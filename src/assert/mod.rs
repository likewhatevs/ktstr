//! Pass/fail evaluation of scenario results.
//!
//! Key types:
//! - [`AssertResult`] -- pass/fail status with diagnostics and statistics
//! - [`Assert`] -- composable assertion config (worker + monitor checks)
//! - [`ScenarioStats`] / [`CgroupStats`] -- aggregated telemetry
//! - [`NumaMapsEntry`] -- parsed `/proc/self/numa_maps` VMA entry
//! - [`Verdict`] -- pointwise-claim accumulator (built via
//!   [`Assert::verdict`] / [`Verdict::new`]; comparators routed through
//!   [`ClaimBuilder`] / [`SetClaim`] / [`SeqClaim`])
//!
//! NUMA assertion functions:
//! - [`parse_numa_maps`] -- parse numa_maps content into per-VMA entries
//! - [`page_locality`] -- compute page locality fraction from entries
//! - [`parse_vmstat_numa_pages_migrated`] -- extract vmstat migration counter
//! - [`assert_page_locality`] / [`assert_cross_node_migration`] -- threshold checks
//!
//! Assertion uses a three-layer merge: [`Assert::default_checks()`] ->
//! `Scheduler.assert` -> per-test `assert`.
//!
//! # Statistical conventions
//!
//! - **Percentiles / medians**: nearest-rank (see `percentile`),
//!   value at index `ceil(n * p) - 1`. Unlike interpolated
//!   percentiles, every reported p99 is an actual observed sample,
//!   not a synthetic midpoint. Consistent across every
//!   [`CgroupStats`] and [`ScenarioStats`] latency field.
//! - **CV (coefficient of variation)** is stddev/mean computed over
//!   the pooled latency samples, not as a mean of per-worker CVs —
//!   see [`CgroupStats::wake_latency_cv`] for the masking caveat.
//!
//! See the [Checking](https://likewhatevs.github.io/ktstr/guide/concepts/checking.html)
//! chapter of the guide.

use crate::workload::WorkerReport;
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};

thread_local! {
    /// Thread-local active phase label. Set by the [`PhaseGuard`]
    /// scope helper at scenario-driver `run_step` entry and read by
    /// [`AssertDetail::new`] / [`PassDetail::binary`] /
    /// [`PassDetail::unary`] / [`NoteValue`] producers so every
    /// detail constructed under a guarded scope auto-stamps its
    /// `phase` field with the active label without the producer
    /// having to thread context through every `with_phase` chain.
    /// `None` outside any guarded scope (boot, BASELINE settle,
    /// non-scenario test fixtures).
    static ACTIVE_PHASE: RefCell<Option<std::borrow::Cow<'static, str>>> =
        const { RefCell::new(None) };
}

/// Snapshot the active phase label installed by the most recent
/// [`PhaseGuard::install`] on this thread. `None` outside any
/// guarded scope. Construction sites for [`AssertDetail`] /
/// [`PassDetail`] / [`NoteValue`] call this to auto-stamp the
/// `phase` field; the test author can still override via the
/// builder `with_phase(...)` chain when an explicit value is
/// preferred.
pub fn current_phase_label() -> Option<std::borrow::Cow<'static, str>> {
    ACTIVE_PHASE.with(|p| p.borrow().clone())
}

/// RAII scope guard for the `ACTIVE_PHASE` thread-local. Install
/// at scenario-driver `run_step` entry; the guard's `Drop` restores
/// the prior phase label, supporting cleanly-nested scenario
/// dispatch (sub-scenarios layer over a parent's phase context
/// without leaking).
///
/// ```ignore
/// let _guard = PhaseGuard::install_step(0); // Step[0] → "Step[0]"
/// // ... apply_ops + hold, every assert constructed here stamps
/// //     phase = Some("Step[0]") automatically ...
/// // drop on scope exit restores the prior label (BASELINE outside
/// // any nested Step).
/// ```
#[must_use = "PhaseGuard restores the prior phase on Drop — bind it to a local"]
pub struct PhaseGuard {
    /// The phase label that was active before this guard installed.
    /// Restored on Drop so nested guards stack cleanly.
    previous: Option<std::borrow::Cow<'static, str>>,
}

impl PhaseGuard {
    /// Install `label` as the active phase. Captures the
    /// previously-active label for restoration on Drop. Use
    /// [`Self::install_step`] / [`Self::install_baseline`] for the
    /// scenario-driver call sites — they produce the standard
    /// `"Step[k]"` / `"BASELINE"` labels matching the rest of the
    /// pipeline.
    pub fn install(label: impl Into<std::borrow::Cow<'static, str>>) -> Self {
        let previous = ACTIVE_PHASE.with(|p| p.replace(Some(label.into())));
        Self { previous }
    }

    /// Convenience: install the `"Step[k]"` label for the
    /// `zero_indexed`-th scenario Step. Matches the label
    /// [`PhaseBucket`] embeds + the [`Phase::step`] display
    /// (`Step[0]`, `Step[1]`, ...).
    pub fn install_step(zero_indexed: u16) -> Self {
        Self::install(format!("Step[{}]", zero_indexed))
    }

    /// Convenience: install the `"BASELINE"` label for the
    /// pre-first-Step settle window. Matches the label
    /// [`PhaseBucket`] uses for `step_index = 0`.
    pub fn install_baseline() -> Self {
        Self::install(std::borrow::Cow::Borrowed("BASELINE"))
    }
}

impl Drop for PhaseGuard {
    fn drop(&mut self) {
        ACTIVE_PHASE.with(|p| {
            *p.borrow_mut() = self.previous.take();
        });
    }
}

/// Per-VMA entry parsed from `/proc/self/numa_maps`.
#[derive(Debug, Clone, Default)]
pub struct NumaMapsEntry {
    /// Virtual address of the VMA.
    pub addr: u64,
    /// Per-node page counts (node_id -> page_count).
    pub node_pages: BTreeMap<usize, u64>,
}

/// Parse `/proc/self/numa_maps` content into per-VMA entries.
///
/// Each line has the format:
///   `<hex_addr> <policy> [key=val ...]`
/// where per-node page counts appear as `N<node>=<count>`.
pub fn parse_numa_maps(content: &str) -> Vec<NumaMapsEntry> {
    let mut entries = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut parts = line.split_whitespace();
        let addr = match parts.next().and_then(|s| u64::from_str_radix(s, 16).ok()) {
            Some(a) => a,
            None => continue,
        };
        // Skip policy field.
        let _ = parts.next();

        let mut entry = NumaMapsEntry {
            addr,
            ..Default::default()
        };

        for token in parts {
            if let Some(rest) = token.strip_prefix('N')
                && let Some((node_str, count_str)) = rest.split_once('=')
                && let (Ok(node), Ok(count)) = (node_str.parse::<usize>(), count_str.parse::<u64>())
            {
                *entry.node_pages.entry(node).or_insert(0) += count;
            }
        }

        if !entry.node_pages.is_empty() {
            entries.push(entry);
        }
    }
    entries
}

/// Compute page locality fraction from parsed numa_maps entries.
///
/// Returns the fraction of pages residing on any node in
/// `expected_nodes` (0.0-1.0). Returns 0.0 when no pages are observed
/// — a zero-allocation workload is not vacuously local; reporting 1.0
/// would let `min_page_locality` thresholds silently pass on broken
/// runs that produced no NUMA signal. The expected node set is
/// derived from the worker's
/// [`MemPolicy`](crate::workload::MemPolicy) at evaluation time.
pub fn page_locality(entries: &[NumaMapsEntry], expected_nodes: &BTreeSet<usize>) -> f64 {
    let mut total: u64 = 0;
    let mut local: u64 = 0;
    for entry in entries {
        for (&node, &count) in &entry.node_pages {
            total += count;
            if expected_nodes.contains(&node) {
                local += count;
            }
        }
    }
    if total > 0 {
        local as f64 / total as f64
    } else {
        0.0
    }
}

/// Extract `numa_pages_migrated` from `/proc/vmstat` content.
///
/// Returns `None` if the counter is not present. The counter is
/// cumulative; callers diff pre- and post-workload snapshots to
/// get migration count during the test.
pub fn parse_vmstat_numa_pages_migrated(content: &str) -> Option<u64> {
    for line in content.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("numa_pages_migrated") {
            let rest = rest.trim();
            if let Ok(v) = rest.parse::<u64>() {
                return Some(v);
            }
        }
    }
    None
}

fn gap_threshold_ms() -> u64 {
    // Unoptimized debug builds have higher scheduling overhead.
    if cfg!(debug_assertions) { 3000 } else { 2000 }
}

fn spread_threshold_pct() -> f64 {
    // Debug builds in small VMs (especially under EEVDF) show higher
    // spread than optimized builds under sched_ext schedulers.
    if cfg!(debug_assertions) { 35.0 } else { 15.0 }
}

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
    Starved,
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
    /// Monitor-subsystem anomaly (imbalance, DSQ depth, rq_clock stall).
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
    /// rq_clock-stall): this kind flags individual event-counter
    /// regressions surfaced by [`assert_scx_events_clean`]. The
    /// counters themselves originate in the kernel's per-task
    /// `scx_event_stats` (see `kernel/sched/ext.c` —
    /// `SCX_EV_*` macros); ktstr reads aggregated deltas via
    /// `monitor::ScxEventDeltas` and presents them to the
    /// assertion as `(name, count)` pairs.
    SchedulerEvent,
    /// Temporal assertion failure on a periodic-capture
    /// [`SampleSeries`](crate::scenario::sample::SampleSeries).
    /// One of the six built-in patterns
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
    /// Host-mode worker stall detected by
    /// [`crate::scenario::host_stall`]. The polling thread
    /// observed `Δnr_switches == 0` AND `Δsum_exec_runtime == 0`
    /// across the configured window for a worker pid — the task
    /// neither got picked nor preempted for at least
    /// `STALL_WINDOW * poll_interval` ms. Distinct from
    /// [`DetailKind::Stuck`] (worker-side report: a worker was
    /// off-CPU longer than the in-test gap threshold): this kind
    /// fires from the host-side polling thread when running
    /// host-mode (no VM boot) and is the only stall signal
    /// available in that mode.
    WorkerStalled,
    /// Skip notification (scenario could not run under this topology/flags).
    Skip,
    /// Uncategorized — falls through when a detail has no specific kind.
    Other,
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
        }
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
/// `MAX_BULK_FRAME_PAYLOAD` per vmm/bulk.rs:53.
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
/// b.record_fail(AssertDetail::new(DetailKind::Starved, "worker starved"));
///
/// a.merge(b);
/// assert!(a.is_fail());
/// assert!(a.failure_details().any(|d| d.kind == DetailKind::Starved));
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
/// programmatic consumption (sidecar parsers, `stats compare`,
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
/// (which pins both the JSON shape and the postcard byte
/// sequence) so a silent-shift regression trips at test time.
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

/// Verdict for a single test scenario.
///
/// # Reading the verdict
///
/// Inspect the terminal verdict via [`Self::outcome`] (returns the
/// folded [`Outcome`] enum) or the convenience accessors
/// [`Self::is_pass`] / [`Self::is_fail`] / [`Self::is_inconclusive`] /
/// [`Self::is_skip`]. Iterate the per-variant payloads via
/// [`Self::failure_details`] (all [`Outcome::Fail`] payloads),
/// [`Self::inconclusive_details`] (all [`Outcome::Inconclusive`]
/// payloads), and [`Self::skip_details`] (all [`Outcome::Skip`]
/// payloads). All four bool accessors mirror
/// [`Outcome::is_pass`] / [`Outcome::is_fail`] /
/// [`Outcome::is_inconclusive`] / [`Outcome::is_skip`].
///
/// # Recording outcomes
///
/// Producers use the atomic mutators [`Self::record_fail`] /
/// [`Self::record_skip`] / [`Self::record_inconclusive`] /
/// [`Self::record_pass`] (each pushes a single [`Outcome`] variant
/// onto [`Self::outcomes`]) and the escape hatch
/// [`Self::record_outcome`] for pre-folded values. Constructors
/// [`Self::pass`] / [`Self::skip`] / [`Self::fail`] seed the
/// outcomes vec with the corresponding variant; [`Self::pass`] is
/// zero-allocation (empty vec; the Pass identity element).
///
/// **Wire-format stability**: this struct is postcard-serialized as
/// part of the in-VM `MSG_TYPE_TEST_RESULT` payload and as
/// sidecar artifacts under `~/.cache/ktstr`. The wire format is
/// **not stable across crate versions** — pre-1.0, fields can be
/// added, removed, or reshaped at any time, and old sidecars must
/// be regenerated after upgrades (re-running the affected tests
/// produces a fresh sidecar). Per the project's pre-1.0 no-compat
/// stance ([`crate::scenario`] module-level doc), no
/// `#[serde(default)]` shims are added for old payloads.
#[must_use = "test verdict is lost if not checked"]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AssertResult {
    /// Recorded terminal verdicts in emission order, one entry per
    /// check that explicitly called [`Self::record_pass`],
    /// [`Self::record_skip`], [`Self::record_inconclusive`], or
    /// [`Self::record_fail`] (plus the single entry seeded by
    /// [`Self::skip`] / [`Self::fail`] constructors).
    ///
    /// **Empty `outcomes` is the Pass identity** — [`Self::pass`]
    /// constructs with `outcomes: vec![]`, [`Self::outcome`] folds
    /// the vec via [`Outcome::merge`] starting from
    /// [`Outcome::Pass`], so a never-touched accumulator naturally
    /// resolves to Pass without any allocation. `record_pass()` is
    /// for the rare case where a test explicitly records a passing
    /// check (e.g. per-check helpers that document what passed);
    /// `pass()` is the zero-state "nothing failed so far"
    /// constructor.
    ///
    /// The folded terminal verdict is computed by [`Self::outcome`]
    /// per the precedence `Fail > Inconclusive > Pass > Skip`. Use
    /// [`Self::is_pass`] / [`Self::is_fail`] /
    /// [`Self::is_inconclusive`] / [`Self::is_skip`] for bool
    /// checks; use [`Self::failure_details`] /
    /// [`Self::inconclusive_details`] / [`Self::skip_details`] to
    /// iterate the per-variant [`AssertDetail`] payloads.
    pub outcomes: Vec<Outcome>,
    /// Structured records of every passing claim. Counterpart to
    /// [`Self::outcomes`]: where `outcomes` carries terminal-verdict
    /// records (Fail/Skip/Pass per-check), `passes` carries the
    /// positive confirmations every comparator's pass arm emits via
    /// [`Verdict`]'s `record_pass_unary` / `record_pass_binary`
    /// helpers.
    /// Empty in tests that don't exercise the structured-pass path
    /// (the no-claim base case), populated whenever a [`Verdict`]
    /// records claims. The auto-repro renderer iterates both vecs
    /// to compose the bracketed phase-grouped output that surfaces
    /// passing context alongside failing assertions.
    ///
    /// **Bounded by [`MAX_RECORDED_PASSES`]** — past that count,
    /// further pushes drop on the floor and a single sentinel
    /// record named [`PASSES_TRUNCATION_SENTINEL_NAME`] appears at
    /// the tail. Use the sentinel-name check (not `len()`
    /// arithmetic) to detect truncation.
    ///
    /// **Test-author convention**: do NOT pin `result.passes` shape
    /// or contents in test assertions unless the test exists
    /// specifically to verify the structured-pass surface (e.g.
    /// the auto-repro renderer's own coverage tests). The field
    /// exists for the renderer's consumption; pinning it
    /// elsewhere makes the test surface viral — every new
    /// comparator that fires under the test starts churning the
    /// pin. Pin `outcome()`, `failure_details()`, and `measurements` for
    /// scenario verification.
    pub passes: Vec<PassDetail>,
    /// Aggregated stats from all workers in this scenario.
    pub stats: ScenarioStats,
    /// Structured measurements attached via [`Self::note_value`] /
    /// [`Verdict::note_value`]. Distinct from [`Self::outcomes`] —
    /// outcomes carry typed verdict variants with `AssertDetail`
    /// payloads for operator triage, `measurements` carries typed
    /// `(key, NoteValue)` pairs for programmatic consumption (sidecar
    /// parsers, `stats compare`, regression dashboards).
    pub measurements: std::collections::BTreeMap<String, NoteValue>,
    /// Informational annotations attached via [`Self::note`] /
    /// [`Verdict::note`]. Structurally separated from [`Self::outcomes`]
    /// so the failure stream stays purely failure-shaped: sidecar
    /// consumers iterating `details` count real failures without
    /// the "forgot to filter notes" silent-miscount class of bug
    /// that the prior `DetailKind::Note` variant on [`AssertDetail`]
    /// invited. The auto-repro renderer surfaces these alongside the
    /// failure summary so the operator still sees them on a failing
    /// run.
    pub info_notes: Vec<InfoNote>,
}

/// Per-cgroup statistics from worker telemetry.
///
/// # Percentile convention
///
/// `p99_wake_latency_us` and `median_wake_latency_us` are computed
/// by `percentile` using the NEAREST-RANK (Type 1) definition:
/// the value at `ceil(n * p) - 1` in sorted order. No interpolation
/// between samples. This matches the percentile convention used
/// throughout schbench and the BPF latency histograms the project
/// cross-references, so a `ktstr` p99 reading aligns with a
/// schbench `lat99` without adjustment. For small `n` (wake
/// reservoirs cap at `MAX_WAKE_SAMPLES = 100_000` per worker —
/// see `workload.rs`) nearest-rank is also numerically stable —
/// interpolation between the two nearest ranks would be
/// implementation-defined at sample-set boundaries.
///
/// # CV pooling scope
///
/// `wake_latency_cv` is POOLED across every sample from every
/// worker in the cgroup, not a per-worker CV averaged back. That
/// collapses per-worker dispersion into the cgroup-wide signal:
/// two workers with uniformly low jitter but different means
/// produce a high pooled CV (mean-shift between workers inflates
/// stddev), while per-worker CV would show neither worker as
/// bad. This is intentional for the fairness threshold
/// (`max_wake_latency_cv`): a scheduler that gives worker A
/// 10µs wakes and worker B 1ms wakes is failing fairness even if
/// each worker on its own is tight. Tests comparing single-worker
/// behavior should scope their assertions to per-worker data
/// rather than this aggregate.
///
/// # Derived ratios
///
/// Two metrics are DERIVED rather than measured and live as
/// `&self` methods, NOT as serde-serialized fields:
/// [`Self::wake_latency_tail_ratio`] (= p99/median) and
/// [`Self::iterations_per_worker`] (= total_iterations/num_workers).
/// Pre-1.0 cleanup eliminated the prior stored-field shadow and
/// `derive_ratios` stamper. Consumers always recompute on read,
/// so a hand-constructed fixture or a deserialized sidecar from an
/// older build cannot silently carry a stale ratio. The run-level
/// worst-cgroup tail ratio (`crate::stats::MetricKind::WakeLatencyTailRatio`,
/// an `ext_metrics` entry) and the iterations efficiencies
/// (`worst_iterations_per_worker` / `worst_iterations_per_cpu_sec`) are all
/// re-pooled POST-merge by [`populate_run_distribution_metrics`] — the tail
/// ratio as the max over [`Self::wake_latency_tail_ratio`] across per-cgroup
/// [`Self`] entries, the efficiencies lowest-wins from
/// [`Self::iterations_per_worker`] / [`Self::iterations_per_cpu_sec`].
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize, crate::Claim)]
pub struct CgroupStats {
    /// Cgroup name (the workload-handle label this telemetry belongs to),
    /// or empty for unlabeled call sites (`collect_all`, bare
    /// `assert_cgroup`). Set post-hoc by `collect_handles` (in
    /// `crate::scenario`) where the name is in scope; `cgroup_stats`
    /// itself has only the reports and leaves it empty. Lets a PASSING-run
    /// consumer say which cgroup's work landed on which CPUs.
    pub cgroup_name: String,
    /// Number of workers in this cgroup.
    pub num_workers: usize,
    /// Distinct CPUs the workers in this cgroup actually ran on (union of
    /// each [`crate::workload::WorkerReport::cpus_used`]). `num_cpus` is
    /// its length, kept for the existing rollups; this set surfaces WHICH
    /// CPUs (not just how many) on every run, pass or fail.
    pub cpus_used: BTreeSet<usize>,
    /// Distinct CPUs used across all workers in this cgroup
    /// (`cpus_used.len()`).
    pub num_cpus: usize,
    /// Mean off-CPU percentage across workers (off_cpu_ns /
    /// wall_time_ns * 100). `None` when no worker reported a
    /// positive `wall_time_ns` (off-CPU% is undefined without wall
    /// time) — distinct from `Some(0.0)`, a measured "never off
    /// CPU". The `Option` keeps a not-measured cgroup from reading
    /// as a perfectly-on-CPU one in the telemetry consumers
    /// (`ScenarioStats.cgroups`).
    pub avg_off_cpu_pct: Option<f64>,
    /// Minimum off-CPU percentage across workers. `None` under the
    /// same no-measurable-wall-time condition as `avg_off_cpu_pct`.
    pub min_off_cpu_pct: Option<f64>,
    /// Maximum off-CPU percentage across workers. `None` under the
    /// same no-measurable-wall-time condition as `avg_off_cpu_pct`.
    pub max_off_cpu_pct: Option<f64>,
    /// `max_off_cpu_pct - min_off_cpu_pct`. Measures scheduling
    /// fairness within the cgroup. `None` when off-CPU% was not
    /// measured (no worker with positive wall time) — a not-measured
    /// cgroup is inconclusive for fairness, NOT "spread 0 = perfectly
    /// fair". `Some(0.0)` means a real measured zero spread.
    pub spread: Option<f64>,
    /// Longest scheduling gap across all workers (ms).
    pub max_gap_ms: u64,
    /// CPU where the longest scheduling gap occurred.
    pub max_gap_cpu: usize,
    /// Sum of CPU migration counts across all workers.
    pub total_migrations: u64,
    /// Migrations per iteration (total_migrations / total_iterations).
    pub migration_ratio: f64,
    /// 99th percentile wake latency across all workers (microseconds).
    pub p99_wake_latency_us: f64,
    /// Median wake latency across all workers (microseconds).
    pub median_wake_latency_us: f64,
    /// Coefficient of variation (stddev / mean) of wake latencies.
    ///
    /// Computed over the POOLED latency samples from every worker in
    /// the cgroup, not as a mean of per-worker CVs. Per-worker
    /// dispersion is therefore masked: a cgroup with one tight
    /// worker and one wildly variable worker can report a moderate
    /// pooled CV that looks healthier than either constituent. Use
    /// [`WorkerReport::wake_latencies_ns`] directly if per-worker
    /// CV is needed.
    pub wake_latency_cv: f64,
    /// Sum of iteration counts across all workers.
    pub total_iterations: u64,
    /// Sum of per-worker on-CPU time (nanoseconds), from each worker's
    /// schedstat run time ([`crate::workload::WorkerReport::schedstat_cpu_time_ns`]
    /// — `task->se.sum_exec_runtime`, the FIRST `/proc/<pid>/schedstat` field
    /// (`sched_info` supplies only the run_delay/pcount fields 2/3, not the
    /// on-CPU time), the summable per-thread proxy for the cgroup's
    /// `cpu.stat usage_usec`).
    /// Denominator for [`Self::iterations_per_cpu_sec`], the
    /// overcommit-invariant per-cell rate. `0` when no worker reported on-CPU
    /// time (the accessor then returns `None`).
    pub total_cpu_time_ns: u64,
    /// Mean schedstat run delay across workers (microseconds).
    pub mean_run_delay_us: f64,
    /// Worst schedstat run delay across workers (microseconds).
    pub worst_run_delay_us: f64,
    /// Fraction of pages on the expected NUMA node(s) (0.0-1.0).
    /// Derived from `/proc/self/numa_maps` and the worker's
    /// [`MemPolicy`](crate::workload::MemPolicy).
    pub page_locality: f64,
    /// Cross-node page migration ratio from `/proc/vmstat`
    /// `numa_pages_migrated` delta divided by total allocated pages.
    pub cross_node_migration_ratio: f64,
    /// Extensible metrics for the generic comparison pipeline.
    pub ext_metrics: BTreeMap<String, f64>,
}

/// Per-phase per-cgroup raw telemetry components — the per-phase analogue of
/// [`CgroupStats`]. Holds RAW components (sample vectors + counters), NOT the
/// reduced ratios/percentiles [`CgroupStats`] computes, so whole-run and
/// cross-run aggregates RE-POOL from the components at every level (the
/// per-phase telemetry thesis: an aggregate is recomputed over the pooled
/// components, never averaged from ready-made per-phase reductions — a
/// percentile or weighted ratio cannot be recovered from per-phase scalars).
/// Covers every TYPED [`CgroupStats`] reduction: avg/min/max off-CPU% and
/// spread from `off_cpu_pcts`; p99/median/CV wake latency from
/// `wake_latencies_ns`; mean/worst run-delay from `run_delays_ns`;
/// migration_ratio, iterations_per_cpu_sec, iterations_per_worker,
/// page_locality, cross_node_migration_ratio from their counter components;
/// the COUPLED worst gap (ms + the CPU that owned it) from `max_gap_ms` /
/// `max_gap_cpu`; cpus_used / num_cpus from `cpus_used`. EXCLUDES
/// [`CgroupStats::ext_metrics`] (the generic extensible map — a per-phase
/// per-cgroup custom metric is a future extension, not part of the typed
/// carrier). Lives in [`PhaseBucket::per_cgroup`], keyed by cgroup name. The
/// structural carrier is empty until a capture path populates it per phase.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize, crate::Claim)]
pub struct PhaseCgroupStats {
    /// Worker count in this cgroup for the phase — the denominator for the
    /// re-pooled per-worker iteration rate (`iterations_per_worker` =
    /// `total_iterations` / this). This is a set CARDINALITY (`reports.len()`),
    /// not a kernel counter, but it SUMs in `merge` because a single cgroup name
    /// can emit MULTIPLE carriers in one step — `collect_handles` builds one per
    /// `WorkloadHandle`, and a `CgroupDef` with several `WorkSpec` entries
    /// (`.work(..).work(..)`) spawns one handle per `WorkSpec` under the same
    /// name (`apply_setup`). Those carriers cover DISJOINT worker subsets, so the
    /// cardinality of their union is the SUM (4 + 2 → 6), matching [`cgroup_stats`]
    /// over the pooled reports (`reports.len()`); a MAX would understate the count
    /// and inflate `iterations_per_worker`. (The disjointness is the real
    /// justification — were carriers ever to overlap, the SUM would over-count.)
    pub num_workers: usize,
    /// Distinct CPUs the cgroup's workers ran on in the phase (union of each
    /// worker's `cpus_used`). Re-pools [`CgroupStats::cpus_used`] / `num_cpus`
    /// (= the set / its length) via a set UNION.
    pub cpus_used: std::collections::BTreeSet<usize>,
    /// Pooled per-wakeup latency samples (ns) across the cgroup's workers in
    /// the phase, un-reduced so p99 / median / CV re-pool over the combined set.
    /// The POOL is reservoir-capped at `MAX_WAKE_SAMPLES` (the per-worker bound,
    /// re-applied when same-name carriers merge so the carrier payload stays
    /// bounded on the size-limited guest bulk port — without it the pool would be
    /// `workers × MAX_WAKE_SAMPLES`); `wake_sample_total` carries the true
    /// pre-cap population. The CV / stddev re-pool divides by
    /// `wake_latencies_ns.len()` (this capped pool size), NOT by
    /// `wake_sample_total` — mirroring [`cgroup_stats`], which computes
    /// `cv = stddev/mean` with `n = all_latencies.len()`.
    ///
    /// PARITY CONTRACT (the one component whose parity is size-dependent): for
    /// pools ≤ `MAX_WAKE_SAMPLES` the reservoir IS the full concatenation, so the
    /// p99 / median / CV re-pool reproduces [`cgroup_stats`] VALUE-FOR-VALUE.
    /// Above the cap the carrier holds a distribution-preserving reservoir
    /// SUBSAMPLE while [`cgroup_stats`] reduces over the full per-worker concat,
    /// so the re-pool is DISTRIBUTION-EQUIVALENT, not byte-identical (the bounded
    /// bulk-port frame forbids carrying the full pool; staged reservoirs cannot be
    /// byte-identical to a single full-pool reduction). This is BY DESIGN:
    /// `cgroup_stats` stays the uncapped run-level authority (capping it to match
    /// the carrier would discard most of a multi-worker cgroup's samples to chase
    /// a sub-display-precision artifact), and the carrier's >cap merge is WEIGHTED
    /// by `wake_sample_total` (`Self::weighted_merge_reservoirs`) so the subsample
    /// is an UNBIASED sample of the combined population — no smaller-population
    /// skew. (That unbiasedness is the carrier MERGE layer only; the cross-PHASE
    /// run-level pool in `populate_run_distribution_metrics` still concatenates
    /// per-phase carriers length-weighted — a separate residual skew when one
    /// cgroup exceeds the cap in multiple phases, not addressed here.)
    pub wake_latencies_ns: Vec<u64>,
    /// True wakeup count before reservoir clamping (`wake_latencies_ns` is
    /// capped), so the re-pool can report the real population size. An
    /// intentional ADDITION over [`CgroupStats`] (which has no such field), NOT
    /// a mirrored reduction — do not strip it in a strict-parity audit; it is
    /// the only source of the true wakeup population once `wake_latencies_ns` is
    /// reservoir-clamped, and it is for REPORTING, not the CV denominator.
    pub wake_sample_total: u64,
    /// Pooled per-worker schedstat run-delay samples (RAW ns) for the phase,
    /// un-reduced so mean / worst run-delay re-pool over the combined set; the
    /// re-pool converts ns → µs to match [`CgroupStats`]'s run-delay-µs fields.
    /// Stored as raw kernel ns (like `wake_latencies_ns`), not pre-converted,
    /// per the raw-component thesis. GRANULARITY: unlike `wake_latencies_ns`
    /// (one per WAKEUP), each entry here is ONE per-worker value — the
    /// worker's cumulative `sched_info.run_delay` delta over its execution
    /// (`schedstat_run_delay_ns`, end−start). So the pool size is the worker
    /// count, the mean is the average per-worker total queued-to-run delay, and
    /// `worst_run_delay_us` selects the single worker with the largest total
    /// queued-to-run delay (NOT the worst single dispatch).
    pub run_delays_ns: Vec<u64>,
    /// Per-worker off-CPU% samples for the phase, un-reduced. Carried for the
    /// per-phase per-cgroup off-CPU% RENDER (Item 8) — the avg / min / max /
    /// spread of the combined set. NOT consumed by the Item 7 run-level
    /// distributional re-pool: off-CPU% has no run-level Distribution metric
    /// (off-CPU%/spread is intrinsically per-cgroup, so the run-level
    /// `worst_spread` stays the cross-cgroup max of per-cgroup
    /// [`CgroupStats::spread`] via the typed [`AssertResult::merge`] fold, not a
    /// pooled distribution). An EMPTY vec is the not-measured state (no worker
    /// with positive wall time), preserving the not-measured vs measured-zero
    /// distinction [`CgroupStats`] keeps. Stored as raw samples, not pre-reduced
    /// extremes, because the mean is unrecoverable from min/max alone for >2
    /// workers. Each sample is `off_cpu_ns /
    /// wall_time_ns * 100`, where `off_cpu_ns = wall_time_ns - cpu_time_ns` and
    /// `cpu_time_ns` is the `CLOCK_THREAD_CPUTIME_ID` thread on-CPU time
    /// (workload/worker `off_cpu_ns` at report build). `total_cpu_time_ns` is a
    /// DISTINCT on-CPU measurement (`schedstat_cpu_time_ns`, the `/proc`
    /// schedstat `se.sum_exec_runtime`): both ultimately track on-CPU runtime but
    /// are sampled at different points (the `CLOCK_THREAD_CPUTIME_ID` read folds
    /// the in-flight delta; the schedstat field reads the stored value), so the
    /// two need not be byte-identical and must not be cross-wired in a re-pool.
    pub off_cpu_pcts: Vec<f64>,
    /// Sum of per-worker CPU-migration counts in the phase (Counter).
    pub total_migrations: u64,
    /// Sum of per-worker iteration counts in the phase (Counter).
    pub total_iterations: u64,
    /// Sum of per-worker on-CPU time (ns) in the phase — the
    /// overcommit-invariant rate denominator (Counter). Sourced from
    /// `schedstat_cpu_time_ns` (the `/proc` schedstat `se.sum_exec_runtime`,
    /// rq-charged on-CPU ns) — a DISTINCT on-CPU-time sample from the
    /// `CLOCK_THREAD_CPUTIME_ID` time behind `off_cpu_pcts` (different sample
    /// point; not byte-identical), so do not cross-wire the two in a re-pool.
    pub total_cpu_time_ns: u64,
    /// Pages on the expected NUMA node(s) — page-locality numerator. A genuine
    /// per-thread numa_maps count (Counter, SUM across workers/sources).
    pub numa_pages_local: u64,
    /// Total allocated pages — the SHARED denominator for BOTH page_locality
    /// (`numa_pages_local` / this) AND cross_node_migration_ratio
    /// (`cross_node_migrated` / this). A genuine per-thread numa_maps count
    /// (Counter, SUM); the kernel computes both ratios over the identical page
    /// total, so one field serves both — a separate cross_node_total would
    /// invite a silent desync.
    pub numa_pages_total: u64,
    /// Cross-node migrated pages — cross_node_migration_ratio numerator
    /// (denominator is `numa_pages_total`). A SYSTEM-WIDE
    /// `/proc/vmstat numa_pages_migrated` delta each worker observes
    /// redundantly, so this is a PEAK (MAX across workers/sources), NOT a
    /// Counter — summing would inflate it by the worker count (mirrors
    /// [`CgroupStats`]'s deliberate max-fold of the same quantity).
    pub cross_node_migrated: u64,
    /// Longest scheduling gap (ms) across the cgroup's workers in the phase,
    /// coupled with `max_gap_cpu`. A Peak folded as an ARGMAX of the (ms, cpu)
    /// pair so the worst gap and its CPU survive together — mirrors
    /// [`CgroupStats`]'s `max_gap_ms` / `max_gap_cpu` coupling (a bare
    /// independent max would desync the gap from its CPU).
    pub max_gap_ms: u64,
    /// CPU that owned the worst scheduling gap — `max_gap_ms`'s argmax
    /// companion. Folded together with `max_gap_ms`, never independently.
    pub max_gap_cpu: usize,
    /// True when this carrier's raw sample vectors (`wake_latencies_ns` /
    /// `run_delays_ns` / `off_cpu_pcts`) were dropped by
    /// [`AssertResult::strip_phase_cgroup_samples`] to fit the size-limited guest
    /// bulk frame — distinct from a carrier that genuinely measured no samples.
    /// The reduced counters survive; only the per-phase distribution render
    /// (Item 8) loses its source, so the render shows "samples stripped" rather
    /// than the not-measured "n/a". Defaults to `false` (not stripped) and is set
    /// only on a carrier that actually HAD samples to drop; ORs across `merge` so
    /// a merged carrier is stripped if either input was.
    pub stripped: bool,
}

impl PhaseCgroupStats {
    /// Component-wise union of two per-phase per-cgroup data for the SAME
    /// cgroup name (same `step_index`). Fold rule by component class:
    /// - sample vectors (`wake_latencies_ns`, `run_delays_ns`, `off_cpu_pcts`)
    ///   CONCAT, so the re-pool sees the combined set, never a mean of
    ///   per-source reductions;
    /// - the CPU set (`cpus_used`) UNIONs;
    /// - genuine Counters (`num_workers`, `wake_sample_total`,
    ///   `total_migrations`, `total_iterations`, `total_cpu_time_ns`,
    ///   `numa_pages_local`, `numa_pages_total`) SUM — `num_workers` included,
    ///   because a multi-`WorkSpec` cgroup emits one carrier per handle covering
    ///   DISJOINT worker subsets, so summing reproduces the pooled count (see
    ///   the `num_workers` field doc);
    /// - the one Peak, `cross_node_migrated`, takes the MAX (a system-wide
    ///   vmstat delta observed redundantly per worker, so summing would inflate
    ///   it);
    /// - the COUPLED worst gap (`max_gap_ms`, `max_gap_cpu`) folds as an
    ///   ARGMAX — the pair from whichever side has the larger ms (b's on tie,
    ///   matching the builders' `max_by_key` last-wins) so the gap and its CPU
    ///   stay bound together.
    ///
    /// The counter SUMs use plain `+`: debug builds panic on overflow rather
    /// than wrapping. The realistic magnitudes (iteration / ns counts far
    /// below `u64::MAX` even pooled across a long run) keep overflow
    /// unreachable; a loud debug panic is preferred over a silently wrong
    /// re-pool denominator.
    fn merge(a: PhaseCgroupStats, b: PhaseCgroupStats) -> PhaseCgroupStats {
        // Merge the two capped wake-latency reservoirs. Same-name carriers (a
        // multi-`WorkSpec` cgroup's per-handle carriers) merge ON THE GUEST before
        // the AssertResult is serialized over the bulk port, so K carriers must
        // not concat to K × MAX_WAKE_SAMPLES (it could overrun the 16 MiB frame,
        // flipping a PASS to a truncated FAIL).
        //
        // ≤cap: the concatenation IS the true combined population, so it passes
        // through unchanged — value-for-value parity with cgroup_stats for small
        // pools (only >cap pools become a subsample; see the `wake_latencies_ns`
        // field doc). >cap: a WEIGHTED reservoir merge weighted by each carrier's
        // true pre-cap population (`wake_sample_total`), so the merged sample is an
        // UNBIASED uniform sample of the combined population — NOT the
        // smaller-population-skewed reservoir-of-reservoirs an unweighted
        // concat-and-re-cap produced (which weighted by reservoir LENGTH ≈ 50/50,
        // ignoring the true populations).
        let cap = crate::workload::MAX_WAKE_SAMPLES;
        let wake_latencies_ns = if a.wake_latencies_ns.len() + b.wake_latencies_ns.len() <= cap {
            let mut v = a.wake_latencies_ns;
            v.extend(b.wake_latencies_ns);
            v
        } else {
            Self::weighted_merge_reservoirs(
                &a.wake_latencies_ns,
                a.wake_sample_total,
                &b.wake_latencies_ns,
                b.wake_sample_total,
                cap,
            )
        };
        let mut run_delays_ns = a.run_delays_ns;
        run_delays_ns.extend(b.run_delays_ns);
        let mut off_cpu_pcts = a.off_cpu_pcts;
        off_cpu_pcts.extend(b.off_cpu_pcts);
        let mut cpus_used = a.cpus_used;
        cpus_used.extend(b.cpus_used);
        // Coupled worst-gap ARGMAX: take the (ms, cpu) pair together from the
        // side with the larger gap (b's on tie, matching the builders'
        // max_by_key last-wins) so the CPU stays bound to the gap it owned — a
        // bare independent max would desync them. The last-wins tie-break is
        // parity-coupled to fold order: AssertResult::merge folds same-name
        // carriers in the order reports are pooled (handle iteration order), so
        // on an equal-gap tie this yields the same CPU as a single cgroup_stats
        // over the concatenated reports. A reordered fold would break that parity.
        let (max_gap_ms, max_gap_cpu) = if b.max_gap_ms >= a.max_gap_ms {
            (b.max_gap_ms, b.max_gap_cpu)
        } else {
            (a.max_gap_ms, a.max_gap_cpu)
        };
        PhaseCgroupStats {
            num_workers: a.num_workers + b.num_workers,
            cpus_used,
            wake_latencies_ns,
            wake_sample_total: a.wake_sample_total + b.wake_sample_total,
            run_delays_ns,
            off_cpu_pcts,
            total_migrations: a.total_migrations + b.total_migrations,
            total_iterations: a.total_iterations + b.total_iterations,
            total_cpu_time_ns: a.total_cpu_time_ns + b.total_cpu_time_ns,
            numa_pages_local: a.numa_pages_local + b.numa_pages_local,
            numa_pages_total: a.numa_pages_total + b.numa_pages_total,
            cross_node_migrated: a.cross_node_migrated.max(b.cross_node_migrated),
            max_gap_ms,
            max_gap_cpu,
            stripped: a.stripped || b.stripped,
        }
    }

    /// Merge two CAPPED uniform reservoirs into one of size ≤ `cap` that is a
    /// uniform sample of the COMBINED population. `a` is a uniform reservoir of
    /// `w_a` true samples, `b` of `w_b` (their `wake_sample_total` weights). Each
    /// output slot is drawn from `a` with probability `w_a / (w_a + w_b)` and from
    /// `b` otherwise; within a source the index is uniform. Composing the
    /// source-level uniform reservoir with the within-source uniform draw makes
    /// each output a uniform draw from the combined population, so the merged
    /// A-fraction is the TRUE `w_a / (w_a + w_b)`. This removes the equal-slot
    /// ("reservoir-of-reservoirs") skew an unweighted concat-and-re-cap imposes:
    /// two already-capped inputs concat ≈ 50/50 by LENGTH regardless of their true
    /// populations, over-counting the smaller-population carrier. Sampling WITH
    /// replacement is the correct estimator once the inputs are capped (each
    /// reservoir element stands for `w/len` population units; the pre-cap samples
    /// are gone).
    ///
    /// DETERMINISTIC: the xorshift64 stream is seeded from the inputs (populations
    /// + lengths) so the merge is a PURE function of its arguments — unlike
    /// `crate::workload::reservoir_push`, whose stream is gettid-seeded
    /// thread-local (a merge run twice would otherwise differ). The triple-shift
    /// mirrors the codebase's inline xorshift64 (`reservoir_push` /
    /// `io::xorshift64`).
    ///
    /// Assumes `w_a + w_b < 2^64` — a realistic wake population is far below it
    /// (2^64 wakeups is physically unreachable), so the single-u64 `s % total`
    /// draw spans `[0, total)`. Callers gate on `a.len() + b.len() > cap`, which
    /// (each input ≤ cap) guarantees both sources non-empty; the per-slot guards
    /// below stay safe for a degenerate hand-built input regardless.
    fn weighted_merge_reservoirs(a: &[u64], w_a: u64, b: &[u64], w_b: u64, cap: usize) -> Vec<u64> {
        if a.is_empty() && b.is_empty() {
            return Vec::new();
        }
        // Weights are the true populations; fall back to reservoir lengths if a
        // (hand-built) carrier reports zero population alongside non-empty samples,
        // keeping the split well-defined instead of dividing by a zero total. The
        // mixed case (one weight 0, the other > 0) is left as-is: a zero weight
        // sends every draw to the other source, the only defensible split for a
        // source claiming zero population. Production maintains wake_sample_total
        // >= len (reservoir_push counts every push), so neither edge is reachable
        // on the capture path.
        let (wa, wb) = if w_a == 0 && w_b == 0 {
            (a.len() as u128, b.len() as u128)
        } else {
            (w_a as u128, w_b as u128)
        };
        let total = wa + wb;
        // Loud-panic on the documented `w_a + w_b < 2^64` assumption (a realistic
        // wake population is far below it): if total exceeded u64::MAX the
        // `s as u128 % total` draw — s spans [0, 2^64) — could not reach
        // [2^64, total) and would silently bias the source split. Matches the merge
        // SUM's debug-panic-on-overflow discipline (loud over silently wrong).
        debug_assert!(
            total <= u64::MAX as u128,
            "weighted_merge_reservoirs: w_a + w_b overflows u64 ({total}); source draw would bias",
        );
        // Golden-ratio Weyl multiplier (the codebase's standard PRNG seed mixer);
        // a non-zero, input-derived seed makes the merge deterministic. xorshift64
        // has 0 as a fixed point, hence the fallback.
        const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut s = (w_a ^ w_b.rotate_left(32) ^ (a.len() as u64).rotate_left(16) ^ (b.len() as u64))
            .wrapping_mul(GOLDEN);
        if s == 0 {
            s = GOLDEN;
        }
        let step = |x: u64| {
            let mut v = x;
            v ^= v << 13;
            v ^= v >> 7;
            v ^= v << 17;
            v
        };
        let mut out = Vec::with_capacity(cap);
        for _ in 0..cap {
            s = step(s);
            // Defensive empty-source guards: caller gates ensure both non-empty,
            // but a stripped / zero-population fixture must never index an empty
            // slice.
            let from_a = if a.is_empty() {
                false
            } else if b.is_empty() {
                true
            } else {
                (s as u128 % total) < wa
            };
            s = step(s);
            if from_a {
                out.push(a[(s % a.len() as u64) as usize]);
            } else {
                out.push(b[(s % b.len() as u64) as usize]);
            }
        }
        out
    }

    /// Off-CPU% reduction for the per-phase per-cgroup render (Item 8):
    /// `(avg, min, max, spread)` over [`Self::off_cpu_pcts`], or `None` when
    /// the vec is empty — the NOT-measured state (no worker had positive wall
    /// time). Reduces the SAME per-worker pcts [`cgroup_stats`] reduces
    /// (off_cpu_ns / wall_time_ns × 100), so for a phase spanning the whole run
    /// it reproduces that whole-run reduction; `spread = max − min`.
    /// `Some((0.0, ..))` is a MEASURED zero (distinct from the `None`
    /// not-measured state), preserving the discipline the empty-vec contract on
    /// `off_cpu_pcts` keeps. Display-only: never written back into a re-pool.
    pub fn off_cpu_summary(&self) -> Option<(f64, f64, f64, f64)> {
        let pcts = &self.off_cpu_pcts;
        if pcts.is_empty() {
            return None;
        }
        let min = pcts.iter().cloned().reduce(f64::min).expect("non-empty");
        let max = pcts.iter().cloned().reduce(f64::max).expect("non-empty");
        let avg = pcts.iter().sum::<f64>() / pcts.len() as f64;
        Some((avg, min, max, max - min))
    }

    /// Wake-latency reduction for the per-phase render (Item 8):
    /// `(p99_us, median_us)` over the pooled [`Self::wake_latencies_ns`], or
    /// `None` when the pool is empty. Nearest-rank percentile via `percentile`
    /// (ns→µs once), reproducing [`cgroup_stats`]'s p99/median value-for-value
    /// for the ≤cap pool (and the Item 7 re-pool's `reduce_sorted_distribution`).
    /// Above `MAX_WAKE_SAMPLES` the pool is a distribution-preserving reservoir
    /// subsample (see [`Self::wake_latencies_ns`]), so p99/median is then
    /// distribution-equivalent, NOT byte-identical, to the full-pool reduction —
    /// the rendered tail stays accurate, only exact parity is size-bounded.
    /// `None`-on-empty omits the wake segment from the render rather than
    /// painting a misleading 0µs (the display analogue of `cgroup_stats`'s
    /// 0.0-sentinel, which has no Option to carry not-measured).
    pub fn wake_summary(&self) -> Option<(f64, f64)> {
        if self.wake_latencies_ns.is_empty() {
            return None;
        }
        let mut sorted = self.wake_latencies_ns.clone();
        sorted.sort_unstable();
        let p99 = percentile(&sorted, 0.99) as f64 / 1000.0;
        let median = percentile(&sorted, 0.5) as f64 / 1000.0;
        Some((p99, median))
    }

    /// Run-delay reduction for the per-phase render (Item 8):
    /// `(mean_us, worst_us)` over the per-worker [`Self::run_delays_ns`] (raw
    /// ns), or `None` when empty. Divides ns→µs ONCE on the summed / maxed ns.
    /// `worst` reproduces [`cgroup_stats`]'s value-for-value (`max(ns)/1000 ==
    /// max(ns/1000)`, division is monotone). `mean` reproduces it to f64 ULP,
    /// not bit-exactly: this f64-sums then divides once (`Σns/n/1000`), while
    /// `cgroup_stats` divides each worker's ns by 1000 first then sums
    /// (`Σ(ns/1000)/n`) — the same value reassociated, differing only
    /// sub-display-precision (a divergent-input parity test bounds it at 1e-9).
    /// Each sample is
    /// one worker's whole-phase cumulative `sched_info.run_delay` delta, so
    /// `mean` is the average per-worker total queued-to-run delay and `worst`
    /// the largest. `None`-on-empty omits the segment.
    pub fn run_delay_summary(&self) -> Option<(f64, f64)> {
        if self.run_delays_ns.is_empty() {
            return None;
        }
        let n = self.run_delays_ns.len() as f64;
        // Sum in f64, NOT u64-then-cast: matches cgroup_stats's f64 accumulation
        // and cannot integer-overflow (an f64 sum saturates toward +inf; a u64
        // sum would panic in debug / silently wrap in release on a pathological
        // pool). Values are identical within the documented 1e-9 ULP bound.
        let mean = self.run_delays_ns.iter().map(|&v| v as f64).sum::<f64>() / n / 1000.0;
        let worst = *self.run_delays_ns.iter().max().expect("non-empty") as f64 / 1000.0;
        Some((mean, worst))
    }
}

impl CgroupStats {
    /// Wake-latency tail amplification:
    /// `p99_wake_latency_us / median_wake_latency_us`. Returns `0.0`
    /// when `median_wake_latency_us <= 0.0` so the result never
    /// propagates `NaN` / `Infinity` into downstream
    /// `finite_or_zero` filters. Method-only access (no stored
    /// shadow) — recomputed every call from the raw fields.
    ///
    /// Unitless; ≥1.0 by definition of order statistics (p99 cannot
    /// undershoot the median on the same sample set). Values far
    /// above 1.0 signal a long tail — the scheduler wakes most
    /// workers promptly but occasionally stalls some, a regression
    /// axis that neither `median_*` nor `p99_*` exposes in
    /// isolation.
    pub fn wake_latency_tail_ratio(&self) -> f64 {
        if self.median_wake_latency_us > 0.0 {
            self.p99_wake_latency_us / self.median_wake_latency_us
        } else {
            0.0
        }
    }

    /// Throughput per parallel degree:
    /// `total_iterations / num_workers`. `None` when
    /// `num_workers == 0` (no worker reported, so per-worker
    /// throughput is undefined — distinct from a measured zero);
    /// `Some(0.0)` when workers ran but completed zero iterations
    /// (a real throughput collapse). The `None` / `Some(0.0)` split
    /// is load-bearing: the run-level worst-cgroup re-pool in
    /// [`populate_run_distribution_metrics`] (the
    /// `MetricKind::WorstLowest` arm) must treat a measured zero as
    /// the worst reading (it wins the "lowest" bucket) while skipping
    /// a no-data cgroup — collapsing both to `0.0` would hide a
    /// starved cgroup behind the no-data sentinel. Method-only
    /// access (no stored shadow) — recomputed every call from the
    /// raw fields.
    ///
    /// Only meaningful across runs of the SAME variant (equal
    /// scenario duration): cross-variant comparison is misleading
    /// because this metric is NOT rate-normalized — a longer-
    /// running scenario racks up more iterations per worker even if
    /// the scheduler is identical. `stats compare`-style
    /// comparisons hold scenario, topology, and work_type constant
    /// before reading this method.
    pub fn iterations_per_worker(&self) -> Option<f64> {
        if self.num_workers > 0 {
            Some(self.total_iterations as f64 / self.num_workers as f64)
        } else {
            None
        }
    }

    /// Worker iterations per CPU-second of on-CPU time consumed by this
    /// cgroup's workers — `total_iterations / (total_cpu_time_ns / 1e9)`.
    ///
    /// Unlike [`Self::iterations_per_worker`] (raw work, which scales with
    /// the host-CPU budget delivered to the guest) and a wall-time rate
    /// (which also drops under host oversubscription), this is
    /// OVERCOMMIT-INVARIANT: under `cpu_budget < vcpus` a cell completes
    /// proportionally fewer iterations AND consumes proportionally less
    /// on-CPU time, so the ratio cancels the lost host-CPU-time factor. Use
    /// it to compare per-cgroup throughput across `cpu_budget` settings.
    ///
    /// `None` when `num_workers == 0` (no worker — undefined, distinct from a
    /// measured zero) or `total_cpu_time_ns == 0` (no on-CPU time captured;
    /// returns inconclusive rather than `Inf`). For a pure busy-spin
    /// workload this rate is ~constant by construction, so it measures
    /// CPU-time EFFICIENCY; for the cross-cell ALLOCATION balance use
    /// [`ScenarioStats::cgroup_balance_ratio`] over `iterations_per_worker`.
    pub fn iterations_per_cpu_sec(&self) -> Option<f64> {
        if self.num_workers == 0 || self.total_cpu_time_ns == 0 {
            return None;
        }
        Some(self.total_iterations as f64 / (self.total_cpu_time_ns as f64 / 1e9))
    }
}

/// Identifier for a scenario phase. Newtype over `u16` carrying
/// the same 1-indexed encoding documented on every other
/// phase-touching site: `Phase::BASELINE` is the pre-first-Step
/// settle window (`u16` 0); `Phase::step(k)` is scenario Step `k`
/// at 1-indexed `u16` `k + 1`. The newtype catches the bug class
/// where a raw `u16` flows between sites that disagree about
/// 0-indexed vs 1-indexed Step encoding, and gives operators
/// readable construction at consumer sites (`Phase::BASELINE` /
/// `Phase::step(2)` instead of magic `0u16` / `3u16`).
///
/// Wire-format identical to a `u16` via `#[serde(transparent)]` —
/// the on-disk sidecar shape is unchanged from the bare-`u16`
/// pipeline, and existing JSON / typeshare consumers see the same
/// scalar field. `.phase_raw()` exposes the inner `u16` for paths
/// that hand the value to a serializer or formatter that does not
/// understand the newtype.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    Default,
    serde::Serialize,
    serde::Deserialize,
)]
#[serde(transparent)]
pub struct Phase(u16);

impl Phase {
    /// Pre-first-Step settle window. The framework writes
    /// `Phase::BASELINE` to `Ctx::current_step` at scenario start
    /// (before any Step's `current_step.store` advance), so any
    /// capture taken before the first Step transition stamps with
    /// this value.
    pub const BASELINE: Self = Self(0);

    /// Construct a `Phase` for the `zero_indexed`-th scenario Step.
    /// The 1-indexed encoding (Step 0 → `u16` 1, Step 1 → `u16` 2,
    /// ...) keeps `BASELINE` unambiguous at `u16` 0. Saturates at
    /// `u16::MAX` rather than overflowing — a scenario with > 65k
    /// Steps is pathological and the saturating value still
    /// distinguishes "well past any real Step" from BASELINE.
    pub const fn step(zero_indexed: u16) -> Self {
        Self(zero_indexed.saturating_add(1))
    }

    /// True iff this is `Phase::BASELINE` (the pre-first-Step
    /// settle window).
    pub const fn is_baseline(&self) -> bool {
        self.0 == 0
    }

    /// Inner `u16`. Use this when handing the value to a
    /// serializer / formatter / external consumer that does not
    /// understand the newtype. Production callers that build a
    /// `Phase` for downstream comparison should prefer
    /// `Phase::BASELINE` / `Phase::step(k)` over wrapping a raw
    /// `u16` themselves.
    pub const fn as_u16(self) -> u16 {
        self.0
    }
}

impl std::fmt::Display for Phase {
    /// `"BASELINE"` for [`Phase::BASELINE`], `"Step[k]"` for
    /// [`Phase::step`] (decoded back via the 1-indexed
    /// encoding). Matches the labels [`PhaseBucket`] embeds in
    /// `label` so operators see consistent phase identifiers
    /// across structured-sidecar reads and ad-hoc `format!`
    /// output.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_baseline() {
            write!(f, "BASELINE")
        } else {
            write!(f, "Step[{}]", self.0 - 1)
        }
    }
}

impl From<u16> for Phase {
    /// Wrap a raw 1-indexed encoded value as a [`Phase`]. Production
    /// paths that already have the encoded value (e.g. drained from
    /// the host-side mirror of `current_step`, or read out of a
    /// deserialized sidecar) construct the typed wrapper via this
    /// conversion without re-deriving the encoding.
    fn from(value: u16) -> Self {
        Self(value)
    }
}

impl From<Phase> for u16 {
    fn from(value: Phase) -> Self {
        value.0
    }
}

/// Per-phase metric bucket — one entry per scenario phase in
/// [`ScenarioStats::phases`].
///
/// A scenario with N Steps yields `N + 1` phases: phase 0 is the
/// BASELINE (pre-first-Step settle window), and phases 1..=N
/// correspond to Step 0..Step N-1 in scenario order. The
/// 1-indexed Step encoding (instead of 0-indexed) lets BASELINE
/// own `step_index = 0` unambiguously — a `step_index = 0` sample
/// is always settle, not first-Step.
///
/// Each bucket carries the metric values reduced over the phase's
/// sample window. For `crate::stats::MetricKind::Counter`
/// metrics the reduction is `last - first` across the phase's
/// periodic samples (cumulative-counter delta); for `Gauge` /
/// `Peak` / `Timestamp` it dispatches per the kind via
/// `crate::stats::aggregate_samples`. Missing metric keys mean
/// the phase had no finite samples for that metric.
///
/// Metric keys match `crate::stats::MetricDef::name` — see
/// `crate::stats::METRICS` for the canonical list of registered
/// metric names a `get` / `phase_metric` lookup expects.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize, crate::Claim)]
pub struct PhaseBucket {
    /// Phase index. `0` = BASELINE (pre-first-Step settle window).
    /// `1..=N` align with Step ordinals (1-indexed): Step 0 of the
    /// scenario lives at `step_index = 1`, Step 1 at
    /// `step_index = 2`, etc. The encoding avoids the collision
    /// where a 0-indexed Step would share `step_index = 0` with
    /// the BASELINE settle window.
    pub step_index: u16,
    /// Human-readable label. `"BASELINE"` for `step_index = 0`,
    /// `"Step[0]"` / `"Step[1]"` / ... for `step_index = 1..=N`.
    /// Mirrors the formatting used by
    /// `crate::timeline::Timeline`'s phase rendering so operator
    /// inspection of the formatted diagnostic and the structured
    /// sidecar yield the same phase identifiers.
    pub label: String,
    /// Phase window start: the MINIMUM per-sample time anchor in the
    /// phase — each sample's `boundary_offset_ms`, falling back to its
    /// `elapsed_ms`. Samples with neither anchor (both `None` — a
    /// not-measured timestamp) are excluded from the min.
    pub start_ms: u64,
    /// Phase window end: the MAXIMUM per-sample time anchor in the
    /// phase (the same `boundary_offset_ms`-or-`elapsed_ms` key as
    /// `start_ms`). A phase whose every sample is unanchored yields the
    /// inverted window `(start_ms = u64::MAX, end_ms = 0)`, which folds
    /// no monitor samples. Downstream renderers should not assume the
    /// value is closed against a stimulus event.
    pub end_ms: u64,
    /// Number of periodic samples bucketed into this phase. Zero
    /// when the phase fired no captures: BASELINE when the settle window
    /// was shorter than the periodic interval, OR a synthesized
    /// capture-free interior step (the
    /// `build_phase_buckets_with_stimulus` seam — a `StepStart`-step
    /// whose window held no periodic boundary still gets a bucket so its
    /// capture-independent `iteration_rate` is not dropped).
    pub sample_count: usize,
    /// Per-metric phase-aggregated values. See the [`PhaseBucket`]
    /// struct doc for the registry key source and per-kind reduction
    /// dispatch; missing keys mean the phase carried no finite
    /// samples for that metric (sentinel-free: `None` from the
    /// reducer surfaces as "key absent" rather than "value 0.0").
    pub metrics: std::collections::BTreeMap<String, f64>,
    /// Per-cgroup raw telemetry components for this phase, keyed by cgroup
    /// name (see [`PhaseCgroupStats`]). Empty until a capture path populates
    /// it; the structural carrier for the per-phase per-cgroup distributional
    /// re-pool. Whole-run = aggregate of these per-phase per-cgroup components.
    ///
    /// An ORPHAN bucket — a guest carrier whose `step_index` has NO paired host
    /// bucket (a dropped/absent StepStart frame, or a stimulus-less host/fixture
    /// path; NOT merely a short step, since `build_phase_buckets_with_stimulus`
    /// synthesizes a bucket for every StepStart so a captured-but-short step
    /// takes the matched arm) — is carried by
    /// `fold_guest_per_cgroup_into_host_buckets` with the shape
    /// `(start_ms, end_ms) == (0, 0)` AND empty `metrics` AND non-empty
    /// `per_cgroup` (it carries only these components). On every non-zero-duration
    /// window that shape is the orphan arm's, so the timeline render keys on it
    /// to surface "window not measured" rather than a misleading `0ms` (see
    /// `crate::timeline::phase_from_bucket`): a captured bucket has metrics. A
    /// zero-duration step at scenario start (`StepStart==StepEnd==0`) can also
    /// produce it via the matched arm, but harmlessly — a zero-duration step has
    /// no window, so "not measured" reads the same as "0ms".
    pub per_cgroup: std::collections::BTreeMap<String, PhaseCgroupStats>,
}

impl PhaseBucket {
    /// Look up the phase-aggregated value for `metric_name` (see
    /// [`PhaseBucket::metrics`] for the registry source). Returns
    /// `None` when the phase carried no finite samples for that
    /// metric — distinct from `Some(0.0)` which means the reducer
    /// produced a real zero from finite samples.
    pub fn get(&self, metric_name: &str) -> Option<f64> {
        self.metrics.get(metric_name).copied()
    }

    /// Like [`Self::get`], but panics with a diagnostic message citing
    /// the bucket's `step_index` + `label` + `sample_count` + the set
    /// of metric keys actually present when the metric is absent. Use
    /// when the caller knows the metric MUST be in the bucket (the
    /// phase fired samples and the metric is registered — see
    /// [`PhaseBucket::metrics`]) — the panic message tells the operator whether the cause is
    /// "phase produced no samples" (sample_count of 0) or "metric key
    /// typo" (positive sample_count but the key isn't in `metrics`).
    ///
    /// ```ignore
    /// let bucket = r.stats.step(0).expect("Step[0] phase");
    /// let throughput = bucket.expect_metric("throughput");
    /// ```
    pub fn expect_metric(&self, metric_name: &str) -> f64 {
        self.get(metric_name).unwrap_or_else(|| {
            panic!(
                "PhaseBucket::expect_metric: metric '{}' absent from phase \
                 step_index={} ('{}') with sample_count={}. \
                 metric keys present in this bucket: {:?}. \
                 Possible causes: (a) phase carried 0 samples for this \
                 metric (sample_count==0 means no captures landed in the \
                 phase at all; sample_count>0 means captures landed but \
                 the metric extracted no finite values from them); \
                 (b) metric name typo (verify against \
                 ScenarioStats::is_known_metric / known_metrics).",
                metric_name,
                self.step_index,
                self.label,
                self.sample_count,
                self.metrics.keys().collect::<Vec<_>>(),
            )
        })
    }
}

/// Merge two [`PhaseBucket`]s sharing the same `step_index` per
/// the per-MetricKind dispatch in [`crate::stats::MergeKind`].
/// Called by [`AssertResult::merge`] for matched buckets;
/// unmatched buckets are appended verbatim by the caller.
///
/// Window-invariant merge:
/// - `step_index`: equal by precondition (caller pairs buckets by
///   `step_index`), kept from `a`.
/// - `label`: kept from `a`. By construction the label is derived
///   purely from `step_index` (`"BASELINE"` / `"Step[k]"`) so both
///   sides agree.
/// - `start_ms`: `min(a.start_ms, b.start_ms)` so the merged
///   window covers the earliest start of either side.
/// - `end_ms`: `max(a.end_ms, b.end_ms)` so the merged window
///   covers the latest end. Drives the [`crate::stats::MergeKind::NonCommutative`]
///   tiebreak on Gauge(Last) / Timestamp metrics — the value
///   from the bucket whose `end_ms` is later wins.
/// - `sample_count`: `a + b`. Used as the weighting denominator
///   for the `MetricKind::Gauge(GaugeAgg::Avg)` weighted mean.
///
/// Per-metric merge dispatches on the metric's `crate::stats::MetricKind`
/// from the registry via [`crate::stats::metric_def`]:
/// - `MetricKind::Counter` → `a + b` (the two reduced values are
///   per-phase deltas; the merge across cgroups sums per-cgroup
///   contributions to the phase delta, mirroring how
///   `ScenarioStats::total_migrations` adds across cgroups).
/// - `MetricKind::Peak` and `MetricKind::Gauge(GaugeAgg::Max)` →
///   `max(a, b)` (the worst-case "peak that fired" survives).
/// - `MetricKind::Gauge(GaugeAgg::Avg)` → weighted mean
///   `(a * a_w + b * b_w) / (a_w + b_w)` where `a_w = a_count.max(1)`
///   and `b_w = b_count.max(1)` — the unbiased combination of both
///   sides' per-phase means weighted by sample population, each weight
///   floored at 1. The `.max(1)` floor (mirroring
///   `populate_run_ext_metrics_from_phases`) keeps a synthesized
///   zero-capture bucket's capture-independent Gauge(Avg) value
///   contributing one phase-observation of weight rather than being
///   zero-weighted out of
///   the merge — the silent-drop the synthesize seam exists to prevent.
///   With both counts > 0 the floor is a no-op (the plain
///   sample-population weighting); both counts zero degenerates to
///   `(a + b) / 2.0`.
/// - `MetricKind::Gauge(GaugeAgg::Last)` and `MetricKind::Timestamp`
///   → value from the bucket with the larger `end_ms`; ties keep
///   `a`'s value. Captures the "latest-sample-wins" semantic per
///   the [`crate::stats::MergeKind::NonCommutative`] contract.
/// - `MetricKind::Rate { .. }` → SKIPPED in the per-key fold and
///   re-derived from the pooled components by
///   [`crate::stats::derive_rate_metrics`] as a post-pass, so the
///   merged rate is `Σnumerator / Σdenominator` (each component
///   folds by its own kind first) rather than a fold of two
///   ready-made per-phase ratios.
///
/// Unregistered metric names (not in `crate::stats::METRICS`)
/// fall back to a commutative arithmetic mean
/// `(a + b) / 2.0`. The mean is the safest default for an unknown
/// kind: sum would over-count Gauge / Timestamp values, max would
/// lose Counter / Avg signal, and "last" requires a tiebreak the
/// caller can't compute without the kind. Producers attaching
/// unregistered metrics to a `PhaseBucket` should add them to
/// `METRICS` to get the typed merge instead of the fallback.
fn merge_matched_phase_buckets(a: PhaseBucket, b: PhaseBucket) -> PhaseBucket {
    assert_eq!(
        a.step_index, b.step_index,
        "merge_matched_phase_buckets: caller must pair by step_index",
    );
    let mut metrics = std::collections::BTreeMap::new();
    // Collect every key present on either side; iterate once,
    // dispatching per the kind of the key (or the unregistered
    // mean fallback) so the merge is single-pass.
    let mut keys: std::collections::BTreeSet<&String> = a.metrics.keys().collect();
    keys.extend(b.metrics.keys());
    for key in keys {
        // Derived metrics (Rate / Distribution / WorstLowest) are NOT merged
        // here: a Rate re-derives from the merged components in the post-pass
        // below, and Distribution / WorstLowest are re-pooled run-level by
        // `populate_run_distribution_metrics` (they never appear in
        // phase.metrics — `aggregate_samples_for_phase` returns None — so this
        // skip is also a structural guard). Folding a ready-made derived value
        // would lose the re-pool.
        if crate::stats::metric_def(key).is_some_and(|m| m.kind.is_derived()) {
            continue;
        }
        let av = a.metrics.get(key).copied();
        let bv = b.metrics.get(key).copied();
        let merged = match (av, bv) {
            (Some(av), Some(bv)) => {
                let kind = crate::stats::metric_def(key).map(|m| m.kind);
                merge_metric_values(
                    kind,
                    av,
                    bv,
                    a.sample_count,
                    b.sample_count,
                    a.end_ms,
                    b.end_ms,
                )
            }
            (Some(v), None) | (None, Some(v)) => v,
            (None, None) => continue,
        };
        metrics.insert(key.clone(), merged);
    }
    // Re-derive Rate metrics from the now-pooled components: each
    // component merged by its own kind above (a Counter numerator
    // summed), so the rate becomes Σnumerator / Σdenominator — the
    // correct re-pool, not a mean of the two phases' ready-made ratios.
    crate::stats::derive_rate_metrics(&mut metrics);
    // Union per_cgroup by cgroup name: a cgroup present on both sides folds
    // its raw components per PhaseCgroupStats::merge (concat samples, sum
    // counters, combine extremes); a cgroup on only one side is carried
    // verbatim. Empty ∪ empty = empty, so this is a no-op until a capture
    // path populates per_cgroup (the structural-carrier invariant).
    let mut per_cgroup = a.per_cgroup;
    for (name, b_cg) in b.per_cgroup {
        match per_cgroup.remove(&name) {
            Some(a_cg) => {
                per_cgroup.insert(name, PhaseCgroupStats::merge(a_cg, b_cg));
            }
            None => {
                per_cgroup.insert(name, b_cg);
            }
        }
    }
    PhaseBucket {
        step_index: a.step_index,
        label: a.label,
        start_ms: a.start_ms.min(b.start_ms),
        end_ms: a.end_ms.max(b.end_ms),
        sample_count: a.sample_count + b.sample_count,
        metrics,
        per_cgroup,
    }
}

/// Fold the guest-collected per-phase `per_cgroup` carriers into the
/// host-rebuilt phase buckets, keyed by `step_index`.
///
/// 6b: the host rebuilds phase buckets from the periodic-capture series
/// (window + metric folds), but those buckets carry an empty `per_cgroup`
/// by construction. The guest collects per-cgroup RAW components per step
/// ([`crate::scenario::collect_handles`] under `collect_step`) into carrier
/// buckets whose only payload is `per_cgroup` — a merge-neutral
/// `(u64::MAX, 0)` window and empty `metrics`. Guest and host `step_index`
/// are the SAME 1-indexed value: the step loop stamps
/// `phase_step_index = step_idx + 1` onto BOTH the `StepStart` frames the
/// host rebuilds buckets from AND the `collect_step` carrier, so pairing by
/// `step_index` is exact and cannot drift.
///
/// Each guest carrier whose `step_index` matches a host bucket folds its
/// `per_cgroup` in via [`merge_matched_phase_buckets`] — a no-op on the
/// host's window (`min`/`max` against `MAX`/`0`), metrics (the carrier has
/// none, so each host key is carried verbatim), and `sample_count` (`+ 0`),
/// contributing ONLY the unioned `per_cgroup`. A guest `step_index` with no
/// host bucket — a DEFENSIVE case: the carrier's `step_index` has no `StepStart`
/// frame in the host stimulus timeline (a dropped/absent stimulus frame, or a
/// stimulus-less host/fixture path), since `build_phase_buckets_with_stimulus`
/// SYNTHESIZES a capture-free bucket for every StepStart-step, so a
/// captured-but-short step takes the matched arm above, not this one — is carried
/// verbatim with its window normalized to `(0, 0)` so duration consumers
/// (`end_ms - start_ms`) never underflow the merge-neutral sentinel — no
/// `per_cgroup` datum is silently dropped. With no guest carriers (a run
/// with no step-local cgroups) the host buckets pass through unchanged. The
/// returned vec is sorted by `step_index`.
pub(crate) fn fold_guest_per_cgroup_into_host_buckets(
    host_buckets: Vec<PhaseBucket>,
    guest_buckets: Vec<PhaseBucket>,
) -> Vec<PhaseBucket> {
    let host_len = host_buckets.len();
    // No-silent-drops: host buckets have unique step_index
    // (build_phase_buckets_with_stimulus emits one bucket per step_index), but
    // fold same-step_index duplicates via merge rather than a last-wins collect so
    // a future producer that violated the invariant DEGRADES to a merge, never a
    // silent release-mode drop. The debug_assert still trips loudly in test/debug.
    let mut by_idx: std::collections::BTreeMap<u16, PhaseBucket> = std::collections::BTreeMap::new();
    for b in host_buckets {
        match by_idx.remove(&b.step_index) {
            Some(existing) => {
                by_idx.insert(b.step_index, merge_matched_phase_buckets(existing, b));
            }
            None => {
                by_idx.insert(b.step_index, b);
            }
        }
    }
    debug_assert_eq!(
        by_idx.len(),
        host_len,
        "host buckets must have unique step_index; a collision merged (not dropped)",
    );
    for gb in guest_buckets {
        // Every guest carrier MUST carry the merge-neutral (u64::MAX, 0) sentinel
        // window (the step_per_cgroup_bucket invariant). Validate it BEFORE the
        // match so BOTH arms are guarded: the matched arm relies on the window
        // being merge-neutral (min/max no-op against the host window), and the
        // orphan arm normalizes it to (0,0). A future caller handing a
        // real-window carrier (incl. a duplicate orphan via the matched arm)
        // trips loudly instead of silently corrupting the merged window.
        debug_assert!(
            gb.start_ms == u64::MAX && gb.end_ms == 0,
            "guest carrier must carry the merge-neutral (u64::MAX, 0) window; got ({}, {})",
            gb.start_ms,
            gb.end_ms,
        );
        match by_idx.remove(&gb.step_index) {
            Some(hb) => {
                by_idx.insert(gb.step_index, merge_matched_phase_buckets(hb, gb));
            }
            None => {
                // Orphan arm: a guest carrier whose step_index has no host bucket.
                //
                // Invariant: build_phase_buckets_with_stimulus synthesizes a host
                // bucket for every StepStart-step, so a carrier whose step has a
                // StepStart frame always takes the matched arm above. This arm is
                // reached only by a carrier whose step has NO StepStart frame —
                // defensive, not produced by normal capture.
                //
                // Normalize the merge-neutral sentinel window to (0,0) so duration
                // consumers don't underflow it. The resulting (0,0)-window +
                // empty-metrics + non-empty-per_cgroup shape is the orphan
                // signature the timeline render keys on to show "window not
                // measured" instead of a misleading 0ms — the (0,0) means "no host
                // window known", NOT a measured zero-duration step.
                //
                // A zero-duration step at scenario start (StepStart==StepEnd==0)
                // produces the same shape via the matched arm, but harmlessly: a
                // zero-duration step has no window, so the render's "not measured"
                // reads the same as "0ms". See `crate::timeline::phase_from_bucket`.
                let mut orphan = gb;
                orphan.start_ms = 0;
                orphan.end_ms = 0;
                by_idx.insert(orphan.step_index, orphan);
            }
        }
    }
    by_idx.into_values().collect()
}

/// Per-metric merge inner helper used by
/// [`merge_matched_phase_buckets`]. Dispatches on the metric's
/// `crate::stats::MetricKind` (or the unregistered fallback)
/// to combine two reduced values into one.
///
/// `a_count` / `b_count` are the source buckets' `sample_count`
/// fields, used as weights for `Gauge(Avg)`. `a_end_ms` /
/// `b_end_ms` are the source buckets' window-end timestamps,
/// used to pick the later sample for `Gauge(Last)` / `Timestamp`.
fn merge_metric_values(
    kind: Option<crate::stats::MetricKind>,
    a: f64,
    b: f64,
    a_count: usize,
    b_count: usize,
    a_end_ms: u64,
    b_end_ms: u64,
) -> f64 {
    use crate::stats::{GaugeAgg, MetricKind};
    match kind {
        // Counter (cumulative) and DeltaSum (sum of per-read deltas)
        // both merge across AssertResults by summing the reduced values
        // (commutative — see MetricKind::merge_kind).
        Some(MetricKind::Counter) | Some(MetricKind::DeltaSum) => a + b,
        Some(MetricKind::Peak) | Some(MetricKind::Gauge(GaugeAgg::Max)) => a.max(b),
        Some(MetricKind::Gauge(GaugeAgg::Avg)) => {
            // Weight by sample_count, floored at 1: a sample_count==0
            // bucket carrying a capture-independent Gauge(Avg) value must
            // contribute one phase-observation of weight, not be
            // zero-weighted out of the merge. Mirrors the .max(1) floor in
            // populate_run_ext_metrics_from_phases. With both counts > 0
            // the floor is a no-op (the prior sample_count weighting);
            // with both 0 each still floors to weight 1, giving the
            // (a+b)/2 equal-weight mean (the aggregate_samples_weighted
            // zero-total-weight fallback is unreachable from here).
            // (iteration_rate — the original synthesized zero-capture case —
            // is now a MetricKind::Rate: merge_matched_phase_buckets skips it
            // via the Rate `continue` in its key loop (above this fn) and
            // re-pools it from its summed Counter components, so a Rate value
            // never reaches this Gauge(Avg) fold.)
            let a_w = a_count.max(1) as f64;
            let b_w = b_count.max(1) as f64;
            (a * a_w + b * b_w) / (a_w + b_w)
        }
        Some(MetricKind::Gauge(GaugeAgg::Last)) | Some(MetricKind::Timestamp) => {
            if b_end_ms > a_end_ms { b } else { a }
        }
        // Derived kinds (Rate / Distribution / WorstLowest) are skipped in
        // the merge loop (see `merge_matched_phase_buckets`'s `is_derived`
        // continue) and produced post-merge (`derive_rate_metrics` /
        // `populate_run_distribution_metrics`), so a derived value never
        // reaches this per-value merge — folding a ready-made derived value
        // would lose the re-pool.
        Some(MetricKind::Rate { .. })
        | Some(MetricKind::Distribution { .. })
        | Some(MetricKind::WorstLowest { .. })
        | Some(MetricKind::WakeLatencyTailRatio) => unreachable!(
            "derived metrics (Rate/Distribution/WorstLowest/WakeLatencyTailRatio) are produced post-merge, not merged as values"
        ),
        // Unregistered metric: commutative mean fallback. Sum
        // would over-count Gauge values; max would lose Counter
        // signal; "last" needs a tiebreak the caller can't
        // compute without the kind. Mean is the safest commutative
        // default.
        None => (a + b) / 2.0,
    }
}

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
    /// phases. For a single phase's balance in a multi-phase scenario, use
    /// the per-`Phase` variant once per-phase per-cgroup stats land.
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
/// same key into `ext_metrics` would create unread sidecar bloat AND a
/// cross-RUN aggregation-drift class (the typed-path reduction vs the
/// ext-path kind-aware dispatch can produce different values for one metric —
/// e.g. `stuck_count`'s typed whole-run flag vs an ext per-phase stall-count
/// sum). Both run-level ext-metrics populators consult this — the SampleSeries
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
        // from the phase fold: the ext copy would be unread bloat and, for a
        // key whose typed and ext reductions differ (stuck_count: whole-run
        // flag vs per-phase stall-count sum), a cross-RUN aggregation-drift
        // trap. Their per-phase PhaseBucket value still feeds rendering; the
        // run-level value stays the typed path. Mirrors the sibling
        // populate_run_ext_metrics. (Without this, folding max_imbalance_ratio
        // + stuck_count onto captured buckets would leak both into ext_metrics
        // on the common path.)
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
        // completion of the per-step #4 fix (iteration_rate itself is a
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
/// carriers store raw ns, per [`PhaseCgroupStats::run_delays_ns`]).
///
/// CARRIER-LESS FOLD (backdrop coverage + graceful degradation): a cgroup
/// whose raw samples are NOT in the pool — a backdrop (collected with no
/// step_index, so no per-phase carrier; 6c/#36) or a cgroup whose carrier was
/// stripped/empty (`strip_phase_cgroup_samples`) — is NOT dropped. Its
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
/// there, not a proxy. #36 will give backdrops a carrier so their samples join
/// the pool.
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
    let mut wake_pool: Vec<u64> = Vec::new();
    let mut run_delay_pool: Vec<u64> = Vec::new();
    // Names of cgroups that contributed NON-EMPTY samples to each pool. A
    // cgroup absent here — a backdrop (carrier attachment is step-local only,
    // collect_handles gates on a Some step_index; backdrops collect with None,
    // 6c/#36) or a stripped/empty carrier — is NOT dropped from the run-level
    // Distribution: the re-pool folds its surviving per-cgroup CgroupStats
    // reduction worst-wins (see `populate_run_distribution_metrics_from`). #36
    // will give backdrops a carrier so their raw samples join the pool.
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
    // step. #36 (backdrops join the pool) dissolves the dedup entirely.
    let mut wake_carriers: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
    let mut run_delay_carriers: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
    for phase in &stats.phases {
        for (cgname, pcg) in &phase.per_cgroup {
            if !pcg.wake_latencies_ns.is_empty() {
                wake_pool.extend_from_slice(&pcg.wake_latencies_ns);
                wake_carriers.insert(cgname.as_str());
            }
            if !pcg.run_delays_ns.is_empty() {
                run_delay_pool.extend_from_slice(&pcg.run_delays_ns);
                run_delay_carriers.insert(cgname.as_str());
            }
        }
    }
    wake_pool.sort_unstable();
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
fn populate_run_distribution_metrics_from<'a>(
    target: &mut std::collections::BTreeMap<String, f64>,
    metrics: impl Iterator<Item = (&'a str, crate::stats::MetricKind)>,
    wake_pool: &[u64],
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
                let (pool, carriers) = match source {
                    SampleSource::WakeLatencyNs => (wake_pool, wake_carriers),
                    SampleSource::RunDelayNs => (run_delay_pool, run_delay_carriers),
                };
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
                let mut v: Option<f64> = if pool.is_empty() {
                    None
                } else {
                    Some(reduce_sorted_distribution(pool, reduction))
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
fn reduce_sorted_distribution(sorted: &[u64], reduction: crate::stats::SampleReduction) -> f64 {
    use crate::stats::SampleReduction;
    match reduction {
        SampleReduction::P99 => percentile(sorted, 0.99) as f64 / 1000.0,
        SampleReduction::Median => percentile(sorted, 0.5) as f64 / 1000.0,
        SampleReduction::Cv => {
            let n = sorted.len() as f64;
            let mean_ns = sorted.iter().sum::<u64>() as f64 / n;
            if mean_ns > 0.0 {
                let variance =
                    sorted.iter().map(|&v| (v as f64 - mean_ns).powi(2)).sum::<f64>() / n;
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
                debug_assert!(false, "no CgroupStats run-delay reduction for {reduction:?}");
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

/// Phase buckets attributed against the guest stimulus timeline, then
/// enriched with stimulus-event-derived per-phase `iteration_rate`.
///
/// Unlike the plain [`build_phase_buckets`] (which groups by the
/// bridge-stamped step_index), this re-groups each periodic capture by
/// the guest step whose stimulus window contains the capture's
/// workload-relative boundary offset (`Sample::boundary_offset_ms`).
/// That offset is derived from the boundary schedule rather than the
/// fire time, so it is immune to the deferred-fire burst that makes
/// every capture stamp the same late CURRENT_STEP (the
/// `phases.len() == 1` collapse). Captures with no offset (on-demand /
/// fixture) fall back to their stamped step_index. Because the bucket
/// windows are then workload-relative, the run-relative monitor samples
/// are shifted by the stimulus/monitor clock skew before windowing.
///
/// Additionally synthesizes a capture-free `PhaseBucket`
/// (`sample_count == 0`) for any stimulus `StepStart`-step that
/// captured no periodic samples — the uniform whole-workload boundary
/// placement (`compute_periodic_boundaries_ns`) is step-agnostic, so a
/// short interior step can land zero captures and otherwise leave no
/// bucket, silently dropping its capture-independent `iteration_rate`.
/// The synthesized bucket carries the step's full stimulus window so its
/// `iteration_rate` (from `StepStart`/`StepEnd` deltas) and
/// `avg_imbalance_ratio` (from in-window monitor samples) are still
/// recovered. The returned vec therefore holds one bucket per
/// (captured phase ∪ `StepStart`-step), sorted by `step_index` — NOT
/// one-per-captured-phase, so `len()` is no longer "number of captured
/// phases".
///
/// The `iteration_rate` enrichment lets
/// `crate::timeline::Timeline::from_phase_buckets` render the per-phase
/// throughput annotation without going through the legacy
/// `crate::timeline::Timeline::build` path.
///
/// For each `StepStart[k]` -> `StepEnd[k]` pair with
/// `total_iterations: Some(_)`, the per-phase rate is
/// `(later - earlier) / duration_s` where `duration_s` is the
/// elapsed-ms delta BETWEEN THE TWO STIMULUS EVENTS (guest clock),
/// not the PhaseBucket sample window. The rate is attributed to the
/// step the EARLIER event starts (`prev.step_index`); the attribution
/// loop skips any `is_step_end` (or `is_terminal`) `prev`, so only a
/// StepStart is ever the earlier member. Phases that don't overlap a
/// stimulus pair keep their PhaseBucket.metrics map unchanged (no
/// iteration_rate key).
///
/// SEMANTICS: `total_iterations` is the sum of the worker handles
/// alive at each event (see
/// [`crate::timeline::StimulusEvent::total_iterations`]). Each step's
/// rate is its STEP-LOCAL `StepStart[k]` -> `StepEnd[k]` delta — the
/// step's own workers measured over its own hold — so a bucket is
/// sourced ONLY by its own pair (the `is_step_end` guard drops the
/// inter-step `StepEnd[k]` -> `StepStart[k+1]` pair entirely). This
/// measures BOTH fresh-per-step workers (which read ~0 at each
/// StepStart, so the old cross-step delta produced no rate) and
/// persistent (Backdrop) workers (excluding the inter-step teardown
/// wall-time a cross-step window would span). On a clean run the
/// `(StepEnd[N], terminal)` pair is guard-skipped and the trailing
/// `is_terminal` event is not consumed; it supplies a step's right
/// boundary ONLY for legacy/synthetic data carrying a `ScenarioEnd`
/// frame but no `StepEnd` frames. A sched-died step has neither frame
/// (its early return skips both emissions), so the dead step's
/// `StepStart` is never a `prev` with a successor and it reports no
/// rate.
///
/// iteration_rate is registered as `MetricKind::Rate` with the Counter
/// components `total_phase_iterations` / `total_phase_duration_sec` and
/// `HigherBetter` polarity (more throughput is better). The per-step
/// producer below emits those two components (the iteration delta and the
/// window seconds — the ms→s `/1000` applied at the component, since
/// `derive_rate_metrics` does a bare num/den) rather than a ready ratio,
/// and the `derive_rate_metrics` post-pass re-derives `iteration_rate` =
/// Σiterations / Σseconds at every in-map aggregation level. Its per-run
/// run-scalar fold (one run's per-phase values → that run's `ext_metrics`)
/// runs through `populate_run_ext_metrics_from_phases`, which SUMS the
/// Counter components across phases (a synthesized zero-capture phase's
/// components are summed in, not zero-weighted out — the run-aggregate
/// completion of the per-step #4 fix) and re-derives the rate. The
/// cross-sidecar-run rollup `group_and_average_by` likewise re-pools via
/// its `derive_rate_metrics` post-pass. `iteration_rate` has no cross-cgroup
/// axis to re-pool: it is derived from run-level phase buckets and
/// host-injected into the run `ext_metrics` by
/// `populate_run_ext_metrics_from_phases` (the eval layer) AFTER the
/// cross-cgroup `merge`, so `AssertResult::merge`'s worst-case
/// (min/max-by-polarity) `ext_metrics` fold never sees its components. The rate whose components ARE per-cgroup is the separate
/// pooled `iterations_per_cpu_sec`, re-pooled across a run's cgroups by
/// `populate_run_pooled_iterations_per_cpu_sec` (reading `stats.cgroups`
/// post-merge).
///
/// Live caller: `evaluate_vm_result` at `src/test_support/eval/mod.rs`
/// — has both the SampleSeries and the stimulus_events vec in scope.
pub fn build_phase_buckets_with_stimulus(
    samples: &crate::scenario::sample::SampleSeries,
    stimulus_events: &[crate::timeline::StimulusEvent],
) -> Vec<PhaseBucket> {
    let monitor_samples: &[crate::monitor::MonitorSample] =
        samples.monitor().map(|m| m.samples()).unwrap_or(&[]);
    // Re-group periodic captures by the guest step whose stimulus
    // window contains each capture's workload-relative boundary offset,
    // NOT the step_index stamped at (deferred) fire time — see
    // [`crate::scenario::sample::SampleSeries::by_stimulus_phase`] for
    // why the scheduled offset is the timing-independent truth (it
    // survives the deferred-fire burst that collapses the stamped
    // step_index to one phase).
    let by_phase = samples.by_stimulus_phase(stimulus_events);
    // Bucket windows are workload-relative (boundary-offset) here, so the
    // monitor samples (run-relative) are shifted by the stimulus/monitor
    // clock skew before windowing.
    let monitor_to_window_offset_ms = monitor_clock_offset(stimulus_events, monitor_samples);
    let mut buckets = buckets_from_grouped(by_phase, monitor_samples, monitor_to_window_offset_ms);
    // Synthesize a PhaseBucket for any scenario step that has a stimulus
    // StepStart but produced no capture bucket. Periodic boundaries are
    // placed uniformly over the whole workload (step-agnostic — see
    // `compute_periodic_boundaries_ns`), so a short interior step can
    // capture zero samples and leave no bucket. The iteration_rate
    // attribution loop below only mutates EXISTING buckets, so without
    // this seam that step's capture-independent rate (derived purely from
    // the StepStart/StepEnd total_iterations deltas, needing no capture)
    // would be silently dropped. The synthesized bucket carries the
    // step's true stimulus window so `fold_monitor_into_bucket` still
    // recovers its monitor-derived imbalance; `sample_count == 0` marks
    // it capture-free for downstream consumers. `Timeline::from_phase_buckets`
    // COMPARES this bucket's stimulus-derived throughput across the gap
    // (the iteration_rate is real, not a sampling artifact) but GATES the
    // monitor-derived metrics (imbalance/dsq/fallback/keep_last) behind
    // both sides having samples — see `detect_boundary_changes`. BASELINE
    // (step 0) is synthesized only if a StepStart carries
    // `step_index == 0` — it is not special-cased.
    let step_starts = crate::scenario::sample::step_starts_from_stimulus(stimulus_events);
    for &(start_ms, k) in &step_starts {
        if buckets.iter().any(|b| b.step_index == k) {
            continue;
        }
        // Right edge: the step's own StepEnd, CLAMPED to the next
        // step-start so a non-monotonic (corrupt-wire) StepEnd[k] >=
        // StepStart[k+1] can never extend this synthesized window past the
        // next step's start (which would fold the same MonitorSample into
        // two adjacent synthesized buckets). Falls back to the next
        // step-start alone, else open-ended (the last step on data with no
        // StepEnd — e.g. sched-died — which the rate loop leaves rateless).
        // Earliest StepEnd elapsed for k (min, symmetric with start_ms's
        // earliest step-start), so a duplicate/corrupt StepEnd later in the
        // vec can't pick a wrong right edge.
        let step_end = stimulus_events
            .iter()
            .filter(|e| e.is_step_end && e.step_index == Some(k))
            .map(|e| e.elapsed_ms)
            .min();
        // `ms > start_ms` (strict): two DISTINCT step_index sharing the
        // same elapsed is corrupt-wire only (build_stimulus emits StepStarts
        // at distinct monotone guest-clock times), so the equal-elapsed
        // window-overlap edge is out of scope — its worst case is a
        // redundant double-fold of one monitor sample, never a drop or panic.
        let next_start = step_starts
            .iter()
            .filter(|&&(ms, _)| ms > start_ms)
            .map(|&(ms, _)| ms)
            .min();
        // Last-resort right edge for a last step with no StepEnd and no
        // successor: the scenario-end terminal event, bounding the window
        // at run-end instead of folding every trailing teardown monitor
        // sample into the bucket. NOTE the production sched-died path emits
        // NEITHER a StepEnd NOR a ScenarioEnd (its early return skips both),
        // so it carries no terminal here and falls to u64::MAX — matching
        // Timeline::build, which extends the last phase to end-of-monitor.
        // This terminal clamp therefore only binds for legacy/synthetic
        // data carrying a ScenarioEnd frame without StepEnd frames.
        let terminal = stimulus_events
            .iter()
            .find(|e| e.is_terminal)
            .map(|e| e.elapsed_ms);
        let end_ms = match (step_end, next_start) {
            (Some(se), Some(ns)) => se.min(ns),
            (Some(se), None) => se,
            (None, Some(ns)) => ns,
            (None, None) => terminal.unwrap_or(u64::MAX),
        };
        // BASELINE (step 0) is synthesized only if a StepStart carries
        // step_index == 0; build_stimulus 1-indexes scenario Steps (Step 0
        // -> step_index 1), so it never emits step_index 0 and BASELINE is
        // intentionally never synthesized (its pre-first-Step settle window
        // carries no meaningful per-step rate). The branch is kept for the
        // convention, not for a production path.
        let label = if k == 0 {
            "BASELINE".to_string()
        } else {
            format!("Step[{}]", k.saturating_sub(1))
        };
        let mut bucket = PhaseBucket {
            step_index: k,
            label,
            start_ms,
            end_ms,
            sample_count: 0,
            metrics: std::collections::BTreeMap::new(),
            per_cgroup: std::collections::BTreeMap::new(),
        };
        fold_monitor_into_bucket(&mut bucket, monitor_samples, monitor_to_window_offset_ms);
        buckets.push(bucket);
    }
    // Keep buckets in step_index order: buckets_from_grouped emits them
    // sorted (BTreeMap key order), but the synthesize loop appends out of
    // order. Downstream lookups resolve by step_index, but a sorted vec
    // matches the captured-bucket invariant and keeps rendered output
    // stable.
    buckets.sort_by_key(|b| b.step_index);
    // Per-phase iteration_rate from stimulus event total_iterations
    // deltas. Walk events pairwise; for each pair compute the
    // rate. Sort events by elapsed_ms first so an out-of-order
    // arrival from the bulk-port drain doesn't silently lose the
    // delta to saturating_sub (the legacy Timeline::build path at
    // src/timeline.rs sorts the same way; without the sort, an
    // inversion produces duration_ms == 0 → skipped, a silent
    // drop).
    let mut sorted_events: Vec<&crate::timeline::StimulusEvent> = stimulus_events.iter().collect();
    // Total-order on an elapsed_ms tie: StepEnd before StepStart
    // (`!is_step_end` is false=0 for StepEnd) so a zero-length inter-step
    // gap at the guest's coarse-ms clock attributes the step-local
    // StepStart[k]->StepEnd[k] rate to bucket k, never the cross-step
    // StepStart[k]->StepStart[k+1] delta (the is_step_end guard below only
    // stops StepEnd from being `prev`; it does not order a StepEnd before
    // the next StepStart on a tie). Mirrors Timeline::build's sort.
    sorted_events.sort_by_key(|e| (e.elapsed_ms, !e.is_step_end));
    for w in sorted_events.windows(2) {
        let prev = w[0];
        let curr = w[1];
        // The terminal scenario-end event and per-step StepEnd events are
        // right boundaries only — never the EARLIER member of a pair (the
        // step a rate is attributed to). Both carry an end-of-hold count,
        // not a step-start, so pairing them as `prev` would attribute the
        // WRONG window to a bucket:
        // - terminal sorts last so it is naturally only ever `curr`, but
        //   guard explicitly so a future caller producing a non-last or
        //   duplicate terminal can't fall into the `None` step_index
        //   timestamp-window branch below and attach a bogus rate.
        // - StepEnd[k] carries step_index k (same as StepStart[k]); if its
        //   step's own StepStart[k] -> StepEnd[k] pair returned None (a
        //   BACKWARD count, e < s — a counter reset; a stalled step e == s
        //   instead yields a measured-ZERO rate, see
        //   StimulusEvent::rate_components), the bucket k key is still empty,
        //   so without this guard the next pair StepEnd[k] -> StepStart[k+1]
        //   would or_insert an inter-step teardown-gap rate into bucket k,
        //   mislabeling gap throughput as step k's. The guard keeps bucket k
        //   sourced ONLY by its own StepStart[k] -> StepEnd[k] pair (whether
        //   a measured zero or a real rate), never a leaked gap rate.
        if prev.is_terminal || prev.is_step_end {
            continue;
        }
        // Emit the iteration_rate Rate's two components (the per-step
        // iteration delta + window seconds) rather than the ready ratio, so
        // derive_rate_metrics re-pools Σiters/Σseconds at every aggregation
        // level. rate_components is None only when the count went BACKWARD
        // (e < s, a counter reset), an event lacks total_iterations, or the
        // window is zero-length; a non-advancing count (e == s over a
        // positive window) yields Some((0.0, secs)) — a measured zero, not
        // None (the same shared formula Timeline::build's rate_to divides
        // for display).
        let Some((iters, secs)) = prev.rate_components(curr) else {
            continue;
        };
        // Attribute the rate to the step PREV starts: the guards above
        // leave `prev` as a StepStart (or a legacy/synthetic step event),
        // so the pair (prev, curr) measures iterations accumulated from
        // prev's step-start to curr's — the throughput DURING prev's step.
        // Match by prev.step_index, NOT a timestamp window: the bucket
        // windows here are workload-relative capture offsets (the step
        // INTERIOR, 10-90% of the workload per
        // compute_periodic_boundaries_ns), so prev.elapsed_ms — the
        // step-START — lands in the inter-step gap, inside no bucket's
        // [start_ms, end_ms) window; a window match would drop every
        // rate. This mirrors the legacy Timeline::build alignment:
        // phase[i] gets events[i]→events[i+1] where events[i] is
        // phase[i]'s left boundary = prev.
        //
        // Stimulus events without a step stamp (step_index == None:
        // legacy / synthetic callers) fall back to the timestamp
        // window — half-open [start_ms, end_ms) with a single-sample
        // (start == end) equality carve-out so a boundary event lands
        // in exactly one bucket.
        for bucket in buckets.iter_mut() {
            let in_bucket = match prev.step_index {
                Some(k) => bucket.step_index == k,
                None => {
                    if bucket.start_ms == bucket.end_ms {
                        prev.elapsed_ms == bucket.start_ms
                    } else {
                        prev.elapsed_ms >= bucket.start_ms && prev.elapsed_ms < bucket.end_ms
                    }
                }
            };
            if in_bucket {
                // Both components come from the SAME (prev, curr) pair and are
                // inserted together; or_insert keeps the first pair that
                // matched this bucket (mirrors the prior first-wins
                // iteration_rate attribution). The derive_rate_metrics
                // post-pass below produces iteration_rate from them.
                bucket
                    .metrics
                    .entry("total_phase_iterations".to_string())
                    .or_insert(iters);
                bucket
                    .metrics
                    .entry("total_phase_duration_sec".to_string())
                    .or_insert(secs);
                break;
            }
        }
    }
    // Stimulus/monitor components are injected ABOVE, AFTER
    // buckets_from_grouped's own derive_rate_metrics already ran; re-derive
    // here so a Rate over a post-injected component is produced for these
    // buckets too. Idempotent: re-deriving an unchanged map is a no-op.
    for bucket in buckets.iter_mut() {
        crate::stats::derive_rate_metrics(&mut bucket.metrics);
    }
    buckets
}

/// Build per-phase metric buckets from a sample series.
///
/// Walks [`crate::scenario::sample::SampleSeries::by_stamped_phase`]
/// to group every stamped sample under its bridge-stamped
/// `step_index` (NOT re-derived from elapsed-ms windows; the
/// bridge stamp is authoritative because the capture path knows
/// the phase it fired from while the time window cannot recover
/// the phase when stimulus events arrive late or out of order).
///
/// For each phase observed (BASELINE under `step_index = 0`,
/// scenario Steps under `step_index = 1..=N` per the 1-indexed
/// phase convention) emits one [`PhaseBucket`] with `step_index`
/// as the key, `label` derived per the BASELINE/Step\[k\]
/// convention, `start_ms` / `end_ms` from the first / last
/// sample's `elapsed_ms`, `sample_count` from the bucketed
/// samples, and `metrics` from the per-kind reduction described
/// on [`PhaseBucket`]. Metrics whose per-sample reading returns
/// `None` for every sample in the bucket are omitted entirely
/// (absent → "no data") rather than collapsed to `Some(0.0)`
/// (real zero), preserving the sentinel-free contract.
///
/// Returns an empty `Vec` when the input series is empty (no
/// samples captured), distinct from returning a single empty
/// BASELINE bucket — the former means the periodic-capture path
/// never fired, the latter means it fired but no metric reading
/// came back.
///
/// Live production caller: `evaluate_vm_result` in
/// `src/test_support/eval/mod.rs` drains the snapshot bridge, builds
/// a `SampleSeries`, and routes it through this fn to populate
/// `AssertResult.stats.phases`. Exposed `pub` (not `pub(crate)`)
/// so out-of-tree consumers — payload authors writing custom
/// eval paths against the publicly-drainable
/// `result.snapshot_bridge` — can produce the same per-phase
/// aggregate shape without re-implementing the bucketing logic.
pub fn build_phase_buckets(samples: &crate::scenario::sample::SampleSeries) -> Vec<PhaseBucket> {
    // Borrowed per-tick monitor samples (None when no MonitorReport
    // was attached, e.g. host-only fixture tests).
    let monitor_samples: &[crate::monitor::MonitorSample] =
        samples.monitor().map(|m| m.samples()).unwrap_or(&[]);
    // Group by the bridge-stamped step_index. Without a stimulus
    // timeline to remap against, the stamped index is the only phase
    // signal available — see `build_phase_buckets_with_stimulus` (and
    // `SampleSeries::by_stimulus_phase`) for the offset-remap that
    // corrects a deferred-fire burst (every capture stamping the same
    // late CURRENT_STEP). The bucket window and the monitor folding
    // share the run-relative frame here, so the clock offset is `0`;
    // synthetic / legacy fixture samples carry `boundary_offset_ms =
    // None` and the window falls back to `elapsed_ms`.
    buckets_from_grouped(samples.by_stamped_phase(), monitor_samples, 0)
}

/// Per-phase CPU-time delta (ns) for one field family
/// (`stime`/`signal_stime` or `utime`/`signal_utime`), folded host-side
/// from the frozen `task_struct` enrichments captured at the phase's
/// freeze boundaries. Backs the injected `system_time_ns` /
/// `user_time_ns` per-phase metrics.
///
/// The unit is the THREAD GROUP, not the individual task: the kernel's
/// `thread_group_cputime` is `signal_struct.{u,s}time` (the accumulator
/// a dying thread's time is folded into at exit) plus the live threads'
/// `task_struct.{u,s}time`. `signal_struct` is shared across a thread
/// group, so its value is counted once per `tgid`; the live counters are
/// summed across the group's threads. For each `tgid` the group total is
/// taken at the FIRST and LAST sample in which the group had a READABLE
/// total (ordered by capture time) and `last - first` is summed across
/// groups.
///
/// Per-group first-seen/last-seen, NOT a per-sample cross-task SUM then a
/// Counter `last - first`: the captured set changes between freezes
/// (system tasks churn in and out). A task carrying a large cumulative
/// counter that appears only in a LATER sample would dump its entire
/// pre-phase history into a summed delta, inflating the phase value
/// many-fold. Subtracting each group's OWN first-seen total cancels its
/// pre-phase history, so a late-joining group contributes only what it
/// accrued while observed — bounding the result by wall-clock × cores. A
/// group's thread exiting mid-phase does not dip the total: its time
/// moves from a `task_struct` counter into `signal_struct`, both of which
/// the group total includes.
///
/// A `None` `signal_field` is a `signal_struct` translate miss, NOT a
/// real zero (which reads `Some(0)`): such a sample is omitted for that
/// group so every endpoint is a full live+signal total — mixing a
/// live-only endpoint with a live+signal one would otherwise leak the
/// cumulative accumulator as a phantom positive. A numeric `tgid` reused
/// within the phase (process exit + PID realloc) can read lower at last
/// than first; `saturating_sub` clamps that to 0 rather than wrapping.
///
/// Returns `None` when no group was observed with a readable total across
/// at least two samples — no delta is measurable — keeping an absent
/// per-phase bucket key distinct from a real `0` (a qualifying group
/// whose counters did not advance yields `Some(0.0)`). Accumulates in
/// `u128` to stay exact before the final `f64` (a phase can total many
/// task-seconds of ns).
fn phase_group_cpu_delta(
    samples: &[crate::scenario::sample::Sample<'_>],
    task_field: impl Fn(&crate::monitor::task_enrichment::TaskEnrichment) -> u64,
    signal_field: impl Fn(&crate::monitor::task_enrichment::TaskEnrichment) -> Option<u64>,
) -> Option<f64> {
    // Per-tgid `thread_group_cputime` total at one sample: the shared
    // signal_struct accumulator (once per group) plus the sum of the
    // group's live threads' task_struct counter. A group is INCLUDED for
    // a sample only when its signal accumulator was readable there (some
    // thread of the tgid carried `Some`): a `None` is a signal_struct
    // translate miss (see `task_enrichment.rs`), NOT a real zero — a real
    // zero reads `Some(0)`. Omitting unreadable-signal groups keeps every
    // endpoint a FULL group total (live + signal), so the first/last delta
    // can never mix a live-only endpoint with a live+signal endpoint and
    // leak the cumulative accumulator as a phantom positive.
    let group_totals = |s: &crate::scenario::sample::Sample<'_>| {
        let mut live: std::collections::HashMap<i32, u128> = std::collections::HashMap::new();
        let mut signal: std::collections::HashMap<i32, Option<u128>> =
            std::collections::HashMap::new();
        for t in s.snapshot.task_enrichments() {
            *live.entry(t.tgid).or_insert(0) += u128::from(task_field(t));
            match signal_field(t) {
                // Shared across the group — any readable thread fixes it.
                Some(v) => {
                    signal.insert(t.tgid, Some(u128::from(v)));
                }
                None => {
                    signal.entry(t.tgid).or_insert(None);
                }
            }
        }
        live.into_iter()
            .filter_map(|(tgid, l)| {
                signal
                    .get(&tgid)
                    .copied()
                    .flatten()
                    .map(|sig| (tgid, l + sig))
            })
            .collect::<std::collections::HashMap<i32, u128>>()
    };

    // Order by capture time so first/last are the earliest/latest
    // boundary — the grouped vec is not guaranteed sorted (the
    // offset-remap in the stimulus path can reorder samples).
    let mut ordered: Vec<&crate::scenario::sample::Sample<'_>> = samples.iter().collect();
    // `elapsed_ms` is now Option: a sample with neither a
    // boundary offset nor a measured elapsed has no time anchor — sort
    // it LAST (u64::MAX), never first, so an untimestamped sample can't
    // become the spurious earliest first_seen endpoint.
    ordered.sort_by_key(|s| s.boundary_offset_ms.or(s.elapsed_ms).unwrap_or(u64::MAX));

    // Per tgid: its full group total at the first and last sample in which
    // it had a readable total, and how many such samples it had.
    let mut first_seen: std::collections::HashMap<i32, u128> = std::collections::HashMap::new();
    let mut last_seen: std::collections::HashMap<i32, u128> = std::collections::HashMap::new();
    let mut readable_count: std::collections::HashMap<i32, u32> = std::collections::HashMap::new();
    for s in ordered {
        for (tgid, total) in group_totals(s) {
            first_seen.entry(tgid).or_insert(total);
            last_seen.insert(tgid, total);
            *readable_count.entry(tgid).or_insert(0) += 1;
        }
    }

    let mut sum: u128 = 0;
    let mut measured = false;
    for (tgid, last) in &last_seen {
        // A delta needs the group observed with a readable total across
        // TWO boundaries; one readable sample gives first == last and no
        // measurable in-phase growth (and a tgid that appears in only one
        // sample, or whose signal_struct never translated, never reaches
        // 2). saturating_sub clamps a last < first read — a numeric tgid
        // reused within the phase (process exit + PID realloc to a fresh
        // group starting near zero) reads lower at last; clamp to 0 rather
        // than wrap.
        if readable_count.get(tgid).copied().unwrap_or(0) < 2 {
            continue;
        }
        measured = true;
        // first_seen always holds tgid (populated in the same pass).
        let first = first_seen.get(tgid).copied().unwrap_or(*last);
        sum += last.saturating_sub(first);
    }
    // None when no group was observed with a readable total across two
    // samples — unmeasurable, so the bucket key stays ABSENT (distinct
    // from a real 0: a qualifying group whose counters did not advance
    // yields Some(0.0)).
    measured.then_some(sum as f64)
}

/// Assemble [`PhaseBucket`]s from a pre-grouped phase map. Shared by
/// [`build_phase_buckets`] (grouping by the bridge-stamped step_index)
/// and [`build_phase_buckets_with_stimulus`] (grouping by the
/// offset-remapped step).
///
/// `monitor_to_window_offset_ms` is subtracted from each
/// [`crate::monitor::MonitorSample`] `elapsed_ms` before the window
/// test, bringing the monitor sample's run-relative timestamp into the
/// bucket-window frame: `0` when both share the run-relative frame, the
/// stimulus/monitor clock skew (see [`monitor_clock_offset`]) when the
/// window is workload-relative (boundary-offset) but the monitor samples
/// remain run-relative.
///
/// Each bucket folds the monitor samples whose (shifted) elapsed_ms
/// lands in the bucket window — supplying metrics like
/// `avg_imbalance_ratio` that need per-CPU full-class `rq.nr_running`,
/// which the bridge-captured Snapshot does not expose (Snapshot carries
/// scx_rq.nr_running only).
fn buckets_from_grouped(
    by_phase: std::collections::BTreeMap<u16, Vec<crate::scenario::sample::Sample<'_>>>,
    monitor_samples: &[crate::monitor::MonitorSample],
    monitor_to_window_offset_ms: i64,
) -> Vec<PhaseBucket> {
    let mut out: Vec<PhaseBucket> = Vec::with_capacity(by_phase.len());
    for (step_index, samples_in_phase) in by_phase {
        let label = if step_index == 0 {
            "BASELINE".to_string()
        } else {
            // Scenario-Step ordinal lives at `step_index - 1`
            // because phase 0 is BASELINE under the 1-indexed
            // encoding; saturate at 0 if the underflow guard
            // ever fires (unreachable for the current encoding
            // — step_index here came from the bucket key so the
            // `> 0` branch is satisfied — but keep the guard so
            // a future caller that hands in a synthetic
            // `step_index = 0` does not panic).
            format!("Step[{}]", step_index.saturating_sub(1))
        };
        let sample_count = samples_in_phase.len();
        // Bucket window: prefer each capture's workload-relative
        // scheduled boundary offset (stable across a deferred-fire
        // burst) over the run-relative fire time; fall back to
        // `elapsed_ms` per-sample when no offset was stamped (synthetic
        // / on-demand). min/max rather than first/last because the
        // offset-remap in the stimulus path can reorder samples
        // relative to drain order.
        let win = |s: &crate::scenario::sample::Sample<'_>| s.boundary_offset_ms.or(s.elapsed_ms);
        let (start_ms, end_ms) = if samples_in_phase.is_empty() {
            (0, u64::MAX)
        } else {
            let mut lo = u64::MAX;
            let mut hi = 0u64;
            for s in &samples_in_phase {
                // Skip a sample with no time anchor (no boundary offset
                // AND no measured elapsed): coercing it to 0
                // would pull start_ms to 0 and over-fold monitor samples.
                // If EVERY sample in the phase is unanchored, lo/hi stay
                // (u64::MAX, 0) — an inverted window that folds nothing,
                // the correct "no placeable samples" outcome.
                if let Some(w) = win(s) {
                    lo = lo.min(w);
                    hi = hi.max(w);
                }
            }
            (lo, hi)
        };
        let mut metrics: std::collections::BTreeMap<String, f64> =
            std::collections::BTreeMap::new();
        for metric_def in crate::stats::METRICS {
            let per_sample_readings: Vec<f64> = samples_in_phase
                .iter()
                .filter_map(|s| metric_def.read_sample(s))
                .collect();
            if per_sample_readings.is_empty() {
                // No per-sample reading for any sample in this
                // bucket -- the metric is host-side-only
                // (cross-cgroup fold) or its dispatch arm has
                // not landed yet. Omit the key rather than
                // collapsing to `Some(0.0)` so the renderer
                // paints "absent" vs "real zero" distinctly.
                continue;
            }
            if let Some(reduced) =
                crate::stats::aggregate_samples_for_phase(metric_def, &per_sample_readings)
            {
                metrics.insert(metric_def.name.to_string(), reduced);
            }
        }
        // Per-phase system / user CPU time (ns), injected post-hoc as a
        // per-thread-GROUP delta over the phase's freeze samples (NOT a
        // read_sample metric — a per-sample cross-task sum then a Counter
        // delta inflates when the captured task set churns; see
        // `phase_group_cpu_delta`). Observer-free: reads the frozen
        // task_struct.{s,u}time + thread-group signal_struct accumulator
        // already captured in each sample's enrichments.
        if let Some(v) = phase_group_cpu_delta(&samples_in_phase, |t| t.stime, |t| t.signal_stime) {
            metrics.entry("system_time_ns".to_string()).or_insert(v);
        }
        if let Some(v) = phase_group_cpu_delta(&samples_in_phase, |t| t.utime, |t| t.signal_utime) {
            metrics.entry("user_time_ns".to_string()).or_insert(v);
        }
        let mut bucket = PhaseBucket {
            step_index,
            label,
            start_ms,
            end_ms,
            sample_count,
            metrics,
            per_cgroup: std::collections::BTreeMap::new(),
        };
        // Per-phase MonitorSample windowing for monitor-derived metrics
        // (avg_imbalance_ratio). Factored into `fold_monitor_into_bucket`
        // so a synthesized zero-capture bucket (see
        // `build_phase_buckets_with_stimulus`) recovers its in-window
        // imbalance too — same formula and frame, but over the
        // synthesized bucket's FULL stimulus window vs this captured
        // bucket's narrower interior capture-offset span.
        fold_monitor_into_bucket(&mut bucket, monitor_samples, monitor_to_window_offset_ms);
        // Derive Rate metrics AFTER every component source is folded in:
        // the METRICS reductions + system/user_time_ns above AND the
        // monitor-injected components (e.g. avg_imbalance_ratio, whose ONLY
        // source is fold_monitor_into_bucket). Deriving before the monitor
        // fold would silently drop a Rate over a monitor-only component. A
        // Rate has no samples of its own (see MetricKind::Rate); this is its
        // per-phase producer. Inert until a Rate registers. The stimulus
        // sibling build_phase_buckets_with_stimulus re-derives again after
        // its own later injections (idempotent).
        crate::stats::derive_rate_metrics(&mut bucket.metrics);
        out.push(bucket);
    }
    out
}

/// Fold the per-CPU full-class imbalance from the monitor samples whose
/// run-relative timestamp falls in `bucket`'s `[start_ms, end_ms)`
/// window into `bucket.metrics["avg_imbalance_ratio"]`.
///
/// The monitor sample's run-relative `elapsed_ms` is shifted into the
/// bucket-window frame (subtract `monitor_to_window_offset_ms`) before
/// the half-open `[start_ms, end_ms)` test so a MonitorSample whose
/// timestamp equals the boundary lands in exactly one bucket (not both
/// adjacent buckets — the closed-on-right form double-counted boundary
/// samples). Single-sample phases (`start_ms == end_ms`) use explicit
/// equality so the window is not empty.
///
/// Filters via [`crate::monitor::sample_looks_valid`] (implausible-DSQ
/// samples) before the fold; `compute_metrics` additionally drops
/// empty-cpus samples (which would default `imbalance_ratio` to 1.0 and
/// pull the mean toward "perfect balance", masking a real regression) —
/// matching the legacy `Timeline::build` path's filter discipline.
///
/// Folds the FULL monitor-derived metric set the legacy `Timeline::build`
/// reducer (`crate::timeline::compute_metrics`) produces —
/// `avg_imbalance_ratio`, `max_imbalance_ratio`, `avg_dsq_depth`,
/// `max_dsq_depth`, `stuck_count`, and the `total_fallback` /
/// `total_keep_last` counter deltas — over the bucket's window.
/// `avg_imbalance_ratio`, `max_imbalance_ratio`, and `stuck_count` are
/// folded for EVERY bucket with in-window monitor samples: none of the three
/// has a `read_sample` dispatch arm (`crate::stats` `read_sample` has arms
/// only for the dsq / fallback keys; these three fall to `_ => None`), so the
/// per-sample capture path never produces them and monitor is their only
/// per-bucket source on captured AND synthesized buckets alike — a captured
/// (common-case) phase must report its per-phase imbalance peak and stall
/// count, not drop them. (`avg_imbalance_ratio` is genuinely ext-metrics-only;
/// `max_imbalance_ratio` and `stuck_count` ALSO carry a typed `GauntletRow`
/// accessor sourced from the whole-run MonitorSummary, so their per-phase
/// fold here feeds per-phase RENDERING only — the run-level value stays the
/// typed accessor, and both `populate_run_ext_metrics*` skip them via
/// `TYPED_FIELD_NAMES` to avoid a double-source.) The dsq / fallback set
/// (`avg_dsq_depth`,
/// `max_dsq_depth`, `total_fallback`, `total_keep_last`) is folded ONLY for
/// a synthesized (`sample_count == 0`) bucket: a captured bucket sources
/// those from its read_sample captures and keeps its pre-synthesize
/// behavior, while a synthesized bucket has no captures, so monitor is its
/// only source — restoring the rendered timeline to PARITY with the old
/// `Timeline::build` fallback (the path a zero-capture-with-monitor run
/// took before the synthesize seam flipped it onto from_phase_buckets;
/// `format_phases` renders these folded metrics for a `sample_count == 0`
/// bucket via its `has_monitor_metrics` gate). Each key is `or_insert` so
/// it never overwrites a value already present. Parity with
/// `Timeline::build` is exact for the production case; for legacy
/// ScenarioEnd-but-no-StepEnd data the synthesized last-step window clamps
/// to the terminal rather than extending to end-of-monitor, and a
/// synthesized bucket's dsq metrics come from the monitor
/// `CpuSnapshot.local_dsq_depth` axis (vs a captured bucket's DSQ-walker
/// axis) — same metric, different sampling axis.
///
/// `bucket`'s `[start_ms, end_ms)` IS the window basis and differs by
/// bucket kind: a captured bucket's is the min/max of its samples'
/// interior capture offsets; a synthesized bucket's is its full
/// `[StepStart, StepEnd)` stimulus window. The monitor sample's
/// run-relative `elapsed_ms` is shifted into that frame (subtract
/// `monitor_to_window_offset_ms`) before the half-open test so a sample
/// on the boundary lands in exactly one bucket. `compute_metrics` returns
/// fallback / keep_last as RATES; this re-derives the bucket-native
/// counter DELTAS (so `phase_from_bucket` re-rates them over the bucket
/// window like the read_sample path) using the same `counter_delta` clamp.
fn fold_monitor_into_bucket(
    bucket: &mut PhaseBucket,
    monitor_samples: &[crate::monitor::MonitorSample],
    monitor_to_window_offset_ms: i64,
) {
    let start_ms = bucket.start_ms;
    let end_ms = bucket.end_ms;
    let in_window = |monitor_ms: u64| -> bool {
        let shifted = monitor_ms as i64 - monitor_to_window_offset_ms;
        if shifted < 0 {
            return false;
        }
        let m = shifted as u64;
        if start_ms == end_ms {
            m == start_ms
        } else {
            m >= start_ms && m < end_ms
        }
    };
    // Filter via sample_looks_valid (matches Timeline::build) so an invalid
    // sample (empty cpus -> imbalance_ratio 1.0 default) doesn't pull the
    // mean toward "perfect balance" and mask a real regression.
    let phase_monitor_samples: Vec<&crate::monitor::MonitorSample> = monitor_samples
        .iter()
        .filter(|s| in_window(s.elapsed_ms))
        .filter(|s| crate::monitor::sample_looks_valid(s))
        .collect();
    if phase_monitor_samples.is_empty() {
        return;
    }
    let pm = crate::timeline::compute_metrics(&phase_monitor_samples);
    // sample_count == 0 IFF the bucket is synthesized: buckets_from_grouped
    // only emits >=1-sample buckets, the synthesize loop only emits 0-sample
    // ones. A CAPTURED bucket folds the source-less monitor signals (avg/max
    // imbalance + stuck_count — none have a read_sample arm) but NOT the
    // dsq / fallback set, which it sources from its read_sample captures.
    // Restricting only that dsq / fallback set to synthesized buckets
    // preserves captured-bucket behavior; a synthesized bucket has no
    // captures, so monitor is its only source and it takes the full set
    // (Timeline::build render parity).
    let synthesized = bucket.sample_count == 0;
    let mut put = |key: &str, v: f64| {
        if v.is_finite() {
            bucket.metrics.entry(key.to_string()).or_insert(v);
        }
    };
    if let Some(v) = pm.avg_imbalance {
        put("avg_imbalance_ratio", v);
    }
    // max_imbalance_ratio (Peak) and stuck_count (Counter) have NO read_sample
    // dispatch arm (both fall to `_ => None` in crate::stats read_sample), so
    // the per-sample capture path never produces them and a CAPTURED bucket
    // would otherwise never carry them — they would surface only on synthesized
    // (zero-capture) buckets. Fold them for EVERY monitor-bearing bucket — like
    // avg_imbalance_ratio above — so a captured (common-case) phase reports its
    // per-phase imbalance peak and stall count instead of dropping them. (Both
    // are monitor-axis signals: imbalance from full-class rq.nr_running, stalls
    // from non-advancing rq.clock across consecutive samples — neither is in
    // the guest Snapshot read_sample observes. Both ALSO carry a typed
    // run-level GauntletRow accessor, so this per-phase fold feeds per-phase
    // RENDERING only; TYPED_FIELD_NAMES keeps them out of the run-level
    // ext_metrics so the typed accessor stays the single run-level source.)
    // `or_insert` still guards against overwriting a value already present.
    if let Some(v) = pm.max_imbalance {
        put("max_imbalance_ratio", v);
    }
    if pm.stall_count > 0 {
        put("stuck_count", pm.stall_count as f64);
    }
    if synthesized {
        if let Some(v) = pm.avg_dsq_depth {
            put("avg_dsq_depth", v);
        }
        put("max_dsq_depth", pm.max_dsq_depth as f64);
        // Bucket-native counter totals for fallback / keep_last: the first
        // and last in-window samples carrying event counters, clamped with
        // the same counter_delta MonitorSummary uses (a mid-phase scheduler
        // restart can reset the counter, producing a negative raw delta).
        // phase_from_bucket re-rates these over the bucket window, matching
        // the read_sample representation captured buckets carry.
        let has_events =
            |s: &&crate::monitor::MonitorSample| s.cpus.iter().any(|c| c.event_counters.is_some());
        let first_ev = phase_monitor_samples.iter().copied().find(has_events);
        let last_ev = phase_monitor_samples.iter().copied().rev().find(has_events);
        if let (Some(first), Some(last)) = (first_ev, last_ev) {
            let fb = crate::monitor::counter_delta(
                last.sum_event_field(|e| e.select_cpu_fallback).unwrap_or(0),
                first
                    .sum_event_field(|e| e.select_cpu_fallback)
                    .unwrap_or(0),
            );
            let kl = crate::monitor::counter_delta(
                last.sum_event_field(|e| e.dispatch_keep_last).unwrap_or(0),
                first.sum_event_field(|e| e.dispatch_keep_last).unwrap_or(0),
            );
            put("total_fallback", fb as f64);
            put("total_keep_last", kl as f64);
        }
    }
}

/// Clock skew (ms) between the host monitor's run-relative timeline and
/// the guest's scenario-relative stimulus timeline, computed the same
/// way as [`crate::timeline::Timeline::build`]: the first significant
/// monitor sample (elapsed > 500 ms, non-empty cpus) and the earliest
/// stimulus event roughly coincide at scenario start. Returns
/// `first_monitor_ms - first_stimulus_ms`; subtract from a monitor
/// sample's elapsed_ms to reach the scenario-relative (boundary-offset)
/// window frame. `0` when either timeline is empty (nothing to align,
/// so the run-relative frames are used as-is).
fn monitor_clock_offset(
    stimulus_events: &[crate::timeline::StimulusEvent],
    monitor_samples: &[crate::monitor::MonitorSample],
) -> i64 {
    if stimulus_events.is_empty() || monitor_samples.is_empty() {
        return 0;
    }
    let first_stimulus_ms = stimulus_events
        .iter()
        .map(|e| e.elapsed_ms)
        .min()
        .unwrap_or(0);
    let first_monitor_ms = monitor_samples
        .iter()
        .find(|s| s.elapsed_ms > 500 && !s.cpus.is_empty())
        .map(|s| s.elapsed_ms)
        .unwrap_or_else(|| monitor_samples.first().map(|s| s.elapsed_ms).unwrap_or(0));
    first_monitor_ms as i64 - first_stimulus_ms as i64
}

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
    /// 6b is the first path to put raw per-cgroup sample vectors on the
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
    /// is intrinsically per-cgroup, out of the Item 7 re-pool), so dropping it
    /// loses only the per-phase off-CPU% render (Item 8), not any run-level value.
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

        // `not_starved`: default-threshold fairness (Starved / Unfair /
        // Stuck) on top of the telemetry already built above.
        if self.not_starved {
            record_default_fairness(&mut r, &cg, reports);
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
        if self.isolation
            && let Some(cs) = cpuset
        {
            r.merge(assert_isolation(reports, cs));
        }
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
        r
    }
}

/// Check slow-tier page ratio against threshold.
///
/// "Slow tier" nodes are NUMA nodes NOT in the cpuset's NUMA node set.
/// For CXL memory-only nodes, these are the nodes without CPUs.
fn assert_slow_tier_ratio(
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
    fn check_not_starved(mut self) -> Self {
        self.not_starved = true;
        self
    }

    fn check_isolation(mut self) -> Self {
        self.isolation = true;
        self
    }

    fn max_gap_ms(mut self, ms: u64) -> Self {
        self.max_gap_ms = Some(ms);
        self
    }

    fn max_spread_pct(mut self, pct: f64) -> Self {
        self.max_spread_pct = Some(pct);
        self
    }
}

/// Unified assertion configuration. Carries both worker checks and
/// monitor thresholds as a single composable type. Each `Option` field
/// acts as an override — `None` means "inherit from parent layer".
///
/// Construct via [`Assert::NO_OVERRIDES`] (preferred const baseline)
/// or [`Assert::default_checks`] (currently aliases NO_OVERRIDES);
/// chain builder methods on either base (all builders are `const fn`
/// except [`Assert::expect_scx_bpf_error_matches`], which compiles a
/// regex at construction). Use the resulting `Assert` value as the
/// `assert` field of a [`Scheduler`](crate::test_support::Scheduler)
/// declared via [`declare_scheduler!`](crate::declare_scheduler) — the
/// macro accepts `assert = Assert::NO_OVERRIDES.foo()`-style chains
/// at the scheduler level. The `#[ktstr_test]` proc macro does NOT
/// accept an `assert = …` attribute on test entries; per-field
/// attribute shortcuts (`max_gap_ms = N`, `not_starved = true`, …)
/// compose into the equivalent struct literal at expansion time.
///
/// Merge order: `Assert::default_checks()` -> `Scheduler.assert` -> per-test `assert`.
/// `default_checks()` is `NO_OVERRIDES` — all assertions are opt-in.
///
/// ```
/// # use ktstr::assert::Assert;
/// // Scheduler opts into imbalance checking.
/// let sched_assert = Assert::NO_OVERRIDES.max_imbalance_ratio(5.0);
///
/// // Merge: defaults <- scheduler <- test.
/// let merged = Assert::default_checks()
///     .merge(&sched_assert)
///     .merge(&Assert::NO_OVERRIDES.max_gap_ms(5000));
///
/// assert_eq!(merged.not_starved, None);              // not opted in
/// assert_eq!(merged.max_imbalance_ratio, Some(5.0)); // from sched
/// assert_eq!(merged.max_gap_ms, Some(5000));         // from test
/// ```
///
/// # Serde roundtrip — covers the 20 threshold/check fields only
///
/// The serde derive at the struct level covers the 20 threshold +
/// flag fields (every `Option<bool/u64/f64/u32/usize>` plus the bare
/// `enforce_monitor_thresholds: bool`). The two reproducer-matcher
/// fields ([`Self::expect_scx_bpf_error_contains`] and
/// [`Self::expect_scx_bpf_error_matches`]) carry `#[serde(skip)]`
/// because their `&'static str` shape cannot round-trip through a
/// borrowed deserializer — see each field's doc for the rationale.
/// Sidecar consumers comparing
/// threshold config across runs treat reproducer matcher strings as
/// part of the test identity (encoded by name in the sidecar key),
/// not part of the threshold payload, so the skip is operationally
/// transparent today.
#[must_use = "builder methods return a new Assert; discard means config is lost"]
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct Assert {
    // Worker checks
    /// Enable starvation, fairness spread, and gap checks across
    /// worker reports. `Some(true)` enables, `Some(false)` explicitly
    /// disables (overriding any enabling merge from a lower layer),
    /// `None` inherits from the merge parent.
    pub not_starved: Option<bool>,
    /// Enable per-worker CPU isolation checks (ensure workers remain
    /// within their assigned cpuset). Same tri-state semantics as
    /// `not_starved`.
    pub isolation: Option<bool>,
    /// Max per-worker scheduling gap in milliseconds. Fails the
    /// assertion if any worker's longest off-CPU stretch exceeds this.
    pub max_gap_ms: Option<u64>,
    /// Max per-cgroup fairness spread as a percentage. Fails if the
    /// range between the most- and least-served workers exceeds this
    /// fraction of their mean.
    pub max_spread_pct: Option<f64>,

    // Throughput checks
    /// Max coefficient of variation for work_units/cpu_time across workers.
    /// Catches placement unfairness where some workers get less CPU than others.
    pub max_throughput_cv: Option<f64>,
    /// Minimum work_units per CPU-second. Catches cases where all workers
    /// are equally slow (CV passes but absolute throughput is too low).
    pub min_work_rate: Option<f64>,

    // Benchmarking checks
    /// Max p99 wake latency in NANOSECONDS. Fails if the pooled
    /// p99 across every worker's `wake_latencies_ns` exceeds this.
    ///
    /// # Unit-name gotcha
    ///
    /// The threshold is `_ns`, but the paired reporting field on
    /// [`CgroupStats::p99_wake_latency_us`] and the re-pooled run-level
    /// `worst_p99_wake_latency_us` ext metric (a `MetricKind::Distribution`)
    /// are MICROSECONDS. The two surfaces are intentionally split:
    ///   - the threshold uses NS for precision (typical scheduler
    ///     wake latencies are single-digit µs, so sub-µs resolution
    ///     matters for regression gates);
    ///   - the reporting fields use US for readability in
    ///     `stats compare` / dashboard output.
    ///
    /// Both are computed from the same underlying
    /// [`WorkerReport::wake_latencies_ns`] samples — see
    /// [`assert_benchmarks`] for the threshold path and
    /// [`assert_not_starved`] for the reporting path. A bare
    /// comparison of `max_p99_wake_latency_ns` against
    /// `CgroupStats::p99_wake_latency_us` is a unit-mismatch bug;
    /// `assert_benchmarks` never does this — it consumes the raw
    /// `wake_latencies_ns` directly — and
    /// `assert_p99_ns_threshold_compares_against_ns_latencies` pins
    /// that contract.
    pub max_p99_wake_latency_ns: Option<u64>,
    /// Max wake latency coefficient of variation. Fails if CV exceeds this.
    pub max_wake_latency_cv: Option<f64>,
    /// Minimum iterations per wall-clock second. Fails if any worker is below.
    pub min_iteration_rate: Option<f64>,
    /// Max migration ratio (migrations/iterations). Fails if any cgroup exceeds this.
    pub max_migration_ratio: Option<f64>,

    // Monitor checks
    /// Max `nr_running` / LLC imbalance ratio observed by the monitor.
    /// Fails if the worst sample's imbalance exceeds this.
    pub max_imbalance_ratio: Option<f64>,
    /// Max local DSQ depth observed by the monitor. Fails if any
    /// sampled CPU's local DSQ grew beyond this.
    pub max_local_dsq_depth: Option<u32>,
    /// Treat a stall verdict from the monitor as a hard failure. Same
    /// tri-state semantics as `not_starved`.
    pub fail_on_stall: Option<bool>,
    /// Minimum number of consecutive samples that must exceed the
    /// monitor threshold before a verdict is raised. Smooths out
    /// single-sample spikes.
    pub sustained_samples: Option<usize>,
    /// Max `select_cpu_fallback` rate (events/sec). Fails if the
    /// scx event counter delta over the run exceeds this rate.
    pub max_fallback_rate: Option<f64>,
    /// Max `keep_last` rate (events/sec). Fails if the scx event
    /// counter delta over the run exceeds this rate.
    pub max_keep_last_rate: Option<f64>,
    /// Promote monitor threshold violations from report-only to
    /// pass/fail. When `false` (the default), the monitor still walks
    /// every sample and records every violation in the verdict's
    /// `details`, but the verdict's `passed` stays `true`. Tests that
    /// want monitor violations to fail the run call
    /// [`Self::with_monitor_defaults`], which populates each monitor
    /// threshold from `MonitorThresholds::new()`
    /// and sets this flag to `true`.
    pub enforce_monitor_thresholds: bool,

    // NUMA checks
    /// Minimum fraction of pages on the expected NUMA node(s) (0.0-1.0).
    /// Expected nodes are derived from the worker's
    /// [`MemPolicy`](crate::workload::MemPolicy) at evaluation time.
    /// Fails if the observed locality fraction falls below this.
    pub min_page_locality: Option<f64>,
    /// Maximum ratio of NUMA-node-migrated pages to total allocated
    /// pages (0.0-1.0). Distinct from [`max_migration_ratio`](Self::max_migration_ratio)
    /// which measures CPU migrations per iteration. Fails if the
    /// observed migration ratio exceeds this.
    pub max_cross_node_migration_ratio: Option<f64>,
    /// Maximum fraction of pages on slow-tier (memory-only) NUMA nodes
    /// (0.0-1.0). For CXL memory tiering tests: fails if more than
    /// this fraction of pages land on memory-only nodes. Requires
    /// `slow_tier_nodes` to be set at evaluation time.
    pub max_slow_tier_ratio: Option<f64>,

    /// Reproducer-mode literal-substring matcher for the captured
    /// `scx_bpf_error` text. When `Some(literal)`, the eval pipeline
    /// scans the combined scheduler log + sched_ext dump for a
    /// substring match against `literal` and fails the test if the
    /// substring is absent.
    ///
    /// Use this for the common case of pinning an exact error
    /// fragment like `apply_cell_config returned -EINVAL` without
    /// having to escape regex metacharacters. For pattern matching
    /// with anchors / character classes / wildcards, use
    /// [`Self::expect_scx_bpf_error_matches`] instead — the two
    /// fields are orthogonal and can both be set (both must match).
    ///
    /// Requires the entry's `expect_err = true` — a reproducer
    /// matcher only fires on expected-error tests; setting this on
    /// a passing test would assert "the test passed AND contained
    /// this error text," which is contradictory. The eval-time
    /// validation rejects that combination with a clear diagnostic.
    ///
    /// Stored as `&'static str` so the const-fn `Self::merge`
    /// composes without cloning. Empty strings are rejected at
    /// evaluation (an empty literal would silently match every
    /// message and turn this assertion into a no-op).
    /// `#[serde(skip)]` because the field's `&'static str` shape
    /// cannot round-trip through a borrowed deserializer (no source-
    /// string lifetime to bind to). Reproducer matcher strings are
    /// test-author static literals carried in the test definition
    /// itself, not per-run data the sidecar needs to roundtrip — so
    /// skipping them on the wire keeps the JSON shape clean without
    /// losing any sidecar-consumer functionality. Skipped fields
    /// default to `None` on deserialize per `Option::default()`.
    ///
    /// The `Option<Cow<'static, str>>` alternative that WOULD
    /// roundtrip is rejected because it cascade-breaks
    /// `Scheduler::assert`'s const-fn (`Cow` has a heap destructor,
    /// which breaks the const-fn assignment path used by
    /// `declare_scheduler!` macro statics). A future decomposition
    /// into a `ReproducerMatchers` sub-config could revisit this if
    /// sidecar-loaded test definitions ever need the strings to
    /// survive end-to-end.
    #[serde(skip)]
    pub expect_scx_bpf_error_contains: Option<&'static str>,

    /// Reproducer-mode regex matcher for the captured `scx_bpf_error`
    /// text. When `Some(pattern)`, the eval pipeline compiles the
    /// pattern via the `regex` crate, scans the combined scheduler
    /// log + sched_ext dump, and fails the test if the regex does
    /// not match anywhere in the corpus.
    ///
    /// The pattern is a full regex — special characters
    /// (`. * + ? ( ) [ ] { } | ^ $ \`) carry their regex meaning.
    /// For literal-substring matching, prefer
    /// [`Self::expect_scx_bpf_error_contains`] to avoid escape
    /// footguns. The captured corpus is the multi-line concatenation
    /// of the scheduler log and `--- sched_ext dump ---`; the regex
    /// crate's default flags apply: `^` and `$` anchor to the start /
    /// end of the WHOLE corpus (not individual lines), and `.` does
    /// NOT cross `\n`. Opt in to line-level anchoring with `(?m)`
    /// (e.g. `(?m)^apply_cell_config$`) and to newline-spanning
    /// `.` with `(?s)`. A bare `apply_cell_config` matches the
    /// token anywhere in the corpus.
    ///
    /// Requires the entry's `expect_err = true` — same rationale
    /// as [`Self::expect_scx_bpf_error_contains`]. Patterns are
    /// validated at construction: empty literals, invalid regex
    /// syntax, and any pattern satisfying `is_match("")` all
    /// panic via the [`Self::expect_scx_bpf_error_matches`]
    /// builder. The `is_match("")` predicate catches two
    /// no-op classes with one check: patterns that match every
    /// position (e.g. `a?`, `.*`, `(?:)`) trivially pass against
    /// any corpus, and patterns that match only the empty string
    /// (e.g. `^$`) trivially fail against any non-empty corpus —
    /// real captured scheduler-output corpora are non-empty, so
    /// both classes are equally no-op pins. Bare `\b` (word
    /// boundary) slips the gate because the empty string
    /// contains no word characters; see the builder docstring
    /// for the operator-direction.
    /// `#[serde(skip)]` for the same reason as
    /// [`Self::expect_scx_bpf_error_contains`] above: `&'static str`
    /// doesn't roundtrip + the matcher pattern is test-definition
    /// data, not sidecar-roundtrip data. Skipped fields default to
    /// `None` on deserialize.
    #[serde(skip)]
    pub expect_scx_bpf_error_matches: Option<&'static str>,
}

impl Assert {
    /// Human-readable multi-line dump of every threshold field. Each
    /// field renders as `  name: value` (`none` when the option is
    /// `None`, i.e. inherited or unset). Used by
    /// `cargo ktstr show-thresholds <test>` to expose the resolved
    /// merged `Assert` (`default_checks().merge(&entry.scheduler.assert).
    /// merge(&entry.assert)`) without forcing the operator to read
    /// the Debug impl or source. Output is a sequence of indented
    /// `row` lines ending with a newline; the caller owns any
    /// outer section header (the `show-thresholds` CLI already
    /// prints `Test: ...` / `Scheduler: ...` lines above the
    /// threshold block, which together establish context — an
    /// additional `Resolved assertion thresholds:` banner here
    /// would be a redundant third header).
    pub fn format_human(&self) -> String {
        use std::fmt::Write;
        let mut out = String::new();
        fn row<T: std::fmt::Display>(out: &mut String, name: &str, v: &Option<T>) {
            match v {
                Some(x) => writeln!(out, "  {name:<38}: {x}").unwrap(),
                None => writeln!(out, "  {name:<38}: none").unwrap(),
            }
        }
        row(&mut out, "not_starved", &self.not_starved);
        row(&mut out, "isolation", &self.isolation);
        row(&mut out, "max_gap_ms", &self.max_gap_ms);
        row(&mut out, "max_spread_pct", &self.max_spread_pct);
        row(&mut out, "max_throughput_cv", &self.max_throughput_cv);
        row(&mut out, "min_work_rate", &self.min_work_rate);
        row(
            &mut out,
            "max_p99_wake_latency_ns",
            &self.max_p99_wake_latency_ns,
        );
        row(&mut out, "max_wake_latency_cv", &self.max_wake_latency_cv);
        row(&mut out, "min_iteration_rate", &self.min_iteration_rate);
        row(&mut out, "max_migration_ratio", &self.max_migration_ratio);
        row(&mut out, "max_imbalance_ratio", &self.max_imbalance_ratio);
        row(&mut out, "max_local_dsq_depth", &self.max_local_dsq_depth);
        row(&mut out, "fail_on_stall", &self.fail_on_stall);
        row(&mut out, "sustained_samples", &self.sustained_samples);
        row(&mut out, "max_fallback_rate", &self.max_fallback_rate);
        row(&mut out, "max_keep_last_rate", &self.max_keep_last_rate);
        row(&mut out, "min_page_locality", &self.min_page_locality);
        row(
            &mut out,
            "max_cross_node_migration_ratio",
            &self.max_cross_node_migration_ratio,
        );
        row(&mut out, "max_slow_tier_ratio", &self.max_slow_tier_ratio);
        row(
            &mut out,
            "expect_scx_bpf_error_contains",
            &self.expect_scx_bpf_error_contains,
        );
        row(
            &mut out,
            "expect_scx_bpf_error_matches",
            &self.expect_scx_bpf_error_matches,
        );
        out
    }

    /// Identity element for [`Assert::merge`]: every field is `None`,
    /// so neither side of a merge with `NO_OVERRIDES` is altered.
    /// Identical to the value returned by [`Self::default_checks`] —
    /// the const is for spread-into-struct-literal composition, the
    /// const fn is the method-style entry point.
    pub const NO_OVERRIDES: Assert = Assert {
        not_starved: None,
        isolation: None,
        max_gap_ms: None,
        max_spread_pct: None,
        max_throughput_cv: None,
        min_work_rate: None,
        max_p99_wake_latency_ns: None,
        max_wake_latency_cv: None,
        min_iteration_rate: None,
        max_migration_ratio: None,
        max_imbalance_ratio: None,
        max_local_dsq_depth: None,
        fail_on_stall: None,
        sustained_samples: None,
        max_fallback_rate: None,
        max_keep_last_rate: None,
        enforce_monitor_thresholds: false,
        min_page_locality: None,
        max_cross_node_migration_ratio: None,
        max_slow_tier_ratio: None,
        expect_scx_bpf_error_contains: None,
        expect_scx_bpf_error_matches: None,
    };

    /// Baseline of the runtime merge chain
    /// `default_checks().merge(&scheduler.assert).merge(&entry.assert)`.
    ///
    /// All checks are off by default — tests opt in to the assertions
    /// they care about via scheduler-level or per-test `Assert`
    /// overrides.
    ///
    /// For spread-into-struct-literal composition
    /// (`Assert { not_starved: Some(true), ..Assert::NO_OVERRIDES }`)
    /// use the equivalent const [`Self::NO_OVERRIDES`]; this const fn
    /// is the method-style entry point that pairs with `.verdict()`
    /// and the builder setters.
    pub const fn default_checks() -> Assert {
        Self::NO_OVERRIDES
    }

    /// Build a fresh [`Verdict`] under this `Assert`'s threshold
    /// config. The returned accumulator carries no claim records; call
    /// the typed `claim_<field>` methods generated by
    /// [`#[derive(Claim)]`](ktstr_macros::Claim) on stats structs as
    /// `stats.claim_<field>(&mut verdict)`, or use the
    /// [`claim!`](crate::claim) macro on a local/expression, then
    /// call [`Verdict::into_result`] to produce the final
    /// [`AssertResult`].
    ///
    /// This is the entry point of the pointwise-claim API. The
    /// `Assert` itself remains pure threshold config and stays
    /// `Copy`; per-test claims accumulate on the returned `Verdict`,
    /// which owns its own buffers (details, stats).
    ///
    /// ```
    /// # use ktstr::assert::Assert;
    /// let r = Assert::default_checks().verdict().into_result();
    /// assert!(r.is_pass(), "no claims means passing verdict");
    /// ```
    pub fn verdict(self) -> Verdict {
        Verdict::with_assert(self)
    }

    pub const fn check_not_starved(mut self) -> Self {
        self.not_starved = Some(true);
        self
    }

    pub const fn check_isolation(mut self) -> Self {
        self.isolation = Some(true);
        self
    }

    pub const fn max_gap_ms(mut self, ms: u64) -> Self {
        self.max_gap_ms = Some(ms);
        self
    }

    pub const fn max_spread_pct(mut self, pct: f64) -> Self {
        self.max_spread_pct = Some(pct);
        self
    }

    pub const fn max_throughput_cv(mut self, v: f64) -> Self {
        self.max_throughput_cv = Some(v);
        self
    }

    pub const fn min_work_rate(mut self, v: f64) -> Self {
        self.min_work_rate = Some(v);
        self
    }

    pub const fn max_p99_wake_latency_ns(mut self, v: u64) -> Self {
        self.max_p99_wake_latency_ns = Some(v);
        self
    }

    pub const fn max_wake_latency_cv(mut self, v: f64) -> Self {
        self.max_wake_latency_cv = Some(v);
        self
    }

    pub const fn min_iteration_rate(mut self, v: f64) -> Self {
        self.min_iteration_rate = Some(v);
        self
    }

    pub const fn max_migration_ratio(mut self, v: f64) -> Self {
        self.max_migration_ratio = Some(v);
        self
    }

    pub const fn max_imbalance_ratio(mut self, v: f64) -> Self {
        self.max_imbalance_ratio = Some(v);
        self
    }

    pub const fn max_local_dsq_depth(mut self, v: u32) -> Self {
        self.max_local_dsq_depth = Some(v);
        self
    }

    /// Control whether a monitor stall verdict fails the assertion.
    pub const fn fail_on_stall(mut self, v: bool) -> Self {
        self.fail_on_stall = Some(v);
        self
    }

    /// Set the number of consecutive over-threshold samples required
    /// before the monitor raises a verdict.
    pub const fn sustained_samples(mut self, v: usize) -> Self {
        self.sustained_samples = Some(v);
        self
    }

    pub const fn max_fallback_rate(mut self, v: f64) -> Self {
        self.max_fallback_rate = Some(v);
        self
    }

    pub const fn max_keep_last_rate(mut self, v: f64) -> Self {
        self.max_keep_last_rate = Some(v);
        self
    }

    pub const fn min_page_locality(mut self, v: f64) -> Self {
        self.min_page_locality = Some(v);
        self
    }

    pub const fn max_cross_node_migration_ratio(mut self, v: f64) -> Self {
        self.max_cross_node_migration_ratio = Some(v);
        self
    }

    pub const fn max_slow_tier_ratio(mut self, v: f64) -> Self {
        self.max_slow_tier_ratio = Some(v);
        self
    }

    /// True when any worker-level check field is `Some`. A pure query on
    /// the configured checks — it no longer gates per-cgroup telemetry
    /// (telemetry is built unconditionally per handle in
    /// [`crate::scenario`]'s collect path; worker-check assertions are the
    /// only opt-in part). Retained as public API for callers composing an
    /// `Assert` who need to know whether any worker check is set.
    pub const fn has_worker_checks(&self) -> bool {
        self.not_starved.is_some()
            || self.isolation.is_some()
            || self.max_gap_ms.is_some()
            || self.max_spread_pct.is_some()
            || self.max_throughput_cv.is_some()
            || self.min_work_rate.is_some()
            || self.max_p99_wake_latency_ns.is_some()
            || self.max_wake_latency_cv.is_some()
            || self.min_iteration_rate.is_some()
            || self.max_migration_ratio.is_some()
            || self.min_page_locality.is_some()
            || self.max_cross_node_migration_ratio.is_some()
            || self.max_slow_tier_ratio.is_some()
    }

    /// Merge `other` on top of `self`. Each `Some` field in `other`
    /// overrides the corresponding field in `self`; `None` fields
    /// inherit from `self`.
    ///
    /// [`Assert::NO_OVERRIDES`] is the two-sided identity:
    /// `x.merge(&NO_OVERRIDES)` and `NO_OVERRIDES.merge(&x)` both yield
    /// `x`. The runtime composes scheduler- and test-level overrides as
    /// `Assert::default_checks().merge(&scheduler.assert).merge(&test.assert)`,
    /// so a `NO_OVERRIDES` at either override layer leaves the defaults
    /// untouched -- which means "no override," not "no checks."
    pub const fn merge(&self, other: &Assert) -> Assert {
        // `Option::or` is not yet const-stable, so each field expands
        // a match rather than calling `other.x.or(self.x)`. Keep it
        // this way until `const fn` can call `Option::or`; at that
        // point the 19 match blocks collapse to 19 `.or()` calls.
        Assert {
            not_starved: match other.not_starved {
                Some(v) => Some(v),
                None => self.not_starved,
            },
            isolation: match other.isolation {
                Some(v) => Some(v),
                None => self.isolation,
            },
            max_gap_ms: match other.max_gap_ms {
                Some(v) => Some(v),
                None => self.max_gap_ms,
            },
            max_spread_pct: match other.max_spread_pct {
                Some(v) => Some(v),
                None => self.max_spread_pct,
            },
            max_throughput_cv: match other.max_throughput_cv {
                Some(v) => Some(v),
                None => self.max_throughput_cv,
            },
            min_work_rate: match other.min_work_rate {
                Some(v) => Some(v),
                None => self.min_work_rate,
            },
            max_p99_wake_latency_ns: match other.max_p99_wake_latency_ns {
                Some(v) => Some(v),
                None => self.max_p99_wake_latency_ns,
            },
            max_wake_latency_cv: match other.max_wake_latency_cv {
                Some(v) => Some(v),
                None => self.max_wake_latency_cv,
            },
            min_iteration_rate: match other.min_iteration_rate {
                Some(v) => Some(v),
                None => self.min_iteration_rate,
            },
            max_migration_ratio: match other.max_migration_ratio {
                Some(v) => Some(v),
                None => self.max_migration_ratio,
            },
            max_imbalance_ratio: match other.max_imbalance_ratio {
                Some(v) => Some(v),
                None => self.max_imbalance_ratio,
            },
            max_local_dsq_depth: match other.max_local_dsq_depth {
                Some(v) => Some(v),
                None => self.max_local_dsq_depth,
            },
            fail_on_stall: match other.fail_on_stall {
                Some(v) => Some(v),
                None => self.fail_on_stall,
            },
            sustained_samples: match other.sustained_samples {
                Some(v) => Some(v),
                None => self.sustained_samples,
            },
            max_fallback_rate: match other.max_fallback_rate {
                Some(v) => Some(v),
                None => self.max_fallback_rate,
            },
            max_keep_last_rate: match other.max_keep_last_rate {
                Some(v) => Some(v),
                None => self.max_keep_last_rate,
            },
            enforce_monitor_thresholds: self.enforce_monitor_thresholds
                || other.enforce_monitor_thresholds,
            min_page_locality: match other.min_page_locality {
                Some(v) => Some(v),
                None => self.min_page_locality,
            },
            max_cross_node_migration_ratio: match other.max_cross_node_migration_ratio {
                Some(v) => Some(v),
                None => self.max_cross_node_migration_ratio,
            },
            max_slow_tier_ratio: match other.max_slow_tier_ratio {
                Some(v) => Some(v),
                None => self.max_slow_tier_ratio,
            },
            expect_scx_bpf_error_contains: match other.expect_scx_bpf_error_contains {
                Some(v) => Some(v),
                None => self.expect_scx_bpf_error_contains,
            },
            expect_scx_bpf_error_matches: match other.expect_scx_bpf_error_matches {
                Some(v) => Some(v),
                None => self.expect_scx_bpf_error_matches,
            },
        }
    }

    /// Extract an `AssertPlan` for worker-side checks.
    pub(crate) fn worker_plan(&self) -> AssertPlan {
        AssertPlan {
            not_starved: self.not_starved.unwrap_or(false),
            isolation: self.isolation.unwrap_or(false),
            max_gap_ms: self.max_gap_ms,
            max_spread_pct: self.max_spread_pct,
            max_throughput_cv: self.max_throughput_cv,
            min_work_rate: self.min_work_rate,
            max_p99_wake_latency_ns: self.max_p99_wake_latency_ns,
            max_wake_latency_cv: self.max_wake_latency_cv,
            min_iteration_rate: self.min_iteration_rate,
            max_migration_ratio: self.max_migration_ratio,
            min_page_locality: self.min_page_locality,
            max_cross_node_migration_ratio: self.max_cross_node_migration_ratio,
            max_slow_tier_ratio: self.max_slow_tier_ratio,
        }
    }

    /// Run the configured worker checks against one cgroup's reports.
    ///
    /// `cpuset` is the CPU set for isolation checks. `numa_nodes` is
    /// the NUMA node IDs covered by the cpuset (for page locality and
    /// slow-tier checks). Derive via
    /// [`TestTopology::numa_nodes_for_cpuset`](crate::topology::TestTopology::numa_nodes_for_cpuset).
    pub fn assert_cgroup(
        &self,
        reports: &[crate::workload::WorkerReport],
        cpuset: Option<&BTreeSet<usize>>,
    ) -> AssertResult {
        self.worker_plan().assert_cgroup(reports, cpuset, None)
    }

    /// Run worker checks with explicit NUMA node set for page locality.
    pub fn assert_cgroup_with_numa(
        &self,
        reports: &[crate::workload::WorkerReport],
        cpuset: Option<&BTreeSet<usize>>,
        numa_nodes: Option<&BTreeSet<usize>>,
    ) -> AssertResult {
        self.worker_plan()
            .assert_cgroup(reports, cpuset, numa_nodes)
    }

    /// Run NUMA page locality check.
    ///
    /// `observed` is the fraction of pages on expected nodes (0.0-1.0).
    /// `total_pages` and `local_pages` are for diagnostics.
    pub fn assert_page_locality(
        &self,
        observed: f64,
        total_pages: u64,
        local_pages: u64,
    ) -> AssertResult {
        assert_page_locality(observed, self.min_page_locality, total_pages, local_pages)
    }

    /// Run cross-node migration ratio check.
    ///
    /// `migrated_pages` is the `/proc/vmstat` `numa_pages_migrated` delta.
    /// `total_pages` is total allocated pages from numa_maps.
    pub fn assert_cross_node_migration(
        &self,
        migrated_pages: u64,
        total_pages: u64,
    ) -> AssertResult {
        assert_cross_node_migration(
            migrated_pages,
            total_pages,
            self.max_cross_node_migration_ratio,
        )
    }

    /// Extract `MonitorThresholds` for monitor-side evaluation.
    pub(crate) fn has_monitor_thresholds(&self) -> bool {
        self.max_imbalance_ratio.is_some()
            || self.max_local_dsq_depth.is_some()
            || self.fail_on_stall.is_some()
            || self.sustained_samples.is_some()
            || self.max_fallback_rate.is_some()
            || self.max_keep_last_rate.is_some()
    }

    pub(crate) fn monitor_thresholds(&self) -> crate::monitor::MonitorThresholds {
        use crate::monitor::MonitorThresholds;
        let d = MonitorThresholds::new();
        MonitorThresholds {
            max_imbalance_ratio: self.max_imbalance_ratio.unwrap_or(d.max_imbalance_ratio),
            max_local_dsq_depth: self.max_local_dsq_depth.unwrap_or(d.max_local_dsq_depth),
            fail_on_stall: self.fail_on_stall.unwrap_or(d.fail_on_stall),
            sustained_samples: self.sustained_samples.unwrap_or(d.sustained_samples),
            max_fallback_rate: self.max_fallback_rate.unwrap_or(d.max_fallback_rate),
            max_keep_last_rate: self.max_keep_last_rate.unwrap_or(d.max_keep_last_rate),
            enforce: self.enforce_monitor_thresholds,
        }
    }

    /// Opt into pass/fail enforcement for monitor thresholds. Without
    /// this call, monitor violations are reported in the verdict's
    /// `details` but do not fail the test. With it, any monitor
    /// threshold violation fails the test.
    ///
    /// Also populates any unset monitor-threshold field with the
    /// canonical default from `MonitorThresholds::new()`
    /// — so a test that only cares about `max_keep_last_rate` can chain
    /// `.max_keep_last_rate(N).with_monitor_defaults()` and get the
    /// other four enforced at their canonical defaults.
    pub const fn with_monitor_defaults(mut self) -> Self {
        use crate::monitor::MonitorThresholds;
        let d = MonitorThresholds::new();
        if self.max_imbalance_ratio.is_none() {
            self.max_imbalance_ratio = Some(d.max_imbalance_ratio);
        }
        if self.max_local_dsq_depth.is_none() {
            self.max_local_dsq_depth = Some(d.max_local_dsq_depth);
        }
        if self.fail_on_stall.is_none() {
            self.fail_on_stall = Some(d.fail_on_stall);
        }
        if self.sustained_samples.is_none() {
            self.sustained_samples = Some(d.sustained_samples);
        }
        if self.max_fallback_rate.is_none() {
            self.max_fallback_rate = Some(d.max_fallback_rate);
        }
        if self.max_keep_last_rate.is_none() {
            self.max_keep_last_rate = Some(d.max_keep_last_rate);
        }
        self.enforce_monitor_thresholds = true;
        self
    }

    /// Const-fn builder for [`Self::expect_scx_bpf_error_contains`].
    /// Chains with the other const-fn setters so a scheduler-def or
    /// per-test assertion block can compose
    /// `Assert::NO_OVERRIDES.expect_scx_bpf_error_contains(...).check_not_starved()`.
    ///
    /// Empty strings panic at construction (an empty literal would
    /// silently match every message and turn this assertion into a
    /// no-op); pass a non-empty fragment that should appear in the
    /// expected `scx_bpf_error` message.
    ///
    /// # Panics
    /// When `literal` is empty.
    #[must_use = "builder methods consume self; bind the result"]
    pub const fn expect_scx_bpf_error_contains(mut self, literal: &'static str) -> Self {
        assert!(
            !literal.is_empty(),
            "Assert::expect_scx_bpf_error_contains: literal must be non-empty",
        );
        self.expect_scx_bpf_error_contains = Some(literal);
        self
    }

    /// Builder for [`Self::expect_scx_bpf_error_matches`]. The
    /// pattern is a regex; special characters retain their regex
    /// meaning. For literal-substring matching, prefer
    /// [`Self::expect_scx_bpf_error_contains`] to avoid escape
    /// footguns.
    ///
    /// Validates the pattern at construction: rejects empty
    /// patterns, rejects invalid regex syntax, and rejects any
    /// pattern that satisfies `is_match("")`. The empty-string
    /// match predicate catches two related no-op classes:
    /// patterns that match every position (e.g. `a?`, `.*`,
    /// `(?:)`) trivially pass against any corpus, and patterns
    /// that match only the empty string (e.g. `^$`) trivially
    /// fail against any non-empty corpus — every real captured
    /// scheduler-output corpus is non-empty, so the latter is
    /// equally a no-op pin in practice. Both are useless;
    /// `is_match("")` catches both with one check.
    ///
    /// Bare `\b` (word boundary) slips this gate because the
    /// empty string contains no word characters, so `\b` finds
    /// no transition and `is_match("")` returns false; yet `\b`
    /// matches the first word boundary in any real corpus,
    /// turning a bare-`\b` pin into a vacuous "any non-empty
    /// log passes" assertion. Use a substring of the expected
    /// error text rather than a standalone boundary assertion.
    /// All other documented assertions (`\A`, `\z`, `^`, `$`,
    /// `\B`) match the empty string at position 0 and ARE
    /// caught by the gate.
    ///
    /// Unlike the sibling [`Self::expect_scx_bpf_error_contains`]
    /// (which is `const fn`), this builder is non-const because
    /// the construction-time regex compilation requires heap
    /// allocation. Callers needing a const builder for a regex
    /// matcher must build the `Assert` via struct literal —
    /// the evaluator's defense-in-depth catches invalid syntax
    /// reached via that bypass at first evaluation, but the
    /// vacuous-pattern gate only fires on the builder path.
    ///
    /// # Panics
    /// When `pattern` is empty, is invalid regex syntax, or
    /// matches the empty string.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn expect_scx_bpf_error_matches(mut self, pattern: &'static str) -> Self {
        assert!(
            !pattern.is_empty(),
            "Assert::expect_scx_bpf_error_matches: pattern must be non-empty",
        );
        let compiled = regex::Regex::new(pattern).unwrap_or_else(|e| {
            panic!(
                "Assert::expect_scx_bpf_error_matches: pattern {pattern:?} is not valid regex: {e}",
            )
        });
        assert!(
            !compiled.is_match(""),
            "Assert::expect_scx_bpf_error_matches: pattern {pattern:?} matches the empty \
             string (e.g. `a?`, `.*`, `(?:)`, `^$`); such patterns vacuously match any \
             corpus and turn the matcher into a no-op — use a meaningful pattern that \
             requires at least one character",
        );
        self.expect_scx_bpf_error_matches = Some(pattern);
        self
    }

    /// Evaluate the reproducer-mode `scx_bpf_error` matchers against
    /// the captured text corpus. Returns an empty Vec when no matcher
    /// is configured or when every configured matcher matches; returns
    /// a non-empty Vec of `AssertDetail` entries on failure.
    ///
    /// Each configured matcher contributes at most one detail. Both
    /// fields can be set simultaneously (`AND` semantics — both must
    /// match).
    ///
    /// Preconditions enforced by the evaluator:
    /// 1. `expect_err = true` must be set when either matcher is
    ///    configured. Setting the matcher on a passing-test contract
    ///    is a misuse — surfaced with an `expect_err = true` reminder
    ///    diagnostic.
    /// 2. The regex pattern in
    ///    [`Self::expect_scx_bpf_error_matches`] must compile via
    ///    the `regex` crate. Invalid syntax surfaces as a diagnostic
    ///    naming the pattern and the compile error.
    ///
    /// `captured_text` is the concatenation of the raw scheduler-log
    /// stream (the bulk-port merged `SchedLog` frames, or the test
    /// process `output` fallback when no frames arrived) and the
    /// `--- sched_ext dump ---` extract — both surfaces where
    /// `scx_bpf_error` printk lands. The matcher sees the WHOLE
    /// stream, not the marker-extracted section; lines outside the
    /// `SCHED_OUTPUT_START..SCHED_OUTPUT_END` markers are included.
    pub fn evaluate_scx_bpf_error_match(
        &self,
        captured_text: &str,
        expect_err: bool,
    ) -> Vec<AssertDetail> {
        let mut details = Vec::new();
        if self.expect_scx_bpf_error_contains.is_none()
            && self.expect_scx_bpf_error_matches.is_none()
        {
            return details;
        }
        if !expect_err {
            details.push(AssertDetail::new(
                DetailKind::Other,
                "expect_scx_bpf_error_contains or expect_scx_bpf_error_matches \
                 requires expect_err = true on the test entry — the matcher narrows \
                 which failure counts as the expected bug, and only applies to \
                 expected-error tests; set #[ktstr_test(expect_err = true, ...)] or \
                 drop the matcher",
            ));
            return details;
        }
        // Truncate at a UTF-8 char boundary at or below 400 bytes so
        // the excerpt length stays within the "up to 400 bytes follow:"
        // budget. `chars().take(400)` would count codepoints instead,
        // and a multi-byte corpus would exceed the byte budget — the
        // boundary walk steps back to the nearest char start.
        let excerpt = || -> &str {
            let len = captured_text.len();
            if len <= 400 {
                captured_text
            } else {
                let mut end = 400;
                while end > 0 && !captured_text.is_char_boundary(end) {
                    end -= 1;
                }
                &captured_text[..end]
            }
        };
        if let Some(literal) = self.expect_scx_bpf_error_contains
            && !captured_text.contains(literal)
        {
            details.push(AssertDetail::new(
                DetailKind::Other,
                format!(
                    "expect_scx_bpf_error_contains({literal:?}): substring not found \
                     in the scheduler log + sched_ext dump corpus (the expected bug \
                     did not fire, or its message text changed). Captured corpus \
                     {} bytes; up to 400 bytes follow:\n{}",
                    captured_text.len(),
                    excerpt(),
                ),
            ));
        }
        if let Some(pattern) = self.expect_scx_bpf_error_matches {
            match regex::Regex::new(pattern) {
                Ok(re) => {
                    if !re.is_match(captured_text) {
                        details.push(AssertDetail::new(
                            DetailKind::Other,
                            format!(
                                "expect_scx_bpf_error_matches({pattern:?}): regex did \
                                 not match the scheduler log + sched_ext dump corpus \
                                 (the expected bug did not fire, or its message text \
                                 changed). Captured corpus {} bytes; up to 400 bytes \
                                 follow:\n{}",
                                captured_text.len(),
                                excerpt(),
                            ),
                        ));
                    }
                }
                Err(e) => {
                    details.push(AssertDetail::new(
                        DetailKind::Other,
                        format!(
                            "expect_scx_bpf_error_matches({pattern:?}): regex \
                             compilation failed: {e}. Fix the pattern at the test \
                             declaration site — the matcher cannot evaluate against an \
                             invalid pattern",
                        ),
                    ));
                }
            }
        }
        details
    }
}

pub mod claim;
pub mod temporal;

pub use claim::{ClaimBuilder, SeqClaim, SetClaim, Verdict};
pub use temporal::{EachClaim, FracPair, PhaseMapExt, SeriesField};

/// Check that workers only ran on CPUs in `expected`.
///
/// Any worker that used a CPU outside the expected set produces a
/// failure with the unexpected CPU IDs listed.
///
/// ```
/// # use ktstr::assert::assert_isolation;
/// # use ktstr::workload::WorkerReport;
/// # use std::collections::BTreeSet;
/// # let report = WorkerReport {
/// #     tid: 1, cpus_used: [0, 1].into_iter().collect(),
/// #     work_units: 100, cpu_time_ns: 1_000_000, wall_time_ns: 2_000_000,
/// #     off_cpu_ns: 1_000_000, migration_count: 0, migrations: vec![],
/// #     max_gap_ms: 0, max_gap_cpu: 0, max_gap_at_ms: 0,
/// #     wake_latencies_ns: vec![], wake_sample_total: 0,
/// #     iteration_costs_ns: vec![], iteration_cost_sample_total: 0,
/// #     iterations: 0,
/// #     schedstat_run_delay_ns: 0, schedstat_run_count: 0,
/// #     schedstat_cpu_time_ns: 0,
/// #     completed: true,
/// #     numa_pages: std::collections::BTreeMap::new(),
/// #     vmstat_numa_pages_migrated: 0,
/// #     exit_info: None,
/// #     is_messenger: false,
/// #     ..Default::default()
/// # };
/// let expected: BTreeSet<usize> = [0, 1, 2].into_iter().collect();
/// assert!(assert_isolation(&[report], &expected).is_pass());
/// ```
pub fn assert_isolation(reports: &[WorkerReport], expected: &BTreeSet<usize>) -> AssertResult {
    let mut r = AssertResult::pass();
    for w in reports {
        let bad: BTreeSet<usize> = w.cpus_used.difference(expected).copied().collect();
        if !bad.is_empty() {
            r.record_fail(AssertDetail::new(
                DetailKind::Isolation,
                format!("tid {} ran on unexpected CPUs {:?}", w.tid, bad),
            ));
        }
    }
    r
}

/// Nearest-rank percentile of a sorted slice (`p` in `[0.0, 1.0]`).
///
/// Returns the value at index `ceil(n * p) - 1`, clamped into
/// `[0, n-1]`. For `n = 100` and `p = 0.99` this is `sorted[98]` (the
/// 99th element in 1-indexed order), not `sorted[99]` (the max). The
/// previous formulation, `ceil(n * 0.99)` without the `-1`, was
/// off-by-one and returned the max for `n = 100`.
///
/// # Preconditions
///
/// `sorted` must be non-decreasing. The function indexes by rank
/// without checking order, so an unsorted input silently returns
/// the value at the computed index — a meaningless number. A
/// `debug_assert!` enforces this in debug builds; release builds
/// skip the check (the production callers sort immediately upstream
/// — `assert_not_starved` and `assert_benchmarks` both
/// `sorted.sort_unstable()` before this call — so the runtime
/// guard is unnecessary in production paths).
///
/// An empty slice yields `0` (the caller should short-circuit
/// before invoking).
fn percentile(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    debug_assert!(
        sorted.windows(2).all(|w| w[0] <= w[1]),
        "percentile() requires sorted input; got slice with out-of-order pair",
    );
    let n = sorted.len();
    let idx = ((n as f64 * p).ceil() as usize)
        .saturating_sub(1)
        .min(n - 1);
    sorted[idx]
}

/// Build per-cgroup telemetry (pure measurement, no assertions) from
/// worker reports. This is the SINGLE telemetry builder on the assertion
/// path: `AssertPlan::assert_cgroup` calls it unconditionally and
/// [`assert_not_starved`] wraps it with the default fairness checks, so
/// per-cgroup [`CgroupStats`] is never gated behind whether a worker-check
/// assertion was configured. Empty `reports` yield a `num_workers == 0`
/// `CgroupStats` (the reduces below collapse to 0.0/0), so a declared
/// cgroup that collected no reports surfaces as a zero-worker entry rather
/// than silently vanishing from [`ScenarioStats::cgroups`].
pub fn cgroup_stats(reports: &[WorkerReport]) -> CgroupStats {
    let cpus: BTreeSet<usize> = reports
        .iter()
        .flat_map(|w| w.cpus_used.iter().copied())
        .collect();
    let pcts: Vec<f64> = reports
        .iter()
        .filter(|w| w.wall_time_ns > 0)
        .map(|w| w.off_cpu_ns as f64 / w.wall_time_ns as f64 * 100.0)
        .collect();

    // None when no worker had measurable wall time (pcts empty):
    // off-CPU% is undefined, and a not-measured cgroup must not read
    // as a measured 0% / spread-0 (perfectly fair) one. Some(_)
    // otherwise, including a real measured zero.
    let min = pcts.iter().cloned().reduce(f64::min);
    let max = pcts.iter().cloned().reduce(f64::max);
    let avg = if pcts.is_empty() {
        None
    } else {
        Some(pcts.iter().sum::<f64>() / pcts.len() as f64)
    };
    let spread = match (min, max) {
        (Some(lo), Some(hi)) => Some(hi - lo),
        _ => None,
    };

    let worst_gap = reports.iter().max_by_key(|w| w.max_gap_ms);
    let (gap_ms, gap_cpu) = worst_gap
        .map(|w| (w.max_gap_ms, w.max_gap_cpu))
        .unwrap_or((0, 0));

    // Compute benchmarking stats from worker reports.
    let all_latencies: Vec<u64> = reports
        .iter()
        .flat_map(|w| w.wake_latencies_ns.iter().copied())
        .collect();
    let (p99_us, median_us, lat_cv) = if all_latencies.is_empty() {
        (0.0, 0.0, 0.0)
    } else {
        let mut sorted = all_latencies.clone();
        sorted.sort_unstable();
        let p99 = percentile(&sorted, 0.99) as f64 / 1000.0;
        // Median routes through `percentile(sorted, 0.5)` so the
        // nearest-rank algorithm matches every other percentile in
        // the project (p99, schbench's `lat99`, the BPF latency
        // histograms). A bare `sorted[n/2]` would pick the upper of
        // the two middle samples for even `n`, while `percentile`
        // returns the value at `ceil(n * 0.5) - 1` — the lower of
        // the two middles — and that lower-bound convention is what
        // the docs on [`CgroupStats::median_wake_latency_us`] and
        // the schbench cross-reference promise.
        let median = percentile(&sorted, 0.5) as f64 / 1000.0;
        let n = all_latencies.len() as f64;
        let mean_ns = all_latencies.iter().sum::<u64>() as f64 / n;
        let cv = if mean_ns > 0.0 {
            let variance = all_latencies
                .iter()
                .map(|&v| (v as f64 - mean_ns).powi(2))
                .sum::<f64>()
                / n;
            variance.sqrt() / mean_ns
        } else {
            0.0
        };
        (p99, median, cv)
    };

    let total_iters: u64 = reports.iter().map(|w| w.iterations).sum();
    let run_delays: Vec<f64> = reports
        .iter()
        .map(|w| w.schedstat_run_delay_ns as f64 / 1000.0)
        .collect();
    let mean_run_delay = if run_delays.is_empty() {
        0.0
    } else {
        run_delays.iter().sum::<f64>() / run_delays.len() as f64
    };
    let worst_run_delay = run_delays.iter().cloned().reduce(f64::max).unwrap_or(0.0);

    let total_mig: u64 = reports.iter().map(|w| w.migration_count).sum();
    let mig_ratio = if total_iters > 0 {
        total_mig as f64 / total_iters as f64
    } else {
        0.0
    };

    // Cross-node page-migration ratio: pages migrated cross-node over the
    // cgroup's total allocated pages. `vmstat_numa_pages_migrated` is a
    // system-wide delta each worker captured over its own loop; concurrent
    // workers observe overlapping deltas, so take the MAX across the cgroup
    // (summing would inflate by the worker count) over the cgroup-wide
    // total of allocated pages. Pure measurement — populated whenever NUMA
    // pages were seen, 0.0 otherwise. (The `max_cross_node_migration_ratio`
    // CHECK in `AssertPlan::assert_cgroup` recomputes the same raw counts
    // for its diagnostic; this is the always-on telemetry.)
    let total_numa_pages: u64 = reports
        .iter()
        .map(|w| w.numa_pages.values().sum::<u64>())
        .sum();
    let migrated_pages: u64 = reports
        .iter()
        .map(|w| w.vmstat_numa_pages_migrated)
        .max()
        .unwrap_or(0);
    let cross_node_ratio = if total_numa_pages > 0 {
        migrated_pages as f64 / total_numa_pages as f64
    } else {
        0.0
    };

    CgroupStats {
        // Empty here; collect_handles labels the entry post-hoc (it has
        // the cgroup name in scope, this reports-only builder does not).
        cgroup_name: String::new(),
        num_workers: reports.len(),
        num_cpus: cpus.len(),
        cpus_used: cpus,
        avg_off_cpu_pct: avg,
        min_off_cpu_pct: min,
        max_off_cpu_pct: max,
        spread,
        max_gap_ms: gap_ms,
        max_gap_cpu: gap_cpu,
        total_migrations: total_mig,
        migration_ratio: mig_ratio,
        p99_wake_latency_us: p99_us,
        median_wake_latency_us: median_us,
        wake_latency_cv: lat_cv,
        total_iterations: total_iters,
        total_cpu_time_ns: reports.iter().map(|w| w.schedstat_cpu_time_ns).sum(),
        mean_run_delay_us: mean_run_delay,
        worst_run_delay_us: worst_run_delay,
        // page_locality requires the expected NUMA node set (the cpuset's
        // nodes), which this reports-only builder does not have. It is
        // populated by `AssertPlan::assert_cgroup` when `numa_nodes` is
        // supplied; left 0.0 here (no NUMA context).
        page_locality: 0.0,
        cross_node_migration_ratio: cross_node_ratio,
        ext_metrics: BTreeMap::new(),
    }
}

/// Per-phase per-cgroup RAW-component builder — the sibling of [`cgroup_stats`]
/// that emits [`PhaseCgroupStats`]'s un-reduced components instead of the
/// reduced ratios/percentiles, so the distributional re-pool recomputes each
/// aggregate from the pooled components at every level. Every [`CgroupStats`]
/// reduction re-pools from these fields: avg/min/max/spread off-CPU% from
/// `off_cpu_pcts`; p99/median/CV from `wake_latencies_ns`; mean/worst run-delay
/// from `run_delays_ns` (RAW ns, the re-pool divides by 1000); migration_ratio
/// / iterations_per_cpu_sec / iterations_per_worker from the counters;
/// page_locality / cross_node_migration_ratio from the numa counters; the
/// coupled worst gap from the argmax pair; cpus_used / num_cpus from `cpus_used`.
///
/// `expected_nodes` is this cgroup's cpuset NUMA-node set (from
/// [`crate::topology::TestTopology::numa_nodes_for_cpuset`]); `numa_pages_local`
/// is the page count on those nodes (0 when `None`, mirroring [`cgroup_stats`]
/// leaving `page_locality` 0.0 without NUMA context — the partition lives with
/// the caller that has the node set, as [`AssertPlan::assert_cgroup`] does).
/// The whole-run [`cgroup_stats`] reductions stay the run-level authority; this
/// feeds the per-phase [`PhaseBucket::per_cgroup`] carrier.
///
/// RE-POOL GUARD CONTRACT: the div-by-zero / not-measured guards live in
/// [`cgroup_stats`], NOT in these raw components — a future re-pool over them
/// MUST mirror them exactly or ship a NaN/Inf or a not-measured-vs-zero
/// collapse: `migration_ratio` only when `total_iterations > 0`;
/// `cross_node_migration_ratio` / `page_locality` only when `numa_pages_total >
/// 0`; mean/worst run-delay only when `run_delays_ns` is non-empty; and
/// avg/min/max/spread off-CPU% return None (not 0.0) when `off_cpu_pcts` is
/// empty (the not-measured state).
pub(crate) fn phase_cgroup_stats(
    reports: &[WorkerReport],
    expected_nodes: Option<&BTreeSet<usize>>,
) -> PhaseCgroupStats {
    let cpus_used: BTreeSet<usize> = reports
        .iter()
        .flat_map(|w| w.cpus_used.iter().copied())
        .collect();
    // Per-worker off-CPU% (only workers with measurable wall time), un-reduced.
    // EMPTY = not measured: the re-pool then yields None for avg/min/max/spread,
    // preserving the not-measured-vs-measured-zero distinction cgroup_stats keeps.
    let off_cpu_pcts: Vec<f64> = reports
        .iter()
        .filter(|w| w.wall_time_ns > 0)
        .map(|w| w.off_cpu_ns as f64 / w.wall_time_ns as f64 * 100.0)
        .collect();
    // Pool every worker's already per-worker-capped wake-latency vec, RE-CAPPING
    // the concatenation at MAX_WAKE_SAMPLES via the same Algorithm-R reservoir the
    // per-worker path uses. This carrier is the FIRST to serialize raw samples
    // over the size-limited guest bulk port (the AssertResult); without the
    // re-cap the pool would be workers × MAX_WAKE_SAMPLES and could overrun the
    // 16 MiB frame on a many-core host, flipping a PASS to a truncated FAIL. The
    // reservoir is distribution-preserving, so p99 / median / CV re-pool over it
    // as cgroup_stats does over the per-worker pool; `wake_sample_total` keeps the
    // TRUE pre-cap population for the re-pool. PARITY: for pools ≤
    // MAX_WAKE_SAMPLES the reservoir is the full concatenation, so the re-pool is
    // VALUE-FOR-VALUE with cgroup_stats; above the cap it is a distribution-
    // preserving SUBSAMPLE (cgroup_stats keeps the full concat), so the re-pool is
    // distribution-equivalent, not byte-identical — see the wake_latencies_ns
    // field doc for the full contract. PhaseCgroupStats::merge keeps same-name
    // carriers bounded too: ≤cap it concatenates (value-for-value), >cap it uses a
    // population-WEIGHTED reservoir merge (weighted_merge_reservoirs) so the merged
    // subsample is unbiased rather than length-skewed toward the smaller carrier.
    let mut wake_latencies_ns: Vec<u64> = Vec::new();
    let mut pooled_wake_count: u64 = 0;
    for w in reports {
        for &sample in &w.wake_latencies_ns {
            crate::workload::reservoir_push(
                &mut wake_latencies_ns,
                &mut pooled_wake_count,
                sample,
                crate::workload::MAX_WAKE_SAMPLES,
            );
        }
    }
    let wake_sample_total: u64 = reports.iter().map(|w| w.wake_sample_total).sum();
    // RAW ns, one per worker — NOT divided by 1000. cgroup_stats divides at
    // reduction time; the re-pool over the concatenated samples divides once,
    // so pre-dividing here would double-divide (a 1000x error).
    let run_delays_ns: Vec<u64> = reports.iter().map(|w| w.schedstat_run_delay_ns).collect();
    // Coupled worst gap: take (ms, cpu) TOGETHER from the worst worker (argmax),
    // never two independent maxes — keeps the gap bound to its CPU.
    let (max_gap_ms, max_gap_cpu) = reports
        .iter()
        .max_by_key(|w| w.max_gap_ms)
        .map(|w| (w.max_gap_ms, w.max_gap_cpu))
        .unwrap_or((0, 0));
    let total_migrations: u64 = reports.iter().map(|w| w.migration_count).sum();
    let total_iterations: u64 = reports.iter().map(|w| w.iterations).sum();
    // schedstat_cpu_time_ns (task->se.sum_exec_runtime), NOT cpu_time_ns
    // (CLOCK_THREAD_CPUTIME_ID) — matches cgroup_stats's total_cpu_time_ns.
    let total_cpu_time_ns: u64 = reports.iter().map(|w| w.schedstat_cpu_time_ns).sum();
    let numa_pages_total: u64 = reports
        .iter()
        .map(|w| w.numa_pages.values().sum::<u64>())
        .sum();
    // System-wide /proc/vmstat numa_pages_migrated delta each worker observes
    // redundantly -> MAX, not SUM (summing inflates by the worker count).
    let cross_node_migrated: u64 = reports
        .iter()
        .map(|w| w.vmstat_numa_pages_migrated)
        .max()
        .unwrap_or(0);
    // Pages on the cgroup's expected NUMA nodes (page_locality numerator),
    // partitioned exactly as AssertPlan::assert_cgroup does; 0 without a node
    // set (mirrors cgroup_stats leaving page_locality 0.0 absent NUMA context).
    let numa_pages_local: u64 = expected_nodes
        .map(|nodes| {
            let mut local = 0u64;
            for w in reports {
                for (&node, &count) in &w.numa_pages {
                    if nodes.contains(&node) {
                        local += count;
                    }
                }
            }
            local
        })
        .unwrap_or(0);
    PhaseCgroupStats {
        num_workers: reports.len(),
        cpus_used,
        wake_latencies_ns,
        wake_sample_total,
        run_delays_ns,
        off_cpu_pcts,
        total_migrations,
        total_iterations,
        total_cpu_time_ns,
        numa_pages_local,
        numa_pages_total,
        cross_node_migrated,
        max_gap_ms,
        max_gap_cpu,
        // Fresh carrier built from worker reports — never stripped.
        stripped: false,
    }
}

/// Build the single-bucket guest-side per-phase carrier for one step-local
/// cgroup: a [`PhaseBucket`] at `step_index` whose only payload is the
/// `per_cgroup` entry `name -> phase_cgroup_stats(reports, expected_nodes)`.
///
/// 6b: the guest emits one of these per step-local cgroup at `collect_step`
/// teardown ([`crate::scenario::collect_handles`]). The window is the
/// merge-neutral `(u64::MAX, 0)` sentinel and `metrics` is empty: the carrier
/// contributes ONLY `per_cgroup`. When folded into the host-rebuilt bucket of
/// the same `step_index` ([`fold_guest_per_cgroup_into_host_buckets`] via
/// [`merge_matched_phase_buckets`]) the `MAX`/`0` window is a no-op against the
/// host's real window (`min`/`max`), so the host's window and metrics win and
/// only `per_cgroup` is carried. The `label` uses [`Phase`]'s `Display` so an
/// orphan carrier (no host bucket) still reads `BASELINE`/`Step[k]`.
pub(crate) fn step_per_cgroup_bucket(
    name: &str,
    reports: &[WorkerReport],
    expected_nodes: Option<&BTreeSet<usize>>,
    step_index: u16,
) -> PhaseBucket {
    let mut per_cgroup = std::collections::BTreeMap::new();
    per_cgroup.insert(name.to_string(), phase_cgroup_stats(reports, expected_nodes));
    PhaseBucket {
        step_index,
        label: Phase::from(step_index).to_string(),
        start_ms: u64::MAX,
        end_ms: 0,
        sample_count: 0,
        metrics: std::collections::BTreeMap::new(),
        per_cgroup,
    }
}

/// Roll a single cgroup's [`CgroupStats`] up into a one-cgroup
/// [`ScenarioStats`]. The KEPT typed `worst_*` fields carry this cgroup's
/// values and fold across cgroups in [`AssertResult::merge`]: max for the
/// higher-is-worse fields (`worst_spread`, `worst_migration_ratio`,
/// `worst_cross_node_migration_ratio`, and the coupled `worst_gap_ms` /
/// `worst_gap_cpu`) and lowest-non-zero for `worst_page_locality`. The
/// wake-latency / run-delay distributions, the iteration efficiencies, and
/// the wake-latency tail ratio are NOT carried here — they have no typed
/// field and re-pool run-level POST-merge from `stats.phases[].per_cgroup`
/// / `stats.cgroups` (the tail ratio is the max over the per-cgroup
/// `CgroupStats::wake_latency_tail_ratio`) in
/// [`populate_run_distribution_metrics`].
/// `cgroups` carries exactly this one entry so merge appends one per
/// handle without double-counting.
fn scenario_stats_for_cgroup(cg: &CgroupStats) -> ScenarioStats {
    ScenarioStats {
        total_workers: cg.num_workers,
        total_cpus: cg.num_cpus,
        total_migrations: cg.total_migrations,
        // worst_spread is higher-is-worse (merge takes max). A
        // not-measured cgroup (`spread == None`) maps to 0.0 — the
        // neutral element for max, and the gauntlet layer's
        // documented no-data convention. A measured zero stays 0.0.
        worst_spread: cg.spread.unwrap_or(0.0),
        worst_gap_ms: cg.max_gap_ms,
        worst_gap_cpu: cg.max_gap_cpu,
        worst_migration_ratio: cg.migration_ratio,
        total_iterations: cg.total_iterations,
        worst_page_locality: cg.page_locality,
        worst_cross_node_migration_ratio: cg.cross_node_migration_ratio,
        ext_metrics: cg.ext_metrics.clone(),
        cgroups: vec![cg.clone()],
        phases: Vec::new(),
    }
}

/// Record the DEFAULT fairness outcomes (Starved / Unfair / Stuck) for one
/// cgroup against the framework default thresholds
/// ([`spread_threshold_pct`] / [`gap_threshold_ms`]). Telemetry is built
/// separately by [`cgroup_stats`]; this only appends fail outcomes, so it
/// is shared by [`assert_not_starved`] and the `not_starved` arm of
/// [`AssertPlan::assert_cgroup`] without rebuilding stats.
fn record_default_fairness(r: &mut AssertResult, cg: &CgroupStats, reports: &[WorkerReport]) {
    for w in reports {
        if w.work_units == 0 {
            r.record_fail(AssertDetail::new(
                DetailKind::Starved,
                format!("tid {} starved (0 work units)", w.tid),
            ));
        }
    }
    // Off-cpu spread above the default threshold, gated on >=2 workers
    // with measurable wall time (the historical `pcts.len() >= 2`).
    // `cg.spread` is None when off-CPU% was not measured — inconclusive,
    // never flagged unfair.
    let measurable = reports.iter().filter(|w| w.wall_time_ns > 0).count();
    let spread_limit = spread_threshold_pct();
    if let Some(spread) = cg.spread
        && spread > spread_limit
        && measurable >= 2
    {
        r.record_fail(AssertDetail::new(
            DetailKind::Unfair,
            format!(
                "unfair cgroup: spread={:.0}% ({:.0}-{:.0}%) {} workers on {} cpus (threshold {:.0}%)",
                spread,
                cg.min_off_cpu_pct.unwrap_or(0.0),
                cg.max_off_cpu_pct.unwrap_or(0.0),
                cg.num_workers,
                cg.num_cpus,
                spread_limit,
            ),
        ));
    }
    let gap_limit = gap_threshold_ms();
    for w in reports {
        if w.max_gap_ms > gap_limit {
            r.record_fail(AssertDetail::new(
                DetailKind::Stuck,
                format!(
                    "tid {} stuck {}ms on cpu{} at +{}ms (threshold {}ms)",
                    w.tid, w.max_gap_ms, w.max_gap_cpu, w.max_gap_at_ms, gap_limit,
                ),
            ));
        }
    }
}

/// Default fairness check for one cgroup's worker reports: builds the
/// per-cgroup telemetry ([`cgroup_stats`]) and records Starved / Unfair /
/// Stuck against the framework default thresholds. Telemetry is ALWAYS
/// populated — including a `num_workers == 0` entry for empty reports — so
/// `r.stats.cgroups` is never empty for a declared cgroup, independent of
/// whether any fail outcome fired.
pub fn assert_not_starved(reports: &[WorkerReport]) -> AssertResult {
    let cg = cgroup_stats(reports);
    let mut r = AssertResult::pass();
    record_default_fairness(&mut r, &cg, reports);
    r.stats = scenario_stats_for_cgroup(&cg);
    r
}

/// Check throughput parity across workers: coefficient of variation and
/// minimum work rate.
///
/// `max_cv`: maximum allowed coefficient of variation (stddev/mean) for
/// work_units / cpu_time_ns across workers. `None` skips the CV check.
///
/// `min_rate`: minimum work_units per CPU-second. `None` skips the floor check.
///
/// When every worker recorded `cpu_time_ns == 0`, both gates record
/// their OWN Inconclusive outcome (the CV gate emits a "CV cannot be
/// computed" detail; the min_rate gate emits a "rates cannot be
/// computed" detail). Each gate carries its own diagnostic so a
/// caller that supplies only one of the two threshold parameters
/// sees the matching Inconclusive message and an operator reading
/// [`AssertResult::inconclusive_details`] can identify which gate(s)
/// misfired without re-deriving the inputs.
///
/// ```
/// # use ktstr::assert::assert_throughput_parity;
/// # use ktstr::workload::WorkerReport;
/// # let mk = |units, cpu_ns| WorkerReport {
/// #     tid: 1, cpus_used: [0].into_iter().collect(),
/// #     work_units: units, cpu_time_ns: cpu_ns, wall_time_ns: cpu_ns,
/// #     off_cpu_ns: cpu_ns, migration_count: 0, migrations: vec![],
/// #     max_gap_ms: 0, max_gap_cpu: 0, max_gap_at_ms: 0,
/// #     wake_latencies_ns: vec![], wake_sample_total: 0,
/// #     iteration_costs_ns: vec![], iteration_cost_sample_total: 0,
/// #     iterations: 0,
/// #     schedstat_run_delay_ns: 0, schedstat_run_count: 0,
/// #     schedstat_cpu_time_ns: 0,
/// #     completed: true,
/// #     numa_pages: std::collections::BTreeMap::new(),
/// #     vmstat_numa_pages_migrated: 0,
/// #     exit_info: None,
/// #     is_messenger: false,
/// #     ..Default::default()
/// # };
/// // Equal throughput -> low CV -> passes.
/// let reports = [mk(1000, 1_000_000_000), mk(1000, 1_000_000_000)];
/// assert!(assert_throughput_parity(&reports, Some(0.5), None).is_pass());
/// ```
pub fn assert_throughput_parity(
    reports: &[WorkerReport],
    max_cv: Option<f64>,
    min_rate: Option<f64>,
) -> AssertResult {
    let mut r = AssertResult::pass();
    if reports.is_empty() {
        return r;
    }

    // Compute per-worker throughput: work_units / cpu_seconds
    let rates: Vec<f64> = reports
        .iter()
        .map(|w| {
            if w.cpu_time_ns == 0 {
                0.0
            } else {
                w.work_units as f64 / (w.cpu_time_ns as f64 / 1e9)
            }
        })
        .collect();

    let n = rates.len() as f64;
    let mean = rates.iter().sum::<f64>() / n;

    // Detect the all-zero-cpu condition once so a call with both
    // `max_cv` and `min_rate` set surfaces a single Inconclusive
    // listing every threshold that couldn't evaluate, rather than
    // emitting one record per gate (which produced duplicate
    // "denominator is zero" diagnostics for the same root cause).
    let all_zero_cpu = reports.iter().all(|w| w.cpu_time_ns == 0);

    if all_zero_cpu && (max_cv.is_some() || min_rate.is_some()) {
        let mut limits: Vec<String> = Vec::with_capacity(2);
        if let Some(cv_limit) = max_cv {
            limits.push(format!("max_cv {cv_limit:.3}"));
        }
        if let Some(floor) = min_rate {
            limits.push(format!("min_rate {floor:.0}"));
        }
        r.record_inconclusive(AssertDetail::new(
            DetailKind::Benchmark,
            format!(
                "throughput parity inconclusive: all {} workers recorded zero cpu_time_ns — \
                 denominator is zero, rates cannot be computed; {} neither pass nor fail \
                 (was the workload able to run?)",
                reports.len(),
                limits.join(" + "),
            ),
        ));
        return r;
    }

    if let Some(cv_limit) = max_cv
        && mean > 0.0
        && rates.len() >= 2
    {
        let variance = rates.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
        let stddev = variance.sqrt();
        let cv = stddev / mean;
        if cv > cv_limit {
            r.record_fail(AssertDetail::new(
                DetailKind::Benchmark,
                format!(
                    "throughput CV {cv:.3} exceeds limit {cv_limit:.3} (mean={mean:.0} work/cpu_s)"
                ),
            ));
        }
    }

    if let Some(floor) = min_rate {
        // Skip per-worker zero-cpu cases: their rate is forced to
        // 0.0 above, and comparing that to `floor` would synthesize
        // a guaranteed Fail with a misleading "below floor" message
        // when the real story is "this worker recorded no CPU time
        // — the rate is unknowable, not failing". The all-zero-cpu
        // case is already handled at the top of the function as a
        // single combined Inconclusive.
        for (i, &rate) in rates.iter().enumerate() {
            if reports[i].cpu_time_ns == 0 {
                continue;
            }
            if rate < floor {
                r.record_fail(AssertDetail::new(
                    DetailKind::Benchmark,
                    format!(
                        "worker {} throughput {rate:.0} work/cpu_s below floor {floor:.0}",
                        reports[i].tid
                    ),
                ));
            }
        }
    }

    r
}

/// Check benchmarking metrics: p99 wake latency, wake latency CV,
/// and minimum iteration rate.
///
/// ```
/// # use ktstr::assert::assert_benchmarks;
/// # use ktstr::workload::WorkerReport;
/// # let report = WorkerReport {
/// #     tid: 1, cpus_used: [0].into_iter().collect(),
/// #     work_units: 1000, cpu_time_ns: 2_500_000_000,
/// #     wall_time_ns: 5_000_000_000, off_cpu_ns: 2_500_000_000,
/// #     migration_count: 0, migrations: vec![],
/// #     max_gap_ms: 50, max_gap_cpu: 0, max_gap_at_ms: 1000,
/// #     wake_latencies_ns: vec![100, 200, 300, 400, 500],
/// #     wake_sample_total: 5,
/// #     iteration_costs_ns: vec![], iteration_cost_sample_total: 0,
/// #     iterations: 1000,
/// #     schedstat_run_delay_ns: 0, schedstat_run_count: 0,
/// #     schedstat_cpu_time_ns: 0,
/// #     completed: true,
/// #     numa_pages: std::collections::BTreeMap::new(),
/// #     vmstat_numa_pages_migrated: 0,
/// #     exit_info: None,
/// #     is_messenger: false,
/// #     ..Default::default()
/// # };
/// // p99 = 500ns, well under 10000ns limit.
/// assert!(assert_benchmarks(&[report], Some(10000), None, None).is_pass());
/// ```
pub fn assert_benchmarks(
    reports: &[WorkerReport],
    max_p99_ns: Option<u64>,
    max_cv: Option<f64>,
    min_iter_rate: Option<f64>,
) -> AssertResult {
    let mut r = AssertResult::pass();
    if reports.is_empty() {
        // No worker reports means nothing to measure — any benchmark
        // threshold the caller supplied cannot be evaluated. A silent
        // pass would let thresholds look "green" on a broken run that
        // never produced signal; surface it as skip so the operator
        // knows the benchmark was not actually exercised.
        return AssertResult::skip("no worker reports — benchmark skipped");
    }

    // Collect all wake latencies across workers.
    let all_latencies: Vec<u64> = reports
        .iter()
        .flat_map(|w| w.wake_latencies_ns.iter().copied())
        .collect();

    if let Some(p99_limit) = max_p99_ns
        && !all_latencies.is_empty()
    {
        let mut sorted = all_latencies.clone();
        sorted.sort_unstable();
        let p99 = percentile(&sorted, 0.99);
        if p99 > p99_limit {
            r.record_fail(AssertDetail::new(
                DetailKind::Benchmark,
                format!(
                    "p99 wake latency {p99}ns exceeds limit {p99_limit}ns ({} samples)",
                    sorted.len()
                ),
            ));
        }
    }

    if let Some(cv_limit) = max_cv
        && all_latencies.len() >= 2
    {
        let n = all_latencies.len() as f64;
        let mean = all_latencies.iter().sum::<u64>() as f64 / n;
        if mean > 0.0 {
            let variance = all_latencies
                .iter()
                .map(|&v| (v as f64 - mean).powi(2))
                .sum::<f64>()
                / n;
            let cv = variance.sqrt() / mean;
            if cv > cv_limit {
                r.record_fail(AssertDetail::new(
                    DetailKind::Benchmark,
                    format!(
                        "wake latency CV {cv:.3} exceeds limit {cv_limit:.3} (mean={mean:.0}ns)"
                    ),
                ));
            }
        } else {
            // CV is dispersion / mean. With mean == 0 every captured
            // wake-latency sample was zero, so the denominator is
            // zero and CV is undefined — neither pass nor fail is
            // truthful. The same workload that fails to record
            // measurable wake latency at all (typically: nothing
            // actually woke, or every wake landed at <1ns and
            // truncated to zero in the ns counter) previously slid
            // past the gate as a silent pass; surface it as
            // Inconclusive so a broken benchmarking run does not
            // masquerade as a CV-compliant one.
            r.record_inconclusive(AssertDetail::new(
                DetailKind::Benchmark,
                format!(
                    "wake latency CV inconclusive: all {} sample(s) had zero mean wake \
                     latency — denominator is zero, CV cannot be computed; limit \
                     {cv_limit:.3} neither pass nor fail (did any wake event capture a \
                     non-zero latency?)",
                    all_latencies.len(),
                ),
            ));
        }
    }

    if let Some(rate_floor) = min_iter_rate {
        // Skip per-worker zero-wall cases (rate is unknowable when
        // wall_time_ns == 0) but count them: if every worker had
        // zero wall_time, the gate silently passed before — record
        // Inconclusive instead so a broken run that produced no
        // signal at all doesn't masquerade as a passing benchmark.
        let mut zero_wall_count = 0usize;
        for w in reports {
            if w.wall_time_ns == 0 {
                zero_wall_count += 1;
                continue;
            }
            let rate = w.iterations as f64 / (w.wall_time_ns as f64 / 1e9);
            if rate < rate_floor {
                r.record_fail(AssertDetail::new(
                    DetailKind::Benchmark,
                    format!(
                        "worker {} iteration rate {rate:.1}/s below floor {rate_floor:.1}/s",
                        w.tid
                    ),
                ));
            }
        }
        if zero_wall_count == reports.len() {
            r.record_inconclusive(AssertDetail::new(
                DetailKind::Benchmark,
                format!(
                    "min iteration rate inconclusive: all {} workers recorded zero wall_time_ns — \
                     denominator is zero, rate cannot be computed; floor {rate_floor:.1}/s \
                     neither pass nor fail (was the workload able to run?)",
                    reports.len()
                ),
            ));
        }
    }

    r
}

/// Assert that every SCX event counter in `events` is at or below
/// `max_count`. `events` is a slice of `(name, count)` pairs sourced
/// from the kernel's per-task `scx_event_stats` (see `kernel/sched/ext.c`,
/// `SCX_EV_*` macros) — typically aggregated and surfaced via
/// `monitor::ScxEventDeltas` or sidecar `GauntletRow.fallback_count` /
/// `keep_last_count` fields. Pass `None` for `max_count` to require zero
/// (the strict default — error-class events should not fire under a
/// healthy scheduler).
///
/// The assertion is decoupled from the `monitor` module on purpose:
/// callers harvest the counters they care about (via the live monitor
/// path or by reading sidecar JSON post-hoc) and feed name/count
/// pairs in. This keeps the assert API surface decoupled from the
/// kernel-side counter inventory, which evolves across kernel
/// versions — adding a new `SCX_EV_*` does not force an API change
/// here.
///
/// Returns a passing result if every counter is within bound; failures
/// concatenate one [`AssertDetail`] per offending counter under
/// [`DetailKind::SchedulerEvent`] so an operator can identify which
/// events fired without scanning the full counter set.
///
/// ```
/// # use ktstr::assert::assert_scx_events_clean;
/// // Strict default — every counter must be zero.
/// let r = assert_scx_events_clean(&[("enq_skip_exiting", 0), ("dispatch_local_dsq_offline", 0)], None);
/// assert!(r.is_pass());
///
/// // A non-zero error-class counter fails.
/// let r = assert_scx_events_clean(&[("enq_skip_exiting", 7)], None);
/// assert!(r.is_fail());
///
/// // Caller-supplied bound tolerates small counts.
/// let r = assert_scx_events_clean(&[("dispatch_keep_last", 3)], Some(10));
/// assert!(r.is_pass());
/// ```
pub fn assert_scx_events_clean(events: &[(&str, i64)], max_count: Option<i64>) -> AssertResult {
    let mut r = AssertResult::pass();
    for (name, count) in events {
        // Kernel `scx_event_stats` counters are monotonic u64 — a
        // negative i64 here means the source data is corrupted
        // (counter reset, wraparound on a signed conversion, or
        // sidecar JSON bit-loss). Treat negatives as failures rather
        // than letting them silently pass `*count > bound` for any
        // non-negative bound.
        let failed = match max_count {
            // Strict default: every counter must be exactly zero.
            // `*count > 0` would let -5 slip through.
            None => *count != 0,
            // Bounded: reject negatives explicitly, then enforce
            // the upper bound.
            Some(bound) => *count < 0 || *count > bound,
        };
        if failed {
            let bound_desc = match max_count {
                None => "0".to_string(),
                Some(b) => b.to_string(),
            };
            r.record_fail(AssertDetail::new(
                DetailKind::SchedulerEvent,
                format!("scx event `{name}` count {count} exceeds bound {bound_desc}",),
            ));
        }
    }
    r
}

/// Threshold-preset bundle for [`assert_thresholds`]. Captures the
/// guarantees a scheduler-under-test should meet on a healthy run:
/// wake latency stays within bound, per-iteration compute cost stays
/// within bound, CPU migrations stay within bound, and every worker
/// makes some forward progress.
///
/// Each `Option` field is independent — `None` skips that check. A
/// `AbsoluteThresholds` with every field `None` is a no-op (the
/// returned [`AssertResult`] always passes), useful as a starting
/// point for builder-style composition. Construct the all-`None`
/// thresholds via `AbsoluteThresholds::default()` and chain the
/// `max_*` / `min_*` setters (e.g. `AbsoluteThresholds::default().max_migrations(5)`)
/// or spread into a struct literal (`AbsoluteThresholds { max_migrations: Some(5), ..Default::default() }`).
/// Use [`Self::strict`] for the "every check enabled with sane defaults" preset.
///
/// Distinct from [`Assert`]: `Assert` is the merge-tree threshold
/// config consumed by the worker-side `AssertPlan`; `AbsoluteThresholds`
/// is a flat preset designed for direct invocation in test bodies
/// where the test author wants a one-call multi-field check without
/// engaging the merge chain. The two surfaces compose — a test can
/// run `assert_thresholds` against a worker-report slice AND merge the
/// `Assert`-derived result into the same accumulator via
/// [`AssertResult::merge`].
#[must_use = "AbsoluteThresholds only takes effect when passed to assert_thresholds"]
#[derive(Debug, Clone, Copy, Default)]
pub struct AbsoluteThresholds {
    /// Maximum acceptable p99 wake latency (nanoseconds). Compared
    /// against the pooled p99 across every worker's
    /// [`WorkerReport::wake_latencies_ns`]. `None` skips the check.
    /// Same units / semantics as [`Assert::max_p99_wake_latency_ns`].
    pub max_p99_wake_latency_ns: Option<u64>,
    /// Maximum acceptable p99 per-iteration compute cost (nanoseconds).
    /// Compared against the pooled p99 across every worker's
    /// [`WorkerReport::iteration_costs_ns`]. `None` skips the check.
    /// Only meaningful for compute work types that populate the
    /// reservoir (`AluHot`, `SmtSiblingSpin`, `IpcVariance`); blocking
    /// variants report empty `iteration_costs_ns` and the check is a
    /// no-op for those.
    pub max_iteration_cost_p99_ns: Option<u64>,
    /// Maximum acceptable total CPU migrations across every worker.
    /// Compared against the sum of [`WorkerReport::migration_count`].
    /// `None` skips the check. Distinct from
    /// [`Assert::max_migration_ratio`] (migrations per iteration) —
    /// this is an absolute count, useful when the test pins a known
    /// workload size and migrations should stay below a fixed ceiling
    /// regardless of how many iterations completed.
    pub max_migrations: Option<u64>,
    /// Minimum acceptable per-worker work_units. Every worker must
    /// have completed at least this many work units; one starved
    /// worker fails the check. `None` skips. Distinct from
    /// [`assert_not_starved`]'s zero-work-units check, which gates
    /// only against literal zero — this gate accepts a non-zero
    /// floor so a test can reject "barely made progress" runs that
    /// pass the strict starvation gate.
    pub min_work_units: Option<u64>,
}

impl AbsoluteThresholds {
    /// Sane-default preset: p99 wake latency under 10ms, p99
    /// iteration cost under 1ms, total migrations under 1000, every
    /// worker completes ≥1 work unit. The defaults are deliberately
    /// loose — a threshold set tight enough to catch egregious
    /// regressions without flagging every routine scheduler
    /// perturbation. Tests
    /// that need tighter bounds should set the fields explicitly via
    /// the bare-verb builder methods rather than tuning these constants.
    pub const fn strict() -> Self {
        Self {
            max_p99_wake_latency_ns: Some(10_000_000),
            max_iteration_cost_p99_ns: Some(1_000_000),
            max_migrations: Some(1000),
            min_work_units: Some(1),
        }
    }

    /// Builder setter for [`Self::max_p99_wake_latency_ns`].
    pub const fn max_p99_wake_latency_ns(mut self, v: u64) -> Self {
        self.max_p99_wake_latency_ns = Some(v);
        self
    }

    /// Builder setter for [`Self::max_iteration_cost_p99_ns`].
    pub const fn max_iteration_cost_p99_ns(mut self, v: u64) -> Self {
        self.max_iteration_cost_p99_ns = Some(v);
        self
    }

    /// Builder setter for [`Self::max_migrations`].
    pub const fn max_migrations(mut self, v: u64) -> Self {
        self.max_migrations = Some(v);
        self
    }

    /// Builder setter for [`Self::min_work_units`].
    pub const fn min_work_units(mut self, v: u64) -> Self {
        self.min_work_units = Some(v);
        self
    }
}

/// Run every check in `thresholds` against `reports`, merging results
/// into a single [`AssertResult`]. A `None` field on the thresholds
/// skips that check.
///
/// An empty `reports` slice short-circuits to a skip (`"no worker
/// reports to evaluate"`) regardless of thresholds content — silently
/// passing thresholds against zero samples would let them look
/// "green" on a run that produced no measurement.
///
/// Field-to-check mapping:
/// - `max_p99_wake_latency_ns` -> pooled p99 across every worker's
///   `wake_latencies_ns`; tagged [`DetailKind::Benchmark`].
/// - `max_iteration_cost_p99_ns` -> pooled p99 across every worker's
///   `iteration_costs_ns`; tagged [`DetailKind::Benchmark`].
/// - `max_migrations` -> sum of `migration_count` across workers;
///   tagged [`DetailKind::Migration`].
/// - `min_work_units` -> per-worker `work_units >= floor`; tagged
///   [`DetailKind::Starved`] when a worker is below the floor.
///
/// The wake-latency check delegates to [`assert_benchmarks`] for the
/// percentile path so the same nearest-rank algorithm applies; the
/// iteration-cost check uses an inline percentile call against the
/// pooled `iteration_costs_ns` reservoir.
///
/// ```
/// # use ktstr::assert::{AbsoluteThresholds, assert_thresholds};
/// # use ktstr::workload::WorkerReport;
/// # let report = WorkerReport {
/// #     tid: 1, cpus_used: [0].into_iter().collect(),
/// #     work_units: 1000, cpu_time_ns: 2_500_000_000,
/// #     wall_time_ns: 5_000_000_000, off_cpu_ns: 2_500_000_000,
/// #     migration_count: 5, migrations: vec![],
/// #     max_gap_ms: 50, max_gap_cpu: 0, max_gap_at_ms: 1000,
/// #     wake_latencies_ns: vec![100, 200, 300, 400, 500],
/// #     wake_sample_total: 5,
/// #     iteration_costs_ns: vec![1000, 2000, 3000, 4000, 5000],
/// #     iteration_cost_sample_total: 5,
/// #     iterations: 1000,
/// #     schedstat_run_delay_ns: 0, schedstat_run_count: 0,
/// #     schedstat_cpu_time_ns: 0,
/// #     completed: true,
/// #     numa_pages: std::collections::BTreeMap::new(),
/// #     vmstat_numa_pages_migrated: 0,
/// #     exit_info: None,
/// #     affinity_error: None,
/// #     is_messenger: false,
/// #     group_idx: 0,
/// # };
/// // Strict preset on a healthy run — passes.
/// let r = assert_thresholds(&[report], &AbsoluteThresholds::strict());
/// assert!(r.is_pass());
/// ```
pub fn assert_thresholds(
    reports: &[WorkerReport],
    thresholds: &AbsoluteThresholds,
) -> AssertResult {
    // Empty `reports` means nothing was measured. Returning a fresh
    // `pass()` here would silently green-light a broken run that
    // produced no signal; delegating to `assert_benchmarks` and
    // merging its skip would lose the skip flag (`AssertResult::merge`
    // ANDs `skipped`, so `pass.merge(skip) == passed-not-skipped`).
    // Surface the skip directly so the operator sees the thresholds
    // weren't actually exercised.
    if reports.is_empty() {
        return AssertResult::skip("no worker reports to evaluate");
    }

    let mut r = AssertResult::pass();

    // Wake-latency p99: reuse the existing `assert_benchmarks` path
    // so the percentile algorithm stays unified. With `reports`
    // non-empty here, `assert_benchmarks` cannot return a skip —
    // the merge sees only pass/fail, preserving thresholds semantics.
    if thresholds.max_p99_wake_latency_ns.is_some() {
        r.merge(assert_benchmarks(
            reports,
            thresholds.max_p99_wake_latency_ns,
            None,
            None,
        ));
    }

    // Iteration-cost p99: pooled across every worker's reservoir.
    // Skipped when no samples are present — compute work types that
    // populate `iteration_costs_ns` are sparse, so an empty pooled
    // set is the common case for blocking variants and not a failure.
    if let Some(cost_limit) = thresholds.max_iteration_cost_p99_ns {
        let all_costs: Vec<u64> = reports
            .iter()
            .flat_map(|w| w.iteration_costs_ns.iter().copied())
            .collect();
        if !all_costs.is_empty() {
            let mut sorted = all_costs.clone();
            sorted.sort_unstable();
            let p99 = percentile(&sorted, 0.99);
            if p99 > cost_limit {
                r.record_fail(AssertDetail::new(
                    DetailKind::Benchmark,
                    format!(
                        "p99 iteration cost {p99}ns exceeds limit {cost_limit}ns ({} samples)",
                        sorted.len(),
                    ),
                ));
            }
        }
    }

    // Total migrations across all workers: absolute-count gate
    // (distinct from migration_ratio which is a per-iteration rate).
    if let Some(max_mig) = thresholds.max_migrations {
        let total_mig: u64 = reports.iter().map(|w| w.migration_count).sum();
        if total_mig > max_mig {
            r.record_fail(AssertDetail::new(
                DetailKind::Migration,
                format!(
                    "total migrations {total_mig} exceeds limit {max_mig} ({} workers)",
                    reports.len(),
                ),
            ));
        }
    }

    // Per-worker work_units floor: every worker must have completed
    // at least `min` work units. One starved worker fails the check.
    if let Some(min_units) = thresholds.min_work_units {
        for w in reports {
            if w.work_units < min_units {
                r.record_fail(AssertDetail::new(
                    DetailKind::Starved,
                    format!(
                        "tid {} work_units {} below floor {min_units}",
                        w.tid, w.work_units,
                    ),
                ));
            }
        }
    }

    r
}

// (The legacy `Expect` / `Checks` / `CheckBuilder` types previously
// living here were replaced by the [`Verdict`]-based claim API
// (defined further up in this file). The new flow is
// `Assert::default_checks().verdict().claim_<field>(stats).at_most(N)` for
// stats-struct-derived accessors, or `claim!(verdict, expr)` for
// expression-labeled claims. Both produce
// [`ClaimBuilder`]/[`SetClaim`]/[`SeqClaim`] under the hood and
// record outcomes onto the same [`AssertResult`] envelope that
// `assert_not_starved` / `assert_isolation` produce, so the two
// paths compose via [`Verdict::merge`].)

#[cfg(test)]
mod tests_assert;
#[cfg(test)]
mod tests_benchmarks;
#[cfg(test)]
mod tests_common;
#[cfg(test)]
mod tests_merge;
#[cfg(test)]
mod tests_note;
#[cfg(test)]
mod tests_numa;
#[cfg(test)]
mod tests_percentile;
#[cfg(test)]
mod tests_phase_bucket;
#[cfg(test)]
mod tests_plan;
#[cfg(test)]
mod tests_sched_died;
#[cfg(test)]
mod tests_serde;
#[cfg(test)]
mod tests_stats;
#[cfg(test)]
mod tests_verdict;
#[cfg(test)]
mod tests_worker;
