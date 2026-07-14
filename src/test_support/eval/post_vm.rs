//! Host-side post_vm plumbing: the post_vm error marker types
//! (ScxBpfErrorMatcherMismatch,
//! PostVmAssertionFailure, HostSkipRequest, ExpectAutoReproSatisfied),
//! the conditional/unconditional callback combiner + dispatch, the
//! post_vm_skip helper, and skip-sidecar recording. Split out of
//! eval/mod.rs to keep the module under the size ceiling.

use super::*;

/// Marker error type attached as `anyhow::Context` to the failure
/// `Err` produced when an scx_bpf_error matcher
/// ([`crate::assert::Assert::expect_scx_bpf_error_contains`] or
/// [`crate::assert::Assert::expect_scx_bpf_error_matches`]) mismatched
/// the captured scheduler log / sched_ext dump corpus.
///
/// Dispatch (`crate::test_support::dispatch::result_to_exit_code`)
/// downcasts the error chain for this marker in the `expect_err = true`
/// branch and refuses to invert the verdict to a pass — a reproducer
/// that fired the WRONG bug must fail loudly, not silently invert to
/// "test passed" via `expect_err`. Without the marker, the matcher's
/// diagnostic surfaces in stderr but the exit code follows the normal
/// expect_err inversion path.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ScxBpfErrorMatcherMismatch;

impl std::fmt::Display for ScxBpfErrorMatcherMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "scx_bpf_error matcher mismatch — the reproducer matcher rejected \
             this failure mode; expect_err inversion bypassed"
        )
    }
}

impl std::error::Error for ScxBpfErrorMatcherMismatch {}

/// Marker error type attached as `anyhow::Context` to the `Err`
/// `resolve_scheduler` returns when the `Discover` `cargo build -p
/// <sched>` FAILED. A scheduler that cannot be built is a failed test —
/// the resolver never falls back to a possibly-stale pre-built binary.
///
/// Dispatch (`crate::test_support::dispatch`) downcasts the error chain
/// for this marker and forces a hard FAIL EVEN under `expect_err = true`.
/// The semantic boundary mirrors [`PostVmAssertionFailure`]: `expect_err`
/// inverts a GUEST-side expected failure, but a build-infra failure is a
/// HOST-side fault that must never masquerade as the expected guest
/// failure — without the marker an `expect_err` test whose scheduler
/// build broke would silently invert to PASS. Same
/// `anyhow::Context` attachment + `downcast_ref` chain-walk as the
/// sibling markers; the dispatch guard sits with the other host-side
/// hard-fail markers, before the `expect_err` inversion.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SchedulerBuildRefused;

impl std::fmt::Display for SchedulerBuildRefused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "scheduler build refused — the workspace build failed; a \
             scheduler that cannot be built is a failed test \
             (expect_err inversion bypassed)"
        )
    }
}

impl std::error::Error for SchedulerBuildRefused {}

/// Marker error type attached as `anyhow::Context` to the failure
/// `Err` produced by `run_ktstr_test_inner_impl` when a host-side
/// `post_vm` / `post_vm_unconditional` callback returned `Err`
/// (which `evaluate_vm_result` has already folded into the verdict —
/// as an `Other` detail in the parse-success arm, as a message prefix
/// in the parse-fail arms).
///
/// Dispatch (`crate::test_support::dispatch::result_to_exit_code`)
/// downcasts the error chain for this marker and refuses to invert the
/// verdict to a pass — even under `expect_err = true`. The semantic
/// boundary: `expect_err` inverts a GUEST-side expected failure (the
/// scheduler stalled, the workload bailed), but a HOST-side `post_vm`
/// assertion is always honored. A failure-dump render test that
/// triggers an expected stall to PRODUCE the dump, then asserts the
/// dump's contents in `post_vm`, must fail loudly when the dump renders
/// wrong — not silently invert to "passed" because the stall it relied
/// on was "expected". Without the marker, the post_vm diagnostic
/// surfaces in stderr but the exit code follows the normal expect_err
/// inversion path (a false PASS).
///
/// Mirrors [`ScxBpfErrorMatcherMismatch`]: same `anyhow::Context`
/// attachment, same `downcast_ref` chain-walk at the dispatch arm. The
/// dispatch arm is positioned AFTER the resource-contention / topology
/// skip arms (a skip means the test never ran) but BEFORE the
/// [`ExpectAutoReproSatisfied`] and `expect_err` inversion arms, so a
/// real host-side regression wins over any inversion.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PostVmAssertionFailure;

impl std::fmt::Display for PostVmAssertionFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "host-side post_vm assertion failed — expect_err inversion bypassed \
             (a host-side check is honored even when the accompanying guest-side \
             failure is expected)"
        )
    }
}

impl std::error::Error for PostVmAssertionFailure {}

/// Marker error type attached as `anyhow::Context` to a `post_vm` /
/// `post_vm_unconditional` `Err` to request a test SKIP (not a
/// failure): the host-side callback determined the run is
/// INCONCLUSIVE — the VM could not produce the artifact the assertion
/// needs (e.g. a load-starved VM whose BPF probe never attached, so
/// the failure dump is a placeholder), as opposed to a real
/// regression. The eval fn detects this marker (context-aware
/// `downcast_ref`, at the `HostSkipRequest` gate) and returns
/// [`crate::assert::AssertResult::skip`] instead of folding the `Err`
/// into the verdict.
///
/// A real [`PostVmAssertionFailure`] in a sibling callback DOMINATES:
/// [`combine_post_vm_errs`] preserves the skip marker only when BOTH
/// callbacks request skip (or only one callback ran); a genuine
/// failure alongside a skip request collapses to a failure, so a skip
/// request can never mask a regression.
#[derive(Debug, Clone, Copy)]
pub(crate) struct HostSkipRequest;

impl std::fmt::Display for HostSkipRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "host-side post_vm requested skip — the run is inconclusive \
             (the VM could not produce the artifact the assertion needs)"
        )
    }
}

impl std::error::Error for HostSkipRequest {}

/// Marker error type attached as `anyhow::Context` to the failure
/// `Err` produced by `evaluate_vm_result` when
/// [`apply_expect_auto_repro_inversion`] has set
/// `result.expect_auto_repro_satisfied = true`: the primary VM
/// produced a Fail AND a shape-valid `.repro.wprof.pb` artifact
/// landed on disk from the auto-repro VM.
///
/// Dispatch (`crate::test_support::dispatch::result_to_exit_code`)
/// downcasts the error chain for this marker and routes the verdict
/// to `EXIT_PASS`. The underlying `AssertResult` is NOT mutated —
/// the original failure detail still surfaces in stderr/dump
/// rendering so an operator chasing why `expect_auto_repro` fired
/// sees the original failure trail alongside the inversion notice.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ExpectAutoReproSatisfied;

impl std::fmt::Display for ExpectAutoReproSatisfied {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "expect_auto_repro satisfied — the primary test failed and the \
             auto-repro VM produced a shape-valid .repro.wprof.pb artifact; \
             verdict inverted to PASS"
        )
    }
}

impl std::error::Error for ExpectAutoReproSatisfied {}

/// Marker error type attached as `anyhow::Context` to the failure `Err`
/// [`render_failure_verdict_message`]
/// builds when `entry.survives_storm` is set AND the failing
/// [`AssertResult`] carries a scheduler-death
/// `DetailKind` (`SchedulerCrashed` / `SchedulerExitedCleanly` /
/// `SchedulerDiedUnknownReason`). `err_to_exit_code` downcasts it and forces
/// `EXIT_FAIL` with a survival-specific explainer, positioned BEFORE the
/// [`ExpectAutoReproSatisfied`] / `expect_err` inversion arms so a survival
/// violation can never be inverted to PASS (the validate-time
/// `survives_storm`/`expect_err` mutex already forbids that combination;
/// the ordering is defense-in-depth). Mirrors [`ScxBpfErrorMatcherMismatch`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct SurvivesStormViolated;

impl std::fmt::Display for SurvivesStormViolated {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "survives_storm asserted but the scx scheduler did not survive \
             the run — it died or was ejected during a hold"
        )
    }
}

impl std::error::Error for SurvivesStormViolated {}

/// Combine the conditional and unconditional `post_vm` failure
/// signals. When both callbacks fail in the same run, surface
/// BOTH errors in a single chained message so a debugging
/// operator sees both regressions on the first pass — a `.or()`
/// would silently drop the unconditional signal whenever the
/// conditional also fired, defeating the whole point of the
/// unconditional callback.
pub(crate) fn combine_post_vm_errs(
    conditional: Option<anyhow::Error>,
    unconditional: Option<anyhow::Error>,
) -> Option<anyhow::Error> {
    match (conditional, unconditional) {
        (Some(c), Some(u)) => {
            // A genuine failure dominates a skip request: collapse to a
            // skip only when BOTH callbacks requested skip (both
            // inconclusive). Otherwise a real PostVmAssertionFailure
            // must surface, so the chained message wins and the
            // HostSkipRequest marker is intentionally dropped.
            let both_skip = c.downcast_ref::<HostSkipRequest>().is_some()
                && u.downcast_ref::<HostSkipRequest>().is_some();
            let combined = anyhow::anyhow!("post_vm: {c:#}; post_vm_unconditional: {u:#}");
            Some(if both_skip {
                combined.context(HostSkipRequest)
            } else {
                combined
            })
        }
        (Some(c), None) => Some(c),
        (None, Some(u)) => Some(u),
        (None, None) => None,
    }
}

/// Request a test SKIP from a `post_vm` / `post_vm_unconditional`
/// callback: `return Err(post_vm_skip(reason))` when the run is
/// INCONCLUSIVE — the VM could not produce the artifact the assertion
/// needs (e.g. a load-starved VM whose BPF probe never attached,
/// leaving a placeholder failure dump), as distinct from a real
/// regression. The framework detects the attached `HostSkipRequest`
/// marker and converts the run to
/// [`crate::assert::AssertResult::skip`] instead of a failure.
///
/// A genuine `Err` from a sibling callback dominates (see
/// `combine_post_vm_errs`): a skip request never masks a regression.
pub fn post_vm_skip(reason: impl Into<String>) -> anyhow::Error {
    anyhow::anyhow!("{}", reason.into()).context(HostSkipRequest)
}

/// Witnessed-starvation evidence for capture-bearing assertions: the
/// host dilation `D` that PROVES the run was contended enough to
/// explain missing/degraded periodic captures, or `None` on a quiet
/// (or unwitnessed) host.
///
/// The periodic capture chain is readiness-gated (KASLR publish +
/// accessor-init worker) and its boundaries are wall-anchored inside
/// the workload window; under heavy host saturation the cold-cache
/// accessor build alone can outlast a short workload, so every
/// boundary defers past run end and `periodic_fired == 0` with zero
/// real captures — an ENVIRONMENTAL outcome, not a capture-pipeline
/// regression (observed across the periodic/freeze test family on
/// saturated 96-CPU CI runners; the same cells pass idle). Assertions
/// that require captures consult this first and convert their failure
/// into a host-side SKIP (`post_vm_skip`) when starvation is proven,
/// mirroring the tri-state latency gates: a quiet-host miss still
/// FAILS (the regression the assertion exists to catch).
///
/// Witness preference: Body-phase dilation (the capture boundaries
/// live inside the workload window), falling back to the whole-run
/// schedstat `D`. The 1.5 bar follows `PERF_ISOLATION_D_MAX`'s
/// derivation (worst measured-quiet `D` ≈ 1.21; > 1.5 is unambiguous
/// external contention). `None` (no witness sampled) is NOT proof —
/// callers must fail in that case; absence of evidence cannot launder
/// a regression.
pub fn capture_starvation_witness(result: &crate::vmm::VmResult) -> Option<f64> {
    const CAPTURE_STARVATION_D_MAX: f64 = 1.5;
    let d = result
        .contention_witness
        .as_ref()
        .and_then(|w| w.body_dilation())
        .or_else(|| {
            result
                .host_vcpu_schedstat
                .as_ref()
                .and_then(|s| s.dilation())
        })?;
    (d > CAPTURE_STARVATION_D_MAX).then_some(d)
}

/// Environmental skip for a zero-scheduler-activity reading (e.g.
/// `nr_dispatched` / `nr_yielded` read 0 across every periodic sample)
/// that the guest's OWN sched_ext runnable-stall watchdog explains: when
/// the console carries a watchdog [`ScxExitKind::Stall`] exit AND
/// [`capture_starvation_witness`] proves the host was contended, the BPF
/// scheduler was ejected because a descheduling host dilated the 5s
/// GUEST-time stall window past the timeout — not because the scheduler
/// failed to dispatch. Returns `Err(post_vm_skip(..))` in that case; `Ok`
/// otherwise, so a Stall on a quiet host (no witness) or any non-Stall
/// zero still reaches the caller's own regression `ensure!`.
///
/// Callers gate their zero-activity failure on it:
/// ```ignore
/// if !any_progress {
///     stall_ejection_skip(result)?;
/// }
/// anyhow::ensure!(any_progress, "... never dispatched ...");
/// ```
pub fn stall_ejection_skip(result: &crate::vmm::VmResult) -> anyhow::Result<()> {
    let stalled = crate::monitor::dmesg_scx::parse_kmsg_window(&result.stderr)
        .iter()
        .any(|e| e.kind == crate::monitor::dmesg_scx::ScxExitKind::Stall);
    if stalled && let Some(d) = capture_starvation_witness(result) {
        return Err(post_vm_skip(format!(
            "scx scheduler ejected by the guest sched_ext runnable-stall \
             watchdog under witnessed host contention (D={d:.2}): the 5s stall \
             timer measures GUEST time, which a descheduling host dilates into \
             a false stall — the zero-activity reading is that ejection, not a \
             scheduler regression; environmental non-verdict",
        )));
    }
    Ok(())
}

/// Generic sub-minimum starvation skip: `Err(post_vm_skip(..))` when the
/// run produced FEWER data-bearing units than the caller's assertion
/// requires (`have < need`) AND [`capture_starvation_witness`] proves the
/// host was contended enough to explain the shortfall; `Ok(())`
/// otherwise. The caller passes the SAME count its own assertion checks
/// (real captures, PSI-carrying captures, series length, …) so the gate
/// and the assertion cannot disagree about the boundary — an earlier
/// zero-only gate let `1 of 6` runs through to fail on a `>= 2` minimum
/// under the identical environmental condition. Quiet-host shortfalls
/// (zero OR sub-minimum) still fall through and fail with the caller's
/// specific diagnosis.
pub fn starved_below_minimum_skip(
    result: &crate::vmm::VmResult,
    have: usize,
    need: usize,
    what: &str,
) -> anyhow::Result<()> {
    if have >= need {
        return Ok(());
    }
    // PRIMARY arm — readiness vs window: when the periodic prereqs
    // (KASLR publish + accessors) became ready at/after the capture
    // window's end, or never became ready at all, every boundary was
    // structurally unreachable — zero/short captures were inevitable at
    // ANY host dilation. This is the honest signal the D-threshold arm
    // missed in the field: a ~7 s cold accessor build outruns a 4-5 s
    // workload window at D ≈ 1.2, well under any defensible contention
    // bar. Skip carries both timestamps.
    let ready = result.periodic_prereqs_ready;
    let window_end = result.periodic_window_end;
    let structurally_impossible = match (ready, window_end) {
        // Never ready: nothing could ever fire.
        (None, _) => true,
        // Ready, but at/after the window closed.
        (Some(r), Some(w)) => r >= w,
        // Ready but the window never resolved: the boundaries were
        // never computed, so no capture could fire.
        (Some(_), None) => true,
    };
    if result.periodic_target > 0 && structurally_impossible {
        return Err(post_vm_skip(format!(
            "only {have} of the required {need} {what}: the capture prereqs \
             (KASLR publish + accessor init) {} the capture window \
             (prereqs_ready={:?}, window_end={:?}) — captures were \
             structurally impossible this run at any host dilation; \
             environmental non-verdict",
            if ready.is_none() {
                "never became ready within"
            } else {
                "became ready only after"
            },
            ready,
            window_end,
        )));
    }
    // SECONDARY arm — witnessed contention: readiness landed in time
    // but the captures still fell short (e.g. rendezvous degradation,
    // data-bearing rows missing) under a provably saturated host.
    if let Some(d) = capture_starvation_witness(result) {
        return Err(post_vm_skip(format!(
            "only {have} of the required {need} {what} under witnessed host \
             contention (D={d:.2}, prereqs_ready={ready:?}, \
             window_end={window_end:?}): the capture chain was starved below \
             the assertion's minimum — environmental non-verdict; the \
             dependent assertions cannot be evaluated",
        )));
    }
    Ok(())
}

/// Periodic-capture starvation gate for capture-requiring post_vm
/// assertions: [`starved_below_minimum_skip`] keyed on the bridge's
/// REAL (non-placeholder) capture count vs the test's stated minimum.
/// Call at the TOP of a post_vm callback with the same minimum its
/// assertions enforce: `periodic_starvation_gate(result, 2)?;`. Tests
/// whose assertions key on a narrower data-bearing count (e.g. "N
/// captures carrying per-cgroup PSI") should ALSO call
/// [`starved_below_minimum_skip`] with that count at the assertion
/// site. No-op when the entry configured no periodic captures.
pub fn periodic_starvation_gate(
    result: &crate::vmm::VmResult,
    min_real: usize,
) -> anyhow::Result<()> {
    if result.periodic_target == 0 {
        return Ok(());
    }
    starved_below_minimum_skip(
        result,
        result.snapshot_bridge.periodic_real_count() as usize,
        min_real,
        "real (non-placeholder) periodic captures",
    )
}

/// Dispatch the entry's `post_vm` + `post_vm_unconditional`
/// callbacks and combine their failure signals.
///
/// - `post_vm` runs only when the guest reported a non-Fail
///   `AssertResult` (Skip / Inconclusive / Pass) — the
///   `guest_already_failed` parameter folds the
///   `parse_assert_result_from_drain` lookup the call site does.
///   The skip mirrors the suppression contract documented on
///   `KtstrTestEntry::post_vm`.
///
/// - `post_vm_unconditional` ALWAYS runs — bypasses the
///   guest-fail suppression that gates `post_vm`. The callback
///   owns its own skip-on-crash logic (or doesn't, when the
///   intent is "assert on host-side artifact regardless of
///   guest-side outcome").
///
/// Both callbacks route through [`invoke_post_vm_callback`] so a
/// panic in either body becomes an `anyhow::Error` rather than
/// unwinding past the call site (which would leak VM resources;
/// see the helper doc).
///
/// Returns the combined `Option<anyhow::Error>` via
/// [`combine_post_vm_errs`]: when both callbacks fail, the
/// chained message names both errors so the operator sees both
/// regressions on the first pass instead of a two-pass debug
/// cycle. `.or()` would silently drop the unconditional fail
/// when the conditional also fired.
pub(crate) fn run_post_vm_callbacks(
    entry: &KtstrTestEntry,
    result: &crate::vmm::VmResult,
    guest_already_failed: bool,
) -> Option<anyhow::Error> {
    let conditional = if guest_already_failed {
        None
    } else {
        entry
            .post_vm
            .and_then(|cb| invoke_post_vm_callback(cb, result, "post_vm"))
    };
    let unconditional = entry
        .post_vm_unconditional
        .and_then(|cb| invoke_post_vm_callback(cb, result, "post_vm_unconditional"));
    combine_post_vm_errs(conditional, unconditional)
}

/// Invoke a `post_vm` / `post_vm_unconditional` callback with panic
/// catch. Converts a panic to `anyhow::Error` so the panic message
/// surfaces in the test failure output AND the rest of the
/// post-VM teardown (`write_placeholder_failure_dump_if_missing`,
/// `drop(vm)` releasing CPU/LLC flocks + guest memory + kernel-cache
/// reader flock) still runs.
///
/// Without the catch, a panicking callback would unwind past the
/// placeholder-dump emission and past `drop(vm)`, leaking VM
/// resources (flocks, guest memory) until process exit or the next
/// test's drop reclaims them. Same hazard for `Ok` returns from
/// callbacks that subsequently panic in their inner state — both
/// paths fold into this single guard.
///
/// `label` is woven into the error message so the operator sees
/// which callback panicked (`post_vm` vs `post_vm_unconditional`)
/// when both are wired and both fire.
///
/// Returns `Some(err)` when the callback returns `Err` OR panics;
/// returns `None` when the callback returns `Ok(())`. Mirrors the
/// shape `.err()` produces from `Result` so the caller's
/// `.and_then(|cb| ...)` flows unchanged.
///
/// Under `panic = "abort"` (release builds — see `Cargo.toml
/// [profile.release]`), `catch_unwind` is a no-op: a panic aborts
/// the process before this function returns. The wrap is still
/// safe — `catch_unwind` is always defined, just inert — and the
/// debug builds get the leak protection that exposes regressions
/// before they ship.
pub(crate) fn invoke_post_vm_callback(
    cb: super::super::PostVmCallback,
    result: &crate::vmm::VmResult,
    label: &'static str,
) -> Option<anyhow::Error> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| cb(result))) {
        Ok(Ok(())) => None,
        Ok(Err(e)) => Some(e),
        Err(payload) => {
            let msg = if let Some(s) = payload.downcast_ref::<&'static str>() {
                (*s).to_string()
            } else if let Some(s) = payload.downcast_ref::<String>() {
                s.clone()
            } else {
                "<non-string panic payload>".to_string()
            };
            Some(anyhow::anyhow!("{label} callback panicked: {msg}"))
        }
    }
}

/// Write a skip sidecar for `entry`, logging to stderr on failure
/// without propagating the error. Called wherever a run is skipped
/// before producing a real result: the skip-class catch-all and the
/// VM build / VM run arms in [`run_ktstr_test_inner`] (each fires on a
/// host-incapacity class — `TopologyInsufficient` or
/// `PerfModeUnavailable`; transient `ResourceContention` is a
/// retryable failure and records no skip), and the
/// performance-mode / coverage gates at the plain-run entry points in
/// the crate `dispatch` module. All must record the skip for stats
/// tooling but cannot meaningfully handle a sidecar-write failure
/// beyond logging it — the skip itself is still valid; only post-run
/// stats tooling loses visibility.
pub(crate) fn record_skip_sidecar(
    entry: &KtstrTestEntry,
    topo: Option<&crate::test_support::topo::TopoOverride>,
) -> crate::vmm::topology::Topology {
    // Resolve the topology the run of this (entry, override) WOULD boot,
    // via the same resolve_vm_topology the run path uses, so a preset's
    // skip and run record the identical topology -> identical
    // variant_hash -> the retry overwrites instead of coexisting. For a
    // plain test (topo = None) this is entry.topology. Returned so a
    // host-class caller can write its `.host-skip.json` marker under the
    // same variant hash without re-resolving.
    let (resolved_topology, _memory_mib) =
        crate::test_support::runtime::resolve_vm_topology(entry, topo);
    if let Err(e) = write_skip_sidecar(entry, &resolved_topology) {
        // Dual-emit at warn level: an unwritten skip sidecar costs
        // the run no correctness — the test still skipped — but
        // silently drops post-run stats tooling's visibility into
        // the skip, so operators debugging a missing row in a
        // gauntlet report need a loud-enough log to notice. The
        // eprintln surfaces under direct nextest / cargo-ktstr
        // invocations where no tracing subscriber is installed;
        // the tracing::warn lands in every structured-log consumer
        // (cargo-ktstr, downstream pipelines) at warn level rather
        // than the previous implicit debug visibility.
        let entry_name = entry.name;
        let rendered = format!("{e:#}");
        eprintln!("ktstr_test: warn: skip-sidecar write failed for {entry_name}: {rendered}");
        tracing::warn!(
            test = %entry_name,
            err = %rendered,
            "skip-sidecar write failed — stats tooling will not see this skip",
        );
    }
    resolved_topology
}

#[cfg(test)]
mod post_vm_skip_tests {
    //! Locks in the post_vm→skip mechanism. `post_vm_skip` attaches the
    //! [`HostSkipRequest`] marker (found by the context-aware
    //! `downcast_ref` the eval gate uses); `combine_post_vm_errs`
    //! preserves a lone skip request but lets a genuine sibling failure
    //! DOMINATE — a skip request must never mask a real regression. A
    //! revert of either the marker attach or the both-skip gate flips a
    //! cell here.
    use super::{HostSkipRequest, PostVmAssertionFailure, combine_post_vm_errs, post_vm_skip};

    fn real_fail() -> anyhow::Error {
        anyhow::anyhow!("real host-side regression").context(PostVmAssertionFailure)
    }

    #[test]
    fn post_vm_skip_carries_marker() {
        assert!(
            post_vm_skip("inconclusive: placeholder dump")
                .downcast_ref::<HostSkipRequest>()
                .is_some()
        );
    }

    #[test]
    fn combine_lone_unconditional_skip_preserved() {
        let c = combine_post_vm_errs(None, Some(post_vm_skip("ph"))).unwrap();
        assert!(c.downcast_ref::<HostSkipRequest>().is_some());
    }

    #[test]
    fn combine_lone_conditional_skip_preserved() {
        let c = combine_post_vm_errs(Some(post_vm_skip("ph")), None).unwrap();
        assert!(c.downcast_ref::<HostSkipRequest>().is_some());
    }

    #[test]
    fn combine_both_skip_yields_skip() {
        let c = combine_post_vm_errs(Some(post_vm_skip("a")), Some(post_vm_skip("b"))).unwrap();
        assert!(c.downcast_ref::<HostSkipRequest>().is_some());
    }

    #[test]
    fn combine_skip_plus_real_fail_does_not_skip() {
        // A genuine failure alongside a skip request collapses to a
        // failure: the combined Err must NOT carry HostSkipRequest, so the
        // eval gate folds it as a failure (re-attaching PostVmAssertionFailure)
        // rather than skipping — a regression is never masked.
        let c = combine_post_vm_errs(Some(post_vm_skip("ph")), Some(real_fail())).unwrap();
        assert!(c.downcast_ref::<HostSkipRequest>().is_none());
    }

    #[test]
    fn combine_real_fail_plus_skip_does_not_skip() {
        let c = combine_post_vm_errs(Some(real_fail()), Some(post_vm_skip("ph"))).unwrap();
        assert!(c.downcast_ref::<HostSkipRequest>().is_none());
    }
}

#[cfg(test)]
mod starvation_witness_tests {
    use super::*;
    use crate::vmm::{HostVcpuSchedstat, VmResult};

    fn with_dilation(on_cpu_ns: u64, run_delay_ns: u64) -> VmResult {
        VmResult {
            host_vcpu_schedstat: Some(HostVcpuSchedstat {
                total_on_cpu_ns: on_cpu_ns,
                total_run_delay_ns: run_delay_ns,
                sampled_vcpus: 1,
            }),
            ..VmResult::test_fixture()
        }
    }

    /// No witness sampled → None (absence of evidence cannot launder a
    /// regression); quiet host (D ≈ 1.1) → None; contended (D = 3) →
    /// Some(3.0). Whole-run fallback path (fixture has no per-phase
    /// witness).
    #[test]
    fn capture_starvation_witness_thresholds() {
        assert_eq!(capture_starvation_witness(&VmResult::test_fixture()), None);
        assert_eq!(capture_starvation_witness(&with_dilation(10, 1)), None);
        let d = capture_starvation_witness(&with_dilation(10, 20)).expect("D=3 is contended");
        assert!((d - 3.0).abs() < 1e-9);
    }

    /// Fixture with the readiness-vs-window fields set: prereqs ready
    /// at `ready_s`, window ending at `end_s` (None = never/unresolved).
    fn with_window(
        base: VmResult,
        target: u32,
        ready_s: Option<u64>,
        end_s: Option<u64>,
    ) -> VmResult {
        VmResult {
            periodic_target: target,
            periodic_prereqs_ready: ready_s.map(std::time::Duration::from_secs),
            periodic_window_end: end_s.map(std::time::Duration::from_secs),
            ..base
        }
    }

    /// Gate arms, primary (readiness-vs-window) first:
    ///   - readiness never / after the window end → structural skip at
    ///     ANY dilation (the arm the D-threshold missed in the field:
    ///     a ~7 s accessor build vs a 4-5 s window at D ≈ 1.2);
    ///   - readiness in time + contended → witness skip;
    ///   - readiness in time + quiet → Ok (the caller's own assertion
    ///     fails — the real-regression case stays catchable);
    ///   - no periodic target → Ok regardless.
    #[test]
    fn periodic_starvation_gate_arms() {
        // No periodic target: Ok even when contended.
        let r = with_dilation(10, 20);
        assert!(periodic_starvation_gate(&r, 1).is_ok());
        // Readiness AFTER the window end, QUIET host: structural skip.
        let r = with_window(with_dilation(10, 1), 2, Some(9), Some(5));
        let err = periodic_starvation_gate(&r, 1).expect_err("late readiness must gate");
        assert!(err.downcast_ref::<HostSkipRequest>().is_some());
        // Readiness NEVER, quiet host: structural skip.
        let r = with_window(with_dilation(10, 1), 2, None, Some(5));
        assert!(periodic_starvation_gate(&r, 1).is_err());
        // Window never resolved (no boundary could fire): structural skip.
        let r = with_window(with_dilation(10, 1), 2, Some(1), None);
        assert!(periodic_starvation_gate(&r, 1).is_err());
        // Readiness IN TIME + contended: witness skip.
        let r = with_window(with_dilation(10, 20), 2, Some(1), Some(5));
        let err = periodic_starvation_gate(&r, 1).expect_err("contended run must gate");
        assert!(err.downcast_ref::<HostSkipRequest>().is_some());
        // Readiness IN TIME + quiet + zero captures: Ok — the caller
        // fails with its own diagnosis (a capture-pipeline regression).
        let r = with_window(with_dilation(10, 1), 2, Some(1), Some(5));
        assert!(periodic_starvation_gate(&r, 1).is_ok());
    }

    /// stall_ejection_skip three-way: a watchdog Stall exit on a
    /// CONTENDED host (D=3) is an environmental skip; the same Stall on a
    /// QUIET host stays the caller's failure (Ok, so its `ensure!` fires);
    /// a contended host with NO stall line is likewise the caller's to
    /// fail — absence of the watchdog signature cannot launder a real
    /// zero-activity regression.
    #[test]
    fn stall_ejection_skip_arms() {
        const STALL: &str = "sched_ext: BPF scheduler \"scx_test\" disabled (runnable task stall)";

        let contended_stall = VmResult {
            stderr: STALL.into(),
            ..with_dilation(10, 20)
        };
        let err = stall_ejection_skip(&contended_stall).expect_err("stall + contention must skip");
        assert!(err.downcast_ref::<HostSkipRequest>().is_some());

        let quiet_stall = VmResult {
            stderr: STALL.into(),
            ..with_dilation(10, 1)
        };
        assert!(
            stall_ejection_skip(&quiet_stall).is_ok(),
            "a stall on a quiet host is a real fail, not an environmental skip",
        );

        let contended_no_stall = with_dilation(10, 20);
        assert!(
            stall_ejection_skip(&contended_no_stall).is_ok(),
            "contention without a watchdog stall line leaves the caller to fail",
        );
    }

    /// The generic sub-minimum gate: `have < need` + (structural OR
    /// witnessed) = skip; `have >= need` never gates; readiness-in-time
    /// quiet host never gates.
    #[test]
    fn starved_below_minimum_arms() {
        // Sub-minimum + contended (readiness in time): witness skip —
        // the 1-of-6 shape.
        let contended = with_window(with_dilation(10, 20), 6, Some(1), Some(5));
        let err = starved_below_minimum_skip(&contended, 1, 3, "captures")
            .expect_err("sub-minimum under contention must gate");
        assert!(err.downcast_ref::<HostSkipRequest>().is_some());
        // At/above minimum: never gates, even contended + late-ready.
        let late = with_window(with_dilation(10, 20), 6, Some(9), Some(5));
        assert!(starved_below_minimum_skip(&late, 3, 3, "captures").is_ok());
        // Sub-minimum, quiet, readiness in time: falls through (caller
        // fails with its own diagnosis).
        let quiet = with_window(with_dilation(10, 1), 6, Some(1), Some(5));
        assert!(starved_below_minimum_skip(&quiet, 1, 3, "captures").is_ok());
        // Sub-minimum, quiet, readiness LATE: structural skip.
        let quiet_late = with_window(with_dilation(10, 1), 6, Some(9), Some(5));
        assert!(starved_below_minimum_skip(&quiet_late, 1, 3, "captures").is_err());
    }
}
