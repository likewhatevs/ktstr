//! Shared classification of test-body host-insufficiency errors.
//!
//! Single source of truth for the guard ORDER and the per-class
//! skip/fail policy applied to the typed host-resource errors a test
//! build/run can surface. Both consumers call [`classify_host_error`]
//! and only choose the rendering:
//! - `err_to_exit_code` (in `super::dispatch`) maps [`HostClass`] to a
//!   process exit code (skip → `EXIT_PASS`, fail → `EXIT_FAIL`).
//! - the `#[ktstr_test]` macro body maps it to libtest control flow
//!   (skip → `eprintln!` + `return`, fail → `panic!`).
//!
//! Keeping the classification here — not duplicated in each consumer —
//! means a reorder or a new host-class type is a one-function edit both
//! sites inherit, eliminating the dispatch-vs-codegen guard-order
//! divergence (the two sites previously ordered the same guards
//! differently, correct only by the types' mutual exclusivity).
//!
//! Scope: the SIX host-insufficiency types BOTH sites classify —
//! [`KernelUnavailable`] (no kernel image resolved — the harness cannot
//! boot a VM here), [`PerfModeUnavailable`], [`CpuBudgetUnsatisfiable`],
//! [`TopologyUnrepresentable`], [`ResourceContention`], and
//! [`TopologyInsufficient`]. A `KernelUnavailable` reaches this classifier
//! on every nextest invocation: nextest suppresses the plain `#[test]`
//! wrapper, so the entry runs as `ktstr/{name}` via the `--exact` dispatch
//! → `run_named_test` → `err_to_exit_code`, NOT the macro body. It is a SKIP
//! by default — a developer running `cargo nextest run`, or `cargo ktstr
//! test` without `--kernel`, on a kernel-less host gets a clean skip rather
//! than a hard fail on every entry — promoted to a FAIL under
//! `KTSTR_NO_SKIP_MODE`. This cannot mask a CI kernel-build failure: a
//! `--kernel` the orchestrator FAILS to build bails in cargo-ktstr
//! (`resolve_kernel_set`) before nextest is spawned, so `KernelUnavailable`
//! here only ever means "no kernel was requested", never "a requested
//! kernel failed to build".
//!
//! [`PerfModeUnavailable`]: crate::vmm::host_topology::PerfModeUnavailable
//! [`CpuBudgetUnsatisfiable`]: crate::vmm::host_topology::CpuBudgetUnsatisfiable
//! [`TopologyUnrepresentable`]: crate::vmm::host_topology::TopologyUnrepresentable
//! [`ResourceContention`]: crate::vmm::host_topology::ResourceContention
//! [`TopologyInsufficient`]: crate::vmm::host_topology::TopologyInsufficient
//! [`KernelUnavailable`]: crate::test_support::eval::KernelUnavailable

use super::{
    is_cpu_budget_unsatisfiable, is_kernel_unavailable, is_perf_mode_unavailable,
    is_resource_contention, is_topology_insufficient, is_topology_unrepresentable,
};
use crate::test_support::eval::KernelUnavailable;
use crate::vmm::host_topology::{
    CpuBudgetUnsatisfiable, PerfModeUnavailable, ResourceContention, TopologyInsufficient,
    TopologyUnrepresentable,
};

/// Outcome of classifying a test-body error against the
/// host-insufficiency taxonomy.
///
/// The `reason` strings are BARE — they carry NO `ktstr: SKIP:` /
/// `ktstr: FAIL:` prefix. Each consumer adds the prefix in its own
/// channel: dispatch routes [`Skip`](HostClass::Skip) through
/// `report::test_skip` (which prepends `ktstr: SKIP:`) and `eprintln!`s
/// [`Fail`](HostClass::Fail) as `ktstr: FAIL: {reason}`; the macro
/// `eprintln!`s the skip and `panic!`s the fail with the same two
/// prefixes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HostClass {
    /// Not one of the six host-insufficiency types — the consumer
    /// applies its own per-site handling (dispatch: the
    /// `PostVmAssertionFailure` / `ExpectAutoReproSatisfied` /
    /// `expect_err` / catch-all arms; macro: the `expect_err` swallow or
    /// the `expect_ok` panic).
    NotHostClass,
    /// The host cannot run the test and no retry changes that
    /// (`KTSTR_NO_SKIP_MODE` unset). A visible, non-failing skip.
    Skip { reason: String },
    /// A hard failure: an unconditional hard-fail type
    /// (`CpuBudgetUnsatisfiable` / `TopologyUnrepresentable`) OR a
    /// skip-class type promoted to a failure under `KTSTR_NO_SKIP_MODE`.
    Fail { reason: String },
}

/// Walk the error chain for a `T` cause and clone its reason string.
///
/// Chain-aware (mirrors the `is_*` predicates): a typed error wrapped in
/// `.context(...)` (e.g. the eval-layer `"build ktstr_test VM"` /
/// `"run ktstr_test VM"` wrappers) is still found. Falls back to
/// `"<unknown>"` if the cause is somehow absent — only reachable if an
/// `is_*` predicate matched a `T` this extractor then missed, which the
/// shared chain walk makes impossible in practice.
fn extract_reason<T, F>(e: &anyhow::Error, reason: F) -> String
where
    T: std::error::Error + Send + Sync + 'static,
    F: Fn(&T) -> String,
{
    e.chain()
        .find_map(|cause| cause.downcast_ref::<T>().map(&reason))
        .unwrap_or_else(|| "<unknown>".to_string())
}

/// Classify a test-body error against the host-insufficiency taxonomy.
///
/// `no_skip` is `KTSTR_NO_SKIP_MODE` — passed in (not read from the
/// environment here) so the function stays pure and unit-testable
/// without env mutation. Each caller reads the env once
/// (`result_to_exit_code` for dispatch; the generated body for the
/// macro) and threads it in.
///
/// The guard ORDER and the per-class skip/fail policy below are the
/// single source of truth shared by both consumers. `expect_err` is
/// deliberately NOT a parameter: a host-class outcome is invariant under
/// it (a skip stays a skip, a hard fail stays a hard fail) — `expect_err`
/// is a test-outcome concern each consumer handles after a
/// [`HostClass::NotHostClass`] result. The `reason` strings reconstruct
/// the exact banners the two sites emitted before this was extracted
/// (minus the prefix, which the consumer adds).
pub fn classify_host_error(e: &anyhow::Error, no_skip: bool) -> HostClass {
    if is_kernel_unavailable(e) {
        // No kernel image resolved: the harness cannot boot a VM here (the
        // binary was run outside `cargo ktstr test`, or `cargo ktstr test`
        // was run without `--kernel` on a host with no cached/discoverable
        // kernel). A skip by default — a missing kernel on the runner is a
        // "not configured here" condition, not a test failure — promoted to
        // a FAIL under KTSTR_NO_SKIP_MODE for runs that demand execution. A
        // requested-but-unbuildable `--kernel` bails in cargo-ktstr before
        // nextest spawns, so this never masks a CI kernel-build failure.
        let reason = extract_reason::<KernelUnavailable, _>(e, |k| k.diagnostic.clone());
        return if no_skip {
            HostClass::Fail {
                reason: format!(
                    "harness not configured under --no-skip-mode: {reason}. \
                     Provide a kernel via --kernel or KTSTR_TEST_KERNEL, or drop \
                     --no-skip-mode."
                ),
            }
        } else {
            HostClass::Skip {
                reason: format!("harness not configured: {reason}"),
            }
        };
    }
    if is_perf_mode_unavailable(e) {
        let reason = extract_reason::<PerfModeUnavailable, _>(e, |p| p.reason.clone());
        return if no_skip {
            HostClass::Fail {
                reason: format!(
                    "performance mode unavailable under --no-skip-mode: {reason}. \
                     Provision a host with the required CPU / LLC count, narrow the \
                     test topology, or drop --perf-mode / --no-skip-mode."
                ),
            }
        } else {
            HostClass::Skip {
                reason: format!("performance mode unavailable: {reason}"),
            }
        };
    }
    if is_cpu_budget_unsatisfiable(e) {
        let reason = extract_reason::<CpuBudgetUnsatisfiable, _>(e, |b| b.reason.clone());
        return HostClass::Fail {
            reason: format!("cpu budget unsatisfiable: {reason}"),
        };
    }
    if is_topology_unrepresentable(e) {
        let reason = extract_reason::<TopologyUnrepresentable, _>(e, |t| t.reason.clone());
        return HostClass::Fail {
            reason: format!("topology unrepresentable: {reason}"),
        };
    }
    if is_resource_contention(e) {
        let reason = extract_reason::<ResourceContention, _>(e, |rc| rc.reason.clone());
        return if no_skip {
            HostClass::Fail {
                reason: format!(
                    "resource contention under --no-skip-mode: {reason}. \
                     Either provision hardware that satisfies the test's topology \
                     requirement, or drop --no-skip-mode / KTSTR_NO_SKIP_MODE to \
                     accept the skip."
                ),
            }
        } else {
            HostClass::Skip {
                reason: format!("resource contention: {reason}"),
            }
        };
    }
    if is_topology_insufficient(e) {
        let reason = extract_reason::<TopologyInsufficient, _>(e, |ti| ti.reason.clone());
        return if no_skip {
            HostClass::Fail {
                reason: format!(
                    "host topology insufficient under --no-skip-mode: {reason}. \
                     Either provision a host with the required CPU / LLC count, or drop \
                     --no-skip-mode / KTSTR_NO_SKIP_MODE to accept the skip."
                ),
            }
        } else {
            HostClass::Skip {
                reason: format!("host topology insufficient: {reason}"),
            }
        };
    }
    HostClass::NotHostClass
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A no-kernel host (KernelUnavailable) skips by default — a missing
    /// kernel on the runner is "not configured here", not a test failure —
    /// and is promoted to a hard fail under `no_skip`. The bare reason is
    /// the extracted diagnostic.
    #[test]
    fn kernel_unavailable_skip_then_fail() {
        let mk = || {
            anyhow::Error::new(KernelUnavailable {
                diagnostic: "no kernel image resolved".into(),
            })
        };
        match classify_host_error(&mk(), false) {
            HostClass::Skip { reason } => {
                assert_eq!(reason, "harness not configured: no kernel image resolved");
            }
            other => panic!("expected Skip, got {other:?}"),
        }
        match classify_host_error(&mk(), true) {
            HostClass::Fail { reason } => {
                assert!(reason.starts_with("harness not configured under --no-skip-mode:"));
                assert!(reason.contains("no kernel image resolved"));
            }
            other => panic!("expected Fail, got {other:?}"),
        }
    }

    /// A perf-mode-too-small error skips by default and is promoted to a
    /// hard fail only under `no_skip`. The reason text is the bare,
    /// prefix-free form each consumer renders.
    #[test]
    fn perf_mode_unavailable_skip_then_fail() {
        let mk = || {
            anyhow::Error::new(PerfModeUnavailable {
                reason: "host too small for perf topology".into(),
            })
        };
        match classify_host_error(&mk(), false) {
            HostClass::Skip { reason } => {
                assert_eq!(
                    reason,
                    "performance mode unavailable: host too small for perf topology"
                );
            }
            other => panic!("expected Skip, got {other:?}"),
        }
        match classify_host_error(&mk(), true) {
            HostClass::Fail { reason } => {
                assert!(reason.starts_with("performance mode unavailable under --no-skip-mode:"));
                assert!(reason.contains("host too small for perf topology"));
            }
            other => panic!("expected Fail, got {other:?}"),
        }
    }

    /// Resource contention: skip default, fail under `no_skip`.
    #[test]
    fn resource_contention_skip_then_fail() {
        let mk = || {
            anyhow::Error::new(ResourceContention {
                reason: "all 3 LLC slots busy".into(),
            })
        };
        assert_eq!(
            classify_host_error(&mk(), false),
            HostClass::Skip {
                reason: "resource contention: all 3 LLC slots busy".into()
            }
        );
        match classify_host_error(&mk(), true) {
            HostClass::Fail { reason } => {
                assert!(reason.starts_with("resource contention under --no-skip-mode:"));
            }
            other => panic!("expected Fail, got {other:?}"),
        }
    }

    /// Topology insufficient: skip default, fail under `no_skip`.
    #[test]
    fn topology_insufficient_skip_then_fail() {
        let mk = || {
            anyhow::Error::new(TopologyInsufficient {
                reason: "host has too few CPUs".into(),
            })
        };
        assert_eq!(
            classify_host_error(&mk(), false),
            HostClass::Skip {
                reason: "host topology insufficient: host has too few CPUs".into()
            }
        );
        match classify_host_error(&mk(), true) {
            HostClass::Fail { reason } => {
                assert!(reason.starts_with("host topology insufficient under --no-skip-mode:"));
            }
            other => panic!("expected Fail, got {other:?}"),
        }
    }

    /// Cpu-budget-unsatisfiable is an UNCONDITIONAL hard fail — `no_skip`
    /// does not change it (it is already a failure).
    #[test]
    fn cpu_budget_unsatisfiable_always_fails() {
        let mk = || {
            anyhow::Error::new(CpuBudgetUnsatisfiable {
                reason: "--cpu-cap exceeds allowed CPUs".into(),
            })
        };
        for no_skip in [false, true] {
            match classify_host_error(&mk(), no_skip) {
                HostClass::Fail { reason } => {
                    assert_eq!(
                        reason,
                        "cpu budget unsatisfiable: --cpu-cap exceeds allowed CPUs"
                    );
                }
                other => panic!("expected Fail (no_skip={no_skip}), got {other:?}"),
            }
        }
    }

    /// Topology-unrepresentable is an UNCONDITIONAL hard fail.
    #[test]
    fn topology_unrepresentable_always_fails() {
        let mk = || {
            anyhow::Error::new(TopologyUnrepresentable {
                reason: "aarch64 vcpus exceed GICv3 redistributor capacity".into(),
            })
        };
        for no_skip in [false, true] {
            match classify_host_error(&mk(), no_skip) {
                HostClass::Fail { reason } => {
                    assert!(reason.starts_with("topology unrepresentable:"));
                }
                other => panic!("expected Fail (no_skip={no_skip}), got {other:?}"),
            }
        }
    }

    /// A plain (non-typed) error and the test-outcome markers are NOT
    /// host-class — the classifier returns `NotHostClass` so each
    /// consumer's own marker / expect_err / catch-all handling runs. A
    /// classifier that swallowed these would erase real failures.
    #[test]
    fn non_host_error_is_not_host_class() {
        let plain = anyhow::anyhow!("scheduler regression: workload did not get the CPU it needs");
        assert_eq!(classify_host_error(&plain, false), HostClass::NotHostClass);
        assert_eq!(classify_host_error(&plain, true), HostClass::NotHostClass);
    }

    /// Chain-aware: a typed error wrapped in `.context(...)` (the
    /// production shape — the eval layer wraps every build/run error)
    /// still classifies, and the extracted reason is the inner typed
    /// reason, NOT the wrapping context layer.
    #[test]
    fn classifies_through_context_wrap() {
        let wrapped = anyhow::Error::new(ResourceContention {
            reason: "all 3 LLC slots busy".into(),
        })
        .context("build ktstr_test VM")
        .context("run ktstr_test VM");
        assert_eq!(
            classify_host_error(&wrapped, false),
            HostClass::Skip {
                reason: "resource contention: all 3 LLC slots busy".into()
            }
        );
    }
}
