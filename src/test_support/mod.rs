//! Runtime support for `#[ktstr_test]` integration tests.
//!
//! Provides the registration type, distributed slice, VM launcher,
//! and result evaluation. Includes guest-side profraw flush for
//! coverage-instrumented builds.
//!
//! The entry point for test authors is the [`macro@crate::ktstr_test`]
//! attribute macro; see the user-facing Writing Tests guide shipped
//! with the crate's mdbook for end-to-end examples and the full
//! attribute grammar.
//!
//! # Consumer API
//!
//! Test authors interact primarily with the `#[ktstr_test]` proc
//! macro; programmatic test generation can instead populate
//! [`KtstrTestEntry`] values into the [`KTSTR_TESTS`]
//! `linkme` distributed slice. The remaining items in this module
//! are runtime glue invoked by the macro-generated code and the
//! `ktstr` / `cargo-ktstr` binaries.
//!
//! # Module layout
//!
//! Implementation is split across 19 production submodules
//! re-exported at `test_support::*` for a flat public API: `args`
//! (CLI argument extraction), `dispatch` (ktstr / cargo-ktstr CLI
//! entry points), `entry` (scheduler + test-entry types),
//! `entry_validate` (`KtstrTestEntry::validate` + phase helpers split
//! out of `entry.rs`), `eval`
//! (host-side VM result evaluation), `host_class` (shared
//! host-insufficiency error classification), `metrics` (payload stdout →
//! `Metric` list), `output`
//! (guest-output and console parsing), `payload` (`Payload` /
//! `MetricCheck` / `Metric` / `Polarity`), `probe` (auto-repro and
//! BPF probe pipeline), `probe_metrics` (host-side BPF map
//! introspection), `profraw` (coverage flush), `runtime` (`pub mod`
//! — neutral home for verbose/shm-size/config-file-parts shared by
//! eval and probe so they don't circularly depend on each other),
//! `shell_descriptor` (wire-format struct shared between the test
//! binary's `--ktstr-shell-test=<NAME>` producer and cargo-ktstr's
//! shell-mode consumer), `wprof` (`#[cfg(feature = "wprof")]` —
//! Perfetto-trace wire constants + `.wprof.pb` shape assertions),
//! `sidecar` (per-run JSON records), `staged`
//! (`pub(crate) mod` — staged-payload writer), `timefmt` (ISO-8601
//! + run-id helpers), and `topo` (topology override parsing).
//!
//! A `#[cfg(test)] pub(crate) mod test_helpers` exists for cross-file
//! test wiring; it is not part of the production surface.

#[cfg(test)]
use crate::assert::AssertResult;
#[cfg(test)]
use crate::scenario::Ctx;
#[cfg(test)]
use anyhow::Result;

mod args;
// Re-exported for the workload-side CgroupChurn worker, which resolves the
// same workload cgroup root the host-side setup uses but lives outside the
// private `args` module's subtree.
pub(crate) use args::resolve_cgroup_root;
mod boot_retry;
// Shared by the guest (rust_init formats the AP-gap panic around the
// marker) and the host cell layer (verifier + eval retry the whole boot
// when a run's crash_message carries it).
#[cfg(target_arch = "x86_64")]
pub(crate) use boot_retry::guest_kernel_rejected_wide_apic;
pub(crate) use boot_retry::{AP_BRINGUP_GAP_MARKER, run_vm_with_ap_gap_retry};
mod dispatch;
mod entry;
mod entry_validate;
mod eval;
mod host_class;
mod metrics;
mod output;
#[doc(hidden)]
pub use output::strip_ansi_csi;
// Reachable crate-wide (vmm::VmResult::guest_assert_result parses the guest
// AssertResult from its own drained guest_messages via this helper, mirroring
// the eval-layer use). The fn itself is pub(crate); this just lifts it out of
// the private `output` module so the vmm layer can call it by a stable path.
pub(crate) use output::parse_assert_result_from_drain;
mod payload;
mod probe;
// Reachable crate-wide so the freeze coordinator can detect the guest's
// probe-payload end delimiter in the drained bulk-port stdout stream and
// hold teardown until Phase 6b has shipped (see freeze_coord's wprof-ship
// kill gating). The const itself is pub(crate); this lifts it out of the
// private `probe` module onto a stable `test_support::` path.
pub(crate) use probe::PROBE_OUTPUT_END;
mod probe_metrics;
mod profraw;
pub use eval::{
    capture_starvation_witness, periodic_starvation_gate, post_vm_skip, stall_ejection_skip,
    starved_below_minimum_skip,
};
pub use profraw::current_binary_is_coverage_instrumented;
pub mod runtime;
mod shell_descriptor;
pub use shell_descriptor::{SchedulerKind, ShellTestDescriptor};
#[cfg(feature = "wprof")]
pub mod wprof;
#[cfg(feature = "wprof")]
pub use wprof::{PERFETTO_TRACE_PACKETS_TAG, WPROF_PB_MIN_BYTES, assert_wprof_pb_shape};
#[cfg(test)]
pub(crate) mod btf_blob;
mod sidecar;
pub(crate) mod staged;
#[cfg(test)]
pub(crate) mod test_helpers;
mod timefmt;
mod topo;

/// Shared callback signature for the
/// [`KtstrTestEntry::post_vm`](entry::KtstrTestEntry::post_vm) and
/// [`KtstrTestEntry::post_vm_unconditional`](entry::KtstrTestEntry::post_vm_unconditional)
/// host-side hooks. Both fields wrap this same shape in `Option<_>`;
/// the alias collapses the open-coded `fn(&crate::vmm::VmResult)
/// -> anyhow::Result<()>` repetition at the field declarations and
/// at the matching `with_post_vm{,_unconditional}` builder
/// parameters. Future post-VM hooks (e.g. an `expect_auto_repro`
/// artifact-existence checker) plug into the same shape without
/// triplicating the signature.
pub type PostVmCallback = fn(&crate::vmm::VmResult) -> anyhow::Result<()>;

// extract_probe_stack_arg and extract_work_type_arg are reached in
// production via `super::args::` (probe.rs); the re-export here
// preserves the flat-namespace invariant so `test_support::X` resolves
// uniformly across all CLI arg extractors.
#[cfg(feature = "export")]
pub(crate) use args::extract_export_output_arg;
#[allow(unused_imports)]
pub(crate) use args::{
    CellParentCgroupArg, VERIFIER_WORKLOAD_FLAG, cell_parent_path_is_valid,
    extract_export_test_arg, extract_probe_stack_arg, extract_shell_test_arg, extract_test_fn_arg,
    extract_topo_arg, extract_work_type_arg, is_verifier_workload, parse_cell_parent_cgroup,
};
#[allow(unused_imports)]
pub(crate) use runtime::{append_base_sched_args, content_hash, scratch_dir, sys_rdy_budget_ms};
#[cfg(test)]
pub(crate) use sidecar::enriched_parse_error_message_for_test;
pub use sidecar::{
    PerfDeltaAssertionRecord, SidecarResult, ThroughputDenomination, collect_pool,
    detect_kernel_commit, format_run_artifact_footer, newest_run_dir, repo_is_dirty, runs_root,
    sidecar_dir, source_dir_for,
};
pub(crate) use sidecar::{
    SidecarIoError, SidecarParseError, apply_archive_source_override, collect_sidecars,
    collect_sidecars_with_errors, format_callback_profile, format_kvm_stats, format_verifier_stats,
    is_run_directory, is_sidecar_filename, warn_skipped_sidecars,
};

pub use dispatch::{
    DEFAULT_HOST_CGROUP_PARENT, EXIT_FAIL, EXIT_INCONCLUSIVE, EXIT_PASS, analyze_sidecars,
    is_cpu_budget_unsatisfiable, is_kernel_unavailable, is_perf_mode_unavailable,
    is_resource_contention, is_topology_insufficient, is_topology_unrepresentable, ktstr_main,
    ktstr_test_early_dispatch, resolve_host_cgroup_parent, run_ktstr_test, sanitize_kernel_label,
};
pub use entry::{
    BinaryKindJson, BpfMapAgg, BpfMapWrite, CgroupPath, KTSTR_SCHEDULERS, KTSTR_TESTS,
    KtstrTestEntry, MemSideCache, NumaDistance, NumaNode, PerfDeltaAssertion, Scheduler,
    SchedulerJson, SchedulerListEntry, SchedulerSpec, SchedulerTestJson, Sysctl, SysctlJson,
    Topology, TopologyConstraints, TopologyConstraintsJson, TopologyJson, WatchBpfMap,
    default_post_vm_periodic_fired, find_scheduler, find_test,
};
pub use eval::{KernelUnavailable, ResolveSource, resolve_scheduler, resolve_test_kernel};
pub(crate) use eval::{record_skip_sidecar, run_ktstr_test_inner};
pub(crate) use host_class::host_skip_class;
pub use host_class::{HostClass, classify_host_error};
pub use metrics::{
    MAX_WALK_DEPTH, WALK_TRUNCATION_SENTINEL_NAME, extract_metrics, is_truncation_sentinel_name,
    walk_json_leaves,
};
pub(crate) use output::extract_panic_message;
pub use payload::{
    Metric, MetricCheck, MetricHint, MetricStream, OutputFormat, Payload, PayloadKind,
    PayloadMetrics, Polarity,
};
pub(crate) use probe::maybe_dispatch_vm_test;
pub(crate) use probe::{
    PROBE_DRAIN_GRACE, finalize_probe_after_unwind, maybe_dispatch_vm_test_with_args,
    maybe_dispatch_vm_test_with_phase_a, propagate_rust_env_from_cmdline, start_probe_phase_a,
};
pub use probe_metrics::{
    MAX_SCAN_INDEX, ThreadLookup, count_indexed_metrics, find_metric, find_metric_u64,
    flat_metrics_dump, has_metric, lookup_thread, snapshot_count, snapshot_worker_allocated,
    thread_count,
};
pub use profraw::target_dir as profraw_target_dir;
pub(crate) use profraw::{find_symbol_vaddrs, persist_guest_profraw, try_flush_profraw};
pub(crate) use timefmt::now_iso8601;
pub(crate) use topo::{TopoOverride, parse_topo_string};

/// Host capacity triple `(cpus, llcs, max_cpus_per_llc)` used to
/// filter gauntlet topology presets against what the host can actually
/// schedule. Both `dispatch::list_tests_*` (gauntlet variant filter)
/// and `dispatch::list_verifier_cells_all` (verifier sweep filter)
/// share this single source of truth so the two filters never drift.
/// Reads `available_parallelism()` for CPU count + `HostTopology::from_sysfs()`
/// for LLC layout; falls back to single-LLC + single-cpu-per-llc when
/// sysfs is unavailable.
pub fn host_capacity() -> (u32, u32, u32) {
    let host_cpus = std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(1);
    let host_topo = crate::vmm::host_topology::HostTopology::from_sysfs().ok();
    let host_llcs = host_topo
        .as_ref()
        .map(|t| t.llc_groups.len() as u32)
        .unwrap_or(1);
    let host_max_cpus_per_llc = host_topo
        .as_ref()
        .map(|t| t.max_cores_per_llc() as u32)
        .unwrap_or(host_cpus);
    (host_cpus, host_llcs, host_max_cpus_per_llc)
}

// ---------------------------------------------------------------------------
// Test infrastructure requirements
// ---------------------------------------------------------------------------
//
// `require_*` helpers turn missing test infrastructure into a panic with
// an actionable message instead of a silent skip. Use them when a test
// is meaningless without the resource -- a missing kernel, vmlinux,
// scheduler binary, or kernel-symbol resolution means the harness is
// misconfigured, not that the test should pass quietly. CI silently
// passing 100 "tests" that all early-returned because no kernel was
// findable is the failure mode these helpers exist to prevent.
//
// For genuine skips (raw BTF at /sys/kernel/btf/vmlinux, host without
// the architectural dependency the test exercises), call the crate's
// `skip!("reason: {detail}")` macro (see `src/test_macros.rs`). It
// emits the canonical `ktstr: SKIP: ...` line and returns from the
// test.

/// Whether the current test process was launched by a cargo-ktstr
/// orchestration path (`cargo ktstr test`, `cargo ktstr verifier`)
/// vs. a raw `cargo nextest run` / `cargo test`.
///
/// Reads [`crate::KTSTR_ORCHESTRATED_ENV`]; only checks presence,
/// not value (cargo-ktstr always sets it to `"1"`, but the marker
/// semantics are presence-only). Returns `false` when the env var
/// is unset or unreadable.
///
/// Tests that boot real KVM VMs use this to skip when running
/// under raw nextest, where the 7000+-test concurrency starves
/// per-VM resource budgets and produces a misleading "kill set by
/// AP" failure that looks like a real bug. cargo-ktstr's
/// orchestrator constrains the VM-test concurrency so the budgets
/// hold; skipping under raw nextest surfaces the operator-error
/// (wrong runner) without masking real failures during proper
/// orchestrated runs.
///
/// `pub(crate)` — only callers are integration-test helpers under
/// `src/vmm/mod.rs`'s `#[cfg(test)]` mod. The env-var name itself
/// is `pub` via [`crate::KTSTR_ORCHESTRATED_ENV`] for
/// documentation purposes.
#[cfg(test)]
#[allow(dead_code)] // called from x86_64-only tests in vmm/mod.rs
pub(crate) fn cargo_ktstr_orchestrated() -> bool {
    std::env::var(crate::KTSTR_ORCHESTRATED_ENV).is_ok()
}

/// Skip-message body for vmm-boot tests that bail when the test
/// process wasn't launched by cargo-ktstr orchestration. The
/// canonical extended rationale lives inline at the
/// `boot_kernel_with_monitor` site; sibling sites reference back
/// to it via this shared const so a future message tweak lands in
/// one place instead of four. The 4 sibling sites previously
/// carried byte-for-byte-identical copies of this string — per
/// the no-mega-no-dupes policy the 3+-site threshold mandates a
/// shared const.
#[cfg(test)]
#[allow(dead_code)] // referenced from x86_64-only vmm/mod.rs tests
pub(crate) const SKIP_NOT_ORCHESTRATED_MSG: &str = "raw nextest fan-out starves KVM resource budgets — see \
     boot_kernel_with_monitor for the shared rationale. Run via \
     `cargo ktstr test --kernel ../linux`.";

#[cfg(test)]
mod cargo_ktstr_orchestrated_tests {
    //! Pin the env-var-presence detection contract. A regression
    //! that renamed [`crate::KTSTR_ORCHESTRATED_ENV`] silently
    //! would make every vmm-test skip even under cargo-ktstr
    //! orchestration (where the env IS set), turning the
    //! VM-boot test suite into an always-green no-op. Pin the
    //! two-arm contract (set → true, unset → false) so the rename
    //! surfaces here.
    use super::cargo_ktstr_orchestrated;
    use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
    #[test]
    fn cargo_ktstr_orchestrated_true_when_env_set() {
        let _lock = lock_env();
        let _guard = EnvVarGuard::set(crate::KTSTR_ORCHESTRATED_ENV, "1");
        assert!(
            cargo_ktstr_orchestrated(),
            "KTSTR_ORCHESTRATED set → orchestrated check must return true"
        );
    }
    #[test]
    fn cargo_ktstr_orchestrated_false_when_env_unset() {
        let _lock = lock_env();
        let _guard = EnvVarGuard::remove(crate::KTSTR_ORCHESTRATED_ENV);
        assert!(
            !cargo_ktstr_orchestrated(),
            "KTSTR_ORCHESTRATED absent → orchestrated check must return false \
             (otherwise raw nextest invocations would run vmm tests and \
             starve KVM resource budgets)"
        );
    }
}

/// Resolve a kernel image path or panic with an actionable message.
///
/// Wraps [`crate::find_kernel`]: an `Err` (KTSTR_KERNEL points at a
/// path with no kernel image, cache lookup failed) and a successful
/// `Ok(None)` (no kernel discoverable) both panic. Tests that boot a
/// VM cannot proceed without a kernel; silently skipping turns CI
/// breakage into a green run.
#[cfg(test)]
#[allow(dead_code)] // called from x86_64-only tests in vmm/mod.rs
pub(crate) fn require_kernel() -> std::path::PathBuf {
    match crate::find_kernel() {
        Ok(Some(p)) => p,
        Ok(None) => panic!(
            "ktstr_test: test requires a kernel but none was found. {}",
            crate::KTSTR_KERNEL_HINT
        ),
        Err(e) => panic!("ktstr_test: kernel resolution failed: {e:#}"),
    }
}

/// Resolve a vmlinux path next to a kernel image or panic.
///
/// `kernel_path` is the value returned by [`require_kernel`]. The
/// vmlinux is required for symbol address lookup, BTF, and probe
/// source resolution -- a kernel image without vmlinux means the
/// cache entry is corrupt or the build was incomplete, which is an
/// infrastructure failure rather than a legitimate skip.
#[cfg(test)]
#[allow(dead_code)] // called from x86_64-only tests in vmm/mod.rs
pub(crate) fn require_vmlinux(kernel_path: &std::path::Path) -> std::path::PathBuf {
    crate::vmm::find_vmlinux(kernel_path).unwrap_or_else(|| {
        panic!(
            "ktstr_test: no vmlinux found alongside {}. The cache entry or \
             kernel build is incomplete. Rebuild with `cargo ktstr kernel \
             build --force`; the specified kernel must include `vmlinux` \
             alongside the boot image. {}",
            kernel_path.display(),
            crate::KTSTR_KERNEL_HINT,
        )
    })
}

/// Build a workspace package and return its binary path, or panic.
///
/// Wraps [`crate::build_and_find_binary`]. A failed build or missing
/// artifact for a required scheduler binary (e.g. `scx-ktstr`) is an
/// infrastructure failure -- the workspace is broken, not the test.
#[cfg(test)]
pub(crate) fn require_binary(package: &str) -> std::path::PathBuf {
    crate::build_and_find_binary(package, env!("CARGO_MANIFEST_DIR")).unwrap_or_else(|e| {
        panic!(
            "ktstr_test: build of `{package}` failed: {e:#}. \
             Run `cargo build -p {package}` to reproduce and diagnose."
        )
    })
}

/// Resolve [`crate::monitor::symbols::KernelSymbols`] from a vmlinux
/// or panic. The symbol table is required for any host-side memory
/// introspection; an unparseable vmlinux is an infrastructure failure.
#[cfg(test)]
#[allow(dead_code)] // called from x86_64-only tests in vmm/mod.rs
pub(crate) fn require_kernel_symbols(
    vmlinux_path: &std::path::Path,
) -> crate::monitor::symbols::KernelSymbols {
    crate::monitor::symbols::KernelSymbols::from_vmlinux(vmlinux_path).unwrap_or_else(|e| {
        panic!(
            "ktstr_test: kernel symbol resolution from {} failed: {e:#}",
            vmlinux_path.display(),
        )
    })
}

/// Resolve [`crate::monitor::btf_offsets::KernelOffsets`] from a vmlinux
/// or panic. BTF resolution is required for any host-side kernel
/// struct introspection; a vmlinux whose BTF fails to parse is an
/// infrastructure failure, not a test-skip condition.
#[cfg(test)]
pub(crate) fn require_kernel_offsets(
    vmlinux_path: &std::path::Path,
) -> crate::monitor::btf_offsets::KernelOffsets {
    crate::monitor::btf_offsets::KernelOffsets::from_vmlinux(vmlinux_path).unwrap_or_else(|e| {
        panic!(
            "ktstr_test: kernel BTF resolution from {} failed: {e:#}. \
             The kernel must be built with CONFIG_DEBUG_INFO_BTF=y; \
             rebuild with `cargo ktstr kernel build --force` if the \
             cache entry was produced without BTF.",
            vmlinux_path.display(),
        )
    })
}

/// Resolve [`crate::monitor::btf_offsets::BpfMapOffsets`] from a vmlinux
/// or panic. A vmlinux whose BTF fails to yield BPF map offsets is an
/// infrastructure failure, not a test-skip condition.
#[cfg(test)]
pub(crate) fn require_bpf_map_offsets(
    vmlinux_path: &std::path::Path,
) -> crate::monitor::btf_offsets::BpfMapOffsets {
    crate::monitor::btf_offsets::BpfMapOffsets::from_vmlinux(vmlinux_path).unwrap_or_else(|e| {
        panic!(
            "ktstr_test: BpfMapOffsets resolution from {} failed: {e:#}. \
             The kernel must be built with CONFIG_DEBUG_INFO_BTF=y; \
             rebuild with `cargo ktstr kernel build --force` if the \
             cache entry was produced without BTF.",
            vmlinux_path.display(),
        )
    })
}

/// Resolve [`crate::monitor::btf_offsets::BpfProgOffsets`] from a vmlinux
/// or panic. A vmlinux whose BTF fails to yield BPF program offsets is
/// an infrastructure failure, not a test-skip condition.
#[cfg(test)]
pub(crate) fn require_bpf_prog_offsets(
    vmlinux_path: &std::path::Path,
) -> crate::monitor::btf_offsets::BpfProgOffsets {
    crate::monitor::btf_offsets::BpfProgOffsets::from_vmlinux(vmlinux_path).unwrap_or_else(|e| {
        panic!(
            "ktstr_test: BpfProgOffsets resolution from {} failed: {e:#}. \
             The kernel must be built with CONFIG_DEBUG_INFO_BTF=y; \
             rebuild with `cargo ktstr kernel build --force` if the \
             cache entry was produced without BTF.",
            vmlinux_path.display(),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ktstr_test;
    use linkme::distributed_slice;

    // Register a test entry in the distributed slice for unit testing find_test.
    fn __ktstr_inner_unit_test_dummy(_ctx: &Ctx) -> Result<AssertResult> {
        Ok(AssertResult::pass())
    }

    #[distributed_slice(KTSTR_TESTS)]
    static __KTSTR_ENTRY_UNIT_TEST_DUMMY: KtstrTestEntry = KtstrTestEntry {
        name: "__unit_test_dummy__",
        func: __ktstr_inner_unit_test_dummy,
        ..KtstrTestEntry::DEFAULT
    };

    #[test]
    fn find_test_registered_entry() {
        let entry = find_test("__unit_test_dummy__");
        assert!(entry.is_some(), "registered entry should be found");
        let entry = entry.unwrap();
        assert_eq!(entry.name, "__unit_test_dummy__");
        assert_eq!(entry.topology.llcs, 1);
        assert_eq!(entry.topology.cores_per_llc, 2);
    }

    #[test]
    fn find_test_nonexistent() {
        assert!(find_test("__nonexistent_test_xyz__").is_none());
    }

    #[test]
    fn find_test_from_distributed_slice() {
        // KTSTR_TESTS should contain at least the __unit_test_dummy__ entry.
        assert!(!KTSTR_TESTS.is_empty());
    }

    // Macro-codegen coverage for the cpu_budget attr: drive the real
    // #[ktstr_test] expansion (the parse arm + the Some(n)/None field-emit
    // arms) and assert the emitted KtstrTestEntry carries the right
    // cpu_budget. host_only keeps the probe from booting a VM; no_perf_mode
    // satisfies the macro's cpu_budget-requires-no_perf_mode gate. find_test
    // reads the registered entry from KTSTR_TESTS without invoking the body.
    // This RUNS in CI (a src/ #[cfg(test)] unit test), unlike the
    // ktstr-macros crate's own suite which cargo-ktstr does not discover.
    #[ktstr_test(cpu_budget = 7, host_only, no_perf_mode)]
    fn cpu_budget_codegen_probe(_ctx: &Ctx) -> Result<AssertResult> {
        Ok(AssertResult::pass())
    }

    #[ktstr_test(host_only)]
    fn cpu_budget_codegen_probe_none(_ctx: &Ctx) -> Result<AssertResult> {
        Ok(AssertResult::pass())
    }

    #[test]
    fn cpu_budget_attr_emits_some() {
        let e = find_test("cpu_budget_codegen_probe")
            .expect("the cpu_budget probe entry must be registered");
        assert_eq!(
            e.cpu_budget,
            Some(7),
            "#[ktstr_test(cpu_budget = 7)] must emit cpu_budget: Some(7)"
        );
    }

    #[test]
    fn omitted_cpu_budget_defaults_none() {
        let e = find_test("cpu_budget_codegen_probe_none")
            .expect("the no-cpu_budget probe entry must be registered");
        assert_eq!(
            e.cpu_budget, None,
            "omitting cpu_budget must leave the DEFAULT None (no field emitted)"
        );
    }
}
