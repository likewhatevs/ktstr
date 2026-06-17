//! Host-side VM result evaluation for `#[ktstr_test]` runs.
//!
//! The core [`run_ktstr_test_inner`] orchestrates a single test run:
//! boot the guest VM with the scheduler and workload, collect profraw
//! + stimulus events from SHM, then hand off to [`evaluate_vm_result`]
//!   for pass/fail judgment and error-message construction.
//!
//! [`evaluate_vm_result`] is factored out of the VM-boot path so error
//! formatting can be unit-tested with synthetic `VmResult` values.
//!
//! Supporting items:
//! - [`resolve_scheduler`] / [`resolve_test_kernel`] locate the
//!   scheduler binary and kernel image from env + cache + filesystem.
//! - [`reporting::scheduler_label`] formats the `[sched=...]` bracket in error
//!   headers.
//! - [`reporting::format_monitor_section`] and [`reporting::trim_settle_samples`] handle the
//!   `--- monitor ---` block in failed-test output.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

use crate::assert::AssertResult;
use crate::timeline::StimulusEvent;
use crate::vmm;

use super::output::{
    classify_init_stage, extract_exit_from_dump_trace, extract_kernel_version,
    extract_panic_message, extract_sched_ext_dump, format_console_diagnostics,
    parse_assert_result_from_drain, sched_log_fingerprint,
};
use super::probe::attempt_auto_repro;
use super::profraw::write_profraw;
use super::sidecar::{write_sidecar, write_skip_sidecar};
use super::topo::TopoOverride;
use super::{KtstrTestEntry, SchedulerSpec, Topology};
mod kernel;
#[cfg(feature = "llm")]
mod llm_extract;
mod post_vm;
#[cfg(feature = "llm")]
pub(crate) use llm_extract::host_side_llm_extract;
pub use post_vm::post_vm_skip;
pub(crate) use post_vm::{
    ExpectAutoReproSatisfied, HostSkipRequest, LLM_MODEL_LOAD_FAILED_PREFIX,
    PostVmAssertionFailure, ScxBpfErrorMatcherMismatch, record_skip_sidecar, run_post_vm_callbacks,
    should_skip_on_llm_model_load_failure,
};
mod reporting;
mod scheduler;
use crate::verifier::{SCHED_OUTPUT_START, parse_sched_output};
pub use kernel::{KernelUnavailable, resolve_test_kernel};
pub(crate) use kernel::{acquire_test_kernel_lock_if_cached, ensure_kvm};
pub use scheduler::{ResolveSource, resolve_scheduler};
pub(crate) use scheduler::{dedupe_include_files, resolve_staged_schedulers_strict};

use super::runtime::{config_content_parts, config_file_parts, verbose, vm_timeout_from_entry};

// ---------------------------------------------------------------------------
// Failure-message constants
// ---------------------------------------------------------------------------
//
// Shared between the production error-formatting paths in this module
// and the tests that pin those messages. Editing a production string
// here without updating the test (or vice versa) is caught at compile
// time instead of as a runtime test assertion drift.

/// Header body for a timed-out run with no parseable AssertResult.
/// Pinned by `eval_timeout_no_result` and `eval_timeout_with_sched_includes_diagnostics`.
pub(crate) const ERR_TIMED_OUT_NO_RESULT: &str = "timed out (no result via bulk port or COM2)";

/// Header body for a run whose scenario passed but whose monitor
/// verdict failed. Pinned by `eval_monitor_fail_has_fingerprint` and
/// `eval_monitor_fail_includes_sched_log`.
pub(crate) const ERR_MONITOR_FAILED_AFTER_SCENARIO: &str = "passed scenario but monitor failed";

/// Reason body when a scheduler is running but no AssertResult was
/// received from the guest. Pinned by `eval_sched_exits_no_com2_output`
/// and `eval_sched_exits_with_sched_log`.
pub(crate) const ERR_NO_TEST_RESULT_FROM_GUEST: &str = "no test result received from guest \
     (no AssertResult arrived via bulk port or COM2; check kernel log and \
     scheduler exit status)";

/// Reason body when EEVDF (no scheduler) produced no AssertResult.
/// Pinned by `eval_eevdf_no_com2_output` and `eval_payload_exits_no_check_result`.
pub(crate) const ERR_NO_TEST_FUNCTION_OUTPUT: &str =
    "test function produced no output (no test result found)";

/// Prefix for the `guest crashed: ...` reason body. Pinned by
/// `eval_crash_in_output_says_guest_crashed`, `eval_crash_eevdf_says_guest_crashed`,
/// and `eval_crash_message_from_field`.
pub(crate) const ERR_GUEST_CRASHED_PREFIX: &str = "guest crashed:";

// ---------------------------------------------------------------------------
// Host-side OutputFormat::LlmExtract resolution
// ---------------------------------------------------------------------------
//
// The guest's `payload_run::evaluate_llm_extract_deferred` ships
// raw stdout/stderr across the SHM ring under
// `MSG_TYPE_RAW_PAYLOAD_OUTPUT` and emits an empty-metrics
// `PayloadMetrics` placeholder under `MSG_TYPE_PAYLOAD_METRICS` for
// every `OutputFormat::LlmExtract` invocation. The guest does NOT
// load the local model into VM RAM (the model is ~2.55 GiB; the test
// VM's RAM budget cannot accommodate it). The host runs
// `extract_via_llm` here, after VM exit, on the captured text — same
// stdout-primary / stderr-fallback contract that previously lived in
// the prior in-VM extraction path — and replaces the empty `metrics` vec on
// the paired `PayloadMetrics` with the extracted result.

// ---------------------------------------------------------------------------
// Host-state save/restore guards for the VM-run path
// ---------------------------------------------------------------------------
//
// `CpuStateGuard` captures per-thread host state on construction
// and restores it on Drop. Protects `run_ktstr_test_inner` against
// state leaks from KVM — the kernel's fpu_swap_kvm_fpstate should
// keep host CPU state intact across VM exit, but we've confirmed
// MXCSR PE flag leaks. Using a Drop guard rather than fall-through
// restore means error paths (Err from builder.build()/vm.run(),
// panics) cannot bypass the restore.
//
// The x86_64 definition saves XSAVE area + FS/GS_BASE + PKRU
// (each CPUID-gated) plus sigmask + SIGRTMIN sigaction.
//
// The aarch64 definition saves V0..V31 + FPCR + FPSR (the FPSIMD
// register file plus the two control/status registers that are the
// MXCSR analog) plus sigmask + SIGRTMIN sigaction. The kernel's
// `kvm_arch_vcpu_load_fp` / `kvm_arch_vcpu_put_fp`
// (arch/arm64/kvm/fpsimd.c) pair drives a
// `fpsimd_save_and_flush_cpu_state()` save plus `TIF_FOREIGN_FPSTATE`
// reload at return-to-userspace through `task_fpsimd_load`
// (arch/arm64/kernel/fpsimd.c) — so in the absence of a kernel bug
// our restore is redundant. The same was true on x86_64 for MXCSR
// before the PE-flag leak surfaced; this guard is the same
// defense-in-depth on arm64.
//
// SVE Z0..Z31 / P0..P15 / FFR are deliberately NOT saved here. The
// kernel's `task_fpsimd_load` reloads them from
// `current->thread.sve_state` when `TIF_SVE` is set on the eval
// thread, so the same defense covers SVE in the no-bug case. Saving
// them in user space requires `.arch_extension sve` inline asm with
// dynamic vector-length sizing (`prctl(PR_SVE_GET_VL)`) plus a
// per-thread allocation; running an SVE store on a thread with
// `TIF_SVE` clear would itself trip `do_sve_acc` and flip the flag
// as an observer effect, so naive saving is incorrect. If a kernel
// bug surfaces in the SVE restore path analogous to the MXCSR PE
// leak, this guard expands to cover it then.
//
// The bare-fallback arm covers any future host arch that slips
// past the build's x86_64+aarch64 expectation; it preserves the
// sigmask + SIGRTMIN restore so the SIGRTMIN stop-vcpu trampoline
// never leaks into the test runner's main loop.

/// Public-crate entry point. Thin wrapper around
/// [`run_ktstr_test_inner_impl`] that records a skip sidecar as a
/// **defensive late catch-all** whenever the inner pipeline bails
/// with a [`crate::vmm::host_topology::ResourceContention`].
///
/// No pre-build path constructs `ResourceContention` today. Every
/// `ResourceContention` construction site in the crate
/// (`vmm::host_topology` and `vmm::mod`) fires from inside
/// `builder.build()` or `vm.run()` — both already record their
/// own sidecar at the bail point via the per-site
/// `record_skip_sidecar` calls in the match arms below. The
/// pre-build helpers (`ensure_kvm`, `resolve_test_kernel`,
/// `acquire_test_kernel_lock_if_cached`, `resolve_scheduler`)
/// produce plain `anyhow` contexts, `KernelUnavailable`, or
/// `anyhow::bail!` string errors — none of those route through
/// this wrapper's downcast.
///
/// This wrapper exists to guard against future migrations that
/// move a contention-producing site into the pre-build chain
/// (e.g. a flock timeout that today returns plain `anyhow` but
/// later starts surfacing as `ResourceContention`). Catching it
/// here means stats coverage stays correct without requiring a
/// per-site retro-fit at every new producer. Today the wrapper
/// is purely defensive — drop it and stats tooling sees no
/// difference until such a migration lands.
///
/// Double-recording is acceptable but NOT idempotent in the
/// strict sense: the wrapper's write overwrites the per-site
/// write's `run_id` and timestamp with fresh values produced at
/// wrapper-return time. Both recordings carry the same skip
/// classification (the same `ResourceContention` reason flowing
/// through the same `write_skip_sidecar`), so the final sidecar
/// is correct for stats tooling — only the run_id / timestamp
/// shift between the two writes, and downstream tooling keys on
/// (test name, skip classification), not on those volatile
/// fields.
pub(crate) fn run_ktstr_test_inner(
    entry: &KtstrTestEntry,
    topo: Option<&TopoOverride>,
) -> Result<AssertResult> {
    let result = run_ktstr_test_inner_impl(entry, topo);
    if let Err(ref e) = result
        && super::is_resource_contention(e)
    {
        // Late catch-all for ResourceContention from any early-
        // bail path before the existing per-site `record_skip_sidecar`
        // calls in builder.build()/vm.run() arms below. Walks the
        // FULL `anyhow::Error` chain via `is_resource_contention`
        // (which uses `e.chain().any(...)`) so a contention wrapped
        // in `.context(...)` (e.g. the `"build ktstr_test VM"` and
        // `"run ktstr_test VM"` wrappers in `evaluate_vm_result`)
        // is still recognised — without the chain walk, a wrapped
        // contention would skip the catch-all and the run would
        // not record a skip sidecar. Not strictly idempotent — a
        // second write refreshes run_id and timestamp — but the
        // skip classification round-trips identically, so stats
        // tooling sees the same outcome.
        record_skip_sidecar(entry);
    }
    // `expect_auto_repro = true` inversion: when the primary VM
    // failed AND the auto-repro VM landed a shape-valid
    // `.repro.wprof.pb`, `evaluate_vm_result` attached the
    // [`ExpectAutoReproSatisfied`] marker to the failure chain.
    // The dispatch path at `result_to_exit_code` consumes the
    // marker by routing the verdict to `EXIT_PASS`, but the
    // `run_ktstr_test` (the library entry) and the `#[ktstr_test]`
    // macro's emitted `#[test]` body consume this Result directly —
    // NOT through `result_to_exit_code` — so the marker-bypass
    // inversion the dispatch arm performs would be missed on those
    // paths without this conversion. The canonical `cargo ktstr test`
    // / nextest path does NOT reach here: the ctor intercepts
    // `--exact ktstr/<name>` and routes to `run_named_test` →
    // `result_to_exit_code` (dispatch.rs), which honors the markers via
    // its own arms. This conversion covers the direct `cargo test`
    // harness and out-of-tree library callers of `run_ktstr_test`.
    //
    // Marker precedence mirrors `result_to_exit_code`: a
    // `PostVmAssertionFailure` (a host-side post_vm regression) must
    // NOT be inverted, so it is checked FIRST and kept as an `Err`;
    // only `ExpectAutoReproSatisfied` with no post_vm failure converts
    // to `Ok(pass)`. The diagnostic surfaces via the `eprintln!` and
    // the per-test stderr capture's original failure trail.
    match result {
        Err(e) => {
            if e.downcast_ref::<PostVmAssertionFailure>().is_some() {
                Err(e)
            } else if e.downcast_ref::<ExpectAutoReproSatisfied>().is_some() {
                eprintln!("{e:#}");
                Ok(AssertResult::pass())
            } else {
                Err(e)
            }
        }
        Ok(r) => Ok(r),
    }
}

/// Write a placeholder `failure-dump.json` at `path` when the file
/// does not already exist. Called from the primary VM dispatch on
/// failure paths where the freeze coordinator did NOT produce a
/// real BPF-state dump (pre-attach failures: send_sys_rdy timeout,
/// VM boot failure, scheduler binary load failure, post_vm callback
/// failure) so the spec promise ("every failed test writes
/// `<test_name>.failure-dump.json` to the sidecar dir") survives
/// those code paths too.
///
/// The placeholder's `is_placeholder: true` field lets downstream
/// tooling (stats compare, sidecar walkers) distinguish a stub
/// from a real BPF dump. The `reason` string carries both the
/// lifecycle stage classification AND — when extractable — the
/// `BUG SUMMARY` text (per `extract_bug_summary` over the captured
/// scheduler log + sched_ext dump), so the disk artifact carries
/// the same actionable diagnostic the operator sees in stderr
/// rather than a less-informative on-disk stub.
///
/// Durability: opens the `.tmp` file, writes, `fsync()`s, drops,
/// then `rename(2)`s atomically. Matches the freeze coordinator's
/// own atomic-publish pattern. A concurrent reader either sees no
/// file or sees a complete stub, never a truncated one; a host
/// crash between write and writeback preserves the data via the
/// fsync.
///
/// Errors at every step (serialize, mkdir, file create, write,
/// fsync, rename) emit a `tracing::warn` naming the path and the
/// io error. The test failure itself surfaces via the
/// normal stderr path; the stub is a best-effort augment, so the
/// helper does not propagate any io::Error.
fn write_placeholder_failure_dump_if_missing(path: &std::path::Path, result: &vmm::VmResult) {
    if path.exists() {
        return;
    }
    let stage_label =
        crate::test_support::output::classify_init_stage(result.guest_messages.as_ref());
    // Fold BUG SUMMARY into the on-disk reason so the disk artifact
    // matches the stderr summary (the design's stated goal — see
    // src/test_support/output.rs::extract_bug_summary).
    let sched_log_merged = crate::verifier::concat_sched_log_chunks(result.guest_messages.as_ref());
    let sched_log_input: &str = if !sched_log_merged.is_empty() {
        &sched_log_merged
    } else {
        &result.output
    };
    let raw_dump = extract_sched_ext_dump(&result.stderr).unwrap_or_default();
    let bug_summary = crate::test_support::output::extract_bug_summary(sched_log_input, &raw_dump);
    let reason = match bug_summary {
        Some(s) => format!(
            "test failed at stage `{stage_label}`; no BPF state captured \
             (probe did not attach before failure). BUG SUMMARY: {s}"
        ),
        None => format!(
            "test failed at stage `{stage_label}`; no BPF state captured \
             (probe did not attach before failure)"
        ),
    };
    let stub = crate::monitor::dump::FailureDumpReport::placeholder(reason);
    let json = match serde_json::to_string_pretty(&stub) {
        Ok(j) => j,
        Err(e) => {
            tracing::warn!(
                error = %e,
                path = %path.display(),
                "eval: failed to serialize placeholder failure-dump"
            );
            return;
        }
    };
    if let Some(parent) = path.parent()
        && let Err(e) = std::fs::create_dir_all(parent)
    {
        tracing::warn!(
            error = %e,
            path = %parent.display(),
            "eval: failed to create parent dir for placeholder failure-dump"
        );
        return;
    }
    let tmp = path.with_extension("json.tmp");
    let mut file = match std::fs::File::create(&tmp) {
        Ok(f) => f,
        Err(e) => {
            tracing::warn!(
                error = %e,
                path = %tmp.display(),
                "eval: failed to create placeholder failure-dump tmp file"
            );
            return;
        }
    };
    use std::io::Write;
    if let Err(e) = file.write_all(json.as_bytes()) {
        tracing::warn!(
            error = %e,
            path = %tmp.display(),
            "eval: failed to write placeholder failure-dump"
        );
        let _ = std::fs::remove_file(&tmp);
        return;
    }
    if let Err(e) = file.sync_all() {
        tracing::warn!(
            error = %e,
            path = %tmp.display(),
            "eval: failed to fsync placeholder failure-dump tmp file"
        );
        // Don't bail — the data is in the page cache, rename will
        // still publish a consistent file; only crash-safety is
        // weakened. Operators see the warn either way.
    }
    drop(file);
    if let Err(e) = std::fs::rename(&tmp, path) {
        tracing::warn!(
            error = %e,
            tmp = %tmp.display(),
            target = %path.display(),
            "eval: failed to rename placeholder failure-dump tmp file"
        );
        let _ = std::fs::remove_file(&tmp);
    }
}

fn run_ktstr_test_inner_impl(
    entry: &KtstrTestEntry,
    topo: Option<&TopoOverride>,
) -> Result<AssertResult> {
    entry.validate().context("KtstrTestEntry validation")?;
    if let Some(t) = topo {
        t.validate().context("TopoOverride validation")?;
    }
    // Pin rayon's global thread pool to the test's allowed CPUs.
    // The pool is created lazily on first rayon use (initramfs
    // build) — configuring it here ensures workers inherit the
    // test's cpuset instead of spreading across all host CPUs.
    //
    // `build_global` succeeds only once per process. Track the
    // first call's cpuset in a `OnceLock<Vec<usize>>`; subsequent
    // tests with a DIFFERENT cpuset can't repin the pool, so emit
    // a warning so the operator sees that the second test's
    // workers may run on the first test's CPUs.
    static FIRST_RAYON_CPUSET: std::sync::OnceLock<Vec<usize>> = std::sync::OnceLock::new();
    let host_cpus = crate::vmm::host_topology::host_allowed_cpus();
    if !host_cpus.is_empty() {
        let cpus = host_cpus.clone();
        let n = cpus.len();
        let range = format!("{}-{}", cpus[0], cpus[n - 1]);
        let cpus_for_handler = cpus.clone();
        let built = rayon::ThreadPoolBuilder::new()
            .num_threads(n.min(32))
            .start_handler(move |_idx| {
                let mut cpuset = nix::sched::CpuSet::new();
                for &cpu in &cpus_for_handler {
                    let _ = cpuset.set(cpu);
                }
                let _ = nix::sched::sched_setaffinity(nix::unistd::Pid::from_raw(0), &cpuset);
            })
            .build_global()
            .is_ok();
        if built {
            // First successful pin in this process. Record the
            // cpuset so later tests can compare.
            let _ = FIRST_RAYON_CPUSET.set(cpus);
            eprintln!("no_perf_mode: rayon pool pinned to {n} CPUs ({range})");
        } else if let Some(first) = FIRST_RAYON_CPUSET.get()
            && first != &host_cpus
        {
            // build_global fails on every call after the first.
            // When the second test's cpuset differs, warn — the
            // pool is still pinned to `first`, not to the
            // requested `host_cpus`, so workers run on a stale
            // cpuset until process exit.
            let first_n = first.len();
            let first_range = if first_n > 0 {
                format!("{}-{}", first[0], first[first_n - 1])
            } else {
                "empty".to_string()
            };
            eprintln!(
                "no_perf_mode: WARNING: rayon pool already pinned to {first_n} CPUs \
                 ({first_range}); requested {n} CPUs ({range}) won't take effect — \
                 build_global is one-shot per process",
            );
        }
    }
    if entry.performance_mode && super::runtime::no_perf_mode_active() {
        // One canonical reason string for both the stderr banner
        // (prefixed with the entry name for multi-test context)
        // and the structured AssertResult::skip payload (test-name
        // is carried on the surrounding entry). Prior code
        // duplicated the body verbatim across both sites, inviting
        // drift; the shared const keeps them in lockstep.
        const REASON: &str =
            "test requires performance_mode but --no-perf-mode or KTSTR_NO_PERF_MODE is active";
        crate::report::test_skip(format_args!("{}: {REASON}", entry.name));
        // Record the skip so stats tooling sees every skipped run,
        // not just the ones that made it to the VM-run site. A sidecar
        // write failure is logged but not propagated: the skip itself
        // is still valid — only post-run stats tooling loses visibility.
        record_skip_sidecar(entry);
        return Ok(AssertResult::skip(REASON));
    }
    ensure_kvm()?;
    let kernel = resolve_test_kernel()?;
    // Hold a reader flock on the cache entry (if the resolved
    // kernel lives in one). Prevents a concurrent
    // `cargo ktstr kernel build` from swapping the entry under
    // the VM mid-run. Dropped explicitly after VM exit (before
    // LLM inference) so concurrent kernel rebuilds aren't
    // blocked during the multi-second extraction phase. `None`
    // on non-cache kernels — those don't need coordination.
    let kernel_lock = acquire_test_kernel_lock_if_cached(&kernel)?;
    // Drop the ResolveSource on this path — the downstream sites (VM
    // builder, auto_repro) only need the PathBuf. Consumers that want
    // provenance (sidecar stamping, cache-key construction) must call
    // resolve_scheduler directly on the same spec; the source is
    // stable across identical inputs within a single process run.
    let scheduler = resolve_scheduler(&entry.scheduler.binary)?.0;
    let resolved_staged = resolve_staged_schedulers_strict(entry, |spec| {
        resolve_scheduler(spec).map(|(opt, _src)| opt)
    })?;
    let ktstr_bin = crate::resolve_current_exe()?;

    let guest_args = vec![
        "run".to_string(),
        "--ktstr-test-fn".to_string(),
        entry.name.to_string(),
    ];

    let cmdline_extra = super::runtime::build_cmdline_extra(entry);

    let (vm_topology, memory_mib) = super::runtime::resolve_vm_topology(entry, topo);

    let no_perf_mode = super::runtime::no_perf_mode_for_entry(entry);

    // Pre-clear stale failure-dump files before the primary VM
    // boots. A passing rerun after a prior failed invocation must
    // not be masked by the prior failure's leftovers — the E2E
    // test (`tests/failure_dump_e2e.rs`) reads from the primary
    // path and an operator inspecting the sidecar dir looks at
    // both. Both files are scoped per-test (`{name}.failure-
    // dump.json` and `{name}.repro.failure-dump.json`), so we can
    // safely unlink both from this single primary-dispatch
    // entry — auto-repro fires only after a primary failure
    // emits a dump, so any repro file present here is stale by
    // construction. NotFound is silenced; other unlink errors
    // emit `tracing::warn!` and dispatch proceeds (the freeze
    // coord's own write would overwrite a stale file in
    // practice; the warn flags permission / fs anomalies for the
    // operator).
    let primary_dump_path =
        super::sidecar::sidecar_dir().join(format!("{}.failure-dump.json", entry.name));
    let repro_dump_path =
        super::sidecar::sidecar_dir().join(format!("{}.repro.failure-dump.json", entry.name));
    for stale in [&primary_dump_path, &repro_dump_path] {
        match std::fs::remove_file(stale) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => tracing::warn!(
                path = %stale.display(),
                error = %e,
                "eval: failed to pre-clear stale failure-dump file"
            ),
        }
    }

    // Attach the primary failure-dump JSON sink at this dispatch
    // site, NOT inside `build_vm_builder_base` — auto-repro calls
    // the same base builder, so attaching the primary path there
    // and re-attaching for the auto-repro VM would let
    // `attempt_auto_repro`'s `.failure_dump_path(repro)` override
    // chain after a phantom primary attachment, which only
    // matters if the setter is later changed to be impure;
    // keeping the primary attachment here keeps the contract
    // robust against that change. The setter is currently pure;
    // pre-clear above handles stale-file removal.
    let mut builder = super::runtime::build_vm_builder_base(
        entry,
        &kernel,
        &ktstr_bin,
        scheduler.as_deref(),
        &resolved_staged,
        vm_topology,
        memory_mib,
        &cmdline_extra,
        &guest_args,
        no_perf_mode,
    )
    .failure_dump_path(primary_dump_path.clone())
    .performance_mode(entry.performance_mode);

    // Merge order: default_checks -> scheduler.assert -> per-test assert.
    let merged_assert = crate::assert::Assert::default_checks()
        .merge(&entry.scheduler.assert)
        .merge(&entry.assert);

    #[cfg(feature = "wprof")]
    {
        builder = super::runtime::attach_wprof_if_requested(builder, entry, "primary")?;
    }

    if let SchedulerSpec::KernelBuiltin { enable, disable } = &entry.scheduler.binary {
        builder = builder.sched_enable_cmds(enable);
        builder = builder.sched_disable_cmds(disable);
    }
    if entry.scheduler.has_bpf_scheduler() {
        builder = builder.monitor_thresholds(merged_assert.monitor_thresholds());
    }

    let mut sched_args: Vec<String> = Vec::new();
    // Declarative include-files: union every Payload's
    // `include_files` specs (scheduler + test payload + workloads +
    // entry.extra) through the same resolver the CLI `-i` flag uses,
    // then merge with the scheduler config file (if any). Dedupe
    // policy: identical `(archive_path, host_path)` pairs collapse
    // silently; a conflict on the same `archive_path` with
    // differing `host_path` aborts the test with a diagnostic
    // naming both sources — two unrelated declarations resolving
    // to the same archive slot is a real ambiguity the user must
    // resolve manually.
    let declarative_specs: Vec<std::path::PathBuf> = entry
        .all_include_files()
        .into_iter()
        .map(std::path::PathBuf::from)
        .collect();
    let mut resolved_includes: Vec<(String, std::path::PathBuf, &'static str)> =
        if declarative_specs.is_empty() {
            Vec::new()
        } else {
            crate::cli::resolve_include_files(&declarative_specs)
                .context("resolving declarative include_files from Payload definitions")?
                .into_iter()
                .map(|(a, h)| (a, h, "declarative"))
                .collect()
        };
    if let Some((archive_path, host_path, guest_path)) = config_file_parts(entry) {
        resolved_includes.push((archive_path, host_path, "scheduler config_file"));
        sched_args.push("--config".to_string());
        sched_args.push(guest_path);
    }
    if let Some((archive_path, host_path, _guest_path, args)) = config_content_parts(entry) {
        resolved_includes.push((archive_path, host_path, "inline config_content"));
        sched_args.extend(args);
    }
    let unioned = dedupe_include_files(&resolved_includes)?;
    if !unioned.is_empty() {
        builder = builder.include_files(unioned);
    }
    super::runtime::append_base_sched_args(entry, &mut sched_args);
    if !sched_args.is_empty() {
        builder = builder.sched_args(&sched_args);
    }

    // Save all per-thread CPU state that KVM could corrupt.
    // Defense in depth: the kernel's fpu_swap_kvm_fpstate should
    // restore host state on VM exit, but we've confirmed MXCSR PE
    // flag leaks. Full XSAVE + segment bases + PKRU + signal mask
    // guards against any state corruption.
    //
    // Wrapped in a Drop guard so the restore runs on ALL return
    // paths — including early Err returns from builder.build() and
    // vm.run(), and panics. Without the guard, error paths bypass
    // the restore, leaving the host with KVM-corrupted state.
    #[cfg(target_arch = "x86_64")]
    struct CpuStateGuard {
        // Backing allocation for the XSAVE area. align_ptr points
        // into this Vec — the Vec must outlive align_ptr so the
        // xrstor in Drop reads valid memory.
        #[allow(dead_code)]
        xsave_buf: Vec<u8>,
        align_ptr: *mut u8,
        has_xsave: bool,
        has_fsgsbase: bool,
        has_pku: bool,
        fsbase: u64,
        gsbase: u64,
        pkru: u32,
        sigmask: libc::sigset_t,
        sigrtmin_action: libc::sigaction,
    }
    #[cfg(target_arch = "x86_64")]
    impl Drop for CpuStateGuard {
        fn drop(&mut self) {
            unsafe {
                if self.has_xsave {
                    core::arch::asm!(
                        "xrstor [{}]",
                        in(reg) self.align_ptr,
                        in("eax") 0xFFFF_FFFFu32,
                        in("edx") 0xFFFF_FFFFu32,
                        options(nostack),
                    );
                }
                if self.has_fsgsbase {
                    core::arch::asm!("wrfsbase {}", in(reg) self.fsbase, options(nostack));
                    core::arch::asm!("wrgsbase {}", in(reg) self.gsbase, options(nostack));
                }
                if self.has_pku {
                    core::arch::asm!(
                        "xor ecx, ecx", "wrpkru",
                        in("eax") self.pkru, in("edx") 0u32, out("ecx") _,
                        options(nostack),
                    );
                }
                libc::pthread_sigmask(libc::SIG_SETMASK, &self.sigmask, std::ptr::null_mut());
                libc::sigaction(
                    libc::SIGRTMIN(),
                    &self.sigrtmin_action,
                    std::ptr::null_mut(),
                );
            }
        }
    }
    // FPSIMD V0..V31 occupy 32 * 16 = 512 bytes when stored as q
    // registers. `repr(C, align(16))` matches the alignment that
    // `stp q0, q1, [x0]` requires (16-byte for SIMD&FP load/store
    // pairs per ARM ARM C3.3.13). FPCR/FPSR are 32-bit
    // architecturally; `mrs` writes the full 64-bit destination
    // with the upper bits RES0, so a u64 slot is the natural size.
    #[cfg(target_arch = "aarch64")]
    #[repr(C, align(16))]
    struct FpsimdState {
        v: [u128; 32],
        fpcr: u64,
        fpsr: u64,
    }
    // Pin the field offsets the inline asm hardcodes (#0..#520).
    // The save/restore asm writes/reads at these literal byte
    // offsets; if `repr(C)` layout ever drifts (e.g. a future
    // u128 alignment bump or padding insertion) the asserts here
    // fire at compile time instead of producing silent
    // wrong-register reads.
    #[cfg(target_arch = "aarch64")]
    const _: () = {
        assert!(std::mem::offset_of!(FpsimdState, v) == 0);
        assert!(std::mem::offset_of!(FpsimdState, fpcr) == 512);
        assert!(std::mem::offset_of!(FpsimdState, fpsr) == 520);
        assert!(std::mem::align_of::<FpsimdState>() == 16);
    };
    #[cfg(target_arch = "aarch64")]
    struct CpuStateGuard {
        // None when the host CPU lacks FEAT_FP. AArch64 Linux
        // requires FPSIMD in practice, but
        // `is_aarch64_feature_detected!("fp")` keeps us correct on
        // any future trimmed-down host configuration.
        fpsimd: Option<Box<FpsimdState>>,
        sigmask: libc::sigset_t,
        sigrtmin_action: libc::sigaction,
    }
    #[cfg(target_arch = "aarch64")]
    impl Drop for CpuStateGuard {
        fn drop(&mut self) {
            unsafe {
                if let Some(fp) = self.fpsimd.as_deref() {
                    let ptr = fp as *const FpsimdState as *const u8;
                    // Restore V0..V31 (32 * 16 bytes) then FPCR
                    // and FPSR. Keeping the load order identical
                    // to the save order means a partial restore
                    // (e.g. on instruction trap mid-block) leaves
                    // the same prefix coherent rather than mixing
                    // V regs with stale FPCR/FPSR.
                    //
                    // The 32 `out("vN") _` declarations tell the
                    // compiler every V register is clobbered after
                    // the asm; without them V8..V15 (the bottom-64
                    // callee-saved slice per AAPCS64) could carry
                    // a compiler-managed value across the asm and
                    // be silently overwritten with our saved
                    // value. FPCR/FPSR have no Rust-level register
                    // class, so writing them via `msr` is opaque
                    // to the compiler — that is the desired
                    // behavior here (the registers control IEEE
                    // rounding/exception state, not register
                    // liveness).
                    core::arch::asm!(
                        "ldp q0, q1, [{p}, #0]",
                        "ldp q2, q3, [{p}, #32]",
                        "ldp q4, q5, [{p}, #64]",
                        "ldp q6, q7, [{p}, #96]",
                        "ldp q8, q9, [{p}, #128]",
                        "ldp q10, q11, [{p}, #160]",
                        "ldp q12, q13, [{p}, #192]",
                        "ldp q14, q15, [{p}, #224]",
                        "ldp q16, q17, [{p}, #256]",
                        "ldp q18, q19, [{p}, #288]",
                        "ldp q20, q21, [{p}, #320]",
                        "ldp q22, q23, [{p}, #352]",
                        "ldp q24, q25, [{p}, #384]",
                        "ldp q26, q27, [{p}, #416]",
                        "ldp q28, q29, [{p}, #448]",
                        "ldp q30, q31, [{p}, #480]",
                        "ldr {tmp}, [{p}, #512]",
                        "msr FPCR, {tmp}",
                        "ldr {tmp}, [{p}, #520]",
                        "msr FPSR, {tmp}",
                        p = in(reg) ptr,
                        tmp = out(reg) _,
                        out("v0") _, out("v1") _, out("v2") _, out("v3") _,
                        out("v4") _, out("v5") _, out("v6") _, out("v7") _,
                        out("v8") _, out("v9") _, out("v10") _, out("v11") _,
                        out("v12") _, out("v13") _, out("v14") _, out("v15") _,
                        out("v16") _, out("v17") _, out("v18") _, out("v19") _,
                        out("v20") _, out("v21") _, out("v22") _, out("v23") _,
                        out("v24") _, out("v25") _, out("v26") _, out("v27") _,
                        out("v28") _, out("v29") _, out("v30") _, out("v31") _,
                        options(nostack, readonly),
                    );
                }
                libc::pthread_sigmask(libc::SIG_SETMASK, &self.sigmask, std::ptr::null_mut());
                libc::sigaction(
                    libc::SIGRTMIN(),
                    &self.sigrtmin_action,
                    std::ptr::null_mut(),
                );
            }
        }
    }
    // sigmask + SIGRTMIN save/restore is arch-independent — covers
    // any host arch outside the x86_64 + aarch64 set the build
    // expects (no compile_error guard local to this file).
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    struct CpuStateGuard {
        sigmask: libc::sigset_t,
        sigrtmin_action: libc::sigaction,
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    impl Drop for CpuStateGuard {
        fn drop(&mut self) {
            unsafe {
                libc::pthread_sigmask(libc::SIG_SETMASK, &self.sigmask, std::ptr::null_mut());
                libc::sigaction(
                    libc::SIGRTMIN(),
                    &self.sigrtmin_action,
                    std::ptr::null_mut(),
                );
            }
        }
    }
    #[cfg(target_arch = "x86_64")]
    let _cpu_guard = unsafe {
        let has_xsave = std::arch::is_x86_feature_detected!("xsave");
        // CPUID leaf 7, subleaf 0: EBX bit 0 = FSGSBASE
        let has_fsgsbase = core::arch::x86_64::__cpuid_count(7, 0).ebx & 1 != 0;
        // CPUID leaf 7, subleaf 0: ECX bit 3 = PKU (OSPKE)
        // PKU (bit 3) AND OSPKE (bit 4) — both required. OSPKE
        // indicates the OS enabled CR4.PKE; without it rdpkru/wrpkru
        // raise #UD even on PKU-capable hardware.
        let cpuid7 = core::arch::x86_64::__cpuid_count(7, 0);
        let has_pku = (cpuid7.ecx & (1 << 3)) != 0 && (cpuid7.ecx & (1 << 4)) != 0;

        let mut fsbase: u64 = 0;
        let mut gsbase: u64 = 0;
        if has_fsgsbase {
            core::arch::asm!("rdfsbase {}", out(reg) fsbase, options(nostack));
            core::arch::asm!("rdgsbase {}", out(reg) gsbase, options(nostack));
        }
        let mut pkru: u32 = 0;
        if has_pku {
            core::arch::asm!(
                "xor ecx, ecx", "rdpkru",
                out("eax") pkru, out("ecx") _, out("edx") _,
                options(nostack),
            );
        }
        // Size the XSAVE buffer from CPUID.0Dh.0:EBX (max size for
        // currently-enabled features). Minimum 16384 as defensive
        // floor — covers AVX-512 + AMX TILEDATA on Sapphire Rapids+.
        let xsave_size = if has_xsave {
            let cpuid = core::arch::x86_64::__cpuid_count(0xD, 0);
            (cpuid.ebx as usize).max(16384)
        } else {
            0
        };
        let mut xsave_buf = vec![0u8; xsave_size + 64];
        let align_ptr = ((xsave_buf.as_mut_ptr() as usize + 63) & !63) as *mut u8;
        if has_xsave {
            core::arch::asm!(
                "xsave [{}]",
                in(reg) align_ptr,
                in("eax") 0xFFFF_FFFFu32,
                in("edx") 0xFFFF_FFFFu32,
                options(nostack),
            );
        }
        let mut sigmask: libc::sigset_t = std::mem::zeroed();
        libc::pthread_sigmask(libc::SIG_SETMASK, std::ptr::null(), &mut sigmask);
        let mut sigrtmin_action: libc::sigaction = std::mem::zeroed();
        libc::sigaction(libc::SIGRTMIN(), std::ptr::null(), &mut sigrtmin_action);
        CpuStateGuard {
            xsave_buf,
            align_ptr,
            has_xsave,
            has_fsgsbase,
            has_pku,
            fsbase,
            gsbase,
            pkru,
            sigmask,
            sigrtmin_action,
        }
    };
    #[cfg(target_arch = "aarch64")]
    let _cpu_guard = unsafe {
        // FEAT_FP is implied by FEAT_AdvSIMD ("neon") and is
        // mandatory on every AArch64 Linux kernel observed in
        // practice; the runtime check survives a hypothetical
        // FP-stripped host without raising SIGILL on the `mrs`
        // and `stp q*` instructions below.
        let fpsimd = if std::arch::is_aarch64_feature_detected!("fp") {
            let mut state = Box::new(FpsimdState {
                v: [0u128; 32],
                fpcr: 0,
                fpsr: 0,
            });
            let ptr = state.as_mut() as *mut FpsimdState as *mut u8;
            // Save V0..V31 (32 * 16 bytes), then FPCR and FPSR.
            // The same offsets the Drop path's `ldp`/`ldr` chain
            // reads back from. `mrs` of FPCR/FPSR writes the full
            // 64-bit destination with the upper bits architectural
            // RES0, so the u64 slot captures the entire defined
            // state.
            core::arch::asm!(
                "stp q0, q1, [{p}, #0]",
                "stp q2, q3, [{p}, #32]",
                "stp q4, q5, [{p}, #64]",
                "stp q6, q7, [{p}, #96]",
                "stp q8, q9, [{p}, #128]",
                "stp q10, q11, [{p}, #160]",
                "stp q12, q13, [{p}, #192]",
                "stp q14, q15, [{p}, #224]",
                "stp q16, q17, [{p}, #256]",
                "stp q18, q19, [{p}, #288]",
                "stp q20, q21, [{p}, #320]",
                "stp q22, q23, [{p}, #352]",
                "stp q24, q25, [{p}, #384]",
                "stp q26, q27, [{p}, #416]",
                "stp q28, q29, [{p}, #448]",
                "stp q30, q31, [{p}, #480]",
                "mrs {tmp}, FPCR",
                "str {tmp}, [{p}, #512]",
                "mrs {tmp}, FPSR",
                "str {tmp}, [{p}, #520]",
                p = in(reg) ptr,
                tmp = out(reg) _,
                options(nostack),
            );
            Some(state)
        } else {
            None
        };
        let mut sigmask: libc::sigset_t = std::mem::zeroed();
        libc::pthread_sigmask(libc::SIG_SETMASK, std::ptr::null(), &mut sigmask);
        let mut sigrtmin_action: libc::sigaction = std::mem::zeroed();
        libc::sigaction(libc::SIGRTMIN(), std::ptr::null(), &mut sigrtmin_action);
        CpuStateGuard {
            fpsimd,
            sigmask,
            sigrtmin_action,
        }
    };
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let _cpu_guard = unsafe {
        let mut sigmask: libc::sigset_t = std::mem::zeroed();
        libc::pthread_sigmask(libc::SIG_SETMASK, std::ptr::null(), &mut sigmask);
        let mut sigrtmin_action: libc::sigaction = std::mem::zeroed();
        libc::sigaction(libc::SIGRTMIN(), std::ptr::null(), &mut sigrtmin_action);
        CpuStateGuard {
            sigmask,
            sigrtmin_action,
        }
    };

    #[cfg(feature = "wprof")]
    if entry.wprof {
        let sidecar = crate::test_support::sidecar_dir();
        let _ = std::fs::remove_file(sidecar.join(format!("{}.wprof.pb", entry.name)));
        let _ = std::fs::remove_file(sidecar.join(format!("{}.repro.wprof.pb", entry.name)));
    }

    let vm = match builder.build() {
        Ok(vm) => vm,
        Err(e) => {
            if e.downcast_ref::<crate::vmm::host_topology::ResourceContention>()
                .is_some()
            {
                record_skip_sidecar(entry);
            }
            return Err(e.context("build ktstr_test VM"));
        }
    };
    let mut result = match vm.run() {
        Ok(r) => r,
        Err(e) => {
            if e.downcast_ref::<crate::vmm::host_topology::ResourceContention>()
                .is_some()
            {
                record_skip_sidecar(entry);
            }
            return Err(e.context("run ktstr_test VM"));
        }
    };
    // Stamp the macro-emitted test fn name onto the VmResult so
    // post_vm callbacks can derive per-test sidecar paths via
    // `result.wprof_pb_path()` / `result.repro_wprof_pb_path()`
    // instead of hardcoding a fn-name literal that drifts when the
    // test is renamed. The `entry.name` field is the `&'static str`
    // the proc-macro emitted; `crate::vmm::VmResult::entry_name` is
    // `Option<&'static str>` so manually-constructed fixtures bail
    // loud when the derivation methods are called without a stamped
    // name.
    result.entry_name = Some(entry.name);
    // When the primary VM failed but the freeze coordinator never
    // wrote the real failure-dump (i.e. pre-attach failures:
    // send_sys_rdy timeout, VM boot failure, scheduler binary load
    // failure — anything that returns before the BPF probe attaches
    // and the freeze-coord's err_triggered watchpoint can fire),
    // emit a placeholder dump at the documented path so operators
    // querying `find target/ktstr -name '*.failure-dump.json'` always
    // see SOMETHING when a test failed. Spec-promise parity: the
    // sidecar dir guarantees one `<test_name>.failure-dump.json`
    // per failed run regardless of whether real BPF state could be
    // captured. The `is_placeholder: true` field on
    // [`crate::monitor::dump::FailureDumpReport`] lets downstream
    // tooling (stats compare, sidecar walkers) distinguish a stub
    // from a real BPF dump.
    if !result.success {
        write_placeholder_failure_dump_if_missing(&primary_dump_path, &result);
    }

    // Run the test entry's optional host-side post_vm callbacks
    // — TWO independently-dispatched slots: `post_vm` (suppressed
    // on guest-fail) and `post_vm_unconditional` (always runs).
    // The `#[ktstr_test]` scenario function (`entry.func`) runs
    // INSIDE the guest VM and cannot read host-side state — most
    // notably `VmResult.snapshot_bridge`, which the freeze
    // coordinator populates on every `Op::CaptureSnapshot` /
    // `Op::WatchSnapshot` fire. Both post_vm hooks close that
    // gap: they run on the HOST after `vm.run()` returns, with
    // direct access to the full `VmResult`.
    //
    // Suppression contract (applies to `post_vm` only;
    // `post_vm_unconditional` bypasses this gate entirely): skip
    // post_vm when the guest already reported a hard `Fail`
    // `AssertResult` (e.g. `sched_died_during_hold` recorded
    // mid-step at `src/scenario/ops/mod.rs::966`). The host's
    // typical post_vm body asserts on workload-derived state
    // (`snapshot_bridge`, `periodic_fired`, …) — that state is
    // structurally missing when the scheduler died before the
    // workload reached the asserted-on phase, so the post_vm
    // `Err` would surface as a misleading "snapshot bridge
    // captured nothing" wrapper that obscures the actual
    // scheduler crash sitting in `--- scheduler log ---`. The
    // guest-side failure already cascades through
    // `evaluate_vm_result` with the right diagnostic; running
    // post_vm here adds noise, not signal. Skip and Inconclusive
    // guest outcomes deliberately fall through to run post_vm —
    // both leave room for the host check to add evidence the
    // guest did not collect.
    //
    // `post_vm_unconditional` is for the inverse case: a
    // host-side artifact (e.g. a `.repro.wprof.pb` written by
    // the auto-repro VM) that exists EVEN WHEN the guest
    // reported a fail — the operator wants the artifact-shape
    // check regardless of the guest-side outcome. The callback
    // owns its own skip-on-crash logic.
    //
    // The combined `Err` (via `combine_post_vm_errs`) is
    // threaded into `evaluate_vm_result` below so the post_vm
    // failure(s) flow through the SAME failure path as a
    // guest-side fail (same `--- scheduler log ---`,
    // `--- sched_ext dump ---`, `--- monitor ---`, `stage:`
    // sections; same auto-repro dispatch). When BOTH callbacks
    // fail, the combined message names both errors so a
    // debugging operator sees both regressions on the first pass.
    #[cfg(feature = "wprof")]
    if let Some(ref bulk) = result.guest_messages {
        for bulk_entry in &bulk.entries {
            if crate::vmm::wire::MsgType::from_wire(bulk_entry.msg_type)
                == Some(crate::vmm::wire::MsgType::WprofTrace)
                && bulk_entry.crc_ok
                && !bulk_entry.payload.is_empty()
            {
                let wprof_path =
                    crate::test_support::sidecar_dir().join(format!("{}.wprof.pb", entry.name));
                if let Err(e) = std::fs::create_dir_all(
                    wprof_path
                        .parent()
                        .expect("sidecar_dir join always has parent"),
                ) {
                    eprintln!("ktstr_test: create sidecar dir for wprof trace: {e}");
                } else if let Err(e) = std::fs::write(&wprof_path, &bulk_entry.payload) {
                    eprintln!(
                        "ktstr_test: write wprof trace to {}: {e}",
                        wprof_path.display()
                    );
                }
            }
        }
    }

    let guest_already_failed = parse_assert_result_from_drain(result.guest_messages.as_ref())
        .map(|r| r.is_fail())
        .unwrap_or(false);
    let post_vm_err = run_post_vm_callbacks(entry, &result, guest_already_failed);
    if post_vm_err.is_some() {
        // post_vm failure: the guest itself may have returned
        // result.success=true, but the host-side check overrules
        // it as a failure. Emit the placeholder dump too (if not
        // already present from the earlier !result.success path)
        // so spec-promise parity holds even when the failure mode
        // is host-side rather than guest-side.
        write_placeholder_failure_dump_if_missing(&primary_dump_path, &result);
    }

    // Release VM resources (CPU/LLC flocks, guest memory) before
    // the multi-second LLM inference so concurrent peers can
    // acquire the same LLC slots during extraction.
    let post_vm_t = std::time::Instant::now();
    drop(vm);
    // Release the kernel-cache reader flock — the VM no longer
    // maps the kernel image, so concurrent `cargo ktstr kernel
    // build` can proceed.
    drop(kernel_lock);

    // Broaden the calling thread's CPU mask before
    // `host_side_llm_extract` runs. After `vm.run()` the
    // BSP / vCPU 0 thread carries either:
    //   - a single-CPU pin (perf-mode path: the vmm `vm.run`
    //     entry calls `pin_current_thread` to nail the BSP to one
    //     CPU for ipi-latency stability) — LLM inference on 1 CPU
    //     is dramatically slow (10x+ in throughput) and gives the
    //     other free host CPUs no work, OR
    //   - a multi-CPU LLC-aware mask (no-perf-mode path: the vmm
    //     applies `set_thread_cpumask` against the
    //     `no_perf_plan.cpus` set so the BSP can roam within an
    //     LLC) — already pool-style, but narrower than the
    //     host-allowed cpuset.
    // Inference is a host-side post-VM-exit phase that doesn't
    // share the VM's measurement contract; it should use whatever
    // CPUs the host process is permitted on (cgroup cpuset / sudo
    // -u limits / CI runner allocation), which is exactly what
    // `host_allowed_cpus()` returns via `sched_getaffinity(0)`.
    // Use no-perf-mode cpuset for inference: `set_thread_cpumask`
    // against the broader host-allowed pool is the no-perf-mode
    // primitive applied to a wider set than any single LLC-plan
    // would carve out.
    //
    // Empty `host_allowed_cpus()` (sched_getaffinity unavailable,
    // procfs fallback failed) skips the call rather than masking
    // to zero CPUs (which would block forever); inference inherits
    // whatever the test left behind. Logged as a warning by
    // `set_thread_cpumask` itself if the syscall fails.
    let host_cpus = crate::vmm::host_topology::host_allowed_cpus();
    if !host_cpus.is_empty() {
        crate::vmm::set_thread_cpumask(&host_cpus, "test");
    }

    // Log verifier stats count for visibility.
    if !result.verifier_stats.is_empty() {
        eprintln!(
            "ktstr_test: verifier_stats: {} struct_ops programs",
            result.verifier_stats.len(),
        );
    }

    // When running with a struct_ops scheduler, check that host-side
    // BPF program enumeration found programs with non-zero verified_insns.
    // Gated on has_bpf_scheduler (excludes KernelBuiltin) — verifier_stats
    // are a BPF-loader artifact, so KernelBuiltin tests have legitimately
    // empty verifier_stats and would otherwise trip this warning on every
    // run.
    if entry.scheduler.has_bpf_scheduler() && result.success && result.verifier_stats.is_empty() {
        eprintln!("ktstr_test: WARNING: scheduler loaded but verifier_stats is empty");
    }

    // Extract profraw from SHM ring buffer and collect stimulus
    // events + per-payload metrics + raw outputs from
    // `OutputFormat::LlmExtract` payloads.
    //
    // Pairing contract: every guest-side payload-pipeline emit
    // (one per `.run()` / `.wait()` / `.kill()` / `.try_wait()`
    // terminal call) allocates one `payload_index` from
    // `payload_run`'s per-process counter and stamps it onto the
    // emitted `PayloadMetrics`. LlmExtract invocations additionally
    // emit a `RawPayloadOutput` carrying the SAME index. Non-
    // LlmExtract payloads emit only the `PayloadMetrics`. The host
    // pairs an LlmExtract `RawPayloadOutput` to its empty-metrics
    // companion by EQUAL `payload_index`, not by emission order —
    // see `host_side_llm_extract` for the pairing implementation.
    let mut stimulus_events = Vec::new();
    let mut payload_metrics: Vec<crate::test_support::PayloadMetrics> = Vec::new();
    let mut raw_outputs: Vec<crate::test_support::RawPayloadOutput> = Vec::new();
    if let Some(ref bulk) = result.guest_messages {
        // Per-frame typed dispatch on the bucketed bulk drain.
        // Mirrors the freeze coord's TOKEN_TX exhaustive match —
        // adding a new MsgType variant is a compile error here, so
        // the host either decodes it explicitly or silently
        // ignores it via the catch-all arm with intent. CRC
        // failures gate decode for every variant whose semantics
        // depend on payload integrity.
        for bulk_entry in &bulk.entries {
            let kind = crate::vmm::wire::MsgType::from_wire(bulk_entry.msg_type);
            match kind {
                Some(crate::vmm::wire::MsgType::Profraw) => {
                    if bulk_entry.crc_ok
                        && !bulk_entry.payload.is_empty()
                        && let Err(e) = write_profraw(&bulk_entry.payload)
                    {
                        eprintln!("ktstr_test: write guest profraw: {e}");
                    }
                }
                Some(crate::vmm::wire::MsgType::WprofTrace) => {
                    // wprof Perfetto trace captured during the
                    // guest's auto-repro window. Write next to the
                    // failure-dump JSON so the operator finds both
                    // in the same per-test directory. NOTE: the
                    // pre-pass above already wrote this file before
                    // post_vm fired; this arm rewrites the same
                    // bytes to the same path (idempotent) so that
                    // a future bulk-drain consumer added here keeps
                    // the WprofTrace handling colocated with the
                    // rest of MsgType dispatch.
                    if bulk_entry.crc_ok && !bulk_entry.payload.is_empty() {
                        let wprof_path = crate::test_support::sidecar_dir()
                            .join(format!("{}.wprof.pb", entry.name));
                        if let Err(e) = std::fs::create_dir_all(
                            wprof_path
                                .parent()
                                .expect("sidecar_dir join always has parent"),
                        ) {
                            eprintln!("ktstr_test: create sidecar dir for wprof trace: {e}");
                        } else if let Err(e) = std::fs::write(&wprof_path, &bulk_entry.payload) {
                            eprintln!(
                                "ktstr_test: write wprof trace to {}: {e}",
                                wprof_path.display()
                            );
                        }
                    }
                }
                Some(crate::vmm::wire::MsgType::Stimulus) => {
                    if bulk_entry.crc_ok
                        && let Some(ev) =
                            crate::vmm::wire::StimulusEvent::from_payload(&bulk_entry.payload)
                    {
                        stimulus_events.push(crate::timeline::StimulusEvent::from_wire(&ev));
                    }
                }
                Some(crate::vmm::wire::MsgType::PayloadMetrics) => {
                    if bulk_entry.crc_ok {
                        match postcard::from_bytes::<crate::test_support::PayloadMetrics>(
                            &bulk_entry.payload,
                        ) {
                            Ok(pm) => payload_metrics.push(pm),
                            Err(e) => {
                                eprintln!("ktstr_test: decode payload metrics from bulk port: {e}")
                            }
                        }
                    }
                }
                Some(crate::vmm::wire::MsgType::RawPayloadOutput) => {
                    if bulk_entry.crc_ok {
                        match postcard::from_bytes::<crate::test_support::RawPayloadOutput>(
                            &bulk_entry.payload,
                        ) {
                            Ok(raw) => raw_outputs.push(raw),
                            Err(e) => eprintln!(
                                "ktstr_test: decode raw payload output from bulk port: {e}"
                            ),
                        }
                    }
                }
                // The remaining verdict-bearing variants
                // (TestResult, Exit, SchedExit, ScenarioStart,
                // ScenarioEnd, Stdout, Stderr, SchedLog, Lifecycle,
                // ExecExit, Dmesg, ProbeOutput, SnapshotReply,
                // Crash) are consumed by other walkers further down
                // the pipeline (parse_assert_result_from_drain,
                // bulk_exit lookup in collect_results, lifecycle
                // classifier, sched_log concatenator, etc.). No
                // per-entry side effect here.
                Some(
                    crate::vmm::wire::MsgType::TestResult
                    | crate::vmm::wire::MsgType::Exit
                    | crate::vmm::wire::MsgType::SchedExit
                    | crate::vmm::wire::MsgType::ScenarioStart
                    | crate::vmm::wire::MsgType::ScenarioEnd
                    | crate::vmm::wire::MsgType::ScenarioPause
                    | crate::vmm::wire::MsgType::ScenarioResume
                    | crate::vmm::wire::MsgType::Stdout
                    | crate::vmm::wire::MsgType::SchedLog
                    | crate::vmm::wire::MsgType::Lifecycle
                    | crate::vmm::wire::MsgType::ExecExit
                    | crate::vmm::wire::MsgType::Dmesg
                    | crate::vmm::wire::MsgType::ProbeOutput
                    | crate::vmm::wire::MsgType::SnapshotReply
                    | crate::vmm::wire::MsgType::Crash,
                ) => {}
                Some(crate::vmm::wire::MsgType::Stderr) => {
                    if bulk_entry.crc_ok && !bulk_entry.payload.is_empty() {
                        eprint!("GUEST: {}", String::from_utf8_lossy(&bulk_entry.payload));
                    }
                }
                // Coordinator-internal frames are stripped before
                // they reach this loop (see the `is_coordinator_internal`
                // filter in `collect_results`'s post-run drain plus
                // the freeze coord's mid-run TOKEN_TX bucketing).
                // Defensively no-op rather than panic so a future
                // is_coordinator_internal expansion does not
                // require a parallel update here.
                Some(crate::vmm::wire::MsgType::SnapshotRequest)
                | Some(crate::vmm::wire::MsgType::KernelOpRequest)
                | Some(crate::vmm::wire::MsgType::KernelOpReply)
                | Some(crate::vmm::wire::MsgType::SysRdy) => {}
                None => {
                    tracing::warn!(
                        msg_type = bulk_entry.msg_type,
                        len = bulk_entry.payload.len(),
                        crc_ok = bulk_entry.crc_ok,
                        "ktstr_test: unknown MSG_TYPE_* on bulk port; dropping"
                    );
                }
            }
        }
    }

    // Host-side `OutputFormat::LlmExtract` resolution. For every
    // RawPayloadOutput drained from the bulk port, look up its
    // `payload_index` in the PayloadMetrics slice, run the
    // LLM-backed extraction on the host, and replace the empty
    // `metrics` vec on the matched slot with the extracted result.
    // The model lives at the host's cache and the guest VM never
    // had it, so this is the only correct place for the call.
    //
    // Pairing is by explicit `payload_index` equality, not emission
    // order — emission order would conflate a `Json` payload that
    // produced zero numeric leaves with an LlmExtract placeholder.
    // Returns a flat `Vec<AssertDetail>` of host-side failures
    // (model unavailable, universal invariant violation, orphan
    // raw outputs) for the test verdict to fold in.
    #[cfg(feature = "llm")]
    let host_extract_failures = host_side_llm_extract(&mut payload_metrics, &raw_outputs);
    #[cfg(not(feature = "llm"))]
    let host_extract_failures: Vec<crate::assert::AssertDetail> = Vec::new();

    // Gate-skip on a post_vm `HostSkipRequest`: a host-side callback
    // determined the run is inconclusive (the VM could not produce the
    // artifact the assertion needs — e.g. a load-starved VM whose BPF
    // probe never attached, leaving a placeholder failure dump). The
    // marker survives `combine_post_vm_errs` only when no sibling
    // callback reported a genuine failure, so a real regression is
    // never masked. Detected here, ahead of the eval / auto-repro
    // path, so the run skips rather than failing.
    if let Some(err) = &post_vm_err
        && err.downcast_ref::<HostSkipRequest>().is_some()
    {
        let reason = format!("{err:#}");
        crate::report::test_skip(format_args!("{}: {}", entry.name, reason));
        record_skip_sidecar(entry);
        return Ok(AssertResult::skip(reason));
    }

    // Gate-skip on an unloadable host LLM model. The model lives
    // host-side (it is too large to ship into the guest) and is a hard
    // prerequisite for any LlmExtract payload: when it cannot load
    // (cold-cache offline, or a cached GGUF incompatible with the linked
    // llama.cpp), the extraction never runs and the test's metrics are
    // unfulfillable. The prior code folded this into the verdict as a
    // failure, failing the whole test on a missing prereq; convert that
    // same whole-test path to a SKIP so the suite passes where a
    // compatible model is present and skips where it is not. Re-fetching
    // cannot help — the incompatibility is with the linked llama.cpp, not
    // a stale download.
    // The skip-vs-fail decision — including the rule that a host-side
    // post_vm failure DOMINATES (no skip), so a real regression is never
    // masked by a missing model — is should_skip_on_llm_model_load_failure
    // (truth-tabled). A post_vm Err is folded into the verdict below and
    // carries the PostVmAssertionFailure marker downstream.
    if let Some(skip_reason) =
        should_skip_on_llm_model_load_failure(&host_extract_failures, post_vm_err.is_some())
    {
        crate::report::test_skip(format_args!("{}: {}", entry.name, skip_reason));
        record_skip_sidecar(entry);
        return Ok(AssertResult::skip(skip_reason));
    }

    // auto_repro is enabled when:
    // - entry.auto_repro is true (default)
    // - a scheduler is running (not EEVDF)
    // - the test does not expect failure (expect_err = false)
    let effective_auto_repro = entry.auto_repro && scheduler.is_some() && !entry.expect_err;
    // Only the BPF-latched `ktstr_exit_kind_snap` (persisted via the
    // failure-dump JSON's `scx_sched_state.exit_kind`) is authoritative
    // for the stall vs non-stall distinction. The downstream auto-repro
    // path uses this exclusively to gate whether to skip probe
    // attachment in the repro VM — a misclassification suppresses
    // exactly the diagnostic the user runs ktstr to get.
    //
    // The previous text-parsing fallbacks (`parse_kmsg_window` returning
    // `ScxExitKind::Stall`, and the literal `"runnable task stall"`
    // substring match in the sched_ext dump) misclassified non-stall
    // errors whose sched_ext dump happened to contain stalled tasks at
    // exit time — flipping `is_stall` to true and silencing the
    // annotated stall the BPF probe would have produced in the repro
    // VM. Dropped both fallbacks: if the dump JSON is missing,
    // `primary_exit_kind` stays `None` and the auto-repro path falls
    // through to running the probe (the conservative choice — probe
    // attachment on a true stall is a slowdown, not a correctness
    // hazard).
    let primary_exit_kind = {
        let dump_path =
            super::sidecar::sidecar_dir().join(format!("{}.failure-dump.json", entry.name));
        std::fs::read_to_string(&dump_path)
            .ok()
            .and_then(|json| serde_json::from_str::<serde_json::Value>(&json).ok())
            .and_then(|v| {
                v.get("scx_sched_state")
                    .and_then(|s| s.get("exit_kind"))
                    .and_then(|k| k.as_u64())
            })
    };
    // Did the primary VM emit its `PayloadStarting` lifecycle frame?
    // Computed before constructing repro_fn so the closure can capture
    // it. The flag gates the "PRIMARY DID NOT REACH WORKLOAD" label
    // on the auto-repro verdict — see
    // `label_repro_verdict_when_workload_not_reached` in probe.rs.
    let primary_reached_workload =
        crate::test_support::output::primary_reached_workload(result.guest_messages.as_ref());
    let repro_fn = |output: &str| -> Option<String> {
        if !effective_auto_repro {
            return None;
        }
        let repro = attempt_auto_repro(
            entry,
            &kernel,
            scheduler.as_deref(),
            &ktstr_bin,
            output,
            &result.stderr,
            topo,
            primary_exit_kind,
            primary_reached_workload,
        );
        // When auto-repro was attempted but produced no data, return a
        // diagnostic so the user knows it was tried.
        Some(repro.unwrap_or_else(|| {
            "auto-repro: no probe data — the scheduler may have \
             exited before probes could capture events, or the \
             crash did not reproduce in the repro VM. Re-run with \
             RUST_LOG=debug for probe pipeline diagnostics. Check \
             the sched_ext dump and scheduler log sections above \
             for crash details."
                .to_string()
        }))
    };

    eprintln!("post-VM overhead before eval: {:?}", post_vm_t.elapsed());
    let eval_result = evaluate_vm_result(
        entry,
        &result,
        &merged_assert,
        &stimulus_events,
        &payload_metrics,
        &host_extract_failures,
        &vm_topology,
        &repro_fn,
        post_vm_err.as_ref(),
    );
    // Set result.expect_auto_repro_satisfied based on the artifact-on-disk
    // probe. Called AFTER evaluate_vm_result so the auto-repro VM has had
    // a chance to land its .repro.wprof.pb artifact via the host's
    // MsgType::WprofTrace dispatch arm. No-op when expect_auto_repro is
    // unset or the test passed on its own.
    apply_expect_auto_repro_inversion(entry, &mut result);
    // When the helper signaled satisfaction, attach the
    // [`ExpectAutoReproSatisfied`] marker to the failure chain so
    // `result_to_exit_code` routes the verdict to `EXIT_PASS`
    // without mutating the underlying `AssertResult` — the
    // original failure detail still surfaces in stderr/dump
    // rendering. The marker rides as `anyhow::Context` (matches
    // the [`ScxBpfErrorMatcherMismatch`] precedent above) so
    // `downcast_ref::<...>().is_some()` at the dispatch arm finds the
    // marker through anyhow's context-aware chain walk (a raw
    // `e.chain().any(|c| c.is::<...>())` would MISS a context-attached
    // marker — anyhow boxes it as `ContextError<C, E>`).
    let eval_result = if result.expect_auto_repro_satisfied {
        eval_result.map_err(|e| e.context(ExpectAutoReproSatisfied))
    } else {
        eval_result
    };
    // When a host-side post_vm / post_vm_unconditional callback
    // returned Err, attach the [`PostVmAssertionFailure`] marker so
    // result_to_exit_code refuses to invert the verdict under
    // expect_err: a host-side assertion failure is a real regression
    // and must surface even when the guest-side failure it accompanies
    // is "expected". evaluate_vm_result already folded the post_vm Err
    // into the failure message (Other detail / message prefix), so the
    // marker governs only the inversion decision, not message content.
    // Mirrors the ScxBpfErrorMatcherMismatch precedent.
    let eval_result = if post_vm_err.is_some() {
        eval_result.map_err(|e| e.context(PostVmAssertionFailure))
    } else {
        eval_result
    };
    eprintln!(
        "evaluate_vm_result (includes auto-repro): {:?}",
        post_vm_t.elapsed()
    );
    eval_result
}

/// Set `result.expect_auto_repro_satisfied` when the
/// `expect_auto_repro = true` assertion is satisfied: the test
/// failed AND a valid `.repro.wprof.pb` artifact landed on disk.
///
/// Called AFTER `evaluate_vm_result` returns so the artifact's
/// presence is observable. When the field is set, the caller
/// (`run_ktstr_test_inner_impl`) wraps any failure `Err` with the
/// [`ExpectAutoReproSatisfied`] marker; the dispatch arm
/// (`crate::test_support::dispatch::result_to_exit_code`)
/// downcasts the marker and routes the verdict to `EXIT_PASS`
/// WITHOUT mutating `result.success` or stripping the error chain
/// (preserves diagnostic visibility — the original fail trail is
/// available for stderr/dump rendering).
///
/// Uses `crate::test_support::wprof::assert_wprof_pb_shape` for
/// the artifact-on-disk probe — closes the partial-write race that
/// a bare `path.exists()` check would miss: the shape validator
/// only signals satisfied when the file has reached its minimum
/// shape-valid size + leading byte. A regression where wprof
/// crashes mid-write produces a non-shape-valid artifact and the
/// assertion correctly does NOT cause inversion.
///
/// No-op when `entry.expect_auto_repro = false` (preserves prior
/// behavior — `result.expect_auto_repro_satisfied` stays `false`,
/// the dispatch arm sees no inversion signal). No-op when
/// `result.success = true` (test passed on its own; nothing to
/// invert).
pub(crate) fn apply_expect_auto_repro_inversion(
    entry: &KtstrTestEntry,
    result: &mut vmm::VmResult,
) {
    // The inversion is wprof-only: it inspects the repro `.wprof.pb`
    // shape, which exists only under the `wprof` feature. Without it
    // this is a no-op, so `entry` and `result` go unused.
    #[cfg(not(feature = "wprof"))]
    let _ = (entry, result);
    #[cfg(feature = "wprof")]
    {
        if !entry.expect_auto_repro {
            return;
        }
        if result.success {
            return;
        }
        let Ok(repro_path) = result.repro_wprof_pb_path() else {
            return;
        };
        if crate::test_support::wprof::assert_wprof_pb_shape(&repro_path).is_ok() {
            result.expect_auto_repro_satisfied = true;
        }
    }
}

/// Evaluate a VM result and produce the appropriate error or Ok.
///
/// This is the core result-evaluation logic, extracted from
/// `run_ktstr_test_inner` so that error message formatting can be tested
/// without booting a VM. The `repro_fn` callback handles auto-repro
/// (which requires a second VM boot) when provided. `payload_metrics`
/// is the per-invocation accumulator drained from the guest SHM ring;
/// the sidecar writer receives it verbatim so stats tooling sees one
/// entry per `ctx.payload(X).run()` / `.spawn().wait()`.
///
/// `host_extract_failures` carries the universal-invariant +
/// model-load failures produced by [`host_side_llm_extract`] when
/// the run's `OutputFormat::LlmExtract` payloads were resolved on
/// the host. The folded details are appended to the test's
/// AssertResult so a host-side LlmExtract failure surfaces in the
/// same failure-rendering pipeline as a guest-emitted check failure.
#[allow(clippy::too_many_arguments)]
fn evaluate_vm_result(
    entry: &KtstrTestEntry,
    result: &vmm::VmResult,
    merged_assert: &crate::assert::Assert,
    stimulus_events: &[StimulusEvent],
    payload_metrics: &[crate::test_support::PayloadMetrics],
    host_extract_failures: &[crate::assert::AssertDetail],
    topo: &Topology,
    repro_fn: &dyn Fn(&str) -> Option<String>,
    // Optional Err captured from `KtstrTestEntry::post_vm` so the
    // host-side assertion failure flows through the SAME failure
    // path as a guest-side `result.success=false`: same scheduler
    // log / sched_ext dump / monitor diagnostic, same auto-repro
    // dispatch. Folded into `check_result` as an AssertDetail
    // below the parse-success arm so the existing failure-message
    // construction picks it up without a parallel renderer.
    post_vm_err: Option<&anyhow::Error>,
) -> Result<AssertResult> {
    // Build phase buckets early so the failure-message timeline
    // renderer can drive from the unified PhaseBucket source
    // (Timeline::from_phase_buckets) rather than re-deriving phases
    // from raw monitor samples (Timeline::build). The drain
    // consumes the snapshot bridge; success-path consumers below
    // read pre-built buckets + the cached SampleSeries instead of
    // re-draining (the bridge is already empty after this point).
    let drained_for_phases = result.snapshot_bridge.drain_ordered_with_stats();
    let early_sample_series = crate::scenario::sample::SampleSeries::from_drained_typed(
        drained_for_phases,
        result.monitor.clone(),
    );
    let mut early_phase_buckets =
        crate::assert::build_phase_buckets_with_stimulus(&early_sample_series, stimulus_events);
    // Build timeline from the pre-bucketed phases. When no
    // PhaseBuckets exist (scenario had no periodic captures,
    // e.g. single-phase tests) but monitor samples ARE present,
    // fall back to the legacy Timeline::build path so the
    // failure-message timeline still renders monitor-derived
    // metrics. The fallback preserves operator-facing diagnostic
    // continuity for monitor-only runs — the new from_phase_buckets
    // path requires snapshot-bridge captures to materialise
    // PhaseBuckets, which monitor-only runs don't produce.
    let timeline = if !early_phase_buckets.is_empty() {
        Some(crate::timeline::Timeline::from_phase_buckets(
            &early_phase_buckets,
            stimulus_events,
            &crate::timeline::TimelineContext::default(),
        ))
    } else {
        result
            .monitor
            .as_ref()
            .map(|m| crate::timeline::Timeline::build(stimulus_events, &m.samples))
    };

    let sched_label = reporting::scheduler_label(&entry.scheduler.binary);
    let output = &result.output;
    let raw_dump = extract_sched_ext_dump(&result.stderr).unwrap_or_default();
    let dump_section = if raw_dump.is_empty() {
        String::new()
    } else {
        format!("\n\n--- sched_ext dump ---\n{raw_dump}")
    };
    // Concatenate bulk-port `MSG_TYPE_SCHED_LOG` chunks then run
    // the marker-pair extractor on the merged stream — pre-bulk-port
    // migration the markers travelled in `output` (COM2). Either
    // source feeds `parse_sched_output` byte-for-byte; falling back
    // to `output` when the bulk-port drain has no SchedLog frames
    // covers verifier-only paths.
    let sched_log_merged = crate::verifier::concat_sched_log_chunks(result.guest_messages.as_ref());
    let sched_log_input: &str = if !sched_log_merged.is_empty() {
        &sched_log_merged
    } else {
        output
    };
    let sched_log_section = parse_sched_output(sched_log_input)
        .map(|s| {
            let collapsed = crate::verifier::collapse_cycles(s);
            let is_verifier = collapsed.contains("processed") && collapsed.contains("insns");
            let lines: Vec<&str> = collapsed.lines().collect();
            let tail = if !is_verifier && lines.len() > 200 {
                let skipped = lines.len() - 200;
                format!(
                    "[{skipped} lines truncated]\n{}",
                    lines[lines.len() - 200..].join("\n")
                )
            } else {
                collapsed
            };
            format!("\n\n--- scheduler log ---\n{tail}")
        })
        .unwrap_or_default();
    let fingerprint_line = sched_log_fingerprint(sched_log_input)
        .map(|fp| {
            if crate::cli::stderr_color() {
                format!("\x1b[1;31m{fp}\x1b[0m\n")
            } else {
                format!("{fp}\n")
            }
        })
        .unwrap_or_default();
    // Hoist the first actionable `scx_bpf_error`-class line to the
    // TOP of the failure message (above the existing noisy sections
    // like sched_log / sched_ext dump / monitor). Without this hint
    // the test author had to scroll past ~200 lines of trace_pipe
    // dump output to find the line that explains why the scheduler
    // exited; see KTSTR_API_ISSUES_FROM_SCX_MITOSIS.md B4 for the
    // motivating user report. Extraction is suppressed when there
    // is nothing actionable to surface so passing tests stay quiet.
    //
    // Gated behind a closure: every failure return path below renders
    // exactly one failure message and calls the closure exactly once,
    // and a passing test takes the `return Ok(check_result)` at the
    // end of the parse-success arm without invoking any failure
    // formatter — so the `extract_bug_summary` scan over sched_log +
    // dump never runs on the pass path. The eager `match` this
    // replaces ran the scan unconditionally on every test, paying the
    // ANSI strip + line walk for passing tests that would never
    // render the result.
    let bug_summary_line = || -> String {
        match crate::test_support::output::extract_bug_summary(sched_log_input, &raw_dump) {
            Some(text) => {
                if crate::cli::stderr_color() {
                    format!("\x1b[1;31mBUG SUMMARY:\x1b[0m {text}\n")
                } else {
                    format!("BUG SUMMARY: {text}\n")
                }
            }
            None => String::new(),
        }
    };

    let tl_ctx = crate::timeline::TimelineContext {
        kernel: extract_kernel_version(&result.stderr),
        topology: Some(format!("{topo} ({} cpus)", topo.total_cpus())),
        scheduler: Some(entry.scheduler.name.to_string()),
        scenario: Some(entry.name.to_string()),
        duration_s: Some(result.duration.as_secs_f64()),
    };

    // Section builders shared by every error branch in this function.
    // Timeline skips phaseless runs; monitor only reports when an
    // active scheduler exposes rq data (EEVDF reads would be junk).
    let build_timeline_section = || -> String {
        timeline
            .as_ref()
            .filter(|t| !t.phases.is_empty())
            .map(|t| format!("\n\n{}", t.format_with_context(&tl_ctx)))
            .unwrap_or_default()
    };
    let build_monitor_section = || -> String {
        if entry.scheduler.has_active_scheduling()
            && let Some(ref monitor) = result.monitor
        {
            reporting::format_monitor_section(monitor, merged_assert)
        } else {
            String::new()
        }
    };

    if let Ok(mut check_result) = parse_assert_result_from_drain(result.guest_messages.as_ref()) {
        // Fold host-side LlmExtract failures into the guest's
        // AssertResult before the sidecar write so per-run stats
        // tooling sees the host-extracted verdict, not the guest's
        // placeholder pass(). Each host-side failure is appended as
        // an `AssertDetail` exactly as if it had been raised inside
        // the guest's `evaluate_checks` — same kind, same prose
        // shape — so failure-rendering downstream is uniform across
        // sources.
        for detail in host_extract_failures {
            check_result.merge(AssertResult::fail(detail.clone()));
        }

        // Fold the host-side `post_vm` callback's Err into the
        // verdict so it flows through the same failure path as
        // host-extract failures and the guest-stamped check
        // result. The downstream failure formatter renders the
        // `--- scheduler log ---` / `--- sched_ext dump ---` /
        // `--- monitor ---` sections + dispatches `repro_fn`
        // (auto-repro) from this single point — no parallel
        // handler needed.
        if let Some(err) = post_vm_err {
            check_result.merge(AssertResult::fail(crate::assert::AssertDetail::new(
                crate::assert::DetailKind::Other,
                format!("post_vm callback returned Err: {err:#}"),
            )));
        }

        // Cleanup-budget enforcement. When the entry sets
        // `cleanup_budget` and `collect_results` produced a measurement
        // (i.e. `run_vm` returned normally — see
        // `VmResult::cleanup_duration`), fold a failing
        // `AssertDetail` into the test verdict if teardown overran the
        // budget. Skipped when either side is `None`: an absent budget
        // means the entry opted out, an absent measurement means the
        // run never reached `collect_results` (BSP panic propagated
        // through `?`, or any pre-BSP setup error returning an `Err`
        // before `VmRunState` is constructed). Note: a host-watchdog
        // timeout is NOT a `None` case — `run_bsp_loop` exits cleanly
        // with `timed_out = true` and `collect_results` still
        // populates `cleanup_duration` to `Some(_)`, per the field
        // contract documented at `src/vmm/mod.rs` for
        // `VmResult::cleanup_duration`. The surrounding error path
        // (BSP panic propagation, pre-BSP setup `Err`) already
        // produces a failure verdict in the absent-measurement case,
        // so a budget check here would double-report.
        //
        // Contract: this check only fires inside the parse-success arm
        // (the `if let Ok(mut check_result)` above) — i.e. when the
        // guest-side test body emitted a parseable AssertResult into
        // SHM or COM2. Tests whose body panics or fails to write a
        // result skip budget enforcement entirely; the watchdog
        // timeout / no-parseable-result branch below produces its own
        // verdict in those cases. Tests that opt into
        // `cleanup_budget_ms` MUST ensure their body returns
        // `Ok(AssertResult)` (e.g. `Ok(AssertResult::pass())`) before
        // teardown begins, otherwise the budget knob is silently
        // inert.
        if let (Some(budget), Some(measured)) = (entry.cleanup_budget, result.cleanup_duration)
            && measured > budget
        {
            check_result.merge(AssertResult::fail(crate::assert::AssertDetail::new(
                crate::assert::DetailKind::Other,
                format!(
                    "vm cleanup overran budget: measured {:.3}s, budget {:.3}s. \
                     Likely a regression in host-side teardown — investigate \
                     the post-BSP-exit join/drain path \
                     (`vmm::KtstrVm::collect_results`).",
                    measured.as_secs_f64(),
                    budget.as_secs_f64(),
                ),
            )));
        }

        // Reproducer-mode scx_bpf_error matcher. Runs the configured
        // `expect_scx_bpf_error_contains` / `_matches` patterns against
        // the combined scheduler log + sched_ext dump corpus — the two
        // host-side surfaces where scx_bpf_error printk text lands.
        // Matcher details (mismatch + misuse diagnostics) fold into
        // `check_result.details` so the failure-message construction
        // path renders them alongside the rest of the verdict.
        //
        // Gated on at least one matcher being configured so the
        // common-path (no matcher) doesn't allocate the corpus string
        // — the evaluator's own early-return covers the no-matcher
        // case but only after the format! has already run.
        let matcher_configured = merged_assert.expect_scx_bpf_error_contains.is_some()
            || merged_assert.expect_scx_bpf_error_matches.is_some();
        let matcher_details = if matcher_configured {
            let matcher_corpus = format!("{sched_log_input}\n{dump_section}");
            merged_assert.evaluate_scx_bpf_error_match(&matcher_corpus, entry.expect_err)
        } else {
            Vec::new()
        };
        let matcher_mismatch = !matcher_details.is_empty();
        for d in matcher_details {
            check_result.merge(AssertResult::fail(d));
        }

        // Write sidecar before checking pass/fail so both outcomes are captured.
        // A sidecar write failure is logged but not propagated: the test
        // verdict itself is still valid — only post-run stats tooling
        // loses visibility.
        let args: Vec<String> = std::env::args().collect();
        let work_type =
            super::args::extract_work_type_arg(&args).unwrap_or_else(|| "SpinWait".to_string());
        if let Err(e) = write_sidecar(
            entry,
            result,
            stimulus_events,
            &check_result,
            &work_type,
            payload_metrics,
        ) {
            eprintln!("ktstr_test: {e:#}");
        }

        if !check_result.is_pass() {
            let details = check_result
                .failure_details()
                .chain(check_result.inconclusive_details())
                .chain(check_result.skip_details())
                .map(|d| d.message.as_str())
                .collect::<Vec<_>>()
                .join("\n  ");
            // Render info_notes in their own delineated section
            // (mirrors --- stats --- / --- auto-repro --- pattern)
            // so the structural details-vs-info separation that
            // sidecar consumers rely on is also visible at the
            // operator-facing failure-dump boundary. An undelineated
            // append into the failures block would interleave
            // failure messages with context lines and undo the
            // split's "details = failures" invariant at the human
            // surface.
            let info_section = if check_result.info_notes.is_empty() {
                String::new()
            } else {
                let lines: Vec<String> = check_result
                    .info_notes
                    .iter()
                    .map(|n| format!("  {}", n.message))
                    .collect();
                format!("\n\n--- info ---\n{}", lines.join("\n"))
            };
            let repro = if entry.scheduler.has_active_scheduling() {
                repro_fn(output)
            } else {
                None
            };
            let repro_section = repro
                .map(|r| format!("\n\n--- auto-repro ---\n{r}"))
                .unwrap_or_default();
            let timeline_section = build_timeline_section();
            let stats_section = if !check_result.stats.cgroups.is_empty() {
                let s = &check_result.stats;
                let mut lines = vec![format!(
                    "\n\n--- stats ---\n{} workers, {} cpus, {} migrations, worst_spread={:.1}%, worst_gap={}ms",
                    s.total_workers,
                    s.total_cpus,
                    s.total_migrations,
                    s.worst_spread,
                    s.worst_gap_ms,
                )];
                for (i, cg) in s.cgroups.iter().enumerate() {
                    lines.push(format!(
                        "  cg{}: workers={} cpus={} spread={:.1}% gap={}ms migrations={} iter={}",
                        i,
                        cg.num_workers,
                        cg.num_cpus,
                        cg.spread,
                        cg.max_gap_ms,
                        cg.total_migrations,
                        cg.total_iterations,
                    ));
                }
                lines.join("\n")
            } else {
                String::new()
            };
            // Structural filter for the console-dump gate: match on
            // `DetailKind::SchedulerDied` only. Every scheduler-exit
            // emit site in this crate tags its `AssertDetail` with
            // that variant (see the ops.rs / scenario/mod.rs call
            // sites plus the `format_sched_died_*` helpers in
            // `assert.rs`), so filtering by kind is sufficient — the
            // prior `is_scheduler_death()` prefix-match fallback was
            // removed once every production emitter was audited as
            // kind-tagging its details. `verbose()` forces the
            // section on for operator debugging runs.
            let console_section = if check_result.failure_details().any(|d| {
                matches!(
                    d.kind,
                    crate::assert::DetailKind::SchedulerCrashed
                        | crate::assert::DetailKind::SchedulerExitedCleanly
                        | crate::assert::DetailKind::SchedulerDiedUnknownReason
                )
            }) || verbose()
            {
                let init_stage = classify_init_stage(result.guest_messages.as_ref());
                format_console_diagnostics(&result.stderr, result.exit_code, init_stage)
            } else {
                String::new()
            };
            let monitor_section = build_monitor_section();
            // Periodic-sample coverage gauge: fires when the entry
            // configured `num_snapshots > 0`. Renders the
            // fired/target ratio; suppressed when the entry did
            // not request periodic capture so non-periodic tests
            // produce uncluttered failure output.
            let periodic_section =
                crate::test_support::output::format_periodic_samples_section(result);
            // Temporal-assertion summary: aggregates every
            // [`DetailKind::Temporal`] detail into a single block
            // so a test author chasing a violated periodic-sample
            // pattern sees the offending sample tag(s) at the top
            // of the section instead of scrolling through scalar
            // claim failures.
            let temporal_section =
                crate::test_support::output::format_temporal_assertions_section(&check_result);
            // Skip-only results take an early exit through
            // `record_skip_sidecar` upstream, so this block only
            // sees Fail or Inconclusive. Render the lattice verdict
            // accurately — "failed" for a hard Fail, "inconclusive"
            // for a zero-denominator Inconclusive — so a CI human
            // reading the dump can triage without inferring from
            // exit code alone (dispatch.rs projects Inconclusive to
            // exit code 2; the verdict word here mirrors that).
            let verdict_word = if check_result.is_inconclusive() {
                "inconclusive"
            } else {
                "failed"
            };
            let msg = format!(
                "{}{}ktstr_test '{}'{} [topo={}] {verdict_word}:\n  {}{}{}{}{}{}{}{}{}{}{}",
                fingerprint_line,
                bug_summary_line(),
                entry.name,
                sched_label,
                topo,
                details,
                info_section,
                stats_section,
                console_section,
                timeline_section,
                periodic_section,
                temporal_section,
                sched_log_section,
                monitor_section,
                dump_section,
                repro_section,
            );
            // When the scx_bpf_error matcher contributed a mismatch
            // detail, wrap the Err with [`ScxBpfErrorMatcherMismatch`]
            // so the dispatch-time expect_err inversion bypasses this
            // failure (a reproducer with a matcher mismatch fails the
            // test even though expect_err = true would normally invert
            // a failure into a pass). When the matcher matched (or was
            // unset), the normal expect_err inversion path applies.
            let err = anyhow::anyhow!("{msg}");
            return Err(if matcher_mismatch {
                err.context(ScxBpfErrorMatcherMismatch)
            } else {
                err
            });
        }

        // Evaluate monitor data against thresholds when a scheduler is running.
        // Without a scheduler (EEVDF), monitor reads rq data that may be
        // uninitialized or irrelevant — skip evaluation in that case.
        //
        // Skip early monitor warmup samples: during boot, BPF verification,
        // and initramfs unpacking the scheduler tick may not fire for hundreds
        // of milliseconds. These transient stalls are real but not indicative
        // of scheduler bugs.
        if entry.scheduler.has_active_scheduling()
            && merged_assert.has_monitor_thresholds()
            && let Some(ref monitor) = result.monitor
        {
            let eval_report = reporting::trim_settle_samples(monitor);
            let thresholds = merged_assert.monitor_thresholds();
            let verdict = thresholds.evaluate(&eval_report);
            if verdict.is_fail() {
                let details = verdict.details.join("\n  ");
                let timeline_section = build_timeline_section();
                let monitor_section = reporting::format_monitor_section(monitor, merged_assert);
                let msg = format!(
                    "{}{}ktstr_test '{}'{} [topo={}] {ERR_MONITOR_FAILED_AFTER_SCENARIO}:\n  {}{}{}{}{}",
                    fingerprint_line,
                    bug_summary_line(),
                    entry.name,
                    sched_label,
                    topo,
                    details,
                    timeline_section,
                    monitor_section,
                    sched_log_section,
                    dump_section,
                );
                anyhow::bail!("{msg}");
            } else if verdict.is_inconclusive() {
                // Monitor reached the evaluator but had no signal —
                // no samples, or data that failed the plausibility
                // check (uninitialized guest memory). Record on
                // `check_result` as an Inconclusive outcome so the
                // sidecar / exit-code surface reflects "couldn't
                // measure" rather than silently passing. Bailing
                // would conflate Inconclusive with the Fail arm
                // above; folding into the outcome stream lets the
                // 4-state pipeline classify the run correctly.
                check_result.merge(crate::assert::AssertResult::inconclusive(
                    crate::assert::AssertDetail::new(
                        crate::assert::DetailKind::Monitor,
                        format!("monitor evaluation inconclusive: {}", verdict.summary),
                    ),
                ));
            }
        }

        // Auto-populate per-phase metric buckets on the returned
        // AssertResult. Drains the snapshot bridge for periodic
        // captures + on-demand fixture-path captures, builds a
        // SampleSeries, and folds it through
        // `crate::assert::build_phase_buckets` so the test author
        // sees `result.stats.phases` populated without needing to
        // manually stitch the snapshot drain to the metric
        // aggregator. Single-phase scenarios with no Steps still
        // run (the bridge may have a periodic capture or two)
        // but yield a phases vec containing only the BASELINE
        // bucket; the renderer is sentinel-free so an empty
        // metrics map paints as "no data" rather than masquerading
        // as real zeros.
        //
        // The bridge drain here is the framework's contract drain;
        // an integration test that bypasses evaluate_vm_result
        // (e.g. tests/stats_bridge_e2e.rs) still owns its own
        // direct `result.snapshot_bridge.drain*()` call path
        // because those tests instrument the framework rather
        // than depending on it. Within evaluate_vm_result the
        // drain is the final consumer.
        // Phase buckets were built at evaluate_vm_result entry
        // (drain happened there) so the unified PhaseBucket source
        // feeds both the failure-message Timeline render (via
        // Timeline::from_phase_buckets up top) and the stamped
        // ScenarioStats.phases below. Reuse the pre-built vec +
        // SampleSeries rather than re-draining (the bridge was
        // already consumed).
        check_result.stats.phases = std::mem::take(&mut early_phase_buckets);
        let sample_series_for_phases = &early_sample_series;
        // Cross-RUN aggregate fill: for any METRICS entry with a
        // read_sample wire but no typed GauntletRow field, compute
        // the per-RUN aggregate from the same samples and write into
        // stats.ext_metrics. Without this, MetricDef::read returns
        // None on both sides at cargo ktstr stats compare time, the
        // EPSILON guard drops the row, and the operator never sees
        // the metric — a silent data drop. Skips keys already
        // populated as typed fields or by other producers.
        crate::assert::populate_run_ext_metrics(
            sample_series_for_phases,
            &mut check_result.stats.ext_metrics,
        );
        // Sibling fill from per-phase metrics — closes the gap
        // for avg_imbalance_ratio (MonitorSample-sourced) and
        // iteration_rate (stimulus-event-sourced). Their
        // read_sample dispatches return None so the SampleSeries
        // path above misses them; without this call, those keys
        // never appear in ext_metrics and cargo ktstr stats
        // compare silently drops them.
        crate::assert::populate_run_ext_metrics_from_phases(
            &check_result.stats.phases,
            &mut check_result.stats.ext_metrics,
        );

        return Ok(check_result);
    }

    // No parseable result — no AssertResult found via the bulk port
    // or COM2. With an scx scheduler under test this typically
    // means the scheduler exited (crash, BPF verifier reject,
    // scx_bpf_error() exit, sched_ext disablement); on the kernel-
    // default scheduler it means the payload itself failed. Attempt
    // auto-repro if enabled and a scheduler was running.
    // Any scheduler failure that prevents producing a test result
    // warrants repro — BPF verifier failures, scx_bpf_error() exits,
    // crashes, and stalls all land here. Previous code required
    // specific string patterns (`SENTINEL_SCHEDULER_DIED`,
    // "sched_ext:" + "disabled") which missed mid-test exits where
    // the sched_exit_monitor writes guest messages but not COM2.
    let repro_section = if entry.scheduler.has_active_scheduling() {
        repro_fn(output)
            .map(|r| format!("\n\n--- auto-repro ---\n{r}"))
            .unwrap_or_default()
    } else {
        String::new()
    };

    // Build a diagnostic section from COM1 kernel console output and exit code.
    // When COM2 has scheduler output markers, sched_log_section and dump_section
    // carry the diagnostics and the kernel console is noise (BIOS, ACPI boot).
    // When COM2 has NO scheduler output (crash before writing), the kernel console
    // is the ONLY source of crash info — include it unconditionally as a fallback.
    // When a scheduler is active and we landed in the no-parseable-
    // result path, force the console section on regardless of
    // SCHED_OUTPUT_START presence: the init-stage classification it
    // carries is the only signal of where the boot got stuck (BPF
    // verifier reject vs. scheduler attach vs. test-fn launch), and
    // dropping it leaves the operator without that locator.
    // Pre-bulk-port-migration: scheduler logs travelled in COM2
    // bracketed by `SCHED_OUTPUT_START`. The marker now lives
    // inside `MSG_TYPE_SCHED_LOG` chunk bytes, so check both the
    // bulk-port drain and the legacy COM2 fallback (still useful
    // when a path emits SCHED_OUTPUT to neither).
    let bulk_sched_log = crate::verifier::concat_sched_log_chunks(result.guest_messages.as_ref());
    let has_sched_output =
        output.contains(SCHED_OUTPUT_START) || bulk_sched_log.contains(SCHED_OUTPUT_START);
    let console_section =
        if !has_sched_output || verbose() || entry.scheduler.has_active_scheduling() {
            let init_stage = classify_init_stage(result.guest_messages.as_ref());
            format_console_diagnostics(&result.stderr, result.exit_code, init_stage)
        } else {
            String::new()
        };

    let timeline_section = build_timeline_section();

    // Build monitor section for error paths where neither the bulk
    // port nor COM2 had a parseable result.
    let monitor_section = build_monitor_section();

    // When both timed_out and crash_message fire, the prior behavior
    // bailed with the timeout reason and silently dropped the crash
    // backtrace from the freeze coordinator's
    // `extract_panic_message` capture. Render both: timeout stays the
    // primary classification (the host watchdog is what halted the
    // run) but the crash backtrace appends as a `guest crashed:`
    // section so the operator sees the panic frames the guest
    // emitted before the watchdog fired.

    // Fold the host-side `post_vm` callback's Err into the parse-fail
    // diagnostic — used by BOTH the timed-out and no-result arms below.
    // Most parse-fail paths are guest-crashed-before-reporting, in which
    // case post_vm probably wouldn't have run meaningfully — but if the
    // test author's post_vm did fire and return Err on the partial
    // result, the failure surface MUST include it. Prepending the
    // post_vm message keeps the crash diagnostics as the load-bearing
    // body and surfaces post_vm as a leading "host-side check also
    // reported" line so the operator sees both signals together. (The
    // PostVmAssertionFailure marker that governs the expect_err
    // inversion decision is attached separately by the caller; this only
    // controls the rendered message.)
    let post_vm_prefix = post_vm_err
        .map(|e| format!("post_vm callback returned Err: {e:#}\n\n"))
        .unwrap_or_default();

    if result.timed_out {
        let crash_section = if let Some(ref guest_crash) = result.crash_message {
            format!("\n\n{ERR_GUEST_CRASHED_PREFIX}\n{guest_crash}")
        } else {
            String::new()
        };
        // Watchdog diagnostic. The bare `timed out (no result via
        // bulk port or COM2)` line tells the operator nothing about
        // why the deadline fired. Append the four knobs the host
        // watchdog actually consulted: the host VM timeout (the
        // value the watchdog deadline was anchored to,
        // `max(watchdog_timeout, duration)` per
        // `vm_timeout_from_entry`), the scheduler watchdog timeout
        // (`entry.watchdog_timeout`, the scx_sched.watchdog_timeout
        // override applied to the guest kernel), the workload
        // duration (`entry.duration`, what the test body asked for),
        // and the wall-clock the run actually consumed
        // (`result.duration`). The hint mirrors the watchdog-thread
        // diagnostic in `freeze_coord/mod.rs` so the operator sees
        // the same direction whether they read VM stderr or this
        // test-output message.
        let vm_timeout = vm_timeout_from_entry(entry);
        let watchdog_section = format!(
            "\n\n--- watchdog ---\n\
             elapsed={:?} (VM run wall-clock)\n\
             vm_timeout={:?} (host watchdog deadline = max(watchdog_timeout, \
             duration, 1s) + vCPU-scaled vm_boot_headroom [+ 30s cold-BTF \
             budget for bpf_map_write tests])\n\
             watchdog_timeout={:?} (scx_sched.watchdog_timeout override)\n\
             duration={:?} (workload duration)\n\
             hint: if the test body needs more wall time, increase \
             duration (the `duration` field on `KtstrTestEntry` / \
             `#[ktstr_test(duration_ms = ...)]`); the VM timeout adds \
             vCPU-scaled boot headroom on top of max(watchdog_timeout, \
             duration, 1s), so raising duration also extends the host \
             watchdog deadline",
            result.duration, vm_timeout, entry.watchdog_timeout, entry.duration,
        );
        let timeout_reason = {
            let scx_exits = crate::monitor::dmesg_scx::parse_kmsg_window(&result.stderr);
            if let Some(ev) = scx_exits.last() {
                if ev.message.is_empty() {
                    format!("timed out (scheduler {} exited)", ev.scheduler_name)
                } else {
                    format!("timed out (scheduler exited: {})", ev.message)
                }
            } else {
                ERR_TIMED_OUT_NO_RESULT.to_string()
            }
        };
        let msg = format!(
            "{post_vm_prefix}{}{}ktstr_test '{}'{} [topo={}] {}{}{}{}{}{}{}{}{}",
            fingerprint_line,
            bug_summary_line(),
            entry.name,
            sched_label,
            topo,
            timeout_reason,
            watchdog_section,
            crash_section,
            console_section,
            timeline_section,
            sched_log_section,
            dump_section,
            monitor_section,
            repro_section,
        );
        anyhow::bail!("{msg}");
    }

    let reason = if let Some(ref guest_crash) = result.crash_message {
        format!("{ERR_GUEST_CRASHED_PREFIX}\n{guest_crash}")
    } else if let Some(crash_msg) = extract_panic_message(output) {
        format!("{ERR_GUEST_CRASHED_PREFIX} {crash_msg}")
    } else if entry.scheduler.has_active_scheduling() {
        let scx_exits = crate::monitor::dmesg_scx::parse_kmsg_window(&result.stderr);
        if let Some(ev) = scx_exits.last() {
            if ev.message.is_empty() {
                format!("scheduler exited ({})", ev.scheduler_name)
            } else {
                format!("scheduler exited: {}", ev.message)
            }
        } else if let Some(reason) = extract_exit_from_dump_trace(&result.stderr) {
            format!("scheduler exited: {reason}")
        } else {
            ERR_NO_TEST_RESULT_FROM_GUEST.to_string()
        }
    } else {
        ERR_NO_TEST_FUNCTION_OUTPUT.to_string()
    };
    let msg = format!(
        "{post_vm_prefix}{}{}ktstr_test '{}'{} [topo={}] {}{}{}{}{}{}{}",
        fingerprint_line,
        bug_summary_line(),
        entry.name,
        sched_label,
        topo,
        reason,
        console_section,
        timeline_section,
        sched_log_section,
        dump_section,
        monitor_section,
        repro_section,
    );
    anyhow::bail!("{msg}")
}

#[cfg(test)]
mod eval_tests;
#[cfg(test)]
mod eval_tests_eval;
#[cfg(test)]
mod eval_tests_llm;
#[cfg(test)]
mod eval_tests_reporting;
