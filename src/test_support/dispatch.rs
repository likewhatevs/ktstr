//! Process-level dispatch and nextest protocol handling.
//!
//! This module owns every code path that runs before (or in lieu of)
//! the user's `main()`:
//!
//! - [`ktstr_test_early_dispatch`]: the `#[ctor]` that fires in every
//!   ktstr-linked binary. Routes the process to guest init, host-side
//!   VM launch, guest-side test execution, or nextest protocol handling.
//! - [`ktstr_main`]: the nextest protocol handler — `--list` returns
//!   `ktstr/` and `gauntlet/` test names, `--exact` runs a single test.
//! - [`run_ktstr_test`]: programmatic entry point used by library
//!   consumers and the macro-generated `#[test]` wrappers.
//! - [`analyze_sidecars`]: collects sidecar JSON from a run directory
//!   and renders the full gauntlet analysis (rows + verifier + callback
//!   profile + KVM stats) into a string.
//!
//! The heavy lifting lives in sibling submodules: `eval` (host-side
//! result judgment — `run_ktstr_test_inner` and `evaluate_vm_result`),
//! `sidecar` (per-run JSON), `probe` (auto-repro + BPF probe pipeline),
//! `args` (CLI extraction), and the [`crate::vmm`] VM launcher.

use std::path::PathBuf;

use anyhow::{Context, Result};

use crate::assert::AssertResult;

#[cfg(feature = "export")]
use super::extract_export_output_arg;
use super::{
    HostClass, KTSTR_TESTS, KtstrTestEntry, TopoOverride, classify_host_error, collect_sidecars,
    extract_export_test_arg, extract_shell_test_arg, extract_test_fn_arg, extract_topo_arg,
    find_test, format_callback_profile, format_kvm_stats, format_verifier_stats,
    maybe_dispatch_vm_test, parse_topo_string, propagate_rust_env_from_cmdline,
    record_skip_sidecar, resolve_test_kernel, run_ktstr_test_inner, sidecar_dir, try_flush_profraw,
};

/// Check if an error is a host topology mismatch (e.g. test requests
/// 2 LLCs but host has 1, or more CPUs than the host carries).
///
/// Walks the FULL error chain via `e.chain().any(...)` so a
/// [`TopologyInsufficient`] wrapped in `.context(...)` (the
/// `crate::test_support::eval` `"build ktstr_test VM"` / `"run
/// ktstr_test VM"` wrappers) is still recognised — mirrors
/// [`is_resource_contention`]. Replaced a fragile message string-match
/// (`"need"` + `"LLC"`/`"CPU"`) that would misclassify any unrelated
/// error happening to contain those words as a topology skip.
///
/// [`TopologyInsufficient`]: crate::vmm::host_topology::TopologyInsufficient
#[doc(hidden)]
pub fn is_topology_insufficient(e: &anyhow::Error) -> bool {
    e.chain().any(|cause| {
        cause
            .downcast_ref::<crate::vmm::host_topology::TopologyInsufficient>()
            .is_some()
    })
}

/// Check if an `anyhow::Error` carries a [`ResourceContention`].
///
/// Walks the FULL error chain via `e.chain().any(...)` so a
/// `ResourceContention` wrapped in `.context(...)` (e.g. the
/// `crate::test_support::eval` `"build ktstr_test VM"` and `"run ktstr_test VM"`
/// wrappers) is still recognised — the macro's match arm depends on
/// this.
///
/// Used by the `#[ktstr_test]` macro expansion to short-circuit on
/// host-resource contention (LLC slots / CPUs unavailable, KVM fd
/// budget exhausted, ENOMEM): the macro emits the canonical
/// `ktstr: SKIP: resource contention: ...` banner and early-returns
/// so libtest sees pass. The skip sidecar is recorded at every
/// contention site inside `run_ktstr_test_inner`, so stats tooling
/// still sees the skip without a panic-driven nextest retry. `pub`
/// because the macro-generated `#[test]` body in `ktstr-macros`
/// references it by absolute path; `#[doc(hidden)]` keeps it out
/// of rustdoc's public surface — it is plumbing, not user API.
///
/// [`ResourceContention`]: crate::vmm::host_topology::ResourceContention
#[doc(hidden)]
pub fn is_resource_contention(e: &anyhow::Error) -> bool {
    e.chain().any(|cause| {
        cause
            .downcast_ref::<crate::vmm::host_topology::ResourceContention>()
            .is_some()
    })
}

/// Check if an `anyhow::Error` carries a [`PerfModeUnavailable`].
///
/// Chain-aware (walks `e.chain()`), like [`is_topology_insufficient`].
/// A `PerfModeUnavailable` is a HOST-INSUFFICIENCY skip, like RC/TI: the
/// host fundamentally cannot honor an explicitly-requested perf-mode
/// guarantee (too few CPUs for an exclusive host LLC + a service CPU).
/// The VM is never run unisolated (it errors at build), so
/// `result_to_exit_code` and the macro body route it to a VISIBLE skip
/// by default, promoted to a FAIL banner under `KTSTR_NO_SKIP_MODE`.
///
/// [`PerfModeUnavailable`]: crate::vmm::host_topology::PerfModeUnavailable
#[doc(hidden)]
pub fn is_perf_mode_unavailable(e: &anyhow::Error) -> bool {
    e.chain().any(|cause| {
        cause
            .downcast_ref::<crate::vmm::host_topology::PerfModeUnavailable>()
            .is_some()
    })
}

/// Check if an `anyhow::Error` carries a [`CpuBudgetUnsatisfiable`].
///
/// Chain-aware. A `CpuBudgetUnsatisfiable` is a HARD ERROR (an explicit
/// `--cpu-cap` / `cpu_budget` number the host cannot satisfy), NOT a skip.
///
/// [`CpuBudgetUnsatisfiable`]: crate::vmm::host_topology::CpuBudgetUnsatisfiable
#[doc(hidden)]
pub fn is_cpu_budget_unsatisfiable(e: &anyhow::Error) -> bool {
    e.chain().any(|cause| {
        cause
            .downcast_ref::<crate::vmm::host_topology::CpuBudgetUnsatisfiable>()
            .is_some()
    })
}

/// Check if an `anyhow::Error` carries a [`TopologyUnrepresentable`].
///
/// Chain-aware. A `TopologyUnrepresentable` is a HARD ERROR (a topology no
/// host can represent under this VMM's static device layout — the aarch64
/// over-`MAX_VCPUS` GICv3-redistributor case), NOT a skip.
/// `classify_host_error` classifies it as `HostClass::Fail`, checked above
/// the RC/TI skip types and handled above the `expect_err` inversion in
/// both `err_to_exit_code` and the macro body, so a too-wide aarch64
/// topology can neither masquerade as the expected failure nor be turned
/// into a silent skip. Distinct from [`is_topology_insufficient`], which
/// matches the host-DEPENDENT skip type.
///
/// [`TopologyUnrepresentable`]: crate::vmm::host_topology::TopologyUnrepresentable
#[doc(hidden)]
pub fn is_topology_unrepresentable(e: &anyhow::Error) -> bool {
    e.chain().any(|cause| {
        cause
            .downcast_ref::<crate::vmm::host_topology::TopologyUnrepresentable>()
            .is_some()
    })
}

/// Predicate: walks the [`anyhow::Error`] chain looking for a
/// [`KernelUnavailable`] cause. Used by `classify_host_error` to classify
/// a no-kernel host as a skip-class host-insufficiency.
///
/// The harness signals "I have no kernel to boot, the binary was
/// likely invoked outside `cargo ktstr test`" by surfacing
/// [`KernelUnavailable`] rather than a generic `anyhow::bail!`.
/// `classify_host_error` maps it to `HostClass::Skip` (the canonical
/// `ktstr: SKIP: harness not configured: ...` banner), promoted to a FAIL
/// under `KTSTR_NO_SKIP_MODE` — same shape as the resource-contention skip.
/// `pub` + `#[doc(hidden)]`: plumbing re-exported from `test_support`
/// alongside the sibling `is_*` predicates, not user API.
///
/// Both consumers route a `KernelUnavailable` through the shared
/// [`classify_host_error`] (a no-kernel host is a skip-class
/// host-insufficiency): `err_to_exit_code` and the `#[ktstr_test]` macro
/// body both SKIP it by default, promoted to a FAIL under
/// `KTSTR_NO_SKIP_MODE`. Under nextest the plain `#[test]` wrapper is
/// suppressed, so an entry dispatches as `ktstr/{name}` via `run_named_test`
/// → `err_to_exit_code` — meaning a developer running `cargo nextest run`,
/// or `cargo ktstr test` without `--kernel`, on a kernel-less host gets a
/// clean skip rather than a hard fail on every entry. This cannot mask a CI
/// kernel-build failure: a requested `--kernel` that fails to build bails in
/// cargo-ktstr (`resolve_kernel_set`) before nextest is spawned, so a
/// `KernelUnavailable` here only ever means "no kernel was requested".
/// Pinned by `result_to_exit_code_kernel_unavailable_skips_on_dispatch_path`.
///
/// [`classify_host_error`]: crate::test_support::classify_host_error
///
/// [`KernelUnavailable`]: crate::test_support::eval::KernelUnavailable
#[doc(hidden)]
pub fn is_kernel_unavailable(e: &anyhow::Error) -> bool {
    e.chain().any(|cause| {
        cause
            .downcast_ref::<crate::test_support::eval::KernelUnavailable>()
            .is_some()
    })
}

/// A nextest-safe kernel identifier whose construction is gated
/// through [`sanitize_kernel_label`] — once a value of this type
/// exists, the contained string is GUARANTEED to match the
/// `kernel_[a-z0-9_]+` shape that nextest's test-name parsing
/// accepts. The wrapped `String` is private so a future caller
/// cannot bypass [`Self::new`] and stuff a raw label into the
/// invariant.
///
/// Constructed by [`Self::new`] (which always calls
/// [`sanitize_kernel_label`]). Read access is via
/// [`Self::as_str`] / `Display` / `AsRef<str>` — both of which
/// expose the sanitized form unchanged.
///
/// `pub(crate)` because every consumer (this module, the
/// production parser at [`parse_kernel_list`], and the encoder
/// helpers in `cargo-ktstr` that emit the wire format
/// `parse_kernel_list` decodes) lives inside the workspace; no external
/// surface is needed today. If a future external consumer needs
/// to construct a `SanitizedKernelLabel` directly, expose
/// `Self::new` as `pub` then — but the private inner stays a
/// private invariant either way.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct SanitizedKernelLabel(String);

impl SanitizedKernelLabel {
    /// Sanitize `raw` via [`sanitize_kernel_label`] and wrap the
    /// result in the invariant-preserving newtype. The only path
    /// that produces a `SanitizedKernelLabel`; bypassing it is
    /// impossible because the inner field is private to this
    /// module.
    pub(crate) fn new(raw: &str) -> Self {
        Self(sanitize_kernel_label(raw))
    }

    /// Read access to the sanitized identifier. Returns `&str`
    /// rather than `&String` so callers can compose with
    /// `format!` / `starts_with` / `strip_suffix` without
    /// chaining `.as_str().as_str()`.
    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for SanitizedKernelLabel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl AsRef<str> for SanitizedKernelLabel {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

// `PartialEq<&str>` and `PartialEq<str>` impls let `assert_eq!`
// against a string literal stay readable in tests
// (`assert_eq!(entries[0].sanitized, "kernel_6_14_2")`) without
// forcing every consumer to chain `.as_str()`. The wrapped
// `String` is private to this module, so impls comparing
// against external `&str` values cannot break the
// "constructor enforces sanitization" invariant — the
// invariant attaches to value PRODUCTION, not to value
// COMPARISON.
impl PartialEq<&str> for SanitizedKernelLabel {
    fn eq(&self, other: &&str) -> bool {
        self.0 == *other
    }
}

impl PartialEq<str> for SanitizedKernelLabel {
    fn eq(&self, other: &str) -> bool {
        self.0 == other
    }
}

#[cfg(test)]
impl SanitizedKernelLabel {
    /// Test-only escape hatch: wrap a string that's ALREADY in
    /// the sanitized shape (`kernel_[a-z0-9_]+`) without running
    /// the sanitizer. Used by unit-test fixtures that hand-roll
    /// `KernelEntry` values whose `sanitized` field is meant to
    /// be a literal — running [`Self::new`] on `"kernel_6_14_2"`
    /// would double-prefix to `"kernel_kernel_6_14_2"`.
    ///
    /// Production code must NEVER call this — invariant
    /// violation here means callers can stuff arbitrary strings
    /// into the field, defeating the point of the newtype.
    /// `#[cfg(test)]` enforces that at compile time.
    pub(crate) fn from_pre_sanitized_for_test(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// One resolved kernel entry from `KTSTR_KERNEL_LIST` (the multi-
/// kernel fan-out wire format that `cargo ktstr test --kernel A
/// --kernel B` or `cargo ktstr verifier --kernel A --kernel B`
/// exports before exec'ing into `cargo nextest`).
///
/// `label` is the producer-side label string before
/// sanitization — e.g. `"6.14.2"` for Version, `"git_tj_sched_ext_for-next"`
/// for Git, `"6.14.2-tarball-x86_64-kc..."` for CacheKey,
/// `"path_linux_a3f2b1"` for Path. Preserved so the
/// [`crate::test_support::dispatch`] verifier sweep filter can
/// compare against `declare_scheduler!`'s `kernels = [...]`
/// declarations — specifically, range membership
/// (`"6.14..6.16"` vs `"6.14.2"`) needs the raw version string
/// to feed into [`crate::kernel_path::decompose_version_for_compare`],
/// which the sanitized form has lost (slashes / dots → underscores).
///
/// `sanitized` is the nextest-safe identifier appended to test names
/// so `cargo nextest run -E 'test(kernel_6_14_2)'` filters work
/// natively. The producer-side encoder in `cargo-ktstr` emits a
/// semantic, operator-readable label per kernel:
/// - Version / Range expansion: the version string verbatim
///   (`6.14.2`, `6.15-rc3`).
/// - CacheKey: the version prefix (everything before the
///   `-tarball-` / `-git-` source tag).
/// - Git: `git_{owner}_{repo}_{ref}` extracted from the URL.
/// - Path: `path_{basename}_{hash6}` — basename + 6-char crc32 of
///   the canonical path, disambiguating two `linux` directories
///   under different parents.
///
/// [`SanitizedKernelLabel::new`] (which calls [`sanitize_kernel_label`])
/// applies the `kernel_` prefix and `[a-z0-9_]+` normalization
/// downstream. The newtype on this field makes the invariant
/// compile-checked: a future caller cannot construct a
/// `KernelEntry` whose `sanitized` field skipped sanitization.
///
/// `kernel_dir` is the canonical absolute path to the kernel-build
/// directory the per-variant subprocess re-exports as
/// `KTSTR_KERNEL`.
#[derive(Clone, Debug)]
pub(crate) struct KernelEntry {
    pub(crate) label: String,
    pub(crate) sanitized: SanitizedKernelLabel,
    pub(crate) kernel_dir: PathBuf,
}

/// Parse the multi-kernel wire format `KTSTR_KERNEL_LIST` into a
/// `Vec<KernelEntry>`. Format: `label1=path1;label2=path2;...`,
/// semicolon-separated entries, `=` separating label from path. Empty
/// / unset env returns an empty vec — callers treat that as
/// "single-kernel mode" and fall through to `KTSTR_KERNEL`.
///
/// Malformed entries (missing `=`, empty label, empty path) are
/// dropped silently — the producer is `cargo ktstr` which encodes
/// the format under our control, so a malformed entry indicates a
/// regression in the producer rather than operator input that
/// deserves a clear error. Silent drop preserves the `len() <= 1` →
/// "treat as single-kernel" invariant in the readers downstream.
pub(crate) fn parse_kernel_list(raw: &str) -> Vec<KernelEntry> {
    raw.split(';')
        .filter_map(|seg| {
            let seg = seg.trim();
            if seg.is_empty() {
                return None;
            }
            let (label, path) = seg.split_once('=')?;
            let label = label.trim();
            let path = path.trim();
            if label.is_empty() || path.is_empty() {
                return None;
            }
            Some(KernelEntry {
                label: label.to_string(),
                sanitized: SanitizedKernelLabel::new(label),
                kernel_dir: PathBuf::from(path),
            })
        })
        .collect()
}

/// Read [`crate::KTSTR_KERNEL_LIST_ENV`] and parse it into a
/// `Vec<KernelEntry>`. Empty / unset / malformed → empty vec
/// (single-kernel mode at the call site).
pub(crate) fn read_kernel_list() -> Vec<KernelEntry> {
    std::env::var(crate::KTSTR_KERNEL_LIST_ENV)
        .ok()
        .map(|v| parse_kernel_list(&v))
        .unwrap_or_default()
}

/// Sanitise a kernel label (the producer-side identity emitted by
/// `cargo ktstr`'s resolver) into a nextest-safe identifier of the
/// shape `kernel_[a-z0-9_]+`.
///
/// Replaces every `[^A-Za-z0-9]` byte with `_`, lowercases, collapses
/// runs of `_`, and prefixes with `kernel_`. Empty / pathologically-
/// short input collapses to `kernel_` alone, which the parser
/// downstream still recognises as a valid suffix (the empty
/// `sanitized` marker just won't disambiguate two kernels — but the
/// producer side guarantees non-empty labels, so the empty case is
/// defensive only).
///
/// Example mappings:
/// - `6.14.2` → `kernel_6_14_2`
/// - `6.15-rc3` → `kernel_6_15_rc3`
/// - `git_tj_sched_ext_for-next` → `kernel_git_tj_sched_ext_for_next`
/// - `path_linux_a3f2b1` → `kernel_path_linux_a3f2b1`
pub fn sanitize_kernel_label(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len() + 7);
    out.push_str("kernel_");
    let mut last_underscore = true; // suppress leading `_` after `kernel_`
    for ch in raw.chars() {
        let c = ch.to_ascii_lowercase();
        if c.is_ascii_alphanumeric() {
            out.push(c);
            last_underscore = false;
        } else if !last_underscore {
            out.push('_');
            last_underscore = true;
        }
    }
    // Strip a trailing `_` so a label like `for-next-` doesn't
    // produce a dangling separator.
    if out.ends_with('_') && out.len() > "kernel_".len() {
        out.pop();
    }
    out
}

ctor::declarative::ctor! {
/// Early dispatch for `#[ktstr_test]` test execution.
///
/// Runs before `main()` in any binary that links against ktstr.
///
/// When running as PID 1 (the binary is `/init` in the VM), calls
/// `ktstr_guest_init()` which handles the full init lifecycle and never
/// returns.
///
/// - `--ktstr-test-fn=NAME --ktstr-topo=NnNlNcNt`: host-side dispatch —
///   boots a VM with the specified topology and runs the test inside it.
/// - `--ktstr-test-fn=NAME` (without `--ktstr-topo`): guest-side dispatch —
///   runs the test function directly (inside a VM that was already booted).
/// - nextest protocol (`--list`/`--exact`): intercepted when running
///   under nextest (`NEXTEST` env var set), delegates to [`ktstr_main`].
/// - Otherwise: no-op (falls through to the standard test harness).
///
/// ctor 1.0 ships both `#[ctor::ctor(...)]` (proc-macro attribute) and
/// `ctor::declarative::ctor! { ... }` (declarative block). This site
/// uses the declarative form because it sidesteps the TT-muncher
/// recursion-limit cost the proc-macro form would impose on the
/// ktstr_test expansion. The proc-macro form stays reachable via
/// `crate::__private::ctor::ctor` for downstream consumers that prefer
/// the attribute-on-fn shape; see `tests/private_module_paths.rs` for
/// the re-export contract.
#[doc(hidden)]
#[ctor(unsafe)]
pub fn ktstr_test_early_dispatch() {
    // PID 1: the binary is /init in the VM. Perform full init lifecycle
    // (mounts, scheduler, test dispatch, reboot). Never returns.
    if unsafe { libc::getpid() } == 1 {
        crate::vmm::rust_init::ktstr_guest_init();
    }

    // Export-self dispatch runs BEFORE host/guest test dispatch.
    // `cargo ktstr export` is a router that exec's the test binary
    // with `--ktstr-export-test=NAME`; the binary reads its own
    // `KTSTR_TESTS` registry, embeds itself via `current_exe`, and
    // writes the .run file. Running this check first means the
    // export path never accidentally triggers VM boot if the
    // operator simultaneously passes `--ktstr-test-fn` (the export
    // arg wins because export is a one-shot tool, not a test
    // execution).
    if let Some(code) = maybe_dispatch_export() {
        std::process::exit(code);
    }
    if let Some(code) = maybe_dispatch_shell_test() {
        std::process::exit(code);
    }
    if let Some(code) = maybe_dispatch_host_test() {
        std::process::exit(code);
    }
    // Propagate RUST_BACKTRACE / RUST_LOG from /proc/cmdline before
    // `maybe_dispatch_vm_test` runs: ctor context is single-threaded
    // (`.init_array` runs before any user thread exists), so this
    // `set_var` is sound and the later guest-side code that spawns
    // the probe thread observes the correct env.
    propagate_rust_env_from_cmdline();
    if let Some(code) = maybe_dispatch_vm_test() {
        // The LLVM profiling runtime registers its atexit handler via a
        // .init_array entry (C++ global initializer). Our ctor also lives
        // in .init_array, and the execution order between them is
        // non-deterministic. If our ctor runs first, the atexit handler
        // was never registered, so std::process::exit() won't write the
        // profraw. Serialize profraw to a buffer and write it to the SHM
        // ring for host-side extraction.
        try_flush_profraw();
        std::process::exit(code);
    }

    // nextest protocol: intercept --list and --exact when running under
    // nextest. Under cargo test, fall through to the standard harness
    // which runs the #[test] wrappers generated by #[ktstr_test].
    //
    // Binaries with real #[ktstr_test] entries need the ctor to handle
    // listing (gauntlet expansion) and dispatch (VM booting). The lib
    // test binary has only the dummy entry and no gauntlet variants —
    // skip interception so the standard harness discovers #[cfg(test)]
    // module #[test] functions (unit tests).
    //
    // For `--list`, ktstr_main prints the gauntlet/ktstr names and
    // RETURNS so the standard libtest harness can print its own list
    // of `#[test]` items afterward. This makes plain `#[test]`
    // functions inside a ktstr_test integration-test binary visible
    // to nextest — without the fall-through, libtest never runs and
    // those test names are silently dropped from the listing.
    //
    // For `--exact`, ktstr_main runs only when the test name starts
    // with `ktstr/` or `gauntlet/` — names ktstr owns. Other names
    // (libtest #[test] items, including the per-entry wrappers
    // emitted by `#[ktstr_test]` itself) fall through to libtest's
    // dispatch. Without this guard, run_named_test would fail
    // `find_test` for a plain `#[test]` name and exit 1, blocking
    // nextest from running it.
    if std::env::var_os("NEXTEST").is_some() {
        let has_real_tests = KTSTR_TESTS.iter().any(|e| !is_test_sentinel(e.name));
        // A binary may carry only `declare_scheduler!` declarations
        // (no `#[ktstr_test]` entries) — pure verifier-only test
        // binaries. Without the scheduler check below the listing
        // branch would never fire for such a binary and the
        // verifier cells would silently fail to emit under nextest.
        let has_schedulers = !super::KTSTR_SCHEDULERS.is_empty();
        if has_real_tests || has_schedulers {
            let args: Vec<String> = std::env::args().collect();
            if args.iter().any(|a| a == "--list") {
                ktstr_list_only();
                list_verifier_cells_all();
                list_plain_tests(args.iter().any(|a| a == "--ignored"));
                std::process::exit(0);
            } else if let Some(pos) = args.iter().position(|a| a == "--exact")
                && let Some(name) = args.get(pos + 1)
                && name.starts_with("verifier/")
            {
                // verifier/<sched>/<kernel>/<preset> cells bypass
                // libtest entirely — the cell handler resolves the
                // scheduler binary, kernel, and preset topology, runs
                // collect_verifier_output, prints the result, and
                // exits. No #[test] wrapper exists for declared
                // schedulers (declare_scheduler! only emits a static),
                // so it runs directly via run_verifier_cell — the same
                // libtest bypass the ktstr/ branch below uses.
                let code = run_verifier_cell(name);
                try_flush_profraw();
                std::process::exit(code);
            } else if let Some(pos) = args.iter().position(|a| a == "--exact")
                && let Some(name) = args.get(pos + 1)
                && (name.starts_with("ktstr/") || name.starts_with("gauntlet/"))
            {
                let bare = name
                    .strip_prefix("ktstr/")
                    .or_else(|| name.strip_prefix("gauntlet/"))
                    .unwrap_or(name)
                    .split('/')
                    .next()
                    .unwrap_or(name);

                // Reject malformed names like `gauntlet/` (trailing
                // slash, no test name) and `ktstr/` up front, so the
                // operator sees a clear error instead of an opaque
                // "unknown test" from the empty bare name.
                if bare.is_empty() {
                    eprintln!(
                        "ktstr: malformed --exact test name {name:?} \
                         (resolves to an empty bare name after prefix strip)",
                    );
                    std::process::exit(1);
                }

                // Run the entry directly, bypassing libtest — the same
                // pattern as the verifier/ branch above. The previous
                // dispatch rewrote argv to the bare name and relied on a
                // #[test] wrapper (emitted only by the #[ktstr_test]
                // macro) for libtest to match it; raw
                // `#[distributed_slice(KTSTR_TESTS)]` registrations have
                // no wrapper, so libtest matched nothing and printed
                // "running 0 tests" — a silent trivial-pass. run_named_test
                // resolves the entry from KTSTR_TESTS by name and boots it
                // for both registration styles, routing gauntlet/ to
                // run_gauntlet_test (identical topology) and applying the
                // host_only / performance_mode / bpf_map_write gates the
                // wrapper path skipped.
                let code = run_named_test(name);
                try_flush_profraw();
                std::process::exit(code);
            }
        }
    } else {
        // cargo-test-direct path: the standard rustc test harness
        // runs only the bare `#[test]` wrappers `#[ktstr_test]`
        // generates. Gauntlet expansion (topology-preset variants)
        // lives inside `ktstr_main`'s `--list` + `--exact` handlers
        // and is reachable ONLY under nextest. Every real ktstr
        // entry produces topology-preset variants under nextest
        // (`for_each_gauntlet_variant` iterates
        // `crate::gauntlet::gauntlet_presets()`). Without nextest those
        // variants would silently not run — coverage loss with no
        // error. Emit a one-shot stderr `warning:` diagnostic (see
        // the `eprintln!` below) when the binary carries any real
        // entry so the user sees the gap instead of trusting a
        // false green. Print once per process (cargo test invokes
        // one test binary per crate; the ctor runs exactly once per
        // test binary) so there is no need to gate with a
        // std::sync::Once.
        //
        // `KTSTR_CARGO_TEST_MODE=1` opts out of the warning: the
        // operator deliberately picked the cargo-test-direct path
        // (e.g. for a single-test debug iteration without the
        // nextest harness) and accepts that gauntlet variants
        // won't run. The warning is still emitted under bare
        // `cargo test` without the env var set so unaware users
        // see the coverage gap.
        if !crate::cargo_test_mode::cargo_test_mode_active() {
            let total = KTSTR_TESTS.len();
            let real = KTSTR_TESTS
                .iter()
                .filter(|e| !is_test_sentinel(e.name))
                .count();
            if real > 0 {
                eprintln!(
                    "warning: {real} of {total} ktstr test entries registered in this binary \
                     will not generate their topology-preset gauntlet variants — NEXTEST env \
                     var is not set and the standard rustc harness does not expand them. Use \
                     `cargo nextest run` (or `cargo ktstr test`) to exercise the full gauntlet, \
                     or set KTSTR_CARGO_TEST_MODE=1 to opt into single-variant bare-`cargo test` \
                     mode without this warning.",
                );
            }
            // Verifier cells are emitted by `list_verifier_cells_all`
            // which runs ONLY from the NEXTEST listing branch above.
            // A bare `cargo test` invocation on a binary carrying
            // `declare_scheduler!` declarations gets zero verifier
            // coverage — surface the gap with the same opt-out shape
            // as the gauntlet warning so an unaware operator does not
            // trust a green run that never reached the verifier.
            // Eevdf + KernelBuiltin variants don't produce userspace
            // binaries to verify, so they are excluded from the count
            // (matching the emission-time filter in
            // `list_verifier_cells_all`).
            let verifier_schedulers = super::KTSTR_SCHEDULERS
                .iter()
                .filter(|s| {
                    !matches!(
                        s.binary,
                        super::SchedulerSpec::Eevdf | super::SchedulerSpec::KernelBuiltin { .. }
                    )
                })
                .count();
            if verifier_schedulers > 0 {
                eprintln!(
                    "warning: {verifier_schedulers} `declare_scheduler!` declaration(s) in this \
                     binary will not generate verifier cells — NEXTEST env var is not set and \
                     verifier cells are emitted only by ktstr's `--list` handler under nextest. \
                     Use `cargo ktstr verifier` to exercise the verifier sweep, or set \
                     KTSTR_CARGO_TEST_MODE=1 to acknowledge the verifier-cell-free path without \
                     this warning.",
                );
            }
        }
    }
}
}

/// Predicate for "this entry is a unit-test sentinel, not a real
/// `#[ktstr_test]` user entry." The lib-test binary registers a
/// single sentinel entry (currently `"__unit_test_dummy__"`) so
/// the dispatch + gauntlet plumbing has something to exercise
/// under `cargo test --lib`; real user entries look like
/// `"module::test_name"` or similar PascalCase-with-dots names.
///
/// Matching the sentinel by convention (`__` prefix + `__`
/// suffix + `_test_` or `_dummy_` infix) rather than by literal
/// equality keeps the filter robust when the sentinel is
/// renamed, or when future scaffolding adds additional
/// sentinel-shaped entries (e.g. `__unit_test_panics__`,
/// `__unit_test_timeout__`). The literal-equality form would
/// silently admit those future sentinels into the real-entry
/// population and double-fire the "NEXTEST env var not set"
/// warning or spuriously enable --list interception.
fn is_test_sentinel(name: &str) -> bool {
    // Real user-authored `#[ktstr_test]` entry names
    // conventionally do not match the `__unit_test_*__` pattern
    // (Rust's reserved-identifier convention for
    // language-implementation and framework-internal names).
    // The `#[ktstr_test]` proc macro does not validate this, so
    // the predicate admits a real user entry in the unlikely
    // case someone names one with the `__unit_test_*__` shape —
    // collision would double-fire the "NEXTEST env var not set"
    // warning / spuriously enable --list interception, but
    // that's a diagnostic glitch, not a correctness failure.
    name.starts_with("__unit_test_") && name.ends_with("__")
}

/// Export-self dispatch: if `--ktstr-export-test=NAME` is present in
/// argv, look up `NAME` in the binary's own `KTSTR_TESTS` registry,
/// build a self-extracting `.run` file embedding `current_exe()`
/// (this binary), and exit. Returns `Some(exit_code)` when dispatched,
/// `None` when the flag is absent.
///
/// `cargo ktstr export <NAME>` (the cargo-ktstr binary) is a router
/// that compiles the workspace's tests, locates the test binary that
/// owns `NAME`, and exec's it with this arg. The test binary embeds
/// ITSELF — without that indirection, cargo-ktstr would package its
/// own binary, which has no `#[ktstr_test]` registrations from the
/// user's crate and can't reproduce the test on bare metal.
///
/// `--ktstr-export-output=PATH` overrides the default output path
/// (`<NAME>.run` in the cwd). Both flags are leniently parsed by the
/// helpers in `args.rs`; an empty NAME (`--ktstr-export-test=`)
/// surfaces with diagnostic "requires a non-empty test name" and
/// exit 1 so the router moves on to the next candidate.
///
/// # Exit-code contract
///
/// The router (`cargo-ktstr.rs::run_export`) discriminates between
/// "this binary doesn't know the test" (exit 1) and "this binary
/// has the test but rejects it" (exit 2). When ANY candidate exits
/// 2, the router surfaces THAT candidate's stderr (the rejection
/// reason: host_only, bpf_map_write, KernelBuiltin) rather than
/// the generic "not found in any workspace test binary" message.
/// Without the differentiation, an operator who exports a
/// host_only test would see the misleading "not found" diagnostic
/// even though the test exists.
/// Stub for the `export`-feature-disabled build. The router
/// (`cargo-ktstr.rs::run_export`) execs every candidate test binary
/// with `--ktstr-export-test=NAME`; without this stub a binary
/// compiled without `export` would fall through to the nextest
/// harness, which would surface an opaque "unrecognised argument"
/// error against an arg the operator never typed. The stub turns
/// that into an actionable diagnostic by detecting the arg and
/// emitting a build-config hint, then exiting 2 (matches the
/// "registered but rejected" exit code so the router surfaces
/// THIS binary's stderr rather than a sibling's "not registered"
/// fallthrough). Recompile the test binary with the `export`
/// feature (folded into `cli-bins` in the default feature set)
/// to enable the real `cargo ktstr export` flow.
#[cfg(not(feature = "export"))]
fn maybe_dispatch_export() -> Option<i32> {
    let args: Vec<String> = std::env::args().collect();
    let _ = extract_export_test_arg(&args)?;
    eprintln!(
        "ktstr export: this test binary was built without the `export` cargo \
         feature, so `cargo ktstr export <name>` cannot reach the export pipeline \
         from here. Rebuild with the default feature set (or pass \
         `--features cli-bins`) and retry."
    );
    Some(2)
}

#[cfg(feature = "export")]
fn maybe_dispatch_export() -> Option<i32> {
    let args: Vec<String> = std::env::args().collect();
    let name = extract_export_test_arg(&args)?;
    let output = extract_export_output_arg(&args).map(std::path::PathBuf::from);

    // Empty name: surface as a hard error rather than silently
    // succeeding. The router's "first binary that exits 0 wins"
    // protocol relies on the absent-test path returning a non-zero
    // exit so the next candidate is tried.
    if name.is_empty() {
        eprintln!("ktstr export: --ktstr-export-test= requires a non-empty test name");
        return Some(1);
    }

    // Look up the test ourselves so we can discriminate "not
    // registered here" (exit 1, router falls through) from
    // "registered but rejected" (exit 2, router surfaces this
    // stderr). `export_test` itself returns anyhow::Error for both
    // cases, which would conflate them at the exit-code level.
    if find_test(name).is_none() {
        eprintln!("ktstr export: no registered test named '{name}'");
        return Some(1);
    }

    match crate::export::export_test(name, output) {
        Ok(()) => Some(0),
        Err(e) => {
            eprintln!("ktstr export: {e:#}");
            // The test exists in this binary but the export pipeline
            // refused it (host_only / bpf_map_write / KernelBuiltin /
            // I/O error). Exit 2 so the router prefers this stderr
            // over a sibling binary's exit-1 "not registered" miss.
            Some(2)
        }
    }
}

/// Shell-self dispatch: if `--ktstr-shell-test=NAME` is present in
/// argv, look up `NAME` in the binary's own `KTSTR_TESTS` registry,
/// serialize its shell-relevant fields to stdout as JSON, and exit.
/// Returns `Some(exit_code)` when dispatched, `None` when absent.
///
/// `cargo ktstr shell --test <NAME>` (the cargo-ktstr binary) is a
/// router that compiles the workspace's tests, exec's each test
/// binary with this flag, and consumes the first stdout-JSON it
/// gets (the router bails on ambiguous names — same `NAME`
/// registered in two binaries). The router applies the
/// descriptor's topology / memory / extra_include_files to the
/// shell VM, then prints a one-line banner to stderr BEFORE VM
/// boot naming the test + scheduler so the operator can repro the
/// workload manually. (PS1-in-guest is a follow-up.)
///
/// # Stdout contract
///
/// The test binary MUST keep stdout silent on this dispatch path —
/// `tracing` output MUST go to stderr. The router parses the entire
/// stdout as a JSON descriptor; any prefix like an INFO log line
/// will fail the parse.
///
/// # JSON shape
///
/// Serialized from [`crate::test_support::ShellTestDescriptor`] via
/// `serde_json::to_string` — see that struct for the field-by-field
/// contract. The struct lives in
/// `crate::test_support::shell_descriptor` so producer and consumer
/// share a single definition; adding a field there automatically
/// propagates to both sides.
///
/// `scheduler_kind` discriminates `"eevdf" | "discover" | "path" |
/// "kernel_builtin"` so the banner can hint at how to repro the
/// scheduler (Discover/Path = userspace binary at `/bin/<n>`;
/// KernelBuiltin = no binary, the shell-mode boot runs
/// `scheduler_enable_cmds` before drop-to-busybox and
/// `scheduler_disable_cmds` on shell exit; Eevdf = no setup needed).
///
/// # Exit-code contract
///
/// Matches `maybe_dispatch_export`:
/// - `0`: test registered, JSON emitted to stdout.
/// - `1`: test not registered in this binary (router falls
///   through to the next candidate).
/// - `2`: registered but rejected for shell mode (currently:
///   `host_only` — no VM to drop into).
fn maybe_dispatch_shell_test() -> Option<i32> {
    let args: Vec<String> = std::env::args().collect();
    let name = extract_shell_test_arg(&args)?;

    if name.is_empty() {
        eprintln!("ktstr shell: --ktstr-shell-test= requires a non-empty test name");
        return Some(1);
    }

    let entry = match find_test(name) {
        Some(e) => e,
        None => {
            eprintln!("ktstr shell: no registered test named '{name}'");
            return Some(1);
        }
    };

    if entry.host_only {
        eprintln!(
            "ktstr shell: test '{name}' has host_only = true; \
             shell mode requires a guest VM to drop into. \
             Either run the test directly with `cargo ktstr test {name}` \
             (host_only tests don't boot a VM) or pick a non-host_only \
             test for shell mode."
        );
        return Some(2);
    }

    let topo = &entry.topology;
    let scheduler_kind = crate::test_support::SchedulerKind::from(&entry.scheduler.binary);
    let (scheduler_enable_cmds, scheduler_disable_cmds) = match &entry.scheduler.binary {
        crate::test_support::entry::SchedulerSpec::KernelBuiltin { enable, disable } => (
            enable.iter().copied().map(String::from).collect(),
            disable.iter().copied().map(String::from).collect(),
        ),
        _ => (Vec::new(), Vec::new()),
    };

    let descriptor = crate::test_support::ShellTestDescriptor {
        numa_nodes: topo.numa_nodes,
        llcs: topo.llcs,
        cores: topo.cores_per_llc,
        threads: topo.threads_per_core,
        memory_mib: entry.memory_mib,
        wprof: entry.wprof,
        extra_include_files: entry
            .extra_include_files
            .iter()
            .copied()
            .map(String::from)
            .collect(),
        scheduler_name: entry.scheduler.name.to_string(),
        scheduler_kind,
        wprof_args: entry.wprof_args.map(String::from),
        performance_mode: entry.performance_mode,
        scheduler_enable_cmds,
        scheduler_disable_cmds,
    };

    // serde_json::to_string produces RFC-8259-compliant escaping
    // (`\uXXXX` with 4 hex digits, surrogate pairs for SMP code
    // points) which Rust's Debug formatter does NOT — Debug uses
    // `\u{1f4c2}` (braced form) for non-ASCII, breaking
    // operator-supplied paths with non-ASCII chars (test built
    // under `/home/<unicode-name>/proj`, `extra_include_files`
    // listing emoji-named files, etc.). serde_json is already a
    // workspace dep so adding this call doesn't widen the dep graph.
    let payload = serde_json::to_string(&descriptor)
        .expect("ShellTestDescriptor is a plain serde struct with no fallible field types");
    println!("{payload}");
    Some(0)
}

/// Host-side dispatch: if both `--ktstr-test-fn` and `--ktstr-topo` are
/// present, boot a VM with the specified topology and run the test
/// inside it. Returns `Some(exit_code)` if dispatched, `None` otherwise.
fn maybe_dispatch_host_test() -> Option<i32> {
    let args: Vec<String> = std::env::args().collect();
    let name = extract_test_fn_arg(&args)?;
    let topo_str = extract_topo_arg(&args)?;

    let entry = match find_test(name) {
        Some(e) => e,
        None => {
            eprintln!("ktstr_test: unknown test function '{name}'");
            return Some(1);
        }
    };

    let (numa_nodes, llcs, cores, threads) = match parse_topo_string(&topo_str) {
        Some(t) => t,
        None => {
            eprintln!(
                "ktstr_test: invalid --ktstr-topo format '{topo_str}' (expected NnNlNcNt, e.g. 1n2l4c2t)"
            );
            return Some(1);
        }
    };

    let cpus = llcs * cores * threads;
    let memory_mib = super::runtime::derive_test_memory_mib(cpus, entry);
    let topo = TopoOverride {
        numa_nodes,
        llcs,
        cores,
        threads,
        memory_mib,
    };

    match run_ktstr_test_with_topo(entry, &topo) {
        Ok(_) => Some(0),
        Err(e) => {
            eprintln!("ktstr_test: {e:#}");
            Some(1)
        }
    }
}

/// Host-side entry point: build a VM, boot it with `--ktstr-test-fn=NAME`,
/// extract profraw from SHM, and return the test result.
///
/// Validates KVM access and auto-discovers a kernel image via
/// `resolve_test_kernel()` when `KTSTR_TEST_KERNEL` is not set.
pub fn run_ktstr_test(entry: &KtstrTestEntry) -> Result<AssertResult> {
    // Directly-constructed entries bypass the proc-macro's
    // compile-time checks. Call `validate` here so programmatic
    // consumers (library callers pushing into `KTSTR_TESTS`
    // dynamically) hit the same bail messages the macro produces at
    // compile time.
    entry.validate()?;

    if entry.host_only {
        return run_host_only_test_inner(entry);
    }
    if !entry.bpf_map_write.is_empty()
        && let Ok(kernel) = resolve_test_kernel()
        && crate::vmm::find_vmlinux(&kernel).is_none()
    {
        anyhow::bail!("vmlinux not found, bpf_map_write requires vmlinux");
    }
    run_ktstr_test_inner(entry, None)
}

/// Like `run_ktstr_test` but with an explicit topology override.
/// Only consumed inside this module by `maybe_dispatch_host_test`;
/// kept as a named helper so the `--ktstr-test-fn` + `--ktstr-topo`
/// dispatch path reads symmetrically with the zero-override
/// [`run_ktstr_test`] library entry point.
fn run_ktstr_test_with_topo(entry: &KtstrTestEntry, topo: &TopoOverride) -> Result<AssertResult> {
    run_ktstr_test_inner(entry, Some(topo))
}

/// Process exit code for a Pass verdict (and for the Skip path,
/// which degenerates to Pass because the test never ran).
///
/// Defined as a `pub const` so external tooling (CI gates,
/// dashboard aggregators, nextest wrappers) can reference the
/// exit-code triad by name instead of duplicating the integer
/// literals. The trio [`EXIT_PASS`] / [`EXIT_FAIL`] /
/// [`EXIT_INCONCLUSIVE`] cover every verdict produced by the
/// `Fail > Inconclusive > Pass > Skip` lattice when projected
/// to a process exit code.
pub const EXIT_PASS: i32 = 0;

/// Process exit code for a Fail verdict (or any expect_err
/// satisfaction failure).
///
/// See [`EXIT_PASS`] for the full triad rationale.
pub const EXIT_FAIL: i32 = 1;

/// Process exit code for an Inconclusive verdict (a
/// zero-denominator ratio gate that could not evaluate).
///
/// Distinct from [`EXIT_PASS`] (which would silently green an
/// unevaluated gate) and [`EXIT_FAIL`] (which would conflate
/// "could not evaluate" with a real regression). External tooling
/// uses this code to triage Inconclusive runs separately — see
/// the README "Exit codes" section for the full operator contract.
pub const EXIT_INCONCLUSIVE: i32 = 2;

/// Run a test result through expect_err logic and return an exit code.
///
/// Returns [`EXIT_PASS`] on pass, [`EXIT_FAIL`] on failure, and
/// [`EXIT_INCONCLUSIVE`] on Inconclusive — the 4-state lattice
/// `Fail > Inconclusive > Pass > Skip` projects to 3 distinct exit
/// codes (Skip degenerates to [`EXIT_PASS`] because the test never
/// ran, mirroring `ResourceContention`). A Skip routes through the
/// dedicated FIRST match arm (`Ok(r) if r.is_skip()`), ahead of the
/// expect_err arm, so an expect_err test that produced no verdict (e.g.
/// a `post_vm_skip` on a load-starved placeholder dump) is not inverted
/// into a FAIL — a skipped test cannot "produce the expected error."
/// [`EXIT_INCONCLUSIVE`] lets
/// downstream tooling (CI gates, nextest summary aggregation, the
/// operator dashboard) triage zero-denominator runs distinctly from
/// real regressions. `ResourceContention` returns [`EXIT_PASS`] —
/// the test never ran, not a real failure. The skip sidecar for
/// this case is written upstream in `run_ktstr_test_inner` at the
/// ResourceContention propagation site so every caller (including
/// the library entry point `run_ktstr_test`) records it, not just
/// the nextest dispatch path.
///
/// `ResourceContention` detection walks the FULL error chain via
/// [`is_resource_contention`] (chain-walk predicate) plus a
/// matching `e.chain().find_map(...)` extraction for the reason
/// string. The eval-side `crate::test_support::eval` `"build ktstr_test VM"` and
/// `"run ktstr_test VM"` wrappers nest the contention error under
/// `.context(...)`, so a top-level `downcast_ref` on the outer
/// error misses the inner cause. Without the chain walk a wrapped
/// contention would land in the `Err(e)` arm below as a regular
/// failure (exit 1) rather than the skip path (exit 0), turning
/// every host-resource-exhausted run into a hard test failure.
fn result_to_exit_code(
    result: Result<AssertResult>,
    expect_err: bool,
    allow_inconclusive: bool,
) -> i32 {
    let no_skip = std::env::var_os(crate::KTSTR_NO_SKIP_MODE_ENV).is_some();
    match result {
        Ok(r) => ok_to_exit_code(r, expect_err, allow_inconclusive),
        Err(e) => err_to_exit_code(e, expect_err, no_skip),
    }
}

/// Map an `Ok(AssertResult)` verdict to an exit code.
///
/// The sequential guards preserve the original `match` arm precedence
/// (first matching guard wins): `is_skip()` → `expect_err` →
/// `is_inconclusive()` → the trailing `EXIT_PASS` (the former
/// `Ok(_) => EXIT_PASS` arm). Reordering these would change which
/// verdict fires for a result matching more than one guard.
fn ok_to_exit_code(r: AssertResult, expect_err: bool, allow_inconclusive: bool) -> i32 {
    // A Skip degenerates to EXIT_PASS regardless of expect_err — the
    // test never evaluated, so there is no guest failure to "expect"
    // (the `Fail > Inconclusive > Pass > Skip` projection; mirrors the
    // ResourceContention Err branch in `err_to_exit_code`, but on the
    // Ok side). Without this guard a post_vm_skip under expect_err
    // falls into the `expect_err` guard below and surfaces as "expected
    // error but test passed" (EXIT_FAIL) — a load-starvation
    // placeholder-dump skip becomes a flaky failure. End-to-end chain:
    // a post_vm callback returns Err(post_vm_skip(..)) → the eval gate
    // detects the HostSkipRequest marker, reports via report::test_skip,
    // and returns Ok(AssertResult::skip) → this guard maps it to
    // EXIT_PASS. is_skip() is true only when `outcomes` is non-empty and
    // every outcome is Outcome::Skip (assert/plan.rs); the empty-outcomes
    // Pass identity has is_skip()==false and falls through to the
    // trailing `EXIT_PASS`.
    if r.is_skip() {
        return EXIT_PASS;
    }
    if expect_err {
        // expect_err inverts on Pass and on Inconclusive: both
        // are "not a failure" in the operator's mental model,
        // and an expect_err scenario that produces an
        // Inconclusive verdict (denominator zero) failed to
        // produce the expected failure just like a Pass would.
        // Surface the inconclusive as exit code 2 to preserve
        // the distinct verdict, but treat it as expect_err
        // satisfaction failure (exit 1) — the test author
        // wanted a Fail, not "the gate could not run".
        //
        // `allow_inconclusive` does NOT relax the expect_err
        // contract: expect_err demands a real Fail, and an
        // Inconclusive verdict does not satisfy that
        // regardless of how the test author scopes
        // Inconclusive elsewhere. The dominant gate wins;
        // `allow_inconclusive` only relaxes the
        // EXIT_INCONCLUSIVE projection on the no-expect_err
        // path below.
        if r.is_inconclusive() {
            eprintln!(
                "expected error but test produced an Inconclusive verdict — \
                 zero-denominator gate could not evaluate; expect_err is \
                 unsatisfied"
            );
            return EXIT_FAIL;
        } else {
            eprintln!("expected error but test passed");
            return EXIT_FAIL;
        }
    }
    if r.is_inconclusive() {
        // `allow_inconclusive` opt-in: a test author may have
        // declared `#[ktstr_test(allow_inconclusive)]` to
        // signal "this test's Inconclusive arm is acceptable —
        // don't fail the CI gate." Route to EXIT_PASS in that
        // case (Inconclusive is still recorded in the sidecar
        // for stats tooling and the operator-facing failure
        // dump still renders the diagnostic). When the flag
        // is unset (the default) the verdict surfaces as
        // EXIT_INCONCLUSIVE so the operator triages it.
        if allow_inconclusive {
            eprintln!(
                "test produced an Inconclusive verdict but \
                 `allow_inconclusive` is set — routing to EXIT_PASS \
                 for CI gate, sidecar still records Inconclusive"
            );
            return EXIT_PASS;
        } else {
            return EXIT_INCONCLUSIVE;
        }
    }
    EXIT_PASS
}

/// Map an `Err(anyhow::Error)` outcome to an exit code.
///
/// The sequential guards preserve the original `match` arm precedence
/// (first matching guard wins): the host-insufficiency classification
/// ([`classify_host_error`], covering kernel-unavailable → perf-mode →
/// cpu-budget → topology-unrepresentable → resource-contention →
/// topology-insufficient, shared with the `#[ktstr_test]` macro body) runs
/// FIRST, then the
/// marker-typed guards (`PostVmAssertionFailure` → `SchedulerBuildRefused`
/// → `SurvivesStormViolated` → `ExpectAutoReproSatisfied`), then the
/// `expect_err` inversion, then
/// the catch-all (the former `Err(e) => …` arm) operating on the
/// now-owned `e`. Reordering these would change which guard fires for an
/// error matching more than one guard. The host-insufficiency guard
/// order + per-class skip/fail policy live in `classify_host_error`, not
/// here, so this site and the macro cannot drift apart.
fn err_to_exit_code(e: anyhow::Error, expect_err: bool, no_skip: bool) -> i32 {
    // Host-insufficiency classification (kernel-unavailable, perf-mode,
    // cpu-budget, topology-unrepresentable, resource-contention,
    // topology-insufficient) is shared with the `#[ktstr_test]` macro body via
    // `classify_host_error` — the single source of truth for the guard
    // ORDER and the per-class skip/fail policy. This site renders the
    // verdict as an exit code; the macro renders the same `HostClass` as
    // libtest control flow. The bare `reason` carries no prefix: the skip
    // channel (`report::test_skip`) prepends `ktstr: SKIP:`, the fail
    // channel prepends `ktstr: FAIL:`. Placed first so a host-insufficiency
    // returns before the marker / expect_err / catch-all arms below — a
    // skip is a skip and an unconditional hard fail is a hard fail
    // regardless of `expect_err`.
    match classify_host_error(&e, no_skip) {
        HostClass::Skip { reason } => {
            crate::report::test_skip(format_args!("{reason}"));
            return EXIT_PASS;
        }
        HostClass::Fail { reason } => {
            eprintln!("ktstr: FAIL: {reason}");
            return EXIT_FAIL;
        }
        HostClass::NotHostClass => {}
    }
    if e.downcast_ref::<crate::test_support::eval::PostVmAssertionFailure>()
        .is_some()
    {
        // A host-side post_vm / post_vm_unconditional callback
        // failed. This is a real regression that must surface
        // regardless of expect_err / expect_auto_repro inversion —
        // those invert a GUEST-side expected failure, but a
        // HOST-side check is always honored. Positioned AFTER the
        // resource-contention / topology skip guards (a skip means
        // the test never ran, so there was no host-side state to
        // assert) but BEFORE the ExpectAutoReproSatisfied and
        // expect_err inversion guards so the host-side regression
        // wins. `downcast_ref` walks the anyhow context+source
        // chain (the marker rides as `.context(...)` from
        // run_ktstr_test_inner_impl); a raw `chain().any(is::<C>())`
        // would miss it (anyhow boxes context as ContextError<C,E>).
        eprintln!("{e:#}");
        return EXIT_FAIL;
    }
    if e.downcast_ref::<crate::test_support::eval::SchedulerBuildRefused>()
        .is_some()
    {
        // An orchestrated scheduler build expected to succeed FAILED and the
        // resolver refused to validate against a possibly-stale pre-built
        // binary (KTSTR_SCHEDULER_ALLOW_STALE_FALLBACK unset). A host-side
        // build-infra fault — always EXIT_FAIL, never inverted by expect_err
        // (mirrors PostVmAssertionFailure above): an expect_err test must not
        // let a broken build masquerade as the guest-side expected failure.
        eprintln!("{e:#}");
        return EXIT_FAIL;
    }
    if e.downcast_ref::<crate::test_support::eval::SurvivesStormViolated>()
        .is_some()
    {
        // The marker rides ONLY when `entry.survives_storm` was set AND the
        // failure cause was a scheduler death (see
        // `render_failure_verdict_message`), so its presence alone proves the
        // survival assertion was violated — no `survives_storm` param needed
        // (mirrors the marker-presence arms for PostVmAssertionFailure /
        // SchedulerBuildRefused / ExpectAutoReproSatisfied below). Force
        // EXIT_FAIL with a survival-specific explainer. Positioned AFTER the
        // host-insufficiency / PostVmAssertionFailure / SchedulerBuildRefused
        // guards (a skip or host-side fault still dominates) but BEFORE the
        // ExpectAutoReproSatisfied and expect_err inversion arms so a survival
        // violation can never be inverted to PASS (defense-in-depth: the
        // validate-time survives_storm/expect_err mutex already forbids that
        // pairing). `downcast_ref` walks the anyhow context chain (the marker
        // rides as `.context(...)`).
        eprintln!(
            "ktstr: FAIL: survives_storm asserted but the scheduler did not \
             survive the run:\n{e:#}"
        );
        return EXIT_FAIL;
    }
    if e.downcast_ref::<crate::test_support::eval::ExpectAutoReproSatisfied>()
        .is_some()
    {
        // `expect_auto_repro = true` was satisfied: the primary
        // VM produced a Fail AND the auto-repro VM landed a
        // shape-valid `.repro.wprof.pb`. The eval layer attached
        // the marker as `anyhow::Context`. `downcast_ref` walks
        // the anyhow context+source chain (per anyhow's
        // documentation: "For errors with context, this method
        // returns true if E matches the type of the context C or
        // the type of the error on which the context has been
        // attached"). A `chain().any(|c| c.is::<E>())` walk on
        // the raw `&dyn StdError` chain would MISS the marker
        // because anyhow boxes context as `ContextError<C, E>`
        // whose underlying `is::<C>()` check returns false. The
        // diagnostic is printed so the operator sees both the
        // original failure trail and the inversion notice — the
        // verdict flips to PASS without erasing the failure
        // detail. Positioned AFTER the ResourceContention /
        // TopologyInsufficient guards so a skip-class outcome still
        // wins over inversion (a skip is a skip regardless of the
        // satisfaction signal). The macro-parse cross-attribute
        // check rejects `expect_auto_repro` combined with
        // `expect_err`, so the two inversion paths are mutually
        // exclusive at the entry layer.
        eprintln!("{e:#}");
        return EXIT_PASS;
    }
    if expect_err {
        // expect_err inverts a failure into a pass — UNLESS the
        // failure carries the
        // [`crate::test_support::eval::ScxBpfErrorMatcherMismatch`]
        // marker, which signals that the reproducer's scx_bpf_error
        // matcher rejected this particular failure. A matcher-
        // mismatch failure must surface even when expect_err = true:
        // the user authored the matcher to pin THIS specific bug,
        // and a different bug firing is itself a regression.
        //
        // `downcast_ref` walks the anyhow context+source chain
        // (anyhow's documented "For errors with context, this
        // method returns true if E matches the type of the context
        // C or the type of the error on which the context has been
        // attached" semantics). A `chain().any(|c| c.is::<E>())`
        // walk on the raw `&dyn StdError` chain would MISS the
        // marker because anyhow boxes context as
        // `ContextError<C, E>` whose underlying `is::<C>()` check
        // returns false.
        if e.downcast_ref::<crate::test_support::eval::ScxBpfErrorMatcherMismatch>()
            .is_some()
        {
            eprintln!("{e:#}");
            return EXIT_FAIL;
        } else {
            return EXIT_PASS;
        }
    }
    // Catch-all: a non-host-class, non-marker, non-expect_err error is a
    // real failure. (A KernelUnavailable does NOT reach here — it is a
    // skip-class host-insufficiency handled by the classify_host_error match
    // at the top.)
    eprintln!("{e:#}");
    EXIT_FAIL
}

/// The final test verdict — the 4-state lattice `Fail > Inconclusive >
/// Pass > Skip` that [`result_to_exit_code`] projects to a process exit
/// code. Distinct from the exit code because the exit code collapses
/// `Skip` into [`EXIT_PASS`]; the sidecar finalize ([`final_outcome`])
/// needs all four to set the persisted `passed`/`skipped`/`inconclusive`
/// bits to the POST-inversion outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Verdict {
    Pass,
    Fail,
    Skip,
    Inconclusive,
}

impl Verdict {
    /// Project to the process exit code, matching the
    /// `EXIT_PASS`/`EXIT_FAIL`/`EXIT_INCONCLUSIVE` mapping
    /// [`result_to_exit_code`] produces (Skip degenerates to
    /// [`EXIT_PASS`]). Test-only: the anti-drift truth-table test
    /// (`final_outcome_projects_to_result_to_exit_code`) is its sole
    /// caller — production reads the [`Verdict`] directly via
    /// [`Verdict::sidecar_bits`].
    #[cfg(test)]
    pub(crate) fn to_exit_code(self) -> i32 {
        match self {
            Verdict::Pass | Verdict::Skip => EXIT_PASS,
            Verdict::Fail => EXIT_FAIL,
            Verdict::Inconclusive => EXIT_INCONCLUSIVE,
        }
    }

    /// The persisted-sidecar verdict bits `(passed, skipped,
    /// inconclusive)` for this outcome. `Fail` is all-false (the
    /// [`crate::test_support::SidecarResult::is_fail`] "none set"
    /// encoding). Lets the sidecar finalize record the final verdict
    /// without [`crate::test_support::sidecar`] depending on this enum.
    pub(crate) fn sidecar_bits(self) -> (bool, bool, bool) {
        match self {
            Verdict::Pass => (true, false, false),
            Verdict::Skip => (false, true, false),
            Verdict::Inconclusive => (false, false, true),
            Verdict::Fail => (false, false, false),
        }
    }
}

/// Classify a test result into the final [`Verdict`] — the same
/// classification [`result_to_exit_code`] performs, as a 4-state value
/// (it does not collapse `Skip` into `Pass` the way the exit code does)
/// and WITHOUT the operator-facing `eprintln` diagnostics.
///
/// Used to record the FINAL (post-`expect_err` / post-marker) outcome on
/// the sidecar so the footer, `stats` analysis, and `replay` reflect the
/// test's real pass/fail (matching nextest's exit code) rather than the
/// raw scenario verdict written mid-run.
///
/// MUST stay in lockstep with [`result_to_exit_code`]: the truth-table
/// test `final_outcome_projects_to_result_to_exit_code` asserts
/// `final_outcome(...).to_exit_code() == result_to_exit_code(...)` over a
/// matrix including the marker-carrying error arms, so the two cannot
/// drift. The arm order mirrors [`ok_to_exit_code`] / [`err_to_exit_code`]
/// first-match precedence exactly.
pub(crate) fn final_outcome(
    result: &Result<AssertResult>,
    expect_err: bool,
    allow_inconclusive: bool,
) -> Verdict {
    let no_skip = std::env::var_os(crate::KTSTR_NO_SKIP_MODE_ENV).is_some();
    match result {
        Ok(r) => {
            if r.is_skip() {
                return Verdict::Skip;
            }
            if expect_err {
                // expect_err on an Ok result is always a failure
                // (expected an error, got a non-error verdict) — both the
                // Pass and Inconclusive arms of ok_to_exit_code map here.
                return Verdict::Fail;
            }
            if r.is_inconclusive() {
                return if allow_inconclusive {
                    Verdict::Pass
                } else {
                    Verdict::Inconclusive
                };
            }
            Verdict::Pass
        }
        Err(e) => {
            match classify_host_error(e, no_skip) {
                HostClass::Skip { .. } => return Verdict::Skip,
                HostClass::Fail { .. } => return Verdict::Fail,
                HostClass::NotHostClass => {}
            }
            if e.downcast_ref::<crate::test_support::eval::PostVmAssertionFailure>()
                .is_some()
            {
                return Verdict::Fail;
            }
            if e.downcast_ref::<crate::test_support::eval::SchedulerBuildRefused>()
                .is_some()
            {
                return Verdict::Fail;
            }
            if e.downcast_ref::<crate::test_support::eval::SurvivesStormViolated>()
                .is_some()
            {
                // Lockstep with err_to_exit_code's SurvivesStormViolated arm
                // (same position: after SchedulerBuildRefused, before
                // ExpectAutoReproSatisfied / expect_err) so the persisted
                // sidecar verdict matches the exit code for a survival
                // violation — including the defense-in-depth bypass case
                // (marker + expect_err) the mutex normally forbids.
                return Verdict::Fail;
            }
            if e.downcast_ref::<crate::test_support::eval::ExpectAutoReproSatisfied>()
                .is_some()
            {
                return Verdict::Pass;
            }
            if expect_err {
                if e.downcast_ref::<crate::test_support::eval::ScxBpfErrorMatcherMismatch>()
                    .is_some()
                {
                    return Verdict::Fail;
                }
                return Verdict::Pass;
            }
            Verdict::Fail
        }
    }
}

/// Whether a base test entry is "ignored" (skipped by default).
///
/// Tests whose names start with `demo_` are ignored -- they are
/// demonstration/benchmarking tests that require manual opt-in.
fn is_ignored(entry: &KtstrTestEntry) -> bool {
    entry.name.starts_with("demo_")
}

/// Walk [`KTSTR_TESTS`] once per process and emit a stderr
/// `warning:` line for every duplicate `name` found.
///
/// Two entries with the same name would both match `find_test(name)`
/// (which returns the FIRST match), so the second registration is
/// silently shadowed — `cargo ktstr` would dispatch the first entry
/// and the second entry's body would never run, with no diagnostic
/// surfaced. The warning surfaces the collision so an operator can
/// rename one of the `#[ktstr_test]` functions; discovery itself
/// proceeds (find_test's first-wins behavior continues) so nextest's
/// `--list` output still lands in stdout. A panic here would abort
/// the whole listing — nextest would see no tests at all rather
/// than a partial set with a clear warning. The first-wins
/// shadowing remains a real bug, but the diagnostic is louder than
/// silence and the tradeoff (operator sees the warning AND a
/// usable test list) beats the alternative (operator sees a
/// panic backtrace and no test list).
///
/// `OnceLock<()>` gates the walk to fire EXACTLY ONCE per process:
/// every gauntlet variant resolves through `list_tests` (under
/// nextest's discovery and budget paths), so without the gate a
/// run with N variants would re-walk the slice N times and emit
/// the same warning N times. Each duplicate name surfaces exactly
/// once via the inner `seen`/`warned` HashSet pair so a
/// triple-collision (three entries sharing one name) does not
/// double-print the warning.
///
/// The pure detection logic lives in
/// [`warn_duplicate_test_names_inner`] so the duplicate-walker
/// is testable without process-wide global state. This wrapper
/// only owns the `OnceLock<()>` gate and the
/// `(KTSTR_TESTS, stderr)` plumbing.
fn warn_duplicate_test_names_once() {
    static CHECKED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    CHECKED.get_or_init(|| {
        warn_duplicate_test_names_inner(KTSTR_TESTS.iter().map(|e| e.name), &mut std::io::stderr());
    });
}

/// Pure walker behind [`warn_duplicate_test_names_once`]: walks
/// the test-name iterator and emits one `warning:` line per
/// duplicate name to `sink`. Each duplicate name surfaces
/// exactly once (a triple-collision does NOT double-print)
/// via the inner `warned` HashSet.
///
/// Extracted from the OnceLock-gated wrapper so the duplicate
/// detection logic is testable without process-wide global
/// state — the wrapper handles "fire once per process" via its
/// own `OnceLock<()>` gate; this inner is a pure function over
/// `(names, sink)`. The wrapper passes
/// `KTSTR_TESTS.iter().map(|e| e.name)` as the iterator and
/// `std::io::stderr()` as the sink.
///
/// `Result<(), std::io::Error>` is collapsed to ignore-on-write
/// because the production wrapper writes to stderr where IO
/// errors are unrecoverable; tests pass a `Vec<u8>` sink which
/// never errors. The function name says "warn" — diagnostic
/// channel — and matches the wrapper's pre-existing
/// `eprintln!` semantics.
fn warn_duplicate_test_names_inner<'a, W: std::io::Write>(
    names: impl IntoIterator<Item = &'a str>,
    sink: &mut W,
) {
    use std::collections::HashSet;
    let names: Vec<&'a str> = names.into_iter().collect();
    let mut seen: HashSet<&'a str> = HashSet::with_capacity(names.len());
    let mut warned: HashSet<&'a str> = HashSet::new();
    for name in names {
        if !seen.insert(name) && warned.insert(name) {
            let _ = writeln!(
                sink,
                "warning: ktstr_test: duplicate test name {name:?} registered in KTSTR_TESTS — \
                 two `#[ktstr_test]` entries share this name; the SECOND entry is \
                 silently shadowed (find_test returns the first registration). \
                 rename one of the functions to disambiguate.",
            );
        }
    }
}

/// Collect test names for nextest discovery (--list --format terse).
///
/// Nextest calls the binary twice:
/// - Without `--ignored`: prints ALL tests (ignored and non-ignored).
/// - With `--ignored`: prints ONLY ignored tests.
///
/// Gauntlet variants are always ignored. Base tests are ignored when
/// their name starts with `demo_`.
///
/// When `KTSTR_BUDGET_SECS` is set, applies greedy coverage maximization
/// to select the subset of tests that maximizes feature coverage within
/// the time budget. Only selected tests are printed.
///
/// Calls [`warn_duplicate_test_names_once`] on the first invocation per
/// process so duplicate registrations surface a stderr `warning:`
/// line BEFORE any test name is printed (discovery itself proceeds
/// — find_test's first-wins behavior continues, but the operator
/// sees which name collided). Subsequent invocations are no-ops via
/// the inner `OnceLock` gate.
fn list_tests(ignored_only: bool) {
    warn_duplicate_test_names_once();
    let raw = std::env::var(crate::KTSTR_BUDGET_SECS_ENV).ok();
    let budget_secs: Option<f64> = raw.as_deref().and_then(|s| match s.parse::<f64>() {
        Ok(v) if v > 0.0 => Some(v),
        Ok(v) => {
            eprintln!("ktstr_test: KTSTR_BUDGET_SECS={v}: must be positive, ignoring");
            None
        }
        Err(e) => {
            eprintln!("ktstr_test: KTSTR_BUDGET_SECS={s:?}: {e}, ignoring");
            None
        }
    });

    if let Some(budget) = budget_secs {
        list_tests_budget(ignored_only, budget);
    } else {
        list_tests_all(ignored_only);
    }
}

/// Iterate topology presets that both fit the host capacity and
/// match the entry's `TopologyConstraints`. Shared between the
/// eager ("print every name") and budgeted ("push a candidate")
/// listers in `list_tests_*`.
fn for_each_gauntlet_variant<F>(
    entry: &KtstrTestEntry,
    presets: &[crate::gauntlet::TopoPreset],
    host_cpus: u32,
    host_llcs: u32,
    host_max_cpus_per_llc: u32,
    mut visit: F,
) where
    F: FnMut(&crate::gauntlet::TopoPreset),
{
    let no_perf_mode = super::runtime::no_perf_mode_for_entry(entry);
    for preset in presets {
        // No-perf-mode tests run KVM-emulated topology — guest sees the
        // declared NUMA / LLC / per-LLC layout regardless of host
        // hardware — so the host-side LLC count and per-LLC CPU width
        // do not constrain preset eligibility. Only the total-CPU
        // budget survives.
        let accepted = if no_perf_mode {
            entry
                .constraints
                .accepts_no_perf_mode(&preset.topology, host_cpus)
        } else {
            entry.constraints.accepts(
                &preset.topology,
                host_cpus,
                host_llcs,
                host_max_cpus_per_llc,
            )
        };
        if !accepted {
            continue;
        }
        visit(preset);
    }
}

/// List all tests without budget filtering.
///
/// When `KTSTR_KERNEL_LIST` carries 2 or more entries, every test
/// name carries an extra `/{sanitized_kernel_label}` suffix so each
/// (test × kernel) pair becomes a distinct nextest test case;
/// nextest's parallelism, retries, and `-E` filtering all apply
/// natively. Single-kernel mode (0 or 1 entries) emits the
/// `gauntlet/{name}/{preset}` shape with no kernel suffix.
///
/// `KTSTR_CARGO_TEST_MODE=1` skips gauntlet variant emission and
/// the multi-kernel suffix path: each test gets exactly one
/// `ktstr/{name}: test` line. Bare `cargo test` doesn't have
/// access to the cargo-ktstr resolver that produces
/// `KTSTR_KERNEL_LIST`, so the multi-kernel branch can't apply
/// even if it were enabled — pin both behaviors explicitly so
/// the listing matches what the dispatch path will actually run.
fn list_tests_all(ignored_only: bool) {
    let cargo_test_mode = crate::cargo_test_mode::cargo_test_mode_active();
    let presets = crate::gauntlet::gauntlet_presets();
    let has_vmlinux = resolve_test_kernel()
        .ok()
        .and_then(|k| crate::vmm::find_vmlinux(&k))
        .is_some();
    let (host_cpus, host_llcs, host_max_cpus_per_llc) = super::host_capacity();

    let kernel_list = read_kernel_list();
    let multi_kernel = kernel_list.len() > 1 && !cargo_test_mode;
    // Single-kernel mode (no list, or list has exactly one entry)
    // emits one variant per (test × preset) tuple with no kernel
    // suffix. Multi-kernel mode iterates every kernel as an outer
    // loop and appends `/{sanitized}` per variant. The empty-suffix
    // sentinel below is what the single-kernel branch passes to keep
    // the print path uniform.
    let kernel_suffixes: Vec<&str> = if multi_kernel {
        kernel_list.iter().map(|k| k.sanitized.as_str()).collect()
    } else {
        vec![""]
    };

    for entry in KTSTR_TESTS.iter() {
        // bpf_map_write tests require vmlinux to resolve BPF map
        // addresses. Don't list them when vmlinux is unavailable —
        // they cannot run and would produce false PASS results.
        if !entry.bpf_map_write.is_empty() && !has_vmlinux {
            continue;
        }

        if !ignored_only || is_ignored(entry) {
            if entry.host_only {
                println!("ktstr/{}: test", entry.name);
            } else {
                for suffix in &kernel_suffixes {
                    if suffix.is_empty() {
                        println!("ktstr/{}: test", entry.name);
                    } else {
                        println!("ktstr/{}/{suffix}: test", entry.name);
                    }
                }
            }
        }

        // Host-only tests run on the host without a VM -- gauntlet
        // topology variants are meaningless.
        if entry.host_only {
            continue;
        }

        // KTSTR_CARGO_TEST_MODE: skip gauntlet expansion. The
        // operator picked the bare-`cargo test` path; emit only
        // the base name so each `#[ktstr_test]` runs once with its
        // declared topology.
        if cargo_test_mode {
            continue;
        }

        // Gauntlet variants are always ignored — users opt in with
        // --run-ignored. Presets that exceed the host's CPU count or
        // LLC count are filtered from the listing entirely.
        for_each_gauntlet_variant(
            entry,
            &presets,
            host_cpus,
            host_llcs,
            host_max_cpus_per_llc,
            |preset| {
                for suffix in &kernel_suffixes {
                    if suffix.is_empty() {
                        println!("gauntlet/{}/{}: test", entry.name, preset.name);
                    } else {
                        println!("gauntlet/{}/{}/{suffix}: test", entry.name, preset.name,);
                    }
                }
            },
        );
    }
}

/// True iff the given operator-resolved kernel `entry` matches one
/// of the `declared` kernel specs from a scheduler's
/// `declare_scheduler!` `kernels = [...]` declaration. Empty
/// `declared` accepts every entry (no per-scheduler filter).
///
/// Match semantics per spec variant (via [`crate::kernel_path::KernelId::parse`]):
/// - [`crate::kernel_path::KernelId::Version`]: raw-label string equality OR sanitized-label match
///   ([`sanitize_kernel_label`] of the spec string equals the entry's
///   sanitized label). Direct match catches the common case where
///   the dispatcher resolved `--kernel 6.14.2` and the scheduler
///   declared `kernels = ["6.14.2"]`.
/// - [`crate::kernel_path::KernelId::Range`]: range-membership check on the entry's raw
///   label via [`crate::kernel_path::decompose_version_for_compare`].
///   Lets schedulers declaring `kernels = ["6.14..6.16"]` match
///   any operator-supplied kernel whose version falls in
///   `[6.14, 6.16]` inclusive.
/// - [`crate::kernel_path::KernelId::Path`] / [`crate::kernel_path::KernelId::CacheKey`] / [`crate::kernel_path::KernelId::Git`]:
///   sanitized-label equality — the producer-side encoder
///   (`src/bin/cargo_ktstr/kernel/wire_format.rs`) emits a deterministic
///   label per variant (`path_…`, `git_owner_repo_ref`, version
///   prefix from cache key), so identical specs on both sides
///   produce identical sanitized labels.
///
/// [`KernelId`]: crate::kernel_path::KernelId
fn sched_kernel_filter_accepts(declared: &[&'static str], entry: &KernelEntry) -> bool {
    if declared.is_empty() {
        return true;
    }
    declared.iter().any(|spec| entry_matches_spec(entry, spec))
}

/// Single-spec match helper for [`sched_kernel_filter_accepts`].
/// Parses `spec` via [`crate::kernel_path::KernelId::parse`] and
/// dispatches on the variant. Pure logic — no network, no FS.
fn entry_matches_spec(entry: &KernelEntry, spec: &str) -> bool {
    use crate::kernel_path::{KernelId, decompose_version_for_compare};
    match KernelId::parse(spec) {
        KernelId::Version(spec_ver) => {
            entry.label == spec_ver || entry.sanitized.as_str() == sanitize_kernel_label(&spec_ver)
        }
        KernelId::Range { start, end, .. } => {
            let Some(entry_t) = decompose_version_for_compare(&entry.label) else {
                return false;
            };
            let Some(start_t) = decompose_version_for_compare(&start) else {
                return false;
            };
            let Some(end_t) = decompose_version_for_compare(&end) else {
                return false;
            };
            entry_t >= start_t && entry_t <= end_t
        }
        KernelId::CacheKey(_) | KernelId::Path(_) | KernelId::Git { .. } => {
            entry.sanitized.as_str() == sanitize_kernel_label(spec)
        }
    }
}

/// Format the `KTSTR_KERNEL_LIST is empty` diagnostic emitted by
/// [`run_verifier_cell`] when a verifier cell name reaches the cell
/// handler with no kernel-list to look the label up in. Extracted
/// from the inline eprintln! so the exact wording can be pinned in
/// unit tests without spawning a process.
fn format_empty_kernel_list_error(full_name: &str) -> String {
    format!(
        "ktstr verifier: cell {full_name}: KTSTR_KERNEL_LIST is empty. \
         Direct `--exact verifier/...` invocation outside `cargo ktstr verifier` \
         is not supported — the dispatcher owns kernel-set resolution. Run \
         `cargo ktstr verifier [--kernel SPEC]` instead.",
    )
}

/// Format the "kernel label not in KTSTR_KERNEL_LIST" diagnostic.
/// `present` is the slice of sanitized labels actually present in
/// the list, in their KTSTR_KERNEL_LIST ordering. Extracted for the
/// same reason as [`format_empty_kernel_list_error`].
fn format_unknown_kernel_label_error(
    full_name: &str,
    kernel_label: &str,
    sched_name: &str,
    present: &[&str],
) -> String {
    format!(
        "ktstr verifier: cell {full_name}: kernel label {kernel_label:?} \
         not in KTSTR_KERNEL_LIST. Present labels: [{}]. \
         Either add --kernel <SPEC> to the dispatcher invocation so it \
         resolves into this label, or remove the matching entry from \
         declare_scheduler!(... kernels = [...]) for {sched_name}.",
        present.join(", "),
    )
}

/// Emit `verifier/<sched>/<kernel>/<preset>: test` lines — one per
/// (declared scheduler × kernel-list entry × accepted gauntlet
/// preset) cell. Mirrors the gauntlet emission pattern in
/// [`list_tests_all`] but walks [`super::KTSTR_SCHEDULERS`] instead
/// of [`KTSTR_TESTS`]. Cells are paired with the
/// [`run_verifier_cell`] handler registered in
/// [`ktstr_test_early_dispatch`]'s `--exact verifier/...` branch.
///
/// The matrix dimension is `KTSTR_KERNEL_LIST` (always populated by
/// the `cargo ktstr verifier` dispatcher — even with a single
/// auto-discovered kernel, the dispatcher synthesizes a one-entry
/// list with a derived label). Each scheduler's
/// `declare_scheduler!` `kernels = [...]` declaration acts as a
/// per-scheduler filter on the matrix — `Version` / `Range`
/// declarations match entries by raw-label equality / range
/// membership; `Path` / `CacheKey` / `Git` declarations match by
/// sanitized-label equality. An empty `kernels = []` declaration
/// accepts every entry in the list (no filter).
///
/// Acceptance filter mirrors the gauntlet branching in
/// [`for_each_gauntlet_variant`]: perf-mode pinning constrains
/// preset eligibility against the host's LLC width AND per-LLC CPU
/// width, while no-perf-mode (KVM-emulated topology) only needs the
/// total-CPU budget to fit. The mode is global for the verifier path
/// — there is no per-cell `performance_mode` attribute analogous to
/// `KtstrTestEntry::no_perf_mode` because every cell shares the same
/// `cargo ktstr verifier` invocation.
///
/// Schedulers declared with [`super::SchedulerSpec::Eevdf`] or
/// [`super::SchedulerSpec::KernelBuiltin`] are skipped at emission
/// time because neither has a userspace binary to load BPF programs
/// from — emitting cells that would always SKIP at execution wastes
/// nextest's per-cell process budget and clutters the run output.
///
/// Cell names with `/` in `sched.name` or `preset.name` would
/// corrupt the splitn-based parse in [`run_verifier_cell`]. The
/// emission elides such cells with a stderr warning so the operator
/// sees the gap rather than silently dropping cells.
///
/// When `KTSTR_KERNEL_LIST` is absent (direct binary invocation
/// outside the `cargo ktstr verifier` dispatcher), no cells emit.
/// Operators who invoke a test binary directly with `--exact
/// verifier/...` will see the cell handler's "kernel label not in
/// KTSTR_KERNEL_LIST" error.
fn list_verifier_cells_all() {
    use super::SchedulerSpec;
    let kernel_list = read_kernel_list();
    if kernel_list.is_empty() {
        return;
    }
    let presets = crate::gauntlet::gauntlet_presets();
    let (host_cpus, host_llcs, host_max_cpus_per_llc) = super::host_capacity();
    let no_perf_mode = super::runtime::no_perf_mode_active();

    for sched in super::KTSTR_SCHEDULERS.iter() {
        if matches!(
            sched.binary,
            SchedulerSpec::Eevdf | SchedulerSpec::KernelBuiltin { .. }
        ) {
            continue;
        }
        if sched.name.contains('/') {
            eprintln!(
                "ktstr verifier: scheduler name {:?} contains '/' — skipping cell emission (would corrupt verifier/<sched>/<kernel>/<preset> parse)",
                sched.name,
            );
            continue;
        }
        for kernel_entry in &kernel_list {
            if !sched_kernel_filter_accepts(sched.kernels, kernel_entry) {
                continue;
            }
            for preset in presets.iter() {
                if preset.name.contains('/') {
                    eprintln!(
                        "ktstr verifier: preset name {:?} contains '/' — skipping cell (would corrupt parse)",
                        preset.name,
                    );
                    continue;
                }
                let accepted = if no_perf_mode {
                    sched
                        .constraints
                        .accepts_no_perf_mode(&preset.topology, host_cpus)
                } else {
                    sched.constraints.accepts(
                        &preset.topology,
                        host_cpus,
                        host_llcs,
                        host_max_cpus_per_llc,
                    )
                };
                if !accepted {
                    continue;
                }
                println!(
                    "verifier/{}/{}/{}: test",
                    sched.name, kernel_entry.sanitized, preset.name,
                );
            }
        }
    }
}

/// Parse `verifier/<sched_name>/<kernel_label>/<preset_name>`, look
/// up the declared scheduler in [`super::KTSTR_SCHEDULERS`] + the
/// gauntlet preset in [`crate::gauntlet::gauntlet_presets`] + the kernel
/// in [`KTSTR_KERNEL_LIST_ENV`](crate::KTSTR_KERNEL_LIST_ENV),
/// resolve the scheduler binary path per
/// [`super::SchedulerSpec`], boot the verifier VM via
/// [`crate::verifier::collect_verifier_output`], and print the
/// rendered output. Returns 0 on success, 1 on failure /
/// malformed cell name.
///
/// The per-cell kernel directory is resolved by sanitized-label
/// lookup in `KTSTR_KERNEL_LIST` — the
/// `cargo ktstr verifier` dispatcher always populates the list,
/// even with no `--kernel` flag (it synthesizes a single auto-
/// discovered entry). There is no single-kernel-mode fallback.
/// An unrecognised label or an absent list both surface as an
/// exit-1 diagnostic naming the present labels and pointing at
/// the dispatcher.
///
/// Eevdf + KernelBuiltin scheduler variants are filtered out at
/// emission time in [`list_verifier_cells_all`], so nextest
/// dispatch never reaches the SKIP arms in this function. The
/// SKIP arms remain as defense-in-depth for direct
/// `--exact verifier/<eevdf>/...` invocation outside nextest
/// (the only path that bypasses the emission-time filter); in
/// that case they emit a `SKIP` banner + exit 0.
fn run_verifier_cell(full_name: &str) -> i32 {
    use super::SchedulerSpec;

    let rest = match full_name.strip_prefix("verifier/") {
        Some(r) => r,
        None => {
            eprintln!("ktstr verifier: missing 'verifier/' prefix in {full_name:?}");
            return 1;
        }
    };
    let parts: Vec<&str> = rest.splitn(3, '/').collect();
    if parts.len() != 3 {
        eprintln!(
            "ktstr verifier: malformed cell name {full_name:?}; expected verifier/<sched>/<kernel>/<preset>",
        );
        return 1;
    }
    let (sched_name, kernel_label, preset_name) = (parts[0], parts[1], parts[2]);

    // Emit the cell banner BEFORE every SKIP / FAIL branch so the
    // operator always sees which (scheduler, kernel, preset) tuple
    // produced the result. Without it an early-exit SKIP / FAIL would
    // surface as a bare error line nextest tags with the full cell
    // name but no per-axis context.
    println!("\n=== {sched_name} | kernel {kernel_label} | topology {preset_name} ===");

    // Fail-fast on missing KVM with the canonical actionable error
    // (kvm group / kvm-ok hint). Without this preflight the operator
    // gets a deep error inside VM bring-up.
    if let Err(e) = crate::cli::check_kvm() {
        eprintln!("ktstr verifier: cell {full_name}: {e:#}");
        return 1;
    }

    let Some(sched) = super::KTSTR_SCHEDULERS
        .iter()
        .find(|s| s.name == sched_name)
    else {
        eprintln!("ktstr verifier: no declared scheduler {sched_name:?} (cell {full_name:?})",);
        return 1;
    };

    let preset_list = crate::gauntlet::gauntlet_presets();
    let Some(preset) = preset_list.iter().find(|p| p.name == preset_name) else {
        eprintln!("ktstr verifier: no gauntlet preset {preset_name:?} (cell {full_name:?})",);
        return 1;
    };

    // Resolve the per-cell kernel directory by looking the cell's
    // sanitized label up in `KTSTR_KERNEL_LIST`. The
    // `cargo ktstr verifier` dispatcher always populates the list —
    // even with no `--kernel` flag it synthesizes a single auto-
    // discovered entry — so the lookup is the single source of
    // truth and there is no single-kernel-mode fallback that would
    // silently run a cell against an unrelated kernel.
    //
    // An empty list reaching this function means the test binary was
    // invoked outside the dispatcher (direct `--exact verifier/...`
    // under a hand-spawned nextest, for instance). Error with an
    // actionable message rather than fall through to auto-discovery.
    let kernel_list = read_kernel_list();
    let Some(kernel_entry) = kernel_list
        .iter()
        .find(|k| k.sanitized.as_str() == kernel_label)
    else {
        if kernel_list.is_empty() {
            eprintln!("{}", format_empty_kernel_list_error(full_name));
        } else {
            let present: Vec<&str> = kernel_list.iter().map(|k| k.sanitized.as_str()).collect();
            eprintln!(
                "{}",
                format_unknown_kernel_label_error(full_name, kernel_label, sched_name, &present,),
            );
        }
        return 1;
    };

    let sched_bin: std::path::PathBuf = match sched.binary {
        SchedulerSpec::Discover(pkg) => match crate::build_and_find_binary(pkg) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("ktstr verifier: build scheduler {pkg:?}: {e:#}");
                return 1;
            }
        },
        SchedulerSpec::Path(p) => {
            let path = std::path::PathBuf::from(p);
            if !path.exists() {
                eprintln!("ktstr verifier: scheduler binary not found: {p}");
                return 1;
            }
            path
        }
        // Eevdf + KernelBuiltin are filtered at list time in
        // list_verifier_cells_all, so nextest dispatch never reaches
        // these arms. The SKIP arms remain as defense-in-depth for
        // direct `--exact verifier/<eevdf>/...` invocation outside
        // nextest.
        SchedulerSpec::Eevdf => {
            println!(
                "ktstr verifier: SKIP cell {full_name} (Eevdf has no userspace binary to verify)",
            );
            return 0;
        }
        SchedulerSpec::KernelBuiltin { .. } => {
            println!(
                "ktstr verifier: SKIP cell {full_name} (KernelBuiltin has no userspace binary to verify)",
            );
            return 0;
        }
    };

    let ktstr_bin = match std::env::current_exe() {
        Ok(p) => p,
        Err(e) => {
            eprintln!(
                "ktstr verifier: locate ktstr binary via current_exe() (required so the \
                 verifier VM can boot the same test binary as /init for guest-side dispatch): {e}",
            );
            return 1;
        }
    };

    let kernel_path = kernel_entry.kernel_dir.clone();
    let topology = super::TopologyJson::from(preset.topology);
    let sched_args: Vec<String> = sched.sched_args.iter().map(|s| s.to_string()).collect();

    // Raw mode is opt-in via the dispatcher's --raw flag, plumbed
    // through KTSTR_VERIFIER_RAW_ENV. Presence (any value, including
    // empty) enables raw rendering — matches the "set to any value"
    // semantics documented on the const and the dispatcher's
    // `cmd.env(KTSTR_VERIFIER_RAW_ENV, "1")` setter.
    let raw = std::env::var_os(crate::KTSTR_VERIFIER_RAW_ENV).is_some();

    match crate::verifier::collect_verifier_output(
        &sched_bin,
        &ktstr_bin,
        &kernel_path,
        &sched_args,
        topology,
    ) {
        Ok(result) => {
            let output = crate::verifier::format_verifier_output("verifier", &result, raw);
            print!("{output}");
            0
        }
        Err(e) => {
            eprintln!("ktstr verifier: cell {full_name} FAILED: {e:#}");
            1
        }
    }
}

/// List tests with budget-based coverage maximization.
///
/// Collects all eligible tests as candidates, runs greedy selection,
/// and prints only the selected subset. Multi-kernel mode adds the
/// kernel suffix as a feature dimension so the budget selector
/// picks per-kernel coverage; single-kernel mode is unchanged.
///
/// `KTSTR_CARGO_TEST_MODE=1` is treated identically to
/// `list_tests_all`: the budget pipeline runs only over base test
/// candidates (no gauntlet-variant candidates, no multi-kernel
/// fan-out). The greedy selector still applies — a low budget
/// can still trim the base list — but the candidate set is the
/// same set that the dispatch path would actually run.
fn list_tests_budget(ignored_only: bool, budget_secs: f64) {
    use crate::budget::{TestCandidate, estimate_duration, extract_features, select};

    let cargo_test_mode = crate::cargo_test_mode::cargo_test_mode_active();
    let presets = crate::gauntlet::gauntlet_presets();
    let has_vmlinux = resolve_test_kernel()
        .ok()
        .and_then(|k| crate::vmm::find_vmlinux(&k))
        .is_some();
    let (host_cpus, host_llcs, host_max_cpus_per_llc) = super::host_capacity();
    let mut candidates: Vec<TestCandidate> = Vec::new();

    let kernel_list = read_kernel_list();
    let multi_kernel = kernel_list.len() > 1 && !cargo_test_mode;
    let kernel_suffixes: Vec<&str> = if multi_kernel {
        kernel_list.iter().map(|k| k.sanitized.as_str()).collect()
    } else {
        vec![""]
    };

    for entry in KTSTR_TESTS.iter() {
        if !entry.bpf_map_write.is_empty() && !has_vmlinux {
            continue;
        }

        let base_ignored = is_ignored(entry);
        let base_topo = entry.topology;

        // Base test
        if !ignored_only || base_ignored {
            // host_only tests never boot a VM, so the kernel never
            // affects what runs — push one candidate without a
            // kernel suffix even in multi-kernel mode. Otherwise the
            // budget selector would consider N identical copies of
            // the same host-side function.
            if entry.host_only {
                candidates.push(TestCandidate {
                    name: format!("ktstr/{}: test", entry.name),
                    features: extract_features(entry, &base_topo, false, entry.name),
                    estimated_secs: estimate_duration(entry, &base_topo),
                });
            } else {
                for suffix in &kernel_suffixes {
                    let name = if suffix.is_empty() {
                        format!("ktstr/{}: test", entry.name)
                    } else {
                        format!("ktstr/{}/{suffix}: test", entry.name)
                    };
                    candidates.push(TestCandidate {
                        name,
                        features: extract_features(entry, &base_topo, false, entry.name),
                        estimated_secs: estimate_duration(entry, &base_topo),
                    });
                }
            }
        }

        if entry.host_only {
            continue;
        }

        if cargo_test_mode {
            // No gauntlet candidates in cargo-test mode — the
            // dispatch path will never execute them and including
            // them in the budget candidate set would shift greedy
            // selection toward variants that resolve to "no test"
            // at run time.
            continue;
        }

        for_each_gauntlet_variant(
            entry,
            &presets,
            host_cpus,
            host_llcs,
            host_max_cpus_per_llc,
            |preset| {
                for suffix in &kernel_suffixes {
                    let test_name = if suffix.is_empty() {
                        format!("gauntlet/{}/{}", entry.name, preset.name)
                    } else {
                        format!("gauntlet/{}/{}/{suffix}", entry.name, preset.name)
                    };
                    candidates.push(TestCandidate {
                        name: format!("{test_name}: test"),
                        features: extract_features(entry, &preset.topology, true, &test_name),
                        estimated_secs: estimate_duration(entry, &preset.topology),
                    });
                }
            },
        );
    }

    let selected = select(&candidates, budget_secs);
    for &i in &selected {
        println!("{}", candidates[i].name);
    }

    let stats = crate::budget::selection_stats(&candidates, &selected, budget_secs);
    eprintln!(
        "ktstr budget: {}/{} tests, {:.0}/{:.0}s used, {}/{} configurations covered",
        stats.selected,
        stats.total,
        stats.budget_used,
        stats.budget_total,
        stats.bits_covered,
        stats.bits_possible,
    );
}

/// Strip an optional `/{sanitized_kernel_label}` suffix from `name`,
/// look up the matching [`KernelEntry`] in the multi-kernel list,
/// and re-export `KTSTR_KERNEL` to that entry's directory. Returns
/// the prefix-only name for the dispatch caller.
///
/// When `KTSTR_KERNEL_LIST` is unset / single-entry, the function
/// is a no-op pass-through: returns `(name, None)` and does not
/// touch the env. When the list has 2+ entries, the suffix is
/// REQUIRED and missing it surfaces as `Err` (the early-dispatch
/// caller turns that into exit code 1 with an actionable message)
/// — the suffix is part of every test name `--list` emitted, so a
/// `--exact` invocation that omits it can only come from operator
/// hand-construction or tooling that hasn't been taught the
/// multi-kernel naming.
fn strip_kernel_suffix<'a>(
    name: &'a str,
    kernel_list: &'a [KernelEntry],
) -> Result<(&'a str, Option<&'a KernelEntry>), String> {
    if kernel_list.len() <= 1 {
        return Ok((name, None));
    }
    // Multi-kernel: every test name carries `/kernel_…` as its
    // final segment. Iterate the labels rather than splitting on
    // `/` — the suffix always has exactly one extra `/` separator
    // before `kernel_…`, but the body of the test name CAN contain
    // `/` (gauntlet variants already do — `gauntlet/{name}/{preset}`),
    // so a naive `rsplit_once('/')` would accidentally peel the
    // preset segment instead.
    //
    // Distinct kernels in the same `KTSTR_KERNEL_LIST` produce
    // distinct sanitized labels in practice — the producer emits
    // semantic identifiers (version strings, git owner/repo/ref,
    // path basename + 6-char hash) that don't share suffixes
    // among the resolved set. If a future regression DID produce
    // labels where one is a strict suffix of another (e.g.
    // `kernel_6_14` vs `kernel_x_kernel_6_14`), the iterate-and-
    // first-match below would pick whichever appears first in
    // the kernel_list — deterministic but potentially wrong.
    // Producer-side regression detection would catch that
    // class of collision before it reaches this peeler.
    for entry in kernel_list {
        let needle = format!("/{}", entry.sanitized);
        if let Some(stripped) = name.strip_suffix(&needle) {
            return Ok((stripped, Some(entry)));
        }
    }
    Err(format!(
        "test name {name:?} has no recognised kernel suffix (KTSTR_KERNEL_LIST \
         carries {n} kernels — every test name must end with `/kernel_…`)",
        n = kernel_list.len(),
    ))
}

/// Re-export `KTSTR_KERNEL` to the kernel directory carried by a
/// resolved [`KernelEntry`]. Called when a multi-kernel `--exact`
/// dispatch peels off the per-test kernel suffix.
///
/// SAFETY: nextest invokes the test binary's `--exact` handler in a
/// single-threaded context — there are no other readers of the env
/// at this point. The eventual VM-launch site reads `KTSTR_KERNEL`
/// via `find_kernel` after this returns; that read is sequenced
/// after the write per the program order.
fn export_kernel_for_variant(entry: &KernelEntry) {
    // SAFETY: see fn-level doc — single-threaded ctor / nextest
    // dispatch context.
    unsafe { std::env::set_var(crate::KTSTR_KERNEL_ENV, &entry.kernel_dir) };
}

/// Parse a nextest-style test name and run it.
///
/// Handles base tests (`ktstr/{name}`), gauntlet variants
/// (`gauntlet/{name}/{preset}`), and bare names (backward compat).
/// When `KTSTR_KERNEL_LIST` carries 2+ kernels,
/// VM-bound test names additionally end with
/// `/{sanitized_kernel_label}` — that suffix is peeled here and
/// the matching kernel directory is re-exported via
/// [`crate::KTSTR_KERNEL_ENV`] before the dispatch continues. `host_only`
/// tests are short-circuited BEFORE the suffix peel: they never
/// boot a VM, so the kernel-suffix listing path emits one
/// `ktstr/{name}: test` entry without a kernel suffix regardless
/// of the kernel-list cardinality (see `list_tests_all` /
/// `list_tests_budget`), and routing them through
/// `strip_kernel_suffix` would surface as a "no recognised kernel
/// suffix" exit-1 error. Returns an exit code.
pub(crate) fn run_named_test(test_name: &str) -> i32 {
    let kernel_list = read_kernel_list();

    // host_only short-circuit: in multi-kernel mode, host_only tests
    // are listed without a `/{sanitized_kernel_label}` suffix (see
    // `list_tests_all` / `list_tests_budget`, which emit a single
    // `ktstr/{name}: test` line for host_only entries regardless of
    // the kernel-list cardinality — a host_only test never boots a
    // VM, so the kernel never affects what runs). Calling
    // `strip_kernel_suffix` on such a name in multi-kernel mode
    // would fail with the "no recognised kernel suffix" error and
    // misroute every host_only dispatch to exit 1.
    //
    // Resolve the host_only check from `find_test` BEFORE the
    // suffix peel so the multi-kernel branch only applies to
    // VM-bound tests. Single-kernel mode is unaffected — the
    // pass-through arm in `strip_kernel_suffix` returns the input
    // verbatim either way.
    let bare_for_lookup = test_name.strip_prefix("ktstr/").unwrap_or(test_name);

    if let Some(entry) = find_test(bare_for_lookup)
        && entry.host_only
    {
        return run_host_only_test(entry);
    }

    let (test_name, kernel_entry) = match strip_kernel_suffix(test_name, &kernel_list) {
        Ok(pair) => pair,
        Err(e) => {
            eprintln!("{e}");
            return 1;
        }
    };
    if let Some(entry) = kernel_entry {
        export_kernel_for_variant(entry);
    }

    if let Some(rest) = test_name.strip_prefix("gauntlet/") {
        return run_gauntlet_test(rest);
    }

    let bare_name = test_name.strip_prefix("ktstr/").unwrap_or(test_name);
    let entry = match find_test(bare_name) {
        Some(e) => e,
        None => {
            eprintln!("unknown test: {test_name}");
            return 1;
        }
    };

    // Defense-in-depth: host_only re-check after suffix peel for the
    // edge case where the bare_for_lookup pre-strip lookup missed
    // (e.g. a future test name shape that doesn't match the
    // pre-strip form but does after the suffix peel).
    if entry.host_only {
        return run_host_only_test(entry);
    }

    if entry.performance_mode && super::runtime::no_perf_mode_active() {
        crate::report::test_skip(format_args!(
            "{}: test requires performance_mode but --no-perf-mode or KTSTR_NO_PERF_MODE is active",
            bare_name,
        ));
        // See run_ktstr_test_inner for the sidecar-emission rationale.
        // Plain (non-gauntlet) dispatch: no TopoOverride, so the skip
        // records entry.topology (declared == booted for a plain test).
        record_skip_sidecar(entry, None);
        return 0;
    }

    if super::runtime::perf_only_skips_entry(entry) {
        crate::report::test_skip(format_args!(
            "{bare_name}: KTSTR_PERF_ONLY is active and this test is not a performance_mode test",
        ));
        // Skip sidecar so the perf-delta pool records the skip (excluded
        // from the A/B compare) rather than a phantom missing result.
        record_skip_sidecar(entry, None);
        return 0;
    }

    if !entry.bpf_map_write.is_empty()
        && let Ok(kernel) = resolve_test_kernel()
        && crate::vmm::find_vmlinux(&kernel).is_none()
    {
        eprintln!("FAIL: vmlinux not found, bpf_map_write requires vmlinux");
        return 1;
    }

    let result = run_ktstr_test_inner(entry, None);
    result_to_exit_code(result, entry.expect_err, entry.allow_inconclusive)
}

/// Run a host-only test directly without booting a VM.
/// Returns an exit code for nextest dispatch.
fn run_host_only_test(entry: &KtstrTestEntry) -> i32 {
    let result = run_host_only_test_inner(entry);
    result_to_exit_code(result, entry.expect_err, entry.allow_inconclusive)
}

/// Inner host-only dispatch returning `Result<AssertResult>`.
///
/// Builds a minimal Ctx and calls the test function on the host.
/// Used for tests that need host tools (cargo, nested VMs).
///
/// Topology comes from real-host sysfs (`/sys/devices/system/cpu/`)
/// via [`crate::topology::TestTopology::from_system`]; the test's
/// declared VM topology is intentionally ignored for host_only
/// runs because the test author wrote it for a synthetic VM and
/// the host's actual CPU layout is what `WorkSpec::workers_pct` /
/// `AffinityIntent::LlcAligned` resolve against. Bails with an
/// actionable diagnostic when sysfs CPU enumeration fails — the
/// underlying causes are missing `/sys/devices/system/cpu/online`
/// (no /sys mount or container masking), unreadable contents (rare
/// permissions edge), corrupt sysfs string (kernel/hardware bug),
/// or an empty online-CPU set (degenerate cpuset namespace).
///
/// Cgroup parent defaults to `/sys/fs/cgroup/ktstr`; the operator
/// can override via `KTSTR_HOST_CGROUP_PARENT`. The override path
/// is validated upfront: it must be non-empty and rooted under
/// `/sys/fs/cgroup` so an accidental empty/relative/foreign value
/// produces a clear error instead of an opaque cgroupfs failure
/// later. Empty-string env value is treated as "unset" and falls
/// back to the default.
///
/// For cgroup-v2 user delegation (Mode B/C: systemd `Delegate=yes`,
/// container `nsdelegate`), the operator sets
/// `KTSTR_CGROUP_WALK_ROOT` to the delegation boundary so
/// [`crate::cgroup::CgroupManager::setup`]'s ancestor
/// `subtree_control` walk stops there instead of EACCES-ing at
/// `user.slice` / the container root. Defaults to `/sys/fs/cgroup`
/// (Mode A: root-owned tree).
fn run_host_only_test_inner(entry: &KtstrTestEntry) -> Result<AssertResult> {
    let topo = crate::topology::TestTopology::from_system().context(
        "host_only requires real-host topology from sysfs; \
         the sysfs CPU enumeration at /sys/devices/system/cpu/online \
         failed — likely causes: running outside a /sys-mounted \
         environment, sysfs contents unreadable (permissions / \
         container mask), corrupt online-CPU string, or a degenerate \
         cpuset namespace with no online CPUs",
    )?;
    let cgroup_parent = resolve_host_cgroup_parent()?;
    let cgroups = build_host_cgroup_manager(&cgroup_parent)?;
    let merged_assert = crate::assert::Assert::default_checks()
        .merge(&entry.scheduler.assert)
        .merge(&entry.assert);
    let ctx = crate::scenario::Ctx::builder(&cgroups, &topo)
        .duration(entry.duration)
        .settle(std::time::Duration::ZERO)
        .assert(merged_assert)
        .entry_name(entry.name)
        // host_only is host-side with no VM: the resolved topology is
        // the declared entry.topology (resolve_vm_topology(entry, None)),
        // so compute the variant hash directly rather than threading.
        .variant_hash(super::sidecar::variant_hash_from_parts(
            entry,
            &entry.topology,
            &super::args::current_work_type(),
        ))
        .build();
    (entry.func)(&ctx)
}

/// Default cgroup parent path for `host_only` tests when
/// `KTSTR_HOST_CGROUP_PARENT` is unset. Suitable for both root
/// (writable directly) and non-root (operator pre-creates
/// `/sys/fs/cgroup/ktstr` with appropriate ownership, OR overrides via
/// `KTSTR_HOST_CGROUP_PARENT` to point at a path inside a delegated
/// subtree) invocations. See [`resolve_host_cgroup_parent`] for the
/// env-override path and `build_host_cgroup_manager` for the
/// cgroup-v2 Mode B/C delegation wire-up.
///
/// `pub` so tests can pin against it instead of mirroring
/// the literal in their own assertion strings (the
/// `resolve_host_cgroup_parent_*` unit tests in `dispatch_tests.rs`
/// assert unset/empty env falls back to this const). Treat as the
/// canonical default — operators set `KTSTR_HOST_CGROUP_PARENT` to
/// override.
pub const DEFAULT_HOST_CGROUP_PARENT: &str = "/sys/fs/cgroup/ktstr";

/// Resolve the cgroup parent path for `host_only` tests.
///
/// Reads `KTSTR_HOST_CGROUP_PARENT`. Empty / unset falls back to
/// `DEFAULT_HOST_CGROUP_PARENT`. A set value must be rooted under
/// `/sys/fs/cgroup` (no relative paths, no random /tmp dirs) so an
/// accidental misconfiguration surfaces here rather than as an
/// opaque cgroupfs failure inside `CgroupManager::setup`.
///
/// Non-root callers are admitted: cgroup-v2 user delegation (Mode
/// B/C: systemd `Delegate=yes`, container `nsdelegate`) is handled
/// by `build_host_cgroup_manager` threading
/// [`crate::KTSTR_CGROUP_WALK_ROOT_ENV`] into
/// [`crate::cgroup::CgroupManager::with_walk_root`] so the
/// `subtree_control` walk bails at the delegation root instead of
/// EACCES-ing on `user.slice`.
pub fn resolve_host_cgroup_parent() -> Result<String> {
    let parent = match std::env::var(crate::KTSTR_HOST_CGROUP_PARENT_ENV) {
        Ok(s) if !s.is_empty() => s,
        _ => return Ok(DEFAULT_HOST_CGROUP_PARENT.to_string()),
    };
    if !parent.starts_with("/sys/fs/cgroup") || parent == "/sys/fs/cgroup" {
        anyhow::bail!(
            "KTSTR_HOST_CGROUP_PARENT={parent:?}: must be rooted under \
             /sys/fs/cgroup and name a non-root subdirectory \
             (e.g. /sys/fs/cgroup/ktstr or /sys/fs/cgroup/ktstr-foo); \
             unset or empty falls back to {DEFAULT_HOST_CGROUP_PARENT}",
        );
    }
    Ok(parent)
}

/// Build a [`crate::cgroup::CgroupManager`] for a `host_only` test
/// run, threading [`crate::KTSTR_CGROUP_WALK_ROOT_ENV`] into
/// [`crate::cgroup::CgroupManager::with_walk_root`] when set.
///
/// The walk root override bounds [`crate::cgroup::CgroupManager::setup`]'s
/// ancestor `subtree_control` walk for cgroup-v2 Mode B/C
/// delegation: under systemd `Delegate=yes` or a container's
/// `nsdelegate`, the operator owns subtree_control writes only
/// inside the delegated subtree. Without the override the walk
/// starts at `/sys/fs/cgroup` and EACCES-es at `user.slice` or the
/// container root.
///
/// Empty / unset falls through to the default `/sys/fs/cgroup`
/// (Mode A: root-owned tree). [`crate::cgroup::CgroupManager::with_walk_root`]
/// validates that the chosen walk root is a prefix of `parent` —
/// misconfigurations surface as a focused error before the first
/// cgroupfs write rather than as an opaque downstream EACCES.
///
/// Non-root callers with no walk-root override are admitted here — the
/// precondition (root, or a cgroup-v2 delegated walk root) is enforced
/// lazily at [`crate::cgroup::CgroupManager::setup`], the first real
/// cgroup operation. `host_only` tests that never create a cgroup
/// (macro-attribute fixtures, host-topology reads, nested-VM verifier
/// orchestration) therefore run without root; only a test that actually
/// touches a cgroup hits the deferred non-root error.
fn build_host_cgroup_manager(cgroup_parent: &str) -> Result<crate::cgroup::CgroupManager> {
    let cg = crate::cgroup::CgroupManager::new(cgroup_parent);
    match std::env::var(crate::KTSTR_CGROUP_WALK_ROOT_ENV) {
        Ok(walk_root) if !walk_root.is_empty() => {
            // Defense-in-depth: walk_root must be rooted under
            // /sys/fs/cgroup. Mirrors the sibling
            // KTSTR_HOST_CGROUP_PARENT_ENV guard above so an operator
            // typo surfaces here instead of as a downstream cgroupfs
            // fs::write EACCES.
            if !walk_root.starts_with("/sys/fs/cgroup") {
                anyhow::bail!(
                    "{env}={walk_root:?}: walk root must be rooted under /sys/fs/cgroup \
                     (e.g. /sys/fs/cgroup/user.slice/user-$(id -u).slice/user@$(id -u).service \
                     for a systemd user session); the value supplied is outside the cgroup-v2 \
                     mount and would EACCES on the first cgroupfs write",
                    env = crate::KTSTR_CGROUP_WALK_ROOT_ENV,
                );
            }
            cg.with_walk_root(&walk_root).with_context(|| {
                format!(
                    "{env}={walk_root:?}: walk-root override rejected (must be a prefix of \
                     KTSTR_HOST_CGROUP_PARENT={cgroup_parent:?})",
                    env = crate::KTSTR_CGROUP_WALK_ROOT_ENV,
                )
            })
        }
        // No KTSTR_CGROUP_WALK_ROOT override. Return the manager as-is;
        // the non-root precondition for managing cgroups under the
        // kernel-owned default walk root is checked lazily in
        // CgroupManager::setup (first real cgroup use). host_only tests
        // that never create a cgroup — macro-attribute fixtures,
        // host-topology reads, nested-VM verifier orchestration — must
        // not be failed here for a resource they never touch. A
        // non-root test that does create a cgroup gets the deferred
        // setup error pointing at with_walk_root; the operator on-ramp
        // is EITHER to run as root OR to set KTSTR_CGROUP_WALK_ROOT to a
        // delegated cgroup-v2 subtree (handled by the arm above).
        _ => Ok(cg),
    }
}

/// Run a gauntlet variant test. `rest` is `{name}/{preset}`.
pub(crate) fn run_gauntlet_test(rest: &str) -> i32 {
    let parts: Vec<&str> = rest.splitn(2, '/').collect();
    if parts.len() != 2 {
        eprintln!("invalid gauntlet test name: gauntlet/{rest}");
        return 1;
    }
    let (test_name, preset_name) = (parts[0], parts[1]);

    let entry = match find_test(test_name) {
        Some(e) => e,
        None => {
            eprintln!("unknown test: {test_name}");
            return 1;
        }
    };

    let presets = crate::gauntlet::gauntlet_presets();
    let preset = match presets.iter().find(|p| p.name == preset_name) {
        Some(p) => p,
        None => {
            eprintln!("unknown gauntlet preset: {preset_name}");
            return 1;
        }
    };

    let t = &preset.topology;
    let cpus = t.total_cpus();

    let memory_mib = super::runtime::derive_test_memory_mib(cpus, entry);
    let topo = TopoOverride {
        numa_nodes: t.numa_nodes,
        llcs: t.llcs,
        cores: t.cores_per_llc,
        threads: t.threads_per_core,
        memory_mib,
    };

    if entry.performance_mode && super::runtime::no_perf_mode_active() {
        crate::report::test_skip(format_args!(
            "{}: test requires performance_mode but --no-perf-mode or KTSTR_NO_PERF_MODE is active",
            test_name,
        ));
        // Gauntlet preset: record the preset's RESOLVED topology
        // (Topology::from(&topo)) so this skip shares a variant_hash
        // with a run of the same preset and distinguishes other presets.
        record_skip_sidecar(entry, Some(&topo));
        return 0;
    }

    if super::runtime::perf_only_skips_entry(entry) {
        crate::report::test_skip(format_args!(
            "{test_name}: KTSTR_PERF_ONLY is active and this test is not a performance_mode test",
        ));
        // Gauntlet preset: record the preset's RESOLVED topology so the
        // skip shares a variant_hash with a run of the same preset.
        record_skip_sidecar(entry, Some(&topo));
        return 0;
    }

    if !entry.bpf_map_write.is_empty()
        && let Ok(kernel) = resolve_test_kernel()
        && crate::vmm::find_vmlinux(&kernel).is_none()
    {
        eprintln!("FAIL: vmlinux not found, bpf_map_write requires vmlinux");
        return 1;
    }

    let result = run_ktstr_test_inner(entry, Some(&topo));
    result_to_exit_code(result, entry.expect_err, entry.allow_inconclusive)
}

/// Collect sidecar JSON files and return the full gauntlet analysis.
///
/// When `dir` is `Some`, reads sidecars from that directory. Otherwise
/// uses the default sidecar directory (`KTSTR_SIDECAR_DIR` override, or
/// `{CARGO_TARGET_DIR or "target"}/ktstr/{kernel}-{project_commit}/`,
/// where `{project_commit}` is the project HEAD short hex with
/// `-dirty` when the worktree differs).
///
/// Returns the concatenated output of `analyze_rows`, verifier stats,
/// callback profile, and KVM stats. Returns an empty string when no
/// sidecars are found.
pub fn analyze_sidecars(dir: Option<&std::path::Path>) -> String {
    let default_dir;
    let dir = match dir {
        Some(d) => d,
        None => {
            default_dir = sidecar_dir();
            &default_dir
        }
    };
    let sidecars = collect_sidecars(dir);
    if sidecars.is_empty() {
        return String::new();
    }
    let mut out = String::new();
    let rows: Vec<_> = sidecars.iter().map(crate::stats::sidecar_to_row).collect();
    if !rows.is_empty() {
        out.push_str(&crate::stats::analyze_rows(&rows));
    }
    let vstats = format_verifier_stats(&sidecars);
    if !vstats.is_empty() {
        out.push_str(&vstats);
    }
    let cprofile = format_callback_profile(&sidecars);
    if !cprofile.is_empty() {
        out.push_str(&cprofile);
    }
    let kstats = format_kvm_stats(&sidecars);
    if !kstats.is_empty() {
        out.push_str(&kstats);
    }
    out
}

/// Discover plain `#[test]` items by re-invoking the binary without
/// NEXTEST, reading libtest's `--list` output, and printing only
/// names that don't match any KTSTR_TESTS entry. This lets plain
/// tests coexist with `#[ktstr_test]` in the same binary without
/// duplicating the ktstr entries.
///
/// `ignored_only` forwards `--ignored` onto the child `--list` call
/// so the echoed plain-test set matches the bucket nextest is
/// enumerating (all tests vs the `#[ignore]`-only subset). Omitting
/// the flag here lands every plain test in nextest's ignored set and
/// silently skips them by default — see the body comment.
fn list_plain_tests(ignored_only: bool) {
    use std::collections::HashSet;
    let ktstr_names: HashSet<&str> = KTSTR_TESTS.iter().map(|e| e.name).collect();

    let exe = match std::env::current_exe() {
        Ok(p) => p,
        Err(_) => return,
    };
    let mut cmd = std::process::Command::new(exe);
    cmd.env_remove("NEXTEST");
    // Forward `--ignored` so the plain-test set echoed here matches the
    // bucket nextest is asking for. nextest computes its "ignored" set by
    // re-running the binary with `--list --ignored`; if this child always
    // lists ALL plain `#[test]` (no `--ignored`), every plain test lands
    // in nextest's ignored set and is silently skipped by default
    // (footgun #2). With the flag forwarded, only real `#[ignore]` plain
    // tests are reported under `--ignored`, so non-ignored plain tests run
    // by default like any other test.
    let mut list_args: Vec<&str> = vec!["--list", "--format", "terse"];
    if ignored_only {
        list_args.push("--ignored");
    }
    cmd.args(&list_args);
    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::null());
    let output = match cmd.output() {
        Ok(o) => o,
        Err(_) => return,
    };
    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        let name = line.strip_suffix(": test").unwrap_or(line);
        if !ktstr_names.contains(name) && !name.is_empty() {
            println!("{line}");
        }
    }
}

/// `--list` subprotocol: emit ktstr/gauntlet test names without
/// exiting so the standard libtest harness can also print its own
/// test list afterward. This is what makes plain `#[test]` items
/// inside a ktstr_test integration-test binary visible to nextest.
///
/// Honours `--ignored` the same way [`ktstr_main`] does — when set,
/// only the ignored subset (gauntlet variants and `demo_` base
/// tests) is printed. Unlike `ktstr_main`, this function returns to
/// the caller after listing so the ctor's caller can fall through
/// to libtest's `main`.
fn ktstr_list_only() {
    let args: Vec<String> = std::env::args().collect();
    let ignored_only = args.iter().any(|a| a == "--ignored");
    list_tests(ignored_only);
}

/// Nextest protocol handler.
///
/// Called automatically by [`ktstr_test_early_dispatch`] when running
/// under nextest with `--exact <ktstr_or_gauntlet_name>`.
/// Not intended for direct use.
///
/// - `--list --format terse`: output `ktstr/{name}: test\n` for base
///   tests and `gauntlet/{name}/{preset}: test\n` for gauntlet
///   variants. (Discovery uses `ktstr_list_only` instead to allow
///   libtest to print its own list afterward; this branch is
///   preserved for direct callers of `ktstr_main`.)
/// - `--exact NAME --nocapture`: run the named test, exit 0/1.
pub fn ktstr_main() -> ! {
    let args: Vec<String> = std::env::args().collect();

    // Discovery mode: --list --format terse [--ignored]
    if args.iter().any(|a| a == "--list") {
        let ignored_only = args.iter().any(|a| a == "--ignored");
        list_tests(ignored_only);
        std::process::exit(0);
    }

    // Execution mode: --exact NAME [--nocapture] [--ignored] [--bench]
    if let Some(pos) = args.iter().position(|a| a == "--exact") {
        if let Some(name) = args.get(pos + 1) {
            let code = run_named_test(name);
            std::process::exit(code);
        }
        eprintln!("--exact requires a test name");
        std::process::exit(1);
    }

    // Fallback: no recognized arguments.
    eprintln!("usage: <binary> --list --format terse [--ignored]");
    eprintln!("       <binary> --exact <test_name> --nocapture");
    std::process::exit(1)
}

#[cfg(test)]
#[path = "dispatch_tests.rs"]
mod tests;
