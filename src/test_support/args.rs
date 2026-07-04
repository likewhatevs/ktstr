//! CLI argument extraction for the ktstr dispatch path.
//!
//! The ktstr runtime hijacks its host binary's argv in two places — the
//! `#[ctor]` early-dispatch (host and guest) and nextest's `--exact`
//! invocation — so it needs a tiny, dependency-free parser that can
//! pick named values out of `std::env::args()` without getting in the
//! way of the harness's own flag handling.
//!
//! The `extract_*` helpers accept a `&[String]` slice and return the
//! first matching value or `None`. They are intentionally lenient: they
//! only recognize the `--ktstr-*=VALUE` form (or, for `--ktstr-test-fn`,
//! also the space-separated form) and ignore unknown flags entirely.
//! That keeps the dispatch path inert for binaries that aren't built
//! against ktstr.
//!
//! The remaining helpers have their own signatures and flag namespace:
//! [`current_work_type`] takes no argument (it reads `std::env::args()`
//! itself) and returns a `String`; the `--cell-parent-cgroup` helpers
//! ([`parse_cell_parent_cgroup`], [`cell_parent_path_is_valid`]) parse
//! and validate that flag rather than the `--ktstr-*` form.
//!
//! [`resolve_cgroup_root`] is a further outlier: it sources the path
//! from the initramfs-mounted `/sched_args` file first, then falls back
//! to the process argv. Used only from guest-side dispatch to derive the
//! cgroup manager root for the running test.

/// Extract the test function name from `--ktstr-test-fn=NAME` or
/// `--ktstr-test-fn NAME` in the argument list.
pub(crate) fn extract_test_fn_arg(args: &[String]) -> Option<&str> {
    let mut iter = args.iter();
    while let Some(a) = iter.next() {
        if let Some(val) = a.strip_prefix("--ktstr-test-fn=") {
            return Some(val);
        }
        if a == "--ktstr-test-fn" {
            return iter.next().map(|s| s.as_str());
        }
    }
    None
}

/// Extract `--ktstr-probe-stack=func1,func2,...` from the argument list.
pub(crate) fn extract_probe_stack_arg(args: &[String]) -> Option<String> {
    for a in args {
        if let Some(val) = a.strip_prefix("--ktstr-probe-stack=")
            && !val.is_empty()
        {
            return Some(val.to_string());
        }
    }
    None
}

/// Extract `--ktstr-topo=NnNlNcNt` from the argument list.
pub(crate) fn extract_topo_arg(args: &[String]) -> Option<String> {
    for a in args {
        if let Some(val) = a.strip_prefix("--ktstr-topo=")
            && !val.is_empty()
        {
            return Some(val.to_string());
        }
    }
    None
}

/// Extract `--ktstr-work-type=NAME` from the argument list.
pub(crate) fn extract_work_type_arg(args: &[String]) -> Option<String> {
    for a in args {
        if let Some(val) = a.strip_prefix("--ktstr-work-type=")
            && !val.is_empty()
        {
            return Some(val.to_string());
        }
    }
    None
}

/// The current run's `work_type` label, read from this process's args.
///
/// `--ktstr-work-type=NAME` (set per payload-gauntlet variant by the
/// dispatch) when present, else `"SpinWait"` — the default for a plain
/// test that drives no parameterized workload.
///
/// Distinct from [`crate::workload::resolve_work_type`], which resolves
/// a workload's effective [`WorkType`] (override + swappable logic)
/// during scenario setup; this returns the label STRING recorded in the
/// sidecar.
///
/// This is the SINGLE source of the `work_type` that flows into a
/// sidecar's `work_type` field and thus into [`sidecar_variant_hash`].
/// Both the run path (`run_ktstr_test_inner`) and the pre-VM-boot skip
/// path ([`write_skip_sidecar`]) call it, so a skip and a run of the
/// SAME config compute the IDENTICAL `work_type` — and therefore the
/// same variant hash, so a flaky test that skips on one attempt and
/// runs on a retry writes to one sidecar file (the retry overwrites the
/// skip) instead of two coexisting files. `std::env::args()` is the
/// same across nextest retry attempts of one test (same invocation), so
/// the value is stable per config.
///
/// [`WorkType`]: crate::workload::WorkType
/// [`sidecar_variant_hash`]: crate::test_support::sidecar::sidecar_variant_hash
/// [`write_skip_sidecar`]: crate::test_support::sidecar::write_skip_sidecar
pub(crate) fn current_work_type() -> String {
    let args: Vec<String> = std::env::args().collect();
    extract_work_type_arg(&args).unwrap_or_else(|| "SpinWait".to_string())
}

/// Extract `--ktstr-variant-hash=<16-hex>` from the argument list as a
/// `u64`. The host (`run_ktstr_test_inner_impl`) computes the run's
/// authoritative variant hash from the resolved config and injects it
/// into the guest's argv so the in-VM scenario [`Ctx`]'s
/// `failure_dump_path` / `wprof_pb_path` derive the SAME variant-keyed
/// paths the host writes — the guest cannot recompute the hash (its
/// argv lacks `--ktstr-work-type` and its topology is sysfs-observed).
/// `None` when absent (a manually-invoked guest) or unparseable; the
/// consumer falls back to `0`, and the `Ctx` path methods bail on
/// `entry_name == None` before a `0` hash could mislead.
///
/// [`Ctx`]: crate::scenario::Ctx
pub(crate) fn extract_variant_hash_arg(args: &[String]) -> Option<u64> {
    for a in args {
        if let Some(val) = a.strip_prefix("--ktstr-variant-hash=") {
            return u64::from_str_radix(val, 16).ok();
        }
    }
    None
}

/// Extract `--ktstr-export-test=NAME` from the argument list. Used by
/// the test binary's ctor to detect a `cargo ktstr export` self-export
/// dispatch (the binary embeds itself rather than letting cargo-ktstr
/// embed its own binary, which would package the wrong code).
///
/// Empty values resolve to `Some("")` so the ctor can surface an
/// actionable error rather than silently no-op when the operator
/// passes `--ktstr-export-test=`.
pub(crate) fn extract_export_test_arg(args: &[String]) -> Option<&str> {
    for a in args {
        if let Some(val) = a.strip_prefix("--ktstr-export-test=") {
            return Some(val);
        }
    }
    None
}

/// Extract `--ktstr-shell-test=NAME` from the argument list. Used by
/// `crate::test_support::dispatch::maybe_dispatch_shell_test` (the
/// test binary's main-path dispatch) to detect a `cargo ktstr shell
/// --test <NAME>` descriptor probe — the test binary itself owns
/// the `KTSTR_TESTS` distributed_slice, so cargo-ktstr probes each
/// test binary in the workspace to find the registered entry and
/// serialize its shell-relevant fields (topology, memory_mib,
/// extra_include_files, scheduler name + kind) back to stdout as
/// JSON.
///
/// Empty values intentionally return `Some("")`; the consumer at
/// `maybe_dispatch_shell_test` rejects empty names with an
/// actionable error and exit 1 so the router moves on to the next
/// candidate. (`extract_export_test_arg` uses the same shape;
/// the empty-handling lives at the consumer, not the extractor.)
pub(crate) fn extract_shell_test_arg(args: &[String]) -> Option<&str> {
    for a in args {
        if let Some(val) = a.strip_prefix("--ktstr-shell-test=") {
            return Some(val);
        }
    }
    None
}

/// Extract `--ktstr-export-output=PATH` from the argument list. Pairs
/// with [`extract_export_test_arg`] to direct the generated `.run`
/// file at a specific path; absent means "default to `<test>.run` in
/// cwd."
#[cfg(feature = "export")]
pub(crate) fn extract_export_output_arg(args: &[String]) -> Option<&str> {
    for a in args {
        if let Some(val) = a.strip_prefix("--ktstr-export-output=")
            && !val.is_empty()
        {
            return Some(val);
        }
    }
    None
}

/// The bare boolean flag that marks a guest run as a verifier-workload
/// dispatch probe. Written onto the guest `run_args` by
/// [`crate::verifier::collect_verifier_output`] and detected here by
/// [`is_verifier_workload`]. Single source of truth so the writer (host
/// dispatcher) and reader (guest init) never spell the flag by hand.
pub(crate) const VERIFIER_WORKLOAD_FLAG: &str = "--ktstr-verifier-workload";

/// Whether [`VERIFIER_WORKLOAD_FLAG`] is present in the argument list.
/// Consumed at `ktstr_guest_init` Phase 5: instead of dispatching a
/// `#[ktstr_test]` body (the verifier sweep VM has none) it runs the
/// SpinWait dispatch probe and, on confirmed worker progress after the
/// scheduler attaches, emits a
/// [`crate::vmm::wire::LifecyclePhase::WorkloadDispatched`] frame. A bare
/// boolean flag — presence is the whole signal, no value.
pub(crate) fn is_verifier_workload(args: &[String]) -> bool {
    args.iter().any(|a| a == VERIFIER_WORKLOAD_FLAG)
}

/// Canonical name for the cgroup-parent flag scx schedulers accept.
/// The auto-inject (`runtime::append_base_sched_args`), the guest's
/// boot-time cgroup-tree creator (`vmm::rust_init`), and this guest-
/// side resolver all read it through [`parse_cell_parent_cgroup`].
pub(crate) const CELL_PARENT_CGROUP_FLAG: &str = "--cell-parent-cgroup";

/// Compile-time guard: the literal `--cell-parent-cgroup=` prefix
/// used inside [`parse_cell_parent_cgroup`] must equal
/// [`CELL_PARENT_CGROUP_FLAG`] with a trailing `=`. A future rename
/// of the flag that misses the literal would fail this const-eval
/// instead of silently mismatching at runtime.
const _: () = {
    let flag = CELL_PARENT_CGROUP_FLAG.as_bytes();
    let prefix = b"--cell-parent-cgroup=";
    assert!(prefix.len() == flag.len() + 1);
    let mut i = 0;
    while i < flag.len() {
        assert!(prefix[i] == flag[i]);
        i += 1;
    }
    assert!(prefix[flag.len()] == b'=');
};

/// Outcome of `parse_cell_parent_cgroup`. Distinguishes "user didn't
/// supply the flag" from "user supplied the flag but left the value
/// empty/missing" so the auto-inject path can fire only in the
/// genuinely-absent case. Without this distinction, a trailing bare
/// `--cell-parent-cgroup` (no following token) would parse as
/// `Absent` and trigger the auto-inject, producing two copies of the
/// flag in the final argv that clap then rejects with a confused
/// "cannot be used multiple times" diagnostic.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum CellParentCgroupArg<'a> {
    /// `--cell-parent-cgroup` not present anywhere in the argv stream.
    Absent,
    /// `--cell-parent-cgroup{=,SP}<value>` present; `value` is whatever
    /// the user supplied (which may itself be invalid — callers must
    /// run the absolute-path / non-root check before downstream use).
    Value(&'a str),
    /// `--cell-parent-cgroup` present as a bare flag with no
    /// following token (two-token form, trailing-in-argv). Distinct
    /// from `Absent` because the user clearly intended to supply the
    /// flag, just incompletely.
    MissingValue,
}

/// True iff `path` is a valid `--cell-parent-cgroup` value: absolute,
/// non-trivial, and contains no `..` components that would normalize
/// back to (or escape) `/sys/fs/cgroup`.
///
/// The downstream consumer at `vmm/rust_init/mounts.rs::create_cgroup_from_file`
/// concatenates the value to `/sys/fs/cgroup` and mkdirs the result.
/// A value like `/foo/..` produces `/sys/fs/cgroup/foo/..` which the
/// kernel canonicalizes back to `/sys/fs/cgroup` — corrupting the host
/// cgroup root the same way an empty / bare-`/` value would. `/.`
/// behaves identically.
///
/// `Path::components` strips every `CurDir` (`.`) segment from any
/// position in the path (leading, mid, trailing) before yielding the
/// component sequence; only `RootDir`, `Normal`, and `ParentDir`
/// survive. So:
///   - `/.` yields `[RootDir]` (rejected: no Normal component)
///   - `/foo/./bar` yields `[RootDir, Normal("foo"), Normal("bar")]`
///     (accepted; canonical form is a real non-root path)
///   - `/foo/..` yields `[RootDir, Normal("foo"), ParentDir]`
///     (rejected: ParentDir present)
///   - `/./bar/..` yields `[RootDir, Normal("bar"), ParentDir]`
///     (rejected: ParentDir present; leading `/.` stripped first)
///
/// The validator is therefore the conjunction of "has at least one
/// Normal component" AND "no ParentDir component" — both classes of
/// normalize-to-host-root are caught. The `CurDir` arm in the match
/// below is dead under current `Path::components` semantics; it is
/// kept as defense-in-depth in case a future stdlib change starts
/// yielding `CurDir` for some path shape.
///
/// Both validation arms (host-side fail-fast in
/// `runtime::append_base_sched_args`, guest-side defense in
/// `vmm/rust_init/mounts.rs::create_cgroup_from_file`) share this predicate
/// so the host gate and the guest-side defense-in-depth stay aligned.
pub(crate) fn cell_parent_path_is_valid(path: &str) -> bool {
    if !path.starts_with('/') {
        return false;
    }
    let mut has_normal = false;
    for component in std::path::Path::new(path).components() {
        use std::path::Component;
        match component {
            Component::Normal(_) => has_normal = true,
            Component::RootDir => {} // single leading `/`, expected
            // CurDir is currently unreachable (see fn docstring) but
            // kept as defense-in-depth.
            Component::CurDir | Component::ParentDir => return false,
            Component::Prefix(_) => return false, // Windows; not applicable
        }
    }
    has_normal
}

/// Find the value passed to `--cell-parent-cgroup` in an argv stream,
/// accepting both the two-token form (`["--cell-parent-cgroup",
/// "/path"]`) and the combined form (`["--cell-parent-cgroup=/path"]`).
/// First match wins. Returns [`CellParentCgroupArg::Absent`] when the
/// flag is not present, [`CellParentCgroupArg::Value`] for a found
/// value (whose contents callers must independently validate), or
/// [`CellParentCgroupArg::MissingValue`] when the bare flag appears
/// as the last token with no following value.
///
/// Caveat — positional-naive: this walker treats every token equal to
/// the canonical flag (or with the `=` prefix) as our flag without
/// regard for whether the preceding token was a flag-with-value
/// expecting a positional value. If an upstream scheduler argv
/// contains a value-taking flag followed by `--cell-parent-cgroup` as
/// that value (e.g. `["--prev-flag", "--cell-parent-cgroup",
/// "/user"]` where `--prev-flag` consumes the next token), this
/// parser would still grab `/user` as if our flag had been written.
/// A fully correct parse would need a scheduler-flag-spec table; the
/// auto-inject and guest-side resolver intentionally stay flag-spec-
/// agnostic, accepting the false-positive risk for arg shapes that
/// the in-tree schedulers don't currently produce. The combined-form
/// branch (`--cell-parent-cgroup=...`) is unambiguous and unaffected.
pub(crate) fn parse_cell_parent_cgroup<'a>(
    args: impl IntoIterator<Item = &'a str>,
) -> CellParentCgroupArg<'a> {
    // The combined-form prefix is `CELL_PARENT_CGROUP_FLAG` plus `=`.
    // Inlined here so the strip_prefix sees a const-foldable literal;
    // keep the spellings in sync if the flag ever changes.
    const COMBINED_PREFIX: &str = "--cell-parent-cgroup=";
    let mut iter = args.into_iter();
    while let Some(a) = iter.next() {
        if a == CELL_PARENT_CGROUP_FLAG {
            return match iter.next() {
                Some(v) => CellParentCgroupArg::Value(v),
                None => CellParentCgroupArg::MissingValue,
            };
        }
        if let Some(rest) = a.strip_prefix(COMBINED_PREFIX) {
            return CellParentCgroupArg::Value(rest);
        }
    }
    CellParentCgroupArg::Absent
}

/// Derive the CgroupManager root path for guest-side dispatch.
///
/// Reads `/sched_args` to find `--cell-parent-cgroup` (either form),
/// then falls back to process argv. When a valid value (starts with
/// `/`, not bare `/`) is found, constructs `/sys/fs/cgroup{path}`.
/// Falls back to `/sys/fs/cgroup/ktstr` when the arg is absent OR
/// when it's malformed (missing value, empty, bare `/`, or
/// non-absolute) — the host-side gate in `append_base_sched_args`
/// already panics on those shapes, so reaching this fallback means
/// the gate was bypassed (operator hand-edited an exported `.run`
/// script, ad-hoc argv injection). Logging is limited guest-side;
/// surface the bad value via stderr and continue with the default
/// so the test doesn't silently land on the host cgroup root.
pub(crate) fn resolve_cgroup_root(args: &[String]) -> String {
    // Priority 1: per-test `workload_root_cgroup` from
    // `KtstrTestEntry`. The host writes the validated absolute
    // path to `/workload_root_cgroup` in the initramfs suffix
    // (`vmm::initramfs::build_suffix`) when the test entry sets
    // it. Empty file / unreadable / invalid path falls through
    // to the legacy resolution so a stale-image edge case stays
    // backward compatible.
    if let Ok(raw) = std::fs::read_to_string("/workload_root_cgroup") {
        let trimmed = raw.trim();
        if cell_parent_path_is_valid(trimmed) {
            return format!("/sys/fs/cgroup{trimmed}");
        } else if !trimmed.is_empty() {
            eprintln!(
                "ktstr_test: ignoring malformed `/workload_root_cgroup` \
                 value {trimmed:?}; falling back to legacy cgroup-root \
                 resolution. The host-side gate in `CgroupPath::new` \
                 normally rejects this at compile time."
            );
        }
    }
    // Priority 2: `--cell-parent-cgroup` in `/sched_args`. Only
    // present when the scheduler declaration (or per-test
    // `extra_sched_args`) explicitly carries the flag; the
    // framework no longer auto-injects it from
    // `Scheduler::cgroup_parent`.
    let sched_args = std::fs::read_to_string("/sched_args").unwrap_or_default();
    if let Some(path) = absolute_cell_parent_value(
        parse_cell_parent_cgroup(sched_args.split_whitespace()),
        "/sched_args",
    ) {
        return format!("/sys/fs/cgroup{path}");
    }
    // Priority 3: process argv (matches sched_args parsing for
    // direct ad-hoc argv injection through `extra_sched_args`).
    if let Some(path) = absolute_cell_parent_value(
        parse_cell_parent_cgroup(args.iter().map(String::as_str)),
        "process argv",
    ) {
        return format!("/sys/fs/cgroup{path}");
    }
    "/sys/fs/cgroup/ktstr".to_string()
}

/// Defense-in-depth filter for guest-side consumers of
/// `parse_cell_parent_cgroup`. Returns `Some(path)` only for values
/// that pass the same validation the host-side gate enforces in
/// `runtime::append_base_sched_args` (starts with `/`, not bare `/`);
/// returns `None` for `Absent`, `MissingValue`, or invalid `Value`
/// shapes. Each rejection logs to stderr naming the offending source
/// so an operator inspecting guest output can trace the bad value
/// back to its origin even when the host-side gate was bypassed.
fn absolute_cell_parent_value<'a>(
    parsed: CellParentCgroupArg<'a>,
    source: &str,
) -> Option<&'a str> {
    match parsed {
        CellParentCgroupArg::Value(path) if path.starts_with('/') && path != "/" => Some(path),
        CellParentCgroupArg::Value(path) => {
            eprintln!(
                "ktstr_test: ignoring malformed `--cell-parent-cgroup` value {path:?} \
                 from {source}; falling back to default cgroup root. The host-side \
                 gate normally panics on this; reaching this branch means the gate \
                 was bypassed (hand-edited export script, ad-hoc argv injection).",
            );
            None
        }
        CellParentCgroupArg::MissingValue => {
            eprintln!(
                "ktstr_test: ignoring bare `--cell-parent-cgroup` (no following value) \
                 from {source}; falling back to default cgroup root.",
            );
            None
        }
        CellParentCgroupArg::Absent => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- extract_test_fn_arg --

    #[test]
    fn extract_test_fn_arg_equals() {
        let args = vec![
            "ktstr".into(),
            "run".into(),
            "--ktstr-test-fn=my_test".into(),
        ];
        assert_eq!(extract_test_fn_arg(&args), Some("my_test"));
    }

    #[test]
    fn extract_test_fn_arg_space() {
        let args = vec![
            "ktstr".into(),
            "run".into(),
            "--ktstr-test-fn".into(),
            "my_test".into(),
        ];
        assert_eq!(extract_test_fn_arg(&args), Some("my_test"));
    }

    #[test]
    fn extract_test_fn_arg_missing() {
        let args = vec!["ktstr".into(), "run".into()];
        assert!(extract_test_fn_arg(&args).is_none());
    }

    #[test]
    fn extract_test_fn_arg_trailing() {
        let args = vec!["ktstr".into(), "run".into(), "--ktstr-test-fn".into()];
        assert!(extract_test_fn_arg(&args).is_none());
    }

    #[test]
    fn extract_test_fn_arg_empty_value() {
        let args = vec!["ktstr".into(), "run".into(), "--ktstr-test-fn=".into()];
        assert_eq!(extract_test_fn_arg(&args), Some(""));
    }

    #[test]
    fn extract_test_fn_arg_space_form_empty_args() {
        let args: Vec<String> = vec![];
        assert!(extract_test_fn_arg(&args).is_none());
    }

    // -- is_verifier_workload --

    #[test]
    fn is_verifier_workload_present() {
        let args = vec!["/init".into(), VERIFIER_WORKLOAD_FLAG.to_string()];
        assert!(is_verifier_workload(&args));
    }

    #[test]
    fn is_verifier_workload_absent() {
        let args = vec!["/init".into(), "--ktstr-test-fn=foo".into()];
        assert!(!is_verifier_workload(&args));
    }

    #[test]
    fn is_verifier_workload_empty_args() {
        let args: Vec<String> = vec![];
        assert!(!is_verifier_workload(&args));
    }

    // -- extract_probe_stack_arg --

    #[test]
    fn extract_probe_stack_arg_equals() {
        let args = vec![
            "ktstr".into(),
            "run".into(),
            "--ktstr-probe-stack=func_a,func_b".into(),
        ];
        assert_eq!(
            extract_probe_stack_arg(&args),
            Some("func_a,func_b".to_string())
        );
    }

    #[test]
    fn extract_probe_stack_arg_missing() {
        let args = vec!["ktstr".into(), "run".into()];
        assert!(extract_probe_stack_arg(&args).is_none());
    }

    #[test]
    fn extract_probe_stack_arg_empty_value() {
        let args = vec!["ktstr".into(), "--ktstr-probe-stack=".into()];
        assert!(extract_probe_stack_arg(&args).is_none());
    }

    // -- extract_topo_arg --

    #[test]
    fn extract_topo_arg_equals() {
        let args = vec!["bin".into(), "--ktstr-topo=1n2l4c2t".into()];
        assert_eq!(extract_topo_arg(&args), Some("1n2l4c2t".to_string()));
    }

    #[test]
    fn extract_topo_arg_missing() {
        let args = vec!["bin".into(), "--ktstr-test-fn=test".into()];
        assert!(extract_topo_arg(&args).is_none());
    }

    #[test]
    fn extract_topo_arg_empty_value() {
        let args = vec!["bin".into(), "--ktstr-topo=".into()];
        assert!(extract_topo_arg(&args).is_none());
    }

    #[test]
    fn extract_topo_arg_with_other_args() {
        let args = vec![
            "bin".into(),
            "--ktstr-test-fn=my_test".into(),
            "--ktstr-topo=1n1l2c1t".into(),
        ];
        assert_eq!(extract_topo_arg(&args), Some("1n1l2c1t".to_string()));
    }

    // -- extract_work_type_arg --

    #[test]
    fn extract_work_type_arg_equals() {
        let args = vec!["ktstr".into(), "--ktstr-work-type=SpinWait".into()];
        assert_eq!(extract_work_type_arg(&args), Some("SpinWait".to_string()));
    }

    #[test]
    fn extract_work_type_arg_missing() {
        let args = vec!["ktstr".into(), "run".into()];
        assert!(extract_work_type_arg(&args).is_none());
    }

    #[test]
    fn extract_work_type_arg_empty_value() {
        let args = vec!["ktstr".into(), "--ktstr-work-type=".into()];
        assert!(extract_work_type_arg(&args).is_none());
    }

    // -- extract_export_test_arg --

    #[test]
    fn extract_export_test_arg_equals() {
        let args = vec![
            "test_bin".into(),
            "--ktstr-export-test=preempt_regression".into(),
        ];
        assert_eq!(extract_export_test_arg(&args), Some("preempt_regression"),);
    }

    #[test]
    fn extract_export_test_arg_missing() {
        let args = vec!["test_bin".into(), "--list".into()];
        assert!(extract_export_test_arg(&args).is_none());
    }

    /// Empty value resolves to Some("") so the ctor can surface an
    /// actionable diagnostic rather than silently no-op when the
    /// router (or operator) accidentally passes the flag without a
    /// value.
    #[test]
    fn extract_export_test_arg_empty_value() {
        let args = vec!["test_bin".into(), "--ktstr-export-test=".into()];
        assert_eq!(extract_export_test_arg(&args), Some(""));
    }

    // -- extract_export_output_arg --
    //
    // Gated alongside the function itself (`extract_export_output_arg`
    // is `#[cfg(feature = "export")]` because the only call site is
    // the export-feature-gated dispatch path).

    #[cfg(feature = "export")]
    #[test]
    fn extract_export_output_arg_equals() {
        let args = vec![
            "test_bin".into(),
            "--ktstr-export-output=/tmp/foo.run".into(),
        ];
        assert_eq!(extract_export_output_arg(&args), Some("/tmp/foo.run"),);
    }

    #[cfg(feature = "export")]
    #[test]
    fn extract_export_output_arg_missing() {
        let args = vec!["test_bin".into()];
        assert!(extract_export_output_arg(&args).is_none());
    }

    /// Empty value treated as absent — the export path falls back to
    /// the default `<test>.run` in the current directory rather than
    /// trying to write to an empty path string.
    #[cfg(feature = "export")]
    #[test]
    fn extract_export_output_arg_empty_value() {
        let args = vec!["test_bin".into(), "--ktstr-export-output=".into()];
        assert!(extract_export_output_arg(&args).is_none());
    }

    // -- parse_cell_parent_cgroup --

    #[test]
    fn parse_cell_parent_cgroup_two_token_form() {
        let argv = ["--cell-parent-cgroup", "/user", "--other-flag"];
        assert_eq!(
            parse_cell_parent_cgroup(argv.iter().copied()),
            CellParentCgroupArg::Value("/user")
        );
    }

    #[test]
    fn parse_cell_parent_cgroup_combined_form() {
        let argv = ["--other-flag", "--cell-parent-cgroup=/user"];
        assert_eq!(
            parse_cell_parent_cgroup(argv.iter().copied()),
            CellParentCgroupArg::Value("/user")
        );
    }

    /// Combined-form with no characters after `=` parses as `Value("")`
    /// — distinct from `MissingValue`. The combined-form prefix
    /// `--cell-parent-cgroup=` always anchors on `=` and treats
    /// everything after as the value (even if empty), so the user
    /// SUPPLIED a value (just empty); the host gate's absolute-path
    /// check rejects the empty string downstream. MissingValue is
    /// reserved for the two-token form where the bare flag is the
    /// trailing token with no following argv element at all.
    #[test]
    fn parse_cell_parent_cgroup_empty_combined_value() {
        let argv = ["--cell-parent-cgroup="];
        assert_eq!(
            parse_cell_parent_cgroup(argv.iter().copied()),
            CellParentCgroupArg::Value("")
        );
    }

    #[test]
    fn parse_cell_parent_cgroup_absent() {
        let argv = ["--unrelated", "--other-flag=value"];
        assert_eq!(
            parse_cell_parent_cgroup(argv.iter().copied()),
            CellParentCgroupArg::Absent
        );
    }

    /// Bare trailing flag (two-token form, no following token) is now
    /// distinguished from `Absent` so the auto-inject path can fire
    /// only on genuinely-missing flags. Previously the parser returned
    /// `None` for both shapes, which let auto-inject silently fire on
    /// a malformed bare-flag invocation and produce a confusing clap
    /// duplicate-rejection downstream.
    #[test]
    fn parse_cell_parent_cgroup_two_token_missing_value() {
        let argv = ["--cell-parent-cgroup"];
        assert_eq!(
            parse_cell_parent_cgroup(argv.iter().copied()),
            CellParentCgroupArg::MissingValue
        );
    }

    /// Bare flag preceded by an unrelated token still trips the
    /// `MissingValue` case when nothing follows.
    #[test]
    fn parse_cell_parent_cgroup_bare_flag_at_end_after_other() {
        let argv = ["--other-flag", "--cell-parent-cgroup"];
        assert_eq!(
            parse_cell_parent_cgroup(argv.iter().copied()),
            CellParentCgroupArg::MissingValue
        );
    }

    #[test]
    fn parse_cell_parent_cgroup_first_match_wins() {
        let argv = [
            "--cell-parent-cgroup=/first",
            "--cell-parent-cgroup",
            "/second",
        ];
        assert_eq!(
            parse_cell_parent_cgroup(argv.iter().copied()),
            CellParentCgroupArg::Value("/first")
        );
    }

    #[test]
    fn parse_cell_parent_cgroup_first_match_wins_two_token_then_combined() {
        let argv = [
            "--cell-parent-cgroup",
            "/first",
            "--cell-parent-cgroup=/second",
        ];
        assert_eq!(
            parse_cell_parent_cgroup(argv.iter().copied()),
            CellParentCgroupArg::Value("/first")
        );
    }

    /// Sibling long flags like `--cell-parent-cgroup-extra=val` must
    /// NOT match. The combined-form check anchors on `=` immediately
    /// after the canonical name — a longer flag with the same prefix
    /// followed by `-` falls through.
    #[test]
    fn parse_cell_parent_cgroup_no_match_on_sibling_long_flag() {
        let argv = ["--cell-parent-cgroup-extra=val"];
        assert_eq!(
            parse_cell_parent_cgroup(argv.iter().copied()),
            CellParentCgroupArg::Absent
        );
    }

    // -- absolute_cell_parent_value --
    //
    // The filter is the entire defense-in-depth for guest-side
    // `resolve_cgroup_root` consumers — a regression here would
    // silently corrupt downstream `format!("/sys/fs/cgroup{path}")`
    // interpolation when the host-side gate is bypassed. Pin the
    // full return-value contract: Some(path) iff the Value variant
    // passes the host-side validation criterion (starts with `/`,
    // not bare `/`); None for every other shape (Absent,
    // MissingValue, invalid Value). The eprintln! side-effects are
    // not asserted — return-value coverage is sufficient for the
    // function's documented contract.

    #[test]
    fn absolute_cell_parent_value_returns_valid_path() {
        assert_eq!(
            absolute_cell_parent_value(CellParentCgroupArg::Value("/ktstr"), "test"),
            Some("/ktstr")
        );
    }

    #[test]
    fn absolute_cell_parent_value_rejects_empty() {
        assert_eq!(
            absolute_cell_parent_value(CellParentCgroupArg::Value(""), "test"),
            None
        );
    }

    #[test]
    fn absolute_cell_parent_value_rejects_bare_slash() {
        assert_eq!(
            absolute_cell_parent_value(CellParentCgroupArg::Value("/"), "test"),
            None
        );
    }

    #[test]
    fn absolute_cell_parent_value_rejects_relative() {
        assert_eq!(
            absolute_cell_parent_value(CellParentCgroupArg::Value("my_test"), "test"),
            None
        );
    }

    #[test]
    fn absolute_cell_parent_value_rejects_missing_value() {
        assert_eq!(
            absolute_cell_parent_value(CellParentCgroupArg::MissingValue, "test"),
            None
        );
    }

    #[test]
    fn absolute_cell_parent_value_returns_none_on_absent() {
        assert_eq!(
            absolute_cell_parent_value(CellParentCgroupArg::Absent, "test"),
            None
        );
    }
}
