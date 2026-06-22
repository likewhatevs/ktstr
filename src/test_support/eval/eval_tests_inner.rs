//! Part of the eval module's unit-test suite, split across sibling
//! `eval_tests*.rs` files to keep each under the size ceiling. Child of
//! `eval`: reaches the production core via `super::` / `super::super::`.
//!
//! Covers the host-reachable arms of [`run_ktstr_test_inner`] and its
//! `run_ktstr_test_inner_impl` body that fire BEFORE any VM boot:
//! - the two validation gates (`entry.validate()` /
//!   `topo.validate()`), which return `Err` ahead of `ensure_kvm()`;
//! - the two pre-VM skip arms (`performance_mode` under
//!   `KTSTR_NO_PERF_MODE`, and a non-`performance_mode` entry under
//!   `KTSTR_PERF_ONLY`), which return `Ok(AssertResult::skip(..))`
//!   ahead of `ensure_kvm()` and emit a skip sidecar via
//!   `record_skip_sidecar`.
//!
//! Every test drives the public-crate wrapper `run_ktstr_test_inner`
//! (not the private `_impl`) so both the wrapper's post-call
//! marker/skip-class handling AND the body's early-return arms are
//! exercised in one call. The arms that boot a VM (`ensure_kvm`,
//! `resolve_test_kernel`, `builder.build()`, `vm.run()`, the bulk-drain
//! loop, the CpuStateGuard save/restore asm) are NOT host-unit-testable
//! and are intentionally untouched here.
use super::super::test_helpers::{EnvVarGuard, dummy_test_fn, eevdf_entry, lock_env};
use super::super::topo::TopoOverride;
// `use super::*` reaches `eval`, whose own `use super::{KtstrTestEntry}`
// brings the type into scope here (child modules see the parent's
// private `use` aliases via glob); the `SchedulerSpec` / `Topology`
// siblings the production fixtures need ride the same glob.
use super::*;

/// Count `*.ktstr.json` files under `dir` whose name starts with
/// `prefix`. Mirrors `sidecar::tests::find_sidecars_by_prefix` (a
/// sibling test module's private helper, not reachable from here) so
/// the skip-arm tests can confirm `record_skip_sidecar` wrote a real
/// sidecar into the `KTSTR_SIDECAR_DIR`-overridden temp dir.
fn count_sidecars_by_prefix(dir: &std::path::Path, prefix: &str) -> usize {
    std::fs::read_dir(dir)
        .expect("sidecar dir must exist for lookup")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with(prefix) && n.ends_with(".ktstr.json"))
        })
        .count()
}

// -- entry.validate() gate (first statement of run_ktstr_test_inner_impl) --

/// An entry whose `name` contains a path separator fails
/// `KtstrTestEntry::validate` at its `self.name.contains('/')` check, hit
/// by the `entry.validate().context("KtstrTestEntry validation")?` first
/// statement of `run_ktstr_test_inner_impl`, so the
/// wrapper returns `Err` BEFORE the rayon pin / `ensure_kvm` / VM boot.
/// The `.context("KtstrTestEntry validation")` wrapper text rides the
/// error chain, and the validate diagnostic ("must not contain path
/// separators") is the chain root. Pins that a malformed entry is
/// rejected on the host without touching KVM.
#[test]
fn run_ktstr_test_inner_invalid_entry_name_fails_validation() {
    let entry = KtstrTestEntry {
        // `/` trips the path-separator check in validate().
        name: "bad/name",
        func: dummy_test_fn,
        auto_repro: false,
        ..KtstrTestEntry::DEFAULT
    };
    let err = run_ktstr_test_inner(&entry, None).unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("KtstrTestEntry validation"),
        "the .context() wrapper must ride the chain, got: {msg}",
    );
    assert!(
        msg.contains("must not contain path separators"),
        "the validate diagnostic must surface as the chain root, got: {msg}",
    );
}

/// An entry with an empty `name` fails `KtstrTestEntry::validate`
/// at its `self.name.is_empty()` check — the second branch of the same
/// gate. Distinct from
/// the path-separator case: pins that the empty-name diagnostic
/// ("must be non-empty") also propagates through the validation
/// `.context()` wrapper without a VM boot. `KtstrTestEntry::DEFAULT`
/// ships `name: ""`, so the bare default (with only `func` set) is
/// already invalid here.
#[test]
fn run_ktstr_test_inner_empty_entry_name_fails_validation() {
    let entry = KtstrTestEntry {
        name: "",
        func: dummy_test_fn,
        auto_repro: false,
        ..KtstrTestEntry::DEFAULT
    };
    let err = run_ktstr_test_inner(&entry, None).unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("KtstrTestEntry validation"),
        "the validation .context() wrapper must ride the chain, got: {msg}",
    );
    assert!(
        msg.contains("must be non-empty"),
        "the empty-name validate diagnostic must surface, got: {msg}",
    );
}

// -- topo.validate() gate (the `if let Some(t) = topo` block in run_ktstr_test_inner_impl) --

/// A VALID entry paired with an INVALID `TopoOverride`
/// (`memory_mib == 0`) passes the `entry.validate()` first statement but
/// fails the `t.validate().context("TopoOverride validation")?` block
/// (`TopoOverride::validate`'s `self.memory_mib == 0` check). The wrapper
/// returns `Err` carrying the
/// `.context("TopoOverride validation")` wrapper plus the
/// zero-memory diagnostic as the chain root — proving the second gate
/// fires on the `Some(t)` topo path ahead of VM boot.
#[test]
fn run_ktstr_test_inner_invalid_topo_override_fails_validation() {
    let entry = eevdf_entry("__inner_bad_topo__");
    // numa/llc/core/thread all valid; only memory_mib==0 trips
    // TopoOverride::validate's last scalar gate.
    let topo = TopoOverride {
        numa_nodes: 1,
        llcs: 1,
        cores: 2,
        threads: 1,
        memory_mib: 0,
    };
    let err = run_ktstr_test_inner(&entry, Some(&topo)).unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("TopoOverride validation"),
        "the topo-gate .context() wrapper must ride the chain, got: {msg}",
    );
    assert!(
        msg.contains("memory_mib must be > 0"),
        "the zero-memory TopoOverride diagnostic must surface, got: {msg}",
    );
}

// -- performance_mode skip arm under KTSTR_NO_PERF_MODE (the `entry.performance_mode && no_perf_mode_active()` arm) --

/// A `performance_mode` entry under an active `KTSTR_NO_PERF_MODE`
/// takes the `entry.performance_mode && no_perf_mode_active()` skip arm:
/// `run_ktstr_test_inner` returns
/// `Ok(AssertResult::skip(REASON))` and that arm's `record_skip_sidecar`
/// call writes a skip sidecar — all BEFORE the `ensure_kvm()?`
/// statement, so no VM boots. The returned `AssertResult` is a skip
/// carrying the canonical REASON, and the sidecar lands in the
/// `KTSTR_SIDECAR_DIR`-overridden temp dir.
///
/// `KTSTR_PERF_ONLY` is cleared so the sibling perf-only skip arm
/// (the `perf_only_skips_entry(entry)` arm) cannot intercept — this test
/// pins the no-perf-mode arm specifically.
#[test]
fn run_ktstr_test_inner_skips_perf_test_under_no_perf_mode() {
    let _lock = lock_env();
    let sidecar_tmp = tempfile::TempDir::new().expect("sidecar tempdir");
    let _env_sidecar = EnvVarGuard::set(crate::KTSTR_SIDECAR_DIR_ENV, sidecar_tmp.path());
    let _env_no_perf = EnvVarGuard::set(crate::KTSTR_NO_PERF_MODE_ENV, "1");
    let _env_perf_only = EnvVarGuard::remove(crate::KTSTR_PERF_ONLY_ENV);

    let entry = KtstrTestEntry {
        name: "__inner_no_perf_skip__",
        func: dummy_test_fn,
        auto_repro: false,
        performance_mode: true,
        ..KtstrTestEntry::DEFAULT
    };
    let result = run_ktstr_test_inner(&entry, None)
        .expect("the no-perf-mode skip arm must return Ok(skip), not Err");
    assert!(
        result.is_skip(),
        "performance_mode under KTSTR_NO_PERF_MODE must yield a SKIP AssertResult",
    );
    let rendered = result
        .skip_details()
        .map(|d| d.message.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        rendered.contains("requires performance_mode but --no-perf-mode or KTSTR_NO_PERF_MODE"),
        "the skip AssertResult must carry the canonical no-perf-mode REASON, got: {rendered}",
    );
    assert_eq!(
        count_sidecars_by_prefix(sidecar_tmp.path(), "__inner_no_perf_skip__-"),
        1,
        "record_skip_sidecar must write exactly one skip sidecar into the override dir",
    );
}

// -- non-perf skip arm under KTSTR_PERF_ONLY (the `perf_only_skips_entry(entry)` arm) --

/// A non-`performance_mode` entry under an active `KTSTR_PERF_ONLY`
/// takes the second skip arm (the `perf_only_skips_entry(entry)` arm):
/// the run is restricted to
/// perf-mode tests, so this entry is skipped with the perf-only REASON
/// and that arm's `record_skip_sidecar` call emits its sidecar — again
/// BEFORE the `ensure_kvm()?` statement. Pins the `perf_only_skips_entry` gate's
/// integration into the eval entry, distinct from the standalone
/// `perf_only_skips_entry` unit tests in runtime.rs (which never drive
/// the surrounding return + sidecar write).
///
/// `KTSTR_NO_PERF_MODE` is cleared so the first skip arm (the
/// `entry.performance_mode && no_perf_mode_active()` arm)
/// cannot intercept (it would not anyway — this entry is
/// non-perf — but clearing it pins the perf-only arm unambiguously).
#[test]
fn run_ktstr_test_inner_skips_non_perf_test_under_perf_only() {
    let _lock = lock_env();
    let sidecar_tmp = tempfile::TempDir::new().expect("sidecar tempdir");
    let _env_sidecar = EnvVarGuard::set(crate::KTSTR_SIDECAR_DIR_ENV, sidecar_tmp.path());
    let _env_perf_only = EnvVarGuard::set(crate::KTSTR_PERF_ONLY_ENV, "1");
    let _env_no_perf = EnvVarGuard::remove(crate::KTSTR_NO_PERF_MODE_ENV);

    // performance_mode defaults to false on eevdf_entry, so this is the
    // "not a performance_mode test" the perf-only gate skips.
    let entry = eevdf_entry("__inner_perf_only_skip__");
    let result = run_ktstr_test_inner(&entry, None)
        .expect("the perf-only skip arm must return Ok(skip), not Err");
    assert!(
        result.is_skip(),
        "a non-perf entry under KTSTR_PERF_ONLY must yield a SKIP AssertResult",
    );
    let rendered = result
        .skip_details()
        .map(|d| d.message.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        rendered.contains("KTSTR_PERF_ONLY is active and this test is not a performance_mode test"),
        "the skip AssertResult must carry the canonical perf-only REASON, got: {rendered}",
    );
    assert_eq!(
        count_sidecars_by_prefix(sidecar_tmp.path(), "__inner_perf_only_skip__-"),
        1,
        "record_skip_sidecar must write exactly one skip sidecar into the override dir",
    );
}
