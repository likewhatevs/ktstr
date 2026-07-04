//! Unit-test suite for the eval module's post_vm / dedupe / kernel /
//! scheduler areas, split out of `eval/mod.rs` to keep it under the
//! size ceiling; sibling `eval_tests_*.rs` files cover the other areas.
//! As a child module of `eval`, these tests reach the production core
//! via `super::` / `super::super::`.
use super::super::test_helpers::{EnvVarGuard, eevdf_entry, lock_env, make_vm_result};
use super::*;
use crate::sync::MutexExt;
use tempfile::TempDir;

// -- combine_post_vm_errs tests --
//
// Pin the combine semantics: when both conditional and
// unconditional post_vm callbacks fail in the same run,
// both errors MUST surface in the combined message so a
// debugging operator sees both regressions on the first
// pass. A `.or()` shape would silently drop one signal,
// defeating the whole point of the unconditional callback.

/// Both Some → combined message names both errors with the
/// `post_vm:` / `post_vm_unconditional:` prefixes so the
/// operator can route each failure to the right callback.
#[test]
fn combine_post_vm_errs_both_fail_surfaces_both_signals() {
    let c = anyhow::anyhow!("snapshot-bridge captured nothing");
    let u = anyhow::anyhow!("wprof .pb missing at <path>");
    let combined = super::post_vm::combine_post_vm_errs(Some(c), Some(u))
        .expect("both Some inputs produce Some output");
    let rendered = format!("{combined:#}");
    assert!(
        rendered.contains("post_vm:"),
        "combined message must label the conditional fail: {rendered}",
    );
    assert!(
        rendered.contains("snapshot-bridge captured nothing"),
        "conditional fail message must be preserved: {rendered}",
    );
    assert!(
        rendered.contains("post_vm_unconditional:"),
        "combined message must label the unconditional fail: {rendered}",
    );
    assert!(
        rendered.contains("wprof .pb missing"),
        "unconditional fail message must be preserved: {rendered}",
    );
}

/// Conditional Some, unconditional None → return the
/// conditional Err unchanged (no labeling overhead when only
/// one signal fired).
#[test]
fn combine_post_vm_errs_only_conditional_passes_through() {
    let c = anyhow::anyhow!("snapshot-bridge captured nothing");
    let combined = super::post_vm::combine_post_vm_errs(Some(c), None)
        .expect("conditional Some produces Some output");
    let rendered = format!("{combined:#}");
    assert_eq!(rendered, "snapshot-bridge captured nothing");
}

/// Unconditional Some, conditional None → return the
/// unconditional Err unchanged. Pins that the unconditional
/// signal reaches the operator even when post_vm did NOT
/// fire (the typical case: guest passed, host-side check
/// failed).
#[test]
fn combine_post_vm_errs_only_unconditional_passes_through() {
    let u = anyhow::anyhow!("wprof .pb missing at <path>");
    let combined = super::post_vm::combine_post_vm_errs(None, Some(u))
        .expect("unconditional Some produces Some output");
    let rendered = format!("{combined:#}");
    assert_eq!(rendered, "wprof .pb missing at <path>");
}

/// Both None → None. Pins the no-op case so the dispatch
/// site at `run_ktstr_test_inner_impl` correctly skips the
/// placeholder-dump-write when neither callback fired.
#[test]
fn combine_post_vm_errs_both_none_returns_none() {
    let combined = super::post_vm::combine_post_vm_errs(None, None);
    assert!(combined.is_none());
}

// -- run_post_vm_callbacks tests --
//
// Pin the dispatch semantics extracted from
// `run_ktstr_test_inner_impl` into a testable free function.
// Fn-pointer callbacks let the test assert on which slot
// fired (Ok / Err / panic) under each combination of
// `guest_already_failed` + the entry's two callback slots.
// A regression that broke either the guest-fail suppression
// contract or the `catch_unwind` panic-catch around the
// callbacks would surface in the tests below.

fn post_vm_ok(_result: &crate::vmm::VmResult) -> anyhow::Result<()> {
    Ok(())
}
fn post_vm_err_conditional(_result: &crate::vmm::VmResult) -> anyhow::Result<()> {
    Err(anyhow::anyhow!("snapshot-bridge captured nothing"))
}
fn post_vm_err_unconditional(_result: &crate::vmm::VmResult) -> anyhow::Result<()> {
    Err(anyhow::anyhow!("wprof .pb missing at <path>"))
}
fn post_vm_panic(_result: &crate::vmm::VmResult) -> anyhow::Result<()> {
    panic!("simulated callback panic");
}

/// `post_vm_unconditional` fires when guest_already_failed=true
/// (where `post_vm` is suppressed). Pins the contract that the
/// unconditional callback bypasses the guest-fail suppression
/// gate.
#[test]
fn run_post_vm_callbacks_unconditional_fires_on_guest_fail() {
    let mut entry = eevdf_entry("test_unconditional_on_guest_fail");
    entry.post_vm = Some(post_vm_err_conditional);
    entry.post_vm_unconditional = Some(post_vm_err_unconditional);
    let result = make_vm_result("", "", 0, false);
    let combined =
        super::run_post_vm_callbacks(&entry, &result, /*guest_already_failed=*/ true)
            .expect("post_vm_unconditional must produce Some(err) when guest failed");
    let rendered = format!("{combined:#}");
    // post_vm suppressed: its message must NOT appear.
    assert!(
        !rendered.contains("snapshot-bridge captured nothing"),
        "post_vm must be suppressed on guest-fail: {rendered}",
    );
    // post_vm_unconditional fired: its message MUST appear.
    assert!(
        rendered.contains("wprof .pb missing"),
        "post_vm_unconditional must run on guest-fail: {rendered}",
    );
}

/// `post_vm` suppressed when guest_already_failed=true AND
/// `post_vm_unconditional` not set → None. Pins the no-op
/// case so the dispatch site correctly skips the
/// placeholder-dump-write when only `post_vm` is wired and
/// the guest already failed.
#[test]
fn run_post_vm_callbacks_conditional_suppressed_on_guest_fail() {
    let mut entry = eevdf_entry("test_conditional_suppressed_on_guest_fail");
    entry.post_vm = Some(post_vm_err_conditional);
    entry.post_vm_unconditional = None;
    let result = make_vm_result("", "", 0, false);
    let combined =
        super::run_post_vm_callbacks(&entry, &result, /*guest_already_failed=*/ true);
    assert!(
        combined.is_none(),
        "post_vm must be suppressed AND no unconditional → None: {combined:?}",
    );
}

/// Both callbacks return Ok → None.
#[test]
fn run_post_vm_callbacks_both_ok_returns_none() {
    let mut entry = eevdf_entry("test_both_ok");
    entry.post_vm = Some(post_vm_ok);
    entry.post_vm_unconditional = Some(post_vm_ok);
    let result = make_vm_result("", "", 0, false);
    let combined =
        super::run_post_vm_callbacks(&entry, &result, /*guest_already_failed=*/ false);
    assert!(combined.is_none(), "both Ok → None: {combined:?}");
}

/// `post_vm_unconditional` panic is caught and surfaced as an
/// error with the `post_vm_unconditional callback panicked:`
/// prefix. Pins the catch_unwind contract — without the wrap,
/// the panic would unwind past the dispatch site and leak VM
/// resources. The label prefix lets the operator distinguish a
/// conditional panic from an unconditional one when both
/// callbacks are wired.
#[test]
#[cfg(panic = "unwind")]
fn run_post_vm_callbacks_unconditional_panic_caught() {
    let mut entry = eevdf_entry("test_unconditional_panic");
    entry.post_vm = None;
    entry.post_vm_unconditional = Some(post_vm_panic);
    let result = make_vm_result("", "", 0, false);
    let combined = super::run_post_vm_callbacks(&entry, &result, false)
        .expect("panicking callback must produce Some(err)");
    let rendered = format!("{combined:#}");
    assert!(
        rendered.contains("post_vm_unconditional callback panicked:"),
        "panic must carry the slot label: {rendered}",
    );
    assert!(
        rendered.contains("simulated callback panic"),
        "panic message must be preserved: {rendered}",
    );
}

// -- dedupe_include_files tests --
//
// Policy pins for the aggregator downstream of
// `KtstrTestEntry::all_include_files` + `resolve_include_files`:
// identical `(archive, host)` pairs collapse silently, same
// archive with conflicting hosts aborts. Deterministic
// ordering (BTreeMap keys).

/// Empty input → empty result. Pins the identity case so a
/// regression that introduces an invariant init-element
/// (e.g. implicit config file) would break this.
#[test]
fn dedupe_include_files_empty_input() {
    let out = dedupe_include_files(&[]).unwrap();
    assert!(out.is_empty(), "empty in → empty out, got {out:?}");
}

/// Identical pair appearing twice deduplicates silently. The
/// output contains a single entry; no error, no warning. Models
/// the scheduler-and-payload-both-declare-config case.
#[test]
fn dedupe_include_files_identical_pair_collapses() {
    let input = vec![
        (
            "include-files/helper".to_string(),
            std::path::PathBuf::from("/usr/bin/helper"),
            "declarative",
        ),
        (
            "include-files/helper".to_string(),
            std::path::PathBuf::from("/usr/bin/helper"),
            "scheduler config_file",
        ),
    ];
    let out = dedupe_include_files(&input).unwrap();
    assert_eq!(out.len(), 1, "identical pair must dedupe, got {out:?}");
    assert_eq!(out[0].0, "include-files/helper");
    assert_eq!(out[0].1, std::path::PathBuf::from("/usr/bin/helper"));
}

/// Same archive_path with conflicting host_paths is a genuine
/// ambiguity — one declaration would silently overwrite the
/// other's file in the initramfs. Policy: hard error with a
/// diagnostic naming both host paths so the operator knows
/// which declarations need disambiguation.
#[test]
fn dedupe_include_files_archive_collision_errors() {
    let input = vec![
        (
            "include-files/config.json".to_string(),
            std::path::PathBuf::from("/tmp/sched/config.json"),
            "scheduler config_file",
        ),
        (
            "include-files/config.json".to_string(),
            std::path::PathBuf::from("/tmp/payload/config.json"),
            "declarative",
        ),
    ];
    let err = dedupe_include_files(&input).unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("include_files conflict"),
        "diagnostic must mention 'include_files conflict': {msg}",
    );
    assert!(
        msg.contains("/tmp/sched/config.json") && msg.contains("/tmp/payload/config.json"),
        "diagnostic must name both host paths: {msg}",
    );
    assert!(
        msg.contains("origin: scheduler config_file") && msg.contains("origin: declarative"),
        "diagnostic must name both origin labels: {msg}",
    );
}

/// Multiple distinct archive_paths pass through unchanged. Verifies
/// the aggregator doesn't accidentally collapse orthogonal entries
/// (e.g. dropping by coincidental prefix or path-component equality).
#[test]
fn dedupe_include_files_preserves_distinct_entries() {
    let input = vec![
        (
            "include-files/a".to_string(),
            std::path::PathBuf::from("/usr/bin/a"),
            "declarative",
        ),
        (
            "include-files/b".to_string(),
            std::path::PathBuf::from("/usr/bin/b"),
            "declarative",
        ),
        (
            "include-files/c".to_string(),
            std::path::PathBuf::from("/usr/bin/c"),
            "scheduler config_file",
        ),
    ];
    let out = dedupe_include_files(&input).unwrap();
    assert_eq!(out.len(), 3, "three distinct entries must survive");
    let archives: Vec<&str> = out.iter().map(|(a, _)| a.as_str()).collect();
    assert!(archives.contains(&"include-files/a"));
    assert!(archives.contains(&"include-files/b"));
    assert!(archives.contains(&"include-files/c"));
}

// -- resolve_test_kernel tests --

#[test]
fn resolve_test_kernel_with_env_var() {
    let _lock = lock_env();
    let exe = crate::resolve_current_exe().unwrap();
    let _env = EnvVarGuard::set(crate::KTSTR_TEST_KERNEL_ENV, &exe);
    let result = resolve_test_kernel();
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), exe);
}

#[test]
fn resolve_test_kernel_with_nonexistent_env_path() {
    let _lock = lock_env();
    let _env = EnvVarGuard::set(crate::KTSTR_TEST_KERNEL_ENV, "/nonexistent/kernel/path");
    let result = resolve_test_kernel();
    let err = match result {
        Err(e) => e,
        Ok(p) => panic!("expected nonexistent env path to fail, got {p:?}"),
    };
    // A pointed-at path that doesn't exist is an OPERATOR
    // mistake, not a "harness not configured" condition. The
    // macro must NOT swallow this as a SKIP — surface it as
    // a regular anyhow error so the panic arm catches the typo.
    assert!(
        !crate::test_support::is_kernel_unavailable(&err),
        "KTSTR_TEST_KERNEL pointing at a missing path must NOT downcast \
             to KernelUnavailable (operator typo, not harness-misconfigured); \
             got: {err:#}",
    );
}

/// `resolve_test_kernel` must surface `KernelUnavailable` (the
/// typed marker the macro skips on) when neither
/// `KTSTR_TEST_KERNEL` nor any standard cache / sysroot location
/// produced a kernel. Pins the contract that a bare
/// `cargo nextest run` invocation skips cleanly instead of
/// panicking with "no kernel found".
#[test]
fn resolve_test_kernel_no_sources_returns_kernel_unavailable() {
    let _lock = lock_env();
    // Clear every candidate env var the discovery cascade reads
    // so the standard-locations branch can't see anything.
    let _e1 = EnvVarGuard::remove(crate::KTSTR_TEST_KERNEL_ENV);
    let _e2 = EnvVarGuard::remove(crate::KTSTR_KERNEL_ENV);
    let _e3 = EnvVarGuard::remove(crate::KTSTR_KERNEL_LIST_ENV);
    // `find_kernel()` may still resolve from /lib/modules or a
    // local cache on the host. The test environment isn't
    // guaranteed to be empty, so we accept either outcome:
    // an Ok (host has a kernel cached / installed) or an Err
    // that downcasts to `KernelUnavailable`. The negative
    // assertion (Err but not KernelUnavailable) is the
    // contract violation we're guarding against.
    match resolve_test_kernel() {
        Ok(_) => {
            // Host environment provides a kernel — the negative
            // branch we care about can't be exercised here.
            // Skipping the assertion is correct; the unit test
            // for `KernelUnavailable`'s Display below covers
            // the type contract regardless of host state.
        }
        Err(e) => {
            assert!(
                crate::test_support::is_kernel_unavailable(&e),
                "every Err from resolve_test_kernel after env-clearing must \
                     downcast to KernelUnavailable; got: {e:#}",
            );
        }
    }
}

/// `KernelUnavailable::Display` must surface the wrapped
/// diagnostic verbatim — the macro's SKIP banner relays it via
/// `{e:#}`, and a missing or mangled rendering would make the
/// "harness not configured" message unparseable.
#[test]
fn kernel_unavailable_display_renders_diagnostic() {
    let err = KernelUnavailable {
        diagnostic: "test fixture diagnostic".to_string(),
    };
    assert_eq!(format!("{err}"), "test fixture diagnostic");
}

// -- KVM check --

#[test]
fn kvm_accessible_on_test_host() {
    // Checks that /dev/kvm is accessible with read+write permissions.
    ensure_kvm().expect("/dev/kvm not accessible");
}

// -- SIGRTMIN save/restore pin --
//
// [`CpuStateGuard`] (above) saves the calling thread's SIGRTMIN
// sigaction before the VM runs and restores it on the way out
// — on every host arch, since ktstr's VMM installs a SIGRTMIN
// stop-vcpu handler (register_vcpu_signal_handler) on all targets.
// Without an explicit save/restore the handler leaks back into
// the test runner's main loop and a subsequent SIGRTMIN
// delivery (e.g. from a tokio timer wheel that uses realtime
// signals on some libc builds) would jump into the KVM
// stop-vcpu trampoline rather than the runner's own
// disposition.
//
// The pattern is:
//   1. zero a `libc::sigaction` and call `sigaction(SIGRTMIN(),
//      null, &mut saved)` to read the current disposition;
//   2. … run code that mutates SIGRTMIN …;
//   3. call `sigaction(SIGRTMIN(), &saved, null_mut())` to
//      restore the saved disposition.
//
// This test simulates one save/install/restore cycle on a
// dummy custom handler, asserting that the saved sigaction
// round-trips byte-for-byte after the install + restore.

/// Dummy signal handler used only as a probe value — never
/// invoked, only its function-pointer identity matters. Marked
/// `extern "C"` to match the libc handler ABI.
extern "C" fn sigrtmin_handler_probe(_sig: libc::c_int) {}

/// Serializes SIGRTMIN-touching tests against each other.
/// Signal dispositions are process-wide, so two tests installing
/// custom handlers concurrently could see each other's writes
/// between the install and assert steps. Using a dedicated
/// mutex (rather than `lock_env`) lets env-mutation tests run
/// concurrently with this one — the only conflict is between
/// signal-touching tests.
static SIGRTMIN_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// [`CpuStateGuard`] saves and restores the SIGRTMIN sigaction
/// across the VM run. The save/restore relies on the libc
/// semantic that `sigaction(sig, NULL, &mut out)` reads the
/// current disposition, and `sigaction(sig, &saved, NULL)`
/// rewrites it. Pin that round-trip on a dedicated fixture:
/// read pre-existing sigaction, install a probe handler, verify
/// the install landed (sa_sigaction matches the probe address),
/// restore from the saved sigaction, verify the restored
/// disposition matches what was originally saved.
///
/// Without this pin, a regression in [`CpuStateGuard`] that flips
/// the save and restore arguments (passing `&saved` as the OUT
/// parameter would zero the saved struct) or swaps `null` and
/// `null_mut` (the second arg type-checks either way under
/// cast-pointer semantics) would silently leave the kvm
/// stop-vcpu handler in place after VM teardown.
#[test]
fn sigrtmin_save_install_restore_roundtrip() {
    let _serial = SIGRTMIN_TEST_LOCK.lock_unpoisoned();

    // Step 1: save current SIGRTMIN sigaction.
    let mut saved: libc::sigaction = unsafe { std::mem::zeroed() };
    // SAFETY: passing NULL as the new disposition only reads
    // the current state into `saved`. Per `man sigaction(2)`,
    // a NULL `act` pointer leaves the disposition unchanged.
    let rc = unsafe { libc::sigaction(libc::SIGRTMIN(), std::ptr::null(), &mut saved as *mut _) };
    assert_eq!(
        rc,
        0,
        "sigaction(SIGRTMIN, NULL, &mut saved) must succeed; \
             got rc={rc}, errno={}",
        std::io::Error::last_os_error()
    );

    // Step 2: install a probe handler. Use a dedicated
    // `extern "C"` function so the address is stable and
    // distinguishable from SIG_DFL (0) and SIG_IGN (1).
    let mut probe: libc::sigaction = unsafe { std::mem::zeroed() };
    probe.sa_sigaction = sigrtmin_handler_probe as *const () as usize;
    unsafe {
        libc::sigemptyset(&mut probe.sa_mask);
    }
    // SAFETY: probe is fully initialized; we are writing the
    // disposition for SIGRTMIN with no out-pointer.
    let rc = unsafe { libc::sigaction(libc::SIGRTMIN(), &probe, std::ptr::null_mut()) };
    assert_eq!(rc, 0, "install probe handler for SIGRTMIN must succeed");

    // Verify the install landed by reading the disposition
    // back. `sigaction(SIGRTMIN, NULL, &mut current)` matches
    // the read form used in the `CpuStateGuard` construction.
    let mut current: libc::sigaction = unsafe { std::mem::zeroed() };
    unsafe {
        libc::sigaction(libc::SIGRTMIN(), std::ptr::null(), &mut current as *mut _);
    }
    let probe_addr = sigrtmin_handler_probe as *const () as usize;
    assert_eq!(
        current.sa_sigaction, probe_addr,
        "after install, sa_sigaction must point at \
             sigrtmin_handler_probe (0x{:x}); got 0x{:x} — the \
             install path is broken",
        probe_addr, current.sa_sigaction
    );

    // Step 3: restore from the saved sigaction. Mirrors the
    // restore in `CpuStateGuard::drop`:
    // `sigaction(SIGRTMIN, &saved_action, null_mut())`.
    // SAFETY: `saved` was populated in step 1 and is byte-
    // valid as a `libc::sigaction`.
    let rc = unsafe { libc::sigaction(libc::SIGRTMIN(), &saved, std::ptr::null_mut()) };
    assert_eq!(rc, 0, "restore from saved sigaction must succeed");

    // Verify the restored disposition matches what was
    // originally saved — sa_sigaction (the disposition
    // pointer) is the load-bearing field. A regression that
    // wrote `&mut saved` as the IN parameter would have
    // zeroed `saved` in step 1 (turning it into SIG_DFL after
    // the libc copy), so the post-restore disposition would
    // be SIG_DFL rather than the pre-test state.
    let mut after: libc::sigaction = unsafe { std::mem::zeroed() };
    unsafe {
        libc::sigaction(libc::SIGRTMIN(), std::ptr::null(), &mut after as *mut _);
    }
    assert_eq!(
        after.sa_sigaction, saved.sa_sigaction,
        "after restore, sa_sigaction must match the saved \
             value — restore is broken or `saved` got clobbered \
             during install. saved=0x{:x}, after=0x{:x}",
        saved.sa_sigaction, after.sa_sigaction
    );
    // sa_flags also rides through the save/restore — pin it
    // so a regression that copied only sa_sigaction (not the
    // full struct) trips here.
    // Mask out SA_RESTORER (0x04000000) — glibc's sigaction
    // wrapper unconditionally sets it on every call, even when
    // restoring a disposition that had sa_flags=0. The flag is
    // a glibc implementation detail, not part of the signal
    // disposition the test is verifying.
    let mask = !0x04000000i32;
    assert_eq!(
        after.sa_flags & mask,
        saved.sa_flags & mask,
        "after restore, sa_flags must match the saved value \
             (ignoring SA_RESTORER)"
    );
}

// -- resolve_scheduler tests --

#[test]
fn resolve_scheduler_eevdf() {
    let (path, source) = resolve_scheduler(&SchedulerSpec::Eevdf).unwrap();
    assert!(path.is_none());
    assert_eq!(
        source,
        ResolveSource::NotFound,
        "Eevdf has no user-space binary — source must be NotFound",
    );
}

#[test]
fn resolve_scheduler_kernel_builtin_is_not_found() {
    let (path, source) = resolve_scheduler(&SchedulerSpec::KernelBuiltin {
        enable: &[],
        disable: &[],
    })
    .unwrap();
    assert!(path.is_none());
    assert_eq!(
        source,
        ResolveSource::NotFound,
        "KernelBuiltin has no user-space binary — source must be NotFound",
    );
}

#[test]
fn resolve_scheduler_path_exists() {
    let exe = crate::resolve_current_exe().unwrap();
    let (path, source) = resolve_scheduler(&SchedulerSpec::Path(Box::leak(
        exe.to_str().unwrap().to_string().into_boxed_str(),
    )))
    .unwrap();
    assert!(path.is_some());
    assert_eq!(
        source,
        ResolveSource::Path,
        "explicit SchedulerSpec::Path(_) is tagged Path",
    );
}

#[test]
fn resolve_scheduler_path_missing() {
    let result = resolve_scheduler(&SchedulerSpec::Path("/nonexistent/scheduler"));
    assert!(result.is_err());
}

#[test]
fn resolve_scheduler_discover_missing() {
    let _lock = lock_env();
    let _env = EnvVarGuard::remove(crate::KTSTR_SCHEDULER_ENV);
    let result = resolve_scheduler(&SchedulerSpec::Discover("__nonexistent_scheduler_xyz__"));
    assert!(result.is_err());
}

/// Core fix: in the orchestrated (non-cargo-test) Discover path a
/// failed `cargo build -p <name>` REFUSES — it returns the build error
/// rather than silently falling through to a possibly-stale pre-built
/// binary. Pinned at the seam: a non-package name makes
/// `build_and_find_binary` error, and with the opt-out unset
/// `resolve_scheduler` returns that error wrapped with the refusal
/// context, proving it returned BEFORE the sibling/target-dir cascade.
#[test]
fn resolve_scheduler_discover_build_failure_refuses_stale_fallback() {
    let _lock = lock_env();
    let _no_global = EnvVarGuard::remove(crate::KTSTR_SCHEDULER_ENV);
    let _no_cargo_test_mode = EnvVarGuard::remove(crate::KTSTR_CARGO_TEST_MODE_ENV);
    let _no_optout = EnvVarGuard::remove(crate::KTSTR_SCHEDULER_ALLOW_STALE_FALLBACK_ENV);
    let result = resolve_scheduler(&SchedulerSpec::Discover("__nonexistent_scheduler_pkg__"));
    let rendered = format!(
        "{:#}",
        result.expect_err("a failed build must refuse, not serve a stale binary")
    );
    assert!(
        rendered.contains("refusing") && rendered.contains("KTSTR_SCHEDULER_ALLOW_STALE_FALLBACK"),
        "the refusal error must name the hazard + the opt-out; got: {rendered}"
    );
}

/// Escape hatch: with `KTSTR_SCHEDULER_ALLOW_STALE_FALLBACK` set a
/// failed orchestrated build no longer returns early — control falls
/// through to the sibling/target-dir cascade, which (for a bogus name
/// with no env/sibling/target hit) bails with the cascade-exhaustion
/// "not found" error, NOT the refusal. Pins that the opt-out is wired
/// and the legacy fallthrough still works.
#[test]
fn resolve_scheduler_discover_build_failure_env_optout_falls_through() {
    let _lock = lock_env();
    let _no_global = EnvVarGuard::remove(crate::KTSTR_SCHEDULER_ENV);
    let _no_cargo_test_mode = EnvVarGuard::remove(crate::KTSTR_CARGO_TEST_MODE_ENV);
    let _optout = EnvVarGuard::set(crate::KTSTR_SCHEDULER_ALLOW_STALE_FALLBACK_ENV, "1");
    let result = resolve_scheduler(&SchedulerSpec::Discover("__nonexistent_scheduler_pkg__"));
    let rendered = format!(
        "{:#}",
        result.expect_err("a bogus name still bails after the cascade")
    );
    assert!(
        rendered.contains("not found") && !rendered.contains("refusing"),
        "with the opt-out set the error must be the cascade-exhaustion bail, \
         not the refusal; got: {rendered}"
    );
}

/// Empty-string rejection: `KTSTR_SCHEDULER_ALLOW_STALE_FALLBACK=""`
/// (a stray CI pass-through) must NOT enable the stale fallback — the
/// presence-only contract treats empty as unset, so a failed build still
/// refuses. Mirrors `cargo_test_mode_active`'s empty-string rejection.
#[test]
fn resolve_scheduler_allow_stale_fallback_empty_string_rejected() {
    let _lock = lock_env();
    let _no_global = EnvVarGuard::remove(crate::KTSTR_SCHEDULER_ENV);
    let _no_cargo_test_mode = EnvVarGuard::remove(crate::KTSTR_CARGO_TEST_MODE_ENV);
    let _empty = EnvVarGuard::set(crate::KTSTR_SCHEDULER_ALLOW_STALE_FALLBACK_ENV, "");
    let result = resolve_scheduler(&SchedulerSpec::Discover("__nonexistent_scheduler_pkg__"));
    let rendered = format!("{:#}", result.expect_err("empty opt-out must still refuse"));
    assert!(
        rendered.contains("refusing"),
        "empty-string KTSTR_SCHEDULER_ALLOW_STALE_FALLBACK must NOT enable the \
         stale fallback (presence-only contract); got: {rendered}"
    );
}

#[test]
fn resolve_scheduler_discover_via_env() {
    let _lock = lock_env();
    let exe = crate::resolve_current_exe().unwrap();
    let _env = EnvVarGuard::set(crate::KTSTR_SCHEDULER_ENV, &exe);
    let (path, source) = resolve_scheduler(&SchedulerSpec::Discover("anything")).unwrap();
    assert_eq!(path.unwrap(), exe);
    assert_eq!(
        source,
        ResolveSource::EnvVar,
        "KTSTR_SCHEDULER hit must tag the result EnvVar",
    );
}

/// The per-name override `KTSTR_SCHEDULER_BIN_<NAME>` wins over the global
/// `KTSTR_SCHEDULER` so a test declaring distinct Discover
/// schedulers can point each at its own binary. Both vars point at
/// existing-but-different paths; the per-name one must be the resolved
/// result.
#[test]
fn resolve_scheduler_discover_per_name_env_wins_over_global() {
    let _lock = lock_env();
    let exe = crate::resolve_current_exe().unwrap();
    // Global points at a different existing binary; per-name points at
    // `exe`. The per-name override is checked first, so `exe` wins.
    let global_dir = TempDir::new().expect("tempdir");
    let global_bin = global_dir.path().join("global_sched");
    std::fs::write(&global_bin, b"stub").expect("write global stub");
    let _global = EnvVarGuard::set(crate::KTSTR_SCHEDULER_ENV, &global_bin);
    let _per = EnvVarGuard::set(crate::per_name_scheduler_env("scx_demo"), &exe);
    let (path, source) = resolve_scheduler(&SchedulerSpec::Discover("scx_demo")).unwrap();
    assert_eq!(
        path.unwrap(),
        exe,
        "per-name KTSTR_SCHEDULER_BIN_SCX_DEMO must win over the global KTSTR_SCHEDULER",
    );
    assert_eq!(source, ResolveSource::EnvVar);
}

/// A per-name override pointing at a NON-existent path falls through
/// to the global var (the documented lenient behavior, matching the
/// global var's own missing-path handling).
#[test]
fn resolve_scheduler_discover_per_name_missing_falls_back_to_global() {
    let _lock = lock_env();
    let exe = crate::resolve_current_exe().unwrap();
    let _per = EnvVarGuard::set(
        crate::per_name_scheduler_env("scx_demo"),
        "/nonexistent/per_name_scheduler",
    );
    let _global = EnvVarGuard::set(crate::KTSTR_SCHEDULER_ENV, &exe);
    let (path, source) = resolve_scheduler(&SchedulerSpec::Discover("scx_demo")).unwrap();
    assert_eq!(
        path.unwrap(),
        exe,
        "a set-but-missing per-name path must fall through to the global var",
    );
    assert_eq!(source, ResolveSource::EnvVar);
}

/// Two DISTINCT Discover names with two distinct per-name overrides
/// resolve to two distinct binaries — the regression the per-name override fixes (the
/// global var collapsed every Discover scheduler to one path).
#[test]
fn resolve_scheduler_discover_per_name_distinguishes_two_schedulers() {
    let _lock = lock_env();
    let _no_global = EnvVarGuard::remove(crate::KTSTR_SCHEDULER_ENV);
    let dir = TempDir::new().expect("tempdir");
    let bin_a = dir.path().join("sched_a");
    let bin_b = dir.path().join("sched_b");
    std::fs::write(&bin_a, b"a").expect("write a");
    std::fs::write(&bin_b, b"b").expect("write b");
    let _a = EnvVarGuard::set(crate::per_name_scheduler_env("scx_a"), &bin_a);
    let _b = EnvVarGuard::set(crate::per_name_scheduler_env("scx_b"), &bin_b);
    let (pa, _) = resolve_scheduler(&SchedulerSpec::Discover("scx_a")).unwrap();
    let (pb, _) = resolve_scheduler(&SchedulerSpec::Discover("scx_b")).unwrap();
    assert_eq!(pa.unwrap(), bin_a);
    assert_eq!(pb.unwrap(), bin_b);
}

/// The per-name env var name derivation: `KTSTR_SCHEDULER_BIN_` +
/// uppercase + non-alphanumeric to `_`.
#[test]
fn per_name_scheduler_env_derivation() {
    assert_eq!(
        crate::per_name_scheduler_env("scx_layered"),
        "KTSTR_SCHEDULER_BIN_SCX_LAYERED",
    );
    assert_eq!(
        crate::per_name_scheduler_env("scx-ktstr"),
        "KTSTR_SCHEDULER_BIN_SCX_KTSTR",
    );
}

/// The profile-aware target-dir probe order: with
/// `prefer_release` the release dir is probed first (so a scheduler on
/// the release default — or `--profile release` — prefers
/// `target/release/` over a stale `target/debug/` binary); otherwise
/// debug-first (`--profile dev`). Pins the reorder branch without
/// staging a CWD.
#[test]
fn target_dir_probe_order_prefers_profile_match() {
    let rel = super::scheduler::target_dir_probe_order(true);
    assert_eq!(rel[0], ("target/release", ResolveSource::TargetRelease));
    assert_eq!(rel[1], ("target/debug", ResolveSource::TargetDebug));
    let dbg = super::scheduler::target_dir_probe_order(false);
    assert_eq!(dbg[0], ("target/debug", ResolveSource::TargetDebug));
    assert_eq!(dbg[1], ("target/release", ResolveSource::TargetRelease));
}

/// `ResolveSource::as_str` is the load-bearing enum -> persisted-tag
/// bridge: it produces the snake_case string stamped into
/// `SidecarResult::resolve_source` (eval/mod.rs stamps it post-run) and
/// surfaced by `stats list-values` (resolve_source column). The
/// roundtrip / propagation tests pin
/// the String values INDEPENDENTLY (they hardcode the expected tags), so
/// a typo here (e.g. `SiblingDir => "sibling-dir"`) would silently ship
/// a malformed JSON contract without failing those tests. Pin every
/// variant -> tag mapping; the exhaustive match in `as_str` (no wildcard)
/// already forces a new variant to add an arm, and this fixes the string.
#[test]
fn resolve_source_as_str_tags() {
    assert_eq!(ResolveSource::Path.as_str(), "path");
    assert_eq!(ResolveSource::EnvVar.as_str(), "env_var");
    assert_eq!(ResolveSource::PathLookup.as_str(), "path_lookup");
    assert_eq!(ResolveSource::SiblingDir.as_str(), "sibling_dir");
    assert_eq!(ResolveSource::TargetDebug.as_str(), "target_debug");
    assert_eq!(ResolveSource::TargetRelease.as_str(), "target_release");
    assert_eq!(ResolveSource::AutoBuilt.as_str(), "auto_built");
    assert_eq!(ResolveSource::NotFound.as_str(), "not_found");
}

/// The `BIN` infix keeps the per-name namespace disjoint from the
/// `KTSTR_SCHEDULER_*` meta-variables: a scheduler named "profile" must NOT
/// derive the build-profile selector `KTSTR_SCHEDULER_PROFILE`
/// (namespace-collision guard).
#[test]
fn per_name_scheduler_env_does_not_collide_with_meta_vars() {
    assert_ne!(
        crate::per_name_scheduler_env("profile"),
        crate::KTSTR_SCHEDULER_PROFILE_ENV,
        "Discover(\"profile\") must not shadow KTSTR_SCHEDULER_PROFILE",
    );
    assert_ne!(
        crate::per_name_scheduler_env("anything"),
        crate::KTSTR_SCHEDULER_ENV,
        "per-name vars must not collide with the global KTSTR_SCHEDULER",
    );
}

/// `KTSTR_CARGO_TEST_MODE=1` enables the `$PATH` lookup branch of
/// `Discover`. Stage a tempdir containing an executable with the
/// requested name, point `PATH` at it, and verify the resolution
/// tags the result `ResolveSource::PathLookup`. Pins the
/// cargo-test-mode contract: a user can install scx_layered on
/// PATH and run their test without driving the cargo-ktstr
/// build pipeline.
#[test]
fn resolve_scheduler_discover_path_lookup_under_cargo_test_mode() {
    use std::os::unix::fs::PermissionsExt;
    let _lock = lock_env();
    let _no_env = EnvVarGuard::remove(crate::KTSTR_SCHEDULER_ENV);
    let _cargo = EnvVarGuard::set(crate::KTSTR_CARGO_TEST_MODE_ENV, "1");
    let dir = TempDir::new().expect("tempdir");
    let bin_path = dir.path().join("__test_path_scheduler__");
    std::fs::write(&bin_path, b"#!/bin/sh\nexit 0\n").expect("write stub");
    let mut perms = std::fs::metadata(&bin_path).unwrap().permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(&bin_path, perms).expect("chmod 0755");
    let _path_env = EnvVarGuard::set("PATH", dir.path());
    let (path, source) =
        resolve_scheduler(&SchedulerSpec::Discover("__test_path_scheduler__")).unwrap();
    assert_eq!(path.expect("found on PATH"), bin_path);
    assert_eq!(
        source,
        ResolveSource::PathLookup,
        "PATH-lookup hit must tag the result PathLookup",
    );
}

/// Without `KTSTR_CARGO_TEST_MODE`, the `$PATH` lookup branch is
/// inert: the orchestrated path runs the workspace build
/// FIRST, so a bogus package name errors there (the build fails and
/// is refused, `KTSTR_SCHEDULER_ALLOW_STALE_FALLBACK` unset) before
/// the `$PATH` / sibling / target-dir branches are ever consulted.
/// Pins the production-path contract: gauntlet runs land on the
/// workspace-built scheduler revision, never a system-wide install.
/// The test stages a stub on PATH but expects the resolution to NOT
/// pick it up (never report `PathLookup`) — it errors instead.
#[test]
fn resolve_scheduler_discover_path_lookup_inert_without_cargo_test_mode() {
    use std::os::unix::fs::PermissionsExt;
    let _lock = lock_env();
    let _no_env = EnvVarGuard::remove(crate::KTSTR_SCHEDULER_ENV);
    let _cargo = EnvVarGuard::remove(crate::KTSTR_CARGO_TEST_MODE_ENV);
    let dir = TempDir::new().expect("tempdir");
    let bin_path = dir.path().join("__test_inert_path_scheduler__");
    std::fs::write(&bin_path, b"#!/bin/sh\nexit 0\n").expect("write stub");
    let mut perms = std::fs::metadata(&bin_path).unwrap().permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(&bin_path, perms).expect("chmod 0755");
    let _path_env = EnvVarGuard::set("PATH", dir.path());
    // Without the cargo-test-mode flag the orchestrated
    // workspace build runs first; `cargo build -p
    // __test_inert_path_scheduler__` fails (bogus package) and is
    // refused (opt-out unset), so the call errors at the build step — before
    // the $PATH / sibling / target-dir branches — never PathLookup.
    let result = resolve_scheduler(&SchedulerSpec::Discover("__test_inert_path_scheduler__"));
    match result {
        Ok((_, source)) => {
            panic!("PATH lookup must be inert without KTSTR_CARGO_TEST_MODE; got source {source:?}",)
        }
        Err(_) => {
            // Expected: the workspace build of the bogus package failed
            // and was refused, so the $PATH lookup never ran. The
            // staged stub on PATH is deliberately ignored.
        }
    }
}

// -- scheduler_label tests --

#[test]
fn scheduler_label_eevdf_empty() {
    assert_eq!(reporting::scheduler_label(&SchedulerSpec::Eevdf), "");
}

#[test]
fn scheduler_label_discover() {
    assert_eq!(
        reporting::scheduler_label(&SchedulerSpec::Discover("scx_mitosis")),
        " [sched=scx_mitosis]"
    );
}

#[test]
fn scheduler_label_path() {
    assert_eq!(
        reporting::scheduler_label(&SchedulerSpec::Path("/usr/bin/sched")),
        " [sched=/usr/bin/sched]"
    );
}

// -- overcommit_skip wiring tests --
//
// Pin the auto-skip WIRING extracted from run_ktstr_test_inner_impl: a
// severe auto-collapse must yield a SKIP AssertResult (never a
// hard-fail), while the CI ~1.3x ratio and an explicit cpu_budget must
// RUN (None) so wide-SMP boot is validated on a typical runner rather
// than silently skipped. The pure predicate
// runtime::overcommit_skip_reason is unit-tested separately in
// runtime.rs; these pin that Some(reason) -> Ok(AssertResult::skip) and
// None -> proceed-to-boot at the eval call site.

fn wide_smp_topology() -> crate::vmm::topology::Topology {
    // 16 LLCs × 16 cores × 1 thread = 256 vCPUs (the wide-SMP class).
    crate::vmm::topology::Topology {
        llcs: 16,
        cores_per_llc: 16,
        threads_per_core: 1,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    }
}

#[test]
fn overcommit_skip_returns_skip_for_severe_auto_collapse() {
    let mut entry = eevdf_entry("overcommit_skip_severe");
    entry.topology = wide_smp_topology();
    entry.cpu_budget = None;
    // 256 vCPUs auto-collapse onto 8 host CPUs = 32x ≥ 6x → SKIP, not a
    // hard-fail.
    let skip = super::overcommit_skip(&entry, &[0, 1, 2, 3, 4, 5, 6, 7], None)
        .expect("32x auto-collapse must auto-skip");
    assert!(skip.is_skip(), "severe overcommit must yield a SKIP result");
}

#[test]
fn overcommit_skip_runs_at_ci_wide_smp_ratio() {
    let mut entry = eevdf_entry("overcommit_skip_ci");
    entry.topology = wide_smp_topology();
    entry.cpu_budget = None;
    // 256 vCPUs on a 192-CPU CI runner = 1.33x < 6x → RUN (None), so
    // wide-SMP boot is validated on CI, never masked.
    let host: Vec<usize> = (0..192).collect();
    assert!(
        super::overcommit_skip(&entry, &host, None).is_none(),
        "wide-SMP boot must RUN at the CI ~1.3x ratio",
    );
}

#[test]
fn overcommit_skip_never_skips_explicit_budget() {
    let mut entry = eevdf_entry("overcommit_skip_explicit");
    entry.topology = wide_smp_topology();
    entry.cpu_budget = Some(4); // deliberate oversubscription opt-in
    // Even 256 vCPUs on 8 host CPUs: an explicit cpu_budget is a
    // contention-testing opt-in and must never auto-skip.
    assert!(
        super::overcommit_skip(&entry, &[0, 1, 2, 3, 4, 5, 6, 7], None).is_none(),
        "explicit cpu_budget must never auto-skip",
    );
}

#[test]
fn overcommit_skip_uses_stricter_cap_for_expect_auto_repro() {
    // Pin the expect_auto_repro field-threading at the eval call site
    // (entry.expect_auto_repro -> overcommit_skip_reason's stricter
    // EXPECT_AUTO_REPRO_SKIP_RATIO=2.0 cap). The SAME 256-vCPU-on-96-host
    // ratio (2.67x) that a boot-only wide test RUNS at must SKIP when
    // expect_auto_repro is set — a regression that hardcoded `false` or
    // dropped the arg would still pass the false-default tests above but
    // fail here.
    let host: Vec<usize> = (0..96).collect(); // 256 / 96 = 2.67x

    let mut boot_only = eevdf_entry("overcommit_skip_boot_only");
    boot_only.topology = wide_smp_topology();
    boot_only.cpu_budget = None;
    boot_only.expect_auto_repro = false;
    assert!(
        super::overcommit_skip(&boot_only, &host, None).is_none(),
        "boot-only wide test must RUN at 2.67x (< 6x boot cap)",
    );

    let mut auto_repro = eevdf_entry("overcommit_skip_auto_repro");
    auto_repro.topology = wide_smp_topology();
    auto_repro.cpu_budget = None;
    auto_repro.expect_auto_repro = true;
    let skip = super::overcommit_skip(&auto_repro, &host, None)
        .expect("expect_auto_repro chain must auto-skip at 2.67x (>= 2.0x cap)");
    assert!(
        skip.is_skip(),
        "expect_auto_repro overcommit must yield a SKIP result",
    );
}

#[test]
fn overcommit_skip_keys_on_booted_not_declared_topology() {
    // Under a gauntlet/CLI TopoOverride the VM boots the OVERRIDE
    // topology, not entry.topology, so the skip decision must model the
    // BOOTED oversubscription. Every other overcommit_skip test passes
    // topo=None (declared == booted) and so never exercises the
    // divergence; a regression that keys on the declared count would flip
    // the verdict in BOTH directions below.
    let host: Vec<usize> = (0..8).collect(); // 8 host CPUs

    // Small declared (DEFAULT = 2 vCPUs) dispatched onto a large preset
    // (14×9×2 = 252 booted vCPUs): declared 2/8 = 1.0x would RUN, but the
    // booted 252/8 = 31.5x >> 6.0x cap MUST auto-skip.
    let mut small = eevdf_entry("overcommit_skip_booted_small_declared");
    small.cpu_budget = None;
    let big_preset = crate::test_support::topo::TopoOverride {
        numa_nodes: 1,
        llcs: 14,
        cores: 9,
        threads: 2,
        memory_mib: 512,
    };
    let skip = super::overcommit_skip(&small, &host, Some(&big_preset))
        .expect("small declared on a large preset must skip on the BOOTED 31.5x oversubscription");
    assert!(
        skip.is_skip(),
        "booted-oversubscription overcommit must yield a SKIP"
    );

    // Inverse: large declared (256 vCPUs) dispatched onto a tiny preset
    // (1×2×1 = 2 booted vCPUs): declared 256/8 = 32x would SKIP, but the
    // booted 2/8 = 1.0x fits and MUST run.
    let mut big = eevdf_entry("overcommit_skip_booted_big_declared");
    big.topology = wide_smp_topology(); // 256 declared
    big.cpu_budget = None;
    let tiny_preset = crate::test_support::topo::TopoOverride {
        numa_nodes: 1,
        llcs: 1,
        cores: 2,
        threads: 1,
        memory_mib: 512,
    };
    assert!(
        super::overcommit_skip(&big, &host, Some(&tiny_preset)).is_none(),
        "large declared on a tiny preset must RUN (the booted topology fits the host)",
    );
}
