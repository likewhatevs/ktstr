//! End-to-end: `cargo ktstr shell --test <fixture> --exec "<payload>"`
//! exercises the shell-test descriptor's scheduler enable/disable
//! cmd lifecycle.
//!
//! ## What's pinned
//!
//! The shell-test descriptor (populated at
//! `src/test_support/dispatch.rs::maybe_dispatch_shell_test`) plumbs
//! `SchedulerSpec::KernelBuiltin { enable, disable }` strings
//! through to a VM that:
//!
//!   1. packs `enable` cmds into `/sched_enable` and `disable` cmds
//!      into `/sched_disable` in the initramfs
//!      (`src/vmm/initramfs.rs::pack_sched_scripts`),
//!   2. runs `/sched_enable` in shell-mode at
//!      `ktstr_guest_init`'s `exec_shell_script("/sched_enable")`
//!      call BEFORE the `busybox sh -c` payload,
//!   3. runs `/sched_disable` AFTER the payload, BEFORE
//!      `force_reboot()`.
//!
//! These tests pin the ENABLE half of that lifecycle. DISABLE-side
//! observability is gated on a follow-up symmetric drain fix in
//! `ktstr_guest_init` (force_reboot is bare `RB_AUTOBOOT` with no
//! userspace drain — disable's writes after the pre-disable drain
//! race the reboot tearing down the virtio TX ring).
//!
//! Line numbers in citation comments below are accurate as of the
//! commit landing this file; symbol-level cites (`exec_shell_script`,
//! `redirect_all_stdio_to`, `shell_console_device`, `force_reboot`)
//! are the stable references and should be used for navigation
//! when the line numbers drift.
//!
//! ## Marker sink: `/proc/1/fd/1`
//!
//! Markers route through init's fd 1 (whichever char device
//! `shell_console_device()` in `src/vmm/rust_init/modes.rs` returns —
//! `/dev/hvc0` when virtio-console is available, else `/dev/ttyS1`).
//! The `cargo-ktstr` subprocess captures that pipeline as
//! `Output.stdout`.
//!
//! `/proc/1/fd/1` is a procfs symlink to init's fd 1; since init
//! is PID 1 and the dup2 in `redirect_all_stdio_to` runs in-process
//! BEFORE `/sched_enable` executes, an `fs::write` to the symlink
//! resolves to whichever char device init's stdio was dup2'd to —
//! drain-parity guaranteed across virtio-console vs COM2 boot
//! environments, with no fallback path needed in the test fixture.
//!
//! ## Fixture pattern
//!
//! Each `shell_lifecycle_fixture_*` `#[ktstr_test]` exists ONLY as
//! the topology + scheduler source for the matching subprocess test.
//! Fixtures use `ignore = "fixture for shell-mode probe; see tests
//! in this file"` to skip them in regular `cargo nextest` runs
//! while remaining discoverable by `cargo ktstr shell --test <name>`:
//! the shell-mode lookup in
//! `src/test_support/dispatch.rs::maybe_dispatch_shell_test` only
//! filters on `host_only`, never on `is_ignored`.
//!
//! Fixture disable cmds in this file write only to `/proc/1/fd/1`
//! (a benign console echo, never a kernel-state sysctl); the
//! `force_reboot` that immediately follows disable does not depend
//! on kernel state being clean. A future fixture that exercises a
//! real sched-knob via disable would need to verify the post-disable
//! kernel state survives `RB_AUTOBOOT` cleanly.
//!
//! ## Marker shape
//!
//! `exec_shell_line` in `src/vmm/rust_init/dump.rs` ONLY supports literal
//! `echo VALUE > /path`. No subshell expansion, no append `>>`, no
//! other commands. Markers are unique sentinels
//! (`SHELL_LIFECYCLE_ENABLE_MARKER`, etc.) — the "missing initramfs
//! pack silently no-ops the hook" risk is caught by exact-count
//! `assert_eq!(matches.count(), 1)`. The prefix
//! `SHELL_LIFECYCLE_` is fixture-private and is chosen to be
//! unlikely to collide with cargo-ktstr's host-side banner output
//! (which uses words like `scheduler`, `enable`, `disable`, `cmd(s)`
//! but never the `SHELL_LIFECYCLE_` prefix string).

#![cfg(unix)]

mod common;

use anyhow::Result;
use common::cargo_ktstr_subprocess::{
    combined_output, run_cargo_ktstr_shell, run_cargo_ktstr_shell_with_timeout,
};
use ktstr::assert::AssertResult;
use ktstr::ktstr_test;
use ktstr::prelude::{Scheduler, SchedulerSpec};
use ktstr::scenario::Ctx;

// ---- FIXTURES -----------------------------------------------------
//
// Each fixture declares a `SchedulerSpec::KernelBuiltin` scheduler
// with specific enable/disable cmd lists. Marker prefix
// `SHELL_LIFECYCLE_` is a unique fixture-private sentinel: a
// regression that silently dropped the cmd into a missing initramfs
// file would surface as 0 occurrences in subprocess output (the
// exact-count assertion catches it).
//
// Fixtures are `#[ktstr_test(..., ignore)]` so they don't VM-boot
// during regular `cargo nextest` runs. They remain discoverable
// by `cargo ktstr shell --test <name>` because the shell-mode
// dispatcher in
// `src/test_support/dispatch.rs::maybe_dispatch_shell_test` only
// filters on `host_only`, never on `is_ignored`.

const SHELL_LIFECYCLE_FIXTURE_BOTH: Scheduler = Scheduler::named("shell_lifecycle_fixture_both")
    .binary(SchedulerSpec::KernelBuiltin {
        enable: &[
            "echo SHELL_LIFECYCLE_ENABLE_MARKER > /proc/1/fd/1",
            "echo SHELL_LIFECYCLE_PRE_SENTINEL > /tmp/shell_lifecycle_pre_marker",
        ],
        disable: &["echo SHELL_LIFECYCLE_DISABLE_MARKER > /proc/1/fd/1"],
    });

const SHELL_LIFECYCLE_FIXTURE_ENABLE_ONLY: Scheduler =
    Scheduler::named("shell_lifecycle_fixture_enable_only").binary(SchedulerSpec::KernelBuiltin {
        enable: &["echo SHELL_LIFECYCLE_ENABLE_MARKER > /proc/1/fd/1"],
        disable: &[],
    });

const SHELL_LIFECYCLE_FIXTURE_DISABLE_ONLY: Scheduler =
    Scheduler::named("shell_lifecycle_fixture_disable_only").binary(SchedulerSpec::KernelBuiltin {
        enable: &[],
        disable: &["echo SHELL_LIFECYCLE_DISABLE_MARKER > /proc/1/fd/1"],
    });

const SHELL_LIFECYCLE_FIXTURE_PARTIAL_FAIL: Scheduler =
    Scheduler::named("shell_lifecycle_fixture_partial_fail").binary(SchedulerSpec::KernelBuiltin {
        enable: &[
            // First line fails (parent dir doesn't exist); per
            // `exec_shell_script` partial-apply contract in
            // `src/vmm/rust_init/dump.rs`, fail_count increments but
            // the loop continues to the next line.
            "echo bogus > /this/path/does/not/exist",
            "echo SHELL_LIFECYCLE_ENABLE_MARKER > /proc/1/fd/1",
        ],
        disable: &[],
    });

#[ktstr_test(
    scheduler = SHELL_LIFECYCLE_FIXTURE_BOTH,
    llcs = 1,
    cores = 2,
    threads = 1,
    memory_mib = 128,
    duration_s = 1,
    watchdog_timeout_s = 10,
    auto_repro = false,
    ignore,
)]
fn shell_lifecycle_fixture_both(_ctx: &Ctx) -> Result<AssertResult> {
    Ok(AssertResult::pass())
}

#[ktstr_test(
    scheduler = SHELL_LIFECYCLE_FIXTURE_ENABLE_ONLY,
    llcs = 1,
    cores = 2,
    threads = 1,
    memory_mib = 128,
    duration_s = 1,
    watchdog_timeout_s = 10,
    auto_repro = false,
    ignore,
)]
fn shell_lifecycle_fixture_enable_only(_ctx: &Ctx) -> Result<AssertResult> {
    Ok(AssertResult::pass())
}

#[ktstr_test(
    scheduler = SHELL_LIFECYCLE_FIXTURE_DISABLE_ONLY,
    llcs = 1,
    cores = 2,
    threads = 1,
    memory_mib = 128,
    duration_s = 1,
    watchdog_timeout_s = 10,
    auto_repro = false,
    ignore,
)]
fn shell_lifecycle_fixture_disable_only(_ctx: &Ctx) -> Result<AssertResult> {
    Ok(AssertResult::pass())
}

#[ktstr_test(
    scheduler = SHELL_LIFECYCLE_FIXTURE_PARTIAL_FAIL,
    llcs = 1,
    cores = 2,
    threads = 1,
    memory_mib = 128,
    duration_s = 1,
    watchdog_timeout_s = 10,
    auto_repro = false,
    ignore,
)]
fn shell_lifecycle_fixture_partial_fail(_ctx: &Ctx) -> Result<AssertResult> {
    Ok(AssertResult::pass())
}

// ---- SUBPROCESS TESTS ---------------------------------------------

/// A `--exec` payload that never exits must be force-killed at
/// `--exec-timeout`, not block the BSP run loop forever. A panic-less
/// guest hang leaves the guest halted (the BSP `bsp.run()` blocks in
/// KVM_RUN, and flipping `kill` alone never unblocks it), so the
/// run_interactive watchdog kicks the vCPU (immediate_exit + SIGRTMIN)
/// at the deadline. `sleep 600` hangs well past the 8s timeout, so the
/// run must end in the DISTINCT exec-timeout error — not the generic
/// no-EXEC_EXIT-frame message, and not a wedge. 8s is comfortably past
/// the ~4s boot-to-payload so the kill lands during the hang.
#[test]
fn shell_exec_timeout_force_kills_hanging_payload() {
    let out = run_cargo_ktstr_shell_with_timeout("shell_lifecycle_fixture_both", "sleep 600", "8s");
    let combined = combined_output(&out);
    assert!(
        !out.status.success(),
        "a timed-out --exec must exit non-zero (the payload never \
         delivered an exit code); got success.\n{combined}",
    );
    assert!(
        combined.contains("exceeded") && combined.contains("exec-timeout"),
        "expected the distinct exec-timeout error (proves the watchdog \
         fired and the no-frame bail took the timed_out branch), \
         got:\n{combined}",
    );
}

/// Happy path: enable + disable both set, payload runs between
/// them. Pins the canonical lifecycle and the ENABLE-before-payload
/// ordering invariant via byte-offset comparison.
#[test]
fn shell_mode_enable_fires_before_payload() {
    let out = run_cargo_ktstr_shell(
        "shell_lifecycle_fixture_both",
        "echo SHELL_LIFECYCLE_PAYLOAD_MARKER",
    );
    let combined = combined_output(&out);
    assert!(
        out.status.success(),
        "cargo ktstr shell exited non-zero (exit={:?})\n{combined}",
        out.status.code(),
    );
    // Exact-count catches BOTH silent-no-op (regression dropped the
    // /sched_enable initramfs pack → 0 occurrences) AND double-fire
    // (refactor invoked enable twice → 2 occurrences). Either case
    // fails this assertion, which a `count > 0` check would miss
    // on the double-fire side.
    let enable_count = combined.matches("SHELL_LIFECYCLE_ENABLE_MARKER").count();
    assert_eq!(
        enable_count, 1,
        "SHELL_LIFECYCLE_ENABLE_MARKER count = {enable_count}; \
         expected exactly 1. Zero = /sched_enable never ran \
         (initramfs pack broken in `pack_sched_scripts` OR \
         shell-mode dispatch skipped `exec_shell_script` in \
         `ktstr_guest_init`'s shell branch). >1 = duplicate \
         invocation. Combined output:\n{combined}",
    );
    let payload_count = combined.matches("SHELL_LIFECYCLE_PAYLOAD_MARKER").count();
    assert_eq!(
        payload_count, 1,
        "SHELL_LIFECYCLE_PAYLOAD_MARKER count = {payload_count}; \
         expected exactly 1. The --exec payload should have echoed \
         it once. Combined:\n{combined}",
    );
    // Byte-offset ordering pin: enable's marker must precede the
    // payload's output. A regression that reordered
    // exec_shell_script("/sched_enable") and the busybox payload
    // exec — or that ran enable async/post-payload — would surface
    // as enable_pos > payload_pos here. Safe because the count
    // pins above guarantee exactly one occurrence of each: a
    // host-side banner leak that emitted a stray marker would
    // already have failed the count check.
    let enable_pos = combined
        .find("SHELL_LIFECYCLE_ENABLE_MARKER")
        .expect("enable marker located above");
    let payload_pos = combined
        .find("SHELL_LIFECYCLE_PAYLOAD_MARKER")
        .expect("payload marker located above");
    assert!(
        enable_pos < payload_pos,
        "SHELL_LIFECYCLE_ENABLE_MARKER at byte {enable_pos} must \
         precede SHELL_LIFECYCLE_PAYLOAD_MARKER at byte {payload_pos} \
         — proves /sched_enable ran BEFORE the busybox sh payload. \
         Combined output:\n{combined}",
    );
}

/// Counter-test for ordering: enable also writes a sentinel file
/// `/tmp/shell_lifecycle_pre_marker` BEFORE the payload runs. The
/// payload reads the file and emits `SHELL_LIFECYCLE_PAYLOAD_SAW_NO_PRE`
/// ONLY if the file is missing. A regression where enable runs
/// AFTER the payload would produce this string; the assertion's
/// absence-check catches it.
///
/// Distinct from `enable_fires_before_payload` because that test
/// only proves the ENABLE MARKER appears before the payload marker —
/// a regression that reordered the marker WRITES but kept the
/// /sched_enable file-system effects (sysfs writes, etc.) in
/// pre-payload order would still pass that test. The sentinel-file
/// check pins the SCRIPT-EFFECTS ordering: the file MUST exist
/// at the moment the payload runs, not just at the moment the
/// console capture observed the marker.
#[test]
fn shell_mode_enable_side_effects_visible_to_payload() {
    let out = run_cargo_ktstr_shell(
        "shell_lifecycle_fixture_both",
        "if test -f /tmp/shell_lifecycle_pre_marker; then \
         echo SHELL_LIFECYCLE_PAYLOAD_SAW_PRE; \
         else echo SHELL_LIFECYCLE_PAYLOAD_SAW_NO_PRE; fi",
    );
    let combined = combined_output(&out);
    assert!(
        out.status.success(),
        "cargo ktstr shell exited non-zero (exit={:?})\n{combined}",
        out.status.code(),
    );
    assert!(
        combined.contains("SHELL_LIFECYCLE_PAYLOAD_SAW_PRE"),
        "payload did not observe /tmp/shell_lifecycle_pre_marker — \
         /sched_enable's filesystem effects must be visible to the \
         payload that runs AFTER. Combined:\n{combined}",
    );
    assert!(
        !combined.contains("SHELL_LIFECYCLE_PAYLOAD_SAW_NO_PRE"),
        "payload reported SHELL_LIFECYCLE_PAYLOAD_SAW_NO_PRE — \
         /sched_enable's `echo > /tmp/shell_lifecycle_pre_marker` \
         did NOT propagate before payload ran. Either the script \
         never executed OR ordering inverted. Combined:\n{combined}",
    );
}

/// only-enable: descriptor sets enable cmds + EMPTY disable cmds.
/// Per `exec_shell_script` in `src/vmm/rust_init/dump.rs`, a missing
/// /sched_disable file is a legitimate "no script" debug-skip;
/// no error surfaces. Enable still fires normally. The
/// disable-marker absence assertion catches a regression that
/// silently injected a default disable script when none was
/// requested.
#[test]
fn shell_mode_only_enable_set_payload_still_runs() {
    let out = run_cargo_ktstr_shell(
        "shell_lifecycle_fixture_enable_only",
        "echo SHELL_LIFECYCLE_PAYLOAD_MARKER",
    );
    let combined = combined_output(&out);
    assert!(
        out.status.success(),
        "cargo ktstr shell exited non-zero (exit={:?})\n{combined}",
        out.status.code(),
    );
    assert_eq!(
        combined.matches("SHELL_LIFECYCLE_ENABLE_MARKER").count(),
        1,
        "SHELL_LIFECYCLE_ENABLE_MARKER must appear exactly once \
         even when disable cmds are empty. Combined:\n{combined}",
    );
    assert!(
        !combined.contains("SHELL_LIFECYCLE_DISABLE_MARKER"),
        "SHELL_LIFECYCLE_DISABLE_MARKER appeared despite the \
         fixture's empty disable cmds — a regression silently \
         injected a default disable script OR cross-test marker \
         contamination. Combined:\n{combined}",
    );
    assert!(
        combined.contains("SHELL_LIFECYCLE_PAYLOAD_MARKER"),
        "payload must still run when only enable cmds are set. \
         Combined:\n{combined}",
    );
}

/// only-disable: descriptor sets EMPTY enable cmds + disable cmds.
/// Symmetric to only-enable on the enable side: enable marker MUST
/// be absent (proves the empty enable list resulted in a no-op
/// /sched_enable, not a silently-injected default).
///
/// Also pins the positive disable-marker assertion: the disable
/// script's marker MUST appear exactly once in captured output.
/// The post-`/sched_disable` tcdrain in `ktstr_guest_init` is what
/// makes this assertion safe — without that drain, the disable
/// script's `echo > /proc/1/fd/1` write races `force_reboot`'s
/// device_shutdown and the host sees a truncated tail. A regression
/// that reverted the drain (or refactored ktstr_guest_init's
/// shell-mode disable path to lose the tcdrain) would surface here
/// as the disable marker missing or appearing zero times.
#[test]
fn shell_mode_only_disable_set_no_enable_marker() {
    let out = run_cargo_ktstr_shell(
        "shell_lifecycle_fixture_disable_only",
        "echo SHELL_LIFECYCLE_PAYLOAD_MARKER",
    );
    let combined = combined_output(&out);
    assert!(
        out.status.success(),
        "cargo ktstr shell exited non-zero (exit={:?})\n{combined}",
        out.status.code(),
    );
    assert!(
        !combined.contains("SHELL_LIFECYCLE_ENABLE_MARKER"),
        "SHELL_LIFECYCLE_ENABLE_MARKER appeared despite empty enable \
         cmds — either the descriptor leaked a default OR a sibling \
         fixture's marker bled through. Combined:\n{combined}",
    );
    assert!(
        combined.contains("SHELL_LIFECYCLE_PAYLOAD_MARKER"),
        "payload must run independent of enable/disable cmd \
         presence. Combined:\n{combined}",
    );
    // Positive disable-marker assertion: exact-count == 1 catches
    // both no-fire (the post-disable drain regressed and bytes were
    // truncated by force_reboot) AND double-fire (a refactor invoked
    // the disable script twice). The drain is what makes this
    // assertion non-flaky — pre-drain, this assertion would race
    // the reboot syscall on every run.
    assert_eq!(
        combined.matches("SHELL_LIFECYCLE_DISABLE_MARKER").count(),
        1,
        "SHELL_LIFECYCLE_DISABLE_MARKER expected exactly once in \
         captured output. Zero matches means the post-`/sched_disable` \
         tcdrain in ktstr_guest_init's shell-mode disable path \
         regressed; more than one match means a refactor invoked the \
         disable script multiple times. Combined:\n{combined}",
    );
}

/// Payload exits non-zero. Pins the exit-code-propagation contract
/// (subprocess Output.status surfaces the payload's exit code) AND
/// proves enable still fired regardless of the payload's later exit.
/// A regression where shell-mode aborted enable on detection that
/// the payload would fail (impossible by construction — enable runs
/// BEFORE the payload — but a future refactor could introduce a
/// pre-flight check) would fail this test.
#[test]
fn shell_mode_payload_nonzero_exit_propagates_enable_still_fires() {
    let out = run_cargo_ktstr_shell(
        "shell_lifecycle_fixture_both",
        "echo SHELL_LIFECYCLE_PAYLOAD_MARKER; exit 17",
    );
    let combined = combined_output(&out);
    // Per `ktstr_guest_init`'s shell-mode --exec branch, the
    // payload's exit code travels via the bulk data port
    // (`send_exec_exit`), which the host surfaces as the
    // cargo-ktstr subprocess's own exit code.
    assert_eq!(
        out.status.code(),
        Some(17),
        "subprocess exit code = {:?}; expected 17 (the payload's \
         own `exit 17`). The exit-code-propagation chain is \
         payload → send_exec_exit (bulk port) → cargo-ktstr host \
         → subprocess exit. Combined:\n{combined}",
        out.status.code(),
    );
    assert_eq!(
        combined.matches("SHELL_LIFECYCLE_ENABLE_MARKER").count(),
        1,
        "SHELL_LIFECYCLE_ENABLE_MARKER count = {}; expected 1 — \
         enable must fire BEFORE the payload regardless of the \
         payload's eventual exit code. Combined:\n{combined}",
        combined.matches("SHELL_LIFECYCLE_ENABLE_MARKER").count(),
    );
    assert!(
        combined.contains("SHELL_LIFECYCLE_PAYLOAD_MARKER"),
        "payload's pre-exit echo must surface in captured output \
         even on non-zero exit. Combined:\n{combined}",
    );
}

/// Companion to the non-zero case: a payload that exits 0 must
/// propagate as exit 0, NOT be masked into a spurious failure by the
/// missing-frame fail-loud path. Pins the success direction of the
/// exit-code-propagation contract — a regression that always reported
/// non-zero, or that tripped the no-ExecExit-frame bail on a valid
/// `ExecExit(0)` frame, would fail here.
#[test]
fn shell_mode_payload_zero_exit_propagates() {
    let out = run_cargo_ktstr_shell(
        "shell_lifecycle_fixture_both",
        "echo SHELL_LIFECYCLE_PAYLOAD_MARKER; exit 0",
    );
    let combined = combined_output(&out);
    assert_eq!(
        out.status.code(),
        Some(0),
        "subprocess exit code = {:?}; expected 0 (the payload's own \
         `exit 0`) — a CRC-valid ExecExit(0) frame must surface as 0, \
         not trip the missing-frame bail. Combined:\n{combined}",
        out.status.code(),
    );
    assert!(
        combined.contains("SHELL_LIFECYCLE_PAYLOAD_MARKER"),
        "payload's echo must surface in captured output. Combined:\n{combined}",
    );
}

/// Partial-apply: enable cmd list has one line that fails (writes
/// to a non-existent path) and one line that succeeds (writes the
/// marker). Per `exec_shell_script` in `src/vmm/rust_init/dump.rs`,
/// per-line failures increment fail_count but the script continues
/// to the next line. The marker MUST appear (proves the second
/// line ran despite the first failing) AND the partial-apply
/// summary tracing line MUST surface in stderr — pinned via
/// CO-OCCURRENCE of the `partial-apply` summary token AND the
/// failing path, so an unrelated tracing line emitting the word
/// `partial-apply` cannot satisfy the assertion alone.
///
/// `exec_shell_line` uses `fs::write` with no mkdir, so the
/// `/this/path/does/not/exist` write reliably fails with ENOENT
/// rather than silently creating the parent.
#[test]
fn shell_mode_enable_partial_apply_failure_continues_to_payload() {
    let out = run_cargo_ktstr_shell(
        "shell_lifecycle_fixture_partial_fail",
        "echo SHELL_LIFECYCLE_PAYLOAD_MARKER",
    );
    let combined = combined_output(&out);
    assert!(
        out.status.success(),
        "cargo ktstr shell exited non-zero (exit={:?}); the \
         partial-apply contract is that per-line failures are \
         counted but the script + payload continue. A non-zero \
         exit here means abort-on-first-failure regressed. \
         Combined:\n{combined}",
        out.status.code(),
    );
    assert_eq!(
        combined.matches("SHELL_LIFECYCLE_ENABLE_MARKER").count(),
        1,
        "SHELL_LIFECYCLE_ENABLE_MARKER count = {}; expected 1 — \
         the second enable cmd MUST execute despite the first \
         failing. A zero count means the script aborted on the \
         first failure, violating the partial-apply contract in \
         `exec_shell_script`. Combined:\n{combined}",
        combined.matches("SHELL_LIFECYCLE_ENABLE_MARKER").count(),
    );
    assert!(
        combined.contains("SHELL_LIFECYCLE_PAYLOAD_MARKER"),
        "payload must run even when /sched_enable had per-line \
         failures (the entire script returns silently per \
         `exec_shell_script`; the partial-apply summary is \
         informational). Combined:\n{combined}",
    );
    // Co-occurrence pin: `partial-apply` summary AND the failing
    // path must BOTH surface in captured output. The summary alone
    // would match an unrelated tracing line that happens to contain
    // "partial-apply"; pinning the path proves the summary came from
    // THIS test's intentional failure, not drift in some other site.
    assert!(
        combined.contains("partial-apply"),
        "expected `partial-apply` summary in captured output \
         (tracing::error in `exec_shell_script`) — a silent \
         partial-apply violates the no-silent-drops rule. \
         Combined:\n{combined}",
    );
    assert!(
        combined.contains("/this/path/does/not/exist"),
        "expected `/this/path/does/not/exist` in captured output \
         alongside the `partial-apply` summary — the failing path \
         must surface so the operator can diagnose which line(s) \
         dropped. Combined:\n{combined}",
    );
}

/// Full-bracket sanity: enable + payload + disable ALL fire and
/// appear in captured output, in that lexical order. Uses the
/// both-fixture (enable + disable cmds both set) + a payload that
/// echoes its own marker. Pins:
///
/// - all three markers appear exactly once (no-fire catches drain
///   regressions; double-fire catches refactor-double-invoke);
/// - the lexical ordering enable < payload < disable holds (catches
///   any refactor that moved /sched_disable before the payload OR
///   /sched_enable after it — both would invert the bracket
///   semantic without necessarily failing the per-marker pins).
///
/// Distinct from `shell_mode_only_disable_set_no_enable_marker`
/// which uses the disable-only fixture; this test exercises the
/// canonical "all three stages fire in order" shape that's the
/// expected production case.
#[test]
fn shell_mode_enable_and_disable_both_fire_with_payload() {
    let out = run_cargo_ktstr_shell(
        "shell_lifecycle_fixture_both",
        "echo SHELL_LIFECYCLE_PAYLOAD_MARKER",
    );
    let combined = combined_output(&out);
    assert!(
        out.status.success(),
        "cargo ktstr shell exited non-zero (exit={:?})\n{combined}",
        out.status.code(),
    );
    for marker in &[
        "SHELL_LIFECYCLE_ENABLE_MARKER",
        "SHELL_LIFECYCLE_PAYLOAD_MARKER",
        "SHELL_LIFECYCLE_DISABLE_MARKER",
    ] {
        assert_eq!(
            combined.matches(marker).count(),
            1,
            "{marker} expected exactly once in captured output. \
             Zero matches means the corresponding stage failed to \
             fire or its bytes were truncated (drain regression for \
             disable); more than one match means a refactor invoked \
             the stage multiple times. Combined:\n{combined}",
        );
    }
    // Ordering invariant — enable BEFORE payload BEFORE disable.
    // The bracket semantic depends on this; without the ordering
    // pin, a refactor that ran /sched_disable before the payload
    // (silently breaking the "disable-after-payload" contract)
    // would still pass the per-marker presence checks.
    let enable_off = combined
        .find("SHELL_LIFECYCLE_ENABLE_MARKER")
        .expect("enable marker presence asserted above");
    let payload_off = combined
        .find("SHELL_LIFECYCLE_PAYLOAD_MARKER")
        .expect("payload marker presence asserted above");
    let disable_off = combined
        .find("SHELL_LIFECYCLE_DISABLE_MARKER")
        .expect("disable marker presence asserted above");
    assert!(
        enable_off < payload_off,
        "enable marker at byte offset {enable_off} must precede \
         payload marker at byte offset {payload_off} — /sched_enable \
         runs BEFORE the --exec payload per ktstr_guest_init's \
         shell-mode bracket. A refactor that moved the enable script \
         after the payload would invert the bracket semantic without \
         tripping the per-marker count assertions. Combined:\n{combined}",
    );
    assert!(
        payload_off < disable_off,
        "payload marker at byte offset {payload_off} must precede \
         disable marker at byte offset {disable_off} — /sched_disable \
         runs AFTER the --exec payload returns per ktstr_guest_init's \
         shell-mode bracket. A refactor that moved the disable script \
         before the payload would invert the bracket semantic without \
         tripping the per-marker count assertions. Combined:\n{combined}",
    );
}
