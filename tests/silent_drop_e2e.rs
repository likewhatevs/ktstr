//! End-to-end regression tests for the failure-dump pipeline. Three
//! scenarios pin the always-on framing (Captured-baseline plus
//! clean-exit-gate-suppression) within which the silent-drop fixes
//! operate.
//!
//! The two pure decisions that drive silent-drop handling — the
//! exit_kind gate (`exit_kind_warrants_dump`) and the BPF-`.bss`-latch
//! rescue (`bss_latch_rescues_suppressed_dump`) — are locked in by
//! truth-table unit tests in `src/vmm/freeze_coord/mod.rs`. An e2e seam
//! for them is impossible: the host coordinator owns the decision and a
//! guest-set static never reaches the host's copy (the prior approach
//! used `pub static` test seams in the freeze coordinator, since removed
//! as a production risk — a flipped bit would force universal dump
//! suppression).
//!
//! Three scenarios. Each boots its workload, then asserts on the
//! host-written dump from a `post_vm_unconditional` callback — the
//! guest is a separate process and cannot read the host-side dump, and
//! a callback's `Err` is a hard FAIL (`PostVmAssertionFailure`) that
//! `expect_err` does not invert. The dump-reading scenarios use
//! `read_dump_skip_placeholder` (skips only a placeholder dump; an
//! empty `maps` in a real dump is the silent-drop regression they
//! exist to catch, so it FAILS rather than skips):
//!
//! 1. `scenario_watchdog_stall_captured_emit_schema` /
//!    `check_captured_emit_schema` — a normal `--stall-after=1`
//!    watchdog stall reaches the late-trigger Captured dispatch and
//!    must produce `schema = SCHEMA_SINGLE`. Pins the happy
//!    Captured-emit dispatch; a future regression that mis-dispatches
//!    to Degraded/Dual/Suppressed on the normal stall surfaces here.
//!
//! 2. `scenario_clean_exit_gate_suppresses_dump` /
//!    `check_clean_exit_gate_suppresses_dump` — a clean exit (no
//!    `--stall-after`, scheduler ends via `Drop`) drives the exit_kind
//!    gate's `kind < SCX_EXIT_ERROR` branch to suppress dump emit. The
//!    callback asserts no primary dump file AND no snapshot-tagged
//!    sibling files exist on the host, but only after the primary VM
//!    reports a clean successful run. A boot failure or watchdog
//!    timeout legitimately emits a failure dump; in that case the
//!    callback leaves the artifact intact and lets the underlying VM
//!    failure remain the diagnostic. Pins the gate's designed
//!    clean-exit suppression; a regression that over-fires the gate
//!    (emitting dumps on clean exits) surfaces here.
//!
//! 3. `scenario_watchdog_stall_dump_populates_vcpu_regs_and_maps` /
//!    `check_captured_content` — like scenario 1 but asserts the
//!    Captured dump's `vcpu_regs` has at least one entry with a
//!    non-zero `instruction_pointer`, AND `maps` is non-empty. Pins
//!    the content-population invariant against a regression where the
//!    dump file lands with valid JSON but the BPF map enumeration or
//!    vCPU register capture silently dropped (a failure mode that
//!    would have an empty shell pass scenario 1's schema check).
//!
//! Silent-drop fix branches NOT exercised by these scenarios (each
//! deferred because the trigger is unavailable in the always-on
//! toolset):
//!
//! - The `sched_exit_final_pass` guard fires only on the SCHED_EXIT
//!   pidfd POLLIN race with the BPF watchpoint latch; no `scx-ktstr`
//!   flag drives it.
//! - The rendezvous-timeout Degraded emit fires only when the vCPU
//!   rendezvous misses its `FREEZE_RENDEZVOUS_TIMEOUT`. The
//!   `KtstrVmBuilder::rendezvous_timeout` setter unlocks this; no
//!   consumer scenario is covered here.

mod common;

use anyhow::Result;
use common::failure_dump::{failure_dump_artifact, read_dump_skip_placeholder};
use ktstr::assert::AssertResult;
use ktstr::prelude::{SCHEMA_SINGLE, VmResult};
use ktstr::scenario::ops::{HoldSpec, Step, await_accessor_ready, execute_steps};
use ktstr::test_support::{Scheduler, SchedulerSpec, sidecar_dir};

const KTSTR_SCHED: Scheduler =
    Scheduler::named("ktstr_sched").binary(SchedulerSpec::Discover("scx-ktstr"));

/// Enumerate every snapshot-tagged sibling file the freeze coordinator
/// could have written for `test_name`. The tag glob is open-ended
/// (`{test_name}.snapshot.*.json`) so a future `SNAPSHOT_TAG_*`
/// addition inflates the scan without requiring a test-side update.
/// File-name pattern matches `src/vmm/freeze_coord/snapshot.rs`'s
/// `snapshot_tagged_path` (pub(super), so reproduced here). The
/// production helper strips `.failure-dump` from the file stem before
/// appending `.snapshot.{safe_tag}.{ext}` (snapshot.rs:507,519-522),
/// so the prefix-and-suffix match here aligns.
#[track_caller]
fn snapshot_sibling_files(test_name: &str) -> Vec<std::path::PathBuf> {
    let prefix = format!("{test_name}.snapshot.");
    let dir = sidecar_dir();
    let entries = match std::fs::read_dir(&dir) {
        Ok(e) => e,
        Err(_) => return Vec::new(),
    };
    entries
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_name()
                .to_str()
                .map(|n| n.starts_with(&prefix) && n.ends_with(".json"))
                .unwrap_or(false)
        })
        .map(|e| e.path())
        .collect()
}

fn scenario_watchdog_stall_captured_emit_schema(
    ctx: &ktstr::scenario::Ctx,
) -> Result<AssertResult> {
    await_accessor_ready();
    let steps = vec![Step {
        setup: vec![ctx.cgroup_def("cg_0")].into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    execute_steps(ctx, steps)
}

/// Host-side post_vm assertion for `silent_drop_watchdog_stall_captured_schema`.
/// The guest cannot read the host-written dump, so the schema +
/// no-sibling assertions run here; the callback's Err is a hard FAIL
/// (`PostVmAssertionFailure`) that `expect_err` does not invert.
fn check_captured_emit_schema(result: &VmResult) -> Result<()> {
    let test_name = result
        .entry_name
        .ok_or_else(|| anyhow::anyhow!("VmResult.entry_name is None"))?;
    let value = read_dump_skip_placeholder(result)?;
    let dump_artifact = failure_dump_artifact(result);

    let schema = value
        .get("schema")
        .and_then(|s| s.as_str())
        .ok_or_else(|| {
            anyhow::anyhow!("dump JSON missing top-level `schema` field; {dump_artifact}")
        })?;
    anyhow::ensure!(
        schema == SCHEMA_SINGLE,
        "Captured emit must produce SCHEMA_SINGLE ({SCHEMA_SINGLE:?}); got \
         schema={schema:?} (a `degraded` or `dual` schema here indicates the \
         freeze coordinator took a different dispatch arm than the happy \
         Captured path)"
    );

    let siblings = snapshot_sibling_files(test_name);
    anyhow::ensure!(
        siblings.is_empty(),
        "Captured emit must not leave snapshot-tagged sibling files when \
         dual_snapshot is off; found {}: {:?}",
        siblings.len(),
        siblings,
    );

    eprintln!("Captured emit produced schema={SCHEMA_SINGLE} dump");
    Ok(())
}

fn scenario_clean_exit_gate_suppresses_dump(ctx: &ktstr::scenario::Ctx) -> Result<AssertResult> {
    let steps = vec![Step {
        setup: vec![ctx.cgroup_def("cg_0")].into(),
        ops: vec![],
        hold: HoldSpec::fixed(std::time::Duration::from_secs(2)),
    }];
    execute_steps(ctx, steps)
}

/// Assert the clean-exit artifact contract after its premise holds.
///
/// `artifacts` is deliberately lazy: a failed primary run must not even
/// inspect the failure artifacts it legitimately produced. Besides
/// preventing a derivative assertion from obscuring the VM failure,
/// this guarantees the callback never removes or otherwise consumes
/// the evidence needed to diagnose it.
fn assert_clean_exit_dump_suppression(
    primary_run_succeeded: bool,
    artifacts: impl FnOnce() -> Result<(std::path::PathBuf, Vec<std::path::PathBuf>)>,
) -> Result<()> {
    if !primary_run_succeeded {
        return Ok(());
    }

    let (dump_path, siblings) = artifacts()?;

    anyhow::ensure!(
        !dump_path.exists(),
        "exit_kind gate must suppress dump emit on clean (non-error-class) exit \
         — found unexpected primary dump file at {}",
        dump_path.display(),
    );

    anyhow::ensure!(
        siblings.is_empty(),
        "exit_kind gate suppression must not leave snapshot-tagged sibling files; \
         found {}: {:?}",
        siblings.len(),
        siblings,
    );

    eprintln!("clean exit produced no dump artifacts (primary path absent, no tagged siblings)");
    Ok(())
}

/// Host-side post_vm assertion for `silent_drop_clean_exit_gate_suppression`.
/// A clean (non-error-class) exit must suppress the dump; the guest
/// cannot observe the host-side dump path, so the absence check runs
/// here. The callback is wired unconditionally so the host artifact is
/// observable, but it asserts only when `VmResult::success` proves the
/// primary run reached the clean-success premise. On a boot failure,
/// crash, or watchdog timeout, the real failure dump is retained and
/// the underlying VM error remains authoritative.
fn check_clean_exit_gate_suppresses_dump(result: &VmResult) -> Result<()> {
    assert_clean_exit_dump_suppression(result.success, || {
        let test_name = result
            .entry_name
            .ok_or_else(|| anyhow::anyhow!("VmResult.entry_name is None"))?;
        Ok((
            result.failure_dump_path()?,
            snapshot_sibling_files(test_name),
        ))
    })
}

fn scenario_watchdog_stall_dump_populates_vcpu_regs_and_maps(
    ctx: &ktstr::scenario::Ctx,
) -> Result<AssertResult> {
    await_accessor_ready();
    let steps = vec![Step {
        setup: vec![ctx.cgroup_def("cg_0")].into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    execute_steps(ctx, steps)
}

/// Host-side post_vm assertion for `silent_drop_watchdog_stall_captured_content`.
/// The guest cannot read the host-written dump, so the content-population
/// assertions run here. A placeholder dump is skipped (inconclusive),
/// but a real dump that dropped its vcpu_regs or map enumeration is the
/// silent-drop regression this test exists to catch — so the emptiness
/// checks hard FAIL (`PostVmAssertionFailure`, which `expect_err` does
/// not invert). Hence read_dump_skip_placeholder, not read_failure_dump.
fn check_captured_content(result: &VmResult) -> Result<()> {
    let test_name = result
        .entry_name
        .ok_or_else(|| anyhow::anyhow!("VmResult.entry_name is None"))?;
    let value = read_dump_skip_placeholder(result)?;

    let vcpu_regs = value
        .get("vcpu_regs")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow::anyhow!("Captured dump missing top-level `vcpu_regs` array"))?;
    let populated = vcpu_regs.iter().any(|s| {
        s.is_object()
            && s.get("instruction_pointer")
                .and_then(|ip| ip.as_u64())
                .is_some_and(|ip| ip != 0)
    });
    anyhow::ensure!(
        populated,
        "Captured dump's `vcpu_regs` has no entry with a non-zero instruction \
         pointer — vCPU register capture silently dropped before the dump landed"
    );

    let maps = value
        .get("maps")
        .and_then(|m| m.as_array())
        .ok_or_else(|| anyhow::anyhow!("Captured dump missing top-level `maps` array"))?;
    anyhow::ensure!(
        !maps.is_empty(),
        "Captured dump's `maps` array is empty — BPF map enumeration silently \
         dropped every entry, leaving the dump shell with no content"
    );

    let siblings = snapshot_sibling_files(test_name);
    anyhow::ensure!(
        siblings.is_empty(),
        "Captured emit must not leave snapshot-tagged sibling files when \
         dual_snapshot is off; found {}: {:?}",
        siblings.len(),
        siblings,
    );

    eprintln!(
        "Captured dump populated vcpu_regs ({} entries) and maps ({} entries)",
        vcpu_regs.len(),
        maps.len(),
    );
    Ok(())
}

#[ktstr::ktstr_test_entry]
static __KTSTR_ENTRY_SILENT_DROP_WATCHDOG_STALL_CAPTURED_SCHEMA:
    ktstr::test_support::KtstrTestEntry = ktstr::test_support::KtstrTestEntry {
    name: "silent_drop_watchdog_stall_captured_schema",
    func: scenario_watchdog_stall_captured_emit_schema,
    scheduler: &KTSTR_SCHED,
    extra_sched_args: &["--stall-after=1"],
    watchdog_timeout: std::time::Duration::from_secs(3),
    duration: std::time::Duration::from_secs(10),
    // Stall death inverts to PASS; the schema + no-sibling assertions
    // gate in check_captured_emit_schema (post_vm_unconditional, hard
    // FAIL via PostVmAssertionFailure that expect_err does not invert).
    expect_err: true,
    post_vm_unconditional: Some(check_captured_emit_schema),
    ..ktstr::test_support::KtstrTestEntry::DEFAULT
};

#[ktstr::ktstr_test_entry]
static __KTSTR_ENTRY_SILENT_DROP_CLEAN_EXIT_GATE_SUPPRESSION: ktstr::test_support::KtstrTestEntry =
    ktstr::test_support::KtstrTestEntry {
        name: "silent_drop_clean_exit_gate_suppression",
        func: scenario_clean_exit_gate_suppresses_dump,
        scheduler: &KTSTR_SCHED,
        extra_sched_args: &[],
        watchdog_timeout: std::time::Duration::from_secs(10),
        duration: std::time::Duration::from_secs(3),
        // Clean exit (expect_err: false) → the suppression assertion gates
        // in check_clean_exit_gate_suppresses_dump (post_vm_unconditional).
        // A failed primary run bypasses the absence assertion and keeps its
        // legitimate failure artifact; a clean successful run with an
        // artifact still returns a hard FAIL.
        expect_err: false,
        post_vm_unconditional: Some(check_clean_exit_gate_suppresses_dump),
        ..ktstr::test_support::KtstrTestEntry::DEFAULT
    };

#[ktstr::ktstr_test_entry]
static __KTSTR_ENTRY_SILENT_DROP_WATCHDOG_STALL_CAPTURED_CONTENT:
    ktstr::test_support::KtstrTestEntry = ktstr::test_support::KtstrTestEntry {
    name: "silent_drop_watchdog_stall_captured_content",
    func: scenario_watchdog_stall_dump_populates_vcpu_regs_and_maps,
    scheduler: &KTSTR_SCHED,
    extra_sched_args: &["--stall-after=1"],
    watchdog_timeout: std::time::Duration::from_secs(3),
    duration: std::time::Duration::from_secs(10),
    // Stall death inverts to PASS; the content-population assertions gate
    // in check_captured_content (post_vm_unconditional, hard FAIL via
    // PostVmAssertionFailure that expect_err does not invert).
    expect_err: true,
    post_vm_unconditional: Some(check_captured_content),
    ..ktstr::test_support::KtstrTestEntry::DEFAULT
};

#[cfg(test)]
mod clean_exit_gate_tests {
    use super::assert_clean_exit_dump_suppression;

    fn fixture_dump_path(dir: &std::path::Path, stem: &str) -> std::path::PathBuf {
        dir.join(stem).with_extension("failure-dump.json")
    }

    #[test]
    fn failed_primary_run_preserves_dump_without_resolving_artifacts() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let dump_path = tmp.path().join("legitimate-failure-dump.json");
        std::fs::write(&dump_path, b"{\"schema\":\"ktstr.failure_dump.v1\"}")
            .expect("write failure dump");
        let artifacts_resolved = std::cell::Cell::new(false);

        assert_clean_exit_dump_suppression(false, || {
            artifacts_resolved.set(true);
            Ok((dump_path.clone(), Vec::new()))
        })
        .expect("failed primary run must retain its underlying diagnostic");

        assert!(
            !artifacts_resolved.get(),
            "failed primary run must not inspect clean-exit artifacts"
        );
        assert!(
            dump_path.exists(),
            "legitimate failure dump must remain available for diagnosis"
        );
    }

    #[test]
    fn successful_primary_run_rejects_primary_dump() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let dump_path = fixture_dump_path(tmp.path(), "unexpected");
        std::fs::write(&dump_path, b"{}").expect("write unexpected dump");

        let err = assert_clean_exit_dump_suppression(true, || Ok((dump_path.clone(), Vec::new())))
            .expect_err("clean success with a primary dump must fail");

        assert!(
            err.to_string().contains("unexpected primary dump file"),
            "unexpected diagnostic: {err:#}"
        );
        assert!(
            dump_path.exists(),
            "assertion must not delete the unexpected artifact"
        );
    }

    #[test]
    fn successful_primary_run_rejects_snapshot_sibling() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let dump_path = fixture_dump_path(tmp.path(), "absent");
        let sibling = tmp.path().join("unexpected.snapshot.watchpoint.json");
        std::fs::write(&sibling, b"{}").expect("write unexpected sibling");

        let err =
            assert_clean_exit_dump_suppression(true, || Ok((dump_path, vec![sibling.clone()])))
                .expect_err("clean success with a snapshot sibling must fail");

        assert!(
            err.to_string().contains("snapshot-tagged sibling files"),
            "unexpected diagnostic: {err:#}"
        );
        assert!(
            sibling.exists(),
            "assertion must not delete the unexpected artifact"
        );
    }

    #[test]
    fn successful_primary_run_accepts_absent_artifacts() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let dump_path = fixture_dump_path(tmp.path(), "absent");

        assert_clean_exit_dump_suppression(true, || Ok((dump_path, Vec::new())))
            .expect("clean success without dump artifacts must pass");
    }
}
