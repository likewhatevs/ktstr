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
//! Three scenarios:
//!
//! 1. `scenario_watchdog_stall_captured_emit_schema` — a normal
//!    `--stall-after=1` watchdog stall reaches the late-trigger
//!    Captured dispatch and must produce `schema = SCHEMA_SINGLE` on
//!    disk. Pins the happy Captured-emit dispatch; a future
//!    regression that mis-dispatches to Degraded/Dual/Suppressed on
//!    the normal stall would surface here.
//!
//! 2. `scenario_clean_exit_gate_suppresses_dump` — a clean exit (no
//!    `--stall-after`, scheduler ends via `Drop`) drives the
//!    exit_kind gate's `kind < SCX_EXIT_ERROR` branch to suppress
//!    dump emit. Asserts no primary dump file AND no snapshot-tagged
//!    sibling files exist. Pins the gate's designed clean-exit
//!    suppression semantic; a regression that over-fires the gate
//!    (emitting dumps on clean exits) would surface here.
//!
//! 3. `scenario_watchdog_stall_dump_populates_vcpu_regs_and_maps`
//!    — like scenario 1 but asserts the Captured dump's `vcpu_regs`
//!    has at least one entry with a non-zero `instruction_pointer`,
//!    AND `maps` is non-empty. Pins the content-population invariant
//!    against a regression where the dump file lands with valid JSON
//!    but the BPF map enumeration or vCPU register capture silently
//!    dropped (a failure mode that would have an empty shell pass
//!    scenario 1's schema check).
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
use ktstr::assert::AssertResult;
use ktstr::prelude::SCHEMA_SINGLE;
use ktstr::scenario::ops::{HoldSpec, Step, execute_steps};
use ktstr::test_support::{Scheduler, SchedulerSpec, sidecar_dir};

const KTSTR_SCHED: Scheduler =
    Scheduler::named("ktstr_sched").binary(SchedulerSpec::Discover("scx-ktstr"));

/// Enumerate every snapshot-tagged sibling file the freeze coordinator
/// could have written for `test_name`. The tag glob is open-ended
/// (`{test_name}.snapshot.*.json`) so a future `SNAPSHOT_TAG_*`
/// addition inflates the scan without requiring a test-side update.
/// File-name pattern matches `src/vmm/freeze_coord/snapshot.rs`'s
/// `snapshot_tagged_path` (pub(super), so reproduced here). The
/// production helper strips `.failure-dump.json` from the stem before
/// appending `.snapshot.{safe_tag}.json` (snapshot.rs:385,398-399),
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
    let test_name = ctx
        .entry_name
        .ok_or_else(|| anyhow::anyhow!("dispatch must stamp Ctx.entry_name"))?;
    let dump_path = ctx.failure_dump_path()?;

    let steps = vec![Step {
        setup: vec![ctx.cgroup_def("cg_0")].into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    let mut result = execute_steps(ctx, steps)?;

    let json = std::fs::read_to_string(&dump_path).map_err(|e| {
        anyhow::anyhow!(
            "Captured emit silently dropped the dump despite the framework-attached \
             sink at {} ({e})",
            dump_path.display()
        )
    })?;

    let value: serde_json::Value = serde_json::from_str(&json)
        .map_err(|e| anyhow::anyhow!("dump file is not valid JSON: {e}; payload: {json}"))?;

    let schema = value
        .get("schema")
        .and_then(|s| s.as_str())
        .ok_or_else(|| {
            anyhow::anyhow!("dump JSON missing top-level `schema` field; payload: {value}")
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

    result.note(format!(
        "Captured emit produced schema={SCHEMA_SINGLE} dump at {}",
        dump_path.display()
    ));
    Ok(result)
}

fn scenario_clean_exit_gate_suppresses_dump(ctx: &ktstr::scenario::Ctx) -> Result<AssertResult> {
    let test_name = ctx
        .entry_name
        .ok_or_else(|| anyhow::anyhow!("dispatch must stamp Ctx.entry_name"))?;
    let dump_path = ctx.failure_dump_path()?;

    let steps = vec![Step {
        setup: vec![ctx.cgroup_def("cg_0")].into(),
        ops: vec![],
        hold: HoldSpec::fixed(std::time::Duration::from_secs(2)),
    }];
    let mut result = execute_steps(ctx, steps)?;

    anyhow::ensure!(
        !dump_path.exists(),
        "exit_kind gate must suppress dump emit on clean (non-error-class) exit \
         — found unexpected primary dump file at {}",
        dump_path.display(),
    );

    let siblings = snapshot_sibling_files(test_name);
    anyhow::ensure!(
        siblings.is_empty(),
        "exit_kind gate suppression must not leave snapshot-tagged sibling files; \
         found {}: {:?}",
        siblings.len(),
        siblings,
    );

    result.note("clean exit produced no dump artifacts (primary path absent, no tagged siblings)");
    Ok(result)
}

fn scenario_watchdog_stall_dump_populates_vcpu_regs_and_maps(
    ctx: &ktstr::scenario::Ctx,
) -> Result<AssertResult> {
    let test_name = ctx
        .entry_name
        .ok_or_else(|| anyhow::anyhow!("dispatch must stamp Ctx.entry_name"))?;
    let dump_path = ctx.failure_dump_path()?;

    let steps = vec![Step {
        setup: vec![ctx.cgroup_def("cg_0")].into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    let mut result = execute_steps(ctx, steps)?;

    let json = std::fs::read_to_string(&dump_path).map_err(|e| {
        anyhow::anyhow!(
            "Captured invariant test: dump file missing at {} ({e})",
            dump_path.display()
        )
    })?;
    let value: serde_json::Value = serde_json::from_str(&json)
        .map_err(|e| anyhow::anyhow!("dump file is not valid JSON: {e}"))?;

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

    result.note(format!(
        "Captured dump populated vcpu_regs ({} entries) and maps ({} entries) at {}",
        vcpu_regs.len(),
        maps.len(),
        dump_path.display()
    ));
    Ok(result)
}

#[ktstr::distributed_slice(ktstr::test_support::KTSTR_TESTS)]
#[linkme(crate = ktstr::linkme)]
static __KTSTR_ENTRY_SILENT_DROP_WATCHDOG_STALL_CAPTURED_SCHEMA:
    ktstr::test_support::KtstrTestEntry = ktstr::test_support::KtstrTestEntry {
    name: "silent_drop_watchdog_stall_captured_schema",
    func: scenario_watchdog_stall_captured_emit_schema,
    scheduler: &KTSTR_SCHED,
    extra_sched_args: &["--stall-after=1"],
    watchdog_timeout: std::time::Duration::from_secs(3),
    duration: std::time::Duration::from_secs(10),
    expect_err: true,
    ..ktstr::test_support::KtstrTestEntry::DEFAULT
};

#[ktstr::distributed_slice(ktstr::test_support::KTSTR_TESTS)]
#[linkme(crate = ktstr::linkme)]
static __KTSTR_ENTRY_SILENT_DROP_CLEAN_EXIT_GATE_SUPPRESSION: ktstr::test_support::KtstrTestEntry =
    ktstr::test_support::KtstrTestEntry {
        name: "silent_drop_clean_exit_gate_suppression",
        func: scenario_clean_exit_gate_suppresses_dump,
        scheduler: &KTSTR_SCHED,
        extra_sched_args: &[],
        watchdog_timeout: std::time::Duration::from_secs(10),
        duration: std::time::Duration::from_secs(3),
        expect_err: false,
        ..ktstr::test_support::KtstrTestEntry::DEFAULT
    };

#[ktstr::distributed_slice(ktstr::test_support::KTSTR_TESTS)]
#[linkme(crate = ktstr::linkme)]
static __KTSTR_ENTRY_SILENT_DROP_WATCHDOG_STALL_CAPTURED_CONTENT:
    ktstr::test_support::KtstrTestEntry = ktstr::test_support::KtstrTestEntry {
    name: "silent_drop_watchdog_stall_captured_content",
    func: scenario_watchdog_stall_dump_populates_vcpu_regs_and_maps,
    scheduler: &KTSTR_SCHED,
    extra_sched_args: &["--stall-after=1"],
    watchdog_timeout: std::time::Duration::from_secs(3),
    duration: std::time::Duration::from_secs(10),
    expect_err: true,
    ..ktstr::test_support::KtstrTestEntry::DEFAULT
};
