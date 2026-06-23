//! End-to-end tests for the snapshot capture pipeline.
//!
//! Each `#[ktstr_test]` scenario fires a snapshot op from inside a
//! real guest VM. The guest verifies the SHM round-trip succeeded
//! (Op returned Ok). The `post_vm` callback runs on the HOST after
//! `vm.run()` returns and asserts the captured `FailureDumpReport`
//! on the `SnapshotBridge` contains real BTF-rendered BPF state.

use anyhow::Result;
use ktstr::assert::AssertResult;
use ktstr::ktstr_test;
use ktstr::prelude::{RenderedValue, VmResult};
use ktstr::scenario::ops::{HoldSpec, Op, Step, execute_steps};
use ktstr::test_support::{Scheduler, SchedulerSpec};

const KTSTR_SCHED: Scheduler =
    Scheduler::named("ktstr_sched").binary(SchedulerSpec::Discover("scx-ktstr"));

/// Host-side content assertion: verify the bridge has a capture with the
/// scheduler's .bss containing real BTF-rendered globals. When
/// `expect_vcpus` is `Some(n)`, additionally pin (1) that the report
/// enumerated exactly `n` `vcpu_regs` slots — the freeze coordinator pushes
/// one slot per booted vCPU regardless of outcome (`collect_vcpu_regs` =
/// 1 BSP + AP reg snapshots), so the count is the vCPU count — and (2) that
/// all `n` slots are populated (`Some`), which proves the freeze rendezvous
/// reached, parked, and captured every AP (a stalled vCPU leaves a `None`
/// slot). all-`Some` is the direct >255 N-of-N freeze pin; the count alone
/// is only satisfied by thread enumeration.
fn assert_bridge_capture(result: &VmResult, expect_vcpus: Option<usize>) -> Result<()> {
    let captured = result.snapshot_bridge.drain();
    anyhow::ensure!(
        !captured.is_empty(),
        "snapshot bridge is empty — no captures reached the host"
    );
    for (tag, report) in &captured {
        anyhow::ensure!(
            !report.maps.is_empty(),
            "snapshot '{tag}' has 0 maps — capture produced nothing"
        );
        let bss = report.maps.iter().find(|m| m.name.ends_with(".bss"));
        anyhow::ensure!(
            bss.is_some(),
            "snapshot '{tag}' has {} maps but no .bss map. maps: {:?}",
            report.maps.len(),
            report
                .maps
                .iter()
                .map(|m| m.name.as_str())
                .collect::<Vec<_>>()
        );
        let bss = bss.unwrap();
        let has_real_members = bss
            .value
            .as_ref()
            .and_then(|v| match v {
                RenderedValue::Struct { members, .. } => Some(members.len() >= 3),
                _ => None,
            })
            .unwrap_or(false);
        anyhow::ensure!(
            has_real_members,
            "snapshot '{tag}' .bss '{}' has no BTF-rendered members — \
             capture did not produce real scheduler state",
            bss.name
        );
        if let Some(n) = expect_vcpus {
            // The slot COUNT pins the vCPU count: one slot per booted vCPU
            // is enumerated regardless of capture outcome...
            anyhow::ensure!(
                report.vcpu_regs.len() == n,
                "snapshot '{tag}' enumerated {} vcpu_regs slots, expected {n} \
                 (one slot per booted vCPU) — wrong vCPU count",
                report.vcpu_regs.len()
            );
            // ...and the count of POPULATED (Some) slots pins that the
            // freeze rendezvous reached, parked, and captured every one. A
            // vCPU that stalled past the rendezvous leaves a None slot
            // (freeze_coord collect_vcpu_regs), so all-Some proves the >255
            // N-of-N freeze succeeded — a Degraded/partial freeze fails here.
            let captured = report.vcpu_regs.iter().filter(|r| r.is_some()).count();
            anyhow::ensure!(
                captured == n,
                "snapshot '{tag}' captured regs for {captured} of {n} vCPUs \
                 (None slots = vCPUs the freeze rendezvous did not reach + \
                 capture at the split-irqchip ceiling)",
            );
        }
    }
    Ok(())
}

/// `post_vm` for the default-topology capture tests: real content, no
/// vCPU-cardinality requirement.
fn assert_bridge_has_real_capture(result: &VmResult) -> Result<()> {
    assert_bridge_capture(result, None)
}

/// `post_vm` for the 256-vCPU wide-SMP capture: real content AND all 256
/// `vcpu_regs` slots populated (`Some`), pinning that the freeze coordinator
/// reached, parked, and captured every one of the 256 vCPUs (16 LLCs x 16
/// cores x 1 thread) at the split-irqchip ceiling — not merely that the 256
/// slots were enumerated or that "some real data" landed.
fn assert_bridge_has_real_capture_wide_smp(result: &VmResult) -> Result<()> {
    assert_bridge_capture(result, Some(256))
}

#[ktstr_test(
    scheduler = KTSTR_SCHED,
    duration_s = 10,
    watchdog_timeout_s = 15,
    auto_repro = false,
    post_vm = assert_bridge_has_real_capture,
)]
fn snapshot_real_capture_op_snapshot(ctx: &ktstr::scenario::Ctx) -> Result<AssertResult> {
    let steps = vec![Step {
        setup: vec![ctx.cgroup_def("cg_0")].into(),
        ops: vec![Op::capture_snapshot("mid_run")],
        hold: HoldSpec::FULL,
    }];
    let mut result = execute_steps(ctx, steps)?;
    result.note("Op::capture_snapshot('mid_run') SHM request succeeded");
    Ok(result)
}

#[ktstr_test(
    scheduler = KTSTR_SCHED,
    duration_s = 10,
    watchdog_timeout_s = 30,
    auto_repro = false,
    post_vm = assert_bridge_has_real_capture,
)]
fn snapshot_real_capture_op_watch_snapshot(ctx: &ktstr::scenario::Ctx) -> Result<AssertResult> {
    let steps = vec![Step {
        setup: vec![ctx.cgroup_def("cg_0")].into(),
        ops: vec![Op::watch_snapshot("jiffies_64")],
        hold: HoldSpec::FULL,
    }];
    let mut result = execute_steps(ctx, steps)?;
    result.note("Op::watch_snapshot('jiffies_64') SHM request succeeded");
    Ok(result)
}

/// Host-side content assertion for the principled active-scheduler
/// walker: `Snapshot::active()` resolves the active
/// scheduler's obj for every non-placeholder capture. The walker
/// runs at freeze time and may populate `report.active_obj_name`
/// when the struct_ops map name's prefix cross-references a global-
/// section map; for libbpf-named struct_ops maps (`<name>_ops`
/// without an obj prefix) the walker returns None and the projection-
/// side prefix-grouping heuristic resolves the obj from the
/// single global obj prefix. Both paths must produce a non-None
/// `active_obj` through `Snapshot::active()`. The assertion fires
/// when either:
/// - `report.active_obj_name` IS Some but doesn't appear in any
///   global-section map prefix (walker drift), OR
/// - `Snapshot::active()` returns NoActiveScheduler in a capture
///   that contains both `scx_sched_state` AND at least one global-
///   section map (the projection failed to resolve).
fn assert_active_obj_name_resolved(result: &VmResult) -> Result<()> {
    let captured = result.snapshot_bridge.drain();
    anyhow::ensure!(
        !captured.is_empty(),
        "snapshot bridge is empty — no captures reached the host"
    );
    for (tag, report) in &captured {
        anyhow::ensure!(
            !report.is_placeholder,
            "snapshot '{tag}' is a placeholder report — capture path did not produce real data"
        );
        // Inventory: global-section obj prefixes present in this
        // capture. Both the walker (when populated) and the
        // heuristic must resolve to one of these.
        let global_obj_names: std::collections::BTreeSet<&str> = report
            .maps
            .iter()
            .filter(|m| {
                m.name.ends_with(".bss") || m.name.ends_with(".data") || m.name.ends_with(".rodata")
            })
            .filter_map(|m| m.name.split('.').next())
            .filter(|s| !s.is_empty())
            .collect();
        if let Some(active) = report.active_obj_name.as_deref() {
            anyhow::ensure!(
                global_obj_names.contains(active),
                "snapshot '{tag}': report.active_obj_name={active:?} not among \
                 global-section obj prefixes {global_obj_names:?} — walker \
                 cross-check drift"
            );
        }
        // Snapshot::active() — the user-facing API — must resolve a
        // non-None active_obj for a capture that has both scx state
        // and global-section maps. The projection uses
        // active_obj_name as the principled tiebreaker (walker-
        // populated) and falls back to "exactly one global prefix"
        // (heuristic). Either path closes the test.
        anyhow::ensure!(
            !global_obj_names.is_empty(),
            "snapshot '{tag}' has scx state but no global-section maps — \
             the active-scheduler resolution has no obj to land on"
        );
        let view = ktstr::scenario::snapshot::Snapshot::new(report);
        let _ = view.active().map_err(|e| {
            anyhow::anyhow!(
                "Snapshot::active() failed for '{tag}' (global obj \
                 prefixes available: {global_obj_names:?}; walker \
                 active_obj_name: {:?}): {e:?}",
                report.active_obj_name
            )
        })?;
    }
    Ok(())
}

/// End-to-end pin: the principled active-scheduler walker
/// resolves `active_obj_name` at freeze time via
/// `kvalue.data == *scx_root` matching, and `Snapshot::active()`
/// returns the matching obj. A regression that breaks the walker
/// — wrong field offset, wrong KVA comparison, struct_ops map
/// `value_kva` populated incorrectly — drops `active_obj_name` to
/// `None` and the assertion above fires.
#[ktstr_test(
    scheduler = KTSTR_SCHED,
    duration_s = 10,
    watchdog_timeout_s = 15,
    auto_repro = false,
    post_vm = assert_active_obj_name_resolved,
)]
fn principled_active_scheduler_walker_resolves_active_obj(
    ctx: &ktstr::scenario::Ctx,
) -> Result<AssertResult> {
    let steps = vec![Step {
        setup: vec![ctx.cgroup_def("cg_0")].into(),
        ops: vec![Op::capture_snapshot("active_walker_probe")],
        hold: HoldSpec::FULL,
    }];
    let mut result = execute_steps(ctx, steps)?;
    result.note(
        "Op::capture_snapshot('active_walker_probe') succeeded; \
         post_vm verifies active_obj_name and Snapshot::active() round-trip",
    );
    Ok(result)
}

/// Snapshot capture at the xAPIC ceiling: a 256-vCPU (16 LLCs x 16 cores,
/// top APIC ID 255 > 254 -> split-irqchip / userspace-IOAPIC path) guest
/// fires `Op::capture_snapshot`, and the host asserts the bridge captured
/// real BTF-rendered BPF state AND all 256 `vcpu_regs` slots populated.
/// Proves the snapshot pipeline works at >255 vCPUs: the freeze
/// coordinator's rendezvous kicks all 256 vCPU threads (SIGRTMIN to every
/// AP) — all 256 `vcpu_regs` slots being `Some` is the direct pin that the
/// rendezvous reached, parked, and captured every AP (a partial freeze
/// leaves `None` slots) — the BPF-map walk reads scx-ktstr's state, and
/// serialization round-trips through SHM, none of which is otherwise
/// exercised above the 254-APIC-ID split-irqchip boundary.
#[ktstr_test(
    scheduler = KTSTR_SCHED,
    llcs = 16,
    cores = 16,
    threads = 1,
    no_perf_mode,
    duration_s = 10,
    watchdog_timeout_s = 60,
    auto_repro = false,
    post_vm = assert_bridge_has_real_capture_wide_smp,
)]
fn snapshot_real_capture_wide_smp(ctx: &ktstr::scenario::Ctx) -> Result<AssertResult> {
    let total = ctx.topo.total_cpus();
    anyhow::ensure!(
        total > 254,
        "need a >254-vCPU topology to exercise the split-irqchip snapshot \
         path (got {total})"
    );
    let steps = vec![Step {
        setup: vec![ctx.cgroup_def("cg_0")].into(),
        ops: vec![Op::capture_snapshot("wide_smp")],
        hold: HoldSpec::FULL,
    }];
    let mut result = execute_steps(ctx, steps)?;
    result.note("256-vCPU Op::capture_snapshot('wide_smp') SHM request succeeded");
    Ok(result)
}
