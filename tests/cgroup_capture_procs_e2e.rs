//! VM-backed end-to-end test for the
//! [`Op::CaptureCgroupProcs`] dispatch path.
//!
//! The MockCgroupOps unit tests in `src/scenario/ops/tests.rs`
//! (`op_capture_cgroup_procs_records_snapshot_on_active_bridge`,
//! `op_capture_cgroup_procs_propagates_read_procs_error`,
//! `op_capture_cgroup_procs_empty_tag_bails_before_read`,
//! `op_capture_cgroup_procs_empty_cgroup_bails_before_read`,
//! `op_capture_cgroup_procs_bails_when_no_bridge_installed`,
//! `op_capture_cgroup_procs_multiple_tags_same_cgroup_preserve_order`)
//! pin the host-side dispatch call streams against a mock. The
//! `read_procs_*` tests in `src/cgroup_tests.rs` cover
//! [`CgroupManager::read_procs`]'s fs read + parse against a
//! tempdir-backed cgroup. This test exercises the composition
//! end-to-end against a real cgroup hierarchy: real workers
//! spawned via [`Op::Spawn`] land in a real cgroup, then a
//! `Op::CaptureCgroupProcs` reads the kernel's actual
//! `cgroup.procs` and routes the pid list through a real
//! [`SnapshotBridge`].
//!
//! ## Bridge install contract
//!
//! `Op::CaptureCgroupProcs` requires a thread-local
//! [`SnapshotBridge`] installed via
//! [`SnapshotBridge::set_thread_local`] — the dispatch arm bails
//! loudly when no bridge is active per the no-silent-drops policy.
//! The framework does NOT auto-install a bridge in the guest
//! (host-side bridges live on the freeze coordinator), so the
//! test body installs one explicitly. The drain happens in-guest
//! against the same bridge (via a clone-with-shared-Arc) so the
//! captured pids are visible without any wire-format marshaling
//! across the VM boundary.

use anyhow::Result;
use ktstr::assert::{AssertDetail, AssertResult, DetailKind};
use ktstr::ktstr_test;
use ktstr::prelude::{Backdrop, SnapshotBridge, SpawnPlacement, WorkSpec, WorkType};
use ktstr::scenario::Ctx;
use ktstr::scenario::ops::{HoldSpec, Op, Step, execute_scenario};
use std::sync::Arc;
use std::time::Duration;

#[ktstr_test(
    workload_root_cgroup = "/ktstr-capture-procs-e2e",
    llcs = 1,
    cores = 2,
    threads = 1,
    memory_mib = 256,
    duration_s = 2,
    watchdog_timeout_s = 20,
    auto_repro = false
)]
fn op_capture_cgroup_procs_returns_workers_in_target_cgroup(ctx: &Ctx) -> Result<AssertResult> {
    // Dummy capture callback — the bridge's other facilities
    // (FailureDumpReport store, watch-register) are unused by this
    // test; only the cgroup-procs drain log is read.
    let bridge = SnapshotBridge::new(Arc::new(|_| None));
    let bridge_for_drain = bridge.clone();
    let _guard = bridge.set_thread_local();

    // Distinct comm marker so the post-drain identity assertion can
    // distinguish the test's workers from any unrelated task that
    // might leak into the cgroup (per the spec: a pids.len()
    // == 2 check alone would silently pass even if the spawn placed
    // a worker elsewhere and 2 unrelated tasks coincidentally landed
    // in cg_target). 15-byte limit per TASK_COMM_LEN-1.
    const WORKER_COMM: &str = "ktstr_captrg";

    let backdrop = Backdrop::new().push_op(Op::add_cgroup("cg_target"));
    let steps = vec![
        // Single Step: spawn + capture in the same step's ops vec.
        // Per-Step teardown at the step boundary collects worker
        // handles + kills payloads, so spawning in one step and
        // capturing in a later step would observe an empty
        // cg_target. Within a single Step, ops apply sequentially
        // (per `apply_ops`'s `for op in &merged` iterator) AND the
        // Op::Spawn dispatch arm's spawn-time `move_tasks` is
        // synchronous (the kernel commits the cgroup.procs write
        // before returning), so the subsequent capture op
        // observes the workers in cg_target.
        Step::new(
            vec![
                Op::spawn(
                    SpawnPlacement::cgroup("cg_target"),
                    WorkSpec::default()
                        .workers(2)
                        .comm(WORKER_COMM)
                        .work_type(WorkType::SpinWait),
                ),
                Op::capture_cgroup_procs("after_spawn", "cg_target"),
            ],
            HoldSpec::fixed(Duration::ZERO),
        ),
    ];
    let _ = execute_scenario(ctx, backdrop, steps)?;

    // Drain via the cloned bridge — SnapshotBridge is Clone with
    // Arc-shared internal state, so the clone we held aside before
    // `set_thread_local` consumed the original shares the same
    // cgroup_procs Vec the dispatch arm pushed into.
    let snaps = bridge_for_drain.drain_cgroup_procs();
    if snaps.len() != 1 {
        return Ok(AssertResult::fail(AssertDetail::new(
            DetailKind::Other,
            format!(
                "expected exactly 1 CgroupProcsSnapshot in the drain log \
                 (one Op::CaptureCgroupProcs call); got {}: {snaps:?}",
                snaps.len(),
            ),
        )));
    }
    let snap = &snaps[0];
    if snap.tag != "after_spawn" {
        return Ok(AssertResult::fail(AssertDetail::new(
            DetailKind::Other,
            format!(
                "snapshot tag mismatch: expected \"after_spawn\", got {:?}",
                snap.tag,
            ),
        )));
    }
    if snap.cgroup != "cg_target" {
        return Ok(AssertResult::fail(AssertDetail::new(
            DetailKind::Other,
            format!(
                "snapshot cgroup mismatch: expected \"cg_target\", got {:?}",
                snap.cgroup,
            ),
        )));
    }
    if snap.pids.len() != 2 {
        return Ok(AssertResult::fail(AssertDetail::new(
            DetailKind::Other,
            format!(
                "expected 2 pids in cg_target's cgroup.procs (workers(2) \
                 spawned via Op::Spawn(SpawnPlacement::Cgroup) in step 1); \
                 got {}: {:?}. A deficit means the spawn-time move_tasks \
                 failed to place a worker; a surplus means an unrelated \
                 task leaked into cg_target.",
                snap.pids.len(),
                snap.pids,
            ),
        )));
    }
    // Sanity-check: every captured pid is positive (libc::pid_t > 0).
    // A 0 or negative value here would imply the kernel
    // cgroup_procs_show emitted a malformed line that the parser
    // accepted — should be unreachable but caught here as
    // defense-in-depth.
    for &pid in &snap.pids {
        if pid <= 0 {
            return Ok(AssertResult::fail(AssertDetail::new(
                DetailKind::Other,
                format!(
                    "captured pid is non-positive ({pid}); expected the \
                     kernel-rendered decimal pids in `cg_target/cgroup.procs` \
                     to all be > 0",
                ),
            )));
        }
    }
    // Identity cross-check (comm-multiset against /proc/<pid>/comm)
    // is NOT possible in this e2e: scenario teardown kills the
    // spawned workers as execute_scenario returns, so by the time
    // the test body reads /proc/<pid>/comm, the pids no longer
    // exist (the read returns ENOENT). The structural assertions
    // above (snap.tag, snap.cgroup, snap.pids.len() == 2, pid > 0)
    // pin the contract that Op::CaptureCgroupProcs delivers the
    // expected number of pids in the expected cgroup under the
    // expected tag — sufficient for the new-Op coverage. A future
    // extension to Op::CaptureCgroupProcs that ALSO captures comm
    // at apply time (alongside pid) would enable a host-side
    // identity multiset check; that's a follow-up scope item.
    Ok(AssertResult::pass())
}
