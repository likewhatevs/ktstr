//! End-to-end test for `KtstrTestEntry::num_snapshots` periodic
//! capture.
//!
//! Boots a real guest VM with `num_snapshots = 3` and a 10 s
//! workload duration (interior boundaries land at scenario_start
//! plus {3 s, 5 s, 7 s}). The guest just holds the cgroup for the
//! full duration; the freeze coordinator's periodic-capture loop
//! fires the captures from the host side and stores reports on
//! the host-side `SnapshotBridge`.
//!
//! The `post_vm` callback runs on the HOST after `vm.run()`
//! returns and asserts:
//!   * `result.periodic_target == 3` (the configured count)
//!   * `result.periodic_fired >= 1` (best-effort — CI cold-cache
//!     latency or kill-flag races may cut the sequence short, but
//!     a healthy guest should fire at least the first boundary)
//!   * `drain_ordered()` returns reports tagged `periodic_NNN` in
//!     ascending NNN order, with no `periodic_` tags missing
//!     between two stored entries.
//!   * Each successful (non-placeholder) report has at least one
//!     `.maps` entry — the freeze-and-capture path actually walked
//!     scheduler-state BPF maps, not just stored a placeholder.

use anyhow::Result;
use ktstr::assert::AssertResult;
use ktstr::ktstr_test;
use ktstr::prelude::VmResult;
use ktstr::scenario::ops::{HoldSpec, Step, execute_steps};
use ktstr::test_support::{Scheduler, SchedulerSpec};

const KTSTR_SCHED: Scheduler =
    Scheduler::named("ktstr_sched").binary(SchedulerSpec::Discover("scx-ktstr"));

/// Scheduler variant with `cgroup_parent` set. The framework's
/// `cgroup_parent` slot controls the cgroup-root path used by the
/// guest CgroupManager and must NOT auto-inject
/// `--cell-parent-cgroup` into the scheduler's argv — the
/// `periodic_capture_with_cgroup_parent_e2e` test locks in that
/// the periodic-capture pipeline still fires when this slot is
/// populated (the prior auto-inject forced cell-aware schedulers
/// into userspace_managed_cell_mode and silently 0-fired the
/// pipeline).
const KTSTR_SCHED_WITH_CGPAR: Scheduler = Scheduler::named("ktstr_sched_with_cgpar")
    .binary(SchedulerSpec::Discover("scx-ktstr"))
    .cgroup_parent("/ktstr_periodic");

/// Host-side check: every periodic capture stored on the bridge
/// has the expected tag shape, ordering, and non-empty content
/// (when not a placeholder).
fn assert_periodic_captures(result: &VmResult) -> Result<()> {
    // Environmental starvation gate: zero real captures under a
    // witnessed-contended host is a non-verdict (the readiness-gated
    // capture chain was starved past the workload window), not a
    // capture regression — SKIP instead of failing the assertions
    // below. A quiet-host zero-capture run still falls through and
    // fails with the specific diagnosis. See `periodic_starvation_gate`.
    ktstr::prelude::periodic_starvation_gate(result, 1)?;
    anyhow::ensure!(
        result.periodic_target == 3,
        "periodic_target must mirror the configured num_snapshots = 3, got {}",
        result.periodic_target,
    );
    anyhow::ensure!(
        result.periodic_fired >= 1,
        "periodic_fired must be at least 1 — a healthy guest should \
         cross the first boundary at scenario_start + 3 s during \
         the 10 s workload window. Got {} of {}.",
        result.periodic_fired,
        result.periodic_target,
    );
    // periodic_fired counts the time-sliced boundaries (up to
    // periodic_target = num_snapshots) PLUS one capture forced at the start
    // of each scenario step the guest enters — the coordinator guarantees
    // every phase a periodic sample so a dilated run cannot leave a later
    // phase empty. So fired may exceed the time-sliced target by up to the
    // number of steps entered, but no more (no runaway captures).
    let steps_entered = result
        .stimulus_timeline()
        .iter()
        .filter(|e| e.label.starts_with("StepStart"))
        .count() as u32;
    anyhow::ensure!(
        result.periodic_fired <= result.periodic_target + steps_entered,
        "periodic_fired ({}) must not exceed periodic_target ({}) + one \
         forced capture per scenario step entered ({})",
        result.periodic_fired,
        result.periodic_target,
        steps_entered,
    );

    // Drain in insertion order so we can assert the tag sequence
    // is contiguous from periodic_000.
    let captured = result.snapshot_bridge.drain_ordered();
    let periodic_entries: Vec<_> = captured
        .iter()
        .filter(|(tag, _)| tag.starts_with("periodic_"))
        .collect();
    anyhow::ensure!(
        !periodic_entries.is_empty(),
        "bridge has no `periodic_*` entries despite periodic_fired = {}",
        result.periodic_fired,
    );
    anyhow::ensure!(
        periodic_entries.len() == result.periodic_fired as usize,
        "bridge has {} periodic_* entries but periodic_fired = {} — \
         counts must match (each fire stores exactly once)",
        periodic_entries.len(),
        result.periodic_fired,
    );

    // Tags must be `periodic_000`, `periodic_001`, ... contiguous
    // from index 0. Any gap indicates a fire path that advanced
    // `next_periodic_idx` without storing onto the bridge — a bug.
    for (i, (tag, _)) in periodic_entries.iter().enumerate() {
        let expected = format!("periodic_{:03}", i);
        anyhow::ensure!(
            tag.as_str() == expected.as_str(),
            "periodic entry at position {i} has tag {tag:?}; expected \
             {expected:?} (zero-based, contiguous, :03 padded)"
        );
    }

    // At least one entry must be a real capture (non-placeholder).
    // Placeholders set every `*_unavailable` field; real captures
    // populate the maps Vec. A run where every boundary timed out
    // should be flagged so an operator notices the rendezvous
    // problem instead of treating the all-placeholder bridge as a
    // pass.
    let real_captures = periodic_entries
        .iter()
        .filter(|(_, report)| !report.maps.is_empty())
        .count();
    anyhow::ensure!(
        real_captures >= 1,
        "every periodic entry on the bridge is a placeholder \
         (empty .maps) — the freeze coordinator never produced a \
         real capture. Most commonly a parked-vCPU rendezvous \
         timeout repeated past the 2-consecutive abandon \
         threshold; check the trace for \
         'periodic capture abandoned'."
    );

    Ok(())
}

/// 10 s workload with periodic captures at scenario_start + 3 s,
/// 5 s, 7 s. The cgroup holds for the full duration so the
/// workload has live tasks across every boundary; without that
/// the per-CPU runnable_at scanner would log "no aged tasks" and
/// the captures would still happen (periodic capture is not gated
/// on the scanner) but the reports would be sparse.
#[ktstr_test(
    scheduler = KTSTR_SCHED,
    duration_s = 10,
    watchdog_timeout_s = 15,
    num_snapshots = 3,
    auto_repro = false,
    post_vm = assert_periodic_captures,
)]
fn periodic_capture_three_boundaries(ctx: &ktstr::scenario::Ctx) -> Result<AssertResult> {
    let steps = vec![Step {
        setup: vec![ctx.cgroup_def("cg_0")].into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    let mut result = execute_steps(ctx, steps)?;
    result.note("10s workload with num_snapshots=3 finished");
    Ok(result)
}

/// Host-side check that asserts the periodic-capture pipeline
/// fires at least once with a real capture (non-placeholder), and
/// that the bridge holds exactly `periodic_fired` entries. Same
/// rigour as `assert_periodic_captures` but parameterized on the
/// configured `num_snapshots` so multiple `#[ktstr_test]` entries
/// can share the verifier.
fn assert_at_least_one_real_capture(result: &VmResult, expected_target: u32) -> Result<()> {
    anyhow::ensure!(
        result.periodic_target == expected_target,
        "periodic_target must mirror num_snapshots = {}, got {}",
        expected_target,
        result.periodic_target,
    );
    anyhow::ensure!(
        result.periodic_fired >= 1,
        "periodic_fired must be at least 1, got {} of {}",
        result.periodic_fired,
        result.periodic_target,
    );
    let captured = result.snapshot_bridge.drain_ordered();
    let periodic_entries: Vec<_> = captured
        .iter()
        .filter(|(tag, _)| tag.starts_with("periodic_"))
        .collect();
    anyhow::ensure!(
        periodic_entries.len() == result.periodic_fired as usize,
        "bridge has {} periodic_* entries but periodic_fired = {}",
        periodic_entries.len(),
        result.periodic_fired,
    );
    let real_captures = periodic_entries
        .iter()
        .filter(|(_, report)| !report.maps.is_empty())
        .count();
    anyhow::ensure!(
        real_captures >= 1,
        "every periodic entry on the bridge is a placeholder (empty .maps) — \
         the freeze coordinator never produced a real capture"
    );
    Ok(())
}

fn assert_cgroup_parent_captures(result: &VmResult) -> Result<()> {
    // Environmental starvation gate: zero real captures under a
    // witnessed-contended host is a non-verdict (the readiness-gated
    // capture chain was starved past the workload window), not a
    // capture regression — SKIP instead of failing the assertions
    // below. A quiet-host zero-capture run still falls through and
    // fails with the specific diagnosis. See `periodic_starvation_gate`.
    ktstr::prelude::periodic_starvation_gate(result, 1)?;
    assert_at_least_one_real_capture(result, 2)
}

/// `cgroup_parent` set on the scheduler must NOT block the
/// periodic-capture pipeline. Locks in the decoupling: the prior
/// auto-inject of `--cell-parent-cgroup` from `cgroup_parent`
/// switched cell-aware schedulers into userspace_managed_cell_mode
/// and silently 0-fired the pipeline; with the decoupling in
/// `runtime::append_base_sched_args`'s `Absent` branch, the
/// scheduler's argv stays clean and the framework's periodic-
/// capture loop still gets real captures.
#[ktstr_test(
    scheduler = KTSTR_SCHED_WITH_CGPAR,
    duration_s = 6,
    watchdog_timeout_s = 12,
    num_snapshots = 2,
    auto_repro = false,
    post_vm = assert_cgroup_parent_captures,
)]
fn periodic_capture_with_cgroup_parent_e2e(ctx: &ktstr::scenario::Ctx) -> Result<AssertResult> {
    let steps = vec![Step {
        setup: vec![ctx.cgroup_def("cg_0")].into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    let mut result = execute_steps(ctx, steps)?;
    result.note("6s workload with cgroup_parent set + num_snapshots=2 finished");
    Ok(result)
}
