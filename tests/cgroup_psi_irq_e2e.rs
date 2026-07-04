//! End-to-end proof that the per-cgroup PSI-irq host-walk flows through the
//! freeze -> cgroup-hierarchy walk -> per-phase fold -> run-level metric pipeline
//! against a REAL kernel — the real-VM walk proof the capture layer deferred.
//!
//! Boots a VM with scx-ktstr + a NIC and runs a single Step whose setup
//! declares TWO sibling leaf cgroups (`irq_load_a` / `irq_load_b`, each at
//! `/sys/fs/cgroup/ktstr/<name>` under the workload root), each running
//! continuous `NetTraffic` workers (AF_PACKET TX -> in-VMM loopback
//! RX-completion IRQs). `num_snapshots = 6` periodic freezes each descend the
//! workload-root cgroup hierarchy, read every leaf's `cgroup->psi`
//! PSI_IRQ_FULL, and stamp the raw `{cgroup_kva, total_ns, avg10_raw}` rows.
//! The host (`post_vm`) asserts the per-cgroup metrics that the per-phase fold
//! derives and auto-folds run-level:
//!
//! * `max_cgroup_irq_pressure` PRESENT + > 0 — the busiest leaf's PSI_IRQ_FULL
//!   stall DELTA over the phase (decoded µs). PRESENT proves the walk descended
//!   from `cgrp_dfl_root`, matched the workload-root path, reached a workload
//!   leaf, and read its live `cgroup->psi` accumulator; > 0 proves the delta is
//!   a real rising counter (a wrong leaf-cgroup PA reads out-of-bounds 0). This
//!   is the per-cgroup analog of `irq_metrics_e2e`'s system-wide
//!   `total_irq_pressure_us > 0`, on the same sustained-load argument:
//!   psi_account_irqtime (kernel/sched/psi.c:1006-1046) charges each CPU's
//!   irq-time delta to the running task's cgroup + ancestors, so a leaf running
//!   a NetTraffic worker accrues PSI_IRQ_FULL every tick.
//! * `max_cgroup_psi_irq_avg10` PRESENT + in [0,100] — the worst leaf's avg10
//!   IRQ-full GAUGE (decoded percent). PRESENT proves the same walk read the
//!   `avg[]` array; the range is a decode sanity floor.
//! * `max_cgroup_irq_pressure_concentration` PRESENT and at least 1.0 — the
//!   busiest-leaf delta / mean-leaf delta. PRESENT requires at least 2 reporting
//!   leaves (the two declared cgroups, both present in every freeze) with a
//!   positive mean; 1.0 is the structural floor (the max is at least the mean
//!   for non-negative deltas). This pins that the walk distinguishes the two
//!   leaves by `cgroup_kva` (a single merged leaf would omit the concentration).
//!
//! `ok_or_else(absent -> Err)` is the skip-masks-bug guard: an ABSENT metric
//! FAILS loudly (psi_cgroups off / `cgrp_dfl_root` unresolved / the
//! CgroupWalkOffsets not in BTF / the workload-root path not matched), never
//! vacuously SKIPs. The delta metrics need `num_snapshots >= 2` so the step
//! bucket holds >= 2 freezes for the per-leaf last-minus-first delta.
//!
//! The fold math (per-leaf delta / cgroup_kva intersection / max/mean / decode)
//! is pinned separately by the `fold_per_cgroup_psi` unit tests on synthetic
//! samples; THIS proves the host-walk against a real kernel.

use anyhow::{Result, anyhow, ensure};
use ktstr::assert::AssertResult;
use ktstr::ktstr_test;
use ktstr::prelude::{BuiltinMetric, NetConfig, VmResult};
use ktstr::scenario::Ctx;
use ktstr::scenario::ops::{CgroupDef, HoldSpec, Step, execute_steps};
use ktstr::test_support::{Scheduler, SchedulerSpec};
use ktstr::workload::WorkType;

const KTSTR_SCHED: Scheduler =
    Scheduler::named("ktstr_sched").binary(SchedulerSpec::Discover("scx-ktstr"));

/// virtio-net with a deterministic locally-administered MAC. The last byte (0x03)
/// matches `irq_steer_e2e`'s `NET_TEST_STEER`; each VM is isolated, so the
/// collision is harmless.
const NET_TEST: NetConfig = NetConfig::DEFAULT.mac([0x52, 0x54, 0x00, 0x4e, 0x54, 0x03]);

/// Continuous NetTraffic: zero inter-frame interval, min Ethernet frame — the
/// same maximal-load shape `irq_metrics_e2e` uses to drive sustained PSI_IRQ_FULL
/// accrual, declared once and placed in each leaf cgroup.
const IRQ_WORK: WorkType = WorkType::NetTraffic {
    interval_us: 0,
    frame_bytes: 60,
};

/// Host-side check that the per-cgroup PSI-irq metrics resolved from the
/// freeze-driven cgroup walk and rose under the two-cgroup NetTraffic load.
fn assert_cgroup_psi_metrics(result: &VmResult) -> Result<()> {
    // Coverage guard: the per-leaf delta needs >= 2 freezes that actually
    // captured per-cgroup PSI. periodic_fired counts ATTEMPTS — it includes
    // rendezvous-timeout placeholders and dump-degraded reports, both of which
    // carry an empty cgroup_psi — so guarding on it would pass on >= 2 empty
    // fires and then fail at the metric ok_or_else below with a misleading
    // PSI/BTF diagnostic. Count the freezes whose cgroup_psi is non-empty (the
    // exact input the fold differences) so a cold-cache freeze-rendezvous stall
    // surfaces THIS coverage diagnostic, not a wrong-cause one. periodic_series()
    // is cache-backed and idempotent, so this composes with run_metric's own use.
    let psi_captures = result
        .periodic_series()
        .iter_samples()
        .filter(|s| !s.snapshot.cgroup_psi().is_empty())
        .count();
    ensure!(
        psi_captures >= 2,
        "only {psi_captures} of {} periodic captures carried per-cgroup PSI — need \
         >= 2 with non-empty cgroup_psi so the per-leaf last-minus-first delta is \
         measurable; a cold-cache freeze-rendezvous stall (placeholder/degraded \
         captures carry no PSI) can cut the data-bearing sequence short",
        result.periodic_target,
    );

    // The busiest leaf's PSI_IRQ_FULL stall delta (µs). ABSENT => the walk found
    // no workload leaf (psi_cgroups off / cgrp_dfl_root unresolved / workload-root
    // path unmatched / CgroupWalkOffsets missing from BTF) -> FAIL, never SKIP.
    let pressure = result
        .run_metric(BuiltinMetric::MaxCgroupIrqPressure)
        .ok_or_else(|| {
            anyhow!(
                "max_cgroup_irq_pressure absent — the per-cgroup PSI walk captured \
                 no workload leaf (psi_cgroups off, cgrp_dfl_root unresolved, the \
                 workload-root path unmatched, or CgroupWalkOffsets missing from BTF)"
            )
        })?;
    // > 0: a leaf running a NetTraffic worker accrues PSI_IRQ_FULL every tick
    // (psi_account_irqtime charges the CPU irq-time delta to the task's cgroup),
    // so the busiest leaf's cumulative stall rises across the window. A
    // present-but-zero delta is the wrong-leaf-PA signature (an out-of-bounds read
    // returns 0). NOTE: > 0 relies on the sustained two-cgroup load; revisit if
    // the workload is lightened or idled.
    ensure!(
        pressure > 0.0 && pressure.is_finite(),
        "max_cgroup_irq_pressure must RISE under the per-cgroup NetTraffic load \
         (present-but-zero signals a wrong leaf-cgroup PA reading 0), got {pressure}"
    );

    // The worst leaf's avg10 IRQ-full gauge (percent). Present under the same PSI
    // gate; the decode clamps the upper bound, so [0,100] is a sanity floor.
    let avg10 = result
        .run_metric(BuiltinMetric::MaxCgroupPsiIrqAvg10)
        .ok_or_else(|| {
            anyhow!("max_cgroup_psi_irq_avg10 absent under the same per-cgroup PSI gate")
        })?;
    ensure!(
        (0.0..=100.0).contains(&avg10),
        "max_cgroup_psi_irq_avg10 must be a percent in [0,100], got {avg10}"
    );

    // Busiest-leaf / mean-leaf concentration. PRESENT requires >= 2 reporting
    // leaves (the two declared cgroups, present in every freeze) with a positive
    // mean — so its presence proves the walk distinguished the two leaves by
    // cgroup_kva (a single merged leaf, or a < 2-leaf intersection, omits it).
    // >= 1.0 is the structural floor (max >= mean for non-negative deltas).
    let conc = result
        .run_metric(BuiltinMetric::MaxCgroupIrqPressureConcentration)
        .ok_or_else(|| {
            anyhow!(
                "max_cgroup_irq_pressure_concentration absent — the walk found < 2 \
                 reporting leaves (the two NetTraffic cgroups should both report) or \
                 a zero mean delta"
            )
        })?;
    ensure!(
        conc >= 1.0 - 1e-9 && conc.is_finite(),
        "max_cgroup_irq_pressure_concentration is max/mean, structurally >= 1.0, \
         got {conc}"
    );

    Ok(())
}

/// scx-ktstr + a NIC; one Step whose setup declares two sibling leaf cgroups,
/// each running continuous NetTraffic so every periodic freeze descends the
/// workload-root hierarchy onto two PSI-bearing leaves. `num_snapshots = 6` over
/// a 15 s hold gives the step bucket several freezes (>= 2 required for the
/// per-leaf delta). Mirrors `irq_metrics_e2e`'s freeze/NIC shape with the load
/// split across two cgroups.
#[ktstr_test(
    scheduler = KTSTR_SCHED,
    llcs = 1,
    cores = 2,
    threads = 1,
    networks = [NET_TEST],
    duration_s = 15,
    watchdog_timeout_s = 45,
    num_snapshots = 6,
    auto_repro = false,
    post_vm = assert_cgroup_psi_metrics,
)]
fn cgroup_psi_irq_pressure_rises_across_leaves(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![Step {
        setup: vec![
            CgroupDef::named("irq_load_a")
                .workers(1)
                .work_type(IRQ_WORK),
            CgroupDef::named("irq_load_b")
                .workers(1)
                .work_type(IRQ_WORK),
        ]
        .into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    execute_steps(ctx, steps)
}
