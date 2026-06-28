//! End-to-end proof that the IRQ-observability counters/rates flow through the
//! freeze -> read_sample -> phase_metric pipeline.
//!
//! Boots a VM with scx-ktstr + a NIC, runs a single Step of `NetTraffic`
//! workers (continuous AF_PACKET TX -> in-VMM loopback RX-completion IRQs),
//! takes `num_snapshots = 6` periodic freezes across the run, and asserts on
//! the HOST (`post_vm`) that the per-phase IRQ metrics resolved from
//! [`VmResult::phase_metric`] are PRESENT and rose:
//!
//! * `total_hardirqs` > 0 — cross-CPU sum of `kstat.irqs_sum` (freeze
//!   last-minus-first): the hardirq-counter wiring.
//! * `total_softirq_net_rx` > 0 — the NET_RX softirq vector index: the
//!   softirq-index wiring AND the NIC RX path.
//! * `hardirq_rate` > 0 — count / capture-window-seconds: the rate
//!   co-insertion + the wall-window denominator.
//!
//! `ok_or_else(absent -> Err)` + `ensure!(> 0)` is the skip-masks-bug guard: an
//! ABSENT metric FAILS loudly (it does not vacuously pass), and `> 0` proves the
//! counter actually advanced. The counters need `num_snapshots >= 2` so the
//! step bucket holds >= 2 freezes for the Counter last-minus-first delta
//! (`phase_counter_delta` is `None` for fewer than two finite samples).
//!
//! The PSI-irq run-level metrics (`psi_irq_full_avg10` / `total_irq_pressure_us`)
//! ARE asserted here via `result.run_metric` (which also pins that the run_metric
//! boundary folds the monitor-summary metrics into its ext map). `psi_irq_full_avg10`
//! must be PRESENT (proving `psi_system` resolved + the host-walk ran) and a sane
//! percent; `total_irq_pressure_us` must be > 0 — the sustained softirq load that
//! drives `total_softirq_net_rx` also accrues IRQ-full stall, so a present-but-
//! zero cumulative total is the wrong-`psi_system`-PA signature (a wrong PA reads
//! out-of-bounds 0). Together they prove the `.data`-global PA translation +
//! `psi_group` offset math read the live accumulator from a real kernel. The fold
//! math (mean / end-start delta / decode) is pinned separately by
//! `MonitorSummary::from_samples` unit tests. The PELT IRQ gauges (`avg_irq_util`
//! / `max_avg_irq_util`) and `avg_nr_running` remain unit-tested only — their
//! level is gate-/topology-dependent, flaky to pin to a value e2e.

use anyhow::{Result, anyhow, ensure};
use ktstr::assert::{AssertResult, Phase};
use ktstr::ktstr_test;
use ktstr::prelude::{BuiltinMetric, NetConfig, VmResult};
use ktstr::scenario::Ctx;
use ktstr::scenario::ops::{CgroupDef, HoldSpec, Step, execute_steps};
use ktstr::test_support::{Scheduler, SchedulerSpec};
use ktstr::workload::WorkType;

const KTSTR_SCHED: Scheduler =
    Scheduler::named("ktstr_sched").binary(SchedulerSpec::Discover("scx-ktstr"));

/// virtio-net with a deterministic locally-administered MAC. The last byte
/// differs from `net_traffic_e2e`'s so the two are unmistakable in traces; each
/// VM is isolated, so a collision would be harmless regardless.
const NET_TEST: NetConfig = NetConfig::DEFAULT.mac([0x52, 0x54, 0x00, 0x4e, 0x54, 0x02]);

/// Host-side check that the per-phase IRQ counters/rate resolved from the
/// freeze-driven phase buckets and rose under the NetTraffic load.
fn assert_irq_metrics(result: &VmResult) -> Result<()> {
    // >= 2 periodic freezes must land in the step bucket for the Counter
    // last-minus-first delta to be measurable. The single Step holds FULL, so
    // every periodic capture is stamped Phase::step(0).
    ensure!(
        result.periodic_fired >= 2,
        "only {} of {} periodic captures fired — need >= 2 in the step bucket \
         so the IRQ Counter delta (last - first) is measurable; a cold-cache \
         freeze-rendezvous stall can cut the sequence short",
        result.periodic_fired,
        result.periodic_target,
    );

    let step = Phase::step(0);

    // total_hardirqs: cross-CPU sum of kstat.irqs_sum, freeze last-minus-first.
    // ABSENT => the freeze never captured per_cpu_time (cpu_time-walker prereqs
    // unmet) or the bucket held < 2 freezes; either way FAIL, never SKIP.
    let hardirqs = result
        .phase_metric(step, BuiltinMetric::TotalHardirqs)
        .ok_or_else(|| {
            anyhow!(
                "total_hardirqs absent in Phase::step(0) — no per_cpu_time freeze \
                 capture, or fewer than two freezes landed in the bucket"
            )
        })?;
    ensure!(
        hardirqs > 0.0,
        "total_hardirqs must RISE across the NetTraffic window, got {hardirqs}"
    );

    // total_softirq_net_rx: the NET_RX softirq vector. The in-VMM loopback turns
    // each AF_PACKET TX kick into an RX completion -> NAPI -> NET_RX softirq, so
    // sustained traffic accrues a positive cross-CPU delta.
    let net_rx = result
        .phase_metric(step, BuiltinMetric::TotalSoftirqNetRx)
        .ok_or_else(|| {
            anyhow!(
                "total_softirq_net_rx absent in Phase::step(0) — softirq-index \
                 read_sample arm did not fold into the bucket"
            )
        })?;
    ensure!(
        net_rx > 0.0,
        "NET_RX softirqs must rise under NetTraffic loopback, got {net_rx}"
    );

    // hardirq_rate: total_hardirqs / capture-window-seconds. ABSENT means the
    // wall-window denominator co-insertion (phase_build) was missed.
    let rate = result
        .phase_metric(step, BuiltinMetric::HardirqRate)
        .ok_or_else(|| {
            anyhow!(
                "hardirq_rate absent in Phase::step(0) — the wall-window \
                 denominator (total_phase_wall_sec) was not co-inserted"
            )
        })?;
    ensure!(rate > 0.0, "hardirq_rate must be > 0, got {rate}");

    // System-wide PSI-irq pressure, host-walked from the global `psi_system`
    // per monitor sample, folded run-level in MonitorSummary, and surfaced via
    // the run-level metric accessor (result.run_metric). The unit fold tests pin
    // the mean / end-start-delta / decode math on synthetic samples; THIS proves
    // the host-walk against a real kernel: `psi_system` resolves (CONFIG_PSI=y in
    // ktstr.kconfig) and the `.data`-global PA translation + `psi_group` offset
    // math read the LIVE accumulator. Reading via run_metric (not
    // result.monitor.summary directly) ALSO pins that the run_metric boundary
    // folds the monitor-summary metrics into its ext map. ABSENT FAILS loudly
    // (psi_system unresolved / PSI_IRQ_FULL not in BTF), never SKIPs.
    //
    // Present (psi_system resolved + PSI_IRQ_FULL in BTF + >= 1 data_valid
    // sample) and a sane percent. decode_avg10_percent clamps the upper bound, so
    // [0,100] is a floor/sanity here; the wrong-PA signal is `total > 0` below.
    let psi_avg10 = result.run_metric(BuiltinMetric::PsiIrqFullAvg10).ok_or_else(|| {
        anyhow!(
            "psi_irq_full_avg10 absent — psi_system unresolved or PSI_IRQ_FULL \
             missing from BTF (CONFIG_PSI / CONFIG_IRQ_TIME_ACCOUNTING off)"
        )
    })?;
    ensure!(
        (0.0..=100.0).contains(&psi_avg10),
        "psi_irq_full_avg10 must be a percent in [0,100], got {psi_avg10}"
    );
    // > 0, not just present. PSI IRQ-full accrues PER-CPU in psi_account_irqtime
    // (kernel/sched/psi.c:1006-1046): on each tick/schedule it adds the CPU's
    // irq-time delta to psi_system's PSI_IRQ_FULL bucket whenever a real task
    // (curr->pid != 0) was running — it is NOT gated on a multi-CPU simultaneous
    // stall. This test's load is maximal-continuous (2 CPU-bound NetTraffic
    // workers, interval_us=0 → constant NET_RX softirq), so a real worker runs on
    // each CPU while it services softirq → PSI_IRQ_FULL accrues every tick and the
    // cumulative total grows large over 15 s — deterministic, not a transient a
    // freeze might miss. A wrong psi_system PA reads out-of-bounds -> 0, so a
    // present-but-zero cumulative total is the wrong-PA signature (caught here
    // where the clamped avg10 range cannot). NOTE: > 0 relies on this sustained
    // load; revisit the assertion if the workload is lightened/idled.
    let psi_total = result.run_metric(BuiltinMetric::TotalIrqPressureUs).ok_or_else(|| {
        anyhow!("total_irq_pressure_us absent under the same PSI gate as avg10")
    })?;
    ensure!(
        psi_total > 0.0 && psi_total.is_finite(),
        "total_irq_pressure_us must RISE under the NetTraffic softirq load \
         (present-but-zero signals a wrong psi_system PA reading 0), got {psi_total}"
    );

    Ok(())
}

/// scx-ktstr + a NIC; one Step of continuous NetTraffic so the periodic freezes
/// land mid-traffic. `num_snapshots = 6` over a 15 s hold gives the step bucket
/// several freezes (>= 2 required for the Counter delta). Mirrors the proven
/// periodic-capture e2e shape (`api_gaps_e2e` / `periodic_capture_e2e`) for the
/// freeze pipeline and `net_traffic_e2e` for the IRQ generator.
#[ktstr_test(
    scheduler = KTSTR_SCHED,
    llcs = 1,
    cores = 2,
    threads = 1,
    network = NET_TEST,
    duration_s = 15,
    watchdog_timeout_s = 45,
    num_snapshots = 6,
    auto_repro = false,
    post_vm = assert_irq_metrics,
)]
fn irq_counters_rise_under_net_traffic(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![Step {
        setup: vec![
            CgroupDef::named("irq_load")
                .workers(2)
                .work_type(WorkType::NetTraffic {
                    interval_us: 0,
                    frame_bytes: 60,
                }),
        ]
        .into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    execute_steps(ctx, steps)
}
