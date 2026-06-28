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
//! The run-level gauges (`avg_irq_util` / `max_avg_irq_util`) are NOT asserted
//! here: like the pre-existing `avg_nr_running`, run-level-ext metrics have no
//! in-test `VmResult` reader (they are sidecar/perf-delta-compare-only). They
//! are covered by a `MonitorSummary::from_samples` unit test instead; an
//! in-test run-level accessor is a separate follow-up.

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
