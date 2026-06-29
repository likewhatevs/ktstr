//! End-to-end proof that [`Op::SteerIrq`] re-routes a NIC's IRQ to a
//! chosen CPU inside a VM.
//!
//! The IRQ-cluster capstone: it composes the three IRQ halves —
//! steering ([`Op::SteerIrq`], this task), generating
//! ([`WorkType::NetTraffic`], the AF_PACKET traffic source), and
//! observing (the per-CPU `/proc/interrupts` counts). A two-step
//! scenario steers the virtio-net IRQ to CPU 1 BEFORE any traffic
//! (step 0, a zero-hold op-only step), then floods NetTraffic across
//! the full hold (step 1). The in-VMM loopback turns each AF_PACKET
//! TX kick into an RX-completion interrupt on that IRQ, so after the
//! run CPU 1 holds the bulk of the NIC's per-CPU count while CPU 0
//! holds only the handful of pre-steer boot interrupts — proving the
//! `smp_affinity_list` write actually reprogrammed delivery.
//!
//! The selector is [`IrqSelector::by_label`] resolved against the
//! guest's real `/proc/interrupts`, exercising the in-guest
//! label→IRQ resolution path end-to-end. ktstr boots the NIC on the
//! virtio-MMIO transport, which registers ONE shared IRQ whose action
//! name is the bare device basename (e.g. `virtio1`), so the single
//! IRQ resolves uniquely AND carries every RX completion — the
//! `c1 > c0` assertion below relies on that single-queue shape.
//! (Were virtio-net multiqueue + per-queue MSI-X added, a single
//! steered IRQ would carry only its queue's share and this assertion
//! would need revisiting.) The label-parse edge
//! cases and the offline-CPU / unresolvable-label bail-before-write
//! guards are pinned by the CI-runnable unit tests in
//! `src/scenario/ops/tests.rs` (`resolve_irq_label_*`,
//! `op_steer_irq_*_bails_before_write`); this test is the booted-VM
//! half.

use anyhow::{Result, ensure};
use ktstr::assert::AssertResult;
use ktstr::ktstr_test;
use ktstr::prelude::NetConfig;
use ktstr::scenario::Ctx;
use ktstr::scenario::ops::{CgroupDef, HoldSpec, IrqSelector, Op, Step, execute_steps};
use ktstr::workload::WorkType;
use std::time::Duration;

#[path = "common/wide_smp_irq.rs"]
mod wide_smp_irq;
use wide_smp_irq::{irq_count, virtio_net_iface, virtio_net_irq};

/// virtio-net with a deterministic locally-administered MAC, distinct from the
/// other IRQ/NetTraffic e2e MACs so the VMs never alias on a shared host.
const NET_TEST_STEER: NetConfig = NetConfig::DEFAULT.mac([0x52, 0x54, 0x00, 0x4e, 0x54, 0x03]);

/// Steer the NIC IRQ to CPU 1, generate NetTraffic, and assert CPU 1
/// holds the majority of the NIC's per-CPU interrupt count.
#[ktstr_test(
    llcs = 1,
    cores = 2,
    threads = 1,
    network = NET_TEST_STEER,
    no_perf_mode,
    duration_s = 15,
    watchdog_timeout_s = 45
)]
fn irq_steer_concentrates_on_target_cpu(ctx: &Ctx) -> Result<AssertResult> {
    let iface = virtio_net_iface()?;
    let (irq, dev) = virtio_net_irq(&iface)?;

    const TARGET_CPU: usize = 1;
    let steps = vec![
        // Step 0: steer the NIC IRQ to CPU 1 before any traffic. The
        // op-only step holds for zero time (the steer is a one-shot
        // file write). ByLabel resolves `dev` (the /proc/interrupts
        // action name) to the IRQ number in-guest.
        Step::with_op(
            Op::steer_irq(IrqSelector::by_label(dev.clone()), TARGET_CPU),
            HoldSpec::Fixed(Duration::ZERO),
        ),
        // Step 1: flood NetTraffic for the full hold. Every
        // RX-completion IRQ now lands on the steered CPU.
        Step::with_defs(
            vec![
                CgroupDef::named("steer_load")
                    .workers(2)
                    .work_type(WorkType::NetTraffic {
                        interval_us: 0,
                        frame_bytes: 60,
                    }),
            ],
            HoldSpec::FULL,
        ),
    ];
    // A steer bail (offline CPU, unresolvable label, un-steerable IRQ)
    // would surface as an Err here — the run failing IS the steer
    // failing. A non-pass scenario verdict (e.g. a liveness failure
    // during the hold) propagates before the IRQ-concentration check.
    let verdict = execute_steps(ctx, steps)?;
    if !verdict.is_pass() {
        return Ok(verdict);
    }

    let c0 = irq_count(irq, 0)?;
    let c1 = irq_count(irq, TARGET_CPU)?;
    eprintln!("IRQ_STEER iface={iface} irq={irq} ({dev}) cpu0={c0} cpu{TARGET_CPU}={c1}");

    // CPU 1 was the steer target, so the sustained NetTraffic
    // RX-completion interrupts must land there, not on CPU 0. c0 holds
    // only the few pre-steer boot interrupts; c1 holds the flood.
    ensure!(
        c1 > c0,
        "steered CPU {TARGET_CPU} ({c1}) must hold more of IRQ {irq}'s count \
         than CPU 0 ({c0}); the smp_affinity_list write did not re-route \
         delivery (or no RX-completion interrupts were generated)"
    );
    ensure!(
        c1 > 0,
        "the steered CPU recorded zero interrupts on IRQ {irq}; NetTraffic \
         generated no RX-completion interrupts to steer"
    );
    Ok(AssertResult::pass())
}
