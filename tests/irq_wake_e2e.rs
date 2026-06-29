//! End-to-end proof that [`WorkType::IrqWake`] wakes a blocked task from
//! NET_RX **softirq** context inside a VM.
//!
//! Boots a small VM with a NIC attached (`network = ...`), direct-spawns a
//! paired `IrqWake` workload through the production [`WorkloadHandle`] pipeline,
//! and asserts across the run: the receiver was woken + scheduled (its
//! `wake_latencies_ns` reservoir is non-empty), and the NIC's
//! `/proc/interrupts` count rose. The pos==0 worker sends self-addressed
//! AF_PACKET frames; the v0 in-VMM loopback echoes each into RX and fires the
//! RX-completion IRQ, which schedules NAPI; the NET_RX softirq's
//! `packet_rcv` → `sock_def_readable` → `wake_up_interruptible_sync_poll` then
//! `ttwu`s the pos==1 receiver blocked in `recvfrom`. A non-empty wake reservoir
//! is the end-to-end proof a softirq-context wake scheduled the task — the wake
//! class [`WorkType::TimerLatency`] (hardirq hrtimer) and
//! [`WorkType::NetTraffic`] (wakee-less sender) do not exercise.

use anyhow::{Result, ensure};
use ktstr::assert::AssertResult;
use ktstr::ktstr_test;
use ktstr::prelude::NetConfig;
use ktstr::scenario::Ctx;
use ktstr::workload::{WorkType, WorkloadConfig, WorkloadHandle};
use std::time::Duration;

#[path = "common/wide_smp_irq.rs"]
mod wide_smp_irq;
use wide_smp_irq::{irq_count, pin_irq_to_cpu, virtio_net_iface, virtio_net_irq};

/// virtio-net with a deterministic locally-administered MAC, distinct from the
/// `net_traffic_e2e` MAC so the two never alias if scheduled on the same host.
const NET_TEST_IRQ: NetConfig = NetConfig::DEFAULT.mac([0x52, 0x54, 0x00, 0x4e, 0x54, 0x02]);

#[ktstr_test(
    llcs = 1,
    cores = 2,
    threads = 1,
    network = NET_TEST_IRQ,
    no_perf_mode,
    duration_s = 15,
    watchdog_timeout_s = 45
)]
fn irq_wake_records_softirq_wakes(_ctx: &Ctx) -> Result<AssertResult> {
    let iface = virtio_net_iface()?;
    let (irq, dev) = virtio_net_irq(&iface)?;
    // Pin the NIC IRQ to CPU 0 so before/after counts read one stable column.
    pin_irq_to_cpu(irq, 0)?;
    let before = irq_count(irq, 0)?;

    // One sender/receiver pair (group size 2). A 200µs send pace lets the
    // receiver drain + block between frames so each frame is a genuine
    // empty-queue block woken by the next NET_RX softirq — populating the wake
    // reservoir — while still driving a steady IRQ rate.
    let cfg = WorkloadConfig {
        num_workers: 2,
        work_type: WorkType::IrqWake {
            interval_us: 200,
            frame_bytes: 60,
        },
        ..Default::default()
    };
    let mut handle = WorkloadHandle::spawn(&cfg)?;
    handle.start();
    std::thread::sleep(Duration::from_secs(3));
    let reports = handle.stop_and_collect();
    let after = irq_count(irq, 0)?;

    let work: u64 = reports.iter().map(|r| r.work_units).sum();
    let wake_samples: usize = reports.iter().map(|r| r.wake_latencies_ns.len()).sum();
    eprintln!(
        "IRQ_WAKE iface={iface} irq={irq} ({dev}) workers={} work_units={work} \
         wake_samples={wake_samples} irq_count {before}->{after}",
        reports.len()
    );
    // Two-part proof: the rising IRQ count (asserted below) is the "NET_RX
    // softirq fired" evidence; a non-empty wake reservoir is the "receiver was
    // scheduled to run on the delivered frame" evidence. A no-NIC no-op leaves
    // both empty, failing here.
    ensure!(
        wake_samples > 0,
        "IrqWake recorded no wake samples across {} workers; the softirq wake did \
         not schedule the blocked receiver (or no NIC was attached)",
        reports.len()
    );
    // The send loop drove real virtio-net interrupts (the in-VMM loopback turns
    // each TX kick into an RX-completion IRQ — the source of the NET_RX softirq).
    ensure!(
        after > before,
        "IrqWake did not raise the NIC IRQ {irq} ({dev}) count on cpu 0 \
         (before={before} after={after}); no RX-completion interrupts were \
         delivered, so no softirq wake could occur"
    );
    Ok(AssertResult::pass())
}

/// With NO NIC attached (no `network = ...`), the guest has only `lo`, so both
/// sides of `IrqWake` are a LOUD no-op: the sender finds no interface and the
/// receiver cannot open its socket — both warn once and report
/// `work_units == 0` while the dispatch loop still iterates. CI-runnable
/// invariant pairing the host-gated test above; the only no-NIC-branch coverage.
#[ktstr_test(
    llcs = 1,
    cores = 2,
    threads = 1,
    no_perf_mode,
    duration_s = 12,
    watchdog_timeout_s = 40
)]
fn irq_wake_no_nic_is_loud_noop(_ctx: &Ctx) -> Result<AssertResult> {
    let cfg = WorkloadConfig {
        num_workers: 2,
        work_type: WorkType::IrqWake {
            interval_us: 200,
            frame_bytes: 60,
        },
        ..Default::default()
    };
    let mut handle = WorkloadHandle::spawn(&cfg)?;
    handle.start();
    std::thread::sleep(Duration::from_secs(2));
    let reports = handle.stop_and_collect();
    ensure!(!reports.is_empty(), "no IrqWake worker reports came back");
    let work: u64 = reports.iter().map(|r| r.work_units).sum();
    let iters: u64 = reports.iter().map(|r| r.iterations).sum();
    eprintln!(
        "IRQ_WAKE_NOOP workers={} work_units={work} iterations={iters}",
        reports.len()
    );
    ensure!(
        work == 0,
        "IrqWake with no NIC must be a no-op (work_units==0); got {work} across \
         {} workers — the no-NIC guard regressed",
        reports.len()
    );
    ensure!(iters > 0, "the dispatch loop should still have iterated");
    Ok(AssertResult::pass())
}
