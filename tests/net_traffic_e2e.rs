//! End-to-end proof that [`WorkType::NetTraffic`] drives real virtio-net
//! interrupts inside a VM.
//!
//! Boots a small VM with a NIC attached (`network = ...`), direct-spawns a
//! `NetTraffic` workload through the production [`WorkloadHandle`] pipeline,
//! and asserts two things across the run: the workers actually sent frames
//! (`work_units > 0`), and the NIC's `/proc/interrupts` count rose. The
//! v0 virtio-net backend is in-VMM loopback, so each AF_PACKET TX kick the
//! workload issues echoes into RX and fires the RX-completion IRQ — a rising
//! interrupt count is the end-to-end proof the workload generated genuine
//! per-CPU hardirq + NAPI softirq load (the IRQ-side metric wiring is a
//! separate task; this pins the GENERATOR).
//!
//! The NIC IRQ is pinned to CPU 0 so the count is read from a single,
//! stable column; under sustained traffic the bulk of completions land
//! there after the affinity write takes effect. Arch-agnostic: AF_PACKET TX
//! + the virtio RX IRQ work the same on x86 (IOAPIC/MSI) and arm64 (GIC) —
//! only the >255-APIC ext-dest leg in `wide_smp_net_irq_e2e` is x86-gated.

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

/// virtio-net with a deterministic locally-administered MAC (the `network =`
/// macro arg needs a const-evaluable path; `NetConfig::DEFAULT.mac(..)` is
/// the const-fn chain).
const NET_TEST: NetConfig = NetConfig::DEFAULT.mac([0x52, 0x54, 0x00, 0x4e, 0x54, 0x01]);

#[ktstr_test(
    llcs = 1,
    cores = 2,
    threads = 1,
    network = NET_TEST,
    no_perf_mode,
    duration_s = 15,
    watchdog_timeout_s = 45
)]
fn net_traffic_raises_nic_irq_count(_ctx: &Ctx) -> Result<AssertResult> {
    let iface = virtio_net_iface()?;
    let (irq, dev) = virtio_net_irq(&iface)?;
    // Pin the NIC IRQ to CPU 0 so before/after counts read one stable column.
    pin_irq_to_cpu(irq, 0)?;
    let before = irq_count(irq, 0)?;

    // Two NetTraffic workers, continuous (interval_us = 0) so they generate
    // maximum TX-kick / IRQ load, minimum frame (60). Direct-spawn through
    // the production pipeline (no scheduler needed — EEVDF is fine; the test
    // is about traffic generation, not scheduling).
    let cfg = WorkloadConfig {
        num_workers: 2,
        work_type: WorkType::NetTraffic {
            interval_us: 0,
            frame_bytes: 60,
        },
        ..Default::default()
    };
    let mut handle = WorkloadHandle::spawn(&cfg)?;
    handle.start();
    std::thread::sleep(Duration::from_secs(3));
    let reports = handle.stop_and_collect();
    let after = irq_count(irq, 0)?;

    let sent: u64 = reports.iter().map(|r| r.work_units).sum();
    eprintln!(
        "NET_TRAFFIC iface={iface} irq={irq} ({dev}) workers={} frames_sent={sent} \
         irq_count {before}->{after}",
        reports.len()
    );
    // The workers must have actually sent frames — a no-op (no NIC) reports
    // work_units == 0, which would fail here loudly rather than vacuously.
    ensure!(
        sent > 0,
        "NetTraffic workers sent no frames (work_units=0 across {} workers); \
         the NIC was not driven",
        reports.len()
    );
    // The send loop drove real virtio-net interrupts: the in-VMM loopback
    // turns each TX kick into an RX-completion IRQ.
    ensure!(
        after > before,
        "NetTraffic did not raise the NIC IRQ {irq} ({dev}) count on cpu 0 \
         (before={before} after={after}); the AF_PACKET TX-loopback did not \
         deliver RX-completion interrupts"
    );
    Ok(AssertResult::pass())
}

/// With NO NIC attached (no `network = ...`), the guest has only `lo`, so
/// NetTraffic must be a LOUD no-op: the workers find no interface, warn once,
/// and report `work_units == 0` (the contract the send-path e2e's `sent > 0`
/// guard and the starvation gate rely on) while the dispatch loop still
/// iterates. This is the CI-runnable invariant pairing the host-gated
/// send-path e2e above, and the only test that exercises the no-NIC branch.
#[ktstr_test(
    llcs = 1,
    cores = 2,
    threads = 1,
    no_perf_mode,
    duration_s = 12,
    watchdog_timeout_s = 40
)]
fn net_traffic_no_nic_is_loud_noop(_ctx: &Ctx) -> Result<AssertResult> {
    let cfg = WorkloadConfig {
        num_workers: 2,
        work_type: WorkType::NetTraffic {
            interval_us: 0,
            frame_bytes: 60,
        },
        ..Default::default()
    };
    let mut handle = WorkloadHandle::spawn(&cfg)?;
    handle.start();
    std::thread::sleep(Duration::from_secs(2));
    let reports = handle.stop_and_collect();
    ensure!(!reports.is_empty(), "no NetTraffic worker reports came back");
    let work: u64 = reports.iter().map(|r| r.work_units).sum();
    let iters: u64 = reports.iter().map(|r| r.iterations).sum();
    eprintln!(
        "NET_TRAFFIC_NOOP workers={} work_units={work} iterations={iters}",
        reports.len()
    );
    ensure!(
        work == 0,
        "NetTraffic with no NIC must be a no-op (work_units==0); got {work} \
         across {} workers — the no-NIC guard regressed",
        reports.len()
    );
    ensure!(iters > 0, "the dispatch loop should still have iterated");
    Ok(AssertResult::pass())
}
