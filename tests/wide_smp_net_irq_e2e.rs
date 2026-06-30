//! End-to-end: a virtio-net device IRQ delivered to a vCPU whose APIC ID
//! exceeds 255, over the MSI ext-dest-id (`address_hi.destid_8_31`) path.
//!
//! Companion to `wide_smp_device_irq_e2e.rs` (virtio-blk). The >255
//! ext-dest routing is device-agnostic (the same userspace-IOAPIC +
//! `KVM_SET_GSI_ROUTING` path for every virtio device), so virtio-blk
//! already proves the route resolves a >255 destination. This adds the
//! virtio-net device-type leg: it boots a sparse topology whose APIC IDs
//! reach 433, attaches a NIC via the `networks = [...]` test attribute, pins the
//! virtio-net IRQ to a vCPU with APIC ID >= 256, drives the in-VMM
//! TX-loopback over an AF_PACKET raw socket, and asserts that vCPU's
//! interrupt count rose — proving virtio-net's own RX-completion IRQ
//! routes through the >255 destination encoding too.
//!
//! Why AF_PACKET: the v0 virtio-net backend is in-VMM loopback (TX bytes
//! echoed straight into RX). IP-layer self-traffic never reaches the NIC
//! (the guest's `RTN_LOCAL` route sends it to `lo`), so only an AF_PACKET
//! raw socket bound to the interface generates a real virtio TX kick and
//! observes the resulting RX interrupt (see `vmm::net_config`).
//!
//! Topology: 14 LLCs x 9 cores x 2 threads = 252 vCPUs. The sparse
//! APIC-ID encoding gives a max APIC ID of 433, so vCPUs in the upper LLCs
//! have APIC ID >= 256 without needing a > 256-vCPU host (same topology as
//! the virtio-blk sibling).
//!
//! Run: cargo run --bin cargo-ktstr -- ktstr test --kernel ../linux \
//!        -- -E 'test(net_irq_delivers_to_apic_id_above_255)' \
//!        --success-output immediate

// The >255-APIC-ID MSI ext-dest-id path is x86-only: aarch64 uses the GIC
// (SPIs; no APIC/IOAPIC/MSI destination) and /proc/cpuinfo has no `apicid`
// field, so find_apic_above_255 returns Err before any I/O. The arm64
// high-vCPU device-IRQ equivalent (GIC SPI delivery) is a separate test.
#![cfg(target_arch = "x86_64")]

use anyhow::{Result, ensure};
use ktstr::assert::AssertResult;
use ktstr::ktstr_test;
use ktstr::prelude::NetConfig;
use ktstr::scenario::Ctx;

#[path = "common/wide_smp_irq.rs"]
mod wide_smp_irq;
use wide_smp_irq::{
    find_apic_above_255, iface_ifindex, iface_mac, irq_count, open_and_drive, pin_irq_to_cpu,
    recv_looped_frame, virtio_net_iface, virtio_net_irq,
};

/// virtio-net with a deterministic locally-administered MAC. Const because
/// `NetConfig::default` is not const and the `networks = [...]` macro arg needs
/// const-evaluable paths; `NetConfig::DEFAULT.mac(..)` is the const-fn chain.
const KTSTR_NET: NetConfig = NetConfig::DEFAULT.mac([0x52, 0x54, 0x00, 0x12, 0x34, 0x56]);

#[ktstr_test(
    llcs = 14,
    cores = 9,
    threads = 2,
    networks = [KTSTR_NET],
    no_perf_mode,
    duration_s = 4
)]
fn net_irq_delivers_to_apic_id_above_255(_ctx: &Ctx) -> Result<AssertResult> {
    // No vCPU-count gate: the sparse APIC-ID encoding mints IDs up to 433
    // from this 252-vCPU topology (see the module doc), so the requirement
    // is "an APIC ID > 255 exists", not "> 254 vCPUs". find_apic_above_255
    // reads the guest's actual /proc/cpuinfo and errors if none reach 256 --
    // the correct runtime gate. (A > 254-vCPU count is the right invariant
    // for the dense sibling tests -- 16*16*1 = 256, max APIC ID 255 -- but
    // wrong for this sparse topology: 14*9*2 = 252 < 254, so it failed every
    // run that was not host-skipped.)

    let iface = virtio_net_iface()?;
    let ifindex = iface_ifindex(&iface)?;
    let mac = iface_mac(&iface)?;
    let (irq, dev_name) = virtio_net_irq(&iface)?;

    // Pick a vCPU with APIC ID >= 256 and pin the NIC IRQ to it (x2APIC
    // physical mode → the RTE carries that exact APIC ID, exercising the
    // >255 ext-dest encoding for virtio-net specifically).
    let (target_cpu, target_apic) = find_apic_above_255()?;
    pin_irq_to_cpu(irq, target_cpu)?;

    // Baseline after pinning, drive the loopback, then re-read. 64 frames
    // give margin for x86's lazy IRQ-affinity migration (a fresh
    // smp_affinity write takes effect at the next IRQ on the old vector).
    let before = irq_count(irq, target_cpu)?;
    let fd = open_and_drive(&iface, ifindex, mac, 64)?;
    let echoed = recv_looped_frame(fd);
    unsafe { libc::close(fd) };
    let after = irq_count(irq, target_cpu)?;

    eprintln!(
        "NET_IRQ iface={iface} cpu={target_cpu} apic_id={target_apic} (>255) \
         irq={irq} ({dev_name}) count {before}->{after} echoed={echoed}"
    );
    ensure!(
        after > before,
        "virtio-net IRQ {irq} ({dev_name}) count on cpu {target_cpu} \
         (APIC ID {target_apic} > 255) did not rise after AF_PACKET TX-loopback \
         (before={before} after={after}); the ext-dest route did not deliver \
         the virtio-net RX-completion IRQ to the >255 APIC ID"
    );
    Ok(AssertResult::pass())
}
