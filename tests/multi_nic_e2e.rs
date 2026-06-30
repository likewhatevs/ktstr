//! End-to-end: two virtio-net PCI functions, each on its own PCI slot with
//! its own INTx GSI, both delivering RX-completion interrupts independently.
//!
//! Boots a small VM with `networks = [NET_A, NET_B]` and proves the
//! multi-NIC path end to end:
//!   1. two non-loopback interfaces appear (two PCI functions installed at
//!      slots 1 and 2);
//!   2. each carries its configured MAC (per-NIC device-config space, not a
//!      single shared device);
//!   3. each has a distinct IRQ (per-NIC GSI allocation + a DSDT `_PRT`
//!      entry per NIC), and driving a NIC raises ITS irq.
//!
//! (3) is the load-bearing multi-NIC check: each NIC's irqfd must be wired
//! to its own GSI. If the per-NIC irqfd→GSI wiring were crossed, driving
//! NIC A's TX-loopback would raise NIC B's IRQ line instead, so A's own
//! `/proc/interrupts` count would stay flat and the assertion fails.
//!
//! Why AF_PACKET: the v0 virtio-net backend is in-VMM loopback (TX bytes
//! echoed straight into RX). IP-layer self-traffic never reaches the NIC
//! (the guest's `RTN_LOCAL` route sends it to `lo`), so only an AF_PACKET
//! raw socket bound to the interface generates a real virtio TX kick and
//! observes the resulting RX interrupt (see `vmm::net_config`).
//!
//! Run: cargo run --bin cargo-ktstr -- ktstr test --kernel ../linux \
//!        -- -E 'test(two_nics_each_deliver_their_own_irq)' \
//!        --success-output immediate

// Multi-NIC is the virtio-pci transport, which is x86-only for now (aarch64
// keeps a single virtio-MMIO NIC; build() caps `networks` at 1 there). The
// AF_PACKET drive + /proc/interrupts reads are arch-neutral, but there is no
// second NIC to enumerate on aarch64, so gate the whole test to x86.
#![cfg(target_arch = "x86_64")]

use anyhow::{Result, ensure};
use ktstr::assert::AssertResult;
use ktstr::ktstr_test;
use ktstr::prelude::NetConfig;
use ktstr::scenario::Ctx;
use std::collections::BTreeSet;

#[path = "common/wide_smp_irq.rs"]
mod wide_smp_irq;
use wide_smp_irq::{
    iface_ifindex, iface_mac, irq_count_total, open_and_drive, recv_looped_frame,
    virtio_net_ifaces, virtio_net_irq,
};

// Two NICs with deterministic, distinct locally-administered MACs. Const
// because the `networks =` macro args need const-evaluable paths;
// `NetConfig::DEFAULT.mac(..)` is the const-fn chain.
const MAC_A: [u8; 6] = [0x52, 0x54, 0x00, 0x0a, 0x0a, 0x0a];
const MAC_B: [u8; 6] = [0x52, 0x54, 0x00, 0x0b, 0x0b, 0x0b];
const NET_A: NetConfig = NetConfig::DEFAULT.mac(MAC_A);
const NET_B: NetConfig = NetConfig::DEFAULT.mac(MAC_B);

#[ktstr_test(
    llcs = 1,
    cores = 2,
    threads = 1,
    networks = [NET_A, NET_B],
    no_perf_mode,
    duration_s = 4
)]
fn two_nics_each_deliver_their_own_irq(_ctx: &Ctx) -> Result<AssertResult> {
    // 1. Exactly two NICs appeared — two virtio-net PCI functions installed.
    let ifaces = virtio_net_ifaces()?;
    ensure!(
        ifaces.len() == 2,
        "expected 2 virtio-net interfaces (networks = [NET_A, NET_B]); saw {}: {ifaces:?}",
        ifaces.len()
    );

    // 2. Each NIC carries its own configured MAC — the guest MAC set equals
    //    {MAC_A, MAC_B}. A single shared config space would surface one MAC
    //    twice (or one NIC with the wrong MAC).
    let mut macs = BTreeSet::new();
    for iface in &ifaces {
        macs.insert(iface_mac(iface)?);
    }
    let want: BTreeSet<[u8; 6]> = [MAC_A, MAC_B].into_iter().collect();
    ensure!(
        macs == want,
        "guest NIC MACs {macs:02x?} != configured {want:02x?} — per-NIC config-space MAC wrong"
    );

    // 3. Each NIC has its own IRQ (distinct per-NIC GSI + DSDT _PRT entry),
    //    and driving that NIC raises ITS irq — proving each per-NIC irqfd is
    //    wired to its own GSI. A crossed wiring would leave the driven NIC's
    //    own count flat (the interrupt would land on the other NIC's line).
    let mut seen_irqs = BTreeSet::new();
    for iface in &ifaces {
        let ifindex = iface_ifindex(iface)?;
        let mac = iface_mac(iface)?;
        let (irq, dev) = virtio_net_irq(iface)?;
        ensure!(
            seen_irqs.insert(irq),
            "NIC {iface} ({dev}) shares irq {irq} with another NIC — \
             per-NIC GSI allocation failed (each NIC must get a distinct GSI)"
        );
        let before = irq_count_total(irq)?;
        let fd = open_and_drive(iface, ifindex, mac, 64)?;
        let echoed = recv_looped_frame(fd);
        // SAFETY: fd is the raw socket from open_and_drive; closing it once.
        unsafe { libc::close(fd) };
        let after = irq_count_total(irq)?;
        eprintln!(
            "MULTI_NIC iface={iface} dev={dev} irq={irq} count {before}->{after} echoed={echoed}"
        );
        ensure!(
            after > before,
            "NIC {iface} ({dev}) irq {irq} count did not rise after AF_PACKET \
             TX-loopback (before={before} after={after}); its per-NIC irqfd did \
             not deliver the RX-completion interrupt to its own GSI"
        );
    }
    Ok(AssertResult::pass())
}
