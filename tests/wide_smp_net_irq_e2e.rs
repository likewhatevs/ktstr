//! End-to-end: a virtio-net device IRQ delivered to a vCPU whose APIC ID
//! exceeds 255, over the MSI ext-dest-id (`address_hi.destid_8_31`) path.
//!
//! Companion to `wide_smp_device_irq_e2e.rs` (virtio-blk). The >255
//! ext-dest routing is device-agnostic (the same userspace-IOAPIC +
//! `KVM_SET_GSI_ROUTING` path for every virtio device), so virtio-blk
//! already proves the route resolves a >255 destination. This adds the
//! virtio-net device-type leg: it boots a sparse topology whose APIC IDs
//! reach 433, attaches a NIC via the `network =` test attribute, pins the
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

use anyhow::{Result, bail, ensure};
use ktstr::assert::AssertResult;
use ktstr::ktstr_test;
use ktstr::prelude::NetConfig;
use ktstr::scenario::Ctx;
use std::fs;
use std::mem;

#[path = "common/wide_smp_irq.rs"]
mod wide_smp_irq;
use wide_smp_irq::{device_irq_by_action_name, find_apic_above_255, irq_count, pin_irq_to_cpu};

/// virtio-net with a deterministic locally-administered MAC. Const because
/// `NetConfig::default` is not const and the `network =` macro arg needs a
/// const-evaluable path; `NetConfig::DEFAULT.mac(..)` is the const-fn chain.
const KTSTR_NET: NetConfig = NetConfig::DEFAULT.mac([0x52, 0x54, 0x00, 0x12, 0x34, 0x56]);

/// The single non-loopback network interface (the virtio-net NIC). Skips
/// `lo`; the NIC is the one interface backed by a device (a `device`
/// symlink under its sysfs node).
fn virtio_net_iface() -> Result<String> {
    for ent in fs::read_dir("/sys/class/net")? {
        let ent = ent?;
        let name = ent.file_name().to_string_lossy().into_owned();
        if name == "lo" {
            continue;
        }
        if ent.path().join("device").exists() {
            return Ok(name);
        }
    }
    bail!("no non-loopback network interface with a device under /sys/class/net")
}

/// Read `/sys/class/net/<iface>/ifindex`.
fn iface_ifindex(iface: &str) -> Result<i32> {
    let s = fs::read_to_string(format!("/sys/class/net/{iface}/ifindex"))?;
    s.trim()
        .parse::<i32>()
        .map_err(|e| anyhow::anyhow!("ifindex for {iface} not numeric ('{}'): {e}", s.trim()))
}

/// Read `/sys/class/net/<iface>/address` into a 6-byte MAC.
fn iface_mac(iface: &str) -> Result<[u8; 6]> {
    let s = fs::read_to_string(format!("/sys/class/net/{iface}/address"))?;
    let mut mac = [0u8; 6];
    let mut n = 0;
    for (i, octet) in s.trim().split(':').enumerate() {
        ensure!(i < 6, "MAC for {iface} has >6 octets: {:?}", s.trim());
        mac[i] = u8::from_str_radix(octet, 16)
            .map_err(|e| anyhow::anyhow!("MAC octet '{octet}' for {iface} not hex: {e}"))?;
        n = i + 1;
    }
    ensure!(
        n == 6,
        "MAC for {iface} has {n} octets, expected 6: {:?}",
        s.trim()
    );
    Ok(mac)
}

/// The virtio-net IRQ number: the NIC's sysfs device basename (e.g.
/// `virtio1`) is its `/proc/interrupts` action name.
fn virtio_net_irq(iface: &str) -> Result<(u32, String)> {
    let dev = fs::canonicalize(format!("/sys/class/net/{iface}/device"))?;
    let name = dev
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| anyhow::anyhow!("no basename for {dev:?}"))?
        .to_string();
    let irq = device_irq_by_action_name(&name)?;
    Ok((irq, name))
}

/// Bring `iface` administratively up via `SIOCSIFFLAGS` (no `ip`/`ifconfig`
/// dependency). AF_PACKET TX drops on a down interface (`dev_queue_xmit`
/// requires `netif_running`), so this is mandatory before driving traffic.
fn iface_up(fd: i32, iface: &str) -> Result<()> {
    let mut ifr: libc::ifreq = unsafe { mem::zeroed() };
    for (i, &b) in iface.as_bytes().iter().enumerate() {
        ensure!(i < libc::IFNAMSIZ - 1, "iface name {iface} too long");
        ifr.ifr_name[i] = b as libc::c_char;
    }
    // SAFETY: ifr_name is set; SIOCGIFFLAGS reads current flags into the
    // ifru_flags union arm, SIOCSIFFLAGS writes them back with IFF_UP set.
    let rc = unsafe { libc::ioctl(fd, libc::SIOCGIFFLAGS, &mut ifr) };
    ensure!(
        rc == 0,
        "SIOCGIFFLAGS {iface}: {}",
        std::io::Error::last_os_error()
    );
    unsafe {
        ifr.ifr_ifru.ifru_flags |= (libc::IFF_UP | libc::IFF_RUNNING) as libc::c_short;
    }
    let rc = unsafe { libc::ioctl(fd, libc::SIOCSIFFLAGS, &ifr) };
    ensure!(
        rc == 0,
        "SIOCSIFFLAGS up {iface}: {}",
        std::io::Error::last_os_error()
    );
    Ok(())
}

/// Open an AF_PACKET raw socket bound to `ifindex`, bring the iface up, and
/// send `count` minimal Ethernet frames. Each TX is a virtio TX kick; the
/// in-VMM loopback echoes it into RX and fires the RX IRQ, so `count`
/// frames drive `count` RX completions. Returns the raw fd (kept open so a
/// subsequent recv can confirm the loopback echo) — caller closes it.
fn open_and_drive(iface: &str, ifindex: i32, mac: [u8; 6], count: usize) -> Result<i32> {
    let proto = (libc::ETH_P_ALL as u16).to_be() as libc::c_int;
    // SAFETY: standard AF_PACKET raw socket creation.
    let fd = unsafe { libc::socket(libc::AF_PACKET, libc::SOCK_RAW, proto) };
    ensure!(
        fd >= 0,
        "socket(AF_PACKET): {}",
        std::io::Error::last_os_error()
    );

    iface_up(fd, iface).inspect_err(|_| unsafe {
        libc::close(fd);
    })?;

    // Bind so the socket is associated with the NIC (filters RX to it and
    // lets a follow-up recv observe the looped frame).
    let mut sll: libc::sockaddr_ll = unsafe { mem::zeroed() };
    sll.sll_family = libc::AF_PACKET as u16;
    sll.sll_protocol = (libc::ETH_P_ALL as u16).to_be();
    sll.sll_ifindex = ifindex;
    let rc = unsafe {
        libc::bind(
            fd,
            &sll as *const _ as *const libc::sockaddr,
            mem::size_of::<libc::sockaddr_ll>() as libc::socklen_t,
        )
    };
    if rc != 0 {
        let e = std::io::Error::last_os_error();
        unsafe { libc::close(fd) };
        bail!("bind AF_PACKET to {iface}: {e}");
    }

    // Minimal padded Ethernet frame; dst == src == our MAC (the loopback
    // echoes verbatim, no MAC swap), a local-experimental ethertype, and a
    // payload marker. 60 bytes = the min Ethernet frame (sans FCS).
    let mut frame = [0u8; 60];
    frame[0..6].copy_from_slice(&mac);
    frame[6..12].copy_from_slice(&mac);
    frame[12..14].copy_from_slice(&0x88B5u16.to_be_bytes());
    frame[14..18].copy_from_slice(b"KTST");

    let mut dst: libc::sockaddr_ll = unsafe { mem::zeroed() };
    dst.sll_family = libc::AF_PACKET as u16;
    dst.sll_protocol = (libc::ETH_P_ALL as u16).to_be();
    dst.sll_ifindex = ifindex;
    dst.sll_halen = 6;
    dst.sll_addr[0..6].copy_from_slice(&mac);

    for i in 0..count {
        let sent = unsafe {
            libc::sendto(
                fd,
                frame.as_ptr() as *const libc::c_void,
                frame.len(),
                0,
                &dst as *const _ as *const libc::sockaddr,
                mem::size_of::<libc::sockaddr_ll>() as libc::socklen_t,
            )
        };
        if sent < 0 {
            let e = std::io::Error::last_os_error();
            unsafe { libc::close(fd) };
            bail!("sendto frame {i} on {iface}: {e}");
        }
    }
    Ok(fd)
}

/// Best-effort: confirm the in-VMM loopback echoed at least one frame back
/// into RX (a 2s timeout bounds a stall). Proves the TX->loopback->RX data
/// path fired, not just that TX was accepted. Non-fatal on timeout — the
/// IRQ-count assertion is the primary proof — but a received frame is the
/// stronger confirmation that RX (and thus its IRQ) ran.
fn recv_looped_frame(fd: i32) -> bool {
    let tv = libc::timeval {
        tv_sec: 2,
        tv_usec: 0,
    };
    unsafe {
        libc::setsockopt(
            fd,
            libc::SOL_SOCKET,
            libc::SO_RCVTIMEO,
            &tv as *const _ as *const libc::c_void,
            mem::size_of::<libc::timeval>() as libc::socklen_t,
        );
    }
    let mut buf = [0u8; 256];
    let n = unsafe { libc::recv(fd, buf.as_mut_ptr() as *mut libc::c_void, buf.len(), 0) };
    n > 0
}

#[ktstr_test(
    llcs = 14,
    cores = 9,
    threads = 2,
    network = KTSTR_NET,
    no_perf_mode,
    duration_s = 4
)]
fn net_irq_delivers_to_apic_id_above_255(ctx: &Ctx) -> Result<AssertResult> {
    let total = ctx.topo.total_cpus();
    ensure!(
        total > 254,
        "need a >254-vCPU topology to reach APIC IDs above 255 (got {total})"
    );

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
