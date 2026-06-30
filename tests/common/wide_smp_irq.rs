//! Shared helpers for >255-APIC-ID device-IRQ e2e tests (wide-SMP).
//!
//! A wide/sparse topology mints APIC IDs above 255 (the MSI ext-dest-id
//! threshold); these helpers pin a device's IRQ to such a vCPU and read
//! its per-CPU interrupt count to prove the >255 destination route
//! actually delivered. Shared by `wide_smp_device_irq_e2e.rs` (virtio-blk)
//! and `wide_smp_net_irq_e2e.rs` (virtio-net) — the userspace-IOAPIC +
//! KVM_SET_GSI_ROUTING path is device-agnostic, so only each device's
//! IRQ-discovery + drive differ; the APIC-ID/IRQ-count scaffolding is one.
//!
//! `#[path]`-included (not a `mod common` tree) so each test pulls in only
//! this file, matching the `common/cpulist.rs` convention.

use anyhow::{Result, bail, ensure};
use std::fs;
use std::mem;

/// Parse `/proc/cpuinfo` into `(processor_number, apic_id)` pairs. The
/// per-CPU interrupt-count column in `/proc/interrupts` is indexed by the
/// Linux processor number, while the ext-dest path is selected by APIC ID
/// — under the sparse encoding the two differ — so callers map between
/// them here.
#[allow(dead_code)]
pub fn cpu_apicids() -> Result<Vec<(usize, u32)>> {
    let text = fs::read_to_string("/proc/cpuinfo")?;
    let mut out = Vec::new();
    let mut cur_cpu: Option<usize> = None;
    for line in text.lines() {
        let Some((key, val)) = line.split_once(':') else {
            continue;
        };
        match key.trim() {
            "processor" => cur_cpu = val.trim().parse().ok(),
            "apicid" => {
                if let (Some(cpu), Ok(apic)) = (cur_cpu, val.trim().parse::<u32>()) {
                    out.push((cpu, apic));
                }
            }
            _ => {}
        }
    }
    Ok(out)
}

/// A `(processor_number, apic_id)` pair whose APIC ID exceeds 255 (the MSI
/// ext-dest-id threshold), or an error naming the max APIC ID seen.
#[allow(dead_code)]
pub fn find_apic_above_255() -> Result<(usize, u32)> {
    let apicids = cpu_apicids()?;
    apicids
        .iter()
        .copied()
        .find(|&(_, apic)| apic >= 256)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "no vCPU with APIC ID >= 256 (max seen {}); the topology did \
                 not mint a >255 APIC ID, so the ext-dest path cannot be exercised",
                apicids.iter().map(|&(_, a)| a).max().unwrap_or(0)
            )
        })
}

/// The Linux IRQ number whose `/proc/interrupts` action name (the last
/// whitespace token on the line) equals `action`. Device IRQs are named
/// after the owning device (e.g. `virtio0`), so callers pass the device
/// basename rather than hardcoding a GSI.
#[allow(dead_code)]
pub fn device_irq_by_action_name(action: &str) -> Result<u32> {
    let irqs = fs::read_to_string("/proc/interrupts")?;
    for line in irqs.lines() {
        let Some((lhs, rhs)) = line.split_once(':') else {
            continue;
        };
        if rhs.split_whitespace().last() == Some(action) {
            return lhs.trim().parse::<u32>().map_err(|e| {
                anyhow::anyhow!("non-numeric IRQ '{}' for {action}: {e}", lhs.trim())
            });
        }
    }
    bail!("no /proc/interrupts line with action name {action}")
}

/// Pin `irq` to Linux processor `cpu` via `smp_affinity_list`. Under
/// x2APIC physical mode this programs the IOAPIC RTE with that CPU's exact
/// APIC ID in physical dest_mode (no lowest-priority redistribution), so a
/// single-CPU pin to an APIC ID >= 256 forces the >255 ext-dest encoding.
#[allow(dead_code)]
pub fn pin_irq_to_cpu(irq: u32, cpu: usize) -> Result<()> {
    fs::write(
        format!("/proc/irq/{irq}/smp_affinity_list"),
        cpu.to_string(),
    )
    .map_err(|e| anyhow::anyhow!("pin irq {irq} to cpu {cpu}: {e}"))
}

/// Per-CPU interrupt count for `irq` on Linux processor `cpu`, from
/// `/proc/interrupts`. The leading columns after `:` are the per-online-CPU
/// counts (indexed by processor number) on BOTH x86_64 and arm64:
///   `<IRQ>: <c0> <c1> ... <cN-1>  <chip>  <hwirq>  [<Edge/Level>]  <action>`
/// The `Edge/Level` column is gated on CONFIG_GENERIC_IRQ_SHOW_LEVEL — arm64
/// selects it (arch/arm64/Kconfig), x86 does not — but it (with chip/hwirq)
/// TRAILS the per-CPU counts, so it never shifts them: `tokens[cpu]` for
/// `cpu` in `[0, N_online)` is a count on both arches. Callers pin all vCPUs
/// online, so `tokens[cpu]` is never the chip name.
#[allow(dead_code)]
pub fn irq_count(irq: u32, cpu: usize) -> Result<u64> {
    let irqs = fs::read_to_string("/proc/interrupts")?;
    let prefix = format!("{irq}:");
    for line in irqs.lines() {
        if line.trim_start().starts_with(&prefix) {
            let rhs = line.split_once(':').unwrap().1;
            let tokens: Vec<&str> = rhs.split_whitespace().collect();
            let tok = tokens.get(cpu).ok_or_else(|| {
                anyhow::anyhow!("cpu {cpu} column missing for irq {irq} (line: {line:?})")
            })?;
            return tok.parse::<u64>().map_err(|e| {
                anyhow::anyhow!("count column {cpu} for irq {irq} not numeric ('{tok}'): {e}")
            });
        }
    }
    bail!("irq {irq} not found in /proc/interrupts")
}

/// The single non-loopback network interface (the virtio-net NIC). Skips
/// `lo`; the NIC is the one interface backed by a device (a `device` symlink
/// under its sysfs node). Shared by the virtio-net e2es (wide_smp_net_irq +
/// net_traffic).
#[allow(dead_code)]
pub fn virtio_net_iface() -> Result<String> {
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

/// The virtio-net IRQ number + device basename: the NIC's sysfs device
/// basename (e.g. `virtio1`) is its `/proc/interrupts` action name.
#[allow(dead_code)]
pub fn virtio_net_irq(iface: &str) -> Result<(u32, String)> {
    let dev = fs::canonicalize(format!("/sys/class/net/{iface}/device"))?;
    let name = dev
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| anyhow::anyhow!("no basename for {dev:?}"))?
        .to_string();
    let irq = device_irq_by_action_name(&name)?;
    Ok((irq, name))
}

/// Every non-loopback, device-backed network interface, sorted by name (a
/// stable order independent of `read_dir` yield order). The multi-NIC e2e
/// expects one entry per configured `networks = [..]` element.
#[allow(dead_code)]
pub fn virtio_net_ifaces() -> Result<Vec<String>> {
    let mut out = Vec::new();
    for ent in fs::read_dir("/sys/class/net")? {
        let ent = ent?;
        let name = ent.file_name().to_string_lossy().into_owned();
        if name == "lo" {
            continue;
        }
        if ent.path().join("device").exists() {
            out.push(name);
        }
    }
    out.sort();
    Ok(out)
}

/// Total interrupt count for `irq` summed across every per-CPU column in
/// `/proc/interrupts`. Use when the IRQ is not pinned to a known CPU (the
/// multi-NIC e2e drives traffic without pinning, so the RX completion can
/// land on any vCPU). The trailing chip/hwirq/action columns are
/// non-numeric, so summing only the leading numeric columns yields the
/// per-IRQ total without an online-CPU count.
#[allow(dead_code)]
pub fn irq_count_total(irq: u32) -> Result<u64> {
    let irqs = fs::read_to_string("/proc/interrupts")?;
    let prefix = format!("{irq}:");
    for line in irqs.lines() {
        if line.trim_start().starts_with(&prefix) {
            let rhs = line.split_once(':').unwrap().1;
            let mut total = 0u64;
            for tok in rhs.split_whitespace() {
                match tok.parse::<u64>() {
                    Ok(n) => total = total.saturating_add(n),
                    // First non-numeric token is the chip column; the per-CPU
                    // counts are contiguous from the start, so stop here.
                    Err(_) => break,
                }
            }
            return Ok(total);
        }
    }
    bail!("irq {irq} not found in /proc/interrupts")
}

/// Read `/sys/class/net/<iface>/ifindex`.
#[allow(dead_code)]
pub fn iface_ifindex(iface: &str) -> Result<i32> {
    let s = fs::read_to_string(format!("/sys/class/net/{iface}/ifindex"))?;
    s.trim()
        .parse::<i32>()
        .map_err(|e| anyhow::anyhow!("ifindex for {iface} not numeric ('{}'): {e}", s.trim()))
}

/// Read `/sys/class/net/<iface>/address` into a 6-byte MAC.
#[allow(dead_code)]
pub fn iface_mac(iface: &str) -> Result<[u8; 6]> {
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

/// Bring `iface` administratively up via `SIOCSIFFLAGS` (no `ip`/`ifconfig`
/// dependency). AF_PACKET TX drops on a down interface (`dev_queue_xmit`
/// requires `netif_running`), so this is mandatory before driving traffic.
#[allow(dead_code)]
pub fn iface_up(fd: i32, iface: &str) -> Result<()> {
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
#[allow(dead_code)]
pub fn open_and_drive(iface: &str, ifindex: i32, mac: [u8; 6], count: usize) -> Result<i32> {
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
#[allow(dead_code)]
pub fn recv_looped_frame(fd: i32) -> bool {
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
