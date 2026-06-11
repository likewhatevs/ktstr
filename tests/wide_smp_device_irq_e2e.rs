//! End-to-end: a virtio-blk device IRQ delivered to a vCPU whose APIC ID
//! exceeds 255, over the MSI ext-dest-id (`address_hi.destid_8_31`) path.
//!
//! `wide_smp_boot_e2e` tops out at APIC ID 255, so the ext-dest-id encoding
//! for IDs > 255 is never driven by a real device IRQ there. This boots a
//! sparse topology whose APIC IDs reach 433, pins the virtio-blk IRQ to a
//! vCPU with APIC ID >= 256 via `smp_affinity`, drives disk I/O, and asserts
//! that vCPU's interrupt count rose -- proving the completion IRQ was routed
//! using the > 255 destination encoding (KVM decodes the high bits from
//! `address_hi.destid_8_31` under `x2apic_format`).
//!
//! Why pinning to one CPU proves the > 255 path: under x2APIC physical mode
//! a single-CPU `smp_affinity` programs the IOAPIC RTE with that CPU's exact
//! APIC ID in physical dest_mode (no lowest-priority redistribution). With
//! APIC ID >= 256 the high bits land in `address_hi.destid_8_31`, and KVM
//! delivers to exactly that one vCPU. So the count rising on the target CPU
//! means the ext-dest route resolved a > 255 destination correctly.
//!
//! Topology: 14 LLCs x 9 cores x 2 threads = 252 vCPUs. The sparse APIC-ID
//! encoding (core_shift = bits(9) + bits(2) = 5) gives a max APIC ID of
//! (13 << 5) | (8 << 1) | 1 = 433, so vCPUs in the upper LLCs (llc >= 8)
//! have APIC ID >= 256 without needing a > 256-vCPU host.
//!
//! virtio-blk alone is the complete proof: the > 255 ext-dest routing is
//! device-agnostic (the same userspace-IOAPIC + KVM_SET_GSI_ROUTING path for
//! every virtio device). virtio-net adds device-type coverage but needs
//! separate test-attach infra + an AF_PACKET payload (a follow-up).
//!
//! Run: cargo run --bin cargo-ktstr -- ktstr test --kernel ../linux \
//!        -- -E 'test(device_irq_delivers_to_apic_id_above_255)' \
//!        --success-output immediate

use anyhow::{Result, bail, ensure};
use ktstr::assert::AssertResult;
use ktstr::ktstr_test;
use ktstr::prelude::{DiskConfig, DiskThrottle, Filesystem};
use ktstr::scenario::Ctx;
use std::fs::{self, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};

/// Raw 256 MiB virtio-blk disk. Raw (no filesystem) so the guest writes
/// directly to /dev/vda to drive request completions. Struct-literal because
/// `DiskConfig::default` is not const and the `disk =` macro arg needs a
/// const-evaluable path.
const KTSTR_DISK_EXTDEST: DiskConfig = DiskConfig {
    capacity_mib: 256,
    filesystem: Filesystem::Raw,
    throttle: DiskThrottle {
        iops: None,
        bytes_per_sec: None,
        iops_burst_capacity: None,
        bytes_burst_capacity: None,
    },
    read_only: false,
    name: None,
    no_auto_mount: false,
};

/// Parse `/proc/cpuinfo` into `(processor_number, apic_id)` pairs. The
/// per-CPU interrupt-count column in `/proc/interrupts` is indexed by the
/// Linux processor number, while the ext-dest path is selected by APIC ID --
/// and under the sparse encoding the two differ -- so the test maps between
/// them here.
fn cpu_apicids() -> Result<Vec<(usize, u32)>> {
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

/// Discover the virtio-blk IRQ number without hardcoding the GSI. The disk's
/// sysfs parent is the virtio device, whose basename ("virtioN") is the
/// `/proc/interrupts` action name; the matching line's leading `<IRQ>:` is
/// the Linux IRQ number.
fn virtio_blk_irq() -> Result<(u32, String)> {
    let dev = fs::canonicalize("/sys/block/vda/device")?;
    let name = dev
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| anyhow::anyhow!("no basename for {dev:?}"))?
        .to_string();
    let irqs = fs::read_to_string("/proc/interrupts")?;
    for line in irqs.lines() {
        let Some((lhs, rhs)) = line.split_once(':') else {
            continue;
        };
        // The action name is the last whitespace token on the line.
        if rhs.split_whitespace().last() == Some(name.as_str()) {
            let irq: u32 = lhs.trim().parse().map_err(|e| {
                anyhow::anyhow!("non-numeric IRQ '{}' for {name}: {e}", lhs.trim())
            })?;
            return Ok((irq, name));
        }
    }
    bail!("no /proc/interrupts line with action name {name}")
}

/// Read the per-CPU interrupt count for `irq` on Linux processor `cpu`.
///
/// x86_64 `/proc/interrupts` layout (no Edge/Level column -- that field is
/// gated on CONFIG_GENERIC_IRQ_SHOW_LEVEL, which x86 does not select):
///   `<IRQ>: <c0> <c1> ... <cN-1>  IO-APIC  <hwirq>  <action>`
/// The first N tokens after `:` are the per-online-CPU counts, indexed by
/// processor number; the chip name follows. `cpu` is always < N here (all
/// 252 vCPUs are online), so `tokens[cpu]` is a count, never the chip name.
fn irq_count(irq: u32, cpu: usize) -> Result<u64> {
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

/// Drive virtio-blk request completions by writing + fsync'ing + reading raw
/// `/dev/vda`. `sync_all` blocks until the flush completes (the device
/// processed the request and raised its IRQ), so no sleep is needed.
///
/// 64 iterations give margin for x86's lazy IRQ-affinity migration: a fresh
/// `smp_affinity` write takes effect only at the next IRQ on the old vector
/// (irq_complete_move), so the first completion or two may still land on the
/// old CPU. The remaining ~62 land on the pinned CPU, so its count rises.
fn drive_disk_io() -> Result<()> {
    let mut f = OpenOptions::new().read(true).write(true).open("/dev/vda")?;
    let buf = [0xA5u8; 4096];
    for i in 0..64u64 {
        let off = i * 4096;
        f.seek(SeekFrom::Start(off))?;
        f.write_all(&buf)?;
        f.sync_all()?;
        let mut rb = [0u8; 4096];
        f.seek(SeekFrom::Start(off))?;
        f.read_exact(&mut rb)?;
    }
    Ok(())
}

#[ktstr_test(
    llcs = 14,
    cores = 9,
    threads = 2,
    disk = KTSTR_DISK_EXTDEST,
    no_perf_mode,
    duration_s = 4
)]
fn device_irq_delivers_to_apic_id_above_255(ctx: &Ctx) -> Result<AssertResult> {
    let total = ctx.topo.total_cpus();
    ensure!(
        total > 254,
        "need a >254-vCPU topology to reach APIC IDs above 255 (got {total})"
    );

    // Pick a vCPU whose APIC ID exceeds 255 (the ext-dest threshold). cpu# !=
    // apic_id under the sparse encoding, so select by APIC ID via /proc/cpuinfo.
    let apicids = cpu_apicids()?;
    let (target_cpu, target_apic) = apicids
        .iter()
        .copied()
        .find(|&(_, apic)| apic >= 256)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "no vCPU with APIC ID >= 256 (max seen {}); the topology did \
                 not mint a >255 APIC ID, so the ext-dest path cannot be exercised",
                apicids.iter().map(|&(_, a)| a).max().unwrap_or(0)
            )
        })?;

    // Discover the virtio-blk IRQ by device name (no hardcoded GSI).
    let (irq, dev_name) = virtio_blk_irq()?;

    // Pin that IRQ to the target CPU. Under x2APIC physical mode this writes
    // the target CPU's exact APIC ID (>= 256) into the RTE destination.
    fs::write(
        format!("/proc/irq/{irq}/smp_affinity_list"),
        target_cpu.to_string(),
    )
    .map_err(|e| anyhow::anyhow!("pin irq {irq} to cpu {target_cpu}: {e}"))?;

    // Baseline after pinning, drive I/O, then re-read the target CPU's count.
    let before = irq_count(irq, target_cpu)?;
    drive_disk_io()?;
    let after = irq_count(irq, target_cpu)?;

    eprintln!(
        "DEVICE_IRQ cpu={target_cpu} apic_id={target_apic} (>255) irq={irq} \
         ({dev_name}) count {before}->{after}"
    );
    ensure!(
        after > before,
        "virtio-blk IRQ {irq} ({dev_name}) count on cpu {target_cpu} \
         (APIC ID {target_apic} > 255) did not rise after disk I/O \
         (before={before} after={after}); the ext-dest route did not deliver \
         the completion IRQ to the >255 APIC ID"
    );
    Ok(AssertResult::pass())
}
