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
//! virtio-blk proves the routing; the > 255 ext-dest path is device-agnostic
//! (the same userspace-IOAPIC + KVM_SET_GSI_ROUTING for every virtio device).
//! The sibling `wide_smp_net_irq_e2e.rs` adds the virtio-net device-type leg
//! (the `networks =` attr + an AF_PACKET payload); the APIC-ID/IRQ-count
//! scaffolding both share lives in `common/wide_smp_irq.rs`.
//!
//! Run: cargo run --bin cargo-ktstr -- ktstr test --kernel ../linux \
//!        -- -E 'test(device_irq_delivers_to_apic_id_above_255)' \
//!        --success-output immediate

// The >255-APIC-ID MSI ext-dest-id path is x86-only: aarch64 uses the GIC
// (SPIs; no APIC/IOAPIC/MSI destination) and /proc/cpuinfo has no `apicid`
// field, so find_apic_above_255 returns Err before any I/O. The arm64
// high-vCPU device-IRQ equivalent (GIC SPI delivery) is a separate test.
#![cfg(target_arch = "x86_64")]

use anyhow::{Result, ensure};
use ktstr::assert::AssertResult;
use ktstr::ktstr_test;
use ktstr::prelude::{DiskConfig, DiskThrottle, Filesystem};
use ktstr::scenario::Ctx;
use std::fs::{self, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};

#[path = "common/wide_smp_irq.rs"]
mod wide_smp_irq;
use wide_smp_irq::{device_irq_by_action_name, find_apic_above_255, irq_count, pin_irq_to_cpu};

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

/// The virtio-blk IRQ number: /dev/vda's sysfs parent is the virtio device,
/// whose basename ("virtioN") is the `/proc/interrupts` action name.
fn virtio_blk_irq() -> Result<(u32, String)> {
    let dev = fs::canonicalize("/sys/block/vda/device")?;
    let name = dev
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| anyhow::anyhow!("no basename for {dev:?}"))?
        .to_string();
    let irq = device_irq_by_action_name(&name)?;
    Ok((irq, name))
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
fn device_irq_delivers_to_apic_id_above_255(_ctx: &Ctx) -> Result<AssertResult> {
    // No vCPU-count gate: the sparse APIC-ID encoding mints IDs up to 433
    // from this 252-vCPU topology (see the module doc), so the requirement
    // is "an APIC ID > 255 exists", not "> 254 vCPUs". find_apic_above_255
    // reads the guest's actual /proc/cpuinfo and errors if none reach 256 --
    // the correct runtime gate. (A > 254-vCPU count is the right invariant
    // for the dense sibling tests -- 16*16*1 = 256, max APIC ID 255 -- but
    // wrong for this sparse topology: 14*9*2 = 252 < 254, so it failed every
    // run that was not host-skipped.)

    // Pick a vCPU whose APIC ID exceeds 255 (the ext-dest threshold). cpu# !=
    // apic_id under the sparse encoding, so select by APIC ID.
    let (target_cpu, target_apic) = find_apic_above_255()?;

    // Discover the virtio-blk IRQ by device name (no hardcoded GSI).
    let (irq, dev_name) = virtio_blk_irq()?;

    // Pin that IRQ to the target CPU. Under x2APIC physical mode this writes
    // the target CPU's exact APIC ID (>= 256) into the RTE destination.
    pin_irq_to_cpu(irq, target_cpu)?;

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
