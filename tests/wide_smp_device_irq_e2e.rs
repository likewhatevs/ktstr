//! End-to-end: the virtio-blk request-queue completion IRQ is delivered over
//! its PCI + MSI-X transport on a wide / sparse-APIC topology.
//!
//! On x86_64 (post the virtio-blk-PCI migration) virtio-blk's request queue is
//! a blk-mq MANAGED MSI-X interrupt — virtio-blk passes a non-NULL
//! `irq_affinity` descriptor (drivers/block/virtio_blk.c `init_vq`) and the
//! per-vq EACH policy succeeds for its two vectors, so the request-queue vector
//! is affinity-managed; virtio-net passes a NULL descriptor
//! (drivers/net/virtio_net.c `virtnet_find_vqs`), so its vectors are never
//! managed (hence userspace-pinnable). The kernel owns a managed IRQ's
//! affinity, so userspace cannot pin it via `smp_affinity`
//! (`/proc/irq/N/smp_affinity*` writes return `-EPERM` — kernel/irq/proc.c
//! `write_irq_affinity` gates on `irq_can_set_affinity_usr`, which is false for
//! `irqd_affinity_is_managed` IRQs, kernel/irq/manage.c). The >255 APIC MSI ext-dest-id
//! (`address_hi.destid_8_31`) route is device-agnostic — the same
//! userspace-IOAPIC + `KVM_SET_GSI_ROUTING` path for every virtio device — and
//! is proven DETERMINISTICALLY by the sibling `wide_smp_net_irq_e2e.rs`, which
//! pins virtio-net's REGULAR (non-managed) MSI-X IRQ to a vCPU whose APIC ID is
//! 256 or higher. virtio-blk cannot drive that pin under managed MSI-X, so this
//! test
//! proves the complementary leg: that the virtio-blk-PCI request-queue MSI-X
//! completion IRQ actually fires. It boots the same sparse topology (so the MSI
//! route table carries >255-destination routes), drives disk I/O, and asserts
//! the request-queue data vector's `/proc/interrupts` count rose — across all
//! CPUs, since the managed IRQ's delivery CPU is kernel-chosen, not pinnable.
//!
//! Topology: 14 LLCs x 9 cores x 2 threads = 252 vCPUs. The sparse APIC-ID
//! encoding (core_shift = bits(9) + bits(2) = 5) gives a max APIC ID of
//! (13 << 5) | (8 << 1) | 1 = 433. The APIC-ID/IRQ scaffolding shared with the
//! net sibling lives in `common/wide_smp_irq.rs`.
//!
//! Run: cargo run --bin cargo-ktstr -- ktstr test --kernel ../linux \
//!        -- -E 'test(blk_pci_msix_completion_irq_delivered)' \
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
use wide_smp_irq::{device_data_irq, irq_count_total};

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

/// The virtio-blk data IRQ number + its `/proc/interrupts` action name.
/// /dev/vda's sysfs parent is the virtio device, whose basename ("virtioN")
/// drives the action name. The transport is arch-split: x86_64 is virtio-pci +
/// MSI-X, so the request-queue completion IRQ is the data vector `{name}-req.0`
/// (the config vector `{name}-config` never fires on disk I/O); aarch64 is
/// virtio-MMIO + INTx, a single line whose action is the bare `{name}`.
/// [`device_data_irq`] resolves whichever is present (INTx bare name OR the
/// non-config MSI-X data vector) — the same discovery the virtio-net sibling
/// uses.
fn virtio_blk_irq() -> Result<(u32, String)> {
    let dev = fs::canonicalize("/sys/block/vda/device")?;
    let name = dev
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| anyhow::anyhow!("no basename for {dev:?}"))?
        .to_string();
    device_data_irq(&name)
}

/// Drive virtio-blk request completions by writing + fsync'ing + reading raw
/// `/dev/vda`. `sync_all` blocks until the flush completes (the device
/// processed the request and raised its completion IRQ), so no sleep is needed.
///
/// 64 iterations accrue a clear count delta on the request-queue IRQ. That IRQ
/// is a blk-mq MANAGED interrupt whose delivery CPU is kernel-chosen (not
/// pinnable — see the module doc), so the caller asserts the IRQ's TOTAL
/// `/proc/interrupts` count (summed across CPUs) rose, not a per-CPU count.
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
fn blk_pci_msix_completion_irq_delivered(_ctx: &Ctx) -> Result<AssertResult> {
    // Discover the virtio-blk request-queue DATA IRQ by device name (no
    // hardcoded GSI). On x86 (PCI + MSI-X) that is the non-config data vector
    // (`{name}-req.0`); the request-queue IRQ is a blk-mq MANAGED interrupt, so
    // its delivery CPU is kernel-chosen and userspace cannot pin it (see the
    // module doc) — the count is therefore summed across all CPUs rather than
    // pinned to one. The >255 ext-dest leg is covered by the net sibling.
    let (irq, dev_name) = virtio_blk_irq()?;

    // Baseline total count, drive request completions, then re-read. Each
    // `sync_all` blocks until the device processed the request and raised its
    // completion IRQ, so no sleep is needed.
    let before = irq_count_total(irq)?;
    drive_disk_io()?;
    let after = irq_count_total(irq)?;

    eprintln!("BLK_MSIX_IRQ irq={irq} ({dev_name}) total count {before}->{after}");
    ensure!(
        after > before,
        "virtio-blk request-queue MSI-X IRQ {irq} ({dev_name}) total count did \
         not rise after disk I/O (before={before} after={after}); the \
         virtio-blk-PCI MSI-X completion IRQ was not delivered"
    );
    Ok(AssertResult::pass())
}
