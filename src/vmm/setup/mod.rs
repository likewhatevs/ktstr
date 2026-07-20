//! Boot pipeline for `KtstrVm`: virtio-blk wiring, KVM creation,
//! initramfs resolution and compression, COW overlay, deferred memory
//! computation, x86_64 / aarch64 memory and FDT layout, vCPU register
//! setup.
//!
//! These methods run on the calling thread (no vCPU work yet) and
//! produce a [`KtstrKvm`](super::kvm::KtstrKvm) ready for the
//! [`KtstrVm::run_vm`](super::KtstrVm::run_vm) loop. They are reopened
//! as additional [`impl KtstrVm`](super::KtstrVm) blocks; the canonical
//! struct definition lives in [`super`].

use anyhow::{Context, Result};
use std::os::fd::{AsRawFd, OwnedFd};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use vm_memory::{Bytes, GuestAddress, GuestMemory, GuestMemoryMmap};

use super::KtstrVm;
#[cfg(test)]
use super::initramfs_cache::BaseKey;
#[cfg(any(target_arch = "aarch64", test))]
use super::initramfs_cache::PREPARED_MAPPING_GRANULE;
use super::initramfs_cache::{
    PreparedInitrd, PreparedMapping, PreparedOverlay, complete_prepared_initrd,
    get_or_prepare_base, prepare_base_inputs,
};
use super::memory_budget::{
    MemoryBudget, TmpfsFraction, initramfs_min_memory_mib, read_kernel_init_size,
    read_kernel_version, read_kernel_version_from_metadata_sidecar,
};
use super::numa_mem::MemoryBacking;
use super::pi_mutex::PiMutex;
use super::{disk_config, disk_template, initramfs, virtio_blk, virtio_net};
// The virtio-PCI transport, its MSI-X state, and the INTx resample eventfd are
// x86_64-only (aarch64 is virtio-MMIO + GICv3 with no PCI); their only users are
// the `#[cfg(target_arch = "x86_64")]` PCI setup paths below.
#[cfg(target_arch = "x86_64")]
use super::{pci, virtio_msix};
#[cfg(target_arch = "x86_64")]
use vmm_sys_util::eventfd::EventFd;

#[cfg(target_arch = "aarch64")]
use super::aarch64;
#[cfg(target_arch = "aarch64")]
use super::aarch64::boot;
#[cfg(target_arch = "aarch64")]
use super::aarch64::kvm;
#[cfg(target_arch = "x86_64")]
use super::virtio_console;
#[cfg(target_arch = "x86_64")]
use super::x86_64::{acpi, boot, kvm, mptable};

fn framework_infrastructure<T>(result: Result<T>) -> Result<T> {
    result.context(crate::test_support::FrameworkInfrastructureFailure)
}

/// Host-side handles for the x86_64 virtio-net PCI device. The device
/// core lives inside the [`pci::PciBus`] function; these are the pieces
/// the run loop keeps alive and exposes.
#[cfg(target_arch = "x86_64")]
pub(crate) struct NetDeviceHandles {
    /// Cumulative device counters — threaded to the failure-dump run
    /// state and snapshotted into `VmResult::virtio_net_counters`
    /// (the PCI path's replacement for the MMIO handle's `counters()`).
    pub(crate) counters: Arc<virtio_net::VirtioNetCounters>,
    /// The INTx resample eventfd — `Some` only on the full in-kernel
    /// irqchip path (`<=254` max APIC ID), where `register_irqfd_with_resample`
    /// makes KVM track the level GSI and de-assert it on guest EOI,
    /// notifying this fd. KVM holds the raw fd, so when `Some` it MUST
    /// outlive the run (a dropped fd silently stops resampling). `None`
    /// on split-irqchip (`>254` max APIC ID): the kernel's `kvm_arch_irqfd_
    /// allowed` rejects a resample irqfd there (it requires `irqchip_
    /// full`), so a plain edge irqfd is used and there is nothing to bind.
    /// The gate is max APIC ID (`max_apic_id > MAX_XAPIC_ID`), not vCPU count —
    /// APIC IDs are sparse, so a wide-core sub-254-vCPU guest can take the
    /// split path.
    pub(crate) resample_evt: Option<EventFd>,
}

/// Host-side handles for the x86_64 virtio-blk PCI device. Unlike
/// [`NetDeviceHandles`] (which moves its device into the facade and keeps only
/// the counters), virtio-blk shares its device core as an `Arc<PiMutex<_>>`: the
/// run loop keeps `device` to pause/resume the request worker across the freeze
/// rendezvous, snapshot counters (`device.lock().counters()`), and set the
/// parked-eventfd — operations the facade's other Arc clone never serves. The
/// MMIO-dispatch `virtio_blk` handle stays `None` on x86 (guest BAR accesses
/// route through the PCI bus), so `device` is the run loop's sole reach into the
/// device on this transport.
#[cfg(target_arch = "x86_64")]
pub(crate) struct BlkDeviceHandles {
    /// The shared device core — the same `Arc<PiMutex<VirtioBlk>>` whose other
    /// clone backs the installed [`pci::PciBus`] function. The run loop pauses /
    /// resumes the worker, reads counters, and sets the parked-eventfd through
    /// it.
    pub(crate) device: Arc<PiMutex<virtio_blk::VirtioBlk>>,
    /// The INTx resample eventfd — `Some` only on the full in-kernel irqchip
    /// path (`<=254` max APIC ID); KVM holds its raw fd to de-assert the level
    /// GSI on guest EOI, so it MUST outlive the run. `None` on split-irqchip
    /// (`>254`), where a plain edge irqfd is used. Same semantics as
    /// [`NetDeviceHandles::resample_evt`].
    pub(crate) resample_evt: Option<EventFd>,
}

/// Address where initramfs is loaded in guest memory.
#[cfg(target_arch = "x86_64")]
const INITRD_ADDR: u64 = 0x800_0000; // 128 MiB

/// Compute initramfs load address at the high end of DRAM, just below
/// the FDT. Matches Firecracker/Cloud Hypervisor placement pattern —
/// avoids conflicts with early kernel allocations near the kernel image.
///
/// Aligned to the prepared mapping granule (2 MiB), which satisfies both the
/// guest-range geometry and Linux's hugetlb `MAP_FIXED` replacement rules.
/// Destination host virtual addresses are validated separately against the
/// runtime host page size.
#[cfg(target_arch = "aarch64")]
fn aarch64_initrd_addr(memory_mib: u32, total_cpus: u32, initrd_max_size: u64) -> Result<u64> {
    // Ceiling is the PVTIME carve base, NOT the FDT address. setup_pvtime
    // registers the per-vCPU steal-time IPAs in [pvtime_base, fdt_addr) and
    // write_memory shrinks the advertised /memory to END at pvtime_base. An
    // initrd whose top enters that carve is corrupted two independent ways:
    // (1) on the FIRST KVM_RUN, before any guest code executes, the host
    // writes the 8-byte stolen_time field at steal_base+8 (the kvm_update_
    // stolen_time call from check_vcpu_requests) — the full 64-byte struct
    // is zeroed later, guest-triggered via the PV_TIME_ST hypercall, after
    // initrd unpack; and (2) the carve is outside advertised RAM, so the
    // guest kernel never memblock-reserves those pages. Either clobbers the
    // initramfs and the guest's /init never starts. Anchor the whole initrd
    // below pvtime_base so it stays in advertised RAM, clear of the carve.
    let ceiling = aarch64::fdt::pvtime_base(memory_mib, total_cpus);
    let page_size = PREPARED_MAPPING_GRANULE as u64;
    let mask = !(page_size - 1);
    let mapped_size = initrd_max_size
        .checked_add(page_size - 1)
        .map(|size| size & mask)
        .context("compressed initrd rounded mapping length overflows u64")?;
    let aligned_ceiling = ceiling & mask;
    // Place initrd just below the PVTIME carve, mapping-granule-aligned. Use
    // checked_sub: a compressed initramfs larger than the advertised RAM
    // span [DRAM_START, pvtime_base) would otherwise wrap the u64 (debug:
    // panic; release with overflow-checks off: a near-u64::MAX value that
    // would PASS the >= DRAM_START check and advertise a bogus
    // linux,initrd-start). The min-memory budget sizes RAM for the
    // tmpfs/init constraint, not for 'initrd fits below pvtime_base', so
    // this bound is payload-reachable. The initrd must reside entirely
    // within advertised RAM: an initrd above pvtime_base is outside the
    // advertised /memory and the guest kernel never memblock-reserves it
    // (see this function's header comment).
    let load_addr = aligned_ceiling.checked_sub(mapped_size).with_context(|| {
        format!(
            "compressed initrd ({initrd_max_size} bytes) exceeds the \
                 RAM span below the PVTIME carve (pvtime_base={ceiling:#x}): \
                 reduce initramfs size or increase VM memory"
        )
    })?;
    anyhow::ensure!(
        load_addr >= kvm::DRAM_START,
        "initrd load address {load_addr:#x} underflows DRAM_START {:#x} \
         (compressed initrd {initrd_max_size} bytes, pvtime_base {ceiling:#x}): \
         reduce initramfs size or increase VM memory",
        kvm::DRAM_START,
    );
    Ok(load_addr)
}

/// Host page size in bytes. Reads from `sysconf(_SC_PAGESIZE)` once
/// per process and caches the result via `OnceLock`; subsequent calls
/// hit the cache. The kernel reports the actual MMU page size (4 KB
/// on x86_64 / common aarch64, 16 KB on Apple Silicon and some
/// aarch64 server SKUs). Falls back to 4 KB only when `sysconf`
/// returns an error code (≤0), which would itself indicate a libc bug
/// — the fallback exists so a downstream alignment computation never
/// produces 0.
#[allow(dead_code)]
pub(crate) fn host_page_size() -> u64 {
    static CACHED: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        // SAFETY: sysconf is a thread-safe libc function that takes a
        // constant integer argument and returns a long. No invariants
        // on the caller side.
        let sz = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if sz > 0 { sz as u64 } else { 0x1000 }
    })
}

fn validate_prepared_host_address(host_addr: *mut u8) -> Result<()> {
    let host_page = host_page_size() as usize;
    anyhow::ensure!(
        (host_addr as usize).is_multiple_of(host_page),
        "prepared initrd host address is not aligned to the \
         {host_page}-byte host page size"
    );
    Ok(())
}

fn validate_prepared_split_host_address(host_addr: *mut u8, split_alignment: usize) -> Result<()> {
    validate_prepared_host_address(host_addr)?;
    anyhow::ensure!(
        (host_addr as usize).is_multiple_of(split_alignment),
        "prepared initrd host address is not aligned to the \
         {split_alignment}-byte backing boundary"
    );
    Ok(())
}

/// The prepared CAS source is an ordinary file, so Linux `mmap(2)` requires
/// its offset to be aligned to the runtime base page. Replacing a hugetlb VMA
/// additionally constrains the destination address and length to the hugetlb
/// boundary, but does not turn the source fd into a hugetlb file or impose a
/// 2 MiB file-offset requirement.
fn validate_prepared_file_offset(file_offset: u64, host_page_size: usize) -> Result<()> {
    anyhow::ensure!(
        file_offset.is_multiple_of(host_page_size as u64),
        "prepared initrd file offset is not aligned to the \
         {host_page_size}-byte host page size"
    );
    Ok(())
}

fn prepared_region_split_alignment(
    backing: MemoryBacking,
    host_page_size: usize,
    mapping_granule: usize,
) -> usize {
    match backing {
        MemoryBacking::BasePages => host_page_size,
        MemoryBacking::HugeTlb2M => mapping_granule,
    }
}

fn expected_initrd_magic(
    compression: initramfs::InitrdCompression,
) -> (&'static [u8], &'static str) {
    match compression {
        initramfs::InitrdCompression::Lz4 => (&initramfs::LZ4_LEGACY_MAGIC, "LZ4 legacy"),
        initramfs::InitrdCompression::Zstd => (b"\x28\xb5\x2f\xfd", "zstd"),
        initramfs::InitrdCompression::Gzip => (b"\x1f\x8b", "gzip"),
        initramfs::InitrdCompression::Uncompressed => (b"070701", "newc cpio"),
    }
}

fn validate_prepared_stream_magic(
    compression: initramfs::InitrdCompression,
    ranges: &[PreparedMapping],
) -> Result<()> {
    let first = ranges
        .first()
        .context("prepared initrd has no mapping ranges")?;
    let (fd, file_offset) = first
        .overlays
        .iter()
        .rev()
        .find(|overlay| overlay.guest_offset == 0)
        .map_or((&first.fd, first.file_offset), |overlay| {
            (&overlay.fd, overlay.file_offset)
        });
    let (expected, name) = expected_initrd_magic(compression);
    let mut actual = vec![0u8; expected.len()];
    let mut done = 0usize;
    while done < actual.len() {
        let offset = file_offset
            .checked_add(done as u64)
            .context("prepared initrd magic file offset overflow")?;
        let offset = libc::off_t::try_from(offset)
            .context("prepared initrd magic file offset exceeds off_t")?;
        let read = unsafe {
            libc::pread(
                fd.as_raw_fd(),
                actual[done..].as_mut_ptr().cast(),
                actual.len() - done,
                offset,
            )
        };
        if read < 0 {
            let error = std::io::Error::last_os_error();
            if error.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            return Err(error).context("pread prepared initrd compression magic");
        }
        anyhow::ensure!(
            read != 0,
            "prepared initrd backing file is truncated before its compression magic"
        );
        done += read as usize;
    }
    anyhow::ensure!(
        actual == expected,
        "prepared initrd has invalid {name} magic: expected {expected:02x?}, got {actual:02x?}"
    );
    Ok(())
}

fn validate_prepared_backing_extent(
    fd: &OwnedFd,
    file_offset: u64,
    map_len: usize,
    host_page_size: usize,
    description: &str,
) -> Result<()> {
    validate_prepared_file_offset(file_offset, host_page_size)?;
    libc::off_t::try_from(file_offset)
        .with_context(|| format!("{description} file offset exceeds off_t"))?;
    let file_end = file_offset
        .checked_add(u64::try_from(map_len)?)
        .with_context(|| format!("{description} backing-file extent overflow"))?;
    let stat = rustix::fs::fstat(fd).with_context(|| format!("stat {description} backing file"))?;
    anyhow::ensure!(
        stat.st_size >= 0 && file_end <= stat.st_size as u64,
        "{description} exceeds its backing file: end={file_end:#x}, file_len={:#x}",
        stat.st_size
    );
    Ok(())
}

/// Validate every property that can be checked before the first unsafe
/// `MAP_FIXED`. Keeping this pure over prepared-range metadata makes the exact
/// production validator directly testable with malformed gap/overlap/reorder
/// fixtures.
fn validate_prepared_load(
    total_compressed: usize,
    compression: initramfs::InitrdCompression,
    page_size: usize,
    host_page_size: usize,
    load_addr: u64,
    ranges: &[PreparedMapping],
) -> Result<u32> {
    let boot_size =
        u32::try_from(total_compressed).context("compressed initrd exceeds u32 boot-size field")?;
    anyhow::ensure!(
        page_size.is_power_of_two(),
        "prepared initrd mapping granule is not a power of two"
    );
    anyhow::ensure!(
        host_page_size.is_power_of_two() && page_size.is_multiple_of(host_page_size),
        "prepared initrd host-page geometry is incompatible with its mapping granule"
    );
    anyhow::ensure!(
        load_addr & (page_size as u64 - 1) == 0,
        "prepared initrd load address {load_addr:#x} is not aligned to the \
         {page_size}-byte prepared mapping granule"
    );
    anyhow::ensure!(!ranges.is_empty(), "prepared initrd has no mapping ranges");

    let mut mapped_len = 0usize;
    for range in ranges {
        anyhow::ensure!(
            range.guest_offset == mapped_len as u64,
            "prepared initrd ranges contain a gap, overlap, or reordering"
        );
        anyhow::ensure!(
            range.map_len > 0 && range.map_len % page_size == 0,
            "prepared initrd mapping length is not aligned to the prepared granule"
        );
        validate_prepared_backing_extent(
            &range.fd,
            range.file_offset,
            range.map_len,
            host_page_size,
            "prepared initrd mapping",
        )?;
        let range_end = range
            .guest_offset
            .checked_add(u64::try_from(range.map_len)?)
            .context("prepared initrd mapping guest extent overflow")?;
        let mut previous_overlay_end = range.guest_offset;
        for overlay in &range.overlays {
            anyhow::ensure!(
                overlay.map_len > 0 && overlay.map_len % host_page_size == 0,
                "prepared initrd overlay length is not aligned to the host page size"
            );
            anyhow::ensure!(
                overlay.guest_offset.is_multiple_of(host_page_size as u64),
                "prepared initrd overlay guest offset is not aligned to the host page size"
            );
            let overlay_end = overlay
                .guest_offset
                .checked_add(u64::try_from(overlay.map_len)?)
                .context("prepared initrd overlay guest extent overflow")?;
            anyhow::ensure!(
                overlay.guest_offset >= range.guest_offset && overlay_end <= range_end,
                "prepared initrd overlay escapes its primary mapping"
            );
            anyhow::ensure!(
                overlay.guest_offset >= previous_overlay_end,
                "prepared initrd overlays overlap or are reordered"
            );
            validate_prepared_backing_extent(
                &overlay.fd,
                overlay.file_offset,
                overlay.map_len,
                host_page_size,
                "prepared initrd overlay",
            )?;
            previous_overlay_end = overlay_end;
        }
        mapped_len = mapped_len
            .checked_add(range.map_len)
            .context("prepared initrd mapped length overflow")?;
    }
    anyhow::ensure!(
        total_compressed <= mapped_len && mapped_len.saturating_sub(total_compressed) < page_size,
        "prepared initrd mapped padding is inconsistent with compressed length"
    );
    load_addr
        .checked_add(mapped_len as u64)
        .context("prepared initrd final guest address overflow")?;
    validate_prepared_stream_magic(compression, ranges)?;
    Ok(boot_size)
}

#[derive(Clone, Copy, Debug)]
struct ValidatedPreparedSubrange {
    guest_addr: u64,
    host_addr: *mut u8,
    file_offset: u64,
    len: usize,
}

#[derive(Debug)]
struct ValidatedPreparedRange {
    range: PreparedMapping,
    subranges: Vec<ValidatedPreparedSubrange>,
    overlays: Vec<ValidatedPreparedOverlay>,
}

#[derive(Debug)]
struct ValidatedPreparedOverlay {
    overlay: PreparedOverlay,
    subranges: Vec<ValidatedPreparedSubrange>,
}

fn validate_prepared_extent_subranges(
    guest_mem: &GuestMemoryMmap,
    guest_addr: u64,
    map_len: usize,
    file_offset: u64,
    split_alignment: usize,
    host_page_size: usize,
) -> Result<Vec<ValidatedPreparedSubrange>> {
    let mut consumed = 0usize;
    let mut subranges = Vec::new();
    for slice in guest_mem.get_slices(GuestAddress(guest_addr), map_len) {
        let slice = slice.context("prepared initrd crosses a guest-memory hole")?;
        let len = slice.len();
        anyhow::ensure!(
            len > 0 && len % split_alignment == 0,
            "prepared initrd crosses a guest-memory region boundary that is not \
             aligned to the {split_alignment}-byte backing boundary"
        );
        let sub_guest = guest_addr
            .checked_add(consumed as u64)
            .context("prepared initrd subrange guest address overflow")?;
        let host_addr = guest_mem
            .get_host_address(GuestAddress(sub_guest))
            .context("resolve prepared initrd subrange host address")?;
        validate_prepared_split_host_address(host_addr, split_alignment)?;
        let sub_file_offset = file_offset
            .checked_add(consumed as u64)
            .context("prepared initrd subrange file offset overflow")?;
        validate_prepared_file_offset(sub_file_offset, host_page_size)
            .context("prepared initrd split produced a misaligned file offset")?;
        subranges.push(ValidatedPreparedSubrange {
            guest_addr: sub_guest,
            host_addr,
            file_offset: sub_file_offset,
            len,
        });
        consumed = consumed
            .checked_add(len)
            .context("prepared initrd split length overflow")?;
    }
    anyhow::ensure!(
        consumed == map_len,
        "prepared initrd guest-memory split did not cover the complete range"
    );
    Ok(subranges)
}

/// Resolve the complete guest-memory split before mutating any VMA. A mapping
/// range may cross adjacent NUMA slots whose host virtual addresses are
/// unrelated, but every resulting subrange must preserve the allocator's
/// backing alignment and cover the source range exactly.
fn validate_prepared_subranges(
    guest_mem: &GuestMemoryMmap,
    ranges: Vec<PreparedMapping>,
    load_addr: u64,
    split_alignment: usize,
    host_page_size: usize,
) -> Result<Vec<ValidatedPreparedRange>> {
    anyhow::ensure!(
        split_alignment.is_power_of_two()
            && host_page_size.is_power_of_two()
            && split_alignment.is_multiple_of(host_page_size),
        "prepared initrd split alignment is incompatible with the host page size"
    );
    let mut validated = Vec::with_capacity(ranges.len());
    for mut range in ranges {
        let guest_addr = load_addr
            .checked_add(range.guest_offset)
            .context("prepared initrd guest address overflow")?;
        let subranges = validate_prepared_extent_subranges(
            guest_mem,
            guest_addr,
            range.map_len,
            range.file_offset,
            split_alignment,
            host_page_size,
        )?;
        let mut overlays = Vec::with_capacity(range.overlays.len());
        for overlay in std::mem::take(&mut range.overlays) {
            let guest_addr = load_addr
                .checked_add(overlay.guest_offset)
                .context("prepared initrd overlay guest address overflow")?;
            let subranges = validate_prepared_extent_subranges(
                guest_mem,
                guest_addr,
                overlay.map_len,
                overlay.file_offset,
                host_page_size,
                host_page_size,
            )?;
            overlays.push(ValidatedPreparedOverlay { overlay, subranges });
        }
        validated.push(ValidatedPreparedRange {
            range,
            subranges,
            overlays,
        });
    }
    Ok(validated)
}

fn map_one_validated_prepared_extent<Map, RestoreNuma>(
    guards: &mut Vec<initramfs::CowOverlayGuard>,
    fd: OwnedFd,
    subranges: Vec<ValidatedPreparedSubrange>,
    description: &str,
    map: &mut Map,
    restore_numa: &mut RestoreNuma,
) -> Result<()>
where
    Map: FnMut(&ValidatedPreparedSubrange, &OwnedFd) -> Result<()>,
    RestoreNuma: FnMut(&ValidatedPreparedSubrange) -> Result<()>,
{
    for subrange in subranges {
        if let Err(error) = map(&subrange, &fd) {
            // An earlier subrange from this fd may already be live. Move
            // the sole shared-lock owner before unwinding.
            guards.push(initramfs::CowOverlayGuard::new(fd));
            return Err(error).with_context(|| {
                format!(
                    "direct-map {description} subrange at guest {:#x} \
                     (len={}, file_offset={:#x})",
                    subrange.guest_addr, subrange.len, subrange.file_offset
                )
            });
        }
        if let Err(error) = restore_numa(&subrange) {
            // MAP_FIXED already installed this file mapping.
            guards.push(initramfs::CowOverlayGuard::new(fd));
            return Err(error).with_context(|| {
                format!(
                    "restore NUMA policy for direct-map {description} \
                     subrange at guest {:#x} (len={})",
                    subrange.guest_addr, subrange.len
                )
            });
        }
    }
    guards.push(initramfs::CowOverlayGuard::new(fd));
    Ok(())
}

/// Consume prevalidated ranges through the one strict production mapping
/// shape. The callback seam lets unit tests observe ordering and injected
/// failure without invoking `MAP_FIXED`; production supplies only
/// `cow_overlay_file_borrowed`, so there is no byte-copy fallback branch.
fn map_validated_prepared_ranges<Map, RestoreNuma>(
    guards: &mut Vec<initramfs::CowOverlayGuard>,
    validated: Vec<ValidatedPreparedRange>,
    mut map: Map,
    mut restore_numa: RestoreNuma,
) -> Result<()>
where
    Map: FnMut(&ValidatedPreparedSubrange, &OwnedFd) -> Result<()>,
    RestoreNuma: FnMut(&ValidatedPreparedSubrange) -> Result<()>,
{
    for validated_range in validated {
        map_one_validated_prepared_extent(
            guards,
            validated_range.range.fd,
            validated_range.subranges,
            "prepared initrd",
            &mut map,
            &mut restore_numa,
        )?;
        for validated_overlay in validated_range.overlays {
            map_one_validated_prepared_extent(
                guards,
                validated_overlay.overlay.fd,
                validated_overlay.subranges,
                "prepared initrd overlay",
                &mut map,
                &mut restore_numa,
            )?;
        }
    }
    Ok(())
}

/// Build the auto-mount cmdline tokens for one disk. Returns an
/// empty string when no auto-mount is requested (Raw filesystem,
/// or `no_auto_mount` opt-out); otherwise returns the
/// space-prefixed `KTSTR_DISK0_FS=... KTSTR_DISK0_MOUNT=...`
/// pair, with `KTSTR_DISK0_RO=1` appended when `read_only` is
/// set.
///
/// Free fn so cfg(test) unit tests cover all branches without
/// driving a full `setup_memory` call.
///
/// Token contract (consumed by
/// `crate::vmm::rust_init::auto_mount_data_disks`):
/// * `KTSTR_DISK0_FS=<cache_tag>` — fstype string for the
///   `mount(2)` syscall. Reuses `Filesystem::cache_tag()` so the
///   on-disk-format identifier and the cmdline value stay in
///   lockstep.
/// * `KTSTR_DISK0_MOUNT=<path>` — guest-side mount point. Driven
///   by `DiskConfig::auto_mount_path` (`/mnt/<name>` when
///   `name` is set, `/mnt/disk0` otherwise).
/// * `KTSTR_DISK0_RO=1` — emitted only when `read_only` is set
///   (matches the host-side virtio-blk F_RO advertisement). The
///   guest sets `MS_RDONLY` proactively rather than letting the
///   kernel fail with -EROFS when bdev RO meets RW mount.
#[allow(dead_code)]
pub(crate) fn disk_auto_mount_cmdline_tokens(disk: &disk_config::DiskConfig) -> String {
    if disk.filesystem == disk_config::Filesystem::Raw || disk.no_auto_mount {
        return String::new();
    }
    let mut s = format!(
        " KTSTR_DISK0_FS={} KTSTR_DISK0_MOUNT={}",
        disk.filesystem.cache_tag(),
        disk.auto_mount_path(),
    );
    if disk.read_only {
        s.push_str(" KTSTR_DISK0_RO=1");
    }
    s
}

/// Guest kernel cmdline flags common to both arches, with the
/// arch-specific tail spliced in. Centralized so a flag added once
/// applies to x86_64 AND aarch64: a per-arch drift here previously left
/// `sysctl.vm.overcommit_memory=1` on x86 only, OOM-ing the aarch64
/// guest /init on its allocator reservation.
///
/// `arch_extra` is the per-arch tail (x86_64: no_timer_check /
/// clocksource / i8042 / pci=off / reboot=k; aarch64: kfence). Callers
/// append dynamic tokens (earlycon, loglevel, rdinit, disk auto-mount,
/// numa, wprof, cmdline_extra) after this base. Cmdline params are
/// order-independent, so the common-then-arch ordering is irrelevant.
fn base_guest_cmdline(arch_extra: &str) -> String {
    // KASLR is ON by default — ktstr.kconfig pins CONFIG_RANDOMIZE_BASE=y
    // (text-image slide; x86 also CONFIG_RANDOMIZE_MEMORY=y for the
    // direct-map slides). The host derives the runtime virt-KASLR offset
    // (x86: MSR_LSTAR readback + KERN_ADDRS _text; aarch64: KERN_ADDRS
    // _text only — no MSR_LSTAR) and threads it via coord_kaslr_offset()
    // into every kaslr-aware site. Tests opt out via
    // #[ktstr_test(kaslr = false)] / Scheduler::kargs(&["nokaslr"]).
    //
    // vm.overcommit_memory=1 (OVERCOMMIT_ALWAYS): the guest /init is a
    // jemalloc-backed test binary that maps more virtual address space
    // than its resident set. Under the default heuristic (mode 0)
    // __vm_enough_memory rejects a single mapping larger than free RAM,
    // so on a deferred-sized guest the /init aborts with "memory
    // allocation of N bytes failed" before the workload runs. ALWAYS mode
    // admits the mapping; the resident set then stays within guest RAM
    // so no OOM-kill follows (a coverage-instrumented /init's larger
    // resident set — the live __llvm_prf_cnts + __llvm_prf_data
    // sections — is covered by the memory_budget coverage reserve).
    // (arm64's arch_mm_preinit auto-enables
    // ALWAYS only for PAGE_SIZE>=16K with <=128 physpages.)
    format!(
        "console=ttyS0 nomodules mitigations=off random.trust_cpu=on \
         swiotlb=noforce panic=-1 lockdown=none \
         sysctl.kernel.unprivileged_bpf_disabled=0 \
         sysctl.kernel.sched_schedstats=1 delayacct \
         sysctl.kernel.task_delayacct=1 sysctl.vm.overcommit_memory=1 \
         {arch_extra} KTSTR_GUEST=1"
    )
}

/// NUMA-balancing cmdline token. The kernel handler
/// `setup_numabalancing` (mm/mempolicy.c) accepts ONLY the strings
/// "enable" / "disable" via `strcmp`; any other value (e.g. "0")
/// leaves the parse result 0, logs `pr_warn("Unable to parse
/// numa_balancing=")`, and is IGNORED — so the kernel keeps its
/// compiled CONFIG_NUMA_BALANCING_DEFAULT_ENABLED state instead of
/// being turned off. Memory-only (CXL) topologies want balancing ON to
/// migrate pages toward CPU-bearing nodes; the uniform/default case
/// wants it OFF to keep scheduler measurements free of migration
/// noise. Extracted (like `base_guest_cmdline`) so the x86_64
/// (`build_guest_cmdline`) and aarch64 (`finish_aarch64_setup`) sites
/// share one definition and a host unit test can pin both branches.
fn numa_balancing_cmdline_token(topology: &crate::vmm::topology::Topology) -> &'static str {
    if topology.has_memory_only_nodes() {
        " numa_balancing=enable"
    } else {
        " numa_balancing=disable"
    }
}

/// Pure helper: assemble the `extras` slice and the [`BaseKey`] from
/// the resolved scheduler/probe/worker/staged-binary paths. Extracted
/// out of [`KtstrVm::prepare_initramfs`] so the staged-extras
/// path-format contract, the per-staged iteration order, and the
/// shell-mode-vs-non-shell BaseKey threading can be unit-tested
/// without spawning the resolve thread or running the full
/// initramfs build.
///
/// Caller responsibilities:
/// - Pre-compute `staged_extras_names` as
///   `format!("{}/scheduler", staged_scheduler_archive_dir(&s.name))`
///   for each staged scheduler (the helper indexes into this vec by
///   position, so caller MUST keep order identical to
///   `staged_schedulers`). Materialized externally so the borrow
///   lifetime ties to the caller's owned Vec.
/// - Pre-compute `merged_includes` (operator's `include_files` plus
///   the optional alloc-worker binary).
/// - Pre-compute `has_jemalloc_extras` = `probe.is_some() ||
///   worker.is_some()` for shell-mode determination.
///
/// Returns `(extras, base_key)`. The extras vec borrows from
/// `scheduler`, `probe`, `staged_extras_names`, and
/// `staged_schedulers` — all `'a`-tied to the caller's lifetimes.
/// The base_key is owned `BaseKey`.
///
/// `#[allow(clippy::too_many_arguments)]` — the parameter set is
/// intrinsically flat (binaries + staging slice + flags); folding
/// into a builder or struct here would just rename the same
/// positional ordering. Sibling precedent: `build_vm_builder_base`
/// in `src/test_support/runtime.rs` uses the same allow for the
/// same reason.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn assemble_extras_and_key<'a>(
    payload: &'a std::path::Path,
    scheduler: Option<&'a std::path::Path>,
    probe: Option<&'a std::path::Path>,
    worker: Option<&'a std::path::Path>,
    staged_schedulers: &'a [crate::vmm::builder::StagedScheduler],
    staged_extras_names: &'a [String],
    merged_includes: &'a [(String, PathBuf)],
    busybox_bytes: Option<&[u8]>,
    has_jemalloc_extras: bool,
) -> Result<(Vec<(&'a str, &'a std::path::Path)>, BaseKey)> {
    debug_assert_eq!(
        staged_schedulers.len(),
        staged_extras_names.len(),
        "staged_schedulers and staged_extras_names must be co-indexed; \
         caller mis-built the extras-names slice"
    );

    let mut extras: Vec<(&str, &std::path::Path)> = Vec::new();
    if let Some(s) = scheduler {
        extras.push(("scheduler", s));
    }
    if let Some(p) = probe {
        extras.push(("bin/ktstr-jemalloc-probe", p));
    }
    for (idx, staged) in staged_schedulers.iter().enumerate() {
        extras.push((staged_extras_names[idx].as_str(), staged.binary.as_path()));
    }

    // Shell-mode determination: busybox flag, non-empty includes,
    // or any jemalloc extras (probe / worker present). Mirrors the
    // pre-extraction logic in prepare_initramfs — kept
    // explicit here so the helper is a closed unit under test
    // without a hidden dependency on the caller's shell_mode
    // computation.
    let shell_mode = busybox_bytes.is_some() || !merged_includes.is_empty() || has_jemalloc_extras;

    let staged_for_key: Vec<(&str, &std::path::Path)> = staged_schedulers
        .iter()
        .map(|s| (s.name.as_str(), s.binary.as_path()))
        .collect();

    let key = if shell_mode {
        BaseKey::new_shell(
            payload,
            scheduler,
            probe,
            worker,
            &staged_for_key,
            merged_includes,
            busybox_bytes,
        )?
    } else {
        BaseKey::new(payload, scheduler, probe, worker, &staged_for_key)?
    };

    Ok((extras, key))
}

impl KtstrVm {
    /// Open the per-test backing file for `disk`, sized to `capacity`.
    /// Shared by both transports — `init_virtio_blk` (aarch64 MMIO) and
    /// `init_virtio_blk_pci` (x86 PCI) — so the disk-template lifecycle
    /// is identical regardless of how the device is presented to the
    /// guest. The fork is on the configured [`disk_config::Filesystem`],
    /// with one override for the template-build VM driver:
    ///
    ///  - **`template_staging_image` set** (internal-only — see
    ///    [`crate::vmm::KtstrVmBuilder::template_staging_image`]): open the
    ///    caller-supplied path RW and hand it to the device. This
    ///    branch exists exclusively for
    ///    `disk_template::build_template_via_vm`: the driver
    ///    materialises a sparse staging image, points the
    ///    template-build guest at it via this field, and recovers
    ///    the now-formatted file after VM exit for
    ///    [`disk_template::store_atomic`]. Bypasses both the
    ///    `Raw` tempfile and `Btrfs` ensure_template branches so
    ///    the template-build VM cannot recursively re-enter the
    ///    cache it is itself populating.
    ///
    ///  - `Raw`: anonymous sparse `tempfile()`. The kernel
    ///    reclaims storage when the device drops the File. No
    ///    cache, no FICLONE.
    ///
    ///  - `Btrfs`: FICLONE-clones the host-cached, guest-formatted
    ///    template into a per-test tempfile under the cache root
    ///    (so FICLONE source and dest share a filesystem), unlinks
    ///    the dest immediately after open so the device sees the
    ///    same anonymous-file semantics as the `Raw` path, and
    ///    hands the open `File` to the `VirtioBlk` device. See
    ///    [`crate::vmm::disk_template`] module docs.
    fn open_blk_backing(
        &self,
        disk: &disk_config::DiskConfig,
        capacity: u64,
    ) -> Result<std::fs::File> {
        if let Some(staging) = self.template_staging_image.as_ref() {
            let f = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(staging)
                .with_context(|| {
                    format!(
                        "open template staging image {} for virtio-blk",
                        staging.display(),
                    )
                })?;
            // Enforce the file-size = advertised-capacity invariant.
            // The in-tree caller (`disk_template::build_template_via_vm`)
            // sizes the staging file via
            // `create_and_size_staging_image` before invoking the
            // builder, so this is normally a no-op. Calling `set_len`
            // here makes the contract local to the device-init path
            // — a caller-supplied staging image that is too small or
            // too large is normalised to `capacity` instead of
            // letting virtio-blk advertise a size that disagrees with
            // the backing file. Sparse-file semantics match the Raw
            // branch below: holes don't consume disk space until
            // written.
            f.set_len(capacity)
                .context("set template staging image length to capacity")?;
            Ok(f)
        } else {
            match disk.filesystem {
                disk_config::Filesystem::Raw => {
                    let f = tempfile::tempfile()
                        .context("create virtio-blk sparse temp backing file")?;
                    // Make sure the file covers the advertised capacity.
                    // set_len creates a sparse file: holes don't consume
                    // disk space until written.
                    f.set_len(capacity)
                        .context("set virtio-blk backing file length")?;
                    Ok(f)
                }
                disk_config::Filesystem::Btrfs => {
                    let template =
                        disk_template::ensure_template(disk_config::Filesystem::Btrfs, capacity)
                            .context("ensure btrfs disk template")?;
                    let cache_root = disk_template::cache_root()
                        .context("resolve disk-template cache root for per-test clone")?;
                    std::fs::create_dir_all(&cache_root)
                        .with_context(|| format!("create cache root {cache_root:?}"))?;
                    // Generate a unique per-test path under the cache
                    // root. Use pid + timestamp_ns + random_u64 so
                    // concurrent tests in the same process and across
                    // processes never collide.
                    let dest = cache_root.join(format!(
                        ".per-test-{pid}-{ns:x}-{rnd:x}.img",
                        pid = std::process::id(),
                        ns = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map(|d| d.as_nanos())
                            .unwrap_or(0),
                        rnd = rand::random::<u64>(),
                    ));
                    let f = disk_template::clone_to_per_test(&template, &dest)
                        .context("FICLONE template into per-test backing")?;
                    // Unlink the dest path immediately. The open File
                    // keeps the inode alive for the device's lifetime;
                    // the kernel reclaims storage on drop, matching the
                    // `tempfile()` semantics of the Raw branch.
                    //
                    // If the unlink fails (very rare — ENOENT means a
                    // peer beat us to it, EACCES means the operator's
                    // cache permissions are broken, EBUSY can come from
                    // some FUSE backings), we keep the open File and
                    // warn — the device still works on the open fd, the
                    // only consequence is a stale path on disk that the
                    // next cache GC sweeps. Do NOT propagate the error,
                    // because the device's per-test backing is already
                    // valid and aborting VM init would be a regression
                    // versus the Raw branch where `tempfile::tempfile()`
                    // returns an already-unlinked file with no failure
                    // mode.
                    if let Err(e) = std::fs::remove_file(&dest) {
                        tracing::warn!(
                            path = %dest.display(),
                            error = %e,
                            "failed to unlink per-test btrfs backing after \
                             FICLONE; the open File still backs the device, \
                             but the leftover path will accumulate in the \
                             cache directory until manual cleanup or the \
                             next disk-template cache GC pass."
                        );
                    }
                    Ok(f)
                }
            }
        }
    }

    /// Construct the aarch64 virtio-MMIO block device for the configured
    /// disk in `self.disk`. (x86_64 routes the disk over virtio-pci — see
    /// `init_virtio_blk_pci`, which is x86-gated so an intra-doc link would not
    /// resolve on this aarch64-only build; aarch64 PCI is a later increment, so
    /// the disk stays on MMIO here.) Returns `Ok(None)` when no disk is
    /// attached.
    ///
    /// On `Ok(Some(_))`, the returned `Arc<PiMutex<VirtioBlk>>` has:
    ///   - the backing file open (sparse temp file when
    ///     `disk.backing_path` is `None`, otherwise the operator-supplied
    ///     path),
    ///   - the file extended to `disk.capacity_bytes()` (so unallocated
    ///     reads return zeros via short-read padding in `handle_read`),
    ///   - the throttle wired in,
    ///   - the irqfd registered with the VM,
    ///   - guest memory set so subsequent `process_requests` calls can
    ///     read/write descriptor data.
    ///
    /// The framework reserves a single MMIO base + IRQ pair
    /// (`VIRTIO_BLK_MMIO_BASE` / `VIRTIO_BLK_IRQ`); the builder's
    /// `.disk()` enforces the single-disk constraint by overwriting
    /// any previous disk on each call.
    #[cfg(not(target_arch = "x86_64"))]
    pub(super) fn init_virtio_blk(
        &self,
        vm: &kvm::KtstrKvm,
        // The selected run-time placement. Performance admission can choose a
        // different equivalent exact plan than the build-time probe, while
        // no-perf admission can choose a fresh LLC mask.
        effective_placement: super::EffectiveRunPlacement<'_>,
    ) -> Result<Option<Arc<PiMutex<virtio_blk::VirtioBlk>>>> {
        let Some(disk) = self.disk.as_ref() else {
            return Ok(None);
        };
        let capacity = disk.capacity_bytes();

        // Throttle sanity gate. `DiskThrottle::validate` rejects
        // burst capacities below their refill rate (which would
        // silently cap the steady-state at the lower capacity
        // instead of the configured rate) and burst capacities set
        // without a refill rate (a one-shot bucket that never
        // refills). Run BEFORE allocating any backing-file resources
        // so a misconfigured throttle bails before disk-side host
        // commitments.
        //
        // The typed `DiskThrottleValidationError` carries the
        // failing dimension (iops/bytes) so callers downcasting via
        // `err.downcast_ref::<DiskThrottleValidationError>()` can
        // route a programmatic recovery without parsing the
        // rendered message.
        disk.throttle
            .validate()
            .map_err(|e| anyhow::anyhow!(e).context("invalid disk throttle"))?;

        // Per-test backing-file allocation forks on the configured
        // [`disk_config::Filesystem`], with one override for the
        // template-build VM driver:
        //
        //  - **`template_staging_image` set** (internal-only — see
        //    [`crate::vmm::KtstrVmBuilder::template_staging_image`]): open the
        //    caller-supplied path RW and hand it to the device. This
        //    branch exists exclusively for
        //    `disk_template::build_template_via_vm`: the driver
        //    materialises a sparse staging image, points the
        //    template-build guest at it via this field, and recovers
        //    the now-formatted file after VM exit for
        //    [`disk_template::store_atomic`]. Bypasses both the
        //    `Raw` tempfile and `Btrfs` ensure_template branches so
        //    the template-build VM cannot recursively re-enter the
        //    cache it is itself populating.
        //
        //  - `Raw`: anonymous sparse `tempfile()`. The kernel
        //    reclaims storage when the device drops the File. No
        //    cache, no FICLONE.
        //
        //  - `Btrfs`: FICLONE-clones the host-cached, guest-formatted
        //    template into a per-test tempfile under the cache root
        //    (so FICLONE source and dest share a filesystem), unlinks
        //    the dest immediately after open so the device sees the
        //    same anonymous-file semantics as the `Raw` path, and
        //    hands the open `File` to the `VirtioBlk` device. See
        //    [`crate::vmm::disk_template`] module docs.
        let backing = self.open_blk_backing(disk, capacity)?;

        let mut blk =
            virtio_blk::VirtioBlk::with_options(backing, capacity, disk.throttle, disk.read_only);
        // Worker placement extracted from the host-topology plan.
        // Perf-mode produces `pinning_plan.service_cpu` (a dedicated
        // host CPU reserved away from vCPU pins) — the worker pins
        // there to keep its cache footprint out of the workload-
        // measured cpuset. Non-perf + `--cpu-cap` supplies the CPUs
        // from the run-time plan acquired immediately before VM setup;
        // the worker shares the LLC but stays inside the exact resource
        // budget this invocation owns.
        // The two effective placement modes are orthogonal; both `None`
        // means inherit the parent's affinity (degraded-sysfs / non-cap-set
        // fallback). The setter only takes effect on the next worker
        // spawn — `with_options` deferred initial spawn to DRIVER_OK
        // (matching the respawn path), so this call lands inside the
        // window and the first worker observes the placement. The
        // Build-time shape plans are never consulted here, so the worker
        // can only bind to the same LLCs the run-scoped flocks hold.
        let placement = virtio_blk::WorkerPlacement {
            service_cpu: effective_placement.service_cpu,
            shared_cpus: effective_placement.shared_cpus.map(<[usize]>::to_vec),
        };
        blk.set_worker_placement(placement);
        blk.set_mem((*vm.guest_mem).clone());
        let blk_arc = Arc::new(PiMutex::new(blk));

        // irqfd registration. On x86's split-irqchip path (max APIC ID > 254)
        // the device IRQ is delivered via the userspace IOAPIC: register_irqfd
        // binds the GSI's eventfd, and the guest's IOAPIC RTE write installs
        // the matching MSI route (see super::x86_64::ioapic + IoapicHandle).
        // On the in-kernel-irqchip (x86 <=254) and aarch64 (GIC) paths the
        // kernel routes the GSI directly. The call is identical on both arches.
        vm.vm_fd
            .register_irqfd(blk_arc.lock().irq_evt(), kvm::VIRTIO_BLK_IRQ)
            .context("register virtio-blk irqfd")?;

        Ok(Some(blk_arc))
    }

    /// Construct the aarch64 virtio-MMIO net device for the configured
    /// network. (x86_64 routes the NIC over virtio-pci — see
    /// `init_virtio_net_pci`, which is x86-gated so an intra-doc link would
    /// not resolve on this aarch64-only build; aarch64 PCI is a later
    /// increment, so the NIC stays on MMIO here.) Returns `Ok(None)` when no
    /// network is
    /// attached. On `Ok(Some(_))` the MAC is in config space, guest
    /// memory is set, and the edge irqfd is registered at
    /// `VIRTIO_NET_IRQ` (the in-kernel GIC routes the GSI). The builder's
    /// `.network()` appends; aarch64 supports a single virtio-MMIO NIC —
    /// build() errors on more than one (multi-NIC PCI is x86-only for now), so
    /// `self.networks.first()` is the lone configured NIC (or none).
    #[cfg(not(target_arch = "x86_64"))]
    pub(super) fn init_virtio_net(
        &self,
        vm: &kvm::KtstrKvm,
    ) -> Result<Option<Arc<PiMutex<virtio_net::VirtioNet>>>> {
        // aarch64 supports a single virtio-MMIO NIC (build() caps networks at 1
        // here); the PCI multi-NIC transport is x86-only for now.
        let Some(cfg) = self.networks.first() else {
            return Ok(None);
        };
        let mut dev = virtio_net::VirtioNet::new(*cfg);
        dev.set_mem((*vm.guest_mem).clone());
        let net_arc = Arc::new(PiMutex::new(dev));
        vm.vm_fd
            .register_irqfd(net_arc.lock().irq_evt(), kvm::VIRTIO_NET_IRQ)
            .context("register virtio-net irqfd")?;
        Ok(Some(net_arc))
    }

    /// Construct one x86_64 virtio-net PCI function per configured NIC and
    /// install each on `pci_bus` at its own slot (`virtio_net_pci_slot(index)`,
    /// slot 0 = host bridge). Returns one [`NetDeviceHandles`] per NIC — an
    /// empty Vec when no network is attached. For each NIC the MAC is in the
    /// device config region, guest memory is set, and INTx is registered by
    /// irqchip mode (gated on max APIC ID, not vCPU count): full in-kernel
    /// irqchip (`<=254` max APIC ID) →
    /// `register_irqfd_with_resample` (KVM tracks the level GSI and
    /// de-asserts on guest EOI via the resample eventfd; the guest's
    /// read-to-clear ISR lowers the device's pending bits); split irqchip
    /// (`>254` max APIC ID) → a plain edge `register_irqfd` (the userspace IOAPIC
    /// delivers edge MSI routes, one per `irq_evt` assert — a resample
    /// irqfd is rejected there). The serialized in-VMM loopback needs no
    /// active resample drain (each interrupt is consumed before the next;
    /// an async RX backend would need one). The returned
    /// [`NetDeviceHandles`] carries the counters Arc (for the failure-dump
    /// run state + `VmResult`) and, on the full-irqchip path, the resample
    /// eventfd the caller MUST keep alive.
    ///
    /// `msix_sink` is the active GSI-route owner as a
    /// [`virtio_msix::MsixRouteSink`]: the [`kvm::FullIrqchipRouteOwner`] on the
    /// full-irqchip path, the [`kvm::IoapicHandle`] on the split-irqchip path
    /// (both impl the trait) — `Some` whenever this is an x86 PCI guest, `None`
    /// otherwise. When `Some`, each NIC additionally gets MSI-X: one eventfd per
    /// vector registered as an irqfd at `virtio_net_msix_gsi(index, v)`, and the
    /// sink threaded into the facade to install the MSI routes on the guest's
    /// vector-unmask edges. When `None`, the facade omits the MSI-X cap so the
    /// NIC stays on INTx. INTx is registered on BOTH irqchip paths regardless —
    /// the guest's fallback if it declines MSI-X (an older driver, `pci=nomsi`).
    #[cfg(target_arch = "x86_64")]
    pub(super) fn init_virtio_net_pci(
        &self,
        vm: &kvm::KtstrKvm,
        pci_bus: &Arc<PiMutex<pci::PciBus>>,
        msix_sink: Option<Arc<dyn virtio_msix::MsixRouteSink>>,
    ) -> Result<Vec<NetDeviceHandles>> {
        // The BAR aperture is the host-bridge _CRS MMIO grant (same window the
        // DSDT advertises to the guest); bar_window rejects a base outside it so
        // a non-conformant guest BAR cannot shadow the ECAM window. Shared by
        // every NIC (the guest assigns each BAR inside this one grant).
        let bar_aperture = (
            kvm::PCI_MMIO_BAR_BASE,
            kvm::PCI_MMIO_BAR_BASE + kvm::PCI_MMIO_BAR_SIZE,
        );
        // One virtio-pci function per configured NIC, each on its own PCI slot
        // (`virtio_net_pci_slot(index)`, slot 0 = host bridge) and INTx GSI
        // (`virtio_net_gsi(index)`, disjoint from the reserved lines, skipping
        // SCI). The builder caps the count at MAX_VIRTIO_NICS so every GSI fits
        // the IOAPIC budget. Returns one NetDeviceHandles per NIC (empty when no
        // network is attached) — the run loop keeps every resample eventfd alive
        // and snapshots every counters Arc.
        let mut handles = Vec::with_capacity(self.networks.len());
        for (index, cfg) in self.networks.iter().enumerate() {
            let gsi = kvm::virtio_net_gsi(index);
            let slot = kvm::virtio_net_pci_slot(index);
            let mut dev = virtio_net::VirtioNet::new(*cfg);
            dev.set_mem((*vm.guest_mem).clone());
            let counters = dev.counters();
            // INTx — the guest's interrupt FALLBACK now that the MSI-X cap is
            // advertised on both irqchip paths (the MSI-X block below). Registered
            // unconditionally so a guest that declines MSI-X (older driver,
            // `pci=nomsi`) still gets IRQs. Its delivery differs by irqchip mode
            // (the kernel's kvm_arch_irqfd_allowed, arch/x86/kvm/irq.c: a RESAMPLE
            // irqfd requires irqchip_full). The mode is gated on max APIC ID
            // (split_irqchip = max_apic_id > MAX_XAPIC_ID), NOT vCPU count — APIC
            // IDs are sparse, so a wide-core sub-254-vCPU guest can be split:
            // - Full in-kernel irqchip (<=254 max APIC ID): register the GSI as a
            //   LEVEL line via register_irqfd_with_resample. KVM's in-kernel
            //   IOAPIC tracks the level and de-asserts on guest EOI, notifying
            //   the resample eventfd; the guest's read-to-clear ISR lowers the
            //   device's pending bits.
            // - Split irqchip (>254 max APIC ID): no in-kernel IOAPIC — the
            //   userspace IOAPIC delivers device IRQs as edge MSI routes, so each
            //   irq_evt assert is exactly one MSI. A plain edge irqfd is correct;
            //   a RESAMPLE irqfd would be rejected with -EINVAL. The guest still
            //   programs this GSI's RTE level/active-low (acpi_pci_irq_enable
            //   hardcodes ACPI_LEVEL_SENSITIVE for x86 PCI INTx), but the
            //   userspace IOAPIC translates it to a one-shot edge MSI with NO
            //   remote-IRR / level re-injection. Correct ONLY because v0's in-VMM
            //   loopback serializes interrupts (one assert consumed before the
            //   next); level re-injection + the active resample handler are the
            //   follow-up before a >254-max-APIC-ID async-RX or multiqueue NIC.
            let resample_evt = if vm.split_irqchip {
                vm.vm_fd
                    .register_irqfd(dev.irq_evt(), gsi)
                    .with_context(|| {
                        format!(
                            "register virtio-net-PCI INTx irqfd (split-irqchip \
                             edge MSI), NIC {index} slot {slot} GSI {gsi}"
                        )
                    })?;
                None
            } else {
                let evt = EventFd::new(libc::EFD_NONBLOCK)
                    .context("create virtio-net resample eventfd")?;
                vm.vm_fd
                    .register_irqfd_with_resample(dev.irq_evt(), &evt, gsi)
                    .with_context(|| {
                        format!(
                            "register virtio-net-PCI INTx resample irqfd (full \
                             irqchip), NIC {index} slot {slot} GSI {gsi}"
                        )
                    })?;
                // The resample eventfd is HELD ALIVE (returned in
                // NetDeviceHandles) but never drained in v0: KVM's in-kernel
                // IOAPIC performs the EOI deassert itself; the only VMM-side
                // requirement is that the fd stays open so the registration
                // remains valid. An active drain handler that re-asserts when the
                // device still has work after EOI is the follow-up (needed before
                // async-RX/multiqueue, where back-to-back unconsumed asserts can
                // race).
                Some(evt)
            };
            // MSI-X (both irqchip paths): build the shared delivery state, create
            // one eventfd per vector and register it as an irqfd at its GSI. The
            // MSI route is installed LATER, on the guest's vector-unmask edge (an
            // irqfd may be assigned to a GSI with no route yet — KVM leaves its
            // cached entry inactive until the route is added, virt/kvm/eventfd.c;
            // the unmask-edge route install refreshes it — irqchip-mode-independent).
            // The eventfds live in the shared state (held alive for the run) so the
            // device core fires them; the route owner (as a `MsixRouteSink`) + the
            // GSIs go to the facade. `msix_sink` is Some on BOTH the full-irqchip
            // (FullIrqchipRouteOwner) and split-irqchip (IoapicHandle) paths for an
            // x86 PCI guest, and None otherwise (no PCI), in which case the facade
            // omits the MSI-X cap and the NIC stays on INTx (no undeliverable MSI-X
            // is advertised).
            // The shared MSI-X state sizes its table to one vector per virtqueue
            // plus config (`num_queues + 1`), capped at the host's per-NIC GSI
            // budget (`kvm::MSIX_VECTORS_PER_NIC` — itself <= the device's table
            // page capacity). The GSI/eventfd allocation below matches that count
            // (`num_vectors`), so the facade's advertised table size, the
            // registered irqfds, and the device's table stay in lockstep.
            let msix = Arc::new(PiMutex::new(virtio_msix::MsixState::new(
                dev.num_queues(),
                kvm::MSIX_VECTORS_PER_NIC,
            )));
            let num_vectors = msix.lock().num_vectors();
            let mut msix_gsis: Vec<u32> = vec![0u32; num_vectors];
            let route_sink: Option<Arc<dyn virtio_msix::MsixRouteSink>> = match &msix_sink {
                Some(sink) => {
                    for (v, gsi_slot) in msix_gsis.iter_mut().enumerate() {
                        let mgsi = kvm::virtio_net_msix_gsi(index, v);
                        *gsi_slot = mgsi;
                        let evt = EventFd::new(libc::EFD_NONBLOCK)
                            .context("create virtio-net MSI-X vector eventfd")?;
                        vm.vm_fd.register_irqfd(&evt, mgsi).with_context(|| {
                            format!(
                                "register virtio-net-PCI MSI-X irqfd, NIC {index} \
                                 vector {v} GSI {mgsi}"
                            )
                        })?;
                        msix.lock().set_eventfd(v, evt);
                    }
                    Some(Arc::clone(sink))
                }
                None => None,
            };
            // Move the device core into the PCI function and install it at its
            // slot. The PciBus lock serializes the vCPU-thread BAR accesses that
            // drive it. The shared MSI-X state is cloned into the device core
            // inside `new`.
            let func =
                virtio_net::VirtioNetPci::new(dev, bar_aperture, msix, route_sink, msix_gsis);
            pci_bus.lock().add_function(slot, Box::new(func));
            handles.push(NetDeviceHandles {
                counters,
                resample_evt,
            });
        }
        Ok(handles)
    }

    /// Construct the x86_64 virtio-blk PCI function (when a disk is configured)
    /// and install it on `pci_bus` at `virtio_blk_pci_slot()`. Returns
    /// [`BlkDeviceHandles`] (the shared device Arc + the INTx resample eventfd),
    /// or `Ok(None)` when no disk is attached.
    ///
    /// The disk-template lifecycle (throttle gate, backing-file open) is shared
    /// verbatim with the aarch64 MMIO path via [`Self::open_blk_backing`]; only
    /// the transport wiring differs. The device core is a SHARED
    /// `Arc<PiMutex<VirtioBlk>>`: one clone is moved into the [`crate::vmm::virtio_blk::VirtioBlkPci`]
    /// function on `pci_bus` (the vCPU thread drives it through BAR MMIO exits),
    /// the other is returned in `BlkDeviceHandles::device` so the run loop can
    /// pause/resume the request worker across the freeze rendezvous, read
    /// counters, and set the parked-eventfd. The MMIO-dispatch `virtio_blk`
    /// handle stays `None` on x86 (guest BAR accesses route through the PCI bus),
    /// exactly as the NIC does.
    ///
    /// INTx is the guest's interrupt FALLBACK (the MSI-X cap is advertised below
    /// whenever `msix_sink` is `Some`). It is registered unconditionally so a
    /// guest that declines MSI-X (older driver, `pci=nomsi`) still gets IRQs;
    /// delivery differs by irqchip mode (gated on max APIC ID, not vCPU count) —
    /// full in-kernel irqchip (`<=254`) → `register_irqfd_with_resample` (KVM
    /// tracks the level GSI and de-asserts on guest EOI); split (`>254`) → a
    /// plain edge `register_irqfd` (the kernel's `kvm_arch_irqfd_allowed` rejects
    /// a resample irqfd there). The single-asserter worker (one drain per kick,
    /// consumed before the next) needs no active resample drain — see
    /// [`crate::vmm::virtio_blk::VirtioBlkPci`] for the full rationale, which mirrors the NIC's INTx.
    ///
    /// `msix_sink` is the active GSI-route owner ([`kvm::FullIrqchipRouteOwner`]
    /// on full-irqchip, [`kvm::IoapicHandle`] on split — both
    /// [`virtio_msix::MsixRouteSink`]s): `Some` for an x86 PCI guest, `None`
    /// otherwise. When `Some`, MSI-X is offered: one eventfd per vector
    /// registered as an irqfd at `virtio_blk_pci_msix_gsi(v)`, the sink threaded
    /// into the facade to install routes on the guest's vector-unmask edges. When
    /// `None`, the facade omits the MSI-X cap and the device stays on INTx.
    #[cfg(target_arch = "x86_64")]
    pub(super) fn init_virtio_blk_pci(
        &self,
        vm: &kvm::KtstrKvm,
        pci_bus: &Arc<PiMutex<pci::PciBus>>,
        msix_sink: Option<Arc<dyn virtio_msix::MsixRouteSink>>,
        // Same selected run-time placement as the MMIO twin.
        effective_placement: super::EffectiveRunPlacement<'_>,
    ) -> Result<Option<BlkDeviceHandles>> {
        let Some(disk) = self.disk.as_ref() else {
            return Ok(None);
        };
        let capacity = disk.capacity_bytes();

        // Throttle sanity gate — identical to the MMIO path in
        // `init_virtio_blk`. Run BEFORE `open_blk_backing` so a misconfigured
        // throttle bails before any backing-file host commitment.
        disk.throttle
            .validate()
            .map_err(|e| anyhow::anyhow!(e).context("invalid disk throttle"))?;

        let backing = self.open_blk_backing(disk, capacity)?;
        let mut blk =
            virtio_blk::VirtioBlk::with_options(backing, capacity, disk.throttle, disk.read_only);
        // Worker placement + guest memory — same as `init_virtio_blk`. Both land
        // before the deferred initial worker spawn (DRIVER_OK), so the first
        // worker observes the placement and guest memory. The run-time-replan
        // selected run-time placement wins over every build-time probe.
        let placement = virtio_blk::WorkerPlacement {
            service_cpu: effective_placement.service_cpu,
            shared_cpus: effective_placement.shared_cpus.map(<[usize]>::to_vec),
        };
        blk.set_worker_placement(placement);
        blk.set_mem((*vm.guest_mem).clone());
        // Wrap once; one Arc clone backs the PciBus function, the other is
        // returned to the run loop. `set_parked_evt` is called by the caller
        // (after this returns), matching the MMIO path's post-init call site.
        let blk_arc = Arc::new(PiMutex::new(blk));

        let slot = kvm::virtio_blk_pci_slot();
        let gsi = kvm::virtio_blk_pci_gsi();
        // The BAR aperture is the host-bridge _CRS MMIO grant (same window the
        // DSDT advertises); `bar_window` rejects an out-of-grant base so a
        // non-conformant guest BAR cannot shadow the ECAM window.
        let bar_aperture = (
            kvm::PCI_MMIO_BAR_BASE,
            kvm::PCI_MMIO_BAR_BASE + kvm::PCI_MMIO_BAR_SIZE,
        );

        // INTx — registered on BOTH irqchip paths (the guest's fallback if it
        // declines MSI-X). The device's `irq_evt` lives behind the device lock;
        // borrow it for the registration call only. Delivery branches on
        // `vm.split_irqchip` exactly as `init_virtio_net_pci` documents.
        let resample_evt = if vm.split_irqchip {
            vm.vm_fd
                .register_irqfd(blk_arc.lock().irq_evt(), gsi)
                .with_context(|| {
                    format!(
                        "register virtio-blk-PCI INTx irqfd (split-irqchip edge \
                         MSI), slot {slot} GSI {gsi}"
                    )
                })?;
            None
        } else {
            let evt =
                EventFd::new(libc::EFD_NONBLOCK).context("create virtio-blk resample eventfd")?;
            vm.vm_fd
                .register_irqfd_with_resample(blk_arc.lock().irq_evt(), &evt, gsi)
                .with_context(|| {
                    format!(
                        "register virtio-blk-PCI INTx resample irqfd (full \
                         irqchip), slot {slot} GSI {gsi}"
                    )
                })?;
            // Held alive in `BlkDeviceHandles::resample_evt` for the run (KVM
            // performs the EOI deassert; the only VMM requirement is the fd stays
            // open). No active drain in v0 — the single-asserter worker never
            // races back-to-back unconsumed asserts.
            Some(evt)
        };

        // MSI-X (both irqchip paths when a sink is wired): the shared delivery
        // state sizes its table to `num_queues + 1` (one request-queue vector +
        // config), capped at the host's per-blk GSI budget
        // (`kvm::MSIX_VECTORS_PER_BLK`). The GSI/eventfd allocation below matches
        // that count, so the facade's advertised table size, the registered
        // irqfds, and the device's table stay in lockstep.
        let msix = Arc::new(PiMutex::new(virtio_msix::MsixState::new(
            blk_arc.lock().num_queues() as usize,
            kvm::MSIX_VECTORS_PER_BLK,
        )));
        let num_vectors = msix.lock().num_vectors();
        let mut msix_gsis: Vec<u32> = vec![0u32; num_vectors];
        let route_sink: Option<Arc<dyn virtio_msix::MsixRouteSink>> = match &msix_sink {
            Some(sink) => {
                for (v, gsi_slot) in msix_gsis.iter_mut().enumerate() {
                    let mgsi = kvm::virtio_blk_pci_msix_gsi(v);
                    *gsi_slot = mgsi;
                    let evt = EventFd::new(libc::EFD_NONBLOCK)
                        .context("create virtio-blk MSI-X vector eventfd")?;
                    vm.vm_fd.register_irqfd(&evt, mgsi).with_context(|| {
                        format!("register virtio-blk-PCI MSI-X irqfd, vector {v} GSI {mgsi}")
                    })?;
                    msix.lock().set_eventfd(v, evt);
                }
                Some(Arc::clone(sink))
            }
            None => None,
        };
        // Move one device Arc clone into the PCI function; install at its slot.
        // The PciBus lock serializes the vCPU-thread BAR accesses that drive it.
        // The shared MSI-X state is cloned into the device core inside `new`.
        let func = virtio_blk::VirtioBlkPci::new(
            Arc::clone(&blk_arc),
            bar_aperture,
            msix,
            route_sink,
            msix_gsis,
        );
        pci_bus.lock().add_function(slot, Box::new(func));
        Ok(Some(BlkDeviceHandles {
            device: blk_arc,
            resample_evt,
        }))
    }

    /// Create the KVM VM and optionally load the kernel.
    ///
    /// When `memory_mib` is `Some`, allocates guest memory and loads the
    /// kernel immediately (existing path). When `None` (deferred), creates
    /// the VM without memory — allocation and kernel loading happen later
    /// in `setup_memory` after the actual initramfs size is known.
    pub(super) fn create_vm_and_load_kernel(
        &self,
        mbind_node_map: &[Vec<usize>],
    ) -> Result<(kvm::KtstrKvm, Option<boot::KernelLoadResult>)> {
        let t0 = Instant::now();

        // `mut` is used only on x86_64, where `vm.pci_enabled` is assigned below;
        // on aarch64 (no PCI field) `vm` is never mutated after construction.
        #[cfg_attr(not(target_arch = "x86_64"), allow(unused_mut))]
        let mut vm = match self.memory_mib {
            // Performance-mode hugepages are opportunistic. The allocator
            // serializes its free-count check with MAP_HUGETLB so a process
            // storm cannot all spend the same observed pool. Passing the
            // explicit-hugepage bit here would bypass that distinction and
            // turn a racy pre-check into a strict no-fallback request.
            Some(mib) => kvm::KtstrKvm::new(self.topology, mib, self.performance_mode)
                .context("create VM")?,
            None => kvm::KtstrKvm::new_deferred(self.topology, false, self.performance_mode)
                .context("create VM (deferred memory)")?,
        };
        tracing::debug!(elapsed_us = t0.elapsed().as_micros(), "kvm_create");

        // Propagate the builder's PCI-enable flag to the VM so the run loops
        // construct the PCI host bridge and the ACPI/cmdline gate on it.
        // x86_64-only: aarch64 `KtstrKvm` has no `pci_enabled` field (the guest
        // is virtio-MMIO + GICv3 with no PCI transport).
        #[cfg(target_arch = "x86_64")]
        {
            vm.pci_enabled = self.pci_enabled;
        }

        // When memory is already allocated (non-deferred path), do mbind
        // and load kernel now. Deferred path does this in setup_memory.
        let kernel_result = if self.memory_mib.is_some() {
            if self.performance_mode && !mbind_node_map.is_empty() {
                let layout = vm.numa_layout.as_ref().expect(
                    "numa_layout is Some on the non-deferred allocation path: \
                     allocate_and_register_memory ran during `vm_new` because \
                     memory_mib was provided up front, and that call sets \
                     numa_layout to Some(...) in src/vmm/{x86_64,aarch64}/kvm.rs",
                );
                layout.mbind_regions(&vm.guest_mem, mbind_node_map);
            }

            let t0 = Instant::now();
            let kr = boot::load_kernel(&vm.guest_mem, &self.kernel).context("load kernel")?;
            tracing::debug!(elapsed_us = t0.elapsed().as_micros(), "load_kernel");
            Some(kr)
        } else {
            None
        };

        Ok((vm, kernel_result))
    }

    /// Resolve the complete immutable initrd before entering exact topology
    /// admission. Cold cells may contend on a shared CAS builder here, but
    /// none of them can sequester CPUs or LLCs while waiting for it.
    pub(super) fn prepare_initramfs(&self) -> Result<Option<PreparedInitrd>> {
        let Some(bin) = self.init_binary.as_ref() else {
            return Ok(None);
        };
        let payload = bin.clone();
        let scheduler = self.scheduler_binary.clone();
        let probe = self.jemalloc_probe_binary.clone();
        let worker = self.jemalloc_alloc_worker_binary.clone();
        let include_files = self.include_files.clone();
        let kernel_config = crate::cache::kernel_config_include_for_image(&self.kernel);
        let staged_schedulers = self.staged_schedulers.clone();
        let busybox_bytes = self.busybox_bytes.clone();
        let compression = self.initrd_compression;
        #[cfg(feature = "wprof")]
        let wprof_host_path: Option<PathBuf> = self.wprof.as_ref().map(|w| w.host_path.clone());
        // Extras are stripped by `build_initramfs_base` before write. The
        // scheduler and probe can lose their DWARF without functional impact
        // — the probe resolves `tsd_s.thread_allocated` offsets against the
        // TARGET process's `/proc/<pid>/exe`, not against its own binary, so
        // its own DWARF is dead weight. The worker (the probe's target) MUST
        // retain DWARF: a stripped worker has no DWARF for the probe to walk.
        // Route scheduler + probe through `extras` (stripped), worker through
        // `include_files` (verbatim). Packing the probe unstripped inflated
        // the initramfs by ~900MB per run in debug builds, which was enough to
        // time out VM init before the test binary loaded.
        //
        // Staged schedulers ride the same `extras` path, packed under
        // `staging/schedulers/<name>/scheduler` so the cpio extractor's silent
        // parent-dir requirement gets satisfied via the auto-registered
        // ancestor entries (see `build_initramfs_base`'s
        // `register_parent_dirs` loop). Each staged binary contributes its own
        // DT_NEEDED set to the shared-lib resolution chain.
        let staged_extras_names: Vec<String> = staged_schedulers
            .iter()
            .map(|s| {
                format!(
                    "{}/scheduler",
                    crate::test_support::staged::staged_scheduler_archive_dir(&s.name),
                )
            })
            .collect();

        // Merge include_files with worker so both the cache key and the actual
        // archive build see the same worker entry; the probe is added to
        // extras inside `assemble_extras_and_key`. wprof (when set) also rides
        // include_files so DT_NEEDED resolution pulls its dynamic dependencies
        // into the archive alongside the binary.
        let mut merged_includes: Vec<(String, PathBuf)> = include_files;
        if let Some((archive_path, host_path)) = kernel_config {
            merged_includes.push((archive_path, host_path));
        }
        if let Some(w) = worker.as_deref() {
            merged_includes.push((
                "bin/ktstr-jemalloc-alloc-worker".to_string(),
                w.to_path_buf(),
            ));
        }
        #[cfg(feature = "wprof")]
        if let Some(wprof_path) = wprof_host_path.as_deref() {
            merged_includes.push(("bin/wprof".to_string(), wprof_path.to_path_buf()));
        }

        let mut extras: Vec<(&str, &std::path::Path)> = Vec::new();
        if let Some(scheduler) = scheduler.as_deref() {
            extras.push(("scheduler", scheduler));
        }
        if let Some(probe) = probe.as_deref() {
            extras.push(("bin/ktstr-jemalloc-probe", probe));
        }
        for (index, staged) in staged_schedulers.iter().enumerate() {
            extras.push((staged_extras_names[index].as_str(), staged.binary.as_path()));
        }

        let t0 = Instant::now();
        let prepared_base = framework_infrastructure(
            prepare_base_inputs(
                &payload,
                &extras,
                &merged_includes,
                busybox_bytes.as_deref(),
            )
            .and_then(|inputs| get_or_prepare_base(inputs, compression)),
        )?;
        let prepared = framework_infrastructure(complete_prepared_initrd(
            prepared_base,
            &self.suffix_params(),
        ))?;
        tracing::debug!(
            elapsed_us = t0.elapsed().as_micros(),
            uncompressed_bytes = prepared.uncompressed_len(),
            compressed_bytes = prepared.compressed_len(),
            cache_hits = prepared.cache_hits(),
            "prepare_initrd",
        );
        Ok(Some(prepared))
    }

    /// Map a prepared initrd's immutable CAS ranges directly into guest RAM.
    /// Returns `total_compressed_size`.
    ///
    /// Every path after a successful `MAP_FIXED` transfers the backing fd to
    /// `vm.cow_overlay_guards`, including later mmap/NUMA-policy errors. This
    /// preserves the CAS object's shared lock for the complete lifetime of
    /// any partial overlay. VM drop order is structural: `_reservation`
    /// unmaps the COW VMAs before the guards release their locks.
    fn load_prepared_initrd(
        &self,
        vm: &mut kvm::KtstrKvm,
        prepared: PreparedInitrd,
        load_addr: u64,
        mbind_node_map: &[Vec<usize>],
    ) -> Result<u32> {
        framework_infrastructure(self.load_prepared_initrd_inner(
            vm,
            prepared,
            load_addr,
            mbind_node_map,
        ))
    }

    fn load_prepared_initrd_inner(
        &self,
        vm: &mut kvm::KtstrKvm,
        prepared: PreparedInitrd,
        load_addr: u64,
        mbind_node_map: &[Vec<usize>],
    ) -> Result<u32> {
        let total_compressed = prepared.compressed_len();
        let page_size = prepared.mapping_granule();
        let compression = prepared.compression();
        let host_page_size = host_page_size() as usize;
        let backing = vm
            .memory_backing
            .context("prepared initrd load requires allocated guest-memory backing")?;
        let split_alignment = prepared_region_split_alignment(backing, host_page_size, page_size);
        let plan = prepared.plan();
        let cache_hits = prepared.cache_hits();
        let ranges = prepared.into_ranges();
        let boot_size = validate_prepared_load(
            total_compressed,
            compression,
            page_size,
            host_page_size,
            load_addr,
            &ranges,
        )?;

        // GuestMemoryMmap is one region per NUMA memory slot; adjacent guest
        // addresses may therefore have unrelated host VAs. Explicit hugetlb
        // VMAs require 2 MiB replacement boundaries. Base-page/THP regions
        // may split on the runtime host page, including caller-declared
        // odd-MiB NUMA boundaries.
        let validated = validate_prepared_subranges(
            &vm.guest_mem,
            ranges,
            load_addr,
            split_alignment,
            host_page_size,
        )?;
        let t0 = Instant::now();
        let restore_numa = self.performance_mode && !mbind_node_map.is_empty();
        let numa_layout = if restore_numa {
            Some(vm.numa_layout.as_ref().context(
                "performance-mode direct-map initrd has NUMA bindings \
                 but no NUMA memory layout",
            )?)
        } else {
            None
        };
        map_validated_prepared_ranges(
            &mut vm.cow_overlay_guards,
            validated,
            |subrange, fd| unsafe {
                initramfs::cow_overlay_file_borrowed(
                    subrange.host_addr,
                    subrange.len,
                    fd,
                    subrange.file_offset,
                )
            },
            |subrange| {
                if let Some(layout) = numa_layout {
                    layout.mbind_replaced_range(
                        subrange.guest_addr,
                        subrange.host_addr,
                        subrange.len,
                        mbind_node_map,
                    )?;
                }
                Ok(())
            },
        )?;
        tracing::debug!(
            elapsed_us = t0.elapsed().as_micros(),
            range_count = vm.cow_overlay_guards.len(),
            cache_hits,
            part_count = plan.part_count,
            direct_ranges = plan.direct_ranges,
            stitch_pages = plan.stitch_pages,
            "initrd_direct_map"
        );

        // Read back first 8 bytes from guest memory to check write.
        let mut check_buf = [0u8; 8];
        vm.guest_mem
            .read_slice(&mut check_buf, GuestAddress(load_addr))
            .context("read-back initrd check")?;
        tracing::debug!(
            first_8 = format!(
                "{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
                check_buf[0],
                check_buf[1],
                check_buf[2],
                check_buf[3],
                check_buf[4],
                check_buf[5],
                check_buf[6],
                check_buf[7]
            ),
            ?compression,
            "initrd_verify",
        );

        Ok(boot_size)
    }

    /// Select the guest rootfs tmpfs fraction for the budget formula by
    /// reading the kernel's version and gating on the honoring versions
    /// (mainline 6.18+ or a stable series at/above its backport floor)
    /// via [`TmpfsFraction::for_kernel_version`].
    ///
    /// This conservatively-derived value is threaded into [`MemoryBudget`]
    /// at every budget call site. The version is read
    /// from the image's own setup_header where it is embedded
    /// ([`read_kernel_version`], the x86_64 bzImage), falling
    /// back to the cache `metadata.json` sidecar
    /// ([`read_kernel_version_from_metadata_sidecar`]) for images without
    /// an embedded version — notably the aarch64 `Image`. Both sources
    /// return `None` on any uncertainty, so
    /// [`TmpfsFraction::for_kernel_version`] yields the safe
    /// [`TmpfsFraction::Half`] unless the kernel is positively confirmed
    /// to honor the token — 90% is never taken on a guess.
    fn tmpfs_fraction(&self) -> TmpfsFraction {
        let version = read_kernel_version(&self.kernel)
            .or_else(|| read_kernel_version_from_metadata_sidecar(&self.kernel));
        TmpfsFraction::for_kernel_version(version)
    }

    /// Validate and load a prepared initramfs into already allocated guest
    /// memory.
    ///
    /// x86_64-only: aarch64 uses
    /// `Self::validate_and_load_initramfs_aarch64`, which computes the
    /// FDT-relative load address from the prepared stream's compressed size.
    #[cfg(target_arch = "x86_64")]
    fn validate_and_load_initramfs(
        &self,
        vm: &mut kvm::KtstrKvm,
        prepared: PreparedInitrd,
        load_addr: u64,
        mbind_node_map: &[Vec<usize>],
    ) -> Result<(Option<u64>, Option<u32>)> {
        let uncompressed_size = prepared.uncompressed_len();
        let compressed_size = prepared.compressed_len();

        // Enforce minimum memory for initramfs extraction.
        // This path is only reached when memory_mib was set explicitly.
        let memory_mib = self.memory_mib.expect(
            "validate_and_load_initramfs called in deferred mode; \
             use compute_memory_and_allocate instead",
        );
        let kernel_init_size = read_kernel_init_size(&self.kernel)?;
        let (init_coverage_instrumented, instrumented_reserve_bytes) = prepared.coverage();
        let budget = MemoryBudget {
            uncompressed_initramfs_bytes: uncompressed_size as u64,
            compressed_initrd_bytes: compressed_size as u64,
            kernel_init_size,
            init_coverage_instrumented,
            instrumented_reserve_bytes,
            tmpfs_fraction: self.tmpfs_fraction(),
        };
        let min_mib = initramfs_min_memory_mib(&budget);
        if memory_mib < min_mib {
            anyhow::bail!(
                "VM memory {}MiB insufficient for initramfs \
                 (uncompressed={}MiB, compressed={}MiB, \
                 init_size={}MiB): need {}MiB",
                memory_mib,
                uncompressed_size >> 20,
                compressed_size >> 20,
                kernel_init_size >> 20,
                min_mib,
            );
        }

        let size = self.load_prepared_initrd(vm, prepared, load_addr, mbind_node_map)?;
        Ok((Some(load_addr), Some(size)))
    }

    /// Deferred memory path: compute memory from the prepared initramfs and
    /// allocate anonymous guest memory. The caller must apply NUMA
    /// policy/prefault and load the kernel before mapping the returned initrd,
    /// otherwise `MADV_POPULATE_WRITE` would eagerly COW every file-backed
    /// initrd page.
    ///
    /// Returns `(prepared_initrd, memory_mib)`.
    ///
    /// x86_64-only: aarch64 uses
    /// `Self::compute_memory_and_allocate_aarch64`, which orders
    /// the load_addr computation after `allocate_and_register_memory`
    /// (the FDT-relative initrd address depends on `memory_mib`,
    /// which is itself computed from the post-compress total size).
    #[cfg(target_arch = "x86_64")]
    fn compute_memory_and_allocate(
        &self,
        vm: &mut kvm::KtstrKvm,
        prepared: PreparedInitrd,
    ) -> Result<(PreparedInitrd, u32)> {
        let uncompressed_size = prepared.uncompressed_len();
        let compressed_size = prepared.compressed_len();

        // Compute memory from actual sizes, honoring the
        // topology-requested minimum when non-zero.
        let kernel_init_size = read_kernel_init_size(&self.kernel)?;
        let (init_coverage_instrumented, instrumented_reserve_bytes) = prepared.coverage();
        let budget = MemoryBudget {
            uncompressed_initramfs_bytes: uncompressed_size as u64,
            compressed_initrd_bytes: compressed_size as u64,
            kernel_init_size,
            init_coverage_instrumented,
            instrumented_reserve_bytes,
            tmpfs_fraction: self.tmpfs_fraction(),
        };
        let memory_mib = initramfs_min_memory_mib(&budget).max(self.memory_min_mib);
        tracing::debug!(
            uncompressed_mib = uncompressed_size >> 20,
            compressed_mib = compressed_size >> 20,
            init_size_mib = kernel_init_size >> 20,
            coverage_instrumented = init_coverage_instrumented,
            coverage_reserve_mib = instrumented_reserve_bytes >> 20,
            memory_min_mib = self.memory_min_mib,
            memory_mib,
            "deferred_memory_computed",
        );

        // Allocate and register guest memory.
        vm.allocate_and_register_memory(memory_mib)
            .with_context(|| format!("allocate deferred memory ({memory_mib}MiB)"))?;

        Ok((prepared, memory_mib))
    }

    pub(super) fn effective_memory_mib(&self, guest_mem: &GuestMemoryMmap) -> u32 {
        use vm_memory::GuestMemoryRegion;
        match self.memory_mib {
            Some(mib) => mib,
            None => {
                let total_bytes: u64 = guest_mem.iter().map(|r| r.len()).sum();
                (total_bytes >> 20) as u32
            }
        }
    }

    /// Try to COW-overlay the compressed base from LZ4 SHM into guest
    /// memory. Returns `Some(CowOverlayGuard)` on success — the guard
    /// owns the SHM fd and holds `LOCK_SH` for the mapping's lifetime,
    /// and MUST be kept alive as long as the COW overlay is in use
    /// (typically the VM lifetime). Validates the segment starts with
    /// LZ4 legacy magic to reject stale data from a previous
    /// compression format.
    ///
    /// Associated function (no `&self`): the COW path is a pure
    /// transform of `(guest_mem, key, expected_len, load_addr)` —
    /// it reads the SHM segment keyed by `key.0` and maps it into
    /// `guest_mem`, touching no VM instance state. Keeping it
    /// `self`-free lets the unit test drive the real overlay logic
    /// without constructing a full `KtstrVm`.
    #[cfg(test)]
    fn try_cow_overlay(
        guest_mem: &GuestMemoryMmap,
        key: &BaseKey,
        expected_len: usize,
        load_addr: u64,
    ) -> Option<initramfs::CowOverlayGuard> {
        let (fd, len) = initramfs::shm_open_lz4(key.0)?;
        if len != expected_len {
            initramfs::shm_close_fd(fd);
            return None;
        }
        // Validate LZ4 legacy magic before COW-mapping. pread the
        // first 4 bytes directly — no need to mmap the entire segment
        // just to peek at the header.
        let mut magic = [0u8; 4];
        // SAFETY: `fd` is owned by `shm_open_lz4` and remains valid
        // until `shm_close_fd` below; `magic` is a 4-byte stack buffer
        // and the read length is exactly 4. The fd refers to a SHM
        // segment with `len >= expected_len` bytes (verified above and
        // by `shm_open_lz4`'s fstat check).
        let n = unsafe {
            libc::pread(
                fd.as_raw_fd(),
                magic.as_mut_ptr() as *mut libc::c_void,
                4,
                0,
            )
        };
        if n != 4 {
            initramfs::shm_close_fd(fd);
            return None;
        }
        if magic != initramfs::LZ4_LEGACY_MAGIC {
            tracing::warn!(
                magic = format!(
                    "{:02x}{:02x}{:02x}{:02x}",
                    magic[0], magic[1], magic[2], magic[3]
                ),
                "stale compressed shm segment in COW path, skipping"
            );
            initramfs::shm_close_fd(fd);
            return None;
        }
        // Refuse zero-length: mmap(len=0) is EINVAL and serves no
        // purpose; the suffix-write fallback handles empty bases
        // trivially. Also refuse load_addr + len overflow before
        // bounds-checking, since GuestAddress arithmetic wraps
        // silently on u64 overflow.
        if len == 0 || load_addr.checked_add(len as u64).is_none() {
            tracing::debug!(
                load_addr = format!("{:#x}", load_addr),
                len,
                "cow_overlay: invalid range (zero-length or overflow), falling back"
            );
            initramfs::shm_close_fd(fd);
            return None;
        }
        // The MAP_FIXED mmap rounds `len` up to the next host page
        // boundary internally — Apple Silicon kernels run with 16 KB
        // pages, so a 5000-byte segment mapped against a 16 KB-page
        // host actually clobbers 16384 bytes of host VA. Bounds-check
        // against the rounded-up length so we don't accept a mapping
        // that overruns the guest region, and reject load_addr that
        // isn't host-page-aligned (mmap returns EINVAL otherwise).
        #[cfg(target_arch = "aarch64")]
        let host_page = host_page_size();
        // x86_64 hosts always run with 4 KB pages, and the call sites
        // page-align load_addr to 4 KB; the rounded-up length matches
        // `len` exactly. Use the constant instead of paying for a
        // sysconf(2) on every overlay attempt.
        #[cfg(target_arch = "x86_64")]
        let host_page: u64 = 0x1000;
        if load_addr & (host_page - 1) != 0 {
            tracing::debug!(
                load_addr = format!("{:#x}", load_addr),
                host_page,
                "cow_overlay: load_addr not host-page-aligned, falling back"
            );
            initramfs::shm_close_fd(fd);
            return None;
        }
        let rounded_len = (len as u64)
            .checked_add(host_page - 1)
            .map(|v| v & !(host_page - 1));
        let Some(rounded_len) = rounded_len else {
            tracing::debug!(
                load_addr = format!("{:#x}", load_addr),
                len,
                "cow_overlay: rounded length overflows u64, falling back"
            );
            initramfs::shm_close_fd(fd);
            return None;
        };
        // Bounds-check [load_addr, load_addr + rounded_len) against
        // guest memory BEFORE the MAP_FIXED mmap. `get_host_address`
        // only validates the start address — without a length check,
        // MAP_FIXED would silently overwrite whatever host VA happens
        // to follow the region (other guest regions, reserved VA, or
        // unrelated mappings). `get_slice` fails if the range extends
        // past the region's end or spans a region boundary, which is
        // exactly the guarantee MAP_FIXED needs.
        let rounded_usize = match usize::try_from(rounded_len) {
            Ok(v) => v,
            Err(_) => {
                tracing::debug!(
                    load_addr = format!("{:#x}", load_addr),
                    rounded_len,
                    "cow_overlay: rounded length exceeds usize, falling back"
                );
                initramfs::shm_close_fd(fd);
                return None;
            }
        };
        if guest_mem
            .get_slice(GuestAddress(load_addr), rounded_usize)
            .is_err()
        {
            tracing::debug!(
                load_addr = format!("{:#x}", load_addr),
                len,
                rounded_len,
                "cow_overlay: range exceeds guest memory region, falling back"
            );
            initramfs::shm_close_fd(fd);
            return None;
        }
        let Ok(host_addr) = guest_mem.get_host_address(GuestAddress(load_addr)) else {
            initramfs::shm_close_fd(fd);
            return None;
        };
        // cow_overlay takes ownership of `fd` on both Some and None
        // paths: on success the guard carries it; on failure
        // cow_overlay itself closes it. Do NOT call shm_close_fd here.
        unsafe { initramfs::cow_overlay(host_addr, len, fd) }
    }

    /// Write cmdline, boot params, and topology tables to guest memory.
    ///
    /// When `kernel_result` is `None` (deferred memory mode), this method
    /// allocates guest memory from the prepared initramfs's exact size, does
    /// mbind, and loads the kernel before the normal initramfs mapping and boot
    /// parameter setup.
    #[cfg(target_arch = "x86_64")]
    pub(super) fn setup_memory(
        &self,
        vm: &mut kvm::KtstrKvm,
        kernel_result: Option<boot::KernelLoadResult>,
        prepared_initrd: Option<PreparedInitrd>,
        mbind_node_map: &[Vec<usize>],
    ) -> Result<boot::KernelLoadResult> {
        // Deferred memory path: join initramfs first to learn its size,
        // then allocate memory, load kernel, and load initramfs — all in
        // one shot with no estimation.
        let (kernel_result, initrd_addr, initrd_size) = if let Some(kr) = kernel_result {
            // Non-deferred: memory already allocated, kernel already loaded.
            // load_prepared_initrd transfers the CowOverlayGuard
            // directly onto vm.cow_overlay_guards before any fallible
            // operation, so a mid-function `?` cannot drop the guard
            // before the COW VMAs are torn down.
            let (initrd_addr, initrd_size) = match prepared_initrd {
                Some(prepared) => {
                    self.validate_and_load_initramfs(vm, prepared, INITRD_ADDR, mbind_node_map)?
                }
                None => (None, None),
            };
            (kr, initrd_addr, initrd_size)
        } else {
            // Deferred memory path: join initramfs first to learn its size,
            // then allocate memory, load kernel, and load initramfs — all in
            // one shot with no estimation.
            let (prepared_initrd, _memory_mib) = match prepared_initrd {
                Some(prepared) => {
                    let (prepared, memory_mib) = self.compute_memory_and_allocate(vm, prepared)?;
                    (Some(prepared), memory_mib)
                }
                None => {
                    // No initramfs — allocate minimum memory.
                    let memory_mib = 256u32;
                    vm.allocate_and_register_memory(memory_mib)
                        .context("allocate deferred memory (no initramfs)")?;
                    (None, memory_mib)
                }
            };

            // This must precede the prepared MAP_FIXED overlay:
            // mbind_regions performs MADV_POPULATE_WRITE across anonymous
            // guest RAM. Running it after the overlay eagerly COWs every
            // initrd page and defeats machine-wide sharing.
            if self.performance_mode && !mbind_node_map.is_empty() {
                let layout = vm.numa_layout.as_ref().expect(
                    "numa_layout is Some after the deferred allocate_and_register_memory \
                     call above: that call sets numa_layout to Some(...) in \
                     src/vmm/{x86_64,aarch64}/kvm.rs before this branch can reach here",
                );
                layout.mbind_regions(&vm.guest_mem, mbind_node_map);
            }

            // Load kernel into the freshly allocated memory.
            let t0 = Instant::now();
            let kr = boot::load_kernel(&vm.guest_mem, &self.kernel).context("load kernel")?;
            tracing::debug!(elapsed_us = t0.elapsed().as_micros(), "load_kernel");

            let (initrd_addr, initrd_size) = match prepared_initrd {
                Some(prepared) => {
                    let size =
                        self.load_prepared_initrd(vm, prepared, INITRD_ADDR, mbind_node_map)?;
                    (Some(INITRD_ADDR), Some(size))
                }
                None => (None, None),
            };

            (kr, initrd_addr, initrd_size)
        };

        // Resolve effective memory_mib for boot params / ACPI / SHM.
        let memory_mib = self.effective_memory_mib(&vm.guest_mem);

        let cmdline = self.build_guest_cmdline();

        let t0 = Instant::now();
        boot::write_cmdline(&vm.guest_mem, &cmdline)?;
        boot::write_boot_params(
            &vm.guest_mem,
            &cmdline,
            memory_mib,
            initrd_addr,
            initrd_size,
            kernel_result.setup_header.as_ref(),
        )?;
        tracing::debug!(elapsed_us = t0.elapsed().as_micros(), "cmdline_boot_params");

        let t0 = Instant::now();
        mptable::setup_mptable(&vm.guest_mem, &self.topology)?;
        let _acpi_layout = acpi::setup_acpi(
            &vm.guest_mem,
            &self.topology,
            vm.numa_layout.as_ref().expect(
                "numa_layout is Some by the time setup_acpi runs: \
                 memory allocation (whether deferred or not) ran earlier \
                 in this function and set numa_layout via \
                 allocate_and_register_memory in src/vmm/x86_64/kvm.rs",
            ),
            vm.pci_enabled,
            self.networks.len(),
            self.disk.is_some(),
        )?;
        tracing::debug!(elapsed_us = t0.elapsed().as_micros(), "mptable_acpi");

        Ok(kernel_result)
    }

    /// Build the guest kernel cmdline (base flags + per-device `virtio_mmio.device=` tokens).
    #[cfg(target_arch = "x86_64")]
    fn build_guest_cmdline(&self) -> String {
        // Kernel cmdline rationale (per flag):
        //   console=ttyS0        — serial console for host-visible output.
        //   nomodules            — no out-of-tree modules are shipped; skip modprobe paths.
        //   mitigations=off      — skip Spectre/Meltdown mitigations for VM perf.
        //   no_timer_check       — suppress APIC timer-calibration failure under KVM.
        //   clocksource=kvm-clock — stable paravirt clock; avoid TSC drift under KVM.
        //   random.trust_cpu=on  — seed RNG from RDRAND so userspace doesn't block on entropy.
        //   swiotlb=noforce      — skip the IOMMU bounce buffer — no passthrough devices.
        //   i8042.*=noaux/nomux/nopnp/dumbkbd — skip legacy PS/2 probing; no keyboard/mouse in VM.
        //   pci=off              — (virtio-MMIO-only guests) skip the PCI scan; dropped when the virtio-PCI transport is enabled.
        //   reboot=k             — use keyboard-controller reset method.
        //   panic=-1             — reboot immediately on panic; host detects via exit.
        //   lockdown=none        — permit /dev/mem and unrestricted BPF needed by the test runtime.
        //   sysctl.kernel.unprivileged_bpf_disabled=0 — allow BPF load from the test runtime.
        //   sysctl.kernel.sched_schedstats=1          — enable /proc/schedstat for workload reports.
        //   delayacct                                 — bare boot param consumed by the
        //                                              kernel's `__setup("delayacct", ...)`
        //                                              handler at kernel/delayacct.c:43-48.
        //                                              The handler sets `delayacct_on = 1`
        //                                              during EARLY boot, BEFORE
        //                                              `delayacct_init()` (line 50-55) reads
        //                                              the variable to decide whether to
        //                                              enable the static branch. This is the
        //                                              authoritative way to turn the
        //                                              delayacct subsystem on at boot.
        //   sysctl.kernel.task_delayacct=1            — backup runtime toggle that flips the
        //                                              delayacct_key static_branch via the
        //                                              `kernel.task_delayacct` sysctl declared
        //                                              at kernel/delayacct.c:80. This path
        //                                              fires later via deferred sysctl
        //                                              registration + proc_handler invocation,
        //                                              which has timing fragility relative to
        //                                              the early-boot increment paths
        //                                              (delayacct_blkio_start/_end gated by
        //                                              static_branch_unlikely(&delayacct_key)
        //                                              at kernel/delayacct.c). Both forms are
        //                                              specified — belt and suspenders — so
        //                                              the runtime toggle is on regardless of
        //                                              whether the early-boot or the deferred
        //                                              sysctl path runs first. Without either,
        //                                              /proc/<tid>/stat field 42 and the
        //                                              taskstats delay-accounting fields stay
        //                                              zero on every kernel built with
        //                                              CONFIG_TASK_DELAY_ACCT=y but boot-time
        //                                              off (the upstream default since v5.14).
        // `pci=off` is dropped when the virtio-PCI transport is enabled so the
        // guest enumerates the PCI host bridge; otherwise it skips the scan.
        let pci_flag = if self.pci_enabled { "" } else { "pci=off " };
        let mut cmdline = base_guest_cmdline(&format!(
            "no_timer_check clocksource=kvm-clock i8042.noaux i8042.nomux \
             i8042.nopnp i8042.dumbkbd {pci_flag}reboot=k"
        ));
        let verbose = std::env::var(crate::KTSTR_VERBOSE_ENV)
            .map(|v| v == "1")
            .unwrap_or(false)
            || std::env::var("RUST_BACKTRACE").is_ok_and(|v| v == "1" || v == "full");
        if verbose {
            cmdline.push_str(" earlyprintk=serial loglevel=7");
        } else {
            cmdline.push_str(" loglevel=0");
        }
        if self.init_binary.is_some() {
            cmdline.push_str(" rdinit=/init initramfs_options=size=90%");
        }
        // Virtio-console MMIO device on the kernel cmdline. The kernel's
        // virtio_mmio_cmdline_devices driver parses this to register the
        // MMIO transport at the given base address and IRQ.
        cmdline.push_str(&format!(
            " virtio_mmio.device={:#x}@{:#x}:{}",
            virtio_console::VIRTIO_MMIO_SIZE,
            kvm::VIRTIO_CONSOLE_MMIO_BASE,
            kvm::VIRTIO_CONSOLE_IRQ,
        ));
        // Virtio-block on x86_64 is a virtio-pci function (see
        // `init_virtio_blk_pci`), enumerated by the guest over ECAM and bound by
        // the virtio-pci driver — it needs NO `virtio_mmio.device=` token (the
        // same reason the NIC emits none; aarch64 keeps blk on virtio-MMIO and
        // emits the token in `finish_aarch64_setup`). The single PCI block
        // function becomes `/dev/vda`. The auto-mount handshake tokens below are
        // transport-independent and still emitted whenever a disk is attached.
        if let Some(disk) = self.disk.as_ref() {
            // Auto-mount handshake. Emit a `KTSTR_DISK0_FS=<tag>`
            // token whenever the first disk has been pre-formatted so
            // the guest init at
            // [`crate::vmm::rust_init::auto_mount_data_disks`]
            // can mount `/dev/vda` at `/mnt/disk0` before the test
            // dispatch runs. `Filesystem::Raw` skips the emission
            // because there is no on-disk fs to mount; the guest
            // sees only the absent token and short-circuits the
            // mount path.
            //
            // `KTSTR_DISK0_RO=1` is emitted when the disk is
            // configured `read_only`. The virtio_blk device
            // advertises `VIRTIO_BLK_F_RO` for that case so the
            // guest's gendisk is RO; mounting RW would fail with
            // `-EROFS` (kernel `do_mount` path: `__btrfs_open_devices`
            // probes the bdev's `bdev_read_only` and returns EROFS
            // when the RW mount tries to write). The token lets the
            // guest set `MS_RDONLY` proactively, surfacing the
            // intent in the cmdline and avoiding the kernel-side
            // EROFS path.
            //
            // The cache_tag() value is reused as the fstype string
            // because it is already kebab-free, ≤8 chars, and
            // matches the on-disk-format identifier the host
            // selected — using the same value for both keeps the
            // guest mount and host cache key in lockstep, so a
            // future `Filesystem` variant rename only has to update
            // one place (the `cache_tag` match in disk_config.rs)
            // and the cmdline / mount automatically follow.
            cmdline.push_str(&disk_auto_mount_cmdline_tokens(disk));
        }
        // No virtio-net cmdline token on x86_64: the NIC is a virtio-pci
        // function (builder `.network()` sets `pci_enabled`), enumerated
        // by the guest over ECAM and bound by the virtio-pci driver — it
        // needs no `virtio_mmio.device=` token. (aarch64 keeps its NIC on
        // virtio-MMIO and emits the token in `finish_aarch64_setup`.)
        cmdline.push_str(numa_balancing_cmdline_token(&self.topology));
        #[cfg(feature = "wprof")]
        if let Some(wprof) = self.wprof.as_ref() {
            cmdline.push_str(" KTSTR_WPROF_ARGS=");
            cmdline.push_str(&wprof.args_cmdline());
        }
        if !self.bpf_map_writes.is_empty() {
            cmdline.push_str(" KTSTR_AWAIT_BPF_MAP_WRITE_READY=1");
        }
        if !self.cmdline_extra.is_empty() {
            cmdline.push(' ');
            cmdline.push_str(&self.cmdline_extra);
        }
        cmdline
    }

    /// Configure BSP and AP vCPUs.
    #[cfg(target_arch = "x86_64")]
    pub(super) fn setup_vcpus(&self, vm: &kvm::KtstrKvm, kernel_entry: u64) -> Result<()> {
        let t0 = Instant::now();
        boot::setup_sregs(&vm.guest_mem, &vm.vcpus[0], vm.split_irqchip)?;
        boot::setup_regs(&vm.vcpus[0], kernel_entry)?;
        boot::setup_fpu(&vm.vcpus[0])?;
        boot::setup_msrs(&vm.vcpus[0], None)?;
        boot::setup_lapic(&vm.vcpus[0], true)?;
        vm.vcpus[0]
            .set_mp_state(kvm_bindings::kvm_mp_state {
                mp_state: kvm_bindings::KVM_MP_STATE_RUNNABLE,
            })
            .context("set BSP mp_state")?;
        tracing::debug!(elapsed_us = t0.elapsed().as_micros(), "bsp_setup");

        let t0 = Instant::now();
        for vcpu in &vm.vcpus[1..] {
            boot::setup_fpu(vcpu)?;
            boot::setup_lapic(vcpu, false)?;
            vcpu.set_mp_state(kvm_bindings::kvm_mp_state {
                mp_state: kvm_bindings::KVM_MP_STATE_UNINITIALIZED,
            })
            .context("set AP mp_state")?;
        }
        tracing::debug!(
            elapsed_us = t0.elapsed().as_micros(),
            ap_count = vm.vcpus.len().saturating_sub(1),
            "ap_setup"
        );

        Ok(())
    }
}

#[cfg(target_arch = "aarch64")]
impl KtstrVm {
    /// Allocate and register guest memory regions for aarch64, including
    /// NUMA-aware placement.
    ///
    /// Uses the same persistent prepared-initrd CAS and direct COW mapping
    /// path as the x86_64 `Self::setup_memory` flow. The shared helpers
    /// ([`Self::load_prepared_initrd`] and the prepared-initrd CAS range
    /// planner) are arch-neutral; this function differs
    /// from the x86_64 driver only in (a) computing the initrd load
    /// address from the dynamic FDT placement (`aarch64_initrd_addr`)
    /// instead of the fixed `INITRD_ADDR`, and (b) handing off to
    /// `finish_aarch64_setup` for FDT writing instead of boot_params /
    /// ACPI emission.
    pub(super) fn setup_memory_aarch64(
        &self,
        vm: &mut kvm::KtstrKvm,
        kernel_result: Option<boot::KernelLoadResult>,
        prepared_initrd: Option<PreparedInitrd>,
        mbind_node_map: &[Vec<usize>],
    ) -> Result<boot::KernelLoadResult> {
        // Deferred memory path for aarch64.
        let (kernel_result, initrd_addr, initrd_size) = if let Some(kr) = kernel_result {
            // Non-deferred: memory already allocated, kernel already loaded.
            // load_prepared_initrd transfers the CowOverlayGuard
            // directly onto vm.cow_overlay_guards before any fallible
            // operation, so a mid-function `?` cannot drop the guard
            // before the COW VMAs are torn down.
            let (initrd_addr, initrd_size) = match prepared_initrd {
                Some(prepared) => {
                    // `self.memory_mib` is required on the non-deferred
                    // path: deferred boots take the early-return branch
                    // below, so we only reach this site after the builder
                    // accepted a concrete `memory_mib`. Surface it as an
                    // error rather than `unwrap()` so a future refactor
                    // that drops the deferred guard fails loudly with an
                    // actionable diagnostic instead of an opaque panic.
                    let memory_mib = self.memory_mib.context(
                        "internal: non-deferred aarch64 path requires memory_mib to be set",
                    )?;
                    self.validate_and_load_initramfs_aarch64(
                        vm,
                        prepared,
                        memory_mib,
                        mbind_node_map,
                    )?
                }
                None => (None, None),
            };
            (kr, initrd_addr, initrd_size)
        } else {
            // Deferred memory path: join initramfs first to learn its
            // size, allocate memory, then load kernel and initramfs.
            let (prepared_initrd, prepared_load_addr) = match prepared_initrd {
                Some(prepared) => {
                    let (prepared, _memory_mib, load_addr) =
                        self.compute_memory_and_allocate_aarch64(vm, prepared)?;
                    (Some(prepared), Some(load_addr))
                }
                None => {
                    // No initramfs — allocate minimum memory.
                    let memory_mib = 256u32;
                    vm.allocate_and_register_memory(memory_mib)
                        .context("allocate deferred memory (no initramfs, aarch64)")?;
                    (None, None)
                }
            };

            if self.performance_mode && !mbind_node_map.is_empty() {
                let layout = vm
                    .numa_layout
                    .as_ref()
                    .expect("numa_layout is Some after deferred aarch64 memory allocation");
                layout.mbind_regions(&vm.guest_mem, mbind_node_map);
            }

            // Load kernel into the freshly allocated memory.
            let t0 = Instant::now();
            let kr =
                boot::load_kernel(&vm.guest_mem, &self.kernel).context("load kernel (aarch64)")?;
            tracing::debug!(elapsed_us = t0.elapsed().as_micros(), "load_kernel");

            let (initrd_addr, initrd_size) = match (prepared_initrd, prepared_load_addr) {
                (Some(prepared), Some(load_addr)) => {
                    let size =
                        self.load_prepared_initrd(vm, prepared, load_addr, mbind_node_map)?;
                    (Some(load_addr), Some(size))
                }
                (None, None) => (None, None),
                _ => anyhow::bail!("deferred aarch64 initrd state is inconsistent"),
            };

            (kr, initrd_addr, initrd_size)
        };

        self.finish_aarch64_setup(vm, kernel_result, initrd_addr, initrd_size)
    }

    /// Non-deferred aarch64 initramfs load: validate that `memory_mib` is
    /// sufficient, compute the FDT-relative load address, then direct-map the
    /// prepared compressed stream into guest memory.
    fn validate_and_load_initramfs_aarch64(
        &self,
        vm: &mut kvm::KtstrKvm,
        prepared: PreparedInitrd,
        memory_mib: u32,
        mbind_node_map: &[Vec<usize>],
    ) -> Result<(Option<u64>, Option<u32>)> {
        let uncompressed_size = prepared.uncompressed_len();
        let compressed_size = prepared.compressed_len();

        // Validate the operator-supplied memory_mib against the
        // initramfs budget. Mirrors the x86_64 validate_and_load_initramfs
        // contract: a builder with too-small memory_mib fails fast here
        // instead of OOMing during boot.
        let kernel_init_size = read_kernel_init_size(&self.kernel)?;
        let (init_coverage_instrumented, instrumented_reserve_bytes) = prepared.coverage();
        let budget = MemoryBudget {
            uncompressed_initramfs_bytes: uncompressed_size as u64,
            compressed_initrd_bytes: compressed_size as u64,
            kernel_init_size,
            init_coverage_instrumented,
            instrumented_reserve_bytes,
            tmpfs_fraction: self.tmpfs_fraction(),
        };
        let min_mib = initramfs_min_memory_mib(&budget);
        if memory_mib < min_mib {
            anyhow::bail!(
                "VM memory {}MiB insufficient for initramfs \
                 (uncompressed={}MiB, compressed={}MiB, \
                 init_size={}MiB): need {}MiB",
                memory_mib,
                uncompressed_size >> 20,
                compressed_size >> 20,
                kernel_init_size >> 20,
                min_mib,
            );
        }

        let load_addr = aarch64_initrd_addr(
            memory_mib,
            self.topology.total_cpus(),
            compressed_size as u64,
        )?;
        let size = self.load_prepared_initrd(vm, prepared, load_addr, mbind_node_map)?;
        Ok((Some(load_addr), Some(size)))
    }

    /// Deferred aarch64 memory setup: compute the memory budget, allocate
    /// anonymous guest memory, and return the prepared image plus its final
    /// address. The caller applies NUMA policy/prefault and loads the kernel
    /// before installing the file-backed COW mappings.
    fn compute_memory_and_allocate_aarch64(
        &self,
        vm: &mut kvm::KtstrKvm,
        prepared: PreparedInitrd,
    ) -> Result<(PreparedInitrd, u32, u64)> {
        let uncompressed_size = prepared.uncompressed_len();
        let compressed_size = prepared.compressed_len();

        let kernel_init_size = read_kernel_init_size(&self.kernel)?;
        let (init_coverage_instrumented, instrumented_reserve_bytes) = prepared.coverage();
        let budget = MemoryBudget {
            uncompressed_initramfs_bytes: uncompressed_size as u64,
            compressed_initrd_bytes: compressed_size as u64,
            kernel_init_size,
            init_coverage_instrumented,
            instrumented_reserve_bytes,
            tmpfs_fraction: self.tmpfs_fraction(),
        };
        let memory_mib = initramfs_min_memory_mib(&budget).max(self.memory_min_mib);
        tracing::debug!(
            uncompressed_mib = uncompressed_size >> 20,
            compressed_mib = compressed_size >> 20,
            init_size_mib = kernel_init_size >> 20,
            coverage_instrumented = init_coverage_instrumented,
            coverage_reserve_mib = instrumented_reserve_bytes >> 20,
            memory_min_mib = self.memory_min_mib,
            memory_mib,
            "deferred_memory_computed",
        );

        vm.allocate_and_register_memory(memory_mib)
            .with_context(|| format!("allocate deferred memory ({memory_mib}MiB, aarch64)"))?;

        // Compute load_addr only AFTER memory_mib is known: it determines
        // the FDT position and thus pvtime_base, and the initrd now sits
        // just below pvtime_base (the PVTIME carve), not the FDT.
        let load_addr = aarch64_initrd_addr(
            memory_mib,
            self.topology.total_cpus(),
            compressed_size as u64,
        )?;

        Ok((prepared, memory_mib, load_addr))
    }

    #[cfg(target_arch = "aarch64")]
    fn finish_aarch64_setup(
        &self,
        vm: &kvm::KtstrKvm,
        kernel_result: boot::KernelLoadResult,
        initrd_addr: Option<u64>,
        initrd_size: Option<u32>,
    ) -> Result<boot::KernelLoadResult> {
        let memory_mib = self.effective_memory_mib(&vm.guest_mem);

        // Kernel cmdline rationale (per flag) — aarch64 subset of the
        // x86_64 block above. Flags present on both arches carry the
        // same justification; see the x86_64 comment for details.
        // aarch64-specific:
        //   kfence.sample_interval=0 — disable KFENCE sampling; no real
        //                              driver faults to catch in the
        //                              test VM, and KFENCE adds boot-time
        //                              page-allocation pressure.
        let mut cmdline = base_guest_cmdline("kfence.sample_interval=0");
        // earlycon is always enabled so the kernel has a console from
        // the earliest boot stage. Without it, stdout-path auto-detection
        // is the only path to early output — and that can fail silently
        // if the FDT node isn't matched by OF_EARLYCON_DECLARE.
        // earlycon base is derived from SERIAL_MMIO_BASE so it tracks the
        // device-window placement (aarch64/kvm.rs) and can never drift.
        cmdline.push_str(&format!(
            " earlycon=uart,mmio,{:#x}",
            aarch64::kvm::SERIAL_MMIO_BASE
        ));
        let verbose = std::env::var(crate::KTSTR_VERBOSE_ENV)
            .map(|v| v == "1")
            .unwrap_or(false)
            || std::env::var("RUST_BACKTRACE").is_ok_and(|v| v == "1" || v == "full");
        if verbose {
            cmdline.push_str(" loglevel=7");
        } else {
            cmdline.push_str(" loglevel=0");
        }
        if self.init_binary.is_some() {
            cmdline.push_str(" rdinit=/init initramfs_options=size=90%");
        }
        // Auto-mount tokens for the configured disk. aarch64 advertises
        // the virtio-blk MMIO transport via FDT (see
        // `create_fdt(..., self.disk.is_some(), ...)` below), so the
        // `virtio_mmio.device=` cmdline form used on x86_64 is omitted.
        // The `KTSTR_DISK0_*` tokens, however, are env-style markers
        // consumed by the guest init at
        // `crate::vmm::rust_init::auto_mount_data_disks` — they are
        // arch-neutral and required on aarch64 for the same auto-mount
        // contract as x86_64.
        if let Some(disk) = self.disk.as_ref() {
            cmdline.push_str(&disk_auto_mount_cmdline_tokens(disk));
        }
        cmdline.push_str(numa_balancing_cmdline_token(&self.topology));
        #[cfg(feature = "wprof")]
        if let Some(wprof) = self.wprof.as_ref() {
            cmdline.push_str(" KTSTR_WPROF_ARGS=");
            cmdline.push_str(&wprof.args_cmdline());
        }
        if !self.bpf_map_writes.is_empty() {
            cmdline.push_str(" KTSTR_AWAIT_BPF_MAP_WRITE_READY=1");
        }
        if !self.cmdline_extra.is_empty() {
            cmdline.push(' ');
            cmdline.push_str(&self.cmdline_extra);
        }

        let t0 = Instant::now();
        boot::validate_cmdline(&cmdline)?;

        let fdt_addr = aarch64::fdt::fdt_address(memory_mib);

        // Wire KVM PV stolen-time so the guest's /proc/stat steal
        // advances under cpu_budget overcommit. The region is carved
        // from the top of guest RAM (just below the FDT); create_fdt
        // below shrinks the /memory node to pvtime_base via the same
        // helper so the guest never reuses it. setup_pvtime gates on
        // host support (has_device_attr) and skips cleanly otherwise.
        let pvtime_base = aarch64::fdt::pvtime_base(memory_mib, self.topology.total_cpus());
        anyhow::ensure!(
            pvtime_base >= aarch64::kvm::DRAM_START && pvtime_base < fdt_addr,
            "guest RAM too small to carve the PVTIME region \
             (pvtime_base={pvtime_base:#x}, fdt_addr={fdt_addr:#x})"
        );
        vm.setup_pvtime(pvtime_base)
            .context("wire KVM PV stolen-time")?;

        let mpidrs =
            aarch64::topology::read_mpidrs(&vm.vcpus).context("read vCPU MPIDRs for FDT")?;
        let guest_l1_unified = aarch64::topology::host_l1_is_unified();
        let dtb = aarch64::fdt::create_fdt(
            &self.topology,
            &mpidrs,
            memory_mib,
            &cmdline,
            initrd_addr,
            initrd_size,
            guest_l1_unified,
            vm.numa_layout.as_ref().expect(
                "numa_layout is Some by the time FDT creation runs: \
                 memory allocation (whether deferred or not) ran earlier \
                 in this function and set numa_layout via \
                 allocate_and_register_memory in src/vmm/aarch64/kvm.rs",
            ),
            self.disk.is_some(),
            !self.networks.is_empty(),
            vm.has_pmu,
        )
        .context("create FDT")?;
        vm.guest_mem
            .write_slice(&dtb, GuestAddress(fdt_addr))
            .context("write FDT to guest memory")?;
        tracing::debug!(
            elapsed_us = t0.elapsed().as_micros(),
            fdt_addr,
            fdt_len = dtb.len(),
            "cmdline_fdt",
        );

        Ok(kernel_result)
    }

    #[cfg(target_arch = "aarch64")]
    pub(super) fn setup_vcpus_aarch64(&self, vm: &kvm::KtstrKvm, kernel_entry: u64) -> Result<()> {
        let t0 = Instant::now();
        let memory_mib = self.effective_memory_mib(&vm.guest_mem);
        let fdt_addr = aarch64::fdt::fdt_address(memory_mib);
        boot::setup_regs(&vm.vcpus[0], kernel_entry, fdt_addr)?;
        tracing::debug!(elapsed_us = t0.elapsed().as_micros(), "bsp_setup");
        // APs start powered off via PSCI — no register setup needed.
        Ok(())
    }
}

/// Per-VM halt-poll interval to apply via `KVM_CAP_HALT_POLL`, or `None` to
/// leave the host's `kvm.halt_poll_ns` module default in place.
///
/// Keyed on signals resolved by the time `KtstrVm::run` calls this — the
/// build-time mode flags plus the run-time `acquire_run_locks` outcome
/// (`default_shared_cpu_claim` is true for default-style admission, including
/// an exact pin whose lifetime reservation was converted to CPU-SH):
///
/// * `no_perf_mode` → `Some(0)`: the guest deliberately shares host CPUs with
///   peers, so halt polling burns CPU that belongs to others.
/// * `default_shared_cpu_claim` → `Some(0)`: an interactive shell configured
///   from a performance builder still uses default-style shared admission, so
///   the acquired claim outranks the build-time mode flag.
/// * `performance_mode` → `None`: an actually isolated perf run disables HLT exits and enables the
///   guest's own haltpoll cpuidle (see `tune_kvm_caps`), which drives
///   `MSR_KVM_POLL_CONTROL` — host halt polling is redundant, leave the module
///   default.
/// * default mode, exact pin or shared fallback → `Some(0)`: both retain
///   CPU-SH, so compatible peers may overlap later and polling would burn
///   their shared time.
///
/// `no_perf_mode` is checked first: `build()` forces `performance_mode=false`
/// under it, and default-style shared admission is a separate mode outcome.
pub(super) fn halt_poll_policy(
    no_perf_mode: bool,
    performance_mode: bool,
    default_shared_cpu_claim: bool,
) -> Option<u64> {
    if no_perf_mode {
        return Some(0);
    }
    if default_shared_cpu_claim {
        return Some(0);
    }
    if performance_mode {
        return None;
    }
    None
}

#[cfg(test)]
mod tests;
