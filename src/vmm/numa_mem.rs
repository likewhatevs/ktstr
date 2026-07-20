use anyhow::{Context, Result};
use std::os::fd::OwnedFd;
use std::path::{Path, PathBuf};
use vm_memory::mmap::{GuestRegionMmap, MmapRegion};
use vm_memory::{GuestAddress, GuestMemory, GuestMemoryMmap};

use super::topology::Topology;

/// Owns a VA reservation created via `mmap(PROT_NONE)`. Drop calls
/// `munmap` on the entire reservation, releasing all MAP_FIXED
/// sub-mappings within it.
pub(crate) struct ReservationGuard {
    addr: *mut libc::c_void,
    size: usize,
}

unsafe impl Send for ReservationGuard {}
unsafe impl Sync for ReservationGuard {}

impl Drop for ReservationGuard {
    fn drop(&mut self) {
        if !self.addr.is_null() && self.addr != libc::MAP_FAILED {
            unsafe {
                libc::munmap(self.addr, self.size);
            }
        }
    }
}

/// Result of [`NumaMemoryLayout::allocate`].
pub(crate) struct AllocatedMemory {
    pub guest_mem: GuestMemoryMmap,
    pub reservation: ReservationGuard,
    pub backing: MemoryBacking,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MemoryBacking {
    BasePages,
    HugeTlb2M,
}

/// Flags for each anonymous node mapping installed into the VA reservation.
///
/// Base-page guest RAM is demand-paged and intentionally does not reserve
/// swap/commit for its full advertised size. This matters under a VM storm:
/// most of a guest's address space remains untouched, so charging every
/// declared MiB up front would reject useful oversubscription despite ample
/// resident memory. Explicit hugetlb mappings are backed by the reserved
/// hugepage pool and therefore do not use `MAP_NORESERVE`.
fn anonymous_node_map_flags(use_hugepages: bool) -> libc::c_int {
    let mut flags = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_FIXED;
    if use_hugepages {
        flags |= libc::MAP_HUGETLB | libc::MAP_HUGE_2MB;
    } else {
        flags |= libc::MAP_NORESERVE;
    }
    flags
}

const HUGEPAGE_ALLOCATION_LOCK: &str = "ktstr-hugepage-allocation-v1.lock";

fn hugepage_allocation_lock_path() -> PathBuf {
    crate::cache::resolve_lock_dir().join(HUGEPAGE_ALLOCATION_LOCK)
}

fn acquire_hugepage_allocation_lock_at(path: &Path) -> Result<OwnedFd> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).with_context(|| {
            format!(
                "create hugepage allocation lock directory {}",
                parent.display()
            )
        })?;
    }
    crate::flock::block_flock(path, crate::flock::FlockMode::Exclusive)
        .with_context(|| format!("lock hugepage allocation transaction {}", path.display()))
}

fn acquire_hugepage_allocation_lock() -> Result<OwnedFd> {
    acquire_hugepage_allocation_lock_at(&hugepage_allocation_lock_path())
}

#[derive(Debug)]
struct HugepageReservationUnavailable {
    operation: String,
    source: std::io::Error,
}

impl std::fmt::Display for HugepageReservationUnavailable {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}: {}", self.operation, self.source)
    }
}

impl std::error::Error for HugepageReservationUnavailable {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

struct MappedReservation {
    guard: ReservationGuard,
    base: *mut libc::c_void,
    va_spans: Vec<usize>,
}

/// Per-NUMA-node guest physical address range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeRegion {
    /// NUMA node index (0-based), matching the index into
    /// `Topology::nodes`. Nodes with `memory_mib == 0` are omitted.
    pub node_id: u32,
    /// Guest physical address where this node's memory starts.
    pub gpa_start: u64,
    /// Size in bytes.
    pub size: u64,
    /// KVM memory slot index for this region.
    pub slot: u32,
}

/// Per-node GPA layout with per-node MAP_FIXED mmaps within a
/// contiguous VA reservation.
///
/// A PROT_NONE VA reservation covers the total memory range. Each
/// node's sub-range is replaced via MAP_FIXED with a real
/// PROT_READ|PROT_WRITE mapping, individually mbind'd and
/// registered as a separate KVM memory slot. The `ReservationGuard`
/// owns the VA range and munmaps it on drop.
///
/// Contiguity is maintained by the VA reservation: all node regions
/// occupy adjacent sub-ranges of the same contiguous VA.
#[derive(Debug, Clone)]
pub struct NumaMemoryLayout {
    /// Per-node regions sorted by ascending GPA. Regions are
    /// GPA-contiguous EXCEPT they skip the MMIO gap
    /// `[mmio_gap.0, mmio_gap.1)` (the x86 sub-4GB device hole): a node
    /// whose RAM crosses the gap is split into a below-gap region and an
    /// above-gap region relocated to `gap_end`, both carrying the same
    /// `node_id`. Total RAM is preserved (the in-gap bytes move above
    /// the gap, not dropped). The host VA backing stays packed and
    /// contiguous (sum of region sizes); see [`Self::allocate`].
    regions: Vec<NodeRegion>,
}

/// Map a gap-free linear RAM offset (0-based within guest DRAM) to its
/// guest physical address, relocating offsets at/after the MMIO gap to
/// above `gap_end` so no RAM lands in the `[gap_start, gap_end)`
/// device-MMIO hole (where virtio + IOAPIC + LAPIC live). This is the
/// shared split primitive: `compute` (KVM slot GPAs) and the e820
/// builder both derive their below/above boundary from the same gap so
/// the host memslots and the guest e820 agree byte-for-byte.
pub(crate) fn linear_to_gpa(linear: u64, dram_base: u64, mmio_gap: Option<(u64, u64)>) -> u64 {
    match mmio_gap {
        Some((gap_start, gap_end)) if dram_base + linear >= gap_start => {
            gap_end + (dram_base + linear - gap_start)
        }
        _ => dram_base + linear,
    }
}

/// Append the GPA region(s) for one node of `size` bytes occupying the
/// gap-free linear range `[*linear, *linear + size)`, advancing
/// `*linear` by `size`. Yields two regions (same `node_id`, the second
/// relocated to `gap_end`) when the node straddles the MMIO-gap
/// boundary, else one. `slot` is the dense running region index.
fn push_node_regions(
    regions: &mut Vec<NodeRegion>,
    node_id: u32,
    size: u64,
    linear: &mut u64,
    dram_base: u64,
    mmio_gap: Option<(u64, u64)>,
) {
    // Linear offset at which the gap begins; pieces below it stay in
    // place, pieces at/after it relocate above `gap_end`.
    let boundary = mmio_gap.map(|(gap_start, _)| gap_start - dram_base);
    let mut remaining = size;
    while remaining > 0 {
        let ls = *linear;
        let piece = match boundary {
            Some(b) if ls < b => (b - ls).min(remaining),
            _ => remaining,
        };
        regions.push(NodeRegion {
            node_id,
            gpa_start: linear_to_gpa(ls, dram_base, mmio_gap),
            size: piece,
            slot: regions.len() as u32,
        });
        *linear += piece;
        remaining -= piece;
    }
}

/// Split `total_mib` across `nodes` nodes so every node's size is a
/// whole 2 MiB hugepage multiple (the last node absorbs a sub-2 MiB
/// remainder when `total_mib` is odd), the sum is exactly `total_mib`,
/// and the split is as even as possible. Returns per-node MiB.
///
/// Hugepage-aligned per-node sizes keep both the guest GPA and the host
/// VA on 2 MiB boundaries. KVM installs a 2 MiB EPT entry only when a
/// slot's base GPA, end GPA, and the GPA-vs-HVA remainder are all 2 MiB
/// congruent (arch/x86/kvm/x86.c `kvm_alloc_memslot_metadata`); a
/// non-2 MiB node base/end or a GPA/HVA incongruence drops the WHOLE
/// node slot to 4 KiB EPT, defeating the hugepage backing the perf path
/// requested. Snapping the (approximate) even split to hugepage
/// boundaries keeps every node fully 2 MiB-backed.
///
/// When `total_mib < 2 * nodes` a node cannot get even one hugepage, so
/// the plain even split is returned unsnapped: hugepages are pointless
/// at that size and snapping would zero out nodes.
fn hugepage_even_split_mib(total_mib: u32, nodes: u32) -> Vec<u32> {
    let plain_even_split = || -> Vec<u32> {
        let per = total_mib / nodes;
        (0..nodes)
            .map(|i| {
                if i == nodes - 1 {
                    total_mib - per * (nodes - 1)
                } else {
                    per
                }
            })
            .collect()
    };
    if nodes <= 1 {
        return vec![total_mib];
    }
    if total_mib < 2 * nodes {
        // Tripwire: total_mib < nodes makes the plain even split zero out
        // non-last nodes (per = 0), silently dropping declared NUMA nodes
        // from the layout. Unreachable in practice (the memory budget
        // floors per-node RAM well above 1 MiB), but fail loud if it ever
        // becomes reachable rather than silently losing nodes.
        debug_assert!(
            total_mib >= nodes,
            "total_mib ({total_mib}) < numa_nodes ({nodes}): even split would zero-size nodes"
        );
        return plain_even_split();
    }
    let total_hp = total_mib / 2; // whole 2 MiB hugepages
    let leftover_mib = total_mib % 2; // 0 or 1 MiB sub-hugepage tail
    let base_hp = total_hp / nodes;
    let extra = total_hp % nodes; // first `extra` nodes get one more hugepage
    (0..nodes)
        .map(|i| {
            let hp = base_hp + u32::from(i < extra);
            // The last node absorbs the sub-2 MiB tail (its end GPA is
            // then non-2 MiB, costing only its final hugepage, not the
            // whole slot).
            hp * 2 + if i == nodes - 1 { leftover_mib } else { 0 }
        })
        .collect()
}

impl NumaMemoryLayout {
    const HUGE_2MB: u64 = 2 * 1024 * 1024;

    /// Explicit hugetlb VMAs can only be split/replaced at 2 MiB boundaries.
    /// Every internal guest-memory region therefore has to start on that
    /// boundary. An odd final tail is fine: it introduces no later boundary
    /// and Linux rounds its hugetlb VMA span up behind the exact KVM slot.
    fn hugetlb_compatible(&self) -> bool {
        self.regions
            .iter()
            .all(|region| region.gpa_start.is_multiple_of(Self::HUGE_2MB))
    }

    fn hugepages_needed(&self) -> u64 {
        self.regions
            .iter()
            .map(|region| region.size.div_ceil(Self::HUGE_2MB))
            .sum()
    }

    fn choose_memory_backing(
        &self,
        use_hugepages: bool,
        performance_mode: bool,
        hugepages_free: u64,
    ) -> Result<MemoryBacking> {
        anyhow::ensure!(
            !use_hugepages || self.hugetlb_compatible(),
            "explicit hugepage backing requires every internal NUMA boundary \
             to be 2 MiB-aligned"
        );
        if self.hugetlb_compatible()
            && (use_hugepages || (performance_mode && hugepages_free >= self.hugepages_needed()))
        {
            Ok(MemoryBacking::HugeTlb2M)
        } else {
            Ok(MemoryBacking::BasePages)
        }
    }

    /// Compute per-node GPA ranges from a topology and total memory.
    ///
    /// `dram_base`: GPA where guest RAM starts (0 on x86_64,
    /// `DRAM_START` on aarch64).
    ///
    /// `total_memory_mib`: total guest memory in MiB. For `with_nodes`
    /// topologies, must equal the sum of all `NumaNode::memory_mib`.
    /// For uniform topologies, memory is divided evenly across
    /// `numa_nodes` nodes.
    /// `mmio_gap`: `Some((gap_start, gap_end))` on x86_64 (the sub-4GB
    /// device-MMIO hole `[0xC000_0000, 0x1_0000_0000)`); `None` on
    /// aarch64 (no low MMIO hole inside DRAM). RAM that would land in
    /// the gap is relocated above `gap_end` so the host registers no
    /// memslot over the device window (otherwise RAM shadows virtio +
    /// IOAPIC and their MMIO never traps).
    pub fn compute(
        topo: &Topology,
        total_memory_mib: u32,
        dram_base: u64,
        mmio_gap: Option<(u64, u64)>,
    ) -> Result<Self> {
        let total_bytes = (total_memory_mib as u64) << 20;
        let numa_nodes = topo.numa_nodes;

        let mut regions = Vec::new();
        // Gap-free running offset into guest DRAM; GPAs are derived from
        // it via `linear_to_gpa` so the gap is skipped exactly once.
        let mut linear = 0u64;

        match topo.nodes {
            Some(nodes) => {
                let node_total_mib: u32 = nodes.iter().map(|n| n.memory_mib).sum();
                anyhow::ensure!(
                    total_memory_mib == node_total_mib,
                    "total_memory_mib ({total_memory_mib}) must equal \
                     sum of node memory_mib ({node_total_mib})"
                );

                for (i, node) in nodes.iter().enumerate() {
                    let size = (node.memory_mib as u64) << 20;
                    if size == 0 {
                        continue;
                    }
                    push_node_regions(
                        &mut regions,
                        i as u32,
                        size,
                        &mut linear,
                        dram_base,
                        mmio_gap,
                    );
                }

                anyhow::ensure!(
                    !regions.is_empty(),
                    "at least one node must have non-zero memory"
                );
            }
            None => {
                if numa_nodes <= 1 {
                    push_node_regions(
                        &mut regions,
                        0,
                        total_bytes,
                        &mut linear,
                        dram_base,
                        mmio_gap,
                    );
                } else {
                    // Even split, snapped to 2 MiB hugepage multiples so
                    // a perf-mode hugepage backing gets full 2 MiB EPT
                    // per node (see hugepage_even_split_mib). The sum is
                    // preserved, so e820/SRAT advertised RAM is unchanged.
                    for (i, mib) in hugepage_even_split_mib(total_memory_mib, numa_nodes)
                        .into_iter()
                        .enumerate()
                    {
                        let size = (mib as u64) << 20;
                        push_node_regions(
                            &mut regions,
                            i as u32,
                            size,
                            &mut linear,
                            dram_base,
                            mmio_gap,
                        );
                    }
                }
            }
        }

        // Relocate preserves total RAM (in-gap bytes move above the gap,
        // never dropped). This is the qemu invariant ram_size == below +
        // above; it also guards the e820 / SRAT consumers, which read
        // sum(region.size) as the advertised RAM.
        debug_assert_eq!(
            regions.iter().map(|r| r.size).sum::<u64>(),
            total_bytes,
            "relocate must preserve total guest RAM"
        );

        Ok(Self { regions })
    }

    /// Per-node regions sorted by ascending GPA.
    pub fn regions(&self) -> &[NodeRegion] {
        &self.regions
    }

    /// Test helper — total guest memory in bytes (sum of all node
    /// regions). Production code derives per-region hugepage counts
    /// directly in `allocate`'s gate, so the layout total is
    /// only needed by tests asserting the advertised RAM.
    #[cfg(test)]
    pub fn total_bytes(&self) -> u64 {
        self.regions.iter().map(|r| r.size).sum()
    }

    /// GPA where guest DRAM starts (first region's start address).
    pub fn dram_base(&self) -> u64 {
        self.regions[0].gpa_start
    }

    /// Highest GPA backed by RAM (one past the last byte). Robust to
    /// region ordering — takes the max over all regions rather than
    /// assuming `regions` is GPA-sorted.
    ///
    /// x86_64-only: its sole caller is `ram_top_exceeds_phys_bits`,
    /// the CPUID-MAXPHYADDR RAM guard, which has no aarch64 caller.
    #[cfg(target_arch = "x86_64")]
    pub fn top_gpa(&self) -> u64 {
        self.regions
            .iter()
            .map(|r| r.gpa_start + r.size)
            .max()
            .unwrap_or(0)
    }

    /// If the relocated RAM top exceeds the guest's addressable physical
    /// space (`1 << phys_bits`), return that top GPA; otherwise `None`.
    /// Bits above the guest MAXPHYADDR are architecturally reserved in guest
    /// PTEs (KVM sets `reserved_gpa_bits = rsvd_bits(cpuid_maxphyaddr, 63)`),
    /// so a PTE mapping a GPA above `1 << phys_bits` faults on the MMU walk's
    /// reserved-bits check — RAM placed there is unreachable; rejecting it
    /// keeps the guest from booting with RAM it cannot reach. (Distinct from
    /// the kernel's separate, wider e820 cap at `max_arch_pfn =
    /// 1 << MAX_PHYSMEM_BITS` (46/52), which does not key on the CPUID
    /// MAXPHYADDR.) `phys_bits >= 64` means no limit (the full u64 GPA space).
    ///
    /// x86_64-only: `phys_bits` is the guest's CPUID 0x8000_0008
    /// MAXPHYADDR and the sole caller is `x86_64::kvm`. There is
    /// currently no aarch64 caller.
    #[cfg(target_arch = "x86_64")]
    pub fn ram_top_exceeds_phys_bits(&self, phys_bits: u32) -> Option<u64> {
        let limit = if phys_bits >= 64 {
            u64::MAX
        } else {
            1u64 << phys_bits
        };
        let top = self.top_gpa();
        (top > limit).then_some(top)
    }

    /// Test helper — GPA immediately after the last node's memory.
    #[cfg(test)]
    pub fn end_gpa(&self) -> u64 {
        let last = self.regions.last().unwrap();
        last.gpa_start + last.size
    }

    /// Test helper — whether this layout has exactly one region.
    #[cfg(test)]
    pub fn is_single_region(&self) -> bool {
        self.regions.len() == 1
    }

    /// Test helper — next available KVM slot index (after all node
    /// regions).
    #[cfg(test)]
    pub fn next_slot(&self) -> u32 {
        self.regions.last().map_or(0, |r| r.slot + 1)
    }

    /// Reserve contiguous VA, install per-node anonymous mappings, and wrap
    /// them in a multi-region `GuestMemoryMmap`.
    ///
    /// Each node gets its own MAP_FIXED mmap within the reserved VA.
    /// The `MmapRegion` wrappers have `owned=false` (via `build_raw`),
    /// so their Drop is a no-op. The `ReservationGuard` munmaps the
    /// entire reservation on drop, releasing all sub-mappings.
    ///
    /// This deliberately does not publish KVM memory slots. Production VM
    /// setup installs the complete final COW layout first and calls
    /// [`Self::register`] only after no further VMA replacement can occur.
    pub fn allocate(&self, use_hugepages: bool, performance_mode: bool) -> Result<AllocatedMemory> {
        let hugetlb_compatible = self.hugetlb_compatible();
        self.choose_memory_backing(use_hugepages, false, 0)?;
        let opportunistic_hugepages = !use_hugepages && performance_mode && hugetlb_compatible;

        // Serialize the free-count decision with the actual MAP_HUGETLB
        // reservation. The lock is held only during memory allocation, never
        // for the VM lifetime: once the VMAs exist the kernel has already
        // deducted those pages and the next process observes the new count.
        let (mapped, backing) = if use_hugepages || opportunistic_hugepages {
            let allocation_lock = acquire_hugepage_allocation_lock()?;
            let desired_backing = self.choose_memory_backing(
                use_hugepages,
                performance_mode,
                super::host_topology::hugepages_free(),
            )?;
            if desired_backing == MemoryBacking::HugeTlb2M {
                match self.map_reservation(true) {
                    Ok(mapped) => {
                        drop(allocation_lock);
                        (mapped, MemoryBacking::HugeTlb2M)
                    }
                    Err(error)
                        if opportunistic_hugepages
                            && error
                                .downcast_ref::<HugepageReservationUnavailable>()
                                .is_some() =>
                    {
                        // A non-ktstr consumer can still spend pages without
                        // taking our advisory lock. The failed attempt owns no
                        // KVM slots and its ReservationGuard has already
                        // unmapped every partial VMA, so retrying with base
                        // pages is exact and retains the same direct-COW loader.
                        tracing::debug!(
                            %error,
                            "opportunistic hugepage reservation lost to an external consumer; \
                             retrying with base-page backing"
                        );
                        drop(allocation_lock);
                        (self.map_reservation(false)?, MemoryBacking::BasePages)
                    }
                    Err(error) => {
                        if use_hugepages
                            && let Some(unavailable) =
                                error.downcast_ref::<HugepageReservationUnavailable>()
                        {
                            let errno = unavailable.source.raw_os_error().unwrap_or(libc::ENOMEM);
                            return Err(super::map_transient_to_contention(
                                kvm_ioctls::Error::new(errno),
                                unavailable.operation.clone(),
                            ));
                        }
                        return Err(error);
                    }
                }
            } else {
                drop(allocation_lock);
                (self.map_reservation(false)?, MemoryBacking::BasePages)
            }
        } else {
            (self.map_reservation(false)?, MemoryBacking::BasePages)
        };

        let guest_regions = self.wrap_mapped_regions(mapped.base, &mapped.va_spans)?;
        let guest_mem = GuestMemoryMmap::from_regions(guest_regions)
            .context("create multi-region GuestMemoryMmap")?;

        Ok(AllocatedMemory {
            guest_mem,
            reservation: mapped.guard,
            backing,
        })
    }

    /// Reserve one contiguous host-VA span and install every per-node mapping.
    /// No KVM slot is registered until all mmaps succeed, so an opportunistic
    /// hugetlb failure can drop this value and retry with base pages without
    /// leaving any kernel-visible partial state.
    fn map_reservation(&self, use_hugepages: bool) -> Result<MappedReservation> {
        const HUGE_2MB: usize = 2 * 1024 * 1024;
        let va_align = if use_hugepages { HUGE_2MB } else { 1 };
        let round_up = |value: usize| {
            value
                .checked_add(va_align - 1)
                .map(|value| value & !(va_align - 1))
                .context("NUMA host-VA span overflow")
        };
        let va_spans: Vec<usize> = self
            .regions
            .iter()
            .map(|region| round_up(region.size as usize))
            .collect::<Result<_>>()?;
        let reserve_size = va_spans.iter().try_fold(va_align - 1, |total, span| {
            total
                .checked_add(*span)
                .context("NUMA reservation overflow")
        })?;
        let reservation = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                reserve_size,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                -1,
                0,
            )
        };
        if reservation == libc::MAP_FAILED {
            let io_error = std::io::Error::last_os_error();
            let errno = io_error.raw_os_error().unwrap_or(0);
            return Err(super::map_transient_to_contention(
                kvm_ioctls::Error::new(errno),
                format!("mmap VA reservation ({reserve_size} bytes) failed"),
            ));
        }
        let guard = ReservationGuard {
            addr: reservation,
            size: reserve_size,
        };
        let base = round_up(reservation as usize)? as *mut libc::c_void;

        // Host VA is PACKED (gap-free): the reservation is sum-of-sizes,
        // so each region's VA offset is the running cumulative size, NOT
        // gpa_start - dram_base. Under MMIO-gap relocation a high
        // region's gpa_start jumps above gap_end; gpa_start - dram_base
        // would index past the packed reservation end (OOB MAP_FIXED).
        // The KVM slot still pairs guest_phys_addr = gpa_start (gapped)
        // with this packed userspace_addr below.
        let mut va_offset = 0usize;
        for (i, region) in self.regions.iter().enumerate() {
            let offset = va_offset;
            let va_span = va_spans[i];
            va_offset = va_offset
                .checked_add(va_span)
                .context("NUMA mapped host-VA offset overflow")?;
            let node_addr = unsafe { (base as *mut u8).add(offset) as *mut libc::c_void };

            let flags = anonymous_node_map_flags(use_hugepages);

            let node_ptr = unsafe {
                libc::mmap(
                    node_addr,
                    va_span,
                    libc::PROT_READ | libc::PROT_WRITE,
                    flags,
                    -1,
                    0,
                )
            };
            if node_ptr == libc::MAP_FAILED {
                let io_error = std::io::Error::last_os_error();
                let errno = io_error.raw_os_error().unwrap_or(0);
                let operation = format!(
                    "MAP_FIXED mmap for node {} ({} bytes) failed",
                    region.node_id, va_span
                );
                if use_hugepages && matches!(errno, libc::ENOMEM | libc::EAGAIN | libc::ENOSPC) {
                    return Err(anyhow::Error::new(HugepageReservationUnavailable {
                        operation,
                        source: io_error,
                    }));
                }
                return Err(super::map_transient_to_contention(
                    kvm_ioctls::Error::new(errno),
                    operation,
                ));
            }

            // Opportunistic THP: hint the anonymous 4 KiB-backed region for
            // 2 MiB transparent hugepages. Self-serve under the host's default
            // `THP=madvise` policy — with `madvise` the kernel only builds
            // huge pages for VMAs that ask, so this advice is what turns guest
            // RAM into 2 MiB backing (fewer EPT/stage-2 walks, fewer TLB
            // misses). Best-effort by design: a kernel built without THP, or
            // with `THP=never`, returns EINVAL; there is no failure mode, so
            // the result is ignored. NOT applied to the perf-mode MAP_HUGETLB
            // branch above — that already reserves explicit hugetlb 2 MiB pages
            // and `MADV_HUGEPAGE` is rejected on hugetlb mappings anyway.
            //
            // Deferred-min interaction: with `MADV_HUGEPAGE` a single 4 KiB
            // fault can materialize a full 2 MiB page, so a sparsely-touched
            // region could inflate RSS. ktstr's boot touches are dense (the
            // guest faults contiguous early-boot memory), so the 2 MiB pages
            // this collapses are near-fully used rather than padding untouched
            // guest RAM into residency.
            //
            // SAFETY: `node_ptr`/`va_span` is the mapping just returned by the
            // `mmap` above (checked != MAP_FAILED); `madvise` only reads/sets
            // VMA advice over that exact range.
            if !use_hugepages {
                unsafe {
                    libc::madvise(node_ptr, va_span, libc::MADV_HUGEPAGE);
                }
            }
        }

        Ok(MappedReservation {
            guard,
            base,
            va_spans,
        })
    }

    /// Wrap a fully mapped reservation as non-owning `vm-memory` regions.
    /// `base` and `va_spans` come from [`Self::map_reservation`] and remain
    /// owned by its guard.
    fn wrap_mapped_regions(
        &self,
        base: *mut libc::c_void,
        va_spans: &[usize],
    ) -> Result<Vec<GuestRegionMmap>> {
        let mut guest_regions = Vec::with_capacity(self.regions.len());
        let mut va_offset = 0usize;
        for (index, region) in self.regions.iter().enumerate() {
            let node_addr = unsafe { (base as *mut u8).add(va_offset) };
            va_offset = va_offset
                .checked_add(va_spans[index])
                .context("NUMA registration host-VA offset overflow")?;
            let node_size = region.size as usize;

            // Step 5: Wrap as vm-memory types. build_raw sets owned=false.
            let mmap_region = unsafe {
                MmapRegion::build_raw(
                    node_addr,
                    node_size,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                )
                .with_context(|| format!("build MmapRegion for node {}", region.node_id))?
            };
            let guest_region = GuestRegionMmap::new(mmap_region, GuestAddress(region.gpa_start))
                .ok_or_else(|| {
                    anyhow::anyhow!("GuestRegionMmap overflow for node {}", region.node_id)
                })?;
            guest_regions.push(guest_region);
        }
        Ok(guest_regions)
    }

    /// Publish the already-final guest-memory VMA layout as KVM memory slots.
    ///
    /// Callers must not replace or unmap any part of `guest_mem` after this
    /// succeeds. Host writes and private COW faults remain valid; only the VMA
    /// identity/layout is frozen at this boundary.
    pub fn register(&self, vm_fd: &kvm_ioctls::VmFd, guest_mem: &GuestMemoryMmap) -> Result<()> {
        self.register_with(guest_mem, |region, mem_region, rollback| {
            let operation = if rollback {
                format!(
                    "roll back KVM memory slot {} for node {}",
                    region.slot, region.node_id
                )
            } else {
                format!(
                    "set KVM memory slot {} for node {}",
                    region.slot, region.node_id
                )
            };
            // Step 7: Register KVM memory slot. KVM_SET_USER_MEMORY_REGION
            // can fail with the host-resource errnos that
            // [`super::map_transient_to_contention`] classifies as
            // [`super::host_topology::ResourceContention`] — most commonly
            // ENOMEM when a peer is holding the host's GuestMemoryMmap
            // budget at the time we register this slot. Routing through
            // the classifier turns those into a SKIP banner instead of a
            // hard test failure (under default policy — `KTSTR_NO_SKIP_MODE`
            // promotes the classified contention to a FAIL at the dispatch
            // boundary in `test_support::dispatch::result_to_exit_code`
            // and the `#[ktstr_test]` macro's terminal arm); non-transient
            // errnos flow through unchanged so a real bug never gets
            // misclassified as contention. The non-transient set per kernel source:
            //   - EINVAL: bad alignment, untagged-addr mismatch, or
            //     access_ok failure on userspace_addr; bad slot ID;
            //     bad gpa_start (virt/kvm/kvm_main.c:2025-2038).
            //   - EEXIST: requested slot overlaps an existing one
            //     (virt/kvm/kvm_main.c:2094, the -EEXIST return under
            //     the kvm_check_memslot_overlap check).
            //   - EIO: kernel WARN-on-impossible-state path
            //     (virt/kvm/kvm_main.c:2057-2058, the -EIO return under
            //     WARN_ON_ONCE(kvm->nr_memslot_pages < old->npages);
            //     arch/x86 mmu.c).
            //   - EFAULT: arm64/riscv guest-phys-bounds violation
            //     (arch/arm64/kvm/mmu.c, arch/riscv/kvm/mmu.c).
            unsafe {
                vm_fd
                    .set_user_memory_region(mem_region)
                    .map_err(|error| super::map_transient_to_contention(error, operation))
            }
        })
    }

    /// Precompute every memory-slot descriptor, then publish them as one
    /// logical transaction. KVM has no multi-slot atomic ioctl, so a failure
    /// rolls back every earlier slot in reverse order before returning.
    ///
    /// The callback seam makes the partial-failure protocol deterministic in
    /// tests without requiring the kernel to fail a particular slot.
    fn register_with<Update>(&self, guest_mem: &GuestMemoryMmap, mut update: Update) -> Result<()>
    where
        Update: FnMut(&NodeRegion, kvm_bindings::kvm_userspace_memory_region, bool) -> Result<()>,
    {
        let registrations = self
            .regions
            .iter()
            .map(|region| {
                let node_addr = guest_mem
                    .get_host_address(GuestAddress(region.gpa_start))
                    .with_context(|| {
                        format!(
                            "resolve host address for KVM memory slot {} (node {})",
                            region.slot, region.node_id
                        )
                    })?;
                Ok((
                    region,
                    kvm_bindings::kvm_userspace_memory_region {
                        slot: region.slot,
                        guest_phys_addr: region.gpa_start,
                        memory_size: region.size,
                        userspace_addr: node_addr as u64,
                        flags: 0,
                    },
                ))
            })
            .collect::<Result<Vec<_>>>()?;

        for (index, (region, mem_region)) in registrations.iter().enumerate() {
            if let Err(register_error) = update(region, *mem_region, false) {
                let mut rollback_errors = Vec::new();
                for (registered_region, registered_mem_region) in
                    registrations[..index].iter().rev()
                {
                    let removal = kvm_bindings::kvm_userspace_memory_region {
                        memory_size: 0,
                        ..*registered_mem_region
                    };
                    if let Err(error) = update(registered_region, removal, true) {
                        rollback_errors.push(format!("{error:#}"));
                    }
                }
                if rollback_errors.is_empty() {
                    return Err(register_error);
                }
                return Err(register_error).context(format!(
                    "KVM memory-slot registration rollback also failed: {}",
                    rollback_errors.join("; ")
                ));
            }
        }
        Ok(())
    }

    /// Bind each node's region to the corresponding host NUMA node(s),
    /// then pre-fault pages.
    ///
    /// `host_nodes` is indexed by guest node_id. Entries beyond the
    /// slice length or empty entries are skipped (e.g. CXL nodes on
    /// non-NUMA hosts).
    ///
    /// Ordering: mbind before MADV_POPULATE_WRITE ensures pages are
    /// allocated on the target node rather than the faulting CPU's node.
    pub fn mbind_regions(&self, guest_mem: &GuestMemoryMmap, host_nodes: &[Vec<usize>]) {
        for region in &self.regions {
            let idx = region.node_id as usize;
            if idx >= host_nodes.len() {
                continue;
            }
            let nodes = &host_nodes[idx];
            if nodes.is_empty() {
                continue;
            }
            let ptr = match guest_mem.get_host_address(GuestAddress(region.gpa_start)) {
                Ok(addr) => addr,
                Err(_) => continue,
            };

            // Step 3: Per-node mbind (before any page faults).
            // SAFETY: `ptr` was obtained from `guest_mem.get_host_address` for
            // an mmap'd region of `region.size` bytes (validated by
            // GuestMemoryMmap), so the (ptr, len) pair is a valid mapped range.
            unsafe {
                super::host_topology::mbind_to_nodes(ptr, region.size as usize, nodes);
            }

            // Step 4: Pre-fault after mbind.
            let ret = unsafe {
                libc::madvise(
                    ptr as *mut libc::c_void,
                    region.size as usize,
                    libc::MADV_POPULATE_WRITE,
                )
            };
            if ret != 0 {
                eprintln!(
                    "performance_mode: WARNING: MADV_POPULATE_WRITE for node {} failed: {}",
                    region.node_id,
                    std::io::Error::last_os_error()
                );
            }
        }
    }

    /// Reapply NUMA policy to a file-backed range installed into a
    /// guest-memory destination hole.
    ///
    /// Unlike [`Self::mbind_regions`], this deliberately does not populate
    /// or write-fault any page. The immutable prepared-initrd mapping must
    /// stay shared with the host page cache until the guest actually writes
    /// a page; `MADV_POPULATE_WRITE` here would eagerly COW the entire image.
    pub fn mbind_replaced_range(
        &self,
        guest_addr: u64,
        host_addr: *mut u8,
        len: usize,
        host_nodes: &[Vec<usize>],
    ) -> Result<()> {
        anyhow::ensure!(len > 0, "replaced NUMA range is empty");
        let region = self
            .regions
            .iter()
            .find(|region| {
                guest_addr >= region.gpa_start
                    && guest_addr < region.gpa_start.saturating_add(region.size)
            })
            .with_context(|| format!("no NUMA region contains replaced GPA {guest_addr:#x}"))?;
        let guest_end = guest_addr
            .checked_add(len as u64)
            .context("replaced NUMA range end overflows")?;
        anyhow::ensure!(
            guest_end <= region.gpa_start.saturating_add(region.size),
            "replaced NUMA range {guest_addr:#x}..{guest_end:#x} crosses node {} boundary",
            region.node_id
        );
        let nodes = host_nodes.get(region.node_id as usize).with_context(|| {
            format!(
                "missing host NUMA mapping for guest node {}",
                region.node_id
            )
        })?;
        if nodes.is_empty() || len == 0 {
            return Ok(());
        }
        // SAFETY: load_prepared_initrd calls this immediately after installing
        // exactly `(host_addr, len)` into its prevalidated destination hole.
        // The subrange lies wholly within this NodeRegion.
        unsafe {
            super::host_topology::mbind_to_nodes(host_addr, len, nodes);
        }
        Ok(())
    }

    /// Test helper — find the node region containing a GPA.
    /// Regions are sorted by `gpa_start`, so this uses binary search.
    #[cfg(test)]
    pub fn region_for_gpa(&self, gpa: u64) -> Option<&NodeRegion> {
        let idx = self
            .regions
            .partition_point(|r| r.gpa_start <= gpa)
            .checked_sub(1)?;
        let r = &self.regions[idx];
        if gpa < r.gpa_start + r.size {
            Some(r)
        } else {
            None
        }
    }

    /// Test helper — node region by node_id.
    #[cfg(test)]
    pub fn region_for_node(&self, node_id: u32) -> Option<&NodeRegion> {
        self.regions.iter().find(|r| r.node_id == node_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vmm::topology::{NumaNode, Topology};

    #[test]
    fn anonymous_node_mapping_retains_noreserve_for_base_pages() {
        let base = anonymous_node_map_flags(false);
        assert_ne!(
            base & libc::MAP_NORESERVE,
            0,
            "base-page guest RAM must not reserve commit for untouched pages"
        );
        assert_eq!(base & libc::MAP_HUGETLB, 0);
        assert_ne!(base & libc::MAP_FIXED, 0);
        assert_ne!(base & libc::MAP_ANONYMOUS, 0);
        assert_ne!(base & libc::MAP_PRIVATE, 0);

        let huge = anonymous_node_map_flags(true);
        assert_eq!(
            huge & libc::MAP_NORESERVE,
            0,
            "explicit hugetlb RAM is accounted by the hugepage pool"
        );
        assert_ne!(huge & libc::MAP_HUGETLB, 0);
        assert_ne!(huge & libc::MAP_HUGE_2MB, 0);
    }

    #[test]
    fn uniform_single_region() {
        let topo = Topology::new(1, 2, 4, 2);
        let layout = NumaMemoryLayout::compute(&topo, 256, 0, None).unwrap();
        assert!(layout.is_single_region());
        assert_eq!(layout.total_bytes(), 256 << 20);
        assert_eq!(layout.regions().len(), 1);
        assert_eq!(layout.regions()[0].node_id, 0);
        assert_eq!(layout.regions()[0].gpa_start, 0);
        assert_eq!(layout.regions()[0].size, 256 << 20);
        assert_eq!(layout.regions()[0].slot, 0);
        assert_eq!(layout.next_slot(), 1);
    }

    #[test]
    fn uniform_multi_numa_splits_evenly() {
        let topo = Topology::new(2, 4, 2, 1);
        let layout = NumaMemoryLayout::compute(&topo, 512, 0, None).unwrap();
        assert_eq!(layout.regions().len(), 2);
        assert_eq!(layout.regions()[0].node_id, 0);
        assert_eq!(layout.regions()[0].size, 256 << 20);
        assert_eq!(layout.regions()[0].slot, 0);
        assert_eq!(layout.regions()[1].node_id, 1);
        assert_eq!(layout.regions()[1].gpa_start, 256 << 20);
        assert_eq!(layout.regions()[1].size, 256 << 20);
        assert_eq!(layout.regions()[1].slot, 1);
    }

    #[test]
    fn uniform_multi_numa_remainder() {
        // 100 MiB / 3 nodes: the even split is snapped to whole 2 MiB
        // hugepages (34/34/32, not 33/33/34) so a perf-mode hugepage
        // backing gets full 2 MiB EPT per node; the sum is preserved.
        let topo = Topology::new(3, 3, 2, 1);
        let layout = NumaMemoryLayout::compute(&topo, 100, 0, None).unwrap();
        assert_eq!(layout.regions().len(), 3);
        let sizes: Vec<u64> = layout.regions().iter().map(|r| r.size).collect();
        assert_eq!(sizes, vec![34 << 20, 34 << 20, 32 << 20]);
        assert_eq!(layout.total_bytes(), 100 << 20);
        // Every node base GPA + size is 2 MiB-aligned (the snap's whole
        // purpose: GPA/HVA 2 MiB-congruence so KVM keeps 2 MiB EPT).
        const TWO_MIB: u64 = 2 << 20;
        for r in layout.regions() {
            assert_eq!(
                r.gpa_start % TWO_MIB,
                0,
                "node {} GPA not 2 MiB-aligned",
                r.node_id
            );
            assert_eq!(
                r.size % TWO_MIB,
                0,
                "node {} size not a 2 MiB multiple",
                r.node_id
            );
        }
    }

    #[test]
    fn hugepage_even_split_preserves_sum_and_snaps_to_2mib() {
        // For each (total, nodes): node count preserved, sum exact, and
        // every node a 2 MiB multiple (the last node may carry a sub-2
        // MiB remainder for odd totals — checked separately below).
        for (total, nodes) in [(100u32, 3u32), (512, 2), (96, 4), (200, 3), (64, 1)] {
            let split = hugepage_even_split_mib(total, nodes);
            assert_eq!(split.len(), nodes as usize, "{total}/{nodes}: node count");
            assert_eq!(
                split.iter().sum::<u32>(),
                total,
                "{total}/{nodes}: sum changed"
            );
            for (i, &mib) in split.iter().enumerate() {
                assert!(mib > 0, "{total}/{nodes}: node {i} zero-sized");
                if i != nodes as usize - 1 {
                    assert_eq!(
                        mib % 2,
                        0,
                        "{total}/{nodes}: node {i} ({mib}) not a 2 MiB multiple"
                    );
                }
            }
            // Even totals snap with no remainder -> last node also even.
            if total % 2 == 0 {
                assert_eq!(
                    split[nodes as usize - 1] % 2,
                    0,
                    "{total}/{nodes}: even total, last node odd"
                );
            }
        }
    }

    #[test]
    fn hugepage_even_split_odd_total_tail_on_last_node() {
        // Odd total: the 1 MiB sub-hugepage remainder lands on the last
        // node only; earlier nodes stay whole 2 MiB multiples.
        let split = hugepage_even_split_mib(101, 3);
        assert_eq!(split.iter().sum::<u32>(), 101);
        assert_eq!(split[0] % 2, 0);
        assert_eq!(split[1] % 2, 0);
        assert_eq!(split[2] % 2, 1, "odd remainder must land on the last node");
    }

    #[test]
    fn hugepage_even_split_tiny_falls_back_to_plain_even() {
        // total_mib < 2 * nodes: a node can't get even one hugepage, so
        // the plain even split is returned rather than snapping nodes to
        // zero. 4 / 3 = 1,1,2.
        let split = hugepage_even_split_mib(4, 3);
        assert_eq!(split.iter().sum::<u32>(), 4);
        assert_eq!(split, vec![1, 1, 2]);
    }

    static TWO_NODES: [NumaNode; 2] = [NumaNode::new(2, 256), NumaNode::new(2, 256)];

    #[test]
    fn with_nodes_two_regions() {
        let topo = Topology::with_nodes(4, 2, &TWO_NODES);
        let layout = NumaMemoryLayout::compute(&topo, 512, 0, None).unwrap();
        assert!(!layout.is_single_region());
        assert_eq!(layout.regions().len(), 2);

        let r0 = &layout.regions()[0];
        assert_eq!(r0.node_id, 0);
        assert_eq!(r0.gpa_start, 0);
        assert_eq!(r0.size, 256 << 20);
        assert_eq!(r0.slot, 0);

        let r1 = &layout.regions()[1];
        assert_eq!(r1.node_id, 1);
        assert_eq!(r1.gpa_start, 256 << 20);
        assert_eq!(r1.size, 256 << 20);
        assert_eq!(r1.slot, 1);

        assert_eq!(layout.total_bytes(), 512 << 20);
        assert_eq!(layout.end_gpa(), 512 << 20);
        assert_eq!(layout.next_slot(), 2);
    }

    static ASYM_NODES: [NumaNode; 2] = [NumaNode::new(1, 128), NumaNode::new(3, 384)];

    #[test]
    fn asymmetric_node_memory() {
        let topo = Topology::with_nodes(2, 1, &ASYM_NODES);
        let layout = NumaMemoryLayout::compute(&topo, 512, 0, None).unwrap();
        assert_eq!(layout.regions().len(), 2);
        assert_eq!(layout.regions()[0].size, 128 << 20);
        assert_eq!(layout.regions()[1].size, 384 << 20);
        assert_eq!(layout.regions()[1].gpa_start, 128 << 20);
    }

    static CXL_NODES: [NumaNode; 3] = [
        NumaNode::new(2, 256),
        NumaNode::new(2, 256),
        NumaNode::new(0, 128),
    ];

    #[test]
    fn cxl_memory_only_node() {
        let topo = Topology::with_nodes(4, 1, &CXL_NODES);
        let layout = NumaMemoryLayout::compute(&topo, 640, 0, None).unwrap();
        assert_eq!(layout.regions().len(), 3);

        assert_eq!(layout.regions()[0].node_id, 0);
        assert_eq!(layout.regions()[1].node_id, 1);
        assert_eq!(layout.regions()[2].node_id, 2);
        assert_eq!(layout.regions()[2].size, 128 << 20);
    }

    static CXL_ZERO_MEM: [NumaNode; 3] = [
        NumaNode::new(2, 256),
        NumaNode::new(0, 0),
        NumaNode::new(2, 256),
    ];

    #[test]
    fn cxl_zero_memory_node_skipped() {
        let topo = Topology::with_nodes(4, 1, &CXL_ZERO_MEM);
        let layout = NumaMemoryLayout::compute(&topo, 512, 0, None).unwrap();
        assert_eq!(layout.regions().len(), 2);
        assert_eq!(layout.regions()[0].node_id, 0);
        assert_eq!(layout.regions()[1].node_id, 2);
    }

    #[test]
    fn aarch64_dram_base() {
        let topo = Topology::with_nodes(4, 2, &TWO_NODES);
        let dram_base = 0x4000_0000u64;
        let layout = NumaMemoryLayout::compute(&topo, 512, dram_base, None).unwrap();
        assert_eq!(layout.dram_base(), dram_base);
        assert_eq!(layout.regions()[0].gpa_start, dram_base);
        assert_eq!(layout.regions()[1].gpa_start, dram_base + (256 << 20));
        assert_eq!(layout.end_gpa(), dram_base + (512 << 20));
    }

    #[test]
    fn memory_mismatch_error() {
        let topo = Topology::with_nodes(4, 2, &TWO_NODES);
        let err = NumaMemoryLayout::compute(&topo, 1024, 0, None).unwrap_err();
        assert!(format!("{err}").contains("must equal"), "got: {err}");
    }

    #[test]
    fn region_for_gpa_lookup() {
        let topo = Topology::with_nodes(4, 2, &TWO_NODES);
        let layout = NumaMemoryLayout::compute(&topo, 512, 0, None).unwrap();

        let r = layout.region_for_gpa(0).unwrap();
        assert_eq!(r.node_id, 0);

        let r = layout.region_for_gpa((256 << 20) - 1).unwrap();
        assert_eq!(r.node_id, 0);

        let r = layout.region_for_gpa(256 << 20).unwrap();
        assert_eq!(r.node_id, 1);

        assert!(layout.region_for_gpa(512 << 20).is_none());
    }

    #[test]
    fn region_for_gpa_with_dram_base() {
        let dram_base = 0x4000_0000u64;
        let topo = Topology::with_nodes(4, 2, &TWO_NODES);
        let layout = NumaMemoryLayout::compute(&topo, 512, dram_base, None).unwrap();

        assert!(layout.region_for_gpa(0).is_none());
        assert_eq!(layout.region_for_gpa(dram_base).unwrap().node_id, 0);
        assert_eq!(
            layout
                .region_for_gpa(dram_base + (256 << 20))
                .unwrap()
                .node_id,
            1
        );
    }

    #[test]
    fn region_for_node_lookup() {
        let topo = Topology::with_nodes(4, 2, &TWO_NODES);
        let layout = NumaMemoryLayout::compute(&topo, 512, 0, None).unwrap();

        assert_eq!(layout.region_for_node(0).unwrap().gpa_start, 0);
        assert_eq!(layout.region_for_node(1).unwrap().gpa_start, 256 << 20);
        assert!(layout.region_for_node(5).is_none());
    }

    #[test]
    fn slot_assignment_contiguous() {
        let topo = Topology::with_nodes(4, 1, &CXL_NODES);
        let layout = NumaMemoryLayout::compute(&topo, 640, 0, None).unwrap();
        for (i, r) in layout.regions().iter().enumerate() {
            assert_eq!(r.slot, i as u32);
        }
    }

    #[test]
    fn single_node_with_nodes() {
        static ONE: [NumaNode; 1] = [NumaNode::new(4, 512)];
        let topo = Topology::with_nodes(2, 1, &ONE);
        let layout = NumaMemoryLayout::compute(&topo, 512, 0, None).unwrap();
        assert!(layout.is_single_region());
        assert_eq!(layout.regions()[0].size, 512 << 20);
    }

    #[test]
    fn allocate_register_single_region() {
        let topo = Topology::new(1, 1, 1, 1);
        let layout = NumaMemoryLayout::compute(&topo, 64, 0, None).unwrap();

        let kvm = kvm_ioctls::Kvm::new().unwrap();
        let vm_fd = kvm.create_vm().unwrap();

        let alloc = layout.allocate(false, false).unwrap();
        layout.register(&vm_fd, &alloc.guest_mem).unwrap();

        use vm_memory::GuestMemoryRegion;
        let total: u64 = alloc.guest_mem.iter().map(|r| r.len()).sum();
        assert_eq!(total, 64 << 20);
        assert_eq!(alloc.guest_mem.iter().count(), 1);
    }

    #[test]
    fn allocate_register_multi_node_per_region() {
        let topo = Topology::with_nodes(4, 2, &TWO_NODES);
        let layout = NumaMemoryLayout::compute(&topo, 512, 0, None).unwrap();

        let kvm = kvm_ioctls::Kvm::new().unwrap();
        let vm_fd = kvm.create_vm().unwrap();

        let alloc = layout.allocate(false, false).unwrap();
        layout.register(&vm_fd, &alloc.guest_mem).unwrap();

        use vm_memory::GuestMemoryRegion;
        let total: u64 = alloc.guest_mem.iter().map(|r| r.len()).sum();
        assert_eq!(total, 512 << 20);
        // Per-node MAP_FIXED: one GuestMemoryMmap region per node.
        assert_eq!(alloc.guest_mem.iter().count(), 2);
    }

    #[test]
    fn register_transaction_rolls_back_partial_slots_in_reverse_order() {
        let topo = Topology::new(3, 3, 1, 1);
        let layout = NumaMemoryLayout::compute(&topo, 96, 0, None).unwrap();
        let ranges = layout
            .regions()
            .iter()
            .map(|region| {
                (
                    GuestAddress(region.gpa_start),
                    usize::try_from(region.size).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        let guest_mem = GuestMemoryMmap::<()>::from_ranges(&ranges).unwrap();

        let mut operations = Vec::new();
        let error = layout
            .register_with(&guest_mem, |region, mem_region, rollback| {
                operations.push((rollback, region.slot, mem_region.memory_size));
                anyhow::ensure!(
                    rollback || region.slot != 2,
                    "injected third-slot registration failure"
                );
                Ok(())
            })
            .unwrap_err();
        assert!(format!("{error:#}").contains("injected third-slot registration failure"));
        assert_eq!(
            operations,
            [
                (false, 0, 32 << 20),
                (false, 1, 32 << 20),
                (false, 2, 32 << 20),
                (true, 1, 0),
                (true, 0, 0),
            ],
            "a partial publish must delete every live slot in reverse order"
        );
    }

    #[test]
    fn allocate_hugepages_snapped_split_aligns_and_succeeds() {
        // Coverage gap: use_hugepages=true had ZERO coverage.
        // The snapped odd split (100 MiB / 3 -> 34/34/32) mmapped with
        // MAP_HUGETLB | MAP_FIXED must NOT EINVAL (the bug), and every
        // node's host VA + GPA must be 2 MiB-aligned (GPA/HVA congruence
        // keeps full 2 MiB EPT). Host-gated on free 2 MiB hugepages; the
        // CI-runnable guarantee is uniform_multi_numa_remainder's
        // GPA-alignment assertion, so a skip here cannot mask the bug.
        let topo = Topology::new(3, 3, 2, 1);
        let layout = NumaMemoryLayout::compute(&topo, 100, 0, None).unwrap();
        if crate::vmm::host_topology::hugepages_free()
            < crate::vmm::host_topology::hugepages_needed(100)
        {
            eprintln!(
                "SKIP allocate_hugepages_snapped_split: host has < 100 MiB of free 2 MiB hugepages"
            );
            return;
        }
        let kvm = kvm_ioctls::Kvm::new().unwrap();
        let vm_fd = kvm.create_vm().unwrap();
        // The bug manifested as a hard EINVAL from the MAP_HUGETLB |
        // MAP_FIXED mmap; .expect() pins the fix.
        let alloc = layout
            .allocate(true, false)
            .expect("hugepage MAP_FIXED allocate must not EINVAL on a snapped odd split");
        layout.register(&vm_fd, &alloc.guest_mem).unwrap();
        const TWO_MIB: usize = 2 << 20;
        for r in layout.regions() {
            let hva = alloc
                .guest_mem
                .get_host_address(GuestAddress(r.gpa_start))
                .unwrap();
            assert_eq!(
                hva as usize % TWO_MIB,
                0,
                "node {} host VA not 2 MiB-aligned",
                r.node_id
            );
            assert_eq!(
                r.gpa_start as usize % TWO_MIB,
                0,
                "node {} GPA not 2 MiB-aligned",
                r.node_id
            );
        }
    }

    static ODD_NODES: [NumaNode; 2] = [NumaNode::new(2, 33), NumaNode::new(2, 33)];

    const HUGEPAGE_LOCK_CHILD_TEST: &str =
        "vmm::numa_mem::tests::hugepage_allocation_lock_cross_process_child";
    const HUGEPAGE_LOCK_CHILD_ROOT: &str = "KTSTR_HUGEPAGE_LOCK_CHILD_ROOT";
    const HUGEPAGE_LOCK_CHILD_INDEX: &str = "KTSTR_HUGEPAGE_LOCK_CHILD_INDEX";

    #[test]
    fn hugepage_allocation_lock_cross_process_child() {
        let Some(root) = std::env::var_os(HUGEPAGE_LOCK_CHILD_ROOT).map(PathBuf::from) else {
            return;
        };
        let index = std::env::var(HUGEPAGE_LOCK_CHILD_INDEX).unwrap();
        std::fs::write(root.join("ready").join(&index), b"ready").unwrap();

        let _transaction =
            acquire_hugepage_allocation_lock_at(&root.join("hugepage-allocation.lock")).unwrap();
        let active = root.join("active-transaction");
        let _active = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&active)
            .expect("two processes entered the hugepage allocation transaction together");
        // Keep the critical section open long enough that a missing
        // cross-process flock cannot pass merely because the children happened
        // to be scheduled one at a time.
        std::thread::sleep(std::time::Duration::from_millis(5));

        let budget_path = root.join("free-pages");
        let available: u64 = std::fs::read_to_string(&budget_path)
            .unwrap()
            .parse()
            .unwrap();
        if available >= 64 {
            std::fs::write(&budget_path, b"0").unwrap();
            std::fs::write(root.join("winner"), index).unwrap();
        }
        std::fs::remove_file(active).unwrap();
    }

    #[test]
    fn opportunistic_hugepage_budget_is_spent_once_under_52_process_contention() {
        const CONTENDERS: usize = 52;
        const RESERVATION_PAGES: u64 = 64;
        let temp = tempfile::tempdir().unwrap();
        let lock_path = temp.path().join("hugepage-allocation.lock");
        std::fs::create_dir(temp.path().join("ready")).unwrap();
        std::fs::write(
            temp.path().join("free-pages"),
            RESERVATION_PAGES.to_string(),
        )
        .unwrap();
        // Queue every child behind an actual process-shared lock before
        // releasing the herd, so the test exercises one observed budget with
        // the same check -> mmap transaction ordering as production.
        let parent_transaction = acquire_hugepage_allocation_lock_at(&lock_path).unwrap();

        struct Children(Vec<std::process::Child>);
        impl Drop for Children {
            fn drop(&mut self) {
                for child in &mut self.0 {
                    let _ = child.kill();
                    let _ = child.wait();
                }
            }
        }

        let mut children = Children(Vec::with_capacity(CONTENDERS));
        for index in 0..CONTENDERS {
            children.0.push(
                std::process::Command::new(std::env::current_exe().unwrap())
                    .arg("--exact")
                    .arg(HUGEPAGE_LOCK_CHILD_TEST)
                    .arg("--nocapture")
                    .arg("--test-threads=1")
                    .env(HUGEPAGE_LOCK_CHILD_ROOT, temp.path())
                    .env(HUGEPAGE_LOCK_CHILD_INDEX, index.to_string())
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .spawn()
                    .unwrap(),
            );
        }

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        loop {
            let ready = std::fs::read_dir(temp.path().join("ready"))
                .unwrap()
                .count();
            if ready == CONTENDERS {
                break;
            }
            for child in &mut children.0 {
                if let Some(status) = child.try_wait().unwrap() {
                    panic!("hugepage lock child exited before release with {status}");
                }
            }
            assert!(
                std::time::Instant::now() < deadline,
                "only {ready}/{CONTENDERS} hugepage lock children reached the barrier"
            );
            std::thread::yield_now();
        }
        drop(parent_transaction);

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        for child in &mut children.0 {
            loop {
                if let Some(status) = child.try_wait().unwrap() {
                    assert!(status.success(), "hugepage lock child failed with {status}");
                    break;
                }
                assert!(
                    std::time::Instant::now() < deadline,
                    "hugepage lock process storm did not drain"
                );
                std::thread::yield_now();
            }
        }
        children.0.clear();

        assert_eq!(
            std::fs::read_to_string(temp.path().join("free-pages")).unwrap(),
            "0"
        );
        assert!(
            temp.path().join("winner").exists(),
            "exactly one process must spend the shared hugepage budget"
        );
    }

    #[test]
    fn custom_odd_internal_boundary_selects_base_page_backing() {
        let topo = Topology::with_nodes(4, 2, &ODD_NODES);
        let layout = NumaMemoryLayout::compute(&topo, 66, 0, None).unwrap();
        assert_eq!(layout.regions()[0].size, 33 << 20);
        assert_eq!(layout.regions()[1].size, 33 << 20);
        assert_eq!(layout.regions()[1].gpa_start, 33 << 20);
        assert_eq!(
            layout.choose_memory_backing(false, true, u64::MAX).unwrap(),
            MemoryBacking::BasePages
        );
        assert!(
            layout.choose_memory_backing(true, true, u64::MAX).is_err(),
            "an explicit hugepage request must not silently fall back"
        );
    }

    #[test]
    fn odd_final_tail_remains_hugetlb_compatible() {
        let topo = Topology::new(3, 3, 2, 1);
        let layout = NumaMemoryLayout::compute(&topo, 101, 0, None).unwrap();
        assert_eq!(
            layout
                .regions()
                .iter()
                .map(|region| region.size >> 20)
                .collect::<Vec<_>>(),
            vec![34, 34, 33]
        );
        assert_eq!(
            layout.choose_memory_backing(true, false, 0).unwrap(),
            MemoryBacking::HugeTlb2M
        );
    }

    #[test]
    fn contiguous_host_va() {
        let topo = Topology::with_nodes(4, 2, &TWO_NODES);
        let layout = NumaMemoryLayout::compute(&topo, 512, 0, None).unwrap();
        let alloc = layout.allocate(false, false).unwrap();

        let base = alloc.guest_mem.get_host_address(GuestAddress(0)).unwrap();
        let mid = alloc
            .guest_mem
            .get_host_address(GuestAddress(256 << 20))
            .unwrap();
        let offset = unsafe { mid.offset_from(base) };
        assert_eq!(offset, (256isize << 20));
    }

    #[test]
    fn cross_region_write_read() {
        let topo = Topology::with_nodes(4, 2, &TWO_NODES);
        let layout = NumaMemoryLayout::compute(&topo, 512, 0, None).unwrap();
        let alloc = layout.allocate(false, false).unwrap();

        use vm_memory::Bytes;

        let boundary = (256u64 << 20) - 4;
        let data: [u8; 8] = [0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE];
        alloc
            .guest_mem
            .write_slice(&data, GuestAddress(boundary))
            .unwrap();

        let mut readback = [0u8; 8];
        alloc
            .guest_mem
            .read_slice(&mut readback, GuestAddress(boundary))
            .unwrap();
        assert_eq!(data, readback);
    }

    #[test]
    fn uniform_multi_numa_allocate() {
        let topo = Topology::new(2, 2, 2, 1);
        let layout = NumaMemoryLayout::compute(&topo, 128, 0, None).unwrap();
        assert_eq!(layout.regions().len(), 2);
        let alloc = layout.allocate(false, false).unwrap();

        use vm_memory::GuestMemoryRegion;
        let total: u64 = alloc.guest_mem.iter().map(|r| r.len()).sum();
        assert_eq!(total, 128 << 20);
        // Uniform multi-NUMA: one region per node.
        assert_eq!(alloc.guest_mem.iter().count(), 2);
    }

    #[test]
    fn reservation_guard_munmaps_on_drop() {
        let topo = Topology::new(1, 1, 1, 1);
        let layout = NumaMemoryLayout::compute(&topo, 64, 0, None).unwrap();
        let alloc = layout.allocate(false, false).unwrap();

        let addr = alloc.reservation.addr;
        let size = alloc.reservation.size;
        assert!(!addr.is_null());
        assert_eq!(size, 64 << 20);
        // Drop releases the VA reservation.
        drop(alloc);
        // Prove the unmap actually fired: no /proc/self/maps entry may
        // still cover the range. If impl Drop for ReservationGuard were
        // deleted (a 64 MiB VA leak per VM) or munmap'd the wrong base/
        // size, the PROT_NONE reservation would still appear here — the
        // pre-drop assertions only read fields, never the unmap effect.
        let addr_val = addr as usize;
        let maps = std::fs::read_to_string("/proc/self/maps").unwrap();
        for line in maps.lines() {
            let range = line.split_whitespace().next().unwrap_or("");
            if let Some((start, end)) = range.split_once('-')
                && let (Ok(start), Ok(end)) = (
                    usize::from_str_radix(start, 16),
                    usize::from_str_radix(end, 16),
                )
            {
                assert!(
                    !(start <= addr_val && addr_val < end),
                    "VA reservation at {addr_val:#x} still mapped after drop \
                     (ReservationGuard unmap did not fire): {line}",
                );
            }
        }
    }

    #[test]
    fn three_node_allocation() {
        let topo = Topology::with_nodes(4, 1, &CXL_NODES);
        let layout = NumaMemoryLayout::compute(&topo, 640, 0, None).unwrap();
        assert_eq!(layout.regions().len(), 3);
        let alloc = layout.allocate(false, false).unwrap();

        use vm_memory::GuestMemoryRegion;
        assert_eq!(alloc.guest_mem.iter().count(), 3);
        let total: u64 = alloc.guest_mem.iter().map(|r| r.len()).sum();
        assert_eq!(total, 640 << 20);
    }

    // --- MMIO-gap relocate ---

    const X86_GAP: Option<(u64, u64)> = Some((0xC000_0000, 0x1_0000_0000));

    #[test]
    fn relocate_single_node_crossing_gap() {
        // 4 GiB single node on x86: RAM crosses MMIO_GAP_START (3 GiB).
        let topo = Topology::new(1, 2, 4, 2);
        let layout = NumaMemoryLayout::compute(&topo, 4096, 0, X86_GAP).unwrap();
        // Split into two regions, SAME node_id, total preserved (no GiB lost).
        assert_eq!(layout.regions().len(), 2);
        assert_eq!(layout.total_bytes(), 4096 << 20);
        let r0 = &layout.regions()[0];
        let r1 = &layout.regions()[1];
        assert_eq!(r0.node_id, 0);
        assert_eq!(r0.gpa_start, 0);
        assert_eq!(r0.size, 0xC000_0000); // [0, 3 GiB)
        assert_eq!(r1.node_id, 0); // relocated half keeps the node id
        assert_eq!(r1.gpa_start, 0x1_0000_0000); // begins at 4 GiB
        assert_eq!(r1.size, (4096u64 << 20) - 0xC000_0000); // overflow above the gap
        assert_eq!(r0.slot, 0);
        assert_eq!(r1.slot, 1); // dense slots
        // Regression pin: no region overlaps the MMIO gap; the device
        // window (virtio 0xC000_0000, IOAPIC 0xFEC0_0000) is not RAM.
        for r in layout.regions() {
            assert!(
                r.gpa_start >= 0x1_0000_0000 || r.gpa_start + r.size <= 0xC000_0000,
                "region {r:?} overlaps the MMIO gap"
            );
        }
        assert!(layout.region_for_gpa(0xC000_0000).is_none());
        assert!(layout.region_for_gpa(0xFEC0_0000).is_none());
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn ram_top_exceeds_phys_bits_rejects_above_maxphyaddr() {
        // 8 GiB single node on x86 relocates above the 4 GiB MMIO gap -> top
        // GPA ~9 GiB. A 33-bit guest MAXPHYADDR (8 GiB) is exceeded -> must
        // reject (else the RAM above the MAXPHYADDR is unreachable — a guest
        // access faults on the MMU reserved-bits check); a 40-bit one
        // (1 TiB) is not; >=64 means no limit.
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
            llc_cores: None,
        };
        let layout = NumaMemoryLayout::compute(&topo, 8192, 0, X86_GAP).unwrap();
        let top = layout.top_gpa();
        assert!(
            top > (1u64 << 33),
            "8 GiB relocated above the gap should exceed 1<<33: {top:#x}"
        );
        assert_eq!(layout.ram_top_exceeds_phys_bits(33), Some(top));
        assert_eq!(layout.ram_top_exceeds_phys_bits(40), None);
        assert_eq!(layout.ram_top_exceeds_phys_bits(64), None);
    }

    #[test]
    fn relocate_below_gap_no_split() {
        // 2 GiB single node stays below the gap: one region, no relocate.
        let topo = Topology::new(1, 2, 4, 2);
        let layout = NumaMemoryLayout::compute(&topo, 2048, 0, X86_GAP).unwrap();
        assert_eq!(layout.regions().len(), 1);
        assert_eq!(layout.regions()[0].gpa_start, 0);
        assert_eq!(layout.regions()[0].size, 2048 << 20);
    }

    static STRADDLE_NODES: [NumaNode; 2] = [NumaNode::new(2, 2048), NumaNode::new(2, 4096)];

    #[test]
    fn relocate_multi_node_straddle() {
        // node0 = 2 GiB (wholly below gap); node1 = 4 GiB (straddles the
        // 3 GiB boundary) → split, both halves keep node_id 1.
        let topo = Topology::with_nodes(4, 1, &STRADDLE_NODES);
        let layout = NumaMemoryLayout::compute(&topo, 6144, 0, X86_GAP).unwrap();
        assert_eq!(layout.regions().len(), 3);
        assert_eq!(layout.total_bytes(), 6144 << 20);
        // node 0 — below the gap, no split.
        assert_eq!(layout.regions()[0].node_id, 0);
        assert_eq!(layout.regions()[0].gpa_start, 0);
        assert_eq!(layout.regions()[0].size, 2048 << 20);
        // node 1 low — [2 GiB, 3 GiB).
        assert_eq!(layout.regions()[1].node_id, 1);
        assert_eq!(layout.regions()[1].gpa_start, 2048 << 20);
        assert_eq!(layout.regions()[1].size, 0xC000_0000 - (2048 << 20));
        // node 1 high — relocated to 4 GiB, SAME node_id.
        assert_eq!(layout.regions()[2].node_id, 1);
        assert_eq!(layout.regions()[2].gpa_start, 0x1_0000_0000);
        // node 1 keeps its full 4 GiB across the split.
        assert_eq!(
            layout.regions()[1].size + layout.regions()[2].size,
            4096 << 20
        );
        // Dense slots, gap unbacked.
        assert_eq!(layout.regions()[2].slot, 2);
        for r in layout.regions() {
            assert!(r.gpa_start >= 0x1_0000_0000 || r.gpa_start + r.size <= 0xC000_0000);
        }
    }

    #[test]
    fn relocate_none_is_noop_even_when_range_crosses_gap() {
        // aarch64 passes None: a 16 GiB VM at DRAM_START spans
        // [1 GiB, 17 GiB), crossing [3 GiB, 4 GiB) — but that is real RAM
        // on aarch64 (MMIO is below DRAM_START), so it must NOT be carved.
        let topo = Topology::new(1, 2, 4, 2);
        let dram_base = 0x4000_0000u64;
        let layout = NumaMemoryLayout::compute(&topo, 16384, dram_base, None).unwrap();
        assert_eq!(layout.regions().len(), 1);
        assert_eq!(layout.regions()[0].gpa_start, dram_base);
        assert_eq!(layout.regions()[0].size, 16384 << 20);
    }

    #[test]
    fn replaced_ranges_require_and_accept_node_aligned_splits() {
        // A uniform two-node topology needs at least one LLC per node.
        // Keep the fixture at two total CPUs while making that ownership
        // explicit: two LLCs, one single-threaded core each.
        let topo = Topology::new(2, 2, 1, 1);
        let layout = NumaMemoryLayout::compute(&topo, 4, 0, None).unwrap();
        assert_eq!(layout.regions().len(), 2);
        assert_eq!(
            layout
                .regions()
                .iter()
                .map(|region| (region.node_id, region.gpa_start, region.size))
                .collect::<Vec<_>>(),
            vec![(0, 0, 2 << 20), (1, 2 << 20, 2 << 20)],
        );
        let boundary = layout.regions()[1].gpa_start;
        let dummy = std::ptr::dangling_mut::<u8>();
        let empty_policies = vec![Vec::new(), Vec::new()];

        assert!(
            layout
                .mbind_replaced_range(boundary - (1 << 20), dummy, 2 << 20, &empty_policies,)
                .is_err(),
            "one mmap policy operation must not silently cross NodeRegions"
        );
        layout
            .mbind_replaced_range(boundary - (1 << 20), dummy, 1 << 20, &empty_policies)
            .unwrap();
        layout
            .mbind_replaced_range(boundary, dummy, 1 << 20, &empty_policies)
            .unwrap();
        assert!(
            layout
                .mbind_replaced_range(boundary, dummy, 1 << 20, &[Vec::new()])
                .is_err(),
            "a missing host-node mapping must be observable"
        );
    }

    #[test]
    fn relocate_allocate_leaves_gap_unbacked() {
        // End-to-end regression pin at the allocate level: a >3 GiB
        // layout registers NO KVM memslot over the device window, so the
        // guest's MMIO there traps. RAM below and above the gap is backed.
        use vm_memory::{GuestAddress, GuestMemory};
        let topo = Topology::new(1, 2, 4, 2);
        let layout = NumaMemoryLayout::compute(&topo, 4096, 0, X86_GAP).unwrap();
        let kvm = kvm_ioctls::Kvm::new().unwrap();
        let vm_fd = kvm.create_vm().unwrap();
        let alloc = layout.allocate(false, false).unwrap();
        layout.register(&vm_fd, &alloc.guest_mem).unwrap();
        assert!(
            alloc
                .guest_mem
                .get_host_address(GuestAddress(0xC000_0000))
                .is_err()
        );
        assert!(
            alloc
                .guest_mem
                .get_host_address(GuestAddress(0xFEC0_0000))
                .is_err()
        );
        assert!(alloc.guest_mem.get_host_address(GuestAddress(0)).is_ok());
        assert!(
            alloc
                .guest_mem
                .get_host_address(GuestAddress(0x1_0000_0000))
                .is_ok()
        );
    }
}
