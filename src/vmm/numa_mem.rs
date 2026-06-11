use anyhow::{Context, Result};
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

/// Result of `NumaMemoryLayout::allocate_and_register`.
pub(crate) struct AllocatedMemory {
    pub guest_mem: GuestMemoryMmap,
    pub reservation: ReservationGuard,
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
    /// contiguous (sum of region sizes); see `allocate_and_register`.
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

impl NumaMemoryLayout {
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
                    push_node_regions(&mut regions, i as u32, size, &mut linear, dram_base, mmio_gap);
                }

                anyhow::ensure!(
                    !regions.is_empty(),
                    "at least one node must have non-zero memory"
                );
            }
            None => {
                if numa_nodes <= 1 {
                    push_node_regions(&mut regions, 0, total_bytes, &mut linear, dram_base, mmio_gap);
                } else {
                    let per_node_mib = total_memory_mib / numa_nodes;
                    for i in 0..numa_nodes {
                        let mib = if i == numa_nodes - 1 {
                            total_memory_mib - per_node_mib * (numa_nodes - 1)
                        } else {
                            per_node_mib
                        };
                        let size = (mib as u64) << 20;
                        push_node_regions(&mut regions, i, size, &mut linear, dram_base, mmio_gap);
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

    /// Total guest memory in bytes (sum of all node regions).
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
    pub fn top_gpa(&self) -> u64 {
        self.regions
            .iter()
            .map(|r| r.gpa_start + r.size)
            .max()
            .unwrap_or(0)
    }

    /// If the relocated RAM top exceeds the guest's addressable physical
    /// space (`1 << phys_bits`), return that top GPA; otherwise `None`.
    /// Without rejecting this, RAM above the guest MAXPHYADDR is SILENTLY
    /// truncated by the guest kernel (e820__end_ram_pfn caps last_pfn at
    /// max_arch_pfn), so the guest boots with less RAM than advertised.
    /// `phys_bits >= 64` means no limit (the full u64 GPA space).
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

    /// Reserve contiguous VA, per-node MAP_FIXED mmap, register per-node
    /// KVM memory slots, and return the multi-region `GuestMemoryMmap`
    /// with a `ReservationGuard` that owns the VA range.
    ///
    /// Each node gets its own MAP_FIXED mmap within the reserved VA.
    /// The `MmapRegion` wrappers have `owned=false` (via `build_raw`),
    /// so their Drop is a no-op. The `ReservationGuard` munmaps the
    /// entire reservation on drop, releasing all sub-mappings.
    pub fn allocate_and_register(
        &self,
        vm_fd: &kvm_ioctls::VmFd,
        use_hugepages: bool,
        performance_mode: bool,
    ) -> Result<AllocatedMemory> {
        let total = self.total_bytes() as usize;
        let memory_mib = (total >> 20) as u32;

        let use_hugepages = use_hugepages
            || (performance_mode
                && super::host_topology::hugepages_free()
                    >= super::host_topology::hugepages_needed(memory_mib));

        // Step 1: Reserve contiguous VA with PROT_NONE.
        let reservation = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                total,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                -1,
                0,
            )
        };
        if reservation == libc::MAP_FAILED {
            // mmap can fail with the same host-resource errnos that
            // [`super::map_transient_to_contention`] classifies — most
            // commonly ENOMEM under host memory pressure when a peer is
            // holding the GuestMemoryMmap budget. Routing through the
            // classifier turns those into a SKIP banner instead of a
            // hard test failure; non-transient errnos flow through
            // unchanged so a real bug never gets misclassified.
            let io_err = std::io::Error::last_os_error();
            let errno = io_err.raw_os_error().unwrap_or(0);
            return Err(super::map_transient_to_contention(
                kvm_ioctls::Error::new(errno),
                format!("mmap VA reservation ({} bytes) failed", total),
            ));
        }

        let guard = ReservationGuard {
            addr: reservation,
            size: total,
        };

        let mut guest_regions: Vec<GuestRegionMmap> = Vec::with_capacity(self.regions.len());

        // Host VA is PACKED (gap-free): the reservation is sum-of-sizes,
        // so each region's VA offset is the running cumulative size, NOT
        // gpa_start - dram_base. Under MMIO-gap relocation a high
        // region's gpa_start jumps above gap_end; gpa_start - dram_base
        // would index past the packed reservation end (OOB MAP_FIXED).
        // The KVM slot still pairs guest_phys_addr = gpa_start (gapped)
        // with this packed userspace_addr below.
        let mut va_offset = 0usize;
        for region in &self.regions {
            let offset = va_offset;
            let node_size = region.size as usize;
            va_offset += node_size;
            let node_addr = unsafe { (reservation as *mut u8).add(offset) as *mut libc::c_void };

            // Step 2: Per-node MAP_FIXED mmap.
            let mut flags = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_FIXED;
            if use_hugepages {
                flags |= libc::MAP_HUGETLB | libc::MAP_HUGE_2MB;
            }

            let node_ptr = unsafe {
                libc::mmap(
                    node_addr,
                    node_size,
                    libc::PROT_READ | libc::PROT_WRITE,
                    flags,
                    -1,
                    0,
                )
            };
            if node_ptr == libc::MAP_FAILED {
                // mmap can fail with the same host-resource errnos that
                // [`super::map_transient_to_contention`] classifies —
                // most commonly ENOMEM (or EAGAIN under MAP_HUGETLB
                // when 2MiB pages are exhausted). Route through the
                // classifier so transient host pressure SKIPs cleanly
                // instead of failing the test as a hard fault.
                let io_err = std::io::Error::last_os_error();
                let errno = io_err.raw_os_error().unwrap_or(0);
                return Err(super::map_transient_to_contention(
                    kvm_ioctls::Error::new(errno),
                    format!(
                        "MAP_FIXED mmap for node {} ({} bytes) failed",
                        region.node_id, node_size
                    ),
                ));
            }

            // Step 5: Wrap as vm-memory types. build_raw sets owned=false.
            let mmap_region = unsafe {
                MmapRegion::build_raw(
                    node_ptr as *mut u8,
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
            //     (virt/kvm/kvm_main.c:2101).
            //   - EIO: kernel WARN-on-impossible-state path
            //     (virt/kvm/kvm_main.c:2065, arch/x86 mmu.c).
            //   - EFAULT: arm64/riscv guest-phys-bounds violation
            //     (arch/arm64/kvm/mmu.c, arch/riscv/kvm/mmu.c).
            let mem_region = kvm_bindings::kvm_userspace_memory_region {
                slot: region.slot,
                guest_phys_addr: region.gpa_start,
                memory_size: region.size,
                userspace_addr: node_ptr as u64,
                flags: 0,
            };
            unsafe {
                vm_fd.set_user_memory_region(mem_region).map_err(|e| {
                    super::map_transient_to_contention(
                        e,
                        format!(
                            "set KVM memory slot {} for node {}",
                            region.slot, region.node_id
                        ),
                    )
                })?;
            }
        }

        // Step 6: Build multi-region GuestMemoryMmap.
        let guest_mem = GuestMemoryMmap::from_regions(guest_regions)
            .context("create multi-region GuestMemoryMmap")?;

        Ok(AllocatedMemory {
            guest_mem,
            reservation: guard,
        })
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
        let topo = Topology::new(3, 3, 2, 1);
        let layout = NumaMemoryLayout::compute(&topo, 100, 0, None).unwrap();
        assert_eq!(layout.regions().len(), 3);
        let sizes: Vec<u64> = layout.regions().iter().map(|r| r.size).collect();
        assert_eq!(sizes[0], 33 << 20);
        assert_eq!(sizes[1], 33 << 20);
        assert_eq!(sizes[2], 34 << 20);
        assert_eq!(layout.total_bytes(), 100 << 20);
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

        let alloc = layout.allocate_and_register(&vm_fd, false, false).unwrap();

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

        let alloc = layout.allocate_and_register(&vm_fd, false, false).unwrap();

        use vm_memory::GuestMemoryRegion;
        let total: u64 = alloc.guest_mem.iter().map(|r| r.len()).sum();
        assert_eq!(total, 512 << 20);
        // Per-node MAP_FIXED: one GuestMemoryMmap region per node.
        assert_eq!(alloc.guest_mem.iter().count(), 2);
    }

    #[test]
    fn contiguous_host_va() {
        let topo = Topology::with_nodes(4, 2, &TWO_NODES);
        let layout = NumaMemoryLayout::compute(&topo, 512, 0, None).unwrap();

        let kvm = kvm_ioctls::Kvm::new().unwrap();
        let vm_fd = kvm.create_vm().unwrap();

        let alloc = layout.allocate_and_register(&vm_fd, false, false).unwrap();

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

        let kvm = kvm_ioctls::Kvm::new().unwrap();
        let vm_fd = kvm.create_vm().unwrap();

        let alloc = layout.allocate_and_register(&vm_fd, false, false).unwrap();

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

        let kvm = kvm_ioctls::Kvm::new().unwrap();
        let vm_fd = kvm.create_vm().unwrap();

        let alloc = layout.allocate_and_register(&vm_fd, false, false).unwrap();

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

        let kvm = kvm_ioctls::Kvm::new().unwrap();
        let vm_fd = kvm.create_vm().unwrap();

        let alloc = layout.allocate_and_register(&vm_fd, false, false).unwrap();

        let addr = alloc.reservation.addr;
        let size = alloc.reservation.size;
        assert!(!addr.is_null());
        assert_eq!(size, 64 << 20);
        // Drop releases the VA reservation.
        drop(alloc);
    }

    #[test]
    fn three_node_allocation() {
        let topo = Topology::with_nodes(4, 1, &CXL_NODES);
        let layout = NumaMemoryLayout::compute(&topo, 640, 0, None).unwrap();
        assert_eq!(layout.regions().len(), 3);

        let kvm = kvm_ioctls::Kvm::new().unwrap();
        let vm_fd = kvm.create_vm().unwrap();

        let alloc = layout.allocate_and_register(&vm_fd, false, false).unwrap();

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
    fn ram_top_exceeds_phys_bits_rejects_above_maxphyaddr() {
        // 8 GiB single node on x86 relocates above the 4 GiB MMIO gap -> top
        // GPA ~9 GiB. A 33-bit guest MAXPHYADDR (8 GiB) is exceeded -> must
        // reject (else the guest silently truncates RAM); a 40-bit one
        // (1 TiB) is not; >=64 means no limit.
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
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
        assert_eq!(layout.regions()[1].size + layout.regions()[2].size, 4096 << 20);
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
    fn relocate_allocate_leaves_gap_unbacked() {
        // End-to-end regression pin at the allocate level: a >3 GiB
        // layout registers NO KVM memslot over the device window, so the
        // guest's MMIO there traps. RAM below and above the gap is backed.
        use vm_memory::{GuestAddress, GuestMemory};
        let topo = Topology::new(1, 2, 4, 2);
        let layout = NumaMemoryLayout::compute(&topo, 4096, 0, X86_GAP).unwrap();
        let kvm = kvm_ioctls::Kvm::new().unwrap();
        let vm_fd = kvm.create_vm().unwrap();
        let alloc = layout.allocate_and_register(&vm_fd, false, false).unwrap();
        assert!(alloc.guest_mem.get_host_address(GuestAddress(0xC000_0000)).is_err());
        assert!(alloc.guest_mem.get_host_address(GuestAddress(0xFEC0_0000)).is_err());
        assert!(alloc.guest_mem.get_host_address(GuestAddress(0)).is_ok());
        assert!(alloc.guest_mem.get_host_address(GuestAddress(0x1_0000_0000)).is_ok());
    }
}
