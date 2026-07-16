use anyhow::{Context, Result};
use vm_fdt::FdtWriter;

use crate::vmm::aarch64::topology::mpidr_to_fdt_reg;
use crate::vmm::kvm::{
    DRAM_START, FDT_MAX_SIZE, GIC_DIST_BASE, GIC_DIST_SIZE, GIC_REDIST_BASE,
    GIC_REDIST_SIZE_PER_CPU, PMU_PPI, SERIAL_IRQ, SERIAL_MMIO_BASE, SERIAL_MMIO_SIZE, SERIAL2_IRQ,
    SERIAL2_MMIO_BASE, VIRTIO_BLK_IRQ, VIRTIO_BLK_MMIO_BASE, VIRTIO_CONSOLE_IRQ,
    VIRTIO_CONSOLE_MMIO_BASE, VIRTIO_NET_IRQ, VIRTIO_NET_MMIO_BASE,
};
use crate::vmm::numa_mem::NumaMemoryLayout;
use crate::vmm::topology::Topology;
use crate::vmm::virtio_blk;
use crate::vmm::virtio_console;
use crate::vmm::virtio_net;

/// GIC phandle — unique identifier referenced by interrupt-parent properties.
const GIC_PHANDLE: u32 = 1;

/// SPI interrupt type (shared peripheral interrupt).
const GIC_SPI: u32 = 0;

/// PPI interrupt type (private peripheral interrupt).
const GIC_PPI: u32 = 1;

/// IRQ_TYPE_EDGE_RISING for SPI devices driven by irqfd.
/// Edge-triggered avoids the need for resamplefd: KVM sets
/// pending_latch on injection, auto-clears after delivery.
const IRQ_TYPE_EDGE_RISING: u32 = 1;

/// IRQ_TYPE_LEVEL_LOW for timer PPIs (active-low per arm spec).
///
/// cloud-hypervisor, firecracker, AND qemu all emit LEVEL_HIGH (4) for the
/// arch-timer PPIs; ktstr's LEVEL_LOW is nonetheless correct and the
/// divergence is benign. arm arch-timer PPIs are architecturally
/// active-low; the kernel's check_ppi_trigger (drivers/clocksource/
/// arm_arch_timer.c) accepts either polarity as-is (only out-of-range
/// values are forced + warned), and KVM's vgic forces all PPIs to
/// VGIC_CONFIG_LEVEL with no polarity dimension — so HIGH vs LOW is
/// cosmetic for these PPIs under KVM. Do NOT "correct" this to HIGH to
/// match the reference VMMs.
const IRQ_TYPE_LEVEL_LOW: u32 = 8;

/// IRQ_TYPE_LEVEL_HIGH for the PMU PPI. Matches qemu virt
/// (hw/arm/virt.c GIC_FDT_IRQ_FLAGS_LEVEL_HI) and cloud-hypervisor
/// (arch/src/aarch64/fdt.rs IRQ_TYPE_LEVEL_HI).
const IRQ_TYPE_LEVEL_HIGH: u32 = 4;

/// Generate a Flattened Device Tree blob for the guest.
///
/// `mpidrs` contains the MPIDR_EL1 values read from KVM for each vCPU.
/// The FDT cpu node `reg` properties use the affinity fields from these
/// values, ensuring the FDT matches KVM's actual MPIDR assignment.
///
/// When `topo.llcs > 1`, DT cache nodes are emitted so the guest kernel
/// discovers per-LLC cache domains via `next-level-cache` phandle chains
/// in `cache_setup_of_node`. The chain always terminates at an L3 node
/// (see `write_cpus`), so a multi-LLC guest presents its LLC as a shared
/// level-3 cache regardless of the host's real cache-level count —
/// mirroring the x86 path, where the LLC is always synthesized as L3.
///
/// `guest_l1_unified` indicates the host's L1 cache is unified (from
/// sysfs). When true, CPU nodes get `cache-unified` so
/// `of_count_cache_leaves` returns 1 leaf instead of 2, matching
/// CLIDR_EL1's Ctype1=Unified (1 leaf).
///
/// When `topo.numa_nodes > 1`, NUMA topology is described via:
/// - `numa-node-id` properties on cpu and memory nodes
/// - per-NUMA-node memory nodes with disjoint address ranges
/// - a `distance-map` node with `numa-distance-map-v1` compatible
#[allow(clippy::too_many_arguments)]
pub fn create_fdt(
    topo: &Topology,
    mpidrs: &[u64],
    memory_mib: u32,
    cmdline: &str,
    initrd_addr: Option<u64>,
    initrd_size: Option<u32>,
    guest_l1_unified: bool,
    numa_layout: &NumaMemoryLayout,
    has_virtio_blk: bool,
    has_virtio_net: bool,
    has_pmu: bool,
) -> Result<Vec<u8>> {
    let mut fdt = FdtWriter::new_with_mem_reserv(&[]).context("create FDT writer")?;

    let root = fdt.begin_node("").context("begin root node")?;
    fdt.property_string("compatible", "linux,dummy-virt")
        .context("root compatible")?;
    fdt.property_u32("#address-cells", 2)
        .context("root #address-cells")?;
    fdt.property_u32("#size-cells", 2)
        .context("root #size-cells")?;
    fdt.property_u32("interrupt-parent", GIC_PHANDLE)
        .context("root interrupt-parent")?;

    // /chosen — bootargs, stdout, initrd
    write_chosen(&mut fdt, cmdline, initrd_addr, initrd_size)?;

    // /memory — guest physical RAM. When numa_nodes > 1, one memory
    // node per NUMA node with disjoint address ranges and numa-node-id.
    write_memory(&mut fdt, topo, memory_mib, numa_layout)?;

    // /cpus — one node per vCPU with MPIDR from KVM, plus cache
    // topology nodes when the topology has multiple LLCs.
    write_cpus(&mut fdt, topo, mpidrs, guest_l1_unified)?;

    // /intc — GICv3
    let num_cpus = mpidrs.len() as u32;
    write_gic(&mut fdt, num_cpus)?;

    // /serial — two ns16550a UARTs with edge-triggered SPI interrupts via irqfd
    write_serial(&mut fdt, SERIAL_MMIO_BASE, "serial0", SERIAL_IRQ)?;
    write_serial(&mut fdt, SERIAL2_MMIO_BASE, "serial1", SERIAL2_IRQ)?;

    // /virtio_mmio — virtio-console
    write_virtio_mmio(
        &mut fdt,
        VIRTIO_CONSOLE_MMIO_BASE,
        virtio_console::VIRTIO_MMIO_SIZE,
        VIRTIO_CONSOLE_IRQ,
    )?;

    // /virtio_mmio — virtio-block (only when at least one disk
    // is attached; absent FDT node leaves the slot dark so the
    // guest does not probe a non-existent device).
    if has_virtio_blk {
        write_virtio_mmio(
            &mut fdt,
            VIRTIO_BLK_MMIO_BASE,
            virtio_blk::VIRTIO_MMIO_SIZE,
            VIRTIO_BLK_IRQ,
        )?;
    }

    // /virtio_mmio — virtio-net (only when the builder attached a
    // `NetConfig`; absent FDT node leaves the slot dark so the
    // guest does not probe a non-existent device).
    if has_virtio_net {
        write_virtio_mmio(
            &mut fdt,
            VIRTIO_NET_MMIO_BASE,
            virtio_net::VIRTIO_MMIO_SIZE,
            VIRTIO_NET_IRQ,
        )?;
    }

    // /timer — arm generic timer
    write_timer(&mut fdt)?;

    // /pmu — arm,armv8-pmuv3 binding so the guest pmuv3 driver
    // attaches and perf_event_open can return a hardware backend.
    // Required by sched_ext schedulers (scx_layered, scx_cosmos)
    // that read perf counters via BPF kfuncs. KVM emulates the PMU
    // via KVM_ARM_VCPU_PMU_V3; the FDT binding gates driver probe
    // on this node's compatible string.
    //
    // Only emitted when the host KVM advertises Cap::ArmPmuV3
    // (threaded as `has_pmu` from KtstrKvm). On no-PMU hosts the
    // KVM_ARM_VCPU_PMU_V3 feature bit is masked out of vcpu_init,
    // so the in-kernel PMU emulation never runs; advertising the
    // FDT node anyway would have the guest pmuv3 driver attach to
    // the PMU PPI (PPI 7, global intid 23), fail to receive any
    // events, and log a noisy attach failure. Omitting the node
    // lets the driver silently skip.
    if has_pmu {
        write_pmu(&mut fdt)?;
    }

    // /psci — power state coordination interface
    write_psci(&mut fdt)?;

    // /distance-map — NUMA distance matrix (only for multi-NUMA)
    if topo.numa_nodes > 1 {
        write_distance_map(&mut fdt, topo)?;
    }

    fdt.end_node(root).context("end root node")?;
    let dtb = fdt.finish().context("finish FDT")?;

    anyhow::ensure!(
        dtb.len() as u64 <= FDT_MAX_SIZE,
        "FDT too large: {} bytes (max {})",
        dtb.len(),
        FDT_MAX_SIZE,
    );

    Ok(dtb)
}

/// Compute FDT load address: placed at the end of usable guest RAM.
///
/// The FDT is placed in the last `FDT_MAX_SIZE` bytes of the usable
/// DRAM region. The address must be 8-byte aligned (FDT spec requirement).
pub fn fdt_address(memory_mib: u32) -> u64 {
    let mem_size = (memory_mib as u64) << 20;
    let dram_end = DRAM_START + mem_size;
    (dram_end - FDT_MAX_SIZE) & !7
}

/// Base IPA of the per-vCPU KVM PV stolen-time region, carved from the
/// top of guest RAM just below the FDT and 64 KB-aligned (the max
/// aarch64 granule, so the `/memory` shrink lands on a page boundary
/// for every granule). [`write_memory`] excludes `[pvtime_base,
/// dram_end)` from `/memory`, and
/// [`super::kvm::KtstrKvm::setup_pvtime`] points each vCPU's PVTIME IPA
/// at `pvtime_base + cpu_id * PVTIME_SIZE_PER_CPU`. Computed in one
/// place so the host IPAs and the FDT carve-out agree exactly.
///
/// Divergence: qemu places PV-stolen-time in a separate fixed region in
/// device-IO space below DRAM (VIRT_PVTIME = 0x090a0000, hw/arm/virt.c;
/// VIRT_MEM starts at 1 GiB), with its own RAM-backed MemoryRegion. ktstr
/// instead carves from the top of guest RAM so the IPA stays inside the
/// registered guest-RAM memslots — no extra device region or memslot.
/// The cost of that choice: the carve overlaps the address range the
/// initrd would otherwise use, so `aarch64_initrd_addr` anchors the initrd
/// below `pvtime_base` (qemu needs no such bound — its region is below DRAM).
///
/// firecracker uses the SAME carve-and-exclude ktstr does: it reserves a
/// general in-RAM system region (SYSTEM_MEM, at the bottom of DRAM, shared
/// with VMGenID), allocates the steal-time IPA from it
/// (allocate_pvtime_region), and drops the region from `/memory`
/// (create_memory_node starts the node SYSTEM_MEM_SIZE above DRAM_MEM_START),
/// differing from us only in placement (its bottom-of-RAM vs our top-of-RAM).
/// So carve-and-exclude is the
/// mainstream pattern and only qemu's separate-device-region diverges;
/// cloud-hypervisor and libkrun have no aarch64 pvtime at all.
pub(crate) fn pvtime_base(memory_mib: u32, total_cpus: u32) -> u64 {
    let pvtime_size = total_cpus as u64 * super::kvm::PVTIME_SIZE_PER_CPU;
    (fdt_address(memory_mib) - pvtime_size) & !0xFFFF
}

fn write_chosen(
    fdt: &mut FdtWriter,
    cmdline: &str,
    initrd_addr: Option<u64>,
    initrd_size: Option<u32>,
) -> Result<()> {
    let chosen = fdt.begin_node("chosen").context("begin chosen")?;
    fdt.property_string("bootargs", cmdline)
        .context("bootargs")?;
    fdt.property_string("stdout-path", &format!("/serial0@{:x}", SERIAL_MMIO_BASE))
        .context("stdout-path")?;

    // KASLR seed. arm64 kernel-image KASLR seeds from `/chosen/kaslr-seed`
    // (an 8-byte property the kernel consumes-and-zeroes in
    // `kaslr_early_init`), falling back to the CPU RNG (FEAT_RNG / RNDR)
    // when the DT seed is absent. KVM does not expose RNDR to the guest
    // here, so without a DT seed the guest boots "KASLR disabled due to
    // lack of seed" and `kern_kaslr_offset` is 0 — defeating the KASLR-on
    // default and its e2e coverage. Provide a random DT seed so KASLR
    // enables deterministically regardless of guest RNDR. This matches
    // qemu (hw/arm/virt.c `create_randomness` writes an 8-byte BE
    // `kaslr-seed`, default-on) and DIVERGES from cloud-hypervisor,
    // firecracker, and libkrun, whose aarch64 /chosen carries only
    // bootargs + initrd bounds (no seed): they rely on guest FEAT_RNG /
    // RNDR or EFI-stub seeding, which our KVM guest lacks. `property_u64`
    // writes the value big-endian, which `fdt64_to_cpu` in `get_kaslr_seed`
    // reads back correctly. Omitted under `nokaslr` (e.g.
    // `#[ktstr_test(kaslr = false)]`) so the disabled-KASLR path stays
    // genuinely seedless.
    if !cmdline.split_whitespace().any(|tok| tok == "nokaslr") {
        fdt.property_u64("kaslr-seed", rand::random::<u64>())
            .context("kaslr-seed")?;
    }

    if let (Some(addr), Some(size)) = (initrd_addr, initrd_size) {
        fdt.property_u64("linux,initrd-start", addr)
            .context("initrd-start")?;
        fdt.property_u64("linux,initrd-end", addr + size as u64)
            .context("initrd-end")?;
    }

    fdt.end_node(chosen).context("end chosen")?;
    Ok(())
}

fn write_memory(
    fdt: &mut FdtWriter,
    topo: &Topology,
    memory_mib: u32,
    numa_layout: &NumaMemoryLayout,
) -> Result<()> {
    let mem_size = (memory_mib as u64) << 20;
    // Top of guest-usable RAM. The PVTIME steal region is carved just
    // below the FDT (see `pvtime_base`) and MUST be excluded from
    // /memory: the guest memremaps the SMCCC-returned IPA itself, so if
    // these pages stayed in /memory the guest could allocate them while
    // the host writes steal-time into them (silent corruption). The
    // host PVTIME IPAs point here (aarch64/kvm.rs setup_pvtime).
    let pvt = pvtime_base(memory_mib, topo.total_cpus());

    if topo.numa_nodes <= 1 {
        let usable = pvt - DRAM_START;
        let name = format!("memory@{DRAM_START:x}");
        let mem = fdt.begin_node(&name).context("begin memory")?;
        fdt.property_string("device_type", "memory")
            .context("memory device_type")?;
        fdt.property_array_u32(
            "reg",
            &[
                (DRAM_START >> 32) as u32,
                DRAM_START as u32,
                (usable >> 32) as u32,
                usable as u32,
            ],
        )
        .context("memory reg")?;
        fdt.end_node(mem).context("end memory")?;
    } else {
        // Multi-NUMA: one memory node per NumaMemoryLayout region. The
        // PVTIME carve lives in the top region (the one ending at
        // dram_end, where the FDT also sits), so shrink that region to
        // end at `pvt`; the others keep their full size.
        let dram_end = DRAM_START + mem_size;
        let regions = numa_layout.regions();
        for region in regions {
            let base = region.gpa_start;
            let length = if base + region.size == dram_end {
                anyhow::ensure!(
                    pvt > base,
                    "PVTIME carve (pvtime_base={pvt:#x}) underflows the top NUMA \
                     region base ({base:#x}); top node too small for the carve"
                );
                pvt - base
            } else {
                region.size
            };
            let name = format!("memory@{base:x}");
            let mem = fdt.begin_node(&name).context("begin memory")?;
            fdt.property_string("device_type", "memory")
                .context("memory device_type")?;
            fdt.property_array_u32(
                "reg",
                &[
                    (base >> 32) as u32,
                    base as u32,
                    (length >> 32) as u32,
                    length as u32,
                ],
            )
            .context("memory reg")?;
            fdt.property_u32("numa-node-id", region.node_id)
                .context("memory numa-node-id")?;
            fdt.end_node(mem).context("end memory")?;
        }
    }

    Ok(())
}

/// Phandle base for cache nodes. GIC uses phandle 1.
/// Grouped by LLC: each LLC's chain occupies `CACHE_CHAIN_DEPTH`
/// consecutive phandles starting at `BASE + llc * CACHE_CHAIN_DEPTH`.
const CACHE_PHANDLE_BASE: u32 = GIC_PHANDLE + 1;

/// Cache level at which a multi-LLC guest presents its LLC boundary.
///
/// Fixed at 3 (L3), mirroring x86 where the LLC is always synthesized as
/// L3, so the guest's presented LLC level is decoupled from the host's
/// real cache-level count (e.g. arm hosts with only L1+L2). Consumed by
/// `override_clidr` to cap CLIDR_EL1 to the same depth.
pub const GUEST_LLC_CACHE_LEVEL: u32 = 3;

/// Non-L1 cache nodes emitted per LLC: an L2 node chaining to an L3
/// terminal node (`CPU -> L2 -> L3`). The guest kernel assigns each leaf's
/// level by its position in the `next-level-cache` chain
/// (populate_cache_leaves counts up from 1), NOT from the DT `cache-level`
/// property, so reaching level 3 requires two non-L1 hops. On hosts whose
/// CLIDR_EL1 reports fewer than 3 levels, the guest kernel's
/// init_cache_level extends the count from the DT chain via
/// of_find_last_cache_level (the "unified external cache" path), so the L3
/// still materializes. CLIDR is capped to `GUEST_LLC_CACHE_LEVEL` in
/// `override_clidr` so hosts with MORE levels don't report leaves the DT
/// chain lacks (which makes cache_setup_of_node bail and drop DT grouping).
const CACHE_CHAIN_DEPTH: usize = (GUEST_LLC_CACHE_LEVEL - 1) as usize;

fn write_cpus(
    fdt: &mut FdtWriter,
    topo: &Topology,
    mpidrs: &[u64],
    guest_l1_unified: bool,
) -> Result<()> {
    let cpus = fdt.begin_node("cpus").context("begin cpus")?;
    fdt.property_u32("#address-cells", 1)
        .context("cpus #address-cells")?;
    fdt.property_u32("#size-cells", 0)
        .context("cpus #size-cells")?;

    // cache_setup_of_node() walks next-level-cache once per non-L1
    // cache leaf (L1 leaves stay at the CPU node). Each LLC gets its own
    // CPU->L2->L3 chain so CPUs sharing an LLC share the same phandles;
    // cache_leaves_are_shared() compares fw_token pointers set by
    // cache_setup_of_node(), so shared phandles produce a shared L3
    // domain. Single-LLC guests emit no cache nodes (no LLC boundary to
    // describe); the guest falls back to CLIDR-only arch cache info.
    let chain_depth = if topo.llcs > 1 { CACHE_CHAIN_DEPTH } else { 0 };

    for (cpu_id, &mpidr) in mpidrs.iter().enumerate() {
        let reg = mpidr_to_fdt_reg(mpidr) as u32;
        let name = format!("cpu@{cpu_id}");
        let cpu = fdt.begin_node(&name).context("begin cpu")?;
        fdt.property_string("device_type", "cpu")
            .context("cpu device_type")?;
        fdt.property_string("compatible", "arm,arm-v8")
            .context("cpu compatible")?;
        fdt.property_string("enable-method", "psci")
            .context("cpu enable-method")?;
        fdt.property_u32("reg", reg).context("cpu reg")?;
        let (llc_id, _, _) = topo.decompose(cpu_id as u32);
        if topo.numa_nodes > 1 {
            let node_id = topo.numa_node_of(llc_id);
            fdt.property_u32("numa-node-id", node_id)
                .context("cpu numa-node-id")?;
        }
        if chain_depth > 0 {
            let first_phandle = CACHE_PHANDLE_BASE + llc_id * chain_depth as u32;
            fdt.property_u32("next-level-cache", first_phandle)
                .context("cpu next-level-cache")?;
            // L1 cache properties live on the CPU node, NOT on a
            // separate cache node. The kernel's cacheinfo parser
            // (drivers/base/cacheinfo.c) walks `next-level-cache`
            // only for `level != 1`; for L1 it reads cache-unified,
            // {i,d,}-cache-size, etc. directly off the CPU node.
            // Specifically:
            //   * `init_of_cache_level` calls `of_count_cache_leaves(np)`
            //     where np is the CPU node — that helper reads
            //     `cache-unified` to decide L1 leaf count
            //     (unified=1 leaf vs. split=2 leaves).
            //   * `cache_setup_of_node` only walks
            //     `of_find_next_cache_node` once `this_leaf->level != 1`;
            //     L1 leaves keep np pointing at the CPU node.
            // When the host L1 is unified (single leaf in CLIDR),
            // of_count_cache_leaves's default would otherwise be 2
            // (separate I/D); emitting `cache-unified` on the CPU
            // node reduces the OF count to 1 to match CLIDR. This is
            // the canonical placement for L1 metadata; moving it to
            // a sibling cache node would not be picked up because
            // the parser does not walk for L1.
            if guest_l1_unified {
                fdt.property_null("cache-unified")
                    .context("cpu cache-unified")?;
            }
        }
        fdt.end_node(cpu).context("end cpu")?;
    }

    // Cache node chains: for each LLC, create `chain_depth` nodes at
    // levels 2..=(chain_depth+1) — i.e. an L2 node and an L3 terminal.
    // Each non-terminal node chains to the next via next-level-cache.
    // The terminal (L3) node is the LLC boundary — CPUs sharing it are in
    // the same LLC domain. `cache-level` here is descriptive metadata; the
    // guest assigns leaf levels by chain position, not this property.
    for llc in 0..topo.llcs {
        for d in 0..chain_depth {
            let phandle = CACHE_PHANDLE_BASE + llc * chain_depth as u32 + d as u32;
            let level = (d + 2) as u32;
            let name = format!("l{level}-cache{llc}");
            let cache = fdt.begin_node(&name).context("begin cache")?;
            fdt.property_string("compatible", "cache")
                .context("cache compatible")?;
            fdt.property_u32("cache-level", level)
                .context("cache-level")?;
            fdt.property_null("cache-unified")
                .context("cache-unified")?;
            fdt.property_phandle(phandle).context("cache phandle")?;
            if d + 1 < chain_depth {
                fdt.property_u32("next-level-cache", phandle + 1)
                    .context("cache next-level-cache")?;
            }
            fdt.end_node(cache).context("end cache")?;
        }
    }

    fdt.end_node(cpus).context("end cpus")?;
    Ok(())
}

fn write_gic(fdt: &mut FdtWriter, num_cpus: u32) -> Result<()> {
    let redist_size = num_cpus as u64 * GIC_REDIST_SIZE_PER_CPU;

    let intc = fdt
        .begin_node(&format!("intc@{GIC_DIST_BASE:x}"))
        .context("begin intc")?;
    fdt.property_string("compatible", "arm,gic-v3")
        .context("intc compatible")?;
    fdt.property_null("interrupt-controller")
        .context("interrupt-controller")?;
    fdt.property_u32("#interrupt-cells", 3)
        .context("#interrupt-cells")?;
    fdt.property_phandle(GIC_PHANDLE).context("intc phandle")?;
    // reg: distributor region, then redistributor region
    fdt.property_array_u32(
        "reg",
        &[
            (GIC_DIST_BASE >> 32) as u32,
            GIC_DIST_BASE as u32,
            (GIC_DIST_SIZE >> 32) as u32,
            GIC_DIST_SIZE as u32,
            (GIC_REDIST_BASE >> 32) as u32,
            GIC_REDIST_BASE as u32,
            (redist_size >> 32) as u32,
            redist_size as u32,
        ],
    )
    .context("intc reg")?;
    fdt.property_u32("#address-cells", 2)
        .context("intc #address-cells")?;
    fdt.property_u32("#size-cells", 2)
        .context("intc #size-cells")?;
    fdt.property_null("ranges").context("intc ranges")?;

    fdt.end_node(intc).context("end intc")?;
    Ok(())
}

fn write_serial(fdt: &mut FdtWriter, base: u64, alias: &str, irq: u32) -> Result<()> {
    let name = format!("{alias}@{base:x}");
    let serial = fdt.begin_node(&name).context("begin serial")?;
    fdt.property_string("compatible", "ns16550a")
        .context("serial compatible")?;
    fdt.property_array_u32(
        "reg",
        &[
            (base >> 32) as u32,
            base as u32,
            (SERIAL_MMIO_SIZE >> 32) as u32,
            SERIAL_MMIO_SIZE as u32,
        ],
    )
    .context("serial reg")?;
    // Edge-triggered SPI: irqfd writes the eventfd once per interrupt,
    // KVM sets pending_latch and auto-clears after delivery. No
    // resamplefd needed. FDT cell 1 is the SPI number (intid - 32).
    fdt.property_array_u32("interrupts", &[GIC_SPI, irq - 32, IRQ_TYPE_EDGE_RISING])
        .context("serial interrupts")?;
    fdt.property_u32("interrupt-parent", GIC_PHANDLE)
        .context("serial interrupt-parent")?;
    fdt.property_u32("clock-frequency", 1843200)
        .context("serial clock-frequency")?;
    fdt.property_u32("reg-shift", 0)
        .context("serial reg-shift")?;
    fdt.property_u32("reg-io-width", 1)
        .context("serial reg-io-width")?;
    fdt.end_node(serial).context("end serial")?;
    Ok(())
}

/// Emit a `virtio_mmio` device-tree node (compatible / reg / interrupts /
/// interrupt-parent).
///
/// Intentionally NO `dma-coherent` property: none of ktstr's virtio
/// devices (blk, net, console) negotiate VIRTIO_F_ACCESS_PLATFORM
/// (bit 33). Without that bit the
/// guest virtio_ring takes the DMA-quirk path (vring_use_map_api ==
/// false, drivers/virtio/virtio_ring.c): guest-physical addresses + a
/// plain write-back-cacheable ring (alloc_pages_exact, not
/// dma_alloc_coherent), so the DMA API — hence `dma-coherent` /
/// of_dma_is_coherent — is never consulted. Under KVM same-arch the host
/// maps the same physical guest RAM, so no coherency gap exists. We
/// diverge from firecracker (which emits `dma-coherent`) deliberately:
/// for a no-ACCESS_PLATFORM feature set the property is inert, not
/// load-bearing.
fn write_virtio_mmio(fdt: &mut FdtWriter, base: u64, size: u64, irq: u32) -> Result<()> {
    let name = format!("virtio_mmio@{base:x}");
    let node = fdt.begin_node(&name).context("begin virtio_mmio")?;
    fdt.property_string("compatible", "virtio,mmio")
        .context("virtio_mmio compatible")?;
    fdt.property_array_u32(
        "reg",
        &[
            (base >> 32) as u32,
            base as u32,
            (size >> 32) as u32,
            size as u32,
        ],
    )
    .context("virtio_mmio reg")?;
    fdt.property_array_u32("interrupts", &[GIC_SPI, irq - 32, IRQ_TYPE_EDGE_RISING])
        .context("virtio_mmio interrupts")?;
    fdt.property_u32("interrupt-parent", GIC_PHANDLE)
        .context("virtio_mmio interrupt-parent")?;
    fdt.end_node(node).context("end virtio_mmio")?;
    Ok(())
}

fn write_timer(fdt: &mut FdtWriter) -> Result<()> {
    let timer = fdt.begin_node("timer").context("begin timer")?;
    fdt.property_string("compatible", "arm,armv8-timer")
        .context("timer compatible")?;
    fdt.property_null("always-on").context("timer always-on")?;
    // Four PPI interrupts: secure phys, non-secure phys, virt, hyp phys.
    // Standard values from QEMU/Firecracker virt machine.
    fdt.property_array_u32(
        "interrupts",
        &[
            GIC_PPI,
            13,
            IRQ_TYPE_LEVEL_LOW, // secure physical timer
            GIC_PPI,
            14,
            IRQ_TYPE_LEVEL_LOW, // non-secure physical timer
            GIC_PPI,
            11,
            IRQ_TYPE_LEVEL_LOW, // virtual timer
            GIC_PPI,
            10,
            IRQ_TYPE_LEVEL_LOW, // hypervisor physical timer
        ],
    )
    .context("timer interrupts")?;
    fdt.end_node(timer).context("end timer")?;
    Ok(())
}

/// Emit the arm,armv8-pmuv3 PMU node.
///
/// The compatible "arm,armv8-pmuv3" is the s/w-model binding listed in
/// Documentation/devicetree/bindings/arm/pmu.yaml — the only one valid for
/// KVM-emulated PMUs. The interrupt cell is `<GIC_PPI ppi flags>`; PMU_PPI
/// (=7) is in the GIC PPI namespace (FDT cell, not the global intid that
/// KVM_ARM_VCPU_PMU_V3_IRQ takes). Level-high matches qemu virt
/// (GIC_FDT_IRQ_FLAGS_LEVEL_HI) and cloud-hypervisor
/// (create_pmu_node IRQ_TYPE_LEVEL_HI).
///
/// interrupt-affinity is omitted: the PPI itself is per-CPU, so the
/// driver attaches to all CPUs without an explicit phandle list — matches
/// the binding doc note "unless this is already specified by the PPI
/// interrupt specifier itself".
fn write_pmu(fdt: &mut FdtWriter) -> Result<()> {
    let pmu = fdt.begin_node("pmu").context("begin pmu")?;
    fdt.property_string("compatible", "arm,armv8-pmuv3")
        .context("pmu compatible")?;
    fdt.property_array_u32("interrupts", &[GIC_PPI, PMU_PPI, IRQ_TYPE_LEVEL_HIGH])
        .context("pmu interrupts")?;
    fdt.end_node(pmu).context("end pmu")?;
    Ok(())
}

fn write_psci(fdt: &mut FdtWriter) -> Result<()> {
    let psci = fdt.begin_node("psci").context("begin psci")?;
    fdt.property_string("compatible", "arm,psci-0.2")
        .context("psci compatible")?;
    fdt.property_string("method", "hvc")
        .context("psci method")?;
    fdt.end_node(psci).context("end psci")?;
    Ok(())
}

/// Write a `distance-map` node with `numa-distance-map-v1` compatible.
///
/// The kernel parses `distance-matrix` as a flat array of (nodea, nodeb,
/// distance) triples. Distances come from `topo.distance()`, defaulting
/// to 10 (local) / 20 (remote).
fn write_distance_map(fdt: &mut FdtWriter, topo: &Topology) -> Result<()> {
    let dm = fdt
        .begin_node("distance-map")
        .context("begin distance-map")?;
    fdt.property_string("compatible", "numa-distance-map-v1")
        .context("distance-map compatible")?;

    // Build flat (nodea, nodeb, distance) triples for the full NxN matrix.
    let n = topo.numa_nodes;
    let mut matrix = Vec::with_capacity((n * n * 3) as usize);
    for i in 0..n {
        for j in 0..n {
            matrix.push(i);
            matrix.push(j);
            matrix.push(topo.distance(i, j) as u32);
        }
    }
    fdt.property_array_u32("distance-matrix", &matrix)
        .context("distance-matrix")?;

    fdt.end_node(dm).context("end distance-map")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_topo() -> Topology {
        Topology {
            llcs: 1,
            cores_per_llc: 2,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
            llc_cores: None,
        }
    }

    /// Generate fake MPIDRs for testing (bit 31 set, linear Aff0).
    fn fake_mpidrs(count: u32) -> Vec<u64> {
        (0..count).map(|i| (1u64 << 31) | i as u64).collect()
    }

    fn test_layout(topo: &Topology, mib: u32) -> NumaMemoryLayout {
        NumaMemoryLayout::compute(topo, mib, DRAM_START, None).unwrap()
    }

    #[allow(clippy::too_many_arguments)]
    fn test_fdt(
        topo: &Topology,
        mpidrs: &[u64],
        memory_mib: u32,
        cmdline: &str,
        initrd_addr: Option<u64>,
        initrd_size: Option<u32>,
        guest_l1_unified: bool,
    ) -> Result<Vec<u8>> {
        let layout = test_layout(topo, memory_mib);
        create_fdt(
            topo,
            mpidrs,
            memory_mib,
            cmdline,
            initrd_addr,
            initrd_size,
            guest_l1_unified,
            &layout,
            false,
            false,
            true,
        )
    }

    #[test]
    fn create_fdt_minimal() {
        let topo = default_topo();
        let mpidrs = fake_mpidrs(topo.total_cpus());
        let dtb = test_fdt(&topo, &mpidrs, 256, "console=ttyS0", None, None, false);
        assert!(dtb.is_ok(), "FDT creation failed: {:?}", dtb.err());
        let dtb = dtb.unwrap();
        assert_eq!(&dtb[..4], &[0xd0, 0x0d, 0xfe, 0xed]);
    }

    /// The initrd branch in `write_chosen` (fdt.rs `if let (Some, Some)`)
    /// must emit `/chosen/linux,initrd-start = addr` and
    /// `linux,initrd-end = addr + size`. Parse the DTB and pin both
    /// values — in particular the `addr + size` arithmetic for the end
    /// property, which is the whole point of the branch. A regression
    /// that dropped a property, swapped start/end, or got the math wrong
    /// would change the byte values here. The companion
    /// `create_fdt_no_initrd_props_absent` pins the `None` arm.
    #[test]
    fn create_fdt_with_initrd() {
        let topo = default_topo();
        let mpidrs = fake_mpidrs(topo.total_cpus());
        const ADDR: u64 = 0x4020_0000;
        const SIZE: u32 = 0x10_0000;
        let dtb = test_fdt(
            &topo,
            &mpidrs,
            256,
            "console=ttyS0",
            Some(ADDR),
            Some(SIZE),
            false,
        )
        .unwrap();
        let props = parse_dtb_props(&dtb);

        assert_eq!(
            prop_u64(&props, "chosen", "linux,initrd-start"),
            Some(ADDR),
            "linux,initrd-start must equal initrd_addr",
        );
        assert_eq!(
            prop_u64(&props, "chosen", "linux,initrd-end"),
            Some(ADDR + SIZE as u64),
            "linux,initrd-end must equal initrd_addr + initrd_size",
        );
    }

    /// With `initrd_addr = None` the `if let (Some, Some)` gate in
    /// `write_chosen` must NOT emit either initrd property. Pinning the
    /// absence proves the gate, not just the populated branch — a
    /// regression that emitted the properties unconditionally (e.g. with
    /// a zero/garbage address) would trip here.
    #[test]
    fn create_fdt_no_initrd_props_absent() {
        let topo = default_topo();
        let mpidrs = fake_mpidrs(topo.total_cpus());
        let dtb = test_fdt(&topo, &mpidrs, 256, "console=ttyS0", None, None, false).unwrap();
        let props = parse_dtb_props(&dtb);

        assert_eq!(
            prop_u64(&props, "chosen", "linux,initrd-start"),
            None,
            "linux,initrd-start must be absent when initrd_addr is None",
        );
        assert_eq!(
            prop_u64(&props, "chosen", "linux,initrd-end"),
            None,
            "linux,initrd-end must be absent when initrd_addr is None",
        );
    }

    /// Large SMP topology: 2 LLCs * 4 cores * 2 threads = 16 vCPUs.
    /// Every multi-LLC guest presents a fixed CPU -> L2 -> L3 chain
    /// (chain_depth = 2) so the LLC boundary lands at L3. Distinct from
    /// `fdt_cache_topology_multi_llc` (4 vCPUs): this pins the per-CPU node
    /// count for a 16-vCPU shape AND the two-hop L3 terminal chain. Parses
    /// the DTB and asserts generated content rather than mere non-panic.
    #[test]
    fn create_fdt_smp() {
        let topo = Topology {
            llcs: 2,
            cores_per_llc: 4,
            threads_per_core: 2,
            numa_nodes: 1,
            nodes: None,
            distances: None,
            llc_cores: None,
        };
        assert_eq!(topo.total_cpus(), 16);
        let mpidrs = fake_mpidrs(topo.total_cpus());
        let dtb = test_fdt(&topo, &mpidrs, 1024, "console=ttyS0", None, None, false).unwrap();
        let props = parse_dtb_props(&dtb);

        // All 16 cpu@N nodes must exist (device_type present) and point
        // at their LLC's L2 node via next-level-cache.
        for cpu_id in 0..topo.total_cpus() {
            let node_path = format!("cpus/cpu@{cpu_id}");
            assert_eq!(
                prop_str(&props, &node_path, "device_type").as_deref(),
                Some("cpu"),
                "cpu {cpu_id}: missing cpu node",
            );
            assert!(
                prop_u32(&props, &node_path, "next-level-cache").is_some(),
                "cpu {cpu_id}: missing next-level-cache",
            );
        }

        // chain_depth=2: each LLC emits an L2 node (level 2, chaining to
        // L3) and a terminal L3 node (level 3, NO next-level-cache). The
        // two LLCs' L3 nodes carry distinct phandles.
        for llc in 0..topo.llcs {
            let l2_path = format!("cpus/l2-cache{llc}");
            assert_eq!(
                prop_u32(&props, &l2_path, "cache-level"),
                Some(2),
                "L2 cache{llc}: wrong cache-level",
            );
            assert!(
                prop_u32(&props, &l2_path, "next-level-cache").is_some(),
                "L2 cache{llc}: must chain to L3 (chain_depth=2)",
            );
            let l3_path = format!("cpus/l3-cache{llc}");
            assert_eq!(
                prop_u32(&props, &l3_path, "cache-level"),
                Some(3),
                "L3 cache{llc}: wrong cache-level",
            );
            assert!(
                prop_u32(&props, &l3_path, "next-level-cache").is_none(),
                "L3 cache{llc}: terminal node must not chain further",
            );
        }
        let l3_0 = prop_u32(&props, "cpus/l3-cache0", "phandle").unwrap();
        let l3_1 = prop_u32(&props, "cpus/l3-cache1", "phandle").unwrap();
        assert_ne!(l3_0, l3_1, "per-LLC L3 phandles must differ");

        // CPUs sharing an LLC share the L2 phandle; CPUs in different
        // LLCs differ. cpu@0..@7 are LLC 0, cpu@8..@15 are LLC 1
        // (8 vCPUs per LLC: 4 cores * 2 threads).
        let cpu0 = prop_u32(&props, "cpus/cpu@0", "next-level-cache").unwrap();
        let cpu7 = prop_u32(&props, "cpus/cpu@7", "next-level-cache").unwrap();
        let cpu8 = prop_u32(&props, "cpus/cpu@8", "next-level-cache").unwrap();
        assert_eq!(cpu0, cpu7, "cpu@0 and cpu@7 share LLC 0");
        assert_ne!(cpu0, cpu8, "cpu@0 (LLC 0) and cpu@8 (LLC 1) differ");
    }

    #[test]
    fn create_fdt_multi_numa() {
        let topo = Topology {
            llcs: 4,
            cores_per_llc: 2,
            threads_per_core: 1,
            numa_nodes: 2,
            nodes: None,
            distances: None,
            llc_cores: None,
        };
        let mpidrs = fake_mpidrs(topo.total_cpus());
        let dtb = test_fdt(&topo, &mpidrs, 512, "console=ttyS0", None, None, false);
        assert!(dtb.is_ok(), "FDT creation failed: {:?}", dtb.err());
    }

    /// 3-NUMA topology: 6 LLCs across 3 nodes (2 LLCs/node), SMT. The
    /// distance-matrix for this exact shape is covered by
    /// `fdt_distance_map`; here we pin the per-CPU `numa-node-id` and the
    /// three per-node `/memory` regions, which `fdt_cpu_numa_node_ids` /
    /// `fdt_memory_nodes_multi_numa` only exercise for the 2-NUMA case.
    /// `check_cpu_numa_node_ids` / `check_memory_nodes` derive the
    /// expected node assignment and region layout from the production
    /// `numa_node_of` / `NumaMemoryLayout::compute`, so the emitted FDT
    /// is checked against the layout, not mere non-panic.
    #[test]
    fn create_fdt_three_numa_nodes() {
        let topo = Topology {
            llcs: 6,
            cores_per_llc: 4,
            threads_per_core: 2,
            numa_nodes: 3,
            nodes: None,
            distances: None,
            llc_cores: None,
        };
        let mpidrs = fake_mpidrs(topo.total_cpus());
        let memory_mib: u32 = 1024;
        let dtb = test_fdt(
            &topo,
            &mpidrs,
            memory_mib,
            "console=ttyS0",
            None,
            None,
            false,
        )
        .unwrap();
        let props = parse_dtb_props(&dtb);

        check_cpu_numa_node_ids(&topo, &props);
        check_memory_nodes(&topo, &props, memory_mib);

        // Exactly three /memory nodes, one per NUMA node (no extra or
        // missing region for the 3-node case).
        let mem_node_count = props
            .iter()
            .filter(|(n, p, _)| n.starts_with("memory@") && p == "device_type")
            .count();
        assert_eq!(
            mem_node_count, 3,
            "3-NUMA must emit one /memory node per node",
        );
    }

    #[test]
    fn fdt_address_aligned() {
        let addr = fdt_address(256);
        assert_eq!(addr % 8, 0, "FDT address must be 8-byte aligned");
    }

    // -----------------------------------------------------------------------
    // Minimal DTB parser for content validation tests.
    //
    // FDT binary format (big-endian):
    //   Header: magic (0xd00dfeed), totalsize, off_dt_struct, off_dt_strings, ...
    //   Structure block: stream of tokens:
    //     FDT_BEGIN_NODE (1): u32 token, null-terminated name, pad to 4-byte
    //     FDT_END_NODE   (2): u32 token
    //     FDT_PROP       (3): u32 token, u32 len, u32 nameoff, [len bytes data], pad
    //     FDT_NOP        (4): u32 token
    //     FDT_END        (9): u32 token
    // -----------------------------------------------------------------------

    const FDT_MAGIC: u32 = 0xd00dfeed;
    const FDT_BEGIN_NODE: u32 = 1;
    const FDT_END_NODE: u32 = 2;
    const FDT_PROP: u32 = 3;
    const FDT_END: u32 = 9;

    fn read_be32(dtb: &[u8], off: usize) -> u32 {
        u32::from_be_bytes(dtb[off..off + 4].try_into().unwrap())
    }

    /// Walk the DTB structure block and collect (node_path, prop_name, prop_data)
    /// tuples. Only descends into nodes; does not interpret property values.
    fn parse_dtb_props(dtb: &[u8]) -> Vec<(String, String, Vec<u8>)> {
        assert_eq!(read_be32(dtb, 0), FDT_MAGIC, "not a valid DTB");
        let off_struct = read_be32(dtb, 8) as usize;
        let off_strings = read_be32(dtb, 12) as usize;

        let mut pos = off_struct;
        let mut path_stack: Vec<String> = Vec::new();
        let mut results = Vec::new();

        loop {
            let token = read_be32(dtb, pos);
            pos += 4;
            match token {
                FDT_BEGIN_NODE => {
                    // Read null-terminated node name.
                    let name_start = pos;
                    while dtb[pos] != 0 {
                        pos += 1;
                    }
                    let name = std::str::from_utf8(&dtb[name_start..pos])
                        .unwrap()
                        .to_string();
                    pos += 1; // skip null
                    pos = (pos + 3) & !3; // align to 4
                    // Skip the root node (empty name) to avoid a leading
                    // "/" separator in join()-ed paths.
                    if !name.is_empty() {
                        path_stack.push(name);
                    }
                }
                FDT_END_NODE => {
                    path_stack.pop();
                }
                FDT_PROP => {
                    let len = read_be32(dtb, pos) as usize;
                    pos += 4;
                    let nameoff = read_be32(dtb, pos) as usize;
                    pos += 4;
                    let data = dtb[pos..pos + len].to_vec();
                    pos += len;
                    pos = (pos + 3) & !3; // align to 4

                    // Read property name from strings table.
                    let str_start = off_strings + nameoff;
                    let mut str_end = str_start;
                    while dtb[str_end] != 0 {
                        str_end += 1;
                    }
                    let prop_name = std::str::from_utf8(&dtb[str_start..str_end])
                        .unwrap()
                        .to_string();

                    let node_path = path_stack.join("/");
                    results.push((node_path, prop_name, data));
                }
                FDT_END => break,
                _ => {} // FDT_NOP or unknown — skip
            }
        }
        results
    }

    /// Extract a u32 property value from parsed props.
    fn prop_u32(props: &[(String, String, Vec<u8>)], node: &str, name: &str) -> Option<u32> {
        props.iter().find_map(|(n, p, d)| {
            if n == node && p == name && d.len() == 4 {
                Some(u32::from_be_bytes(d[..4].try_into().unwrap()))
            } else {
                None
            }
        })
    }

    /// Extract a u64 property value from parsed props. FDT u64 cells
    /// are big-endian; `property_u64` (used for initrd bounds and the
    /// kaslr seed) writes them that way.
    fn prop_u64(props: &[(String, String, Vec<u8>)], node: &str, name: &str) -> Option<u64> {
        props.iter().find_map(|(n, p, d)| {
            if n == node && p == name && d.len() == 8 {
                Some(u64::from_be_bytes(d[..8].try_into().unwrap()))
            } else {
                None
            }
        })
    }

    /// Extract a string property value from parsed props. FDT
    /// `property_string` writes a null-terminated string; the trailing
    /// NUL is stripped.
    fn prop_str(props: &[(String, String, Vec<u8>)], node: &str, name: &str) -> Option<String> {
        props.iter().find_map(|(n, p, d)| {
            if n == node && p == name {
                Some(
                    std::str::from_utf8(d)
                        .unwrap()
                        .trim_end_matches('\0')
                        .to_string(),
                )
            } else {
                None
            }
        })
    }

    /// Extract a u32-array property from parsed props.
    fn prop_u32_array(
        props: &[(String, String, Vec<u8>)],
        node: &str,
        name: &str,
    ) -> Option<Vec<u32>> {
        props.iter().find_map(|(n, p, d)| {
            if n == node && p == name && d.len() % 4 == 0 {
                Some(
                    d.chunks_exact(4)
                        .map(|c| u32::from_be_bytes(c.try_into().unwrap()))
                        .collect(),
                )
            } else {
                None
            }
        })
    }

    #[test]
    fn parse_dtb_props_paths_no_leading_slash() {
        let topo = default_topo();
        let mpidrs = fake_mpidrs(topo.total_cpus());
        let dtb = test_fdt(&topo, &mpidrs, 256, "console=ttyS0", None, None, false).unwrap();
        let props = parse_dtb_props(&dtb);

        // Top-level node paths must not start with "/".
        let cpus_prop = props
            .iter()
            .find(|(n, p, _)| n == "cpus" && p == "#address-cells");
        assert!(cpus_prop.is_some(), "expected path 'cpus', not '/cpus'");

        // Nested node paths use "/" as separator without leading slash.
        let cpu0_prop = props
            .iter()
            .find(|(n, p, _)| n == "cpus/cpu@0" && p == "device_type");
        assert!(
            cpu0_prop.is_some(),
            "expected path 'cpus/cpu@0', not '/cpus/cpu@0'"
        );

        // No path should start with "/".
        for (path, _, _) in &props {
            assert!(
                !path.starts_with('/'),
                "path {path:?} must not start with '/'"
            );
        }
    }

    fn check_cpu_numa_node_ids(topo: &Topology, props: &[(String, String, Vec<u8>)]) {
        for cpu_id in 0..topo.total_cpus() {
            let node_path = format!("cpus/cpu@{cpu_id}");
            let numa_id = prop_u32(props, &node_path, "numa-node-id")
                .unwrap_or_else(|| panic!("cpu {cpu_id}: missing numa-node-id"));
            let (llc_id, _, _) = topo.decompose(cpu_id);
            let expected = topo.numa_node_of(llc_id);
            assert_eq!(
                numa_id, expected,
                "cpu {cpu_id}: numa-node-id {numa_id} != expected {expected}"
            );
        }
    }

    #[test]
    fn fdt_cpu_numa_node_ids() {
        // No-SMT variant: 4 LLCs, 2 cores/LLC, 1 thread, 2 NUMA nodes.
        let topo = Topology {
            llcs: 4,
            cores_per_llc: 2,
            threads_per_core: 1,
            numa_nodes: 2,
            nodes: None,
            distances: None,
            llc_cores: None,
        };
        let mpidrs = fake_mpidrs(topo.total_cpus());
        let dtb = test_fdt(&topo, &mpidrs, 512, "console=ttyS0", None, None, false).unwrap();
        check_cpu_numa_node_ids(&topo, &parse_dtb_props(&dtb));

        // SMT variant: sibling threads share the same LLC and must get
        // the same numa-node-id.
        let topo_smt = Topology {
            llcs: 4,
            cores_per_llc: 2,
            threads_per_core: 2,
            numa_nodes: 2,
            nodes: None,
            distances: None,
            llc_cores: None,
        };
        let mpidrs_smt = fake_mpidrs(topo_smt.total_cpus());
        let dtb_smt = test_fdt(
            &topo_smt,
            &mpidrs_smt,
            512,
            "console=ttyS0",
            None,
            None,
            false,
        )
        .unwrap();
        check_cpu_numa_node_ids(&topo_smt, &parse_dtb_props(&dtb_smt));
    }

    #[test]
    fn fdt_single_numa_no_numa_props() {
        let topo = Topology {
            llcs: 2,
            cores_per_llc: 4,
            threads_per_core: 2,
            numa_nodes: 1,
            nodes: None,
            distances: None,
            llc_cores: None,
        };
        let mpidrs = fake_mpidrs(topo.total_cpus());
        let dtb = test_fdt(&topo, &mpidrs, 256, "console=ttyS0", None, None, false).unwrap();
        let props = parse_dtb_props(&dtb);

        // CPU nodes must NOT have numa-node-id when numa_nodes == 1.
        for cpu_id in 0..topo.total_cpus() {
            let node_path = format!("cpus/cpu@{cpu_id}");
            assert!(
                prop_u32(&props, &node_path, "numa-node-id").is_none(),
                "cpu {cpu_id}: numa-node-id must be absent for single-NUMA"
            );
        }

        // distance-map node must not exist.
        let has_distance_map = props
            .iter()
            .any(|(n, _, _)| n == "distance-map" || n.starts_with("distance-map/"));
        assert!(
            !has_distance_map,
            "distance-map node must not exist for single-NUMA"
        );
    }

    /// Check multi-NUMA memory nodes: numa-node-id, reg, contiguity, total size.
    fn check_memory_nodes(topo: &Topology, props: &[(String, String, Vec<u8>)], memory_mib: u32) {
        let mem_size = (memory_mib as u64) << 20;
        let dram_end = DRAM_START + mem_size;
        // The top region (the one ending at dram_end) is shrunk to
        // exclude the PVTIME+FDT carve [pvtime_base, dram_end); every
        // other region keeps its full size. See `write_memory`.
        let pvt = pvtime_base(memory_mib, topo.total_cpus());
        let layout = NumaMemoryLayout::compute(topo, memory_mib, DRAM_START, None).unwrap();
        let regions = layout.regions();

        let mut prev_end: Option<u64> = None;
        let mut total_size: u64 = 0;

        for (i, region) in regions.iter().enumerate() {
            let base = region.gpa_start;
            let node_name = format!("memory@{base:x}");

            let numa_id = prop_u32(props, &node_name, "numa-node-id")
                .unwrap_or_else(|| panic!("memory region {i}: missing numa-node-id"));
            assert_eq!(
                numa_id, region.node_id,
                "memory region {i}: wrong numa-node-id"
            );

            let reg = prop_u32_array(props, &node_name, "reg")
                .unwrap_or_else(|| panic!("memory region {i}: missing reg"));
            assert_eq!(reg.len(), 4, "memory region {i}: reg must have 4 cells");

            let reg_base = ((reg[0] as u64) << 32) | reg[1] as u64;
            assert_eq!(reg_base, base, "memory region {i}: wrong base address");

            let reg_size = ((reg[2] as u64) << 32) | reg[3] as u64;
            // The top region carries the carve; all others are full size.
            let expected_size = if base + region.size == dram_end {
                pvt - base
            } else {
                region.size
            };
            assert_eq!(
                reg_size, expected_size,
                "memory region {i}: wrong size (carve-adjusted)"
            );

            if let Some(prev) = prev_end {
                assert_eq!(
                    reg_base, prev,
                    "memory region {i}: not contiguous (base {reg_base:#x} != prev end {prev:#x})"
                );
            }
            prev_end = Some(reg_base + reg_size);

            total_size += reg_size;
        }

        // Advertised RAM is exactly [DRAM_START, pvtime_base): the
        // PVTIME+FDT carve at the top is excluded so the guest never
        // reuses the steal-time region (kvm.rs setup_pvtime points each
        // vCPU's PVTIME IPA there).
        assert_eq!(
            total_size,
            pvt - DRAM_START,
            "total advertised memory {total_size:#x} != pvtime_base-DRAM_START {:#x}",
            pvt - DRAM_START
        );
        assert_eq!(
            prev_end,
            Some(pvt),
            "highest /memory region must end at pvtime_base {pvt:#x}"
        );
    }

    #[test]
    fn fdt_memory_nodes_multi_numa() {
        let topo = Topology {
            llcs: 4,
            cores_per_llc: 2,
            threads_per_core: 1,
            numa_nodes: 2,
            nodes: None,
            distances: None,
            llc_cores: None,
        };
        let mpidrs = fake_mpidrs(topo.total_cpus());

        let memory_mib: u32 = 512;
        let dtb = test_fdt(
            &topo,
            &mpidrs,
            memory_mib,
            "console=ttyS0",
            None,
            None,
            false,
        )
        .unwrap();
        check_memory_nodes(&topo, &parse_dtb_props(&dtb), memory_mib);
    }

    #[test]
    fn fdt_memory_single_numa_excludes_pvtime_carve() {
        // Single-NUMA: the sole /memory node must end at pvtime_base, not
        // dram_end. The PVTIME+FDT carve [pvtime_base, dram_end) is
        // excluded so the guest never allocates the steal-time region the
        // host writes into (kvm.rs setup_pvtime points each vCPU's PVTIME
        // IPA at pvtime_base + cpu_id * PVTIME_SIZE_PER_CPU).
        let topo = Topology {
            llcs: 2,
            cores_per_llc: 4,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
            llc_cores: None,
        };
        let mpidrs = fake_mpidrs(topo.total_cpus());
        let memory_mib: u32 = 512;
        let dtb = test_fdt(
            &topo,
            &mpidrs,
            memory_mib,
            "console=ttyS0",
            None,
            None,
            false,
        )
        .unwrap();
        let props = parse_dtb_props(&dtb);

        let pvt = pvtime_base(memory_mib, topo.total_cpus());
        let node = format!("memory@{DRAM_START:x}");
        let reg = prop_u32_array(&props, &node, "reg").expect("single-NUMA memory reg");
        assert_eq!(reg.len(), 4, "memory reg must have 4 cells");
        let reg_base = ((reg[0] as u64) << 32) | reg[1] as u64;
        let reg_size = ((reg[2] as u64) << 32) | reg[3] as u64;
        assert_eq!(reg_base, DRAM_START, "single-NUMA memory base");
        assert_eq!(
            reg_base + reg_size,
            pvt,
            "single-NUMA /memory must end at pvtime_base (carve excluded)"
        );
    }

    #[test]
    fn pvtime_base_carve_below_fdt_and_fits_all_vcpus() {
        // pvtime_base must be 64KB-aligned (the max aarch64 granule, so
        // the /memory shrink lands on a page boundary for every granule),
        // strictly below the FDT, and leave room for one
        // PVTIME_SIZE_PER_CPU slot per vCPU between it and the FDT — so no
        // per-vCPU PVTIME IPA spills into the FDT region.
        let memory_mib: u32 = 512;
        let per_cpu = crate::vmm::aarch64::kvm::PVTIME_SIZE_PER_CPU;
        for total_cpus in [1u32, 8, 64, 256] {
            let pvt = pvtime_base(memory_mib, total_cpus);
            let fdt = fdt_address(memory_mib);
            assert_eq!(pvt & 0xFFFF, 0, "pvtime_base {pvt:#x} must be 64KB-aligned");
            assert!(
                pvt >= DRAM_START && pvt < fdt,
                "pvtime_base {pvt:#x} must be in [DRAM_START, fdt_address={fdt:#x})"
            );
            assert!(
                fdt - pvt >= total_cpus as u64 * per_cpu,
                "carve [{pvt:#x},{fdt:#x}) too small for {total_cpus} vCPUs * {per_cpu}B"
            );
        }
    }

    #[test]
    fn fdt_distance_map() {
        let topo = Topology {
            llcs: 6,
            cores_per_llc: 4,
            threads_per_core: 2,
            numa_nodes: 3,
            nodes: None,
            distances: None,
            llc_cores: None,
        };
        let mpidrs = fake_mpidrs(topo.total_cpus());
        let dtb = test_fdt(&topo, &mpidrs, 1024, "console=ttyS0", None, None, false).unwrap();
        let props = parse_dtb_props(&dtb);

        let matrix = prop_u32_array(&props, "distance-map", "distance-matrix")
            .expect("missing distance-matrix property");

        let n = topo.numa_nodes;
        // NxN matrix of (nodea, nodeb, distance) triples.
        assert_eq!(
            matrix.len(),
            (n * n * 3) as usize,
            "distance-matrix length: expected {} triples",
            n * n,
        );

        let mut idx = 0;
        for i in 0..n {
            for j in 0..n {
                assert_eq!(matrix[idx], i, "triple ({i},{j}): wrong nodea");
                assert_eq!(matrix[idx + 1], j, "triple ({i},{j}): wrong nodeb");
                let expected_dist = if i == j { 10 } else { 20 };
                assert_eq!(
                    matrix[idx + 2],
                    expected_dist,
                    "triple ({i},{j}): distance {} != expected {expected_dist}",
                    matrix[idx + 2],
                );
                idx += 3;
            }
        }
    }

    #[test]
    fn fdt_cache_topology_multi_llc() {
        // 2 LLCs, 2 cores each. Multi-LLC guests always present L1/L2/L3
        // (chain_depth = 2): CPU -> L2 node -> L3 node per LLC.
        let topo = Topology {
            llcs: 2,
            cores_per_llc: 2,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
            llc_cores: None,
        };
        let mpidrs = fake_mpidrs(topo.total_cpus());
        let dtb = test_fdt(&topo, &mpidrs, 512, "console=ttyS0", None, None, false).unwrap();
        let props = parse_dtb_props(&dtb);

        // Each CPU must have next-level-cache pointing to its LLC's L2 node.
        for cpu_id in 0..topo.total_cpus() {
            let node_path = format!("cpus/cpu@{cpu_id}");
            let nlc = prop_u32(&props, &node_path, "next-level-cache");
            assert!(
                nlc.is_some(),
                "cpu {cpu_id}: missing next-level-cache phandle"
            );
        }

        // CPUs in the same LLC must share the same phandle.
        let cpu0_nlc = prop_u32(&props, "cpus/cpu@0", "next-level-cache").unwrap();
        let cpu1_nlc = prop_u32(&props, "cpus/cpu@1", "next-level-cache").unwrap();
        assert_eq!(
            cpu0_nlc, cpu1_nlc,
            "CPU 0 and 1 (same LLC) must share phandle"
        );

        // CPUs in different LLCs must have different phandles.
        let cpu2_nlc = prop_u32(&props, "cpus/cpu@2", "next-level-cache").unwrap();
        assert_ne!(
            cpu0_nlc, cpu2_nlc,
            "CPU 0 and 2 (different LLC) must differ"
        );

        // L2 cache nodes must exist with correct properties.
        for llc in 0..2u32 {
            let l2_path = format!("cpus/l2-cache{llc}");
            assert_eq!(
                prop_u32(&props, &l2_path, "cache-level"),
                Some(2),
                "L2 cache{llc}: wrong cache-level"
            );
            // L2 must chain to L3 via next-level-cache.
            let l2_nlc = prop_u32(&props, &l2_path, "next-level-cache");
            assert!(l2_nlc.is_some(), "L2 cache{llc}: missing next-level-cache");

            let l3_path = format!("cpus/l3-cache{llc}");
            assert_eq!(
                prop_u32(&props, &l3_path, "cache-level"),
                Some(3),
                "L3 cache{llc}: wrong cache-level"
            );
            // L3 must NOT have next-level-cache (terminal).
            assert!(
                prop_u32(&props, &l3_path, "next-level-cache").is_none(),
                "L3 cache{llc}: should not have next-level-cache"
            );
        }

        // L3 nodes for different LLCs must have different phandles.
        let l3_0_phandle = prop_u32(&props, "cpus/l3-cache0", "phandle").unwrap();
        let l3_1_phandle = prop_u32(&props, "cpus/l3-cache1", "phandle").unwrap();
        assert_ne!(
            l3_0_phandle, l3_1_phandle,
            "L3 phandles must differ per LLC"
        );
    }

    #[test]
    fn fdt_no_cache_nodes_single_llc() {
        // Single LLC: no cache nodes should be emitted.
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 4,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
            llc_cores: None,
        };
        let mpidrs = fake_mpidrs(topo.total_cpus());
        let dtb = test_fdt(&topo, &mpidrs, 256, "console=ttyS0", None, None, false).unwrap();
        let props = parse_dtb_props(&dtb);

        // CPU 0 must NOT have next-level-cache.
        assert!(
            prop_u32(&props, "cpus/cpu@0", "next-level-cache").is_none(),
            "single-LLC: cpu should not have next-level-cache"
        );

        // No cache nodes should exist.
        let has_cache = props
            .iter()
            .any(|(n, _, _)| n.contains("cache") && n.starts_with("cpus/l"));
        assert!(!has_cache, "single-LLC: no cache nodes expected");
    }

    #[test]
    fn fdt_pmu_node_present() {
        // Verify the FDT emits an arm,armv8-pmuv3 pmu node referencing
        // PPI 7 with level-high triggering — the binding the guest
        // pmuv3 driver attaches to so BPF perf-event syscalls succeed.
        let topo = default_topo();
        let mpidrs = fake_mpidrs(topo.total_cpus());
        let dtb = test_fdt(&topo, &mpidrs, 256, "console=ttyS0", None, None, false).unwrap();
        let props = parse_dtb_props(&dtb);

        let compat = props
            .iter()
            .find(|(n, p, _)| n == "pmu" && p == "compatible")
            .expect("pmu node must exist with compatible property");
        assert_eq!(
            std::str::from_utf8(&compat.2)
                .unwrap()
                .trim_end_matches('\0'),
            "arm,armv8-pmuv3",
            "pmu compatible must match the kernel s/w-model binding",
        );

        let irq = prop_u32_array(&props, "pmu", "interrupts").expect("pmu interrupts must exist");
        assert_eq!(irq.len(), 3, "pmu interrupt cell count");
        assert_eq!(irq[0], GIC_PPI, "pmu interrupt type must be PPI");
        assert_eq!(irq[1], 7, "pmu interrupt PPI must be 7");
        assert_eq!(
            irq[2], IRQ_TYPE_LEVEL_HIGH,
            "pmu interrupt flags must be level-high"
        );
    }

    /// `has_pmu = false` MUST omit the pmu node entirely. Pinning the
    /// omission proves the no-PMU-host gating in `create_fdt` honors
    /// the threaded boolean: on a host where KVM masks
    /// KVM_ARM_VCPU_PMU_V3 out of vcpu_init, advertising the FDT node
    /// would have the guest pmuv3 driver attach to the PMU PPI (PPI 7,
    /// global intid 23), fail to receive any events, and log a noisy
    /// attach failure. The test
    /// drives `create_fdt` directly with `has_pmu=false` because the
    /// `test_fdt` shim hard-codes `true`.
    #[test]
    fn fdt_no_pmu_node_when_has_pmu_false() {
        let topo = default_topo();
        let mpidrs = fake_mpidrs(topo.total_cpus());
        let layout = test_layout(&topo, 256);
        let dtb = create_fdt(
            &topo,
            &mpidrs,
            256,
            "console=ttyS0",
            None,
            None,
            false,
            &layout,
            false,
            false,
            false,
        )
        .expect("create_fdt with has_pmu=false must still produce a valid DTB");
        let props = parse_dtb_props(&dtb);

        let pmu_compat = props
            .iter()
            .find(|(n, p, _)| n == "pmu" && p == "compatible");
        assert!(
            pmu_compat.is_none(),
            "pmu node must be absent when has_pmu=false; found compatible={:?}",
            pmu_compat,
        );
        let pmu_intr = props
            .iter()
            .find(|(n, p, _)| n == "pmu" && p == "interrupts");
        assert!(
            pmu_intr.is_none(),
            "pmu interrupts property must be absent when has_pmu=false; found={:?}",
            pmu_intr,
        );
    }

    /// FDT `interrupts` cell[1] for the pmu node is the bare PPI value
    /// (PMU_PPI = 7), NOT the global intid (PMU_INTID = 23). Pin this
    /// distinction explicitly so a refactor that conflates the two
    /// namespaces (e.g. writing PMU_INTID into the FDT cell) trips
    /// here even if the per-vCPU init still works correctly.
    #[test]
    fn fdt_pmu_interrupt_cell_is_ppi_not_intid() {
        use crate::vmm::aarch64::kvm::PMU_INTID;
        let topo = default_topo();
        let mpidrs = fake_mpidrs(topo.total_cpus());
        let dtb = test_fdt(&topo, &mpidrs, 256, "console=ttyS0", None, None, false).unwrap();
        let props = parse_dtb_props(&dtb);
        let irq = prop_u32_array(&props, "pmu", "interrupts").expect("pmu interrupts must exist");
        assert_eq!(
            irq[1], PMU_PPI,
            "FDT pmu interrupts cell[1] must carry PMU_PPI (PPI namespace), \
             not PMU_INTID (global intid namespace)",
        );
        assert_ne!(
            irq[1], PMU_INTID,
            "FDT pmu interrupts cell[1] must NOT be the global intid value — \
             KVM_ARM_VCPU_PMU_V3_IRQ takes the intid; the FDT takes the PPI",
        );
    }
}
