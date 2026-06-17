/// ACPI 2.0 table generation for SMP topology via zerocopy packed structs.
///
/// Generates RSDP rev 2 -> XSDT/RSDT -> {FADT, MADT, SRAT, SLIT[, HMAT]},
/// with FADT referencing DSDT. RSDT with 32-bit pointers coexists with
/// XSDT as an ACPI 1.0 fallback. FADT rev 6 with legacy hardware
/// (PIC, PIT, ISA serial). Per-CPU APIC type: Local APIC (type 0) for
/// apic_id < 255, x2APIC (type 9) for apic_id >= 255.
///
/// HMAT (Heterogeneous Memory Attribute Table) is emitted for all
/// multi-NUMA topologies. It provides latency and bandwidth attributes
/// that the kernel uses to compute abstract distance (adistance) for
/// memory tiering and NUMA optimization.
use anyhow::{Context, Result, ensure};
use vm_memory::{Bytes, GuestAddress, GuestMemoryMmap};
use zerocopy::IntoBytes;

use super::topology::apic_id;
use crate::vmm::kvm::HIMEM_START;
use crate::vmm::numa_mem::NumaMemoryLayout;
use crate::vmm::topology::Topology;

// RSDP at fixed address in BIOS ROM area — firmware scans for it here.
const RSDP_ADDR: u64 = 0x000E_0000;
const RSDP_SIZE: u64 = 36;

/// Addresses and sizes of all ACPI tables after dynamic placement.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct AcpiLayout {
    pub dsdt_addr: u64,
    pub dsdt_size: u64,
    pub madt_addr: u64,
    pub madt_size: u64,
    pub fadt_addr: u64,
    pub fadt_size: u64,
    pub srat_addr: u64,
    pub srat_size: u64,
    pub slit_addr: u64,
    pub slit_size: u64,
    pub hmat_addr: u64,
    pub hmat_size: u64,
    pub rsdt_addr: u64,
    pub rsdt_size: u64,
    pub xsdt_addr: u64,
    pub xsdt_size: u64,
    pub rsdp_addr: u64,
    pub rsdp_size: u64,
}

// FADT flags
const FADT_F_PWR_BUTTON: u32 = 1 << 4;
const FADT_F_SLP_BUTTON: u32 = 1 << 5;

// IOAPIC
const IOAPIC_ADDR: u32 = 0xFEC0_0000;
const IOAPIC_ID: u8 = 0;

// Local APIC
const LAPIC_ADDR: u32 = 0xFEE0_0000;

// ---------------------------------------------------------------------------
// Packed structs — field offsets verified by zerocopy at compile time
// ---------------------------------------------------------------------------

/// ACPI SDT header (36 bytes). Shared prefix for DSDT, RSDT, XSDT, FADT,
/// MADT, SRAT, SLIT, HMAT.
#[repr(C, packed)]
#[derive(Clone, Copy, Default, IntoBytes, zerocopy::Immutable, zerocopy::KnownLayout)]
struct SdtHeader {
    signature: [u8; 4],
    length: u32,
    revision: u8,
    checksum: u8,
    oem_id: [u8; 6],
    oem_table_id: [u8; 8],
    oem_revision: u32,
    creator_id: [u8; 4],
    creator_revision: u32,
}

impl SdtHeader {
    fn new(sig: &[u8; 4], length: u32, revision: u8) -> Self {
        Self {
            signature: *sig,
            length,
            revision,
            oem_id: *b"KTSTR\0",
            oem_table_id: {
                let mut id = [0u8; 8];
                let prefix = b"KTSR";
                id[..prefix.len()].copy_from_slice(prefix);
                id[prefix.len()..prefix.len() + sig.len()].copy_from_slice(sig);
                id
            },
            oem_revision: 1,
            // 4-byte creator_id field; KTSR is the 4-byte prefix of KTSTR.
            creator_id: *b"KTSR",
            creator_revision: 1,
            ..Default::default()
        }
    }
}

/// RSDP rev 2 (36 bytes).
#[repr(C, packed)]
#[derive(Clone, Copy, Default, IntoBytes, zerocopy::Immutable, zerocopy::KnownLayout)]
struct Rsdp {
    signature: [u8; 8],
    checksum: u8,
    oem_id: [u8; 6],
    revision: u8,
    rsdt_address: u32,
    length: u32,
    xsdt_address: u64,
    extended_checksum: u8,
    _reserved: [u8; 3],
}

/// MADT header (44 bytes = 36 SDT + 8 MADT-specific).
#[repr(C, packed)]
#[derive(Clone, Copy, Default, IntoBytes, zerocopy::Immutable, zerocopy::KnownLayout)]
struct MadtHeader {
    sdt: SdtHeader,
    local_apic_address: u32,
    flags: u32,
}

/// Local APIC entry (type 0, 8 bytes).
#[repr(C, packed)]
#[derive(Clone, Copy, Default, IntoBytes, zerocopy::Immutable, zerocopy::KnownLayout)]
struct MadtLocalApic {
    entry_type: u8,
    length: u8,
    processor_id: u8,
    apic_id: u8,
    flags: u32,
}

/// x2APIC entry (type 9, 16 bytes).
#[repr(C, packed)]
#[derive(Clone, Copy, Default, IntoBytes, zerocopy::Immutable, zerocopy::KnownLayout)]
struct MadtX2Apic {
    entry_type: u8,
    length: u8,
    _reserved: u16,
    x2apic_id: u32,
    flags: u32,
    processor_uid: u32,
}

/// IOAPIC entry (type 1, 12 bytes).
#[repr(C, packed)]
#[derive(Clone, Copy, Default, IntoBytes, zerocopy::Immutable, zerocopy::KnownLayout)]
struct MadtIoApic {
    entry_type: u8,
    length: u8,
    io_apic_id: u8,
    _reserved: u8,
    io_apic_address: u32,
    gsi_base: u32,
}

/// Interrupt Source Override (type 2, 10 bytes).
#[repr(C, packed)]
#[derive(Clone, Copy, Default, IntoBytes, zerocopy::Immutable, zerocopy::KnownLayout)]
struct MadtIso {
    entry_type: u8,
    length: u8,
    bus: u8,
    source: u8,
    gsi: u32,
    flags: u16,
}

/// Local APIC NMI (type 4, 6 bytes).
#[repr(C, packed)]
#[derive(Clone, Copy, Default, IntoBytes, zerocopy::Immutable, zerocopy::KnownLayout)]
struct MadtLapicNmi {
    entry_type: u8,
    length: u8,
    processor_id: u8,
    flags: u16,
    lint: u8,
}

/// x2APIC NMI (type 0x0A, 12 bytes).
#[repr(C, packed)]
#[derive(Clone, Copy, Default, IntoBytes, zerocopy::Immutable, zerocopy::KnownLayout)]
struct MadtX2ApicNmi {
    entry_type: u8,
    length: u8,
    flags: u16,
    processor_uid: u32,
    lint: u8,
    _reserved: [u8; 3],
}

/// SRAT CPU affinity: ProcessorLocalX2ApicAffinity (type 2, 24 bytes).
#[repr(C, packed)]
#[derive(Clone, Copy, Default, IntoBytes, zerocopy::Immutable, zerocopy::KnownLayout)]
struct SratCpuAffinity {
    entry_type: u8,
    length: u8,
    _reserved0: u16,
    proximity_domain: u32,
    x2apic_id: u32,
    flags: u32,
    clock_domain: u32,
    _reserved1: u32,
}

/// SRAT memory affinity (type 1, 40 bytes).
///
/// Layout matches kernel `acpi_srat_mem_affinity` (actbl3.h).
/// Proximity domain is a contiguous u32 at offset 2 for SRAT rev >= 2.
#[repr(C, packed)]
#[derive(Clone, Copy, Default, IntoBytes, zerocopy::Immutable, zerocopy::KnownLayout)]
struct SratMemAffinity {
    entry_type: u8,
    length: u8,
    proximity_domain: u32,
    _reserved0: u16,
    base_address: u64,
    address_length: u64,
    _reserved1: u32,
    flags: u32,
    _reserved2: u64,
}

// ---------------------------------------------------------------------------
// Checksum
// ---------------------------------------------------------------------------

fn acpi_checksum(data: &[u8]) -> u8 {
    let sum: u8 = data.iter().fold(0u8, |acc, &b| acc.wrapping_add(b));
    (!sum).wrapping_add(1)
}

/// Apply checksum to a byte buffer at the SDT header checksum offset (byte 9).
fn set_sdt_checksum(buf: &mut [u8]) {
    buf[9] = 0;
    buf[9] = acpi_checksum(buf);
}

// ---------------------------------------------------------------------------
// Table generation
// ---------------------------------------------------------------------------

/// Write ACPI tables to guest memory.
///
/// RSDP is at fixed address 0xE0000; remaining tables pack contiguously
/// after it in order: DSDT, MADT, FADT, SRAT, SLIT, HMAT, RSDT, XSDT.
///
/// SRAT memory affinity uses `NumaMemoryLayout` regions directly,
/// ensuring GPA ranges match KVM memory slots exactly. HMAT is emitted
/// for all multi-NUMA topologies.
pub fn setup_acpi(
    mem: &GuestMemoryMmap,
    topo: &Topology,
    numa_layout: &NumaMemoryLayout,
) -> Result<AcpiLayout> {
    let num_cpus = topo.total_cpus();

    let emit_hmat = topo.numa_nodes > 1;

    // The ACPI tables pack into the legacy ISA hole
    // [RSDP_ADDR + RSDP_SIZE, HIMEM_START). Reject any topology whose
    // tables would overflow it BEFORE the u32 per-table size math, so
    // neither the O(num_cpus) SRAT/MADT terms nor the O(numa_nodes²)
    // SLIT/HMAT terms can wrap u32 (a wrapped size defeats the precise
    // end-of-pack check below and lets a write_* scribble past the hole
    // into guest RAM). setup_acpi is a pub fn: a direct caller can pass a
    // topology the kvm.rs max_vcpus gate would reject, so the guards live
    // here, not only at VM build.
    //
    // The cap is the correct ceiling, not merely a guard: the guest
    // kernel's MAX_NUMNODES = 1 << CONFIG_NODES_SHIFT = 64 (x86_64 default
    // 6; ktstr does not override it), and its SRAT parser drops nodes
    // >= MAX_NUMNODES, so a guest cannot use more NUMA nodes than the
    // ~130 KiB hole trivially fits (a 64-node SLIT is 4 KiB). If
    // CONFIG_NODES_SHIFT is ever raised toward the x86 max of 10
    // (MAX_NUMNODES 1024, a 1 MiB SLIT), revisit: the hole would no longer
    // fit a usable node count and the tables would need relocating to a
    // high reserved region.
    let acpi_table_space = HIMEM_START - (RSDP_ADDR + RSDP_SIZE);

    // CPU axis: SRAT has one CpuAffinity per vCPU and MADT one APIC entry
    // per vCPU, both O(num_cpus). Bound it in u64 so the u32 sizing below
    // cannot wrap.
    let cpu_table_bytes = num_cpus as u64
        * (std::mem::size_of::<SratCpuAffinity>() + std::mem::size_of::<MadtX2Apic>()) as u64;
    ensure!(
        cpu_table_bytes <= acpi_table_space,
        "ACPI CPU tables (SRAT + MADT, {cpu_table_bytes} bytes for {num_cpus} \
         vCPUs) exceed the ISA hole table space ({acpi_table_space} bytes) — \
         would clobber guest RAM",
    );

    // Compute table sizes.
    let dsdt_size: u64 = 36;

    let madt_size = compute_madt_size(topo) as u64;

    let fadt_size: u64 = 276;

    // SRAT: one CPU affinity per vCPU + one memory affinity per layout
    // region (nodes with memory). CPU-only nodes with zero memory do not
    // get a memory affinity entry.
    let num_mem_regions = numa_layout.regions().len() as u64;
    let srat_size: u64 = 48
        + std::mem::size_of::<SratCpuAffinity>() as u64 * num_cpus as u64
        + std::mem::size_of::<SratMemAffinity>() as u64 * num_mem_regions;

    // SLIT: NxN distance matrix where N = NUMA node count.
    let n = topo.numa_nodes as u64;
    let slit_size: u64 = 36 + 8 + n * n;

    // NUMA axis: SLIT is O(numa_nodes²) and dominates; HMAT is also
    // O(numa_nodes²). Reject when the SLIT alone cannot fit, which also
    // bounds numa_nodes so compute_hmat_size's u32 math below cannot wrap.
    ensure!(
        slit_size <= acpi_table_space,
        "ACPI SLIT alone ({slit_size} bytes, numa_nodes={}, O(n²)) exceeds the \
         ISA hole table space ({acpi_table_space} bytes [{:#x}, {HIMEM_START:#x})) \
         — would clobber guest RAM",
        topo.numa_nodes,
        RSDP_ADDR + RSDP_SIZE
    );

    let hmat_size: u64 = if emit_hmat {
        compute_hmat_size(topo, numa_layout) as u64
    } else {
        0
    };

    // Table count: FADT + MADT + SRAT + SLIT + optional HMAT.
    let table_count: u64 = if emit_hmat { 5 } else { 4 };
    let rsdt_size: u64 = 36 + table_count * 4;
    let xsdt_size: u64 = 36 + table_count * 8;

    // Pack tables contiguously after RSDP.
    let mut cursor = RSDP_ADDR + RSDP_SIZE;

    let dsdt_addr = cursor;
    cursor += dsdt_size;

    let madt_addr = cursor;
    cursor += madt_size;

    let fadt_addr = cursor;
    cursor += fadt_size;

    let srat_addr = cursor;
    cursor += srat_size;

    let slit_addr = cursor;
    cursor += slit_size;

    let hmat_addr = cursor;
    cursor += hmat_size;

    let rsdt_addr = cursor;
    cursor += rsdt_size;

    let xsdt_addr = cursor;
    let acpi_end = xsdt_addr + xsdt_size;

    // Precise fit: the full contiguous pack must end within the ISA hole;
    // above HIMEM_START is guest RAM (E820_RAM) the guest allocates over.
    // The CPU and SLIT pre-checks above bound num_cpus and numa_nodes, so
    // every per-table u32 size summed here is wrap-safe.
    ensure!(
        acpi_end <= HIMEM_START,
        "ACPI tables overflow the ISA hole [{RSDP_ADDR:#x}, {HIMEM_START:#x}): \
         end {acpi_end:#x} ({} bytes) for numa_nodes={}, cpus={} — would clobber \
         guest RAM",
        acpi_end - RSDP_ADDR,
        topo.numa_nodes,
        num_cpus
    );

    let layout = AcpiLayout {
        dsdt_addr,
        dsdt_size,
        madt_addr,
        madt_size,
        fadt_addr,
        fadt_size,
        srat_addr,
        srat_size,
        slit_addr,
        slit_size,
        hmat_addr,
        hmat_size,
        rsdt_addr,
        rsdt_size,
        xsdt_addr,
        xsdt_size,
        rsdp_addr: RSDP_ADDR,
        rsdp_size: RSDP_SIZE,
    };

    write_dsdt(mem, dsdt_addr)?;
    write_madt(mem, topo, madt_addr)?;
    write_fadt(mem, &layout)?;
    write_srat(mem, topo, numa_layout, srat_addr)?;
    write_slit(mem, topo, slit_addr)?;
    if emit_hmat {
        write_hmat(mem, topo, numa_layout, hmat_addr)?;
    }
    write_rsdt(mem, &layout)?;
    write_xsdt(mem, &layout)?;
    write_rsdp(mem, &layout)?;
    Ok(layout)
}

fn write_rsdp(mem: &GuestMemoryMmap, layout: &AcpiLayout) -> Result<()> {
    let mut rsdp = Rsdp {
        signature: *b"RSD PTR ",
        oem_id: *b"KTSTR\0",
        revision: 2,
        rsdt_address: layout.rsdt_addr as u32,
        length: 36,
        xsdt_address: layout.xsdt_addr,
        ..Default::default()
    };
    rsdp.checksum = acpi_checksum(&rsdp.as_bytes()[..20]);
    rsdp.extended_checksum = acpi_checksum(rsdp.as_bytes());
    mem.write_slice(rsdp.as_bytes(), GuestAddress(RSDP_ADDR))
        .context("write RSDP")?;
    Ok(())
}

fn write_rsdt(mem: &GuestMemoryMmap, layout: &AcpiLayout) -> Result<()> {
    let len = layout.rsdt_size as usize;
    let mut buf = vec![0u8; len];
    let hdr = SdtHeader::new(b"RSDT", len as u32, 1);
    buf[..36].copy_from_slice(hdr.as_bytes());
    let mut offset = 36;
    for addr in rsdt_entries(layout) {
        buf[offset..offset + 4].copy_from_slice(&(addr as u32).to_le_bytes());
        offset += 4;
    }
    set_sdt_checksum(&mut buf);
    mem.write_slice(&buf, GuestAddress(layout.rsdt_addr))
        .context("write RSDT")?;
    Ok(())
}

fn write_xsdt(mem: &GuestMemoryMmap, layout: &AcpiLayout) -> Result<()> {
    let len = layout.xsdt_size as usize;
    let mut buf = vec![0u8; len];
    let hdr = SdtHeader::new(b"XSDT", len as u32, 1);
    buf[..36].copy_from_slice(hdr.as_bytes());
    let mut offset = 36;
    for addr in rsdt_entries(layout) {
        buf[offset..offset + 8].copy_from_slice(&addr.to_le_bytes());
        offset += 8;
    }
    set_sdt_checksum(&mut buf);
    mem.write_slice(&buf, GuestAddress(layout.xsdt_addr))
        .context("write XSDT")?;
    Ok(())
}

fn rsdt_entries(layout: &AcpiLayout) -> Vec<u64> {
    let mut entries = vec![
        layout.fadt_addr,
        layout.madt_addr,
        layout.srat_addr,
        layout.slit_addr,
    ];
    if layout.hmat_size > 0 {
        entries.push(layout.hmat_addr);
    }
    entries
}

fn write_dsdt(mem: &GuestMemoryMmap, addr: u64) -> Result<()> {
    let mut buf = vec![0u8; 36];
    let hdr = SdtHeader::new(b"DSDT", 36, 2);
    buf[..36].copy_from_slice(hdr.as_bytes());
    set_sdt_checksum(&mut buf);
    mem.write_slice(&buf, GuestAddress(addr))
        .context("write DSDT")?;
    Ok(())
}

fn write_fadt(mem: &GuestMemoryMmap, layout: &AcpiLayout) -> Result<()> {
    let mut buf = vec![0u8; 276];
    let hdr = SdtHeader::new(b"FACP", 276, 6);
    buf[..36].copy_from_slice(hdr.as_bytes());
    // DSDT pointer at offset 40 (32-bit, legacy)
    buf[40..44].copy_from_slice(&(layout.dsdt_addr as u32).to_le_bytes());
    // X_DSDT at offset 140 (64-bit)
    buf[140..148].copy_from_slice(&layout.dsdt_addr.to_le_bytes());
    let flags = FADT_F_PWR_BUTTON | FADT_F_SLP_BUTTON;
    buf[112..116].copy_from_slice(&flags.to_le_bytes());
    buf[131] = 5;
    set_sdt_checksum(&mut buf);
    mem.write_slice(&buf, GuestAddress(layout.fadt_addr))
        .context("write FADT")?;
    Ok(())
}

fn write_srat(
    mem: &GuestMemoryMmap,
    topo: &Topology,
    numa_layout: &NumaMemoryLayout,
    addr: u64,
) -> Result<()> {
    let num_cpus = topo.total_cpus();
    let num_mem_regions = numa_layout.regions().len() as u32;

    let len = 48
        + std::mem::size_of::<SratCpuAffinity>() as u32 * num_cpus
        + std::mem::size_of::<SratMemAffinity>() as u32 * num_mem_regions;
    let mut buf = vec![0u8; len as usize];

    let hdr = SdtHeader::new(b"SRAT", len, 3);
    buf[..36].copy_from_slice(hdr.as_bytes());
    buf[36..40].copy_from_slice(&1u32.to_le_bytes());

    let mut offset = 48;

    for cpu_id in 0..num_cpus {
        let (llc_id, _, _) = topo.decompose(cpu_id);
        let node_id = topo.numa_node_of(llc_id);
        let entry = SratCpuAffinity {
            entry_type: 2,
            length: std::mem::size_of::<SratCpuAffinity>() as u8,
            proximity_domain: node_id,
            x2apic_id: apic_id(topo, cpu_id),
            flags: 1,
            ..Default::default()
        };
        let bytes = entry.as_bytes();
        buf[offset..offset + bytes.len()].copy_from_slice(bytes);
        offset += bytes.len();
    }

    // Memory affinity from NumaMemoryLayout regions. The layout covers
    // the full guest DRAM.
    let regions = numa_layout.regions();
    for region in regions {
        let length = region.size;
        let entry = SratMemAffinity {
            entry_type: 1,
            length: std::mem::size_of::<SratMemAffinity>() as u8,
            proximity_domain: region.node_id,
            base_address: region.gpa_start,
            address_length: length,
            flags: 1,
            ..Default::default()
        };
        let bytes = entry.as_bytes();
        buf[offset..offset + bytes.len()].copy_from_slice(bytes);
        offset += bytes.len();
    }

    set_sdt_checksum(&mut buf);
    mem.write_slice(&buf, GuestAddress(addr))
        .context("write SRAT")?;
    Ok(())
}

fn write_slit(mem: &GuestMemoryMmap, topo: &Topology, addr: u64) -> Result<()> {
    let n = topo.numa_nodes as u64;
    let len = 36 + 8 + n * n;
    let mut buf = vec![0u8; len as usize];

    let hdr = SdtHeader::new(b"SLIT", len as u32, 1);
    buf[..36].copy_from_slice(hdr.as_bytes());
    buf[36..44].copy_from_slice(&n.to_le_bytes());
    let matrix_start = 44;
    for i in 0..n {
        for j in 0..n {
            buf[matrix_start + (i * n + j) as usize] = topo.distance(i as u32, j as u32);
        }
    }

    set_sdt_checksum(&mut buf);
    mem.write_slice(&buf, GuestAddress(addr))
        .context("write SLIT")?;
    Ok(())
}

// ---------------------------------------------------------------------------
// HMAT — Heterogeneous Memory Attribute Table (rev 2)
// ---------------------------------------------------------------------------

/// SLLBI ACCESS_LATENCY entry_base_unit: 100ns expressed in picoseconds.
/// actual_latency = entry_u16 * entry_base_unit. DRAM=1*100000ps=100ns,
/// CXL=3*100000ps=300ns.
const HMAT_LATENCY_BASE_PS: u64 = 100_000;
/// DRAM latency entry value (100ns / 100ns = 1).
const HMAT_DRAM_LATENCY_ENTRY: u16 = 1;
/// CXL latency entry value (300ns / 100ns = 3).
const HMAT_CXL_LATENCY_ENTRY: u16 = 3;

/// SLLBI ACCESS_BANDWIDTH entry_base_unit: 10 GB/s in MB/s.
/// actual_bw = entry_u16 * entry_base_unit. DRAM=5*10240=51200 MB/s=50GB/s,
/// CXL=2*10240=20480 MB/s=20GB/s.
const HMAT_BW_BASE_MBS: u64 = 10_240;
/// DRAM bandwidth entry value (50GB / 10GB = 5).
const HMAT_DRAM_BW_ENTRY: u16 = 5;
/// CXL bandwidth entry value (20GB / 10GB = 2).
const HMAT_CXL_BW_ENTRY: u16 = 2;

/// HMAT Memory Proximity Domain Attributes (type 0, 40 bytes).
#[repr(C, packed)]
#[derive(Clone, Copy, Default, IntoBytes, zerocopy::Immutable, zerocopy::KnownLayout)]
struct HmatMpda {
    hmat_type: u16,
    _reserved0: u16,
    length: u32,
    flags: u16,
    _reserved1: u16,
    initiator_proximity_domain: u32,
    memory_proximity_domain: u32,
    _reserved2: u32,
    _reserved3: u64,
    _reserved4: u64,
}

const _: () = assert!(std::mem::size_of::<HmatMpda>() == 40);

/// HMAT System Locality Latency and Bandwidth Info (type 1).
/// Variable-length: header (32 bytes) + PD lists (4 bytes each)
/// + entry matrix (2 bytes each).
#[repr(C, packed)]
#[derive(Clone, Copy, Default, IntoBytes, zerocopy::Immutable, zerocopy::KnownLayout)]
struct HmatSllbiHeader {
    hmat_type: u16,
    _reserved0: u16,
    length: u32,
    flags: u8,
    data_type: u8,
    min_transfer_size: u8,
    _reserved1: u8,
    num_initiator_pds: u32,
    num_target_pds: u32,
    _reserved2: u32,
    entry_base_unit: u64,
}

const _: () = assert!(std::mem::size_of::<HmatSllbiHeader>() == 32);

/// HMAT Memory Side Cache Information Structure (type 2, 32 bytes).
///
/// Matches kernel `acpi_hmat_cache` (actbl1.h). `cache_attributes`
/// packs line_size in bits \[31:16\], associativity in \[11:8\], write
/// policy in \[15:12\], cache level in \[7:4\], total levels in \[3:0\].
#[repr(C, packed)]
#[derive(Clone, Copy, Default, IntoBytes, zerocopy::Immutable, zerocopy::KnownLayout)]
struct HmatMsci {
    hmat_type: u16,
    _reserved0: u16,
    length: u32,
    memory_proximity_domain: u32,
    _reserved1: u32,
    cache_size: u64,
    /// Bits `[3:0]` = total cache levels,
    /// bits `[7:4]` = cache level being described,
    /// bits `[11:8]` = associativity (0=none,1=direct,2=complex),
    /// bits `[15:12]` = write policy (0=none,1=WB,2=WT),
    /// bits `[31:16]` = cache line size in bytes.
    cache_attributes: u32,
    address_mode: u16,
    _num_smbios_handles: u16,
}

const _: () = assert!(std::mem::size_of::<HmatMsci>() == 32);

/// HMAT targets are per-proximity-domain (per NUMA node), NOT per
/// memory range. A node split across the MMIO gap yields two
/// `NodeRegion`s with the same `node_id`; emitting one HMAT target per
/// region would duplicate the proximity domain (malformed HMAT). Dedup
/// to the distinct `node_id`s, preserving first-seen (ascending-GPA)
/// order. (SRAT, by contrast, is correctly per-range — two
/// memory-affinity entries with the same domain are valid and expected,
/// confirmed against the guest's acpi_parse_memory_affinity.)
fn hmat_target_nodes(numa_layout: &NumaMemoryLayout) -> Vec<u32> {
    let mut nodes = Vec::new();
    for region in numa_layout.regions() {
        if !nodes.contains(&region.node_id) {
            nodes.push(region.node_id);
        }
    }
    nodes
}

fn compute_hmat_size(topo: &Topology, numa_layout: &NumaMemoryLayout) -> u32 {
    let num_initiators = topo.cpu_bearing_nodes();
    let num_targets = hmat_target_nodes(numa_layout).len() as u32;
    let num_mpdas = num_targets;

    // SDT header (36) + 4 reserved bytes = 40.
    let hmat_header = 40u32;

    let mpda_size = std::mem::size_of::<HmatMpda>() as u32 * num_mpdas;

    // Two SLLBI subtables: access_latency and access_bandwidth.
    let sllbi_size = (std::mem::size_of::<HmatSllbiHeader>() as u32
        + 4 * num_initiators
        + 4 * num_targets
        + 2 * num_initiators * num_targets)
        * 2;

    // Type 2: one MSCI per node that has a mem_side_cache.
    let num_msci = topo
        .nodes
        .map(|nodes| nodes.iter().filter(|n| n.mem_side_cache.is_some()).count() as u32)
        .unwrap_or(0);
    let msci_size = std::mem::size_of::<HmatMsci>() as u32 * num_msci;

    hmat_header + mpda_size + sllbi_size + msci_size
}

fn write_hmat(
    mem: &GuestMemoryMmap,
    topo: &Topology,
    numa_layout: &NumaMemoryLayout,
    addr: u64,
) -> Result<()> {
    let len = compute_hmat_size(topo, numa_layout);
    let mut buf = vec![0u8; len as usize];

    // SDT header: HMAT revision 2.
    let hdr = SdtHeader::new(b"HMAT", len, 2);
    buf[..36].copy_from_slice(hdr.as_bytes());
    // 4 reserved bytes at offset 36..40 (already zero).

    let mut offset = 40usize;

    // Collect initiator and target node IDs.
    let initiators: Vec<u32> = (0..topo.numa_nodes)
        .filter(|&n| topo.llcs_in_node(n) > 0)
        .collect();
    let targets: Vec<u32> = hmat_target_nodes(numa_layout);

    // Type 0: MPDA — one per memory target, mapping to its closest
    // CPU-bearing initiator. flags=3: bit 0 (PROCESSOR_PD_VALID) +
    // bit 1 (MEMORY_PD_VALID). Without both bits, hmat_update_target()
    // skips the entry and SLLBI perf data is never associated.
    for &target_node in &targets {
        let initiator = if topo.llcs_in_node(target_node) > 0 {
            target_node
        } else {
            *initiators
                .iter()
                .min_by_key(|&&i| topo.distance(i, target_node))
                .unwrap_or(&initiators[0])
        };
        let mpda = HmatMpda {
            hmat_type: 0,
            length: std::mem::size_of::<HmatMpda>() as u32,
            flags: 3,
            initiator_proximity_domain: initiator,
            memory_proximity_domain: target_node,
            ..Default::default()
        };
        let bytes = mpda.as_bytes();
        buf[offset..offset + bytes.len()].copy_from_slice(bytes);
        offset += bytes.len();
    }

    // Type 1: SLLBI — latency (data_type=0) and bandwidth (data_type=3).
    // actual_value = entry_u16 * entry_base_unit.
    //
    // Cross-node latency scales by SLIT distance ratio: for initiator I
    // accessing target T, latency = target_base_latency * (distance(I,T) / 10).
    // Cross-node bandwidth scales inversely: bw = target_base_bw * (10 / distance(I,T)).
    let ni = initiators.len() as u32;
    let nt = targets.len() as u32;

    for (data_type, base_unit) in [(0u8, HMAT_LATENCY_BASE_PS), (3u8, HMAT_BW_BASE_MBS)] {
        let sllbi_len =
            std::mem::size_of::<HmatSllbiHeader>() as u32 + 4 * ni + 4 * nt + 2 * ni * nt;

        let sllbi_hdr = HmatSllbiHeader {
            hmat_type: 1,
            length: sllbi_len,
            flags: 0,
            data_type,
            min_transfer_size: 0,
            num_initiator_pds: ni,
            num_target_pds: nt,
            entry_base_unit: base_unit,
            ..Default::default()
        };
        let bytes = sllbi_hdr.as_bytes();
        buf[offset..offset + bytes.len()].copy_from_slice(bytes);
        offset += bytes.len();

        for &i in &initiators {
            buf[offset..offset + 4].copy_from_slice(&i.to_le_bytes());
            offset += 4;
        }

        for &t in &targets {
            buf[offset..offset + 4].copy_from_slice(&t.to_le_bytes());
            offset += 4;
        }

        for &init_node in &initiators {
            for &tgt_node in &targets {
                let is_cxl = topo
                    .nodes
                    .is_some_and(|nodes| nodes[tgt_node as usize].is_memory_only());

                let entry = if data_type == 0 {
                    // Latency: base value from per-node config or defaults.
                    let base = topo
                        .nodes
                        .and_then(|nodes| nodes[tgt_node as usize].latency_ns)
                        .map(|ns| {
                            let ps = ns as u64 * 1000;
                            (ps / base_unit).max(1) as u16
                        })
                        .unwrap_or(if is_cxl {
                            HMAT_CXL_LATENCY_ENTRY
                        } else {
                            HMAT_DRAM_LATENCY_ENTRY
                        });
                    let dist = topo.distance(init_node, tgt_node) as u32;
                    ((base as u32 * dist / 10) as u16).max(1)
                } else {
                    // Bandwidth: base value from per-node config or defaults.
                    let base = topo
                        .nodes
                        .and_then(|nodes| nodes[tgt_node as usize].bandwidth_mbs)
                        .map(|mbs| (mbs as u64 / base_unit).max(1) as u16)
                        .unwrap_or(if is_cxl {
                            HMAT_CXL_BW_ENTRY
                        } else {
                            HMAT_DRAM_BW_ENTRY
                        });
                    let dist = topo.distance(init_node, tgt_node) as u32;
                    ((base as u32 * 10 / dist.max(1)) as u16).max(1)
                };
                buf[offset..offset + 2].copy_from_slice(&entry.to_le_bytes());
                offset += 2;
            }
        }
    }

    // Type 2: Memory Side Cache Information — one per node with cache.
    if let Some(nodes) = topo.nodes {
        for (i, node) in nodes.iter().enumerate() {
            if let Some(cache) = &node.mem_side_cache {
                let attrs: u32 = 1 // total_cache_levels = 1
                    | (1 << 4) // cache_level = 1
                    | ((cache.associativity as u32 & 0xF) << 8)
                    | ((cache.write_policy as u32 & 0xF) << 12)
                    | ((cache.line_size as u32) << 16);
                let msci = HmatMsci {
                    hmat_type: 2,
                    length: std::mem::size_of::<HmatMsci>() as u32,
                    memory_proximity_domain: i as u32,
                    cache_size: cache.size,
                    cache_attributes: attrs,
                    ..Default::default()
                };
                let bytes = msci.as_bytes();
                buf[offset..offset + bytes.len()].copy_from_slice(bytes);
                offset += bytes.len();
            }
        }
    }

    debug_assert_eq!(offset, len as usize);

    set_sdt_checksum(&mut buf);
    mem.write_slice(&buf, GuestAddress(addr))
        .context("write HMAT")?;
    Ok(())
}

/// Determine whether a given APIC ID should use x2APIC (type 9) or
/// Local APIC (type 0). APIC ID < 255 uses type 0, >= 255 uses type 9.
fn use_x2apic_entry(apic_id: u32) -> bool {
    // 255 = MAX_XAPIC_ID + 1; matches the split_irqchip threshold in kvm.rs.
    apic_id >= 255
}

/// Compute MADT total size for a given topology.
fn compute_madt_size(topo: &Topology) -> u32 {
    let num_cpus = topo.total_cpus();
    let mut cpu_entries_size: u32 = 0;
    let mut has_x2apic = false;
    let mut has_lapic = false;
    for cpu_id in 0..num_cpus {
        if use_x2apic_entry(apic_id(topo, cpu_id)) {
            cpu_entries_size += std::mem::size_of::<MadtX2Apic>() as u32;
            has_x2apic = true;
        } else {
            cpu_entries_size += std::mem::size_of::<MadtLocalApic>() as u32;
            has_lapic = true;
        }
    }
    let nmi_size: u32 = if has_lapic {
        std::mem::size_of::<MadtLapicNmi>() as u32
    } else {
        0
    } + if has_x2apic {
        std::mem::size_of::<MadtX2ApicNmi>() as u32
    } else {
        0
    };
    std::mem::size_of::<MadtHeader>() as u32
        + cpu_entries_size
        + std::mem::size_of::<MadtIoApic>() as u32
        + std::mem::size_of::<MadtIso>() as u32
        + nmi_size
}

fn write_madt(mem: &GuestMemoryMmap, topo: &Topology, addr: u64) -> Result<()> {
    let num_cpus = topo.total_cpus();

    let mut has_x2apic = false;
    let mut has_lapic = false;
    for cpu_id in 0..num_cpus {
        if use_x2apic_entry(apic_id(topo, cpu_id)) {
            has_x2apic = true;
        } else {
            has_lapic = true;
        }
    }

    let len = compute_madt_size(topo);
    let mut buf = vec![0u8; len as usize];

    // MADT header
    let hdr = MadtHeader {
        sdt: SdtHeader::new(b"APIC", len, 3),
        local_apic_address: LAPIC_ADDR,
        flags: 1, // PCAT_COMPAT
    };
    buf[..std::mem::size_of::<MadtHeader>()].copy_from_slice(hdr.as_bytes());

    let mut offset = std::mem::size_of::<MadtHeader>();

    // CPU entries
    for cpu_id in 0..num_cpus {
        let id = apic_id(topo, cpu_id);
        if use_x2apic_entry(id) {
            let entry = MadtX2Apic {
                entry_type: 9,
                length: std::mem::size_of::<MadtX2Apic>() as u8,
                x2apic_id: id,
                flags: 1,
                processor_uid: cpu_id,
                ..Default::default()
            };
            let bytes = entry.as_bytes();
            buf[offset..offset + bytes.len()].copy_from_slice(bytes);
            offset += bytes.len();
        } else {
            let entry = MadtLocalApic {
                entry_type: 0,
                length: std::mem::size_of::<MadtLocalApic>() as u8,
                processor_id: cpu_id as u8,
                apic_id: id as u8,
                flags: 1,
            };
            let bytes = entry.as_bytes();
            buf[offset..offset + bytes.len()].copy_from_slice(bytes);
            offset += bytes.len();
        }
    }

    // IOAPIC
    let ioapic = MadtIoApic {
        entry_type: 1,
        length: std::mem::size_of::<MadtIoApic>() as u8,
        io_apic_id: IOAPIC_ID,
        io_apic_address: IOAPIC_ADDR,
        gsi_base: 0,
        ..Default::default()
    };
    let bytes = ioapic.as_bytes();
    buf[offset..offset + bytes.len()].copy_from_slice(bytes);
    offset += bytes.len();

    // Interrupt Source Override: IRQ0 -> GSI 2
    let iso = MadtIso {
        entry_type: 2,
        length: std::mem::size_of::<MadtIso>() as u8,
        bus: 0,
        source: 0,
        gsi: 2,
        flags: 0,
    };
    let bytes = iso.as_bytes();
    buf[offset..offset + bytes.len()].copy_from_slice(bytes);
    offset += bytes.len();

    // NMI entries
    if has_lapic {
        let nmi = MadtLapicNmi {
            entry_type: 4,
            length: std::mem::size_of::<MadtLapicNmi>() as u8,
            processor_id: 0xFF,
            flags: 0,
            lint: 1,
        };
        let bytes = nmi.as_bytes();
        buf[offset..offset + bytes.len()].copy_from_slice(bytes);
        offset += bytes.len();
    }
    if has_x2apic {
        let nmi = MadtX2ApicNmi {
            entry_type: 0x0A,
            length: std::mem::size_of::<MadtX2ApicNmi>() as u8,
            flags: 0,
            processor_uid: 0xFFFF_FFFF,
            lint: 1,
            _reserved: [0; 3],
        };
        let bytes = nmi.as_bytes();
        buf[offset..offset + bytes.len()].copy_from_slice(bytes);
    }

    set_sdt_checksum(&mut buf);
    mem.write_slice(&buf, GuestAddress(addr))
        .context("write MADT")?;
    Ok(())
}

#[cfg(test)]
mod tests;
