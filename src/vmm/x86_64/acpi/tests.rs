use super::*;

fn test_mem(mib: u32) -> GuestMemoryMmap {
    GuestMemoryMmap::<()>::from_ranges(&[(GuestAddress(0), (mib as usize) << 20)]).unwrap()
}

fn test_layout(topo: &Topology, mib: u32) -> NumaMemoryLayout {
    NumaMemoryLayout::compute(topo, mib, 0, None).unwrap()
}

fn test_setup(mem: &GuestMemoryMmap, topo: &Topology, mib: u32) -> AcpiLayout {
    let layout = test_layout(topo, mib);
    setup_acpi(mem, topo, &layout, false, 0).unwrap()
}

fn read_table(mem: &GuestMemoryMmap, addr: u64) -> Vec<u8> {
    let mut len_bytes = [0u8; 4];
    mem.read_slice(&mut len_bytes, GuestAddress(addr + 4))
        .unwrap();
    let len = u32::from_le_bytes(len_bytes) as usize;
    let mut buf = vec![0u8; len];
    mem.read_slice(&mut buf, GuestAddress(addr)).unwrap();
    buf
}

fn read_madt(mem: &GuestMemoryMmap, layout: &AcpiLayout) -> Vec<u8> {
    read_table(mem, layout.madt_addr)
}

fn walk_madt_entries(madt: &[u8]) -> Vec<(u8, u8, &[u8])> {
    let hdr_size = std::mem::size_of::<MadtHeader>();
    let mut entries = Vec::new();
    let mut offset = hdr_size;
    while offset < madt.len() {
        let entry_type = madt[offset];
        let entry_len = madt[offset + 1];
        entries.push((
            entry_type,
            entry_len,
            &madt[offset..offset + entry_len as usize],
        ));
        offset += entry_len as usize;
    }
    entries
}

// -- Struct size compile-time assertions --
const _: () = assert!(std::mem::size_of::<SdtHeader>() == 36);
const _: () = assert!(std::mem::size_of::<Rsdp>() == 36);
const _: () = assert!(std::mem::size_of::<MadtHeader>() == 44);
const _: () = assert!(std::mem::size_of::<MadtLocalApic>() == 8);
const _: () = assert!(std::mem::size_of::<MadtX2Apic>() == 16);
const _: () = assert!(std::mem::size_of::<MadtIoApic>() == 12);
const _: () = assert!(std::mem::size_of::<MadtIso>() == 10);
const _: () = assert!(std::mem::size_of::<MadtLapicNmi>() == 6);
const _: () = assert!(std::mem::size_of::<MadtX2ApicNmi>() == 12);
const _: () = assert!(std::mem::size_of::<SratCpuAffinity>() == 24);
const _: () = assert!(std::mem::size_of::<SratMemAffinity>() == 40);

#[test]
fn acpi_rejects_slit_dominated_overflow_at_extreme_numa() {
    // SLIT is O(numa_nodes²); 512 nodes → ~256 KiB, far past the ~128 KiB
    // ISA-hole table space. The early u64 SLIT check must reject it
    // loudly (and keep the u32 HMAT size math from wrapping).
    let mem = test_mem(16);
    let topo = Topology {
        llcs: 1,
        cores_per_llc: 1,
        threads_per_core: 1,
        numa_nodes: 512,
        nodes: None,
        distances: None,
    };
    let layout = test_layout(&topo, 512);
    let err =
        setup_acpi(&mem, &topo, &layout, false, 0).expect_err("512-NUMA SLIT must overflow the ISA hole");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("SLIT alone") && msg.contains("ISA hole"),
        "error must name the SLIT-dominated ISA-hole overflow, got: {msg}"
    );
}

#[test]
fn acpi_rejects_cpu_table_overflow_at_extreme_vcpu_count() {
    // SRAT + MADT scale O(num_cpus) and are summed against the ISA hole;
    // a topology with enough vCPUs to overflow it must be rejected loudly
    // (and before the u32 per-CPU sizing can wrap). 4096 vCPUs × ~40 B
    // exceeds the ~128 KiB hole. setup_acpi is a pub fn, so this guards
    // direct callers that bypass the VM-build max_vcpus gate.
    let mem = test_mem(16);
    let topo = Topology {
        llcs: 4096,
        cores_per_llc: 1,
        threads_per_core: 1,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    let layout = test_layout(&topo, 256);
    let err = setup_acpi(&mem, &topo, &layout, false, 0)
        .expect_err("4096-vCPU ACPI CPU tables must overflow the ISA hole");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("CPU tables") && msg.contains("ISA hole"),
        "error must name the CPU-table ISA-hole overflow, got: {msg}"
    );
}

#[test]
fn acpi_rejects_total_overflow_when_slit_alone_fits() {
    // 361 nodes: SLIT alone (~127 KiB) still fits the ~128 KiB hole, but
    // the full pack (SLIT + SRAT + HMAT + ...) exceeds it. The precise
    // end-of-tables check must catch the case the SLIT pre-check passes.
    let mem = test_mem(16);
    let topo = Topology {
        llcs: 1,
        cores_per_llc: 1,
        threads_per_core: 1,
        numa_nodes: 361,
        nodes: None,
        distances: None,
    };
    let layout = test_layout(&topo, 361);
    let err = setup_acpi(&mem, &topo, &layout, false, 0)
        .expect_err("361-NUMA total tables must overflow the ISA hole");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("tables overflow") && msg.contains("ISA hole"),
        "error must name the total-pack ISA-hole overflow, got: {msg}"
    );
}

#[test]
fn acpi_tables_fit_at_wide_smp_topology() {
    // The widest realistic scheduler-test topology (256 vCPUs, 16 NUMA)
    // must fit the ISA hole with room to spare — the overflow ceiling is
    // far above any real topology.
    let mem = test_mem(16);
    let topo = Topology {
        llcs: 16,
        cores_per_llc: 16,
        threads_per_core: 1,
        numa_nodes: 16,
        nodes: None,
        distances: None,
    };
    let l = test_setup(&mem, &topo, 4096);
    let end = l.xsdt_addr + l.xsdt_size;
    assert!(
        end <= HIMEM_START,
        "wide-SMP ACPI tables must fit the ISA hole: end={end:#x} \
         HIMEM_START={HIMEM_START:#x}"
    );
}

#[test]
fn rsdp_signature_and_checksum() {
    let mem = test_mem(16);
    let topo = Topology {
        llcs: 1,
        cores_per_llc: 1,
        threads_per_core: 1,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    let l = test_setup(&mem, &topo, 256);
    let mut rsdp = [0u8; 20];
    mem.read_slice(&mut rsdp, GuestAddress(l.rsdp_addr))
        .unwrap();
    assert_eq!(&rsdp[..8], b"RSD PTR ");
    let sum: u8 = rsdp.iter().fold(0u8, |acc, &b| acc.wrapping_add(b));
    assert_eq!(sum, 0, "RSDP checksum must be zero");
}

#[test]
fn rsdt_signature_and_checksum() {
    let mem = test_mem(16);
    let topo = Topology {
        llcs: 1,
        cores_per_llc: 1,
        threads_per_core: 1,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    let l = test_setup(&mem, &topo, 256);
    let rsdt = read_table(&mem, l.rsdt_addr);
    assert_eq!(&rsdt[..4], b"RSDT");
    let sum: u8 = rsdt.iter().fold(0u8, |acc, &b| acc.wrapping_add(b));
    assert_eq!(sum, 0, "RSDT checksum must be zero");
}

#[test]
fn madt_signature_and_checksum() {
    let mem = test_mem(16);
    let topo = Topology {
        llcs: 2,
        cores_per_llc: 2,
        threads_per_core: 1,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    let l = test_setup(&mem, &topo, 256);
    let madt = read_madt(&mem, &l);
    assert_eq!(&madt[..4], b"APIC");
    let sum: u8 = madt.iter().fold(0u8, |acc, &b| acc.wrapping_add(b));
    assert_eq!(sum, 0, "MADT checksum must be zero");
}

#[test]
fn madt_has_correct_cpu_count() {
    let mem = test_mem(16);
    let topo = Topology {
        llcs: 2,
        cores_per_llc: 4,
        threads_per_core: 2,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    let l = test_setup(&mem, &topo, 256);
    let madt = read_madt(&mem, &l);
    let entries = walk_madt_entries(&madt);
    let cpu_count = entries
        .iter()
        .filter(|(t, _, _)| *t == 0 || *t == 9)
        .count();
    assert_eq!(cpu_count, 16);
}

#[test]
fn madt_apic_ids_match_topology() {
    let mem = test_mem(16);
    let topo = Topology {
        llcs: 2,
        cores_per_llc: 2,
        threads_per_core: 1,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    let l = test_setup(&mem, &topo, 256);
    let madt = read_madt(&mem, &l);
    let entries = walk_madt_entries(&madt);
    let mut cpu_idx = 0u32;
    for (entry_type, _, data) in &entries {
        match *entry_type {
            0 => {
                assert_eq!(data[3] as u32, apic_id(&topo, cpu_idx));
                cpu_idx += 1;
            }
            9 => {
                assert_eq!(
                    u32::from_le_bytes(data[4..8].try_into().unwrap()),
                    apic_id(&topo, cpu_idx)
                );
                cpu_idx += 1;
            }
            _ => {}
        }
    }
}

#[test]
fn madt_has_ioapic() {
    let mem = test_mem(16);
    let topo = Topology {
        llcs: 1,
        cores_per_llc: 1,
        threads_per_core: 1,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    let l = test_setup(&mem, &topo, 256);
    let madt = read_madt(&mem, &l);
    let entries = walk_madt_entries(&madt);
    let ioapic = entries.iter().find(|(t, _, _)| *t == 1);
    assert!(ioapic.is_some());
    let (_, _, data) = ioapic.unwrap();
    assert_eq!(
        u32::from_le_bytes(data[4..8].try_into().unwrap()),
        IOAPIC_ADDR
    );
}

#[test]
fn rsdp_points_to_rsdt() {
    let mem = test_mem(16);
    let topo = Topology {
        llcs: 1,
        cores_per_llc: 1,
        threads_per_core: 1,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    let l = test_setup(&mem, &topo, 256);
    let mut rsdp = [0u8; 20];
    mem.read_slice(&mut rsdp, GuestAddress(l.rsdp_addr))
        .unwrap();
    assert_eq!(
        u32::from_le_bytes(rsdp[16..20].try_into().unwrap()),
        l.rsdt_addr as u32
    );
}

#[test]
fn rsdt_table_pointers() {
    let mem = test_mem(16);
    let topo = Topology {
        llcs: 1,
        cores_per_llc: 1,
        threads_per_core: 1,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    let l = test_setup(&mem, &topo, 256);
    let mut entry = [0u8; 4];
    mem.read_slice(&mut entry, GuestAddress(l.rsdt_addr + 36))
        .unwrap();
    assert_eq!(u32::from_le_bytes(entry), l.fadt_addr as u32);
    mem.read_slice(&mut entry, GuestAddress(l.rsdt_addr + 40))
        .unwrap();
    assert_eq!(u32::from_le_bytes(entry), l.madt_addr as u32);
    mem.read_slice(&mut entry, GuestAddress(l.rsdt_addr + 44))
        .unwrap();
    assert_eq!(u32::from_le_bytes(entry), l.srat_addr as u32);
    mem.read_slice(&mut entry, GuestAddress(l.rsdt_addr + 48))
        .unwrap();
    assert_eq!(u32::from_le_bytes(entry), l.slit_addr as u32);
}

/// `setup_acpi` with the PCI transport enabled (MCFG + the `_SB.PCI0` /
/// `_PRT` DSDT body), the integration path the booted-guest NIC e2es exercise
/// but no unit test covered (every other `test_setup` runs `pci_enabled=false`).
fn test_setup_pci(
    mem: &GuestMemoryMmap,
    topo: &Topology,
    mib: u32,
    nic_count: usize,
) -> AcpiLayout {
    let layout = test_layout(topo, mib);
    setup_acpi(mem, topo, &layout, true, nic_count).unwrap()
}

#[test]
fn pci_enabled_emits_mcfg_and_prt_dsdt() {
    let topo = Topology {
        llcs: 1,
        cores_per_llc: 1,
        threads_per_core: 1,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    // Non-PCI baseline: header-only DSDT, no MCFG, 4-entry RSDT.
    let mem_base = test_mem(16);
    let base = test_setup(&mem_base, &topo, 256);
    assert_eq!(base.dsdt_size, 36, "non-PCI DSDT is header-only");
    assert_eq!(base.mcfg_size, 0, "non-PCI emits no MCFG");
    let base_rsdt = read_table(&mem_base, base.rsdt_addr);
    assert_eq!(
        (base_rsdt.len() - 36) / 4,
        4,
        "non-PCI RSDT: FADT/MADT/SRAT/SLIT only",
    );

    // PCI-enabled: the DSDT carries the host-bridge AML (with _PRT), and the
    // MCFG (ECAM base) is emitted and linked into RSDT/XSDT.
    let mem = test_mem(16);
    let l = test_setup_pci(&mem, &topo, 256, 1);
    assert!(l.dsdt_size > 36, "PCI DSDT carries the host-bridge AML body");
    let dsdt = read_table(&mem, l.dsdt_addr);
    let dsdt_has = |needle: &[u8]| dsdt.windows(needle.len()).any(|w| w == needle);
    assert!(dsdt_has(b"_PRT"), "PCI DSDT contains the _PRT routing object");
    assert!(
        dsdt_has(&[0x0C, 0x41, 0xD0, 0x0A, 0x08]),
        "PCI DSDT contains the PNP0A08 host-bridge _HID",
    );

    // MCFG: signature, zero checksum, ECAM base at offset 44.
    assert!(l.mcfg_size > 0, "MCFG emitted when PCI enabled");
    let mcfg = read_table(&mem, l.mcfg_addr);
    assert_eq!(&mcfg[..4], b"MCFG");
    let sum: u8 = mcfg.iter().fold(0u8, |a, &b| a.wrapping_add(b));
    assert_eq!(sum, 0, "MCFG checksum must be zero");
    assert_eq!(
        u64::from_le_bytes(mcfg[44..52].try_into().unwrap()),
        PCI_ECAM_BASE,
        "MCFG allocation base must be the ECAM window base",
    );

    // RSDT/XSDT gain the MCFG entry (5 = FADT/MADT/SRAT/SLIT/MCFG); MCFG is
    // last (rsdt_entries appends it after the optional HMAT).
    let rsdt = read_table(&mem, l.rsdt_addr);
    let n = (rsdt.len() - 36) / 4;
    assert_eq!(n, 5, "PCI RSDT adds MCFG to the 4 base tables");
    let last = u32::from_le_bytes(rsdt[36 + (n - 1) * 4..36 + n * 4].try_into().unwrap());
    assert_eq!(last as u64, l.mcfg_addr, "MCFG is the last RSDT entry");
    let xsdt = read_table(&mem, l.xsdt_addr);
    assert_eq!((xsdt.len() - 36) / 8, 5, "PCI XSDT adds MCFG (8-byte entries)");

    // The full contiguous pack still ends within the ISA hole.
    assert!(
        l.xsdt_addr + l.xsdt_size <= HIMEM_START,
        "ACPI tables must pack within the ISA hole",
    );
}

#[test]
fn pci_with_hmat_table_count() {
    // pci + hmat (multi-NUMA) guards the table_count = 4 + hmat + pci
    // arithmetic and the rsdt_entries append order (MCFG after HMAT).
    let mem = test_mem(16);
    let topo = Topology {
        llcs: 4,
        cores_per_llc: 2,
        threads_per_core: 1,
        numa_nodes: 2,
        nodes: None,
        distances: None,
    };
    let l = test_setup_pci(&mem, &topo, 256, 1);
    assert!(l.hmat_size > 0, "HMAT emitted for multi-NUMA");
    assert!(l.mcfg_size > 0, "MCFG emitted for PCI");
    let rsdt = read_table(&mem, l.rsdt_addr);
    let n = (rsdt.len() - 36) / 4;
    assert_eq!(n, 6, "FADT/MADT/SRAT/SLIT/HMAT/MCFG");
    let last = u32::from_le_bytes(rsdt[36 + (n - 1) * 4..36 + n * 4].try_into().unwrap());
    assert_eq!(
        last as u64, l.mcfg_addr,
        "MCFG is the last RSDT entry, appended after HMAT",
    );
    assert!(
        l.xsdt_addr + l.xsdt_size <= HIMEM_START,
        "ACPI tables must pack within the ISA hole",
    );
}

/// setup_acpi with N NICs emits one DSDT `_PRT` routing package per NIC, each
/// at `(virtio_net_pci_slot(i), virtio_net_gsi(i))`. Decodes the real DSDT
/// (not the aml encoder in isolation) with the expected (slot, GSI) DERIVED
/// from the allocator, so the SCI-skip at i>=2 (NIC 2 -> GSI 10, not 9) is
/// proven to flow allocator -> setup_acpi -> DSDT. nic_count=3 exercises the
/// skip; the aml `prt_emits_one_entry_per_route` test covers the encoder.
#[test]
fn setup_acpi_emits_one_prt_route_per_nic() {
    use crate::vmm::kvm::{virtio_net_gsi, virtio_net_pci_slot};
    let topo = Topology {
        llcs: 1,
        cores_per_llc: 1,
        threads_per_core: 1,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    let mem = test_mem(16);
    let l = test_setup_pci(&mem, &topo, 256, 3);
    let dsdt = read_table(&mem, l.dsdt_addr);
    let contains = |needle: &[u8]| dsdt.windows(needle.len()).any(|w| w == needle);
    for i in 0..3usize {
        let slot = virtio_net_pci_slot(i) as u32;
        let gsi = virtio_net_gsi(i);
        // Inner routing package (mirror prt_entry_encoding): Package(4) of
        // Address DWord (slot<<16)|0xFFFF, Pin Zero, Source Zero, SourceIndex
        // GSI byte. GSI 7..23 always encodes as a Byte (0x0A prefix).
        let entry = [
            0x12, 0x0B, 0x04, // PackageOp, PkgLength, NumElements=4
            0x0C, 0xFF, 0xFF, slot as u8, 0x00, // Address DWord (slot)
            0x00, // Pin = INTA#
            0x00, // Source = static GSI routing
            0x0A, gsi as u8, // SourceIndex = GSI (Byte)
        ];
        assert!(
            contains(&entry),
            "DSDT _PRT must route NIC {i} at slot {slot} -> GSI {gsi}",
        );
    }
}

#[test]
fn madt_has_iso_irq0_gsi2() {
    let mem = test_mem(16);
    let topo = Topology {
        llcs: 1,
        cores_per_llc: 1,
        threads_per_core: 1,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    let l = test_setup(&mem, &topo, 256);
    let madt = read_madt(&mem, &l);
    let entries = walk_madt_entries(&madt);
    let iso = entries.iter().find(|(t, _, _)| *t == 2).unwrap();
    assert_eq!(iso.2[3], 0);
    assert_eq!(u32::from_le_bytes(iso.2[4..8].try_into().unwrap()), 2);
}

#[test]
fn madt_has_nmi() {
    let mem = test_mem(16);
    let topo = Topology {
        llcs: 1,
        cores_per_llc: 1,
        threads_per_core: 1,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    let l = test_setup(&mem, &topo, 256);
    let madt = read_madt(&mem, &l);
    let entries = walk_madt_entries(&madt);
    assert!(entries.iter().any(|(t, _, _)| *t == 4 || *t == 0x0A));
}

#[test]
fn small_topology_uses_lapic_entries() {
    let mem = test_mem(16);
    let topo = Topology {
        llcs: 2,
        cores_per_llc: 4,
        threads_per_core: 2,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    let l = test_setup(&mem, &topo, 256);
    let madt = read_madt(&mem, &l);
    let entries = walk_madt_entries(&madt);
    assert_eq!(entries.iter().filter(|(t, _, _)| *t == 9).count(), 0);
    assert_eq!(entries.iter().filter(|(t, _, _)| *t == 0).count(), 16);
    assert!(entries.iter().any(|(t, _, _)| *t == 4));
    assert!(!entries.iter().any(|(t, _, _)| *t == 0x0A));
}

#[test]
fn large_topology_uses_mixed_entries() {
    let mem = test_mem(16);
    let topo = Topology {
        llcs: 14,
        cores_per_llc: 9,
        threads_per_core: 2,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    let mut has_low = false;
    let mut has_high = false;
    for cpu_id in 0..topo.total_cpus() {
        let id = apic_id(&topo, cpu_id);
        if id < 255 {
            has_low = true;
        } else {
            has_high = true;
        }
    }
    assert!(has_low && has_high);
    let l = test_setup(&mem, &topo, 256);
    let madt = read_madt(&mem, &l);
    let entries = walk_madt_entries(&madt);
    let lapic_count = entries.iter().filter(|(t, _, _)| *t == 0).count();
    let x2apic_count = entries.iter().filter(|(t, _, _)| *t == 9).count();
    assert!(lapic_count > 0);
    assert!(x2apic_count > 0);
    assert_eq!(lapic_count + x2apic_count, topo.total_cpus() as usize);
    assert!(entries.iter().any(|(t, _, _)| *t == 4));
    assert!(entries.iter().any(|(t, _, _)| *t == 0x0A));
}

#[test]
fn x2apic_nmi_fields_correct() {
    let mem = test_mem(16);
    let topo = Topology {
        llcs: 14,
        cores_per_llc: 9,
        threads_per_core: 2,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    let l = test_setup(&mem, &topo, 256);
    let madt = read_madt(&mem, &l);
    let entries = walk_madt_entries(&madt);
    let (_, len, data) = entries.iter().find(|(t, _, _)| *t == 0x0A).unwrap();
    assert_eq!(*len, 12);
    assert_eq!(u16::from_le_bytes(data[2..4].try_into().unwrap()), 0);
    assert_eq!(
        u32::from_le_bytes(data[4..8].try_into().unwrap()),
        0xFFFF_FFFF
    );
    assert_eq!(data[8], 1);
}

#[test]
fn lapic_nmi_fields_correct() {
    let mem = test_mem(16);
    let topo = Topology {
        llcs: 1,
        cores_per_llc: 1,
        threads_per_core: 1,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    let l = test_setup(&mem, &topo, 256);
    let madt = read_madt(&mem, &l);
    let entries = walk_madt_entries(&madt);
    let (_, len, data) = entries.iter().find(|(t, _, _)| *t == 4).unwrap();
    assert_eq!(*len, 6);
    assert_eq!(data[2], 0xFF);
    assert_eq!(u16::from_le_bytes(data[3..5].try_into().unwrap()), 0);
    assert_eq!(data[5], 1);
}

#[test]
fn madt_checksum_representative_topologies() {
    let topos = [
        (1, 1, 1),   // degenerate single CPU
        (2, 1, 1),   // minimal multi-LLC
        (3, 3, 1),   // odd non-power-of-2
        (1, 1, 2),   // minimal SMT
        (2, 4, 2),   // standard multi-LLC with SMT
        (7, 5, 3),   // all dimensions non-power-of-2
        (15, 16, 1), // large scale no SMT
        (14, 9, 2),  // large with SMT, mixed LAPIC/x2APIC
        (2, 128, 1), // x2APIC boundary (max APIC ID = 255)
        (14, 18, 1), // large no SMT, mixed LAPIC/x2APIC
    ];
    for (llcs, cores, threads) in topos {
        let mem = test_mem(16);
        let topo = Topology {
            llcs,
            cores_per_llc: cores,
            threads_per_core: threads,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        };
        let l = test_setup(&mem, &topo, 256);
        let madt = read_madt(&mem, &l);
        let sum: u8 = madt.iter().fold(0u8, |acc, &b| acc.wrapping_add(b));
        assert_eq!(
            sum, 0,
            "MADT checksum failed for {llcs}l/{cores}c/{threads}t"
        );
        let entries = walk_madt_entries(&madt);
        let cpu_count = entries
            .iter()
            .filter(|(t, _, _)| *t == 0 || *t == 9)
            .count();
        assert_eq!(cpu_count, topo.total_cpus() as usize);
        assert!(entries.iter().any(|(t, _, _)| *t == 4 || *t == 0x0A));
    }
}

#[test]
fn cpu_entry_type_matches_apic_id() {
    for (llcs, cores, threads) in [(1, 4, 1), (2, 2, 2), (15, 8, 2), (14, 9, 2)] {
        let mem = test_mem(16);
        let topo = Topology {
            llcs,
            cores_per_llc: cores,
            threads_per_core: threads,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        };
        let l = test_setup(&mem, &topo, 256);
        let madt = read_madt(&mem, &l);
        let entries = walk_madt_entries(&madt);
        let mut cpu_idx = 0u32;
        for (entry_type, _, data) in &entries {
            match *entry_type {
                0 => {
                    let id = data[3] as u32;
                    assert!(id < 255);
                    assert_eq!(id, apic_id(&topo, cpu_idx));
                    cpu_idx += 1;
                }
                9 => {
                    let id = u32::from_le_bytes(data[4..8].try_into().unwrap());
                    assert!(id >= 255);
                    assert_eq!(id, apic_id(&topo, cpu_idx));
                    cpu_idx += 1;
                }
                _ => {}
            }
        }
    }
}

#[test]
fn madt_entry_lengths_valid() {
    let mem = test_mem(16);
    let topo = Topology {
        llcs: 14,
        cores_per_llc: 9,
        threads_per_core: 2,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    let l = test_setup(&mem, &topo, 256);
    let madt = read_madt(&mem, &l);
    let entries = walk_madt_entries(&madt);
    for (entry_type, entry_len, _) in &entries {
        let expected = match *entry_type {
            0 => 8,
            1 => 12,
            2 => 10,
            4 => 6,
            9 => 16,
            0x0A => 12,
            t => panic!("unexpected MADT entry type {t}"),
        };
        assert_eq!(*entry_len, expected);
    }
}

#[test]
fn madt_total_length_matches_entries() {
    let mem = test_mem(16);
    let topo = Topology {
        llcs: 14,
        cores_per_llc: 9,
        threads_per_core: 2,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    let l = test_setup(&mem, &topo, 256);
    let madt = read_madt(&mem, &l);
    let declared_len = u32::from_le_bytes(madt[4..8].try_into().unwrap()) as usize;
    assert_eq!(declared_len, madt.len());
    let entries = walk_madt_entries(&madt);
    let entries_size: usize = entries.iter().map(|(_, l, _)| *l as usize).sum();
    assert_eq!(
        std::mem::size_of::<MadtHeader>() + entries_size,
        declared_len
    );
}

#[test]
fn cpu_flags_enabled() {
    let mem = test_mem(16);
    let topo = Topology {
        llcs: 2,
        cores_per_llc: 2,
        threads_per_core: 2,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    let l = test_setup(&mem, &topo, 256);
    let madt = read_madt(&mem, &l);
    let entries = walk_madt_entries(&madt);
    for (entry_type, _, data) in &entries {
        match *entry_type {
            0 => assert_eq!(u32::from_le_bytes(data[4..8].try_into().unwrap()) & 1, 1),
            9 => assert_eq!(u32::from_le_bytes(data[8..12].try_into().unwrap()) & 1, 1),
            _ => {}
        }
    }
}

#[test]
fn rsdp_rev2_structure() {
    let mem = test_mem(16);
    let topo = Topology {
        llcs: 1,
        cores_per_llc: 1,
        threads_per_core: 1,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    let l = test_setup(&mem, &topo, 256);
    let mut rsdp = [0u8; 36];
    mem.read_slice(&mut rsdp, GuestAddress(l.rsdp_addr))
        .unwrap();
    assert_eq!(&rsdp[..8], b"RSD PTR ");
    assert_eq!(rsdp[15], 2);
    assert_eq!(
        u32::from_le_bytes(rsdp[16..20].try_into().unwrap()),
        l.rsdt_addr as u32
    );
    assert_eq!(u32::from_le_bytes(rsdp[20..24].try_into().unwrap()), 36);
    assert_eq!(
        u64::from_le_bytes(rsdp[24..32].try_into().unwrap()),
        l.xsdt_addr
    );
    let sum20: u8 = rsdp[..20].iter().fold(0u8, |acc, &b| acc.wrapping_add(b));
    assert_eq!(sum20, 0);
    let sum36: u8 = rsdp.iter().fold(0u8, |acc, &b| acc.wrapping_add(b));
    assert_eq!(sum36, 0);
}

#[test]
fn xsdt_signature_and_checksum() {
    let mem = test_mem(16);
    let topo = Topology {
        llcs: 1,
        cores_per_llc: 1,
        threads_per_core: 1,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    let l = test_setup(&mem, &topo, 256);
    let xsdt = read_table(&mem, l.xsdt_addr);
    assert_eq!(&xsdt[..4], b"XSDT");
    assert_eq!(xsdt.len(), 68);
    let sum: u8 = xsdt.iter().fold(0u8, |acc, &b| acc.wrapping_add(b));
    assert_eq!(sum, 0);
}

#[test]
fn xsdt_table_pointers() {
    let mem = test_mem(16);
    let topo = Topology {
        llcs: 1,
        cores_per_llc: 1,
        threads_per_core: 1,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    let l = test_setup(&mem, &topo, 256);
    let mut entry = [0u8; 8];
    mem.read_slice(&mut entry, GuestAddress(l.xsdt_addr + 36))
        .unwrap();
    assert_eq!(u64::from_le_bytes(entry), l.fadt_addr);
    mem.read_slice(&mut entry, GuestAddress(l.xsdt_addr + 44))
        .unwrap();
    assert_eq!(u64::from_le_bytes(entry), l.madt_addr);
    mem.read_slice(&mut entry, GuestAddress(l.xsdt_addr + 52))
        .unwrap();
    assert_eq!(u64::from_le_bytes(entry), l.srat_addr);
    mem.read_slice(&mut entry, GuestAddress(l.xsdt_addr + 60))
        .unwrap();
    assert_eq!(u64::from_le_bytes(entry), l.slit_addr);
}

#[test]
fn fadt_signature_and_checksum() {
    let mem = test_mem(16);
    let topo = Topology {
        llcs: 1,
        cores_per_llc: 1,
        threads_per_core: 1,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    let l = test_setup(&mem, &topo, 256);
    let mut fadt = [0u8; 276];
    mem.read_slice(&mut fadt, GuestAddress(l.fadt_addr))
        .unwrap();
    assert_eq!(&fadt[..4], b"FACP");
    assert_eq!(u32::from_le_bytes(fadt[4..8].try_into().unwrap()), 276);
    assert_eq!(fadt[8], 6);
    let sum: u8 = fadt.iter().fold(0u8, |acc, &b| acc.wrapping_add(b));
    assert_eq!(sum, 0);
}

#[test]
fn fadt_hw_reduced_flags() {
    let mem = test_mem(16);
    let topo = Topology {
        llcs: 1,
        cores_per_llc: 1,
        threads_per_core: 1,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    let l = test_setup(&mem, &topo, 256);
    let mut fadt = [0u8; 276];
    mem.read_slice(&mut fadt, GuestAddress(l.fadt_addr))
        .unwrap();
    let flags = u32::from_le_bytes(fadt[112..116].try_into().unwrap());
    assert_eq!(flags & (1 << 20), 0, "HW_REDUCED_ACPI must not be set");
    assert_ne!(flags & FADT_F_PWR_BUTTON, 0);
    assert_ne!(flags & FADT_F_SLP_BUTTON, 0);
}

#[test]
fn fadt_pm_register_blocks() {
    // Non-hardware-reduced full ACPI needs populated PM register blocks.
    // With them zero, ACPICA faults (AE_BAD_ADDRESS) initializing fixed
    // events, ACPI never enables, and PCI INTx falls back to legacy MP-table
    // routing that can't route the virtio-net PCI function. Pin every PM-block
    // field so a regression that zeroes them is caught at unit time, not by a
    // booted-guest e2e.
    let mem = test_mem(16);
    let topo = Topology {
        llcs: 1,
        cores_per_llc: 1,
        threads_per_core: 1,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    let l = test_setup(&mem, &topo, 256);
    let mut fadt = [0u8; 276];
    mem.read_slice(&mut fadt, GuestAddress(l.fadt_addr))
        .unwrap();
    // SCI_INT (offset 46): GSI 9, the conventional SCI line.
    assert_eq!(u16::from_le_bytes(fadt[46..48].try_into().unwrap()), 9);
    // PM1a_EVT_BLK / PM1a_CNT_BLK / PM_TMR_BLK (offsets 56/64/76): 32-bit
    // legacy I/O port addresses. Must be non-zero or ACPICA faults.
    assert_eq!(
        u32::from_le_bytes(fadt[56..60].try_into().unwrap()),
        u32::from(ACPI_PM1_EVT_PORT)
    );
    assert_eq!(
        u32::from_le_bytes(fadt[64..68].try_into().unwrap()),
        u32::from(ACPI_PM1_CNT_PORT)
    );
    assert_eq!(
        u32::from_le_bytes(fadt[76..80].try_into().unwrap()),
        u32::from(ACPI_PM_TMR_PORT)
    );
    // PM1_EVT_LEN / PM1_CNT_LEN / PM_TMR_LEN (offsets 88/89/91).
    assert_eq!(fadt[88], 4);
    assert_eq!(fadt[89], 2);
    assert_eq!(fadt[91], 4);
    // The optional SEPARATE_LENGTH blocks are deliberately left fully zero
    // (acpi_tb_convert_fadt silently accepts a fully-zero optional block; ktstr
    // emulates no PM2 control or GPE block). Pin them so a regression that
    // populated one — which acpi_tb_convert_fadt would then validate and could
    // fault on a malformed pair — is caught: PM2_CNT_BLK@72 / GPE0_BLK@80 (u32),
    // PM2_CNT_LEN@90 / GPE0_BLK_LEN@92 (u8).
    assert_eq!(u32::from_le_bytes(fadt[72..76].try_into().unwrap()), 0, "PM2_CNT_BLK");
    assert_eq!(u32::from_le_bytes(fadt[80..84].try_into().unwrap()), 0, "GPE0_BLK");
    assert_eq!(fadt[90], 0, "PM2_CNT_LEN");
    assert_eq!(fadt[92], 0, "GPE0_BLK_LEN");
    // IA-PC Boot Architecture Flags (offset 109): bit1 = i8042 present,
    // bit2 = VGA not present.
    assert_eq!(
        u16::from_le_bytes(fadt[109..111].try_into().unwrap()),
        0x0006
    );
}

#[test]
fn fadt_extended_gas_blocks_are_zero() {
    // We populate ONLY the legacy 32-bit I/O-port PM blocks (offsets 56/64/76)
    // and leave every 64-bit extended GAS field zero. ACPICA's
    // acpi_tb_convert_fadt (drivers/acpi/acpica/tbfadt.c) synthesizes the X_*
    // GAS from the legacy block whenever the X_ address is zero and the legacy
    // block is non-zero. Populating BOTH risks an inconsistent pair that
    // acpi_tb_convert_fadt warns on (legacy-vs-X mismatch); zeroing the X_
    // blocks is the spec-clean way to let ACPICA derive them. Pin that the
    // X_PM*/X_GPE*/SLEEP_*/HypervisorVendorId region (offset 148 to the end of
    // the 276-byte ACPI-6 FADT) stays all-zero so a regression that starts
    // writing a partial GAS is caught here.
    let mem = test_mem(16);
    let topo = Topology {
        llcs: 1,
        cores_per_llc: 1,
        threads_per_core: 1,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    let l = test_setup(&mem, &topo, 256);
    let mut fadt = [0u8; 276];
    mem.read_slice(&mut fadt, GuestAddress(l.fadt_addr))
        .unwrap();
    assert!(
        fadt[148..].iter().all(|&b| b == 0),
        "extended GAS region (offset 148..) must be zero so ACPICA synthesizes \
         X_* blocks from the legacy ports; found {:?}",
        &fadt[148..]
    );
}

#[test]
fn fadt_minor_version() {
    let mem = test_mem(16);
    let topo = Topology {
        llcs: 1,
        cores_per_llc: 1,
        threads_per_core: 1,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    let l = test_setup(&mem, &topo, 256);
    let mut fadt = [0u8; 276];
    mem.read_slice(&mut fadt, GuestAddress(l.fadt_addr))
        .unwrap();
    assert_eq!(fadt[131], 5);
}

#[test]
fn fadt_dsdt_pointers() {
    let mem = test_mem(16);
    let topo = Topology {
        llcs: 1,
        cores_per_llc: 1,
        threads_per_core: 1,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    let l = test_setup(&mem, &topo, 256);
    let mut fadt = [0u8; 276];
    mem.read_slice(&mut fadt, GuestAddress(l.fadt_addr))
        .unwrap();
    assert_eq!(
        u32::from_le_bytes(fadt[40..44].try_into().unwrap()),
        l.dsdt_addr as u32
    );
    assert_eq!(
        u64::from_le_bytes(fadt[140..148].try_into().unwrap()),
        l.dsdt_addr
    );
}

#[test]
fn dsdt_signature_and_checksum() {
    let mem = test_mem(16);
    let topo = Topology {
        llcs: 1,
        cores_per_llc: 1,
        threads_per_core: 1,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    let l = test_setup(&mem, &topo, 256);
    let mut dsdt = [0u8; 36];
    mem.read_slice(&mut dsdt, GuestAddress(l.dsdt_addr))
        .unwrap();
    assert_eq!(&dsdt[..4], b"DSDT");
    assert_eq!(u32::from_le_bytes(dsdt[4..8].try_into().unwrap()), 36);
    let sum: u8 = dsdt.iter().fold(0u8, |acc, &b| acc.wrapping_add(b));
    assert_eq!(sum, 0, "DSDT checksum must be zero");
}

#[test]
fn rsdp_points_to_xsdt() {
    let mem = test_mem(16);
    let topo = Topology {
        llcs: 1,
        cores_per_llc: 1,
        threads_per_core: 1,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    let l = test_setup(&mem, &topo, 256);
    let mut rsdp = [0u8; 36];
    mem.read_slice(&mut rsdp, GuestAddress(l.rsdp_addr))
        .unwrap();
    assert_eq!(
        u64::from_le_bytes(rsdp[24..32].try_into().unwrap()),
        l.xsdt_addr
    );
}

fn walk_srat_entries(srat: &[u8]) -> Vec<(u8, u8, &[u8])> {
    let hdr_size = 48; // 36-byte SDT + 12-byte SRAT-specific
    let mut entries = Vec::new();
    let mut offset = hdr_size;
    while offset < srat.len() {
        let entry_type = srat[offset];
        let entry_len = srat[offset + 1] as usize;
        entries.push((
            entry_type,
            entry_len as u8,
            &srat[offset..offset + entry_len],
        ));
        offset += entry_len;
    }
    entries
}

#[test]
fn srat_cpu_affinity_multi_numa() {
    for (numa_nodes, llcs, cores, threads) in
        [(2, 4, 2, 1), (2, 4, 2, 2), (4, 8, 1, 1), (3, 6, 2, 2)]
    {
        let mem = test_mem(16);
        let topo = Topology {
            llcs,
            cores_per_llc: cores,
            threads_per_core: threads,
            numa_nodes,
            nodes: None,
            distances: None,
        };
        let l = test_setup(&mem, &topo, 256);
        let srat = read_table(&mem, l.srat_addr);
        let entries = walk_srat_entries(&srat);
        let mut cpu_idx = 0u32;
        for (entry_type, _, data) in &entries {
            if *entry_type == 2 {
                let prox_domain = u32::from_le_bytes(data[4..8].try_into().unwrap());
                let (llc_id, _, _) = topo.decompose(cpu_idx);
                let expected_node = topo.numa_node_of(llc_id);
                assert_eq!(
                    prox_domain, expected_node,
                    "cpu {cpu_idx}: proximity_domain {prox_domain} != expected {expected_node} \
                     (topo: {numa_nodes}n/{llcs}l/{cores}c/{threads}t)"
                );
                let x2apic = u32::from_le_bytes(data[8..12].try_into().unwrap());
                assert_eq!(
                    x2apic,
                    apic_id(&topo, cpu_idx),
                    "cpu {cpu_idx}: x2apic_id mismatch"
                );
                cpu_idx += 1;
            }
        }
        assert_eq!(cpu_idx, topo.total_cpus());
    }
}

#[test]
fn srat_memory_split_multi_numa() {
    for (numa_nodes, llcs) in [(2, 4), (3, 6), (4, 8)] {
        let mem = test_mem(16);
        let topo = Topology {
            llcs,
            cores_per_llc: 2,
            threads_per_core: 1,
            numa_nodes,
            nodes: None,
            distances: None,
        };
        let mem_bytes = 256u64 << 20;
        let l = test_setup(&mem, &topo, 256);
        let srat = read_table(&mem, l.srat_addr);
        let entries = walk_srat_entries(&srat);
        let mem_entries: Vec<_> = entries.iter().filter(|(t, _, _)| *t == 1).collect();
        assert_eq!(mem_entries.len(), numa_nodes as usize);

        let mut prev_end: u64 = 0;
        let mut total: u64 = 0;
        for (i, (_, _, data)) in mem_entries.iter().enumerate() {
            let prox_domain = u32::from_le_bytes(data[2..6].try_into().unwrap());
            assert_eq!(
                prox_domain, i as u32,
                "node {i}: proximity_domain {prox_domain} != {i} \
                 (topo: {numa_nodes}n/{llcs}l)"
            );
            let base = u64::from_le_bytes(data[8..16].try_into().unwrap());
            let length = u64::from_le_bytes(data[16..24].try_into().unwrap());
            assert_eq!(
                base, prev_end,
                "node {i}: base {base:#x} != prev_end {prev_end:#x} \
                 (topo: {numa_nodes}n/{llcs}l)"
            );
            assert!(length > 0, "node {i}: zero-length memory region");
            prev_end = base + length;
            total += length;
        }
        assert_eq!(
            total, mem_bytes,
            "total memory mismatch for {numa_nodes}n/{llcs}l"
        );
    }
}

#[test]
fn slit_distance_matrix_multi_numa() {
    for (numa_nodes, llcs) in [(2, 2), (3, 3), (4, 4), (2, 4), (2, 6), (3, 9)] {
        let mem = test_mem(16);
        let topo = Topology {
            llcs,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes,
            nodes: None,
            distances: None,
        };
        let l = test_setup(&mem, &topo, 256);
        let slit = read_table(&mem, l.slit_addr);
        assert_eq!(&slit[..4], b"SLIT", "SLIT signature mismatch");
        let n = u64::from_le_bytes(slit[36..44].try_into().unwrap());
        assert_eq!(n, numa_nodes as u64);
        let matrix_start = 44;
        for i in 0..n {
            for j in 0..n {
                let dist = slit[matrix_start + (i * n + j) as usize];
                if i == j {
                    assert_eq!(dist, 10, "diagonal ({i},{j}) != 10");
                } else {
                    assert_eq!(dist, 20, "off-diagonal ({i},{j}) != 20");
                }
            }
        }
    }
}

#[test]
fn srat_slit_checksum_multi_numa() {
    for (numa_nodes, llcs, cores, threads) in
        [(2, 2, 2, 1), (2, 4, 2, 2), (3, 3, 1, 1), (4, 8, 4, 2)]
    {
        let mem = test_mem(16);
        let topo = Topology {
            llcs,
            cores_per_llc: cores,
            threads_per_core: threads,
            numa_nodes,
            nodes: None,
            distances: None,
        };
        let l = test_setup(&mem, &topo, 256);
        let srat = read_table(&mem, l.srat_addr);
        let srat_sum: u8 = srat.iter().fold(0u8, |acc, &b| acc.wrapping_add(b));
        assert_eq!(
            srat_sum, 0,
            "SRAT checksum failed for {numa_nodes}n/{llcs}l/{cores}c/{threads}t"
        );
        let slit = read_table(&mem, l.slit_addr);
        let slit_sum: u8 = slit.iter().fold(0u8, |acc, &b| acc.wrapping_add(b));
        assert_eq!(
            slit_sum, 0,
            "SLIT checksum failed for {numa_nodes}n/{llcs}l/{cores}c/{threads}t"
        );
    }
}

#[test]
fn srat_memory_split_remainder() {
    // 257 MiB / 3 nodes, snapped to 2 MiB hugepages by
    // NumaMemoryLayout: KVM installs a 2 MiB EPT entry only when a
    // node slot's base/end GPA are 2 MiB-congruent, so the split is
    // hugepage-aligned rather than dividing raw MiB (see
    // hugepage_even_split_mib). total_hp = 257/2 = 128; base =
    // 128/3 = 42 hp/node; extra = 128 % 3 = 2, so the first two
    // nodes get 43 hp (86 MiB) and the last node gets 42 hp plus
    // the 1 MiB sub-hugepage tail (85 MiB). The first nodes absorb
    // the extra hugepages; the last node absorbs the sub-2 MiB
    // remainder, so the last node is SMALLER than the others.
    let memory_mib = 257u32;
    let mem = test_mem(memory_mib);
    let topo = Topology {
        llcs: 3,
        cores_per_llc: 1,
        threads_per_core: 1,
        numa_nodes: 3,
        nodes: None,
        distances: None,
    };
    let mem_bytes = (memory_mib as u64) << 20;
    let first_two = 86u64 << 20; // 43 hugepages each
    let last_bytes = 85u64 << 20; // 42 hugepages + 1 MiB tail
    let l = test_setup(&mem, &topo, memory_mib);
    let srat = read_table(&mem, l.srat_addr);
    let entries = walk_srat_entries(&srat);
    let mem_entries: Vec<_> = entries.iter().filter(|(t, _, _)| *t == 1).collect();
    assert_eq!(mem_entries.len(), 3);
    let mut total: u64 = 0;
    for (i, (_, _, data)) in mem_entries.iter().enumerate() {
        let length = u64::from_le_bytes(data[16..24].try_into().unwrap());
        if i < 2 {
            assert_eq!(
                length, first_two,
                "node {i}: expected {first_two}, got {length}"
            );
        } else {
            assert_eq!(
                length, last_bytes,
                "last node: expected {last_bytes}, got {length}"
            );
            assert!(
                length < first_two,
                "last node should be smaller — it absorbs the \
                 sub-2 MiB tail while the first nodes get the extra \
                 hugepages"
            );
        }
        total += length;
    }
    assert_eq!(total, mem_bytes);
}

#[test]
fn srat_total_memory() {
    let mem = test_mem(16);
    let topo = Topology {
        llcs: 2,
        cores_per_llc: 2,
        threads_per_core: 1,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    let l = test_setup(&mem, &topo, 256);
    let srat = read_table(&mem, l.srat_addr);
    let entries = walk_srat_entries(&srat);
    let total_mem: u64 = entries
        .iter()
        .filter(|(t, _, _)| *t == 1)
        .map(|(_, _, data)| u64::from_le_bytes(data[16..24].try_into().unwrap()))
        .sum();
    let expected = 256u64 << 20;
    assert_eq!(total_mem, expected);
}

// -- HMAT tests --

use crate::vmm::topology::NumaNode;

static CXL_NODES: [NumaNode; 3] = [
    NumaNode::new(2, 256),
    NumaNode::new(2, 256),
    NumaNode::new(0, 128),
];

#[test]
fn hmat_not_emitted_single_node() {
    let mem = test_mem(16);
    let topo = Topology {
        llcs: 2,
        cores_per_llc: 2,
        threads_per_core: 1,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    let l = test_setup(&mem, &topo, 256);
    assert_eq!(l.hmat_size, 0, "HMAT must not be emitted for single-node");
}

#[test]
fn hmat_emitted_multi_numa_without_cxl() {
    let mem = test_mem(16);
    let topo = Topology {
        llcs: 4,
        cores_per_llc: 2,
        threads_per_core: 1,
        numa_nodes: 2,
        nodes: None,
        distances: None,
    };
    let l = test_setup(&mem, &topo, 256);
    assert!(l.hmat_size > 0, "HMAT must be emitted for multi-NUMA");
}

#[test]
fn hmat_emitted_with_cxl() {
    let topo = Topology::with_nodes(4, 1, &CXL_NODES);
    let mem = test_mem(16);
    let layout = NumaMemoryLayout::compute(&topo, 640, 0, None).unwrap();
    let l = setup_acpi(&mem, &topo, &layout, false, 0).unwrap();
    assert!(l.hmat_size > 0, "HMAT must be emitted with CXL nodes");
}

#[test]
fn hmat_checksum() {
    let topo = Topology::with_nodes(4, 1, &CXL_NODES);
    let mem = test_mem(16);
    let layout = NumaMemoryLayout::compute(&topo, 640, 0, None).unwrap();
    let l = setup_acpi(&mem, &topo, &layout, false, 0).unwrap();
    let hmat = read_table(&mem, l.hmat_addr);
    let sum: u8 = hmat.iter().fold(0u8, |acc, &b| acc.wrapping_add(b));
    assert_eq!(sum, 0, "HMAT checksum must be zero");
}

#[test]
fn hmat_header_fields() {
    let topo = Topology::with_nodes(4, 1, &CXL_NODES);
    let mem = test_mem(16);
    let layout = NumaMemoryLayout::compute(&topo, 640, 0, None).unwrap();
    let l = setup_acpi(&mem, &topo, &layout, false, 0).unwrap();
    let hmat = read_table(&mem, l.hmat_addr);
    assert_eq!(&hmat[..4], b"HMAT");
    assert_eq!(hmat[8], 2, "HMAT revision must be 2");
    assert_eq!(
        &hmat[36..40],
        &[0, 0, 0, 0],
        "4 reserved bytes after SDT header"
    );
}

#[test]
fn hmat_mpda_count_and_flags() {
    let topo = Topology::with_nodes(4, 1, &CXL_NODES);
    let mem = test_mem(16);
    let layout = NumaMemoryLayout::compute(&topo, 640, 0, None).unwrap();
    let l = setup_acpi(&mem, &topo, &layout, false, 0).unwrap();
    let hmat = read_table(&mem, l.hmat_addr);

    let num_targets = layout.regions().len();
    let mut offset = 40;
    let mut mpda_count = 0;
    while offset < hmat.len() {
        let hmat_type = u16::from_le_bytes(hmat[offset..offset + 2].try_into().unwrap());
        if hmat_type != 0 {
            break;
        }
        let length = u32::from_le_bytes(hmat[offset + 4..offset + 8].try_into().unwrap());
        assert_eq!(length, 40, "MPDA length must be 40");
        let flags = u16::from_le_bytes(hmat[offset + 8..offset + 10].try_into().unwrap());
        assert_eq!(
            flags, 3,
            "MPDA flags must be 3 (PROCESSOR_PD_VALID | MEMORY_PD_VALID)"
        );
        mpda_count += 1;
        offset += length as usize;
    }
    assert_eq!(
        mpda_count, num_targets,
        "one MPDA per memory target (layout region)"
    );
}

#[test]
fn hmat_mpda_cxl_initiator() {
    let topo = Topology::with_nodes(4, 1, &CXL_NODES);
    let mem = test_mem(16);
    let layout = NumaMemoryLayout::compute(&topo, 640, 0, None).unwrap();
    let l = setup_acpi(&mem, &topo, &layout, false, 0).unwrap();
    let hmat = read_table(&mem, l.hmat_addr);

    let mut offset = 40;
    for region in layout.regions() {
        let hmat_type = u16::from_le_bytes(hmat[offset..offset + 2].try_into().unwrap());
        assert_eq!(hmat_type, 0);
        let initiator = u32::from_le_bytes(hmat[offset + 12..offset + 16].try_into().unwrap());
        let memory_pd = u32::from_le_bytes(hmat[offset + 16..offset + 20].try_into().unwrap());
        assert_eq!(memory_pd, region.node_id);
        if topo.llcs_in_node(region.node_id) > 0 {
            assert_eq!(
                initiator, region.node_id,
                "CPU-bearing node {}: initiator must be self",
                region.node_id
            );
        } else {
            assert_ne!(
                topo.llcs_in_node(initiator),
                0,
                "CXL node {}: initiator {} must be CPU-bearing",
                region.node_id,
                initiator
            );
        }
        offset += 40;
    }
}

#[test]
fn hmat_sllbi_latency_and_bandwidth() {
    let topo = Topology::with_nodes(4, 1, &CXL_NODES);
    let mem = test_mem(16);
    let layout = NumaMemoryLayout::compute(&topo, 640, 0, None).unwrap();
    let l = setup_acpi(&mem, &topo, &layout, false, 0).unwrap();
    let hmat = read_table(&mem, l.hmat_addr);

    let num_targets = layout.regions().len();
    let mut offset = 40 + 40 * num_targets;

    for expected_data_type in [0u8, 3u8] {
        let hmat_type = u16::from_le_bytes(hmat[offset..offset + 2].try_into().unwrap());
        assert_eq!(hmat_type, 1, "SLLBI type must be 1");
        let length = u32::from_le_bytes(hmat[offset + 4..offset + 8].try_into().unwrap());
        let data_type = hmat[offset + 9];
        assert_eq!(data_type, expected_data_type);
        let ni = u32::from_le_bytes(hmat[offset + 12..offset + 16].try_into().unwrap());
        let nt = u32::from_le_bytes(hmat[offset + 16..offset + 20].try_into().unwrap());
        assert_eq!(ni, topo.cpu_bearing_nodes());
        assert_eq!(nt, num_targets as u32);
        let base_unit = u64::from_le_bytes(hmat[offset + 24..offset + 32].try_into().unwrap());
        if data_type == 0 {
            assert_eq!(base_unit, 100_000, "latency base must be 100000 ps");
        } else {
            assert_eq!(base_unit, 10_240, "bandwidth base must be 10240 MB/s");
        }
        offset += length as usize;
    }
}

#[test]
fn hmat_sllbi_cxl_entries_differ() {
    let topo = Topology::with_nodes(4, 1, &CXL_NODES);
    let mem = test_mem(16);
    let layout = NumaMemoryLayout::compute(&topo, 640, 0, None).unwrap();
    let l = setup_acpi(&mem, &topo, &layout, false, 0).unwrap();
    let hmat = read_table(&mem, l.hmat_addr);

    let num_targets = layout.regions().len();
    let ni = topo.cpu_bearing_nodes() as usize;
    let nt = num_targets;

    // Skip to first SLLBI (latency) entry matrix.
    let sllbi_offset = 40 + 40 * nt;
    let matrix_offset = sllbi_offset + 32 + 4 * ni + 4 * nt;

    // CXL target is the last region (node 2, memory-only).
    // Read first initiator's entries for DRAM target 0 and CXL target 2.
    let dram_entry = u16::from_le_bytes(hmat[matrix_offset..matrix_offset + 2].try_into().unwrap());
    let cxl_entry = u16::from_le_bytes(
        hmat[matrix_offset + 2 * (nt - 1)..matrix_offset + 2 * nt]
            .try_into()
            .unwrap(),
    );
    // Local DRAM: base=1, distance=10 → 1*10/10 = 1
    assert_eq!(dram_entry, 1, "local DRAM latency entry must be 1");
    // Remote CXL: base=3, distance=20 → 3*20/10 = 6
    assert_eq!(
        cxl_entry, 6,
        "remote CXL latency entry must be 6 (distance-scaled)"
    );
}

#[test]
fn hmat_rsdt_xsdt_include_pointer() {
    let topo = Topology::with_nodes(4, 1, &CXL_NODES);
    let mem = test_mem(16);
    let layout = NumaMemoryLayout::compute(&topo, 640, 0, None).unwrap();
    let l = setup_acpi(&mem, &topo, &layout, false, 0).unwrap();

    // RSDT should have 5 entries (FADT, MADT, SRAT, SLIT, HMAT).
    let rsdt = read_table(&mem, l.rsdt_addr);
    let rsdt_entries = (rsdt.len() - 36) / 4;
    assert_eq!(rsdt_entries, 5, "RSDT must have 5 table pointers with HMAT");
    let hmat_ptr = u32::from_le_bytes(rsdt[36 + 16..36 + 20].try_into().unwrap());
    assert_eq!(hmat_ptr, l.hmat_addr as u32);

    // XSDT should have 5 entries.
    let xsdt = read_table(&mem, l.xsdt_addr);
    let xsdt_entries = (xsdt.len() - 36) / 8;
    assert_eq!(xsdt_entries, 5, "XSDT must have 5 table pointers with HMAT");
    let hmat_ptr64 = u64::from_le_bytes(xsdt[36 + 32..36 + 40].try_into().unwrap());
    assert_eq!(hmat_ptr64, l.hmat_addr);
}

#[test]
fn no_hmat_rsdt_has_4_entries() {
    let mem = test_mem(16);
    let topo = Topology::new(1, 2, 2, 1);
    let l = test_setup(&mem, &topo, 256);
    let rsdt = read_table(&mem, l.rsdt_addr);
    let rsdt_entries = (rsdt.len() - 36) / 4;
    assert_eq!(
        rsdt_entries, 4,
        "RSDT must have 4 table pointers without HMAT"
    );
}
