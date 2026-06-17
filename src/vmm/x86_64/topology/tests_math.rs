use super::*;

#[test]
fn bits_needed_values() {
    assert_eq!(bits_needed(1), 0);
    assert_eq!(bits_needed(2), 1);
    assert_eq!(bits_needed(3), 2);
    assert_eq!(bits_needed(4), 2);
    assert_eq!(bits_needed(5), 3);
    assert_eq!(bits_needed(8), 3);
    assert_eq!(bits_needed(9), 4);
    assert_eq!(bits_needed(16), 4);
}

/// Build a kvm_cpuid_entry2 with the given cache level in EAX[7:5]
/// and arbitrary host values in the sharing/core fields to verify
/// patch_cache_topology_eax overwrites them.
fn make_cache_entry(level: u32) -> kvm_cpuid_entry2 {
    kvm_cpuid_entry2 {
        function: 0x4, // doesn't matter for the helper
        index: 0,
        flags: 0,
        // Set cache level in bits [7:5], type=2 (unified) in [4:0],
        // and fill sharing/core fields with 0xfff / 0x3f (max host values)
        // to verify the helper overwrites them.
        eax: (level << 5) | 2 | (0xfff << 14) | (0x3f << 26),
        ebx: 0,
        ecx: 0,
        edx: 0,
        ..Default::default()
    }
}

#[test]
fn patch_cache_l1_smt_scoped() {
    // 4 cores, 2 threads: smt=1, core=3
    let mut entry = make_cache_entry(1);
    patch_cache_topology_eax(&mut entry, 1, 3, 4);
    let sharing = (entry.eax >> 14) & 0xfff;
    // L1: (1 << smt) - 1 = (1 << 1) - 1 = 1
    assert_eq!(sharing, 1, "L1 sharing should be SMT-scoped");
}

#[test]
fn patch_cache_l2_smt_scoped() {
    let mut entry = make_cache_entry(2);
    patch_cache_topology_eax(&mut entry, 1, 3, 4);
    let sharing = (entry.eax >> 14) & 0xfff;
    assert_eq!(sharing, 1, "L2 sharing should be SMT-scoped");
}

#[test]
fn patch_cache_l3_llc_scoped() {
    let mut entry = make_cache_entry(3);
    patch_cache_topology_eax(&mut entry, 1, 3, 4);
    let sharing = (entry.eax >> 14) & 0xfff;
    // L3: (1 << core) - 1 = (1 << 3) - 1 = 7
    assert_eq!(sharing, 7, "L3 sharing should be LLC-scoped");
}

#[test]
fn patch_cache_unknown_level_zero() {
    // Level 0
    let mut entry0 = make_cache_entry(0);
    patch_cache_topology_eax(&mut entry0, 1, 3, 4);
    assert_eq!((entry0.eax >> 14) & 0xfff, 0, "level 0 sharing should be 0");

    // Level 4
    let mut entry4 = make_cache_entry(4);
    patch_cache_topology_eax(&mut entry4, 1, 3, 4);
    assert_eq!((entry4.eax >> 14) & 0xfff, 0, "level 4 sharing should be 0");

    // Level 7
    let mut entry7 = make_cache_entry(7);
    patch_cache_topology_eax(&mut entry7, 1, 3, 4);
    assert_eq!((entry7.eax >> 14) & 0xfff, 0, "level 7 sharing should be 0");
}

#[test]
fn patch_cache_single_core_single_thread() {
    // smt=0, core=0, cores_per_llc=1
    let mut entry_l1 = make_cache_entry(1);
    patch_cache_topology_eax(&mut entry_l1, 0, 0, 1);
    // (1 << 0) - 1 = 0
    assert_eq!(
        (entry_l1.eax >> 14) & 0xfff,
        0,
        "1c/1t L1 sharing should be 0"
    );
    // core_bits = bits_needed(1) = 0, max_core_ids = 0
    assert_eq!((entry_l1.eax >> 26) & 0x3f, 0, "1c/1t core IDs should be 0");

    let mut entry_l3 = make_cache_entry(3);
    patch_cache_topology_eax(&mut entry_l3, 0, 0, 1);
    assert_eq!(
        (entry_l3.eax >> 14) & 0xfff,
        0,
        "1c/1t L3 sharing should be 0"
    );
}

#[test]
fn patch_cache_large_topology() {
    // 16 cores, 2 threads: smt=1, core=5, cores_per_llc=16
    let smt = bits_needed(2); // 1
    let core = smt + bits_needed(16); // 1 + 4 = 5
    assert_eq!(smt, 1);
    assert_eq!(core, 5);

    let mut entry_l3 = make_cache_entry(3);
    patch_cache_topology_eax(&mut entry_l3, smt, core, 16);
    let sharing = (entry_l3.eax >> 14) & 0xfff;
    // (1 << 5) - 1 = 31
    assert_eq!(sharing, 31, "16c/2t L3 sharing");
    let core_ids = (entry_l3.eax >> 26) & 0x3f;
    // bits_needed(16) = 4, (1 << 4) - 1 = 15
    assert_eq!(core_ids, 15, "16c/2t core IDs");
    // Verify fields fit: sharing (31) fits in 12 bits, core_ids (15) fits in 6 bits
    assert!(sharing < (1 << 12));
    assert!(core_ids < (1 << 6));

    let mut entry_l1 = make_cache_entry(1);
    patch_cache_topology_eax(&mut entry_l1, smt, core, 16);
    assert_eq!(
        (entry_l1.eax >> 14) & 0xfff,
        1,
        "16c/2t L1 sharing (SMT-scoped)"
    );
}

#[test]
fn patch_cache_preserves_lower_bits() {
    // EAX[4:0] (type) and [13:8] should be preserved
    let mut entry = kvm_cpuid_entry2 {
        eax: (3 << 5) | 0b10101 | (0x2a << 8), // level=3, type=0b10101, bits[13:8]=0x2a
        ..Default::default()
    };
    patch_cache_topology_eax(&mut entry, 1, 3, 4);
    assert_eq!(entry.eax & 0x1f, 0b10101, "type bits [4:0] preserved");
    assert_eq!((entry.eax >> 5) & 0x7, 3, "level bits [7:5] preserved");
    assert_eq!((entry.eax >> 8) & 0x3f, 0x2a, "bits [13:8] preserved");
}

#[test]
fn patch_cache_leaf4_and_8000001d_identical() {
    // The core invariant: for the same topology, both leaves produce
    // identical EAX values.
    let topos = [(1, 1, 1), (2, 4, 1), (2, 4, 2), (4, 8, 2), (8, 16, 2)];
    for (llcs, cores, threads) in topos {
        let smt = bits_needed(threads);
        let core = smt + bits_needed(cores);
        for level in 1..=3 {
            let mut leaf4 = make_cache_entry(level);
            leaf4.function = 0x4;
            let mut leaf_amd = make_cache_entry(level);
            leaf_amd.function = 0x8000_001d;
            patch_cache_topology_eax(&mut leaf4, smt, core, cores);
            patch_cache_topology_eax(&mut leaf_amd, smt, core, cores);
            assert_eq!(
                leaf4.eax, leaf_amd.eax,
                "{llcs}l/{cores}c/{threads}t L{level}: leaf 0x4 and 0x8000001D \
                 EAX should be identical"
            );
        }
    }
}

#[test]
fn apic_ids_unique() {
    let t = Topology {
        llcs: 2,
        cores_per_llc: 4,
        threads_per_core: 2,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    let ids: Vec<u32> = (0..t.total_cpus()).map(|i| apic_id(&t, i)).collect();
    let unique: std::collections::HashSet<u32> = ids.iter().copied().collect();
    assert_eq!(ids.len(), unique.len(), "APIC IDs must be unique: {ids:?}");
}

#[test]
fn apic_ids_smt_siblings_adjacent() {
    let t = Topology {
        llcs: 2,
        cores_per_llc: 2,
        threads_per_core: 2,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    // SMT siblings should differ only in thread_id bits
    let smt_mask = (1u32 << smt_shift(&t)) - 1;
    for core_start in (0..t.total_cpus()).step_by(t.threads_per_core as usize) {
        let base = apic_id(&t, core_start) & !smt_mask;
        for thread in 0..t.threads_per_core {
            let apic = apic_id(&t, core_start + thread);
            assert_eq!(
                apic & !smt_mask,
                base,
                "SMT siblings should share upper bits: cpu {}, apic {apic:#x}",
                core_start + thread
            );
        }
    }
}

#[test]
fn apic_ids_same_llc_share_upper_bits() {
    let t = Topology {
        llcs: 2,
        cores_per_llc: 4,
        threads_per_core: 2,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    let pkg_mask = !((1u32 << core_shift(&t)) - 1);
    let cpus_per_llc = t.cores_per_llc * t.threads_per_core;
    for llc in 0..t.llcs {
        let start = llc * cpus_per_llc;
        let llc_bits = apic_id(&t, start) & pkg_mask;
        for cpu in start..start + cpus_per_llc {
            assert_eq!(
                apic_id(&t, cpu) & pkg_mask,
                llc_bits,
                "CPU {cpu} should be in LLC {llc}"
            );
        }
    }
    let s0 = apic_id(&t, 0) & pkg_mask;
    let s1 = apic_id(&t, cpus_per_llc) & pkg_mask;
    assert_ne!(s0, s1, "different LLCs should have different package IDs");
}

#[test]
fn smt_shift_values() {
    assert_eq!(
        smt_shift(&Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        }),
        0
    );
    assert_eq!(
        smt_shift(&Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 2,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        }),
        1
    );
    assert_eq!(
        smt_shift(&Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 4,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        }),
        2
    );
}

#[test]
fn core_shift_values() {
    // 1 thread, 4 cores: smt_shift=0, core_bits=2, core_shift=2
    assert_eq!(
        core_shift(&Topology {
            llcs: 1,
            cores_per_llc: 4,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        }),
        2
    );
    // 2 threads, 4 cores: smt_shift=1, core_bits=2, core_shift=3
    assert_eq!(
        core_shift(&Topology {
            llcs: 1,
            cores_per_llc: 4,
            threads_per_core: 2,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        }),
        3
    );
}

#[test]
fn generate_cpuid_produces_entries() {
    let kvm = match kvm_ioctls::Kvm::new() {
        Ok(k) => k,
        Err(_) => return, // skip if no KVM
    };
    let topo = Topology {
        llcs: 2,
        cores_per_llc: 2,
        threads_per_core: 2,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    let cpuid = generate_cpuid(
        kvm.get_supported_cpuid(kvm_bindings::KVM_MAX_CPUID_ENTRIES)
            .unwrap()
            .as_slice(),
        &topo,
        0,
        false,
    );
    assert!(!cpuid.is_empty());

    // Verify leaf 0x1 has correct APIC ID in EBX[31:24]
    let leaf1 = cpuid.iter().find(|e| e.function == 1);
    if let Some(entry) = leaf1 {
        let apic_from_cpuid = (entry.ebx >> 24) & 0xff;
        assert_eq!(apic_from_cpuid, apic_id(&topo, 0));
    }
}

#[test]
fn generate_cpuid_different_per_vcpu() {
    let kvm = match kvm_ioctls::Kvm::new() {
        Ok(k) => k,
        Err(_) => return,
    };
    let topo = Topology {
        llcs: 2,
        cores_per_llc: 2,
        threads_per_core: 1,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    let cpuid0 = generate_cpuid(
        kvm.get_supported_cpuid(kvm_bindings::KVM_MAX_CPUID_ENTRIES)
            .unwrap()
            .as_slice(),
        &topo,
        0,
        false,
    );
    let cpuid1 = generate_cpuid(
        kvm.get_supported_cpuid(kvm_bindings::KVM_MAX_CPUID_ENTRIES)
            .unwrap()
            .as_slice(),
        &topo,
        1,
        false,
    );

    // Different vCPUs should have different APIC IDs in leaf 0xB
    let leaf_b_0 = cpuid0.iter().find(|e| e.function == 0xb && e.index == 0);
    let leaf_b_1 = cpuid1.iter().find(|e| e.function == 0xb && e.index == 0);
    if let (Some(e0), Some(e1)) = (leaf_b_0, leaf_b_1) {
        assert_ne!(
            e0.edx, e1.edx,
            "different vCPUs should have different x2APIC IDs"
        );
    }
}

#[test]
fn topology_odd_counts() {
    let t = Topology {
        llcs: 3,
        cores_per_llc: 3,
        threads_per_core: 1,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    assert_eq!(t.total_cpus(), 9);
    let ids: Vec<u32> = (0..9).map(|i| apic_id(&t, i)).collect();
    let unique: std::collections::HashSet<u32> = ids.iter().copied().collect();
    assert_eq!(
        unique.len(),
        9,
        "odd topology APIC IDs must be unique: {ids:?}"
    );
}

#[test]
fn leaf1_lpc_is_package_total() {
    let kvm = match kvm_ioctls::Kvm::new() {
        Ok(k) => k,
        Err(_) => return,
    };
    let topo = Topology {
        llcs: 4,
        cores_per_llc: 4,
        threads_per_core: 2,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    assert_eq!(topo.total_cpus(), 32);
    let cpuid = generate_cpuid(
        kvm.get_supported_cpuid(kvm_bindings::KVM_MAX_CPUID_ENTRIES)
            .unwrap()
            .as_slice(),
        &topo,
        0,
        false,
    );
    let leaf1 = cpuid.iter().find(|e| e.function == 1);
    if let Some(entry) = leaf1 {
        let lpc = (entry.ebx >> 16) & 0xff;
        assert_eq!(
            lpc,
            32, // logical processors in the package = all CPUs (pow2)
            "EBX[23:16] is the package logical-processor count (all 32 CPUs), \
             not one LLC's worth"
        );
    }
}

#[test]
fn apic_ids_unique_representative_topologies() {
    let topos = [
        (1, 1, 1),   // degenerate single CPU
        (2, 1, 1),   // minimal multi-LLC
        (3, 3, 1),   // odd non-power-of-2
        (1, 1, 2),   // minimal SMT
        (2, 4, 2),   // standard multi-LLC with SMT
        (7, 5, 3),   // all dimensions non-power-of-2
        (15, 16, 1), // large scale no SMT
        (14, 9, 2),  // large with SMT, max APIC > 255
        (2, 128, 1), // x2APIC boundary (max APIC ID = 255)
    ];
    for (llcs, cores, threads) in topos {
        let t = Topology {
            llcs,
            cores_per_llc: cores,
            threads_per_core: threads,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        };
        let ids: Vec<u32> = (0..t.total_cpus()).map(|i| apic_id(&t, i)).collect();
        let unique: std::collections::HashSet<u32> = ids.iter().copied().collect();
        assert_eq!(
            ids.len(),
            unique.len(),
            "topology {llcs}l/{cores}c/{threads}t: APIC IDs not unique"
        );
    }
}

#[test]
fn leaf0b_subleaf0_ebx_is_threads_per_core() {
    let kvm = match kvm_ioctls::Kvm::new() {
        Ok(k) => k,
        Err(_) => return,
    };
    let topo = Topology {
        llcs: 2,
        cores_per_llc: 4,
        threads_per_core: 2,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    let cpuid = generate_cpuid(
        kvm.get_supported_cpuid(kvm_bindings::KVM_MAX_CPUID_ENTRIES)
            .unwrap()
            .as_slice(),
        &topo,
        0,
        false,
    );
    let leaf_b_0 = cpuid.iter().find(|e| e.function == 0xb && e.index == 0);
    if let Some(entry) = leaf_b_0 {
        assert_eq!(
            entry.ebx & 0xffff,
            2, // threads_per_core
            "leaf 0xB subleaf 0 EBX should be threads per core"
        );
        assert_eq!(
            entry.eax,
            smt_shift(&topo),
            "leaf 0xB subleaf 0 EAX should be smt_shift"
        );
    }
}

#[test]
fn leaf0b_subleaf1_core_spans_package() {
    let kvm = match kvm_ioctls::Kvm::new() {
        Ok(k) => k,
        Err(_) => return,
    };
    let topo = Topology {
        llcs: 2,
        cores_per_llc: 4,
        threads_per_core: 2,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    let cpuid = generate_cpuid(
        kvm.get_supported_cpuid(kvm_bindings::KVM_MAX_CPUID_ENTRIES)
            .unwrap()
            .as_slice(),
        &topo,
        0,
        false,
    );
    let leaf_b_1 = cpuid.iter().find(|e| e.function == 0xb && e.index == 1);
    if let Some(entry) = leaf_b_1 {
        assert_eq!(
            entry.ebx & 0xffff,
            topo.total_cpus(),
            "leaf 0xB Core subleaf EBX is the whole package's logical CPUs"
        );
        assert_eq!(
            entry.eax,
            bits_needed(max_apic_id(&topo) + 1),
            "leaf 0xB Core subleaf EAX is the package shift (apic >> it == 0)"
        );
    }
}

#[test]
fn leaf0b_ecx_includes_subleaf_index() {
    let kvm = match kvm_ioctls::Kvm::new() {
        Ok(k) => k,
        Err(_) => return,
    };
    let topo = Topology {
        llcs: 2,
        cores_per_llc: 2,
        threads_per_core: 2,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    let cpuid = generate_cpuid(
        kvm.get_supported_cpuid(kvm_bindings::KVM_MAX_CPUID_ENTRIES)
            .unwrap()
            .as_slice(),
        &topo,
        0,
        false,
    );
    // Subleaf 0: ECX should have level_type=1 (SMT) in bits 15:8 and index=0 in bits 7:0
    if let Some(entry) = cpuid.iter().find(|e| e.function == 0xb && e.index == 0) {
        assert_eq!(entry.ecx & 0xff, 0, "subleaf 0 ECX[7:0] should be 0");
        assert_eq!(
            (entry.ecx >> 8) & 0xff,
            1,
            "subleaf 0 ECX[15:8] should be 1 (SMT)"
        );
    }
    // Subleaf 1: ECX should have level_type=2 (Core) in bits 15:8 and index=1 in bits 7:0
    if let Some(entry) = cpuid.iter().find(|e| e.function == 0xb && e.index == 1) {
        assert_eq!(entry.ecx & 0xff, 1, "subleaf 1 ECX[7:0] should be 1");
        assert_eq!(
            (entry.ecx >> 8) & 0xff,
            2,
            "subleaf 1 ECX[15:8] should be 2 (Core)"
        );
    }
}

#[test]
fn leaf4_l3_shared_within_llc() {
    let kvm = match kvm_ioctls::Kvm::new() {
        Ok(k) => k,
        Err(_) => return,
    };
    let topo = Topology {
        llcs: 2,
        cores_per_llc: 4,
        threads_per_core: 2,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    let cpuid = generate_cpuid(
        kvm.get_supported_cpuid(kvm_bindings::KVM_MAX_CPUID_ENTRIES)
            .unwrap()
            .as_slice(),
        &topo,
        0,
        false,
    );
    let l3 = cpuid
        .iter()
        .find(|e| e.function == 0x4 && ((e.eax >> 5) & 0x7) == 3);
    if let Some(entry) = l3 {
        // EAX[25:14] = max addressable IDs sharing this cache - 1
        // For 4c/2t: core_shift=3, (1<<3)-1 = 7
        let max_sharing = ((entry.eax >> 14) & 0xfff) + 1;
        let expected = 1u32 << core_shift(&topo); // APIC-ID-space rounded
        assert_eq!(max_sharing, expected, "L3 max sharing: APIC-ID-space value");
    }
}

#[test]
fn leaf4_core_ids_apic_space() {
    let kvm = match kvm_ioctls::Kvm::new() {
        Ok(k) => k,
        Err(_) => return,
    };
    // 3 cores: needs 2 bits, so (1<<2)-1 = 3 addressable core IDs
    let topo = Topology {
        llcs: 1,
        cores_per_llc: 3,
        threads_per_core: 1,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    let cpuid = generate_cpuid(
        kvm.get_supported_cpuid(kvm_bindings::KVM_MAX_CPUID_ENTRIES)
            .unwrap()
            .as_slice(),
        &topo,
        0,
        false,
    );
    // Find any valid leaf 0x4 entry
    let leaf4 = cpuid
        .iter()
        .find(|e| e.function == 0x4 && ((e.eax >> 5) & 0x7) > 0);
    if let Some(entry) = leaf4 {
        let max_core_ids = ((entry.eax >> 26) & 0x3f) + 1;
        let core_bits = bits_needed(topo.cores_per_llc);
        assert_eq!(
            max_core_ids,
            1 << core_bits,
            "leaf 0x4 EAX[31:26]+1 should be power-of-2 from APIC ID space"
        );
    }
}

#[test]
fn leaf1_hypervisor_bit_set() {
    let kvm = match kvm_ioctls::Kvm::new() {
        Ok(k) => k,
        Err(_) => return,
    };
    let topo = Topology {
        llcs: 1,
        cores_per_llc: 1,
        threads_per_core: 1,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    let cpuid = generate_cpuid(
        kvm.get_supported_cpuid(kvm_bindings::KVM_MAX_CPUID_ENTRIES)
            .unwrap()
            .as_slice(),
        &topo,
        0,
        false,
    );
    let leaf1 = cpuid.iter().find(|e| e.function == 1);
    if let Some(entry) = leaf1 {
        assert_ne!(
            entry.ecx & (1 << 31),
            0,
            "hypervisor bit (ECX.31) should be set"
        );
    }
}

#[test]
fn leaf1_clflush_set() {
    let kvm = match kvm_ioctls::Kvm::new() {
        Ok(k) => k,
        Err(_) => return,
    };
    let topo = Topology {
        llcs: 1,
        cores_per_llc: 1,
        threads_per_core: 1,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    let cpuid = generate_cpuid(
        kvm.get_supported_cpuid(kvm_bindings::KVM_MAX_CPUID_ENTRIES)
            .unwrap()
            .as_slice(),
        &topo,
        0,
        false,
    );
    let leaf1 = cpuid.iter().find(|e| e.function == 1);
    if let Some(entry) = leaf1 {
        let clflush = (entry.ebx >> 8) & 0xff;
        assert_eq!(clflush, 8, "CLFLUSH should be 8 (64-byte cache lines)");
    }
}

#[test]
fn leaf_0xa_pmu_v2_synthesized() {
    let kvm = match kvm_ioctls::Kvm::new() {
        Ok(k) => k,
        Err(_) => return,
    };
    let topo = Topology {
        llcs: 1,
        cores_per_llc: 1,
        threads_per_core: 1,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    let cpuid = generate_cpuid(
        kvm.get_supported_cpuid(kvm_bindings::KVM_MAX_CPUID_ENTRIES)
            .unwrap()
            .as_slice(),
        &topo,
        0,
        false,
    );
    let leaf_a = cpuid.iter().find(|e| e.function == 0xa);
    if let Some(entry) = leaf_a {
        // Bit-field decode per arch/x86/include/asm/perf_event.h
        // union cpuid10_eax: version_id, num_counters, bit_width, mask_length
        // (each 8 bits, little end first).
        let version = entry.eax & 0xff;
        // Host has no PMU (kvm.enable_pmu=0 or PMU-less hardware): the
        // gating in generate_cpuid leaves the entry zero on purpose so
        // the guest's intel_pmu_init sees version=0 and graceful-fails.
        // Skip the v2-shape assertions in that case — they apply only
        // when the host advertises a PMU and we synthesized over it.
        if version == 0 {
            return;
        }
        let num_gp = (entry.eax >> 8) & 0xff;
        let gp_width = (entry.eax >> 16) & 0xff;
        let mask_len = (entry.eax >> 24) & 0xff;
        assert_eq!(version, PMU_ARCH_PERFMON_VERSION, "PMU v2 version");
        assert_eq!(num_gp, PMU_NUM_GP_COUNTERS, "PMU v2 GP counter count");
        assert_eq!(gp_width, PMU_GP_COUNTER_WIDTH, "PMU v2 GP counter width");
        // mask_length must be >= ARCH_PERFMON_EVENTS_COUNT (7) or
        // intel_pmu_init in arch/x86/events/intel/core.c returns -ENODEV.
        assert_eq!(
            mask_len, PMU_EVENT_MASK_LENGTH,
            "mask_length must equal ARCH_PERFMON_EVENTS_COUNT"
        );

        // EBX = unsupported-event bitmap; 0 means all 7 arch events available.
        assert_eq!(entry.ebx, 0, "no architectural events disabled");
        assert_eq!(entry.ecx, 0, "ECX reserved for PMU v2");

        // EDX: union cpuid10_edx — num_counters_fixed[4:0], bit_width_fixed[12:5].
        let num_fixed = entry.edx & 0x1f;
        let fixed_width = (entry.edx >> 5) & 0xff;
        assert_eq!(
            num_fixed, PMU_NUM_FIXED_COUNTERS,
            "PMU v2 fixed counter count"
        );
        assert_eq!(
            fixed_width, PMU_FIXED_COUNTER_WIDTH,
            "PMU v2 fixed counter width"
        );
        assert_eq!(
            entry.edx & !0x1fff,
            0,
            "EDX bits[31:13] must be zero for PMU v2"
        );
    }
}

/// Synthetic-base test: build a base CPUID with leaf 0xA absent and
/// verify generate_cpuid leaves the result without leaf 0xA. This is
/// the host-absent-leaf scenario — KVM may omit leaf 0xA entirely
/// (rather than zeroing it) on hosts where intel_pmu_refresh decides
/// no PMU is available. The handler in generate_cpuid only fires
/// when the leaf is present in entries; if it isn't, the result
/// must also lack the leaf (no fabrication).
#[test]
fn leaf_0xa_absent_from_base_stays_absent() {
    let base = vec![kvm_cpuid_entry2 {
        function: 0,
        index: 0,
        flags: 0,
        eax: 0,
        ebx: 0x756e_6547,
        edx: 0x4965_6e69,
        ecx: 0x6c65_746e,
        ..Default::default()
    }];
    let topo = Topology {
        llcs: 1,
        cores_per_llc: 1,
        threads_per_core: 1,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    let cpuid = generate_cpuid(&base, &topo, 0, false);
    assert!(
        cpuid.iter().all(|e| e.function != 0xa),
        "leaf 0xA must not be fabricated when absent from base"
    );
}

/// Synthetic-base test: a base entry with version=0 (the kvm.enable_pmu=0
/// signal) must remain zero. Overwriting it would lie to the guest
/// about PMU availability while intel_pmu_refresh clamps every counter
/// back to 0.
#[test]
fn leaf_0xa_zero_version_preserved() {
    let base = vec![
        kvm_cpuid_entry2 {
            function: 0,
            index: 0,
            flags: 0,
            eax: 0,
            ebx: 0x756e_6547,
            edx: 0x4965_6e69,
            ecx: 0x6c65_746e,
            ..Default::default()
        },
        // Leaf 0xA with EAX[7:0]=0 — host PMU is disabled.
        kvm_cpuid_entry2 {
            function: 0xa,
            index: 0,
            flags: 0,
            eax: 0,
            ebx: 0,
            ecx: 0,
            edx: 0,
            ..Default::default()
        },
    ];
    let topo = Topology {
        llcs: 1,
        cores_per_llc: 1,
        threads_per_core: 1,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    let cpuid = generate_cpuid(&base, &topo, 0, false);
    let leaf_a = cpuid
        .iter()
        .find(|e| e.function == 0xa)
        .expect("leaf 0xA preserved from base");
    assert_eq!(leaf_a.eax, 0, "EAX must stay zero (host PMU disabled)");
    assert_eq!(leaf_a.ebx, 0, "EBX must stay zero");
    assert_eq!(leaf_a.ecx, 0, "ECX must stay zero");
    assert_eq!(leaf_a.edx, 0, "EDX must stay zero");
}

/// Synthetic-base test: a base entry with non-zero version is
/// overwritten with the synthesized PMU v2 surface using the
/// named consts at the top of the file. This covers vendor-agnostic
/// behavior — the leaf-0xA handler is unconditional on vendor.
#[test]
fn leaf_0xa_nonzero_version_synthesized_to_v2() {
    let base = vec![
        kvm_cpuid_entry2 {
            function: 0,
            index: 0,
            flags: 0,
            eax: 0,
            ebx: 0x756e_6547,
            edx: 0x4965_6e69,
            ecx: 0x6c65_746e,
            ..Default::default()
        },
        // Synthetic host PMU: version=5, 8 GP counters, 56-bit width,
        // mask_length=10. Our synthesizer must overwrite all four
        // registers to the conservative v2 surface.
        kvm_cpuid_entry2 {
            function: 0xa,
            index: 0,
            flags: 0,
            eax: 5 | (8 << 8) | (56 << 16) | (10 << 24),
            ebx: 0xdead_beef,
            ecx: 0xcafe_f00d,
            edx: 0xface_d00d,
            ..Default::default()
        },
    ];
    let topo = Topology {
        llcs: 1,
        cores_per_llc: 1,
        threads_per_core: 1,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    let cpuid = generate_cpuid(&base, &topo, 0, false);
    let leaf_a = cpuid
        .iter()
        .find(|e| e.function == 0xa)
        .expect("leaf 0xA present from base");
    let expected_eax = PMU_ARCH_PERFMON_VERSION
        | (PMU_NUM_GP_COUNTERS << 8)
        | (PMU_GP_COUNTER_WIDTH << 16)
        | (PMU_EVENT_MASK_LENGTH << 24);
    let expected_edx = PMU_NUM_FIXED_COUNTERS | (PMU_FIXED_COUNTER_WIDTH << 5);
    assert_eq!(leaf_a.eax, expected_eax, "EAX synthesized to PMU v2");
    assert_eq!(leaf_a.ebx, 0, "EBX cleared (no events disabled)");
    assert_eq!(leaf_a.ecx, 0, "ECX cleared (reserved)");
    assert_eq!(leaf_a.edx, expected_edx, "EDX synthesized to PMU v2");
}

/// Synthetic-base test: AMD vendor + non-zero leaf 0xA must follow
/// the same synthesis path as Intel. AMD CPUs ignore leaf 0xA at
/// runtime (they use MSR-based counters) but the leaf is still
/// present in CPUID enumeration and KVM may populate it. Our
/// vendor-agnostic handler must not introduce vendor divergence.
#[test]
fn leaf_0xa_amd_vendor_same_synthesis() {
    let base = vec![
        kvm_cpuid_entry2 {
            function: 0,
            index: 0,
            flags: 0,
            eax: 0,
            ebx: 0x6874_7541, // "Auth"
            edx: 0x6974_6e65, // "enti"
            ecx: 0x444d_4163, // "cAMD"
            ..Default::default()
        },
        kvm_cpuid_entry2 {
            function: 0xa,
            index: 0,
            flags: 0,
            eax: 1 | (2 << 8) | (32 << 16) | (4 << 24),
            ebx: 0x1111_1111,
            ecx: 0x2222_2222,
            edx: 0x3333_3333,
            ..Default::default()
        },
    ];
    let topo = Topology {
        llcs: 1,
        cores_per_llc: 1,
        threads_per_core: 1,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    let cpuid = generate_cpuid(&base, &topo, 0, false);
    let leaf_a = cpuid
        .iter()
        .find(|e| e.function == 0xa)
        .expect("leaf 0xA present from base");
    let expected_eax = PMU_ARCH_PERFMON_VERSION
        | (PMU_NUM_GP_COUNTERS << 8)
        | (PMU_GP_COUNTER_WIDTH << 16)
        | (PMU_EVENT_MASK_LENGTH << 24);
    let expected_edx = PMU_NUM_FIXED_COUNTERS | (PMU_FIXED_COUNTER_WIDTH << 5);
    assert_eq!(
        leaf_a.eax, expected_eax,
        "AMD synthesized identically to Intel"
    );
    assert_eq!(leaf_a.ebx, 0, "EBX cleared on AMD");
    assert_eq!(leaf_a.ecx, 0, "ECX cleared on AMD");
    assert_eq!(
        leaf_a.edx, expected_edx,
        "AMD synthesized identically to Intel"
    );
}

/// Synthetic-base test: when the base hypervisor max-leaf
/// (0x40000000.EAX) is already higher than 0x40000001 (e.g.
/// 0x40000010 from a host that exposes more KVM hypervisor leaves),
/// performance_mode must NOT lower it — we use .max() to bump but
/// preserve a higher existing value.
#[test]
fn max_hypervisor_leaf_not_lowered_in_performance_mode() {
    let base = vec![
        kvm_cpuid_entry2 {
            function: 0,
            index: 0,
            flags: 0,
            eax: 0,
            ebx: 0x756e_6547,
            edx: 0x4965_6e69,
            ecx: 0x6c65_746e,
            ..Default::default()
        },
        kvm_cpuid_entry2 {
            function: 0x4000_0000,
            index: 0,
            flags: 0,
            eax: 0x4000_0010, // already higher than 0x40000001
            ebx: 0,
            ecx: 0,
            edx: 0,
            ..Default::default()
        },
        kvm_cpuid_entry2 {
            function: 0x4000_0001,
            index: 0,
            flags: 0,
            eax: 0,
            ebx: 0,
            ecx: 0,
            edx: 0,
            ..Default::default()
        },
    ];
    let topo = Topology {
        llcs: 1,
        cores_per_llc: 1,
        threads_per_core: 1,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    let cpuid = generate_cpuid(&base, &topo, 0, true);
    let leaf40 = cpuid
        .iter()
        .find(|e| e.function == 0x4000_0000)
        .expect("leaf 0x40000000 present");
    assert_eq!(
        leaf40.eax, 0x4000_0010,
        "max hypervisor leaf must not regress when already > 0x40000001"
    );
}

#[test]
fn hypervisor_leaf_present() {
    let kvm = match kvm_ioctls::Kvm::new() {
        Ok(k) => k,
        Err(_) => return,
    };
    let topo = Topology {
        llcs: 1,
        cores_per_llc: 1,
        threads_per_core: 1,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    let cpuid = generate_cpuid(
        kvm.get_supported_cpuid(kvm_bindings::KVM_MAX_CPUID_ENTRIES)
            .unwrap()
            .as_slice(),
        &topo,
        0,
        false,
    );
    let leaf_40 = cpuid.iter().find(|e| e.function == 0x4000_0000);
    assert!(leaf_40.is_some(), "hypervisor leaf 0x40000000 should exist");
}

#[test]
fn decompose_roundtrip_representative_topologies() {
    let topos = [
        (1, 1, 1),   // degenerate single CPU
        (2, 1, 1),   // minimal multi-LLC
        (3, 3, 1),   // odd non-power-of-2
        (1, 1, 2),   // minimal SMT
        (2, 4, 2),   // standard multi-LLC with SMT
        (7, 5, 3),   // all dimensions non-power-of-2
        (15, 16, 1), // large scale no SMT
        (14, 9, 2),  // large with SMT, max APIC > 255
    ];
    for (llcs, cores, threads) in topos {
        let t = Topology {
            llcs,
            cores_per_llc: cores,
            threads_per_core: threads,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        };
        for cpu in 0..t.total_cpus() {
            let (s, c, th) = t.decompose(cpu);
            assert!(s < llcs, "cpu {cpu}: llc {s} >= {llcs}");
            assert!(c < cores, "cpu {cpu}: core {c} >= {cores}");
            assert!(th < threads, "cpu {cpu}: thread {th} >= {threads}");
            let recomposed = s * cores * threads + c * threads + th;
            assert_eq!(
                recomposed, cpu,
                "decompose roundtrip failed for {llcs}l/{cores}c/{threads}t cpu {cpu}"
            );
        }
    }
}
