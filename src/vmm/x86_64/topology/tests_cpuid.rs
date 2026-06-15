use super::*;


#[test]
fn leaf_80000008_amd_topology() {
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
    let leaf = cpuid.iter().find(|e| e.function == 0x8000_0008);
    if let Some(entry) = leaf {
        // ECX[7:0] = CPUs per package - 1 (all CPUs, package-scoped)
        let nc = entry.ecx & 0xff;
        assert_eq!(
            nc,
            topo.total_cpus() - 1,
            "NC should be cpus_per_package - 1"
        );
        // ECX[15:12] = APIC-ID core-id size = the package shift
        let apic_id_size = (entry.ecx >> 12) & 0xf;
        assert_eq!(
            apic_id_size,
            bits_needed(max_apic_id(&topo) + 1),
            "ApicIdCoreIdSize is the package shift"
        );
    }
}

#[test]
fn leaf_8000001e_amd_extended_apic() {
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

    // Check CPU 0
    let cpuid0 = generate_cpuid(
        kvm.get_supported_cpuid(kvm_bindings::KVM_MAX_CPUID_ENTRIES)
            .unwrap()
            .as_slice(),
        &topo,
        0,
        false,
    );
    let leaf0 = cpuid0.iter().find(|e| e.function == 0x8000_001e);
    if let Some(entry) = leaf0 {
        assert_eq!(entry.eax, apic_id(&topo, 0), "EAX = extended APIC ID");
        assert_eq!(entry.ebx & 0xff, 0, "core ID for cpu 0 should be 0");
        assert_eq!(
            (entry.ebx >> 8) & 0xff,
            1,
            "threads per core - 1 should be 1"
        );
        // Single NUMA: node_id=0, nodes_per_processor-1=0
        assert_eq!(entry.ecx & 0xff, 0, "single node: node ID = 0");
        assert_eq!(
            (entry.ecx >> 8) & 0x7,
            0,
            "single node: nodes per proc - 1 = 0"
        );
        assert_eq!(entry.edx, 0, "EDX reserved");
    }

    // Check CPU 3 (LLC 0, core 1, thread 1)
    let cpuid3 = generate_cpuid(
        kvm.get_supported_cpuid(kvm_bindings::KVM_MAX_CPUID_ENTRIES)
            .unwrap()
            .as_slice(),
        &topo,
        3,
        false,
    );
    let leaf3 = cpuid3.iter().find(|e| e.function == 0x8000_001e);
    if let Some(entry) = leaf3 {
        assert_eq!(entry.eax, apic_id(&topo, 3), "EAX = extended APIC ID");
        assert_eq!(entry.ebx & 0xff, 1, "core ID for cpu 3 should be 1");
    }
}

#[test]
fn leaf_8000001e_multi_numa_node_id() {
    let kvm = match kvm_ioctls::Kvm::new() {
        Ok(k) => k,
        Err(_) => return,
    };
    // 4 LLCs, 2 NUMA nodes -> LLCs 0,1 in node 0; LLCs 2,3 in node 1
    let topo = Topology {
        llcs: 4,
        cores_per_llc: 2,
        threads_per_core: 2,
        numa_nodes: 2,
        nodes: None,
        distances: None,
    };
    let base = kvm
        .get_supported_cpuid(kvm_bindings::KVM_MAX_CPUID_ENTRIES)
        .unwrap();
    let vendor = detect_vendor(base.as_slice());
    if vendor != CpuVendor::Amd {
        return;
    }

    let cpus_per_llc = topo.cores_per_llc * topo.threads_per_core;
    // CPU 0: LLC 0, NUMA node 0
    let cpuid0 = generate_cpuid(base.as_slice(), &topo, 0, false);
    let leaf0 = cpuid0.iter().find(|e| e.function == 0x8000_001e);
    if let Some(entry) = leaf0 {
        assert_eq!(entry.ecx & 0xff, 0, "cpu 0: node ID = 0");
        assert_eq!((entry.ecx >> 8) & 0x7, 1, "nodes per processor - 1 = 1");
    }

    // First CPU in LLC 2: NUMA node 1
    let cpu_in_node1 = 2 * cpus_per_llc;
    let cpuid1 = generate_cpuid(base.as_slice(), &topo, cpu_in_node1, false);
    let leaf1 = cpuid1.iter().find(|e| e.function == 0x8000_001e);
    if let Some(entry) = leaf1 {
        assert_eq!(entry.ecx & 0xff, 1, "cpu {cpu_in_node1}: node ID = 1");
        assert_eq!((entry.ecx >> 8) & 0x7, 1, "nodes per processor - 1 = 1");
    }
}

#[test]
fn leaf_80000008_single_cpu() {
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
    let leaf = cpuid.iter().find(|e| e.function == 0x8000_0008);
    if let Some(entry) = leaf {
        // Single CPU: ECX should be 0
        assert_eq!(entry.ecx & 0xff, 0, "single CPU: NC = 0");
        assert_eq!((entry.ecx >> 12) & 0xf, 0, "single CPU: ApicIdSize = 0");
    }
}

#[test]
fn leaf1f_matches_leaf0b() {
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

    // For subleaves 0 and 1, leaf 0x1F should produce the same topology
    // data as leaf 0xB (EAX, EBX, EDX match; ECX may differ only in type encoding)
    for sub in 0..2 {
        let leaf_b = cpuid.iter().find(|e| e.function == 0xb && e.index == sub);
        let leaf_1f = cpuid.iter().find(|e| e.function == 0x1f && e.index == sub);
        if let (Some(b), Some(f)) = (leaf_b, leaf_1f) {
            assert_eq!(b.eax, f.eax, "subleaf {sub}: EAX should match");
            assert_eq!(b.ebx, f.ebx, "subleaf {sub}: EBX should match");
            assert_eq!(b.edx, f.edx, "subleaf {sub}: EDX should match");
            assert_eq!(b.ecx, f.ecx, "subleaf {sub}: ECX should match");
        }
    }
}

#[test]
fn leaf1_htt_not_set_for_single_cpu() {
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
        // HTT bit should not be forcibly set when threads_per_pkg == 1.
        // KVM may still set it in its supported CPUID, but we should not
        // add it when unnecessary.
        let threads_per_pkg = (entry.ebx >> 16) & 0xff;
        assert_eq!(threads_per_pkg, 1, "single CPU: threads per pkg = 1");
    }
}

#[test]
fn leaf_80000001_cmplegacy_and_topoext_multi_cpu() {
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
    let leaf = cpuid.iter().find(|e| e.function == 0x8000_0001);
    if let Some(entry) = leaf {
        assert_ne!(entry.ecx & (1 << 1), 0, "CmpLegacy (bit 1) should be set");
        assert_ne!(
            entry.ecx & (1 << 22),
            0,
            "TopologyExtensions (bit 22) should be set"
        );
    }
}

#[test]
fn leaf_80000001_not_set_for_single_cpu() {
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
    let leaf = cpuid.iter().find(|e| e.function == 0x8000_0001);
    if let Some(entry) = leaf {
        // We should not forcibly set CmpLegacy or TopologyExtensions
        // when there is only one logical CPU in the package.
        // KVM may set these in its supported CPUID on AMD hosts,
        // but our code should not add them for single-CPU topologies.
        let our_bits = (1u32 << 1) | (1u32 << 22);
        // Get the host baseline to compare
        let host_cpuid = kvm
            .get_supported_cpuid(kvm_bindings::KVM_MAX_CPUID_ENTRIES)
            .expect("get_supported_cpuid");
        let host_leaf = host_cpuid
            .as_slice()
            .iter()
            .find(|e| e.function == 0x8000_0001);
        if let Some(host_entry) = host_leaf {
            // Our code should not have added bits beyond what the host provides
            let added = entry.ecx & !host_entry.ecx;
            assert_eq!(
                added & our_bits,
                0,
                "single CPU: should not add CmpLegacy or TopologyExtensions"
            );
        }
    }
}

#[test]
fn leaf_80000008_apic_id_size_representative() {
    let kvm = match kvm_ioctls::Kvm::new() {
        Ok(k) => k,
        Err(_) => return,
    };
    let topos = [
        (1, 1, 1),   // degenerate single CPU (ECX=0 path)
        (2, 1, 1),   // minimal multi-LLC
        (3, 3, 1),   // odd non-power-of-2
        (1, 1, 2),   // minimal SMT
        (2, 4, 2),   // standard multi-LLC with SMT
        (7, 5, 3),   // all dimensions non-power-of-2
        (15, 16, 1), // large scale no SMT
        (14, 9, 2),  // large with SMT, max APIC > 255
        (1, 64, 1),  // EAX[31:26] boundary (bits_needed(64)=6)
        (1, 18, 1),  // non-power-of-2 large threads_per_llc
    ];
    for (llcs, cores, threads) in topos {
        let topo = Topology {
            llcs,
            cores_per_llc: cores,
            threads_per_core: threads,
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
        let leaf = cpuid.iter().find(|e| e.function == 0x8000_0008);
        if let Some(entry) = leaf {
            let total_cpus = llcs * cores * threads;
            let apic_id_size = (entry.ecx >> 12) & 0xf;
            let nc = entry.ecx & 0xff;

            if total_cpus > 1 {
                assert!(
                    (1u32 << apic_id_size) >= total_cpus,
                    "{llcs}l/{cores}c/{threads}t: ApicIdSize {apic_id_size} too small \
                     for {total_cpus} cpus (2^{apic_id_size} = {})",
                    1u32 << apic_id_size
                );
                assert_eq!(
                    nc,
                    (total_cpus - 1) & 0xff,
                    "{llcs}l/{cores}c/{threads}t: NC should be total_cpus - 1"
                );
            } else {
                assert_eq!(
                    entry.ecx & 0xf0ff,
                    0,
                    "{llcs}l/{cores}c/{threads}t: single CPU ECX should be 0"
                );
            }
        }
    }
}

#[test]
fn detect_vendor_intel() {
    let entries = [kvm_cpuid_entry2 {
        function: 0,
        index: 0,
        flags: 0,
        eax: 0,
        ebx: 0x756e_6547, // "Genu"
        edx: 0x4965_6e69, // "ineI"
        ecx: 0x6c65_746e, // "ntel"
        ..Default::default()
    }];
    assert_eq!(detect_vendor(&entries), CpuVendor::Intel);
}

#[test]
fn detect_vendor_amd() {
    let entries = [kvm_cpuid_entry2 {
        function: 0,
        index: 0,
        flags: 0,
        eax: 0,
        ebx: 0x6874_7541, // "Auth"
        edx: 0x6974_6e65, // "enti"
        ecx: 0x444d_4163, // "cAMD"
        ..Default::default()
    }];
    assert_eq!(detect_vendor(&entries), CpuVendor::Amd);
}

#[test]
fn detect_vendor_unknown() {
    let entries = [kvm_cpuid_entry2 {
        function: 0,
        index: 0,
        flags: 0,
        eax: 0,
        ebx: 0,
        edx: 0,
        ecx: 0,
        ..Default::default()
    }];
    assert_eq!(detect_vendor(&entries), CpuVendor::Unknown);
}

#[test]
fn detect_vendor_missing_leaf0() {
    let entries = [kvm_cpuid_entry2 {
        function: 1,
        index: 0,
        ..Default::default()
    }];
    assert_eq!(detect_vendor(&entries), CpuVendor::Unknown);
}

#[test]
fn detect_vendor_from_kvm() {
    let kvm = match kvm_ioctls::Kvm::new() {
        Ok(k) => k,
        Err(_) => return,
    };
    let cpuid = kvm
        .get_supported_cpuid(kvm_bindings::KVM_MAX_CPUID_ENTRIES)
        .expect("get_supported_cpuid");
    let vendor = detect_vendor(cpuid.as_slice());
    assert_ne!(vendor, CpuVendor::Unknown, "host should be Intel or AMD");
}

#[test]
fn brand_string_not_clobbered() {
    let kvm = match kvm_ioctls::Kvm::new() {
        Ok(k) => k,
        Err(_) => return,
    };
    let host_cpuid = kvm
        .get_supported_cpuid(kvm_bindings::KVM_MAX_CPUID_ENTRIES)
        .expect("get_supported_cpuid");
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
    // Brand string leaves 0x80000002-0x80000004 should match host
    for leaf_fn in [0x8000_0002u32, 0x8000_0003, 0x8000_0004] {
        let host_leaf = host_cpuid.as_slice().iter().find(|e| e.function == leaf_fn);
        let guest_leaf = cpuid.iter().find(|e| e.function == leaf_fn);
        match (host_leaf, guest_leaf) {
            (Some(h), Some(g)) => {
                assert_eq!(
                    (h.eax, h.ebx, h.ecx, h.edx),
                    (g.eax, g.ebx, g.ecx, g.edx),
                    "brand string leaf {leaf_fn:#x} should pass through from host"
                );
            }
            (None, None) => {}
            _ => panic!(
                "leaf {leaf_fn:#x}: host has it = {}, guest has it = {}",
                host_leaf.is_some(),
                guest_leaf.is_some()
            ),
        }
    }
}

#[test]
fn vendor_conditional_leaf4_on_intel() {
    let kvm = match kvm_ioctls::Kvm::new() {
        Ok(k) => k,
        Err(_) => return,
    };
    let host_cpuid = kvm
        .get_supported_cpuid(kvm_bindings::KVM_MAX_CPUID_ENTRIES)
        .expect("get_supported_cpuid");
    let vendor = detect_vendor(host_cpuid.as_slice());
    if vendor != CpuVendor::Intel {
        return; // test only meaningful on Intel
    }
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
    // On Intel, leaf 0x4 should have been patched
    let l3 = cpuid
        .iter()
        .find(|e| e.function == 0x4 && ((e.eax >> 5) & 0x7) == 3);
    if let Some(entry) = l3 {
        let max_sharing = (entry.eax >> 14) & 0xfff;
        assert_eq!(
            max_sharing,
            (1u32 << core_shift(&topo)) - 1,
            "Intel leaf 0x4 L3 sharing should be patched"
        );
    }
}

#[test]
fn vendor_conditional_leaf8000001e_on_amd() {
    let kvm = match kvm_ioctls::Kvm::new() {
        Ok(k) => k,
        Err(_) => return,
    };
    let host_cpuid = kvm
        .get_supported_cpuid(kvm_bindings::KVM_MAX_CPUID_ENTRIES)
        .expect("get_supported_cpuid");
    let vendor = detect_vendor(host_cpuid.as_slice());
    if vendor != CpuVendor::Amd {
        return; // test only meaningful on AMD
    }
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
    let leaf = cpuid.iter().find(|e| e.function == 0x8000_001e);
    if let Some(entry) = leaf {
        assert_eq!(
            entry.eax,
            apic_id(&topo, 0),
            "AMD leaf 0x8000001E EAX should be patched"
        );
    }
}

#[test]
fn vendor_conditional_leaf8000001d_on_amd() {
    let kvm = match kvm_ioctls::Kvm::new() {
        Ok(k) => k,
        Err(_) => return,
    };
    let host_cpuid = kvm
        .get_supported_cpuid(kvm_bindings::KVM_MAX_CPUID_ENTRIES)
        .expect("get_supported_cpuid");
    let vendor = detect_vendor(host_cpuid.as_slice());
    if vendor != CpuVendor::Amd {
        return;
    }
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
    let entry = cpuid
        .iter()
        .find(|e| e.function == 0x8000_001d && ((e.eax >> 5) & 0x7) == 3)
        .expect("synthesized AMD L3 subleaf (type=3) must be present");
    let max_sharing = (entry.eax >> 14) & 0xfff;
    assert_eq!(
        max_sharing,
        (1u32 << core_shift(&topo)) - 1,
        "AMD leaf 0x8000001D L3 sharing spans the LLC (core_shift scope)"
    );
    // EAX[31:26] (num_cores) is left 0 on 0x8000001D, matching qemu's
    // encode_cache_cpuid8000001d (only the Intel leaf 0x4 encodes it;
    // the guest kernel reads only num_threads_sharing[25:14] for llc_id).
    assert_eq!(
        (entry.eax >> 26) & 0x3f,
        0,
        "0x8000001D leaves EAX[31:26] num_cores = 0 (qemu parity)"
    );
}

#[test]
fn leaf8000001d_l1_l2_sharing_per_core() {
    let kvm = match kvm_ioctls::Kvm::new() {
        Ok(k) => k,
        Err(_) => return,
    };
    let host_cpuid = kvm
        .get_supported_cpuid(kvm_bindings::KVM_MAX_CPUID_ENTRIES)
        .expect("get_supported_cpuid");
    let vendor = detect_vendor(host_cpuid.as_slice());
    if vendor != CpuVendor::Amd {
        return;
    }
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
    // L1 and L2 should share at core level (SMT siblings only)
    for level in [1u32, 2] {
        let entry = cpuid
            .iter()
            .find(|e| e.function == 0x8000_001d && ((e.eax >> 5) & 0x7) == level)
            .unwrap_or_else(|| panic!("synthesized AMD cache subleaf for level {level}"));
        let max_sharing = (entry.eax >> 14) & 0xfff;
        assert_eq!(
            max_sharing,
            (1u32 << smt_shift(&topo)) - 1,
            "AMD leaf 0x8000001D L{level} sharing is per-core (SMT level)"
        );
    }
}

#[test]
fn leaf8000001d_cache_ids_differ_across_llcs() {
    let kvm = match kvm_ioctls::Kvm::new() {
        Ok(k) => k,
        Err(_) => return,
    };
    let host_cpuid = kvm
        .get_supported_cpuid(kvm_bindings::KVM_MAX_CPUID_ENTRIES)
        .expect("get_supported_cpuid");
    let vendor = detect_vendor(host_cpuid.as_slice());
    if vendor != CpuVendor::Amd {
        return;
    }
    let topo = Topology {
        llcs: 2,
        cores_per_llc: 4,
        threads_per_core: 1,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    // Generate CPUID for cpu 0 (LLC 0) and cpu 4 (LLC 1)
    let cpuid0 = generate_cpuid(
        kvm.get_supported_cpuid(kvm_bindings::KVM_MAX_CPUID_ENTRIES)
            .unwrap()
            .as_slice(),
        &topo,
        0,
        false,
    );
    let cpuid4 = generate_cpuid(
        kvm.get_supported_cpuid(kvm_bindings::KVM_MAX_CPUID_ENTRIES)
            .unwrap()
            .as_slice(),
        &topo,
        4,
        false,
    );
    let e0 = cpuid0
        .iter()
        .find(|e| e.function == 0x8000_001d && ((e.eax >> 5) & 0x7) == 3)
        .expect("synthesized AMD L3 subleaf for cpu 0");
    let e4 = cpuid4
        .iter()
        .find(|e| e.function == 0x8000_001d && ((e.eax >> 5) & 0x7) == 3)
        .expect("synthesized AMD L3 subleaf for cpu 4");
    {
        // Both should have the same sharing field (per-LLC scope)
        let sharing0 = (e0.eax >> 14) & 0xfff;
        let sharing4 = (e4.eax >> 14) & 0xfff;
        assert_eq!(sharing0, sharing4, "L3 sharing field should be identical");
        // Verify that get_cache_id would produce different IDs
        // by simulating the kernel's apicid >> get_count_order(n)
        // where get_count_order(n) = fls(n - 1) for n > 1.
        let num_threads_sharing = sharing0 + 1;
        let index_msb = 32 - (num_threads_sharing - 1).leading_zeros();
        let cache_id_0 = apic_id(&topo, 0) >> index_msb;
        let cache_id_4 = apic_id(&topo, 4) >> index_msb;
        assert_ne!(
            cache_id_0,
            cache_id_4,
            "CPUs in different LLCs should have different L3 cache IDs \
             (apic0={}, apic4={}, shift={index_msb})",
            apic_id(&topo, 0),
            apic_id(&topo, 4),
        );
    }
}

/// For multi-LLC topologies, verify that CPUs in different LLCs
/// produce different L3 cache IDs via the kernel's get_cache_id formula
/// (apicid >> get_count_order(sharing+1)). This invariant makes
/// from_system().split_by_llc() return the correct LLC count in guests.
#[test]
fn cache_ids_distinct_per_llc_representative() {
    let kvm = match kvm_ioctls::Kvm::new() {
        Ok(k) => k,
        Err(_) => return,
    };
    let host_cpuid = kvm
        .get_supported_cpuid(kvm_bindings::KVM_MAX_CPUID_ENTRIES)
        .expect("get_supported_cpuid");
    let vendor = detect_vendor(host_cpuid.as_slice());

    // The cache leaf depends on vendor
    let cache_leaf: u32 = match vendor {
        CpuVendor::Intel => 0x4,
        CpuVendor::Amd => 0x8000_001d,
        CpuVendor::Unknown => return,
    };

    // Multi-LLC only — single-LLC has no cross-LLC invariant.
    let topos = [
        (2, 1, 1),   // minimal multi-LLC
        (3, 3, 1),   // odd non-power-of-2
        (2, 4, 2),   // standard multi-LLC with SMT
        (7, 5, 3),   // all dimensions non-power-of-2
        (5, 3, 2),   // prime LLCs, odd cores, SMT
        (15, 16, 1), // large scale no SMT
        (14, 9, 2),  // large with SMT, max APIC > 255
    ];
    for (llcs, cores, threads) in topos {
        let topo = Topology {
            llcs,
            cores_per_llc: cores,
            threads_per_core: threads,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        };
        let cpus_per_llc = cores * threads;

        // Get L3 sharing from CPU 0's CPUID
        let cpuid0 = generate_cpuid(host_cpuid.as_slice(), &topo, 0, false);
        let l3 = cpuid0
            .iter()
            .find(|e| e.function == cache_leaf && ((e.eax >> 5) & 0x7) == 3);
        let Some(l3_entry) = l3 else { continue };

        let sharing = (l3_entry.eax >> 14) & 0xfff;
        let num_threads_sharing = sharing + 1;
        let index_msb = 32 - (num_threads_sharing - 1).leading_zeros();

        // Compute cache ID for the first CPU in each LLC
        let mut cache_ids: std::collections::HashSet<u32> = std::collections::HashSet::new();
        for l in 0..llcs {
            let cpu = l * cpus_per_llc;
            let apic = apic_id(&topo, cpu);
            let cache_id = apic >> index_msb;
            cache_ids.insert(cache_id);
        }
        assert_eq!(
            cache_ids.len(),
            llcs as usize,
            "{llcs}l/{cores}c/{threads}t: expected {llcs} distinct L3 cache IDs, \
             got {} (shift={index_msb})",
            cache_ids.len(),
        );
    }
}

/// Host-independent (synthetic base): the synthesized 0xB Core
/// subleaf (index 1) must EXIST — the exact regression this fix prevents
/// (an absent Core subleaf collapses every CPU into its own package).
/// Also asserts exactly one entry per (function, index) and the vendor
/// 0x1F split (Intel emits it, AMD does not).
#[test]
fn leaf0b_core_subleaf_emitted_both_vendors() {
    // (label, leaf-0 ebx, edx, ecx) — the vendor string.
    for (vendor, ebx, edx, ecx) in [
        ("intel", 0x756e_6547u32, 0x4965_6e69u32, 0x6c65_746eu32),
        ("amd", 0x6874_7541u32, 0x6974_6e65u32, 0x444d_4163u32),
    ] {
        let base = vec![kvm_cpuid_entry2 {
            function: 0,
            ebx,
            edx,
            ecx,
            ..Default::default()
        }];
        let topo = Topology {
            llcs: 2,
            cores_per_llc: 4,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        };
        let cpuid = generate_cpuid(&base, &topo, 0, false);
        // The Core (index 1) subleaf MUST be present (not vacuous).
        let core_sub = cpuid
            .iter()
            .find(|e| e.function == 0xb && e.index == 1)
            .unwrap_or_else(|| {
                panic!("{vendor}: 0xB Core subleaf (index 1) must be synthesized")
            });
        assert_eq!(
            (core_sub.ecx >> 8) & 0xff,
            2,
            "{vendor}: 0xB index 1 must be the Core level (type 2)"
        );
        // Exactly one entry per (function, index) — no host subleaf leaks.
        for idx in 0..3u32 {
            let n = cpuid
                .iter()
                .filter(|e| e.function == 0xb && e.index == idx)
                .count();
            assert_eq!(
                n, 1,
                "{vendor}: exactly one 0xB subleaf at index {idx}, got {n}"
            );
        }
        // 0x1F is emitted for Intel guests only; AMD falls through to 0xB.
        let has_1f = cpuid.iter().any(|e| e.function == 0x1f);
        if vendor == "intel" {
            assert!(has_1f, "intel: 0x1F must be emitted");
            assert!(
                cpuid.iter().any(|e| e.function == 0x1f && e.index == 1),
                "intel: 0x1F Core subleaf (index 1) must be present"
            );
        } else {
            assert!(!has_1f, "amd: 0x1F must NOT be emitted");
        }
    }
}

/// Host-independence (synthetic base): on a forced-AMD base that
/// OMITS 0x8000001D and masks the 0x80000006 L3-size field to 0,
/// generate_cpuid must still emit a full L1/L2/L3 chain + a nonzero
/// L3-detection gate, so the guest's llc_id is correct regardless of the
/// host CPUID. (The 0x80000006-absent case is covered separately by
/// leaf_80000006_synthesized_when_absent.)
#[test]
fn amd_cache_leaves_synthesized_host_independent() {
    let base = vec![
        kvm_cpuid_entry2 {
            function: 0,
            ebx: 0x6874_7541, // "Auth"
            edx: 0x6974_6e65, // "enti"
            ecx: 0x444d_4163, // "cAMD"
            ..Default::default()
        },
        // Host 0x80000006 present but the L3-size field masked to 0.
        kvm_cpuid_entry2 {
            function: 0x8000_0006,
            ..Default::default()
        },
        // No 0x8000001D in the base — the host omits the cache chain.
    ];
    let topo = Topology {
        llcs: 2,
        cores_per_llc: 4,
        threads_per_core: 1,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    let cpuid = generate_cpuid(&base, &topo, 0, false);
    // L1d(0), L1i(1), L2(2), L3(3) chain must be emitted in order.
    let chain: Vec<(u32, u32)> = (0..4)
        .map(|i| {
            let e = cpuid
                .iter()
                .find(|e| e.function == 0x8000_001d && e.index == i)
                .unwrap_or_else(|| panic!("0x8000001D subleaf {i} must be synthesized"));
            (e.eax & 0x1f, (e.eax >> 5) & 0x7) // (type, level)
        })
        .collect();
    assert_eq!(
        chain,
        vec![(1, 1), (2, 1), (3, 2), (3, 3)],
        "0x8000001D must emit L1d, L1i, L2, L3 (type,level) in order"
    );
    // Index 4 is the type-0 terminator; L3 (index 3) is the highest
    // non-null subleaf the kernel's find_num_cache_leaves stops after.
    let term = cpuid
        .iter()
        .find(|e| e.function == 0x8000_001d && e.index == 4)
        .expect("0x8000001D terminator subleaf");
    assert_eq!(
        term.eax & 0x1f,
        0,
        "0x8000001D index 4 must be the type-0 terminator"
    );
    // L3 num_threads_sharing spans the LLC (the field driving llc_id).
    let l3 = cpuid
        .iter()
        .find(|e| e.function == 0x8000_001d && e.index == 3)
        .unwrap();
    assert_eq!(
        (l3.eax >> 14) & 0xfff,
        (1u32 << core_shift(&topo)) - 1,
        "L3 num_threads_sharing must span the LLC (core_shift scope)"
    );
    // The L3-detection gate: 0x80000006 EDX must be nonzero despite the
    // host reporting 0 (cpuid_amd_hygon_has_l3_cache checks EDX != 0).
    let l6 = cpuid
        .iter()
        .find(|e| e.function == 0x8000_0006)
        .expect("0x80000006 must be present");
    assert_ne!(
        l6.edx, 0,
        "0x80000006 EDX (the AMD L3-detection gate) must be synthesized nonzero"
    );
}

/// qemu-parity guard (host-independent, synthetic AMD base): pin the EDX
/// cache-property flags the synthesized 0x8000001D subleaves carry. These
/// mirror qemu's `legacy_amd_cache_info` (target/i386/cpu.c): L1d/L1i =
/// no-write-invalidate (0x1), L2 = none (0x0), L3 = inclusive|complex
/// (0x6). The kernel reads only EAX/EBX/ECX from 0x8000001D, so EDX is
/// informational — but a regression in these flag bytes would silently
/// diverge from qemu's emitted CPUID. Asserting the subleaf ORDER (index
/// 0=L1d, 1=L1i, 2=L2, 3=L3, by type/level) catches a reorder that would
/// mis-attribute the flags. (EAX/EBX/ECX geometry and the chain order are
/// covered by amd_cache_leaves_synthesized_host_independent.)
#[test]
fn amd_cache_0x8000001d_edx_flags_pinned() {
    let base = vec![kvm_cpuid_entry2 {
        function: 0,
        ebx: 0x6874_7541, // "Auth"
        edx: 0x6974_6e65, // "enti"
        ecx: 0x444d_4163, // "cAMD"
        ..Default::default()
    }];
    let topo = Topology {
        llcs: 2,
        cores_per_llc: 4,
        threads_per_core: 2,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    let cpuid = generate_cpuid(&base, &topo, 0, false);
    // (type, level, edx) per subleaf index 0..3 — type from EAX[4:0],
    // level from EAX[7:5], edx = the cache-property flags.
    let subleaves: Vec<(u32, u32, u32)> = (0..4)
        .map(|i| {
            let e = cpuid
                .iter()
                .find(|e| e.function == 0x8000_001d && e.index == i)
                .unwrap_or_else(|| panic!("0x8000001D subleaf {i} must be synthesized"));
            (e.eax & 0x1f, (e.eax >> 5) & 0x7, e.edx)
        })
        .collect();
    assert_eq!(
        subleaves,
        vec![
            (1, 1, 0x1), // index 0: L1d, level 1, no-write-invalidate
            (2, 1, 0x1), // index 1: L1i, level 1, no-write-invalidate
            (3, 2, 0x0), // index 2: L2 unified, level 2, no flags
            (3, 3, 0x6), // index 3: L3 unified, level 3, inclusive|complex
        ],
        "0x8000001D subleaves must emit L1d/L1i/L2/L3 in order with qemu \
         legacy_amd_cache_info EDX flags (0x1/0x1/0x0/0x6)"
    );
}

/// Host-independence: on a forced-AMD base that OMITS 0x80000006
/// entirely (vs masking its EDX to 0), the synthesis must PUSH the leaf
/// so the L3-detection gate (cpuid_amd_hygon_has_l3_cache = EDX != 0)
/// still holds — symmetric with the 0x8000001D synthesis.
#[test]
fn leaf_80000006_synthesized_when_absent() {
    let base = vec![kvm_cpuid_entry2 {
        function: 0,
        ebx: 0x6874_7541, // "Auth"
        edx: 0x6974_6e65, // "enti"
        ecx: 0x444d_4163, // "cAMD"
        ..Default::default()
    }];
    let topo = Topology {
        llcs: 2,
        cores_per_llc: 4,
        threads_per_core: 1,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    let cpuid = generate_cpuid(&base, &topo, 0, false);
    let l6 = cpuid
        .iter()
        .find(|e| e.function == 0x8000_0006)
        .expect("0x80000006 must be synthesized when absent from the base");
    assert_ne!(
        l6.edx, 0,
        "0x80000006 EDX (the L3-detection gate) must be nonzero when synthesized"
    );
}

#[test]
fn max_apic_id_single_cpu() {
    let t = Topology {
        llcs: 1,
        cores_per_llc: 1,
        threads_per_core: 1,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    assert_eq!(max_apic_id(&t), 0);
}

#[test]
fn max_apic_id_equals_last_cpu() {
    let t = Topology {
        llcs: 2,
        cores_per_llc: 4,
        threads_per_core: 2,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    assert_eq!(max_apic_id(&t), apic_id(&t, t.total_cpus() - 1));
}

#[test]
fn max_apic_id_large_topology() {
    // 14 LLCs x 9 cores x 2 threads = 252 CPUs (near KVM vCPU limit)
    let t = Topology {
        llcs: 14,
        cores_per_llc: 9,
        threads_per_core: 2,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    // core_bits = bits_needed(9) = 4, thread_bits = 1
    // last cpu = 251: LLC 13, core 8, thread 1
    // apic_id = 13 << 5 | 8 << 1 | 1 = 433
    assert_eq!(max_apic_id(&t), 433);
    assert!(max_apic_id(&t) > 254);
}

#[test]
fn topology_single_thread_per_core() {
    let t = Topology {
        llcs: 4,
        cores_per_llc: 4,
        threads_per_core: 1,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    assert_eq!(smt_shift(&t), 0);
    // APIC IDs should still be unique
    let ids: Vec<u32> = (0..t.total_cpus()).map(|i| apic_id(&t, i)).collect();
    let unique: std::collections::HashSet<u32> = ids.iter().cloned().collect();
    assert_eq!(ids.len(), unique.len());
}

#[test]
fn topology_1x1x1() {
    let t = Topology {
        llcs: 1,
        cores_per_llc: 1,
        threads_per_core: 1,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    assert_eq!(t.total_cpus(), 1);
    assert_eq!(apic_id(&t, 0), 0);
    assert_eq!(max_apic_id(&t), 0);
}

#[test]
fn kvm_hints_realtime_set_in_performance_mode() {
    let kvm = match kvm_ioctls::Kvm::new() {
        Ok(k) => k,
        Err(_) => return,
    };
    let topo = Topology {
        llcs: 1,
        cores_per_llc: 2,
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
        true,
    );
    let leaf = cpuid.iter().find(|e| e.function == 0x4000_0001);
    assert!(
        leaf.is_some(),
        "leaf 0x40000001 should exist in performance_mode"
    );
    let entry = leaf.unwrap();
    assert_ne!(
        entry.edx & 1,
        0,
        "KVM_HINTS_REALTIME (EDX bit 0) should be set"
    );

    // Max hypervisor leaf must advertise 0x40000001 so the guest
    // enumerates the hints leaf.
    let leaf40 = cpuid
        .iter()
        .find(|e| e.function == 0x4000_0000)
        .expect("leaf 0x40000000 should exist");
    assert!(
        leaf40.eax >= 0x4000_0001,
        "0x40000000.EAX should be >= 0x40000001, got {:#x}",
        leaf40.eax,
    );
}

#[test]
fn kvm_hints_realtime_not_set_without_performance_mode() {
    let kvm = match kvm_ioctls::Kvm::new() {
        Ok(k) => k,
        Err(_) => return,
    };
    let topo = Topology {
        llcs: 1,
        cores_per_llc: 2,
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
    // Leaf 0x40000001 may or may not exist from KVM's base CPUID.
    // If it exists, EDX bit 0 must not be set by our code.
    // KVM's base leaf has EDX=0, so bit 0 should be clear.
    if let Some(entry) = cpuid.iter().find(|e| e.function == 0x4000_0001) {
        assert_eq!(
            entry.edx & 1,
            0,
            "KVM_HINTS_REALTIME should not be set without performance_mode"
        );
    }
}

#[test]
fn msi_ext_dest_id_set_for_wide_topology() {
    // A topology whose max APIC ID exceeds the xAPIC limit (>254) must
    // advertise KVM_FEATURE_MSI_EXT_DEST_ID (0x40000001 EAX bit 15) so
    // the guest raises its APIC limit and can address CPUs above 255.
    let base = vec![
        kvm_cpuid_entry2 {
            function: 0,
            ebx: 0x756e_6547, // "Genu"
            edx: 0x4965_6e69, // "ineI"
            ecx: 0x6c65_746e, // "ntel"
            ..Default::default()
        },
        kvm_cpuid_entry2 {
            function: 0x4000_0000,
            eax: 0x4000_0000, // max leaf = 0x40000000 initially
            ..Default::default()
        },
        kvm_cpuid_entry2 {
            function: 0x4000_0001,
            ..Default::default()
        },
    ];
    let wide = Topology {
        llcs: 16,
        cores_per_llc: 16,
        threads_per_core: 2,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    // Precondition: this topology actually exceeds the xAPIC limit.
    assert!(
        max_apic_id(&wide) > crate::vmm::x86_64::kvm::MAX_XAPIC_ID,
        "test topology must exceed the xAPIC limit; max_apic_id={}",
        max_apic_id(&wide),
    );
    let cpuid = generate_cpuid(&base, &wide, 0, false);
    let entry = cpuid
        .iter()
        .find(|e| e.function == 0x4000_0001)
        .expect("leaf 0x40000001 should exist");
    assert_ne!(
        entry.eax & (1 << 15),
        0,
        "MSI_EXT_DEST_ID (EAX bit 15) must be set for >254 APIC IDs"
    );
    // The guest enumerates 0x40000001 only if 0x40000000 advertises it.
    let leaf40 = cpuid
        .iter()
        .find(|e| e.function == 0x4000_0000)
        .expect("leaf 0x40000000 should exist");
    assert!(
        leaf40.eax >= 0x4000_0001,
        "0x40000000.EAX must advertise 0x40000001, got {:#x}",
        leaf40.eax,
    );
}

#[test]
fn msi_ext_dest_id_absent_for_narrow_topology() {
    // A topology within the xAPIC limit (<=254) must NOT advertise
    // MSI_EXT_DEST_ID — its 0x40000001 EAX is unchanged from the base.
    let base = vec![
        kvm_cpuid_entry2 {
            function: 0,
            ebx: 0x756e_6547,
            edx: 0x4965_6e69,
            ecx: 0x6c65_746e,
            ..Default::default()
        },
        kvm_cpuid_entry2 {
            function: 0x4000_0000,
            eax: 0x4000_0000,
            ..Default::default()
        },
        kvm_cpuid_entry2 {
            function: 0x4000_0001,
            ..Default::default()
        },
    ];
    let narrow = Topology {
        llcs: 1,
        cores_per_llc: 2,
        threads_per_core: 1,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    assert!(max_apic_id(&narrow) <= crate::vmm::x86_64::kvm::MAX_XAPIC_ID);
    let cpuid = generate_cpuid(&base, &narrow, 0, false);
    let entry = cpuid
        .iter()
        .find(|e| e.function == 0x4000_0001)
        .expect("leaf 0x40000001 should exist");
    assert_eq!(
        entry.eax & (1 << 15),
        0,
        "MSI_EXT_DEST_ID must not be set for <=254 APIC IDs"
    );
}

#[test]
fn kvm_hints_realtime_preserves_other_edx_bits() {
    // Synthetic test: if the base CPUID has other EDX bits set in
    // leaf 0x40000001, performance_mode should OR bit 0, not replace.
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
            eax: 0x4000_0000, // max leaf = 0x40000000 initially
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
            edx: 0xdead_0000, // hypothetical other bits
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
    let entry = cpuid
        .iter()
        .find(|e| e.function == 0x4000_0001)
        .expect("leaf 0x40000001 should exist");
    assert_eq!(
        entry.edx, 0xdead_0001,
        "should OR bit 0 into existing EDX, not replace"
    );

    // Max hypervisor leaf bumped from 0x40000000 to 0x40000001.
    let leaf40 = cpuid
        .iter()
        .find(|e| e.function == 0x4000_0000)
        .expect("leaf 0x40000000 should exist");
    assert_eq!(
        leaf40.eax, 0x4000_0001,
        "0x40000000.EAX should be bumped to 0x40000001"
    );

    // Without performance_mode, EDX should be untouched.
    let cpuid_no_perf = generate_cpuid(&base, &topo, 0, false);
    let entry_no_perf = cpuid_no_perf
        .iter()
        .find(|e| e.function == 0x4000_0001)
        .expect("leaf should still exist");
    assert_eq!(
        entry_no_perf.edx, 0xdead_0000,
        "without performance_mode, EDX should be unchanged"
    );
}
