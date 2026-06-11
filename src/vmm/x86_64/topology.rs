use kvm_bindings::{KVM_CPUID_FLAG_SIGNIFCANT_INDEX, kvm_cpuid_entry2};

use crate::vmm::topology::Topology;

// ---- Leaf 0xA: Architectural Performance Monitoring (Intel) ----
//
// Field layout per arch/x86/include/asm/perf_event.h:
//   union cpuid10_eax { version_id:8, num_counters:8, bit_width:8, mask_length:8 }
//   union cpuid10_edx { num_counters_fixed:5, bit_width_fixed:8, reserved1:2, anythread_deprecated:1, reserved2:16 }
//
// We synthesize a conservative PMU v2 surface so guest sched_ext
// schedulers (scx_layered, scx_cosmos) get usable perf counters
// regardless of host hardware. KVM's intel_pmu_refresh
// (arch/x86/kvm/vmx/pmu_intel.c) clamps these against the host's
// actual PMU capabilities, so the guest sees min(synthesized, host).

/// PMU architectural version reported in `EAX\[7:0\]`. Version 2
/// matches Intel SDM "Architectural Performance Monitoring v2".
const PMU_ARCH_PERFMON_VERSION: u32 = 2;

/// Number of general-purpose counters per logical CPU, `EAX\[15:8\]`.
/// Conservative — Intel hardware ranges from 4 to 8.
const PMU_NUM_GP_COUNTERS: u32 = 4;

/// General-purpose counter bit width, `EAX\[23:16\]`. PMU v2 spec value.
const PMU_GP_COUNTER_WIDTH: u32 = 48;

/// Bit-vector length for the unsupported-event mask, `EAX\[31:24\]`.
/// Must equal ARCH_PERFMON_EVENTS_COUNT (7) from
/// arch/x86/include/asm/perf_event.h or intel_pmu_init in
/// arch/x86/events/intel/core.c returns -ENODEV.
const PMU_EVENT_MASK_LENGTH: u32 = 7;

/// Number of fixed-function counters, `EDX\[4:0\]`. PMU v2 surface = 3
/// (instructions, cycles, ref-cycles).
const PMU_NUM_FIXED_COUNTERS: u32 = 3;

/// Fixed-function counter bit width, `EDX\[12:5\]`. PMU v2 spec value.
const PMU_FIXED_COUNTER_WIDTH: u32 = 48;

/// CPU vendor, detected from CPUID leaf 0x0 EBX:EDX:ECX.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuVendor {
    Intel,
    Amd,
    Unknown,
}

/// Detect CPU vendor from leaf 0x0 in the given CPUID entries.
/// Vendor string is encoded across EBX:EDX:ECX (note: not EBX:ECX:EDX).
fn detect_vendor(entries: &[kvm_cpuid_entry2]) -> CpuVendor {
    let leaf0 = entries.iter().find(|e| e.function == 0 && e.index == 0);
    match leaf0 {
        Some(e) => {
            // "GenuineIntel" = EBX:0x756e6547 EDX:0x49656e69 ECX:0x6c65746e
            // "AuthenticAMD" = EBX:0x68747541 EDX:0x69746e65 ECX:0x444d4163
            match (e.ebx, e.edx, e.ecx) {
                (0x756e_6547, 0x4965_6e69, 0x6c65_746e) => CpuVendor::Intel,
                (0x6874_7541, 0x6974_6e65, 0x444d_4163) => CpuVendor::Amd,
                _ => CpuVendor::Unknown,
            }
        }
        None => CpuVendor::Unknown,
    }
}

/// Minimum number of bits to represent values 0..n-1.
/// Returns 0 for n <= 1.
fn bits_needed(n: u32) -> u32 {
    if n <= 1 {
        return 0;
    }
    32 - (n - 1).leading_zeros()
}

/// Compute the x2APIC ID for a logical CPU.
/// Encoding: llc_id << (core_bits + thread_bits) | core_id << thread_bits | thread_id
pub fn apic_id(topo: &Topology, cpu_id: u32) -> u32 {
    let (llc_id, core_id, thread_id) = topo.decompose(cpu_id);
    let thread_bits = bits_needed(topo.threads_per_core);
    let core_bits = bits_needed(topo.cores_per_llc);
    (llc_id << (core_bits + thread_bits)) | (core_id << thread_bits) | thread_id
}

/// Highest APIC ID across all logical CPUs in this topology.
pub fn max_apic_id(topo: &Topology) -> u32 {
    let total = topo.total_cpus();
    if total == 0 {
        return 0;
    }
    apic_id(topo, total - 1)
}

/// Number of bits needed to represent thread ID within a core.
pub fn smt_shift(topo: &Topology) -> u32 {
    bits_needed(topo.threads_per_core)
}

/// Number of bits needed to represent core+thread ID within an LLC.
pub fn core_shift(topo: &Topology) -> u32 {
    bits_needed(topo.threads_per_core) + bits_needed(topo.cores_per_llc)
}

/// Build one Extended-Topology (leaf 0xB/0x1F) subleaf entry.
/// `shift_to_next` is the x2APIC-ID right-shift that yields the next
/// level's id; `level_type` is 1=SMT, 2=Core, 0=invalid (terminator).
fn topo_subleaf(
    function: u32,
    index: u32,
    shift_to_next: u32,
    count: u32,
    level_type: u32,
    apic: u32,
) -> kvm_cpuid_entry2 {
    kvm_cpuid_entry2 {
        function,
        index,
        flags: KVM_CPUID_FLAG_SIGNIFCANT_INDEX,
        eax: shift_to_next,
        ebx: count & 0xffff,
        ecx: (level_type << 8) | (index & 0xff),
        edx: apic,
        ..Default::default()
    }
}

/// Patch cache topology fields in a CPUID EAX register (leaf 0x4 or 0x8000001D).
/// Sets `EAX\[25:14\]` (num_threads_sharing) and `EAX\[31:26\]` (num_cores_on_die)
/// based on the cache level and VM topology.
fn patch_cache_topology_eax(entry: &mut kvm_cpuid_entry2, smt: u32, core: u32, cores_per_llc: u32) {
    let cache_level = (entry.eax >> 5) & 0x7;
    let max_sharing = match cache_level {
        1 | 2 => (1u32 << smt).saturating_sub(1),
        3 => (1u32 << core).saturating_sub(1),
        _ => 0,
    };
    entry.eax = (entry.eax & 0xfc003fff) | ((max_sharing & 0xfff) << 14);
    let core_bits = bits_needed(cores_per_llc);
    let max_core_ids = (1u32 << core_bits).saturating_sub(1);
    entry.eax = (entry.eax & 0x03ffffff) | ((max_core_ids & 0x3f) << 26);
}

/// AMD cache geometry for the synthesized cache leaves (0x8000001D and
/// 0x80000006), modeled on qemu's `legacy_amd_cache_info`
/// (target/i386/cpu.c): 64-byte lines, 1 partition per cache. The guest
/// kernel derives `llc_id` solely from `num_threads_sharing` (computed
/// per-level from the VM topology), via
/// `get_cache_id = apicid >> order(num_threads_sharing + 1)`, so these
/// sizes/associativities are informational — they shape only the cache
/// sizes the guest reports in `/sys`, not the LLC grouping.
const CACHE_LINE_SIZE: u32 = 64;
const L1_CACHE_SIZE_KIB: u32 = 64;
const L1_CACHE_WAYS: u32 = 2;
const L2_CACHE_SIZE_KIB: u32 = 512;
const L2_CACHE_WAYS: u32 = 16;
const L3_CACHE_SIZE_KIB: u32 = 16 * 1024;
const L3_CACHE_WAYS: u32 = 16;
/// AMD 0x80000006 ECX/EDX associativity encoding for 16-way (qemu
/// `X86_ENC_ASSOC` / AMD APM): 16-way -> 0x8.
const ASSOC_ENC_16WAY: u32 = 0x8;
/// Leaf 0x80000006 ECX (L2) / EDX (L3) descriptors, matching the
/// 0x8000001D L2/L3 geometry. EDX\[31:18\]=L3 size/512KiB nonzero is the AMD
/// L3-detection gate (cpuid_amd_hygon_has_l3_cache). ECX\[31:16\]=L2 size KiB.
const L80000006_ECX: u32 =
    (L2_CACHE_SIZE_KIB << 16) | (ASSOC_ENC_16WAY << 12) | (1 << 8) | CACHE_LINE_SIZE;
const L80000006_EDX: u32 =
    ((L3_CACHE_SIZE_KIB / 512) << 18) | (ASSOC_ENC_16WAY << 12) | (1 << 8) | CACHE_LINE_SIZE;

/// Build one AMD cache-topology subleaf (leaf 0x8000001D), encoded per the
/// AMD APM / qemu `encode_cache_cpuid8000001d`. `cache_type` is 1=data,
/// 2=instruction, 3=unified. `num_threads_sharing` is the count MINUS ONE
/// of logical CPUs sharing this cache (EAX\[25:14\]) — the field the guest
/// kernel reads to compute llc_id. EAX\[31:26\] (num_cores) and EAX\[9\]
/// (fully-associative) are left 0, matching qemu's 0x8000001D (only the
/// Intel leaf 0x4 encodes num_cores). EBX\[11:0\]=line-1, \[21:12\]=partitions-1
/// (0, one partition), \[31:22\]=ways-1; ECX=sets-1; EDX=property flags.
#[allow(clippy::too_many_arguments)]
fn amd_cache_subleaf(
    index: u32,
    cache_type: u32,
    level: u32,
    self_init: bool,
    size_kib: u32,
    ways: u32,
    num_threads_sharing: u32,
    flags: u32,
) -> kvm_cpuid_entry2 {
    // size = line * ways * partitions(1) * sets  =>  sets = size / (line * ways)
    let sets = (size_kib * 1024) / (CACHE_LINE_SIZE * ways);
    let eax = cache_type
        | (level << 5)
        | (u32::from(self_init) << 8)
        | ((num_threads_sharing & 0xfff) << 14);
    let ebx = (CACHE_LINE_SIZE - 1) | ((ways - 1) << 22);
    kvm_cpuid_entry2 {
        function: 0x8000_001d,
        index,
        flags: KVM_CPUID_FLAG_SIGNIFCANT_INDEX,
        eax,
        ebx,
        ecx: sets - 1,
        edx: flags,
        ..Default::default()
    }
}

/// Generate CPUID entries for a specific vCPU with topology information.
/// Takes a pre-fetched base CPUID (from `get_supported_cpuid`) and patches
/// topology-related leaves. The base should be fetched once and reused for
/// all vCPUs — each call clones and patches per-vCPU fields (APIC ID etc).
///
/// Leaf 0x8000001E ECX is NUMA-aware: sets node ID from
/// `Topology::numa_node_of` and nodes-per-processor from `numa_nodes`.
///
/// When `performance_mode` is true, sets KVM_HINTS_REALTIME (CPUID leaf
/// 0x40000001 EDX bit 0). This disables PV spinlocks, PV TLB flush, and
/// PV sched_yield in the guest, and enables haltpoll cpuidle. PV spinlocks
/// require CONFIG_PARAVIRT_SPINLOCKS (not in ktstr.kconfig, so no-op for
/// ktstr guests).
pub fn generate_cpuid(
    base_cpuid: &[kvm_cpuid_entry2],
    topo: &Topology,
    cpu_id: u32,
    performance_mode: bool,
) -> Vec<kvm_cpuid_entry2> {
    let mut entries: Vec<kvm_cpuid_entry2> = base_cpuid.to_vec();

    let vendor = detect_vendor(&entries);
    let apic = apic_id(topo, cpu_id);
    let smt = smt_shift(topo);
    let core = core_shift(topo);
    // The whole machine is one package: the LLCs are sub-domains carved by
    // the cache leaf (0x4 / 0x8000001D), not separate packages. `pkg_shift`
    // is the APIC-ID width below the package, so `apic >> pkg_shift == 0`
    // for every CPU -> the guest kernel groups all CPUs into one package.
    // One multi-core package is the precondition for the kernel to build
    // multi-core sibling masks at all (has_mp); the LLC sub-domain within it
    // is then carved by llc_id from the cache leaf, not by the package id.
    let total_cpus = topo.total_cpus();
    let pkg_shift = bits_needed(max_apic_id(topo) + 1);

    for entry in entries.iter_mut() {
        match entry.function {
            // Leaf 0x1: Feature Information (vendor-independent)
            0x1 => {
                // EBX[31:24] = initial APIC ID (8-bit)
                entry.ebx = (entry.ebx & 0x00ffffff) | ((apic & 0xff) << 24);
                // EBX[23:16] = max addressable logical processors in the
                // package (all CPUs, rounded up to a power of two, clamped to
                // the field's 8-bit max of 255). LLC-scoped here would tell
                // the guest the package is one LLC. For >255 CPUs the guest
                // uses leaf 0xB EDX (32-bit) under x2APIC, not this legacy
                // field.
                let lpc = total_cpus.next_power_of_two().min(255);
                entry.ebx = (entry.ebx & 0xff00ffff) | (lpc << 16);
                // EBX[15:8] = CLFLUSH line size — preserved from KVM
                // ECX.31 = hypervisor — preserved from KVM
                // EDX bit 28 = HTT
                if total_cpus > 1 {
                    entry.edx |= 1 << 28;
                }
            }

            // Leaf 0x4: Deterministic Cache Parameters (Intel only)
            0x4 if vendor == CpuVendor::Intel => {
                patch_cache_topology_eax(entry, smt, core, topo.cores_per_llc);
            }

            // Leaves 0xB / 0x1F (Extended Topology) are SYNTHESIZED after
            // this loop, not patched here: KVM's get_supported_cpuid zeroes
            // these leaves (eax=ebx=ecx=0, no Core subleaf 1 — KVM leaves a
            // "valid topology ... subleaf 1" for the VMM to populate), so
            // patching in place cannot add the Core-level subleaf the guest
            // needs to form a single package. See below.

            // Leaf 0x8000001D (AMD Cache Topology) is SYNTHESIZED after this
            // loop, not patched here: patching only rewrites host-provided
            // subleaves, so a host whose 0x8000001D omits the L3 (type=3,
            // level=3) subleaf would leave the guest's llc_id unset and
            // collapse every CPU into one LLC. See the synthesis block below.

            // Leaf 0xA: Architectural Performance Monitoring (Intel SDM,
            // Architectural Performance Monitoring). Synthesized to a
            // conservative PMU v2 surface so guest sched_ext schedulers
            // (scx_layered, scx_cosmos) get usable perf counters
            // regardless of host hardware. AMD CPUs ignore leaf 0xA and
            // use MSR-based counters; populating it is a no-op on AMD.
            // See PMU_* consts at the top of this file for field
            // semantics.
            //
            // Gated on the ORIGINAL entry's version (EAX[7:0]) being non-zero.
            // On a kvm.enable_pmu=0 host, KVM zeros leaf 0xA before exposing
            // it via get_supported_cpuid; overwriting with v2 would tell the
            // guest "PMU available" while intel_pmu_refresh clamps every
            // counter count back to 0 — silent failures inside the guest.
            // Leaving zeros lets the guest's intel_pmu_init see version=0 and
            // graceful-fail the same way it does on a no-PMU bare-metal host.
            0xa => {
                if entry.eax & 0xff != 0 {
                    entry.eax = PMU_ARCH_PERFMON_VERSION
                        | (PMU_NUM_GP_COUNTERS << 8)
                        | (PMU_GP_COUNTER_WIDTH << 16)
                        | (PMU_EVENT_MASK_LENGTH << 24);
                    entry.ebx = 0;
                    entry.ecx = 0;
                    entry.edx = PMU_NUM_FIXED_COUNTERS | (PMU_FIXED_COUNTER_WIDTH << 5);
                }
            }

            // Leaf 0x80000001: AMD extended feature identification (AMD only)
            0x8000_0001 if vendor == CpuVendor::Amd && total_cpus > 1 => {
                // ECX bit 1 = CmpLegacy: multi-core chip
                // ECX bit 22 = TopologyExtensions: enables leaves 0x8000001D/1E
                entry.ecx |= (1 << 1) | (1 << 22);
            }

            // Leaf 0x80000006: AMD L2 (ECX) and L3 (EDX) cache descriptors.
            // EDX MUST be non-zero: the guest kernel gates AMD L3 detection
            // on cpuid_amd_hygon_has_l3_cache() == (cpuid_edx(0x80000006) != 0)
            // (arch/x86/include/asm/cpuid/api.h). KVM passes the host value
            // through; a host that masks the L3-size field to 0 makes the
            // guest see no L3 and collapse every CPU into its own LLC.
            // Synthesize L2/L3 to match the 0x8000001D geometry below so the
            // gate holds host-independently; EAX/EBX (TLB) are left as the
            // host reported them. 16-way associativity encodes to 0x8.
            0x8000_0006 if vendor == CpuVendor::Amd => {
                entry.ecx = L80000006_ECX; // L2: size KiB<<16 | assoc<<12 | lines/tag<<8 | line
                entry.edx = L80000006_EDX; // L3: size/512KiB<<18 | assoc<<12 | lines/tag<<8 | line
            }

            // Leaf 0x80000008: virtual/physical address sizes (vendor-independent)
            // ECX[7:0] = number of physical threads - 1
            // ECX[15:12] = APIC ID size (bits needed for thread IDs in package)
            0x8000_0008 => {
                if total_cpus > 1 {
                    // ECX[15:12] = APIC-ID bits covering all CPUs in the
                    // package (= the CORE domain shift the AMD topology
                    // parser uses for the package boundary, apic >> shift).
                    // ECX[7:0] = threads-per-package - 1, SATURATED to the
                    // 8-bit field: a >256-CPU package would otherwise wrap to
                    // a small NC and collapse the package. NC is only the
                    // AMD fallback when leaf 0xB is absent (we always emit
                    // 0xB), but saturating avoids advertising a wrong, small
                    // count. Both must be package-scoped, not LLC-scoped.
                    entry.ecx = (pkg_shift << 12) | ((total_cpus - 1).min(0xff));
                } else {
                    entry.ecx = 0;
                }
            }

            // Leaf 0x8000001E: AMD Extended APIC ID / Topology (AMD only)
            0x8000_001e if vendor == CpuVendor::Amd => {
                // EAX = Extended APIC ID
                entry.eax = apic;
                // EBX[7:0] = Compute Unit (core) ID
                // EBX[15:8] = Threads per compute unit - 1
                let (llc_id, core_id, _) = topo.decompose(cpu_id);
                entry.ebx = ((topo.threads_per_core - 1) << 8) | (core_id & 0xff);
                // ECX[7:0] = Node ID
                // ECX[10:8] = Nodes per processor - 1
                let node_id = topo.numa_node_of(llc_id);
                entry.ecx = node_id | ((topo.numa_nodes - 1) << 8);
                // EDX = reserved
                entry.edx = 0;
            }

            _ => {}
        }
    }

    // Synthesize the Extended-Topology leaves (0xB, and 0x1F for Intel).
    // KVM's get_supported_cpuid zeroes these leaves (eax=ebx=ecx=0, no Core
    // subleaf 1 — it leaves a "valid topology ... subleaf 1" for the VMM to
    // populate) — so a Core-level subleaf whose shift spans the WHOLE
    // package must be EMITTED (patching can't add a missing subleaf), else
    // the guest kernel leaves every CPU in its own package and
    // cpu_llc_shared_mask collapses to per-CPU (per-CPU L3). The LLC
    // sub-domain within the package is carved by the cache leaf
    // (0x4 / 0x8000001D). 0x1F is emitted only for Intel guests; AMD does
    // not enumerate it and the kernel falls through to 0xB.
    let topo_leaves: &[u32] = if vendor == CpuVendor::Intel {
        &[0xb, 0x1f]
    } else {
        &[0xb]
    };
    entries.retain(|e| e.function != 0xb && e.function != 0x1f);
    for &func in topo_leaves {
        // Subleaf 0: SMT level — shift to the Core id is the SMT width.
        entries.push(topo_subleaf(func, 0, smt, topo.threads_per_core, 1, apic));
        // Subleaf 1: Core level — shift to the Package id spans the whole
        // package, so apic >> pkg_shift == 0 for every CPU (one package).
        entries.push(topo_subleaf(func, 1, pkg_shift, total_cpus, 2, apic));
        // Subleaf 2: terminator (level type 0 ends enumeration).
        entries.push(topo_subleaf(func, 2, 0, 0, 0, apic));
    }

    // Synthesize the AMD cache-topology leaf 0x8000001D (host-independent).
    // The old code PATCHED the host's 0x8000001D subleaves in place, which
    // depends on the host exposing a complete L1/L2/L3 chain; a host whose
    // 0x8000001D lacks an L3 (type=3, level=3) subleaf would leave the
    // guest's llc_id unset (BAD_APICID) and collapse every CPU into one
    // LLC. Emit the full L1d/L1i/L2/L3 chain + a type-0 terminator so
    // find_num_cache_leaves (arch/x86/kernel/cpu/cacheinfo.c) always finds
    // L3 as the highest subleaf and amd_fill_cpuid4_info reads its
    // num_threads_sharing. L1/L2 share at the SMT level ((1<<smt)-1); L3
    // spans the LLC ((1<<core_shift)-1) — the span that makes get_cache_id
    // (apicid >> order(num_threads_sharing+1)) place each LLC on its own
    // cache id. Subleaf ORDER matters: L3 must be the highest non-null
    // index. AMD only — Intel uses leaf 0x4 (patched above) and 0x80000006
    // EDX is reserved on Intel.
    if vendor == CpuVendor::Amd {
        let smt_sharing = (1u32 << smt).saturating_sub(1);
        let llc_sharing = (1u32 << core).saturating_sub(1);
        entries.retain(|e| e.function != 0x8000_001d);
        // args: (index, type, level, self_init, size_kib, ways, sharing, flags).
        // EDX flags per qemu legacy_amd_cache_info: L1d/L1i=no-invd(0x1),
        // L2=none(0x0), L3=inclusive|complex(0x6). The kernel reads only
        // EAX/EBX/ECX from 0x8000001D, so EDX flags are informational.
        entries.push(amd_cache_subleaf(
            0, 1, 1, true, L1_CACHE_SIZE_KIB, L1_CACHE_WAYS, smt_sharing, 0x1,
        )); // L1 data
        entries.push(amd_cache_subleaf(
            1, 2, 1, true, L1_CACHE_SIZE_KIB, L1_CACHE_WAYS, smt_sharing, 0x1,
        )); // L1 instruction
        entries.push(amd_cache_subleaf(
            2, 3, 2, false, L2_CACHE_SIZE_KIB, L2_CACHE_WAYS, smt_sharing, 0x0,
        )); // L2 unified
        entries.push(amd_cache_subleaf(
            3, 3, 3, true, L3_CACHE_SIZE_KIB, L3_CACHE_WAYS, llc_sharing, 0x6,
        )); // L3 unified (the LLC)
        // Terminator: type 0 (EAX[4:0]=0) ends the kernel's subleaf walk.
        entries.push(kvm_cpuid_entry2 {
            function: 0x8000_001d,
            index: 4,
            flags: KVM_CPUID_FLAG_SIGNIFCANT_INDEX,
            ..Default::default()
        });
        // 0x80000006 is patched in-loop when present (preserving the host's
        // L2-TLB EAX/EBX); if the base omits it entirely, push a synthesized
        // one so the L3 gate (EDX != 0) still holds — symmetric with the
        // 0x8000001D synthesis. (Real KVM always enumerates 0x80000006; this
        // guards a base that omits an architectural leaf.)
        if !entries.iter().any(|e| e.function == 0x8000_0006) {
            entries.push(kvm_cpuid_entry2 {
                function: 0x8000_0006,
                ecx: L80000006_ECX,
                edx: L80000006_EDX,
                ..Default::default()
            });
        }
    }

    // Add hypervisor identification leaf (0x40000000) if not present.
    // Guest OS uses leaf 0x1 ECX.31 to detect hypervisor, then reads
    // 0x40000000 for the hypervisor signature. KVM's supported CPUID
    // may already include this; only add if missing.
    if !entries.iter().any(|e| e.function == 0x4000_0000) {
        entries.push(kvm_cpuid_entry2 {
            function: 0x4000_0000,
            index: 0,
            flags: 0,
            eax: 0x4000_0001, // max hypervisor leaf
            // "KVMKVMKVM\0\0\0" signature
            ebx: 0x4b56_4d4b, // "KVMK"
            ecx: 0x564b_4d56, // "VMKV"
            edx: 0x0000_004d, // "M\0\0\0"
            ..Default::default()
        });
    }

    // Topologies above the xAPIC limit (max APIC ID > 254 — the same
    // threshold that switches the VMM to split-irqchip) need x2APIC with
    // extended MSI destination IDs to address CPUs above 255.
    // KVM_FEATURE_MSI_EXT_DEST_ID (0x40000001 EAX bit 15) makes the guest's
    // try_to_enable_x2apic raise apic_limit from 255 to 32767
    // (arch/x86/kernel/apic/apic.c) and pack the high destination bits into
    // the IOAPIC RTE / MSI address; host KVM decodes them via
    // x86_msi_msg_get_destid. Without it the guest refuses to online any CPU
    // whose APIC ID exceeds 255. Gated on the topology so smaller guests'
    // CPUID is byte-identical to before.
    let wide_smp = max_apic_id(topo) > crate::vmm::x86_64::kvm::MAX_XAPIC_ID;
    if wide_smp {
        if let Some(entry) = entries.iter_mut().find(|e| e.function == 0x4000_0001) {
            entry.eax |= 1 << 15; // KVM_FEATURE_MSI_EXT_DEST_ID
        } else {
            // Defensive: KVM always enumerates 0x40000001 (its PV-features
            // leaf), but a base CPUID that omitted it would otherwise drop
            // the bit silently.
            entries.push(kvm_cpuid_entry2 {
                function: 0x4000_0001,
                eax: 1 << 15,
                ..Default::default()
            });
        }
    }

    // KVM_HINTS_REALTIME: CPUID leaf 0x40000001 EDX bit 0.
    // Disables PV spinlocks, PV TLB flush, and PV sched_yield in the
    // guest, and enables haltpoll cpuidle. PV spinlocks require
    // CONFIG_PARAVIRT_SPINLOCKS (not in ktstr.kconfig, so no-op for ktstr
    // guests). Only set in performance_mode to avoid disabling PV
    // optimizations in functional tests.
    if performance_mode {
        if let Some(entry) = entries.iter_mut().find(|e| e.function == 0x4000_0001) {
            entry.edx |= 1;
        }
    }

    // Both paths above populate leaf 0x40000001 (wide_smp -> EAX
    // MSI_EXT_DEST_ID; performance_mode -> EDX HINTS_REALTIME); the guest
    // only enumerates it if 0x40000000 advertises it as the max hypervisor
    // leaf. Bump once for whichever ran.
    if wide_smp || performance_mode {
        if let Some(entry) = entries.iter_mut().find(|e| e.function == 0x4000_0000) {
            entry.eax = entry.eax.max(0x4000_0001);
        }
    }

    entries
}

#[cfg(test)]
mod tests {
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
            assert_eq!(nc, topo.total_cpus() - 1, "NC should be cpus_per_package - 1");
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
                .unwrap_or_else(|| panic!("{vendor}: 0xB Core subleaf (index 1) must be synthesized"));
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
                assert_eq!(n, 1, "{vendor}: exactly one 0xB subleaf at index {idx}, got {n}");
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
}
