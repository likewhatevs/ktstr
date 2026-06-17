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
            0,
            1,
            1,
            true,
            L1_CACHE_SIZE_KIB,
            L1_CACHE_WAYS,
            smt_sharing,
            0x1,
        )); // L1 data
        entries.push(amd_cache_subleaf(
            1,
            2,
            1,
            true,
            L1_CACHE_SIZE_KIB,
            L1_CACHE_WAYS,
            smt_sharing,
            0x1,
        )); // L1 instruction
        entries.push(amd_cache_subleaf(
            2,
            3,
            2,
            false,
            L2_CACHE_SIZE_KIB,
            L2_CACHE_WAYS,
            smt_sharing,
            0x0,
        )); // L2 unified
        entries.push(amd_cache_subleaf(
            3,
            3,
            3,
            true,
            L3_CACHE_SIZE_KIB,
            L3_CACHE_WAYS,
            llc_sharing,
            0x6,
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
    if performance_mode && let Some(entry) = entries.iter_mut().find(|e| e.function == 0x4000_0001)
    {
        entry.edx |= 1;
    }

    // Both paths above populate leaf 0x40000001 (wide_smp -> EAX
    // MSI_EXT_DEST_ID; performance_mode -> EDX HINTS_REALTIME); the guest
    // only enumerates it if 0x40000000 advertises it as the max hypervisor
    // leaf. Bump once for whichever ran.
    if (wide_smp || performance_mode)
        && let Some(entry) = entries.iter_mut().find(|e| e.function == 0x4000_0000)
    {
        entry.eax = entry.eax.max(0x4000_0001);
    }

    entries
}

#[cfg(test)]
mod tests_cpuid;
#[cfg(test)]
mod tests_math;
