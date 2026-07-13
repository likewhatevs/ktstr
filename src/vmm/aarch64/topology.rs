use anyhow::{Context, Result};
use kvm_bindings::{
    KVM_REG_ARM64, KVM_REG_ARM64_SYSREG, KVM_REG_ARM64_SYSREG_CRM_MASK,
    KVM_REG_ARM64_SYSREG_CRM_SHIFT, KVM_REG_ARM64_SYSREG_CRN_MASK, KVM_REG_ARM64_SYSREG_CRN_SHIFT,
    KVM_REG_ARM64_SYSREG_OP0_MASK, KVM_REG_ARM64_SYSREG_OP0_SHIFT, KVM_REG_ARM64_SYSREG_OP1_MASK,
    KVM_REG_ARM64_SYSREG_OP1_SHIFT, KVM_REG_ARM64_SYSREG_OP2_MASK, KVM_REG_ARM64_SYSREG_OP2_SHIFT,
    KVM_REG_SIZE_U64,
};
use kvm_ioctls::VcpuFd;

#[cfg(test)]
use crate::vmm::topology::Topology;

/// Maximum cache level supported by arm64 CLIDR_EL1 (7 Ctype fields).
pub const MAX_CACHE_LEVEL: u32 = 7;

/// MPIDR_EL1 register ID for KVM_GET_ONE_REG / KVM_SET_ONE_REG.
/// Encoded as system register (3, 0, 0, 0, 5) per the kernel's
/// arch/arm64/include/uapi/asm/kvm.h.
pub const MPIDR_EL1: u64 = KVM_REG_ARM64
    | KVM_REG_SIZE_U64
    | KVM_REG_ARM64_SYSREG as u64
    | ((3u64 << KVM_REG_ARM64_SYSREG_OP0_SHIFT) & KVM_REG_ARM64_SYSREG_OP0_MASK as u64)
    | ((0u64 << KVM_REG_ARM64_SYSREG_OP1_SHIFT) & KVM_REG_ARM64_SYSREG_OP1_MASK as u64)
    | ((0u64 << KVM_REG_ARM64_SYSREG_CRN_SHIFT) & KVM_REG_ARM64_SYSREG_CRN_MASK as u64)
    | ((0u64 << KVM_REG_ARM64_SYSREG_CRM_SHIFT) & KVM_REG_ARM64_SYSREG_CRM_MASK as u64)
    | ((5u64 << KVM_REG_ARM64_SYSREG_OP2_SHIFT) & KVM_REG_ARM64_SYSREG_OP2_MASK as u64);

/// CLIDR_EL1 register ID for KVM_GET_ONE_REG / KVM_SET_ONE_REG.
/// Encoded as system register (3, 1, 0, 0, 1).
const CLIDR_EL1: u64 = KVM_REG_ARM64
    | KVM_REG_SIZE_U64
    | KVM_REG_ARM64_SYSREG as u64
    | ((3u64 << KVM_REG_ARM64_SYSREG_OP0_SHIFT) & KVM_REG_ARM64_SYSREG_OP0_MASK as u64)
    | ((1u64 << KVM_REG_ARM64_SYSREG_OP1_SHIFT) & KVM_REG_ARM64_SYSREG_OP1_MASK as u64)
    | ((0u64 << KVM_REG_ARM64_SYSREG_CRN_SHIFT) & KVM_REG_ARM64_SYSREG_CRN_MASK as u64)
    | ((0u64 << KVM_REG_ARM64_SYSREG_CRM_SHIFT) & KVM_REG_ARM64_SYSREG_CRM_MASK as u64)
    | ((1u64 << KVM_REG_ARM64_SYSREG_OP2_SHIFT) & KVM_REG_ARM64_SYSREG_OP2_MASK as u64);

// CLIDR_EL1 Ctype field values.
const CLIDR_CTYPE_NO_CACHE: u64 = 0;
const CLIDR_CTYPE_INSTRUCTION: u64 = 1;
const CLIDR_CTYPE_DATA: u64 = 2;
const CLIDR_CTYPE_SEPARATE: u64 = 3;
const CLIDR_CTYPE_UNIFIED: u64 = 4;

// CLIDR_EL1 field positions.
const CLIDR_CTYPE_BITS: u32 = 3;
const CLIDR_LOC_SHIFT: u32 = 24;

/// Return true if the host's L1 cache is unified (from sysfs).
///
/// When the host L1 is unified, the CLIDR Ctype1 field is Unified (1
/// leaf). The DT's `of_count_cache_leaves` defaults to 2 for CPU nodes
/// without cache properties, so the CPU node needs `cache-unified` to
/// reduce the OF count to 1.
pub fn host_l1_is_unified() -> bool {
    let cache_dir = "/sys/devices/system/cpu/cpu0/cache";
    let entries = match std::fs::read_dir(cache_dir) {
        Ok(e) => e,
        Err(_) => return false,
    };
    let mut has_data = false;
    let mut has_instruction = false;
    let mut has_unified = false;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with("index") {
            continue;
        }
        let level_path = entry.path().join("level");
        let type_path = entry.path().join("type");
        let Ok(level_str) = std::fs::read_to_string(&level_path) else {
            continue;
        };
        let Ok(level) = level_str.trim().parse::<u32>() else {
            continue;
        };
        if level != 1 {
            continue;
        }
        let Ok(type_str) = std::fs::read_to_string(&type_path) else {
            continue;
        };
        match type_str.trim() {
            "Data" => has_data = true,
            "Instruction" => has_instruction = true,
            "Unified" => has_unified = true,
            _ => {}
        }
    }
    has_unified && !has_data && !has_instruction
}

/// Build a CLIDR_EL1 value from the host's sysfs cache topology.
///
/// Reads /sys/devices/system/cpu/cpu0/cache/index*/level and type to
/// determine Ctype fields for each cache level. Sets LoC to the
/// highest level found.
fn build_clidr_from_sysfs() -> u64 {
    let cache_dir = "/sys/devices/system/cpu/cpu0/cache";
    let entries = match std::fs::read_dir(cache_dir) {
        Ok(e) => e,
        Err(_) => return 0,
    };

    // Collect cache types per level.
    let mut level_types: [u8; MAX_CACHE_LEVEL as usize + 1] = [0; MAX_CACHE_LEVEL as usize + 1];
    // Bit flags: 1=Data, 2=Instruction, 4=Unified
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with("index") {
            continue;
        }
        let level_path = entry.path().join("level");
        let type_path = entry.path().join("type");
        let Ok(level_str) = std::fs::read_to_string(&level_path) else {
            continue;
        };
        let Ok(level) = level_str.trim().parse::<u32>() else {
            continue;
        };
        if level == 0 || level > MAX_CACHE_LEVEL {
            continue;
        }
        let Ok(type_str) = std::fs::read_to_string(&type_path) else {
            continue;
        };
        match type_str.trim() {
            "Data" => level_types[level as usize] |= 1,
            "Instruction" => level_types[level as usize] |= 2,
            "Unified" => level_types[level as usize] |= 4,
            _ => {}
        }
    }

    let mut clidr: u64 = 0;
    let mut max_level: u32 = 0;
    for level in 1..=MAX_CACHE_LEVEL {
        let flags = level_types[level as usize];
        if flags == 0 {
            break;
        }
        let ctype = if flags & 4 != 0 {
            CLIDR_CTYPE_UNIFIED
        } else if flags & 1 != 0 && flags & 2 != 0 {
            CLIDR_CTYPE_SEPARATE
        } else if flags & 2 != 0 {
            CLIDR_CTYPE_INSTRUCTION
        } else if flags & 1 != 0 {
            CLIDR_CTYPE_DATA
        } else {
            CLIDR_CTYPE_NO_CACHE
        };
        let shift = CLIDR_CTYPE_BITS * (level - 1);
        clidr |= ctype << shift;
        max_level = level;
    }

    // Set LoC (Level of Coherence) to the highest cache level.
    clidr |= (max_level as u64) << CLIDR_LOC_SHIFT;

    clidr
}

/// Merge sysfs-derived Ctype and LoC fields into an existing CLIDR_EL1
/// value, preserving LoUU, LoUIS, ICB, and Ttype fields from the
/// original.
fn merge_clidr(current: u64, sysfs: u64) -> u64 {
    // Ctype fields: bits [20:0] (7 levels x 3 bits = 21 bits).
    // LoC field: bits [26:24].
    const CTYPE_MASK: u64 = 0x001F_FFFF;
    const LOC_MASK: u64 = 0x0700_0000;
    const REPLACE_MASK: u64 = CTYPE_MASK | LOC_MASK;
    (current & !REPLACE_MASK) | (sysfs & REPLACE_MASK)
}

/// Clear Ctype fields above `max_level` and cap LoC to `max_level`.
///
/// Used when the guest presents a fixed cache depth (multi-LLC guests
/// terminate their DT chain at L3) but the host has more cache levels.
/// Without this cap, CLIDR would report leaves the DT chain lacks:
/// `init_cache_level` sizes `num_leaves` from CLIDR, then
/// `cache_setup_of_node` walks the DT chain expecting one node per non-L1
/// leaf and bails with -ENOENT when it runs out, dropping DT grouping
/// entirely (the guest falls back to per-CPU arch cache info — no shared
/// LLC domain). Capping CLIDR to the DT depth keeps the two consistent.
/// Levels the DT declares *beyond* CLIDR are handled the other way, by the
/// kernel's `of_find_last_cache_level` external-cache path, so a low cap
/// here never hides an L3 the DT still adds.
fn cap_clidr_levels(clidr: u64, max_level: u32) -> u64 {
    let mut out = clidr;
    for level in (max_level + 1)..=MAX_CACHE_LEVEL {
        let shift = CLIDR_CTYPE_BITS * (level - 1);
        out &= !(0x7u64 << shift);
    }
    let loc = (out >> CLIDR_LOC_SHIFT) & 0x7;
    if loc > max_level as u64 {
        out = (out & !(0x7u64 << CLIDR_LOC_SHIFT)) | ((max_level as u64) << CLIDR_LOC_SHIFT);
    }
    out
}

/// Override CLIDR_EL1 on each vCPU to match the host's real cache
/// topology from sysfs, capped to `max_level` cache levels.
///
/// KVM's `reset_clidr` (since host kernel 6.3) fabricates CLIDR_EL1
/// from CTR_EL0 flags, which can report fewer cache levels than the
/// host actually has. The DT is built from sysfs and may describe
/// more levels. When CLIDR and DT disagree on cache leaf counts,
/// `cache_setup_of_node` fails and the guest sees no cache topology.
///
/// This reads the current (possibly fabricated) CLIDR from vCPU 0,
/// replaces only the Ctype and LoC fields with values from sysfs,
/// preserves LoUU/LoUIS/ICB/Ttype, and writes back to all vCPUs.
/// On pre-6.3 kernels where CLIDR already passes through the real
/// value, the write is effectively a no-op.
///
/// `max_level` caps the reported levels so CLIDR agrees with a DT cache
/// chain that terminates early (multi-LLC guests present an L3 boundary;
/// pass 3). Single-LLC guests emit no DT cache chain — pass
/// `MAX_CACHE_LEVEL` to leave the host's levels untouched.
pub fn override_clidr(vcpus: &[VcpuFd], max_level: u32) -> Result<()> {
    let mut sysfs_clidr = build_clidr_from_sysfs();
    if sysfs_clidr == 0 {
        tracing::warn!("no cache info from sysfs, skipping CLIDR override");
        return Ok(());
    }
    if max_level < MAX_CACHE_LEVEL {
        sysfs_clidr = cap_clidr_levels(sysfs_clidr, max_level);
    }

    let mut cur_clidr_bytes = [0u8; 8];
    if let Err(e) = vcpus[0].get_one_reg(CLIDR_EL1, &mut cur_clidr_bytes) {
        tracing::warn!("failed to read CLIDR_EL1, skipping override: {e}");
        return Ok(());
    }
    let cur_clidr = u64::from_le_bytes(cur_clidr_bytes);
    let new_clidr = merge_clidr(cur_clidr, sysfs_clidr);

    if new_clidr != cur_clidr {
        let new_bytes = new_clidr.to_le_bytes();
        for (i, vcpu) in vcpus.iter().enumerate() {
            vcpu.set_one_reg(CLIDR_EL1, &new_bytes)
                .with_context(|| format!("set CLIDR_EL1 on vCPU {i}"))?;
        }
        tracing::debug!(
            cur = format_args!("{cur_clidr:#x}"),
            new = format_args!("{new_clidr:#x}"),
            "CLIDR_EL1 override applied",
        );
    }

    Ok(())
}

/// Mask for the affinity fields used in FDT cpu node `reg` property.
/// Bits \[23:0\] of MPIDR: Aff0 \[7:0\], Aff1 \[15:8\], Aff2 \[23:16\].
pub const MPIDR_AFF_MASK: u64 = 0xFF_FFFF;

/// Read MPIDR_EL1 from a vCPU after vcpu_init.
pub fn read_mpidr(vcpu: &VcpuFd) -> Result<u64> {
    let mut buf = [0u8; 8];
    vcpu.get_one_reg(MPIDR_EL1, &mut buf)
        .context("read MPIDR_EL1")?;
    Ok(u64::from_le_bytes(buf))
}

/// Read MPIDRs for all vCPUs.
pub fn read_mpidrs(vcpus: &[VcpuFd]) -> Result<Vec<u64>> {
    vcpus.iter().map(read_mpidr).collect()
}

/// Extract the FDT `reg` value from an MPIDR: affinity fields only.
pub fn mpidr_to_fdt_reg(mpidr: u64) -> u64 {
    mpidr & MPIDR_AFF_MASK
}

/// Compute MPIDR affinity encoding from topology decomposition.
/// Aff0 = thread, Aff1 = core, Aff2 = LLC.
/// This is our own MPIDR layout. KVM's default reset_mpidr
/// (arch/arm64/kvm/sys_regs.c) maps vcpu_id linearly into affinity
/// levels (Aff0 bits [3:0], Aff1 bits [11:4], Aff2 bits [19:12]), which
/// does not match this layout.
#[cfg(test)]
fn mpidr_from_topology(topo: &Topology, cpu_id: u32) -> u64 {
    let (llc, core, thread) = topo.decompose(cpu_id);
    let aff0 = thread as u64;
    let aff1 = core as u64;
    let aff2 = llc as u64;
    (1u64 << 31) | (aff2 << 16) | (aff1 << 8) | aff0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mpidr_el1_reg_id() {
        // MPIDR_EL1 = sys_reg(3, 0, 0, 0, 5):
        //   KVM_REG_ARM64 | KVM_REG_SIZE_U64 | KVM_REG_ARM64_SYSREG
        //   | (3 << 14) | (0 << 11) | (0 << 7) | (0 << 3) | 5
        assert_eq!(
            MPIDR_EL1, 0x6030_0000_0013_C005,
            "MPIDR_EL1 register ID encoding"
        );
    }

    #[test]
    fn mpidr_from_topology_single() {
        let t = Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
            llc_cores: None,
        };
        let mpidr = mpidr_from_topology(&t, 0);
        assert_eq!(mpidr & MPIDR_AFF_MASK, 0);
        assert_ne!(mpidr & (1 << 31), 0, "bit 31 must be set");
    }

    #[test]
    fn mpidr_from_topology_multi() {
        let t = Topology {
            llcs: 2,
            cores_per_llc: 4,
            threads_per_core: 2,
            numa_nodes: 1,
            nodes: None,
            distances: None,
            llc_cores: None,
        };
        // cpu 0: LLC 0, core 0, thread 0
        let m0 = mpidr_from_topology(&t, 0);
        assert_eq!(m0 & 0xFF, 0, "aff0 (thread) = 0");
        assert_eq!((m0 >> 8) & 0xFF, 0, "aff1 (core) = 0");
        assert_eq!((m0 >> 16) & 0xFF, 0, "aff2 (LLC) = 0");

        // cpu 1: LLC 0, core 0, thread 1
        let m1 = mpidr_from_topology(&t, 1);
        assert_eq!(m1 & 0xFF, 1, "aff0 (thread) = 1");
        assert_eq!((m1 >> 8) & 0xFF, 0, "aff1 (core) = 0");

        // cpu 2: LLC 0, core 1, thread 0
        let m2 = mpidr_from_topology(&t, 2);
        assert_eq!(m2 & 0xFF, 0, "aff0 (thread) = 0");
        assert_eq!((m2 >> 8) & 0xFF, 1, "aff1 (core) = 1");

        // cpu 8: LLC 1, core 0, thread 0
        let m8 = mpidr_from_topology(&t, 8);
        assert_eq!(m8 & 0xFF, 0, "aff0 (thread) = 0");
        assert_eq!((m8 >> 8) & 0xFF, 0, "aff1 (core) = 0");
        assert_eq!((m8 >> 16) & 0xFF, 1, "aff2 (LLC) = 1");
    }

    #[test]
    fn mpidr_to_fdt_reg_masks() {
        let mpidr = (1u64 << 31) | (2 << 16) | (3 << 8) | 1;
        let reg = mpidr_to_fdt_reg(mpidr);
        assert_eq!(reg, (2 << 16) | (3 << 8) | 1);
        assert_eq!(reg & (1 << 31), 0, "bit 31 should be masked out");
    }

    #[test]
    fn mpidr_unique_representative_topologies() {
        let topos = [
            (1, 1, 1),   // degenerate single CPU
            (2, 1, 1),   // minimal multi-LLC
            (3, 3, 1),   // odd non-power-of-2
            (1, 1, 2),   // minimal SMT
            (2, 4, 2),   // standard multi-LLC with SMT
            (7, 5, 3),   // all dimensions non-power-of-2
            (15, 16, 1), // large scale no SMT
            (14, 9, 2),  // large with SMT
            (255, 1, 1), // max LLCs before Aff2 overflow
            (1, 255, 1), // max cores before Aff1 overflow
            (4, 32, 1),  // many cores, multi-LLC
        ];
        for (llcs, cores, threads) in topos {
            let t = Topology {
                llcs,
                cores_per_llc: cores,
                threads_per_core: threads,
                numa_nodes: 1,
                nodes: None,
                distances: None,
                llc_cores: None,
            };
            let mpidrs: Vec<u64> = (0..t.total_cpus())
                .map(|i| mpidr_from_topology(&t, i))
                .collect();
            let unique: std::collections::HashSet<u64> = mpidrs.iter().copied().collect();
            assert_eq!(
                mpidrs.len(),
                unique.len(),
                "topology {llcs}l/{cores}c/{threads}t: MPIDRs not unique"
            );
        }
    }

    #[test]
    fn mpidr_bit31_always_set() {
        let t = Topology {
            llcs: 2,
            cores_per_llc: 4,
            threads_per_core: 2,
            numa_nodes: 1,
            nodes: None,
            distances: None,
            llc_cores: None,
        };
        for cpu in 0..t.total_cpus() {
            let mpidr = mpidr_from_topology(&t, cpu);
            assert_ne!(mpidr & (1 << 31), 0, "cpu {cpu}: MPIDR bit 31 must be set");
        }
    }

    #[test]
    fn mpidr_aff_mask_covers_three_levels() {
        assert_eq!(MPIDR_AFF_MASK, 0xFF_FFFF);
        assert_eq!(MPIDR_AFF_MASK & 0xFF, 0xFF, "Aff0 fully covered");
        assert_eq!((MPIDR_AFF_MASK >> 8) & 0xFF, 0xFF, "Aff1 fully covered");
        assert_eq!((MPIDR_AFF_MASK >> 16) & 0xFF, 0xFF, "Aff2 fully covered");
    }

    #[test]
    fn decompose_matches_mpidr_fields() {
        let t = Topology {
            llcs: 3,
            cores_per_llc: 5,
            threads_per_core: 2,
            numa_nodes: 1,
            nodes: None,
            distances: None,
            llc_cores: None,
        };
        for cpu in 0..t.total_cpus() {
            let (llc, core, thread) = t.decompose(cpu);
            let mpidr = mpidr_from_topology(&t, cpu);
            assert_eq!(mpidr & 0xFF, thread as u64, "cpu {cpu}: aff0 = thread");
            assert_eq!((mpidr >> 8) & 0xFF, core as u64, "cpu {cpu}: aff1 = core");
            assert_eq!((mpidr >> 16) & 0xFF, llc as u64, "cpu {cpu}: aff2 = LLC");
        }
    }

    #[test]
    fn clidr_el1_reg_id() {
        // CLIDR_EL1 = sys_reg(3, 1, 0, 0, 1):
        //   KVM_REG_ARM64 | KVM_REG_SIZE_U64 | KVM_REG_ARM64_SYSREG
        //   | (3 << 14) | (1 << 11) | (0 << 7) | (0 << 3) | 1
        assert_eq!(
            CLIDR_EL1, 0x6030_0000_0013_C801,
            "CLIDR_EL1 register ID encoding"
        );
    }

    #[test]
    fn build_clidr_from_sysfs_nonzero() {
        let clidr = build_clidr_from_sysfs();
        assert_ne!(clidr, 0, "sysfs should produce a non-zero CLIDR");
        // L1 must be present.
        let ctype1 = clidr & 0x7;
        assert_ne!(ctype1, 0, "L1 Ctype must be non-zero");
        // LoC must be at least 1.
        let loc = (clidr >> CLIDR_LOC_SHIFT) & 0x7;
        assert!(loc >= 1, "LoC must be >= 1, got {loc}");
    }

    #[test]
    fn merge_clidr_replaces_ctype_and_loc() {
        // current: LoUU=2 [29:27], LoUIS=1 [23:21], Ctype1=Unified(4), LoC=1
        let current: u64 = (2 << 27) | (1 << 21) | (1 << 24) | 4;
        // sysfs: Ctype1=Separate(3), Ctype2=Unified(4), LoC=2
        let sysfs: u64 = (2 << 24) | (4 << 3) | 3;
        let merged = merge_clidr(current, sysfs);

        // Ctype and LoC from sysfs.
        assert_eq!(merged & 0x001F_FFFF, sysfs & 0x001F_FFFF);
        assert_eq!((merged >> 24) & 0x7, 2, "LoC from sysfs");
        // LoUIS and LoUU preserved from current.
        assert_eq!((merged >> 21) & 0x7, 1, "LoUIS preserved");
        assert_eq!((merged >> 27) & 0x7, 2, "LoUU preserved");
    }

    #[test]
    fn merge_clidr_identity_when_equal() {
        let val = 0x0000_0000_0200_0023_u64;
        assert_eq!(merge_clidr(val, val), val);
    }

    #[test]
    fn cap_clidr_levels_noop_when_within_cap() {
        // 2-level host (L1 separate, L2 unified, LoC=2). Capping to 3
        // leaves it untouched — the DT chain adds the L3 beyond LoC.
        let clidr: u64 = (2 << CLIDR_LOC_SHIFT) | (4 << 3) | 3;
        assert_eq!(cap_clidr_levels(clidr, 3), clidr);
    }

    #[test]
    fn cap_clidr_levels_clears_levels_above_cap() {
        // 4-level host: Ctype1..4 = Unified, LoC=4. Cap to 3 must clear
        // Ctype4 and pull LoC down to 3 so CLIDR agrees with a DT chain
        // that terminates at L3.
        let clidr: u64 = (4 << CLIDR_LOC_SHIFT)
            | (CLIDR_CTYPE_UNIFIED << (CLIDR_CTYPE_BITS * 3))  // Ctype4
            | (CLIDR_CTYPE_UNIFIED << (CLIDR_CTYPE_BITS * 2))  // Ctype3
            | (CLIDR_CTYPE_UNIFIED << CLIDR_CTYPE_BITS)        // Ctype2
            | CLIDR_CTYPE_UNIFIED; // Ctype1
        let capped = cap_clidr_levels(clidr, 3);
        assert_eq!(
            (capped >> (CLIDR_CTYPE_BITS * 3)) & 0x7,
            0,
            "Ctype4 must be cleared"
        );
        assert_eq!(
            (capped >> (CLIDR_CTYPE_BITS * 2)) & 0x7,
            CLIDR_CTYPE_UNIFIED,
            "Ctype3 preserved"
        );
        assert_eq!((capped >> CLIDR_LOC_SHIFT) & 0x7, 3, "LoC capped to 3");
    }
}
