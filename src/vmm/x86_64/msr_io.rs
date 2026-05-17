//! Single-MSR read primitive over `KVM_GET_MSRS`.
//!
//! Batched MSR writes for the boot-time seed live in
//! [`super::boot::setup_msrs`]; this module is the single-MSR read
//! counterpart used by runtime readers (e.g. virt-KASLR derivation
//! in [`super::msr_kaslr`]) and per-MSR test probes. v0 exposes
//! [`read_one_msr`] only.

use anyhow::{Context, Result, ensure};
use kvm_bindings::{Msrs, kvm_msr_entry};
use kvm_ioctls::VcpuFd;

/// Read a single MSR from `vcpu`.
///
/// Returns `Ok(None)` when `KVM_GET_MSRS` returns 0 entries — the
/// host KVM build does not expose `msr_index`. Returns `Err` on
/// ioctl failure or KVM ABI violation (count > 1 for a 1-entry
/// batch).
///
/// Kernel contract: `__msr_io` at `arch/x86/kvm/x86.c` returns the
/// count of successfully-read entries and stops on first per-MSR
/// failure. For a 1-MSR batch the count collapses to either 0 or 1;
/// any other count would violate the KVM ABI and trips `ensure!`.
///
/// For multi-MSR batches (not exposed here), the same kernel contract
/// uses the returned count as an index into the entry array to
/// identify the first failing MSR: `entries[count].index` names the
/// MSR the kernel stopped on. A future `read_many_msrs` helper would
/// need to surface that index rather than collapse to `Option` like
/// this single-MSR form.
pub(crate) fn read_one_msr(vcpu: &VcpuFd, msr_index: u32) -> Result<Option<u64>> {
    // The slice literal `&[entry]` is statically non-empty, so
    // `Msrs::from_entries` cannot fail on the empty-slice path here;
    // the `?` surfaces the Result for forward-compat with any future
    // capacity check the kvm-bindings crate might add.
    let mut msrs = Msrs::from_entries(&[kvm_msr_entry {
        index: msr_index,
        ..Default::default()
    }])
    .with_context(|| format!("Msrs::from_entries for MSR {msr_index:#x}"))?;
    let n = vcpu
        .get_msrs(&mut msrs)
        .with_context(|| format!("KVM_GET_MSRS for MSR {msr_index:#x}"))?;
    if n == 0 {
        return Ok(None);
    }
    ensure!(
        n == 1,
        "KVM_GET_MSRS returned {n} entries for 1-MSR batch \
         (MSR {msr_index:#x}) — KVM ABI violation"
    );
    Ok(Some(msrs.as_slice()[0].data))
}

/// Test convenience: read a single architectural MSR that KVM must
/// expose. Panics with `label` if the MSR is missing or the read
/// errors. Use this from tests where `None` indicates a test-setup
/// bug, not a runtime degradation case.
#[cfg(test)]
pub(crate) fn read_one_msr_required(vcpu: &VcpuFd, msr_index: u32, label: &str) -> u64 {
    read_one_msr(vcpu, msr_index)
        .unwrap_or_else(|e| panic!("read_one_msr({msr_index:#x}) [{label}]: {e}"))
        .unwrap_or_else(|| panic!("KVM did not expose MSR {msr_index:#x} ({label})"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vmm::x86_64::msr_indices::MSR_IA32_MISC_ENABLE;
    use crate::vmm::x86_64::test_helpers::single_vcpu_kvm;

    #[test]
    fn read_one_msr_returns_some_on_supported_msr() {
        // MSR_IA32_MISC_ENABLE is architectural on every x86_64 host
        // KVM build; KVM_GET_MSRS must return 1 entry with the seeded
        // value (KVM hands out a default at vCPU init).
        let vm = single_vcpu_kvm();
        let value = read_one_msr(&vm.vcpus[0], MSR_IA32_MISC_ENABLE).unwrap();
        assert!(
            value.is_some(),
            "MSR_IA32_MISC_ENABLE must be exposed by KVM"
        );
    }

    #[test]
    fn read_one_msr_returns_none_on_unsupported_msr() {
        // MSR index 0x2 is unassigned in arch/x86/include/asm/msr-index.h
        // and KVM does not expose it; KVM_GET_MSRS returns 0 entries.
        // Per the helper's contract that collapses to `Ok(None)`.
        // Firecracker uses the same probe index in its
        // test_get_msrs_with_invalid_msr_index test.
        let vm = single_vcpu_kvm();
        let value = read_one_msr(&vm.vcpus[0], 0x2).unwrap();
        assert!(value.is_none(), "MSR 0x2 must not be exposed by KVM");
    }

    #[test]
    fn read_one_msr_matches_raw_get_msrs_value() {
        // Pin that the helper returns bit-identical data to a raw
        // KVM_GET_MSRS through the same kvm_msr_entry path. Guards
        // against an accidental transform inside read_one_msr after
        // future refactors.
        let vm = single_vcpu_kvm();
        let via_helper = read_one_msr(&vm.vcpus[0], MSR_IA32_MISC_ENABLE).unwrap();
        let mut raw_msrs = Msrs::from_entries(&[kvm_msr_entry {
            index: MSR_IA32_MISC_ENABLE,
            ..Default::default()
        }])
        .unwrap();
        let n = vm.vcpus[0].get_msrs(&mut raw_msrs).unwrap();
        assert_eq!(n, 1);
        let via_raw = raw_msrs.as_slice()[0].data;
        assert_eq!(via_helper, Some(via_raw));
    }
}
