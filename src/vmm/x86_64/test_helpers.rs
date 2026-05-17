//! Shared `#[cfg(test)]` fixtures for sibling `vmm::x86_64` test
//! modules. Lives here so the same KVM topology shapes don't get
//! re-inlined in every test file under the directory.

use crate::vmm::kvm::KtstrKvm;
use crate::vmm::topology::Topology;

/// 1-LLC / 1-core / 1-thread / 1-NUMA KVM instance with 64 MiB
/// guest memory — the smallest VM that boots through `KtstrKvm::new`
/// and exposes a single vCPU at `vm.vcpus[0]`.
///
/// Used by every test that doesn't care about topology shape and
/// just needs a vCPU to drive an ioctl through (boot MSR seed,
/// single-MSR read, FPU/regs setup, ...). Tests that exercise
/// multi-core paths (SMT pairing, per-LLC partitioning) keep their
/// `Topology` literal inline so the topology shape stays visible
/// at the call site.
pub(super) fn single_vcpu_kvm() -> KtstrKvm {
    let topo = Topology {
        llcs: 1,
        cores_per_llc: 1,
        threads_per_core: 1,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    };
    KtstrKvm::new(topo, 64, false).unwrap()
}
