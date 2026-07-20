//! Tests for `handle_freeze` that do require a real KVM VcpuFd —
//! they boot a minimal `KtstrKvm` (1 vCPU, 64 MB) so the freeze
//! drain dance and reg-capture paths exercise real KVM ioctls.
//!
//! These tests skip the parking loop by leaving `freeze=false`,
//! so the function falls straight through after the drain dance,
//! reg capture, and parked Release-store. That covers:
//!   - has_immediate_exit=true → vcpu.run() returns EINTR (the
//!     KVM_CAP_IMMEDIATE_EXIT contract) and the drain swallows it
//!   - capture_vcpu_regs writes a Some(_) to regs_slot
//!   - parked AtomicBool gets a Release-then-Release toggle
//!     (true on entry, false on exit)
//!   - parked_evt.write(1) writes one edge

use super::*;
use std::sync::atomic::{AtomicBool, Ordering};

#[test]
fn handle_freeze_drain_swallows_eintr_and_resets_state() {
    // Boot a minimal VM, set up sregs so capture_vcpu_regs has a
    // valid CR3, then call handle_freeze. With freeze=false, the
    // park loop falls through; the drain dance runs because we
    // pass has_immediate_exit=true. The KVM_CAP_IMMEDIATE_EXIT
    // contract returns EINTR from vcpu.run() — handle_freeze logs
    // and continues, NOT panics.
    use crate::vmm::kvm::KtstrKvm;
    use crate::vmm::topology::Topology;
    let topo = Topology {
        llcs: 1,
        cores_per_llc: 1,
        threads_per_core: 1,
        numa_nodes: 1,
        nodes: None,
        distances: None,
        llc_cores: None,
    };
    let mut vm = KtstrKvm::new(topo, 64, false).unwrap();
    // This path enters KVM_RUN (with immediate_exit already armed). Keep the
    // test on the same one-way registration boundary as a real VM.
    vm.register_memory().unwrap();
    crate::vmm::x86_64::boot::setup_sregs(&vm.guest_mem, &vm.vcpus[0], false).unwrap();
    let run_size = vm.vm_fd.run_size();
    let (mut vcpu, _) =
        crate::vmm::vcpu::ImmediateExitVcpu::new(vm.vcpus.remove(0), true, run_size).unwrap();

    let kill = std::sync::Arc::new(AtomicBool::new(false));
    let freeze = std::sync::Arc::new(AtomicBool::new(false));
    let parked = std::sync::Arc::new(AtomicBool::new(false));
    let regs_slot = std::sync::Arc::new(std::sync::Mutex::new(None));

    // freeze=false → the park loop exits immediately; the function
    // returns after the drain + reg capture + parked toggle.
    // has_immediate_exit=true exercises the EINTR drain path.
    handle_freeze(
        &mut vcpu, true, &kill, &freeze, &parked, &regs_slot, None, None, None,
    );

    // After handle_freeze returns:
    //   - parked must be false (Released on exit so subsequent
    //     freeze cycles are observable).
    //   - regs_slot must hold Some(_) — capture_vcpu_regs ran
    //     successfully on the freshly-init'd vCPU.
    assert!(
        !parked.load(Ordering::Acquire),
        "parked must be cleared on exit so subsequent freeze \
         cycles can observe a fresh true→false edge"
    );
    let snapshot = regs_slot.lock().unwrap();
    assert!(
        snapshot.is_some(),
        "capture_vcpu_regs must populate regs_slot for a freshly-\
         init'd vCPU — None means KVM_GET_REGS failed unexpectedly"
    );
}

#[test]
fn handle_freeze_no_drain_when_immediate_exit_unsupported() {
    // With has_immediate_exit=false, the drain dance is skipped
    // entirely (the doc explains: calling vcpu.run() with the cap
    // absent would re-enter the guest instead of returning EINTR).
    // Verify the function still completes, captures regs, and
    // toggles parked correctly.
    use crate::vmm::kvm::KtstrKvm;
    use crate::vmm::topology::Topology;
    let topo = Topology {
        llcs: 1,
        cores_per_llc: 1,
        threads_per_core: 1,
        numa_nodes: 1,
        nodes: None,
        distances: None,
        llc_cores: None,
    };
    let mut vm = KtstrKvm::new(topo, 64, false).unwrap();
    crate::vmm::x86_64::boot::setup_sregs(&vm.guest_mem, &vm.vcpus[0], false).unwrap();
    let run_size = vm.vm_fd.run_size();
    let (mut vcpu, _) =
        crate::vmm::vcpu::ImmediateExitVcpu::new(vm.vcpus.remove(0), false, run_size).unwrap();

    let kill = std::sync::Arc::new(AtomicBool::new(false));
    let freeze = std::sync::Arc::new(AtomicBool::new(false));
    let parked = std::sync::Arc::new(AtomicBool::new(false));
    let regs_slot = std::sync::Arc::new(std::sync::Mutex::new(None));

    // has_immediate_exit=false skips the drain dance.
    handle_freeze(
        &mut vcpu, false, &kill, &freeze, &parked, &regs_slot, None, None, None,
    );

    assert!(!parked.load(Ordering::Acquire));
    assert!(regs_slot.lock().unwrap().is_some());
}

#[test]
fn handle_freeze_writes_parked_evt_edge() {
    // The parked_evt write is a wake edge for the freeze
    // coordinator's epoll loop. Verify exactly one byte/edge gets
    // written by reading back the eventfd counter post-call —
    // an EFD_NONBLOCK eventfd returns the accumulated count on
    // read and resets to 0.
    use crate::vmm::kvm::KtstrKvm;
    use crate::vmm::topology::Topology;
    let topo = Topology {
        llcs: 1,
        cores_per_llc: 1,
        threads_per_core: 1,
        numa_nodes: 1,
        nodes: None,
        distances: None,
        llc_cores: None,
    };
    let mut vm = KtstrKvm::new(topo, 64, false).unwrap();
    crate::vmm::x86_64::boot::setup_sregs(&vm.guest_mem, &vm.vcpus[0], false).unwrap();
    let run_size = vm.vm_fd.run_size();
    let (mut vcpu, _) =
        crate::vmm::vcpu::ImmediateExitVcpu::new(vm.vcpus.remove(0), false, run_size).unwrap();

    let kill = std::sync::Arc::new(AtomicBool::new(false));
    let freeze = std::sync::Arc::new(AtomicBool::new(false));
    let parked = std::sync::Arc::new(AtomicBool::new(false));
    let regs_slot = std::sync::Arc::new(std::sync::Mutex::new(None));
    // EFD_NONBLOCK matches production usage in freeze_coord.
    let parked_evt = EventFd::new(vmm_sys_util::eventfd::EFD_NONBLOCK).unwrap();

    handle_freeze(
        &mut vcpu,
        false,
        &kill,
        &freeze,
        &parked,
        &regs_slot,
        Some(&parked_evt),
        None,
        None,
    );

    // EventFd::read returns the accumulated counter and resets it.
    // We expect exactly 1 (one edge from the write(1) inside
    // handle_freeze).
    let counter = parked_evt.read().unwrap();
    assert_eq!(
        counter, 1,
        "handle_freeze must write exactly one wake edge to \
         parked_evt — coordinator depends on this to advance \
         from epoll_wait without spurious extra wakes"
    );
}
