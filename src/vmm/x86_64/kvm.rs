use anyhow::{Context, Result};
use kvm_bindings::{
    KVM_CAP_HALT_POLL, KVM_CAP_SPLIT_IRQCHIP, KVM_CAP_X2APIC_API, KVM_CAP_X86_DISABLE_EXITS,
    KVM_CLOCK_TSC_STABLE, KVM_IRQ_ROUTING_IRQCHIP, KVM_IRQ_ROUTING_MSI, KVM_IRQCHIP_IOAPIC,
    KVM_IRQCHIP_PIC_MASTER, KVM_IRQCHIP_PIC_SLAVE, KVM_PIT_SPEAKER_DUMMY,
    KVM_X2APIC_API_DISABLE_BROADCAST_QUIRK, KVM_X2APIC_API_USE_32BIT_IDS,
    KVM_X86_DISABLE_EXITS_HLT, KVM_X86_DISABLE_EXITS_PAUSE, KvmIrqRouting, kvm_enable_cap,
    kvm_irq_routing, kvm_irq_routing_entry, kvm_irq_routing_entry__bindgen_ty_1,
    kvm_irq_routing_irqchip, kvm_irq_routing_msi, kvm_irq_routing_msi__bindgen_ty_1,
    kvm_pit_config,
};
use kvm_ioctls::{Cap, Kvm, VcpuFd, VmFd};
use std::mem::ManuallyDrop;
use std::sync::Arc;
use vm_memory::{GuestAddress, GuestMemoryMmap};

use super::ioapic::{IOAPIC_BASE, IOAPIC_SIZE, Ioapic, MsiRoute};
use super::topology::{apic_id, generate_cpuid, max_apic_id};
use crate::vmm::numa_mem::{MemoryBacking, NumaMemoryLayout, ReservationGuard};
use crate::vmm::pi_mutex::PiMutex;
use crate::vmm::topology::Topology;

/// Physical address where the kernel is loaded.
pub(crate) const KERNEL_LOAD_ADDR: u64 = 0x100000; // 1 MB

/// Physical address of boot parameters (zero page).
pub(crate) const BOOT_PARAMS_ADDR: u64 = 0x7000;

/// Physical address of the kernel command line.
pub(crate) const CMDLINE_ADDR: u64 = 0x20000;

/// Maximum command line length.
pub(crate) const CMDLINE_MAX: usize = 4096;

// ---- Memory layout constants shared by boot.rs and acpi.rs ----

/// End of Extended BIOS Data Area (640K - 1K).
pub(crate) const EBDA_START: u64 = 0x9FC00;

/// Start of high memory (1 MB).
pub(crate) const HIMEM_START: u64 = 0x10_0000;

/// Start of PCI MMIO gap (3 GB). Memory below this is usable RAM.
pub(crate) const MMIO_GAP_START: u64 = 0xC000_0000;

/// End of PCI MMIO gap (4 GB). Memory above this resumes as RAM.
pub(crate) const MMIO_GAP_END: u64 = 0x1_0000_0000;

/// Virtio-console MMIO base: start of the MMIO gap.
pub(crate) const VIRTIO_CONSOLE_MMIO_BASE: u64 = MMIO_GAP_START;

/// Virtio-block MMIO base: one page above virtio-console.
/// Each virtio-mmio device occupies `VIRTIO_MMIO_SIZE = 0x1000`.
pub(crate) const VIRTIO_BLK_MMIO_BASE: u64 = MMIO_GAP_START + 0x1000;

/// Virtio-net MMIO base: one page above virtio-blk.
/// Each virtio-mmio device occupies `VIRTIO_MMIO_SIZE = 0x1000`.
pub(crate) const VIRTIO_NET_MMIO_BASE: u64 = MMIO_GAP_START + 0x2000;

/// IOAPIC MMIO base. Lives inside the MMIO gap; the PCI windows below must
/// avoid it. Mirrors the canonical definitions (ioapic.rs `IOAPIC_BASE`, acpi
/// MADT `IOAPIC_ADDR`) for the compile-time disjointness checks.
const IOAPIC_MMIO_BASE: u64 = 0xFEC0_0000;

/// Local APIC / MSI MMIO base. Also inside the gap; the PCI windows avoid it.
/// Mirrors acpi `LAPIC_ADDR`.
const LAPIC_MMIO_BASE: u64 = 0xFEE0_0000;

/// PCI ECAM (Enhanced Configuration Access Mechanism) window base. Holds one
/// PCI bus (256 functions × 4 KiB = 1 MiB), carved from the MMIO gap above the
/// three virtio-mmio device pages and below the IOAPIC. The ACPI MCFG table
/// publishes this base to the guest, which maps it for extended config access.
pub(crate) const PCI_ECAM_BASE: u64 = 0xE000_0000;

/// PCI ECAM window size: one bus.
pub(crate) const PCI_ECAM_SIZE: u64 = crate::vmm::pci::ECAM_BYTES_PER_BUS;

/// PCI 32-bit MMIO BAR window base. Device BARs are placed here; sits above
/// the ECAM window and below the IOAPIC. Advertised to the guest via the DSDT
/// `_SB.PCI0` `_CRS` so Linux assigns BARs within it.
pub(crate) const PCI_MMIO_BAR_BASE: u64 = PCI_ECAM_BASE + PCI_ECAM_SIZE;

/// PCI 32-bit MMIO BAR window size: from the BAR base up to the IOAPIC.
pub(crate) const PCI_MMIO_BAR_SIZE: u64 = IOAPIC_MMIO_BASE - PCI_MMIO_BAR_BASE;

// Compile-time PCI address-space disjointness. x86_64 had no equivalent
// (only struct-size asserts), so this block is new: the ECAM and BAR windows
// must lie within the MMIO gap, above the three virtio-mmio device pages
// [MMIO_GAP_START, MMIO_GAP_START + 0x3000), not overlap each other, and end
// before both the IOAPIC and the LAPIC/MSI region (both inside the gap).
const _: () = {
    assert!(PCI_ECAM_BASE >= MMIO_GAP_START + 0x3000);
    assert!(PCI_ECAM_BASE + PCI_ECAM_SIZE <= PCI_MMIO_BAR_BASE);
    assert!(PCI_MMIO_BAR_BASE + PCI_MMIO_BAR_SIZE <= IOAPIC_MMIO_BASE);
    assert!(PCI_ECAM_BASE + PCI_ECAM_SIZE <= LAPIC_MMIO_BASE);
    assert!(PCI_MMIO_BAR_BASE + PCI_MMIO_BAR_SIZE <= LAPIC_MMIO_BASE);
    assert!(IOAPIC_MMIO_BASE < MMIO_GAP_END);
    assert!(LAPIC_MMIO_BASE < MMIO_GAP_END);
};

/// GSI for virtio-console. On the in-kernel-irqchip path the in-kernel
/// IOAPIC routes this GSI; on split-irqchip (>254 APIC IDs) the userspace
/// IOAPIC translates the guest's RTE for it into an MSI route.
pub(crate) const VIRTIO_CONSOLE_IRQ: u32 = 5;

/// GSI for virtio-block. Routed via the in-kernel IOAPIC (<=254 max APIC ID)
/// or the userspace IOAPIC (split-irqchip, >254 max APIC ID). The IOAPIC's
/// 24-line cap leaves ample free slots after COM1=4, COM2=3, virtio-console=5,
/// virtio-blk=6.
pub(crate) const VIRTIO_BLK_IRQ: u32 = 6;

/// Maximum number of virtio-net NICs on the PCI bus (x86_64). Bounded by the
/// IOAPIC's 24-line budget for INTx: each NIC needs one INTx GSI disjoint from
/// the reserved lines (PIT GSI2, COM2=3, COM1=4, virtio-console=5,
/// virtio-blk=6, SCI=9), leaving 7,8,10..=23 = 16 usable. (Per-queue MSI-X
/// routes do not consume IOAPIC pins, so MSI-X does not lift this cap; it also
/// bounds the per-NIC MSI-X GSI budget — see [`MSIX_VECTORS_PER_NIC`].) The
/// builder rejects more than this many `.network()` calls.
pub(crate) const MAX_VIRTIO_NICS: usize = 16;

/// PCI slot (bus 0) for virtio-net NIC `index` (0-based): slot 0 is the host
/// bridge, NICs take slots 1..=MAX_VIRTIO_NICS. One source of truth shared by
/// the install site (`PciBus::add_function`) and the DSDT `_PRT` route, so the
/// routed slot always matches the installed slot.
pub(crate) const fn virtio_net_pci_slot(index: usize) -> usize {
    1 + index
}

/// INTx GSI for virtio-net NIC `index` (0-based). Routed via the in-kernel
/// IOAPIC (<=254 max APIC ID) or the userspace IOAPIC (split-irqchip, >254 max
/// APIC ID). Allocated from 7 upward, SKIPPING the SCI GSI ([`ACPI_SCI_IRQ`]=9,
/// which the guest programs at acpi_enable), so the NIC lines stay disjoint from
/// COM2=3/COM1=4/console=5/blk=6/SCI=9: index 0->7, 1->8, 2->10, ... 15->23 (the
/// IOAPIC's last line). The NIC delivers legacy INTx on this GSI (level,
/// active-low); the DSDT `_PRT` routes `(virtio_net_pci_slot(index), INTA#)`
/// here. PCI device GSIs deliberately get NO MADT Interrupt Source Override: an
/// ISO overrides only the ISA-bus IRQ->GSI identity/polarity defaults, and these
/// are PCI lines whose level/active-low polarity the guest programs from `_PRT`
/// via acpi_pci_irq_enable (the lone MADT ISO is IRQ0->GSI2 for the PIT).
pub(crate) const fn virtio_net_gsi(index: usize) -> u32 {
    let raw = 7 + index as u32;
    // Skip the SCI line: every GSI at or above it shifts up by one.
    if raw >= ACPI_SCI_IRQ as u32 {
        raw + 1
    } else {
        raw
    }
}

/// KVM's hard cap on GSI routing-table entries and the exclusive upper bound on
/// any routed GSI: `kvm_set_irq_routing` rejects `gsi >= KVM_MAX_IRQ_ROUTES`
/// (virt/kvm/irqchip.c) and the `KVM_SET_GSI_ROUTING` ioctl rejects a table with
/// more than this many entries (virt/kvm/kvm_main.c). 4096 per
/// include/linux/kvm_host.h. Hardcoded — the value is not in the uapi headers
/// (the host also surfaces it via the `KVM_CAP_IRQ_ROUTING` extension) — and
/// guarded by the const-asserts below.
pub(crate) const KVM_MAX_IRQ_ROUTES: usize = 4096;

/// Per-NIC MSI-X GSI budget: the stride between NICs' vector ranges and the
/// per-NIC cap on advertised MSI-X vectors. Each NIC's table vectors occupy
/// `[NUM_IOAPIC_PINS + nic*MSIX_VECTORS_PER_NIC, +MSIX_VECTORS_PER_NIC)`,
/// disjoint across NICs. Sized as the largest stride for which even
/// `MAX_VIRTIO_NICS` fully-populated NICs fit KVM's route table
/// (`NUM_IOAPIC_PINS + MAX_VIRTIO_NICS * stride <= KVM_MAX_IRQ_ROUTES`), so the
/// host never overflows the table and no config needs a runtime bail. With the
/// 24-pin IOAPIC base and 16 NICs this is `(4096 - 24) / 16 = 254`. The device
/// caps its advertised vector count at this (and at the MSI-X table-page
/// capacity [`crate::vmm::virtio_msix::MSIX_TABLE_MAX`]); a NIC wanting more
/// queue-pairs than `254` vectors allow (> 126 pairs) falls back to the guest's
/// SHARED vector policy, which the per-queue signal path serves transparently.
pub(crate) const MSIX_VECTORS_PER_NIC: usize =
    (KVM_MAX_IRQ_ROUTES - NUM_IOAPIC_PINS as usize) / MAX_VIRTIO_NICS;

// The per-NIC GSI budget must not exceed the device's MSI-X table-page capacity
// (the device cannot advertise more vectors than its one-page table holds, so
// MsixState::new caps num_vectors at this stride knowing it is the binding one).
const _: () = assert!(
    MSIX_VECTORS_PER_NIC <= crate::vmm::virtio_msix::MSIX_TABLE_MAX,
    "per-NIC MSI-X GSI budget exceeds the device MSI-X table-page capacity"
);
// All NICs' vector ranges plus the IOAPIC pin range must fit KVM's routing
// table (every routed GSI < KVM_MAX_IRQ_ROUTES); the floor division above makes
// this hold, and the assert pins it against a future MAX_VIRTIO_NICS bump.
const _: () = assert!(
    NUM_IOAPIC_PINS as usize + MAX_VIRTIO_NICS * MSIX_VECTORS_PER_NIC <= KVM_MAX_IRQ_ROUTES,
    "virtio-net MSI-X GSIs overflow the KVM routing table"
);

/// MSI-X vector GSI for virtio-net NIC `nic` (0-based), table vector `vector`
/// (`vector < MSIX_VECTORS_PER_NIC`). MSI-X vectors deliver as
/// `KVM_IRQ_ROUTING_MSI` routes, which — unlike INTx IOAPIC routes — are NOT
/// bounded by the 24-pin IOAPIC budget, only by [`KVM_MAX_IRQ_ROUTES`]. So the
/// MSI-X GSIs live ABOVE the IOAPIC pin range, disjoint from every INTx GSI
/// ([`virtio_net_gsi`] ∈ [7, 24)): `NUM_IOAPIC_PINS + nic*MSIX_VECTORS_PER_NIC +
/// vector`. Each NIC's range is `MSIX_VECTORS_PER_NIC` wide, so the ranges are
/// disjoint and the highest GSI (`NUM_IOAPIC_PINS + MAX_VIRTIO_NICS *
/// MSIX_VECTORS_PER_NIC - 1`) stays below the budget. The host registers one
/// irqfd per (NIC, vector) at this GSI at VM bring-up (an irqfd may be assigned
/// to a GSI with no route yet — virt/kvm/eventfd.c `irqfd_update` leaves the
/// cache inactive); the MSI route is installed on the guest's vector-unmask edge
/// by the route owner, which refreshes the irqfd's cached entry via
/// `kvm_irq_routing_update`.
pub(crate) const fn virtio_net_msix_gsi(nic: usize, vector: usize) -> u32 {
    NUM_IOAPIC_PINS as u32 + (nic * MSIX_VECTORS_PER_NIC + vector) as u32
}

/// PCI slot (bus 0) for the single virtio-block PCI function: after the NIC
/// slots (1..=MAX_VIRTIO_NICS), so blk takes slot `MAX_VIRTIO_NICS + 1`. One
/// source of truth shared by `PciBus::add_function` and the DSDT `_PRT`.
pub(crate) const fn virtio_blk_pci_slot() -> usize {
    1 + MAX_VIRTIO_NICS
}

/// INTx GSI for the virtio-block PCI function: reuses [`VIRTIO_BLK_IRQ`] (=6) —
/// the IOAPIC pin the MMIO blk transport uses. On the x86 PCI path blk is on PCI
/// (not MMIO), so that pin is free; the DSDT `_PRT` routes
/// `(virtio_blk_pci_slot(), INTA#)` here (level, active-low, programmed by the
/// guest via acpi_pci_irq_enable, like the NIC INTx lines).
pub(crate) const fn virtio_blk_pci_gsi() -> u32 {
    VIRTIO_BLK_IRQ
}

/// Per-device MSI-X GSI budget for the virtio-block PCI function — the cap on its
/// advertised vectors. blk is single-queue (advertises `num_queues + 1` = 2),
/// so the budget is the GSIs left after the IOAPIC pins and all NIC MSI-X ranges
/// (`[24 + 16*254, 4096)` = 8 GSIs); ample for 2 vectors. Caps `MsixState::new`.
pub(crate) const MSIX_VECTORS_PER_BLK: usize =
    KVM_MAX_IRQ_ROUTES - NUM_IOAPIC_PINS as usize - MAX_VIRTIO_NICS * MSIX_VECTORS_PER_NIC;

/// MSI-X vector GSI for the virtio-block PCI function, table vector `vector`
/// (`vector < MSIX_VECTORS_PER_BLK`). Lives ABOVE the IOAPIC pins AND every NIC
/// MSI-X range, disjoint from both: `NUM_IOAPIC_PINS + MAX_VIRTIO_NICS *
/// MSIX_VECTORS_PER_NIC + vector`. A single blk device, so no per-device stride.
pub(crate) const fn virtio_blk_pci_msix_gsi(vector: usize) -> u32 {
    (NUM_IOAPIC_PINS as usize + MAX_VIRTIO_NICS * MSIX_VECTORS_PER_NIC + vector) as u32
}

// The blk MSI-X range plus the IOAPIC pins and all NIC MSI-X ranges must fit
// KVM's routing table. By construction MSIX_VECTORS_PER_BLK is exactly the
// remainder, so the total equals KVM_MAX_IRQ_ROUTES; pinned against a future
// MAX_VIRTIO_NICS / MSIX_VECTORS_PER_NIC change.
const _: () = assert!(
    NUM_IOAPIC_PINS as usize + MAX_VIRTIO_NICS * MSIX_VECTORS_PER_NIC + MSIX_VECTORS_PER_BLK
        <= KVM_MAX_IRQ_ROUTES,
    "virtio-blk MSI-X GSIs overflow the KVM routing table"
);
// blk advertises num_queues + 1 = 2 vectors; the budget must hold them and not
// exceed the MSI-X table-page capacity.
const _: () = assert!(
    MSIX_VECTORS_PER_BLK >= 2 && MSIX_VECTORS_PER_BLK <= crate::vmm::virtio_msix::MSIX_TABLE_MAX,
    "virtio-blk MSI-X GSI budget must hold >=2 vectors and fit the table page"
);

/// GSI the FADT advertises as the ACPI SCI (system control interrupt). The
/// non-hardware-reduced FADT must name one; ktstr never arms a fixed event or
/// a GPE block, so the SCI never fires — this GSI is reserved (inert) and has
/// no MADT Interrupt Source Override (an ISO declaring it level/active-low
/// would only matter if the SCI could fire). Inert does NOT mean untouched:
/// the guest still PROGRAMS the GSI-9 IOAPIC RTE (level/active-low) during
/// `acpi_enable` (acpi_os_install_interrupt_handler / acpi_sci_ioapic_setup),
/// and on the split-irqchip path that triggers a KVM_SET_GSI_ROUTING rebuild
/// for GSI 9 — but the line maps to no device irqfd, so the programmed route
/// delivers nothing. Named here so its disjointness from the device GSIs is
/// checkable in one place: serial COM1=4/COM2=3, virtio-MMIO console=5,
/// virtio-blk [`VIRTIO_BLK_IRQ`]=6, virtio-net base GSI 7 (`virtio_net_gsi(0)`)
/// — 9 collides with none. If a fixed event or GPE is ever armed, add a MADT ISO
/// declaring this GSI level/active-low.
pub(crate) const ACPI_SCI_IRQ: u16 = 9;

/// ACPI PM register-block I/O ports (non-hardware-reduced FADT). The guest's
/// ACPICA writes/reads these during `acpi_enable` + fixed-event init; ktstr
/// emulates them as stateless stubs (no real power management). Advertising
/// them is what lets ACPI FULLY enable — required so the guest routes PCI
/// INTx via the DSDT `_PRT` (`acpi_pci_irq_enable`); a partially-enabled
/// ACPI (zeroed PM blocks → `AE_BAD_ADDRESS`) aborts and falls back to
/// legacy MP-table PCI IRQ routing, which has no NIC entry. Layout (qemu
/// i440fx/q35 model — full ACPI + PM blocks + legacy IRQs, NOT the
/// firecracker/cloud-hypervisor hardware-reduced microvm path which drops
/// the legacy 8259 IRQs ktstr's cmdline virtio-MMIO devices depend on):
///   PM1 event block  (4 bytes: status u16 @ +0, enable u16 @ +2)
///   PM1 control block (2 bytes: SCI_EN bit0)
///   PM timer block    (4 bytes: 24-bit free-running counter)
pub(crate) const ACPI_PM1_EVT_PORT: u16 = 0x600;
pub(crate) const ACPI_PM1_CNT_PORT: u16 = 0x604;
pub(crate) const ACPI_PM_TMR_PORT: u16 = 0x608;

/// E820 memory type: usable RAM.
pub(crate) const E820_RAM: u32 = 1;

/// Offset from code32_start to 64-bit entry point in bzImage.
pub(crate) const STARTUP64_OFFSET: u64 = 0x200;

/// TSS address — same as Firecracker/libkrun.
const KVM_TSS_ADDRESS: u64 = 0xfffb_d000;

/// Identity map address — placed immediately after the 3-page TSS region.
/// KVM requires this to be set before creating vCPUs on x86_64.
const KVM_IDENTITY_MAP_ADDRESS: u64 = KVM_TSS_ADDRESS + 3 * 4096;

/// IOAPIC supports 24 input pins (IRQ 0-23).
const NUM_IOAPIC_PINS: u64 = 24;

/// APIC IDs above this require x2APIC mode (8-bit xAPIC limit).
pub(crate) const MAX_XAPIC_ID: u32 = 254;

/// Required KVM capabilities — Firecracker checks these 14.
const REQUIRED_CAPS: &[Cap] = &[
    Cap::Irqchip,
    Cap::Ioeventfd,
    Cap::Irqfd,
    Cap::UserMemory,
    Cap::SetTssAddr,
    Cap::Pit2,
    Cap::PitState2,
    Cap::AdjustClock,
    Cap::Debugregs,
    Cap::MpState,
    Cap::VcpuEvents,
    Cap::Xcrs,
    Cap::Xsave,
    Cap::ExtCpuid,
];

/// A KVM virtual machine with configured topology.
#[allow(dead_code)] // configuration fields read conditionally; reservation held for RAII drop
pub struct KtstrKvm {
    pub kvm: ManuallyDrop<Kvm>,
    pub vm_fd: ManuallyDrop<VmFd>,
    pub vcpus: Vec<VcpuFd>,
    pub guest_mem: ManuallyDrop<GuestMemoryMmap>,
    pub topology: Topology,
    /// Per-node GPA layout used by ACPI SRAT/HMAT generation. `None`
    /// in deferred mode before `allocate_and_register_memory()`.
    pub(crate) numa_layout: Option<NumaMemoryLayout>,
    /// Actual backing selected by the allocator. `None` only before deferred
    /// memory allocation.
    pub(crate) memory_backing: Option<MemoryBacking>,
    /// Whether KVM supports the immediate_exit mechanism (KVM_CAP_IMMEDIATE_EXIT).
    pub has_immediate_exit: bool,
    /// Split IRQ chip mode: LAPIC in kernel, PIC/IOAPIC emulated in userspace.
    /// Enabled when any APIC ID exceeds the 8-bit xAPIC limit (254).
    pub(crate) split_irqchip: bool,
    /// Userspace IOAPIC device, present only on the split-irqchip path.
    /// The run loops wrap it in an [`IoapicHandle`] (device + raw VM fd) to
    /// service IOAPIC MMIO and reprogram MSI routes; `None` for <=254-max-APIC-ID
    /// guests, which use the in-kernel IOAPIC.
    pub(crate) ioapic: Option<Arc<PiMutex<Ioapic>>>,
    /// Whether hugepages were requested at construction time.
    /// Stored so deferred memory allocation uses the same backing.
    use_hugepages: bool,
    /// Performance mode flag. Stored so deferred memory allocation
    /// can check hugepage availability fresh when memory_mib was
    /// unknown at construction time.
    performance_mode: bool,
    /// Owns the VA reservation for per-node MAP_FIXED mmaps.
    /// Drop munmaps the entire reservation.
    _reservation: Option<ReservationGuard>,
    /// RAII guards for direct-COW initramfs mappings. Each guard holds its
    /// immutable CAS object fd with `LOCK_SH`. Must drop AFTER `_reservation`
    /// so GC cannot unlink the backing before the COW VMAs are torn down.
    pub(crate) cow_overlay_guards: Vec<crate::vmm::initramfs::CowOverlayGuard>,
    /// Whether this VM exposes the virtio-PCI transport (a PCI host bridge with
    /// ECAM/CAM config access). `false` keeps the guest on virtio-MMIO only —
    /// no PCI ACPI tables, `pci=off` retained — byte-identical to a non-PCI
    /// boot. Set when a NIC is placed on PCI (or by a transport test).
    pub(crate) pci_enabled: bool,
}

impl Drop for KtstrKvm {
    fn drop(&mut self) {
        unsafe {
            // Ordered teardown: vCPU fds → VM fd → guest memory →
            // VA reservation → COW flock guards → /dev/kvm.
            //
            // Closing VmFd triggers kvm_destroy_vm which calls
            // mmu_notifier_unregister (synchronous SRCU wait). All
            // KVM references to this process's page tables are removed
            // before the guest memory munmap fires, preventing stale
            // mmu_notifier callbacks from racing with the unmap.
            let vcpus = std::mem::take(&mut self.vcpus);
            drop(vcpus);
            ManuallyDrop::drop(&mut self.vm_fd);
            ManuallyDrop::drop(&mut self.guest_mem);
            let reservation = self._reservation.take();
            drop(reservation);
            let cow_guards = std::mem::take(&mut self.cow_overlay_guards);
            drop(cow_guards);
            ManuallyDrop::drop(&mut self.kvm);
        }
    }
}

impl KtstrKvm {
    /// Create a new KVM VM with the given topology and memory size.
    pub fn new(topo: Topology, memory_mib: u32, performance_mode: bool) -> Result<Self> {
        Self::new_inner(topo, Some(memory_mib), false, performance_mode)
    }

    /// Create a new KVM VM with hugepage-backed guest memory.
    pub fn new_with_hugepages(
        topo: Topology,
        memory_mib: u32,
        performance_mode: bool,
    ) -> Result<Self> {
        Self::new_inner(topo, Some(memory_mib), true, performance_mode)
    }

    /// Create a KVM VM without allocating guest memory.
    ///
    /// Sets up /dev/kvm, VM fd, TSS, identity map, IRQ chip, vCPUs, and
    /// CPUID — none of which depend on guest memory size. Memory is
    /// allocated later via `allocate_and_register_memory`.
    pub fn new_deferred(
        topo: Topology,
        use_hugepages: bool,
        performance_mode: bool,
    ) -> Result<Self> {
        Self::new_inner(topo, None, use_hugepages, performance_mode)
    }

    /// Allocate guest memory and register it with KVM.
    ///
    /// Should be called exactly once on a VM created with
    /// `new_deferred`; calling twice unconditionally replaces the
    /// backing memory. Replaces the placeholder guest memory with a
    /// real allocation of `memory_mib` mebibytes and sets
    /// `numa_layout` to the computed per-node GPA layout. Re-checks
    /// hugepage availability when performance_mode is set, since
    /// memory_mib was unknown at construction time and `use_hugepages`
    /// may have been false.
    pub fn allocate_and_register_memory(&mut self, memory_mib: u32) -> Result<()> {
        let layout = NumaMemoryLayout::compute(
            &self.topology,
            memory_mib,
            0,
            Some((MMIO_GAP_START, MMIO_GAP_END)),
        )?;
        let alloc =
            layout.allocate_and_register(&self.vm_fd, self.use_hugepages, self.performance_mode)?;
        // SAFETY: this is the only call to ManuallyDrop::drop on
        // self.guest_mem; the next line replaces it with
        // ManuallyDrop::new(...).
        unsafe { ManuallyDrop::drop(&mut self.guest_mem) };
        self.guest_mem = ManuallyDrop::new(alloc.guest_mem);
        self._reservation = Some(alloc.reservation);
        self.memory_backing = Some(alloc.backing);
        self.numa_layout = Some(layout);
        Ok(())
    }

    fn new_inner(
        topo: Topology,
        memory_mib: Option<u32>,
        use_hugepages: bool,
        performance_mode: bool,
    ) -> Result<Self> {
        let kvm = Kvm::new().context("open /dev/kvm")?;

        // Check required capabilities (Firecracker pattern)
        for &cap in REQUIRED_CAPS {
            anyhow::ensure!(
                kvm.check_extension(cap),
                "KVM missing required capability: {:?}",
                cap
            );
        }

        let has_immediate_exit = kvm.check_extension(Cap::ImmediateExit);

        let vm_fd = crate::vmm::create_vm_with_retry(&kvm)?;

        // TSS (required on x86_64 before creating vCPUs). Maps
        // transient host errnos (ENOMEM, EBUSY) into
        // ResourceContention so the macro SKIPs cleanly instead of
        // panicking under host-resource pressure.
        vm_fd
            .set_tss_address(KVM_TSS_ADDRESS as usize)
            .map_err(|e| crate::vmm::map_transient_to_contention(e, "set TSS"))?;

        // Identity map — one page after the 3-page TSS region.
        // Must be set before creating vCPUs.
        vm_fd
            .set_identity_map_address(KVM_IDENTITY_MAP_ADDRESS)
            .map_err(|e| crate::vmm::map_transient_to_contention(e, "set identity map address"))?;

        // Determine whether any APIC ID exceeds the 8-bit xAPIC limit.
        // If so, use split IRQ chip (LAPIC-only in kernel) + x2APIC API.
        let max_apic_id = max_apic_id(&topo);
        let split_irqchip = max_apic_id > MAX_XAPIC_ID;

        let ioapic = Self::setup_irqchip(&vm_fd, split_irqchip)?;

        Self::tune_kvm_caps(&vm_fd, performance_mode)?;

        let (guest_mem, numa_layout, reservation, memory_backing) = match memory_mib {
            Some(mb) => {
                let layout =
                    NumaMemoryLayout::compute(&topo, mb, 0, Some((MMIO_GAP_START, MMIO_GAP_END)))?;
                let alloc =
                    layout.allocate_and_register(&vm_fd, use_hugepages, performance_mode)?;
                (
                    alloc.guest_mem,
                    Some(layout),
                    Some(alloc.reservation),
                    Some(alloc.backing),
                )
            }
            None => {
                let placeholder = GuestMemoryMmap::<()>::from_ranges(&[(GuestAddress(0), 4096)])
                    .context("allocate placeholder guest memory")?;
                (placeholder, None, None, None)
            }
        };

        let vcpus = Self::create_vcpus(
            &kvm,
            &vm_fd,
            &topo,
            &numa_layout,
            max_apic_id,
            performance_mode,
        )?;

        Ok(KtstrKvm {
            kvm: ManuallyDrop::new(kvm),
            vm_fd: ManuallyDrop::new(vm_fd),
            vcpus,
            guest_mem: ManuallyDrop::new(guest_mem),
            topology: topo,
            numa_layout,
            memory_backing,
            has_immediate_exit,
            split_irqchip,
            ioapic,
            use_hugepages,
            performance_mode,
            _reservation: reservation,
            cow_overlay_guards: Vec::new(),
            pci_enabled: false,
        })
    }

    /// Set up the IRQ chip: split (LAPIC-only + x2APIC API + userspace
    /// IOAPIC) when `split_irqchip`, else full in-kernel PIC/IOAPIC/LAPIC + PIT.
    fn setup_irqchip(vm_fd: &VmFd, split_irqchip: bool) -> Result<Option<Arc<PiMutex<Ioapic>>>> {
        if split_irqchip {
            // Split IRQ chip: only LAPIC is emulated in kernel.
            // PIC and IOAPIC are not created — userspace handles them.
            let mut cap = kvm_enable_cap {
                cap: KVM_CAP_SPLIT_IRQCHIP,
                ..Default::default()
            };
            cap.args[0] = NUM_IOAPIC_PINS;
            // KVM_CAP_SPLIT_IRQCHIP allocates the in-kernel LAPIC
            // tables and sets up the userspace IRQ routing slots.
            // ENOMEM under host pressure is transient — route through
            // the contention classifier so the macro SKIPs cleanly.
            vm_fd
                .enable_cap(&cap)
                .map_err(|e| crate::vmm::map_transient_to_contention(e, "enable split IRQ chip"))?;

            // Enable x2APIC API for 32-bit destination IDs and correct
            // broadcast behavior with APIC IDs > 254.
            let mut cap = kvm_enable_cap {
                cap: KVM_CAP_X2APIC_API,
                ..Default::default()
            };
            cap.args[0] =
                (KVM_X2APIC_API_USE_32BIT_IDS | KVM_X2APIC_API_DISABLE_BROADCAST_QUIRK) as u64;
            vm_fd.enable_cap(&cap).context("enable x2APIC API")?;
        } else {
            // Full IRQ chip (PIC + IOAPIC + LAPIC) — must exist before KVM_CREATE_VCPU
            vm_fd
                .create_irq_chip()
                .map_err(|e| crate::vmm::map_transient_to_contention(e, "create IRQ chip"))?;

            // PIT (timer) with dummy speaker port.
            // Only created with full IRQ chip — PIT routes through the in-kernel
            // IOAPIC (IRQ 0 -> GSI 2). With split IRQ chip there is no in-kernel
            // IOAPIC, so PIT creation fails.
            let pit_config = kvm_pit_config {
                flags: KVM_PIT_SPEAKER_DUMMY,
                ..Default::default()
            };
            vm_fd
                .create_pit2(pit_config)
                .map_err(|e| crate::vmm::map_transient_to_contention(e, "create PIT"))?;
        }

        // Userspace IOAPIC device for the split-irqchip path (no in-kernel
        // IOAPIC there). `None` for <=254-max-APIC-ID guests. The run loops build an
        // IoapicHandle around this to translate guest RTE writes into MSI
        // routes; see `super::ioapic` and `IoapicHandle`.
        Ok(split_irqchip.then(|| Arc::new(PiMutex::new(Ioapic::new()))))
    }

    /// Tune per-VM KVM caps: disable PAUSE/HLT exits in performance mode.
    ///
    /// The per-VM halt-poll interval is NOT set here. It keys on the
    /// run-lock outcome (1:1 pin vs overcommit fallback vs no-perf), which
    /// is not resolved until `acquire_run_locks` at run time, so
    /// `KtstrVm::run` sets `KVM_CAP_HALT_POLL` after that resolves — the cap
    /// is settable at any time per the KVM API (Documentation/virt/kvm/
    /// api.rst, 7.20: "This capability can be invoked at any time and any
    /// number of times"). See `KtstrKvm::set_halt_poll` and
    /// `KtstrVm::halt_poll_policy`.
    fn tune_kvm_caps(vm_fd: &VmFd, performance_mode: bool) -> Result<()> {
        // Disable PAUSE and HLT VM exits in performance mode.
        // Two separate enable_cap calls: kvm_disable_exits() uses |=
        // (additive), so multiple calls accumulate. Separate calls
        // ensure PAUSE succeeds unconditionally even if HLT is rejected.
        //
        // PAUSE: reduces vmexit overhead during guest spinlocks.
        //        Unconditionally allowed by KVM.
        // HLT:   eliminates the most frequent exit type during boot/idle.
        //        BSP shutdown uses I8042 reset (port 0x64, 0xFE via
        //        reboot=k) and VcpuExit::Shutdown, not VcpuExit::Hlt.
        //        KVM blocks HLT disable when mitigate_smt_rsb is active
        //        (host has X86_BUG_SMT_RSB and cpu_smt_possible()).
        if performance_mode {
            let mut cap = kvm_enable_cap {
                cap: KVM_CAP_X86_DISABLE_EXITS,
                ..Default::default()
            };

            // 1. PAUSE — always allowed.
            cap.args[0] = KVM_X86_DISABLE_EXITS_PAUSE as u64;
            if let Err(e) = vm_fd.enable_cap(&cap) {
                eprintln!(
                    "performance_mode: WARNING: \
                     KVM_CAP_X86_DISABLE_EXITS (PAUSE) not supported: {e}"
                );
            }

            // 2. HLT — may fail on mitigate_smt_rsb hosts.
            cap.args[0] = KVM_X86_DISABLE_EXITS_HLT as u64;
            if let Err(e) = vm_fd.enable_cap(&cap) {
                eprintln!(
                    "performance_mode: WARNING: \
                     KVM_CAP_X86_DISABLE_EXITS (HLT) rejected: {e}"
                );
            }
        }

        Ok(())
    }

    /// Set the per-VM halt-poll interval via `KVM_CAP_HALT_POLL`.
    ///
    /// Called from `KtstrVm::run` after `acquire_run_locks` resolves the
    /// mode/outcome the policy keys on (see `KtstrVm::halt_poll_policy`). The
    /// cap is a VM-target capability settable at any time per the KVM API
    /// (Documentation/virt/kvm/api.rst, 7.20), so setting it post-vCPU-create
    /// is valid. A `halt_poll_ns` of 0 disables host halt polling for this
    /// VM; the policy never SETS the module default (a fitting 1:1 default-mode
    /// VM is left at whatever `kvm.halt_poll_ns` the host module chose,
    /// matching the prior 200_000 == KVM_HALT_POLL_NS_DEFAULT on stock x86).
    ///
    /// Best-effort: a pre-5.9 host without the cap warns once and continues
    /// on the module default rather than failing the run.
    pub(crate) fn set_halt_poll(&self, halt_poll_ns: u64) {
        let mut cap = kvm_enable_cap {
            cap: KVM_CAP_HALT_POLL,
            ..Default::default()
        };
        cap.args[0] = halt_poll_ns;
        if let Err(e) = self.vm_fd.enable_cap(&cap) {
            static WARNED: std::sync::Once = std::sync::Once::new();
            WARNED.call_once(|| {
                eprintln!(
                    "kvm: WARNING: KVM_CAP_HALT_POLL not supported ({e}), using kernel default"
                );
            });
        }
    }

    /// Create the per-vCPU fds with topology-specific CPUID, after the
    /// RAM-top / max-vcpus / max-vcpu-id host-capability checks, then probe
    /// TSC stability via a KVM_GET_CLOCK roundtrip in performance mode.
    fn create_vcpus(
        kvm: &Kvm,
        vm_fd: &VmFd,
        topo: &Topology,
        numa_layout: &Option<NumaMemoryLayout>,
        max_apic_id: u32,
        performance_mode: bool,
    ) -> Result<Vec<VcpuFd>> {
        // Fetch host CPUID once, reuse for all vCPUs (Firecracker pattern).
        let base_cpuid = kvm
            .get_supported_cpuid(kvm_bindings::KVM_MAX_CPUID_ENTRIES)
            .context("get_supported_cpuid")?;

        // Reject a VM whose RAM relocates above the guest's addressable
        // physical space. generate_cpuid leaves CPUID leaf 0x8000_0008 EAX =
        // the host's MAXPHYADDR (it patches only ECX), so the guest's
        // phys-addr width is the host's. Bits above the guest MAXPHYADDR are
        // architecturally reserved in guest PTEs (KVM sets reserved_gpa_bits
        // = rsvd_bits(cpuid_maxphyaddr, 63)), so a PTE mapping a GPA above
        // 1<<phys_bits faults on the MMU walk's reserved-bits check — RAM with
        // a top above it is unreachable even though it is advertised. Reject
        // pre-create: a permanent host-capability
        // shortfall (the host MAXPHYADDR is fixed hardware), same skip class
        // as the max_vcpus check below. (Distinct from the kernel's separate,
        // wider e820 cap at max_arch_pfn = 1<<MAX_PHYSMEM_BITS (46/52), which
        // does not key on the CPUID MAXPHYADDR.)
        //
        // Reference note (how other VMMs treat the host MAXPHYADDR): qemu
        // hard-fails ("phys-bits too low") against a WIDER bound — phys_bits
        // vs pc_max_used_gpa = max(RAM top, high-MMIO top). cloud-hypervisor
        // caps the guest MAXPHYADDR at min(host, max_phys_bits) (default 46)
        // and sizes the MMIO area from it, with no RAM-top rejection.
        // firecracker and libkrun leave the host MAXPHYADDR with no
        // 1<<phys_bits RAM bound. ktstr exposes
        // the host MAXPHYADDR unmodified and rejects: a SUBSET of qemu's wider
        // bound (equal today — ktstr's sole MMIO window is the sub-4G gap,
        // below relocated RAM, so RAM top is the true max GPA) and a superset
        // of the firecracker/libkrun no-bound case. Bounds the RAM top only;
        // widen to max(RAM, MMIO) top if a high MMIO window above RAM is ever
        // added.
        if let Some(layout) = &numa_layout {
            let phys_bits = base_cpuid
                .as_slice()
                .iter()
                .find(|e| e.function == 0x8000_0008)
                .map(|e| e.eax & 0xff)
                .unwrap_or(36);
            if let Some(top) = layout.ram_top_exceeds_phys_bits(phys_bits) {
                return Err(anyhow::Error::new(
                    crate::vmm::host_topology::TopologyInsufficient {
                        reason: format!(
                            "guest RAM top {top:#x} exceeds the guest MAXPHYADDR \
                             (1<<{phys_bits}); this host's physical-address \
                             width cannot back a VM this large — RAM above the \
                             MAXPHYADDR is unreachable (a guest access faults on \
                             the MMU reserved-bits check)"
                        ),
                    },
                ));
            }
        }

        // Create vCPUs with topology-specific CPUID. KVM_CREATE_VCPU
        // allocates per-vCPU kernel memory (struct kvm_vcpu, kvm_run
        // page, posted-interrupt descriptor); EMFILE / ENOMEM here is
        // host-resource pressure, not a test fault — route through
        // the contention classifier so the macro SKIPs cleanly.
        let total = topo.total_cpus();
        // A topology wider than the host's KVM_CAP_MAX_VCPUS cannot run
        // here; surface it as a clean skip (a host-capability limit, same
        // SKIP class as overcommit) before the per-vCPU create loop, rather
        // than a mid-loop create_vcpu errno. KVM_CAP_MAX_VCPUS returns the
        // per-VM kvm->max_vcpus (arch/x86/kvm/x86.c), which defaults to
        // KVM_MAX_VCPUS (= CONFIG_KVM_MAX_NR_VCPUS, commonly 1024) — a
        // host-dependent value. Because the cap is host-dependent, an over-cap
        // topology is a skippable TopologyInsufficient — a host with a larger
        // cap could run it. This is the deliberate counterpart to aarch64,
        // where the vCPU bound is a VMM-layout constant (the GICv3 redistributor
        // window) that is host-independent (no host can run a wider topology
        // under this VMM), so an over-bound topology there is a hard error, not
        // a skip.
        let max_vcpus = kvm.get_max_vcpus();
        if total as usize > max_vcpus {
            return Err(anyhow::Error::new(
                crate::vmm::host_topology::TopologyInsufficient {
                    reason: format!(
                        "topology requires {total} vCPUs but this host's \
                     KVM_CAP_MAX_VCPUS is {max_vcpus}; cannot run a VM this wide"
                    ),
                },
            ));
        }
        // The vcpu_id passed below is apic_id(topo, cpu_id), whose sparse
        // range can exceed the vCPU count; KVM_CREATE_VCPU requires the id be
        // < KVM_CAP_MAX_VCPU_ID. Skip cleanly if the host's cap is too low,
        // same class as the max_vcpus check above.
        // `max_apic_id` is passed in from `new_inner`, the same u32 it
        // computed for the split-irqchip decision.
        let max_vcpu_id = kvm.get_max_vcpu_id();
        if (max_apic_id as usize) >= max_vcpu_id {
            return Err(anyhow::Error::new(
                crate::vmm::host_topology::TopologyInsufficient {
                    reason: format!(
                        "topology's max APIC ID {max_apic_id} (the KVM vcpu_id) is \
                         >= this host's KVM_CAP_MAX_VCPU_ID {max_vcpu_id}; cannot \
                         create a vCPU at that ID"
                    ),
                },
            ));
        }
        let mut vcpus = Vec::with_capacity(total as usize);
        for cpu_id in 0..total {
            // vcpu_id = apic_id, not cpu_id: KVM hardwires the in-kernel LAPIC
            // x2apic_id to vcpu_id (arch/x86/kvm/lapic.c kvm_apic_set_x2apic_id,
            // read-only), and an MSI/IPI dest plus the guest's read_apic_id()
            // resolve against it. The CPUID/MADT advertise the sparse apic_id,
            // so vcpu_id must equal it or sparse APIC IDs are unrouteable. The
            // vcpus Vec stays indexed by cpu_id (push order); only the KVM
            // vcpu_id changes (a no-op for dense topologies where apic==cpu_id).
            let aid = apic_id(topo, cpu_id);
            let vcpu = vm_fd.create_vcpu(aid as u64).map_err(|e| {
                crate::vmm::map_transient_to_contention(
                    e,
                    format!("create vCPU cpu_id={cpu_id} apic_id={aid}"),
                )
            })?;

            let cpuid_entries =
                generate_cpuid(base_cpuid.as_slice(), topo, cpu_id, performance_mode);
            let cpuid = kvm_bindings::CpuId::from_entries(&cpuid_entries).context("build CpuId")?;
            vcpu.set_cpuid2(&cpuid)
                .with_context(|| format!("set CPUID for vCPU {cpu_id}"))?;

            vcpus.push(vcpu);
        }

        // Check TSC stability via KVM_GET_CLOCK. An unstable TSC
        // (missing KVM_CLOCK_TSC_STABLE) means kvmclock falls back to
        // host-side timekeeping per-vCPU, adding overhead to
        // clock_gettime and degrading timer accuracy. Common in nested
        // virtualization where the L0 hypervisor does not expose
        // constant TSC to L1.
        //
        // Only checked in performance_mode: non-perf tests use binary
        // pass/fail (cpuset, starvation) where timing precision doesn't
        // affect results.
        //
        // A get→set→get roundtrip is required: use_master_clock
        // starts false and is only evaluated by
        // pvclock_update_vm_gtod_copy(). That function is called by
        // kvm_vm_ioctl_set_clock() but NOT by kvm_vm_ioctl_get_clock()
        // or vCPU creation. Without the set_clock() call, get_clock()
        // always returns flags=0 regardless of actual TSC stability.
        //
        // Flags must be cleared before set_clock(): get_clock() may
        // set KVM_CLOCK_REALTIME, and set_clock() applies a realtime
        // adjustment when that flag is present (x86.c:7209-7215),
        // double-counting elapsed time. KVM_CLOCK_TSC_STABLE and
        // KVM_CLOCK_HOST_TSC are output-only and ignored by set_clock().
        if performance_mode {
            match vm_fd.get_clock() {
                Ok(clock) => {
                    let mut set_data = clock;
                    set_data.flags = 0;
                    if let Err(e) = vm_fd.set_clock(&set_data) {
                        eprintln!(
                            "performance_mode: WARNING: KVM_SET_CLOCK failed ({e}), \
                             cannot check TSC stability"
                        );
                    } else {
                        match vm_fd.get_clock() {
                            Ok(clock2) => {
                                if clock2.flags & KVM_CLOCK_TSC_STABLE == 0 {
                                    eprintln!(
                                        "performance_mode: WARNING: TSC not stable \
                                         (KVM_CLOCK_TSC_STABLE not set), \
                                         timing measurements may have higher variance \
                                         (nested virt?)."
                                    );
                                }
                            }
                            Err(e) => {
                                eprintln!(
                                    "performance_mode: WARNING: KVM_GET_CLOCK failed ({e}), \
                                     cannot check TSC stability"
                                );
                            }
                        }
                    }
                }
                Err(e) => {
                    eprintln!(
                        "performance_mode: WARNING: KVM_GET_CLOCK failed ({e}), \
                         cannot check TSC stability"
                    );
                }
            }
        }

        Ok(vcpus)
    }
}

/// Call `KVM_GET_CLOCK` via a raw VM fd (libc::ioctl direct).
/// Companion to the safe-wrapper boot-time probe above — used by
/// the freeze coordinator (see [`crate::vmm::freeze_coord`]) for
/// the freeze rendezvous save/restore where the coordinator's
/// `freeze_and_dispatch` + `thaw_and_barrier` sibling closures
/// can't borrow `&vm.vm_fd` (vm is consumed by a downstream closure
/// in the same scope) and therefore use the raw fd (Copy) cached
/// at coord-thread spawn time.
///
/// Mirrors `kvm_ioctls::VmFd::get_clock` — same ioctl number
/// (`KVM_GET_CLOCK = KVMIO | 0x7c`), same `kvm_clock_data` payload,
/// same error mapping. The underlying ioctl path
/// (`arch/x86/kvm/x86.c kvm_vm_ioctl_get_clock` → `get_kvmclock`)
/// is a pure seqcount read on the host side with no lock
/// acquisition. The save/restore pairing keeps the guest's
/// post-resume kvm_clock view at the parked-state value rather
/// than the freeze-advanced host monotonic; the planned per-vCPU
/// `KVM_KVMCLOCK_CTRL` emit at freeze entry is complementary —
/// it sets `PVCLOCK_GUEST_STOPPED` so the guest's soft-lockup
/// watchdog (`pvclock_touch_watchdogs` in
/// `arch/x86/kernel/pvclock.c`) skips the freeze interval and
/// does not fire on long freezes.
pub(crate) fn kvm_get_clock_via_raw_fd(
    vm_fd: i32,
) -> std::io::Result<kvm_bindings::kvm_clock_data> {
    // KVMIO | 0x7c, ioctl_ior_nr! per kvm-ioctls 0.24.0
    // kvm_ioctls.rs:109. `kvm_clock_data` size is 8 (clock) +
    // 4 (flags) + 4 (pad0) + 8 (realtime) + 8 (host_tsc) + 4*4
    // (pad) = 48 bytes; `_IOC_SIZE` (0x30 = 48) is encoded into
    // the ioctl number. If kvm-bindings ever bumps the struct
    // past 48 bytes, the encoded size in our ioctl constant
    // diverges from the kernel's expectation and the syscall
    // returns EINVAL silently — guard the size at compile time.
    const _: () = assert!(std::mem::size_of::<kvm_bindings::kvm_clock_data>() == 48);
    const KVM_GET_CLOCK_IOCTL: libc::c_ulong = 0x8030_ae7c;
    let mut clock = kvm_bindings::kvm_clock_data::default();
    // SAFETY: `vm_fd` is a valid kvm_vmfd (caller is the freeze
    // coordinator, which got the fd from vm.vm_fd.as_raw_fd() at
    // closure-definition time and the fd is alive for the
    // duration of `run_vm`). `kvm_clock_data` is `#[repr(C)]`
    // POD; the kernel writes <= sizeof::<kvm_clock_data>() bytes.
    let rc = unsafe {
        libc::ioctl(
            vm_fd,
            KVM_GET_CLOCK_IOCTL,
            &mut clock as *mut kvm_bindings::kvm_clock_data,
        )
    };
    if rc < 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(clock)
    }
}

/// Call `KVM_SET_CLOCK` via a raw VM fd (libc::ioctl direct).
/// Sibling of [`kvm_get_clock_via_raw_fd`] for the restore-side of
/// the freeze rendezvous kvm_clock save/restore. Mirrors
/// `kvm_ioctls::VmFd::set_clock`. The underlying ioctl path
/// (`arch/x86/kvm/x86.c kvm_vm_ioctl_set_clock`) takes the
/// `pvclock_sc` seqcount write side, recomputes
/// `master_kernel_ns`, sets `ka->kvmclock_offset = data.clock -
/// now_raw_ns`, then queues `KVM_REQ_CLOCK_UPDATE` on every vCPU
/// (processed at the next KVM_RUN entry per-vCPU).
///
/// Caller MUST clear `flags` to 0 before calling (per the
/// boot-time precedent above) — leaving `KVM_CLOCK_REALTIME` in
/// flags causes the kernel to apply a realtime adjustment that
/// double-counts elapsed time.
pub(crate) fn kvm_set_clock_via_raw_fd(
    vm_fd: i32,
    clock: &kvm_bindings::kvm_clock_data,
) -> std::io::Result<()> {
    // KVMIO | 0x7b, ioctl_iow_nr! per kvm-ioctls 0.24.0
    // kvm_ioctls.rs:106.
    const KVM_SET_CLOCK_IOCTL: libc::c_ulong = 0x4030_ae7b;
    // SAFETY: `vm_fd` is a valid kvm_vmfd (see SAFETY note on
    // [`kvm_get_clock_via_raw_fd`]). The kernel reads exactly
    // sizeof::<kvm_clock_data>() bytes from the pointer; the
    // payload is `#[repr(C)]` POD.
    let rc = unsafe {
        libc::ioctl(
            vm_fd,
            KVM_SET_CLOCK_IOCTL,
            clock as *const kvm_bindings::kvm_clock_data,
        )
    };
    if rc < 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}

/// Call `KVM_SET_GSI_ROUTING` via a raw VM fd (libc::ioctl direct).
/// Sibling of [`kvm_set_clock_via_raw_fd`] for the userspace-IOAPIC
/// (split-irqchip / >255-vCPU) path: an AP run loop holds only a cached
/// Copy raw vm fd, not `&vm.vm_fd`, and reprograms the device MSI routes
/// when the guest writes the IOAPIC redirection table. Mirrors
/// `kvm_ioctls::VmFd::set_gsi_routing` — same ioctl, same `kvm_irq_routing`
/// FAM payload. `KVM_SET_GSI_ROUTING` is a whole-table replace
/// (virt/kvm/irqchip.c `kvm_set_irq_routing`, under `kvm->irq_lock` plus an
/// SRCU grace period), so the caller passes the COMPLETE route set.
pub(crate) fn kvm_set_gsi_routing_via_raw_fd(
    vm_fd: i32,
    routing: &KvmIrqRouting,
) -> std::io::Result<()> {
    // ioctl_iow_nr!(KVM_SET_GSI_ROUTING, KVMIO=0xAE, 0x6a, kvm_irq_routing)
    // = _IOW(0xAE, 0x6a, size_of::<kvm_irq_routing>()) = 0x4008_AE6A. The
    // encoded size is the FAM HEADER (nr:u32 + flags:u32 = 8); the kernel
    // reads `nr` entries past the pointer.
    const _: () = assert!(std::mem::size_of::<kvm_irq_routing>() == 8);
    const KVM_SET_GSI_ROUTING_IOCTL: libc::c_ulong = 0x4008_AE6A;
    // SAFETY: `vm_fd` is a live kvm vm fd (cached from vm.vm_fd.as_raw_fd(),
    // valid for the run loop's lifetime). `as_fam_struct_ref()` points at a
    // `kvm_irq_routing` whose `nr` matches the entries the kernel reads.
    let rc = unsafe {
        libc::ioctl(
            vm_fd,
            KVM_SET_GSI_ROUTING_IOCTL,
            routing.as_fam_struct_ref() as *const kvm_irq_routing,
        )
    };
    if rc < 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}

/// Build a `KVM_SET_GSI_ROUTING` table from the IOAPIC's `(gsi, MsiRoute)`
/// set. Each entry is a `KVM_IRQ_ROUTING_MSI` route carrying the
/// extended-destination MSI the IOAPIC translated the guest's RTE into.
fn build_device_msi_routing(routes: &[(u32, MsiRoute)]) -> Result<KvmIrqRouting> {
    let mut routing = KvmIrqRouting::new(routes.len()).map_err(|e| {
        anyhow::anyhow!(
            "allocate kvm_irq_routing for {} routes: {e:?}",
            routes.len()
        )
    })?;
    let slice = routing.as_mut_slice();
    for (i, (gsi, msi)) in routes.iter().enumerate() {
        slice[i] = kvm_irq_routing_entry {
            gsi: *gsi,
            type_: KVM_IRQ_ROUTING_MSI,
            flags: 0,
            pad: 0,
            u: kvm_irq_routing_entry__bindgen_ty_1 {
                msi: kvm_irq_routing_msi {
                    address_lo: msi.address_lo,
                    address_hi: msi.address_hi,
                    data: msi.data,
                    __bindgen_anon_1: kvm_irq_routing_msi__bindgen_ty_1 { pad: 0 },
                },
            },
        };
    }
    Ok(routing)
}

/// Build a `KVM_SET_GSI_ROUTING` table for the FULL-irqchip path: the explicit
/// x86 default routes (IOAPIC pins for GSI 0..24, plus 8259 PIC master/slave for
/// GSI 0..16) FOLLOWED BY the device MSI routes. Required because
/// `KVM_SET_GSI_ROUTING` is a whole-table replace — installing only the MSI
/// routes would wipe KVM's implicit default IOAPIC/PIC routes and break INTx
/// delivery (console / blk / serial). Replicates the kernel's own
/// `default_routing` (arch/x86/kvm/irq.c: `ROUTING_ENTRY2` = IOAPIC + PIC for
/// GSI 0..16, `ROUTING_ENTRY1` = IOAPIC-only for GSI 16..24; PIC pin = gsi % 8,
/// master for gsi < 8 else slave) so the explicit table is route-identical to
/// the implicit default, then appends one `KVM_IRQ_ROUTING_MSI` per device
/// vector.
fn build_full_irqchip_routing(msi_routes: &[(u32, MsiRoute)]) -> Result<KvmIrqRouting> {
    const N_IOAPIC_PINS: u32 = 24;
    const N_PIC_PINS: u32 = 16;
    let total = N_IOAPIC_PINS as usize + N_PIC_PINS as usize + msi_routes.len();
    let mut routing = KvmIrqRouting::new(total).map_err(|e| {
        anyhow::anyhow!("allocate kvm_irq_routing for {total} full-irqchip routes: {e:?}")
    })?;
    let irqchip = |gsi: u32, chip: u32, pin: u32| kvm_irq_routing_entry {
        gsi,
        type_: KVM_IRQ_ROUTING_IRQCHIP,
        flags: 0,
        pad: 0,
        u: kvm_irq_routing_entry__bindgen_ty_1 {
            irqchip: kvm_irq_routing_irqchip { irqchip: chip, pin },
        },
    };
    let slice = routing.as_mut_slice();
    let mut i = 0;
    // IOAPIC pins 0..24 (gsi == pin) — the IOAPIC half of every default entry.
    for gsi in 0..N_IOAPIC_PINS {
        slice[i] = irqchip(gsi, KVM_IRQCHIP_IOAPIC, gsi);
        i += 1;
    }
    // 8259 PIC for GSI 0..16: master (0..8) / slave (8..16), pin = gsi % 8.
    for gsi in 0..N_PIC_PINS {
        let chip = if gsi < 8 {
            KVM_IRQCHIP_PIC_MASTER
        } else {
            KVM_IRQCHIP_PIC_SLAVE
        };
        slice[i] = irqchip(gsi, chip, gsi % 8);
        i += 1;
    }
    // Device MSI routes (same entry shape as `build_device_msi_routing`).
    for (gsi, msi) in msi_routes {
        slice[i] = kvm_irq_routing_entry {
            gsi: *gsi,
            type_: KVM_IRQ_ROUTING_MSI,
            flags: 0,
            pad: 0,
            u: kvm_irq_routing_entry__bindgen_ty_1 {
                msi: kvm_irq_routing_msi {
                    address_lo: msi.address_lo,
                    address_hi: msi.address_hi,
                    data: msi.data,
                    __bindgen_anon_1: kvm_irq_routing_msi__bindgen_ty_1 { pad: 0 },
                },
            },
        };
        i += 1;
    }
    Ok(routing)
}

/// Full-irqchip shared GSI route owner. Holds the device MSI routes and installs
/// the COMPLETE routing table — the explicit IOAPIC/PIC defaults from
/// [`build_full_irqchip_routing`] PLUS the MSI routes — via `KVM_SET_GSI_ROUTING`
/// on every change. Because that ioctl is a whole-table replace, ONE owner holds
/// every MSI-capable device's routes on the full-irqchip path; the split-irqchip
/// path routes through [`IoapicHandle`] instead. Constructed `Some` only when
/// `!split_irqchip` (the inverse of `IoapicHandle`). [`Self::install_defaults`]
/// runs once at VM init to replace KVM's implicit default routing with a
/// route-identical EXPLICIT table, so INTx keeps working once MSI routes are
/// added. Cloned (`Arc`) into each MSI-X-capable PCI function, which calls
/// [`Self::set_route`] on a vector mask/unmask edge.
pub(crate) struct FullIrqchipRouteOwner {
    vm_fd_raw: i32,
    /// Device MSI routes keyed by GSI. The 40 default IOAPIC/PIC routes are
    /// constant (re-emitted by the builder each install); only these change.
    routes: PiMutex<Vec<(u32, MsiRoute)>>,
    /// Count of failed `KVM_SET_GSI_ROUTING` installs (a failure leaves a device
    /// IRQ unrouted — it will not deliver). Read at teardown for diagnostics.
    routing_failures: std::sync::atomic::AtomicU64,
    /// The route set most recently installed; skips the redundant
    /// (SRCU-grace-period) ioctl when an install would be byte-identical. Mirrors
    /// [`IoapicHandle`]'s `last_installed` dedup. A separate lock from `routes`
    /// so the route snapshot is taken + the `routes` lock released BEFORE the
    /// install ioctl (keeps the slow ioctl off any other vCPU's `set_route`).
    last_installed: PiMutex<Option<Vec<(u32, MsiRoute)>>>,
}

impl FullIrqchipRouteOwner {
    pub(crate) fn new(vm_fd_raw: i32) -> Self {
        FullIrqchipRouteOwner {
            vm_fd_raw,
            routes: PiMutex::new(Vec::new()),
            routing_failures: std::sync::atomic::AtomicU64::new(0),
            last_installed: PiMutex::new(None),
        }
    }

    /// Install the complete table (explicit defaults + the current MSI routes).
    /// Called once at VM init (no MSI routes yet → defaults only) so the explicit
    /// table replaces KVM's implicit default before any device adds a route.
    pub(crate) fn install_defaults(&self) -> Result<()> {
        let snapshot = self.routes.lock().clone();
        self.install_snapshot(snapshot, |r| {
            kvm_set_gsi_routing_via_raw_fd(self.vm_fd_raw, r)
        })
    }

    /// Set (`Some`) or clear (`None`) the device MSI route for `gsi`, then
    /// reinstall the complete table. The route snapshot is taken under the
    /// `routes` lock, which is released before the install ioctl.
    pub(crate) fn set_route(&self, gsi: u32, route: Option<MsiRoute>) -> Result<()> {
        let snapshot = {
            let mut routes = self.routes.lock();
            routes.retain(|(g, _)| *g != gsi);
            if let Some(m) = route {
                routes.push((gsi, m));
            }
            routes.clone()
        };
        self.install_snapshot(snapshot, |r| {
            kvm_set_gsi_routing_via_raw_fd(self.vm_fd_raw, r)
        })
    }

    /// Build the full table from `snapshot` and install it via `install` (unless
    /// byte-identical to the last install). The seam exists only so a host-side
    /// test injects a counting/failing installer; production passes the real
    /// `KVM_SET_GSI_ROUTING`. `install` runs at most once → `FnOnce`.
    fn install_snapshot(
        &self,
        snapshot: Vec<(u32, MsiRoute)>,
        install: impl FnOnce(&KvmIrqRouting) -> std::io::Result<()>,
    ) -> Result<()> {
        let mut last = self.last_installed.lock();
        if last.as_deref() == Some(snapshot.as_slice()) {
            return Ok(());
        }
        let routing = build_full_irqchip_routing(&snapshot)?;
        if let Err(e) = install(&routing) {
            self.routing_failures
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Err(anyhow::anyhow!("KVM_SET_GSI_ROUTING (full-irqchip): {e}"));
        }
        *last = Some(snapshot);
        Ok(())
    }

    /// Failed-install count, for teardown diagnostics.
    pub(crate) fn routing_failures(&self) -> u64 {
        self.routing_failures
            .load(std::sync::atomic::Ordering::Relaxed)
    }
}

/// Translate a guest-composed MSI message from the `virt_ext_dest_id` encoding
/// into the format KVM decodes, so MSI-X delivers to APIC IDs > 255 WITHOUT
/// interrupt remapping. A guest with `KVM_FEATURE_MSI_EXT_DEST_ID` advertised
/// (we set it for wide-SMP — see `host_topology`) and no IOMMU composes a >255
/// destination by packing `dest[14:8]` into `address_lo` bits `[11:5]` (the
/// "virt-destid" field; `address_hi` stays 0) — Linux `__irq_msi_compose_msg`
/// (arch/x86/kernel/apic/apic.c). But KVM, under `KVM_X2APIC_API_USE_32BIT_IDS`
/// (which we enable), reads the extended destination from `address_hi[31:8]` and
/// ignores those `address_lo` bits — so a verbatim guest message decodes to
/// `destid_0_7` only (e.g. 256 & 0xff = 0 → APIC 0), silently mis-delivering.
/// Move the virt-destid bits into `address_hi` (`[11:5] << 3` lands them at
/// `address_hi[14:8]`, matching `(dest>>8)<<8` — the exact format
/// [`Ioapic::redtbl_to_msi`] already emits for INTx and KVM decodes).
///
/// Guards (mirrors cloud-hypervisor `translate_msi_ext_dest_id` / qemu
/// `kvm_swizzle_msi_ext_dest_id`): only swizzle when `address_hi == 0` (an
/// already-translated message, or a host-composed IOAPIC route whose
/// `address_hi` redtbl_to_msi set for dest>255, is left untouched — no
/// double-swizzle) and the interrupt-format bit (`address_lo` bit 4 —
/// `x86_msi_addr_lo.reserved_1`, set as `dmar_format` only by the IR/remappable
/// path; bit 3, not this, is `redirect_hint`) is clear (never rewrite a
/// remappable-format message; the kernel sets `virt_destid` only when it is
/// clear). A dest ≤ 255 has no virt-destid bits set, so the swizzle is a no-op
/// there.
fn translate_msi_ext_dest_id(address_lo: u32, address_hi: u32) -> (u32, u32) {
    /// `address_lo` bit 4 — the interrupt-format / `dmar_format` bit (kernel
    /// `x86_msi_addr_lo.reserved_1` in the compatibility format, set only by the
    /// IR/remappable path; not `redirect_hint`, which is bit 3). Matches qemu's
    /// `MSI_ADDR_DEST_IDX_SHIFT` ext-dest gate.
    const MSI_ADDR_RF_BIT: u32 = 0x10;
    /// `address_lo` bits `[11:5]` — the guest's virt-destid (`dest[14:8]`) field.
    const MSI_ADDR_VIRT_DESTID: u32 = 0xfe0;
    if address_hi == 0 && address_lo & MSI_ADDR_RF_BIT == 0 {
        let ext = address_lo & MSI_ADDR_VIRT_DESTID;
        (address_lo & !MSI_ADDR_VIRT_DESTID, address_hi | (ext << 3))
    } else {
        (address_lo, address_hi)
    }
}

/// Build an [`MsiRoute`] from a guest-composed MSI-X message tuple, applying the
/// `virt_ext_dest_id` → KVM swizzle ([`translate_msi_ext_dest_id`]) so the
/// route's destination survives KVM decode for APIC IDs > 255. Shared by both
/// [`MsixRouteSink`](crate::vmm::virtio_msix::MsixRouteSink) impls (the two
/// per-irqchip-mode route owners) so the guest-message → KVM-route conversion is
/// identical on both paths.
fn msi_route_from_guest_msg(address_lo: u32, address_hi: u32, data: u32) -> MsiRoute {
    let (address_lo, address_hi) = translate_msi_ext_dest_id(address_lo, address_hi);
    MsiRoute {
        address_lo,
        address_hi,
        data,
    }
}

/// Bridge the full-irqchip route owner to the transport-neutral
/// [`crate::vmm::virtio_msix::MsixRouteSink`] the virtio-net PCI facade calls on a
/// vector mask/unmask edge: map the guest's MSI message dwords into an
/// [`MsiRoute`] (applying the ext-dest swizzle, [`msi_route_from_guest_msg`]) and
/// (un)install it via the inherent [`Self::set_route`]. Errors are counted in
/// `routing_failures` (read at teardown), not propagated — a config-space MMIO
/// write has no error channel back to the guest.
impl crate::vmm::virtio_msix::MsixRouteSink for FullIrqchipRouteOwner {
    fn set_route(&self, gsi: u32, msg: Option<(u32, u32, u32)>) {
        let route = msg.map(|(lo, hi, data)| msi_route_from_guest_msg(lo, hi, data));
        let _ = FullIrqchipRouteOwner::set_route(self, gsi, route);
    }
}

/// Owns the userspace IOAPIC device plus the cached raw VM fd needed to
/// reprogram KVM's MSI routing table. Cloned (via `Arc`) into each AP run
/// loop on the split-irqchip path; `None` on the in-kernel-irqchip path
/// (<=254 max APIC ID), where the kernel IOAPIC delivers device IRQs directly.
///
/// On split-irqchip this is the SINGLE owner of the whole KVM GSI routing
/// table (`KVM_SET_GSI_ROUTING` is a whole-table replace), holding BOTH halves:
/// the userspace IOAPIC's RTE→MSI translations ([`Ioapic::gsi_routes`], GSIs
/// `0..NUM_IOAPIC_PINS`) AND the device MSI-X routes (`msix_routes`, GSIs
/// `>= NUM_IOAPIC_PINS`) installed via [`crate::vmm::virtio_msix::MsixRouteSink`]
/// on a vector mask/unmask edge. Both halves are `KVM_IRQ_ROUTING_MSI` entries
/// built by [`build_device_msi_routing`] — split-irqchip has no in-kernel
/// IOAPIC/PIC, so (unlike [`FullIrqchipRouteOwner`]) it never emits the explicit
/// irqchip defaults. Every install rebuilds the COMBINED table from both halves
/// (see [`Self::install_combined`]) so a change to one never drops the other
/// from the whole-table replace.
pub(crate) struct IoapicHandle {
    ioapic: Arc<PiMutex<Ioapic>>,
    vm_fd_raw: i32,
    /// Count of failed `KVM_SET_GSI_ROUTING` installs. A failure leaves a
    /// guest-programmed device IRQ unrouted — it will not deliver and the
    /// device hangs on first use. Bumped on every failed install (an IOAPIC RTE
    /// write OR an MSI-X route edge), read at teardown (`routing_failures`) so a
    /// hung-device test reports the count instead of an opaque timeout.
    routing_failures: std::sync::atomic::AtomicU64,
    /// Device MSI-X routes keyed by GSI (`>= NUM_IOAPIC_PINS`), the second half
    /// of the combined table. Maintained by [`Self::set_route_with`] (retain +
    /// push, mirroring [`FullIrqchipRouteOwner`]'s `routes`) on each vector
    /// mask/unmask edge; appended to the IOAPIC half in [`Self::install_combined`].
    /// A separate lock from `ioapic` so the two halves are snapshotted
    /// independently (never both held) — no lock-order inversion between the
    /// IOAPIC-write and MSI-X-edge install paths.
    msix_routes: PiMutex<Vec<(u32, MsiRoute)>>,
    /// The COMBINED route set most recently installed via `KVM_SET_GSI_ROUTING`
    /// (IOAPIC translations ++ device MSI-X routes). An install skips the ioctl
    /// (which waits an SRCU grace period) when the freshly-rebuilt combined set
    /// is byte-identical — the guest programs each 64-bit IOAPIC RTE as two
    /// 32-bit MMIO writes, and the high-word write of a still-masked entry
    /// yields the same `is_masked`-filtered IOAPIC half as before it, so roughly
    /// half the per-RTE installs are redundant. Guarded by its own mutex, not the
    /// `ioapic` lock: the compare + ioctl + cache update run as one critical
    /// section so the cache can never diverge from KVM's actual routing table
    /// under concurrent IOAPIC programming, while the `ioapic` lock is
    /// released first so a slow install never stalls another vCPU's IOAPIC
    /// MMIO access.
    ///
    /// This dedup is intentionally stronger than the reference userspace
    /// IOAPICs: qemu, cloud-hypervisor, and libkrun all re-issue
    /// `KVM_SET_GSI_ROUTING` on every redtbl write (qemu's change-counting
    /// dedup applies only to its PCI-MSI path, not the IOAPIC). Skipping an
    /// unchanged table is safe because the ioctl is a whole-table replace
    /// KVM applies idempotently -- an identical re-install only burns an
    /// SRCU grace period (virt/kvm/irqchip.c `kvm_set_irq_routing` has no
    /// unchanged-table early-out).
    last_installed: PiMutex<Option<Vec<(u32, MsiRoute)>>>,
}

impl IoapicHandle {
    pub(crate) fn new(ioapic: Arc<PiMutex<Ioapic>>, vm_fd_raw: i32) -> Self {
        IoapicHandle {
            ioapic,
            vm_fd_raw,
            routing_failures: std::sync::atomic::AtomicU64::new(0),
            msix_routes: PiMutex::new(Vec::new()),
            last_installed: PiMutex::new(None),
        }
    }

    /// Service a guest MMIO read of the IOAPIC window.
    pub(crate) fn mmio_read(&self, offset: u64, data: &mut [u8]) {
        self.ioapic.lock().mmio_read(offset, data);
    }

    /// Service a guest MMIO write of the IOAPIC window. If the write changed
    /// a redirection entry, rebuild the COMBINED MSI routing table — the IOAPIC
    /// translations PLUS the current device MSI-X routes (see
    /// [`Self::install_combined`]) — and install it, unless it is byte-identical
    /// to the last install, in which case the (SRCU-grace-period) ioctl is
    /// skipped (see the `last_installed` cache).
    ///
    /// The IOAPIC-half snapshot is taken under the `ioapic` lock atomically with
    /// the change-detection (so it matches the write that triggered it); the
    /// `ioapic` lock is then released, the MSI-X half appended under its own
    /// lock, and the compare + ioctl + cache update run under the separate
    /// `last_installed` lock. So a slow install never stalls another vCPU's
    /// IOAPIC MMIO access (the `ioapic` lock is free during the ioctl), and
    /// the cache stays consistent with KVM's table — installs serialize on
    /// `last_installed` (one whole-table replace at a time) and the cache is
    /// updated in the same critical section as each install.
    ///
    /// Cross-vCPU atomicity is traded for that latency, diverging from every
    /// reference userspace IOAPIC: qemu (one BQL across redtbl-write →
    /// route-commit), cloud-hypervisor (one device mutex across the install),
    /// and libkrun (device mutex held across a synchronous worker-hop install)
    /// are all single-lock atomic and accept the peer-vCPU IOAPIC-MMIO stall
    /// for the SRCU-grace-period ioctl; we release the `ioapic` lock first to
    /// keep that ioctl off the vCPU blocking budget. The resulting window —
    /// two vCPUs racing the IOAPIC, an older snapshot installing after a newer
    /// one and leaving KVM's table briefly stale — is unreachable for a
    /// spec-compliant guest: Linux serializes every IOAPIC RTE program under
    /// one global `ioapic_lock` (the `ioapic_write_entry`,
    /// `ioapic_set_affinity`, and `eoi_ioapic_pin` paths in
    /// arch/x86/kernel/apic/io_apic.c), so only one vCPU programs the IOAPIC
    /// at a time. A guest that races its own IOAPIC can transiently install a
    /// stale-but-valid whole-table replace built only from its own programmed
    /// routes; it self-corrects on the next RTE write (which re-snapshots the
    /// current register file) and can only mis-route its own device IRQs to
    /// its own APICs — no host memory is touched and no unprogrammed route is
    /// installable.
    ///
    /// Delegates to [`Self::mmio_write_with`] passing the real
    /// `KVM_SET_GSI_ROUTING` installer; the seam exists only so a host-side
    /// test can inject a counting/failing installer.
    pub(crate) fn mmio_write(&self, offset: u64, data: &[u8]) -> Result<()> {
        let fd = self.vm_fd_raw;
        self.mmio_write_with(offset, data, move |routing| {
            kvm_set_gsi_routing_via_raw_fd(fd, routing)
        })
    }

    /// [`Self::mmio_write`] with the routing install injected as `install`,
    /// so a host-side test drives the combined-table dedup + cache-on-success
    /// logic with a counting/failing closure instead of a live KVM fd. The
    /// production caller passes the real `KVM_SET_GSI_ROUTING` installer.
    /// `install` runs at most once per call — only when the write changed a
    /// route AND the resulting combined set differs from `last_installed` —
    /// hence `FnOnce`. (No reference VMM exposes such a seam or unit-tests this
    /// path; they re-install unconditionally — see the `last_installed`
    /// divergence note.)
    fn mmio_write_with(
        &self,
        offset: u64,
        data: &[u8],
        install: impl FnOnce(&KvmIrqRouting) -> std::io::Result<()>,
    ) -> Result<()> {
        // Snapshot the IOAPIC half atomically with the change-detection (one
        // `ioapic` critical section), then release the lock before the install.
        let ioapic_routes = {
            let mut io = self.ioapic.lock();
            if io.mmio_write(offset, data) {
                Some(io.gsi_routes())
            } else {
                None
            }
        };
        match ioapic_routes {
            Some(routes) => self.install_combined(routes, install),
            None => Ok(()),
        }
    }

    /// Append the current device MSI-X routes to the given IOAPIC-half snapshot,
    /// rebuild the COMBINED `KVM_IRQ_ROUTING_MSI` table
    /// ([`build_device_msi_routing`] — NO in-kernel-irqchip defaults; the
    /// split-irqchip path has no in-kernel IOAPIC/PIC, unlike
    /// [`FullIrqchipRouteOwner`]), and install it via `install` unless
    /// byte-identical to the last successful install. The caller passes the
    /// IOAPIC half already snapshotted (atomically with its own change-detection
    /// on the MMIO path, freshly on the MSI-X path); the MSI-X half is read here
    /// under its own lock — never while holding `ioapic`, so the two halves are
    /// never both locked and the MMIO-write and MSI-X-edge paths cannot invert
    /// lock order. The compare + ioctl + cache run under `last_installed` alone;
    /// cache-on-success-only so a failed install never makes a later identical
    /// attempt skip a needed retry. Either trigger rebuilds the whole combined
    /// table, so a change to one half never drops the other from the
    /// whole-table replace. `install` runs at most once → `FnOnce`.
    fn install_combined(
        &self,
        ioapic_routes: Vec<(u32, MsiRoute)>,
        install: impl FnOnce(&KvmIrqRouting) -> std::io::Result<()>,
    ) -> Result<()> {
        let combined = {
            let mut c = ioapic_routes;
            c.extend(self.msix_routes.lock().iter().cloned());
            c
        };
        let mut last = self.last_installed.lock();
        if last.as_deref() == Some(combined.as_slice()) {
            // Unchanged from the last successful install — skip the
            // whole-table replace and its SRCU grace period.
            return Ok(());
        }
        let routing = build_device_msi_routing(&combined)?;
        if let Err(e) = install(&routing) {
            self.routing_failures
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Err(anyhow::anyhow!("KVM_SET_GSI_ROUTING: {e}"));
        }
        // Cache only after a successful install so a failed ioctl never
        // makes a later identical attempt skip a needed retry.
        *last = Some(combined);
        Ok(())
    }

    /// Set (`Some`) or clear (`None`) the device MSI-X route for `gsi` (the
    /// second half of the combined table), then reinstall the combined table via
    /// `install`. The `msix_routes` mutation runs under its own lock (released
    /// before the install); [`Self::install_combined`] then snapshots the IOAPIC
    /// half separately — neither lock is held across the other or across the
    /// ioctl. The seam exists only so a host-side test injects a counting/failing
    /// installer; production reaches this via the
    /// [`crate::vmm::virtio_msix::MsixRouteSink`] impl with the real
    /// `KVM_SET_GSI_ROUTING` installer. `install` runs at most once → `FnOnce`.
    fn set_route_with(
        &self,
        gsi: u32,
        route: Option<MsiRoute>,
        install: impl FnOnce(&KvmIrqRouting) -> std::io::Result<()>,
    ) -> Result<()> {
        {
            let mut routes = self.msix_routes.lock();
            routes.retain(|(g, _)| *g != gsi);
            if let Some(m) = route {
                routes.push((gsi, m));
            }
        }
        let ioapic_routes = self.ioapic.lock().gsi_routes();
        self.install_combined(ioapic_routes, install)
    }

    /// Service a `KVM_EXIT_IOAPIC_EOI` for `vector` (clears remote-IRR on a
    /// matching level entry; a no-op for the edge device pins of v0).
    pub(crate) fn eoi(&self, vector: u8) {
        // v0 is edge-only, so end_of_interrupt's returned pending-pins Vec is
        // always empty (edge entries never set remote-IRR, and this IOAPIC is a
        // register-file + MSI translator that never *services* a pin, so
        // remote-IRR is never set at all). The Vec is intentionally dropped;
        // the debug_assert is a tripwire so that adding a level-triggered
        // device WITHOUT completing level re-injection fails loudly in tests
        // instead of silently dropping the re-assert and wedging the line.
        let pending = self.ioapic.lock().end_of_interrupt(vector);
        debug_assert!(
            pending.is_empty(),
            "v0 IOAPIC is edge-only but EOI returned {} pin(s) needing level \
             re-injection — a level-triggered device was added without \
             completing level support (re-injection is dropped here)",
            pending.len()
        );
    }

    /// Number of failed `KVM_SET_GSI_ROUTING` installs since VM start. Read at
    /// teardown to surface routing failures into the result (a nonzero count
    /// explains a device that never delivered IRQs).
    pub(crate) fn routing_failures(&self) -> u64 {
        self.routing_failures
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// If `addr` lies in the IOAPIC MMIO window, the offset within it;
    /// otherwise `None`. Lets the run-loop dispatcher route an MMIO exit
    /// without importing the device's base/size constants.
    pub(crate) fn in_range(&self, addr: u64) -> Option<u64> {
        (IOAPIC_BASE..IOAPIC_BASE + IOAPIC_SIZE)
            .contains(&addr)
            .then(|| addr - IOAPIC_BASE)
    }
}

/// Bridge the split-irqchip route owner to the transport-neutral
/// [`crate::vmm::virtio_msix::MsixRouteSink`] the virtio-net PCI facade calls on a
/// vector mask/unmask edge: map the guest's MSI message dwords into an
/// [`MsiRoute`] and (un)install it via the inherent [`Self::set_route_with`],
/// which rebuilds the COMBINED IOAPIC + MSI-X table. Errors are counted in
/// `routing_failures` (read at teardown), not propagated — a config-space MMIO
/// write has no error channel back to the guest. Mirrors the identical bridge on
/// [`FullIrqchipRouteOwner`]; the two are the per-irqchip-mode route owners
/// (split → `IoapicHandle`, full → `FullIrqchipRouteOwner`).
impl crate::vmm::virtio_msix::MsixRouteSink for IoapicHandle {
    fn set_route(&self, gsi: u32, msg: Option<(u32, u32, u32)>) {
        let route = msg.map(|(lo, hi, data)| msi_route_from_guest_msg(lo, hi, data));
        let fd = self.vm_fd_raw;
        let _ = self.set_route_with(gsi, route, move |routing| {
            kvm_set_gsi_routing_via_raw_fd(fd, routing)
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The per-NIC GSI/slot allocators are the single source of truth the
    /// whole multi-NIC stack routes through (the install loop in setup, the
    /// DSDT `_PRT` routes in acpi). Pin their invariants directly so a
    /// regression — dropping the SCI skip, an off-by-one past the IOAPIC line
    /// budget, or changing the base — fails here at compile-test time rather
    /// than silently mis-routing INTx in a booted guest.
    #[test]
    fn virtio_net_gsi_slot_allocation_invariants() {
        let gsis: Vec<u32> = (0..MAX_VIRTIO_NICS).map(virtio_net_gsi).collect();

        // (a) every GSI within the IOAPIC line budget.
        for (i, &g) in gsis.iter().enumerate() {
            assert!(
                u64::from(g) < NUM_IOAPIC_PINS,
                "virtio_net_gsi({i}) = {g} exceeds the IOAPIC budget ({NUM_IOAPIC_PINS})",
            );
        }

        // (b) disjoint from every reserved line: PIT=2, COM2=3, COM1=4,
        //     virtio-MMIO console=5, virtio-blk=6, ACPI SCI=9.
        let reserved = [
            2u32,
            3,
            4,
            VIRTIO_CONSOLE_IRQ,
            VIRTIO_BLK_IRQ,
            ACPI_SCI_IRQ as u32,
        ];
        for &g in &gsis {
            assert!(
                !reserved.contains(&g),
                "virtio_net_gsi produced {g}, colliding with a reserved line {reserved:?}",
            );
        }

        // (c) all distinct and (d) the exact set {7,8,10..=23} = (7..24) minus
        //     the skipped SCI line.
        let set: std::collections::BTreeSet<u32> = gsis.iter().copied().collect();
        assert_eq!(set.len(), gsis.len(), "GSIs must be distinct: {gsis:?}");
        let mut want: std::collections::BTreeSet<u32> = (7..NUM_IOAPIC_PINS as u32).collect();
        want.remove(&(ACPI_SCI_IRQ as u32));
        assert_eq!(set, want, "GSI set must be (7..24) minus the SCI line");

        // (e) slots 0..MAX map to 1..=MAX (slot 0 reserved for the host bridge).
        let slots: Vec<usize> = (0..MAX_VIRTIO_NICS).map(virtio_net_pci_slot).collect();
        let want_slots: Vec<usize> = (1..=MAX_VIRTIO_NICS).collect();
        assert_eq!(
            slots, want_slots,
            "slots must be 1..=MAX_VIRTIO_NICS (slot 0 = host bridge)",
        );
    }

    /// MSI-X vector GSIs ([`virtio_net_msix_gsi`]) live ABOVE the IOAPIC pin
    /// range (MSI routes are bounded only by KVM_MAX_IRQ_ROUTES, not the 24-pin
    /// IOAPIC budget), are disjoint from every INTx GSI + reserved line, are
    /// globally unique across (NIC, vector), and fit the KVM routing budget. A
    /// regression — overlapping the IOAPIC range, colliding with an INTx GSI, or
    /// a non-unique mapping — would silently misroute an MSI-X vector.
    #[test]
    fn virtio_net_msix_gsi_invariants() {
        let intx: std::collections::BTreeSet<u32> =
            (0..MAX_VIRTIO_NICS).map(virtio_net_gsi).collect();
        let reserved = [
            2u32,
            3,
            4,
            VIRTIO_CONSOLE_IRQ,
            VIRTIO_BLK_IRQ,
            ACPI_SCI_IRQ as u32,
        ];

        // Walk the FULL per-NIC vector budget (not just a single-pair count): the
        // disjointness/uniqueness/budget invariants must hold for every vector a
        // NIC could ever advertise.
        let mut seen = std::collections::BTreeSet::new();
        for nic in 0..MAX_VIRTIO_NICS {
            for v in 0..MSIX_VECTORS_PER_NIC {
                let g = virtio_net_msix_gsi(nic, v);
                // (a) above the IOAPIC pin range.
                assert!(
                    u64::from(g) >= NUM_IOAPIC_PINS,
                    "msix_gsi(nic={nic}, v={v}) = {g} is inside the IOAPIC range"
                );
                // (b) disjoint from every INTx GSI and reserved line.
                assert!(!intx.contains(&g), "msix_gsi {g} collides with an INTx GSI");
                assert!(
                    !reserved.contains(&g),
                    "msix_gsi {g} collides with a reserved line"
                );
                // (c) globally unique across (NIC, vector).
                assert!(
                    seen.insert(g),
                    "msix_gsi {g} duplicated at (nic={nic}, v={v})"
                );
                // (d) within the KVM routing budget (every routed GSI < the cap).
                assert!(
                    (g as usize) < KVM_MAX_IRQ_ROUTES,
                    "msix_gsi {g} exceeds KVM_MAX_IRQ_ROUTES"
                );
            }
        }
        // Contiguous block [NUM_IOAPIC_PINS, NUM_IOAPIC_PINS + N*stride).
        assert_eq!(*seen.iter().next().unwrap(), NUM_IOAPIC_PINS as u32);
        assert_eq!(
            *seen.iter().next_back().unwrap(),
            NUM_IOAPIC_PINS as u32 + (MAX_VIRTIO_NICS * MSIX_VECTORS_PER_NIC) as u32 - 1
        );
        // The whole budget — IOAPIC pins plus every NIC's full vector range —
        // fits below the KVM routing-table cap, so no config overflows it.
        assert!(
            NUM_IOAPIC_PINS as usize + MAX_VIRTIO_NICS * MSIX_VECTORS_PER_NIC <= KVM_MAX_IRQ_ROUTES,
            "per-NIC MSI-X GSI budget overflows the KVM routing table"
        );
    }

    /// `IoapicHandle` dedups a redundant GSI-routing install (skips the
    /// ioctl when the route set is byte-identical to the last successful
    /// install) AND caches only on success (a failed install must not poison
    /// a later identical retry). Drives the real `mmio_write_with` dedup path
    /// with an injected counting/failing installer — no live KVM fd. No
    /// reference VMM unit-tests this path (they re-install unconditionally).
    #[test]
    fn ioapic_handle_dedups_install_and_caches_on_success_only() {
        use crate::vmm::x86_64::ioapic::{IOREGSEL, IOWIN, REG_REDTBL_BASE};
        use std::cell::Cell;

        let handle = IoapicHandle::new(std::sync::Arc::new(PiMutex::new(Ioapic::new())), -1);
        // Pin 6's low-dword redtbl register (entry i lo = REG_REDTBL_BASE + 2i).
        let lo_reg = (REG_REDTBL_BASE + 2 * 6) as u8;
        let installs = Cell::new(0u32);

        // Drive one IOAPIC MMIO write through the dedup path with an injected
        // installer that bumps `installs` and returns Ok/Err per `ok`.
        let step = |off: u64, data: &[u8], ok: bool| -> Result<()> {
            handle.mmio_write_with(off, data, |_routing| {
                installs.set(installs.get() + 1);
                if ok {
                    Ok(())
                } else {
                    Err(std::io::Error::other("injected install failure"))
                }
            })
        };

        // Program pin 6's RTE: select the lo reg (not a route change → no
        // install), then write vector 0x40 with the mask bit clear (an
        // unmasked route) → route change → install #1.
        step(IOREGSEL, &[lo_reg], true).unwrap();
        step(IOWIN, &0x40u32.to_le_bytes(), true).unwrap();
        assert_eq!(
            installs.get(),
            1,
            "programming an unmasked RTE installs once"
        );

        // Rewrite the identical lo dword: the register file reports dirty, but
        // the route set is byte-identical → dedup SKIPS the install ioctl.
        step(IOREGSEL, &[lo_reg], true).unwrap();
        step(IOWIN, &0x40u32.to_le_bytes(), true).unwrap();
        assert_eq!(
            installs.get(),
            1,
            "a redundant RTE rewrite must dedup (no second install)"
        );

        // Cache-on-success-only: change the vector (0x50) with a FAILING
        // installer → the install is attempted (count 2) but errors, so
        // `last_installed` must NOT be updated to the 0x50 route set.
        step(IOREGSEL, &[lo_reg], true).unwrap();
        assert!(
            step(IOWIN, &0x50u32.to_le_bytes(), false).is_err(),
            "an injected install failure propagates as an error"
        );
        assert_eq!(installs.get(), 2, "the changed RTE attempts an install");
        assert_eq!(
            handle.routing_failures(),
            1,
            "the failed install is counted"
        );

        // Retry the SAME changed RTE with a succeeding installer. Because the
        // failed install did not cache 0x50, this must install AGAIN (count 3)
        // rather than dedup-skip — proving a failed install never wedges a
        // device behind a poisoned cache.
        step(IOREGSEL, &[lo_reg], true).unwrap();
        step(IOWIN, &0x50u32.to_le_bytes(), true).unwrap();
        assert_eq!(
            installs.get(),
            3,
            "a failed install must not poison the cache — the identical retry re-installs"
        );
    }

    /// On split-irqchip `IoapicHandle` owns the COMBINED routing table: the
    /// userspace-IOAPIC RTE translations PLUS the device MSI-X routes. Either
    /// trigger — an IOAPIC RTE write (`mmio_write_with`) or an MSI-X route edge
    /// (`set_route_with`) — rebuilds the whole table from both halves, so a
    /// change to one half never drops the other from the `KVM_SET_GSI_ROUTING`
    /// whole-table replace. Drives both seams with a capturing installer that
    /// records the installed GSI set.
    #[test]
    fn ioapic_handle_combines_ioapic_and_msix_routes_each_edge_preserves_the_other() {
        use crate::vmm::x86_64::ioapic::{IOREGSEL, IOWIN, REG_REDTBL_BASE};
        use std::cell::RefCell;
        use std::collections::BTreeSet;

        let handle = IoapicHandle::new(std::sync::Arc::new(PiMutex::new(Ioapic::new())), -1);
        // The GSI set of the most recent install (the whole-table replace).
        let last: RefCell<BTreeSet<u32>> = RefCell::new(BTreeSet::new());

        // Program an IOAPIC pin's RTE to `vector`, unmasked (mask bit clear) — a
        // route change → install; capture the installed whole-table GSI set.
        let program_pin = |pin: u32, vector: u32| {
            let lo_reg = (REG_REDTBL_BASE + 2 * pin) as u8;
            handle
                .mmio_write_with(IOREGSEL, &[lo_reg], |r| {
                    *last.borrow_mut() = r.as_slice().iter().map(|e| e.gsi).collect();
                    Ok(())
                })
                .unwrap();
            handle
                .mmio_write_with(IOWIN, &vector.to_le_bytes(), |r| {
                    *last.borrow_mut() = r.as_slice().iter().map(|e| e.gsi).collect();
                    Ok(())
                })
                .unwrap();
        };
        let msi = |data: u32| MsiRoute {
            address_lo: 0xFEE0_0000,
            address_hi: 0,
            data,
        };
        let set_msix = |gsi: u32, route: Option<MsiRoute>| {
            handle
                .set_route_with(gsi, route, |r| {
                    *last.borrow_mut() = r.as_slice().iter().map(|e| e.gsi).collect();
                    Ok(())
                })
                .unwrap();
        };

        // 1) IOAPIC pin 6 only → the table has the IOAPIC half.
        program_pin(6, 0x40);
        assert_eq!(
            *last.borrow(),
            BTreeSet::from([6]),
            "an IOAPIC RTE install carries the IOAPIC half"
        );

        // 2) Add an MSI-X route at GSI 25 → the table now has BOTH halves
        //    (the MSI-X edge appends to, not replaces, the IOAPIC half).
        set_msix(25, Some(msi(0x4001)));
        assert_eq!(
            *last.borrow(),
            BTreeSet::from([6, 25]),
            "an MSI-X edge appends to — does not replace — the IOAPIC half"
        );

        // 3) Program another IOAPIC pin (7) → the MSI-X half SURVIVES the
        //    whole-table replace triggered by the IOAPIC change.
        program_pin(7, 0x41);
        assert_eq!(
            *last.borrow(),
            BTreeSet::from([6, 7, 25]),
            "an IOAPIC change must not drop the MSI-X half"
        );

        // 4) Remove the MSI-X route (None) → the IOAPIC half SURVIVES; only the
        //    MSI-X GSI is dropped.
        set_msix(25, None);
        assert_eq!(
            *last.borrow(),
            BTreeSet::from([6, 7]),
            "removing an MSI-X route drops only its GSI, keeping the IOAPIC half"
        );
    }

    /// The MSI-X route edge (`set_route_with`) reuses the same combined-table
    /// dedup + cache-on-success-only logic as the IOAPIC path: an identical
    /// route set skips the (SRCU-grace) ioctl, and a failed install neither
    /// caches nor poisons a later identical retry. Drives the `install` FnOnce
    /// seam with an injected counting/failing closure (no live KVM fd).
    #[test]
    fn ioapic_handle_msix_set_route_dedups_and_caches_on_success_only() {
        use std::cell::Cell;

        let handle = IoapicHandle::new(std::sync::Arc::new(PiMutex::new(Ioapic::new())), -1);
        let installs = Cell::new(0u32);
        let msi = |data: u32| MsiRoute {
            address_lo: 0xFEE0_0000,
            address_hi: 0,
            data,
        };

        // First MSI-X route at GSI 25 (empty IOAPIC half) → install #1.
        handle
            .set_route_with(25, Some(msi(0x4000)), |_r| {
                installs.set(installs.get() + 1);
                Ok(())
            })
            .unwrap();
        assert_eq!(installs.get(), 1, "the first MSI-X route installs");

        // Identical route → the combined table is byte-identical → dedup.
        handle
            .set_route_with(25, Some(msi(0x4000)), |_r| {
                installs.set(installs.get() + 1);
                Ok(())
            })
            .unwrap();
        assert_eq!(installs.get(), 1, "an identical MSI-X route dedups");

        // Changed route (data 0x4001) with a FAILING installer → attempts
        // (count 2), errors, counts the failure, and must NOT cache.
        assert!(
            handle
                .set_route_with(25, Some(msi(0x4001)), |_r| {
                    installs.set(installs.get() + 1);
                    Err(std::io::Error::other("injected install failure"))
                })
                .is_err(),
            "an injected install failure propagates as an error"
        );
        assert_eq!(
            installs.get(),
            2,
            "the changed MSI-X route attempts an install"
        );
        assert_eq!(
            handle.routing_failures(),
            1,
            "the failed MSI-X install is counted"
        );

        // Retry the SAME changed route with a succeeding installer → re-installs
        // (count 3), proving the failed install did not poison the cache.
        handle
            .set_route_with(25, Some(msi(0x4001)), |_r| {
                installs.set(installs.get() + 1);
                Ok(())
            })
            .unwrap();
        assert_eq!(
            installs.get(),
            3,
            "a failed MSI-X install must not poison the cache — the identical retry re-installs"
        );
    }

    #[test]
    fn build_device_msi_routing_lays_out_fam_entries() {
        let routes = vec![
            (
                4u32,
                MsiRoute {
                    address_lo: 0xFEE0_1004,
                    address_hi: 0x0000_0100,
                    data: 0x0000_8030,
                },
            ),
            (
                6u32,
                MsiRoute {
                    address_lo: 0xFEE0_2000,
                    address_hi: 0x0000_0000,
                    data: 0x0000_0040,
                },
            ),
        ];
        let mut routing = build_device_msi_routing(&routes).expect("build routing");
        let entries = routing.as_mut_slice();
        assert_eq!(entries.len(), 2, "one FAM entry per route");
        for (i, (gsi, msi)) in routes.iter().enumerate() {
            let e = &entries[i];
            assert_eq!(e.gsi, *gsi, "entry {i} gsi");
            assert_eq!(e.type_, KVM_IRQ_ROUTING_MSI, "entry {i} type is MSI");
            assert_eq!(e.flags, 0, "entry {i} flags");
            // SAFETY: every entry was built as the `.msi` union variant above.
            let m = unsafe { e.u.msi };
            assert_eq!(m.address_lo, msi.address_lo, "entry {i} address_lo");
            assert_eq!(m.address_hi, msi.address_hi, "entry {i} address_hi");
            assert_eq!(m.data, msi.data, "entry {i} data");
        }
    }

    #[test]
    fn build_device_msi_routing_empty_is_valid() {
        // All-masked IOAPIC -> empty route set -> nr=0 table (the re-mask-all
        // case: FamStructWrapper::new(0) yields a
        // valid header-only kvm_irq_routing{nr:0}).
        let mut routing = build_device_msi_routing(&[]).expect("empty routing");
        assert_eq!(routing.as_mut_slice().len(), 0, "no entries for empty set");
    }

    /// `FullIrqchipRouteOwner::install_snapshot` dedups a byte-identical reinstall
    /// AND caches only on success (a failed install must not poison a later
    /// identical retry) — the same invariants the IoapicHandle test guards, on
    /// the new full-irqchip owner. Drives the `install` FnOnce seam with an
    /// injected counting/failing closure (no live KVM fd).
    #[test]
    fn full_irqchip_route_owner_dedups_and_caches_on_success_only() {
        use std::cell::Cell;
        let owner = FullIrqchipRouteOwner::new(-1);
        let installs = Cell::new(0u32);
        let route_a = vec![(
            24u32,
            MsiRoute {
                address_lo: 0xFEE0_0000,
                address_hi: 0,
                data: 0x4000,
            },
        )];
        let route_b = vec![(
            24u32,
            MsiRoute {
                address_lo: 0xFEE0_0000,
                address_hi: 0,
                data: 0x4001, // different data → different route set than A
            },
        )];

        owner
            .install_snapshot(route_a.clone(), |_r| {
                installs.set(installs.get() + 1);
                Ok(())
            })
            .unwrap();
        assert_eq!(installs.get(), 1, "first install runs");

        // Byte-identical reinstall → dedup, no ioctl.
        owner
            .install_snapshot(route_a.clone(), |_r| {
                installs.set(installs.get() + 1);
                Ok(())
            })
            .unwrap();
        assert_eq!(installs.get(), 1, "byte-identical reinstall dedups");

        // Changed route set with a FAILING installer → attempts (count 2),
        // errors, and must NOT cache route_b.
        assert!(
            owner
                .install_snapshot(route_b.clone(), |_r| {
                    installs.set(installs.get() + 1);
                    Err(std::io::Error::other("injected install failure"))
                })
                .is_err(),
            "an injected install failure propagates as an error"
        );
        assert_eq!(
            installs.get(),
            2,
            "the changed route set attempts an install"
        );
        assert_eq!(owner.routing_failures(), 1, "the failed install is counted");

        // Retry the SAME route_b with a succeeding installer: because the failed
        // install did not cache route_b, this re-installs (count 3) rather than
        // dedup-skip — a failed install must not wedge a device behind a poisoned
        // cache.
        owner
            .install_snapshot(route_b.clone(), |_r| {
                installs.set(installs.get() + 1);
                Ok(())
            })
            .unwrap();
        assert_eq!(
            installs.get(),
            3,
            "a failed install must not poison the cache — the identical retry re-installs"
        );
    }

    /// `build_full_irqchip_routing` emits the explicit x86 default routing (24
    /// IOAPIC pins for GSI 0..24 + 16 PIC master/slave for GSI 0..16) followed by
    /// the device MSI routes — byte-route-identical to the kernel's
    /// `default_routing` (arch/x86/kvm/irq.c) so the first MSI-X
    /// KVM_SET_GSI_ROUTING does not drop the legacy INTx routes.
    #[test]
    fn build_full_irqchip_routing_lays_out_defaults_plus_msi() {
        let msi = vec![(
            24u32,
            MsiRoute {
                address_lo: 0xFEE0_1000,
                address_hi: 0,
                data: 0x0000_4030,
            },
        )];
        let mut routing = build_full_irqchip_routing(&msi).expect("build full-irqchip routing");
        let entries = routing.as_mut_slice();
        assert_eq!(entries.len(), 24 + 16 + 1, "24 IOAPIC + 16 PIC + 1 MSI");

        // IOAPIC pins: GSI 0..24, irqchip = IOAPIC, pin == gsi.
        for gsi in 0..24u32 {
            let e = &entries[gsi as usize];
            assert_eq!(e.gsi, gsi, "IOAPIC entry {gsi} gsi");
            assert_eq!(e.type_, KVM_IRQ_ROUTING_IRQCHIP, "IOAPIC entry {gsi} type");
            // SAFETY: built as the `.irqchip` union variant in build_full_irqchip_routing.
            let c = unsafe { e.u.irqchip };
            assert_eq!(c.irqchip, KVM_IRQCHIP_IOAPIC, "IOAPIC entry {gsi} chip");
            assert_eq!(c.pin, gsi, "IOAPIC entry {gsi} pin == gsi");
        }
        // PIC: GSI 0..16, master (<8) / slave, pin = gsi % 8.
        for gsi in 0..16u32 {
            let e = &entries[24 + gsi as usize];
            assert_eq!(e.gsi, gsi, "PIC entry {gsi} gsi");
            assert_eq!(e.type_, KVM_IRQ_ROUTING_IRQCHIP, "PIC entry {gsi} type");
            // SAFETY: built as the `.irqchip` union variant.
            let c = unsafe { e.u.irqchip };
            let want = if gsi < 8 {
                KVM_IRQCHIP_PIC_MASTER
            } else {
                KVM_IRQCHIP_PIC_SLAVE
            };
            assert_eq!(c.irqchip, want, "PIC entry {gsi} master/slave");
            assert_eq!(c.pin, gsi % 8, "PIC entry {gsi} pin == gsi%8");
        }
        // MSI route appended after the 40 defaults, carrying the dwords verbatim.
        let e = &entries[40];
        assert_eq!(e.gsi, 24, "MSI entry gsi");
        assert_eq!(e.type_, KVM_IRQ_ROUTING_MSI, "MSI entry type");
        // SAFETY: built as the `.msi` union variant.
        let m = unsafe { e.u.msi };
        assert_eq!(m.address_lo, 0xFEE0_1000, "MSI address_lo verbatim");
        assert_eq!(m.data, 0x0000_4030, "MSI data verbatim");
    }

    /// [`translate_msi_ext_dest_id`] moves a guest's `virt_ext_dest_id` high
    /// destination bits (address_lo `[11:5]`) into address_hi `[31:8]` so KVM's
    /// `x86_msi_msg_get_destid` (which reads the extended dest from address_hi
    /// under `KVM_X2APIC_API_USE_32BIT_IDS`) decodes the full APIC ID > 255. The
    /// post-swizzle address_hi must byte-match what [`Ioapic::redtbl_to_msi`]
    /// emits for the same dest — INTx and swizzled MSI-X produce the identical
    /// KVM route, so both deliver to the same APIC.
    #[test]
    fn translate_msi_ext_dest_id_swizzles_virt_destid_into_addr_hi() {
        // dest 256: guest packs dest>>8 = 1 into virt_destid (addr_lo [11:5] =
        // 1<<5 = 0x20), destid_0_7 = 256 & 0xff = 0, addr_hi = 0.
        let (lo, hi) = translate_msi_ext_dest_id(0xFEE0_0000 | 0x20, 0);
        assert_eq!(
            hi, 0x100,
            "dest>>8 lands at addr_hi[31:8] (destid_8_31 LSB)"
        );
        assert_eq!(lo & 0xfe0, 0, "virt-destid bits cleared from addr_lo");
        // == redtbl_to_msi's (dest>>8)<<8 for dest 256 → INTx and MSI-X agree.
        assert_eq!(
            hi,
            (256u32 >> 8) << 8,
            "matches the host-composed INTx addr_hi"
        );

        // Max virt_ext_dest_id dest (0x7FFF, the <0x8000 cap): dest>>8 = 0x7F →
        // virt_destid = 0x7F << 5 = 0xFE0 → addr_hi = 0x7F << 8 = 0x7F00.
        let (_lo, hi) = translate_msi_ext_dest_id(0xFEE0_0000 | 0xFE0, 0);
        assert_eq!(
            hi, 0x7F00,
            "full 7-bit virt-destid range swizzles correctly"
        );

        // dest <= 255: no virt-destid bits set → no-op.
        let lo_in = 0xFEE0_0000 | (200u32 << 12); // destid_0_7 = 200, no [11:5]
        assert_eq!(
            translate_msi_ext_dest_id(lo_in, 0),
            (lo_in, 0),
            "a <=255 dest (no virt-destid bits) is unchanged"
        );

        // Already-composed addr_hi (e.g. a host-built IOAPIC route from
        // redtbl_to_msi for dest>255) is left untouched — no double-swizzle.
        assert_eq!(
            translate_msi_ext_dest_id(0xFEE0_0000 | 0x20, 0x100),
            (0xFEE0_0000 | 0x20, 0x100),
            "addr_hi != 0 skips the swizzle (no double-translate)"
        );

        // Interrupt-format bit (addr_lo bit 4 = dmar_format) set → a
        // remappable-format message we must never rewrite; left untouched even
        // with addr_hi == 0.
        assert_eq!(
            translate_msi_ext_dest_id(0xFEE0_0000 | 0x10 | 0x20, 0),
            (0xFEE0_0000 | 0x10 | 0x20, 0),
            "the RF bit skips the swizzle"
        );
    }

    use std::os::fd::AsRawFd;
    use vm_memory::GuestMemory;

    #[test]
    fn create_vm_basic() {
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 2,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
            llc_cores: None,
        };
        let vm = KtstrKvm::new(topo, 128, false);
        assert!(vm.is_ok(), "VM creation failed: {:?}", vm.err());
        let vm = vm.unwrap();
        assert_eq!(vm.vcpus.len(), 2);
    }

    #[test]
    fn create_vm_multi_llc() {
        let topo = Topology {
            llcs: 2,
            cores_per_llc: 2,
            threads_per_core: 2,
            numa_nodes: 1,
            nodes: None,
            distances: None,
            llc_cores: None,
        };
        let vm = KtstrKvm::new(topo, 256, false);
        assert!(vm.is_ok(), "multi-LLC VM creation failed: {:?}", vm.err());
        let vm = vm.unwrap();
        assert_eq!(vm.vcpus.len(), 8);
    }

    #[test]
    fn create_vm_single_cpu() {
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
            llc_cores: None,
        };
        let vm = KtstrKvm::new(topo, 64, false);
        assert!(vm.is_ok());
        assert_eq!(vm.unwrap().vcpus.len(), 1);
    }

    #[test]
    fn create_vm_large_topology() {
        let topo = Topology {
            llcs: 4,
            cores_per_llc: 4,
            threads_per_core: 2,
            numa_nodes: 1,
            nodes: None,
            distances: None,
            llc_cores: None,
        };
        let vm = KtstrKvm::new(topo, 512, false);
        assert!(vm.is_ok(), "large topology failed: {:?}", vm.err());
        assert_eq!(vm.unwrap().vcpus.len(), 32);
    }

    #[test]
    fn create_vm_odd_topology() {
        let topo = Topology {
            llcs: 3,
            cores_per_llc: 3,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
            llc_cores: None,
        };
        let vm = KtstrKvm::new(topo, 128, false);
        assert!(vm.is_ok(), "odd topology failed: {:?}", vm.err());
        assert_eq!(vm.unwrap().vcpus.len(), 9);
    }

    #[test]
    fn memory_size_correct() {
        use vm_memory::GuestMemoryRegion;
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
            llc_cores: None,
        };
        let vm = KtstrKvm::new(topo, 256, false).unwrap();
        let total: u64 = vm.guest_mem.iter().map(|r| r.len()).sum();
        assert_eq!(total, 256 << 20);
    }

    #[test]
    fn tss_address_matches_firecracker() {
        assert_eq!(KVM_TSS_ADDRESS, 0xfffb_d000);
    }

    #[test]
    fn identity_map_follows_tss() {
        assert_eq!(KVM_IDENTITY_MAP_ADDRESS, KVM_TSS_ADDRESS + 3 * 4096);
        assert_eq!(KVM_IDENTITY_MAP_ADDRESS, 0xfffc_0000);
    }

    #[test]
    fn required_caps_non_empty() {
        assert!(!REQUIRED_CAPS.is_empty());
        assert!(REQUIRED_CAPS.len() >= 14);
    }

    #[test]
    fn small_topology_uses_full_irqchip() {
        let topo = Topology {
            llcs: 2,
            cores_per_llc: 4,
            threads_per_core: 2,
            numa_nodes: 1,
            nodes: None,
            distances: None,
            llc_cores: None,
        };
        // max APIC ID = apic_id(15) = 1<<3 | 3<<1 | 1 = 15, well under 254
        assert!(max_apic_id(&topo) <= MAX_XAPIC_ID);
        let vm = KtstrKvm::new(topo, 256, false).unwrap();
        assert!(!vm.split_irqchip, "small topology should use full IRQ chip");
    }

    #[test]
    fn large_topology_uses_split_irqchip() {
        // 15 LLCs x 8 cores x 2 threads = 240 vCPUs
        // max APIC ID = apic_id(239) = 14<<4 | 7<<1 | 1 = 239, under 254
        // So try bigger: 14 LLCs x 9 cores x 2 threads = 252 vCPUs
        // core_bits = bits_needed(9) = 4, thread_bits = 1, core_shift = 5
        // max APIC ID = apic_id(251) = 13<<5 | 8<<1 | 1 = 433
        let topo = Topology {
            llcs: 14,
            cores_per_llc: 9,
            threads_per_core: 2,
            numa_nodes: 1,
            nodes: None,
            distances: None,
            llc_cores: None,
        };
        assert!(
            max_apic_id(&topo) > MAX_XAPIC_ID,
            "max APIC ID {} should exceed {}",
            max_apic_id(&topo),
            MAX_XAPIC_ID,
        );
        let vm = match KtstrKvm::new(topo, 4096, false) {
            Ok(v) => v,
            Err(e) => {
                // Some hosts reject 252-vCPU VMs (EEXIST from
                // KVM_CREATE_VCPU when split irqchip + x2APIC
                // interact with host KVM limitations). The APIC ID
                // assertion above validates the split irqchip logic;
                // skip the VM creation test on those hosts.
                skip!("large_topology VM creation: {e:#}");
            }
        };
        assert!(vm.split_irqchip, "large topology should use split IRQ chip");
        assert_eq!(vm.vcpus.len(), 252);
    }

    #[test]
    fn split_irqchip_boundary() {
        // Find a topology that is exactly at the boundary.
        // 8 LLCs x 8 cores x 2 threads: core_shift = 4, max APIC ID = 7<<4 | 7<<1 | 1 = 127
        let small = Topology {
            llcs: 8,
            cores_per_llc: 8,
            threads_per_core: 2,
            numa_nodes: 1,
            nodes: None,
            distances: None,
            llc_cores: None,
        };
        assert!(
            max_apic_id(&small) <= MAX_XAPIC_ID,
            "8l/8c/2t max APIC ID {} should be <= 254",
            max_apic_id(&small),
        );
        let vm = KtstrKvm::new(small, 2048, false).unwrap();
        assert!(!vm.split_irqchip);

        // 15 LLCs x 8 cores x 2 threads: core_shift = 4, max APIC ID = 14<<4 | 7<<1 | 1 = 239
        let still_small = Topology {
            llcs: 15,
            cores_per_llc: 8,
            threads_per_core: 2,
            numa_nodes: 1,
            nodes: None,
            distances: None,
            llc_cores: None,
        };
        assert!(
            max_apic_id(&still_small) <= MAX_XAPIC_ID,
            "15l/8c/2t max APIC ID {} should be <= 254",
            max_apic_id(&still_small),
        );
        let vm = KtstrKvm::new(still_small, 4096, false).unwrap();
        assert!(!vm.split_irqchip);
    }

    #[test]
    fn immediate_exit_cap_detected() {
        use crate::vmm::x86_64::test_helpers::single_vcpu_kvm;
        let vm = single_vcpu_kvm();
        // KVM_CAP_IMMEDIATE_EXIT is available since Linux 4.12.
        assert!(vm.has_immediate_exit);
    }

    #[test]
    fn performance_mode_succeeds() {
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 2,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
            llc_cores: None,
        };
        let vm = KtstrKvm::new(topo, 128, true);
        assert!(
            vm.is_ok(),
            "performance_mode VM creation failed: {:?}",
            vm.err()
        );
    }

    #[test]
    fn performance_mode_does_not_affect_vcpu_count() {
        let topo = Topology {
            llcs: 2,
            cores_per_llc: 2,
            threads_per_core: 2,
            numa_nodes: 1,
            nodes: None,
            distances: None,
            llc_cores: None,
        };
        let vm_normal = KtstrKvm::new(topo, 256, false).unwrap();
        let vm_perf = KtstrKvm::new(topo, 256, true).unwrap();
        assert_eq!(vm_normal.vcpus.len(), vm_perf.vcpus.len());
    }

    #[test]
    fn non_perf_mode_vm_constructs() {
        // Halt polling is no longer set at construction (it moved to
        // run() keyed on the run-lock outcome); a non-perf VM must still
        // build cleanly.
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 2,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
            llc_cores: None,
        };
        let vm = KtstrKvm::new(topo, 128, false);
        assert!(
            vm.is_ok(),
            "non-perf VM construction failed: {:?}",
            vm.err()
        );
    }

    #[test]
    fn set_halt_poll_on_real_vm() {
        // The cap is settable at any time (api.rst 7.20); 0 disables host
        // polling for this VM. Best-effort — no panic on an old host.
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
            llc_cores: None,
        };
        let vm = KtstrKvm::new(topo, 64, false).unwrap();
        vm.set_halt_poll(0);
    }

    #[test]
    fn disable_exits_hlt_bit_value() {
        // KVM_X86_DISABLE_EXITS_HLT is bit 1 (value 2) in the kernel ABI.
        assert_eq!(KVM_X86_DISABLE_EXITS_HLT, 2);
    }

    #[test]
    fn disable_exits_pause_and_hlt_no_overlap() {
        assert_ne!(
            KVM_X86_DISABLE_EXITS_PAUSE, KVM_X86_DISABLE_EXITS_HLT,
            "PAUSE and HLT bits must be distinct"
        );
        assert_eq!(
            KVM_X86_DISABLE_EXITS_PAUSE & KVM_X86_DISABLE_EXITS_HLT,
            0,
            "PAUSE and HLT bits must not overlap"
        );
    }

    #[test]
    fn tsc_stability_check_roundtrip() {
        // Check the get→set→get roundtrip succeeds with
        // performance_mode=true (which enables the TSC check).
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 2,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
            llc_cores: None,
        };
        let vm = KtstrKvm::new(topo, 64, true).unwrap();
        let clock = vm.vm_fd.get_clock().unwrap();
        let mut set_data = clock;
        set_data.flags = 0;
        vm.vm_fd.set_clock(&set_data).unwrap();
        let clock2 = vm.vm_fd.get_clock().unwrap();
        // On bare-metal with invariant TSC, KVM_CLOCK_TSC_STABLE
        // should be set after the roundtrip forces
        // pvclock_update_vm_gtod_copy. In nested virt it may not be.
        // Either way, the roundtrip must not fail.
        let _ = clock2.flags & KVM_CLOCK_TSC_STABLE;
    }

    #[test]
    fn kvm_clock_data_default_is_zeroed() {
        let clock = kvm_bindings::kvm_clock_data::default();
        assert_eq!(clock.clock, 0);
        assert_eq!(clock.flags, 0);
        assert_eq!(clock.pad0, 0);
        assert_eq!(clock.realtime, 0);
        assert_eq!(clock.host_tsc, 0);
        assert_eq!(clock.pad, [0u32; 4]);
    }

    #[test]
    fn kvm_clock_data_size_matches_ioctl_encoding() {
        // The hand-encoded `_IOC_SIZE = 0x30 = 48` in the
        // KVM_GET_CLOCK / KVM_SET_CLOCK ioctl-number constants in
        // this file presumes this exact size. A compile-time
        // `const _: () = assert!(...)` next to the first constant
        // guards builds; this runtime check is a belt-and-
        // suspenders guard against a future split of kvm-bindings
        // that drops the compile-time assert.
        assert_eq!(std::mem::size_of::<kvm_bindings::kvm_clock_data>(), 48);
    }

    #[test]
    fn raw_fd_get_clock_matches_safe_wrapper() {
        // Cross-check: the hand-encoded ioctl number 0x8030_ae7c
        // hits the same kernel path as kvm_ioctls::VmFd::get_clock.
        // If the number were wrong, libc::ioctl would return ENOTTY
        // (-22), which surfaces as Err and the assertion below would
        // observe it. If the number aimed at a different ioctl, the
        // returned clock value would not advance monotonically and
        // the safe-vs-raw comparison would diverge dramatically.
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
            llc_cores: None,
        };
        let vm = KtstrKvm::new(topo, 64, false).unwrap();
        let raw_fd = vm.vm_fd.as_raw_fd();
        let via_safe = vm.vm_fd.get_clock().expect("safe GET_CLOCK");
        let via_raw = super::kvm_get_clock_via_raw_fd(raw_fd).expect("raw GET_CLOCK");
        // Both reads hit the same in-kernel pvclock via separate
        // seqcount reads; the later one must be >= the earlier
        // (kvm_clock is monotonic non-decreasing).
        assert!(
            via_raw.clock >= via_safe.clock,
            "raw-fd GET regressed below safe GET (raw={}, safe={}) — ioctl number drift",
            via_raw.clock,
            via_safe.clock,
        );
        // < 1 second drift means we are reading the same ioctl,
        // not some unrelated kernel time source.
        assert!(
            via_raw.clock - via_safe.clock < 1_000_000_000,
            "raw-fd vs safe GET differ by >1s (raw={}, safe={}) — likely different kernel state",
            via_raw.clock,
            via_safe.clock,
        );
    }

    #[test]
    fn raw_fd_set_clock_roundtrip_with_flags_zero() {
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
            llc_cores: None,
        };
        let vm = KtstrKvm::new(topo, 64, false).unwrap();
        let raw_fd = vm.vm_fd.as_raw_fd();
        let mut clock = super::kvm_get_clock_via_raw_fd(raw_fd).expect("raw GET_CLOCK");
        clock.flags = 0;
        super::kvm_set_clock_via_raw_fd(raw_fd, &clock).expect("raw SET_CLOCK");
        let after = super::kvm_get_clock_via_raw_fd(raw_fd).expect("raw GET_CLOCK after");
        assert!(after.clock >= clock.clock);
    }

    #[test]
    fn raw_fd_set_gsi_routing_with_msi_route_succeeds() {
        // Cross-check: the hand-encoded ioctl 0x4008_AE6A hits
        // KVM_SET_GSI_ROUTING. A wrong direction bit, _IOC_SIZE, or a
        // typo in the constant fails at runtime — ENOTTY (wrong
        // number) or EINVAL (wrong direction/size) — surfacing as Err
        // here. This is the only one of the three hand-encoded raw-fd
        // ioctls without a cross-check (the two clock siblings are
        // covered above); a wrong encoding would otherwise fail
        // silently on the split-irqchip device-MSI-routing path (no
        // IRQ delivery → guest hang on first device use), surfaced only
        // by `routing_failures` at runtime.
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
            llc_cores: None,
        };
        let vm = KtstrKvm::new(topo, 64, false).unwrap();
        let raw_fd = vm.vm_fd.as_raw_fd();
        // KVM_SET_GSI_ROUTING is a whole-table replace; install one
        // valid MSI route. This small topology uses the in-kernel
        // irqchip (split-irqchip engages only above MAX_XAPIC_ID), and
        // an MSI route is valid under either irqchip mode — the ioctl
        // encoding under test is irqchip-mode-independent.
        let routing = build_device_msi_routing(&[(
            0u32,
            MsiRoute {
                address_lo: 0xFEE0_0000,
                address_hi: 0,
                data: 0x0000_0040,
            },
        )])
        .expect("build one-MSI-route routing");
        super::kvm_set_gsi_routing_via_raw_fd(raw_fd, &routing)
            .expect("raw KVM_SET_GSI_ROUTING with one MSI route");
    }

    #[test]
    fn performance_mode_with_hlt_disable_succeeds() {
        // performance_mode issues two separate enable_cap calls:
        // PAUSE (always succeeds) then HLT (may be rejected by
        // mitigate_smt_rsb). Either way, VM creation must succeed.
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 2,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
            llc_cores: None,
        };
        let vm = KtstrKvm::new(topo, 128, true);
        assert!(
            vm.is_ok(),
            "performance_mode with HLT disable failed: {:?}",
            vm.err()
        );
    }
}
