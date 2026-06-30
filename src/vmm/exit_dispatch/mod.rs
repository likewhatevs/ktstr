//! vCPU exit classification and per-arch I/O dispatch.
//!
//! Shared between BSP and AP run loops. Each exit gets classified into
//! [`ExitAction`] (Continue / Shutdown / Fatal); arch-specific I/O is
//! dispatched inline so the surrounding loop only sees the action.
//!
//! - x86_64: serial via port I/O (`dispatch_io_out` / `dispatch_io_in`),
//!   virtio-console via MMIO inside [`classify_exit`], i8042 reset for reboot.
//! - aarch64: serial + virtio-console both via MMIO (`dispatch_mmio_write`
//!   / `dispatch_mmio_read`).

use crate::sync::MutexExt;
use crate::vmm::IoapicHandle;
use crate::vmm::PiMutex;
use crate::vmm::vcpu::{SCX_EXIT_ERROR_THRESHOLD, WatchpointArm, self_arm_watchpoint};
use crate::vmm::{console, kvm, pci, virtio_blk, virtio_console, virtio_net};
use kvm_ioctls::VcpuExit;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use vmm_sys_util::eventfd::EventFd;

/// Snapshot of a vCPU's architectural state, captured by the vCPU
/// thread itself at freeze time (just before it parks). Surfaced in
/// the failure-dump report so an operator can correlate the observed
/// guest-memory state with where each vCPU was executing.
///
/// Field naming is arch-neutral: each value is set from the matching
/// per-arch register so the layout is identical across x86_64 and
/// aarch64 in JSON / Display output.
///
/// Capture must run ON the vCPU thread (not cross-thread) because
/// `KVM_GET_REGS` / `KVM_GET_SREGS` are vCPU-fd-bound ioctls; their
/// thread-affinity is a KVM API contract documented in the kernel
/// vCPU lifecycle (Documentation/virt/kvm/api.rst). Calling them
/// from a different thread reads stale state at best and races KVM's
/// internal vCPU state machine at worst.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[non_exhaustive]
pub struct VcpuRegSnapshot {
    /// Instruction pointer at freeze time (`rip` on x86_64,
    /// `pc` on aarch64). Identifies the kernel/userspace function
    /// the vCPU was executing when the freeze coordinator's kick
    /// arrived.
    pub instruction_pointer: u64,
    /// Kernel-side stack pointer at freeze time (`rsp` on x86_64,
    /// `sp_el1` on aarch64 — explicitly NOT `sp_el0`, which is the
    /// userspace stack). Captures the EL1/CPL0 stack frame an
    /// operator can unwind against the BPF map dump for sched_ext
    /// failures, which fire in kernel context.
    pub stack_pointer: u64,
    /// Page-table root at freeze time. Captures arch-specific
    /// kernel-side state suitable for correlating the BPF map
    /// dump with the active address space:
    ///
    ///   - On x86_64: `cr3` — per-process pgd. Distinct from
    ///     [`crate::monitor::guest::GuestKernel::cr3_pa`], which
    ///     captures the boot-time `init_top_pgt` at coordinator
    ///     start. This snapshot field reflects whatever pgd the
    ///     vCPU was running on at freeze time (typically the
    ///     current task's mm); the boot-time value is what the
    ///     freeze coordinator uses for its own page-walks.
    ///
    ///   - On aarch64: `ttbr1_el1` — the kernel pgd. Stays
    ///     stable across context switches (TTBR0_EL1 swaps
    ///     per-task; see [`Self::user_page_table_root`] for the
    ///     userspace half).
    ///
    /// Raw register value with arch-specific flag bits intact
    /// (PCID/PCD/PWT on x86_64 CR3, ASID on aarch64 TTBR);
    /// consumers must mask before walking as a physical address.
    pub page_table_root: u64,
    /// Userspace page-table root at freeze time. arch-specific:
    ///
    ///   - On x86_64: always `None`. CR3 already covers both the
    ///     kernel and userspace halves of the address space —
    ///     [`Self::page_table_root`] alone identifies the active
    ///     mm.
    ///
    ///   - On aarch64: `Some(ttbr0_el1)` when capture succeeds,
    ///     `None` when KVM_GET_ONE_REG fails (mid-shutdown vCPU,
    ///     sysreg gated by the host kernel). TTBR0_EL1 holds the
    ///     userspace pgd that switches per-task, so it
    ///     identifies which task's userspace was active at
    ///     freeze time — useful for diagnosing user-context
    ///     traps. For sched_ext failures (kernel context),
    ///     TTBR1_EL1 in `page_table_root` is the primary signal.
    ///
    /// Raw register value with arch-specific flag bits intact
    /// (PCID/PCD/PWT on x86_64 CR3, ASID on aarch64 TTBR);
    /// consumers must mask before walking as a physical address.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_page_table_root: Option<u64>,
    /// aarch64 TCR_EL1 register at freeze time. Drives the
    /// granule-agnostic page-table walker
    /// ([`crate::monitor::reader::GuestMem::translate_kva`]):
    /// TG1 bits `[31:30]` select the high-half granule (4 KB / 16 KB
    /// / 64 KB) and T1SZ bits `[21:16]` determine the high-half VA
    /// width (`64 - T1SZ`). Stable after kernel MMU bring-up.
    /// `None` on x86_64 (the register does not exist) and on
    /// aarch64 if the KVM_GET_ONE_REG read fails mid-shutdown.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tcr_el1: Option<u64>,
}

/// Capture the vCPU's RIP/RSP/CR3 (or PC/SP/TTBR1 on aarch64) on
/// the calling thread. Invoked from `handle_freeze` after the drain
/// dance and before the `parked = true` Release store, so the
/// values reach the freeze coordinator via the same happens-before
/// edge the coordinator relies on for guest-memory reads.
///
/// `None` on capture failure — the get_regs / get_one_reg ioctls
/// can fail mid-shutdown when KVM has begun tearing down the vCPU.
/// The caller stores `None` into the per-vCPU slot in that case;
/// the dump reflects "registers unavailable" rather than panicking
/// the freeze path.
#[cfg(target_arch = "x86_64")]
pub(crate) fn capture_vcpu_regs(vcpu: &mut kvm_ioctls::VcpuFd) -> Option<VcpuRegSnapshot> {
    let regs = vcpu.get_regs().ok()?;
    let sregs = vcpu.get_sregs().ok()?;
    Some(VcpuRegSnapshot {
        instruction_pointer: regs.rip,
        stack_pointer: regs.rsp,
        page_table_root: sregs.cr3,
        // x86_64 has a single CR3 covering both halves of the
        // address space; no separate userspace pgd to capture.
        user_page_table_root: None,
        // TCR_EL1 is an aarch64 register; not present on x86_64.
        tcr_el1: None,
    })
}

#[cfg(target_arch = "aarch64")]
pub(crate) fn capture_vcpu_regs(vcpu: &mut kvm_ioctls::VcpuFd) -> Option<VcpuRegSnapshot> {
    // ARM core register IDs encode
    // `(offsetof(struct kvm_regs, field) / sizeof(u32))` in the low
    // bits, OR'd with KVM_REG_ARM64 + KVM_REG_SIZE_U64 +
    // KVM_REG_ARM_CORE (per kernel uapi/asm/kvm.h
    // `KVM_REG_ARM_CORE_REG` macro). The offset is into
    // `struct kvm_regs`, NOT directly into `struct user_pt_regs`;
    // the two coincide for the first 272 bytes because
    // `kvm_regs.regs` is at offset 0, but adding fields past
    // `user_pt_regs` (e.g. `sp_el1` below) requires the
    // `kvm_regs`-relative encoding.
    //
    // struct kvm_regs (arch/arm64/include/uapi/asm/kvm.h):
    //   struct user_pt_regs regs;     // offset 0..272
    //     u64 regs[31];               //   offset   0..248
    //     u64 sp;       (= sp_el0)    //   offset 248
    //     u64 pc;                     //   offset 256
    //     u64 pstate;                 //   offset 264
    //   u64 sp_el1;                   // offset 272
    //   u64 elr_el1;                  // offset 280
    //   u64 spsr[KVM_NR_SPSR];        // offset 288..
    //   ...
    //
    // The kernel-side stack pointer is `sp_el1`, NOT `regs.sp`
    // (which is `sp_el0` — the userspace stack pointer per the
    // comment at arch/arm64/include/uapi/asm/kvm.h:47). sched_ext
    // exits fire in EL1 (kernel context), so capturing sp_el1
    // yields the kernel stack frame an operator can unwind
    // against the BPF map dump. Capturing sp_el0 here would
    // leak the userspace stack of whatever task happened to be
    // current — typically irrelevant for kernel-side debugging
    // and confusing in the JSON output.
    //
    // Each u32 step is +1 in the encoded ID.
    const KVM_REG_ARM64: u64 = 0x6000_0000_0000_0000;
    const KVM_REG_SIZE_U64: u64 = 0x0030_0000_0000_0000;
    const KVM_REG_ARM_CORE: u64 = 0x0010_0000;
    // SP_EL1 lives at offset 272 in struct kvm_regs (right after
    // the 272-byte user_pt_regs). 272 / 4 = 68.
    const SP_EL1_ID: u64 = KVM_REG_ARM64 | KVM_REG_SIZE_U64 | KVM_REG_ARM_CORE | (272 / 4);
    // PC at offset 256 in user_pt_regs (= same offset in kvm_regs
    // because user_pt_regs.regs is at offset 0). 256 / 4 = 64.
    const PC_ID: u64 = KVM_REG_ARM64 | KVM_REG_SIZE_U64 | KVM_REG_ARM_CORE | (256 / 4);
    // ARM64 system registers encoded under KVM_REG_ARM64_SYSREG.
    // The 16-bit packing is
    //   (Op0 << 14) | (Op1 << 11) | (CRn << 7) | (CRm << 3) | Op2
    // per arch/arm64/include/uapi/asm/kvm.h `ARM64_SYS_REG` macro.
    const KVM_REG_ARM64_SYSREG: u64 = 0x0013_0000;
    // TTBR0_EL1: Op0=3, Op1=0, CRn=2, CRm=0, Op2=0
    // = (3 << 14) | (0 << 11) | (2 << 7) | (0 << 3) | 0 = 0xC100
    const TTBR0_EL1_ID: u64 = KVM_REG_ARM64 | KVM_REG_SIZE_U64 | KVM_REG_ARM64_SYSREG | 0xC100;
    // TTBR1_EL1: Op0=3, Op1=0, CRn=2, CRm=0, Op2=1
    // = (3 << 14) | (0 << 11) | (2 << 7) | (0 << 3) | 1 = 0xC101
    const TTBR1_EL1_ID: u64 = KVM_REG_ARM64 | KVM_REG_SIZE_U64 | KVM_REG_ARM64_SYSREG | 0xC101;
    // TCR_EL1: Op0=3, Op1=0, CRn=2, CRm=0, Op2=2
    // = (3 << 14) | (0 << 11) | (2 << 7) | (0 << 3) | 2 = 0xC102
    const TCR_EL1_ID: u64 = KVM_REG_ARM64 | KVM_REG_SIZE_U64 | KVM_REG_ARM64_SYSREG | 0xC102;

    let mut buf = [0u8; 8];
    let pc = vcpu
        .get_one_reg(PC_ID, &mut buf)
        .ok()
        .map(|_| u64::from_le_bytes(buf))?;
    let sp = vcpu
        .get_one_reg(SP_EL1_ID, &mut buf)
        .ok()
        .map(|_| u64::from_le_bytes(buf))?;
    // TTBR1 read is best-effort; some kernels gate sysreg access.
    // A failure leaves page_table_root = 0 — the boot-time
    // GuestKernel::cr3_pa is still available to the dump.
    let ttbr1 = vcpu
        .get_one_reg(TTBR1_EL1_ID, &mut buf)
        .ok()
        .map(|_| u64::from_le_bytes(buf))
        .unwrap_or(0);
    // TTBR0 read is best-effort. Stored in user_page_table_root
    // so a failure surfaces as None — distinct from a successful
    // read of 0, which means "no userspace mapping active at
    // freeze time" (e.g. the vCPU was running pure kernel code).
    let ttbr0 = vcpu
        .get_one_reg(TTBR0_EL1_ID, &mut buf)
        .ok()
        .map(|_| u64::from_le_bytes(buf));
    // TCR_EL1 carries the granule (TG1[31:30]) and high-half VA
    // size (T1SZ[21:16]) the page-table walker needs. Best-effort:
    // a failure leaves None and the walker falls back to the
    // boot-time cached value the freeze coordinator latched at
    // GuestKernel construction.
    let tcr_el1 = vcpu
        .get_one_reg(TCR_EL1_ID, &mut buf)
        .ok()
        .map(|_| u64::from_le_bytes(buf));
    Some(VcpuRegSnapshot {
        instruction_pointer: pc,
        stack_pointer: sp,
        page_table_root: ttbr1,
        user_page_table_root: ttbr0,
        tcr_el1,
    })
}

/// Read TCR_EL1 directly from a vCPU. Used at GuestKernel
/// construction time to feed the page-table walker its granule and
/// VA-width settings (TG1 in bits `[31:30]`, T1SZ in bits `[21:16]`).
///
/// Returns `None` on x86_64 (the register does not exist) and on
/// aarch64 if `KVM_GET_ONE_REG` fails. The caller treats `None` as
/// "no walker context yet"; on aarch64 that surfaces as a 0 stored
/// in the GuestKernel's `tcr_el1` field — the walker rejects T1SZ=0
/// and the affected lookups skip cleanly.
#[cfg(target_arch = "x86_64")]
pub(crate) fn read_tcr_el1(_vcpu: &mut kvm_ioctls::VcpuFd) -> Option<u64> {
    None
}

#[cfg(target_arch = "aarch64")]
pub(crate) fn read_tcr_el1(vcpu: &mut kvm_ioctls::VcpuFd) -> Option<u64> {
    // Same encoding constants as `capture_vcpu_regs`. TCR_EL1 packs
    // to (Op0=3, Op1=0, CRn=2, CRm=0, Op2=2) under the
    // KVM_REG_ARM64_SYSREG namespace.
    const KVM_REG_ARM64: u64 = 0x6000_0000_0000_0000;
    const KVM_REG_SIZE_U64: u64 = 0x0030_0000_0000_0000;
    const KVM_REG_ARM64_SYSREG: u64 = 0x0013_0000;
    const TCR_EL1_ID: u64 = KVM_REG_ARM64 | KVM_REG_SIZE_U64 | KVM_REG_ARM64_SYSREG | 0xC102;
    let mut buf = [0u8; 8];
    vcpu.get_one_reg(TCR_EL1_ID, &mut buf)
        .ok()
        .map(|_| u64::from_le_bytes(buf))
}

/// Read the BSP's kernel-half page-table root directly from a vCPU.
/// On x86_64 this is `CR3` from `KVM_GET_SREGS`; on aarch64 it is
/// `TTBR1_EL1` from `KVM_GET_ONE_REG`.
///
/// Used by the BSP loop's lazy-CAS to populate the
/// `super::freeze_coord::run_bsp_loop` cr3 cache once the guest
/// kernel has installed its post-randomization page tables. The
/// monitor and BPF map writer threads consume the cached value to
/// resolve `phys_base` via a page-table walk that breaks the
/// chicken-and-egg with text-symbol PA translation.
///
/// Returns `None` on KVM_GET_SREGS / KVM_GET_ONE_REG failure
/// (transient EINTR mid-shutdown); callers retry on the next
/// iteration. A successful return of `0` means the kernel has not
/// yet installed its page tables (very-early boot before
/// `__startup_64` / `__cpu_setup`); callers MUST gate the CAS on a
/// non-zero value so a stale `0` does not displace a previously
/// latched non-zero CR3.
#[cfg(target_arch = "x86_64")]
pub(crate) fn read_cr3(vcpu: &mut kvm_ioctls::VcpuFd) -> Option<u64> {
    vcpu.get_sregs().ok().map(|s| s.cr3)
}

#[cfg(target_arch = "aarch64")]
pub(crate) fn read_cr3(vcpu: &mut kvm_ioctls::VcpuFd) -> Option<u64> {
    // TTBR1_EL1 holds the kernel-half page-table base (matches the
    // `page_table_root` field in `VcpuRegSnapshot`). Same encoding
    // as `capture_vcpu_regs`: (Op0=3, Op1=0, CRn=2, CRm=0, Op2=1)
    // under the KVM_REG_ARM64_SYSREG namespace.
    const KVM_REG_ARM64: u64 = 0x6000_0000_0000_0000;
    const KVM_REG_SIZE_U64: u64 = 0x0030_0000_0000_0000;
    const KVM_REG_ARM64_SYSREG: u64 = 0x0013_0000;
    const TTBR1_EL1_ID: u64 = KVM_REG_ARM64 | KVM_REG_SIZE_U64 | KVM_REG_ARM64_SYSREG | 0xC101;
    let mut buf = [0u8; 8];
    vcpu.get_one_reg(TTBR1_EL1_ID, &mut buf)
        .ok()
        .map(|_| u64::from_le_bytes(buf))
}

impl std::fmt::Display for VcpuRegSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ip=0x{:016x} sp=0x{:016x} ptroot=0x{:016x}",
            self.instruction_pointer, self.stack_pointer, self.page_table_root
        )?;
        // user_page_table_root is x86_64-None / aarch64-Optional;
        // when present, render it inline so an aarch64 failure
        // dump shows both halves of the address space.
        if let Some(uptr) = self.user_page_table_root {
            write!(f, " uptroot=0x{uptr:016x}")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// aarch64 MMIO dispatch — serial and virtio over MMIO
// ---------------------------------------------------------------------------

/// Dispatch an MMIO write to serial and virtio devices.
///
/// aarch64 reboot is signalled via PSCI (`VcpuExit::SystemEvent`), not
/// MMIO, so there is no shutdown return distinct from the normal write
/// path: this function always handles the write and returns to the
/// run loop.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_mmio_write(
    com1: &PiMutex<console::Serial>,
    com2: &PiMutex<console::Serial>,
    virtio_con: Option<&PiMutex<virtio_console::VirtioConsole>>,
    virtio_blk: Option<&PiMutex<virtio_blk::VirtioBlk>>,
    virtio_net: Option<&PiMutex<virtio_net::VirtioNet>>,
    addr: u64,
    data: &[u8],
) {
    if let Some(offset) = mmio_serial_offset(addr, kvm::SERIAL_MMIO_BASE) {
        if let Some(&byte) = data.first() {
            com1.lock().inner_write(offset, byte);
        }
    } else if let Some(offset) = mmio_serial_offset(addr, kvm::SERIAL2_MMIO_BASE)
        && let Some(&byte) = data.first()
    {
        com2.lock().inner_write(offset, byte);
    } else if let Some(vc) = virtio_con
        && (kvm::VIRTIO_CONSOLE_MMIO_BASE
            ..kvm::VIRTIO_CONSOLE_MMIO_BASE + virtio_console::VIRTIO_MMIO_SIZE)
            .contains(&addr)
    {
        vc.lock()
            .mmio_write(addr - kvm::VIRTIO_CONSOLE_MMIO_BASE, data);
    } else if let Some(vb) = virtio_blk
        && (kvm::VIRTIO_BLK_MMIO_BASE..kvm::VIRTIO_BLK_MMIO_BASE + virtio_blk::VIRTIO_MMIO_SIZE)
            .contains(&addr)
    {
        vb.lock().mmio_write(addr - kvm::VIRTIO_BLK_MMIO_BASE, data);
    } else if let Some(vn) = virtio_net
        && (kvm::VIRTIO_NET_MMIO_BASE..kvm::VIRTIO_NET_MMIO_BASE + virtio_net::VIRTIO_MMIO_SIZE)
            .contains(&addr)
    {
        vn.lock().mmio_write(addr - kvm::VIRTIO_NET_MMIO_BASE, data);
    }
}

/// Dispatch an MMIO read from serial and virtio-console devices.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_mmio_read(
    com1: &PiMutex<console::Serial>,
    com2: &PiMutex<console::Serial>,
    virtio_con: Option<&PiMutex<virtio_console::VirtioConsole>>,
    virtio_blk: Option<&PiMutex<virtio_blk::VirtioBlk>>,
    virtio_net: Option<&PiMutex<virtio_net::VirtioNet>>,
    addr: u64,
    data: &mut [u8],
) {
    if let Some(offset) = mmio_serial_offset(addr, kvm::SERIAL_MMIO_BASE) {
        if let Some(first) = data.first_mut() {
            *first = com1.lock().inner_read(offset);
        }
    } else if let Some(offset) = mmio_serial_offset(addr, kvm::SERIAL2_MMIO_BASE) {
        if let Some(first) = data.first_mut() {
            *first = com2.lock().inner_read(offset);
        }
    } else if let Some(vc) = virtio_con
        && (kvm::VIRTIO_CONSOLE_MMIO_BASE
            ..kvm::VIRTIO_CONSOLE_MMIO_BASE + virtio_console::VIRTIO_MMIO_SIZE)
            .contains(&addr)
    {
        vc.lock()
            .mmio_read(addr - kvm::VIRTIO_CONSOLE_MMIO_BASE, data);
    } else if let Some(vb) = virtio_blk
        && (kvm::VIRTIO_BLK_MMIO_BASE..kvm::VIRTIO_BLK_MMIO_BASE + virtio_blk::VIRTIO_MMIO_SIZE)
            .contains(&addr)
    {
        vb.lock().mmio_read(addr - kvm::VIRTIO_BLK_MMIO_BASE, data);
    } else if let Some(vn) = virtio_net
        && (kvm::VIRTIO_NET_MMIO_BASE..kvm::VIRTIO_NET_MMIO_BASE + virtio_net::VIRTIO_MMIO_SIZE)
            .contains(&addr)
    {
        vn.lock().mmio_read(addr - kvm::VIRTIO_NET_MMIO_BASE, data);
    } else {
        for b in data.iter_mut() {
            *b = 0xff;
        }
    }
}

/// Compute register offset for an MMIO address within a serial region.
///
/// The serial MMIO region is page-sized (`kvm::SERIAL_MMIO_SIZE` =
/// 0x1000) so each UART sits on its own guest page, but the
/// underlying ns16550a register window is only 8 bytes wide
/// (DATA/IER/IIR/LCR/MCR/LSR/MSR/SCR at offsets 0..=7 per the
/// `vm-superio` `Serial::{read,write}` switch arms; out-of-range
/// offsets fall through to `_ => 0` / `_ => {}`). The legal-input
/// bound here is therefore the `u8` representable range, not the
/// page size: an offset above 0xFF cannot fit in `u8` and the cast
/// would silently wrap modulo 256 — a guest write to register 0x100
/// would land at register 0x00 (DATA), corrupting the TX path.
///
/// Tightening the upper bound to `base + 256` returns `None` for
/// the [0x100, SERIAL_MMIO_SIZE) sub-region, which falls through
/// to the next dispatch arm (and ultimately to the unmapped-MMIO
/// 0xFF fill on read / silent drop on write — the correct "no
/// device here" semantics for the unused tail of the page). The
/// kernel's 8250 driver only emits accesses to offsets 0..=7, so
/// no production guest hits the > 0x100 region; this gate is a
/// hostile-guest defense against truncation-induced register
/// aliasing.
#[cfg(target_arch = "aarch64")]
fn mmio_serial_offset(addr: u64, base: u64) -> Option<u8> {
    // Bound is `u8::MAX as u64 + 1` (= 256) so the `as u8` cast
    // is total within the kept range. The page-sized
    // `SERIAL_MMIO_SIZE` is intentionally NOT used as the bound;
    // see the rationale on this function.
    const MAX_REG_OFFSET: u64 = u8::MAX as u64 + 1;
    // Compile-time guarantee: the region we accept fits inside the
    // declared MMIO window. If a future change shrinks
    // `SERIAL_MMIO_SIZE` below 256, this constant breaks the
    // build instead of letting the function admit offsets that
    // step into a neighbouring device's region.
    const _: () = assert!(
        kvm::SERIAL_MMIO_SIZE >= MAX_REG_OFFSET,
        "SERIAL_MMIO_SIZE must cover at least the 256-byte u8-representable \
         register window mmio_serial_offset accepts"
    );
    if addr >= base && addr < base + MAX_REG_OFFSET {
        Some((addr - base) as u8)
    } else {
        None
    }
}

// -- watchpoint hit dispatch ------------------------------------------
//
// Shared between the AP (`vcpu_run_loop_unified`) and BSP
// (`run_bsp_loop`) paths. Identifies which watchpoint slot fired
// from the per-arch `kvm_debug_exit_arch` payload, gates the slot-0
// trigger on the post-store `exit_kind` value (so a clean
// SCX_EXIT_DONE does not generate a failure dump), and latches the
// matched user slot for the freeze coordinator's epoll loop.

/// `ESR_ELx_EC` decoded value for a watchpoint exception taken from a
/// lower exception level (the only EC that surfaces guest-side data
/// watchpoint hits to userspace via `KVM_EXIT_DEBUG`). Pinned per
/// `arch/arm64/include/asm/esr.h` `ESR_ELx_EC_WATCHPT_LOW = 0x34`.
///
/// `WATCHPT_CUR` (EC=0x35, watchpoint taken at the current EL) is
/// not handled because the kernel's `arm_exit_handlers` table in
/// `arch/arm64/kvm/handle_exit.c` (which routes guest exceptions to
/// userspace via `KVM_EXIT_DEBUG`) only registers a handler for
/// `ESR_ELx_EC_WATCHPT_LOW` — `WATCHPT_CUR` has no entry and is
/// therefore never surfaced to userspace. The guest runs at EL0/EL1
/// and KVM hosts at EL2; from KVM's perspective every guest-side
/// watchpoint trap is "from a lower EL" (LOW), and only EL2's own
/// debug traps would be CUR (which KVM does not arm).
#[cfg(target_arch = "aarch64")]
const ESR_ELX_EC_WATCHPT_LOW: u32 = 0x34;
/// `ESR_ELx_EC` decoded value for a software-step exception taken
/// from a lower exception level. KVM raises this through
/// `KVM_EXIT_DEBUG` after a `KVM_GUESTDBG_SINGLESTEP`-armed
/// `KVM_RUN` retires exactly one instruction (kernel
/// `arch/arm64/kvm/handle_exit.c::kvm_handle_guest_debug` switches
/// on `ESR_ELx_EC_SOFTSTP_LOW` and toggles `DBG_SPSR_SS`). We use
/// it to detect "the offending store has retired" after stepping
/// past a watchpoint trap, so the next `self_arm_watchpoint` call
/// can restore the slot's WCR.E=1. Pinned per
/// `arch/arm64/include/asm/esr.h` `ESR_ELx_EC_SOFTSTP_LOW = 0x32`.
#[cfg(target_arch = "aarch64")]
const ESR_ELX_EC_SOFTSTP_LOW: u32 = 0x32;
/// Bit shift of the `ESR_ELx_EC` field within the lower 32 bits of
/// the ESR_EL2 value KVM hands userspace as
/// `kvm_debug_exit_arch.hsr`. Pinned per `arch/arm64/include/asm/
/// esr.h` `ESR_ELx_EC_SHIFT = 26`.
#[cfg(target_arch = "aarch64")]
const ESR_ELX_EC_SHIFT: u32 = 26;
/// Mask applied to `(hsr >> ESR_ELx_EC_SHIFT)` to extract the
/// 6-bit EC field. Pinned per `ESR_ELx_EC(esr) = (esr & ESR_ELx_
/// EC_MASK) >> ESR_ELx_EC_SHIFT` in the same kernel header.
#[cfg(target_arch = "aarch64")]
const ESR_ELX_EC_MASK: u32 = 0x3F;

/// Dispatch a `KVM_EXIT_DEBUG` watchpoint trap to the matching slot's
/// latch. Reads the per-arch identifier (DR6 on x86_64; ESR EC + FAR
/// on aarch64), gates the slot-0 trigger on the post-store
/// `exit_kind` value, and writes the appropriate `hit` flag for the
/// freeze coordinator to observe.
///
/// `armed_slots` is the per-thread mirror of currently-armed KVAs
/// (one entry per slot) maintained by `self_arm_watchpoint`. On
/// aarch64 it carries the original (un-aligned) KVA so the
/// FAR range check covers the exact 4 bytes the watchpoint targets,
/// not the 8-byte block DBGWVR addresses. On x86_64 the entry is
/// also the requested KVA (DR0..DR3 hold full addresses) but is not
/// used by this helper — DR6 alone identifies which slot fired.
///
/// `single_step_pending` and `single_step_slot` are the per-vCPU
/// loop-local single-step bookkeeping the aarch64 path uses to step
/// past a fired watchpoint:
///
///   - On `EC = ESR_ELx_EC_WATCHPT_LOW (0x34)` with at least one
///     slot whose 4-byte FAR window contains the fault address,
///     after latching `hit` on every matched slot the helper sets
///     `*single_step_pending = true` and stores a 4-bit bitmap of
///     matched slot indices into `*single_step_slot` (bit i set
///     ⇒ slot i was matched). The next loop iteration's
///     `self_arm_watchpoint` call notices the flipped flag,
///     reissues KVM_SET_GUEST_DEBUG with WCR.E=0 on EVERY matched
///     slot (peer arms stay enabled), asserts
///     `KVM_GUESTDBG_SINGLESTEP`, and the following KVM_RUN
///     executes exactly one instruction past the offending store.
///     A 4-bit bitmap is sufficient because there are only four
///     hardware watchpoint slots; multiple matches happen when
///     `arm_user_watchpoint` placed two slots on overlapping
///     KVAs (no duplicate-rejection — see comment on the FAR
///     range loop).
///   - On `EC = ESR_ELx_EC_SOFTSTP_LOW (0x32)` with the flag set,
///     the helper clears `*single_step_pending` (without latching).
///     The next `self_arm_watchpoint` call restores WCR.E=1 on
///     every previously-disabled slot and drops the singlestep
///     bit. The mask in `*single_step_slot` is functionally not
///     consulted on this transition because the per-slot E
///     selector at `vcpu.rs::self_arm_watchpoint` short-circuits
///     on `single_step_pending == false`; restoring WCR.E
///     globally is correct because all slots have valid
///     `request_kva` published.
///
/// Both fields are inert on x86_64 — the x86 watchpoint trap is
/// taken AFTER the store retires (Intel SDM Vol. 3B 17.2.4
/// "Trap-class debug exceptions"), so re-entry advances normally
/// without the disable-step-rearm dance.
pub(crate) fn dispatch_watchpoint_hit(
    watchpoint: &WatchpointArm,
    debug_arch: &kvm_bindings::kvm_debug_exit_arch,
    armed_slots: &[u64; 4],
    single_step_pending: &mut bool,
    single_step_slot: &mut usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        // DR6 layout (Intel SDM Vol. 3B 17.2.5): bits 0-3 (B0..B3)
        // indicate which DR fired. Bit 14 (BS) signals single-step.
        // KVM populates `kvm_run.debug.arch.dr6` from the just-
        // fired exit's qualification field (`vmx_get_exit_qual`
        // in `arch/x86/kvm/vmx/vmx.c::handle_exception_nmi`),
        // not from the architectural DR6 register, so the bits we
        // see reflect ONLY the slots that fired on THIS exit —
        // not stale "sticky" bits from prior exits. The dedup
        // gate on `WatchpointArm::hit` (CAS in `latch_*`) handles
        // the cross-vCPU race where two vCPUs each fire on the
        // same slot before either has been processed by the
        // freeze coordinator: only the first false→true transition
        // wakes the coordinator's `hit_evt`.
        //
        // Single-step is aarch64-only — the x86 watchpoint trap
        // is taken AFTER the offending store retires (Intel SDM
        // Vol. 3B 17.2.4 "Trap-class debug exceptions"), so re-
        // entering KVM_RUN advances normally without the
        // disable-step-rearm dance. Consume the unused inputs to
        // keep the per-arch helper signature shared.
        let _ = armed_slots;
        let _ = (&mut *single_step_pending, &mut *single_step_slot);
        let dr6 = debug_arch.dr6;
        let trap_bits = (dr6 & 0xF) as u8;
        if trap_bits == 0 {
            // KVM exited via KVM_EXIT_DEBUG with no DR0..3 trap
            // bits set — possible when a single-step (BS, bit 14)
            // or task-switch (BT, bit 15) fired without a data/
            // code breakpoint match. ktstr never arms BS/BT, so
            // this is either a host-side debug stub leaking
            // through or a synthetic exit — log and ignore.
            // Mirrors the aarch64 "no FAR match" debug log so
            // both arches surface unexpected debug exits the
            // same way.
            tracing::debug!(
                dr6,
                "KVM_EXIT_DEBUG fired with no DR0..DR3 trap bit set \
                 (BS/BT or spurious); not latching"
            );
            return;
        }
        if trap_bits & 0x1 != 0 {
            latch_slot0_with_gate(watchpoint);
        }
        for idx in 0..3 {
            if trap_bits & (1u8 << (idx + 1)) != 0 {
                watchpoint.latch_user_hit(idx);
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // ARM debug exit payload (kernel
        // `arch/arm64/kvm/handle_exit.c::kvm_handle_guest_debug`):
        //   `hsr` = lower 32 bits of ESR_EL2
        //   `far` = FAR_EL2 (set only when ESR.EC == WATCHPT_LOW)
        // The EC field at bits [31:26] of ESR distinguishes
        // watchpoint exceptions from breakpoints / soft-step / BRK.
        let ec = (debug_arch.hsr >> ESR_ELX_EC_SHIFT) & ESR_ELX_EC_MASK;
        if ec == ESR_ELX_EC_SOFTSTP_LOW {
            // Software-step exception following a watchpoint hit.
            // The kernel sets cpsr.SS in
            // `kvm_handle_guest_debug` to advertise that exactly
            // one instruction retired since the prior fire; we
            // clear `single_step_pending` so the next
            // `self_arm_watchpoint` call restores the slot's
            // WCR.E=1 and drops `KVM_GUESTDBG_SINGLESTEP`. Do NOT
            // latch any `hit` flag here — the original
            // WATCHPT_LOW exit already latched the freeze
            // trigger; this exit only signals "one instruction
            // executed cleanly past the watched store".
            //
            // If the flag is NOT set we got a soft-step exit
            // we did not request (e.g. host kernel quirk, peer
            // tooling); log and ignore — there is no slot to
            // restore.
            if *single_step_pending {
                *single_step_pending = false;
                // Zero `single_step_slot` defensively. The
                // per-slot E selector in
                // `vcpu.rs::self_arm_watchpoint` already
                // short-circuits on `single_step_pending ==
                // false`, so a stale mask cannot disable a slot
                // — but a future regression that drops the
                // short-circuit would silently disable whatever
                // slots the stale mask still flags. Zeroing
                // here makes the post-step state purely
                // reflect "no slots pending step" so downstream
                // readers cannot trip on a leftover bitmap.
                *single_step_slot = 0;
            } else {
                tracing::debug!(
                    hsr = debug_arch.hsr,
                    "KVM_EXIT_DEBUG soft-step EC with no \
                     single-step pending; ignoring (likely \
                     spurious kernel-side step exit)"
                );
            }
            return;
        }
        if ec != ESR_ELX_EC_WATCHPT_LOW {
            tracing::debug!(
                hsr = debug_arch.hsr,
                ec,
                "KVM_EXIT_DEBUG with non-watchpoint EC; ignoring \
                 (breakpoint/BRK paths are not used by ktstr)"
            );
            return;
        }
        let far = debug_arch.far;
        // ARM ARM D2.10.5: FAR may be imprecise for unaligned
        // accesses. This exact-range check is correct for atomic_t
        // writes (aligned 4-byte stores via atomic_set/cmpxchg) but
        // would miss imprecise hits from unaligned multi-byte
        // accesses spanning the watched range.
        //
        // Range-match FAR against each armed slot's 4-byte
        // window. `armed_slots[i]` is the requested KVA (un-
        // aligned); the watch covers `[kva, kva + 4)`. Multiple
        // slots may match if their watched ranges overlap (e.g.
        // two `Op::WatchSnapshot` registrations on adjacent
        // 4-byte fields of the same struct word). `arm_user_
        // watchpoint` allocates by free-slot search and does NOT
        // reject duplicate KVAs, so overlapping arms are
        // possible. The loop iterates all four slots and latches
        // every match — overlapping arms are never silently
        // dropped.
        let mut matched_mask: u8 = 0;
        for (i, kva) in armed_slots.iter().enumerate() {
            if *kva == 0 {
                continue;
            }
            if far >= *kva && far < kva.saturating_add(4) {
                matched_mask |= 1 << i;
                if i == 0 {
                    latch_slot0_with_gate(watchpoint);
                } else {
                    watchpoint.latch_user_hit(i - 1);
                }
            }
        }
        if matched_mask == 0 {
            tracing::debug!(
                hsr = debug_arch.hsr,
                far,
                armed = ?armed_slots,
                "KVM_EXIT_DEBUG watchpoint fired but FAR matched no \
                 armed slot (possible KVM watchpoint match-distance \
                 fallback or stale arm); not latching"
            );
            return;
        }
        // The aarch64 watchpoint trap fires BEFORE the offending
        // store retires (ARM ARM D2.10.5: "the exception is
        // taken on the instruction that would have made the
        // access"). Re-entering KVM_RUN without intervention
        // replays the same store and re-trips the watchpoint
        // forever. Mirror the kernel's
        // `arch/arm64/kernel/hw_breakpoint.c::do_watchpoint`
        // recipe with a two-mechanism dance:
        //
        //   - `KVM_GUESTDBG_SINGLESTEP` (which sets MDSCR_EL1.SS
        //     in the kernel's `setup_external_mdscr`) is what
        //     causes KVM to retire EXACTLY ONE guest instruction
        //     and exit with `EC = ESR_ELx_EC_SOFTSTP_LOW (0x32)`.
        //     This advances the PC past the watched store. MDSCR.
        //     SS does NOT suppress watchpoint exceptions on the
        //     stepped instruction (per ARM ARM D2.12, software-
        //     step state still respects WCR.E for watchpoints
        //     that match the stepped access).
        //
        //   - Clearing WCR.E=0 on every matched slot is what
        //     prevents the watched store from re-tripping the
        //     watchpoint on the single-step pass. Without this,
        //     the same instruction that originally trapped would
        //     re-trap on its replay (the trap is taken BEFORE
        //     the access; the access has not retired yet).
        //
        // `single_step_slot` carries a 4-bit mask of every
        // matched slot index (bit i set ⇒ slot i must have its
        // WCR.E cleared during the single-step pass). Multiple
        // bits can be set: `arm_user_watchpoint` allocates by
        // free-slot search and does NOT reject duplicate KVAs,
        // so two slots may watch overlapping ranges and fire
        // simultaneously on the same store. `self_arm_watch
        // point` walks the mask and clears WCR.E on every set
        // bit; when `single_step_pending` clears on the
        // following SOFTSTP_LOW exit, all slots get WCR.E=1
        // restored.
        *single_step_pending = true;
        *single_step_slot = matched_mask as usize;
    }
}

/// Slot-0 latch with the post-store `exit_kind` value gate. Reads
/// the host pointer the freeze coordinator published, compares
/// against [`SCX_EXIT_ERROR_THRESHOLD`], and latches the failure-
/// trigger only on error-class transitions.
///
/// `kind_host_ptr` is guaranteed non-null when this helper runs.
/// The freeze coordinator publishes the pair in `kind_host_ptr →
/// request_kva` order with matching `Release` stores (see
/// `freeze_coord/watchpoint.rs::republish_watchpoint_on_rebind`, where
/// the err_exit publish issues the `kind_host_ptr` store BEFORE the
/// `request_kva`
/// store). [`super::vcpu::self_arm_watchpoint`] only programs the
/// hardware watchpoint after observing a non-zero `request_kva`
/// via an `Acquire` load — that load synchronises-with the
/// publisher's `Release`, which makes the prior `kind_host_ptr`
/// store visible too. Once armed, a fire reaches this helper only
/// when both stores are visible. The pointer is never invalidated
/// for the VM lifetime: `vm.guest_mem` (which backs the host
/// mapping) is dropped only after every vCPU thread has joined, so
/// the host-side mapping at this address strictly outlives every
/// reader of this pointer.
///
/// On aarch64 an `Acquire` fence pairs with the guest's store: by
/// the time KVM_RUN returns `KVM_EXIT_DEBUG` the trap-into-EL2 +
/// host-context-restore path has already issued an architectural
/// context-synchronization event (ERET, eret-to-EL1 from the
/// hypervisor save/restore), but Rust's memory model does not
/// know about those. The fence makes the `read_volatile` of the
/// host pointer ordered-after that synchronization in Rust's
/// happens-before graph, matching what the x86_64 path gets for
/// free from TSO.
///
/// A null observation here would be a publication-invariant
/// violation; we still check at runtime so a regression in the
/// publisher cannot be turned into a `read_volatile` of a null
/// pointer (UB). The check costs one branch on the cold debug
/// trap path — negligible — and surfaces the invariant break as
/// a `tracing::error!` instead of crashing the run.
fn latch_slot0_with_gate(watchpoint: &WatchpointArm) {
    let host_ptr = watchpoint.kind_host_ptr.load(Ordering::Acquire);
    if host_ptr.is_null() {
        tracing::error!(
            "latch_slot0_with_gate: kind_host_ptr null at fire time — \
             publication invariant broken (request_kva non-zero must \
             imply kind_host_ptr non-null per the Release-store \
             ordering in \
             freeze_coord/watchpoint.rs::republish_watchpoint_on_rebind). \
             Skipping \
             slot-0 latch; the BPF .bss late-trigger fallback in the \
             freeze coordinator's poll loop remains active."
        );
        return;
    }
    // Publish ordering: the guest's store is globally visible by
    // the time KVM exits to userspace, but Rust's memory model
    // requires an explicit Acquire fence on weakly-ordered hosts
    // (aarch64) to make the host-pointer read happens-after the
    // guest store in the Rust abstract machine. On x86_64 TSO
    // gives us this for free; the explicit fence is a no-op
    // codegen-wise but keeps the operation ordered in std::sync
    // terms across both arches.
    std::sync::atomic::fence(Ordering::Acquire);
    // SAFETY: `kind_host_ptr` was published by the freeze
    // coordinator before `request_kva` (Release), and the
    // `request_kva` non-zero load that triggered the arm is the
    // synchronizes-with edge for this read. The pointer addresses
    // a u32 inside the guest's `scx_sched` slab page, which stays
    // mapped for the VM lifetime per the `ReservationGuard`
    // contract. Non-null per the `is_null` early-return above.
    let kind = unsafe { std::ptr::read_volatile(host_ptr) };
    if kind >= SCX_EXIT_ERROR_THRESHOLD {
        watchpoint.latch_hit();
    } else {
        tracing::debug!(
            kind,
            threshold = SCX_EXIT_ERROR_THRESHOLD,
            "watchpoint fired on non-error exit_kind transition \
             (e.g. SCX_EXIT_DONE on clean shutdown); skipping \
             freeze trigger"
        );
    }
}

/// Unified per-vCPU KVM_RUN loop for AP threads.
///
/// HLT on APs: check kill + continue on both arches (KVM delivers
/// interrupts to wake the vCPU). Shutdown sets the kill flag so all
/// other vCPUs exit.
///
/// `watchpoint` carries the failure-dump trigger contract: each
/// iteration polls `watchpoint.request_kva` and self-arms a hardware
/// data-write watchpoint on `*scx_root->exit_kind` once the freeze
/// coordinator has resolved its KVA. When the kernel later writes
/// the field, KVM exits via `VcpuExit::Debug`; this loop sets
/// `watchpoint.hit` so the freeze coordinator's late-trigger poll
/// fires immediately. The arm is one-shot per KVA value (the
/// per-vCPU `armed_kva` slot suppresses re-arms after the ioctl
/// lands).
///
/// Freeze handling: when the freeze flag is set, the vCPU thread
/// performs the Cloud Hypervisor pause/snapshot drain dance
/// (set_immediate_exit(1) → vcpu.run() → set_immediate_exit(0)) so
/// any in-flight PIO/MMIO operation completes inside the KVM_RUN
/// ioctl before the thread parks. The drain is necessary because
/// KVM_EXIT_IO/MMIO leave the operation only partially complete on
/// the kernel side; userspace must re-enter KVM_RUN to commit it.
/// After draining, the thread sets `parked=true` (Release-ordered so
/// the host's subsequent guest-memory reads happen-after the
/// drain), then polls freeze on park_timeout. The Acquire load on
/// `parked` from the freeze coordinator IS the memory barrier that
/// makes external-thread guest-memory reads correct on weakly
/// ordered architectures (matches Cloud Hypervisor's pause
/// pattern). The kick that triggers freeze observation uses
/// Firecracker's SIGRTMIN+immediate_exit pattern, but the drain
/// dance itself is Cloud Hypervisor-specific.
#[allow(clippy::too_many_arguments)]
pub(crate) fn vcpu_run_loop_unified(
    vcpu: &mut kvm_ioctls::VcpuFd,
    com1: &Arc<PiMutex<console::Serial>>,
    com2: &Arc<PiMutex<console::Serial>>,
    virtio_con: Option<&Arc<PiMutex<virtio_console::VirtioConsole>>>,
    virtio_blk: Option<&Arc<PiMutex<virtio_blk::VirtioBlk>>>,
    virtio_net: Option<&Arc<PiMutex<virtio_net::VirtioNet>>>,
    ioapic: Option<&Arc<IoapicHandle>>,
    pci_bus: Option<&Arc<PiMutex<pci::PciBus>>>,
    kill: &Arc<AtomicBool>,
    kill_evt: &Arc<EventFd>,
    freeze: &Arc<AtomicBool>,
    parked: &Arc<AtomicBool>,
    regs_slot: &Arc<std::sync::Mutex<Option<VcpuRegSnapshot>>>,
    watchpoint: &Arc<WatchpointArm>,
    has_immediate_exit: bool,
    parked_evt: Option<&Arc<EventFd>>,
    thaw_evt: Option<&Arc<EventFd>>,
) {
    // Per-AP `armed_slots` mirrors the BSP-side slot array in
    // `freeze_coord::run_bsp_loop`. Index 0 = DR0 (err_exit watchpoint
    // for `*scx_root->exit_kind`); indices 1..=3 = DR1/DR2/DR3 (user
    // `Op::WatchSnapshot` arms). All start at `0` until the freeze
    // coordinator publishes resolved KVAs. The array is a per-thread
    // local so the per-iteration arm check is four Acquire loads
    // with no cross-thread synchronization beyond the published
    // requests. `arm_failures` counts consecutive non-EINTR ioctl
    // failures; EINTR is transient (SIGRTMIN kick race) and does NOT
    // increment, so a kicked-mid-arm vCPU retries instead of
    // permanently disabling the watchpoint.
    let mut armed_slots: [u64; 4] = [0; 4];
    let mut arm_failures: u8 = 0;
    // aarch64 watchpoint single-step bookkeeping. On aarch64 the
    // hardware watchpoint trap is taken BEFORE the offending store
    // retires (ARM ARM D2.10.5: "the exception is taken on the
    // instruction that would have made the access"), so re-entering
    // KVM_RUN replays the same instruction and re-trips the
    // watchpoint forever. Mirroring the kernel's
    // `arch/arm64/kernel/hw_breakpoint.c` recipe, we disable the
    // fired slot's WCR.E and assert KVM_GUESTDBG_SINGLESTEP for
    // exactly one KVM_RUN; the next KVM_EXIT_DEBUG carries
    // EC=ESR_ELx_EC_SOFTSTP_LOW (0x32), at which point we clear
    // `single_step_pending` and `self_arm_watchpoint` reissues
    // KVM_SET_GUEST_DEBUG with WCR.E restored to 1 and
    // KVM_GUESTDBG_SINGLESTEP cleared. Inert on x86_64 (the trap
    // there is taken AFTER the store, so re-entry advances
    // normally); the locals still pass through to keep the
    // per-arch helper signatures shared.
    let mut single_step_pending: bool = false;
    let mut single_step_slot: usize = 0;
    let mut armed_single_step: bool = false;
    loop {
        if kill.load(Ordering::Acquire) {
            break;
        }
        // Honour a pending freeze before re-entering KVM_RUN.
        if freeze.load(Ordering::Acquire) {
            handle_freeze(
                vcpu,
                has_immediate_exit,
                kill,
                freeze,
                parked,
                regs_slot,
                parked_evt.map(|a| a.as_ref()),
                thaw_evt.map(|a| a.as_ref()),
                Some(kill_evt.as_ref()),
            );
            if kill.load(Ordering::Acquire) {
                break;
            }
        }
        // Self-arm the failure-dump watchpoint when the coordinator
        // publishes (or republishes) a request KVA. Cheap (atomic
        // load + compare against `armed_kva`) when no new arm is
        // pending. Mirrors `run_bsp_loop`'s arm-before-run pattern;
        // both paths share `WatchpointArm` so a fire on either
        // triggers the late-snapshot rendezvous. Also drives the
        // aarch64 watchpoint single-step transition: when
        // `single_step_pending` is set by the prior watchpoint
        // exit, this call reissues KVM_SET_GUEST_DEBUG with the
        // fired slot's WCR.E cleared and KVM_GUESTDBG_SINGLESTEP
        // asserted; when the SOFTSTP_LOW exit clears the flag, the
        // next call restores WCR.E=1 and drops the singlestep bit.
        self_arm_watchpoint(
            vcpu,
            watchpoint,
            &mut armed_slots,
            &mut arm_failures,
            single_step_pending,
            single_step_slot,
            &mut armed_single_step,
        );

        match vcpu.run() {
            Ok(mut exit) => {
                if matches!(exit, VcpuExit::Hlt) {
                    if kill.load(Ordering::Acquire) {
                        break;
                    }
                    continue;
                }
                // KVM_EXIT_DEBUG fires when the armed hardware
                // data-write watchpoint trips on a guest write to
                // `*scx_root->exit_kind`. The kernel writes the
                // field on BOTH error transitions
                // (`scx_error -> SCX_EXIT_ERROR/_BPF/_STALL >=
                // 1024`) AND clean shutdown
                // (`scx_unregister -> SCX_EXIT_DONE = 1`). Only the
                // error transitions should trigger the failure-dump
                // freeze; firing on every clean test exit is a
                // regression. Read the post-store value from the
                // host pointer the coordinator published and gate
                // `hit` on the error threshold. The watchpoint is
                // left armed regardless: the coordinator's freeze +
                // thaw is synchronous with the dump emission, and a
                // future error after a clean transition would still
                // fire (slab page lifetime — the scheduler's
                // `scx_sched` is not freed until well after the
                // last `exit_kind` write).
                if let VcpuExit::Debug(debug_arch) = &exit {
                    dispatch_watchpoint_hit(
                        watchpoint,
                        debug_arch,
                        &armed_slots,
                        &mut single_step_pending,
                        &mut single_step_slot,
                    );
                    if kill.load(Ordering::Acquire) {
                        break;
                    }
                    continue;
                }
                match classify_exit(
                    com1,
                    com2,
                    virtio_con.map(|a| a.as_ref()),
                    virtio_blk.map(|a| a.as_ref()),
                    virtio_net.map(|a| a.as_ref()),
                    ioapic.map(|a| a.as_ref()),
                    pci_bus.map(|a| a.as_ref()),
                    &mut exit,
                ) {
                    Some(ExitAction::Continue) | None => {}
                    Some(ExitAction::Shutdown) => {
                        kill.store(true, Ordering::Release);
                        // Wake the freeze coordinator's epoll loop
                        // so it sees the kill flag without waiting
                        // up to one full epoll timeout. Failure
                        // (EAGAIN under EFD_NONBLOCK from a
                        // saturated counter) is benign — any prior
                        // pending edge already wakes the coord, and
                        // the AtomicBool above remains the source
                        // of truth.
                        let _ = kill_evt.write(1);
                        break;
                    }
                    Some(ExitAction::Fatal(_)) => {
                        // AP fatal exit (FailEntry / InternalError):
                        // surface in tracing AND propagate the kill
                        // signal. Without `kill.store(true)` and the
                        // kill_evt write, the AP thread silently
                        // exits while peer vCPUs and the freeze
                        // coordinator stay running — peers eventually
                        // hit FREEZE_RENDEZVOUS_TIMEOUT instead of
                        // shutting down promptly. Mirrors the
                        // Shutdown arm's kill-propagation pattern.
                        tracing::error!("AP fatal exit");
                        kill.store(true, Ordering::Release);
                        let _ = kill_evt.write(1);
                        break;
                    }
                }
            }
            Err(e) => {
                if e.errno() == libc::EINTR || e.errno() == libc::EAGAIN {
                    vcpu.set_kvm_immediate_exit(0);
                    if kill.load(Ordering::Acquire) {
                        break;
                    }
                    continue;
                }
                if kill.load(Ordering::Acquire) {
                    break;
                }
            }
        }

        if kill.load(Ordering::Acquire) {
            break;
        }
    }
}

/// Drain pending PIO/MMIO state and park the vCPU until freeze
/// clears. Called from the run loop when the freeze flag is observed,
/// and from `super::freeze_coord::run_bsp_loop` for the same purpose
/// on the BSP thread.
///
/// The drain dance — `set_immediate_exit(1) → vcpu.run() →
/// set_immediate_exit(0)` — is the Cloud Hypervisor pause/snapshot
/// pattern for completing in-flight I/O before pausing. KVM_RUN with
/// immediate_exit=1 returns -EINTR without entering the guest but
/// still commits any pending PIO/MMIO state from the previous exit
/// (per the KVM API contract: pending I/O is committed at the start
/// of KVM_RUN even when immediate_exit prevents guest entry).
/// `has_immediate_exit` gates the dance — without
/// KVM_CAP_IMMEDIATE_EXIT, calling `vcpu.run()` here would re-enter
/// the guest instead of returning EINTR, so the drain step is
/// skipped on kernels that lack the cap. The freeze rendezvous
/// itself still works (set parked, await thaw); only the I/O drain
/// is skipped.
///
/// After the drain, the thread sets `parked=true` with Release
/// ordering and polls freeze on `park_timeout(10ms)` until the
/// coordinator clears it. The thaw path uses no explicit unpark —
/// the 10ms park_timeout cadence picks up the cleared freeze flag
/// within at most 10 ms, which is well below the dump latency
/// budget.
///
/// `kill` is honoured throughout: a shutdown signal during the park
/// loop wins over freeze and the function returns to the caller's
/// kill-check at the top of the loop.
#[allow(clippy::too_many_arguments)]
pub(crate) fn handle_freeze(
    vcpu: &mut kvm_ioctls::VcpuFd,
    has_immediate_exit: bool,
    kill: &Arc<AtomicBool>,
    freeze: &Arc<AtomicBool>,
    parked: &Arc<AtomicBool>,
    regs_slot: &Arc<std::sync::Mutex<Option<VcpuRegSnapshot>>>,
    parked_evt: Option<&EventFd>,
    thaw_evt: Option<&EventFd>,
    kill_evt: Option<&EventFd>,
) {
    // Drain dance: complete any pending PIO/MMIO before parking.
    // Skipped on kernels without KVM_CAP_IMMEDIATE_EXIT, where
    // calling vcpu.run() with the cap absent would re-enter the
    // guest instead of returning EINTR.
    if has_immediate_exit {
        vcpu.set_kvm_immediate_exit(1);
        // Drain dance: KVM_RUN with immediate_exit=1 commits any
        // pending PIO/MMIO from the prior exit and returns EINTR
        // without entering the guest (per the KVM API contract). EINTR
        // is the expected outcome; any other error means KVM rejected
        // the ioctl (e.g. KVM_RUN unsupported state, vCPU corruption)
        // and the freeze coordinator's subsequent guest-memory reads
        // may observe partial state. Log non-EINTR explicitly so a
        // real KVM regression is not silently swallowed.
        if let Err(e) = vcpu.run()
            && e.errno() != libc::EINTR
        {
            tracing::warn!(
                err = %e,
                "handle_freeze: drain KVM_RUN failed with non-EINTR — \
                 pending PIO/MMIO may not have committed before park"
            );
        }
        vcpu.set_kvm_immediate_exit(0);
    }

    // Capture vCPU registers BEFORE the Release store on `parked`.
    // KVM_GET_REGS / KVM_GET_SREGS are vCPU-fd-bound ioctls — they
    // must run on the vCPU thread (not cross-thread from the
    // coordinator). Capturing here means the regs slot's Mutex
    // store is happens-before the coordinator's Acquire on
    // `parked`, so the coordinator can read the slot via the same
    // synchronizes-with edge that makes its guest-memory reads
    // correct. A failed capture stores `None`; the dump shows
    // "registers unavailable" rather than panicking the freeze.
    let snapshot = capture_vcpu_regs(vcpu);
    *regs_slot.lock_unpoisoned() = snapshot;

    // Acknowledge frozen state. The Release store synchronizes-with
    // the coordinator's Acquire load on `parked`, providing the
    // happens-before edge that makes the coordinator's subsequent
    // guest-memory reads correct.
    parked.store(true, Ordering::Release);

    // Wake the freeze coordinator's rendezvous wait — write to the
    // shared `parked_evt` AFTER the Release store on `parked`. The
    // coordinator drains the eventfd once and then re-checks every
    // vCPU's `parked` flag plus the worker's `paused` flag. The
    // ordering is load-bearing: the coordinator's Acquire load on
    // `parked` happens-after this Release, so its subsequent
    // guest-memory reads observe every queue mutation the vCPU
    // performed before the drain dance.
    //
    // EAGAIN under EFD_NONBLOCK from a saturated counter is benign:
    // the AtomicBool is the source of truth, and any prior pending
    // edge already wakes the coordinator. Other errnos (EBADF,
    // EINVAL) signal real eventfd breakage and warrant a higher-
    // severity log than the benign-saturation case. Either way we do
    // not propagate — the parked AtomicBool is the source of truth
    // for the freeze rendezvous, and the coordinator's epoll-wait
    // backstop bounds wake latency without the eventfd edge.
    if let Some(evt) = parked_evt
        && let Err(e) = evt.write(1)
    {
        if e.raw_os_error() == Some(libc::EAGAIN) {
            tracing::debug!(
                err = %e,
                "handle_freeze: parked_evt write returned EAGAIN \
                 (eventfd counter saturated; benign — coordinator \
                 already has a pending wake edge)"
            );
        } else {
            tracing::warn!(
                err = %e,
                "handle_freeze: parked_evt write failed with non-EAGAIN \
                 errno — eventfd may be broken; freeze coordinator wake \
                 falls back to epoll backstop"
            );
        }
    }

    // Park until freeze clears or shutdown wins. The thaw_evt
    // is written by the freeze coordinator alongside
    // `freeze.store(false, Release)`; poll on [thaw_evt, kill_evt]
    // with a 100 ms backstop so a missed eventfd write (counter
    // overflow / EAGAIN) still drops the parked vCPU within
    // bounded latency. Without the thaw_evt the legacy
    // park_timeout(10 ms) cadence applies as the only source of
    // wake.
    use std::os::fd::AsRawFd;
    while freeze.load(Ordering::Acquire) {
        if kill.load(Ordering::Acquire) {
            break;
        }
        match (thaw_evt, kill_evt) {
            (Some(thaw), kev) => {
                let mut pfds = [
                    libc::pollfd {
                        fd: thaw.as_raw_fd(),
                        events: libc::POLLIN,
                        revents: 0,
                    },
                    libc::pollfd {
                        fd: kev.map_or(-1, |k| k.as_raw_fd()),
                        events: libc::POLLIN,
                        revents: 0,
                    },
                ];
                let nfds = if kev.is_some() { 2 } else { 1 };
                unsafe {
                    libc::poll(pfds.as_mut_ptr(), nfds as libc::nfds_t, 100);
                }
                // Do NOT drain the shared `thaw_evt` here. The
                // coordinator writes to thaw_evt ONCE per thaw and
                // every parked AP polls the SAME fd; if the first
                // wake-winner drains the counter, every other AP's
                // poll blocks for the full 100 ms backstop instead
                // of waking immediately. Leaving the eventfd level
                // high means poll returns immediately for every AP
                // — fast thaw across all peers. The `freeze.load
                // (Acquire)` re-check at the top of the loop is the
                // source of truth: once `freeze` clears the loop
                // exits regardless of eventfd state.
            }
            (None, _) => {
                // No thaw_evt plumbed (e.g. interactive shell path
                // that doesn't run a freeze coordinator). Fall back
                // to the legacy park_timeout cadence — the freeze
                // flag will never be set in that path so this
                // branch is structurally unreachable for real
                // shutdowns, but the safe-by-construction fallback
                // keeps the function callable with all None.
                std::thread::park_timeout(std::time::Duration::from_millis(10));
            }
        }
    }

    // Resume: clear parked so subsequent freeze cycles are observable.
    parked.store(false, Ordering::Release);
}

// ---------------------------------------------------------------------------
// I/O dispatch — shared between BSP and AP run loops
// ---------------------------------------------------------------------------

const KVM_SYSTEM_EVENT_SHUTDOWN: u32 = 1;
const KVM_SYSTEM_EVENT_RESET: u32 = 2;

/// Classified vCPU exit action from `classify_exit`.
pub(crate) enum ExitAction {
    /// Continue running (I/O handled, etc.).
    Continue,
    /// Clean shutdown (system reset, VcpuExit::Shutdown, etc.).
    Shutdown,
    /// Fatal error. `Some(reason)` for FailEntry, `None` for InternalError.
    Fatal(Option<u64>),
}

/// Classify a VcpuExit into an ExitAction, dispatching arch-specific I/O.
///
/// Returns `None` for HLT (caller handles: check kill flag, continue).
/// Takes the exit by mutable reference so IoIn/MmioRead data buffers
/// can be written back.
///
/// On aarch64, serial and virtio-console are dispatched via MMIO.
/// On x86_64, serial is dispatched via port I/O; virtio-console via MMIO.
#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(target_arch = "x86_64"), allow(unused_variables))]
pub(crate) fn classify_exit(
    com1: &PiMutex<console::Serial>,
    com2: &PiMutex<console::Serial>,
    virtio_con: Option<&PiMutex<virtio_console::VirtioConsole>>,
    virtio_blk: Option<&PiMutex<virtio_blk::VirtioBlk>>,
    virtio_net: Option<&PiMutex<virtio_net::VirtioNet>>,
    ioapic: Option<&IoapicHandle>,
    pci_bus: Option<&PiMutex<pci::PciBus>>,
    exit: &mut VcpuExit,
) -> Option<ExitAction> {
    match exit {
        #[cfg(target_arch = "x86_64")]
        VcpuExit::IoOut(port, data) => {
            if dispatch_io_out(com1, com2, pci_bus, *port, data) {
                Some(ExitAction::Shutdown)
            } else {
                Some(ExitAction::Continue)
            }
        }
        #[cfg(target_arch = "x86_64")]
        VcpuExit::IoIn(port, data) => {
            dispatch_io_in(com1, com2, pci_bus, *port, data);
            Some(ExitAction::Continue)
        }
        #[cfg(target_arch = "aarch64")]
        VcpuExit::MmioWrite(addr, data) => {
            // aarch64 has no MMIO-side shutdown signal — guest reboot
            // arrives as VcpuExit::SystemEvent (PSCI), handled below.
            dispatch_mmio_write(com1, com2, virtio_con, virtio_blk, virtio_net, *addr, data);
            Some(ExitAction::Continue)
        }
        #[cfg(target_arch = "aarch64")]
        VcpuExit::MmioRead(addr, data) => {
            dispatch_mmio_read(com1, com2, virtio_con, virtio_blk, virtio_net, *addr, data);
            Some(ExitAction::Continue)
        }
        VcpuExit::Hlt => None,
        VcpuExit::Shutdown => Some(ExitAction::Shutdown),
        VcpuExit::SystemEvent(event_type, _) => {
            if *event_type == KVM_SYSTEM_EVENT_SHUTDOWN || *event_type == KVM_SYSTEM_EVENT_RESET {
                Some(ExitAction::Shutdown)
            } else {
                Some(ExitAction::Continue)
            }
        }
        VcpuExit::FailEntry(reason, _cpu) => Some(ExitAction::Fatal(Some(*reason))),
        VcpuExit::InternalError => Some(ExitAction::Fatal(None)),
        #[cfg(target_arch = "x86_64")]
        VcpuExit::MmioRead(addr, data) => {
            if let Some(vc) = virtio_con {
                let base = kvm::VIRTIO_CONSOLE_MMIO_BASE;
                if *addr >= base && *addr < base + virtio_console::VIRTIO_MMIO_SIZE {
                    vc.lock().mmio_read(*addr - base, data);
                    return Some(ExitAction::Continue);
                }
            }
            if let Some(vb) = virtio_blk {
                let base = kvm::VIRTIO_BLK_MMIO_BASE;
                if *addr >= base && *addr < base + virtio_blk::VIRTIO_MMIO_SIZE {
                    vb.lock().mmio_read(*addr - base, data);
                    return Some(ExitAction::Continue);
                }
            }
            if let Some(vn) = virtio_net {
                let base = kvm::VIRTIO_NET_MMIO_BASE;
                if *addr >= base && *addr < base + virtio_net::VIRTIO_MMIO_SIZE {
                    vn.lock().mmio_read(*addr - base, data);
                    return Some(ExitAction::Continue);
                }
            }
            // PCI BAR windows (virtio-pci device register regions:
            // common/notify/isr/device cfg). The guest drives a PCI
            // function here after programming BAR0 + COMMAND.MEMORY; on
            // x86 this is the NIC's transport (the MMIO net arm above is
            // inert — the caller passes `virtio_net: None` when the NIC
            // is on PCI). Hot path, so it precedes the cold ECAM range.
            // The read takes a mutable bus lock because the virtio ISR
            // region is read-to-clear.
            if let Some(bus) = pci_bus {
                let mut guard = bus.lock();
                if guard.bar_mmio_contains(*addr) {
                    guard.bar_mmio_read(*addr, data);
                    return Some(ExitAction::Continue);
                }
            }
            // PCI ECAM (extended config space, registers 0..4095). Cold: the
            // guest reads config space during enumeration. Checked after the
            // hot virtio ranges.
            if let Some(bus) = pci_bus {
                let guard = bus.lock();
                if guard.ecam_contains(*addr) {
                    guard.ecam_read(*addr, data);
                    return Some(ExitAction::Continue);
                }
            }
            // Userspace IOAPIC (split-irqchip). Cold: the guest reads the
            // redirection table only during IRQ setup. Checked after the hot
            // virtio ranges so a virtio MMIO exit never reaches here.
            if let Some(io) = ioapic
                && let Some(off) = io.in_range(*addr)
            {
                io.mmio_read(off, data);
                return Some(ExitAction::Continue);
            }
            for b in data.iter_mut() {
                *b = 0xff;
            }
            Some(ExitAction::Continue)
        }
        #[cfg(target_arch = "x86_64")]
        VcpuExit::MmioWrite(addr, data) => {
            if let Some(vc) = virtio_con {
                let base = kvm::VIRTIO_CONSOLE_MMIO_BASE;
                if *addr >= base && *addr < base + virtio_console::VIRTIO_MMIO_SIZE {
                    vc.lock().mmio_write(*addr - base, data);
                    return Some(ExitAction::Continue);
                }
            }
            if let Some(vb) = virtio_blk {
                let base = kvm::VIRTIO_BLK_MMIO_BASE;
                if *addr >= base && *addr < base + virtio_blk::VIRTIO_MMIO_SIZE {
                    vb.lock().mmio_write(*addr - base, data);
                    return Some(ExitAction::Continue);
                }
            }
            if let Some(vn) = virtio_net {
                let base = kvm::VIRTIO_NET_MMIO_BASE;
                if *addr >= base && *addr < base + virtio_net::VIRTIO_MMIO_SIZE {
                    vn.lock().mmio_write(*addr - base, data);
                    return Some(ExitAction::Continue);
                }
            }
            // PCI BAR windows (virtio-pci device register regions). The
            // guest drives a PCI function here (notify/queue kicks land in
            // the NOTIFY region); on x86 this is the NIC's transport (the
            // MMIO net arm above is inert when the NIC is on PCI). Hot
            // path, so it precedes the cold ECAM range.
            if let Some(bus) = pci_bus {
                let mut guard = bus.lock();
                if guard.bar_mmio_contains(*addr) {
                    guard.bar_mmio_write(*addr, data);
                    return Some(ExitAction::Continue);
                }
            }
            // PCI ECAM (extended config space). Cold: the guest writes config
            // space during enumeration (BAR sizing, COMMAND). Checked after
            // the hot virtio ranges.
            if let Some(bus) = pci_bus {
                let mut guard = bus.lock();
                if guard.ecam_contains(*addr) {
                    guard.ecam_write(*addr, data);
                    return Some(ExitAction::Continue);
                }
            }
            // Userspace IOAPIC (split-irqchip), checked after the hot virtio
            // ranges. An RTE write rebuilds the MSI routing table (cold path).
            if let Some(io) = ioapic
                && let Some(off) = io.in_range(*addr)
            {
                if let Err(e) = io.mmio_write(off, data) {
                    // A failed KVM_SET_GSI_ROUTING leaves the guest's
                    // just-programmed device IRQ unrouted — it will not
                    // deliver and the device hangs on first use. Loud +
                    // counted (surfaced at teardown via routing_failures())
                    // so a hung-device test reports the cause instead of an
                    // opaque timeout.
                    tracing::error!(
                        count = io.routing_failures(),
                        "ioapic: KVM_SET_GSI_ROUTING failed: {e:#}"
                    );
                }
                return Some(ExitAction::Continue);
            }
            Some(ExitAction::Continue)
        }
        #[cfg(target_arch = "x86_64")]
        VcpuExit::IoapicEoi(vector) => {
            // Split-irqchip level-EOI exit. Edge device pins (v0) never raise
            // it; serviced defensively so a level entry's remote-IRR is
            // cleared rather than wedging the line.
            if let Some(io) = ioapic {
                io.eoi(*vector);
            }
            Some(ExitAction::Continue)
        }
        _ => None,
    }
}

/// I8042 ports and commands — minimal emulation for x86 guest reboot.
/// The kernel's default reboot method (`reboot=k`) writes CMD_RESET_CPU
/// (0xFE) to the i8042 command port (0x64).
#[cfg(target_arch = "x86_64")]
const I8042_DATA_PORT: u16 = 0x60;
#[cfg(target_arch = "x86_64")]
const I8042_CMD_PORT: u16 = 0x64;
#[cfg(target_arch = "x86_64")]
const I8042_CMD_RESET_CPU: u8 = 0xFE;

/// PCI type-1 configuration I/O ports (x86): CONFIG_ADDRESS latch (0xCF8) and
/// the CONFIG_DATA window (0xCFC..0xD00). The guest's `pci_direct_conf1`
/// drives these for all base config (registers 0..255).
#[cfg(target_arch = "x86_64")]
const PCI_CONFIG_ADDRESS: u16 = 0xCF8;
#[cfg(target_arch = "x86_64")]
const PCI_CONFIG_DATA: u16 = 0xCFC;

/// The ACPI PM timer's current 24-bit value (3.579545 MHz, ~279 ns/tick),
/// derived from the host monotonic clock so it advances. ktstr has no real
/// PM hardware — the guest's clocksource is kvm-clock; this counter exists
/// only so the FADT-advertised PM timer reads as a moving value (a stuck
/// value trips the guest's clocksource watchdog).
///
/// vCPU-thread budget: this runs on the vCPU thread (a guest IN from the
/// PM_TMR port), but `clock_gettime(CLOCK_MONOTONIC)` resolves via the vDSO
/// (no syscall entry / kernel transition), so its worst-case latency is
/// sub-microsecond — it cannot approach the freeze-rendezvous timeout and
/// does not delay SIGRTMIN delivery.
#[cfg(target_arch = "x86_64")]
fn acpi_pm_timer_value() -> u32 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `clock_gettime` writes only `ts` (a valid out-param) and reads
    // no caller memory; CLOCK_MONOTONIC is always available on Linux.
    unsafe {
        libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts);
    }
    let nanos = (ts.tv_sec as u64)
        .wrapping_mul(1_000_000_000)
        .wrapping_add(ts.tv_nsec as u64);
    // 279 is a deliberate integer approximation of the 279.365 ns/tick period
    // (3.579545 MHz) — the synthesized counter runs ~0.13% fast, accepted
    // because it is liveness-only (the guest's clocksource watchdog wants a
    // moving value; kvm-clock, not this, is the timekeeping source). The
    // 24-bit mask matches the FADT leaving TMR_VAL_EXT clear (32-bit register,
    // 24-bit counter). Because the value is recomputed from host-monotonic time
    // on every read (not a stored counter), it ADVANCES across a freeze window
    // — host time keeps running while vCPUs are paused — unlike kvm-clock, which
    // is save/restored across freeze. Benign: acpi_pm is never the selected
    // clocksource (rating-outranked by kvm-clock) and the only consumer is the
    // watchdog, which wants forward motion, not freeze-consistency — and vCPUs
    // are paused for the whole freeze window, so no watchdog comparison ever runs
    // inside it to observe the host-monotonic advance.
    ((nanos / 279) & 0x00FF_FFFF) as u32
}

/// True when `port` is one of the three ACPI PM register-block ports the FADT
/// advertises: PM1a_EVT (0x600, len 4) and PM1a_CNT (0x604, len 2) are
/// contiguous (0x600..0x606); PM_TMR (0x608, len 4) is disjoint
/// (0x608..0x60C). The 0x606-0x607 gap between CNT and TMR is NOT advertised
/// (PM1_CNT_LEN=2), so it is deliberately excluded — the guard claims exactly
/// the advertised register footprint, leaving the gap free for any future I/O
/// resource rather than shadowing it behind the PM stub.
#[cfg(target_arch = "x86_64")]
fn acpi_pm_port(port: u16) -> bool {
    (kvm::ACPI_PM1_EVT_PORT..kvm::ACPI_PM1_CNT_PORT + 2).contains(&port)
        || (kvm::ACPI_PM_TMR_PORT..kvm::ACPI_PM_TMR_PORT + 4).contains(&port)
}

/// Dispatch an I/O out to serial ports or system devices.
/// Returns `true` if the caller should exit (system reset detected).
#[cfg(target_arch = "x86_64")]
fn dispatch_io_out(
    com1: &PiMutex<console::Serial>,
    com2: &PiMutex<console::Serial>,
    pci_bus: Option<&PiMutex<pci::PciBus>>,
    port: u16,
    data: &[u8],
) -> bool {
    // I8042 reset: kernel writes 0xFE to port 0x64 during reboot.
    if port == I8042_CMD_PORT && data.first() == Some(&I8042_CMD_RESET_CPU) {
        return true;
    }
    // PCI type-1 CAM: CONFIG_ADDRESS (0xCF8, 4-byte latch) and CONFIG_DATA
    // (0xCFC..0xD00 byte-addressable data window). CONFIG_ADDRESS is handled
    // only as an aligned 4-byte access: the guest's pci_direct_conf1 always
    // drives 0xCF8 with outl/inl (arch/x86/pci/direct.c). Any sub-dword write
    // into 0xCF8..0xCFC (incl. the legacy outb(0x01, 0xCFB) poke in
    // pci_check_type1, and a hypothetical len<4 write to 0xCF8 itself) is NOT
    // latched — it falls through to the drop, which is harmless because
    // pci_direct_conf1 never issues such an access. CONFIG_DATA, by contrast,
    // accepts any sub-dword access within 0xCFC..0xD00 (the data window IS
    // byte-addressable).
    if let Some(bus) = pci_bus {
        if port == PCI_CONFIG_ADDRESS && data.len() == 4 {
            bus.lock()
                .cam_set_address(u32::from_le_bytes([data[0], data[1], data[2], data[3]]));
            return false;
        }
        if (PCI_CONFIG_DATA..PCI_CONFIG_DATA + 4).contains(&port) {
            bus.lock()
                .cam_data_write((port - PCI_CONFIG_DATA) as u8, data);
            return false;
        }
    }
    // ACPI PM register blocks (stub — no real power management). Every write
    // is a no-op: PM1 status is write-1-to-clear of bits that never set; PM1
    // enable arms fixed events ktstr never raises; PM1 control's SCI_EN is
    // reported by the read stub regardless; the PM timer is read-only.
    if acpi_pm_port(port) {
        return false;
    }
    // Only lock the matching serial port based on port range.
    if (console::COM1_BASE..console::COM1_BASE + 8).contains(&port) {
        com1.lock().handle_out(port, data);
    } else if (console::COM2_BASE..console::COM2_BASE + 8).contains(&port) {
        com2.lock().handle_out(port, data);
    }
    false
}

/// Dispatch an I/O in from serial ports or system devices.
/// Handles i8042 reads to satisfy the kernel's keyboard probe.
#[cfg(target_arch = "x86_64")]
fn dispatch_io_in(
    com1: &PiMutex<console::Serial>,
    com2: &PiMutex<console::Serial>,
    pci_bus: Option<&PiMutex<pci::PciBus>>,
    port: u16,
    data: &mut [u8],
) {
    // PCI type-1 CAM reads: CONFIG_ADDRESS echoes the latched value (the
    // guest's pci_check_type1 writes 0x80000000 and reads it back to confirm
    // type-1 access is present); CONFIG_DATA reads the addressed register.
    if let Some(bus) = pci_bus {
        if port == PCI_CONFIG_ADDRESS && data.len() == 4 {
            data.copy_from_slice(&bus.lock().cam_get_address().to_le_bytes());
            return;
        }
        if (PCI_CONFIG_DATA..PCI_CONFIG_DATA + 4).contains(&port) {
            bus.lock()
                .cam_data_read((port - PCI_CONFIG_DATA) as u8, data);
            return;
        }
    }
    // ACPI PM register blocks (stateless stub — no real power management).
    if acpi_pm_port(port) {
        let val: u32 = if port == kvm::ACPI_PM1_CNT_PORT {
            // PM1 control: report SCI_EN (bit0) set. ktstr leaves FADT
            // SMI_CMD=0, so the guest's acpi_enable() returns AE_OK on its
            // first acpi_hw_get_mode() check — acpi_hw_get_mode() reports
            // ACPI_SYS_MODE_ACPI directly when smi_command==0
            // (drivers/acpi/acpica/hwacpi.c) and acpi_hw_set_mode() is never
            // called (evxfevnt.c). So this SCI_EN=1 read does NOT gate
            // acpi_enable; it is belt-and-suspenders for any other ACPICA
            // reader of PM1_CNT. A constant 1 needs no stored state (ktstr
            // never disables ACPI — it reboots via the i8042). The EXACT-port
            // match (==0x604) is deliberate, NOT a range: ACPICA reads PM1_CNT
            // as a 16-bit register at the block base, so SCI_EN (bit0) lands in
            // the low byte; the high byte (0x605) correctly reads 0 via the else
            // arm. A range match would wrongly return SCI_EN=1 for a 0x605 read.
            0x0001
        } else if (kvm::ACPI_PM_TMR_PORT..kvm::ACPI_PM_TMR_PORT + 4).contains(&port) {
            // PM timer: a moving 24-bit counter so the guest's clocksource
            // watchdog doesn't flag it stuck (the guest's real clocksource is
            // kvm-clock; this exists only so ACPI enumerates the timer).
            acpi_pm_timer_value()
        } else {
            // PM1 status (PM1a_EVT_BLK @0x600, u16) and PM1 enable (@0x602, the
            // upper half of the 4-byte PM1_EVT block) both fall here: no fixed
            // event is ever pending or armed, so both registers read 0. A 2-byte
            // guest access to either offset lands correctly within this arm
            // (acpi_pm_port covers the whole 0x600..0x606 PM1 EVT+CNT range).
            0
        };
        let bytes = val.to_le_bytes();
        let n = data.len().min(4);
        data[..n].copy_from_slice(&bytes[..n]);
        return;
    }
    match port {
        // I8042 status: return 0 (no data, buffer empty).
        I8042_CMD_PORT => {
            if let Some(b) = data.first_mut() {
                *b = 0;
            }
        }
        // I8042 data: return 0 (no keypress).
        I8042_DATA_PORT => {
            if let Some(b) = data.first_mut() {
                *b = 0;
            }
        }
        // Only lock the matching serial port based on port range.
        p if (console::COM1_BASE..console::COM1_BASE + 8).contains(&p) => {
            com1.lock().handle_in(port, data);
        }
        p if (console::COM2_BASE..console::COM2_BASE + 8).contains(&p) => {
            com2.lock().handle_in(port, data);
        }
        // Unhandled port: return all-ones, the "no device responded" I/O
        // convention. Leaving `data` untouched would surface the previous
        // KVM_RUN buffer's stale bytes to the guest as a phantom read.
        _ => data.fill(0xFF),
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests;

/// Arch-neutral classify_exit coverage. `classify_exit` is shared
/// across x86_64 and aarch64; these tests construct synthetic
/// `VcpuExit` values that exist on both arches and assert the
/// dispatch table's behavior. Kept in a separate `cfg(test)` module
/// (no arch gate) so the same coverage runs on both targets — the
/// surrounding x86_64-gated `mod tests` carries port-I/O and
/// I8042-reset tests that ARE arch-specific.
#[cfg(test)]
mod tests_arch_neutral;

#[cfg(all(test, target_arch = "aarch64"))]
mod tests_aarch64;

#[cfg(all(test, target_arch = "x86_64"))]
mod handle_freeze_tests;

#[cfg(test)]
mod vcpu_reg_snapshot_tests;
