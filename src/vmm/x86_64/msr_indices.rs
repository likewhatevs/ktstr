//! Centralized x86_64 MSR index constants.
//!
//! Every `MSR_*: u32` that ktstr's VMM code touches lives here. Both
//! the boot-time MSR seed in [`super::boot::setup_msrs`] and the
//! runtime MSR readback in [`super::msr_io`] import from this module
//! so the numeric values appear in exactly one place. Value-bit
//! consts (e.g. `MSR_IA32_MISC_ENABLE_FAST_STRING`) and feature-bit
//! masks live in their consumer modules — only MSR identity numbers
//! live here.
//!
//! This is a consumer-only list, not a full mirror of
//! `arch/x86/include/asm/msr-index.h`. Reference VMMs like
//! firecracker and libkrun vendor the entire header via bindgen;
//! ktstr lists only the MSRs it actually touches so a reader sees
//! the relevant ABI surface in one short file. New readers/writers
//! add their MSR's index here rather than reaching for a generated
//! crate.
//!
//! Per-const documentation cites the kernel header that defines the
//! value so a reader can chase the constant to its kernel-source
//! authority without leaving the file.
//!
//! ## Adding a new MSR
//!
//! Add the `pub(crate) const` in numeric order by MSR address with
//! the kernel-header cite. The `// Owner:` comment convention is
//! reserved for MSRs whose semantic role is owned by a non-generic
//! consumer module (e.g. `MSR_LSTAR` — owned by [`super::msr_kaslr`]
//! for virt-KASLR derivation); MSRs consumed only by the boot-time
//! seed in [`super::boot::setup_msrs`] do not need an owner comment.

/// `MSR_IA32_TSC` — time-stamp counter. arch/x86/include/asm/msr-index.h
pub(crate) const MSR_IA32_TSC: u32 = 0x10;

/// `MSR_IA32_SYSENTER_CS` — SYSENTER target code segment.
/// arch/x86/include/asm/msr-index.h
pub(crate) const MSR_IA32_SYSENTER_CS: u32 = 0x174;

/// `MSR_IA32_SYSENTER_ESP` — SYSENTER target stack pointer.
/// arch/x86/include/asm/msr-index.h
pub(crate) const MSR_IA32_SYSENTER_ESP: u32 = 0x175;

/// `MSR_IA32_SYSENTER_EIP` — SYSENTER target instruction pointer.
/// arch/x86/include/asm/msr-index.h
pub(crate) const MSR_IA32_SYSENTER_EIP: u32 = 0x176;

/// `MSR_IA32_MISC_ENABLE` — Intel feature-control MSR (incl.
/// `FAST_STRING` bit 0). arch/x86/include/asm/msr-index.h
pub(crate) const MSR_IA32_MISC_ENABLE: u32 = 0x1a0;

/// `MSR_MTRR_DEF_TYPE` — default memory type for ranges not covered
/// by a variable MTRR. arch/x86/include/asm/msr-index.h
///
/// The kernel header itself spells the symbol `MSR_MTRRdefType`
/// (camelCase). ktstr uses the SCREAMING_SNAKE_CASE form for
/// consistency with the other MSR_* consts in this module; the
/// numeric value matches the kernel verbatim.
pub(crate) const MSR_MTRR_DEF_TYPE: u32 = 0x2ff;

/// `MSR_STAR` — legacy SYSCALL/SYSRET segment selectors (32-bit
/// compat). arch/x86/include/asm/msr-index.h
pub(crate) const MSR_STAR: u32 = 0xc000_0081;

/// `MSR_LSTAR` — long-mode SYSCALL target RIP. Holds the runtime
/// virtual address of `entry_SYSCALL_64` on non-FRED kernels.
/// arch/x86/include/asm/msr-index.h
///
/// Owner: super::msr_kaslr — virt-KASLR derivation reads this MSR
/// and subtracts the link-time `entry_SYSCALL_64` KVA to recover the
/// runtime KASLR offset.
pub(crate) const MSR_LSTAR: u32 = 0xc000_0082;

/// `MSR_CSTAR` — compat-mode SYSCALL target RIP. Unused at the
/// 64-bit-only entry path but seeded to 0 alongside `MSR_LSTAR` for
/// determinism. arch/x86/include/asm/msr-index.h
pub(crate) const MSR_CSTAR: u32 = 0xc000_0083;

/// `MSR_SYSCALL_MASK` — RFLAGS bits cleared on SYSCALL entry.
/// arch/x86/include/asm/msr-index.h
pub(crate) const MSR_SYSCALL_MASK: u32 = 0xc000_0084;

/// `MSR_KERNEL_GS_BASE` — base address swapped into `%gs` on
/// SWAPGS. arch/x86/include/asm/msr-index.h
pub(crate) const MSR_KERNEL_GS_BASE: u32 = 0xc000_0102;
