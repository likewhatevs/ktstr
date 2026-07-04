//! Runtime derivation of the virtual KASLR offset on x86_64 via
//! `MSR_LSTAR` readback from a paused vCPU.
//!
//! # Why MSR_LSTAR?
//!
//! Non-FRED x86_64 kernels populate `MSR_LSTAR` with the post-
//! relocation kernel virtual address of the SYSCALL entry handler:
//!
//! ```text
//! // arch/x86/kernel/cpu/common.c:2257 (kernel rev 9636d2ea)
//! wrmsrq(MSR_LSTAR, (unsigned long)entry_SYSCALL_64);
//! ```
//!
//! The cast resolves to the runtime KVA because
//! `handle_relocations` in `arch/x86/boot/compressed/misc.c` patches
//! every absolute 64-bit reloc with the KASLR delta before the
//! kernel's C code runs. Subtracting the link-time `entry_SYSCALL_64`
//! KVA (from the vmlinux symbol table) yields the virtual KASLR
//! offset — independent of physical-KASLR.
//!
//! # Why not phys_base subtraction?
//!
//! A prior inline `freeze_coord` derivation subtracted
//! `phys_base - real_phys_base`, but both reads targeted the same
//! kernel global and the difference reduced to a structural zero,
//! so that path was deleted (`freeze_coord/mod.rs` notes the
//! removal). Virtual and physical KASLR are
//! INDEPENDENTLY randomized by `choose_random_location` in
//! `arch/x86/boot/compressed/kaslr.c` — `find_random_phys_addr` and
//! `find_random_virt_addr` are separate slot-pickers. The virtual
//! delta is what per-CPU template arithmetic in `compute_rq_pas`
//! needs; the physical delta does not substitute.
//!
//! # FRED caveat
//!
//! On `CONFIG_X86_FRED=y` kernels, `syscall_init` at
//! `arch/x86/kernel/cpu/common.c:2303-2304` skips `idt_syscall_init`
//! entirely. The FRED parallel init writes
//! `MSR_IA32_FRED_CONFIG`, not `MSR_LSTAR` — so `MSR_LSTAR` reads
//! back as 0 (the value KVM seeded at vCPU init). [`derive_virt_kaslr`]
//! returns `Err(LstarZero)` in that case so the caller can fall back
//! to the guest-channel `KERN_ADDRS` `_text` derivation
//! (`freeze_coord/dispatch.rs`) or skip virt-KASLR-dependent reads.
//!
//! `ktstr.kconfig` defensively asserts `# CONFIG_X86_FRED is not set`
//! to keep this gate dormant for the kernels ktstr produces.
//!
//! # Timing
//!
//! `MSR_LSTAR` is meaningful only AFTER the guest's BSP has reached
//! `cpu_init → syscall_init → idt_syscall_init` in `start_kernel`.
//! Reading before that point yields 0 (the boot-time KVM seed). The
//! caller should gate reads on a guest-side readiness signal — the
//! existing `kern_phys_base` evt-fd handshake in `freeze_coord` is
//! the natural site, since by that point the guest has executed
//! enough kernel code to have populated `MSR_LSTAR` on the BSP.
//!
//! # Adversarial divergence from reference VMMs (intentional)
//!
//! None of {cloud-hypervisor, firecracker, qemu, libkrun} derives
//! KASLR from MSR_LSTAR readback. They use one of: KVM_TRANSLATE
//! (pause-gated; cannot satisfy continuous monitor at 100 ms
//! cadence), vmcoreinfo (one-shot at dump time; not live), or
//! MSR_KERNEL_GS_BASE per-vCPU (pause-gated). ktstr's live
//! host-side monitor REQUIRES non-pause-gated host-computed
//! KASLR-aware PA derivation — the LSTAR readback path here is
//! one of two such mechanisms (the other being the KERN_ADDRS
//! guest channel — the `MSG_TYPE_KERN_ADDRS` arm at
//! `src/vmm/freeze_coord/dispatch.rs`).
//! Divergence is INTENTIONAL: documented here so future reviewers
//! don't propose collapsing to a ref-VMM pattern that breaks
//! continuous monitoring.
//!
//! # `MSR_LSTAR == entry_SYSCALL_64_link` ambiguity
//!
//! When KASLR is disabled (`nokaslr` cmdline or
//! `CONFIG_RANDOMIZE_BASE=n`), the runtime LSTAR equals the link-time
//! KVA and the derived offset is legitimately 0. The same arithmetic
//! also produces 0 on a corrupted LSTAR that happens to equal the
//! link KVA — this primitive cannot distinguish "KASLR-off honest
//! zero" from "garbage equal to link". Callers that need the
//! distinction (e.g. the freeze-coord wire-in deciding whether to
//! trust the virt-KASLR path) should cross-check with an EXTERNAL
//! signal:
//! kernel cmdline `nokaslr` or `CONFIG_RANDOMIZE_BASE`. (The
//! former `phys_base`-derived cross-check no longer exists — that
//! derivation reduced to a structural zero and was deleted.)
//!
//! # BSP-only read
//!
//! `MSR_LSTAR` is a per-vCPU MSR (each KVM_GET_VCPU_EVENTS holds its
//! own copy), but every CPU's LSTAR points to the same
//! `entry_SYSCALL_64` KVA — the kernel writes the same value on the
//! BSP via `cpu_init → syscall_init → idt_syscall_init` and on
//! every AP via `start_secondary → cpu_init → syscall_init`. So the
//! derived virt-KASLR offset is CPU-INVARIANT. **Callers should read
//! from BSP (vCPU 0)** — AP LSTAR writes race the host's freeze
//! rendezvous (an AP that hasn't completed `cpu_init` yet still
//! holds the KVM-seeded 0). Reading from BSP avoids the race.

// `read_and_derive` is wired into `freeze_coord`'s BSP wait-loop
// (`crate::vmm::x86_64::msr_kaslr::read_and_derive` at
// `freeze_coord/mod.rs`), but `is_retryable` has no production
// caller today (only the unit tests below exercise it), so it
// would warn as dead-code; suppress at module scope. The module
// itself is `#[cfg(target_arch = "x86_64")]`-gated, so it is not
// compiled on other architectures.
#![allow(dead_code)]

use anyhow::Result;
use kvm_ioctls::VcpuFd;

use super::msr_indices::MSR_LSTAR;
use super::msr_io::read_one_msr;

/// KASLR randomization alignment, per `CONFIG_PHYSICAL_ALIGN` default
/// in `arch/x86/Kconfig`. `find_random_virt_addr` in
/// `arch/x86/boot/compressed/kaslr.c:840-855` picks slots as
/// multiples of this alignment.
const PHYSICAL_ALIGN: u64 = 2 * 1024 * 1024;

/// Kernel-half canonical address threshold for `entry_SYSCALL_64`
/// KVAs on x86_64. `__START_KERNEL_map = 0xffff_ffff_8000_0000` is
/// declared unconditionally in `arch/x86/include/asm/page_64_types.h`
/// — kernel-text anchors at the same base on both 4-level and
/// 5-level paging. Virt-KASLR picks a slot within `KERNEL_IMAGE_SIZE`
/// (1 GiB with `CONFIG_RANDOMIZE_BASE`) above that base, so every
/// valid runtime `entry_SYSCALL_64` KVA falls in
/// `[0xffff_ffff_8000_0000, 0xffff_ffff_c000_0000)` — well above
/// this threshold on both paging modes.
///
/// A non-zero `MSR_LSTAR` value below this threshold is either a
/// non-canonical/garbage read or the boot firmware's pre-init value
/// — not a valid `entry_SYSCALL_64` KVA.
///
/// **Scope:** this threshold is correct for `entry_SYSCALL_64`-class
/// KVAs only. Generic kernel-half detection for arbitrary KVAs
/// (direct-map base `__PAGE_OFFSET_BASE_L5 = 0xff11_0000_0000_0000`
/// or vmalloc base `__VMALLOC_BASE_L5 = 0xffa0_0000_0000_0000` both
/// FAIL this threshold and would be wrongly rejected as
/// non-canonical) needs a paging-aware variant driven by
/// `KernelSymbols::pgtable_l5_enabled` — that's the canonical-KVA
/// validity guard's scope, not this module's.
pub(crate) const KERNEL_HALF_CANONICAL_4LEVEL: u64 = 0xFFFF_8000_0000_0000;

/// Reasons the LSTAR-based virt-KASLR derivation can decline a
/// reading. Each is recoverable — the caller falls back to an
/// alternate KASLR source (the guest-channel `KERN_ADDRS` `_text`
/// derivation) or skips virt-KASLR-dependent reads for this dump.
///
/// `#[non_exhaustive]` so future paging-mode-specific gates (5-level
/// canonical-half tightening, FRED-vs-early-boot disambiguation)
/// can split existing variants without breaking match arms.
#[derive(Debug)]
#[non_exhaustive]
pub enum LstarDeriveError {
    /// KVM_GET_MSRS returned 0 entries for MSR_LSTAR (the host
    /// kernel's KVM build doesn't expose it). Should never happen
    /// on x86_64 hosts since LSTAR is mandatory for long-mode
    /// SYSCALL, but defensive against future KVM-build oddities.
    /// Same caller-side behavior as [`Self::LstarZero`]: skip
    /// virt-KASLR derivation, fall back to alternate source.
    LstarUnsupported,
    /// `MSR_LSTAR == 0`. Either the guest hasn't reached
    /// `idt_syscall_init` yet (read too early), or the kernel is
    /// `CONFIG_X86_FRED=y` and `idt_syscall_init` was skipped
    /// (FRED uses `MSR_IA32_FRED_CONFIG` instead). Both cases are
    /// distinguishable from a successful read by the zero value
    /// (the KVM-seeded boot value), but not from each other
    /// without separate FRED detection.
    LstarZero,
    /// `entry_syscall_64_link_kva == 0`. The caller fed a zero
    /// link-time KVA — usually a vmlinux that lacked the
    /// `entry_SYSCALL_64` symbol (callers should check
    /// [`crate::monitor::symbols::KernelSymbols::entry_syscall_64_kva`]
    /// for `Some` before calling). Without a real link KVA the
    /// subtraction would silently return LSTAR itself as a
    /// nonsense "offset".
    LinkUnknown,
    /// `MSR_LSTAR` is non-zero but below the kernel-half canonical
    /// threshold (bits 63..47 not all set). The kernel has written
    /// SOMETHING but not a valid `entry_SYSCALL_64` KVA — usually a
    /// transitional state or a corrupted MSR.
    NonCanonical { lstar: u64 },
    /// Link-time KVA is ABOVE the runtime `MSR_LSTAR` — the
    /// subtraction would wrap into a huge u64. Impossible under
    /// correct KASLR (offset is always non-negative), so this
    /// indicates the link-time KVA from vmlinux doesn't match the
    /// running kernel's build (wrong vmlinux for this guest).
    LinkAboveLstar { lstar: u64, link: u64 },
    /// Derived offset is not a multiple of `PHYSICAL_ALIGN` (2 MiB).
    /// KASLR slots are always 2-MiB-aligned per
    /// `find_random_virt_addr` — misalignment indicates the inputs
    /// don't agree on which kernel image is loaded.
    Misaligned { offset: u64 },
}

impl std::fmt::Display for LstarDeriveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LstarUnsupported => write!(
                f,
                "KVM_GET_MSRS returned 0 entries for MSR_LSTAR; host KVM \
                 build doesn't expose long-mode SYSCALL MSRs"
            ),
            Self::LstarZero => write!(
                f,
                "MSR_LSTAR == 0; guest may not have reached idt_syscall_init, \
                 or CONFIG_X86_FRED=y kernel skipped the LSTAR write"
            ),
            Self::LinkUnknown => write!(
                f,
                "entry_syscall_64_link_kva == 0; vmlinux lacks the \
                 entry_SYSCALL_64 symbol — callers must check \
                 KernelSymbols::entry_syscall_64_kva is Some before calling"
            ),
            Self::NonCanonical { lstar } => write!(
                f,
                "MSR_LSTAR = {lstar:#x}; not in kernel-half canonical range \
                 (expected bits 63..47 all set, ≥ {KERNEL_HALF_CANONICAL_4LEVEL:#x})"
            ),
            Self::LinkAboveLstar { lstar, link } => write!(
                f,
                "MSR_LSTAR = {lstar:#x} < entry_SYSCALL_64_link = {link:#x}; \
                 wrong vmlinux for the running guest kernel"
            ),
            Self::Misaligned { offset } => write!(
                f,
                "derived virt_kaslr_offset = {offset:#x} not aligned to \
                 PHYSICAL_ALIGN ({PHYSICAL_ALIGN:#x}); inputs disagree on \
                 the loaded kernel image"
            ),
        }
    }
}

impl std::error::Error for LstarDeriveError {}

impl LstarDeriveError {
    /// Whether re-reading LSTAR might produce a different result.
    ///
    /// `LstarUnsupported`, `LstarZero`, `NonCanonical` are likely-
    /// transient (host KVM build / early-boot timing / transitional
    /// state — re-read after a delay may succeed). `LinkUnknown`,
    /// `LinkAboveLstar`, `Misaligned` are permanent (wrong vmlinux
    /// / corrupted symbol table — re-reading won't change the
    /// inputs). The retry-or-fall-back wiring uses this to decide
    /// mechanically rather than per-variant doc reading.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::LstarUnsupported | Self::LstarZero | Self::NonCanonical { .. }
        )
    }
}

/// Derive the virtual KASLR offset from a runtime `MSR_LSTAR` value
/// and the link-time `entry_SYSCALL_64` KVA from vmlinux.
///
/// Returns `Err(LstarDeriveError)` with a specific variant for each
/// failure mode so the caller can log + fall back rather than
/// silently zeroing.
pub fn derive_virt_kaslr(lstar: u64, entry_syscall_64_link: u64) -> Result<u64, LstarDeriveError> {
    if lstar == 0 {
        return Err(LstarDeriveError::LstarZero);
    }
    if entry_syscall_64_link == 0 {
        return Err(LstarDeriveError::LinkUnknown);
    }
    if lstar < KERNEL_HALF_CANONICAL_4LEVEL {
        return Err(LstarDeriveError::NonCanonical { lstar });
    }
    if lstar < entry_syscall_64_link {
        return Err(LstarDeriveError::LinkAboveLstar {
            lstar,
            link: entry_syscall_64_link,
        });
    }
    let offset = lstar - entry_syscall_64_link;
    if offset & (PHYSICAL_ALIGN - 1) != 0 {
        return Err(LstarDeriveError::Misaligned { offset });
    }
    Ok(offset)
}

/// Convenience helper: read `MSR_LSTAR` from `vcpu` and derive the
/// virt-KASLR offset from the supplied link-time KVA. Used by the
/// freeze-coordinator at the `kern_phys_base` handshake site —
/// guarantees the guest has reached `idt_syscall_init` before the
/// read (otherwise LSTAR is the KVM-seeded 0).
///
/// Errors fall into two classes:
/// - Read failures (ioctl error, KVM ABI violation) bubble as
///   `anyhow::Error` and indicate a KVM-layer problem.
/// - Derivation failures (any [`LstarDeriveError`] variant —
///   `LstarUnsupported`, `LstarZero`, `LinkUnknown`, `NonCanonical`,
///   `LinkAboveLstar`, `Misaligned`) bubble as `LstarDeriveError`
///   wrapped in `anyhow::Error` and indicate the kernel's LSTAR
///   isn't usable for KASLR derivation on this guest (FRED, early
///   boot, wrong vmlinux, missing symbol, ...). Callers that want
///   to dispatch per-variant should call
///   [`super::msr_io::read_one_msr`] with [`MSR_LSTAR`] and
///   [`derive_virt_kaslr`] directly to get the typed error.
///
/// Callers typically fall back to the guest-channel `KERN_ADDRS`
/// `_text` derivation on any error here — that path is independent
/// of LSTAR and works whether or not LSTAR is meaningful.
///
/// # Confidential-compute caveat (SEV-ES / SEV-SNP / TDX)
///
/// `KVM_GET_MSRS` for MSR_LSTAR FAILS on confidential-compute
/// guests — fail-loud rather than silent-garbage. ktstr does NOT
/// target SEV-ES / SEV-SNP / TDX guests today; the gates below
/// catch it correctly via the existing `is_retryable()` /
/// `LstarUnsupported` / `LstarZero` path (`is_retryable` at
/// L243-248, the `LstarUnsupported` gate in `read_and_derive` at
/// L340-349; the wait-loop itself lives in `freeze_coord/mod.rs`,
/// not this file), fall
/// through to the wait-loop, and let the guest-channel
/// `KERN_ADDRS` `_text` publisher
/// (`src/vmm/freeze_coord/dispatch.rs`) or the `nokaslr` cmdline
/// (`src/vmm/setup/mod.rs`) become the authoritative source.
///
/// Kernel-source paths the gates rely on:
///
/// - SEV-ES / SEV-SNP: `svm_get_msr` (arch/x86/kvm/svm/svm.c:2733-2738)
///   calls `sev_es_prevent_msr_access` FIRST
///   (arch/x86/kvm/svm/svm.c:2725-2731). When
///   `sev_es_guest(kvm) && vcpu->arch.guest_state_protected &&
///    msr_index != MSR_IA32_XSS && !msr_write_intercepted`, data
///   is zeroed AND returns `-EINVAL` if `has_protected_state`,
///   else `0` (success with data=0). ioctl FAILS or returns
///   zero — never garbage.
///
/// - TDX: `vt_get_msr` (arch/x86/kvm/vmx/main.c:183-189) dispatches
///   to `tdx_get_msr` (arch/x86/kvm/vmx/tdx.c:2157-2180) for TD
///   vCPUs. `tdx_get_msr` falls through to `kvm_get_msr_common`
///   ONLY if `tdx_has_emulated_msr(index)` returns true
///   (arch/x86/kvm/vmx/tdx.c:2104-2149) — an explicit allow-list
///   (UCODE_REV, ARCH_CAPABILITIES, MTRR*, EFER, FEAT_CTL, etc.).
///   MSR_LSTAR is NOT in the list, so `tdx_get_msr` returns 1,
///   ioctl FAILS — never garbage.
///
/// Future cc support remains a fallback-engagement question, not a
/// data-correctness one. If a confidential-compute backend is
/// added, the existing fail-loud → KERN_ADDRS / nokaslr fallback
/// chain Just Works; document the assumption explicitly at the
/// backend's BSP-init site.
pub(crate) fn read_and_derive(vcpu: &VcpuFd, entry_syscall_64_link: u64) -> Result<u64> {
    let lstar = read_one_msr(vcpu, MSR_LSTAR)?.ok_or_else(|| {
        anyhow::anyhow!(
            "derive virt_kaslr_offset: {}",
            LstarDeriveError::LstarUnsupported
        )
    })?;
    derive_virt_kaslr(lstar, entry_syscall_64_link)
        .map_err(|e| anyhow::anyhow!("derive virt_kaslr_offset: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    const LINK_KVA_TYPICAL: u64 = 0xFFFF_FFFF_8100_0000;

    fn aligned_lstar(offset_mb: u64) -> u64 {
        LINK_KVA_TYPICAL + (offset_mb * 1024 * 1024)
    }

    #[test]
    fn derive_happy_path() {
        let lstar = aligned_lstar(20); // 20 MiB virt-KASLR offset
        let offset = derive_virt_kaslr(lstar, LINK_KVA_TYPICAL).expect("derive succeeds");
        assert_eq!(offset, 20 * 1024 * 1024);
    }

    #[test]
    fn derive_zero_offset_is_valid() {
        // KASLR-off (nokaslr cmdline or CONFIG_RANDOMIZE_BASE=n)
        // produces lstar == link → offset == 0. Valid result.
        let offset =
            derive_virt_kaslr(LINK_KVA_TYPICAL, LINK_KVA_TYPICAL).expect("zero-offset derive ok");
        assert_eq!(offset, 0);
    }

    #[test]
    fn derive_lstar_zero_fred_or_early_boot() {
        let err =
            derive_virt_kaslr(0, LINK_KVA_TYPICAL).expect_err("zero LSTAR must produce LstarZero");
        assert!(matches!(err, LstarDeriveError::LstarZero));
    }

    #[test]
    fn derive_non_canonical_below_high_half() {
        // A non-zero LSTAR that lacks the kernel-half bits.
        // Defends against KVM returning a transitional/garbage value
        // that happens to be non-zero but is not a valid kernel KVA.
        let bogus = 0x0000_FFFF_8100_0000u64;
        let err = derive_virt_kaslr(bogus, LINK_KVA_TYPICAL).expect_err("non-canonical must fail");
        assert!(matches!(err, LstarDeriveError::NonCanonical { lstar } if lstar == bogus));
    }

    #[test]
    fn derive_link_above_lstar_when_link_too_high() {
        // Indicates wrong vmlinux for this guest — refuse to compute
        // a wrapped-around-negative u64 offset.
        let lstar = LINK_KVA_TYPICAL - (4 * 1024 * 1024);
        let err = derive_virt_kaslr(lstar, LINK_KVA_TYPICAL).expect_err("lstar < link must error");
        assert!(matches!(
            err,
            LstarDeriveError::LinkAboveLstar { lstar: l, link: k }
                if l == lstar && k == LINK_KVA_TYPICAL
        ));
    }

    #[test]
    fn derive_at_canonical_threshold_uses_link_guard() {
        // lstar exactly == KERNEL_HALF_CANONICAL_4LEVEL passes the canonical
        // guard (strict <); falls through to the link guard. With a
        // higher link KVA this surfaces LinkAboveLstar.
        let err = derive_virt_kaslr(KERNEL_HALF_CANONICAL_4LEVEL, LINK_KVA_TYPICAL)
            .expect_err("lstar < link must produce LinkAboveLstar");
        assert!(matches!(err, LstarDeriveError::LinkAboveLstar { .. }));
    }

    #[test]
    fn derive_minimum_aligned_offset() {
        // The smallest nonzero KASLR offset is one PHYSICAL_ALIGN
        // step (2 MiB). Accepting this pins the boundary at the
        // alignment gate.
        let offset = derive_virt_kaslr(aligned_lstar(2), LINK_KVA_TYPICAL)
            .expect("2 MiB offset is the minimum valid nonzero");
        assert_eq!(offset, 2 * 1024 * 1024);
    }

    #[test]
    fn derive_lstar_max_u64_falls_to_misaligned() {
        // u64::MAX passes canonical (≥ threshold) + link (> link).
        // The subtraction u64::MAX - LINK_KVA_TYPICAL has low bits
        // set (LINK has low bits ≠ 0 when subtracted from all-ones),
        // so the alignment guard catches it.
        let err = derive_virt_kaslr(u64::MAX, LINK_KVA_TYPICAL)
            .expect_err("u64::MAX produces misaligned offset");
        assert!(matches!(err, LstarDeriveError::Misaligned { .. }));
    }

    #[test]
    fn derive_errors_render_diagnostic_payload() {
        // Each error variant's Display includes the input values so
        // operators can debug from the rendered message alone. A
        // refactor that strips the hex payload would silently
        // degrade ops debuggability; this test guards.
        let lstar_zero = format!("{}", LstarDeriveError::LstarZero);
        assert!(lstar_zero.contains("MSR_LSTAR == 0"));

        let non_canonical = format!(
            "{}",
            LstarDeriveError::NonCanonical {
                lstar: 0x1234_5678_9abc_def0
            }
        );
        assert!(non_canonical.contains("0x1234"));

        let link_above = format!(
            "{}",
            LstarDeriveError::LinkAboveLstar {
                lstar: 0x1,
                link: 0x2
            }
        );
        assert!(link_above.contains("0x1") && link_above.contains("0x2"));

        let misaligned = format!(
            "{}",
            LstarDeriveError::Misaligned {
                offset: 0xdead_beef
            }
        );
        assert!(misaligned.contains("0xdeadbeef"));
    }

    #[test]
    fn is_retryable_partitions_variants_correctly() {
        // Likely-transient (re-read might change result):
        assert!(LstarDeriveError::LstarUnsupported.is_retryable());
        assert!(LstarDeriveError::LstarZero.is_retryable());
        assert!(LstarDeriveError::NonCanonical { lstar: 0 }.is_retryable());
        // Permanent (wrong inputs, re-read can't fix):
        assert!(!LstarDeriveError::LinkUnknown.is_retryable());
        assert!(!LstarDeriveError::LinkAboveLstar { lstar: 0, link: 0 }.is_retryable());
        assert!(!LstarDeriveError::Misaligned { offset: 0 }.is_retryable());
    }

    #[test]
    fn derive_link_unknown_when_link_is_zero() {
        // vmlinux that lacks entry_SYSCALL_64 → caller-side
        // KernelSymbols::entry_syscall_64_kva is None, but a careless
        // caller could pass `.unwrap_or(0)` and get a nonsense
        // derived offset (literally lstar itself). The LinkUnknown
        // guard catches this before subtraction.
        let err =
            derive_virt_kaslr(aligned_lstar(8), 0).expect_err("link == 0 must error LinkUnknown");
        assert!(matches!(err, LstarDeriveError::LinkUnknown));
    }

    #[test]
    fn derive_misalignment_rejected() {
        // KASLR is 2-MiB-aligned per CONFIG_PHYSICAL_ALIGN. A 4-KiB-
        // misaligned offset can't be a valid KASLR result.
        let lstar = LINK_KVA_TYPICAL + (20 * 1024 * 1024) + 4096;
        let err = derive_virt_kaslr(lstar, LINK_KVA_TYPICAL).expect_err("misalign must fail");
        assert!(matches!(err, LstarDeriveError::Misaligned { offset }
                if offset == (20 * 1024 * 1024) + 4096));
    }
}
