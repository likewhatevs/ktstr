//! Lazy retry wrapper for the boot-race-sensitive per-CPU offsets.
//!
//! One coordinator-state cache needs to be populated lazily because its
//! construction depends on guest-memory bootstrap symbols that the
//! kernel only writes during boot (`page_offset_base`,
//! `pgtable_l5_enabled`, `init_top_pgt`, `__per_cpu_offset[]`): the
//! per-CPU offset array, read once via
//! [`crate::monitor::symbols::read_per_cpu_offsets`] and cached for the
//! rest of the run; gated on every entry having bit 63 set (kernel-half
//! pointer) so a partially-online VM doesn't poison the cache with a
//! CPU 0 alias for not-yet-online CPUs (the rq PA invariant).
//!
//! # No state-machine semantics change
//!
//! The helper is byte-for-byte identical to the inline retry block it
//! replaced: tcr_el1 is loaded with Acquire INSIDE the helper so a
//! fresh value is observed each iteration — capturing it pre-call would
//! freeze a stale snapshot the BSP loop hasn't refined yet. It returns
//! `Option` rather than `Result` because its failure mode includes the
//! "any slot still fails the bit-63 gate" non-error retry condition.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::monitor;

/// Resolve and cache the per-CPU offset array. Returns `Some(offsets)`
/// only when every slot has bit 63 set (kernel-half pointer) so a
/// partially-online VM does
/// not poison the cache with a CPU 0 alias for not-yet-online CPUs
/// (rq PA invariant; fix for `compute_rq_pas` wraparound when a
/// `pco_offset == 0` is fed downstream). Returns `None` when:
///
/// * `per_cpu_offset_kva == 0` (caller's symbol cache had no entry
///   for `__per_cpu_offset` — typically a stripped vmlinux image),
///   OR
/// * any slot fails the bit-63 (kernel-half) gate — a still-zero
///   slot or a non-zero-but-bit-63-clear value like `0x3` (caller
///   leaves cache `None`, retries next scan tick).
///
/// Takes a pre-resolved `per_cpu_offset_kva` from the coordinator's
/// `dump_cpu_time_symbols` cache instead of re-running
/// `KernelSymbols::from_vmlinux` on every scan tick. The previous
/// in-helper parse re-read the entire vmlinux ELF (50 MB+) and re-
/// built every symbol-table entry every 250 ms while waiting for
/// the per-CPU areas to come up — visible as ~MB/s of constant
/// post-boot file I/O on every ktstr run. The KVA is fixed at
/// kernel link time and the caller already resolved it once at
/// coord start; passing it through eliminates the redundant work
/// without changing the post-resolution invariants.
///
/// `phys_base` is sourced by the caller from the owned accessor's
/// `GuestKernel::phys_base()` when it has landed; otherwise `0`
/// (correct on non-KASLR boots and the bootstrap value before the
/// accessor's page-table walk has resolved phys_base for the live
/// kernel).
pub(super) fn try_init_prog_per_cpu_offsets(
    mem: &monitor::reader::GuestMem,
    per_cpu_offset_kva: u64,
    tcr_el1: Option<&Arc<AtomicU64>>,
    phys_base: u64,
    num_cpus: u32,
) -> Option<Vec<u64>> {
    if per_cpu_offset_kva == 0 {
        return None;
    }
    let tcr_val = tcr_el1.map(|c| c.load(Ordering::Acquire)).unwrap_or(0);
    let start_kernel_map = monitor::symbols::start_kernel_map_for_tcr(tcr_val)
        .unwrap_or(monitor::symbols::START_KERNEL_MAP);
    let pco_pa =
        monitor::symbols::text_kva_to_pa_with_base(per_cpu_offset_kva, start_kernel_map, phys_base);
    let offsets = monitor::symbols::read_per_cpu_offsets(mem, pco_pa, num_cpus);
    // Defer caching until every offset slot looks like a legitimate
    // kernel-half pointer (bit 63 set). Two failure modes the gate
    // catches:
    //   * Still-booting guest: not-yet-initialised CPUs read 0 from
    //     the array (BSS pre-zero before `setup_per_cpu_areas`).
    //     Caching zero would alias every such CPU's stats to CPU 0.
    //   * Wrong-PA garbage: an off-by-N or stale-phys_base resolution
    //     can leave a slot reading a small-magnitude value (the
    //     arm64 `pco0 = 0x3` shape observed in CI). Treating that as
    //     a real per-CPU offset would propagate the wrong KVA into
    //     every downstream prog_runtime_stats walker and produce
    //     zero-filled samples from random guest pages.
    //
    // Every legitimate `__per_cpu_offset[cpu]` value has bit 63 set
    // on both supported architectures (kernel-half direct-map KVA on
    // x86_64; high-half delta `pcpu_base_addr - __per_cpu_start +
    // pcpu_unit_offsets[cpu]` whose subtraction inherits bit 63 on
    // aarch64). The sibling bit-63 gate on the DATA_VALID latch in
    // monitor/reader.rs uses the same check — keeping them in lockstep
    // avoids a class of bugs where one gate accepts a value the
    // other rejects.
    //
    // A retry is cheap; a cached miss is permanent for the run. For
    // prog_runtime_stats this means stats for CPUs that haven't
    // booted yet (or that resolve through a wrong PA) are simply
    // missing — degraded, not corrupt.
    if offsets.is_empty() || !offsets.iter().all(|&v| v & (1u64 << 63) != 0) {
        // Diagnostic for the post-cluster-D/B failure shape: emit
        // pco_pa + the first two slot reads so a failing CI run
        // pinpoints whether (a) `pco_pa` lands at a wrong address
        // (off-by-N) and one slot reads garbage, (b) the guest's
        // `setup_per_cpu_areas` hasn't populated every slot yet,
        // OR (c) cached `phys_base` is stale so `pco_pa` resolves
        // to an unrelated DRAM page reading all-zeros. Emitted at
        // `tracing::warn` with target `ktstr::failure_dump` so it
        // lands in the CI failure-dump log; bounded
        // to once per `try_init_prog_per_cpu_offsets` invocation —
        // the caller already gates the retry on a 250 ms
        // scan_tick, so volume is naturally rate-limited.
        tracing::warn!(
            target: "ktstr::failure_dump",
            per_cpu_offset_kva = format_args!("{:#x}", per_cpu_offset_kva),
            start_kernel_map = format_args!("{:#x}", start_kernel_map),
            phys_base = format_args!("{:#x}", phys_base),
            pco_pa = format_args!("{:#x}", pco_pa),
            num_cpus,
            slot0 = format_args!("{:#x}", offsets.first().copied().unwrap_or(0)),
            slot1 = format_args!("{:#x}", offsets.get(1).copied().unwrap_or(0)),
            "freeze-coord: try_init_prog_per_cpu_offsets returning None — one or more slots fail the kernel-half (bit-63) gate (will retry next scan_tick)"
        );
        None
    } else {
        Some(offsets)
    }
}
