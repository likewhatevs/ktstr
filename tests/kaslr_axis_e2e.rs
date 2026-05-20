//! Post-boot KASLR offset is non-zero, 2 MiB aligned, within the
//! kernel's RANDOMIZE_BASE_MAX_OFFSET (1 GiB). Validates the kconfig
//! + cmdline + Arc-publish + collect_results chain end to end.
//!
//! Failure mode hierarchy:
//! 1. `kern_kaslr_offset == 0` — either the cmdline still ships
//!    `nokaslr`, the kconfig didn't enable CONFIG_RANDOMIZE_BASE, or
//!    the MSR_LSTAR-derived offset never published to the Arc
//!    (msr_kaslr.rs → kern_virt_kaslr_for_result chain).
//! 2. Alignment violation — kernel guarantees 2 MiB slot alignment
//!    per `arch/x86/boot/compressed/kaslr.c::process_mem_region`;
//!    violation means MSR_LSTAR readback corrupted (subtracted the
//!    wrong base symbol) or the publish path arithmetic is wrong.
//! 3. Range violation — `RANDOMIZE_BASE_MAX_OFFSET = 1 GiB`
//!    (arch/x86/include/asm/setup.h); violation indicates the
//!    derivation subtracted from the wrong link-time anchor.

use anyhow::Result;
use ktstr::assert::AssertResult;
use ktstr::ktstr_test;
use ktstr::prelude::*;
use ktstr::scenario::Ctx;
use ktstr::scenario::ops::{HoldSpec, KernelTarget, KernelValueWidth, Op, Step, execute_steps};

declare_scheduler!(KTSTR_SCHED, {
    name = "ktstr_sched",
    binary = "scx-ktstr",
});

#[ktstr_test(
    scheduler = KTSTR_SCHED,
    duration_s = 2,
    auto_repro = false,
    post_vm = assert_kaslr_on,
)]
fn kaslr_offset_nonzero_post_boot(_ctx: &Ctx) -> Result<AssertResult> {
    Ok(AssertResult::pass())
}

fn assert_kaslr_on(result: &VmResult) -> Result<()> {
    let off = result.kern_kaslr_offset;
    anyhow::ensure!(
        off != 0,
        "kern_kaslr_offset == 0 under KASLR-on default — \
         flip didn't reach guest (cmdline still ships `nokaslr`, \
         kconfig lacks CONFIG_RANDOMIZE_BASE=y, or the \
         MSR_LSTAR→kern_virt_kaslr publish chain at \
         msr_kaslr.rs → kern_virt_kaslr_for_result)"
    );
    // 2 MiB slot alignment per arch/x86/boot/compressed/kaslr.c.
    const SLOT: u64 = 2 * 1024 * 1024;
    anyhow::ensure!(
        off.is_multiple_of(SLOT),
        "kern_kaslr_offset {off:#x} not 2 MiB aligned — kernel \
         guarantees 2 MiB slot alignment per arch/x86/boot/compressed/\
         kaslr.c::process_mem_region"
    );
    // KERNEL_IMAGE_SIZE = 1 GiB per arch/x86/include/asm/page_64_types.h:85
    // when CONFIG_RANDOMIZE_BASE=y. RANDOMIZE_BASE picks slots in
    // [0, KERNEL_IMAGE_SIZE) per arch/x86/boot/compressed/kaslr.c.
    const MAX_OFFSET: u64 = 1 << 30;
    anyhow::ensure!(
        off <= MAX_OFFSET,
        "kern_kaslr_offset {off:#x} exceeds 1 GiB max — outside \
         RANDOMIZE_BASE_MAX_OFFSET; derivation likely subtracted \
         from the wrong link-time anchor"
    );
    anyhow::ensure!(
        result.kaslr_enabled(),
        "kaslr_enabled() returned false but kern_kaslr_offset = {off:#x}"
    );
    Ok(())
}

#[ktstr_test(
    scheduler = KTSTR_SCHED,
    duration_s = 2,
    auto_repro = false,
    kaslr = false,
    post_vm = assert_kaslr_off,
)]
fn kaslr_disabled_via_macro_attribute(_ctx: &Ctx) -> Result<AssertResult> {
    Ok(AssertResult::pass())
}

fn assert_kaslr_off(result: &VmResult) -> Result<()> {
    anyhow::ensure!(
        result.kern_kaslr_offset == 0,
        "kaslr=false attribute did not reach guest: kern_kaslr_offset={:#x} \
         (expected 0; nokaslr cmdline should have pinned KASLR off)",
        result.kern_kaslr_offset,
    );
    anyhow::ensure!(
        !result.kaslr_enabled(),
        "kaslr_enabled() returned true under kaslr=false attribute — \
         derivation chain or VmResult plumbing leaked a non-zero value"
    );
    Ok(())
}

// =========================================================================
// kaslr_compute_rq_pas_e2e — exercises the coord_kaslr_offset() thread-
// through into resolve_per_cpu_field_pa (kernel_op_dispatch.rs).
// =========================================================================

const TAG_PER_CPU: &str = "verify_kaslr_rq_nr_running";

#[ktstr_test(
    scheduler = KTSTR_SCHED,
    duration_s = 3,
    auto_repro = false,
    post_vm = assert_compute_rq_pas,
)]
fn kaslr_compute_rq_pas_e2e(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![Step::new(
        vec![Op::read_kernel_cold(
            TAG_PER_CPU,
            KernelTarget::per_cpu_field("runqueues", "nr_running", 0),
            KernelValueWidth::u32(),
        )],
        HoldSpec::FULL,
    )];
    execute_steps(ctx, steps)
}

fn assert_compute_rq_pas(result: &VmResult) -> Result<()> {
    anyhow::ensure!(
        result.kern_kaslr_offset != 0,
        "kern_kaslr_offset == 0 — cannot pin the PA-derivation path \
         under KASLR-on when KASLR is off. Either the cmdline flip \
         did not reach the guest, or this test ran under nokaslr."
    );
    let replies = result.snapshot_bridge.drain_kernel_ops();
    let reply = replies
        .iter()
        .find(|(t, _)| t == TAG_PER_CPU)
        .map(|(_, r)| r)
        .ok_or_else(|| {
            let tags: Vec<&str> = replies.iter().map(|(t, _)| t.as_str()).collect();
            anyhow::anyhow!(
                "no reply for tag '{TAG_PER_CPU}'; captured={tags:?} — \
                 Op::read_kernel_cold never round-tripped"
            )
        })?;
    anyhow::ensure!(
        reply.success,
        "PerCpuField read FAILED under KASLR-on: {} — the \
         kaslr_offset thread-through in kernel_op_dispatch.rs \
         did not land or did not cover this read path",
        reply.reason
    );
    let val = reply
        .read_values
        .first()
        .ok_or_else(|| anyhow::anyhow!("PerCpuField reply read_values empty"))?;
    let n = match val {
        KernelOpValue::U32(v) => *v,
        other => anyhow::bail!("PerCpuField expected U32, got {other:?}"),
    };
    anyhow::ensure!(
        n <= 1000,
        "nr_running = {n} on cpu 0 implausible (>1000 queued tasks \
         on a quiescent fixture) — likely wrong PA (kva_to_pa \
         wrapped or kaslr_offset misapplied)"
    );
    Ok(())
}

// =========================================================================
// aarch64 KASLR variant — KERN_ADDRS-only derivation path.
// =========================================================================

#[cfg(target_arch = "aarch64")]
#[ktstr_test(
    scheduler = KTSTR_SCHED,
    duration_s = 2,
    auto_repro = false,
    post_vm = assert_kaslr_aarch64,
)]
fn kaslr_offset_nonzero_post_boot_aarch64(_ctx: &Ctx) -> Result<AssertResult> {
    Ok(AssertResult::pass())
}

#[cfg(target_arch = "aarch64")]
fn assert_kaslr_aarch64(result: &VmResult) -> Result<()> {
    let off = result.kern_kaslr_offset;
    anyhow::ensure!(
        off != 0,
        "aarch64 KASLR-on but kern_kaslr_offset == 0 — KERN_ADDRS \
         derivation path did not publish (msr_kaslr.rs is x86-only; \
         the aarch64 path is in vmm/aarch64/* via KERN_ADDRS _text \
         readback minus link-time _text)"
    );
    anyhow::ensure!(
        off % 4096 == 0,
        "aarch64 kern_kaslr_offset {off:#x} not 4 KiB aligned — \
         page-aligned slide invariant violated"
    );
    // NOTE: no upper bound — arm64 RANDOMIZE_BASE entropy varies by
    // config (MODULES_VSIZE / 4096); page-aligned + non-zero is the
    // load-bearing pin per arch/arm64/kernel/kaslr.c. Don't add a
    // 1 GiB cap here (x86-only invariant).
    Ok(())
}

// =========================================================================
// kaslr_page_offset_derivation_nonzero — assert that under
// CONFIG_RANDOMIZE_MEMORY=y the runtime page_offset_base value lives at
// a PUD-aligned non-DEFAULT slot. kernel_randomize_memory picks slots in
// `[__PAGE_OFFSET_BASE_L4, __PAGE_OFFSET_BASE_L4 + remain_entropy)` with
// PUD alignment (1 GiB granularity) — see arch/x86/mm/kaslr.c. ~1/30000
// slot-0 re-roll risk — accepted as test flake floor (emit WARN, still
// pass).
//
// Read mechanism: Op::ReadKernelCold(KernelTarget::symbol("page_offset_base"),
// u64). Symbol read path uses text_kva_to_pa under the hood — already
// KASLR-aware via kern_syms.phys_base + start_kernel_map.
//
// X86_64-ONLY: arm64 has no page_offset_base global (PAGE_OFFSET is
// compile-time per arch/arm64/include/asm/memory.h:43-45).
// =========================================================================

#[cfg(target_arch = "x86_64")]
const TAG_POB: &str = "page_offset_base_value";
#[cfg(target_arch = "x86_64")]
const TAG_JIFFIES_CROSSCHECK: &str = "jiffies_64_crosscheck";

#[cfg(target_arch = "x86_64")]
#[ktstr_test(
    scheduler = KTSTR_SCHED,
    duration_s = 2,
    auto_repro = false,
    post_vm = assert_page_offset_randomized,
)]
fn kaslr_page_offset_derivation_nonzero(ctx: &Ctx) -> Result<AssertResult> {
    // Two reads in the same Step (one freeze rendezvous):
    //   1. page_offset_base — the target of this test
    //   2. jiffies_64 cross-check — proves the symbol-read chain
    //      (phys_base + text_kva_to_pa + read_u64) is healthy BEFORE
    //      we assert on page_offset_base's value. If jiffies_64
    //      reads 0, the symbol-read chain is broken upstream and the
    //      page_offset assertion is vacuous.
    let steps = vec![Step::new(
        vec![
            Op::read_kernel_cold(
                TAG_POB,
                KernelTarget::symbol("page_offset_base"),
                KernelValueWidth::u64(),
            ),
            Op::read_kernel_cold(
                TAG_JIFFIES_CROSSCHECK,
                KernelTarget::symbol("jiffies_64"),
                KernelValueWidth::u64(),
            ),
        ],
        HoldSpec::FULL,
    )];
    execute_steps(ctx, steps)
}

#[cfg(target_arch = "x86_64")]
fn assert_page_offset_randomized(result: &VmResult) -> Result<()> {
    anyhow::ensure!(
        result.kern_kaslr_offset != 0,
        "kern_kaslr_offset == 0 — symbol-read path uses text_kva_to_pa \
         which is slide-aware; without kaslr_offset != 0 we can't pin \
         that the test exercised the CONFIG_RANDOMIZE_MEMORY chain"
    );
    let replies = result.snapshot_bridge.drain_kernel_ops();
    // Belt-and-suspenders: cross-check the
    // symbol-read chain via jiffies_64 BEFORE asserting on
    // page_offset_base. If jiffies_64 reads 0 or fails, the chain
    // is broken upstream and the page_offset assertion below would
    // be vacuous.
    let jiffies_reply = replies
        .iter()
        .find(|(t, _)| t == TAG_JIFFIES_CROSSCHECK)
        .map(|(_, r)| r)
        .ok_or_else(|| anyhow::anyhow!("no reply for tag '{TAG_JIFFIES_CROSSCHECK}'"))?;
    anyhow::ensure!(
        jiffies_reply.success,
        "jiffies_64 cross-check read FAILED: {} — symbol-read chain \
         (phys_base + text_kva_to_pa + read_u64) broken upstream; \
         page_offset_base assertion would be vacuous",
        jiffies_reply.reason
    );
    let jiffies_v = match jiffies_reply.read_values.first() {
        Some(KernelOpValue::U64(v)) => *v,
        other => anyhow::bail!("jiffies cross-check not U64: {other:?}"),
    };
    anyhow::ensure!(
        jiffies_v != 0,
        "jiffies_64 cross-check returned 0 — symbol-read chain broken \
         upstream (wrong PA from text_kva_to_pa OR read_u64 returned \
         silent zero on bounds-reject); page_offset_base assertion \
         would be vacuous"
    );

    let reply = replies
        .iter()
        .find(|(t, _)| t == TAG_POB)
        .map(|(_, r)| r)
        .ok_or_else(|| anyhow::anyhow!("no reply for tag '{TAG_POB}'"))?;
    anyhow::ensure!(
        reply.success,
        "page_offset_base symbol read FAILED: {} — either the symbol \
         is absent (CONFIG_RANDOMIZE_MEMORY=n; ktstr.kconfig pins =y \
         so this would mean kconfig regressed) OR text_kva_to_pa \
         translation produced an unmapped PA",
        reply.reason
    );
    let pob = match reply.read_values.first() {
        Some(KernelOpValue::U64(v)) => *v,
        other => anyhow::bail!("expected U64, got {other:?}"),
    };
    // 4- vs 5-level paging changes the kaslr_regions[0] base AND
    // the region's entropy range. The kernel picks the appropriate
    // `__PAGE_OFFSET_BASE_L{4,5}` at boot based on the LA57 CPUID
    // bit (arch/x86/include/asm/page_64_types.h); for the entropy
    // upper bound, kaslr_regions[0].size_tb is dynamically set to
    // `1 << (MAX_PHYSMEM_BITS - TB_SHIFT)` (arch/x86/mm/kaslr.c)
    // where MAX_PHYSMEM_BITS = 46 (L4) or 52 (L5). The actual
    // entropy a given boot uses is `remain_entropy / N_regions`
    // bounded by that size — page_offset_base can land anywhere
    // in `[base, base + size_tb*TiB)`.
    //
    // Detection: observed `pob`'s upper byte. 0xff matches L5
    // (`0xff11_0000_0000_0000` base); anything else (in practice
    // 0xffff) falls back to L4 (`0xffff_8880_0000_0000` base).
    const DEFAULT_PAGE_OFFSET_L4: u64 = 0xffff_8880_0000_0000;
    const DEFAULT_PAGE_OFFSET_L5: u64 = 0xff11_0000_0000_0000;
    // kaslr_regions[0].size_tb per paging mode — the upper bound
    // on the entropy added to the base. MAX_PHYSMEM_BITS values
    // are stable kernel constants per arch/x86/include/asm/page_*
    // _types.h. Use the slightly-loose >> instead of `1 << (m - 40)`
    // expressed as TiB to keep the byte arithmetic readable.
    const REGION_SIZE_L4: u64 = 64u64 * (1u64 << 40); // 1 << (46 - 40) = 64 TiB
    const REGION_SIZE_L5: u64 = 4096u64 * (1u64 << 40); // 1 << (52 - 40) = 4096 TiB = 4 PiB
    const PUD_SIZE: u64 = 1 << 30; // 1 GiB
    anyhow::ensure!(
        pob != 0,
        "page_offset_base value == 0 — derivation chain failed entirely"
    );
    let (default_page_offset, region_size) = if (pob >> 56) == 0xff {
        (DEFAULT_PAGE_OFFSET_L5, REGION_SIZE_L5)
    } else {
        (DEFAULT_PAGE_OFFSET_L4, REGION_SIZE_L4)
    };
    anyhow::ensure!(
        (default_page_offset..default_page_offset.wrapping_add(region_size)).contains(&pob),
        "page_offset_base = {pob:#x} outside {default_page_offset:#x}.. + {} TiB range \
         (kaslr_regions[0].size_tb for this paging mode) — picked outside its assigned zone",
        region_size / (1u64 << 40),
    );
    anyhow::ensure!(
        (pob - default_page_offset).is_multiple_of(PUD_SIZE),
        "page_offset_base = {pob:#x}, delta from DEFAULT = {:#x} not \
         PUD-aligned (1 GiB) — kernel guarantees PUD alignment per \
         arch/x86/mm/kaslr.c (& PUD_MASK)",
        pob - default_page_offset
    );
    if pob == default_page_offset {
        eprintln!(
            "WARN: kaslr_page_offset rolled slot 0 (~1/30000 prob); \
             re-roll if reproducible across 3 boots"
        );
    }
    Ok(())
}

// =========================================================================
// kaslr_watch_snapshot_jiffies_64_arms — under KASLR-on default,
// Op::WatchSnapshot("jiffies_64") must NOT return Err from
// arm_user_watchpoint. Under KASLR-on the derivation chain publishes a
// non-zero offset, arm proceeds at the correct KVA, and execute_steps
// returns Ok.
//
// Asserts the arm-success path via execute_steps Ok return + KASLR-on
// precondition. Fire-and-capture round-trip is covered by
// tests/snapshot_e2e.rs's manual-fire pattern.
// =========================================================================

#[ktstr_test(
    scheduler = KTSTR_SCHED,
    duration_s = 2,
    auto_repro = false,
    post_vm = assert_kaslr_for_watch_arm,
)]
fn kaslr_watch_snapshot_jiffies_64_arms(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![Step::new(
        vec![Op::watch_snapshot("jiffies_64")],
        HoldSpec::FULL,
    )];
    execute_steps(ctx, steps)
}

fn assert_kaslr_for_watch_arm(result: &VmResult) -> Result<()> {
    anyhow::ensure!(
        result.kern_kaslr_offset != 0,
        "kern_kaslr_offset == 0 — this test must run under KASLR-on \
         to exercise the slide-aware arm path; the arm-success branch \
         is meaningless when no slide needs to apply"
    );
    Ok(())
}

// =========================================================================
// Negative pin: under `kaslr = false`, Op::WatchSnapshot("jiffies_64")
// MUST return Err from the cold-path execute_steps. The link-time
// `jiffies_64` KVA is in the kernel high-half (>= 0xffff_8000_0000_0000),
// and with `kaslr_offset == 0` the snapshot arm path
// (src/vmm/freeze_coord/snapshot.rs) promotes a silent-misfire to Err
// with a kaslr_offset == 0 + high-half diagnostic.
//
// Without this negative pin, the arm path could regress to warn-then-arm
// and kaslr_watch_snapshot_jiffies_64_arms (positive path) would still
// pass (under KASLR-on the warn-then-arm path also Oks). This test
// gates against that regression.
// =========================================================================

#[ktstr_test(
    scheduler = KTSTR_SCHED,
    duration_s = 2,
    auto_repro = false,
    kaslr = false,
)]
fn kaslr_off_watch_snapshot_jiffies_64_errs(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![Step::new(
        vec![Op::watch_snapshot("jiffies_64")],
        HoldSpec::FULL,
    )];
    // execute_steps wraps step-level errors into Ok(failed_AssertResult)
    // via run_scenario's err→stamped-fail conversion at
    // src/scenario/ops/mod.rs:939-955. Match the canonical pattern
    // pinned in tests/snapshot_e2e.rs (watch_snapshot_op_unresolvable_
    // symbol_bails_immediately): unwrap Ok, assert is_fail(), grep the
    // recorded detail for the typed diagnostic substring.
    let result = execute_steps(ctx, steps)
        .expect("execute_steps returns Ok with stamped error under kaslr=false");
    anyhow::ensure!(
        result.is_fail(),
        "expected stamped failure under kaslr=false for high-half symbol \
         jiffies_64 — snapshot.rs did not promote silent-misfire to Err. \
         Without the promotion the arm path would silently arm at the \
         link-KVA and never fire."
    );
    let messages: Vec<String> = result
        .failure_details()
        .map(|d| d.message.clone())
        .collect();
    anyhow::ensure!(
        messages
            .iter()
            .any(|m| { m.contains("kaslr_offset") || m.contains("kern_virt_kaslr") }),
        "Err message does not cite the kaslr_offset/kern_virt_kaslr \
         diagnostic — the typed Err format in snapshot.rs drifted. \
         Recorded details: {:?}",
        messages,
    );
    Ok(AssertResult::pass())
}
