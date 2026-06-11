//! End-to-end: a >254-APIC-ID guest (the split-irqchip / userspace-IOAPIC
//! path) boots fully and brings every vCPU online.
//!
//! This exercises the wide-SMP stack at the xAPIC ceiling:
//!   - x2APIC AP bring-up (required once any APIC ID exceeds 254; this
//!     guest's top APIC ID is 255), and
//!   - the userspace IOAPIC + `KVM_SET_GSI_ROUTING` MSI routing that delivers
//!     the virtio / serial device IRQs the guest needs to boot.
//!
//! `KVM_FEATURE_MSI_EXT_DEST_ID` is advertised, but this guest's top APIC ID
//! is 255 (fits the 8-bit MSI destination), so the ext-dest-id path for IDs
//! >255 is not exercised by this boot test.
//!
//! The proof is end-to-end by construction: the test body itself runs *inside
//! the guest* (it reads the guest's `/sys`). For the body to run at all, the
//! guest must have booted to userspace over the virtio console / block path —
//! which means the userspace IOAPIC routed those device IRQs. If the IOAPIC
//! routing were broken the virtio/serial IRQs would never fire and the guest
//! would hang in boot, so reaching the assertions is itself the signal.
//!
//! Run: cargo run --bin cargo-ktstr -- ktstr test --kernel ../linux \
//!        -- -E 'test(wide_smp_guest_boots_all_cpus_online)' \
//!        --success-output immediate

use anyhow::{Result, ensure};
use ktstr::assert::AssertResult;
use ktstr::ktstr_test;
use ktstr::scenario::Ctx;
use std::fs::read_to_string;

/// Count the CPUs named by a Linux cpulist string ("0-255", "0,2,4", "0-3,8").
fn count_cpulist(s: &str) -> usize {
    s.split(',')
        .filter(|p| !p.trim().is_empty())
        .map(|p| match p.trim().split_once('-') {
            Some((a, b)) => {
                let a: usize = a.trim().parse().unwrap_or(0);
                let b: usize = b.trim().parse().unwrap_or(0);
                b.saturating_sub(a) + 1
            }
            None => 1,
        })
        .sum()
}

// 16 LLCs x 16 cores x 1 thread = 256 vCPUs. The max APIC ID is
// (15 << 4) | 15 = 255 > 254 (MAX_XAPIC_ID), so this guest takes the
// split-irqchip + userspace-IOAPIC path — the wide-SMP (>254-APIC-ID)
// machinery under test. Memory is omitted: the cpus*64 floor (256 * 64 =
// 16 GiB) dominates, which also crosses the sub-4GB MMIO gap and exercises
// the relocate.
#[ktstr_test(llcs = 16, cores = 16, threads = 1, no_perf_mode, duration_s = 4)]
fn wide_smp_guest_boots_all_cpus_online(ctx: &Ctx) -> Result<AssertResult> {
    let total = ctx.topo.total_cpus();
    ensure!(
        total > 254,
        "test must exceed the 254 xAPIC limit to exercise split-irqchip (got {total})"
    );

    // Every vCPU must be online. AP bring-up above APIC ID 255 needs x2APIC
    // (chunk A); a broken device-IRQ path (chunk B) would hang boot before the
    // guest ever schedules this payload, so reaching here already proves the
    // userspace IOAPIC delivered the virtio/serial IRQs.
    let online = read_to_string("/sys/devices/system/cpu/online")
        .map_err(|e| anyhow::anyhow!("read /sys/devices/system/cpu/online: {e}"))?;
    let n_online = count_cpulist(online.trim());
    eprintln!("WIDE_SMP total_cpus={total} online='{}' n_online={n_online}", online.trim());
    ensure!(
        n_online == total,
        "expected all {total} vCPUs online, got {n_online} (online='{}')",
        online.trim()
    );
    Ok(AssertResult::pass())
}

/// The same 256-vCPU, >254-APIC-ID guest as
/// [`wide_smp_guest_boots_all_cpus_online`], booted with `cpu_budget = 64`:
/// the no-perf path masks all 256 vCPU threads onto 64 host CPUs (4x
/// oversubscription on a host with >= 64 allowed CPUs; clamped to the host
/// allowance on smaller hosts). This pins that the `cpu_budget` knob does
/// not BREAK wide-SMP boot under a constrained host mask — every vCPU still
/// comes online despite the oversubscription (an oversubscription-wedged AP
/// bring-up would leave vCPUs offline or hang). It does not itself observe
/// the mask cardinality — the guest can't read host affinity, so n_online
/// is 256 whether the mask is 64 or 256; the mask APPLICATION is covered
/// host-side (`builder_cpu_budget_setter` + the `KtstrTestEntry::validate`
/// tests). `cpu_budget` requires `no_perf_mode` (it sizes the no-perf
/// shared vCPU mask), so both attributes are set.
///
/// Run: cargo run --bin cargo-ktstr -- ktstr test --kernel ../linux \
///        -- -E 'test(wide_smp_guest_boots_all_cpus_online_overcommit)' \
///        --success-output immediate
#[ktstr_test(llcs = 16, cores = 16, threads = 1, no_perf_mode, cpu_budget = 64, duration_s = 4)]
fn wide_smp_guest_boots_all_cpus_online_overcommit(ctx: &Ctx) -> Result<AssertResult> {
    let total = ctx.topo.total_cpus();
    ensure!(
        total > 254,
        "test must exceed the 254 xAPIC limit to exercise split-irqchip (got {total})"
    );

    // Every vCPU must come online even though the 256 vCPU threads share
    // only `cpu_budget` host CPUs. If the oversubscription wedged AP
    // bring-up, vCPUs would be offline or the boot would hang before this
    // payload runs. (A silently-dropped budget is not caught here — it
    // would still boot all-online; mask application is covered host-side.)
    let online = read_to_string("/sys/devices/system/cpu/online")
        .map_err(|e| anyhow::anyhow!("read /sys/devices/system/cpu/online: {e}"))?;
    let n_online = count_cpulist(online.trim());
    eprintln!(
        "WIDE_SMP_OVERCOMMIT total_cpus={total} online='{}' n_online={n_online}",
        online.trim()
    );
    ensure!(
        n_online == total,
        "expected all {total} vCPUs online under cpu_budget overcommit, got \
         {n_online} (online='{}')",
        online.trim()
    );
    Ok(AssertResult::pass())
}
