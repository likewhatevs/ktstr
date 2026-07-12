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
use ktstr::test_support::{Scheduler, SchedulerSpec};
use std::fs::read_to_string;

#[path = "common/cpulist.rs"]
mod cpulist;
use cpulist::count_cpulist;

/// scx-ktstr as the boot scheduler for the wide-SMP scheduler-path test.
const WIDE_SMP_SCHED: Scheduler =
    Scheduler::named("wide_smp_sched").binary(SchedulerSpec::Discover("scx-ktstr"));

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
    eprintln!(
        "WIDE_SMP total_cpus={total} online='{}' n_online={n_online}",
        online.trim()
    );
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
/// oversubscription on a host with >= 64 allowed CPUs; on a host with < 64
/// allowed CPUs this `cpu_budget` is a `CpuBudgetUnsatisfiable` hard error
/// (`resolve_cpu_budget`), so the test only runs where >= 64 CPUs are
/// allowed). This pins that the `cpu_budget` knob does
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
#[ktstr_test(
    llcs = 16,
    cores = 16,
    threads = 1,
    no_perf_mode,
    cpu_budget = 64,
    duration_s = 4
)]
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

/// The same 256-vCPU wide-SMP guest as
/// [`wide_smp_guest_boots_all_cpus_online`], but booted UNDER A SCHEDULER
/// (scx-ktstr) to pin the scheduler-path boot-ordering invariant that
/// `CONFIG_HOTPLUG_PARALLEL` must not perturb: PID-1 init gates the scheduler
/// spawn on `all_possible_cpus_online()` (the AP-gap check that precedes
/// Phase 3 `start_scheduler` in `src/vmm/rust_init/init.rs`), so a scheduler
/// only ever attaches after every possible CPU is online.
///
/// The proof is by construction. For this body to run at all, the guest must
/// have booted through Phase 3 with the scheduler attached — and the AP-gap
/// gate would have PANICked the guest before Phase 3 (surfacing host-side as a
/// crash, failing the run) had any possible CPU been offline. So reaching the
/// body under a bound scheduler already witnesses "all CPUs online BEFORE
/// scheduler spawn". The body then asserts the guest's `online` set still
/// equals `possible` at scenario time, making the invariant explicit rather
/// than only implied. Parallel AP bring-up changes only HOW FAST the APs
/// online, so this must hold identically with `HOTPLUG_PARALLEL` active.
///
/// `no_perf_mode` + `cpu_budget = 64` mirror the overcommit sibling so the
/// 256-vCPU guest runs on hosts that cannot 1:1-pin; the scheduler runs under
/// that oversubscription.
///
/// Run: cargo run --bin cargo-ktstr -- ktstr test --kernel ../linux \
///        -- -E 'test(wide_smp_scheduler_observes_all_cpus_online)' \
///        --success-output immediate
#[ktstr_test(
    scheduler = WIDE_SMP_SCHED,
    llcs = 16,
    cores = 16,
    threads = 1,
    no_perf_mode,
    cpu_budget = 64,
    duration_s = 5,
    watchdog_timeout_s = 60,
    auto_repro = false
)]
fn wide_smp_scheduler_observes_all_cpus_online(ctx: &Ctx) -> Result<AssertResult> {
    let total = ctx.topo.total_cpus();
    ensure!(
        total > 254,
        "test must exceed the 254 xAPIC limit to exercise wide-SMP bring-up (got {total})"
    );

    // Compare the kernel's `online` set against its `possible` set directly:
    // every possible CPU must be online. Reaching here under a bound scheduler
    // already proves the AP-gap gate (which precedes scheduler spawn) passed;
    // this makes the all-online state explicit at scenario time.
    let possible = read_to_string("/sys/devices/system/cpu/possible")
        .map_err(|e| anyhow::anyhow!("read /sys/devices/system/cpu/possible: {e}"))?;
    let online = read_to_string("/sys/devices/system/cpu/online")
        .map_err(|e| anyhow::anyhow!("read /sys/devices/system/cpu/online: {e}"))?;
    let n_possible = count_cpulist(possible.trim());
    let n_online = count_cpulist(online.trim());
    eprintln!(
        "WIDE_SMP_SCHED total_cpus={total} possible='{}' online='{}' \
         n_possible={n_possible} n_online={n_online}",
        possible.trim(),
        online.trim()
    );
    ensure!(
        n_online == n_possible && n_online == total,
        "expected all {total} vCPUs online under a scheduler (possible={n_possible}, \
         online={n_online}) — the scheduler-spawn gate on all_possible_cpus_online \
         did not hold",
    );
    Ok(AssertResult::pass())
}
