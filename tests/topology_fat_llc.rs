//! Coverage for the guest LLC topology contract: a multi-LLC VM must
//! present FAT shared-L3 domains — each LLC's level-3 cache shared by all
//! `cores_per_llc` CPUs of that LLC, with exactly `num_llcs` distinct L3
//! domains. (Schedulers read `/sys/.../cache/index3/shared_cpu_list` to
//! build their LLC map; a per-CPU L3 makes every CPU its own LLC.)
//!
//! Runs under the default kernel scheduler (no scx attach) so it exercises
//! ktstr's CPUID topology generation directly.
//!
//! Run: `cargo run --bin cargo-ktstr -- ktstr test --kernel ../linux \
//!        -- -E 'test(multi_llc_guest_presents_fat_l3)' --success-output immediate`

use anyhow::{Result, ensure};
use ktstr::assert::AssertResult;
use ktstr::ktstr_test;
use ktstr::scenario::Ctx;
use std::collections::BTreeSet;
use std::fs::read_to_string;

#[path = "common/cpulist.rs"]
mod cpulist;
use cpulist::count_cpulist;

/// Assert the guest presents fat shared-L3 domains: exactly `num_llcs`
/// distinct level-3 `shared_cpu_list` domains, each shared by all
/// `cores_per_llc` CPUs of its LLC. (Schedulers read index3/shared_cpu_list
/// to build their LLC map; a per-CPU L3 makes every CPU its own LLC.)
fn assert_fat_l3_presentation(ctx: &Ctx) -> Result<AssertResult> {
    let num_llcs = ctx.topo.num_llcs();
    let total = ctx.topo.total_cpus();
    ensure!(num_llcs >= 2, "test needs >=2 LLCs (got {num_llcs})");
    ensure!(
        total.is_multiple_of(num_llcs),
        "uneven topology: {total} cpus / {num_llcs} llcs"
    );
    let cores_per_llc = total / num_llcs;

    // Collect each CPU's level-3 cache shared_cpu_list.
    let mut l3_groups: BTreeSet<String> = BTreeSet::new();
    let mut details = String::new();
    for cpu in 0..total {
        let mut l3 = None;
        for idx in 0..8 {
            let dir = format!("/sys/devices/system/cpu/cpu{cpu}/cache/index{idx}");
            let Ok(level) = read_to_string(format!("{dir}/level")) else {
                break;
            };
            if level.trim() == "3" {
                let scl = read_to_string(format!("{dir}/shared_cpu_list")).unwrap_or_default();
                l3 = Some(scl.trim().to_string());
                break;
            }
        }
        let scl = l3.ok_or_else(|| anyhow::anyhow!("cpu{cpu} has no level-3 cache in /sys"))?;
        details.push_str(&format!("cpu{cpu} L3 shared_cpu_list={scl}\n"));
        l3_groups.insert(scl);
    }

    eprintln!(
        "FATLLC num_llcs={num_llcs} total_cpus={total} cores_per_llc={cores_per_llc} \
         distinct_L3_domains={}\n{details}",
        l3_groups.len()
    );

    ensure!(
        l3_groups.len() == num_llcs,
        "expected {num_llcs} distinct L3 shared_cpu_list domains (one per LLC), got {} — \
         the guest is not presenting fat shared L3 (per-CPU L3 means every CPU is its own LLC)",
        l3_groups.len()
    );
    for g in &l3_groups {
        let n = count_cpulist(g);
        ensure!(
            n == cores_per_llc,
            "L3 domain '{g}' is shared by {n} CPUs, expected {cores_per_llc} (cores_per_llc)"
        );
    }
    Ok(AssertResult::pass())
}

#[ktstr_test(llcs = 4, cores = 4, threads = 1, no_perf_mode, duration_s = 4)]
fn multi_llc_guest_presents_fat_l3(ctx: &Ctx) -> Result<AssertResult> {
    assert_fat_l3_presentation(ctx)
}

/// The same fat-L3 contract at the wide-SMP scale: a 256-vCPU guest
/// (16 LLCs x 16 cores, top APIC ID 255 > 254 -> split-irqchip boot path).
/// Pins that the host-synthesized cache-topology CPUID presents 16 fat-L3
/// domains, each shared by all 16 CPUs of its LLC, at 256 vCPUs.
///
/// The >255-unique coverage is the boot and the grouping, NOT the cache
/// synthesis: `num_threads_sharing` is encoded purely from cores-per-LLC
/// (x86_64/topology/mod.rs `patch_cache_topology_eax`), independent of vCPU count, so
/// a per-CPU-collapse or mis-size regression is already caught by the
/// 16-vCPU sibling. This variant adds (a) all 256 APs coming online — the
/// per-CPU /sys scan reads every CPU's L3 only if the split-irqchip boot
/// brought them all up — and (b) the guest's `llc_id = apicid >> order(..)`
/// grouping holding across the upper-half (128-255) APIC-ID range the
/// sibling (APIC IDs 0-15) never reaches, where an apic-id-width or
/// truncation bug would surface.
#[ktstr_test(llcs = 16, cores = 16, threads = 1, no_perf_mode, duration_s = 4)]
fn multi_llc_guest_presents_fat_l3_wide_smp(ctx: &Ctx) -> Result<AssertResult> {
    ensure!(
        ctx.topo.total_cpus() > 254,
        "wide-smp variant needs >254 vCPUs to exercise the split-irqchip \
         cache presentation (got {})",
        ctx.topo.total_cpus()
    );
    assert_fat_l3_presentation(ctx)
}
