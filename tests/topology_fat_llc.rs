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

/// Count the CPUs named by a Linux cpulist string ("0-3", "0,2,4", "0-3,8").
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

#[ktstr_test(llcs = 4, cores = 4, threads = 1, no_perf_mode, duration_s = 4)]
fn multi_llc_guest_presents_fat_l3(ctx: &Ctx) -> Result<AssertResult> {
    let num_llcs = ctx.topo.num_llcs();
    let total = ctx.topo.total_cpus();
    ensure!(num_llcs >= 2, "test needs >=2 LLCs (got {num_llcs})");
    ensure!(
        total % num_llcs == 0,
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
