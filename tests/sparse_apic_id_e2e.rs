//! Regression: the guest's runtime APIC ID must equal the declared
//! (CPUID/MADT) apic_id for a SPARSE topology.
//!
//! ktstr advertises `apic_id(topo, cpu_id)` in CPUID leaf 1/0xB and the MADT,
//! which is SPARSE for non-power-of-2 core/thread counts (gaps in the
//! bit-field encoding). KVM hardwires the in-kernel LAPIC x2apic_id to the KVM
//! `vcpu_id` (arch/x86/kvm/lapic.c `kvm_apic_set_x2apic_id`, read-only,
//! `WARN_ON_ONCE(id != vcpu_id)`), and the guest's runtime apic_id =
//! `read_apic_id()` = that `vcpu_id` (cpu/topology_common.c uses it as
//! authoritative, warning if the CPUID disagrees). So unless ktstr creates
//! each vCPU with `vcpu_id = apic_id(topo, cpu_id)`, the guest runs DENSE
//! apic_ids (= cpu_id) that diverge from the advertised sparse ones: sparse
//! APIC IDs (including any > 255) become unrouteable, AP bringup to the
//! gap/overflow IDs fails, and an MSI/IPI dest never matches the intended
//! vCPU. The fix is `create_vcpu(apic_id(topo, cpu_id))` in
//! `src/vmm/x86_64/kvm.rs`.
//!
//! This pins the invariant on a MINIMAL sparse topology — 2 LLC x 3 core x 1
//! thread = 6 vCPUs; 3 cores need a 2-bit field, so `apic_id = (llc<<2)|core`
//! = {0,1,2,4,5,6} (gap at 3, max 6 > 5) — so it runs on any host without a
//! \>255-vCPU box. The >255 ext-dest path itself is exercised by the
//! `wide_smp_*_irq_e2e` tests on a >=252-CPU host; this is the CI-runnable
//! root-cause guard. `/proc/cpuinfo` exposes both `apicid` (runtime,
//! `read_apic_id()`) and `initial apicid` (CPUID leaf 1) per CPU; the fix
//! makes them equal.
//!
//! Run: cargo run --bin cargo-ktstr -- ktstr test --kernel ../linux \
//!        -- -E 'test(sparse_topology_runtime_apicid_matches_declared)' \
//!        --success-output immediate

use anyhow::{Result, ensure};
use ktstr::assert::AssertResult;
use ktstr::ktstr_test;
use ktstr::scenario::Ctx;
use std::collections::BTreeSet;
use std::fs::read_to_string;

/// Parse `/proc/cpuinfo` into `(runtime apicid, initial apicid)` pairs, one
/// per processor block. `apicid` is `read_apic_id()` (the in-kernel LAPIC ID
/// an MSI/IPI resolves against); `initial apicid` is the CPUID leaf-1 value
/// ktstr advertised.
fn read_apicids() -> Result<Vec<(u32, u32)>> {
    let text = read_to_string("/proc/cpuinfo")?;
    let mut pairs = Vec::new();
    let mut apicid: Option<u32> = None;
    let mut initial: Option<u32> = None;
    let mut flush = |apicid: &mut Option<u32>, initial: &mut Option<u32>| {
        if let (Some(a), Some(i)) = (apicid.take(), initial.take()) {
            pairs.push((a, i));
        }
    };
    for line in text.lines() {
        if line.trim().is_empty() {
            // Blank line separates processor blocks.
            flush(&mut apicid, &mut initial);
            continue;
        }
        let Some((key, val)) = line.split_once(':') else {
            continue;
        };
        match key.trim() {
            "apicid" => apicid = Some(val.trim().parse()?),
            "initial apicid" => initial = Some(val.trim().parse()?),
            _ => {}
        }
    }
    // Flush a trailing block with no terminating blank line.
    flush(&mut apicid, &mut initial);
    Ok(pairs)
}

#[ktstr_test(llcs = 2, cores = 3, threads = 1, no_perf_mode, duration_s = 4)]
fn sparse_topology_runtime_apicid_matches_declared(ctx: &Ctx) -> Result<AssertResult> {
    let total = ctx.topo.total_cpus();
    ensure!(
        total == 6,
        "expected 6 vCPUs (2 LLC x 3 core x 1 thread), got {total}"
    );

    let pairs = read_apicids()?;

    // All vCPUs must come online. Pre-fix, the BSP's INIT-SIPI to a sparse
    // MADT apic_id (e.g. 6) matches no LAPIC (KVM x2apic_ids are dense
    // {0..5}), so that AP never boots and the count is short.
    ensure!(
        pairs.len() == total,
        "all {total} vCPUs must report an apicid in /proc/cpuinfo, got {} — \
         sparse APIC IDs unrouteable, so AP bringup to the gap/overflow IDs failed",
        pairs.len()
    );

    let declared: BTreeSet<u32> = pairs.iter().map(|(_, i)| *i).collect();
    let max_declared = declared.iter().copied().max().unwrap_or(0);

    // Guard: the topology must be genuinely SPARSE, else the invariant below
    // holds trivially (dense apic_id == cpu_id) and the test is vacuous. The
    // 2x3x1 encoding declares a max apic_id of 6 over 6 CPUs, so a declared
    // value > total-1 proves the bit-field gap is present.
    ensure!(
        max_declared > (total as u32) - 1,
        "topology is not sparse (max declared apic_id {max_declared} <= {}); \
         the test would be vacuous — pick a non-power-of-2 core/thread count",
        (total as u32) - 1
    );

    // The invariant: each vCPU's runtime apic_id (what an MSI/IPI resolves
    // against) must equal the declared CPUID apic_id. With vcpu_id=cpu_id the
    // runtime set is dense {0..5} while declared is {0,1,2,4,5,6}.
    eprintln!(
        "SPARSE_APICID total={total} pairs={pairs:?} declared={declared:?} max_declared={max_declared}"
    );
    for (apicid, initial) in &pairs {
        ensure!(
            apicid == initial,
            "a vCPU's runtime apic_id {apicid} != its declared CPUID apic_id {initial}: \
             the in-kernel LAPIC ID (KVM vcpu_id) diverges from the advertised apic_id, \
             so sparse (incl. >255) APIC IDs are unrouteable"
        );
    }

    Ok(AssertResult::pass())
}
