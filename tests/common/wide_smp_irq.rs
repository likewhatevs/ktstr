//! Shared helpers for >255-APIC-ID device-IRQ e2e tests (wide-SMP).
//!
//! A wide/sparse topology mints APIC IDs above 255 (the MSI ext-dest-id
//! threshold); these helpers pin a device's IRQ to such a vCPU and read
//! its per-CPU interrupt count to prove the >255 destination route
//! actually delivered. Shared by `wide_smp_device_irq_e2e.rs` (virtio-blk)
//! and `wide_smp_net_irq_e2e.rs` (virtio-net) — the userspace-IOAPIC +
//! KVM_SET_GSI_ROUTING path is device-agnostic, so only each device's
//! IRQ-discovery + drive differ; the APIC-ID/IRQ-count scaffolding is one.
//!
//! `#[path]`-included (not a `mod common` tree) so each test pulls in only
//! this file, matching the `common/cpulist.rs` convention.

use anyhow::{Result, bail};
use std::fs;

/// Parse `/proc/cpuinfo` into `(processor_number, apic_id)` pairs. The
/// per-CPU interrupt-count column in `/proc/interrupts` is indexed by the
/// Linux processor number, while the ext-dest path is selected by APIC ID
/// — under the sparse encoding the two differ — so callers map between
/// them here.
#[allow(dead_code)]
pub fn cpu_apicids() -> Result<Vec<(usize, u32)>> {
    let text = fs::read_to_string("/proc/cpuinfo")?;
    let mut out = Vec::new();
    let mut cur_cpu: Option<usize> = None;
    for line in text.lines() {
        let Some((key, val)) = line.split_once(':') else {
            continue;
        };
        match key.trim() {
            "processor" => cur_cpu = val.trim().parse().ok(),
            "apicid" => {
                if let (Some(cpu), Ok(apic)) = (cur_cpu, val.trim().parse::<u32>()) {
                    out.push((cpu, apic));
                }
            }
            _ => {}
        }
    }
    Ok(out)
}

/// A `(processor_number, apic_id)` pair whose APIC ID exceeds 255 (the MSI
/// ext-dest-id threshold), or an error naming the max APIC ID seen.
#[allow(dead_code)]
pub fn find_apic_above_255() -> Result<(usize, u32)> {
    let apicids = cpu_apicids()?;
    apicids
        .iter()
        .copied()
        .find(|&(_, apic)| apic >= 256)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "no vCPU with APIC ID >= 256 (max seen {}); the topology did \
                 not mint a >255 APIC ID, so the ext-dest path cannot be exercised",
                apicids.iter().map(|&(_, a)| a).max().unwrap_or(0)
            )
        })
}

/// The Linux IRQ number whose `/proc/interrupts` action name (the last
/// whitespace token on the line) equals `action`. Device IRQs are named
/// after the owning device (e.g. `virtio0`), so callers pass the device
/// basename rather than hardcoding a GSI.
#[allow(dead_code)]
pub fn device_irq_by_action_name(action: &str) -> Result<u32> {
    let irqs = fs::read_to_string("/proc/interrupts")?;
    for line in irqs.lines() {
        let Some((lhs, rhs)) = line.split_once(':') else {
            continue;
        };
        if rhs.split_whitespace().last() == Some(action) {
            return lhs
                .trim()
                .parse::<u32>()
                .map_err(|e| anyhow::anyhow!("non-numeric IRQ '{}' for {action}: {e}", lhs.trim()));
        }
    }
    bail!("no /proc/interrupts line with action name {action}")
}

/// Pin `irq` to Linux processor `cpu` via `smp_affinity_list`. Under
/// x2APIC physical mode this programs the IOAPIC RTE with that CPU's exact
/// APIC ID in physical dest_mode (no lowest-priority redistribution), so a
/// single-CPU pin to an APIC ID >= 256 forces the >255 ext-dest encoding.
#[allow(dead_code)]
pub fn pin_irq_to_cpu(irq: u32, cpu: usize) -> Result<()> {
    fs::write(format!("/proc/irq/{irq}/smp_affinity_list"), cpu.to_string())
        .map_err(|e| anyhow::anyhow!("pin irq {irq} to cpu {cpu}: {e}"))
}

/// Per-CPU interrupt count for `irq` on Linux processor `cpu`, from
/// `/proc/interrupts`. x86_64 layout (no Edge/Level column — that field is
/// gated on CONFIG_GENERIC_IRQ_SHOW_LEVEL, which x86 does not select):
///   `<IRQ>: <c0> <c1> ... <cN-1>  <chip>  <hwirq>  <action>`
/// The first N tokens after `:` are the per-online-CPU counts indexed by
/// processor number; the chip name follows. Callers pin all vCPUs online,
/// so `tokens[cpu]` is a count, never the chip name.
#[allow(dead_code)]
pub fn irq_count(irq: u32, cpu: usize) -> Result<u64> {
    let irqs = fs::read_to_string("/proc/interrupts")?;
    let prefix = format!("{irq}:");
    for line in irqs.lines() {
        if line.trim_start().starts_with(&prefix) {
            let rhs = line.split_once(':').unwrap().1;
            let tokens: Vec<&str> = rhs.split_whitespace().collect();
            let tok = tokens.get(cpu).ok_or_else(|| {
                anyhow::anyhow!("cpu {cpu} column missing for irq {irq} (line: {line:?})")
            })?;
            return tok.parse::<u64>().map_err(|e| {
                anyhow::anyhow!("count column {cpu} for irq {irq} not numeric ('{tok}'): {e}")
            });
        }
    }
    bail!("irq {irq} not found in /proc/interrupts")
}
