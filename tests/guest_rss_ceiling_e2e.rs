//! Directional guard for the guest-RAM THP change (W2e).
//!
//! `numa_mem::map_and_register_regions` hints the anonymous guest-RAM
//! regions with `MADV_HUGEPAGE` so they self-serve 2 MiB transparent
//! hugepages under the host's default `THP=madvise` policy. THP only
//! materializes a 2 MiB page for a fault inside an already-touched 2 MiB
//! window, so it changes page SIZE, not the set of touched pages — a small
//! idle guest should not balloon host residency.
//!
//! This test pins that directionally. nextest runs each `#[ktstr_test]` in
//! its own process, so the orchestrator process that hosted the guest owns a
//! `VmHWM` (peak RSS) that includes the faulted-in guest pages. After the run
//! the `post_vm` callback reads its own `/proc/self/status` and asserts the
//! peak stayed under a generous ceiling.
//!
//! The bound is intentionally loose: the host peak is dominated by the
//! one-shot vmlinux ELF + BTF parse (a debug vmlinux is tens-to-hundreds of
//! MB), not by the 512 MiB guest, so this is a catastrophic-over-
//! materialization guard (e.g. a regression that faulted the entire guest
//! RAM into 2 MiB pages plus large slack), not a precise guest-RSS
//! measurement — the harness exposes no host-VMM-RSS hook for a tighter one.
//!
//! Run: cargo run --bin cargo-ktstr -- ktstr test --kernel ../linux \
//!        -- -E 'test(small_idle_guest_host_rss_stays_bounded)' \
//!        --success-output immediate

use anyhow::{Context, Result, ensure};
use ktstr::assert::AssertResult;
use ktstr::ktstr_test;
use ktstr::prelude::VmResult;
use ktstr::scenario::Ctx;

/// Guest RAM for the idle guest under test.
const GUEST_MIB: u64 = 512;

/// Generous peak-RSS ceiling (KiB): 4x the guest RAM plus 4 GiB of fixed
/// host overhead (vmlinux/BTF parse, monitor, allocator). A regression that
/// over-materializes guest RAM well beyond what an idle boot touches trips
/// this; a normal run sits far below it.
const RSS_CEILING_KB: u64 = GUEST_MIB * 1024 * 4 + 4 * 1024 * 1024;

/// Parse the `VmHWM:` (peak resident set size) line from a
/// `/proc/<pid>/status` blob, returning kilobytes. `None` when the line is
/// absent or malformed. Pure over its input so it is unit-testable without a
/// live process.
fn parse_vm_hwm_kb(status: &str) -> Option<u64> {
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            // Format: "VmHWM:\t   12345 kB"
            return rest.split_whitespace().next()?.parse().ok();
        }
    }
    None
}

/// Host-side gate: the orchestrator process's peak RSS (which includes the
/// guest pages faulted in during the run) stayed under the generous ceiling.
fn assert_host_rss_under_ceiling(result: &VmResult) -> Result<()> {
    ensure!(
        result.success,
        "guest run did not succeed (timed_out={}, exit_code={}, crash={:?}); \
         RSS ceiling is only meaningful on a clean boot",
        result.timed_out,
        result.exit_code,
        result.crash_message,
    );
    let status =
        std::fs::read_to_string("/proc/self/status").context("read /proc/self/status")?;
    let hwm_kb = parse_vm_hwm_kb(&status)
        .context("no parseable VmHWM in /proc/self/status")?;
    eprintln!("GUEST_RSS host VmHWM={hwm_kb} kB ceiling={RSS_CEILING_KB} kB");
    ensure!(
        hwm_kb < RSS_CEILING_KB,
        "host peak RSS {hwm_kb} kB exceeded the {RSS_CEILING_KB} kB ceiling for a \
         {GUEST_MIB} MiB idle guest — the guest-RAM MADV_HUGEPAGE hint may be \
         over-materializing residency",
    );
    Ok(())
}

/// Boots a small idle guest and asserts (host-side) that peak RSS stayed
/// bounded. `no_perf_mode` so it runs on any host without a 1:1-pin
/// requirement; no scheduler (default in-kernel EEVDF) keeps the run minimal.
#[ktstr_test(
    llcs = 1,
    cores = 1,
    threads = 1,
    memory_mib = 512,
    no_perf_mode,
    duration_s = 2,
    watchdog_timeout_s = 30,
    auto_repro = false,
    post_vm = assert_host_rss_under_ceiling,
)]
fn small_idle_guest_host_rss_stays_bounded(ctx: &Ctx) -> Result<AssertResult> {
    let _ = ctx;
    Ok(AssertResult::pass())
}

#[test]
fn parse_vm_hwm_kb_extracts_peak() {
    let status = "Name:\tktstr\nVmPeak:\t 9999999 kB\nVmHWM:\t   123456 kB\nVmRSS:\t   65432 kB\n";
    assert_eq!(parse_vm_hwm_kb(status), Some(123456));
}

#[test]
fn parse_vm_hwm_kb_absent_is_none() {
    let status = "Name:\tktstr\nVmRSS:\t   65432 kB\n";
    assert_eq!(parse_vm_hwm_kb(status), None);
}

#[test]
fn parse_vm_hwm_kb_malformed_is_none() {
    assert_eq!(parse_vm_hwm_kb("VmHWM:\t   notanumber kB\n"), None);
}
