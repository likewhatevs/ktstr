//! Shared test helpers for `monitor` submodules.
//!
//! Helpers that two or more sibling test modules need — e.g.
//! [`name_from_str`] used by both `bpf_map::tests` and
//! `dump::tests` — live here to avoid duplicate copies that drift.
//! `#[cfg(test)] pub(crate)` so descendants of `crate::monitor` can
//! reach in (siblings cannot see each other's private `tests`
//! modules; a shared parent module is the only path).

use super::bpf_map::BPF_OBJ_NAME_LEN;
use super::{CpuSnapshot, MonitorSample};

/// Pack a `&str` into the inline name representation
/// (`name_bytes`, `name_len`) used by
/// [`super::bpf_map::BpfMapInfo`]. Truncates to
/// `BPF_OBJ_NAME_LEN` when the input exceeds that — matches the
/// kernel's own bookkeeping (`bpf_obj_name_cpy` in
/// `kernel/bpf/syscall.c` rejects names longer than the field, but
/// for tests we silently truncate so call sites can use whatever
/// length is convenient without precomputing the cap).
pub(crate) fn name_from_str(s: &str) -> ([u8; BPF_OBJ_NAME_LEN], u8) {
    let mut buf = [0u8; BPF_OBJ_NAME_LEN];
    let bytes = s.as_bytes();
    let n = bytes.len().min(BPF_OBJ_NAME_LEN);
    buf[..n].copy_from_slice(&bytes[..n]);
    (buf, n as u8)
}

/// Two-CPU sample whose load is balanced under every default
/// monitor check: `nr_running=2` on both CPUs, small
/// `local_dsq_depth`, advancing `rq_clock`. Used as the
/// "no-violation baseline" by `thresholds_tests`,
/// `validity_tests`, `event_rates_tests`, and
/// `schedstat_tests` — keeping it in one place avoids drift
/// between copies that would silently change what "balanced"
/// means across test modules.
pub(crate) fn balanced_sample(elapsed_ms: u64, clock_base: u64) -> MonitorSample {
    MonitorSample {
        prog_stats: None,
        psi_irq: None,
        elapsed_ms,
        cpus: vec![
            CpuSnapshot {
                nr_running: 2,
                rq_clock: clock_base,
                local_dsq_depth: 3,
                ..Default::default()
            },
            CpuSnapshot {
                nr_running: 2,
                rq_clock: clock_base + 100,
                local_dsq_depth: 2,
                ..Default::default()
            },
        ],
    }
}
