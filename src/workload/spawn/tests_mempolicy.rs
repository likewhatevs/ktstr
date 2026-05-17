//! Spawn-pipeline tests — mempolicy group.

#![cfg(test)]
#![allow(unused_imports)]

use super::super::affinity::*;
use super::super::config::*;
use super::super::types::*;
use super::super::worker::*;
use super::testing::*;
use super::*;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

#[test]
fn build_nodemask_empty() {
    let (mask, maxnode) = build_nodemask(&BTreeSet::new());
    assert!(mask.is_empty());
    assert_eq!(maxnode, 0);
}
#[test]
fn build_nodemask_single() {
    let (mask, maxnode) = build_nodemask(&[0].into_iter().collect());
    // kernel get_nodes() does --maxnode, so maxnode = max_node + 2
    assert_eq!(maxnode, 2);
    assert_eq!(mask.len(), 1);
    assert_eq!(mask[0], 1);
}
#[test]
fn build_nodemask_multiple() {
    let (mask, maxnode) = build_nodemask(&[0, 2].into_iter().collect());
    assert_eq!(maxnode, 4); // max_node=2, +2 = 4
    assert_eq!(mask[0] & 1, 1); // node 0
    assert_eq!(mask[0] & 4, 4); // node 2
    assert_eq!(mask[0] & 2, 0); // node 1 not set
}
#[test]
fn build_nodemask_high_node() {
    let bits_per_word = std::mem::size_of::<libc::c_ulong>() * 8;
    let high = bits_per_word + 3;
    let (mask, maxnode) = build_nodemask(&[high].into_iter().collect());
    assert_eq!(maxnode, (high + 2) as libc::c_ulong);
    assert_eq!(mask.len(), 2);
    assert_eq!(mask[0], 0);
    assert_eq!(mask[1], 1 << 3);
}
#[test]
fn apply_mempolicy_default_is_noop() {
    apply_mempolicy_with_flags(&MemPolicy::Default, MpolFlags::NONE);
}
#[test]
fn apply_mempolicy_empty_bind_skipped() {
    apply_mempolicy_with_flags(&MemPolicy::Bind(BTreeSet::new()), MpolFlags::NONE);
}
#[test]
fn apply_mempolicy_empty_interleave_skipped() {
    apply_mempolicy_with_flags(&MemPolicy::Interleave(BTreeSet::new()), MpolFlags::NONE);
}
/// `WorkType::NumaWorkingSetSweep` smoke test. Empty
/// `target_nodes` disables binding (per the variant's doc:
/// "Empty list disables binding ... no migration is
/// triggered"); the worker still touches the region every
/// iteration. Sufficient for the pathology smoke check —
/// real multi-node migration tests live under
/// `tests/numa_tests.rs` (see #143/#146).
#[test]
fn pathology_numa_working_set_sweep_iterates() {
    let cfg = WorkloadConfig {
        num_workers: 2,
        work_type: WorkType::NumaWorkingSetSweep {
            region_kib: 256,
            sweep_period_ms: 100,
            target_nodes: vec![],
        },
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&cfg).expect("NumaWorkingSetSweep must spawn");
    h.start();
    std::thread::sleep(Duration::from_millis(200));
    let reports = h.stop_and_collect();
    assert_eq!(reports.len(), 2);
    for r in &reports {
        assert!(
            r.iterations > 0,
            "NumaWorkingSetSweep worker must iterate: {r:?}"
        );
    }
}

/// Pins that [`WorkloadHandle::spawn`] invokes
/// [`WorkloadConfig::validate`] before any worker setup. A regression
/// that drops the `config.validate()?` call at the top of
/// [`WorkloadHandle::spawn`] would silently restore the pre-validate
/// behavior: unit tests on [`WorkloadConfig::validate`] still pass
/// (they call the validator directly), but production `spawn(&bad)`
/// would silently coast through to `apply_mempolicy_with_flags`'s
/// permissive silent-skip arm. Anchor on the validate-wrap prefix
/// ("WorkloadConfig.mem_policy") to prove `spawn` called the
/// structural gate, not some other reject site downstream.
///
/// `WorkloadHandle` does not derive `Debug` (intentional — wrapping
/// raw fds + mmaps + child reapers in a printable type would invite
/// accidental fmt-on-failure paths), so the let-else shape is used
/// instead of `expect_err` (which has a `T: Debug` bound).
#[test]
fn spawn_rejects_invalid_mempolicy_config() {
    let bad = WorkloadConfig::default().mem_policy(MemPolicy::Bind(BTreeSet::new()));
    let Err(err) = WorkloadHandle::spawn(&bad) else {
        panic!(
            "WorkloadHandle::spawn must reject invalid mem_policy via WorkloadConfig::validate",
        );
    };
    let msg = err.to_string();
    assert!(
        msg.contains("WorkloadConfig.mem_policy"),
        "spawn must surface the WorkloadConfig::validate() wrap prefix \
         (proves the validator was actually called): got {msg}",
    );
    assert!(
        msg.contains("primary"),
        "diagnostic must name the slot (primary group): got {msg}",
    );
    assert!(
        msg.contains("Bind"),
        "diagnostic must name the offending variant (Bind): got {msg}",
    );
}

/// Pins that [`WorkloadHandle::spawn_pcomm_cgroup`] — the second
/// public spawn entry point — also validates mem_policy. Same
/// reasoning as [`spawn_rejects_invalid_mempolicy_config`]: a
/// regression that drops the validate loop at the top of
/// `spawn_pcomm_cgroup` would silently bypass the structural gate
/// for the pcomm entry, allowing invalid mem_policy through to
/// `apply_mempolicy_with_flags`'s silent-skip arm. The let-else
/// shape mirrors the sibling test (`WorkloadHandle` is not
/// `Debug`, so `expect_err` cannot apply).
#[test]
fn spawn_pcomm_cgroup_rejects_invalid_mempolicy() {
    let bad = WorkSpec::default()
        .workers(1)
        .work_type(WorkType::SpinWait)
        .mem_policy(MemPolicy::Bind(BTreeSet::new()));
    let Err(err) = WorkloadHandle::spawn_pcomm_cgroup("ktstr-test", None, None, &[bad]) else {
        panic!("spawn_pcomm_cgroup must reject invalid mem_policy on works[0]");
    };
    let msg = err.to_string();
    assert!(
        msg.contains("spawn_pcomm_cgroup"),
        "diagnostic must name the entry point: got {msg}",
    );
    assert!(
        msg.contains("works[0].mem_policy"),
        "diagnostic must name the offending slot: got {msg}",
    );
    assert!(
        msg.contains("Bind"),
        "diagnostic must name the offending variant: got {msg}",
    );
}
