//! Unit tests for the scenario module. Co-located with production
//! code via the `tests` submodule pattern.

#![cfg(test)]

use super::*;
use crate::assert;

#[test]
fn resolve_affinity_inherit() {
    let t = crate::topology::TestTopology::synthetic(8, 2);
    assert!(matches!(
        resolve_affinity_for_cgroup(&AffinityIntent::Inherit, None, &t).unwrap(),
        ResolvedAffinity::None
    ));
}

#[test]
fn resolve_affinity_single_cpu() {
    let t = crate::topology::TestTopology::synthetic(8, 2);
    match resolve_affinity_for_cgroup(&AffinityIntent::SingleCpu, None, &t).unwrap() {
        ResolvedAffinity::SingleCpu(c) => assert_eq!(c, 0),
        other => panic!("expected SingleCpu, got {:?}", other),
    }
}

#[test]
fn resolve_affinity_cross_cgroup() {
    let t = crate::topology::TestTopology::synthetic(8, 2);
    match resolve_affinity_for_cgroup(&AffinityIntent::CrossCgroup, None, &t).unwrap() {
        ResolvedAffinity::Fixed(cpus) => assert_eq!(cpus.len(), 8),
        other => panic!("expected Fixed, got {:?}", other),
    }
}

#[test]
fn resolve_affinity_llc_aligned() {
    let t = crate::topology::TestTopology::synthetic(8, 2);
    // No cpuset: both LLCs cover the full pool equally. LLC 0
    // is found first with max overlap, so result is LLC 0 CPUs.
    match resolve_affinity_for_cgroup(&AffinityIntent::LlcAligned, None, &t).unwrap() {
        ResolvedAffinity::Fixed(cpus) => assert_eq!(cpus, [0, 1, 2, 3].into_iter().collect()),
        other => panic!("expected Fixed, got {:?}", other),
    }
}

#[test]
fn resolve_affinity_llc_aligned_with_cpuset() {
    let t = crate::topology::TestTopology::synthetic(8, 2);
    // Cpuset restricted to LLC 1 CPUs: LlcAligned picks LLC 1.
    let cpusets: Vec<BTreeSet<usize>> = vec![
        [0, 1, 2, 3].into_iter().collect(),
        [4, 5, 6, 7].into_iter().collect(),
    ];
    match resolve_affinity_for_cgroup(&AffinityIntent::LlcAligned, cpusets.get(1), &t).unwrap() {
        ResolvedAffinity::Fixed(cpus) => assert_eq!(cpus, [4, 5, 6, 7].into_iter().collect()),
        other => panic!("expected Fixed, got {:?}", other),
    }
}

#[test]
fn resolve_affinity_random_subset() {
    let t = crate::topology::TestTopology::synthetic(8, 2);
    let cpusets: Vec<BTreeSet<usize>> = vec![[0, 1, 2, 3].into_iter().collect()];
    // Caller pre-builds the pool from topology; the resolver
    // intersects with the cgroup's cpuset so the effective sample
    // pool stays within the cgroup's CPU budget. Sample size
    // is half the cpuset (`(pool.len() / 2).max(1)`).
    let intent = AffinityIntent::random_subset(t.all_cpus().iter().copied(), 2);
    match resolve_affinity_for_cgroup(&intent, cpusets.first(), &t).unwrap() {
        ResolvedAffinity::Random { from, count } => {
            assert_eq!(from, cpusets[0]);
            assert_eq!(count, 2); // half of 4
        }
        other => panic!("expected Random, got {:?}", other),
    }
}

#[test]
fn cgroup_work_default() {
    let cw = WorkSpec::default();
    assert_eq!(cw.num_workers, None);
    assert!(matches!(cw.work_type, WorkType::SpinWait));
    assert!(matches!(cw.sched_policy, SchedPolicy::Normal));
    assert!(matches!(cw.affinity, AffinityIntent::Inherit));
    assert!(matches!(cw.mem_policy, MemPolicy::Default));
}

#[test]
fn resolve_affinity_random_no_cpusets() {
    let t = crate::topology::TestTopology::synthetic(8, 2);
    // No cgroup cpuset → pool is the caller-supplied set verbatim.
    let intent = AffinityIntent::random_subset(t.all_cpus().iter().copied(), 4);
    match resolve_affinity_for_cgroup(&intent, None, &t).unwrap() {
        ResolvedAffinity::Random { from, count } => {
            assert_eq!(from.len(), 8); // all CPUs
            assert_eq!(count, 4); // half
        }
        other => panic!("expected Random, got {:?}", other),
    }
}

#[test]
fn resolve_affinity_random_subset_empty_pool_bails() {
    // Empty cpuset intersection on RandomSubset must surface as
    // a returnable Err — an unsatisfiable affinity intent is the
    // operator's bug to fix (widen the cpuset, change the
    // intent), not a silent "default placement" the test
    // happily accepts. (Earlier code paths produced
    // ResolvedAffinity::None or even an empty Random.from that
    // sched_setaffinity rejected with EINVAL; both forms
    // masked the configuration error.)
    let t = crate::topology::TestTopology::synthetic(4, 1);
    let empty: BTreeSet<usize> = BTreeSet::new();
    let intent = AffinityIntent::random_subset(t.all_cpus().iter().copied(), 1);
    let err = resolve_affinity_for_cgroup(&intent, Some(&empty), &t)
        .expect_err("empty cpuset intersection must bail");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("AffinityIntent::RandomSubset"),
        "diagnostic must name the intent variant: got {msg}",
    );
    assert!(
        msg.contains("empty cpuset"),
        "diagnostic must name what narrowed the pool (empty cpuset): got {msg}",
    );
    assert!(
        msg.contains("AffinityIntent::Inherit"),
        "diagnostic must name the recommended fix (Inherit fallback): got {msg}",
    );
}

#[test]
fn resolve_affinity_random_subset_empty_from_no_cpuset_bails() {
    // Companion to `_empty_pool_bails` above — that one pins the
    // `cpuset=Some(non_empty)` + `from=non_empty` → empty intersection
    // path. This test pins the OTHER way pool can be empty: caller
    // passes RandomSubset { from: empty } directly and no cpuset is
    // active. The bail diag must distinguish "intersected to empty"
    // (cpuset present) from "from-pool was empty with no cpuset to
    // narrow it" (no cpuset) so operators see actionable wording.
    let t = crate::topology::TestTopology::synthetic(4, 1);
    let intent = AffinityIntent::random_subset(std::iter::empty(), 1);
    let err = resolve_affinity_for_cgroup(&intent, None, &t)
        .expect_err("empty from-pool with no cpuset must bail");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("AffinityIntent::RandomSubset"),
        "diagnostic must name the intent variant: got {msg}",
    );
    assert!(
        msg.contains("empty `from` pool with no cgroup cpuset"),
        "diagnostic must distinguish this case from the cpuset-intersection \
             case so the operator sees the right remediation: got {msg}",
    );
    assert!(
        msg.contains("AffinityIntent::Inherit"),
        "diagnostic must name the recommended fix (Inherit fallback): got {msg}",
    );
}

#[test]
fn resolve_affinity_random_subset_zero_count_bails() {
    let t = crate::topology::TestTopology::synthetic(4, 1);
    let intent = AffinityIntent::random_subset(t.all_cpus().iter().copied(), 0);
    let err = resolve_affinity_for_cgroup(&intent, None, &t).expect_err("count=0 must bail");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("RandomSubset count=0"),
        "diagnostic must name count=0: got {msg}",
    );
}

#[test]
fn resolve_affinity_llc_aligned_disjoint_cpuset_bails() {
    // LLC 0 = {0,1,2,3}, LLC 1 = {4,5,6,7}. Cpuset = {} (empty)
    // → every LLC's intersection is empty → bail.
    let t = crate::topology::TestTopology::synthetic(8, 2);
    let empty: BTreeSet<usize> = BTreeSet::new();
    let err = resolve_affinity_for_cgroup(&AffinityIntent::LlcAligned, Some(&empty), &t)
        .expect_err("empty cpuset on LlcAligned must bail");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("AffinityIntent::LlcAligned"),
        "diagnostic must name the intent variant: got {msg}",
    );
    assert!(
        msg.contains("No LLC has any CPU"),
        "diagnostic must name the failure mode: got {msg}",
    );
    assert!(
        msg.contains("AffinityIntent::Inherit"),
        "diagnostic must name the recommended fix: got {msg}",
    );
}

#[test]
fn resolve_affinity_single_cpu_empty_cpuset_bails() {
    let t = crate::topology::TestTopology::synthetic(4, 1);
    let empty: BTreeSet<usize> = BTreeSet::new();
    let err = resolve_affinity_for_cgroup(&AffinityIntent::SingleCpu, Some(&empty), &t)
        .expect_err("empty cpuset on SingleCpu must bail");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("AffinityIntent::SingleCpu"),
        "diagnostic must name the intent variant: got {msg}",
    );
    assert!(
        msg.contains("empty cgroup cpuset"),
        "diagnostic must name the failure mode (empty cpuset): got {msg}",
    );
    assert!(
        msg.contains("AffinityIntent::Inherit"),
        "diagnostic must name the recommended fix: got {msg}",
    );
}

#[test]
fn resolve_affinity_exact_disjoint_from_cpuset_bails() {
    let t = crate::topology::TestTopology::synthetic(4, 1);
    // Cpuset = {0, 1}; Exact = {2, 3}. Intersection is empty.
    let cpuset: BTreeSet<usize> = [0, 1].into_iter().collect();
    let exact: BTreeSet<usize> = [2, 3].into_iter().collect();
    let err = resolve_affinity_for_cgroup(&AffinityIntent::Exact(exact), Some(&cpuset), &t)
        .expect_err("disjoint Exact must bail");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("AffinityIntent::Exact"),
        "diagnostic must name the intent variant: got {msg}",
    );
    assert!(
        msg.contains("disjoint"),
        "diagnostic must name the failure mode (disjoint): got {msg}",
    );
    assert!(
        msg.contains("AffinityIntent::Inherit"),
        "diagnostic must name the recommended fix: got {msg}",
    );
}

#[test]
fn resolve_affinity_exact_empty_bails_regardless_of_cpuset() {
    // Empty Exact is the most-explicit way a user can say "I made
    // a mistake" — must bail whether or not a cpuset is active,
    // otherwise the no-cpuset branch silently produces
    // ResolvedAffinity::Fixed({}) which flatten_for_spawn used to
    // coerce to Inherit.
    let t = crate::topology::TestTopology::synthetic(4, 1);
    let empty: BTreeSet<usize> = BTreeSet::new();
    // (a) no cpuset
    let err_no_cs = resolve_affinity_for_cgroup(&AffinityIntent::Exact(empty.clone()), None, &t)
        .expect_err("empty Exact must bail even without cpuset");
    let msg_no_cs = format!("{err_no_cs:#}");
    assert!(
        msg_no_cs.contains("AffinityIntent::Exact(BTreeSet::new())"),
        "diagnostic must name the empty-Exact form: got {msg_no_cs}",
    );
    assert!(
        msg_no_cs.contains("unsatisfiable"),
        "diagnostic must name the failure mode: got {msg_no_cs}",
    );
    assert!(
        msg_no_cs.contains("AffinityIntent::Inherit"),
        "diagnostic must name the recommended fix: got {msg_no_cs}",
    );
    // (b) with cpuset — same bail, doesn't reach the intersection arm
    let cpuset: BTreeSet<usize> = [0, 1].into_iter().collect();
    let err_with_cs = resolve_affinity_for_cgroup(&AffinityIntent::Exact(empty), Some(&cpuset), &t)
        .expect_err("empty Exact must bail with cpuset too");
    let msg_with_cs = format!("{err_with_cs:#}");
    assert!(
        msg_with_cs.contains("AffinityIntent::Exact(BTreeSet::new())"),
        "diagnostic must name the empty-Exact form (not the disjoint message): got {msg_with_cs}",
    );
}

#[test]
fn resolve_affinity_oob_cgroup_idx_falls_back_to_unrestricted() {
    // The wrapper that bounds-checked cgroup_idx against cpusets
    // was inlined into run_scenario (see the tracing::warn in
    // run_scenario's affinity-resolve block). The underlying
    // `cpusets.get(idx)` returns None on OOB, and
    // resolve_affinity_for_cgroup's None arm delivers the
    // unrestricted fallback. This test pins the fallback contract.
    let t = crate::topology::TestTopology::synthetic(4, 1);
    let cpusets: Vec<BTreeSet<usize>> = vec![[0, 1].into_iter().collect()];
    let oob_idx = 5;
    // Mirror the inlined expression in run_scenario:
    let cpuset = cpusets.get(oob_idx);
    assert!(cpuset.is_none(), "OOB index must yield None cpuset");
    let intent = AffinityIntent::random_subset(t.all_cpus().iter().copied(), 2);
    match resolve_affinity_for_cgroup(&intent, cpuset, &t).unwrap() {
        ResolvedAffinity::Random { from, count } => {
            assert_eq!(from.len(), 4, "OOB idx falls back to full topology");
            assert_eq!(count, 2);
        }
        other => panic!("expected Random with full pool, got {:?}", other),
    }
}

/// `SmtSiblingPair` resolves to a 2-CPU `Fixed` set drawn from
/// the first physical core whose siblings live inside the
/// effective cpuset. The sequential VM topology constructor
/// numbers core `c` of LLC `l` with siblings at
/// `(l*cores_per_llc + c) * threads_per_core ..`, so a
/// `(1 numa, 1 llc, 2 cores, 2 threads)` topology produces
/// core 0's pair `{0, 1}` first.
#[test]
fn resolve_affinity_smt_sibling_pair_uses_first_core() {
    let vmt = crate::vmm::topology::Topology::new(1, 1, 2, 2);
    let t = crate::topology::TestTopology::from_vm_topology(&vmt);
    match resolve_affinity_for_cgroup(&AffinityIntent::SmtSiblingPair, None, &t).unwrap() {
        ResolvedAffinity::Fixed(cpus) => {
            assert_eq!(
                cpus,
                [0usize, 1].into_iter().collect(),
                "SmtSiblingPair must pick the first core's siblings"
            );
        }
        other => panic!("expected Fixed({{0, 1}}), got {:?}", other),
    }
}

/// When the cpuset excludes one of core 0's siblings, the
/// resolver must skip that core and look for another core whose
/// siblings are both present. With 2 cores per LLC and a cpuset
/// of `{2, 3}`, core 1's pair `{2, 3}` is selected.
#[test]
fn resolve_affinity_smt_sibling_pair_skips_partial_cores() {
    let vmt = crate::vmm::topology::Topology::new(1, 1, 2, 2);
    let t = crate::topology::TestTopology::from_vm_topology(&vmt);
    let cpuset: BTreeSet<usize> = [2usize, 3].into_iter().collect();
    match resolve_affinity_for_cgroup(&AffinityIntent::SmtSiblingPair, Some(&cpuset), &t).unwrap() {
        ResolvedAffinity::Fixed(cpus) => {
            assert_eq!(
                cpus,
                [2usize, 3].into_iter().collect(),
                "SmtSiblingPair must skip core 0 when cpuset excludes one of its \
                     siblings and pick the next eligible pair"
            );
        }
        other => panic!("expected Fixed({{2, 3}}), got {:?}", other),
    }
}

/// `threads_per_core == 1` means no SMT. The resolver must
/// return an explicit error rather than silently degrading to
/// `None` — running an SMT-pair workload without true SMT
/// produces a misleading test result.
#[test]
fn resolve_affinity_smt_sibling_pair_errors_without_smt() {
    let vmt = crate::vmm::topology::Topology::new(1, 1, 4, 1);
    let t = crate::topology::TestTopology::from_vm_topology(&vmt);
    let err = resolve_affinity_for_cgroup(&AffinityIntent::SmtSiblingPair, None, &t)
        .expect_err("threads_per_core=1 must produce an error, not silent fallback");
    let msg = err.to_string();
    assert!(
        msg.contains("SmtSiblingPair"),
        "diagnostic must name the variant, got: {msg}"
    );
    assert!(
        msg.contains("two SMT siblings"),
        "diagnostic must explain the missing precondition, got: {msg}"
    );
    // With cpuset=None, the scope must read as "full topology" —
    // NOT "<no cpuset>" — so the operator looks at topology config
    // (threads_per_core), not cgroup cpuset config.
    assert!(
        msg.contains("full topology"),
        "diagnostic with no cpuset must name the topology as the \
             search scope (not '<no cpuset>'), got: {msg}",
    );
}

/// When the cpuset isolates each sibling onto a different
/// cgroup (e.g. `{0, 2}` keeps one sibling from each core but
/// no full pair), the resolver must error rather than silently
/// producing a 1-CPU set.
#[test]
fn resolve_affinity_smt_sibling_pair_errors_when_cpuset_breaks_pairs() {
    let vmt = crate::vmm::topology::Topology::new(1, 1, 2, 2);
    let t = crate::topology::TestTopology::from_vm_topology(&vmt);
    let cpuset: BTreeSet<usize> = [0usize, 2].into_iter().collect();
    let err = resolve_affinity_for_cgroup(&AffinityIntent::SmtSiblingPair, Some(&cpuset), &t)
        .expect_err("cpuset that breaks every sibling pair must error");
    let msg = err.to_string();
    assert!(
        msg.contains("SmtSiblingPair"),
        "diagnostic must name the variant, got: {msg}"
    );
    // The cpuset must be echoed via format_cpuset_for_diag so an
    // operator sees which sibling-isolating cpuset they passed.
    // Pin the exact rendered shape (not just a digit) so a future
    // format change can't slip past on a coincidental substring.
    assert!(
        msg.contains("effective cpuset (cpuset {0, 2})"),
        "diagnostic with cpuset=Some must echo the cpuset value \
             verbatim via format_cpuset_for_diag, got: {msg}",
    );
}

/// SmtSiblingPair bail with `cpuset=Some(empty)` should still take
/// the cpuset-branch of the scope conditional (since `is_some()`
/// is true) and render "the effective cpuset (empty cpuset {})".
/// Pin the wording so a refactor that collapses Some(empty) into
/// the None branch (e.g. via `cpuset.map_or(false, |cs|
/// !cs.is_empty())`) doesn't silently change the operator-facing
/// scope from "the effective cpuset (empty cpuset {})" to "the
/// full topology (no cgroup cpuset is active)".
#[test]
fn resolve_affinity_smt_sibling_pair_errors_with_empty_cpuset() {
    let vmt = crate::vmm::topology::Topology::new(1, 1, 2, 2);
    let t = crate::topology::TestTopology::from_vm_topology(&vmt);
    let empty_cpuset: BTreeSet<usize> = BTreeSet::new();
    let err = resolve_affinity_for_cgroup(&AffinityIntent::SmtSiblingPair, Some(&empty_cpuset), &t)
        .expect_err("SmtSiblingPair with empty cpuset must bail");
    let msg = err.to_string();
    assert!(
        msg.contains("SmtSiblingPair"),
        "diagnostic must name the variant, got: {msg}"
    );
    // Some(empty) → cpuset-branch ("the effective cpuset (empty cpuset {})"),
    // NOT the None branch ("the full topology"). Inversion would be silent.
    assert!(
        msg.contains("effective cpuset (empty cpuset") && !msg.contains("full topology"),
        "diagnostic with cpuset=Some(empty) must take the cpuset \
             branch (not the topology branch), got: {msg}",
    );
}

/// `TestTopology::synthetic` builds an empty per-core sibling
/// map (no SMT info), so `SmtSiblingPair` must error there as
/// well — the resolver depends on `LlcInfo::cores`, not on the
/// raw CPU list.
#[test]
fn resolve_affinity_smt_sibling_pair_errors_on_synthetic_topology() {
    let t = crate::topology::TestTopology::synthetic(8, 2);
    let err = resolve_affinity_for_cgroup(&AffinityIntent::SmtSiblingPair, None, &t)
        .expect_err("synthetic topology has no per-core sibling data — must error");
    let msg = err.to_string();
    assert!(
        msg.contains("SmtSiblingPair"),
        "diagnostic must name the variant, got: {msg}"
    );
    // No cpuset → scope reads as "full topology", not "<no cpuset>".
    assert!(
        msg.contains("full topology"),
        "diagnostic with no cpuset must name the topology as the \
             search scope, got: {msg}",
    );
}

#[test]
fn split_half_even() {
    let t = crate::topology::TestTopology::synthetic(8, 2);
    let ctx_cg = crate::cgroup::CgroupManager::new("/nonexistent");
    let ctx = Ctx {
        cgroups: &ctx_cg,
        topo: &t,
        duration: std::time::Duration::from_secs(1),
        workers_per_cgroup: 4,
        sched_pid: None,
        settle: Duration::from_millis(3000),
        work_type_override: None,
        assert: assert::Assert::default_checks(),
        wait_for_map_write: false,
    };
    let (a, b) = split_half(&ctx);
    // Last CPU reserved for cgroup 0 → 7 usable, split 3/4
    assert_eq!(a.len() + b.len(), 7);
    assert!(a.intersection(&b).count() == 0, "halves should not overlap");
}

#[test]
fn split_half_small() {
    let t = crate::topology::TestTopology::synthetic(2, 1);
    let ctx_cg = crate::cgroup::CgroupManager::new("/nonexistent");
    let ctx = Ctx {
        cgroups: &ctx_cg,
        topo: &t,
        duration: std::time::Duration::from_secs(1),
        workers_per_cgroup: 1,
        sched_pid: None,
        settle: Duration::from_millis(3000),
        work_type_override: None,
        assert: assert::Assert::default_checks(),
        wait_for_map_write: false,
    };
    let (a, b) = split_half(&ctx);
    assert_eq!(a.len() + b.len(), 2);
}

#[test]
fn dfl_wl_propagates_workers() {
    let t = crate::topology::TestTopology::synthetic(8, 2);
    let ctx_cg = crate::cgroup::CgroupManager::new("/nonexistent");
    let ctx = Ctx {
        cgroups: &ctx_cg,
        topo: &t,
        duration: std::time::Duration::from_secs(1),
        workers_per_cgroup: 7,
        sched_pid: None,
        settle: Duration::from_millis(3000),
        work_type_override: None,
        assert: assert::Assert::default_checks(),
        wait_for_map_write: false,
    };
    let wl = dfl_wl(&ctx);
    assert_eq!(wl.num_workers, 7);
    assert!(matches!(wl.work_type, WorkType::SpinWait));
}

#[test]
fn process_alive_self_is_true() {
    let pid: libc::pid_t = unsafe { libc::getpid() };
    assert!(process_alive(pid));
}

/// `Ctx::active_sched_pid` filters `Some(0)` — and any negative
/// pid — out of the "configured scheduler" set. Without the
/// filter, a `Some(0)` would reach `process_alive(0)`, which
/// returns `false`, and the three liveness call sites
/// (`run_scenario` post-settle, workload-phase polling,
/// `setup_cgroups` post-settle) would each raise a spurious
/// scheduler-died diagnostic on a test that never had a
/// scheduler to begin with. The "configured" definition has to
/// agree across all three sites, so the gate lives on `Ctx`
/// rather than being re-inlined three times.
#[test]
fn ctx_active_sched_pid_treats_nonpositive_as_unconfigured() {
    let cg = crate::cgroup::CgroupManager::new("/nonexistent");
    let topo = crate::topology::TestTopology::synthetic(1, 1);

    // Some(0) — the most likely mistake when a caller confuses
    // `Ctx::sched_pid` (Option<pid_t>) with the workload TLS's
    // 0-sentinel.
    let ctx_zero = Ctx::builder(&cg, &topo).sched_pid(Some(0)).build();
    assert_eq!(
        ctx_zero.sched_pid,
        Some(0),
        "builder must preserve the literal value — the gate lives in the accessor",
    );
    assert_eq!(
        ctx_zero.active_sched_pid(),
        None,
        "Some(0) must be treated as unconfigured, otherwise the liveness \
             bails fire on tests that never ran a scheduler",
    );

    // Negative pids: `kill(negative, sig)` is a process-group
    // broadcast, not a live-process query — also unconfigured.
    let ctx_neg = Ctx::builder(&cg, &topo).sched_pid(Some(-1)).build();
    assert_eq!(
        ctx_neg.active_sched_pid(),
        None,
        "negative pid must be treated as unconfigured",
    );

    // `pid_t::MIN` — the most-negative representable value.
    // Pin the lower boundary explicitly so a future filter
    // change that accidentally uses `p >= 0` (a common mis-read
    // of "non-negative") keeps treating the edge as
    // unconfigured. `Some(0)` above covers the `p > 0` vs
    // `p >= 0` distinction; `pid_t::MIN` covers "does the
    // `p > 0` predicate survive an overflow-adjacent input?"
    let ctx_min = Ctx::builder(&cg, &topo)
        .sched_pid(Some(libc::pid_t::MIN))
        .build();
    assert_eq!(
        ctx_min.active_sched_pid(),
        None,
        "pid_t::MIN must be treated as unconfigured — the filter \
             is `p > 0`, and the most-negative pid_t stays unconfigured \
             under that predicate by construction",
    );

    // Sanity: a positive pid survives the filter.
    let ctx_pos = Ctx::builder(&cg, &topo).sched_pid(Some(1234)).build();
    assert_eq!(
        ctx_pos.active_sched_pid(),
        Some(1234),
        "positive pid must pass through unchanged",
    );

    // `pid_t::MAX` — the most-positive representable value.
    // Linux caps live pids at PID_MAX_LIMIT (2^22) so
    // `pid_t::MAX` (2^31 - 1) cannot be allocated, but the
    // filter operates on pure value-polarity rather than
    // kernel allocability. A positive value — even one
    // guaranteed not to exist — must pass through unchanged,
    // because the liveness callsites downstream
    // (`process_alive`) are what will see the pid and report
    // it as dead; the `active_sched_pid` filter is a
    // configured-vs-unconfigured gate, not a liveness gate.
    let ctx_max = Ctx::builder(&cg, &topo)
        .sched_pid(Some(libc::pid_t::MAX))
        .build();
    assert_eq!(
        ctx_max.active_sched_pid(),
        Some(libc::pid_t::MAX),
        "pid_t::MAX must pass the filter — `p > 0` accepts it. \
             Liveness determination is the responsibility of the \
             downstream `process_alive` call, not this accessor.",
    );

    // Sanity: None stays None.
    let ctx_none = Ctx::builder(&cg, &topo).sched_pid(None).build();
    assert_eq!(
        ctx_none.active_sched_pid(),
        None,
        "None must pass through unchanged",
    );
}

#[test]
fn process_alive_zero_is_false() {
    // kill(0, sig) targets the caller's process group, so without
    // an explicit guard kill(0, 0) succeeds and would falsely
    // report "process 0" as alive.
    assert!(!process_alive(0));
}

#[test]
fn process_alive_negative_is_false() {
    // kill(negative, sig) targets a process group (or, for -1,
    // every process the caller can signal). A pid_t <= 0 is
    // never a live-process query and must return false.
    assert!(!process_alive(-1));
    assert!(!process_alive(libc::pid_t::MIN));
}

#[test]
fn process_alive_nonexistent_pid() {
    // Use `pid_t::MAX` (2^31 - 1) as a guaranteed-non-existent pid.
    // Linux's `pid_max` is capped at 2^22 (4,194,304) per
    // `include/linux/threads.h` (PID_MAX_LIMIT), so any pid above
    // that threshold cannot be allocated — kill(2) returns ESRCH
    // unconditionally. The previous formulation (fork + waitpid +
    // probe-the-freed-pid) had a PID-reuse race: between waitpid
    // returning and process_alive probing, the kernel could
    // allocate the freed pid to a concurrent fork() on the host
    // (heavily-forking CI runners, container hosts, etc.) and
    // turn this test into a flake. Using a pid above PID_MAX_LIMIT
    // removes the race entirely — no syscall ordering can place a
    // live process on a pid the kernel refuses to allocate.
    assert!(!process_alive(libc::pid_t::MAX));
}

#[test]
fn cgroup_group_new_empty() {
    let cg = crate::cgroup::CgroupManager::new("/nonexistent");
    let group = CgroupGroup::new(&cg);
    assert!(group.names().is_empty());
}

// -- resolve_affinity_for_cgroup edge cases --

#[test]
fn resolve_affinity_single_cpu_with_cpuset() {
    let t = crate::topology::TestTopology::synthetic(4, 1);
    // Cpuset restricts to CPUs {2,3}: SingleCpu picks first in cpuset.
    let cpusets: Vec<BTreeSet<usize>> = vec![[2, 3].into_iter().collect()];
    match resolve_affinity_for_cgroup(&AffinityIntent::SingleCpu, cpusets.first(), &t).unwrap() {
        ResolvedAffinity::SingleCpu(c) => assert_eq!(c, 2),
        other => panic!("expected SingleCpu, got {:?}", other),
    }
}

#[test]
fn resolve_affinity_llc_aligned_picks_best_overlap() {
    let t = crate::topology::TestTopology::synthetic(8, 2);
    // Cpuset spans both LLCs but has more CPUs in LLC 1.
    // LLC 0 = {0,1,2,3}, LLC 1 = {4,5,6,7}.
    // Cpuset = {3, 4, 5, 6, 7}: LLC 1 has 4 CPUs in cpuset, LLC 0 has 1.
    let cpusets: Vec<BTreeSet<usize>> = vec![[3, 4, 5, 6, 7].into_iter().collect()];
    match resolve_affinity_for_cgroup(&AffinityIntent::LlcAligned, cpusets.first(), &t).unwrap() {
        ResolvedAffinity::Fixed(cpus) => {
            // LLC 1 has best overlap; result is intersection {4,5,6,7}.
            assert_eq!(cpus, [4, 5, 6, 7].into_iter().collect());
        }
        other => panic!("expected Fixed, got {:?}", other),
    }
}

#[test]
fn resolve_num_workers_zero_rejected_with_label() {
    let w = WorkSpec {
        num_workers: Some(0),
        ..Default::default()
    };
    let err = resolve_num_workers(&w, 4, "victim").unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("cgroup 'victim'"),
        "label must appear in error: {msg}"
    );
    assert!(
        msg.contains("num_workers=0"),
        "error must name the offending field: {msg}"
    );
}

#[test]
fn resolve_num_workers_zero_default_also_rejected() {
    // When num_workers is unset AND the default is 0, still reject.
    let w = WorkSpec {
        num_workers: None,
        ..Default::default()
    };
    assert!(resolve_num_workers(&w, 0, "cg").is_err());
}

#[test]
fn resolve_num_workers_falls_back_to_default() {
    let w = WorkSpec {
        num_workers: None,
        ..Default::default()
    };
    assert_eq!(resolve_num_workers(&w, 3, "cg").unwrap(), 3);
}

#[test]
fn resolve_num_workers_explicit_wins_over_default() {
    let w = WorkSpec {
        num_workers: Some(7),
        ..Default::default()
    };
    assert_eq!(resolve_num_workers(&w, 3, "cg").unwrap(), 7);
}

/// Minimal `CgroupOps` double for `CgroupGroup::drop` error-path
/// tests. Injects a caller-supplied `io::Error` into the
/// `remove_cgroup` chain so the test can cover both the ENOENT
/// (benign TOCTOU) branch and the non-ENOENT warn branch without
/// touching cgroupfs. Every other trait method is a no-op — the
/// drop path only calls `remove_cgroup`.
struct DropErrCgroupOps {
    parent: std::path::PathBuf,
    remove_kind: std::io::ErrorKind,
    raw_os_error: Option<i32>,
    remove_calls: std::sync::Mutex<Vec<String>>,
}

impl DropErrCgroupOps {
    fn new(kind: std::io::ErrorKind, raw: Option<i32>) -> Self {
        Self {
            parent: std::path::PathBuf::from("/mock/cgroup"),
            remove_kind: kind,
            raw_os_error: raw,
            remove_calls: std::sync::Mutex::new(Vec::new()),
        }
    }
    fn calls(&self) -> Vec<String> {
        self.remove_calls.lock().unwrap().clone()
    }
}

impl crate::cgroup::CgroupOps for DropErrCgroupOps {
    fn parent_path(&self) -> &std::path::Path {
        &self.parent
    }
    fn setup(&self, _: &std::collections::BTreeSet<crate::cgroup::Controller>) -> Result<()> {
        Ok(())
    }
    fn create_cgroup(&self, _: &str) -> Result<()> {
        Ok(())
    }
    fn remove_cgroup(&self, name: &str) -> Result<()> {
        self.remove_calls.lock().unwrap().push(name.to_string());
        let io = match self.raw_os_error {
            Some(errno) => std::io::Error::from_raw_os_error(errno),
            None => std::io::Error::from(self.remove_kind),
        };
        // Wrap in anyhow::Context the same way the real CgroupManager
        // does, so `err.root_cause().downcast_ref::<io::Error>()`
        // traverses the chain identically to production.
        Err(anyhow::Error::new(io).context("remove_dir cgroup"))
    }
    fn set_cpuset(&self, _: &str, _: &BTreeSet<usize>) -> Result<()> {
        Ok(())
    }
    fn clear_cpuset(&self, _: &str) -> Result<()> {
        Ok(())
    }
    fn set_cpuset_mems(&self, _: &str, _: &BTreeSet<usize>) -> Result<()> {
        Ok(())
    }
    fn clear_cpuset_mems(&self, _: &str) -> Result<()> {
        Ok(())
    }
    fn set_cpu_max(&self, _: &str, _: Option<u64>, _: u64) -> Result<()> {
        Ok(())
    }
    fn set_cpu_weight(&self, _: &str, _: u32) -> Result<()> {
        Ok(())
    }
    fn set_memory_max(&self, _: &str, _: Option<u64>) -> Result<()> {
        Ok(())
    }
    fn set_memory_high(&self, _: &str, _: Option<u64>) -> Result<()> {
        Ok(())
    }
    fn set_memory_low(&self, _: &str, _: Option<u64>) -> Result<()> {
        Ok(())
    }
    fn set_io_weight(&self, _: &str, _: u16) -> Result<()> {
        Ok(())
    }
    fn set_freeze(&self, _: &str, _: bool) -> Result<()> {
        Ok(())
    }
    fn set_pids_max(&self, _: &str, _: Option<u64>) -> Result<()> {
        Ok(())
    }
    fn set_memory_swap_max(&self, _: &str, _: Option<u64>) -> Result<()> {
        Ok(())
    }
    fn move_task(&self, _: &str, _: libc::pid_t) -> Result<()> {
        Ok(())
    }
    fn move_tasks(&self, _: &str, _: &[libc::pid_t]) -> Result<()> {
        Ok(())
    }
    fn clear_subtree_control(&self, _: &str) -> Result<()> {
        Ok(())
    }
    fn drain_tasks(&self, _: &str) -> Result<()> {
        Ok(())
    }
    fn cleanup_all(&self) -> Result<()> {
        Ok(())
    }
}

/// Drop must iterate every tracked name regardless of error kind —
/// ENOENT is classified benign and dropped silently; any other
/// error path (EBUSY, EACCES, generic IO) takes the warn branch.
/// Neither path may panic (panic in Drop under `panic = "abort"`
/// aborts the whole process). This test is a panic-free crash-test
/// plus a call-count pin: every tracked cgroup must see exactly
/// one `remove_cgroup` call, in reverse-insertion order, in every
/// branch — so the iteration contract doesn't silently shrink.
#[test]
fn cgroup_group_drop_is_panic_free_on_every_error_kind() {
    for (label, kind, raw) in [
        ("ENOENT", std::io::ErrorKind::NotFound, Some(libc::ENOENT)),
        ("EBUSY", std::io::ErrorKind::Other, Some(libc::EBUSY)),
        (
            "EACCES",
            std::io::ErrorKind::PermissionDenied,
            Some(libc::EACCES),
        ),
        ("generic-IO", std::io::ErrorKind::Other, None),
    ] {
        let mock = DropErrCgroupOps::new(kind, raw);
        {
            let mut group = CgroupGroup::new(&mock);
            group.names.push("child-a".to_string());
            group.names.push("child-b".to_string());
            // drop at end of scope must not panic.
        }
        let calls = mock.calls();
        // Reverse-insertion order: child-b first, then child-a —
        // pins the nested-cleanup invariant documented in Drop.
        assert_eq!(
            calls,
            vec!["child-b".to_string(), "child-a".to_string()],
            "[{label}] Drop must call remove_cgroup for every tracked name in reverse order",
        );
    }
}

/// `is_io_not_found` is the one-place classifier for the "benign
/// TOCTOU ENOENT" branch in both `CgroupGroup::drop` and
/// `Op::RemoveCgroup`. Pin the contract: NotFound → true; every
/// other kind → false. A regression that mis-classifies EBUSY or
/// EACCES as "not found" would silently swallow a real teardown
/// failure.
#[test]
fn is_io_not_found_matches_only_notfound() {
    let wrap = |k: std::io::ErrorKind| -> anyhow::Error {
        anyhow::Error::new(std::io::Error::from(k)).context("wrap")
    };
    assert!(is_io_not_found(&wrap(std::io::ErrorKind::NotFound)));
    assert!(!is_io_not_found(&wrap(
        std::io::ErrorKind::PermissionDenied
    )));
    assert!(!is_io_not_found(&wrap(std::io::ErrorKind::Other)));
    // anyhow::anyhow! without an underlying io::Error must not
    // look like NotFound even if the message contains "not found".
    let no_io = anyhow::anyhow!("cgroup not found in parent");
    assert!(!is_io_not_found(&no_io));
}

/// `remove_cgroup_errno_hint` gives actionable remediation for
/// the two errnos users can fix (EBUSY, EACCES) and stays quiet
/// for everything else. The hint strings end up rendered into
/// warn output; pin their presence/absence so a regression that
/// drops the hint (or hallucinates one) surfaces in test output.
#[test]
fn remove_cgroup_errno_hint_covers_ebusy_and_eacces() {
    let busy = anyhow::Error::new(std::io::Error::from_raw_os_error(libc::EBUSY)).context("wrap");
    let acces = anyhow::Error::new(std::io::Error::from_raw_os_error(libc::EACCES)).context("wrap");
    let enotempty =
        anyhow::Error::new(std::io::Error::from_raw_os_error(libc::ENOTEMPTY)).context("wrap");
    let non_io = anyhow::anyhow!("not an io error");

    assert!(
        remove_cgroup_errno_hint(&busy).is_some_and(|h| h.contains("EBUSY") && h.contains("drain")),
        "EBUSY hint must name the errno and the drain remediation",
    );
    assert!(
        remove_cgroup_errno_hint(&acces)
            .is_some_and(|h| h.contains("EACCES") && h.contains("permission")),
        "EACCES hint must name the errno and the permission angle",
    );
    assert_eq!(
        remove_cgroup_errno_hint(&enotempty),
        None,
        "unclassified errnos must yield no hint so warn stays terse",
    );
    assert_eq!(
        remove_cgroup_errno_hint(&non_io),
        None,
        "non-io root causes must yield no hint",
    );
}

// -- flatten_for_spawn / intent_for_spawn coverage --
//
// Every arm of `flatten_for_spawn` must round-trip a known
// [`ResolvedAffinity`] into the matching [`AffinityIntent`]
// shape. The flatten step gates the scenario engine's output
// against the spawn-time gate in `workload::resolve_spawn_affinity`.
//
// Contract: `resolve_affinity_for_cgroup` bails on every
// path that would produce an empty pool — empty `Fixed`,
// `Random { from: empty }`, `Random { count: 0 }` — so
// `flatten_for_spawn` upholds that contract with `unreachable!()`
// on those arms. The `_panics` tests pin the invariant: a future
// caller that bypasses the resolver and constructs an empty pool
// directly trips the panic at the construction site, never
// silently degrades to `Inherit`.

#[test]
fn flatten_for_spawn_none_to_inherit() {
    let out = flatten_for_spawn(ResolvedAffinity::None);
    assert!(
        matches!(out, AffinityIntent::Inherit),
        "ResolvedAffinity::None must flatten to Inherit, got {out:?}"
    );
}

#[test]
fn flatten_for_spawn_fixed_to_exact() {
    let set: BTreeSet<usize> = [1usize, 3, 5].into_iter().collect();
    let out = flatten_for_spawn(ResolvedAffinity::Fixed(set.clone()));
    match out {
        AffinityIntent::Exact(got) => {
            assert_eq!(got, set, "Fixed payload must round-trip into Exact");
        }
        other => panic!("expected Exact, got {other:?}"),
    }
}

/// Invariant: `resolve_affinity_for_cgroup` bails on
/// every path that would produce an empty `Fixed`, so
/// `flatten_for_spawn` upholds that contract with `unreachable!()`.
/// This test pins that contract — if a future caller bypasses the
/// resolver and constructs `ResolvedAffinity::Fixed(BTreeSet::new())`
/// directly, the panic surfaces the bug at the call site instead of
/// silently degrading to `Inherit`.
#[test]
#[should_panic(expected = "ResolvedAffinity::Fixed(empty) reached flatten_for_spawn")]
fn flatten_for_spawn_fixed_empty_panics() {
    let _ = flatten_for_spawn(ResolvedAffinity::Fixed(BTreeSet::new()));
}

#[test]
fn flatten_for_spawn_single_cpu_to_exact_singleton() {
    let out = flatten_for_spawn(ResolvedAffinity::SingleCpu(7));
    match out {
        AffinityIntent::Exact(got) => {
            let expected: BTreeSet<usize> = [7usize].into_iter().collect();
            assert_eq!(got, expected, "SingleCpu must flatten to a 1-CPU Exact set");
        }
        other => panic!("expected Exact({{7}}), got {other:?}"),
    }
}

#[test]
fn flatten_for_spawn_random_to_random_subset() {
    let from: BTreeSet<usize> = [0usize, 1, 2, 3].into_iter().collect();
    let out = flatten_for_spawn(ResolvedAffinity::Random {
        from: from.clone(),
        count: 2,
    });
    match out {
        AffinityIntent::RandomSubset {
            from: got_from,
            count: got_count,
        } => {
            assert_eq!(got_from, from, "Random.from must round-trip verbatim");
            assert_eq!(got_count, 2, "Random.count must round-trip verbatim");
        }
        other => panic!("expected RandomSubset, got {other:?}"),
    }
}

/// Invariant: `resolve_affinity_for_cgroup` bails on
/// `RandomSubset` empty pools (after cpuset intersection), so
/// `flatten_for_spawn` upholds that contract with `unreachable!()`.
/// Pins the contract — see `flatten_for_spawn_fixed_empty_panics`.
#[test]
#[should_panic(expected = "reached flatten_for_spawn with count==0 or empty pool")]
fn flatten_for_spawn_random_empty_pool_panics() {
    let _ = flatten_for_spawn(ResolvedAffinity::Random {
        from: BTreeSet::new(),
        count: 4,
    });
}

/// Invariant: `resolve_affinity_for_cgroup` bails on
/// `RandomSubset { count: 0 }`. Pins the contract — see
/// `flatten_for_spawn_fixed_empty_panics`.
#[test]
#[should_panic(expected = "reached flatten_for_spawn with count==0 or empty pool")]
fn flatten_for_spawn_random_zero_count_panics() {
    let from: BTreeSet<usize> = [0usize, 1, 2, 3].into_iter().collect();
    let _ = flatten_for_spawn(ResolvedAffinity::Random { from, count: 0 });
}

/// End-to-end: `intent_for_spawn` chains
/// `resolve_affinity_for_cgroup` into `flatten_for_spawn`. Verify
/// the full pipeline produces a spawn-gate-acceptable intent for
/// each top-level [`AffinityIntent`] variant on the happy path.
/// Topology-aware variants (`SingleCpu`, `CrossCgroup`,
/// `SmtSiblingPair`, `LlcAligned`) flatten to `Exact`; `Inherit`
/// and `RandomSubset` (with a valid pool) round-trip. The bail
/// propagation for unsatisfiable inputs is covered by the
/// dedicated `intent_for_spawn_propagates_*_bail` sibling tests.
#[test]
fn intent_for_spawn_full_pipeline() {
    // Use a real VM topology so the per-LLC sibling map is
    // populated — `synthetic` leaves `LlcInfo::cores` empty,
    // which forces the SmtSiblingPair arm into its no-SMT
    // error branch. `Topology::new(numa, llcs, cores, threads)`
    // with threads=2 produces 2 SMT siblings per core.
    let vmt = crate::vmm::topology::Topology::new(1, 2, 2, 2);
    let t = crate::topology::TestTopology::from_vm_topology(&vmt);

    // Inherit → Inherit
    let out = intent_for_spawn(&AffinityIntent::Inherit, None, &t).unwrap();
    assert!(
        matches!(out, AffinityIntent::Inherit),
        "Inherit must round-trip, got {out:?}"
    );

    // SingleCpu → Exact({some_cpu})
    let out = intent_for_spawn(&AffinityIntent::SingleCpu, None, &t).unwrap();
    match out {
        AffinityIntent::Exact(set) => {
            assert_eq!(set.len(), 1, "SingleCpu flattens to a 1-CPU Exact set");
        }
        other => panic!("expected Exact, got {other:?}"),
    }

    // CrossCgroup → Exact(<all CPUs>)
    let out = intent_for_spawn(&AffinityIntent::CrossCgroup, None, &t).unwrap();
    match out {
        AffinityIntent::Exact(set) => {
            assert_eq!(set.len(), 8, "CrossCgroup flattens to all-CPU Exact set");
        }
        other => panic!("expected Exact, got {other:?}"),
    }

    // SmtSiblingPair → Exact({sibling_a, sibling_b})
    let out = intent_for_spawn(&AffinityIntent::SmtSiblingPair, None, &t).unwrap();
    match out {
        AffinityIntent::Exact(set) => {
            assert_eq!(set.len(), 2, "SmtSiblingPair flattens to a 2-CPU Exact set");
            // First core's siblings are CPUs 0 and 1 in the
            // sequential VM topology (cores 0..2 within LLC 0:
            // core 0 = {0, 1}, core 1 = {2, 3}).
            assert_eq!(
                set,
                [0usize, 1].into_iter().collect(),
                "SmtSiblingPair must pick the first core's siblings"
            );
        }
        other => panic!("expected Exact, got {other:?}"),
    }

    // LlcAligned → Exact(<LLC's CPUs>) — picks the LLC with most
    // overlap against the cpuset (or `all_cpuset` when no cpuset
    // is active). On the 1-NUMA / 2-LLC / 2-core / 2-thread
    // topology, each LLC owns 4 CPUs.
    let out = intent_for_spawn(&AffinityIntent::LlcAligned, None, &t).unwrap();
    match out {
        AffinityIntent::Exact(set) => {
            assert_eq!(
                set.len(),
                4,
                "LlcAligned flattens to one LLC's worth of CPUs"
            );
        }
        other => panic!("expected Exact, got {other:?}"),
    }

    // Exact(non-empty) → Exact(round-trip when within cpuset)
    let exact_cpus: BTreeSet<usize> = [0usize, 2, 4].into_iter().collect();
    let out = intent_for_spawn(&AffinityIntent::Exact(exact_cpus.clone()), None, &t).unwrap();
    match out {
        AffinityIntent::Exact(set) => {
            assert_eq!(
                set, exact_cpus,
                "Exact round-trip must preserve the CPU set"
            );
        }
        other => panic!("expected Exact, got {other:?}"),
    }

    // RandomSubset with valid pool → RandomSubset round-trip
    let pool: BTreeSet<usize> = [0usize, 1, 2, 3].into_iter().collect();
    let intent = AffinityIntent::random_subset(pool.iter().copied(), 2);
    let out = intent_for_spawn(&intent, None, &t).unwrap();
    match out {
        AffinityIntent::RandomSubset { from, count } => {
            assert_eq!(from, pool, "RandomSubset.from must round-trip");
            assert_eq!(count, 2, "RandomSubset.count must round-trip");
        }
        other => panic!("expected RandomSubset, got {other:?}"),
    }
}

/// Invariant: `intent_for_spawn` must propagate every
/// `Err` from the inner `resolve_affinity_for_cgroup` rather than
/// swallowing it. These tests pin the `?`-propagation path for
/// each unsatisfiable bail.
#[test]
fn intent_for_spawn_propagates_random_empty_intersection_bail() {
    let vmt = crate::vmm::topology::Topology::new(1, 2, 2, 2);
    let t = crate::topology::TestTopology::from_vm_topology(&vmt);
    let empty_cpuset: BTreeSet<usize> = BTreeSet::new();
    let intent = AffinityIntent::random_subset(t.all_cpus().iter().copied(), 1);
    let err = intent_for_spawn(&intent, Some(&empty_cpuset), &t)
        .expect_err("RandomSubset with empty cpuset intersection must bail");
    let msg = err.to_string();
    assert!(
        msg.contains("AffinityIntent::RandomSubset") && msg.contains("after intersecting"),
        "bail diag should name the intent and the intersection, got: {msg}"
    );
}

#[test]
fn intent_for_spawn_propagates_random_zero_count_bail() {
    let vmt = crate::vmm::topology::Topology::new(1, 2, 2, 2);
    let t = crate::topology::TestTopology::from_vm_topology(&vmt);
    let pool: BTreeSet<usize> = [0usize, 1].into_iter().collect();
    let intent = AffinityIntent::random_subset(pool.iter().copied(), 0);
    let err = intent_for_spawn(&intent, None, &t).expect_err("RandomSubset { count: 0 } must bail");
    let msg = err.to_string();
    assert!(
        msg.contains("count=0"),
        "bail diag should name the count=0 condition, got: {msg}"
    );
}

#[test]
fn intent_for_spawn_propagates_exact_empty_bail() {
    let vmt = crate::vmm::topology::Topology::new(1, 2, 2, 2);
    let t = crate::topology::TestTopology::from_vm_topology(&vmt);
    let intent = AffinityIntent::Exact(BTreeSet::new());
    let err = intent_for_spawn(&intent, None, &t).expect_err("Exact(empty) must bail");
    let msg = err.to_string();
    assert!(
        msg.contains("AffinityIntent::Exact(BTreeSet::new())"),
        "bail diag should name the empty Exact, got: {msg}"
    );
}

#[test]
fn intent_for_spawn_propagates_single_cpu_empty_cpuset_bail() {
    let vmt = crate::vmm::topology::Topology::new(1, 2, 2, 2);
    let t = crate::topology::TestTopology::from_vm_topology(&vmt);
    let empty_cpuset: BTreeSet<usize> = BTreeSet::new();
    let err = intent_for_spawn(&AffinityIntent::SingleCpu, Some(&empty_cpuset), &t)
        .expect_err("SingleCpu with empty cpuset must bail");
    let msg = err.to_string();
    assert!(
        msg.contains("AffinityIntent::SingleCpu") && msg.contains("empty"),
        "bail diag should name the intent and the empty cpuset, got: {msg}"
    );
}

#[test]
fn intent_for_spawn_propagates_llc_aligned_no_overlap_bail() {
    let vmt = crate::vmm::topology::Topology::new(1, 2, 2, 2);
    let t = crate::topology::TestTopology::from_vm_topology(&vmt);
    let empty_cpuset: BTreeSet<usize> = BTreeSet::new();
    let err = intent_for_spawn(&AffinityIntent::LlcAligned, Some(&empty_cpuset), &t)
        .expect_err("LlcAligned with empty cpuset must bail");
    let msg = err.to_string();
    assert!(
        msg.contains("AffinityIntent::LlcAligned") && msg.contains("No LLC has any CPU"),
        "bail diag should name the intent and the no-overlap failure mode, got: {msg}"
    );
}

#[test]
fn intent_for_spawn_propagates_exact_disjoint_bail() {
    let vmt = crate::vmm::topology::Topology::new(1, 2, 2, 2);
    let t = crate::topology::TestTopology::from_vm_topology(&vmt);
    // Cpuset = {0, 1}; Exact = {6, 7}. Both subsets of all_cpuset
    // (so the empty-Exact bail doesn't fire) but their intersection
    // with cpuset is empty.
    let cpuset: BTreeSet<usize> = [0, 1].into_iter().collect();
    let exact: BTreeSet<usize> = [6, 7].into_iter().collect();
    let err = intent_for_spawn(&AffinityIntent::Exact(exact), Some(&cpuset), &t)
        .expect_err("Exact disjoint from cpuset must bail");
    let msg = err.to_string();
    assert!(
        msg.contains("AffinityIntent::Exact") && msg.contains("disjoint"),
        "bail diag should name the intent and the disjoint failure mode, got: {msg}"
    );
}

#[test]
fn intent_for_spawn_propagates_smt_sibling_pair_bail() {
    // synthetic() leaves LlcInfo::cores empty so SmtSiblingPair's
    // walker finds no core with two siblings in the pool and bails.
    let t = crate::topology::TestTopology::synthetic(8, 2);
    let err = intent_for_spawn(&AffinityIntent::SmtSiblingPair, None, &t)
        .expect_err("SmtSiblingPair on a topology without per-core sibling data must bail");
    let msg = err.to_string();
    assert!(
        msg.contains("AffinityIntent::SmtSiblingPair") && msg.contains("two SMT siblings"),
        "bail diag should name the intent and the no-SMT-pair failure mode, got: {msg}"
    );
}

/// Exact non-empty against cpuset=Some(empty) must fall through to
/// the disjoint bail (since cpus ∩ empty = empty), NOT to the
/// empty-Exact bail (the Exact set itself is non-empty). Pin the
/// path so a refactor that swaps the bail order doesn't silently
/// change the operator-facing message.
#[test]
fn resolve_affinity_exact_nonempty_disjoint_from_empty_cpuset_bails() {
    let t = crate::topology::TestTopology::synthetic(4, 1);
    let empty_cpuset: BTreeSet<usize> = BTreeSet::new();
    let exact: BTreeSet<usize> = [0usize, 1].into_iter().collect();
    let err = resolve_affinity_for_cgroup(
        &AffinityIntent::Exact(exact.clone()),
        Some(&empty_cpuset),
        &t,
    )
    .expect_err("Exact non-empty against empty cpuset must bail");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("disjoint"),
        "bail diag should name the disjoint failure mode (NOT the empty-Exact form, since the Exact set itself is non-empty): got {msg}",
    );
    assert!(
        msg.contains("empty cpuset"),
        "bail diag should render the empty cpuset via format_cpuset_for_diag: got {msg}",
    );
}

/// LlcAligned with a non-empty cpuset that doesn't overlap ANY LLC
/// must bail. The existing `_disjoint_cpuset_bails` test uses
/// cpuset=empty; this one uses a non-empty cpuset referencing CPUs
/// outside topo.all_cpuset() so the bail fires on the
/// non-empty-but-still-disjoint path.
#[test]
fn resolve_affinity_llc_aligned_nonempty_disjoint_cpuset_bails() {
    // LLC 0 = {0,1,2,3}, LLC 1 = {4,5,6,7}. Cpuset = {99} (a CPU
    // ID outside any LLC's range). Every LLC's intersection with
    // the pool is empty → bail.
    let t = crate::topology::TestTopology::synthetic(8, 2);
    let cpuset: BTreeSet<usize> = [99usize].into_iter().collect();
    let err = resolve_affinity_for_cgroup(&AffinityIntent::LlcAligned, Some(&cpuset), &t)
        .expect_err("LlcAligned with non-empty disjoint cpuset must bail");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("AffinityIntent::LlcAligned"),
        "diagnostic must name the intent variant: got {msg}",
    );
    assert!(
        msg.contains("No LLC has any CPU"),
        "diagnostic must name the failure mode: got {msg}",
    );
    // Pin that the cpuset is rendered with its CPU IDs (not the
    // "<no cpuset>" sentinel) so format_cpuset_for_diag picked
    // the Some(non_empty) branch.
    assert!(
        msg.contains("99"),
        "bail diag should render the non-empty cpuset's CPU IDs via format_cpuset_for_diag: got {msg}",
    );
}

#[test]
fn settled_hold_returns_settle_plus_fraction_of_duration() {
    let cg = crate::cgroup::CgroupManager::new("/nonexistent");
    let topo = crate::topology::TestTopology::synthetic(1, 1);
    let ctx = Ctx::builder(&cg, &topo)
        .settle(Duration::from_millis(200))
        .duration(Duration::from_secs(1))
        .build();
    assert_eq!(
        ctx.settled_hold(0.5),
        crate::scenario::ops::HoldSpec::fixed(Duration::from_millis(700)),
    );
}

#[test]
fn settled_hold_full_fraction_returns_settle_plus_full_duration() {
    let cg = crate::cgroup::CgroupManager::new("/nonexistent");
    let topo = crate::topology::TestTopology::synthetic(1, 1);
    let ctx = Ctx::builder(&cg, &topo)
        .settle(Duration::from_millis(100))
        .duration(Duration::from_secs(2))
        .build();
    assert_eq!(
        ctx.settled_hold(1.0),
        crate::scenario::ops::HoldSpec::fixed(Duration::from_millis(2100)),
    );
}

#[test]
fn settled_hold_zero_fraction_returns_settle_only() {
    let cg = crate::cgroup::CgroupManager::new("/nonexistent");
    let topo = crate::topology::TestTopology::synthetic(1, 1);
    let ctx = Ctx::builder(&cg, &topo)
        .settle(Duration::from_millis(500))
        .duration(Duration::from_secs(1))
        .build();
    assert_eq!(
        ctx.settled_hold(0.0),
        crate::scenario::ops::HoldSpec::fixed(Duration::from_millis(500)),
    );
}

#[test]
fn settled_hold_third_fraction_matches_integer_division() {
    let cg = crate::cgroup::CgroupManager::new("/nonexistent");
    let topo = crate::topology::TestTopology::synthetic(1, 1);
    let ctx = Ctx::builder(&cg, &topo)
        .settle(Duration::from_millis(100))
        .duration(Duration::from_secs(3))
        .build();
    // 3s * (1/3) under mul_f64 may produce 2.999_999_999s due to
    // f64 representation of 1/3. The hold-evaluation boundary
    // tolerates the sub-nanosecond drift; the test pins that
    // the result stays within 1 nanosecond of the integer-div
    // formulation (3s / 3 = 1s exact, + 100ms settle = 1100ms).
    let hold = ctx.settled_hold(1.0 / 3.0);
    let crate::scenario::ops::HoldSpec::Fixed(d) = hold else {
        panic!("expected Fixed variant, got {hold:?}");
    };
    let expected = Duration::from_millis(1100);
    let diff = d.abs_diff(expected);
    assert!(
        diff <= Duration::from_nanos(1),
        "settled_hold(1.0/3.0) drift > 1ns: {d:?} vs {expected:?}",
    );
}

#[test]
#[should_panic(expected = "cannot convert float seconds to Duration")]
fn settled_hold_panics_on_nan() {
    let cg = crate::cgroup::CgroupManager::new("/nonexistent");
    let topo = crate::topology::TestTopology::synthetic(1, 1);
    let ctx = Ctx::builder(&cg, &topo)
        .settle(Duration::from_millis(100))
        .duration(Duration::from_secs(1))
        .build();
    let _ = ctx.settled_hold(f64::NAN);
}

/// `ctx.cgroup_def(name)` produces a [`CgroupDef`](crate::scenario::ops::CgroupDef) carrying the
/// builder's `workers_per_cgroup` value verbatim. Pin so a
/// regression that pulled from a stale field or hardcoded a
/// literal would surface here.
#[test]
fn cgroup_def_carries_workers_per_cgroup() {
    let cg = crate::cgroup::CgroupManager::new("/nonexistent");
    let topo = crate::topology::TestTopology::synthetic(2, 1);
    let ctx = Ctx::builder(&cg, &topo).workers_per_cgroup(7).build();
    let def = ctx.cgroup_def("cg_0");
    assert_eq!(def.name, "cg_0", "name must thread through verbatim");
    assert_eq!(
        def.works.len(),
        1,
        "the default workers(n) call seeds exactly one WorkSpec; \
             a regression that doubled or skipped the seeding would \
             surface here: {:?}",
        def.works,
    );
    assert_eq!(
        def.works[0].num_workers,
        Some(7),
        "WorkSpec must carry ctx.workers_per_cgroup as num_workers: {:?}",
        def.works[0],
    );
}

/// Default `workers_per_cgroup = 1` from `CtxBuilder` flows
/// through unchanged. A regression that defaulted differently
/// would surface here even when the test author doesn't set
/// `workers_per_cgroup` explicitly.
#[test]
fn cgroup_def_default_workers_per_cgroup_is_one() {
    let cg = crate::cgroup::CgroupManager::new("/nonexistent");
    let topo = crate::topology::TestTopology::synthetic(2, 1);
    let ctx = Ctx::builder(&cg, &topo).build();
    let def = ctx.cgroup_def("cg_default");
    assert_eq!(def.works[0].num_workers, Some(1));
}

/// The helper accepts both `&'static str` and `String` for the
/// name argument (via `Into<Cow<'static, str>>`), matching
/// `CgroupDef::named`'s signature so the migration is
/// type-compatible at every call site.
#[test]
fn cgroup_def_accepts_static_str_and_string() {
    let cg = crate::cgroup::CgroupManager::new("/nonexistent");
    let topo = crate::topology::TestTopology::synthetic(2, 1);
    let ctx = Ctx::builder(&cg, &topo).build();
    let from_static = ctx.cgroup_def("static_name");
    let from_string = ctx.cgroup_def(String::from("owned_name"));
    assert_eq!(from_static.name, "static_name");
    assert_eq!(from_string.name, "owned_name");
}

/// Helper-returned [`CgroupDef`] chains additional builders
/// (`.cpuset`, `.work_type`) — the docstring promises this
/// composability so the migration can collapse multi-line
/// chains. A regression to a newtype wrapper or `&CgroupDef`
/// return that broke move-based chaining would surface here.
#[test]
fn cgroup_def_chains_further_builders() {
    use crate::scenario::ops::CpusetSpec;
    use crate::workload::WorkType;
    let cg = crate::cgroup::CgroupManager::new("/nonexistent");
    let topo = crate::topology::TestTopology::synthetic(4, 1);
    let ctx = Ctx::builder(&cg, &topo).workers_per_cgroup(3).build();
    let def = ctx
        .cgroup_def("cg_chained")
        .cpuset(CpusetSpec::range(0.0, 1.0))
        .work_type(WorkType::SpinWait);
    assert_eq!(def.name, "cg_chained");
    assert_eq!(
        def.works[0].num_workers,
        Some(3),
        "ctx default propagates through subsequent chained builders",
    );
    assert!(
        def.cpuset.is_some(),
        "cpuset chains correctly after the helper",
    );
    assert!(
        matches!(def.works[0].work_type, WorkType::SpinWait),
        "work_type chains after the helper-seeded WorkSpec; got {:?}",
        def.works[0].work_type,
    );
}

/// `ctx.workers_per_cgroup = 0` passes through verbatim — the
/// helper does not reject or substitute. Matches the pre-helper
/// inlined `CgroupDef::named(name).workers(0)` semantics so
/// migration is lossless. For an empty move-target cgroup
/// without workers, [`CgroupDef::named`]'s docstring directs
/// authors to [`Op::AddCgroup`] instead — that idiomatic path is
/// orthogonal to this helper's dedup contract.
#[test]
fn cgroup_def_passes_through_zero_workers_per_cgroup() {
    let cg = crate::cgroup::CgroupManager::new("/nonexistent");
    let topo = crate::topology::TestTopology::synthetic(2, 1);
    let ctx = Ctx::builder(&cg, &topo).workers_per_cgroup(0).build();
    let def = ctx.cgroup_def("zero_workers");
    assert_eq!(
        def.works[0].num_workers,
        Some(0),
        "helper preserves Ctx::workers_per_cgroup=0 verbatim; \
             for an empty move-target cgroup use Op::AddCgroup per \
             CgroupDef::named's docstring",
    );
}

/// A trailing `.workers(N)` overrides the helper's default —
/// produces the same [`CgroupDef`](crate::scenario::ops::CgroupDef) as [`CgroupDef::named`](crate::scenario::ops::CgroupDef::named)`(name).workers(N)`.
/// Pin so a regression that appended a SECOND WorkSpec instead
/// of replacing num_workers on works[0] would surface here.
#[test]
fn cgroup_def_override_workers_replaces_default() {
    let cg = crate::cgroup::CgroupManager::new("/nonexistent");
    let topo = crate::topology::TestTopology::synthetic(2, 1);
    let ctx = Ctx::builder(&cg, &topo).workers_per_cgroup(3).build();
    let def = ctx.cgroup_def("cg_override").workers(99);
    assert_eq!(
        def.works[0].num_workers,
        Some(99),
        "explicit workers(N) overrides ctx default",
    );
    assert_eq!(def.works.len(), 1, "no second WorkSpec is added");
}

/// A subsequent `.work(WorkSpec::default().workers(N))` APPENDS
/// a second WorkSpec rather than replacing the helper's seed.
/// Pin the composition shape so a regression that overwrote
/// works[0] (instead of pushing) would surface here.
#[test]
fn cgroup_def_with_second_workspec_preserves_helper_seed() {
    use crate::workload::WorkSpec;
    let cg = crate::cgroup::CgroupManager::new("/nonexistent");
    let topo = crate::topology::TestTopology::synthetic(2, 1);
    let ctx = Ctx::builder(&cg, &topo).workers_per_cgroup(3).build();
    let def = ctx
        .cgroup_def("cg_two_works")
        .work(WorkSpec::default().workers(5));
    assert_eq!(
        def.works.len(),
        2,
        "second WorkSpec appended, not replaced; helper seed stays at works[0]",
    );
    assert_eq!(
        def.works[0].num_workers,
        Some(3),
        "helper's ctx default preserved as works[0]",
    );
    assert_eq!(
        def.works[1].num_workers,
        Some(5),
        "appended WorkSpec lands as works[1]",
    );
}
