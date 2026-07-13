use super::super::*;
use super::*;

/// `CpuCap::new(1)` succeeds — minimum legal cap.
#[test]
fn cpu_cap_new_accepts_one() {
    let cap = CpuCap::new(1).expect("cap of 1 must succeed");
    assert_eq!(cap.effective_count(4).unwrap(), 1);
}

/// `CpuCap::new(usize::MAX)` is accepted at construction time
/// and clamped later by `effective_count`. Pins the contract
/// that construction never consults the host.
#[test]
fn cpu_cap_new_accepts_usize_max() {
    let cap = CpuCap::new(usize::MAX).expect("MAX accepted at construction");
    // Actual clamping surfaces at effective_count; see
    // `cpu_cap_effective_count_exceeds_host` below.
    assert!(cap.effective_count(usize::MAX).is_ok());
}

/// `effective_count` returns the inner value when it fits.
#[test]
fn cpu_cap_effective_count_fits() {
    let cap = CpuCap::new(3).unwrap();
    assert_eq!(cap.effective_count(4).unwrap(), 3);
    assert_eq!(cap.effective_count(3).unwrap(), 3);
}

/// `effective_count` when cap exceeds the allowed-CPU count
/// returns a `CpuBudgetUnsatisfiable` hard error naming both
/// numbers. An explicit `--cpu-cap` the host cannot satisfy is a
/// user-input error (FAIL so the operator fixes the flag), NOT
/// transient contention (skip/retry).
#[test]
fn cpu_cap_effective_count_exceeds_host() {
    let cap = CpuCap::new(8).unwrap();
    let err = cap.effective_count(4).expect_err("8 > 4 must error");
    let msg = format!("{err:#}");
    assert!(msg.contains("8"), "msg must name requested cap: {msg}");
    assert!(msg.contains("4"), "msg must name allowed-CPU count: {msg}");
    // Must downcast to CpuBudgetUnsatisfiable: a hard FAIL (the typed
    // cap does not fit the host), NOT a ResourceContention skip/retry.
    assert!(
        err.downcast_ref::<CpuBudgetUnsatisfiable>().is_some(),
        "must be a CpuBudgetUnsatisfiable hard error: {msg}",
    );
}

/// `effective_count` at the boundary: cap == allowed_cpus is OK.
#[test]
fn cpu_cap_effective_count_at_host_boundary() {
    let cap = CpuCap::new(4).unwrap();
    assert_eq!(cap.effective_count(4).unwrap(), 4);
}

/// CLI flag supplied → wins over env var. `resolve(Some(N))`
/// ignores `KTSTR_CPU_CAP` entirely. Pins the precedence
/// contract documented on `CpuCap::resolve`.
#[test]
fn cpu_cap_resolve_cli_wins_over_env() {
    let _lock = env_lock();
    let _env = EnvGuard::set(crate::KTSTR_CPU_CAP_ENV, "99");
    let cap = CpuCap::resolve(Some(3)).unwrap().expect("CLI flag set");
    assert_eq!(cap.effective_count(4).unwrap(), 3, "CLI wins");
}

/// No CLI flag, no env var → `None` (the 30%-of-allowed default
/// is applied at acquire time — `resolve` never synthesizes a
/// cap here).
#[test]
fn cpu_cap_resolve_no_cli_no_env_returns_none() {
    let _lock = env_lock();
    let _env = EnvGuard::remove(crate::KTSTR_CPU_CAP_ENV);
    assert!(CpuCap::resolve(None).unwrap().is_none());
}

/// Env var set to a valid integer, no CLI flag → resolves to
/// that value.
#[test]
fn cpu_cap_resolve_env_set() {
    let _lock = env_lock();
    let _env = EnvGuard::set(crate::KTSTR_CPU_CAP_ENV, "2");
    let cap = CpuCap::resolve(None)
        .expect("resolve must succeed")
        .expect("env-set cap must yield Some");
    assert_eq!(cap.effective_count(8).unwrap(), 2);
}

/// Env var set to the empty string → treated as absent
/// (matches `Ok(s) if s.is_empty()` arm).
#[test]
fn cpu_cap_resolve_empty_env_is_absent() {
    let _lock = env_lock();
    let _env = EnvGuard::set(crate::KTSTR_CPU_CAP_ENV, "");
    assert!(CpuCap::resolve(None).unwrap().is_none());
}

/// Env var set to a non-numeric value → parse error with the
/// variable name in the message.
#[test]
fn cpu_cap_resolve_non_numeric_env_errors() {
    let _lock = env_lock();
    let _env = EnvGuard::set(crate::KTSTR_CPU_CAP_ENV, "not-a-number");
    let err = CpuCap::resolve(None).expect_err("non-numeric must error");
    let msg = format!("{err:#}");
    assert!(msg.contains(crate::KTSTR_CPU_CAP_ENV), "msg={msg}");
}

/// Env var set to `"0"` flows through `CpuCap::new(0)` and
/// surfaces the same "--cpu-cap must be ≥ 1 CPU (got 0)" error.
/// Regression guard: typos like `KTSTR_CPU_CAP=0` must NOT
/// silently fall back to "no cap".
#[test]
fn cpu_cap_resolve_zero_env_rejected() {
    let _lock = env_lock();
    let _env = EnvGuard::set(crate::KTSTR_CPU_CAP_ENV, "0");
    let err = CpuCap::resolve(None).expect_err("zero must error");
    let msg = format!("{err:#}");
    assert!(msg.contains("≥ 1"), "msg={msg}");
    assert!(msg.contains("got 0"), "msg={msg}");
}

/// CLI flag of 0 is the same rejection path as env var of 0 —
/// both feed `CpuCap::new(0)`. Pins that precedence doesn't
/// let a valid env var "save" an invalid CLI zero.
#[test]
fn cpu_cap_resolve_zero_cli_rejected_even_with_valid_env() {
    let _lock = env_lock();
    let _env = EnvGuard::set(crate::KTSTR_CPU_CAP_ENV, "2");
    let err = CpuCap::resolve(Some(0)).expect_err("cli=0 must error");
    let msg = format!("{err:#}");
    assert!(msg.contains("≥ 1"), "msg={msg}");
}

/// `EnvGuard::set` applies the value, and `Drop` removes the
/// variable even if the test body panics mid-scope. Pins the
/// RAII contract so a refactor that accidentally drops the
/// Drop impl leaks env state across tests.
#[test]
fn env_guard_set_and_drop_removes_variable() {
    let _lock = env_lock();
    let probe = "KTSTR_CPU_CAP_ENV_GUARD_TEST";
    {
        let _env = EnvGuard::set(probe, "abc");
        assert_eq!(
            std::env::var(probe).ok().as_deref(),
            Some("abc"),
            "set must apply immediately",
        );
    }
    // Drop ran — variable must be gone.
    assert!(
        std::env::var(probe).is_err(),
        "EnvGuard::drop must remove the variable",
    );
}

/// Single-node host: one entry in host_llcs_by_numa_node with
/// every LLC index in ascending order.
#[test]
fn host_llcs_by_numa_node_single_node() {
    let topo = synth_host_topo(&[(vec![0, 1], 0), (vec![2, 3], 0), (vec![4, 5], 0)]);
    let map = topo.host_llcs_by_numa_node();
    assert_eq!(map.len(), 1, "single-node host has one entry");
    assert_eq!(map.get(&0), Some(&vec![0, 1, 2]));
}

/// Dual-node host: two entries, each with its own LLC indices
/// in ascending order.
#[test]
fn host_llcs_by_numa_node_dual_node() {
    let topo = synth_host_topo(&[
        (vec![0, 1], 0),
        (vec![2, 3], 1),
        (vec![4, 5], 0),
        (vec![6, 7], 1),
    ]);
    let map = topo.host_llcs_by_numa_node();
    assert_eq!(map.len(), 2);
    assert_eq!(map.get(&0), Some(&vec![0, 2]));
    assert_eq!(map.get(&1), Some(&vec![1, 3]));
}

/// Asymmetric: node 0 has 3 LLCs, node 1 has 1 LLC.
/// `numa_nodes_with_capacity(2)` returns only node 0.
#[test]
fn numa_nodes_with_capacity_asymmetric() {
    let topo = synth_host_topo(&[(vec![0], 0), (vec![1], 0), (vec![2], 0), (vec![3], 1)]);
    let cap2: Vec<usize> = topo
        .numa_nodes_with_capacity(2)
        .into_iter()
        .map(|(node, _)| node)
        .collect();
    assert_eq!(cap2, vec![0], "only node 0 has ≥ 2 LLCs");
    let cap1: Vec<usize> = topo
        .numa_nodes_with_capacity(1)
        .into_iter()
        .map(|(node, _)| node)
        .collect();
    assert_eq!(cap1, vec![0, 1], "both nodes have ≥ 1 LLC");
}

/// `numa_nodes_with_capacity` with min_llcs > every node's
/// count returns empty — no candidates.
#[test]
fn numa_nodes_with_capacity_over_max_returns_empty() {
    let topo = synth_host_topo(&[(vec![0], 0), (vec![1], 1)]);
    assert!(topo.numa_nodes_with_capacity(99).is_empty());
}

/// `numa_nodes_sorted_by_distance` with identity closure:
/// anchor == node → 10, else 20. Anchor sorts first; remaining
/// nodes preserve BTreeMap ascending order (stable sort over
/// equal distances).
#[test]
fn numa_nodes_sorted_by_distance_identity_closure() {
    let topo = synth_host_topo(&[(vec![0], 0), (vec![1], 1), (vec![2], 2)]);
    let order = topo.numa_nodes_sorted_by_distance(1, |from, to| if from == to { 10 } else { 20 });
    // Anchor node 1 first; nodes 0 and 2 tied at distance 20,
    // stable over BTreeMap-ascending order.
    assert_eq!(order[0], 1, "anchor node first");
    assert_eq!(
        &order[1..],
        &[0, 2],
        "tied-distance nodes in ascending order"
    );
}

/// `numa_nodes_sorted_by_distance` demotes unreachable nodes
/// (distance 255 per Linux convention) to the end even when
/// the node has LLCs. Pins the unreachable-last contract.
#[test]
fn numa_nodes_sorted_by_distance_unreachable_demoted() {
    let topo = synth_host_topo(&[(vec![0], 0), (vec![1], 1), (vec![2], 2)]);
    // Node 2 unreachable from anchor 0, node 1 at distance 20.
    let order = topo.numa_nodes_sorted_by_distance(0, |from, to| match (from, to) {
        (0, 0) => 10,
        (0, 1) => 20,
        (0, 2) => 255,
        _ => 20,
    });
    assert_eq!(order, vec![0, 1, 2]);
    // The key invariant: unreachable at end even though its
    // numeric id (2) would naturally sort mid-range.
    assert_eq!(*order.last().unwrap(), 2, "unreachable node is last");
}

/// `numa_nodes_sorted_by_distance` skips nodes not in
/// host_node_llcs — a node with no LLCs is excluded entirely.
/// "Nodes without any LLCs on this host are skipped — spilling
/// to an empty node has no value" per the doc.
#[test]
fn numa_nodes_sorted_by_distance_skips_empty_nodes() {
    // Only node 0 has LLCs. Anchor 99 never appears in output.
    let topo = synth_host_topo(&[(vec![0], 0)]);
    let order = topo.numa_nodes_sorted_by_distance(99, |_, _| 20);
    assert_eq!(order, vec![0], "only node 0 is in host_node_llcs");
}

// ---------------------------------------------------------------
// acquire_llc_plan — cap semantics (host-integration-light)
// ---------------------------------------------------------------

/// `acquire_llc_plan` with `cpu_cap == Some(cap)` and
/// `cap > allowed-CPU count` fails at `effective_count` with a
/// `CpuBudgetUnsatisfiable` hard error — before any /tmp
/// side-effects. An explicit cap the host cannot satisfy is a
/// user-input FAIL, not retry-routed contention. Pins that
/// over-cap fails cleanly without touching the lock pool. The
/// test pins a 2-CPU allowed set and caps at 3 CPUs, the minimum
/// pair that exercises the "N > allowed" branch.
#[test]
fn acquire_llc_plan_rejects_cap_over_allowed_cpus() {
    let _allowed = AllowedCpusGuard::new(vec![0, 1]);
    // Two real LLC groups (one CPU each), cap of 3 CPUs.
    let topo = synth_host_topo(&[(vec![0], 0), (vec![1], 0)]);
    let test_topo = crate::topology::TestTopology::synthetic(4, 1);
    let cap = CpuCap::new(3).unwrap();
    let err = acquire_llc_plan(&topo, &test_topo, Some(cap), PlacementPolicy::Consolidate)
        .expect_err("cap > allowed_cpus must error");
    assert!(
        err.downcast_ref::<CpuBudgetUnsatisfiable>().is_some(),
        "must be CpuBudgetUnsatisfiable: {err:#}"
    );
}

// ---------------------------------------------------------------
// BuildSandbox supplementary coverage lives in
// src/vmm/cgroup_sandbox.rs's mod tests — see
// `cpuset_sets_equal_identity`, `cpuset_sets_equal_narrower_effective`,
// `sandbox_degraded_display_text` (includes RootCgroupRefused),
// `parent_controllers_include_missing_file`, and
// `read_cpuset_effective_missing_file_returns_none`. The
// try_create RootCgroupRefused path is regression-tested via the
// `is_root_cgroup` seam test
// (`is_root_cgroup_handles_slash_empty_and_whitespace` in
// src/vmm/cgroup_sandbox.rs), in addition to the variant's
// Display coverage.
// ---------------------------------------------------------------

// ---------------------------------------------------------------
// Deadlock guards — plan_from_snapshots produces ascending
// llc_idx for livelock-proof acquire order
// ---------------------------------------------------------------

/// `plan_from_snapshots` returns selected LLC indices in
/// ascending order — pinned at step e of the algorithm. Two
/// concurrent callers with the same target see the same
/// sequence, so their `try_acquire_llc_plan_locks` walk each
/// flock in the same order. Reverse-order acquire would
/// deadlock if one caller grabbed LLC N first while another
/// grabbed LLC 0 first and they competed for each other's
/// next targets. Ascending order eliminates that possibility.
///
/// The expected output `[0, 2, 3]` catches TWO independent
/// regressions at once:
///   1. Consolidation dropped (filter on `holder_count > 0`
///      removed). Output would become `[0, 1, 2]` because the
///      fresh LLCs at indices 0 and 1 would rank equal to LLC
///      2 without the consolidation preference.
///   2. Final `sort_unstable` dropped. Output would preserve
///      the interior walk order, typically `[2, 3, 0]` once
///      consolidation promoted the peer-held LLCs.
///
/// Either regression fails this test. See
/// `plan_from_snapshots_always_ascending_across_target_range`
/// for the broader property-based guard.
#[test]
fn plan_from_snapshots_returns_ascending_indices() {
    let topo = synth_host_topo(&[(vec![0], 0), (vec![1], 0), (vec![2], 0), (vec![3], 0)]);
    // Synthetic snapshots — holder_count higher on "later"
    // LLCs so consolidation score would put them first if the
    // algorithm didn't re-sort ascending at the end.
    let snapshots: Vec<LlcSnapshot> = (0..4)
        .map(|idx| LlcSnapshot {
            llc_idx: idx,
            lockfile_path: std::path::PathBuf::from(format!("/tmp/ktstr-llc-{idx}.lock")),
            holders: Vec::new(),
            holder_count: if idx >= 2 { 5 } else { 0 },
        })
        .collect();
    let allowed: std::collections::BTreeSet<usize> = (0..4).collect();
    let selected = plan_from_snapshots(
        &snapshots,
        3,
        &topo,
        &allowed,
        |_, _| 10, // everything same-node
        PlacementPolicy::Consolidate,
    );
    // Step e of plan_from_snapshots is
    // `selected.sort_unstable()` — guarantees ascending llc_idx
    // regardless of consolidation score or seed ordering. Two
    // concurrent callers with the same snapshots see the same
    // acquire order, eliminating reverse-order deadlock.
    assert_eq!(selected, vec![0, 2, 3], "step e sorts ascending");
}

/// `plan_from_snapshots` with `target_cpus >= sum of allowed
/// CPUs across every LLC` short-circuits to "select every LLC
/// with at least one allowed CPU" in ascending order. Pins the
/// saturation-case behaviour: the CPU budget covers or exceeds
/// the total schedulable capacity, so the walk picks every
/// eligible LLC without running the scoring pass.
#[test]
fn plan_from_snapshots_target_ge_all_selects_every_llc() {
    let topo = synth_host_topo(&[(vec![0], 0), (vec![1], 1), (vec![2], 2)]);
    let snapshots: Vec<LlcSnapshot> = (0..3)
        .map(|idx| LlcSnapshot {
            llc_idx: idx,
            lockfile_path: std::path::PathBuf::from(format!("/tmp/ktstr-llc-{idx}.lock")),
            holders: Vec::new(),
            holder_count: 0,
        })
        .collect();
    let allowed: std::collections::BTreeSet<usize> = (0..3).collect();
    let selected = plan_from_snapshots(
        &snapshots,
        3,
        &topo,
        &allowed,
        |_, _| 10,
        PlacementPolicy::Consolidate,
    );
    assert_eq!(selected, vec![0, 1, 2]);
    let selected_over = plan_from_snapshots(
        &snapshots,
        999,
        &topo,
        &allowed,
        |_, _| 10,
        PlacementPolicy::Consolidate,
    );
    assert_eq!(selected_over, vec![0, 1, 2], "target > len clamps");
}

/// `plan_from_snapshots` with `target == 0` returns empty —
/// early return in the algorithm. Pins the degenerate case
/// so a future "optimization" that assumes selected[0] exists
/// fails here first.
#[test]
fn plan_from_snapshots_target_zero_returns_empty() {
    let topo = synth_host_topo(&[(vec![0], 0)]);
    let snapshots: Vec<LlcSnapshot> = vec![LlcSnapshot {
        llc_idx: 0,
        lockfile_path: std::path::PathBuf::from("/tmp/ktstr-llc-0.lock"),
        holders: Vec::new(),
        holder_count: 0,
    }];
    let allowed: std::collections::BTreeSet<usize> = [0].into_iter().collect();
    let selected = plan_from_snapshots(
        &snapshots,
        0,
        &topo,
        &allowed,
        |_, _| 10,
        PlacementPolicy::Consolidate,
    );
    assert!(selected.is_empty());
}

/// `plan_from_snapshots` prefers LLCs with `holder_count > 0`
/// over fresh LLCs on the same NUMA node — the consolidation
/// half of the composite sort ("consolidation candidates
/// first, then fresh candidates"). Two same-node LLCs,
/// holder_count [0, 5],
/// target=1 → must pick the holder=5 LLC (index 1), not the
/// fresh one (index 0). A future bug that flipped the partition
/// order (fresh-first) or dropped the holder_count tiebreaker
/// would pick LLC 0 instead and fail this test.
///
/// Distinct from `plan_from_snapshots_returns_ascending_indices`
/// which only asserted the post-sort ordering — that test
/// accepted EITHER consolidation ordering because its output
/// happened to be ascending in both cases. This one rejects
/// the non-consolidation output.
#[test]
fn plan_from_snapshots_prefers_higher_holder_count() {
    let topo = synth_host_topo(&[(vec![0], 0), (vec![1], 0)]);
    let snapshots: Vec<LlcSnapshot> = vec![
        LlcSnapshot {
            llc_idx: 0,
            lockfile_path: std::path::PathBuf::from("/tmp/ktstr-llc-0.lock"),
            holders: Vec::new(),
            holder_count: 0,
        },
        LlcSnapshot {
            llc_idx: 1,
            lockfile_path: std::path::PathBuf::from("/tmp/ktstr-llc-1.lock"),
            holders: Vec::new(),
            holder_count: 5,
        },
    ];
    // Same-node distance closure so placement doesn't bias by
    // NUMA — isolates the consolidation preference signal.
    let allowed: std::collections::BTreeSet<usize> = (0..2).collect();
    let selected = plan_from_snapshots(
        &snapshots,
        1,
        &topo,
        &allowed,
        |_, _| 10,
        PlacementPolicy::Consolidate,
    );
    assert_eq!(
        selected,
        vec![1],
        "target=1 with holders [0,5] must pick LLC 1 \
         (consolidation preference), not LLC 0 (fresh)"
    );
}

/// Invariant-based ascending-order property: for every target
/// in 1..=snapshots.len(), `selected.windows(2)` all satisfy
/// `w[0] < w[1]`. This pins the step-e sort_unstable invariant
/// independent of the consolidation / node-spill traversal —
/// a future refactor that restructures the inner walk but
/// forgets the final sort will fail this test at SOME target,
/// not just the specific one `_returns_ascending_indices` pins.
#[test]
fn plan_from_snapshots_always_ascending_across_target_range() {
    let topo = synth_host_topo(&[(vec![0], 0), (vec![1], 1), (vec![2], 0), (vec![3], 1)]);
    // Mixed holder_counts so consolidation ordering varies.
    let snapshots: Vec<LlcSnapshot> = vec![
        LlcSnapshot {
            llc_idx: 0,
            lockfile_path: std::path::PathBuf::from("/tmp/ktstr-llc-0.lock"),
            holders: Vec::new(),
            holder_count: 3,
        },
        LlcSnapshot {
            llc_idx: 1,
            lockfile_path: std::path::PathBuf::from("/tmp/ktstr-llc-1.lock"),
            holders: Vec::new(),
            holder_count: 0,
        },
        LlcSnapshot {
            llc_idx: 2,
            lockfile_path: std::path::PathBuf::from("/tmp/ktstr-llc-2.lock"),
            holders: Vec::new(),
            holder_count: 7,
        },
        LlcSnapshot {
            llc_idx: 3,
            lockfile_path: std::path::PathBuf::from("/tmp/ktstr-llc-3.lock"),
            holders: Vec::new(),
            holder_count: 1,
        },
    ];
    let allowed: std::collections::BTreeSet<usize> = (0..4).collect();
    // Each LLC has 1 CPU, so target_cpus == #LLCs to select. The
    // ascending-order invariant is agnostic to CPU-count vs
    // LLC-count semantics — the post-step-e sort holds regardless.
    for target_cpus in 1..=snapshots.len() {
        let selected = plan_from_snapshots(
            &snapshots,
            target_cpus,
            &topo,
            &allowed,
            |_, _| 10,
            PlacementPolicy::Consolidate,
        );
        assert_eq!(
            selected.len(),
            target_cpus,
            "target_cpus={target_cpus} must produce {target_cpus} selections, got {selected:?}"
        );
        assert!(
            selected.windows(2).all(|w| w[0] < w[1]),
            "target_cpus={target_cpus}: selection {selected:?} is not strictly ascending",
        );
    }
}

/// `make_jobs_for_plan` returns `plan.cpus.len().max(1)` so the
/// `-jN` hint to make matches the reserved CPU count — gcc
/// doesn't fan out beyond the cgroup budget.
#[test]
fn make_jobs_for_plan_matches_cpu_count() {
    let plan = LlcPlan {
        locked_llcs: vec![0, 1],
        cpus: vec![0, 1, 2, 3],
        mems: std::collections::BTreeSet::new(),
        snapshot: Vec::new(),
        locks: Vec::new(),
    };
    assert_eq!(make_jobs_for_plan(&plan), 4);
}

/// Edge: empty `plan.cpus` must yield `1`, never `0` — `make
/// -j0` on GNU make produces unbounded parallelism, exactly
/// the pathology the cap is supposed to prevent. The `.max(1)`
/// floor pins this.
#[test]
fn make_jobs_for_plan_empty_cpus_floors_to_one() {
    let plan = LlcPlan {
        locked_llcs: Vec::new(),
        cpus: Vec::new(),
        mems: std::collections::BTreeSet::new(),
        snapshot: Vec::new(),
        locks: Vec::new(),
    };
    assert_eq!(
        make_jobs_for_plan(&plan),
        1,
        "empty-cpus must floor to 1, not 0 — -j0 is unbounded",
    );
}

/// `format_llc_list` renders LLC indices with per-entry NUMA
/// node annotation when `cpu_to_node` is populated. Two
/// locked LLCs on different nodes → "0 (node 0), 2 (node 1)".
#[test]
fn format_llc_list_with_numa_info() {
    let topo = synth_host_topo(&[(vec![0], 0), (vec![1], 0), (vec![2], 1), (vec![3], 1)]);
    let rendered = format_llc_list(&[0, 2], &topo);
    assert!(
        rendered.contains("0 (node 0)"),
        "must annotate LLC 0 with its node: {rendered}",
    );
    assert!(
        rendered.contains("2 (node 1)"),
        "must annotate LLC 2 with its node: {rendered}",
    );
    // Full bracket form — enforces "[...]" wrapping so the
    // warning message reads naturally.
    assert_eq!(rendered, "[0 (node 0), 2 (node 1)]");
}

/// `format_llc_list` single-LLC case — no comma, no cross-node
/// spill, bracket-wrapped. Pins the rendering shape for the
/// warning that fires on non-spilling plans (which don't
/// actually emit the cross-node warning, but the helper may
/// still be called by future tooling).
#[test]
fn format_llc_list_single_llc() {
    let topo = synth_host_topo(&[(vec![0], 0)]);
    let rendered = format_llc_list(&[0], &topo);
    assert_eq!(rendered, "[0 (node 0)]");
}

/// `format_llc_list` on a degraded host with empty
/// `cpu_to_node` drops the `(node N)` annotation per the doc
/// ("[0, 2] on degraded hosts whose cpu_to_node map is empty").
/// Synth helper populates cpu_to_node — mimic the degraded
/// case by clearing it before calling.
#[test]
fn format_llc_list_without_numa_info() {
    let mut topo = synth_host_topo(&[(vec![0], 0), (vec![1], 0)]);
    topo.cpu_to_node.clear();
    let rendered = format_llc_list(&[0, 1], &topo);
    assert_eq!(
        rendered, "[0, 1]",
        "degraded-host form drops node annotation"
    );
}

/// `should_warn_cross_node` polarity pin: empty set or
/// single-node set → false; two or more nodes → true.
/// Splits the decision out of the eprintln! side-channel so
/// regression tests can assert the condition without capturing
/// stderr.
#[test]
fn should_warn_cross_node_polarity() {
    use std::collections::BTreeSet;
    let empty: BTreeSet<usize> = BTreeSet::new();
    assert!(
        !should_warn_cross_node(&empty),
        "empty mems must NOT warn (degenerate plan with no NUMA info)",
    );
    let single: BTreeSet<usize> = [0].into_iter().collect();
    assert!(
        !should_warn_cross_node(&single),
        "single-node plan must NOT warn — the whole point of the cap \
         is to fit on one node when possible",
    );
    let dual: BTreeSet<usize> = [0, 1].into_iter().collect();
    assert!(
        should_warn_cross_node(&dual),
        "two-node plan MUST warn — operator picked a cap that \
         couldn't fit on one node and deserves to hear about it",
    );
    let triple: BTreeSet<usize> = [0, 1, 2].into_iter().collect();
    assert!(
        should_warn_cross_node(&triple),
        "three-node plan MUST warn — same rationale as dual",
    );
}

/// `warn_if_cross_node_spill` is a thin `eprintln!` wrapper over
/// [`cross_node_spill_warning`], which holds the gate-and-format
/// logic and returns the EXACT bytes the wrapper emits (sans
/// newline). Asserting on that `Option<String>` pins both halves of
/// the contract that the wrapper alone can only smoke-test:
///   - multi-node plan → `Some(message)` whose body contains the
///     `format_llc_list` rendering (`0 (node 0)`, `1 (node 1)`) and
///     the NUMA node count, so a refactor that dropped the
///     `format_llc_list` call, miscounted `mems.len()`, or inverted
///     the gate would change the returned string and fail here;
///   - single-node plan → `None`, so a refactor that flipped the
///     predicate to fire on single-node plans surfaces as an
///     unexpected `Some`.
///
/// `warn_if_cross_node_spill` itself is still invoked to prove it
/// stays a pure no-op on the single-node path (it must not panic and
/// has no other observable side effect when the warning is `None`).
#[test]
fn warn_if_cross_node_spill_predicate_gates_stderr() {
    let topo = synth_host_topo(&[(vec![0], 0), (vec![1], 1)]);
    let multi_plan = LlcPlan {
        locked_llcs: vec![0, 1],
        cpus: vec![0, 1],
        mems: [0usize, 1].into_iter().collect(),
        snapshot: Vec::new(),
        locks: Vec::new(),
    };
    let msg = cross_node_spill_warning(&multi_plan, &topo)
        .expect("multi-node plan must produce a warning");
    // The message must equal exactly what the wrapper eprintln!s and
    // must thread the rendered LLC list + the node count through.
    let expected = format!(
        "ktstr: reserving LLCs {list} across {n} NUMA nodes \
         (preferred single-node contiguous unavailable). Build \
         will run; memory-access latency may be higher.",
        list = format_llc_list(&multi_plan.locked_llcs, &topo),
        n = multi_plan.mems.len(),
    );
    assert_eq!(msg, expected, "warning body must match the wrapper's emit");
    assert!(
        msg.contains("0 (node 0)") && msg.contains("1 (node 1)"),
        "warning must carry the format_llc_list NUMA annotations: {msg}",
    );
    assert!(
        msg.contains("across 2 NUMA nodes"),
        "warning must name the spanned node count: {msg}",
    );
    // The wrapper threads the same Some through to eprintln! — invoke
    // it to confirm the Some path does not panic.
    warn_if_cross_node_spill(&multi_plan, &topo);

    let single_plan = LlcPlan {
        locked_llcs: vec![0],
        cpus: vec![0],
        mems: [0usize].into_iter().collect(),
        snapshot: Vec::new(),
        locks: Vec::new(),
    };
    assert!(
        cross_node_spill_warning(&single_plan, &topo).is_none(),
        "single-node plan must suppress the warning (None), not emit",
    );
    // Wrapper must be a pure no-op on the None path.
    warn_if_cross_node_spill(&single_plan, &topo);
}

/// `CpuCap::new(1).effective_count(0)` errors: `n=1 > host=0`.
/// Degenerate "host has zero LLCs" edge — unlikely on a real
/// machine but critical to pin the boundary so a future bug
/// that flipped the comparison to `n >= host_llcs` (rejecting
/// cap == total) OR `n > host_llcs - 1` (overflow on 0) fails
/// here first.
#[test]
fn cpu_cap_effective_count_on_zero_llc_host() {
    let cap = CpuCap::new(1).unwrap();
    let err = cap.effective_count(0).expect_err("1 > 0 must error");
    assert!(
        err.downcast_ref::<CpuBudgetUnsatisfiable>().is_some(),
        "must be CpuBudgetUnsatisfiable: an explicit cap > host is a hard error",
    );
}

/// Multi-process concurrent `acquire_llc_plan`: a child process
/// holds `LOCK_SH` on one LLC's lockfile via `flock(1)` (SHELL
/// utility), then the parent calls `acquire_llc_plan` with a
/// cap forcing the planner to consolidate onto an LLC that has
/// holders. The consolidation invariant (`holder_count DESC`
/// ordering in `plan_from_snapshots`) requires the parent's
/// plan to include the child's LLC.
///
/// Uses `flock(1)` + `sleep 10` rather than Rust fork() so the
/// holder is a different process (different pid, different OFD)
/// than the test thread — proving the /proc/locks cross-process
/// enumeration path is exercised.
///
/// `flock(1)` is expected on every Linux host that runs ktstr
/// tests (it's in util-linux, part of the minimum viable CI
/// image). If it's absent the test short-circuits rather than
/// failing — the invariant is real but the test infrastructure
/// depends on a userspace utility.
#[test]
fn acquire_llc_plan_consolidates_on_peer_held_llc() {
    let _llc_prefix = LlcLockPrefixGuard::new();
    // 2 LLCs on the same node so NUMA-locality doesn't bias
    // against consolidation.
    let topo = HostTopology::new_for_tests(&[(vec![0], 0), (vec![1], 0)]);

    // Child process holds SH on LLC 1's lockfile via flock(1),
    // sleeping long enough for the parent's acquire to complete
    // inside the same SH window.
    let target_lock = llc_lock_path(1);
    // Ensure the lockfile exists so flock(1) opens the right
    // inode (not a fresh one that /proc/locks would attribute
    // to the flock(1) pid on a different inode than the parent
    // sees).
    crate::flock::materialize(&target_lock).expect("materialize lockfile");

    let child = std::process::Command::new("flock")
        .args(["-s", "-n", &target_lock, "sleep", "2"])
        .spawn();
    let mut child = match child {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            // flock(1) missing — skip rather than fail.
            eprintln!(
                "acquire_llc_plan_consolidates_on_peer_held_llc: \
                 flock(1) not available, skipping ({e})"
            );
            return;
        }
        Err(e) => panic!("spawn flock(1): {e}"),
    };

    // Brief sleep to let the child acquire SH before the parent
    // reads /proc/locks in discover. 50ms is well past util-linux's
    // exec + flock NB path and short enough that the child's
    // `sleep 2` still covers the parent's acquire window.
    std::thread::sleep(std::time::Duration::from_millis(50));

    let test_topo = crate::topology::TestTopology::synthetic(2, 1);
    let cap = CpuCap::new(1).expect("cap=1 valid");
    let plan = acquire_llc_plan(&topo, &test_topo, Some(cap), PlacementPolicy::Consolidate)
        .expect("SH is reentrant — parent SH must coexist with child SH");

    // Consolidation picked LLC 1 (the one with a holder) over
    // LLC 0 (fresh). The `holder_count DESC` ordering in
    // `plan_from_snapshots` makes this deterministic.
    assert_eq!(
        plan.locked_llcs,
        vec![1],
        "cap=1 with child holding SH on LLC 1 must pick LLC 1 \
         (consolidation over fresh LLC 0); got {:?}",
        plan.locked_llcs,
    );

    drop(plan);
    // Child exits naturally after sleep 2; reap it so we don't
    // leave zombies.
    let _ = child.wait();
}

/// `discover_llc_snapshots` EXCLUDES the calling process from the
/// PLAN-driving `holder_count`, but keeps it in the diagnostic
/// `holders` vec — while a PEER process's hold DOES count.
///
/// Rationale: no-perf `build()` now holds each planned LLC's
/// `LOCK_SH` from build through run (the fd strip was removed), so at
/// the run-time replan our own build-time fd is present in
/// `/proc/locks` for the very LLCs we already reserved. If
/// `holder_count` counted it, the Spread policy — which prefers the
/// LEAST-held LLCs — would flee our own reservation onto a different
/// LLC, the exact opposite of the truthful-holder-count fix. This
/// test holds `LOCK_SH` on LLC 0 from the test PROCESS and on LLC 1
/// from a `flock(1)` CHILD, then asserts LLC 0's `holder_count` is 0
/// (self excluded) with self still in the `holders` vec, and LLC 1's
/// `holder_count` is 1 (the peer counts).
///
/// Uses `flock(1)` for the peer so the holder is a genuinely
/// different pid; absent util-linux the test skips rather than fails
/// (same convention as `acquire_llc_plan_consolidates_on_peer_held_llc`).
#[test]
fn discover_excludes_self_pid_from_holder_count() {
    let _llc_prefix = LlcLockPrefixGuard::new();
    // 2 LLCs on the same node; both CPUs in `allowed` so neither LLC
    // is skipped by discover's allowed-overlap filter.
    let topo = HostTopology::new_for_tests(&[(vec![0], 0), (vec![1], 0)]);
    let allowed: std::collections::BTreeSet<usize> = [0usize, 1].into_iter().collect();

    // Self-hold SH on LLC 0 from this process/thread.
    let self_lock_path = llc_lock_path(0);
    crate::flock::materialize(&self_lock_path).expect("materialize self lockfile");
    let _self_hold = try_flock(&self_lock_path, FlockMode::Shared)
        .expect("open self lockfile")
        .expect("self SH on a fresh LLC 0 lockfile must succeed");

    // Peer child holds SH on LLC 1 via flock(1). Materialize first so
    // the child opens the same inode the parent's discover stats.
    let peer_lock_path = llc_lock_path(1);
    crate::flock::materialize(&peer_lock_path).expect("materialize peer lockfile");
    let child = std::process::Command::new("flock")
        .args(["-s", "-n", &peer_lock_path, "sleep", "2"])
        .spawn();
    let mut child = match child {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            eprintln!(
                "discover_excludes_self_pid_from_holder_count: \
                 flock(1) not available, skipping ({e})"
            );
            return;
        }
        Err(e) => panic!("spawn flock(1): {e}"),
    };
    std::thread::sleep(std::time::Duration::from_millis(50));

    let mountinfo = crate::flock::read_mountinfo().expect("read mountinfo");
    let snapshots =
        discover_llc_snapshots(&topo, &allowed, &mountinfo).expect("discover must succeed");

    let llc0 = snapshots
        .iter()
        .find(|s| s.llc_idx == 0)
        .expect("LLC 0 snapshot present");
    let llc1 = snapshots
        .iter()
        .find(|s| s.llc_idx == 1)
        .expect("LLC 1 snapshot present");

    let self_pid = std::process::id();
    assert!(
        llc0.holders.iter().any(|h| h.pid == self_pid),
        "self must remain in the diagnostic holders vec for LLC 0; \
         holders={:?}",
        llc0.holders,
    );
    assert_eq!(
        llc0.holder_count, 0,
        "self-held SH must be EXCLUDED from LLC 0's holder_count (got \
         {}, holders={:?})",
        llc0.holder_count, llc0.holders,
    );
    assert_eq!(
        llc1.holder_count, 1,
        "a peer process's SH must COUNT toward LLC 1's holder_count \
         (got {}, holders={:?})",
        llc1.holder_count, llc1.holders,
    );
    assert!(
        llc1.holders.iter().all(|h| h.pid != self_pid),
        "self never locked LLC 1, so must not appear there; holders={:?}",
        llc1.holders,
    );

    // Child exits after sleep 2; reap it so we don't leave zombies.
    let _ = child.wait();
}

/// The plan `acquire_llc_plan` returns carries its `LOCK_SH` flock
/// fds LIVE: held through the plan's lifetime, visible to a peer's
/// `/proc/locks` read, and released on drop. This is the precondition
/// the no-perf `build()` fix relies on — `build()` now forwards the
/// plan's `locks` onto `KtstrVm::no_perf_plan` UNSTRIPPED (the
/// historical `drop(std::mem::take(&mut plan.locks))` is gone), so a
/// concurrent peer's DISCOVER observes a truthful holder count from
/// build through run. With the fds stripped (the old behavior) the
/// Spread policy's holder-count feedback was dead.
///
/// Asserts: (1) `plan.locks` is non-empty; (2) our pid is a holder of
/// the locked LLC's lockfile while the plan lives; (3) after
/// `drop(plan)` our pid is gone from that lockfile's holders.
#[test]
fn acquire_llc_plan_holds_locks_live_until_drop() {
    let _llc_prefix = LlcLockPrefixGuard::new();
    let _allowed = AllowedCpusGuard::new(vec![0, 1]);
    let topo = HostTopology::new_for_tests(&[(vec![0], 0), (vec![1], 0)]);
    let test_topo = crate::topology::TestTopology::synthetic(2, 1);
    let cap = CpuCap::new(1).expect("cap=1 valid");

    let plan = acquire_llc_plan(
        &topo,
        &test_topo,
        Some(cap),
        PlacementPolicy::spread_for_process(),
    )
    .expect("acquire_llc_plan must succeed on a fresh two-LLC host");
    assert!(
        !plan.locks.is_empty(),
        "the plan must carry its live LOCK_SH fds (the build-time fd \
         strip is gone); got empty locks",
    );

    let locked = *plan
        .locked_llcs
        .first()
        .expect("a cap=1 plan locks exactly one LLC");
    let lock_path = std::path::PathBuf::from(llc_lock_path(locked));
    let self_pid = std::process::id();

    let mountinfo = crate::flock::read_mountinfo().expect("read mountinfo while plan alive");
    let held = crate::flock::read_holders_with_mountinfo(&lock_path, &mountinfo)
        .expect("read holders while plan alive");
    assert!(
        held.iter().any(|h| h.pid == self_pid),
        "our LOCK_SH must be visible in /proc/locks while the plan \
         lives; holders={held:?}",
    );

    drop(plan);

    let mountinfo = crate::flock::read_mountinfo().expect("read mountinfo after drop");
    let after = crate::flock::read_holders_with_mountinfo(&lock_path, &mountinfo)
        .expect("read holders after drop");
    assert!(
        after.iter().all(|h| h.pid != self_pid),
        "dropping the plan must release our LOCK_SH; holders still \
         list us: {after:?}",
    );
}

/// The build-time no-perf fds live in a `Mutex<Vec<OwnedFd>>` on
/// `KtstrVm` (not on the plan) so `acquire_run_locks`, which only has
/// `&self`, can release them the moment its replan adopts fresh locks.
/// This test exercises exactly that release path: park a held
/// `LOCK_SH` fd in such a slot, confirm our pid is a holder, then run
/// the `drop(std::mem::take(&mut *slot.lock().unwrap()))` release
/// through a SHARED (`&`) borrow of the slot — the same expression the
/// no-perf replan arm runs — and confirm our pid drops out of
/// `/proc/locks`. Without the interior-mutable slot the fd could only
/// release at `KtstrVm` drop, leaving a phantom hold on any LLC the
/// replan abandoned for the whole run.
#[test]
fn parked_build_locks_release_through_shared_borrow() {
    let _llc_prefix = LlcLockPrefixGuard::new();
    let lock_path_s = llc_lock_path(0);
    let lock_path = std::path::PathBuf::from(&lock_path_s);
    crate::flock::materialize(&lock_path_s).expect("materialize lockfile");
    let fd = try_flock(&lock_path_s, FlockMode::Shared)
        .expect("open lockfile")
        .expect("SH on a fresh lockfile must succeed");

    // The slot as it lives on `KtstrVm`. Bind by shared ref to prove
    // the release needs only `&self`, not `&mut`.
    let slot: std::sync::Mutex<Vec<std::os::fd::OwnedFd>> = std::sync::Mutex::new(vec![fd]);
    let slot_ref: &std::sync::Mutex<Vec<std::os::fd::OwnedFd>> = &slot;

    let self_pid = std::process::id();
    let mountinfo = crate::flock::read_mountinfo().expect("read mountinfo while parked");
    let held = crate::flock::read_holders_with_mountinfo(&lock_path, &mountinfo)
        .expect("read holders while parked");
    assert!(
        held.iter().any(|h| h.pid == self_pid),
        "the parked fd must hold LOCK_SH; holders={held:?}",
    );

    // The exact release expression from `acquire_run_locks`' no-perf arm.
    drop(std::mem::take(&mut *slot_ref.lock().unwrap()));

    let mountinfo = crate::flock::read_mountinfo().expect("read mountinfo after release");
    let after = crate::flock::read_holders_with_mountinfo(&lock_path, &mountinfo)
        .expect("read holders after release");
    assert!(
        after.iter().all(|h| h.pid != self_pid),
        "take-and-drop through the shared borrow must release our \
         LOCK_SH; holders still list us: {after:?}",
    );
    assert!(
        slot_ref.lock().unwrap().is_empty(),
        "the slot must be drained after the take",
    );
}

/// `ACQUIRE_MAX_TOCTOU_RETRIES` pins the retry budget at 3 —
/// one DISCOVER + up to three retry DISCOVERs (four total
/// attempts), each separated by an ascending micro-sleep
/// (10ms, 50ms, 200ms — see [`TOCTOU_RETRY_DELAYS`]) so a
/// racing peer has time to drop its fds before the next
/// snapshot. Regression guard against a future "just retry
/// harder" tweak that would amplify livelock cost without
/// adding coordination signal.
#[test]
fn acquire_max_toctou_retries_pinned() {
    assert_eq!(
        ACQUIRE_MAX_TOCTOU_RETRIES, 3,
        "retry budget must be 3 — micro-sleeps absorb mid-sized races",
    );
    assert_eq!(
        TOCTOU_RETRY_DELAYS.len(),
        ACQUIRE_MAX_TOCTOU_RETRIES as usize,
        "one sleep per retry — TOCTOU_RETRY_DELAYS length must \
         match ACQUIRE_MAX_TOCTOU_RETRIES exactly",
    );
}

/// TOCTOU retry SUCCESS path via the acquire-fn seam: attempt 0
/// returns `Ok(None)` (simulating a peer holding EX during the
/// first ACQUIRE), attempt 1 returns `Ok(Some(Vec::new()))`
/// (peer released, shared acquire succeeds). The outer
/// `acquire_llc_plan_with_acquire_fn` must re-run DISCOVER +
/// PLAN and retry — not propagate the first `None` upward.
///
/// Uses two real LLC groups with empty CPU lists so
/// `discover_llc_snapshots` succeeds without touching any real
/// `/tmp` lockfile (the seam consumes the snapshots instead of
/// handing off to the real flock code). LLC indices 93500/93501
/// are in the reserved 93000-99999 test range per the module's
/// SYNTHETIC-TOPOLOGY OFFSET CONVENTION.
#[test]
fn acquire_llc_plan_retry_succeeds_on_attempt_one() {
    let _llc_prefix = LlcLockPrefixGuard::new();
    let _allowed = AllowedCpusGuard::new(vec![93500, 93501]);
    let topo = synth_host_topo(&[(vec![93500], 0), (vec![93501], 0)]);

    let test_topo = crate::topology::TestTopology::synthetic(2, 1);
    let counter = std::cell::Cell::new(0u32);
    let plan = acquire_llc_plan_with_acquire_fn(
        &topo,
        &test_topo,
        None,
        PlacementPolicy::Consolidate,
        |_selected, _snapshots| {
            let n = counter.get();
            counter.set(n + 1);
            if n == 0 {
                // Attempt 0: simulate peer winning EX race.
                Ok(None)
            } else {
                // Attempt 1: peer released, acquire succeeds
                // with an empty fd set (production would have
                // actual OwnedFd values; the LlcPlan RAII
                // contract is exercised elsewhere).
                Ok(Some(Vec::new()))
            }
        },
    )
    .expect("retry on attempt 1 must succeed");
    // Attempt 1 produced locks (empty vec is fine — the plan
    // constructor accepts any Vec<OwnedFd>).
    assert_eq!(counter.get(), 2, "acquire_fn called exactly twice");
    // 30% of 2 allowed CPUs = ceil(0.6) = 1 CPU → pick 1 LLC
    // (seed-node first: LLC 0). `selected` holds only LLC 0;
    // the second LLC stays unlocked.
    assert_eq!(plan.locked_llcs, vec![0]);
}

/// TOCTOU retry EXHAUSTED path via the acquire-fn seam: every
/// attempt returns `Ok(None)`. After
/// `ACQUIRE_MAX_TOCTOU_RETRIES + 1` attempts, the outer loop
/// bails with a `ResourceContention` whose message names the
/// retry count.
///
/// Pins: (a) the retry budget is respected — the acquire
/// closure is called exactly `ACQUIRE_MAX_TOCTOU_RETRIES + 1`
/// times before the error is returned; (b) the error surfaces
/// as `ResourceContention` for nextest-retry routing; (c) the
/// holder diagnostic block runs (the final DISCOVER read).
#[test]
fn acquire_llc_plan_retry_exhausted_bails_with_resource_contention() {
    let _llc_prefix = LlcLockPrefixGuard::new();
    let _allowed = AllowedCpusGuard::new(vec![93600]);
    let topo = synth_host_topo(&[(vec![93600], 0)]);
    let test_topo = crate::topology::TestTopology::synthetic(1, 1);

    let counter = std::cell::Cell::new(0u32);
    let err = acquire_llc_plan_with_acquire_fn(
        &topo,
        &test_topo,
        None,
        PlacementPolicy::Consolidate,
        |_selected, _snapshots| {
            counter.set(counter.get() + 1);
            Ok(None)
        },
    )
    .expect_err("every attempt returns None — must bail after retries");

    // The retry budget consumes exactly ACQUIRE_MAX_TOCTOU_RETRIES
    // + 1 acquire-fn calls. Attempt index 0 is the first
    // acquire; attempt reaches MAX before incrementing, so the
    // failure occurs on call MAX+1.
    assert_eq!(
        counter.get(),
        ACQUIRE_MAX_TOCTOU_RETRIES + 1,
        "acquire_fn called exactly ACQUIRE_MAX_TOCTOU_RETRIES + 1 times",
    );

    assert!(
        err.downcast_ref::<ResourceContention>().is_some(),
        "must downcast to ResourceContention for retry routing: {err:#}",
    );
    let msg = format!("{err:#}");
    assert!(
        msg.contains("attempts"),
        "message must name the attempt count: {msg}",
    );
}

/// `plan_from_snapshots` MUST-CONSOLIDATE invariant: on a
/// single-node host where every fresh LLC is ascending, the
/// single peer-held LLC at index 3 MUST be selected over any
/// lower-index fresh LLC when target=1. A future refactor that
/// accidentally flipped the partition order (fresh-first) or
/// dropped the `holder_count > 0` filter would pick LLC 0
/// instead and fail this test.
///
/// Complements `plan_from_snapshots_prefers_higher_holder_count`
/// (same-node, two LLCs) by proving the peer-held LLC wins
/// even when it sits at the TAIL of the ascending fresh order,
/// not just adjacent — the `holder_count > 0` partition MUST
/// override the fresh-LLC ordering.
#[test]
fn plan_from_snapshots_consolidation_overrides_fresh_ordering() {
    let topo = synth_host_topo(&[(vec![0], 0), (vec![1], 0), (vec![2], 0), (vec![3], 0)]);
    let snapshots: Vec<LlcSnapshot> = (0..4)
        .map(|idx| LlcSnapshot {
            llc_idx: idx,
            lockfile_path: std::path::PathBuf::from(format!("/tmp/ktstr-llc-{idx}.lock")),
            holders: Vec::new(),
            holder_count: if idx == 3 { 5 } else { 0 },
        })
        .collect();
    let allowed: std::collections::BTreeSet<usize> = (0..4).collect();
    let selected = plan_from_snapshots(
        &snapshots,
        1,
        &topo,
        &allowed,
        |_, _| 10,
        PlacementPolicy::Consolidate,
    );
    assert_eq!(
        selected,
        vec![3],
        "target=1 with peer-held LLC 3 must pick LLC 3, not the \
         lowest-index fresh LLC 0 — consolidation overrides fresh",
    );
}

/// `plan_from_snapshots` NUMA-locality invariant: a single-node
/// fit (target ≤ seed-node capacity) must NEVER spill. 4 LLCs
/// split 2+2 across nodes 0/1, all fresh, target=2 → selected
/// must be both LLCs on the seed node. A future refactor that
/// accidentally spanned both nodes (e.g. by iterating every
/// node's LLCs before checking selected.len()) would fail here.
///
/// Walk seed node first, exhaust it
/// before spilling to nearest-by-distance nodes. This test
/// pins that the seed-node-fits-fully short-circuit works.
#[test]
fn plan_from_snapshots_single_node_fit_no_spill() {
    // LLCs 0,1 on node 0; LLCs 2,3 on node 1. CPUs disjoint so
    // synth_host_topo populates cpu_to_node cleanly.
    let topo = synth_host_topo(&[(vec![0], 0), (vec![1], 0), (vec![2], 1), (vec![3], 1)]);
    // All fresh so neither node has a consolidation signal —
    // isolates the NUMA-locality bias.
    let snapshots: Vec<LlcSnapshot> = (0..4)
        .map(|idx| LlcSnapshot {
            llc_idx: idx,
            lockfile_path: std::path::PathBuf::from(format!("/tmp/ktstr-llc-{idx}.lock")),
            holders: Vec::new(),
            holder_count: 0,
        })
        .collect();
    // Canonical distance: same-node 10, cross-node 20.
    let allowed: std::collections::BTreeSet<usize> = (0..4).collect();
    let selected = plan_from_snapshots(
        &snapshots,
        2,
        &topo,
        &allowed,
        |from, to| if from == to { 10 } else { 20 },
        PlacementPolicy::Consolidate,
    );
    assert_eq!(
        selected,
        vec![0, 1],
        "target=2 must stay on seed node 0 (LLCs 0,1); seed-node \
         capacity (2) covers the request, no spill to node 1 allowed",
    );
}

/// `plan_from_snapshots` tie-break invariant: when every
/// consolidation score is identical (all holder_count=5),
/// selection tiebreaks on `llc_idx ASC`. target=2 on 4 equal
/// LLCs → selected == [0, 1]. A future refactor that made the
/// consolidation sort unstable, or that used `sort_by_key`
/// without the secondary ASC tiebreak, would pick a non-
/// deterministic pair and fail this test.
///
/// The `holder_count DESC, llc_idx ASC` composite key — the
/// second key is mandatory for cross-run determinism.
#[test]
fn plan_from_snapshots_equal_scores_tiebreak_ascending() {
    let topo = synth_host_topo(&[(vec![0], 0), (vec![1], 0), (vec![2], 0), (vec![3], 0)]);
    let snapshots: Vec<LlcSnapshot> = (0..4)
        .map(|idx| LlcSnapshot {
            llc_idx: idx,
            lockfile_path: std::path::PathBuf::from(format!("/tmp/ktstr-llc-{idx}.lock")),
            holders: Vec::new(),
            holder_count: 5,
        })
        .collect();
    let allowed: std::collections::BTreeSet<usize> = (0..4).collect();
    let selected = plan_from_snapshots(
        &snapshots,
        2,
        &topo,
        &allowed,
        |_, _| 10,
        PlacementPolicy::Consolidate,
    );
    assert_eq!(
        selected,
        vec![0, 1],
        "equal consolidation scores must tiebreak on llc_idx ASC \
         — selected={selected:?}",
    );
}

/// `default_cpu_budget` math: 30% rounded UP with min-1 floor.
/// Covers the small-host edge (1 CPU → 1 CPU budget), the
/// rounding boundary (3 CPUs → ceil(0.9) = 1 CPU), the
/// non-trivial case (10 CPUs → 3 CPUs), and the large case
/// (100 CPUs → 30 CPUs). Zero-input is pinned at min-1 for
/// defense in depth even though production callers bail
/// upstream on empty allowed sets.
#[test]
fn default_cpu_budget_30_percent_rounded_up_min_one() {
    assert_eq!(default_cpu_budget(0), 1, "min-1 floor");
    assert_eq!(default_cpu_budget(1), 1, "ceil(0.3) = 1");
    assert_eq!(default_cpu_budget(3), 1, "ceil(0.9) = 1");
    assert_eq!(default_cpu_budget(4), 2, "ceil(1.2) = 2");
    assert_eq!(default_cpu_budget(10), 3, "ceil(3.0) = 3");
    assert_eq!(default_cpu_budget(100), 30, "exact 30%");
}

/// `no_perf_cpu_budget` sizes a no-perf VM to its EXACT vCPU count
/// (clamped to the allowed cpuset, min-1) — the topology's real need,
/// with the 30% `default_cpu_budget` acting as NEITHER a floor nor a
/// ceiling. The regression this pins: a 2-vCPU interactive shell on a
/// ~192-CPU host must ask ~3 CPUs (vCPUs + 1 service), never the 30% (58) it did when 30%
/// was a `.max()` floor — that over-reservation walked ~30% of the
/// host's LLCs and exhausted `acquire_llc_plan` under peer `LOCK_EX`
/// contention (`could not reserve 58 CPU(s)`).
#[test]
fn no_perf_cpu_budget_sizes_to_vcpus_not_30_percent() {
    // The exact CI-failure shape: a 2-vCPU guest on a 192-CPU box.
    // Old (buggy) `max(30%, min(vcpus, allowed))` = max(58, 2) = 58;
    // the clamp must yield vCPUs + 1 (the service-thread CPU).
    assert_eq!(
        no_perf_cpu_budget(192, 2),
        3,
        "2-vCPU VM on a 192-CPU host asks 3 CPUs (2 vCPUs + 1 service), never the 30% (58) floor",
    );
    // A small VM never inflates to 30% on a large host.
    assert_eq!(
        no_perf_cpu_budget(100, 4),
        5,
        "vcpus + 1 service, no 30%-of-100 = 30 floor"
    );
    // A WIDE VM still gets its full vCPU count plus the service CPU
    // (test-validity: budget >= vcpus so the guest scheduler isn't
    // confounded by host oversubscription; +1 keeps the host-side
    // sensing threads off the vCPUs' CPUs).
    assert_eq!(
        no_perf_cpu_budget(100, 50),
        51,
        "wide VM keeps its vCPU count + 1 service CPU"
    );
    // `vcpus > allowed` clamps to the process cpuset (the
    // overcommit-warning case). min-1 floor on degenerate inputs.
    assert_eq!(
        no_perf_cpu_budget(8, 32),
        8,
        "clamped to the allowed cpuset"
    );
    assert_eq!(no_perf_cpu_budget(0, 4), 1, "min-1 floor (empty allowed)");
    assert_eq!(no_perf_cpu_budget(4, 0), 1, "min-1 floor (zero vCPUs)");
}

/// `overcommit_warning`: `None` when the budget covers the vCPUs; `Some`
/// with the right severity (explicit opt-in vs silent auto-collapse) and
/// the tight-watchdog caveat when the budget is below the vCPU count.
#[test]
fn overcommit_warning_severity_and_polarity() {
    // Budget >= vcpus: not oversubscribed -> no warning, either severity.
    assert_eq!(overcommit_warning(32, 32, false, None), None);
    assert_eq!(overcommit_warning(40, 32, true, None), None);

    // Explicit opt-in (cpu_budget / --cpu-cap below vcpus): an opt-in note
    // that points at the overcommit-invariant rate.
    let m = overcommit_warning(4, 32, true, None).expect("4 < 32 => Some");
    assert!(
        m.contains("opt-in"),
        "explicit case must read as opt-in: {m}"
    );
    assert!(
        m.contains("worst_iterations_per_cpu_sec"),
        "must point at the per-cgroup overcommit-invariant rate \
         (worst_iterations_per_cpu_sec, matching the stats.rs compare hint); \
         the bare iterations_per_cpu_sec is the pooled cohort rate: {m}",
    );
    assert!(
        !m.contains("NOTHING opted into"),
        "explicit case must not use the silent-case wording: {m}",
    );

    // Auto-collapse (process cpuset smaller than the guest): the loud,
    // nobody-opted-in case.
    let m = overcommit_warning(4, 32, false, None).expect("4 < 32 => Some");
    assert!(
        m.contains("WARNING"),
        "auto case must be a louder WARNING: {m}"
    );
    assert!(
        m.contains("NOTHING opted into"),
        "auto case must name the silent condition: {m}",
    );

    // A tight watchdog folds in the false-eject caveat; a roomy one omits it.
    let tight = overcommit_warning(4, 32, true, Some(5)).unwrap();
    assert!(tight.contains("watchdog"), "tight watchdog caveat: {tight}");
    let roomy = overcommit_warning(4, 32, true, Some(30)).unwrap();
    assert!(
        !roomy.contains("watchdog"),
        "roomy watchdog must not add the caveat: {roomy}",
    );
}

/// `acquire_llc_plan` bails with a diagnostic when the allowed
/// CPU set has no overlap with ANY host LLC — a misconfigured
/// host where sysfs and sched_getaffinity disagree. Pins the
/// plan_from_snapshots-returns-empty → bail path so a future
/// refactor that silently produces an empty plan surfaces as a
/// test failure rather than an "no-op" VM boot.
#[test]
fn acquire_llc_plan_bails_when_no_llc_overlaps_allowed() {
    let _llc_prefix = LlcLockPrefixGuard::new();
    // Allowed CPUs {100, 101} don't overlap ANY of the host's
    // LLCs (CPUs 0, 1). plan_from_snapshots returns empty →
    // acquire_llc_plan bails with the no-overlap diagnostic.
    let _allowed = AllowedCpusGuard::new(vec![100, 101]);
    let topo = HostTopology::new_for_tests(&[(vec![0], 0), (vec![1], 0)]);
    let test_topo = crate::topology::TestTopology::synthetic(4, 1);
    let err = acquire_llc_plan(&topo, &test_topo, None, PlacementPolicy::Consolidate)
        .expect_err("no LLC overlap must bail, not silently run");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("no host LLC overlaps"),
        "err must name the no-overlap condition: {msg}"
    );
}

/// Allowed-cpu filter invariant — LLCs whose CPUs are entirely
/// outside the allowed set MUST NOT appear in `selected`, even
/// when their consolidation score would otherwise promote them.
///
/// Four LLCs, two CPUs each. Allowed set = {0, 1, 4, 5} —
/// contains every CPU of LLCs 0 and 2, NONE of LLCs 1 or 3.
/// target_cpus=3 → planner picks LLC 0 (2 allowed CPUs,
/// accumulated 2 < 3 keeps walking) then LLC 2 (1 more CPU is
/// enough to cover the budget once materialization
/// partial-takes; the plan_from_snapshots walk itself stops
/// once accumulated ≥ target, which here fires at accumulated
/// == 4 ≥ 3). `selected` is [0, 2]; LLCs 1 and 3 must stay
/// out of the list.
///
/// Regresses any refactor that drops the eligibility filter —
/// e.g. a cleaner that collapses the `filter(eligible)` pass
/// into the sort closure would produce a plan containing an
/// LLC with zero schedulable CPUs, which sched_setaffinity on
/// the resulting mask would reject.
#[test]
fn plan_from_snapshots_filters_llcs_outside_allowed_set() {
    let topo = synth_host_topo(&[
        (vec![0, 1], 0),
        (vec![2, 3], 0),
        (vec![4, 5], 0),
        (vec![6, 7], 0),
    ]);
    let snapshots: Vec<LlcSnapshot> = (0..4)
        .map(|idx| LlcSnapshot {
            llc_idx: idx,
            lockfile_path: std::path::PathBuf::from(format!("/tmp/ktstr-llc-{idx}.lock")),
            holders: Vec::new(),
            holder_count: 0,
        })
        .collect();
    let allowed: std::collections::BTreeSet<usize> = [0, 1, 4, 5].into_iter().collect();
    let selected = plan_from_snapshots(
        &snapshots,
        3,
        &topo,
        &allowed,
        |_, _| 10,
        PlacementPolicy::Consolidate,
    );
    assert_eq!(
        selected,
        vec![0, 2],
        "planner must skip LLCs 1 and 3 (no allowed-CPU overlap) \
         and pick LLCs 0 and 2 whose CPUs are fully in allowed; \
         got {selected:?}"
    );
}

/// Partial-take on the last selected LLC — when the budget
/// falls mid-LLC, `plan.cpus` contains only the budget-needed
/// prefix of that LLC's allowed CPUs, not the whole LLC. Two
/// 4-CPU LLCs, cpu_cap = 5 → LLC 0 contributes 4 CPUs, LLC 1
/// contributes 1 CPU, `plan.cpus.len() == 5`, both LLCs are
/// flocked. Regresses any refactor that reverts to the
/// round-up-whole-LLC policy (which would produce 8 CPUs,
/// over-reserving).
#[test]
fn acquire_llc_plan_partial_take_last_llc_matches_exact_budget() {
    let _llc_prefix = LlcLockPrefixGuard::new();
    let _allowed = AllowedCpusGuard::new(vec![0, 1, 2, 3, 4, 5, 6, 7]);
    let topo = HostTopology::new_for_tests(&[(vec![0, 1, 2, 3], 0), (vec![4, 5, 6, 7], 0)]);
    let test_topo = crate::topology::TestTopology::synthetic(4, 1);
    let cap = CpuCap::new(5).expect("cap=5 valid");
    let plan = acquire_llc_plan(&topo, &test_topo, Some(cap), PlacementPolicy::Consolidate)
        .expect("clean pool must allow SH on both LLCs");

    assert_eq!(
        plan.locked_llcs,
        vec![0, 1],
        "budget of 5 CPUs crosses LLC boundary — both must be flocked"
    );
    assert_eq!(
        plan.cpus.len(),
        5,
        "plan.cpus is EXACTLY the budget, not rounded up: {:?}",
        plan.cpus,
    );
    // Partial-take is deterministic: first LLC fully, then the
    // ordered prefix of the second.
    assert_eq!(plan.cpus, vec![0, 1, 2, 3, 4]);
}

/// Partial-LLC allowed overlap — an LLC that contains SOME
/// allowed CPUs is still selectable, and its contribution to
/// the CPU budget is the size of the intersection, not the
/// full LLC. Two LLCs with 2 CPUs each; allowed = {0, 2} (one
/// CPU from each LLC). target_cpus=2 → both LLCs must be
/// selected (each contributes 1 allowed CPU, total 2 meets the
/// budget).
#[test]
fn plan_from_snapshots_partial_llc_overlap_counted_correctly() {
    let topo = synth_host_topo(&[(vec![0, 1], 0), (vec![2, 3], 0)]);
    let snapshots: Vec<LlcSnapshot> = (0..2)
        .map(|idx| LlcSnapshot {
            llc_idx: idx,
            lockfile_path: std::path::PathBuf::from(format!("/tmp/ktstr-llc-{idx}.lock")),
            holders: Vec::new(),
            holder_count: 0,
        })
        .collect();
    let allowed: std::collections::BTreeSet<usize> = [0, 2].into_iter().collect();
    let selected = plan_from_snapshots(
        &snapshots,
        2,
        &topo,
        &allowed,
        |_, _| 10,
        PlacementPolicy::Consolidate,
    );
    assert_eq!(
        selected,
        vec![0, 1],
        "target_cpus=2 with 1 allowed CPU per LLC must pick \
         BOTH LLCs — each contributes 1, total 2 meets budget"
    );
}

/// Full `LlcPlan.mems` invariant (I1) — on a cross-node spill,
/// `mems` MUST equal the union of NUMA nodes hosting every
/// selected LLC. 4 LLCs split 2+2 across nodes 0/1, cap=3
/// forces exactly one LLC from node 1 to spill after node 0
/// exhausts. Assert `locked_llcs.len() == 3` AND
/// `mems == {0, 1}`.
///
/// Without this guard, a broken mems computation could produce
/// an empty set (cgroup cpuset.mems write rejects → SIGKILL on
/// mem alloc), OR the wrong nodes (forcing cross-socket
/// allocation that defeats the LLC reservation).
///
/// Uses a per-test lockfile prefix via [`LlcLockPrefixGuard`] so
/// the topology can use small indices (0..4) instead of padding
/// to 94004 entries to avoid colliding with production LLC
/// lockfile paths.
#[test]
fn acquire_llc_plan_cross_node_spill_mems_union() {
    let _llc_prefix = LlcLockPrefixGuard::new();
    let _allowed = AllowedCpusGuard::new(vec![0, 1, 2, 3]);
    // LLC 0,1 on node 0 (CPUs 0,1); LLC 2,3 on node 1 (CPUs 2,3).
    let topo =
        HostTopology::new_for_tests(&[(vec![0], 0), (vec![1], 0), (vec![2], 1), (vec![3], 1)]);

    let test_topo = crate::topology::TestTopology::synthetic(4, 2);
    // Each LLC has 1 CPU, so cap=3 CPUs → exactly 3 LLCs.
    let cap = CpuCap::new(3).expect("cap=3 valid");
    let plan = acquire_llc_plan(&topo, &test_topo, Some(cap), PlacementPolicy::Consolidate)
        .expect("clean pool must allow 3-CPU acquisition");

    assert_eq!(
        plan.locked_llcs.len(),
        3,
        "cap=3 CPUs with 1-CPU LLCs must reserve exactly 3 LLCs, got {:?}",
        plan.locked_llcs,
    );
    assert_eq!(
        plan.mems.len(),
        2,
        "3 LLCs split across 2 nodes → mems must span BOTH nodes; \
         got {:?} (locked_llcs={:?})",
        plan.mems,
        plan.locked_llcs,
    );
    assert!(
        plan.mems.contains(&0) && plan.mems.contains(&1),
        "mems must contain BOTH node 0 and node 1 after cross-node \
         spill; got {:?}",
        plan.mems,
    );
}

// ─── KTSTR_CARGO_TEST_MODE bypass ────────────────────────────
//
// Tests at indices 95000-95999 cover the cargo-test-mode flock
// bypass. Use the `crate::test_support::test_helpers::lock_env`
// pattern to serialise env mutation across tests; otherwise a
// concurrent test could observe a transiently-set
// `KTSTR_CARGO_TEST_MODE` and short-circuit its own assertions.

/// `acquire_resource_locks` returns `Acquired { locks: vec![] }`
/// in cargo-test mode regardless of the requested `LlcLockMode`
/// or whether a peer holds the same lockfile. Pins the bypass
/// contract: bare `cargo test` doesn't share the cross-process
/// LLC reservation contract that nextest / `cargo ktstr test`
/// peers rely on.
#[test]
fn acquire_resource_locks_cargo_test_mode_bypasses_flock() {
    use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
    let _lock = lock_env();
    let _env = EnvVarGuard::set(crate::KTSTR_CARGO_TEST_MODE_ENV, "1");
    let plan = PinningPlan {
        assignments: vec![(0, 95100)],
        service_cpu: None,
        llc_indices: vec![95100],
        locks: Vec::new(),
    };
    let outcome = acquire_resource_locks(&plan, &[95100usize], LlcLockMode::Exclusive).unwrap();
    let (llc_offset, locks) = unwrap_acquired(outcome, Some("in cargo-test mode"));
    assert_eq!(llc_offset, 95100);
    assert!(
        locks.is_empty(),
        "cargo-test-mode bypass must NOT take any flocks; \
         got {} held fds",
        locks.len(),
    );
}

/// Empty `KTSTR_CARGO_TEST_MODE` does NOT activate the bypass.
/// The standard `acquire_resource_locks` path runs and returns an
/// `Acquired` with the actual fd vector. Mirrors the empty-string
/// rejection on `cargo_test_mode_active` so a stray `--env`
/// pass-through can't silently degrade the locking contract.
#[test]
fn acquire_resource_locks_cargo_test_mode_empty_string_inert() {
    use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
    let _lock = lock_env();
    let _env = EnvVarGuard::set(crate::KTSTR_CARGO_TEST_MODE_ENV, "");
    let _llc_prefix = LlcLockPrefixGuard::new();
    let plan = PinningPlan {
        assignments: vec![(0, 95200)],
        service_cpu: None,
        llc_indices: vec![95200],
        locks: Vec::new(),
    };
    let outcome = acquire_resource_locks(&plan, &[95200usize], LlcLockMode::Exclusive).unwrap();
    let (_, locks) = unwrap_acquired(outcome, Some("with empty-string bypass inert"));
    assert_eq!(
        locks.len(),
        1,
        "empty-string cargo-test-mode is inert — expected the \
         standard `Exclusive` path to take exactly one LLC fd, \
         got {}",
        locks.len(),
    );
}

// ---------------------------------------------------------------
// Spread placement — no-perf VM plans fan out instead of stacking
// ---------------------------------------------------------------

/// Snapshot fixture for the Spread tests: `n` single-CPU LLCs on
/// node 0 with the given per-LLC holder counts.
fn spread_snapshots(holder_counts: &[usize]) -> Vec<LlcSnapshot> {
    holder_counts
        .iter()
        .enumerate()
        .map(|(idx, &holder_count)| LlcSnapshot {
            llc_idx: idx,
            lockfile_path: std::path::PathBuf::from(format!("/tmp/ktstr-llc-{idx}.lock")),
            holders: Vec::new(),
            holder_count,
        })
        .collect()
}

/// Spread inverts the consolidation preference: with peers holding
/// LLCs 0-1, a Spread plan lands on the FRESH LLCs 2-3 (a
/// Consolidate plan picks the held ones — pinned by
/// `plan_from_snapshots_returns_ascending_indices`). This is the
/// heart of the scx-sweep fix: a VM cell must move AWAY from the
/// load, not toward it.
#[test]
fn plan_from_snapshots_spread_prefers_least_held_llcs() {
    let topo = synth_host_topo(&[(vec![0], 0), (vec![1], 0), (vec![2], 0), (vec![3], 0)]);
    let snapshots = spread_snapshots(&[5, 5, 0, 0]);
    let allowed: std::collections::BTreeSet<usize> = (0..4).collect();
    let selected = plan_from_snapshots(
        &snapshots,
        2,
        &topo,
        &allowed,
        |_, _| 10,
        PlacementPolicy::Spread { rotation: 0 },
    );
    assert_eq!(
        selected,
        vec![2, 3],
        "spread must select the zero-holder LLCs, not the peer-held ones",
    );
}

/// The pid-derived rotation breaks the zero-knowledge symmetry:
/// concurrent planners that all snapshot ZERO holders (plans are
/// computed at build() while the LOCK_SH set defers to run(), so a
/// simultaneous fan-out sees no peers) select DIFFERENT LLC windows
/// instead of all stacking on the LLC-0 prefix. Also pins the
/// wrap-around (rotation near the end wraps to LLC 0) and the step-e
/// ascending re-sort under Spread.
#[test]
fn plan_from_snapshots_spread_rotation_fans_out_zero_holder_snapshots() {
    let topo = synth_host_topo(&[
        (vec![0], 0),
        (vec![1], 0),
        (vec![2], 0),
        (vec![3], 0),
        (vec![4], 0),
        (vec![5], 0),
    ]);
    let snapshots = spread_snapshots(&[0; 6]);
    let allowed: std::collections::BTreeSet<usize> = (0..6).collect();
    let window = |rotation: usize| {
        plan_from_snapshots(
            &snapshots,
            2,
            &topo,
            &allowed,
            |_, _| 10,
            PlacementPolicy::Spread { rotation },
        )
    };
    assert_eq!(window(0), vec![0, 1], "rotation 0 starts at LLC 0");
    assert_eq!(window(3), vec![3, 4], "rotation 3 starts at LLC 3");
    assert_eq!(
        window(5),
        vec![0, 5],
        "rotation 5 wraps (LLC 5 then LLC 0) and step e still \
         returns ascending acquire order",
    );
    assert_eq!(
        window(9),
        vec![3, 4],
        "rotation reduces modulo the eligible count (9 % 6 == 3)",
    );
}

/// Rotation positions are computed over the ELIGIBLE list, not raw
/// llc_idx values: with LLC 1 filtered out by the allowed cpuset,
/// rotation 1 must start at the second ELIGIBLE LLC (idx 2), and a
/// full-wrap rotation must cover exactly the eligible set.
#[test]
fn plan_from_snapshots_spread_rotation_uses_eligible_positions() {
    let topo = synth_host_topo(&[(vec![0], 0), (vec![1], 0), (vec![2], 0), (vec![3], 0)]);
    let snapshots = spread_snapshots(&[0; 4]);
    // LLC 1's only CPU is outside the allowed set — ineligible.
    let allowed: std::collections::BTreeSet<usize> = [0usize, 2, 3].into_iter().collect();
    let selected = plan_from_snapshots(
        &snapshots,
        1,
        &topo,
        &allowed,
        |_, _| 10,
        PlacementPolicy::Spread { rotation: 1 },
    );
    assert_eq!(
        selected,
        vec![2],
        "rotation 1 must land on the second eligible LLC (2), \
         not raw index 1 (ineligible)",
    );
}

/// Spread still honours NUMA seeding: the rotated first choice
/// anchors the seed node and the walk fills that node before
/// spilling, so a rotated plan is NUMA-coherent rather than a
/// round-robin scatter.
#[test]
fn plan_from_snapshots_spread_rotated_seed_fills_its_node_first() {
    // Two nodes, two single-CPU LLCs each.
    let topo = synth_host_topo(&[(vec![0], 0), (vec![1], 0), (vec![2], 1), (vec![3], 1)]);
    let snapshots = spread_snapshots(&[0; 4]);
    let allowed: std::collections::BTreeSet<usize> = (0..4).collect();
    let selected = plan_from_snapshots(
        &snapshots,
        2,
        &topo,
        &allowed,
        |from, to| if from == to { 10 } else { 20 },
        PlacementPolicy::Spread { rotation: 2 },
    );
    assert_eq!(
        selected,
        vec![2, 3],
        "rotation 2 seeds on LLC 2 (node 1) and fills node 1 \
         before any spill back to node 0",
    );
}

/// `spread_for_process` is stable within a process (pure pid hash):
/// the same process always plans the same windows, so its own
/// affinity masks and lock sets stay aligned across build()-time
/// planning and any diagnostics that re-derive the rotation.
#[test]
fn spread_for_process_rotation_is_stable() {
    assert_eq!(
        PlacementPolicy::spread_for_process(),
        PlacementPolicy::spread_for_process(),
    );
}
