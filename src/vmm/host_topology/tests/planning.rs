use super::super::protocol as admission_protocol;
use super::super::*;
use super::*;
use crate::vmm::topology::Topology;

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
    let err = acquire_llc_plan(
        &topo,
        &test_topo,
        Some(cap),
        PlacementPolicy::Consolidate,
        false,
    )
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
            holders: Vec::new(),
            holder_count: if idx >= 2 { 5 } else { 0 },
            exclusive_held: false,
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
            holders: Vec::new(),
            holder_count: 0,
            exclusive_held: false,
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

/// Saturation must preserve the LLC identities carried by the snapshots.
/// Sparse snapshots arise when the caller's allowed CPU set excludes whole
/// LLCs; treating snapshot positions as LLC indices would turn `[1, 3]` into
/// `[0, 1]` and reserve the wrong lockfiles.
#[test]
fn plan_from_snapshots_sparse_saturation_preserves_llc_indices() {
    let topo = synth_host_topo(&[(vec![0], 0), (vec![1], 0), (vec![2], 0), (vec![3], 0)]);
    let snapshots = [1usize, 3]
        .into_iter()
        .map(|llc_idx| LlcSnapshot {
            llc_idx,
            holders: Vec::new(),
            holder_count: 0,
            exclusive_held: false,
        })
        .collect::<Vec<_>>();
    let allowed = (0..4).collect::<std::collections::BTreeSet<_>>();

    let selected = plan_from_snapshots(
        &snapshots,
        2,
        &topo,
        &allowed,
        |_, _| 10,
        PlacementPolicy::Consolidate,
    );

    assert_eq!(selected, vec![1, 3]);
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
        holders: Vec::new(),
        holder_count: 0,
        exclusive_held: false,
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

#[test]
fn overlapping_llc_groups_materialize_a_distinct_cpu_budget() {
    let topo = HostTopology::new_for_tests_unchecked(&[(vec![0, 1], 0), (vec![1, 2], 0)]);
    let snapshots = (0..2)
        .map(|llc_idx| LlcSnapshot {
            llc_idx,
            holders: Vec::new(),
            holder_count: 0,
            exclusive_held: false,
        })
        .collect::<Vec<_>>();
    let allowed = [0usize, 1, 2]
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    let selected = plan_from_snapshots(
        &snapshots,
        3,
        &topo,
        &allowed,
        |from, to| if from == to { 10 } else { 20 },
        PlacementPolicy::Consolidate,
    );
    let states = std::collections::BTreeMap::new();
    let (cpus, _) = materialize_plan_cpus(
        &selected,
        &topo,
        &allowed,
        &states,
        3,
        PlacementPolicy::Consolidate,
    )
    .expect("the union of overlapping LLC groups carries three CPUs");
    assert_eq!(cpus, vec![0, 1, 2]);
    assert_eq!(
        cpus.iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        3,
        "the exact CPU claim must not shrink when converted to a set",
    );
}

#[test]
fn sparse_cpu_rotation_uses_eligible_ordinals_not_cpu_ids() {
    let topo = synth_host_topo(&[(vec![2, 8, 32], 0)]);
    let eligible = [2usize, 8, 32]
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    let (cpus, _) = materialize_plan_cpus(
        &[0],
        &topo,
        &eligible,
        &std::collections::BTreeMap::new(),
        3,
        PlacementPolicy::Spread { rotation: 1 },
    )
    .expect("one sparse LLC carries the full budget");
    assert_eq!(
        cpus,
        vec![8, 32, 2],
        "rotation is over sorted eligible positions, independent of sparse CPU IDs",
    );
}

#[test]
fn missing_cpu_observation_is_advisory_and_keeps_the_cpu_eligible() {
    let allowed = [2usize, 8]
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    let eligible =
        cpu_eligible_allowed(&allowed, &std::collections::BTreeMap::new(), |_| Ok(false))
            .expect("unknown CPU state remains plannable");
    assert_eq!(eligible, allowed);
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
            holders: Vec::new(),
            holder_count: 0,
            exclusive_held: false,
        },
        LlcSnapshot {
            llc_idx: 1,
            holders: Vec::new(),
            holder_count: 5,
            exclusive_held: false,
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
            holders: Vec::new(),
            holder_count: 3,
            exclusive_held: false,
        },
        LlcSnapshot {
            llc_idx: 1,
            holders: Vec::new(),
            holder_count: 0,
            exclusive_held: false,
        },
        LlcSnapshot {
            llc_idx: 2,
            holders: Vec::new(),
            holder_count: 7,
            exclusive_held: false,
        },
        LlcSnapshot {
            llc_idx: 3,
            holders: Vec::new(),
            holder_count: 1,
            exclusive_held: false,
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
         (preferred single-node contiguous unavailable). Work \
         will proceed; memory-access latency may be higher.",
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
/// Uses `flock(1)` + `sleep` rather than Rust fork() so the
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

    // Child process holds SH on LLC 1's lockfile via flock(1).
    // Keep it alive until explicitly killed so host load cannot
    // close the observation window before the parent's acquire.
    let target_lock = llc_lock_path(1);
    // Ensure the lockfile exists so flock(1) opens the right
    // inode (not a fresh one that /proc/locks would attribute
    // to the flock(1) pid on a different inode than the parent
    // sees).
    crate::flock::materialize(&target_lock).expect("materialize lockfile");

    use std::os::unix::process::CommandExt as _;
    let child = std::process::Command::new("flock")
        .args(["-s", "-n", &target_lock, "sleep", "300"])
        .process_group(0)
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

    // Wait for the peer's LOCK_SH to become observable. A fixed sleep
    // races process scheduling on a saturated runner: the child can
    // remain runnable but unscheduled past any guessed delay.
    let allowed: std::collections::BTreeSet<usize> = [0usize, 1].into_iter().collect();
    let mountinfo = crate::flock::read_mountinfo().expect("read mountinfo");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    let peer_seen = loop {
        let snapshots =
            discover_llc_snapshots(&topo, &allowed, &mountinfo).expect("discover must succeed");
        if snapshots
            .iter()
            .find(|s| s.llc_idx == 1)
            .is_some_and(|s| s.holder_count == 1)
        {
            break true;
        }
        if std::time::Instant::now() >= deadline {
            break false;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    };

    let test_topo = crate::topology::TestTopology::synthetic(2, 1);
    let cap = CpuCap::new(1).expect("cap=1 valid");
    let plan = peer_seen.then(|| {
        acquire_llc_plan(
            &topo,
            &test_topo,
            Some(cap),
            PlacementPolicy::Consolidate,
            false,
        )
    });

    // Sweep both flock and its sleep child before any assertion can
    // panic, then reap flock so the test leaves no process or zombie.
    if let Some(pgid) = libc::pid_t::try_from(child.id())
        .ok()
        .filter(|&p| p > 0)
        .map(nix::unistd::Pid::from_raw)
    {
        let _ = nix::sys::signal::killpg(pgid, nix::sys::signal::Signal::SIGKILL);
    }
    let _ = child.wait();

    assert!(
        peer_seen,
        "peer LOCK_SH on LLC 1 did not become visible within 30s"
    );
    let plan = plan
        .expect("peer was visible, so acquire was attempted")
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
}

#[test]
fn acquire_llc_plan_skips_an_exclusive_held_llc_when_a_ready_alternative_exists() {
    let _llc_prefix = LlcLockPrefixGuard::new();
    let _allowed = AllowedCpusGuard::new(vec![0, 1]);
    let topo = HostTopology::new_for_tests(&[(vec![0], 0), (vec![1], 0)]);
    let peer_path = llc_lock_path(1);
    crate::flock::materialize(&peer_path).expect("materialize peer EX lockfile");
    use std::os::unix::process::CommandExt as _;
    let child = std::process::Command::new("flock")
        .args(["-x", "-n", &peer_path, "sleep", "300"])
        .process_group(0)
        .spawn();
    let mut child = match child {
        Ok(child) => child,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            eprintln!(
                "acquire_llc_plan_skips_an_exclusive_held_llc_when_a_ready_alternative_exists: \
                 flock(1) not available, skipping ({error})"
            );
            return;
        }
        Err(error) => panic!("spawn flock(1): {error}"),
    };
    let allowed: std::collections::BTreeSet<usize> = [0usize, 1].into_iter().collect();
    let mountinfo = crate::flock::read_mountinfo().expect("read mountinfo");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    let peer_seen = loop {
        let snapshots =
            discover_llc_snapshot_counts(&topo, &allowed, &mountinfo).expect("discover EX peer");
        if snapshots
            .iter()
            .find(|snapshot| snapshot.llc_idx == 1)
            .is_some_and(|snapshot| snapshot.exclusive_held)
        {
            break true;
        }
        if std::time::Instant::now() >= deadline {
            break false;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    };
    let test_topo = crate::topology::TestTopology::synthetic(2, 1);
    let plan = peer_seen.then(|| {
        acquire_llc_plan(
            &topo,
            &test_topo,
            CpuCap::new(1).ok(),
            PlacementPolicy::Consolidate,
            false,
        )
    });
    if let Some(pgid) = libc::pid_t::try_from(child.id())
        .ok()
        .filter(|&pid| pid > 0)
        .map(nix::unistd::Pid::from_raw)
    {
        let _ = nix::sys::signal::killpg(pgid, nix::sys::signal::Signal::SIGKILL);
    }
    let _ = child.wait();

    assert!(peer_seen, "peer LLC EX hold did not become observable");
    let plan = plan
        .expect("peer was visible, so acquisition was attempted")
        .expect("the free LLC must be selected without waiting");
    assert_eq!(
        plan.locked_llcs,
        vec![0],
        "Consolidate must not mistake an incompatible EX holder for compatible occupancy",
    );
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
    // Lead a fresh process group so the kill below reaches `flock`'s
    // `sleep` grandchild too, not just `flock` (mirrors the make(1)
    // spawn+killpg pattern in cli::kernel_build::make).
    use std::os::unix::process::CommandExt as _;
    let child = std::process::Command::new("flock")
        .args(["-s", "-n", &peer_lock_path, "sleep", "300"])
        .process_group(0)
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
    // The peer child must be SCHEDULED and take its LOCK_SH before we
    // stat /proc/locks. A fixed sleep races that acquisition on a loaded
    // host — a freshly spawned process can sit unscheduled far longer
    // than any guessed delay — so poll discover until the peer's hold is
    // visible instead of assuming a fixed settle time. The child holds
    // for the poll's whole lifetime (killed below), so the observation
    // window can never fall outside the peer's hold.
    let mountinfo = crate::flock::read_mountinfo().expect("read mountinfo");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    let snapshots = loop {
        let snaps =
            discover_llc_snapshots(&topo, &allowed, &mountinfo).expect("discover must succeed");
        let peer_seen = snaps
            .iter()
            .find(|s| s.llc_idx == 1)
            .is_some_and(|s| s.holder_count == 1);
        if peer_seen || std::time::Instant::now() >= deadline {
            break snaps;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    };

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
    assert!(
        !llc0.exclusive_held,
        "self-held SH remains compatible with another SH requester",
    );
    assert_eq!(
        llc1.holder_count, 1,
        "a peer process's SH must COUNT toward LLC 1's holder_count \
         (got {}, holders={:?})",
        llc1.holder_count, llc1.holders,
    );
    assert!(
        !llc1.exclusive_held,
        "peer-held SH remains compatible with another SH requester",
    );
    assert!(
        llc1.holders.iter().all(|h| h.pid != self_pid),
        "self never locked LLC 1, so must not appear there; holders={:?}",
        llc1.holders,
    );

    // The child holds until killed (sleep 300); sweep its whole process
    // group so `flock` AND its `sleep` grandchild die, then reap `flock`
    // so we leave neither a lingering process nor a zombie. Guard the
    // pgid cast as make's killpg does: a non-positive pgid would broadcast
    // to the caller's group.
    if let Some(pgid) = libc::pid_t::try_from(child.id())
        .ok()
        .filter(|&p| p > 0)
        .map(nix::unistd::Pid::from_raw)
    {
        let _ = nix::sys::signal::killpg(pgid, nix::sys::signal::Signal::SIGKILL);
    }
    let _ = child.wait();
}

#[test]
fn discover_tracks_self_and_peer_exclusive_llc_holds_separately_from_occupancy() {
    let _llc_prefix = LlcLockPrefixGuard::new();
    let topo = HostTopology::new_for_tests(&[(vec![0], 0), (vec![1], 0)]);
    let allowed: std::collections::BTreeSet<usize> = [0usize, 1].into_iter().collect();

    let self_path = llc_lock_path(0);
    crate::flock::materialize(&self_path).expect("materialize self EX lockfile");
    let _self_hold = try_flock(&self_path, FlockMode::Exclusive)
        .expect("open self EX lockfile")
        .expect("take self EX lock");

    let peer_path = llc_lock_path(1);
    crate::flock::materialize(&peer_path).expect("materialize peer EX lockfile");
    use std::os::unix::process::CommandExt as _;
    let child = std::process::Command::new("flock")
        .args(["-x", "-n", &peer_path, "sleep", "300"])
        .process_group(0)
        .spawn();
    let mut child = match child {
        Ok(child) => child,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            eprintln!(
                "discover_tracks_self_and_peer_exclusive_llc_holds_separately_from_occupancy: \
                 flock(1) not available, skipping ({error})"
            );
            return;
        }
        Err(error) => panic!("spawn flock(1): {error}"),
    };

    let mountinfo = crate::flock::read_mountinfo().expect("read mountinfo");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    let snapshots = loop {
        let snapshots =
            discover_llc_snapshots(&topo, &allowed, &mountinfo).expect("discover EX holders");
        if snapshots
            .iter()
            .find(|snapshot| snapshot.llc_idx == 1)
            .is_some_and(|snapshot| snapshot.exclusive_held && snapshot.holder_count == 1)
            || std::time::Instant::now() >= deadline
        {
            break snapshots;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    };

    if let Some(pgid) = libc::pid_t::try_from(child.id())
        .ok()
        .filter(|&pid| pid > 0)
        .map(nix::unistd::Pid::from_raw)
    {
        let _ = nix::sys::signal::killpg(pgid, nix::sys::signal::Signal::SIGKILL);
    }
    let _ = child.wait();

    let self_snapshot = snapshots
        .iter()
        .find(|snapshot| snapshot.llc_idx == 0)
        .expect("self-held LLC snapshot");
    assert_eq!(
        (self_snapshot.exclusive_held, self_snapshot.holder_count),
        (true, 0),
        "self EX is incompatible but excluded only from occupancy scoring",
    );
    let peer_snapshot = snapshots
        .iter()
        .find(|snapshot| snapshot.llc_idx == 1)
        .expect("peer-held LLC snapshot");
    assert_eq!(
        (peer_snapshot.exclusive_held, peer_snapshot.holder_count),
        (true, 1),
        "peer EX is both incompatible and counted for occupancy",
    );
}

#[test]
fn count_only_discover_resolves_no_holder_cmdlines() {
    let _llc_prefix = LlcLockPrefixGuard::new();
    let topo = HostTopology::new_for_tests(&[(vec![0], 0)]);
    let allowed: std::collections::BTreeSet<usize> = [0usize].into_iter().collect();
    let path = llc_lock_path(0);
    crate::flock::materialize(&path).expect("materialize count-only lockfile");
    let _held = try_flock(&path, FlockMode::Shared)
        .expect("open count-only lockfile")
        .expect("hold count-only lockfile");
    let mountinfo = crate::flock::read_mountinfo().expect("read mountinfo");
    let before = crate::flock::proc_locks::batch_holder_info_resolution_count_for_tests();
    let snapshots = discover_llc_snapshot_counts(&topo, &allowed, &mountinfo)
        .expect("count-only LLC discovery");
    let after = crate::flock::proc_locks::batch_holder_info_resolution_count_for_tests();
    assert_eq!(
        after, before,
        "normal placement must not resolve HolderInfo/cmdline data",
    );
    assert_eq!(
        snapshots[0].holder_count, 0,
        "PID-only discovery must still exclude this process from placement counts",
    );
    assert!(
        snapshots[0].holders.is_empty(),
        "count-only snapshots intentionally carry no diagnostic enrichment",
    );
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
        false,
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

    // Another test thread may fork while this fd is live. CLOEXEC closes the
    // inherited copy at exec, but cannot prevent inheritance by the raw-fork
    // queue test, whose children retain it until `_exit`. The kernel's flock
    // entry can therefore outlive our local close briefly in a parallel
    // cargo-test process. Require disappearance within a tight diagnostic
    // deadline rather than in the same instant.
    let mountinfo = crate::flock::read_mountinfo().expect("read mountinfo after release");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    let after = loop {
        let holders = crate::flock::read_holders_with_mountinfo(&lock_path, &mountinfo)
            .expect("read holders after release");
        if holders.is_empty() || std::time::Instant::now() >= deadline {
            break holders;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    };
    assert!(
        after.is_empty(),
        "take-and-drop through the shared borrow must release our \
         LOCK_SH and every fork-inherited copy; holders remain: {after:?}",
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
        false,
        None,
        |_selected, _cpus, _snapshots| {
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

/// Once the outer LLC-plan path owns the real reservation fds, acquisition is
/// committed. Cancellation racing immediately after the final flock succeeds
/// must not replace that success with `Interrupted` and silently discard the
/// reservation. The protocol-level granted/coordinator tests pin the registry
/// commit boundary; this test pins the caller above those branches.
#[test]
fn acquire_llc_plan_commit_is_terminal_when_cancellation_arrives_after_lock_acquire() {
    let _llc_prefix = LlcLockPrefixGuard::new();
    let _allowed = AllowedCpusGuard::new(vec![93550]);
    let topo = synth_host_topo(&[(vec![93550], 0)]);
    let test_topo = crate::topology::TestTopology::synthetic(1, 1);
    let cancelled = AtomicBool::new(false);

    let plan = acquire_llc_plan_with_acquire_fn(
        &topo,
        &test_topo,
        None,
        PlacementPolicy::Consolidate,
        false,
        Some(&cancelled),
        |selected, cpus, snapshots| {
            let acquired =
                match try_acquire_llc_plan_locks_with_evidence(selected, cpus, snapshots)? {
                    LlcLockAttempt::Acquired(locks) => Some(locks),
                    LlcLockAttempt::Contended(_) | LlcLockAttempt::Unavailable => None,
                };
            assert!(
                acquired.is_some(),
                "fresh isolated lock pool must commit on its first probe",
            );
            cancelled.store(true, Ordering::Release);
            Ok(acquired)
        },
    )
    .expect("post-commit cancellation must not replace LLC-plan success");

    assert!(
        cancelled.load(Ordering::Acquire),
        "seam must inject cancellation after the real flock succeeds",
    );
    assert_eq!(plan.locked_llcs, vec![0]);
    assert_eq!(plan.locks.len(), 2);
    assert!(
        try_flock(llc_lock_path(0), FlockMode::Exclusive)
            .expect("probe live LLC reservation")
            .is_none(),
        "returned plan must retain ownership despite post-commit cancellation",
    );

    drop(plan);
    let released = try_flock(llc_lock_path(0), FlockMode::Exclusive)
        .expect("probe dropped LLC reservation")
        .expect("dropping the successful plan must release its reservation");
    drop(released);
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
        false,
        None,
        |_selected, _cpus, _snapshots| {
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

#[test]
fn live_ex_holder_uses_fresh_holder_diagnostics_not_registered_claim_error() {
    let _llc_prefix = LlcLockPrefixGuard::new();
    let _allowed = AllowedCpusGuard::new(vec![93650]);
    let topo = synth_host_topo(&[(vec![93650], 0)]);
    let test_topo = crate::topology::TestTopology::synthetic(1, 1);
    let _held = try_flock(llc_lock_path(0), FlockMode::Exclusive)
        .expect("open live LLC blocker")
        .expect("take live LLC blocker");
    let error = acquire_llc_plan(
        &topo,
        &test_topo,
        CpuCap::new(1).ok(),
        PlacementPolicy::Consolidate,
        false,
    )
    .expect_err("the only LLC is held EX");
    let message = format!("{error:#}");
    assert!(
        !message.contains("registered reservation claims"),
        "a real flock holder is not a registry-claim-only fast failure: {message}",
    );
    assert!(
        message.contains("holders:") && message.contains("LLC 0"),
        "retry exhaustion must rebuild fresh holder diagnostics: {message}",
    );
}

/// A waiting caller performs one real fast attempt, then uses the registry as
/// its retry mechanism. The seam always bounces while the real lockfile pool is
/// free, so the elected coordinator completes immediately.
#[test]
fn acquire_llc_plan_wait_phase_acquires_beyond_the_seam() {
    let _llc_prefix = LlcLockPrefixGuard::new();
    let _allowed = AllowedCpusGuard::new(vec![93700]);
    let topo = synth_host_topo(&[(vec![93700], 0)]);
    let test_topo = crate::topology::TestTopology::synthetic(1, 1);

    let counter = std::cell::Cell::new(0u32);
    let plan = acquire_llc_plan_with_acquire_fn(
        &topo,
        &test_topo,
        None,
        PlacementPolicy::Consolidate,
        true,
        None,
        |_selected, _cpus, _snapshots| {
            counter.set(counter.get() + 1);
            // Fast phase always bounces; the wait phase must succeed
            // without this seam.
            Ok(None)
        },
    )
    .expect("wait phase must acquire via real lockfiles after the fast phase exhausts");
    assert_eq!(
        counter.get(),
        1,
        "waiting acquisition must make exactly one fast attempt before registry admission",
    );
    assert_eq!(plan.locked_llcs, vec![0]);
    assert_eq!(
        plan.locks.len(),
        2,
        "head-acquired LLC and CPU LOCK_SH fds ride the plan"
    );
}

#[test]
fn plan_only_falls_back_to_a_full_static_shape_when_every_llc_is_exclusive_held() {
    let _llc_prefix = LlcLockPrefixGuard::new();
    let _allowed = AllowedCpusGuard::new(vec![93720, 93721]);
    let topo = synth_host_topo(&[(vec![93720], 0), (vec![93721], 0)]);
    let test_topo = crate::topology::TestTopology::synthetic(2, 1);
    let _held: Vec<_> = [0usize, 1]
        .into_iter()
        .map(|llc| {
            try_flock(llc_lock_path(llc), FlockMode::Exclusive)
                .expect("open LLC EX blocker")
                .expect("take LLC EX blocker")
        })
        .collect();
    let plan = plan_llc_selection_only(
        &topo,
        &test_topo,
        CpuCap::new(2).ok(),
        PlacementPolicy::Spread { rotation: 0 },
    )
    .expect("temporary EX contention must not make plan-only fail");
    assert_eq!(plan.locked_llcs, vec![0, 1]);
    assert_eq!(plan.cpus, vec![93720, 93721]);
    assert!(plan.locks.is_empty(), "plan-only never owns resource fds");
}

#[test]
fn registered_claim_fast_fails_without_acquire_or_diagnostic_enrichment() {
    let _llc_prefix = LlcLockPrefixGuard::new();
    let _allowed = AllowedCpusGuard::new(vec![93750]);
    let topo = synth_host_topo(&[(vec![93750], 0)]);
    let test_topo = crate::topology::TestTopology::synthetic(1, 1);
    let claim =
        admission_protocol::ClaimSet::new([0usize], std::iter::empty(), FlockMode::Exclusive);
    let coordinator =
        match admission_protocol::register_ticket_or_acquire(claim.clone(), claim, None, |_| {
            Ok::<Option<()>, anyhow::Error>(None)
        })
        .expect("register exclusive LLC claim")
        {
            admission_protocol::TicketWork::Coordinator(coordinator) => coordinator,
            admission_protocol::TicketWork::Acquired(()) => {
                panic!("fresh registry must elect a coordinator")
            }
        };
    let attempts = std::cell::Cell::new(0usize);
    let enrichment_before =
        crate::flock::proc_locks::batch_holder_info_resolution_count_for_tests();
    let error = acquire_llc_plan_with_acquire_fn(
        &topo,
        &test_topo,
        None,
        PlacementPolicy::Consolidate,
        false,
        None,
        |_, _, _| {
            attempts.set(attempts.get() + 1);
            Ok(None)
        },
    )
    .expect_err("a covering registered claim must fail fast");
    assert!(error.downcast_ref::<ResourceContention>().is_some());
    assert_eq!(
        attempts.get(),
        0,
        "aggregate-proven claim contention must not touch the real acquire seam",
    );
    assert_eq!(
        crate::flock::proc_locks::batch_holder_info_resolution_count_for_tests(),
        enrichment_before,
        "aggregate-proven contention must not run enriched final diagnostics",
    );
    drop(coordinator);
}

#[test]
fn waiting_handoff_publishes_ticket_before_releasing_old_locks() {
    let _llc_prefix = LlcLockPrefixGuard::new();
    let _allowed = AllowedCpusGuard::new(vec![93775]);
    let topo = synth_host_topo(&[(vec![93775], 0)]);
    let test_topo = crate::topology::TestTopology::synthetic(1, 1);
    let blocker = try_flock(llc_lock_path(0), FlockMode::Exclusive)
        .expect("open build-time blocker")
        .expect("hold build-time blocker");
    let handed_off = std::cell::Cell::new(false);
    let attempts = std::cell::Cell::new(0usize);
    let enrichment_before =
        crate::flock::proc_locks::batch_holder_info_resolution_count_for_tests();
    let plan = acquire_llc_plan_with_acquire_fn_and_handoff(
        LlcPlanAcquireRequest {
            topo: &topo,
            test_topo: &test_topo,
            cpu_cap: None,
            policy: PlacementPolicy::Spread { rotation: 0 },
            wait: true,
            cancelled: None,
        },
        Some(|| {
            let records = admission_protocol::ticket_registry_snapshot_for_tests()
                .expect("handoff hook must be able to read the published ticket");
            assert_eq!(
                records.len(),
                1,
                "replacement ticket must be durable before inherited locks release",
            );
            assert_eq!(
                records[0].2.llcs,
                [0usize].into_iter().collect(),
                "an all-busy wait must retain a nonempty static LLC designation",
            );
            assert_eq!(
                records[0].2.cpus,
                [93775usize].into_iter().collect(),
                "an all-busy wait must retain the full-budget static CPU designation",
            );
            handed_off.set(true);
            drop(blocker);
        }),
        |selected, cpus, snapshots| {
            assert!(
                !handed_off.get(),
                "old reservations must remain held through the fast probe",
            );
            attempts.set(attempts.get() + 1);
            Ok(
                match try_acquire_llc_plan_locks_with_evidence(selected, cpus, snapshots)? {
                    LlcLockAttempt::Acquired(locks) => Some(locks),
                    LlcLockAttempt::Contended(_) | LlcLockAttempt::Unavailable => None,
                },
            )
        },
    )
    .expect("handoff release must let the registered coordinator acquire");
    assert!(
        handed_off.get(),
        "contention must invoke the post-publication handoff"
    );
    assert_eq!(
        attempts.get(),
        1,
        "handoff waiting path must make exactly one fast acquire call",
    );
    assert_eq!(
        crate::flock::proc_locks::batch_holder_info_resolution_count_for_tests(),
        enrichment_before,
        "waiting handoff must use count-only discovery",
    );
    assert_eq!(plan.locked_llcs, vec![0]);
}

#[test]
fn waiting_handoff_releases_inherited_locks_on_acquire_error() {
    let _llc_prefix = LlcLockPrefixGuard::new();
    let _allowed = AllowedCpusGuard::new(vec![93776]);
    let topo = synth_host_topo(&[(vec![93776], 0)]);
    let test_topo = crate::topology::TestTopology::synthetic(1, 1);
    let inherited = try_flock(llc_lock_path(0), FlockMode::Shared)
        .expect("open inherited reservation")
        .expect("hold inherited reservation");
    let handed_off = std::cell::Cell::new(false);
    let handed_off_in_hook = &handed_off;

    let error = acquire_llc_plan_with_acquire_fn_and_handoff(
        LlcPlanAcquireRequest {
            topo: &topo,
            test_topo: &test_topo,
            cpu_cap: None,
            policy: PlacementPolicy::Spread { rotation: 0 },
            wait: true,
            cancelled: None,
        },
        Some(move || {
            handed_off_in_hook.set(true);
            drop(inherited);
        }),
        |_, _, _| -> anyhow::Result<Option<Vec<std::os::fd::OwnedFd>>> {
            anyhow::bail!("injected runtime replan failure")
        },
    )
    .expect_err("injected acquisition error must propagate");
    assert!(
        error
            .to_string()
            .contains("injected runtime replan failure"),
        "unexpected handoff error: {error:#}"
    );
    assert!(
        handed_off.get(),
        "every error exit must drain inherited no-perf reservations"
    );
    let released = try_flock(llc_lock_path(0), FlockMode::Exclusive)
        .expect("probe released inherited reservation")
        .expect("inherited reservation must not survive an acquisition error");
    drop(released);
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
            holders: Vec::new(),
            holder_count: if idx == 3 { 5 } else { 0 },
            exclusive_held: false,
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
            holders: Vec::new(),
            holder_count: 0,
            exclusive_held: false,
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
            holders: Vec::new(),
            holder_count: 5,
            exclusive_held: false,
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
/// with the right severity for intentional sharing vs default fallback.
#[test]
fn overcommit_warning_severity_and_polarity() {
    // Budget >= vcpus: not oversubscribed -> no warning, either severity.
    assert_eq!(overcommit_warning(32, 32, false), None);
    assert_eq!(overcommit_warning(40, 32, true), None);

    // No-perf / explicit cpu-budget sharing is a compact factual note.
    let m = overcommit_warning(4, 32, true).expect("4 < 32 => Some");
    assert!(
        m.contains("no-perf/cpu-budget mode"),
        "intentional case must name the mode: {m}"
    );
    assert!(
        !m.contains("worst_iterations_per_cpu_sec"),
        "placement note must not prescribe a workload metric: {m}",
    );
    assert!(
        !m.contains("watchdog"),
        "placement note must not guess at watchdog tuning: {m}",
    );

    // Default fallback remains visibly a warning.
    let m = overcommit_warning(4, 32, false).expect("4 < 32 => Some");
    assert!(
        m.contains("WARNING"),
        "auto case must be a louder WARNING: {m}"
    );
    assert!(
        m.contains("host capacity is below guest width"),
        "fallback case must name the capacity mismatch: {m}",
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
    let err = acquire_llc_plan(&topo, &test_topo, None, PlacementPolicy::Consolidate, false)
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
            holders: Vec::new(),
            holder_count: 0,
            exclusive_held: false,
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
    let plan = acquire_llc_plan(
        &topo,
        &test_topo,
        Some(cap),
        PlacementPolicy::Consolidate,
        false,
    )
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
            holders: Vec::new(),
            holder_count: 0,
            exclusive_held: false,
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
    let plan = acquire_llc_plan(
        &topo,
        &test_topo,
        Some(cap),
        PlacementPolicy::Consolidate,
        false,
    )
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
    let outcome = acquire_resource_locks_waiting_impl(
        &[95100usize],
        LlcLockMode::Exclusive,
        &[95100],
        FlockMode::Exclusive,
        false,
        None,
    )
    .unwrap();
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
    let outcome = acquire_resource_locks_waiting_impl(
        &[95200usize],
        LlcLockMode::Exclusive,
        &[95200],
        FlockMode::Exclusive,
        false,
        None,
    )
    .unwrap();
    let (_, locks) = unwrap_acquired(outcome, Some("with empty-string bypass inert"));
    assert_eq!(
        locks.len(),
        2,
        "empty-string cargo-test-mode is inert — expected the \
         explicit `Exclusive` path to take one LLC fd plus its CPU bridge, \
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
            holders: Vec::new(),
            holder_count,
            exclusive_held: false,
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

/// A queue-only predecessor has no `/proc/locks` holder yet. Folding its
/// exact claim into the planning snapshot must move a flexible no-perf waiter
/// to a free alternative instead of republishing the same blocked claim.
#[test]
fn spread_deprioritizes_predecessor_claims_missing_from_proc_locks() {
    let topo = synth_host_topo(&[(vec![0], 0), (vec![1], 0), (vec![2], 0)]);
    let allowed: std::collections::BTreeSet<usize> = (0..3).collect();
    let snapshots = avoid_preceding_claims_when_possible(
        &spread_snapshots(&[0, 0, 0]),
        1,
        &topo,
        &allowed,
        |candidate| Ok(candidate.llcs.contains(&0)),
    )
    .expect("fold predecessor reservation into snapshots");
    let selected = plan_from_snapshots(
        &snapshots,
        1,
        &topo,
        &allowed,
        |_, _| 10,
        PlacementPolicy::Spread { rotation: 0 },
    );
    assert_eq!(
        selected,
        vec![1],
        "LLC 0's queue-only reservation must not create head-of-line blocking",
    );
}

#[test]
fn consolidate_avoids_predecessor_claims_missing_from_proc_locks() {
    let topo = synth_host_topo(&[(vec![0], 0), (vec![1], 0), (vec![2], 0)]);
    let allowed: std::collections::BTreeSet<usize> = (0..3).collect();
    let snapshots = avoid_preceding_claims_when_possible(
        &spread_snapshots(&[0, 0, 0]),
        1,
        &topo,
        &allowed,
        |candidate| Ok(candidate.llcs.contains(&0)),
    )
    .expect("fold predecessor reservation into snapshots");
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
        "Consolidate must not mistake a queue-only reservation for useful load",
    );
}

#[test]
fn predecessor_claims_remain_eligible_when_exact_budget_needs_them() {
    let topo = synth_host_topo(&[(vec![0], 0), (vec![1], 0)]);
    let allowed: std::collections::BTreeSet<usize> = (0..2).collect();
    let snapshots = avoid_preceding_claims_when_possible(
        &spread_snapshots(&[0, 0]),
        2,
        &topo,
        &allowed,
        |candidate| Ok(candidate.llcs.contains(&0)),
    )
    .expect("retain a predecessor when the whole host is required");
    let selected = plan_from_snapshots(
        &snapshots,
        2,
        &topo,
        &allowed,
        |_, _| 10,
        PlacementPolicy::Spread { rotation: 0 },
    );
    assert_eq!(selected, vec![0, 1]);
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

#[test]
fn bounded_planner_stays_within_prime_host_resource_count() {
    let groups = (0..7)
        .map(|llc| {
            let start = llc * 5;
            (((start)..(start + 5)).collect::<Vec<_>>(), llc % 2)
        })
        .collect::<Vec<_>>();
    let host = HostTopology::new_for_tests(&groups);
    let allowed = host.online_cpus.clone();
    let candidates = host
        .default_pinning_candidates_for_cpus(&Topology::new(1, 3, 1, 1), &allowed)
        .expect("prime/coprime host shape must be plannable");
    assert!(!candidates.is_empty());
    assert!(
        candidates.len() <= allowed.len().max(host.llc_groups.len()),
        "planner work/output must be bounded by physical resources",
    );
}

#[test]
fn planner_selects_fragmented_preferred_cpus_exactly() {
    let host = synth_host_topo(&[(vec![0, 1, 2, 3], 0)]);
    let allowed = vec![0, 1, 2, 3];
    let preferred = std::collections::BTreeSet::from([0usize, 2]);
    let llcs = std::collections::BTreeSet::from([0usize]);
    let candidates = host
        .topology_pinning_candidates(
            &Topology::new(1, 1, 2, 1),
            PinningKind::Default,
            &allowed,
            Some(PinningPreferences {
                cpus: &preferred,
                shared_llcs: &llcs,
                exclusive_llcs: &llcs,
            }),
        )
        .expect("fragmented exact CPU set must fit");
    assert!(
        candidates.iter().any(|candidate| {
            candidate
                .plan
                .assignments
                .iter()
                .map(|&(_, cpu)| cpu)
                .collect::<std::collections::BTreeSet<_>>()
                == preferred
        }),
        "availability-aware matching must not require a contiguous window",
    );
}

#[test]
fn planner_rotates_sparse_cpu_ids_by_dense_rank() {
    let host = synth_host_topo(&[(vec![0, 2, 4], 0)]);
    let candidates = host
        .default_pinning_candidates_for_cpus(&Topology::new(1, 1, 1, 1), &[0, 2, 4])
        .expect("sparse CPU IDs must expose every exact one-CPU placement");
    let footprints = candidates
        .iter()
        .map(|candidate| candidate.cpu_reservations.clone())
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        footprints,
        std::collections::BTreeSet::from([vec![0], vec![2], vec![4]]),
        "numeric CPU IDs modulo CPU count must not collapse candidate rotation",
    );
}

#[test]
fn planner_rotates_sparse_host_numa_ids_by_dense_rank() {
    let host = synth_host_topo(&[(vec![0], 0), (vec![1], 3), (vec![2], 6)]);
    let candidates = host
        .default_pinning_candidates_for_cpus(&Topology::new(2, 2, 1, 1), &[0, 1, 2])
        .expect("two guest nodes fit on every pair of three sparse host nodes");
    let host_node_pairs = candidates
        .iter()
        .map(|candidate| {
            candidate
                .plan
                .assignments
                .iter()
                .map(|&(_, cpu)| host.cpu_to_node[&cpu])
                .collect::<std::collections::BTreeSet<_>>()
        })
        .collect::<std::collections::BTreeSet<_>>();
    assert!(
        host_node_pairs.iter().any(|nodes| nodes.contains(&6)),
        "raw sparse node IDs modulo node count must not leave node 6 unseen",
    );
    assert_eq!(
        host_node_pairs,
        std::collections::BTreeSet::from([
            std::collections::BTreeSet::from([0, 3]),
            std::collections::BTreeSet::from([0, 6]),
            std::collections::BTreeSet::from([3, 6]),
        ]),
    );
}

#[test]
fn llc_partition_validation_rejects_every_malformed_shape() {
    let error = validate_llc_partition(&[], &[]).unwrap_err();
    assert!(error.to_string().contains("empty online CPU set"));

    let error = validate_llc_partition(&[], &[0]).unwrap_err();
    assert!(error.to_string().contains("zero LLC groups"));

    let error = validate_llc_partition(&[LlcGroup { cpus: Vec::new() }], &[0]).unwrap_err();
    assert!(error.to_string().contains("empty"));

    let error = validate_llc_partition(&[LlcGroup { cpus: vec![0, 0] }], &[0]).unwrap_err();
    assert!(error.to_string().contains("repeats CPU 0"));

    let error = validate_llc_partition(
        &[LlcGroup { cpus: vec![0, 1] }, LlcGroup { cpus: vec![1, 2] }],
        &[0, 1, 2],
    )
    .unwrap_err();
    assert!(error.to_string().contains("more than one"));

    let error = validate_llc_partition(&[LlcGroup { cpus: vec![0] }], &[0, 1]).unwrap_err();
    assert!(error.to_string().contains("omit online CPUs [1]"));

    let error = validate_llc_partition(&[LlcGroup { cpus: vec![0, 2] }], &[0, 1]).unwrap_err();
    assert!(error.to_string().contains("non-online CPU 2"));
}

#[test]
fn planner_collapses_a_full_domain_to_one_material_seed() {
    let cpus = (0..192).collect::<Vec<_>>();
    let host = synth_host_topo(&[(cpus.clone(), 0)]);
    let candidates = host
        .default_pinning_candidates_for_cpus(&Topology::new(1, 1, 192, 1), &cpus)
        .expect("a full-domain guest must fit exactly once");
    assert_eq!(
        candidates.len(),
        1,
        "a 192-vCPU full-domain fit must not materialize 192 identical \
         192-by-192 CPU matchings",
    );
    assert_eq!(candidates[0].cpu_reservations, cpus);
    assert_eq!(
        candidates[0]
            .plan
            .assignments
            .iter()
            .map(|&(_, cpu)| cpu)
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        192,
        "direct per-bin selection must retain one globally distinct host CPU \
         per guest vCPU",
    );
}

#[test]
fn whole_domain_identity_and_cpu_bridge_are_cpuset_independent() {
    let host = synth_host_topo(&[(vec![0, 1, 2], 0), (vec![3, 4, 5], 0)]);
    let topology = Topology::new(1, 1, 1, 1);
    let left = host
        .performance_pinning_candidates_for_cpus(&topology, &[0, 1])
        .expect("left slice of LLC 0 must fit")
        .into_iter()
        .find(|candidate| candidate.plan.llc_indices == vec![0])
        .expect("left cpuset must retain physical LLC 0 identity");
    let right = host
        .performance_pinning_candidates_for_cpus(&topology, &[1, 2])
        .expect("right slice of LLC 0 must fit")
        .into_iter()
        .find(|candidate| candidate.plan.llc_indices == vec![0])
        .expect("right cpuset must retain physical LLC 0 identity");

    assert_eq!(left.llc_mode, LlcLockMode::Exclusive);
    assert_eq!(right.llc_mode, LlcLockMode::Exclusive);
    assert_eq!(left.plan.llc_indices, right.plan.llc_indices);
    assert_eq!(left.cpu_reservations, vec![0, 1, 2]);
    assert_eq!(right.cpu_reservations, vec![0, 1, 2]);
    assert_eq!(
        left.claim().cpus,
        [0, 1, 2].into_iter().collect(),
        "the production whole-EX candidate claim must include the sibling \
         outside either caller's allowed mask",
    );
}

#[test]
fn planner_seed_schedule_is_linear_on_heterogeneous_llcs() {
    let mut groups = vec![((0..96).collect::<Vec<_>>(), 0)];
    groups.extend((96..192).map(|cpu| (vec![cpu], 0)));
    let host = synth_host_topo(&groups);
    let allowed = (0..192).collect::<Vec<_>>();
    let allowed_set = allowed.iter().copied().collect();
    let demands = [GuestLlcDemand {
        guest_llc: 0,
        guest_node: 0,
        vcpu_start: 0,
        cpus: 1,
    }];
    let seeds = planner_seed_schedule(&host, &demands, PinningKind::Default, &allowed_set);
    assert_eq!(
        seeds.len(),
        192,
        "one 96-CPU LLC plus 96 singleton LLCs must produce a flattened \
         192-grain schedule, not a 97-by-96 Cartesian product",
    );
    let candidates = host
        .default_pinning_candidates_for_cpus(&Topology::new(1, 1, 1, 1), &allowed)
        .expect("every physical one-CPU grain must be materialized");
    assert_eq!(candidates.len(), 192);
}

#[test]
fn bin_matching_displaces_flexible_requests_instead_of_getting_stuck_greedily() {
    let edges = std::collections::BTreeMap::from([
        (0usize, vec![0usize, 1]),
        (1usize, vec![0]),
        (2usize, vec![1, 2]),
    ]);
    let matched = match_distinct_bins(&edges, &[0, 1, 2])
        .expect("augmenting paths must find the complete 0→1, 1→0, 2→2 match");
    assert_eq!(
        matched,
        std::collections::BTreeMap::from([(0, 1), (1, 0), (2, 2)]),
    );
}

#[test]
fn preferred_snapshot_synthesizes_nonadjacent_ready_llc_combination() {
    let host = synth_host_topo(&[(vec![0], 0), (vec![1], 0), (vec![2], 0), (vec![3], 0)]);
    let preferred = std::collections::BTreeSet::from([0usize, 2]);
    let all_llcs = std::collections::BTreeSet::from([0usize, 1, 2, 3]);
    let candidates = host
        .topology_pinning_candidates(
            &Topology::new(1, 2, 1, 1),
            PinningKind::Default,
            &[0, 1, 2, 3],
            Some(PinningPreferences {
                cpus: &preferred,
                shared_llcs: &all_llcs,
                exclusive_llcs: &all_llcs,
            }),
        )
        .expect("two nonadjacent ready LLCs must form an exact candidate");
    assert!(
        candidates.iter().any(|candidate| {
            candidate.plan.llc_indices == vec![0, 2] && candidate.cpu_reservations == vec![0, 2]
        }),
        "availability must outrank cyclic seed rotation so a coordinator can \
         synthesize the simultaneously ready {{0,2}} placement",
    );
}

#[test]
fn preferred_llc_modes_synthesize_nonadjacent_whole_perf_combination() {
    let host = synth_host_topo(&[
        (vec![0, 1], 0),
        (vec![2, 3], 0),
        (vec![4, 5], 0),
        (vec![6, 7], 0),
    ]);
    let cpus = (0..8).collect::<std::collections::BTreeSet<_>>();
    let shared_llcs = std::collections::BTreeSet::from([0usize, 1, 2, 3]);
    let exclusive_llcs = std::collections::BTreeSet::from([0usize, 2]);
    let candidates = host
        .topology_pinning_candidates(
            &Topology::new(1, 2, 1, 1),
            PinningKind::Performance,
            &(0..8).collect::<Vec<_>>(),
            Some(PinningPreferences {
                cpus: &cpus,
                shared_llcs: &shared_llcs,
                exclusive_llcs: &exclusive_llcs,
            }),
        )
        .expect("whole-LLC availability must synthesize the free 0+2 pair");
    assert!(candidates.iter().any(|candidate| {
        candidate.llc_mode == LlcLockMode::Exclusive && candidate.plan.llc_indices == vec![0, 2]
    }));
}

#[test]
fn preferred_llc_modes_cover_service_only_whole_perf_domain() {
    let host = synth_host_topo(&[(vec![0], 0), (vec![1], 0), (vec![2], 0), (vec![3], 0)]);
    let cpus = std::collections::BTreeSet::from([0usize, 1, 2, 3]);
    let shared_llcs = std::collections::BTreeSet::from([0usize, 1, 2, 3]);
    let exclusive_llcs = std::collections::BTreeSet::from([0usize, 2]);
    let candidates = host
        .topology_pinning_candidates(
            &Topology::new(1, 1, 1, 1),
            PinningKind::Performance,
            &[0, 1, 2, 3],
            Some(PinningPreferences {
                cpus: &cpus,
                shared_llcs: &shared_llcs,
                exclusive_llcs: &exclusive_llcs,
            }),
        )
        .expect("vCPU and service may occupy the two nonadjacent free LLCs");
    assert!(candidates.iter().any(|candidate| {
        candidate.llc_mode == LlcLockMode::Exclusive
            && candidate.plan.llc_indices == vec![0, 2]
            && candidate.cpu_reservations == vec![0, 2]
    }));
}

#[test]
fn service_cpu_uses_a_ready_global_llc_before_a_busy_local_fallback() {
    let host = synth_host_topo(&[
        ((0..32).collect::<Vec<_>>(), 0),
        ((32..64).collect::<Vec<_>>(), 0),
    ]);
    let preferred_cpus = std::collections::BTreeSet::from([0usize, 1, 32]);
    let ready_llcs = std::collections::BTreeSet::from([0usize, 1]);
    let candidates = host
        .topology_pinning_candidates(
            &Topology::new(1, 1, 2, 1),
            PinningKind::Performance,
            &[0, 1, 2, 32],
            Some(PinningPreferences {
                cpus: &preferred_cpus,
                shared_llcs: &ready_llcs,
                exclusive_llcs: &ready_llcs,
            }),
        )
        .expect("a free service CPU in another LLC must remain eligible");
    assert!(candidates.iter().any(|candidate| {
        candidate.llc_mode == LlcLockMode::Shared
            && candidate.plan.service_cpu == Some(32)
            && candidate.plan.llc_indices == vec![0, 1]
            && candidate.cpu_reservations == vec![0, 1, 32]
    }));
}

#[test]
fn heterogeneous_grain_service_placement_is_demand_order_independent() {
    let host = synth_host_topo(&[
        ((0..32).collect::<Vec<_>>(), 0),
        ((32..64).collect::<Vec<_>>(), 0),
    ]);
    let allowed = (0..15).chain([32, 33]).collect::<Vec<_>>();
    let preferred_cpus = allowed.iter().copied().collect();
    let shared_llcs = std::collections::BTreeSet::from([0usize, 1]);
    let exclusive_llcs = std::collections::BTreeSet::new();

    let plan = |llc_cores: &'static [u32]| {
        let mut topology = Topology::new(1, 2, 16, 1);
        topology.llc_cores = Some(llc_cores);
        host.topology_pinning_candidates(
            &topology,
            PinningKind::Performance,
            &allowed,
            Some(PinningPreferences {
                cpus: &preferred_cpus,
                shared_llcs: &shared_llcs,
                exclusive_llcs: &exclusive_llcs,
            }),
        )
        .expect("service must fit on the one-CPU guest LLC's spare grain")
        .into_iter()
        .find(|candidate| {
            candidate.llc_mode == LlcLockMode::Shared
                && candidate.cpu_reservations == (0..15).chain([32, 33]).collect::<Vec<_>>()
        })
        .expect("SH-ready 15/2 footprint must be preferred over whole EX")
    };

    let large_first = plan(&[15, 1]);
    let small_first = plan(&[1, 15]);
    assert_eq!(large_first.plan.service_cpu, Some(33));
    assert_eq!(small_first.plan.service_cpu, Some(33));
    assert_eq!(large_first.plan.llc_indices, vec![0, 1]);
    assert_eq!(small_first.plan.llc_indices, vec![0, 1]);
    assert_eq!(
        large_first.cpu_reservations, small_first.cpu_reservations,
        "reversing guest LLC declaration order must not change the material \
         SH-grain claim",
    );

    let mut static_topology = Topology::new(1, 2, 16, 1);
    static_topology.llc_cores = Some(&[15, 1]);
    let static_candidates = host
        .topology_pinning_candidates(
            &static_topology,
            PinningKind::Performance,
            &(0..64).collect::<Vec<_>>(),
            None,
        )
        .expect("idle-host planning must preserve the available shared grain");
    let static_first = static_candidates
        .first()
        .expect("idle-host planning must produce a candidate");
    assert_eq!(
        static_first.llc_mode,
        LlcLockMode::Shared,
        "when SH and EX are both ready, service locality must not turn a valid \
         small-bin grain into a whole-domain reservation",
    );
    let small_guest_cpu = static_first
        .plan
        .assignments
        .iter()
        .find_map(|&(vcpu, cpu)| (vcpu == 15).then_some(cpu))
        .expect("one-vCPU guest LLC assignment");
    let service_cpu = static_first.plan.service_cpu.expect("service CPU");
    assert!(
        host.llc_groups
            .iter()
            .any(|group| group.cpus.contains(&small_guest_cpu)
                && group.cpus.contains(&service_cpu)),
        "the service CPU must preserve the shared grain by joining the \
         one-vCPU guest LLC",
    );
}

#[test]
fn preferred_snapshot_guides_strict_numa_matching_to_nonadjacent_nodes() {
    let host = synth_host_topo(&[(vec![0], 0), (vec![1], 1), (vec![2], 2), (vec![3], 3)]);
    let preferred = std::collections::BTreeSet::from([0usize, 2]);
    let preferred_llcs = std::collections::BTreeSet::from([0usize, 2]);
    let all_llcs = std::collections::BTreeSet::from([0usize, 1, 2, 3]);
    let candidates = host
        .topology_pinning_candidates(
            &Topology::new(2, 2, 1, 1),
            PinningKind::Default,
            &[0, 1, 2, 3],
            Some(PinningPreferences {
                cpus: &preferred,
                shared_llcs: &preferred_llcs,
                exclusive_llcs: &all_llcs,
            }),
        )
        .expect("strict NUMA matching must use the simultaneously ready nodes");
    assert!(candidates.iter().any(|candidate| {
        candidate
            .plan
            .assignments
            .iter()
            .map(|&(_, cpu)| host.cpu_to_node[&cpu])
            .collect::<std::collections::BTreeSet<_>>()
            == std::collections::BTreeSet::from([0usize, 2])
    }));
}

#[test]
fn planner_matches_nonuniform_guest_llcs_largest_demand_first() {
    let host = synth_host_topo(&[(vec![0], 0), (vec![1, 2, 3], 0)]);
    let mut guest = Topology::new(1, 2, 3, 1);
    guest.llc_cores = Some(&[3, 1]);
    let candidate = host
        .default_pinning_candidates_for_cpus(&guest, &[0, 1, 2, 3])
        .expect("heterogeneous bins fit")
        .into_iter()
        .next()
        .unwrap();
    let large = candidate.plan.assignments[..3]
        .iter()
        .map(|&(_, cpu)| cpu)
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(large, std::collections::BTreeSet::from([1, 2, 3]));
    assert_eq!(candidate.plan.assignments[3].1, 0);
}

#[test]
fn planner_uses_strict_numa_mapping_then_global_fallback() {
    let strict_host = synth_host_topo(&[
        (vec![0, 1], 0),
        (vec![2, 3], 0),
        (vec![4, 5], 1),
        (vec![6, 7], 1),
    ]);
    let guest = Topology::new(2, 2, 1, 1);
    let strict = strict_host
        .default_pinning_candidates_for_cpus(&guest, &(0..8).collect::<Vec<_>>())
        .unwrap();
    assert!(strict.iter().all(|candidate| {
        let first = strict_host.cpu_to_node[&candidate.plan.assignments[0].1];
        let second = strict_host.cpu_to_node[&candidate.plan.assignments[1].1];
        first != second
    }));

    let one_node_host = synth_host_topo(&[(vec![0, 1], 0), (vec![2, 3], 0), (vec![4, 5], 0)]);
    assert!(
        one_node_host
            .default_pinning_candidates_for_cpus(&guest, &(0..6).collect::<Vec<_>>())
            .is_ok(),
        "strict NUMA failure must fall back to a global distinct-LLC mapping",
    );
}

#[test]
fn planner_covers_service_only_domain() {
    let service_host = synth_host_topo(&[(vec![0], 0), (vec![1], 0)]);
    let performance = service_host
        .performance_pinning_candidates_for_cpus(&Topology::new(1, 1, 1, 1), &[0, 1])
        .unwrap();
    assert!(performance.iter().all(|candidate| {
        candidate.plan.llc_indices.len() == 2
            && candidate.cpu_reservations == vec![0, 1]
            && candidate.plan.service_cpu.is_some()
    }));
}

#[test]
fn planner_reservation_matrix_is_explicit_and_canonical() {
    let small = synth_host_topo(&[(vec![0, 1, 2, 3], 0)]);
    let default = small
        .default_pinning_candidates_for_cpus(&Topology::new(1, 1, 2, 1), &[0, 1, 2, 3])
        .unwrap();
    assert_eq!(
        default[0].claim().llc_mode,
        admission_protocol::ClaimMode::Shared
    );
    assert_eq!(
        default[0].claim().cpu_mode,
        admission_protocol::ClaimMode::Exclusive,
    );
    assert_eq!(default[0].cpu_reservations.len(), 2);

    let whole = small
        .performance_pinning_candidates_for_cpus(&Topology::new(1, 1, 2, 1), &[0, 1, 2, 3])
        .unwrap();
    assert!(whole.iter().all(|candidate| {
        candidate.llc_mode == LlcLockMode::Exclusive
            && candidate.cpu_reservations == vec![0, 1, 2, 3]
            && candidate.claim().llc_mode == admission_protocol::ClaimMode::Exclusive
            && candidate.claim().cpu_mode == admission_protocol::ClaimMode::Exclusive
    }));

    let restricted = small
        .performance_pinning_candidates_for_cpus(&Topology::new(1, 1, 1, 1), &[0, 1])
        .unwrap();
    assert!(
        restricted
            .iter()
            .all(|candidate| { candidate.cpu_reservations == vec![0, 1, 2, 3] }),
        "whole-LLC CPU bridge must cover sibling CPUs outside this process cpuset"
    );
    let unknown_topology_bridge = resource_claim_with_modes(
        &[],
        LlcLockMode::Shared,
        &[0, 1, 2, 3],
        FlockMode::Exclusive,
    );
    assert!(
        unknown_topology_bridge.conflicts_with(&restricted[0].claim()),
        "CPU EX alone must bridge a whole-perf reservation when LLCs are unknown",
    );
    let known_topology_bridge =
        resource_claim_with_modes(&[0], LlcLockMode::Shared, &[0, 1], FlockMode::Exclusive);
    assert_eq!(
        known_topology_bridge.llc_mode,
        admission_protocol::ClaimMode::Shared,
    );
    assert_eq!(
        known_topology_bridge.cpu_mode,
        admission_protocol::ClaimMode::Exclusive,
    );

    let monolith = synth_host_topo(&[((0..36).collect(), 0)]);
    let grain = monolith
        .performance_pinning_candidates_for_cpus(
            &Topology::new(1, 1, 2, 1),
            &(0..36).collect::<Vec<_>>(),
        )
        .unwrap();
    assert!(grain.iter().all(|candidate| {
        candidate.llc_mode == LlcLockMode::Shared
            && candidate.cpu_reservations.len() == 3
            && candidate.claim().cpu_mode == admission_protocol::ClaimMode::Exclusive
    }));
}

#[test]
fn build_and_runtime_entry_points_return_identical_claim_sets() {
    let host = synth_host_topo(&[((0..36).collect(), 0), ((36..72).collect(), 1)]);
    let guest = Topology::new(1, 1, 2, 1);
    let allowed = (0..72).collect::<Vec<_>>();
    let build = host
        .performance_pinning_candidates_for_cpus(&guest, &allowed)
        .unwrap()
        .into_iter()
        .map(|candidate| candidate.claim())
        .collect::<Vec<_>>();
    let runtime = host
        .performance_pinning_candidates_for_cpus(&guest, &allowed)
        .unwrap()
        .into_iter()
        .map(|candidate| candidate.claim())
        .collect::<Vec<_>>();
    assert_eq!(build, runtime);
}
