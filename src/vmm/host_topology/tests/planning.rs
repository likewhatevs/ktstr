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
    let err = plan_llc_selection_only(&topo, &test_topo, Some(cap), PlacementPolicy::Consolidate)
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
            holder_count: if idx >= 2 { 5 } else { 0 },
            exclusive_held: false,
            granted_count: 0,
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
            holder_count: 0,
            exclusive_held: false,
            granted_count: 0,
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
            holder_count: 0,
            exclusive_held: false,
            granted_count: 0,
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
        holder_count: 0,
        exclusive_held: false,
        granted_count: 0,
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
            holder_count: 0,
            exclusive_held: false,
            granted_count: 0,
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
        CpuSelectionPolicy::WithinEachLlc(PlacementPolicy::Consolidate),
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
        CpuSelectionPolicy::WithinEachLlc(PlacementPolicy::Spread { rotation: 1 }),
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
            holder_count: 0,
            exclusive_held: false,
            granted_count: 0,
        },
        LlcSnapshot {
            llc_idx: 1,
            holder_count: 5,
            exclusive_held: false,
            granted_count: 0,
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
            holder_count: 3,
            exclusive_held: false,
            granted_count: 0,
        },
        LlcSnapshot {
            llc_idx: 1,
            holder_count: 0,
            exclusive_held: false,
            granted_count: 0,
        },
        LlcSnapshot {
            llc_idx: 2,
            holder_count: 7,
            exclusive_held: false,
            granted_count: 0,
        },
        LlcSnapshot {
            llc_idx: 3,
            holder_count: 1,
            exclusive_held: false,
            granted_count: 0,
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
        permits: Vec::new(),
        mems: std::collections::BTreeSet::new(),
        locks: admission_protocol::Acquired::untracked(Vec::new()),
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
        permits: Vec::new(),
        mems: std::collections::BTreeSet::new(),
        locks: admission_protocol::Acquired::untracked(Vec::new()),
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
        permits: Vec::new(),
        mems: [0usize, 1].into_iter().collect(),
        locks: admission_protocol::Acquired::untracked(Vec::new()),
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
        permits: Vec::new(),
        mems: [0usize].into_iter().collect(),
        locks: admission_protocol::Acquired::untracked(Vec::new()),
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

/// A current-version HELD publication supplies consolidation occupancy without
/// any procfs scan. The planner must prefer that compatible SH-held LLC over a
/// fresh peer when the cap selects only one.
#[test]
fn acquire_llc_plan_consolidates_on_peer_held_llc() {
    let _llc_prefix = LlcLockPrefixGuard::new();
    let _allowed = AllowedCpusGuard::new(vec![0, 1]);
    let topo = HostTopology::new_for_tests(&[(vec![0], 0), (vec![1], 0)]);
    let target_lock = llc_lock_path(1);
    let physical = crate::flock::try_flock(&target_lock, FlockMode::Shared)
        .expect("open peer LLC lock")
        .expect("acquire peer LLC SH");
    let claim = admission_protocol::ClaimSet::with_modes(
        [1usize],
        std::iter::empty(),
        FlockMode::Shared,
        FlockMode::Shared,
    );
    let peer =
        admission_protocol::publish_acquired(&claim, physical).expect("publish peer LLC HELD");

    let test_topo = crate::topology::TestTopology::synthetic(2, 1);
    let cap = CpuCap::new(1).expect("cap=1 valid");
    let proc_reads_before = crate::flock::proc_locks::proc_locks_read_count_for_tests();
    let plan = acquire_llc_plan_interruptible(
        &topo,
        &test_topo,
        Some(cap),
        PlacementPolicy::Consolidate,
        false,
        None,
        None,
        None,
    )
    .expect("current-version SH occupancy remains compatible");
    assert_eq!(
        crate::flock::proc_locks::proc_locks_read_count_for_tests(),
        proc_reads_before,
        "successful current-version placement must not scan /proc/locks",
    );

    // Consolidation picked LLC 1 (the one with a holder) over
    // LLC 0 (fresh). The `holder_count DESC` ordering in
    // `plan_from_snapshots` makes this deterministic.
    assert_eq!(
        plan.locked_llcs,
        vec![1],
        "cap=1 with a published SH holder on LLC 1 must pick LLC 1 \
         (consolidation over fresh LLC 0); got {:?}",
        plan.locked_llcs,
    );

    drop(plan);
    drop(peer);
}

/// No-perf build planning must remain ownership-free, while the matching
/// run-time acquisition must own the exact LLC and CPU resources it returns.
/// This pins the phase boundary that keeps initrd/CAS preparation outside
/// topology admission without weakening run-time isolation.
#[test]
fn plan_only_holds_nothing_and_runtime_plan_holds_exact_resources() {
    let _llc_prefix = LlcLockPrefixGuard::new();
    let _cpu_prefix = CpuLockPrefixGuard::new();
    let _allowed = AllowedCpusGuard::new(vec![93640, 93641]);
    let topo = HostTopology::new_for_tests(&[(vec![93640], 0), (vec![93641], 0)]);
    let test_topo = crate::topology::TestTopology::synthetic(2, 1);
    let cap = CpuCap::new(1).expect("cap=1 valid");
    let registry_reads_before =
        crate::vmm::host_topology::protocol::aggregate_snapshot_read_count_for_tests();

    let build_plan = plan_llc_selection_only(
        &topo,
        &test_topo,
        Some(cap),
        PlacementPolicy::spread_for_process(),
    )
    .expect("build-time plan-only selection must succeed");
    assert!(
        build_plan.locks.is_empty(),
        "build-time no-perf planning must retain zero real resource fds",
    );
    assert_eq!(
        crate::vmm::host_topology::protocol::aggregate_snapshot_read_count_for_tests(),
        registry_reads_before,
        "shape-only planning must not consult the live admission registry",
    );
    let build_llc = build_plan.locked_llcs[0];
    let build_cpu = build_plan.cpus[0];
    assert!(
        !std::path::Path::new(&llc_lock_path(build_llc)).exists(),
        "shape-only planning must not materialize its selected LLC lockfile",
    );
    assert!(
        !std::path::Path::new(&cpu_lock_path(build_cpu)).exists(),
        "shape-only planning must not materialize its selected CPU lockfile",
    );
    let build_llc_probe = try_flock(llc_lock_path(build_llc), FlockMode::Exclusive)
        .expect("probe build-time LLC")
        .expect("build-time plan must not own its selected LLC");
    let build_cpu_probe = try_flock(cpu_lock_path(build_cpu), FlockMode::Exclusive)
        .expect("probe build-time CPU")
        .expect("build-time plan must not own its selected CPU");
    drop((build_llc_probe, build_cpu_probe, build_plan));

    let runtime_plan = acquire_llc_plan_interruptible(
        &topo,
        &test_topo,
        Some(cap),
        PlacementPolicy::spread_for_process(),
        false,
        None,
        None,
        None,
    )
    .expect("run-time acquisition must succeed on a fresh two-LLC host");
    assert!(
        !runtime_plan.locks.is_empty(),
        "run-time plan must carry its live resource fds",
    );
    assert!(
        try_flock(
            llc_lock_path(runtime_plan.locked_llcs[0]),
            FlockMode::Exclusive,
        )
        .expect("probe run-time LLC")
        .is_none(),
        "run-time plan must hold its selected LLC",
    );
    assert!(
        try_flock(cpu_lock_path(runtime_plan.cpus[0]), FlockMode::Exclusive)
            .expect("probe run-time CPU")
            .is_none(),
        "run-time plan must hold its selected CPU",
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
                // actual admission owners; the LlcPlan RAII
                // contract is exercised elsewhere).
                Ok(Some(Vec::new()))
            }
        },
    )
    .expect("retry on attempt 1 must succeed");
    // Attempt 1 produced locks (empty vec is fine — the plan
    // constructor accepts an empty admission-owner vector).
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
    let error = acquire_llc_plan_interruptible(
        &topo,
        &test_topo,
        CpuCap::new(1).ok(),
        PlacementPolicy::Consolidate,
        false,
        None,
        None,
        None,
    )
    .expect_err("the only LLC is held EX");
    let message = format!("{error:#}");
    assert!(
        !message.contains("registered reservation claims"),
        "a real flock holder is not a registry-claim-only fast failure: {message}",
    );
    assert!(
        message.contains("contended:") && message.contains("LLC 0") && message.contains("held"),
        "a live EX flock holder must be named by the per-needle probe bail: {message}",
    );
}

/// A waiting caller performs one real fast attempt, then uses the registry as
/// its retry mechanism. The seam always bounces while the real lockfile pool is
/// free, so the elected coordinator completes immediately.
#[test]
fn acquire_llc_plan_wait_phase_acquires_beyond_the_seam() {
    let _llc_prefix = LlcLockPrefixGuard::new();
    // Queue-registry CPU identities are real host CPU indices and therefore
    // live below `/sys/devices/system/cpu/possible`. The per-test lock prefix
    // already provides isolation, so use a valid low synthetic identity here.
    let _allowed = AllowedCpusGuard::new(vec![0]);
    let topo = synth_host_topo(&[(vec![0], 0)]);
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
fn plan_only_ignores_live_exclusive_holders() {
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
fn plan_only_cargo_test_mode_matches_runtime_full_allowed_cpuset() {
    let _env_guard = env_lock();
    let _cargo = EnvGuard::set(crate::KTSTR_CARGO_TEST_MODE_ENV, "1");
    let _llc_prefix = LlcLockPrefixGuard::new();
    let _allowed = AllowedCpusGuard::new(vec![93730, 93731]);
    let topo = synth_host_topo(&[(vec![93730], 0), (vec![93731], 1)]);
    let test_topo = crate::topology::TestTopology::synthetic(2, 1);
    let cap = CpuCap::new(1).expect("cap=1");

    let build_plan = plan_llc_selection_only(
        &topo,
        &test_topo,
        Some(cap),
        PlacementPolicy::Spread { rotation: 0 },
    )
    .expect("cargo-test build plan");
    let runtime_plan = acquire_llc_plan_interruptible(
        &topo,
        &test_topo,
        Some(cap),
        PlacementPolicy::Spread { rotation: 0 },
        true,
        None,
        None,
        None,
    )
    .expect("cargo-test runtime plan");

    assert_eq!(build_plan.cpus, vec![93730, 93731]);
    assert_eq!(
        build_plan.cpus, runtime_plan.cpus,
        "build budget and runtime mask must describe the same full allowed cpuset",
    );
    assert_eq!(build_plan.locked_llcs, runtime_plan.locked_llcs);
    assert!(build_plan.locks.is_empty());
    assert!(runtime_plan.locks.is_empty());
    for path in [
        llc_lock_path(0),
        llc_lock_path(1),
        cpu_lock_path(93730),
        cpu_lock_path(93731),
    ] {
        assert!(
            !std::path::Path::new(&path).exists(),
            "cargo-test shape/acquire bypass must not materialize {path}",
        );
    }
}

#[test]
fn registered_claim_fast_fails_without_acquire_or_diagnostic_enrichment() {
    let _llc_prefix = LlcLockPrefixGuard::new();
    let _allowed = AllowedCpusGuard::new(vec![0]);
    let topo = synth_host_topo(&[(vec![0], 0)]);
    let test_topo = crate::topology::TestTopology::synthetic(1, 1);
    let claim =
        admission_protocol::ClaimSet::new([0usize], std::iter::empty(), FlockMode::Exclusive);
    let watch = admission_protocol::ClaimSet::with_modes(
        [0usize],
        [0usize],
        FlockMode::Exclusive,
        FlockMode::Shared,
    );
    let coordinator =
        match admission_protocol::register_ticket_or_acquire(claim, watch, None, |_| {
            Ok::<Option<()>, anyhow::Error>(None)
        })
        .expect("register exclusive LLC claim")
        {
            admission_protocol::TicketWork::Coordinator(coordinator) => coordinator,
            admission_protocol::TicketWork::Acquired(_) => {
                panic!("fresh registry must elect a coordinator")
            }
        };
    let attempts = std::cell::Cell::new(0usize);
    let proc_reads_before = crate::flock::proc_locks::proc_locks_read_count_for_tests();
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
    // The wait=false bail diagnostic probes contended LLC lock files with a
    // non-blocking flock; it never walks host-global /proc/locks. (Here the
    // fence is registry-only, so no LLC is physically held and none is named.)
    assert_eq!(
        crate::flock::proc_locks::proc_locks_read_count_for_tests(),
        proc_reads_before,
        "the contention bail must not scan /proc/locks",
    );
    drop(coordinator);
}

/// The wait=false contention bail names the physically-held LLC via a
/// per-needle non-blocking flock probe, never a host-global `/proc/locks`
/// walk.
#[test]
fn contention_bail_names_held_llc_without_scanning_proc_locks() {
    let _llc_prefix = LlcLockPrefixGuard::new();
    let _allowed = AllowedCpusGuard::new(vec![0]);
    let topo = HostTopology::new_for_tests(&[(vec![0], 0)]);
    let test_topo = crate::topology::TestTopology::synthetic(1, 1);
    // A peer physically holds LLC 0's lock EX but registers no claim, so the
    // registry-based discover sees LLC 0 free and plans it; the physical
    // acquire then fails, yielding an LLC-0 contention marker the bail names.
    let held = crate::flock::try_flock(llc_lock_path(0), FlockMode::Exclusive)
        .expect("open LLC 0 lock")
        .expect("hold LLC 0 EX");
    let proc_reads_before = crate::flock::proc_locks::proc_locks_read_count_for_tests();
    let error = acquire_llc_plan_interruptible(
        &topo,
        &test_topo,
        Some(CpuCap::new(1).expect("cap=1 valid")),
        PlacementPolicy::Consolidate,
        false,
        None,
        None,
        None,
    )
    .expect_err("a physically held LLC 0 must fail the nonblocking plan");
    let reason = error.to_string();
    assert!(
        reason.contains("LLC 0") && reason.contains("held"),
        "the bail must name the physically-held LLC 0; got: {reason}",
    );
    assert_eq!(
        crate::flock::proc_locks::proc_locks_read_count_for_tests(),
        proc_reads_before,
        "the bail must probe the LLC lock files, not scan /proc/locks",
    );
    drop(held);
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
            holder_count: if idx == 3 { 5 } else { 0 },
            exclusive_held: false,
            granted_count: 0,
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

/// Elastic planners retain held-first Consolidate ordering while rotating the
/// fresh suffix. Distinct process-stable rotations must therefore fan out both
/// on a cold host and when a partially usable held LLC cannot cover the whole
/// target. Returned plans remain sorted in canonical acquisition order.
#[test]
fn elastic_fresh_rotations_fan_out_cold_prefix_and_warm_spill_suffix() {
    let topo = synth_host_topo(&[
        (vec![0], 0),
        (vec![1], 0),
        (vec![2], 0),
        (vec![3], 0),
        (vec![4], 0),
        (vec![5], 0),
        (vec![6], 0),
        (vec![7], 0),
    ]);
    let cold_snapshots: Vec<LlcSnapshot> = (0..8)
        .map(|llc_idx| LlcSnapshot {
            llc_idx,
            holder_count: 0,
            exclusive_held: false,
            granted_count: 0,
        })
        .collect();
    let allowed: std::collections::BTreeSet<usize> = (0..8).collect();
    let barrier = std::sync::Barrier::new(3);

    let (first, second) = std::thread::scope(|scope| {
        let first = scope.spawn(|| {
            barrier.wait();
            plan_from_snapshots_with_fresh_rotation(
                &cold_snapshots,
                2,
                &topo,
                &allowed,
                |_, _| 10,
                PlacementPolicy::Consolidate,
                Some(0),
            )
        });
        let second = scope.spawn(|| {
            barrier.wait();
            plan_from_snapshots_with_fresh_rotation(
                &cold_snapshots,
                2,
                &topo,
                &allowed,
                |_, _| 10,
                PlacementPolicy::Consolidate,
                Some(2),
            )
        });
        barrier.wait();
        (
            first.join().expect("first cold planner"),
            second.join().expect("second cold planner"),
        )
    });

    assert_eq!(first, vec![0, 1]);
    assert_eq!(second, vec![2, 3]);
    let first: std::collections::BTreeSet<_> = first.into_iter().collect();
    let second: std::collections::BTreeSet<_> = second.into_iter().collect();
    assert!(
        first.is_disjoint(&second),
        "distinct cold rotations must choose disjoint LLC prefixes",
    );

    let warm_topo = synth_host_topo(&[
        (vec![0, 1, 2, 3], 0),
        (vec![4, 5, 6, 7], 0),
        (vec![8, 9, 10, 11], 0),
        (vec![12, 13, 14, 15], 0),
        (vec![16, 17, 18, 19], 0),
    ]);
    let warm_snapshots: Vec<LlcSnapshot> = (0..5)
        .map(|llc_idx| LlcSnapshot {
            llc_idx,
            holder_count: if llc_idx == 4 { 3 } else { 0 },
            exclusive_held: false,
            granted_count: 0,
        })
        .collect();
    // The held LLC contributes only CPUs 16 and 17 inside this cpuset. A
    // six-CPU target must consolidate there first, then take one whole fresh
    // four-CPU LLC. Distinct rotations should choose distinct spill LLCs.
    let warm_allowed: std::collections::BTreeSet<usize> = (0..18).collect();
    let first = plan_from_snapshots_with_fresh_rotation(
        &warm_snapshots,
        6,
        &warm_topo,
        &warm_allowed,
        |_, _| 10,
        PlacementPolicy::Consolidate,
        Some(0),
    );
    let second = plan_from_snapshots_with_fresh_rotation(
        &warm_snapshots,
        6,
        &warm_topo,
        &warm_allowed,
        |_, _| 10,
        PlacementPolicy::Consolidate,
        Some(2),
    );
    assert_eq!(
        first,
        vec![0, 4],
        "held LLC must score first, with the final plan sorted for acquisition",
    );
    assert_eq!(
        second,
        vec![2, 4],
        "the held LLC stays selected while the fresh spill suffix rotates",
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
            holder_count: 0,
            exclusive_held: false,
            granted_count: 0,
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
            holder_count: 5,
            exclusive_held: false,
            granted_count: 0,
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

#[test]
fn elastic_sizing_uses_every_available_cpu_up_to_its_maximum() {
    let sizing = LlcPlanSizing::Elastic;
    assert_eq!(sizing.target_for_capacity(8, 0), None);
    assert_eq!(sizing.target_for_capacity(8, 1), Some(1));
    assert_eq!(sizing.target_for_capacity(8, 5), Some(5));
    assert_eq!(sizing.target_for_capacity(8, 8), Some(8));
    assert_eq!(sizing.target_for_capacity(8, 13), Some(8));
    assert_eq!(sizing.queued_target(8), 1);

    let exact = LlcPlanSizing::Exact;
    assert_eq!(exact.target_for_capacity(8, 7), None);
    assert_eq!(exact.target_for_capacity(8, 8), Some(8));
    assert_eq!(exact.target_for_capacity(8, 13), Some(8));
    assert_eq!(exact.queued_target(8), 8);
}

#[test]
fn cooperative_permits_admit_four_host_width_claims_and_reject_a_fifth() {
    let host_width = 8;
    let pool = AdmissionPermitPool::for_host(host_width);
    assert_eq!(pool.len(), host_width * COOPERATIVE_OVERSUBSCRIPTION);
    let mut held = std::collections::BTreeSet::new();

    for lane in 0..COOPERATIVE_OVERSUBSCRIPTION {
        let selection = select_admission_permits(
            PermitAdmission::Cooperative,
            &pool,
            host_width,
            host_width,
            lane * host_width,
            &[],
            |candidate| Ok(candidate.permits.is_disjoint(&held)),
            |_| Ok(false),
        )
        .unwrap()
        .expect("each of four host-width cooperative lanes must fit");
        assert_eq!(selection.permits.len(), host_width);
        assert!(selection.permits.iter().all(|permit| held.insert(*permit)));
    }

    assert_eq!(held.len(), pool.len());
    assert!(
        select_admission_permits(
            PermitAdmission::Cooperative,
            &pool,
            host_width,
            host_width,
            0,
            &[],
            |candidate| Ok(candidate.permits.is_disjoint(&held)),
            |_| Ok(false),
        )
        .unwrap()
        .is_none(),
        "a fifth host-width claim must wait until one of the four lanes releases",
    );
}

/// A zero-width request names no permits. The selection loop only stops on
/// `permits.len() == maximum`, so a `maximum` of 0 would otherwise run to the
/// end of the pool and name every ready permit in it — a whole-host admission
/// claim funding no CPUs.
#[test]
fn zero_width_permit_request_names_no_permits() {
    let host_width = 8;
    let pool = AdmissionPermitPool::for_host(host_width);
    assert_eq!(pool.len(), host_width * COOPERATIVE_OVERSUBSCRIPTION);

    let mut probes = 0usize;
    let selection = select_admission_permits(
        PermitAdmission::Cooperative,
        &pool,
        0,
        0,
        0,
        &[],
        |_| {
            probes += 1;
            Ok(true)
        },
        |_| Ok(false),
    )
    .expect("select a zero-width shape")
    .expect("a zero-width request with no floor is met by an empty selection");
    assert!(
        selection.permits.is_empty(),
        "zero-width selection must name no permits, got {:?}",
        selection.permits,
    );
    assert_eq!(
        selection.admission_class,
        admission_protocol::AdmissionClass::Ordinary,
    );
    assert_eq!(
        probes, 0,
        "a zero-width request must not probe pool permits at all",
    );

    assert!(
        select_admission_permits(
            PermitAdmission::Cooperative,
            &pool,
            0,
            1,
            0,
            &[],
            |_| Ok(true),
            |_| Ok(false),
        )
        .expect("select a zero-width shape under a one-permit floor")
        .is_none(),
        "an empty selection cannot clear a one-permit floor",
    );
}

/// A claim-fenced attempt selects no CPUs, and permit selection is skipped
/// for that zero-width plan. Both acquisition arms already refuse an empty
/// selection, so the skip changes no outcome — its only observable effect is
/// the registry snapshot the permit watch would otherwise take. Pin that:
/// `discover_registered_placement_states` takes the single aggregate snapshot
/// a fenced no-wait attempt is allowed, so the delta is exactly 1 (2 without
/// the short-circuit). Uses `acquire_llc_plan_interruptible` because it is
/// the entry point supplying `PermitAdmission::Cooperative`;
/// `acquire_llc_plan_with_acquire_fn` hardcodes `PermitAdmission::None` and
/// takes a different arm that never selected permits.
#[test]
fn claim_fenced_plan_takes_only_the_discover_registry_snapshot() {
    let _llc_prefix = LlcLockPrefixGuard::new();
    let _allowed = AllowedCpusGuard::new(vec![0]);
    let topo = synth_host_topo(&[(vec![0], 0)]);
    let test_topo = crate::topology::TestTopology::synthetic(1, 1);
    let claim =
        admission_protocol::ClaimSet::new([0usize], std::iter::empty(), FlockMode::Exclusive);
    let watch = admission_protocol::ClaimSet::with_modes(
        [0usize],
        [0usize],
        FlockMode::Exclusive,
        FlockMode::Shared,
    );
    let coordinator =
        match admission_protocol::register_ticket_or_acquire(claim, watch, None, |_| {
            Ok::<Option<()>, anyhow::Error>(None)
        })
        .expect("register exclusive LLC claim")
        {
            admission_protocol::TicketWork::Coordinator(coordinator) => coordinator,
            admission_protocol::TicketWork::Acquired(_) => {
                panic!("fresh registry must elect a coordinator")
            }
        };
    let registry_reads_before =
        crate::vmm::host_topology::protocol::aggregate_snapshot_read_count_for_tests();
    let error = acquire_llc_plan_interruptible(
        &topo,
        &test_topo,
        Some(CpuCap::new(1).expect("cap=1 valid")),
        PlacementPolicy::Consolidate,
        false,
        None,
        None,
        None,
    )
    .expect_err("a covering registered claim must fail fast");
    assert!(error.downcast_ref::<ResourceContention>().is_some());
    assert_eq!(
        crate::vmm::host_topology::protocol::aggregate_snapshot_read_count_for_tests()
            - registry_reads_before,
        1,
        "a claim-fenced attempt must read the aggregate exactly once (DISCOVER); \
         selecting permits for a plan that funds no CPUs costs a second read",
    );
    drop(coordinator);
}

#[test]
fn preparation_private_working_set_uses_two_chunks_and_preserves_conversion_headroom() {
    assert_eq!(MEMORY_PERMIT_CHUNK_MIB, 256);
    assert_eq!(PREPARATION_PRIVATE_WORKING_SET_MIB, 512);
    assert_eq!(preparation_memory_chunks(), 2);

    // Match the 376 GiB CI hosts which exposed the old 2 GiB-per-process
    // preparation bottleneck. Host reserve policy leaves 1,353 memory chunks;
    // a quarter can fund 169 two-chunk preparation owners instead of only 42
    // eight-chunk owners. CPU capacity remains a separate upper bound.
    let possible_width = 192usize;
    let (usable_mib, memory_chunks) = memory_capacity_from_total(376 * 1024);
    assert_eq!(usable_mib, 346_521);
    assert_eq!(memory_chunks, 1_353);
    let cpu_chunks = AdmissionPermitPool::for_host(possible_width).len();
    let slots = preparation_slot_capacity(memory_chunks, cpu_chunks, possible_width);
    assert_eq!(slots, 169);

    let preparation_chunks = slots * preparation_memory_chunks();
    assert!(
        preparation_chunks <= memory_chunks / PREPARATION_MEMORY_FRACTION,
        "a full preparation wave must never consume more than its quarter",
    );
    assert!(
        memory_chunks - preparation_chunks >= memory_chunks * 3 / 4,
        "at least three quarters of memory permits remain for guest conversion",
    );
}

/// The stacked permit pools (cooperative CPU, memory, preparation, build)
/// modeled from their pure sizing functions. Kept in lockstep with the real
/// `permit_namespace_end` by `permit_namespace_end_matches_the_stacked_pool_model`.
fn modeled_permit_namespace_end(possible_width: usize, memory_chunks: usize) -> usize {
    let cooperative = AdmissionPermitPool::for_host(possible_width).len();
    let preparation = preparation_slot_capacity(memory_chunks, cooperative, possible_width);
    let build = cooperative
        .saturating_mul(BUILD_RESERVED_PERCENT)
        .div_ceil(100)
        .max(1);
    memory_permit_base_for_possible_width(possible_width) + memory_chunks + preparation + build
}

#[test]
fn permit_namespace_end_matches_the_stacked_pool_model() {
    let Ok(end) = permit_namespace_end() else {
        // A host too small to fund one preparation slot has no namespace to
        // check; the registry then falls back to the CPU-only overprovision.
        return;
    };
    let (_, memory_chunks) = memory_capacity_from_total(host_mem_total_mib().unwrap());
    assert_eq!(
        end,
        modeled_permit_namespace_end(possible_cpu_width(), memory_chunks),
        "the registry creation width would silently stop covering the permit \
         namespace if the stacked pools grew without permit_namespace_end \
         following them",
    );
}

/// A ~400-possible-CPU host with 1 TiB of RAM is the smallest common server
/// shape whose widest permit claim overflows the registry's 4096-bit creation
/// floor. A CPU/LLC-only registrant can create the registry first, so its
/// creation width must already cover every same-host permit index — with the
/// old `max(2 * cpus, 4096)` overprovision the file came up 4096 bits wide
/// and every later run admission bailed against it.
#[test]
fn registry_creation_width_covers_the_permit_namespace_on_big_hosts() {
    let possible_width = 400;
    let (_, memory_chunks) = memory_capacity_from_total(1024 * 1024);
    let namespace_end = modeled_permit_namespace_end(possible_width, memory_chunks);
    let widest_claim_bits = possible_width + namespace_end;
    assert!(
        widest_claim_bits > 4096,
        "fixture host must overflow the creation floor, got {widest_claim_bits} bits",
    );
    assert!(
        admission_protocol::registry_overprovision_bits_for_tests(
            possible_width,
            Some(namespace_end),
        ) >= widest_claim_bits,
        "a CPU/LLC-only creator must size the registry for the full permit namespace",
    );
    assert_eq!(
        admission_protocol::registry_overprovision_bits_for_tests(possible_width, None),
        possible_width * 2,
        "an unresolvable permit namespace must fall back to the CPU-only overprovision",
    );
}

/// Preparation and run must derive the CPU-permit pool from the same host
/// width. Permit identities are one host-wide lockfile namespace, so a run
/// pool sized from the caller's (cgroup-narrowed) cpuset would stop short of
/// the identities this process's own preparation phase already holds.
#[test]
fn run_permit_pool_covers_the_preparation_pool_under_a_narrow_cpuset() {
    let _allowed = AllowedCpusGuard::new(vec![0, 1]);
    let preparation = AdmissionPermitPool::for_host(possible_cpu_width());
    let run = VmPermitPool::new_with_preparation(1, 256, None).expect("construct run permit pool");
    assert_eq!(
        run.cpu.all().collect::<Vec<_>>(),
        preparation.all().collect::<Vec<_>>(),
        "the run CPU-permit pool must be the preparation pool",
    );
    assert_eq!(run.cpu.general, preparation.general);
    assert_eq!(run.cpu.reserved, preparation.reserved);
}

#[test]
fn required_chunks_charges_touch_ceiling_not_sized_ram_on_wide_cells() {
    // The run claim charges `required_chunks(permit_memory)` where
    // `permit_memory` is the touch ceiling (max'd with declared), not the
    // sized RAM. On the 384 GB host the usable pool is ~1,380 chunks; a
    // 224-vCPU cell sized at the 64 MiB/vCPU floor (14.3 GiB) would charge
    // 56 chunks, capping concurrency at 24. Charging its ~2 GiB touch
    // ceiling instead charges 8 chunks and lifts concurrency to ~170.
    let (_usable_mib, memory_chunks) = memory_capacity_from_total(384 * 1024);
    let memory = MemoryPermitPool {
        permits: (0..memory_chunks).collect(),
        usable_mib: memory_chunks * MEMORY_PERMIT_CHUNK_MIB,
    };

    let sized_mib = 224 * 64; // 14336 MiB, the 64 MiB/vCPU floor
    let touch_mib = 2041; // touch ceiling for the wide cell

    let sized_chunks = memory.required_chunks(sized_mib).expect("sized demand");
    let touch_chunks = memory.required_chunks(touch_mib).expect("touch demand");
    assert_eq!(sized_chunks, 56, "sized RAM charges 56 chunks");
    assert_eq!(touch_chunks, 8, "touch ceiling charges 8 chunks");

    // Projected concurrency from the same usable pool.
    assert_eq!(memory_chunks / sized_chunks, 24);
    assert!(
        (168..=176).contains(&(memory_chunks / touch_chunks)),
        "touch-ceiling concurrency should be ~170 (got {})",
        memory_chunks / touch_chunks,
    );
}

#[test]
fn computed_guest_memory_replaces_fixed_preparation_weight() {
    let cpu = AdmissionPermitPool::for_host(1);
    let memory = MemoryPermitPool {
        permits: (100..108).collect(),
        usable_mib: 8 * MEMORY_PERMIT_CHUNK_MIB,
    };
    let preparation_owned = std::collections::BTreeSet::from([100usize, 101usize]);
    let preferred_memory = preparation_owned.iter().copied().collect::<Vec<_>>();
    let candidate_ready = |candidate: &admission_protocol::ClaimSet,
                           externally_blocked: Option<usize>| {
        let Some(external) = claim_without_owned_permits(candidate, &preparation_owned) else {
            return Ok(true);
        };
        Ok(externally_blocked.is_none_or(|permit| !external.permits.contains(&permit)))
    };

    // A 256 MiB guest needs one chunk, not both chunks in the fixed
    // preparation estimate. The surplus preparation OFD must not leak into
    // the run reservation.
    let small = select_vm_permits_grant_aware(
        PermitAdmission::Cooperative,
        &cpu,
        Some(&memory),
        1,
        1,
        memory.required_chunks(256).expect("256 MiB demand"),
        0,
        0,
        &[],
        &preferred_memory,
        |candidate| candidate_ready(candidate, None),
        |_| Ok(false),
    )
    .expect("select 256 MiB guest permits")
    .expect("256 MiB guest must fit");
    assert_eq!(small.memory_permits, vec![100]);

    // Conversely, a 2 GiB guest must acquire all eight actual chunks. Owning
    // the two preparation chunks is not permission to allocate the larger
    // guest: one unavailable external chunk keeps the conversion pending.
    let full_demand = memory.required_chunks(2048).expect("2 GiB demand");
    assert_eq!(full_demand, 8);
    assert!(
        select_vm_permits_grant_aware(
            PermitAdmission::Cooperative,
            &cpu,
            Some(&memory),
            1,
            1,
            full_demand,
            0,
            0,
            &[],
            &preferred_memory,
            |candidate| candidate_ready(candidate, Some(107)),
            |_| Ok(false),
        )
        .expect("probe partially available 2 GiB demand")
        .is_none(),
        "fixed preparation ownership cannot authorize an under-reserved guest",
    );

    let large = select_vm_permits_grant_aware(
        PermitAdmission::Cooperative,
        &cpu,
        Some(&memory),
        1,
        1,
        full_demand,
        0,
        0,
        &[],
        &preferred_memory,
        |candidate| candidate_ready(candidate, None),
        |_| Ok(false),
    )
    .expect("select fully available 2 GiB demand")
    .expect("2 GiB guest must fit once every actual chunk is available");
    assert_eq!(large.memory_permits.len(), full_demand);
    assert!(
        preparation_owned.is_subset(&large.memory_permits.iter().copied().collect()),
        "the atomic conversion should reuse preparation OFDs where possible",
    );
}

#[test]
fn owned_or_absent_permits_never_issue_an_empty_registry_query() {
    let pool = AdmissionPermitPool::for_host(1);
    let callback_count = std::cell::Cell::new(0usize);
    let selection = select_vm_permits_grant_aware(
        PermitAdmission::None,
        &pool,
        None,
        1,
        1,
        0,
        0,
        0,
        &[],
        &[],
        |candidate| {
            callback_count.set(callback_count.get() + 1);
            assert!(
                !candidate.is_empty(),
                "registry readiness callbacks require a real resource claim",
            );
            Ok(true)
        },
        |_| Ok(false),
    )
    .expect("select a permitless admission shape")
    .expect("permitless admission remains available");
    assert!(selection.all_permits().is_empty());
    assert_eq!(
        callback_count.get(),
        0,
        "a permitless selection has no registry resource delta to query",
    );

    let candidate = permit_only_claim(&PermitSelection {
        permits: vec![11],
        admission_class: admission_protocol::AdmissionClass::Ordinary,
    });
    assert!(
        claim_without_owned_permits(&candidate, &std::collections::BTreeSet::from([11usize]))
            .is_none(),
        "a fully reused preparation permit has no external contention claim",
    );
    let external = claim_without_owned_permits(
        &permit_only_claim(&PermitSelection {
            permits: vec![11, 12],
            admission_class: admission_protocol::AdmissionClass::Ordinary,
        }),
        &std::collections::BTreeSet::from([11usize]),
    )
    .expect("a non-owned permit remains externally visible");
    assert_eq!(
        external.permits,
        std::collections::BTreeSet::from([12usize])
    );
}

#[test]
fn build_permits_are_bounded_but_never_blocked_by_live_default_borrowers() {
    let cooperative = AdmissionPermitPool::for_host(8);
    let borrowed = cooperative
        .reserved
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    assert!(
        !borrowed.is_empty(),
        "the fixture must expose borrowable default capacity"
    );

    let build = AdmissionPermitPool::for_build_host(8).expect("construct build-only namespace");
    assert!(
        build
            .all()
            .all(|permit| !cooperative.general.contains(&permit)
                && !cooperative.reserved.contains(&permit)),
        "build permits must be disjoint from both general and borrowed default capacity",
    );
    let selection = select_admission_permits(
        PermitAdmission::Build,
        &build,
        1,
        1,
        0,
        &[],
        |candidate| Ok(candidate.permits.is_disjoint(&borrowed)),
        |_| Ok(false),
    )
    .expect("select one build permit")
    .expect("a live default borrower must not delay build admission");
    assert_eq!(
        selection.admission_class,
        admission_protocol::AdmissionClass::Build
    );
    assert_eq!(selection.permits.len(), 1);
}

#[test]
fn elastic_build_falls_back_to_one_cpu_when_every_build_permit_is_busy() {
    let pool = AdmissionPermitPool::for_build_host(4).expect("construct build-only namespace");
    let busy = pool.all().collect::<std::collections::BTreeSet<_>>();
    let selection = select_plan_permits_grant_aware(
        PermitAdmission::Build,
        LlcPlanSizing::Elastic,
        &pool,
        None,
        4,
        0,
        0,
        0,
        &[],
        &[],
        |candidate| Ok(candidate.permits.is_disjoint(&busy)),
        |_| Ok(false),
    )
    .expect("select an elastic build shape")
    .expect("busy build permits must retain serial forward progress");

    assert_eq!(selection.cpu_width, 1);
    assert!(
        selection.permits.all_permits().is_empty(),
        "the serial fallback must not queue behind the saturated permit pool",
    );
    assert_eq!(
        selection.permits.admission_class,
        admission_protocol::AdmissionClass::Build,
    );
}

#[test]
fn elastic_build_uses_the_one_free_build_permit_before_serial_fallback() {
    let pool = AdmissionPermitPool::for_build_host(4).expect("construct build-only namespace");
    let free = pool.all().next().expect("build pool is non-empty");
    let busy = pool
        .all()
        .filter(|permit| *permit != free)
        .collect::<std::collections::BTreeSet<_>>();
    let selection = select_plan_permits_grant_aware(
        PermitAdmission::Build,
        LlcPlanSizing::Elastic,
        &pool,
        None,
        4,
        0,
        0,
        0,
        &[],
        &[],
        |candidate| Ok(candidate.permits.is_disjoint(&busy)),
        |_| Ok(false),
    )
    .expect("select an elastic build shape")
    .expect("one free build permit must remain usable");

    assert_eq!(selection.cpu_width, 1);
    assert_eq!(selection.permits.cpu_permits, vec![free]);
    assert_eq!(selection.permits.all_permits(), vec![free]);
}

#[test]
fn elastic_build_acquires_serial_shared_plan_while_build_permits_are_saturated() {
    let _prefixes = LockPrefixesGuard::new();
    let _allowed = AllowedCpusGuard::new(vec![0, 1, 2, 3]);
    let topo = synth_host_topo(&[(vec![0], 0), (vec![1], 0), (vec![2], 0), (vec![3], 0)]);
    let test_topo = crate::topology::TestTopology::synthetic(4, 1);
    // Saturate the namespace the planner itself derives: build permits are
    // sized from the host possible width, not this test's allowed cpuset.
    let build_pool = AdmissionPermitPool::for_build_host(possible_cpu_width())
        .expect("construct build-only namespace");
    let busy_permits = build_pool.all().collect::<Vec<_>>();
    let busy_claim = resource_claim_with_permits(
        &[],
        LlcLockMode::Shared,
        &[],
        FlockMode::Shared,
        &busy_permits,
        admission_protocol::AdmissionClass::Build,
    );
    let busy_locks = busy_permits
        .iter()
        .map(|permit| {
            try_flock(permit_lock_path(*permit), FlockMode::Exclusive)
                .expect("open build-permit lock")
                .expect("reserve build permit")
        })
        .collect::<Vec<_>>();
    let busy_holder = admission_protocol::publish_acquired(&busy_claim, busy_locks)
        .expect("publish saturated build-permit pool");

    let plan =
        acquire_elastic_build_llc_plan(&topo, &test_topo, Some(CpuCap::new(4).unwrap()), None)
            .expect("soft build permits must not block SH-compatible serial progress");

    assert_eq!(plan.cpus.len(), 1);
    assert_eq!(plan.locked_llcs.len(), 1);
    assert!(plan.permits.is_empty());
    assert_eq!(make_jobs_for_plan(&plan), 1);
    assert_eq!(
        plan.locks.len(),
        plan.locked_llcs.len() + plan.cpus.len(),
        "the fallback retains only its physical shared ownership",
    );

    drop(plan);
    drop(busy_holder);
}

#[test]
fn cooperative_selection_borrows_reserved_capacity_when_general_capacity_is_exhausted() {
    let pool = AdmissionPermitPool::for_host(8);
    assert!(!pool.general.is_empty() && !pool.reserved.is_empty());
    let exhausted_general = pool
        .general
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    let selection = select_admission_permits(
        PermitAdmission::Cooperative,
        &pool,
        1,
        1,
        0,
        &[],
        |candidate| Ok(candidate.permits.is_disjoint(&exhausted_general)),
        |_| Ok(false),
    )
    .expect("select cooperative fallback capacity")
    .expect("reserved capacity must remain a soft default fallback");
    assert_eq!(
        selection.admission_class,
        admission_protocol::AdmissionClass::DefaultBorrow,
    );
    assert!(
        selection
            .permits
            .iter()
            .all(|permit| pool.reserved.contains(permit)),
    );
}

/// A heavily SH-held LLC can satisfy the complete build maximum, but must not
/// hide a disjoint idle LLC. Free-first eligibility is applied before LLC
/// consolidation, so the build uses the idle domain rather than reproducing
/// the busy-prefix/idle-host collapse.
#[test]
fn elastic_build_chooses_idle_cpus_before_a_sufficient_shared_llc() {
    let _prefixes = LockPrefixesGuard::new();
    let _allowed = AllowedCpusGuard::new(vec![0, 1, 2, 3]);
    let topo = synth_host_topo(&[(vec![0, 1], 0), (vec![2, 3], 0)]);
    let test_topo = crate::topology::TestTopology::synthetic(4, 1);
    let peer_claim = admission_protocol::ClaimSet::with_modes(
        [0usize],
        [0usize, 1],
        FlockMode::Shared,
        FlockMode::Shared,
    );
    let peer_physical = [llc_lock_path(0), cpu_lock_path(0), cpu_lock_path(1)]
        .into_iter()
        .map(|path| {
            try_flock(path, FlockMode::Shared)
                .expect("open peer SH resource")
                .expect("acquire peer SH resource")
        })
        .collect::<Vec<_>>();
    let peer = admission_protocol::publish_acquired(&peer_claim, peer_physical)
        .expect("publish concurrent SH holder");

    let plan =
        acquire_elastic_build_llc_plan(&topo, &test_topo, Some(CpuCap::new(2).unwrap()), None)
            .expect("two unshared CPUs must admit a two-CPU elastic build");

    assert_eq!(
        plan.locked_llcs,
        vec![1],
        "the shared LLC must be absent while the idle LLC carries the maximum",
    );
    assert_eq!(
        plan.cpus
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>(),
        [2usize, 3].into_iter().collect(),
        "admitted CPUs must come entirely from the disjoint idle LLC",
    );
    drop(plan);
    drop(peer);
}

#[test]
fn elastic_build_width_prefers_unshared_capacity_then_falls_back_to_shared() {
    let topo = synth_host_topo(&[(vec![0, 1], 0), (vec![2, 3], 0)]);
    let snapshots = vec![
        LlcSnapshot {
            llc_idx: 0,
            holder_count: 1,
            exclusive_held: false,
            granted_count: 0,
        },
        LlcSnapshot {
            llc_idx: 1,
            holder_count: 1,
            exclusive_held: false,
            granted_count: 0,
        },
    ];
    let compatible = (0usize..4).collect::<std::collections::BTreeSet<_>>();
    let mut states = (0usize..4)
        .map(|cpu| (cpu, CpuPlacementState::default()))
        .collect::<std::collections::BTreeMap<_, _>>();
    states.get_mut(&0).unwrap().other_holders = 2;
    states.get_mut(&1).unwrap().other_holders = 1;

    let unshared = live_cpu_capacity(
        LlcPlanSizing::Elastic,
        4,
        &snapshots,
        &topo,
        &compatible,
        &states,
    )
    .expect("two unshared CPUs provide live capacity");
    assert_eq!(
        unshared.target, 2,
        "two unshared CPUs contract a four-CPU elastic maximum to two",
    );
    assert_eq!(
        unshared.eligible,
        [2usize, 3].into_iter().collect(),
        "LLC planning and CPU selection must both see only unshared CPUs",
    );

    for state in states.values_mut() {
        state.other_holders = 1;
    }
    let fallback = live_cpu_capacity(
        LlcPlanSizing::Elastic,
        4,
        &snapshots,
        &topo,
        &compatible,
        &states,
    )
    .expect("compatible shared CPUs provide fallback capacity");
    assert_eq!(
        fallback.target, 4,
        "a fully occupied cooperative host retains the SH-compatible fallback",
    );
    assert_eq!(
        fallback.eligible, compatible,
        "zero unshared capacity restores the complete SH-compatible set",
    );
}

/// A performance-shaped EX claim remains a hard fence even though elastic
/// builds use SH locks and can overlap default/no-perf/build peers.
#[test]
fn elastic_build_excludes_a_hard_exclusive_llc() {
    let _prefixes = LockPrefixesGuard::new();
    let _allowed = AllowedCpusGuard::new(vec![0, 1, 2, 3]);
    let topo = synth_host_topo(&[(vec![0, 1], 0), (vec![2, 3], 0)]);
    let test_topo = crate::topology::TestTopology::synthetic(4, 1);
    let peer_claim = admission_protocol::ClaimSet::with_modes(
        [0usize],
        [0usize],
        FlockMode::Exclusive,
        FlockMode::Exclusive,
    );
    let peer_physical = [llc_lock_path(0), cpu_lock_path(0)]
        .into_iter()
        .map(|path| {
            try_flock(path, FlockMode::Exclusive)
                .expect("open peer EX resource")
                .expect("acquire peer EX resource")
        })
        .collect::<Vec<_>>();
    let peer = admission_protocol::publish_acquired(&peer_claim, peer_physical)
        .expect("publish hard-exclusive holder");

    let plan =
        acquire_elastic_build_llc_plan(&topo, &test_topo, Some(CpuCap::new(1).unwrap()), None)
            .expect("disjoint compatible LLC must admit the elastic build");
    assert_eq!(
        plan.locked_llcs,
        vec![1],
        "the build must not enter the hard-exclusive LLC",
    );
    assert!(
        plan.cpus.iter().all(|cpu| [2usize, 3].contains(cpu)),
        "the build CPU must come from the disjoint LLC: {:?}",
        plan.cpus,
    );
    drop(plan);
    drop(peer);
}

/// Once every compatible CPU already has a SH holder, an elastic build must
/// still start by sharing rather than manufacturing an EX-like serialization
/// point. The LLC footprint remains consolidated to one domain.
#[test]
fn elastic_build_uses_shared_capacity_when_no_cpu_is_unshared() {
    let _prefixes = LockPrefixesGuard::new();
    let _allowed = AllowedCpusGuard::new(vec![0, 1, 2, 3]);
    let topo = synth_host_topo(&[(vec![0, 1], 0), (vec![2, 3], 0)]);
    let test_topo = crate::topology::TestTopology::synthetic(4, 1);
    let peer_claim = admission_protocol::ClaimSet::with_modes(
        [0usize, 1],
        [0usize, 1, 2, 3],
        FlockMode::Shared,
        FlockMode::Shared,
    );
    let peer_physical = [
        llc_lock_path(0),
        llc_lock_path(1),
        cpu_lock_path(0),
        cpu_lock_path(1),
        cpu_lock_path(2),
        cpu_lock_path(3),
    ]
    .into_iter()
    .map(|path| {
        try_flock(path, FlockMode::Shared)
            .expect("open peer SH resource")
            .expect("acquire peer SH resource")
    })
    .collect::<Vec<_>>();
    let peer = admission_protocol::publish_acquired(&peer_claim, peer_physical)
        .expect("publish whole-host cooperative holder");

    let plan =
        acquire_elastic_build_llc_plan(&topo, &test_topo, Some(CpuCap::new(2).unwrap()), None)
            .expect("SH fallback must admit a fully occupied cooperative host");
    assert_eq!(
        plan.locked_llcs,
        vec![0],
        "equal held LLCs consolidate deterministically onto the first domain",
    );
    assert_eq!(
        plan.cpus
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>(),
        [0usize, 1].into_iter().collect(),
        "fallback CPUs must remain inside the consolidated footprint",
    );
    drop(plan);
    drop(peer);
}

/// An elastic build must start immediately on every currently compatible CPU
/// instead of joining the fixed-budget queue. Two exact VM-shaped holders own
/// CPUs/LLCs 0 and 1; a 3-CPU build therefore contracts to the two free CPUs,
/// carries matching SH locks plus build permits, and reports `-j2`.
#[test]
fn elastic_build_plan_takes_largest_immediately_available_subset() {
    let _prefixes = LockPrefixesGuard::new();
    let _allowed = AllowedCpusGuard::new(vec![0, 1, 2, 3]);
    let topo = synth_host_topo(&[(vec![0], 0), (vec![1], 0), (vec![2], 0), (vec![3], 0)]);
    let test_topo = crate::topology::TestTopology::synthetic(4, 1);

    let exact_holders: Vec<_> = [0usize, 1]
        .into_iter()
        .flat_map(|index| [llc_lock_path(index), cpu_lock_path(index)])
        .map(|path| {
            try_flock(&path, FlockMode::Exclusive)
                .expect("open exact-holder lock")
                .expect("reserve exact-holder resource")
        })
        .collect();

    let plan =
        acquire_elastic_build_llc_plan(&topo, &test_topo, Some(CpuCap::new(3).unwrap()), None)
            .expect("two free CPUs must start an elastic three-CPU-max build");
    assert_eq!(plan.locked_llcs, vec![2, 3]);
    assert_eq!(plan.cpus, vec![2, 3]);
    assert_eq!(
        plan.permits.len(),
        plan.cpus.len(),
        "elastic width is bounded by one build permit per admitted CPU",
    );
    assert_eq!(
        plan.locks.len(),
        plan.locked_llcs.len() + plan.cpus.len() + plan.permits.len(),
        "the elastic build retains its LLC, CPU, and build-permit ownership",
    );
    assert_eq!(
        make_jobs_for_plan(&plan),
        2,
        "build parallelism must follow the contracted reservation",
    );
    for index in &plan.cpus {
        assert!(
            try_flock(cpu_lock_path(*index), FlockMode::Exclusive)
                .expect("probe returned CPU lock")
                .is_none(),
            "the elastic plan must retain its exact SH CPU reservation",
        );
    }

    drop(plan);
    drop(exact_holders);
    let full =
        acquire_elastic_build_llc_plan(&topo, &test_topo, Some(CpuCap::new(3).unwrap()), None)
            .expect("idle capacity must restore the configured elastic maximum");
    assert_eq!(full.cpus.len(), 3);
    assert_eq!(make_jobs_for_plan(&full), 3);
}

/// With every resource initially owned by exact VM-shaped EX holders, an
/// elastic build queues. Releasing two of four CPUs must wake it into a 2-CPU
/// plan even though its configured maximum is three; it must neither wait for
/// a fixed three-CPU shape nor overlap the two holders that remain.
#[test]
fn elastic_build_wait_replans_to_largest_partial_release() {
    let _prefixes = LockPrefixesGuard::new_real_wake();
    let _allowed = AllowedCpusGuard::new(vec![0, 1, 2, 3]);
    let topo = synth_host_topo(&[(vec![0], 0), (vec![1], 0), (vec![2], 0), (vec![3], 0)]);
    let test_topo = crate::topology::TestTopology::synthetic(4, 1);

    let mut exact_holders = Vec::new();
    for index in 0usize..4 {
        let llc = try_flock(llc_lock_path(index), FlockMode::Exclusive)
            .expect("open exact-holder LLC lock")
            .expect("reserve exact-holder LLC");
        let cpu = try_flock(cpu_lock_path(index), FlockMode::Exclusive)
            .expect("open exact-holder CPU lock")
            .expect("reserve exact-holder CPU");
        exact_holders.push((llc, cpu));
    }
    let mut released = Some(exact_holders.split_off(2));
    let retained = exact_holders;
    let proc_reads_before = crate::flock::proc_locks::proc_locks_read_count_for_tests();
    let plan = acquire_elastic_build_llc_plan_with_coordinator_step_hook(
        &topo,
        &test_topo,
        Some(CpuCap::new(3).unwrap()),
        None,
        || {
            // Release exactly two CPUs only after the waiter has crossed
            // registration and is executing its first coordinator replan.
            // This replaces both the guessed 250 ms startup delay and the
            // five-second "drop everything" escape hatch: success while
            // `retained` is still live proves the partial release itself
            // satisfied the elastic request.
            drop(released.take());
        },
    )
    .expect("the two-CPU release must satisfy an elastic build");
    assert_eq!(plan.locked_llcs, vec![2, 3]);
    assert_eq!(plan.cpus, vec![2, 3]);
    assert_eq!(make_jobs_for_plan(&plan), 2);
    assert_eq!(
        crate::flock::proc_locks::proc_locks_read_count_for_tests(),
        proc_reads_before,
        "queued admission and coordinator wakes must not scan /proc/locks",
    );

    drop(plan);
    drop(retained);
}

/// A direct (non-elastic, exact-width) kernel build must WAIT for its fixed
/// width through the host queue instead of bailing after the four TOCTOU
/// retries — the give-up-instead-of-wait regression that failed a CI lane
/// (`acquire_llc_plan: could not reserve N CPU(s) after 4 attempts`). Every
/// LLC/CPU is held so the fast phase cannot satisfy the request; with the
/// wait routing the build registers and re-plans, and the release performed
/// during its first coordinator step — while other holders stay live — is what
/// satisfies it. Success without a bail proves the queue, not a retry timeout.
#[test]
fn direct_build_waits_for_exact_width_on_release_instead_of_bailing() {
    let _prefixes = LockPrefixesGuard::new_real_wake();
    let _allowed = AllowedCpusGuard::new(vec![0, 1, 2, 3]);
    let topo = synth_host_topo(&[(vec![0], 0), (vec![1], 0), (vec![2], 0), (vec![3], 0)]);
    let test_topo = crate::topology::TestTopology::synthetic(4, 1);

    let mut holders = Vec::new();
    for index in 0usize..4 {
        let llc = try_flock(llc_lock_path(index), FlockMode::Exclusive)
            .expect("open holder LLC lock")
            .expect("reserve holder LLC");
        let cpu = try_flock(cpu_lock_path(index), FlockMode::Exclusive)
            .expect("open holder CPU lock")
            .expect("reserve holder CPU");
        holders.push((llc, cpu));
    }
    // An exact designation names a complete CANONICAL set even while busy
    // (Consolidate -> the lowest two LLCs, 0 and 1), so release exactly those
    // after the waiter has crossed registration and is executing its first
    // coordinator replan; LLCs 2 and 3 stay held, so success proves the release
    // of the designated width satisfied the queued exact build, not a timeout.
    let retained = holders.split_off(2);
    let mut released = Some(holders);
    let plan = acquire_build_llc_plan_with_coordinator_step_hook(
        &topo,
        &test_topo,
        Some(CpuCap::new(2).unwrap()),
        None,
        || {
            drop(released.take());
        },
    )
    .expect("a two-CPU release must satisfy a waiting exact build, never a bail");
    assert_eq!(
        plan.cpus,
        vec![0, 1],
        "the exact build takes its canonical width"
    );
    assert_eq!(make_jobs_for_plan(&plan), 2);

    drop(plan);
    drop(retained);
}

/// N direct kernel builds contending for the SAME exact width must all
/// eventually reserve and complete by serializing through the host queue —
/// none may give up the lane. Each wants the whole two-CPU LLC, so only one
/// can hold at a time; with the wait routing the losers queue and each is
/// granted in turn as the current holder releases. Before the fix they raced
/// four TOCTOU attempts and failed a lane under exactly this overlap.
#[test]
fn many_contenders_for_one_exact_build_width_all_reserve_without_bailing() {
    use std::sync::{Arc, Barrier};
    let _prefixes = LockPrefixesGuard::new_real_wake();
    let _allowed = AllowedCpusGuard::new(vec![0, 1]);
    let llc_prefix = LLC_LOCK_PREFIX_OVERRIDE
        .with(|slot| slot.borrow().clone())
        .expect("parent LLC prefix");
    let cpu_prefix = CPU_LOCK_PREFIX_OVERRIDE
        .with(|slot| slot.borrow().clone())
        .expect("parent CPU prefix");

    const CONTENDERS: usize = 4;
    let barrier = Arc::new(Barrier::new(CONTENDERS));
    let handles: Vec<_> = (0..CONTENDERS)
        .map(|_| {
            let llc_prefix = llc_prefix.clone();
            let cpu_prefix = cpu_prefix.clone();
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                // Thread-local test overrides do not inherit across spawn.
                LLC_LOCK_PREFIX_OVERRIDE.with(|slot| *slot.borrow_mut() = Some(llc_prefix));
                CPU_LOCK_PREFIX_OVERRIDE.with(|slot| *slot.borrow_mut() = Some(cpu_prefix));
                ALLOWED_CPUS_OVERRIDE.with(|slot| *slot.borrow_mut() = Some(vec![0, 1]));
                let topo = synth_host_topo(&[(vec![0, 1], 0)]);
                let test_topo = crate::topology::TestTopology::synthetic(2, 1);
                // Start together to maximize the overlap the fast phase cannot
                // satisfy — the exact shape that previously bailed a lane.
                barrier.wait();
                let plan = acquire_build_llc_plan(
                    &topo,
                    &test_topo,
                    Some(CpuCap::new(2).unwrap()),
                    true,
                    None,
                )
                .expect("every contender must wait and reserve, never bail");
                assert_eq!(plan.cpus, vec![0, 1]);
                // Hold briefly, then release so the next queued contender is
                // granted; the drop releases the LLC/CPU flocks and wakes the
                // coordinator.
                std::thread::sleep(std::time::Duration::from_millis(5));
                drop(plan);
            })
        })
        .collect();
    for handle in handles {
        handle
            .join()
            .expect("every contender must reserve and complete without a bail");
    }
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
    let err = plan_llc_selection_only(&topo, &test_topo, None, PlacementPolicy::Consolidate)
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
            holder_count: 0,
            exclusive_held: false,
            granted_count: 0,
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
    let plan = acquire_llc_plan_interruptible(
        &topo,
        &test_topo,
        Some(cap),
        PlacementPolicy::Consolidate,
        false,
        None,
        None,
        None,
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
            holder_count: 0,
            exclusive_held: false,
            granted_count: 0,
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
    let plan = acquire_llc_plan_interruptible(
        &topo,
        &test_topo,
        Some(cap),
        PlacementPolicy::Consolidate,
        false,
        None,
        None,
        None,
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
            holder_count,
            exclusive_held: false,
            granted_count: 0,
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
        admission_protocol::AdmissionClass::Ordinary,
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
        admission_protocol::AdmissionClass::Ordinary,
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
        admission_protocol::AdmissionClass::Ordinary,
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

#[test]
fn physical_replan_candidates_preserve_the_build_watch_class() {
    let class = admission_protocol::AdmissionClass::Build;
    let cpu = physical_candidate_for_watch(
        std::iter::empty(),
        [7usize],
        FlockMode::Shared,
        FlockMode::Shared,
        class,
    );
    let llc = physical_candidate_for_watch(
        [3usize],
        std::iter::empty(),
        FlockMode::Shared,
        FlockMode::Shared,
        class,
    );
    assert_eq!(cpu.admission_class, class);
    assert_eq!(llc.admission_class, class);

    let topo = synth_host_topo(&[(vec![0], 0)]);
    let allowed = std::collections::BTreeSet::from([0usize]);
    let mut observed = Vec::new();
    avoid_preceding_claims_when_possible(
        &spread_snapshots(&[0]),
        1,
        &topo,
        &allowed,
        class,
        |candidate| {
            observed.push(candidate.admission_class);
            Ok(false)
        },
    )
    .expect("filter build-watch predecessor claims");
    assert_eq!(observed, vec![class]);
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
    let matched = match_distinct(&edges, &[0, 1, 2])
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

#[test]
fn physical_performance_reserve_is_deterministic_and_numa_balanced() {
    let host = synth_host_topo(&[((0..32).collect(), 0), ((32..64).collect(), 1)]);
    let allowed = (0..64).collect::<Vec<_>>();
    let first = host.performance_reserved_cpus(&allowed);
    let second = host.performance_reserved_cpus(&allowed);

    assert_eq!(
        first, second,
        "the reserve cannot rotate by process or call"
    );
    assert_eq!(first.len(), 20, "30% of 64 CPUs rounds up to 20");
    let per_node = first.iter().fold([0usize; 2], |mut counts, cpu| {
        counts[host.cpu_to_node[cpu]] += 1;
        counts
    });
    assert_eq!(
        per_node,
        [10, 10],
        "CPU-grain reserve selection must round-robin NUMA nodes",
    );
}

#[test]
fn physical_performance_reserve_keeps_modest_llcs_indivisible() {
    let host = synth_host_topo(&[
        ((0..4).collect(), 0),
        ((4..8).collect(), 1),
        ((8..12).collect(), 0),
        ((12..16).collect(), 1),
    ]);
    let allowed = (0..16).collect::<Vec<_>>();
    let reserved = host.performance_reserved_cpus(&allowed);

    assert_eq!(reserved.len(), 8, "whole-LLC rounding may exceed 30%");
    for group in &host.llc_groups {
        let selected = group
            .cpus
            .iter()
            .filter(|cpu| reserved.contains(cpu))
            .count();
        assert!(
            selected == 0 || selected == group.cpus.len(),
            "a modest LLC reserve must contain either all or none of its CPUs",
        );
    }
    let node0 = reserved
        .iter()
        .filter(|cpu| host.cpu_to_node[cpu] == 0)
        .count();
    let node1 = reserved.len() - node0;
    assert_eq!((node0, node1), (4, 4));
}

#[test]
fn every_performance_claim_mode_preserves_one_physical_reserve() {
    // Ordering the huge LLC first makes the 30% reserve a CPU-grain subset of
    // that domain. Small-LLC candidates remain whole-EX while huge-LLC
    // candidates remain SH+CPU-EX, exercising both final footprint shapes.
    let host = synth_host_topo(&[
        ((0..64).collect(), 0),
        ((64..72).collect(), 0),
        ((72..80).collect(), 0),
        ((80..88).collect(), 0),
        ((88..96).collect(), 0),
    ]);
    let allowed = (0..96).collect::<Vec<_>>();
    let reserved = host.performance_reserved_cpus(&allowed);
    let candidates = host
        .performance_pinning_candidates_for_cpus(&Topology::new(1, 1, 2, 1), &allowed)
        .expect("both whole-LLC and grain performance placements must remain");

    assert_eq!(reserved.len(), 29);
    assert!(
        candidates
            .iter()
            .any(|candidate| candidate.llc_mode == LlcLockMode::Shared),
        "the huge LLC must expose CPU-grain performance candidates",
    );
    assert!(
        candidates
            .iter()
            .any(|candidate| candidate.llc_mode == LlcLockMode::Exclusive),
        "the modest LLCs must expose whole-domain performance candidates",
    );
    assert!(candidates.iter().all(|candidate| {
        candidate
            .cpu_reservations
            .iter()
            .all(|cpu| !reserved.contains(cpu))
    }));

    let defaults = host
        .default_pinning_candidates_for_cpus(&Topology::new(1, 1, 1, 1), &allowed)
        .expect("default pinning still sees the complete allowed host");
    assert!(
        defaults.iter().any(|candidate| {
            candidate
                .cpu_reservations
                .iter()
                .any(|cpu| reserved.contains(cpu))
        }),
        "the reserve is retained from performance, not forbidden to default work",
    );
}

#[test]
fn indivisible_small_host_preserves_performance_instead_of_reserving_it_all() {
    let host = synth_host_topo(&[((0..4).collect(), 0)]);
    let allowed = (0..4).collect::<Vec<_>>();
    assert!(host.performance_reserved_cpus(&allowed).is_empty());
    assert!(
        host.performance_pinning_candidates_for_cpus(&Topology::new(1, 1, 2, 1), &allowed)
            .is_ok(),
        "an indivisible host LLC must retain its existing performance semantics",
    );
}

#[test]
fn performance_permit_pool_cannot_select_the_reserved_suffix() {
    let ordinary = AdmissionPermitPool::for_host(8);
    let performance = AdmissionPermitPool::for_performance_host(8);
    assert!(!ordinary.reserved.is_empty());
    assert_eq!(performance.general, ordinary.general);
    assert!(performance.reserved.is_empty());

    let selection = select_admission_permits(
        PermitAdmission::Cooperative,
        &performance,
        performance.general.len(),
        1,
        0,
        &ordinary.reserved,
        |_| Ok(true),
        |_| Ok(false),
    )
    .unwrap()
    .expect("general performance permits remain selectable");
    assert_eq!(selection.permits, performance.general);
    assert_eq!(
        selection.admission_class,
        admission_protocol::AdmissionClass::Ordinary,
    );
}

// ---------------------------------------------------------------
// Grant-aware planning bias — in-flight grant charge is a
// subordinate preference, never a fence or a primary key
// ---------------------------------------------------------------

/// CPU rank: among CPUs whose HELD holder count ties, the in-flight
/// grant charge decides with a fixed ASC sign under BOTH policies,
/// and the HELD-only primary key keeps its existing per-policy sign.
#[test]
fn cpu_rank_prefers_free_over_granted_within_equal_holders() {
    let topo = synth_host_topo(&[(vec![0, 1, 2, 3], 0)]);
    let eligible: std::collections::BTreeSet<usize> = (0..4).collect();
    let states: std::collections::BTreeMap<usize, CpuPlacementState> = [
        // cpu0: free but grant-charged once.
        (
            0,
            CpuPlacementState {
                exclusive_held: false,
                other_holders: 0,
                granted_holders: 1,
            },
        ),
        // cpu1: completely free.
        (
            1,
            CpuPlacementState {
                exclusive_held: false,
                other_holders: 0,
                granted_holders: 0,
            },
        ),
        // cpu2: free but grant-charged twice.
        (
            2,
            CpuPlacementState {
                exclusive_held: false,
                other_holders: 0,
                granted_holders: 2,
            },
        ),
        // cpu3: one live HELD holder, no grant charge.
        (
            3,
            CpuPlacementState {
                exclusive_held: false,
                other_holders: 1,
                granted_holders: 0,
            },
        ),
    ]
    .into_iter()
    .collect();
    let (spread, _) = materialize_plan_cpus(
        &[0],
        &topo,
        &eligible,
        &states,
        3,
        CpuSelectionPolicy::WithinEachLlc(PlacementPolicy::Spread { rotation: 0 }),
    )
    .expect("spread materialization must satisfy the target");
    assert_eq!(
        spread,
        vec![1, 0, 2],
        "Spread: least-held first, then grant charge ASC among the free CPUs",
    );
    let (consolidate, _) = materialize_plan_cpus(
        &[0],
        &topo,
        &eligible,
        &states,
        3,
        CpuSelectionPolicy::WithinEachLlc(PlacementPolicy::Consolidate),
    )
    .expect("consolidate materialization must satisfy the target");
    assert_eq!(
        consolidate,
        vec![3, 1, 0],
        "Consolidate: HELD holders keep the DESC primary key (grant charge \
         must not outrank them), then grant charge ASC among the free CPUs",
    );
}

/// LLC rank: a granted-only LLC stays in the FRESH Consolidate
/// partition (grant charge never joins the `holder_count > 0`
/// primary partition key), and within each partition the charge is
/// only a subordinate ASC preference.
#[test]
fn consolidate_llc_ordering_ignores_grant_charge_in_primary_key() {
    let topo = synth_host_topo(&[(vec![0], 0), (vec![1], 0), (vec![2], 0)]);
    let allowed: std::collections::BTreeSet<usize> = (0..3).collect();
    // llc0: held by a peer. llc1: heavily grant-charged, no holder.
    // llc2: completely free.
    let snapshots = vec![
        LlcSnapshot {
            llc_idx: 0,
            holder_count: 1,
            exclusive_held: false,
            granted_count: 0,
        },
        LlcSnapshot {
            llc_idx: 1,
            holder_count: 0,
            exclusive_held: false,
            granted_count: 5,
        },
        LlcSnapshot {
            llc_idx: 2,
            holder_count: 0,
            exclusive_held: false,
            granted_count: 0,
        },
    ];
    // Target 1: the held LLC must still win — a granted-only LLC must
    // not displace the consolidation partition.
    assert_eq!(
        plan_from_snapshots(
            &snapshots,
            1,
            &topo,
            &allowed,
            |_, _| 10,
            PlacementPolicy::Consolidate,
        ),
        vec![0],
        "grant charge must not evict the peer-held consolidation winner",
    );
    // Target 2: within the fresh partition the grant-free LLC wins.
    assert_eq!(
        plan_from_snapshots(
            &snapshots,
            2,
            &topo,
            &allowed,
            |_, _| 10,
            PlacementPolicy::Consolidate,
        ),
        vec![0, 2],
        "the fresh partition must prefer the grant-free LLC",
    );
    // Spread: holder count stays primary (llc1/llc2 before llc0), and
    // the grant charge breaks the tie toward llc2.
    assert_eq!(
        plan_from_snapshots(
            &snapshots,
            1,
            &topo,
            &allowed,
            |_, _| 10,
            PlacementPolicy::Spread { rotation: 0 },
        ),
        vec![2],
        "Spread must prefer the grant-free LLC among the least-held",
    );
}

/// Two-tier permit selection: the grant-aware tier avoids charged
/// permits while alternatives exist, and the grant-blind fallback
/// fires — returning charged permits instead of `None` — when the
/// charge covers the whole pool. The fallback is the livelock guard:
/// a senior must be able to publish an overlapping claim to trigger
/// the scan's ticket-order revoke.
#[test]
fn permit_selection_prefers_grant_free_and_falls_back() {
    let pool = AdmissionPermitPool::for_host(2);
    let all: Vec<usize> = pool.all().collect();
    assert!(all.len() >= 3, "fixture needs at least 3 permits");
    let charged: std::collections::BTreeSet<usize> = [all[0]].into_iter().collect();
    let grant_conflicts = |charged: &std::collections::BTreeSet<usize>,
                           candidate: &admission_protocol::ClaimSet| {
        Ok(candidate
            .permits
            .iter()
            .any(|permit| charged.contains(permit)))
    };
    let avoided = select_plan_permits_grant_aware(
        PermitAdmission::Cooperative,
        LlcPlanSizing::Exact,
        &pool,
        None,
        2,
        0,
        0,
        0,
        &[],
        &[],
        |_| Ok(true),
        |candidate| grant_conflicts(&charged, candidate),
    )
    .expect("grant-aware permit selection")
    .expect("a mostly-free pool must satisfy the request");
    assert_eq!(avoided.permits.cpu_permits.len(), 2);
    assert!(
        !avoided
            .permits
            .cpu_permits
            .iter()
            .any(|permit| charged.contains(permit)),
        "the grant-aware tier must avoid the charged permit while \
         alternatives exist: {:?}",
        avoided.permits.cpu_permits,
    );
    let all_charged: std::collections::BTreeSet<usize> = all.iter().copied().collect();
    let fallback = select_plan_permits_grant_aware(
        PermitAdmission::Cooperative,
        LlcPlanSizing::Exact,
        &pool,
        None,
        2,
        0,
        0,
        0,
        &[],
        &[],
        |_| Ok(true),
        |candidate| grant_conflicts(&all_charged, candidate),
    )
    .expect("grant-aware permit selection with a fully charged pool")
    .expect("the grant-blind fallback must fire instead of returning None");
    assert_eq!(
        fallback.permits.cpu_permits.len(),
        2,
        "under total charge the fallback must restore today's selection",
    );
}

/// The registration seed — the exact permit set a waiter publishes when it
/// joins the registry — is sized exactly, so the grant charge can only reorder
/// its walk. It must prefer the grant-free permits of each class while any
/// remain, and must hand back exactly the grant-blind selection (width, class,
/// and identities) once the charge covers the whole pool: a senior that never
/// publishes an overlapping claim never triggers the scan's ticket-order
/// revoke.
#[test]
fn registered_permit_seed_prefers_grant_free_permits_without_narrowing() {
    let pool = VmPermitPool {
        cpu: AdmissionPermitPool {
            general: (0..4).collect(),
            reserved: (4..6).collect(),
        },
        memory: MemoryPermitPool {
            permits: (100..104).collect(),
            usable_mib: 1024,
        },
        cpu_required: 2,
        memory_required: 1,
        cpu_rotation: 0,
        memory_rotation: 0,
        preferred_cpu: Vec::new(),
        preferred_memory: Vec::new(),
    };
    let charges = |charged: std::collections::BTreeSet<usize>| {
        move |candidate: &admission_protocol::ClaimSet| {
            Ok(candidate
                .permits
                .iter()
                .any(|permit| charged.contains(permit)))
        }
    };
    let blind = pool
        .select(|_| Ok(true))
        .expect("grant-blind seed")
        .expect("a free pool must seed a designation");
    assert_eq!(blind.cpu_permits, vec![0, 1]);
    assert_eq!(blind.memory_permits, vec![100]);

    let biased = pool
        .select_grant_aware(|_| Ok(true), charges([0, 1, 100].into_iter().collect()))
        .expect("grant-aware seed")
        .expect("a partly charged pool must still seed a designation");
    assert_eq!(
        (biased.cpu_permits, biased.memory_permits),
        (vec![2, 3], vec![101]),
        "the seed must walk past the permits an in-flight grant counts on",
    );
    assert_eq!(biased.admission_class, blind.admission_class);

    let saturated = pool
        .select_grant_aware(
            |_| Ok(true),
            charges(
                (0..6)
                    .chain(100..104)
                    .collect::<std::collections::BTreeSet<_>>(),
            ),
        )
        .expect("grant-aware seed against a fully charged pool")
        .expect("a fully charged pool must still seed a complete designation");
    assert_eq!(
        (
            saturated.cpu_permits,
            saturated.memory_permits,
            saturated.admission_class
        ),
        (
            blind.cpu_permits,
            blind.memory_permits,
            blind.admission_class
        ),
        "under total charge the bias must restore the grant-blind seed exactly",
    );
}

/// Width preservation: grant avoidance is a preference, never a filter
/// that shrinks the request. `LlcPlanSizing::Elastic` floors a permit
/// request at one, so a charge covering all but one permit lets a
/// filtering tier answer a multi-permit request from that single permit
/// — and an elastic selection's permit count *is* the CPU width
/// `apply_plan_permit_width` truncates the plan to. The grant charge
/// would then be deciding how many CPUs the plan funds, which is a
/// capacity decision a subordinate bias must never make. Avoidance has
/// to fill the requested width from the charged permits instead.
#[test]
fn grant_aware_permit_selection_preserves_elastic_plan_width() {
    let pool = AdmissionPermitPool::for_host(2);
    let all: Vec<usize> = pool.all().collect();
    assert!(all.len() >= 3, "fixture needs at least 3 permits");
    let grant_free = all[2];
    let charged: std::collections::BTreeSet<usize> = all
        .iter()
        .copied()
        .filter(|permit| *permit != grant_free)
        .collect();
    let selection = select_plan_permits_grant_aware(
        PermitAdmission::Cooperative,
        LlcPlanSizing::Elastic,
        &pool,
        None,
        2,
        0,
        0,
        0,
        &[],
        &[],
        |_| Ok(true),
        |candidate: &admission_protocol::ClaimSet| {
            Ok(candidate
                .permits
                .iter()
                .any(|permit| charged.contains(permit)))
        },
    )
    .expect("grant-aware permit selection")
    .expect("a fully available pool must satisfy the request");
    assert_eq!(
        selection.permits.cpu_permits.len(),
        2,
        "grant avoidance must fill the requested width from charged \
         permits rather than return a short set: {:?}",
        selection.permits.cpu_permits,
    );
    assert_eq!(
        selection.cpu_width, 2,
        "the funded CPU width must still be the requested width",
    );
    assert!(
        selection.permits.cpu_permits.contains(&grant_free),
        "the one grant-free permit must still be preferred: {:?}",
        selection.permits.cpu_permits,
    );
    // The consequence the width guards: the plan keeps every CPU the
    // selection was asked to fund.
    let topo = HostTopology::new_for_tests(&[(vec![0, 1], 0)]);
    let mut selected_llcs = vec![0];
    let mut selected_cpus = vec![0, 1];
    apply_plan_permit_width(
        LlcPlanSizing::Elastic,
        &selection,
        &topo,
        &mut selected_llcs,
        &mut selected_cpus,
    );
    assert_eq!(
        selected_cpus,
        vec![0, 1],
        "a short permit selection truncates the CPU plan",
    );
    assert_eq!(selected_llcs, vec![0]);
}
