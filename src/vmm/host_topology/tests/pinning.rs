use super::super::*;
use super::*;
use crate::vmm::topology::Topology;

#[test]
fn parse_cpu_list_range() {
    assert_eq!(parse_cpu_list_lenient("0-3"), vec![0, 1, 2, 3]);
}

#[test]
fn parse_cpu_list_single() {
    assert_eq!(parse_cpu_list_lenient("5"), vec![5]);
}

#[test]
fn parse_cpu_list_mixed() {
    assert_eq!(
        parse_cpu_list_lenient("0-2,5,7-9"),
        vec![0, 1, 2, 5, 7, 8, 9]
    );
}

#[test]
fn parse_cpu_list_empty() {
    assert!(parse_cpu_list_lenient("").is_empty());
}

#[test]
fn parse_cpu_list_whitespace() {
    assert_eq!(parse_cpu_list_lenient("0-3\n"), vec![0, 1, 2, 3]);
}

#[test]
fn host_topology_from_sysfs() {
    let topo = HostTopology::from_sysfs();
    assert!(topo.is_ok(), "should read host topology: {:?}", topo.err());
    let topo = topo.unwrap();
    assert!(!topo.online_cpus.is_empty());
    assert!(!topo.llc_groups.is_empty());
}

#[test]
fn pinning_plan_simple() {
    let topo = HostTopology::from_sysfs().unwrap();
    if topo.total_cpus() < 2 {
        return; // skip on single-CPU hosts
    }
    let plan = topo.compute_pinning(&Topology::new(1, 1, 2, 1), false, 0);
    assert!(plan.is_ok(), "pinning should succeed: {:?}", plan.err());
    let plan = plan.unwrap();
    assert_eq!(plan.assignments.len(), 2);
    // All assigned CPUs should be distinct.
    let cpus: Vec<usize> = plan.assignments.iter().map(|a| a.1).collect();
    let unique: std::collections::HashSet<usize> = cpus.iter().copied().collect();
    assert_eq!(cpus.len(), unique.len());
}

#[test]
fn pinning_plan_oversubscribed() {
    let topo = HostTopology::from_sysfs().unwrap();
    let too_many = topo.total_cpus() as u32 + 1;
    let plan = topo.compute_pinning(&Topology::new(1, 1, too_many, 1), false, 0);
    assert!(plan.is_err());
}

#[test]
fn hugepages_needed_values() {
    assert_eq!(hugepages_needed(2), 1);
    assert_eq!(hugepages_needed(4), 2);
    assert_eq!(hugepages_needed(2048), 1024);
    assert_eq!(hugepages_needed(3), 2);
}

#[test]
fn hugepages_free_parses_count_and_falls_back_to_zero() {
    // Drive the path-parameterized core `hugepages_free_from` against
    // fixture files so the parse and the documented 0-fallback are
    // pinned independent of the host's hugetlbfs configuration. The
    // public `hugepages_free()` is this function applied to the fixed
    // 2 MiB-pool sysfs path.
    let tmp_dir = tempfile::TempDir::new().unwrap();

    // Happy parse: a trimmed integer count round-trips exactly.
    let present = tmp_dir.path().join("free_hugepages");
    std::fs::write(&present, "7\n").unwrap();
    assert_eq!(
        hugepages_free_from(&present),
        7,
        "trimmed count must parse to its u64 value",
    );

    // Documented 0-fallback: an absent file yields 0, never Err/panic.
    let missing = tmp_dir.path().join("nonexistent_free_hugepages");
    assert_eq!(
        hugepages_free_from(&missing),
        0,
        "absent sysfs file must fall back to 0",
    );

    // Non-numeric contents also fall back to 0 (parse failure path).
    let garbage = tmp_dir.path().join("garbage_free_hugepages");
    std::fs::write(&garbage, "not-a-number\n").unwrap();
    assert_eq!(
        hugepages_free_from(&garbage),
        0,
        "unparseable contents must fall back to 0",
    );
}

#[test]
fn host_load_estimate_runs() {
    let result = host_load_estimate();
    // `host_load_estimate` reads `/proc/stat` (scanning for
    // the `procs_running` line) and
    // `/sys/devices/system/cpu/online`. Both are mandatory on
    // any Linux kernel with CONFIG_PROC_FS + CONFIG_SYSFS, so
    // `Some(_)` is guaranteed when the test runs on a Linux
    // host.
    assert!(result.is_some());
    let (running, total) = result.unwrap();
    assert!(total > 0);
    // `running` is the `procs_running` counter from
    // `/proc/stat` — number of processes currently in state
    // `R`. This test thread itself is running at observation
    // time, so the floor is 1.
    assert!(running >= 1);
}

// -- parse_cpu_list edge cases --

#[test]
fn parse_cpu_list_trailing_comma() {
    assert_eq!(parse_cpu_list_lenient("0,1,2,"), vec![0, 1, 2]);
}

#[test]
fn parse_cpu_list_leading_comma() {
    assert_eq!(parse_cpu_list_lenient(",0,1"), vec![0, 1]);
}

#[test]
fn parse_cpu_list_single_zero() {
    assert_eq!(parse_cpu_list_lenient("0"), vec![0]);
}

#[test]
fn parse_cpu_list_large_ids() {
    assert_eq!(parse_cpu_list_lenient("127,255"), vec![127, 255]);
}

#[test]
fn parse_cpu_list_reversed_range() {
    // "5-3" parses as start=5, end=3 — 5..=3 is empty.
    assert!(parse_cpu_list_lenient("5-3").is_empty());
}

#[test]
fn parse_cpu_list_non_numeric() {
    // Garbage is silently ignored.
    assert!(parse_cpu_list_lenient("abc").is_empty());
}

#[test]
fn compute_pinning_single_llc() {
    // 1 LLC with 4 CPUs, request 1 LLC x 2 cores x 1 thread.
    let topo = synthetic_topo(vec![vec![0, 1, 2, 3]]);
    let plan = topo
        .compute_pinning(&Topology::new(1, 1, 2, 1), false, 0)
        .unwrap();
    assert_eq!(plan.assignments.len(), 2);
    assert_eq!(plan.assignments[0], (0, 0));
    assert_eq!(plan.assignments[1], (1, 1));
}

#[test]
fn compute_pinning_two_llcs() {
    // 2 LLCs, each with 4 CPUs. Request 2l2c1t.
    let topo = synthetic_topo(vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]]);
    let plan = topo
        .compute_pinning(&Topology::new(1, 2, 2, 1), false, 0)
        .unwrap();
    assert_eq!(plan.assignments.len(), 4);
    // LLC 0 vCPUs (0,1) should map to LLC group 0 CPUs (0,1).
    assert_eq!(plan.assignments[0], (0, 0));
    assert_eq!(plan.assignments[1], (1, 1));
    // LLC 1 vCPUs (2,3) should map to LLC group 1 CPUs (4,5).
    assert_eq!(plan.assignments[2], (2, 4));
    assert_eq!(plan.assignments[3], (3, 5));
}

#[test]
fn compute_pinning_with_smt() {
    // 1 LLC with 8 CPUs, request 1l2c2t = 4 vCPUs.
    let topo = synthetic_topo(vec![vec![0, 1, 2, 3, 4, 5, 6, 7]]);
    let plan = topo
        .compute_pinning(&Topology::new(1, 1, 2, 2), false, 0)
        .unwrap();
    assert_eq!(plan.assignments.len(), 4);
    // All 4 vCPUs map to distinct CPUs within the same LLC.
    let cpus: Vec<usize> = plan.assignments.iter().map(|a| a.1).collect();
    let unique: std::collections::HashSet<usize> = cpus.iter().copied().collect();
    assert_eq!(cpus.len(), unique.len());
}

#[test]
fn compute_pinning_exact_fit() {
    // 2 LLCs with exactly 2 CPUs each, request 2l2c1t = 4 total.
    let topo = synthetic_topo(vec![vec![0, 1], vec![2, 3]]);
    let plan = topo
        .compute_pinning(&Topology::new(1, 2, 2, 1), false, 0)
        .unwrap();
    assert_eq!(plan.assignments.len(), 4);
    // All host CPUs consumed.
    let assigned: std::collections::HashSet<usize> = plan.assignments.iter().map(|a| a.1).collect();
    let all_cpus: std::collections::HashSet<usize> = topo.online_cpus.iter().copied().collect();
    assert_eq!(assigned, all_cpus, "exact fit must consume all host CPUs");
    // No duplicates (unique count == total count).
    assert_eq!(
        assigned.len(),
        plan.assignments.len(),
        "all assignments must be unique",
    );
}

#[test]
fn compute_pinning_error_too_many_vcpus() {
    // 1 LLC with 2 CPUs, request 4 vCPUs.
    let topo = synthetic_topo(vec![vec![0, 1]]);
    let err = topo
        .compute_pinning(&Topology::new(1, 1, 4, 1), false, 0)
        .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("4 vCPUs") && msg.contains("2 host CPUs"),
        "error should mention CPU counts: {msg}",
    );
    assert!(
        err.downcast_ref::<TopologyInsufficient>().is_some(),
        "must be a typed TopologyInsufficient so the skip routing is \
         not a fragile message string-match: {err:#}",
    );
}

#[test]
fn compute_pinning_error_too_many_llcs() {
    // 1 LLC, request 2 LLCs.
    let topo = synthetic_topo(vec![vec![0, 1, 2, 3]]);
    let err = topo
        .compute_pinning(&Topology::new(1, 2, 1, 1), false, 0)
        .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("2 LLCs") && msg.contains("1 LLC groups"),
        "error should mention LLC count mismatch: {msg}",
    );
    assert!(
        err.downcast_ref::<TopologyInsufficient>().is_some(),
        "LLC-shortage must be a typed TopologyInsufficient: {err:#}",
    );
}

#[test]
fn compute_pinning_error_llc_too_small() {
    // 2 LLCs: first has 4 CPUs, second has only 1. Request 2l2c1t.
    let topo = synthetic_topo(vec![vec![0, 1, 2, 3], vec![4]]);
    let err = topo
        .compute_pinning(&Topology::new(1, 2, 2, 1), false, 0)
        .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("LLC group 1") && msg.contains("1 available"),
        "error should identify the undersized LLC: {msg}",
    );
    assert!(
        err.downcast_ref::<TopologyInsufficient>().is_some(),
        "per-LLC shortage must be a typed TopologyInsufficient: {err:#}",
    );
}

#[test]
fn compute_pinning_no_cross_llc_sharing() {
    // Verify vCPUs in different LLCs never share an LLC's CPUs.
    let topo = synthetic_topo(vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7], vec![8, 9, 10, 11]]);
    let plan = topo
        .compute_pinning(&Topology::new(1, 3, 2, 1), false, 0)
        .unwrap();
    // LLC 0 should only use CPUs 0-3, LLC 1 only 4-7, LLC 2 only 8-11.
    for (vcpu_id, host_cpu) in &plan.assignments {
        let llc_idx = vcpu_id / 2; // 2 vCPUs per LLC
        let llc_start = llc_idx as usize * 4;
        let llc_end = llc_start + 3;
        assert!(
            *host_cpu >= llc_start && *host_cpu <= llc_end,
            "vCPU {vcpu_id} (LLC {llc_idx}) pinned to CPU {host_cpu}, \
             expected range {llc_start}..={llc_end}",
        );
    }
}

#[test]
fn compute_pinning_all_assignments_unique() {
    let topo = synthetic_topo(vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]]);
    let plan = topo
        .compute_pinning(&Topology::new(1, 2, 4, 1), false, 0)
        .unwrap();
    let cpus: Vec<usize> = plan.assignments.iter().map(|a| a.1).collect();
    let unique: std::collections::HashSet<usize> = cpus.iter().copied().collect();
    assert_eq!(
        cpus.len(),
        unique.len(),
        "all host CPU assignments must be unique: {:?}",
        cpus,
    );
}

#[test]
fn compute_pinning_vcpu_ids_sequential() {
    let topo = synthetic_topo(vec![vec![0, 1, 2, 3]]);
    let plan = topo
        .compute_pinning(&Topology::new(1, 1, 4, 1), false, 0)
        .unwrap();
    let vcpu_ids: Vec<u32> = plan.assignments.iter().map(|a| a.0).collect();
    assert_eq!(vcpu_ids, vec![0, 1, 2, 3]);
}

#[test]
fn compute_pinning_single_vcpu() {
    let topo = synthetic_topo(vec![vec![42]]);
    let plan = topo
        .compute_pinning(&Topology::new(1, 1, 1, 1), false, 0)
        .unwrap();
    assert_eq!(plan.assignments.len(), 1);
    assert_eq!(plan.assignments[0], (0, 42));
}

// -- sysfs-based tests with real host topology --

#[test]
fn sysfs_llc_groups_cover_all_cpus() {
    let topo = HostTopology::from_sysfs().unwrap();
    let llc_cpus: Vec<usize> = topo
        .llc_groups
        .iter()
        .flat_map(|g| g.cpus.iter().copied())
        .collect();
    for cpu in &topo.online_cpus {
        assert!(
            llc_cpus.contains(cpu),
            "CPU {} is online but not in any LLC group",
            cpu,
        );
    }
}

#[test]
fn sysfs_llc_groups_nonempty() {
    let topo = HostTopology::from_sysfs().unwrap();
    for (i, group) in topo.llc_groups.iter().enumerate() {
        assert!(
            !group.cpus.is_empty(),
            "LLC group {} should have at least one CPU",
            i,
        );
    }
}

#[test]
fn sysfs_pinning_respects_llc_boundaries() {
    let topo = HostTopology::from_sysfs().unwrap();
    if topo.llc_groups.len() < 2 || topo.total_cpus() < 4 {
        return; // need at least 2 LLCs with 2+ CPUs each
    }
    let min_llc_size = topo.llc_groups.iter().map(|g| g.cpus.len()).min().unwrap();
    if min_llc_size < 2 {
        return;
    }
    let plan = topo
        .compute_pinning(&Topology::new(1, 2, 2, 1), false, 0)
        .unwrap();
    // LLC 0 vCPUs should be in LLC group 0.
    for (vcpu_id, host_cpu) in &plan.assignments {
        let llc_idx = vcpu_id / 2;
        let group = &topo.llc_groups[llc_idx as usize];
        assert!(
            group.cpus.contains(host_cpu),
            "vCPU {} mapped to CPU {} which is not in LLC group {}",
            vcpu_id,
            host_cpu,
            llc_idx,
        );
    }
}

// -- hugepages_needed edge cases --

#[test]
fn hugepages_needed_boundary() {
    assert_eq!(hugepages_needed(1), 1); // 1 MB -> ceil(1/2) = 1
    assert_eq!(hugepages_needed(0), 0);
}

#[test]
fn hugepages_needed_exact_multiple() {
    assert_eq!(hugepages_needed(1024), 512);
}

// -- service CPU reservation tests --

#[test]
fn compute_pinning_service_cpu_picks_unpinned() {
    // 4 CPUs in one LLC, request 2 vCPUs + service CPU.
    let topo = synthetic_topo(vec![vec![0, 1, 2, 3]]);
    let plan = topo
        .compute_pinning(&Topology::new(1, 1, 2, 1), true, 0)
        .unwrap();
    assert_eq!(plan.assignments.len(), 2);
    let service = plan.service_cpu.expect("service_cpu should be set");
    // Service CPU must not overlap with any vCPU assignment.
    let vcpu_cpus: std::collections::HashSet<usize> =
        plan.assignments.iter().map(|a| a.1).collect();
    assert!(
        !vcpu_cpus.contains(&service),
        "service CPU {service} must not be assigned to a vCPU",
    );
}

#[test]
fn compute_pinning_service_cpu_false_returns_none() {
    let topo = synthetic_topo(vec![vec![0, 1, 2, 3]]);
    let plan = topo
        .compute_pinning(&Topology::new(1, 1, 2, 1), false, 0)
        .unwrap();
    assert!(plan.service_cpu.is_none());
}

#[test]
fn compute_pinning_service_cpu_exact_fit() {
    // 3 CPUs total, request 2 vCPUs + 1 service = exact fit.
    let topo = synthetic_topo(vec![vec![0, 1, 2]]);
    let plan = topo
        .compute_pinning(&Topology::new(1, 1, 2, 1), true, 0)
        .unwrap();
    let service = plan.service_cpu.expect("service_cpu should be set");
    // vCPUs consume CPUs 0,1. The only remaining CPU is 2.
    assert_eq!(service, 2, "service CPU should be the only remaining CPU");
    // Service CPU must not overlap with vCPU assignments.
    let vcpu_cpus: std::collections::HashSet<usize> =
        plan.assignments.iter().map(|a| a.1).collect();
    assert!(
        !vcpu_cpus.contains(&service),
        "service CPU {service} must not overlap vCPU assignments",
    );
}

#[test]
fn compute_pinning_service_cpu_insufficient_fails() {
    // 2 CPUs, request 2 vCPUs + 1 service = 3 needed, only 2 available.
    let topo = synthetic_topo(vec![vec![0, 1]]);
    let err = topo
        .compute_pinning(&Topology::new(1, 1, 2, 1), true, 0)
        .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("3 CPUs") && msg.contains("2 host CPUs"),
        "error should mention CPU shortage: {msg}",
    );
}

#[test]
fn compute_pinning_service_cpu_multi_llc() {
    // 2 LLCs with 3 CPUs each, request 2l2c1t + service = 5 CPUs needed.
    let topo = synthetic_topo(vec![vec![0, 1, 2], vec![3, 4, 5]]);
    let plan = topo
        .compute_pinning(&Topology::new(1, 2, 2, 1), true, 0)
        .unwrap();
    let service = plan.service_cpu.unwrap();
    let vcpu_cpus: std::collections::HashSet<usize> =
        plan.assignments.iter().map(|a| a.1).collect();
    assert!(!vcpu_cpus.contains(&service));
}

// -- NUMA node discovery tests --

#[test]
fn sysfs_cpu_to_node_populated() {
    let topo = HostTopology::from_sysfs().unwrap();
    // On any Linux host, at least some CPUs should have NUMA info.
    // On single-node systems the map may map everything to node 0.
    if !topo.cpu_to_node.is_empty() {
        for (&cpu, &node) in &topo.cpu_to_node {
            assert!(
                topo.online_cpus.contains(&cpu),
                "NUMA mapping for CPU {cpu} but not in online set",
            );
            // NUMA node IDs are typically small (0-N).
            assert!(node < 1024, "unexpected NUMA node ID {node} for CPU {cpu}");
        }
    }
}

#[test]
fn max_cores_per_llc_synthetic() {
    let topo = synthetic_topo(vec![vec![0, 1, 2, 3], vec![4, 5]]);
    assert_eq!(topo.max_cores_per_llc(), 4);
}

#[test]
fn max_cores_per_llc_uniform() {
    let topo = synthetic_topo(vec![vec![0, 1, 2], vec![3, 4, 5]]);
    assert_eq!(topo.max_cores_per_llc(), 3);
}

#[test]
fn mbind_to_nodes_short_circuits_on_empty_nodes_or_zero_len() {
    // `mbind_to_nodes` short-circuits before the `mbind(2)` syscall
    // (and before dereferencing `addr`) exactly when
    // `mbind_should_skip` returns true: an empty node set or a
    // zero-length region. Assert the pure predicate directly — the
    // through-`mbind_to_nodes` path swallows syscall errors, so a
    // not-crashing call would pass regardless of whether the guard
    // was present, defeating the test's purpose.
    //
    // Skip cases (no policy target / no bytes to bind):
    assert!(mbind_should_skip(0, &[]), "empty nodes + zero len skips");
    assert!(
        mbind_should_skip(4096, &[]),
        "empty nodes skips regardless of len",
    );
    assert!(
        mbind_should_skip(0, &[0, 1]),
        "zero len skips regardless of nodes",
    );

    // Do-work case: a non-empty node set with a non-zero region must
    // NOT skip — this is the branch that reaches the syscall.
    assert!(
        !mbind_should_skip(4096, &[0, 1]),
        "non-empty nodes + non-zero len must reach the mbind syscall",
    );
}

#[test]
fn llc_numa_node_synthetic() {
    // 4 LLCs: 0,1 on node 0; 2,3 on node 1.
    let topo = synthetic_topo_numa(vec![
        (0, vec![0, 1]),
        (0, vec![2, 3]),
        (1, vec![4, 5]),
        (1, vec![6, 7]),
    ]);
    assert_eq!(topo.llc_numa_node(0), 0);
    assert_eq!(topo.llc_numa_node(1), 0);
    assert_eq!(topo.llc_numa_node(2), 1);
    assert_eq!(topo.llc_numa_node(3), 1);
}

#[test]
fn compute_pinning_numa_two_nodes() {
    // Host: 4 LLCs, 2 per NUMA node. LLCs 0,1 on node 0; LLCs 2,3 on node 1.
    // Guest: 2 NUMA nodes, 4 LLCs (2 per node), 2 cores each.
    let topo = synthetic_topo_numa(vec![
        (0, vec![0, 1, 2, 3]),
        (0, vec![4, 5, 6, 7]),
        (1, vec![8, 9, 10, 11]),
        (1, vec![12, 13, 14, 15]),
    ]);
    let plan = topo
        .compute_pinning(&Topology::new(2, 4, 2, 1), false, 0)
        .unwrap();
    assert_eq!(plan.assignments.len(), 8);

    // Guest NUMA node 0 (vLLCs 0,1) should map to host LLCs on the
    // same physical NUMA node.
    let node_0_cpus: Vec<usize> = plan
        .assignments
        .iter()
        .filter(|(vcpu, _)| *vcpu < 4) // vLLC 0,1 = vCPUs 0-3
        .map(|(_, cpu)| *cpu)
        .collect();
    let node_0_host_nodes = numa_nodes_for_cpus(&topo, &node_0_cpus);
    assert_eq!(
        node_0_host_nodes.len(),
        1,
        "guest NUMA 0 LLCs should all be on one host NUMA node, got {:?}",
        node_0_host_nodes,
    );

    // Guest NUMA node 1 (vLLCs 2,3) should map to host LLCs on the
    // same physical NUMA node.
    let node_1_cpus: Vec<usize> = plan
        .assignments
        .iter()
        .filter(|(vcpu, _)| *vcpu >= 4) // vLLC 2,3 = vCPUs 4-7
        .map(|(_, cpu)| *cpu)
        .collect();
    let node_1_host_nodes = numa_nodes_for_cpus(&topo, &node_1_cpus);
    assert_eq!(
        node_1_host_nodes.len(),
        1,
        "guest NUMA 1 LLCs should all be on one host NUMA node, got {:?}",
        node_1_host_nodes,
    );

    // The two guest NUMA nodes should map to different host NUMA nodes.
    assert_ne!(
        node_0_host_nodes.iter().next(),
        node_1_host_nodes.iter().next(),
        "guest NUMA nodes should map to different host NUMA nodes",
    );
}

#[test]
fn numa_aware_llc_order_uneven_llcs_preserves_remainder() {
    // With llcs=5 and numa_nodes=2, naive integer division would
    // yield llcs_per_node=2 → only 4 entries in `order`, dropping
    // the remainder LLC. The implementation distributes the
    // remainder across the first `llcs % numa_nodes` guest
    // nodes: node 0 → 3 LLCs, node 1 → 2 LLCs, total 5.
    //
    // Host: 6 LLCs, 3 per NUMA node — satisfies the ceiling
    // eligibility check (each host node must supply max_per_node=3).
    let topo = synthetic_topo_numa(vec![
        (0, vec![0, 1]),
        (0, vec![2, 3]),
        (0, vec![4, 5]),
        (1, vec![6, 7]),
        (1, vec![8, 9]),
        (1, vec![10, 11]),
    ]);
    let order = topo.numa_aware_llc_order(2, 5, 0);
    assert_eq!(
        order.len(),
        5,
        "uneven llc distribution must preserve all LLCs, got {order:?}"
    );
    // First 3 entries belong to the host's first eligible NUMA
    // node; last 2 to the second.
    let first_three_nodes: std::collections::BTreeSet<usize> = order[..3]
        .iter()
        .map(|&idx| topo.llc_numa_node(idx))
        .collect();
    let last_two_nodes: std::collections::BTreeSet<usize> = order[3..]
        .iter()
        .map(|&idx| topo.llc_numa_node(idx))
        .collect();
    assert_eq!(
        first_three_nodes.len(),
        1,
        "first 3 LLCs must share a host node"
    );
    assert_eq!(
        last_two_nodes.len(),
        1,
        "last 2 LLCs must share a host node"
    );
    assert_ne!(
        first_three_nodes, last_two_nodes,
        "two guest NUMA nodes must map to distinct host nodes"
    );
}

#[test]
fn numa_aware_llc_order_zero_numa_nodes_is_safe() {
    // numa_nodes=0 would divide-by-zero on `llcs / numa_nodes`.
    // Falls back to sequential mapping instead.
    let topo = synthetic_topo_numa(vec![(0, vec![0, 1]), (0, vec![2, 3])]);
    let order = topo.numa_aware_llc_order(0, 2, 0);
    // Pin the fallback CONTENT, not just length: a len-only check passes
    // for [0,0] (two guest LLCs double-pinned to host LLC 0), a real
    // misconfiguration. Sequential fallback over 2 host LLCs is [0,1].
    assert_eq!(
        order,
        vec![0, 1],
        "zero-numa fallback must map sequentially"
    );
}

#[test]
fn numa_aware_llc_order_fewer_llcs_than_nodes_falls_back() {
    // llcs < numa_nodes would give base_per_node = 0 and leave
    // some guest nodes empty. Falls back to sequential mapping
    // so all requested LLCs land.
    let topo = synthetic_topo_numa(vec![
        (0, vec![0, 1]),
        (0, vec![2, 3]),
        (1, vec![4, 5]),
        (1, vec![6, 7]),
    ]);
    let order = topo.numa_aware_llc_order(4, 2, 0);
    // Sequential fallback over 4 host LLCs is [0,1]. Pin content +
    // distinctness: [0,0] (len 2) would double-pin two guest LLCs to
    // host LLC 0 and never use LLC 1 — a real misconfiguration a
    // len-only check admits.
    assert_eq!(
        order,
        vec![0, 1],
        "fewer-llcs fallback must map to distinct LLCs"
    );
    let unique: std::collections::HashSet<usize> = order.iter().copied().collect();
    assert_eq!(
        unique.len(),
        order.len(),
        "no two guest LLCs may pin to the same host LLC"
    );
}

#[test]
fn compute_pinning_numa_fallback_insufficient_nodes() {
    // Host: 4 LLCs all on NUMA node 0; guest requests 2 NUMA
    // nodes. The host cannot distribute LLCs across distinct
    // nodes (there is only one), so `compute_pinning` falls
    // back to the single-node sequential mapping rather than
    // erroring. The fallback still produces 8 unique host-CPU
    // assignments so the VM boots on the same host memory
    // throughout.
    let topo = synthetic_topo_numa(vec![
        (0, vec![0, 1]),
        (0, vec![2, 3]),
        (0, vec![4, 5]),
        (0, vec![6, 7]),
    ]);
    let plan = topo
        .compute_pinning(&Topology::new(2, 4, 2, 1), false, 0)
        .unwrap();
    assert_eq!(plan.assignments.len(), 8);
    let cpus: Vec<usize> = plan.assignments.iter().map(|a| a.1).collect();
    let unique: std::collections::HashSet<usize> = cpus.iter().copied().collect();
    assert_eq!(cpus.len(), unique.len());
}

#[test]
fn compute_pinning_numa_single_node_unchanged() {
    // numa_nodes=1 should behave identically to the original sequential
    // mapping regardless of host NUMA layout.
    let topo = synthetic_topo_numa(vec![(0, vec![0, 1, 2, 3]), (1, vec![4, 5, 6, 7])]);
    let plan = topo
        .compute_pinning(&Topology::new(1, 2, 2, 1), false, 0)
        .unwrap();
    assert_eq!(plan.assignments.len(), 4);
    // Sequential: vLLC 0 -> host LLC 0, vLLC 1 -> host LLC 1.
    assert_eq!(plan.assignments[0], (0, 0));
    assert_eq!(plan.assignments[1], (1, 1));
    assert_eq!(plan.assignments[2], (2, 4));
    assert_eq!(plan.assignments[3], (3, 5));
}

#[test]
fn compute_pinning_numa_three_nodes() {
    // Host: 6 LLCs, 2 per NUMA node (nodes 0,1,2).
    // Guest: 3 NUMA nodes, 6 LLCs.
    let topo = synthetic_topo_numa(vec![
        (0, vec![0, 1]),
        (0, vec![2, 3]),
        (1, vec![4, 5]),
        (1, vec![6, 7]),
        (2, vec![8, 9]),
        (2, vec![10, 11]),
    ]);
    let plan = topo
        .compute_pinning(&Topology::new(3, 6, 1, 1), false, 0)
        .unwrap();
    assert_eq!(plan.assignments.len(), 6);

    // Each guest NUMA node's vCPUs should be on one host NUMA node.
    for guest_node in 0..3u32 {
        let start = guest_node * 2;
        let end = start + 2;
        let cpus: Vec<usize> = plan
            .assignments
            .iter()
            .filter(|(vcpu, _)| *vcpu >= start && *vcpu < end)
            .map(|(_, cpu)| *cpu)
            .collect();
        let nodes = numa_nodes_for_cpus(&topo, &cpus);
        assert_eq!(
            nodes.len(),
            1,
            "guest NUMA {} should be on one host NUMA node, got {:?}",
            guest_node,
            nodes,
        );
    }
}

#[test]
fn compute_pinning_numa_with_service_cpu() {
    // 2 NUMA nodes, 4 LLCs, request 2 NUMA nodes + service CPU.
    let topo = synthetic_topo_numa(vec![
        (0, vec![0, 1, 2, 3]),
        (0, vec![4, 5, 6, 7]),
        (1, vec![8, 9, 10, 11]),
        (1, vec![12, 13, 14, 15]),
    ]);
    let plan = topo
        .compute_pinning(&Topology::new(2, 4, 2, 1), true, 0)
        .unwrap();
    assert_eq!(plan.assignments.len(), 8);
    let service = plan.service_cpu.expect("service_cpu should be set");
    let vcpu_cpus: std::collections::HashSet<usize> =
        plan.assignments.iter().map(|a| a.1).collect();
    assert!(
        !vcpu_cpus.contains(&service),
        "service CPU {service} must not overlap vCPU assignments",
    );
}

#[test]
fn llc_numa_node_empty_map() {
    // Empty cpu_to_node should default to node 0 (implicit via
    // the unwrap_or(0) in `llc_numa_node`).
    //
    // Construct manually rather than via `new_for_tests` because
    // this test pins behavior when `cpu_to_node` is EMPTY — our
    // fixture helper always populates node ids. Emptying
    // `cpu_to_node` after a `new_for_tests` call is clearer than
    // a special-case seam.
    let mut topo = HostTopology::new_for_tests(&[(vec![0, 1], 0)]);
    topo.cpu_to_node.clear();
    topo.host_node_llcs.clear();
    assert_eq!(topo.llc_numa_node(0), 0);
}

// -- llc_offset pinning tests --

#[test]
fn compute_pinning_offset_single_llc_wraps() {
    // 1 host LLC with 4 CPUs, request 1l2c1t, offset 1.
    // (0 + 1) % 1 = 0 — wraps back to the only LLC.
    let topo = synthetic_topo(vec![vec![0, 1, 2, 3]]);
    let plan = topo
        .compute_pinning(&Topology::new(1, 1, 2, 1), false, 1)
        .unwrap();
    assert_eq!(plan.assignments.len(), 2);
    assert_eq!(plan.assignments[0], (0, 0));
    assert_eq!(plan.assignments[1], (1, 1));
}

#[test]
fn compute_pinning_offset_two_llcs_shifts() {
    // 2 host LLCs, request 2l2c1t, offset 1.
    // vLLC 0 -> (0+1)%2 = host LLC 1 (CPUs 4,5).
    // vLLC 1 -> (1+1)%2 = host LLC 0 (CPUs 0,1).
    let topo = synthetic_topo(vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]]);
    let plan = topo
        .compute_pinning(&Topology::new(1, 2, 2, 1), false, 1)
        .unwrap();
    assert_eq!(plan.assignments.len(), 4);
    assert_eq!(plan.assignments[0], (0, 4));
    assert_eq!(plan.assignments[1], (1, 5));
    assert_eq!(plan.assignments[2], (2, 0));
    assert_eq!(plan.assignments[3], (3, 1));
}

#[test]
fn compute_pinning_offset_wraps_modulo() {
    // 2 host LLCs, request 2l2c1t, offset 2.
    // (0+2)%2 = 0, (1+2)%2 = 1 — same as offset 0.
    let topo = synthetic_topo(vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]]);
    let plan = topo
        .compute_pinning(&Topology::new(1, 2, 2, 1), false, 2)
        .unwrap();
    assert_eq!(plan.assignments.len(), 4);
    assert_eq!(plan.assignments[0], (0, 0));
    assert_eq!(plan.assignments[1], (1, 1));
    assert_eq!(plan.assignments[2], (2, 4));
    assert_eq!(plan.assignments[3], (3, 5));
}

#[test]
fn compute_pinning_offset_three_llcs_partial() {
    // 3 host LLCs (4 CPUs each), request 2l2c1t, offset 1.
    // vLLC 0 -> (0+1)%3 = host LLC 1 (CPUs 4,5).
    // vLLC 1 -> (1+1)%3 = host LLC 2 (CPUs 8,9).
    let topo = synthetic_topo(vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7], vec![8, 9, 10, 11]]);
    let plan = topo
        .compute_pinning(&Topology::new(1, 2, 2, 1), false, 1)
        .unwrap();
    assert_eq!(plan.assignments.len(), 4);
    assert_eq!(plan.assignments[0], (0, 4));
    assert_eq!(plan.assignments[1], (1, 5));
    assert_eq!(plan.assignments[2], (2, 8));
    assert_eq!(plan.assignments[3], (3, 9));
}

#[test]
fn compute_pinning_offset_large_wraps() {
    // 3 host LLCs, request 1l2c1t, offset 5.
    // (0 + 5) % 3 = 2 — maps to host LLC 2 (CPUs 8,9).
    let topo = synthetic_topo(vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7], vec![8, 9, 10, 11]]);
    let plan = topo
        .compute_pinning(&Topology::new(1, 1, 2, 1), false, 5)
        .unwrap();
    assert_eq!(plan.assignments.len(), 2);
    assert_eq!(plan.assignments[0], (0, 8));
    assert_eq!(plan.assignments[1], (1, 9));
}

#[test]
fn compute_pinning_offset_numa_within_rotation() {
    // 4 host LLCs across 2 NUMA nodes, offset 1.
    // node_offset = 1/2 = 0 (no node rotation).
    // within_offset = 1 % 2 = 1 (rotate within each node).
    // Guest node 0 → host node 0: LLCs [1, 0].
    // Guest node 1 → host node 1: LLCs [3, 2].
    // LLC order: [1, 0, 3, 2].
    let topo = synthetic_topo_numa(vec![
        (0, vec![0, 1, 2, 3]),
        (0, vec![4, 5, 6, 7]),
        (1, vec![8, 9, 10, 11]),
        (1, vec![12, 13, 14, 15]),
    ]);
    let plan = topo
        .compute_pinning(&Topology::new(2, 4, 2, 1), false, 1)
        .unwrap();
    assert_eq!(plan.assignments.len(), 8);
    // vLLC 0 → host LLC 1 (CPUs 4,5).
    assert_eq!(plan.assignments[0], (0, 4));
    assert_eq!(plan.assignments[1], (1, 5));
    // vLLC 1 → host LLC 0 (CPUs 0,1).
    assert_eq!(plan.assignments[2], (2, 0));
    assert_eq!(plan.assignments[3], (3, 1));
    // vLLC 2 → host LLC 3 (CPUs 12,13).
    assert_eq!(plan.assignments[4], (4, 12));
    assert_eq!(plan.assignments[5], (5, 13));
    // vLLC 3 → host LLC 2 (CPUs 8,9).
    assert_eq!(plan.assignments[6], (6, 8));
    assert_eq!(plan.assignments[7], (7, 9));
}

#[test]
fn compute_pinning_offset_numa_node_rotation() {
    // 4 host LLCs across 2 NUMA nodes, offset 2.
    // node_offset = 2/2 = 1 (rotates guest→host node mapping).
    // within_offset = 2 % 2 = 0 (no within-node rotation).
    // Guest node 0 → host node 1: LLCs [2, 3].
    // Guest node 1 → host node 0: LLCs [0, 1].
    // LLC order: [2, 3, 0, 1].
    let topo = synthetic_topo_numa(vec![
        (0, vec![0, 1, 2, 3]),
        (0, vec![4, 5, 6, 7]),
        (1, vec![8, 9, 10, 11]),
        (1, vec![12, 13, 14, 15]),
    ]);
    let plan = topo
        .compute_pinning(&Topology::new(2, 4, 2, 1), false, 2)
        .unwrap();
    assert_eq!(plan.assignments.len(), 8);
    // vLLC 0 → host LLC 2 (CPUs 8,9).
    assert_eq!(plan.assignments[0], (0, 8));
    assert_eq!(plan.assignments[1], (1, 9));
    // vLLC 1 → host LLC 3 (CPUs 12,13).
    assert_eq!(plan.assignments[2], (2, 12));
    assert_eq!(plan.assignments[3], (3, 13));
    // vLLC 2 → host LLC 0 (CPUs 0,1).
    assert_eq!(plan.assignments[4], (4, 0));
    assert_eq!(plan.assignments[5], (5, 1));
    // vLLC 3 → host LLC 1 (CPUs 4,5).
    assert_eq!(plan.assignments[6], (6, 4));
    assert_eq!(plan.assignments[7], (7, 5));
}

#[test]
fn compute_pinning_offset_with_service_cpu() {
    // 2 host LLCs, offset 1, reserve_service_cpu=true.
    // LLC order: [1, 0]. vCPUs consume 4,5,0,1.
    // Service CPU: first online_cpus entry not in {0,1,4,5} → 2.
    let topo = synthetic_topo(vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]]);
    let plan = topo
        .compute_pinning(&Topology::new(1, 2, 2, 1), true, 1)
        .unwrap();
    assert_eq!(plan.assignments.len(), 4);
    assert_eq!(plan.assignments[0], (0, 4));
    assert_eq!(plan.assignments[1], (1, 5));
    assert_eq!(plan.assignments[2], (2, 0));
    assert_eq!(plan.assignments[3], (3, 1));
    let service = plan.service_cpu.expect("service_cpu should be set");
    assert_eq!(service, 2);
    let vcpu_cpus: std::collections::HashSet<usize> =
        plan.assignments.iter().map(|a| a.1).collect();
    assert!(!vcpu_cpus.contains(&service));
}

#[test]
fn compute_pinning_offset_numa_combined_rotation() {
    // 4 host LLCs across 2 NUMA nodes, offset 3.
    // node_offset = 3/2 = 1 (rotates node mapping).
    // within_offset = 3 % 2 = 1 (rotates within each node).
    // Guest node 0 → host node 1: LLCs [3, 2].
    // Guest node 1 → host node 0: LLCs [1, 0].
    // LLC order: [3, 2, 1, 0].
    let topo = synthetic_topo_numa(vec![
        (0, vec![0, 1, 2, 3]),
        (0, vec![4, 5, 6, 7]),
        (1, vec![8, 9, 10, 11]),
        (1, vec![12, 13, 14, 15]),
    ]);
    let plan = topo
        .compute_pinning(&Topology::new(2, 4, 2, 1), false, 3)
        .unwrap();
    assert_eq!(plan.assignments.len(), 8);
    // vLLC 0 → host LLC 3 (CPUs 12,13).
    assert_eq!(plan.assignments[0], (0, 12));
    assert_eq!(plan.assignments[1], (1, 13));
    // vLLC 1 → host LLC 2 (CPUs 8,9).
    assert_eq!(plan.assignments[2], (2, 8));
    assert_eq!(plan.assignments[3], (3, 9));
    // vLLC 2 → host LLC 1 (CPUs 4,5).
    assert_eq!(plan.assignments[4], (4, 4));
    assert_eq!(plan.assignments[5], (5, 5));
    // vLLC 3 → host LLC 0 (CPUs 0,1).
    assert_eq!(plan.assignments[6], (6, 0));
    assert_eq!(plan.assignments[7], (7, 1));
}

// -- resource lock tests --

#[test]
fn resource_lock_exclusive_acquires() {
    let _tempfile_keep_alive = tempfile::Builder::new()
        .prefix("ktstr-test-flock-excl-acquires-")
        .suffix(".lock")
        .tempfile()
        .unwrap();
    let path = _tempfile_keep_alive.path().to_str().unwrap();
    let fd = try_flock(path, FlockMode::Exclusive).expect("open should succeed");
    assert!(fd.is_some(), "exclusive lock on fresh file should succeed");
}

// -- per-CPU-grain perf reservation: threshold + placement --
//
// Anchors the grain switch against the MEASURED validated-host LLC
// sizes and perf-cell footprints so a future ratio/floor tweak that
// would reclassify a validated host trips a red test:
//   * dev box:  16-CPU L3s
//   * x86 CI:   ~8-CPU L3s
//   * armly:     4-CPU L2s
//   * Graviton: one 96-CPU L3   (the box the grain switch exists for)
// Perf cells are `cores = 2..4, threads = 1` (2-4 vCPUs per LLC).

/// A single monolithic-LLC host of `n` CPUs on one NUMA node — the
/// Graviton shape.
fn monolith_host(n: usize) -> HostTopology {
    HostTopology::new_for_tests(&[((0..n).collect::<Vec<_>>(), 0)])
}

/// `k` LLCs of `size` CPUs each on one NUMA node — the x86/dev shape.
fn multi_llc_host(k: usize, size: usize) -> HostTopology {
    let groups: Vec<(Vec<usize>, usize)> = (0..k)
        .map(|i| ((i * size..i * size + size).collect(), 0))
        .collect();
    HostTopology::new_for_tests(&groups)
}

#[test]
fn perf_grain_switch_fires_only_on_a_huge_llc() {
    // Graviton: a 2-vCPU cell on a 96-CPU L3 → per-CPU grain (Shared).
    let host = monolith_host(96);
    let plan = host
        .compute_pinning(&Topology::new(1, 1, 2, 1), true, 0)
        .unwrap();
    assert_eq!(perf_llc_lock_mode(&host, &plan), LlcLockMode::Shared);
}

#[test]
fn perf_grain_stays_exclusive_on_validated_x86_and_dev_and_arm() {
    // Every validated host keeps whole-LLC Exclusive for BOTH real perf
    // cell sizes — the behaviour the dilation campaign measured. The
    // absolute floor (32) makes this hold regardless of how small the
    // cell is: a pure 0.5-occupancy ratio would flip the 2-vCPU cases.
    for (k, size) in [(24usize, 8usize), (4, 16), (6, 4)] {
        // (x86 CI ~8, dev box 16, armly 4)
        let host = multi_llc_host(k, size);
        for cores in [2u32, 4] {
            if cores as usize > size {
                continue; // cell cannot map — skip the impossible pin
            }
            let plan = host
                .compute_pinning(&Topology::new(1, 1, cores, 1), true, 0)
                .unwrap();
            assert_eq!(
                perf_llc_lock_mode(&host, &plan),
                LlcLockMode::Exclusive,
                "validated host {k}×{size}-CPU LLC, {cores}-core cell must stay Exclusive",
            );
        }
    }
}

#[test]
fn perf_grain_keeps_exclusive_when_cell_fills_most_of_a_huge_llc() {
    // A cell wanting >= 1/2 of even a 96-CPU L3 keeps the whole cache:
    // the occupancy ratio, not just the absolute floor, gates the switch.
    let host = monolith_host(96);
    // 48 vCPUs + 1 service = 49 footprint → 49/96 >= 1/2 → Exclusive.
    let fills = host
        .compute_pinning(&Topology::new(1, 1, 48, 1), true, 0)
        .unwrap();
    assert_eq!(perf_llc_lock_mode(&host, &fills), LlcLockMode::Exclusive);
    // 46 vCPUs + 1 = 47 footprint → 47/96 < 1/2 → Shared (minority slice).
    let minority = host
        .compute_pinning(&Topology::new(1, 1, 46, 1), true, 0)
        .unwrap();
    assert_eq!(perf_llc_lock_mode(&host, &minority), LlcLockMode::Shared);
}

#[test]
fn compute_pinning_grain_blocks_are_disjoint_including_service() {
    // Distinct blocks yield fully disjoint host-CPU sets — vCPUs AND
    // service — the property that lets two perf cells coexist.
    let host = monolith_host(96);
    let topo = Topology::new(1, 1, 2, 1); // V = 2 → block size 3
    let b0 = host.compute_pinning_grain(&topo, 0).unwrap();
    let b1 = host.compute_pinning_grain(&topo, 1).unwrap();

    // Block 0: vCPUs {0,1}, service 2.  Block 1: vCPUs {3,4}, service 5.
    assert_eq!(b0.assignments, vec![(0, 0), (1, 1)]);
    assert_eq!(b0.service_cpu, Some(2));
    assert_eq!(b1.assignments, vec![(0, 3), (1, 4)]);
    assert_eq!(b1.service_cpu, Some(5));

    let cpus = |p: &PinningPlan| -> std::collections::HashSet<usize> {
        let mut s: std::collections::HashSet<usize> =
            p.assignments.iter().map(|&(_, c)| c).collect();
        s.extend(p.service_cpu);
        s
    };
    assert!(
        cpus(&b0).is_disjoint(&cpus(&b1)),
        "grain blocks must not share any CPU (vCPU or service)",
    );
    // And the grain plan itself re-derives as Shared (run-path parity).
    assert_eq!(perf_llc_lock_mode(&host, &b0), LlcLockMode::Shared);
}

#[test]
fn compute_pinning_grain_windows_are_cache_coherent_contiguous() {
    // Each guest LLC's vCPUs stay a contiguous slice of ONE host LLC —
    // topology mirroring, never scattered across the cache.
    let host = monolith_host(96);
    let topo = Topology::new(1, 1, 4, 1); // V = 4
    let b2 = host.compute_pinning_grain(&topo, 2).unwrap();
    // block_size = 5, block 2 starts at CPU 10: vCPUs 10..14, service 14.
    let cpus: Vec<usize> = b2.assignments.iter().map(|&(_, c)| c).collect();
    assert_eq!(cpus, vec![10, 11, 12, 13]);
    assert_eq!(b2.service_cpu, Some(14));
}

#[test]
fn compute_pinning_grain_reports_end_of_blocks() {
    // A 96-CPU LLC / block size 3 seats exactly 32 blocks; block 32 does
    // not fit and returns TopologyInsufficient (the caller's break).
    let host = monolith_host(96);
    let topo = Topology::new(1, 1, 2, 1);
    assert!(host.compute_pinning_grain(&topo, 31).is_ok());
    let past = host.compute_pinning_grain(&topo, 32).unwrap_err();
    assert!(
        past.downcast_ref::<TopologyInsufficient>().is_some(),
        "past-last-block must be TopologyInsufficient, got: {past:#}",
    );
}
