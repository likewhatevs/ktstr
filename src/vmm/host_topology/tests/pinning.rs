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

// -- sysfs-based tests with real host topology --

#[test]
fn sysfs_llc_groups_cover_all_cpus() {
    let topo = HostTopology::from_sysfs().unwrap();
    let llc_cpus: Vec<usize> = topo
        .llc_groups
        .iter()
        .flat_map(|g| g.cpus.iter().copied())
        .collect();
    assert_eq!(
        llc_cpus.len(),
        topo.online_cpus.len(),
        "flattened LLC membership must name every online CPU exactly once",
    );
    assert_eq!(
        llc_cpus
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>(),
        topo.online_cpus
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>(),
        "LLC membership must be an exact partition of the online CPU set",
    );
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

/// `k` LLCs of `size` CPUs each on one NUMA node — the x86/dev shape.
fn multi_llc_host(k: usize, size: usize) -> HostTopology {
    let groups: Vec<(Vec<usize>, usize)> = (0..k)
        .map(|i| ((i * size..i * size + size).collect(), 0))
        .collect();
    HostTopology::new_for_tests(&groups)
}

#[test]
fn performance_candidates_include_numa_rotations_skipped_by_slot_stride() {
    // With one guest LLC per NUMA node, offsets 0 and 1 select the two
    // disjoint host windows. The old `slot * guest_llcs` scan used offsets
    // 0 and 2; NUMA node rotation made offset 2 alias offset 0 and hid the
    // second half of the machine.
    let host = synthetic_topo_numa(vec![
        (0, vec![0, 1, 2, 3]),
        (0, vec![4, 5, 6, 7]),
        (1, vec![8, 9, 10, 11]),
        (1, vec![12, 13, 14, 15]),
    ]);
    let candidates = host
        .performance_pinning_candidates(&Topology::new(2, 2, 1, 1))
        .expect("enumerate NUMA-aware exclusive placements");
    assert!(
        candidates
            .iter()
            .all(|candidate| candidate.llc_mode == LlcLockMode::Exclusive),
    );
    let mut llcs = candidates
        .iter()
        .map(|candidate| candidate.plan.llc_indices.clone())
        .collect::<Vec<_>>();
    llcs.sort();
    assert_eq!(llcs, vec![vec![0, 2], vec![1, 3]]);

    let footprints = candidates
        .iter()
        .map(|candidate| {
            candidate
                .plan
                .assignments
                .iter()
                .map(|&(_, cpu)| cpu)
                .chain(candidate.plan.service_cpu)
                .collect::<std::collections::BTreeSet<_>>()
        })
        .collect::<Vec<_>>();
    assert!(footprints[0].is_disjoint(&footprints[1]));
}

#[test]
fn performance_grain_candidates_cover_every_llc_window_and_block() {
    let host = multi_llc_host(2, 36);
    let candidates = host
        .performance_pinning_candidates(&Topology::new(1, 1, 2, 1))
        .expect("enumerate offset-aware grain placements");
    assert!(
        candidates
            .iter()
            .all(|candidate| candidate.llc_mode == LlcLockMode::Shared),
    );
    let per_llc = candidates.iter().fold(
        std::collections::BTreeMap::<usize, std::collections::BTreeSet<Vec<usize>>>::new(),
        |mut footprints, candidate| {
            assert_eq!(candidate.plan.llc_indices.len(), 1);
            footprints
                .entry(candidate.plan.llc_indices[0])
                .or_default()
                .insert(candidate.cpu_reservations.clone());
            footprints
        },
    );
    assert_eq!(candidates.len(), 24);
    for llc in 0..2 {
        let base = llc * 36;
        let expected = (0..12)
            .map(|grain| {
                let start = base + grain * 3;
                vec![start, start + 1, start + 2]
            })
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            per_llc[&llc], expected,
            "each 36-CPU LLC must expose exactly twelve pairwise-disjoint \
             vCPU+service grains",
        );
    }
}

#[test]
fn multi_llc_performance_candidates_retain_twelve_disjoint_grains() {
    let host = multi_llc_host(2, 36);
    let candidates = host
        .performance_pinning_candidates(&Topology::new(1, 2, 2, 1))
        .expect("enumerate two-LLC performance grains");
    assert!(
        candidates
            .iter()
            .all(|candidate| candidate.llc_mode == LlcLockMode::Shared)
    );

    let footprints = candidates
        .iter()
        .map(|candidate| {
            candidate
                .cpu_reservations
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>()
        })
        .collect::<std::collections::BTreeSet<_>>();
    let mut selected = Vec::new();
    for grain in 0..12 {
        let low = grain * 3;
        let high = 36 + grain * 3;
        let service_low = std::collections::BTreeSet::from([low, low + 1, low + 2, high, high + 1]);
        let service_high =
            std::collections::BTreeSet::from([low, low + 1, high, high + 1, high + 2]);
        let footprint = if footprints.contains(&service_low) {
            service_low
        } else {
            assert!(
                footprints.contains(&service_high),
                "missing grain {grain} with service CPU in either mapped LLC",
            );
            service_high
        };
        assert!(
            selected
                .iter()
                .all(|prior: &std::collections::BTreeSet<usize>| prior.is_disjoint(&footprint)),
            "static grain {grain} overlaps an earlier material footprint",
        );
        selected.push(footprint);
    }
    assert_eq!(selected.len(), 12);
}
