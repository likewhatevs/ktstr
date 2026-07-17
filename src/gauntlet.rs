//! Gauntlet topology presets.
//!
//! See the [Gauntlet](https://ktstr.dev/guide/running-tests/gauntlet.html)
//! chapter of the guide.

use crate::vmm::Topology;

/// A gauntlet topology preset.
///
/// Each preset defines a specific CPU topology for matrix testing.
/// See [`gauntlet_presets()`] for the full set.
pub struct TopoPreset {
    pub name: &'static str,
    /// Human-readable description; read by preset-audit tests only.
    #[allow(dead_code)]
    pub description: &'static str,
    pub topology: Topology,
    /// Memory budget for this preset's VM; read by preset-audit tests only.
    #[allow(dead_code)]
    pub memory_mib: usize,
    /// Forced no-perf host-CPU budget for cells running this preset.
    /// `None` (every stock preset) leaves budget resolution to the
    /// normal path (test's `cpu_budget`, else auto-size to vCPU count).
    /// `Some(n)` pins the no-perf CPU mask to `n` host CPUs regardless
    /// of host size, so a preset whose vCPU count exceeds `n` ALWAYS
    /// time-slices (deliberate, continuous overcommit). Consumed by the
    /// verifier cell path ([`crate::verifier::collect_verifier_output`]).
    /// The only preset that sets it (uneven-11llc) is also non-uniform,
    /// hence verifier-only (the gauntlet path skips `llc_cores` presets —
    /// see `for_each_gauntlet_variant`), so the forced budget is realized
    /// exclusively in the verifier battery. A future uniform forced-budget
    /// preset would additionally need the gauntlet variant path to honor
    /// this field.
    pub forced_cpu_budget: Option<u32>,
}

/// Topology presets used by gauntlet mode.
///
/// Covers topologies from `tiny-1llc` (4 CPUs) up through the
/// `252cpu-14llc-smt2` / `252cpu-14llc-nosmt` presets (252 CPUs,
/// near the KVM vCPU limit), plus `uneven-11llc` (192 CPUs across 11
/// NON-uniform LLCs), spanning SMT, non-SMT (`-nosmt`), and multi-NUMA
/// (`numa2-*`, `numa4-*`) families. The stock presets are built from
/// the `defs` / `numa_defs` tuple tables; `uneven-11llc` is appended
/// explicitly because it carries per-LLC core counts and a forced CPU
/// budget the uniform tuple shape cannot express.
///
/// `uneven-11llc` targets schedulers whose per-LLC math assumes
/// equal-sized caches: ten LLCs of 18 logical CPUs and one of 12
/// (`llc_cores = [9;10] + [6]`, packing width 9). It is VERIFIER-ONLY —
/// the gauntlet execution path reconstructs topology from a uniform
/// `TopoOverride` that cannot express `llc_cores`, so
/// `for_each_gauntlet_variant` skips non-uniform presets. The verifier
/// reaches it via
/// [`crate::test_support::TopologyConstraints::accepts_verifier`], which
/// ignores default caps and imposes NO host-size bound. Its
/// `forced_cpu_budget = Some(96)` guarantees continuous overcommit (192
/// vCPUs time-slicing over <=96 host CPUs) every time the battery runs.
///
/// The full set is returned unconditionally; the only filter applied
/// here is the aarch64 retain below, which drops SMT presets
/// (`threads_per_core > 1`) because ARM64 CPUs have no SMT (the
/// non-SMT medium/large/max presets keep ARM64's topology scale
/// coverage). The multi-NUMA presets are *not* filtered here: the
/// default [`crate::test_support::TopologyConstraints`]
/// (`max_numa_nodes: Some(1)`) excludes them at test-selection time
/// (via `accepts` / `accepts_no_perf_mode`) unless a test raises
/// the bound.
pub fn gauntlet_presets() -> Vec<TopoPreset> {
    let defs: &[(&str, &str, u32, u32, u32, usize)] = &[
        ("tiny-1llc", "4 CPUs, 1 LLC", 1, 4, 1, 2048),
        ("tiny-2llc", "4 CPUs, 2 LLCs", 2, 2, 1, 2048),
        ("odd-3llc", "9 CPUs, 3 LLCs (odd)", 3, 3, 1, 2048),
        ("odd-5llc", "15 CPUs, 5 LLCs (prime)", 5, 3, 1, 2048),
        ("odd-7llc", "14 CPUs, 7 LLCs (prime)", 7, 2, 1, 2048),
        ("smt-2llc", "8 CPUs, 2 LLCs with SMT", 2, 2, 2, 2048),
        ("smt-3llc", "12 CPUs, 3 LLCs with SMT", 3, 2, 2, 2048),
        ("medium-4llc", "32 CPUs, 4 LLCs", 4, 4, 2, 2048),
        ("medium-8llc", "64 CPUs, 8 LLCs", 8, 4, 2, 2048),
        ("large-4llc", "128 CPUs, 4 LLCs", 4, 16, 2, 2048),
        ("large-8llc", "128 CPUs, 8 LLCs", 8, 8, 2, 2048),
        (
            "240cpu-15llc-smt2",
            "240 CPUs, 15 LLCs with SMT",
            15,
            8,
            2,
            2048,
        ),
        (
            "252cpu-14llc-smt2",
            "252 CPUs, 14 LLCs (near KVM vCPU limit)",
            14,
            9,
            2,
            4096,
        ),
        // Non-SMT medium/large/scale presets for ARM64 coverage.
        // These also run on x86_64 to test non-SMT topologies at scale.
        (
            "medium-4llc-nosmt",
            "32 CPUs, 4 LLCs (no SMT)",
            4,
            8,
            1,
            2048,
        ),
        (
            "medium-8llc-nosmt",
            "64 CPUs, 8 LLCs (no SMT)",
            8,
            8,
            1,
            2048,
        ),
        (
            "large-4llc-nosmt",
            "128 CPUs, 4 LLCs (no SMT)",
            4,
            32,
            1,
            2048,
        ),
        (
            "large-8llc-nosmt",
            "128 CPUs, 8 LLCs (no SMT)",
            8,
            16,
            1,
            2048,
        ),
        (
            "240cpu-15llc-nosmt",
            "240 CPUs, 15 LLCs (no SMT)",
            15,
            16,
            1,
            2048,
        ),
        (
            "252cpu-14llc-nosmt",
            "252 CPUs, 14 LLCs (no SMT, near KVM vCPU limit)",
            14,
            18,
            1,
            4096,
        ),
    ];
    let numa_defs: &[(&str, &str, u32, u32, u32, u32, usize)] = &[
        // One LLC per node, SMT. The 2 total LLCs split evenly across
        // the 2 nodes, so a scheduler that conflates node identity with
        // LLC identity has nowhere to hide the confusion. Excluded from
        // the default battery by `max_numa_nodes = Some(1)`; reached by
        // the verifier under `accepts_verifier` (default numa cap
        // ignored).
        (
            "numa2-2llc",
            "32 CPUs, 2 NUMA nodes, 2 LLCs (one LLC per node, SMT)",
            2,
            2,
            8,
            2,
            2048,
        ),
        (
            "numa2-4llc",
            "16 CPUs, 2 NUMA nodes, 4 LLCs",
            2,
            4,
            4,
            1,
            2048,
        ),
        (
            "numa2-8llc",
            "128 CPUs, 2 NUMA nodes, 8 LLCs",
            2,
            8,
            8,
            2,
            2048,
        ),
        (
            "numa2-8llc-nosmt",
            "128 CPUs, 2 NUMA nodes, 8 LLCs (no SMT)",
            2,
            8,
            16,
            1,
            2048,
        ),
        (
            "numa4-8llc",
            "32 CPUs, 4 NUMA nodes, 8 LLCs",
            4,
            8,
            4,
            1,
            2048,
        ),
        (
            "numa4-12llc",
            "192 CPUs, 4 NUMA nodes, 12 LLCs",
            4,
            12,
            8,
            2,
            4096,
        ),
    ];

    let mut presets: Vec<TopoPreset> = defs
        .iter()
        .map(|&(n, d, s, c, t, m)| TopoPreset {
            name: n,
            description: d,
            topology: Topology {
                llcs: s,
                cores_per_llc: c,
                threads_per_core: t,
                numa_nodes: 1,
                nodes: None,
                distances: None,
                llc_cores: None,
            },
            memory_mib: m,
            forced_cpu_budget: None,
        })
        .chain(numa_defs.iter().map(|&(n, d, nn, s, c, t, m)| TopoPreset {
            name: n,
            description: d,
            topology: Topology {
                llcs: s,
                cores_per_llc: c,
                threads_per_core: t,
                numa_nodes: nn,
                nodes: None,
                distances: None,
                llc_cores: None,
            },
            memory_mib: m,
            forced_cpu_budget: None,
        }))
        .collect();

    // uneven-11llc: 11 LLCs, ten with 9 cores (18 logical CPUs) and one
    // with 6 (12 logical CPUs) = 192 CPUs. NON-uniform LLC sizing (the
    // point): schedulers that assume equal-sized LLCs mis-place work on
    // the short cache. Packing width `cores_per_llc = 9` reserves a
    // 9-core (32-APIC-ID) block per LLC; `llc_cores` populates ten fully
    // and one to 6, so the guest sees the uneven layout via the sparse
    // APIC-block mechanism. `forced_cpu_budget = Some(96)` pins the
    // no-perf mask to <=96 host CPUs so its 192 vCPUs ALWAYS overcommit
    // (>=2x) — continuous exercise of the time-slicing path, modeling a
    // chip larger than the host. Appended out-of-band (the tuple tables
    // express only uniform shapes with no forced budget).
    static UNEVEN_11LLC_CORES: [u32; 11] = [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 6];
    presets.push(TopoPreset {
        name: "uneven-11llc",
        description: "192 CPUs, 11 LLCs (uneven: ten 18-CPU + one 12-CPU, SMT)",
        topology: Topology {
            llcs: 11,
            cores_per_llc: 9,
            threads_per_core: 2,
            numa_nodes: 1,
            nodes: None,
            distances: None,
            llc_cores: Some(&UNEVEN_11LLC_CORES),
        },
        memory_mib: 4096,
        forced_cpu_budget: Some(96),
    });

    // ARM64 has no SMT -- exclude presets with threads_per_core > 1.
    if cfg!(target_arch = "aarch64") {
        presets.retain(|p| p.topology.threads_per_core <= 1);
    }

    presets
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gauntlet_presets_unique_names() {
        let p = gauntlet_presets();
        let names: Vec<&str> = p.iter().map(|p| p.name).collect();
        let unique: std::collections::HashSet<&&str> = names.iter().collect();
        assert_eq!(names.len(), unique.len());
    }

    #[test]
    fn gauntlet_presets_total_cpus_match() {
        for p in &gauntlet_presets() {
            let cpus = p.topology.total_cpus();
            assert!(
                p.description.contains(&cpus.to_string()),
                "{}: description '{}' doesn't mention {} CPUs",
                p.name,
                p.description,
                cpus
            );
        }
    }

    #[test]
    fn gauntlet_presets_memory_sane() {
        for p in &gauntlet_presets() {
            assert!(
                p.memory_mib >= 512,
                "{} has too little memory: {}MiB",
                p.name,
                p.memory_mib
            );
            let cpus = p.topology.total_cpus() as usize;
            assert!(
                p.memory_mib >= cpus * 8,
                "{} has {}MiB for {} CPUs",
                p.name,
                p.memory_mib,
                cpus
            );
        }
    }

    #[test]
    fn gauntlet_presets_topology_pinned() {
        // (name, expected LLCs, expected total CPUs)
        let expected: &[(&str, u32, u32)] = &[
            ("tiny-1llc", 1, 4),
            ("tiny-2llc", 2, 4),
            ("odd-3llc", 3, 9),
            ("odd-5llc", 5, 15),
            ("odd-7llc", 7, 14),
            #[cfg(not(target_arch = "aarch64"))]
            ("smt-2llc", 2, 8),
            #[cfg(not(target_arch = "aarch64"))]
            ("smt-3llc", 3, 12),
            #[cfg(not(target_arch = "aarch64"))]
            ("medium-4llc", 4, 32),
            #[cfg(not(target_arch = "aarch64"))]
            ("medium-8llc", 8, 64),
            #[cfg(not(target_arch = "aarch64"))]
            ("large-4llc", 4, 128),
            #[cfg(not(target_arch = "aarch64"))]
            ("large-8llc", 8, 128),
            #[cfg(not(target_arch = "aarch64"))]
            ("240cpu-15llc-smt2", 15, 240),
            #[cfg(not(target_arch = "aarch64"))]
            ("252cpu-14llc-smt2", 14, 252),
            #[cfg(not(target_arch = "aarch64"))]
            ("uneven-11llc", 11, 192),
            ("medium-4llc-nosmt", 4, 32),
            ("medium-8llc-nosmt", 8, 64),
            ("large-4llc-nosmt", 4, 128),
            ("large-8llc-nosmt", 8, 128),
            ("240cpu-15llc-nosmt", 15, 240),
            ("252cpu-14llc-nosmt", 14, 252),
            ("numa2-4llc", 4, 16),
            #[cfg(not(target_arch = "aarch64"))]
            ("numa2-2llc", 2, 32),
            #[cfg(not(target_arch = "aarch64"))]
            ("numa2-8llc", 8, 128),
            ("numa2-8llc-nosmt", 8, 128),
            ("numa4-8llc", 8, 32),
            #[cfg(not(target_arch = "aarch64"))]
            ("numa4-12llc", 12, 192),
        ];
        let presets = gauntlet_presets();
        assert_eq!(
            expected.len(),
            presets.len(),
            "pinned list and preset list have different lengths"
        );
        for &(name, llcs, cpus) in expected {
            let p = presets.iter().find(|p| p.name == name).unwrap();
            assert_eq!(
                p.topology.num_llcs(),
                llcs,
                "{}: expected {} LLCs, got {}",
                name,
                llcs,
                p.topology.num_llcs()
            );
            assert_eq!(
                p.topology.total_cpus(),
                cpus,
                "{}: expected {} CPUs, got {}",
                name,
                cpus,
                p.topology.total_cpus()
            );
        }
    }

    #[test]
    fn gauntlet_presets_topology_valid() {
        for p in &gauntlet_presets() {
            p.topology
                .validate()
                .unwrap_or_else(|e| panic!("{}: {e}", p.name));
        }
    }

    #[test]
    fn gauntlet_presets_252_cpu_near_limit() {
        let presets = gauntlet_presets();
        let max_presets: Vec<_> = presets
            .iter()
            .filter(|p| p.name.starts_with("252cpu-14llc-"))
            .collect();
        assert!(
            !max_presets.is_empty(),
            "at least one 252cpu-14llc preset must exist"
        );
        for p in &max_presets {
            let cpus = p.topology.total_cpus();
            assert!(
                cpus <= 255,
                "{} has {} CPUs, exceeds KVM vCPU limit",
                p.name,
                cpus
            );
            assert!(
                cpus >= 200,
                "{} should be near the limit: {} CPUs",
                p.name,
                cpus
            );
        }
    }

    #[test]
    fn topology_single_cpu() {
        let t = Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
            llc_cores: None,
        };
        assert_eq!(t.total_cpus(), 1);
        assert_eq!(t.num_llcs(), 1);
    }

    #[test]
    #[cfg(not(target_arch = "aarch64"))]
    fn gauntlet_presets_smt_presets_have_threads() {
        let presets = gauntlet_presets();
        for p in &presets {
            if p.name.starts_with("smt-") {
                assert_eq!(
                    p.topology.threads_per_core, 2,
                    "{} should have 2 threads per core",
                    p.name
                );
            }
        }
    }

    #[test]
    fn gauntlet_presets_odd_presets_are_odd() {
        let presets = gauntlet_presets();
        for p in &presets {
            if p.name.starts_with("odd-") {
                assert!(
                    p.topology.llcs % 2 != 0,
                    "{}: odd-* presets must have odd LLC count, got {} LLCs",
                    p.name,
                    p.topology.llcs
                );
            }
        }
    }

    #[test]
    fn gauntlet_presets_numa_presets_have_correct_nodes() {
        for p in &gauntlet_presets() {
            if p.name.starts_with("numa2") {
                assert_eq!(
                    p.topology.numa_nodes, 2,
                    "{}: expected 2 NUMA nodes",
                    p.name
                );
            } else if p.name.starts_with("numa4") {
                assert_eq!(
                    p.topology.numa_nodes, 4,
                    "{}: expected 4 NUMA nodes",
                    p.name
                );
            }
        }
    }

    #[test]
    fn gauntlet_presets_description_non_empty() {
        for p in &gauntlet_presets() {
            assert!(
                !p.description.is_empty(),
                "{} has empty description",
                p.name
            );
        }
    }

    #[test]
    fn gauntlet_presets_forced_budget_only_on_uneven_11llc() {
        // Regression pin: the forced-overcommit budget is opt-in per
        // preset. Exactly `uneven-11llc` carries it; every stock preset
        // leaves it `None` so their cell budgets resolve unchanged.
        for p in &gauntlet_presets() {
            match p.name {
                "uneven-11llc" => assert_eq!(
                    p.forced_cpu_budget,
                    Some(96),
                    "uneven-11llc must force a 96-CPU budget"
                ),
                other => assert_eq!(
                    p.forced_cpu_budget, None,
                    "{other} must not carry a forced budget"
                ),
            }
        }
    }

    #[test]
    #[cfg(not(target_arch = "aarch64"))]
    fn gauntlet_presets_uneven_11llc_is_non_uniform_and_overcommits() {
        let p = gauntlet_presets()
            .into_iter()
            .find(|p| p.name == "uneven-11llc")
            .expect("uneven-11llc present on x86_64");
        let t = &p.topology;
        t.validate().expect("uneven-11llc must validate");
        assert_eq!(t.num_llcs(), 11);
        assert_eq!(t.total_cpus(), 192);
        // Ten 18-CPU LLCs + one 12-CPU LLC: NOT uniform.
        let sizes: Vec<u32> = (0..t.num_llcs())
            .map(|l| t.cores_in_llc(l) * t.threads_per_core)
            .collect();
        assert_eq!(sizes.iter().filter(|&&s| s == 18).count(), 10);
        assert_eq!(sizes.iter().filter(|&&s| s == 12).count(), 1);
        // Forced budget guarantees overcommit: budget < vCPU count.
        assert!(
            p.forced_cpu_budget.unwrap() < t.total_cpus(),
            "forced budget must be below vCPU count to force overcommit"
        );
    }
}
