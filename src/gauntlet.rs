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
    /// Memory budget for this preset's verifier VM.
    ///
    /// This caps the topology-derived 64 MiB/vCPU floor; deferred initramfs
    /// sizing may still raise the actual guest allocation when the test
    /// binary, kernel, or boot modules require more memory.
    pub memory_mib: usize,
    /// Forced no-perf host-CPU budget for cells running this preset.
    /// `None` (every stock preset) leaves budget resolution to the
    /// normal path (test's `cpu_budget`, else auto-size to vCPU count plus
    /// one service CPU, clamped to the allowed cpuset).
    /// `Some(n)` pins the no-perf CPU mask to `min(n, allowed host CPUs)`,
    /// so a preset whose vCPU count exceeds `n` ALWAYS time-slices
    /// (deliberate, continuous overcommit). On a host with fewer than `n`
    /// allowed CPUs the budget collapses to the allowed set — deeper
    /// overcommit, never a skip or hard error (a forced budget exists to
    /// force overcommit; a smaller host just forces more of it). The
    /// admission preflight and the verifier cell path apply the same
    /// clamp. Consumed by the
    /// verifier cell path ([`crate::verifier::collect_verifier_output`]).
    /// The only preset that sets it (192cpu-11llc-smt) is also non-uniform,
    /// hence verifier-only (the gauntlet path skips `llc_cores` presets —
    /// see `for_each_gauntlet_variant`), so the forced budget is realized
    /// exclusively in the verifier battery. A future uniform forced-budget
    /// preset would additionally need the gauntlet variant path to honor
    /// this field.
    pub forced_cpu_budget: Option<u32>,
}

/// Topology presets used by gauntlet mode.
///
/// Covers topologies from `4cpu-1llc-nosmt` (4 CPUs) up through the
/// `252cpu-14llc-smt` / `252cpu-14llc-nosmt` presets (252 CPUs,
/// near the KVM vCPU limit), plus `192cpu-11llc-smt` (192 CPUs across 11
/// NON-uniform LLCs), spanning SMT, non-SMT (`-nosmt`), and multi-NUMA
/// (`2numa-*`, `4numa-*`) families. The stock presets are built from
/// the `defs` / `numa_defs` tuple tables; `192cpu-11llc-smt` is appended
/// explicitly because it carries per-LLC core counts and a forced CPU
/// budget the uniform tuple shape cannot express.
///
/// `192cpu-11llc-smt` targets schedulers whose per-LLC math assumes
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
/// non-SMT scale presets keep ARM64's topology scale
/// coverage). The multi-NUMA presets are *not* filtered here: the
/// default [`crate::test_support::TopologyConstraints`]
/// (`max_numa_nodes: Some(1)`) excludes them at test-selection time
/// (via `accepts` / `accepts_no_perf_mode`) unless a test raises
/// the bound.
pub fn gauntlet_presets() -> Vec<TopoPreset> {
    let defs: &[(&str, &str, u32, u32, u32, usize)] = &[
        ("4cpu-1llc-nosmt", "4 CPUs, 1 LLC", 1, 4, 1, 2048),
        ("4cpu-2llc-nosmt", "4 CPUs, 2 LLCs", 2, 2, 1, 2048),
        ("9cpu-3llc-nosmt", "9 CPUs, 3 LLCs (odd)", 3, 3, 1, 2048),
        ("15cpu-5llc-nosmt", "15 CPUs, 5 LLCs (prime)", 5, 3, 1, 2048),
        ("14cpu-7llc-nosmt", "14 CPUs, 7 LLCs (prime)", 7, 2, 1, 2048),
        ("8cpu-2llc-smt", "8 CPUs, 2 LLCs with SMT", 2, 2, 2, 2048),
        ("12cpu-3llc-smt", "12 CPUs, 3 LLCs with SMT", 3, 2, 2, 2048),
        ("32cpu-4llc-smt", "32 CPUs, 4 LLCs", 4, 4, 2, 2048),
        ("64cpu-8llc-smt", "64 CPUs, 8 LLCs", 8, 4, 2, 2048),
        ("128cpu-4llc-smt", "128 CPUs, 4 LLCs", 4, 16, 2, 2048),
        ("128cpu-8llc-smt", "128 CPUs, 8 LLCs", 8, 8, 2, 2048),
        (
            "240cpu-15llc-smt",
            "240 CPUs, 15 LLCs with SMT",
            15,
            8,
            2,
            2048,
        ),
        (
            "252cpu-14llc-smt",
            "252 CPUs, 14 LLCs (near KVM vCPU limit)",
            14,
            9,
            2,
            4096,
        ),
        // Non-SMT scale presets for ARM64 coverage.
        // These also run on x86_64 to test non-SMT topologies at scale.
        (
            "32cpu-4llc-nosmt",
            "32 CPUs, 4 LLCs (no SMT)",
            4,
            8,
            1,
            2048,
        ),
        (
            "64cpu-8llc-nosmt",
            "64 CPUs, 8 LLCs (no SMT)",
            8,
            8,
            1,
            2048,
        ),
        (
            "128cpu-4llc-nosmt",
            "128 CPUs, 4 LLCs (no SMT)",
            4,
            32,
            1,
            2048,
        ),
        (
            "128cpu-8llc-nosmt",
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
            "2numa-32cpu-2llc-smt",
            "32 CPUs, 2 NUMA nodes, 2 LLCs (one LLC per node, SMT)",
            2,
            2,
            8,
            2,
            2048,
        ),
        (
            "2numa-16cpu-4llc-nosmt",
            "16 CPUs, 2 NUMA nodes, 4 LLCs",
            2,
            4,
            4,
            1,
            2048,
        ),
        (
            "2numa-128cpu-8llc-smt",
            "128 CPUs, 2 NUMA nodes, 8 LLCs",
            2,
            8,
            8,
            2,
            2048,
        ),
        (
            "2numa-128cpu-8llc-nosmt",
            "128 CPUs, 2 NUMA nodes, 8 LLCs (no SMT)",
            2,
            8,
            16,
            1,
            2048,
        ),
        (
            "4numa-32cpu-8llc-nosmt",
            "32 CPUs, 4 NUMA nodes, 8 LLCs",
            4,
            8,
            4,
            1,
            2048,
        ),
        (
            "4numa-192cpu-12llc-smt",
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

    // 192cpu-11llc-smt: 11 LLCs, ten with 9 cores (18 logical CPUs) and one
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
        name: "192cpu-11llc-smt",
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
            ("4cpu-1llc-nosmt", 1, 4),
            ("4cpu-2llc-nosmt", 2, 4),
            ("9cpu-3llc-nosmt", 3, 9),
            ("15cpu-5llc-nosmt", 5, 15),
            ("14cpu-7llc-nosmt", 7, 14),
            #[cfg(not(target_arch = "aarch64"))]
            ("8cpu-2llc-smt", 2, 8),
            #[cfg(not(target_arch = "aarch64"))]
            ("12cpu-3llc-smt", 3, 12),
            #[cfg(not(target_arch = "aarch64"))]
            ("32cpu-4llc-smt", 4, 32),
            #[cfg(not(target_arch = "aarch64"))]
            ("64cpu-8llc-smt", 8, 64),
            #[cfg(not(target_arch = "aarch64"))]
            ("128cpu-4llc-smt", 4, 128),
            #[cfg(not(target_arch = "aarch64"))]
            ("128cpu-8llc-smt", 8, 128),
            #[cfg(not(target_arch = "aarch64"))]
            ("240cpu-15llc-smt", 15, 240),
            #[cfg(not(target_arch = "aarch64"))]
            ("252cpu-14llc-smt", 14, 252),
            #[cfg(not(target_arch = "aarch64"))]
            ("192cpu-11llc-smt", 11, 192),
            ("32cpu-4llc-nosmt", 4, 32),
            ("64cpu-8llc-nosmt", 8, 64),
            ("128cpu-4llc-nosmt", 4, 128),
            ("128cpu-8llc-nosmt", 8, 128),
            ("240cpu-15llc-nosmt", 15, 240),
            ("252cpu-14llc-nosmt", 14, 252),
            ("2numa-16cpu-4llc-nosmt", 4, 16),
            #[cfg(not(target_arch = "aarch64"))]
            ("2numa-32cpu-2llc-smt", 2, 32),
            #[cfg(not(target_arch = "aarch64"))]
            ("2numa-128cpu-8llc-smt", 8, 128),
            ("2numa-128cpu-8llc-nosmt", 8, 128),
            ("4numa-32cpu-8llc-nosmt", 8, 32),
            #[cfg(not(target_arch = "aarch64"))]
            ("4numa-192cpu-12llc-smt", 12, 192),
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
    fn gauntlet_preset_names_encode_topology() {
        for p in &gauntlet_presets() {
            let smt = if p.topology.threads_per_core > 1 {
                "smt"
            } else {
                "nosmt"
            };
            let expected = if p.topology.numa_nodes > 1 {
                format!(
                    "{}numa-{}cpu-{}llc-{smt}",
                    p.topology.numa_nodes,
                    p.topology.total_cpus(),
                    p.topology.num_llcs(),
                )
            } else {
                format!(
                    "{}cpu-{}llc-{smt}",
                    p.topology.total_cpus(),
                    p.topology.num_llcs(),
                )
            };
            assert_eq!(p.name, expected, "preset name must encode its topology");
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
    fn gauntlet_presets_forced_budget_only_on_192cpu_11llc_smt() {
        // Regression pin: the forced-overcommit budget is opt-in per
        // preset. Exactly `192cpu-11llc-smt` carries it; every stock preset
        // leaves it `None` so their cell budgets resolve unchanged.
        for p in &gauntlet_presets() {
            match p.name {
                "192cpu-11llc-smt" => assert_eq!(
                    p.forced_cpu_budget,
                    Some(96),
                    "192cpu-11llc-smt must force a 96-CPU budget"
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
    fn gauntlet_presets_192cpu_11llc_smt_is_non_uniform_and_overcommits() {
        let p = gauntlet_presets()
            .into_iter()
            .find(|p| p.name == "192cpu-11llc-smt")
            .expect("192cpu-11llc-smt present on x86_64");
        let t = &p.topology;
        t.validate().expect("192cpu-11llc-smt must validate");
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
