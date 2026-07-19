//! Host CPU topology discovery for performance_mode.
//!
//! Wraps [`TestTopology`](crate::topology::TestTopology) for LLC-aware
//! vCPU pinning and host resource validation.

use anyhow::{Context, Result};
use std::sync::atomic::{AtomicBool, Ordering};

// Advisory flock primitives live in `crate::flock` so both LLC +
// per-CPU coordination here and per-cache-entry coordination in
// `crate::cache` share one `try_flock` implementation (with a single
// `O_CLOEXEC` source of truth) plus one `HolderInfo` /proc/locks
// parser. Re-importing the names keeps existing in-module call sites
// (production + `super::*` tests) compiling unchanged.
use crate::flock::{FlockMode, try_flock};

// Cross-invocation acquisition protocol: the ticket queue, the head
// license + claim visibility, the inotify wait, and lifecycle-bound
// blocking. See protocol.rs's module doc for the full model.
pub(crate) mod protocol;

/// Resource contention error — LLC slots or CPUs unavailable.
/// Downcast via `anyhow::Error::downcast_ref::<ResourceContention>()`
/// to distinguish from fatal errors.
#[derive(Debug)]
pub struct ResourceContention {
    pub reason: String,
}

impl std::fmt::Display for ResourceContention {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.reason)
    }
}

impl std::error::Error for ResourceContention {}

/// The requested topology cannot be realized on this host, and no retry
/// changes that. Surfaced as a SKIP by the x86_64 VM-creation caps (guest
/// RAM top above the host MAXPHYADDR, vCPU count above KVM_CAP_MAX_VCPUS,
/// or max APIC id at/above KVM_CAP_MAX_VCPU_ID): these fire for ANY VM of
/// this shape, perf-mode or not, so the test cannot run here. Also
/// returned by the `performance_mode` planner (`compute_pinning`) when the
/// host has too few physical CPUs / LLC groups — but that perf-mode caller
/// RE-MAPS it to [`PerfModeUnavailable`] (a host-insufficiency: skip by
/// default, fail under `KTSTR_NO_SKIP_MODE`). Also raised by
/// `resolve_cpu_budget` when an author's per-test `cpu_budget` exceeds the
/// allowed-CPU count — the author-attribute half of a provenance split (a
/// capability requirement a bigger host satisfies → skip), mirroring the
/// operator-knob half [`CpuBudgetUnsatisfiable`] (a concrete `--cpu-cap`
/// number the host cannot satisfy → hard fail). Distinct
/// from [`ResourceContention`] (a transient slot/resource shortage a retry
/// resolves → skip); a too-small host is permanent, so the operator must
/// provision different hardware or narrow the topology rather than retry.
///
/// Downcast via `anyhow::Error::downcast_ref::<TopologyInsufficient>()`
/// (chain-aware: the `#[ktstr_test]` dispatch and `skip_on_contention!`
/// walk the full error chain so a `.context(...)`-wrapped instance is
/// still recognised). This typed error replaced a fragile message
/// string-match (`"need"` + `"LLC"`/`"CPU"`) that would misclassify any
/// unrelated error happening to contain those words.
#[derive(Debug)]
pub struct TopologyInsufficient {
    pub reason: String,
}

impl std::fmt::Display for TopologyInsufficient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.reason)
    }
}

impl std::error::Error for TopologyInsufficient {}

/// The host cannot honor the `performance_mode` guarantee (an exclusive
/// host LLC for the test's virtual LLC topology + a service CPU), and no
/// retry changes that — a permanent host-insufficiency (e.g. a single-LLC
/// host whose LLC spans every CPU, so LLC + 1 service never fits). Treated
/// like [`TopologyInsufficient`] / [`ResourceContention`]: a SKIP by
/// default (the VM never runs unisolated — it errors at build, so a
/// visible skip informs the operator without reddening CI on a host that
/// can never satisfy perf-mode), promoted to a hard FAIL under
/// `KTSTR_NO_SKIP_MODE` for runs that demand perf-mode execution. The
/// remedy is unchanged: provision a host with a spare LLC/CPU, narrow the
/// topology, or drop `--perf-mode`.
///
/// Downcast via `anyhow::Error::downcast_ref::<PerfModeUnavailable>()`
/// (chain-aware: the dispatch + macro predicates walk the full error
/// chain, so a `.context(...)`-wrapped instance is still recognised).
#[derive(Debug)]
pub struct PerfModeUnavailable {
    pub reason: String,
}

impl std::fmt::Display for PerfModeUnavailable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.reason)
    }
}

impl std::error::Error for PerfModeUnavailable {}

/// An operator `--cpu-cap N` (or `KTSTR_CPU_CAP`) the host cannot satisfy: N
/// exceeds the CPUs this process is allowed on. A HARD ERROR, not a skip —
/// the operator typed a concrete number that does not exist on this host (a
/// user-input error). This is the OPERATOR-knob half of a provenance split:
/// an author's per-test `cpu_budget` over the allowance is instead a
/// [`TopologyInsufficient`] SKIP (a capability request a bigger host would
/// satisfy), raised in `resolve_cpu_budget`. Contrast [`ResourceContention`]
/// (a transient shortage of an otherwise-satisfiable budget → skip/retry).
///
/// Downcast via `anyhow::Error::downcast_ref::<CpuBudgetUnsatisfiable>()`
/// (chain-aware).
#[derive(Debug)]
pub struct CpuBudgetUnsatisfiable {
    pub reason: String,
}

impl std::fmt::Display for CpuBudgetUnsatisfiable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.reason)
    }
}

impl std::error::Error for CpuBudgetUnsatisfiable {}

/// The requested topology cannot be represented by this VMM's static
/// device layout, and the limit is host-INDEPENDENT, so no retry and no
/// different host changes that. Concretely: the aarch64 vCPU count
/// exceeds `MAX_VCPUS` (the capacity of the statically sized GICv3
/// redistributor MMIO window) — with more vCPUs the redistributor region
/// overruns the device MMIO window and shadows serial/virtio.
///
/// A HARD ERROR — distinct from [`TopologyInsufficient`], its deliberate
/// counterpart. `TopologyInsufficient` is host-DEPENDENT (the VM cannot
/// boot on *this* host, but a bigger host could → skip);
/// `TopologyUnrepresentable` is a fixed VMM-layout limit no aarch64 host
/// can satisfy under this VMM, so it is a test misconfiguration — the
/// author must narrow the topology, not provision different hardware.
/// Routes to `EXIT_FAIL` via a DEDICATED hard-fail arm (the
/// `is_topology_unrepresentable` predicate) in both `result_to_exit_code`
/// and the `#[ktstr_test]` macro body, placed ABOVE the `expect_err`
/// inversion and the skip arms — mirroring `CpuBudgetUnsatisfiable` (the
/// other dedicated hard-fail). That placement is what makes it fail even in
/// an `expect_err` test (the generic `expect_err` arm would otherwise
/// invert it to a pass) and keeps it out of the `skip_on_contention!` /
/// `is_topology_insufficient` skip paths, so the misconfiguration can
/// never masquerade as the expected failure or be turned into a skip.
///
/// Downcast via `anyhow::Error::downcast_ref::<TopologyUnrepresentable>()`
/// (chain-aware: walks `e.chain()`, so a `.context(...)`-wrapped instance
/// is still recognised) to identify it programmatically — e.g. tests
/// asserting the over-`MAX_VCPUS` bail is this hard-fault and not a bare
/// string-matched error.
// Constructed only on aarch64 (the GICv3-layout over-MAX_VCPUS bail in
// aarch64::kvm) and in cross-arch routing tests; a non-aarch64 lib-only
// build sees no construction site. Keep the dead-code check live on
// aarch64 (where the bail MUST construct it — a real regression if it
// stops) and allow it only off-arch.
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
#[derive(Debug)]
pub struct TopologyUnrepresentable {
    pub reason: String,
}

impl std::fmt::Display for TopologyUnrepresentable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.reason)
    }
}

impl std::error::Error for TopologyUnrepresentable {}

/// A physical LLC group on the host, identified by its cache ID.
#[derive(Debug, Clone)]
pub struct LlcGroup {
    /// CPUs sharing this LLC.
    pub cpus: Vec<usize>,
}

/// Host CPU topology: LLC groups, NUMA nodes, and online CPU set.
#[derive(Debug, Clone)]
pub struct HostTopology {
    /// LLC groups indexed by their order of discovery.
    pub llc_groups: Vec<LlcGroup>,
    /// All online CPUs.
    pub online_cpus: Vec<usize>,
    /// NUMA node ID for each online CPU, indexed by CPU ID.
    /// CPUs not in the map default to node 0.
    pub cpu_to_node: std::collections::HashMap<usize, usize>,
    /// LLC indices grouped by their NUMA node. Memoized at construction
    /// time from `llc_groups + cpu_to_node` so repeated NUMA-aware
    /// placement queries (perf-mode rotation, `--cpu-cap` consolidation
    /// PLAN) don't re-walk every LLC's CPU list on every call. Access
    /// via [`HostTopology::host_llcs_by_numa_node`]. `BTreeMap` (not
    /// `HashMap`) for deterministic iteration order — two ktstr
    /// invocations on the same host MUST produce identical LLC
    /// selections so their ACQUIRE phases converge on the same indices.
    pub(crate) host_node_llcs: std::collections::BTreeMap<usize, Vec<usize>>,
}

/// Pinning plan: maps each vCPU index to a host CPU, plus a dedicated
/// CPU for service threads (monitor, watchdog).
#[derive(Debug)]
pub struct PinningPlan {
    /// vcpu_index -> host_cpu
    pub assignments: Vec<(u32, usize)>,
    /// Dedicated host CPU for monitor/watchdog threads. Set when
    /// `reserve_service_cpu` is true in `compute_pinning`.
    pub service_cpu: Option<usize>,
    /// Host LLC group indices used by this plan, sorted.
    pub llc_indices: Vec<usize>,
    /// Held flock fds for resource reservation. Dropped when the plan
    /// (and the KtstrVm holding it) is dropped, releasing all locks.
    #[allow(dead_code)] // RAII: flock fds released on Drop, not read after construction.
    pub(crate) locks: Vec<std::os::fd::OwnedFd>,
}

impl PinningPlan {
    /// Duplicate the plan's DESCRIPTION (assignments, service CPU,
    /// LLC indices) with an EMPTY lock set. `PinningPlan` cannot be
    /// `Clone` — the fds are RAII lock holders — but candidate plans
    /// (pure `compute_pinning` output, no locks yet) need copying
    /// between the fast path's scan list and the acquired result.
    pub(crate) fn clone_unlocked(&self) -> PinningPlan {
        PinningPlan {
            assignments: self.assignments.clone(),
            service_cpu: self.service_cpu,
            llc_indices: self.llc_indices.clone(),
            locks: Vec::new(),
        }
    }
}

/// Process-wide cache for [`HostTopology::cached`]. Only
/// populated on success — a failed sysfs probe retries on the
/// next call instead of poisoning the cache.
static CACHED_HOST_TOPOLOGY: std::sync::OnceLock<HostTopology> = std::sync::OnceLock::new();

impl HostTopology {
    /// Read host topology from sysfs via [`TestTopology::from_system()`](crate::topology::TestTopology::from_system).
    pub fn from_sysfs() -> Result<Self> {
        let topo = crate::topology::TestTopology::from_system()
            .context("read host topology from sysfs")?;
        let online_cpus = topo.all_cpus().to_vec();
        let llc_groups: Vec<LlcGroup> = topo
            .llcs()
            .iter()
            .map(|llc| LlcGroup {
                cpus: llc.cpus().to_vec(),
            })
            .collect();
        let cpu_to_node: std::collections::HashMap<usize, usize> = topo
            .llcs()
            .iter()
            .flat_map(|llc| llc.cpus().iter().map(|&cpu| (cpu, llc.numa_node())))
            .collect();
        let host_node_llcs = Self::compute_host_node_llcs(&llc_groups, &cpu_to_node);
        Ok(Self {
            llc_groups,
            online_cpus,
            cpu_to_node,
            host_node_llcs,
        })
    }

    /// Return a cached host topology, populating the cache on first
    /// successful call. Failed reads retry on the next call — the
    /// cache only stores success so a transient sysfs issue at
    /// process start doesn't poison every subsequent build().
    pub fn cached() -> Result<Self> {
        if let Some(topo) = CACHED_HOST_TOPOLOGY.get() {
            return Ok(topo.clone());
        }
        let topo = Self::from_sysfs()?;
        let _ = CACHED_HOST_TOPOLOGY.set(topo.clone());
        Ok(topo)
    }

    /// Build a synthetic `HostTopology` from `(cpu_list, node_id)`
    /// pairs for tests. One pair per LLC group; within a pair the
    /// `cpu_list` becomes the group's CPUs and the `node_id` is the
    /// NUMA node every CPU in that group is assigned to.
    /// `online_cpus` is the flattened concatenation of every group's
    /// CPUs in input order; `cpu_to_node` is built by broadcasting
    /// each group's node over its CPUs; `host_node_llcs` goes through
    /// the same [`compute_host_node_llcs`] path production uses, so
    /// tests never diverge from the sysfs-derived memoization.
    ///
    /// Intended for test fixtures that want a deterministic in-memory
    /// topology without stubbing `/sys/devices/system/cpu/*`.
    /// Previously this logic was duplicated across three helper
    /// functions (`synthetic_topo`, `synthetic_topo_numa`,
    /// `synth_host_topo`) — consolidated here so the
    /// `HostTopology` invariant is maintained in one place. The
    /// `#[cfg(test)]` gate keeps the symbol out of release builds.
    #[cfg(test)]
    pub(crate) fn new_for_tests(groups: &[(Vec<usize>, usize)]) -> Self {
        let llc_groups: Vec<LlcGroup> = groups
            .iter()
            .map(|(cpus, _)| LlcGroup { cpus: cpus.clone() })
            .collect();
        let cpu_to_node: std::collections::HashMap<usize, usize> = groups
            .iter()
            .flat_map(|(cpus, node)| cpus.iter().map(move |&cpu| (cpu, *node)))
            .collect();
        let online_cpus: Vec<usize> = groups
            .iter()
            .flat_map(|(cpus, _)| cpus.iter().copied())
            .collect();
        let host_node_llcs = HostTopology::compute_host_node_llcs(&llc_groups, &cpu_to_node);
        HostTopology {
            llc_groups,
            online_cpus,
            cpu_to_node,
            host_node_llcs,
        }
    }

    /// Compute the memoized `host_node_llcs` map from `llc_groups` +
    /// `cpu_to_node`. Uses the same majority-vote NUMA-assignment rule
    /// as [`Self::llc_numa_node`], so the memoized map and the one-off query
    /// method never disagree. Separate fn (not inlined) so
    /// `from_sysfs` and synthetic-test constructors share one path.
    fn compute_host_node_llcs(
        llc_groups: &[LlcGroup],
        cpu_to_node: &std::collections::HashMap<usize, usize>,
    ) -> std::collections::BTreeMap<usize, Vec<usize>> {
        let mut node_llcs: std::collections::BTreeMap<usize, Vec<usize>> =
            std::collections::BTreeMap::new();
        for (idx, group) in llc_groups.iter().enumerate() {
            // Majority-vote NUMA node for this LLC — matches
            // `llc_numa_node` exactly. We inline the logic here rather
            // than calling the method because we don't yet have `self`.
            let mut counts: std::collections::HashMap<usize, usize> =
                std::collections::HashMap::new();
            for &cpu in &group.cpus {
                let node = cpu_to_node.get(&cpu).copied().unwrap_or(0);
                *counts.entry(node).or_insert(0) += 1;
            }
            let node = counts
                .into_iter()
                .max_by_key(|&(_, count)| count)
                .map(|(node, _)| node)
                .unwrap_or(0);
            node_llcs.entry(node).or_default().push(idx);
        }
        // Within-node LLC ordering: ascending llc_idx. Callers that
        // walk `host_node_llcs[node]` rely on this for deterministic
        // output — two ktstr invocations with identical topology see
        // the same walk order.
        for llcs in node_llcs.values_mut() {
            llcs.sort_unstable();
        }
        node_llcs
    }

    /// Maximum cores per LLC group on the host.
    pub fn max_cores_per_llc(&self) -> usize {
        self.llc_groups
            .iter()
            .map(|g| g.cpus.len())
            .max()
            .unwrap_or(0)
    }

    /// Total available host CPUs.
    pub fn total_cpus(&self) -> usize {
        self.online_cpus.len()
    }

    // ------------------------------------------------------------------
    // Shared NUMA-placement primitives
    // ------------------------------------------------------------------
    //
    // Used by the existing perf-mode pinning path
    // ([`numa_aware_llc_order`]) AND the `--cpu-cap` consolidation
    // PLAN phase. Both callers implement DIFFERENT selection algorithms
    // on top of these queries:
    //
    // - Perf-mode distributes virtual NUMA nodes across host NUMA
    //   nodes with modulo rotation; uses primitive 2
    //   (eligibility-by-capacity). No distance lookup.
    // - Consolidation seeds from a scored LLC list then greedily
    //   expands within the seed's node, spilling to nearest-by-distance
    //   when needed; uses primitive 3 (plus llc_numa_node).
    //
    // Kept as small orthogonal queries rather than a single mega-selector
    // — the two algorithms genuinely do different things, but they both
    // need the same three topology lookups.

    /// Memoized map of NUMA node → LLC indices on that node. Returned
    /// by reference so callers can iterate without cloning; `BTreeMap`
    /// gives deterministic iteration so two invocations on identical
    /// topologies produce identical walks.
    ///
    /// In-tree callers currently reach the same data via
    /// [`Self::numa_nodes_sorted_by_distance`] and [`Self::numa_nodes_with_capacity`]
    /// — both iterate `host_node_llcs` internally — so this accessor
    /// has no direct consumer today. Kept as a stable handle for
    /// future callers (e.g. a planned `ktstr topo --json` NUMA
    /// section) and downstream tooling that wants the raw map.
    #[allow(dead_code)]
    pub(crate) fn host_llcs_by_numa_node(&self) -> &std::collections::BTreeMap<usize, Vec<usize>> {
        &self.host_node_llcs
    }

    /// Return every NUMA node that has `>= min_llcs` LLCs, paired with
    /// that node's LLC-index slice. Callers filter through this when
    /// their algorithm requires per-node capacity guarantees (perf-mode
    /// passes `ceil(llcs/numa_nodes)` so any guest node can land on any
    /// host node; consolidation passes 1 so every node with at least
    /// one free LLC is a valid spill candidate). Iteration order
    /// follows the underlying `BTreeMap` — ascending by node id.
    pub(crate) fn numa_nodes_with_capacity(&self, min_llcs: usize) -> Vec<(usize, &Vec<usize>)> {
        self.host_node_llcs
            .iter()
            .filter(|(_, llcs)| llcs.len() >= min_llcs)
            .map(|(&node, llcs)| (node, llcs))
            .collect()
    }

    /// Return NUMA node ids sorted by distance from `anchor` ascending,
    /// with unreachable nodes (distance 255 per Linux convention)
    /// demoted to the end. Caller supplies the distance lookup via
    /// `distance_fn` so this primitive stays independent of any
    /// specific distance source — consolidation threads
    /// `TestTopology::numa_distance` through a closure, while callers
    /// without a distance matrix can pass
    /// `|from, to| if from == to { 10 } else { 20 }` for a trivial
    /// near/far split.
    ///
    /// `anchor` is included in the output (distance to self = 10 on
    /// the Linux convention, sorting first). Nodes without any LLCs
    /// on this host are skipped — spilling to an empty node has no
    /// value.
    pub(crate) fn numa_nodes_sorted_by_distance(
        &self,
        anchor: usize,
        distance_fn: impl Fn(usize, usize) -> u8,
    ) -> Vec<usize> {
        let mut nodes: Vec<(usize, u8)> = self
            .host_node_llcs
            .keys()
            .map(|&node| (node, distance_fn(anchor, node)))
            .collect();
        // Sort: unreachable (255) last; among reachable, ascending
        // distance; ties broken by ascending node id via the stable
        // sort applied over a pre-sorted (BTreeMap-ordered) input.
        nodes.sort_by(|a, b| {
            let a_unreachable = a.1 == 255;
            let b_unreachable = b.1 == 255;
            match (a_unreachable, b_unreachable) {
                (true, false) => std::cmp::Ordering::Greater,
                (false, true) => std::cmp::Ordering::Less,
                _ => a.1.cmp(&b.1),
            }
        });
        nodes.into_iter().map(|(node, _)| node).collect()
    }

    /// NUMA node for a host LLC group, determined by majority vote of
    /// its CPUs' NUMA assignments. Returns 0 when the map is empty
    /// (single-node systems).
    ///
    /// Production callers pre-compute the node-to-LLC mapping once at
    /// [`HostTopology::from_sysfs`] via
    /// [`compute_host_node_llcs`](Self::compute_host_node_llcs)
    /// (memoized in [`host_node_llcs`](Self::host_node_llcs)); use
    /// [`Self::host_llcs_by_numa_node`](Self::host_llcs_by_numa_node) to
    /// iterate the pre-built map. This method stays exposed for
    /// external callers (future `ktstr locks` NUMA column + any
    /// downstream tooling that needs a single-LLC lookup) and
    /// synthetic-topology tests that assert per-LLC node assignment.
    pub fn llc_numa_node(&self, llc_idx: usize) -> usize {
        let group = &self.llc_groups[llc_idx];
        let mut counts: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        for &cpu in &group.cpus {
            let node = self.cpu_to_node.get(&cpu).copied().unwrap_or(0);
            *counts.entry(node).or_insert(0) += 1;
        }
        counts
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(node, _)| node)
            .unwrap_or(0)
    }

    /// Compute a pinning plan that maps virtual LLCs to physical LLC groups.
    ///
    /// Each virtual LLC's vCPUs are assigned to cores within a single physical LLC.
    /// `llc_offset` rotates the starting LLC group so concurrent VMs pin to
    /// different physical cores. When `reserve_service_cpu` is true, one
    /// additional host CPU is reserved for service threads (monitor, watchdog).
    ///
    /// When `topo.numa_nodes > 1`, virtual LLCs are grouped by guest NUMA
    /// node and each group is placed on host LLCs within the same physical
    /// NUMA node. Falls back to sequential placement when the host lacks
    /// enough NUMA-aligned LLCs.
    ///
    /// Returns an error if the host cannot satisfy the topology.
    pub fn compute_pinning(
        &self,
        topo: &super::topology::Topology,
        reserve_service_cpu: bool,
        llc_offset: usize,
    ) -> Result<PinningPlan> {
        self.compute_pinning_at(topo, reserve_service_cpu, llc_offset, 0)
    }

    /// [`Self::compute_pinning`] with an additional INTRA-LLC slot:
    /// `intra_offset` selects which disjoint `vcpus_per_llc`-sized CPU
    /// window inside each mapped LLC the plan assigns. `intra_offset
    /// == 0` is exactly `compute_pinning`. This is the default run
    /// path's candidate-DIVERSITY lever: without it, every 1-vCPU
    /// cell's candidate for a given LLC offset is the same CPU prefix,
    /// so a 64-CPU / 4-LLC host serializes 1-vCPU cells FOUR wide (one
    /// per LLC) while 60 CPUs idle — a placement concentration the old
    /// skip-on-contention regime silently masked. With intra slots, an
    /// LLC with 16 CPUs offers 16 disjoint 1-vCPU candidates, and the
    /// suite's effective default-cell parallelism scales with CPUs,
    /// not LLC count. Returns `TopologyInsufficient` when the window
    /// does not fit (the caller enumerating candidates treats that as
    /// end-of-slots).
    pub fn compute_pinning_at(
        &self,
        topo: &super::topology::Topology,
        reserve_service_cpu: bool,
        llc_offset: usize,
        intra_offset: usize,
    ) -> Result<PinningPlan> {
        let cores = topo.cores_per_llc;
        let threads = topo.threads_per_core;
        let llcs = topo.llcs;
        let vcpus_per_llc = cores * threads;
        let total_vcpus = llcs * vcpus_per_llc;
        let total_needed = total_vcpus as usize + if reserve_service_cpu { 1 } else { 0 };

        if total_needed > self.total_cpus() {
            return Err(anyhow::Error::new(TopologyInsufficient {
                reason: format!(
                    "performance_mode: need {} CPUs ({} vCPUs + {} service) \
                     but only {} host CPUs available",
                    total_needed,
                    total_vcpus,
                    if reserve_service_cpu { 1 } else { 0 },
                    self.total_cpus(),
                ),
            }));
        }

        let num_llcs = self.llc_groups.len();
        if llcs as usize > num_llcs {
            return Err(anyhow::Error::new(TopologyInsufficient {
                reason: format!(
                    "performance_mode: need {} LLCs for {} virtual LLCs, \
                     but host has {} LLC groups",
                    llcs, llcs, num_llcs,
                ),
            }));
        }

        // Build the virtual-to-host LLC index mapping. When numa_nodes > 1,
        // try to place each guest NUMA node's LLCs on host LLCs within
        // the same physical NUMA node.
        let llc_order = self.numa_aware_llc_order(topo.numa_nodes, llcs, llc_offset);

        let mut assignments = Vec::with_capacity(total_vcpus as usize);
        let mut used_cpus = std::collections::HashSet::new();

        for llc in 0..llcs {
            let llc_idx = llc_order[llc as usize];
            let group = &self.llc_groups[llc_idx];
            let available: Vec<usize> = group
                .cpus
                .iter()
                .copied()
                .filter(|c| !used_cpus.contains(c))
                .collect();

            // The intra window shifts the assigned slice by whole
            // vcpus_per_llc-sized steps so distinct intra slots never
            // overlap (disjoint per-CPU lock sets by construction).
            let window_start = intra_offset * vcpus_per_llc as usize;
            if available.len() < window_start + vcpus_per_llc as usize {
                return Err(anyhow::Error::new(TopologyInsufficient {
                    reason: format!(
                        "LLC group {} has {} available CPUs, need {} at \
                         intra slot {} for virtual LLC {}",
                        llc_idx,
                        available.len(),
                        vcpus_per_llc,
                        intra_offset,
                        llc,
                    ),
                }));
            }

            for vcpu_in_llc in 0..vcpus_per_llc {
                let vcpu_id = llc * vcpus_per_llc + vcpu_in_llc;
                let host_cpu = available[window_start + vcpu_in_llc as usize];
                used_cpus.insert(host_cpu);
                assignments.push((vcpu_id, host_cpu));
            }
        }

        let service_cpu = if reserve_service_cpu {
            let cpu = self
                .online_cpus
                .iter()
                .copied()
                .find(|c| !used_cpus.contains(c));
            // Defensive: the total-CPU check above already folds the +1
            // service CPU into `total_needed`, so a passing host always
            // has at least one online CPU beyond the assigned vCPUs and
            // this never fires today. Typed as TopologyInsufficient (not
            // plain anyhow) so that if a future refactor of that check ever
            // lets it through, it is handled identically to its three
            // sibling shortfall checks: the perf-mode caller
            // (acquire_slot_with_locks) re-maps every compute_pinning
            // TopologyInsufficient to PerfModeUnavailable (a host-insufficiency
            // skip, fail under KTSTR_NO_SKIP_MODE), and
            // the non-perf caller passes reserve_service_cpu=false so this
            // site is unreachable there.
            if cpu.is_none() {
                return Err(anyhow::Error::new(TopologyInsufficient {
                    reason: format!(
                        "performance_mode: no free host CPU for service threads \
                         after assigning {total_vcpus} vCPUs"
                    ),
                }));
            }
            cpu
        } else {
            None
        };

        // Deduplicate LLC indices (multiple virtual LLCs may map to the
        // same host LLC at different offsets, but that's prevented by the
        // used_cpus check above — each virtual LLC consumes distinct CPUs).
        let mut llc_indices = llc_order;
        llc_indices.sort_unstable();
        llc_indices.dedup();

        Ok(PinningPlan {
            assignments,
            service_cpu,
            llc_indices,
            locks: Vec::new(),
        })
    }

    /// Per-CPU-GRAIN perf placement: tile each occupied host LLC into
    /// disjoint `(vcpus_per_llc + 1)`-CPU BLOCKS and map the guest onto
    /// block index `block`.
    ///
    /// Block `b` in a host LLC owns CPUs
    /// `[b*(V+1) .. b*(V+1)+V)` (V = `vcpus_per_llc`) as a contiguous,
    /// cache-coherent vCPU window — the SAME topology-mirroring
    /// [`Self::compute_pinning_at`] gives (each guest LLC's vCPUs stay a
    /// coherent group inside one real host LLC; guest LLCs never
    /// interleave or straddle a cache boundary). The guest's single
    /// service CPU is the block's SPARE `+V` slot in the first occupied
    /// host LLC, so it too lies INSIDE block `b`. Because every CPU a
    /// block consumes — vCPUs AND service — is contained in that block,
    /// DISTINCT block indices produce fully DISJOINT host-CPU sets: two
    /// perf cells at different blocks take non-overlapping per-CPU
    /// `LOCK_EX` sets and COEXIST under a shared (`LOCK_SH`) LLC lock.
    ///
    /// This is the placement half of the per-CPU-grain reservation (see
    /// [`perf_llc_lock_mode`]) that unblocks perf-cell parallelism on a
    /// host whose LLC DWARFS the cell — the AWS Graviton's single
    /// 96-CPU L3 being the motivating case, where whole-LLC `LOCK_EX`
    /// (the default [`Self::compute_pinning`] path) serializes EVERY
    /// perf cell onto the whole machine. It exists as a SEPARATE method
    /// rather than a `compute_pinning_at` flag precisely so the
    /// whole-LLC-Exclusive placement the validated dilation campaign
    /// measured stays byte-for-byte unchanged.
    ///
    /// Contrast [`Self::compute_pinning_at`], whose service CPU is the
    /// global-first-free CPU (fine when the whole LLC is locked `LOCK_EX`,
    /// but it would COLLIDE across grain blocks and break disjointness).
    ///
    /// Returns `TopologyInsufficient` when block `block` does not fit in
    /// every occupied host LLC — the caller enumerating blocks treats
    /// that as end-of-blocks (mirroring the `intra_offset` overflow in
    /// [`Self::compute_pinning_at`]).
    pub(crate) fn compute_pinning_grain(
        &self,
        topo: &super::topology::Topology,
        block: usize,
    ) -> Result<PinningPlan> {
        let cores = topo.cores_per_llc;
        let threads = topo.threads_per_core;
        let llcs = topo.llcs;
        let vcpus_per_llc = cores * threads;
        // A block reserves V vCPUs + 1 service slot so the WHOLE cell
        // footprint (including service) tiles at this stride and blocks
        // never overlap — the property that lets disjoint-block cells
        // coexist.
        let block_size = vcpus_per_llc as usize + 1;

        let num_llcs = self.llc_groups.len();
        if llcs as usize > num_llcs {
            return Err(anyhow::Error::new(TopologyInsufficient {
                reason: format!(
                    "performance_mode (per-CPU grain): need {llcs} LLCs for \
                     {llcs} virtual LLCs, but host has {num_llcs} LLC groups"
                ),
            }));
        }

        // Distinct host LLCs per guest LLC (offset 0 — grain diversity
        // comes from the BLOCK index within each LLC, not the LLC
        // offset), so no two guest LLCs share a host LLC and the block
        // windows never alias between guest LLCs.
        let llc_order = self.numa_aware_llc_order(topo.numa_nodes, llcs, 0);
        let block_start = block * block_size;

        let mut assignments = Vec::with_capacity((llcs * vcpus_per_llc) as usize);
        for llc in 0..llcs {
            let llc_idx = llc_order[llc as usize];
            let group = &self.llc_groups[llc_idx];
            // The entire block (V vCPUs + the service slot) must fit in
            // this host LLC's CPU list.
            if group.cpus.len() < block_start + block_size {
                return Err(anyhow::Error::new(TopologyInsufficient {
                    reason: format!(
                        "LLC group {} has {} CPUs, need block {} of size {} \
                         (end {}) for virtual LLC {}",
                        llc_idx,
                        group.cpus.len(),
                        block,
                        block_size,
                        block_start + block_size,
                        llc,
                    ),
                }));
            }
            for vcpu_in_llc in 0..vcpus_per_llc {
                let vcpu_id = llc * vcpus_per_llc + vcpu_in_llc;
                let host_cpu = group.cpus[block_start + vcpu_in_llc as usize];
                assignments.push((vcpu_id, host_cpu));
            }
        }

        // Service CPU: the block's spare `+V` slot in the FIRST occupied
        // host LLC — inside block `b`, so disjoint from every other
        // block's footprint. The fit check above already reserved it.
        let first_group = &self.llc_groups[llc_order[0]];
        let service_cpu = Some(first_group.cpus[block_start + vcpus_per_llc as usize]);

        let mut llc_indices = llc_order;
        llc_indices.sort_unstable();
        llc_indices.dedup();

        Ok(PinningPlan {
            assignments,
            service_cpu,
            llc_indices,
            locks: Vec::new(),
        })
    }

    /// Build the virtual LLC to host LLC index mapping.
    ///
    /// Falls back to sequential offset mapping when any of these hold:
    /// `numa_nodes == 0` (avoids divide-by-zero), `numa_nodes == 1`
    /// (no NUMA-awareness needed), `cpu_to_node` is empty (no NUMA
    /// map available), `llcs < numa_nodes` (base-per-node would be 0
    /// and leave guest nodes empty), or the host lacks enough
    /// NUMA-aligned LLCs.
    ///
    /// Otherwise, distributes `llcs` across `numa_nodes` guest nodes:
    /// the first `llcs % numa_nodes` guest nodes receive
    /// `base + 1 = ceil(llcs / numa_nodes)` LLCs each; the rest
    /// receive `base = floor(llcs / numa_nodes)` LLCs. This preserves
    /// the remainder that floor-only division would silently drop
    /// (e.g. `llcs=5, numa_nodes=2` yields counts 3+2 = 5).
    /// Eligibility requires each host NUMA node to supply at least
    /// `ceil(llcs / numa_nodes)` (the max any single guest node will
    /// claim) — stricter than the prior floor-based check, so the
    /// "+1" guest nodes always land on a node with capacity.
    ///
    /// Implementation composes [`Self::numa_nodes_with_capacity`],
    /// which iterates the memoized `host_node_llcs` map. The
    /// `--cpu-cap` consolidation PLAN phase instead composes
    /// [`Self::numa_nodes_sorted_by_distance`] plus
    /// [`Self::llc_numa_node`], so the two callers share the memoized
    /// `host_node_llcs` map rather than the same accessor calls. The
    /// two callers' SELECTION algorithms also differ: perf-mode does
    /// modulo rotation of guest onto host nodes; consolidation does
    /// score-driven greedy expansion.
    pub(crate) fn numa_aware_llc_order(
        &self,
        numa_nodes: u32,
        llcs: u32,
        llc_offset: usize,
    ) -> Vec<usize> {
        let num_host_llcs = self.llc_groups.len();

        // Sequential fallback used by the degenerate cases below.
        let sequential_fallback = || -> Vec<usize> {
            (0..llcs as usize)
                .map(|i| (i + llc_offset) % num_host_llcs)
                .collect()
        };

        // Defensive: zero NUMA nodes would divide-by-zero below. Also
        // handles the single-node case (no NUMA-awareness needed) and
        // the "cpu_to_node map unavailable" case.
        if numa_nodes == 0 || numa_nodes == 1 || self.cpu_to_node.is_empty() {
            return sequential_fallback();
        }

        // If the guest has fewer LLCs than NUMA nodes, a per-node base
        // of 0 would leave some guest nodes empty. Fall back rather
        // than silently dropping those nodes' LLCs.
        if llcs < numa_nodes {
            return sequential_fallback();
        }

        // Distribute LLCs across guest NUMA nodes. Integer division
        // alone drops the remainder (e.g. llcs=5, numa_nodes=2 gave
        // 2 per node = 4 LLCs assigned, 5th dropped). Fix: the first
        // `remainder` nodes get `base + 1`, the rest get `base`.
        let base_per_node = (llcs / numa_nodes) as usize;
        let remainder = (llcs % numa_nodes) as usize;
        // Ceiling-per-node — the largest count any single guest node
        // will claim. Host NUMA nodes must supply at least this many
        // to remain eligible.
        let max_per_node = base_per_node + if remainder > 0 { 1 } else { 0 };

        // Collect host NUMA nodes that can supply the ceiling (max)
        // per-node count — so any guest node can land there regardless
        // of whether it's one of the `remainder` "+1" nodes. Shared
        // primitive: `numa_nodes_with_capacity` filters the memoized
        // group-by-node map.
        let eligible_nodes = self.numa_nodes_with_capacity(max_per_node);

        // Need at least numa_nodes distinct host NUMA nodes with enough
        // LLCs each.
        if eligible_nodes.len() < numa_nodes as usize {
            return sequential_fallback();
        }

        // Assign guest NUMA nodes to host NUMA nodes, rotating by
        // llc_offset to spread concurrent VMs.
        let mut order = Vec::with_capacity(llcs as usize);
        let node_offset = llc_offset / max_per_node.max(1);
        for guest_node in 0..numa_nodes as usize {
            let host_idx = (guest_node + node_offset) % eligible_nodes.len();
            let (_, host_llcs) = &eligible_nodes[host_idx];
            let within_offset = llc_offset % host_llcs.len();
            // First `remainder` guest nodes get `base + 1` LLCs; rest
            // get `base`. Total assigned == llcs (remainder preserved).
            let count = if guest_node < remainder {
                base_per_node + 1
            } else {
                base_per_node
            };
            for i in 0..count {
                let llc_idx = host_llcs[(i + within_offset) % host_llcs.len()];
                order.push(llc_idx);
            }
        }

        order
    }
}

/// Lock mode for LLC reservation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlcLockMode {
    /// Exclusive access to the entire LLC (performance_mode tests).
    /// Returns unavailable when any shared or exclusive holder exists.
    Exclusive,
    /// Shared access to the LLC (non-perf pinned tests).
    /// Multiple shared holders coexist; returns unavailable when
    /// exclusive holder exists.
    #[allow(dead_code)]
    Shared,
}

/// Absolute lower bound, in host CPUs, on an LLC before perf-mode
/// reservation may drop from whole-LLC exclusion to per-CPU grain.
///
/// This is the GUARD on the occupancy ratio below, not the decision
/// axis itself. It is anchored ABOVE every validated host LLC — the
/// dev box's 16-CPU L3s, x86 CI's (`scx-x64`) ~8-CPU L3s, and native
/// arm's (`armly`) 4-CPU L2s — with margin, and far BELOW the
/// pathological monolithic L3 the grain switch exists for (the AWS
/// Graviton's single 96-CPU L3). Below this bar the reservation is
/// ALWAYS whole-LLC [`LlcLockMode::Exclusive`], so the entire measured
/// perf campaign (dilation_validation.md, the perf-isolation gates) is
/// byte-for-byte unchanged on every host it was validated on.
///
/// Why a floor at all, when the author's model is a pure occupancy
/// ratio: the real perf cells are SMALL (`cores = 2..4`, `threads = 1`
/// → 2-4 vCPUs per LLC — see the `performance_mode` e2e fixtures), so on
/// the dev box's 16-CPU L3 a cell occupies as little as `2/16 = 0.125`.
/// A pure `>= 0.5`-occupancy gate (which assumes ~LLC-filling cells)
/// would therefore reclassify the dev box AND x86 CI (`2/8 = 0.25`) to
/// per-CPU grain and invalidate their measurements. The floor makes the
/// ratio safe precisely because no validated host has an LLC anywhere
/// near this size — 32 sits at 2× the largest validated LLC (16) and
/// 1/3 of the Graviton's 96 — so the ratio only ever DECIDES on a
/// genuinely monolithic cache domain, and the validated hosts are
/// off-limits regardless of how small a cell they run.
pub(crate) const PERF_GRAIN_LLC_MIN_CPUS: usize = 32;

/// Maximum LLC OCCUPANCY (cell's per-LLC pinned CPUs ÷ host LLC CPUs) at
/// which perf-mode still drops to per-CPU grain, expressed as an exact
/// integer ratio to avoid float. A cell occupying `< 1/2` of the LLC is
/// a minority slice: on a huge L3 its per-cell cache share is ample, and
/// a per-CPU-grain neighbour is at most a comparable slice, so the
/// perf-mode `D≈1` isolation contract still effectively holds (the
/// dilation witness + the `D>1.5` perf-isolation gate catch any residual
/// perturbation). A cell occupying `>= 1/2` of even a huge LLC wants the
/// cache to itself — a neighbour would be a large fraction of its
/// working set — so it keeps whole-LLC [`LlcLockMode::Exclusive`]
/// regardless of the LLC's absolute size. This is the author's ratio
/// gate; on our small cells it is trivially satisfied on the Graviton
/// (`2/96 ≈ 0.02`) and only bites a hypothetical LLC-filling perf cell.
const PERF_GRAIN_MAX_OCCUPANCY_NUM: usize = 1;
const PERF_GRAIN_MAX_OCCUPANCY_DEN: usize = 2;

/// Choose the LLC lock mode for a perf-mode reservation of `plan`.
///
/// The default perf-mode reservation is whole-LLC
/// [`LlcLockMode::Exclusive`]: the cell owns its entire cache domain —
/// the strongest isolation, and what the dilation campaign measured. On
/// a host whose LLC DWARFS the cell (a single monolithic L3 spanning
/// scores of CPUs — the AWS Graviton's 96-CPU L3), whole-LLC `LOCK_EX`
/// means one small cell exclusively locks the WHOLE machine, so perf
/// cells serialize globally (the 20-60× arm-CI makespan blowup). There
/// we drop to per-CPU GRAIN: a SHARED (`LOCK_SH`) LLC lock so cells
/// coexist on the giant L3, plus EXCLUSIVE per-CPU locks over exactly
/// the pinned cores + service CPU — the SAME composition
/// [`LlcLockMode::Shared`] already produces (`fixed_set_cpus` +
/// `claim_for`), so the returned claim names the ACTUAL CPUs held, not
/// the whole LLC, and peers see the freed capacity.
///
/// The switch requires BOTH gates, evaluated PER occupied host LLC and
/// taken conservatively (any occupied LLC that is modest, or that the
/// cell fills, keeps the WHOLE plan Exclusive):
///
/// 1. [`PERF_GRAIN_LLC_MIN_CPUS`] — the LLC is far larger than any
///    validated CI/dev cache domain (the absolute guard that keeps
///    every validated host on the unchanged whole-LLC path).
/// 2. [`PERF_GRAIN_MAX_OCCUPANCY_NUM`]/[`PERF_GRAIN_MAX_OCCUPANCY_DEN`]
///    — the cell occupies a minority (`< 1/2`) of that LLC (a cell that
///    wants most of even a huge LLC keeps the whole L3).
///
/// On every validated host gate 1 fails (LLC ≤ 16 < 32), so this
/// returns `Exclusive` there no matter how small the cell — behaviour
/// is identical to the pre-change whole-LLC campaign.
pub(crate) fn perf_llc_lock_mode(host_topo: &HostTopology, plan: &PinningPlan) -> LlcLockMode {
    if plan.llc_indices.is_empty() {
        return LlcLockMode::Exclusive;
    }
    // Per-host-LLC pinned footprint: vCPU assignments landing in the
    // LLC, plus the service CPU if it lives there.
    let footprint_in = |llc_idx: usize| -> usize {
        let cpus = &host_topo.llc_groups[llc_idx].cpus;
        let in_llc = |c: usize| cpus.contains(&c);
        let vcpus = plan.assignments.iter().filter(|&&(_, c)| in_llc(c)).count();
        vcpus + usize::from(plan.service_cpu.is_some_and(in_llc))
    };
    // Grain only when EVERY occupied host LLC is both huge and a
    // minority-occupied slice — the fullest / smallest occupied LLC
    // dominates the decision, so a plan touching one modest LLC stays
    // Exclusive.
    let all_grain = plan.llc_indices.iter().all(|&idx| {
        let llc_cpus = host_topo.llc_groups[idx].cpus.len();
        let footprint = footprint_in(idx);
        let huge = llc_cpus >= PERF_GRAIN_LLC_MIN_CPUS;
        // footprint / llc_cpus < NUM / DEN  ⇔  footprint*DEN < llc_cpus*NUM.
        let minority = footprint.saturating_mul(PERF_GRAIN_MAX_OCCUPANCY_DEN)
            < llc_cpus.saturating_mul(PERF_GRAIN_MAX_OCCUPANCY_NUM);
        huge && minority
    });
    if all_grain {
        LlcLockMode::Shared
    } else {
        LlcLockMode::Exclusive
    }
}

/// Resource lock acquisition outcome.
#[derive(Debug)]
pub enum LockOutcome {
    /// All locks acquired successfully.
    Acquired {
        /// LLC offset consumed; read only by the locking test fixtures.
        #[allow(dead_code)]
        llc_offset: usize,
        locks: Vec<std::os::fd::OwnedFd>,
    },
    /// Resources busy. The inner string carries the diagnostic reason
    /// surfaced to test fixtures; production callers only match the
    /// variant tag.
    Unavailable(#[allow(dead_code)] String),
}

/// Acquire resource locks for a pinning plan (non-blocking).
///
/// **LLC locks** (`{lock_dir}/ktstr-llc-{N}.lock`):
/// - `Exclusive`: `flock(LOCK_EX | LOCK_NB)` — sole access to the LLC.
/// - `Shared`: `flock(LOCK_SH | LOCK_NB)` — multiple holders coexist.
///
/// **CPU locks** (`{lock_dir}/ktstr-cpu-{C}.lock`):
/// - Always `flock(LOCK_EX | LOCK_NB)` — exclusive per CPU.
/// - Skipped for `Exclusive` LLC mode (the LLC lock already provides
///   exclusivity over all CPUs in the group).
///
/// Single non-blocking, all-or-nothing attempt (the fast path of the
/// acquisition protocol — see [`protocol`]). Returns
/// `LockOutcome::Unavailable` immediately when any resource is busy,
/// having released every lock it took (protocol rule: no fast-path
/// partial ever persists; only the queue head may hold partials).
/// Locks are walked in the canonical global order (LLC index
/// ascending, then CPU index ascending).
///
/// Claim-aware: when a live queue head has published a claim
/// intersecting this request, the attempt reports `Unavailable`
/// WITHOUT touching the claimed locks — fast-path callers subtract
/// the head's target from their view of free capacity so disjoint
/// invocations don't snipe the slots the head is accumulating.
///
/// Used by the offset-scan probe in
/// [`crate::vmm::KtstrVm::acquire_default_run_locks`] (which needs a
/// fast "is this offset free?" answer per candidate), by perf mode's
/// fast path, and by the locking tests.
///
/// `KTSTR_CARGO_TEST_MODE` short-circuits the entire flock dance and
/// returns `Acquired` with an empty fd list — bare `cargo test`
/// invocations don't share the cross-process LLC reservation
/// contract that nextest / `cargo ktstr test` peers rely on. Tests
/// run on whatever CPUs the OS schedules them onto.
pub fn acquire_resource_locks(
    plan: &PinningPlan,
    llc_indices: &[usize],
    llc_mode: LlcLockMode,
) -> Result<LockOutcome> {
    acquire_resource_locks_waiting(plan, llc_indices, llc_mode, false)
}

/// Fixed-set reservation acquire with optional queue-and-wait.
///
/// `wait == false` is the single non-blocking fast-path attempt — the
/// interactive / one-shot `ktstr shell` path (which degrades a busy
/// host to a lock-free overcommit rather than parking a human), the
/// per-offset probe in the default run path, and build-time
/// pre-checks.
///
/// `wait == true` is the TEST run path: on a busy reservation the
/// caller joins the cross-invocation acquisition queue
/// ([`protocol::wait_for_queue_turn`]) and, once head, accumulates
/// the fixed target set incrementally under the head license —
/// holding partials, waking on lock-dir inotify events as holders
/// release, re-sweeping on every wake. This is the resource-budget
/// model's contract — a budgeted / shared acquirer WAITS for an
/// exclusive holder. It waits for the authoritative flock release;
/// holder crashes are cleaned up by the kernel, while the holder's VM
/// watchdog and nextest process rail own lifecycle bounds. The lock
/// scheduler never guesses that a live holder is wedged from elapsed
/// wall time.
///
/// The fixed set is the degenerate re-plan case: the target cannot
/// change, so "re-plan on every wake" reduces to re-sweeping the same
/// canonical-order lock list against live state. Queued, this caller
/// holds NO resource locks; as head, its partials are fenced by the
/// published claim.
///
/// `KTSTR_CARGO_TEST_MODE` short-circuits to `Acquired` with an empty
/// fd list, same as [`acquire_resource_locks`].
pub fn acquire_resource_locks_waiting(
    plan: &PinningPlan,
    llc_indices: &[usize],
    llc_mode: LlcLockMode,
    wait: bool,
) -> Result<LockOutcome> {
    if crate::cargo_test_mode::cargo_test_mode_active() {
        return Ok(LockOutcome::Acquired {
            llc_offset: llc_indices.first().copied().unwrap_or(0),
            locks: Vec::new(),
        });
    }
    let llc_offset = llc_indices.first().copied().unwrap_or(0);
    // Fast path: claim-subtracted, non-blocking, all-or-nothing.
    let first_reason = match try_acquire_all(plan, llc_indices, llc_mode) {
        Ok(locks) => return Ok(LockOutcome::Acquired { llc_offset, locks }),
        Err(reason) => reason,
    };
    if !wait {
        return Ok(LockOutcome::Unavailable(first_reason));
    }
    // Contended: queue up (arrival order, crash-safe ticket), then
    // accumulate as head.
    let _queue = protocol::wait_for_queue_turn()?;
    let target = protocol::canonical_lock_order(
        llc_indices,
        match llc_mode {
            LlcLockMode::Exclusive => FlockMode::Exclusive,
            LlcLockMode::Shared => FlockMode::Shared,
        },
        &fixed_set_cpus(plan, llc_mode),
    );
    let claim = claim_for(llc_indices, plan, llc_mode);
    let outcome = protocol::acquire_as_head(|held| {
        held.sweep(&target)?;
        if held.covers(&target) {
            Ok(protocol::HeadStep::Complete(held.take(&target)))
        } else {
            Ok(protocol::HeadStep::Waiting {
                claim: claim.clone(),
            })
        }
    })?;
    Ok(match outcome {
        protocol::HeadOutcome::Acquired(locks) => LockOutcome::Acquired { llc_offset, locks },
        protocol::HeadOutcome::Aborted { reason } => LockOutcome::Unavailable(reason),
    })
}

/// The CPU-lock set for a fixed reservation: assignment CPUs plus the
/// service CPU, or empty for `Exclusive` LLC mode (the LLC lock
/// already covers its CPUs).
fn fixed_set_cpus(plan: &PinningPlan, llc_mode: LlcLockMode) -> Vec<usize> {
    if llc_mode == LlcLockMode::Exclusive {
        return Vec::new();
    }
    let mut cpus: Vec<usize> = plan.assignments.iter().map(|&(_, c)| c).collect();
    cpus.extend(plan.service_cpu);
    cpus
}

/// The published claim for a fixed reservation.
fn claim_for(
    llc_indices: &[usize],
    plan: &PinningPlan,
    llc_mode: LlcLockMode,
) -> protocol::ClaimSet {
    let flock_mode = match llc_mode {
        LlcLockMode::Exclusive => FlockMode::Exclusive,
        LlcLockMode::Shared => FlockMode::Shared,
    };
    protocol::ClaimSet::new(
        llc_indices.iter().copied(),
        fixed_set_cpus(plan, llc_mode),
        flock_mode,
    )
}

/// Compose the LLC lockfile prefix from the resolved lock directory.
/// Returns `{lock_dir}/ktstr-llc-`.
fn llc_lock_prefix() -> String {
    format!("{}/ktstr-llc-", crate::cache::resolve_lock_dir().display())
}

/// Compose the per-CPU lockfile prefix from the resolved lock directory.
/// Returns `{lock_dir}/ktstr-cpu-`.
fn cpu_lock_prefix() -> String {
    format!("{}/ktstr-cpu-", crate::cache::resolve_lock_dir().display())
}

#[cfg(test)]
thread_local! {
    /// Thread-local override for the LLC lock prefix. Tests set this
    /// to a per-test tempdir so the acquire path operates on its
    /// own lockfile pool instead of padding the `LlcGroup` vector
    /// to 90,000+ entries just to avoid collision with production
    /// indices at 0..<host-llcs>. See tests `acquire_llc_plan_*`
    /// that build a small synth topo and point the prefix at a
    /// `TempDir`.
    pub(crate) static LLC_LOCK_PREFIX_OVERRIDE: std::cell::RefCell<Option<String>> =
        const { std::cell::RefCell::new(None) };

    /// Thread-local override for the per-CPU lock prefix. Symmetric
    /// with `LLC_LOCK_PREFIX_OVERRIDE`.
    pub(crate) static CPU_LOCK_PREFIX_OVERRIDE: std::cell::RefCell<Option<String>> =
        const { std::cell::RefCell::new(None) };
}

/// Compose the LLC lockfile path for `llc_idx`. Production resolves
/// via `KTSTR_LOCK_DIR` (fallback `/tmp`); tests can override the
/// prefix via `LLC_LOCK_PREFIX_OVERRIDE` to keep their lockfile
/// pool isolated.
pub(crate) fn llc_lock_path(llc_idx: usize) -> String {
    #[cfg(test)]
    {
        if let Some(p) = LLC_LOCK_PREFIX_OVERRIDE.with(|p| p.borrow().clone()) {
            return format!("{p}{llc_idx}.lock");
        }
    }
    format!("{}{llc_idx}.lock", llc_lock_prefix())
}

/// Compose the per-CPU lockfile path for `cpu`. Symmetric with
/// [`llc_lock_path`] — production resolves via `KTSTR_LOCK_DIR`;
/// tests can override via `CPU_LOCK_PREFIX_OVERRIDE`.
pub(crate) fn cpu_lock_path(cpu: usize) -> String {
    #[cfg(test)]
    {
        if let Some(p) = CPU_LOCK_PREFIX_OVERRIDE.with(|p| p.borrow().clone()) {
            return format!("{p}{cpu}.lock");
        }
    }
    format!("{}{cpu}.lock", cpu_lock_prefix())
}

/// Try to acquire all resource locks (all-or-nothing, non-blocking,
/// canonical order — see [`protocol`] rule 3). Returns the held fds
/// on success, or an error string describing which resource was
/// busy; on ANY failure every lock taken so far is released before
/// returning, so no fast-path partial ever persists (only the queue
/// head may hold partials).
///
/// Claim-aware: a LIVE head claim ([`protocol::read_live_claim`])
/// intersecting this request fails the attempt up front WITHOUT
/// touching the claimed lockfiles — fast-path callers must not snipe
/// the slots the head is accumulating. Claim staleness is tolerated:
/// a caller acting on an outdated read at worst bounces once against
/// the real flocks.
fn try_acquire_all(
    plan: &PinningPlan,
    llc_indices: &[usize],
    llc_mode: LlcLockMode,
) -> std::result::Result<Vec<std::os::fd::OwnedFd>, String> {
    let flock_mode = match llc_mode {
        LlcLockMode::Exclusive => FlockMode::Exclusive,
        LlcLockMode::Shared => FlockMode::Shared,
    };
    let claim = protocol::read_live_claim();
    if !claim.is_empty() {
        if let Some(&idx) = llc_indices
            .iter()
            .find(|&&i| claim.conflicts_with_llc(i, flock_mode))
        {
            return Err(format!("LLC {idx} claimed by the queue head"));
        }
        if let Some(&cpu) = fixed_set_cpus(plan, llc_mode)
            .iter()
            .find(|c| claim.cpus.contains(c))
        {
            return Err(format!("CPU {cpu} claimed by the queue head"));
        }
    }
    let target =
        protocol::canonical_lock_order(llc_indices, flock_mode, &fixed_set_cpus(plan, llc_mode));
    let mut locks = Vec::with_capacity(target.len());
    for (path, mode) in &target {
        match try_flock(path, *mode) {
            Ok(Some(fd)) => locks.push(fd),
            // Dropping `locks` on return releases everything taken
            // so far — the all-or-nothing contract.
            Ok(None) => return Err(format!("{path} busy")),
            Err(e) => return Err(format!("{path}: {e}")),
        }
    }
    Ok(locks)
}

/// Diffuse a pid across `[0, max_start)` so adjacent pids do not
/// land on adjacent offsets. Used by the default-else run-lock path
/// (`KtstrVm::acquire_default_run_locks`) to pick a starting LLC slot so
/// two ktstr invocations launching simultaneously don't both probe slot 0
/// first.
///
/// Bare `pid % max_start` collapses adjacent pids onto adjacent
/// offsets (Linux's pid allocator walks `pid_max` sequentially),
/// which is the worst spread shape for the common batch-spawn
/// case: nextest forks N test processes back-to-back, every pid
/// lands within a small contiguous range, every `pid % max_start`
/// lands within an equally small contiguous slice of the offset
/// space, and they all probe overlapping slots on the first
/// pass. AHasher avalanche on the pid bytes diffuses adjacent
/// pids across the whole `[0, max_start)` range, so the
/// slot-rotation loop has a fair chance of finding a free slot
/// without burning the entire lockfile pool.
///
/// The hasher is `ahash::AHasher` keyed with fixed zero seeds
/// (`RandomState::with_seeds(0, 0, 0, 0)`); a per-run random
/// seed would defeat reproducibility for unit-test fixtures and
/// for any future debug logging that wants to confirm "pid X
/// picks offset Y for slot N".
///
/// Caller invariant: `max_start >= 1`. Panics on `max_start == 0`
/// (modulo-by-zero); callers must enforce this upstream (the
/// run-lock path floors `max_slots` at 1).
pub(crate) fn pid_window_offset(pid: u32, max_start: usize) -> usize {
    use std::hash::{BuildHasher, Hasher};
    let mut hasher = ahash::RandomState::with_seeds(0, 0, 0, 0).build_hasher();
    hasher.write(&pid.to_le_bytes());
    (hasher.finish() as usize) % max_start
}

// ===========================================================================
// --cpu-cap PLAN pipeline — CpuCap / LlcSnapshot / LlcPlan + discover/plan/acquire
// ===========================================================================
//
// Entry point [`acquire_llc_plan`] is the single non-perf-mode
// reservation path: kernel builds and no-perf-mode VMs both call it
// with or without `--cpu-cap N`. `--cpu-cap` is a CPU-count budget:
// the planner reserves exactly N host CPUs by walking whole LLCs in
// contention- / NUMA-aware order and partial-taking the last LLC
// so `plan.cpus.len() == N`. The flock is per-LLC even when the
// last LLC is only partially used — coordination with concurrent
// ktstr peers is unchanged at LLC granularity. When `--cpu-cap`
// is absent the planner defaults to 30% of the calling process's
// sched_getaffinity cpuset (see [`default_cpu_budget`] and
// [`host_allowed_cpus`]) — not 30% of the host's online CPU count,
// because a CI runner whose parent cgroup pins ktstr to a 4-CPU
// subset must plan within THAT subset or sched_setaffinity on the
// resulting mask produces an empty effective set.
// Perf-mode never reaches this path; it stays on
// [`acquire_resource_locks`] for its `LOCK_EX` reservation contract.
//
// The pipeline has three phases: discover (snapshot holders per
// LLC, filtered to the process's allowed cpuset), plan (NUMA-aware
// selection under the caller's [`PlacementPolicy`] — Consolidate
// for builds, Spread for no-perf VMs), acquire (non-blocking `LOCK_SH`
// on each selected LLC). Up to ACQUIRE_MAX_TOCTOU_RETRIES retries
// absorb the window between the discover snapshot and the
// non-blocking acquire; between retries the loop sleeps for an
// ascending micro-budget (TOCTOU_RETRY_DELAYS) so a peer that
// raced us has time to drop its fds before the next snapshot.
// If every retry fails, the contention is persistent and the
// caller falls back to nextest-retry / operator-wait.

/// Return the CPUs the calling process is allowed to run on, per
/// `sched_getaffinity(2)` with a `/proc/self/status` Cpus_allowed_list
/// fallback. Every consumer of the `--cpu-cap` pipeline plans against
/// this set instead of `HostTopology::online_cpus` so
/// `sched_setaffinity` on the plan's CPU list never produces an empty
/// effective mask under a cgroup-restricted runner (CI hosts, systemd
/// slices, sudo -u under a limited cpuset).
///
/// Returns an empty vec only when BOTH the syscall AND procfs fail —
/// a pathological host that can't enumerate its own affinity. Callers
/// treat that as a bail reason, not a fallback "every CPU" permission:
/// guessing on a misconfigured host is worse than failing visibly.
///
/// Tests override the return value via `ALLOWED_CPUS_OVERRIDE` so
/// the 30% default and allowed-cpu filtering are deterministic in
/// unit tests regardless of the CI runner's real cpuset.
pub(crate) fn host_allowed_cpus() -> Vec<usize> {
    #[cfg(test)]
    {
        if let Some(override_set) = ALLOWED_CPUS_OVERRIDE.with(|p| p.borrow().clone()) {
            return override_set;
        }
    }
    if let Some(cpus) = crate::cpu_util::read_affinity(0) {
        return cpus.into_iter().map(|c| c as usize).collect();
    }
    if let Ok(raw) = std::fs::read_to_string("/proc/self/status") {
        for line in raw.lines() {
            if let Some(v) = line.strip_prefix("Cpus_allowed_list:")
                && let Some(parsed) = crate::cpu_util::parse_cpu_list(v.trim())
            {
                return parsed.into_iter().map(|c| c as usize).collect();
            }
        }
    }
    Vec::new()
}

#[cfg(test)]
thread_local! {
    /// Test-only override for [`host_allowed_cpus`]. Set via
    /// [`AllowedCpusGuard`] to make 30%-of-allowed calculations and
    /// plan filtering deterministic in unit tests. Mirrors the
    /// `LLC_LOCK_PREFIX_OVERRIDE` pattern.
    pub(crate) static ALLOWED_CPUS_OVERRIDE: std::cell::RefCell<Option<Vec<usize>>> =
        const { std::cell::RefCell::new(None) };
}

/// Default CPU budget when `--cpu-cap` is not set: 30% of the
/// allowed-CPU count, rounded up, with a min-1 floor for small or
/// degenerate hosts. 30% leaves enough headroom for concurrent peers
/// (tests, builds) while still reserving a non-trivial slice; the
/// min-1 floor prevents returning 0 on a 1- or 2-CPU host, where
/// ceil(×0.30) ≥ 1 anyway — the `.max(1)` is defense in depth for
/// future ratio tweaks.
fn default_cpu_budget(allowed_cpus: usize) -> usize {
    allowed_cpus.saturating_mul(30).div_ceil(100).max(1)
}

/// No-perf CPU budget when no explicit `--cpu-cap` (or `cpu_budget` knob) is
/// set: the VM's own vCPU count PLUS ONE service CPU, clamped to the allowed
/// cpuset (min-1). That is the topology's actual need: one host CPU per
/// guest vCPU, plus headroom for the host-side service threads (monitor,
/// watchdog, virtio, workload coordination) that live INSIDE the cell's
/// cgroup and would otherwise CFS-contend with spinning vCPUs on an
/// exactly-vcpus cpuset — the budgeted-path analog of perf mode's
/// `reserve_service_cpu` +1 in `compute_pinning`. Sensing timeliness rides
/// on this: on hosts without CAP_SYS_NICE the sensing threads run
/// SCHED_OTHER, and a starved monitor thins the contention witness the
/// latency verdicts consume.
///
/// The rationale is TEST VALIDITY, not boot speed — do not "optimize" this
/// back to a flat 30%. A scheduler test measures how the GUEST scheduler
/// places tasks across the guest's CPUs. If the VM's vCPU threads are
/// oversubscribed on the host (256 vCPUs sharing the 30% default mask is
/// ~95 pCPUs = 2.7x), the HOST scheduler time-slices them, so guest vCPUs
/// stall for reasons unrelated to the workload — a host-contention confound
/// that invalidates the guest-scheduler measurement (the silent-wrong-answer
/// class the project guards against). Sizing the budget to `vcpus + 1` when
/// host capacity permits gives the guest's CPUs real host CPUs and leaves
/// service-thread headroom, so its scheduler view tracks real concurrency.
/// (A wide boot also drops ~0.7s as the kernel's parallel AP bring-up runs
/// unthrottled, but that is incidental.)
///
/// NOT floored at the 30% `default_cpu_budget`. That floor was a bug: it made
/// a SMALL VM over-reserve on a LARGE host — a 2-vCPU interactive shell on a
/// ~192-CPU box resolved to `max(58, 2) = 58` CPUs (~30%). The reservation
/// walks WHOLE LLCs to reach the budget, so a bigger budget needs more LLCs
/// that are free for `LOCK_SH`; when peer runners' perf-mode work holds LLCs
/// `LOCK_EX` the plan cannot find 58 CPUs' worth and bails
/// (`acquire_llc_plan: could not reserve 58 CPU(s) after 4 attempts`). A
/// SMALLER reservation frees LLCs for peers, not the reverse, so a small VM
/// asking ~30% is backwards. 30% is the right default for throughput-elastic
/// kernel builds (`default_cpu_budget`, `make -j`), never a floor for a VM
/// whose need is fixed by its vCPU count. `vcpus + 1 > allowed` clamps to
/// `allowed` (the process-cpuset-too-small case), which
/// [`overcommit_warning`] surfaces. An explicit cap LOWER than vcpus stays
/// the deliberate opt-in to oversubscribe for contention testing.
pub(crate) fn no_perf_cpu_budget(allowed_cpus: usize, vm_vcpus: usize) -> usize {
    vm_vcpus.saturating_add(1).min(allowed_cpus).max(1)
}

/// Parsed `--cpu-cap N` value. N is a CPU count: the planner reserves
/// exactly N host CPUs by walking whole LLCs in contention- /
/// NUMA-aware order (filtered to the calling process's allowed
/// cpuset) and partial-taking the last LLC so `plan.cpus.len() == N`.
/// The flock set is still per-LLC (the last LLC is flocked whole
/// even when only a prefix of its CPUs enters `plan.cpus`).
/// Bounded to `1..=usize::MAX` at the constructor — a cap of 0 is
/// nonsensical (reserving zero CPUs is just "don't run") and
/// rejected upstream by the CLI layer, but we enforce the bound in
/// the type system via `NonZeroUsize` so callers can
/// `CpuCap::new(...)?` without a follow-up bounds check.
///
/// The runtime upper bound — "don't exceed the process's allowed
/// CPU count" — is enforced at acquire time via
/// [`CpuCap::effective_count`] because the allowed set is not known
/// until `host_allowed_cpus` reads `sched_getaffinity`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuCap {
    n: std::num::NonZeroUsize,
}

impl CpuCap {
    /// Construct from a raw `usize` CPU count. Returns `Err` on `0`;
    /// `usize::MAX` is accepted here and clamped later by
    /// `effective_count`.
    pub fn new(n: usize) -> Result<Self> {
        std::num::NonZeroUsize::new(n)
            .map(|n| CpuCap { n })
            .ok_or_else(|| anyhow::anyhow!("--cpu-cap must be ≥ 1 CPU (got 0)"))
    }

    /// Three-tier resolution: explicit CLI flag wins over env var,
    /// which wins over "not set". Returns `None` when neither is present,
    /// meaning "use the caller's auto-sized default": the
    /// kernel-build/planner path expands `None` to `default_cpu_budget`
    /// (30% of the allowed set); the no-perf VM-builder path expands it to
    /// `no_perf_cpu_budget` (`vcpus + 1`, clamped to the allowed cpuset and
    /// at least 1). The extra CPU leaves room for the VMM/control threads;
    /// this default is NOT floored at 30%.
    ///
    /// Env var is `KTSTR_CPU_CAP` (integer ≥ 1, CPU count). An empty
    /// or unset env var is treated as absent; a non-numeric value
    /// OR the numeric value `0` is an error — `KTSTR_CPU_CAP=0`
    /// flows through `CpuCap::new(0)` which rejects with "--cpu-cap
    /// must be ≥ 1 CPU (got 0)". Zero is not a silent fallback to
    /// "no cap"; it surfaces as a parse-time error so typos and
    /// scripting mistakes don't accidentally disable the resource
    /// contract.
    pub fn resolve(cli_flag: Option<usize>) -> Result<Option<CpuCap>> {
        if let Some(n) = cli_flag {
            return Ok(Some(CpuCap::new(n)?));
        }
        match std::env::var(crate::KTSTR_CPU_CAP_ENV) {
            Ok(s) if s.is_empty() => Ok(None),
            Ok(s) => {
                let n: usize = s
                    .parse()
                    .with_context(|| format!("KTSTR_CPU_CAP is not a valid integer: {s:?}"))?;
                Ok(Some(CpuCap::new(n)?))
            }
            Err(std::env::VarError::NotPresent) => Ok(None),
            Err(std::env::VarError::NotUnicode(raw)) => {
                anyhow::bail!(
                    "KTSTR_CPU_CAP contains non-UTF-8 bytes ({} bytes): {raw:?}. \
                     Set an integer value or unset.",
                    raw.len(),
                )
            }
        }
    }

    /// Runtime-bounded cap: returns the inner count unless it exceeds
    /// `allowed_cpus` (the calling process's sched_getaffinity cpuset
    /// count), in which case a `CpuBudgetUnsatisfiable` hard error (an
    /// explicit cap the host cannot satisfy is a FAIL, not a transient
    /// skip) steers the caller toward an actionable message. This check
    /// lives at acquire time — not at construction — because the allowed
    /// set is not known until `host_allowed_cpus` reads the syscall.
    pub fn effective_count(&self, allowed_cpus: usize) -> Result<usize> {
        let n = self.n.get();
        if n > allowed_cpus {
            // An explicit --cpu-cap the host cannot satisfy is a hard ERROR
            // (the author typed a concrete number that does not exist here),
            // not transient contention: CpuBudgetUnsatisfiable, not
            // ResourceContention, so it fails rather than skips.
            return Err(anyhow::Error::new(CpuBudgetUnsatisfiable {
                reason: format!(
                    "--cpu-cap N = {n} exceeds the {allowed_cpus} CPUs this \
                     process is allowed on (from sched_getaffinity / \
                     Cpus_allowed_list). Pick a value ≤ {allowed_cpus}, \
                     release the cgroup/taskset constraint restricting this \
                     process, or omit --cpu-cap to use the auto-sized default \
                     (30% of the allowed set for kernel builds; \
                     min(vCPUs + 1, allowed CPUs) for no-perf VMs)."
                ),
            }));
        }
        Ok(n)
    }
}

/// Placement policy for the PLAN phase of [`acquire_llc_plan`]:
/// which LLCs a reservation prefers when more are eligible than the
/// budget needs.
///
/// `Consolidate` packs onto the LLCs already holding peers
/// (holder_count DESC, llc_idx ASC) so the rest of the host stays
/// whole for exclusive perf-mode reservations. Right for kernel-build
/// sandboxes: a build is throughput-elastic and indifferent to
/// sharing its cache domain with another build.
///
/// `Spread` picks the LEAST-held LLCs (holder_count ASC), breaking
/// ties by the eligible-list position rotated by `rotation`. Right
/// for no-perf VM vCPU placement, where Consolidate is
/// pathological: every concurrent VM cell would mask its vCPU
/// threads onto the same most-held LLCs — and since plans are
/// computed at `build()` while the `LOCK_SH` set is deferred to
/// `run()`, a fan-out of simultaneously-planning cells all snapshot
/// ZERO holders and the llc_idx-ASC tiebreak stacks every one of
/// them onto the identical LLC-0-upward prefix. Observed in the scx
/// verifier sweep: a 30-cell matrix on a 16-LLC runner piled every
/// small and mid-sized cell's mask onto the same low-LLC prefix
/// (the widest cells' whole-host masks overlapping it) while the
/// high LLCs ran near-idle — guests starved hard enough that
/// attached schedulers tripped the guest kernel's sched_ext stall
/// watchdog and wide-topology boots outran the VM deadline. The
/// per-process `rotation` (derive it via `pid_window_offset`) is
/// what breaks that zero-knowledge symmetry; holder_count ASC then
/// keeps later arrivals off whichever LLCs the earlier ones
/// actually locked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlacementPolicy {
    /// Pack onto already-held LLCs (kernel builds).
    Consolidate,
    /// Fan out across least-held LLCs, tie-broken by `rotation`
    /// (no-perf VM placement). `rotation` is reduced modulo the
    /// eligible-LLC count internally; any process-stable value works.
    Spread { rotation: usize },
}

impl PlacementPolicy {
    /// `Spread` seeded for this process: rotation from
    /// `pid_window_offset` over a wide domain so the later
    /// modulo-by-eligible-count stays uniform for any LLC count.
    pub fn spread_for_process() -> Self {
        PlacementPolicy::Spread {
            rotation: pid_window_offset(std::process::id(), 1 << 16),
        }
    }
}

/// Per-LLC discover snapshot: identity + current holder set.
/// Constructed by [`discover_llc_snapshots`] before the PLAN phase.
/// `pub(crate)` so the in-crate PLAN pipeline and this module's tests
/// can construct and inspect it; the `ktstr locks` observational
/// command shares only [`crate::flock::HolderInfo`], not this
/// structure. External callers have no reason to construct one.
#[derive(Debug, Clone)]
pub(crate) struct LlcSnapshot {
    /// Host LLC index — matches [`HostTopology::llc_groups`] ordering.
    pub(crate) llc_idx: usize,
    /// Canonical `{lock_dir}/ktstr-llc-{N}.lock` path. Stored so the
    /// ACQUIRE phase doesn't re-format the string per LLC.
    pub(crate) lockfile_path: std::path::PathBuf,
    /// Processes currently holding this LLC's flock (any mode). Empty
    /// when no peer holds the lock. Derived from a single `/proc/locks`
    /// read shared across every LLC in the discover phase.
    pub(crate) holders: Vec<crate::flock::HolderInfo>,
    /// `holders.len()`, cached so the PLAN sort can access it without
    /// re-traversing the holder list per candidate.
    pub(crate) holder_count: usize,
}

/// Output of [`acquire_llc_plan`]: the concrete LLC reservation plus
/// every piece of diagnostic context a downstream consumer could
/// want.
///
/// `mems` is the union of NUMA nodes containing the selected CPUs —
/// `BuildSandbox::try_create` writes this to the child cgroup's
/// `cpuset.mems` so memory allocations respect the same NUMA locality
/// the CPU reservation already implies.
///
/// `locks` holds the RAII file descriptors whose `OwnedFd::drop`
/// releases the kernel-side flock; the field is `pub(crate)` because
/// direct manipulation from outside the crate would defeat the drop
/// guarantee.
#[derive(Debug)]
pub struct LlcPlan {
    /// Selected host LLC indices, sorted ASCENDING. Acquire order
    /// matches this slice — two callers with the same target see the
    /// same ordering and converge on the same one-wins-the-others-retry
    /// livelock-proof sequence.
    pub locked_llcs: Vec<usize>,
    /// Flattened host CPU list, sized exactly `target_cpus`. The last
    /// locked LLC may contribute only a prefix of its allowed CPUs.
    /// Preserves LLC ordering: CPUs from `locked_llcs[0]` come
    /// before CPUs from `locked_llcs[1]`, etc.
    pub cpus: Vec<usize>,
    /// Union of NUMA nodes hosting the locked LLCs. When the plan
    /// spans > 1 node (cross-node spill — seed node exhausted, plan
    /// spilled to nearest-by-distance neighbors), `mems`
    /// contains every node — not just the seed node's.
    pub mems: std::collections::BTreeSet<usize>,
    /// Per-LLC discovery trail. Preserved through the lifetime of the
    /// plan so error-formatting (via `acquire_llc_plan`'s final
    /// fresh snapshot) and future `ktstr locks` rendering don't
    /// re-probe `/proc/locks`. In-tree consumers currently re-read
    /// the snapshot only on the TOCTOU failure path; the field is
    /// kept populated so downstream tooling can inspect the
    /// plan-at-acquire holder set without a second pass.
    #[allow(dead_code)]
    pub(crate) snapshot: Vec<LlcSnapshot>,
    /// RAII flock holders. Dropped when the plan goes out of scope,
    /// releasing each LLC's `LOCK_SH` in declared order.
    #[allow(dead_code)] // RAII only — Drop releases flocks, no reads.
    pub(crate) locks: Vec<std::os::fd::OwnedFd>,
}

/// Maximum TOCTOU retry budget for the DISCOVER → PLAN → ACQUIRE
/// pipeline. Production sees up to `RETRIES + 1 = 4` attempts: one
/// initial DISCOVER and three retries. Between retries the caller
/// sleeps for an ascending micro-budget (10ms, 50ms, 200ms — see
/// [`TOCTOU_RETRY_DELAYS`]) so two peers that initially raced on the
/// same LLC have time to drop their fds before the next snapshot.
/// Without the sleep the second DISCOVER often sees the same holder
/// state and bails on a transient race; the in-process micro-sleep
/// absorbs that without paying the nextest-retry cost.
const ACQUIRE_MAX_TOCTOU_RETRIES: u32 = 3;

/// Per-retry sleep durations between DISCOVER attempts. Indexed by
/// the retry index: after attempt 0 fails the loop sleeps
/// `TOCTOU_RETRY_DELAYS[0]`, after attempt 1 fails it sleeps
/// `TOCTOU_RETRY_DELAYS[1]`, etc. Length must equal
/// [`ACQUIRE_MAX_TOCTOU_RETRIES`] — there are exactly that many
/// sleeps before the final attempt that can still bail.
const TOCTOU_RETRY_DELAYS: [std::time::Duration; ACQUIRE_MAX_TOCTOU_RETRIES as usize] = [
    std::time::Duration::from_millis(10),
    std::time::Duration::from_millis(50),
    std::time::Duration::from_millis(200),
];

/// DISCOVER phase — read-only LLC snapshot.
///
/// Walks ONLY the LLCs whose CPUs overlap `allowed` (the calling
/// process's `sched_getaffinity` cpuset). LLCs entirely outside the
/// cpuset are skipped — locking one would never contribute a
/// schedulable CPU to `plan.cpus`, and on a heavily-pinned runner
/// (CI cgroup with N out of M CPUs allowed) skipping them avoids
/// O(host_llcs - allowed_llcs) lockfile materializations and
/// /proc/locks lookups per attempt. The PLAN phase still receives a
/// snapshot vector indexed by `LlcSnapshot.llc_idx`, not by
/// position, so a sparse snapshot set works without any further
/// adjustment downstream.
///
/// For every selected LLC: stat the canonical lockfile (materializing
/// it with `O_CREAT | O_CLOEXEC | 0o666` if absent so subsequent
/// ACQUIRE has a stable inode), then parse one `/proc/locks` read to
/// populate every snapshot's holder list in a single pass. No flock
/// acquires — DISCOVER never contends.
///
/// `mountinfo` is the `/proc/self/mountinfo` text read once per
/// `acquire_llc_plan` invocation at [`acquire_llc_plan_with_acquire_fn`]
/// and threaded through here so a host with N LLCs pays for exactly
/// one mountinfo read per DISCOVER pass (DISCOVER runs once per retry
/// attempt — up to ACQUIRE_MAX_TOCTOU_RETRIES+1 — plus once on the
/// retry-exhausted diagnostic path, up to 5 passes, hence caching at
/// the plan level rather than per snapshot walk).
///
/// Returns `Ok(snapshots)` on success. Propagates opening + stat
/// errors so a missing `/tmp` or permission failure surfaces
/// actionably.
fn discover_llc_snapshots(
    topo: &HostTopology,
    allowed: &std::collections::BTreeSet<usize>,
    mountinfo: &str,
) -> Result<Vec<LlcSnapshot>> {
    let mut snapshots: Vec<LlcSnapshot> = Vec::with_capacity(topo.llc_groups.len());
    for llc_idx in 0..topo.llc_groups.len() {
        // Skip LLCs whose CPUs are entirely outside the calling
        // process's allowed cpuset — they cannot contribute a
        // schedulable CPU to `plan.cpus`, and locking one would just
        // pay for a lockfile + /proc/locks pass without coordination
        // value. The sparse snapshot vector keeps llc_idx as the
        // identity key, so PLAN's index-based iteration is
        // unaffected.
        if !topo.llc_groups[llc_idx]
            .cpus
            .iter()
            .any(|c| allowed.contains(c))
        {
            continue;
        }
        let path = std::path::PathBuf::from(llc_lock_path(llc_idx));
        // Ensure the lockfile inode exists so `read_holders_with_mountinfo`
        // can key /proc/locks lookups on it. Deliberately takes no
        // flock — DISCOVER is observational. Also runs the NFS/FUSE
        // reject check inside `materialize`, so a misconfigured
        // `/tmp` mount surfaces here instead of silently at ACQUIRE
        // time.
        crate::flock::materialize(&path)?;
        let holders =
            crate::flock::read_holders_with_mountinfo(&path, mountinfo).unwrap_or_default();
        // Exclude the calling process from the PLAN-driving holder
        // count. No-perf `build()` now holds this LLC's `LOCK_SH` from
        // build through run, so at the run-time replan our own
        // build-time fd shows up in `/proc/locks` for this very LLC.
        // Counting it would make every LLC we already reserved look one
        // holder busier to ourselves and push the Spread policy to FLEE
        // our own reservation onto a different LLC — the opposite of the
        // truthful-holder-count fix. At build time we hold nothing yet,
        // so this filter is a no-op there. The full `holders` vec is
        // kept intact for diagnostics / `ktstr locks`; only the sort key
        // drops self.
        let holder_count = holders
            .iter()
            .filter(|h| h.pid != std::process::id())
            .count();
        snapshots.push(LlcSnapshot {
            llc_idx,
            lockfile_path: path,
            holders,
            holder_count,
        });
    }
    Ok(snapshots)
}

/// PLAN phase — NUMA-aware placement over discover snapshots.
///
/// Composite sort driven by three ordered keys:
///   1. Placement policy — [`PlacementPolicy::Consolidate`] prefers
///      LLCs already holding peers (holder_count DESC);
///      [`PlacementPolicy::Spread`] prefers the least-held LLCs
///      (holder_count ASC) with a caller-rotated tiebreak.
///   2. NUMA locality — after seeding on the highest-scored LLC's
///      node, greedily fill the seed node before spilling.
///   3. LLC index tiebreak — ASC for Consolidate, ASC rotated by the
///      policy's offset for Spread. The final ACQUIRE ordering is
///      plain ASC under BOTH policies (step e re-sorts), so every
///      concurrent acquirer walks lockfiles in one global order and
///      the livelock-safety argument is policy-independent.
///
/// `target_cpus` is the exact number of allowed CPUs the plan
/// reserves. The walk selects whole LLCs (filtered to their
/// allowed-CPU overlap) until the accumulated contribution meets
/// the budget. The LAST selected LLC may contribute more allowed
/// CPUs than the remaining budget needs; the materialization layer
/// at [`acquire_llc_plan_with_acquire_fn`] takes only the needed
/// prefix of that LLC's allowed CPUs into `plan.cpus`. The flock
/// is always held at LLC granularity — coordination with concurrent
/// ktstr peers happens per-LLC, regardless of how many of the LLC's
/// CPUs are consumed here. LLCs whose CPUs are all outside
/// `allowed` are skipped entirely — locking one would never
/// contribute a schedulable CPU to `plan.cpus`.
///
/// Distance fallback: callers without a distance matrix pass a closure
/// that returns `10` for equal nodes and `20` otherwise — primitive 3
/// keeps the spill order reasonable even on hosts whose
/// `/sys/devices/system/node/*/distance` is unavailable.
fn plan_from_snapshots(
    snapshots: &[LlcSnapshot],
    target_cpus: usize,
    topo: &HostTopology,
    allowed: &std::collections::BTreeSet<usize>,
    distance_fn: impl Fn(usize, usize) -> u8,
    policy: PlacementPolicy,
) -> Vec<usize> {
    if target_cpus == 0 {
        return Vec::new();
    }

    // Allowed-CPU count contributed by each LLC. An LLC with zero
    // overlap contributes no schedulable CPUs to `plan.cpus`, so
    // reserving it adds a useless flock and no planning value — drop
    // those up front so every subsequent walk only considers
    // candidates that can actually carry budget.
    let llc_allowed_cpus = |idx: usize| -> usize {
        topo.llc_groups[idx]
            .cpus
            .iter()
            .filter(|c| allowed.contains(c))
            .count()
    };
    let total_allowed_in_llcs: usize = snapshots
        .iter()
        .map(|snapshot| llc_allowed_cpus(snapshot.llc_idx))
        .sum();
    if target_cpus >= total_allowed_in_llcs {
        // Budget ≥ sum of per-LLC contributions: select every LLC
        // that has at least one allowed CPU, in ascending order.
        // Short-circuits the scoring walk when the cap degenerates
        // to "reserve everything we can schedule on."
        let mut all: Vec<usize> = snapshots
            .iter()
            .map(|snapshot| snapshot.llc_idx)
            .filter(|&idx| llc_allowed_cpus(idx) > 0)
            .collect();
        all.sort_unstable();
        return all;
    }

    // Step a: partition + sort. Only LLCs with at least one allowed
    // CPU are eligible — locking an out-of-cpuset LLC is useless.
    let eligible = |s: &&LlcSnapshot| -> bool { llc_allowed_cpus(s.llc_idx) > 0 };
    let ranked: Vec<&LlcSnapshot> = match policy {
        // Consolidation candidates first (holder_count DESC, llc_idx
        // ASC); fresh candidates after, sorted by llc_idx ASC. A
        // single composite sort would do the same work but the
        // two-partition form is easier to read and lets future
        // "prefer consolidation only if score ≥ threshold" tweaks
        // slot in.
        PlacementPolicy::Consolidate => {
            let mut consolidation: Vec<&LlcSnapshot> = snapshots
                .iter()
                .filter(|s| s.holder_count > 0)
                .filter(eligible)
                .collect();
            let mut fresh: Vec<&LlcSnapshot> = snapshots
                .iter()
                .filter(|s| s.holder_count == 0)
                .filter(eligible)
                .collect();
            consolidation.sort_by(|a, b| {
                b.holder_count
                    .cmp(&a.holder_count)
                    .then(a.llc_idx.cmp(&b.llc_idx))
            });
            fresh.sort_by_key(|s| s.llc_idx);
            consolidation.into_iter().chain(fresh).collect()
        }
        // Least-held first, ties broken by rotated eligible position
        // so simultaneous planners with identical (typically
        // all-zero) holder snapshots fan out across the host instead
        // of converging on the same LLC-0-upward prefix. The rotation
        // is reduced modulo the ELIGIBLE count — rotating over raw
        // LLC indices would bias toward whichever indices survive the
        // cpuset filter.
        PlacementPolicy::Spread { rotation } => {
            let mut spread: Vec<&LlcSnapshot> = snapshots.iter().filter(eligible).collect();
            spread.sort_by_key(|s| s.llc_idx);
            let n = spread.len();
            if n > 0 {
                let start = rotation % n;
                // Rotated position of each eligible snapshot in the
                // llc_idx-ASC ordering (`start` maps to 0), keyed by
                // llc_idx. Precomputed: the comparator below cannot
                // itself search `spread` mid-sort.
                let rotated_pos: std::collections::HashMap<usize, usize> = spread
                    .iter()
                    .enumerate()
                    .map(|(p, s)| (s.llc_idx, (p + n - start) % n))
                    .collect();
                spread.sort_by(|a, b| {
                    a.holder_count
                        .cmp(&b.holder_count)
                        .then(rotated_pos[&a.llc_idx].cmp(&rotated_pos[&b.llc_idx]))
                });
            }
            spread
        }
    };
    if ranked.is_empty() {
        // No LLC on this host overlaps the caller's allowed cpuset.
        // Bail upstream handles this as ResourceContention; here we
        // just return empty so the caller can surface the diagnostic.
        return Vec::new();
    }

    // Step b: seed. Highest-scored eligible LLC; its NUMA node
    // anchors the greedy expansion.
    let seed = ranked[0];
    let seed_node = topo.llc_numa_node(seed.llc_idx);

    // Step c–d: walk seed-node LLCs first, then spill to
    // nearest-by-distance nodes. Primitives 1 + 3 drive the node
    // ordering; the per-node LLC lists come from primitive 1. Within
    // each node, we still honour the composite score by walking
    // `ranked` and skipping LLCs not on the current target node.
    // Accumulation is by allowed-CPU contribution — an LLC with 4
    // CPUs of which 2 are in `allowed` counts as 2 toward the
    // budget and the other 2 never appear in `plan.cpus`.
    let node_order = topo.numa_nodes_sorted_by_distance(seed_node, distance_fn);
    let mut selected: Vec<usize> = Vec::new();
    let mut picked: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut accumulated: usize = 0;
    for node in node_order {
        if accumulated >= target_cpus {
            break;
        }
        // Ranked walk, taking every candidate on this node in
        // score-order until we've filled `target_cpus` or exhausted
        // the node.
        for snap in &ranked {
            if accumulated >= target_cpus {
                break;
            }
            if picked.contains(&snap.llc_idx) {
                continue;
            }
            if topo.llc_numa_node(snap.llc_idx) != node {
                continue;
            }
            selected.push(snap.llc_idx);
            picked.insert(snap.llc_idx);
            accumulated += llc_allowed_cpus(snap.llc_idx);
        }
    }

    // Step e: livelock-proof acquire order — ascending index.
    selected.sort_unstable();
    selected
}

/// ACQUIRE phase — non-blocking `LOCK_SH` on every selected LLC.
///
/// All-or-nothing. A single `EWOULDBLOCK` releases every held fd (via
/// `drop(locks)`) and returns `Ok(None)` so the caller re-runs
/// discover + plan with a fresh snapshot. Non-retryable errors
/// (unexpected errno, path open failures) propagate unchanged.
fn try_acquire_llc_plan_locks(
    selected: &[usize],
    snapshots: &[LlcSnapshot],
) -> Result<Option<Vec<std::os::fd::OwnedFd>>> {
    let mut locks: Vec<std::os::fd::OwnedFd> = Vec::with_capacity(selected.len());
    for &idx in selected {
        let snap = snapshots
            .iter()
            .find(|s| s.llc_idx == idx)
            .expect("selected index must come from snapshots — plan invariant");
        match crate::flock::try_flock(&snap.lockfile_path, FlockMode::Shared)? {
            Some(fd) => locks.push(fd),
            None => {
                // Drop previously-held fds so the peer racing us sees
                // a consistent post-bail state, then signal "retry".
                drop(locks);
                return Ok(None);
            }
        }
    }
    Ok(Some(locks))
}

/// Entry point for the `--cpu-cap` PLAN pipeline.
///
/// Runs DISCOVER → PLAN → ACQUIRE with up to
/// [`ACQUIRE_MAX_TOCTOU_RETRIES`] retries (each separated by a
/// per-retry sleep from [`TOCTOU_RETRY_DELAYS`]) as the acquisition
/// protocol's non-blocking FAST PHASE — claim-subtracted and
/// all-or-nothing (see [`protocol`]). `wait == false` (builds, the
/// interactive shell) bails with `ResourceContention` the moment that
/// budget is spent. `wait == true` (the test run path) then joins the
/// cross-invocation queue and, as head, RE-PLANS AGAINST LIVE HOLDER
/// STATE ON EVERY WAKE — plans are never cached across waits — waiting
/// for a genuine holder's authoritative flock release rather than
/// skipping.
/// On
/// success returns an [`LlcPlan`] holding the selected LLCs, their
/// flattened CPUs (intersected with the calling process's allowed
/// cpuset), the derived `mems` set, the diagnostic snapshot, and the
/// RAII flock handles.
///
/// `cpu_cap == None` means "reserve 30% of the allowed-CPU set" (see
/// [`default_cpu_budget`]). `cpu_cap == Some(cap)` where
/// `cap > allowed_cpus` errors at acquire time via
/// [`CpuCap::effective_count`]. The allowed-CPU set comes from
/// [`host_allowed_cpus`] — `sched_getaffinity(0)` with a procfs
/// fallback — so plans are always schedulable under cgroup-restricted
/// runners (CI hosts, systemd slices, sudo under a limited cpuset).
///
/// `policy` picks the placement preference among eligible LLCs —
/// [`PlacementPolicy::Consolidate`] for builds,
/// [`PlacementPolicy::Spread`] for no-perf VM placement (see the enum
/// docs for the clustering failure Spread exists to prevent). Both
/// policies use the host distance matrix from [`crate::topology::TestTopology`]
/// so spill order matches actual NUMA cost. Hosts whose
/// `/sys/devices/system/node/*/distance` failed to parse degrade to a
/// numerically-adjacent ordering via the distance closure (`10` for
/// same-node, `20` for cross-node).
pub fn acquire_llc_plan(
    topo: &HostTopology,
    test_topo: &crate::topology::TestTopology,
    cpu_cap: Option<CpuCap>,
    policy: PlacementPolicy,
    wait: bool,
) -> Result<LlcPlan> {
    acquire_llc_plan_impl(topo, test_topo, cpu_cap, policy, wait, None)
}

/// Cancellation-aware waiting LLC-plan acquisition.
///
/// This is the harness-build reservation path: normal contention retains one
/// FIFO queue ticket, while a published cancellation interrupts either the
/// queue flock or the head's inotify sleep and releases every partial hold.
pub fn acquire_llc_plan_interruptible(
    topo: &HostTopology,
    test_topo: &crate::topology::TestTopology,
    cpu_cap: Option<CpuCap>,
    policy: PlacementPolicy,
    cancelled: &AtomicBool,
) -> Result<LlcPlan> {
    acquire_llc_plan_impl(topo, test_topo, cpu_cap, policy, true, Some(cancelled))
}

fn acquire_llc_plan_impl(
    topo: &HostTopology,
    test_topo: &crate::topology::TestTopology,
    cpu_cap: Option<CpuCap>,
    policy: PlacementPolicy,
    wait: bool,
    cancelled: Option<&AtomicBool>,
) -> Result<LlcPlan> {
    check_acquire_cancelled(cancelled)?;
    if crate::cargo_test_mode::cargo_test_mode_active() {
        // Bare `cargo test` mode: no peer-coordination contract.
        // Synthesise a degenerate plan that names every LLC and
        // every allowed CPU but holds no flocks. The vmm caller
        // strips `locks` after build (see `KtstrVmBuilder::build`)
        // and re-acquires via `acquire_resource_locks` at run time
        // — also short-circuited above. `cpus` is the calling
        // process's allowed cpuset so the `sched_setaffinity`
        // sites inside the vmm have a valid mask to apply
        // (allowed cpuset = whatever the OS schedules us onto).
        let allowed = host_allowed_cpus();
        if allowed.is_empty() {
            return Err(ResourceContention {
                reason: "could not determine allowed CPU set \
                         (sched_getaffinity and /proc/self/status both failed)"
                    .into(),
            }
            .into());
        }
        let _ = test_topo;
        let _ = cpu_cap;
        let allowed_set: std::collections::BTreeSet<usize> = allowed.iter().copied().collect();
        let locked_llcs: Vec<usize> = topo
            .llc_groups
            .iter()
            .enumerate()
            .filter_map(|(idx, group)| {
                if group.cpus.iter().any(|c| allowed_set.contains(c)) {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect();
        let mems: std::collections::BTreeSet<usize> = locked_llcs
            .iter()
            .filter_map(|&idx| {
                topo.llc_groups
                    .get(idx)
                    .and_then(|g| g.cpus.first().copied())
                    .and_then(|c| topo.cpu_to_node.get(&c).copied())
            })
            .collect();
        let plan = LlcPlan {
            locked_llcs,
            cpus: allowed,
            mems,
            snapshot: Vec::new(),
            locks: Vec::new(),
        };
        check_acquire_cancelled(cancelled)?;
        return Ok(plan);
    }
    acquire_llc_plan_with_acquire_fn(
        topo,
        test_topo,
        cpu_cap,
        policy,
        wait,
        cancelled,
        try_acquire_llc_plan_locks,
    )
}

fn check_acquire_cancelled(cancelled: Option<&AtomicBool>) -> Result<()> {
    if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        Err(std::io::Error::new(
            std::io::ErrorKind::Interrupted,
            "ktstr LLC-plan acquisition interrupted",
        )
        .into())
    } else {
        Ok(())
    }
}

/// Parameterized form of [`acquire_llc_plan`] that takes the
/// ACQUIRE closure as a seam. Production calls this with
/// [`try_acquire_llc_plan_locks`] (non-blocking `LOCK_SH` per LLC);
/// tests can pass a closure that returns `Ok(None)` on attempt 0 and
/// forwards on attempt 1 to simulate a peer winning the first race,
/// or an attempt-counting closure that always fails to exercise the
/// retry-exhausted error path.
///
/// `acquire_fn` receives `(selected, snapshots)` and returns
/// `Ok(Some(locks))` on success, `Ok(None)` to trigger a retry, or
/// propagates hard errors unchanged. Production closure is the
/// free-standing [`try_acquire_llc_plan_locks`]; the test closure
/// can track its own attempt counter via interior mutability
/// ([`std::cell::Cell`], `Mutex`, atomic int).
///
/// The FAST-PHASE loop body — DISCOVER, PLAN, retry budget, final
/// holder diagnostics — is shared between both entry points so the
/// test seam exercises the exact retry-and-diagnose sequence
/// production uses, not a parallel implementation. (`acquire_fn` is
/// a fast-phase seam only: the wait phase acquires through the
/// protocol head engine against real lockfiles.)
///
/// `wait` separates the two contention regimes. The first
/// [`ACQUIRE_MAX_TOCTOU_RETRIES`] retries always fire (with the short
/// [`TOCTOU_RETRY_DELAYS`]) as the protocol fast path: non-blocking,
/// all-or-nothing, claim-subtracted — they absorb plan/acquire
/// *races* (a peer that grabbed a slot between our DISCOVER and
/// ACQUIRE). Beyond that budget the behaviour forks: `wait == false`
/// (builds / interactive) bails with `ResourceContention`; `wait ==
/// true` (the test run path) joins the acquisition queue and, once
/// head, re-runs DISCOVER→PLAN against LIVE holder state on every
/// lock-dir wake — the re-plan-on-wake contract — accumulating
/// `LOCK_SH` on the freshly selected LLCs under the head license
/// (partials retained across re-plans exactly when the new plan still
/// selects them) until the plan completes. Waiting is event-driven
/// (inotify on the lock dir), never polled; lifecycle bounds belong to
/// the holder's VM watchdog and the nextest process rail.
fn acquire_llc_plan_with_acquire_fn<F>(
    topo: &HostTopology,
    test_topo: &crate::topology::TestTopology,
    cpu_cap: Option<CpuCap>,
    policy: PlacementPolicy,
    wait: bool,
    cancelled: Option<&AtomicBool>,
    mut acquire_fn: F,
) -> Result<LlcPlan>
where
    F: FnMut(&[usize], &[LlcSnapshot]) -> Result<Option<Vec<std::os::fd::OwnedFd>>>,
{
    check_acquire_cancelled(cancelled)?;
    // Resolve the calling process's allowed cpuset. Plans must fit
    // inside this set — sched_setaffinity against a mask outside the
    // process's cgroup cpuset either fails outright or produces an
    // empty effective set (the vCPU thread then cannot run). Reading
    // the syscall ONCE here and threading it through means every
    // TOCTOU retry sees the same baseline; a cgroup change mid-plan
    // is a host-reconfiguration event the retry budget does not
    // attempt to absorb.
    let allowed_vec = host_allowed_cpus();
    if allowed_vec.is_empty() {
        return Err(ResourceContention {
            reason: "could not determine allowed CPU set \
                     (sched_getaffinity and /proc/self/status both failed)"
                .into(),
        }
        .into());
    }
    let allowed: std::collections::BTreeSet<usize> = allowed_vec.iter().copied().collect();
    let allowed_cpus = allowed.len();

    let target_cpus = match cpu_cap {
        Some(cap) => cap.effective_count(allowed_cpus)?,
        None => default_cpu_budget(allowed_cpus),
    };
    if target_cpus == 0 {
        // Defense in depth. `default_cpu_budget` has a `.max(1)`
        // floor and `effective_count` on a `NonZeroUsize` cap can
        // never return 0, but surfacing this as an explicit bail
        // catches future regressions (e.g. someone wires a signed
        // integer into the budget math) instead of silently
        // producing a plan with no locks.
        return Err(ResourceContention {
            reason: "CPU budget resolved to zero".into(),
        }
        .into());
    }

    // Read /proc/self/mountinfo ONCE per acquire_llc_plan invocation.
    // Every DISCOVER pass re-uses this text to derive per-LLC
    // /proc/locks needles (major:minor:inode). Without this cache, a
    // host with N LLCs would re-read mountinfo N× per DISCOVER pass,
    // and DISCOVER itself runs up to ACQUIRE_MAX_TOCTOU_RETRIES+1
    // times in the retry loop, plus once on the retry-exhausted
    // diagnostic path (up to 5 total). Mount points are
    // effectively static during a plan acquisition — a bind mount
    // changing under us mid-acquire is a host-reconfiguration event
    // that invalidates every parallel acquirer anyway, not something
    // we need to re-read to observe.
    let mountinfo = crate::flock::read_mountinfo().map_err(|e| ResourceContention {
        reason: format!("read /proc/self/mountinfo: {e}"),
    })?;
    check_acquire_cancelled(cancelled)?;

    // ---- FAST PHASE: TOCTOU-bounded, non-blocking, all-or-nothing,
    // claim-subtracted (protocol rules 2-4). Bounded attempts, then
    // either bail (no-wait callers) or join the queue.
    let mut attempt: u32 = 0;
    loop {
        check_acquire_cancelled(cancelled)?;
        let snapshots =
            discover_llc_snapshots(topo, &allowed, &mountinfo).map_err(|e| ResourceContention {
                reason: format!("discover LLC snapshots: {e}"),
            })?;
        // Subtract a live queue head's claim from the eligible set:
        // fast-path planners must not target the LLCs the head is
        // accumulating (protocol rule 2). If the remainder cannot
        // carry the budget, the attempt is a bounce — never an
        // under-budget plan.
        let claim = protocol::read_live_claim();
        let eligible: Vec<LlcSnapshot> = snapshots
            .iter()
            .filter(|snap| !claim.conflicts_with_llc(snap.llc_idx, FlockMode::Shared))
            .cloned()
            .collect();
        let claim_filtered = eligible.len() != snapshots.len();
        let eligible_capacity: usize = eligible
            .iter()
            .map(|snap| {
                topo.llc_groups[snap.llc_idx]
                    .cpus
                    .iter()
                    .filter(|c| allowed.contains(c))
                    .count()
            })
            .sum();
        let selected = if eligible_capacity >= target_cpus {
            plan_from_snapshots(
                &eligible,
                target_cpus,
                topo,
                &allowed,
                |from, to| test_topo.numa_distance(from, to),
                policy,
            )
        } else {
            Vec::new()
        };
        if selected.is_empty() && !claim_filtered {
            // Every LLC's CPU set lies outside the allowed cpuset —
            // sysfs disagrees with sched_getaffinity. This is a host
            // misconfiguration (stale sysfs after hotplug, cgroup
            // pinned to a CPU range the kernel no longer reports in
            // llc_groups, etc.). Bail with actionable text rather
            // than looping through retries that cannot change the
            // outcome.
            return Err(ResourceContention {
                reason: format!(
                    "no host LLC overlaps the process's \
                     {allowed_cpus}-CPU allowed set — sysfs LLC groups \
                     and sched_getaffinity disagree"
                ),
            }
            .into());
        }
        let acquired = if selected.is_empty() {
            // Claim-blocked: bounce without touching any lockfile.
            None
        } else {
            acquire_fn(&selected, &snapshots).map_err(|e| ResourceContention {
                reason: format!("acquire LLC locks: {e}"),
            })?
        };
        if let Some(locks) = acquired {
            let plan =
                materialize_llc_plan(selected, snapshots, locks, topo, &allowed, target_cpus);
            check_acquire_cancelled(cancelled)?;
            return Ok(plan);
        }
        if attempt >= ACQUIRE_MAX_TOCTOU_RETRIES {
            break;
        }
        // Short backoff between fast-phase attempts so a racing peer
        // has time to drop its fds before the next DISCOVER.
        std::thread::sleep(TOCTOU_RETRY_DELAYS[attempt as usize]);
        check_acquire_cancelled(cancelled)?;
        attempt = attempt.saturating_add(1);
    }

    if !wait {
        // Rebuild holder diagnostics from a FRESH read so the error
        // points at the peer that actually won.
        let final_snapshots = discover_llc_snapshots(topo, &allowed, &mountinfo)?;
        let holders: Vec<String> = final_snapshots
            .iter()
            .filter(|s| !s.holders.is_empty())
            .map(|s| {
                format!(
                    "LLC {}: {}",
                    s.llc_idx,
                    crate::flock::format_holder_list(&s.holders)
                )
            })
            .collect();
        let holder_text = if holders.is_empty() {
            "<none recorded>".to_string()
        } else {
            holders.join("; ")
        };
        return Err(anyhow::Error::new(ResourceContention {
            reason: format!(
                "acquire_llc_plan: could not reserve {target_cpus} \
                 CPU(s) after {attempts} attempts; holders: \
                 {holder_text}. Run `ktstr locks --json` to see \
                 every ktstr lock on this host.",
                attempts = ACQUIRE_MAX_TOCTOU_RETRIES + 1,
            ),
        }));
    }

    // ---- WAIT PHASE: queue up, then accumulate as head with
    // re-plan-on-every-wake (see `protocol`). Queued, this caller
    // holds NO resource locks (every fast-phase attempt above was
    // all-or-nothing); as head, its `LOCK_SH` partials are retained
    // across re-plans exactly when the fresh plan still selects them.
    check_acquire_cancelled(cancelled)?;
    let _queue = match cancelled {
        Some(flag) => protocol::wait_for_queue_turn_interruptible(flag)?,
        None => protocol::wait_for_queue_turn()?,
    };
    let step = |held: &mut protocol::HeldLocks| {
        // RE-PLAN against live holder state on every wake — plans are
        // never cached across waits. The freed capacity may satisfy a
        // different selection than the one that was busy last wake.
        let snapshots =
            discover_llc_snapshots(topo, &allowed, &mountinfo).map_err(|e| ResourceContention {
                reason: format!("discover LLC snapshots: {e}"),
            })?;
        let selected = plan_from_snapshots(
            &snapshots,
            target_cpus,
            topo,
            &allowed,
            |from, to| test_topo.numa_distance(from, to),
            policy,
        );
        if selected.is_empty() {
            return Ok(protocol::HeadStep::Abort {
                reason: format!(
                    "no host LLC overlaps the process's {allowed_cpus}-CPU \
                     allowed set — sysfs LLC groups and sched_getaffinity \
                     disagree"
                ),
            });
        }
        let target = protocol::canonical_lock_order(&selected, FlockMode::Shared, &[]);
        held.retain_paths(&target.iter().map(|(p, _)| p.clone()).collect());
        held.sweep(&target)?;
        if held.covers(&target) {
            let locks = held.take(&target);
            Ok(protocol::HeadStep::Complete((selected, snapshots, locks)))
        } else {
            Ok(protocol::HeadStep::Waiting {
                claim: protocol::ClaimSet::new(
                    selected.iter().copied(),
                    std::iter::empty(),
                    FlockMode::Shared,
                ),
            })
        }
    };
    let outcome = match cancelled {
        Some(flag) => protocol::acquire_as_head_interruptible(flag, step)?,
        None => protocol::acquire_as_head(step)?,
    };
    match outcome {
        protocol::HeadOutcome::Acquired((selected, snapshots, locks)) => {
            let plan =
                materialize_llc_plan(selected, snapshots, locks, topo, &allowed, target_cpus);
            check_acquire_cancelled(cancelled)?;
            Ok(plan)
        }
        protocol::HeadOutcome::Aborted { reason } => {
            Err(anyhow::Error::new(ResourceContention { reason }))
        }
    }
}

/// PLAN-ONLY variant of [`acquire_llc_plan`]: DISCOVER → PLAN →
/// materialize, taking NO locks. Used by the no-perf build path when
/// its non-blocking reservation is contended (a perf `LOCK_EX` head
/// or its claim covering the pool): the build plan only shapes setup
/// — budget size and affinity masks — and the run-time replan
/// re-acquires real locks through the acquisition queue, so failing
/// the build over a transient claim would only feed retry storms.
/// The returned plan's `locks` is empty by construction; peers'
/// DISCOVER will not see this cell's reservation until the run-time
/// replan takes real fds (an accepted truthfulness gap — the cell
/// owns nothing yet).
pub fn plan_llc_selection_only(
    topo: &HostTopology,
    test_topo: &crate::topology::TestTopology,
    cpu_cap: Option<CpuCap>,
    policy: PlacementPolicy,
) -> Result<LlcPlan> {
    let allowed_vec = host_allowed_cpus();
    if allowed_vec.is_empty() {
        return Err(ResourceContention {
            reason: "could not determine allowed CPU set \
                     (sched_getaffinity and /proc/self/status both failed)"
                .into(),
        }
        .into());
    }
    let allowed: std::collections::BTreeSet<usize> = allowed_vec.iter().copied().collect();
    let target_cpus = match cpu_cap {
        Some(cap) => cap.effective_count(allowed.len())?,
        None => default_cpu_budget(allowed.len()),
    };
    let mountinfo = crate::flock::read_mountinfo().map_err(|e| ResourceContention {
        reason: format!("read /proc/self/mountinfo: {e}"),
    })?;
    let snapshots =
        discover_llc_snapshots(topo, &allowed, &mountinfo).map_err(|e| ResourceContention {
            reason: format!("discover LLC snapshots: {e}"),
        })?;
    let selected = plan_from_snapshots(
        &snapshots,
        target_cpus,
        topo,
        &allowed,
        |from, to| test_topo.numa_distance(from, to),
        policy,
    );
    if selected.is_empty() {
        return Err(ResourceContention {
            reason: format!(
                "no host LLC overlaps the process's {}-CPU allowed set — \
                 sysfs LLC groups and sched_getaffinity disagree",
                allowed.len(),
            ),
        }
        .into());
    }
    Ok(materialize_llc_plan(
        selected,
        snapshots,
        Vec::new(),
        topo,
        &allowed,
        target_cpus,
    ))
}

/// Materialize the final [`LlcPlan`] from a selected LLC set and its
/// held locks: flatten each selected LLC's CPUs, intersecting with
/// `allowed` so `plan.cpus` never contains a CPU the process cannot
/// run on, and TRUNCATING at exactly `target_cpus` so the last-LLC
/// overshoot contributes only the prefix the budget needs. The full
/// LLC is still flocked (the coordination unit is per-LLC), but the
/// CPUs beyond `target_cpus` never appear in `plan.cpus` —
/// sched_setaffinity masks and cgroup cpuset.cpus writes reflect the
/// exact budget. `mems` collects the NUMA nodes of CPUs that actually
/// appear in `plan.cpus`; an LLC that contributes a partial slice on
/// a cross-node split only registers the nodes of its actually-used
/// CPUs. Shared by the fast-phase and head-phase success paths so the
/// two cannot drift.
fn materialize_llc_plan(
    selected: Vec<usize>,
    snapshots: Vec<LlcSnapshot>,
    locks: Vec<std::os::fd::OwnedFd>,
    topo: &HostTopology,
    allowed: &std::collections::BTreeSet<usize>,
    target_cpus: usize,
) -> LlcPlan {
    let mut cpus: Vec<usize> = Vec::new();
    let mut mems: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
    'outer: for &idx in &selected {
        let group = &topo.llc_groups[idx];
        for &cpu in &group.cpus {
            if !allowed.contains(&cpu) {
                continue;
            }
            if cpus.len() >= target_cpus {
                break 'outer;
            }
            cpus.push(cpu);
            let node = topo.cpu_to_node.get(&cpu).copied().unwrap_or(0);
            mems.insert(node);
        }
    }
    LlcPlan {
        locked_llcs: selected,
        cpus,
        mems,
        snapshot: snapshots,
        locks,
    }
}

/// Parallelism hint for `make -j{N}` when running under an
/// [`LlcPlan`] reservation. Returns the flattened host-CPU count
/// (`plan.cpus.len()`), clamped to at least 1 so a pathological empty
/// plan still produces a runnable command.
///
/// Rationale: without this hint, `make -j$(nproc)` fans gcc
/// children across every online CPU, defeating the --cpu-cap
/// reservation — the build escapes the cgroup cpuset in scheduling
/// terms even though the kernel enforces CPU membership. Passing
/// `plan.cpus.len()` to make keeps gcc's parallel width aligned with
/// the reserved capacity.
pub fn make_jobs_for_plan(plan: &LlcPlan) -> usize {
    plan.cpus.len().max(1)
}

/// Render selected LLC indices for user-facing warning text.
///
/// Format is compact and stable: `[0 (node 0), 2 (node 1)]` when the
/// host exposes NUMA information, `[0, 2]` on degraded hosts whose
/// `cpu_to_node` map is empty. Used by
/// [`warn_if_cross_node_spill`] to render the `ktstr: reserving LLCs
/// …` message when an `--cpu-cap` plan spills across nodes.
pub fn format_llc_list(locked: &[usize], topo: &HostTopology) -> String {
    let parts: Vec<String> = locked
        .iter()
        .map(|&idx| {
            if topo.cpu_to_node.is_empty() {
                idx.to_string()
            } else {
                let node = topo.llc_numa_node(idx);
                format!("{idx} (node {node})")
            }
        })
        .collect();
    format!("[{}]", parts.join(", "))
}

/// Emit the cross-node spill warning when an `--cpu-cap` plan's
/// `mems` set spans more than one NUMA node. No-op for single-node
/// plans.
///
/// `eprintln!`, not `tracing::warn!`: this is user-visible
/// UX feedback (the operator picked a cap that couldn't fit in one
/// NUMA node), not operational instrumentation. Fires at most once
/// per plan — there is nothing in the plan lifecycle that causes a
/// re-trigger. Single-node plans (including single-socket hosts and
/// caps that fit within a single node) never emit.
///
/// Placement: called by `kernel_build_pipeline` and friends right
/// after [`acquire_llc_plan`] returns, before the sandbox mount.
/// Extracting this into a helper rather than inlining at the call
/// site lets the message body be unit-tested via
/// [`cross_node_spill_warning`] without capturing stderr.
pub fn warn_if_cross_node_spill(plan: &LlcPlan, topo: &HostTopology) {
    if let Some(msg) = cross_node_spill_warning(plan, topo) {
        eprintln!("{msg}");
    }
}

/// Build the cross-node spill warning string for `plan`, or `None`
/// when the plan fits within a single NUMA node (the suppression
/// case). [`warn_if_cross_node_spill`] is a thin wrapper that
/// `eprintln!`s the `Some` value; this function holds the actual
/// gate-and-format logic so a test can pin both halves — the
/// predicate gate AND the rendered message — without a stderr
/// capture seam. The returned string is exactly the bytes
/// `warn_if_cross_node_spill` would emit (sans the trailing newline
/// that `eprintln!` appends).
fn cross_node_spill_warning(plan: &LlcPlan, topo: &HostTopology) -> Option<String> {
    if !should_warn_cross_node(&plan.mems) {
        return None;
    }
    Some(format!(
        "ktstr: reserving LLCs {list} across {n} NUMA nodes \
         (preferred single-node contiguous unavailable). Work \
         will proceed; memory-access latency may be higher.",
        list = format_llc_list(&plan.locked_llcs, topo),
        n = plan.mems.len(),
    ))
}

/// Pure predicate backing [`warn_if_cross_node_spill`]. Returns
/// `true` when the plan spans more than one NUMA node
/// (`mems.len() > 1`); the warning suppression for single-node
/// plans follows directly from this.
///
/// Split out so tests can pin the polarity of the single-node /
/// multi-node decision without capturing stderr. A refactor that
/// accidentally flipped the comparison (`>= 1` or `== 1`) would
/// either warn on every plan (noise) or never warn (silent cost),
/// both of which the test suite catches here before the stderr
/// capture layer sees it.
fn should_warn_cross_node(mems: &std::collections::BTreeSet<usize>) -> bool {
    mems.len() > 1
}

/// Diagnostic text when the effective host-CPU budget is below the guest's
/// vCPU count, else `None`.
///
/// `intentional` splits severity: no-perf / an explicit CPU budget requested
/// shared execution, while the default placement fallback only overcommits
/// because the host cannot provide a 1:1 mapping. Keep the text factual: this
/// is a placement observation, not a performance-model recommendation or a
/// claim about the guest scheduler's own watchdog tuning.
///
/// Pure (returns the text) so a test pins the message + the
/// `None`-when-not-oversubscribed polarity without capturing stderr; the
/// caller eprintln's it once at build time, mirroring
/// [`warn_if_cross_node_spill`].
pub(crate) fn overcommit_warning(
    effective_host_cpus: usize,
    vcpus: usize,
    intentional: bool,
) -> Option<String> {
    if effective_host_cpus >= vcpus {
        return None;
    }
    let oversub = vcpus as f64 / effective_host_cpus.max(1) as f64;
    let msg = if intentional {
        format!(
            "ktstr: {vcpus} guest vCPUs share {effective_host_cpus} host CPUs \
             ({oversub:.1}x oversubscribed; no-perf/cpu-budget mode)"
        )
    } else {
        format!(
            "ktstr: WARNING: {vcpus} guest vCPUs share {effective_host_cpus} host \
             CPUs ({oversub:.1}x oversubscribed; host capacity is below guest width)"
        )
    };
    Some(msg)
}

/// Whether [`mbind_to_nodes`] must short-circuit before touching `addr`
/// or invoking the `mbind(2)` syscall. Returns `true` when there is no
/// work to do — an empty node set (no policy target) or a zero-length
/// region. This is the exact guard [`mbind_to_nodes`] consults; it is a
/// pure predicate so the short-circuit decision can be asserted directly
/// instead of inferred from a not-crashing call (whose pass condition the
/// syscall's own error-swallowing would satisfy regardless of the guard).
fn mbind_should_skip(len: usize, nodes: &[usize]) -> bool {
    nodes.is_empty() || len == 0
}

/// Bind a memory region to specific NUMA nodes using `mbind(MPOL_BIND)`.
/// `nodes` is the set of NUMA node IDs. Logs a warning on error
/// (single-node systems, missing capabilities).
///
/// # Safety
///
/// The caller must ensure that `addr` points to a valid mmap'd region
/// of at least `len` bytes. The kernel will read this range via the
/// `mbind(2)` syscall to set its NUMA memory policy; passing a stale,
/// unmapped, or out-of-bounds pointer is undefined behavior from the
/// process's perspective (the syscall itself returns EFAULT, but the
/// surrounding Rust contract is violated).
///
/// When `nodes.is_empty()` or `len == 0`, the function short-circuits
/// without dereferencing `addr`, so a null or dangling pointer is
/// permitted in those cases.
pub unsafe fn mbind_to_nodes(addr: *mut u8, len: usize, nodes: &[usize]) {
    if mbind_should_skip(len, nodes) {
        return;
    }
    let node_set: std::collections::BTreeSet<usize> = nodes.iter().copied().collect();
    let (nodemask, maxnode) = crate::workload::build_nodemask(&node_set);

    let rc = unsafe {
        libc::syscall(
            libc::SYS_mbind,
            addr as *mut libc::c_void,
            len,
            libc::MPOL_BIND,
            nodemask.as_ptr(),
            maxnode,
            0u32,
        )
    };
    if rc == 0 {
        eprintln!(
            "performance_mode: mbind {} MB to NUMA node(s) {:?}",
            len >> 20,
            nodes,
        );
    } else {
        let err = std::io::Error::last_os_error();
        eprintln!(
            "performance_mode: WARNING: mbind to node(s) {:?} failed: {err}",
            nodes,
        );
    }
}

use crate::topology::parse_cpu_list_lenient;

/// Number of free 2MB hugepages on the host.
pub fn hugepages_free() -> u64 {
    hugepages_free_from(std::path::Path::new(
        "/sys/kernel/mm/hugepages/hugepages-2048kB/free_hugepages",
    ))
}

/// Path-parameterized core of [`hugepages_free`]. Reads the
/// `free_hugepages` sysfs file at `path`, parses the trimmed count, and
/// returns 0 when the file is absent, unreadable, or contains a value
/// that does not parse as a `u64`. Exposes a path seam so the parse and
/// the documented 0-fallback can be tested against fixture files without
/// depending on the host's hugetlbfs configuration.
fn hugepages_free_from(path: &std::path::Path) -> u64 {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(0)
}

/// Estimate the number of 2 MiB hugepages needed for a given memory size in MiB.
pub fn hugepages_needed(memory_mib: u32) -> u64 {
    // 2 MiB per hugepage.
    (memory_mib as u64).div_ceil(2)
}

/// Estimate current host CPU load by checking /proc/stat.
/// Returns (busy_cpus, total_cpus) as a rough estimate.
pub fn host_load_estimate() -> Option<(usize, usize)> {
    // Count processes in R state from /proc/stat.
    let stat = std::fs::read_to_string("/proc/stat").ok()?;
    let procs_running = stat
        .lines()
        .find(|l| l.starts_with("procs_running "))?
        .split_whitespace()
        .nth(1)?
        .parse::<usize>()
        .ok()?;
    let online = std::fs::read_to_string("/sys/devices/system/cpu/online").ok()?;
    let total = parse_cpu_list_lenient(&online).len();
    Some((procs_running, total))
}

#[cfg(test)]
mod tests;
