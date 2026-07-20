//! Host CPU topology discovery for performance_mode.
//!
//! Wraps [`TestTopology`](crate::topology::TestTopology) for LLC-aware
//! vCPU pinning and host resource validation.

use anyhow::{Context, Result};
use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};

// Advisory flock primitives live in `crate::flock` so both LLC +
// per-CPU coordination here and per-cache-entry coordination in
// `crate::cache` share one `try_flock` implementation (with a single
// `O_CLOEXEC` source of truth) plus one `HolderInfo` /proc/locks
// parser. Re-importing the names keeps existing in-module call sites
// (production + `super::*` tests) compiling unchanged.
#[cfg(test)]
use crate::flock::try_flock;
use crate::flock::{FlockMode, TryFlockOutcome, try_flock_with_witness};

// Cross-invocation acquisition protocol: exact fixed-record claims,
// work-conserving grants, the coordinator watch, and lifecycle-bound blocking.
// See protocol.rs's module doc for the full model.
pub(crate) mod protocol;

const RESERVATION_WAIT_PROGRESS_POLL: std::time::Duration = std::time::Duration::from_secs(1);

thread_local! {
    /// Optional same-thread progress callbacks around a blocking admission
    /// acquire. A stack keeps nested library callers well-defined without a
    /// process-global reporter or helper thread.
    static RESERVATION_WAIT_PROGRESS: RefCell<Vec<Box<dyn FnMut()>>> =
        RefCell::new(Vec::new());
}

struct ReservationWaitProgressScope;

impl Drop for ReservationWaitProgressScope {
    fn drop(&mut self) {
        RESERVATION_WAIT_PROGRESS.with(|callbacks| {
            drop(
                callbacks
                    .borrow_mut()
                    .pop()
                    .expect("reservation wait progress scope remains installed"),
            );
        });
    }
}

pub(crate) fn with_reservation_wait_progress<T>(
    progress: impl FnMut() + 'static,
    operation: impl FnOnce() -> T,
) -> T {
    RESERVATION_WAIT_PROGRESS.with(|callbacks| {
        callbacks.borrow_mut().push(Box::new(progress));
    });
    let _scope = ReservationWaitProgressScope;
    operation()
}

fn tick_reservation_wait_progress() {
    RESERVATION_WAIT_PROGRESS.with(|callbacks| {
        if let Some(progress) = callbacks.borrow_mut().last_mut() {
            progress();
        }
    });
}

fn reservation_wait_progress_poll() -> Option<std::time::Duration> {
    RESERVATION_WAIT_PROGRESS.with(|callbacks| {
        (!callbacks.borrow().is_empty()).then_some(RESERVATION_WAIT_PROGRESS_POLL)
    })
}

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
/// returned by the `performance_mode` topology planner when the
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
    /// `BTreeMap` (not `HashMap`) gives deterministic iteration order — two ktstr
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
    /// the planner selects performance-mode isolation.
    pub service_cpu: Option<usize>,
    /// Host LLC group indices used by this plan, sorted.
    pub llc_indices: Vec<usize>,
    /// Held flock fds for resource reservation. Dropped when the plan
    /// (and the KtstrVm holding it) is dropped, releasing all locks.
    #[allow(dead_code)] // RAII: flock fds released on Drop, not read after construction.
    pub(crate) locks: Vec<std::os::fd::OwnedFd>,
}

/// Which run path is asking the shared topology planner for placements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PinningKind {
    /// Default runs share touched LLCs and exclusively own the exact CPUs
    /// assigned to guest vCPUs.
    Default,
    /// Performance runs additionally reserve a service CPU and choose whole
    /// LLC or CPU-grain isolation from the occupied cache domains.
    Performance,
}

/// Non-authoritative, mode-aware availability snapshot used to order one
/// bounded planner pass. Complete registry claims and flocks remain the
/// authority; these sets only steer the matcher toward resources the
/// coordinator just observed ready.
#[derive(Clone, Copy)]
pub(crate) struct PinningPreferences<'a> {
    pub(crate) cpus: &'a std::collections::BTreeSet<usize>,
    pub(crate) shared_llcs: &'a std::collections::BTreeSet<usize>,
    pub(crate) exclusive_llcs: &'a std::collections::BTreeSet<usize>,
}

/// One exact placement and its complete reservation footprint.
///
/// Keeping the CPU footprint beside the plan is important for whole-LLC
/// performance reservations: those take both the LLC EX lock and CPU EX locks
/// for every CPU in that domain (including siblings outside this process's
/// cpuset). The CPU locks are the admission
/// bridge to topology-unavailable overcommitters, which can name CPUs but not
/// LLCs.
pub(crate) struct PerformancePinningCandidate {
    pub(crate) plan: PinningPlan,
    pub(crate) llc_mode: LlcLockMode,
    pub(crate) cpu_mode: FlockMode,
    pub(crate) cpu_reservations: Vec<usize>,
}

impl PerformancePinningCandidate {
    #[cfg(test)]
    pub(crate) fn claim(&self) -> protocol::ClaimSet {
        resource_claim_with_modes(
            &self.plan.llc_indices,
            self.llc_mode,
            &self.cpu_reservations,
            self.cpu_mode,
        )
    }
}

#[derive(Debug, Clone, Copy)]
struct GuestLlcDemand {
    guest_llc: u32,
    guest_node: u32,
    vcpu_start: u32,
    cpus: usize,
}

#[derive(Clone, Copy)]
struct PlannerInputs<'a> {
    demands: &'a [GuestLlcDemand],
    kind: PinningKind,
    allowed: &'a std::collections::BTreeSet<usize>,
    allowed_rank: &'a std::collections::BTreeMap<usize, usize>,
    preferred_cpus: &'a std::collections::BTreeSet<usize>,
    preferred_shared_llcs: &'a std::collections::BTreeSet<usize>,
    preferred_exclusive_llcs: &'a std::collections::BTreeSet<usize>,
    performance_prefers_shared: bool,
}

impl PinningPlan {
    /// Duplicate the plan's DESCRIPTION (assignments, service CPU,
    /// LLC indices) with an EMPTY lock set. `PinningPlan` cannot be
    /// `Clone` — the fds are RAII lock holders — but candidate plans
    /// (pure topology-planner output, no locks yet) need copying
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

fn validate_llc_partition(llc_groups: &[LlcGroup], online_cpus: &[usize]) -> Result<()> {
    if online_cpus.is_empty() {
        anyhow::bail!("host CPU discovery returned an empty online CPU set");
    }
    if llc_groups.is_empty() {
        anyhow::bail!("host LLC discovery returned zero LLC groups");
    }
    let online = online_cpus
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    if online.len() != online_cpus.len() {
        anyhow::bail!("online CPU discovery contains duplicate CPU IDs");
    }

    let mut partition = std::collections::BTreeSet::new();
    for (llc, group) in llc_groups.iter().enumerate() {
        if group.cpus.is_empty() {
            anyhow::bail!("host LLC group {llc} is empty");
        }
        let mut within = std::collections::BTreeSet::new();
        for &cpu in &group.cpus {
            if !within.insert(cpu) {
                anyhow::bail!("host LLC group {llc} repeats CPU {cpu}");
            }
            if !online.contains(&cpu) {
                anyhow::bail!("host LLC group {llc} contains non-online CPU {cpu}");
            }
            if !partition.insert(cpu) {
                anyhow::bail!("online CPU {cpu} appears in more than one host LLC group");
            }
        }
    }
    if partition != online {
        let missing = online.difference(&partition).copied().collect::<Vec<_>>();
        anyhow::bail!("host LLC groups omit online CPUs {missing:?}");
    }
    Ok(())
}

impl HostTopology {
    /// Read the host-global physical topology from sysfs. Per-process affinity
    /// is applied later by placement, never while assigning LLC lock identity.
    pub fn from_sysfs() -> Result<Self> {
        let topo = crate::topology::TestTopology::from_system_unfiltered()
            .context("read host-global topology from sysfs")?;
        let online_cpus = topo.all_cpus().to_vec();
        let llc_groups: Vec<LlcGroup> = topo
            .llcs()
            .iter()
            .map(|llc| LlcGroup {
                cpus: llc.cpus().to_vec(),
            })
            .collect();
        validate_llc_partition(&llc_groups, &online_cpus)
            .context("host LLC discovery did not produce an exact CPU partition")?;
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
        let topology = Self::new_for_tests_unchecked(groups);
        validate_llc_partition(&topology.llc_groups, &topology.online_cpus)
            .expect("synthetic host LLC groups must exactly partition online CPUs");
        topology
    }

    /// Build an intentionally malformed synthetic topology for focused tests
    /// of discovery validation and CPU-union behavior. Topology-planner tests
    /// use [`Self::new_for_tests`], whose disjoint-partition invariant matches
    /// sysfs discovery.
    #[cfg(test)]
    pub(crate) fn new_for_tests_unchecked(groups: &[(Vec<usize>, usize)]) -> Self {
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
    #[cfg(test)]
    pub fn total_cpus(&self) -> usize {
        self.online_cpus.len()
    }

    // ------------------------------------------------------------------
    // Shared NUMA-placement primitives
    // ------------------------------------------------------------------
    //
    // Used by the bounded topology planner and `--cpu-cap` consolidation.

    /// Memoized map of NUMA node → LLC indices on that node. Returned
    /// by reference so callers can iterate without cloning; `BTreeMap`
    /// gives deterministic iteration so two invocations on identical
    /// topologies produce identical walks.
    ///
    /// Test fixtures use this to assert the same memoized map production
    /// placement traverses.
    #[cfg(test)]
    pub(crate) fn host_llcs_by_numa_node(&self) -> &std::collections::BTreeMap<usize, Vec<usize>> {
        &self.host_node_llcs
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
    /// Bulk placement traverses the memoized [`host_node_llcs`](Self::host_node_llcs)
    /// map; this one-off query is used where a selected LLC's node is needed.
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

    /// Enumerate performance placements against the synthetic topology's CPU
    /// set. Production uses [`Self::performance_pinning_candidates_for_cpus`]
    /// with the process affinity set; this wrapper keeps topology-only tests
    /// independent of the machine running them.
    #[cfg(test)]
    pub(crate) fn performance_pinning_candidates(
        &self,
        topo: &super::topology::Topology,
    ) -> Result<Vec<PerformancePinningCandidate>> {
        self.performance_pinning_candidates_for_cpus(topo, &self.online_cpus)
    }

    /// Enumerate performance placements using only CPUs allowed by the
    /// caller's affinity/cgroup. Ownership-free build mapping and run-time
    /// admission both call this exact entry point.
    pub(crate) fn performance_pinning_candidates_for_cpus(
        &self,
        topo: &super::topology::Topology,
        allowed_cpus: &[usize],
    ) -> Result<Vec<PerformancePinningCandidate>> {
        self.topology_pinning_candidates(topo, PinningKind::Performance, allowed_cpus, None)
    }

    /// Enumerate default 1:1 placements using the same mapper as performance
    /// mode. The only policy differences are no service CPU and SH LLC locks.
    pub(crate) fn default_pinning_candidates_for_cpus(
        &self,
        topo: &super::topology::Topology,
        allowed_cpus: &[usize],
    ) -> Result<Vec<PerformancePinningCandidate>> {
        self.topology_pinning_candidates(topo, PinningKind::Default, allowed_cpus, None)
    }

    /// Bounded availability-aware topology planner.
    ///
    /// The old enumerator multiplied several unrelated rotation periods with
    /// LCM and then nested an open-ended intra-LLC window loop. Coprime host
    /// shapes could therefore perform enormous work before deduplication. This
    /// planner evaluates at most one seed per actual allowed CPU (or host LLC
    /// when larger), and canonical claim deduplication bounds its output by
    /// that same physical-resource count.
    ///
    /// Each guest LLC contributes its real demand (`llc_cores[i] * threads`).
    /// Largest demands claim distinct fitting host LLC bins first. When both
    /// sides expose usable NUMA information, a bipartite match maps each
    /// CPU-bearing guest node to a distinct host node and bins are selected
    /// within that node. If no complete strict match exists, the planner
    /// deliberately falls back once to a global distinct-LLC match: running
    /// with correct CPU/LLC identity is preferable to rejecting a host solely
    /// because its NUMA/cache geometry cannot mirror the guest.
    ///
    /// `preferences` is a non-authoritative, mode-aware availability snapshot.
    /// Exact placement prefers its CPUs and LLC modes but may use the
    /// remainder; final admission still validates the complete claim under the
    /// registry fence. Supplying fragmented preferences therefore yields one
    /// exact globally-distinct plan rather than requiring a contiguous free
    /// window.
    pub(crate) fn topology_pinning_candidates(
        &self,
        topo: &super::topology::Topology,
        kind: PinningKind,
        allowed_cpus: &[usize],
        preferences: Option<PinningPreferences<'_>>,
    ) -> Result<Vec<PerformancePinningCandidate>> {
        use std::collections::BTreeSet;

        let allowed = allowed_cpus.iter().copied().collect::<BTreeSet<_>>();
        let total_needed =
            topo.total_cpus() as usize + usize::from(kind == PinningKind::Performance);
        if total_needed > allowed.len() {
            return Err(anyhow::Error::new(TopologyInsufficient {
                reason: format!(
                    "{}: need {total_needed} distinct host CPUs but only {} are allowed",
                    if kind == PinningKind::Performance {
                        "performance_mode"
                    } else {
                        "default pinning"
                    },
                    allowed.len(),
                ),
            }));
        }
        if topo.llcs as usize > self.llc_groups.len() {
            return Err(anyhow::Error::new(TopologyInsufficient {
                reason: format!(
                    "need {} distinct host LLC groups for {} guest LLCs, but host has {}",
                    topo.llcs,
                    topo.llcs,
                    self.llc_groups.len(),
                ),
            }));
        }

        let mut vcpu_start = 0u32;
        let demands = (0..topo.llcs)
            .map(|guest_llc| {
                let cpus = topo
                    .cores_in_llc(guest_llc)
                    .saturating_mul(topo.threads_per_core) as usize;
                let demand = GuestLlcDemand {
                    guest_llc,
                    guest_node: topo.numa_node_of(guest_llc),
                    vcpu_start,
                    cpus,
                };
                vcpu_start = vcpu_start.saturating_add(cpus as u32);
                demand
            })
            .collect::<Vec<_>>();
        if demands.iter().any(|demand| demand.cpus == 0) {
            return Err(anyhow::Error::new(TopologyInsufficient {
                reason: "guest topology contains an empty LLC CPU demand".into(),
            }));
        }
        let performance_prefers_shared =
            kind == PinningKind::Performance && grain_mapping_possible(self, &demands, &allowed);

        let all_llcs = (0..self.llc_groups.len()).collect::<BTreeSet<_>>();
        let (preferred_cpus, preferred_shared_llcs, preferred_exclusive_llcs) = match preferences {
            Some(preferences) => (
                preferences.cpus,
                preferences.shared_llcs,
                preferences.exclusive_llcs,
            ),
            None => (&allowed, &all_llcs, &all_llcs),
        };
        let allowed_rank = allowed
            .iter()
            .copied()
            .enumerate()
            .map(|(rank, cpu)| (cpu, rank))
            .collect::<std::collections::BTreeMap<_, _>>();
        let inputs = PlannerInputs {
            demands: &demands,
            kind,
            allowed: &allowed,
            allowed_rank: &allowed_rank,
            preferred_cpus,
            preferred_shared_llcs,
            preferred_exclusive_llcs,
            performance_prefers_shared,
        };
        let seeds = planner_seed_schedule(self, &demands, kind, &allowed);
        let mut seen = BTreeSet::new();
        let mut candidates = Vec::new();
        for (seed, local_seed) in seeds {
            let Some((plan, mut exact_cpus)) = self.plan_topology_seed(inputs, seed, local_seed)
            else {
                continue;
            };
            let llc_mode = match kind {
                PinningKind::Default => LlcLockMode::Shared,
                PinningKind::Performance => perf_llc_lock_mode(self, &plan),
            };
            if kind == PinningKind::Performance && llc_mode == LlcLockMode::Exclusive {
                exact_cpus = plan
                    .llc_indices
                    .iter()
                    .flat_map(|&llc| self.llc_groups[llc].cpus.iter().copied())
                    .collect();
            }
            exact_cpus.sort_unstable();
            exact_cpus.dedup();
            let llc_tag = u8::from(llc_mode == LlcLockMode::Exclusive);
            let identity = (llc_tag, plan.llc_indices.clone(), exact_cpus.clone());
            if seen.insert(identity) {
                candidates.push(PerformancePinningCandidate {
                    plan,
                    llc_mode,
                    cpu_mode: FlockMode::Exclusive,
                    cpu_reservations: exact_cpus,
                });
            }
        }
        if candidates.is_empty() {
            return Err(anyhow::Error::new(TopologyInsufficient {
                reason: "no exact distinct-LLC/CPU host placement fits".into(),
            }));
        }

        // Claim identity, not seed traversal, defines uniqueness.
        candidates.sort_by_key(|candidate| {
            let availability_miss = preferences.is_some_and(|preferences| {
                let llcs_ready = match candidate.llc_mode {
                    LlcLockMode::Shared => candidate
                        .plan
                        .llc_indices
                        .iter()
                        .all(|llc| preferences.shared_llcs.contains(llc)),
                    LlcLockMode::Exclusive => candidate
                        .plan
                        .llc_indices
                        .iter()
                        .all(|llc| preferences.exclusive_llcs.contains(llc)),
                };
                let cpus_ready = candidate.llc_mode == LlcLockMode::Exclusive
                    || candidate
                        .cpu_reservations
                        .iter()
                        .all(|cpu| preferences.cpus.contains(cpu));
                !(llcs_ready && cpus_ready)
            });
            (
                availability_miss,
                u8::from(candidate.llc_mode == LlcLockMode::Exclusive),
                candidate.plan.llc_indices.clone(),
                candidate.cpu_reservations.clone(),
            )
        });
        Ok(candidates)
    }

    /// Materialize one deterministic tie rotation of the bounded planner.
    fn plan_topology_seed(
        &self,
        inputs: PlannerInputs<'_>,
        seed: usize,
        local_seed: usize,
    ) -> Option<(PinningPlan, Vec<usize>)> {
        use std::collections::{BTreeMap, BTreeSet};

        let PlannerInputs {
            demands,
            kind,
            allowed,
            allowed_rank,
            preferred_cpus,
            preferred_shared_llcs,
            preferred_exclusive_llcs,
            performance_prefers_shared: _,
        } = inputs;
        let mut guest_nodes = BTreeMap::<u32, Vec<usize>>::new();
        for (index, demand) in demands.iter().enumerate() {
            guest_nodes
                .entry(demand.guest_node)
                .or_default()
                .push(index);
        }
        let guest_node_rank = guest_nodes
            .keys()
            .copied()
            .enumerate()
            .map(|(rank, node)| (node, rank))
            .collect::<BTreeMap<_, _>>();
        let host_node_rank = self
            .host_node_llcs
            .keys()
            .copied()
            .enumerate()
            .map(|(rank, node)| (node, rank))
            .collect::<BTreeMap<_, _>>();

        // First attempt a strict guest-node -> distinct host-node bipartite
        // mapping. Feasibility is evaluated from the actual per-LLC CPU
        // capacities after the process's allowed mask is applied.
        let strict_nodes = if guest_nodes.len() > 1 && !self.cpu_to_node.is_empty() {
            let mut edges = BTreeMap::<u32, Vec<usize>>::new();
            let host_count = self.host_node_llcs.len().max(1);
            for (&guest_node, request_indices) in &guest_nodes {
                let mut hosts = self
                    .host_node_llcs
                    .iter()
                    .filter_map(|(&host_node, llcs)| {
                        host_bins_fit(self, demands, request_indices, llcs, allowed)
                            .then_some(host_node)
                    })
                    .collect::<Vec<_>>();
                hosts.sort_by_key(|host_node| {
                    (
                        !host_bins_preferred_fit(
                            self,
                            inputs,
                            request_indices,
                            &self.host_node_llcs[host_node],
                        ),
                        rotated_tie(host_node_rank[host_node], seed, host_count),
                    )
                });
                edges.insert(guest_node, hosts);
            }
            match_distinct_nodes(&edges)
        } else {
            None
        };

        let mut mapped = vec![usize::MAX; demands.len()];
        let mut used_llcs = BTreeSet::new();
        let strict_fit = strict_nodes.as_ref().is_some_and(|node_map| {
            let mut trial_mapped = mapped.clone();
            let mut trial_used = BTreeSet::new();
            let mut nodes = guest_nodes.keys().copied().collect::<Vec<_>>();
            nodes.sort_by_key(|guest_node| {
                let max_demand = guest_nodes
                    .get(guest_node)
                    .expect("guest node came from this map")
                    .iter()
                    .map(|&index| demands[index].cpus)
                    .max()
                    .unwrap_or(0);
                (
                    std::cmp::Reverse(max_demand),
                    rotated_tie(guest_node_rank[guest_node], seed, guest_nodes.len()),
                )
            });
            for guest_node in nodes {
                let host_node = node_map[&guest_node];
                let bins = &self.host_node_llcs[&host_node];
                if !assign_distinct_bins(
                    self,
                    inputs,
                    guest_nodes
                        .get(&guest_node)
                        .expect("guest node came from this map"),
                    bins,
                    seed,
                    &mut trial_used,
                    &mut trial_mapped,
                ) {
                    return false;
                }
            }
            mapped = trial_mapped;
            used_llcs = trial_used;
            true
        });

        if !strict_fit {
            // Documented global fallback: strict NUMA mirroring is optional,
            // distinct cache bins and exact CPU identity are not.
            let all_requests = (0..demands.len()).collect::<Vec<_>>();
            let all_bins = (0..self.llc_groups.len()).collect::<Vec<_>>();
            if !assign_distinct_bins(
                self,
                inputs,
                &all_requests,
                &all_bins,
                seed,
                &mut used_llcs,
                &mut mapped,
            ) {
                return None;
            }
        }

        // Each mapped LLC is distinct and host discovery validates LLC CPU
        // membership as a partition. Selecting the first demand-sized prefix
        // from each independently ordered bin is therefore globally distinct;
        // no recursive per-vCPU bipartite match is needed.
        let mut assignments = Vec::new();
        let mut occupied = BTreeSet::new();
        // Static candidates are capacity grains, not every overlapping cyclic
        // window. Advancing by the largest material footprint emits a
        // pairwise-disjoint family per LLC (for example twelve 3-CPU
        // vCPU+service grains on a 36-CPU domain). Availability-aware planning
        // can still synthesize an arbitrary fragmented footprint on demand.
        for (index, demand) in demands.iter().enumerate() {
            let llc = mapped[index];
            let mut cpus = self.llc_groups[llc]
                .cpus
                .iter()
                .copied()
                .filter(|cpu| allowed.contains(cpu))
                .collect::<Vec<_>>();
            cpus.sort_unstable();
            cpus.dedup();
            let local_rank = cpus
                .iter()
                .copied()
                .enumerate()
                .map(|(rank, cpu)| (cpu, rank))
                .collect::<BTreeMap<_, _>>();
            let cpu_count = cpus.len().max(1);
            cpus.sort_by_key(|cpu| {
                (
                    !preferred_cpus.contains(cpu),
                    rotated_tie(local_rank[cpu], local_seed, cpu_count),
                )
            });
            if cpus.len() < demand.cpus {
                return None;
            }
            for (local, cpu) in cpus.into_iter().take(demand.cpus).enumerate() {
                if !occupied.insert(cpu) {
                    // This is unreachable for a validated LLC partition and
                    // keeps intentionally malformed test fixtures from
                    // producing duplicate host CPU assignments.
                    return None;
                }
                assignments.push((demand.vcpu_start + local as u32, cpu));
            }
        }
        assignments.sort_unstable_by_key(|&(vcpu, _)| vcpu);

        let service_cpu = if kind == PinningKind::Performance {
            let mut mapped_choice_rank = BTreeMap::<usize, (usize, usize)>::new();
            for (mapped_rank, &llc) in mapped.iter().enumerate() {
                let mut cpus = self.llc_groups[llc]
                    .cpus
                    .iter()
                    .copied()
                    .filter(|cpu| allowed.contains(cpu))
                    .collect::<Vec<_>>();
                cpus.sort_unstable();
                cpus.dedup();
                let count = cpus.len().max(1);
                for (rank, cpu) in cpus.into_iter().enumerate() {
                    let rank = (mapped_rank, rotated_tie(rank, local_seed, count));
                    mapped_choice_rank
                        .entry(cpu)
                        .and_modify(|current| *current = (*current).min(rank))
                        .or_insert(rank);
                }
            }
            // Include every allowed unoccupied host CPU. Locality is a
            // preference, not an eligibility gate: one busy mapped-local
            // leftover must not hide a free service CPU in another LLC.
            let mut service_choices = self
                .llc_groups
                .iter()
                .flat_map(|group| group.cpus.iter().copied())
                .filter(|cpu| allowed.contains(cpu) && !occupied.contains(cpu))
                .collect::<Vec<_>>();
            service_choices.sort_unstable();
            service_choices.dedup();
            service_choices.sort_by_key(|cpu| {
                let service_llcs = self
                    .llc_groups
                    .iter()
                    .enumerate()
                    .filter(|(_, group)| group.cpus.contains(cpu))
                    .map(|(llc, _)| llc)
                    .collect::<Vec<_>>();
                let candidate_prefers_shared = mapped.iter().enumerate().all(|(index, &llc)| {
                    let footprint = demands[index].cpus + usize::from(service_llcs.contains(&llc));
                    perf_grain_capable_for_footprint(self, llc, footprint)
                }) && service_llcs
                    .iter()
                    .filter(|&&llc| !mapped.contains(&llc))
                    .all(|&llc| perf_grain_capable_for_footprint(self, llc, 1));
                let llc_ready = mapped
                    .iter()
                    .copied()
                    .chain(service_llcs.iter().copied())
                    .all(|llc| {
                        if candidate_prefers_shared {
                            preferred_shared_llcs.contains(&llc)
                        } else {
                            preferred_exclusive_llcs.contains(&llc)
                        }
                    });
                let preferred = preferred_cpus.contains(cpu);
                let mapped_local = mapped_choice_rank.contains_key(cpu);
                (
                    !llc_ready,
                    !preferred,
                    !candidate_prefers_shared,
                    !mapped_local,
                    mapped_choice_rank.get(cpu).copied().unwrap_or((
                        usize::MAX,
                        rotated_tie(allowed_rank[cpu], seed, allowed.len().max(1)),
                    )),
                )
            });
            let cpu = *service_choices.first()?;
            occupied.insert(cpu);
            Some(cpu)
        } else {
            None
        };

        let mut llc_indices = mapped;
        if let Some(service_cpu) = service_cpu {
            // Prefer an already-mapped cache domain. Discovery validates that
            // each CPU belongs to exactly one LLC, while the fallback keeps
            // this helper defensive for intentionally malformed test fixtures.
            let service_llc = llc_indices
                .iter()
                .copied()
                .find(|&llc| self.llc_groups[llc].cpus.contains(&service_cpu))
                .or_else(|| {
                    (0..self.llc_groups.len())
                        .filter(|&llc| self.llc_groups[llc].cpus.contains(&service_cpu))
                        .min_by_key(|&llc| rotated_tie(llc, seed, self.llc_groups.len().max(1)))
                })?;
            llc_indices.push(service_llc);
        }
        llc_indices.sort_unstable();
        llc_indices.dedup();
        let exact_cpus = occupied.into_iter().collect();
        Some((
            PinningPlan {
                assignments,
                service_cpu,
                llc_indices,
                locks: Vec::new(),
            },
            exact_cpus,
        ))
    }
}

fn rotated_tie(value: usize, seed: usize, cardinality: usize) -> usize {
    value
        .wrapping_add(cardinality)
        .wrapping_sub(seed % cardinality.max(1))
        % cardinality.max(1)
}

fn planner_seed_schedule(
    host: &HostTopology,
    demands: &[GuestLlcDemand],
    kind: PinningKind,
    allowed: &std::collections::BTreeSet<usize>,
) -> Vec<(usize, usize)> {
    let stride = planner_local_stride(demands, kind);
    let minimum_demand = demands.iter().map(|demand| demand.cpus).min().unwrap_or(1);
    let mut seeds = host
        .llc_groups
        .iter()
        .enumerate()
        .flat_map(|(llc, group)| {
            let capacity = group
                .cpus
                .iter()
                .filter(|cpu| allowed.contains(cpu))
                .count();
            let grains = if capacity >= minimum_demand {
                (capacity / stride).max(1)
            } else {
                0
            };
            (0..grains).map(move |grain| (llc, grain.saturating_mul(stride)))
        })
        .collect::<Vec<_>>();
    if seeds.is_empty() {
        seeds.push((0, 0));
    }
    seeds
}

fn planner_local_stride(demands: &[GuestLlcDemand], kind: PinningKind) -> usize {
    demands
        .iter()
        .map(|demand| demand.cpus)
        .max()
        .unwrap_or(1)
        // The service thread may land in any mapped LLC. Reserve its width in
        // every static grain so multi-LLC and heterogeneous topologies do not
        // generate overlapping starts on the eventual service domain.
        .saturating_add(usize::from(kind == PinningKind::Performance))
        .max(1)
}

fn llc_allowed_cpus(
    host: &HostTopology,
    llc: usize,
    allowed: &std::collections::BTreeSet<usize>,
) -> std::collections::BTreeSet<usize> {
    host.llc_groups[llc]
        .cpus
        .iter()
        .copied()
        .filter(|cpu| allowed.contains(cpu))
        .collect()
}

fn host_bins_fit(
    host: &HostTopology,
    demands: &[GuestLlcDemand],
    request_indices: &[usize],
    bins: &[usize],
    allowed: &std::collections::BTreeSet<usize>,
) -> bool {
    let mut request_sizes = request_indices
        .iter()
        .map(|&index| demands[index].cpus)
        .collect::<Vec<_>>();
    let mut capacities = bins
        .iter()
        .map(|&llc| llc_allowed_cpus(host, llc, allowed).len())
        .collect::<Vec<_>>();
    request_sizes.sort_unstable_by(|a, b| b.cmp(a));
    capacities.sort_unstable_by(|a, b| b.cmp(a));
    request_sizes.len() <= capacities.len()
        && request_sizes
            .iter()
            .zip(capacities)
            .all(|(demand, capacity)| *demand <= capacity)
}

fn perf_grain_capable_for_footprint(host: &HostTopology, llc: usize, footprint: usize) -> bool {
    let llc_cpus = host.llc_groups[llc].cpus.len();
    llc_cpus >= PERF_GRAIN_LLC_MIN_CPUS
        && footprint.saturating_mul(PERF_GRAIN_MAX_OCCUPANCY_DEN)
            < llc_cpus.saturating_mul(PERF_GRAIN_MAX_OCCUPANCY_NUM)
}

fn grain_mapping_possible(
    host: &HostTopology,
    demands: &[GuestLlcDemand],
    allowed: &std::collections::BTreeSet<usize>,
) -> bool {
    let matches_with_service =
        |service_request: Option<usize>, separate_service_llc: Option<usize>| {
            let mut edges = std::collections::BTreeMap::new();
            for (request, demand) in demands.iter().enumerate() {
                let footprint = demand.cpus + usize::from(service_request == Some(request));
                let choices = (0..host.llc_groups.len())
                    .filter(|&llc| {
                        Some(llc) != separate_service_llc
                            && llc_allowed_cpus(host, llc, allowed).len() >= footprint
                            && perf_grain_capable_for_footprint(host, llc, footprint)
                    })
                    .collect::<Vec<_>>();
                if choices.is_empty() {
                    return false;
                }
                edges.insert(request, choices);
            }
            let mut order = (0..demands.len()).collect::<Vec<_>>();
            order.sort_by_key(|request| edges[request].len());
            match_distinct_bins(&edges, &order).is_some()
        };

    // The one service CPU may share any guest-mapped LLC...
    if (0..demands.len()).any(|request| matches_with_service(Some(request), None)) {
        return true;
    }
    // ...or occupy its own otherwise-unmapped grain-capable LLC.
    (0..host.llc_groups.len()).any(|llc| {
        !llc_allowed_cpus(host, llc, allowed).is_empty()
            && perf_grain_capable_for_footprint(host, llc, 1)
            && matches_with_service(None, Some(llc))
    })
}

fn preferred_llc_ready(
    host: &HostTopology,
    llc: usize,
    demand: usize,
    kind: PinningKind,
    preferred_shared_llcs: &std::collections::BTreeSet<usize>,
    preferred_exclusive_llcs: &std::collections::BTreeSet<usize>,
    performance_prefers_shared: bool,
) -> bool {
    match kind {
        PinningKind::Default => preferred_shared_llcs.contains(&llc),
        PinningKind::Performance
            if performance_prefers_shared
                && perf_grain_capable_for_footprint(host, llc, demand) =>
        {
            preferred_shared_llcs.contains(&llc)
        }
        PinningKind::Performance => preferred_exclusive_llcs.contains(&llc),
    }
}

/// Whether a host NUMA node can satisfy every request entirely from the
/// mode-aware resources in the coordinator's latest availability snapshot.
/// This is preference input only; the completed candidate's full claim is
/// checked authoritatively before acquisition.
fn host_bins_preferred_fit(
    host: &HostTopology,
    inputs: PlannerInputs<'_>,
    request_indices: &[usize],
    bins: &[usize],
) -> bool {
    let PlannerInputs {
        demands,
        kind,
        allowed,
        preferred_cpus,
        preferred_shared_llcs,
        preferred_exclusive_llcs,
        performance_prefers_shared,
        ..
    } = inputs;
    let mut edges = std::collections::BTreeMap::new();
    for &request in request_indices {
        let demand = demands[request].cpus;
        let choices = bins
            .iter()
            .copied()
            .filter(|&llc| {
                let allowed_cpus = llc_allowed_cpus(host, llc, allowed);
                allowed_cpus.len() >= demand
                    && allowed_cpus
                        .iter()
                        .filter(|cpu| preferred_cpus.contains(cpu))
                        .count()
                        >= demand
                    && preferred_llc_ready(
                        host,
                        llc,
                        demand,
                        kind,
                        preferred_shared_llcs,
                        preferred_exclusive_llcs,
                        performance_prefers_shared,
                    )
            })
            .collect::<Vec<_>>();
        if choices.is_empty() {
            return false;
        }
        edges.insert(request, choices);
    }
    let mut order = request_indices.to_vec();
    order.sort_by_key(|request| edges[request].len());
    match_distinct_bins(&edges, &order).is_some()
}

/// Maximum-cardinality bipartite matching for strict guest-NUMA placement.
fn match_distinct_nodes(
    edges: &std::collections::BTreeMap<u32, Vec<usize>>,
) -> Option<std::collections::BTreeMap<u32, usize>> {
    fn augment(
        guest: u32,
        edges: &std::collections::BTreeMap<u32, Vec<usize>>,
        owner: &mut std::collections::BTreeMap<usize, u32>,
        seen: &mut std::collections::BTreeSet<usize>,
    ) -> bool {
        for &host in edges.get(&guest).into_iter().flatten() {
            if !seen.insert(host) {
                continue;
            }
            let displaced = owner.get(&host).copied();
            if displaced.is_none_or(|other| augment(other, edges, owner, seen)) {
                owner.insert(host, guest);
                return true;
            }
        }
        false
    }

    let mut guests = edges.keys().copied().collect::<Vec<_>>();
    guests.sort_by_key(|guest| {
        edges
            .get(guest)
            .expect("guest came from this edge map")
            .len()
    });
    let mut owner = std::collections::BTreeMap::new();
    for guest in guests {
        if !augment(
            guest,
            edges,
            &mut owner,
            &mut std::collections::BTreeSet::new(),
        ) {
            return None;
        }
    }
    Some(
        owner
            .into_iter()
            .map(|(host, guest)| (guest, host))
            .collect(),
    )
}

fn assign_distinct_bins(
    host: &HostTopology,
    inputs: PlannerInputs<'_>,
    request_indices: &[usize],
    eligible_bins: &[usize],
    seed: usize,
    used: &mut std::collections::BTreeSet<usize>,
    mapped: &mut [usize],
) -> bool {
    let PlannerInputs {
        demands,
        kind,
        allowed,
        preferred_cpus,
        preferred_shared_llcs,
        preferred_exclusive_llcs,
        performance_prefers_shared,
        ..
    } = inputs;
    let mut eligible = eligible_bins
        .iter()
        .copied()
        .filter(|llc| !used.contains(llc))
        .collect::<Vec<_>>();
    eligible.sort_unstable();
    eligible.dedup();
    let bin_rank = eligible
        .iter()
        .copied()
        .enumerate()
        .map(|(rank, llc)| (llc, rank))
        .collect::<std::collections::BTreeMap<_, _>>();

    let mut edges = std::collections::BTreeMap::new();
    for &request in request_indices {
        let demand = demands[request].cpus;
        let mut choices = eligible
            .iter()
            .copied()
            .filter_map(|llc| {
                let cpus = llc_allowed_cpus(host, llc, allowed);
                (cpus.len() >= demand).then(|| {
                    let ready = cpus
                        .iter()
                        .filter(|cpu| preferred_cpus.contains(cpu))
                        .count();
                    let llc_ready = preferred_llc_ready(
                        host,
                        llc,
                        demand,
                        kind,
                        preferred_shared_llcs,
                        preferred_exclusive_llcs,
                        performance_prefers_shared,
                    );
                    (llc, cpus.len(), ready, llc_ready)
                })
            })
            .collect::<Vec<_>>();
        // A complete ready footprint outranks cyclic rotation. This lets the
        // coordinator synthesize an arbitrary simultaneously-free combination
        // (for example LLCs 0+2 of four) without pre-enumerating every subset.
        // When no availability snapshot was supplied, `preferred == allowed`,
        // so every fitting bin ties on the capped readiness terms and bounded
        // seed rotation still chooses alternate complete matchings. Subsequent
        // seeds can therefore recover when exact CPU matching rejects one
        // because synthetic LLC descriptions overlap.
        choices.sort_by_key(|&(llc, capacity, ready, llc_ready)| {
            let phase = bin_rank
                .get(&seed)
                .copied()
                .unwrap_or(seed % eligible.len().max(1));
            (
                !llc_ready,
                ready < demand,
                std::cmp::Reverse(ready.min(demand)),
                rotated_tie(bin_rank[&llc], phase, eligible.len().max(1)),
                capacity,
                std::cmp::Reverse(ready),
            )
        });
        if choices.is_empty() {
            return false;
        }
        edges.insert(
            request,
            choices
                .into_iter()
                .map(|(llc, _, _, _)| llc)
                .collect::<Vec<_>>(),
        );
    }

    let mut requests = request_indices.to_vec();
    requests.sort_by_key(|&request| {
        (
            edges[&request].len(),
            std::cmp::Reverse(demands[request].cpus),
            rotated_tie(
                demands[request].guest_llc as usize,
                seed,
                demands.len().max(1),
            ),
        )
    });
    let Some(assignment) = match_distinct_bins(&edges, &requests) else {
        return false;
    };
    for (&request, &llc) in &assignment {
        used.insert(llc);
        mapped[request] = llc;
    }
    true
}

/// Maximum-cardinality request-to-LLC matching. Unlike a greedy "first bin
/// that fits" walk, augmenting paths can displace an earlier flexible request
/// when a later request has only that bin available.
fn match_distinct_bins(
    edges: &std::collections::BTreeMap<usize, Vec<usize>>,
    request_order: &[usize],
) -> Option<std::collections::BTreeMap<usize, usize>> {
    fn augment(
        request: usize,
        edges: &std::collections::BTreeMap<usize, Vec<usize>>,
        owner: &mut std::collections::BTreeMap<usize, usize>,
        seen: &mut std::collections::BTreeSet<usize>,
    ) -> bool {
        for &bin in edges.get(&request).into_iter().flatten() {
            if !seen.insert(bin) {
                continue;
            }
            let displaced = owner.get(&bin).copied();
            if displaced.is_none_or(|other| augment(other, edges, owner, seen)) {
                owner.insert(bin, request);
                return true;
            }
        }
        false
    }

    let mut owner = std::collections::BTreeMap::new();
    for &request in request_order {
        if !augment(
            request,
            edges,
            &mut owner,
            &mut std::collections::BTreeSet::new(),
        ) {
            return None;
        }
    }
    Some(
        owner
            .into_iter()
            .map(|(bin, request)| (request, bin))
            .collect(),
    )
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
/// the pinned cores + service CPU. The returned explicit claim therefore
/// names the actual CPUs held, not the whole LLC, and peers see the freed
/// capacity.
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

/// Admission bridge for a default overcommit run. The run cannot promise
/// one-host-CPU-per-vCPU, but it still owns its complete allowed CPU footprint
/// exclusively and shares every LLC that footprint is known to touch. Whole
/// performance reservations also lock every CPU in their claimed
/// LLCs, so this CPU class remains sufficient when sysfs topology discovery
/// failed and `llc_indices` is empty.
pub(crate) fn acquire_overcommit_bridge(
    llc_indices: &[usize],
    cpus: &[usize],
    wait: bool,
    cancelled: Option<&AtomicBool>,
) -> Result<LockOutcome> {
    acquire_resource_locks_waiting_impl(
        llc_indices,
        LlcLockMode::Shared,
        cpus,
        FlockMode::Exclusive,
        wait,
        cancelled,
    )
}

fn acquire_resource_locks_waiting_impl(
    llc_indices: &[usize],
    llc_mode: LlcLockMode,
    cpus: &[usize],
    cpu_mode: FlockMode,
    wait: bool,
    cancelled: Option<&AtomicBool>,
) -> Result<LockOutcome> {
    if crate::cargo_test_mode::cargo_test_mode_active() {
        return Ok(LockOutcome::Acquired {
            llc_offset: llc_indices.first().copied().unwrap_or(0),
            locks: Vec::new(),
        });
    }
    let llc_offset = llc_indices.first().copied().unwrap_or(0);
    // Fast path: claim-subtracted, non-blocking, all-or-nothing.
    let (first_reason, first_evidence) =
        match try_acquire_resources(llc_indices, llc_mode, cpus, cpu_mode)? {
            TryAcquireAll::Acquired(locks) => {
                return Ok(LockOutcome::Acquired { llc_offset, locks });
            }
            TryAcquireAll::Contended { reason, evidence } => (reason, evidence),
        };
    if !wait {
        return Ok(LockOutcome::Unavailable(first_reason));
    }
    let target = protocol::canonical_lock_order_with_modes(
        llc_indices,
        llc_flock_mode(llc_mode),
        cpus,
        cpu_mode,
    );
    let claim = resource_claim_with_modes(llc_indices, llc_mode, cpus, cpu_mode);
    // Contended: retain one exact ticket while retrying this fixed set only
    // when it is compatible with every earlier live waiter claim. This keeps
    // disjoint host capacity busy without weakening the reservation: the
    // actual flock set is still all-or-nothing and byte-for-byte identical.
    let register = |probe: &mut protocol::GrantedProbe| {
        probe.try_acquire(&claim, || {
            match try_acquire_resources_unfenced(llc_indices, llc_mode, cpus, cpu_mode)? {
                TryAcquireAll::Acquired(locks) => Ok(protocol::ProbeOutcome::Acquired(locks)),
                TryAcquireAll::Contended {
                    evidence: Some(evidence),
                    ..
                } => Ok(protocol::ProbeOutcome::Contended(evidence)),
                TryAcquireAll::Contended { evidence: None, .. } => {
                    anyhow::bail!("unfenced resource probe lost exact contention evidence")
                }
            }
        })
    };
    let ticket = match first_evidence {
        Some(evidence) => protocol::register_ticket_after_contention(
            claim.clone(),
            claim.clone(),
            evidence,
            cancelled,
            register,
        )?,
        None => {
            protocol::register_ticket_or_acquire(claim.clone(), claim.clone(), cancelled, register)?
        }
    };
    let coordinator = match ticket {
        protocol::TicketWork::Acquired(locks) => {
            return Ok(LockOutcome::Acquired { llc_offset, locks });
        }
        protocol::TicketWork::Coordinator(coordinator) => coordinator,
    };
    let mut step = |held: &mut protocol::HeldLocks| {
        if let Some(locks) = held.probe_complete_if_ready(&claim, &target)? {
            Ok(protocol::CoordinatorStep::Complete {
                claim: claim.clone(),
                value: locks,
            })
        } else {
            Ok(protocol::CoordinatorStep::Waiting {
                claim: claim.clone(),
            })
        }
    };
    let outcome = match cancelled {
        Some(cancelled) => {
            protocol::acquire_as_coordinator_interruptible(coordinator, cancelled, &mut step)?
        }
        None => protocol::acquire_as_coordinator(coordinator, &mut step)?,
    };
    Ok(match outcome {
        protocol::CoordinatorOutcome::Acquired(locks) => {
            LockOutcome::Acquired { llc_offset, locks }
        }
        protocol::CoordinatorOutcome::Aborted { reason } => LockOutcome::Unavailable(reason),
    })
}

fn llc_flock_mode(llc_mode: LlcLockMode) -> FlockMode {
    match llc_mode {
        LlcLockMode::Exclusive => FlockMode::Exclusive,
        LlcLockMode::Shared => FlockMode::Shared,
    }
}

/// The published claim for an explicit two-resource-class reservation.
pub(crate) fn resource_claim_with_modes(
    llc_indices: &[usize],
    llc_mode: LlcLockMode,
    cpus: &[usize],
    cpu_mode: FlockMode,
) -> protocol::ClaimSet {
    protocol::ClaimSet::with_modes(
        llc_indices.iter().copied(),
        cpus.iter().copied(),
        llc_flock_mode(llc_mode),
        cpu_mode,
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
/// canonical order — see [`protocol`]). Returns the held fds
/// on success, or an error string describing which resource was
/// busy; on ANY failure every lock taken so far is released before
/// returning, so no partial reservation ever persists.
///
/// Claim-aware: an incompatible live ticket claim fails the attempt before
/// touching its reserved lockfiles.
pub(crate) enum TryAcquireAll {
    Acquired(Vec<std::os::fd::OwnedFd>),
    Contended {
        reason: String,
        evidence: Option<protocol::ContentionEvidence>,
    },
}

#[cfg(test)]
thread_local! {
    static AUTHORITATIVE_REGISTRY_PROBES: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn authoritative_registry_probe_count_for_tests() -> usize {
    AUTHORITATIVE_REGISTRY_PROBES.with(std::cell::Cell::get)
}

pub(crate) fn try_acquire_resources(
    llc_indices: &[usize],
    llc_mode: LlcLockMode,
    cpus: &[usize],
    cpu_mode: FlockMode,
) -> Result<TryAcquireAll> {
    #[cfg(test)]
    AUTHORITATIVE_REGISTRY_PROBES.with(|probes| probes.set(probes.get() + 1));
    let request = resource_claim_with_modes(llc_indices, llc_mode, cpus, cpu_mode);
    if request.is_empty() {
        return Ok(TryAcquireAll::Acquired(Vec::new()));
    }
    match protocol::with_registry_fence(&request, || {
        try_acquire_resources_unfenced(llc_indices, llc_mode, cpus, cpu_mode)
    })? {
        protocol::RegistryFence::Fenced => Ok(TryAcquireAll::Contended {
            reason: "reservation claimed by an earlier registered ticket".into(),
            evidence: None,
        }),
        protocol::RegistryFence::Ran {
            value: TryAcquireAll::Acquired(locks),
            watched,
        } => {
            if watched {
                protocol::publish_acquired(&request)?;
            }
            Ok(TryAcquireAll::Acquired(locks))
        }
        protocol::RegistryFence::Ran { value: outcome, .. } => Ok(outcome),
    }
}

fn try_acquire_resources_unfenced(
    llc_indices: &[usize],
    llc_mode: LlcLockMode,
    cpus: &[usize],
    cpu_mode: FlockMode,
) -> Result<TryAcquireAll> {
    let target = protocol::canonical_lock_order_with_modes(
        llc_indices,
        llc_flock_mode(llc_mode),
        cpus,
        cpu_mode,
    );
    let mut locks = Vec::with_capacity(target.len());
    for lock in &target {
        match try_flock_with_witness(&lock.path, lock.mode)? {
            TryFlockOutcome::Acquired(fd) => locks.push(fd),
            // Dropping `locks` on return releases everything taken
            // so far — the all-or-nothing contract.
            TryFlockOutcome::Contended(witness) => {
                drop(locks);
                return Ok(TryAcquireAll::Contended {
                    reason: format!("{} busy", lock.path),
                    evidence: Some(protocol::ContentionEvidence {
                        blocker: lock.resource,
                        mode: lock.mode,
                        _witness: witness,
                    }),
                });
            }
        }
    }
    Ok(TryAcquireAll::Acquired(locks))
}

pub(crate) fn acquire_resources_granted_with_evidence(
    llc_indices: &[usize],
    llc_mode: LlcLockMode,
    cpus: &[usize],
    cpu_mode: FlockMode,
) -> Result<protocol::ProbeOutcome<Vec<std::os::fd::OwnedFd>>> {
    Ok(
        match try_acquire_resources_unfenced(llc_indices, llc_mode, cpus, cpu_mode)? {
            TryAcquireAll::Acquired(locks) => protocol::ProbeOutcome::Acquired(locks),
            TryAcquireAll::Contended {
                evidence: Some(evidence),
                ..
            } => protocol::ProbeOutcome::Contended(evidence),
            TryAcquireAll::Contended { evidence: None, .. } => {
                anyhow::bail!("unfenced resource probe lost exact contention evidence")
            }
        },
    )
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
// Entry point [`acquire_llc_plan`] is the workload-time non-perf-mode
// reservation path: kernel-build workloads and no-perf VM runs both use it.
// The separate KtstrVm builder's no-perf sizing uses the ownership-free
// [`plan_llc_selection_only`] twin. `--cpu-cap` is a CPU-count budget:
// the planner reserves exactly N host CPUs by walking whole LLCs in
// contention- / NUMA-aware order and partial-taking the last LLC
// so `plan.cpus.len() == N`. Each plan holds `LOCK_SH` on the selected
// LLCs and on exactly those N CPUs. The last LLC is still covered as a
// whole cache domain while the exact CPU locks let perf-mode `LOCK_EX`
// reservations conflict with only the CPU prefix actually used. When `--cpu-cap`
// is absent the planner defaults to 30% of the calling process's
// sched_getaffinity cpuset (see [`default_cpu_budget`] and
// [`host_allowed_cpus`]) — not 30% of the host's online CPU count,
// because a CI runner whose parent cgroup pins ktstr to a 4-CPU
// subset must plan within THAT subset or sched_setaffinity on the
// resulting mask produces an empty effective set.
// Perf-mode never reaches this path; it uses topology-candidate admission
// with LLC and CPU modes recorded explicitly in each exact claim.
//
// The acquisition pipeline has three phases: discover (snapshot holders per
// LLC, filtered to the process's allowed cpuset), plan (NUMA-aware
// selection under the caller's [`PlacementPolicy`] — Consolidate for
// kernel-build workloads, Spread for no-perf VM runs), acquire (non-blocking `LOCK_SH`
// on each selected LLC and exact selected CPU). Up to
// ACQUIRE_MAX_TOCTOU_RETRIES retries
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
/// exactly-vcpus cpuset — the budgeted-path analog of the performance
/// planner's dedicated service CPU. Sensing timeliness rides on this: on
/// hosts without CAP_SYS_NICE the sensing threads run
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
/// The last LLC is flocked whole, while CPU `LOCK_SH` covers only the exact
/// prefix entering `plan.cpus`.
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
/// computed at `build()` without live holder state, a fan-out of
/// simultaneously-planning cells all see the same zero-occupancy shape and
/// the llc_idx-ASC tiebreak stacks every one of them onto the identical
/// LLC-0-upward prefix. Observed in the scx
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
    /// Processes currently holding this LLC's flock (any mode). Empty
    /// when no peer holds the lock. Derived from a single `/proc/locks`
    /// read shared across every LLC in the discover phase.
    pub(crate) holders: Vec<crate::flock::HolderInfo>,
    /// Count of holder PIDs other than this process, cached for PLAN scoring.
    /// Count-only discovery deliberately leaves `holders` empty, so this is
    /// not derived from `holders.len()`.
    pub(crate) holder_count: usize,
    /// Whether `/proc/locks` observed an incompatible `LOCK_EX` holder.
    /// Advisory planning excludes these LLCs when a complete ready
    /// alternative exists; the authoritative nonblocking flock still
    /// resolves races and hosts where procfs observation is unavailable.
    pub(crate) exclusive_held: bool,
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
/// `acquire_llc_plan` invocation at [`acquire_llc_plan_impl`]
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
    discover_llc_snapshots_impl(topo, allowed, mountinfo, true)
}

fn discover_llc_snapshot_counts(
    topo: &HostTopology,
    allowed: &std::collections::BTreeSet<usize>,
    mountinfo: &str,
) -> Result<Vec<LlcSnapshot>> {
    discover_llc_snapshots_impl(topo, allowed, mountinfo, false)
}

fn discover_llc_snapshots_impl(
    topo: &HostTopology,
    allowed: &std::collections::BTreeSet<usize>,
    mountinfo: &str,
    enrich_holders: bool,
) -> Result<Vec<LlcSnapshot>> {
    let mut paths = Vec::with_capacity(topo.llc_groups.len());
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
        // Existing files need only a metadata lookup. Opening every inode
        // O_RDWR during DISCOVER manufactures IN_CLOSE_WRITE traffic on the
        // coordinator's resource watch, so materialize only the genuinely
        // missing paths.
        match std::fs::metadata(&path) {
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                crate::flock::materialize(&path)?;
            }
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("inspect LLC reservation lock {}", path.display()));
            }
        }
        paths.push((llc_idx, path));
    }

    // Compatibility mode plus holder PID counts come from one /proc/locks
    // scan. The observation is advisory: on locked-down hosts, exact flock
    // acquisition remains authoritative and placement falls back to a
    // zero-holder view.
    let states = match crate::flock::read_flock_states_batch_with_mountinfo(
        paths.iter().map(|(_, path)| path.as_path()),
        mountinfo,
    ) {
        Ok(states) => states,
        Err(error) => {
            tracing::debug!(
                %error,
                "cannot observe LLC flock holders; planning with unknown holder state"
            );
            vec![crate::flock::FlockResourceState::default(); paths.len()]
        }
    };
    // Full HolderInfo/cmdline enrichment is reserved for explicit
    // diagnostics. Normal storm placement avoids one /proc/<pid>/cmdline read
    // per holder.
    let enriched = if enrich_holders {
        Some(
            match crate::flock::proc_locks::read_holders_batch_with_mountinfo(
                paths.iter().map(|(_, path)| path.as_path()),
                mountinfo,
            ) {
                Ok(holders) => holders,
                Err(error) => {
                    tracing::debug!(
                        %error,
                        "cannot attribute LLC flock holders; planning with empty holder snapshots"
                    );
                    (0..paths.len()).map(|_| Vec::new()).collect()
                }
            },
        )
    } else {
        None
    };
    if states.len() != paths.len()
        || enriched
            .as_ref()
            .is_some_and(|holders| holders.len() != paths.len())
    {
        anyhow::bail!(
            "batched LLC holder snapshot returned {} state rows for {} lockfiles",
            states.len(),
            paths.len()
        );
    }

    let mut snapshots = Vec::with_capacity(paths.len());
    for (row, (llc_idx, _)) in paths.into_iter().enumerate() {
        // Exclude the calling process from the PLAN-driving holder count.
        // A process may already own a related probe or another run plan;
        // treating that as peer load would make Spread flee its own claim.
        // The full `holders` vec is kept intact for diagnostics /
        // `ktstr locks`; only the sort key drops self.
        let holder_count = states[row]
            .holder_pids
            .iter()
            .filter(|pid| **pid != std::process::id())
            .count();
        let holders = enriched
            .as_ref()
            .map(|holders| holders[row].clone())
            .unwrap_or_default();
        snapshots.push(LlcSnapshot {
            llc_idx,
            holders,
            holder_count,
            exclusive_held: states[row].summary.exclusive_holder,
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
/// at [`materialize_llc_plan`] takes only the needed
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
            .copied()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
    };
    let total_allowed_in_llcs = snapshots
        .iter()
        .flat_map(|snapshot| topo.llc_groups[snapshot.llc_idx].cpus.iter().copied())
        .filter(|cpu| allowed.contains(cpu))
        .collect::<std::collections::BTreeSet<_>>()
        .len();
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
    let mut accumulated = std::collections::BTreeSet::new();
    for node in node_order {
        if accumulated.len() >= target_cpus {
            break;
        }
        // Ranked walk, taking every candidate on this node in
        // score-order until we've filled `target_cpus` or exhausted
        // the node.
        for snap in &ranked {
            if accumulated.len() >= target_cpus {
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
            accumulated.extend(
                topo.llc_groups[snap.llc_idx]
                    .cpus
                    .iter()
                    .copied()
                    .filter(|cpu| allowed.contains(cpu)),
            );
        }
    }

    // Step e: livelock-proof acquire order — ascending index.
    selected.sort_unstable();
    selected
}

/// Remove queue-only predecessor reservations when the remaining LLCs can
/// still carry the requested budget.
///
/// These claims have no `/proc/locks` row until their owner acquires a real
/// flock. Filtering them before policy scoring makes both Spread and
/// Consolidate prefer disjoint capacity. If the filtered capacity cannot
/// satisfy the exact budget, return the full snapshot so predecessor resources
/// remain eligible as the necessary last resort.
fn avoid_preceding_claims_when_possible(
    snapshots: &[LlcSnapshot],
    target_cpus: usize,
    topo: &HostTopology,
    allowed: &std::collections::BTreeSet<usize>,
    mut conflicts: impl FnMut(&protocol::ClaimSet) -> Result<bool>,
) -> Result<Vec<LlcSnapshot>> {
    let mut unreserved = Vec::with_capacity(snapshots.len());
    for snapshot in snapshots {
        let candidate =
            protocol::ClaimSet::new([snapshot.llc_idx], std::iter::empty(), FlockMode::Shared);
        if !conflicts(&candidate)? {
            unreserved.push(snapshot.clone());
        }
    }
    let available_cpus = unreserved
        .iter()
        .flat_map(|snapshot| topo.llc_groups[snapshot.llc_idx].cpus.iter().copied())
        .filter(|cpu| allowed.contains(cpu))
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    if available_cpus >= target_cpus {
        Ok(unreserved)
    } else {
        Ok(snapshots.to_vec())
    }
}

enum LlcLockAttempt {
    Acquired(Vec<std::os::fd::OwnedFd>),
    Contended(protocol::ContentionEvidence),
    #[cfg(test)]
    Unavailable,
}

trait IntoLlcLockAttempt {
    fn into_llc_lock_attempt(self) -> LlcLockAttempt;
}

impl IntoLlcLockAttempt for LlcLockAttempt {
    fn into_llc_lock_attempt(self) -> LlcLockAttempt {
        self
    }
}

#[cfg(test)]
impl IntoLlcLockAttempt for Option<Vec<std::os::fd::OwnedFd>> {
    fn into_llc_lock_attempt(self) -> LlcLockAttempt {
        self.map_or(LlcLockAttempt::Unavailable, LlcLockAttempt::Acquired)
    }
}

fn try_acquire_llc_plan_locks_with_evidence(
    selected: &[usize],
    cpus: &[usize],
    snapshots: &[LlcSnapshot],
) -> Result<LlcLockAttempt> {
    debug_assert!(
        selected
            .iter()
            .all(|idx| snapshots.iter().any(|snapshot| snapshot.llc_idx == *idx))
    );
    let target = protocol::canonical_lock_order_with_modes(
        selected,
        FlockMode::Shared,
        cpus,
        FlockMode::Shared,
    );
    let mut locks: Vec<std::os::fd::OwnedFd> = Vec::with_capacity(target.len());
    for lock in target {
        match try_flock_with_witness(&lock.path, lock.mode)? {
            TryFlockOutcome::Acquired(fd) => locks.push(fd),
            TryFlockOutcome::Contended(witness) => {
                // Drop previously-held fds so the peer racing us sees
                // a consistent post-bail state, then signal "retry".
                drop(locks);
                return Ok(LlcLockAttempt::Contended(protocol::ContentionEvidence {
                    blocker: lock.resource,
                    mode: lock.mode,
                    _witness: witness,
                }));
            }
        }
    }
    Ok(LlcLockAttempt::Acquired(locks))
}

/// Entry point for the `--cpu-cap` PLAN pipeline.
///
/// Runs DISCOVER → PLAN → ACQUIRE with up to
/// [`ACQUIRE_MAX_TOCTOU_RETRIES`] retries (each separated by a
/// per-retry sleep from [`TOCTOU_RETRY_DELAYS`]) as the acquisition
/// protocol's non-blocking FAST PHASE — claim-subtracted and
/// all-or-nothing (see [`protocol`]). `wait == false` (non-waiting kernel
/// builds, the interactive shell, and other non-blocking probes) bails with
/// `ResourceContention` the moment that
/// budget is spent. `wait == true` (the test run path) then joins the
/// cross-invocation registry and, as coordinator, RE-PLANS AGAINST LIVE HOLDER
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
/// exact ticket, while cancellation interrupts either its futex wait or the
/// coordinator's inotify sleep.
pub fn acquire_llc_plan_interruptible(
    topo: &HostTopology,
    test_topo: &crate::topology::TestTopology,
    cpu_cap: Option<CpuCap>,
    policy: PlacementPolicy,
    cancelled: &AtomicBool,
) -> Result<LlcPlan> {
    acquire_llc_plan_impl(topo, test_topo, cpu_cap, policy, true, Some(cancelled))
}

#[derive(Clone, Copy)]
struct LlcPlanAcquireRequest<'a> {
    topo: &'a HostTopology,
    test_topo: &'a crate::topology::TestTopology,
    cpu_cap: Option<CpuCap>,
    policy: PlacementPolicy,
    wait: bool,
    cancelled: Option<&'a AtomicBool>,
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
        // Bare `cargo test` mode: no peer-coordination contract. Both
        // build-time shape planning and run-time acquisition use this exact
        // helper so the stamped CPU budget and applied mask cannot diverge.
        let _ = test_topo;
        let _ = cpu_cap;
        let plan = cargo_test_mode_llc_plan(topo)?;
        check_acquire_cancelled(cancelled)?;
        return Ok(plan);
    }
    acquire_llc_plan_with_acquire_fn_impl(
        LlcPlanAcquireRequest {
            topo,
            test_topo,
            cpu_cap,
            policy,
            wait,
            cancelled,
        },
        try_acquire_llc_plan_locks_with_evidence,
    )
}

/// Ownership-free plan used by both build and run when
/// `KTSTR_CARGO_TEST_MODE` is active.
///
/// Cargo-test mode deliberately ignores CPU caps and names the complete
/// allowed cpuset, matching its "no host coordination" contract. Keeping this
/// in one helper prevents the build-side budget stamp from describing a
/// cap-truncated mask while the run side actually uses the full cpuset.
fn cargo_test_mode_llc_plan(topo: &HostTopology) -> Result<LlcPlan> {
    let allowed = host_allowed_cpus();
    if allowed.is_empty() {
        return Err(ResourceContention {
            reason: "could not determine allowed CPU set \
                     (sched_getaffinity and /proc/self/status both failed)"
                .into(),
        }
        .into());
    }
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
    let mems = plan_mems(&allowed, topo);
    Ok(LlcPlan {
        locked_llcs,
        cpus: allowed,
        mems,
        snapshot: Vec::new(),
        locks: Vec::new(),
    })
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
/// [`try_acquire_llc_plan_locks_with_evidence`] (non-blocking `LOCK_SH` per resource);
/// tests can pass a closure that returns `Ok(None)` on attempt 0 and
/// forwards on attempt 1 to simulate a peer winning the first race,
/// or an attempt-counting closure that always fails to exercise the
/// retry-exhausted error path.
///
/// `acquire_fn` receives `(selected, snapshots)` and returns
/// `Ok(Some(locks))` on success, `Ok(None)` to trigger a retry, or
/// propagates hard errors unchanged. Production closure is the
/// free-standing [`try_acquire_llc_plan_locks_with_evidence`]; the test closure
/// can track its own attempt counter via interior mutability
/// ([`std::cell::Cell`], `Mutex`, atomic int).
///
/// The FAST-PHASE loop body — DISCOVER, PLAN, retry budget, final
/// holder diagnostics — is shared between both entry points so the
/// test seam exercises the exact retry-and-diagnose sequence
/// production uses, not a parallel implementation. (`acquire_fn` is
/// a fast-phase seam only: the wait phase acquires through the
/// protocol coordinator engine against real lockfiles.)
///
/// `wait` separates the two contention regimes. No-wait callers retain the
/// short [`ACQUIRE_MAX_TOCTOU_RETRIES`] race budget. A waiting test run joins
/// the admission registry after its first failed all-or-nothing attempt;
/// repeating DISCOVER→PLAN against a published claim or real flock cannot beat
/// that holder and only multiplies procfs and registry traffic. Once elected
/// coordinator, the waiting path re-runs DISCOVER→PLAN against LIVE holder state
/// on every lock-dir wake — the re-plan-on-wake contract — and probes the
/// freshly selected LLCs and CPUs all-or-nothing. An incomplete probe releases
/// its N-1 available resources before sleeping on the blocker. Ordinary tickets
/// wait on targeted futexes; the coordinator waits on filtered inotify events.
#[cfg(test)]
fn acquire_llc_plan_with_acquire_fn<F>(
    topo: &HostTopology,
    test_topo: &crate::topology::TestTopology,
    cpu_cap: Option<CpuCap>,
    policy: PlacementPolicy,
    wait: bool,
    cancelled: Option<&AtomicBool>,
    acquire_fn: F,
) -> Result<LlcPlan>
where
    F: FnMut(&[usize], &[usize], &[LlcSnapshot]) -> Result<Option<Vec<std::os::fd::OwnedFd>>>,
{
    acquire_llc_plan_with_acquire_fn_impl(
        LlcPlanAcquireRequest {
            topo,
            test_topo,
            cpu_cap,
            policy,
            wait,
            cancelled,
        },
        acquire_fn,
    )
}

fn acquire_llc_plan_with_acquire_fn_impl<F, A>(
    request: LlcPlanAcquireRequest<'_>,
    mut acquire_fn: F,
) -> Result<LlcPlan>
where
    F: FnMut(&[usize], &[usize], &[LlcSnapshot]) -> Result<A>,
    A: IntoLlcLockAttempt,
{
    let LlcPlanAcquireRequest {
        topo,
        test_topo,
        cpu_cap,
        policy,
        wait,
        cancelled,
    } = request;
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
    // claim-fenced. Bounded attempts, then either bail (no-wait callers) or
    // join the registry.
    let mut attempt: u32 = 0;
    let mut queue_seed_snapshots = None;
    let mut queue_seed_cpu_states = None;
    let mut queue_seed_universe = None;
    let mut contention = protocol::ContentionSet::default();
    loop {
        check_acquire_cancelled(cancelled)?;
        let snapshots = discover_llc_snapshot_counts(topo, &allowed, &mountinfo).map_err(|e| {
            ResourceContention {
                reason: format!("discover LLC snapshots: {e}"),
            }
        })?;
        let cpu_states = discover_cpu_placement_states(&allowed, &mountinfo)?;
        // Subtract resources fenced by incompatible exact ticket claims. LLC
        // and CPU SH requests coexist with earlier SH reservations; only EX
        // claims are removed.
        let required = protocol::ClaimSet::with_modes(
            snapshots.iter().map(|snapshot| snapshot.llc_idx),
            allowed.iter().copied(),
            FlockMode::Shared,
            FlockMode::Shared,
        );
        let aggregate = protocol::registered_claim_snapshot(&required)?;
        let mut eligible: Vec<LlcSnapshot> = Vec::with_capacity(snapshots.len());
        let real_llc_filtered = snapshots.iter().any(|snapshot| snapshot.exclusive_held);
        let mut registry_llc_filtered = false;
        for snapshot in &snapshots {
            if snapshot.exclusive_held {
                continue;
            }
            let candidate =
                protocol::ClaimSet::new([snapshot.llc_idx], std::iter::empty(), FlockMode::Shared);
            if aggregate.conflicts(&candidate)? {
                registry_llc_filtered = true;
            } else {
                eligible.push(snapshot.clone());
            }
        }
        let contended_cpus = contention
            .marker_vec()
            .into_iter()
            .filter_map(|marker| match marker.blocker {
                protocol::ResourceKey::Cpu(cpu) => Some(cpu),
                protocol::ResourceKey::Llc(_) => None,
            })
            .collect::<std::collections::BTreeSet<_>>();
        let real_cpu_filtered = allowed.iter().any(|cpu| {
            contended_cpus.contains(cpu)
                || cpu_states
                    .get(cpu)
                    .is_some_and(|state| state.exclusive_held)
        });
        let mut registry_cpu_filtered = false;
        let eligible_allowed = cpu_eligible_allowed(&allowed, &cpu_states, |cpu| {
            if contended_cpus.contains(&cpu) {
                return Ok(true);
            }
            let conflicts = aggregate.conflicts(&protocol::ClaimSet::with_modes(
                std::iter::empty(),
                [cpu],
                FlockMode::Shared,
                FlockMode::Shared,
            ))?;
            registry_cpu_filtered |= conflicts;
            Ok(conflicts)
        })?;
        let registry_claim_filtered = registry_llc_filtered || registry_cpu_filtered;
        let real_holder_filtered = real_llc_filtered || real_cpu_filtered;
        let dynamic_filtered = registry_claim_filtered || real_holder_filtered;
        let eligible_capacity = eligible
            .iter()
            .flat_map(|snapshot| topo.llc_groups[snapshot.llc_idx].cpus.iter().copied())
            .filter(|cpu| eligible_allowed.contains(cpu))
            .collect::<std::collections::BTreeSet<_>>()
            .len();
        let claim_capacity_insufficient = eligible_capacity < target_cpus;
        let selected = if !claim_capacity_insufficient {
            plan_from_snapshots(
                &eligible,
                target_cpus,
                topo,
                &eligible_allowed,
                |from, to| test_topo.numa_distance(from, to),
                policy,
            )
        } else {
            Vec::new()
        };
        if selected.is_empty() && !dynamic_filtered {
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
        let selected_materialized = if selected.is_empty() {
            None
        } else {
            materialize_plan_cpus(
                &selected,
                topo,
                &eligible_allowed,
                &cpu_states,
                target_cpus,
                policy,
            )
        };
        if !selected.is_empty() && selected_materialized.is_none() {
            return Err(ResourceContention {
                reason: format!(
                    "selected LLCs did not contain the required {target_cpus} distinct eligible CPUs"
                ),
            }
            .into());
        }
        let (selected_cpus, selected_mems) = selected_materialized.unwrap_or_default();
        let mut registry_fenced = false;
        let acquired = if selected.is_empty() {
            // Claim-blocked: bounce without touching any lockfile.
            None
        } else {
            let exact = protocol::ClaimSet::with_modes(
                selected.iter().copied(),
                selected_cpus.iter().copied(),
                FlockMode::Shared,
                FlockMode::Shared,
            );
            match protocol::with_registry_fence(&exact, || {
                acquire_fn(&selected, &selected_cpus, &snapshots)
            })
            .map_err(|e| ResourceContention {
                reason: format!("acquire LLC and CPU locks: {e}"),
            })? {
                protocol::RegistryFence::Fenced => {
                    registry_fenced = true;
                    None
                }
                protocol::RegistryFence::Ran {
                    value: outcome,
                    watched,
                } => match outcome.into_llc_lock_attempt() {
                    LlcLockAttempt::Acquired(locks) => {
                        if watched {
                            protocol::publish_acquired(&exact).map_err(|error| {
                                ResourceContention {
                                    reason: format!(
                                        "publish acquired LLC reservation state: {error}"
                                    ),
                                }
                            })?;
                        }
                        Some(locks)
                    }
                    LlcLockAttempt::Contended(evidence) => {
                        contention.insert(evidence);
                        None
                    }
                    #[cfg(test)]
                    LlcLockAttempt::Unavailable => None,
                },
            }
        };
        if let Some(locks) = acquired {
            let plan =
                materialize_llc_plan(selected, snapshots, locks, selected_cpus, selected_mems);
            return Ok(plan);
        }
        if !wait
            && (registry_fenced
                || (claim_capacity_insufficient
                    && registry_claim_filtered
                    && !real_holder_filtered))
        {
            return Err(anyhow::Error::new(ResourceContention {
                reason: format!(
                    "registered reservation claims currently cover capacity needed for a {target_cpus}-CPU LLC plan"
                ),
            }));
        }
        if wait {
            // The registry is the retry mechanism for blocking callers. Join
            // immediately instead of multiplying an already-observed
            // contention event into four procfs scans and backoff sleeps.
            queue_seed_snapshots = Some(snapshots);
            queue_seed_cpu_states = Some(cpu_states);
            // The queued designation and immutable watch must cover a valid
            // full-budget candidate from the static topology. Live
            // exclusions are transient contention; seeding from them could
            // create an empty/short claim precisely when waiting is needed.
            queue_seed_universe = Some(allowed.clone());
            break;
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

    // ---- WAIT PHASE: register, then re-plan on every coordinator wake (see
    // `protocol`). Every target probe remains all-or-nothing: the exact claim
    // stays published while an incomplete attempt releases its real flocks.
    check_acquire_cancelled(cancelled)?;
    let queued_snapshots = queue_seed_snapshots
        .expect("waiting acquisition must preserve its failed fast-phase snapshot");
    let queued_cpu_states = queue_seed_cpu_states
        .expect("waiting acquisition must preserve its failed fast-phase CPU snapshot");
    let queued_allowed = queue_seed_universe
        .expect("waiting acquisition must preserve its full static CPU universe");
    let queued_selected = plan_from_snapshots(
        &queued_snapshots,
        target_cpus,
        topo,
        &queued_allowed,
        |from, to| test_topo.numa_distance(from, to),
        policy,
    );
    if queued_selected.is_empty() {
        return Err(ResourceContention {
            reason: format!(
                "no host LLC overlaps the process's {allowed_cpus}-CPU \
                 allowed set — sysfs LLC groups and sched_getaffinity disagree"
            ),
        }
        .into());
    }
    let Some((queued_cpus, _)) = materialize_plan_cpus(
        &queued_selected,
        topo,
        &queued_allowed,
        &queued_cpu_states,
        target_cpus,
        policy,
    ) else {
        return Err(ResourceContention {
            reason: format!(
                "host LLC topology contains fewer than the required {target_cpus} distinct allowed CPUs"
            ),
        }
        .into());
    };
    let queued_claim = protocol::ClaimSet::with_modes(
        queued_selected.iter().copied(),
        queued_cpus.iter().copied(),
        FlockMode::Shared,
        FlockMode::Shared,
    );
    // Watch the immutable static resource universe for releases, while
    // fencing successors with only the exact plan currently designated in
    // the registry. A transiently busy resource must remain in the watch or
    // its release cannot wake this ticket.
    let queued_watch = protocol::ClaimSet::with_modes(
        queued_snapshots.iter().map(|snapshot| snapshot.llc_idx),
        allowed.iter().copied(),
        FlockMode::Shared,
        FlockMode::Shared,
    );
    let register = |probe: &mut protocol::GrantedProbe| {
        let mut snapshots =
            discover_llc_snapshot_counts(topo, &allowed, &mountinfo).map_err(|e| {
                ResourceContention {
                    reason: format!("discover LLC snapshots while queued: {e}"),
                }
            })?;
        let static_capacity = snapshots
            .iter()
            .flat_map(|snapshot| topo.llc_groups[snapshot.llc_idx].cpus.iter().copied())
            .filter(|cpu| allowed.contains(cpu))
            .collect::<std::collections::BTreeSet<_>>()
            .len();
        if static_capacity < target_cpus {
            return Err(ResourceContention {
                reason: format!(
                    "host LLC topology exposes only {static_capacity} distinct CPUs from the process's allowed set, fewer than the required {target_cpus}"
                ),
            }
            .into());
        }
        let cpu_states = discover_cpu_placement_states(&allowed, &mountinfo)?;
        let designated = probe.designated().clone();
        let selected: Vec<usize> = designated.llcs.iter().copied().collect();
        let cpus: Vec<usize> = designated.cpus.iter().copied().collect();
        let designated_is_live = selected
            .iter()
            .all(|idx| snapshots.iter().any(|snapshot| snapshot.llc_idx == *idx));
        if designated_is_live
            && let Some(acquired) = probe.try_acquire(&designated, || {
                Ok(
                    match try_acquire_llc_plan_locks_with_evidence(&selected, &cpus, &snapshots)? {
                        LlcLockAttempt::Acquired(locks) => protocol::ProbeOutcome::Acquired((
                            selected.clone(),
                            snapshots.clone(),
                            locks,
                            cpus.clone(),
                        )),
                        LlcLockAttempt::Contended(evidence) => {
                            protocol::ProbeOutcome::Contended(evidence)
                        }
                        #[cfg(test)]
                        LlcLockAttempt::Unavailable => protocol::ProbeOutcome::Unavailable,
                    },
                )
            })?
        {
            return Ok(Some(acquired));
        }

        // `/proc/locks` shows only real flock holders. An earlier queue
        // ticket may reserve an LLC without holding that flock yet, so a
        // holder-only replan would repeatedly choose the same unavailable
        // LLC while disjoint capacity sits idle. Exclude conflicting prefix
        // claims before either policy scores the snapshot when the remaining
        // capacity can carry the full budget; otherwise retain them as the
        // necessary last resort.
        let eligible_allowed = cpu_eligible_allowed(&allowed, &cpu_states, |cpu| {
            probe.conflicts_with_predecessors(&protocol::ClaimSet::with_modes(
                std::iter::empty(),
                [cpu],
                FlockMode::Shared,
                FlockMode::Shared,
            ))
        })?;
        snapshots.retain(|snapshot| !snapshot.exclusive_held);
        snapshots = avoid_preceding_claims_when_possible(
            &snapshots,
            target_cpus,
            topo,
            &eligible_allowed,
            |candidate| probe.conflicts_with_predecessors(candidate),
        )?;
        let next_selected = plan_from_snapshots(
            &snapshots,
            target_cpus,
            topo,
            &eligible_allowed,
            |from, to| test_topo.numa_distance(from, to),
            policy,
        );
        if next_selected.is_empty() {
            // Every statically valid alternative is temporarily excluded by
            // a real EX holder or an earlier exact claim. Keep the designated
            // reservation; the contention evidence recorded by its probe
            // routes the relevant release serial back to this ticket.
            return Ok(None);
        }
        let Some((next_cpus, _)) = materialize_plan_cpus(
            &next_selected,
            topo,
            &eligible_allowed,
            &cpu_states,
            target_cpus,
            policy,
        ) else {
            // The ready subset lost capacity between planning dimensions.
            // This is transient contention, not a malformed static topology.
            return Ok(None);
        };
        let next_claim = protocol::ClaimSet::with_modes(
            next_selected.iter().copied(),
            next_cpus,
            FlockMode::Shared,
            FlockMode::Shared,
        );
        probe.reserve(&next_claim)?;
        Ok(None)
    };
    let coordinator_seed_claim = queued_claim.clone();
    let ticket = if contention.is_empty() {
        protocol::register_ticket_or_acquire(queued_claim, queued_watch, cancelled, register)?
    } else {
        protocol::register_ticket_after_contentions(
            queued_claim,
            queued_watch,
            contention,
            cancelled,
            register,
        )?
    };
    let coordinator = match ticket {
        protocol::TicketWork::Acquired((selected, snapshots, locks, cpus)) => {
            let mems = plan_mems(&cpus, topo);
            let plan = materialize_llc_plan(selected, snapshots, locks, cpus, mems);
            return Ok(plan);
        }
        protocol::TicketWork::Coordinator(coordinator) => coordinator,
    };
    let mut coordinator_claim = coordinator_seed_claim;
    let step = |held: &mut protocol::HeldLocks| {
        // RE-PLAN against live holder state on every wake — plans are
        // never cached across waits. The freed capacity may satisfy a
        // different selection than the one that was busy last wake.
        let snapshots = discover_llc_snapshot_counts(topo, &allowed, &mountinfo).map_err(|e| {
            ResourceContention {
                reason: format!("discover LLC snapshots: {e}"),
            }
        })?;
        let static_capacity = snapshots
            .iter()
            .flat_map(|snapshot| topo.llc_groups[snapshot.llc_idx].cpus.iter().copied())
            .filter(|cpu| allowed.contains(cpu))
            .collect::<std::collections::BTreeSet<_>>()
            .len();
        if static_capacity < target_cpus {
            return Ok(protocol::CoordinatorStep::Abort {
                reason: format!(
                    "host LLC topology exposes only {static_capacity} distinct CPUs from the process's allowed set, fewer than the required {target_cpus}"
                ),
            });
        }
        let cpu_states = discover_cpu_placement_states(&allowed, &mountinfo)?;
        let eligible_allowed = cpu_eligible_allowed(&allowed, &cpu_states, |_| Ok(false))?;
        let ready_snapshots: Vec<_> = snapshots
            .iter()
            .filter(|snapshot| !snapshot.exclusive_held)
            .cloned()
            .collect();
        let selected = plan_from_snapshots(
            &ready_snapshots,
            target_cpus,
            topo,
            &eligible_allowed,
            |from, to| test_topo.numa_distance(from, to),
            policy,
        );
        if let Some((cpus, _)) = materialize_plan_cpus(
            &selected,
            topo,
            &eligible_allowed,
            &cpu_states,
            target_cpus,
            policy,
        ) {
            coordinator_claim = protocol::ClaimSet::with_modes(
                selected,
                cpus,
                FlockMode::Shared,
                FlockMode::Shared,
            );
        }
        // If real EX holders leave no full ready alternative, retain the
        // previous exact designation and probe it. The resulting contention
        // evidence selects the blocker serial that wakes this coordinator;
        // temporary scarcity is never a terminal topology error.
        let selected: Vec<usize> = coordinator_claim.llcs.iter().copied().collect();
        let cpus: Vec<usize> = coordinator_claim.cpus.iter().copied().collect();
        debug_assert_eq!(cpus.len(), target_cpus);
        let mems = plan_mems(&cpus, topo);
        let target = protocol::canonical_lock_order_with_modes(
            &selected,
            FlockMode::Shared,
            &cpus,
            FlockMode::Shared,
        );
        if let Some(locks) = held.probe_complete_if_ready(&coordinator_claim, &target)? {
            Ok(protocol::CoordinatorStep::Complete {
                claim: protocol::ClaimSet::with_modes(
                    selected.iter().copied(),
                    cpus.iter().copied(),
                    FlockMode::Shared,
                    FlockMode::Shared,
                ),
                value: (selected, snapshots, locks, cpus, mems),
            })
        } else {
            Ok(protocol::CoordinatorStep::Waiting {
                claim: coordinator_claim.clone(),
            })
        }
    };
    let outcome = match cancelled {
        Some(flag) => protocol::acquire_as_coordinator_interruptible(coordinator, flag, step)?,
        None => protocol::acquire_as_coordinator(coordinator, step)?,
    };
    match outcome {
        protocol::CoordinatorOutcome::Acquired((selected, snapshots, locks, cpus, mems)) => {
            Ok(materialize_llc_plan(selected, snapshots, locks, cpus, mems))
        }
        protocol::CoordinatorOutcome::Aborted { reason } => {
            Err(anyhow::Error::new(ResourceContention { reason }))
        }
    }
}

/// Static, ownership-free shape planner for no-perf builds.
///
/// This path consults only the cached host topology, the calling process's
/// allowed cpuset, and the NUMA distance matrix. It performs no registry read,
/// lockfile materialization, `/proc/locks` scan, or flock attempt. Live holder
/// state is intentionally irrelevant: the resulting plan sizes diagnostics
/// and other build-time metadata, while the run-time replan performs
/// authoritative admission after immutable preparation.
///
/// The returned plan's `locks` is empty by construction.
pub fn plan_llc_selection_only(
    topo: &HostTopology,
    test_topo: &crate::topology::TestTopology,
    cpu_cap: Option<CpuCap>,
    policy: PlacementPolicy,
) -> Result<LlcPlan> {
    if crate::cargo_test_mode::cargo_test_mode_active() {
        let _ = test_topo;
        let _ = cpu_cap;
        return cargo_test_mode_llc_plan(topo);
    }
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
    let snapshots: Vec<_> = topo
        .llc_groups
        .iter()
        .enumerate()
        .filter(|(_, group)| group.cpus.iter().any(|cpu| allowed.contains(cpu)))
        .map(|(llc_idx, _)| LlcSnapshot {
            llc_idx,
            holders: Vec::new(),
            holder_count: 0,
            exclusive_held: false,
        })
        .collect();
    let cpu_states = allowed
        .iter()
        .copied()
        .map(|cpu| (cpu, CpuPlacementState::default()))
        .collect::<std::collections::BTreeMap<_, _>>();
    let selected = plan_from_snapshots(
        &snapshots,
        target_cpus,
        topo,
        &allowed,
        |from, to| test_topo.numa_distance(from, to),
        policy,
    );
    let materialized =
        materialize_plan_cpus(&selected, topo, &allowed, &cpu_states, target_cpus, policy);
    let Some((cpus, mems)) = materialized else {
        return Err(ResourceContention {
            reason: format!(
                "host LLC topology contains fewer than the required {target_cpus} distinct CPUs from the process's {}-CPU allowed set",
                allowed.len()
            ),
        }
        .into());
    };
    Ok(materialize_llc_plan(
        selected,
        snapshots,
        Vec::new(),
        cpus,
        mems,
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
/// CPUs. Shared by the fast-phase and coordinator-phase success paths so the
/// two cannot drift.
#[derive(Debug, Clone, Copy, Default)]
struct CpuPlacementState {
    exclusive_held: bool,
    other_holders: usize,
}

fn discover_cpu_placement_states(
    allowed: &std::collections::BTreeSet<usize>,
    mountinfo: &str,
) -> Result<std::collections::BTreeMap<usize, CpuPlacementState>> {
    let mut paths = Vec::with_capacity(allowed.len());
    for &cpu in allowed {
        let path = std::path::PathBuf::from(cpu_lock_path(cpu));
        // Existing files need only a metadata lookup. Opening every inode
        // O_RDWR manufactures IN_CLOSE_WRITE traffic for the coordinator's
        // resource watch and can wake the entire admission storm.
        match std::fs::metadata(&path) {
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                crate::flock::materialize(&path).with_context(|| {
                    format!("materialize CPU reservation lock {}", path.display())
                })?;
            }
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("inspect CPU reservation lock {}", path.display()));
            }
        }
        paths.push((cpu, path));
    }
    // Holder discovery is a placement hint, not the authority. Locked-down
    // hosts may deny /proc/locks or mountinfo-derived inode observation; the
    // exact flock/evidence path below still decides compatibility correctly.
    let states = match crate::flock::read_flock_states_batch_with_mountinfo(
        paths.iter().map(|(_, path)| path.as_path()),
        mountinfo,
    ) {
        Ok(states) => states,
        Err(error) => {
            tracing::debug!(
                %error,
                "cannot observe CPU reservation holders; planning with unknown holder state"
            );
            vec![crate::flock::FlockResourceState::default(); paths.len()]
        }
    };
    if states.len() != paths.len() {
        anyhow::bail!(
            "batched CPU holder snapshot returned {} rows for {} lockfiles",
            states.len(),
            paths.len(),
        );
    }
    let self_pid = std::process::id();
    Ok(paths
        .into_iter()
        .zip(states)
        .map(|((cpu, _), state)| {
            (
                cpu,
                CpuPlacementState {
                    exclusive_held: state.summary.exclusive_holder,
                    other_holders: state
                        .holder_pids
                        .iter()
                        .filter(|&&pid| pid != self_pid)
                        .count(),
                },
            )
        })
        .collect())
}

fn cpu_eligible_allowed(
    allowed: &std::collections::BTreeSet<usize>,
    states: &std::collections::BTreeMap<usize, CpuPlacementState>,
    mut conflicts: impl FnMut(usize) -> Result<bool>,
) -> Result<std::collections::BTreeSet<usize>> {
    let mut eligible = std::collections::BTreeSet::new();
    for &cpu in allowed {
        if states.get(&cpu).is_some_and(|state| state.exclusive_held) {
            continue;
        }
        if !conflicts(cpu)? {
            eligible.insert(cpu);
        }
    }
    Ok(eligible)
}

fn materialize_plan_cpus(
    selected: &[usize],
    topo: &HostTopology,
    eligible: &std::collections::BTreeSet<usize>,
    states: &std::collections::BTreeMap<usize, CpuPlacementState>,
    target_cpus: usize,
    policy: PlacementPolicy,
) -> Option<(Vec<usize>, std::collections::BTreeSet<usize>)> {
    let mut cpus: Vec<usize> = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    'outer: for &idx in selected {
        let group = &topo.llc_groups[idx];
        let mut candidates = group
            .cpus
            .iter()
            .copied()
            .filter(|cpu| eligible.contains(cpu))
            .collect::<Vec<_>>();
        candidates.sort_unstable();
        let rotation = match policy {
            PlacementPolicy::Consolidate => 0,
            PlacementPolicy::Spread { rotation } => rotation,
        };
        let count = candidates.len();
        let start = rotation % count.max(1);
        let rotated_pos: std::collections::HashMap<usize, usize> = candidates
            .iter()
            .enumerate()
            .map(|(position, cpu)| (*cpu, (position + count - start) % count.max(1)))
            .collect();
        candidates.sort_by(|a, b| {
            let a_holders = states.get(a).map_or(0, |state| state.other_holders);
            let b_holders = states.get(b).map_or(0, |state| state.other_holders);
            let occupancy = match policy {
                PlacementPolicy::Consolidate => b_holders.cmp(&a_holders),
                PlacementPolicy::Spread { .. } => a_holders.cmp(&b_holders),
            };
            occupancy.then_with(|| rotated_pos[a].cmp(&rotated_pos[b]).then(a.cmp(b)))
        });
        for cpu in candidates {
            if !seen.insert(cpu) {
                continue;
            }
            if cpus.len() >= target_cpus {
                break 'outer;
            }
            cpus.push(cpu);
        }
    }
    if cpus.len() != target_cpus {
        return None;
    }
    let mems = plan_mems(&cpus, topo);
    Some((cpus, mems))
}

fn plan_mems(cpus: &[usize], topo: &HostTopology) -> std::collections::BTreeSet<usize> {
    cpus.iter()
        .map(|cpu| topo.cpu_to_node.get(cpu).copied().unwrap_or(0))
        .collect()
}

fn materialize_llc_plan(
    selected: Vec<usize>,
    snapshots: Vec<LlcSnapshot>,
    locks: Vec<std::os::fd::OwnedFd>,
    cpus: Vec<usize>,
    mems: std::collections::BTreeSet<usize>,
) -> LlcPlan {
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
