//! NUMA memory-placement policy types for the workload pipeline.
//!
//! Holds [`MemPolicy`] (the `MPOL_*` enum the worker process
//! applies via `set_mempolicy(2)` after fork) and [`MpolFlags`]
//! (the `MPOL_F_*` flag bag OR'd into the mode argument). Validation
//! lives on [`MemPolicy::validate`] — each variant that takes a
//! node set rejects empty sets with an actionable diagnostic.

use std::collections::BTreeSet;

/// NUMA memory placement policy for worker processes.
///
/// Applied via `set_mempolicy(2)` after fork, before the work loop.
/// Maps to Linux `MPOL_*` constants. When `Default`, no syscall is
/// made (inherits the parent's policy).
///
/// Optional [`MpolFlags`] modify behavior (e.g. `STATIC_NODES` to
/// keep the nodemask absolute across cpuset changes).
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemPolicy {
    /// Inherit the parent process's memory policy (no syscall).
    #[default]
    Default,
    /// Allocate only from the specified NUMA nodes (`MPOL_BIND`).
    Bind(BTreeSet<usize>),
    /// Prefer allocations from the specified node, falling back to
    /// others when the preferred node is full (`MPOL_PREFERRED`).
    Preferred(usize),
    /// Interleave allocations round-robin across the specified nodes
    /// (`MPOL_INTERLEAVE`).
    Interleave(BTreeSet<usize>),
    /// Prefer the nearest node to the CPU where the allocation occurs
    /// (`MPOL_LOCAL`). No nodemask.
    Local,
    /// Prefer allocations from any of the specified nodes, falling back
    /// to others when all preferred nodes are full
    /// (`MPOL_PREFERRED_MANY`, kernel 5.15+).
    PreferredMany(BTreeSet<usize>),
    /// Weighted interleave across the specified nodes. Page distribution
    /// is proportional to per-node weights set via
    /// `/sys/kernel/mm/mempolicy/weighted_interleave/nodeN`
    /// (`MPOL_WEIGHTED_INTERLEAVE`, kernel 6.9+).
    WeightedInterleave(BTreeSet<usize>),
}

/// Optional mode flags for `set_mempolicy(2)`.
///
/// OR'd into the mode argument. See `MPOL_F_*` in
/// `include/uapi/linux/mempolicy.h`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub struct MpolFlags(u32);

impl MpolFlags {
    /// No flags.
    pub const NONE: Self = Self(0);
    /// `MPOL_F_STATIC_NODES` (1 << 15): nodemask is absolute, not
    /// remapped when the task's cpuset changes.
    pub const STATIC_NODES: Self = Self(1 << 15);
    /// `MPOL_F_RELATIVE_NODES` (1 << 14): nodemask is relative to
    /// the task's current cpuset.
    pub const RELATIVE_NODES: Self = Self(1 << 14);
    /// `MPOL_F_NUMA_BALANCING` (1 << 13): enable NUMA balancing
    /// optimization for this policy.
    pub const NUMA_BALANCING: Self = Self(1 << 13);

    /// Test-only raw-bit constructor. Lets unknown-bit guards
    /// (e.g. `validate_mempolicy_cpuset` in src/scenario/ops/mod.rs)
    /// be tested against bit patterns that are not reachable via
    /// the documented `STATIC_NODES | RELATIVE_NODES |
    /// NUMA_BALANCING` constants. Production callers must use the
    /// named constants + `union` / `BitOr` so the model stays in
    /// sync with the validator's known-bits mask.
    #[cfg(test)]
    pub(crate) const fn from_bits_for_test(bits: u32) -> Self {
        Self(bits)
    }

    /// Combine two flag sets.
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Raw flag bits for passing to the syscall.
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// Whether every bit in `other` is set in `self`.
    ///
    /// Set-theoretic, not syntactic: `contains(NONE)` returns `true`
    /// for any `self` (vacuous truth — the empty set is a subset of
    /// everything). Callers who want "has a non-empty intersection
    /// with `other`" must compare `self.bits() & other.bits() != 0`
    /// explicitly; using `contains` for that query silently returns
    /// `true` when the operand is `NONE` regardless of `self`.
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}

impl std::ops::BitOr for MpolFlags {
    type Output = Self;
    /// Delegates to [`MpolFlags::union`] so the bitwise-OR logic
    /// lives in one place. `union` is `const fn` (usable in
    /// const contexts like `const` initializers); `BitOr::bitor`
    /// cannot currently be `const` on stable, so keeping both
    /// entry points is necessary, but they must never diverge.
    fn bitor(self, rhs: Self) -> Self {
        self.union(rhs)
    }
}

impl MemPolicy {
    /// Construct a `Bind` policy from any iterator of NUMA node IDs.
    ///
    /// Accepts arrays, ranges, `Vec`, `BTreeSet`, or any `IntoIterator<Item = usize>`.
    pub fn bind(nodes: impl IntoIterator<Item = usize>) -> Self {
        MemPolicy::Bind(nodes.into_iter().collect())
    }

    /// Construct a `Preferred` policy for a single NUMA node.
    pub const fn preferred(node: usize) -> Self {
        MemPolicy::Preferred(node)
    }

    /// Construct an `Interleave` policy from any iterator of NUMA node IDs.
    ///
    /// Accepts arrays, ranges, `Vec`, `BTreeSet`, or any `IntoIterator<Item = usize>`.
    pub fn interleave(nodes: impl IntoIterator<Item = usize>) -> Self {
        MemPolicy::Interleave(nodes.into_iter().collect())
    }

    /// Construct a `PreferredMany` policy from any iterator of NUMA node IDs.
    pub fn preferred_many(nodes: impl IntoIterator<Item = usize>) -> Self {
        MemPolicy::PreferredMany(nodes.into_iter().collect())
    }

    /// Construct a `WeightedInterleave` policy from any iterator of NUMA node IDs.
    pub fn weighted_interleave(nodes: impl IntoIterator<Item = usize>) -> Self {
        MemPolicy::WeightedInterleave(nodes.into_iter().collect())
    }

    /// NUMA node IDs referenced by this policy.
    ///
    /// Returns the node set for `Bind`, `Interleave`, `PreferredMany`,
    /// and `WeightedInterleave`, a single-element set for `Preferred`,
    /// and an empty set for `Default`/`Local`.
    pub fn node_set(&self) -> BTreeSet<usize> {
        match self {
            MemPolicy::Default | MemPolicy::Local => BTreeSet::new(),
            MemPolicy::Bind(nodes)
            | MemPolicy::Interleave(nodes)
            | MemPolicy::PreferredMany(nodes)
            | MemPolicy::WeightedInterleave(nodes) => nodes.clone(),
            MemPolicy::Preferred(node) => [*node].into_iter().collect(),
        }
    }

    /// Validate that this policy's node set is non-empty where required.
    ///
    /// Returns `Err` with a description when a node-set-bearing policy
    /// has an empty set. Each diagnostic names the offending variant,
    /// the constraint ("at least one NUMA node"), and the actionable
    /// fix — both the matching constructor and the recommended
    /// fallbacks for "I don't want node-restricted placement" — so the
    /// operator sees the next mouse-click inline rather than having to
    /// `cargo doc --open` to discover the constructor menu. Mirrors the
    /// workers_pct rejection trailer at `WorkSpec::resolve_workers_pct`.
    pub fn validate(&self) -> std::result::Result<(), String> {
        match self {
            MemPolicy::Default | MemPolicy::Local => Ok(()),
            MemPolicy::Preferred(_) => Ok(()),
            MemPolicy::Bind(nodes) if nodes.is_empty() => Err(
                "Bind policy requires at least one NUMA node — \
                 use `MemPolicy::bind([node, ...])` with one or more node IDs, \
                 or `MemPolicy::Default` / `MemPolicy::Local` for non-bound placement"
                    .into(),
            ),
            MemPolicy::Interleave(nodes) if nodes.is_empty() => Err(
                "Interleave policy requires at least one NUMA node — \
                 use `MemPolicy::interleave([node, ...])` with one or more node IDs, \
                 or `MemPolicy::Default` for unrestricted placement"
                    .into(),
            ),
            MemPolicy::PreferredMany(nodes) if nodes.is_empty() => Err(
                "PreferredMany policy requires at least one NUMA node — \
                 use `MemPolicy::preferred_many([node, ...])` with one or more node IDs, \
                 or `MemPolicy::Preferred(node)` / `MemPolicy::Default` / `MemPolicy::Local`"
                    .into(),
            ),
            MemPolicy::WeightedInterleave(nodes) if nodes.is_empty() => Err(
                "WeightedInterleave policy requires at least one NUMA node — \
                 use `MemPolicy::weighted_interleave([node, ...])` with one or more node IDs, \
                 or `MemPolicy::interleave([node, ...])` / `MemPolicy::Default` for unweighted placement"
                    .into(),
            ),
            _ => Ok(()),
        }
    }
}
