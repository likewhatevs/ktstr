//! Top-level workload configuration: the [`WorkloadConfig`] struct,
//! its `Default`, and the chainable builder methods on
//! `impl WorkloadConfig`.
//!
//! WorkloadConfig is the primary group's shape — what each worker
//! does (`work_type`), how many (`num_workers`), what scheduler
//! policy / memory policy / nice / clone mode they get, and an
//! optional `composed` list of secondary [`WorkSpec`] groups that
//! spawn alongside.
//!
//! Validation lives on [`WorkloadConfig::validate`]: it gates
//! invariants that must hold BEFORE any worker context exists —
//! currently `mem_policy` empty-nodemask rejection on the primary
//! group plus every composed entry.

use std::borrow::Cow;

use super::super::{AffinityIntent, WorkType};
use super::{CloneMode, MemPolicy, MpolFlags, SchedPolicy, WorkSpec};

/// Configuration for spawning a group of worker processes.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
// See [`WorkType`]'s `#[serde(bound(...))]` comment — embedding
// `WorkType` propagates the same lifetime-bound issue, so we pass
// through the same explicit empty bound.
#[serde(bound(deserialize = ""))]
pub struct WorkloadConfig {
    /// Number of worker processes to fork.
    pub num_workers: usize,
    /// Per-worker affinity intent. Resolved at spawn time via the
    /// same gate as composed entries (see [`Self::composed`]):
    /// [`AffinityIntent::Inherit`] (resolved to
    /// `ResolvedAffinity::None`),
    /// [`AffinityIntent::Exact`] (resolved to
    /// `ResolvedAffinity::Fixed`), and
    /// [`AffinityIntent::RandomSubset`] (resolved to
    /// `ResolvedAffinity::Random` — sampling deferred per-worker
    /// at spawn time) are accepted at `WorkloadHandle::spawn`.
    /// Topology-aware variants (`SingleCpu`, `LlcAligned`,
    /// `CrossCgroup`, `SmtSiblingPair`) require scenario context
    /// and are rejected with an actionable diagnostic.
    /// Type-unified with [`WorkSpec::affinity`] so a test author
    /// writes the same affinity expression at the top level and
    /// inside `composed` entries.
    pub affinity: AffinityIntent,
    /// What each worker does.
    pub work_type: WorkType,
    /// Linux scheduling policy.
    pub sched_policy: SchedPolicy,
    /// NUMA memory placement policy.
    pub mem_policy: MemPolicy,
    /// Optional mode flags for `set_mempolicy(2)`.
    pub mpol_flags: MpolFlags,
    /// Per-worker nice value applied via `setpriority(2)` after
    /// fork, before the work loop. Range `-20..=19` per `MIN_NICE`
    /// / `MAX_NICE` in `kernel/sys.c`'s `setpriority` syscall;
    /// values outside this window are clamped kernel-side. `None`
    /// (the default) skips the syscall entirely so the worker
    /// inherits the parent's nice value; `Some(n)` invokes
    /// `setpriority(PRIO_PROCESS, 0, n)` unconditionally — a user
    /// who wants the worker to land on nice 0 regardless of the
    /// parent's nice (or a cgroup-level default stored at
    /// [`CgroupDef::default_nice`](crate::scenario::ops::CgroupDef::default_nice))
    /// writes `Some(0)`, distinct from `None`.
    ///
    /// Values below the calling task's current nice require
    /// `CAP_SYS_NICE` (the kernel's `can_nice` check fires on
    /// `niceval < task_nice(p)`, not only on negatives — the
    /// `set_one_prio` gate at `kernel/sys.c` returns `EACCES` to
    /// unprivileged callers when `is_nice_reduction` rejects the
    /// requested value). With `Some(0)` on a parent at `nice=5`,
    /// `setpriority` returns `EACCES` without the capability.
    /// `None` (inherit) is always safe. Failures are logged once
    /// via stderr and do not abort the worker — the
    /// scheduling-policy and affinity sites use the same idiom.
    pub nice: Option<i32>,
    /// How to create each worker. Defaults to [`CloneMode::Fork`].
    pub clone_mode: CloneMode,
    /// Worker process name set via `prctl(PR_SET_NAME)` after fork.
    /// Kernel truncates to 15 bytes (TASK_COMM_LEN - 1). `None`
    /// inherits the binary name. Mirrors [`WorkSpec::comm`] so the
    /// primary group exposes the same scheduler-matcher knob composed
    /// entries already do.
    pub comm: Option<Cow<'static, str>>,
    /// Effective UID set via `setresuid(uid, uid, uid)` after fork.
    /// `None` inherits the parent's euid. Mirrors [`WorkSpec::uid`].
    pub uid: Option<u32>,
    /// Effective GID set via `setresgid(gid, gid, gid)` after fork.
    /// `None` inherits the parent's egid. Mirrors [`WorkSpec::gid`].
    pub gid: Option<u32>,
    /// Restrict worker affinity to the CPUs of this NUMA node.
    /// Applied via `sched_setaffinity` after fork. Mirrors
    /// [`WorkSpec::numa_node`].
    pub numa_node: Option<u32>,
    /// Secondary worker groups spawned alongside the primary group
    /// described by the top-level fields. Each entry is a
    /// [`WorkSpec`] with its own `work_type`, `num_workers`,
    /// `sched_policy`, `affinity`, etc. Composed groups are spawned
    /// in declaration order after the primary group; their workers
    /// run concurrently with the primary's for the lifetime of the
    /// `WorkloadHandle`. The default (an empty vec) skips the
    /// composed pass and behaves exactly as the pre-composition
    /// spawn.
    ///
    /// All groups share the same stop signal —
    /// `WorkloadHandle::stop_and_collect` terminates primary plus
    /// every composed group atomically. Per-group stop is not
    /// supported.
    ///
    /// Reports carry `WorkerReport::group_idx` = 0 for the primary
    /// group and 1..=N for composed entries in declaration order.
    ///
    /// # Worked example
    ///
    /// Build a multi-group workload — primary `SpinWait(2)` plus
    /// one `PipeIo(2)` composed group plus one `YieldHeavy(1)`
    /// composed group — using either the replacing
    /// [`composed`](Self::composed) setter or the appending
    /// [`push_composed`](Self::push_composed) chain:
    ///
    /// ```
    /// use ktstr::workload::{WorkSpec, WorkType, WorkloadConfig};
    ///
    /// // Append style: each call adds one group to the existing list.
    /// let cfg = WorkloadConfig::default()
    ///     .work_type(WorkType::SpinWait)
    ///     .workers(2)
    ///     .push_composed(
    ///         WorkSpec::default()
    ///             .work_type(WorkType::pipe_io(64))
    ///             .workers(2),
    ///     )
    ///     .push_composed(
    ///         WorkSpec::default()
    ///             .work_type(WorkType::YieldHeavy)
    ///             .workers(1),
    ///     );
    /// assert_eq!(cfg.composed.len(), 2);
    ///
    /// // Replace style: one call passes every composed group at once.
    /// let cfg2 = WorkloadConfig::default()
    ///     .work_type(WorkType::SpinWait)
    ///     .workers(2)
    ///     .composed([
    ///         WorkSpec::default().work_type(WorkType::pipe_io(64)).workers(2),
    ///         WorkSpec::default().work_type(WorkType::YieldHeavy).workers(1),
    ///     ]);
    /// assert_eq!(cfg2.composed.len(), 2);
    /// ```
    ///
    /// # Resolution rules at spawn time
    ///
    /// Composed [`WorkSpec`] entries must specify
    /// [`WorkSpec::num_workers`] (`Some(n)`); the `None` default
    /// resolved by the scenario engine via
    /// `Ctx::workers_per_cgroup` is unreachable from
    /// `WorkloadHandle::spawn` and is rejected with an actionable
    /// diagnostic.
    ///
    /// Composed [`WorkSpec::affinity`] accepts the no-context
    /// variants [`AffinityIntent::Inherit`] (resolved to
    /// `ResolvedAffinity::None`), [`AffinityIntent::Exact`]
    /// (resolved to `ResolvedAffinity::Fixed`), and
    /// [`AffinityIntent::RandomSubset`] (resolved to
    /// `ResolvedAffinity::Random` — sampling deferred per-worker
    /// at spawn time). The topology-aware variants (`SingleCpu`,
    /// `LlcAligned`, `CrossCgroup`, `SmtSiblingPair`) are rejected
    /// because spawn() has no access to the
    /// [`crate::topology::TestTopology`] / cpuset state that the
    /// scenario engine threads in.
    ///
    /// Composed entries inherit the parent
    /// [`WorkloadConfig::clone_mode`] — the dispatch path
    /// (fork vs thread) is a workload-wide property, so
    /// [`WorkSpec`] carries no `clone_mode` field of its own.
    ///
    /// Composition is single-level — a [`WorkSpec`] inside
    /// `composed` has no `composed` field of its own.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub composed: Vec<WorkSpec>,
}

impl Default for WorkloadConfig {
    fn default() -> Self {
        Self {
            num_workers: 1,
            affinity: AffinityIntent::Inherit,
            work_type: WorkType::SpinWait,
            sched_policy: SchedPolicy::Normal,
            mem_policy: MemPolicy::Default,
            mpol_flags: MpolFlags::NONE,
            nice: None,
            clone_mode: CloneMode::Fork,
            comm: None,
            uid: None,
            gid: None,
            numa_node: None,
            composed: Vec::new(),
        }
    }
}

impl WorkloadConfig {
    /// Validate the config before spawn. Fails loud on invariants
    /// that the worker-spawn path otherwise handles by silent
    /// degradation — in particular `mem_policy` variants that
    /// require a non-empty nodemask (Bind / Interleave / PreferredMany /
    /// WeightedInterleave with an empty BTreeSet).
    ///
    /// # Why a config-layer gate
    ///
    /// `apply_mempolicy_with_flags` (called from the worker's hot
    /// path in BOTH forked-child and thread-mode contexts) currently
    /// handles an empty node-set by logging to `stderr` and
    /// returning — the worker silently proceeds with default kernel
    /// placement instead of the requested NUMA binding. That
    /// silent-skip is a silent-drop bug (the test reports success
    /// while the actual workload ran with the wrong placement).
    ///
    /// A hypothetical fix-it-in-the-worker design — `libc::_exit(1)`
    /// on an empty node-set inside the worker — was rejected because
    /// it is unsound for thread-mode workers: `_exit` invokes
    /// `exit_group(2)` (verified at kernel/exit.c::do_group_exit →
    /// `zap_other_threads`) which terminates EVERY thread in the
    /// caller's tgid. A thread-mode worker shares its tgid with the
    /// test runner, so an inner `_exit(1)` would kill the runner.
    /// Rejecting at the config layer keeps the failure visible as a
    /// returnable `Result` BEFORE any worker context exists,
    /// regardless of clone-mode dispatch, and avoids the exit_group
    /// hazard entirely.
    ///
    /// # What is validated
    ///
    /// The primary group's `mem_policy` plus every composed
    /// [`WorkSpec`]'s `mem_policy`. Per-entry errors name the
    /// offending slot (`"primary"` or `"composed[N] (group_idx M)"`) so
    /// the test author can locate the misconfigured group.
    ///
    /// # Scope
    ///
    /// Currently validates only `mem_policy` on the primary group +
    /// each composed [`WorkSpec`]. Other field invariants are
    /// validated at their own use sites: `num_workers` via
    /// `WorkSpec::resolve_workers_pct` (and the spawn-time
    /// `WorkloadHandle::spawn` derivation cascade); [`WorkType`]
    /// payloads via per-variant constructors and
    /// `validate_workload_admission`; [`AffinityIntent`] topology
    /// rules at the scenario-engine
    /// `resolve_affinity_for_cgroup` resolver. This method is the
    /// home for invariants that must hold BEFORE any worker context
    /// (threads, forks, cgroups) exists — `mem_policy` qualifies
    /// because of the silent-skip + `exit_group` hazard noted
    /// above; future fields with the same "must-fail-before-spawn"
    /// shape belong here too.
    ///
    /// # Return type
    ///
    /// Returns [`anyhow::Result`] (composite-layer convention used
    /// by sibling composite validators
    /// `crate::test_support::entry::KtstrTestEntry::validate` and
    /// `crate::test_support::entry::TopologyConstraints::validate`
    /// — they wrap leaf validators that return
    /// `Result<(), String>` with slot-context). The leaf validator
    /// [`MemPolicy::validate`] returns `Result<(), String>` to match
    /// the leaf convention used by every per-spec validator in the
    /// project.
    pub fn validate(&self) -> anyhow::Result<()> {
        self.mem_policy
            .validate()
            .map_err(|e| anyhow::anyhow!("WorkloadConfig.mem_policy (primary group): {e}",))?;
        for (idx, spec) in self.composed.iter().enumerate() {
            spec.mem_policy.validate().map_err(|e| {
                anyhow::anyhow!(
                    "WorkloadConfig.composed[{idx}].mem_policy (group_idx {}): {e}",
                    idx + 1,
                )
            })?;
        }
        Ok(())
    }

    /// Set the number of worker processes.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn workers(mut self, n: usize) -> Self {
        self.num_workers = n;
        self
    }

    /// Set the per-worker affinity intent.
    ///
    /// At `WorkloadHandle::spawn`, [`AffinityIntent::Inherit`],
    /// [`AffinityIntent::Exact`], and [`AffinityIntent::RandomSubset`]
    /// are accepted; topology-aware variants (`SingleCpu`,
    /// `LlcAligned`, `CrossCgroup`, `SmtSiblingPair`) require
    /// scenario context and are rejected.
    ///
    /// Idiomatic short form for an exact CPU set:
    /// `cfg.affinity(AffinityIntent::exact([0, 1]))`.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn affinity(mut self, a: AffinityIntent) -> Self {
        self.affinity = a;
        self
    }

    /// Set the work type.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn work_type(mut self, wt: WorkType) -> Self {
        self.work_type = wt;
        self
    }

    /// Set the Linux scheduling policy.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn sched_policy(mut self, p: SchedPolicy) -> Self {
        self.sched_policy = p;
        self
    }

    /// Set the NUMA memory placement policy.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn mem_policy(mut self, p: MemPolicy) -> Self {
        self.mem_policy = p;
        self
    }

    /// Set the NUMA memory policy mode flags.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn mpol_flags(mut self, f: MpolFlags) -> Self {
        self.mpol_flags = f;
        self
    }

    /// Set the per-worker nice value applied via `setpriority(2)`.
    ///
    /// Stores `Some(n)` on the config; the spawn pipeline calls
    /// `setpriority(PRIO_PROCESS, 0, n)` unconditionally (including
    /// `n == 0`). The "skip the syscall, inherit the parent's nice"
    /// state is the type-level default `None` — set the field via
    /// `..Default::default()` (or leave the builder unchained) when
    /// you want inherit semantics. Values below the calling task's
    /// current nice require `CAP_SYS_NICE`; see
    /// [`WorkloadConfig::nice`] for the full `can_nice` rule.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn nice(mut self, n: i32) -> Self {
        self.nice = Some(n);
        self
    }

    /// Set the clone mode used when spawning each worker.
    ///
    /// [`CloneMode::Fork`] (the default) preserves historical
    /// behavior. See [`CloneMode`] for the full menu and dispatch
    /// status.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn clone_mode(mut self, m: CloneMode) -> Self {
        self.clone_mode = m;
        self
    }

    /// Set the worker process name via `prctl(PR_SET_NAME)`.
    /// Kernel truncates to 15 bytes.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn comm(mut self, name: impl Into<Cow<'static, str>>) -> Self {
        self.comm = Some(name.into());
        self
    }

    /// Set the worker's effective UID via `setresuid`.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn uid(mut self, uid: u32) -> Self {
        self.uid = Some(uid);
        self
    }

    /// Set the worker's effective GID via `setresgid`.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn gid(mut self, gid: u32) -> Self {
        self.gid = Some(gid);
        self
    }

    /// Restrict worker affinity to a NUMA node's CPU set.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn numa_node(mut self, node: u32) -> Self {
        self.numa_node = Some(node);
        self
    }

    /// Replace the composed worker groups (replacing setter).
    ///
    /// Pass an iterator of [`WorkSpec`] entries; the existing
    /// `composed` vec is REPLACED with the supplied entries. Each
    /// will be spawned as an independent group alongside the
    /// primary described by the top-level fields. Pass an empty
    /// iterator to clear any previously-set composed groups.
    ///
    /// Use this when you have all groups in hand at once. To add
    /// one group at a time to an existing list, use the appending
    /// [`push_composed`](Self::push_composed) instead.
    ///
    /// See [`Self::composed`] for the resolution rules applied to
    /// each entry's `num_workers` / `affinity` fields at spawn time.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn composed(mut self, specs: impl IntoIterator<Item = WorkSpec>) -> Self {
        self.composed = specs.into_iter().collect();
        self
    }

    /// Append a single composed worker group to the existing list
    /// (appending setter).
    ///
    /// The supplied [`WorkSpec`] is PUSHED onto the existing
    /// `composed` vec; previously-set groups are preserved.
    /// Convenience for chained construction:
    /// `cfg.push_composed(a).push_composed(b)` produces
    /// `composed: [a, b]`.
    ///
    /// Use this when building the group list incrementally. To
    /// replace the entire list in one call, use the replacing
    /// [`composed`](Self::composed) instead.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn push_composed(mut self, spec: WorkSpec) -> Self {
        self.composed.push(spec);
        self
    }
}
