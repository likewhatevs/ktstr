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
//! currently `num_workers > 0` (primary + every composed entry;
//! rejects vacuously-passing zero-worker workloads where every
//! assertion would trivially pass) and `mem_policy` empty-nodemask
//! rejection (primary + every composed entry).

use std::borrow::Cow;

use super::super::{AffinityIntent, WorkType};
use super::{CloneMode, MemPolicy, MpolFlags, SchedPolicy, WorkSpec};

/// Configuration for spawning a group of worker processes.
//
// PartialEq (not Eq): the [`Self::composed`] field is
// `Vec<WorkSpec>`, and `WorkSpec` is `PartialEq`-only because of
// its `workers_pct: Option<f64>` field — see the derive comment on
// [`WorkSpec`] for the IEEE-754 NaN rationale. Production
// WorkSpec values are NaN-free at construction (the
// `WorkSpec::workers_pct` builder rejects NaN), so the inherited
// f64 semantics do not surface at typical call sites.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
// See [`WorkType`]'s `#[serde(bound(...))]` comment — embedding
// `WorkType` propagates the same lifetime-bound issue, so we pass
// through the same explicit empty bound.
#[serde(bound(deserialize = ""))]
pub struct WorkloadConfig {
    /// Number of worker processes to fork. Concrete `usize` (not
    /// `Option`): `WorkloadConfig` is the spawn-time configuration
    /// passed to `WorkloadHandle::spawn`, by which point the worker
    /// count must be known. The Option-to-usize coalescing happens
    /// upstream at `resolve_num_workers`, which
    /// reads [`crate::workload::WorkSpec::num_workers`]
    /// (`Option<usize>` — `None` falls back to
    /// `Ctx::workers_per_cgroup`) and produces the resolved value
    /// passed to `Self::for_scenario_engine`. The type asymmetry
    /// is deliberate: `WorkSpec` is the user-facing declarative
    /// spec where `None` means "inherit the cgroup-level default",
    /// `WorkloadConfig` is the spawn-time concrete config where
    /// `usize` is the only sensible type.
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
    /// The setter rejects > 15 bytes (TASK_COMM_LEN-1) at
    /// construction so the operator sees the cap at the call site
    /// instead of debugging a truncated comm. `None` inherits the
    /// binary name. Mirrors [`WorkSpec::comm`] so the primary group
    /// exposes the same scheduler-matcher knob composed entries
    /// already do.
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
    /// Construct a `WorkloadConfig` for scenario-engine spawn
    /// dispatch (apply_setup non-pcomm, Op::Spawn). The signature
    /// pins `clone_mode =
    /// CloneMode::Fork` in the constructor body so callers can't
    /// accidentally route a Thread-mode workload through these
    /// sites — a Thread-mode spawn would migrate the scenario
    /// runner's tgid into the test cgroup when move_tasks fires,
    /// per `kernel/cgroup/cgroup.c::cgroup_procs_write_start`
    /// (cgroup.procs writes are process-scoped). The
    /// previously-needed `debug_assert_eq!(wl.clone_mode, Fork)`
    /// guards collapse into the type system.
    ///
    /// `work_type` is taken as an arg (not pulled from `work.work_type`)
    /// because apply_setup's per-WorkSpec resolution layers a
    /// `ctx.work_type_override` over the spec-declared type; passing
    /// the resolved value keeps that call site honest.
    ///
    /// `affinity` is taken as an arg (not `work.affinity`) because
    /// all three call sites pre-resolve the intent via
    /// `intent_for_spawn` against the cgroup cpuset before
    /// dispatch.
    ///
    /// **Do NOT pass `work.num_workers.unwrap()` / `work.affinity` /
    /// `work.work_type` directly** — that bypasses the resolution
    /// layer (workers_pct → ceil(cpuset * pct), cgroup-cpuset-aware
    /// intent_for_spawn, work_type override) and silently produces
    /// wrong counts / wrong affinity / wrong type. Always pass the
    /// resolved values from `resolve_num_workers` /
    /// `intent_for_spawn` / `resolve_work_type`; the args/fields
    /// asymmetry is deliberate and the resolution layer is
    /// load-bearing.
    ///
    /// `composed` always empty: the scenario-engine spawn dispatch
    /// emits one WorkloadConfig per WorkSpec from the resolved
    /// `non_pcomm_works` list, so composition is upstream. A future
    /// "composed in single spawn" optimization would need a
    /// sibling constructor, not a parameterized one.
    ///
    /// Bails when `work.pcomm.is_some()`: the scenario-engine spawn
    /// dispatch (apply_setup non-pcomm path, Op::Spawn) forks one
    /// process per worker and does not
    /// route through `spawn_pcomm_cgroup`, so `task->group_leader->comm`
    /// would be left at the binary name rather than the requested
    /// pcomm value — workers spawn but scheduler matchers filtering
    /// on the group_leader comm see zero matches. Mirrors the
    /// `composed[i].pcomm.is_some()` bail at
    /// `WorkloadHandle::spawn`. Test authors wanting pcomm route via
    /// `CgroupDef::pcomm` + apply_setup pcomm-aware fan-out, or call
    /// `WorkloadHandle::spawn_pcomm_cgroup` directly.
    pub(crate) fn for_scenario_engine(
        work: &WorkSpec,
        num_workers: usize,
        affinity: AffinityIntent,
        work_type: WorkType,
    ) -> anyhow::Result<Self> {
        if work.pcomm.is_some() {
            anyhow::bail!(
                "WorkSpec::pcomm is unsupported in the scenario-engine \
                 spawn dispatch (apply_setup non-pcomm path, \
                 Op::Spawn) — those sites fork \
                 one process per worker rather than threading inside \
                 a pcomm container, so `task->group_leader->comm` \
                 would stay at the binary name. To run with a \
                 specific group-leader comm, declare \
                 `CgroupDef::pcomm` (apply_setup picks up the pcomm-\
                 aware coalesce path) or call \
                 `WorkloadHandle::spawn_pcomm_cgroup` directly.",
            );
        }
        Ok(Self {
            num_workers,
            affinity,
            work_type,
            sched_policy: work.sched_policy,
            mem_policy: work.mem_policy.clone(),
            mpol_flags: work.mpol_flags,
            nice: work.nice,
            clone_mode: CloneMode::Fork,
            comm: work.comm.clone(),
            uid: work.uid,
            gid: work.gid,
            numa_node: work.numa_node,
            composed: Vec::new(),
        })
    }

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
    /// Two gates, in order:
    /// 1. `num_workers > 0` on the primary group and on every
    ///    composed [`WorkSpec`] entry — zero workers emit no
    ///    `WorkerReport`s and downstream assertions would vacuously
    ///    pass. Composed entries also route through
    ///    `WorkloadHandle::spawn` (via `GroupParams::from_composed`)
    ///    directly, bypassing the scenario-engine's
    ///    `resolve_num_workers` resolver, so the gate must live
    ///    here to catch `composed[i].num_workers=0` before the spawn
    ///    cascade forks anything.
    /// 2. `mem_policy` on the primary group and on every composed
    ///    [`WorkSpec`] entry.
    ///
    /// Per-entry errors name the offending slot (`"primary"` or
    /// `"composed[N] (group_idx M)"`) so the test author can
    /// locate the misconfigured group. Gate (1) runs first so the
    /// more-fundamental "no workers" diagnostic surfaces before a
    /// secondary mem_policy failure (which becomes moot when no
    /// worker exists to bind).
    ///
    /// # Scope
    ///
    /// Validates `mem_policy` and `num_workers > 0`. Other field
    /// invariants are validated at their own use sites:
    /// `workers_pct` via `WorkSpec::resolve_workers_pct`,
    /// [`WorkType`] payloads via per-variant constructors and
    /// `validate_workload_admission`, [`AffinityIntent`] topology
    /// rules at the scenario-engine `resolve_affinity_for_cgroup`
    /// resolver. This method is the home for invariants that must
    /// hold BEFORE any worker context (threads, forks, cgroups)
    /// exists — `mem_policy` qualifies because of the silent-skip +
    /// `exit_group` hazard noted above; `num_workers == 0`
    /// qualifies because every downstream gate becomes
    /// vacuous-pass. Future fields with the same
    /// "must-fail-before-spawn" shape belong here too.
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
        // num_workers gate runs FIRST so the operator sees the more-
        // fundamental "no workers" diagnostic before mem_policy
        // failures (which become moot when no worker exists to bind).
        if self.num_workers == 0 {
            anyhow::bail!(
                "WorkloadConfig.num_workers=0 is not allowed — \
                 zero workers emit no WorkerReports and downstream \
                 assertions would vacuously pass. Use at least 1 \
                 worker or drop the WorkloadConfig entirely."
            );
        }
        for (idx, spec) in self.composed.iter().enumerate() {
            // composed entries route through `WorkloadHandle::spawn`
            // (via `GroupParams::from_composed`) directly, bypassing
            // the scenario engine's `resolve_num_workers` —
            // the gate must live here for the spawn entry to catch
            // composed[i].num_workers=0 before forking.
            if spec.num_workers == Some(0) {
                anyhow::bail!(
                    "WorkloadConfig.composed[{idx}].num_workers=0 \
                     (group_idx {}): zero workers in a composed group \
                     emit no WorkerReports for the group; drop the \
                     entry or use >= 1 worker",
                    idx + 1,
                );
            }
        }
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
    ///
    /// # Panics
    ///
    /// Panics on programmer-error inputs — mirrors
    /// [`crate::workload::WorkSpec::pcomm`]'s `# Panics`:
    /// - Empty string.
    /// - Interior NUL byte (prctl C-string truncation).
    /// - More than 15 bytes (`TASK_COMM_LEN - 1` cap).
    ///
    /// See
    /// `validate_task_comm_string`
    /// for the centralized rationale; `name.len()` is the BYTE
    /// length (UTF-8 multi-byte chars count as their byte width).
    #[must_use = "builder methods consume self; bind the result"]
    pub fn comm(mut self, name: impl Into<Cow<'static, str>>) -> Self {
        let name: Cow<'static, str> = name.into();
        crate::workload::validate_task_comm_string("WorkloadConfig::comm", &name);
        self.comm = Some(name);
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn validate_rejects_zero_num_workers_on_primary() {
        let cfg = WorkloadConfig {
            num_workers: 0,
            ..Default::default()
        };
        let err = cfg.validate().expect_err("num_workers=0 must bail");
        let msg = format!("{err:#}");
        assert!(msg.contains("num_workers=0 is not allowed"), "{msg}");
        assert!(msg.contains("vacuously pass"), "{msg}");
    }

    #[test]
    fn validate_rejects_zero_num_workers_on_composed_entry() {
        let cfg = WorkloadConfig {
            num_workers: 1,
            composed: vec![WorkSpec::default().workers(0)],
            ..Default::default()
        };
        let err = cfg
            .validate()
            .expect_err("composed[0].num_workers=0 must bail");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("composed[0]"),
            "must cite the entry idx: {msg}"
        );
        assert!(msg.contains("group_idx 1"), "1-indexed group_idx: {msg}");
    }

    #[test]
    fn validate_accepts_one_or_more_workers_on_primary_and_composed() {
        let cfg = WorkloadConfig {
            num_workers: 1,
            composed: vec![WorkSpec::default().workers(2)],
            ..Default::default()
        };
        cfg.validate().expect("1+composed(2) must validate ok");
    }

    #[test]
    fn validate_rejects_zero_workers_before_mempolicy() {
        // Zero workers + invalid mem_policy: zero-worker check must
        // fire first so the operator's primary diagnostic is the
        // more-fundamental "no workers" rather than the secondary
        // "bad mempolicy" message.
        let cfg = WorkloadConfig {
            num_workers: 0,
            mem_policy: MemPolicy::Bind(BTreeSet::new()), // invalid
            ..Default::default()
        };
        let err = cfg
            .validate()
            .expect_err("zero workers + bad policy must bail on num_workers first");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("num_workers=0"),
            "zero-workers msg surfaces: {msg}"
        );
        assert!(!msg.contains("mem_policy"), "mempolicy msg deferred: {msg}");
    }

    #[test]
    #[should_panic(expected = "WorkloadConfig::comm: empty string rejected")]
    fn workload_config_comm_rejects_empty() {
        let _ = WorkloadConfig::default().comm("");
    }

    #[test]
    #[should_panic(expected = "interior NUL byte")]
    fn workload_config_comm_rejects_interior_nul() {
        let _ = WorkloadConfig::default().comm("foo\0bar");
    }

    /// Per-builder boundary pin: a future refactor that re-routes
    /// WorkloadConfig::comm around the shared
    /// `validate_task_comm_string` helper would surface here even
    /// if the helper-level tests still pass.
    #[test]
    fn workload_config_comm_accepts_15_byte_boundary() {
        let fifteen = "a".repeat(15);
        let cfg = WorkloadConfig::default().comm(fifteen.clone());
        assert_eq!(cfg.comm.as_deref(), Some(fifteen.as_str()));
    }

    #[test]
    #[should_panic(expected = "16 bytes")]
    fn workload_config_comm_rejects_16_byte_overflow() {
        let _ = WorkloadConfig::default().comm("a".repeat(16));
    }
}
