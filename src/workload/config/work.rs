//! Per-group worker specification: the [`WorkSpec`] struct, its
//! `Default`, and the chainable builder methods on `impl WorkSpec`.
//!
//! WorkSpec is the per-cgroup-group worker shape that composes into
//! [`WorkloadConfig::composed`](super::WorkloadConfig::composed) or
//! `CgroupDef`'s `merged_works`. Each WorkSpec spawns its own set of
//! worker processes with its own work_type, sched_policy, affinity,
//! mem_policy, nice, comm, pcomm, uid, gid, numa_node, and
//! workers_pct.
//!
//! WorkSpec deliberately omits `clone_mode` — clone-mode is a
//! workload-wide property carried by `WorkloadConfig`.
//!
//! Per-spec validation lives at apply-setup / spawn time:
//! `mem_policy` is validated by
//! [`WorkloadConfig::validate`](super::WorkloadConfig::validate)
//! before any worker context exists; `workers_pct` is resolved
//! per-cpuset by [`WorkSpec::resolve_workers_pct`] at dispatch.

use std::borrow::Cow;

use super::super::{AffinityIntent, WorkType};
use super::{MemPolicy, MpolFlags, SchedPolicy};

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
// See [`WorkType`]'s `#[serde(bound(...))]` comment — embedding
// `WorkType` here propagates the same lifetime-bound issue, so we
// pass through the same explicit empty bound.
#[serde(bound(deserialize = ""))]
pub struct WorkSpec {
    /// What each worker does.
    pub work_type: WorkType,
    /// Linux scheduling policy.
    pub sched_policy: SchedPolicy,
    /// Number of workers. `None` means use `Ctx::workers_per_cgroup`.
    ///
    /// Composition-sensitive: different work groups within the same
    /// cgroup commonly want different worker counts (e.g. an
    /// antagonist with 4 spinners alongside a victim with 1
    /// SCHED_FIFO worker). For that reason `CgroupDef` does NOT
    /// expose a cgroup-level default for `num_workers` — multi-group
    /// cgroups set the count per-[`WorkSpec`] here.
    pub num_workers: Option<usize>,
    /// Per-worker affinity intent. Resolved to `ResolvedAffinity` at
    /// runtime via [`resolve_affinity_for_cgroup()`](crate::scenario::resolve_affinity_for_cgroup).
    pub affinity: AffinityIntent,
    /// NUMA memory placement policy. Applied via `set_mempolicy(2)`
    /// after fork, before the work loop.
    ///
    /// Validated against the resolved cpuset per-WorkSpec at
    /// apply-setup time. Because validation is per-group, a
    /// cgroup-level default would mask per-group failures with
    /// confusing diagnostics — `CgroupDef` deliberately does not
    /// expose a cgroup-level default for `mem_policy`; multi-group
    /// cgroups set it per-[`WorkSpec`] here.
    pub mem_policy: MemPolicy,
    /// Optional mode flags for `set_mempolicy(2)`.
    pub mpol_flags: MpolFlags,
    /// Per-worker nice value applied via `setpriority(2)` after
    /// fork, before the work loop. See [`WorkloadConfig::nice`]
    /// for range, `None`-vs-`Some(n)` semantics, and `CAP_SYS_NICE`
    /// rules.
    ///
    /// To inherit a cgroup-level default stored at
    /// [`CgroupDef::default_nice`](crate::scenario::ops::CgroupDef::default_nice),
    /// leave this `None`. `Some(0)` opts out of the cgroup-level
    /// merge — see [`WorkloadConfig::nice`] for the underlying
    /// `setpriority(PRIO_PROCESS, 0, 0)` semantics.
    pub nice: Option<i32>,
    /// Per-worker comm set via `prctl(PR_SET_NAME)` at thread
    /// creation time (the kernel truncates to `TASK_COMM_LEN - 1 =
    /// 15` bytes inside `__set_task_comm`). `None` inherits the
    /// binary name. Useful for scheduler matchers that filter on
    /// `task->comm` (e.g. layered's `CommPrefix`). The comm is
    /// applied once per worker; it is NOT live-propagated after
    /// the worker enters its work loop.
    pub comm: Option<Cow<'static, str>>,
    /// The thread-group leader's comm — what schedulers read as
    /// `task->group_leader->comm`. When set, `apply_setup` coalesces
    /// every WorkSpec sharing this `pcomm` value (within one
    /// CgroupDef) into ONE forked thread-group leader. The leader's
    /// `task->comm` is set via `prctl(PR_SET_NAME)` (kernel
    /// truncates to `TASK_COMM_LEN - 1 = 15` bytes inside
    /// `__set_task_comm`), so every worker thread inside observes
    /// `task->group_leader->comm == pcomm` for the leader's
    /// lifetime. WorkSpecs with `pcomm = None` (or empty pcomm
    /// string, treated as `None`) spawn via the conventional fork
    /// path — one process per worker.
    ///
    /// **Dispatch is `apply_setup`-only.** Direct calls to
    /// [`crate::workload::WorkloadHandle::spawn`] and
    /// [`crate::scenario::ops::Op::Spawn`] do NOT honor `pcomm` —
    /// they always spawn one process per worker (fork mode). To
    /// drive the pcomm container path without going through
    /// `CgroupDef`, callers may invoke
    /// [`crate::workload::WorkloadHandle::spawn_pcomm_cgroup`]
    /// directly with a `&[WorkSpec]` slice.
    ///
    /// This is the AUTHORITATIVE source for the pcomm dispatch:
    /// `apply_setup` reads it directly from each WorkSpec.
    /// `crate::scenario::ops::types::CgroupDef::pcomm` is a
    /// convenience method that writes the same value into every
    /// WorkSpec at builder time; there is no separate cgroup-level
    /// pcomm field.
    ///
    /// Per-thread comm goes through [`Self::comm`] and the worker's
    /// own `prctl(PR_SET_NAME)` at thread creation time. Models
    /// real workloads like `chrome` (pcomm) hosting
    /// `ThreadPoolForeg` and `GPU Process` worker threads
    /// (per-thread comm), or `java` (pcomm) hosting `GC Thread`
    /// and `C2 CompilerThre` worker threads.
    pub pcomm: Option<Cow<'static, str>>,
    /// Effective UID set via `setresuid(uid, uid, uid)` after fork.
    /// `None` inherits the parent's euid. Useful for scheduler
    /// matchers that filter on `task->real_cred->euid` (e.g.
    /// layered's `UIDEquals`).
    pub uid: Option<u32>,
    /// Effective GID set via `setresgid(gid, gid, gid)` after fork.
    /// `None` inherits the parent's egid.
    pub gid: Option<u32>,
    /// Restrict worker affinity to the CPUs of this NUMA node.
    /// Applied via `sched_setaffinity` after fork. Useful for
    /// scheduler matchers that check `bpf_cpumask_subset(cpus_ptr,
    /// node_cpumask)` (e.g. layered's `NumaNode`).
    pub numa_node: Option<u32>,
    /// Optional fraction-of-cpuset worker count. When `Some(p)`, the
    /// dispatch site computes `ceil(cpuset_cpus * p)` and writes the
    /// result into `num_workers`. The denominator is the cgroup's
    /// currently-recorded cpuset at dispatch time:
    ///
    /// - `apply_setup` dispatch: the cgroup was just created and its
    ///   cpuset just resolved via `CpusetSpec::resolve(ctx)` (or
    ///   inherited from `ctx.topo.usable_cpuset()` when the
    ///   `CgroupDef` has no `.cpuset(...)`), so the denominator
    ///   matches the declared `CpusetSpec`.
    /// - `Op::Spawn` dispatch: the denominator is whatever cpuset is
    ///   currently recorded for the cgroup. A prior `Op::SetCpuset`
    ///   that narrowed the cgroup will narrow the denominator too.
    ///   Workers already spawned by a prior `apply_setup` are not
    ///   re-counted.
    ///
    /// Cannot coexist with `num_workers = Some(_)` — validation
    /// rejects that combination because it's ambiguous which source
    /// wins. Values > 1.0 are accepted as deliberate oversubscription
    /// (e.g. `workers_pct(2.0)` on a 10-CPU cpuset produces 20
    /// workers). NaN/Inf/negative are rejected at construction time.
    pub workers_pct: Option<f64>,
}

impl Default for WorkSpec {
    /// Single SpinWait worker under the kernel's default scheduling
    /// class — the framework's no-customization baseline. Every
    /// other field is `None` / inherit so a test that needs a
    /// specific knob (`affinity`, `mem_policy`, `nice`, etc.) sets
    /// only that one via the corresponding `WorkSpec::with_*`
    /// builder. `num_workers = None` defers count selection to
    /// `CgroupDef`'s merged-works contract (the cgroup-level
    /// default applies; see `CgroupDef::workers` /
    /// `CgroupDef::merged_works`). The `workers_pct` mutex with
    /// `num_workers` only fires when BOTH are `Some(_)` — at
    /// default neither is set, so the
    /// `CgroupDef::resolve_workers_pct` arm that emits the
    /// `WorkSpec sets BOTH workers(...) and workers_pct(...)` bail
    /// does not trigger.
    fn default() -> Self {
        Self {
            work_type: WorkType::SpinWait,
            sched_policy: SchedPolicy::Normal,
            num_workers: None,
            affinity: AffinityIntent::Inherit,
            mem_policy: MemPolicy::Default,
            mpol_flags: MpolFlags::NONE,
            nice: None,
            comm: None,
            pcomm: None,
            uid: None,
            gid: None,
            numa_node: None,
            workers_pct: None,
        }
    }
}

impl WorkSpec {
    /// Set the number of workers.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn workers(mut self, n: usize) -> Self {
        self.num_workers = Some(n);
        self
    }

    /// Set the worker count as a fraction of the resolved cpuset
    /// CPU count. Apply-setup computes `ceil(cpuset_cpus * pct)` and
    /// writes the result into `num_workers`. Use this when the worker
    /// count should scale with the cpuset rather than hardcoding a
    /// per-topology constant.
    ///
    /// Setting BOTH `workers(n)` and `workers_pct(p)` on the same
    /// WorkSpec is rejected at apply-setup time because the two sources
    /// would silently fight; pick one. Values > 1.0 are accepted as
    /// deliberate oversubscription; NaN, infinite, and non-positive
    /// values are rejected here at construction time via an assertion.
    ///
    /// # Panics
    ///
    /// Panics when `pct` is NaN, infinite, or `<= 0.0`. The builder
    /// returns `Self`, so the construction-time gate uses `assert!`
    /// rather than a fallible `Result`. Negative or zero fractions
    /// would resolve to zero workers — caught at apply-setup time by
    /// `resolve_num_workers`'s zero-workers rejection anyway, but the
    /// construction-time message is more actionable.
    ///
    /// Extreme finite values (e.g. `1e100`) pass the construction gate
    /// and saturate to `usize::MAX` via the `as` cast in
    /// `resolve_workers_pct` (RFC 2484 / Rust 1.45+). Attempting to
    /// spawn that many workers would OOM the host. The framework
    /// imposes no upper cap; as a rule of thumb keep `pct` near the
    /// intended oversubscription factor (e.g. `1.0`, `2.0`, `4.0`).
    #[must_use = "builder methods consume self; bind the result"]
    pub fn workers_pct(mut self, pct: f64) -> Self {
        assert!(
            pct.is_finite() && pct > 0.0,
            "WorkSpec::workers_pct({pct}): pct must be finite and > 0.0",
        );
        self.workers_pct = Some(pct);
        self
    }

    /// Resolve `workers_pct` against a cpuset size into a concrete
    /// `num_workers` count and clear the fractional state, leaving
    /// `num_workers = Some(scaled)` and `workers_pct = None`. Used by
    /// both `apply_setup` (per-CgroupDef WorkSpec) and `Op::Spawn`
    /// (mid-step ad-hoc spawn) so the two paths produce identical
    /// counts for the same `(pct, cpuset_size)` pair.
    ///
    /// Rejects the ambiguous `(num_workers = Some, workers_pct =
    /// Some)` combination with an `anyhow::bail!` naming the cgroup.
    /// Rejects a computed count of zero (e.g. empty cpuset, or
    /// fraction so small it rounds down) with an actionable diagnostic
    /// naming the cgroup, the cpuset size, and the requested fraction.
    /// Returns the original [`WorkSpec`] unchanged when `workers_pct` is
    /// `None`.
    pub(crate) fn resolve_workers_pct(
        mut self,
        cpuset_cpus: usize,
        cgroup_name: &str,
    ) -> anyhow::Result<Self> {
        let Some(pct) = self.workers_pct else {
            return Ok(self);
        };
        if let Some(n) = self.num_workers {
            anyhow::bail!(
                "cgroup '{}': WorkSpec sets BOTH workers({n}) and \
                 workers_pct({pct}); pick one — workers_pct resolves the \
                 cpuset fraction at apply-setup time and is incompatible \
                 with an explicit count",
                cgroup_name,
            );
        }
        let scaled = (cpuset_cpus as f64 * pct).ceil() as usize;
        if scaled == 0 {
            anyhow::bail!(
                "cgroup '{cgroup_name}': workers_pct({pct}) on a cpuset of \
                 {cpuset_cpus} CPU(s) resolved to 0 workers \
                 (ceil({cpuset_cpus} * {pct}) = 0); the cgroup would \
                 have no workers and downstream assertions would \
                 vacuously pass — narrow the cpuset, raise the fraction, \
                 or use `workers(N)` instead",
            );
        }
        self.num_workers = Some(scaled);
        self.workers_pct = None;
        Ok(self)
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

    /// Set the per-worker affinity intent.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn affinity(mut self, a: AffinityIntent) -> Self {
        self.affinity = a;
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
    /// Stores `Some(n)` on the spec; the spawn pipeline calls
    /// `setpriority(PRIO_PROCESS, 0, n)` unconditionally (including
    /// `n == 0`). The "skip the syscall, inherit the parent's nice"
    /// state is the type-level default `None` — leave the builder
    /// unchained for inherit semantics. Values below the calling
    /// task's current nice require `CAP_SYS_NICE`; see
    /// [`WorkloadConfig::nice`] for the full `can_nice` rule.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn nice(mut self, n: i32) -> Self {
        self.nice = Some(n);
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

    /// Set the thread-group leader's comm. Triggers fork-then-thread
    /// spawn through `apply_setup` (or via
    /// [`crate::workload::WorkloadHandle::spawn_pcomm_cgroup`] for
    /// the direct entry point): one forked leader process whose
    /// `task->comm` is `name`, threads spawned inside it. Each
    /// thread additionally sets its own `task->comm` via
    /// [`Self::comm`] at thread creation time.
    ///
    /// # Panics
    ///
    /// Panics on programmer-error inputs:
    /// - Empty string — the empty pcomm has no observable effect
    ///   (kernel sets task->comm to ""), so it's a no-op disguised
    ///   as configuration. `apply_setup` treats empty as `None` to
    ///   keep the dispatch contract unambiguous, but accepting the
    ///   builder call would silently drop user intent. Reject up
    ///   front.
    /// - Interior NUL byte — `prctl(PR_SET_NAME)` takes a C string;
    ///   any embedded NUL truncates the kernel-side comm at the
    ///   first NUL silently, producing a comm value the caller
    ///   didn't ask for. Reject so the operator sees the error
    ///   immediately instead of debugging a truncated comm.
    ///
    /// `name.len() > 15` is NOT a panic — the kernel truncates to
    /// `TASK_COMM_LEN - 1 = 15` bytes inside `__set_task_comm`, and
    /// some test fixtures intentionally exercise the truncation
    /// boundary. `apply_setup` emits a `tracing::warn!` at
    /// dispatch time so operators see the truncation; the actual
    /// kernel truncation is silent.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn pcomm(mut self, name: impl Into<Cow<'static, str>>) -> Self {
        let name: Cow<'static, str> = name.into();
        assert!(
            !name.is_empty(),
            "WorkSpec::pcomm: empty pcomm string rejected — \
             use `None` (default) for no pcomm, not an empty value",
        );
        assert!(
            !name.contains('\0'),
            "WorkSpec::pcomm: pcomm string {name:?} contains an interior NUL byte; \
             prctl(PR_SET_NAME) treats it as a C string and would truncate \
             at the NUL — strip it before calling .pcomm()",
        );
        self.pcomm = Some(name);
        self
    }
}
