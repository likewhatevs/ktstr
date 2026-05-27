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

/// Validate a `comm` / `pcomm` builder argument.
///
/// Centralizes the rejection contract for the task-name fields the
/// framework writes via `prctl(PR_SET_NAME)`:
///
/// - Empty: produces an empty kernel comm (surprising; breaks
///   scheduler matchers that look for non-empty names).
/// - Interior NUL: `prctl` takes a C string and would truncate at
///   the NUL silently.
/// - Length > 15: `__set_task_comm` (fs/exec.c) writes
///   `min(strlen(buf), sizeof(tsk->comm) - 1)` bytes and
///   `TASK_COMM_LEN = 16` (include/linux/sched.h), so any 16th
///   byte (and beyond) is silently dropped. Rejecting at builder
///   time means the operator sees the limit at the call site
///   instead of debugging a truncated comm.
///
/// `field` names the call site for the panic message (e.g.
/// `"WorkSpec::comm"`, `"CgroupDef::pcomm"`) so the operator
/// can grep the offending builder from the panic.
///
/// Panics intentionally — these are builder-time input errors
/// that the test author must fix at the source. Returning a
/// `Result` would force every caller to `unwrap` and lose the
/// site context.
pub(crate) fn validate_task_comm_string(field: &str, name: &str) {
    assert!(
        !name.is_empty(),
        "{field}: empty string rejected — use `None` (default) for no override, not an empty value",
    );
    assert!(
        !name.contains('\0'),
        "{field}: string {name:?} contains an interior NUL byte; \
         prctl(PR_SET_NAME) treats it as a C string and would \
         truncate at the NUL — strip it before calling .{field}()",
    );
    assert!(
        name.len() <= 15,
        "{field}: name {name:?} is {} bytes; kernel TASK_COMM_LEN \
         limit is 15 bytes (TASK_COMM_LEN-1=15 in include/linux/sched.h; \
         `__set_task_comm` truncates at that cap) — shorten before \
         calling .{field}()",
        name.len(),
    );
}

// PartialEq (not Eq): the [`Self::workers_pct`] field is `Option<f64>`
// and `f64` is `PartialEq` only — `f64::partial_cmp(NaN, NaN)` is
// `None` (IEEE-754 semantics). The [`Self::workers_pct`] builder
// rejects NaN at construction (see the `assert!` near the top of
// `impl WorkSpec::workers_pct`), so production `WorkSpec` values
// are NaN-free in practice — the derive inherits f64's standard
// semantics without surfacing them at typical call sites. Tests
// that synthesize WorkSpec values via struct-literal syntax can
// still introduce NaN; concretely, `assert_eq!(spec, spec)` will
// FAIL (panic) for a spec containing NaN `workers_pct` because
// NaN != NaN per IEEE-754. Avoid synthesizing NaN even in tests.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
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
    ///
    /// Type asymmetry with [`crate::workload::WorkloadConfig::num_workers`]
    /// (`usize`, no Option) is deliberate. `WorkSpec` is the
    /// declarative spec layer where `None` is a meaningful
    /// "inherit the cgroup-level default" sentinel;
    /// [`crate::scenario::resolve_num_workers`] coalesces it to a
    /// concrete `usize` against the `Ctx` before
    /// [`crate::workload::WorkloadConfig::for_scenario_engine`]
    /// constructs the spawn-time config. The coalesce happens at
    /// the resolution boundary, not silently inside any builder.
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
    /// fork, before the work loop. See [`crate::workload::WorkloadConfig::nice`]
    /// for range, `None`-vs-`Some(n)` semantics, and `CAP_SYS_NICE`
    /// rules.
    ///
    /// To inherit a cgroup-level default stored at
    /// [`CgroupDef::default_nice`](crate::scenario::ops::CgroupDef::default_nice),
    /// leave this `None`. `Some(0)` opts out of the cgroup-level
    /// merge — see [`crate::workload::WorkloadConfig::nice`] for the underlying
    /// `setpriority(PRIO_PROCESS, 0, 0)` semantics.
    pub nice: Option<i32>,
    /// Per-worker comm set via `prctl(PR_SET_NAME)` at thread
    /// creation time. The setter rejects > 15 bytes
    /// (TASK_COMM_LEN-1) at construction so the operator sees the
    /// cap at the call site instead of debugging a kernel-truncated
    /// comm — see [`validate_task_comm_string`]. `None` inherits the
    /// binary name. Useful for scheduler matchers that filter on
    /// `task->comm` (e.g. layered's `CommPrefix`). The comm is
    /// applied once per worker; it is NOT live-propagated after
    /// the worker enters its work loop.
    pub comm: Option<Cow<'static, str>>,
    /// The thread-group leader's comm — what schedulers read as
    /// `task->group_leader->comm`. When set, `apply_setup` coalesces
    /// every WorkSpec sharing this `pcomm` value (within one
    /// CgroupDef) into ONE forked thread-group leader. The leader's
    /// `task->comm` is set via `prctl(PR_SET_NAME)`; the setter
    /// rejects > 15 bytes (TASK_COMM_LEN-1) at construction so the
    /// `task->group_leader->comm == pcomm` invariant every worker
    /// thread observes for the leader's lifetime matches the
    /// requested string exactly (no silent kernel truncation).
    /// WorkSpecs with `pcomm = None` (or empty pcomm string,
    /// treated as `None`) spawn via the conventional fork path —
    /// one process per worker.
    ///
    /// **Dispatch is `apply_setup`-only.** The `WorkSpec::pcomm`
    /// setter itself always accepts a valid value (subject to the
    /// existing 15-byte / NUL / empty-string checks). The bail
    /// fires later at **`WorkloadConfig` dispatch-construction
    /// time** — direct calls to
    /// [`crate::workload::WorkloadHandle::spawn`] (composed entries)
    /// and the scenario-engine spawn-dispatch sites
    /// ([`crate::scenario::ops::Op::Spawn`] / `apply_setup` non-pcomm
    /// path) all reject a pcomm-bearing
    /// WorkSpec when they synthesize the per-spawn `WorkloadConfig`.
    /// Those paths always fork one process per worker (fork mode),
    /// so `task->group_leader->comm` would be left at the parent's
    /// task->comm at fork time (the scenario runner's binary name)
    /// and scheduler matchers filtering on the leader's comm would
    /// see zero matches. The bail surfaces the misuse at the call
    /// site instead of producing a workload that silently fails to
    /// match its fixture. To drive the pcomm container path without
    /// going through `CgroupDef`, callers may invoke
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
    ///
    /// Declarative-only field — absent from
    /// [`crate::workload::WorkloadConfig`] by design (same shape as
    /// [`Self::num_workers`] / [`Self::workers_pct`]). pcomm is the
    /// operator-facing intent that drives the apply_setup pcomm-
    /// aware coalesce path; by the time a `WorkloadConfig` is
    /// constructed for spawn, the per-WorkSpec pcomm has either
    /// routed through `spawn_pcomm_cgroup` (then no longer needed
    /// at the WorkloadConfig layer) or hit the dispatch-construction
    /// bail at
    /// [`crate::workload::WorkloadConfig::for_scenario_engine`].
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
    /// - `Op::Spawn(SpawnPlacement::Cgroup)` dispatch: the denominator
    ///   is whatever cpuset is currently recorded for the cgroup. A
    ///   prior `Op::SetCpuset` that narrowed the cgroup will narrow
    ///   the denominator too. Workers already spawned by a prior
    ///   `apply_setup` are not re-counted.
    ///
    /// Cannot coexist with `num_workers = Some(_)` — validation
    /// rejects that combination because it's ambiguous which source
    /// wins. Values > 1.0 are accepted as deliberate oversubscription
    /// (e.g. `workers_pct(2.0)` on a 10-CPU cpuset produces 20
    /// workers). NaN/Inf/negative are rejected at construction time.
    ///
    /// Declarative-only field — absent from
    /// [`crate::workload::WorkloadConfig`] by design. The same
    /// asymmetry as [`Self::num_workers`]: `WorkSpec` is the
    /// operator-facing declarative spec where `workers_pct(p)` is
    /// a meaningful "scale with the cpuset" intent;
    /// [`Self::resolve_workers_pct`] (called at apply_setup and at
    /// each dispatch site) computes the concrete worker count
    /// against the dispatch-time cpuset and writes the result into
    /// the WorkSpec's `num_workers` field, which then flows through
    /// the standard `resolve_num_workers` →
    /// [`crate::workload::WorkloadConfig::num_workers`] migration
    /// boundary. There is no `workers_pct` field on `WorkloadConfig`
    /// because by spawn time the scale-with-cpuset intent has
    /// already collapsed to a concrete count.
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
    /// both `apply_setup` (per-CgroupDef WorkSpec) and
    /// `Op::Spawn(SpawnPlacement::Cgroup)` (mid-step ad-hoc spawn)
    /// so the two paths produce identical counts for the same
    /// `(pct, cpuset_size)` pair.
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
    /// [`crate::workload::WorkloadConfig::nice`] for the full `can_nice` rule.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn nice(mut self, n: i32) -> Self {
        self.nice = Some(n);
        self
    }

    /// Set the worker process name via `prctl(PR_SET_NAME)`.
    ///
    /// # Panics
    ///
    /// Panics on programmer-error inputs — same three cases as
    /// [`Self::pcomm`]:
    /// - Empty string (silent kernel-comm clobber).
    /// - Interior NUL byte (prctl C-string truncation).
    /// - More than 15 bytes (`TASK_COMM_LEN - 1` —
    ///   `__set_task_comm` truncates at 15 so the framework rejects
    ///   at construction to keep the kernel-observed comm equal to
    ///   the requested value).
    ///
    /// See [`validate_task_comm_string`] for the centralized
    /// rationale; `name.len()` is the BYTE length (UTF-8 multi-byte
    /// chars count as their byte width, not their codepoint count).
    #[must_use = "builder methods consume self; bind the result"]
    pub fn comm(mut self, name: impl Into<Cow<'static, str>>) -> Self {
        let name: Cow<'static, str> = name.into();
        validate_task_comm_string("WorkSpec::comm", &name);
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
    /// - More than 15 bytes — `__set_task_comm` writes
    ///   `min(strlen(buf), sizeof(tsk->comm) - 1)` bytes and
    ///   `TASK_COMM_LEN = 16`, so the 16th byte (and beyond) is
    ///   silently dropped. Rejecting at construction time means the
    ///   `task->group_leader->comm == pcomm` invariant the rest of
    ///   the framework relies on holds exactly, and the operator
    ///   sees the cap at the call site instead of debugging a
    ///   truncated comm.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn pcomm(mut self, name: impl Into<Cow<'static, str>>) -> Self {
        let name: Cow<'static, str> = name.into();
        validate_task_comm_string("WorkSpec::pcomm", &name);
        self.pcomm = Some(name);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "WorkSpec::comm: empty string rejected")]
    fn work_spec_comm_rejects_empty() {
        let _ = WorkSpec::default().comm("");
    }

    #[test]
    #[should_panic(expected = "interior NUL byte")]
    fn work_spec_comm_rejects_interior_nul() {
        let _ = WorkSpec::default().comm("foo\0bar");
    }

    #[test]
    #[should_panic(expected = "WorkSpec::pcomm: empty string rejected")]
    fn work_spec_pcomm_rejects_empty() {
        let _ = WorkSpec::default().pcomm("");
    }

    #[test]
    #[should_panic(expected = "interior NUL byte")]
    fn work_spec_pcomm_rejects_interior_nul() {
        let _ = WorkSpec::default().pcomm("foo\0bar");
    }

    #[test]
    fn work_spec_comm_accepts_15_byte_boundary() {
        let fifteen = "a".repeat(15);
        let spec = WorkSpec::default().comm(fifteen.clone());
        assert_eq!(spec.comm.as_deref(), Some(fifteen.as_str()));
    }

    #[test]
    #[should_panic(expected = "WorkSpec::comm: name")]
    fn work_spec_comm_rejects_16_byte_overflow() {
        let _ = WorkSpec::default().comm("a".repeat(16));
    }

    #[test]
    fn work_spec_pcomm_accepts_15_byte_boundary() {
        let fifteen = "a".repeat(15);
        let spec = WorkSpec::default().pcomm(fifteen.clone());
        assert_eq!(spec.pcomm.as_deref(), Some(fifteen.as_str()));
    }

    #[test]
    #[should_panic(expected = "WorkSpec::pcomm: name")]
    fn work_spec_pcomm_rejects_16_byte_overflow() {
        let _ = WorkSpec::default().pcomm("a".repeat(16));
    }

    /// UTF-8 boundary: the length cap is byte-counted, not
    /// codepoint-counted. 5 Cyrillic chars = 10 bytes (each char in
    /// U+0400-U+04FF is 2 bytes) — accepts. 8 Cyrillic chars = 16
    /// bytes — panics. Sanity-check the assumption with `s.len()`
    /// at runtime so a future Cyrillic literal in this test that
    /// drifts off the 2-byte assumption surfaces immediately
    /// instead of as a false acceptance.
    #[test]
    fn work_spec_comm_accepts_10_byte_utf8_within_cap() {
        let cyr10 = "приве";
        assert_eq!(
            cyr10.len(),
            10,
            "test fixture: cyrillic 5-char must be 10 bytes"
        );
        let spec = WorkSpec::default().comm(cyr10);
        assert_eq!(spec.comm.as_deref(), Some(cyr10));
    }

    #[test]
    #[should_panic(expected = "16 bytes")]
    fn work_spec_comm_rejects_16_byte_utf8_overflow() {
        let cyr16 = "приветик";
        assert_eq!(
            cyr16.len(),
            16,
            "test fixture: cyrillic 8-char must be 16 bytes"
        );
        let _ = WorkSpec::default().comm(cyr16);
    }
}
