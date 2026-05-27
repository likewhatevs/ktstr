//! Declarative cgroup blueprint — [`CgroupDef`] struct + the
//! full builder-method surface (Group I per-WorkSpec setters,
//! Group II `default_*` merges, Group III in-place `pcomm`
//! stamping, plus the cpu / memory / io / pids controller knobs).
//! See the type-level doc on [`CgroupDef`] for the per-controller
//! summary table and the three builder-pattern groups.
//!
//! `CgroupDef` deliberately has NO `Default` impl — see the note at
//! the foot of this file (and `tests::assert_not_impl_default!` in
//! `super::tests`) for the rationale (`name = "cg_0"` would
//! silently collide with the conventional first cgroup name).

use std::borrow::Cow;
use std::collections::BTreeSet;
use std::time::Duration;

use crate::workload::{WorkSpec, WorkType};

#[allow(unused_imports)] // referenced by intra-doc links
use super::Op;
use super::{CpuLimits, CpusetSpec, IoLimits, MemoryLimits, PidsLimits};

// ---------------------------------------------------------------------------
// CgroupDef
// ---------------------------------------------------------------------------

/// Declarative cgroup definition: name + cpuset + synthetic
/// [`WorkSpec`] groups + optional userspace [`Payload`](crate::test_support::Payload).
///
/// Bundles the ops that always go together (AddCgroup + SetCpuset +
/// Spawn) into a single value. The executor creates the cgroup, optionally
/// sets its cpuset, spawns workers for each [`WorkSpec`] entry, and moves
/// them into the cgroup.
///
/// Multiple [`WorkSpec`] entries run in parallel within the cgroup. Each
/// entry spawns its own set of worker processes. The optional
/// [`Self::payload`] slot is a *single* userspace binary that runs
/// alongside those synthetic [`WorkSpec`] groups (hence "plural works,
/// singular payload" — the pluralization in the legacy "workload(s)"
/// prose elided this distinction).
///
/// Use `CgroupDef` in `Step::with_defs` for scenarios where cgroups are
/// created once and run for the step duration. Use `Op::add_cgroup` +
/// `Op::spawn(SpawnPlacement::cgroup(name), work)` directly when you
/// need mid-step cgroup creation, removal, or other dynamic operations
/// between spawn and collect.
///
/// # Resource controllers overview
///
/// `CgroupDef` exposes one builder method per cgroup v2 controller
/// knob, each writing the corresponding `cgroup.*` / `*.max` /
/// `*.weight` file at `apply_setup` time. The full surface:
///
/// | Controller | One-line description | Builder methods | Underlying file(s) |
/// |------------|----------------------|-----------------|--------------------|
/// | cpuset | Bind to a CPU subset and NUMA-node memory affinity. | [`Self::cpuset`], [`Self::cpuset_mems`] | `cpuset.cpus`, `cpuset.mems` |
/// | cpu    | Bandwidth ceiling (`cpu.max` quota/period) plus relative-share weight. | [`Self::cpu_quota_pct`], [`Self::cpu_quota`], [`Self::cpu_unlimited`], [`Self::cpu_weight`] | `cpu.max`, `cpu.weight` |
/// | memory | Hard ceiling, soft throttle threshold, soft protection floor, swap cap. | [`Self::memory_max`], [`Self::memory_high`], [`Self::memory_low`], [`Self::memory_swap_max`], [`Self::memory_swap_unlimited`], [`Self::memory_unlimited`] | `memory.max`, `memory.high`, `memory.low`, `memory.swap.max` |
/// | io     | Relative IO share (BFQ / io.cost) when the io controller is enabled. | [`Self::io_weight`] | `io.weight` |
/// | pids   | Task-count ceiling — fork(2)/clone(2) returns EAGAIN once the cap is hit. | [`Self::pids_max`], [`Self::pids_unlimited`] | `pids.max` |
/// | freeze | Pause/resume every task in the cgroup mid-run via the JOBCTL freeze path. | (Op-level) [`Op::freeze_cgroup`], [`Op::unfreeze_cgroup`] | `cgroup.freeze` |
///
/// `CgroupDef` covers steady-state resource limits — knobs that
/// hold for the cgroup's whole lifetime. The freeze knob is
/// intentionally exposed at the [`Op`] layer instead, because
/// freeze/unfreeze describe transitions over time (suspend
/// mid-step, resume later) rather than the cgroup's identity; see
/// the "See also" section below for the full Op-variants list.
///
/// All builders are additive — a `CgroupDef` accumulates an
/// optional [`CpuLimits`] / [`MemoryLimits`] / [`IoLimits`] /
/// [`PidsLimits`] block. When a block is set (e.g. `def.memory`
/// is `Some`), **all** knobs in that block are written —
/// `None`-valued fields emit their kernel-default sentinel
/// (`"max"` for `memory.max`/`memory.high`, `"0"` for
/// `memory.low`). Only `memory.swap.max` is gated: `None` means
/// no write (for `CONFIG_SWAP=n` compatibility). The "*_unlimited"
/// builders explicitly rewind a knob to its sentinel value
/// (`"max"` / `"0"`) so a base `CgroupDef` factory can cap a
/// resource and a per-test extension can clear that cap without
/// rewriting the whole `CgroupDef`.
///
/// Validation runs at `apply_setup` time (before any worker
/// spawn): out-of-range weights, `cpu.max period == 0`, and
/// `pids.max == Some(0)` all produce actionable bails before the
/// syscall fires. The kernel is the final authority on
/// per-controller numeric ranges; framework-level checks catch
/// only the foot-cannons documented per-builder.
///
/// # Builder semantics
///
/// The setters fall into three groups:
///
/// **Group I — per-WorkSpec fan to `works[0]`:**
/// [`workers`](Self::workers), [`workers_pct`](Self::workers_pct),
/// [`work_type`](Self::work_type), [`sched_policy`](Self::sched_policy),
/// [`affinity`](Self::affinity), [`mem_policy`](Self::mem_policy),
/// [`mpol_flags`](Self::mpol_flags). Each mutates `self.works[0]`,
/// auto-inserting a default [`WorkSpec`] when `works` is empty. There
/// is NO cgroup-level default for these knobs — per-group identity (or
/// per-group cpuset validation) makes fan-out semantically ambiguous.
/// Use [`work`](Self::work) + per-`WorkSpec` setters for multi-group
/// cgroups.
///
/// **Group II — cgroup-level `default_*` merge:**
/// [`nice`](Self::nice), [`comm`](Self::comm), [`uid`](Self::uid),
/// [`gid`](Self::gid), [`numa_node`](Self::numa_node). Each stores a
/// value in a `default_*` field on `CgroupDef`. Every [`WorkSpec`] in
/// [`works`](Self::works) whose corresponding `Option`-typed field is
/// `None` inherits the default at [`merged_works`](Self::merged_works)
/// time — ORDER-INDEPENDENT with [`work`](Self::work). `Some(_)`
/// (including `Some(0)`) opts out.
///
/// **Group III — [`pcomm`](Self::pcomm):** mutates `works` in-place
/// at call time, NOT order-independent — by design. See
/// [`pcomm`](Self::pcomm) for the coalescing rationale.
///
/// Other setters ([`cpuset`](Self::cpuset),
/// [`cpuset_mems`](Self::cpuset_mems), the
/// [`cpu_quota`](Self::cpu_quota) / [`memory_max`](Self::memory_max)
/// / [`io_weight`](Self::io_weight) / [`pids_max`](Self::pids_max)
/// controller families, [`workload`](Self::workload),
/// [`swappable`](Self::swappable)) set cgroup-level state directly
/// and do not participate in either merge pattern.
///
/// # See also
///
/// `CgroupDef` only expresses the steady-state shape of a cgroup
/// (name, cpuset, work groups, payload). State changes that need
/// to happen DURING a step — without tearing the cgroup down and
/// recreating it — go through dedicated [`Op`] variants instead:
///
/// * [`Op::FreezeCgroup`] / [`Op::UnfreezeCgroup`] — pause and
///   resume every task in the cgroup via `cgroup.freeze` (the
///   kernel-side asynchronous freeze path; not a SIGSTOP).
///   Useful for scheduler suspend/resume tests that observe
///   how the scheduler handles a workload that goes idle
///   mid-step. **Do not freeze a cgroup hosting the test's own
///   observers** — see the deadlock warning on
///   [`Op::FreezeCgroup`].
/// * [`Op::SetCpuset`] — re-pin an existing cgroup's cpuset to
///   exercise the scheduler's response to a moving CPU mask
///   without disrupting the worker tasks themselves.
/// * [`Op::AddCgroup`] / [`Op::RemoveCgroup`] — add or destroy
///   cgroups mid-step when a `CgroupDef`'s lifecycle is
///   tied to step duration but the test wants a different
///   (e.g. nested) cgroup to appear or disappear partway
///   through.
///
/// These describe transitions over time rather than the cgroup's
/// identity, which is why they live as `Op` variants alongside
/// the rest of the operation vocabulary rather than as
/// `CgroupDef` builders.
///
/// ```
/// # use ktstr::scenario::ops::{CgroupDef, CpusetSpec};
/// # use ktstr::workload::{WorkSpec, WorkType};
/// // Single work group via convenience methods.
/// let def = CgroupDef::named("workers")
///     .cpuset(CpusetSpec::disjoint(0, 2))
///     .workers(4)
///     .work_type(WorkType::SpinWait);
///
/// assert_eq!(def.name, "workers");
/// assert_eq!(def.works[0].num_workers, Some(4));
///
/// // Multiple concurrent work groups via .work().
/// let def = CgroupDef::named("mixed")
///     .work(WorkSpec::default().workers(4).work_type(WorkType::SpinWait))
///     .work(WorkSpec::default().workers(2).work_type(WorkType::YieldHeavy));
///
/// assert_eq!(def.works.len(), 2);
///
/// // Synthetic work + userspace binary side-by-side via .workload(&X).
/// // The binary runs inside the same cgroup as the WorkSpec handles;
/// // both spawn in apply_setup, the WorkSpec groups first, then the
/// // Payload after the cpuset settles.
/// # use ktstr::test_support::{OutputFormat, Payload, PayloadKind};
/// # const BENCH: Payload = Payload {
/// #     name: "bench",
/// #     kind: PayloadKind::Binary("bench"),
/// #     output: OutputFormat::ExitCode,
/// #     default_args: &[],
/// #     default_checks: &[],
/// #     metrics: &[],
/// #     include_files: &[],
/// #     uses_parent_pgrp: false,
/// #     known_flags: None,
/// # };
/// let def = CgroupDef::named("io_and_spin")
///     .cpuset(CpusetSpec::disjoint(0, 2))
///     .workers(2)
///     .work_type(WorkType::SpinWait)
///     .workload(&BENCH);
///
/// assert!(def.payload.is_some());
/// assert_eq!(def.works[0].num_workers, Some(2));
/// ```
#[derive(Clone, Debug)]
pub struct CgroupDef {
    /// Cgroup name relative to the scenario's parent cgroup. Must be a
    /// valid cgroupfs filename.
    pub name: Cow<'static, str>,
    /// Optional cpuset assignment. `None` inherits the parent cgroup's
    /// cpuset (typically the scenario's usable CPU set).
    pub cpuset: Option<CpusetSpec>,
    /// WorkSpec groups to spawn. Empty means use a single default WorkSpec
    /// (SpinWait, Normal, `ctx.workers_per_cgroup` workers — defaults to 1
    /// from `CtxBuilder` unless the scenario overrides it explicitly).
    pub works: Vec<WorkSpec>,
    /// When true, the gauntlet work_type override replaces each WorkSpec's
    /// work_type (applied per-WorkSpec via resolve_work_type).
    pub swappable: bool,
    /// Optional userspace [`Payload`](crate::test_support::Payload) to
    /// launch inside this cgroup.
    ///
    /// **Spawn order within `apply_setup`**: the cgroup is created
    /// (`add_cgroup_no_cpuset`), its cpuset is resolved + set, then
    /// each `WorkSpec` entry is spawned and moved into the cgroup in
    /// declaration order, and finally — after every synthetic
    /// `WorkSpec` handle has started — the `Payload` is spawned via
    /// `PayloadRun::new(ctx, p).in_cgroup(name).spawn()`. This
    /// fixed order lets the cgroup cpuset and mempolicy settle on
    /// the `WorkSpec` handles before the binary inherits placement, so
    /// the binary sees a stable topology. Once spawned, all three
    /// (cgroup, works, payload) run concurrently until teardown.
    ///
    /// Only
    /// [`PayloadKind::Binary`](crate::test_support::PayloadKind::Binary)
    /// payloads are accepted — scheduler-kind payloads are rejected
    /// at construction time via [`Self::workload`]. The payload is
    /// killed at step-teardown (before cgroup removal) so the cgroup
    /// removal does not fail with EBUSY.
    pub payload: Option<&'static crate::test_support::Payload>,
    /// Optional cpuset.mems NUMA node binding. `None` inherits the
    /// parent cgroup's `cpuset.mems`. Set via
    /// [`Self::cpuset_mems`].
    pub cpuset_mems: Option<BTreeSet<usize>>,
    /// Optional cpu controller limits (`cpu.max`, `cpu.weight`).
    /// `None` leaves both kernel defaults in place. Set via
    /// [`Self::cpu_quota_pct`] / [`Self::cpu_quota`] /
    /// [`Self::cpu_weight`].
    pub cpu: Option<CpuLimits>,
    /// Optional memory controller limits (`memory.max`,
    /// `memory.high`, `memory.low`, `memory.swap.max`). `None`
    /// leaves all four at the kernel defaults. Set via
    /// [`Self::memory_max`] / [`Self::memory_high`] /
    /// [`Self::memory_low`] / [`Self::memory_swap_max`].
    pub memory: Option<MemoryLimits>,
    /// Optional io controller limits (`io.weight`). `None` leaves
    /// the kernel default in place. Set via [`Self::io_weight`].
    pub io: Option<IoLimits>,
    /// Optional pids controller limits (`pids.max`). `None` leaves
    /// the kernel default in place (no ceiling). Set via
    /// [`Self::pids_max`].
    pub pids: Option<PidsLimits>,
    /// Cgroup-level default for [`WorkSpec::nice`]. When `Some(n)`,
    /// every [`WorkSpec`] in [`Self::works`] whose own `nice` field
    /// is `None` (the framework's "skip setpriority(2)" state — see
    /// [`WorkloadConfig::nice`](crate::workload::WorkloadConfig::nice))
    /// inherits `Some(n)` at apply-setup time. Set via [`Self::nice`];
    /// merged in [`Self::merged_works`].
    ///
    /// Order-independent with [`Self::work`]: `def.work(spec).nice(n)`
    /// and `def.nice(n).work(spec)` produce identical effective
    /// `WorkSpec` values because the merge runs at `merged_works()`
    /// call time, not at builder-method call time.
    pub default_nice: Option<i32>,
    /// Cgroup-level default for [`WorkSpec::comm`]. Merged into any
    /// [`WorkSpec`] whose own `comm` is `None` at apply-setup time.
    /// Set via [`Self::comm`]; merged in [`Self::merged_works`].
    pub default_comm: Option<Cow<'static, str>>,
    /// Cgroup-level default for [`WorkSpec::uid`]. Merged into any
    /// [`WorkSpec`] whose own `uid` is `None` at apply-setup time.
    /// Set via [`Self::uid`]; merged in [`Self::merged_works`].
    pub default_uid: Option<u32>,
    /// Cgroup-level default for [`WorkSpec::gid`]. Merged into any
    /// [`WorkSpec`] whose own `gid` is `None` at apply-setup time.
    /// Set via [`Self::gid`]; merged in [`Self::merged_works`].
    pub default_gid: Option<u32>,
    /// Cgroup-level default for [`WorkSpec::numa_node`]. Merged into
    /// any [`WorkSpec`] whose own `numa_node` is `None` at
    /// apply-setup time. Set via [`Self::numa_node`]; merged in
    /// [`Self::merged_works`].
    pub default_numa_node: Option<u32>,
}

impl CgroupDef {
    /// Create a CgroupDef with defaults (empty works, no cpuset).
    ///
    /// `apply_setup` fills an empty `works` slice with one default
    /// [`WorkSpec`] (SpinWait, SCHED_NORMAL, `ctx.workers_per_cgroup`
    /// workers — defaults to 1 from `CtxBuilder`). For an empty
    /// move-target cgroup with no workers, declare it via
    /// [`Op::AddCgroup`] at step or Backdrop level. For the common
    /// `CgroupDef::named(name).workers(ctx.workers_per_cgroup)`
    /// pattern use [`Ctx::cgroup_def`](crate::scenario::Ctx::cgroup_def).
    #[must_use = "dropping a CgroupDef discards the cgroup specification"]
    pub fn named(name: impl Into<Cow<'static, str>>) -> Self {
        Self {
            name: name.into(),
            cpuset: None,
            works: vec![],
            swappable: false,
            payload: None,
            cpuset_mems: None,
            cpu: None,
            memory: None,
            io: None,
            pids: None,
            default_nice: None,
            default_comm: None,
            default_uid: None,
            default_gid: None,
            default_numa_node: None,
        }
    }

    /// Set [`Self::cpuset`]; see [`Op::SetCpuset`] for mid-run changes.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn cpuset(mut self, cpus: CpusetSpec) -> Self {
        self.cpuset = Some(cpus);
        self
    }

    /// Append a [`WorkSpec`] group (multiple calls yield concurrent groups within this cgroup).
    #[must_use = "builder methods consume self; bind the result"]
    pub fn work(mut self, w: WorkSpec) -> Self {
        self.works.push(w);
        self
    }

    /// Ensure `works[0]` exists for single-WorkSpec builder methods.
    fn ensure_default_work(&mut self) {
        if self.works.is_empty() {
            self.works.push(WorkSpec::default());
        }
    }

    /// Set [`WorkSpec::num_workers`] on `works[0]` (Group I).
    ///
    /// `n` MUST be `>= 1`. `n == 0` is rejected at
    /// [`WorkloadConfig::validate`] time with an actionable
    /// diagnostic — a zero-worker spawn would silently produce
    /// no workload load, vacuously passing scheduler assertions
    /// that rely on observable contention. Pass `n >= 1`; for
    /// fraction-of-cpuset sizing use [`Self::workers_pct`].
    #[must_use = "builder methods consume self; bind the result"]
    pub fn workers(mut self, n: usize) -> Self {
        self.ensure_default_work();
        self.works[0].num_workers = Some(n);
        self
    }

    /// Set [`WorkSpec::workers_pct`] on `works[0]` (Group I). Resolved
    /// against the cgroup's cpuset at apply-setup via
    /// `ceil(cpuset_cpus * pct)`. Mutually exclusive with
    /// [`Self::workers`] — see [`WorkSpec::workers_pct`].
    ///
    /// # Panics
    ///
    /// Panics when `pct` is NaN, infinite, or `<= 0.0`. Extreme
    /// finite values (e.g. `1e100`) pass the gate and saturate to
    /// `usize::MAX` via the `as` cast in `resolve_workers_pct`
    /// (RFC 2484 / Rust 1.45+) — attempting to spawn that many
    /// workers would OOM the host. Keep `pct` near the intended
    /// oversubscription factor (e.g. `1.0`, `2.0`, `4.0`).
    #[must_use = "builder methods consume self; bind the result"]
    pub fn workers_pct(mut self, pct: f64) -> Self {
        assert!(
            pct.is_finite() && pct > 0.0,
            "CgroupDef::workers_pct({pct}): pct must be finite and > 0.0",
        );
        self.ensure_default_work();
        self.works[0].workers_pct = Some(pct);
        self
    }

    /// Set [`WorkSpec::work_type`] on `works[0]` (Group I).
    #[must_use = "builder methods consume self; bind the result"]
    pub fn work_type(mut self, wt: WorkType) -> Self {
        self.ensure_default_work();
        self.works[0].work_type = wt;
        self
    }

    /// Set [`WorkSpec::sched_policy`] on `works[0]` (Group I).
    #[must_use = "builder methods consume self; bind the result"]
    pub fn sched_policy(mut self, p: crate::workload::SchedPolicy) -> Self {
        self.ensure_default_work();
        self.works[0].sched_policy = p;
        self
    }

    /// Set [`WorkSpec::affinity`] on `works[0]` (Group I).
    #[must_use = "builder methods consume self; bind the result"]
    pub fn affinity(mut self, a: crate::workload::AffinityIntent) -> Self {
        self.ensure_default_work();
        self.works[0].affinity = a;
        self
    }

    /// Set [`WorkSpec::mem_policy`] on `works[0]` (Group I). Validated
    /// against the resolved cpuset per-group.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn mem_policy(mut self, p: crate::workload::MemPolicy) -> Self {
        self.ensure_default_work();
        self.works[0].mem_policy = p;
        self
    }

    /// Set [`WorkSpec::mpol_flags`] on `works[0]` (Group I).
    #[must_use = "builder methods consume self; bind the result"]
    pub fn mpol_flags(mut self, f: crate::workload::MpolFlags) -> Self {
        self.ensure_default_work();
        self.works[0].mpol_flags = f;
        self
    }

    /// Set [`Self::default_nice`] (Group II). Note: `WorkSpec::nice(0)`
    /// stores `Some(0)` and opts out of this default — the worker's
    /// nice is explicitly set to 0 via `setpriority(2)` rather than
    /// inheriting.
    #[must_use = "builder methods consume self; bind the result"]
    pub const fn nice(mut self, n: i32) -> Self {
        self.default_nice = Some(n);
        self
    }

    /// Set [`Self::default_comm`] (Group II).
    ///
    /// # Panics
    ///
    /// Panics on programmer-error inputs — mirrors
    /// [`crate::workload::WorkSpec::pcomm`]'s `# Panics`:
    /// - Empty string.
    /// - Interior NUL byte.
    /// - More than 15 bytes (`TASK_COMM_LEN - 1` cap).
    ///
    /// See
    /// [`validate_task_comm_string`](crate::workload::validate_task_comm_string)
    /// for the centralized rationale; `name.len()` is the BYTE
    /// length (UTF-8 multi-byte chars count as their byte width).
    #[must_use = "builder methods consume self; bind the result"]
    pub fn comm(mut self, name: impl Into<std::borrow::Cow<'static, str>>) -> Self {
        let name: std::borrow::Cow<'static, str> = name.into();
        crate::workload::validate_task_comm_string("CgroupDef::comm", &name);
        self.default_comm = Some(name);
        self
    }

    /// Set [`Self::default_uid`] (Group II).
    #[must_use = "builder methods consume self; bind the result"]
    pub const fn uid(mut self, uid: u32) -> Self {
        self.default_uid = Some(uid);
        self
    }

    /// Set [`Self::default_gid`] (Group II).
    #[must_use = "builder methods consume self; bind the result"]
    pub const fn gid(mut self, gid: u32) -> Self {
        self.default_gid = Some(gid);
        self
    }

    /// Set the thread-group leader's comm on every WorkSpec in
    /// this CgroupDef. Each affected WorkSpec gets `pcomm =
    /// Some(name)`; existing per-WorkSpec `pcomm` values (set
    /// before this call) are overwritten. Calling on an empty
    /// `works` list pushes a default WorkSpec carrying the value.
    ///
    /// The pcomm string is applied via `prctl(PR_SET_NAME)` on
    /// the forked thread-group leader. The builder rejects > 15
    /// bytes (TASK_COMM_LEN-1) at construction so the
    /// `task->group_leader->comm == pcomm` invariant the framework
    /// relies on holds exactly.
    /// Setting this triggers the fork-then-thread spawn path in
    /// `apply_setup`: WorkSpecs sharing a `pcomm` value coalesce
    /// into ONE thread-group leader per group; every worker
    /// thread inside observes `task->group_leader->comm == pcomm`.
    /// Each worker thread additionally sets its own `task->comm`
    /// via `.comm()` on the per-WorkSpec [`WorkSpec::comm`] at
    /// thread creation time.
    ///
    /// `pcomm` lives ONLY on [`WorkSpec`] — there is no
    /// CgroupDef-level field. This builder writes the value into
    /// every WorkSpec directly so `apply_setup` has a single
    /// authoritative source per WorkSpec.
    ///
    /// **Not order-independent with [`Self::work`] — by design.**
    /// Unlike Group II setters, `pcomm` mutates `works` in-place when
    /// called: it stamps every WorkSpec that EXISTS at call time and
    /// then returns. WorkSpecs added via subsequent [`Self::work`]
    /// calls are not retroactively touched, and a WorkSpec that
    /// already carried its own `pcomm` is OVERWRITTEN if it was
    /// pushed before `.pcomm(..)` ran. This is intentional — `pcomm`
    /// determines the thread-group leader's coalescing key in
    /// `apply_setup`, so the framework needs the value baked onto
    /// each WorkSpec by the time `merged_works()` runs. Storing it
    /// as a default and merging at read time would break the
    /// coalescing contract for the empty-works case (the synthesised
    /// `WorkSpec::default()` would have to carry the pcomm without
    /// distinguishing "default" from "explicit override").
    ///
    /// # Panics
    ///
    /// Panics on programmer-error inputs — mirrors
    /// [`crate::workload::WorkSpec::pcomm`]'s `# Panics`:
    /// - Empty string.
    /// - Interior NUL byte.
    /// - More than 15 bytes (`TASK_COMM_LEN - 1` cap).
    ///
    /// See
    /// [`validate_task_comm_string`](crate::workload::validate_task_comm_string)
    /// for the centralized rationale; `name.len()` is the BYTE
    /// length (UTF-8 multi-byte chars count as their byte width).
    #[must_use = "builder methods consume self; bind the result"]
    pub fn pcomm(mut self, name: impl Into<Cow<'static, str>>) -> Self {
        let name: Cow<'static, str> = name.into();
        // Validate ONCE before the in-place loop so a bad input
        // never partially writes the `works` vec — the per-builder
        // assert fires with a single named site rather than firing
        // mid-mutation across N entries.
        crate::workload::validate_task_comm_string("CgroupDef::pcomm", &name);
        if self.works.is_empty() {
            self.works.push(WorkSpec::default());
        }
        for w in &mut self.works {
            w.pcomm = Some(name.clone());
        }
        self
    }

    /// Set [`Self::default_numa_node`] (Group II).
    #[must_use = "builder methods consume self; bind the result"]
    pub const fn numa_node(mut self, node: u32) -> Self {
        self.default_numa_node = Some(node);
        self
    }

    /// Set [`Self::swappable`] (gauntlet work_type override).
    #[must_use = "builder methods consume self; bind the result"]
    pub const fn swappable(mut self, swappable: bool) -> Self {
        self.swappable = swappable;
        self
    }

    /// Attach a userspace payload binary that runs inside this cgroup
    /// alongside any synthetic [`WorkSpec`] groups. The payload spawns
    /// when the step enters `apply_setup` and is killed during
    /// step-teardown so the cgroup can be removed cleanly.
    ///
    /// # Panics
    ///
    /// Panics when `p.is_scheduler()` (i.e. `p` is a scheduler-kind
    /// [`Payload`](crate::test_support::Payload) — `KERNEL_DEFAULT`
    /// or any other `PayloadKind::Scheduler*` variant). Only
    /// [`PayloadKind::Binary`](crate::test_support::PayloadKind::Binary)
    /// payloads are accepted; `CgroupDef.workload` is for userspace
    /// binary payloads only, and scheduler placement uses
    /// `#[ktstr_test(scheduler = ...)]` instead.
    ///
    /// **Why panic at declaration time, not at spawn time?** Three
    /// reasons, all of which favor failing fast:
    /// 1. **Discovery-time surfacing.** `CgroupDef` builders run
    ///    during test construction, which nextest's `--list`
    ///    invocation reaches BEFORE any VM boot. A panic here
    ///    emits a full backtrace inside the test binary and
    ///    surfaces the offending call site immediately; a deferred
    ///    runtime error would require a KVM-capable host + a
    ///    kernel image + an initramfs build to observe — a 30+
    ///    second feedback loop for what is purely a
    ///    typed-API misuse.
    /// 2. **No side effects.** The panic happens before
    ///    `CgroupDef.payload = Some(p)` assignment runs, so the
    ///    in-progress builder is left in its prior (no-payload)
    ///    state. A caller that catches the panic via
    ///    `catch_unwind` sees a valid CgroupDef either way.
    /// 3. **Scheduler-kind is always a programming error here.**
    ///    `Payload::KERNEL_DEFAULT` in `CgroupDef::workload` is never a
    ///    legitimate use case — it means the author confused the
    ///    `scheduler` slot (test-level) with the `workload` slot
    ///    (cgroup-level). There is no recovery path; the only
    ///    resolution is editing the source.
    ///
    /// Scheduler-kind payloads in the step-level `Op::RunPayload`
    /// path bail with an `anyhow::Error` instead of panicking —
    /// that path runs during scenario execution where one bad op
    /// should not crash a whole test run.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn workload(mut self, p: &'static crate::test_support::Payload) -> Self {
        assert!(
            !p.is_scheduler(),
            "CgroupDef::workload called with a scheduler-kind Payload ({}); \
             CgroupDef.workload is for userspace binary payloads only. \
             Use #[ktstr_test(scheduler = ...)] for scheduler placement.",
            p.name,
        );
        self.payload = Some(p);
        self
    }

    /// Bind `cpuset.mems` for this cgroup. Mirrors [`Self::cpuset`]
    /// for NUMA memory placement: the cgroup's tasks may only
    /// allocate memory on the listed NUMA nodes. `None` (default)
    /// inherits the parent's `cpuset.mems`.
    ///
    /// Required when the cgroup spans CPUs on a NUMA node whose
    /// memory is NOT in the parent's `cpuset.mems` — allocations
    /// from the cgroup's tasks are constrained to the parent's
    /// allowed nodes per kernel/cgroup/cpuset.c. The framework
    /// writes `cpuset.mems` immediately after `cpuset.cpus` so the
    /// binding is in effect before any worker is moved in.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn cpuset_mems(mut self, nodes: BTreeSet<usize>) -> Self {
        self.cpuset_mems = Some(nodes);
        self
    }

    /// Set `cpu.max` quota as a percentage of one CPU's
    /// throughput, with a default 100 ms `period`. `100` means
    /// "one full CPU" (quota=100_000, period=100_000); `200` means
    /// "two CPUs". Use [`Self::cpu_quota`] for non-default periods.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn cpu_quota_pct(mut self, pct: u32) -> Self {
        let cpu = self.cpu.get_or_insert_with(CpuLimits::default);
        cpu.max_period_us = 100_000;
        cpu.max_quota_us = Some((pct as u64) * 1_000);
        self
    }

    /// Set `cpu.max` quota and period directly. `quota` may exceed
    /// `period` (multi-CPU concurrency, see [`CpuLimits::max_quota_us`]).
    /// Both arguments are converted to microseconds; sub-microsecond
    /// fractions in the supplied [`Duration`]s are truncated.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn cpu_quota(mut self, quota: Duration, period: Duration) -> Self {
        let cpu = self.cpu.get_or_insert_with(CpuLimits::default);
        cpu.max_quota_us = Some(quota.as_micros() as u64);
        cpu.max_period_us = period.as_micros() as u64;
        self
    }

    /// Clear any previously-set `cpu.max` quota (writes `"max"`),
    /// leaving `cpu.weight` (if set) intact. Useful when a base
    /// CgroupDef builder applied a default cap and the test wants
    /// only weight-based bias.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn cpu_unlimited(mut self) -> Self {
        let cpu = self.cpu.get_or_insert_with(CpuLimits::default);
        cpu.max_quota_us = None;
        self
    }

    /// Set `cpu.weight` (`CGROUP_WEIGHT_MIN..=CGROUP_WEIGHT_MAX`,
    /// 1..=10000; `CGROUP_WEIGHT_DFL` = 100; enforced by
    /// `cpu_weight_write_u64` in kernel/sched/core.c). Larger values
    /// get a larger share under contention. Independent of `cpu.max`.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn cpu_weight(mut self, weight: u32) -> Self {
        let cpu = self.cpu.get_or_insert_with(CpuLimits::default);
        cpu.weight = Some(weight);
        self
    }

    /// Set `memory.max` hard ceiling in bytes. Crossing this
    /// triggers reclaim first (`try_charge_memcg` in
    /// mm/memcontrol.c); the cgroup OOM killer fires only after
    /// `MAX_RECLAIM_RETRIES` failed retries, and is skipped when
    /// the allocation carries `__GFP_NORETRY` or
    /// `__GFP_RETRY_MAYFAIL`.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn memory_max(mut self, bytes: u64) -> Self {
        let m = self.memory.get_or_insert_with(MemoryLimits::default);
        m.max = Some(bytes);
        self
    }

    /// Set `memory.high` soft throttle threshold in bytes. Crossing
    /// this triggers reclaim throttling but NOT OOM-kill — per
    /// `__mem_cgroup_handle_over_high` in mm/memcontrol.c:
    /// "memory.high enforcement isn't as strict, and there is no
    /// OOM killer involved".
    #[must_use = "builder methods consume self; bind the result"]
    pub fn memory_high(mut self, bytes: u64) -> Self {
        let m = self.memory.get_or_insert_with(MemoryLimits::default);
        m.high = Some(bytes);
        self
    }

    /// Set `memory.low` soft protection threshold in bytes.
    /// Reclaim prefers other cgroups before this one's memory
    /// drops below `low`.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn memory_low(mut self, bytes: u64) -> Self {
        let m = self.memory.get_or_insert_with(MemoryLimits::default);
        m.low = Some(bytes);
        self
    }

    /// Clear all three memory limits (writes `"max"` for max/high
    /// and `"0"` for low). Equivalent to leaving `memory` unset
    /// at construction; provided for symmetry with
    /// [`Self::cpu_unlimited`].
    #[must_use = "builder methods consume self; bind the result"]
    pub fn memory_unlimited(mut self) -> Self {
        self.memory = Some(MemoryLimits::default());
        self
    }

    /// Set `io.weight` (`CGROUP_WEIGHT_MIN..=CGROUP_WEIGHT_MAX`,
    /// 1..=10000; `CGROUP_WEIGHT_DFL` = 100; enforced by
    /// `ioc_weight_write` in block/blk-iocost.c). Biases relative
    /// IO share when the io controller is enabled. `io.max`
    /// per-device caps are not surfaced — see [`IoLimits`].
    #[must_use = "builder methods consume self; bind the result"]
    pub fn io_weight(mut self, weight: u16) -> Self {
        let io = self.io.get_or_insert_with(IoLimits::default);
        io.weight = Some(weight);
        self
    }

    /// Set `memory.swap.max` ceiling in bytes. The kernel parses the
    /// wire value via `page_counter_memparse` and accepts a decimal
    /// byte count (`swap_max_write` in `mm/memcontrol.c`). Distinct
    /// from `memory.max`: this caps how much of the cgroup's memory
    /// can spill to swap, separate from total memory consumption.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn memory_swap_max(mut self, bytes: u64) -> Self {
        let m = self.memory.get_or_insert_with(MemoryLimits::default);
        m.swap_max = Some(bytes);
        self
    }

    /// Clear any previously-set `memory.swap.max` (writes `"max"`).
    /// Mirrors [`Self::cpu_unlimited`] / [`Self::memory_unlimited`]
    /// for a single memory-knob unset; useful when a base
    /// `CgroupDef` builder applied a swap cap and the test wants to
    /// remove only that knob while preserving `memory.max`/`high`/
    /// `low`.
    ///
    /// No-ops when `self.memory == None` — the default state already
    /// means "no swap cap" (apply_setup emits no memory writes for an
    /// unset `memory` field), so creating a fresh `MemoryLimits` just
    /// to set `swap_max = None` would (a) be redundant and (b)
    /// trigger 3 unwanted writes for `memory.max` / `memory.high` /
    /// `memory.low` at apply_setup time. The no-op short-circuit
    /// keeps "fresh CgroupDef + memory_swap_unlimited()" semantically
    /// identical to "fresh CgroupDef".
    #[must_use = "builder methods consume self; bind the result"]
    pub fn memory_swap_unlimited(mut self) -> Self {
        if let Some(m) = self.memory.as_mut() {
            m.swap_max = None;
        }
        self
    }

    /// Set `pids.max` task-count ceiling. `n` is the maximum number
    /// of processes the cgroup may host before subsequent
    /// `fork()` / `clone()` calls return EAGAIN. Existing tasks are
    /// NOT killed when the limit lands below the current count
    /// (per the `pids_max_write` kernel comment: "Limit updates
    /// don't need to be mutex'd, since it isn't critical that any
    /// racing fork()s follow the new limit").
    ///
    /// `n = 0` is rejected at `apply_setup` time: a 0-limit cgroup
    /// halts every fork/clone inside, including the worker spawn
    /// under `CloneMode::Fork` and the `ForkExit` per-iteration
    /// child fork. There is no kernel sentinel for "no fork ever";
    /// `pids_max=0` silently fails every `fork()` inside with
    /// `EAGAIN`, which is almost certainly a configuration bug.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn pids_max(mut self, n: u64) -> Self {
        let pids = self.pids.get_or_insert_with(PidsLimits::default);
        pids.max = Some(n);
        self
    }

    /// Clear any previously-set `pids.max` (writes `"max"`).
    /// Mirrors [`Self::cpu_unlimited`] / [`Self::memory_unlimited`].
    #[must_use = "builder methods consume self; bind the result"]
    pub fn pids_unlimited(mut self) -> Self {
        let pids = self.pids.get_or_insert_with(PidsLimits::default);
        pids.max = None;
        self
    }

    /// Materialize [`Self::works`] with cgroup-level defaults
    /// merged into each entry. Called by `apply_setup` to resolve
    /// the per-WorkSpec values before spawning workers.
    ///
    /// For every [`WorkSpec`] in [`Self::works`] (or a single
    /// [`WorkSpec::default()`] when `works` is empty, matching
    /// `apply_setup`'s default-substitution rule), each cgroup-level
    /// default in [`Self::default_nice`] / [`Self::default_comm`] /
    /// [`Self::default_uid`] / [`Self::default_gid`] /
    /// [`Self::default_numa_node`] fills the corresponding
    /// `WorkSpec` field when that field is "unset" at the WorkSpec
    /// level.
    ///
    /// "Unset" means `None` for every `Option`-typed field —
    /// `nice`, `comm`, `uid`, `gid`, `numa_node` are all
    /// `Option<_>`. The framework's "skip setpriority(2)" state per
    /// [`WorkloadConfig::nice`](crate::workload::WorkloadConfig::nice)
    /// is `None`. A `WorkSpec` that explicitly sets `Some(n)`
    /// (including `Some(0)`) keeps its value; the cgroup-level
    /// default applies only when the WorkSpec is at the framework
    /// default of `None`.
    ///
    /// `pcomm` is NOT propagated through `merged_works`. The
    /// [`Self::pcomm`] convenience method writes `pcomm` directly
    /// into every WorkSpec at builder time so coalescing in
    /// `apply_setup` reads the per-WorkSpec value (the
    /// authoritative source).
    ///
    /// Decoupling this merge from the convenience-method call sites
    /// makes the builder order-independent —
    /// `def.nice(5).work(spec)` and `def.work(spec).nice(5)`
    /// produce identical effective `WorkSpec` values.
    pub fn merged_works(&self) -> Vec<WorkSpec> {
        let base: Vec<WorkSpec> = if self.works.is_empty() {
            vec![WorkSpec::default()]
        } else {
            self.works.clone()
        };
        base.into_iter()
            .map(|mut w| {
                if w.nice.is_none()
                    && let Some(n) = self.default_nice
                {
                    w.nice = Some(n);
                }
                if w.comm.is_none() {
                    w.comm = self.default_comm.clone();
                }
                if w.uid.is_none() {
                    w.uid = self.default_uid;
                }
                if w.gid.is_none() {
                    w.gid = self.default_gid;
                }
                if w.numa_node.is_none() {
                    w.numa_node = self.default_numa_node;
                }
                w
            })
            .collect()
    }
}

// `CgroupDef` deliberately has NO `Default` impl. The previous
// derived/hand-rolled default produced `name = "cg_0"`, which
// collides with the conventional first cgroup name in nearly every
// scenario (a test calling `..Default::default()` would silently
// share a cgroup with the scenario's first named entry). Forcing
// every construction site to go through [`CgroupDef::named`] makes
// the name explicit and eliminates the footgun. The pattern is
// documented in the type-level docstring and operator-facing
// guidance at `doc/guide/src/architecture/workload-handle.md` under
// the spread-default warning. The compile-time pin of the absence
// lives in the `#[cfg(test)]` mod below (`assert_not_impl_default!`
// from `src/test_macros.rs`).

#[cfg(test)]
mod tests {
    use super::*;

    assert_not_impl_default!(CgroupDef);

    #[test]
    #[should_panic(expected = "CgroupDef::comm: empty string rejected")]
    fn cgroup_def_comm_rejects_empty() {
        let _ = CgroupDef::named("cg").comm("");
    }

    #[test]
    #[should_panic(expected = "interior NUL byte")]
    fn cgroup_def_comm_rejects_interior_nul() {
        let _ = CgroupDef::named("cg").comm("foo\0bar");
    }

    /// Pins the validate-on-builder contract: CgroupDef::pcomm
    /// previously wrote `w.pcomm = Some(...)` directly, bypassing
    /// WorkSpec::pcomm's asserts. Both builders now route through a
    /// shared `validate_task_comm_string` helper — this test would
    /// FAIL the pre-helper implementation and PASS the post-helper
    /// implementation.
    #[test]
    #[should_panic(expected = "CgroupDef::pcomm: empty string rejected")]
    fn cgroup_def_pcomm_rejects_empty() {
        let _ = CgroupDef::named("cg").pcomm("");
    }

    #[test]
    #[should_panic(expected = "interior NUL byte")]
    fn cgroup_def_pcomm_rejects_interior_nul() {
        let _ = CgroupDef::named("cg").pcomm("foo\0bar");
    }

    /// Per-builder boundary pins: a future refactor that re-routes
    /// CgroupDef::comm or CgroupDef::pcomm around the shared
    /// `validate_task_comm_string` helper would surface here even if
    /// the helper-level tests still pass.
    #[test]
    fn cgroup_def_comm_accepts_15_byte_boundary() {
        let fifteen = "a".repeat(15);
        let def = CgroupDef::named("cg").comm(fifteen.clone());
        assert_eq!(def.default_comm.as_deref(), Some(fifteen.as_str()));
    }

    #[test]
    #[should_panic(expected = "16 bytes")]
    fn cgroup_def_comm_rejects_16_byte_overflow() {
        let _ = CgroupDef::named("cg").comm("a".repeat(16));
    }

    #[test]
    fn cgroup_def_pcomm_accepts_15_byte_boundary() {
        let fifteen = "a".repeat(15);
        let def = CgroupDef::named("cg").pcomm(fifteen.clone());
        // pcomm stamps every WorkSpec; default works is one entry.
        assert_eq!(def.works.len(), 1);
        assert_eq!(def.works[0].pcomm.as_deref(), Some(fifteen.as_str()));
    }

    #[test]
    #[should_panic(expected = "16 bytes")]
    fn cgroup_def_pcomm_rejects_16_byte_overflow() {
        let _ = CgroupDef::named("cg").pcomm("a".repeat(16));
    }
}
