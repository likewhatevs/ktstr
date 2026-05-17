//! Scenario definitions and test execution.
//!
//! Most tests use the declarative ops API from the [`ops`] submodule:
//! - [`ops::CgroupDef`] -- declarative cgroup definition (name + cpuset + workload)
//! - [`ops::Step`] -- a sequence of ops followed by a hold period
//! - [`ops::Op`] -- atomic cgroup topology operation
//! - [`ops::CpusetSpec`] -- how to compute a cpuset from topology
//! - [`ops::HoldSpec`] -- how long to hold after a step
//! - [`backdrop::Backdrop`] -- persistent scenario state shared across every Step
//! - [`ops::execute_defs`] -- run cgroup definitions for the full duration
//! - [`ops::execute_steps`] -- run a multi-step sequence
//! - [`ops::execute_scenario`] -- run a Backdrop + Steps sequence
//!
//! Types defined in this module:
//! - [`Ctx`] -- runtime context passed to scenario functions
//! - [`CgroupGroup`] -- RAII guard that removes cgroups on drop
//!
//! The [`scenarios`] submodule provides curated canned scenarios.
//!
//! See the [Scenarios](https://likewhatevs.github.io/ktstr/guide/concepts/scenarios.html)
//! and [Writing Tests](https://likewhatevs.github.io/ktstr/guide/writing-tests.html)
//! chapters of the guide.

pub mod affinity;
pub mod backdrop;
pub mod basic;
pub mod cpuset;
pub mod dynamic;
pub mod interaction;
pub mod nested;
pub mod ops;
pub mod payload_run;
pub mod performance;
pub mod sample;
pub mod scenarios;
pub mod snapshot;
pub mod stress;

pub use backdrop::Backdrop;

use std::collections::BTreeSet;
use std::thread;
use std::time::Duration;

use anyhow::Result;

use nix::sys::signal::kill;
use nix::unistd::Pid;

use crate::assert::AssertResult;
use crate::topology::TestTopology;
use crate::workload::*;

/// Check if a process is alive via kill(pid, 0).
///
/// Returns `false` for pid 0: `kill(0, ...)` targets the caller's
/// process group rather than a single process, so the syscall would
/// always report success and falsely mark "no process" as alive.
///
/// Returns `false` for `pid <= 0`. Non-positive pid_t values are
/// invalid targets — `kill(0, ...)` signals the caller's process
/// group and `kill(-1, ...)` signals every process the caller is
/// permitted to signal. Neither matches "is this specific process
/// alive?", so we refuse rather than probe.
///
/// # EPERM: foreign-UID processes report as dead
///
/// `kill(pid, 0)` returns one of three things for `pid > 0`:
///
/// 1. `Ok(())` — pid exists and the caller is permitted to signal it
///    (same UID, or the caller has `CAP_KILL`). This maps to `true`.
/// 2. `Err(ESRCH)` — no process with that pid. Maps to `false`.
/// 3. `Err(EPERM)` — the pid exists but belongs to a different UID
///    (or is otherwise unsignalable by the caller). Per `kill(2)`,
///    "EPERM implies the process exists" — a live process. This
///    implementation treats EPERM as `false` (via `.is_ok()`) because
///    ktstr's callers use `process_alive` to ask "is the scheduler /
///    payload *I launched* still running?", not "does any process
///    with this pid exist?". A foreign-UID process sharing the pid is
///    not the one the caller is tracking and is correctly classified
///    as "no, not *my* process."
///
/// If a future caller needs to distinguish "dead" from "alive but
/// unsignalable," switch to `Errno::ESRCH` discrimination on the
/// `kill` result instead of `.is_ok()` — do NOT change this function
/// silently, because existing callers rely on the EPERM-as-false
/// behavior when walking /proc on heavily-forking hosts where pid
/// reuse can land a foreign-UID process on the old slot.
fn process_alive(pid: libc::pid_t) -> bool {
    if pid <= 0 {
        return false;
    }
    kill(Pid::from_raw(pid), None).is_ok()
}

// Re-export AffinityIntent from workload so existing `use super::*` in
// submodules (affinity.rs, etc.) can find it.
pub use crate::workload::AffinityIntent;

// ---------------------------------------------------------------------------
// RAII cgroup group
// ---------------------------------------------------------------------------

/// RAII guard that removes cgroups on drop.
///
/// Prevents cgroup leaks when workload spawning or other operations fail
/// between cgroup creation and cleanup.
#[must_use = "dropping a CgroupGroup immediately destroys the cgroups it manages"]
pub struct CgroupGroup<'a> {
    cgroups: &'a dyn crate::cgroup::CgroupOps,
    names: Vec<String>,
}

impl std::fmt::Debug for CgroupGroup<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CgroupGroup")
            .field("cgroups", &self.cgroups.parent_path())
            .field("names", &self.names)
            .finish()
    }
}

impl<'a> CgroupGroup<'a> {
    /// Create an empty group. Cgroups added via `add_cgroup` or
    /// `add_cgroup_no_cpuset` are removed when the group is dropped.
    pub fn new(cgroups: &'a dyn crate::cgroup::CgroupOps) -> Self {
        Self {
            cgroups,
            names: Vec::new(),
        }
    }

    /// Create a cgroup and set its cpuset. The cgroup is tracked for cleanup on drop.
    ///
    /// Auto-enables [`Controller::Cpuset`](crate::cgroup::Controller::Cpuset)
    /// on the parent's `cgroup.subtree_control` before creating the
    /// child so the child's `cpuset.cpus` file is exposed and the
    /// subsequent [`set_cpuset`](crate::cgroup::CgroupOps::set_cpuset)
    /// write lands. Direct CgroupGroup users (the `custom_*` scenarios
    /// in [`crate::scenario::nested`] / [`crate::scenario::stress`])
    /// don't go through [`run_scenario`](crate::scenario::ops::execute_steps)'s
    /// controller-resolution hook, so the controller enable has to
    /// happen here. The setup call is idempotent on real cgroupfs (a
    /// `+cpuset` write into `cgroup.subtree_control` that already
    /// contains `cpuset` is a no-op at the kernel level per
    /// `cgroup_subtree_control_write` in kernel/cgroup/cgroup.c).
    pub fn add_cgroup(&mut self, name: &str, cpuset: &BTreeSet<usize>) -> Result<()> {
        let mut required = BTreeSet::new();
        required.insert(crate::cgroup::Controller::Cpuset);
        self.cgroups.setup(&required)?;
        self.cgroups.create_cgroup(name)?;
        self.cgroups.set_cpuset(name, cpuset)?;
        self.names.push(name.to_string());
        Ok(())
    }

    /// Create a cgroup without a cpuset. The cgroup is tracked for cleanup on drop.
    ///
    /// No controller enablement: callers explicitly opting out of a
    /// cpuset signal that they don't need any cgroup v2 controller
    /// surface beyond the cgroup-core knobs (`cgroup.procs`,
    /// `cgroup.freeze`) which are ungated. If a future caller needs
    /// e.g. memory limits on a no-cpuset cgroup, add a
    /// `with_controllers` overload rather than auto-enabling — the
    /// "no-cpuset" name is load-bearing for the absent-controller
    /// behavior pinned by tests in
    /// [`crate::scenario::nested::custom_nested_cgroup_no_ctrl`].
    pub fn add_cgroup_no_cpuset(&mut self, name: &str) -> Result<()> {
        self.cgroups.create_cgroup(name)?;
        self.names.push(name.to_string());
        Ok(())
    }

    /// Names of all tracked cgroups.
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// Forget a tracked cgroup name without touching cgroupfs. Used
    /// by `Op::RemoveCgroup` immediately BEFORE invoking the kernel
    /// rmdir, so a later `Op::AddCgroup` with the same name can
    /// re-create the cgroup instead of colliding against the stale
    /// tracking entry, and the teardown-on-drop path skips a
    /// now-redundant rmdir of a dir that the in-progress (or
    /// already-completed) kernel call is removing.
    pub(crate) fn forget(&mut self, name: &str) {
        self.names.retain(|n| n != name);
    }
}

/// True when `err`'s root cause is an `io::Error` with kind
/// `NotFound` (ENOENT). Used by `CgroupGroup::drop` and
/// `Op::RemoveCgroup` to classify a TOCTOU ENOENT as benign
/// (post-condition "no dir" already holds) so it is filtered
/// from warn output. Extracting the predicate keeps the two
/// sites in lock-step — a classification change only edits
/// this function, not both call sites.
pub(crate) fn is_io_not_found(err: &anyhow::Error) -> bool {
    err.root_cause()
        .downcast_ref::<std::io::Error>()
        .is_some_and(|io| io.kind() == std::io::ErrorKind::NotFound)
}

/// Map a cgroup `remove_cgroup` error's root-cause errno to a
/// short remediation hint appended to warn messages. Only
/// EBUSY and EACCES — the two errnos callers can act on — get
/// specific hints; every other errno yields `None` so the warn
/// stays terse with just the underlying error chain. Extracted
/// so both `CgroupGroup::drop` and `Op::RemoveCgroup` stay
/// synchronized; a new hint (e.g. ENOTEMPTY for un-cleaned
/// children) only needs to be wired here.
pub(crate) fn remove_cgroup_errno_hint(err: &anyhow::Error) -> Option<&'static str> {
    let raw = err
        .root_cause()
        .downcast_ref::<std::io::Error>()?
        .raw_os_error()?;
    match raw {
        libc::EBUSY => {
            Some("EBUSY: cgroup still has live tasks — workloads were not drained before teardown")
        }
        libc::EACCES => {
            Some("EACCES: permission denied — check cgroup owner / `user.slice` delegation")
        }
        _ => None,
    }
}

impl Drop for CgroupGroup<'_> {
    fn drop(&mut self) {
        // Reverse-iterate so nested cgroups (children created AFTER
        // their parents) are removed before their parents. Removing a
        // cgroup directory that still has child cgroup directories
        // under it fails with ENOTEMPTY.
        //
        // ENOENT is expected: `CgroupManager::remove_cgroup` returns
        // Ok when the dir is already gone, so the only way ENOENT
        // reaches here is the narrow TOCTOU race where another process
        // unlinks between `exists()` and `remove_dir` — the post-
        // condition (no dir) still holds and no cleanup is owed. Every
        // other error (EBUSY from a surviving task, EACCES, broken
        // cgroupfs mount) surfaces via `tracing::warn!` so a teardown
        // failure is visible instead of silently swallowed; mirrors
        // the same handling in `Op::RemoveCgroup` so the two paths
        // stay consistent.
        for name in self.names.iter().rev() {
            if let Err(err) = self.cgroups.remove_cgroup(name) {
                if is_io_not_found(&err) {
                    continue;
                }
                let hint = remove_cgroup_errno_hint(&err).unwrap_or("");
                tracing::warn!(
                    cgroup = %name,
                    err = %format!("{err:#}"),
                    hint,
                    "CgroupGroup::drop: remove_cgroup returned non-ENOENT error",
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Runtime context and interpreter
// ---------------------------------------------------------------------------

/// Runtime context passed to scenario functions.
///
/// Provides access to cgroup management, topology information, and
/// test configuration. Custom scenarios (`Action::Custom`) receive
/// this as their sole parameter.
pub struct Ctx<'a> {
    /// Cgroup filesystem operations. `&dyn CgroupOps` (not `&CgroupManager`)
    /// so scenario code can be driven by an in-memory test double without
    /// touching `/sys/fs/cgroup`. Production callers pass
    /// `&CgroupManager` and the auto-coercion is transparent at the call
    /// site — `ctx.cgroups.set_cpuset(...)` works unchanged.
    pub cgroups: &'a dyn crate::cgroup::CgroupOps,
    /// VM CPU topology.
    pub topo: &'a TestTopology,
    /// How long to run the workload.
    pub duration: Duration,
    /// Default number of workers per cgroup.
    pub workers_per_cgroup: usize,
    /// PID of the running scheduler (for liveness checks), or `None`
    /// when no scheduler is attached. Stored as `Option<pid_t>` so
    /// the "no scheduler" state is a distinct variant rather than a
    /// 0-sentinel — `run_scenario` and step-level liveness probes
    /// destructure via `if let Some(pid)` instead of `!= 0` guards.
    pub sched_pid: Option<libc::pid_t>,
    /// Time to wait after cgroup creation for scheduler stabilization.
    pub settle: Duration,
    /// Override work type for scenarios that use `SpinWait` by default.
    pub work_type_override: Option<WorkType>,
    /// Merged assertion config (default_checks + scheduler + per-test).
    /// Used by `run_scenario` for data-driven scenarios and by
    /// `execute_steps` as the default when no explicit checks are
    /// passed to `execute_steps_with`.
    pub assert: crate::assert::Assert,
    /// When true, `execute_steps` polls SHM signal slot 0 after writing
    /// the scenario start marker, blocking until the host confirms its
    /// BPF map write is complete. Set automatically by the framework
    /// when a `KtstrTestEntry` declares `bpf_map_write`; custom
    /// scenarios typically do not flip this manually.
    pub wait_for_map_write: bool,
}

impl std::fmt::Debug for Ctx<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `&dyn CgroupOps` is not Debug (dropped the supertrait to
        // avoid bloating the test-double surface); render the parent
        // path instead so debug prints are still informative.
        f.debug_struct("Ctx")
            .field("cgroups", &self.cgroups.parent_path())
            .field("topo", &self.topo)
            .field("duration", &self.duration)
            .field("workers_per_cgroup", &self.workers_per_cgroup)
            .field("sched_pid", &self.sched_pid)
            .field("settle", &self.settle)
            .field("work_type_override", &self.work_type_override)
            .field("assert", &self.assert)
            .field("wait_for_map_write", &self.wait_for_map_write)
            .finish()
    }
}

impl Ctx<'_> {
    /// Scheduler pid, filtered to the `> 0` range that
    /// [`process_alive`] treats as signalable.
    ///
    /// `Ctx::sched_pid` documents `None` as the "no scheduler
    /// configured" state, and the liveness sites destructure with
    /// `if let Some(pid)`. Nothing in the builder, however, prevents
    /// a caller from passing `Some(0)` or a negative pid — an easy
    /// mistake for callers used to the workload module's internal
    /// 0-sentinel pid slot (see the note on `sched_pid` above — the
    /// sentinel lives on a module-private `AtomicI32` in
    /// `src/workload.rs`, not on this `Option<pid_t>`). A bare
    /// `Some(0)` would reach
    /// `process_alive`, which returns `false` for any pid `<= 0`,
    /// and the liveness sites would then bail with `scheduler died`
    /// even though no scheduler was ever running — a false
    /// positive that turns a misconfiguration into a misleading
    /// scheduler-death diagnostic.
    ///
    /// Centralising the filter here means every liveness callsite
    /// (`run_scenario` post-settle bail, workload-phase polling,
    /// `setup_cgroups` post-settle bail) uses the same predicate:
    /// only a positive pid is "configured". Callers must use this
    /// accessor rather than destructuring `sched_pid` directly.
    ///
    /// A `Some(n)` where `n <= 0` is a caller bug — the builder
    /// documents `None` as the unconfigured shape, and every
    /// positive value flows through unchanged. When the accessor
    /// squashes such a value to `None`, it emits a `tracing::warn!`
    /// naming the offending pid so the misuse surfaces in
    /// structured logs instead of manifesting downstream as a
    /// silent "scheduler died" verdict or, worse, a `kill(0, …)`
    /// reaching the caller's own process group. The warn is
    /// bounded: there are exactly three callsites
    /// (`run_scenario` post-settle bail, workload-phase polling,
    /// `setup_cgroups` post-settle bail), so the volume is O(3)
    /// per scenario run even for a sustained
    /// misconfiguration — tight enough to leave in place without
    /// a rate limiter.
    pub(crate) fn active_sched_pid(&self) -> Option<libc::pid_t> {
        match self.sched_pid {
            Some(p) if p > 0 => Some(p),
            Some(p) => {
                tracing::warn!(
                    pid = p,
                    "Ctx::active_sched_pid: sched_pid=Some({p}) squashed to None; \
                     only positive pids are configured-scheduler values — use \
                     None for the unconfigured shape instead of a 0-sentinel or \
                     negative pid"
                );
                None
            }
            None => None,
        }
    }

    /// Resolve a [`CpusetSpec`] against this context's topology and
    /// return the CPU count. Convenience accessor for tests that need
    /// to size work counts proportional to a cpuset without computing
    /// the topology denominator by hand. Mirrors the framework's own
    /// resolution: the count is exactly the size of the BTreeSet
    /// `spec.resolve(self)` returns, so any
    /// `CpusetSpec`-aware code path (cgroup cpuset assignment,
    /// affinity intent resolution, [`WorkSpec::workers_pct`]) sees the
    /// same denominator. Uses the TOPOLOGY-level cpuset, not the
    /// currently-effective cgroup cpuset — narrowing via mid-scenario
    /// `Op::SetCpuset` does not change the value this returns.
    pub fn cpuset_cpus(&self, spec: &crate::scenario::ops::CpusetSpec) -> usize {
        spec.resolve(self).len()
    }

    /// `HoldSpec::fixed(settle + duration * fraction_of_duration)` —
    /// the dominant Step hold-time pattern across scenarios. A Step
    /// typically holds for the settle window (so the scheduler can
    /// reach steady state) plus some fraction of the workload
    /// duration (often `1.0` for whole-test Steps, or `0.5`/`1.0/3.0`
    /// for multi-Step scenarios that subdivide the duration budget).
    ///
    /// The multiplication routes through [`Duration::mul_f64`], so a
    /// fraction like `1.0 / 3.0` may yield a Duration that differs
    /// from an integer-division formulation by ≤1 nanosecond — below
    /// Linux thread sleep granularity and so unobservable at the
    /// hold-evaluation boundary, but worth noting if a test ever
    /// byte-pins a Duration value.
    ///
    /// # Panics
    /// When `fraction_of_duration` is NaN, infinite, or negative
    /// (per the `Duration::mul_f64` contract).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// Step::new(vec![], ctx.settled_hold(0.5));  // settle + half duration
    /// Step::new(vec![], ctx.settled_hold(1.0));  // settle + full duration
    /// ```
    pub fn settled_hold(&self, fraction_of_duration: f64) -> crate::scenario::ops::HoldSpec {
        crate::scenario::ops::HoldSpec::fixed(
            self.settle + self.duration.mul_f64(fraction_of_duration),
        )
    }

    /// Construct a [`CgroupDef`] with `self.workers_per_cgroup`
    /// workers — the most common scenario shape, dedupe of 40+
    /// `CgroupDef::named(name).workers(ctx.workers_per_cgroup)` call
    /// sites across `src/scenario/` and `tests/`.
    ///
    /// Equivalent to:
    ///
    /// ```ignore
    /// CgroupDef::named(name).workers(ctx.workers_per_cgroup)
    /// ```
    ///
    /// Returns a fresh [`CgroupDef`] so the test author can chain
    /// further builders (`.cpuset`, `.work`, etc.) on the
    /// result. For non-default worker counts call
    /// `CgroupDef::named(name).workers(N)` directly — the helper
    /// pins ONLY the `ctx.workers_per_cgroup` default path.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Before (42+ sites):
    /// vec![CgroupDef::named("cg_0").workers(ctx.workers_per_cgroup)].into()
    /// // After:
    /// vec![ctx.cgroup_def("cg_0")].into()
    ///
    /// // With additional builders:
    /// ctx.cgroup_def("cg_0").cpuset(...)
    /// ```
    pub fn cgroup_def(
        &self,
        name: impl Into<std::borrow::Cow<'static, str>>,
    ) -> crate::scenario::ops::CgroupDef {
        crate::scenario::ops::CgroupDef::named(name).workers(self.workers_per_cgroup)
    }
}

/// Fluent builder for [`Ctx`].
///
/// Scenario unit tests reach for a [`Ctx`] with sane defaults so they
/// can exercise scenario logic without booting a VM. The direct
/// struct-literal construction at ~14 call sites forces every test to
/// repeat the full 9-field init and keeps diverging defaults in sync
/// by hand; this builder centralises those defaults and keeps required
/// fields (borrowed `cgroups`/`topo`) in their types.
///
/// Defaults:
/// - `duration`: 1 s — matches the `scenario::basic` test helper
///   (`scenario::stress` uses 2 s and sets it explicitly)
/// - `workers_per_cgroup`: 1
/// - `sched_pid`: `None` — [`run_scenario`] short-circuits the
///   liveness checks when `sched_pid.is_none()`.
/// - `settle`: 0 ms — tests do not need to wait for scheduler stabilisation
/// - `work_type_override`: `None`
/// - `assert`: [`crate::assert::Assert::default_checks()`] —
///   the same policy production paths merge through
/// - `wait_for_map_write`: `false`
///
/// Override any default via the corresponding method, then materialise
/// the context with [`CtxBuilder::build`].
///
/// # Example
/// ```ignore
/// let cgroups = CgroupManager::new("/nonexistent");
/// let topo = TestTopology::synthetic(4, 1);
/// let ctx = Ctx::builder(&cgroups, &topo)
///     .workers_per_cgroup(3)
///     .duration(Duration::from_secs(2))
///     .build();
/// ```
pub struct CtxBuilder<'a> {
    cgroups: &'a dyn crate::cgroup::CgroupOps,
    topo: &'a TestTopology,
    duration: Duration,
    workers_per_cgroup: usize,
    sched_pid: Option<libc::pid_t>,
    settle: Duration,
    work_type_override: Option<WorkType>,
    assert: crate::assert::Assert,
    wait_for_map_write: bool,
}

impl<'a> CtxBuilder<'a> {
    /// Wall-clock budget for the workload phase of the scenario.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn duration(mut self, d: Duration) -> Self {
        self.duration = d;
        self
    }

    /// Number of worker threads started per cgroup by the default workload.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn workers_per_cgroup(mut self, n: usize) -> Self {
        self.workers_per_cgroup = n;
        self
    }

    /// PID of the scheduler process; `None` disables the liveness
    /// checks in [`run_scenario`].
    #[must_use = "builder methods consume self; bind the result"]
    pub fn sched_pid(mut self, pid: Option<libc::pid_t>) -> Self {
        self.sched_pid = pid;
        self
    }

    /// Time to wait after cgroup creation for scheduler stabilisation.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn settle(mut self, s: Duration) -> Self {
        self.settle = s;
        self
    }

    /// Override the default work type for scenarios that would
    /// otherwise use `SpinWait`.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn work_type_override(mut self, wt: Option<WorkType>) -> Self {
        self.work_type_override = wt;
        self
    }

    /// Merged assertion config. Callers that want the production
    /// layering should pass `Assert::default_checks().merge(&...)`;
    /// tests that pin a specific policy can pass
    /// [`crate::assert::Assert::NO_OVERRIDES`] directly.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn assert(mut self, a: crate::assert::Assert) -> Self {
        self.assert = a;
        self
    }

    /// When true, `execute_steps` polls the SHM signal slot after
    /// writing the scenario start marker. See the field doc on
    /// [`Ctx::wait_for_map_write`].
    #[must_use = "builder methods consume self; bind the result"]
    pub fn wait_for_map_write(mut self, v: bool) -> Self {
        self.wait_for_map_write = v;
        self
    }

    /// Materialise the configured [`Ctx`].
    #[must_use = "dropping a Ctx without running the scenario discards the test setup"]
    pub fn build(self) -> Ctx<'a> {
        Ctx {
            cgroups: self.cgroups,
            topo: self.topo,
            duration: self.duration,
            workers_per_cgroup: self.workers_per_cgroup,
            sched_pid: self.sched_pid,
            settle: self.settle,
            work_type_override: self.work_type_override,
            assert: self.assert,
            wait_for_map_write: self.wait_for_map_write,
        }
    }
}

impl<'a> Ctx<'a> {
    /// Start a new [`CtxBuilder`] with required `cgroups` and `topo`
    /// borrows and sane defaults for every other field. See
    /// [`CtxBuilder`] for the full default set.
    #[must_use = "discarding a CtxBuilder drops the scenario context defaults; chain setters and call .build()"]
    pub fn builder(
        cgroups: &'a dyn crate::cgroup::CgroupOps,
        topo: &'a TestTopology,
    ) -> CtxBuilder<'a> {
        CtxBuilder {
            cgroups,
            topo,
            duration: Duration::from_secs(1),
            workers_per_cgroup: 1,
            sched_pid: None,
            settle: Duration::from_millis(0),
            work_type_override: None,
            assert: crate::assert::Assert::default_checks(),
            wait_for_map_write: false,
        }
    }

    /// Start a [`PayloadRun`](crate::scenario::payload_run::PayloadRun)
    /// builder for the given [`Payload`](crate::test_support::Payload).
    ///
    /// The builder inherits `payload.default_args` and
    /// `payload.default_checks`; chained `.arg(...)` / `.check(...)`
    /// calls extend them; `.clear_args()` / `.clear_checks()` wipe
    /// both defaults and prior appends. Terminal `.run()` blocks and
    /// returns `Result<(AssertResult, PayloadMetrics)>`.
    ///
    /// Only `PayloadKind::Binary` payloads are runnable here;
    /// `.run()` on a `PayloadKind::Scheduler` payload returns `Err`.
    #[must_use = "dropping a PayloadRun discards the payload configuration; chain setters and call .run()"]
    pub fn payload(
        &'a self,
        p: &'static crate::test_support::Payload,
    ) -> crate::scenario::payload_run::PayloadRun<'a> {
        crate::scenario::payload_run::PayloadRun::new(self, p)
    }
}

/// Spawn workers per cgroup, move each handle's worker pids into
/// its cgroup, then start all handles in a second pass.
///
/// Shared scaffolding for [`run_scenario`] and [`setup_cgroups`] —
/// both defer `.start()` until every handle has been spawned and
/// every worker pid moved, so workers see a stable cgroup
/// membership at first run. [`spawn_diverse`] does NOT use this
/// helper because it starts each handle inline (eager-start
/// semantics required for its IoSyncWrite/SpinWait mix — workload
/// ordering matters when the mix includes I/O-bound and CPU-bound
/// cgroups).
///
/// `cfg_fn` builds the per-cgroup [`WorkloadConfig`] from its
/// index + name; callers own the per-cgroup customization logic.
///
/// `move_tasks` is ESRCH-tolerant — a worker that exits between
/// fork and cgroup placement is warned and skipped, unlike the
/// original per-pid `move_task` which propagated ESRCH.
fn spawn_and_move<F>(ctx: &Ctx, names: &[String], mut cfg_fn: F) -> Result<Vec<WorkloadHandle>>
where
    F: FnMut(usize, &str) -> Result<WorkloadConfig>,
{
    let mut handles = Vec::with_capacity(names.len());
    for (i, name) in names.iter().enumerate() {
        let wl = cfg_fn(i, name.as_str())?;
        let h = WorkloadHandle::spawn(&wl)?;
        tracing::debug!(
            cgroup = %name,
            workers = wl.num_workers,
            pids = h.worker_pids().len(),
            "spawned workers",
        );
        ctx.cgroups
            .move_tasks(name.as_str(), &h.worker_pids_for_cgroup_procs()?)?;
        handles.push(h);
    }
    for h in &mut handles {
        h.start();
    }
    Ok(handles)
}

/// Resolve an [`AffinityIntent`] to a concrete [`ResolvedAffinity`] for workers
/// in a cgroup with the given effective cpuset.
///
/// When a cpuset is active, affinity masks are intersected with it so the
/// effective `sched_setaffinity` mask matches what the kernel will enforce.
/// Without a cpuset, the full topology is used.
/// Resolve a [`WorkSpec`]'s `num_workers`, falling back to `default_n` when unset,
/// and reject `num_workers=0`.
///
/// A cgroup with no workers emits no `WorkerReport`s, so every downstream
/// assertion vacuously passes. Callers that want "no load" on a cgroup
/// should either drop the `WorkSpec` entry entirely (letting the default apply)
/// or use a single sentinel worker so assertions have something to check.
pub(crate) fn resolve_num_workers(work: &WorkSpec, default_n: usize, label: &str) -> Result<usize> {
    let n = work.num_workers.unwrap_or(default_n);
    if n == 0 {
        anyhow::bail!(
            "cgroup '{}': num_workers=0 is not allowed — assertions would \
             vacuously pass with no WorkerReports; use at least 1 worker or \
             drop this WorkSpec entry",
            label,
        );
    }
    Ok(n)
}

/// Resolve an [`AffinityIntent`] to a concrete [`ResolvedAffinity`]
/// for workers in a cgroup with the given effective cpuset.
///
/// # Errors
///
/// Returns `Err` when the test author's affinity intent cannot be
/// satisfied against the cgroup's effective cpuset. Per the
/// project-wide no-silent-drops invariant, an unsatisfiable
/// intent must surface as a returnable error rather than silently
/// degrading to "no affinity applied" — silent degradation lets
/// the workload run with the wrong placement while the test
/// reports success (vacuously-passing assertions).
///
/// The unsatisfiable cases by variant:
/// - [`AffinityIntent::RandomSubset`]: `from` pool empty after
///   cpuset intersection, or `count == 0`.
/// - [`AffinityIntent::LlcAligned`]: every LLC's CPUs disjoint
///   from the cpuset (no LLC has any CPU inside the cpuset).
/// - [`AffinityIntent::SingleCpu`]: cpuset is empty.
/// - [`AffinityIntent::Exact`]: requested CPU set is empty
///   (`Exact(BTreeSet::new())` is intent-only unsatisfiable),
///   or requested CPU set disjoint from the cpuset
///   (intersection empty).
/// - [`AffinityIntent::SmtSiblingPair`]: no physical core with
///   ≥2 SMT siblings inside the cpuset.
///
/// Every error diagnostic names the offending intent and a
/// remediation hint. Diagnostics for cpuset-narrowed pools
/// (`RandomSubset` empty intersection, `LlcAligned`, `SingleCpu`,
/// `Exact` disjoint-intersection, `SmtSiblingPair`) also render the
/// cpuset that narrowed the pool. The intent-only errors —
/// `RandomSubset { count: 0 }` and `Exact(BTreeSet::new())` — omit
/// the cpuset because the cpuset is irrelevant to the failure (the
/// intent itself names zero CPUs). Remediation hints include
/// switching to [`AffinityIntent::Inherit`] to deliberately inherit
/// the cpuset, widening the cgroup's cpuset, or picking CPUs inside
/// the cpuset.
pub fn resolve_affinity_for_cgroup(
    kind: &AffinityIntent,
    cpuset: Option<&BTreeSet<usize>>,
    topo: &TestTopology,
) -> Result<ResolvedAffinity> {
    match kind {
        AffinityIntent::Inherit => Ok(ResolvedAffinity::None),
        AffinityIntent::RandomSubset { from, count } => {
            // Validate the intent itself (count > 0) before doing any
            // resource work — an intent-only bug (count==0) doesn't
            // need an allocation to diagnose.
            if *count == 0 {
                anyhow::bail!(
                    "AffinityIntent::RandomSubset count=0 cannot satisfy any sample. \
                     Switch to `AffinityIntent::Inherit` to deliberately inherit the \
                     cgroup cpuset, or pass `count >= 1`.",
                );
            }
            // The pool is already resolved by the caller (typed
            // `from`). Intersect with the cgroup's cpuset if one is
            // active so the resolved pool stays within the
            // scenario's CPU budget — same intersection semantic
            // applied to `Exact` below.
            let pool = if let Some(cs) = cpuset {
                from.intersection(cs).copied().collect::<BTreeSet<usize>>()
            } else {
                from.clone()
            };
            if pool.is_empty() {
                if cpuset.is_some() {
                    let cpuset_repr = format_cpuset_for_diag(cpuset);
                    anyhow::bail!(
                        "AffinityIntent::RandomSubset has no CPUs after intersecting \
                         `from={from:?}` with the cgroup cpuset ({cpuset_repr}). \
                         Switch to `AffinityIntent::Inherit` to deliberately inherit \
                         the cgroup cpuset, widen the cgroup's cpuset, or pick a \
                         `from` set that overlaps the cpuset.",
                    );
                } else {
                    anyhow::bail!(
                        "AffinityIntent::RandomSubset has an empty `from` pool with \
                         no cgroup cpuset to narrow it — there is no CPU to sample. \
                         Switch to `AffinityIntent::Inherit` to deliberately inherit \
                         the scenario's CPU budget, or pass a non-empty `from` set.",
                    );
                }
            }
            Ok(ResolvedAffinity::Random {
                from: pool,
                count: *count,
            })
        }
        AffinityIntent::LlcAligned => {
            let pool = cpuset.cloned().unwrap_or_else(|| topo.all_cpuset());
            // Find the LLC that has the most overlap with the cpuset.
            let mut best_llc = topo.llc_aligned_cpuset(0);
            let mut best_overlap = best_llc.intersection(&pool).count();
            for idx in 1..topo.num_llcs() {
                let llc = topo.llc_aligned_cpuset(idx);
                let overlap = llc.intersection(&pool).count();
                if overlap > best_overlap {
                    best_llc = llc;
                    best_overlap = overlap;
                }
            }
            // Intersect with cpuset so effective affinity matches kernel behavior.
            let effective: BTreeSet<usize> = best_llc.intersection(&pool).copied().collect();
            if effective.is_empty() {
                let cpuset_repr = format_cpuset_for_diag(cpuset);
                anyhow::bail!(
                    "AffinityIntent::LlcAligned has no CPUs after intersecting every \
                     LLC with the cgroup cpuset ({cpuset_repr}). No LLC has any CPU \
                     inside the cpuset. Switch to `AffinityIntent::Inherit` to \
                     deliberately inherit the cpuset, widen the cgroup's cpuset to \
                     include CPUs from at least one LLC, or pick a different \
                     affinity intent that doesn't require LLC alignment.",
                );
            }
            Ok(ResolvedAffinity::Fixed(effective))
        }
        AffinityIntent::CrossCgroup => {
            // When a cpuset is active, crossing cgroup boundaries is the intent,
            // but the kernel will intersect. Use all CPUs -- the kernel enforces
            // the cpuset constraint.
            Ok(ResolvedAffinity::Fixed(topo.all_cpuset()))
        }
        AffinityIntent::SingleCpu => {
            let pool = cpuset.cloned().unwrap_or_else(|| topo.all_cpuset());
            if let Some(&cpu) = pool.iter().next() {
                Ok(ResolvedAffinity::SingleCpu(cpu))
            } else {
                // Pool is empty only when cpuset is Some(empty) — `all_cpuset()`
                // returns at least the boot CPU for any non-degenerate topology.
                anyhow::bail!(
                    "AffinityIntent::SingleCpu cannot pick a CPU from an empty \
                     cgroup cpuset. Switch to `AffinityIntent::Inherit` to \
                     deliberately inherit (the empty cpuset is itself the \
                     problem), or assign a non-empty cpuset to the cgroup.",
                );
            }
        }
        AffinityIntent::Exact(cpus) => {
            if cpus.is_empty() {
                // Empty Exact is the most-explicit way a user can say
                // "I made a mistake" — silently degrading it to
                // Inherit is the same no-silent-drop violation as the
                // disjoint-intersection case below.
                anyhow::bail!(
                    "AffinityIntent::Exact(BTreeSet::new()) is unsatisfiable — an \
                     empty CPU set pins workers to nothing. Switch to \
                     `AffinityIntent::Inherit` to deliberately inherit the cgroup \
                     cpuset (or the full topology when no cpuset is active), or \
                     pass at least one CPU ID.",
                );
            }
            if let Some(cs) = cpuset {
                let effective: BTreeSet<usize> = cpus.intersection(cs).copied().collect();
                if effective.is_empty() {
                    let cpuset_repr = format_cpuset_for_diag(cpuset);
                    anyhow::bail!(
                        "AffinityIntent::Exact({cpus:?}) is disjoint from the cgroup \
                         cpuset ({cpuset_repr}); intersection is empty. Switch to \
                         `AffinityIntent::Inherit` to deliberately inherit the cpuset, \
                         widen the cgroup's cpuset to include the requested CPUs, or \
                         narrow the `Exact` set to CPUs inside the cpuset.",
                    );
                }
                Ok(ResolvedAffinity::Fixed(effective))
            } else {
                Ok(ResolvedAffinity::Fixed(cpus.clone()))
            }
        }
        AffinityIntent::SmtSiblingPair => resolve_smt_sibling_pair(cpuset, topo),
    }
}

/// Render a cgroup cpuset for the bail diagnostics on
/// [`resolve_affinity_for_cgroup`]'s unsatisfiable arms. `None`
/// renders as `<no cpuset>` so the operator can distinguish
/// "cpuset is empty" from "no cpuset is active" — both can produce
/// an empty intersection on different intents.
fn format_cpuset_for_diag(cpuset: Option<&BTreeSet<usize>>) -> String {
    match cpuset {
        Some(cs) if cs.is_empty() => "empty cpuset {}".to_string(),
        Some(cs) => format!("cpuset {cs:?}"),
        None => "<no cpuset>".to_string(),
    }
}

/// Resolve [`AffinityIntent::SmtSiblingPair`] against the cgroup's
/// effective cpuset.
///
/// Walks every LLC's per-core sibling map looking for a physical
/// core whose SMT siblings are all present in the pool (cgroup's
/// cpuset, or the full topology when no cpuset is active). Returns
/// the first matching pair as [`ResolvedAffinity::Fixed`] containing
/// the two sibling CPU IDs.
///
/// Returns `Err` when no core has 2+ siblings in the pool —
/// `threads_per_core == 1` (SMT disabled or non-SMT host), the
/// cpuset isolates each sibling onto a different cgroup, or the
/// topology was constructed without per-core sibling data
/// (`LlcInfo::cores` empty — see [`TestTopology::synthetic`]). The
/// error path is explicit, not a silent fallback, because
/// [`WorkType::SmtSiblingSpin`] and other paired-on-siblings
/// workloads produce meaningless results without true SMT
/// contention.
///
/// All workers in the group resolve to the same 2-CPU set; for
/// `num_workers == 2` the kernel runs one worker on each sibling,
/// which is the contention pattern this intent targets. For
/// `num_workers > 2` (multiple pairs in one group) every worker
/// shares the same pair — the kernel time-slices them, which
/// approximates pair contention but does not place each pair on
/// distinct cores. Strict per-pair distribution across cores
/// requires per-worker affinity that the current
/// [`ResolvedAffinity`] model does not express; track via a
/// follow-up if a test author needs it.
///
/// [`WorkType::SmtSiblingSpin`]: crate::workload::WorkType::SmtSiblingSpin
/// [`AffinityIntent::SmtSiblingPair`]: crate::workload::AffinityIntent::SmtSiblingPair
fn resolve_smt_sibling_pair(
    cpuset: Option<&BTreeSet<usize>>,
    topo: &TestTopology,
) -> Result<ResolvedAffinity> {
    let pool = cpuset.cloned().unwrap_or_else(|| topo.all_cpuset());
    for llc in topo.llcs() {
        for siblings in llc.cores().values() {
            // Take the first two sibling CPUs that are both in the
            // pool. `cores()` is sorted; pairing the lowest two
            // present siblings gives a deterministic choice for a
            // given (topology, cpuset) input.
            let mut iter = siblings.iter().copied().filter(|cpu| pool.contains(cpu));
            if let (Some(a), Some(b)) = (iter.next(), iter.next()) {
                let pair: BTreeSet<usize> = [a, b].into_iter().collect();
                return Ok(ResolvedAffinity::Fixed(pair));
            }
        }
    }
    anyhow::bail!(
        "AffinityIntent::SmtSiblingPair requires a physical core with at \
         least two SMT siblings present in the effective cpuset. The \
         current topology and cpuset expose no such pair — \
         threads_per_core may be 1 (SMT disabled or non-SMT host), the \
         cpuset may have isolated each sibling onto a different cgroup, \
         or the topology was built without per-core sibling data. \
         Switch to a different AffinityIntent for non-SMT scheduling \
         tests, or run on a host whose VM topology has \
         threads_per_core >= 2.",
    );
}

/// Resolve an [`AffinityIntent`] for direct storage in
/// [`crate::workload::WorkloadConfig::affinity`].
///
/// [`crate::workload::WorkloadConfig::affinity`] is an
/// [`AffinityIntent`] (type-unified with [`crate::workload::WorkSpec::affinity`])
/// and its spawn-time gate (see
/// [`crate::workload::WorkloadHandle::spawn`]) accepts
/// [`AffinityIntent::Inherit`], [`AffinityIntent::Exact`], and
/// [`AffinityIntent::RandomSubset`]. The scenario engine holds the
/// topology and cpuset that the spawn-time gate lacks, so it
/// pre-resolves topology-aware variants here:
///
/// - [`ResolvedAffinity::None`] → [`AffinityIntent::Inherit`]
/// - [`ResolvedAffinity::Fixed(set)`](ResolvedAffinity::Fixed) →
///   [`AffinityIntent::Exact(set)`](AffinityIntent::Exact)
/// - [`ResolvedAffinity::SingleCpu(cpu)`](ResolvedAffinity::SingleCpu) →
///   [`AffinityIntent::Exact`] containing `cpu`
/// - [`ResolvedAffinity::Random { from, count }`](ResolvedAffinity::Random) →
///   [`AffinityIntent::RandomSubset { from, count }`](AffinityIntent::RandomSubset)
///   — the resolved pool is forwarded verbatim and per-worker
///   sampling stays deferred to spawn time (each worker gets an
///   independent draw from `from`).
///
/// # Errors
///
/// Forwards every `Err` from the inner [`resolve_affinity_for_cgroup`]
/// — see that function's `# Errors` section for the full list of
/// unsatisfiable cases (RandomSubset empty pool / count=0,
/// LlcAligned no-overlap, SingleCpu empty cpuset, Exact empty or
/// disjoint, SmtSiblingPair no-pair-in-cpuset). The empty-pool
/// "silent degrade to Inherit" policy that previously lived here
/// was removed — empty pools are operator bugs, not "soft" fallbacks.
pub(crate) fn intent_for_spawn(
    kind: &AffinityIntent,
    cpuset: Option<&BTreeSet<usize>>,
    topo: &TestTopology,
) -> Result<AffinityIntent> {
    Ok(flatten_for_spawn(resolve_affinity_for_cgroup(
        kind, cpuset, topo,
    )?))
}

fn flatten_for_spawn(resolved: ResolvedAffinity) -> AffinityIntent {
    match resolved {
        ResolvedAffinity::None => AffinityIntent::Inherit,
        ResolvedAffinity::Fixed(set) => {
            if set.is_empty() {
                // Invariant: resolve_affinity_for_cgroup bails before
                // constructing an empty Fixed (LlcAligned
                // empty-effective bail, Exact empty-input bail, Exact
                // disjoint-intersection bail). Reaching here means a
                // future constructor of ResolvedAffinity::Fixed
                // bypassed those checks — panic loudly so the
                // regression surfaces at the construction site, not
                // as a silent inheritance downstream.
                unreachable!(
                    "ResolvedAffinity::Fixed(empty) reached flatten_for_spawn — \
                     resolve_affinity_for_cgroup is supposed to bail on every \
                     path that produces an empty Fixed (no-silent-drops \
                     invariant). Audit the new caller that constructed it.",
                )
            } else {
                AffinityIntent::Exact(set)
            }
        }
        ResolvedAffinity::SingleCpu(cpu) => AffinityIntent::Exact([cpu].into_iter().collect()),
        ResolvedAffinity::Random { from, count } => {
            // Round-trip the resolved pool through
            // [`AffinityIntent::RandomSubset`] so per-worker
            // sampling stays deferred to spawn time
            // (`workload::resolve_affinity` samples each worker
            // independently).
            if count == 0 || from.is_empty() {
                // Invariant: resolve_affinity_for_cgroup bails on
                // RandomSubset { count: 0 } and on empty intersected
                // pools. Same regression-surface contract as the
                // Fixed arm above.
                unreachable!(
                    "ResolvedAffinity::Random {{ count={count}, from={from:?} }} \
                     reached flatten_for_spawn with count==0 or empty pool — \
                     resolve_affinity_for_cgroup is supposed to bail on those \
                     cases (no-silent-drops invariant). Audit the new caller \
                     that constructed it.",
                )
            } else {
                AffinityIntent::RandomSubset { from, count }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Custom scenario helpers
// ---------------------------------------------------------------------------

/// Create N cgroups, spawn workers in each, and start them.
///
/// Returns the worker handles and an RAII [`CgroupGroup`] that removes
/// the cgroups on drop. Workers are moved into their target cgroups
/// before being signaled to start.
pub fn setup_cgroups<'a>(
    ctx: &'a Ctx,
    n: usize,
    wl: &WorkloadConfig,
) -> Result<(Vec<WorkloadHandle>, CgroupGroup<'a>)> {
    let mut guard = CgroupGroup::new(ctx.cgroups);
    for i in 0..n {
        guard.add_cgroup_no_cpuset(&format!("cg_{i}"))?;
    }
    thread::sleep(ctx.settle);
    // `active_sched_pid()` returns `None` when no scheduler was
    // configured (kernel-default path) OR when the caller planted a
    // `<= 0` sentinel; both cases skip the liveness-based bail.
    if let Some(pid) = ctx.active_sched_pid()
        && !process_alive(pid)
    {
        anyhow::bail!(
            "{} after cgroup creation (pid={})",
            crate::assert::SCHED_DIED_PREFIX,
            pid,
        );
    }
    let names: Vec<String> = (0..n).map(|i| format!("cg_{i}")).collect();
    let handles = spawn_and_move(ctx, &names, |_, _| Ok(wl.clone()))?;
    Ok((handles, guard))
}

/// Stop workers, collect reports, and merge assertion results.
///
/// Each item is a `(WorkloadHandle, Option<&BTreeSet<usize>>)` pair
/// where the optional cpuset is passed through to
/// [`Assert::assert_cgroup`](crate::assert::Assert::assert_cgroup)
/// for isolation checks. When `checks` has no worker-level checks,
/// workers are collected but no assertions run.
pub(crate) fn collect_handles<'a>(
    handles: impl IntoIterator<Item = (WorkloadHandle, Option<&'a BTreeSet<usize>>)>,
    checks: &crate::assert::Assert,
    topo: Option<&crate::topology::TestTopology>,
) -> AssertResult {
    let mut r = AssertResult::pass();
    for (h, cpuset) in handles {
        let reports = h.stop_and_collect();
        if checks.has_worker_checks() {
            let numa_nodes = cpuset.and_then(|cs| topo.map(|t| t.numa_nodes_for_cpuset(cs)));
            r.merge(checks.assert_cgroup_with_numa(&reports, cpuset, numa_nodes.as_ref()));
        }
    }
    r
}

/// Stop all workers, collect reports, and run assertion checks.
///
/// Uses `checks` for worker evaluation. Returns a merged
/// [`AssertResult`] across all workers.
pub fn collect_all(handles: Vec<WorkloadHandle>, checks: &crate::assert::Assert) -> AssertResult {
    collect_handles(handles.into_iter().map(|h| (h, None)), checks, None)
}

/// Default [`WorkloadConfig`] with `ctx.workers_per_cgroup` workers.
pub fn dfl_wl(ctx: &Ctx) -> WorkloadConfig {
    WorkloadConfig {
        num_workers: ctx.workers_per_cgroup,
        ..Default::default()
    }
}

#[cfg(test)]
pub fn split_half(ctx: &Ctx) -> (BTreeSet<usize>, BTreeSet<usize>) {
    let usable = ctx.topo.usable_cpus();
    let mid = usable.len() / 2;
    (
        usable[..mid].iter().copied().collect(),
        usable[mid..].iter().copied().collect(),
    )
}

/// Spawn diverse workloads across N cgroups: SpinWait, Bursty,
/// IoSyncWrite, Mixed, YieldHeavy. Each cgroup uses
/// `ctx.workers_per_cgroup` workers except IoSyncWrite cgroups,
/// which always use 2 workers to avoid drowning the scenario in
/// blocking IO.
pub fn spawn_diverse(ctx: &Ctx, cgroup_names: &[&str]) -> Result<Vec<WorkloadHandle>> {
    let types = [
        WorkType::SpinWait,
        WorkType::bursty(Duration::from_millis(50), Duration::from_millis(100)),
        WorkType::IoSyncWrite,
        WorkType::Mixed,
        WorkType::YieldHeavy,
    ];
    let mut handles = Vec::new();
    for (i, name) in cgroup_names.iter().enumerate() {
        let wt = types[i % types.len()].clone();
        let n = if matches!(wt, WorkType::IoSyncWrite) {
            2
        } else {
            ctx.workers_per_cgroup
        };
        let mut h = WorkloadHandle::spawn(&WorkloadConfig {
            num_workers: n,
            work_type: wt,
            ..Default::default()
        })?;
        ctx.cgroups
            .move_tasks(name, &h.worker_pids_for_cgroup_procs()?)?;
        h.start();
        handles.push(h);
    }
    Ok(handles)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert;

    #[test]
    fn resolve_affinity_inherit() {
        let t = crate::topology::TestTopology::synthetic(8, 2);
        assert!(matches!(
            resolve_affinity_for_cgroup(&AffinityIntent::Inherit, None, &t).unwrap(),
            ResolvedAffinity::None
        ));
    }

    #[test]
    fn resolve_affinity_single_cpu() {
        let t = crate::topology::TestTopology::synthetic(8, 2);
        match resolve_affinity_for_cgroup(&AffinityIntent::SingleCpu, None, &t).unwrap() {
            ResolvedAffinity::SingleCpu(c) => assert_eq!(c, 0),
            other => panic!("expected SingleCpu, got {:?}", other),
        }
    }

    #[test]
    fn resolve_affinity_cross_cgroup() {
        let t = crate::topology::TestTopology::synthetic(8, 2);
        match resolve_affinity_for_cgroup(&AffinityIntent::CrossCgroup, None, &t).unwrap() {
            ResolvedAffinity::Fixed(cpus) => assert_eq!(cpus.len(), 8),
            other => panic!("expected Fixed, got {:?}", other),
        }
    }

    #[test]
    fn resolve_affinity_llc_aligned() {
        let t = crate::topology::TestTopology::synthetic(8, 2);
        // No cpuset: both LLCs cover the full pool equally. LLC 0
        // is found first with max overlap, so result is LLC 0 CPUs.
        match resolve_affinity_for_cgroup(&AffinityIntent::LlcAligned, None, &t).unwrap() {
            ResolvedAffinity::Fixed(cpus) => assert_eq!(cpus, [0, 1, 2, 3].into_iter().collect()),
            other => panic!("expected Fixed, got {:?}", other),
        }
    }

    #[test]
    fn resolve_affinity_llc_aligned_with_cpuset() {
        let t = crate::topology::TestTopology::synthetic(8, 2);
        // Cpuset restricted to LLC 1 CPUs: LlcAligned picks LLC 1.
        let cpusets: Vec<BTreeSet<usize>> = vec![
            [0, 1, 2, 3].into_iter().collect(),
            [4, 5, 6, 7].into_iter().collect(),
        ];
        match resolve_affinity_for_cgroup(&AffinityIntent::LlcAligned, cpusets.get(1), &t).unwrap()
        {
            ResolvedAffinity::Fixed(cpus) => assert_eq!(cpus, [4, 5, 6, 7].into_iter().collect()),
            other => panic!("expected Fixed, got {:?}", other),
        }
    }

    #[test]
    fn resolve_affinity_random_subset() {
        let t = crate::topology::TestTopology::synthetic(8, 2);
        let cpusets: Vec<BTreeSet<usize>> = vec![[0, 1, 2, 3].into_iter().collect()];
        // Caller pre-builds the pool from topology; the resolver
        // intersects with the cgroup's cpuset so the effective sample
        // pool stays within the cgroup's CPU budget. Sample size
        // is half the cpuset (`(pool.len() / 2).max(1)`).
        let intent = AffinityIntent::random_subset(t.all_cpus().iter().copied(), 2);
        match resolve_affinity_for_cgroup(&intent, cpusets.first(), &t).unwrap() {
            ResolvedAffinity::Random { from, count } => {
                assert_eq!(from, cpusets[0]);
                assert_eq!(count, 2); // half of 4
            }
            other => panic!("expected Random, got {:?}", other),
        }
    }

    #[test]
    fn cgroup_work_default() {
        let cw = WorkSpec::default();
        assert_eq!(cw.num_workers, None);
        assert!(matches!(cw.work_type, WorkType::SpinWait));
        assert!(matches!(cw.sched_policy, SchedPolicy::Normal));
        assert!(matches!(cw.affinity, AffinityIntent::Inherit));
        assert!(matches!(cw.mem_policy, MemPolicy::Default));
    }

    #[test]
    fn resolve_affinity_random_no_cpusets() {
        let t = crate::topology::TestTopology::synthetic(8, 2);
        // No cgroup cpuset → pool is the caller-supplied set verbatim.
        let intent = AffinityIntent::random_subset(t.all_cpus().iter().copied(), 4);
        match resolve_affinity_for_cgroup(&intent, None, &t).unwrap() {
            ResolvedAffinity::Random { from, count } => {
                assert_eq!(from.len(), 8); // all CPUs
                assert_eq!(count, 4); // half
            }
            other => panic!("expected Random, got {:?}", other),
        }
    }

    #[test]
    fn resolve_affinity_random_subset_empty_pool_bails() {
        // Empty cpuset intersection on RandomSubset must surface as
        // a returnable Err — per the project no-silent-drops
        // invariant, an unsatisfiable affinity intent is the
        // operator's bug to fix (widen the cpuset, change the
        // intent), not a silent "default placement" the test
        // happily accepts. (Earlier code paths produced
        // ResolvedAffinity::None or even an empty Random.from that
        // sched_setaffinity rejected with EINVAL; both forms
        // masked the configuration error.)
        let t = crate::topology::TestTopology::synthetic(4, 1);
        let empty: BTreeSet<usize> = BTreeSet::new();
        let intent = AffinityIntent::random_subset(t.all_cpus().iter().copied(), 1);
        let err = resolve_affinity_for_cgroup(&intent, Some(&empty), &t)
            .expect_err("empty cpuset intersection must bail");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("AffinityIntent::RandomSubset"),
            "diagnostic must name the intent variant: got {msg}",
        );
        assert!(
            msg.contains("empty cpuset"),
            "diagnostic must name what narrowed the pool (empty cpuset): got {msg}",
        );
        assert!(
            msg.contains("AffinityIntent::Inherit"),
            "diagnostic must name the recommended fix (Inherit fallback): got {msg}",
        );
    }

    #[test]
    fn resolve_affinity_random_subset_empty_from_no_cpuset_bails() {
        // Companion to `_empty_pool_bails` above — that one pins the
        // `cpuset=Some(non_empty)` + `from=non_empty` → empty intersection
        // path. This test pins the OTHER way pool can be empty: caller
        // passes RandomSubset { from: empty } directly and no cpuset is
        // active. The bail diag must distinguish "intersected to empty"
        // (cpuset present) from "from-pool was empty with no cpuset to
        // narrow it" (no cpuset) so operators see actionable wording.
        let t = crate::topology::TestTopology::synthetic(4, 1);
        let intent = AffinityIntent::random_subset(std::iter::empty(), 1);
        let err = resolve_affinity_for_cgroup(&intent, None, &t)
            .expect_err("empty from-pool with no cpuset must bail");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("AffinityIntent::RandomSubset"),
            "diagnostic must name the intent variant: got {msg}",
        );
        assert!(
            msg.contains("empty `from` pool with no cgroup cpuset"),
            "diagnostic must distinguish this case from the cpuset-intersection \
             case so the operator sees the right remediation: got {msg}",
        );
        assert!(
            msg.contains("AffinityIntent::Inherit"),
            "diagnostic must name the recommended fix (Inherit fallback): got {msg}",
        );
    }

    #[test]
    fn resolve_affinity_random_subset_zero_count_bails() {
        let t = crate::topology::TestTopology::synthetic(4, 1);
        let intent = AffinityIntent::random_subset(t.all_cpus().iter().copied(), 0);
        let err = resolve_affinity_for_cgroup(&intent, None, &t)
            .expect_err("count=0 must bail");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("RandomSubset count=0"),
            "diagnostic must name count=0: got {msg}",
        );
    }

    #[test]
    fn resolve_affinity_llc_aligned_disjoint_cpuset_bails() {
        // LLC 0 = {0,1,2,3}, LLC 1 = {4,5,6,7}. Cpuset = {} (empty)
        // → every LLC's intersection is empty → bail.
        let t = crate::topology::TestTopology::synthetic(8, 2);
        let empty: BTreeSet<usize> = BTreeSet::new();
        let err = resolve_affinity_for_cgroup(&AffinityIntent::LlcAligned, Some(&empty), &t)
            .expect_err("empty cpuset on LlcAligned must bail");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("AffinityIntent::LlcAligned"),
            "diagnostic must name the intent variant: got {msg}",
        );
        assert!(
            msg.contains("No LLC has any CPU"),
            "diagnostic must name the failure mode: got {msg}",
        );
        assert!(
            msg.contains("AffinityIntent::Inherit"),
            "diagnostic must name the recommended fix: got {msg}",
        );
    }

    #[test]
    fn resolve_affinity_single_cpu_empty_cpuset_bails() {
        let t = crate::topology::TestTopology::synthetic(4, 1);
        let empty: BTreeSet<usize> = BTreeSet::new();
        let err = resolve_affinity_for_cgroup(&AffinityIntent::SingleCpu, Some(&empty), &t)
            .expect_err("empty cpuset on SingleCpu must bail");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("AffinityIntent::SingleCpu"),
            "diagnostic must name the intent variant: got {msg}",
        );
        assert!(
            msg.contains("empty cgroup cpuset"),
            "diagnostic must name the failure mode (empty cpuset): got {msg}",
        );
        assert!(
            msg.contains("AffinityIntent::Inherit"),
            "diagnostic must name the recommended fix: got {msg}",
        );
    }

    #[test]
    fn resolve_affinity_exact_disjoint_from_cpuset_bails() {
        let t = crate::topology::TestTopology::synthetic(4, 1);
        // Cpuset = {0, 1}; Exact = {2, 3}. Intersection is empty.
        let cpuset: BTreeSet<usize> = [0, 1].into_iter().collect();
        let exact: BTreeSet<usize> = [2, 3].into_iter().collect();
        let err = resolve_affinity_for_cgroup(
            &AffinityIntent::Exact(exact),
            Some(&cpuset),
            &t,
        )
        .expect_err("disjoint Exact must bail");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("AffinityIntent::Exact"),
            "diagnostic must name the intent variant: got {msg}",
        );
        assert!(
            msg.contains("disjoint"),
            "diagnostic must name the failure mode (disjoint): got {msg}",
        );
        assert!(
            msg.contains("AffinityIntent::Inherit"),
            "diagnostic must name the recommended fix: got {msg}",
        );
    }

    #[test]
    fn resolve_affinity_exact_empty_bails_regardless_of_cpuset() {
        // Empty Exact is the most-explicit way a user can say "I made
        // a mistake" — must bail whether or not a cpuset is active,
        // otherwise the no-cpuset branch silently produces
        // ResolvedAffinity::Fixed({}) which flatten_for_spawn used to
        // coerce to Inherit.
        let t = crate::topology::TestTopology::synthetic(4, 1);
        let empty: BTreeSet<usize> = BTreeSet::new();
        // (a) no cpuset
        let err_no_cs = resolve_affinity_for_cgroup(
            &AffinityIntent::Exact(empty.clone()),
            None,
            &t,
        )
        .expect_err("empty Exact must bail even without cpuset");
        let msg_no_cs = format!("{err_no_cs:#}");
        assert!(
            msg_no_cs.contains("AffinityIntent::Exact(BTreeSet::new())"),
            "diagnostic must name the empty-Exact form: got {msg_no_cs}",
        );
        assert!(
            msg_no_cs.contains("unsatisfiable"),
            "diagnostic must name the failure mode: got {msg_no_cs}",
        );
        assert!(
            msg_no_cs.contains("AffinityIntent::Inherit"),
            "diagnostic must name the recommended fix: got {msg_no_cs}",
        );
        // (b) with cpuset — same bail, doesn't reach the intersection arm
        let cpuset: BTreeSet<usize> = [0, 1].into_iter().collect();
        let err_with_cs = resolve_affinity_for_cgroup(
            &AffinityIntent::Exact(empty),
            Some(&cpuset),
            &t,
        )
        .expect_err("empty Exact must bail with cpuset too");
        let msg_with_cs = format!("{err_with_cs:#}");
        assert!(
            msg_with_cs.contains("AffinityIntent::Exact(BTreeSet::new())"),
            "diagnostic must name the empty-Exact form (not the disjoint message): got {msg_with_cs}",
        );
    }

    #[test]
    fn resolve_affinity_oob_cgroup_idx_falls_back_to_unrestricted() {
        // The wrapper that bounds-checked cgroup_idx against cpusets
        // was inlined into run_scenario (see the tracing::warn in
        // run_scenario's affinity-resolve block). The underlying
        // `cpusets.get(idx)` returns None on OOB, and
        // resolve_affinity_for_cgroup's None arm delivers the
        // unrestricted fallback. This test pins the fallback contract.
        let t = crate::topology::TestTopology::synthetic(4, 1);
        let cpusets: Vec<BTreeSet<usize>> = vec![[0, 1].into_iter().collect()];
        let oob_idx = 5;
        // Mirror the inlined expression in run_scenario:
        let cpuset = cpusets.get(oob_idx);
        assert!(cpuset.is_none(), "OOB index must yield None cpuset");
        let intent = AffinityIntent::random_subset(t.all_cpus().iter().copied(), 2);
        match resolve_affinity_for_cgroup(&intent, cpuset, &t).unwrap() {
            ResolvedAffinity::Random { from, count } => {
                assert_eq!(from.len(), 4, "OOB idx falls back to full topology");
                assert_eq!(count, 2);
            }
            other => panic!("expected Random with full pool, got {:?}", other),
        }
    }

    /// `SmtSiblingPair` resolves to a 2-CPU `Fixed` set drawn from
    /// the first physical core whose siblings live inside the
    /// effective cpuset. The sequential VM topology constructor
    /// numbers core `c` of LLC `l` with siblings at
    /// `(l*cores_per_llc + c) * threads_per_core ..`, so a
    /// `(1 numa, 1 llc, 2 cores, 2 threads)` topology produces
    /// core 0's pair `{0, 1}` first.
    #[test]
    fn resolve_affinity_smt_sibling_pair_uses_first_core() {
        let vmt = crate::vmm::topology::Topology::new(1, 1, 2, 2);
        let t = crate::topology::TestTopology::from_vm_topology(&vmt);
        match resolve_affinity_for_cgroup(&AffinityIntent::SmtSiblingPair, None, &t).unwrap() {
            ResolvedAffinity::Fixed(cpus) => {
                assert_eq!(
                    cpus,
                    [0usize, 1].into_iter().collect(),
                    "SmtSiblingPair must pick the first core's siblings"
                );
            }
            other => panic!("expected Fixed({{0, 1}}), got {:?}", other),
        }
    }

    /// When the cpuset excludes one of core 0's siblings, the
    /// resolver must skip that core and look for another core whose
    /// siblings are both present. With 2 cores per LLC and a cpuset
    /// of `{2, 3}`, core 1's pair `{2, 3}` is selected.
    #[test]
    fn resolve_affinity_smt_sibling_pair_skips_partial_cores() {
        let vmt = crate::vmm::topology::Topology::new(1, 1, 2, 2);
        let t = crate::topology::TestTopology::from_vm_topology(&vmt);
        let cpuset: BTreeSet<usize> = [2usize, 3].into_iter().collect();
        match resolve_affinity_for_cgroup(&AffinityIntent::SmtSiblingPair, Some(&cpuset), &t)
            .unwrap()
        {
            ResolvedAffinity::Fixed(cpus) => {
                assert_eq!(
                    cpus,
                    [2usize, 3].into_iter().collect(),
                    "SmtSiblingPair must skip core 0 when cpuset excludes one of its \
                     siblings and pick the next eligible pair"
                );
            }
            other => panic!("expected Fixed({{2, 3}}), got {:?}", other),
        }
    }

    /// `threads_per_core == 1` means no SMT. The resolver must
    /// return an explicit error rather than silently degrading to
    /// `None` — running an SMT-pair workload without true SMT
    /// produces a misleading test result.
    #[test]
    fn resolve_affinity_smt_sibling_pair_errors_without_smt() {
        let vmt = crate::vmm::topology::Topology::new(1, 1, 4, 1);
        let t = crate::topology::TestTopology::from_vm_topology(&vmt);
        let err = resolve_affinity_for_cgroup(&AffinityIntent::SmtSiblingPair, None, &t)
            .expect_err("threads_per_core=1 must produce an error, not silent fallback");
        let msg = err.to_string();
        assert!(
            msg.contains("SmtSiblingPair"),
            "diagnostic must name the variant, got: {msg}"
        );
        assert!(
            msg.contains("two SMT siblings"),
            "diagnostic must explain the missing precondition, got: {msg}"
        );
    }

    /// When the cpuset isolates each sibling onto a different
    /// cgroup (e.g. `{0, 2}` keeps one sibling from each core but
    /// no full pair), the resolver must error rather than silently
    /// producing a 1-CPU set.
    #[test]
    fn resolve_affinity_smt_sibling_pair_errors_when_cpuset_breaks_pairs() {
        let vmt = crate::vmm::topology::Topology::new(1, 1, 2, 2);
        let t = crate::topology::TestTopology::from_vm_topology(&vmt);
        let cpuset: BTreeSet<usize> = [0usize, 2].into_iter().collect();
        let err = resolve_affinity_for_cgroup(&AffinityIntent::SmtSiblingPair, Some(&cpuset), &t)
            .expect_err("cpuset that breaks every sibling pair must error");
        let msg = err.to_string();
        assert!(
            msg.contains("SmtSiblingPair"),
            "diagnostic must name the variant, got: {msg}"
        );
    }

    /// `TestTopology::synthetic` builds an empty per-core sibling
    /// map (no SMT info), so `SmtSiblingPair` must error there as
    /// well — the resolver depends on `LlcInfo::cores`, not on the
    /// raw CPU list.
    #[test]
    fn resolve_affinity_smt_sibling_pair_errors_on_synthetic_topology() {
        let t = crate::topology::TestTopology::synthetic(8, 2);
        let err = resolve_affinity_for_cgroup(&AffinityIntent::SmtSiblingPair, None, &t)
            .expect_err("synthetic topology has no per-core sibling data — must error");
        let msg = err.to_string();
        assert!(
            msg.contains("SmtSiblingPair"),
            "diagnostic must name the variant, got: {msg}"
        );
    }

    #[test]
    fn split_half_even() {
        let t = crate::topology::TestTopology::synthetic(8, 2);
        let ctx_cg = crate::cgroup::CgroupManager::new("/nonexistent");
        let ctx = Ctx {
            cgroups: &ctx_cg,
            topo: &t,
            duration: std::time::Duration::from_secs(1),
            workers_per_cgroup: 4,
            sched_pid: None,
            settle: Duration::from_millis(3000),
            work_type_override: None,
            assert: assert::Assert::default_checks(),
            wait_for_map_write: false,
        };
        let (a, b) = split_half(&ctx);
        // Last CPU reserved for cgroup 0 → 7 usable, split 3/4
        assert_eq!(a.len() + b.len(), 7);
        assert!(a.intersection(&b).count() == 0, "halves should not overlap");
    }

    #[test]
    fn split_half_small() {
        let t = crate::topology::TestTopology::synthetic(2, 1);
        let ctx_cg = crate::cgroup::CgroupManager::new("/nonexistent");
        let ctx = Ctx {
            cgroups: &ctx_cg,
            topo: &t,
            duration: std::time::Duration::from_secs(1),
            workers_per_cgroup: 1,
            sched_pid: None,
            settle: Duration::from_millis(3000),
            work_type_override: None,
            assert: assert::Assert::default_checks(),
            wait_for_map_write: false,
        };
        let (a, b) = split_half(&ctx);
        assert_eq!(a.len() + b.len(), 2);
    }

    #[test]
    fn dfl_wl_propagates_workers() {
        let t = crate::topology::TestTopology::synthetic(8, 2);
        let ctx_cg = crate::cgroup::CgroupManager::new("/nonexistent");
        let ctx = Ctx {
            cgroups: &ctx_cg,
            topo: &t,
            duration: std::time::Duration::from_secs(1),
            workers_per_cgroup: 7,
            sched_pid: None,
            settle: Duration::from_millis(3000),
            work_type_override: None,
            assert: assert::Assert::default_checks(),
            wait_for_map_write: false,
        };
        let wl = dfl_wl(&ctx);
        assert_eq!(wl.num_workers, 7);
        assert!(matches!(wl.work_type, WorkType::SpinWait));
    }

    #[test]
    fn process_alive_self_is_true() {
        let pid: libc::pid_t = unsafe { libc::getpid() };
        assert!(process_alive(pid));
    }

    /// `Ctx::active_sched_pid` filters `Some(0)` — and any negative
    /// pid — out of the "configured scheduler" set. Without the
    /// filter, a `Some(0)` would reach `process_alive(0)`, which
    /// returns `false`, and the three liveness call sites
    /// (`run_scenario` post-settle, workload-phase polling,
    /// `setup_cgroups` post-settle) would each raise a spurious
    /// scheduler-died diagnostic on a test that never had a
    /// scheduler to begin with. The "configured" definition has to
    /// agree across all three sites, so the gate lives on `Ctx`
    /// rather than being re-inlined three times.
    #[test]
    fn ctx_active_sched_pid_treats_nonpositive_as_unconfigured() {
        let cg = crate::cgroup::CgroupManager::new("/nonexistent");
        let topo = crate::topology::TestTopology::synthetic(1, 1);

        // Some(0) — the most likely mistake when a caller confuses
        // `Ctx::sched_pid` (Option<pid_t>) with the workload TLS's
        // 0-sentinel.
        let ctx_zero = Ctx::builder(&cg, &topo).sched_pid(Some(0)).build();
        assert_eq!(
            ctx_zero.sched_pid,
            Some(0),
            "builder must preserve the literal value — the gate lives in the accessor",
        );
        assert_eq!(
            ctx_zero.active_sched_pid(),
            None,
            "Some(0) must be treated as unconfigured, otherwise the liveness \
             bails fire on tests that never ran a scheduler",
        );

        // Negative pids: `kill(negative, sig)` is a process-group
        // broadcast, not a live-process query — also unconfigured.
        let ctx_neg = Ctx::builder(&cg, &topo).sched_pid(Some(-1)).build();
        assert_eq!(
            ctx_neg.active_sched_pid(),
            None,
            "negative pid must be treated as unconfigured",
        );

        // `pid_t::MIN` — the most-negative representable value.
        // Pin the lower boundary explicitly so a future filter
        // change that accidentally uses `p >= 0` (a common mis-read
        // of "non-negative") keeps treating the edge as
        // unconfigured. `Some(0)` above covers the `p > 0` vs
        // `p >= 0` distinction; `pid_t::MIN` covers "does the
        // `p > 0` predicate survive an overflow-adjacent input?"
        let ctx_min = Ctx::builder(&cg, &topo)
            .sched_pid(Some(libc::pid_t::MIN))
            .build();
        assert_eq!(
            ctx_min.active_sched_pid(),
            None,
            "pid_t::MIN must be treated as unconfigured — the filter \
             is `p > 0`, and the most-negative pid_t stays unconfigured \
             under that predicate by construction",
        );

        // Sanity: a positive pid survives the filter.
        let ctx_pos = Ctx::builder(&cg, &topo).sched_pid(Some(1234)).build();
        assert_eq!(
            ctx_pos.active_sched_pid(),
            Some(1234),
            "positive pid must pass through unchanged",
        );

        // `pid_t::MAX` — the most-positive representable value.
        // Linux caps live pids at PID_MAX_LIMIT (2^22) so
        // `pid_t::MAX` (2^31 - 1) cannot be allocated, but the
        // filter operates on pure value-polarity rather than
        // kernel allocability. A positive value — even one
        // guaranteed not to exist — must pass through unchanged,
        // because the liveness callsites downstream
        // (`process_alive`) are what will see the pid and report
        // it as dead; the `active_sched_pid` filter is a
        // configured-vs-unconfigured gate, not a liveness gate.
        let ctx_max = Ctx::builder(&cg, &topo)
            .sched_pid(Some(libc::pid_t::MAX))
            .build();
        assert_eq!(
            ctx_max.active_sched_pid(),
            Some(libc::pid_t::MAX),
            "pid_t::MAX must pass the filter — `p > 0` accepts it. \
             Liveness determination is the responsibility of the \
             downstream `process_alive` call, not this accessor.",
        );

        // Sanity: None stays None.
        let ctx_none = Ctx::builder(&cg, &topo).sched_pid(None).build();
        assert_eq!(
            ctx_none.active_sched_pid(),
            None,
            "None must pass through unchanged",
        );
    }

    #[test]
    fn process_alive_zero_is_false() {
        // kill(0, sig) targets the caller's process group, so without
        // an explicit guard kill(0, 0) succeeds and would falsely
        // report "process 0" as alive.
        assert!(!process_alive(0));
    }

    #[test]
    fn process_alive_negative_is_false() {
        // kill(negative, sig) targets a process group (or, for -1,
        // every process the caller can signal). A pid_t <= 0 is
        // never a live-process query and must return false.
        assert!(!process_alive(-1));
        assert!(!process_alive(libc::pid_t::MIN));
    }

    #[test]
    fn process_alive_nonexistent_pid() {
        // Use `pid_t::MAX` (2^31 - 1) as a guaranteed-non-existent pid.
        // Linux's `pid_max` is capped at 2^22 (4,194,304) per
        // `include/linux/threads.h` (PID_MAX_LIMIT), so any pid above
        // that threshold cannot be allocated — kill(2) returns ESRCH
        // unconditionally. The previous formulation (fork + waitpid +
        // probe-the-freed-pid) had a PID-reuse race: between waitpid
        // returning and process_alive probing, the kernel could
        // allocate the freed pid to a concurrent fork() on the host
        // (heavily-forking CI runners, container hosts, etc.) and
        // turn this test into a flake. Using a pid above PID_MAX_LIMIT
        // removes the race entirely — no syscall ordering can place a
        // live process on a pid the kernel refuses to allocate.
        assert!(!process_alive(libc::pid_t::MAX));
    }

    #[test]
    fn cgroup_group_new_empty() {
        let cg = crate::cgroup::CgroupManager::new("/nonexistent");
        let group = CgroupGroup::new(&cg);
        assert!(group.names().is_empty());
    }

    // -- resolve_affinity_for_cgroup edge cases --

    #[test]
    fn resolve_affinity_single_cpu_with_cpuset() {
        let t = crate::topology::TestTopology::synthetic(4, 1);
        // Cpuset restricts to CPUs {2,3}: SingleCpu picks first in cpuset.
        let cpusets: Vec<BTreeSet<usize>> = vec![[2, 3].into_iter().collect()];
        match resolve_affinity_for_cgroup(&AffinityIntent::SingleCpu, cpusets.first(), &t).unwrap()
        {
            ResolvedAffinity::SingleCpu(c) => assert_eq!(c, 2),
            other => panic!("expected SingleCpu, got {:?}", other),
        }
    }

    #[test]
    fn resolve_affinity_llc_aligned_picks_best_overlap() {
        let t = crate::topology::TestTopology::synthetic(8, 2);
        // Cpuset spans both LLCs but has more CPUs in LLC 1.
        // LLC 0 = {0,1,2,3}, LLC 1 = {4,5,6,7}.
        // Cpuset = {3, 4, 5, 6, 7}: LLC 1 has 4 CPUs in cpuset, LLC 0 has 1.
        let cpusets: Vec<BTreeSet<usize>> = vec![[3, 4, 5, 6, 7].into_iter().collect()];
        match resolve_affinity_for_cgroup(&AffinityIntent::LlcAligned, cpusets.first(), &t).unwrap()
        {
            ResolvedAffinity::Fixed(cpus) => {
                // LLC 1 has best overlap; result is intersection {4,5,6,7}.
                assert_eq!(cpus, [4, 5, 6, 7].into_iter().collect());
            }
            other => panic!("expected Fixed, got {:?}", other),
        }
    }

    #[test]
    fn resolve_num_workers_zero_rejected_with_label() {
        let w = WorkSpec {
            num_workers: Some(0),
            ..Default::default()
        };
        let err = resolve_num_workers(&w, 4, "victim").unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("cgroup 'victim'"),
            "label must appear in error: {msg}"
        );
        assert!(
            msg.contains("num_workers=0"),
            "error must name the offending field: {msg}"
        );
    }

    #[test]
    fn resolve_num_workers_zero_default_also_rejected() {
        // When num_workers is unset AND the default is 0, still reject.
        let w = WorkSpec {
            num_workers: None,
            ..Default::default()
        };
        assert!(resolve_num_workers(&w, 0, "cg").is_err());
    }

    #[test]
    fn resolve_num_workers_falls_back_to_default() {
        let w = WorkSpec {
            num_workers: None,
            ..Default::default()
        };
        assert_eq!(resolve_num_workers(&w, 3, "cg").unwrap(), 3);
    }

    #[test]
    fn resolve_num_workers_explicit_wins_over_default() {
        let w = WorkSpec {
            num_workers: Some(7),
            ..Default::default()
        };
        assert_eq!(resolve_num_workers(&w, 3, "cg").unwrap(), 7);
    }

    /// Minimal `CgroupOps` double for `CgroupGroup::drop` error-path
    /// tests. Injects a caller-supplied `io::Error` into the
    /// `remove_cgroup` chain so the test can cover both the ENOENT
    /// (benign TOCTOU) branch and the non-ENOENT warn branch without
    /// touching cgroupfs. Every other trait method is a no-op — the
    /// drop path only calls `remove_cgroup`.
    struct DropErrCgroupOps {
        parent: std::path::PathBuf,
        remove_kind: std::io::ErrorKind,
        raw_os_error: Option<i32>,
        remove_calls: std::sync::Mutex<Vec<String>>,
    }

    impl DropErrCgroupOps {
        fn new(kind: std::io::ErrorKind, raw: Option<i32>) -> Self {
            Self {
                parent: std::path::PathBuf::from("/mock/cgroup"),
                remove_kind: kind,
                raw_os_error: raw,
                remove_calls: std::sync::Mutex::new(Vec::new()),
            }
        }
        fn calls(&self) -> Vec<String> {
            self.remove_calls.lock().unwrap().clone()
        }
    }

    impl crate::cgroup::CgroupOps for DropErrCgroupOps {
        fn parent_path(&self) -> &std::path::Path {
            &self.parent
        }
        fn setup(&self, _: &std::collections::BTreeSet<crate::cgroup::Controller>) -> Result<()> {
            Ok(())
        }
        fn create_cgroup(&self, _: &str) -> Result<()> {
            Ok(())
        }
        fn remove_cgroup(&self, name: &str) -> Result<()> {
            self.remove_calls.lock().unwrap().push(name.to_string());
            let io = match self.raw_os_error {
                Some(errno) => std::io::Error::from_raw_os_error(errno),
                None => std::io::Error::from(self.remove_kind),
            };
            // Wrap in anyhow::Context the same way the real CgroupManager
            // does, so `err.root_cause().downcast_ref::<io::Error>()`
            // traverses the chain identically to production.
            Err(anyhow::Error::new(io).context("remove_dir cgroup"))
        }
        fn set_cpuset(&self, _: &str, _: &BTreeSet<usize>) -> Result<()> {
            Ok(())
        }
        fn clear_cpuset(&self, _: &str) -> Result<()> {
            Ok(())
        }
        fn set_cpuset_mems(&self, _: &str, _: &BTreeSet<usize>) -> Result<()> {
            Ok(())
        }
        fn clear_cpuset_mems(&self, _: &str) -> Result<()> {
            Ok(())
        }
        fn set_cpu_max(&self, _: &str, _: Option<u64>, _: u64) -> Result<()> {
            Ok(())
        }
        fn set_cpu_weight(&self, _: &str, _: u32) -> Result<()> {
            Ok(())
        }
        fn set_memory_max(&self, _: &str, _: Option<u64>) -> Result<()> {
            Ok(())
        }
        fn set_memory_high(&self, _: &str, _: Option<u64>) -> Result<()> {
            Ok(())
        }
        fn set_memory_low(&self, _: &str, _: Option<u64>) -> Result<()> {
            Ok(())
        }
        fn set_io_weight(&self, _: &str, _: u16) -> Result<()> {
            Ok(())
        }
        fn set_freeze(&self, _: &str, _: bool) -> Result<()> {
            Ok(())
        }
        fn set_pids_max(&self, _: &str, _: Option<u64>) -> Result<()> {
            Ok(())
        }
        fn set_memory_swap_max(&self, _: &str, _: Option<u64>) -> Result<()> {
            Ok(())
        }
        fn move_task(&self, _: &str, _: libc::pid_t) -> Result<()> {
            Ok(())
        }
        fn move_tasks(&self, _: &str, _: &[libc::pid_t]) -> Result<()> {
            Ok(())
        }
        fn clear_subtree_control(&self, _: &str) -> Result<()> {
            Ok(())
        }
        fn drain_tasks(&self, _: &str) -> Result<()> {
            Ok(())
        }
        fn cleanup_all(&self) -> Result<()> {
            Ok(())
        }
    }

    /// Drop must iterate every tracked name regardless of error kind —
    /// ENOENT is classified benign and dropped silently; any other
    /// error path (EBUSY, EACCES, generic IO) takes the warn branch.
    /// Neither path may panic (panic in Drop under `panic = "abort"`
    /// aborts the whole process). This test is a panic-free crash-test
    /// plus a call-count pin: every tracked cgroup must see exactly
    /// one `remove_cgroup` call, in reverse-insertion order, in every
    /// branch — so the iteration contract doesn't silently shrink.
    #[test]
    fn cgroup_group_drop_is_panic_free_on_every_error_kind() {
        for (label, kind, raw) in [
            ("ENOENT", std::io::ErrorKind::NotFound, Some(libc::ENOENT)),
            ("EBUSY", std::io::ErrorKind::Other, Some(libc::EBUSY)),
            (
                "EACCES",
                std::io::ErrorKind::PermissionDenied,
                Some(libc::EACCES),
            ),
            ("generic-IO", std::io::ErrorKind::Other, None),
        ] {
            let mock = DropErrCgroupOps::new(kind, raw);
            {
                let mut group = CgroupGroup::new(&mock);
                group.names.push("child-a".to_string());
                group.names.push("child-b".to_string());
                // drop at end of scope must not panic.
            }
            let calls = mock.calls();
            // Reverse-insertion order: child-b first, then child-a —
            // pins the nested-cleanup invariant documented in Drop.
            assert_eq!(
                calls,
                vec!["child-b".to_string(), "child-a".to_string()],
                "[{label}] Drop must call remove_cgroup for every tracked name in reverse order",
            );
        }
    }

    /// `is_io_not_found` is the one-place classifier for the "benign
    /// TOCTOU ENOENT" branch in both `CgroupGroup::drop` and
    /// `Op::RemoveCgroup`. Pin the contract: NotFound → true; every
    /// other kind → false. A regression that mis-classifies EBUSY or
    /// EACCES as "not found" would silently swallow a real teardown
    /// failure.
    #[test]
    fn is_io_not_found_matches_only_notfound() {
        let wrap = |k: std::io::ErrorKind| -> anyhow::Error {
            anyhow::Error::new(std::io::Error::from(k)).context("wrap")
        };
        assert!(is_io_not_found(&wrap(std::io::ErrorKind::NotFound)));
        assert!(!is_io_not_found(&wrap(
            std::io::ErrorKind::PermissionDenied
        )));
        assert!(!is_io_not_found(&wrap(std::io::ErrorKind::Other)));
        // anyhow::anyhow! without an underlying io::Error must not
        // look like NotFound even if the message contains "not found".
        let no_io = anyhow::anyhow!("cgroup not found in parent");
        assert!(!is_io_not_found(&no_io));
    }

    /// `remove_cgroup_errno_hint` gives actionable remediation for
    /// the two errnos users can fix (EBUSY, EACCES) and stays quiet
    /// for everything else. The hint strings end up rendered into
    /// warn output; pin their presence/absence so a regression that
    /// drops the hint (or hallucinates one) surfaces in test output.
    #[test]
    fn remove_cgroup_errno_hint_covers_ebusy_and_eacces() {
        let busy =
            anyhow::Error::new(std::io::Error::from_raw_os_error(libc::EBUSY)).context("wrap");
        let acces =
            anyhow::Error::new(std::io::Error::from_raw_os_error(libc::EACCES)).context("wrap");
        let enotempty =
            anyhow::Error::new(std::io::Error::from_raw_os_error(libc::ENOTEMPTY)).context("wrap");
        let non_io = anyhow::anyhow!("not an io error");

        assert!(
            remove_cgroup_errno_hint(&busy)
                .is_some_and(|h| h.contains("EBUSY") && h.contains("drain")),
            "EBUSY hint must name the errno and the drain remediation",
        );
        assert!(
            remove_cgroup_errno_hint(&acces)
                .is_some_and(|h| h.contains("EACCES") && h.contains("permission")),
            "EACCES hint must name the errno and the permission angle",
        );
        assert_eq!(
            remove_cgroup_errno_hint(&enotempty),
            None,
            "unclassified errnos must yield no hint so warn stays terse",
        );
        assert_eq!(
            remove_cgroup_errno_hint(&non_io),
            None,
            "non-io root causes must yield no hint",
        );
    }

    // -- flatten_for_spawn / intent_for_spawn coverage --
    //
    // Every arm of `flatten_for_spawn` must round-trip a known
    // [`ResolvedAffinity`] into the matching [`AffinityIntent`]
    // shape. The flatten step gates the scenario engine's output
    // against the spawn-time gate in `workload::resolve_spawn_affinity`.
    //
    // Post-#48 contract: `resolve_affinity_for_cgroup` bails on every
    // path that would produce an empty pool — empty `Fixed`,
    // `Random { from: empty }`, `Random { count: 0 }` — so
    // `flatten_for_spawn` upholds that contract with `unreachable!()`
    // on those arms. The `_panics` tests pin the invariant: a future
    // caller that bypasses the resolver and constructs an empty pool
    // directly trips the panic at the construction site, never
    // silently degrades to `Inherit`.

    #[test]
    fn flatten_for_spawn_none_to_inherit() {
        let out = flatten_for_spawn(ResolvedAffinity::None);
        assert!(
            matches!(out, AffinityIntent::Inherit),
            "ResolvedAffinity::None must flatten to Inherit, got {out:?}"
        );
    }

    #[test]
    fn flatten_for_spawn_fixed_to_exact() {
        let set: BTreeSet<usize> = [1usize, 3, 5].into_iter().collect();
        let out = flatten_for_spawn(ResolvedAffinity::Fixed(set.clone()));
        match out {
            AffinityIntent::Exact(got) => {
                assert_eq!(got, set, "Fixed payload must round-trip into Exact");
            }
            other => panic!("expected Exact, got {other:?}"),
        }
    }

    /// Post-#48 invariant: `resolve_affinity_for_cgroup` bails on
    /// every path that would produce an empty `Fixed`, so
    /// `flatten_for_spawn` upholds that contract with `unreachable!()`.
    /// This test pins that contract — if a future caller bypasses the
    /// resolver and constructs `ResolvedAffinity::Fixed(BTreeSet::new())`
    /// directly, the panic surfaces the bug at the call site instead of
    /// silently degrading to `Inherit`.
    #[test]
    #[should_panic(
        expected = "ResolvedAffinity::Fixed(empty) reached flatten_for_spawn"
    )]
    fn flatten_for_spawn_fixed_empty_panics() {
        let _ = flatten_for_spawn(ResolvedAffinity::Fixed(BTreeSet::new()));
    }

    #[test]
    fn flatten_for_spawn_single_cpu_to_exact_singleton() {
        let out = flatten_for_spawn(ResolvedAffinity::SingleCpu(7));
        match out {
            AffinityIntent::Exact(got) => {
                let expected: BTreeSet<usize> = [7usize].into_iter().collect();
                assert_eq!(got, expected, "SingleCpu must flatten to a 1-CPU Exact set");
            }
            other => panic!("expected Exact({{7}}), got {other:?}"),
        }
    }

    #[test]
    fn flatten_for_spawn_random_to_random_subset() {
        let from: BTreeSet<usize> = [0usize, 1, 2, 3].into_iter().collect();
        let out = flatten_for_spawn(ResolvedAffinity::Random {
            from: from.clone(),
            count: 2,
        });
        match out {
            AffinityIntent::RandomSubset {
                from: got_from,
                count: got_count,
            } => {
                assert_eq!(got_from, from, "Random.from must round-trip verbatim");
                assert_eq!(got_count, 2, "Random.count must round-trip verbatim");
            }
            other => panic!("expected RandomSubset, got {other:?}"),
        }
    }

    /// Post-#48 invariant: `resolve_affinity_for_cgroup` bails on
    /// `RandomSubset` empty pools (after cpuset intersection), so
    /// `flatten_for_spawn` upholds that contract with `unreachable!()`.
    /// Pins the contract — see `flatten_for_spawn_fixed_empty_panics`.
    #[test]
    #[should_panic(
        expected = "reached flatten_for_spawn with count==0 or empty pool"
    )]
    fn flatten_for_spawn_random_empty_pool_panics() {
        let _ = flatten_for_spawn(ResolvedAffinity::Random {
            from: BTreeSet::new(),
            count: 4,
        });
    }

    /// Post-#48 invariant: `resolve_affinity_for_cgroup` bails on
    /// `RandomSubset { count: 0 }`. Pins the contract — see
    /// `flatten_for_spawn_fixed_empty_panics`.
    #[test]
    #[should_panic(
        expected = "reached flatten_for_spawn with count==0 or empty pool"
    )]
    fn flatten_for_spawn_random_zero_count_panics() {
        let from: BTreeSet<usize> = [0usize, 1, 2, 3].into_iter().collect();
        let _ = flatten_for_spawn(ResolvedAffinity::Random { from, count: 0 });
    }

    /// End-to-end: `intent_for_spawn` chains
    /// `resolve_affinity_for_cgroup` into `flatten_for_spawn`. Verify
    /// the full pipeline produces a spawn-gate-acceptable intent for
    /// each top-level [`AffinityIntent`] variant on the happy path.
    /// Topology-aware variants (`SingleCpu`, `CrossCgroup`,
    /// `SmtSiblingPair`, `LlcAligned`) flatten to `Exact`; `Inherit`
    /// and `RandomSubset` (with a valid pool) round-trip. The bail
    /// propagation for unsatisfiable inputs is covered by the
    /// dedicated `intent_for_spawn_propagates_*_bail` sibling tests.
    #[test]
    fn intent_for_spawn_full_pipeline() {
        // Use a real VM topology so the per-LLC sibling map is
        // populated — `synthetic` leaves `LlcInfo::cores` empty,
        // which forces the SmtSiblingPair arm into its no-SMT
        // error branch. `Topology::new(numa, llcs, cores, threads)`
        // with threads=2 produces 2 SMT siblings per core.
        let vmt = crate::vmm::topology::Topology::new(1, 2, 2, 2);
        let t = crate::topology::TestTopology::from_vm_topology(&vmt);

        // Inherit → Inherit
        let out = intent_for_spawn(&AffinityIntent::Inherit, None, &t).unwrap();
        assert!(
            matches!(out, AffinityIntent::Inherit),
            "Inherit must round-trip, got {out:?}"
        );

        // SingleCpu → Exact({some_cpu})
        let out = intent_for_spawn(&AffinityIntent::SingleCpu, None, &t).unwrap();
        match out {
            AffinityIntent::Exact(set) => {
                assert_eq!(set.len(), 1, "SingleCpu flattens to a 1-CPU Exact set");
            }
            other => panic!("expected Exact, got {other:?}"),
        }

        // CrossCgroup → Exact(<all CPUs>)
        let out = intent_for_spawn(&AffinityIntent::CrossCgroup, None, &t).unwrap();
        match out {
            AffinityIntent::Exact(set) => {
                assert_eq!(set.len(), 8, "CrossCgroup flattens to all-CPU Exact set");
            }
            other => panic!("expected Exact, got {other:?}"),
        }

        // SmtSiblingPair → Exact({sibling_a, sibling_b})
        let out = intent_for_spawn(&AffinityIntent::SmtSiblingPair, None, &t).unwrap();
        match out {
            AffinityIntent::Exact(set) => {
                assert_eq!(set.len(), 2, "SmtSiblingPair flattens to a 2-CPU Exact set");
                // First core's siblings are CPUs 0 and 1 in the
                // sequential VM topology (cores 0..2 within LLC 0:
                // core 0 = {0, 1}, core 1 = {2, 3}).
                assert_eq!(
                    set,
                    [0usize, 1].into_iter().collect(),
                    "SmtSiblingPair must pick the first core's siblings"
                );
            }
            other => panic!("expected Exact, got {other:?}"),
        }

        // LlcAligned → Exact(<LLC's CPUs>) — picks the LLC with most
        // overlap against the cpuset (or `all_cpuset` when no cpuset
        // is active). On the 1-NUMA / 2-LLC / 2-core / 2-thread
        // topology, each LLC owns 4 CPUs.
        let out = intent_for_spawn(&AffinityIntent::LlcAligned, None, &t).unwrap();
        match out {
            AffinityIntent::Exact(set) => {
                assert_eq!(set.len(), 4, "LlcAligned flattens to one LLC's worth of CPUs");
            }
            other => panic!("expected Exact, got {other:?}"),
        }

        // Exact(non-empty) → Exact(round-trip when within cpuset)
        let exact_cpus: BTreeSet<usize> = [0usize, 2, 4].into_iter().collect();
        let out = intent_for_spawn(
            &AffinityIntent::Exact(exact_cpus.clone()),
            None,
            &t,
        )
        .unwrap();
        match out {
            AffinityIntent::Exact(set) => {
                assert_eq!(set, exact_cpus, "Exact round-trip must preserve the CPU set");
            }
            other => panic!("expected Exact, got {other:?}"),
        }

        // RandomSubset with valid pool → RandomSubset round-trip
        let pool: BTreeSet<usize> = [0usize, 1, 2, 3].into_iter().collect();
        let intent = AffinityIntent::random_subset(pool.iter().copied(), 2);
        let out = intent_for_spawn(&intent, None, &t).unwrap();
        match out {
            AffinityIntent::RandomSubset { from, count } => {
                assert_eq!(from, pool, "RandomSubset.from must round-trip");
                assert_eq!(count, 2, "RandomSubset.count must round-trip");
            }
            other => panic!("expected RandomSubset, got {other:?}"),
        }
    }

    /// Post-#48 invariant: `intent_for_spawn` must propagate every
    /// `Err` from the inner `resolve_affinity_for_cgroup` rather than
    /// swallowing it. These tests pin the `?`-propagation path for
    /// each unsatisfiable bail.
    #[test]
    fn intent_for_spawn_propagates_random_empty_intersection_bail() {
        let vmt = crate::vmm::topology::Topology::new(1, 2, 2, 2);
        let t = crate::topology::TestTopology::from_vm_topology(&vmt);
        let empty_cpuset: BTreeSet<usize> = BTreeSet::new();
        let intent = AffinityIntent::random_subset(t.all_cpus().iter().copied(), 1);
        let err = intent_for_spawn(&intent, Some(&empty_cpuset), &t)
            .expect_err("RandomSubset with empty cpuset intersection must bail");
        let msg = err.to_string();
        assert!(
            msg.contains("AffinityIntent::RandomSubset")
                && msg.contains("after intersecting"),
            "bail diag should name the intent and the intersection, got: {msg}"
        );
    }

    #[test]
    fn intent_for_spawn_propagates_random_zero_count_bail() {
        let vmt = crate::vmm::topology::Topology::new(1, 2, 2, 2);
        let t = crate::topology::TestTopology::from_vm_topology(&vmt);
        let pool: BTreeSet<usize> = [0usize, 1].into_iter().collect();
        let intent = AffinityIntent::random_subset(pool.iter().copied(), 0);
        let err = intent_for_spawn(&intent, None, &t)
            .expect_err("RandomSubset { count: 0 } must bail");
        let msg = err.to_string();
        assert!(
            msg.contains("count=0"),
            "bail diag should name the count=0 condition, got: {msg}"
        );
    }

    #[test]
    fn intent_for_spawn_propagates_exact_empty_bail() {
        let vmt = crate::vmm::topology::Topology::new(1, 2, 2, 2);
        let t = crate::topology::TestTopology::from_vm_topology(&vmt);
        let intent = AffinityIntent::Exact(BTreeSet::new());
        let err = intent_for_spawn(&intent, None, &t)
            .expect_err("Exact(empty) must bail");
        let msg = err.to_string();
        assert!(
            msg.contains("AffinityIntent::Exact(BTreeSet::new())"),
            "bail diag should name the empty Exact, got: {msg}"
        );
    }

    #[test]
    fn intent_for_spawn_propagates_single_cpu_empty_cpuset_bail() {
        let vmt = crate::vmm::topology::Topology::new(1, 2, 2, 2);
        let t = crate::topology::TestTopology::from_vm_topology(&vmt);
        let empty_cpuset: BTreeSet<usize> = BTreeSet::new();
        let err = intent_for_spawn(&AffinityIntent::SingleCpu, Some(&empty_cpuset), &t)
            .expect_err("SingleCpu with empty cpuset must bail");
        let msg = err.to_string();
        assert!(
            msg.contains("AffinityIntent::SingleCpu")
                && msg.contains("empty"),
            "bail diag should name the intent and the empty cpuset, got: {msg}"
        );
    }

    #[test]
    fn settled_hold_returns_settle_plus_fraction_of_duration() {
        let cg = crate::cgroup::CgroupManager::new("/nonexistent");
        let topo = crate::topology::TestTopology::synthetic(1, 1);
        let ctx = Ctx::builder(&cg, &topo)
            .settle(Duration::from_millis(200))
            .duration(Duration::from_secs(1))
            .build();
        assert_eq!(
            ctx.settled_hold(0.5),
            crate::scenario::ops::HoldSpec::fixed(Duration::from_millis(700)),
        );
    }

    #[test]
    fn settled_hold_full_fraction_returns_settle_plus_full_duration() {
        let cg = crate::cgroup::CgroupManager::new("/nonexistent");
        let topo = crate::topology::TestTopology::synthetic(1, 1);
        let ctx = Ctx::builder(&cg, &topo)
            .settle(Duration::from_millis(100))
            .duration(Duration::from_secs(2))
            .build();
        assert_eq!(
            ctx.settled_hold(1.0),
            crate::scenario::ops::HoldSpec::fixed(Duration::from_millis(2100)),
        );
    }

    #[test]
    fn settled_hold_zero_fraction_returns_settle_only() {
        let cg = crate::cgroup::CgroupManager::new("/nonexistent");
        let topo = crate::topology::TestTopology::synthetic(1, 1);
        let ctx = Ctx::builder(&cg, &topo)
            .settle(Duration::from_millis(500))
            .duration(Duration::from_secs(1))
            .build();
        assert_eq!(
            ctx.settled_hold(0.0),
            crate::scenario::ops::HoldSpec::fixed(Duration::from_millis(500)),
        );
    }

    #[test]
    fn settled_hold_third_fraction_matches_integer_division() {
        let cg = crate::cgroup::CgroupManager::new("/nonexistent");
        let topo = crate::topology::TestTopology::synthetic(1, 1);
        let ctx = Ctx::builder(&cg, &topo)
            .settle(Duration::from_millis(100))
            .duration(Duration::from_secs(3))
            .build();
        // 3s * (1/3) under mul_f64 may produce 2.999_999_999s due to
        // f64 representation of 1/3. The hold-evaluation boundary
        // tolerates the sub-nanosecond drift; the test pins that
        // the result stays within 1 nanosecond of the integer-div
        // formulation (3s / 3 = 1s exact, + 100ms settle = 1100ms).
        let hold = ctx.settled_hold(1.0 / 3.0);
        let crate::scenario::ops::HoldSpec::Fixed(d) = hold else {
            panic!("expected Fixed variant, got {hold:?}");
        };
        let expected = Duration::from_millis(1100);
        let diff = d.abs_diff(expected);
        assert!(
            diff <= Duration::from_nanos(1),
            "settled_hold(1.0/3.0) drift > 1ns: {d:?} vs {expected:?}",
        );
    }

    #[test]
    #[should_panic(expected = "cannot convert float seconds to Duration")]
    fn settled_hold_panics_on_nan() {
        let cg = crate::cgroup::CgroupManager::new("/nonexistent");
        let topo = crate::topology::TestTopology::synthetic(1, 1);
        let ctx = Ctx::builder(&cg, &topo)
            .settle(Duration::from_millis(100))
            .duration(Duration::from_secs(1))
            .build();
        let _ = ctx.settled_hold(f64::NAN);
    }

    /// `ctx.cgroup_def(name)` produces a `CgroupDef` carrying the
    /// builder's `workers_per_cgroup` value verbatim. Pin so a
    /// regression that pulled from a stale field or hardcoded a
    /// literal would surface here.
    #[test]
    fn cgroup_def_carries_workers_per_cgroup() {
        let cg = crate::cgroup::CgroupManager::new("/nonexistent");
        let topo = crate::topology::TestTopology::synthetic(2, 1);
        let ctx = Ctx::builder(&cg, &topo).workers_per_cgroup(7).build();
        let def = ctx.cgroup_def("cg_0");
        assert_eq!(def.name, "cg_0", "name must thread through verbatim");
        assert_eq!(
            def.works.len(),
            1,
            "the default workers(n) call seeds exactly one WorkSpec; \
             a regression that doubled or skipped the seeding would \
             surface here: {:?}",
            def.works,
        );
        assert_eq!(
            def.works[0].num_workers,
            Some(7),
            "WorkSpec must carry ctx.workers_per_cgroup as num_workers: {:?}",
            def.works[0],
        );
    }

    /// Default `workers_per_cgroup = 1` from `CtxBuilder` flows
    /// through unchanged. A regression that defaulted differently
    /// would surface here even when the test author doesn't set
    /// `workers_per_cgroup` explicitly.
    #[test]
    fn cgroup_def_default_workers_per_cgroup_is_one() {
        let cg = crate::cgroup::CgroupManager::new("/nonexistent");
        let topo = crate::topology::TestTopology::synthetic(2, 1);
        let ctx = Ctx::builder(&cg, &topo).build();
        let def = ctx.cgroup_def("cg_default");
        assert_eq!(def.works[0].num_workers, Some(1));
    }

    /// The helper accepts both `&'static str` and `String` for the
    /// name argument (via `Into<Cow<'static, str>>`), matching
    /// `CgroupDef::named`'s signature so the migration is
    /// type-compatible at every call site.
    #[test]
    fn cgroup_def_accepts_static_str_and_string() {
        let cg = crate::cgroup::CgroupManager::new("/nonexistent");
        let topo = crate::topology::TestTopology::synthetic(2, 1);
        let ctx = Ctx::builder(&cg, &topo).build();
        let from_static = ctx.cgroup_def("static_name");
        let from_string = ctx.cgroup_def(String::from("owned_name"));
        assert_eq!(from_static.name, "static_name");
        assert_eq!(from_string.name, "owned_name");
    }

    /// Helper-returned [`CgroupDef`] chains additional builders
    /// (`.cpuset`, `.work_type`) — the docstring promises this
    /// composability so the migration can collapse multi-line
    /// chains. A regression to a newtype wrapper or `&CgroupDef`
    /// return that broke move-based chaining would surface here.
    #[test]
    fn cgroup_def_chains_further_builders() {
        use crate::scenario::ops::CpusetSpec;
        use crate::workload::WorkType;
        let cg = crate::cgroup::CgroupManager::new("/nonexistent");
        let topo = crate::topology::TestTopology::synthetic(4, 1);
        let ctx = Ctx::builder(&cg, &topo).workers_per_cgroup(3).build();
        let def = ctx
            .cgroup_def("cg_chained")
            .cpuset(CpusetSpec::range(0.0, 1.0))
            .work_type(WorkType::SpinWait);
        assert_eq!(def.name, "cg_chained");
        assert_eq!(
            def.works[0].num_workers,
            Some(3),
            "ctx default propagates through subsequent chained builders",
        );
        assert!(
            def.cpuset.is_some(),
            "cpuset chains correctly after the helper",
        );
        assert!(
            matches!(def.works[0].work_type, WorkType::SpinWait),
            "work_type chains after the helper-seeded WorkSpec; got {:?}",
            def.works[0].work_type,
        );
    }

    /// `ctx.workers_per_cgroup = 0` passes through verbatim — the
    /// helper does not reject or substitute. Matches the pre-helper
    /// inlined `CgroupDef::named(name).workers(0)` semantics so
    /// migration is lossless. For an empty move-target cgroup
    /// without workers, [`CgroupDef::named`]'s docstring directs
    /// authors to [`Op::AddCgroup`] instead — that idiomatic path is
    /// orthogonal to this helper's dedup contract.
    #[test]
    fn cgroup_def_passes_through_zero_workers_per_cgroup() {
        let cg = crate::cgroup::CgroupManager::new("/nonexistent");
        let topo = crate::topology::TestTopology::synthetic(2, 1);
        let ctx = Ctx::builder(&cg, &topo).workers_per_cgroup(0).build();
        let def = ctx.cgroup_def("zero_workers");
        assert_eq!(
            def.works[0].num_workers,
            Some(0),
            "helper preserves Ctx::workers_per_cgroup=0 verbatim; \
             for an empty move-target cgroup use Op::AddCgroup per \
             CgroupDef::named's docstring",
        );
    }

    /// A trailing `.workers(N)` overrides the helper's default —
    /// produces the same `CgroupDef` as `CgroupDef::named(name).workers(N)`.
    /// Pin so a regression that appended a SECOND WorkSpec instead
    /// of replacing num_workers on works[0] would surface here.
    #[test]
    fn cgroup_def_override_workers_replaces_default() {
        let cg = crate::cgroup::CgroupManager::new("/nonexistent");
        let topo = crate::topology::TestTopology::synthetic(2, 1);
        let ctx = Ctx::builder(&cg, &topo).workers_per_cgroup(3).build();
        let def = ctx.cgroup_def("cg_override").workers(99);
        assert_eq!(
            def.works[0].num_workers,
            Some(99),
            "explicit workers(N) overrides ctx default",
        );
        assert_eq!(def.works.len(), 1, "no second WorkSpec is added");
    }

    /// A subsequent `.work(WorkSpec::default().workers(N))` APPENDS
    /// a second WorkSpec rather than replacing the helper's seed.
    /// Pin the composition shape so a regression that overwrote
    /// works[0] (instead of pushing) would surface here.
    #[test]
    fn cgroup_def_with_second_workspec_preserves_helper_seed() {
        use crate::workload::WorkSpec;
        let cg = crate::cgroup::CgroupManager::new("/nonexistent");
        let topo = crate::topology::TestTopology::synthetic(2, 1);
        let ctx = Ctx::builder(&cg, &topo).workers_per_cgroup(3).build();
        let def = ctx
            .cgroup_def("cg_two_works")
            .work(WorkSpec::default().workers(5));
        assert_eq!(
            def.works.len(),
            2,
            "second WorkSpec appended, not replaced; helper seed stays at works[0]",
        );
        assert_eq!(
            def.works[0].num_workers,
            Some(3),
            "helper's ctx default preserved as works[0]",
        );
        assert_eq!(
            def.works[1].num_workers,
            Some(5),
            "appended WorkSpec lands as works[1]",
        );
    }
}
