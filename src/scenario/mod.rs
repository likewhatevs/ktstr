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
//! ## Builder method conventions
//!
//! Every builder type in the scenario API (Setup, Step, Backdrop,
//! WorkloadConfig, …) names its methods by what they do, not by
//! what they return. The three-prefix vocabulary is uniform across
//! the scenario surface so a reader can predict semantics from
//! the prefix alone:
//!
//! - **`with_X(arg) -> Self`** — alternate constructor that returns
//!   a fresh value with `X` already set (e.g.
//!   [`ops::Step::with_defs`], [`ops::Step::with_payload`],
//!   [`ops::Setup::with_factory`]). Distinct from `Self::new(...)`
//!   which is the base ctor; `with_X` constructors compose without
//!   reaching for `Default::default()` then chaining setters.
//! - **`set_X(self, value) -> Self`** — field REPLACE on an
//!   existing builder. Consumes `self`, writes `X`, returns the
//!   updated value (e.g. [`ops::Step::set_ops`],
//!   [`ops::Step::set_hold`]). Previous contents of `X` are
//!   discarded.
//! - **`push_X(self, value) -> Self`** / **`extend_X<I>(self, iter)
//!   -> Self`** — field APPEND. `push_X` adds one element,
//!   `extend_X` adds many from any `IntoIterator` (e.g.
//!   [`backdrop::Backdrop::push_cgroup`] /
//!   [`backdrop::Backdrop::extend_cgroups`]).
//!
//! Naming an APPEND method `set_X` (or a REPLACE method `push_X`)
//! mis-encodes the semantics and is a defect — flag at review.
//!
//! See the [Scenarios](https://likewhatevs.github.io/ktstr/guide/concepts/scenarios.html)
//! and [Writing Tests](https://likewhatevs.github.io/ktstr/guide/writing-tests.html)
//! chapters of the guide.

pub mod affinity;
pub mod backdrop;
pub mod basic;
pub mod bpf_pin;
pub mod cpuset;
pub mod dynamic;
pub mod host_stall;
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
use std::sync::Arc;
use std::sync::atomic::AtomicU16;
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
///
/// # Method groups
///
/// ## Time helpers
///
/// - [`Self::settled_hold`] — `HoldSpec::fixed(settle + duration * f)`
///   sugar for the dominant Step hold-time pattern.
///
/// ## Cgroup construction
///
/// - [`Self::cgroup_def`] — `CgroupDef::named(name).workers(workers_per_cgroup)`
///   sugar that pins the default-worker-count shape across 40+ call
///   sites.
///
/// ## Topology accessors
///
/// - [`Self::cpuset_cpus`] — resolve a
///   [`CpusetSpec`](crate::scenario::ops::CpusetSpec) against this
///   context's topology and return the CPU count.
///
/// ## Constructors
///
/// - [`Self::builder`] — start a [`CtxBuilder`] with sane defaults for
///   unit-test scenarios.
/// - [`Self::payload`] — start a
///   [`PayloadRun`](crate::scenario::payload_run::PayloadRun) for a
///   given [`Payload`](crate::test_support::Payload).
///
/// # Field groups
///
/// Each pub field's doc is prefixed with its sub-concern label so the
/// rustdoc table groups visibly. The six groups are:
///
/// - **VM environment** — `cgroups`, `topo`. The host-side
///   filesystem + topology handles the scenario interacts with.
/// - **Test timing** — `duration`, `settle`. The wall-clock
///   budgets that shape every Step's hold-time math.
/// - **Cgroup defaults** — `workers_per_cgroup`, `work_type_override`.
///   The merge-time defaults `CgroupDef::merged_works` applies when a
///   `WorkSpec` leaves them unset.
/// - **Scheduler state** — `sched_pid`. Liveness-probe target for
///   inter-step scheduler-death detection.
/// - **Assertion policy** — `assert`. The merged
///   default+scheduler+per-test verdict checks
///   `run_scenario` / `execute_steps` apply.
/// - **Runtime coordination** — `wait_for_map_write`. Framework-set
///   gate that custom scenarios typically do not flip.
#[non_exhaustive]
pub struct Ctx<'a> {
    /// **VM environment.** Cgroup filesystem operations. `&dyn CgroupOps`
    /// (not `&CgroupManager`) so scenario code can be driven by an
    /// in-memory test double without touching `/sys/fs/cgroup`.
    /// Production callers pass `&CgroupManager` and the auto-coercion
    /// is transparent at the call site — `ctx.cgroups.set_cpuset(...)`
    /// works unchanged.
    pub cgroups: &'a dyn crate::cgroup::CgroupOps,
    /// **VM environment.** VM CPU topology.
    pub topo: &'a TestTopology,
    /// **Test timing.** How long to run the workload.
    pub duration: Duration,
    /// **Cgroup defaults.** Default number of workers per cgroup.
    pub workers_per_cgroup: usize,
    /// **Scheduler state.** PID of the running scheduler (for liveness
    /// checks), or `None` when no scheduler is attached. Stored as
    /// `Option<pid_t>` so the "no scheduler" state is a distinct
    /// variant rather than a 0-sentinel — `run_scenario` and
    /// step-level liveness probes destructure via `if let Some(pid)`
    /// instead of `!= 0` guards.
    pub sched_pid: Option<libc::pid_t>,
    /// **Test timing.** Time to wait after cgroup creation for
    /// scheduler stabilization.
    pub settle: Duration,
    /// **Cgroup defaults.** Override work type for scenarios that use
    /// `SpinWait` by default.
    pub work_type_override: Option<WorkType>,
    /// **Assertion policy.** Merged assertion config (default_checks +
    /// scheduler + per-test). Used by `run_scenario` for data-driven
    /// scenarios and by `execute_steps` as the default when no explicit
    /// checks are passed to `execute_steps_with`.
    pub assert: crate::assert::Assert,
    /// **Runtime coordination.** When true, `execute_steps` polls SHM
    /// signal slot 0 after writing the scenario start marker, blocking
    /// until the host confirms its BPF map write is complete. Set
    /// automatically by the framework when a `KtstrTestEntry` declares
    /// `bpf_map_write`; custom scenarios typically do not flip this
    /// manually.
    pub wait_for_map_write: bool,
    /// **Phase coordination.** Per-VM atomic publishing the current
    /// scenario step index. Written by the scenario driver immediately
    /// before each `run_step` call and read by three stamping sites
    /// so each captured sample carries the step it belongs to:
    /// (1) the host-side freeze-coordinator periodic-capture path
    /// stamps at periodic-fire time;
    /// (2) the on-demand `Op::CaptureSnapshot` apply arm stamps at
    /// apply time (the apply happens in the same phase as the
    /// capture);
    /// (3) the host-side user-watchpoint trip handler stamps at
    /// TRIP time, not at registration — the user issues
    /// `Op::WatchSnapshot` from some Step k, but the actual write
    /// that fires the watchpoint and triggers the snapshot can
    /// happen at any later phase, so the trip-time stamp pins the
    /// sample to the bucket matching when the observation actually
    /// occurred.
    ///
    /// Encoded per the framework's 1-indexed phase convention: `0` is
    /// the BASELINE settle window (the initial value), `1..=N` align
    /// with scenario Step ordinals (`step_idx + 1`). This matches
    /// [`crate::assert::PhaseBucket::step_index`] so a phase-aware
    /// sample drops directly into the correct bucket without a
    /// reindex.
    ///
    /// Stored as `AtomicU16` because the wire `StimulusPayload`
    /// step-index field is also `u16`, so a single shared width
    /// keeps the host-side bridge map and the guest-published wire
    /// value type-compatible without narrowing.
    ///
    /// Wrapped in `Arc` so the same per-VM publisher can be cloned
    /// into every consumer thread (scenario driver, freeze-coord,
    /// on-demand-capture apply arms) without a process-global
    /// static — multiple in-process VMs (e.g. parallel gauntlet
    /// variants) each get an independent atomic instead of racing
    /// on shared global state.
    pub current_step: Arc<AtomicU16>,
    /// **Drift-safe path derivation.** The `&'static str` name of
    /// the [`KtstrTestEntry`](crate::test_support::KtstrTestEntry)
    /// the running test body was dispatched as, stamped by the
    /// guest-side `maybe_dispatch_vm_test_with_args` and the
    /// host-only dispatch path before the test body runs. Drives
    /// the body-side path-derivation methods
    /// [`failure_dump_path`](Self::failure_dump_path),
    /// [`wprof_pb_path`](Self::wprof_pb_path), and
    /// [`repro_wprof_pb_path`](Self::repro_wprof_pb_path) — the
    /// drift-safe replacement for the legacy pattern of
    /// hardcoding the test fn name as a string literal at the
    /// callsite. When `Some(name)`, those methods derive the
    /// sidecar paths from the macro-stamped value at call time,
    /// so a future test rename surfaces the resulting
    /// `Result<PathBuf>` bail at compile-time-equivalent-failure
    /// (a deterministic Err) rather than as a runtime ENOENT
    /// against a stale literal.
    ///
    /// `None` is the manually-constructed-Ctx escape hatch — ad-hoc
    /// scenario tests that build `Ctx` via
    /// [`CtxBuilder::build`](CtxBuilder::build) without calling
    /// [`CtxBuilder::entry_name`](CtxBuilder::entry_name) get
    /// `None` and a path-derivation method invocation bails with an
    /// actionable diagnostic naming the missing-stamp scenario.
    /// Sibling to [`crate::vmm::VmResult::entry_name`] which carries
    /// the same `&'static str` on the post-VM result struct (the
    /// two ends of the test-name chain — pre-VM body context vs
    /// post-VM result — store the same shape so the body-side
    /// `ctx.failure_dump_path()` and the host-side
    /// `result.failure_dump_path()` resolve to identical paths).
    pub entry_name: Option<&'static str>,
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
            .field(
                "current_step",
                &self.current_step.load(std::sync::atomic::Ordering::Relaxed),
            )
            .field("entry_name", &self.entry_name)
            .finish()
    }
}

impl Ctx<'_> {
    /// Scheduler pid, filtered to the `> 0` range that
    /// `process_alive` treats as signalable.
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
    /// Read the live scheduler identity published by the
    /// `Op::AttachScheduler` / `Op::ReplaceScheduler` /
    /// `Op::DetachScheduler` dispatch arms. Returns `None` when no
    /// scheduler is currently attached (the pre-attach state at
    /// process start and the post-`Op::DetachScheduler` state).
    ///
    /// Distinct from `entry.scheduler` (the boot-time descriptor
    /// from the `#[ktstr_test]` macro): `entry.scheduler` stays
    /// the same across `Op::ReplaceScheduler` swaps, while
    /// `ctx.current_scheduler()` reflects the LIVE identity after
    /// any runtime swap. Consumer sites that care about the
    /// currently-attached BPF binary (verifier_stats wiring,
    /// monitor thresholds, auto-repro probe gates) want this
    /// method; sites that care about the test's declared
    /// scheduler (test-runner skip/include filtering, sidecar
    /// `scheduler_name` metadata) want `entry.scheduler`.
    ///
    /// v0 limitation: the boot path does not publish the boot
    /// scheduler into the side channel, so the first observable
    /// `Some` arrives after the first `Op::AttachScheduler` /
    /// `Op::ReplaceScheduler` runs. Consumer sites that want a
    /// fallback should call `.unwrap_or(&entry.scheduler.binary)`
    /// to combine the live view with the boot descriptor.
    pub fn current_scheduler(&self) -> Option<&'static crate::test_support::SchedulerSpec> {
        crate::vmm::rust_init::current_scheduler()
    }

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

    /// Resolve a `CpusetSpec` against this context's topology and
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

    /// Construct a [`CgroupDef`](crate::scenario::ops::CgroupDef) with `self.workers_per_cgroup`
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
    /// Returns a fresh [`CgroupDef`](crate::scenario::ops::CgroupDef) so the test author can chain
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

    /// Per-test failure-dump sidecar path. Derives
    /// `{sidecar_dir()}/{entry_name}.failure-dump.json` from the
    /// macro-stamped [`Self::entry_name`] — the drift-safe
    /// replacement for the legacy pattern of hardcoding the test
    /// fn name as a string literal at the callsite.
    ///
    /// # Sibling to [`crate::vmm::VmResult::failure_dump_path`]
    ///
    /// The post-VM result struct carries its own copy of the
    /// macro-stamped entry name + computes the same path string.
    /// A test body invocation `ctx.failure_dump_path()` and a
    /// post-VM `result.failure_dump_path()` resolve to identical
    /// paths because both stamp from the same
    /// `entry.name: &'static str` source — proc-macro emission at
    /// the `#[ktstr_test]` site.
    ///
    /// # Errors
    ///
    /// Bails when `self.entry_name` is `None`. The `None` shape is
    /// the manually-constructed-Ctx escape hatch — ad-hoc scenario
    /// unit tests build [`Ctx`] via [`CtxBuilder::build`] without
    /// calling [`CtxBuilder::entry_name`]; such a context cannot
    /// compute a drift-safe path. The bail diagnostic names the
    /// missing-stamp scenario explicitly so test authors who hit it
    /// know exactly which builder method to call.
    pub fn failure_dump_path(&self) -> anyhow::Result<std::path::PathBuf> {
        let name = self.entry_name.ok_or_else(|| {
            anyhow::anyhow!(
                "Ctx::failure_dump_path requires entry_name set by the \
                 macro-stamped dispatch path \
                 (`maybe_dispatch_vm_test_with_args`); reached with \
                 entry_name = None, which means the Ctx was \
                 constructed via CtxBuilder::build without calling \
                 .entry_name(...). Call ctx_builder.entry_name(name) \
                 explicitly, OR if this is a scenario unit-test fixture \
                 that has no test-entry context, derive the path inline \
                 (sidecar_dir().join(format!(\"{{name}}.failure-dump.json\"))) \
                 — the method form is for tests dispatched via \
                 #[ktstr_test], not for builder-driven fixtures."
            )
        })?;
        Ok(crate::test_support::sidecar_dir().join(format!("{name}.failure-dump.json")))
    }

    /// Per-test wprof Perfetto-trace sidecar path. Mirror of
    /// [`Self::failure_dump_path`] for the wprof artifact —
    /// derives `{sidecar_dir()}/{entry_name}.wprof.pb` from the
    /// macro-stamped [`Self::entry_name`].
    ///
    /// Sibling to [`crate::vmm::VmResult::wprof_pb_path`] —
    /// the post-VM and pre-VM derivations produce identical paths.
    /// See [`Self::failure_dump_path`] for the broader contract +
    /// the manually-constructed-Ctx None-bail diagnostic.
    ///
    /// # Errors
    ///
    /// Bails when `self.entry_name` is `None` per the same shape
    /// as [`Self::failure_dump_path`].
    pub fn wprof_pb_path(&self) -> anyhow::Result<std::path::PathBuf> {
        let name = self.entry_name.ok_or_else(|| {
            anyhow::anyhow!(
                "Ctx::wprof_pb_path requires entry_name set by the \
                 macro-stamped dispatch path; reached with \
                 entry_name = None — see Ctx::failure_dump_path \
                 for the manually-constructed-Ctx workaround."
            )
        })?;
        Ok(crate::test_support::sidecar_dir().join(format!("{name}.wprof.pb")))
    }

    /// Per-test auto-repro wprof Perfetto-trace sidecar path.
    /// Mirror of [`Self::wprof_pb_path`] for the auto-repro
    /// artifact — derives `{sidecar_dir()}/{entry_name}.repro.wprof.pb`
    /// from the macro-stamped [`Self::entry_name`]. The
    /// auto-repro variant lands ONLY when the primary VM's
    /// scenario reports a failure AND the test's
    /// `#[ktstr_test(auto_repro = true)]` gates the second-VM
    /// rerun; an absent `.repro.wprof.pb` is the steady state for
    /// passing tests.
    ///
    /// Sibling to [`crate::vmm::VmResult::repro_wprof_pb_path`].
    ///
    /// # Errors
    ///
    /// Bails when `self.entry_name` is `None` per the same shape
    /// as [`Self::failure_dump_path`].
    pub fn repro_wprof_pb_path(&self) -> anyhow::Result<std::path::PathBuf> {
        let name = self.entry_name.ok_or_else(|| {
            anyhow::anyhow!(
                "Ctx::repro_wprof_pb_path requires entry_name set by \
                 the macro-stamped dispatch path; reached with \
                 entry_name = None — see Ctx::failure_dump_path for \
                 the manually-constructed-Ctx workaround."
            )
        })?;
        Ok(crate::test_support::sidecar_dir().join(format!("{name}.repro.wprof.pb")))
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
/// - `sched_pid`: `None` — `run_scenario` short-circuits the
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
    current_step: Arc<AtomicU16>,
    entry_name: Option<&'static str>,
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
    /// checks in `run_scenario`.
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

    /// Inject a caller-owned per-VM step-index publisher. The
    /// default `Ctx::builder` already constructs a fresh
    /// `Arc<AtomicU16>` initialised to `0`, so most callers do
    /// not need this setter; it exists so the host-side VM runner
    /// can hand the same Arc to both the scenario driver `Ctx` and
    /// the freeze-coordinator thread, giving both halves a single
    /// per-VM source of truth for the current phase.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn current_step(mut self, cs: Arc<AtomicU16>) -> Self {
        self.current_step = cs;
        self
    }

    /// **Drift-safe path derivation.** Stamp the
    /// `&'static str` name of the
    /// [`KtstrTestEntry`](crate::test_support::KtstrTestEntry)
    /// the dispatched test body was registered as. Drives the
    /// body-side path-derivation methods on [`Ctx`]
    /// ([`failure_dump_path`](Ctx::failure_dump_path),
    /// [`wprof_pb_path`](Ctx::wprof_pb_path),
    /// [`repro_wprof_pb_path`](Ctx::repro_wprof_pb_path)) so test
    /// authors get the drift-safe per-test sidecar path without
    /// re-hardcoding the test fn name in the body — a future test
    /// rename surfaces a deterministic `Result<PathBuf>` bail
    /// rather than a runtime ENOENT against a stale literal.
    ///
    /// The framework's macro-stamped dispatch path
    /// (`maybe_dispatch_vm_test_with_args` + the host-only
    /// dispatcher) calls this with the entry name at Ctx
    /// construction time, before the test body runs. Ad-hoc
    /// scenario unit tests that build [`Ctx`] without the dispatch
    /// path skip this setter, and the path-derivation methods bail
    /// with an actionable diagnostic — see
    /// [`Ctx::failure_dump_path`] for the None-case bail shape.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn entry_name(mut self, name: &'static str) -> Self {
        self.entry_name = Some(name);
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
            current_step: self.current_step,
            entry_name: self.entry_name,
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
            current_step: Arc::new(AtomicU16::new(0)),
            entry_name: None,
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
/// Shared scaffolding for `run_scenario` and `setup_cgroups` —
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
/// A cgroup with no workers emits no [`crate::workload::WorkerReport`]s, so every downstream
/// assertion vacuously passes. Callers that want "no load" on a cgroup
/// should either drop the [`crate::workload::WorkSpec`] entry entirely (letting the default apply)
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
/// - [`AffinityIntent::CrossCgroup`]: topology exposes zero CPUs.
///   The public [`crate::topology::TestTopology`] constructors all
///   reject this at construction; reaching this case requires a
///   private-field construction or a future API addition.
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
            let all = topo.all_cpuset();
            if all.is_empty() {
                // Defense-in-depth against zero-CPU topologies. The two
                // public TestTopology constructors (`synthetic` +
                // `from_vm_topology`) both reject `num_cpus == 0` at
                // construction, so reaching this branch requires a
                // private-field construction or a future API addition
                // that produces a zero-CPU topology. Without this bail
                // an empty `Fixed` would either trip the
                // `flatten_for_spawn` unreachable!() OR (if reached via
                // a path that bypassed flatten) silently produce an
                // empty `sched_setaffinity` mask the kernel rejects
                // with EINVAL after the cgroup intersection.
                anyhow::bail!(
                    "AffinityIntent::CrossCgroup cannot satisfy any worker — \
                     the topology exposes zero CPUs. The public \
                     TestTopology constructors (`synthetic` + \
                     `from_vm_topology`) reject this at construction; \
                     reaching this bail means a direct private-field \
                     construction or a future API addition produced a \
                     zero-CPU topology. Build the test against a \
                     topology with at least one CPU, or switch to \
                     `AffinityIntent::Inherit` to defer to the cgroup \
                     cpuset.",
                );
            }
            Ok(ResolvedAffinity::Fixed(all))
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
/// (`LlcInfo::cores` empty — see `crate::topology::TestTopology::synthetic`). The
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
    // Render the search scope: when a cpuset narrowed the pool, name
    // it (operator can widen / pick siblings inside it); when no
    // cpuset is active, the scope IS the full topology (operator must
    // adjust topology or switch intents — naming "<no cpuset>" would
    // mislead by implying cpuset config is relevant).
    let scope = if cpuset.is_some() {
        format!("the effective cpuset ({})", format_cpuset_for_diag(cpuset))
    } else {
        "the full topology (no cgroup cpuset is active)".to_string()
    };
    anyhow::bail!(
        "AffinityIntent::SmtSiblingPair requires a physical core with at \
         least two SMT siblings present in {scope}. The current topology \
         and cpuset expose no such pair — threads_per_core may be 1 (SMT \
         disabled or non-SMT host), the cpuset may have isolated each \
         sibling onto a different cgroup, or the topology was built \
         without per-core sibling data. Switch to a different \
         AffinityIntent for non-SMT scheduling tests, or run on a host \
         whose VM topology has threads_per_core >= 2.",
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
/// disjoint, SmtSiblingPair no-pair-in-cpuset, CrossCgroup on
/// zero-CPU topology). The empty-pool "silent degrade to Inherit"
/// policy that previously lived here was removed — empty pools are
/// operator bugs, not "soft" fallbacks.
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
                // disjoint-intersection bail, CrossCgroup zero-CPU
                // topology bail). Reaching here means a future
                // constructor of ResolvedAffinity::Fixed bypassed
                // those checks — panic loudly so the regression
                // surfaces at the construction site, not as a silent
                // inheritance downstream.
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
mod tests;
