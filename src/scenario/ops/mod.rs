//! Composable ops/steps system for dynamic cgroup topology changes.
//!
//! [`Op`] is an atomic cgroup operation. [`Step`] sequences ops with a
//! hold period. [`CgroupDef`] bundles create + cpuset + spawn into a
//! single declaration. [`execute_steps()`] runs a step sequence with
//! scheduler liveness checks and stimulus event recording.
//!
//! See the [Ops and Steps](https://likewhatevs.github.io/ktstr/guide/concepts/ops.html)
//! chapter for a guide.
//!
//! # Cgroup tooling at a glance
//!
//! ktstr exposes the cgroup v2 surface across two layers — declarative
//! steady-state via [`CgroupDef`] (set at scenario-setup time, holds
//! for the cgroup's lifetime) and imperative state-transitions via
//! [`Op`] (applied mid-step, describe transitions over time):
//!
//! | Knob | Layer | API entry | Underlying file | When to use |
//! |------|-------|-----------|-----------------|-------------|
//! | CPU affinity | setup | [`CgroupDef::cpuset`] | `cpuset.cpus` | Bind workers to a CPU subset for the whole run. |
//! | NUMA-mem affinity | setup | [`CgroupDef::cpuset_mems`] | `cpuset.mems` | Constrain allocations to specific NUMA nodes. |
//! | CPU bandwidth | setup | [`CgroupDef::cpu_quota_pct`] / [`CgroupDef::cpu_quota`] / [`CgroupDef::cpu_unlimited`] | `cpu.max` | Cap CPU time per period (1 CPU at 50% / 2 CPU at 100% / etc). |
//! | CPU share weight | setup | [`CgroupDef::cpu_weight`] | `cpu.weight` | Bias relative CPU share when siblings contend. |
//! | Memory ceiling | setup | [`CgroupDef::memory_max`] / [`CgroupDef::memory_unlimited`] | `memory.max` | Hard ceiling — exceeding triggers cgroup OOM. |
//! | Memory throttle | setup | [`CgroupDef::memory_high`] | `memory.high` | Soft throttle: triggers reclaim, not OOM. |
//! | Memory protection | setup | [`CgroupDef::memory_low`] | `memory.low` | Soft protection: kernel reclaims from siblings first. |
//! | Swap cap | setup | [`CgroupDef::memory_swap_max`] / [`CgroupDef::memory_swap_unlimited`] | `memory.swap.max` | Cap how much memory can spill to swap (CONFIG_SWAP=y). |
//! | IO share | setup | [`CgroupDef::io_weight`] | `io.weight` | Bias relative IO share when siblings contend. |
//! | Task ceiling | setup | [`CgroupDef::pids_max`] / [`CgroupDef::pids_unlimited`] | `pids.max` | Cap process+thread count — fork/clone returns EAGAIN at limit. |
//! | Mid-run cpuset rebind | mid-step | [`Op::set_cpuset`] / [`Op::clear_cpuset`] / [`Op::swap_cpusets`] | `cpuset.cpus` | Move cpuset on a live cgroup mid-scenario. |
//! | Mid-run task migration | mid-step | [`Op::move_all_tasks`] | `cgroup.procs` | Move workers from one cgroup to another. |
//! | Pause/resume | mid-step | [`Op::freeze_cgroup`] / [`Op::unfreeze_cgroup`] | `cgroup.freeze` | Suspend every task in the cgroup; resume later. |
//! | Add/remove cgroup | mid-step | [`Op::add_cgroup`] / [`Op::remove_cgroup`] / [`Op::stop_cgroup`] | (cgroupfs mkdir/rmdir) | Spawn / tear down a cgroup mid-scenario. |
//!
//! # Worked examples
//!
//! * **Static topology** (one cgroup, fixed cpuset, weight-biased
//!   compute): [`CgroupDef`] type-level docs.
//! * **Suspend/resume** (3-Step idiom — run, freeze, run again):
//!   [`Op::FreezeCgroup`] doc.
//! * **Memory-cap teardown** (rewind a base CgroupDef's swap cap):
//!   [`CgroupDef::memory_swap_unlimited`] doc.
//!
//! # Implementation entry points
//!
//! Every knob ends in [`crate::cgroup::CgroupOps`] (production:
//! [`crate::cgroup::CgroupManager`]; tests: a recording `MockCgroupOps`
//! double). `apply_setup` runs the [`CgroupDef`] passes; `apply_ops`
//! dispatches the [`Op`] variants. Both share `ctx.cgroups` so a test
//! that uses both layers writes through the same RAII teardown
//! (`crate::scenario::CgroupGroup::Drop`).
//!
//! # File layout
//!
//! `types` holds the data model: [`Op`], [`CgroupDef`], [`Step`],
//! [`HoldSpec`], [`Setup`], [`CpusetSpec`], the per-controller limits
//! structs, and every builder constructor. Re-exported from this module
//! so external paths remain `crate::scenario::ops::Op` etc. The executor
//! in this file drives that model against [`crate::cgroup::CgroupOps`]
//! via `apply_setup` / `apply_ops` and exposes the [`execute_steps`] /
//! [`execute_scenario`] family of public entry points.

mod types;
pub use types::*;

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result};

use crate::assert::AssertResult;
use crate::scenario::backdrop;
use crate::scenario::{CgroupGroup, Ctx, process_alive};
use crate::vmm::guest_comms;
use crate::vmm::wire::StimulusPayload;
use crate::workload::{MemPolicy, ResolvedAffinity, WorkloadConfig, WorkloadHandle};

/// Latched once `Op::CaptureSnapshot` / `Op::WatchSnapshot` observes a
/// [`crate::vmm::wire::SnapshotRequestResult::TransportError`].
/// Process-scoped because the underlying transport (virtio-console
/// bulk port + SHM ring) is process-shared: once the host's freeze
/// coordinator stops draining, every subsequent guest-side request
/// will time out the same 30-second window. A `Loop` step that
/// fires `Op::CaptureSnapshot` every iteration would otherwise burn 30 s
/// per iteration on a permanently dead transport. After the first
/// timeout the flag short-circuits later attempts back to a
/// `tracing::warn!` no-op so the loop continues exercising the
/// scheduler workload at near-full cadence.
///
/// The flag is never cleared inside `apply_ops` — recovering the
/// transport requires fresh process state. New scenarios in the
/// same guest process inherit "transport dead" because the
/// underlying virtio-console port and host coordinator are the
/// same instance.
static SNAPSHOT_TRANSPORT_DEAD: AtomicBool = AtomicBool::new(false);

// ---------------------------------------------------------------------------
// Step executor
// ---------------------------------------------------------------------------

/// Persistent scenario-wide state owned by
/// [`execute_scenario_with`]. Lives for the entire step sequence;
/// cgroups, workload handles, and payload handles declared by the
/// [`Backdrop`](backdrop::Backdrop) go here and only tear
/// down at scenario end (success or Err). See [`StepState`] for
/// the step-local counterpart.
struct BackdropState<'a> {
    /// RAII cgroup guard for persistent cgroups — removes them on drop.
    cgroups: CgroupGroup<'a>,
    /// Active workload handles in persistent cgroups, keyed by cgroup name.
    handles: Vec<(String, WorkloadHandle)>,
    /// Resolved cpusets per persistent cgroup name.
    cpusets: std::collections::HashMap<String, BTreeSet<usize>>,
    /// Active payload-binary handles owned by the backdrop. Drained
    /// via `.kill()` at scenario teardown so the metric-emission
    /// pipeline still fires.
    payload_handles: Vec<PayloadEntry>,
}

impl<'a> BackdropState<'a> {
    /// Empty backdrop state (no persistent entities), scoped to `ctx.cgroups`.
    fn empty(ctx: &'a Ctx) -> Self {
        Self {
            cgroups: CgroupGroup::new(ctx.cgroups),
            handles: Vec::new(),
            cpusets: std::collections::HashMap::new(),
            payload_handles: Vec::new(),
        }
    }
}

/// Step-local execution state. Fresh per step, torn down at step
/// boundary: cgroups removed (via RAII drop), workload handles
/// collected, payload handles killed with metric emission. Any ops
/// in the step that reference a cgroup name look here first before
/// falling through to [`BackdropState`].
struct StepState<'a> {
    /// RAII cgroup guard — removes step-local cgroups on drop.
    cgroups: CgroupGroup<'a>,
    /// Active workload handles keyed by step-local cgroup name.
    handles: Vec<(String, WorkloadHandle)>,
    /// Resolved cpusets per step-local cgroup name, for isolation checks.
    cpusets: std::collections::HashMap<String, BTreeSet<usize>>,
    /// Active payload-binary handles keyed by cgroup name. Each entry
    /// came from either a [`CgroupDef::workload`] spawn in
    /// `apply_setup` or an explicit [`Op::RunPayload`] invocation;
    /// `source` tags which path spawned it so the duplicate-name
    /// dedup in `Op::RunPayload` can point at the original site. All
    /// are killed during step-teardown / cgroup removal so cgroupfs
    /// cleanup never trips EBUSY on a live process.
    payload_handles: Vec<PayloadEntry>,
}

impl<'a> StepState<'a> {
    /// Empty step state scoped to `ctx.cgroups`.
    fn empty(ctx: &'a Ctx) -> Self {
        Self {
            cgroups: CgroupGroup::new(ctx.cgroups),
            handles: Vec::new(),
            cpusets: std::collections::HashMap::new(),
            payload_handles: Vec::new(),
        }
    }
}

/// Combined mutable view over step-local and backdrop state.
///
/// Every function that touches execution state (apply_setup,
/// apply_ops, the drain helpers) receives a
/// `ScenarioState`; lookups prefer step-local, falling through to
/// backdrop. New state created via ops/setup inside a step writes
/// to step-local by default — that is the primary mechanism
/// enforcing per-step bounded lifetime. Setup for the Backdrop
/// itself (run once before the step loop) writes straight to the
/// backdrop side via [`ScenarioState::with_target_backdrop`].
struct ScenarioState<'a, 'b> {
    step: &'b mut StepState<'a>,
    backdrop: &'b mut BackdropState<'a>,
    /// When true, all mutations route to [`Self::backdrop`] instead
    /// of [`Self::step`]. Set by [`Self::with_target_backdrop`] when
    /// running the Backdrop's initial `apply_setup` / `apply_ops`
    /// before the first step.
    target_backdrop: bool,
}

impl<'a, 'b> ScenarioState<'a, 'b> {
    /// Build a combined scenario view. Starts with the step-local
    /// slot as the mutation target — call [`Self::with_target_backdrop`]
    /// to flip into backdrop-setup mode for Backdrop's own
    /// apply_setup / apply_ops pass.
    fn new(step: &'b mut StepState<'a>, backdrop: &'b mut BackdropState<'a>) -> Self {
        Self {
            step,
            backdrop,
            target_backdrop: false,
        }
    }

    /// Run `f` with writes routed to the backdrop side.
    fn with_target_backdrop<R>(&mut self, f: impl FnOnce(&mut Self) -> R) -> R {
        let prev = self.target_backdrop;
        self.target_backdrop = true;
        let r = f(self);
        self.target_backdrop = prev;
        r
    }

    /// `cgroups` group that receives newly-created cgroups. Step-local
    /// by default; backdrop when [`Self::with_target_backdrop`] is active.
    fn target_cgroups(&mut self) -> &mut CgroupGroup<'a> {
        if self.target_backdrop {
            &mut self.backdrop.cgroups
        } else {
            &mut self.step.cgroups
        }
    }

    /// `handles` vec that receives newly-spawned workload handles.
    fn target_handles(&mut self) -> &mut Vec<(String, WorkloadHandle)> {
        if self.target_backdrop {
            &mut self.backdrop.handles
        } else {
            &mut self.step.handles
        }
    }

    /// `cpusets` map that receives resolved cpusets for new cgroups.
    fn target_cpusets(&mut self) -> &mut std::collections::HashMap<String, BTreeSet<usize>> {
        if self.target_backdrop {
            &mut self.backdrop.cpusets
        } else {
            &mut self.step.cpusets
        }
    }

    /// `payload_handles` vec that receives newly-spawned payload handles.
    fn target_payload_handles(&mut self) -> &mut Vec<PayloadEntry> {
        if self.target_backdrop {
            &mut self.backdrop.payload_handles
        } else {
            &mut self.step.payload_handles
        }
    }

    /// Resolved cpuset for a cgroup name, looked up step-first then backdrop.
    fn lookup_cpuset(&self, name: &str) -> Option<&BTreeSet<usize>> {
        self.step
            .cpusets
            .get(name)
            .or_else(|| self.backdrop.cpusets.get(name))
    }

    /// Returns the live payload handle matching the composite key
    /// (`payload_name`, `cgroup_key`) from either step-local or
    /// backdrop state, or `None` when no entry matches. Used for
    /// the `Op::RunPayload` duplicate guard, which now treats
    /// "same payload in a different cgroup" as legitimate rather
    /// than a name collision.
    fn find_live_payload_with_cgroup(
        &self,
        payload_name: &str,
        cgroup_key: &str,
    ) -> Option<&PayloadEntry> {
        let matches =
            |e: &&PayloadEntry| e.handle.payload_name() == payload_name && e.cgroup == cgroup_key;
        self.step
            .payload_handles
            .iter()
            .find(matches)
            .or_else(|| self.backdrop.payload_handles.iter().find(matches))
    }

    /// Drop a payload handle by composite key (`name`, optional
    /// `cgroup`). Checks step-local first, then backdrop.
    ///
    /// - `cgroup = Some(c)`: exact match on both name and cgroup.
    /// - `cgroup = None`: if exactly one entry matches `name` across
    ///   both slots, consume it (backward-compat for
    ///   `Op::wait_payload(name)` / `Op::kill_payload(name)` when
    ///   only one copy is live). If two or more match, returns
    ///   `Err(ambiguous_cgroups)` where `ambiguous_cgroups` is the
    ///   list of cgroup keys for the candidates so the caller can
    ///   produce an actionable error.
    ///
    /// Returns `Ok(None)` when no entry matches.
    fn take_payload_by_name(
        &mut self,
        name: &str,
        cgroup: Option<&str>,
    ) -> std::result::Result<Option<PayloadEntry>, Vec<String>> {
        if let Some(c) = cgroup {
            // Composite-key path: exact match on both.
            if let Some(idx) = self
                .step
                .payload_handles
                .iter()
                .position(|e| e.handle.payload_name() == name && e.cgroup == c)
            {
                return Ok(Some(self.step.payload_handles.swap_remove(idx)));
            }
            if let Some(idx) = self
                .backdrop
                .payload_handles
                .iter()
                .position(|e| e.handle.payload_name() == name && e.cgroup == c)
            {
                return Ok(Some(self.backdrop.payload_handles.swap_remove(idx)));
            }
            return Ok(None);
        }
        // Name-only path: disambiguate across both slots before
        // consuming, so a mid-test wait on an ambiguous name
        // surfaces the caller's bug rather than silently waiting
        // on the first match.
        let mut step_idx: Option<usize> = None;
        let mut backdrop_idx: Option<usize> = None;
        let mut cgroups: Vec<String> = Vec::new();
        for (i, e) in self.step.payload_handles.iter().enumerate() {
            if e.handle.payload_name() == name {
                if step_idx.is_none() {
                    step_idx = Some(i);
                }
                cgroups.push(e.cgroup.clone());
            }
        }
        for (i, e) in self.backdrop.payload_handles.iter().enumerate() {
            if e.handle.payload_name() == name {
                if backdrop_idx.is_none() && step_idx.is_none() {
                    backdrop_idx = Some(i);
                }
                cgroups.push(e.cgroup.clone());
            }
        }
        if cgroups.len() > 1 {
            return Err(cgroups);
        }
        if let Some(i) = step_idx {
            return Ok(Some(self.step.payload_handles.swap_remove(i)));
        }
        if let Some(i) = backdrop_idx {
            return Ok(Some(self.backdrop.payload_handles.swap_remove(i)));
        }
        Ok(None)
    }

    /// Drain every live payload handle in step + backdrop state by
    /// calling `.kill()` so the metric-emission pipeline fires. Used
    /// on error paths in the step loop so mid-scenario failure still
    /// leaves a usable sidecar.
    fn drain_all_payloads(&mut self) {
        drain_all_payload_handles(&mut self.step.payload_handles);
        drain_all_payload_handles(&mut self.backdrop.payload_handles);
    }

    /// Kill every payload handle (step-first, then backdrop) whose
    /// cgroup matches `cgroup`. Called before a cgroup removal so
    /// cgroupfs cleanup does not trip EBUSY on a live process.
    fn drain_payloads_for_cgroup(&mut self, cgroup: &str) {
        drain_payload_handles_for_cgroup(&mut self.step.payload_handles, cgroup);
        drain_payload_handles_for_cgroup(&mut self.backdrop.payload_handles, cgroup);
    }

    /// Remove every workload handle whose key matches `cgroup`. The
    /// handles themselves drop (which SIGKILLs the workers) — this is
    /// appropriate for `Op::StopCgroup` and `Op::RemoveCgroup`.
    fn drop_handles_for_cgroup(&mut self, cgroup: &str) {
        self.step.handles.retain(|(n, _)| n.as_str() != cgroup);
        self.backdrop.handles.retain(|(n, _)| n.as_str() != cgroup);
    }

    /// Forget a tracked cpuset (step-first, then backdrop) for a cgroup.
    fn forget_cpuset(&mut self, cgroup: &str) {
        self.step.cpusets.remove(cgroup);
        self.backdrop.cpusets.remove(cgroup);
    }

    /// Record / overwrite the resolved cpuset for a cgroup. If the
    /// cgroup is known to step-local state, the step-local entry
    /// updates; if it's known to backdrop, the backdrop entry
    /// updates; otherwise the entry goes into the currently-active
    /// target (step-local, or backdrop inside `with_target_backdrop`).
    fn record_cpuset(&mut self, cgroup: &str, cpuset: BTreeSet<usize>) {
        if self.step.cpusets.contains_key(cgroup) {
            self.step.cpusets.insert(cgroup.to_string(), cpuset);
        } else if self.backdrop.cpusets.contains_key(cgroup) {
            self.backdrop.cpusets.insert(cgroup.to_string(), cpuset);
        } else {
            self.target_cpusets().insert(cgroup.to_string(), cpuset);
        }
    }

    /// Re-key every workload handle from `from` to `to`. When `to`
    /// names a Backdrop-owned cgroup, step-local handles are also
    /// transferred into [`Self::backdrop`] so their lifetime extends
    /// to scenario end instead of dying at step teardown. Backdrop
    /// handles stay in the backdrop slot regardless of `to`.
    ///
    /// Called by `Op::MoveAllTasks` after the kernel-side
    /// `cgroup.procs` writes succeed so subsequent ops that address
    /// the moved workers by cgroup name find them under the new key
    /// and in the correct state slot.
    fn rename_handles(&mut self, from: &str, to: &str) {
        let to_is_backdrop = self.cgroup_name_is_backdrop(to);
        if to_is_backdrop {
            // Move step-local handles keyed under `from` into the
            // backdrop slot, re-keyed to `to`. Iterate in reverse so
            // swap_remove indices stay stable.
            let mut i = self.step.handles.len();
            while i > 0 {
                i -= 1;
                if self.step.handles[i].0.as_str() == from {
                    let (_, handle) = self.step.handles.swap_remove(i);
                    self.backdrop.handles.push((to.to_string(), handle));
                }
            }
        } else {
            // Step-local destination: keep ownership, just rename.
            for (name, _) in &mut self.step.handles {
                if name.as_str() == from {
                    *name = to.to_string();
                }
            }
        }
        // Backdrop handles are never demoted to step-local ownership
        // regardless of destination — a backdrop worker is declared
        // persistent and stays persistent for the scenario. Rename
        // in place so subsequent ops still find it under the new key.
        for (name, _) in &mut self.backdrop.handles {
            if name.as_str() == from {
                *name = to.to_string();
            }
        }
    }

    /// Iterate every live workload handle across step + backdrop.
    /// Used by `Op::MoveAllTasks` / `Op::SetAffinity` which act on
    /// whichever cgroup owns the handle without caring about which
    /// state slot it's in.
    fn all_handles(&self) -> impl Iterator<Item = &(String, WorkloadHandle)> {
        self.step.handles.iter().chain(self.backdrop.handles.iter())
    }

    /// True iff a cgroup with the given name is already tracked by
    /// either step-local or backdrop state. Used to reject duplicate
    /// names at `apply_setup` time so a user can't accidentally
    /// shadow a Backdrop cgroup with a step-local [`CgroupDef`].
    fn cgroup_name_is_tracked(&self, name: &str) -> bool {
        self.step.cgroups.names().iter().any(|n| n == name)
            || self.backdrop.cgroups.names().iter().any(|n| n == name)
    }

    /// True iff a cgroup with the given name is tracked by backdrop
    /// (persistent) state. Used by `Op::MoveAllTasks` to decide
    /// handle-ownership transfer direction (step→backdrop transfers
    /// the handle into the persistent slot; backdrop→step-local is
    /// rejected because it would orphan workers at step teardown).
    fn cgroup_name_is_backdrop(&self, name: &str) -> bool {
        self.backdrop.cgroups.names().iter().any(|n| n == name)
    }
}

/// Whether a live payload handle was spawned by an explicit
/// [`Op::RunPayload`] inside the step or by a
/// [`CgroupDef::workload`] attachment at `apply_setup`. Held by
/// every [`PayloadEntry`] so the dedup path in `Op::RunPayload`
/// can name the original source when rejecting a second spawn of
/// the same name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PayloadSource {
    /// Spawned by `CgroupDef::workload(&payload)` during `apply_setup`.
    CgroupDefWorkload,
    /// Spawned by `Op::RunPayload { payload, .. }` inside the step's ops.
    OpRunPayload,
}

impl PayloadSource {
    /// Human-readable tag for error output. Describes the API surface
    /// that originated the spawn, not the internal dispatch site.
    fn describe(self) -> &'static str {
        match self {
            PayloadSource::CgroupDefWorkload => "CgroupDef::workload",
            PayloadSource::OpRunPayload => "Op::RunPayload",
        }
    }
}

/// One live payload handle plus the cgroup it runs inside and the
/// API surface that spawned it. `cgroup` is empty iff
/// `source == PayloadSource::OpRunPayload` was invoked without a
/// `cgroup = Some(...)` argument — in which case the payload runs
/// in whatever cgroup its parent process inherited (no explicit
/// placement).
struct PayloadEntry {
    cgroup: String,
    source: PayloadSource,
    handle: crate::scenario::payload_run::PayloadHandle,
}

/// Map the BPF probe's current scheduler-exit classification onto
/// the [`crate::assert::DetailKind`] variant the three liveness
/// emission sites push. Reads [`crate::probe::process::sched_exit_kind`]
/// which mirrors the probe's `ktstr_err_exit_detected` BSS latch
/// across threads.
///
/// Returns:
/// - `SchedulerCrashed` when the probe observed a non-clean kernel
///   exit (any path that latched `ktstr_err_exit_detected`).
/// - `SchedulerExitedCleanly` when the probe ran but never observed
///   the latch (clean `SCX_EXIT_NONE` teardown, or the scheduler
///   exited for a benign reason).
/// - `SchedulerDiedUnknownReason` when the probe has not classified
///   yet — typically the probe pipeline never wired for this run
///   (host-only test, no scheduler attached) or the poll thread has
///   not completed a first iteration since the prior reset.
fn sched_died_detail_kind() -> crate::assert::DetailKind {
    use crate::assert::DetailKind;
    use crate::probe::process::{SchedExitKind, sched_exit_kind};
    match sched_exit_kind() {
        SchedExitKind::Crashed => DetailKind::SchedulerCrashed,
        SchedExitKind::Clean => DetailKind::SchedulerExitedCleanly,
        SchedExitKind::Unknown => DetailKind::SchedulerDiedUnknownReason,
    }
}

/// Execute a single step with CgroupDefs that hold for the full duration.
///
/// Convenience wrapper around [`execute_steps`] for the common pattern
/// of creating cgroups and running them for [`HoldSpec::FULL`].
pub fn execute_defs(ctx: &Ctx, defs: Vec<CgroupDef>) -> Result<AssertResult> {
    execute_steps(ctx, vec![Step::with_defs(defs, HoldSpec::FULL)])
}

/// Execute a sequence of steps against the given context.
///
/// Convenience wrapper around [`execute_steps_with`] that passes
/// `None` for checks, falling back to `ctx.assert`. Use
/// [`execute_steps_with`] when you need to override `ctx.assert`.
pub fn execute_steps(ctx: &Ctx, steps: Vec<Step>) -> Result<AssertResult> {
    execute_steps_with(ctx, steps, None)
}

/// Execute a [`Backdrop`](backdrop::Backdrop) + Steps sequence
/// against the given context.
///
/// The Backdrop declares persistent scenario-wide state
/// (long-running payloads, cgroups referenced by many Steps) while
/// Steps express bounded per-phase behavior. The runtime sets up
/// the Backdrop before the first Step, runs the Step sequence
/// with per-Step teardown (cgroups removed, workload handles
/// collected, payload handles killed at step boundary), and tears
/// the Backdrop down at the end.
pub fn execute_scenario(
    ctx: &Ctx,
    backdrop: backdrop::Backdrop,
    steps: Vec<Step>,
) -> Result<AssertResult> {
    execute_scenario_with(ctx, backdrop, steps, None)
}

/// [`execute_scenario`] with an explicit
/// [`Assert`](crate::assert::Assert) override — the Backdrop
/// equivalent of [`execute_steps_with`].
pub fn execute_scenario_with(
    ctx: &Ctx,
    backdrop: backdrop::Backdrop,
    steps: Vec<Step>,
    checks: Option<&crate::assert::Assert>,
) -> Result<AssertResult> {
    run_scenario(ctx, backdrop, steps, checks)
}

/// Execute steps with an explicit [`Assert`](crate::assert::Assert) for
/// worker checks. When `checks` is `Some`, it overrides `ctx.assert`.
/// When `None`, uses `ctx.assert` (the merged three-layer config).
///
/// Thin wrapper around [`execute_scenario_with`] with an empty
/// [`Backdrop`](backdrop::Backdrop) — every Step's effects
/// (cgroups, workloads, payloads) tear down at the step boundary.
pub fn execute_steps_with(
    ctx: &Ctx,
    steps: Vec<Step>,
    checks: Option<&crate::assert::Assert>,
) -> Result<AssertResult> {
    execute_scenario_with(ctx, backdrop::Backdrop::new(), steps, checks)
}

/// Compute the union of cgroup v2 controllers required by a
/// Backdrop and Step sequence. Walks every [`CgroupDef`] declaration
/// and every [`Op`] variant, returning the smallest set of
/// controllers that must be enabled in `cgroup.subtree_control` for
/// the scenario's per-knob writes to land.
///
/// Mapping:
/// - [`CgroupDef::cpuset`] / [`CgroupDef::cpuset_mems`] → [`Controller::Cpuset`]
/// - [`CgroupDef::cpu`] → [`Controller::Cpu`]
/// - [`CgroupDef::memory`] → [`Controller::Memory`]
/// - [`CgroupDef::pids`] → [`Controller::Pids`]
/// - [`CgroupDef::io`] → [`Controller::Io`]
/// - [`Op::SetCpuset`] / [`Op::ClearCpuset`] / [`Op::SwapCpusets`] /
///   [`Op::SetAffinity`] → [`Controller::Cpuset`]
/// - Every other [`Op`] variant ([`Op::FreezeCgroup`],
///   [`Op::AddCgroup`], [`Op::SpawnWorkers`], [`Op::MoveAllTasks`], etc.)
///   touches cgroup-core knobs (`cgroup.freeze`, `cgroup.procs`,
///   `mkdir`/`rmdir`) which are ungated by any controller and
///   contribute nothing to this set.
///
/// Returning the SMALLEST set lets a test that intentionally
/// requires the absence of a controller (e.g. testing behavior on
/// a kernel without `+cpu`) get an empty subtree_control write.
fn required_controllers(
    ctx: &Ctx,
    backdrop: &backdrop::Backdrop,
    steps: &[Step],
) -> BTreeSet<crate::cgroup::Controller> {
    use crate::cgroup::Controller;
    fn absorb_def(set: &mut BTreeSet<Controller>, def: &CgroupDef) {
        if def.cpuset.is_some() || def.cpuset_mems.is_some() {
            set.insert(Controller::Cpuset);
        }
        if def.cpu.is_some() {
            set.insert(Controller::Cpu);
        }
        if def.memory.is_some() {
            set.insert(Controller::Memory);
        }
        if def.io.is_some() {
            set.insert(Controller::Io);
        }
        if def.pids.is_some() {
            set.insert(Controller::Pids);
        }
    }
    fn absorb_op(set: &mut BTreeSet<Controller>, op: &Op) {
        if matches!(
            op,
            Op::SetCpuset { .. }
                | Op::ClearCpuset { .. }
                | Op::SwapCpusets { .. }
                | Op::SetAffinity { .. }
        ) {
            set.insert(Controller::Cpuset);
        }
        // AddCgroupDef carries a full CgroupDef whose knobs may
        // require any of the same controllers absorb_def covers. The
        // op-applied def goes through apply_setup at op-execute time,
        // which writes to those controller files; the parent's
        // subtree_control must already have the controllers enabled
        // by then, so absorb the def's needs into the pre-scenario
        // controller setup the same way step-local CgroupDefs do.
        if let Op::AddCgroupDef { def } = op {
            absorb_def(set, def);
        }
    }
    let mut set = BTreeSet::new();
    for def in &backdrop.cgroups {
        absorb_def(&mut set, def);
    }
    for op in &backdrop.ops {
        absorb_op(&mut set, op);
    }
    for step in steps {
        for def in step.setup.resolve(ctx) {
            absorb_def(&mut set, &def);
        }
        for op in &step.ops {
            absorb_op(&mut set, op);
        }
    }
    set
}

/// Internal driver: runs Backdrop setup, the Step loop with
/// per-Step teardown, and final Backdrop teardown.
fn run_scenario(
    ctx: &Ctx,
    backdrop: backdrop::Backdrop,
    steps: Vec<Step>,
    checks: Option<&crate::assert::Assert>,
) -> Result<AssertResult> {
    // Validate every step's hold spec up front so a typo doesn't
    // reach `Duration::from_secs_f64(NaN)` / `thread::sleep(ZERO)` /
    // a no-yield Loop busy-wait after ops have already been applied.
    for (i, step) in steps.iter().enumerate() {
        if let Err(reason) = step.hold.validate() {
            anyhow::bail!("step {i} hold validation: {reason}");
        }
    }
    // Validate Backdrop payloads before creating any runtime state.
    // Only binary payloads can be spawned by Op::RunPayload, which
    // is what the Backdrop setup uses under the hood. Reject
    // scheduler-kind payloads here so the failure surface is the
    // Backdrop declaration, not a mid-scenario spawn error after
    // cgroups have already been created.
    for p in &backdrop.payloads {
        if p.is_scheduler() {
            anyhow::bail!(
                "Backdrop::push_payload received scheduler-kind Payload '{}' — \
                 only PayloadKind::Binary payloads run in the Backdrop; \
                 place scheduler-kind payloads on the #[ktstr_test(scheduler = ...)] \
                 attribute instead",
                p.name,
            );
        }
    }
    // Scheduler-kind payloads smuggled via Backdrop::push_op(Op::RunPayload { ... })
    // would otherwise bypass the check above and only bail deep inside
    // apply_ops. Reject them here with a Backdrop-specific error so
    // the failure surface matches the declaration surface.
    for op in &backdrop.ops {
        if let Op::RunPayload { payload, .. } = op
            && payload.is_scheduler()
        {
            anyhow::bail!(
                "Backdrop::push_op(Op::RunPayload) received scheduler-kind Payload '{}' — \
                 only PayloadKind::Binary payloads run in the Backdrop; \
                 place scheduler-kind payloads on the #[ktstr_test(scheduler = ...)] \
                 attribute instead",
                payload.name,
            );
        }
    }
    let effective_checks = checks.unwrap_or(&ctx.assert);

    // Enable the controllers this scenario actually needs in
    // `cgroup.subtree_control` BEFORE any cgroupfs writes land. The
    // union is computed from every CgroupDef and Op declared in the
    // backdrop+steps; tests that declare no controller-gated knobs
    // get an empty set (parent dir created, no subtree_control walk).
    let required = required_controllers(ctx, &backdrop, &steps);
    ctx.cgroups
        .setup(&required)
        .context("enable cgroup controllers in subtree_control")?;

    let mut backdrop_state = BackdropState::empty(ctx);
    let mut result = AssertResult::pass();

    let scenario_start = std::time::Instant::now();

    // ScenarioStart marker. `is_guest` short-circuits in host
    // contexts (unit tests) where the bulk port and SHM ring are
    // both absent and `send_scenario_start` would log a no-op warning.
    if guest_comms::is_guest() {
        crate::vmm::guest_comms::send_scenario_start();
    }

    // When a host-side BPF map write is configured the test framework
    // sets `wait_for_map_write=true`; in that case block until the
    // guest's `hvc0_poll_loop` observes
    // [`crate::vmm::virtio_console::SIGNAL_BPF_WRITE_DONE`] (pushed by
    // the host's `bpf-map-write` thread after every queued
    // `bpf_map_write` lands) and fires the `bpf_map_write_done` latch.
    // Without this gate the workload phase races against the host's
    // map writes and may observe a stale BPF map value.
    //
    // Guest-only path. On the host (unit tests) the latch is never
    // armed, so we skip the wait entirely. The 60 s timeout matches
    // the bpf-map-write thread's combined phase 1 + phase 2 budget
    // (30 s accessor init + 30 s map discovery in
    // `freeze_coord::start_bpf_map_write`); a real timeout means the
    // host failed to resolve a map. The scenario continues anyway
    // (rather than `bail!`) because the legacy rendezvous also let
    // the guest proceed under its own timeout, and a bail here would
    // mask the underlying host-side resolution failure with a
    // test-side `Err`.
    if ctx.wait_for_map_write && guest_comms::is_guest() {
        let latch = crate::vmm::rust_init::bpf_map_write_done_latch();
        if !latch.wait_timeout(Duration::from_secs(60)) {
            tracing::warn!(
                "wait_for_map_write timed out after 60s — host bpf-map-write \
                 thread may have failed to resolve a queued map; proceeding \
                 with the workload regardless"
            );
        }
    }

    // --- Backdrop setup (persistent) ---
    // Run before the first Step. Cgroups + payloads declared on
    // `backdrop` land in `backdrop_state` so they survive every
    // Step's teardown. On error, drain Backdrop payload handles
    // (metric emission) and propagate.
    if !backdrop.is_empty() {
        let mut step_staging = StepState::empty(ctx);
        let mut scratch = ScenarioState::new(&mut step_staging, &mut backdrop_state);
        let setup_res = scratch.with_target_backdrop(|s| {
            // Order: cgroups → ops → payloads. CgroupDefs go first so
            // a later `Op::add_cgroup` / `Op::run_payload_in_cgroup`
            // can target cgroups that `apply_setup` just created.
            // Payloads spawn last so `run_payload` resolving a cgroup
            // placement lands inside a cgroup that either apply pass
            // already built.
            if !backdrop.cgroups.is_empty() {
                apply_setup(ctx, s, &backdrop.cgroups)?;
            }
            // Raw ops: typically `Op::AddCgroup` for empty move-target
            // cgroups (can't be expressed via CgroupDef because
            // apply_setup forces a worker spawn), or placement-aware
            // `Op::RunPayload` targeting a just-created backdrop
            // cgroup.
            if !backdrop.ops.is_empty() {
                apply_ops(ctx, s, &backdrop.ops)?;
            }
            // Shorthand payloads: one Op::RunPayload per entry,
            // inherited cgroup placement.
            if !backdrop.payloads.is_empty() {
                let ops: Vec<Op> = backdrop
                    .payloads
                    .iter()
                    .map(|p| Op::run_payload(p, Vec::<String>::new()))
                    .collect();
                apply_ops(ctx, s, &ops)?;
            }
            Ok::<(), anyhow::Error>(())
        });
        if let Err(err) = setup_res {
            // Collect any workers that DID spawn before the failure
            // so their stats reach the final result instead of being
            // discarded by `WorkloadHandle::drop` (which SIGKILLs
            // without gathering scheduler-side data). `collect_*`
            // drain `payload_handles` internally, so the backdrop-
            // and step-side payloads still get `.kill()` (SHM metric
            // emission) on the error path.
            //
            // `with_target_backdrop` routes every target writer to
            // the backdrop slot, so `step_staging` normally holds
            // nothing — but collect defensively so a partial-failure
            // path that leaks a non-backdrop write surfaces here
            // rather than disappearing into `StepState::drop`.
            let mut r =
                collect_backdrop(&mut backdrop_state, effective_checks, ctx.topo, ctx.cgroups);
            let staging_result =
                collect_step(&mut step_staging, effective_checks, ctx.topo, ctx.cgroups);
            r.merge(staging_result);
            r.merge(result);
            // step_staging's CgroupGroup RAII still drops here,
            // removing any cgroups the failed Backdrop setup routed
            // into step-local state.
            r.record_fail(crate::assert::AssertDetail::new(
                crate::assert::DetailKind::Other,
                format!("Backdrop setup failed: {err:#}"),
            ));
            return Ok(r);
        }
        // `step_staging` should not have accumulated anything
        // because `with_target_backdrop` routed every target writer
        // to the backdrop side. Collect any stray handles defensively
        // before dropping so a future refactor that leaks a non-
        // backdrop write here surfaces as a missed teardown rather
        // than silently discarded state.
        drain_all_payload_handles(&mut step_staging.payload_handles);
    }

    // --- Step loop with per-Step teardown ---
    for (step_idx, step) in steps.iter().enumerate() {
        // Check scheduler liveness between steps (skip before first).
        // Live `crate::vmm::rust_init::sched_pid()` read instead of
        // `ctx.sched_pid` snapshot so a mid-scenario
        // `Op::ReplaceScheduler` swap is reflected — the swap
        // dispatcher updates `SCHED_PID` to the new child via
        // `set_sched_pid`, and this check then observes the new
        // pid's liveness (not the dead boot pid). `None` means
        // either no scheduler was configured at boot or
        // `Op::DetachScheduler` cleared the pid; the liveness probe
        // cannot meaningfully report on a pid that doesn't exist.
        if step_idx > 0
            && let Some(pid) = crate::vmm::rust_init::sched_pid()
            && !process_alive(pid)
        {
            // Collect backdrop-owned workload handles into the
            // result before reporting the crash so whatever the
            // persistent workers produced is still assertable.
            let mut r =
                collect_backdrop(&mut backdrop_state, effective_checks, ctx.topo, ctx.cgroups);
            r.merge(result);
            r.record_fail(crate::assert::AssertDetail::new(
                sched_died_detail_kind(),
                crate::assert::format_sched_died_after_step(
                    step_idx,
                    steps.len(),
                    scenario_start.elapsed().as_secs_f64(),
                ),
            ));
            return Ok(r);
        }

        let mut step_state = StepState::empty(ctx);
        let mut sched_died_during_hold = false;
        // Publish the 1-indexed phase number for this Step so the
        // freeze-coordinator periodic-capture path and the on-demand
        // Op::CaptureSnapshot / Op::WatchSnapshot apply arms all
        // stamp the captures they take with the correct scenario
        // phase. The 1-indexed encoding (scenario Step k -> phase
        // k + 1) reserves phase 0 for the pre-first-Step BASELINE
        // settle window. `Release` pairs with the consumers'
        // `Acquire` load so a sample stamped with this value
        // happens-after any state the Step has set up before
        // calling run_step.
        let phase_step_index = u16::try_from(step_idx)
            .ok()
            .and_then(|i| i.checked_add(1))
            .unwrap_or(u16::MAX);
        ctx.current_step
            .store(phase_step_index, std::sync::atomic::Ordering::Release);
        let step_res = run_step(
            ctx,
            step,
            step_idx,
            &mut step_state,
            &mut backdrop_state,
            scenario_start,
            effective_checks,
            &mut sched_died_during_hold,
        );

        if guest_comms::is_guest() {
            crate::vmm::guest_comms::send_scenario_pause();
        }

        let step_result = collect_step(&mut step_state, effective_checks, ctx.topo, ctx.cgroups);
        result.merge(step_result);

        // A step-level error is converted into a failure on the
        // accumulated result after teardown has run so every step
        // boundary leaves clean state behind even on failure. The
        // caller keeps the prior-steps' merged AssertResult plus
        // the error context as a detail, instead of an opaque Err
        // that discards everything.
        if let Err(err) = step_res {
            // Collect Backdrop-owned workload handles into a fresh
            // result first, then merge the accumulated step result
            // on top. `collect_backdrop` drains
            // `backdrop_state.payload_handles` internally, so the
            // backdrop-side payloads still get `.kill()` (metric
            // emission) on the error path. Ordering mirrors the
            // scheduler-crash path above so detail order is
            // consistent across both Ok(failed) returns.
            let mut r =
                collect_backdrop(&mut backdrop_state, effective_checks, ctx.topo, ctx.cgroups);
            r.merge(result);
            r.record_fail(crate::assert::AssertDetail::new(
                crate::assert::DetailKind::Other,
                format!("step {step_idx} failed: {err:#}"),
            ));
            return Ok(r);
        }

        // Scheduler exited during the step's hold-period sleep —
        // [`run_step`] cut the hold short and stamped
        // `sched_died_during_hold`. Emit the in-step
        // sched-died message before continuing to the next step
        // boundary; otherwise the post-loop probe would fire after
        // the full scenario duration and stamp a misleading elapsed
        // time. Same Backdrop-then-step merge order as the
        // inter-step path above so detail ordering stays consistent.
        if sched_died_during_hold {
            let mut r =
                collect_backdrop(&mut backdrop_state, effective_checks, ctx.topo, ctx.cgroups);
            r.merge(result);
            r.record_fail(crate::assert::AssertDetail::new(
                sched_died_detail_kind(),
                crate::assert::format_sched_died_during_workload(
                    scenario_start.elapsed().as_secs_f64(),
                ),
            ));
            return Ok(r);
        }
    }

    // ScenarioEnd marker. Routes through `send_scenario_end`
    // (virtio-console port-1 with COM2 fallback for early-boot).
    if guest_comms::is_guest() {
        let elapsed = scenario_start.elapsed().as_millis() as u64;
        crate::vmm::guest_comms::send_scenario_end(elapsed);
    }

    // Final liveness check. Live `crate::vmm::rust_init::sched_pid()`
    // read instead of `ctx.sched_pid` snapshot so a mid-scenario
    // Op::ReplaceScheduler swap reflects the new pid here too.
    // sched_pid() == None ⇒ no scheduler configured (kernel-default
    // path) OR Op::DetachScheduler cleared it; no liveness to
    // report on either case.
    let sched_dead = crate::vmm::rust_init::sched_pid().is_some_and(|pid| !process_alive(pid));

    // --- Backdrop teardown ---
    let backdrop_result =
        collect_backdrop(&mut backdrop_state, effective_checks, ctx.topo, ctx.cgroups);
    result.merge(backdrop_result);

    if sched_dead {
        result.record_fail(crate::assert::AssertDetail::new(
            sched_died_detail_kind(),
            crate::assert::format_sched_died_after_all_steps(
                steps.len(),
                scenario_start.elapsed().as_secs_f64(),
            ),
        ));
    }

    Ok(result)
}

/// Sleep up to `dur`, returning early if `sched_pid` exits.
///
/// Returns `true` the first time the scheduler is observed dead,
/// `false` if the full duration elapsed with no death observed.
/// When `sched_pid` is `None` (kernel-default scheduling, no
/// scheduler process to monitor), behaves exactly like
/// [`thread::sleep`] and always returns `false`.
///
/// Implementation uses `pidfd_open(2)` + `epoll_wait` so the waiter
/// is kernel-blocked on the pidfd until either the scheduler exits
/// (pidfd becomes readable) or the per-step hold elapses. This
/// drops crash-detection latency from one poll-tick (the previous
/// 100 ms cadence) to ~0: the kernel wakes the epoll waiter as
/// soon as the task transitions to EXIT_ZOMBIE. Mirrors
/// [`crate::scenario::payload_run`]'s `wait_with_deadline` shape.
/// Minimum kernel: Linux 5.3.
///
/// Deadline honoring: the `epoll_wait` timeout is re-derived from
/// `saturating_duration_since` each iteration so `EINTR` restarts
/// narrow the remaining window rather than extending it.
///
/// Failure handling: if `pidfd_open` returns `ESRCH`, the scheduler
/// is already gone — return `true` immediately without sleeping. If
/// it returns any other unexpected errno (e.g. on a kernel without
/// `pidfd_open` support), log a warning and fall back to a
/// `thread::sleep` for the full duration plus a final
/// [`process_alive`] check. The fallback loses sub-poll-tick
/// detection latency but preserves the boolean contract.
///
/// Scheduling jitter under load can leave the actual elapsed time
/// modestly above `dur`.
fn sleep_or_sched_died(dur: Duration, sched_pid: Option<libc::pid_t>) -> bool {
    use nix::sys::epoll::{Epoll, EpollCreateFlags, EpollEvent, EpollFlags, EpollTimeout};
    use std::os::fd::{AsFd, FromRawFd, OwnedFd};

    if dur.is_zero() {
        return sched_pid.is_some_and(|pid| !process_alive(pid));
    }
    let Some(pid) = sched_pid else {
        thread::sleep(dur);
        return false;
    };

    // `pidfd_open(pid, 0)`: returns an fd that becomes readable when
    // the pid exits. Only meaningful on a thread-group leader, which
    // every `sched_pid` already is (it is the scheduler binary's
    // top-level pid as recorded in `Ctx::sched_pid`). No
    // `PIDFD_NONBLOCK` flag — epoll is the gate.
    let pidfd_raw = unsafe { libc::syscall(libc::SYS_pidfd_open, pid, 0i32) };
    if pidfd_raw < 0 {
        let err = std::io::Error::last_os_error();
        if err.raw_os_error() == Some(libc::ESRCH) {
            // pidfd_open observed the pid as gone before we could
            // even attach a waiter — sched is already dead.
            return true;
        }
        // Unexpected pidfd_open failure — fall back to a single
        // sleep + final liveness probe. Logged so unexpected errnos
        // don't disappear silently.
        tracing::warn!(
            target: "ktstr::scenario",
            pid,
            error = %err,
            "pidfd_open failed; falling back to sleep + final process_alive check"
        );
        thread::sleep(dur);
        return !process_alive(pid);
    }
    // SAFETY: the syscall succeeded and returned a fresh fd; it is
    // not registered with any other owner.
    let pidfd: OwnedFd = unsafe { OwnedFd::from_raw_fd(pidfd_raw as i32) };

    // epoll setup. EPOLL_CLOEXEC matches `wait_with_deadline` to
    // avoid leaking the epoll fd into any post-fork descendant.
    let epoll = match Epoll::new(EpollCreateFlags::EPOLL_CLOEXEC) {
        Ok(e) => e,
        Err(e) => {
            tracing::warn!(
                target: "ktstr::scenario",
                pid,
                error = %e,
                "epoll_create1 failed for pidfd waiter; falling back to sleep + final process_alive check"
            );
            drop(pidfd);
            thread::sleep(dur);
            return !process_alive(pid);
        }
    };
    // `data` field is unused — we only ever watch one fd. The add()
    // syscall still needs an `EpollEvent` with populated events.
    let event = EpollEvent::new(EpollFlags::EPOLLIN, 0);
    if let Err(e) = epoll.add(pidfd.as_fd(), event) {
        tracing::warn!(
            target: "ktstr::scenario",
            pid,
            error = %e,
            "epoll_ctl ADD pidfd failed; falling back to sleep + final process_alive check"
        );
        drop(pidfd);
        thread::sleep(dur);
        return !process_alive(pid);
    }

    let deadline = std::time::Instant::now() + dur;
    let mut events = [EpollEvent::empty()];
    loop {
        let remaining = deadline.saturating_duration_since(std::time::Instant::now());
        if remaining.is_zero() {
            // Hold elapsed without a wakeup. Re-probe once via
            // `process_alive` to catch a race where the pid exited
            // between the last `epoll_wait` return and the deadline
            // check (e.g. during EINTR re-entry).
            return !process_alive(pid);
        }

        // `PollTimeout` (aliased as `EpollTimeout`) stores the value
        // as `i32`, so `TryFrom<u32>` rejects any input larger than
        // `i32::MAX` (~24.8 days of milliseconds). Clamp `u128 →
        // u32` and then `u32 → i32`-range so a `Duration::MAX`
        // remainder saturates to the max accepted value instead of
        // bubbling up a conversion error.
        let ms_u32 = u32::try_from(remaining.as_millis()).unwrap_or(u32::MAX);
        let ms_u32 = std::cmp::min(ms_u32, i32::MAX as u32);
        let timeout_param = match EpollTimeout::try_from(ms_u32) {
            Ok(t) => t,
            Err(e) => {
                tracing::warn!(
                    target: "ktstr::scenario",
                    pid,
                    error = %e,
                    "epoll timeout conversion failed; falling back to sleep + final process_alive check"
                );
                thread::sleep(remaining);
                return !process_alive(pid);
            }
        };

        match epoll.wait(&mut events, timeout_param) {
            Ok(0) => {
                // Timeout fired with no ready events. Loop back so
                // `remaining.is_zero()` at the top handles the
                // deadline path uniformly.
            }
            Ok(_) => {
                // pidfd became readable — task transitioned to
                // EXIT_ZOMBIE. Scheduler is dead.
                return true;
            }
            Err(nix::errno::Errno::EINTR) => {
                // Signal interrupted the wait; loop and re-compute
                // the remaining window.
            }
            Err(e) => {
                tracing::warn!(
                    target: "ktstr::scenario",
                    pid,
                    error = %e,
                    "epoll_wait failed; falling back to sleep + final process_alive check"
                );
                thread::sleep(remaining);
                return !process_alive(pid);
            }
        }
    }
}

/// Run a single step's setup + ops + hold against step-local state.
///
/// On error, the caller is expected to invoke `collect_step` for
/// per-step teardown (which runs regardless) and then propagate.
///
/// `sched_died_during_hold` is set to `true` when the hold-period
/// liveness poll observes the scheduler process exiting; the caller
/// uses this to emit [`crate::assert::format_sched_died_during_workload`]
/// instead of waiting for the post-loop probe to fire (which would
/// stamp the message with the full scenario duration even though
/// the death happened mid-step).
#[allow(clippy::too_many_arguments)]
fn run_step<'a>(
    ctx: &Ctx,
    step: &Step,
    step_idx: usize,
    step_state: &mut StepState<'a>,
    backdrop_state: &mut BackdropState<'a>,
    scenario_start: std::time::Instant,
    _effective_checks: &crate::assert::Assert,
    sched_died_during_hold: &mut bool,
) -> Result<()> {
    let mut scenario = ScenarioState::new(step_state, backdrop_state);

    // Any `?` out of apply_ops / apply_setup would bypass the
    // per-step teardown ordering; `drain_on_err!` kills payload
    // handles across step + backdrop (metric-emitting) before
    // propagating so a mid-scenario spawn failure still leaves a
    // usable sidecar.
    macro_rules! drain_on_err {
        ($scenario:expr, $e:expr) => {
            match $e {
                Ok(v) => v,
                Err(err) => {
                    $scenario.drain_all_payloads();
                    return Err(err);
                }
            }
        };
    }

    match step.hold {
        HoldSpec::Loop { interval } => {
            // Setup runs once before the loop.
            if !step.setup.is_empty() {
                let defs = step.setup.resolve(ctx);
                drain_on_err!(scenario, apply_setup(ctx, &mut scenario, &defs));
            }
            // Loop mode: apply ops repeatedly at interval until
            // the remaining scenario time is exhausted, or the
            // scheduler process exits — whichever fires first.
            let deadline = scenario_start + ctx.duration;
            while std::time::Instant::now() < deadline {
                drain_on_err!(scenario, apply_ops(ctx, &mut scenario, &step.ops));
                let remaining = deadline.saturating_duration_since(std::time::Instant::now());
                // Live `sched_pid()` read so a mid-loop
                // Op::ReplaceScheduler swap is watched at the NEW
                // pid, not the stale boot snapshot in ctx.
                if sleep_or_sched_died(remaining.min(interval), crate::vmm::rust_init::sched_pid())
                {
                    *sched_died_during_hold = true;
                    return Ok(());
                }
            }
        }
        _ => {
            // Ops first (e.g. parent cgroup creation), then
            // CgroupDef setup (children with workers).
            drain_on_err!(scenario, apply_ops(ctx, &mut scenario, &step.ops));
            if !step.setup.is_empty() {
                let defs = step.setup.resolve(ctx);
                drain_on_err!(scenario, apply_setup(ctx, &mut scenario, &defs));
            }

            // Write stimulus event after applying ops. Routes through
            // `crate::vmm::guest_comms::send_stimulus` (virtio-console
            // port-1 bulk channel). `is_guest` keeps the
            // `build_stimulus` walk off the host where the write would
            // no-op.
            if guest_comms::is_guest() {
                let payload = build_stimulus(&scenario_start, step_idx, &step.ops, &scenario);
                crate::vmm::guest_comms::send_stimulus(zerocopy::IntoBytes::as_bytes(&payload));
            }

            if guest_comms::is_guest() {
                crate::vmm::guest_comms::send_scenario_resume();
            }
            let hold_dur = match step.hold {
                HoldSpec::Frac(f) => Duration::from_secs_f64(ctx.duration.as_secs_f64() * f),
                HoldSpec::Fixed(d) => d,
                HoldSpec::Loop { .. } => unreachable!(),
            };
            let remaining = (scenario_start + ctx.duration)
                .saturating_duration_since(std::time::Instant::now());
            let hold_dur = hold_dur.min(remaining);
            // Live `sched_pid()` read — matches the loop arm above
            // so the hold watches the post-Op::ReplaceScheduler
            // pid, not the stale boot snapshot.
            if sleep_or_sched_died(hold_dur, crate::vmm::rust_init::sched_pid()) {
                *sched_died_during_hold = true;
                return Ok(());
            }
        }
    }

    Ok(())
}

/// Build a StimulusPayload from the current scenario state (step + backdrop).
///
/// # step_idx u16 saturation
///
/// `step_idx` is a `usize` on the caller side but the wire
/// `StimulusPayload.step_index` is a `u16` — the slot is sized for
/// realistic scenarios (≤ 65 536 distinct indices, `0..=u16::MAX`).
/// Any `step_idx` > `u16::MAX as usize` is clamped to `u16::MAX` by
/// `to_u16` below, with a `tracing::warn!` that names the overflow.
/// Downstream consumers of the StepStart wire frame therefore see
/// every step past index `u16::MAX` collapsed onto the same
/// `step_index` value (`u16::MAX`) — the ordering is preserved for
/// the first 65 536 steps (indices `0..=u16::MAX`), but labels
/// saturate and become ambiguous once the scenario crosses the
/// boundary. Scenarios that need to distinguish individual steps
/// past `u16::MAX` must widen the wire schema field; the
/// saturating-clip preserves visible wake ordering at the cost of
/// individuality in the deep tail.
fn build_stimulus(
    scenario_start: &std::time::Instant,
    step_idx: usize,
    ops: &[Op],
    state: &ScenarioState<'_, '_>,
) -> StimulusPayload {
    let mut op_kinds: u32 = 0;
    for op in ops {
        op_kinds |= 1 << op.discriminant();
    }

    let total_iterations: u64 = state
        .all_handles()
        .flat_map(|(_, h)| h.snapshot_iterations())
        .sum();

    let cgroup_count = state.step.cgroups.names().len() + state.backdrop.cgroups.names().len();
    let worker_count = state.step.handles.len() + state.backdrop.handles.len();

    // Saturate narrowing conversions for the wire schema: the
    // StimulusPayload fields are sized for realistic scenarios
    // (u32 ms, u16 counts) but `as u32` / `as u16` silently
    // wrap on overflow, poisoning downstream consumers. Log the
    // overflow so the operator sees which field exceeded its
    // bound and substitute MAX — clipped-high is a safer wire
    // value than silently wrapping to a small number.
    let to_u32 = |field: &str, v: u128| -> u32 {
        u32::try_from(v).unwrap_or_else(|_| {
            tracing::warn!(
                field,
                value = %v,
                "StimulusPayload field overflowed u32; saturating to u32::MAX",
            );
            u32::MAX
        })
    };
    let to_u16 = |field: &str, v: usize| -> u16 {
        u16::try_from(v).unwrap_or_else(|_| {
            tracing::warn!(
                field,
                value = v,
                "StimulusPayload field overflowed u16; saturating to u16::MAX",
            );
            u16::MAX
        })
    };

    // Encode the 1-indexed phase number per the framework's
    // phase convention -- the BASELINE (pre-first-Step) window owns
    // 0, scenario Step k publishes k + 1. Saturate at u16::MAX
    // (rather than wrap) so a pathological 65k-step scenario still
    // produces a clipped-high value the host parser can recognise
    // instead of silently rolling over.
    let phase_step_index: u16 = u16::try_from(step_idx)
        .ok()
        .and_then(|i| i.checked_add(1))
        .unwrap_or_else(|| {
            tracing::warn!(
                field = "step_index",
                value = step_idx,
                "StimulusPayload step_index overflowed u16 after 1-indexed encoding; saturating to u16::MAX",
            );
            u16::MAX
        });
    StimulusPayload {
        elapsed_ms: to_u32("elapsed_ms", scenario_start.elapsed().as_millis()),
        step_index: phase_step_index,
        op_count: to_u16("op_count", ops.len()),
        op_kinds,
        cgroup_count: to_u16("cgroup_count", cgroup_count),
        worker_count: to_u16("worker_count", worker_count),
        total_iterations,
    }
}

/// Validate that a MemPolicy's node set is consistent with the
/// cgroup's scenario intent — the cpuset the cgroup runs in and
/// the host topology.
///
/// # Empty-nodemask early return
///
/// Policies with no nodemask — [`MemPolicy::Default`] and
/// [`MemPolicy::Local`] — carry no node IDs to validate against,
/// so this function returns `Ok(())` unconditionally for them
/// (after the unknown-bit and mutual-exclusion flag guards run).
/// Every other variant — any variant carrying a nodemask,
/// currently [`MemPolicy::Bind`], [`MemPolicy::Preferred`],
/// [`MemPolicy::PreferredMany`], [`MemPolicy::Interleave`], and
/// [`MemPolicy::WeightedInterleave`] — reaches the cpuset /
/// host-topology coverage logic below.
///
/// # Why this is a scenario-intent check, not a kernel guard
///
/// ktstr writes `cpuset.cpus` on each cgroup but never writes
/// `cpuset.mems`, so `cpuset.mems` keeps its inherited default —
/// the permissive "all nodes" set in every ktstr deployment
/// shape (PID 1 inside the guest VM, cgroup root on the host).
/// The kernel's `set_mempolicy(2)` path always runs the policy's
/// nodemask through `mpol_set_nodemask` in `mm/mempolicy.c`, which
/// intersects it with the caller's `mems_allowed` before it is
/// stored on the task; because ktstr never narrows `mems_allowed`,
/// that intersection is an identity operation under ktstr's
/// deployment — the stored nodemask equals the one the caller
/// supplied, and the kernel never rejects or silently trims the
/// policy the way it would if `mems_allowed` were disjoint from
/// the requested set. Rejection of a mismatched policy is
/// therefore validator-only: if this function does not bail, the
/// policy lands on the syscall unchanged and `run_steps` commits
/// to running the worker with a misconfigured allocation target.
///
/// What the validator catches is a **scenario-design mismatch**:
/// you pinned CPUs on NUMA node X (via `CpusetSpec::Numa(X)`) but
/// asked the mempolicy to bind/prefer/interleave a disjoint node Y,
/// meaning the worker's compute is local to node X while its
/// allocations live on node Y — producing cross-socket traffic
/// that the test author almost certainly did not intend. Surface
/// the mismatch here before `run_steps` commits to the policy.
///
/// `MpolFlags::STATIC_NODES` is the rebind-behavior flag. Two
/// kernel sites encode the semantics: `mpol_set_nodemask` in
/// `mm/mempolicy.c` consumes the flag during policy creation (it
/// determines whether the supplied nodemask is stored absolute or
/// remapped against the caller's cpuset at install time), and
/// `mpol_rebind_policy` (same file) branches on the flag when the
/// cpuset's `mems_allowed` changes after the policy was installed
/// — with `STATIC_NODES` set, the stored nodemask is unchanged;
/// without it, the kernel remaps the nodemask against the new
/// `mems_allowed`. Since ktstr never rebinds `cpuset.mems` mid-run,
/// only the install-time semantics applies, and the flag is
/// effectively a cross-node-intent declaration for the validator's
/// purposes — a sign the author knows the intent is "allocations on
/// a node outside the CPU-affinity cpuset" and has opted in to
/// that shape.
///
/// # Flag-specific handling (in order of evaluation)
///
/// - `STATIC_NODES | RELATIVE_NODES` both set → bail: the kernel
///   rejects this combination with `EINVAL`; surfacing it here
///   names the offender before the syscall.
/// - `STATIC_NODES` only → the caller has declared intentional
///   cross-node placement. Skip the cpuset-intent check, but each
///   referenced node must exist on the host topology or the
///   kernel will reject the policy. Verify existence; bail with
///   the missing nodes if any.
/// - `RELATIVE_NODES` only → the nodemask is an ordinal into the
///   cpuset's allowed-nodes set. Cpuset coverage does not apply in
///   absolute-id terms, so bypass.
/// - No relevant flag set → enforce cpuset-intent coverage:
///   every policy node must appear in the cpuset's covered NUMA
///   nodes. Bail naming the uncovered nodes AND both escape
///   hatches (STATIC_NODES opt-in; widening the cpuset).
///
/// Reject `--flag` args whose bare name is not in the payload's
/// `known_flags` allowlist. Returns `Ok(())` when the payload
/// declared no allowlist (`known_flags: None`) — the opt-in
/// contract defaults to "permissive" so payloads wrapping
/// open-ended binaries (stress-ng, fio, schbench) aren't forced
/// to enumerate every flag their upstream tool accepts.
///
/// Recognises two flag shapes: `--foo` (flag-only) and
/// `--foo=value` (flag-with-attached-value). Non-flag args
/// (positional, `-short`, everything else) are passed through
/// without inspection — the allowlist scopes to long flags only.
///
/// Extracted out of `apply_ops`'s `Op::RunPayload` arm so the
/// validation is unit-testable without standing up a full Ctx
/// / scenario state. See the caller for how the allowlist is
/// threaded through Op::RunPayload execution.
fn validate_known_flags(payload: &crate::test_support::Payload, args: &[String]) -> Result<()> {
    let Some(allowlist) = payload.known_flags else {
        return Ok(());
    };
    for arg in args {
        let Some(flag_body) = arg.strip_prefix("--") else {
            continue;
        };
        // `split('=').next()` is infallible: `str::split` always
        // yields at least one element (the full string when no
        // separator is present). The prior `unwrap_or("")` fallback
        // was dead code — the empty-name branch below never fired
        // via this path since `flag_body` had already passed the
        // `strip_prefix("--")` filter above (leaving at least one
        // character). Kept the `name.is_empty()` guard in place
        // only to handle the degenerate `"--"` bare-dashes case,
        // which produces `flag_body = ""` → `name = ""`.
        let name = flag_body
            .split('=')
            .next()
            .expect("str::split always yields at least one element");
        if name.is_empty() {
            continue;
        }
        if !allowlist.contains(&name) {
            anyhow::bail!(
                "Op::RunPayload: payload '{}' received unknown flag \
                 '--{name}' — not in its known_flags allowlist \
                 {allowlist:?}. Check the spelling against the \
                 payload's declared flags; if '--{name}' is a new \
                 legitimate flag, add it to `Payload::known_flags`.",
                payload.name,
            );
        }
    }
    Ok(())
}

fn validate_mempolicy_cpuset(
    policy: &MemPolicy,
    flags: crate::workload::MpolFlags,
    cpuset: &BTreeSet<usize>,
    ctx: &Ctx,
    cgroup_name: &str,
) -> Result<()> {
    use crate::workload::MpolFlags;

    // Reject unknown bits before any other check. The `MpolFlags`
    // type is a `u32` bitfield covering three documented bits
    // (STATIC_NODES, RELATIVE_NODES, NUMA_BALANCING); any other bit
    // set in `flags` is either a user typo (raw-constructing the
    // struct with an arbitrary integer) or forward-compat from a
    // future kernel flag that this validator hasn't learned yet.
    // Either way, surfacing unknown bits here prevents a silent
    // semantic mismatch — the kernel would either reject with
    // EINVAL or (worse) treat the bit as a flag we don't model.
    let known_bits = MpolFlags::STATIC_NODES.bits()
        | MpolFlags::RELATIVE_NODES.bits()
        | MpolFlags::NUMA_BALANCING.bits();
    let unknown_bits = flags.bits() & !known_bits;
    if unknown_bits != 0 {
        anyhow::bail!(
            "cgroup '{}': MpolFlags contains unknown bit(s) {:#x} (known bits: \
             STATIC_NODES={:#x}, RELATIVE_NODES={:#x}, NUMA_BALANCING={:#x}); \
             refusing to forward to the kernel — update MpolFlags to model the \
             new bit before using it, or clear the bit at the call site",
            cgroup_name,
            unknown_bits,
            MpolFlags::STATIC_NODES.bits(),
            MpolFlags::RELATIVE_NODES.bits(),
            MpolFlags::NUMA_BALANCING.bits(),
        );
    }

    // `STATIC_NODES | RELATIVE_NODES` is a kernel-rejected combination —
    // `MPOL_F_STATIC_NODES` and `MPOL_F_RELATIVE_NODES` are mutually
    // exclusive (see `include/uapi/linux/mempolicy.h` + the
    // `sanitize_mpol_flags` helper in `mm/mempolicy.c`, which bails
    // with `EINVAL` if both are set). Fail early here instead of
    // letting the syscall return a generic error — the scenario
    // caller almost certainly meant one or the other, not both.
    if flags.contains(MpolFlags::STATIC_NODES) && flags.contains(MpolFlags::RELATIVE_NODES) {
        anyhow::bail!(
            "cgroup '{}': MpolFlags::STATIC_NODES and MpolFlags::RELATIVE_NODES are \
             mutually exclusive (the kernel will reject the set_mempolicy syscall with \
             EINVAL); pick whichever matches the intended semantics — STATIC_NODES \
             for absolute node ids that survive cpuset changes, RELATIVE_NODES for \
             cpuset-relative indices",
            cgroup_name,
        );
    }

    let policy_nodes = policy.node_set();
    if policy_nodes.is_empty() {
        return Ok(());
    }

    // `STATIC_NODES`: nodemask is treated as absolute node ids and NOT
    // intersected with the cpuset. The cpuset-coverage check below
    // does not apply, but we DO need to verify the referenced nodes
    // actually exist on the host — a policy pinning node 7 on a
    // 2-node host would fail at syscall time; surfacing it here
    // names the offender.
    if flags.contains(MpolFlags::STATIC_NODES) {
        let host_nodes = ctx.topo.numa_node_ids();
        let missing: Vec<usize> = policy_nodes
            .iter()
            .copied()
            .filter(|n| !host_nodes.contains(n))
            .collect();
        if !missing.is_empty() {
            anyhow::bail!(
                "cgroup '{}': MemPolicy with MpolFlags::STATIC_NODES references \
                 NUMA node(s) {:?} that do not exist on this host (host nodes: {:?}); \
                 the kernel will reject or silently drop the policy (Preferred can \
                 silently fall back to local allocation; Bind/Interleave reject with \
                 EINVAL) — fix the MemPolicy or pick a host with the required nodes",
                cgroup_name,
                missing,
                host_nodes,
            );
        }
        return Ok(());
    }

    // `RELATIVE_NODES`: nodemask is an ordinal into the cpuset's
    // allowed nodes, not an absolute node id set. The cpuset-coverage
    // check compares absolute ids, so it does not apply here — the
    // kernel does the relative-to-absolute remap internally. Trust
    // the caller and bypass the coverage bail, same shape as the
    // STATIC_NODES early return.
    if flags.contains(MpolFlags::RELATIVE_NODES) {
        return Ok(());
    }

    let cpuset_numa = ctx.topo.numa_nodes_for_cpuset(cpuset);
    let uncovered: Vec<usize> = policy_nodes
        .iter()
        .copied()
        .filter(|n| !cpuset_numa.contains(n))
        .collect();
    if !uncovered.is_empty() {
        anyhow::bail!(
            "cgroup '{}': MemPolicy references NUMA node(s) {:?} \
             outside the cpuset's coverage (cpuset covers node(s) \
             {:?}) — some or all of the worker's allocations would \
             live on NUMA nodes its CPUs cannot reach locally, \
             producing cross-socket allocation traffic that is \
             almost certainly unintended. Two fixes: \
             (a) add .mpol_flags(MpolFlags::STATIC_NODES) to \
             declare the cross-node placement intentional (the \
             flag survives cpuset rebinds; see MpolFlags doc), or \
             (b) widen the cpuset to cover the policy's nodes \
             (e.g. CpusetSpec::Numa(N) for each referenced N, or \
             a CpusetSpec::Exact set that spans both).",
            cgroup_name,
            uncovered,
            cpuset_numa,
        );
    }
    Ok(())
}

/// Each CgroupDef's `works` vec is iterated, spawning one WorkloadHandle
/// per WorkSpec entry. Multiple Works for the same cgroup produce multiple
/// handle entries with the same name key; Ops that filter by cgroup name
/// (StopCgroup, SetAffinity, etc.) naturally apply to all of them.
///
/// When `works` is empty, a single default WorkSpec is used (SpinWait, Normal,
/// ctx.workers_per_cgroup workers).
///
/// Cgroups created here route into step-local or backdrop state per
/// `state.target_backdrop`. A duplicate name (already tracked by
/// either state) bails — a [`CgroupDef`] must not silently shadow a
/// cgroup that another state slot has already created.
fn apply_setup(ctx: &Ctx, state: &mut ScenarioState<'_, '_>, defs: &[CgroupDef]) -> Result<()> {
    for def in defs {
        if state.cgroup_name_is_tracked(&def.name) {
            anyhow::bail!(
                "CgroupDef '{}' collides with a cgroup already tracked \
                 (by a prior Backdrop or step-local CgroupDef) — declare it \
                 in exactly one place; use a fresh name for the step-local cgroup",
                def.name,
            );
        }
        state.target_cgroups().add_cgroup_no_cpuset(&def.name)?;
        if let Some(ref cpuset_spec) = def.cpuset {
            let resolved = cpuset_spec.resolve_quiet(ctx);
            // workers_pct + empty cpuset combinations produce more
            // actionable diagnostics than the generic CpusetSpec
            // empty-mask rejection — surface them here before
            // validate's broader empty-Exact reject preempts the
            // per-pct context.
            //
            // Two distinct empty-cpuset misconfigurations:
            //
            //   (1) any WorkSpec sets BOTH workers(N) and
            //       workers_pct(P): the dual-set is the more
            //       fundamental error and must be resolved before
            //       the cpuset semantics matter. Surface "BOTH
            //       workers ... workers_pct" here rather than letting
            //       validate's empty-mask rejection mask it.
            //
            //   (2) one or more WorkSpecs set workers_pct only:
            //       enumerate every configured pct value so a
            //       multi-WorkSpec cgroup doesn't silently drop all
            //       but the first.
            if resolved.is_empty() {
                let works = def.merged_works();
                if let Some(dual_work) = works
                    .iter()
                    .find(|w| w.workers_pct.is_some() && w.num_workers.is_some())
                {
                    let n = dual_work
                        .num_workers
                        .expect("dual_work selected via num_workers.is_some()");
                    let pct = dual_work
                        .workers_pct
                        .expect("dual_work selected via workers_pct.is_some()");
                    anyhow::bail!(
                        "cgroup '{}': WorkSpec sets BOTH workers({n}) \
                         and workers_pct({pct}); pick one — \
                         workers_pct resolves the cpuset fraction at \
                         apply-setup time and is incompatible with an \
                         explicit count. The empty cpuset would \
                         otherwise mask this conflict; resolve the \
                         workers/workers_pct conflict first",
                        def.name,
                    );
                }
                let pcts: Vec<(usize, f64)> = works
                    .iter()
                    .enumerate()
                    .filter_map(|(i, w)| w.workers_pct.map(|p| (i, p)))
                    .collect();
                if !pcts.is_empty() {
                    let pct_display = if pcts.len() == 1 {
                        format!("workers_pct({})", pcts[0].1)
                    } else {
                        // Include positional indices so the operator
                        // can disambiguate when the same fraction is
                        // configured on multiple WorkSpecs (e.g.
                        // `[works[0]=0.5, works[2]=0.5]` shows which
                        // entries to adjust without grepping the test).
                        let list = pcts
                            .iter()
                            .map(|(i, p)| format!("works[{i}]={p}"))
                            .collect::<Vec<_>>()
                            .join(", ");
                        format!("workers_pct values [{list}]")
                    };
                    anyhow::bail!(
                        "cgroup '{}': {pct_display} on a cpuset of 0 \
                         CPU(s) would resolve to 0 workers; the cgroup \
                         would have no workers and downstream \
                         assertions would vacuously pass — narrow the \
                         cpuset, raise the fraction, or use \
                         `workers(N)` instead",
                        def.name,
                    );
                }
                // Fall-through for cpusets that resolve to empty
                // without workers_pct — i.e. cases where the slice
                // math (or topology shape) yields an empty BTreeSet
                // even though `validate` accepts the spec. Examples:
                // `Range { 0.0, 0.1 }` on a small usable set (the
                // truncated `(len * 0.1) as usize` rounds to 0, see
                // the `op_set_cpuset_narrow_to_empty_bails` test),
                // or `Llc(N)` on a pathological topology where LLC
                // N has no associated CPUs (memory-only NUMA node
                // attached to a separate LLC). Cases like
                // `Range { 0.0, 0.0 }` or `Disjoint { of: 0 }` do
                // NOT reach this branch in `Op::SetCpuset` — they
                // get rejected by validate first — but they DO
                // reach this branch here because apply_setup runs
                // resolve before validate (intentional: the Bundle
                // H workers_pct diagnostic at the dual_work / pcts
                // probes above needs to fire on empty-Exact +
                // workers_pct combinations before validate's
                // generic empty-Exact rejection preempts it). An
                // empty cpuset reaching `set_cpuset` silently
                // writes an empty mask to the cgroup; subsequent
                // worker spawns get no CPUs and every CPU-pinned
                // assertion vacuously passes. Bail here with the
                // cpuset_spec context so the operator sees which
                // spec resolved to empty and can adjust.
                anyhow::bail!(
                    "cgroup '{}': cpuset_spec {:?} resolved to 0 \
                     CPU(s); the cgroup would have no CPUs assigned \
                     and downstream worker spawns would fail or \
                     produce vacuous assertions — adjust the spec \
                     so it resolves to a non-empty cpuset on this \
                     topology",
                    def.name,
                    cpuset_spec,
                );
            }
            if let Err(reason) = cpuset_spec.validate(ctx) {
                anyhow::bail!(
                    "cgroup '{}': CpusetSpec validation failed: {}",
                    def.name,
                    reason
                );
            }
            ctx.cgroups.set_cpuset(&def.name, &resolved)?;
            state.record_cpuset(&def.name, resolved);
        }
        if let Some(ref nodes) = def.cpuset_mems {
            // The cpuset.mems write must succeed before any task
            // moves into the cgroup; cpuset_update_task_spread will
            // SIGKILL or fail allocations otherwise. Surfacing the
            // error here (instead of at move_tasks time) lets the
            // operator see the bad NUMA spec at setup, before the
            // worker spawn pays its cost.
            ctx.cgroups.set_cpuset_mems(&def.name, nodes)?;
        }
        if let Some(ref cpu) = def.cpu {
            // cpu.weight: kernel range is 1..=10000 per
            // Documentation/admin-guide/cgroup-v2.rst. Reject at
            // setup so a 0 / 12000 from a typo fails fast instead
            // of returning EINVAL from the kernel write.
            if let Some(w) = cpu.weight {
                if !(1..=10_000).contains(&w) {
                    anyhow::bail!(
                        "cgroup '{}': cpu.weight {w} out of range 1..=10000",
                        def.name,
                    );
                }
                ctx.cgroups.set_cpu_weight(&def.name, w)?;
            }
            // cpu.max: writing requires `+cpu` in subtree_control;
            // CgroupManager::setup with enable_cpu_controller=true
            // turns it on. quota=0 with period>0 would reject every
            // schedule slice in the kernel; reject here with a
            // clearer message.
            if cpu.max_period_us == 0 {
                anyhow::bail!("cgroup '{}': cpu.max period must be > 0 (got 0)", def.name,);
            }
            if let Some(q) = cpu.max_quota_us
                && q == 0
            {
                anyhow::bail!(
                    "cgroup '{}': cpu.max quota must be > 0 when set; \
                     use cpu_unlimited() to remove the cap",
                    def.name,
                );
            }
            // Always emit the cpu.max write so the period field is
            // recorded even when quota is None. Aligns with the
            // kernel's `"max <period>"` write format.
            ctx.cgroups
                .set_cpu_max(&def.name, cpu.max_quota_us, cpu.max_period_us)?;
        }
        if let Some(ref mem) = def.memory {
            // Order: max first, then high (high <= max is the
            // operator-meaningful constraint per cgroup-v2 docs;
            // kernel allows any ordering but writing max first
            // means a high write fails clearly when high>max).
            // swap_max is independent of the max/high/low triple
            // and lands last in the memory block.
            ctx.cgroups.set_memory_max(&def.name, mem.max)?;
            ctx.cgroups.set_memory_high(&def.name, mem.high)?;
            ctx.cgroups.set_memory_low(&def.name, mem.low)?;
            // memory.swap.max only exists when the kernel was built
            // with CONFIG_SWAP. On a swap-disabled kernel the file is
            // absent and write returns ENOENT. Match the per-knob
            // explicit-set semantics of the pids block: emit the
            // write only when the user opted in via memory_swap_max
            // / memory_swap_unlimited. swap_max=None means "the
            // user never asked for a swap cap" — in that case the
            // kernel default (unlimited on swap-enabled, no file on
            // swap-disabled) is exactly what we want, and skipping
            // the write keeps swap-disabled kernels viable for
            // tests that just set memory_max.
            if mem.swap_max.is_some() {
                ctx.cgroups.set_memory_swap_max(&def.name, mem.swap_max)?;
            }
        }
        if let Some(ref io) = def.io
            && let Some(w) = io.weight
        {
            if !(1..=10_000).contains(&w) {
                anyhow::bail!(
                    "cgroup '{}': io.weight {w} out of range 1..=10000",
                    def.name,
                );
            }
            ctx.cgroups.set_io_weight(&def.name, w)?;
        }
        if let Some(ref pids) = def.pids {
            // pids.max: zero is a foot-cannon (no fork ever), so
            // reject before the syscall — the kernel would accept
            // 0 but the workload would silently halt every fork
            // including the futex-helper threads spawned by some
            // WorkType variants. There's no kernel sentinel for
            // "no fork ever"; the explicit None path writes "max".
            if let Some(0) = pids.max {
                anyhow::bail!(
                    "cgroup '{}': pids.max must be > 0; use \
                     pids_unlimited() to remove the cap",
                    def.name,
                );
            }
            ctx.cgroups.set_pids_max(&def.name, pids.max)?;
        }
        // Materialize the per-WorkSpec values with cgroup-level
        // defaults merged in. `merged_works` substitutes a single
        // `WorkSpec::default()` when `def.works` is empty (matching
        // the historical default-substitution rule pinned by
        // `apply_setup_substitutes_default_workspec_when_works_empty`)
        // and merges `default_nice` / `default_comm` / `default_uid`
        // / `default_gid` / `default_numa_node` into each WorkSpec
        // whose own field is unset, regardless of the order builder
        // methods were called in. This is what makes
        // `def.nice(5).work(spec)` and `def.work(spec).nice(5)`
        // equivalent. `pcomm` lives ONLY on `WorkSpec`; the
        // `CgroupDef::pcomm` builder writes it into every WorkSpec
        // directly, so the per-WorkSpec value below is the
        // authoritative source for the pcomm dispatch.
        let effective_works = def.merged_works();
        for work in &effective_works {
            if let Err(reason) = work.mem_policy.validate() {
                anyhow::bail!("cgroup '{}': {}", def.name, reason);
            }
        }
        // Clone the cpuset out so we don't keep a borrow into
        // `state` across the mutable spawn calls below.
        let cgroup_cpuset: Option<BTreeSet<usize>> = state.lookup_cpuset(&def.name).cloned();
        if let Some(ref resolved) = cgroup_cpuset {
            for work in &effective_works {
                validate_mempolicy_cpuset(
                    &work.mem_policy,
                    work.mpol_flags,
                    resolved,
                    ctx,
                    &def.name,
                )?;
            }
        }
        // Per-WorkSpec pcomm dispatch. A WorkSpec with `pcomm =
        // Some(value)` joins a thread-group leader keyed on
        // `value`: every WorkSpec sharing the same `pcomm` value
        // coalesces into ONE forked leader per group, and every
        // thread inside the leader observes
        // `task->group_leader->comm == pcomm`. WorkSpecs with
        // `pcomm = None` (or an empty pcomm string, which is
        // treated as `None`) spawn via the conventional fork path
        // — one process per worker.
        //
        // Coalescing key: the pcomm string itself. Different pcomm
        // values inside the same CgroupDef produce different
        // leaders (more flexible than rejecting heterogeneity, and
        // matches the "model real workloads like `chrome` next to
        // `java` in one cgroup" use case).
        //
        // pcomm > 15 bytes triggers a one-shot warning so operators
        // see the kernel-side truncation that
        // `__set_task_comm`/`prctl(PR_SET_NAME)` performs at
        // `TASK_COMM_LEN - 1`. The warning fires at apply-setup
        // before any spawn, so a misconfigured pcomm is visible
        // before any work runs.
        //
        // Resolve every WorkSpec's `num_workers` / `work_type` /
        // `affinity` once up front; the same triple is used by
        // both dispatch paths and the resolution context (ctx,
        // cgroup_cpuset) is identical for every WorkSpec inside
        // this CgroupDef.
        let mut resolved_works: Vec<crate::workload::WorkSpec> =
            Vec::with_capacity(effective_works.len());
        for work in &effective_works {
            // Resolve `workers_pct` against the cgroup's cpuset (or
            // the topology-usable cpuset when the cgroup inherits)
            // and synthesize a `num_workers` value before the rest of
            // the dispatch. Shares the resolution helper with
            // Op::SpawnWorkers so the two paths produce identical worker
            // counts for the same `(pct, cpuset_size)` pair.
            let cpuset_size = cgroup_cpuset
                .as_ref()
                .map_or_else(|| ctx.topo.usable_cpuset().len(), |s| s.len());
            let work = work.clone().resolve_workers_pct(cpuset_size, &def.name)?;
            let n = crate::scenario::resolve_num_workers(&work, ctx.workers_per_cgroup, &def.name)?;
            let effective_work_type = crate::workload::resolve_work_type(
                &work.work_type,
                ctx.work_type_override.as_ref(),
                def.swappable,
                n,
            );
            let affinity = crate::scenario::intent_for_spawn(
                &work.affinity,
                cgroup_cpuset.as_ref(),
                ctx.topo,
            )?;
            resolved_works.push(crate::workload::WorkSpec {
                work_type: effective_work_type,
                sched_policy: work.sched_policy,
                num_workers: Some(n),
                affinity,
                mem_policy: work.mem_policy.clone(),
                mpol_flags: work.mpol_flags,
                nice: work.nice,
                comm: work.comm.clone(),
                pcomm: work.pcomm.clone(),
                uid: work.uid,
                gid: work.gid,
                numa_node: work.numa_node,
                workers_pct: None,
            });
        }

        // Partition by pcomm value. `pcomm_groups` is keyed on the
        // pcomm string; insertion order tracked via a parallel
        // `pcomm_order` vec so the spawn order is stable
        // (BTreeMap iteration would reorder by string sort).
        let mut pcomm_groups: std::collections::HashMap<String, Vec<crate::workload::WorkSpec>> =
            std::collections::HashMap::new();
        let mut pcomm_order: Vec<String> = Vec::new();
        let mut non_pcomm_works: Vec<crate::workload::WorkSpec> = Vec::new();
        for work in resolved_works {
            match &work.pcomm {
                Some(value) if !value.is_empty() => {
                    let key = value.to_string();
                    if !pcomm_groups.contains_key(&key) {
                        pcomm_order.push(key.clone());
                    }
                    pcomm_groups.entry(key).or_default().push(work);
                }
                _ => non_pcomm_works.push(work),
            }
        }

        // Spawn non-pcomm WorkSpecs via the conventional fork path
        // (one WorkloadHandle per WorkSpec, one move_tasks call
        // per spawn).
        for work in non_pcomm_works {
            let n = work.num_workers.expect("num_workers resolved above");
            let wl = WorkloadConfig {
                num_workers: n,
                affinity: work.affinity.clone(),
                work_type: work.work_type.clone(),
                sched_policy: work.sched_policy,
                mem_policy: work.mem_policy.clone(),
                mpol_flags: work.mpol_flags,
                nice: work.nice,
                // scenario-engine spawns are always Fork; pcomm is the only thread-mode path.
                clone_mode: Default::default(),
                comm: work.comm.clone(),
                uid: work.uid,
                gid: work.gid,
                numa_node: work.numa_node,
                composed: Vec::new(),
            };
            let mut h = WorkloadHandle::spawn(&wl)?;
            ctx.cgroups.move_tasks(&def.name, &h.worker_pids())?;
            h.start();
            state.target_handles().push((def.name.to_string(), h));
        }

        // Spawn one thread-group leader per unique pcomm value.
        // Each leader hosts every WorkSpec that shares its pcomm.
        for pcomm in pcomm_order {
            if pcomm.len() > 15 {
                tracing::warn!(
                    cgroup = %def.name,
                    pcomm = %pcomm,
                    len = pcomm.len(),
                    "WorkSpec::pcomm exceeds TASK_COMM_LEN-1 (15 bytes); kernel \
                     `__set_task_comm` will truncate to the leading 15 bytes",
                );
            }
            let works_for_pcomm = pcomm_groups
                .remove(&pcomm)
                .expect("pcomm key inserted during partition pass");
            // glibc setresuid/setresgid broadcasts via NPTL signalling
            // to every thread in the tgid; coalesced WorkSpecs that
            // disagree on uid/gid would race the leader's credentials
            // out from under the other group's threads. Reject mixed
            // values upfront so the misconfiguration surfaces here
            // rather than as a runtime credential flap.
            if works_for_pcomm.len() > 1 {
                let first_uid = works_for_pcomm[0].uid;
                let first_gid = works_for_pcomm[0].gid;
                for (i, w) in works_for_pcomm.iter().enumerate().skip(1) {
                    if w.uid != first_uid {
                        anyhow::bail!(
                            "cgroup '{}' pcomm '{}': WorkSpec[0].uid={:?} differs from \
                             WorkSpec[{}].uid={:?}; pcomm-coalesced WorkSpecs must \
                             agree on uid (NPTL setresuid is broadcast to every thread \
                             in the tgid)",
                            def.name,
                            pcomm,
                            first_uid,
                            i,
                            w.uid,
                        );
                    }
                    if w.gid != first_gid {
                        anyhow::bail!(
                            "cgroup '{}' pcomm '{}': WorkSpec[0].gid={:?} differs from \
                             WorkSpec[{}].gid={:?}; pcomm-coalesced WorkSpecs must \
                             agree on gid (NPTL setresgid is broadcast to every thread \
                             in the tgid)",
                            def.name,
                            pcomm,
                            first_gid,
                            i,
                            w.gid,
                        );
                    }
                }
            }
            // Container-leader credentials. Fall back to the first
            // WorkSpec's uid/gid when no CgroupDef-level default is
            // set: glibc's `setresuid` is broadcast to every thread
            // in the tgid via NPTL signalling, so a worker thread's
            // setresuid would eventually drop the leader's
            // credentials anyway. Pre-applying it on the leader
            // closes the root-uid window between fork and the first
            // worker's setresuid call. When the WorkSpec also has
            // `uid = None` the container stays at the parent's
            // credentials (root in the test harness, the harness's
            // euid otherwise) — the WorkSpec's lack-of-uid means
            // "inherit the parent" anyway.
            let container_uid = def
                .default_uid
                .or_else(|| works_for_pcomm.first().and_then(|w| w.uid));
            let container_gid = def
                .default_gid
                .or_else(|| works_for_pcomm.first().and_then(|w| w.gid));
            let mut h = WorkloadHandle::spawn_pcomm_cgroup(
                &pcomm,
                container_uid,
                container_gid,
                &works_for_pcomm,
            )?;
            ctx.cgroups.move_tasks(&def.name, &h.worker_pids())?;
            h.start();
            state.target_handles().push((def.name.to_string(), h));
        }
        // After synthetic workers are in place, spawn the optional
        // userspace payload inside the same cgroup. The payload runs
        // concurrently with the WorkSpec groups; its metrics are recorded
        // to the sidecar via the guest-to-host SHM ring when the
        // handle is killed at step-teardown. Spawning after the WorkSpec
        // handles lets the cgroup cpuset + mempolicy settle first so
        // the binary inherits a stable placement.
        if let Some(payload) = def.payload {
            // Composite-key dedup: the same payload CAN live in a
            // different cgroup, but two copies in THIS cgroup would
            // collide on teardown (one handle masks the other in
            // the sidecar). Reject upfront with the same error
            // shape as the Op::RunPayload path.
            if let Some(existing) =
                state.find_live_payload_with_cgroup(payload.name, def.name.as_ref())
            {
                anyhow::bail!(
                    "CgroupDef::workload: payload '{}' already running in cgroup '{}' (spawned by {}) — \
                     declare it in exactly one place per cgroup",
                    payload.name,
                    def.name,
                    existing.source.describe(),
                );
            }
            let handle = crate::scenario::payload_run::PayloadRun::new(ctx, payload)
                .in_cgroup(def.name.clone())
                .spawn()
                .map_err(|e| {
                    anyhow::anyhow!(
                        "cgroup '{}': spawn payload '{}': {:#}",
                        def.name,
                        payload.name,
                        e,
                    )
                })?;
            state.target_payload_handles().push(PayloadEntry {
                cgroup: def.name.to_string(),
                source: PayloadSource::CgroupDefWorkload,
                handle,
            });
        }
    }
    Ok(())
}

/// Apply a slice of Ops to the running state.
///
/// Ops that create new entities (`AddCgroup`, `Spawn`, `SpawnHost`,
/// `RunPayload`) route into step-local state by default, or into
/// backdrop when the Backdrop's initial setup phase is active.
/// Ops that read or mutate existing entities (`SetCpuset`,
/// `ClearCpuset`, `SwapCpusets`, `SetAffinity`, `MoveAllTasks`,
/// `RemoveCgroup`, `StopCgroup`, `WaitPayload`, `KillPayload`)
/// resolve the target name against step-local first, then backdrop
/// — so a Step's ops can reach into Backdrop-declared cgroups by
/// name without the Backdrop leaking implementation details.
fn apply_ops(ctx: &Ctx, state: &mut ScenarioState<'_, '_>, ops: &[Op]) -> Result<()> {
    // Pre-pass: fold runs of adjacent `Op::WriteKernelCold`
    // singletons into one merged op so that multi-CPU seeds
    // (e.g. `with_uptime` writing per-CPU `rq.clock` on every CPU
    // at the same instant) land in ONE freeze rendezvous rather
    // than N — N separate rendezvous cycles would produce
    // observable inter-CPU skew.
    //
    // Only `Op::WriteKernelCold` merges in this pre-pass; reads
    // stay one-per-rendezvous until a wire-format follow-up adds
    // per-entry direction + tag (needed so multi-read batches can
    // route each reply back to its caller's tag). Any non-cold-
    // write op is a hard barrier — including hot variants, every
    // other Op variant, and `Op::WriteKernelHot`. Caller-supplied
    // `Op::WriteKernelCold` already containing multiple writes
    // passes through unchanged.
    let merged = merge_adjacent_cold_writes(ops);
    for op in &merged {
        match op {
            Op::AddCgroup { name } => {
                // Mirror the collision check in `apply_setup`
                // (`CgroupDef`) so the same name declared via `Op`
                // is rejected the same way. Without this, an
                // `Op::AddCgroup` could silently shadow a
                // Backdrop-owned or step-local `CgroupDef`-created
                // cgroup and the two writers could clobber each
                // other's cpuset / subtree_control state.
                if state.cgroup_name_is_tracked(name) {
                    anyhow::bail!(
                        "Op::AddCgroup '{}' collides with a cgroup already \
                         tracked (by a prior Backdrop or step-local CgroupDef) — \
                         declare it in exactly one place; use a fresh name for \
                         the step-local cgroup",
                        name,
                    );
                }
                state.target_cgroups().add_cgroup_no_cpuset(name)?;
            }
            Op::AddCgroupDef { def } => {
                // Delegate to `apply_setup` so cpuset, cpu / memory /
                // io / pids knobs, and worker spawning all run
                // through the same code path that a Step's
                // `with_defs` setup pass uses. The collision check
                // and workers_pct / empty-cpuset diagnostics carry
                // over via the delegation; controller-required
                // tracking is a sibling concern wired separately at
                // `required_controllers` (see absorb_op's
                // Op::AddCgroupDef arm) so the parent's
                // subtree_control has the def's controllers enabled
                // before this dispatch runs. The only difference
                // from Step::with_defs is the timing (apply-ops vs
                // setup).
                apply_setup(ctx, state, std::slice::from_ref(def))?;
            }
            Op::RemoveCgroup { cgroup } => {
                // Stop workers + payload binaries in this cgroup
                // before the cgroupfs removal. A live process in the
                // cgroup makes `rmdir` fail with EBUSY; kill the
                // payload handles first so the cgroup frees up.
                state.drain_payloads_for_cgroup(cgroup);
                state.drop_handles_for_cgroup(cgroup);
                state.forget_cpuset(cgroup);
                // Diagnostic breadcrumbs for the typo-late-surfacing
                // failure mode that permissive RemoveCgroup makes
                // possible: a typo'd cgroup name now no-ops silently
                // against the kernel, then a downstream op
                // referencing the intended Backdrop cgroup hits
                // kernel-level "cgroup missing" with no obvious link
                // back to the typo. Two complementary warns:
                //
                // (1) RemoveCgroup against a Backdrop-tracked name
                //     — operator can grep the log to correlate a
                //     later "cgroup missing" error with the
                //     intentional removal source.
                // (2) RemoveCgroup against a name NOT in any tracked
                //     set — could be a typo OR a second-remove of a
                //     name already forgotten by a prior RemoveCgroup;
                //     dump both the Backdrop and step-local cgroup
                //     name lists so the operator can compare and
                //     find the off-by-one. Fires unconditionally on
                //     unknown names (no `backdrop non-empty` gate)
                //     so typos in step-local-only scenarios are also
                //     caught.
                //
                // Order matters: these membership checks must run
                // BEFORE the `forget` calls below. Reordering them
                // after `forget` would prune the name from both
                // `names()` lists, making `in_backdrop` and `in_step`
                // both observe `false` — warn (1) would never fire and
                // warn (2) would fire spuriously on every RemoveCgroup.
                let in_backdrop = state
                    .backdrop
                    .cgroups
                    .names()
                    .iter()
                    .any(|n| n == &**cgroup);
                let in_step = state.step.cgroups.names().iter().any(|n| n == &**cgroup);
                if in_backdrop {
                    tracing::warn!(
                        cgroup = %cgroup,
                        "Op::RemoveCgroup removed a Backdrop-owned cgroup mid-scenario; \
                         unless this name is re-added by a later Op::AddCgroup, \
                         downstream ops referencing it will see kernel-level \
                         `cgroup missing` errors. If this removal was unintended \
                         (e.g. typo'd cgroup name that coincidentally matched a \
                         Backdrop entry), check the test source for the intended \
                         Backdrop cgroup.",
                    );
                } else if !in_step {
                    tracing::warn!(
                        cgroup = %cgroup,
                        backdrop_cgroups = ?state.backdrop.cgroups.names(),
                        step_cgroups = ?state.step.cgroups.names(),
                        "Op::RemoveCgroup target '{cgroup}' matches no step-local \
                         or Backdrop-owned cgroup — could be a typo or a \
                         second-remove of an already-forgotten name. Compare \
                         against the listed Backdrop and step cgroups; if a \
                         downstream op later hits kernel-level `cgroup missing` \
                         on a similar name, the typo here is the probable source.",
                    );
                }
                // Drop the name from step/backdrop tracking BEFORE
                // the rmdir so a later AddCgroup with the same name
                // doesn't collide against a stale entry, and the
                // CgroupGroup::drop teardown path doesn't attempt
                // to rmdir an already-removed dir.
                state.step.cgroups.forget(cgroup);
                state.backdrop.cgroups.forget(cgroup);
                // ENOENT is expected here only as a TOCTOU outcome:
                // `CgroupManager::remove_cgroup` first checks
                // `p.exists()` and returns `Ok(())` when the dir is
                // already gone, so a clean "already removed by a
                // prior op" case never reaches this error arm. The
                // remaining ENOENT path is the narrow race where the
                // dir is unlinked by another process between
                // `exists()` and `fs::remove_dir(&p)`, which is
                // benign — the post-condition we want (no dir) still
                // holds. Every other error — EBUSY from a surviving
                // task, EACCES from a permissions regression, I/O
                // errors from a broken cgroupfs mount — gets logged
                // so the failure surfaces in test output instead of
                // being swallowed by `let _ = `.
                if let Err(err) = ctx.cgroups.remove_cgroup(cgroup)
                    && !crate::scenario::is_io_not_found(&err)
                {
                    let hint = crate::scenario::remove_cgroup_errno_hint(&err).unwrap_or("");
                    tracing::warn!(
                        cgroup = %cgroup,
                        err = %format!("{err:#}"),
                        hint,
                        "Op::RemoveCgroup: remove_cgroup returned non-ENOENT error",
                    );
                }
            }
            Op::SetCpuset { cgroup, cpus } => {
                if let Err(reason) = cpus.validate(ctx) {
                    anyhow::bail!(
                        "cgroup '{}': CpusetSpec validation failed: {}",
                        cgroup,
                        reason
                    );
                }
                let resolved = cpus.resolve_quiet(ctx);
                // Symmetric with apply_setup's empty-resolved bail.
                // An Op::SetCpuset that narrows mid-scenario to 0
                // CPUs would silently re-mask the cgroup to empty
                // and break every running worker that depended on
                // it. Example cases that pass validate but resolve
                // to empty: `Range { start, end }` where the slice
                // math truncates to an empty range on a small
                // topology (the `op_set_cpuset_narrow_to_empty_bails`
                // test exercises `Range { 0.0, 0.1 }` on 4 CPUs),
                // or `Llc(N)` on a pathological topology where the
                // Nth LLC has no associated CPUs (memory-only NUMA
                // node attached to a separate LLC). Bail with the
                // spec context so the operator can see which
                // mid-scenario narrow produced the empty
                // resolution.
                if resolved.is_empty() {
                    anyhow::bail!(
                        "cgroup '{}': Op::SetCpuset spec {:?} \
                         resolved to 0 CPU(s); narrowing a live \
                         cgroup to empty would leave running \
                         workers without CPUs and downstream \
                         assertions would vacuously pass — adjust \
                         the spec so it resolves to a non-empty \
                         cpuset on this topology, or use \
                         Op::ClearCpuset if the intent was to \
                         release the cpuset restriction (allow all \
                         CPUs)",
                        cgroup,
                        cpus,
                    );
                }
                ctx.cgroups.set_cpuset(cgroup, &resolved)?;
                state.record_cpuset(cgroup, resolved);
            }
            Op::ClearCpuset { cgroup } => {
                ctx.cgroups.clear_cpuset(cgroup)?;
                state.forget_cpuset(cgroup);
            }
            Op::SwapCpusets { a, b } => {
                // Read current cpusets from the cgroup filesystem, swap them.
                let cpus_a = read_cpuset(ctx, a);
                let cpus_b = read_cpuset(ctx, b);
                if let Some(ca) = cpus_a {
                    ctx.cgroups.set_cpuset(b, &ca)?;
                    state.record_cpuset(b, ca);
                }
                if let Some(cb) = cpus_b {
                    ctx.cgroups.set_cpuset(a, &cb)?;
                    state.record_cpuset(a, cb);
                }
            }
            Op::SpawnWorkers { cgroup, work } => {
                if let Err(reason) = work.mem_policy.validate() {
                    anyhow::bail!("cgroup '{}': {}", cgroup, reason);
                }
                let cgroup_cpuset: Option<BTreeSet<usize>> = state.lookup_cpuset(cgroup).cloned();
                let cpuset_size = cgroup_cpuset
                    .as_ref()
                    .map_or_else(|| ctx.topo.usable_cpuset().len(), |s| s.len());
                let work = work.clone().resolve_workers_pct(cpuset_size, cgroup)?;
                let n =
                    crate::scenario::resolve_num_workers(&work, ctx.workers_per_cgroup, cgroup)?;
                if let Some(ref resolved) = cgroup_cpuset {
                    validate_mempolicy_cpuset(
                        &work.mem_policy,
                        work.mpol_flags,
                        resolved,
                        ctx,
                        cgroup,
                    )?;
                }
                let affinity = crate::scenario::intent_for_spawn(
                    &work.affinity,
                    cgroup_cpuset.as_ref(),
                    ctx.topo,
                )?;
                let wl = WorkloadConfig {
                    num_workers: n,
                    affinity,
                    work_type: work.work_type.clone(),
                    sched_policy: work.sched_policy,
                    mem_policy: work.mem_policy.clone(),
                    mpol_flags: work.mpol_flags,
                    nice: work.nice,
                    // scenario-engine spawns are always Fork; pcomm is the only thread-mode path.
                    clone_mode: Default::default(),
                    comm: work.comm.clone(),
                    uid: work.uid,
                    gid: work.gid,
                    numa_node: work.numa_node,
                    composed: Vec::new(),
                };
                let mut h = WorkloadHandle::spawn(&wl)?;
                ctx.cgroups.move_tasks(cgroup, &h.worker_pids())?;
                h.start();
                state.target_handles().push((cgroup.to_string(), h));
            }
            Op::StopCgroup { cgroup } => {
                state.drain_payloads_for_cgroup(cgroup);
                state.drop_handles_for_cgroup(cgroup);
            }
            Op::SetAffinity { cgroup, affinity } => {
                let cgroup_cpuset: Option<BTreeSet<usize>> = state.lookup_cpuset(cgroup).cloned();
                let resolved = crate::scenario::resolve_affinity_for_cgroup(
                    affinity,
                    cgroup_cpuset.as_ref(),
                    ctx.topo,
                )?;
                // Materialise the Random pool into a Vec once before
                // walking handles. `IndexedRandom::sample` requires
                // slice indexing, which `BTreeSet` does not provide;
                // without this hoist the per-handle inner arm would
                // re-collect the same pool on every matching handle
                // (and a single cgroup name can carry multiple handles
                // when a `CgroupDef::works` vec or repeated `Op::SpawnWorkers`
                // populates more than one). The pool is invariant
                // across handles for a given resolved affinity.
                //
                // Invariant: `resolve_affinity_for_cgroup` bails on
                // `RandomSubset` with an empty pool or `count == 0`
                // before this match, so the Random arm here always
                // sees a non-empty pool and count > 0. The match
                // guard is gone; the former defensive no-op arm is
                // replaced with an `unreachable!()` inside the Random
                // arm — a regression that reintroduced the empty case
                // would trip the panic at this site (both debug and
                // release) instead of silently no-op'ing a
                // SetAffinity. Mirrors the same enforcement in
                // `flatten_for_spawn` at
                // `crate::scenario::flatten_for_spawn`, so both
                // consumer sites of ResolvedAffinity::Random share
                // identical regression surfaces.
                let random_pool: Vec<usize> =
                    if let ResolvedAffinity::Random { from, .. } = &resolved {
                        from.iter().copied().collect()
                    } else {
                        Vec::new()
                    };
                for (name, handle) in state.all_handles() {
                    if name.as_str() == *cgroup {
                        match &resolved {
                            ResolvedAffinity::None => {}
                            ResolvedAffinity::Fixed(cpus) => {
                                for idx in 0..handle.worker_pids().len() {
                                    if let Err(e) = handle.set_affinity(idx, cpus) {
                                        tracing::warn!(
                                            cgroup = %cgroup,
                                            idx,
                                            err = %format!("{e:#}"),
                                            "Op::SetAffinity Fixed: handle.set_affinity failed; \
                                             worker keeps prior affinity"
                                        );
                                    }
                                }
                            }
                            ResolvedAffinity::Random { from, count } => {
                                if from.is_empty() || *count == 0 {
                                    // Invariant: resolve_affinity_for_cgroup
                                    // bails on empty pool / count=0 before
                                    // this match. Reaching here means a
                                    // future caller constructed
                                    // ResolvedAffinity::Random directly
                                    // (bypassing the resolver). Panic loudly
                                    // so the regression surfaces at the
                                    // construction site instead of producing
                                    // an empty sched_setaffinity mask that
                                    // the kernel rejects with EINVAL —
                                    // matches the unreachable!() pattern in
                                    // flatten_for_spawn (scenario/mod.rs).
                                    unreachable!(
                                        "ResolvedAffinity::Random {{ from={from:?}, count={count} }} \
                                         reached Op::SetAffinity with empty pool or count==0 — \
                                         resolve_affinity_for_cgroup is supposed to bail on those \
                                         cases (no-silent-drops invariant). Audit the new caller \
                                         that constructed it.",
                                    );
                                }
                                use rand::seq::IndexedRandom;
                                for idx in 0..handle.worker_pids().len() {
                                    let chosen: BTreeSet<usize> = random_pool
                                        .sample(&mut rand::rng(), *count)
                                        .copied()
                                        .collect();
                                    if let Err(e) = handle.set_affinity(idx, &chosen) {
                                        tracing::warn!(
                                            cgroup = %cgroup,
                                            idx,
                                            err = %format!("{e:#}"),
                                            "Op::SetAffinity Random: handle.set_affinity failed; \
                                             worker keeps prior affinity"
                                        );
                                    }
                                }
                            }
                            ResolvedAffinity::SingleCpu(cpu) => {
                                let cpus: BTreeSet<usize> = [*cpu].into_iter().collect();
                                for idx in 0..handle.worker_pids().len() {
                                    if let Err(e) = handle.set_affinity(idx, &cpus) {
                                        tracing::warn!(
                                            cgroup = %cgroup,
                                            idx,
                                            cpu = *cpu,
                                            err = %format!("{e:#}"),
                                            "Op::SetAffinity SingleCpu: handle.set_affinity failed; \
                                             worker keeps prior affinity"
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Op::SpawnHost { work } => {
                if let Err(reason) = work.mem_policy.validate() {
                    anyhow::bail!("SpawnHost: {}", reason);
                }
                let n =
                    crate::scenario::resolve_num_workers(work, ctx.workers_per_cgroup, "<host>")?;
                let affinity = crate::scenario::intent_for_spawn(&work.affinity, None, ctx.topo)?;
                let wl = WorkloadConfig {
                    num_workers: n,
                    affinity,
                    work_type: work.work_type.clone(),
                    sched_policy: work.sched_policy,
                    mem_policy: work.mem_policy.clone(),
                    mpol_flags: work.mpol_flags,
                    nice: work.nice,
                    // scenario-engine spawns are always Fork; pcomm is the only thread-mode path.
                    clone_mode: Default::default(),
                    comm: work.comm.clone(),
                    uid: work.uid,
                    gid: work.gid,
                    numa_node: work.numa_node,
                    composed: Vec::new(),
                };
                let mut h = WorkloadHandle::spawn(&wl)?;
                h.start();
                // Empty string key: workers in parent cgroup, not a managed cgroup.
                state.target_handles().push((String::new(), h));
            }
            Op::MoveAllTasks { from, to } => {
                // A step-local MoveAllTasks that pulls from a
                // Backdrop-owned cgroup into a step-local cgroup
                // would strand persistent workers inside a cgroup
                // that gets rmdir'd at step boundary. Reject
                // explicitly. Ops running inside the Backdrop's
                // own setup pass (`target_backdrop`) stay exempt.
                if !state.target_backdrop
                    && state.cgroup_name_is_backdrop(from)
                    && !state.cgroup_name_is_backdrop(to)
                {
                    anyhow::bail!(
                        "Op::MoveAllTasks from Backdrop-owned '{}' to step-local '{}' \
                         would leave persistent workers in a cgroup that disappears \
                         at step boundary; declare `{}` in the Backdrop too, or \
                         move the workers back into a Backdrop-owned cgroup",
                        from,
                        to,
                        to,
                    );
                }
                // Clear subtree_control on the destination before moving
                // tasks. The kernel's no-internal-process constraint
                // (cgroup_migrate_vet_dst) returns EBUSY when writing to
                // cgroup.procs of a cgroup with subtree_control set.
                if let Err(e) = ctx.cgroups.clear_subtree_control(to) {
                    tracing::warn!(
                        cgroup = to.as_ref(),
                        err = %e,
                        "failed to clear subtree_control before task move"
                    );
                }
                // Collect every matching handle's pid list first so
                // partial-failure semantics are bounded: if any per-pid
                // cgroup.procs write fails, we have not yet mutated
                // `state`, so handles remain keyed under `from`. The
                // kernel side may still be partially migrated (writes
                // before the failing pid succeeded), but the in-process
                // tracking does not also drift — subsequent ops looking
                // up by `from` find the same set they would have found
                // before this op ran.
                let pid_batches: Vec<Vec<libc::pid_t>> = state
                    .all_handles()
                    .filter(|(name, _)| name.as_str() == *from)
                    .map(|(_, handle)| handle.worker_pids())
                    .collect();
                for pids in &pid_batches {
                    ctx.cgroups.move_tasks(to, pids)?;
                }
                // Re-key handles under `to` and transfer ownership
                // when required. A step-local handle whose `to`
                // names a Backdrop cgroup moves into the backdrop
                // slot so its lifetime extends with the destination
                // cgroup — without the transfer, the step's
                // teardown would SIGKILL the worker even though the
                // user moved it into a persistent cgroup. Backdrop
                // handles always stay in the backdrop slot
                // regardless of `to`; "Backdrop is persistent" does
                // not degrade to step-local ownership because a
                // later MoveAllTasks targets a step-local cgroup.
                // Only run after every kernel write succeeded —
                // partial failure leaves `state` un-renamed.
                state.rename_handles(from, to);
            }
            Op::RunPayload {
                payload,
                args,
                cgroup,
            } => {
                if payload.is_scheduler() {
                    anyhow::bail!(
                        "Op::RunPayload called with scheduler-kind Payload ('{}'); \
                         only PayloadKind::Binary payloads can be spawned by step ops",
                        payload.name,
                    );
                }
                // Known-flags allowlist: if the Payload declared
                // one, surface typos as scenario-execution-time
                // errors instead of silent no-ops at payload
                // runtime.
                validate_known_flags(payload, args)?;
                // Compute the cgroup key now so the composite-key
                // dedup sees the same `(name, cgroup)` pair the
                // spawn is about to record.
                let cgroup_key = cgroup.as_ref().map(|c| c.to_string()).unwrap_or_default();
                if let Some(existing) =
                    state.find_live_payload_with_cgroup(payload.name, &cgroup_key)
                {
                    // Same payload in the same cgroup is still a
                    // collision: two concurrent runs would write
                    // overlapping metrics to the sidecar and there's
                    // no way for a subsequent WaitPayload / KillPayload
                    // to tell them apart. Same payload in a DIFFERENT
                    // cgroup is now legitimate (placement-disambiguated).
                    // Name the surface that spawned the live handle
                    // so the user can find the original site without
                    // guessing.
                    anyhow::bail!(
                        "Op::RunPayload: payload '{}' already running in cgroup {} (spawned by {}) — \
                         WaitPayload/KillPayload it before spawning another with the same name in the same cgroup",
                        payload.name,
                        render_cgroup_key(&existing.cgroup),
                        existing.source.describe(),
                    );
                }
                let mut run = crate::scenario::payload_run::PayloadRun::new(ctx, payload);
                if !args.is_empty() {
                    run = run.args(args.iter().cloned());
                }
                if let Some(c) = cgroup {
                    run = run.in_cgroup(c.clone());
                }
                let handle = run.spawn().with_context(|| {
                    format!(
                        "Op::RunPayload: spawn payload '{}' in cgroup {}",
                        payload.name,
                        render_cgroup_key(&cgroup_key),
                    )
                })?;
                state.target_payload_handles().push(PayloadEntry {
                    cgroup: cgroup_key,
                    source: PayloadSource::OpRunPayload,
                    handle,
                });
            }
            Op::WaitPayload { name, cgroup } => {
                let entry = take_payload_for_op(
                    state,
                    "Op::WaitPayload",
                    "waiting",
                    "Op::wait_payload_in_cgroup",
                    name,
                    cgroup.as_deref(),
                )?;
                // Check verdicts + metrics are recorded to the sidecar
                // via the SHM ring inside `handle.wait()`; the returned
                // tuple is discarded here because step-ops surface per-
                // payload results through the sidecar, not the ops API.
                let _result = entry
                    .handle
                    .wait()
                    .with_context(|| format!("Op::WaitPayload: wait payload '{name}'"))?;
            }
            Op::KillPayload { name, cgroup } => {
                let entry = take_payload_for_op(
                    state,
                    "Op::KillPayload",
                    "killing",
                    "Op::kill_payload_in_cgroup",
                    name,
                    cgroup.as_deref(),
                )?;
                let _result = entry
                    .handle
                    .kill()
                    .with_context(|| format!("Op::KillPayload: kill payload '{name}'"))?;
            }
            Op::FreezeCgroup { cgroup } => {
                ctx.cgroups
                    .set_freeze(cgroup, true)
                    .with_context(|| format!("Op::FreezeCgroup: cgroup '{cgroup}'"))?;
            }
            Op::UnfreezeCgroup { cgroup } => {
                ctx.cgroups
                    .set_freeze(cgroup, false)
                    .with_context(|| format!("Op::UnfreezeCgroup: cgroup '{cgroup}'"))?;
            }
            Op::CaptureSnapshot { name } => {
                // Two execution contexts:
                //   1. Test fixture: a thread-local SnapshotBridge is
                //      installed (e.g. by the `snapshot_e2e.rs`
                //      smoke tests). Drive its capture callback
                //      directly — no SHM, no doorbell — so the
                //      pure-host unit tests still exercise the
                //      executor + bridge wiring.
                //   2. Production: the scenario runs inside the
                //      guest VM. The freeze coordinator owns the
                //      bridge on the host. Publish a request
                //      through SHM, fire the doorbell, and wait
                //      for the host to stamp a matching reply id.
                //      The host's coordinator stores the captured
                //      report on its bridge; the test code drains
                //      the bridge after VM exit.
                // Stamp the capture with the current scenario phase
                // (1-indexed: 0 = BASELINE, 1..=N = Step ordinals)
                // so the drained sample buckets directly into the
                // matching PhaseBucket without a later reindex.
                // Reads the per-VM `Ctx::current_step` Arc the Step
                // loop publishes via `Release` just before
                // `run_step`; `Acquire` here pairs with that store
                // for a happens-after on Step state setup.
                let phase = ctx.current_step.load(std::sync::atomic::Ordering::Acquire);
                let invoked = crate::scenario::snapshot::with_active_bridge(|b| {
                    let captured = b.capture_with_step(name, phase);
                    if captured {
                        tracing::info!(
                            name = %name,
                            stored = b.len(),
                            step_index = phase,
                            "Op::CaptureSnapshot: captured diagnostic snapshot"
                        );
                    }
                    captured
                });
                if invoked.is_none() {
                    if crate::vmm::guest_comms::is_guest() {
                        if SNAPSHOT_TRANSPORT_DEAD.load(Ordering::Relaxed) {
                            // A prior request observed a transport
                            // failure. Skip the 30 s host-reply wait
                            // — the latch only flips on
                            // TransportError, so the host-side
                            // coordinator is unreachable until the
                            // process restarts.
                            tracing::warn!(
                                name = %name,
                                "Op::CaptureSnapshot: snapshot transport latched dead; skipping host \
                                 request to avoid the 30 s timeout per attempt"
                            );
                        } else {
                            let timeout = std::time::Duration::from_secs(30);
                            match crate::vmm::guest_comms::request_snapshot(
                                crate::vmm::wire::SNAPSHOT_KIND_CAPTURE,
                                name,
                                timeout,
                            ) {
                                crate::vmm::wire::SnapshotRequestResult::Ok => {
                                    tracing::info!(
                                        name = %name,
                                        "Op::CaptureSnapshot: host captured diagnostic snapshot via TLV stream"
                                    );
                                }
                                crate::vmm::wire::SnapshotRequestResult::HostError { reason } => {
                                    anyhow::bail!(
                                        "Op::CaptureSnapshot('{name}'): host rejected capture: {reason}"
                                    );
                                }
                                crate::vmm::wire::SnapshotRequestResult::TransportError {
                                    reason,
                                } => {
                                    SNAPSHOT_TRANSPORT_DEAD.store(true, Ordering::Relaxed);
                                    anyhow::bail!(
                                        "Op::CaptureSnapshot('{name}'): port-1 transport failure: {reason}"
                                    );
                                }
                            }
                        }
                    } else {
                        tracing::warn!(
                            name = %name,
                            "Op::CaptureSnapshot: no SnapshotBridge installed on the executor's \
                             thread and not running in a guest VM — skipping capture"
                        );
                    }
                }
            }
            Op::WatchSnapshot { symbol } => {
                // Two execution contexts mirroring `Op::CaptureSnapshot`:
                //   1. Test fixture: thread-local SnapshotBridge
                //      drives the register callback directly.
                //   2. Production: in-guest scenario sends a
                //      `SNAPSHOT_KIND_WATCH` request through the
                //      virtio-console port-1 TLV stream. The host
                //      coordinator resolves the
                //      symbol via the parsed vmlinux ELF +
                //      direct-mapping translation, allocates a
                //      free user watchpoint slot, programs the
                //      hardware watchpoint via
                //      `KVM_SET_GUEST_DEBUG` on every vCPU, and
                //      replies OK. A future guest write to the
                //      resolved KVA fires the corresponding debug
                //      exit; the vCPU dispatcher identifies the
                //      slot and latches `WatchpointSlot::hit`. The
                //      coordinator then runs
                //      `freeze_and_capture(false)` and stores the
                //      report on the bridge keyed by the symbol.
                let registered =
                    crate::scenario::snapshot::with_active_bridge(|b| b.register_watch(symbol));
                match registered {
                    Some(Ok(())) => {
                        tracing::info!(
                            symbol = %symbol,
                            "Op::WatchSnapshot: registered hardware-watchpoint snapshot"
                        );
                    }
                    Some(Err(err)) => {
                        anyhow::bail!(
                            "Op::WatchSnapshot: register watch on '{symbol}' failed: {err}",
                        );
                    }
                    None => {
                        if crate::vmm::guest_comms::is_guest() {
                            if SNAPSHOT_TRANSPORT_DEAD.load(Ordering::Relaxed) {
                                tracing::warn!(
                                    symbol = %symbol,
                                    "Op::WatchSnapshot: snapshot transport latched dead; skipping \
                                     host request to avoid the 30 s timeout per attempt"
                                );
                            } else {
                                let timeout = std::time::Duration::from_secs(30);
                                match crate::vmm::guest_comms::request_snapshot(
                                    crate::vmm::wire::SNAPSHOT_KIND_WATCH,
                                    symbol,
                                    timeout,
                                ) {
                                    crate::vmm::wire::SnapshotRequestResult::Ok => {
                                        tracing::info!(
                                            symbol = %symbol,
                                            "Op::WatchSnapshot: host armed hardware-watchpoint via TLV stream"
                                        );
                                    }
                                    crate::vmm::wire::SnapshotRequestResult::HostError {
                                        reason,
                                    } => {
                                        anyhow::bail!(
                                            "Op::WatchSnapshot('{symbol}'): host rejected: {reason}"
                                        );
                                    }
                                    crate::vmm::wire::SnapshotRequestResult::TransportError {
                                        reason,
                                    } => {
                                        SNAPSHOT_TRANSPORT_DEAD.store(true, Ordering::Relaxed);
                                        anyhow::bail!(
                                            "Op::WatchSnapshot('{symbol}'): port-1 transport failure: {reason}"
                                        );
                                    }
                                }
                            }
                        } else {
                            tracing::warn!(
                                symbol = %symbol,
                                "Op::WatchSnapshot: no SnapshotBridge installed and not in \
                                 guest VM — skipping watch registration"
                            );
                        }
                    }
                }
            }
            Op::WriteKernelHot { writes } => {
                let payload = build_kernel_op_request(
                    crate::vmm::wire::KernelOpMode::Hot,
                    crate::vmm::wire::KernelOpDirection::Write,
                    String::new(),
                    write_entries_from_writes(writes),
                );
                dispatch_kernel_op_request("Op::WriteKernelHot", payload)?;
            }
            Op::WriteKernelCold { writes } => {
                let payload = build_kernel_op_request(
                    crate::vmm::wire::KernelOpMode::Cold,
                    crate::vmm::wire::KernelOpDirection::Write,
                    String::new(),
                    write_entries_from_writes(writes),
                );
                dispatch_kernel_op_request("Op::WriteKernelCold", payload)?;
            }
            Op::ReadKernelHot { tag, target, width } => {
                let payload = build_kernel_op_request(
                    crate::vmm::wire::KernelOpMode::Hot,
                    crate::vmm::wire::KernelOpDirection::Read,
                    tag.to_string(),
                    vec![crate::vmm::wire::KernelOpEntry {
                        target: target.into(),
                        value: width.into(),
                    }],
                );
                dispatch_kernel_op_request("Op::ReadKernelHot", payload)?;
            }
            Op::ReadKernelCold { tag, target, width } => {
                let payload = build_kernel_op_request(
                    crate::vmm::wire::KernelOpMode::Cold,
                    crate::vmm::wire::KernelOpDirection::Read,
                    tag.to_string(),
                    vec![crate::vmm::wire::KernelOpEntry {
                        target: target.into(),
                        value: width.into(),
                    }],
                );
                dispatch_kernel_op_request("Op::ReadKernelCold", payload)?;
            }
            // Scheduler-lifecycle Op dispatch. apply_ops runs
            // guest-side (the test scenario executes inside the VM
            // as part of the guest binary), so each arm calls into
            // the `vmm::rust_init` spawn/kill primitives directly —
            // no host-to-guest wire format is needed. The Op variant
            // payload carries the target `&'static Scheduler`; the
            // composer derives staging archive paths via the
            // `test_support::staged` helpers so the spawn path
            // matches the cpio entries packed by the initramfs
            // composer.
            //
            // SCHED_PID is the single source of truth for "which
            // scheduler is currently running". Each arm reads it
            // (Detach/Replace/Restart) or writes it (Attach via the
            // spawn helper's internal store) to keep the existing
            // monitor / sched-stats / probe consumers consistent.
            Op::AttachScheduler { scheduler } => {
                dispatch_attach_scheduler(scheduler)?;
            }
            Op::DetachScheduler => {
                dispatch_detach_scheduler()?;
            }
            Op::RestartScheduler => {
                dispatch_restart_scheduler()?;
            }
            Op::ReplaceScheduler { scheduler } => {
                dispatch_replace_scheduler(scheduler)?;
            }
        }
    }
    Ok(())
}

/// Fold runs of adjacent [`Op::WriteKernelCold`] singleton ops
/// into one merged `Op::WriteKernelCold` with the concatenated
/// `writes` vec. Caller-supplied multi-write `Op::WriteKernelCold`
/// ops also fold (their writes vec appends onto the running batch).
///
/// Merge eligibility is strictly `Op::WriteKernelCold` adjacent to
/// `Op::WriteKernelCold` — any other op (including
/// [`Op::ReadKernelCold`], [`Op::WriteKernelHot`],
/// [`Op::CaptureSnapshot`], or any unrelated op) is a hard
/// barrier and starts a new batch.
///
/// Reads do NOT merge in this pre-pass — each
/// [`Op::ReadKernelCold`] still triggers its own freeze
/// rendezvous. Folding reads requires per-entry tags on the wire
/// (so each entry's reply lands under its caller's tag), which
/// lands in a follow-up batch.
///
/// Pre-pass cost: one allocation per `apply_ops` call (the
/// returned `Vec<Op>`). When the input contains no adjacent cold
/// writes the output is structurally equivalent to the input.
fn merge_adjacent_cold_writes(ops: &[Op]) -> Vec<Op> {
    let mut out: Vec<Op> = Vec::with_capacity(ops.len());
    let mut pending_writes: Option<Vec<(KernelTarget, KernelValue)>> = None;
    for op in ops {
        match op {
            Op::WriteKernelCold { writes } => {
                // Fold into the running batch; this collapses N
                // adjacent singletons into one merged op.
                match &mut pending_writes {
                    Some(buf) => buf.extend(writes.iter().cloned()),
                    None => pending_writes = Some(writes.clone()),
                }
            }
            _ => {
                // Barrier — flush the in-flight cold-write batch
                // before emitting the non-mergeable op.
                if let Some(buf) = pending_writes.take() {
                    out.push(Op::WriteKernelCold { writes: buf });
                }
                out.push(op.clone());
            }
        }
    }
    if let Some(buf) = pending_writes.take() {
        out.push(Op::WriteKernelCold { writes: buf });
    }
    out
}

/// Build a [`crate::vmm::wire::KernelOpRequestPayload`] from the
/// per-arm bits — mode, direction, tag, entries. The `request_id` is
/// stamped 0 here and overwritten by the wire transport
/// ([`crate::vmm::guest_comms::request_kernel_op`]) before publishing;
/// the bridge path ignores it (the in-process callback round-trips
/// whatever id the caller supplied).
fn build_kernel_op_request(
    mode: crate::vmm::wire::KernelOpMode,
    direction: crate::vmm::wire::KernelOpDirection,
    tag: String,
    entries: Vec<crate::vmm::wire::KernelOpEntry>,
) -> crate::vmm::wire::KernelOpRequestPayload {
    crate::vmm::wire::KernelOpRequestPayload {
        request_id: 0,
        mode,
        direction,
        tag,
        entries,
    }
}

/// Convert an Op-side `(KernelTarget, KernelValue)` write batch into
/// the wire-side [`crate::vmm::wire::KernelOpEntry`] list, using the
/// `From<&KernelTarget>` / `From<&KernelValue>` impls in
/// [`super::types::op`] for the 1:1 enum mapping.
fn write_entries_from_writes(
    writes: &[(KernelTarget, KernelValue)],
) -> Vec<crate::vmm::wire::KernelOpEntry> {
    writes
        .iter()
        .map(|(target, value)| crate::vmm::wire::KernelOpEntry {
            target: target.into(),
            value: value.into(),
        })
        .collect()
}

/// Dispatch a built [`crate::vmm::wire::KernelOpRequestPayload`] via
/// the bridge-first / wire-fallback / hard-fail pattern:
///
/// 1. **Test fixture path**: if a thread-local
///    [`crate::scenario::snapshot::SnapshotBridge`] is installed
///    with a kernel-op callback, route the request through it. The
///    callback can record the request, synthesise a reply, and
///    return without touching real guest memory — the host-side
///    coordinator / freeze-coord paths are not invoked.
/// 2. **Production path**: if the executor is running inside a
///    guest VM (no in-process bridge callback), forward the
///    request via the port-1 TLV stream through
///    [`crate::vmm::guest_comms::request_kernel_op`]. The host-side
///    handler that consumes the request (freeze-coord cold-path
///    for `Cold` mode, host-worker for `Hot` mode) lands in
///    dedicated follow-up sub-batches; until those handlers exist
///    the wire fallback will surface a `TransportError` after the
///    deadline elapses.
/// 3. **Neither**: a hard `anyhow::bail!` with an actionable hint.
///    Per the project "no silent drops" rule the dispatcher
///    refuses to no-op; a no-bridge-no-guest call is always a
///    misconfigured test fixture. The bail names both recovery
///    paths so the test author can install a callback via
///    `SnapshotBridge::new(...).with_kernel_op(...).set_thread_local()`
///    or run the scenario inside a guest VM.
///
/// On any success-path reply the function checks
/// [`crate::vmm::wire::KernelOpReplyPayload::success`] and converts
/// `false` to an `anyhow::Error` so the caller's `?` propagation
/// surfaces the host-side failure.
///
/// Per-spawn sequence number used by [`staged_scheduler_log_path`]
/// to keep successive Op-dispatched spawns of the SAME staged
/// scheduler from overwriting each other's logs. Monotonic across
/// the entire scenario lifetime; each call to
/// `staged_scheduler_log_path` consumes one seq value.
fn next_sched_spawn_seq() -> u64 {
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// Per-staged-scheduler log path with a monotonic seq suffix so
/// mid-experiment swaps don't overwrite each other's logs. The
/// boot scheduler keeps `/tmp/sched.log` (one spawn, one log);
/// every staged scheduler gets a name-and-seq-keyed log so
/// successive Op::ReplaceScheduler or Op::AttachScheduler
/// dispatches with the SAME staged name don't truncate the first
/// spawn's failure-dump payload.
///
/// Path scheme `/tmp/sched_<name>_<seq>.log` is collision-free
/// under the validated name shape (no path separators, no leading
/// `.`, length-capped to 128 bytes per `validate_staged_scheduler_name`)
/// plus the per-call monotonic seq suffix.
fn staged_scheduler_log_path(name: &str) -> String {
    format!("/tmp/sched_{name}_{seq}.log", seq = next_sched_spawn_seq())
}

/// SIGTERM grace window for scheduler-lifecycle Op kill paths.
/// 10s comfortably exceeds the real-world scx_disable_workfn
/// detach latency (kernel/sched/ext.c:5923) — the kernel tears
/// down the BPF prog graph on refcount drop from a workqueue and
/// the scheduler's SIGTERM handler returns from main once that
/// completes. The 2s initial cut produced
/// `StillAliveAfterSigkill` in the e2e because scx_disable_workfn
/// took longer than the SIGTERM budget AND the SIGKILL post-grace
/// also exceeded (`POST_SIGKILL_GRACE` at 2s) — neither could
/// service a process stuck mid-BPF-detach in D-state. 10s gives
/// the SIGTERM-handled clean exit path enough room to complete
/// without escalating to SIGKILL on the common scx_* scheduler
/// shape; the SIGKILL escalation inside `kill_scheduler_process`
/// still covers any pathological hang past this budget.
const SCHED_LIFECYCLE_KILL_GRACE: std::time::Duration = std::time::Duration::from_secs(10);

/// Path of the sched_ext kernel state sysfs node. Reading returns
/// the string-form state: `disabled`, `enabling`, `enabled`,
/// `disabling` (kernel/sched/ext.c). Op dispatch polls this
/// between kill and spawn so the next scheduler's BPF skeleton
/// load doesn't hit `-EBUSY` from
/// `kernel/sched/ext.c:6643`'s `scx_enable_state() != SCX_DISABLED`
/// guard at enable entry.
const SCX_STATE_SYSFS: &str = "/sys/kernel/sched_ext/state";

/// Block until `/sys/kernel/sched_ext/state` reads `disabled` or
/// the timeout elapses. Polls at 50ms — small enough to keep the
/// Op dispatch latency tight when the kernel finishes the detach
/// quickly, large enough that the busy-wait doesn't measurably
/// pressure the scheduler workqueue running the BPF detach.
///
/// Returns `Ok(elapsed)` when the state reaches `disabled`,
/// `Err` with the last observed state when the timeout fires.
/// Absent sysfs node (kernel without sched_ext or non-Linux
/// platform) returns `Ok(Duration::ZERO)` — the no-scx case has
/// no detach to wait for and the next spawn will fail later with
/// a sharper diagnostic if scx is genuinely required.
fn wait_for_scx_disabled(timeout: std::time::Duration) -> Result<std::time::Duration> {
    let start = std::time::Instant::now();
    let interval = std::time::Duration::from_millis(50);
    let path = std::path::Path::new(SCX_STATE_SYSFS);
    if !path.exists() {
        return Ok(std::time::Duration::ZERO);
    }
    loop {
        let state = std::fs::read_to_string(SCX_STATE_SYSFS).unwrap_or_default();
        let state = state.trim();
        if state == "disabled" {
            return Ok(start.elapsed());
        }
        if start.elapsed() >= timeout {
            anyhow::bail!(
                "wait_for_scx_disabled: state '{state}' did not reach 'disabled' \
                 within {timeout:?}; the kernel scx state machine is stuck — \
                 the next scheduler spawn will hit -EBUSY at the enable path. \
                 Inspect /sys/kernel/sched_ext/state + dmesg for the stuck \
                 disable transition.",
            );
        }
        std::thread::sleep(interval);
    }
}

/// Common kill helper for the Detach / Restart / Replace arms.
/// Reads SCHED_PID, sends SIGTERM, waits for the scx kernel state
/// to transition to `disabled` (the load-bearing barrier per
/// `wait_for_scx_disabled`'s doc), and clears SCHED_PID on
/// success so subsequent reads observe "no scheduler".
///
/// Direct `libc::kill(SIGTERM)` rather than the
/// `vmm::rust_init::kill_scheduler_process` helper because the
/// latter's strict /proc-absence verification can fire
/// `StillAliveAfterSigkill` when the scheduler's exit blocks on
/// BPF detach. The scheduler PROCESS being gone is not the same
/// signal as the BPF state being `SCX_DISABLED` — operatively the
/// sysfs state is what gates the next spawn, not /proc removal.
/// Both signals normally resolve together but a slow workqueue
/// can decouple them under load. The scx state machine reaches
/// `disabled` BEFORE the userspace process completes its libbpf
/// cleanup syscalls (which are what holds /proc/{pid} alive).
fn kill_current_scheduler(op_label: &str) -> Result<libc::pid_t> {
    let pid = crate::vmm::rust_init::sched_pid().ok_or_else(|| {
        anyhow::anyhow!(
            "{op_label}: no scheduler attached (SCHED_PID is 0); \
             attach a scheduler via boot-time `scheduler` field or \
             `Op::AttachScheduler` before invoking this Op"
        )
    })?;
    // Suppress the guest sched_exit_monitor's SchedExit message —
    // the host's dispatch.rs SchedExit arm promotes the message
    // into the run-wide kill flag and would fail the test as
    // scheduler-died even though the kill is an INTENTIONAL
    // lifecycle Op. The dispatch_* arms clear this flag after the
    // post-kill sequence completes. See SCHED_EXIT_SUPPRESS doc at
    // src/vmm/rust_init.rs:90 for the v0 limitation.
    crate::vmm::rust_init::SCHED_EXIT_SUPPRESS.store(true, std::sync::atomic::Ordering::Release);
    // Trigger async scx_disable via sysrq-'S' so the kernel-side
    // disable cascade runs OUT OF BAND from the scheduler's exit
    // path. Without this, `bpf_scx_unreg`
    // (kernel/sched/ext.c:7375-7382) holds the dying process in
    // D-state inside the bpf_link refcount-drop chain via
    // `kthread_flush_work(&sch->disable_work)` — SIGKILL kills
    // userspace but cannot remove /proc/{pid} until that block
    // finishes, which is how the `kill_scheduler_process` helper
    // sees `StillAliveAfterSigkill` under realistic load.
    // The sysrq-'S' handler at ext.c:7508 runs scx_disable directly
    // via RCU-protected scx_root (registered at ext.c:7791
    // `register_sysrq_key('S', &sysrq_sched_ext_reset_op)`), so the
    // disable_work irq_work fires asynchronously and the scheduler
    // process can exit cleanly without holding the bpf_link across
    // the slow disable cascade. Best-effort write — sysrq absence
    // or write failure is silently tolerated because the SIGTERM
    // below still drives the standard detach path (slower but
    // correct).
    let _ = std::fs::write("/proc/sysrq-trigger", "S");

    // SIGTERM lets the scheduler's userspace handler invoke its
    // libbpf cleanup (drops BPF prog refcounts, returns from
    // main). With sysrq-'S' already in flight above, the bpf_link
    // refcount drop's `bpf_scx_unreg` finds disable_work near
    // completion and `kthread_flush_work` returns quickly — no
    // D-state stall. We still wait for the SCX_DISABLED state
    // below rather than for the userspace process to exit so the
    // next scheduler's BPF skeleton load doesn't hit -EBUSY at
    // kernel/sched/ext.c:6643.
    let r = unsafe { libc::kill(pid, libc::SIGTERM) };
    if r != 0 {
        let errno = std::io::Error::last_os_error();
        anyhow::bail!("{op_label}: SIGTERM to pid {pid} failed: {errno}");
    }
    let elapsed = wait_for_scx_disabled(SCHED_LIFECYCLE_KILL_GRACE).map_err(|e| {
        anyhow::anyhow!("{op_label}: wait_for_scx_disabled(pid={pid}) failed: {e:#}")
    })?;
    tracing::debug!(
        op = op_label,
        pid = pid,
        elapsed_ms = elapsed.as_millis() as u64,
        "scx state reached 'disabled' after SIGTERM",
    );
    crate::vmm::rust_init::set_sched_pid(0);
    Ok(pid)
}

/// Spawn helper shared by Attach / Restart / Replace arms.
/// Calls `try_spawn_scheduler` (the Result-returning variant) so
/// the boot-path force_reboot semantics don't apply — a failed
/// spawn / startup-died / not-attached surfaces as a typed
/// `anyhow::Error` that bubbles up through `apply_ops` to fail
/// the test cleanly instead of rebooting the VM. The helper
/// stores SCHED_PID on successful spawn via the internal
/// `SCHED_PID.store` call site in `try_spawn_scheduler`.
fn spawn_scheduler_for_op(
    op_label: &str,
    binary_path: &str,
    args_path: &str,
    log_path: &str,
    expected_scheduler_name: &str,
) -> Result<()> {
    match crate::vmm::rust_init::try_spawn_scheduler(binary_path, args_path, log_path) {
        Ok(Some(_)) => Ok(()),
        Ok(None) => anyhow::bail!(
            "{op_label}: scheduler binary for '{expected_scheduler_name}' is missing at \
             {binary_path}. The staging cpio pack at initramfs build time should have \
             materialised it via staged_scheduler_binary_path — check that \
             KtstrTestEntry.staged_schedulers contains the named entry and the host-side \
             resolve_staged_schedulers_strict found its binary."
        ),
        Err(e) => anyhow::bail!(
            "{op_label}: scheduler '{expected_scheduler_name}' spawn failed: {e}. The boot \
             path would force_reboot on this; the Op dispatch path surfaces it as a typed \
             test-failure so the operator sees the specific failure mode (spawn vs \
             startup-died vs not-attached) instead of a bare reboot signal."
        ),
    }
}

/// Op::AttachScheduler dispatch. Spawns the named staged scheduler
/// at its `/staging/schedulers/<name>/` archive paths. The boot
/// scheduler (if any) is NOT auto-detached — callers must issue a
/// preceding `Op::DetachScheduler` if they intend to swap rather
/// than co-attach. Sidecar swap tagging emits a `tracing::info!`
/// event with structured fields for the phase-aware sidecar
/// pipeline to pick up; the full sidecar schema wire-in lands
/// alongside the rest of the phase pipeline.
fn dispatch_attach_scheduler(scheduler: &'static crate::test_support::Scheduler) -> Result<()> {
    let binary = crate::test_support::staged::staged_scheduler_binary_path(scheduler.name);
    let args = crate::test_support::staged::staged_scheduler_args_path(scheduler.name);
    let log = staged_scheduler_log_path(scheduler.name);
    spawn_scheduler_for_op("Op::AttachScheduler", &binary, &args, &log, scheduler.name)?;
    tracing::info!(
        op = "AttachScheduler",
        scheduler_name = scheduler.name,
        binary_path = %binary,
        log_path = %log,
        "scheduler attached",
    );
    Ok(())
}

/// Op::DetachScheduler dispatch. Kills the currently-running
/// scheduler via the shared kill helper and clears SCHED_PID.
fn dispatch_detach_scheduler() -> Result<()> {
    let pid = kill_current_scheduler("Op::DetachScheduler")?;
    tracing::info!(
        op = "DetachScheduler",
        killed_pid = pid,
        "scheduler detached"
    );
    Ok(())
}

/// Op::RestartScheduler dispatch. Kills the currently-running
/// scheduler and respawns the BOOT scheduler at `/scheduler` +
/// `/sched_args`. v0 limitation: assumes the boot scheduler is the
/// intended restart target — a future iteration tracking
/// "currently-attached scheduler paths" can restart staged
/// schedulers in place. The common test pattern (validate boot
/// scheduler survives detach + reattach cleanly) is covered.
fn dispatch_restart_scheduler() -> Result<()> {
    let prev_pid = kill_current_scheduler("Op::RestartScheduler")?;
    spawn_scheduler_for_op(
        "Op::RestartScheduler",
        "/scheduler",
        "/sched_args",
        "/tmp/sched.log",
        "boot",
    )?;
    tracing::info!(
        op = "RestartScheduler",
        prev_pid = prev_pid,
        "boot scheduler restarted",
    );
    Ok(())
}

/// Op::ReplaceScheduler dispatch. Atomically (from the user-visible
/// scenario's perspective) detaches the currently-running scheduler
/// and attaches the named staged scheduler. Emits a sidecar-tagging
/// event with both prev and new scheduler context so phase-aware
/// analysis can attribute pre-swap vs post-swap metrics.
fn dispatch_replace_scheduler(scheduler: &'static crate::test_support::Scheduler) -> Result<()> {
    let prev_pid = kill_current_scheduler("Op::ReplaceScheduler")?;
    let binary = crate::test_support::staged::staged_scheduler_binary_path(scheduler.name);
    let args = crate::test_support::staged::staged_scheduler_args_path(scheduler.name);
    let log = staged_scheduler_log_path(scheduler.name);
    spawn_scheduler_for_op("Op::ReplaceScheduler", &binary, &args, &log, scheduler.name)?;
    tracing::info!(
        op = "ReplaceScheduler",
        prev_pid = prev_pid,
        new_scheduler_name = scheduler.name,
        binary_path = %binary,
        log_path = %log,
        "scheduler replaced",
    );
    Ok(())
}

/// **Timeout choice.** The 30 s wire-fallback timeout is sized for
/// the cold path's freeze-rendezvous round-trip (matches the
/// `FREEZE_RENDEZVOUS_TIMEOUT` budget in CLAUDE.md). The hot path
/// completes sub-microsecond and treats the timeout strictly as an
/// upper bound; a regression that stalls the host-worker would
/// surface as a deferred 30 s wait, not a missed bug.
fn dispatch_kernel_op_request(
    op_label: &str,
    payload: crate::vmm::wire::KernelOpRequestPayload,
) -> Result<()> {
    // `with_active_bridge` returns `Option<Option<reply>>` — outer
    // `None` means no bridge active on the thread; inner `None`
    // means bridge active but no kernel-op callback installed.
    // Both collapse to "no bridge-routed reply" via `.flatten()`.
    let bridge_reply =
        crate::scenario::snapshot::with_active_bridge(|b| b.dispatch_kernel_op(&payload)).flatten();
    if let Some(reply) = bridge_reply {
        return check_kernel_op_reply(op_label, &payload, &reply);
    }
    if !crate::vmm::guest_comms::is_guest() {
        // No bridge callback AND not in a guest VM — refuse to
        // no-op. The actionable hint names both recovery paths so
        // the test author can pick the one matching their context
        // (per the project "no silent drops" rule).
        anyhow::bail!(
            "{op_label}('{}'): no SnapshotBridge kernel-op callback is installed on this \
             thread and not running in a guest VM. Install a callback via \
             SnapshotBridge::new(...).with_kernel_op(...).set_thread_local() for host-side \
             tests, or run the scenario inside a ktstr guest VM where the port-1 wire path \
             provides dispatch.",
            payload.tag,
        );
    }
    let timeout = std::time::Duration::from_secs(30);
    match crate::vmm::guest_comms::request_kernel_op(payload.clone(), timeout) {
        crate::vmm::wire::KernelOpRequestResult::Ok(reply) => {
            check_kernel_op_reply(op_label, &payload, &reply)
        }
        crate::vmm::wire::KernelOpRequestResult::TransportError { reason } => {
            anyhow::bail!(
                "{op_label}('{}'): port-1 transport failure: {reason}",
                payload.tag,
            );
        }
    }
}

/// Inspect a [`crate::vmm::wire::KernelOpReplyPayload`]. Logs success
/// at info level (with entry count + tag for diagnostics), converts
/// `success = false` into an `anyhow::Error` so the executor's `?`
/// propagation bails the step.
fn check_kernel_op_reply(
    op_label: &str,
    request: &crate::vmm::wire::KernelOpRequestPayload,
    reply: &crate::vmm::wire::KernelOpReplyPayload,
) -> Result<()> {
    if !reply.success {
        anyhow::bail!(
            "{op_label}('{}'): host reported failure: {}",
            request.tag,
            reply.reason,
        );
    }
    tracing::info!(
        op = op_label,
        tag = %request.tag,
        mode = ?request.mode,
        direction = ?request.direction,
        entries = request.entries.len(),
        read_values = reply.read_values.len(),
        "{op_label}: host completed kernel-op batch",
    );
    Ok(())
}

/// Shared lookup for `Op::WaitPayload` / `Op::KillPayload`.
///
/// Consumes the payload handle matching the composite key
/// (`name`, `cgroup`). Produces the op-specific not-found /
/// ambiguous errors so the match arms stay short.
///
/// Callers pass the static trio that shapes the error text:
///
/// - `op_tag` — the user-facing op name (e.g. `"Op::WaitPayload"`).
/// - `verb_ing` — the `-ing` form of the action for "before
///   waiting" / "before killing" prose (no trailing
///   `to_lowercase` munging so two-word op names don't collide
///   into one word).
/// - `ctor_path` — the fully-qualified constructor the user
///   should switch to on ambiguity, e.g.
///   `"Op::wait_payload_in_cgroup"`. Copying this hint into
///   source must produce a callable path.
fn take_payload_for_op(
    state: &mut ScenarioState<'_, '_>,
    op_tag: &str,
    verb_ing: &str,
    ctor_path: &str,
    name: &str,
    cgroup: Option<&str>,
) -> Result<PayloadEntry> {
    match state.take_payload_by_name(name, cgroup) {
        Ok(Some(entry)) => Ok(entry),
        Ok(None) => match cgroup {
            Some(c) => anyhow::bail!(
                "{op_tag}: no running payload named '{name}' in cgroup {} \
                 (spawn it via Op::RunPayload or CgroupDef::workload before {verb_ing})",
                render_cgroup_key(c),
            ),
            None => anyhow::bail!(
                "{op_tag}: no running payload named '{name}' \
                 (spawn it via Op::RunPayload or CgroupDef::workload before {verb_ing})",
            ),
        },
        Err(cgroups) => {
            // Name-only lookup matched >1 live payload. Enumerate
            // the candidate cgroups so the caller knows which
            // qualified form they need.
            let rendered: Vec<String> = cgroups.iter().map(|c| render_cgroup_key(c)).collect();
            anyhow::bail!(
                "{op_tag}: payload '{name}' is ambiguous — {} live copies in cgroups {} — \
                 use {ctor_path}(name, cgroup) to disambiguate",
                rendered.len(),
                rendered.join(", "),
            )
        }
    }
}

/// Read the effective cpuset for a cgroup by reading cpuset.cpus.
fn read_cpuset(ctx: &Ctx, name: &str) -> Option<BTreeSet<usize>> {
    let path = ctx.cgroups.parent_path().join(name).join("cpuset.cpus");
    let content = std::fs::read_to_string(&path).ok()?;
    let content = content.trim();
    if content.is_empty() {
        return None;
    }
    let cpus: BTreeSet<usize> = crate::topology::parse_cpu_list_lenient(content)
        .into_iter()
        .collect();
    Some(cpus)
}

/// Collect step-local worker results and produce an AssertResult.
///
/// Drains step-local handles + payload handles; backdrop state is
/// untouched. Called at every step boundary (success AND error
/// paths) as the "Step is fully bounded" teardown. The
/// `step_state` goes out of scope at the end of this step's
/// iteration, so its `CgroupGroup` drop removes every step-local
/// cgroup immediately after `run_scenario` propagates the result
/// of this call.
///
/// Before draining handles, every step-local cgroup is unfrozen
/// (`cgroup.freeze` ← 0). An [`Op::FreezeCgroup`] without a paired
/// [`Op::UnfreezeCgroup`] would leave step-local tasks frozen at
/// step boundary; killpg/SIGKILL on a frozen task is queued but
/// never delivered (the task is parked off the runqueue), so
/// [`drain_all_payload_handles`] hangs and the subsequent
/// `CgroupGroup::Drop` rmdir hits EBUSY because workers are still
/// resident. Pre-emptive unfreeze restores the run-state
/// precondition every cleanup path expects. Failures are logged
/// at warn level only — a missing freezer file or a cgroup that
/// was already torn down is benign at teardown time, and
/// propagating would mask the real workload result.
fn collect_step(
    step_state: &mut StepState<'_>,
    checks: &crate::assert::Assert,
    topo: &crate::topology::TestTopology,
    cgroups: &dyn crate::cgroup::CgroupOps,
) -> AssertResult {
    // Unfreeze every step-local cgroup before draining handles or
    // letting the CgroupGroup RAII guard rmdir them. A live
    // `cgroup.freeze == 1` blocks SIGKILL delivery (frozen tasks
    // are off the runqueue) and EBUSYs the rmdir.
    for name in step_state.cgroups.names() {
        if let Err(e) = cgroups.set_freeze(name, false) {
            tracing::warn!(
                cgroup = %name,
                err = %format!("{e:#}"),
                "collect_step: pre-teardown unfreeze failed; rmdir may EBUSY"
            );
        }
    }
    // Kill any CgroupDef::workload / Op::RunPayload payload binaries
    // still live at step teardown so cgroupfs cleanup does not trip
    // EBUSY. Metrics are emitted to the SHM ring by PayloadHandle::kill
    // via the `evaluate()` pipeline.
    drain_all_payload_handles(&mut step_state.payload_handles);
    let handles = std::mem::take(&mut step_state.handles);
    crate::scenario::collect_handles(
        handles
            .into_iter()
            .map(|(name, h)| (h, step_state.cpusets.get(&name))),
        checks,
        Some(topo),
    )
}

/// Collect backdrop (persistent) worker results. Called once at
/// scenario end after every Step has torn down. The
/// `backdrop_state.cgroups` RAII guard drops persistent cgroups
/// when `backdrop_state` itself drops.
///
/// Mirrors [`collect_step`]'s pre-teardown unfreeze pass over every
/// tracked cgroup. A backdrop cgroup left frozen at scenario end
/// blocks SIGKILL delivery to its tasks (frozen tasks are off the
/// runqueue, see `kernel/cgroup/freezer.c::cgroup_freeze_task`),
/// which then EBUSYs the rmdir issued by the
/// `BackdropState::cgroups` RAII drop. The asymmetry between
/// step-local and backdrop teardown — only the former unfreezing —
/// would surface as backdrop cgroups leaking on every scenario
/// whose Backdrop froze a cgroup and never unfroze it. Symmetric
/// unfreeze pre-rmdir is the same bug class
/// [`super::CgroupGroup::drop`] already prevents at the
/// CgroupGroup level for the per-step path; this prologue brings
/// the backdrop path back in line.
fn collect_backdrop(
    backdrop_state: &mut BackdropState<'_>,
    checks: &crate::assert::Assert,
    topo: &crate::topology::TestTopology,
    cgroups: &dyn crate::cgroup::CgroupOps,
) -> AssertResult {
    // Unfreeze every backdrop cgroup before draining handles or
    // letting the CgroupGroup RAII guard rmdir them. Same rationale
    // as `collect_step`: a live `cgroup.freeze == 1` blocks SIGKILL
    // delivery (frozen tasks are off the runqueue) and EBUSYs the
    // rmdir.
    for name in backdrop_state.cgroups.names() {
        if let Err(e) = cgroups.set_freeze(name, false) {
            tracing::warn!(
                cgroup = %name,
                err = %format!("{e:#}"),
                "collect_backdrop: pre-teardown unfreeze failed; rmdir may EBUSY"
            );
        }
    }
    drain_all_payload_handles(&mut backdrop_state.payload_handles);
    let handles = std::mem::take(&mut backdrop_state.handles);
    crate::scenario::collect_handles(
        handles
            .into_iter()
            .map(|(name, h)| (h, backdrop_state.cpusets.get(&name))),
        checks,
        Some(topo),
    )
}

/// Kill every payload handle whose cgroup matches `cgroup` and drop
/// the matched entries from `handles`. Runs before the cgroup is
/// removed or stopped; failures are logged to stderr but do not
/// propagate — the cgroup removal is best-effort already, and the
/// payload-kill failure is never the primary error.
///
/// **Metric emission depends on the explicit `.kill()` call** —
/// if a future refactor replaces the `.kill()` below with plain
/// `drop(handle)`, the `PayloadHandle::drop` SIGKILLs the child
/// but skips the evaluate-and-emit pipeline that records metrics
/// to the SHM ring. Test helpers that drain payload handles
/// likewise route through `drain_all_payload_handles` for the
/// same reason. Preserve `.kill()` on every path that claims to
/// drain handles for metric capture.
///
/// Drop order across matched entries is LIFO (last pushed, first
/// dropped) — the loop walks indices from the tail toward index 0
/// using `Vec::remove` so newer matched entries' embedded
/// `SigchldScope`s restore the SIGCHLD disposition before older
/// matches do, matching the save-and-restore chain documented on
/// `PayloadHandle` in `payload_run.rs`. `Vec::swap_remove` would
/// rotate the tail into the freed slot and break LIFO across
/// matches; `Vec::remove` preserves the relative order of the
/// remaining (unmatched) survivors. Note: SIGCHLD scope LIFO across
/// the FULL vec is structurally unsalvageable in any partial-drain
/// helper — unmatched entries that stay alive in `handles` outlive
/// their younger matched siblings whose scopes already restored.
/// The full-vec LIFO contract holds only when every handle is
/// dropped together via [`drain_all_payload_handles`].
fn drain_payload_handles_for_cgroup(handles: &mut Vec<PayloadEntry>, cgroup: &str) {
    let mut i = handles.len();
    while i > 0 {
        i -= 1;
        if handles[i].cgroup.as_str() == cgroup {
            let entry = handles.remove(i);
            if let Err(e) = entry.handle.kill() {
                eprintln!("ktstr: kill payload in cgroup '{cgroup}': {e:#}");
            }
        }
    }
}

/// Kill every payload handle regardless of cgroup and clear the
/// vector. Called at step-sequence teardown so every handle gets a
/// terminal `.kill()` (and therefore a sidecar metric emission) even
/// when no explicit `RemoveCgroup`/`StopCgroup` op targeted it.
///
/// Drop order is LIFO (last pushed, first dropped) — `Vec::pop`
/// returns the tail first, so `PayloadHandle::drop` runs in reverse
/// creation order. Each handle's embedded `SigchldScope` captured the
/// `SIGCHLD` disposition that was live at construction time (the
/// previous scope's installed `SIG_DFL`). Restoring in LIFO unwinds
/// the save-and-restore chain back to the original disposition; FIFO
/// drop (e.g. `Vec::drain(..)`) restores intermediate `SIG_DFL` values
/// out of order and leaks `SIG_DFL` past the outermost scope. See the
/// DROP-ORDER-CRITICAL note on `PayloadHandle` in `payload_run.rs`.
fn drain_all_payload_handles(handles: &mut Vec<PayloadEntry>) {
    while let Some(entry) = handles.pop() {
        if let Err(e) = entry.handle.kill() {
            eprintln!(
                "ktstr: teardown kill payload in cgroup {}: {e:#}",
                render_cgroup_key(&entry.cgroup),
            );
        }
    }
}

/// Render a cgroup key for inclusion in user-facing error text.
/// An empty string is replaced with `(no cgroup)` so
/// `Op::RunPayload { cgroup: None }` failures don't produce messages
/// like `cgroup ''` that look like a corrupt log line. Non-empty
/// keys are quoted so they read clearly next to surrounding prose.
fn render_cgroup_key(cgroup: &str) -> String {
    if cgroup.is_empty() {
        "(no cgroup)".to_string()
    } else {
        format!("'{cgroup}'")
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::ops::RangeInclusive;

    use super::*;
    use crate::workload::{AffinityIntent, WorkSpec, WorkType};
    use strum::IntoEnumIterator;

    /// Exhaustiveness guard for [`OpKind::bit_index`]. A new [`Op`]
    /// variant auto-generates a matching [`OpKind`] variant (via
    /// `#[derive(strum::EnumDiscriminants)]`), which the match arms
    /// in `bit_index` must cover — adding an `Op` variant without
    /// extending `bit_index` fails compilation. But the arms could
    /// still drift in a way the compiler cannot see: two variants
    /// accidentally mapped to the same index, or the contiguous-
    /// from-zero invariant broken by a typo.
    ///
    /// This test iterates every `OpKind` via `EnumIter` and pins
    /// both invariants:
    /// - Every variant produces a distinct bit index.
    /// - Indices are contiguous `0..N` where N = variant count.
    ///
    /// A regression — duplicate index, gap, or an off-by-one —
    /// surfaces here before it silently corrupts the `op_kinds`
    /// bitmask semantics elsewhere in the crate.
    #[test]
    fn op_kind_bit_indices_are_unique_and_contiguous() {
        let kinds: Vec<OpKind> = OpKind::iter().collect();
        let indices: Vec<u32> = kinds.iter().copied().map(OpKind::bit_index).collect();

        // Unique: every kind has a distinct index.
        let unique: std::collections::BTreeSet<u32> = indices.iter().copied().collect();
        assert_eq!(
            unique.len(),
            indices.len(),
            "OpKind::bit_index produced duplicates. \
             Pairs (OpKind, bit_index): {:?}. Fix the match in \
             OpKind::bit_index so every variant maps to a distinct \
             bit.",
            kinds.iter().zip(&indices).collect::<Vec<_>>(),
        );

        // Contiguous: indices form `0..N`.
        let expected: Vec<u32> = (0..kinds.len() as u32).collect();
        let mut sorted = indices.clone();
        sorted.sort_unstable();
        assert_eq!(
            sorted,
            expected,
            "OpKind::bit_index indices must be contiguous from 0 \
             (no gaps, no duplicates). Got sorted indices {sorted:?} \
             for {} OpKind variants; expected {expected:?}.",
            kinds.len(),
        );
    }

    /// `OpKind::iter()` order matches `bit_index` ascending order.
    /// strum's `EnumIter` derive follows declaration order by default
    /// — this test pins that contract so a future strum upgrade or
    /// an enum reorder that decouples the two orderings surfaces
    /// here instead of silently reshuffling bitmask traversal.
    ///
    /// Complements `op_kind_bit_indices_are_unique_and_contiguous`
    /// (which proves bit_index forms 0..N but not that iter() yields
    /// them ascending) and the discriminant tests (which don't
    /// exercise iter order at all).
    #[test]
    fn op_kind_iter_order_matches_bit_index_ascending() {
        let kinds: Vec<OpKind> = OpKind::iter().collect();
        let pairs: Vec<(usize, u32)> = kinds
            .iter()
            .enumerate()
            .map(|(i, k)| (i, k.bit_index()))
            .collect();
        for (i, bit) in &pairs {
            assert_eq!(
                *bit as usize, *i,
                "OpKind::iter()[{i}] (variant {:?}) has bit_index {bit}; \
                 expected iter-index to match bit_index. Pairs: {pairs:?}",
                kinds[*i],
            );
        }
    }

    // -- Traverse combinator (test-only) --

    /// Layout strategy for Traverse phases.
    #[derive(Debug)]
    enum Layout {
        Disjoint,
        /// Overlapping cpusets. (min_frac, max_frac) — PRNG picks a value in range.
        Overlap(f64, f64),
    }

    /// Generates a random walk of cgroup topology changes across phases.
    ///
    /// Each phase picks a random (cgroup_count, layout) pair, generates SetCpuset
    /// ops, spawns workers in new cgroups, and holds for phase_duration.
    ///
    /// `persistent_cgroups` cgroups are created in phase 0 and never removed.
    /// Only cgroups at index >= `persistent_cgroups` are added/removed by the
    /// random walk. The `cgroup_count` range applies to the total cgroup count
    /// (persistent + ephemeral).
    ///
    /// `cgroup_workloads` controls the workload for each cgroup index. If the
    /// vec has fewer entries than the cgroup index, the last entry repeats.
    #[derive(Debug)]
    struct Traverse {
        seed: Option<u64>,
        cgroup_count: RangeInclusive<usize>,
        layouts: Vec<Layout>,
        phases: usize,
        phase_duration: Duration,
        settle: Duration,
        /// Cgroups [0..persistent_cgroups) are created once and never removed.
        persistent_cgroups: usize,
        /// WorkSpec definition per cgroup index. Last entry repeats for higher indices.
        cgroup_workloads: Vec<WorkSpec>,
    }

    impl Traverse {
        /// Generate a `Vec<Step>` from the Traverse configuration.
        fn generate(&self, ctx: &Ctx) -> Vec<Step> {
            use rand::RngExt;

            let seed = self.seed.unwrap_or_else(|| std::process::id() as u64);
            let mut rng = seeded_rng(seed);

            let usable_len = ctx.topo.usable_cpus().len();
            let max_cgroups = (*self.cgroup_count.end()).min(usable_len / 2).max(1);
            let min_cgroups = (*self.cgroup_count.start()).max(1).min(max_cgroups);

            let mut steps = Vec::with_capacity(self.phases + 1);
            let mut live_cgroups: Vec<Cow<'static, str>> = Vec::new();

            let names: Vec<Cow<'static, str>> = (0..max_cgroups)
                .map(|i| Cow::Owned(format!("cg_{i}")))
                .collect();

            for phase in 0..self.phases {
                let range = max_cgroups - min_cgroups + 1;
                let target_count = min_cgroups + rng.random_range(0..range);
                let layout_idx = rng.random_range(0..self.layouts.len());
                let layout = &self.layouts[layout_idx];

                let mut ops = Vec::new();

                // Add cgroups if needed.
                while live_cgroups.len() < target_count {
                    let idx = live_cgroups.len();
                    let name = names[idx].clone();
                    let w = self
                        .cgroup_workloads
                        .get(idx)
                        .or(self.cgroup_workloads.last())
                        .cloned()
                        .unwrap_or_default();
                    ops.push(Op::AddCgroup { name: name.clone() });
                    ops.push(Op::SpawnWorkers {
                        cgroup: name.clone(),
                        work: w,
                    });
                    live_cgroups.push(name);
                }

                // Remove cgroups if needed (never remove persistent cgroups).
                while live_cgroups.len() > target_count
                    && live_cgroups.len() > self.persistent_cgroups
                {
                    if let Some(name) = live_cgroups.pop() {
                        ops.push(Op::StopCgroup {
                            cgroup: name.clone(),
                        });
                        ops.push(Op::RemoveCgroup { cgroup: name });
                    }
                }

                // Apply cpuset layout.
                for (i, name) in live_cgroups.iter().enumerate() {
                    let spec = match layout {
                        Layout::Disjoint => CpusetSpec::Disjoint {
                            index: i,
                            of: live_cgroups.len(),
                        },
                        Layout::Overlap(min_frac, max_frac) => {
                            let frac = min_frac
                                + rng.random_range(0..100) as f64 / 100.0 * (max_frac - min_frac);
                            CpusetSpec::Overlap {
                                index: i,
                                of: live_cgroups.len(),
                                frac,
                            }
                        }
                    };
                    ops.push(Op::SetCpuset {
                        cgroup: name.clone(),
                        cpus: spec,
                    });
                }

                let hold = if phase == 0 {
                    // First phase includes settle time.
                    HoldSpec::fixed(self.settle + self.phase_duration)
                } else {
                    HoldSpec::fixed(self.phase_duration)
                };

                steps.push(Step {
                    setup: vec![].into(),
                    ops,
                    hold,
                });
            }

            steps
        }
    }

    /// Seeded PRNG for deterministic topology generation.
    fn seeded_rng(seed: u64) -> rand::rngs::StdRng {
        use rand::SeedableRng;
        rand::rngs::StdRng::seed_from_u64(seed)
    }

    // -- validate_known_flags tests --

    /// Declared allowlist, every `--flag` in args is on the
    /// allowlist → `Ok(())`. Covers both `--foo` and
    /// `--foo=value` shapes to pin the flag-body split.
    #[test]
    fn validate_known_flags_accepts_listed_long_flags() {
        use crate::test_support::{OutputFormat, Payload, PayloadKind};
        static WITH_ALLOWLIST: Payload = Payload {
            name: "with_allowlist",
            kind: PayloadKind::Binary("/bin/true"),
            output: OutputFormat::ExitCode,
            default_args: &[],
            default_checks: &[],
            metrics: &[],
            include_files: &[],
            uses_parent_pgrp: false,
            known_flags: Some(&["runtime", "threads", "verbose"]),
            metric_bounds: None,
        };
        let args: Vec<String> = vec![
            "--runtime=30".into(),
            "--threads".into(),
            "4".into(),
            "--verbose".into(),
            "positional_arg".into(),
            "-s".into(), // short flags aren't inspected
            // Degenerate forms: the bare `--` (end-of-flags
            // marker used by many CLIs) and `--=value` (empty
            // name before `=`) both skip the allowlist check
            // because the extracted flag name is empty. Pin the
            // empty-name skip path so a future refactor can't
            // accidentally treat them as unknown long flags.
            "--".into(),
            "--=value".into(),
        ];
        validate_known_flags(&WITH_ALLOWLIST, &args)
            .expect("all long flags in allowlist must pass");
    }

    /// Fail-fast ordering: when args contain a known flag, a
    /// typo, then another known flag, the error must name ONLY
    /// the typo — the validator bails on the first unknown flag
    /// without continuing to inspect later args.
    #[test]
    fn validate_known_flags_fails_fast_on_first_unknown() {
        use crate::test_support::{OutputFormat, Payload, PayloadKind};
        static WITH_ALLOWLIST: Payload = Payload {
            name: "with_allowlist",
            kind: PayloadKind::Binary("/bin/true"),
            output: OutputFormat::ExitCode,
            default_args: &[],
            default_checks: &[],
            metrics: &[],
            include_files: &[],
            uses_parent_pgrp: false,
            known_flags: Some(&["runtime", "threads", "verbose"]),
            metric_bounds: None,
        };
        let args = vec!["--runtime=30".into(), "--threds".into(), "--verbose".into()];
        let err = validate_known_flags(&WITH_ALLOWLIST, &args)
            .expect_err("typo between two known flags must be rejected");
        let msg = format!("{err:#}");
        assert!(msg.contains("--threds"), "error must name the typo: {msg}");
        assert!(
            !msg.contains("--verbose"),
            "error must not mention the later known flag '--verbose' \
             — fail-fast broke: {msg}",
        );
    }

    /// A `--flag` whose bare name is not on the allowlist bails
    /// with a message naming both the offending flag and the
    /// allowlist — the loud-typo-detection contract.
    #[test]
    fn validate_known_flags_rejects_unknown_long_flag() {
        use crate::test_support::{OutputFormat, Payload, PayloadKind};
        static WITH_ALLOWLIST: Payload = Payload {
            name: "with_allowlist",
            kind: PayloadKind::Binary("/bin/true"),
            output: OutputFormat::ExitCode,
            default_args: &[],
            default_checks: &[],
            metrics: &[],
            include_files: &[],
            uses_parent_pgrp: false,
            known_flags: Some(&["runtime", "threads"]),
            metric_bounds: None,
        };
        // "threds" is a typo for "threads" — the exact failure
        // the allowlist exists to catch.
        let args = vec!["--threds".to_string(), "4".to_string()];
        let err = validate_known_flags(&WITH_ALLOWLIST, &args).expect_err("typo must be rejected");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("--threds"),
            "error must name the offending flag: {msg}",
        );
        assert!(
            msg.contains("known_flags allowlist"),
            "error must mention the allowlist surface: {msg}",
        );
    }

    /// `known_flags: None` (the default on every Payload that
    /// doesn't opt in) lets every `--flag` through without
    /// inspection. Required for payloads that wrap binaries with
    /// open-ended flag surfaces (stress-ng, fio, schbench).
    #[test]
    fn validate_known_flags_none_is_permissive() {
        use crate::test_support::{OutputFormat, Payload, PayloadKind};
        static NO_ALLOWLIST: Payload = Payload {
            name: "no_allowlist",
            kind: PayloadKind::Binary("/bin/true"),
            output: OutputFormat::ExitCode,
            default_args: &[],
            default_checks: &[],
            metrics: &[],
            include_files: &[],
            uses_parent_pgrp: false,
            known_flags: None,
            metric_bounds: None,
        };
        let args: Vec<String> = vec![
            "--anything".into(),
            "--whatever=x".into(),
            "--threds".into(),
        ];
        validate_known_flags(&NO_ALLOWLIST, &args).expect("None allowlist must pass any flag");
    }

    // -- Op discriminant tests --

    #[test]
    fn op_discriminant_unique() {
        use crate::test_support::{OutputFormat, Payload, PayloadKind};
        static TRUE_BIN: Payload = Payload {
            name: "true_bin",
            kind: PayloadKind::Binary("/bin/true"),
            output: OutputFormat::ExitCode,
            default_args: &[],
            default_checks: &[],
            metrics: &[],
            include_files: &[],
            uses_parent_pgrp: false,
            known_flags: None,
            metric_bounds: None,
        };
        let ops: Vec<Op> = vec![
            Op::AddCgroup { name: "a".into() },
            Op::AddCgroupDef {
                def: CgroupDef::named("a"),
            },
            Op::RemoveCgroup { cgroup: "a".into() },
            Op::SetCpuset {
                cgroup: "a".into(),
                cpus: CpusetSpec::exact([]),
            },
            Op::ClearCpuset { cgroup: "a".into() },
            Op::SwapCpusets {
                a: "a".into(),
                b: "b".into(),
            },
            Op::SpawnWorkers {
                cgroup: "a".into(),
                work: Default::default(),
            },
            Op::StopCgroup { cgroup: "a".into() },
            Op::SetAffinity {
                cgroup: "a".into(),
                affinity: Default::default(),
            },
            Op::SpawnHost {
                work: Default::default(),
            },
            Op::MoveAllTasks {
                from: "a".into(),
                to: "b".into(),
            },
            Op::RunPayload {
                payload: &TRUE_BIN,
                args: vec![],
                cgroup: None,
            },
            Op::WaitPayload {
                name: "p".into(),
                cgroup: None,
            },
            Op::KillPayload {
                name: "p".into(),
                cgroup: None,
            },
            Op::FreezeCgroup { cgroup: "a".into() },
            Op::UnfreezeCgroup { cgroup: "a".into() },
            Op::CaptureSnapshot {
                name: "snap".into(),
            },
            Op::WatchSnapshot {
                symbol: "kernel.x".into(),
            },
            Op::WriteKernelHot {
                writes: vec![(KernelTarget::symbol("x"), KernelValue::u64(0))],
            },
            Op::WriteKernelCold {
                writes: vec![(KernelTarget::symbol("x"), KernelValue::u64(0))],
            },
            Op::ReadKernelHot {
                tag: "t".into(),
                target: KernelTarget::symbol("x"),
                width: KernelValueWidth::u64(),
            },
            Op::ReadKernelCold {
                tag: "t".into(),
                target: KernelTarget::symbol("x"),
                width: KernelValueWidth::u64(),
            },
        ];
        let mut seen = std::collections::BTreeSet::new();
        for op in &ops {
            assert!(seen.insert(op.discriminant()), "duplicate discriminant");
        }
    }

    /// Pins every Op variant's exact discriminant value against the
    /// canonical `OpKind::bit_index` match in types.rs. A renumbering
    /// or reordering surfaces here naming the specific variant that
    /// moved — complementing `op_kind_bit_indices_are_unique_and_contiguous`
    /// (whose contiguity arm surfaces gaps as sorted indices only,
    /// not the offending variant; its uniqueness arm DOES name
    /// variants via the `{:?}` of `(OpKind, bit_index)` pairs) and
    /// `op_discriminant_unique` (which proves no collisions via the
    /// `BTreeSet::insert` "duplicate discriminant" panic). The
    /// variant-name label on each `assert_eq!` 3rd arg makes a
    /// multi-variant failure operator-readable: the cargo-test output
    /// names each variant whose discriminant drifted, no
    /// source-cross-reference needed.
    #[test]
    fn op_discriminant_values() {
        use crate::test_support::{OutputFormat, Payload, PayloadKind};
        static TRUE_BIN: Payload = Payload {
            name: "true_bin",
            kind: PayloadKind::Binary("/bin/true"),
            output: OutputFormat::ExitCode,
            default_args: &[],
            default_checks: &[],
            metrics: &[],
            include_files: &[],
            uses_parent_pgrp: false,
            known_flags: None,
            metric_bounds: None,
        };
        assert_eq!(
            Op::AddCgroup { name: "a".into() }.discriminant(),
            0,
            "AddCgroup",
        );
        assert_eq!(
            Op::AddCgroupDef {
                def: CgroupDef::named("a")
            }
            .discriminant(),
            1,
            "AddCgroupDef",
        );
        assert_eq!(
            Op::RemoveCgroup { cgroup: "a".into() }.discriminant(),
            2,
            "RemoveCgroup",
        );
        assert_eq!(
            Op::SetCpuset {
                cgroup: "a".into(),
                cpus: CpusetSpec::Llc(0),
            }
            .discriminant(),
            3,
            "SetCpuset",
        );
        assert_eq!(
            Op::ClearCpuset { cgroup: "a".into() }.discriminant(),
            4,
            "ClearCpuset",
        );
        assert_eq!(
            Op::SwapCpusets {
                a: "a".into(),
                b: "b".into(),
            }
            .discriminant(),
            5,
            "SwapCpusets",
        );
        assert_eq!(
            Op::SpawnWorkers {
                cgroup: "a".into(),
                work: WorkSpec::default(),
            }
            .discriminant(),
            6,
            "Spawn",
        );
        assert_eq!(
            Op::StopCgroup { cgroup: "a".into() }.discriminant(),
            7,
            "StopCgroup",
        );
        assert_eq!(
            Op::SetAffinity {
                cgroup: "a".into(),
                affinity: AffinityIntent::Inherit,
            }
            .discriminant(),
            8,
            "SetAffinity",
        );
        assert_eq!(
            Op::SpawnHost {
                work: Default::default()
            }
            .discriminant(),
            9,
            "SpawnHost",
        );
        assert_eq!(
            Op::MoveAllTasks {
                from: "a".into(),
                to: "b".into()
            }
            .discriminant(),
            10,
            "MoveAllTasks",
        );
        assert_eq!(
            Op::RunPayload {
                payload: &TRUE_BIN,
                args: vec![],
                cgroup: None,
            }
            .discriminant(),
            11,
            "RunPayload",
        );
        assert_eq!(
            Op::WaitPayload {
                name: "p".into(),
                cgroup: None,
            }
            .discriminant(),
            12,
            "WaitPayload",
        );
        assert_eq!(
            Op::KillPayload {
                name: "p".into(),
                cgroup: None,
            }
            .discriminant(),
            13,
            "KillPayload",
        );
        assert_eq!(
            Op::FreezeCgroup { cgroup: "a".into() }.discriminant(),
            14,
            "FreezeCgroup",
        );
        assert_eq!(
            Op::UnfreezeCgroup { cgroup: "a".into() }.discriminant(),
            15,
            "UnfreezeCgroup",
        );
        assert_eq!(
            Op::CaptureSnapshot {
                name: "snap".into()
            }
            .discriminant(),
            16,
            "Snapshot",
        );
        assert_eq!(
            Op::WatchSnapshot {
                symbol: "kernel.x".into()
            }
            .discriminant(),
            17,
            "WatchSnapshot",
        );
        assert_eq!(
            Op::WriteKernelHot {
                writes: vec![(KernelTarget::symbol("x"), KernelValue::u64(0))]
            }
            .discriminant(),
            18,
            "WriteKernelHot",
        );
        assert_eq!(
            Op::WriteKernelCold {
                writes: vec![(KernelTarget::symbol("x"), KernelValue::u64(0))]
            }
            .discriminant(),
            19,
            "WriteKernelCold",
        );
        assert_eq!(
            Op::ReadKernelHot {
                tag: "t".into(),
                target: KernelTarget::symbol("x"),
                width: KernelValueWidth::u64(),
            }
            .discriminant(),
            20,
            "ReadKernelHot",
        );
        assert_eq!(
            Op::ReadKernelCold {
                tag: "t".into(),
                target: KernelTarget::symbol("x"),
                width: KernelValueWidth::u64(),
            }
            .discriminant(),
            21,
            "ReadKernelCold",
        );
        static SCHED_FIXTURE: crate::test_support::Scheduler =
            crate::test_support::Scheduler::EEVDF;
        assert_eq!(
            Op::AttachScheduler {
                scheduler: &SCHED_FIXTURE,
            }
            .discriminant(),
            22,
            "AttachScheduler",
        );
        assert_eq!(Op::DetachScheduler.discriminant(), 23, "DetachScheduler",);
        assert_eq!(Op::RestartScheduler.discriminant(), 24, "RestartScheduler",);
        assert_eq!(
            Op::ReplaceScheduler {
                scheduler: &SCHED_FIXTURE,
            }
            .discriminant(),
            25,
            "ReplaceScheduler",
        );
    }

    // -- seeded_rng tests --

    #[test]
    fn seeded_rng_deterministic() {
        use rand::RngExt;
        let mut rng1 = seeded_rng(42);
        let mut rng2 = seeded_rng(42);
        for _ in 0..100 {
            assert_eq!(rng1.random::<u64>(), rng2.random::<u64>());
        }
    }

    #[test]
    fn seeded_rng_different_seeds_differ() {
        use rand::RngExt;
        let mut rng1 = seeded_rng(1);
        let mut rng2 = seeded_rng(2);
        let same = (0..10).all(|_| rng1.random::<u64>() == rng2.random::<u64>());
        assert!(!same);
    }

    // -- HoldSpec validate --

    #[test]
    fn holdspec_validate_accepts_valid() {
        HoldSpec::Frac(0.5).validate().unwrap();
        HoldSpec::Frac(1.0).validate().unwrap();
        HoldSpec::Fixed(Duration::from_millis(1))
            .validate()
            .unwrap();
        HoldSpec::Loop {
            interval: Duration::from_millis(100),
        }
        .validate()
        .unwrap();
    }

    #[test]
    fn holdspec_validate_accepts_fixed_zero() {
        HoldSpec::Fixed(Duration::ZERO)
            .validate()
            .expect("Duration::ZERO is valid for settle/op-only steps");
    }

    #[test]
    fn holdspec_validate_rejects_frac_zero() {
        let err = HoldSpec::Frac(0.0).validate().unwrap_err();
        assert!(err.contains("Frac") && err.contains("> 0"), "got: {err}");
    }

    #[test]
    fn holdspec_validate_rejects_frac_negative() {
        let err = HoldSpec::Frac(-0.5).validate().unwrap_err();
        assert!(err.contains("Frac") && err.contains("> 0"), "got: {err}");
    }

    #[test]
    fn holdspec_validate_rejects_frac_nan() {
        let err = HoldSpec::Frac(f64::NAN).validate().unwrap_err();
        assert!(
            err.contains("not finite") || err.contains("NaN"),
            "got: {err}"
        );
    }

    #[test]
    fn holdspec_validate_rejects_frac_inf() {
        let err = HoldSpec::Frac(f64::INFINITY).validate().unwrap_err();
        assert!(
            err.contains("not finite") || err.contains("Inf"),
            "got: {err}"
        );
    }

    #[test]
    fn holdspec_validate_rejects_loop_zero_interval() {
        let err = HoldSpec::Loop {
            interval: Duration::ZERO,
        }
        .validate()
        .unwrap_err();
        assert!(err.contains("Loop") && err.contains("busy"), "got: {err}");
    }

    // -- HoldSpec variants (exercise constructors + Step storage + PartialEq) --

    #[test]
    fn holdspec_frac() {
        let step = Step::new(vec![], HoldSpec::frac(0.5));
        assert_eq!(step.hold, HoldSpec::Frac(0.5));
    }

    #[test]
    fn holdspec_fixed() {
        let step = Step::new(vec![], HoldSpec::fixed(Duration::from_secs(3)));
        assert_eq!(step.hold, HoldSpec::Fixed(Duration::from_secs(3)));
    }

    #[test]
    fn holdspec_loop() {
        let step = Step::new(vec![], HoldSpec::loop_at(Duration::from_millis(100)));
        assert_eq!(
            step.hold,
            HoldSpec::Loop {
                interval: Duration::from_millis(100)
            }
        );
    }

    /// Drive `HoldSpec::Loop` end-to-end via `execute_steps` against
    /// the mock CgroupOps. The Loop arm of `run_step` (mod.rs:1163-1180)
    /// fires `apply_ops` repeatedly at `interval` until `ctx.duration`
    /// elapses; each iteration's SetCpuset op records a
    /// `CgroupCall::SetCpuset` in the mock. After the scenario
    /// completes, the mock's SetCpuset count proves the loop actually
    /// repeated — distinguishing the Loop path from the Fixed/Frac
    /// single-apply path. `sched_pid = None` (inherited from `mock_ctx`)
    /// makes `sleep_or_sched_died` a plain sleep with no liveness probe
    /// (verified at mod.rs:993-996), so the loop exits cleanly on the
    /// duration deadline rather than on a spurious dead-scheduler signal.
    /// `duration` is overridden to 150ms (vs `mock_ctx`'s 1-second
    /// default) to keep the unit-test runtime short. Lower bound is
    /// loose (>= 2) to absorb CI timing variance — the contract being
    /// pinned is "repeats at least once", not "fires exactly N times".
    #[test]
    fn holdspec_loop_apply_path_repeats_ops_until_duration_elapses() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let mut ctx = mock_ctx(&mock, &topo);
        ctx.duration = Duration::from_millis(150);
        let steps = vec![Step::new(
            vec![Op::set_cpuset("loop_test", CpusetSpec::Llc(0))],
            HoldSpec::loop_at(Duration::from_millis(30)),
        )];
        let result = execute_steps(&ctx, steps)
            .expect("HoldSpec::Loop apply path must succeed against mock cgroups");
        assert!(
            result.is_pass(),
            "scenario must pass with no failing assertions; got: {:?}",
            result.outcomes,
        );
        let set_cpuset_calls = mock
            .calls()
            .iter()
            .filter(|c| matches!(c, CgroupCall::SetCpuset(name, _) if name == "loop_test"))
            .count();
        assert!(
            set_cpuset_calls >= 2,
            "HoldSpec::Loop with interval=30ms over duration=150ms must fire \
             SetCpuset at least twice; got {set_cpuset_calls} calls. The Loop \
             arm of run_step (mod.rs:1163) must invoke apply_ops repeatedly \
             until the deadline; a regression that single-shotted the ops \
             would surface here as exactly 1 call.",
        );
    }

    /// The Loop arm's setup pass at mod.rs:1165-1168 runs `apply_setup`
    /// ONCE before entering the while loop, NOT per-iteration. A
    /// regression that moved the `if !step.setup.is_empty()` block
    /// inside the loop would attempt to re-create the same cgroup
    /// every iteration and bail on the second iteration's collision
    /// check (apply_setup's `cgroup_name_is_tracked` at mod.rs:1568).
    /// Test pins this by counting `CreateCgroup` calls — must be
    /// exactly 1 even though the loop body iterates multiple times
    /// (verified separately via the SetCpuset count).
    #[test]
    fn holdspec_loop_apply_path_setup_runs_once_not_per_iteration() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let mut ctx = mock_ctx(&mock, &topo);
        ctx.duration = Duration::from_millis(150);
        let steps = vec![
            Step::with_defs(
                vec![CgroupDef::named("setup_cg")],
                HoldSpec::loop_at(Duration::from_millis(30)),
            )
            .set_ops(vec![Op::set_cpuset("setup_cg", CpusetSpec::Llc(0))]),
        ];
        let result = execute_steps(&ctx, steps)
            .expect("HoldSpec::Loop with setup must succeed against mock cgroups");
        assert!(
            result.is_pass(),
            "scenario must pass with no failing assertions; got: {:?}",
            result.outcomes,
        );
        let create_calls = mock
            .calls()
            .iter()
            .filter(|c| matches!(c, CgroupCall::CreateCgroup(name) if name == "setup_cg"))
            .count();
        assert_eq!(
            create_calls, 1,
            "Loop arm's setup pass must run exactly ONCE before the loop body; \
             got {create_calls} CreateCgroup calls. A regression that moved \
             the `if !step.setup.is_empty()` block inside the while loop \
             (mod.rs:1165) would surface here as N > 1 calls (the second \
             iteration's apply_setup would also fail the collision check, \
             but counting reveals the bug source).",
        );
        let set_cpuset_calls = mock
            .calls()
            .iter()
            .filter(|c| matches!(c, CgroupCall::SetCpuset(name, _) if name == "setup_cg"))
            .count();
        assert!(
            set_cpuset_calls >= 2,
            "Loop body must repeat SetCpuset >= 2 times despite setup running \
             once; got {set_cpuset_calls}. Pairs with the create-once check \
             above to pin the full setup-once + ops-many contract.",
        );
    }

    /// `interval > duration` is a degenerate-but-valid Loop config:
    /// the while loop body runs exactly ONCE (deadline reached after
    /// the first apply_ops + sleep). Pins the exact-iteration
    /// contract via `assert_eq!(..., 1)` — catches BOTH a regression
    /// that skipped the first apply_ops (0 calls) AND a regression
    /// in the deadline-min logic at mod.rs:1175 that let the second
    /// iteration's sleep underflow (2+ calls). The boundary behavior
    /// at mod.rs:1175-1179 (`sleep_or_sched_died(remaining.min(interval), ...)`)
    /// ensures sleep is capped at the remaining time so the loop
    /// exits promptly on the next deadline check.
    #[test]
    fn holdspec_loop_apply_path_fires_exactly_once_when_interval_exceeds_duration() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let mut ctx = mock_ctx(&mock, &topo);
        ctx.duration = Duration::from_millis(30);
        let steps = vec![Step::new(
            vec![Op::set_cpuset("brief_loop", CpusetSpec::Llc(0))],
            HoldSpec::loop_at(Duration::from_millis(100)),
        )];
        let result = execute_steps(&ctx, steps)
            .expect("HoldSpec::Loop with interval > duration must succeed against mock");
        assert!(
            result.is_pass(),
            "scenario must pass with no failing assertions; got: {:?}",
            result.outcomes,
        );
        let set_cpuset_calls = mock
            .calls()
            .iter()
            .filter(|c| matches!(c, CgroupCall::SetCpuset(name, _) if name == "brief_loop"))
            .count();
        assert_eq!(
            set_cpuset_calls, 1,
            "interval (100ms) > duration (30ms) must fire SetCpuset exactly \
             once; got {set_cpuset_calls}. The loop body should run a single \
             iteration: enter loop (now < deadline) → apply_ops → sleep \
             min(remaining, interval) = ~30ms → next deadline check fails. \
             0 calls = a regression that skipped the first apply_ops; 2+ \
             calls = a regression in the deadline-min logic at mod.rs:1175 \
             that let the second iteration's sleep underflow.",
        );
    }

    /// The Loop arm's sched-died-early-exit path (mod.rs:1177-1180)
    /// fires when `sleep_or_sched_died` observes the scheduler pid
    /// has exited mid-loop. Setting `sched_died_during_hold = true`
    /// and returning `Ok(())` is the contract — the outer caller
    /// (mod.rs:911-922) reads the flag and stamps one of
    /// `DetailKind::SchedulerCrashed` /
    /// `DetailKind::SchedulerExitedCleanly` /
    /// `DetailKind::SchedulerDiedUnknownReason` (chosen by
    /// `sched_died_detail_kind` reading the probe BSS latch) with
    /// `format_sched_died_during_workload`,
    /// then marks the AssertResult `passed = false`.
    ///
    /// Implementation: use `libc::pid_t::MAX` as the dead pid. The
    /// kernel's PID_MAX_LIMIT (include/linux/threads.h) caps real
    /// pids well below `i32::MAX`, so `pidfd_open` on `pid_t::MAX`
    /// always returns ESRCH, which `sleep_or_sched_died` maps to
    /// "dead, return true." This pattern matches
    /// [`crate::scenario::process_alive_nonexistent_pid`] (the same
    /// trick is used to assert process-alive's no-such-pid path
    /// without a fork+reap race window).
    ///
    /// Pins: (1) `sched_pid` carrying a dead pid into the Loop arm
    /// exits the while-loop after the first apply_ops iteration;
    /// (2) the `sched_died_during_hold = true` write at mod.rs:1178
    /// reaches the outer caller; (3) the outer caller pushes
    /// one of the three sched-died `DetailKind` variants and marks
    /// `passed = false`. A regression that DROPPED the early-exit
    /// (loop runs all iterations after the death is observed) would
    /// surface as multiple SetCpuset calls; a regression that
    /// DROPPED the `sched_died_during_hold = true` write would
    /// surface as passed=true with no sched-died detail. Note that
    /// `return Ok(())` vs `break` produce identical observable
    /// state here because the Loop arm is the last operation in
    /// run_step's match block — both exit the while loop and fall
    /// through to the same return — so the count assertion catches
    /// loss of the early-exit BEHAVIOR, not the specific keyword
    /// chosen to implement it.
    #[test]
    fn holdspec_loop_arm_exits_early_when_sched_dies_during_hold() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let mut ctx = mock_ctx(&mock, &topo);
        // Short total duration keeps the test fast; the loop should
        // exit on iteration 1 long before this deadline anyway.
        ctx.duration = Duration::from_millis(150);
        // libc::pid_t::MAX is above kernel PID_MAX_LIMIT, so
        // pidfd_open inside sleep_or_sched_died returns ESRCH
        // immediately. Same trick as scenario::process_alive's
        // no-such-pid test. Publishes via the SCHED_PID atomic
        // because the death-detection sites in apply_ops read
        // `crate::vmm::rust_init::sched_pid()` live (swap-aware
        // for Op::ReplaceScheduler) rather than the
        // `ctx.sched_pid` snapshot.
        ctx.sched_pid = Some(libc::pid_t::MAX);
        crate::vmm::rust_init::set_sched_pid(libc::pid_t::MAX);
        // SCHED_PID is a process-global atomic — restore to 0 on
        // exit so this test doesn't pollute the empty-pid contract
        // of neighbor tests (e.g. apply_ops_detach_scheduler_bails_when_no_scheduler_attached)
        // that read sched_pid() and expect None.
        struct ResetSchedPid;
        impl Drop for ResetSchedPid {
            fn drop(&mut self) {
                crate::vmm::rust_init::set_sched_pid(0);
            }
        }
        let _reset = ResetSchedPid;
        let steps = vec![Step::new(
            vec![Op::set_cpuset("died_test", CpusetSpec::Llc(0))],
            HoldSpec::loop_at(Duration::from_millis(30)),
        )];
        let result = execute_steps(&ctx, steps).expect(
            "Loop arm must return Ok even when sched dies — the death \
             is surfaced via sched_died_during_hold + one of the three sched-died DetailKind variants, \
             NOT as an Err out of run_step",
        );
        assert!(
            !result.is_pass(),
            "sched-died during the Loop hold must mark passed=false; \
             got passed=true with details: {:?}",
            result.outcomes,
        );
        let sched_died_details: Vec<_> = result
            .failure_details()
            .filter(|d| {
                matches!(
                    d.kind,
                    crate::assert::DetailKind::SchedulerCrashed
                        | crate::assert::DetailKind::SchedulerExitedCleanly
                        | crate::assert::DetailKind::SchedulerDiedUnknownReason
                )
            })
            .collect();
        assert_eq!(
            sched_died_details.len(),
            1,
            "must push exactly one sched-died DetailKind detail (from \
             mod.rs:911-922); got {} sched-died failures out of {} total \
             failures: {:?}",
            sched_died_details.len(),
            result.failure_details().count(),
            result.outcomes,
        );
        let set_cpuset_calls = mock
            .calls()
            .iter()
            .filter(|c| matches!(c, CgroupCall::SetCpuset(name, _) if name == "died_test"))
            .count();
        // First iteration's apply_ops at mod.rs:1175 fires BEFORE
        // sleep_or_sched_died at mod.rs:1177, so a sched-died-from-
        // entry still records exactly one SetCpuset call. A
        // regression that DROPPED the early-exit (loop runs all
        // iterations after the death is observed) would surface as
        // multiple calls here; a regression that skipped the first
        // apply_ops would surface as zero.
        assert_eq!(
            set_cpuset_calls, 1,
            "sched-died-on-entry must apply ops once (iter 1) then exit; \
             got {set_cpuset_calls} SetCpuset calls. > 1 means the loop \
             continued past the sched-died signal (early-exit dropped); \
             0 means apply_ops was gated on liveness (would surface as \
             a missing-apply regression).",
        );
    }

    /// The Loop arm's apply_ops error-propagation path: an
    /// `apply_ops` Err on iteration N at mod.rs:1175 exits the loop
    /// via the `drain_on_err!` macro (mod.rs:1151-1161) which
    /// propagates the Err up through `run_step`. The outer caller
    /// at mod.rs:883-901 converts the Err to
    /// `Ok(AssertResult { passed: false, details: [...
    /// DetailKind::Other ...] })` so a mid-scenario failure still
    /// returns the merged prior-step results plus the error context
    /// rather than an opaque Err.
    ///
    /// Implementation: `MockCgroupOps::fail_call_at(2, "...")` fails
    /// the third cgroup call. The cgroup call sequence is:
    /// - Index 0: `Setup` (run_scenario at mod.rs:706-708 calls
    ///   `cgroups.setup(&required)` before any step runs)
    /// - Index 1: Iteration 1's SetCpuset → Ok
    /// - Index 2: Iteration 2's SetCpuset → Err (injected)
    ///
    /// Expected post-state: exactly 2 SetCpuset calls (iter 1 ok +
    /// iter 2 fail, no third iteration), result.passed=false,
    /// DetailKind::Other detail containing the injected message.
    /// A regression that allowed the loop to continue past the
    /// failing iteration would surface as 3+ calls.
    ///
    /// SCOPE NOTE: this test does NOT verify the
    /// `scenario.drain_all_payloads()` side effect inside
    /// `drain_on_err!` because the fixture has no live payloads.
    /// The drain-on-err contract at the macro level is verified by
    /// `apply_ops_error_does_not_lose_live_payload_handles`
    /// (sibling test in this module — grep by name) which checks
    /// `apply_ops` itself doesn't drain (so `execute_steps`'
    /// `drain_on_err!` is responsible). A dedicated Loop-arm drain
    /// test with a live payload fixture is a follow-up (see queue
    /// task for "Loop-arm drain verification with live payload
    /// fixture") because the test infrastructure requires a custom
    /// PayloadHandle observer to distinguish drain-side `.kill()`
    /// from `Drop`-side SIGKILL.
    #[test]
    fn holdspec_loop_arm_propagates_apply_ops_error() {
        let mock = MockCgroupOps::new();
        // Inject an error at the THIRD cgroup call (index 2). The
        // sequence is: Setup (index 0) + iter-1 SetCpuset (index 1,
        // Ok) + iter-2 SetCpuset (index 2, Err injected). See
        // run_scenario at mod.rs:706-708 for the Setup-first call.
        mock.fail_call_at(2, "injected SetCpuset error mid-iteration");
        let topo = mock_topo();
        let mut ctx = mock_ctx(&mock, &topo);
        // Long enough for >2 iterations at 30ms interval if the
        // loop incorrectly continued past the failing iteration.
        ctx.duration = Duration::from_millis(200);
        let steps = vec![Step::new(
            vec![Op::set_cpuset("err_drain_test", CpusetSpec::Llc(0))],
            HoldSpec::loop_at(Duration::from_millis(30)),
        )];
        let result = execute_steps(&ctx, steps).expect(
            "execute_steps converts step Err to Ok(passed=false) per \
             mod.rs:883-901; the Err must NOT propagate to the caller",
        );
        assert!(
            !result.is_pass(),
            "injected apply_ops error must mark passed=false; got \
             passed=true with details: {:?}",
            result.outcomes,
        );
        let other_details: Vec<_> = result
            .failure_details()
            .filter(|d| {
                matches!(d.kind, crate::assert::DetailKind::Other)
                    && d.message.contains("injected SetCpuset error mid-iteration")
            })
            .collect();
        assert_eq!(
            other_details.len(),
            1,
            "step Err must surface exactly once as DetailKind::Other \
             carrying the injected message; got {} matching details out \
             of {} total: {:?}",
            other_details.len(),
            result.outcomes.len(),
            result.outcomes,
        );
        let set_cpuset_calls = mock
            .calls()
            .iter()
            .filter(|c| matches!(c, CgroupCall::SetCpuset(name, _) if name == "err_drain_test"))
            .count();
        // 1 ok + 1 fail = 2. A third call would mean the loop body
        // ignored the Err and continued to the next interval —
        // a regression in either drain_on_err! propagation or the
        // run_step Loop arm's `?`-out behavior.
        assert_eq!(
            set_cpuset_calls, 2,
            "loop must stop at the failed iteration (1 ok + 1 fail = 2); \
             got {set_cpuset_calls} SetCpuset calls. Any value > 2 means \
             the apply_ops Err was swallowed and the Loop arm continued, \
             which would also bypass the drain_on_err! payload-kill path \
             (silent metric loss).",
        );
    }

    /// Loop-arm `drain_on_err!` invokes `.kill()` on every live
    /// payload handle when `apply_ops` returns Err mid-iteration,
    /// rather than letting `PayloadHandle::Drop` SIGKILL them via
    /// the process-group fallback. The two paths differ in
    /// observable behavior at `payload_run.rs::PayloadHandle::drop`:
    /// `.kill()` calls `self.child.take()` before reaping, so by the
    /// time Drop runs `self.child.is_none()` and the diagnostic
    /// `eprintln!("ktstr: PayloadHandle for 'X' dropped without
    /// wait/kill — process group SIGKILLed, metrics not recorded.")`
    /// does NOT fire. If `drain_on_err!` regressed to a bare `?`-out
    /// (or any path that doesn't drain), Drop would see
    /// `self.child.is_some()`, fire the eprintln, and the captured
    /// stderr would contain "dropped without wait/kill".
    ///
    /// Pairs with `holdspec_loop_arm_propagates_apply_ops_error`:
    /// that test verifies Err PROPAGATION via the macro (loop stops,
    /// passed=false, exactly 2 SetCpuset calls). This test verifies
    /// the SIDE EFFECT (kill-not-Drop) using a live `/bin/sleep`
    /// fixture spawned by `Op::run_payload`. Both tests must pass —
    /// drain_on_err! must both propagate Err AND drain payloads, and
    /// covering only one half misses regressions in the other.
    #[test]
    fn holdspec_loop_arm_drain_on_err_kills_live_payload_via_kill_not_drop() {
        use crate::test_support::{OutputFormat, Payload, PayloadKind};
        static SLEEP: Payload = Payload {
            name: "drain_on_err_observer",
            kind: PayloadKind::Binary("/bin/sleep"),
            output: OutputFormat::ExitCode,
            default_args: &[],
            default_checks: &[],
            metrics: &[],
            include_files: &[],
            uses_parent_pgrp: false,
            known_flags: None,
            metric_bounds: None,
        };

        let mock = MockCgroupOps::new();
        // Index sequence (cgroup-op counts only):
        //   0 = run_scenario Setup (mod.rs:706-708)
        //   1 = iter-1 SetCpuset (Ok)
        //   2 = iter-2 SetCpuset (Err injected here)
        // Op::run_payload without an explicit cgroup arg does NOT go
        // through MockCgroupOps — the SLEEP child spawns directly,
        // so it's live in state.payload_handles when the iter-2
        // SetCpuset Err triggers drain_on_err!.
        mock.fail_call_at(2, "injected SetCpuset error to trigger drain_on_err");
        let topo = mock_topo();
        let mut ctx = mock_ctx(&mock, &topo);
        // Wide enough for iter-2 to reach the SetCpuset failure point
        // at the 30ms interval below; the Err short-circuits the
        // remaining iterations.
        ctx.duration = Duration::from_millis(200);

        let steps = vec![Step::new(
            vec![
                Op::run_payload(&SLEEP, vec!["3600".into()]),
                Op::set_cpuset("drain_observer_cg", CpusetSpec::Llc(0)),
            ],
            HoldSpec::loop_at(Duration::from_millis(30)),
        )];

        let (_, captured_stderr) = crate::test_support::test_helpers::capture_stderr(|| {
            let _ = execute_steps(&ctx, steps).expect(
                "execute_steps converts step Err to Ok(passed=false); the \
                     Err must NOT propagate to the caller",
            );
        });

        let stderr_text = String::from_utf8_lossy(&captured_stderr);
        assert!(
            !stderr_text.contains("dropped without wait/kill"),
            "drain_on_err! must invoke .kill() on every live payload \
             handle, not let them fall through to PayloadHandle::Drop's \
             process-group SIGKILL. Observed the Drop-path eprintln in \
             captured stderr — drain_on_err! regressed (Err propagated \
             but payloads were leaked to Drop). Full captured stderr: \
             {stderr_text:?}",
        );
    }

    // -- HoldSpec PartialEq (load-bearing semantic pins) --

    /// Payload participates in equality across every variant. A
    /// derived PartialEq guarantees this, but pinning it explicitly
    /// catches a hypothetical hand-rolled `|_, _| true` regression
    /// AND a partial-derive that ignores struct-variant field
    /// contents.
    #[test]
    fn holdspec_partialeq_payload_participates_in_equality() {
        assert_ne!(HoldSpec::Frac(0.5), HoldSpec::Frac(0.75));
        assert_ne!(
            HoldSpec::Fixed(Duration::from_secs(1)),
            HoldSpec::Fixed(Duration::from_secs(2))
        );
        assert_ne!(
            HoldSpec::Loop {
                interval: Duration::from_millis(100)
            },
            HoldSpec::Loop {
                interval: Duration::from_millis(200)
            }
        );
    }

    /// IEEE 754: 0.1 + 0.2 != 0.3. PartialEq on Frac inherits strict
    /// float equality so a Frac built from arithmetic does NOT
    /// compare equal to a Frac with the rounded literal. Pins the
    /// documented behavior so a future "fuzzy PartialEq" rewrite
    /// doesn't silently change the contract.
    #[test]
    fn holdspec_partialeq_frac_float_strict_equality() {
        assert_ne!(HoldSpec::Frac(0.1 + 0.2), HoldSpec::Frac(0.3));
    }

    /// IEEE 754: NaN != NaN, even against itself. PartialEq on Frac
    /// inherits the non-reflexive behavior. `HoldSpec::validate`
    /// rejects Frac(NaN) at intake so production code paths don't
    /// see this, but the type-level PartialEq contract must hold —
    /// pinned against a future "treat NaN as reflexive" rewrite.
    #[test]
    fn holdspec_partialeq_frac_nan_self_unequal() {
        let nan = HoldSpec::Frac(f64::NAN);
        assert_ne!(nan, nan);
    }

    /// `FULL` is an alias for `Frac(1.0)`. The public-API const
    /// shouldn't drift from the variant it expands to.
    #[test]
    fn holdspec_full_equals_frac_one() {
        assert_eq!(HoldSpec::FULL, HoldSpec::Frac(1.0));
    }

    // Compile-time proof the constructor signatures stay `const fn`.
    // If a future refactor demotes any of these (e.g. by introducing
    // a non-const call internally), the module fails to compile here
    // — surfaces the regression at the layer where const usability
    // matters. Discarded via `_` to avoid namespace pollution.
    const _: HoldSpec = HoldSpec::fixed(Duration::from_secs(1));
    const _: HoldSpec = HoldSpec::frac(0.5);
    const _: HoldSpec = HoldSpec::loop_at(Duration::from_millis(50));

    // -- CpusetSpec::Exact --

    #[test]
    fn cpusetspec_exact_is_passthrough() {
        let cpus: BTreeSet<usize> = [0, 2, 4].iter().copied().collect();
        let spec = CpusetSpec::Exact(cpus.clone());
        let topo = crate::topology::TestTopology::from_vm_topology(
            &crate::vmm::topology::Topology::new(1, 1, 4, 1),
        );
        let cgroups = crate::cgroup::CgroupManager::new("/nonexistent");
        let ctx = Ctx {
            cgroups: &cgroups,
            topo: &topo,
            duration: Duration::from_secs(10),
            workers_per_cgroup: 4,
            sched_pid: None,
            settle: Duration::from_millis(1000),
            work_type_override: None,
            assert: crate::assert::Assert::default_checks(),
            wait_for_map_write: false,
            current_step: std::sync::Arc::new(std::sync::atomic::AtomicU16::new(0)),
        };
        let resolved = spec.resolve(&ctx);
        assert_eq!(resolved, cpus);
    }

    // -- Defense-in-depth: resolve must not panic on spec shapes that
    // -- validate rejects. Each test exercises a concrete panic the
    // -- resolver's hardening guards against.

    #[test]
    fn resolve_disjoint_of_zero_returns_empty_instead_of_panicking() {
        // `usable.len() / of` with of=0 would panic without hardening.
        // Current behavior: returns an empty BTreeSet with a
        // tracing::warn.
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let spec = CpusetSpec::Disjoint { index: 0, of: 0 };
        assert!(spec.resolve(&ctx).is_empty());
    }

    #[test]
    fn resolve_overlap_of_zero_returns_empty_instead_of_panicking() {
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let spec = CpusetSpec::Overlap {
            index: 0,
            of: 0,
            frac: 0.5,
        };
        assert!(spec.resolve(&ctx).is_empty());
    }

    #[test]
    fn resolve_range_inverted_fracs_returns_empty_instead_of_panicking() {
        // Without hardening, `usable[start.min(len)..end.min(len)]`
        // with start_frac > end_frac produced start > end after
        // clamping and panicked the slice operation. Current
        // behavior: the slice is clamped to length-zero instead.
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let spec = CpusetSpec::Range {
            start_frac: 0.8,
            end_frac: 0.2,
        };
        assert!(spec.resolve(&ctx).is_empty());
    }

    #[test]
    fn resolve_range_nan_fracs_clamps_to_zero_instead_of_panicking() {
        // NaN as usize saturates to 0 on stable Rust, but inverted
        // start/end after both saturate is still fine post-fix.
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let spec = CpusetSpec::Range {
            start_frac: f64::NAN,
            end_frac: f64::NAN,
        };
        assert!(spec.resolve(&ctx).is_empty());
    }

    #[test]
    fn resolve_overlap_nonfinite_frac_clamps_to_zero() {
        // NaN frac pre-fix flowed through `(chunk as f64 * frac) as
        // usize` and could produce an out-of-range overlap. Post-fix
        // clamps NaN to 0, yielding the same partition boundaries as
        // Disjoint.
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let spec = CpusetSpec::Overlap {
            index: 0,
            of: 2,
            frac: f64::NAN,
        };
        // No panic; result must be non-empty because index/of are valid.
        let result = spec.resolve(&ctx);
        assert!(!result.is_empty());
    }

    // -- CpusetSpec resolution helpers --

    fn make_ctx(
        llcs: u32,
        cores: u32,
        threads: u32,
    ) -> (crate::cgroup::CgroupManager, crate::topology::TestTopology) {
        let cgroups = crate::cgroup::CgroupManager::new("/nonexistent");
        let topo = crate::topology::TestTopology::from_vm_topology(
            &crate::vmm::topology::Topology::new(1, llcs, cores, threads),
        );
        (cgroups, topo)
    }

    fn ctx_from<'a>(
        cgroups: &'a crate::cgroup::CgroupManager,
        topo: &'a crate::topology::TestTopology,
    ) -> Ctx<'a> {
        Ctx {
            cgroups,
            topo,
            duration: Duration::from_secs(10),
            workers_per_cgroup: 4,
            sched_pid: None,
            settle: Duration::ZERO,
            work_type_override: None,
            assert: crate::assert::Assert::default_checks(),
            wait_for_map_write: false,
            current_step: std::sync::Arc::new(std::sync::atomic::AtomicU16::new(0)),
        }
    }

    // -- CpusetSpec::Disjoint --

    #[test]
    fn cpusetspec_disjoint_two_partitions() {
        let (cg, topo) = make_ctx(1, 8, 1);
        let ctx = ctx_from(&cg, &topo);
        let a = CpusetSpec::Disjoint { index: 0, of: 2 }.resolve(&ctx);
        let b = CpusetSpec::Disjoint { index: 1, of: 2 }.resolve(&ctx);
        // Partitions must be disjoint.
        assert!(a.is_disjoint(&b), "partitions overlap: {:?} vs {:?}", a, b);
        // Together they cover all usable CPUs.
        let usable = ctx.topo.usable_cpuset();
        let union: BTreeSet<usize> = a.union(&b).copied().collect();
        assert_eq!(union, usable);
    }

    #[test]
    fn cpusetspec_disjoint_remainder_to_last() {
        // 7 usable CPUs / 3 partitions = chunk=2, so partition 0=[0,1], 1=[2,3], 2=[4,5,6].
        // Last partition gets the remainder.
        let (cg, topo) = make_ctx(1, 8, 1);
        let ctx = ctx_from(&cg, &topo);
        let usable_len = ctx.topo.usable_cpus().len();
        let c = CpusetSpec::Disjoint { index: 2, of: 3 }.resolve(&ctx);
        let chunk = usable_len / 3;
        // Last partition should be >= chunk size (gets remainder).
        assert!(
            c.len() >= chunk,
            "last partition {}: expected >= {}",
            c.len(),
            chunk
        );
    }

    #[test]
    fn cpusetspec_disjoint_single_partition() {
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let all = CpusetSpec::Disjoint { index: 0, of: 1 }.resolve(&ctx);
        let usable = ctx.topo.usable_cpuset();
        assert_eq!(all, usable);
    }

    #[test]
    fn cpusetspec_disjoint_index_beyond_of_returns_empty() {
        // Defense-in-depth: `validate` rejects index >= of with a clear
        // error, but callers that skip validation (e.g. programmatic
        // spec construction) must not hit the div-by-zero or panic in
        // `resolve`. With index = 5 and of = 3 on 3 usable CPUs
        // (4 total, 1 reserved by `usable_cpus`), chunk = 1 and
        // start = 5 clamps past `usable.len()` to yield an empty set
        // — a safe fallback, not a panic.
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let cpus = CpusetSpec::Disjoint { index: 5, of: 3 }.resolve(&ctx);
        assert!(
            cpus.is_empty(),
            "Disjoint with index beyond `of` must return an empty \
             cpuset rather than panicking, got: {cpus:?}",
        );
    }

    // -- CpusetSpec::Range --

    #[test]
    fn cpusetspec_range_first_half() {
        let (cg, topo) = make_ctx(1, 8, 1);
        let ctx = ctx_from(&cg, &topo);
        let cpus = CpusetSpec::Range {
            start_frac: 0.0,
            end_frac: 0.5,
        }
        .resolve(&ctx);
        let usable = ctx.topo.usable_cpus();
        let expected_len = usable.len() / 2;
        assert_eq!(cpus.len(), expected_len);
        // Should contain the first usable CPUs.
        for &cpu in &cpus {
            assert!(usable.contains(&cpu));
        }
    }

    #[test]
    fn cpusetspec_range_second_half() {
        let (cg, topo) = make_ctx(1, 8, 1);
        let ctx = ctx_from(&cg, &topo);
        let a = CpusetSpec::Range {
            start_frac: 0.0,
            end_frac: 0.5,
        }
        .resolve(&ctx);
        let b = CpusetSpec::Range {
            start_frac: 0.5,
            end_frac: 1.0,
        }
        .resolve(&ctx);
        assert!(a.is_disjoint(&b));
    }

    #[test]
    fn cpusetspec_range_full() {
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let cpus = CpusetSpec::Range {
            start_frac: 0.0,
            end_frac: 1.0,
        }
        .resolve(&ctx);
        let usable = ctx.topo.usable_cpuset();
        assert_eq!(cpus, usable);
    }

    #[test]
    fn cpusetspec_range_empty() {
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let cpus = CpusetSpec::Range {
            start_frac: 0.5,
            end_frac: 0.5,
        }
        .resolve(&ctx);
        assert!(cpus.is_empty());
    }

    #[test]
    fn cpusetspec_range_clamps_to_bounds() {
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        // end_frac > 1.0 should be clamped to usable.len().
        let cpus = CpusetSpec::Range {
            start_frac: 0.0,
            end_frac: 2.0,
        }
        .resolve(&ctx);
        let usable = ctx.topo.usable_cpuset();
        assert_eq!(cpus, usable);
    }

    // -- CpusetSpec::Overlap --

    #[test]
    fn cpusetspec_overlap_neighbors_share_cpus() {
        let (cg, topo) = make_ctx(1, 8, 1);
        let ctx = ctx_from(&cg, &topo);
        let a = CpusetSpec::Overlap {
            index: 0,
            of: 2,
            frac: 0.5,
        }
        .resolve(&ctx);
        let b = CpusetSpec::Overlap {
            index: 1,
            of: 2,
            frac: 0.5,
        }
        .resolve(&ctx);
        let shared: BTreeSet<usize> = a.intersection(&b).copied().collect();
        assert!(!shared.is_empty(), "overlap=0.5 should produce shared CPUs");
    }

    #[test]
    fn cpusetspec_overlap_zero_frac_is_disjoint() {
        let (cg, topo) = make_ctx(1, 8, 1);
        let ctx = ctx_from(&cg, &topo);
        let a = CpusetSpec::Overlap {
            index: 0,
            of: 2,
            frac: 0.0,
        }
        .resolve(&ctx);
        let b = CpusetSpec::Overlap {
            index: 1,
            of: 2,
            frac: 0.0,
        }
        .resolve(&ctx);
        assert!(a.is_disjoint(&b), "frac=0 should be disjoint");
    }

    #[test]
    fn cpusetspec_overlap_last_partition_covers_tail() {
        let (cg, topo) = make_ctx(1, 8, 1);
        let ctx = ctx_from(&cg, &topo);
        let last = CpusetSpec::Overlap {
            index: 2,
            of: 3,
            frac: 0.5,
        }
        .resolve(&ctx);
        let usable = ctx.topo.usable_cpus();
        // Last partition should include the last usable CPU.
        assert!(last.contains(usable.last().unwrap()));
    }

    #[test]
    fn cpusetspec_overlap_first_partition_starts_at_zero() {
        let (cg, topo) = make_ctx(1, 8, 1);
        let ctx = ctx_from(&cg, &topo);
        let first = CpusetSpec::Overlap {
            index: 0,
            of: 3,
            frac: 0.5,
        }
        .resolve(&ctx);
        let usable = ctx.topo.usable_cpus();
        assert!(first.contains(&usable[0]));
    }

    // -- CpusetSpec::Llc --

    #[test]
    fn cpusetspec_llc_index_zero() {
        let (cg, topo) = make_ctx(2, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let cpus = CpusetSpec::Llc(0).resolve(&ctx);
        assert!(!cpus.is_empty());
        // All CPUs in the set should belong to LLC 0.
        let llc0 = ctx.topo.llc_aligned_cpuset(0);
        assert_eq!(cpus, llc0);
    }

    #[test]
    fn cpusetspec_llc_two_llcs_disjoint() {
        let (cg, topo) = make_ctx(2, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let llc0 = CpusetSpec::Llc(0).resolve(&ctx);
        let llc1 = CpusetSpec::Llc(1).resolve(&ctx);
        assert!(llc0.is_disjoint(&llc1), "LLCs should be disjoint");
    }

    // -- CpusetSpec::Numa --

    fn make_numa_ctx(
        numa_nodes: u32,
        llcs: u32,
        cores: u32,
        threads: u32,
    ) -> (crate::cgroup::CgroupManager, crate::topology::TestTopology) {
        let cgroups = crate::cgroup::CgroupManager::new("/nonexistent");
        let topo = crate::topology::TestTopology::from_vm_topology(
            &crate::vmm::topology::Topology::new(numa_nodes, llcs, cores, threads),
        );
        (cgroups, topo)
    }

    #[test]
    fn cpusetspec_numa_node_zero() {
        // 2 NUMA nodes, 4 LLCs (2 per NUMA), 4 cores, 1 thread
        // LLCs 0,1 -> NUMA 0 (CPUs 0-7), LLCs 2,3 -> NUMA 1 (CPUs 8-15)
        let (cg, topo) = make_numa_ctx(2, 4, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let cpus = CpusetSpec::Numa(0).resolve(&ctx);
        let expected: BTreeSet<usize> = (0..8).collect();
        assert_eq!(cpus, expected);
    }

    #[test]
    fn cpusetspec_numa_node_one() {
        let (cg, topo) = make_numa_ctx(2, 4, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let cpus = CpusetSpec::Numa(1).resolve(&ctx);
        let expected: BTreeSet<usize> = (8..16).collect();
        assert_eq!(cpus, expected);
    }

    #[test]
    fn cpusetspec_numa_disjoint() {
        let (cg, topo) = make_numa_ctx(2, 4, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let node0 = CpusetSpec::Numa(0).resolve(&ctx);
        let node1 = CpusetSpec::Numa(1).resolve(&ctx);
        assert!(
            node0.is_disjoint(&node1),
            "NUMA nodes should be disjoint: {:?} vs {:?}",
            node0,
            node1
        );
        let union: BTreeSet<usize> = node0.union(&node1).copied().collect();
        assert_eq!(union, ctx.topo.all_cpuset());
    }

    #[test]
    fn cpusetspec_numa_single_node_returns_all() {
        let (cg, topo) = make_numa_ctx(1, 2, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let cpus = CpusetSpec::Numa(0).resolve(&ctx);
        assert_eq!(cpus, ctx.topo.all_cpuset());
    }

    #[test]
    fn cpusetspec_numa_validate_out_of_range() {
        let (cg, topo) = make_numa_ctx(2, 4, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let spec = CpusetSpec::Numa(5);
        let err = spec.validate(&ctx).unwrap_err();
        assert!(err.contains("out of range"), "got: {err}");
    }

    #[test]
    fn cpusetspec_numa_validate_valid() {
        let (cg, topo) = make_numa_ctx(2, 4, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        assert!(CpusetSpec::Numa(0).validate(&ctx).is_ok());
        assert!(CpusetSpec::Numa(1).validate(&ctx).is_ok());
    }

    #[test]
    fn cpusetspec_numa_convenience_constructor() {
        let spec = CpusetSpec::numa(0);
        assert!(matches!(spec, CpusetSpec::Numa(0)));
    }

    // -- Traverse::generate --

    #[test]
    fn traverse_generate_produces_steps() {
        let (cg, topo) = make_ctx(1, 8, 1);
        let ctx = ctx_from(&cg, &topo);
        let t = Traverse {
            seed: Some(42),
            cgroup_count: 2..=4,
            layouts: vec![Layout::Disjoint],
            phases: 3,
            phase_duration: Duration::from_millis(100),
            settle: Duration::from_millis(50),
            persistent_cgroups: 0,
            cgroup_workloads: vec![WorkSpec::default()],
        };
        let steps = t.generate(&ctx);
        assert_eq!(steps.len(), 3);
        for step in &steps {
            assert!(!step.ops.is_empty(), "each phase should have ops");
        }
    }

    #[test]
    fn traverse_generate_deterministic() {
        let (cg, topo) = make_ctx(1, 8, 1);
        let ctx = ctx_from(&cg, &topo);
        let t = Traverse {
            seed: Some(99),
            cgroup_count: 2..=4,
            layouts: vec![Layout::Disjoint, Layout::Overlap(0.2, 0.5)],
            phases: 5,
            phase_duration: Duration::from_millis(100),
            settle: Duration::from_millis(50),
            persistent_cgroups: 1,
            cgroup_workloads: vec![WorkSpec::default()],
        };
        let steps1 = t.generate(&ctx);
        let steps2 = t.generate(&ctx);
        assert_eq!(steps1.len(), steps2.len());
        for (s1, s2) in steps1.iter().zip(steps2.iter()) {
            assert_eq!(
                s1.ops.len(),
                s2.ops.len(),
                "deterministic seed should produce same ops"
            );
        }
    }

    #[test]
    fn traverse_generate_persistent_cgroups_preserved() {
        let (cg, topo) = make_ctx(1, 8, 1);
        let ctx = ctx_from(&cg, &topo);
        let t = Traverse {
            seed: Some(42),
            cgroup_count: 1..=4,
            layouts: vec![Layout::Disjoint],
            phases: 5,
            phase_duration: Duration::from_millis(100),
            settle: Duration::from_millis(50),
            persistent_cgroups: 2,
            cgroup_workloads: vec![WorkSpec::default()],
        };
        let steps = t.generate(&ctx);
        // Every phase should have at least persistent_cgroups worth of SetCpuset ops
        // (cg_0, cg_1 are never removed).
        for step in &steps {
            let remove_ops: Vec<&Op> = step.ops.iter()
                .filter(|op| matches!(op, Op::RemoveCgroup { cgroup } if cgroup == "cg_0" || cgroup == "cg_1"))
                .collect();
            assert!(
                remove_ops.is_empty(),
                "persistent cgroups should never be removed"
            );
        }
    }

    // -- CgroupDef builder --

    #[test]
    fn cgroup_def_builder_chain() {
        let d = CgroupDef::named("test")
            .cpuset(CpusetSpec::llc(0))
            .workers(8)
            .work_type(WorkType::bursty(
                Duration::from_millis(50),
                Duration::from_millis(100),
            ))
            .sched_policy(crate::workload::SchedPolicy::Batch)
            .swappable(true);
        assert_eq!(d.name, "test");
        assert!(d.cpuset.is_some());
        assert_eq!(d.works.len(), 1);
        assert_eq!(d.works[0].num_workers, Some(8));
        assert!(d.swappable);
    }

    #[test]
    fn cgroup_def_multi_work() {
        let d = CgroupDef::named("multi")
            .work(WorkSpec::default().workers(4).work_type(WorkType::SpinWait))
            .work(
                WorkSpec::default()
                    .workers(2)
                    .work_type(WorkType::YieldHeavy),
            );
        assert_eq!(d.works.len(), 2);
        assert_eq!(d.works[0].num_workers, Some(4));
        assert_eq!(d.works[1].num_workers, Some(2));
    }

    #[test]
    fn cgroup_def_old_api_then_work() {
        let d = CgroupDef::named("mixed")
            .workers(4)
            .work(WorkSpec::default().workers(2));
        assert_eq!(d.works.len(), 2);
        assert_eq!(d.works[0].num_workers, Some(4));
        assert_eq!(d.works[1].num_workers, Some(2));
    }

    #[test]
    fn cgroup_def_work_only_no_phantom() {
        let d = CgroupDef::named("explicit").work(WorkSpec::default().workers(3));
        assert_eq!(d.works.len(), 1);
        assert_eq!(d.works[0].num_workers, Some(3));
    }

    // -- Setup --

    #[test]
    fn setup_defs_resolves() {
        let defs = vec![CgroupDef::named("a"), CgroupDef::named("b")];
        let setup = Setup::Defs(defs);
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let resolved = setup.resolve(&ctx);
        assert_eq!(resolved.len(), 2);
        assert!(!setup.is_empty());
    }

    #[test]
    fn setup_defs_empty() {
        let setup = Setup::Defs(vec![]);
        assert!(setup.is_empty());
    }

    #[test]
    fn setup_factory_not_empty() {
        let setup = Setup::Factory(|_| vec![CgroupDef::named("generated")]);
        assert!(!setup.is_empty());
    }

    // -- Step::with_defs / with_ops --

    #[test]
    fn step_with_defs_empty() {
        let step = Step::with_defs(vec![], HoldSpec::frac(0.5));
        assert!(step.setup.is_empty());
        assert!(step.ops.is_empty());
    }

    #[test]
    fn step_with_defs_populated() {
        let step = Step::with_defs(
            vec![CgroupDef::named("cg_0"), CgroupDef::named("cg_1")],
            HoldSpec::fixed(Duration::from_secs(5)),
        );
        assert!(!step.setup.is_empty());
        assert!(step.ops.is_empty());
    }

    #[test]
    fn step_with_defs_then_ops() {
        let step = Step::with_defs(vec![CgroupDef::named("cg_0")], HoldSpec::FULL).set_ops(vec![
            Op::AddCgroup {
                name: "cg_1".into(),
            },
        ]);
        assert!(!step.setup.is_empty());
        assert_eq!(step.ops.len(), 1);
    }

    #[test]
    fn step_set_ops_replaces() {
        let step = Step::new(
            vec![Op::AddCgroup { name: "a".into() }],
            HoldSpec::frac(0.5),
        )
        .set_ops(vec![
            Op::AddCgroup { name: "b".into() },
            Op::RemoveCgroup { cgroup: "c".into() },
        ]);
        assert_eq!(step.ops.len(), 2);
    }

    // -- CpusetSpec::validate --

    #[test]
    fn cpusetspec_validate_disjoint_of_zero() {
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let spec = CpusetSpec::Disjoint { index: 0, of: 0 };
        let err = spec.validate(&ctx).unwrap_err();
        assert!(err.contains("must be > 0"), "got: {err}");
    }

    #[test]
    fn cpusetspec_validate_disjoint_index_ge_of() {
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let spec = CpusetSpec::Disjoint { index: 3, of: 3 };
        let err = spec.validate(&ctx).unwrap_err();
        assert!(err.contains("index 3 >= partition count 3"), "got: {err}");
    }

    #[test]
    fn cpusetspec_validate_overlap_of_zero() {
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let spec = CpusetSpec::Overlap {
            index: 0,
            of: 0,
            frac: 0.5,
        };
        let err = spec.validate(&ctx).unwrap_err();
        assert!(err.contains("must be > 0"), "got: {err}");
    }

    #[test]
    fn cpusetspec_validate_overlap_index_ge_of() {
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let spec = CpusetSpec::Overlap {
            index: 2,
            of: 2,
            frac: 0.5,
        };
        let err = spec.validate(&ctx).unwrap_err();
        assert!(err.contains("index 2 >= partition count 2"), "got: {err}");
    }

    #[test]
    fn cpusetspec_validate_range_start_ge_end() {
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let spec = CpusetSpec::Range {
            start_frac: 0.8,
            end_frac: 0.2,
        };
        let err = spec.validate(&ctx).unwrap_err();
        assert!(err.contains("start_frac"), "got: {err}");
    }

    #[test]
    fn cpusetspec_validate_range_rejects_nan() {
        // Regression: IEEE 754 comparisons with NaN always return false, so
        // `start_frac >= end_frac` failed to reject it. validate() now
        // rejects non-finite fracs explicitly.
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let spec = CpusetSpec::Range {
            start_frac: 0.8,
            end_frac: f64::NAN,
        };
        let err = spec.validate(&ctx).unwrap_err();
        assert!(err.contains("not finite"), "got: {err}");
    }

    #[test]
    fn cpusetspec_validate_range_rejects_infinity() {
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let spec = CpusetSpec::Range {
            start_frac: 0.0,
            end_frac: f64::INFINITY,
        };
        let err = spec.validate(&ctx).unwrap_err();
        assert!(err.contains("not finite"), "got: {err}");
    }

    #[test]
    fn cpusetspec_validate_range_rejects_negative() {
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let spec = CpusetSpec::Range {
            start_frac: -0.5,
            end_frac: 0.5,
        };
        let err = spec.validate(&ctx).unwrap_err();
        assert!(err.contains("[0.0, 1.0]"), "got: {err}");
    }

    #[test]
    fn cpusetspec_validate_range_rejects_above_one() {
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let spec = CpusetSpec::Range {
            start_frac: 0.5,
            end_frac: 1.5,
        };
        let err = spec.validate(&ctx).unwrap_err();
        assert!(err.contains("[0.0, 1.0]"), "got: {err}");
    }

    #[test]
    fn cpusetspec_validate_overlap_rejects_nan_frac() {
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let spec = CpusetSpec::Overlap {
            index: 0,
            of: 2,
            frac: f64::NAN,
        };
        let err = spec.validate(&ctx).unwrap_err();
        assert!(err.contains("not finite"), "got: {err}");
    }

    #[test]
    fn cpusetspec_validate_overlap_rejects_infinity_frac() {
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let spec = CpusetSpec::Overlap {
            index: 0,
            of: 2,
            frac: f64::INFINITY,
        };
        let err = spec.validate(&ctx).unwrap_err();
        assert!(err.contains("not finite"), "got: {err}");
    }

    #[test]
    fn cpusetspec_validate_overlap_rejects_out_of_range_frac() {
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let spec = CpusetSpec::Overlap {
            index: 0,
            of: 2,
            frac: 1.5,
        };
        let err = spec.validate(&ctx).unwrap_err();
        assert!(err.contains("[0.0, 1.0]"), "got: {err}");
    }

    #[test]
    fn cpusetspec_validate_too_few_cpus_for_partitions() {
        // 1 LLC, 2 cores, 1 thread => 2 total cpus, 2 usable
        let (cg, topo) = make_ctx(1, 2, 1);
        let ctx = ctx_from(&cg, &topo);
        let spec = CpusetSpec::Disjoint { index: 0, of: 5 };
        let err = spec.validate(&ctx).unwrap_err();
        assert!(err.contains("not enough usable CPUs"), "got: {err}");
    }

    #[test]
    fn cpusetspec_validate_exact_in_range_ok() {
        // 1 LLC * 4 cores * 1 thread = CPUs 0..=3 physically present.
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let spec = CpusetSpec::exact([0, 2]);
        assert!(spec.validate(&ctx).is_ok());
    }

    #[test]
    fn cpusetspec_validate_exact_empty_rejected() {
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let spec = CpusetSpec::Exact(BTreeSet::new());
        let err = spec.validate(&ctx).unwrap_err();
        assert!(err.contains("Exact") && err.contains("empty"), "got: {err}");
    }

    #[test]
    fn cpusetspec_validate_exact_out_of_range_rejected() {
        // Topology has CPUs 0..=3; 99 is not physically present.
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let spec = CpusetSpec::exact([99]);
        let err = spec.validate(&ctx).unwrap_err();
        assert!(
            err.contains("99") && err.contains("physical CPU set"),
            "error must name the offending CPU and call it physical: {err}"
        );
    }

    /// Regression: the reserved last CPU (when `total_cpus > 2`,
    /// `usable_cpus` drops the last one to leave the root cgroup a
    /// home) is still PHYSICALLY present. A scheduler author pinning
    /// a cgroup to that CPU for testing is legitimate — validate
    /// must NOT reject on `usable_cpuset` membership. Accepting it
    /// here is the contract that lets isolated-CPU tests compile.
    #[test]
    fn cpusetspec_validate_exact_accepts_reserved_last_cpu() {
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let total = ctx.topo.all_cpus().len();
        assert!(total > 2, "test requires a topology that reserves a CPU");
        let reserved_cpu = total - 1;
        assert!(
            !ctx.topo.usable_cpuset().contains(&reserved_cpu),
            "precondition: reserved CPU {reserved_cpu} must sit outside usable_cpuset",
        );
        assert!(
            ctx.topo.all_cpuset().contains(&reserved_cpu),
            "precondition: reserved CPU {reserved_cpu} must be physically present",
        );
        let spec = CpusetSpec::exact([reserved_cpu]);
        assert!(
            spec.validate(&ctx).is_ok(),
            "validate must accept the reserved CPU — physical presence, not \
             usable-set membership, is the bar",
        );
    }

    /// Regression guard for the HoldSpec pre-loop validation:
    /// execute_steps_with must bail on a vacuous hold BEFORE running
    /// any op. Failure mode without the pre-loop check: ops mutate
    /// cgroup state, then `Duration::from_secs_f64` / `thread::sleep`
    /// hit the downstream panic, leaving orphan cgroups on disk.
    #[test]
    fn execute_steps_with_bails_on_invalid_hold_before_ops() {
        let parent =
            std::env::temp_dir().join(format!("ktstr-hold-validate-{}", std::process::id()));
        // Pre-clean in case a prior failing test left a directory.
        let _ = std::fs::remove_dir_all(&parent);
        std::fs::create_dir_all(&parent).unwrap();
        let cgroups = crate::cgroup::CgroupManager::new(parent.to_str().unwrap());
        let topo = crate::topology::TestTopology::from_vm_topology(
            &crate::vmm::topology::Topology::new(1, 1, 4, 1),
        );
        let ctx = ctx_from(&cgroups, &topo);
        let cg_name = "should_never_exist";
        let step = Step::new(vec![Op::add_cgroup(cg_name)], HoldSpec::Frac(0.0));
        let err = execute_steps_with(&ctx, vec![step], None).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("hold validation") && msg.contains("Frac"),
            "error must cite hold validation + variant: {msg}"
        );
        assert!(
            !parent.join(cg_name).exists(),
            "AddCgroup op ran before hold validation — cgroup dir '{}' exists",
            parent.join(cg_name).display()
        );
        let _ = std::fs::remove_dir_all(&parent);
    }

    /// The SetAffinity dispatcher's `ResolvedAffinity::Random` arm is
    /// guarded by `!from.is_empty() && *count > 0` (see the
    /// `ResolvedAffinity::Random` arm with that same guard in
    /// `apply_ops`). This test mirrors that classification to lock
    /// the contract in place: future refactors that drop either
    /// side of the AND must update this test alongside the dispatch.
    /// The live dispatcher path is partially covered by the
    /// `apply_setup_*` tests via `MockCgroupOps`, but the SetAffinity
    /// arm specifically still requires a running workload handle to
    /// exercise end-to-end and is therefore only covered by its
    /// classification guard here.
    #[test]
    fn set_affinity_random_no_op_conditions() {
        fn should_apply(from: &BTreeSet<usize>, count: usize) -> bool {
            !from.is_empty() && count > 0
        }
        let pool: BTreeSet<usize> = [0, 1, 2].into_iter().collect();
        let empty: BTreeSet<usize> = BTreeSet::new();
        assert!(should_apply(&pool, 2));
        assert!(!should_apply(&pool, 0), "count=0 → no-op");
        assert!(!should_apply(&empty, 1), "empty pool → no-op");
        assert!(!should_apply(&empty, 0), "both zero → no-op");
    }

    #[test]
    fn cpusetspec_validate_llc_out_of_range() {
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let spec = CpusetSpec::Llc(5);
        let err = spec.validate(&ctx).unwrap_err();
        assert!(err.contains("out of range"), "got: {err}");
    }

    #[test]
    fn cpusetspec_validate_valid_disjoint_ok() {
        let (cg, topo) = make_ctx(1, 8, 1);
        let ctx = ctx_from(&cg, &topo);
        let spec = CpusetSpec::Disjoint { index: 1, of: 2 };
        assert!(spec.validate(&ctx).is_ok());
    }

    // -- MemPolicy + cpuset validation tests --

    #[test]
    fn validate_mempolicy_default_always_ok() {
        // 2 NUMA nodes, 2 LLCs (1 per node), 4 cores, 1 thread = 8 CPUs
        let (cg, topo) = make_numa_ctx(2, 2, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let cpuset: BTreeSet<usize> = (0..4).collect();
        assert!(
            validate_mempolicy_cpuset(
                &MemPolicy::Default,
                crate::workload::MpolFlags::NONE,
                &cpuset,
                &ctx,
                "cg_0",
            )
            .is_ok()
        );
    }

    #[test]
    fn validate_mempolicy_local_always_ok() {
        let (cg, topo) = make_numa_ctx(2, 2, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let cpuset: BTreeSet<usize> = (0..4).collect();
        assert!(
            validate_mempolicy_cpuset(
                &MemPolicy::Local,
                crate::workload::MpolFlags::NONE,
                &cpuset,
                &ctx,
                "cg_0",
            )
            .is_ok()
        );
    }

    #[test]
    fn validate_mempolicy_bind_covered() {
        // 2 NUMA nodes, 2 LLCs, 4 cores each = 8 CPUs total
        // LLC 0 (CPUs 0-3) = NUMA 0, LLC 1 (CPUs 4-7) = NUMA 1
        let (cg, topo) = make_numa_ctx(2, 2, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let cpuset: BTreeSet<usize> = (0..8).collect(); // covers both nodes
        let policy = MemPolicy::Bind([0, 1].into_iter().collect());
        assert!(
            validate_mempolicy_cpuset(
                &policy,
                crate::workload::MpolFlags::NONE,
                &cpuset,
                &ctx,
                "cg_0",
            )
            .is_ok()
        );
    }

    #[test]
    fn validate_mempolicy_bind_uncovered() {
        let (cg, topo) = make_numa_ctx(2, 2, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let cpuset: BTreeSet<usize> = (0..4).collect(); // NUMA node 0 only
        let policy = MemPolicy::Bind([1].into_iter().collect()); // node 1 not in cpuset
        let err = validate_mempolicy_cpuset(
            &policy,
            crate::workload::MpolFlags::NONE,
            &cpuset,
            &ctx,
            "cg_bind_test",
        )
        .unwrap_err()
        .to_string();
        // Cgroup name must appear so multi-cgroup scenarios can
        // triage which entry triggered the bail.
        assert!(err.contains("cg_bind_test"), "bail must name cgroup: {err}");
        // Uncovered node (1) and the covering cpuset node (0) must
        // both appear so the reader sees the exact disjoint pair.
        assert!(
            err.contains("[1]"),
            "bail must name uncovered node 1: {err}"
        );
        assert!(err.contains("{0}"), "bail must name cpuset node 0: {err}");
        // Both escape hatches must surface — pin the enumerated
        // `(a)` / `(b)` markers so a regression that collapses them
        // into one option trips this test before a user sees a
        // vague diagnostic.
        assert!(
            err.contains("(a) add .mpol_flags(MpolFlags::STATIC_NODES)"),
            "bail must call out hatch (a) STATIC_NODES opt-in by name: {err}",
        );
        assert!(
            err.contains("(b) widen the cpuset"),
            "bail must call out hatch (b) cpuset widening: {err}",
        );
        assert!(
            err.contains("CpusetSpec::Numa(N)"),
            "bail must name CpusetSpec::Numa(N) as a widening example: {err}",
        );
        assert!(
            err.contains("CpusetSpec::Exact"),
            "bail must name the CpusetSpec::Exact cpuset-widening escape hatch: {err}",
        );
        // The mismatch framing ("cross-socket allocation traffic
        // that is almost certainly unintended") must survive doc
        // edits — it's what makes the bail actionable for an
        // author who wrote the policy assuming the kernel would
        // silently intersect.
        assert!(
            err.contains("cross-socket allocation traffic"),
            "bail must name the cross-socket framing: {err}",
        );
        assert!(
            err.contains("almost certainly unintended"),
            "bail must frame the mismatch as unintended: {err}",
        );
    }

    #[test]
    fn validate_mempolicy_preferred_covered() {
        let (cg, topo) = make_numa_ctx(2, 2, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let cpuset: BTreeSet<usize> = (4..8).collect(); // NUMA node 1
        let policy = MemPolicy::Preferred(1);
        assert!(
            validate_mempolicy_cpuset(
                &policy,
                crate::workload::MpolFlags::NONE,
                &cpuset,
                &ctx,
                "cg_0",
            )
            .is_ok()
        );
    }

    #[test]
    fn validate_mempolicy_preferred_uncovered() {
        let (cg, topo) = make_numa_ctx(2, 2, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let cpuset: BTreeSet<usize> = (0..4).collect(); // NUMA node 0 only
        let policy = MemPolicy::Preferred(1);
        let err = validate_mempolicy_cpuset(
            &policy,
            crate::workload::MpolFlags::NONE,
            &cpuset,
            &ctx,
            "cg_preferred_test",
        )
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("cg_preferred_test"),
            "bail must name cgroup: {err}"
        );
        assert!(
            err.contains("[1]"),
            "bail must name uncovered node 1: {err}"
        );
        assert!(err.contains("{0}"), "bail must name cpuset node 0: {err}");
        assert!(
            err.contains("(a) add .mpol_flags(MpolFlags::STATIC_NODES)"),
            "bail must enumerate hatch (a): {err}",
        );
        assert!(
            err.contains("(b) widen the cpuset"),
            "bail must enumerate hatch (b): {err}",
        );
        assert!(
            err.contains("CpusetSpec::Numa(N)"),
            "bail must cite CpusetSpec::Numa(N) example: {err}",
        );
        assert!(
            err.contains("CpusetSpec::Exact"),
            "bail must name CpusetSpec::Exact widening: {err}",
        );
        assert!(
            err.contains("almost certainly unintended"),
            "bail must frame mismatch as unintended: {err}",
        );
    }

    #[test]
    fn validate_mempolicy_interleave_partial_uncovered() {
        let (cg, topo) = make_numa_ctx(2, 2, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let cpuset: BTreeSet<usize> = (0..4).collect(); // NUMA node 0 only
        let policy = MemPolicy::Interleave([0, 1].into_iter().collect());
        let err = validate_mempolicy_cpuset(
            &policy,
            crate::workload::MpolFlags::NONE,
            &cpuset,
            &ctx,
            "cg_interleave_test",
        )
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("cg_interleave_test"),
            "bail must name cgroup: {err}"
        );
        // Only node 1 is uncovered (node 0 is covered by cpuset); the
        // bail should not list node 0 in the uncovered set.
        assert!(
            err.contains("[1]"),
            "bail must name uncovered node 1: {err}"
        );
        assert!(err.contains("{0}"), "bail must name cpuset node 0: {err}");
        assert!(
            err.contains("(a) add .mpol_flags(MpolFlags::STATIC_NODES)"),
            "bail must enumerate hatch (a): {err}",
        );
        assert!(
            err.contains("(b) widen the cpuset"),
            "bail must enumerate hatch (b): {err}",
        );
        assert!(
            err.contains("CpusetSpec::Numa(N)"),
            "bail must cite CpusetSpec::Numa(N) example: {err}",
        );
        assert!(
            err.contains("CpusetSpec::Exact"),
            "bail must name CpusetSpec::Exact widening: {err}",
        );
    }

    /// `MPOL_F_STATIC_NODES` is the kernel's explicit opt-in for
    /// keeping a mempolicy nodemask absolute across cpuset changes,
    /// so the validator must NOT reject a policy referencing nodes
    /// outside the cpuset when that flag is set — the caller has
    /// signaled intentional cross-node placement.
    #[test]
    fn validate_mempolicy_static_nodes_bypasses_cpuset_check() {
        let (cg, topo) = make_numa_ctx(2, 2, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let cpuset: BTreeSet<usize> = (0..4).collect(); // NUMA node 0 only
        let policy = MemPolicy::Interleave([0, 1].into_iter().collect());
        assert!(
            validate_mempolicy_cpuset(
                &policy,
                crate::workload::MpolFlags::STATIC_NODES,
                &cpuset,
                &ctx,
                "cg_0",
            )
            .is_ok()
        );
    }

    /// `STATIC_NODES | RELATIVE_NODES` is a kernel-rejected
    /// combination — `sanitize_mpol_flags` in `mm/mempolicy.c`
    /// returns `EINVAL` if both bits are set. The validator must
    /// surface this with a named diagnostic before the syscall,
    /// not let it collapse into a generic EINVAL at runtime.
    #[test]
    fn validate_mempolicy_rejects_static_and_relative_conflict() {
        let (cg, topo) = make_numa_ctx(2, 2, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let cpuset: BTreeSet<usize> = (0..4).collect();
        let policy = MemPolicy::Bind([0].into_iter().collect());
        let flags =
            crate::workload::MpolFlags::STATIC_NODES | crate::workload::MpolFlags::RELATIVE_NODES;
        let err = validate_mempolicy_cpuset(&policy, flags, &cpuset, &ctx, "cg_0")
            .expect_err("STATIC_NODES | RELATIVE_NODES must be rejected");
        let rendered = format!("{err:#}");
        assert!(
            rendered.contains("mutually exclusive"),
            "error must name the mutual-exclusion contract; got: {rendered}"
        );
    }

    /// The unknown-bit guard must reject any `MpolFlags` bit that
    /// isn't one of the three documented constants. Without this
    /// test, a regression that accidentally widened `known_bits` or
    /// skipped the guard would land silently — the kernel would
    /// either EINVAL or (worse) interpret the bit as a flag the
    /// validator doesn't model. Uses the `#[cfg(test)]`
    /// `from_bits_for_test` constructor to synthesize a bit pattern
    /// (1 << 10) that no production `MpolFlags` call path can
    /// produce via the named constants.
    #[test]
    fn validate_mempolicy_rejects_unknown_bits() {
        let (cg, topo) = make_numa_ctx(2, 2, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let cpuset: BTreeSet<usize> = (0..8).collect();
        let unknown = crate::workload::MpolFlags::from_bits_for_test(1 << 10);
        let err = validate_mempolicy_cpuset(
            &MemPolicy::Default,
            unknown,
            &cpuset,
            &ctx,
            "cg_unknown_bit",
        )
        .expect_err("unknown bit must bail");
        let s = err.to_string();
        assert!(s.contains("cg_unknown_bit"), "bail must name cgroup: {s}");
        assert!(
            s.contains("unknown bit"),
            "bail must name the unknown-bit contract: {s}"
        );
        assert!(
            s.contains("STATIC_NODES"),
            "bail must enumerate the known bits so the user sees what IS supported: {s}",
        );
    }

    /// `RELATIVE_NODES` treats the policy nodemask as an ordinal
    /// into the cpuset's allowed-nodes set — the kernel performs
    /// the relative→absolute remap internally, so cpuset coverage
    /// in absolute-id terms does not apply. The validator must
    /// bypass the uncovered-node bail path that the default
    /// (no-flag) case enforces; otherwise every RELATIVE_NODES
    /// policy referencing an ordinal beyond the cpuset's first
    /// node would false-positive.
    #[test]
    fn validate_mempolicy_relative_nodes_bypasses_cpuset_check() {
        let (cg, topo) = make_numa_ctx(2, 2, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        // cpuset covers NUMA node 0 only; policy references
        // "node 1" which would fail the absolute-id coverage
        // check in the default path. RELATIVE_NODES must bypass.
        let cpuset: BTreeSet<usize> = (0..4).collect();
        let policy = MemPolicy::Bind([1].into_iter().collect());
        assert!(
            validate_mempolicy_cpuset(
                &policy,
                crate::workload::MpolFlags::RELATIVE_NODES,
                &cpuset,
                &ctx,
                "cg_0",
            )
            .is_ok(),
            "RELATIVE_NODES must bypass the absolute-id cpuset coverage check"
        );
    }

    /// Under `STATIC_NODES` the nodemask is absolute, so the
    /// validator must verify every referenced node actually exists
    /// on the host topology. A policy pinning node 7 on a 2-node
    /// host would fail at syscall time; surfacing it here names
    /// the offender before the failure.
    #[test]
    fn validate_mempolicy_static_nodes_rejects_missing_host_node() {
        let (cg, topo) = make_numa_ctx(2, 2, 4, 1); // host has nodes {0, 1}
        let ctx = ctx_from(&cg, &topo);
        let cpuset: BTreeSet<usize> = (0..8).collect();
        // Reference a node that does not exist on this synthetic host.
        let policy = MemPolicy::Bind([7].into_iter().collect());
        let err = validate_mempolicy_cpuset(
            &policy,
            crate::workload::MpolFlags::STATIC_NODES,
            &cpuset,
            &ctx,
            "cg_0",
        )
        .expect_err("STATIC_NODES policy referencing missing host node must be rejected");
        let rendered = format!("{err:#}");
        assert!(
            rendered.contains("do not exist on this host"),
            "error must name the missing-host-node condition; got: {rendered}"
        );
    }

    #[test]
    fn cgroupdef_mem_policy_builder() {
        let def = CgroupDef::named("test").mem_policy(MemPolicy::Bind([0].into_iter().collect()));
        assert!(matches!(def.works[0].mem_policy, MemPolicy::Bind(_)));
    }

    // ---------------------------------------------------------------
    // apply_setup tests via MockCgroupOps
    // ---------------------------------------------------------------
    //
    // MockCgroupOps is a recording implementor of crate::cgroup::CgroupOps
    // that stores every call it receives in an internal Vec and can be
    // primed to return an error from the next call. This lets
    // apply_setup tests assert on the sequence of cgroup operations
    // without touching /sys/fs/cgroup, so they run as regular userspace
    // unit tests.
    //
    // apply_setup still calls WorkloadHandle::spawn, which forks real
    // worker processes. That's intentional: fork does not require root,
    // and the cgroup.procs write (which would require root in the real
    // kernel) is abstracted behind the mock. The test subject is the
    // orchestration logic — "for each def, call create_cgroup, then
    // set_cpuset if spec.is_some(), then move_tasks after spawn".
    //
    // Parallel-nextest behavior: verified non-flaky over repeated
    // `cargo nextest run --lib -E 'test(apply_setup)' --test-threads 8`
    // invocations and back-to-back full-suite runs. Each `MockCgroupOps`
    // owns its own `Mutex<Vec<CgroupCall>>`, so cross-test recording
    // cannot contend. `apply_setup` does call `WorkloadHandle::start`
    // (see top of this file) — workers wake, run briefly, and are then
    // SIGKILL'd when the owning `WorkloadHandle` drops via
    // `cleanup_state(&mut state)` / `state.handles.clear()` at the tail
    // of each test. No test assertion depends on worker output, only
    // on mock-recorded cgroup calls, so worker timing is not
    // observable. Fd footprint is 4 pipes × `workers()` per test — 8
    // fds for the 2-worker tests, well inside any RLIMIT_NOFILE the
    // harness sets.

    use crate::cgroup::CgroupOps;
    use std::path::Path;
    use std::sync::Mutex;

    /// A call captured by MockCgroupOps during apply_setup execution.
    /// Equality-comparable so tests can assert on the exact sequence.
    /// `MoveTasks` stores the pid count rather than the full `pids` Vec
    /// because PIDs are unpredictable between runs.
    #[derive(Debug, Clone, PartialEq, Eq)]
    enum CgroupCall {
        Setup(BTreeSet<crate::cgroup::Controller>),
        CreateCgroup(String),
        RemoveCgroup(String),
        SetCpuset(String, BTreeSet<usize>),
        ClearCpuset(String),
        SetCpusetMems(String, BTreeSet<usize>),
        #[allow(dead_code)] // Emitted by CgroupOps::clear_cpuset_mems; no test asserts on it yet.
        ClearCpusetMems(String),
        // (name, quota_us, period_us); quota=None means "max".
        SetCpuMax(String, Option<u64>, u64),
        SetCpuWeight(String, u32),
        SetMemoryMax(String, Option<u64>),
        SetMemoryHigh(String, Option<u64>),
        SetMemoryLow(String, Option<u64>),
        SetMemorySwapMax(String, Option<u64>),
        SetIoWeight(String, u16),
        SetFreeze(String, bool),
        SetPidsMax(String, Option<u64>),
        MoveTask(String, libc::pid_t),
        MoveTasks(String, usize), // (cgroup name, number of pids)
        ClearSubtreeControl(String),
        DrainTasks(String),
        CleanupAll,
    }

    struct MockCgroupOps {
        parent: std::path::PathBuf,
        calls: Mutex<Vec<CgroupCall>>,
        // When Some, the Nth call (indexed from 0 at insertion time)
        // returns an error and decrements; otherwise all calls return Ok.
        fail_at: Mutex<Option<(usize, String)>>,
    }

    impl MockCgroupOps {
        fn new() -> Self {
            Self {
                parent: std::path::PathBuf::from("/mock/cgroup"),
                calls: Mutex::new(Vec::new()),
                fail_at: Mutex::new(None),
            }
        }

        /// Return an error from the Nth call (0-indexed from now) with
        /// the given message. Used by tests that check error
        /// propagation through apply_setup.
        fn fail_call_at(&self, index: usize, message: &str) {
            *self.fail_at.lock().unwrap() = Some((index, message.to_string()));
        }

        fn calls(&self) -> Vec<CgroupCall> {
            self.calls.lock().unwrap().clone()
        }

        /// Record a call and decide whether to return Ok or inject an
        /// error. Centralizes the fail_at logic so every trait method
        /// gets it for free.
        fn record(&self, call: CgroupCall) -> Result<()> {
            let mut calls = self.calls.lock().unwrap();
            let current_index = calls.len();
            calls.push(call);
            drop(calls);
            let mut fail = self.fail_at.lock().unwrap();
            if let Some((index, ref message)) = *fail
                && current_index == index
            {
                let err_msg = message.clone();
                *fail = None;
                return Err(anyhow::anyhow!(err_msg));
            }
            Ok(())
        }
    }

    impl CgroupOps for MockCgroupOps {
        fn parent_path(&self) -> &Path {
            &self.parent
        }
        fn setup(&self, controllers: &BTreeSet<crate::cgroup::Controller>) -> Result<()> {
            self.record(CgroupCall::Setup(controllers.clone()))
        }
        fn create_cgroup(&self, name: &str) -> Result<()> {
            self.record(CgroupCall::CreateCgroup(name.to_string()))
        }
        fn remove_cgroup(&self, name: &str) -> Result<()> {
            self.record(CgroupCall::RemoveCgroup(name.to_string()))
        }
        fn set_cpuset(&self, name: &str, cpus: &BTreeSet<usize>) -> Result<()> {
            self.record(CgroupCall::SetCpuset(name.to_string(), cpus.clone()))
        }
        fn clear_cpuset(&self, name: &str) -> Result<()> {
            self.record(CgroupCall::ClearCpuset(name.to_string()))
        }
        fn set_cpuset_mems(&self, name: &str, nodes: &BTreeSet<usize>) -> Result<()> {
            self.record(CgroupCall::SetCpusetMems(name.to_string(), nodes.clone()))
        }
        fn clear_cpuset_mems(&self, name: &str) -> Result<()> {
            self.record(CgroupCall::ClearCpusetMems(name.to_string()))
        }
        fn set_cpu_max(&self, name: &str, quota_us: Option<u64>, period_us: u64) -> Result<()> {
            self.record(CgroupCall::SetCpuMax(name.to_string(), quota_us, period_us))
        }
        fn set_cpu_weight(&self, name: &str, weight: u32) -> Result<()> {
            self.record(CgroupCall::SetCpuWeight(name.to_string(), weight))
        }
        fn set_memory_max(&self, name: &str, bytes: Option<u64>) -> Result<()> {
            self.record(CgroupCall::SetMemoryMax(name.to_string(), bytes))
        }
        fn set_memory_high(&self, name: &str, bytes: Option<u64>) -> Result<()> {
            self.record(CgroupCall::SetMemoryHigh(name.to_string(), bytes))
        }
        fn set_memory_low(&self, name: &str, bytes: Option<u64>) -> Result<()> {
            self.record(CgroupCall::SetMemoryLow(name.to_string(), bytes))
        }
        fn set_io_weight(&self, name: &str, weight: u16) -> Result<()> {
            self.record(CgroupCall::SetIoWeight(name.to_string(), weight))
        }
        fn set_freeze(&self, name: &str, frozen: bool) -> Result<()> {
            self.record(CgroupCall::SetFreeze(name.to_string(), frozen))
        }
        fn set_pids_max(&self, name: &str, max: Option<u64>) -> Result<()> {
            self.record(CgroupCall::SetPidsMax(name.to_string(), max))
        }
        fn set_memory_swap_max(&self, name: &str, bytes: Option<u64>) -> Result<()> {
            self.record(CgroupCall::SetMemorySwapMax(name.to_string(), bytes))
        }
        fn move_task(&self, name: &str, pid: libc::pid_t) -> Result<()> {
            self.record(CgroupCall::MoveTask(name.to_string(), pid))
        }
        fn move_tasks(&self, name: &str, pids: &[libc::pid_t]) -> Result<()> {
            self.record(CgroupCall::MoveTasks(name.to_string(), pids.len()))
        }
        fn clear_subtree_control(&self, name: &str) -> Result<()> {
            self.record(CgroupCall::ClearSubtreeControl(name.to_string()))
        }
        fn drain_tasks(&self, name: &str) -> Result<()> {
            self.record(CgroupCall::DrainTasks(name.to_string()))
        }
        fn cleanup_all(&self) -> Result<()> {
            self.record(CgroupCall::CleanupAll)
        }
    }

    /// Build a Ctx backed by MockCgroupOps so apply_setup can be driven
    /// without cgroup filesystem access. Topology fixed at 1 NUMA /
    /// 1 LLC / 4 cores / 1 thread = 4 CPUs — enough range to cover
    /// per-cpu cpuset assertions without making the mock brittle.
    fn mock_ctx<'a>(mock: &'a MockCgroupOps, topo: &'a crate::topology::TestTopology) -> Ctx<'a> {
        Ctx {
            cgroups: mock,
            topo,
            duration: Duration::from_secs(1),
            workers_per_cgroup: 1,
            sched_pid: None,
            settle: Duration::ZERO,
            work_type_override: None,
            assert: crate::assert::Assert::default_checks(),
            wait_for_map_write: false,
            current_step: std::sync::Arc::new(std::sync::atomic::AtomicU16::new(0)),
        }
    }

    fn mock_topo() -> crate::topology::TestTopology {
        crate::topology::TestTopology::from_vm_topology(&crate::vmm::topology::Topology::new(
            1, 1, 4, 1,
        ))
    }

    /// Drop workload + payload handles inside state so apply_setup
    /// tests don't leak worker or payload processes. Synthetic
    /// `WorkloadHandle`s SIGKILL their workers on Drop, so a
    /// `handles.clear()` is enough; `PayloadHandle` likewise
    /// SIGKILLs its child on Drop (with an eprintln warning about
    /// metrics not being recorded — acceptable in the test path
    /// where metrics aren't what's under test). Calling
    /// `drain_all_payload_handles` routes through `.kill()` so the
    /// metric-emission branch runs and the test doesn't trigger
    /// the Drop-warning banner on stderr.
    fn cleanup_state(state: &mut StepState<'_>) {
        state.handles.clear();
        drain_all_payload_handles(&mut state.payload_handles);
    }

    /// Test helper: call `apply_setup` against a step-local-only
    /// [`ScenarioState`]. Constructs a throwaway backdrop state
    /// pointing at the same mock-cgroups handle `state` uses so
    /// tests that only exercise step-local semantics stay terse.
    fn apply_setup_test<'a>(
        ctx: &'a Ctx<'a>,
        state: &mut StepState<'a>,
        defs: &[CgroupDef],
    ) -> Result<()> {
        let mut backdrop = BackdropState::empty(ctx);
        let mut scenario = ScenarioState::new(state, &mut backdrop);
        apply_setup(ctx, &mut scenario, defs)
    }

    /// Test helper: call `apply_ops` against a step-local-only
    /// [`ScenarioState`]. Mirrors [`apply_setup_test`] for ops.
    fn apply_ops_test<'a>(ctx: &'a Ctx<'a>, state: &mut StepState<'a>, ops: &[Op]) -> Result<()> {
        let mut backdrop = BackdropState::empty(ctx);
        let mut scenario = ScenarioState::new(state, &mut backdrop);
        apply_ops(ctx, &mut scenario, ops)
    }

    #[test]
    fn apply_setup_empty_defs_is_noop() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        apply_setup_test(&ctx, &mut state, &[]).unwrap();
        assert!(
            mock.calls().is_empty(),
            "apply_setup on zero defs must not call any cgroup op, got: {:?}",
            mock.calls()
        );
        cleanup_state(&mut state);
    }

    #[test]
    fn apply_setup_creates_cgroup_per_def() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        let defs = vec![
            CgroupDef::named("cg_a").workers(1),
            CgroupDef::named("cg_b").workers(1),
        ];
        apply_setup_test(&ctx, &mut state, &defs).unwrap();
        let calls = mock.calls();
        let creates: Vec<&CgroupCall> = calls
            .iter()
            .filter(|c| matches!(c, CgroupCall::CreateCgroup(_)))
            .collect();
        assert_eq!(
            creates,
            vec![
                &CgroupCall::CreateCgroup("cg_a".to_string()),
                &CgroupCall::CreateCgroup("cg_b".to_string()),
            ],
            "one create_cgroup call per def, in order"
        );
        cleanup_state(&mut state);
    }

    #[test]
    fn apply_setup_sets_cpuset_when_spec_present() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        let cpus: BTreeSet<usize> = [0, 1].into_iter().collect();
        let defs = vec![
            CgroupDef::named("cg_0")
                .cpuset(CpusetSpec::Exact(cpus.clone()))
                .workers(1),
        ];
        apply_setup_test(&ctx, &mut state, &defs).unwrap();
        let calls = mock.calls();
        assert!(
            calls.contains(&CgroupCall::SetCpuset("cg_0".to_string(), cpus.clone())),
            "set_cpuset must be called with exactly the resolved cpu set, got: {calls:?}"
        );
        // state.cpusets should mirror the set so later SetAffinity /
        // MemPolicy checks see the resolved cpuset.
        assert_eq!(
            state.cpusets.get("cg_0"),
            Some(&cpus),
            "state.cpusets must record the resolved set"
        );
        cleanup_state(&mut state);
    }

    #[test]
    fn apply_setup_skips_cpuset_when_none() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        // cpuset: None → inherit parent's set, apply_setup must not
        // emit a set_cpuset call.
        let defs = vec![CgroupDef::named("cg_inherit").workers(1)];
        apply_setup_test(&ctx, &mut state, &defs).unwrap();
        let calls = mock.calls();
        let has_set_cpuset = calls
            .iter()
            .any(|c| matches!(c, CgroupCall::SetCpuset(_, _)));
        assert!(
            !has_set_cpuset,
            "no set_cpuset should be emitted when CgroupDef.cpuset is None, got: {calls:?}"
        );
        assert!(
            state.cpusets.is_empty(),
            "state.cpusets should stay empty when no CpusetSpec was resolved"
        );
        cleanup_state(&mut state);
    }

    #[test]
    fn apply_setup_moves_spawned_tasks_into_cgroup() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        // workers(2): after spawn, apply_setup must call move_tasks
        // with 2 pids.
        let defs = vec![CgroupDef::named("cg_move").workers(2)];
        apply_setup_test(&ctx, &mut state, &defs).unwrap();
        let calls = mock.calls();
        assert!(
            calls.contains(&CgroupCall::MoveTasks("cg_move".to_string(), 2)),
            "move_tasks must be called with the 2 spawned worker pids, got: {calls:?}"
        );
        // Ordering invariant: move_tasks follows create_cgroup, and
        // set_cpuset (when present) follows create_cgroup but precedes
        // move_tasks. Here with no cpuset, just assert create precedes
        // move.
        let create_idx = calls
            .iter()
            .position(|c| matches!(c, CgroupCall::CreateCgroup(n) if n == "cg_move"))
            .expect("create_cgroup for cg_move");
        let move_idx = calls
            .iter()
            .position(|c| matches!(c, CgroupCall::MoveTasks(n, _) if n == "cg_move"))
            .expect("move_tasks for cg_move");
        assert!(
            create_idx < move_idx,
            "create_cgroup must precede move_tasks for the same cgroup: {calls:?}"
        );
        cleanup_state(&mut state);
    }

    #[test]
    fn apply_setup_sets_cpuset_before_move_tasks() {
        // Ordering invariant: for a cgroup with both a cpuset spec and
        // workers, `set_cpuset` MUST precede `move_tasks` so the
        // kernel enforces the cpu mask on the first scheduling
        // decision after the task enters the cgroup. Moving first
        // would let tasks briefly run on cpus outside the intended
        // set.
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        let cpus: BTreeSet<usize> = [0, 1].into_iter().collect();
        let defs = vec![
            CgroupDef::named("cg_ordered")
                .cpuset(CpusetSpec::Exact(cpus.clone()))
                .workers(2),
        ];
        apply_setup_test(&ctx, &mut state, &defs).unwrap();
        let calls = mock.calls();
        let set_idx = calls
            .iter()
            .position(|c| matches!(c, CgroupCall::SetCpuset(n, _) if n == "cg_ordered"))
            .expect("set_cpuset for cg_ordered");
        let move_idx = calls
            .iter()
            .position(|c| matches!(c, CgroupCall::MoveTasks(n, _) if n == "cg_ordered"))
            .expect("move_tasks for cg_ordered");
        assert!(
            set_idx < move_idx,
            "set_cpuset must precede move_tasks for the same cgroup: {calls:?}"
        );
        cleanup_state(&mut state);
    }

    #[test]
    fn apply_setup_bails_on_invalid_cpuset_spec() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        // Llc(99) on a 1-LLC topology is out of range; CpusetSpec::validate
        // bails after create_cgroup runs but before set_cpuset / move_tasks
        // fire.
        let defs = vec![CgroupDef::named("cg_bad").cpuset(CpusetSpec::Llc(99))];
        let err = apply_setup_test(&ctx, &mut state, &defs).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("CpusetSpec validation failed"),
            "expected validation error, got: {msg}"
        );
        // create_cgroup runs before cpuset validation — record that
        // here so future refactors notice if the order flips.
        let calls = mock.calls();
        assert_eq!(
            calls,
            vec![CgroupCall::CreateCgroup("cg_bad".to_string())],
            "current ordering: create_cgroup first, then cpuset validation"
        );
        cleanup_state(&mut state);
    }

    #[test]
    fn apply_setup_propagates_set_cpuset_error() {
        let mock = MockCgroupOps::new();
        // Inject failure at call index 1. Index 0 is the create_cgroup
        // emitted before the cpuset write; index 1 is the set_cpuset
        // itself.
        mock.fail_call_at(1, "set_cpuset kernel EBUSY");
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        let cpus: BTreeSet<usize> = [0, 1].into_iter().collect();
        let defs = vec![
            CgroupDef::named("cg_setfail")
                .cpuset(CpusetSpec::Exact(cpus))
                .workers(1),
        ];
        let err = apply_setup_test(&ctx, &mut state, &defs).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("set_cpuset kernel EBUSY"),
            "set_cpuset error must propagate, got: {msg}"
        );
        // Check the failure halted apply_setup before reaching spawn:
        // no MoveTasks call should have been recorded.
        let calls = mock.calls();
        let has_move = calls
            .iter()
            .any(|c| matches!(c, CgroupCall::MoveTasks(_, _)));
        assert!(
            !has_move,
            "no move_tasks call should follow a failed set_cpuset, got: {calls:?}"
        );
        cleanup_state(&mut state);
    }

    #[test]
    fn apply_setup_validates_mempolicy_against_cpuset() {
        let mock = MockCgroupOps::new();
        // 2 NUMA / 2 LLCs (1 per node) / 4 cores / 1 thread = 8 CPUs
        let topo = crate::topology::TestTopology::from_vm_topology(
            &crate::vmm::topology::Topology::new(2, 2, 4, 1),
        );
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        // cpuset = NUMA node 0 only (CPUs 0-3); mem_policy binds to
        // node 1 — must bail, no downstream spawn.
        let cpus: BTreeSet<usize> = (0..4).collect();
        let bind: BTreeSet<usize> = [1].into_iter().collect();
        let defs = vec![
            CgroupDef::named("cg_memfail")
                .cpuset(CpusetSpec::Exact(cpus))
                .mem_policy(MemPolicy::Bind(bind))
                .workers(1),
        ];
        let err = apply_setup_test(&ctx, &mut state, &defs).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("cg_memfail"),
            "error must name the bad cgroup, got: {msg}"
        );
        // set_cpuset was called before the mempolicy check (order
        // documented by apply_setup). Assert move_tasks did not run —
        // that would mean the pre-validation guard failed.
        let calls = mock.calls();
        let has_move = calls
            .iter()
            .any(|c| matches!(c, CgroupCall::MoveTasks(_, _)));
        assert!(
            !has_move,
            "mempolicy validation must bail before spawn, got: {calls:?}"
        );
        cleanup_state(&mut state);
    }

    // -- CgroupDef::workload --

    /// Default CgroupDef has no payload attached — every test that
    /// doesn't opt in stays Payload-free so the synthetic-workload
    /// path is unaffected.
    #[test]
    fn cgroup_def_default_payload_is_none() {
        let def = CgroupDef::named("cg_0");
        assert!(def.payload.is_none());
    }

    /// The `.workload(&FIO)` builder stores the reference on the
    /// CgroupDef so apply_setup can spawn it. Because `Payload` is
    /// `Copy`, the builder preserves identity through pointer
    /// equality after conversion to `&'static` refs.
    #[test]
    fn cgroup_def_workload_stores_payload() {
        use crate::test_support::{OutputFormat, Payload, PayloadKind};
        static FIO: Payload = Payload {
            name: "fio",
            kind: PayloadKind::Binary("fio"),
            output: OutputFormat::Json,
            default_args: &[],
            default_checks: &[],
            metrics: &[],
            include_files: &[],
            uses_parent_pgrp: false,
            known_flags: None,
            metric_bounds: None,
        };
        let def = CgroupDef::named("cg_0").workload(&FIO);
        let p = def.payload.expect("workload was attached");
        assert_eq!(p.name, "fio");
        assert!(!p.is_scheduler());
    }

    /// Scheduler-kind payloads are rejected at builder time — the
    /// `workload` slot is exclusively for userspace binaries that
    /// run *under* a scheduler, not for scheduler placement itself.
    #[test]
    #[should_panic(expected = "CgroupDef::workload called with a scheduler-kind Payload")]
    fn cgroup_def_workload_rejects_scheduler_kind_payload() {
        use crate::test_support::Payload;
        let _ = CgroupDef::named("cg_0").workload(&Payload::KERNEL_DEFAULT);
    }

    /// The drain helper kills + removes entries whose cgroup name
    /// matches the target. Non-matching entries stay in the vector
    /// so subsequent step teardown (via `collect_step`) or scenario
    /// end (via `collect_backdrop`) kills them in turn.
    #[test]
    fn drain_payload_handles_for_cgroup_removes_matching_only() {
        use crate::cgroup::CgroupManager;
        use crate::scenario::payload_run::PayloadRun;
        use crate::test_support::{OutputFormat, Payload, PayloadKind};
        use crate::topology::TestTopology;

        static TRUE_BIN: Payload = Payload {
            name: "true_bin",
            kind: PayloadKind::Binary("/bin/true"),
            output: OutputFormat::ExitCode,
            default_args: &[],
            default_checks: &[],
            metrics: &[],
            include_files: &[],
            uses_parent_pgrp: false,
            known_flags: None,
            metric_bounds: None,
        };

        let cgroups = CgroupManager::new("/nonexistent");
        let topo = TestTopology::synthetic(4, 1);
        let ctx = crate::scenario::Ctx::builder(&cgroups, &topo).build();

        let h_a = PayloadRun::new(&ctx, &TRUE_BIN)
            .spawn()
            .expect("spawn /bin/true for cg_a");
        let h_b = PayloadRun::new(&ctx, &TRUE_BIN)
            .spawn()
            .expect("spawn /bin/true for cg_b");

        let mut handles = vec![
            PayloadEntry {
                cgroup: "cg_a".to_string(),
                source: PayloadSource::CgroupDefWorkload,
                handle: h_a,
            },
            PayloadEntry {
                cgroup: "cg_b".to_string(),
                source: PayloadSource::CgroupDefWorkload,
                handle: h_b,
            },
        ];
        drain_payload_handles_for_cgroup(&mut handles, "cg_a");

        assert_eq!(handles.len(), 1);
        assert_eq!(handles[0].cgroup, "cg_b");

        drain_all_payload_handles(&mut handles);
        assert!(handles.is_empty());
    }

    // -- Step::with_payload + Op::RunPayload/WaitPayload/KillPayload --

    /// Step::with_payload emits a step whose ops consist of a single
    /// Op::RunPayload carrying the supplied payload. Hold passes
    /// through unchanged.
    #[test]
    fn step_with_payload_emits_runpayload_op() {
        use crate::test_support::{OutputFormat, Payload, PayloadKind};
        static FIO: Payload = Payload {
            name: "fio",
            kind: PayloadKind::Binary("fio"),
            output: OutputFormat::Json,
            default_args: &[],
            default_checks: &[],
            metrics: &[],
            include_files: &[],
            uses_parent_pgrp: false,
            known_flags: None,
            metric_bounds: None,
        };
        let step = Step::with_payload(&FIO, HoldSpec::fixed(Duration::from_millis(50)));
        assert_eq!(step.ops.len(), 1);
        match &step.ops[0] {
            Op::RunPayload {
                payload,
                args,
                cgroup,
            } => {
                assert_eq!(payload.name, "fio");
                assert!(args.is_empty());
                assert!(cgroup.is_none());
            }
            other => panic!("expected RunPayload, got {other:?}"),
        }
        assert!(matches!(step.hold, HoldSpec::Fixed(d) if d == Duration::from_millis(50)));
        assert!(matches!(&step.setup, Setup::Defs(d) if d.is_empty()));
    }

    /// Op convenience constructors — `run_payload`, `wait_payload`,
    /// `kill_payload`, `run_payload_in_cgroup` — build the expected
    /// enum shapes with the right field contents.
    #[test]
    fn op_payload_constructors_produce_expected_variants() {
        use crate::test_support::{OutputFormat, Payload, PayloadKind};
        static FIO: Payload = Payload {
            name: "fio",
            kind: PayloadKind::Binary("fio"),
            output: OutputFormat::Json,
            default_args: &[],
            default_checks: &[],
            metrics: &[],
            include_files: &[],
            uses_parent_pgrp: false,
            known_flags: None,
            metric_bounds: None,
        };

        let op = Op::run_payload(&FIO, vec!["--warmup".into()]);
        match op {
            Op::RunPayload {
                payload,
                args,
                cgroup,
            } => {
                assert_eq!(payload.name, "fio");
                assert_eq!(args, vec!["--warmup".to_string()]);
                assert!(cgroup.is_none());
            }
            other => panic!("expected RunPayload, got {other:?}"),
        }

        let op = Op::run_payload_in_cgroup(&FIO, vec![], "cg_0");
        match op {
            Op::RunPayload {
                payload,
                args,
                cgroup,
            } => {
                assert_eq!(payload.name, "fio");
                assert!(args.is_empty());
                assert_eq!(cgroup.as_deref(), Some("cg_0"));
            }
            other => panic!("expected RunPayload, got {other:?}"),
        }

        let op = Op::wait_payload("fio");
        assert!(matches!(
            op,
            Op::WaitPayload { ref name, ref cgroup } if name.as_ref() == "fio" && cgroup.is_none(),
        ));

        let op = Op::kill_payload("fio");
        assert!(matches!(
            op,
            Op::KillPayload { ref name, ref cgroup } if name.as_ref() == "fio" && cgroup.is_none(),
        ));

        let op = Op::wait_payload_in_cgroup("fio", "cg_0");
        assert!(matches!(
            op,
            Op::WaitPayload { ref name, cgroup: Some(ref c) } if name.as_ref() == "fio" && c.as_ref() == "cg_0",
        ));

        let op = Op::kill_payload_in_cgroup("fio", "cg_0");
        assert!(matches!(
            op,
            Op::KillPayload { ref name, cgroup: Some(ref c) } if name.as_ref() == "fio" && c.as_ref() == "cg_0",
        ));
    }

    /// Op::RunPayload rejects scheduler-kind payloads at apply time
    /// with an actionable error message. The existing CgroupDef
    /// path panics at builder time; the Op path runs at scenario
    /// time and must bail instead of panicking so one bad step in
    /// a sequence doesn't crash the harness.
    #[test]
    fn apply_ops_runpayload_rejects_scheduler_kind() {
        use crate::test_support::Payload;
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        let ops = vec![Op::RunPayload {
            payload: &Payload::KERNEL_DEFAULT,
            args: vec![],
            cgroup: None,
        }];
        let err = apply_ops_test(&ctx, &mut state, &ops).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("scheduler-kind Payload") && msg.contains("kernel_default"),
            "error must name the scheduler-kind reason AND the payload name, got: {msg}"
        );
        assert!(
            state.payload_handles.is_empty(),
            "no handle should be stored when RunPayload rejects the kind"
        );
    }

    /// Op::WaitPayload with no matching handle surfaces a descriptive
    /// error rather than silently no-op'ing. Ditto KillPayload. A
    /// silent no-op would let test authors wait for ghosts and pass
    /// scenarios that never ran what they claim.
    #[test]
    fn apply_ops_wait_unknown_payload_bails() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        let err = apply_ops_test(
            &ctx,
            &mut state,
            &[Op::WaitPayload {
                name: "ghost".into(),
                cgroup: None,
            }],
        )
        .unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("no running payload named 'ghost'"),
            "error must name the missing payload, got: {msg}"
        );
    }

    #[test]
    fn apply_ops_kill_unknown_payload_bails() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        let err = apply_ops_test(
            &ctx,
            &mut state,
            &[Op::KillPayload {
                name: "ghost".into(),
                cgroup: None,
            }],
        )
        .unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("no running payload named 'ghost'"),
            "error must name the missing payload, got: {msg}"
        );
    }

    // -- Scheduler-lifecycle Op dispatch error-path tests --
    //
    // The 4 scheduler-lifecycle Op variants (AttachScheduler /
    // DetachScheduler / RestartScheduler / ReplaceScheduler) dispatch
    // through `dispatch_attach_scheduler` / `dispatch_detach_scheduler`
    // / `dispatch_restart_scheduler` / `dispatch_replace_scheduler`
    // helpers in this module. Unit tests cover the error paths that
    // don't require a real running scheduler: AttachScheduler with a
    // missing staged binary, and Detach / Restart / Replace with no
    // scheduler attached (SCHED_PID == 0). The success paths require
    // a real spawn + libbpf attach, which the e2e VM integration
    // suite exercises with scx-ktstr as the staged target.
    //
    // Every error path emits an actionable message naming the op
    // AND the specific failure mode — copy-paste regression across
    // the 4 arms surfaces as a distinct substring per arm.

    /// Op::AttachScheduler against a Scheduler whose `name` doesn't
    /// resolve to a staged binary file must bail with an actionable
    /// error naming the missing path + suggesting the staging
    /// pipeline check. EEVDF.name = "eevdf" — its `/staging/schedulers/
    /// eevdf/scheduler` path does NOT exist in the test environment
    /// (test harness has no initramfs mounted), so the inline
    /// existence probe in `spawn_scheduler_from_paths` returns
    /// `(None, None)` and the dispatch arm bails.
    #[test]
    fn apply_ops_attach_scheduler_bails_when_staged_binary_missing() {
        static SCHED: crate::test_support::Scheduler = crate::test_support::Scheduler::EEVDF;
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        let err = apply_ops_test(&ctx, &mut state, &[Op::attach_scheduler(&SCHED)]).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("Op::AttachScheduler"),
            "error must name the op (catches copy-paste regression across the 4 arms): {msg}"
        );
        assert!(
            msg.contains("staging") || msg.contains("staged"),
            "error must point at the staging pipeline so the operator knows where to look: {msg}"
        );
        assert!(
            msg.contains("eevdf"),
            "error must include the scheduler name so the operator can identify which entry: {msg}"
        );
    }

    /// Op::DetachScheduler with no scheduler currently attached
    /// (SCHED_PID == 0 sentinel) bails with an actionable error.
    /// Distinct from AttachScheduler's pin because each arm's error
    /// message must be per-variant.
    #[test]
    fn apply_ops_detach_scheduler_bails_when_no_scheduler_attached() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        // Test environment has no scheduler spawned — SCHED_PID is 0.
        let err = apply_ops_test(&ctx, &mut state, &[Op::detach_scheduler()]).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("Op::DetachScheduler"),
            "error must name the op: {msg}"
        );
        assert!(
            msg.contains("no scheduler attached") || msg.contains("SCHED_PID"),
            "error must name the no-scheduler failure mode: {msg}"
        );
    }

    /// Op::RestartScheduler with no scheduler attached bails with
    /// an actionable error.
    #[test]
    fn apply_ops_restart_scheduler_bails_when_no_scheduler_attached() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        let err = apply_ops_test(&ctx, &mut state, &[Op::restart_scheduler()]).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("Op::RestartScheduler"),
            "error must name the op: {msg}"
        );
        assert!(
            msg.contains("no scheduler attached") || msg.contains("SCHED_PID"),
            "error must name the no-scheduler failure mode: {msg}"
        );
    }

    /// Op::ReplaceScheduler with no scheduler attached bails BEFORE
    /// attempting to spawn the replacement — the detach phase fails
    /// fast on the SCHED_PID == 0 check so the operator sees the
    /// "no scheduler to replace" error rather than a confusing
    /// post-spawn diagnostic.
    #[test]
    fn apply_ops_replace_scheduler_bails_when_no_scheduler_attached() {
        static SCHED: crate::test_support::Scheduler = crate::test_support::Scheduler::EEVDF;
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        let err = apply_ops_test(&ctx, &mut state, &[Op::replace_scheduler(&SCHED)]).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("Op::ReplaceScheduler"),
            "error must name the op: {msg}"
        );
        assert!(
            msg.contains("no scheduler attached") || msg.contains("SCHED_PID"),
            "error must name the no-scheduler failure mode (detach phase fails fast): {msg}"
        );
    }

    /// `staged_scheduler_log_path` produces collision-free per-name
    /// plus per-seq paths so successive Op::AttachScheduler or
    /// Op::ReplaceScheduler dispatches with the SAME staged name
    /// don't overwrite each other's logs. Pins the
    /// `/tmp/sched_<name>_<seq>.log` scheme against a regression
    /// that drops either the per-name keying or the per-seq seq
    /// suffix.
    #[test]
    fn staged_scheduler_log_path_is_per_name_and_seq_keyed() {
        let a1 = staged_scheduler_log_path("scx_mitosis_a");
        let a2 = staged_scheduler_log_path("scx_mitosis_a");
        let b1 = staged_scheduler_log_path("scx_mitosis_b");
        // Same name on consecutive calls must produce distinct
        // paths via the seq suffix — protects against repeated
        // Op::ReplaceScheduler with the same staged name losing
        // the first spawn's failure-dump payload.
        assert_ne!(a1, a2, "same-name consecutive calls must differ via seq");
        // Different names must also differ — name keying defends
        // against parallel dispatch with distinct staged entries.
        assert_ne!(a1, b1, "different names must produce distinct paths");
        // Path shape: prefix + name + underscore + numeric seq +
        // .log. Asserts the seq suffix is purely numeric.
        assert!(
            a1.starts_with("/tmp/sched_scx_mitosis_a_"),
            "missing name + underscore prefix: {a1}"
        );
        assert!(a1.ends_with(".log"), "missing .log extension: {a1}");
        let seq_part = a1
            .strip_prefix("/tmp/sched_scx_mitosis_a_")
            .unwrap()
            .strip_suffix(".log")
            .unwrap();
        assert!(
            seq_part.chars().all(|c| c.is_ascii_digit()),
            "seq suffix must be all digits: {seq_part:?}"
        );
    }

    /// End-to-end on a real payload binary: Op::RunPayload spawns
    /// a long-running `/bin/sleep`, Op::KillPayload matches by
    /// payload.name and consumes the handle. The handle should
    /// disappear from state.payload_handles so later teardown
    /// drains don't double-consume.
    #[test]
    fn apply_ops_run_then_kill_consumes_handle() {
        use crate::test_support::{OutputFormat, Payload, PayloadKind};
        static SLEEP: Payload = Payload {
            // Name distinct from binary so the payload_name lookup
            // path is exercised against a non-basename key.
            name: "sleeper",
            kind: PayloadKind::Binary("/bin/sleep"),
            output: OutputFormat::ExitCode,
            default_args: &[],
            default_checks: &[],
            metrics: &[],
            include_files: &[],
            uses_parent_pgrp: false,
            known_flags: None,
            metric_bounds: None,
        };

        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        apply_ops_test(
            &ctx,
            &mut state,
            &[Op::run_payload(&SLEEP, vec!["3600".into()])],
        )
        .expect("spawn /bin/sleep");
        assert_eq!(state.payload_handles.len(), 1, "one payload is live");
        assert_eq!(state.payload_handles[0].handle.payload_name(), "sleeper");

        apply_ops_test(&ctx, &mut state, &[Op::kill_payload("sleeper")])
            .expect("kill the live payload");
        assert!(
            state.payload_handles.is_empty(),
            "handle must be consumed by KillPayload"
        );
    }

    /// Spawning a second payload with the same name while the first
    /// is still live is a caller bug — the `WaitPayload`/
    /// `KillPayload` lookup would hit the first match and leave the
    /// second leaked. Reject at RunPayload time.
    #[test]
    fn apply_ops_run_duplicate_payload_name_bails() {
        use crate::test_support::{OutputFormat, Payload, PayloadKind};
        static SLEEP: Payload = Payload {
            name: "sleeper",
            kind: PayloadKind::Binary("/bin/sleep"),
            output: OutputFormat::ExitCode,
            default_args: &[],
            default_checks: &[],
            metrics: &[],
            include_files: &[],
            uses_parent_pgrp: false,
            known_flags: None,
            metric_bounds: None,
        };

        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        apply_ops_test(
            &ctx,
            &mut state,
            &[Op::run_payload(&SLEEP, vec!["3600".into()])],
        )
        .expect("first spawn");

        let err = apply_ops_test(
            &ctx,
            &mut state,
            &[Op::run_payload(&SLEEP, vec!["3600".into()])],
        )
        .unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("payload 'sleeper' already running"),
            "error must flag the duplicate, got: {msg}"
        );
        // The dup error must identify the surface that spawned the
        // live handle so the user knows where to go to fix it. The
        // first spawn was via Op::RunPayload, not CgroupDef::workload.
        assert!(
            msg.contains("Op::RunPayload"),
            "dup error must name the originating surface, got: {msg}"
        );
        // The Op::RunPayload in this test ran without a
        // `cgroup = Some(..)`, so the rendered cgroup key must be
        // `(no cgroup)`, not an empty-quoted `''`.
        assert!(
            msg.contains("(no cgroup)"),
            "empty-cgroup key must render as '(no cgroup)', got: {msg}"
        );
        assert!(
            !msg.contains("cgroup ''"),
            "empty-cgroup key must not render as quoted empty, got: {msg}"
        );
        assert_eq!(
            state.payload_handles.len(),
            1,
            "second spawn must not add a handle on failure"
        );

        // Clean up the live handle so the test process doesn't leak
        // a /bin/sleep.
        apply_ops_test(&ctx, &mut state, &[Op::kill_payload("sleeper")]).expect("teardown kill");
    }

    /// When the first spawn came from `CgroupDef::workload` in
    /// `cg_def` and a subsequent `Op::run_payload_in_cgroup` targets
    /// the same `cg_def` with the same payload name, the composite-
    /// key dup check fires and names `CgroupDef::workload` as the
    /// originating surface. A cross-cgroup duplicate (same name,
    /// different cgroup) is legitimate and tested separately.
    #[test]
    fn apply_ops_run_rejects_payload_already_owned_by_cgroup_def() {
        use crate::test_support::{OutputFormat, Payload, PayloadKind};
        static SLEEP: Payload = Payload {
            name: "sleeper",
            kind: PayloadKind::Binary("/bin/sleep"),
            output: OutputFormat::ExitCode,
            default_args: &[],
            default_checks: &[],
            metrics: &[],
            include_files: &[],
            uses_parent_pgrp: false,
            known_flags: None,
            metric_bounds: None,
        };

        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        // Simulate the def-owned handle directly — apply_setup pushes
        // entries with PayloadSource::CgroupDefWorkload, so construct
        // the equivalent here without invoking the real spawn path
        // (apply_setup needs workers(N) and cgroupfs ops which MockCgroupOps
        // does not implement for this test shape).
        let h = crate::scenario::payload_run::PayloadRun::new(&ctx, &SLEEP)
            .args(["3600".to_string()])
            .spawn()
            .expect("manual def-source spawn");
        state.payload_handles.push(PayloadEntry {
            cgroup: "def_cg".to_string(),
            source: PayloadSource::CgroupDefWorkload,
            handle: h,
        });

        // Targeting the SAME cgroup as the pre-existing entry: dup.
        let err = apply_ops_test(
            &ctx,
            &mut state,
            &[Op::run_payload_in_cgroup(
                &SLEEP,
                vec!["1".into()],
                "def_cg",
            )],
        )
        .unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("CgroupDef::workload"),
            "dup error must name the def-source surface, got: {msg}"
        );
        assert!(
            msg.contains("'def_cg'"),
            "dup error must name the cgroup the live handle is in, got: {msg}"
        );
        // Only the original handle remains — op branch bailed pre-spawn.
        assert_eq!(state.payload_handles.len(), 1);

        apply_ops_test(
            &ctx,
            &mut state,
            &[Op::kill_payload_in_cgroup("sleeper", "def_cg")],
        )
        .expect("teardown kill");
    }

    /// [`render_cgroup_key`] renders an empty string as
    /// `(no cgroup)` and a populated name as single-quoted prose.
    /// Pins the formatting so every error path that echoes the
    /// cgroup key through this helper stays consistent.
    #[test]
    fn render_cgroup_key_handles_empty_and_populated() {
        assert_eq!(render_cgroup_key(""), "(no cgroup)");
        assert_eq!(render_cgroup_key("cg_a"), "'cg_a'");
    }

    // -- payload_handles drain on error paths in execute_steps_with --

    /// An Err return from `execute_steps_with` (here: a vacuous
    /// `HoldSpec::Frac(0.0)` caught by up-front validation — `Frac`
    /// rejects `f <= 0.0` per types.rs:1859, while `Fixed(ZERO)` is
    /// deliberately valid for op-only settle steps per types.rs:1854)
    /// leaves no live payload_handles because no setup/ops ran.
    /// Pins the invariant that the pre-ops validation path does
    /// not spawn anything that could then leak.
    #[test]
    fn execute_steps_with_early_validation_err_has_nothing_to_drain() {
        use crate::cgroup::CgroupManager;
        let cgroups = CgroupManager::new("/nonexistent");
        let topo = mock_topo();
        let ctx = crate::scenario::Ctx::builder(&cgroups, &topo).build();
        let step = Step::new(vec![], HoldSpec::Frac(0.0));
        let err = execute_steps_with(&ctx, vec![step], None).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("hold validation") && msg.contains("Frac"),
            "expected pre-ops validation err, got: {msg}"
        );
    }

    /// When a live payload has been spawned and a later op returns
    /// Err, the drain-on-err path consumes the payload handles via
    /// `.kill()` (which emits metrics) rather than leaking them to
    /// `PayloadHandle::drop` (which SIGKILLs without recording).
    ///
    /// This test exercises the drain path directly by spawning a
    /// /bin/sleep, then calling `apply_ops` with an op that forces
    /// an error (unknown-name `WaitPayload`). After the Err, the
    /// state's payload_handles must still be consulted by the
    /// drain — verified by checking the live count before +
    /// explicit teardown after.
    #[test]
    fn apply_ops_error_does_not_lose_live_payload_handles() {
        use crate::test_support::{OutputFormat, Payload, PayloadKind};
        static SLEEP: Payload = Payload {
            name: "sleeper_drain",
            kind: PayloadKind::Binary("/bin/sleep"),
            output: OutputFormat::ExitCode,
            default_args: &[],
            default_checks: &[],
            metrics: &[],
            include_files: &[],
            uses_parent_pgrp: false,
            known_flags: None,
            metric_bounds: None,
        };
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        apply_ops_test(
            &ctx,
            &mut state,
            &[Op::run_payload(&SLEEP, vec!["3600".into()])],
        )
        .expect("spawn");
        assert_eq!(state.payload_handles.len(), 1);
        // Trigger an Err via WaitPayload on an unknown name. Before
        // the fix, execute_steps_with would propagate the Err via
        // `?` and leave the SLEEP handle to be SIGKILLed by Drop
        // (losing the metric emission).
        let err =
            apply_ops_test(&ctx, &mut state, &[Op::wait_payload("never_spawned")]).unwrap_err();
        assert!(
            format!("{err:#}").contains("no running payload named 'never_spawned'"),
            "expected wait-unknown-name err",
        );
        // The live handle is still in state — apply_ops itself does
        // not drain on Err (that's execute_steps_with's
        // responsibility). Manually drain via the helper to
        // terminate the child cleanly.
        drain_all_payload_handles(&mut state.payload_handles);
        assert!(state.payload_handles.is_empty());
    }

    // ---------------------------------------------------------------
    // Step/Backdrop ruling invariants
    // ---------------------------------------------------------------

    /// `Op::RemoveCgroup` and `Op::StopCgroup` reach the cgroup ops
    /// for Backdrop-owned targets from both step-local apply and
    /// Backdrop's own setup pass. RemoveCgroup also drops the
    /// Backdrop tracking entry so a later AddCgroup with the same
    /// name does not collide against a stale slot.
    ///
    /// Regression class: a future re-introduction of the Backdrop-
    /// target rejection (e.g. as a "safety" re-add by a contributor
    /// who didn't see the rationale) would surface here as the
    /// `apply_ops` call returning Err. The framework intentionally
    /// trades the early-bail for permissive removal — tests that
    /// mistype a cgroup name will silently succeed at the
    /// RemoveCgroup site and surface the typo later as a kernel-
    /// layer `cgroup missing` error on the next op that references
    /// the name.
    #[test]
    fn remove_and_stop_cgroup_permit_backdrop_target_from_step() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);

        let mut step_state = StepState::empty(&ctx);
        let mut backdrop_state = BackdropState::empty(&ctx);
        backdrop_state
            .cgroups
            .add_cgroup_no_cpuset("bd_cg")
            .expect("add backdrop cgroup");

        {
            let mut scenario = ScenarioState::new(&mut step_state, &mut backdrop_state);
            apply_ops(&ctx, &mut scenario, &[Op::remove_cgroup("bd_cg")])
                .expect("step-local RemoveCgroup permitted against Backdrop target");
        }
        let calls = mock.calls();
        assert!(
            calls
                .iter()
                .any(|c| matches!(c, CgroupCall::RemoveCgroup(n) if n == "bd_cg")),
            "step-local remove must reach the cgroup ops, got: {calls:?}"
        );
        assert!(
            !backdrop_state.cgroups.names().iter().any(|n| n == "bd_cg"),
            "post-RemoveCgroup must drop backdrop tracking entry, got: {:?}",
            backdrop_state.cgroups.names()
        );

        // Slot is free — re-adding the same name must succeed.
        {
            let mut scenario = ScenarioState::new(&mut step_state, &mut backdrop_state);
            apply_ops(&ctx, &mut scenario, &[Op::add_cgroup("bd_cg")])
                .expect("AddCgroup with previously-removed name must succeed");
        }

        backdrop_state
            .cgroups
            .add_cgroup_no_cpuset("bd_cg_2")
            .expect("add second backdrop cgroup");
        {
            let mut scenario = ScenarioState::new(&mut step_state, &mut backdrop_state);
            apply_ops(&ctx, &mut scenario, &[Op::stop_cgroup("bd_cg_2")])
                .expect("step-local StopCgroup permitted against Backdrop target");
        }

        backdrop_state
            .cgroups
            .add_cgroup_no_cpuset("bd_cg_3")
            .expect("add third backdrop cgroup");
        {
            let mut scenario = ScenarioState::new(&mut step_state, &mut backdrop_state);
            scenario
                .with_target_backdrop(|s| apply_ops(&ctx, s, &[Op::remove_cgroup("bd_cg_3")]))
                .expect("backdrop-pass RemoveCgroup permitted against Backdrop target");
        }

        cleanup_state(&mut step_state);
    }

    /// `Op::MoveAllTasks` from a step-local cgroup to a Backdrop
    /// cgroup must transfer the handle from step-local slot to
    /// backdrop slot so the worker survives the step boundary. A
    /// step-to-step move keeps ownership step-local. A backdrop-to-
    /// step move keeps the handle in the backdrop slot (persistent
    /// does not degrade).
    #[test]
    fn move_all_tasks_transfers_handle_ownership_step_to_backdrop() {
        use crate::workload::{WorkSpec, WorkloadConfig, WorkloadHandle};

        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);

        let mut step_state = StepState::empty(&ctx);
        let mut backdrop_state = BackdropState::empty(&ctx);
        // Backdrop owns "bd_cg"; the step owns "step_cg" and a
        // handle keyed under it.
        backdrop_state
            .cgroups
            .add_cgroup_no_cpuset("bd_cg")
            .unwrap();
        step_state.cgroups.add_cgroup_no_cpuset("step_cg").unwrap();
        let w = WorkSpec::default();
        let wl = WorkloadConfig {
            num_workers: 1,
            affinity: crate::workload::AffinityIntent::Inherit,
            work_type: w.work_type,
            sched_policy: w.sched_policy,
            mem_policy: w.mem_policy,
            mpol_flags: w.mpol_flags,
            nice: None,
            clone_mode: Default::default(),
            comm: None,
            uid: None,
            gid: None,
            numa_node: None,
            composed: Vec::new(),
        };
        let h = WorkloadHandle::spawn(&wl).expect("spawn worker");
        step_state.handles.push(("step_cg".to_string(), h));
        assert_eq!(step_state.handles.len(), 1);
        assert_eq!(backdrop_state.handles.len(), 0);

        // Move tasks from step_cg to bd_cg: ownership transfers.
        {
            let mut scenario = ScenarioState::new(&mut step_state, &mut backdrop_state);
            apply_ops(
                &ctx,
                &mut scenario,
                &[Op::move_all_tasks("step_cg", "bd_cg")],
            )
            .expect("move into backdrop");
        }
        assert_eq!(
            step_state.handles.len(),
            0,
            "step-local handle must leave the step slot after transfer",
        );
        assert_eq!(
            backdrop_state.handles.len(),
            1,
            "backdrop slot must receive the transferred handle",
        );
        assert_eq!(
            backdrop_state.handles[0].0, "bd_cg",
            "transferred handle must be re-keyed to `to`",
        );

        // Clear the handles before the test drops (handles SIGKILL on
        // drop — avoid leaking the worker process).
        backdrop_state.handles.clear();
        step_state.handles.clear();
    }

    /// Step→step move does NOT cross state slots (companion to the
    /// step→backdrop transfer test above).
    #[test]
    fn move_all_tasks_step_to_step_keeps_step_ownership() {
        use crate::workload::{WorkSpec, WorkloadConfig, WorkloadHandle};

        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut step_state = StepState::empty(&ctx);
        let mut backdrop_state = BackdropState::empty(&ctx);
        step_state.cgroups.add_cgroup_no_cpuset("src").unwrap();
        step_state.cgroups.add_cgroup_no_cpuset("dst").unwrap();
        let w = WorkSpec::default();
        let wl = WorkloadConfig {
            num_workers: 1,
            affinity: crate::workload::AffinityIntent::Inherit,
            work_type: w.work_type,
            sched_policy: w.sched_policy,
            mem_policy: w.mem_policy,
            mpol_flags: w.mpol_flags,
            nice: None,
            clone_mode: Default::default(),
            comm: None,
            uid: None,
            gid: None,
            numa_node: None,
            composed: Vec::new(),
        };
        let h = WorkloadHandle::spawn(&wl).expect("spawn");
        step_state.handles.push(("src".to_string(), h));
        {
            let mut scenario = ScenarioState::new(&mut step_state, &mut backdrop_state);
            apply_ops(&ctx, &mut scenario, &[Op::move_all_tasks("src", "dst")])
                .expect("step-to-step move");
        }
        assert_eq!(step_state.handles.len(), 1);
        assert_eq!(step_state.handles[0].0, "dst");
        assert_eq!(backdrop_state.handles.len(), 0);
        step_state.handles.clear();
    }

    /// A step-local `Op::MoveAllTasks` that
    /// pulls from a Backdrop-owned cgroup into a step-local cgroup
    /// must bail before touching cgroupfs. The persistent worker
    /// would otherwise be stranded in a cgroup that gets rmdir'd at
    /// the step boundary. Backdrop-setup ops (`target_backdrop`)
    /// stay exempt.
    #[test]
    fn move_all_tasks_backdrop_to_step_rejected() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut step_state = StepState::empty(&ctx);
        let mut backdrop_state = BackdropState::empty(&ctx);
        backdrop_state.cgroups.add_cgroup_no_cpuset("bd").unwrap();
        step_state.cgroups.add_cgroup_no_cpuset("step").unwrap();

        let mut scenario = ScenarioState::new(&mut step_state, &mut backdrop_state);
        let err = apply_ops(&ctx, &mut scenario, &[Op::move_all_tasks("bd", "step")]).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("Backdrop-owned 'bd'") && msg.contains("step-local 'step'"),
            "error must name both cgroups and the direction, got: {msg}"
        );
        // The mock must not have seen a cgroup.procs write — the
        // guard bails before any kernel-side work.
        let calls = mock.calls();
        assert!(
            !calls
                .iter()
                .any(|c| matches!(c, CgroupCall::MoveTasks(_, _))),
            "pre-bail path must not invoke move_tasks, got: {calls:?}"
        );
    }

    /// `run_scenario` rejects a scheduler-kind payload in
    /// `Backdrop::payloads` before running any setup.
    #[test]
    fn run_scenario_rejects_scheduler_kind_backdrop_payload() {
        use crate::cgroup::CgroupManager;
        use crate::test_support::Payload;
        let cgroups = CgroupManager::new("/nonexistent");
        let topo = mock_topo();
        let ctx = crate::scenario::Ctx::builder(&cgroups, &topo).build();
        let backdrop =
            crate::scenario::backdrop::Backdrop::new().push_payload(&Payload::KERNEL_DEFAULT);
        let err = execute_scenario_with(
            &ctx,
            backdrop,
            vec![Step::new(vec![], HoldSpec::fixed(Duration::from_millis(1)))],
            None,
        )
        .unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("scheduler-kind") && msg.contains("Backdrop"),
            "error must name the kind mismatch and the Backdrop surface, got: {msg}"
        );
    }

    /// `apply_setup` rejects a step-local CgroupDef whose name
    /// collides with a Backdrop-tracked cgroup.
    #[test]
    fn apply_setup_rejects_name_collision_with_backdrop() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut step_state = StepState::empty(&ctx);
        let mut backdrop_state = BackdropState::empty(&ctx);
        backdrop_state
            .cgroups
            .add_cgroup_no_cpuset("shared")
            .unwrap();
        let defs = vec![CgroupDef::named("shared").workers(1)];
        let mut scenario = ScenarioState::new(&mut step_state, &mut backdrop_state);
        let err = apply_setup(&ctx, &mut scenario, &defs).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("already tracked") && msg.contains("shared"),
            "error must cite the collision and the offending name, got: {msg}"
        );
        cleanup_state(&mut step_state);
    }

    // ---------------------------------------------------------------
    // composite-key (name, cgroup) dedup for Op::RunPayload
    // ---------------------------------------------------------------

    /// Push a synthetic live PayloadEntry into `state`'s step slot
    /// so tests can exercise dedup / lookup paths without paying
    /// the cost of a real cgroupfs-backed spawn (which fails inside
    /// the MockCgroupOps test harness because `/mock/cgroup/...`
    /// doesn't exist on disk).
    fn push_fake_payload_entry<'a>(
        ctx: &'a Ctx<'a>,
        state: &mut StepState<'a>,
        payload: &'static crate::test_support::Payload,
        cgroup: &str,
        source: PayloadSource,
    ) {
        let h = crate::scenario::payload_run::PayloadRun::new(ctx, payload)
            .args(["3600".to_string()])
            .spawn()
            .expect("manual spawn (no cgroup placement)");
        state.payload_handles.push(PayloadEntry {
            cgroup: cgroup.to_string(),
            source,
            handle: h,
        });
    }

    /// Same payload live in `cg_a` AND `cg_b`; a third
    /// `Op::RunPayload` targeting a brand-new `cg_c` must NOT trip
    /// the composite-key dedup because the (name, cgroup) pair is
    /// fresh. Simulated via direct state injection so the test
    /// doesn't depend on cgroupfs.
    #[test]
    fn apply_ops_run_duplicate_name_different_cgroups_allowed() {
        use crate::test_support::{OutputFormat, Payload, PayloadKind};
        static SLEEP: Payload = Payload {
            name: "sleeper",
            kind: PayloadKind::Binary("/bin/sleep"),
            output: OutputFormat::ExitCode,
            default_args: &[],
            default_checks: &[],
            metrics: &[],
            include_files: &[],
            uses_parent_pgrp: false,
            known_flags: None,
            metric_bounds: None,
        };
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        push_fake_payload_entry(
            &ctx,
            &mut state,
            &SLEEP,
            "cg_a",
            PayloadSource::OpRunPayload,
        );
        push_fake_payload_entry(
            &ctx,
            &mut state,
            &SLEEP,
            "cg_b",
            PayloadSource::OpRunPayload,
        );

        let mut backdrop = BackdropState::empty(&ctx);
        let scenario = ScenarioState::new(&mut state, &mut backdrop);
        // The `find_live_payload_with_cgroup` lookup for ("sleeper", "cg_c")
        // returns None because no live entry matches that pair — so
        // the dup check passes and run_scenario would let the spawn
        // proceed. We check the lookup directly (spawning against
        // MockCgroupOps would fail on the pre_exec cgroup write).
        assert!(
            scenario
                .find_live_payload_with_cgroup("sleeper", "cg_c")
                .is_none(),
            "fresh (name, cgroup) pair must not collide with live entries in other cgroups",
        );
        // And the existing same-cgroup entry still collides.
        assert!(
            scenario
                .find_live_payload_with_cgroup("sleeper", "cg_a")
                .is_some(),
            "same (name, cgroup) still matches — only the pair matters",
        );

        cleanup_state(&mut state);
    }

    /// `take_payload_by_name` in composite mode matches only the
    /// exact `(name, cgroup)` pair and leaves sibling copies alone.
    #[test]
    fn take_payload_by_composite_key_matches_exact_cgroup() {
        use crate::test_support::{OutputFormat, Payload, PayloadKind};
        static SLEEP: Payload = Payload {
            name: "sleeper",
            kind: PayloadKind::Binary("/bin/sleep"),
            output: OutputFormat::ExitCode,
            default_args: &[],
            default_checks: &[],
            metrics: &[],
            include_files: &[],
            uses_parent_pgrp: false,
            known_flags: None,
            metric_bounds: None,
        };
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        push_fake_payload_entry(
            &ctx,
            &mut state,
            &SLEEP,
            "cg_a",
            PayloadSource::OpRunPayload,
        );
        push_fake_payload_entry(
            &ctx,
            &mut state,
            &SLEEP,
            "cg_b",
            PayloadSource::OpRunPayload,
        );

        let mut backdrop = BackdropState::empty(&ctx);
        let mut scenario = ScenarioState::new(&mut state, &mut backdrop);
        let taken = scenario
            .take_payload_by_name("sleeper", Some("cg_a"))
            .expect("composite lookup does not bail on ambiguity")
            .expect("one entry matches");
        assert_eq!(taken.cgroup, "cg_a");
        // The cg_b entry survives.
        assert_eq!(state.payload_handles.len(), 1);
        assert_eq!(state.payload_handles[0].cgroup, "cg_b");
        // Drain to avoid leaking the live child.
        drain_all_payload_handles(&mut state.payload_handles);
        let _ = taken.handle.kill();
    }

    /// Bare `take_payload_by_name(name, None)` returns
    /// `Err(ambiguous_cgroups)` when two or more copies are live,
    /// surfacing both cgroup keys so the caller can disambiguate.
    #[test]
    fn take_payload_by_bare_name_reports_ambiguous_cgroups() {
        use crate::test_support::{OutputFormat, Payload, PayloadKind};
        static SLEEP: Payload = Payload {
            name: "sleeper",
            kind: PayloadKind::Binary("/bin/sleep"),
            output: OutputFormat::ExitCode,
            default_args: &[],
            default_checks: &[],
            metrics: &[],
            include_files: &[],
            uses_parent_pgrp: false,
            known_flags: None,
            metric_bounds: None,
        };
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        push_fake_payload_entry(
            &ctx,
            &mut state,
            &SLEEP,
            "cg_a",
            PayloadSource::OpRunPayload,
        );
        push_fake_payload_entry(
            &ctx,
            &mut state,
            &SLEEP,
            "cg_b",
            PayloadSource::OpRunPayload,
        );

        let mut backdrop = BackdropState::empty(&ctx);
        let mut scenario = ScenarioState::new(&mut state, &mut backdrop);
        let err = match scenario.take_payload_by_name("sleeper", None) {
            Err(cgroups) => cgroups,
            Ok(_) => panic!("bare lookup over multi-copy must surface ambiguity"),
        };
        assert_eq!(err.len(), 2);
        assert!(err.contains(&"cg_a".to_string()) && err.contains(&"cg_b".to_string()));
        // No handle consumed — both still live.
        assert_eq!(state.payload_handles.len(), 2);
        drain_all_payload_handles(&mut state.payload_handles);
    }

    /// Bare `take_payload_by_name(name, None)` succeeds when
    /// exactly one copy is live, so `Op::wait_payload(name)` and
    /// `Op::kill_payload(name)` don't need to carry a cgroup
    /// argument in the single-copy case.
    #[test]
    fn take_payload_by_bare_name_succeeds_on_single_copy() {
        use crate::test_support::{OutputFormat, Payload, PayloadKind};
        static SLEEP: Payload = Payload {
            name: "sleeper",
            kind: PayloadKind::Binary("/bin/sleep"),
            output: OutputFormat::ExitCode,
            default_args: &[],
            default_checks: &[],
            metrics: &[],
            include_files: &[],
            uses_parent_pgrp: false,
            known_flags: None,
            metric_bounds: None,
        };
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        push_fake_payload_entry(
            &ctx,
            &mut state,
            &SLEEP,
            "cg_a",
            PayloadSource::OpRunPayload,
        );

        let mut backdrop = BackdropState::empty(&ctx);
        let mut scenario = ScenarioState::new(&mut state, &mut backdrop);
        let taken = scenario
            .take_payload_by_name("sleeper", None)
            .expect("single-copy bare lookup returns Ok")
            .expect("one entry matches");
        assert_eq!(taken.cgroup, "cg_a");
        assert!(state.payload_handles.is_empty());
        let _ = taken.handle.kill();
    }

    /// The apply_ops ambiguity hint must spell the full snake_case
    /// constructor path so a user copying the hint into source
    /// writes something that actually compiles. Covers both
    /// `Op::wait_payload` and `Op::kill_payload` entry points
    /// because they route through the same helper.
    #[test]
    fn apply_ops_bare_wait_and_kill_ambiguity_hint_names_full_constructor() {
        use crate::test_support::{OutputFormat, Payload, PayloadKind};
        static SLEEP: Payload = Payload {
            name: "sleeper",
            kind: PayloadKind::Binary("/bin/sleep"),
            output: OutputFormat::ExitCode,
            default_args: &[],
            default_checks: &[],
            metrics: &[],
            include_files: &[],
            uses_parent_pgrp: false,
            known_flags: None,
            metric_bounds: None,
        };
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);

        // WaitPayload path.
        let mut state = StepState::empty(&ctx);
        push_fake_payload_entry(
            &ctx,
            &mut state,
            &SLEEP,
            "cg_a",
            PayloadSource::OpRunPayload,
        );
        push_fake_payload_entry(
            &ctx,
            &mut state,
            &SLEEP,
            "cg_b",
            PayloadSource::OpRunPayload,
        );
        let err = apply_ops_test(&ctx, &mut state, &[Op::wait_payload("sleeper")]).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("ambiguous"),
            "wait ambiguity message must flag ambiguity, got: {msg}"
        );
        assert!(
            msg.contains("Op::wait_payload_in_cgroup(name, cgroup)"),
            "wait ambiguity hint must name the full snake_case constructor \
             so a copy-paste into source compiles, got: {msg}"
        );
        drain_all_payload_handles(&mut state.payload_handles);

        // KillPayload path.
        let mut state = StepState::empty(&ctx);
        push_fake_payload_entry(
            &ctx,
            &mut state,
            &SLEEP,
            "cg_a",
            PayloadSource::OpRunPayload,
        );
        push_fake_payload_entry(
            &ctx,
            &mut state,
            &SLEEP,
            "cg_b",
            PayloadSource::OpRunPayload,
        );
        let err = apply_ops_test(&ctx, &mut state, &[Op::kill_payload("sleeper")]).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("Op::kill_payload_in_cgroup(name, cgroup)"),
            "kill ambiguity hint must name the full snake_case constructor, got: {msg}"
        );
        drain_all_payload_handles(&mut state.payload_handles);
    }

    /// The not-found arm uses `-ing` verb form ("before waiting" /
    /// "before killing"), not the collapsed single-word lowercase
    /// a previous implementation emitted.
    #[test]
    fn apply_ops_not_found_message_uses_gerund_verb() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        let err = apply_ops_test(&ctx, &mut state, &[Op::wait_payload("ghost")]).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("before waiting"),
            "wait not-found message must say 'before waiting', got: {msg}"
        );
        assert!(
            !msg.contains("before waitpayload"),
            "must not collapse 'wait payload' into 'waitpayload', got: {msg}"
        );

        let err = apply_ops_test(&ctx, &mut state, &[Op::kill_payload("ghost")]).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("before killing"),
            "kill not-found message must say 'before killing', got: {msg}"
        );
    }

    // ---------------------------------------------------------------
    // Step-local vs Backdrop state invariants
    // ---------------------------------------------------------------

    /// Op::RemoveCgroup prunes the name from CgroupGroup's tracked
    /// `names` vec via the `forget` helper BEFORE dispatching
    /// `ctx.cgroups.remove_cgroup`. Without the prune, the stale
    /// tracking entry would re-trigger the AddCgroup collision check
    /// for a same-name re-create, and CgroupGroup's Drop would invoke
    /// a redundant rmdir against an already-removed dir. Pin both:
    /// names() reflects only the surviving cgroup, and the mock
    /// observes exactly one RemoveCgroup call for the dropped name
    /// (from the Op — Drop does not see it).
    #[test]
    fn remove_cgroup_forgets_name_so_drop_does_not_double_rmdir() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        apply_ops_test(
            &ctx,
            &mut state,
            &[Op::add_cgroup("cg_keep"), Op::add_cgroup("cg_drop")],
        )
        .unwrap();
        // Op::RemoveCgroup records on the mock AND prunes `cg_drop`
        // from the tracked names — only `cg_keep` survives.
        apply_ops_test(&ctx, &mut state, &[Op::remove_cgroup("cg_drop")]).unwrap();
        assert_eq!(
            state.cgroups.names(),
            &["cg_keep".to_string()],
            "Op::RemoveCgroup must prune the removed name from \
             CgroupGroup::names so a later AddCgroup with the same \
             name can re-create the cgroup without colliding against \
             a stale tracking entry",
        );
        // After Drop, the mock observed exactly one RemoveCgroup
        // call for cg_drop (from the Op itself). Drop iterates only
        // surviving names so it does not re-issue rmdir against
        // cg_drop, and it issues exactly one rmdir for the
        // surviving cg_keep.
        drop(state);
        let calls = mock.calls();
        let cg_drop_removes: Vec<&CgroupCall> = calls
            .iter()
            .filter(|c| matches!(c, CgroupCall::RemoveCgroup(n) if n == "cg_drop"))
            .collect();
        assert_eq!(
            cg_drop_removes.len(),
            1,
            "Op::RemoveCgroup must be the sole rmdir dispatcher for \
             cg_drop; Drop must not re-issue rmdir against a forgotten \
             name: {calls:?}",
        );
        let cg_keep_removes: Vec<&CgroupCall> = calls
            .iter()
            .filter(|c| matches!(c, CgroupCall::RemoveCgroup(n) if n == "cg_keep"))
            .collect();
        assert_eq!(
            cg_keep_removes.len(),
            1,
            "Drop must rmdir the surviving cg_keep exactly once: {calls:?}",
        );
    }

    /// Install a minimal tracing subscriber that records every
    /// event's (Level, concatenated-field-values) pair, run `f`,
    /// and return the captured events. The MessageVisitor handles
    /// both record_debug (Debug-formatted `?` fields wrapped via
    /// `DebugValue`, Display-formatted `%` fields wrapped via
    /// `DisplayValue`, and the format-string message itself via
    /// `fmt::Arguments` — all three route through record_debug) and
    /// record_str (raw string field values), so the returned message
    /// string contains the warn body concatenated with every
    /// structured-field value.
    fn capture_tracing_events<F: FnOnce()>(f: F) -> Vec<(tracing::Level, String)> {
        use std::sync::{Arc, Mutex};
        use tracing::field::{Field, Visit};
        use tracing::span::{Attributes, Id, Record};
        use tracing::{Event, Level, Metadata, Subscriber};

        #[derive(Default)]
        struct CaptureSubscriber {
            events: Arc<Mutex<Vec<(Level, String)>>>,
        }
        struct MessageVisitor<'a>(&'a mut String);
        impl<'a> Visit for MessageVisitor<'a> {
            fn record_debug(&mut self, _field: &Field, value: &dyn std::fmt::Debug) {
                use std::fmt::Write;
                let _ = write!(self.0, "{value:?} ");
            }
            fn record_str(&mut self, _field: &Field, value: &str) {
                use std::fmt::Write;
                let _ = write!(self.0, "{value} ");
            }
        }
        impl Subscriber for CaptureSubscriber {
            fn enabled(&self, _: &Metadata<'_>) -> bool {
                true
            }
            fn new_span(&self, _: &Attributes<'_>) -> Id {
                Id::from_u64(1)
            }
            fn record(&self, _: &Id, _: &Record<'_>) {}
            fn record_follows_from(&self, _: &Id, _: &Id) {}
            fn event(&self, event: &Event<'_>) {
                let mut msg = String::new();
                event.record(&mut MessageVisitor(&mut msg));
                self.events
                    .lock()
                    .unwrap()
                    .push((*event.metadata().level(), msg));
            }
            fn enter(&self, _: &Id) {}
            fn exit(&self, _: &Id) {}
        }
        let events: Arc<Mutex<Vec<(Level, String)>>> = Arc::new(Mutex::new(Vec::new()));
        let sub = CaptureSubscriber {
            events: events.clone(),
        };
        tracing::subscriber::with_default(sub, f);
        events.lock().unwrap().clone()
    }

    /// Branch 1 of Op::RemoveCgroup's typo-late-surfacing warn
    /// fires when the removed name was tracked in Backdrop. The
    /// warn correlates a later kernel-level "cgroup missing" with
    /// the intentional removal source. Pin the predicate so a
    /// future refactor that inverts the membership check, swallows
    /// the warn, or routes the Backdrop path to branch 2 surfaces
    /// here.
    #[test]
    fn remove_cgroup_warn_branch_1_fires_on_backdrop_tracked_name() {
        let events = capture_tracing_events(|| {
            let mock = MockCgroupOps::new();
            let topo = mock_topo();
            let ctx = mock_ctx(&mock, &topo);
            let mut step_state = StepState::empty(&ctx);
            let mut backdrop_state = BackdropState::empty(&ctx);
            backdrop_state
                .cgroups
                .add_cgroup_no_cpuset("bd_cg")
                .expect("add backdrop cgroup");
            let mut scenario = ScenarioState::new(&mut step_state, &mut backdrop_state);
            apply_ops(&ctx, &mut scenario, &[Op::remove_cgroup("bd_cg")])
                .expect("remove_cgroup of backdrop target must succeed");
        });
        let warns: Vec<&(tracing::Level, String)> = events
            .iter()
            .filter(|(lvl, _)| *lvl == tracing::Level::WARN)
            .collect();
        assert_eq!(
            warns.len(),
            1,
            "exactly one warn expected from branch 1; got: {warns:?}",
        );
        assert!(
            warns[0]
                .1
                .contains("removed a Backdrop-owned cgroup mid-scenario"),
            "warn must include branch-1 text identifying Backdrop-owned removal; got: {:?}",
            warns[0].1,
        );
        assert!(
            warns[0].1.contains("bd_cg"),
            "warn must include the cgroup name; got: {:?}",
            warns[0].1,
        );
    }

    /// Branch 2 of Op::RemoveCgroup's typo-late-surfacing warn
    /// fires when the removed name matches NEITHER step-local NOR
    /// Backdrop tracking. The warn dumps both lists so the
    /// operator can compare against the test source and find a
    /// typo. Pin the dump-on-typo behavior so a future refactor
    /// that drops a list or fires on the wrong predicate surfaces
    /// here.
    #[test]
    fn remove_cgroup_warn_branch_2_fires_on_unknown_typo_name() {
        let events = capture_tracing_events(|| {
            let mock = MockCgroupOps::new();
            let topo = mock_topo();
            let ctx = mock_ctx(&mock, &topo);
            let mut step_state = StepState::empty(&ctx);
            let mut backdrop_state = BackdropState::empty(&ctx);
            backdrop_state
                .cgroups
                .add_cgroup_no_cpuset("bd_real_name")
                .expect("add backdrop cgroup");
            let mut scenario = ScenarioState::new(&mut step_state, &mut backdrop_state);
            // Add a step-local cgroup so the step_cgroups field
            // in the warn has a non-empty value to substring-match
            // against — guards against a future refactor dropping
            // the step_cgroups field from the warn.
            apply_ops(&ctx, &mut scenario, &[Op::add_cgroup("step_local_real")])
                .expect("add step-local cgroup");
            apply_ops(&ctx, &mut scenario, &[Op::remove_cgroup("bd_typoed_name")])
                .expect("remove_cgroup of unknown name must succeed (permissive)");
        });
        let warns: Vec<&(tracing::Level, String)> = events
            .iter()
            .filter(|(lvl, _)| *lvl == tracing::Level::WARN)
            .collect();
        assert_eq!(
            warns.len(),
            1,
            "exactly one warn expected from branch 2; got: {warns:?}",
        );
        assert!(
            warns[0].1.contains("matches no step-local"),
            "warn must include branch-2 text identifying unknown-name typo; got: {:?}",
            warns[0].1,
        );
        assert!(
            warns[0].1.contains("bd_real_name"),
            "warn must dump backdrop_cgroups list including the real name; got: {:?}",
            warns[0].1,
        );
        assert!(
            warns[0].1.contains("step_local_real"),
            "warn must dump step_cgroups list including the step-local name; got: {:?}",
            warns[0].1,
        );
        assert!(
            warns[0].1.contains("bd_typoed_name"),
            "warn must include the typo'd cgroup target name; got: {:?}",
            warns[0].1,
        );
    }

    /// Branch 2 also fires for a legitimate second-remove of a
    /// name already pruned by a prior Op::RemoveCgroup. The
    /// wording must acknowledge this as one of two possible
    /// causes (typo or double-remove) so a test author seeing the
    /// warn doesn't immediately assume bug. Pin both the warn
    /// emission and the wording.
    #[test]
    fn remove_cgroup_warn_branch_2_fires_on_double_remove_already_forgotten() {
        let events = capture_tracing_events(|| {
            let mock = MockCgroupOps::new();
            let topo = mock_topo();
            let ctx = mock_ctx(&mock, &topo);
            let mut state = StepState::empty(&ctx);
            apply_ops_test(&ctx, &mut state, &[Op::add_cgroup("cg_once")]).unwrap();
            // First remove: in_step is true → branch 2 gated off.
            apply_ops_test(&ctx, &mut state, &[Op::remove_cgroup("cg_once")]).unwrap();
            // Second remove: name already pruned by the prior
            // remove's `forget` → matches neither tracking set →
            // branch 2 fires.
            apply_ops_test(&ctx, &mut state, &[Op::remove_cgroup("cg_once")]).unwrap();
        });
        let warns: Vec<&(tracing::Level, String)> = events
            .iter()
            .filter(|(lvl, _)| *lvl == tracing::Level::WARN)
            .collect();
        assert_eq!(
            warns.len(),
            1,
            "exactly one warn expected — first remove gated by in_step, second remove fires branch 2 once; got: {warns:?}",
        );
        assert!(
            warns[0]
                .1
                .contains("second-remove of an already-forgotten name"),
            "branch-2 wording must acknowledge double-remove as legitimate cause alongside typo; got: {:?}",
            warns[0].1,
        );
    }

    /// Neither warn branch fires on the happy step-local
    /// add-then-remove path. Pin the suppression so a future
    /// refactor that flips the membership predicate (so step-local
    /// removals would log "unknown name") surfaces here.
    #[test]
    fn remove_cgroup_emits_no_warn_on_happy_step_local_path() {
        let events = capture_tracing_events(|| {
            let mock = MockCgroupOps::new();
            let topo = mock_topo();
            let ctx = mock_ctx(&mock, &topo);
            let mut state = StepState::empty(&ctx);
            apply_ops_test(&ctx, &mut state, &[Op::add_cgroup("cg_local")]).unwrap();
            apply_ops_test(&ctx, &mut state, &[Op::remove_cgroup("cg_local")]).unwrap();
        });
        let warns: Vec<&(tracing::Level, String)> = events
            .iter()
            .filter(|(lvl, _)| *lvl == tracing::Level::WARN)
            .collect();
        assert!(
            warns.is_empty(),
            "happy step-local add-then-remove path must emit zero warns; got: {warns:?}",
        );
    }

    /// Step-local `Op::AddCgroup` with a name that already lives
    /// in the Backdrop must route through the same
    /// `cgroup_name_is_tracked` collision guard as `apply_setup`
    /// — otherwise the CgroupGroup would push a shadow step-local
    /// entry that later steps could address, silently racing the
    /// Backdrop's own writes to cpuset / subtree_control on the
    /// same cgroupfs path.
    #[test]
    fn op_add_cgroup_step_local_rejects_collision_with_backdrop() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut step_state = StepState::empty(&ctx);
        let mut backdrop_state = BackdropState::empty(&ctx);
        backdrop_state
            .cgroups
            .add_cgroup_no_cpuset("shared")
            .expect("add backdrop cgroup");
        let mut scenario = ScenarioState::new(&mut step_state, &mut backdrop_state);
        let err = apply_ops(&ctx, &mut scenario, &[Op::add_cgroup("shared")]).expect_err(
            "apply_ops must reject a step-local AddCgroup whose \
                         name already lives in the Backdrop",
        );
        let msg = format!("{err:?}");
        assert!(
            msg.contains("'shared'") && msg.contains("collides"),
            "error must name the colliding cgroup and explain the collision; got: {msg}",
        );
        // Step-local names must NOT gain a shadow entry after the
        // guard fires.
        assert!(
            step_state.cgroups.names().iter().all(|n| n != "shared"),
            "step-local names must not contain the rejected name; got: {:?}",
            step_state.cgroups.names(),
        );
        // Backdrop copy is untouched.
        assert!(
            backdrop_state.cgroups.names().iter().any(|n| n == "shared"),
            "backdrop copy must survive the rejected op",
        );
    }

    /// `Op::AddCgroup` applied twice in one step with the same name
    /// is rejected by the `cgroup_name_is_tracked` collision guard.
    /// The first op adds the name to step-local tracking; the second
    /// sees it already tracked and bails, so the CgroupGroup's name
    /// vec gains exactly one entry and Drop's remove_cgroup runs
    /// once per unique name.
    #[test]
    fn op_add_cgroup_duplicate_in_same_step_is_rejected() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        let err = apply_ops_test(
            &ctx,
            &mut state,
            &[Op::add_cgroup("cg_dup"), Op::add_cgroup("cg_dup")],
        )
        .expect_err("second AddCgroup must fail against the same step-local name");
        let msg = format!("{err:?}");
        assert!(
            msg.contains("'cg_dup'") && msg.contains("collides"),
            "error must name the colliding cgroup and explain the collision; got: {msg}",
        );
        let names = state.cgroups.names();
        assert_eq!(
            names.iter().filter(|n| n.as_str() == "cg_dup").count(),
            1,
            "the first op must register the name exactly once; the second op \
             must not push a shadow entry; got: {names:?}",
        );
    }

    /// `Op::add_cgroup_def(def)` constructor wraps the [`CgroupDef`]
    /// in the `AddCgroupDef` variant without mutation. Pins the
    /// constructor contract so a future refactor that, e.g., merges
    /// AddCgroup and AddCgroupDef into one variant or splits the def
    /// into separate fields surfaces here.
    #[test]
    fn op_add_cgroup_def_constructor_wraps_def_unmutated() {
        let def = CgroupDef::named("midstep").workers(3);
        let op = Op::add_cgroup_def(def.clone());
        match op {
            Op::AddCgroupDef { def: out } => {
                assert_eq!(out.name, def.name);
                assert_eq!(out.merged_works().len(), def.merged_works().len());
                assert_eq!(out.merged_works()[0].num_workers, Some(3));
            }
            other => panic!("expected AddCgroupDef, got {other:?}"),
        }
    }

    /// `Op::AddCgroupDef` dispatches through `apply_setup` so the
    /// cgroup is created in cgroupfs and the def's name is tracked
    /// in step-local state — same observable result as declaring the
    /// def in `Step::with_defs`, just at apply-ops time instead of
    /// the step's setup pass.
    #[test]
    fn op_add_cgroup_def_creates_cgroup_through_apply_setup() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        apply_ops_test(
            &ctx,
            &mut state,
            &[Op::add_cgroup_def(CgroupDef::named("cg_midstep"))],
        )
        .expect("AddCgroupDef must succeed for a fresh name");
        assert!(
            state.cgroups.names().iter().any(|n| n == "cg_midstep"),
            "step-local tracking must record the AddCgroupDef name; got: {:?}",
            state.cgroups.names(),
        );
    }

    /// `Op::AddCgroupDef` reuses `apply_setup`'s dedup check, so a
    /// name that already lives on the Backdrop is rejected with the
    /// same collision diagnostic operators see from a step-local
    /// `Step::with_defs` collision.
    #[test]
    fn op_add_cgroup_def_rejects_collision_with_backdrop() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut step_state = StepState::empty(&ctx);
        let mut backdrop_state = BackdropState::empty(&ctx);
        backdrop_state
            .cgroups
            .add_cgroup_no_cpuset("persistent")
            .expect("add backdrop cgroup");
        let mut scenario = ScenarioState::new(&mut step_state, &mut backdrop_state);
        let err = apply_ops(
            &ctx,
            &mut scenario,
            &[Op::add_cgroup_def(CgroupDef::named("persistent"))],
        )
        .expect_err("AddCgroupDef must reject a name already tracked by the Backdrop");
        let msg = format!("{err:?}");
        assert!(
            msg.contains("'persistent'") && msg.contains("collides"),
            "error must name the colliding cgroup and explain the collision; got: {msg}",
        );
    }

    /// `required_controllers` picks up the controllers needed by a
    /// [`CgroupDef`] embedded in `Op::AddCgroupDef`, so the parent's
    /// `subtree_control` enables Cpuset before the op runs and the
    /// cpuset write at apply-ops time doesn't fail with ENOENT on
    /// the controller file. Without this absorb pass, a scenario
    /// whose only cpuset user is an `Op::AddCgroupDef` would skip
    /// Cpuset controller enablement entirely.
    #[test]
    fn required_controllers_absorbs_add_cgroup_def_cpuset() {
        use crate::cgroup::Controller;
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let steps = vec![Step::new(
            vec![Op::add_cgroup_def(
                CgroupDef::named("cg_pinned").cpuset(CpusetSpec::disjoint(0, 2)),
            )],
            HoldSpec::fixed(Duration::from_millis(1)),
        )];
        let needed = required_controllers(&ctx, &backdrop::Backdrop::new(), &steps);
        assert!(
            needed.contains(&Controller::Cpuset),
            "AddCgroupDef carrying a cpuset must require Cpuset controller; got: {needed:?}",
        );
    }

    /// `Op::AddCgroupDef` carrying `workers(N)` spawns N workers
    /// and emits the resulting `MoveTasks(_, N)` call against the
    /// embedded def's cgroup — proves the delegation to
    /// `apply_setup` invokes the same worker-spawn + move-into-cgroup
    /// path that step-local CgroupDefs use. Mirrors
    /// `apply_setup_moves_spawned_tasks_into_cgroup` (which exercises
    /// the setup-time entry) but enters via the apply-ops entry.
    #[test]
    fn op_add_cgroup_def_spawns_workers_and_moves_into_cgroup() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        apply_ops_test(
            &ctx,
            &mut state,
            &[Op::add_cgroup_def(
                CgroupDef::named("cg_workers").workers(2),
            )],
        )
        .expect("AddCgroupDef with workers must succeed against mock");
        let calls = mock.calls();
        assert!(
            calls.contains(&CgroupCall::MoveTasks("cg_workers".to_string(), 2)),
            "AddCgroupDef must move 2 spawned worker pids into 'cg_workers'; got: {calls:?}",
        );
    }

    /// `Op::AddCgroupDef` carrying a cpuset spec emits a SetCpuset
    /// mock call against the embedded def's resolved CPU set —
    /// proves the delegation to `apply_setup` writes the cpuset
    /// through `CgroupOps::set_cpuset`, not just stages controller
    /// state. Regression class: a future refactor that bypasses
    /// apply_setup's cpuset-write loop for the AddCgroupDef path
    /// would slip past `required_controllers_absorbs_add_cgroup_def_cpuset`
    /// (which only verifies the controller-bitmask side) — this
    /// test pins the actual write.
    #[test]
    fn op_add_cgroup_def_writes_embedded_cpuset_to_mock() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        apply_ops_test(
            &ctx,
            &mut state,
            &[Op::add_cgroup_def(
                CgroupDef::named("cg_pinned").cpuset(CpusetSpec::disjoint(0, 2)),
            )],
        )
        .expect("AddCgroupDef with cpuset must succeed against mock");
        let calls = mock.calls();
        let has_set_cpuset = calls
            .iter()
            .any(|c| matches!(c, CgroupCall::SetCpuset(name, _) if name == "cg_pinned"));
        assert!(
            has_set_cpuset,
            "AddCgroupDef must emit SetCpuset for 'cg_pinned' via apply_setup; got: {calls:?}",
        );
    }

    /// `Op::AddCgroupDef` whose embedded def configures
    /// `workers_pct` against a cpuset that resolves to zero CPUs
    /// surfaces the same diagnostic apply_setup produces in the
    /// step-setup path — the per-pct error message naming the
    /// cgroup, the pct value, and the empty-cpuset condition.
    /// Regression class: a refactor that short-circuits the
    /// workers_pct empty-cpuset check for the AddCgroupDef path
    /// would let a misconfigured def silently spawn 0 workers.
    #[test]
    fn op_add_cgroup_def_workers_pct_empty_cpuset_bails() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        // CpusetSpec::Exact with an empty set resolves to 0 CPUs;
        // workers_pct on top of that hits the dedicated diagnostic.
        let err = apply_ops_test(
            &ctx,
            &mut state,
            &[Op::add_cgroup_def(
                CgroupDef::named("cg_pct")
                    .cpuset(CpusetSpec::exact(std::iter::empty::<usize>()))
                    .workers_pct(0.5),
            )],
        )
        .expect_err("workers_pct + empty cpuset must bail through AddCgroupDef");
        let msg = format!("{err:?}");
        assert!(
            msg.contains("cg_pct") && msg.contains("workers_pct"),
            "diagnostic must name the cgroup and the workers_pct condition; got: {msg}",
        );
    }

    /// `Op::AddCgroupDef` after a prior `Op::AddCgroup` with the
    /// same name in one step is rejected via apply_setup's
    /// collision check (delegation transmits the dedup contract).
    #[test]
    fn op_add_cgroup_def_collides_with_prior_add_cgroup_in_same_step() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        let err = apply_ops_test(
            &ctx,
            &mut state,
            &[
                Op::add_cgroup("shared"),
                Op::add_cgroup_def(CgroupDef::named("shared")),
            ],
        )
        .expect_err("AddCgroupDef must reject a name already tracked by a prior AddCgroup");
        let msg = format!("{err:?}");
        assert!(
            msg.contains("'shared'") && msg.contains("collides"),
            "error must name the colliding cgroup and explain the collision; got: {msg}",
        );
    }

    /// Two `Op::AddCgroupDef` ops with the same name in one step
    /// are rejected — second op hits apply_setup's collision check
    /// against the first op's step-local tracking entry.
    #[test]
    fn op_add_cgroup_def_collides_with_prior_add_cgroup_def_in_same_step() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        let err = apply_ops_test(
            &ctx,
            &mut state,
            &[
                Op::add_cgroup_def(CgroupDef::named("dup")),
                Op::add_cgroup_def(CgroupDef::named("dup")),
            ],
        )
        .expect_err("second AddCgroupDef must reject the duplicated name");
        let msg = format!("{err:?}");
        assert!(
            msg.contains("'dup'") && msg.contains("collides"),
            "error must name the duplicated cgroup and explain the collision; got: {msg}",
        );
    }

    /// `Op::AddCgroup` after a prior `Op::AddCgroupDef` with the
    /// same name in one step is rejected via the AddCgroup arm's
    /// `cgroup_name_is_tracked` check — symmetric of the
    /// AddCgroup-then-AddCgroupDef ordering. Without symmetric
    /// coverage, a refactor that scoped tracking differently
    /// per-arm could let a name escape the dedup in one ordering
    /// only.
    #[test]
    fn op_add_cgroup_collides_with_prior_add_cgroup_def_in_same_step() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        let err = apply_ops_test(
            &ctx,
            &mut state,
            &[
                Op::add_cgroup_def(CgroupDef::named("shared")),
                Op::add_cgroup("shared"),
            ],
        )
        .expect_err("AddCgroup must reject a name already tracked by a prior AddCgroupDef");
        let msg = format!("{err:?}");
        assert!(
            msg.contains("'shared'") && msg.contains("collides"),
            "error must name the colliding cgroup and explain the collision; got: {msg}",
        );
    }

    /// `MoveAllTasks` must re-key EVERY workload handle whose
    /// current name matches `from`, not just the first. Multiple
    /// handles on the same cgroup arise when a scenario issues two
    /// `Op::SpawnWorkers` ops on the same cgroup name.
    #[test]
    fn move_all_tasks_renames_every_handle_keyed_under_from() {
        use crate::workload::{AffinityIntent, WorkType, WorkloadConfig, WorkloadHandle};

        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut step_state = StepState::empty(&ctx);
        let mut backdrop_state = BackdropState::empty(&ctx);
        step_state.cgroups.add_cgroup_no_cpuset("src").unwrap();
        step_state.cgroups.add_cgroup_no_cpuset("dst").unwrap();

        // Push THREE handles all keyed under "src" — simulates two
        // Op::SpawnWorkers ops in the same cgroup + one from CgroupDef.
        for _ in 0..3 {
            let wl = WorkloadConfig {
                num_workers: 1,
                affinity: AffinityIntent::Inherit,
                work_type: WorkType::SpinWait,
                ..Default::default()
            };
            let h = WorkloadHandle::spawn(&wl).expect("spawn worker");
            step_state.handles.push(("src".to_string(), h));
        }
        assert_eq!(step_state.handles.len(), 3);

        {
            let mut scenario = ScenarioState::new(&mut step_state, &mut backdrop_state);
            apply_ops(&ctx, &mut scenario, &[Op::move_all_tasks("src", "dst")]).expect("move");
        }

        assert_eq!(step_state.handles.len(), 3, "no handles lost");
        assert!(
            step_state.handles.iter().all(|(name, _)| name == "dst"),
            "every handle must be re-keyed to 'dst': {:?}",
            step_state
                .handles
                .iter()
                .map(|(n, _)| n.as_str())
                .collect::<Vec<_>>(),
        );
        // SIGKILL before drop so the synthetic workers don't leak.
        step_state.handles.clear();
    }

    /// Per-step teardown is observable via the mock's call log.
    /// `execute_scenario` runs Step::cgroups Drop at step boundary;
    /// with MockCgroupOps we can pin that the rmdir calls happen
    /// (a) only on step-local cgroups, (b) in REVERSE order of
    /// addition — nested-cgroup-safe teardown.
    #[test]
    fn per_step_teardown_removes_step_local_cgroups_in_reverse_order() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        apply_ops_test(
            &ctx,
            &mut state,
            &[
                Op::add_cgroup("cg_a"),
                Op::add_cgroup("cg_a/sub"),
                Op::add_cgroup("cg_b"),
            ],
        )
        .unwrap();
        // Simulate step boundary: drop the state to run CgroupGroup::Drop.
        drop(state);
        let calls = mock.calls();
        let removes: Vec<&str> = calls
            .iter()
            .filter_map(|c| match c {
                CgroupCall::RemoveCgroup(n) => Some(n.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(
            removes,
            vec!["cg_b", "cg_a/sub", "cg_a"],
            "per-step teardown must rmdir in reverse addition order so a \
             child cgroup's directory is gone before its parent's rmdir \
             runs",
        );
    }

    /// `build_stimulus` encodes the 1-indexed phase number
    /// (`step_idx + 1`) into the wire `step_index` slot, saturating
    /// to `u16::MAX` (with a `tracing::warn!`) when the +1 would
    /// overflow u16. The 0 slot is reserved for the BASELINE
    /// pre-first-Step window the framework never emits a stimulus
    /// for, so the lowest wire value `build_stimulus` ever produces
    /// is 1. Exercise the three interesting values:
    ///
    /// - `step_idx == 0` -> wire `step_index == 1` (first Step,
    ///   not BASELINE).
    /// - `step_idx == u16::MAX as usize - 1` -> wire
    ///   `step_index == u16::MAX` (highest 1-indexed value that
    ///   fits without saturation).
    /// - `step_idx == u16::MAX as usize` -> wire
    ///   `step_index == u16::MAX` (the +1 overflows; must saturate
    ///   instead of wrapping to 0).
    #[test]
    fn build_stimulus_saturates_step_idx_at_u16_max() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut step_state = StepState::empty(&ctx);
        let mut backdrop_state = BackdropState::empty(&ctx);
        let scenario = ScenarioState::new(&mut step_state, &mut backdrop_state);
        let start = std::time::Instant::now();

        let zero = build_stimulus(&start, 0, &[], &scenario);
        assert_eq!(
            zero.step_index, 1,
            "scenario step_idx=0 publishes wire step_index=1 \
             per the 1-indexed phase encoding (BASELINE owns 0)",
        );

        let last_unsaturated = build_stimulus(&start, u16::MAX as usize - 1, &[], &scenario);
        assert_eq!(
            last_unsaturated.step_index,
            u16::MAX,
            "scenario step_idx=u16::MAX - 1 publishes wire step_index=u16::MAX \
             without saturation (highest 1-indexed value that fits)",
        );

        let overflow = build_stimulus(&start, u16::MAX as usize, &[], &scenario);
        assert_eq!(
            overflow.step_index,
            u16::MAX,
            "scenario step_idx=u16::MAX would publish wire step_index=u16::MAX+1 \
             after the 1-indexed +1, so the encoder must saturate to u16::MAX \
             rather than wrap to 0",
        );

        // Far-overflow smoke check: the helper handles values
        // orders of magnitude past u16::MAX without panicking or
        // returning nonsense. The saturated value is u16::MAX
        // regardless of how far past the boundary `step_idx`
        // landed.
        let far = build_stimulus(&start, u32::MAX as usize, &[], &scenario);
        assert_eq!(
            far.step_index,
            u16::MAX,
            "far-overflow step_idx must saturate to u16::MAX",
        );
    }

    /// Saturation without a warn would silently clip the wire field;
    /// the `tracing::warn!` inside `to_u16` is the only observable
    /// signal an operator gets when a scenario blew past `u16::MAX`.
    /// Install a minimal capturing subscriber, run a saturation-
    /// triggering call, and assert the warn event fired.
    #[test]
    fn build_stimulus_warns_on_step_idx_saturation() {
        use std::sync::{Arc, Mutex};
        use tracing::field::{Field, Visit};
        use tracing::span::{Attributes, Id, Record};
        use tracing::{Event, Subscriber};
        use tracing::{Level, Metadata};

        // Capturing subscriber that records `(level, message)` pairs
        // for every event. Span-related methods are implemented as
        // no-ops; the test only cares about event emission.
        #[derive(Default)]
        struct CaptureSubscriber {
            events: Arc<Mutex<Vec<(Level, String)>>>,
        }
        struct MessageVisitor<'a>(&'a mut String);
        impl<'a> Visit for MessageVisitor<'a> {
            fn record_debug(&mut self, _field: &Field, value: &dyn std::fmt::Debug) {
                use std::fmt::Write;
                let _ = write!(self.0, "{value:?} ");
            }
            fn record_str(&mut self, _field: &Field, value: &str) {
                use std::fmt::Write;
                let _ = write!(self.0, "{value} ");
            }
        }
        impl Subscriber for CaptureSubscriber {
            fn enabled(&self, _: &Metadata<'_>) -> bool {
                true
            }
            fn new_span(&self, _: &Attributes<'_>) -> Id {
                Id::from_u64(1)
            }
            fn record(&self, _: &Id, _: &Record<'_>) {}
            fn record_follows_from(&self, _: &Id, _: &Id) {}
            fn event(&self, event: &Event<'_>) {
                let mut msg = String::new();
                event.record(&mut MessageVisitor(&mut msg));
                self.events
                    .lock()
                    .unwrap()
                    .push((*event.metadata().level(), msg));
            }
            fn enter(&self, _: &Id) {}
            fn exit(&self, _: &Id) {}
        }

        let events: Arc<Mutex<Vec<(Level, String)>>> = Arc::new(Mutex::new(Vec::new()));
        let sub = CaptureSubscriber {
            events: events.clone(),
        };

        tracing::subscriber::with_default(sub, || {
            let mock = MockCgroupOps::new();
            let topo = mock_topo();
            let ctx = mock_ctx(&mock, &topo);
            let mut step_state = StepState::empty(&ctx);
            let mut backdrop_state = BackdropState::empty(&ctx);
            let scenario = ScenarioState::new(&mut step_state, &mut backdrop_state);
            let start = std::time::Instant::now();

            // In-range call: no saturation, no warn expected.
            let _ = build_stimulus(&start, 0, &[], &scenario);
            // Saturating call: must emit a warn naming the
            // overflowing field and the offending value. The
            // 1-indexed encoding (`step_idx + 1`) saturates when
            // the +1 would exceed u16::MAX, which kicks in at
            // `step_idx == u16::MAX as usize`.
            let _ = build_stimulus(&start, u16::MAX as usize, &[], &scenario);
        });

        let captured = events.lock().unwrap();
        let warn_hits: Vec<&String> = captured
            .iter()
            .filter(|(lvl, _)| *lvl == Level::WARN)
            .map(|(_, msg)| msg)
            .collect();
        assert!(
            warn_hits.iter().any(|m| m.contains("step_index")
                && m.contains("StimulusPayload step_index overflowed u16")),
            "saturation must emit a tracing::warn naming step_index; got warns: {warn_hits:?}",
        );
        // Sanity: no warn should fire for the in-range 0 call.
        // Since we can't easily partition the two calls, we assert
        // the total count is exactly one: saturating call fires
        // once, in-range call fires zero.
        assert_eq!(
            warn_hits.len(),
            1,
            "exactly one saturation warn expected; got: {warn_hits:?}",
        );
    }

    // -- Op variant constructor coverage --
    //
    // `Op` is `#[non_exhaustive]` — its doc directs downstream
    // authors to use the per-op constructors (`Op::add_cgroup`,
    // `Op::run_payload`, …) rather than naming variants directly so
    // new variants can land without breaking matchers. This test is
    // the enforcement seam: it exercises every documented constructor
    // once AND pattern-matches the produced value against every Op
    // variant without a wildcard arm. Either half failing catches a
    // different regression:
    //
    // - A new variant added without a constructor fails the match
    //   compilation (non-exhaustive pattern).
    // - A new variant with a constructor but no test coverage
    //   survives compilation but the constructor block below won't
    //   cover it — a reviewer adding a variant + constructor must
    //   also add a call here.
    //
    // The guard is build-time rather than runtime: removing the
    // wildcard `_ =>` arm makes the rustc exhaustiveness checker
    // own the constructor-per-variant contract.

    /// Static binary-kind Payload used only to address the
    /// `RunPayload` / `WaitPayload` / `KillPayload` constructors.
    /// The test never spawns or runs this payload — only the
    /// `&'static Payload` reference is consumed.
    static CONSTRUCTOR_TEST_PAYLOAD: crate::test_support::Payload =
        crate::test_support::Payload::binary("constructor-test", "/bin/true");

    /// Static Scheduler used only to address the AttachScheduler /
    /// ReplaceScheduler constructors. The test never spawns or
    /// attaches this scheduler — only the `&'static Scheduler`
    /// reference is consumed. `EEVDF` is the zero-binary baseline
    /// so the fixture has no init-time cost.
    static CONSTRUCTOR_TEST_SCHEDULER: crate::test_support::Scheduler =
        crate::test_support::Scheduler::EEVDF;

    #[test]
    fn op_constructor_coverage_is_exhaustive() {
        let w = WorkSpec::default();
        let constructed: Vec<Op> = vec![
            Op::add_cgroup("a"),
            Op::add_cgroup_def(CgroupDef::named("midstep")),
            Op::remove_cgroup("a"),
            Op::set_cpuset("a", CpusetSpec::Llc(0)),
            Op::clear_cpuset("a"),
            Op::swap_cpusets("a", "b"),
            Op::spawn_workers("a", w.clone()),
            Op::stop_cgroup("a"),
            Op::set_affinity("a", AffinityIntent::Inherit),
            Op::spawn_host(w.clone()),
            Op::move_all_tasks("a", "b"),
            Op::run_payload(&CONSTRUCTOR_TEST_PAYLOAD, Vec::new()),
            Op::run_payload_in_cgroup(&CONSTRUCTOR_TEST_PAYLOAD, Vec::new(), "a"),
            Op::wait_payload("constructor-test"),
            Op::wait_payload_in_cgroup("constructor-test", "a"),
            Op::kill_payload("constructor-test"),
            Op::kill_payload_in_cgroup("constructor-test", "a"),
            Op::freeze_cgroup("a"),
            Op::unfreeze_cgroup("a"),
            Op::capture_snapshot("constructor-test"),
            Op::watch_snapshot("kernel.constructor_test"),
            Op::write_kernel_hot(
                KernelTarget::symbol("constructor_test_symbol"),
                KernelValue::u64(0),
            ),
            Op::write_kernel_cold(
                KernelTarget::symbol("constructor_test_symbol"),
                KernelValue::u64(0),
            ),
            Op::read_kernel_hot(
                "constructor-test-hot",
                KernelTarget::symbol("constructor_test_symbol"),
                KernelValueWidth::u64(),
            ),
            Op::read_kernel_cold(
                "constructor-test-cold",
                KernelTarget::symbol("constructor_test_symbol"),
                KernelValueWidth::u32(),
            ),
            Op::attach_scheduler(&CONSTRUCTOR_TEST_SCHEDULER),
            Op::detach_scheduler(),
            Op::restart_scheduler(),
            Op::replace_scheduler(&CONSTRUCTOR_TEST_SCHEDULER),
        ];

        // Track which variants we observed. Adding a variant to `Op`
        // without a constructor call above leaves one slot `false`,
        // and adding a variant without a match arm below fails to
        // compile (no `_ =>` on purpose). **Bump the array size when
        // the bit_index high-water-mark in `OpKind::bit_index`
        // changes** — the runtime index check at `seen[idx] = true`
        // will panic if the new variant's index >= the array length.
        let mut seen = [false; 26];
        for op in &constructed {
            let idx = match op {
                Op::AddCgroup { .. } => 0,
                Op::AddCgroupDef { .. } => 1,
                Op::RemoveCgroup { .. } => 2,
                Op::SetCpuset { .. } => 3,
                Op::ClearCpuset { .. } => 4,
                Op::SwapCpusets { .. } => 5,
                Op::SpawnWorkers { .. } => 6,
                Op::StopCgroup { .. } => 7,
                Op::SetAffinity { .. } => 8,
                Op::SpawnHost { .. } => 9,
                Op::MoveAllTasks { .. } => 10,
                Op::RunPayload { .. } => 11,
                Op::WaitPayload { .. } => 12,
                Op::KillPayload { .. } => 13,
                Op::FreezeCgroup { .. } => 14,
                Op::UnfreezeCgroup { .. } => 15,
                Op::CaptureSnapshot { .. } => 16,
                Op::WatchSnapshot { .. } => 17,
                Op::WriteKernelHot { .. } => 18,
                Op::WriteKernelCold { .. } => 19,
                Op::ReadKernelHot { .. } => 20,
                Op::ReadKernelCold { .. } => 21,
                Op::AttachScheduler { .. } => 22,
                Op::DetachScheduler => 23,
                Op::RestartScheduler => 24,
                Op::ReplaceScheduler { .. } => 25,
            };
            seen[idx] = true;
        }

        let missing: Vec<usize> = seen
            .iter()
            .enumerate()
            .filter(|(_, hit)| !**hit)
            .map(|(i, _)| i)
            .collect();
        assert!(
            missing.is_empty(),
            "Op variant discriminants with no constructor coverage: {missing:?}. \
             Every Op variant must have a public constructor under impl Op per the \
             non_exhaustive convention documented on the enum.",
        );
    }

    #[test]
    fn cpuset_spec_constructor_coverage_is_exhaustive() {
        let constructed = [
            CpusetSpec::llc(0),
            CpusetSpec::numa(0),
            CpusetSpec::range(0.0, 1.0),
            CpusetSpec::disjoint(0, 2),
            CpusetSpec::overlap(0, 2, 0.25),
            CpusetSpec::exact([0usize]),
        ];
        let mut seen = [false; 6];
        for spec in &constructed {
            let idx = match spec {
                CpusetSpec::Llc(_) => 0,
                CpusetSpec::Numa(_) => 1,
                CpusetSpec::Range { .. } => 2,
                CpusetSpec::Disjoint { .. } => 3,
                CpusetSpec::Overlap { .. } => 4,
                CpusetSpec::Exact(_) => 5,
            };
            seen[idx] = true;
        }
        assert!(
            seen.iter().all(|s| *s),
            "every CpusetSpec variant must have a matching constructor, seen={seen:?}",
        );
    }

    // -- CgroupDef cgroup-v2 resource builders -----------------------

    /// `.cpu_quota_pct(50)` populates `cpu.max_quota_us = 50_000`
    /// with the default 100 ms period. Pins the percentage-to-µs
    /// conversion factor so a future refactor that shifts to
    /// nanoseconds trips this test.
    #[test]
    fn cgroup_def_cpu_quota_pct_uses_100ms_period_and_correct_quota() {
        let def = CgroupDef::named("cg_a").cpu_quota_pct(50);
        let cpu = def.cpu.expect("cpu_quota_pct must populate `cpu`");
        assert_eq!(cpu.max_quota_us, Some(50_000));
        assert_eq!(cpu.max_period_us, 100_000);
        assert!(cpu.weight.is_none(), "weight must remain unset");
    }

    /// `.cpu_quota(quota, period)` accepts arbitrary Durations and
    /// converts to microseconds.
    #[test]
    fn cgroup_def_cpu_quota_accepts_explicit_durations() {
        let def = CgroupDef::named("cg_a")
            .cpu_quota(Duration::from_micros(7_500), Duration::from_millis(10));
        let cpu = def.cpu.unwrap();
        assert_eq!(cpu.max_quota_us, Some(7_500));
        assert_eq!(cpu.max_period_us, 10_000);
    }

    /// `.cpu_unlimited()` clears the quota but preserves `weight`.
    /// Pins the "weight survives clear" guarantee documented on
    /// the builder.
    #[test]
    fn cgroup_def_cpu_unlimited_clears_quota_keeps_weight() {
        let def = CgroupDef::named("cg_a")
            .cpu_quota_pct(80)
            .cpu_weight(200)
            .cpu_unlimited();
        let cpu = def.cpu.unwrap();
        assert!(cpu.max_quota_us.is_none());
        assert_eq!(cpu.max_period_us, 100_000);
        assert_eq!(cpu.weight, Some(200));
    }

    /// All three memory builders compose into a single MemoryLimits.
    #[test]
    fn cgroup_def_memory_builders_compose() {
        let def = CgroupDef::named("cg_a")
            .memory_max(1_000_000)
            .memory_high(800_000)
            .memory_low(400_000);
        let m = def.memory.unwrap();
        assert_eq!(m.max, Some(1_000_000));
        assert_eq!(m.high, Some(800_000));
        assert_eq!(m.low, Some(400_000));
    }

    /// `.memory_unlimited()` resets every memory knob to None,
    /// undoing prior `.memory_max/high/low` calls.
    #[test]
    fn cgroup_def_memory_unlimited_clears_all_three() {
        let def = CgroupDef::named("cg_a")
            .memory_max(1_000_000)
            .memory_high(800_000)
            .memory_low(400_000)
            .memory_unlimited();
        let m = def.memory.unwrap();
        assert!(m.max.is_none());
        assert!(m.high.is_none());
        assert!(m.low.is_none());
    }

    /// `.io_weight(N)` populates the IoLimits.
    #[test]
    fn cgroup_def_io_weight_populates() {
        let def = CgroupDef::named("cg_a").io_weight(750);
        assert_eq!(def.io.unwrap().weight, Some(750));
    }

    /// `.cpuset_mems(nodes)` populates the new field without
    /// disturbing the cpuset.cpus side.
    #[test]
    fn cgroup_def_cpuset_mems_populates_independent_field() {
        let nodes: BTreeSet<usize> = [0usize, 1].into_iter().collect();
        let def = CgroupDef::named("cg_a").cpuset_mems(nodes.clone());
        assert_eq!(def.cpuset_mems, Some(nodes));
        assert!(def.cpuset.is_none());
    }

    // -- apply_setup wires builder values to CgroupOps calls ----------
    //
    // These tests drive `apply_setup` against a `MockCgroupOps` that
    // records every call into the existing `CgroupCall` enum, then
    // assert on the recorded sequence. The apply_setup site emits
    // the new resource-control writes between cpuset assignment and
    // worker spawn, so the tests pin both presence (the calls fire)
    // and ordering (cpu/memory/io land BEFORE move_tasks so the
    // limits are in effect when workers join).

    /// A bare CgroupDef with `.cpu_quota_pct(50)` records exactly
    /// one SetCpuMax call with the converted u64 quota and the
    /// default 100 ms period.
    #[test]
    fn apply_setup_records_set_cpu_max_for_cpu_quota_pct_builder() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        let defs = vec![CgroupDef::named("cg_cap").cpu_quota_pct(75)];
        apply_setup_test(&ctx, &mut state, &defs).expect("apply_setup must succeed");
        let calls = mock.calls();
        assert!(
            calls.contains(&CgroupCall::SetCpuMax(
                "cg_cap".to_string(),
                Some(75_000),
                100_000,
            )),
            "expected SetCpuMax(cg_cap, Some(75000), 100000); got {calls:?}",
        );
        cleanup_state(&mut state);
    }

    /// `.memory_max(N)` records SetMemoryMax(Some(N)) AND clears
    /// the unset high/low to None — the apply_setup loop emits all
    /// three writes whenever the `memory` field is `Some` so a
    /// prior cgroup's residue can't bleed through.
    #[test]
    fn apply_setup_records_three_memory_writes_when_memory_field_set() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        let defs = vec![CgroupDef::named("cg_mem").memory_max(1_000_000)];
        apply_setup_test(&ctx, &mut state, &defs).expect("apply_setup must succeed");
        let calls = mock.calls();
        let max_idx = calls
            .iter()
            .position(|c| matches!(c, CgroupCall::SetMemoryMax(n, _) if n == "cg_mem"))
            .expect("SetMemoryMax must fire");
        let high_idx = calls
            .iter()
            .position(|c| matches!(c, CgroupCall::SetMemoryHigh(n, _) if n == "cg_mem"))
            .expect("SetMemoryHigh must fire");
        let low_idx = calls
            .iter()
            .position(|c| matches!(c, CgroupCall::SetMemoryLow(n, _) if n == "cg_mem"))
            .expect("SetMemoryLow must fire");
        assert!(
            max_idx < high_idx && high_idx < low_idx,
            "memory writes must land in (max, high, low) order; got max={max_idx} high={high_idx} low={low_idx}",
        );
        // Specific values: max=Some, high=None (writes "max"),
        // low=None (writes "0") — pin both the SET and the
        // implicit-clear.
        assert!(calls.contains(&CgroupCall::SetMemoryMax(
            "cg_mem".to_string(),
            Some(1_000_000)
        )),);
        assert!(calls.contains(&CgroupCall::SetMemoryHigh("cg_mem".to_string(), None)));
        assert!(calls.contains(&CgroupCall::SetMemoryLow("cg_mem".to_string(), None)));
        cleanup_state(&mut state);
    }

    /// Ordering pin: every resource-control write MUST land before
    /// the first MoveTasks for the same cgroup so workers join an
    /// already-configured environment. Reverse ordering is a kernel
    /// race per Documentation/admin-guide/cgroup-v2.rst — tasks
    /// admitted before cpuset.mems is set may fail allocation per
    /// `cpuset_update_task_spread`.
    #[test]
    fn apply_setup_resource_writes_land_before_move_tasks() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        let mems: BTreeSet<usize> = [0usize].into_iter().collect();
        let defs = vec![
            CgroupDef::named("cg_full")
                .cpuset_mems(mems)
                .cpu_quota_pct(40)
                .cpu_weight(200)
                .memory_max(2_000_000)
                .io_weight(150),
        ];
        apply_setup_test(&ctx, &mut state, &defs).expect("apply_setup must succeed");
        let calls = mock.calls();
        let move_idx = calls
            .iter()
            .position(|c| matches!(c, CgroupCall::MoveTasks(n, _) if n == "cg_full"));
        // No workers here means no MoveTasks — but every resource
        // write must still appear, in the documented order. Pin
        // each kind's presence and then assert the inter-kind
        // ordering relative to the (possibly absent) MoveTasks.
        let kinds: Vec<usize> = calls
            .iter()
            .enumerate()
            .filter_map(|(i, c)| match c {
                CgroupCall::SetCpusetMems(n, _) if n == "cg_full" => Some(i),
                CgroupCall::SetCpuMax(n, _, _) if n == "cg_full" => Some(i),
                CgroupCall::SetCpuWeight(n, _) if n == "cg_full" => Some(i),
                CgroupCall::SetMemoryMax(n, _) if n == "cg_full" => Some(i),
                CgroupCall::SetMemoryHigh(n, _) if n == "cg_full" => Some(i),
                CgroupCall::SetMemoryLow(n, _) if n == "cg_full" => Some(i),
                CgroupCall::SetIoWeight(n, _) if n == "cg_full" => Some(i),
                _ => None,
            })
            .collect();
        assert!(
            kinds.len() >= 7,
            "expected at least 7 resource writes (mems + cpu.max + cpu.weight + 3 memory + io.weight); got {} ({calls:?})",
            kinds.len(),
        );
        if let Some(mi) = move_idx {
            assert!(
                kinds.iter().all(|k| *k < mi),
                "every resource write must precede MoveTasks; kinds={kinds:?} move_idx={mi}",
            );
        }
        cleanup_state(&mut state);
    }

    /// `cpu.weight = 0` (out of kernel range 1..=10000) MUST be
    /// rejected at apply_setup with a clear error message naming
    /// the cgroup and the offending value.
    #[test]
    fn apply_setup_rejects_cpu_weight_out_of_range() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        let defs = vec![CgroupDef::named("cg_bad").cpu_weight(0)];
        let err = apply_setup_test(&ctx, &mut state, &defs)
            .expect_err("apply_setup must reject weight=0");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("cg_bad") && msg.contains("cpu.weight"),
            "error must name cgroup and field; got: {msg}",
        );
        cleanup_state(&mut state);
    }

    /// `cpu.max` with `period_us = 0` MUST be rejected — the
    /// kernel writes `quota period` and divide-by-zero in the CFS
    /// scheduler is a guaranteed bug.
    #[test]
    fn apply_setup_rejects_cpu_max_period_zero() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        let defs =
            vec![CgroupDef::named("cg_bad").cpu_quota(Duration::from_millis(50), Duration::ZERO)];
        let err = apply_setup_test(&ctx, &mut state, &defs)
            .expect_err("apply_setup must reject period=0");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("cg_bad") && msg.contains("period"),
            "error must name cgroup and period; got: {msg}",
        );
        cleanup_state(&mut state);
    }

    // -- pids.max + memory.swap.max + cgroup.freeze ------------------

    /// `.memory_swap_max(bytes)` populates the swap_max field on the
    /// MemoryLimits inner; `.memory_swap_unlimited()` clears it back
    /// to None. Mirrors the cpu_quota_pct / cpu_unlimited convention.
    #[test]
    fn cgroup_def_memory_swap_max_builder_round_trip() {
        let d = CgroupDef::named("cg_a").memory_swap_max(2 * 1024 * 1024);
        assert_eq!(d.memory.as_ref().unwrap().swap_max, Some(2 * 1024 * 1024));

        let d = d.memory_swap_unlimited();
        assert_eq!(d.memory.as_ref().unwrap().swap_max, None);
    }

    /// `memory_swap_unlimited()` on a fresh CgroupDef (no prior
    /// `memory_*` calls) MUST NOT inflate `self.memory` from `None`
    /// to `Some(MemoryLimits::default())` — that would trigger 3
    /// unwanted apply_setup writes (`memory.max`, `memory.high`,
    /// `memory.low`) for a user who only asked to clear the swap
    /// cap. Pin the no-op short-circuit so a regression that drops
    /// the `if let Some` guard surfaces here.
    #[test]
    fn cgroup_def_memory_swap_unlimited_on_fresh_def_is_noop() {
        let d = CgroupDef::named("cg_a").memory_swap_unlimited();
        assert!(
            d.memory.is_none(),
            "memory_swap_unlimited() on a fresh CgroupDef must leave \
             self.memory == None; got: {:?}",
            d.memory,
        );
    }

    /// `memory_unlimited()` then `memory_swap_unlimited()` — the
    /// chain cln-preread flagged. memory_unlimited sets
    /// `self.memory = Some(MemoryLimits::default())` (already has
    /// `swap_max = None`); the subsequent memory_swap_unlimited
    /// must not redundantly recreate the MemoryLimits. After both
    /// calls, the inner is `Some(default)` with all four knobs
    /// `None`, mirroring memory_unlimited's intent. Pin both ends
    /// of the chain.
    #[test]
    fn cgroup_def_memory_unlimited_then_swap_unlimited_is_idempotent() {
        let d = CgroupDef::named("cg_a")
            .memory_unlimited()
            .memory_swap_unlimited();
        let m = d.memory.expect("memory_unlimited installs Some(default)");
        assert!(m.max.is_none());
        assert!(m.high.is_none());
        assert!(m.low.is_none());
        assert!(m.swap_max.is_none());
    }

    /// `apply_setup` against a CgroupDef with `memory_swap_unlimited()`
    /// alone (no other memory builders) must NOT emit any memory
    /// writes — the no-op short-circuit keeps `self.memory == None`,
    /// so the apply_setup `if let Some(ref mem)` block is skipped.
    /// Without the fix, a fresh `MemoryLimits::default()` would land
    /// in `self.memory` and fire `set_memory_max(None)` +
    /// `set_memory_high(None)` + `set_memory_low(None)` — a silent
    /// regression for tests that just want to clear a swap cap
    /// inherited from a base CgroupDef factory.
    #[test]
    fn apply_setup_memory_swap_unlimited_on_fresh_def_emits_no_memory_writes() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        let defs = vec![CgroupDef::named("cg_swap_clear").memory_swap_unlimited()];
        apply_setup_test(&ctx, &mut state, &defs).expect("apply_setup must succeed");
        let calls = mock.calls();
        assert!(
            !calls.iter().any(|c| matches!(
                c,
                CgroupCall::SetMemoryMax(_, _)
                    | CgroupCall::SetMemoryHigh(_, _)
                    | CgroupCall::SetMemoryLow(_, _)
                    | CgroupCall::SetMemorySwapMax(_, _)
            )),
            "memory_swap_unlimited() on a fresh CgroupDef must emit zero memory writes; got: {calls:?}"
        );
        cleanup_state(&mut state);
    }

    /// `.pids_max(n)` populates the pids field; `.pids_unlimited()`
    /// clears it. The pids field is independent of memory/cpu/io.
    #[test]
    fn cgroup_def_pids_max_builder_round_trip() {
        let d = CgroupDef::named("cg_a").pids_max(1024);
        assert_eq!(d.pids.as_ref().unwrap().max, Some(1024));

        let d = d.pids_unlimited();
        assert_eq!(d.pids.as_ref().unwrap().max, None);
    }

    /// apply_setup with `.memory_swap_max(N)` records exactly one
    /// SetMemorySwapMax call. swap_max defaults to None on a
    /// MemoryLimits constructed by `memory_max` alone — pin both
    /// shapes so a regression that always emits swap_max writes
    /// (or never emits them) surfaces here.
    #[test]
    fn apply_setup_records_set_memory_swap_max() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        let defs = vec![CgroupDef::named("cg_swap").memory_swap_max(4 * 1024 * 1024)];
        apply_setup_test(&ctx, &mut state, &defs).expect("apply_setup must succeed");
        let calls = mock.calls();
        assert!(
            calls.contains(&CgroupCall::SetMemorySwapMax(
                "cg_swap".to_string(),
                Some(4 * 1024 * 1024),
            )),
            "swap_max with bytes must record SetMemorySwapMax(Some(N)), got: {calls:?}",
        );
        cleanup_state(&mut state);

        // memory_max alone: swap_max stays None — apply_setup must
        // SKIP the SetMemorySwapMax write entirely. memory.swap.max
        // only exists on CONFIG_SWAP kernels; the per-knob
        // explicit-set semantics (write only when the user opted in)
        // keeps swap-disabled kernels viable for tests that just set
        // memory_max. This mirrors the pids block's "only write when
        // pids.max.is_some()" gate.
        let mock = MockCgroupOps::new();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        let defs = vec![CgroupDef::named("cg_nosw").memory_max(1_000_000)];
        apply_setup_test(&ctx, &mut state, &defs).expect("apply_setup must succeed");
        let calls = mock.calls();
        assert!(
            !calls.iter().any(|c| matches!(
                c,
                CgroupCall::SetMemorySwapMax(n, _) if n == "cg_nosw",
            )),
            "memory_max-only must NOT record SetMemorySwapMax (would \
             ENOENT on CONFIG_SWAP=n kernels); got: {calls:?}",
        );
        // Memory-write order pin: max → high → low. The ordering
        // matters because max must precede high so a high-above-max
        // user error surfaces with a clearer kernel error. swap_max
        // is excluded from the order check here because it only
        // emits when explicitly opted in (see test
        // `apply_setup_orders_memory_swap_max_after_low` for the
        // 4-write order pin).
        let max_idx = calls
            .iter()
            .position(|c| matches!(c, CgroupCall::SetMemoryMax(n, _) if n == "cg_nosw"))
            .expect("SetMemoryMax must fire");
        let high_idx = calls
            .iter()
            .position(|c| matches!(c, CgroupCall::SetMemoryHigh(n, _) if n == "cg_nosw"))
            .expect("SetMemoryHigh must fire");
        let low_idx = calls
            .iter()
            .position(|c| matches!(c, CgroupCall::SetMemoryLow(n, _) if n == "cg_nosw"))
            .expect("SetMemoryLow must fire");
        assert!(
            max_idx < high_idx && high_idx < low_idx,
            "memory writes must land in (max, high, low) order; \
             got max={max_idx} high={high_idx} low={low_idx}",
        );
        cleanup_state(&mut state);
    }

    /// When the user opts in via `.memory_swap_max(N)`, apply_setup
    /// emits SetMemorySwapMax AFTER the max/high/low triple. Pins the
    /// 4-write order across the full memory block so a regression
    /// that re-orders swap_max relative to the other knobs surfaces
    /// here. Distinct from `apply_setup_records_set_memory_swap_max`
    /// which pins presence/absence under the swap-disabled-kernel
    /// gate; this test pins ordering under the swap-enabled path.
    #[test]
    fn apply_setup_orders_memory_swap_max_after_low() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        let defs = vec![
            CgroupDef::named("cg_full_mem")
                .memory_max(2_000_000)
                .memory_high(1_500_000)
                .memory_low(500_000)
                .memory_swap_max(8 * 1024 * 1024),
        ];
        apply_setup_test(&ctx, &mut state, &defs).expect("apply_setup must succeed");
        let calls = mock.calls();
        let max_idx = calls
            .iter()
            .position(|c| matches!(c, CgroupCall::SetMemoryMax(n, _) if n == "cg_full_mem"))
            .expect("SetMemoryMax must fire");
        let high_idx = calls
            .iter()
            .position(|c| matches!(c, CgroupCall::SetMemoryHigh(n, _) if n == "cg_full_mem"))
            .expect("SetMemoryHigh must fire");
        let low_idx = calls
            .iter()
            .position(|c| matches!(c, CgroupCall::SetMemoryLow(n, _) if n == "cg_full_mem"))
            .expect("SetMemoryLow must fire");
        let swap_idx = calls
            .iter()
            .position(|c| matches!(c, CgroupCall::SetMemorySwapMax(n, _) if n == "cg_full_mem"))
            .expect("SetMemorySwapMax must fire when swap_max is opted in");
        assert!(
            max_idx < high_idx && high_idx < low_idx && low_idx < swap_idx,
            "memory writes must land in (max, high, low, swap_max) order; \
             got max={max_idx} high={high_idx} low={low_idx} swap={swap_idx}",
        );
        cleanup_state(&mut state);
    }

    /// apply_setup with `.pids_max(N)` records SetPidsMax(Some(N)).
    /// Without `pids` set, no SetPidsMax call is emitted.
    #[test]
    fn apply_setup_records_set_pids_max_only_when_set() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        let defs = vec![CgroupDef::named("cg_pids").pids_max(512)];
        apply_setup_test(&ctx, &mut state, &defs).expect("apply_setup must succeed");
        let calls = mock.calls();
        assert!(
            calls.contains(&CgroupCall::SetPidsMax("cg_pids".to_string(), Some(512))),
            "pids_max(N) must record SetPidsMax(Some(N)), got: {calls:?}",
        );
        cleanup_state(&mut state);

        // No pids — no SetPidsMax call.
        let mock = MockCgroupOps::new();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        let defs = vec![CgroupDef::named("cg_nopids").memory_max(1_000_000)];
        apply_setup_test(&ctx, &mut state, &defs).expect("apply_setup must succeed");
        let calls = mock.calls();
        assert!(
            !calls
                .iter()
                .any(|c| matches!(c, CgroupCall::SetPidsMax(_, _))),
            "no SetPidsMax expected when pids field is None, got: {calls:?}",
        );
        cleanup_state(&mut state);
    }

    /// `pids_max(0)` must be rejected at apply_setup with a clear
    /// error naming the cgroup and the offending value. A 0-limit
    /// cgroup silently halts every fork inside, including the
    /// futex-helper threads spawned by some WorkType variants —
    /// kernel accepts it but the workload would silently halt.
    #[test]
    fn apply_setup_rejects_pids_max_zero() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        let defs = vec![CgroupDef::named("cg_zero").pids_max(0)];
        let err = apply_setup_test(&ctx, &mut state, &defs)
            .expect_err("apply_setup must reject pids_max(0)");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("cg_zero") && msg.contains("pids.max"),
            "error must name cgroup and pids.max; got: {msg}",
        );
        // Pin the full diagnostic wording: the actionable hint
        // ("must be > 0" + "pids_unlimited") is what tells a user
        // to switch builders rather than rewrite their config.
        // Drift in either substring makes the diagnostic less
        // actionable; surface it here at test time, not at
        // user-debugging time.
        assert!(
            msg.contains("must be > 0"),
            "error must spell out the constraint; got: {msg}",
        );
        assert!(
            msg.contains("pids_unlimited"),
            "error must name the escape hatch (pids_unlimited()); got: {msg}",
        );
        cleanup_state(&mut state);
    }

    /// `Op::FreezeCgroup` against a cgroup the framework has never
    /// created routes through `ctx.cgroups.set_freeze` and
    /// surfaces the underlying kernel error as a step-level
    /// failure. The MockCgroupOps double records the call but
    /// returns Ok by default; pin the call sequence so a future
    /// regression that swallows the FreezeCgroup op (or routes it
    /// through a different code path that masks the error from a
    /// real cgroupfs ENOENT) trips here. The "real" fail-on-ENOENT
    /// path is exercised at the [`crate::cgroup`] layer's
    /// `set_freeze_returns_err_with_enoent_when_freeze_file_missing`
    /// test; this test pins the apply_ops dispatch shape.
    #[test]
    fn apply_ops_freeze_undefined_cgroup_dispatches_set_freeze() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        // The cgroup name "ghost_cg" is never declared via
        // CgroupDef or Op::AddCgroup. apply_ops still dispatches —
        // the framework does not gate FreezeCgroup on prior
        // creation; the kernel is the final authority on whether
        // the cgroup directory exists.
        apply_ops_test(&ctx, &mut state, &[Op::freeze_cgroup("ghost_cg")])
            .expect("apply_ops must dispatch FreezeCgroup even for an undeclared name");
        let calls = mock.calls();
        assert!(
            calls.contains(&CgroupCall::SetFreeze("ghost_cg".to_string(), true)),
            "FreezeCgroup must reach set_freeze regardless of declaration state, got: {calls:?}"
        );
    }

    /// `Op::FreezeCgroup` propagates the underlying ops error with
    /// an `Op::FreezeCgroup: cgroup '<name>'` context prefix so a
    /// failure dump names both the op and the offender. Inject an
    /// error from the mock and verify the context chain.
    #[test]
    fn apply_ops_freeze_propagates_set_freeze_error_with_context() {
        let mock = MockCgroupOps::new();
        // Index 0 is the SetFreeze call from the FreezeCgroup op.
        mock.fail_call_at(0, "kernel ENOENT — cgroup directory does not exist");
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        let err = apply_ops_test(&ctx, &mut state, &[Op::freeze_cgroup("ghost_cg")])
            .expect_err("set_freeze failure must surface as Err");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("Op::FreezeCgroup") && msg.contains("ghost_cg"),
            "error must name the op and the cgroup, got: {msg}"
        );
        assert!(
            msg.contains("ENOENT"),
            "error must propagate the underlying cause, got: {msg}"
        );
    }

    /// `Op::FreezeCgroup` dispatches to set_freeze(true);
    /// `Op::UnfreezeCgroup` to set_freeze(false). The mock records
    /// both shapes verbatim so a regression that swaps the bool
    /// surfaces here. Direct apply_ops dispatch — no workers needed.
    #[test]
    fn apply_ops_freeze_and_unfreeze_record_set_freeze() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        apply_ops_test(
            &ctx,
            &mut state,
            &[Op::freeze_cgroup("cg_x"), Op::unfreeze_cgroup("cg_x")],
        )
        .expect("freeze/unfreeze ops must succeed");
        let calls = mock.calls();
        assert!(
            calls.contains(&CgroupCall::SetFreeze("cg_x".to_string(), true)),
            "FreezeCgroup must dispatch SetFreeze(true), got: {calls:?}",
        );
        assert!(
            calls.contains(&CgroupCall::SetFreeze("cg_x".to_string(), false)),
            "UnfreezeCgroup must dispatch SetFreeze(false), got: {calls:?}",
        );
        // Sanity: the order must be (true, false) — the ops were
        // applied in that order.
        let true_idx = calls
            .iter()
            .position(|c| matches!(c, CgroupCall::SetFreeze(_, true)))
            .expect("found freeze");
        let false_idx = calls
            .iter()
            .position(|c| matches!(c, CgroupCall::SetFreeze(_, false)))
            .expect("found unfreeze");
        assert!(
            true_idx < false_idx,
            "freeze (true) must come before unfreeze (false): {calls:?}",
        );
        cleanup_state(&mut state);
    }

    /// `apply_setup` rejects `io.weight` outside the kernel's
    /// `1..=10000` range BEFORE issuing the syscall. The kernel's
    /// `cgrp_dfl_io_weight_write` parses via `kstrtouint` and
    /// returns -ERANGE for values outside the documented bound; the
    /// framework intercepts at apply-setup time so the operator
    /// gets a structured error naming the offending cgroup and
    /// value, rather than a raw ERANGE on cgroupfs.
    ///
    /// Pin both ends (0 and 10001) so a refactor that loosens the
    /// check in either direction surfaces here.
    #[test]
    fn apply_setup_rejects_io_weight_out_of_range() {
        for (weight, label) in [(0u16, "zero"), (10_001u16, "above-max")] {
            let mock = MockCgroupOps::new();
            let topo = mock_topo();
            let ctx = mock_ctx(&mock, &topo);
            let mut state = StepState::empty(&ctx);
            let defs = vec![CgroupDef::named("cg_io").io_weight(weight)];
            let err = apply_setup_test(&ctx, &mut state, &defs)
                .expect_err(&format!("io.weight={weight} ({label}) must reject"));
            let msg = format!("{err:#}");
            assert!(
                msg.contains("io.weight") && msg.contains("out of range"),
                "error must name the offending knob and constraint; got: {msg}",
            );
            assert!(
                msg.contains("cg_io"),
                "error must name the offending cgroup; got: {msg}",
            );
            // The reject must fire BEFORE the kernel write — no
            // SetIoWeight call should have been recorded.
            let calls = mock.calls();
            assert!(
                !calls
                    .iter()
                    .any(|c| matches!(c, CgroupCall::SetIoWeight(n, _) if n == "cg_io")),
                "rejected weight must not reach the cgroupfs write: {calls:?}",
            );
            cleanup_state(&mut state);
        }
    }

    /// Range boundary acceptance: `io.weight=1` and `io.weight=10000`
    /// (the kernel's documented endpoints) MUST be accepted by the
    /// framework's range gate. Pinned alongside the rejection test so
    /// a future refactor that flips a `<` to `<=` (or vice versa)
    /// breaks one of the two tests instead of silently widening or
    /// narrowing the accepted set.
    #[test]
    fn apply_setup_accepts_io_weight_range_endpoints() {
        for weight in [1u16, 10_000u16] {
            let mock = MockCgroupOps::new();
            let topo = mock_topo();
            let ctx = mock_ctx(&mock, &topo);
            let mut state = StepState::empty(&ctx);
            let defs = vec![CgroupDef::named("cg_io").io_weight(weight)];
            apply_setup_test(&ctx, &mut state, &defs).unwrap_or_else(|e| {
                panic!("io.weight={weight} (boundary) must be accepted: {e:#}")
            });
            let calls = mock.calls();
            assert!(
                calls.contains(&CgroupCall::SetIoWeight("cg_io".to_string(), weight)),
                "boundary weight must reach the cgroupfs write; got: {calls:?}",
            );
            cleanup_state(&mut state);
        }
    }

    /// Empty `works` substitution: a `CgroupDef` declared without a
    /// `.work(...)` or `.workload(...)` call falls back to a single
    /// default [`WorkSpec`](crate::workload::WorkSpec) (SpinWait, Normal, ctx.workers_per_cgroup
    /// workers) at apply-setup time. Pin the substitution by
    /// asserting that workers were spawned and migrated into the
    /// cgroup — without the fallback, no MoveTasks call would fire
    /// and the cgroup would sit empty.
    ///
    /// The tests above (e.g. `apply_setup_creates_cgroup_per_def`)
    /// drive `CgroupDef::named` directly without a workload; this
    /// test pins the fallback explicitly with a comment naming the
    /// invariant so a future refactor that drops the default-work
    /// substitution surfaces here with a clear failure message
    /// rather than a generic "no MoveTasks" symptom.
    #[test]
    fn apply_setup_substitutes_default_workspec_when_works_empty() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        // No .work(...) and no .workload(...) — empty `works` vec.
        let def = CgroupDef::named("cg_default_work");
        assert!(
            def.works.is_empty(),
            "test premise: CgroupDef without .work() must start with empty works",
        );
        apply_setup_test(&ctx, &mut state, &[def])
            .expect("apply_setup with default-work substitution must succeed");
        let calls = mock.calls();
        // The substitution surfaces as a real worker spawn → at
        // least one MoveTasks call into the cgroup with a non-zero
        // pid count. MoveTasks records the count (usize) rather
        // than the Vec; matching `count > 0` pins both the call
        // presence and the fact that the spawned workload had
        // workers to migrate.
        assert!(
            calls.iter().any(|c| matches!(
                c,
                CgroupCall::MoveTasks(name, count) if name == "cg_default_work" && *count > 0
            )),
            "default-WorkSpec substitution must spawn workers and migrate them into the \
             cgroup; without it the empty `works` would leave the cgroup taskless. \
             Got: {calls:?}",
        );
        cleanup_state(&mut state);
    }

    // -- pcomm coalescing tests -----------------------------------------
    //
    // [`CgroupDef::pcomm`] propagates `pcomm` to every WorkSpec in the
    // group AND records it on the CgroupDef itself. At apply_setup time,
    // pcomm-bearing WorkSpecs trigger the fork-then-thread spawn path
    // in [`WorkloadHandle::spawn`]: ONE container child is forked, its
    // comm is set to `pcomm`, then N thread workers are spawned inside.
    //
    // Verification at this layer:
    // - `move_tasks` receives a single PID per pcomm group (the
    //   container), not one PID per thread.
    // - `/proc/<container>/comm` carries `pcomm`, kernel-truncated at
    //   15 bytes (TASK_COMM_LEN-1 from include/linux/sched.h:325 — the
    //   write site is __set_task_comm in fs/exec.c:1075 which calls
    //   `min(strlen, sizeof(tsk->comm) - 1)`).
    // - Mixed pcomm/non-pcomm CgroupDefs in the same setup keep their
    //   move_tasks shapes distinct: pcomm group → 1 PID, non-pcomm
    //   group → N PIDs (one per worker fork).
    // - pcomm + num_workers=0 is rejected by `resolve_num_workers`
    //   like any other cgroup: a 0-worker cgroup emits no
    //   `WorkerReport`s, so every downstream assertion would
    //   vacuously pass. The pcomm path receives no exception —
    //   the rejection happens before pcomm dispatch runs.

    /// Read the `Tgid:` line from `/proc/<tid>/status`. Returns the
    /// pid_t of the thread group leader. Panics on read or parse
    /// failure — the live thread should always be observable; either
    /// failure indicates a wider problem (worker died early, /proc
    /// unmounted) outside this test's scope.
    fn read_status_tgid(tid: libc::pid_t) -> libc::pid_t {
        let status = std::fs::read_to_string(format!("/proc/{tid}/status"))
            .expect("/proc/<tid>/status must be readable for live thread");
        let line = status
            .lines()
            .find(|l| l.starts_with("Tgid:"))
            .expect("/proc/<tid>/status must include Tgid line");
        line.trim_start_matches("Tgid:")
            .trim()
            .parse()
            .expect("Tgid must be a parseable pid_t")
    }

    /// Read `/proc/<pid>/comm`. The kernel emits the comm bytes
    /// followed by a single newline (see `comm_show` in
    /// `fs/proc/base.c:1750`). The trailing newline is stripped.
    fn read_proc_comm(pid: libc::pid_t) -> String {
        let raw = std::fs::read_to_string(format!("/proc/{pid}/comm"))
            .expect("/proc/<pid>/comm must be readable for live task");
        raw.trim_end_matches('\n').to_string()
    }

    /// `CgroupDef::named(...).pcomm("X").workers(2)` propagates
    /// pcomm into the group's single (default) WorkSpec, and the
    /// resulting spawn forks ONE container process — observable
    /// here as a single PID delivered to `move_tasks`. Without
    /// fork-then-thread coalescing, `move_tasks` would receive
    /// 2 distinct fork-mode worker PIDs.
    #[test]
    fn apply_setup_pcomm_via_cgroup_def_forks_one_container() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        let defs = vec![CgroupDef::named("cg_pcomm").pcomm("leader").workers(2)];
        apply_setup_test(&ctx, &mut state, &defs).expect("pcomm apply_setup must succeed");
        let calls = mock.calls();
        // pcomm coalescing: exactly ONE PID is moved into the
        // cgroup (the container), not 2 (one per worker fork).
        assert!(
            calls.iter().any(|c| matches!(
                c,
                CgroupCall::MoveTasks(name, 1) if name == "cg_pcomm"
            )),
            "pcomm group must move exactly 1 PID (the container) into the cgroup; \
             got: {calls:?}",
        );
        cleanup_state(&mut state);
    }

    /// pcomm + per-thread comm coexist. The container holds `pcomm`
    /// as its comm; each worker thread sets its own comm via the
    /// post-spawn `prctl(PR_SET_NAME)`. Observable through
    /// `/proc/<leader>/comm == pcomm` while each per-thread file
    /// at `/proc/<leader>/task/<tid>/comm` carries the per-thread
    /// `comm` (except the leader-thread's own task entry, whose
    /// comm tracks `pcomm` since the leader called the
    /// container-wide prctl).
    ///
    /// `worker_pids()` for a pcomm group returns ONLY the leader
    /// pid (the parent has no per-thread tids exported across the
    /// process boundary). To verify per-thread comm we enumerate
    /// `/proc/<leader>/task/` directly: every directory entry is
    /// a kernel TID inside the container's tgid, and its `comm`
    /// file is the kernel-side authoritative per-thread comm.
    #[test]
    fn apply_setup_pcomm_with_per_thread_comm() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        let defs = vec![
            CgroupDef::named("cg_named")
                .pcomm("leader")
                .comm("worker")
                .workers(2),
        ];
        apply_setup_test(&ctx, &mut state, &defs).expect("pcomm + comm apply_setup must succeed");
        // Wait for the container's post-fork init (prctl for leader
        // comm) before reading /proc. Same race as truncation test.
        std::thread::sleep(Duration::from_millis(200));
        // Take handles so the workers are observable before drop.
        let mut handles = std::mem::take(&mut state.handles);
        assert_eq!(handles.len(), 1, "one CgroupDef → one handle");
        let (_name, handle) = handles
            .pop()
            .expect("apply_setup must have pushed a handle for cg_named");
        let pids = handle.worker_pids();
        assert_eq!(
            pids.len(),
            1,
            "pcomm handle must report exactly 1 pid (the leader); got {}",
            pids.len(),
        );
        // For pcomm groups, worker_pids()[0] IS the leader pid
        // directly (the parent never observes the per-thread tids).
        let leader_pid = pids[0];
        // Container's leader-thread comm is the pcomm value.
        assert_eq!(
            read_proc_comm(leader_pid),
            "leader",
            "/proc/<leader>/comm must equal pcomm",
        );
        // Wait briefly for thread workers to install their
        // per-thread comm via prctl in worker_main. 100 ms is
        // generous against scheduler jitter on contended hosts;
        // sleep is intentional: waits for prctl to propagate.
        std::thread::sleep(Duration::from_millis(100));
        // Enumerate /proc/<leader>/task/ — every directory entry
        // is a TID inside the container's tgid. Read each TID's
        // comm. The leader-thread's own task entry tracks the
        // container-wide prctl (== "leader"); every other TID is
        // a worker thread that ran worker_main's prctl == "worker".
        let task_dir = format!("/proc/{leader_pid}/task");
        let entries: Vec<libc::pid_t> = std::fs::read_dir(&task_dir)
            .expect("/proc/<leader>/task must be readable for live container")
            .flatten()
            .filter_map(|e| e.file_name().to_str().and_then(|n| n.parse().ok()))
            .collect();
        assert!(
            entries.len() >= 3,
            "leader pid {leader_pid} must have leader + 2 worker threads in /proc/<leader>/task; \
             observed {} entries: {entries:?}",
            entries.len(),
        );
        let mut leader_seen = false;
        let mut worker_seen = 0usize;
        for tid in entries {
            let tcomm = read_proc_comm(tid);
            if tid == leader_pid {
                assert_eq!(
                    tcomm, "leader",
                    "/proc/<leader>/task/<leader>/comm must equal pcomm; got {tcomm:?}",
                );
                leader_seen = true;
            } else {
                assert_eq!(
                    tcomm, "worker",
                    "/proc/<leader>/task/{tid}/comm must equal per-thread comm 'worker'; \
                     got {tcomm:?}",
                );
                worker_seen += 1;
            }
        }
        assert!(
            leader_seen,
            "leader's own task entry must appear in /proc/<leader>/task",
        );
        assert_eq!(
            worker_seen, 2,
            "must observe exactly 2 worker threads with per-thread comm 'worker'; \
             saw {worker_seen}",
        );
        // Drop the handle (reaps container + threads) and clean up
        // the rest of state.
        drop(handle);
        cleanup_state(&mut state);
    }

    /// Mixed cgroup behavior: one CgroupDef has `pcomm`, another
    /// does not. The pcomm group spawns via fork-then-thread
    /// (1 PID into its cgroup), the non-pcomm group spawns via
    /// normal fork mode (N PIDs into its cgroup). Pin both shapes
    /// so the implementer cannot regress either path.
    #[test]
    fn apply_setup_mixed_pcomm_and_non_pcomm_groups() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        let defs = vec![
            // Group 1: pcomm — fork-then-thread, one container PID.
            CgroupDef::named("cg_pcomm").pcomm("threaded").workers(2),
            // Group 2: no pcomm — normal fork mode, two PIDs.
            CgroupDef::named("cg_fork").workers(2),
        ];
        apply_setup_test(&ctx, &mut state, &defs).expect("mixed apply_setup must succeed");
        let calls = mock.calls();
        // pcomm group: 1 PID move.
        assert!(
            calls.iter().any(|c| matches!(
                c,
                CgroupCall::MoveTasks(name, 1) if name == "cg_pcomm"
            )),
            "cg_pcomm must move 1 PID (container only) into its cgroup; \
             got: {calls:?}",
        );
        // Non-pcomm group: 2 PID move.
        assert!(
            calls.iter().any(|c| matches!(
                c,
                CgroupCall::MoveTasks(name, 2) if name == "cg_fork"
            )),
            "cg_fork must move 2 PIDs (one per fork worker) into its cgroup; \
             got: {calls:?}",
        );
        cleanup_state(&mut state);
    }

    /// `pcomm` longer than 15 bytes is silently truncated by the
    /// kernel: `__set_task_comm` (fs/exec.c:1075) writes
    /// `min(strlen(buf), sizeof(tsk->comm) - 1)` bytes, with
    /// `TASK_COMM_LEN = 16` (include/linux/sched.h:325) so the
    /// write cap is exactly 15 bytes. Pin the boundary so a future
    /// caller passing a >15-byte pcomm sees the documented
    /// truncation, not an error.
    #[test]
    fn apply_setup_pcomm_kernel_truncation_at_15_bytes() {
        let long_name = "this_is_a_very_long_name";
        assert!(
            long_name.len() > 15,
            "test fixture must exceed TASK_COMM_LEN-1=15",
        );
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        let defs = vec![CgroupDef::named("cg_trunc").pcomm(long_name).workers(1)];
        apply_setup_test(&ctx, &mut state, &defs).expect("long-pcomm apply_setup must succeed");
        // The fork returns to the parent before the child executes
        // prctl(PR_SET_NAME). Wait for the child's post-fork init
        // (PDEATHSIG, setpgid, /proc/self/fd sweep, then prctl) to
        // complete. Matches tests_pcomm.rs:116 pattern.
        std::thread::sleep(Duration::from_millis(200));
        let mut handles = std::mem::take(&mut state.handles);
        let (_, handle) = handles.pop().expect("one handle");
        let pids = handle.worker_pids();
        let container_pid = read_status_tgid(pids[0]);
        let observed = read_proc_comm(container_pid);
        assert_eq!(
            observed.len(),
            15,
            "kernel must truncate pcomm to TASK_COMM_LEN-1=15 bytes; \
             observed length {} for {observed:?}",
            observed.len(),
        );
        assert_eq!(
            observed,
            &long_name[..15],
            "truncated comm must be the leading 15 bytes of pcomm input",
        );
        drop(handle);
        cleanup_state(&mut state);
    }

    /// `CgroupDef::pcomm("x").workers(0)` is rejected at
    /// `apply_setup` like any other 0-worker cgroup. The pcomm
    /// path receives no exception: `resolve_num_workers` runs
    /// before pcomm dispatch and rejects `num_workers=0`
    /// because a worker-less cgroup emits no [`WorkerReport`](crate::workload::WorkerReport)s,
    /// vacuously passing every downstream assertion. The
    /// rejection error names the cgroup and the offending
    /// field so a typo'd worker count surfaces at setup
    /// rather than as a silent green test.
    ///
    /// Pin the rejection here so a regression that silently
    /// no-ops the call (or forks an empty container) surfaces
    /// as a passing `apply_setup_test` instead of the expected
    /// `Err`.
    #[test]
    fn apply_setup_pcomm_with_zero_workers_is_rejected() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        let defs = vec![CgroupDef::named("cg_zero").pcomm("empty").workers(0)];
        let err = apply_setup_test(&ctx, &mut state, &defs)
            .expect_err("pcomm + 0 workers apply_setup must be rejected");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("cg_zero"),
            "rejection error must name the cgroup: {msg}",
        );
        assert!(
            msg.contains("num_workers=0"),
            "rejection error must name the offending field: {msg}",
        );
        // No spawn ever happened: no PIDs were moved into the
        // cgroup. The cgroup itself may have been created
        // (`add_cgroup_no_cpuset` runs before the WorkSpec
        // resolution loop) — only `MoveTasks` is forbidden.
        let calls = mock.calls();
        let any_move = calls.iter().any(|c| {
            matches!(
                c,
                CgroupCall::MoveTasks(name, _) if name == "cg_zero"
            )
        });
        assert!(
            !any_move,
            "rejection must short-circuit before any move_tasks call \
             into cg_zero; got: {calls:?}",
        );
        cleanup_state(&mut state);
    }

    /// `CgroupDef::workers_pct(0.5)` on a cgroup with no explicit
    /// cpuset resolves against the topology-usable cpuset and
    /// produces `ceil(usable_cpus * 0.5)` workers. The mock_topo's
    /// 4-CPU topology reserves the last CPU so usable=3 → ceil(3*0.5)=2.
    /// Pins the no-cpuset path through the apply_setup pre-resolution.
    #[test]
    fn workers_pct_no_cpuset_resolves_against_usable_topology() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        let def = CgroupDef::named("cg_p").workers_pct(0.5);
        apply_setup_test(&ctx, &mut state, std::slice::from_ref(&def)).unwrap();
        let handle = &state
            .handles
            .iter()
            .find(|(n, _)| n == "cg_p")
            .expect("workload spawned")
            .1;
        assert_eq!(
            handle.worker_pids().len(),
            2,
            "workers_pct(0.5) on usable=3 CPUs must resolve to ceil(3*0.5)=2 \
             workers; got {} workers",
            handle.worker_pids().len(),
        );
        cleanup_state(&mut state);
    }

    /// `CgroupDef::workers_pct(0.34)` on an LLC-restricted cpuset
    /// (size 4 here because the mock topology is 1 LLC × 4 cores)
    /// resolves to ceil(4 * 0.34) = 2 workers. Pins the with-cpuset
    /// path: workers_pct denominator is the resolved cpuset size,
    /// not the full topology.
    #[test]
    fn workers_pct_with_cpuset_resolves_against_cpuset_size() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        let def = CgroupDef::named("cg_p")
            .cpuset(CpusetSpec::Llc(0))
            .workers_pct(0.34);
        apply_setup_test(&ctx, &mut state, std::slice::from_ref(&def)).unwrap();
        let handle = &state
            .handles
            .iter()
            .find(|(n, _)| n == "cg_p")
            .expect("workload spawned")
            .1;
        assert_eq!(
            handle.worker_pids().len(),
            2,
            "workers_pct(0.34) on Llc(0)=4 CPUs must resolve to ceil(4*0.34)=2 \
             workers; got {} workers",
            handle.worker_pids().len(),
        );
        cleanup_state(&mut state);
    }

    /// `workers_pct(2.0)` accepts oversubscription. 4-CPU LLC * 2.0 = 8.
    /// Pins that >1.0 fractions are NOT rejected at apply time.
    #[test]
    fn workers_pct_above_one_accepts_oversubscription() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        let def = CgroupDef::named("cg_p")
            .cpuset(CpusetSpec::Llc(0))
            .workers_pct(2.0);
        apply_setup_test(&ctx, &mut state, std::slice::from_ref(&def)).unwrap();
        let handle = &state
            .handles
            .iter()
            .find(|(n, _)| n == "cg_p")
            .expect("workload spawned")
            .1;
        assert_eq!(
            handle.worker_pids().len(),
            8,
            "workers_pct(2.0) on Llc(0)=4 CPUs must resolve to ceil(4*2.0)=8 \
             workers (oversubscription); got {}",
            handle.worker_pids().len(),
        );
        cleanup_state(&mut state);
    }

    /// Setting both `workers(N)` and `workers_pct(p)` is rejected at
    /// apply-setup time regardless of the builder-call order. Pins
    /// BOTH orderings per adversary's mutex-asymmetry concern.
    #[test]
    fn workers_pct_then_workers_rejected_at_apply() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        let def = CgroupDef::named("cg_p")
            .cpuset(CpusetSpec::Llc(0))
            .workers_pct(0.5)
            .workers(2);
        let err = apply_setup_test(&ctx, &mut state, std::slice::from_ref(&def))
            .expect_err("workers_pct + workers must be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("workers_pct") && msg.contains("workers(2)"),
            "error must name both workers and workers_pct: {msg}",
        );
        cleanup_state(&mut state);
    }

    #[test]
    fn workers_then_workers_pct_rejected_at_apply() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        let def = CgroupDef::named("cg_p")
            .cpuset(CpusetSpec::Llc(0))
            .workers(2)
            .workers_pct(0.5);
        let err = apply_setup_test(&ctx, &mut state, std::slice::from_ref(&def))
            .expect_err("workers + workers_pct must be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("workers_pct") && msg.contains("workers(2)"),
            "error must name both workers and workers_pct: {msg}",
        );
        cleanup_state(&mut state);
    }

    /// `CgroupDef::workers_pct(p)` stores the fraction on
    /// `works[0].workers_pct` without pre-resolving, and leaves
    /// `works[0].num_workers` unset. Pins the construction-time
    /// invariant: resolution is deferred to apply-setup (which has
    /// access to the cpuset size). A future regression that
    /// eagerly resolved at construction would silently break the
    /// apply-time-resolution contract; this test catches it.
    #[test]
    fn workers_pct_construction_stores_pct_without_resolving() {
        let def = CgroupDef::named("cg_p").workers_pct(0.5);
        let work = &def.works[0];
        assert_eq!(
            work.workers_pct,
            Some(0.5),
            "workers_pct must be stored verbatim at construction; got {:?}",
            work.workers_pct,
        );
        assert_eq!(
            work.num_workers, None,
            "num_workers must be left unset at construction (apply-setup resolves); got {:?}",
            work.num_workers,
        );
    }

    /// `workers_pct` uses ceil() for the cpuset→worker count
    /// resolution. Pin the rounding across four cases covering the
    /// integer / fractional / just-above / just-below boundaries:
    /// an exact integer product stays at that integer; any non-zero
    /// remainder rounds UP regardless of which side of the half it
    /// falls on. Catches a future regression to round() or floor()
    /// rounding modes that would produce off-by-one worker counts at
    /// boundary fractions (`round` and `floor` differ from `ceil`
    /// in opposite directions, so these four cases pin ceil
    /// uniquely).
    #[test]
    fn workers_pct_rounding_is_ceil_not_round_or_floor() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);

        // Exact integer product: 4 * 0.5 = 2.0 (exact in IEEE 754
        // because 0.5 = 2^-1 is exactly representable) → ceil(2.0) = 2.
        // round and floor also give 2 here; this case doesn't
        // distinguish ceil, it's a baseline.
        let mut state = StepState::empty(&ctx);
        let def_exact = CgroupDef::named("cg_exact")
            .cpuset(CpusetSpec::Llc(0))
            .workers_pct(0.5);
        apply_setup_test(&ctx, &mut state, std::slice::from_ref(&def_exact)).unwrap();
        let handle = &state
            .handles
            .iter()
            .find(|(n, _)| n == "cg_exact")
            .expect("workload spawned")
            .1;
        assert_eq!(
            handle.worker_pids().len(),
            2,
            "workers_pct(0.5) on 4 CPUs (exact 2.0) must produce 2 workers",
        );
        cleanup_state(&mut state);

        // Mid-fractional product: 4 * 0.6 ≈ 2.3999... → ceil = 3.
        // round (nearest) gives 2; floor gives 2. ceil gives 3.
        // Distinguishes ceil from round.
        let mut state = StepState::empty(&ctx);
        let def_mid = CgroupDef::named("cg_mid")
            .cpuset(CpusetSpec::Llc(0))
            .workers_pct(0.6);
        apply_setup_test(&ctx, &mut state, std::slice::from_ref(&def_mid)).unwrap();
        let handle = &state
            .handles
            .iter()
            .find(|(n, _)| n == "cg_mid")
            .expect("workload spawned")
            .1;
        assert_eq!(
            handle.worker_pids().len(),
            3,
            "workers_pct(0.6) on 4 CPUs (≈2.4) must ceil to 3 workers; round (2) or floor (2) would be wrong",
        );
        cleanup_state(&mut state);

        // Just above an integer: 4 * 0.51 ≈ 2.04 → ceil = 3. Pins
        // that ANY non-zero remainder rounds up, not just near-half.
        let mut state = StepState::empty(&ctx);
        let def_just_over = CgroupDef::named("cg_over")
            .cpuset(CpusetSpec::Llc(0))
            .workers_pct(0.51);
        apply_setup_test(&ctx, &mut state, std::slice::from_ref(&def_just_over)).unwrap();
        let handle = &state
            .handles
            .iter()
            .find(|(n, _)| n == "cg_over")
            .expect("workload spawned")
            .1;
        assert_eq!(
            handle.worker_pids().len(),
            3,
            "workers_pct(0.51) on 4 CPUs (≈2.04) must ceil to 3 workers; round (2) or floor (2) would be wrong",
        );
        cleanup_state(&mut state);

        // Just below an integer: 4 * 0.49 ≈ 1.96 → ceil = 2.
        // floor gives 1; round gives 2. Distinguishes ceil/round
        // from floor — completes the rounding-mode coverage.
        let mut state = StepState::empty(&ctx);
        let def_just_under = CgroupDef::named("cg_under")
            .cpuset(CpusetSpec::Llc(0))
            .workers_pct(0.49);
        apply_setup_test(&ctx, &mut state, std::slice::from_ref(&def_just_under)).unwrap();
        let handle = &state
            .handles
            .iter()
            .find(|(n, _)| n == "cg_under")
            .expect("workload spawned")
            .1;
        assert_eq!(
            handle.worker_pids().len(),
            2,
            "workers_pct(0.49) on 4 CPUs (≈1.96) must ceil to 2 workers; floor (1) would be wrong",
        );
        cleanup_state(&mut state);
    }

    /// Setup-spawned workers (workers from apply_setup-time
    /// `workers_pct` resolution) keep their pid set across
    /// subsequent `Op::SetCpuset` cpuset changes. Pins that
    /// `Op::SetCpuset`'s apply arm at mod.rs:2063-2074 is NOT a
    /// `resolve_workers_pct` call site — the arm validates +
    /// resolves the CpusetSpec, calls `ctx.cgroups.set_cpuset`,
    /// and records the new cpuset via `state.record_cpuset`, but
    /// touches no WorkSpec / handle state.
    ///
    /// `resolve_workers_pct` does have TWO call sites overall
    /// (apply_setup at mod.rs:1772 and Op::SpawnWorkers at mod.rs:2100),
    /// so a test author who issues an `Op::SpawnWorkers` AFTER an
    /// `Op::SetCpuset` will get fresh resolution against the
    /// then-current cpuset — that's the Op::SpawnWorkers integration
    /// layer's responsibility and is verified by `op_spawn_*` tests,
    /// not here. This test catches a future regression that adds
    /// re-resolution INTO the Op::SetCpuset apply branch
    /// (re-counting apply-setup workers when the cpuset narrows).
    ///
    /// The test drives Op::SetCpuset through `apply_ops_test` (the
    /// real Op-dispatch wrapper) instead of calling
    /// `ctx.cgroups.set_cpuset` directly — that distinction matters
    /// because a regression that added `resolve_workers_pct` inside
    /// the Op match arm would NOT be caught by a direct set_cpuset
    /// call that bypasses dispatch.
    #[test]
    fn workers_pct_setup_workers_survive_op_setcpuset_narrowing() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        let def = CgroupDef::named("cg_stable")
            .cpuset(CpusetSpec::Llc(0))
            .workers_pct(0.5);
        apply_setup_test(&ctx, &mut state, std::slice::from_ref(&def)).unwrap();
        let initial_count = state
            .handles
            .iter()
            .find(|(n, _)| n == "cg_stable")
            .expect("workload spawned")
            .1
            .worker_pids()
            .len();
        assert_eq!(
            initial_count, 2,
            "baseline: workers_pct(0.5) on Llc(0)=4 CPUs → ceil(4*0.5)=2 workers",
        );

        // Drive Op::SetCpuset through the real apply_ops dispatch
        // (NOT just ctx.cgroups.set_cpuset, which would bypass the
        // Op match arm where a regression might add re-resolution).
        let narrower: std::collections::BTreeSet<usize> = [0usize, 1].into_iter().collect();
        apply_ops_test(
            &ctx,
            &mut state,
            &[Op::SetCpuset {
                cgroup: "cg_stable".into(),
                cpus: CpusetSpec::Exact(narrower.clone()),
            }],
        )
        .expect("Op::SetCpuset applies");

        // Verify the narrowing actually took effect — without this
        // assertion, a silently-no-op set_cpuset would make the
        // worker-stability claim trivially true. StepState's
        // `cpusets` HashMap is the step-local cpuset bookkeeping
        // that Op::SetCpuset's `state.record_cpuset` call writes to.
        assert_eq!(
            state
                .cpusets
                .get("cg_stable")
                .expect("cg_stable has recorded cpuset"),
            &narrower,
            "Op::SetCpuset must persist the narrower set in state.cpusets via record_cpuset",
        );

        let after_count = state
            .handles
            .iter()
            .find(|(n, _)| n == "cg_stable")
            .expect("workload still present")
            .1
            .worker_pids()
            .len();
        assert_eq!(
            after_count, initial_count,
            "Op::SetCpuset apply arm must NOT re-resolve workers_pct; \
             setup-spawned worker count must remain {initial_count}; got {after_count}",
        );
        cleanup_state(&mut state);
    }

    /// Pathological `workers_pct` values rejected at construction:
    /// NaN, INFINITY, negative values, and zero all panic via
    /// `CgroupDef::workers_pct`'s `assert!` at types.rs:1097-1100.
    /// Pin all four rejection paths so a future regression that
    /// loosens the gate (e.g. accepts NaN as "use default") fails
    /// here loudly.
    #[test]
    fn workers_pct_pathological_finite_rejected_at_construction() {
        // Non-finite NaN → CgroupDef::workers_pct panics; std::panic::catch_unwind verifies.
        let nan_panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = CgroupDef::named("cg_nan").workers_pct(f64::NAN);
        }));
        assert!(
            nan_panic.is_err(),
            "CgroupDef::workers_pct(NaN) must panic at construction",
        );

        let inf_panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = CgroupDef::named("cg_inf").workers_pct(f64::INFINITY);
        }));
        assert!(
            inf_panic.is_err(),
            "CgroupDef::workers_pct(INFINITY) must panic at construction",
        );

        let neg_panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = CgroupDef::named("cg_neg").workers_pct(-1.0);
        }));
        assert!(
            neg_panic.is_err(),
            "CgroupDef::workers_pct(-1.0) must panic at construction",
        );

        let zero_panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = CgroupDef::named("cg_zero").workers_pct(0.0);
        }));
        assert!(
            zero_panic.is_err(),
            "CgroupDef::workers_pct(0.0) must panic at construction",
        );
    }

    /// A very large finite `workers_pct` (e.g. `1e100`) passes the
    /// finite + positive construction gate but produces `usize::MAX`
    /// when `resolve_workers_pct` evaluates
    /// `(cpuset_cpus as f64 * pct).ceil() as usize` — Rust's
    /// saturating float-to-int cast (RFC 2484, stable since 1.45)
    /// clamps any finite f64 exceeding the integer range to the
    /// bound. The product `4.0 * 1e100 = 4e100` is finite (well
    /// below `f64::MAX ≈ 1.798e308`) but far exceeds `usize::MAX`
    /// (~`1.844e19` on 64-bit), so the cast saturates.
    ///
    /// Calls `resolve_workers_pct` directly on a constructed
    /// [`WorkSpec`](crate::workload::WorkSpec) — pins the FRAMEWORK contract (current behavior
    /// returns `Ok(num_workers=Some(usize::MAX))`). A future
    /// regression that added a saturation guard
    /// (e.g. `if scaled == usize::MAX { bail!("too large") }`)
    /// would flip this to `Err` and trip the test. The spawn path
    /// is NOT exercised — spawning `usize::MAX` workers would hang
    /// the host. See follow-up task for the OOM-risk doc warning
    /// the framework currently lacks on `workers_pct`.
    #[test]
    fn workers_pct_pathological_finite_large_saturates_usize() {
        let work = crate::workload::WorkSpec::default().workers_pct(1e100);
        let resolved = work
            .resolve_workers_pct(4, "cg_saturate")
            .expect("current framework does not gate against usize::MAX saturation");
        assert_eq!(
            resolved.num_workers,
            Some(usize::MAX),
            "extreme pct saturates `num_workers` to `usize::MAX` per Rust's saturating \
             float-to-int `as` cast (RFC 2484); got {:?}",
            resolved.num_workers,
        );
    }

    /// Empty cpuset + MULTIPLE [`WorkSpec`](crate::workload::WorkSpec)s with distinct `workers_pct`
    /// values: the diagnostic must enumerate ALL pct values, not just
    /// the first. An earlier diagnostic used
    /// `find_map(|w| w.workers_pct)` which dropped subsequent pcts and
    /// hid that other WorkSpecs in the cgroup also had pct configured.
    #[test]
    fn workers_pct_empty_cpuset_multi_workspec_lists_all_pcts() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        let def = CgroupDef::named("cg_multi")
            .cpuset(CpusetSpec::Exact(std::collections::BTreeSet::new()))
            .workers_pct(0.3)
            .work(crate::workload::WorkSpec::default().workers_pct(0.7))
            .work(crate::workload::WorkSpec::default().workers_pct(0.5));
        let err = apply_setup_test(&ctx, &mut state, std::slice::from_ref(&def))
            .expect_err("multi-workspec workers_pct on empty cpuset must reject");
        let msg = format!("{err}");
        assert!(
            msg.contains("0.3") && msg.contains("0.7") && msg.contains("0.5"),
            "diagnostic must name ALL configured workers_pct values, not just the first: {msg}",
        );
        assert!(
            msg.contains("cpuset of 0"),
            "diagnostic must still name the empty cpuset size: {msg}",
        );
        cleanup_state(&mut state);
    }

    /// Empty cpuset + a single [`WorkSpec`](crate::workload::WorkSpec) that sets BOTH `workers(N)`
    /// AND `workers_pct(p)`: the framework must emit a dual-set-specific
    /// bail (the more fundamental misconfiguration) rather than letting
    /// validate's empty-Exact mask preempt it OR the workers_pct-only
    /// empty-cpuset diagnostic claim "would resolve to 0 workers" (which
    /// is misleading when workers(N) explicitly sets the count). The
    /// operator must pick one of `workers` or `workers_pct` before the
    /// empty-cpuset question is meaningful. Case (1) of the
    /// empty-cpuset handling in apply_setup surfaces this bail inline
    /// with the "BOTH workers ... empty cpuset would otherwise mask"
    /// wording.
    #[test]
    fn workers_pct_empty_cpuset_dual_set_bails_with_dedicated_error() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        let def = CgroupDef::named("cg_both")
            .cpuset(CpusetSpec::Exact(std::collections::BTreeSet::new()))
            .workers(2)
            .workers_pct(0.5);
        let err = apply_setup_test(&ctx, &mut state, std::slice::from_ref(&def))
            .expect_err("workers + workers_pct on empty cpuset must reject");
        let msg = format!("{err}");
        assert!(
            msg.contains("BOTH workers"),
            "dual-set error must fire first; got the empty-cpuset diagnostic instead: {msg}",
        );
        assert!(
            !msg.contains("cpuset of 0"),
            "workers_pct-only empty-cpuset diagnostic must NOT preempt the more fundamental dual-set error: {msg}",
        );
        assert!(
            msg.contains("empty cpuset would otherwise mask"),
            "dual-set bail must include the case-(1)-specific trailing context that \
             explains why this fired at apply_setup rather than at the deeper resolve \
             path: {msg}",
        );
        cleanup_state(&mut state);
    }

    /// A cgroup whose `cpuset_spec` resolves to an empty CPU set
    /// AND that does NOT use `workers_pct` must still bail at
    /// apply_setup — silently writing an empty mask would leave the
    /// cgroup with no CPUs assigned, downstream worker spawns would
    /// fail or produce vacuous assertions, and the operator would
    /// have no signal that they misconfigured the spec. Uses
    /// `Range { 0.0, 0.1 }` on a 4-CPU mock topology: validate
    /// accepts because `0.0 < 0.1` and both fracs are in `[0, 1]`,
    /// but resolve computes `start = 4 * 0.0 = 0` and `end =
    /// (4 * 0.1) as usize = 0`, yielding an empty slice. This is
    /// the canonical "passes validate but resolves to empty" case
    /// — `Range { 0.0, 0.0 }` would be rejected by validate's
    /// `start_frac >= end_frac` guard at types.rs:2149, so the
    /// fraction must be small but non-zero to thread the needle.
    /// Distinct from the `workers_pct`-driven empty-cpuset bails:
    /// no fraction is set, so the diagnostic should cite the
    /// cpuset_spec itself rather than a fraction-on-zero-CPUs
    /// framing.
    #[test]
    fn empty_resolved_cpuset_without_workers_pct_bails_in_apply_setup() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        let def = CgroupDef::named("cg_empty_range").cpuset(CpusetSpec::Range {
            start_frac: 0.0,
            end_frac: 0.1,
        });
        let err = apply_setup_test(&ctx, &mut state, std::slice::from_ref(&def))
            .expect_err("empty-resolved cpuset must reject even without workers_pct");
        let msg = format!("{err}");
        assert!(
            msg.contains("cg_empty_range"),
            "diagnostic must name the cgroup: {msg}",
        );
        assert!(
            msg.contains("resolved to 0 CPU(s)"),
            "diagnostic must name the zero-CPU resolution: {msg}",
        );
        assert!(
            !msg.contains("workers_pct"),
            "diagnostic must NOT cite workers_pct when none is set; \
             that would mis-direct the operator to a knob they didn't \
             configure: {msg}",
        );
        cleanup_state(&mut state);
    }

    /// `Op::SetCpuset` mid-scenario must also bail when the new spec
    /// resolves to an empty CPU set, symmetric with apply_setup.
    /// Silently re-masking a live cgroup to empty would leave its
    /// running workers without CPUs and downstream assertions would
    /// vacuously pass. The diagnostic must cite the target cgroup
    /// and the spec that resolved to empty so the operator knows
    /// which mid-scenario narrow produced the empty resolution.
    #[test]
    fn op_set_cpuset_narrow_to_empty_bails() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        // Establish a live cgroup with a valid cpuset first.
        apply_setup_test(
            &ctx,
            &mut state,
            std::slice::from_ref(&CgroupDef::named("cg_narrow").cpuset(CpusetSpec::Llc(0))),
        )
        .unwrap();
        // Now try to narrow it via Op::SetCpuset to an empty range.
        // `Range { 0.0, 0.1 }` passes validate (start < end, both in
        // [0, 1]) but resolves empty on a 4-CPU topology: end =
        // (4 * 0.1) as usize = 0, so the slice is [0..0] = empty.
        let err = apply_ops_test(
            &ctx,
            &mut state,
            &[Op::SetCpuset {
                cgroup: std::borrow::Cow::Borrowed("cg_narrow"),
                cpus: CpusetSpec::Range {
                    start_frac: 0.0,
                    end_frac: 0.1,
                },
            }],
        )
        .expect_err("Op::SetCpuset narrowing to empty must reject");
        let msg = format!("{err}");
        assert!(
            msg.contains("cg_narrow"),
            "diagnostic must name the target cgroup: {msg}",
        );
        assert!(
            msg.contains("resolved to 0 CPU(s)"),
            "diagnostic must name the zero-CPU resolution: {msg}",
        );
        assert!(
            msg.contains("Op::SetCpuset"),
            "diagnostic must identify the Op layer so the operator \
             knows this came from a mid-scenario narrow, not setup: \
             {msg}",
        );
        assert!(
            msg.contains("Op::ClearCpuset"),
            "diagnostic must point the operator at the right \
             primitive for the 'release cpuset restriction' intent \
             so a regression that drops the Op::ClearCpuset \
             direction (leading users to the workaround \
             `Range {{ 0.0, 1.0 }}` instead) is caught: {msg}",
        );
        cleanup_state(&mut state);
    }

    /// `workers_pct` against an empty cpuset (Exact({})) resolves to
    /// 0 workers and bails with a diagnostic that names the cpuset
    /// size and the requested fraction. Pin per adversary's V1+V3
    /// "loud reject with diagnostic" caveat — the message must carry
    /// the diagnostic fields so a future refactor that drops them is
    /// caught here, not by a confused user.
    #[test]
    fn workers_pct_empty_cpuset_rejects_with_diagnostic() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        let def = CgroupDef::named("cg_e")
            .cpuset(CpusetSpec::Exact(std::collections::BTreeSet::new()))
            .workers_pct(0.9);
        let err = apply_setup_test(&ctx, &mut state, std::slice::from_ref(&def))
            .expect_err("workers_pct on empty cpuset must reject");
        let msg = format!("{err}");
        assert!(
            msg.contains("workers_pct(0.9)") && msg.contains("cpuset of 0"),
            "diagnostic must name the requested fraction AND cpuset size: {msg}",
        );
        cleanup_state(&mut state);
    }

    /// `Op::SpawnWorkers` with `WorkSpec::workers_pct` resolves against the
    /// cgroup's currently-recorded cpuset, mirroring the apply_setup
    /// path. Pin so a future regression that drops the workers_pct
    /// pre-resolution from Op::SpawnWorkers (silently falling back to
    /// `ctx.workers_per_cgroup` and ignoring the user's fraction) is
    /// caught.
    #[test]
    fn op_spawn_workers_pct_resolves_against_cgroup_cpuset() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        // Set up an empty cgroup with an explicit cpuset first.
        apply_setup_test(
            &ctx,
            &mut state,
            std::slice::from_ref(&CgroupDef::named("cg_spawn").cpuset(CpusetSpec::Llc(0))),
        )
        .unwrap();
        // Drop the apply_setup default-spawned workload so the Spawn
        // we issue below is the only handle for cg_spawn.
        state.handles.clear();
        // Now Spawn a WorkSpec that uses workers_pct(0.5).
        // Llc(0) = 4 CPUs → ceil(4 * 0.5) = 2 workers.
        let work = crate::workload::WorkSpec::default().workers_pct(0.5);
        apply_ops_test(
            &ctx,
            &mut state,
            &[Op::SpawnWorkers {
                cgroup: std::borrow::Cow::Borrowed("cg_spawn"),
                work,
            }],
        )
        .unwrap();
        let handle = &state
            .handles
            .iter()
            .find(|(n, _)| n == "cg_spawn")
            .expect("Op::SpawnWorkers workload registered")
            .1;
        assert_eq!(
            handle.worker_pids().len(),
            2,
            "Op::SpawnWorkers workers_pct(0.5) on Llc(0)=4 must resolve to 2 workers; \
             got {}",
            handle.worker_pids().len(),
        );
        cleanup_state(&mut state);
    }

    /// `Op::SpawnWorkers` with BOTH workers and workers_pct set is rejected
    /// the same way apply_setup rejects it — the resolution helper
    /// is shared so the diagnostic is identical.
    #[test]
    fn op_spawn_workers_pct_dual_set_rejected() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        apply_setup_test(
            &ctx,
            &mut state,
            std::slice::from_ref(&CgroupDef::named("cg_x").cpuset(CpusetSpec::Llc(0))),
        )
        .unwrap();
        state.handles.clear();
        let work = crate::workload::WorkSpec::default()
            .workers(2)
            .workers_pct(0.5);
        let err = apply_ops_test(
            &ctx,
            &mut state,
            &[Op::SpawnWorkers {
                cgroup: std::borrow::Cow::Borrowed("cg_x"),
                work,
            }],
        )
        .expect_err("Op::SpawnWorkers dual-set must reject");
        let msg = format!("{err}");
        assert!(
            msg.contains("workers_pct") && msg.contains("workers(2)"),
            "Op::SpawnWorkers diagnostic must name both knobs: {msg}",
        );
        cleanup_state(&mut state);
    }

    /// `Ctx::cpuset_cpus(&spec)` returns the size of
    /// `spec.resolve(ctx)` for every CpusetSpec variant. Pinned via a
    /// single-pass equivalence check across all variants so a future
    /// CpusetSpec variant added without updating cpuset_cpus stays
    /// detectable.
    #[test]
    fn ctx_cpuset_cpus_matches_resolve_len() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let specs = [
            CpusetSpec::Llc(0),
            CpusetSpec::Numa(0),
            CpusetSpec::Range {
                start_frac: 0.0,
                end_frac: 0.5,
            },
            CpusetSpec::Disjoint { index: 0, of: 2 },
            CpusetSpec::Overlap {
                index: 0,
                of: 2,
                frac: 0.5,
            },
            CpusetSpec::Exact([0usize, 1, 2].iter().copied().collect()),
        ];
        for spec in &specs {
            assert_eq!(
                ctx.cpuset_cpus(spec),
                spec.resolve(&ctx).len(),
                "ctx.cpuset_cpus drift on {spec:?}",
            );
        }
    }

    // -----------------------------------------------------------------
    // Kernel-op integration: 4 Op::*Kernel* arms dispatch through
    // apply_ops + a thread-local SnapshotBridge kernel-op callback.
    // -----------------------------------------------------------------

    /// `Op::WriteKernelHot` dispatched via `apply_ops` invokes the
    /// installed bridge kernel-op callback with the correct mode +
    /// direction + entries, and the bridge's drain log records the
    /// reply. Pins the executor arm's mapping from variant fields
    /// to wire payload — a regression that flipped Hot↔Cold or
    /// Write↔Read or dropped a write entry surfaces here.
    #[test]
    fn apply_ops_write_kernel_hot_dispatches_via_bridge() {
        use std::sync::Arc;
        let captured = Arc::new(std::sync::Mutex::new(
            None::<crate::vmm::wire::KernelOpRequestPayload>,
        ));
        let captured_clone = captured.clone();
        let kernel_op_cb: crate::scenario::snapshot::KernelOpCallback = Arc::new(move |req| {
            *captured_clone.lock().unwrap() = Some(req.clone());
            crate::vmm::wire::KernelOpReplyPayload {
                request_id: req.request_id,
                success: true,
                reason: String::new(),
                read_values: vec![],
            }
        });
        let bridge = crate::scenario::snapshot::SnapshotBridge::new(Arc::new(|_| None))
            .with_kernel_op(kernel_op_cb);
        let bridge_clone = bridge.clone();
        let _bg = bridge.set_thread_local();
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        let ops = vec![Op::write_kernel_hot(
            KernelTarget::symbol("test_field"),
            KernelValue::u64(42),
        )];
        apply_ops_test(&ctx, &mut state, &ops).expect("WriteKernelHot must dispatch");
        let req = captured.lock().unwrap().take().expect("callback must fire");
        assert_eq!(req.mode, crate::vmm::wire::KernelOpMode::Hot);
        assert_eq!(req.direction, crate::vmm::wire::KernelOpDirection::Write);
        assert_eq!(req.entries.len(), 1);
        match &req.entries[0].target {
            crate::vmm::wire::KernelOpTarget::Symbol(s) => assert_eq!(s, "test_field"),
            other => panic!("unexpected target shape: {other:?}"),
        }
        match req.entries[0].value {
            crate::vmm::wire::KernelOpValue::U64(42) => {}
            ref other => panic!("unexpected value shape: {other:?}"),
        }
        assert_eq!(bridge_clone.drain_kernel_ops().len(), 1);
        cleanup_state(&mut state);
    }

    /// `Op::WriteKernelCold` dispatches with `KernelOpMode::Cold`
    /// (vs Hot) — pins the per-arm mode mapping. A regression that
    /// reused Hot's payload-build path for Cold would surface here.
    #[test]
    fn apply_ops_write_kernel_cold_dispatches_with_cold_mode() {
        use std::sync::Arc;
        let captured = Arc::new(std::sync::Mutex::new(
            None::<crate::vmm::wire::KernelOpRequestPayload>,
        ));
        let captured_clone = captured.clone();
        let kernel_op_cb: crate::scenario::snapshot::KernelOpCallback = Arc::new(move |req| {
            *captured_clone.lock().unwrap() = Some(req.clone());
            crate::vmm::wire::KernelOpReplyPayload {
                request_id: req.request_id,
                success: true,
                reason: String::new(),
                read_values: vec![],
            }
        });
        let bridge = crate::scenario::snapshot::SnapshotBridge::new(Arc::new(|_| None))
            .with_kernel_op(kernel_op_cb);
        let _bg = bridge.set_thread_local();
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        let ops = vec![Op::write_kernel_cold_batch(vec![
            (
                KernelTarget::per_cpu_field("runqueues", "clock", 0),
                KernelValue::u64(100),
            ),
            (
                KernelTarget::per_cpu_field("runqueues", "clock", 1),
                KernelValue::u64(200),
            ),
        ])];
        apply_ops_test(&ctx, &mut state, &ops).expect("WriteKernelCold must dispatch");
        let req = captured.lock().unwrap().take().expect("callback must fire");
        assert_eq!(req.mode, crate::vmm::wire::KernelOpMode::Cold);
        assert_eq!(req.direction, crate::vmm::wire::KernelOpDirection::Write);
        assert_eq!(req.entries.len(), 2, "batch must carry both entries");
        cleanup_state(&mut state);
    }

    /// `Op::ReadKernelHot` dispatches with the right tag + width
    /// hint. The wire payload's value-slot mirrors the
    /// `KernelValueWidth` chosen at the variant level: U32 picks
    /// the u32 read family, U64 picks u64, Bytes(N) picks the
    /// N-byte read.
    #[test]
    fn apply_ops_read_kernel_hot_dispatches_with_width_u32() {
        use std::sync::Arc;
        let captured = Arc::new(std::sync::Mutex::new(
            None::<crate::vmm::wire::KernelOpRequestPayload>,
        ));
        let captured_clone = captured.clone();
        let kernel_op_cb: crate::scenario::snapshot::KernelOpCallback = Arc::new(move |req| {
            *captured_clone.lock().unwrap() = Some(req.clone());
            crate::vmm::wire::KernelOpReplyPayload {
                request_id: req.request_id,
                success: true,
                reason: String::new(),
                read_values: vec![crate::vmm::wire::KernelOpValue::U32(7)],
            }
        });
        let bridge = crate::scenario::snapshot::SnapshotBridge::new(Arc::new(|_| None))
            .with_kernel_op(kernel_op_cb);
        let bridge_clone = bridge.clone();
        let _bg = bridge.set_thread_local();
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        let ops = vec![Op::read_kernel_hot(
            "scratch_u32",
            KernelTarget::symbol("some_u32"),
            KernelValueWidth::u32(),
        )];
        apply_ops_test(&ctx, &mut state, &ops).expect("ReadKernelHot must dispatch");
        let req = captured.lock().unwrap().take().expect("callback must fire");
        assert_eq!(req.mode, crate::vmm::wire::KernelOpMode::Hot);
        assert_eq!(req.direction, crate::vmm::wire::KernelOpDirection::Read);
        assert_eq!(req.tag, "scratch_u32");
        match req.entries[0].value {
            crate::vmm::wire::KernelOpValue::U32(_) => {}
            ref other => panic!("u32 width hint must emit U32 slot, got {other:?}"),
        }
        // Single-tag convenience accessor returns the U32 read-back.
        match bridge_clone.kernel_op_value("scratch_u32") {
            Some(crate::vmm::wire::KernelOpValue::U32(7)) => {}
            other => panic!("kernel_op_value lookup mismatch: {other:?}"),
        }
        cleanup_state(&mut state);
    }

    /// `Op::ReadKernelCold` mirrors `Op::ReadKernelHot` with cold
    /// mode + Bytes width. Pins the Bytes width hint passing
    /// through to the wire payload's value slot.
    #[test]
    fn apply_ops_read_kernel_cold_dispatches_with_width_bytes() {
        use std::sync::Arc;
        let captured = Arc::new(std::sync::Mutex::new(
            None::<crate::vmm::wire::KernelOpRequestPayload>,
        ));
        let captured_clone = captured.clone();
        let kernel_op_cb: crate::scenario::snapshot::KernelOpCallback = Arc::new(move |req| {
            *captured_clone.lock().unwrap() = Some(req.clone());
            crate::vmm::wire::KernelOpReplyPayload {
                request_id: req.request_id,
                success: true,
                reason: String::new(),
                read_values: vec![crate::vmm::wire::KernelOpValue::Bytes(vec![0xAA; 16])],
            }
        });
        let bridge = crate::scenario::snapshot::SnapshotBridge::new(Arc::new(|_| None))
            .with_kernel_op(kernel_op_cb);
        let _bg = bridge.set_thread_local();
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        let ops = vec![Op::read_kernel_cold(
            "scratch_bytes",
            KernelTarget::kva(0xffff_c900_0000_1000),
            KernelValueWidth::bytes(16),
        )];
        apply_ops_test(&ctx, &mut state, &ops).expect("ReadKernelCold must dispatch");
        let req = captured.lock().unwrap().take().expect("callback must fire");
        assert_eq!(req.mode, crate::vmm::wire::KernelOpMode::Cold);
        assert_eq!(req.direction, crate::vmm::wire::KernelOpDirection::Read);
        match &req.entries[0].value {
            crate::vmm::wire::KernelOpValue::Bytes(b) => {
                assert_eq!(b.len(), 16, "Bytes(16) width hint must emit a 16-byte slot");
            }
            other => panic!("Bytes width hint must emit Bytes slot, got {other:?}"),
        }
        cleanup_state(&mut state);
    }

    /// Three singleton `Op::WriteKernelCold` ops dispatched
    /// through `apply_ops` produce ONE bridge callback with all 3
    /// writes — confirms the executor's pre-pass folds adjacent
    /// singletons into a single freeze rendezvous end-to-end,
    /// not just at the helper level. Pins the freeze-rendezvous-
    /// batching contract the [`Op::WriteKernelCold`] doc names
    /// as a "hard correctness requirement" (no inter-CPU skew).
    #[test]
    fn apply_ops_merges_three_adjacent_cold_write_singletons_into_one_dispatch() {
        use std::sync::Arc;
        let captured = Arc::new(std::sync::Mutex::new(Vec::<
            crate::vmm::wire::KernelOpRequestPayload,
        >::new()));
        let captured_clone = captured.clone();
        let kernel_op_cb: crate::scenario::snapshot::KernelOpCallback = Arc::new(move |req| {
            captured_clone.lock().unwrap().push(req.clone());
            crate::vmm::wire::KernelOpReplyPayload {
                request_id: req.request_id,
                success: true,
                reason: String::new(),
                read_values: vec![],
            }
        });
        let bridge = crate::scenario::snapshot::SnapshotBridge::new(Arc::new(|_| None))
            .with_kernel_op(kernel_op_cb);
        let _bg = bridge.set_thread_local();
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        let ops = vec![
            Op::write_kernel_cold(
                KernelTarget::per_cpu_field("runqueues", "clock", 0),
                KernelValue::u64(100),
            ),
            Op::write_kernel_cold(
                KernelTarget::per_cpu_field("runqueues", "clock", 1),
                KernelValue::u64(200),
            ),
            Op::write_kernel_cold(
                KernelTarget::per_cpu_field("runqueues", "clock", 2),
                KernelValue::u64(300),
            ),
        ];
        apply_ops_test(&ctx, &mut state, &ops).expect("merged cold-write batch must dispatch");
        let payloads = captured.lock().unwrap();
        assert_eq!(
            payloads.len(),
            1,
            "3 adjacent singletons must collapse into ONE bridge dispatch, got {} dispatches",
            payloads.len()
        );
        assert_eq!(payloads[0].mode, crate::vmm::wire::KernelOpMode::Cold);
        assert_eq!(
            payloads[0].direction,
            crate::vmm::wire::KernelOpDirection::Write
        );
        assert_eq!(
            payloads[0].entries.len(),
            3,
            "merged batch must carry all 3 writes in input order"
        );
        cleanup_state(&mut state);
    }

    /// `Op::CaptureSnapshot` between two cold-write singletons
    /// acts as a hard barrier — the snapshot must observe state
    /// AFTER the first write but BEFORE the second. Pins the
    /// "any non-cold-write op is a barrier" generalization in the
    /// pre-pass: a regression that narrowed the predicate to
    /// "only kernel ops barrier" would fold across CaptureSnapshot
    /// and silently break the captured-state-between-writes
    /// contract.
    #[test]
    fn merge_adjacent_cold_writes_capture_snapshot_is_barrier() {
        use super::merge_adjacent_cold_writes;
        let ops = vec![
            Op::write_kernel_cold(KernelTarget::symbol("a"), KernelValue::u64(1)),
            Op::CaptureSnapshot { name: "mid".into() },
            Op::write_kernel_cold(KernelTarget::symbol("b"), KernelValue::u64(2)),
        ];
        let merged = merge_adjacent_cold_writes(&ops);
        assert_eq!(merged.len(), 3, "CaptureSnapshot must split cold writes");
        assert!(matches!(merged[0], Op::WriteKernelCold { ref writes } if writes.len() == 1));
        assert!(matches!(merged[1], Op::CaptureSnapshot { .. }));
        assert!(matches!(merged[2], Op::WriteKernelCold { ref writes } if writes.len() == 1));
    }

    /// A generic non-kernel op (e.g. `Op::AddCgroup`) between two
    /// cold-write singletons acts as a hard barrier. Pins the
    /// "any non-cold-write op is a barrier" predicate so a
    /// regression that narrowed to "only kernel ops barrier" or
    /// "only Op::Write/Read variants barrier" silently breaks
    /// sequencing with cgroup setup / payload spawn / etc.
    #[test]
    fn merge_adjacent_cold_writes_non_kernel_op_is_barrier() {
        use super::merge_adjacent_cold_writes;
        let ops = vec![
            Op::write_kernel_cold(KernelTarget::symbol("a"), KernelValue::u64(1)),
            Op::AddCgroup {
                name: "cg_mid".into(),
            },
            Op::write_kernel_cold(KernelTarget::symbol("b"), KernelValue::u64(2)),
        ];
        let merged = merge_adjacent_cold_writes(&ops);
        assert_eq!(
            merged.len(),
            3,
            "non-kernel cgroup op must split cold writes"
        );
        assert!(matches!(merged[0], Op::WriteKernelCold { ref writes } if writes.len() == 1));
        assert!(matches!(merged[1], Op::AddCgroup { .. }));
        assert!(matches!(merged[2], Op::WriteKernelCold { ref writes } if writes.len() == 1));
    }
}

#[cfg(test)]
mod workers_pct_construction_tests {
    use super::types::CgroupDef;
    use crate::workload::WorkSpec;

    /// Builder rejects NaN at construction.
    #[test]
    #[should_panic(expected = "must be finite and > 0.0")]
    fn cgroup_def_workers_pct_panics_on_nan() {
        let _ = CgroupDef::named("x").workers_pct(f64::NAN);
    }

    #[test]
    #[should_panic(expected = "must be finite and > 0.0")]
    fn cgroup_def_workers_pct_panics_on_inf() {
        let _ = CgroupDef::named("x").workers_pct(f64::INFINITY);
    }

    #[test]
    #[should_panic(expected = "must be finite and > 0.0")]
    fn cgroup_def_workers_pct_panics_on_zero() {
        let _ = CgroupDef::named("x").workers_pct(0.0);
    }

    #[test]
    #[should_panic(expected = "must be finite and > 0.0")]
    fn cgroup_def_workers_pct_panics_on_negative() {
        let _ = CgroupDef::named("x").workers_pct(-0.5);
    }

    #[test]
    #[should_panic(expected = "must be finite and > 0.0")]
    fn work_spec_workers_pct_panics_on_nan() {
        let _ = WorkSpec::default().workers_pct(f64::NAN);
    }

    #[test]
    #[should_panic(expected = "must be finite and > 0.0")]
    fn work_spec_workers_pct_panics_on_inf() {
        let _ = WorkSpec::default().workers_pct(f64::INFINITY);
    }

    #[test]
    #[should_panic(expected = "must be finite and > 0.0")]
    fn work_spec_workers_pct_panics_on_zero() {
        let _ = WorkSpec::default().workers_pct(0.0);
    }

    #[test]
    #[should_panic(expected = "must be finite and > 0.0")]
    fn work_spec_workers_pct_panics_on_negative() {
        let _ = WorkSpec::default().workers_pct(-0.5);
    }
}

#[cfg(test)]
mod kernel_op_dispatch_tests {
    //! Coverage for the kernel-op dispatch surface: the four
    //! `KernelTarget`/`KernelValue` conversion helpers plus
    //! `dispatch_kernel_op_request`'s bridge-first /
    //! wire-fallback / hard-bail routing.
    //!
    //! These exercise the host-side dispatch surface only. The
    //! in-guest wire path (`request_kernel_op` end-to-end) is
    //! covered separately in `src/vmm/guest_comms.rs::tests` and
    //! the future end-to-end integration suite.

    use std::sync::Arc;

    use super::{
        KernelTarget, KernelValue, Op, build_kernel_op_request, dispatch_kernel_op_request,
        merge_adjacent_cold_writes, write_entries_from_writes,
    };
    use crate::scenario::snapshot::{CaptureCallback, KernelOpCallback, SnapshotBridge};

    /// 1:1 mapping of every `KernelTarget` variant via the
    /// `From<&KernelTarget> for KernelOpTarget` impl — pins the
    /// Cow-to-String coercion + per-CPU field decomposition. A
    /// regression that flipped a variant tag, dropped a Cow→String
    /// conversion, or swapped per-cpu-field fields surfaces here.
    #[test]
    fn kernel_target_into_wire_maps_every_variant() {
        let cases: &[(KernelTarget, crate::vmm::wire::KernelOpTarget)] = &[
            (
                KernelTarget::symbol("jiffies"),
                crate::vmm::wire::KernelOpTarget::Symbol("jiffies".into()),
            ),
            (
                KernelTarget::direct(0xffff_8000_0000_2000),
                crate::vmm::wire::KernelOpTarget::Direct(0xffff_8000_0000_2000),
            ),
            (
                KernelTarget::kva(0xffff_c000_dead_beef),
                crate::vmm::wire::KernelOpTarget::Kva(0xffff_c000_dead_beef),
            ),
            (
                KernelTarget::per_cpu_field("runqueues", "clock", 5),
                crate::vmm::wire::KernelOpTarget::PerCpuField {
                    symbol: "runqueues".into(),
                    field: "clock".into(),
                    cpu: 5,
                },
            ),
            (
                KernelTarget::task_field(42, 1_700_000_000_000, "scx.dsq_vtime"),
                crate::vmm::wire::KernelOpTarget::TaskField {
                    pid: 42,
                    expected_start_time_ns: 1_700_000_000_000,
                    field: "scx.dsq_vtime".into(),
                },
            ),
        ];
        for (src, want) in cases {
            let got: crate::vmm::wire::KernelOpTarget = src.into();
            assert_eq!(&got, want, "wire mapping mismatch for {src:?}");
        }
    }

    /// 1:1 mapping of every `KernelValue` variant via the
    /// `From<&KernelValue> for KernelOpValue` impl — pins the
    /// Bytes-clone semantic and the numeric-width identity. A
    /// regression that swapped U32/U64 width or skipped the Bytes
    /// clone surfaces here.
    #[test]
    fn kernel_value_into_wire_maps_every_variant() {
        let u32_val: crate::vmm::wire::KernelOpValue = (&KernelValue::u32(42)).into();
        assert_eq!(u32_val, crate::vmm::wire::KernelOpValue::U32(42));
        let u64_val: crate::vmm::wire::KernelOpValue =
            (&KernelValue::u64(0xDEAD_BEEF_CAFE_F00D)).into();
        assert_eq!(
            u64_val,
            crate::vmm::wire::KernelOpValue::U64(0xDEAD_BEEF_CAFE_F00D)
        );
        let bytes = vec![1u8, 2, 3, 4, 5];
        let bytes_val: crate::vmm::wire::KernelOpValue =
            (&KernelValue::bytes(bytes.clone())).into();
        assert_eq!(bytes_val, crate::vmm::wire::KernelOpValue::Bytes(bytes));
        // OrU32: width is u32 not u64 (struct scx_rq.flags is u32
        // per kernel/sched/sched.h:802); a regression that mapped
        // OrU32 to wire-side U32 or OrU64 (had one existed) would
        // silently lose the RMW intent at the dispatcher and
        // either drop the OR or corrupt the adjacent field.
        let or_val: crate::vmm::wire::KernelOpValue = (&KernelValue::or_u32(1 << 5)).into();
        assert_eq!(or_val, crate::vmm::wire::KernelOpValue::OrU32(1 << 5));
        // Edge mask values per tester pass-1 spec: degenerate
        // zero mask (a tempting wrong-optimization to skip the OR
        // on `mask == 0` would trip here), all-bits-set mask
        // (RMW degenerates to a full overwrite — still must
        // route through OrU32 wire variant, not U32), and a
        // multi-bit non-power-of-2 mask (catches a regression
        // that treated OrU32 as single-bit-only).
        let zero_or: crate::vmm::wire::KernelOpValue = (&KernelValue::or_u32(0)).into();
        assert_eq!(zero_or, crate::vmm::wire::KernelOpValue::OrU32(0));
        let max_or: crate::vmm::wire::KernelOpValue = (&KernelValue::or_u32(u32::MAX)).into();
        assert_eq!(max_or, crate::vmm::wire::KernelOpValue::OrU32(u32::MAX));
        let multi_bit_or: crate::vmm::wire::KernelOpValue =
            (&KernelValue::or_u32(0xA5A5_A5A5)).into();
        assert_eq!(
            multi_bit_or,
            crate::vmm::wire::KernelOpValue::OrU32(0xA5A5_A5A5)
        );
    }

    /// `KernelValue::OrU32` participates in `PartialEq` distinct
    /// from its `U32` sibling and distinct from other OrU32 masks.
    /// A regression that custom-impl'd PartialEq ignoring the
    /// variant tag (so `OrU32(1) == U32(1)`) would silently
    /// conflate write modes — `assert_eq!` on collected
    /// `Vec<KernelValue>` would compare-equal across variants.
    #[test]
    fn kernel_value_partial_eq_distinguishes_oru32_from_u32_and_other_masks() {
        assert_eq!(KernelValue::or_u32(1), KernelValue::or_u32(1));
        assert_ne!(KernelValue::or_u32(1), KernelValue::or_u32(2));
        assert_ne!(KernelValue::or_u32(1), KernelValue::u32(1));
    }

    /// `KernelValue::or_u32` is `const fn` so static contexts can
    /// declare flag constants without a runtime construction. A
    /// regression that dropped the `const` modifier would push
    /// every static use to a runtime build (or a compile error
    /// at the callsite). This compile-time `const _` assertion
    /// trips on the regression at the file boundary, not at the
    /// downstream caller.
    #[test]
    fn kernel_value_or_u32_is_const_constructible() {
        const _OR_AT_COMPILE_TIME: KernelValue = KernelValue::or_u32(1 << 5);
    }

    /// `write_entries_from_writes` preserves order and produces one
    /// wire entry per source pair — the cold-batch contract relies
    /// on every entry surviving the conversion in the supplied
    /// sequence (a reorder would change which CPU's `rq.clock`
    /// landed in which freeze-rendezvous slot).
    #[test]
    fn write_entries_from_writes_preserves_order_and_count() {
        let writes = vec![
            (
                KernelTarget::per_cpu_field("runqueues", "clock", 0),
                KernelValue::u64(100),
            ),
            (
                KernelTarget::per_cpu_field("runqueues", "clock", 1),
                KernelValue::u64(200),
            ),
            (KernelTarget::symbol("jiffies"), KernelValue::u32(0xDEAD)),
        ];
        let entries = write_entries_from_writes(&writes);
        assert_eq!(entries.len(), 3);
        match (&entries[0].target, &entries[0].value) {
            (
                crate::vmm::wire::KernelOpTarget::PerCpuField { cpu: 0, .. },
                crate::vmm::wire::KernelOpValue::U64(100),
            ) => {}
            other => panic!("entry[0] mismatch: {other:?}"),
        }
        match (&entries[1].target, &entries[1].value) {
            (
                crate::vmm::wire::KernelOpTarget::PerCpuField { cpu: 1, .. },
                crate::vmm::wire::KernelOpValue::U64(200),
            ) => {}
            other => panic!("entry[1] mismatch: {other:?}"),
        }
        match (&entries[2].target, &entries[2].value) {
            (
                crate::vmm::wire::KernelOpTarget::Symbol(s),
                crate::vmm::wire::KernelOpValue::U32(0xDEAD),
            ) if s == "jiffies" => {}
            other => panic!("entry[2] mismatch: {other:?}"),
        }
    }

    /// `dispatch_kernel_op_request` hard-bails when no bridge is
    /// installed AND we're not in a guest VM — per the project
    /// "no silent drops" rule. The bail message must name both
    /// recovery paths (install bridge callback / run in guest VM)
    /// so the misconfigured-test signal is actionable. A regression
    /// that reverted to silent warn-skip would re-introduce the
    /// vacuous-test footgun.
    #[test]
    fn dispatch_kernel_op_request_no_bridge_no_guest_hard_bails() {
        let payload = build_kernel_op_request(
            crate::vmm::wire::KernelOpMode::Hot,
            crate::vmm::wire::KernelOpDirection::Write,
            "missing_setup".into(),
            vec![],
        );
        let r = dispatch_kernel_op_request("Op::TestNoBridge", payload);
        let err = r.expect_err("no-bridge/non-guest must bail loudly, not warn-skip");
        let msg = err.to_string();
        assert!(
            msg.contains("Op::TestNoBridge"),
            "error must name the op label: {msg}"
        );
        assert!(
            msg.contains("missing_setup"),
            "error must name the request tag: {msg}"
        );
        assert!(
            msg.contains("with_kernel_op"),
            "error must point at SnapshotBridge::with_kernel_op recovery path: {msg}"
        );
        assert!(
            msg.contains("guest VM"),
            "error must mention the guest-VM recovery path: {msg}"
        );
    }

    /// `dispatch_kernel_op_request` invokes the bridge callback on
    /// success, propagates `reply.success == true` as `Ok(())`, and
    /// the bridge's drain log captures the (tag, reply) pair.
    #[test]
    fn dispatch_kernel_op_request_bridge_success_path() {
        let cb: CaptureCallback = Arc::new(|_| None);
        let kernel_op_cb: KernelOpCallback =
            Arc::new(|req| crate::vmm::wire::KernelOpReplyPayload {
                request_id: req.request_id,
                success: true,
                reason: String::new(),
                read_values: vec![],
            });
        let bridge = SnapshotBridge::new(cb).with_kernel_op(kernel_op_cb);
        let bridge_clone = bridge.clone();
        let _g = bridge.set_thread_local();
        let payload = build_kernel_op_request(
            crate::vmm::wire::KernelOpMode::Cold,
            crate::vmm::wire::KernelOpDirection::Write,
            "test_tag".into(),
            vec![crate::vmm::wire::KernelOpEntry {
                target: crate::vmm::wire::KernelOpTarget::Symbol("jiffies".into()),
                value: crate::vmm::wire::KernelOpValue::U64(42),
            }],
        );
        let r = dispatch_kernel_op_request("Op::TestSuccess", payload);
        assert!(r.is_ok(), "bridge success path must Ok, got {r:?}");
        let log = bridge_clone.drain_kernel_ops();
        assert_eq!(log.len(), 1);
        assert_eq!(log[0].0, "test_tag");
        assert!(log[0].1.success);
    }

    /// `dispatch_kernel_op_request` propagates `reply.success ==
    /// false` as an `anyhow::Error` carrying the reason — the
    /// executor's `?` propagation thus surfaces host-side op
    /// failures (e.g. "symbol not found") to the caller.
    #[test]
    fn dispatch_kernel_op_request_bridge_failure_path_bails() {
        let cb: CaptureCallback = Arc::new(|_| None);
        let kernel_op_cb: KernelOpCallback =
            Arc::new(|req| crate::vmm::wire::KernelOpReplyPayload {
                request_id: req.request_id,
                success: false,
                reason: "host: symbol 'bogus' not found".into(),
                read_values: vec![],
            });
        let bridge = SnapshotBridge::new(cb).with_kernel_op(kernel_op_cb);
        let _g = bridge.set_thread_local();
        let payload = build_kernel_op_request(
            crate::vmm::wire::KernelOpMode::Hot,
            crate::vmm::wire::KernelOpDirection::Read,
            "failing_tag".into(),
            vec![crate::vmm::wire::KernelOpEntry {
                target: crate::vmm::wire::KernelOpTarget::Symbol("bogus".into()),
                value: crate::vmm::wire::KernelOpValue::U64(0),
            }],
        );
        let r = dispatch_kernel_op_request("Op::TestFailure", payload);
        let err = r.expect_err("reply.success=false must bail");
        let msg = err.to_string();
        assert!(
            msg.contains("Op::TestFailure"),
            "error must name the op label: {msg}"
        );
        assert!(
            msg.contains("failing_tag"),
            "error must name the request tag: {msg}"
        );
        assert!(
            msg.contains("symbol 'bogus' not found"),
            "error must surface the host reason: {msg}"
        );
    }

    /// Three adjacent `Op::WriteKernelCold` singletons fold into
    /// one merged op with the 3 writes concatenated in input
    /// order. Pins the multi-CPU-seed shape — `with_uptime`
    /// writing per-CPU `rq.clock` on every CPU must land in ONE
    /// freeze rendezvous, not N.
    #[test]
    fn merge_adjacent_cold_writes_folds_three_singletons() {
        let ops = vec![
            Op::write_kernel_cold(
                KernelTarget::per_cpu_field("runqueues", "clock", 0),
                KernelValue::u64(100),
            ),
            Op::write_kernel_cold(
                KernelTarget::per_cpu_field("runqueues", "clock", 1),
                KernelValue::u64(200),
            ),
            Op::write_kernel_cold(
                KernelTarget::per_cpu_field("runqueues", "clock", 2),
                KernelValue::u64(300),
            ),
        ];
        let merged = merge_adjacent_cold_writes(&ops);
        assert_eq!(
            merged.len(),
            1,
            "3 adjacent cold-write singletons must fold to 1 op"
        );
        match &merged[0] {
            Op::WriteKernelCold { writes } => {
                assert_eq!(writes.len(), 3, "merged batch must carry all 3 writes");
                match &writes[0].1 {
                    KernelValue::U64(100) => {}
                    other => panic!("first entry value mismatch: {other:?}"),
                }
                match &writes[2].1 {
                    KernelValue::U64(300) => {}
                    other => panic!("third entry value mismatch: {other:?}"),
                }
            }
            other => panic!("expected merged WriteKernelCold, got {other:?}"),
        }
    }

    /// An `Op::WriteKernelHot` between two cold-write singletons
    /// is a hard barrier — the two cold-write singletons emerge as
    /// 2 separate `Op::WriteKernelCold` ops (one on each side of
    /// the hot barrier), preserving rendezvous boundaries.
    #[test]
    fn merge_adjacent_cold_writes_hot_is_barrier() {
        let ops = vec![
            Op::write_kernel_cold(KernelTarget::symbol("a"), KernelValue::u64(1)),
            Op::write_kernel_hot(KernelTarget::symbol("h"), KernelValue::u64(2)),
            Op::write_kernel_cold(KernelTarget::symbol("b"), KernelValue::u64(3)),
        ];
        let merged = merge_adjacent_cold_writes(&ops);
        assert_eq!(merged.len(), 3, "hot barrier must split cold writes");
        assert!(matches!(merged[0], Op::WriteKernelCold { ref writes } if writes.len() == 1));
        assert!(matches!(merged[1], Op::WriteKernelHot { .. }));
        assert!(matches!(merged[2], Op::WriteKernelCold { ref writes } if writes.len() == 1));
    }

    /// An `Op::ReadKernelCold` between two cold-write singletons
    /// is a hard barrier — reads don't fold into write batches
    /// in this pre-pass (a follow-up adds per-entry direction +
    /// tag for mixed-direction folding). The two cold-write
    /// singletons emerge as 2 separate ops + the read in between.
    #[test]
    fn merge_adjacent_cold_writes_read_is_barrier() {
        let ops = vec![
            Op::write_kernel_cold(KernelTarget::symbol("a"), KernelValue::u64(1)),
            Op::read_kernel_cold(
                "r",
                KernelTarget::symbol("r"),
                super::KernelValueWidth::u64(),
            ),
            Op::write_kernel_cold(KernelTarget::symbol("b"), KernelValue::u64(2)),
        ];
        let merged = merge_adjacent_cold_writes(&ops);
        assert_eq!(merged.len(), 3, "cold-read barrier must split cold writes");
        assert!(matches!(merged[0], Op::WriteKernelCold { ref writes } if writes.len() == 1));
        assert!(matches!(merged[1], Op::ReadKernelCold { .. }));
        assert!(matches!(merged[2], Op::WriteKernelCold { ref writes } if writes.len() == 1));
    }

    /// Caller-supplied multi-write `Op::WriteKernelCold` (via
    /// `Op::write_kernel_cold_batch`) merges with adjacent
    /// singletons — the multi-write's `writes` vec appends onto
    /// the running batch in input order.
    #[test]
    fn merge_adjacent_cold_writes_appends_multi_write_op() {
        let ops = vec![
            Op::write_kernel_cold(KernelTarget::symbol("pre"), KernelValue::u64(0)),
            Op::write_kernel_cold_batch(vec![
                (KernelTarget::symbol("a"), KernelValue::u64(1)),
                (KernelTarget::symbol("b"), KernelValue::u64(2)),
            ]),
            Op::write_kernel_cold(KernelTarget::symbol("post"), KernelValue::u64(3)),
        ];
        let merged = merge_adjacent_cold_writes(&ops);
        assert_eq!(
            merged.len(),
            1,
            "singleton+batch+singleton must fold to 1 op"
        );
        match &merged[0] {
            Op::WriteKernelCold { writes } => {
                assert_eq!(writes.len(), 4);
                let names: Vec<&str> = writes
                    .iter()
                    .map(|(t, _)| match t {
                        KernelTarget::Symbol(s) => s.as_ref(),
                        _ => panic!("non-Symbol target"),
                    })
                    .collect();
                assert_eq!(names, vec!["pre", "a", "b", "post"]);
            }
            other => panic!("expected merged WriteKernelCold, got {other:?}"),
        }
    }

    /// Empty input → empty output. Single cold-write → single
    /// cold-write (no spurious wrapping). Pre-pass is structurally
    /// equivalent on inputs with no fold opportunities.
    #[test]
    fn merge_adjacent_cold_writes_passes_through_when_nothing_to_fold() {
        assert!(merge_adjacent_cold_writes(&[]).is_empty());
        let single = vec![Op::write_kernel_cold(
            KernelTarget::symbol("x"),
            KernelValue::u64(42),
        )];
        let merged = merge_adjacent_cold_writes(&single);
        assert_eq!(merged.len(), 1);
        match &merged[0] {
            Op::WriteKernelCold { writes } => assert_eq!(writes.len(), 1),
            other => panic!("expected single WriteKernelCold, got {other:?}"),
        }
    }
}
