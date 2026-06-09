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
//! drives that model against [`crate::cgroup::CgroupOps`] via `apply_setup`
//! (sibling `setup` module) and `apply_ops` (sibling `dispatch` module),
//! and exposes the [`execute_steps`] / [`execute_scenario`] family of
//! public entry points (this file).

mod types;
pub use types::*;

mod setup;
pub use setup::PLACEMENT_LOG_PATH;
use setup::apply_setup;

mod dispatch;
#[cfg(test)]
use dispatch::{
    REPLACE_NOT_TRYING_DEADLINE_S, build_kernel_op_request, dispatch_kernel_op_request,
    merge_adjacent_cold_writes, staged_scheduler_log_path, wait_for_accessor_publish_or_bail,
    wait_for_worker_state_not_trying_or_bail, write_entries_from_writes,
};
use dispatch::{apply_ops, render_cgroup_key};

use std::collections::BTreeSet;
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result};

use crate::assert::AssertResult;
use crate::scenario::backdrop;
use crate::scenario::{CgroupGroup, Ctx, process_alive};
use crate::vmm::guest_comms;
use crate::vmm::wire::StimulusPayload;
use crate::workload::{MemPolicy, WorkloadHandle};

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
    /// BPF map fds opened via [`crate::scenario::ops::types::Op::PinBpfMap`].
    /// Keyed by the map name the caller requested; the
    /// [`std::os::fd::OwnedFd`] holds an extra refcount on the
    /// kernel-side `struct bpf_map` so the map survives any
    /// scheduler-process teardown (including
    /// [`crate::scenario::ops::types::Op::ReplaceScheduler`]) until
    /// scenario end. Drops close the fds and release the refcount.
    pinned_bpf_maps: std::collections::HashMap<String, std::os::fd::OwnedFd>,
}

impl<'a> BackdropState<'a> {
    /// Empty backdrop state (no persistent entities), scoped to `ctx.cgroups`.
    fn empty(ctx: &'a Ctx) -> Self {
        Self {
            cgroups: CgroupGroup::new(ctx.cgroups),
            handles: Vec::new(),
            cpusets: std::collections::HashMap::new(),
            payload_handles: Vec::new(),
            pinned_bpf_maps: std::collections::HashMap::new(),
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
    /// Host-mode worker stall monitor, started lazily at the end of
    /// the first successful [`apply_setup`] when running outside a
    /// VM (no `is_guest`, no `cargo_test_mode`) and at least one
    /// worker exists. The handle's [`Drop`] joins the polling
    /// thread when [`StepState`] drops; [`collect_step`] drains
    /// any accumulated reports before that drop so they reach the
    /// scenario's [`AssertResult`]. `None` in every guest-side
    /// scenario and in `cargo_test_mode` runs — the host-side
    /// monitor is the only stall signal available in host-mode,
    /// where the freeze coordinator / KVM-side stall plumbing is
    /// not running. See [`crate::scenario::host_stall`] for the
    /// signal definition and detection latency contract.
    stall_monitor: Option<crate::scenario::host_stall::StallMonitorHandle>,
}

impl<'a> StepState<'a> {
    /// Empty step state scoped to `ctx.cgroups`.
    fn empty(ctx: &'a Ctx) -> Self {
        Self {
            cgroups: CgroupGroup::new(ctx.cgroups),
            handles: Vec::new(),
            cpusets: std::collections::HashMap::new(),
            payload_handles: Vec::new(),
            stall_monitor: None,
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

/// Block until the host freeze coordinator has ADOPTED its
/// kernel-symbol accessor — signalled via the
/// `SIGNAL_ACCESSOR_READY` wake byte →
/// `accessor_ready_latch` (set by `hvc0_poll_loop`). A failure dump
/// captured by a stall AFTER this returns renders real BPF map values
/// instead of placeholders, because the coordinator's `owned_accessor`
/// is adopted before the stall fires. Call this in a dump-asserting
/// scenario before triggering its stall (e.g. before [`execute_steps`]
/// with a `--stall-after` scheduler).
///
/// Guest-only: a no-op on the host (unit tests), where the latch is
/// never armed. Warn-and-proceed on a 60s timeout — a never-adopted
/// accessor is a worker failure, and surfacing it as a placeholder dump
/// is more useful than blocking the test forever (mirrors the
/// `wait_for_map_write` gate's soft-timeout policy). On that timeout the
/// "renders real values" guarantee above does NOT hold: the wait returns
/// without adoption and a subsequent stall may dump placeholders, which
/// the test's post-VM dump assertions then surface as a failure.
///
/// The latch is sticky (level-triggered) and sets once, on the FIRST
/// adoption. A re-init publish after a scheduler swap
/// (`Op::ReplaceScheduler`) reuses the same latch, so a later
/// `await_accessor_ready` returns immediately and does NOT re-synchronise
/// on the post-swap adoption — today's only caller gates the first stall,
/// before any swap.
pub fn await_accessor_ready() {
    if guest_comms::is_guest() {
        let latch = crate::vmm::rust_init::accessor_ready_latch();
        if !latch.wait_timeout(Duration::from_secs(60)) {
            tracing::warn!(
                "await_accessor_ready timed out after 60s — host freeze \
                 coordinator did not signal accessor adoption; a dump from a \
                 stall after this point may render placeholder map values"
            );
        }
    }
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
/// - [`CgroupDef::cpuset`] / [`CgroupDef::cpuset_mems`] → `Controller::Cpuset`
/// - [`CgroupDef::cpu`] → `Controller::Cpu`
/// - [`CgroupDef::memory`] → `Controller::Memory`
/// - [`CgroupDef::pids`] → `Controller::Pids`
/// - [`CgroupDef::io`] → `Controller::Io`
/// - [`Op::SetCpuset`] / [`Op::ClearCpuset`] / [`Op::SwapCpusets`] /
///   [`Op::SetAffinity`] → `Controller::Cpuset`
/// - Every other [`Op`] variant ([`Op::FreezeCgroup`],
///   [`Op::AddCgroup`], [`Op::Spawn`], [`Op::MoveAllTasks`], etc.)
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
                apply_ops(ctx, s, &backdrop.ops, false)?;
            }
            // Shorthand payloads: one Op::RunPayload per entry,
            // inherited cgroup placement.
            if !backdrop.payloads.is_empty() {
                let ops: Vec<Op> = backdrop
                    .payloads
                    .iter()
                    .map(|p| Op::run_payload(p, Vec::<String>::new()))
                    .collect();
                apply_ops(ctx, s, &ops, false)?;
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
            // Scheduler died between steps: the kernel has disabled
            // sched_ext and the backdrop workers have fallen back to
            // the builtin scheduler. SIGKILL them up front so the
            // collect reap below isn't scheduling-gated behind
            // still-spinning workers (see `sigkill_handles`).
            sigkill_handles(&backdrop_state.handles);
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
        // Install the assert-side phase guard for the scenario
        // driver's thread for the duration of this Step. Every
        // AssertDetail / PassDetail / InfoNote constructed under
        // the run_step call below auto-stamps its `phase` field
        // with "Step[<step_idx>]" via the thread-local snapshot
        // in `crate::assert::current_phase_label`. On Drop the
        // prior label is restored (BASELINE outside any Step), so
        // assertions evaluated post-loop (e.g. at scenario
        // teardown) stamp with the right outer scope.
        let _phase_guard = crate::assert::PhaseGuard::install_step(step_idx as u16);
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

        // Scheduler died during this step's hold: the kernel has
        // disabled sched_ext and the workers (step and backdrop) have
        // fallen back to the builtin scheduler. If CPU-bound they
        // would CFS-starve the per-worker reap in the collect calls
        // below, so SIGKILL them all up front — the reaps then find
        // already-exiting workers (see `sigkill_handles`). Gated on
        // the death flag so the normal teardown keeps its graceful
        // cooperative-stop + report-collection path.
        if sched_died_during_hold {
            sigkill_handles(&step_state.handles);
            sigkill_handles(&backdrop_state.handles);
        }

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

    // Scheduler died after the last hold: SIGKILL the backdrop
    // workers up front so the teardown reap below isn't
    // scheduling-gated behind still-spinning workers that fell back
    // to the builtin scheduler (see `sigkill_handles`).
    if sched_dead {
        sigkill_handles(&backdrop_state.handles);
    }

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
/// is already gone — return `true` immediately without sleeping. Any
/// other failure mode (pidfd_open non-ESRCH, epoll_create1,
/// epoll_ctl ADD, EpollTimeout::try_from, epoll_wait) panics with an
/// operator-actionable message. Polling fallbacks were removed per
/// the project-wide "no polling fallbacks for evented paths" rule:
/// pidfd_open has shipped since Linux 5.3 and epoll has been
/// universally available for longer, so a failure here indicates a
/// catastrophic environment defect (memory pressure exhausting fds,
/// kernel feature compiled out) rather than a recoverable transient.
/// A loud panic surfaces the defect immediately; the prior silent
/// sleep+probe fallback masked it as test flakiness.
///
/// Scheduling jitter under load can leave the actual elapsed time
/// modestly above `dur`.
/// Loud panic on env- or code-defect failures inside [`hold_or_sched_died`].
/// Centralised so every site emits the same module-qualified prefix
/// `ktstr::scenario::hold_or_sched_died` — operator can grep that exact
/// string to land at the panic source. `op` names the failed primitive,
/// `pid` carries the in-scope pid for cross-reference with /proc, `err`
/// renders the underlying errno or nix error, `advice` is the one-line
/// remediation classified by failure class (env vs framework code defect).
#[cold]
#[track_caller]
fn panic_evented_hold_defect(
    op: &str,
    pid: libc::pid_t,
    err: impl std::fmt::Display,
    advice: &str,
) -> ! {
    panic!("ktstr::scenario::hold_or_sched_died: {op} failed (pid={pid}): {err} — {advice}");
}

fn hold_or_sched_died(dur: Duration, sched_pid: Option<libc::pid_t>) -> bool {
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
        // pidfd_open shipped unconditionally in Linux 5.3 and ktstr's
        // kernel floor is well above that. A non-ESRCH failure (ENOMEM,
        // ENFILE, EPERM) means the test environment is broken in a way
        // polling cannot recover from. Panic loudly so the operator
        // sees the env defect instead of silently losing sched-died
        // detection for the rest of the hold.
        panic_evented_hold_defect(
            "pidfd_open",
            pid,
            format_args!("{err} (errno {:?})", err.raw_os_error()),
            "pidfd_open is unconditional from Linux 5.3; failure on a \
             5.3+ kernel = env defect — check ulimit -n / memory pressure / \
             cgroup pids.max",
        );
    }
    // SAFETY: the syscall succeeded and returned a fresh fd; it is
    // not registered with any other owner.
    let pidfd: OwnedFd = unsafe { OwnedFd::from_raw_fd(pidfd_raw as i32) };

    // epoll setup. EPOLL_CLOEXEC matches `wait_with_deadline` to
    // avoid leaking the epoll fd into any post-fork descendant.
    let epoll = match Epoll::new(EpollCreateFlags::EPOLL_CLOEXEC) {
        Ok(e) => e,
        Err(e) => {
            let fd = std::os::fd::AsRawFd::as_raw_fd(&pidfd);
            panic_evented_hold_defect(
                "epoll_create1(EPOLL_CLOEXEC)",
                pid,
                format_args!("{e} (pidfd={fd})"),
                "epoll has been universally available since 2.6; failure = \
                 env defect — check ulimit -n and CONFIG_EPOLL",
            );
        }
    };
    // `data` field is unused — we only ever watch one fd. The add()
    // syscall still needs an `EpollEvent` with populated events.
    let event = EpollEvent::new(EpollFlags::EPOLLIN, 0);
    if let Err(e) = epoll.add(pidfd.as_fd(), event) {
        panic_evented_hold_defect(
            "epoll_ctl(ADD)",
            pid,
            e,
            "epoll_ctl(ADD) on a freshly-opened pidfd should never fail \
             in a healthy kernel; documented errors (EBADF/EEXIST/ENOMEM) \
             are env defects — likely fd exhaustion or memory pressure",
        );
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
        // as `i32`. Single-pass clamp via `u128 → i32::MAX` so a
        // `Duration::MAX` remainder saturates at the max accepted
        // value instead of overflowing through the intermediate u32.
        let ms_i32 = remaining.as_millis().min(i32::MAX as u128) as i32;
        let timeout_param = match EpollTimeout::try_from(ms_i32) {
            Ok(t) => t,
            Err(e) => {
                // ms_i32 was clamped to the i32 range above so
                // EpollTimeout::try_from (which accepts i32) cannot
                // overflow. Reaching this arm means the EpollTimeout API
                // changed shape — code defect, not env transient.
                panic_evented_hold_defect(
                    "EpollTimeout::try_from",
                    pid,
                    format_args!("{e} (input={ms_i32})"),
                    "input was pre-clamped to fit i32; failure indicates an \
                     upstream nix EpollTimeout API change requiring code update",
                );
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
                panic_evented_hold_defect(
                    "epoll_wait",
                    pid,
                    e,
                    "epoll_wait on a freshly-created epoll with a single \
                     valid pidfd cannot legitimately fail outside EINTR; \
                     documented errors (EBADF/EFAULT/EINVAL) are framework- \
                     internal memory-safety defects — investigate concurrent \
                     fd mutation or stack-frame corruption",
                );
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
                drain_on_err!(scenario, apply_ops(ctx, &mut scenario, &step.ops, true));
                let remaining = deadline.saturating_duration_since(std::time::Instant::now());
                // Live `sched_pid()` read so a mid-loop
                // Op::ReplaceScheduler swap is watched at the NEW
                // pid, not the stale boot snapshot in ctx.
                if hold_or_sched_died(remaining.min(interval), crate::vmm::rust_init::sched_pid()) {
                    *sched_died_during_hold = true;
                    return Ok(());
                }
            }
        }
        _ => {
            // Ops first (e.g. parent cgroup creation), then
            // CgroupDef setup (children with workers).
            //
            // Footgun: a workload-producing `CgroupDef` in
            // `step.setup` is invisible to `step.ops` operating on
            // it, because step.ops runs BEFORE apply_setup creates
            // the cgroup AND registers the WorkloadHandle in
            // `state.all_handles()`. The failure mode is per-Op:
            // `Op::MoveAllTasks` against the absent source filters
            // `state.all_handles()` by name, finds zero matches,
            // iterates an empty `pid_batches`, and exits with no
            // work done — silent (no error surfaced to the
            // operator); `Op::CaptureCgroupProcs` reading the
            // missing `cgroup.procs` via `read_procs` returns the
            // ENOENT `with_context` and propagates `?`, bailing the
            // ops phase — loud (the test crashes with the cgroup
            // name in the error). Tests that need a freshly-spawned
            // worker pool to feed step ops must declare the
            // producing `CgroupDef` in the Backdrop (which runs
            // before any Step) when `step.setup` is `Setup::Defs(_)`.
            // When `step.setup` is `Setup::Factory(_)` (runtime-
            // generated defs that can't be hoisted at edit time),
            // the only viable remediation is restructuring into
            // multiple Steps — producer factory in Step N's setup,
            // consumer ops in Step N+1.
            drain_on_err!(scenario, apply_ops(ctx, &mut scenario, &step.ops, false));
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
            if hold_or_sched_died(hold_dur, crate::vmm::rust_init::sched_pid()) {
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

/// SIGKILL every worker in `handles` without reaping. Used only on
/// the scheduler-death paths in [`run_scenario`]: when the scheduler
/// under test dies, the kernel disables sched_ext and its workers
/// fall back to the builtin scheduler, so a CPU-bound worker pool
/// would otherwise make the per-worker `waitpid` reap in the
/// subsequent [`collect_step`] / [`collect_backdrop`]
/// scheduling-gated (teardown time scaling with worker count).
/// Delivering SIGKILL to every worker first lets that reap find
/// already-exiting workers. See
/// [`crate::workload::WorkloadHandle::sigkill_workers`].
fn sigkill_handles(handles: &[(String, WorkloadHandle)]) {
    for (_, h) in handles {
        h.sigkill_workers();
    }
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
    // Drain any host-mode stall reports that accumulated during the
    // step BEFORE dropping the monitor handle (Drop joins the
    // polling thread). Reports get folded into the merged
    // [`AssertResult`] below as
    // [`crate::assert::DetailKind::WorkerStalled`] failures. `take`
    // ensures the handle drops here (joining the thread) so the
    // polling thread exits before per-step teardown returns.
    let stall_reports = if let Some(handle) = step_state.stall_monitor.take() {
        let reports = handle.drain();
        drop(handle);
        reports
    } else {
        Vec::new()
    };
    let handles = std::mem::take(&mut step_state.handles);
    let mut result = crate::scenario::collect_handles(
        handles
            .into_iter()
            .map(|(name, h)| (h, step_state.cpusets.get(&name))),
        checks,
        Some(topo),
    );
    for report in stall_reports {
        result.record_fail(crate::assert::AssertDetail::new(
            crate::assert::DetailKind::WorkerStalled,
            format_stall_report(&report),
        ));
    }
    result
}

/// Render a [`crate::scenario::host_stall::StallReport`] as a
/// human-readable single-string assertion detail. Emits the stalled
/// pid + comm, the sample-window summary (first and last values
/// for both counters PLUS the computed `last - first` delta — both
/// expected to be zero for a true stall but rendered from the
/// samples themselves so a predicate-tolerance refactor stays
/// observable), and the diagnostic subset (state, wchan, syscall,
/// cgroup, host loadavg, optional kernel stack). The diagnostic's
/// `status_full` field is intentionally OMITTED — the parsed
/// `state` letter carries the actionable signal and the full
/// status file is verbose; sidecar consumers keying off
/// [`crate::assert::DetailKind::WorkerStalled`] can match on the
/// kind discriminator and read the full StallReport (carries the
/// status_full field) without parsing this message.
///
/// Format is multi-line so the operator can read the trip at a
/// glance without parsing structured output.
fn format_stall_report(report: &crate::scenario::host_stall::StallReport) -> String {
    use std::fmt::Write as _;
    let mut s = String::new();
    let _ = writeln!(
        s,
        "worker stall detected: pid={} comm={:?} (host-mode /proc/<pid>/sched polling)",
        report.pid, report.comm,
    );
    if let (Some(first), Some(last)) = (report.samples.first(), report.samples.last()) {
        // Compute deltas from the actual sample values so a
        // predicate-tolerance refactor (e.g. "fire on Δ <= N
        // switches rather than == 0") stays observable in the
        // rendered output. Saturating subtraction handles the
        // pathological counter-rollover case without panic.
        let nr_delta = last.nr_switches.saturating_sub(first.nr_switches);
        let rt_delta = last
            .sum_exec_runtime_ns
            .saturating_sub(first.sum_exec_runtime_ns);
        let _ = writeln!(
            s,
            "  sample window: nr_switches {} -> {} (delta {nr_delta}), sum_exec_runtime_ns {} -> {} (delta {rt_delta}), {} samples",
            first.nr_switches,
            last.nr_switches,
            first.sum_exec_runtime_ns,
            last.sum_exec_runtime_ns,
            report.samples.len(),
        );
    }
    let d = &report.diagnostic;
    let _ = writeln!(s, "  state: {}", d.state);
    let _ = writeln!(s, "  wchan: {}", d.wchan);
    let _ = writeln!(s, "  syscall: {}", d.syscall);
    let _ = writeln!(s, "  cgroup: {}", d.cgroup);
    let _ = writeln!(s, "  host loadavg: {}", d.host_loadavg);
    if let Some(stack) = &d.stack {
        let _ = writeln!(s, "  kernel stack:\n{stack}");
    }
    s
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

#[cfg(test)]
mod tests;

#[cfg(test)]
mod workers_pct_construction_tests;

#[cfg(test)]
mod kernel_op_dispatch_tests;
