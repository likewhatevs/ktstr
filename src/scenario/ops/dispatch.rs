//! Op-dispatch layer for [`crate::scenario::ops::Op`]-driven scenarios.
//!
//! [`apply_ops`] is the per-step dispatch loop that drives the [`Op`]
//! enum against a [`super::ScenarioState`] view, plus its tightly-coupled
//! per-op helper bundle: scheduler lifecycle ([`spawn_scheduler_for_op`],
//! [`kill_current_scheduler`], the four `dispatch_*_scheduler` fns,
//! the two `wait_for_*_or_bail` quiesce helpers, [`wait_for_scx_disabled`]),
//! kernel-Op wire helpers ([`build_kernel_op_request`],
//! [`write_entries_from_writes`], [`merge_adjacent_cold_writes`],
//! [`dispatch_kernel_op_request`], [`check_kernel_op_reply`]), staged-path
//! helpers ([`staged_scheduler_log_path`], [`next_sched_spawn_seq`]),
//! the payload-handle lookup ([`take_payload_for_op`]), the cpuset reader
//! ([`read_cpuset`]), the cgroup-key renderer ([`render_cgroup_key`]),
//! the snapshot-transport latch ([`SNAPSHOT_TRANSPORT_DEAD`]), and the
//! three timing/path consts ([`SCHED_LIFECYCLE_KILL_GRACE`],
//! [`REPLACE_NOT_TRYING_DEADLINE_S`], [`SCX_STATE_SYSFS`]).
//!
//! Sibling to [`super::setup`]; both mutate the same
//! [`super::ScenarioState`] view over step-local + backdrop state.
//! `apply_ops` re-enters `setup::apply_setup` on [`Op::AddCgroupDef`].

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result};

use crate::scenario::Ctx;
use crate::workload::{ResolvedAffinity, WorkloadConfig, WorkloadHandle};

use super::setup::{append_placement_log, apply_setup};
use super::{
    KernelTarget, KernelValue, Op, PayloadEntry, PayloadSource, ScenarioState, SpawnPlacement,
    validate_known_flags, validate_mempolicy_cpuset,
};

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

/// Apply a slice of Ops to the running state.
///
/// Ops that create new entities (`AddCgroup`, `AddCgroupDef`,
/// `Spawn`, `RunPayload`) route into step-local state by default, or
/// into backdrop when the Backdrop's initial setup phase is active.
/// Ops that read or mutate existing entities (`SetCpuset`,
/// `ClearCpuset`, `SwapCpusets`, `SetAffinity`, `MoveAllTasks`,
/// `RemoveCgroup`, `StopCgroup`, `WaitPayload`, `KillPayload`)
/// resolve the target name against step-local first, then backdrop
/// — so a Step's ops can reach into Backdrop-declared cgroups by
/// name without the Backdrop leaking implementation details.
pub(super) fn apply_ops(
    ctx: &Ctx,
    state: &mut ScenarioState<'_, '_>,
    ops: &[Op],
    in_loop: bool,
) -> Result<()> {
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
            Op::Spawn { placement, work } => match placement {
                SpawnPlacement::RunnerCgroup => {
                    if let Err(reason) = work.mem_policy.validate() {
                        anyhow::bail!("Op::Spawn(RunnerCgroup): {}", reason);
                    }
                    // RunnerCgroup placement has no managed cgroup
                    // whose cpuset would scale `workers_pct`. Bail
                    // loud — silent fallback to ctx.workers_per_cgroup
                    // would discard the operator's intent.
                    if work.workers_pct.is_some() {
                        anyhow::bail!(
                            "Op::Spawn with SpawnPlacement::RunnerCgroup does not support \
                             `WorkSpec::workers_pct` — RunnerCgroup spawns workers in the \
                             test runner's own cgroup, with no managed cgroup whose cpuset \
                             would scale the fraction against (workers_pct = `ceil(cpuset_cpus \
                             * pct)`). Either set an explicit `.workers(N)` count, or switch \
                             to SpawnPlacement::Cgroup(name) against a cgroup whose cpuset \
                             gives `workers_pct` a denominator.",
                        );
                    }
                    let n = crate::scenario::resolve_num_workers(
                        work,
                        ctx.workers_per_cgroup,
                        "<runner>",
                    )?;
                    let affinity =
                        crate::scenario::intent_for_spawn(&work.affinity, None, ctx.topo)?;
                    let wl = WorkloadConfig::for_scenario_engine(
                        work,
                        n,
                        affinity,
                        work.work_type.clone(),
                    )?;
                    let mut h = WorkloadHandle::spawn(&wl)?;
                    h.start();
                    // Empty-string key: workers stay in the runner's
                    // own cgroup (no managed cgroup membership) —
                    // see SpawnPlacement::RunnerCgroup doc.
                    state.target_handles().push((String::new(), h));
                }
                SpawnPlacement::Cgroup(cgroup) => {
                    // Reject empty-string cgroup at the dispatch
                    // entry: `move_tasks` further down also rejects
                    // it, but the resulting error mentions
                    // `cgroup ''` after workers have already been
                    // spawned (Drop SIGKILLs them, but the wasted
                    // spawn churns the kernel). Bail upfront with an
                    // actionable redirect.
                    if cgroup.is_empty() {
                        anyhow::bail!(
                            "Op::Spawn(SpawnPlacement::Cgroup): cgroup name is empty — \
                             use SpawnPlacement::runner_cgroup() to spawn workers in \
                             the test runner's own cgroup, or pass a non-empty name \
                             via SpawnPlacement::cgroup(name)",
                        );
                    }
                    // Pre-validate that the cgroup exists in step or
                    // backdrop tracking BEFORE WorkloadHandle::spawn
                    // burns clone(2) syscalls. Without this gate
                    // `move_tasks` would fail with kernel ENOENT
                    // AFTER the workers spawn (and get SIGKILLed via
                    // Drop on the failure path) — wasted spawn
                    // churn AND a less actionable error (kernel
                    // ENOENT does not include the registry of
                    // known cgroup names). Bail upfront with a
                    // diagnostic that lists the tracked names so
                    // the test author spots a typo immediately.
                    if !state.cgroup_name_is_tracked(cgroup) {
                        anyhow::bail!(
                            "Op::Spawn(SpawnPlacement::Cgroup('{cgroup}')): \
                             cgroup '{cgroup}' is not tracked by the scenario state — \
                             declare it via CgroupDef in Step.setup, \
                             Op::add_cgroup / Op::add_cgroup_def earlier in the \
                             same step, or on the persistent Backdrop. Tracked \
                             step-local cgroups: {step:?}; tracked Backdrop \
                             cgroups: {backdrop:?}",
                            step = state.step.cgroups.names(),
                            backdrop = state.backdrop.cgroups.names(),
                        );
                    }
                    if let Err(reason) = work.mem_policy.validate() {
                        anyhow::bail!("Op::Spawn(Cgroup '{}'): {}", cgroup, reason);
                    }
                    let cgroup_cpuset: Option<BTreeSet<usize>> =
                        state.lookup_cpuset(cgroup).cloned();
                    let cpuset_size = cgroup_cpuset
                        .as_ref()
                        .map_or_else(|| ctx.topo.usable_cpuset().len(), |s| s.len());
                    let work = work.clone().resolve_workers_pct(cpuset_size, cgroup)?;
                    let n = crate::scenario::resolve_num_workers(
                        &work,
                        ctx.workers_per_cgroup,
                        cgroup,
                    )?;
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
                    // for_scenario_engine pins Fork in the constructor
                    // so the runtime assert that used to live here is
                    // redundant.  WorkSpec::pcomm is intentionally
                    // dropped on this path — see WorkSpec::pcomm doc:
                    // Fork-mode worker pcomm derives from the parent
                    // process's name, not from WorkSpec, so the
                    // pcomm field has no effect for spawn ops.
                    let wl = WorkloadConfig::for_scenario_engine(
                        &work,
                        n,
                        affinity,
                        work.work_type.clone(),
                    )?;
                    let mut h = WorkloadHandle::spawn(&wl)?;
                    ctx.cgroups.move_tasks(cgroup, &h.worker_pids())?;
                    h.start();
                    state.target_handles().push((cgroup.to_string(), h));
                }
            },
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
                // when a `CgroupDef::works` vec or repeated
                // `Op::Spawn(SpawnPlacement::Cgroup)` populates more
                // than one). The pool is invariant
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
            Op::MoveAllTasks { from, to } => {
                // Self-move is a silent no-op: cgroup.procs writes are
                // idempotent on same-cgroup targets and rename_handles
                // collapses to identity when from == to. The op
                // burns a freeze cycle + worker-state walk +
                // clear_subtree_control roundtrip for zero observable
                // effect. Bail so the test author surfaces real intent
                // (remove the stale op, or fix a typo where the two
                // names accidentally collapsed onto the same string)
                // instead of debugging an absent move. Symmetric with
                // the empty-string RunnerCgroup-capture pattern: an
                // explicit `from == to == ""` self-move is also
                // rejected here — moving RunnerCgroup-placement
                // workers to "self" doesn't materialize them under a
                // named cgroup.
                if from == to {
                    anyhow::bail!(
                        "Op::MoveAllTasks from '{}' to '{}' is a self-move \
                         and a silent no-op (cgroup.procs is idempotent on \
                         same-cgroup writes); remove the op or correct the \
                         typo on whichever side was intended to differ",
                        from,
                        to,
                    );
                }
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
                // Surface destination typos with a warn that dumps
                // both tracking sets. The kernel-level ENOENT from
                // `move_tasks` below names `to` literally but
                // doesn't compare against tracked names, so an
                // operator who typed "dts" for "dst" otherwise
                // hits a bare ENOENT with no hint that a similar
                // tracked name exists. Don't bail — the operator
                // may be moving into a hand-mkdir'd cgroup outside
                // ktstr's tracking, in which case the warn is
                // informational only. Mirrors Op::RemoveCgroup's
                // typo-late-surfacing pattern.
                if !state.cgroup_name_is_tracked(to) {
                    tracing::warn!(
                        cgroup = %to,
                        backdrop_cgroups = ?state.backdrop.cgroups.names(),
                        step_cgroups = ?state.step.cgroups.names(),
                        "Op::MoveAllTasks destination '{to}' matches no \
                         step-local or Backdrop-owned cgroup — could be a \
                         typo or a move into an externally-managed cgroup. \
                         Compare against the listed Backdrop and step \
                         cgroups; if the subsequent move fails with a \
                         kernel-level `cgroup.procs ENOENT`, the typo here \
                         is the probable source.",
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
                // Snapshot the source cgroup.procs right before we
                // dispatch the move so the apply_setup log records
                // whether the kernel still sees the workers we are
                // about to migrate. A divergence between pid_batches
                // (in-process handle's worker list) and the source-
                // side cgroup.procs (kernel's per-cgroup tasks set)
                // proves workers exited between apply_setup completion
                // and this op — the silent self-exit window that
                // move_tasks_inner's per-pid ESRCH tolerance would
                // otherwise mask.
                let from_procs_path = ctx
                    .cgroups
                    .parent_path()
                    .join(from.as_ref())
                    .join("cgroup.procs");
                let from_procs_pre = std::fs::read_to_string(&from_procs_path)
                    .unwrap_or_else(|e| format!("<read {}: {e}>", from_procs_path.display()));
                append_placement_log(&format!(
                    "Op::MoveAllTasks pre-move from={} to={} batches={:?} src_cgroup.procs={:?}",
                    from,
                    to,
                    pid_batches,
                    from_procs_pre.trim(),
                ));
                for pids in &pid_batches {
                    ctx.cgroups.move_tasks(to, pids)?;
                }
                // Post-move snapshot of the destination cgroup.procs.
                // Combined with the pre-move source snapshot, this
                // shows whether the kernel observed the migration.
                // Workers present in both src_cgroup.procs pre-move
                // AND dest cgroup.procs post-move means migration
                // succeeded and any later disappearance is downstream
                // (post-move exit, scheduler eviction, etc.).
                let to_procs_path = ctx
                    .cgroups
                    .parent_path()
                    .join(to.as_ref())
                    .join("cgroup.procs");
                let to_procs_post = std::fs::read_to_string(&to_procs_path)
                    .unwrap_or_else(|e| format!("<read {}: {e}>", to_procs_path.display()));
                append_placement_log(&format!(
                    "Op::MoveAllTasks post-move to={} dest_cgroup.procs={:?}",
                    to,
                    to_procs_post.trim(),
                ));
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
                // Reject CaptureSnapshot inside any HoldSpec::Loop
                // step. Each capture forces a freeze
                // rendezvous; inside a Loop generating a high-rate
                // pattern (e.g. 100Hz iteration burst), one capture
                // per iteration would destroy the workload pattern
                // via 100 freezes/sec. Boundary captures emitted from
                // a non-Loop Step before/after the Loop step still
                // give the operator pre/post bracketing.
                if in_loop {
                    anyhow::bail!(
                        "Op::CaptureSnapshot('{name}') inside HoldSpec::Loop forces a freeze \
                         rendezvous every loop iteration, freezing every vCPU on each iteration \
                         so the workload no longer runs at the rate you wrote; move the capture \
                         into a non-Loop Step before or after the Loop step"
                    );
                }
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
                            // failure. The host-side coordinator is
                            // unreachable until the process restarts.
                            // Fail loud: returning Ok here would
                            // silently mask a structural transport
                            // loss — tests would pass with no
                            // captured snapshot. Bail with a typed
                            // reason instead.
                            anyhow::bail!(
                                "Op::CaptureSnapshot('{name}'): snapshot transport latched dead; \
                                 a prior request observed TransportError and the latch only flips \
                                 on transport failure (host-side coordinator unreachable until \
                                 process restart)"
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
                        // host_only-vs-snapshot is a documented mutex
                        // (the host-only-mode contract); the
                        // C2 host-mode lever made this path reachable
                        // at scenario-engine dispatch for the first
                        // time. Per the no-silent-drops policy, bail
                        // loud rather than tracing::warn-then-skip —
                        // a test author writing
                        // `Op::capture_snapshot("phase1")` in a
                        // host_only test would otherwise see PASS
                        // with no captured snapshot data backing the
                        // assertions that rely on it.
                        anyhow::bail!(
                            "Op::CaptureSnapshot('{name}'): not supported in host_only mode \
                             (no guest VM, no test-fixture SnapshotBridge installed); \
                             snapshot capture is mutually exclusive with host_only — \
                             either drop the snapshot op or convert the test to non-host_only",
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
                                // Fail loud: returning Ok here would
                                // silently mask a structural transport
                                // loss — a WatchSnapshot that never
                                // arms looks identical to a healthy
                                // passing run.
                                anyhow::bail!(
                                    "Op::WatchSnapshot('{symbol}'): snapshot transport latched \
                                     dead; a prior request observed TransportError and the latch \
                                     only flips on transport failure (host-side coordinator \
                                     unreachable until process restart)"
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
                            // Same host_only-vs-snapshot mutex as
                            // Op::CaptureSnapshot above — C2 made the
                            // host-mode dispatch path reachable for
                            // both ops at once, so this sibling site
                            // needs the same loud bail per the
                            // no-silent-drops policy. A
                            // test author writing
                            // `Op::watch_snapshot("kernel_symbol")`
                            // in a host_only test would otherwise see
                            // PASS with no watchpoint armed, then
                            // wonder why every subsequent assertion
                            // that relied on the watch firing
                            // vacuously passes.
                            anyhow::bail!(
                                "Op::WatchSnapshot('{symbol}'): not supported in host_only mode \
                                 (no guest VM, no test-fixture SnapshotBridge installed); \
                                 hardware-watchpoint snapshots are mutually exclusive with \
                                 host_only — either drop the watch op or convert the test to \
                                 non-host_only",
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
                crate::vmm::rust_init::set_current_scheduler(Some(&scheduler.binary));
            }
            Op::DetachScheduler => {
                dispatch_detach_scheduler()?;
                crate::vmm::rust_init::set_current_scheduler(None);
            }
            Op::RestartScheduler => {
                dispatch_restart_scheduler()?;
                // RestartScheduler re-spawns the BOOT scheduler;
                // CURRENT_SCHEDULER stays as the prior identity
                // (whatever the last Attach/Replace published, or
                // None if the test only ever used the boot
                // scheduler). A future iteration tracking the
                // "currently-attached scheduler identity" on the
                // boot path (matching the spawn_scheduler_from_paths
                // side channel in rust_init) would let
                // RestartScheduler restore the published identity
                // explicitly; for v0 the slot keeps its last value
                // and consumers reading the post-Op state see the
                // same identity they would have seen pre-Op.
            }
            Op::ReplaceScheduler { scheduler } => {
                dispatch_replace_scheduler(scheduler)?;
                crate::vmm::rust_init::set_current_scheduler(Some(&scheduler.binary));
            }
            Op::PinBpfMap { name } => {
                // Idempotent — pinning the same name twice no-ops.
                // The first pin's OwnedFd already holds one extra
                // refcount via `__bpf_map_inc_not_zero`
                // (kernel/bpf/syscall.c:4859), which is sufficient
                // to keep the map alive past any scheduler teardown.
                // A second pin would bump the kernel refcount AGAIN
                // and consume another host fd slot, but the test
                // cannot observe the difference (one extra refcount
                // and two are both "alive"), so we skip the work.
                let name_key = name.as_ref().to_string();
                if let std::collections::hash_map::Entry::Vacant(slot) =
                    state.backdrop.pinned_bpf_maps.entry(name_key)
                {
                    let fd = crate::scenario::bpf_pin::open_bpf_map_fd_by_name(name.as_ref())
                        .map_err(|e| {
                            anyhow::anyhow!(
                                "Op::PinBpfMap({name:?}): {e:#}\n\
                                 \n\
                                 Common causes:\n  \
                                 (a) Target scheduler's BPF object hasn't finished \
                                 loading. Place this op AFTER a hold long enough for \
                                 the scheduler to attach (typically ~100ms for the \
                                 small scx-ktstr fixture, longer for heavyweight \
                                 schedulers).\n  \
                                 (b) Step ran before any `Op::AttachScheduler` or \
                                 before the boot scheduler started; pin must come \
                                 after the scheduler that owns the map is up.\n  \
                                 (c) Name exceeds the 15-char usable cap of \
                                 `BPF_OBJ_NAME_LEN` and was truncated by libbpf when \
                                 loaded — compare against the observed names in the \
                                 error above; the kernel-visible name is the \
                                 truncated form."
                            )
                        })?;
                    slot.insert(fd);
                }
            }
            Op::CaptureCgroupProcs { tag, cgroup } => {
                if tag.is_empty() {
                    anyhow::bail!(
                        "Op::CaptureCgroupProcs: tag is empty; the tag is the \
                         snapshot key consumers use to find the capture in \
                         `SnapshotBridge::drain_cgroup_procs` — supply a \
                         non-empty identifier (e.g. \"after_spawn\", \
                         \"post_migrate\")"
                    );
                }
                if cgroup.is_empty() {
                    anyhow::bail!(
                        "Op::CaptureCgroupProcs(tag={tag:?}): cgroup name is empty. \
                         Provide a cgroup name registered via `Op::AddCgroup`, a \
                         `CgroupDef` in setup, or pushed on the Backdrop; an empty \
                         name would resolve to the runner's own cgroup, which is \
                         almost certainly not the test author's intent"
                    );
                }
                // Bridge-presence pre-check BEFORE the read_procs syscall.
                // Burning the syscall when there's no recipient hides
                // the actionable missing-bridge diagnostic behind any
                // unrelated read failure (e.g. flaky cgroupfs); the
                // pattern matches Op::Spawn(SpawnPlacement::Cgroup(""))
                // bail-before-spawn at L2540-2553.
                if crate::scenario::snapshot::with_active_bridge(|_| ()).is_none() {
                    anyhow::bail!(
                        "Op::CaptureCgroupProcs(tag={tag:?}, cgroup={cgroup:?}): \
                         no SnapshotBridge installed for this thread; bailing \
                         before the cgroup.procs read so the misconfiguration \
                         surfaces without burning a syscall (a silent record \
                         would also leave subsequent `drain_cgroup_procs` empty, \
                         masking the missing-bridge bug). Install a bridge via \
                         `SnapshotBridge::set_thread_local` (RAII via \
                         `BridgeGuard`) before `execute_scenario` runs the op"
                    );
                }
                let pids = ctx.cgroups.read_procs(cgroup).with_context(|| {
                    format!(
                        "Op::CaptureCgroupProcs(tag={tag:?}, cgroup={cgroup:?}): \
                         CgroupOps::read_procs failed",
                    )
                })?;
                let pid_count = pids.len();
                // The bridge-presence pre-check above guarantees the
                // bridge is installed; record without re-checking.
                crate::scenario::snapshot::with_active_bridge(|b| {
                    b.record_cgroup_procs(tag.to_string(), cgroup.to_string(), pids);
                });
                tracing::info!(
                    tag = %tag,
                    cgroup = %cgroup,
                    pid_count,
                    "Op::CaptureCgroupProcs: captured cgroup.procs snapshot"
                );
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
/// is not yet implemented.
///
/// Pre-pass cost: one allocation per `apply_ops` call (the
/// returned `Vec<Op>`). When the input contains no adjacent cold
/// writes the output is structurally equivalent to the input.
pub(super) fn merge_adjacent_cold_writes(ops: &[Op]) -> Vec<Op> {
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
pub(super) fn build_kernel_op_request(
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
/// `super::types::op` for the 1:1 enum mapping.
pub(super) fn write_entries_from_writes(
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
pub(super) fn staged_scheduler_log_path(name: &str) -> String {
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

/// Deadline for `Op::ReplaceScheduler`'s worker-not-trying gate
/// (`SnapshotBridge::wait_for_worker_state_not_trying` inside
/// `dispatch_replace_scheduler`). 5 s matches one freeze-coord scan
/// tick + `from_elf_with_hint` at tens of ms scale. Asymmetric
/// against `Op::AttachScheduler`'s first-boot 60 s slab-settle
/// budget — Replace runs after the prior scheduler held the slab
/// alive for its whole hold, so no extra warm-up margin is needed.
/// Bumping this risks pushing total dispatch latency past the
/// test's `duration_s` budget; lowering it risks spurious
/// "stayed in TRYING past deadline" bails under cold-cache CI.
/// A regression that silently changes this value would surface as
/// either a flake (too tight) or a slow test (too loose), neither
/// of which produces an actionable signal — the unit test pinning
/// this const is the canary.
pub(super) const REPLACE_NOT_TRYING_DEADLINE_S: u64 = 5;

/// Path of the sched_ext kernel state sysfs node. Reading returns
/// the string-form state: `disabled`, `enabling`, `enabled`,
/// `disabling` (kernel/sched/ext.c). Op dispatch polls this
/// between kill and spawn so the next scheduler's BPF skeleton
/// load doesn't hit `-EBUSY` from
/// `kernel/sched/ext.c:6643`'s `scx_enable_state() != SCX_DISABLED`
/// guard at enable entry.
const SCX_STATE_SYSFS: &str = "/sys/kernel/sched_ext/state";

/// The sched_ext enable-state the kernel exposes as the one-line
/// content of [`SCX_STATE_SYSFS`] (`scx_enable_state_str`,
/// kernel `kernel/sched/ext_internal.h`). The enable path is
/// `Disabled → Enabling → Enabled`; the disable path — taken on BOTH a
/// runtime error/crash AND a clean unregister — is
/// `Enabled → Disabling → Disabled`. So [`ScxState::Disabling`] /
/// [`ScxState::Disabled`] are reached ONLY by going down, never by
/// coming up (a scheduler mid-spawn reads `Enabling`). The state is
/// KIND-LESS — a crash and a clean unregister produce the same string;
/// telling them apart is the caller's job.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ScxState {
    Enabling,
    Enabled,
    Disabling,
    Disabled,
}

impl ScxState {
    /// The exact sysfs string for this state (matches the kernel's
    /// `scx_enable_state_str`), for diagnostics.
    fn as_str(self) -> &'static str {
        match self {
            ScxState::Enabling => "enabling",
            ScxState::Enabled => "enabled",
            ScxState::Disabling => "disabling",
            ScxState::Disabled => "disabled",
        }
    }
}

/// Test override for [`scx_state`]: host unit tests have no live scx
/// scheduler, so they force a state here instead of reading sysfs.
/// `0` = no override; `1..=4` select a state. Mirrors
/// [`crate::probe::process::set_probe_sched_exit_state`] — a
/// process-global atomic, so tests must reset it (RAII) on drop.
#[cfg(test)]
static TEST_SCX_STATE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Force [`scx_state`] to return `state` (or `None` to clear the
/// override and resume reading sysfs). Test-only.
#[cfg(test)]
pub(crate) fn set_test_scx_state(state: Option<ScxState>) {
    use std::sync::atomic::Ordering;
    let v = match state {
        None => 0,
        Some(ScxState::Enabling) => 1,
        Some(ScxState::Enabled) => 2,
        Some(ScxState::Disabling) => 3,
        Some(ScxState::Disabled) => 4,
    };
    TEST_SCX_STATE.store(v, Ordering::Release);
}

/// Read [`SCX_STATE_SYSFS`] and map it to [`ScxState`]. `None` when the
/// file is absent (kernel without `CONFIG_SCHED_CLASS_EXT`, or the scx
/// kobject not registered) or the content is unrecognized — callers
/// treat `None` as "no scx state to act on". A single lock-free kernfs
/// read (the kernel value is an `atomic_read`), cheap to poll.
pub(crate) fn scx_state() -> Option<ScxState> {
    #[cfg(test)]
    {
        use std::sync::atomic::Ordering;
        match TEST_SCX_STATE.load(Ordering::Acquire) {
            1 => return Some(ScxState::Enabling),
            2 => return Some(ScxState::Enabled),
            3 => return Some(ScxState::Disabling),
            4 => return Some(ScxState::Disabled),
            _ => {} // 0 = no override; fall through to the real sysfs read
        }
    }
    let mut buf = String::with_capacity(16);
    std::fs::File::open(SCX_STATE_SYSFS)
        .and_then(|mut f| {
            use std::io::Read;
            f.read_to_string(&mut buf)
        })
        .ok()?;
    parse_scx_state(buf.trim_end())
}

/// Map a raw `/sys/kernel/sched_ext/state` string to [`ScxState`].
fn parse_scx_state(s: &str) -> Option<ScxState> {
    match s {
        "enabling" => Some(ScxState::Enabling),
        "enabled" => Some(ScxState::Enabled),
        "disabling" => Some(ScxState::Disabling),
        "disabled" => Some(ScxState::Disabled),
        _ => None,
    }
}

/// True when sched_ext is going down or down (`Disabling`/`Disabled`) —
/// the probe-independent live crash-or-unregister signal readable in the
/// PRIMARY test VM. (The BPF err-exit latch's host mirror is only
/// populated by the auto-repro/dump VM's probe poll loop, so it stays
/// `Unknown` for the whole primary scenario — the crash signal the hold
/// needs is dead there. The sysfs state needs no probe.) Down-states
/// ONLY (NOT `!= Enabled`): the enable-ramp `Enabling` and steady
/// `Enabled` read false, so a scheduler coming up after an
/// `Op::ReplaceScheduler` swap is never misread as a crash. The
/// crash-vs-clean-unregister distinction is the caller's (the state is
/// kind-less): callers gate this on a scheduler still being expected to
/// run (`sched_pid` set) so a clean `Op::DetachScheduler` — which clears
/// the pid before this would observe `disabled` — does not trip it.
pub(crate) fn scx_down() -> bool {
    matches!(scx_state(), Some(ScxState::Disabling | ScxState::Disabled))
}

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
    use crate::vmm::freeze_coord::evented_wait::{KernfsWaitOutcome, kernfs_evented_wait};
    use nix::sys::inotify::AddWatchFlags;

    let start = std::time::Instant::now();
    let path = std::path::Path::new(SCX_STATE_SYSFS);
    if !path.exists() {
        return Ok(std::time::Duration::ZERO);
    }

    // `last_state` carries the most recent observed state into the
    // timeout error below. Routes through the canonical `scx_state`
    // reader so there is a single parser for the sysfs file.
    let mut last_state = String::new();
    let check_done = || -> Option<()> {
        let state = scx_state();
        last_state.clear();
        last_state.push_str(state.map_or("<absent>", ScxState::as_str));
        if state == Some(ScxState::Disabled) {
            Some(())
        } else {
            None
        }
    };

    // Evented wake sources managed by kernfs_evented_wait:
    //   - inotify on /sys/kernel/sched_ext/ for IN_DELETE (fires
    //     when scx_root_disable's kobject_del at
    //     kernel/sched/ext.c:5859 removes the "root" entry)
    //   - POLLPRI on /sys/kernel/sched_ext/state (future-proofed
    //     for kernels that add `sysfs_notify` on the attribute)
    //
    // REQUIRED CADENCE. Verified at kernel/sched/ext.c:5735
    // scx_root_disable: between kobject_del (L5859, fires
    // IN_DELETE) and the state flip (L5865,
    // scx_set_enable_state(SCX_DISABLED)) the kernel does
    // free_kick_syncs() + mutex_unlock(&scx_enable_mutex) + the
    // atomic_xchg. Sub-microsecond gap in the common case but
    // theoretically present, AND scx emits NO further event for
    // the state-attribute transition (no sysfs_notify, IN_DELETE
    // already fired). Without the 50ms cadence cap, an IN_DELETE
    // wake catching the pre-flip state would block until the
    // overall deadline before re-reading. The cap is the necessary
    // guard rail for the kernel-side ordering gap — NOT a polling
    // fallback for failed evented setup.
    let cadence = std::time::Duration::from_millis(50);
    let outcome = kernfs_evented_wait(
        "/sys/kernel/sched_ext/",
        AddWatchFlags::IN_DELETE,
        Some("/sys/kernel/sched_ext/state"),
        cadence,
        start + timeout,
        check_done,
    );

    match outcome {
        KernfsWaitOutcome::Done(()) => Ok(start.elapsed()),
        KernfsWaitOutcome::Timeout => {
            anyhow::bail!(
                "wait_for_scx_disabled: state '{last_state}' did not reach 'disabled' \
                 within {timeout:?}; the kernel scx state machine is stuck — \
                 the next scheduler spawn will hit -EBUSY at the enable path. \
                 Inspect /sys/kernel/sched_ext/state + dmesg for the stuck \
                 disable transition.",
            );
        }
        KernfsWaitOutcome::NoEventedSource => {
            // Both state_fd open and inotify_add_watch failed. We
            // target kernel 6.12+ where kernfs + inotify are
            // universal, so the path-exists-but-attribute-fd-
            // unopenable branch indicates a fundamental env defect.
            anyhow::bail!(
                "wait_for_scx_disabled: could not subscribe to evented wake \
                 sources (state fd open failed AND inotify_add_watch on \
                 /sys/kernel/sched_ext/ failed). Diagnose: \
                 (1) does '/sys/kernel/sched_ext/state' exist AND contain \
                 'disabled' or 'enabled'? If absent/garbage the kernel was \
                 built without CONFIG_SCHED_CLASS_EXT — rebuild with that \
                 config. \
                 (2) zcat /proc/config.gz | grep CONFIG_INOTIFY_USER must be \
                 =y — without it the framework's evented wake can't \
                 subscribe; rebuild with CONFIG_INOTIFY_USER=y."
            );
        }
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
    // Stop the guest sched_exit_monitor BEFORE SIGTERM so the
    // monitor thread exits without sending the SchedExit message
    // that the host's dispatch.rs SchedExit arm would otherwise
    // promote into the run-wide kill flag (failing the test as
    // scheduler-died even though the kill is an INTENTIONAL
    // lifecycle Op). The post-spawn Attach / Restart / Replace
    // paths re-install a fresh monitor for the new SCHED_PID via
    // `restart_sched_exit_monitor_with_log`; Detach leaves the
    // slot empty because there is no replacement scheduler.
    // SchedExitStop.stop_and_join joins the monitor thread before
    // returning, so the kill below is safe from the prior
    // monitor's pidfd race.
    crate::vmm::rust_init::stop_sched_exit_monitor();
    // Pin the post-stop invariant for Restart+Replace dispatch
    // sites: after `stop_sched_exit_monitor` returns, the slot
    // MUST be empty so the subsequent spawn's
    // `restart_sched_exit_monitor_with_log` call lands a fresh
    // monitor without first stop+joining a stale handle. The
    // `Op::AttachScheduler` site does NOT flow through here (no
    // prior scheduler to kill); its possibly-non-empty entry is
    // handled silently by `restart_sched_exit_monitor_with_log`'s
    // defensive `take()`. debug-only — release builds rely on
    // that defensive take() as the safety net.
    debug_assert!(
        crate::vmm::rust_init::sched_exit_monitor_slot_is_empty(),
        "kill_current_scheduler did not clear sched_exit_monitor slot — \
         stop_sched_exit_monitor() must precede the kill (called from {op_label})",
    );
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
pub(super) fn dispatch_attach_scheduler(
    scheduler: &'static crate::test_support::Scheduler,
) -> Result<()> {
    // Serialize against any in-flight worker publish BEFORE the
    // dispatcher captures `seqno_before`. The accessor-init worker
    // has a 60 s boot budget for its first publish; if the user's
    // test has an auto-boot scheduler config, the worker may be
    // mid-init for the boot scheduler when the user's
    // Op::AttachScheduler dispatches. Without this gate, the
    // dispatcher's read-spawn-wait sequence could be satisfied by
    // the BOOT scheduler's first publish (seqno 0 → 1) and return
    // success before the user's scheduler is actually live.
    //
    // After this wait, worker_state ∈ {SUCCEEDED, FAILED_PERMANENTLY}
    // (FAILED → bail) — the boot publish has either landed and
    // the slab settled, OR the worker has given up and a fresh
    // attach from a stuck state will hang the same way regardless.
    // Either way, the seqno captured AFTER this wait reflects a
    // quiescent slab and any subsequent advance MUST belong to the
    // worker re-init triggered by THIS op's spawn (the coord's
    // Published-arm pulse fires reinit_evt for the new scheduler).
    let boot_deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    wait_for_worker_state_not_trying_or_bail("Op::AttachScheduler", boot_deadline)?;
    let seqno_before =
        crate::scenario::snapshot::with_active_bridge(|b| b.accessor_publish_seqno()).unwrap_or(0);
    let binary = crate::test_support::staged::staged_scheduler_binary_path(scheduler.name);
    let args = crate::test_support::staged::staged_scheduler_args_path(scheduler.name);
    let log = staged_scheduler_log_path(scheduler.name);
    spawn_scheduler_for_op("Op::AttachScheduler", &binary, &args, &log, scheduler.name)?;
    // Install a fresh sched_exit_monitor against the just-spawned
    // SCHED_PID so the post-Op scheduler retains death detection.
    // No prior monitor existed for this slot (Attach is the
    // "first scheduler" or post-Detach attach); the restart helper
    // handles the empty-slot case as a fresh install.
    crate::vmm::rust_init::restart_sched_exit_monitor_with_log(Some(&log));
    // 30 s deadline: the coord's Published-arm pulse + worker's
    // no-deadline reinit budget (post-boot the 60 s gate is gone)
    // typically lands in <500 ms (one scan tick + tens of ms for
    // from_elf_with_hint). 30 s gives 60× margin for slow hosts /
    // cold-cache vmlinux reads on the FIRST post-boot attach,
    // where the slab may still be settling. Replace uses 5 s
    // because the kill flushes the slab and the re-init lands
    // faster; attach onto a freshly-booted system can be slower.
    wait_for_accessor_publish_or_bail(
        "Op::AttachScheduler",
        seqno_before,
        std::time::Duration::from_secs(30),
    )?;
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
pub(super) fn dispatch_detach_scheduler() -> Result<()> {
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
pub(super) fn dispatch_restart_scheduler() -> Result<()> {
    let prev_pid = kill_current_scheduler("Op::RestartScheduler")?;
    let log = staged_scheduler_log_path("boot");
    spawn_scheduler_for_op(
        "Op::RestartScheduler",
        "/scheduler",
        "/sched_args",
        &log,
        "boot",
    )?;
    crate::vmm::rust_init::restart_sched_exit_monitor_with_log(Some(&log));
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
pub(super) fn dispatch_replace_scheduler(
    scheduler: &'static crate::test_support::Scheduler,
) -> Result<()> {
    let prev_pid = kill_current_scheduler("Op::ReplaceScheduler")?;
    let binary = crate::test_support::staged::staged_scheduler_binary_path(scheduler.name);
    let args = crate::test_support::staged::staged_scheduler_args_path(scheduler.name);
    let log = staged_scheduler_log_path(scheduler.name);
    spawn_scheduler_for_op("Op::ReplaceScheduler", &binary, &args, &log, scheduler.name)?;
    // Re-install monitor against the replacement scheduler's pid
    // so death detection persists past the swap. The seq-suffixed
    // log path matches the spawn_scheduler_for_op log arg above so
    // failure-dump output goes to the new scheduler's own file.
    crate::vmm::rust_init::restart_sched_exit_monitor_with_log(Some(&log));
    // Quiesce the worker before capturing the baseline seqno.
    // Symmetric with Op::AttachScheduler's wait at L3377-3382:
    // without this gate, a coord scan tick that fired during
    // kill_current_scheduler (which can take up to
    // SCHED_LIFECYCLE_KILL_GRACE = 10 s) may still be in a
    // TRYING worker-state when we capture seqno_before. If that
    // in-flight publish completes between our seqno_before snapshot
    // and the wait below, the wait's seqno-advance gate fires on a
    // publish against OLD scheduler state — the operator's
    // post-swap `Snapshot::active()` then surfaces stale OLD data
    // even though the new scheduler is running fine. 5 s deadline
    // matches Attach's per-pulse expectation (coord scan tick +
    // from_elf_with_hint at tens-of-ms scale); FAILED arms bail
    // with the worker-state diagnostic for the caller's `?` to
    // surface as a typed Op failure. The deadline is tighter than
    // Attach's first-boot budget because by the time we reach
    // Replace the slab has been live for the prior scheduler's
    // whole hold so no first-boot 60 s slab-settle headroom is
    // needed.
    let not_trying_deadline =
        std::time::Instant::now() + std::time::Duration::from_secs(REPLACE_NOT_TRYING_DEADLINE_S);
    wait_for_worker_state_not_trying_or_bail("Op::ReplaceScheduler", not_trying_deadline)?;
    // Capture the publish seqno AFTER the worker-not-trying gate
    // above. The gate guarantees worker_state ∈ {SUCCEEDED,
    // FAILED_PERMANENTLY}, so any seqno advance the wait below
    // observes MUST originate from the worker re-init triggered
    // by our newly-spawned scheduler (the coord's Published-arm
    // pulse fires reinit_evt for the new BPF object). None →
    // bridge isn't installed (caught upfront by
    // wait_for_worker_state_not_trying_or_bail above); this
    // accessor_publish_seqno fallback to 0 is the cold-start case
    // when no prior publish has happened.
    let seqno_before =
        crate::scenario::snapshot::with_active_bridge(|b| b.accessor_publish_seqno()).unwrap_or(0);
    // 10s deadline: a cold-cache vmlinux re-parse during the worker
    // reinit can run several seconds on a slow CI runner. The prior 5s
    // ceiling produced sporadic timeouts when the kernel cache was cold.
    // 10s aligns with SCHED_LIFECYCLE_KILL_GRACE — still asymmetric vs
    // Op::AttachScheduler's 30s budget (which covers a from-scratch
    // attach with no warm worker state to leverage) but covers the
    // realistic post-swap reinit window with margin. A timeout surfaces
    // as an actionable anyhow::bail with the worker-state sentinel
    // attached.
    wait_for_accessor_publish_or_bail(
        "Op::ReplaceScheduler",
        seqno_before,
        std::time::Duration::from_secs(10),
    )?;
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

/// Wait until the bridge's worker thread is NOT in TRYING state.
/// Called before `Op::AttachScheduler` / `Op::ReplaceScheduler`
/// capture the publish seqno, so any subsequent seqno advance is
/// attributable to the OP's spawn — not an in-flight publish from
/// before the op started.
///
/// Bails loud when no SnapshotBridge is installed on the executor
/// thread. Same rationale as
/// [`wait_for_accessor_publish_or_bail`]: a silent fallback would
/// pretend the not-trying gate succeeded without verifying the
/// worker state, masking a future change that exposes this path.
/// `KtstrTestEntry::validate` rejects host_only + non-EEVDF at
/// entry-construction time, so the op never reaches dispatch in
/// host_only mode today — the bail is defense-in-depth + closes
/// the silent-drop hole by construction.
pub(super) fn wait_for_worker_state_not_trying_or_bail(
    op_label: &str,
    deadline: std::time::Instant,
) -> Result<()> {
    let res = crate::scenario::snapshot::with_active_bridge(|b| {
        b.wait_for_worker_state_not_trying(deadline, op_label)
    });
    match res {
        Some(Ok(_)) => Ok(()),
        Some(Err(e)) => Err(e),
        None => {
            // Guest mode: the SnapshotBridge's `accessor_worker_state`
            // is an Arc<AtomicU8> populated by the freeze coordinator's
            // accessor-init worker, which reads guest memory from the
            // HOST process (see `freeze_coord`'s `accessor_worker_state`
            // allocation for the construction site, and its
            // `accessor_init_handle` spawn for the worker spawn). The
            // bridge is not transferable across
            // the host→guest process boundary, so a guest-side dispatcher
            // legitimately has no bridge installed. The wait observes a
            // host-side signal; in guest mode it's structurally inapplicable.
            // Actual attach correctness in guest mode is enforced by
            // `spawn_scheduler_for_op` via
            // `crate::vmm::rust_init::poll_scx_attached` polling
            // `/sys/kernel/sched_ext/root/ops` directly (see `try_spawn_scheduler`
            // in src/vmm/rust_init/scheduler.rs). Skip with a debug trace so the
            // operator can see the intentional bypass.
            if crate::vmm::guest_comms::is_guest() {
                tracing::debug!(
                    op = %op_label,
                    "wait_for_worker_state_not_trying skipped: guest mode (host-side accessor not installable in guest executor thread; attach correctness verified by poll_scx_attached)",
                );
                return Ok(());
            }
            anyhow::bail!(
                "{op_label}: no SnapshotBridge installed on the executor's thread; \
                 cannot observe worker-state-not-trying gate — the pre-spawn quiesce \
                 would be silently skipped, letting a stale in-flight publish from \
                 the prior scheduler corrupt the post-spawn seqno baseline. Recovery \
                 options: (1) install a SnapshotBridge on the executor thread \
                 (test-fixture path via `SnapshotBridge::new(cb).set_thread_local()`), \
                 (2) drop the scheduler op, or (3) convert host_only tests to \
                 non-host_only so the VM-orchestrated bridge install applies \
                 (snapshot mutex per the host-only-mode contract)",
            )
        }
    }
}

/// Wait helper used by `dispatch_replace_scheduler`. Reads the
/// active bridge's `wait_for_accessor_publish_advance` and threads
/// the timeout budget per op. Returns `Ok(())` when the accessor-
/// init worker has advanced the publish seqno past `seqno_before`.
/// Bails loud when no SnapshotBridge is installed (see body
/// comment for rationale + recovery paths). Surfaces the wait
/// error via `?` on the inner `anyhow::Result` so the op dispatch
/// returns the worker-state diagnostic verbatim.
pub(super) fn wait_for_accessor_publish_or_bail(
    op_label: &str,
    seqno_before: u64,
    budget: std::time::Duration,
) -> Result<()> {
    let deadline = std::time::Instant::now() + budget;
    let res = crate::scenario::snapshot::with_active_bridge(|b| {
        b.wait_for_accessor_publish_advance(seqno_before, deadline, op_label)
    });
    match res {
        Some(Ok(_)) => Ok(()),
        Some(Err(e)) => Err(e),
        // No bridge installed on the executor thread → no observer
        // for the accessor-publish seqno advance.
        None => {
            // Guest mode: same rationale as
            // `wait_for_worker_state_not_trying_or_bail` — the bridge's
            // `accessor_publish_seqno` Arc<AtomicU64> is host-side
            // (populated by the freeze-coord accessor-init worker reading
            // guest memory from host), structurally inapplicable to a
            // guest-side dispatcher. Scheduler-liveness verification in
            // guest mode is provided by `poll_scx_attached`'s
            // `/sys/kernel/sched_ext/root/ops` poll inside
            // `spawn_scheduler_for_op` (the spawn-side check runs BEFORE
            // we reach this wait).
            if crate::vmm::guest_comms::is_guest() {
                tracing::debug!(
                    op = %op_label,
                    "wait_for_accessor_publish skipped: guest mode (host-side accessor not installable in guest executor thread; scheduler-liveness verified by poll_scx_attached)",
                );
                return Ok(());
            }
            // Returning Ok in HOST mode without a bridge would silently
            // pretend the wait succeeded without ever verifying that
            // {op_label}'s scheduler attach actually published — per
            // the no-silent-drops policy (defense in depth). Bail
            // explicitly so the silent-drop hole stays closed by
            // construction in host_only test contexts.
            anyhow::bail!(
                "{op_label}: no SnapshotBridge installed on the executor's thread; \
                 cannot observe accessor-publish seqno advance — scheduler-liveness \
                 verification would be silently skipped. Recovery options: (1) \
                 install a SnapshotBridge on the executor thread (test-fixture \
                 path via `SnapshotBridge::new(cb).set_thread_local()`), (2) drop \
                 the scheduler-attach op, or (3) convert host_only tests to \
                 non-host_only so the VM-orchestrated bridge install applies \
                 (snapshot mutex per the host-only-mode contract)",
            )
        }
    }
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
/// **Timeout choice.** The 30 s wire-fallback timeout is sized for
/// the cold path's freeze-rendezvous round-trip (matches the
/// `FREEZE_RENDEZVOUS_TIMEOUT` budget). The hot path
/// completes sub-microsecond and treats the timeout strictly as an
/// upper bound; a regression that stalls the host-worker would
/// surface as a deferred 30 s wait, not a missed bug.
pub(super) fn dispatch_kernel_op_request(
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
/// Render a cgroup key for inclusion in user-facing error text.
/// An empty string is replaced with `(no cgroup)` so
/// `Op::RunPayload { cgroup: None }` failures don't produce messages
/// like `cgroup ''` that look like a corrupt log line. Non-empty
/// keys are quoted so they read clearly next to surrounding prose.
pub(super) fn render_cgroup_key(cgroup: &str) -> String {
    if cgroup.is_empty() {
        "(no cgroup)".to_string()
    } else {
        format!("'{cgroup}'")
    }
}

#[cfg(test)]
mod scx_state_tests {
    use super::{ScxState, parse_scx_state, scx_down, set_test_scx_state};

    /// T1 regression: `scx_down()` (the hold-abort / crash signal) fires
    /// on the DOWN-states ONLY. An `Op::ReplaceScheduler` enable-ramp
    /// reads `Enabling` and a steady scheduler reads `Enabled`; neither
    /// must be misread as a crash, or the hold would abort while a new
    /// scheduler is coming up. (Guards against the "abort on `!= Enabled`"
    /// bug, which would trip on `Enabling`.)
    #[test]
    fn scx_down_only_on_down_states() {
        for (state, want_down) in [
            (ScxState::Enabling, false),
            (ScxState::Enabled, false),
            (ScxState::Disabling, true),
            (ScxState::Disabled, true),
        ] {
            set_test_scx_state(Some(state));
            assert_eq!(scx_down(), want_down, "scx_down() for {state:?}");
        }
        // Clear the override (hygiene; nextest runs each test in its own
        // process so it can't leak, but reset anyway). NOT asserting the
        // post-clear `scx_down()`: that falls through to a real sysfs read
        // and a host running the tests with sched_ext loaded reads
        // `disabled` → down — environment-dependent, not the invariant
        // under test (which is the override-driven table above).
        set_test_scx_state(None);
    }

    #[test]
    fn parse_scx_state_maps_the_kernel_strings() {
        assert_eq!(parse_scx_state("enabling"), Some(ScxState::Enabling));
        assert_eq!(parse_scx_state("enabled"), Some(ScxState::Enabled));
        assert_eq!(parse_scx_state("disabling"), Some(ScxState::Disabling));
        assert_eq!(parse_scx_state("disabled"), Some(ScxState::Disabled));
        assert_eq!(parse_scx_state(""), None);
        assert_eq!(parse_scx_state("bogus"), None);
    }
}
