//! Setup-phase orchestration for [`crate::scenario::ops::Op`]-driven scenarios.
//!
//! [`apply_setup`] runs the [`CgroupDef`] passes that materialise each
//! declared cgroup, resolve its cpuset / cpu / memory / io / pids
//! controllers, partition `WorkSpec`s by `pcomm`, spawn worker handles,
//! and start any [`CgroupDef::workload`] payload. After the per-def
//! loop finishes it triggers [`maybe_start_stall_monitor`] for the
//! host-mode worker-stall poller.
//!
//! Sibling to [`super::apply_ops`]; both mutate the same
//! [`super::ScenarioState`] view over step-local + backdrop state.

use std::collections::BTreeSet;

use anyhow::Result;

use crate::scenario::Ctx;
use crate::vmm::guest_comms;
use crate::workload::{WorkloadConfig, WorkloadHandle};

use super::{CgroupDef, PayloadEntry, PayloadSource, ScenarioState, validate_mempolicy_cpuset};

/// Path of the scenario placement-history log. Captures placement
/// events from both `apply_setup` (worker / payload spawn) and
/// `apply_ops` (`Op::MoveAllTasks` pre/post cgroup.procs snapshots).
/// Tests that assert on per-cgroup worker placement can read this
/// file on failure to see exactly which PIDs landed in which cgroup
/// at spawn time AND how `Op::MoveAllTasks` migrated them. The file
/// lives in tmpfs and survives the scenario teardown (only the
/// cgroup directories are rmdir'd; this file stays put). Cleared by
/// the runtime at the start of each test run so an assertion in
/// test N never reads stale placement from test N-1.
pub const PLACEMENT_LOG_PATH: &str = "/tmp/ktstr-placement.log";

/// Append `msg` (with a trailing newline) to the placement-history
/// log. See [`PLACEMENT_LOG_PATH`] for the transport rationale
/// (every guest-side stderr/serial route was empirically confirmed
/// to drop scenario-era diagnostic bytes in test mode). Errors are
/// swallowed because this is a best-effort diagnostic — if the
/// file write fails, the placement record just goes missing and the
/// operator falls back to the existing post-mortem cgroup.procs
/// snapshot.
pub(super) fn append_placement_log(msg: &str) {
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(PLACEMENT_LOG_PATH)
    {
        let _ = writeln!(f, "{msg}");
    }
}

/// Walk every [`CgroupDef`] in `defs` and register it against the
/// step or backdrop cgroup slot selected by `state.target_backdrop`.
/// A duplicate name (already tracked by either state) bails — a
/// [`CgroupDef`] must not silently shadow a cgroup that another state
/// slot has already created.
///
/// Each `CgroupDef`'s `works` vec is iterated, spawning one
/// [`WorkloadHandle`] per [`crate::workload::WorkSpec`] entry. Multiple
/// `WorkSpec`s for the same cgroup produce multiple handle entries with
/// the same name key; ops that filter by cgroup name (`StopCgroup`,
/// `SetAffinity`, etc.) naturally apply to all of them. When `works`
/// is empty, a single default `WorkSpec` is used (`SpinWait`, `Normal`,
/// `ctx.workers_per_cgroup` workers). Cgroups created here route into
/// step-local or backdrop state per the `target_backdrop` flag.
pub(super) fn apply_setup(
    ctx: &Ctx,
    state: &mut ScenarioState<'_, '_>,
    defs: &[CgroupDef],
) -> Result<()> {
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
        // Per-controller application, in the order the kernel docs and
        // the operator-meaningful constraints want: cpuset (sets the
        // CPUs every later task inherits) before cpuset.mems (NUMA
        // node binding) before cpu (weight + cpu.max) before memory
        // (max/high/low/swap) before io (weight) before pids
        // (pids.max). Each helper is a no-op early-return when its
        // `def.<controller>` field is `None`.
        apply_cgroup_cpuset(ctx, state, def)?;
        apply_cgroup_cpuset_mems(ctx, def)?;
        apply_cgroup_cpu(ctx, def)?;
        apply_cgroup_memory(ctx, def)?;
        apply_cgroup_io(ctx, def)?;
        apply_cgroup_pids(ctx, def)?;
        // Materialize the per-WorkSpec values with cgroup-level
        // defaults merged in, resolve workers_pct / num_workers /
        // work_type / affinity, then spawn the synthetic worker
        // handles and the optional userspace payload.
        let resolved_works = resolve_def_works(ctx, state, def)?;
        spawn_def_workers(ctx, state, def, resolved_works)?;
        spawn_def_payload(ctx, state, def)?;
    }
    // Start the host-mode stall monitor once we've spawned workers
    // for the first apply_setup in this step. Skip when:
    // - running inside the guest (the VM-side freeze coordinator
    //   owns stall detection there; the host-mode poller would
    //   read its own guest /proc/sched, which has no relevance);
    // - running under cargo_test_mode (in-process VMM tests share
    //   the host's /proc with the test harness itself and a poller
    //   would observe the harness threads, not the scenario);
    // - the apply_setup pass is routing to the Backdrop slot
    //   (Backdrop workers are scenario-persistent; the per-step
    //   monitor's lifetime would be wrong for them, and a future
    //   task can add a parallel Backdrop monitor if needed);
    // - the monitor is already started for this StepState (an
    //   `Op::AddCgroupDef` that re-enters apply_setup mid-step
    //   should not spawn a second poller — the existing one
    //   already runs);
    // - no workers exist (degenerate scenarios with all worker
    //   spawns deferred to apply_ops or with zero spawns at all).
    maybe_start_stall_monitor(state);
    Ok(())
}

/// Apply [`CgroupDef::cpuset`] for `def`: resolve the spec against
/// the topology, surface the workers_pct / empty-cpuset diagnostics,
/// validate, write the cpuset, and record it on `state`. No-op when
/// `def.cpuset` is `None` (the cgroup inherits the parent cpuset).
fn apply_cgroup_cpuset(
    ctx: &Ctx,
    state: &mut ScenarioState<'_, '_>,
    def: &CgroupDef,
) -> Result<()> {
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
    Ok(())
}

/// Apply [`CgroupDef::cpuset_mems`] for `def`: write the NUMA node
/// binding to `cpuset.mems`. No-op when `def.cpuset_mems` is `None`
/// (the cgroup inherits the parent `cpuset.mems`).
fn apply_cgroup_cpuset_mems(ctx: &Ctx, def: &CgroupDef) -> Result<()> {
    if let Some(ref nodes) = def.cpuset_mems {
        // The cpuset.mems write must succeed before any task
        // moves into the cgroup; cpuset_update_task_spread will
        // SIGKILL or fail allocations otherwise. Surfacing the
        // error here (instead of at move_tasks time) lets the
        // operator see the bad NUMA spec at setup, before the
        // worker spawn pays its cost.
        ctx.cgroups.set_cpuset_mems(&def.name, nodes)?;
    }
    Ok(())
}

/// Apply [`CgroupDef::cpu`] for `def`: validate `cpu.weight` against
/// the kernel range, reject a zero `cpu.max` period / quota, and
/// write `cpu.weight` + `cpu.max`. No-op when `def.cpu` is `None`
/// (both kernel defaults stay in place).
fn apply_cgroup_cpu(ctx: &Ctx, def: &CgroupDef) -> Result<()> {
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
    Ok(())
}

/// Apply [`CgroupDef::memory`] for `def`: write `memory.max`,
/// `memory.high`, `memory.low`, then (only when the user opted in)
/// `memory.swap.max`. No-op when `def.memory` is `None` (all four
/// stay at the kernel defaults).
fn apply_cgroup_memory(ctx: &Ctx, def: &CgroupDef) -> Result<()> {
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
    Ok(())
}

/// Apply [`CgroupDef::io`] for `def`: validate `io.weight` against
/// the kernel range and write it. No-op when `def.io` (or its
/// `weight`) is `None` (the kernel default stays in place).
fn apply_cgroup_io(ctx: &Ctx, def: &CgroupDef) -> Result<()> {
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
    Ok(())
}

/// Apply [`CgroupDef::pids`] for `def`: reject a zero `pids.max` and
/// write `pids.max`. No-op when `def.pids` is `None` (no ceiling).
fn apply_cgroup_pids(ctx: &Ctx, def: &CgroupDef) -> Result<()> {
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
    Ok(())
}

/// Resolve the effective worker specs for `def`: merge cgroup-level
/// defaults into each `WorkSpec`, validate each work's mempolicy
/// (and the mempolicy-vs-cpuset compatibility), then resolve
/// `workers_pct` / `num_workers` / `work_type` / `affinity` into a
/// concrete `WorkSpec` per entry. Returns the resolved specs in
/// declaration order for [`spawn_def_workers`] to spawn.
fn resolve_def_works(
    ctx: &Ctx,
    state: &ScenarioState<'_, '_>,
    def: &CgroupDef,
) -> Result<Vec<crate::workload::WorkSpec>> {
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
            validate_mempolicy_cpuset(&work.mem_policy, work.mpol_flags, resolved, ctx, &def.name)?;
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
        // Op::Spawn(SpawnPlacement::Cgroup) so the two paths
        // produce identical worker counts for the same
        // `(pct, cpuset_size)` pair.
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
        let affinity =
            crate::scenario::intent_for_spawn(&work.affinity, cgroup_cpuset.as_ref(), ctx.topo)?;
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
    Ok(resolved_works)
}

/// Spawn the synthetic worker handles for `def` from the
/// already-resolved `resolved_works`: partition by `pcomm`, spawn
/// the non-pcomm WorkSpecs via the conventional fork path, then
/// spawn one thread-group leader per unique `pcomm` value. Each
/// spawned handle is moved into `def`'s cgroup, started, and pushed
/// onto `state`'s target handle vec.
fn spawn_def_workers(
    ctx: &Ctx,
    state: &mut ScenarioState<'_, '_>,
    def: &CgroupDef,
    resolved_works: Vec<crate::workload::WorkSpec>,
) -> Result<()> {
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
        // `for_scenario_engine` pins `clone_mode = Fork` in the
        // constructor body so the previously-needed
        // `assert_eq!(wl.clone_mode, Fork)` collapses into the
        // type system. Move_tasks below relies on Fork semantics:
        // each worker is its own tgid leader, so writing a
        // worker's TID to `<cgroup>/cgroup.procs` migrates only
        // that worker. Thread mode would drag the scenario
        // runner's tgid into def.name's cgroup per
        // kernel/cgroup/cgroup.c::cgroup_procs_write_start.
        let wl = WorkloadConfig::for_scenario_engine(
            &work,
            n,
            work.affinity.clone(),
            work.work_type.clone(),
        )?;
        tracing::debug!(
            cgroup = %def.name,
            expected_workers = n,
            comm = ?work.comm,
            work_type = ?work.work_type,
            "apply_setup: about to spawn non-pcomm workers"
        );
        let mut h = WorkloadHandle::spawn(&wl)?;
        let pids = h.worker_pids();
        // Append the placement record to a guest-side log file.
        // Empirical reruns ruled out every stderr/serial route
        // for surfacing apply_setup diagnostics in libtest's
        // captured stderr (tracing::*/eprintln drop in the pre-
        // multiport-handshake window; COM1/COM2 PIO writes do
        // not surface either — the host's test-mode serial
        // capture only scans for panic-prefixed lines). A
        // tmpfs file is the established pattern in this test
        // module (the shell probe at cgroup_ops_placement_e2e
        // already writes /tmp/ktstr-cgroup-ops-procs and the
        // test body reads it on assertion failure).
        append_placement_log(&format!(
            "apply_setup: spawned non-pcomm workers cgroup={} count={} pids={:?}",
            def.name,
            pids.len(),
            pids,
        ));
        ctx.cgroups.move_tasks(&def.name, &pids)?;
        h.start();
        state.target_handles().push((def.name.to_string(), h));
    }

    // Spawn one thread-group leader per unique pcomm value.
    // Each leader hosts every WorkSpec that shares its pcomm.
    for pcomm in pcomm_order {
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
        tracing::debug!(
            cgroup = %def.name,
            pcomm = %pcomm,
            workspec_count = works_for_pcomm.len(),
            "apply_setup: about to spawn pcomm-coalesced workers"
        );
        let mut h = WorkloadHandle::spawn_pcomm_cgroup(
            &pcomm,
            container_uid,
            container_gid,
            &works_for_pcomm,
        )?;
        let pids = h.worker_pids();
        // Append the placement record to a guest-side log file —
        // see the non-pcomm sibling above for the routing
        // rationale.
        append_placement_log(&format!(
            "apply_setup: spawned pcomm workers cgroup={} pcomm={} count={} pids={:?}",
            def.name,
            pcomm,
            pids.len(),
            pids,
        ));
        ctx.cgroups.move_tasks(&def.name, &pids)?;
        tracing::debug!(
            cgroup = %def.name,
            pcomm = %pcomm,
            "apply_setup: move_tasks succeeded; about to h.start()"
        );
        h.start();
        tracing::debug!(
            cgroup = %def.name,
            pcomm = %pcomm,
            "apply_setup: h.start() returned; handle pushed"
        );
        state.target_handles().push((def.name.to_string(), h));
    }
    Ok(())
}

/// Spawn the optional userspace payload for `def` inside its cgroup,
/// after the synthetic worker handles are already in place. No-op
/// when `def.payload` is `None`. Rejects a duplicate payload already
/// live in the same cgroup; on success pushes a [`PayloadEntry`]
/// onto `state`'s target payload-handle vec.
fn spawn_def_payload(ctx: &Ctx, state: &mut ScenarioState<'_, '_>, def: &CgroupDef) -> Result<()> {
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
        if let Some(existing) = state.find_live_payload_with_cgroup(payload.name, def.name.as_ref())
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
    Ok(())
}

/// Helper for [`apply_setup`]: spawn the host-mode stall monitor
/// against every step-local worker pid the just-completed setup
/// pass left in [`super::StepState::handles`]. Idempotent —
/// re-invocation when the monitor is already started is a no-op.
///
/// The set of pids is snapshot once at start time; workers added
/// by subsequent ops (e.g. an `Op::Spawn` later in the same step)
/// are NOT picked up by the running thread. This matches the
/// dominant CgroupDef workload shape (every worker for the step
/// is spawned in one apply_setup pass) and avoids the
/// add-while-polling synchronization complexity of a dynamic
/// pid set.
pub(super) fn maybe_start_stall_monitor(state: &mut ScenarioState<'_, '_>) {
    // Backdrop-setup pass: monitor is scoped to the step, not the
    // backdrop. Skip and let the next per-step apply_setup install
    // it if/when step-local workers spawn.
    if state.target_backdrop {
        return;
    }
    // Already started for this step.
    if state.step.stall_monitor.is_some() {
        return;
    }
    // VM-side scenarios use the freeze coordinator's stall plumbing,
    // not the host-mode poller. cargo_test_mode runs the harness
    // itself in the same /proc namespace, so a poller would see
    // unrelated harness threads.
    if guest_comms::is_guest() || crate::cargo_test_mode::cargo_test_mode_active() {
        return;
    }
    // Collect every worker pid in scope — step-local handles
    // spawned by the just-completed apply_setup pass AND
    // backdrop-owned handles that survive across steps. The
    // monitor's lifetime is per-step (re-installed at each step
    // boundary), but the pid set must include backdrop workers
    // because a stalled backdrop worker (e.g. a long-running
    // payload pinned to a contended cpuset) would otherwise be
    // silent across the entire scenario.
    let pids: Vec<libc::pid_t> = state
        .step
        .handles
        .iter()
        .chain(state.backdrop.handles.iter())
        .flat_map(|(_, h)| h.worker_pids())
        .collect();
    if pids.is_empty() {
        return;
    }
    match crate::scenario::host_stall::spawn_monitor(&pids) {
        Ok(handle) => {
            state.step.stall_monitor = Some(handle);
        }
        Err(e) => {
            // Spawn failure is non-fatal — the scenario itself can
            // still run and report worker results. Surface the
            // defect via tracing so an operator can spot the
            // missing stall coverage.
            tracing::warn!(err = %format!("{e:#}"), "host_stall::spawn_monitor failed; stall monitor disabled for this step");
        }
    }
}
