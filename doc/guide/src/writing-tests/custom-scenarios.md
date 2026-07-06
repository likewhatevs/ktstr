# Custom Scenarios

The body of a `#[ktstr_test]` function *is* the scenario — there is
no separate registration step. Most bodies hand control to a canned
scenario or to `execute_defs` / `execute_steps`; a custom scenario
is the same function keeping control and driving cgroups, workers,
and assertions itself.

<div class="kt-doc-grid">
<div class="kt-doc-card"><strong>Prefer ops</strong><p>Use <code>Step</code> and <code>Op</code> for fixed cgroup changes, captures, holds, and checks.</p></div>
<div class="kt-doc-card"><strong>Use custom code</strong><p>Branch on runtime state, compute cpusets dynamically, or assert on raw reports.</p></div>
<div class="kt-doc-card"><strong>Return one result</strong><p>Collect every worker group, merge assertion results, and keep teardown deterministic.</p></div>
</div>

For dynamic scenarios (cgroup creation/removal, cpuset changes),
prefer the [ops/steps system](../concepts/ops.md) over a
hand-written scenario. Reach for custom code only when ops cannot
express the logic:

- Cgroups created, removed, or resized at fixed points in the run —
  ops cover it.
- Different work types, worker counts, or phase-scoped checks per
  step — ops cover it.
- Snapshot captures at chosen points — ops cover it
  ([Snapshots](snapshots.md)).
- Branching on state observed mid-run, computing cpusets from
  runtime conditions, or asserting directly on raw `WorkerReport`s —
  ops cannot; write a custom scenario.

## A worked custom scenario

Shrink one cgroup's cpuset mid-run — a decision ops cannot make,
because the second half's cpuset and the assertion both depend on
runtime state — then assert the scheduler actually moved the
workers:

```rust,ignore
use ktstr::prelude::*;
use ktstr::scenario::*;

#[ktstr_test(llcs = 2, cores = 4, threads = 1, duration_s = 10)]
fn workers_follow_cpuset_shrink(ctx: &Ctx) -> Result<AssertResult> {
    let wl = dfl_wl(ctx);
    // Creates cg_0 and cg_1, spawns and starts workers in each.
    let (mut handles, _guard) = setup_cgroups(ctx, 2, &wl)?;

    // First half: full topology.
    std::thread::sleep(ctx.duration / 2);

    // Mid-run: pin cg_0 to LLC 1 only.
    let llc1 = ctx.topo.llc_aligned_cpuset(1);
    ctx.cgroups.set_cpuset("cg_0", &llc1)?;

    // Second half: workers must migrate off LLC 0.
    std::thread::sleep(ctx.duration / 2);

    let cg0_reports = handles.remove(0).stop_and_collect();
    let migrations: u64 = cg0_reports.iter().map(|r| r.migration_count).sum();
    anyhow::ensure!(
        migrations > 0,
        "cpuset shrink forced no migrations — cg_0 workers never moved"
    );

    let mut result = ctx.assert.assert_cgroup(&cg0_reports, None);
    result.merge(collect_all(handles, &ctx.assert));
    Ok(result)
}
```

Bind the `CgroupGroup` to a named variable (`_guard`) so the cgroups
live until end of scope — see
[CgroupGroup](../architecture/cgroup-group.md) for drop semantics.
Sleeping `ctx.duration` (rather than a hard-coded period) keeps the
scenario composable with `duration_s = N` overrides and the gauntlet
budget controller.

> **Imports:** `setup_cgroups`, `dfl_wl`, `collect_all`, and
> `spawn_diverse` live in `ktstr::scenario`, not in the prelude. The
> `use ktstr::scenario::*;` line is required —
> `use ktstr::prelude::*;` alone does not bring them into scope.

## Helper functions {#helper-functions}

**`setup_cgroups(ctx, n, wl)`** — creates cgroups `cg_0..cg_{n-1}`,
spawns and starts workers in each, and returns
`(Vec<WorkloadHandle>, CgroupGroup)` with handles in cgroup order.

**`collect_all(handles, checks)`** — stops all workers and collects
reports. Per-cgroup telemetry is always produced; only the checks
the caller enabled record assertion outcomes, and with no checks
enabled the result stays `pass` (there is no implicit worker-progress
fallback).

**`dfl_wl(ctx)`** — a `WorkloadConfig` with `ctx.workers_per_cgroup`
workers and default settings (`WorkType::SpinWait`).

**`spawn_diverse(ctx, cgroup_names)`** — spawns rotating
[work types](../concepts/work-types.md) across cgroups (SpinWait,
Bursty, IoSyncWrite, Mixed, YieldHeavy); IoSyncWrite cgroups always
get 2 workers so blocking IO does not drown the scenario.

## Custom work functions

When the built-in [work types](../concepts/work-types.md) don't
generate the load pattern you need, `WorkType::Custom` runs a
user-supplied work function inside each worker. The framework
handles fork, cgroup placement, affinity, and signal setup; the
function owns the work loop and all `WorkerReport` population —
framework telemetry (migration tracking, gap detection, schedstat
deltas) is not provided.

```rust,ignore
use std::sync::atomic::Ordering;
use ktstr::workload::{WorkType, WorkerCtx, WorkerReport};

fn my_workload(ctx: &WorkerCtx) -> WorkerReport {
    let tid: i32 = std::process::id() as i32; // one worker = one process
    let start = std::time::Instant::now();
    let mut work_units = 0u64;
    while !ctx.stop().load(Ordering::Relaxed) {
        // ... custom work ...
        work_units += 1;
    }
    // Start from default() so unpopulated fields stay zero/empty.
    WorkerReport {
        tid,
        work_units,
        iterations: work_units,
        wall_time_ns: start.elapsed().as_nanos() as u64,
        ..WorkerReport::default()
    }
}

let wt = WorkType::custom("my_workload", my_workload);
```

`WorkerCtx` exposes the stop flag (`ctx.stop()`), `ctx.cpus()`,
`ctx.sibling_pids()`, `ctx.cgroup_dir()`, and `ctx.cfg()`. Only
plain function pointers are accepted — they carry no captured state
across the fork boundary; closures are not supported. To pass
per-worker configuration, build the work type with
`WorkType::custom_with(name, run, cfg)`: `CustomCfg` is a Copy POD
payload inherited byte-faithfully across `fork`. For genuinely
shared state, allocate a `MAP_SHARED` region and pass its address
through a `u64` slot.

> [!WARNING]
> Every worker calls `setpgid(0, 0)` after fork, and teardown
> SIGKILLs the worker's whole process group — twice (at collect and
> at handle drop). Any child a custom function spawns inherits that
> pgid and dies with it. A child that must outlive the worker needs
> `setpgid(child_pid, 0)` after fork, or an explicit wait before the
> function returns.

## The Ctx fields scenario authors use

- **`ctx.cgroups`** — create/remove cgroups, set cpusets, move
  tasks. A `&dyn CgroupOps` trait object;
  [CgroupManager](../architecture/cgroup-manager.md) is the
  production implementation.
- **`ctx.topo`** — CPU/LLC/NUMA queries and cpuset generation. See
  [Topology](../concepts/topology.md).
- **`ctx.duration`** — the workload wall-clock budget; sleep against
  this, not a literal.
- **`ctx.settle`** — time to wait after cgroup creation for the
  scheduler to stabilize.
- **`ctx.workers_per_cgroup`** — default per-cgroup worker count
  (`dfl_wl` reads it; there is no `workers` test attribute — set
  counts via `CgroupDef::named("x").workers(n)`).
- **`ctx.sched_pid`** — scheduler PID for liveness checks; `None`
  when running under kernel-default EEVDF.
- **`ctx.assert`** — the merged check set (defaults → scheduler →
  per-test). Pass to `collect_all` / `assert_cgroup` so attribute
  overrides actually apply.
- **`ctx.work_type_override`** — gauntlet-supplied work type applied
  to `CgroupDef`s marked `swappable`; it does not affect `dfl_wl`.
- **`ctx.current_step`** — live phase counter (0 = baseline, 1..=N =
  step ordinal), readable via
  `ctx.current_step.load(Ordering::Acquire)` to gate behavior on
  phase; periodic captures are stamped with the same value.

The remaining fields are framework wiring; see the
[Ctx rustdoc](https://ktstr.dev/rustdoc/ktstr/scenario/struct.Ctx.html).
