# Custom Scenarios

For dynamic scenarios (cgroup creation/removal, cpuset changes), prefer
the [ops/steps system](../concepts/ops.md) over a hand-written custom
scenario function.

A custom scenario is just a `fn(&Ctx) -> Result<AssertResult>` that
runs in place of one of the canned `scenarios::*` entry points. Use
this shape only when you need logic that the ops system cannot
express.

## Writing a custom scenario

```rust,ignore
use ktstr::prelude::*;
use ktstr::scenario::*;

fn my_custom_scenario(ctx: &Ctx) -> Result<AssertResult> {
    let wl = dfl_wl(ctx);
    let (handles, _guard) = setup_cgroups(ctx, 2, &wl)?;

    // Custom logic: resize cpusets, move workers, etc.
    std::thread::sleep(ctx.duration);

    Ok(collect_all(handles, &ctx.assert))
}
```

## Helper functions

**`setup_cgroups(ctx, n, wl)`** -- creates N cgroups, spawns workers,
returns `Result<(Vec<`[`WorkloadHandle`](../architecture/workload-handle.md)`>, `[`CgroupGroup`](../architecture/cgroup-group.md)`)>`.
Bind the `CgroupGroup` to a named variable (e.g. `_guard`) so it
lives until end of scope.
See [CgroupGroup](../architecture/cgroup-group.md) for drop semantics.

> **Imports:** `setup_cgroups` and `dfl_wl` live in `ktstr::scenario`,
> not in the prelude. The `use ktstr::scenario::*;` line in the
> example above is required — `use ktstr::prelude::*;` alone does
> not bring them into scope.

**`collect_all(handles, checks)`** -- stops all workers, collects
reports, runs worker-level checks when `checks.has_worker_checks()`
is true; when no worker checks are configured the merge step is
skipped entirely and the result stays `pass` (no implicit
"`assert_not_starved` fallback"). Merges results: if any worker
group fails an enabled check, the overall result fails.

**`dfl_wl(ctx)`** -- creates a `WorkloadConfig` with
`ctx.workers_per_cgroup` workers and default settings.

**`spawn_diverse(ctx, cgroup_names)`** -- spawns different
[work types](../concepts/work-types.md) across cgroups, rotating
through (SpinWait, Bursty{50ms burst / 100ms sleep}, IoSyncWrite,
Mixed, YieldHeavy). Each cgroup uses `ctx.workers_per_cgroup`
workers except IoSyncWrite cgroups, which always use 2 workers so
blocking IO does not drown the scenario.

## The Ctx struct

Custom scenarios receive a `Ctx` reference:

```rust,ignore
#[non_exhaustive]
pub struct Ctx<'a> {
    pub cgroups: &'a dyn CgroupOps,
    pub topo: &'a TestTopology,
    pub duration: Duration,
    pub workers_per_cgroup: usize,
    pub sched_pid: Option<libc::pid_t>,
    pub settle: Duration,
    pub work_type_override: Option<WorkType>,
    pub assert: Assert,
    pub wait_for_map_write: bool,
    pub current_step: Arc<AtomicU16>,
    pub entry_name: Option<&'static str>,
}
```

`current_step` is the live phase counter — `0` during BASELINE,
`1..=N` while step ordinal N is running. Read via
`ctx.current_step.load(Ordering::Acquire)` to gate scenario-side
behavior on phase. The host-side periodic-capture pipeline stamps
each capture with the same value.

**`cgroups`** -- create/remove cgroups, set cpusets, move tasks. The
slot is a `&dyn CgroupOps` trait object, not a concrete
[`CgroupManager`](../architecture/cgroup-manager.md), so tests can
substitute a no-op double for host-only scenarios while production
paths receive the real manager. Method signatures are defined on
`CgroupOps`; see `CgroupManager` for the production implementation.

**`topo`** -- query CPU topology (LLCs, NUMA nodes, memory info,
distances). Provides CPU enumeration, LLC/NUMA partitioning, cpuset
generation, and inter-node distance queries. See
[TestTopology](../concepts/topology.md) for the full API reference.

**`sched_pid`** -- scheduler process ID (`Option<libc::pid_t>`) for
liveness checks. `None` when the test runs without an scx scheduler
(the EEVDF default path has no userspace scheduler binary). Unwrap
or `is_some_and(...)` before passing to `process_alive` or
`kill(Pid::from_raw(pid), None)`.

**`settle`** -- time to wait after cgroup creation for the scheduler
to stabilize.

**`duration`** -- total scenario wall-clock budget. Custom scenarios
that hold workers running for a fixed period should sleep for this
duration after `start()` and before `stop_and_collect()`. Honoring
this value keeps custom scenarios composable with the gauntlet
budget controller and with `#[ktstr_test(duration_s = N)]` overrides.

**`workers_per_cgroup`** -- default per-cgroup worker count derived
from the test's `workers` attribute (or the topology when unset).
`dfl_wl(ctx)` pre-fills this. Custom scenarios that hand-build a
`WorkloadConfig` should source `num_workers` from this field unless
the test explicitly wants a different value.

**`work_type_override`** -- gauntlet-supplied or programmatic
`WorkType` to swap in for any `CgroupDef` whose `swappable = true`
and for the `dfl_wl` default. `None` keeps the per-cgroup work type
the scenario specified.

**`assert`** -- the merged `Assert` (`default_checks() →
scheduler → per-test`) the test framework expects the scenario to
evaluate against. Pass to `collect_all(handles, &ctx.assert)` or
the manual `assert.assert_cgroup(...)` paths.

**`wait_for_map_write`** -- when `true`, the framework will not
spawn workers until the test's `bpf_map_write` declarations have
been applied. Custom scenarios that don't gate worker spawn on BPF
map state can ignore this field; the framework wires it
automatically.

## Checking in custom scenarios

Use `Assert` for both direct report checking and ops-based scenarios.
Call `assert.assert_cgroup(reports, cpuset)` for manual report
collection, or use `execute_steps_with()` for ops-based scenarios. See
[Checking](../concepts/checking.md#worker-checks-via-assert).
