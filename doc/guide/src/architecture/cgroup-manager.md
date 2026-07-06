# CgroupManager and CgroupGroup

sched_ext schedulers see cgroups — weights, cpusets, hierarchy — so
ktstr scenarios create, mutate, and destroy cgroups mid-test, and the
cleanup has to survive kernel-side hangs a buggy scheduler can cause.
`CgroupManager` is that layer: cgroup v2 filesystem operations under a
parent directory, with timeouts and failure caps where the kernel can
wedge. [`CgroupGroup`](#cgroupgroup) is the RAII wrapper that makes
those operations leak-safe under early returns.

<div class="kt-doc-grid">
<div class="kt-doc-card"><strong>Create</strong><p>Set up the ktstr parent, enable controllers, and create scenario cgroups.</p></div>
<div class="kt-doc-card"><strong>Mutate</strong><p>Move tasks, resize cpusets, update weights, and exercise scheduler callbacks.</p></div>
<div class="kt-doc-card"><strong>Clean up</strong><p>Bound teardown so a broken scheduler does not hang the host-side test forever.</p></div>
</div>

Scenarios reach it through `Ctx.cgroups`. The typical pattern pairs it
with the RAII guard:

```rust,ignore
fn custom_scenario(ctx: &Ctx) -> Result<AssertResult> {
    let mut guard = CgroupGroup::new(ctx.cgroups);
    guard.add_cgroup("cg_0", &cpuset)?;

    let mut h = WorkloadHandle::spawn(&config)?;
    ctx.cgroups.move_tasks("cg_0", &h.worker_pids_for_cgroup_procs()?)?;
    h.start(); // workers block until start() is called

    // ... run workload ...

    // `guard` drops at end of scope and removes cg_0 even on error.
    Ok(result)
}
```

Bypass [`CgroupGroup`](#cgroupgroup) only when the cgroup's lifetime
must outlive the current scope; the RAII wrapper removes the cgroup on
every error path, not just the happy one.

## Construction

```rust,ignore
use std::collections::BTreeSet;
use ktstr::cgroup::Controller;

let cgroups = CgroupManager::new("/sys/fs/cgroup/ktstr");
let mut controllers = BTreeSet::new();
controllers.insert(Controller::Cpuset);
controllers.insert(Controller::Cpu);
cgroups.setup(&controllers)?; // create parent dir, enable cpuset + cpu
```

`setup()` creates the parent directory, checks each requested
controller (`Cpuset`, `Cpu`, `Memory`, `Pids`, `Io`) against
`/sys/fs/cgroup/cgroup.controllers`, and enables the requested set on
every ancestor down to and including the parent. A missing controller
fails early with a diagnostic of this shape rather than a later
`ENOENT`:

```text
cgroup controller 'memory' not available at /sys/fs/cgroup/cgroup.controllers;
cgroup.controllers reports {...}. CONFIG_MEMORY_CONTROLLER may be unset, or
the controller is masked at this level of the hierarchy
```

**Walk root.** By default the ancestor walk and task-drain destination
is `/sys/fs/cgroup` (a root-owned tree). `with_walk_root(root)`
retargets both for cgroup-v2 user delegation (systemd `Delegate=yes`,
container `nsdelegate`): the walk stops at the delegated subtree, and
the constructor validates that `parent` sits at or below it.

## Routine operations

| Method | Effect |
|--------|--------|
| `create_cgroup(name)` | Create a child directory; idempotent; supports nested paths |
| `set_cpuset(name, cpus)` / `clear_cpuset(name)` | Write `cpuset.cpus` as a compact range string (`"0-3,5"`); clear inherits the parent |
| `set_cpuset_mems` / `clear_cpuset_mems` | NUMA-node analogue (`cpuset.mems`) |
| `move_task(name, pid)` | Write one PID to the child's `cgroup.procs` |
| `set_cpu_max` / `set_cpu_weight` | cpu controller knobs |
| `set_memory_max` / `set_memory_high` / `set_memory_low` / `set_memory_swap_max` | memory controller knobs |
| `set_io_weight` / `set_pids_max` / `set_freeze` | io / pids / freezer knobs |

The `CgroupDef` builder routes its per-controller setters through
these, and the `CgroupOps` trait abstracts the surface so scenarios
consume `&dyn CgroupOps` (test doubles substitute cleanly). Cgroup
names are validated at every entry point: empty names, leading
slashes, NUL bytes, and `..`/`.` components are rejected.

> [!WARNING]
> For nested paths (`"nested/leaf"`), only `+cpuset` is propagated to
> intermediate cgroups' `subtree_control` — `+cpu`, `+memory`,
> `+pids`, and `+io` are not. A nested leaf exposes `cpuset.*` knobs,
> but driving a memory/pids/io knob on it (e.g.
> `CgroupDef::named("nested/leaf").memory_max(N)`) fails with
> `ENOENT` at apply-setup time. See
> [Troubleshooting](../troubleshooting.md) for the operator-facing
> diagnostic.

## Operations with a story

**`move_tasks(name, pids)`** — moves a batch of PIDs into a child
cgroup. Tolerates `ESRCH` (a task exited between listing and
migration) with a warning, but bails when *every* supplied pid
vanished — silence there would mask a dead-worker cascade. Retries
transient `EBUSY` from sched_ext `cgroup_prep_move` callbacks up to 3
attempts with 100ms backoff, then propagates. And it refuses to write
`cgroup.procs` at all when the destination has `cpuset.cpus` set but
`cpuset.mems.effective` reads empty — a half-configured cgroup whose
kernel behavior is path-dependent; the refusal names the fix
(`set_cpuset_mems` or widen an ancestor).

**`remove_cgroup(name)`** — auto-unfreezes frozen tasks (a frozen task
cannot be reparented), drains tasks to the walk root, waits for
`cgroup.events` to report `populated 0` (inotify-driven, 1s
deadline), then removes the directory. Draining targets the walk root
because the parent has `subtree_control` set, and the kernel's
no-internal-process constraint rejects task writes to a cgroup with
active controllers. Removing a cgroup that does not exist is `Ok`.

**`drain_tasks(name)`** / **`cleanup_all()`** — the pieces of the
above: drain one cgroup's tasks to the walk root; or recursively
remove every child under the parent, depth-first, draining at each
level.

## Failure modes

**Write timeout.** Every cgroup filesystem write runs under a
2-second timeout in a helper thread. A write the kernel never
completes (scheduler bug, wedged freezer) errors with

```text
cgroup write to <path> timed out after 2000ms
```

instead of hanging the test forever.

**Stuck-cgroup cap.** Each failed remove increments an
outstanding-removes counter (successful removes decrement it). Past
10 outstanding, further `remove_cgroup` calls fail fast with a
message of this shape:

```text
remove_cgroup 'cg_42' refused: 11 cgroups outstanding (cap 10); cgroup.procs
draining wedged or churn loop outpacing the kernel's RCU grace period —
bailing to avoid unbounded cgroupfs accumulation
```

This bounds the leak from a churn scenario outrunning the kernel's
cleanup instead of accumulating writer threads without limit.
`outstanding_removes()` exposes the count for diagnostics.

## CgroupGroup

`CgroupGroup` is the RAII guard over `CgroupManager`: it removes every
cgroup it created when it drops, so a workload spawn or any other
operation that fails between cgroup creation and cleanup cannot leak a
cgroup. It is the standard cgroup-lifecycle pattern for custom
scenarios — the worked example at the top of this page is the shape in
full.

<div class="kt-doc-grid">
<div class="kt-doc-card"><strong>Track</strong><p>Remember every cgroup created during a scenario scope.</p></div>
<div class="kt-doc-card"><strong>Guard</strong><p>Drop cleanup runs on both happy paths and early errors.</p></div>
<div class="kt-doc-card"><strong>Warn</strong><p>Teardown errors are logged with context instead of panicking in <code>Drop</code>.</p></div>
</div>

```rust,ignore
#[must_use = "dropping a CgroupGroup immediately destroys the cgroups it manages"]
pub struct CgroupGroup<'a> { /* ... */ }
```

The `#[must_use]` is deliberate: binding the guard to `_` (rather
than `_guard`) drops it immediately and destroys the cgroups before
the workload runs.

**`new(cgroups: &dyn CgroupOps)`** — creates an empty group bound to
any `CgroupOps` implementor (`CgroupManager` in production, an
in-memory fake in tests).

**`add_cgroup(name, cpuset)`** — creates a cgroup and sets its
cpuset. Auto-enables the `Cpuset` controller on the parent's
`cgroup.subtree_control` first — the difference that matters vs
`add_cgroup_no_cpuset`, which creates the cgroup without a cpuset and
without touching controllers. Both track the cgroup for removal on
drop.

**`names()`** — the names of all tracked cgroups.

The helper `setup_cgroups(ctx, n, &wl)` bundles the pattern: it
creates `n` cgroups, spawns workers in each, and returns the handles
alongside the guard.

### Drop behavior

On drop, the group calls `remove_cgroup()` on each tracked cgroup in
reverse insertion order, so nested children are removed before their
parents (a parent still holding child directories fails with
`ENOTEMPTY`).

`ENOENT` is the one errno the drop swallows silently: it means the
directory is already gone, so the post-condition already holds and no
cleanup is owed. (It can legitimately appear via a narrow race
between the existence check and `remove_dir`.) Every other error
surfaces as a `tracing::warn!` record carrying the cgroup name and
the full error chain — the drop never panics, but teardown failures
are visible in logs rather than silently swallowed. The record's
shape:

```text
CgroupGroup::drop: remove_cgroup returned non-ENOENT error
  cgroup=<name> err=<error chain>
  hint=EBUSY: cgroup still has live tasks — workloads were not drained before teardown
```

`EBUSY` at drop means exactly what the hint says: something is still
running in the cgroup — typically a `WorkloadHandle` that outlives
the guard, so its workers were never stopped before teardown. Drop
(or `stop_and_collect`) the handle before the guard goes out of
scope. `EACCES` gets its own hint pointing at cgroup ownership and
delegation.

See also: [Workers and Workloads](workers.md) for worker lifecycle,
[Topology](../concepts/topology.md) for cpuset generation.
