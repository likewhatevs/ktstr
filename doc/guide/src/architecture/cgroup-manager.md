# CgroupManager

`CgroupManager` manages cgroup v2 filesystem operations. It creates,
configures, and removes cgroups under a parent directory.

```rust,ignore
use ktstr::prelude::*;

pub struct CgroupManager {
    parent: PathBuf,
    walk_root: PathBuf,
    outstanding_removes: AtomicUsize,
}
```

The `outstanding_removes` counter is a stuck-cgroup safety cap.
When a `remove_cgroup` write times out (the kernel is sometimes
slow to reap freezer or BPF state), the counter increments. Once
more than `MAX_OUTSTANDING_REMOVES` (= 10) removes have failed,
subsequent `remove_cgroup` calls return `Err` immediately rather
than continuing to leak per-call writer threads. `outstanding_removes()`
exposes the count for diagnostics.

## Construction

```rust,ignore
use std::collections::BTreeSet;
use ktstr::cgroup::Controller;

let cgroups = CgroupManager::new("/sys/fs/cgroup/ktstr");
let mut controllers = BTreeSet::new();
controllers.insert(Controller::Cpuset);
controllers.insert(Controller::Cpu);
cgroups.setup(&controllers)?; // create parent dir, enable cpuset + cpu controllers
```

`new()` sets the parent path. `setup()` takes a
`&BTreeSet<Controller>` (variants: `Cpuset`, `Cpu`, `Memory`,
`Pids`, `Io` — the controller name tokens written to
`cgroup.subtree_control`), creates the parent directory if it does
not exist, checks each requested controller against
`/sys/fs/cgroup/cgroup.controllers` and bails with a clear
"controller X not available" error if the kernel didn't expose it,
then enables the requested controllers on every ancestor from
`/sys/fs/cgroup` down to AND INCLUDING the parent by writing to
each level's `cgroup.subtree_control`. An empty set creates the
directory and returns without touching `subtree_control`. The
deterministic `BTreeSet` iteration order keeps the rendered
subtree_control write stable between runs.

## Walk root

`walk_root` defaults to `/sys/fs/cgroup` (Mode A, a root-owned tree).
`with_walk_root(root)` retargets it for cgroup-v2 user delegation
(systemd `Delegate=yes` / container `nsdelegate`, Mode B/C): it bounds
both the `setup` `cgroup.subtree_control` ancestor walk and the drain
destination to the delegated subtree, and the constructor validates
that `parent` is at or below `walk_root` and rejects `..` components.

## Methods

**`parent_path() -> &Path`** -- returns the parent cgroup directory path.

**`walk_root() -> &Path`** -- returns the walk-root directory (default
`/sys/fs/cgroup`; the delegation boundary when set via `with_walk_root`).

**`with_walk_root(root) -> Result<Self>`** -- retargets the walk root for
cgroup-v2 user delegation; validates the parent is at or below `root`.

**`create_cgroup(name)`** -- creates a child cgroup directory.
Idempotent: no error if the directory already exists. Supports nested
paths (e.g. `"nested/deep"`). For nested paths, enables ONLY
`+cpuset` on intermediate cgroups' `subtree_control` — `+cpu`,
`+memory`, `+io`, and `+pids` are NOT propagated. Tests that drive
non-cpuset controllers on a nested leaf
(e.g. `CgroupDef::named("nested/leaf").memory_max(N)`) get `ENOENT`
at apply-setup time when the missing controller knob is written; see
[Cgroup controller not enabled](../troubleshooting.md#cgroup-controller-not-enabled)
for the operator-facing diagnostic shape.

**`remove_cgroup(name)`** -- auto-unfreezes any frozen tasks (a
frozen task cannot be reparented), drains tasks from the child
cgroup to the walk-root cgroup (default `/sys/fs/cgroup`; the
delegated subtree root under cgroup-v2 delegation), then waits for
`cgroup.events` to report `populated 0` via inotify (1s deadline)
before removing the directory. No error if the cgroup does not
exist. Returns `Err` once the outstanding-remove count exceeds the
cap (10).

**`set_cpuset(name, cpus)`** -- writes `cpuset.cpus` for a child cgroup.
The `BTreeSet<usize>` is formatted as a compact range string via
`TestTopology::cpuset_string()` (e.g. `"0-3,5,7-9"`).

**`clear_cpuset(name)`** -- writes an empty string to `cpuset.cpus`,
which inherits the parent's cpuset.

**`move_task(name, pid)`** -- writes a single PID to the child cgroup's
`cgroup.procs`.

**`move_tasks(name, pids)`** -- moves all PIDs from a slice into the
child cgroup. Tolerates ESRCH (task exited between listing and
migration) with a warning. Up to 3 attempts total (2 retries after
the initial try) with 100ms backoff for transient EBUSY from
sched_ext BPF `cgroup_prep_move` callbacks. Bails BEFORE writing
`cgroup.procs` when `cpuset.cpus` is non-empty but
`cpuset.mems.effective` reads empty — that combination would
silently strand the move with no actionable error, so the bail is
load-bearing. Propagates EBUSY after retries exhausted.

In addition to the public methods above, `CgroupManager` exposes a
broader knob surface for each controller: `set_cpuset_mems` /
`clear_cpuset_mems` (`cpuset.mems` — the NUMA-node analogue of
set_cpuset), `set_cpu_max`
/ `set_cpu_weight` (cpu controller), `set_memory_max` / `set_memory_high`
/ `set_memory_low` / `set_memory_swap_max` (memory + memory.swap),
`set_io_weight` (io controller), `set_freeze` (cgroup.freeze), and
`set_pids_max` (pids controller). The `CgroupDef` builder routes its
per-controller setters through these. The `CgroupOps` trait abstracts
the surface so test scenarios consume `&dyn CgroupOps` (allowing
test-double substitution); `validate_cgroup_name` rejects empty
names, leading slash, NUL bytes, `..`/`.` components, and
leading-dot components at all entry points.

**`drain_tasks(name)`** -- moves all tasks from a child cgroup to the
walk-root cgroup (default `/sys/fs/cgroup`; the delegated subtree root
under cgroup-v2 delegation) by reading `cgroup.procs` and writing each
PID to the walk-root's `cgroup.procs`. Drains to the walk root
because the parent has `subtree_control` set and the kernel's
no-internal-process constraint rejects writes to a cgroup with
active controllers.

**`cleanup_all()`** -- recursively removes all child cgroups under the
parent (depth-first), draining tasks at each level. Keeps the parent
directory itself.

## Timeout protection

All cgroup filesystem writes use a 2-second timeout. The write runs
in a spawned thread; if it does not complete within the timeout, the
caller gets an error. This prevents test hangs when cgroup operations
block in the kernel (e.g. during scheduler reconfigurations).

## Usage in scenarios

Scenarios access `CgroupManager` through `Ctx.cgroups`. The typical
pattern is:

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

Bypass [`CgroupGroup`](cgroup-group.md) only when you need to hand the
cgroup's lifetime to a different owner; the RAII wrapper is the default
because it removes the cgroup on every error path, not just the happy
path.

See also: [CgroupGroup](cgroup-group.md) for RAII cleanup,
[WorkloadHandle](workload-handle.md) for worker lifecycle,
[TestTopology](../concepts/topology.md) for cpuset generation.
