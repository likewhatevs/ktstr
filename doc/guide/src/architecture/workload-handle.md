# WorkloadHandle

`WorkloadHandle` is the RAII handle to spawned worker processes. It
manages the lifecycle of forked workers: spawning, start signaling,
stop/collection, and cleanup.

```rust,ignore
use ktstr::prelude::*;

#[must_use = "dropping a WorkloadHandle immediately tears down all worker tasks"]
pub struct WorkloadHandle { /* ... */ }
```

## Spawning

```rust,ignore
let config = WorkloadConfig {
    num_workers: 4,
    work_type: WorkType::Mixed,
    ..Default::default()
};
let mut handle = WorkloadHandle::spawn(&config)?;
```

Set only the fields that matter for the test and let
`..Default::default()` fill in the rest. The spread-default form is the
canonical style for `WorkloadConfig` — it keeps examples pinned to
intent (`num_workers`, `work_type`) and has already absorbed additions
to `WorkloadConfig` (the NUMA memory-policy fields) without rotting.
Consult the `WorkloadConfig` rustdoc for the current field list.

The spread-default pattern is safe for `WorkloadConfig` specifically
because its `Default::default()` produces a known-good single-worker
`SpinWait` baseline that runs without further setup. Do NOT extrapolate
this guidance to every ktstr type without checking each type's
`Default` semantics — some have known footguns (e.g. `CgroupDef::default()`
produces `name = "cg_0"` which collides with the conventional first
cgroup name in most scenarios). Prefer the named constructors
(`CgroupDef::named(...)`, `Setup::defs(...)`, etc.) for types where the
`Default` is not unambiguously useful.

`spawn()` forks `num_workers` child processes. Each child installs a
SIGUSR1 handler, then blocks on a pipe waiting for the start signal.
Workers do not begin their workload until `start()` is called.

For grouped work types (the full set: `PipeIo`, `CachePipe`,
`FutexPingPong`, `FutexFanOut`, `FanOutCompute`, `MutexContention`,
`ThunderingHerd`, `PriorityInversion`, `ProducerConsumerImbalance`,
`RtStarvation`, `AsymmetricWaker`, `WakeChain`, `SignalStorm`,
`PreemptStorm`, `EpollStorm`, `SmtSiblingSpin`), `spawn()` validates
that `num_workers` is divisible by the work type's group size (each
variant exposes a `worker_group_size()` accessor) and sets up the
inter-worker communication the variant requires: pipes for
`PipeIo`/`CachePipe`, shared mmap pages for the futex / waker /
contention families (`FutexPingPong`, `FutexFanOut`, `FanOutCompute`,
`MutexContention`, `ThunderingHerd`, `PriorityInversion`,
`ProducerConsumerImbalance`, `AsymmetricWaker`, `WakeChain`,
`RtStarvation`, `SignalStorm`, `PreemptStorm`, `EpollStorm`).

## Methods

**`worker_pids() -> Vec<libc::pid_t>`** -- PIDs of all worker
processes. Used with `CgroupManager::move_task()` or `move_tasks()`
to place workers in cgroups before starting them.

**`start()`** -- signals all workers to begin their workload by writing
to their start pipes. Idempotent: calling it twice has no effect.
Call this after moving workers into their target cgroups.

**`set_affinity(idx, cpus) -> Result<()>`** -- sets CPU affinity for
the worker at index `idx` via `sched_setaffinity`. Use this for
per-worker pinning outside any cgroup, or when you need to change one
worker's affinity without disturbing the rest. When all workers in a
cgroup should share the same CPU set, prefer
[`CgroupGroup::add_cgroup`](cgroup-group.md) — it creates the cgroup,
writes `cpuset.cpus` once for the whole cgroup, and RAII-removes the
cgroup on drop (including error paths). Reach for
[`CgroupManager::set_cpuset`](cgroup-manager.md) directly only when
the cgroup's lifetime must outlive the current scope; the RAII
wrapper is the default because it cleans up on every error path.

**`snapshot_iterations() -> Vec<u64>`** -- reads all workers' current
iteration counts from a shared memory region (MAP_SHARED). Each count
is monotonically increasing, read with relaxed ordering. Returns an
empty vec if no workers were spawned. Call periodically during the
workload's run window to sample forward progress (e.g. to detect stalls
or compute instantaneous rates); the final per-worker totals come back
through `stop_and_collect()`.

**`stop_and_collect(self) -> Vec<WorkerReport>`** -- sends SIGUSR1 to
all workers, reads their serialized `WorkerReport` from report pipes,
and waits for exit. Auto-starts workers if `start()` was not called.
Workers that do not respond within a shared 5-second deadline are
killed with SIGKILL. Consumes the handle.

## Typical usage

```rust,ignore
// 1. Spawn workers (blocked, waiting for start signal)
let mut handle = WorkloadHandle::spawn(&config)?;

// 2. Move workers into their target cgroup. `cgroup.procs` is
//    tgid-scoped, so use `worker_pids_for_cgroup_procs()` — it
//    bails for Thread-mode workers (whose pids share the harness's
//    tgid) and points at `cgroup.threads` instead. Plain
//    `worker_pids()` returns the raw pid set without the
//    cgroup-procs safety check.
ctx.cgroups.move_tasks("cg_0", &handle.worker_pids_for_cgroup_procs()?)?;

// 3. Signal workers to start
handle.start();

// 4. Wait for workload duration
std::thread::sleep(ctx.duration);

// 5. Stop workers and collect telemetry
let reports: Vec<WorkerReport> = handle.stop_and_collect();
```

## Drop behavior

Dropping a `WorkloadHandle` without calling `stop_and_collect()` sends
SIGKILL to all child processes and waits for them. This prevents
orphaned worker processes on error paths. Shared mmap regions (futex
pages and iteration counters) are unmapped on drop.

See also: [CgroupManager](cgroup-manager.md) for cgroup operations,
[CgroupGroup](cgroup-group.md) for RAII cleanup,
[TestTopology](../concepts/topology.md) for cpuset generation,
[Worker Processes](workers.md) for the two-phase start protocol and
telemetry details.
