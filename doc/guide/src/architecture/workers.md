# Workers and Workloads

Workers are the processes that generate load for scenarios. They run
inside the VM, each placed in a cgroup, and each one reports detailed
telemetry (`WorkerReport`) when the workload stops. `WorkloadHandle`
is the RAII handle that owns their whole lifecycle: spawn → place →
start → stop and collect → drop.

<div class="kt-figure"><svg width="700" height="300" viewBox="0 0 700 300" role="img" aria-label="Worker and cgroup process model. Inside the guest VM, workers run as fork-mode child processes, each in its own process group (pid = pgid), grouped into per-cgroup cpuset boxes. Each worker produces a WorkerReport, collected by the host and rolled up per cgroup into stats. At teardown, SIGKILL is delivered to each worker's process group unconditionally — a process-group sweep. A footgun: a child a Custom work function spawns inherits the worker's pgid and is SIGKILLed too unless setpgid detaches it.">
  <defs><marker id="kt-arr11" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="var(--fg)"/></marker></defs>
  <rect x="14" y="34" width="456" height="212" rx="12" fill="none" stroke="var(--kt-rule)" stroke-width="1.4"/>
  <text x="30" y="26" font-size="11" font-weight="700" fill="var(--fg)">guest VM</text>
  <rect x="32" y="54" width="200" height="150" rx="10" fill="var(--kt-accent-soft)" stroke="var(--kt-rule)" stroke-width="1.2"/>
  <text x="46" y="74" font-size="10.5" font-weight="700" fill="var(--fg)">cg_0 · cpuset</text>
  <rect x="48" y="84" width="168" height="30" rx="6" fill="var(--bg)" stroke="var(--kt-accent)" stroke-width=".8"/>
  <text x="60" y="103" font-size="9.5" fill="var(--fg)">worker · pid = pgid</text>
  <rect x="48" y="120" width="168" height="30" rx="6" fill="var(--bg)" stroke="var(--kt-accent)" stroke-width=".8"/>
  <text x="60" y="139" font-size="9.5" fill="var(--fg)">worker · pid = pgid</text>
  <text x="46" y="172" font-size="8.5" fill="var(--fg)" opacity=".7">fork: own tgid → cgroup.procs</text>
  <text x="46" y="186" font-size="8.5" fill="var(--fg)" opacity=".7">each calls setpgid(0, 0)</text>
  <rect x="252" y="54" width="200" height="150" rx="10" fill="var(--kt-accent-soft)" stroke="var(--kt-rule)" stroke-width="1.2"/>
  <text x="266" y="74" font-size="10.5" font-weight="700" fill="var(--fg)">cg_1 · cpuset</text>
  <rect x="268" y="84" width="168" height="30" rx="6" fill="var(--bg)" stroke="var(--kt-accent)" stroke-width=".8"/>
  <text x="280" y="103" font-size="9.5" fill="var(--fg)">worker · pid = pgid</text>
  <rect x="268" y="120" width="168" height="30" rx="6" fill="var(--bg)" stroke="var(--kt-accent)" stroke-width=".8"/>
  <text x="280" y="139" font-size="9.5" fill="var(--fg)">worker (Custom) · pgid</text>
  <rect x="286" y="160" width="150" height="26" rx="6" fill="none" stroke="var(--kt-rule)" stroke-width="1.1" stroke-dasharray="4 3"/>
  <text x="298" y="177" font-size="9" fill="var(--fg)" opacity=".8">child: execv / subshell</text>
  <path d="M352 150 L 352 158" stroke="var(--fg)" stroke-width="1.1" opacity=".6" marker-end="url(#kt-arr11)"/>
  <rect x="512" y="70" width="172" height="86" rx="11" fill="var(--kt-accent-soft)" stroke="var(--kt-accent)" stroke-width="1.6"/>
  <text x="528" y="94" font-size="11" font-weight="700" fill="var(--kt-accent)">host · collector</text>
  <text x="528" y="112" font-size="9" fill="var(--fg)" opacity=".85">WorkerReport per worker,</text>
  <text x="528" y="126" font-size="9" fill="var(--fg)" opacity=".85">rolled up per cgroup</text>
  <text x="528" y="140" font-size="9" fill="var(--fg)" opacity=".85">→ stats</text>
  <path d="M470 113 L 508 113" stroke="var(--kt-accent)" stroke-width="1.5" marker-end="url(#kt-arr11)"/>
  <text x="598" y="63" font-size="8.5" fill="var(--kt-accent)" text-anchor="middle">stop_and_collect</text>
  <line x1="14" y1="256" x2="684" y2="256" stroke="var(--kt-rule)" stroke-width="1"/>
  <text x="14" y="274" font-size="9.5" fill="var(--fg)" opacity=".8">teardown: SIGKILL → each worker's process group (pgid) — a process-group sweep.</text>
  <text x="14" y="289" font-size="9.5" fill="var(--fg)" opacity=".8">footgun: a Custom child inherits the worker's pgid and is killed too, unless setpgid detaches it.</text>
</svg></div>

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
`..Default::default()` fill in the rest — `WorkloadConfig`'s default
is a known-good single-worker `SpinWait` baseline, and the spread
form keeps examples pinned to intent as fields are added. Consult the
`WorkloadConfig` rustdoc for the current field list. (Do not
extrapolate this to every ktstr type: `CgroupDef` deliberately has no
`Default` because a derived empty name would silently produce an
invalid cgroup — use `CgroupDef::named(...)`.)

The worker-creation primitive is selected by `CloneMode`:

- **`CloneMode::Fork`** (default): forks N child processes; each
  child installs a SIGUSR1 handler, then blocks on a pipe waiting for
  the start signal. Each worker has its own tgid, so `cgroup.procs`
  placement is per-worker.
- **`CloneMode::Thread`**: spawns N threads inside the harness; each
  blocks on a rendezvous channel until `start()`. Workers share the
  harness's tgid, so cgroup placement must go through
  `cgroup.threads` (see Placement below).

For grouped work types (`PipeIo`, `FutexPingPong`, `FutexFanOut`,
`MutexContention`, `ThunderingHerd`, and the rest of the
communicating families), `spawn()` validates that `num_workers` is
divisible by the variant's group size and sets up the inter-worker
plumbing the variant requires (pipes, shared futex pages). See
[Work Types](../concepts/work-types.md) for choosing a variant.

`pcomm` containers are not created by `spawn()` — it bails when a
composed `WorkSpec::pcomm` is set, pointing at
`WorkloadHandle::spawn_pcomm_cgroup` (or the `CgroupDef::pcomm`
path), which spawns one thread-group-leader process hosting N worker
threads so the group's `comm` matches the pcomm name.

## Two-phase start

Workers wait for a "start" signal after spawn:

1. Parent spawns the worker (fork or thread), which blocks.
2. Parent moves the worker to its target cgroup.
3. Parent calls `start()`, releasing all workers at once.

This ensures workers run inside their target cgroup from the first
instruction of their workload — there is no window where load runs in
the wrong cgroup and pollutes the measurement.

## Placement

```rust,ignore
// 1. Spawn workers (blocked, waiting for start signal)
let mut handle = WorkloadHandle::spawn(&config)?;

// 2. Move workers into their target cgroup. `cgroup.procs` is
//    tgid-scoped, so use `worker_pids_for_cgroup_procs()` — it
//    bails for Thread-mode workers (whose pids share the harness's
//    tgid) and points at `cgroup.threads` instead. Plain
//    `worker_pids()` returns the raw pid set without that check.
ctx.cgroups.move_tasks("cg_0", &handle.worker_pids_for_cgroup_procs()?)?;

// 3. Signal workers to start
handle.start();

// 4. Wait for the workload duration
std::thread::sleep(ctx.duration);

// 5. Stop workers and collect telemetry
let reports: Vec<WorkerReport> = handle.stop_and_collect();
```

Step 2's Thread-mode bail exists because the kernel resolves any pid
written to `cgroup.procs` to its thread-group leader — writing a
Thread-mode worker's pid there would migrate the *entire harness*
into the test cgroup.

Which placement tool, when:

| You want to | Use |
|-------------|-----|
| Pin one worker to CPUs | `handle.set_affinity(idx, cpus)` |
| Pin a whole cgroup of workers | [`CgroupGroup::add_cgroup`](cgroup-group.md) (writes `cpuset.cpus` once, RAII-removes on drop) |
| A cgroup that outlives the current scope | [`CgroupManager`](cgroup-manager.md) directly |

## Start and observing progress

**`start()`** signals all workers to begin (a start-pipe byte for
fork children, a channel send for threads). Idempotent — the second
call is a no-op. Call it after cgroup placement.

**`snapshot_iterations()`** reads every worker's current iteration
count from a shared-memory region without stopping anything. Call it
periodically during the run window to detect stuck workers or compute
instantaneous rates; final totals come from `stop_and_collect()`.

## Stop and collect

**`stop_and_collect(self)`** signals workers to stop (SIGUSR1 flips a
stop flag in fork children; a per-thread flag for thread workers),
then collects each worker's `WorkerReport` — read from a report pipe
under a shared 5-second deadline for fork children, returned from the
thread join for thread workers. It auto-starts workers if `start()`
was never called, and consumes the handle — workers cannot be
restarted.

A worker that fails to produce a report (died, timed out, wrote
corrupt data) gets a zeroed sentinel report: `completed: false`,
`work_units: 0`, and `exit_info: Some(_)` preserving how it ended
(`Exited(code)` / `Signaled(sig)` / `TimedOut` / `WaitFailed` /
`Panicked`). Live-worker reports always carry `exit_info: None`, so
consumers can distinguish "ran to completion and did nothing" from
"died before reporting" — and the zero-work-units gate counts dead
workers as failed-progress reports instead of silently passing.

After collection, SIGKILL is delivered to each fork worker's process
group unconditionally to reap stragglers.

> [!WARNING]
> The teardown SIGKILL is a process-*group* sweep. Every worker calls
> `setpgid(0, 0)` after fork, so any child a `Custom` work function
> spawns (a helper via `execv`, a subshell) inherits the worker's
> pgid and is SIGKILLed at teardown. A child that must outlive the
> worker needs `setpgid(child_pid, 0)` after fork, or an explicit
> wait before the worker returns its report. Details in
> [Work Types — Custom](../concepts/work-types.md).

## Drop behavior

Dropping a `WorkloadHandle` without calling `stop_and_collect()`
sends SIGKILL to all child processes (the same process-group sweep)
and waits for them, so error paths never leak orphaned workers.
Shared mmap regions (futex pages, iteration counters) are unmapped on
drop. The type is `#[must_use]` — an accidentally dropped handle
tears its workload down immediately.

## Telemetry: WorkerReport

Each worker produces one `WorkerReport`. The fields you will actually
assert on:

| Field | Meaning | Populated by |
|-------|---------|--------------|
| `work_units` | Cumulative work counter; feeds the zero-work-units gate | Every framework work type |
| `iterations` | Outer-loop count; feeds throughput rates | Every framework work type |
| `cpu_time_ns` / `wall_time_ns` / `off_cpu_ns` | On-CPU vs total vs off-CPU time | Every framework work type |
| `migration_count`, `migrations`, `cpus_used` | Cross-CPU movement | Checked every 1024 work units |
| `max_gap_ms` (+ `_cpu`, `_at_ms`) | Longest wall-clock gap between checkpoints — the stuck/preemption tell | Every framework work type |
| `wake_latencies_ns` + `wake_sample_total` | Per-wakeup latency samples | Blocking work types only (futex, pipe, I/O, yield, sleep) |
| `iteration_costs_ns` + `iteration_cost_sample_total` | Per-iteration wall-clock cost | Pure-compute variants (`AluHot`, `SmtSiblingSpin`, `IpcVariance`) |
| `timer_latencies_ns` + `timer_sample_total` | Timer-wake jitter vs absolute deadline | `TimerLatency` only |
| `schedstat_run_delay_ns` / `schedstat_run_count` / `schedstat_cpu_time_ns` | `/proc/self/schedstat` deltas over the work loop | Every framework work type |
| `numa_pages`, `vmstat_numa_pages_migrated` | Per-node residency and migration counters | Every framework work type; feed the [NUMA checks](../concepts/checking.md) |
| `completed`, `exit_info` | Natural end vs sentinel (see above) | Framework |
| `affinity_error`, `sched_policy_error` | Setup calls that failed; worker ran anyway | Framework |

Consult the `WorkerReport` rustdoc for the full field list and
per-field semantics — the table above summarizes, the rustdoc is
authoritative.

Semantics worth knowing before asserting:

- **Sampling caps.** `wake_latencies_ns` is reservoir-sampled and
  capped at 100,000 entries; `wake_sample_total` keeps counting past
  the cap. Report "total wakeups" from the total; compute percentiles
  from the vector. (The cap is pinned by a unit test —
  `max_wake_samples_pins_doc_value` — so this paragraph cannot
  silently rot.)
- **`schedstat_run_count` is pcount, not context switches.** It
  increments each time the scheduler picks the task to run; a task
  that keeps running on one CPU does not advance it. For true
  context-switch counts read `/proc/<pid>/status`.
- **Checkpoint cadence.** Migration and gap checks run when
  `work_units` is a multiple of 1024, so a variant contributing N
  units per outer iteration checks every `1024 / gcd(N, 1024)`
  iterations. Per-variant unit contributions live in the worker
  source and its rustdoc; the key defaults are pinned by unit tests.
- **`Custom` populates nothing.** The framework fills no telemetry
  for `WorkType::Custom` — migration tracking, gap detection,
  schedstat deltas, and iteration counts exist only if the user's
  `run` function fills them.

## What the reports become

Test output rolls `WorkerReport`s up per cgroup. From a real failing
run:

<!-- captured: cargo ktstr test --kernel 7.0 -E 'test(throughput_gate)' | ktstr 0.23.0 | kernel 7.0.14 -->
```text
--- stats ---
2 workers, 4 cpus, 2 migrations, worst_spread=0.0%, worst_gap=21ms
  cg0: workers=1 cpus=2 spread=0.0% gap=10ms migrations=1 iter=209600
  cg1: workers=1 cpus=2 spread=0.0% gap=21ms migrations=1 iter=189252
```

`iter` sums `iterations`, `gap` is the worst `max_gap_ms`,
`migrations` sums `migration_count`, and `cpus` counts distinct
`cpus_used` entries. Reading this one: both cgroups made steady
progress with sub-25ms worst gaps — the workers were scheduled fine;
this failure came from a throughput floor, not the zero-work-units gate. A report
showing `migrations=0` plus a growing `gap` on a multi-CPU cpuset
would tell the opposite story: the scheduler is not spreading.

How reports become verdicts — thresholds, defaults, and the merge
rules — is [Checking](../concepts/checking.md)'s territory.
