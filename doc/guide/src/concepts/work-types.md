# Work Types

`WorkType` decides what each worker process does — and each variant
targets a specific kernel scheduling path, so a test pins down the
code path a regression lives in. `ForkExit` hammers
`wake_up_new_task` and the exit path (`do_group_exit` /
`wait_task_zombie`); `AffinityChurn` drives `affine_move_task` and
`migration_cpu_stop`. If you know which path your scheduler change
touches, there is usually a work type aimed at it.

<div class="kt-doc-grid">
<div class="kt-doc-card"><strong>CPU paths</strong><p>Spin, yield, fork/exit, affinity churn, scheduling-class transitions, and SMT pressure.</p></div>
<div class="kt-doc-card"><strong>Wake paths</strong><p>Pipe, futex, epoll, fan-out, timers, IRQ/softirq, and request/response workloads.</p></div>
<div class="kt-doc-card"><strong>Locality + I/O</strong><p>Cache pressure, NUMA working sets, page faults, block I/O, taobench, and schbench parity.</p></div>
</div>

## Choosing a work type

| Scheduler behavior to test | Work type |
|---|---|
| Basic load balancing / fairness | `SpinWait` (default) |
| Wake placement / sleep-wake cycles | `YieldHeavy`, `FutexPingPong` |
| CPU borrowing / idle balance | `Bursty`, `IdleChurn` |
| Cross-CPU wake latency | `PipeIo`, `CachePipe`, `WakeChain` |
| Cache-aware scheduling | `CachePressure`, `CacheYield` |
| Fan-out wake storms | `FutexFanOut`, `FanOutCompute` |
| Broadcast wakeups (thundering herd) | `ThunderingHerd` |
| epoll exclusive-wake paths | `EpollStorm` |
| Timer (hrtimer) wake-to-run latency | `TimerLatency` |
| IRQ/softirq wake paths | `IrqWake`, `NetTraffic` |
| Wakeup + request latency (schbench parity) | `Schbench` |
| KV object-cache request mix (taobench parity) | `Taobench` |
| Task creation/destruction pressure | `ForkExit` |
| Priority reweighting / nice dynamics | `NiceSweep` |
| Affinity churn / forced migration | `AffinityChurn`, `CrossAffinityChurn`, `NumaMigrationChurn` |
| Scheduling-class transitions | `PolicyChurn` |
| Cgroup migration paths | `CgroupChurn`, `CgroupAttachStorm` |
| Page fault / TLB pressure | `PageFaultChurn` |
| NUMA locality under migration | `NumaWorkingSetSweep` |
| Lock contention / convoy effect | `MutexContention` |
| Priority inversion | `PriorityInversion` |
| RT monopolizing or preempting CFS | `RtStarvation`, `PreemptStorm` |
| Signal delivery pressure | `SignalStorm` |
| Producer/consumer imbalance | `ProducerConsumerImbalance` |
| Block-I/O D-state cycles | `IoSyncWrite`, `IoRandRead`, `IoConvoy` |
| Mixed / phased real-world patterns | `Mixed`, `Sequence` |
| High-IPC compute, SMT interference | `AluHot`, `SmtSiblingSpin`, `IpcVariance` |
| Arbitrary user-defined workload | `Custom` |

## Variants by intent

The `WorkType` enum in `ktstr::workload` is the source of truth —
run `cargo doc` for full per-variant semantics, parameters, and
kernel-path citations. The shape of each family:

**CPU primitives.** `SpinWait` — tight spin loop, pure CPU. `YieldHeavy`
— `sched_yield` every iteration, exercising wake/sleep paths. `Mixed`
— spin burst then yield. `AluHot { width }` — parallel multiply
chains at high IPC, optionally SIMD. `SmtSiblingSpin` — paired
PAUSE-spin on two SMT siblings. `IpcVariance { hot_iters, cold_iters,
period_iters }` — alternating high-IPC and cache-miss phases.

**Block I/O** (against `/dev/vda`; per-worker tempfile fallback when
absent). `IoSyncWrite` — striped `O_SYNC` pwrites + `fdatasync`,
fsync-heavy D-state cycles. `IoRandRead` — 4 KB `O_DIRECT` preads at
random offsets, high-IOPS short D-states. `IoConvoy` — interleaved
sequential writes and random reads with periodic `fdatasync`.

**Burst-and-sleep.** `Bursty { burst_duration, sleep_duration }` —
CPU burst then sleep, freeing CPUs for borrowing. `IdleChurn` — burst
then `nanosleep`, exercising hrtimer + idle-class paths.

**Cache pressure.** `CachePressure { size_kib, stride }` — strided
read-modify-write sized to pressure L1. `CacheYield` — the same plus
`sched_yield`, testing re-placement with a cache-hot working set.

**Wake placement and cross-CPU paths.** `PipeIo` — CPU burst then
1-byte pipe exchange with a partner. `FutexPingPong { spin_iters }` —
paired futex wait/wake (non-WF_SYNC path). `CachePipe` — cache-hot
working set + pipe wake. `FutexFanOut { fan_out, spin_iters }` — one
messenger wakes N receivers, which measure wake-to-run latency.
`FanOutCompute` — fan-out plus matrix-multiply think time per
receiver. `WakeChain { depth, wake, work_per_hop }` — a ring of
waker-wakee hops via pipe (WF_SYNC) or futex. `AsymmetricWaker` —
paired workers in mismatched scheduling classes sharing a futex.
`EpollStorm { producers, consumers, events_per_burst }` — eventfd
producers + `epoll_wait` consumers (exclusive autoremove wake).
`ThunderingHerd { waiters, batches, inter_batch_ms }` — N waiters on
one futex word, broadcast-woken.

**Timer and IRQ wakes** (the AF_PACKET variants need
`#[ktstr_test(network = ...)]`). `TimerLatency { interval_us }` —
cyclictest-style absolute-deadline hrtimer wake. `NetTraffic` —
AF_PACKET self-traffic driving virtio-net RX hardirq + NAPI softirq.
`IrqWake` — paired sender/receiver; the receiver blocked in
`recvfrom` is woken from NET_RX softirq context.

**Lifecycle and class churn.** `ForkExit` — rapid fork + `_exit` +
`waitpid` cycles. `NiceSweep` — nice level cycled -20..19
(`reweight_task`; negative values skipped without `CAP_SYS_NICE`).
`AffinityChurn { spin_iters }` — self-directed `sched_setaffinity`
to random CPUs. `CrossAffinityChurn` — workers rewrite their cgroup
siblings' affinity; needs a dedicated cgroup. `PolicyChurn` —
`SCHED_OTHER` → `BATCH` → `IDLE` (→ `FIFO`/`RR` with `CAP_SYS_NICE`)
via `__sched_setscheduler`. `NumaMigrationChurn { period_ms }` —
affinity rotated across NUMA nodes. `CgroupChurn { groups, cycle_ms }`
— membership cycled between sibling cgroups. `CgroupAttachStorm`
— transient children migrated into a sibling cgroup mid-exit
(attach-path leader race).

**Memory pressure / NUMA.** `PageFaultChurn { region_kib,
touches_per_cycle, spin_iters }` — mmap, fault random 4 KiB pages
through `do_anonymous_page`, `MADV_DONTNEED`, repeat.
`NumaWorkingSetSweep` — the working set rotated across NUMA nodes
via `mbind`.

**Lock contention.** `MutexContention { contenders, hold_iters,
work_iters }` — N-way futex mutex contention (convoy effect,
lock-holder preemption). `PriorityInversion` — three priority tiers
contending for one lock, PI or plain futex mode.

**Signal / preemption pressure.** `SignalStorm` — paired workers
fire `tkill(partner, SIGUSR2)` between bursts. `PreemptStorm` — one
`SCHED_FIFO` worker preempts CFS spinners at ~kHz. `RtStarvation` —
`SCHED_FIFO` workers monopolize CPUs while CFS workers get no runtime.

**Compound.** `Sequence { first, rest }` — ordered `WorkPhase`s
(`Spin` / `Sleep` / `Yield` / `Io` / `AluHot`, each with a
`Duration`) looped for the run:

```rust,ignore
WorkType::Sequence {
    first: WorkPhase::Spin(Duration::from_millis(100)),
    rest: vec![
        WorkPhase::Sleep(Duration::from_millis(50)),
        WorkPhase::Yield(Duration::from_millis(20)),
    ],
}
```

**User-supplied.** `Custom` — your own work function, with a
fork-safe config payload. The how-to lives in
[Custom Scenarios](../writing-tests/custom-scenarios.md).

> `use ktstr::prelude::*;` brings in `WorkType`, `WorkSpec`,
> `WorkPhase`, `SchedPolicy`, and the parameter enums
> (`SchbenchConfig`, `TaobenchConfig`, `FutexLockMode`,
> `WakeMechanism`, `SchedClass`, `ReapMode`, `AluWidth`). Note the
> prelude also exports an unrelated `Phase` from the assertion layer
> — `WorkType::Sequence` uses `WorkPhase`.

## Constructors and defaults

Every parameterized variant has a snake-case constructor —
`WorkType::bursty(burst, sleep)`, `WorkType::mutex_contention(4, 256,
1024)`, `WorkType::wake_chain(depth, wake, work_per_hop)`, and so on
— with parameter validation where zero values are meaningless.
Duration-typed parameters (`Bursty`, `IdleChurn`, `WakeChain`) take
`std::time::Duration`, not raw integers.

`WorkType::from_name("FutexPingPong")` resolves a PascalCase name to
a default-parameterized instance; the per-variant defaults are the
constants in the
[`ktstr::workload::defaults`](https://ktstr.dev/rustdoc/ktstr/workload/defaults/index.html)
module. `Sequence` and `Custom` require explicit construction and
return `None` from name lookup. `WorkType::ALL_NAMES` lists every
name; `WorkType::name()` maps back.

## Grouped work types

`PipeIo`, `FutexPingPong`, and `CachePipe` pair workers and require
even `num_workers`. `FutexFanOut` and `FanOutCompute` require
`num_workers` divisible by `fan_out + 1` (one messenger + N
receivers per group). `MutexContention` requires divisibility by
`contenders`. `WorkType::worker_group_size()` returns the group size
for these variants, `None` for ungrouped types.

## Schbench

`Schbench` re-expresses schbench natively, in-process: message
threads batch-wake worker threads (wakeup latency), each worker
think-sleeps then does matrix work under a per-CPU lock (request
latency). `SchbenchConfig`'s fields map schbench's
`-m`/`-t`/`-F`/`-n`/`-s`/`-L`/`-R`/`-A`/`--split`/`-p` flags — the
rustdoc has the full CLI-parity table, including which knobs ktstr's
topology sets for you. Use a single ktstr worker (`workers(1)`): the
message/worker parallelism is this variant's internal thread
topology, not ktstr worker processes.

Beyond the default matrix mode: `requests_per_sec` (`-R`) injects a
fixed request rate, `auto_rps` (`-A`) searches for the saturation
rate, `split_percent` (`--split`) shifts matrix work between shared
and private cache footprints, and `pipe_transfer_bytes` (`-p`)
replaces matrix work with pipe transfers and reports per-worker
throughput. Each phase carries `wakeup_*_latency_us` and
`request_*_latency_us` percentiles, `schbench_loop_count`, and
message/worker run-delay — assertable per phase (see
[schbench_pipe_e2e.rs](https://github.com/likewhatevs/ktstr/blob/main/tests/schbench_pipe_e2e.rs)
and
[schbench_split_e2e.rs](https://github.com/likewhatevs/ktstr/blob/main/tests/schbench_split_e2e.rs)).

The port's fidelity is measured, not assumed:
[validation.md](https://github.com/likewhatevs/ktstr/blob/main/src/workload/schbench/validation.md)
compares it per-flag against gcc and clang builds of upstream
`schbench.c` — its percentiles and throughput land inside the
gcc↔clang envelope on nearly every axis, i.e. it behaves like a third
compilation of the same benchmark. Check it yourself on the host, no
VM:

<!-- captured: ktstr-schbench-validate | ktstr 0.23.0 | host, no VM -->
```text
average rps: 407.88
sched delay: message 37 (usec) worker 317 (usec)
```

## Taobench

`Taobench` is a native port of Meta's TaoBench cache-service
benchmark: a bounded, sharded, FIFO-evicting KV cache with a fast
in-cache hit path and a slow dispatcher miss path, settling to a
target hit ratio by the eviction↔refill equilibrium. `TaobenchConfig`
runs closed-loop by default; `arrival_rate` switches to open-loop
fixed-rate arrival with coordinated-omission serve latency, and
`slow_path_p99_us` adds a heavy-tailed Pareto service time to the
miss path. Same `workers(1)` rule as schbench.

Per phase and whole-run it emits `taobench_total_qps`,
`taobench_{fast,slow}_qps`, `taobench_hit_ratio`, and — open-loop
only — `taobench_serve_{p50,p90,p99,p999}_us` measured from intended
arrival, so queueing delay is charged to the scheduler rather than
hidden (see
[taobench_e2e.rs](https://github.com/likewhatevs/ktstr/blob/main/tests/taobench_e2e.rs)).
Its
[validation.md](https://github.com/likewhatevs/ktstr/blob/main/src/workload/taobench/validation.md)
documents the structural match against the reference (same hit-ratio
equilibrium, ~9:1 fast:slow, same qps/hit-ratio formulas) and the
deliberate divergences (no sockets/TLS; open-loop + Pareto tail are
additions). The host-side check:

<!-- captured: ktstr-taobench-validate (closed-loop, then -R 200000 open-loop) | ktstr 0.23.0 | host, no VM -->
```text
total_qps = 131870.7
hit_ratio = 0.8999  (fast / (fast + slow))
serve_us (coordinated-omission, from intended arrival): min=4 p50=3526656 p99=6545408
```

## Worker teardown and process groups

Every worker calls `setpgid(0, 0)` after fork, and teardown SIGKILLs
the worker's whole process group — on graceful stop, on escalation,
and again at handle drop. Anything a worker spawns that inherits its
pgid (a helper binary, a subshell) dies with it. A child that must
outlive the worker needs its own process group
(`setpgid(child_pid, 0)`) or an explicit wait before the worker
returns.

## Clone mode and pcomm

Workers fork by default (`CloneMode::Fork`: one process per worker);
`CloneMode::Thread` runs them as threads sharing the parent's thread
group. Setting `pcomm` on a `WorkSpec` or `CgroupDef` routes workers
through a fork-then-thread path: one forked leader whose `comm` is
the `pcomm` value hosts the matching workers as its threads — the
per-process-leader shape schedulers expect from real applications.
See the `WorkloadConfig` and `WorkSpec` rustdoc for the mechanics.

## WorkloadConfig

`WorkloadConfig` is the low-level spawn spec `CgroupDef` builds
internally; use it directly only when calling `WorkloadHandle::spawn`
from a custom scenario. Its default is one `SpinWait` worker with
inherited affinity and policy. The `composed` field carries secondary
`WorkSpec` groups spawned alongside the primary; reports identify
them by `group_idx`. Topology-aware `AffinityIntent` variants
(`SingleCpu`, `LlcAligned`, `CrossCgroup`, `SmtSiblingPair`) need
scenario context and are rejected at the direct-spawn gate. See
[Workers and Workloads](../architecture/workers.md).

## Scheduling policies

```rust,ignore
pub enum SchedPolicy {
    Normal,
    Batch,
    Idle,
    Fifo(u32),       // priority 1-99
    RoundRobin(u32), // priority 1-99
    Deadline { runtime: Duration, deadline: Duration, period: Duration },
    Ext,             // SCHED_EXT — route through the loaded BPF scheduler
}
```

`Fifo`, `RoundRobin`, and `Deadline` require `CAP_SYS_NICE`. A
malformed `Deadline` (`runtime <= deadline <= period` violated) fails
with a diagnostic before the syscall. `Ext` is `SCHED_EXT`: it routes
the worker through the loaded sched_ext scheduler even under a
`SCX_OPS_SWITCH_PARTIAL` scheduler that leaves other tasks in fair,
and requires `CONFIG_SCHED_CLASS_EXT` in the guest kernel.
