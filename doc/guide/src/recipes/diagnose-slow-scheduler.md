# Diagnose a Slow Scheduler with ctprof

When a scheduler change makes the workload slower but the test
suite still passes, the regression is usually buried in per-thread
off-CPU time. `ktstr ctprof capture` snapshots every live thread's
scheduling, memory, I/O, and taskstats delay counters;
`ktstr ctprof compare` diffs two snapshots and surfaces the buckets
where time went. This recipe walks through a typical A/B
comparison.

See the [ctprof reference](../reference/ctprof.md) for the full
metric registry, aggregation rules, derived-metric formulas, and
taskstats kconfig gating.

## Capture before and after

Before capturing on a fresh boot, run the
[taskstats pre-flight](#confirm-taskstats-data-is-actually-populated)
below — it saves a capture round-trip.

```sh
# Baseline: scheduler A loaded, workload running.
ktstr ctprof capture --output baseline.ctprof.zst

# Switch schedulers, restart workload, wait for steady state.
# ...

# Candidate: scheduler B, same workload.
ktstr ctprof capture --output candidate.ctprof.zst
```

`capture` walks `/proc` once and writes the snapshot. The
scheduling, I/O, and taskstats delay data are read from procfs and
netlink (genetlink) with no kernel tracing. The jemalloc memory
counters require a `ptrace(PTRACE_SEIZE)` attach that briefly stops
each probed thread — an observer effect bounded per thread; every
counter is cumulative-from-birth, so the recorded values are
unbiased by attach timing. The default capture covers every live
tgid; on a busy host this is hundreds of threads. The snapshot is
zstd-compressed JSON, typically a few MB, and capturing is fast: in
the session below (≈800 processes, ≈1,200 threads, no
jemalloc-probed targets), each snapshot completed in under a
second.

## Reading the compare table

Each compared metric renders as a `baseline → candidate` arrow
cell plus `delta`, `%`, and `%uptime` (what fraction of the
snapshot interval the group was alive) — the default `arrow`
layout; `--display-format full` splits baseline and candidate into
separate columns. Rows group by process family and sort so the
biggest movers land on top. This excerpt brackets a compile job, and
the table points straight at it — the `mm_percpu_wq` kworkers did
~10× more waiting:

<!-- captured: ktstr ctprof compare /tmp/ktstr-docs-base.ctprof.zst /tmp/ktstr-docs-cand.ctprof.zst --limit 24 | ktstr 0.23.0 | host capture, no VM -->
<div class="kt-term"><div class="kt-term-bar"><span class="kt-term-title">ktstr ctprof compare baseline.ctprof.zst candidate.ctprof.zst --limit 24</span></div>

<pre>## Primary metrics
<span class="t-dim"> comm                              threads  metric             value                delta      %         %uptime </span>
 kworker/{N}:{N}-mm_percpu_wq
     kworker/{N}:{N}-mm_percpu_wq  11→37    voluntary_csw      8.697K → 101.154K    +92.457K   +1063.1%  93%
     kworker/{N}:{N}-mm_percpu_wq  11→37    timeslices         8.699K → 101.166K    +92.467K   +1063.0%  93%
     <span class="t-yel">kworker/{N}:{N}-mm_percpu_wq  11→37    wait_time_ns       2.684s → 27.653s     +24.969s   +930.2%   93%</span>
     kworker/{N}:{N}-mm_percpu_wq  11→37    stime_clock_ticks  22ticks → 217ticks   +195ticks  +886.4%   93%
     kworker/{N}:{N}-mm_percpu_wq  11→37    run_time_ns        243.378ms → 2.320s   +2.077s    +853.4%   93%
...
 kworker/{N}:{N}-events
     kworker/{N}:{N}-events        87→60    nonvoluntary_csw   22 → 11              -11        -50.0%    95%
     kworker/{N}:{N}-events        87→60    timeslices         222.140K → 127.813K  -94.327K   -42.5%    95%
     kworker/{N}:{N}-events        87→60    voluntary_csw      222.118K → 127.802K  -94.316K   -42.5%    95%
     kworker/{N}:{N}-events        87→60    wait_time_ns       64.861s → 39.243s    -25.618s   -39.5%    95%
...</pre></div>

Large positive deltas on a process that should not have moved are
the suspects — here `wait_time_ns` grew 930% (runqueue wait, not
work) while `run_time_ns` grew less, the signature of queueing
pressure. The taskstats-delay lens below renders rows in exactly
the same shape.

## Compare with the taskstats lens

The `taskstats-delay` section bundles the eight kernel
delay-accounting buckets (CPU, blkio, swapin, freepages, thrashing,
compact, wpcopy, irq) plus their nine derived metrics
(`avg_*_delay_ns` per bucket, and the `total_offcpu_delay_ns`
rollup). Filter the output down to just the off-CPU view:

```sh
ktstr ctprof compare baseline.ctprof.zst candidate.ctprof.zst \
    --sections taskstats-delay \
    --sort-by total_offcpu_delay_ns:desc
```

The sort puts the processes with the largest absolute off-CPU
growth at the top. The `total_offcpu_delay_ns` derivation is:

```text
cpu + blkio + freepages + compact + wpcopy + irq + max(swapin, thrashing)
```

`max(swapin, thrashing)` rather than `swapin + thrashing` because
every thrashing event is also a swapin event from the syscall
perspective; summing both would double-count.

## Drill into the per-bucket averages

If `total_offcpu_delay_ns` jumped on a process, the per-bucket
`avg_*_delay_ns` derivations identify *which* off-CPU phase grew
(the `--sections taskstats-delay` filter keeps the raw counters and
all nine derivations together):

| Bucket | Average derivation | Meaning |
|---|---|---|
| CPU runqueue wait | `avg_cpu_delay_ns` | Time waiting for the scheduler to pick the task |
| Block I/O wait | `avg_blkio_delay_ns` | Synchronous block-device wait; the canonical delay-accounting reading, distinct from schedstat `iowait_sum` |
| Swap-in / Thrashing | `avg_swapin_delay_ns` / `avg_thrashing_delay_ns` | Memory pressure; the two overlap (a thrashing event is also a swapin) |
| Direct memory reclaim | `avg_freepages_delay_ns` | Allocator hit the `__alloc_pages` slowpath |
| Memory compaction | `avg_compact_delay_ns` | High-order allocation stalled on compaction |
| CoW page-fault | `avg_wpcopy_delay_ns` | Write-protect-copy fault, e.g. fork-then-write |
| IRQ handling | `avg_irq_delay_ns` | Time charged to the task by IRQ accounting |

One caveat on `avg_cpu_delay_ns`: the kernel updates its count and
total locklessly, so a reader can catch one ahead of the other —
the quotient is approximate at the sub-event scale and stable at
the integrated scale.

A growing `avg_cpu_delay_ns` with flat blkio/swap/freepages
suggests the new scheduler is making poor placement choices — the
task is queueing more often or for longer, but no other subsystem
is to blame. A growing `avg_blkio_delay_ns` with flat
`avg_cpu_delay_ns` points away from the scheduler entirely (disk,
network filesystem, or a userspace lock pattern).

## Cross-reference the primary table

Once a bucket is identified, look at the underlying counters
without the section filter:

```sh
ktstr ctprof compare baseline.ctprof.zst candidate.ctprof.zst \
    --metrics nr_wakeups,nr_migrations,wait_sum,wait_count,run_time_ns,timeslices
```

Useful pairings when the suspect bucket is CPU runqueue wait:

- `wait_sum / wait_count` — schedstat's average wait per scheduling
  event (the `avg_wait_ns` derivation). If this confirms
  `avg_cpu_delay_ns`, both delay-accounting paths agree.
- `nr_migrations` — the new scheduler may be moving the task more
  aggressively; cross-CPU migrations cost wall-clock time even when
  `run_time_ns` is identical.
- `nr_wakeups_affine / nr_wakeups_affine_attempts` — the
  `affine_success_ratio` derivation (CFS-only). A large drop with
  growing `avg_cpu_delay_ns` is a strong signal for cache-unfriendly
  placement.

## Confirm taskstats data is actually populated

**Pre-flight (save a capture round-trip):** verify the host has the
delayacct runtime toggle enabled before capturing on a fresh boot:

```sh
sysctl kernel.task_delayacct       # must report `kernel.task_delayacct = 1`
zcat /proc/config.gz | grep -E 'CONFIG_TASKSTATS|CONFIG_TASK_DELAY_ACCT'
                                   # both must read `=y` for delayacct to fire
```

If `task_delayacct` is `0`, set it with
`sysctl -w kernel.task_delayacct=1` (or persist via
`/etc/sysctl.d/`) before capture. If the kconfig lines are absent,
the running kernel was built without delayacct support and
re-capturing won't help. `capture` also logs read-failure tallies
with a hint when a whole class of files is unreadable — a real
example from a kernel missing two kconfigs:

<!-- captured: ktstr ctprof capture -o /tmp/ktstr-docs-cand.ctprof.zst (stderr log line, ANSI color stripped) | ktstr 0.23.0 | host capture, no VM -->
```text
2026-07-04T22:00:48.765819Z  INFO ktstr::ctprof: ctprof parse: 1215 tids walked, 1646 read failures (dominant: io); hint: schedstat / io read failures dominate — kernel may be built without CONFIG_SCHED_INFO and/or CONFIG_TASK_IO_ACCOUNTING
```

If every taskstats column reads zero *after* the pre-flight, the
snapshot likely hit a gating problem rather than a real "no delay"
reading. The snapshot records a structured per-capture tally; no
subcommand prints it, but the snapshot is zstd-compressed JSON, so
read it directly:

```sh
zstd -dc candidate.ctprof.zst | \
    jq '{taskstats: .taskstats_summary, tids_walked: .parse_summary.tids_walked}'
```

- `eperm_count > 0` — the capturing process lacked `CAP_NET_ADMIN`.
  Re-run as root, or grant `cap_net_admin+eip` via `setcap`.
- `esrch_count` near `tids_walked` — every tid raced exit before
  the per-tid query landed. Lengthen the workload's steady-state
  window and re-capture.
- `ok_count == 0` and `eperm_count == 0` — the netlink open failed,
  almost always meaning the kernel was built without
  `CONFIG_TASKSTATS`. Rebuild with the kconfig.
- `ok_count > 0` but every delay column reads zero — kernel built
  with the kconfigs but launched without the runtime toggle. Add
  `delayacct` to the kernel cmdline, or set
  `sysctl kernel.task_delayacct=1` and re-capture.

## What next

- If the candidate scheduler is a *branch* of the baseline, gate
  the regression with
  [`cargo ktstr perf-delta`](ab-compare.md) so CI catches the next
  one.
- If the question is "is this scheduler worth its overhead at
  all", run the
  [scheduler-vs-EEVDF comparison](scheduler-vs-eevdf.md) — one
  test, both schedulers, per-phase deltas.
- [ctprof reference](../reference/ctprof.md) — full metric registry
  and gating documentation.
