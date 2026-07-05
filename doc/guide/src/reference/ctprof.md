# ctprof

The ctprof profiler captures a host-wide per-thread snapshot
of scheduling counters, memory / I/O accounting, CPU affinity,
cgroup state, and thread identity, then compares two snapshots to
surface what changed. It is a manually-invoked CLI companion to
the automated scheduler tests — useful when a run passes on one
machine and fails on another, or for A/B comparing host behavior
across kernel / sysctl / workload changes.

This is a **different tool** from `cargo ktstr show-host`,
which captures the host *context* (kernel, CPU model, sched_\*
tunables, NUMA layout, kernel cmdline) — aggregate state that
does not change between scenarios. The profiler captures
*per-thread* cumulative counters that do change, and its
comparison surface is designed for the thread-level diff.

## When to use it

- **Workload investigation** — you observe a regression and want
  to know which process / thread pool moved in run time,
  context-switch rate, or migration count.
- **Kernel / sysctl A/B** — capture before and after flipping a
  sched_\* tunable on an otherwise-identical workload; the
  compare output surfaces every counter that responded.
- **Host baselining** — capture on a known-good host, capture on
  a failing host, compare to isolate what differs at the
  thread-behavior level.

The profiler is **not** invoked automatically by scenarios or the
gauntlet. It is opt-in and operator-driven via the
`ktstr ctprof` subcommand.

## Capture, then compare

The whole workflow is three commands: snapshot, change something,
snapshot again, diff.

```sh
ktstr ctprof capture --output base.ctprof.zst
# ... run a workload, flip a tunable, swap a scheduler ...
ktstr ctprof capture --output cand.ctprof.zst
ktstr ctprof compare base.ctprof.zst cand.ctprof.zst
```

`capture` walks `/proc` for every live thread group, enumerates
each thread, and reads a handful of procfs sources for each one.
The output is a zstd-compressed JSON snapshot (conventional
extension: `.ctprof.zst`). On a workstation with ~1,200 live
threads, each snapshot in the run below took about a second.

Here is a real compare — two captures taken a couple of seconds
apart on a busy workstation. Rows sort by largest absolute
percent delta, so the biggest movers are the first thing you see:

<!-- captured: ktstr ctprof capture --output /tmp/ktstr-docs-{base,cand}.ctprof.zst (×2, ~2s apart); ktstr ctprof compare /tmp/ktstr-docs-base.ctprof.zst /tmp/ktstr-docs-cand.ctprof.zst --sections primary --limit 20 | ktstr 0.23.0 | host kernel 7.0.14 (no VM) -->
<div class="kt-term"><div class="kt-term-bar"><span class="kt-term-title">ktstr ctprof compare base.ctprof.zst cand.ctprof.zst --sections primary --limit 20</span></div>

<pre>## Primary metrics
 comm                              threads  metric             value                delta      %         %uptime
 kworker/{N}:{N}-mm_percpu_wq
<span class="t-red">     kworker/{N}:{N}-mm_percpu_wq  11→37    voluntary_csw      8.697K → 101.154K    +92.457K   +1063.1%  93%</span>
     kworker/{N}:{N}-mm_percpu_wq  11→37    timeslices         8.699K → 101.166K    +92.467K   +1063.0%  93%
     kworker/{N}:{N}-mm_percpu_wq  11→37    wait_time_ns       2.684s → 27.653s     +24.969s   +930.2%   93%
     kworker/{N}:{N}-mm_percpu_wq  11→37    stime_clock_ticks  22ticks → 217ticks   +195ticks  +886.4%   93%
     kworker/{N}:{N}-mm_percpu_wq  11→37    run_time_ns        243.378ms → 2.320s   +2.077s    +853.4%   93%
     kworker/{N}:{N}-mm_percpu_wq  11→37    nonvoluntary_csw   2 → 12               +10        +500.0%   93%
<span class="t-b">     kworker/{N}:{N}-mm_percpu_wq  11→37    thread_count       11 → 37              +26        +236.4%   93%</span>
     kworker/{N}:{N}-mm_percpu_wq  11→37    nr_migrations      11 → 34              +23        +209.1%   93%
 kworker/{N}:{N}-events
     kworker/{N}:{N}-events        87→60    nonvoluntary_csw   22 → 11              -11        -50.0%    95%
<span class="t-grn">     kworker/{N}:{N}-events        87→60    timeslices         222.140K → 127.813K  -94.327K   -42.5%    95%</span>
 user.slice
   user-{N}.slice
     session-{H}.scope
       ktstr
           ktstr                   1        processor          9 → 43               +34        +377.8%   0%
           ktstr                   1        wait_time_ns       6.850µs → 22.693µs   +15.843µs  +231.3%   0%
           ktstr                   1        nonvoluntary_csw   2 → 6                +4         +200.0%   0%
<span class="t-dim">... 22 more lines truncated (use --limit 0 for unlimited)</span></pre></div>

Reading it:

- **value** is `baseline → candidate` for the group's aggregated
  reading; **delta** and **%** carry the signed move. The
  `mm_percpu_wq` pool grew from 11 to 37 threads and its
  voluntary context switches went up 11×, all inside the capture
  window — the eye lands there first because the sort put it
  first.
- **threads** shows the group population on each side
  (`11→37`). A population change is often the story by itself.
- **%uptime** is the group's average thread lifetime relative to
  the longest-lived group in the snapshot — low values flag
  young threads whose counters had little time to accumulate.
- `{N}` / `{H}` placeholders come from name-pattern
  normalization: `kworker/3:1` and `kworker/7:0` are the same
  logical pool, so they land in one `kworker/{N}:{N}` bucket.

Groups present on only one side surface as **unmatched** — a row is
missing because the process did not exist, not because it did zero
work. A full (unfiltered) compare lists them in a trailer:

<!-- captured: ktstr ctprof compare /tmp/ktstr-docs-base.ctprof.zst /tmp/ktstr-docs-cand.ctprof.zst (trailer) | ktstr 0.23.0 | host kernel 7.0.14 (no VM) -->
```text
1 group(s) only in baseline (/tmp/ktstr-docs-base.ctprof.zst):
  kworker/u{N}:{N}-flush-btrfs-{N} kworker/u{N}:{N}-flush-btrfs-{N}

2 group(s) only in candidate (/tmp/ktstr-docs-cand.ctprof.zst):
  kworker/u{N}:{N}-writeback kworker/u{N}:{N}-writeback
  kworker/{N}:{N}-events_freezable kworker/{N}:{N}-events_freezable
```

### What is captured per thread

- **Identity** — tid, tgid, process and thread name, cgroup v2
  path, start time, scheduling policy, nice, CPU affinity mask.
- **Scheduling counters** (cumulative, from `/proc/<tid>/sched`
  and `/proc/<tid>/schedstat`) — run / wait / sleep / block /
  iowait time, context switches, wakeups with locality splits,
  migrations, plus lifetime peaks (`wait_max`, `slice_max`, …).
- **Memory** — page faults; jemalloc per-thread
  allocated/deallocated counters read via ptrace +
  `process_vm_readv` (jemalloc-linked processes only — other
  allocators read zero rather than failing capture); per-process
  `smaps_rollup`.
- **I/O** — `rchar` / `wchar`, syscall counts, and block-level
  byte counters from `/proc/<tid>/io` (requires
  `CONFIG_TASK_IO_ACCOUNTING`).
- **Taskstats delay accounting + watermarks** — eight delay
  categories plus peak-RSS/VM watermarks via the TASKSTATS
  genetlink family; see
  [Taskstats delay accounting](#taskstats-delay-accounting) for
  gating and semantics.
- **PSI and cgroup aggregates** — host-level and per-cgroup
  pressure (`CONFIG_PSI`), `cpu.stat` / `memory.*` / `pids.*` per
  cgroup that hosted a sampled thread — read from cgroup files
  directly, not derived from per-thread data.
- **sched_ext sysfs** — `state`, `switch_all`, `nr_rejected`, and
  the hotplug/enable sequence counters, when
  `CONFIG_SCHED_CLASS_EXT` is built.

Three timing families matter when interpreting a diff:

- **Cumulative counters** (the majority) only increase, so probe
  attachment time does not bias the reading — a diff between two
  captures measures exactly the activity in the window.
- **Lifetime extrema** (`*_max`, `hiwater_*`, `*_delay_min_ns`)
  are per-event peaks kept by the kernel, not sums over the
  window.
- **Instantaneous gauges** (`nr_threads`, `fair_slice_ns`,
  `state`, affinity, `processor`) are sampled at capture time
  and can legitimately differ between two probes of the same
  thread.

Metrics that reset on attachment (perf_event_open counters, BPF
tracing samples) are intentionally absent — they require
long-lived instrumentation the capture layer cannot install
without disturbing the system it is measuring.

### Capture is best-effort

Each internal reader returns `Option`; a kernel missing a config
gate (no `CONFIG_SCHED_DEBUG`, no `CONFIG_SCHEDSTATS`) yields
`None` from that reader without failing the rest of the thread.
Counters collapse to `0`, identity strings collapse to empty.
**A missing reading is indistinguishable from a genuine zero in
the output** — the contract is "never fail the snapshot." The
capture summary lines on stderr tally read failures and hint at
the likely missing kconfig.

Pulling the jemalloc counters briefly stops each probed thread
via `ptrace(PTRACE_SEIZE)`, which needs root, `CAP_SYS_PTRACE`,
or `kernel.yama.ptrace_scope=0`; without the privilege those
fields fall through to zero and the rest of the snapshot still
populates.

### Cgroup namespace caveat

The per-thread `cgroup` path is read verbatim from
`/proc/<tid>/cgroup` — it is relative to the **cgroup namespace
root the capturing process sees**, not the system-global v2
mount root. A process inside a nested cgroup namespace sees a
truncated path. Cross-namespace comparison requires external
canonicalization; the capture layer deliberately does not
attempt it.

## Compare options

### Grouping

`compare` defaults to `--group-by all`: all three pattern-aware
axes (cgroup, pcomm, comm) contribute to one view — cgroup-grouped
rows render as an indented path tree, name-pattern buckets render
flat, as in the excerpt above — and renamed-but-identical cgroups
are joined for diffing (a `[fudged: <leaf>]` marker) instead of
surfacing as orphans.

- `--group-by pcomm` — aggregate every thread of the same
  process together (the `show` default).
- `--group-by cgroup` — aggregate by cgroup path; enables the
  per-cgroup sections. Use `--cgroup-flatten '<glob>'` to
  collapse dynamic segments (pod UUIDs, session scopes) so the
  same logical workload lands on the same row across runs.
- `--group-by comm` — aggregate by thread-name pattern across
  every process (`tokio-worker-{0..N}` → one bucket). Choose it
  when a thread-pool name spans many binaries.
- `--group-by comm-exact` — literal thread names, no pattern
  collapse, for when distinct token values carry meaning (each
  `kworker/u8:N` tracked independently).
- `--no-thread-normalize` disables the pattern collapse on the
  name axes.

Rule of thumb: start with the default `all` to find which axis
moved, then re-run with that single axis plus `--sections` /
`--metrics` filters for a narrow, pasteable table.

### Filtering: `--sections` vs `--metrics`

- **`--sections`** picks which sub-tables render: `primary`,
  `taskstats-delay`, `derived`, `cgroup-stats`, `cgroup-limits`,
  `memory-stat`, `memory-events`, `pressure`, `host-pressure`,
  `smaps-rollup`, `sched-ext`. The five cgroup sections require
  `--group-by cgroup`. The taskstats rows render inside the
  primary/derived tables but match the `taskstats-delay` section
  name, so you can scope to them alone or exclude them.
- **`--metrics`** picks which rows render inside the primary and
  derived tables, by metric name from the
  [`metric-list`](#metric-list) vocabulary. Secondary sub-tables
  have fixed shapes and ignore it.

They compose: `--sections primary --metrics run_time_ns` shows a
single row and nothing else. `--sort-by
'wait_sum:desc,run_time_ns:desc'` re-ranks rows by your own key
instead of the default biggest-|Δ%| ordering; `--limit N` caps
lines per section.

### How groups aggregate

Every metric declares how per-thread values reduce into a group
row; the registry binds each metric to exactly one reduction so
a nonsensical fold (summing peaks) cannot be expressed.

| Metric class | Group reduction | Why |
|---|---|---|
| Cumulative counters (csw, wakeups, migrations, run/wait time, io bytes, delay totals) | sum | totals compose; deltas stay meaningful |
| Lifetime peaks (`wait_max`, `*_delay_max_ns`, `hiwater_*`) | max | summing peaks conflates one 1 s spike with 1000 × 1 ms spikes |
| Instantaneous gauges (`nr_threads`, `fair_slice_ns`) | max | summing a sampled instant has no physical meaning |
| Bounded ordinals (`nice`, `priority`, `processor`) | `[min, max]` range | a shift on either end stays visible |
| Categorical (`policy`, `state`) | mode + count/total | no arithmetic on categories; delta is `same` / `differs` |
| CPU affinity | min/max CPU count + uniform flag | heterogeneous groups render `N-M cpus (mixed)` |

Three cross-cutting caveats, stated once:

#### Swapin / thrashing overlap {#swapin-thrashing-overlap}

Every thrashing event is also a swapin event at the syscall
layer. Never sum the two families; rollups OR them with `max()`.

#### The `*_delay_min_ns` sentinel {#min-ns-sentinel}

The kernel keeps the smallest *non-zero* observation, so `0`
means "no events observed", not "saw a zero-ns event".
Disambiguate against the matching `*_count`.

#### Shared-mm watermarks {#hiwater-shared-mm}

The kernel reads `hiwater_rss_bytes` / `hiwater_vm_bytes` from
the shared `mm_struct`, so sibling threads of one process all
report the same value; kernel threads read zero by design.

### Derived metrics

Derived metrics combine already-aggregated inputs into a scalar
with its own scale — they render in a separate `## Derived
metrics` table on both `compare` and `show`. A missing input or
zero denominator yields `-` (not computable), distinct from a
computed zero. Representative entries:

| Metric | Formula | Reading it |
|---|---|---|
| `cpu_efficiency` | `run / (run + wait)` | fraction of scheduler-tracked time on-CPU; lower = more runqueue waiting |
| `avg_slice_ns` | `run_time_ns / timeslices` | average on-CPU slice; catches timeslice-tuning regressions |
| `involuntary_csw_ratio` | `nonvol / (vol + nonvol)` | preemption pressure vs cooperative blocking |
| `avg_cpu_delay_ns` | `cpu_delay_total_ns / count` | runqueue wait per event, from the delayacct path |
| `live_heap_estimate` | `allocated - deallocated` | jemalloc-only live heap; zero is genuine for other allocators |
| `total_offcpu_delay_ns` | sum of delay buckets, swapin/thrashing OR'd | one off-CPU number; `-` when delayacct is off entirely |

The full registry (17 derived + every primary metric) is
enumerated by [`metric-list`](#metric-list); all names are valid
`--sort-by` and `--metrics` keys.

### Output and interpretation

The comparison prints raw numbers and percent delta. There are
no judgment labels (regression vs. improvement) — whether
"run_time went up 15%" is good depends on whether you measured a
CPU-bound workload (more work done) or a spin-wait pathology
(more time wasted). The interpretation is scheduler-specific and
left to the operator. Ratio-valued rows suppress the `%` column:
the absolute delta of a `[0, 1]` quantity already carries
percentage-point semantics.

## show

`show` renders a single snapshot as a per-group table — same
grouping, filtering, and sorting surface as `compare`, minus the
diff columns. Default grouping is `pcomm`.

<!-- captured: ktstr ctprof show /tmp/ktstr-docs-base.ctprof.zst --group-by pcomm --sections primary --metrics run_time_ns,voluntary_csw,nr_migrations --sort-by run_time_ns:desc --limit 8 | ktstr 0.23.0 | host kernel 7.0.14 (no VM) -->
```text
## Primary metrics
 pcomm                                    threads  metric         value
...
 kworker/u{N}:{N}-btrfs-endio-write       23       run_time_ns    41.186s
 kworker/u{N}:{N}-btrfs-endio-write       23       voluntary_csw  1.249M
 kworker/u{N}:{N}-btrfs-endio-write       23       nr_migrations  26.759K
 kworker/u{N}:{N}-btrfs-endio             47       run_time_ns    40.940s
... 464 more lines truncated (use --limit 0 for unlimited)
```

`--columns` controls the rendered column set; `show` accepts
`group`, `threads`, `metric`, `value`, `tags`, `uptime`, while
`compare` accepts `group`, `threads`, `metric`, `baseline`,
`candidate`, `delta`, `%`, `arrow`, `tags`, `uptime` — each side
rejects the other's diff/value columns.

## metric-list

`metric-list` prints every registered metric with its tags and a
one-line description — the authoritative vocabulary for
`--metrics` and `--sort-by`, plus a tag legend explaining which
kernels populate each counter:

<!-- captured: ktstr ctprof metric-list | ktstr 0.23.0 | host-side (no VM) -->
```text
## Metrics

 metric                        tags                           description
...
 run_time_ns                   [SCHED_INFO]                   Cumulative on-CPU time, ns; /proc/<tid>/schedstat field 1.
 wait_time_ns                  [SCHED_INFO]                   Cumulative time waiting on the runqueue, ns; schedstat field 2.
 timeslices                    [SCHED_INFO]                   Number of times the task was run on a CPU; schedstat field 3.
 voluntary_csw                                                Voluntary context switches (task gave up the CPU itself).
 nonvoluntary_csw                                             Involuntary context switches (task was preempted).
 nr_wakeups                    [SCHEDSTATS]                   Total wakeups via try_to_wake_up().
...
 nr_wakeups_affine             [cfs-only] [SCHEDSTATS]        Wakeups that succeeded under the wake_affine() heuristic.
```

The bracketed tags mark scheduler-class gates (`[cfs-only]`
counters stay zero under sched_ext) and kconfig gates
(`[SCHEDSTATS]`, `[TASK_DELAY_ACCT]`, …) so you know whether a
zero means "idle" or "not compiled in".

## Taskstats delay accounting

The kernel's TASKSTATS genetlink family delivers per-task
delay-accounting and memory-watermark fields that are not
exposed via `/proc/<tid>/sched` or `/proc/<tid>/stat`. The 34
captured fields (8 delay categories × 4 bucket fields + 2
watermarks) all tag the `taskstats-delay` section so they can be
filtered as a unit.

### Capability and kconfig gating

Querying the netlink family requires **`CAP_NET_ADMIN`** on the
capturing process. A non-root operator running
`ktstr ctprof capture` hits `EPERM` on the first query and every
taskstats field collapses to zero per the best-effort contract.

- **Delay-accounting fields** require `CONFIG_TASKSTATS=y` and
  `CONFIG_TASK_DELAY_ACCT=y` **and** the runtime `delayacct=on`
  toggle (boot param or `kernel.task_delayacct=1`). A kernel
  built with both configs but launched without the toggle
  produces all-zero delay readings. ktstr's standard kernel
  build includes both kconfigs, and the test harness adds
  `delayacct` to the guest cmdline.
- **Memory-watermark fields** (`hiwater_rss_bytes`,
  `hiwater_vm_bytes`) require `CONFIG_TASKSTATS=y` and
  `CONFIG_TASK_XACCT=y`, and do not respond to the runtime
  toggle. See the [shared-mm caveat](#hiwater-shared-mm).

The structured tally on `CtprofSnapshot::taskstats_summary`
(`ok_count` / `eperm_count` / `esrch_count` / `other_err_count`)
distinguishes "kernel doesn't expose this" (netlink open failed,
all counters zero) from "every tid raced exit" (high
`esrch_count`) from "`CAP_NET_ADMIN` missing" (high
`eperm_count`). There is no CLI lens for it yet — read it from
the snapshot JSON (`zstd -d < snap.ctprof.zst | jq
.taskstats_summary`).

### The eight delay categories

| Category | Kernel source | Notes |
|---|---|---|
| `cpu_delay_*` | `tsk->sched_info` via `delayacct_add_tsk` (`kernel/delayacct.c`) | Runqueue wait. Count and total update locklessly, so a reader can transiently observe one ahead of the other — averages are approximate at sub-event scale, stable integrated. Same bucket as schedstat `wait_*` via a different path. |
| `blkio_delay_*` | `delayacct_blkio_start/_end` | Synchronous block-I/O wait; updates serialize through the task's delay lock. The canonical delayacct block-I/O reading, distinct from schedstat `iowait_sum`. |
| `swapin_delay_*` | `delayacct_swapin_start/_end` | Swap-in wait. [Overlaps thrashing](#swapin-thrashing-overlap). |
| `freepages_delay_*` | called from `mm/vmscan.c` | Direct-reclaim wait. |
| `thrashing_delay_*` | called from `mm/filemap.c`, `mm/page_io.c` | Thrashing wait; refines swapin tracking. |
| `compact_delay_*` | called from `mm/page_alloc.c` | Memory-compaction wait. |
| `wpcopy_delay_*` | called from `mm/memory.c`, `mm/hugetlb.c` | Write-protect-copy (CoW) fault wait. Taskstats v13+. |
| `irq_delay_*` | `delayacct_irq` | IRQ-handler windows charged to the task. Taskstats v14+. On kernels predating a bucket, the missing fields read zero from the truncated payload. |

Each category carries four fields: `*_count` (windows observed),
`*_delay_total_ns` (cumulative), `*_delay_max_ns` (longest single
window), and `*_delay_min_ns` (shortest non-zero window — mind
the [sentinel](#min-ns-sentinel)).

## File format

`.ctprof.zst` is zstd-compressed JSON of `CtprofSnapshot`. The
schema is `#[non_exhaustive]` so field additions do not break
existing snapshots. The top level carries `threads`,
`cgroup_stats`, host-level `psi`, optional `sched_ext` sysfs
state, the capture summaries (`probe_summary`, `parse_summary`,
`taskstats_summary`), and an embedded `HostContext` — the same
structure `show-host` prints — for round-trip tooling. Thread
start times are recorded in USER_HZ (100 on x86_64 and aarch64),
so cross-host comparison between differently-configured kernels
on those architectures is meaningful.

## Extending ctprof

Adding a metric to the registry is a typed three-step change
(field newtype → capture wiring → registry entry) designed so a
mismatched aggregation fails to compile. See the module
documentation for `ktstr::ctprof_compare` in the
[rustdoc](https://ktstr.dev/rustdoc/ktstr/ctprof_compare/index.html).

## Related

- [Diagnose a Slow Scheduler with ctprof](../recipes/diagnose-slow-scheduler.md) —
  the worked investigation recipe built on this tool.
- [`cargo ktstr show-host`](../running-tests/cargo-ktstr.md) —
  host *context* capture (kernel, CPU, tunables), without the
  per-thread walk; [Capture and Compare Host State](../recipes/host-state.md)
  compares it across runs.
