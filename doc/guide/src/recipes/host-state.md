# Capture and Compare Host State

> **Disambiguation**: this recipe covers **host context**
> (kernel build, CPU model, sched_\* tunables, NUMA layout)
> via `cargo ktstr show-host`. For **per-thread profiling**
> (scheduling counters, memory / I/O accounting, taskstats
> delay accounting per thread), see the
> [ctprof reference](../reference/ctprof.md) and the
> [Diagnose a Slow Scheduler with ctprof](diagnose-slow-scheduler.md)
> recipe. For **scheduler-behavior diffs between branches**
> (per-metric scheduler measurements via `cargo ktstr stats
> compare` on two `#[ktstr_test]` runs against different worktrees),
> see [A/B Compare Branches](ab-compare.md).

When a gauntlet run passes on one machine and fails on another —
or passes on Monday and fails on Wednesday — the first thing to
check is whether the host itself changed. `cargo ktstr show-host`
captures a snapshot of the kernel, CPU, memory, scheduler
tunables, and kernel cmdline; `cargo ktstr stats compare`
surfaces the changes between two sidecars in a host-delta
section of its output so you can see what moved.

## Two `show-host` commands: live vs archived

Two distinct subcommands print host context, and they are NOT
interchangeable — pick the one whose target matches your question:

- **`cargo ktstr show-host`** captures the **live** host context
  by reading `/proc`, `/sys`, and `uname()` at invocation time.
  Use this when you want to inspect the current machine, e.g.
  before running a benchmark, after a sysctl change, or to
  confirm what `cargo ktstr stats compare` would record on the
  next run produced here. No prior runs needed.
- **`cargo ktstr stats show-host --run RUN_ID`** prints the
  **archived** host context captured at sidecar-write time for
  the named run. Use this when investigating a regression in a
  past run — what looked like a code change might trace back to
  a host change at the time the sidecar was produced. Resolves
  `--run` against `target/ktstr/` (or `--dir`) and renders the
  first sidecar in the run that carries a populated `host`
  field via the same `HostContext::format_human` formatter the
  live `show-host` uses, so the two outputs are byte-for-byte
  comparable when the host is unchanged.

The sections below cover the live `show-host`. For the archived
variant's flag table see
[`stats show-host`](../running-tests/cargo-ktstr.md#show-host).

## Capture: `show-host`

```sh
cargo ktstr show-host
```

Prints a `key: value` report covering:

- CPU model + vendor (first `/proc/cpuinfo` entry).
- Total memory, hugepages total / free, hugepage size (from
  `/proc/meminfo`).
- Transparent hugepage policy (`thp_enabled`, `thp_defrag`) with
  the bracketed selection preserved verbatim.
- Every `/proc/sys/kernel/sched_*` tunable, one entry per line.
- `online_cpus` — count from `HostTopology::online_cpus.len()`.
- `cpufreq_governor` — per-CPU governor from
  `/sys/devices/system/cpu/cpuN/cpufreq/scaling_governor`,
  one line per CPU.
- NUMA node count (from CPU→node mapping; memory-only nodes
  without CPUs are not counted).
- `kernel_name` / `kernel_release` / `arch` (from the `uname()`
  syscall).
- `/proc/cmdline` verbatim.
- `heap_state` — nested jemalloc allocator state, populated whenever
  jemalloc is installed as the process's `#[global_allocator]`. Every
  `cli-bins` binary (`ktstr`, `cargo-ktstr`) installs
  `tikv_jemallocator::Jemalloc`, so `cargo ktstr show-host` always
  populates it. It collapses to `None` only for downstream library
  consumers that link jemalloc but do not install it as the global
  allocator (allocated and active bytes both zero).

Absent fields render as `(unknown)` — an empty `sched_*` map
renders as `(empty)` and a missing map renders as `(unknown)`.
The distinction matters when you want to know whether a
dimension was inspected but absent, vs failed to populate.

Sidecars written before the `uname_sysname` / `uname_release` /
`uname_machine` → `kernel_name` / `kernel_release` / `arch`
rename render the renamed fields as `(unknown)` in `show-host`
and in `stats compare`'s host-delta section, and re-running the
test against the current binary regenerates the sidecar with
the new field names populated. Mechanically: the old sidecar
deserializes only because the rename was field-additive — the
new fields are `Option<T>` and default to `None` when the
old-name keys are absent. Cross-version-incompatible changes
(field deletion, type change) would fail deserialization
loudly. Per the pre-1.0 sidecar-disposable rule, re-run the
test to regenerate the sidecar with the current schema rather
than relying on the additive-rename path holding.

This output is human-oriented. For programmatic access, read
the `host` field of any sidecar JSON (same schema, identical
values — `show-host` prints the live snapshot the sidecar
writer would attach to a fresh test run).

## Compare: `stats compare`

```sh
cargo ktstr stats compare --a-project-commit <baseline> --b-project-commit <current>
```

Per-side filter flags (`--a-X` / `--b-X`) partition the
sidecar pool into the two sides of the contrast — slice on
`project-commit`, `kernel`, `scheduler`, `run-source`, etc.
depending on what you are diffing. `compare` picks the first
sidecar with `Some(host)` from each side, then prints one of
five host-section shapes depending on what survived capture:

- Neither side carried host context — the host section is
  omitted entirely (no banner, no rows).
- One side missing — `host: captured in 'A' only, delta
  unavailable` (or `'B' only`), so a one-sided pipeline failure
  is visible rather than silently dropped.
- Both sides present, every field agrees — `host: identical
  between 'A' and 'B'` (plus ` (arch: x86_64)` when both sides
  carry a matching `arch` field).
- Both sides present, fields differ — the `host delta ('A' →
  'B'):` banner followed by one indented `key: A → B` row per
  changed field:

```text
host delta ('A' → 'B'):
  kernel_release: 6.14.2 → 6.15.0
  thp_enabled: always [madvise] never → always madvise [never]
  sched_tunables.sched_rt_runtime_us: 950000 → 980000
```

Fields that match in both runs are suppressed by design — this
is a diff, not a snapshot. Missing-on-one-side rendering differs
by layer: top-level `Option<T>` host fields (e.g. `kernel_release`,
`thp_enabled`, the whole `sched_tunables` map) render with
`(unknown)` on the None side so a regression in the capture
pipeline surfaces instead of silently hiding. Per-key diffs
inside the `sched_tunables` map use `(absent)` instead, to
distinguish "the map was captured and this key is not in it"
from "the whole map was unknown at capture time".

### CI integration

Gauntlet runs emit the host block automatically in every
sidecar. To diff the host state across two CI runs, slice the
pool on whatever dimension separates them (typically
`--a-project-commit` / `--b-project-commit` or `--a-kernel` /
`--b-kernel`) — the host-delta section appears automatically
in the compare output when any host field differs between the
two sides. A CI job can:

1. Run the gauntlet on the candidate commit and the baseline.
2. Invoke `stats compare` slicing on the dimension that
   separates the two runs (e.g.
   `--a-project-commit <baseline> --b-project-commit <current>`)
   and inspect the host-delta section of its output.
3. Fail (or annotate the PR) if any host dimension changed —
   an unchanged host set is the precondition for a clean A/B
   of scheduler behavior.

## Typical hits

Each bullet names the `show-host` field that carries the signal so
you can `cargo ktstr show-host | grep <field>` directly, or pluck
the same key out of a sidecar via `jq '.host.<field>'`.

- `thp_enabled` (and its companion `thp_defrag`) changed between
  runs → explains latency-sensitive regressions that vanish when
  you pin THP via `transparent_hugepage=` on the kernel cmdline.
  The bracketed selection inside the value is the active setting;
  compare the bracket position, not just the full string.
- A `sched_tunables.*` key differs → the kernel's scheduler tunables
  changed between runs, which can shift the idle-steal pressure on
  `scx_*` schedulers that depend on them. `sched_tunables` captures
  only the surviving `/proc/sys/kernel/sched_*` sysctls (e.g.
  `sched_rt_period_us`, `sched_rt_runtime_us`, `sched_rr_timeslice_ms`,
  `sched_cfs_bandwidth_slice_us`, `sched_schedstats`,
  `sched_util_clamp_*`) — the full set is whatever
  `/proc/sys/kernel/sched_*` lists at capture time. Note: the
  CFS/EEVDF granularity, latency, migration-cost, and base-slice knobs
  (`min_granularity_ns`, `latency_ns`, `wakeup_granularity_ns`,
  `migration_cost_ns`, `base_slice_ns`) live in debugfs
  (`/sys/kernel/debug/sched/`), not `/proc/sys/kernel`, and were moved
  there in Linux 5.13 (well before EEVDF) — `show-host` reads only
  `/proc/sys/kernel`, so it never captures them on any kernel.
- `kernel_cmdline` diverges → `isolcpus=` / `nohz_full=` /
  `mitigations=` / `transparent_hugepage=` / `numa_balancing=`
  are all boot-time and change the whole scheduling surface.
  Rebooting the host to match is the correct remediation when
  you need the comparison to hold. The field is named
  `kernel_cmdline` (not `cmdline`) in both `show-host`'s printed
  output and the sidecar JSON to disambiguate from
  `SidecarResult.kargs`, which carries the extra kargs the ktstr
  VMM appended when booting the guest rather than the running
  host's boot line.
- `kernel_release` differs (also check the companion
  `kernel_name` and `arch` fields) → the kernel itself changed;
  every other host dimension is suspect under cross-kernel
  comparison. A `kernel_name` change (`uname -s` reporting a
  different OS family — `Linux` vs `FreeBSD`, say) is a harder
  stop than a same-family version bump and usually means the
  two sidecars were produced on entirely different systems.
- `hugepages_total` / `hugepages_free` / `hugepages_size_kib`
  deltas → benchmark throughput that depends on 2 MiB pages
  (performance_mode tests) flips outcome when the pool shrinks
  or the page size changes. All three are reported by `show-host`
  in the meminfo-derived block.
- `numa_nodes` differs → cpusets and cross-node migration signals
  only make sense within the CPU→node mapping captured at
  sidecar-write time; a host reconfigured to expose or hide
  nodes changes what `cpus_used` and `numa_pages` mean across
  the two runs. See the
  [capture caveat](#capture-show-host) — `numa_nodes` counts
  only nodes that host at least one CPU (memory-only nodes are
  not counted), so a delta here can reflect either a hardware /
  firmware change or a topology reconfiguration that left the
  memory-only nodes untouched.
- CPU-level skew (`cpu_model` / `cpu_vendor`) → microarchitectural
  differences affect cache-sensitive benchmarks. Always inspect
  alongside `cmdline` because a different CPU usually comes with
  a different bootloader.

## Seeing the raw sidecar field

`show-host` reads the live host; the sidecar carries whatever
`show-host` would have captured at sidecar-write time. To see
the sidecar's host block directly:

```sh
jq '.host' path/to/sidecar.ktstr.json
```

The field is emitted on every gauntlet run.
