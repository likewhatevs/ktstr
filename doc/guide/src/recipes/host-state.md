# Capture and Compare Host State

When a gauntlet run passes on one machine and fails on another — or
passes on Monday and fails on Wednesday — the first thing to check
is whether the host itself changed. `cargo ktstr show-host`
captures a snapshot of the kernel, CPU, memory, scheduler tunables,
and kernel cmdline; `cargo ktstr perf-delta` surfaces the changes
between two runs in a host-delta section so you can see what moved.
(For per-thread profiling see
[ctprof](diagnose-slow-scheduler.md); for scheduler-behavior diffs
between commits see [A/B Compare Branches](ab-compare.md).)

<div class="kt-doc-grid">
<div class="kt-doc-card"><strong>Live host</strong><p><code>show-host</code> reads the machine exactly as it is before the next run.</p></div>
<div class="kt-doc-card"><strong>Archived run</strong><p><code>stats show-host --run</code> shows the host context stored with a sidecar.</p></div>
<div class="kt-doc-card"><strong>Delta</strong><p><code>perf-delta</code> surfaces host changes next to metric changes.</p></div>
</div>

## Live vs archived

Two subcommands print host context; pick the one whose target
matches your question:

- **`cargo ktstr show-host`** reads the live host (`/proc`, `/sys`,
  `uname()`) at invocation time. Use it to inspect the current
  machine — before a benchmark, after a sysctl change, or to
  confirm what the next run here would record.
- **`cargo ktstr stats show-host --run RUN_ID`** prints the
  archived host context captured at sidecar-write time for a past
  run (run keys from `cargo ktstr stats list`). Use it when
  investigating a regression in a past run — what looked like a
  code change might trace back to a host change.

Both render through the same formatter, so the two outputs are
byte-for-byte comparable when the host is unchanged.

## Capture: `show-host`

```sh
cargo ktstr show-host
```

Prints a `key: value` report, one field per line, in a fixed order
(the order and field set are pinned by unit tests):

`kernel_name`, `kernel_release`, `arch` — the `uname()` triple ·
`cpu_model`, `cpu_vendor` — first `/proc/cpuinfo` entry ·
`total_memory_kib`, `hugepages_total`, `hugepages_free`,
`hugepages_size_kib` — from `/proc/meminfo` · `online_cpus`,
`numa_nodes` — node count from the CPU→node mapping (memory-only
nodes are not counted) · `thp_enabled`, `thp_defrag` — transparent
hugepage policy with the bracketed selection preserved verbatim ·
`kernel_cmdline` — `/proc/cmdline` verbatim · `task_delayacct` —
delay-accounting state (`on`, `runtime-off`, or `config-off`);
gates which taskstats delay fields populate ·
`config_task_xacct` — `CONFIG_TASK_XACCT` build state; gates the
taskstats memory-watermark fields · `cpufreq_governor` — one line
per CPU · `sched_tunables` — every `/proc/sys/kernel/sched_*`
sysctl, one entry per line · `heap_state` — the process's own
jemalloc allocator state (rarely relevant to host comparison).

Missing-value rendering is consistent everywhere: a field that
failed to populate prints `(unknown)`; a map that was captured but
empty prints `(empty)`; in per-key diffs, a key present on only one
side prints `(absent)`. The distinction matters — `(empty)` means
the dimension was inspected and had nothing, `(unknown)` means the
capture itself failed.

The output is human-oriented. The same data, same schema, is
attached to every gauntlet-run sidecar under its `host` field:

```sh
jq '.host' path/to/sidecar.ktstr.json
```

## Compare: `perf-delta`'s host-delta section

```sh
cargo ktstr perf-delta --noise-adjust 5 --kernel 6.14
```

[`perf-delta`](ab-compare.md) picks the first sidecar with a
populated `host` field from each side and prints one of: nothing
(neither side carried host context), `host: captured in 'A' only,
delta unavailable` (a one-sided capture failure — your capture
broke, not the host), `host: identical between 'A' and 'B'
(arch: x86_64)`, or a diff. The diff suppresses fields that match —
it is a diff, not a snapshot — and renders one row per changed
field, in this shape:

```text
host delta ('<baseline>' → '<candidate>'):
  <field>: <baseline value> → <candidate value>
  ...
```

An unchanged host is the precondition for a clean A/B of scheduler
behavior. A CI perf-gate that runs `perf-delta` on a pull request
surfaces this section automatically whenever any host field differs
— treat its appearance as a signal the comparison may not hold, and
fail or annotate the PR.

## Typical hits

Each row names the `show-host` field carrying the signal, so you
can `cargo ktstr show-host | grep <field>` or
`jq '.host.<field>' sidecar.ktstr.json` directly.

| Field | Symptom | Fix / interpretation |
|---|---|---|
| `thp_enabled` / `thp_defrag` | Latency-sensitive regressions that come and go between runs | Compare the bracket position (the active setting), not the whole string; pin via `transparent_hugepage=` on the kernel cmdline |
| `sched_tunables.*` | Idle-steal pressure shifted on `scx_*` schedulers that read these sysctls | Restore the changed sysctl; the captured set is whatever `/proc/sys/kernel/sched_*` lists at capture time |
| `kernel_cmdline` | Whole scheduling surface changed (`isolcpus=`, `nohz_full=`, `mitigations=`, `numa_balancing=` are all boot-time) | Reboot the host to match — the only remediation that makes the comparison hold |
| `kernel_release` (with `kernel_name`, `arch`) | Everything is suspect | Cross-kernel comparison; rebuild the baseline on the same kernel |
| `hugepages_total` / `hugepages_free` / `hugepages_size_kib` | `performance_mode` throughput flips when the 2 MiB pool shrinks | Restore the hugepage reservation |
| `numa_nodes` | Cross-node migration and locality signals mean different things across the runs | Hardware/firmware or topology reconfiguration; note memory-only nodes are not counted |
| `cpu_model` / `cpu_vendor` | Cache-sensitive benchmarks moved | Different machine — inspect alongside `kernel_cmdline`, which will usually differ too |

Two disambiguations worth knowing:

- The field is named `kernel_cmdline` (not `cmdline`) in both the
  printed output and the sidecar JSON, to distinguish it from
  `SidecarResult.kargs` — the extra kargs the ktstr VMM appended
  when booting the *guest*, not the running host's boot line.
- The CFS/EEVDF tuning knobs (`base_slice_ns`, `migration_cost_ns`,
  `latency_ns`, …) live in debugfs (`/sys/kernel/debug/sched/`),
  where they moved in Linux 5.13 — not in `/proc/sys/kernel`.
  `show-host` reads only `/proc/sys/kernel`, so `sched_tunables`
  never captures them on any kernel.

## The three-line investigation

The shape this recipe usually takes: a suite regresses with no
scheduler change in the diff. Compare `cargo ktstr stats show-host
--run <old-run>` against live `cargo ktstr show-host`; one delta
appears — say, the `thp_enabled` bracket moved after a distro
update. Pin the setting on the kernel cmdline, rerun, baseline
restored: it was never a scheduler bug.
