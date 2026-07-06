# ktstr (standalone)

`ktstr` is the standalone debugging companion to the
[`#[ktstr_test]`](../writing-tests/ktstr-test-macro.md) test harness.
It owns interactive VM shells, host topology inspection, host-wide
per-thread profiling, and lock introspection — the operations a
scheduler author reaches for when investigating a test failure.

<div class="kt-doc-grid">
<div class="kt-doc-card"><strong>Inspect</strong><p><code>topo</code>, <code>kernel</code>, and <code>locks</code> explain the host and cache state.</p></div>
<div class="kt-doc-card"><strong>Debug</strong><p><code>shell</code> boots an interactive VM with the same kernel and topology plumbing.</p></div>
<div class="kt-doc-card"><strong>Profile</strong><p><code>ctprof</code> captures host scheduler state before and after suspicious workloads.</p></div>
</div>

To run the test suite, use
[`cargo ktstr test`](cargo-ktstr.md#test); to reproduce a test as a
self-contained script without a VM, use
[`cargo ktstr export`](cargo-ktstr.md#export).

A typical failure investigation chains the two binaries: a test
fails ([read the output](failures.md)), you boot the same
environment interactively with `cargo ktstr shell --test NAME`, and
if the question is "what else changed on this host?", you bracket
the workload with `ktstr ctprof capture` and diff the snapshots.

Build from the workspace:

```sh
cargo build --bin ktstr
```

## Subcommands

### topo

Show the host CPU topology — the same view the resource planner and
performance mode use:

<!-- captured: ktstr topo | ktstr 0.23.0 -->
```text
$ ktstr topo
CPUs:       64
LLCs:       4
NUMA nodes: 1
  LLC 0 (node 0): [0, 1, 2, 3, 4, 5, 6, 7, 32, 33, 34, 35, 36, 37, 38, 39]
  LLC 1 (node 0): [8, 9, 10, 11, 12, 13, 14, 15, 40, 41, 42, 43, 44, 45, 46, 47]
  LLC 2 (node 0): [16, 17, 18, 19, 20, 21, 22, 23, 48, 49, 50, 51, 52, 53, 54, 55]
  LLC 3 (node 0): [24, 25, 26, 27, 28, 29, 30, 31, 56, 57, 58, 59, 60, 61, 62, 63]
```

This box is `1n4l8c2t` in ktstr's
[topology notation](../concepts/topology.md): 1 NUMA node, 4 LLCs,
8 cores per LLC, 2 threads per core — note the SMT siblings (CPU 0
pairs with CPU 32).

### kernel

Manage cached kernel images: `list`, `build`, `clean`. Identical to
the [cargo-ktstr kernel](cargo-ktstr.md#kernel) subcommands — see
there for full documentation.

### shell

Boot an interactive shell in a KVM VM. The guest is a busybox
userland with your files mounted at `/include-files/`:

```sh
ktstr shell
ktstr shell --kernel ../linux
ktstr shell --kernel 6.14.2 --topology 1,2,4,1
ktstr shell -i ./my-binary -i strace
ktstr shell --exec 'cat /proc/schedstat'
```

Files and directories passed via `-i` land at
`/include-files/<name>` inside the guest. Directories are walked
recursively; bare names (no path separator) are resolved via `PATH`;
dynamically-linked ELF binaries get automatic shared-library
resolution, and non-ELF files are copied as-is.

Stdin must be a terminal: the host terminal enters raw mode for
bidirectional forwarding, and the saved terminal state is restored
on exit paths ktstr controls (normal exit, errors, catchable fatal
signals). A SIGKILL cannot be intercepted — run `reset` if the
terminal is left raw.

| Flag | Default | Description |
|------|---------|-------------|
| `--kernel ID` | auto | Same kernel grammar as [`cargo ktstr test --kernel`](cargo-ktstr.md#common-flags) (path, version, cache key, range, git source), resolving to a single kernel; raw image files are rejected here. When absent, resolves via cache then filesystem, falling back to downloading the latest stable kernel. |
| `--topology N,L,C,T` | `1,1,1,1` | Virtual topology as `numa_nodes,llcs,cores,threads`. All values must be >= 1. |
| `-i, --include-files PATH` | — | Files or directories to include in the guest. Repeatable. |
| `--memory-mib MiB` | auto | Guest memory in MiB (minimum 128). When absent, estimated from payload and include-file sizes. |
| `--dmesg` | off | Forward the guest kernel console (COM1/dmesg) to stderr in real time; sets `loglevel=7`. |
| `--exec CMD` | — | Run a command instead of an interactive shell; the VM exits when it completes. |
| `--exec-timeout DURATION` | `120s` | Max wall-clock for a `--exec` payload before the VM is killed (`30s`, `5m`, `1h`). |
| `--no-perf-mode` | off | Disable all [performance mode](../concepts/performance-mode.md) features. Also via `KTSTR_NO_PERF_MODE`. |
| `--cpu-cap N` | unset | Reserve only N host CPUs for the shell VM; requires `--no-perf-mode`. See [Resource Budget](../concepts/resource-budget.md). |
| `--disk SIZE` | unset | Attach a raw virtio-blk disk at `/dev/vda`, backed by a fresh sparse tempfile. IEC sizes only (`256mib`, `1gib`). |

[`cargo ktstr shell`](cargo-ktstr.md#shell) runs the same boot flow
with two additions: it accepts raw kernel image files for
`--kernel`, and it has `--test NAME` to derive topology, memory, and
include files from a registered `#[ktstr_test]`.

### ctprof

Capture or compare a host-wide per-thread snapshot — for "the
scheduler looks fine but something on the host is still behaving
oddly". Every visible thread's scheduling, memory, and I/O counters
are snapshotted as zstd-compressed JSON:

```sh
ktstr ctprof capture --output baseline.ctprof.zst
# ... run the workload of interest ...
ktstr ctprof capture --output candidate.ctprof.zst
ktstr ctprof compare baseline.ctprof.zst candidate.ctprof.zst
```

Cumulative counters and lifetime peaks are probe-timing-invariant —
sampled twice, a value either increased monotonically or stayed at
its high-water mark — so a diff between two snapshots measures
exactly the activity over the window. Capture uses no kprobes or
kernel tracing and does not modify thread state; the only exception
is the jemalloc-only memory fields, read by briefly ptrace-attaching
jemalloc-linked processes (needs root, `CAP_SYS_PTRACE`, or
`ptrace_scope=0`; recorded as zero when denied).

`compare` joins two snapshots on a grouping axis and renders
per-metric baseline/candidate/delta rows, sorted by largest relative
change. Real output (a cargo build ran between the snapshots):

<!-- captured: ktstr ctprof compare /tmp/ktstr-docs-base.ctprof.zst /tmp/ktstr-docs-cand.ctprof.zst --limit 24 | ktstr 0.23.0 -->
```text
## Primary metrics
 comm                              threads  metric             value                delta      %         %uptime
 kworker/{N}:{N}-mm_percpu_wq
     kworker/{N}:{N}-mm_percpu_wq  11→37    voluntary_csw      8.697K → 101.154K    +92.457K   +1063.1%  93%
     kworker/{N}:{N}-mm_percpu_wq  11→37    timeslices         8.699K → 101.166K    +92.467K   +1063.0%  93%
     kworker/{N}:{N}-mm_percpu_wq  11→37    wait_time_ns       2.684s → 27.653s     +24.969s   +930.2%   93%
     kworker/{N}:{N}-mm_percpu_wq  11→37    stime_clock_ticks  22ticks → 217ticks   +195ticks  +886.4%   93%
     kworker/{N}:{N}-mm_percpu_wq  11→37    run_time_ns        243.378ms → 2.320s   +2.077s    +853.4%   93%
...
```

Thread names are token-normalized (`kworker/3:1` and `kworker/7:0`
fold into `kworker/{N}:{N}`), so the join key survives across
process restarts and even across hosts — deltas reflect the named
workload, not a specific pid.

Choosing `--group-by`: start with the default `all` (cgroup, then
process, then thread pattern — it folds renamed-but-identical
cgroups together); use `pcomm` when you think in processes, `cgroup`
when comparing services or containers, and `comm` / `comm-exact`
when a single thread pool is the suspect. Most-used compare flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--group-by AXIS` | `all` | `all`, `pcomm`, `cgroup`, `comm`, or `comm-exact` (literal thread names). |
| `--sections NAMES` | every | Sub-tables to render, e.g. `primary`, `taskstats-delay`, `derived`, `pressure`, `smaps-rollup`. |
| `--metrics NAMES` | every | Metric allowlist (names from `ktstr ctprof metric-list`). |
| `--sort-by SPEC` | largest `\|delta_pct\|` | Multi-key sort: `metric[:asc\|desc],...`. |
| `--limit N` | `500` | Max rendered lines per section; `0` disables truncation. |

`show` renders a single snapshot without diff math, and
`metric-list` prints the metric vocabulary — see the
[ctprof reference](../reference/ctprof.md) for those, the full flag
tables, aggregation rules, and taskstats kconfig gating.

### locks

Enumerate every ktstr flock held on this host — read-only, never
acquires anything. When a build or test is blocked behind a peer's
reservation, `ktstr locks` names the peer without disturbing it:

<!-- captured: ktstr locks | ktstr 0.23.0 -->
```text
$ ktstr locks
LLC locks:
 LLC  NODE  LOCKFILE               HOLDERS
 0    0     /tmp/ktstr-llc-0.lock  <none recorded>
 1    0     /tmp/ktstr-llc-1.lock  <none recorded>
...

Run-dir locks:
 RUN KEY               LOCKFILE                                       HOLDERS
 7.0.14-73730e0-dirty  target/ktstr/.locks/7.0.14-73730e0-dirty.lock  <none recorded>
```

An idle host shows `<none recorded>`; while a lock is held, the
`HOLDERS` column names the holder's PID and cmdline
(cross-referenced against `/proc/locks`). Four lock-file roots are
scanned:

- `{KTSTR_LOCK_DIR}/ktstr-llc-*.lock` (default `/tmp`) — per-LLC
  reservations held by perf-mode test runs and `--cpu-cap`-bounded
  builds.
- `{KTSTR_LOCK_DIR}/ktstr-cpu-*.lock` — per-CPU reservations from
  the same flow.
- `{cache_root}/.locks/*.lock` — kernel-cache entry locks held
  during `kernel build` writes, plus per-source-tree locks held
  while building from a path.
- `{runs_root}/.locks/{kernel}-{project_commit}.lock` — sidecar
  write locks serializing concurrent runs targeting the same run
  directory.

| Flag | Default | Description |
|------|---------|-------------|
| `--json` | off | JSON snapshot (pretty in one-shot mode; ndjson under `--watch`). |
| `--watch DURATION` | unset | Redraw at the interval until SIGINT (`100ms`, `1s`, `5m`). |

Available identically as `cargo ktstr locks`. The reservation model
behind these locks is documented in
[Resource Budget](../concepts/resource-budget.md).

### completions

Generate shell completions (`bash`, `zsh`, `fish`, `elvish`,
`powershell`):

```sh
ktstr completions bash
```

`--binary NAME` overrides the registered name when invoking ktstr
through a differently-named symlink. The same subcommand exists as
`cargo ktstr completions`.
