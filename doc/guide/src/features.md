<style>
:root { --content-max-width: 1000px; }
</style>

# ktstr in Action

The feature tour: each capability in a few sentences, with real output
where output is the point. Captures are real runs, trimmed for focus;
commands show the intended shape rather than a frozen version promise.

## Testing

### Real kernel, clean slate, x86/arm parity

Every VM test boots its own Linux kernel in KVM — fresh state each run,
no shared daemons, no containers. Topology is configurable per test:
NUMA nodes, LLCs, cores per LLC, threads per core, with real ACPI
SRAT/SLIT tables on x86_64 and FDT cpu nodes on aarch64 — 24 topology
presets on x86_64, 14 on aarch64. The framework drives the whole
lifecycle: boot, attach, scenario, collect, teardown. See
[Topology](concepts/topology.md).

### Fast boot

Initramfs base, payload, and module parts are compressed once per
content-addressed recipe and stored in a persistent regular-file cache.
Concurrent VMs privately COW-map the cached ranges instead of rebuilding or
copying them, so boot is dominated by kernel init rather than initramfs
preparation. This applies to every supported initrd compression format.
Measured on a 64-CPU host:

<div class="kt-kpis">
<div class="kt-kpi"><strong>55.583µs</strong><span>initramfs spawn</span></div>
<div class="kt-kpi"><strong>867.005µs</strong><span>KVM + kernel setup</span></div>
<div class="kt-kpi"><strong>1.409s</strong><span>VM setup total</span></div>
</div>

<!-- captured: cargo ktstr test --kernel 7.0 (VM setup timings, stderr) | ktstr 0.23.0 | kernel 7.0.14 -->
<div class="kt-term"><div class="kt-term-bar"><span class="kt-term-title">VM setup timings</span></div>

<pre>  initramfs spawn: <span class="t-cyn">55.583µs</span>
  kvm+kernel: <span class="t-cyn">867.005µs</span>
  setup_memory (joins initramfs): 1.409360963s
  setup_vcpus: 1.409565321s
<span class="t-b">VM setup total: 1.409619773s</span></pre></div>

### Declarative scheduler registration

One macro declares a scheduler — binary, default topology, kernel filter
for the verifier sweep, assertion overrides, always-on CLI args — and
tests reference the const it emits:

```rust,ignore
use ktstr::declare_scheduler;

declare_scheduler!(MY_SCHED, {
    name = "my_sched",
    binary = "scx_my_sched",
    topology = (1, 2, 4, 1),
    sched_args = ["--exit-dump-len", "1048576"],
});
```

Schedulers that take a `--config` JSON file declare the arg template
once; each test supplies its config inline via `config = …`. See
[Scheduler Definitions](writing-tests/scheduler-definitions.md) and
[The #\[ktstr_test\] Attribute](writing-tests/ktstr-test-macro.md).

### Data-driven scenarios

You declare intent as data — cgroups, cpusets, workloads, mid-run ops —
and the framework creates cgroups, spawns workers, applies policies and
affinity, and tears it all down. Canned scenarios grouped by scheduling
concern — affinity, cpusets, dynamic cgroups, nesting, contention,
stress — cover the common patterns; the `ops` DSL underneath (`Step`,
`Op`, `Backdrop`) expresses the rest. See
[Scenarios](writing-tests.md#scenarios) and
[Ops, Steps, and Backdrop](concepts/ops.md).

### 45 work types

Each work type targets a specific scheduling pressure, so a test can pin
the kernel path a regression lives in. A sample of the 45:

| Pressure | Examples |
|---|---|
| CPU and IPC | `SpinWait`, `AluHot`, `IpcVariance` |
| Wakeup placement | `FutexPingPong`, `WakeChain`, `PipeIo` |
| Task churn | `ForkExit`, `CgroupAttachStorm`, `AffinityChurn` |
| Priority and RT monopolization | `PreemptStorm`, `RtStarvation`, `PriorityInversion` |
| Memory and NUMA | `CachePressure`, `PageFaultChurn`, `NumaWorkingSetSweep` |
| IRQ and I/O | `IrqWake`, `NetTraffic`, `IoConvoy` |
| Benchmarks | `Schbench`, `Taobench` |

Workers can also set `comm`, nice level, and thread-group-leader name
(`pcomm`) to model real applications, and `Custom` takes a user-supplied
work function. Full catalog: [Work Types](concepts/work-types.md).

### Gauntlet

A single `#[ktstr_test]` auto-expands across topology presets, and
multi-kernel runs (`--kernel A --kernel B`) add the kernel as another
matrix dimension. Budget-based selection maximizes coverage within a CI
time limit; multi-NUMA and very large presets are opt-in via constraint
attributes. See [Gauntlet](running-tests/gauntlet.md).

### Real-kernel BPF verifier analysis

The verifier sweep boots a VM per (scheduler × kernel × topology
preset), loads the scheduler through struct_ops — the same path
production uses — and reads actual verified instruction counts from
guest memory. Topology is a real verification axis: values baked into
`.rodata` (like CPU counts) change what the verifier explores, so a
scheduler can attach on one topology and be rejected on another.

<!-- captured: cargo ktstr verifier --kernel <local sched_ext dev tree> --test docs_real_scheds 4cpu-1llc-nosmt 4cpu-2llc-nosmt 9cpu-3llc-nosmt 8cpu-2llc-smt | ktstr 0.23.0 | kernel sched_ext-for-7.2 b4dc42d2 -->
<div class="kt-term"><div class="kt-term-bar"><span class="kt-term-title">cargo ktstr verifier --kernel ../linux --test my_schedulers</span></div>

<pre>verifier verified_insns (per scheduler; rows: kernel, cols: BPF program, cell: range across topologies):

scx_p2dq:
 kernel        p2dq_dequeue  p2dq_dispatch  p2dq_enqueue  p2dq_exit  p2dq_exit_task  p2dq_init  p2dq_init_task  p2dq_running  p2dq_select_cpu  ...
 kernel_local  5             1159..2130     2026..5118    25         419             2121       27601           609           801..887         ...

verifier results (per scheduler; rows: topology, cols: kernel):

scx_bpfland: 4 ✅  0 ❌
┌─────────────────┬──────────────┐
│ topology        │ kernel_local │
╞═════════════════╪══════════════╡
│ 4cpu-1llc-nosmt │ <span class="t-grn">✓</span>            │
├─────────────────┼──────────────┤
│ 4cpu-2llc-nosmt │ <span class="t-grn">✓</span>            │
├─────────────────┼──────────────┤
│ 8cpu-2llc-smt   │ <span class="t-grn">✓</span>            │
├─────────────────┼──────────────┤
│ 9cpu-3llc-nosmt │ <span class="t-grn">✓</span>            │
└─────────────────┴──────────────┘

scx_lavd: 0 ✅  <span class="t-red">4 ❌</span>
┌─────────────────┬──────────────┐
│ topology        │ kernel_local │
╞═════════════════╪══════════════╡
│ 4cpu-1llc-nosmt │ <span class="t-red">✗</span>            │
├─────────────────┼──────────────┤
│ 4cpu-2llc-nosmt │ <span class="t-red">✗</span>            │
├─────────────────┼──────────────┤
│ 8cpu-2llc-smt   │ <span class="t-red">✗</span>            │
├─────────────────┼──────────────┤
│ 9cpu-3llc-nosmt │ <span class="t-red">✗</span>            │
└─────────────────┴──────────────┘

scx_p2dq: 4 ✅  0 ❌
┌─────────────────┬──────────────┐
│ topology        │ kernel_local │
╞═════════════════╪══════════════╡
│ 4cpu-1llc-nosmt │ <span class="t-grn">✓</span>            │
├─────────────────┼──────────────┤
│ 4cpu-2llc-nosmt │ <span class="t-grn">✓</span>            │
├─────────────────┼──────────────┤
│ 8cpu-2llc-smt   │ <span class="t-grn">✓</span>            │
├─────────────────┼──────────────┤
│ 9cpu-3llc-nosmt │ <span class="t-grn">✓</span>            │
└─────────────────┴──────────────┘</pre></div>

That sweep is real: on this development kernel `scx_bpfland` and
`scx_p2dq` verify, attach, and dispatch on all four topologies, while
`scx_lavd` fails every cell — the kernel removed the deprecated
`scx_bpf_cpu_rq()` kfunc that lavd still requires, and the failing
cells' logs name it (`extern (func ksym) 'scx_bpf_cpu_rq': not found
in kernel or module BTFs`). See
[BPF Verifier Sweep](running-tests/verifier.md) for the full run.

On rejection, the log is cycle-collapsed — repeated loop-unrolling
iterations are deduplicated so the offending access is readable:

<!-- captured: cargo ktstr verifier --kernel 7.0 --scheduler ktstr_broken --test verifier_pipeline 4cpu-1llc-nosmt | ktstr 0.23.0 | kernel 7.0.14 -->
<div class="kt-term"><div class="kt-term-bar"><span class="kt-term-title">cycle-collapsed verifier rejection</span></div>

<pre><span class="t-red">Global function ktstr_dispatch() doesn't return scalar. Only those are supported.</span>
...
--- 8x of the following 25 lines ---
; u64 t = bpf_ktime_get_ns(); @ main.bpf.c:453
38: (85) call bpf_ktime_get_ns#5      ; R0=scalar()
...
--- 6 identical iterations omitted ---
...
--- end repeat ---
192: (63) *(u32 *)(r1 +0) = r2
<span class="t-red">R1 invalid mem access 'scalar'</span>
processed 186 insns (limit 1000000) max_states_per_insn 0 total_states 7 peak_states 7 mark_read 0</pre></div>

### Bare-metal export

`cargo ktstr export` packages a registered test as a self-extracting
`.run` script that reproduces the scenario on real hardware, no VM. The
script freezes the scheduler binary, its args and config files, and the
required topology; it validates the host and refuses to displace an
already-attached sched_ext scheduler.

<!-- captured: cargo ktstr export sched_basic_proportional --package ktstr -o /tmp/sched_basic_proportional.run | ktstr 0.23.0 | host-side, no VM -->
<div class="kt-term"><div class="kt-term-bar"><span class="kt-term-title">cargo ktstr export sched_basic_proportional</span></div>

<pre><span class="t-grn">wrote /tmp/sched_basic_proportional.run (90074903 bytes archive, 0 include files)</span>
----- head -40 of the generated script -----
#!/bin/bash
# Generated by `cargo ktstr export`. Do not edit; regenerate to update.
...
# --- frozen test specification ---
KTSTR_TEST_NAME=sched_basic_proportional
KTSTR_SCHED_NAME=ktstr_sched
KTSTR_GIT_HASH=73730e0
NEED_LLCS=1
NEED_CORES_PER_LLC=2
NEED_THREADS_PER_CORE=1
NEED_NUMA_NODES=1
...
</pre></div>

See [cargo ktstr](running-tests/cargo-ktstr.md).

## Observability

### Zero-perturbation introspection

Everything is built on direct reads of guest physical memory from the
host via the KVM memory mapping. Kernel state — per-CPU runqueues,
sched_domain trees, schedstat and sched_ext event counters — is read
through BTF-resolved struct offsets; BPF maps get typed field access via
program BTF. No guest-side instrumentation, no BPF syscalls: the
observer does not perturb the scheduler under test. See
[Monitor](architecture/monitor.md).

### Cast analysis

Schedulers stash kernel and arena pointers in BPF map fields declared as
`u64`, because BTF cannot express those pointer types. The cast analyzer
walks the scheduler's instruction stream and proves which fields are
really pointers and to what — so dumps chase through them and print
typed structs annotated `(cast→arena)` or `(cast→kernel)` instead of raw
hex. It runs on every scheduler load, no configuration (the failure-dump
excerpt below shows it in action). See
[Monitor](architecture/monitor.md).

### Periodic capture and temporal assertions

`#[ktstr_test(num_snapshots = N)]` samples BPF map fields and scheduler
stats at N points across the workload window, from outside the guest.
Temporal patterns — `nondecreasing`, `rate_within`, `steady_within`,
`converges_to`, and friends — assert over the whole series; on-demand
and write-triggered snapshots share the same machinery. See
[Snapshots and Live Capture](writing-tests/snapshots.md) and
[Projections and Temporal Assertions](writing-tests/temporal-assertions.md).

### Statistical regression detection

Every run writes machine-readable results; cross-run comparison with
dual-gate significance thresholds (absolute and relative) catches
regressions single-run assertions miss. Gated metrics include
`worst_spread`, `worst_gap_ms`, `worst_p99_wake_latency_us`, and the
duration-invariant `total_run_delay_ns_per_sched`; run
`cargo ktstr stats list-metrics` for the registry. See
[Runs and Regression Gates](running-tests/runs.md) and
[Assertable Metrics](reference/assertable-metrics.md).

## Debugging

### Failure dumps

A failing test's stderr carries the whole story: the tripped check,
per-cgroup stats, a phase timeline, the scheduler log, the monitor's
verdict. On scheduler crash, ktstr also snapshots every BPF map with
fields rendered by name through BTF — `.bss` globals, arena allocator
state, typed pointer chases — plus vCPU registers at the instant of
death:

<!-- captured: cargo ktstr test --kernel 7.0 -- --features integration -E 'test(=ktstr/throughput_gate)' | ktstr 0.23.0 | kernel 7.0.14 -->
<div class="kt-term"><div class="kt-term-bar"><span class="kt-term-title">cargo ktstr test — failure dump excerpt</span></div>

<pre>--- repro VM failure dump ---
DualFailureDumpReport: early=absent (max_age never crossed threshold (peak=66j, threshold=2500j)), late=(12 maps, 2 vcpu_regs)
...
map bpf_bpf.bss (type=array, value_size=448, max_entries=1)
<span class="t-yel">.bss:</span>
  scx_arena_verify_once=true   ktstr_alloc_count=76   nr_dispatched=907
  nr_enqueued=495              nr_select_cpu=372      stats_magic=6004496034161779060
...
  scx_task_allocator scx_allocator:
...
    root 0x100000006000 → sdt_desc:
      nr_free=512
<span class="t-grn">      chunk 0x100000007000 (sdt_alloc) → ktstr_arena_ctx{}</span>
...
<span class="t-b">vcpu_regs:</span>
  vcpu 0: ip=0xffffffff96347fbf sp=0xffffffff97203e78 ptroot=0x0000000001e85003
  vcpu 1: ip=0xffffffff9560bdc5 sp=0xff3b18cb8000f778 ptroot=0x0000000001e85003</pre></div>

The same report is written as a JSON artifact next to the run's stats
sidecar. See [Reading Failure Output](running-tests/failures.md) and
[Snapshots](writing-tests/snapshots.md).

### Auto-repro

On a scheduler crash, ktstr extracts the crash stack, discovers the
struct_ops callbacks, and reruns the scenario in a second VM with BPF
probes attached along the crash path — decoded function arguments and
struct state at each call site, `→` arrows marking entry-to-exit
changes:

<!-- captured: cargo ktstr test (ktstr/bpf_crash_auto_repro_e2e) — prior-run sample preserved from running-tests/auto-repro.md | ktstr 0.23.0 | kernel with the scheduler-exit probe trigger -->
<div class="kt-term"><div class="kt-term-bar"><span class="kt-term-title">auto-repro probe excerpt</span></div>

<pre>
enqueue_task_scx                                              kernel/sched/ext.c
  rq *rq
    cpu         1
  task_struct *p
    pid         97
    cpus_ptr    0xf(0-3)
    <span class="t-grn">dsq_id      SCX_DSQ_INVALID          →  SCX_DSQ_LOCAL</span>
...
    <span class="t-grn">scx_flags   QUEUED|DEQD_FOR_SLEEP    →  QUEUED</span></pre></div>

On by default. ktstr selects the kernel's native scheduler-exit hook:
`tp_btf/sched_ext_exit` on the newest kernels,
a raw `scx_vexit` entry/return pair on the preceding generation, or filtered
`fentry/scx_dump_state` on global-era kernels.
See [Auto-Repro](running-tests/auto-repro.md).

### Interactive shell

`ktstr shell` boots a busybox VM and drops you into it.
`--include-files` injects host binaries with their shared-library
closure resolved automatically (recursive `DT_NEEDED` discovery);
`--exec "cmd"` runs one command non-interactively. For debugging, not
tests. See [the standalone ktstr binary](running-tests/cargo-ktstr.md#the-standalone-ktstr-binary).

### ctprof

`ktstr ctprof capture` snapshots per-task and per-cgroup scheduler
telemetry on the host — no VM involved. Capture before and after a
change, then `compare` to see which processes scheduled differently:

<!-- captured: ktstr ctprof compare base.ctprof.zst cand.ctprof.zst --limit 24 | ktstr 0.23.0 | host-side, no VM -->
<div class="kt-term"><div class="kt-term-bar"><span class="kt-term-title">ktstr ctprof compare base.ctprof.zst cand.ctprof.zst</span></div>

<pre><span class="t-b">## Primary metrics</span>
 comm                              threads  metric             value                delta      %         %uptime 
 kworker/{N}:{N}-mm_percpu_wq                                                                                    
     kworker/{N}:{N}-mm_percpu_wq  11→37    voluntary_csw      8.697K → 101.154K    <span class="t-yel">+92.457K</span>   +1063.1%  93%     
     kworker/{N}:{N}-mm_percpu_wq  11→37    timeslices         8.699K → 101.166K    <span class="t-yel">+92.467K</span>   +1063.0%  93%     
     kworker/{N}:{N}-mm_percpu_wq  11→37    wait_time_ns       2.684s → 27.653s     <span class="t-yel">+24.969s</span>   +930.2%   93%     
...
</pre></div>

See [ctprof](reference/ctprof.md).

## Infrastructure

### Supported kernels

| Capability | Kernel requirement |
|---|---|
| CI-tested series | 6.14 and 7.1, on x86_64 and aarch64, every push |
| Watchdog-timeout override | 7.1+ via BTF (`scx_sched.watchdog_timeout`); older kernels via the static `scx_watchdog_timeout` symbol |
| sched_ext event counters | 6.16+ (two BTF layouts); sampling is disabled when neither is present |
| Auto-repro probe trigger | `sched_ext_exit` typed tracepoint (newest), raw `scx_vexit` entry/return pair (preceding generation), or two-argument `scx_dump_state` (global era) |

Outside the CI-tested series, the monitor degrades feature by feature
rather than failing: tests still run, and unavailable capabilities are
reported as absent.

### Kernel management

`cargo ktstr kernel build` builds and caches kernel images from version
numbers, local source paths, or git URLs; automatic discovery resolves
cached images, host kernels, and CI-provided paths, and an optional GHA
cache backend shares built kernels across CI runs. See
[cargo ktstr](running-tests/cargo-ktstr.md) and [CI](ci.md).

### Performance mode

Performance mode pins vCPUs to reserved host cores, pre-faults 2 MB
hugepages, runs vCPU threads under SCHED_FIFO, and suppresses PAUSE/HLT
exits — removing host scheduling noise so a latency spike in the guest
points at the scheduler under test, not the host. Guest-side jitter from
shared LLCs and memory bandwidth remains, so compare performance-mode
runs only against the same host. When the host can't physically satisfy
a declared topology, `no_perf_mode` builds the VM anyway and skips the
isolation. See [Performance Mode](concepts/performance-mode.md).

### Resource-budget coordination

`--cpu-cap N` confines kernel builds and no-perf-mode VMs to `N` host
CPUs, reserved whole-LLC-at-a-time and coordinated between concurrent
ktstr processes through per-LLC locks — so a kernel build and a
performance run can share a box without trampling each other.
`ktstr locks` lists every held lock with its holder. See
[Resource Budget](concepts/resource-budget.md) and
[cargo ktstr locks](running-tests/cargo-ktstr.md#locks).

### Change-scoped selection

`cargo ktstr affected` attributes a `base..HEAD` diff to the schedulers
it touches and emits a JSON array ready for a GitHub Actions dynamic
matrix — one job per affected scheduler instead of the whole fleet.
`--relevant` applies the same attribution to the local working tree for
a fast inner loop. Any uncertainty widens to "run all": silently
skipping an affected scheduler is the worst outcome. See [CI](ci.md).

### Guest coverage

Profraw data is collected inside the VM over shared memory and merged
with host coverage, so guest-side code paths count toward the same
`cargo llvm-cov` report as host-side ones.

---

Ready to try it? [Getting Started](getting-started.md) sets up your
first test; the [Recipes](recipes.md) cover complete workflows.
