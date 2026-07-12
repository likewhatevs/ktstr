# Monitor

When your scheduler leaves runnable work stuck on a CPU, the monitor is what
notices. It runs on the host while the guest executes and reads
scheduler state directly out of guest memory — the scheduler under
test never executes an extra instruction to be observed, and no BPF
probe perturbs the decisions being measured.

Every test report ends with the monitor's summary. From a real run:

<!-- captured: cargo ktstr test --kernel 7.0 -E 'test(throughput_gate)' | ktstr 0.23.0 | kernel 7.0.14 -->
```text
--- monitor ---
samples=41 max_imbalance=2.00 max_dsq_depth=0 stuck=0
avg: imbalance=1.32 nr_running/cpu=1.2 dsq/cpu=0.0
events: fallback=0 (0.0/s) keep_last=210 (52.5/s) offline=0
events+: refill_slice_dfl=210
schedstat: csw=586 (146/s) run_delay=381246314ns/s ttwu=204 goidle=1
bpf: ktstr_select_cp cnt=189 145ns/call
bpf: ktstr_enqueue cnt=373 34ns/call
bpf: ktstr_dispatch cnt=584 237ns/call
verdict: monitor OK
```

Reading it: `samples` is how many point-in-time snapshots the monitor
took; `max_imbalance`/`max_dsq_depth`/`stuck` are the peaks the
threshold checks evaluate; `events` are sched_ext event-counter rates
(select-cpu fallbacks, keep-last dispatches); the `bpf:` lines are
per-callback invocation counts and mean cost from the guest kernel's
BPF program-runtime stats; `verdict` is what folds into the test
result.


<div class="kt-figure"><svg width="700" height="220" viewBox="0 0 700 220" role="img" aria-label="The host-side monitor resolves struct offsets from BTF and reads guest memory directly; nothing is injected into the guest">
  <defs><marker id="kt-arr3" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="var(--kt-accent)"/></marker></defs>
  <rect x="10" y="14" width="300" height="188" rx="12" fill="var(--kt-accent-soft)" stroke="var(--kt-accent)" stroke-width="1.4"/>
  <text x="28" y="40" font-size="12.5" font-weight="700" fill="var(--kt-accent)">guest VM · untouched</text>
  <g font-size="10.5" fill="var(--fg)">
    <rect x="28" y="54" width="264" height="38" rx="7" fill="var(--bg)" stroke="var(--kt-rule)"/>
    <text x="42" y="77" font-family="var(--mono-font)">runqueues · DSQs · schedstat</text>
    <rect x="28" y="100" width="264" height="38" rx="7" fill="var(--bg)" stroke="var(--kt-rule)"/>
    <text x="42" y="123" font-family="var(--mono-font)">scheduler BPF maps · .bss · arena</text>
    <rect x="28" y="146" width="264" height="38" rx="7" fill="var(--bg)" stroke="var(--kt-rule)"/>
    <text x="42" y="169" font-family="var(--mono-font)">task_structs · cgroups</text>
  </g>
  <g>
    <path d="M395 74 L 314 74" stroke="var(--kt-accent)" stroke-width="1.5" marker-end="url(#kt-arr3)"/>
    <path d="M395 120 L 314 120" stroke="var(--kt-accent)" stroke-width="1.5" marker-end="url(#kt-arr3)"/>
    <path d="M395 166 L 314 166" stroke="var(--kt-accent)" stroke-width="1.5" marker-end="url(#kt-arr3)"/>
    <text x="322" y="62" font-size="9.5" fill="var(--fg)" opacity=".65">reads only</text>
  </g>
  <rect x="400" y="14" width="290" height="188" rx="12" fill="none" stroke="var(--kt-rule)" stroke-width="1.4"/>
  <text x="418" y="40" font-size="12.5" font-weight="700" fill="var(--fg)" opacity=".85">host · monitor</text>
  <g font-size="10.5" fill="var(--fg)">
    <text x="418" y="70" opacity=".8">BTF resolves struct offsets</text>
    <text x="418" y="90" opacity=".8">guest memory mapped read-only</text>
    <text x="418" y="110" opacity=".8">samples on its own clock</text>
    <text x="418" y="140" font-weight="700">no BPF injected, no guest agent —</text>
    <text x="418" y="158" font-weight="700">observation cannot perturb</text>
    <text x="418" y="176" font-weight="700">the scheduler under test</text>
  </g>
</svg></div>

## What it reads

The monitor resolves kernel structure offsets from the guest kernel's
BTF — nothing is hardcoded per kernel version. Per CPU, it reads the
runqueue's `nr_running`, `scx_nr_running`, `rq_clock`,
`local_dsq_depth`, and `scx_flags`, plus the sched_ext event counters
(select-cpu fallback, dispatch keep-last, bypass activity, and the
rest of the family). When the guest kernel has `CONFIG_SCHEDSTATS`, it
also reads per-CPU `struct rq` schedstat fields (run_delay, pcount,
ttwu_count, …).

It also walks the `struct sched_domain` tree — from `rq->sd` up the
`sd->parent` chain — whenever the BTF exposes it, capturing per-level
topology metadata (`level`, `name`, `flags`, `span_weight`) and
runtime fields (`balance_interval`, `nr_balance_failed`,
`max_newidle_lb_cost`), plus load-balancing stats when
`CONFIG_SCHEDSTATS` is enabled. Fields that newer kernels added (the
proportional-newidle counters `newidle_call` / `newidle_success` /
`newidle_ratio`, new in 7.0 with some stable backports) resolve as
optional: on kernels whose BTF lacks them they are simply absent, and
the rest of the walk proceeds.

## Sampling

The monitor takes periodic snapshots (`MonitorSample`) of all per-CPU
state; each sample is a point-in-time view of every CPU.
`MonitorSummary` aggregates samples into peak values (max imbalance
ratio, max DSQ depth, stuck-task detection), per-sample averages, and
event-counter deltas. Averages are computed over valid samples only
(excluding uninitialized guest memory — see below).

Each tick the monitor also samples the HOST side: the vCPU threads'
own `/proc/self/task/<tid>/schedstat` (on-CPU vs run-delay), which it
attributes to the guest's current lifecycle phase. This is what feeds
the per-cell host dilation `D` on verdict lines and the per-phase
contention witness — the guest-side `struct rq` reads measure the
scheduler *under test*, the host-side reads measure the weather the
cell ran in. See [Run Modes](../concepts/run-modes.md#dilation-reporting).

## Threshold evaluation

`MonitorThresholds` defines the pass/fail conditions:

| Threshold | Default | Trips when |
|-----------|---------|------------|
| `max_imbalance_ratio` | 4.0 | max/min per-CPU `nr_running` exceeds the ratio |
| `max_local_dsq_depth` | 50 | any CPU's local DSQ exceeds the depth |
| `fail_on_rq_clock_stuck` | true | a CPU's `rq_clock` stops advancing (exemptions below) |
| `max_fallback_rate` | 200.0/s | sustained select-cpu-fallback event rate |
| `max_keep_last_rate` | 100.0/s | sustained dispatch-keep-last event rate |
| `sustained_samples` | 5 | — window: a violation must persist this many consecutive samples |

A violation must persist for `sustained_samples` consecutive samples
before it counts — at the ~100ms sample interval, the default 5 means
roughly 500ms of sustained violation. This filters transient spikes
from cpuset transitions and cgroup creation/destruction. The
reasoning behind each default value lives in
[Checking](../concepts/checking.md).

Test authors do not construct `MonitorThresholds` directly: the
`#[ktstr_test]` threshold attributes (`max_imbalance_ratio`,
`fail_on_rq_clock_stuck`, …) and `Assert::with_monitor_defaults()` feed it —
see [the macro reference](../writing-tests/ktstr-test-macro.md).

**`enforce` is the on/off gate for the threshold-violation path.**
The default is report-only: monitor evaluations record every
violation in the verdict's details, but the verdict passes. Opting in
— via `Assert::with_monitor_defaults()`, which fills unset threshold
fields and sets `enforce` — promotes recorded violations to failures.
Setting a field like `fail_on_rq_clock_stuck` without enforcement is a no-op
for the violation path: the violation appears in the monitor report,
the verdict still passes, and the summary carries a report-only
advisory flagging the missing enforcement.

The no-signal arms bypass `enforce` entirely. An empty sample buffer,
or data that fails the plausibility check below, always produces a
verdict with `passed: false, inconclusive: true`, which folds into
the test's `AssertResult` as Inconclusive (exit code 2). "Couldn't
evaluate" is not the same as "evaluated and OK," so the no-signal
path always surfaces distinct from Pass. Only threshold *violations*
are gated by `enforce`.

### Stuck-task detection

A stuck task is detected when a CPU's `rq_clock` does not advance
between consecutive samples. Three exemptions prevent false positives:

- **Idle CPUs**: when `nr_running == 0` in both the current and
  previous sample, the CPU has no runnable tasks. The kernel stops
  the tick (NOHZ) on idle CPUs, so `rq_clock` legitimately does not
  advance.
- **Preempted vCPUs**: when the vCPU thread's CPU time did not
  advance past the preemption threshold between samples, the host
  preempted the vCPU — the guest never got a chance to run, which is
  not the scheduler's fault.
- **Sustained window**: stuck-task detection uses per-CPU consecutive
  counters and the `sustained_samples` threshold, matching the other
  checks. A single stuck sample does not trigger failure.

### Uninitialized memory detection

Before the guest kernel initializes per-CPU structures, monitor reads
return garbage. Two layers handle this: summary computation skips
individual samples where any CPU's `local_dsq_depth` exceeds a
plausibility ceiling (10,000), and threshold evaluation checks the
whole report — if all `rq_clock` values are identical across every
CPU and sample, or any sample exceeds the ceiling, the report is
classified "not yet initialized" and no per-threshold checks run
(this is one of the Inconclusive arms above).

## The monitor never instruments the guest

Everything above is passive memory reading. The one ktstr feature
that *does* load BPF probes into a guest is
[auto-repro](../running-tests/auto-repro.md) — and it runs them in a
separate, disposable repro VM booted after the original test failed,
never in the VM whose behavior is being measured. The run your
verdict is based on is unperturbed.

## How guest memory is read

Three address-translation modes cover the kernel's address spaces:

- **Text/data/bss** — linear offset from the kernel's static map, for
  statically-linked kernel variables.
- **Direct mapping** — `kva - PAGE_OFFSET`, for SLAB allocations and
  per-CPU data.
- **Vmalloc/vmap** — a real page-table walk through the guest's CR3
  (4- and 5-level paging on x86_64; 4/16/64 KB granules on aarch64),
  for BPF maps and vmalloc'd memory.

All reads are bounds-checked and volatile (the guest modifies memory
concurrently), and the runtime KASLR offset is recovered at startup
so ELF symbols from the matching `vmlinux` resolve to live guest
addresses. `vmlinux` must match the guest kernel — it supplies both
the symbol table and the BTF. The implementing types (`GuestMem`,
`GuestKernel`, `GuestMemMapAccessor`) are documented in the `monitor`
module's rustdoc (`cargo doc --document-private-items`).

## BPF map access

The monitor also discovers and reads/writes the scheduler's BPF maps
directly through guest physical memory — no guest cooperation, no BPF
syscalls. Maps are found by walking the kernel's `map_idr` and
matched by name suffix (`".bss"` matches `"my_sched.bss"`); values are
read at BTF-resolved offsets, including per-CPU array maps. When a
map carries program BTF, the dump renderer uses it to render the
value struct field by field — which is why failure dumps show your
BPF globals by name.

Tests use this through `BpfMapWrite`: a host-side write to a BPF map
during VM execution. The test runner waits for the scheduler to load
(the map becomes discoverable), writes the value, then signals the
guest to start the scenario:

```rust,ignore
const BPF_CRASH: BpfMapWrite = BpfMapWrite::new(".bss", "crash", 1);

#[ktstr_test(bpf_map_write = BPF_CRASH, expect_err = true)]
fn crash_test(ctx: &Ctx) -> Result<AssertResult> {
    Ok(AssertResult::pass())
}
```

The field's byte offset and width are resolved from the map's program
BTF at write time (which also disambiguates same-suffix maps by
picking the one whose BTF names the field). Only array maps and
4-byte scalar fields are supported. For reading map state from test
code, see [Snapshots](../writing-tests/snapshots.md).

## What a dump shows: cast analysis

When a test fails, the failure dump renders the scheduler's BPF map
state — with BTF names, not raw bytes. From a real failure:

<!-- captured: cargo ktstr test --kernel 7.0 -E 'test(throughput_gate)' | ktstr 0.23.0 | kernel 7.0.14 -->
```text
map bpf_bpf.bss (type=array, value_size=448, max_entries=1)
.bss:
  scx_arena_verify_once=true   ktstr_alloc_count=76   nr_dispatched=907
  nr_enqueued=495              nr_select_cpu=372      stats_magic=6004496034161779060
...
    root 0x100000006000 → sdt_desc:
      nr_free=512
      chunk 0x100000007000 (sdt_alloc) → ktstr_arena_ctx{}
  ktstr_bss_arena_holder ktstr_bss_arena_holder:
    bss_plain_counter=76
    arena_target 0x10000000aa80 (cast→arena) [chase: arena chase: STX-flow path tagged slot as Arena with deferred resolve; bridge had no entry for 0x10000000aa80]
```

BPF schedulers frequently store kernel and arena pointers in `u64`
fields, because BTF cannot express a pointer to a per-allocation
type. Without help, the renderer would print those as meaningless
integers. The cast analyzer closes that gap by analyzing the
scheduler binary's BPF bytecode to learn which `u64` fields actually
hold pointers, so the renderer can chase them. The annotations tell
you what happened:

- **`(cast→arena)` / `(cast→kernel)`** — the pointer was recovered by
  cast analysis and chased into arena or kernel memory; the rendered
  fields after it are real dereferenced state, visually distinct from
  natively BTF-typed pointers.
- **`(sdt_alloc)`** — the chase resolved the pointee's type through a
  live arena-allocator slot (the common pattern for
  `scx_task_data()`-style per-task state).
- **`[chase: …]`** — the chase stopped, and this is why. A stopped
  chase falls back to showing the raw value; the analyzer is
  deliberately conservative, preferring a raw `u64` over chasing
  garbage.

The analysis is unconditional — no opt-in, no test-author
configuration — and applies to every snapshot, periodic capture, and
failure dump. Failure dumps are also written machine-readable as a
JSON artifact next to the test outputs, carrying the same
BTF-resolved field names and cast annotations:

<!-- captured: cargo ktstr test --kernel 7.0 -E 'test(throughput_gate)' | ktstr 0.23.0 | kernel 7.0.14 (throughput_gate-*.failure-dump.json) -->
```json
{
  "name": "bpf_bpf.bss",
  "map_kva": 18400526959283003096,
  "map_type": 2,
  "value_size": 448,
  "max_entries": 1,
  "value": {
    "kind": "struct",
    "type_name": ".bss",
    "members": [
      {
        "name": "scx_arena_verify_once",
        "value": {
          "kind": "bool",
          "value": true
        }
      },
...
      {
        "name": "nr_dispatched",
        "value": {
          "kind": "uint",
          "bits": 64,
          "value": 849
        }
      },
...
```

Every field name here was resolved on the host from the guest's BTF —
the guest wrote nothing but its normal map state. See
[Reading Failure Output](../running-tests/failures.md) for the full
anatomy of a failure report.
