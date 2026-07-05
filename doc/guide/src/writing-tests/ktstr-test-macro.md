# The #\[ktstr_test\] Attribute

`#[ktstr_test]` registers a function as an integration test that
boots a VM with a declared topology and runs the function body inside
it. This page is the attribute reference.

Most tests need only a handful of attributes: a scheduler, a
topology dimension or two, a duration, and the checking thresholds
the test is actually about.

```rust,ignore
use ktstr::prelude::*;

declare_scheduler!(MY_SCHED, {
    name = "my_sched",
    binary = "scx_my_sched",
    topology = (1, 2, 4, 1),
});

#[ktstr_test(
    scheduler = MY_SCHED,   // scheduler under test (default: kernel EEVDF)
    threads = 2,            // override one dimension; the rest inherit
    duration_s = 10,        // workload window (default 12 s)
    not_starved = true,     // enable starvation / spread / gap checks
    max_spread_pct = 20.0,  // tighten the fairness-spread threshold
)]
fn smt_fairness(ctx: &Ctx) -> Result<AssertResult> {
    execute_defs(ctx, vec![ctx.cgroup_def("cg_a"), ctx.cgroup_def("cg_b")])
}
```

The function signature is
`fn(&Ctx) -> anyhow::Result<AssertResult>`. `scheduler =` expects the
bare const emitted by `declare_scheduler!` — see
[Scheduler Definitions](scheduler-definitions.md).

## Attribute forms

All attributes are optional, with defaults, and most take
`key = value`. The sixteen bool attributes (`auto_repro`,
`expect_auto_repro`, `not_starved`, `isolation`, `performance_mode`,
`pci`, `no_perf_mode`, `requires_smt`, `expect_err`,
`survives_storm`, `allow_inconclusive`, `fail_on_stall`, `host_only`,
`ignore`, `kaslr`, `wprof`) also accept a bare form as shorthand for
`= true` — `#[ktstr_test(host_only)]` equals
`#[ktstr_test(host_only = true)]`. `auto_repro` and `kaslr` default
to `true`, so their meaningful spelling is `auto_repro = false` /
`kaslr = false`; the other fourteen default to `false` (or unset).

Each attribute key may appear at most once per invocation; a
duplicate key fails at macro expansion rather than silently letting
the later value win.

## Topology

| Attribute | Default | Description |
|---|---|---|
| `numa_nodes` | inherited | NUMA nodes |
| `llcs` | inherited | Total LLCs (not per node) |
| `cores` | inherited | Cores per LLC |
| `threads` | inherited | Threads per core |
| `memory_mib` | 2048 | VM memory floor in MiB (see below) |

Each dimension independently inherits from `Scheduler.topology` when
a `scheduler` is specified and that dimension is not set. Without a
scheduler, unset dimensions use the macro defaults (`numa_nodes = 1`,
`llcs = 1`, `cores = 2`, `threads = 1`). See
[Topology](../concepts/topology.md) for the notation and what the
guest actually gets.

### Memory

`memory_mib` is one of three floors: the framework allocates
`max(total_cpus * 64, 256, memory_mib)` MiB at VM launch. Above 32
vCPUs the CPU-based floor dominates the default 2048, so a 126-vCPU
test gets 8064 MiB regardless. Raise `memory_mib` only when the test
needs more headroom than the per-CPU budget provides.

## Timing

| Attribute | Default | Description |
|---|---|---|
| `duration_s` | 12 | Workload window in seconds (`ctx.duration`) |
| `watchdog_timeout_s` | 5 | sched_ext watchdog override in seconds |

The watchdog override is applied via `scx_sched.watchdog_timeout` on
7.1+ kernels and via the static `scx_watchdog_timeout` symbol on
earlier kernels; when neither path is available the override logs a
warning and the kernel default stands.

## Checking thresholds {#checking}

Checking attributes override the merged check set
(library defaults → scheduler-level `assert` → per-test attributes).
[Checking](../concepts/checking.md) explains the two evaluation
channels; [Customize Checking](../recipes/custom-checking.md) owns
the merge rules and worked overrides. Everything here is inherited
when unset.

Worker checks — evaluated from per-worker telemetry after the
scenario:

| Attribute | Unit | Example | Fails when |
|---|---|---|---|
| `not_starved` | bool | `not_starved = true` | any worker finishes with zero work units; also enables the spread and gap checks |
| `isolation` | bool | `isolation = true` | a worker ran on a CPU outside its cgroup's cpuset |
| `max_gap_ms` | ms | `max_gap_ms = 500` | a worker's longest scheduling gap exceeds the cap |
| `max_spread_pct` | percentage points | `max_spread_pct = 20.0` | max−min worker off-CPU% exceeds the cap |
| `max_throughput_cv` | coefficient of variation | `max_throughput_cv = 0.35` | per-worker throughput CV exceeds the cap |
| `min_work_rate` | work units / CPU-second | `min_work_rate = 1000.0` | a worker's work rate falls below the floor |
| `min_iteration_rate` | iterations / second | `min_iteration_rate = 50000.0` | a worker's wall-clock iteration rate falls below the floor |
| `max_migration_ratio` | migrations / iteration | `max_migration_ratio = 0.5` | a cgroup's migration ratio exceeds the cap |
| `max_p99_wake_latency_ns` | ns | `max_p99_wake_latency_ns = 2000000` | p99 wake latency exceeds the cap |
| `max_wake_latency_cv` | coefficient of variation | `max_wake_latency_cv = 1.0` | wake-latency CV exceeds the cap |
| `min_page_locality` | fraction 0.0–1.0 | `min_page_locality = 0.9` | fraction of pages on expected NUMA nodes falls below the floor |
| `max_cross_node_migration_ratio` | fraction 0.0–1.0 | `max_cross_node_migration_ratio = 0.1` | NUMA-migrated pages / total pages exceeds the cap |
| `max_slow_tier_ratio` | fraction 0.0–1.0 | `max_slow_tier_ratio = 0.05` | pages on memory-only (CXL) nodes exceed the cap |

Monitor thresholds — evaluated from the host monitor's samples of
guest scheduler state:

| Attribute | Unit | Example | Fails when |
|---|---|---|---|
| `max_imbalance_ratio` | ratio | `max_imbalance_ratio = 2.0` | observed run-queue imbalance exceeds the cap |
| `max_local_dsq_depth` | tasks | `max_local_dsq_depth = 8` | a local DSQ grows deeper than the cap |
| `fail_on_stall` | bool | `fail_on_stall` | the monitor's stuck-task detection fails the test instead of reporting |
| `sustained_samples` | samples | `sustained_samples = 3` | window size a violation must persist for before it counts |
| `max_fallback_rate` | events/s | `max_fallback_rate = 5.0` | fallback-dispatch event rate exceeds the cap |
| `max_keep_last_rate` | events/s | `max_keep_last_rate = 100.0` | keep-last event rate exceeds the cap |

### What a failing gate looks like

A deliberately unreachable floor, to show the output shape — 50M
iterations/s is far beyond what two workers deliver:

```rust,ignore
declare_scheduler!(MY_SCHED, { name = "my_sched", binary = "scx-ktstr" });

#[ktstr_test(scheduler = MY_SCHED, llcs = 1, cores = 2, threads = 1, duration_s = 5)]
fn throughput_gate(ctx: &Ctx) -> Result<AssertResult> {
    let checks = Assert::default_checks().min_iteration_rate(50_000_000.0);
    let steps = vec![Step {
        setup: vec![ctx.cgroup_def("cg_a"), ctx.cgroup_def("cg_b")].into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    execute_steps_with(ctx, steps, Some(&checks))
}
```

<!-- captured: cargo ktstr test --kernel 7.0 -- -E 'test(throughput_gate)' | ktstr 0.23.0 | kernel 7.0.14 -->
<div class="kt-term"><div class="kt-term-bar"><span class="kt-term-title">cargo ktstr test --kernel 7.0 -- -E 'test(throughput_gate)'</span></div>

<pre>  TRY 1 FAIL [  31.810s] (───) ktstr::docs_demo ktstr/throughput_gate
  stderr ───
...
    <span class="t-b">ktstr_test 'throughput_gate' [sched=scx-ktstr] [topo=1n1l2c1t] failed:</span>
      <span class="t-red">worker 71 iteration rate 41903.3/s below floor 50000000.0/s</span>
      <span class="t-red">worker 73 iteration rate 37834.5/s below floor 50000000.0/s</span>

    --- stats ---
    2 workers, 4 cpus, 2 migrations, worst_spread=0.0%, worst_gap=21ms
      cg0: workers=1 cpus=2 spread=0.0% gap=10ms migrations=1 iter=209600
      cg1: workers=1 cpus=2 spread=0.0% gap=21ms migrations=1 iter=189252
...
    --- monitor ---
    samples=41 max_imbalance=2.00 max_dsq_depth=0 stuck=0
    avg: imbalance=1.32 nr_running/cpu=1.2 dsq/cpu=0.0
    events: fallback=0 (0.0/s) keep_last=210 (52.5/s) offline=0
...
    <span class="t-dim">verdict: monitor OK</span></pre></div>

The failure names the worker, the measured value, and the threshold
it crossed; the stats and monitor sections that follow are the
context for deciding whether the threshold or the scheduler is
wrong. [Reading Failure Output](../running-tests/failures.md) walks
the full transcript.

### Expected-error matchers

Two attributes narrow which failure counts as the expected bug in an
`expect_err = true` reproducer test (both require `expect_err`; both
may be set, composing with AND semantics):

- `expect_scx_bpf_error_contains = "literal"` — the captured
  `scx_bpf_error` text must contain the literal substring. Empty
  strings panic at construction.
- `expect_scx_bpf_error_matches = "regex"` — the text must match the
  regex. Empty patterns, invalid syntax, and any pattern that
  matches the empty string (`a?`, `.*`, `^$`) panic at construction,
  so a vacuous matcher can never silently pass. `^`/`$` anchor to
  the whole string by default (use `(?m)` for line anchors), and a
  bare `\b` slips the vacuity gate — prefer a substring.

See [Investigate a Crash](../recipes/investigate-crash.md) for the
pin-an-error-as-a-regression-test workflow these serve.

## Topology constraints {#topology-constraints}

These filter which gauntlet presets a test expands into; the base
`ktstr/` variant is unaffected.

| Attribute | Default | Description |
|---|---|---|
| `min_llcs` / `max_llcs` | 1 / 12 | LLC-count bounds |
| `min_numa_nodes` / `max_numa_nodes` | 1 / 1 | NUMA-node bounds — multi-NUMA presets are opt-in |
| `min_cpus` / `max_cpus` | 1 / 192 | Total-CPU bounds |
| `requires_smt` | `false` | Only SMT (threads > 1) presets; skips the test entirely on aarch64, which ships no SMT presets |

The gauntlet skips presets that fail any bound. See
[Gauntlet](../running-tests/gauntlet.md) for the preset table,
filtering rules, and a worked expansion.

## Execution attributes {#execution}

### `auto_repro` / `expect_auto_repro`

On scheduler crash, `auto_repro` (default `true`) boots a second VM
with probes attached to capture the state on the way to the crash —
see [Auto-Repro](../running-tests/auto-repro.md). Set
`auto_repro = false` for faster iteration; `expect_err = true` also
disables it. `expect_auto_repro = true` inverts the assertion: the
test fails unless the auto-repro path actually fired (used to pin
the repro machinery itself). It requires a scheduler and `wprof`,
and is rejected alongside `auto_repro = false`, `expect_err`, or
`host_only`.

### `expect_err` / `survives_storm` / `allow_inconclusive`

`expect_err = true` asserts the run returns `Err` — the negative
test: a scheduler crash or scenario failure is the expected outcome,
and a clean pass fails the test. `survives_storm = true` is the
positive inverse: the scx scheduler must stay attached and alive
through every hold; a death or ejection fails with a
survival-specific explainer. It requires a scheduler and is mutually
exclusive with `expect_err` and `expect_auto_repro`.
`allow_inconclusive = true` lets an Inconclusive verdict pass
instead of exiting 2 — see
[Checking](../concepts/checking.md) for when Inconclusive arises.

### `performance_mode` / `no_perf_mode` / `cpu_budget`

`performance_mode = true` pins vCPUs to reserved host cores with
hugepages, NUMA binding, and RT scheduling, for runs whose numbers
must be comparable — see
[Performance Mode](../concepts/performance-mode.md).
`no_perf_mode = true` goes the other way: build the VM with the
declared topology even on a smaller host, skipping all pinning.
The two are mutually exclusive (rejected at compile time).
`cpu_budget = N` overrides the auto-derived host-CPU mask size in
no-perf mode (must be > 0, requires `no_perf_mode`; an explicit
`--cpu-cap` / `KTSTR_CPU_CAP` still wins) — see
[Resource Budget](../concepts/resource-budget.md).

### `kaslr`

Default `true`: the guest boots with KASLR enabled, so tests run
against the memory layout real systems have. `kaslr = false`
appends `nokaslr` to the guest command line for the rare workflow
that needs stable kernel addresses.

### `host_only`

Run the function directly on the host — no VM. For tests that need
host tools (cargo, nested VMs) unavailable in the guest initramfs.
Mutually exclusive with `scheduler`, `num_snapshots > 0`,
`auto_repro = true`, `disk`, and `networks` — everything that
requires a VM to exist.

### `ignore`

Emits `#[ignore]` on the generated test, so it is skipped by default
and runs only under nextest's `--run-ignored`. Use for
slow-by-design tests that should not gate every local run.

### `pci` / `disk` / `networks`

`disk = CONST` attaches a virtio-blk device from a `const
DiskConfig` (built via `DiskConfig::DEFAULT` chained setters); the
framework owns the backing file. `networks = [CONST, …]` attaches
one virtio-net device per `const NetConfig` (aarch64 supports at
most one). On x86_64 either device auto-enables the virtio-PCI
transport; `pci = true` forces the PCI host bridge on without a
device (default: no bridge, `pci=off` on the guest command line).

### `bpf_map_write`

`bpf_map_write = CONST` (or `[A, B]`) writes a u32 into a named
scheduler BPF-map field from the host, once, after the scheduler
loads — pre-seeding guest state before workers start. See
[Snapshots — composing reads with writes](snapshots.md#composing-reads-with-writes).

### `watch_bpf_maps`

`watch_bpf_maps = CONST` (or `[A, B]`) samples named scheduler
BPF-map fields observer-effect-free during the run and surfaces each
as a run-level metric. Full semantics in
[Watching BPF-map fields](#watching-bpf-map-fields) below.

### `perf_delta_assertions`

`perf_delta_assertions = CONST` (or `[A, B]`) declares per-test
performance-regression gates. Inert in a normal `cargo ktstr test`
run — enforced only under `cargo ktstr perf-delta --noise-adjust`.
Requires `performance_mode` (rejected at compile time otherwise,
since unpinned numbers would make the gate misfire). See
[Assertable Metrics](../reference/assertable-metrics.md).

### `num_snapshots`

`num_snapshots = N` fires N periodic BPF-state captures inside the
workload's 10%–90% window. `0` (default) disables periodic capture.
Validated against the 64-capture bridge cap, `host_only`, and a
100 ms minimum boundary spacing. See
[Periodic Capture](periodic-capture.md).

### `post_vm` / `post_vm_unconditional`

Host-side callbacks (`fn(&VmResult) -> anyhow::Result<()>`) invoked
after the VM exits — the place to drain snapshot bridges, read run
metrics, and run [temporal assertions](temporal-assertions.md).
`post_vm` is suppressed when the guest reported failure;
`post_vm_unconditional` always runs (guard it with
`if !result.success { return Ok(()); }` when it reads state a crash
may not have produced, and note it never turns a guest failure into
a pass). When `num_snapshots > 0` and `post_vm` is omitted, the
macro installs a default callback asserting at least one periodic
capture landed with real BPF state.

### `cleanup_budget_ms`

Caps host-side VM teardown wall time; exceeding the budget folds a
failing detail into the test result. Unset disables the check; `0`
is rejected.

### `staged_schedulers`

`staged_schedulers = [PATH, …]` packs additional `&'static
Scheduler` binaries into the guest at boot. Required for scenarios
that invoke `Op::ReplaceScheduler` / `Op::AttachScheduler` — the
swap target must already be on disk in the guest. See
[Ops](../concepts/ops.md).

### `workload_root_cgroup`

`workload_root_cgroup = "/path"` places the per-test workload
cgroups under a specific guest cgroup path, decoupled from the
scheduler's `cgroup_parent` (which roots scheduler-side cells).

### `wprof` / `wprof_args`

`wprof = true` attaches the wprof BPF tracer to the workload VM;
`wprof_args = "..."` passes space-separated CLI args. Both require
the `wprof` cargo feature.

### `payload` / `workloads` / `extra_include_files` / `extra_sched_args`

`payload = CONST` declares the test's primary benchmark binary and
`workloads = [A, B]` composes more alongside it; the include-file
pipeline packs every referenced binary into the guest. See
[Payloads and Included Files](payloads.md).
`extra_include_files = ["path", …]` adds test-level host files that
belong to no particular payload. `extra_sched_args = ["--flag", …]`
appends scheduler CLI args after the scheduler's own `sched_args`.

### `config`

Inline scheduler config content, paired with a scheduler that
declares `config_file_def` — covered next.

## Inline scheduler config {#inline-scheduler-config}

Some schedulers (e.g. `scx_layered`, `scx_lavd`) accept a JSON
config file via a CLI argument like `--config /path/to/config.json`.
Two pieces wire this into a test:

1. **Scheduler declaration** — declares the arg template and the
   guest path via `config_file_def`:

   ```rust,ignore
   const LAYERED_SCHED: Scheduler = Scheduler::named("layered")
       .binary(SchedulerSpec::Discover("scx_layered"))
       .config_file_def("--config {file}", "/include-files/layered.json");
   ```

   `{file}` in the arg template is replaced with the guest path. The
   framework writes the config content to that path inside the guest
   before the scheduler binary starts.

2. **Test attribute** — supplies the inline content:

   ```rust,ignore
   const LAYERED_CONFIG: &str = r#"{ "layers": [...] }"#;

   #[ktstr_test(scheduler = LAYERED_SCHED, config = LAYERED_CONFIG)]
   fn layered_test(ctx: &Ctx) -> Result<AssertResult> {
       Ok(AssertResult::pass())
   }
   ```

   Both a string literal and a path to a `const &'static str` are
   accepted.

The pairing gate is bidirectional and enforced at compile time (and
again at runtime for programmatic entry construction): a scheduler
with `config_file_def` requires `config = …` on every test, and a
scheduler without it rejects `config = …` — the content would
otherwise be silently dropped.

For schedulers that take the same config file on every test, use
`Scheduler::config_file(host_path)` instead — see
[Scheduler Definitions](scheduler-definitions.md).

## Watching BPF-map fields {#watching-bpf-map-fields}

`watch_bpf_maps` turns "the scheduler computed X" into a post-VM
assertion. The free-running host monitor reads the named field from
the running guest's BPF-map memory via BTF — without freezing vCPUs
— and folds the samples into a run-level metric.

Each declared const is one `WatchBpfMap::new(map_name_suffix, field,
agg, label)`:

- `map_name_suffix` — matched against a loaded BPF map by
  `ends_with` (`".bss"` for a section global, or a named map like
  `"cpu_ctx_stor"`).
- `field` — a dot-path into the map's value type
  (`"sys_stat.avg_lat_cri"`, or a bare global like `"lat_headroom"`).
- `agg` — pick by the field's semantic class:
  - `BpfMapAgg::Scalar` — a gauge; folded as the mean over the
    run's samples.
  - `BpfMapAgg::ScalarCounter` — a monotonic counter; folded as the
    value at the last sample (the final total, not a mean of a
    rising series).
  - `BpfMapAgg::PerCpu` — a per-CPU gauge array; folded into a
    cross-CPU mean and max.
  - `BpfMapAgg::PerCpuCounter` — a per-CPU counter array; folded as
    the cross-CPU sum at the last sample. Watch at u64 width so no
    per-CPU slot truncates before the sum.
- `label` — the metric-key leaf; must be unique within a test.

The metric key is `<scheduler-obj>_<label>` (per-CPU gauges get
`_avg` / `_max` variants). The prefix is libbpf's object name from
the scheduler's global-section map, which can differ from the ops
name — scx-ktstr's object is `bpf_bpf`, so its prefix is `bpf_bpf`.
Read the metric back with `VmResult::run_metric` in a `post_vm`
hook; an absent metric returns `None`, never a false `0.0`.

```rust,ignore
const AVG_LAT_CRI: WatchBpfMap =
    WatchBpfMap::new(".bss", "sys_stat.avg_lat_cri", BpfMapAgg::Scalar, "avg_lat_cri");
const LAT_HEADROOM: WatchBpfMap =
    WatchBpfMap::new("cpu_ctx_stor", "lat_headroom", BpfMapAgg::PerCpu, "lat_headroom");

fn check(result: &VmResult) -> anyhow::Result<()> {
    let avg_lat_cri = result.run_metric("scx_lavd_avg_lat_cri")
        .ok_or_else(|| anyhow::anyhow!("avg_lat_cri absent"))?;
    let headroom_max = result.run_metric("scx_lavd_lat_headroom_max")
        .ok_or_else(|| anyhow::anyhow!("lat_headroom_max absent"))?;
    anyhow::ensure!(avg_lat_cri.is_finite() && headroom_max.is_finite());
    Ok(())
}

#[ktstr_test(
    scheduler = SCX_LAVD,
    watch_bpf_maps = [AVG_LAT_CRI, LAT_HEADROOM],
    post_vm = check,
)]
fn lat_metrics_surface(ctx: &Ctx) -> anyhow::Result<AssertResult> { /* workload */ }
```

Resolution is lazy: the maps appear only after the scheduler
attaches, so the monitor retries until the named map is present,
then caches the resolved offset and re-reads only the leaf bytes
each tick.

## What the macro generates

The macro renames the function, registers it in the `KTSTR_TESTS`
distributed slice, and emits a `#[test]` wrapper that boots the VM
and dispatches. Details in the
[attribute rustdoc](https://ktstr.dev/rustdoc/ktstr/attr.ktstr_test.html).
