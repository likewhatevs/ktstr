# Zero to ktstr

This tutorial walks through writing a complete `#[ktstr_test]` from
scratch. By the end you'll have a scheduler test that runs two
cgroups with different lifecycle patterns across a multi-LLC
topology, asserts fairness, throughput parity, and cpuset isolation
— and you'll have broken it on purpose once, so real failures look
familiar.

> **Already have a scheduler binary?** This tutorial teaches ktstr
> from the ground up. If you have an existing `scx_X` you want to
> test, jump to one of the targeted recipes instead:
> [Test a New Scheduler](recipes/test-new-scheduler.md) (5 minutes,
> validates basic behavior), [A/B Compare Branches](recipes/ab-compare.md)
> (compare two scheduler builds), or
> [Diagnose a Slow Scheduler](recipes/diagnose-slow-scheduler.md)
> (debug performance regressions).

## What you'll build

A test named `mixed_workloads` that:

<div class="kt-doc-grid">
<div class="kt-doc-card"><strong>Two cgroups</strong><p><code>background_spinner</code> runs continuously; <code>phased_worker</code> loops through <code>Spin -> Yield</code> phases.</p></div>
<div class="kt-doc-card"><strong>Real topology</strong><p>A 2-LLC, 4-core VM gives the scheduler a cache boundary to respect.</p></div>
<div class="kt-doc-card"><strong>Explicit checks</strong><p>Fairness, throughput parity, cpuset isolation, and one deliberate failure.</p></div>
<div class="kt-doc-card"><strong>Scheduler state</strong><p>A post-workload snapshot captures BPF state for inspection and assertions.</p></div>
</div>

The complete test is at the [end of this page](#the-complete-test).

## Prerequisites

[Getting Started](getting-started.md) covers the toolchain, KVM
access, the dev-dependency, and building a bootable kernel
([Build a kernel](getting-started.md#build-a-kernel)). With those in
place, create a file under your crate's `tests/` directory (e.g.
`tests/mixed_workloads.rs`) and follow along.

## Step 1: The skeleton

Every `#[ktstr_test]` is a Rust function that takes `&Ctx` and
returns `Result<AssertResult>`. Start with an empty body that passes
unconditionally:

```rust,ignore
use ktstr::prelude::*;

#[ktstr_test(llcs = 1, cores = 2, threads = 1)]
fn mixed_workloads(ctx: &Ctx) -> Result<AssertResult> {
    let _ = ctx;
    Ok(AssertResult::pass())
}
```

`let _ = ctx;` keeps the unused-variable lint quiet at the skeleton
stage; Step 2 onward uses `ctx`.

> **Try it.** Once this file compiles, run just this test with
> `cargo ktstr test --kernel 7.0 -- -E 'test(mixed_workloads)'`.
> A bare-skeleton test passes immediately — the rest of the tutorial
> adds the workload and assertions on top.

`use ktstr::prelude::*;` brings in every type the test body needs —
`Ctx`, `AssertResult`, `CgroupDef`, `WorkType`, `CpusetSpec`,
`execute_defs`, and the `Result` alias from `anyhow`. The
`#[ktstr_test]` attribute registers the function so `cargo ktstr
test` discovers it and boots a VM with the requested topology.

A test without a `scheduler = …` attribute runs under the kernel's
default EEVDF scheduler — a useful baseline (see
[Overview](overview.md)). Step 2 swaps in a sched_ext scheduler so
the rest of the tutorial exercises that scheduler instead.

For the full attribute reference, see
[The #\[ktstr_test\] Attribute](writing-tests/ktstr-test-macro.md).

## Step 2: Define your scheduler

To target a sched_ext scheduler, declare it with
`declare_scheduler!` and reference the generated const from
`#[ktstr_test(scheduler = …)]`. The example uses `scx-ktstr`, the
test-fixture scheduler shipped in the ktstr workspace; substitute
your own binary name to target a different scheduler.

```rust,ignore
use ktstr::prelude::*;

declare_scheduler!(KTSTR_SCHED, {
    name = "ktstr_sched",
    binary = "scx-ktstr",
});

#[ktstr_test(scheduler = KTSTR_SCHED, llcs = 1, cores = 2, threads = 1)]
fn mixed_workloads(ctx: &Ctx) -> Result<AssertResult> {
    let _ = ctx;
    Ok(AssertResult::pass())
}
```

`declare_scheduler!` emits a `pub static KTSTR_SCHED: Scheduler` and
registers it so the [verifier sweep](running-tests/verifier.md)
discovers it automatically. The `scheduler =` slot expects the bare
const name. The fields used here:

- `name` — scheduler name for display and result files.
- `binary` — binary name, resolved on the host: a `KTSTR_SCHEDULER`
  override path, else built via `cargo build -p <name>` in the
  declaring crate's workspace. The resolved binary is packed into the
  VM's initramfs.

Other commonly used fields: `topology = (numa, llcs, cores,
threads)` sets a default VM topology that per-test attributes can
override; `sched_args = ["--flag"]` prepends CLI args to every test
using this scheduler; `kernels = [...]` lists kernel specs for the
verifier sweep; `verifier_exclude_topologies = ["preset"]` removes
named topology exceptions from that sweep only. For the full surface (`sysctls`, `kargs`,
`config_file`, gauntlet constraints, scheduler-level assertion
overrides) and the manual-builder path for programmatic composition,
see [Scheduler Definitions](writing-tests/scheduler-definitions.md).

## Step 3: Add workloads

A `CgroupDef` declares a cgroup along with the workers that will run
inside it. The builder methods configure worker count, the work each
worker performs, scheduling policy, and cpuset assignment.

Add two cgroups — both running tight CPU spinners for now. Step 5
will swap one of them for a phased workload:

```rust,ignore
use ktstr::prelude::*;

#[ktstr_test(scheduler = KTSTR_SCHED, llcs = 1, cores = 2, threads = 1)]
fn mixed_workloads(ctx: &Ctx) -> Result<AssertResult> {
    execute_defs(ctx, vec![
        CgroupDef::named("background_spinner")
            .workers(2)
            .work_type(WorkType::SpinWait),
        CgroupDef::named("phased_worker")
            .workers(2)
            .work_type(WorkType::SpinWait),
    ])
}
```

Without `.cpuset(...)`, a cgroup's workers run on every CPU in the
test's topology — they share the VM's full CPU set with all other
cgroups. `.cpuset(CpusetSpec::Llc(idx))` (introduced in Step 4)
restricts a cgroup to one LLC's CPUs.

`WorkType::SpinWait` runs a tight CPU spin loop; it is one of many
work primitives, each targeting a different kernel scheduling path —
see [Work Types](concepts/work-types.md) for the full set and how to
choose one.

`execute_defs` runs each cgroup concurrently for the test's full
duration. Use `execute_steps` when you need to add cgroups mid-run
or swap cpusets between phases — see
[Ops, Steps, and Backdrop](concepts/ops.md).

## Step 4: Set topology

The `#[ktstr_test]` attribute carries the VM's CPU topology.
Dimensions are big-to-little: `numa_nodes` (default 1), `llcs`
(total across all NUMA nodes), `cores` per LLC, and `threads` per
core. Total CPU count is `llcs * cores * threads`.

LLC count matters because the last-level cache is the primary
scheduling boundary — tasks sharing an LLC benefit from shared
cache lines, while cross-LLC migration carries a cold-cache penalty.
A scheduler that ignores LLC topology will look fine on `llcs = 1`
and start failing as soon as there is a real cache boundary to
respect.

Bump the topology to two LLCs with two cores each (4 CPUs total) so
each cgroup can own its own LLC:

```rust,ignore
#[ktstr_test(scheduler = KTSTR_SCHED, llcs = 2, cores = 2, threads = 1)]
fn mixed_workloads(ctx: &Ctx) -> Result<AssertResult> {
    execute_defs(ctx, vec![
        CgroupDef::named("background_spinner")
            .workers(2)
            .work_type(WorkType::SpinWait)
            .cpuset(CpusetSpec::Llc(0)),
        CgroupDef::named("phased_worker")
            .workers(2)
            .work_type(WorkType::SpinWait)
            .cpuset(CpusetSpec::Llc(1)),
    ])
}
```

`CpusetSpec::Llc(idx)` confines a cgroup to the CPUs that belong to
LLC `idx`. Other variants (`Numa`, `Range`, `Disjoint`, `Overlap`,
`Exact`) cover NUMA-node binding, fractional partitioning, and
hand-built CPU sets — see [Topology](concepts/topology.md).

## Step 5: Compose phased work inside a cgroup

So far both cgroups run identical CPU spinners. The point of this
test is to exercise a scheduler against **different lifecycle
patterns** at once, so swap `phased_worker` for a worker that loops
through explicit phases.

`WorkType::Sequence` runs each phase for its specified duration and
then advances to the next; when the last phase ends the loop
restarts. Phases: `WorkPhase::Spin(Duration)`,
`WorkPhase::Sleep(Duration)`, `WorkPhase::Yield(Duration)`,
`WorkPhase::Io(Duration)`, and `WorkPhase::AluHot { .. }`. Use the
`WorkType::sequence(first, rest)` constructor. Only
`std::time::Duration` needs an extra `use` line:

```rust,ignore
use std::time::Duration;
use ktstr::prelude::*;

#[ktstr_test(scheduler = KTSTR_SCHED, llcs = 2, cores = 2, threads = 1)]
fn mixed_workloads(ctx: &Ctx) -> Result<AssertResult> {
    execute_defs(ctx, vec![
        // Persistent CPU pressure on LLC 0 for the whole run.
        CgroupDef::named("background_spinner")
            .workers(2)
            .work_type(WorkType::SpinWait)
            .cpuset(CpusetSpec::Llc(0)),
        // Phased worker on LLC 1: spin 100 ms, yield for 20 ms,
        // then loop. Stresses the scheduler's wake-after-yield
        // placement repeatedly while the LLC-0 spinner keeps
        // runqueue pressure constant.
        CgroupDef::named("phased_worker")
            .workers(2)
            .work_type(WorkType::sequence(
                WorkPhase::Spin(Duration::from_millis(100)),
                [WorkPhase::Yield(Duration::from_millis(20))],
            ))
            .cpuset(CpusetSpec::Llc(1)),
    ])
}
```

The two cgroups now exercise distinct paths concurrently:
`background_spinner` keeps two CPUs continuously busy on LLC 0,
while `phased_worker` alternates between burning CPU and yielding on
LLC 1, exercising voluntary preemption and wakeup placement.

Both cgroups still run for the entire scenario duration: the phasing
happens *within* each `phased_worker` worker's loop. To express
phasing *across* cgroups (e.g. add `phased_worker` only for the
second half of the run), use `execute_steps` with multiple `Step`
entries — see [Ops, Steps, and Backdrop](concepts/ops.md).

## Step 6: Tune execution

Several `#[ktstr_test]` attributes control how the VM runs the
scenario. The defaults are tuned for fast iteration:

| Attribute | Default | What it does |
|---|---|---|
| `duration_s` | `12` | Per-scenario wall-clock seconds. Workers run for this long, then stop and report. |
| `watchdog_timeout_s` | `5` | sched_ext watchdog fire threshold. |
| `memory_mib` | `256` | Minimum VM memory in MiB; actual memory is raised for topology and initramfs size. |

`watchdog_timeout_s` is sched_ext's per-task stall threshold — if a
runnable task is not picked for that many seconds, the scheduler
exits with `SCX_EXIT_ERROR_STALL`. The scenario duration and
watchdog are independent; a 12 s scenario with a 5 s watchdog is
normal. Tune the watchdog only when the scheduler under test is
expected to legitimately leave a runnable task parked longer than
the default 5 s.

For the run we're building, set the duration to 20 s (so each phase
iteration repeats many times):

```rust,ignore
#[ktstr_test(
    scheduler = KTSTR_SCHED,
    llcs = 2,
    cores = 2,
    threads = 1,
    duration_s = 20,
)]
fn mixed_workloads(ctx: &Ctx) -> Result<AssertResult> {
    // body unchanged from Step 5 — two cgroups via execute_defs
}
```

For the full attribute reference (auto-repro, performance mode,
topology constraints, etc.), see
[The #\[ktstr_test\] Attribute](writing-tests/ktstr-test-macro.md).

## Step 7: Add assertions

Every check is opt-in — no threshold is compared until you turn its
check on, either at the scheduler level or on the per-test attribute
([Checking](concepts/checking.md) explains the model, and
[Customize Checking](recipes/custom-checking.md) the override
chain). The first check to opt into is `not_stuck = true`, which
enables three related worker-level checks together:

- **Zero work units** — any worker with no measured work fails the test.
- **Fairness spread** — per-cgroup `max(off-CPU%) - min(off-CPU%)`
  must stay under the spread threshold (release default 15%; debug
  default 35% — debug builds in small VMs show higher spread, so
  the threshold loosens automatically in debug builds).
- **Scheduling gaps** — the longest wall-clock gap observed at
  work-unit checkpoints must stay under the gap threshold (release
  default 2000 ms; debug default 3000 ms).

Cpuset isolation is separate — enable it with `isolation = true`.
Override the spread threshold and add throughput-parity gates:

```rust,ignore
use std::time::Duration;
use ktstr::prelude::*;

#[ktstr_test(
    scheduler = KTSTR_SCHED,
    llcs = 2,
    cores = 2,
    threads = 1,
    duration_s = 20,
    isolation = true,
    not_stuck = true,
    max_spread_pct = 20.0,
    max_throughput_cv = 0.5,
    min_work_rate = 1.0,
)]
fn mixed_workloads(ctx: &Ctx) -> Result<AssertResult> {
    execute_defs(ctx, vec![
        CgroupDef::named("background_spinner")
            .workers(2)
            .work_type(WorkType::SpinWait)
            .cpuset(CpusetSpec::Llc(0)),
        CgroupDef::named("phased_worker")
            .workers(2)
            .work_type(WorkType::sequence(
                WorkPhase::Spin(Duration::from_millis(100)),
                [WorkPhase::Yield(Duration::from_millis(20))],
            ))
            .cpuset(CpusetSpec::Llc(1)),
    ])
}
```

What each new attribute gates:

- `isolation = true` — workers must only run on CPUs in their
  assigned cpuset; any execution on an unexpected CPU fails the
  test.
- `not_stuck = true` — enables the zero-work, spread, and stuck-gap trio
  described above, at the default thresholds.
- `max_spread_pct = 20.0` — custom fairness threshold. It replaces
  the default-threshold spread verdict from `not_stuck` with your
  limit (and enables the spread check on its own even without
  `not_stuck`). 20.0 loosens the release default of 15.0 slightly
  to absorb noise from the phased worker's yield-driven
  re-placement.
- `max_throughput_cv = 0.5` — coefficient of variation of
  `work_units / cpu_time` across workers. Catches a scheduler that
  gives some workers disproportionately less effective CPU.
- `min_work_rate = 1.0` — minimum work units per CPU-second per
  worker. Catches the case where every worker is equally slow (CV
  passes but absolute throughput is too low).

Host-side monitor checks (imbalance ratio, DSQ depth, stuck-task
detection, event rates) also run on every test, but they are
report-only by default — [Checking](concepts/checking.md) covers
what they observe and how to make them enforce.

## Step 8: Run it

Run the test with `cargo ktstr test`, scoped to this one test name:

```sh
cargo ktstr test --kernel 7.0 -- -E 'test(mixed_workloads)'
```

`cargo ktstr test` resolves the kernel image, boots a VM with the
declared topology, runs the test as the guest's init, and reports
the result. A real passing run against a local kernel tree looks like
this (transcript captured from ktstr's own suite — your run shows
`ktstr/mixed_workloads` on the PASS line instead):

<!-- captured: cargo ktstr test --kernel ../linux --no-perf-mode -- --features integration -E 'test(=ktstr/scx_empty_run_exits_under_watchdog)' | ktstr 0.24.0-dev | kernel ../linux (b4dc42d2, sched_ext-for-7.2) | verified by independent rerun -->
<div class="kt-term"><div class="kt-term-bar"><span class="kt-term-title">cargo ktstr test --kernel ../linux --no-perf-mode -- -E 'test(=ktstr/scx_empty_run_exits_under_watchdog)'</span></div>

<pre><span class="t-dim">cargo ktstr: resolved kernel "path_linux_b5562c"
cargo ktstr: BTF type anchor at target/ktstr_btf_anchor.h</span>
...
────────────
 Nextest run ID d79da4e4-df49-4b40-a415-c1dff2739ce6 with nextest profile: default
    Starting 1 test across 120 binaries (13597 tests skipped)
        <span class="t-grn">PASS [   8.601s] (1/1) ktstr::scx_cleanup_test ktstr/scx_empty_run_exits_under_watchdog</span>
────────────
     <span class="t-b">Summary [   8.642s] 1 test run: 1 passed, 13597 skipped</span>

cargo ktstr: test outputs
...
    (1 stats sidecar(s), 0 wprof trace(s) written this run)</pre></div>

The VM portion took about nine seconds; Rust may add compile time
when the test binary changed. The `ktstr/` prefix on the test name
marks the base variant; see
[Running Tests](running-tests.md) for the name shapes and the
sidecar files each run writes.

If something goes wrong instead:

- **"kernel not found"** — the `--kernel` argument points at a
  directory without a built kernel, or at a version the cache
  cannot locate. Run `cargo ktstr kernel build` to populate the
  cache — see
  [Getting Started: Build a kernel](getting-started.md#build-a-kernel).
- **scheduler build failed** — the declared `binary = "..."` from
  Step 2 could not be built via `cargo build -p <name>`. Fix the
  scheduler build, or set `KTSTR_SCHEDULER=/path/to/binary` to pin an
  already-built binary.
- **probe-related errors** ("probe skeleton load failed", "trigger
  attach failed") — re-run with `RUST_LOG=ktstr=debug` to see the
  underlying libbpf reason; see
  [Troubleshooting](troubleshooting.md).

## Step 9: Break it on purpose

A green run tells you the harness works; it doesn't teach you to
read a failure. Crank one threshold to an impossible value and watch
what comes out. Add an iteration-rate floor no 2-core VM can meet:

```rust,ignore
#[ktstr_test(
    scheduler = KTSTR_SCHED,
    llcs = 2,
    cores = 2,
    threads = 1,
    duration_s = 20,
    isolation = true,
    not_stuck = true,
    max_spread_pct = 20.0,
    max_throughput_cv = 0.5,
    min_work_rate = 1.0,
    min_iteration_rate = 50_000_000.0,   // deliberately impossible
)]
```

Below is a real capture of exactly this experiment — a demo test
with the same impossible floor, on a 2-CPU topology:

<!-- captured: cargo ktstr test --kernel 7.0 -- --features integration -E 'test(=ktstr/throughput_gate)' | ktstr 0.23.0 | kernel 7.0.14 -->
```text
ktstr_test 'throughput_gate' [sched=scx-ktstr] [topo=1n1l2c1t] failed:
  worker 71 iteration rate 41903.3/s below floor 50000000.0/s
  worker 73 iteration rate 37834.5/s below floor 50000000.0/s

--- stats ---
2 workers, 4 cpus, 2 migrations, worst_spread=0.0%, worst_gap=21ms
  cg0: workers=1 cpus=2 spread=0.0% gap=10ms migrations=1 iter=209600
  cg1: workers=1 cpus=2 spread=0.0% gap=21ms migrations=1 iter=189252
...
--- monitor ---
samples=41 max_imbalance=2.00 max_dsq_depth=0 stuck=0
avg: imbalance=1.32 nr_running/cpu=1.2 dsq/cpu=0.0
events: fallback=0 (0.0/vcpu-s) keep_last=210 (52.5/vcpu-s) offline=0
...
verdict: monitor OK

...
cargo ktstr: test outputs
...
    FAILED  throughput_gate  [my_sched 1n1l2c1t]
      ...
      replay        cargo ktstr replay --filter throughput_gate --exec
```

How to read it:

- The header names the test, the scheduler, and the topology
  variant. Every detail line under it names the check that tripped,
  the observed value, and the threshold — here, workers managing
  ~40k iterations/s against a 50M floor.
- `--- stats ---` gives the per-cgroup roll-up: worker counts, CPUs
  touched, fairness spread, worst scheduling gap, migrations, and
  iteration totals.
- `verdict: monitor OK` is worth noticing: the host-side monitor saw
  nothing wrong. The scheduler behaved fine — the test's own gate
  was impossible. When a real scheduler bug trips a check, the
  monitor and timeline sections are usually where the story is.
- The footer hands you a ready-to-paste `cargo ktstr replay` line to
  re-run exactly the failing variant.

The full failure anatomy — timeline, scheduler log, auto-repro,
failure-dump artifacts — is covered in
[Reading Failure Output](running-tests/failures.md). Now delete the
`min_iteration_rate` line and the test goes green again.

## Step 10: Capture a snapshot

Threshold assertions tell you something is off; snapshots tell you
what the scheduler's state actually was.
`Op::capture_snapshot(name)` freezes every vCPU long enough to read
the scheduler's BPF map state, vCPU registers, and per-CPU counters
into a named report, then resumes the guest.

`execute_defs` (used so far) takes a flat list of cgroups. To inject
a snapshot, switch to `execute_steps`, which takes a list of `Step`s
— each with `setup` cgroups, an `ops` list, and a `hold` duration.

> [!WARNING]
> Within a step, ops fire *before* the `setup` cgroups are created.
> A single step with both the workload and a snapshot op named
> "after_workload" would capture an empty guest. Use two steps: a
> setup step that holds the workload, then a follow-up step whose op
> fires after the hold ends.

```rust,ignore
use std::time::Duration;
use ktstr::prelude::*;

#[ktstr_test(scheduler = KTSTR_SCHED, llcs = 2, cores = 2, threads = 1, duration_s = 20)]
fn mixed_workloads(ctx: &Ctx) -> Result<AssertResult> {
    execute_steps(ctx, vec![
        Step {
            setup: Setup::Defs(vec![
                CgroupDef::named("background_spinner")
                    .workers(2)
                    .work_type(WorkType::SpinWait)
                    .cpuset(CpusetSpec::Llc(0)),
                CgroupDef::named("phased_worker")
                    .workers(2)
                    .work_type(WorkType::sequence(
                        WorkPhase::Spin(Duration::from_millis(100)),
                        [WorkPhase::Yield(Duration::from_millis(20))],
                    ))
                    .cpuset(CpusetSpec::Llc(1)),
            ]),
            ops: vec![],
            hold: HoldSpec::FULL,
        },
        Step {
            setup: Setup::Defs(Vec::new()),
            ops: vec![Op::capture_snapshot("after_workload")],
            hold: HoldSpec::Fixed(Duration::ZERO),
        },
    ])
}
```

The first step creates the cgroups and holds them for the full
scenario duration; the second step's op runs after that hold
finishes, so the snapshot reflects the post-workload guest state.
Downstream code reads the captured report by name and walks fields
with a dotted-path accessor — e.g.
`snap.var("nr_dispatched").as_u64()?` reads a scheduler global. For
the traversal API, error handling, and the write-driven
`Op::watch_snapshot` variant, see
[Snapshots](writing-tests/snapshots.md).

## The complete test

The shape exercised by every step above, in one file — the Step 7
assertions plus the Step 10 snapshot steps:

```rust,ignore
use std::time::Duration;
use ktstr::prelude::*;

declare_scheduler!(KTSTR_SCHED, {
    name = "ktstr_sched",
    binary = "scx-ktstr",
});

#[ktstr_test(
    scheduler = KTSTR_SCHED,
    llcs = 2,
    cores = 2,
    threads = 1,
    duration_s = 20,
    isolation = true,
    not_stuck = true,
    max_spread_pct = 20.0,
    max_throughput_cv = 0.5,
    min_work_rate = 1.0,
)]
fn mixed_workloads(ctx: &Ctx) -> Result<AssertResult> {
    execute_steps(ctx, vec![
        Step {
            setup: Setup::Defs(vec![
                CgroupDef::named("background_spinner")
                    .workers(2)
                    .work_type(WorkType::SpinWait)
                    .cpuset(CpusetSpec::Llc(0)),
                CgroupDef::named("phased_worker")
                    .workers(2)
                    .work_type(WorkType::sequence(
                        WorkPhase::Spin(Duration::from_millis(100)),
                        [WorkPhase::Yield(Duration::from_millis(20))],
                    ))
                    .cpuset(CpusetSpec::Llc(1)),
            ]),
            ops: vec![],
            hold: HoldSpec::FULL,
        },
        Step {
            setup: Setup::Defs(Vec::new()),
            ops: vec![Op::capture_snapshot("after_workload")],
            hold: HoldSpec::Fixed(Duration::ZERO),
        },
    ])
}
```

Run it:

```sh
cargo ktstr test --kernel 7.0 -- -E 'test(mixed_workloads)'
```

## Going further

Each of these builds directly on the test you just wrote.

- **Gauntlet.** `#[ktstr_test]` doesn't emit just one test — it also
  generates variants that run the same body across every accepted
  topology preset (`gauntlet/mixed_workloads/8cpu-2llc-smt`, …), catching
  the bugs only odd LLC counts, SMT siblings, or NUMA crossings
  expose. See [Gauntlet](running-tests/gauntlet.md).
- **Worker identity.** `.comm("name")`, `.nice(n)`, and
  `.pcomm("name")` on `CgroupDef` give workers realistic names and
  priorities for schedulers that key on `task->comm` or nice values.
  See [Work Types](concepts/work-types.md).
- **Inline scheduler config.** Schedulers like `scx_layered` take a
  JSON config file; `config_file_def` on the scheduler plus
  `config = …` on the test writes it into the guest. See
  [The #\[ktstr_test\] Attribute](writing-tests/ktstr-test-macro.md).
- **Periodic capture and temporal assertions.** `num_snapshots = N`
  captures BPF state at evenly spaced points across the run, and a
  `post_vm` callback asserts temporal patterns over the series
  (nondecreasing counters, bounded rates, convergence). See
  [Periodic capture](writing-tests/snapshots.md#periodic-capture) and
  [Projections and Temporal Assertions](writing-tests/temporal-assertions.md).
- **Performance mode.** For benchmark-grade runs, ktstr pins vCPUs
  to reserved host cores and strips host scheduling noise; for
  topologies your host can't mirror, `no_perf_mode = true` builds
  the virtual topology as declared. See
  [Performance Mode](concepts/performance-mode.md).
- **Stats and regression gates.** Every run writes machine-readable
  sidecars; `cargo ktstr stats` aggregates them and
  `cargo ktstr perf-delta` gates HEAD against a baseline. See
  [Runs and Regression Gates](running-tests/runs.md).
- **Custom scenarios.** When the declarative ops can't express your
  scenario, the test body is arbitrary Rust — resize cpusets based
  on observed telemetry, assert on migrations directly. See
  [Custom Scenarios](writing-tests/custom-scenarios.md) and
  [Ops, Steps, and Backdrop](concepts/ops.md).
