# Writing Tests

Tests are Rust functions annotated with `#[ktstr_test]`. Each test
boots a KVM VM, runs the scenario inside it, and evaluates results
on the host.

<div class="kt-doc-grid">
<div class="kt-doc-card"><strong><a href="writing-tests/ktstr-test-macro.html">Attribute shape</a></strong><p>Topology, timing, checks, snapshots, scheduler selection, and execution knobs.</p></div>
<div class="kt-doc-card"><strong><a href="writing-tests/scheduler-definitions.html">Scheduler setup</a></strong><p>Declare an <code>scx_*</code> binary once, then reuse it across tests and verifier sweeps.</p></div>
<div class="kt-doc-card"><strong><a href="concepts/checking.html">Assertions</a></strong><p>Checks are opt-in: choose worker progress, spread, stuck gaps, throughput, and temporal gates deliberately.</p></div>
</div>

```rust,ignore
use ktstr::prelude::*;

#[ktstr_test(llcs = 1, cores = 2, threads = 1)]
fn my_test(ctx: &Ctx) -> Result<AssertResult> {
    execute_defs(ctx, vec![
        ctx.cgroup_def("cg_0"),
        ctx.cgroup_def("cg_1"),
    ])
}
```

`ctx.cgroup_def("name")` is shorthand for
`CgroupDef::named("name").workers(ctx.workers_per_cgroup)` — the
common case. Use `CgroupDef::named(...).workers(N).work_type(...)`
directly when the test needs to customize worker count or work type.

Run with `cargo ktstr test --kernel ../linux` or a released kernel
version such as `7.0` (see [Getting Started](getting-started.md) for
setup). A passing run is one nextest line per test; the VM boot,
scenario, and teardown all happen inside the reported duration —
[Getting Started](getting-started.md#run-it) shows a full transcript.

Every test gets the same machinery for free: a fresh VM per test (no
state shared between tests), a failure dump with BTF-rendered
scheduler BPF state if the scheduler crashes (see
[Reading Failure Output](running-tests/failures.md)), and an
automatic second-VM reproduction run with probes attached
([Auto-Repro](running-tests/auto-repro.md)). Each test also expands
into gauntlet variants across topology presets — see
[Gauntlet](running-tests/gauntlet.md).

> [!WARNING]
> No worker checks run by default. The example above passes as long
> as nothing crashes — it does not assert worker progress, fairness,
> or stuck gaps. Opt in with `not_stuck = true` and the threshold
> attributes; see [Checking](concepts/checking.md) for the model.

## Scenarios {#scenarios}

A scenario is the scheduling condition a test creates — which cgroups
exist, which CPUs they may use, what their workers do, and what
changes mid-run. The canned scenarios in `scenarios::*` exist so those
conditions have names: `scenarios::steady(ctx)` produces the same
reproducible condition against every scheduler you point it at, which
is what makes results comparable across schedulers and commits.

<div class="kt-doc-grid">
<div class="kt-doc-card"><strong>Start simple</strong><p><code>steady</code> proves the scheduler survives ordinary balanced load.</p></div>
<div class="kt-doc-card"><strong>Add movement</strong><p>Cpuset, affinity, and cgroup-change scenarios exercise dynamic callbacks.</p></div>
<div class="kt-doc-card"><strong>Mix signals</strong><p>Specialized scenarios combine workload types when one primitive is too narrow.</p></div>
</div>

Scenarios graduate in three stages; move down only when the stage
above can't express the condition:

1. **Canned** — call a `scenarios::*` function. Zero setup, named,
   comparable.

   ```rust,ignore
   // Baseline: run the canned steady scenario under EEVDF.
   #[ktstr_test(llcs = 1, cores = 2, threads = 1)]
   fn baseline(ctx: &Ctx) -> Result<AssertResult> {
       scenarios::steady(ctx)
   }

   // Scheduler smoke test: crash/watchdog failures plus worker-progress checks.
   #[ktstr_test(scheduler = MY_SCHED, llcs = 2, cores = 2, threads = 1, not_stuck = true)]
   fn scheduler_smoke(ctx: &Ctx) -> Result<AssertResult> {
       scenarios::steady(ctx)
   }
   ```

2. **Your own cgroup layout** — `execute_defs(ctx, vec![...])` with
   `CgroupDef`s you declare: your worker counts, work types, cpusets,
   still one static phase.

   ```rust,ignore
   // Two cgroups pinned to different LLCs.
   #[ktstr_test(scheduler = MY_SCHED, llcs = 2, cores = 2, threads = 1)]
   fn split_llc(ctx: &Ctx) -> Result<AssertResult> {
       execute_defs(ctx, vec![
           CgroupDef::named("left").cpuset(CpusetSpec::Llc(0)),
           CgroupDef::named("right").cpuset(CpusetSpec::Llc(1)),
       ])
   }
   ```

3. **Steps** — `execute_steps` / `execute_scenario` with `Step`s and
   `Op`s for anything that changes mid-run (cpuset swaps, scheduler
   replacement, snapshots, kernel-memory reads), plus a
   [`Backdrop`](concepts/ops.md#backdrop) for state that must outlive
   the steps.

[Ops, Steps, and Backdrop](concepts/ops.md) documents stage 2 and 3 —
the `CgroupDef` builder, every `Op`, and all the `execute_*` entry
points. A custom scenario is just the `#[ktstr_test]` function body
itself; [Custom Scenarios](writing-tests/custom-scenarios.md) covers
writing bodies that go beyond `Step`s entirely. Start with the
smallest shape that can fail for the behavior you care about — add
topology, workers, snapshots, and temporal assertions only when they
give you a new signal; otherwise they just make the run slower and the
failure harder to read.

### The canned catalog

| Function | Condition tested | Setup |
|---|---|---|
| `steady` | Baseline fairness | 2 cgroups, no cpusets, equal CPU-spin load |
| `steady_llc` | LLC-boundary scheduling | 2 cgroups on different LLCs (skips on 1-LLC topologies) |
| `oversubscribed` | Dispatch under oversubscription | 2 cgroups, 32 mixed workers each |
| `cpuset_apply` | Cpuset assignment on running tasks | Disjoint cpusets applied mid-run |
| `cpuset_clear` | Cpuset removal on confined tasks | Cpusets cleared mid-run |
| `cpuset_resize` | Cpuset resizing adaptation | Cpusets shrink then grow |
| `cgroup_add` | Scheduler reaction to a new cgroup | Cgroups created while others run |
| `cgroup_remove` | Scheduler reaction to cgroup removal | Cgroups torn down while others run |
| `affinity_change` | Affinity mask changes | Worker affinities randomized mid-run |
| `affinity_pinned` | Narrow-affinity contention | Workers pinned to a 2-CPU subset |
| `host_contention` | Cgroup vs host-task fairness | Root-cgroup workers beside managed cgroups |
| `mixed_workloads` | Mixed workload fairness | Heavy + bursty + IO cgroups |
| `nested_steady` | Nested cgroup hierarchy | Workers in nested sub-cgroups |
| `nested_task_move` | Cross-level task migration | Tasks moved between nested cgroups |

More specialized `custom_*` functions live in the
`ktstr::scenario::{affinity, basic, cpuset, dynamic, interaction,
nested, performance, stress}` modules — see the
[API docs](https://ktstr.dev/rustdoc/ktstr/scenario/index.html).

Against a new scheduler, run `steady` first — it is the smallest
condition that can fail (two cgroups, spin load, nothing dynamic).
Then `steady_llc` on a 2-LLC topology to see cache-boundary
placement, then `mixed_workloads` and `oversubscribed` for load
diversity. The dynamic scenarios (`cpuset_*`, `cgroup_*`,
`affinity_*`) each isolate one reconfiguration path; reach for the
one matching the code you changed.

### How a scenario runs

A scenario body does not pick its own duration or topology — the
`#[ktstr_test]` attribute does. The workload runs for the test's
`duration_s` (see the
[macro reference](writing-tests/ktstr-test-macro.md)), on the
topology the attribute declares, and a
[gauntlet](running-tests/gauntlet.md) run re-executes the same body
across a whole topology matrix. Worker counts and cpusets come from
the scenario's own `CgroupDef`s.

Every scenario ends the same way: worker reports are collected and
the opted-in [checks](concepts/checking.md) run against them. A run's
stats roll-up looks like:

<!-- captured: cargo ktstr test --kernel 7.0 (throughput_gate demo test) | ktstr 0.23.0 | kernel 7.0.14 -->
```text
--- stats ---
2 workers, 4 cpus, 2 migrations, worst_spread=0.0%, worst_gap=21ms
  cg0: workers=1 cpus=2 spread=0.0% gap=10ms migrations=1 iter=209600
  cg1: workers=1 cpus=2 spread=0.0% gap=21ms migrations=1 iter=189252
```

[Reading Failure Output](running-tests/failures.md) walks the full
anatomy.

## Where to go next

- [The #\[ktstr_test\] Attribute](writing-tests/ktstr-test-macro.md)
  — the full attribute reference: topology, timing, checking
  thresholds, execution knobs.
- [Scheduler Definitions](writing-tests/scheduler-definitions.md) —
  `declare_scheduler!`: how the scheduler under test is named,
  found, configured, and launched.
- [Payloads and Included Files](writing-tests/payloads.md) — run
  benchmark binaries (`schbench`, `fio`, …) alongside workers and
  extract their metrics.
- [Custom Scenarios](writing-tests/custom-scenarios.md) — scenario
  logic the ops system cannot express, written directly in the test
  body.
- [Snapshots and Live Capture](writing-tests/snapshots.md) — freeze
  and read scheduler BPF state on demand, when the kernel writes a
  chosen symbol, or on a cadence across the workload window.
- [Temporal Assertions](writing-tests/temporal-assertions.md) —
  assert on trajectories: counters that only advance, metrics that
  hold steady, systems that converge.
