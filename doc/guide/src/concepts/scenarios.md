# Scenarios

A scenario is the scheduling condition a test creates — which cgroups
exist, which CPUs they may use, what their workers do, and what
changes mid-run. The canned scenarios in `scenarios::*` exist so those
conditions have names: `scenarios::steady(ctx)` produces the same
reproducible condition against every scheduler you point it at, which
is what makes results comparable across schedulers and commits.

```rust,ignore
use ktstr::prelude::*;

#[ktstr_test(llcs = 1, cores = 2, threads = 1)]
fn my_test(ctx: &Ctx) -> Result<AssertResult> {
    scenarios::steady(ctx)
}
```

## Canned scenarios (`scenarios::*`)

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

## Start here

Against a new scheduler, run `steady` first — it is the smallest
condition that can fail (two cgroups, spin load, nothing dynamic).
Then `steady_llc` on a 2-LLC topology to see cache-boundary
placement, then `mixed_workloads` and `oversubscribed` for load
diversity. The dynamic scenarios (`cpuset_*`, `cgroup_*`,
`affinity_*`) each isolate one reconfiguration path; reach for the
one matching the code you changed.

## Run parameters

A scenario body does not pick its own duration or topology — the
`#[ktstr_test]` attribute does. The workload runs for the test's
`duration_s` (see the
[macro reference](../writing-tests/ktstr-test-macro.md)), on the
topology the attribute declares, and a
[gauntlet](../running-tests/gauntlet.md) run re-executes the same
body across a whole topology matrix. Worker counts and cpusets come
from the scenario's own `CgroupDef`s.

Every scenario ends the same way: worker reports are collected and
the opted-in [checks](checking.md) run against them. A run's stats
roll-up looks like:

<!-- captured: cargo ktstr test --kernel 7.0 (throughput_gate demo test) | ktstr 0.23.0 | kernel 7.0.14 -->
```text
--- stats ---
2 workers, 4 cpus, 2 migrations, worst_spread=0.0%, worst_gap=21ms
  cg0: workers=1 cpus=2 spread=0.0% gap=10ms migrations=1 iter=209600
  cg1: workers=1 cpus=2 spread=0.0% gap=21ms migrations=1 iter=189252
```

[Reading Failure Output](../running-tests/failures.md) walks the full
anatomy.

## From canned to custom

Scenarios graduate in three stages; move down only when the stage
above can't express the condition:

1. **Canned** — call a `scenarios::*` function. Zero setup, named,
   comparable.
2. **Your own cgroup layout** — `execute_defs(ctx, vec![...])` with
   `CgroupDef`s you declare: your worker counts, work types, cpusets,
   still one static phase.
3. **Steps** — `execute_steps` / `execute_scenario` with `Step`s and
   `Op`s for anything that changes mid-run (cpuset swaps, scheduler
   replacement, snapshots, kernel-memory reads), plus a
   [`Backdrop`](ops.md#backdrop) for state that must outlive the
   steps.

[Ops, Steps, and Backdrop](ops.md) documents stage 2 and 3 — the
`CgroupDef` builder, every `Op`, and all the `execute_*` entry
points. A custom scenario is just the `#[ktstr_test]` function body
itself; [Custom Scenarios](../writing-tests/custom-scenarios.md)
covers writing bodies that go beyond `Step`s entirely.
