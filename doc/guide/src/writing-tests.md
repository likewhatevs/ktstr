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

## Common shapes

```rust,ignore
// Baseline: run the canned steady scenario under EEVDF.
#[ktstr_test(llcs = 1, cores = 2, threads = 1)]
fn baseline(ctx: &Ctx) -> Result<AssertResult> {
    scenarios::steady(ctx)
}

// Scheduler smoke test: crash/watchdog failures plus worker-progress checks.
#[ktstr_test(
    scheduler = MY_SCHED,
    llcs = 2,
    cores = 2,
    threads = 1,
    not_stuck = true,
)]
fn scheduler_smoke(ctx: &Ctx) -> Result<AssertResult> {
    scenarios::steady(ctx)
}

// Custom shape: two cgroups pinned to different LLCs.
#[ktstr_test(scheduler = MY_SCHED, llcs = 2, cores = 2, threads = 1)]
fn split_llc(ctx: &Ctx) -> Result<AssertResult> {
    execute_defs(ctx, vec![
        CgroupDef::named("left").cpuset(CpusetSpec::Llc(0)),
        CgroupDef::named("right").cpuset(CpusetSpec::Llc(1)),
    ])
}
```

Start with the smallest shape that can fail for the behavior you care
about. Add topology, workers, snapshots, and temporal assertions when
they give you a new signal; otherwise they only make the run slower and
the failure harder to read.

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
- [Snapshots](writing-tests/snapshots.md) — capture scheduler BPF
  state on demand mid-scenario and assert on it.
- [Watch Snapshots](writing-tests/watch-snapshots.md) — capture at
  the exact instant the kernel writes a chosen symbol.
- [Periodic Capture](writing-tests/periodic-capture.md) — cadenced
  BPF-state sampling across the workload window, no scenario code
  required.
- [Temporal Assertions](writing-tests/temporal-assertions.md) —
  assert on trajectories: counters that only advance, metrics that
  hold steady, systems that converge.
