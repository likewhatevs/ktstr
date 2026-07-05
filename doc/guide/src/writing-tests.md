# Writing Tests

Tests are Rust functions annotated with `#[ktstr_test]`. Each test
boots a KVM VM, runs the scenario inside it, and evaluates results
on the host.

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

Run with `cargo ktstr test --kernel 7.0` (see
[Getting Started](getting-started.md) for setup). A passing run is
one nextest line per test; the VM boot, scenario, and teardown all
happen inside the reported duration:

<!-- captured: cargo ktstr test --kernel 7.0 -- -E 'test(failure_dump_renders_bss_fields)' | ktstr 0.23.0 | kernel 7.0.14 -->
```text
cargo ktstr: resolved kernel "7.0"
...
 Nextest run ID 24c18577-... with nextest profile: default
    Starting 1 test across 121 binaries (12531 tests skipped)
        PASS [  34.451s] (1/1) ktstr::failure_dump_e2e ktstr/failure_dump_renders_bss_fields
...
     Summary [  34.490s] 1 test run: 1 passed, 12531 skipped
```

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
> as nothing crashes — it does not assert fairness, starvation, or
> gaps. Opt in with `not_starved = true` and the threshold
> attributes; see [Checking](concepts/checking.md) for the model.

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
