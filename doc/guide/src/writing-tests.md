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

Run with `cargo ktstr test --kernel ../linux`. See
[Getting Started](getting-started.md) for setup and
[The #\[ktstr_test\] Macro](writing-tests/ktstr-test-macro.md) for all
available attributes. Each test also generates gauntlet variants across
topology presets and flag profiles. See
[Gauntlet Tests](writing-tests/gauntlet-tests.md).
For scenarios that need logic beyond what the ops system can express,
see [Custom Scenarios](writing-tests/custom-scenarios.md).
