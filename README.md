# ktstr

[![CI](https://github.com/likewhatevs/ktstr/actions/workflows/ci.yml/badge.svg)](https://github.com/likewhatevs/ktstr/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/likewhatevs/ktstr/graph/badge.svg?token=E7GRAO2KZM)](https://codecov.io/gh/likewhatevs/ktstr)
[![crates.io](https://img.shields.io/crates/v/ktstr.svg)](https://crates.io/crates/ktstr)
[![tutorial](https://img.shields.io/badge/docs-tutorial-blue)](https://ktstr.dev/guide/tutorial.html)
[![api](https://img.shields.io/badge/docs-api-blue)](https://ktstr.dev/rustdoc/ktstr/)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)](https://github.com/likewhatevs/ktstr/issues)

Test Linux schedulers like code. Every test boots a real kernel in a
KVM micro-VM with the topology it declares — and ktstr watches what
your scheduler does from the host, without touching the guest.

Scheduler bugs hide in topology: the fairness regression that only
shows up on an odd LLC count, the starvation that needs SMT siblings,
the crash that wants a NUMA crossing. Testing a
[sched_ext](https://github.com/sched-ext/scx) scheduler against those
shapes has meant scrounging hardware and hand-running repro scripts.
ktstr turns it into `cargo test`: declare the topology on the test,
and the VM actually has it.

- **Topology as code** — NUMA nodes, LLCs, cores, SMT declared on the
  test attribute, realized in the guest. The same test builds the same
  machine on any x86_64 or aarch64 host.
- **Workloads and assertions as data** — cgroups, cpusets, workers,
  and checks (starvation, fairness, isolation, throughput floors)
  declared on the test; the framework runs and judges them.
- **[Gauntlet](https://ktstr.dev/guide/running-tests/gauntlet.html)** —
  one test declaration fans out across a matrix of topology presets,
  with budget-aware selection for CI. Multiple `--kernel` flags add
  the kernel as another axis.
- **[Verifier sweep](https://ktstr.dev/guide/running-tests/verifier.html)** —
  every declared scheduler load-tested against the real kernel
  verifier across topologies, with per-program instruction counts and
  cycle-collapsed rejection logs.
- **[Auto-repro](https://ktstr.dev/guide/running-tests/auto-repro.html)** —
  crashes rerun themselves in a probe VM that captures function
  arguments and struct state along the crash path.
- **[Regression gates](https://ktstr.dev/guide/running-tests/runs.html)** —
  per-run metric sidecars, `stats` aggregation, and
  `perf-delta --noise-adjust` gating for CI.

The full tour with real output: [ktstr in
Action](https://ktstr.dev/guide/features.html).

## Install

```toml
[dev-dependencies]
ktstr = "0.23.0"
```

```sh
cargo install --locked cargo-nextest
cargo install --locked ktstr
```

ktstr is pre-1.0: pin the exact patch version, and install the
matching CLI with `cargo install --locked --bin cargo-ktstr
ktstr@X.Y.Z`. Minor bumps may break the test-facing API.

**Requirements:** Linux (x86_64 or aarch64) with `/dev/kvm` access,
Rust via `rust-toolchain.toml`, clang, and the usual build tooling:

```sh
# Ubuntu/Debian
sudo apt install clang pkg-config make gcc autoconf autopoint flex bison gawk

# Fedora
sudo dnf install clang pkgconf make gcc autoconf gettext-devel flex bison gawk
```

[Getting Started](https://ktstr.dev/guide/getting-started.html) covers
kernel discovery, building a test kernel, and `/dev/kvm` permissions.

## Quick start

Tests are standard Rust integration tests in `tests/`. Declare
cgroups and workers as data — no scheduler required (the guest runs
EEVDF, the kernel's default):

```rust
use ktstr::prelude::*;

#[ktstr_test(llcs = 1, cores = 2, threads = 1)]
fn two_cgroups(ctx: &Ctx) -> Result<AssertResult> {
    execute_defs(ctx, vec![
        CgroupDef::named("cg_0").workers(2),
        CgroupDef::named("cg_1").workers(2),
    ])
}
```

To test your own sched_ext scheduler, declare it and reference it
from the test:

```rust
use ktstr::prelude::*;

declare_scheduler!(MY_SCHED, {
    name = "my_sched",
    binary = "scx_my_sched",
});

#[ktstr_test(scheduler = MY_SCHED, llcs = 2, cores = 4, threads = 1)]
fn sched_two_cgroups(ctx: &Ctx) -> Result<AssertResult> {
    execute_defs(ctx, vec![
        CgroupDef::named("cg_0").workers(2),
        CgroupDef::named("cg_1").workers(2),
    ])
}
```

Run against any kernel — a released version, a local source tree, or
a git URL:

```sh
cargo ktstr test --kernel 7.0
```

Each test boots its own VM, builds the declared cgroups and workers,
runs the workload, and applies the default checks (starvation,
scheduling gaps, fairness). A pass looks like a pass:

```text
    PASS [  34.451s] (1/1) ktstr::failure_dump_e2e ktstr/failure_dump_renders_bss_fields
```

A failure names the check, the worker, and the numbers:

```text
ktstr_test 'throughput_gate' [sched=scx-ktstr] [topo=1n1l2c1t] failed:
  worker 71 iteration rate 41903.3/s below floor 50000000.0/s
  worker 73 iteration rate 37834.5/s below floor 50000000.0/s

--- stats ---
2 workers, 4 cpus, 2 migrations, worst_spread=0.0%, worst_gap=21ms
```

— followed by a timeline, the kernel's sched_ext dump, the scheduler
log, and (for crashes) the auto-repro probe report. See [Reading
Failure Output](https://ktstr.dev/guide/running-tests/failures.html).

From here:

- binary workloads (`schbench`, `fio`, anything) via
  [payloads](https://ktstr.dev/guide/writing-tests/payloads.html);
- dynamic scenarios (cgroups created, resized, and destroyed
  mid-run) via
  [ops and steps](https://ktstr.dev/guide/concepts/ops.html);
- topology fan-out via
  [gauntlet](https://ktstr.dev/guide/running-tests/gauntlet.html);
- hosting ktstr tests in your scheduler's own crate via [Test a New
  Scheduler](https://ktstr.dev/guide/recipes/test-new-scheduler.html).

## Documentation

- **[Tutorial: Zero to ktstr](https://ktstr.dev/guide/tutorial.html)** —
  define a scheduler, write a test, break it, read the wreckage.
- **[Guide](https://ktstr.dev/guide/)** — getting started, writing and
  running tests, concepts, recipes, architecture.
- **[API docs](https://ktstr.dev/rustdoc/ktstr/)** — rustdoc for all
  workspace crates.

## Contributing

Pull requests welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the
workflow and how to run the test suite locally.

## License

GPL-2.0-only
