# ktstr

[![CI](https://github.com/likewhatevs/ktstr/actions/workflows/ci.yml/badge.svg)](https://github.com/likewhatevs/ktstr/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/likewhatevs/ktstr/graph/badge.svg?token=E7GRAO2KZM)](https://codecov.io/gh/likewhatevs/ktstr)
[![crates.io](https://img.shields.io/crates/v/ktstr.svg)](https://crates.io/crates/ktstr)
[![tutorial](https://img.shields.io/badge/docs-tutorial-blue)](https://ktstr.dev/guide/tutorial.html)
[![api](https://img.shields.io/badge/docs-api-blue)](https://ktstr.dev/rustdoc/ktstr/)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)](https://github.com/likewhatevs/ktstr/issues)

> **Early stage.** APIs, CLI, and internals are actively evolving.
> Expect breaking changes between releases.

Test harness for Linux process schedulers, with a focus on
[sched_ext](https://github.com/sched-ext/scx).

## Why ktstr?

sched_ext lets you write Linux process schedulers as BPF programs.
A scheduler runs on every CPU and
affects every process -- bugs cause system-wide stalls or crashes.
Scheduler behavior depends on CPU topology, cgroup hierarchy, workload
mix, and kernel version. You cannot test this with unit tests because
the relevant state only exists inside a running kernel. ktstr also
tests under EEVDF (the kernel's built-in scheduler) as a baseline.

ktstr boots a real Linux kernel image under KVM (KASLR on by
default) and observes scheduler state by reading guest kernel
memory and BPF maps directly. The state your assertions see is
the same state `/proc` and `bpftool` would see if you logged
into the guest.

Without ktstr, testing means manually booting a VM, setting up cgroups,
running workloads, and eyeballing whether things went wrong -- with no
reproducibility across machines because topology varies per host. ktstr
automates this:

- **Clean slate** -- each test boots its own kernel in a KVM VM. No
  shared state between tests.
- **Topology as code** -- `topology = (1, 2, 4, 2)` gives you 1 NUMA
  node, 2 LLCs (last-level caches), 4 cores/LLC, 2 threads. The same
  test produces the same topology on any x86_64 or aarch64 host.
- **Declarative scenarios** -- tests declare cgroups, cpusets, and
  workloads as data (`CgroupDef`, `Step`, `Op`). The framework
  handles the rest.
- **Automated assertions** -- checks for starvation, cgroup
  isolation violations, and CPU time fairness. No manual inspection.
- **[Gauntlet](https://ktstr.dev/guide/running-tests/gauntlet.html)** --
  one `#[ktstr_test]` expands across topology presets (4-252 vCPUs,
  1-15 LLCs, optional SMT and multi-NUMA), filtered by per-test
  constraints. Multi-kernel runs (`--kernel A --kernel B`) add the
  kernel as an additional dimension.
- **Host-side introspection** -- reads kernel state and BPF maps
  from guest memory without guest-side instrumentation.
- **Per-thread profile diff** -- `ktstr ctprof capture` walks
  every live thread's scheduling, memory, I/O, and taskstats
  delay counters into a snapshot; `ktstr ctprof compare` diffs
  two snapshots for thread-level scheduling/memory/I/O
  regression hunting.
- **Auto-repro** -- on failure, reruns the scenario with BPF probes
  on the crash call chain, capturing arguments and struct state at
  each call site.
- **[Features](https://ktstr.dev/guide/features.html)** --
  testing, observability, debugging, and infrastructure.

## Installation

Add ktstr as a dev-dependency:

```toml
[dev-dependencies]
ktstr = "0.21.0"
```

The library is the test-author surface. The `anyhow::Result`
referenced in examples below is re-exported through
`ktstr::prelude`; no separate `anyhow` dev-dependency needed.

Tests are then run through `cargo ktstr test`, which wraps
[cargo-nextest](https://nexte.st/) with kernel resolution. Install
both binaries:

```sh
cargo install --locked cargo-nextest
cargo install --locked ktstr
```

This installs:

- `cargo-ktstr` -- the test-runner entry point invoked as
  `cargo ktstr test`. Wraps nextest with kernel resolution,
  coverage, verifier stats, shell access, and `cargo ktstr export`
  for reproducing test scenarios as self-contained shell scripts.
- `ktstr` -- standalone CLI for kernel cache management,
  interactive VM shells, host-wide per-thread profiling, and
  lock introspection. Optional unless you want the standalone
  diagnostic commands; `cargo-ktstr` already covers the test
  flow.

**Version compatibility:** pin the EXACT ktstr patch version across
`[dev-dependencies] ktstr = "X.Y.Z"` and
`cargo install --locked --bin cargo-ktstr ktstr@X.Y.Z`. ktstr is
pre-1.0 — minor-version bumps may break the test-facing API, and
patch bumps may break unstable internal surfaces (the CI matrix
runs against the locked patch). Examples below assume 0.21.0; an
example from a different release may not compile against the crate
this README documents.

To host ktstr tests in an *external* scheduler crate, add ktstr as an
optional dependency behind a feature gate (not a plain dev-dependency) —
see [Host a ktstr test in an external scheduler
crate](doc/guide/src/recipes/test-new-scheduler.md#9-host-a-ktstr-test-in-an-external-scheduler-crate).

When building from this repo, `scx-ktstr` (the test fixture
scheduler) is built automatically by the workspace. Downstream
consumers don't get `scx-ktstr` from `cargo install ktstr`; only
this repo's own integration tests use it.

## Setup

**Linux only (x86_64, aarch64).** ktstr boots KVM virtual machines;
it does not build or run on other platforms.

**Required:**

- Linux host with `/dev/kvm`
- Rust 1.94.1 (exact pin via `rust-toolchain.toml`; rustup uses it automatically)
- [cargo-nextest](https://nexte.st/) -- `cargo ktstr test` delegates
  to nextest internally.
- clang (BPF skeleton compilation)
- pkg-config, make, gcc
- autotools (autoconf, autopoint, flex, bison, gawk) -- vendored
  libbpf/libelf/zlib build
- BTF (`/sys/kernel/btf/vmlinux` -- present by default on most
  distros; needed for host kernel introspection in some tooling)
- Internet access on first build (downloads busybox source)

**Optional:**

- Test kernel with sched_ext for scheduler tests;
  `cargo ktstr kernel build` fetches and caches one. The build
  step runs `validate_kernel_config` which requires `CONFIG_SCHED_CLASS_EXT`
  (present from 6.12); kernels older than that fail the check at
  build time rather than running with a missing scheduler class.
  See [Supported kernels](https://ktstr.dev/guide/features.html#supported-kernels).
  Prebuilt kernels resolved via `KTSTR_KERNEL` must contain a
  vmlinux with embedded BTF.

```sh
# Ubuntu/Debian
sudo apt install clang pkg-config make gcc autoconf autopoint flex bison gawk

# Fedora
sudo dnf install clang pkgconf make gcc autoconf gettext-devel flex bison gawk
```

**liblzma note:** ktstr links `xz2` with the `static` feature — no
separate `liblzma-dev` / `xz-devel` package is needed. See
[CONTRIBUTING.md](CONTRIBUTING.md#liblzma-build-configuration)
for the dynamic-link path if you're modifying the workspace.

**Test files** go in `tests/` as standard Rust integration tests. Use `#[ktstr_test]` from `ktstr::prelude::*`.

See the [getting started guide](https://ktstr.dev/guide/getting-started.html) for kernel discovery and building a test kernel.

## Quick start

After the install + setup above, the full new-test path is:
paste an example below into `tests/<name>.rs`, then run
`cargo ktstr kernel build && cargo ktstr test`.

### Write a test

Declare cgroups and workers as data. No scheduler setup required:

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

Each test boots a KVM VM, creates the declared cgroups and workers,
runs the workload, and checks for starvation and fairness. For
canned scenarios, see `scenarios::steady` in the
[getting started guide](https://ktstr.dev/guide/getting-started.html).

### Define a scheduler

To test a custom sched_ext scheduler, declare it with
`declare_scheduler!`:

```rust
use ktstr::declare_scheduler;
use ktstr::prelude::*;

declare_scheduler!(MY_SCHED, {
    name = "my_sched",
    binary = "scx_my_sched",
    // (numa_nodes, llcs, cores_per_llc, threads_per_core)
    topology = (1, 2, 4, 1),
    sched_args = ["--enable-llc", "--enable-stealing"],
});
```

`binary = "scx_my_sched"` tells ktstr to auto-discover the scheduler
binary. The resolution cascade is: explicit `KTSTR_SCHEDULER` env
var → `$PATH` (when invoked under `cargo test`) →
sibling-of-test-binary → `target/debug/` → `target/release/` →
on-demand build from the workspace. If the scheduler is a `[[bin]]`
target in the same workspace, `cargo build` places it where the
sibling/target steps find it, so discovery is automatic. The
resolved binary is packed into the VM's initramfs. Tests without a
`scheduler` attribute run under EEVDF (the kernel's default
scheduler).

`topology = (numa_nodes, llcs, cores_per_llc, threads_per_core)`
sets the VM's CPU topology — `topology = (1, 2, 4, 1)` creates 1
NUMA node, 2 LLCs, 4 cores per LLC, 1 thread per core (8 vCPUs).
Topologies display as `NnNlNcNt` (e.g. `1n2l4c1t`). In
`#[ktstr_test]`, use named attributes instead: `llcs = 2, cores =
4, threads = 1, numa_nodes = 1`. Unset dimensions inherit from the
scheduler's topology. For non-uniform NUMA, see
`Topology::with_nodes()` in the
[topology guide](https://ktstr.dev/guide/concepts/topology.html).

`sched_args = [...]` are CLI args prepended to every test using
this scheduler. Per-test `#[ktstr_test(extra_sched_args = [...])]`
appends additional args after these.

The macro emits `pub static MY_SCHED: Scheduler` (referenceable by
the bare ident `MY_SCHED`) and ALSO registers an internal
`#[linkme::distributed_slice(KTSTR_SCHEDULERS)]` entry whose name
is mangled (`__KTSTR_SCHED_REG_MY_SCHED`) so `cargo ktstr verifier`
auto-discovers the scheduler. Test authors interact with the
public `MY_SCHED` static only; the distributed-slice entry is a
framework-internal hook. Add
`scheduler = MY_SCHED` to `#[ktstr_test]` to target it:

```rust
#[ktstr_test(scheduler = MY_SCHED)]
fn sched_two_cgroups(ctx: &Ctx) -> Result<AssertResult> {
    execute_defs(ctx, vec![
        CgroupDef::named("cg_0").workers(2),
        CgroupDef::named("cg_1").workers(2),
    ])
}
```

Tests referencing `MY_SCHED` inherit its topology. The macro
requires `llcs` to be an exact multiple of `numa_nodes`;
`topology = (1, 2, 4, 1)` (2 LLCs, 1 NUMA node) is fine,
`topology = (2, 3, ...)` is rejected at compile time.

### Multi-step scenarios

For dynamic topology changes, use `execute_steps` with `Step` and
`HoldSpec`:

```rust
use ktstr::prelude::*;

#[ktstr_test(scheduler = MY_SCHED, llcs = 1, cores = 4, threads = 1)]
fn cpuset_split(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![Step::with_defs(
        vec![
            CgroupDef::named("cg_0").cpuset(CpusetSpec::disjoint(0, 2)),
            CgroupDef::named("cg_1").cpuset(CpusetSpec::disjoint(1, 2)),
        ],
        HoldSpec::FULL,
    )];
    execute_steps(ctx, steps)
}
```

### Run a binary payload

To run a binary workload (`schbench`, `fio`, `stress-ng`,
anything else) as part of a test, declare a `Payload` and
reference it via `payload = ...` (primary slot) or
`workloads = [...]` (additional slots):

Each test crate declares its own `Payload` consts. Below is a
self-contained schbench example: the `#[derive(Payload)]` struct
defines the const, then the test references it. The macro
generates a `pub const SCHBENCH: Payload` whose name matches the
struct (uppercased, `Payload` suffix stripped).

```rust
use ktstr::prelude::*;

#[derive(Payload)]
#[payload(binary = "schbench", output = Json)]
#[default_args("--runtime", "5", "--message-threads", "2", "--json", "-")]
#[default_check(exit_code_eq(0))]
pub struct SchbenchPayload;

#[ktstr_test(scheduler = MY_SCHED, payload = SCHBENCH)]
fn schbench_under_my_sched(ctx: &Ctx) -> Result<AssertResult> {
    let (result, _metrics) = ctx.payload(&SCHBENCH).run()?;
    Ok(result)
}
```

See
[Payload Definitions](https://ktstr.dev/guide/writing-tests/scheduler-definitions.html#derive-payload)
for the `#[derive(Payload)]` macro and the full field surface
(`default_args`, `default_checks`, `metrics`, `include_files`).
This repo's `tests/common/fixtures.rs` carries reusable
in-tree examples (`FIO`, `FIO_JSON`, `STRESS_NG`, `SCHBENCH_JSON`)
that other ktstr tests inside this repo import via `mod common`;
they are not part of the published `ktstr` crate.

### Run

```sh
cargo ktstr test --kernel ../linux
```

`--kernel` accepts a kernel source tree path (e.g. `../linux`,
auto-built on first use), a version (`6.14.2`, or `6.14` for
latest patch), a cache key (see `kernel list`), a version
range (`6.12..6.14`), or a git source (`git+URL#tag=NAME`,
`git+URL#branch=NAME`, or `git+URL#sha=<40-hex>`). A range
expands only across series still listed in kernel.org's active
releases; add `--include-eol` to also cover end-of-life stable
series (enumerated from the gregkh linux-stable mirror).

`cargo ktstr test` wraps `cargo nextest run` with kernel
resolution (source tree, version, or cache key), kconfig
fragment merging, and shell access. Prefer it over a bare
`cargo nextest run` — the bare invocation only finds a kernel
when one is already cached under the exact key the test binary
asks for.

Requires `/dev/kvm` accessible to the invoking user. On most
distros that means adding the user to the `kvm` group; the
[Troubleshooting](https://ktstr.dev/guide/troubleshooting.html#devkvm-not-accessible)
page covers permission errors and nested-virt setup for CI
runners.

Passing tests:

```
    PASS [  11.34s] my_crate::my_sched_tests ktstr/two_cgroups
    PASS [  14.02s] my_crate::my_sched_tests ktstr/sched_two_cgroups
    PASS [  13.87s] my_crate::my_sched_tests ktstr/cpuset_split
```

A failing test prints assertion details:

```
    FAIL [  12.05s] my_crate::my_sched_tests ktstr/two_cgroups

--- STDERR ---
ktstr_test 'two_cgroups' [topo=1n1l2c1t] failed:
  stuck 3500ms on cpu1 at +1200ms

--- stats ---
4 workers, 2 cpus, 8 migrations, worst_spread=12.3%, worst_gap=3500ms
  cg0: workers=2 cpus=2 spread=5.1% gap=3500ms migrations=4 iter=15230
  cg1: workers=2 cpus=2 spread=12.3% gap=890ms migrations=4 iter=14870
```

### Exit codes

Per-test process exit codes project the
`Fail > Inconclusive > Pass > Skip` verdict lattice to three
values. CI gates and dashboard aggregators triage runs by
exit code:

| Code | Verdict      | Meaning                                                                                                                                                |
|------|--------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|
| `0`  | Pass / Skip  | Test passed, or the test never ran (host topology insufficient, resource contention).                                                                  |
| `1`  | Fail         | At least one assertion failed, OR `expect_err = true` and the test produced a Pass / Inconclusive (an `expect_err` test whose gate could not evaluate is unsatisfied just as it would be on a Pass). |
| `2`  | Inconclusive | A zero-denominator ratio gate could not evaluate — neither pass nor fail is truthful.                                                                  |

Exit code `2` is distinct from `1` so CI gates can treat
"could not evaluate" (often a workload that produced no signal —
zero iterations, zero pages, zero wake events) separately from
"evaluated and regressed." The constants are exposed as
`ktstr::prelude::{EXIT_PASS, EXIT_FAIL, EXIT_INCONCLUSIVE}`
for tooling that drives the harness programmatically.

See the [verdict outcomes guide](https://ktstr.dev/guide/concepts/checking.html#verdict-outcomes)
for the full four-state lattice (`Fail > Inconclusive > Pass > Skip`)
and CI-gate patterns.

Tests that have a reason to accept an Inconclusive arm as
not-a-failure (e.g. exploratory benchmarks whose ratio gate may
legitimately see no signal under certain host topologies) can
opt in with `#[ktstr_test(allow_inconclusive)]`. The dispatch
layer routes that test's Inconclusive verdict to exit code `0`
instead of `2`. Inconclusive is still recorded in the sidecar
and rendered in the failure dump — the flag only relaxes the
per-test exit-code projection. `expect_err` still dominates: an
`expect_err` test whose result is Inconclusive remains exit `1`
regardless of `allow_inconclusive`.

### cargo-ktstr subcommands

`cargo ktstr` wraps the full workflow and has subcommands beyond
`test`:

```sh
cargo ktstr test                                           # build/resolve kernel + run tests
cargo ktstr nextest                                        # visible alias for `test`
cargo ktstr test --kernel ~/linux -- -E 'test(my_test)'    # local source tree + nextest filter
cargo ktstr replay                                         # print a nextest filter for last session's failures (add --exec to re-run)
cargo ktstr coverage                                       # tests under cargo-llvm-cov nextest
cargo ktstr llvm-cov report --lcov --output-path lcov.info # raw llvm-cov passthrough (report/clean/show-env)
cargo ktstr kernel build --kernel 6.14.2                   # cache a specific version
cargo ktstr kernel build --kernel ~/linux                  # build from a local source tree
cargo ktstr kernel build --kernel git+URL#tag=v6.14        # build from a git tag
cargo ktstr kernel list                                    # list cached kernels (shows (EOL) tags)
cargo ktstr kernel clean --keep 3                          # keep 3 most recent
cargo ktstr verifier                                       # BPF verifier sweep (auto-discover kernel)
cargo ktstr verifier --kernel 6.14.2 --kernel 6.15.0       # sweep across multiple kernels
cargo ktstr stats                                          # aggregate gauntlet sidecars
cargo ktstr perf-delta --dual-run --kernel 6.14            # gate HEAD vs merge-base on performance_mode metrics
cargo ktstr stats show-host --run <key>                    # print archived HostContext for a run
cargo ktstr stats list-metrics                             # discover the metric vocabulary in archived sidecars
cargo ktstr stats list-values                              # list distinct values per filterable dimension in the sidecar pool
cargo ktstr stats explain-sidecar --run <key>              # explain per-field None causes across a run's sidecars
cargo ktstr show-host                                      # print current host context
cargo ktstr show-thresholds my_test                        # print resolved Assert thresholds for a test
cargo ktstr export my_test                                 # write a self-contained .run for bare-metal repro
cargo ktstr shell --kernel 6.14.2                          # interactive VM shell
cargo ktstr shell --kernel 6.14.2 --no-perf-mode           # shell on shared runners (skip flock/pinning/RT)
cargo ktstr completions bash                               # shell completions
```

`cargo ktstr --help` is the authoritative listing if anything
in the table above looks stale.

### Standalone CLI

`ktstr` is the debugging companion to the `#[ktstr_test]`
test harness. It owns kernel cache management, interactive VM
shells, host-wide per-thread profiling, and lock introspection.

Every `ktstr kernel ...` subcommand is identical to the corresponding
`cargo ktstr kernel ...`.

```sh
ktstr topo                                                 # show host CPU topology
ktstr shell --kernel 6.14.2                                # interactive VM shell (kernel optional)
ktstr kernel list                                          # manage cached kernels
ktstr kernel build --kernel 6.14.2
ktstr kernel build --kernel ../linux
ktstr kernel build --kernel git+URL#tag=v6.14
ktstr kernel clean --keep 3
ktstr ctprof capture --output baseline.ctprof.zst         # snapshot every live thread's counters
# ctprof capture pulls per-thread jemalloc counters via ptrace; needs root,
# `sudo setcap cap_sys_ptrace+eip $(which ktstr)`, or `kernel.yama.ptrace_scope=0`
ktstr ctprof compare baseline.ctprof.zst candidate.ctprof.zst # diff two snapshots on the selected grouping axis
ktstr ctprof show baseline.ctprof.zst                     # render one snapshot, no diff math
ktstr ctprof metric-list                                  # discover the metric vocabulary (--sort-by / --metrics)
ktstr locks                                               # enumerate held flocks (read-only)
ktstr completions bash
```

To reproduce a test scenario as a bare-metal shell script
without the test harness, use `cargo ktstr export`.

## Documentation

**[Zero to ktstr](https://ktstr.dev/guide/tutorial.html)** --
hands-on tutorial: define a scheduler, write a test, run it.

**[Guide](https://ktstr.dev/guide/)** -- getting started, concepts,
writing tests, recipes, architecture.

**[ctprof reference](https://ktstr.dev/guide/reference/ctprof.html)** --
metric registry, aggregation rules, taskstats kconfig gating,
adding-a-metric guide.

**[API docs](https://ktstr.dev/rustdoc/ktstr/)** -- rustdoc for all workspace crates.

## Contributing

Pull requests welcome. See [CONTRIBUTING.md](CONTRIBUTING.md)
for the workflow, coding conventions, and how to run the test
suite locally.

## License

GPL-2.0-only
