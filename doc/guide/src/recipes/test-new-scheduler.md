# Test a New Scheduler

End-to-end workflow: define a scheduler, write tests, run them,
sweep the BPF verifier. At the end you have a scheduler that boots
under real kernels on declared topologies, a test suite that fails
when its behavior regresses, and (optionally) all of it hosted in
your own crate.

## 1. Define the scheduler

`declare_scheduler!` generates a `pub static MY_SCHED: Scheduler`
and registers it so `cargo ktstr verifier` discovers it
automatically. Tests reference the bare `MY_SCHED` ident via
`#[ktstr_test(scheduler = MY_SCHED)]`.

```rust,ignore
use ktstr::declare_scheduler;
use ktstr::prelude::*;

declare_scheduler!(MY_SCHED, {
    name = "my_sched",
    binary = "scx_my_sched",
    topology = (1, 2, 4, 1),
    kernels = ["6.14", "6.15..=7.0"],
    sched_args = ["--exit-dump-len", "1048576"],
});
```

`topology = (1, 2, 4, 1)` is 1 NUMA node, 2 LLCs, 4 cores per LLC,
1 thread per core — written `1n2l4c1t` in test names and output. See
[Topology](../concepts/topology.md) for the notation and
[Scheduler Definitions](../writing-tests/scheduler-definitions.md)
for every supported field.

## 2. Write integration tests

Tests inherit the scheduler's topology. Override with explicit
`llcs`, `cores`, or `threads` when needed.

```rust,ignore
use ktstr::prelude::*;

#[ktstr_test(scheduler = MY_SCHED)]
fn basic_steady(ctx: &Ctx) -> Result<AssertResult> {
    // Inherits 1n2l4c1t from MY_SCHED
    scenarios::steady(ctx)
}

#[ktstr_test(scheduler = MY_SCHED, threads = 2)]
fn smt_steady(ctx: &Ctx) -> Result<AssertResult> {
    // Inherits llcs=2, cores=4; overrides threads to exercise SMT
    scenarios::steady(ctx)
}
```

While iterating on a single test, mark the others with
`#[ktstr_test(scheduler = MY_SCHED, ignore = true)]`: nextest skips
them by default, but they stay registered so the verifier sweep
still sees them. Clear the attribute when the test is ready to
land — leaving it on permanently silently drops coverage.

## 3. Build a kernel

Build a kernel with sched_ext support:

```sh
cargo ktstr kernel build
```

See [Getting Started](../getting-started.md) for version selection
and local source builds.

## 4. Run

`cargo ktstr test` resolves the kernel from `KTSTR_KERNEL`, the
cache, or an explicit `--kernel <spec>` — a version like `7.0`, a
cache key from `cargo ktstr kernel list`, or a path to a kernel
source tree. (It does not accept a prebuilt `bzImage`/`Image`;
only `cargo ktstr shell` does.)

```sh
cargo ktstr test                    # auto-discover from cache / KTSTR_KERNEL
cargo ktstr test --kernel 7.0       # pin to a version (latest 7.0.x)
cargo ktstr test --kernel ../linux  # pin to a local source checkout
```

A run looks like this — each `PASS` line is a fresh VM that booted,
ran the scenario, and shut down:

<!-- captured: cargo ktstr test --kernel 7.0 -- --features integration -E 'test(=ktstr/failure_dump_renders_bss_fields)' | ktstr 0.23.0 | kernel 7.0.14 -->
```text
cargo ktstr: fetching latest 7.0.x kernel version
cargo ktstr: latest 7.0.x kernel: 7.0.14
cargo ktstr: resolved kernel "7.0"
...
 Nextest run ID 24c18577-cd34-43bd-9d14-b0197701c187 with nextest profile: default
    Starting 1 test across 121 binaries (12531 tests skipped)
        PASS [  34.451s] (1/1) ktstr::failure_dump_e2e ktstr/failure_dump_renders_bss_fields
────────────
     Summary [  34.490s] 1 test run: 1 passed, 12531 skipped
...
```

The run footer names the output directory
(`target/ktstr/{kernel}-{project_commit}`) where per-test stats
sidecars land — see
[Runs and Regression Gates](../running-tests/runs.md).

## 5. Sweep the BPF verifier

The verifier sweep loads your scheduler's BPF programs under the
real kernel verifier, on every accepted topology preset, and
reports per-program verified-instruction counts. Instruction counts
vary with topology when `nr_cpus` bakes into `.rodata` — a
scheduler can attach on one topology and wedge on another, which is
exactly what the sweep catches.

```sh
cargo ktstr verifier                        # kernel from KTSTR_KERNEL / cache
cargo ktstr verifier --kernel ../linux      # pin one kernel
cargo ktstr verifier --kernel 6.14 --kernel 7.0   # sweep several
```

Each scheduler's `kernels = [...]` declaration filters the
operator-supplied set; an empty or omitted `kernels` field runs
against every kernel in the sweep.

<!-- captured: cargo ktstr verifier --kernel 7.0 --scheduler ktstr_sched --test kaslr_axis_e2e tiny-1llc tiny-2llc odd-3llc smt-2llc | ktstr 0.23.0 | kernel 7.0.14 -->
<div class="kt-term"><div class="kt-term-bar"><span class="kt-term-title">cargo ktstr verifier --kernel 7.0 --scheduler ktstr_sched</span></div>

<pre>cargo ktstr: resolved kernel "7.0"
...
    Starting 4 tests across 1 binary (55 tests skipped)
        PASS [  12.406s] (1/4) ktstr::kaslr_axis_e2e verifier/ktstr_sched/kernel_7_0/odd-3llc
        PASS [  12.432s] (2/4) ktstr::kaslr_axis_e2e verifier/ktstr_sched/kernel_7_0/smt-2llc
        PASS [  12.656s] (3/4) ktstr::kaslr_axis_e2e verifier/ktstr_sched/kernel_7_0/tiny-1llc
        PASS [  12.929s] (4/4) ktstr::kaslr_axis_e2e verifier/ktstr_sched/kernel_7_0/tiny-2llc
────────────
     Summary [  12.929s] 4 tests run: 4 passed, 55 skipped

verifier verified_insns (per scheduler; rows: kernel, cols: BPF program, cell: range across topologies):

ktstr_sched:
 kernel      ktstr_dispatch  ktstr_dump  ktstr_dump_cpu  ktstr_dump_task  ktstr_enqueue  ktstr_exit  ktstr_exit_task  ktstr_init  ktstr_init_task  ktstr_select_cp  ktstr_yield
 kernel_7_0  102             81          13              70               74             25          419              2296        <span class="t-yel">29077</span>            39               8

<span class="t-grn">verifier summary: 4 ✅  0 ❌  0 🇽</span>
 topology   ktstr_sched
 odd-3llc   ✅
 smt-2llc   ✅
 tiny-1llc  ✅
 tiny-2llc  ✅</pre></div>

One glance shows where the complexity lives (`ktstr_init_task` at
~29k verified instructions dwarfs every other program) and that all
four topologies attach cleanly. See
[BPF Verifier Sweep](../running-tests/verifier.md) for the output
format, cycle collapse on rejections, and the kernel-matching
contract. Kernel cache hygiene (`kernel list` / `kernel clean`)
lives in the [cargo ktstr](../running-tests/cargo-ktstr.md)
reference.

## 6. Debug failures

Boot an interactive shell with the scheduler binary packed into the
guest. `-i` (`--include-files`) adds host-side files to the guest's
`/include-files/` directory:

```sh
cargo ktstr shell -i ./target/debug/scx_my_sched
```

Inside the guest, run `/include-files/scx_my_sched` manually to
inspect behavior. Use `--exec CMD` to run a single command
non-interactively instead. See
[ktstr (standalone)](../running-tests/ktstr.md) and the
[cargo ktstr](../running-tests/cargo-ktstr.md) reference for all
flags, and [Investigate a Crash](investigate-crash.md) for the
crash-report workflow.

## 7. Write a crash test

Schedulers ship their own failure-handling paths; a negative test
pins them. The pattern: define a `BpfMapWrite` constant naming a
`.bss` global in your scheduler, have ktstr write a trigger value
into it after the scheduler loads, and have the scheduler's error
path read the global and call `scx_bpf_error(...)` with a known
message. The test passes only when that exact error fires — a
wrong or missing error fails, so silent regressions in the error
path become visible.

```rust,ignore
use ktstr::prelude::*;

// ".bss" names the libbpf-named .bss map; "crash" is the global the
// host writes; 1 is the trigger value. The field offset is resolved
// from the map's program BTF at write time.
static BPF_CRASH: BpfMapWrite = BpfMapWrite::new(".bss", "crash", 1);

#[ktstr_test(
    scheduler = MY_SCHED,
    bpf_map_write = BPF_CRASH,
    expect_err = true,
    expect_scx_bpf_error_contains = "my_sched: host-triggered crash",
)]
fn crash_path_emits_expected_error(ctx: &Ctx) -> Result<AssertResult> {
    ktstr::scenario::basic::custom_sched_mixed(ctx)
}
```

The substring contract is yours to define — the framework only
enforces that what you declare matches what the scheduler emits.
Use `expect_scx_bpf_error_matches = r"…"` for regex matching; the
full matcher semantics live in the
[`#[ktstr_test]` reference](../writing-tests/ktstr-test-macro.md).

## 8. Host a ktstr test in an external scheduler crate

A scheduler that lives in its own crate (outside the ktstr
workspace) can host ktstr tests directly. Gate ktstr behind a
feature so it never enters a normal build — ktstr pulls in
Linux-only, heavyweight dependencies (KVM, libbpf, a kernel loader,
guest memory) that would break non-Linux development and bloat
ordinary CI.

In the scheduler crate's `Cargo.toml`:

```toml
[dependencies]
# Pin the exact installed cargo-ktstr version — ktstr is pre-1.0 and
# a minor bump can break the test-facing API (see the README's
# version-compatibility note). Must be an optional [dependencies]
# entry, not a dev-dependency: `dep:ktstr` in [features] only
# resolves an optional normal dep (Cargo has no optional
# dev-dependencies).
ktstr = { version = "=X.Y.Z", optional = true }

[features]
ktstr-tests = ["dep:ktstr"]

[dev-dependencies]
# Only if a test body uses raw libc (e.g. fork / _exit); the
# prelude does not re-export libc.
libc = "0.2"
```

The test file gates its whole contents on the feature, so the crate
compiles to nothing extra when the feature is off:

```rust,ignore
#![cfg(feature = "ktstr-tests")]

use ktstr::prelude::*;

// `MY_SCHED` is your `declare_scheduler!(MY_SCHED, { ... })` constant
// (section 1). Drop the `scheduler =` attribute to run under the
// kernel's default scheduler instead.
#[ktstr_test(scheduler = MY_SCHED)]
fn my_sched_runs(ctx: &Ctx) -> Result<AssertResult> {
    ktstr::scenario::basic::custom_sched_mixed(ctx)
}
```

Build a kernel once (section 3), then run the gated tests. The
feature flag rides the nextest passthrough after `--`:

```sh
cargo ktstr kernel build --kernel /path/to/linux
cargo ktstr test --kernel /path/to/linux -- --features ktstr-tests
```

`cargo ktstr test` forwards everything after `--` to
`cargo nextest run`, which routes `--features ktstr-tests` to the
test compile.
