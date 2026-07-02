# Test a New Scheduler

End-to-end workflow: define a scheduler, write tests, run them.

## 1. Define the scheduler

Use `declare_scheduler!` to register a scheduler in the
`KTSTR_SCHEDULERS` distributed slice. The verifier sweep picks
it up automatically.

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

The macro generates `pub static MY_SCHED: Scheduler` plus a
private `linkme` registration so `cargo ktstr verifier`
discovers the scheduler automatically. Tests reference the
bare `MY_SCHED` ident via
`#[ktstr_test(scheduler = MY_SCHED)]`.

See [Scheduler Definitions](../writing-tests/scheduler-definitions.md)
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
`#[ktstr_test(scheduler = MY_SCHED, ignore = true)]` so the
distributed-slice walker still discovers them but
`cargo ktstr test` skips them in the sweep. The macro emits
`#[ignore]` on the generated `#[test]` wrapper (so nextest skips
the test by default) while still registering the entry in the
`KTSTR_TESTS` distributed slice (so the verifier sweep and other
slice walkers see it). Clear the attribute when the test is ready
to land — leaving it on permanently silently drops coverage from
test-runner enumeration.

## 3. Build a kernel

Build a kernel with sched_ext support:

```sh
cargo ktstr kernel build
```

See [Getting Started: Build a kernel](../getting-started.md#build-a-kernel)
for version selection and local source builds.

## 4. Run

`cargo ktstr test` resolves the kernel from `KTSTR_KERNEL`, the cache,
or an explicit `--kernel <spec>` (a version like `6.14`, a cache key
from `cargo ktstr kernel list`, or a path to a kernel source tree).
`cargo ktstr test` does not accept a prebuilt `bzImage`/`Image` (only
`cargo ktstr shell` does). Step 3 populated the cache with the
declared kernels, so the bare form is sufficient when a single kernel
is available:

```sh
cargo ktstr test            # auto-discover from cache / KTSTR_KERNEL
cargo ktstr test --kernel 6.14    # pin to a specific cached version
cargo ktstr test --kernel ../linux  # pin to a local source checkout
```

## 5. Check BPF complexity (optional)

Collect per-program verifier statistics across the declared
kernels and accepted topology presets:

```sh
# Use the kernel auto-discovered via KTSTR_KERNEL / cache.
cargo ktstr verifier

# Pin to a specific kernel build.
cargo ktstr verifier --kernel ../linux

# Sweep across multiple kernels. Each scheduler's
# `kernels = [...]` declaration acts as a per-scheduler filter on
# the operator-supplied set; an empty (or omitted) `kernels` field
# means the scheduler runs against every kernel in the sweep.
cargo ktstr verifier --kernel 6.14 --kernel 7.0
```

See [BPF Verifier](../running-tests/verifier.md) for output
format, cycle collapse, and the cell-name → kernel matching
contract.

## 6. Manage the kernel cache

Cached kernel images accumulate under
`$XDG_CACHE_HOME/ktstr/kernels/`. Keep a handful of recent
builds and drop the rest when disk pressure grows:

```sh
cargo ktstr kernel list                # inspect cache contents
cargo ktstr kernel clean --keep 3      # keep the 3 most recent images
cargo ktstr kernel clean --force       # remove everything (non-interactive)
```

## 7. Debug failures

Boot an interactive shell with the scheduler binary packed into
the guest. `cargo ktstr shell` is interactive by default; the
`-i` (`--include-files`) flag adds host-side files (binaries,
configs, scripts) to the guest's `/include-files/` directory so
they're available inside the VM:

```sh
cargo ktstr shell -i ./target/debug/scx_my_sched
```

Inside the guest, run `/include-files/scx_my_sched` manually to
inspect behavior. Use `--exec CMD` to opt out of the interactive
prompt and run a single command non-interactively. See
[cargo-ktstr shell](../running-tests/cargo-ktstr.md#shell) for
all flags.

## 8. Write a crash test

Schedulers ship with their own failure-handling paths. A
negative test pins those paths: tell ktstr to force a BPF-map
write into the scheduler that produces a known `scx_bpf_error`,
declare the test as expected-error, and assert on the rendered
error message. The test passes when the scheduler emits the
expected error and fails when it doesn't (or emits the wrong
one — silent regressions become visible).

The test author defines a `BpfMapWrite` constant naming the
scheduler-side `.bss` slot to write, then names it in the
`bpf_map_write` macro attribute. The scheduler under test must
expose the slot AND emit a deterministic `scx_bpf_error(...)`
when it sees the host-written value:

```rust,ignore
use ktstr::prelude::*;

// User-defined trigger: ".bss" suffix matches the libbpf-named
// .bss map; field name and value name the global + payload the host
// writes after the scheduler loads. The scheduler reads this global
// in its error path and calls scx_bpf_error(...). The field's offset
// is resolved from the map's program BTF at write time.
static BPF_CRASH: BpfMapWrite = BpfMapWrite::new(".bss", "crash", 1);

#[ktstr_test(
    scheduler = MY_SCHED,
    bpf_map_write = BPF_CRASH,
    expect_err = true,
    expect_scx_bpf_error_contains = "ktstr: host-triggered crash",
)]
fn crash_path_emits_expected_error(ctx: &Ctx) -> Result<AssertResult> {
    ktstr::scenario::basic::custom_sched_mixed(ctx)
}
```

The host writes the trigger value into the scheduler's `.bss` slot
after the scheduler loads. The scheduler-side error path reads the
slot and calls `scx_bpf_error(...)` with a message containing
the documented substring (`"ktstr: host-triggered crash"` is the
convention used by the scx-ktstr fixture). The substring contract
is yours to define for your scheduler — the framework only enforces
that whatever you declare in `expect_scx_bpf_error_contains` matches
what the scheduler emits.

Use `expect_scx_bpf_error_matches = r"…"` (regex) for richer
matching against escape-sequence-rich messages. Both attributes
gate against the same expected-error-on-pass contract — the
test fails if the scheduler exits without emitting the matching
error.

See [The #\[ktstr_test\] Macro](../writing-tests/ktstr-test-macro.md)
for all available attributes and
[Scheduler Definitions](../writing-tests/scheduler-definitions.md)
for the full `Scheduler` type and the `declare_scheduler!` macro.

## 9. Host a ktstr test in an external scheduler crate

A scheduler that lives in its own crate (outside the ktstr
workspace) can host ktstr tests directly. Gate ktstr behind a
feature so it never enters a normal build — ktstr pulls in
Linux-only, heavyweight dependencies (KVM, libbpf, a kernel
loader, guest memory) that would break non-Linux development and
bloat ordinary CI.

In the scheduler crate's `Cargo.toml`:

```toml
[dependencies]
# Optional + exact-pinned: ktstr is pre-1.0, so a minor bump can
# break the test-facing API. The pin MUST equal the installed
# cargo-ktstr (see "Version compatibility" in the README). It must
# be an optional [dependencies] entry, not a dev-dependency, because
# `dep:ktstr` in [features] only resolves an optional normal dep
# (Cargo has no optional dev-dependencies).
ktstr = { version = "=0.21.0", optional = true }

[features]
# Enabling the feature pulls ktstr in; without it the crate builds
# with no ktstr dependency at all.
ktstr-tests = ["dep:ktstr"]

[dev-dependencies]
# Only if a test body uses raw libc (e.g. fork / _exit); the
# prelude does not re-export libc.
libc = "0.2"
```

The test file gates its whole contents on the feature, so the
crate compiles to nothing extra when the feature is off:

```rust,ignore
#![cfg(feature = "ktstr-tests")]

use ktstr::prelude::*;

// `MY_SCHED` is your scheduler's `declare_scheduler!(MY_SCHED, { ... })`
// constant (see section 1). Drop the `scheduler =` attribute to run the
// test under the kernel's default EEVDF instead.
#[ktstr_test(scheduler = MY_SCHED)]
fn my_sched_runs(ctx: &Ctx) -> Result<AssertResult> {
    ktstr::scenario::basic::custom_sched_mixed(ctx)
}
```

Build a kernel once (see section 3), then run the gated tests.
The feature flag rides the nextest passthrough after `--`:

```sh
cargo ktstr kernel build --source /path/to/linux
cargo ktstr test --kernel /path/to/linux -- --features ktstr-tests
```

`cargo ktstr test` forwards everything after `--` to `cargo nextest
run`, which routes `--features ktstr-tests` to the test compile.
