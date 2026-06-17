# Running Tests

Tests run via `cargo ktstr test --kernel ../linux`, which resolves the
kernel and wraps `cargo nextest run` to boot KVM virtual machines for
each `#[ktstr_test]` entry. Raw `cargo nextest run` remains available
as a fallback once a kernel is in place via the discovery chain.

## Quick reference

```sh
# Run all tests
cargo ktstr test --kernel ../linux

# Run a specific test
cargo ktstr test --kernel ../linux -- -E 'test(sched_basic_proportional)'

# Run all ktstr-managed tests, skipping non-ktstr tests in the same crate
cargo ktstr test --kernel ../linux -- -E 'test(/^ktstr/)'

# Run ignored gauntlet tests
cargo ktstr test --kernel ../linux -- --run-ignored ignored-only -E 'test(gauntlet/)'
```

## Test name shapes

Tests registered through `#[ktstr_test]` show up in nextest output
under one of four prefixes, all routed through ktstr's dispatch
layer:

- `ktstr/{name}` — single-variant form. Single-kernel mode (no
  `KTSTR_KERNEL_LIST` or exactly one entry), or any `host_only` test
  regardless of kernel-list size (host_only tests never boot a VM,
  so kernel identity does not multiply them).
- `ktstr/{name}/{kernel}` — multi-kernel single-variant form. Active
  when `KTSTR_KERNEL_LIST` carries 2+ kernels, so each (test ×
  kernel) pair becomes its own nextest case.
- `gauntlet/{name}/{preset}` — single-kernel gauntlet expansion. For
  each `#[ktstr_test]`, ktstr emits one entry per gauntlet preset so
  topology variants run as distinct nextest cases.
- `gauntlet/{name}/{preset}/{kernel}` — multi-kernel gauntlet
  expansion. Active when `KTSTR_KERNEL_LIST` carries 2+ kernels, so
  each (test × preset × kernel) triple becomes its own case.

Filter by prefix with `-E 'test(/^ktstr/)'` (all ktstr-managed
single-variant tests) or `-E 'test(/^gauntlet/)'` (all gauntlet
expansions). Substring matches like `-E 'test(ktstr/)'` also work;
the `^` anchor avoids accidentally matching test names like
`verify_ktstr/bar` from a sibling crate or `something_ktstr/foo`
from a manually-named test that happens to contain `ktstr/` as a
substring.

The `#[ktstr_test]` attribute itself isn't a nextest filter
selector — nextest filters on test names, and the prefix routing
above is how `#[ktstr_test]`-marked tests surface as filterable
names.

The `{kernel}` suffix is a sanitized kernel label: `kernel_`
prefix, lowercase, every non-alphanumeric ASCII character collapsed
to `_`, consecutive underscores collapsed, trailing underscores
stripped. Version specs render as `kernel_<version>` — `6.16.1`
becomes `kernel_6_16_1`. Path specs render as
`kernel_path_<basename>_<hash6>` — `../linux` becomes
`kernel_path_linux_<hash6>` (and `..._dirty` when the source tree
is dirty). The 6-char path hash (a CRC32 over the canonical path
string) disambiguates two different source paths that share a
basename.

## Run analysis

Each test invocation writes a `*.ktstr.json` sidecar per variant
into `{CARGO_TARGET_DIR or "target"}/ktstr/{kernel}-{project_commit}/`.
`cargo ktstr stats list` enumerates runs; `cargo ktstr stats
compare`, `list-values`, and `show-host` operate on those sidecars.
`list-metrics` is independent of the sidecar pool — it enumerates the
static `ktstr::stats::METRICS` regression-metric registry (metric
names, polarities, default abs/rel thresholds, units) and takes no
run input. See [Runs](running-tests/runs.md) for the directory
layout, last-writer-wins semantics, and the comparison workflow.

## Budget-based test selection

Set `KTSTR_BUDGET_SECS` to select the subset of tests that maximizes
feature coverage within a time budget. Useful for CI pipelines or
quick smoke tests.

```sh
# Run the best 5 minutes of tests
KTSTR_BUDGET_SECS=300 cargo ktstr test --kernel ../linux

# Budget applies to gauntlet variants too
KTSTR_BUDGET_SECS=600 cargo ktstr test --kernel ../linux -- --run-ignored all
```

The selector encodes each test as a bitset of properties (scheduler,
topology class, SMT, workload characteristics) and greedily picks
tests with the highest marginal coverage per estimated second.
Duration estimates account for VM boot overhead based on vCPU count.

A summary is printed to stderr during budget-mode `--list` (only
when `KTSTR_BUDGET_SECS` is set):

```text
ktstr budget: 42/1200 tests, 295/300s used, 38/38 configurations covered
```

When `KTSTR_BUDGET_SECS` is not set, all tests are listed as usual
with no budget summary.

## Custom scheduler

Declare a scheduler with `declare_scheduler!` and reference the
bare const from `#[ktstr_test(scheduler = ...)]`:

```rust,ignore
use ktstr::declare_scheduler;
use ktstr::prelude::*;

declare_scheduler!(MY_SCHED, {
    name = "my_sched",
    binary = "scx_my_sched",
});

#[ktstr_test(scheduler = MY_SCHED)]
fn my_sched_test(ctx: &Ctx) -> Result<AssertResult> {
    Ok(AssertResult::pass())
}
```

The binary is injected into the VM's initramfs and started before
scenarios run. See [Test a New Scheduler](recipes/test-new-scheduler.md)
for the full end-to-end workflow, and
[Payload Definitions](writing-tests/scheduler-definitions.md#derive-payload)
for the `#[derive(Payload)]` macro that handles binary-kind
workloads (`schbench`, `fio`, etc.) — distinct from the
scheduler-under-test surface.
