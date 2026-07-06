# Running Tests

Every `#[ktstr_test]` boots a fresh KVM microVM with the topology the
test declares, on the exact kernel you target. `cargo ktstr test`
resolves that kernel (building and caching it when needed) and wraps
`cargo nextest run`, so nextest's filtering, retries, and parallelism
all apply.

<div class="kt-doc-grid">
<div class="kt-doc-card"><strong><a href="running-tests/failures.html">A run failed</a></strong><p>Read the assertion, timeline, scheduler log, monitor verdict, dumps, and replay command.</p></div>
<div class="kt-doc-card"><strong><a href="running-tests/gauntlet.html">Need topology coverage?</a></strong><p>Expand one test across presets, then select variants with nextest filters or a CI budget.</p></div>
<div class="kt-doc-card"><strong><a href="running-tests/runs.html">Need regression gates?</a></strong><p>Use run sidecars and <code>perf-delta</code> instead of comparing terminal output by eye.</p></div>
</div>

## Quick reference

```sh
# Run all tests
cargo ktstr test --kernel ../linux

# Run a specific test
cargo ktstr test --kernel ../linux -- -E 'test(sched_basic_proportional)'

# Run all ktstr-managed tests, skipping non-ktstr tests in the same crate
cargo ktstr test --kernel ../linux -- -E 'test(/^ktstr/)'

# Run ignored gauntlet variants
cargo ktstr test --kernel ../linux -- --run-ignored ignored-only -E 'test(gauntlet/)'
```

More patterns:

| Goal | Command |
|---|---|
| Re-run one exact ktstr case | `cargo ktstr test --kernel 7.0 -- -E 'test(=ktstr/my_test)'` |
| Compare two kernels | `cargo ktstr test --kernel 6.14 --kernel 7.0` |
| Stay inside a CI time budget | `KTSTR_BUDGET_SECS=300 cargo ktstr test --kernel 7.0` |
| Open the same VM shape manually | `cargo ktstr shell --kernel 7.0 --test my_test` |

## What's in this chapter

- [cargo ktstr](running-tests/cargo-ktstr.md) — the host-side command:
  kernel resolution, test dispatch, replay, coverage, export, plus the
  interactive shell, `topo`, `ctprof`, and `locks` debugging tools
  (and the standalone `ktstr` binary that carries them without cargo).
- [Gauntlet](running-tests/gauntlet.md) — run every test across a
  matrix of topology presets.
- [BPF Verifier Sweep](running-tests/verifier.md) — verify, attach,
  and dispatch every declared scheduler across topologies.
- [Reading Failure Output](running-tests/failures.md) — what a failed
  test prints, section by section, and how to investigate.
- [Auto-Repro](running-tests/auto-repro.md) — the second VM that
  replays a scheduler crash with probes attached.
- [Runs and Regression Gates](running-tests/runs.md) — result
  sidecars, `stats`, and `perf-delta`.

## Test names and variants {#test-name-shapes}

Tests registered through `#[ktstr_test]` show up in nextest output
under one of four prefixes:

- `ktstr/{name}` — single-kernel run (or any `host_only` test, which
  never boots a VM and so never multiplies across kernels).
- `ktstr/{name}/{kernel}` — one case per (test × kernel) when
  `--kernel` resolves to two or more kernels.
- `gauntlet/{name}/{preset}` — one case per topology preset
  (see [Gauntlet](running-tests/gauntlet.md)).
- `gauntlet/{name}/{preset}/{kernel}` — the full (test × preset ×
  kernel) expansion under a multi-kernel run.

This is what those names look like in a real run:

<!-- captured: cargo ktstr test --kernel 7.0 -- --features integration -E 'test(=ktstr/failure_dump_renders_bss_fields)' | ktstr 0.23.0 | kernel 7.0.14 -->
<div class="kt-term"><div class="kt-term-bar"><span class="kt-term-title">nextest case name: base ktstr variant</span></div>

<pre>
 Nextest run ID 98581174-246f-4824-a170-50992df166d7 with nextest profile: default
    Starting 1 test across 121 binaries (12531 tests skipped)
        <span class="t-grn">PASS [  34.459s] (1/1) ktstr::failure_dump_e2e ktstr/failure_dump_renders_bss_fields</span></pre></div>

<!-- captured: KTSTR_KERNEL=7.0 cargo nextest list --features integration -E 'test(gauntlet/) & binary(worktype_coverage_fork_gauntlet_e2e)' | ktstr 0.23.0 | kernel 7.0.14 -->
<div class="kt-term"><div class="kt-term-bar"><span class="kt-term-title">nextest list: gauntlet variants</span></div>

<pre>
...
ktstr::worktype_coverage_fork_gauntlet_e2e gauntlet/worktype_fork_gauntlet_covers_all_arms/smt-3llc
ktstr::worktype_coverage_fork_gauntlet_e2e gauntlet/worktype_fork_gauntlet_covers_all_arms/tiny-1llc
ktstr::worktype_coverage_fork_gauntlet_e2e gauntlet/worktype_fork_gauntlet_covers_all_arms/tiny-2llc
</pre></div>

Filter by prefix with `-E 'test(/^ktstr/)'` or `-E 'test(/^gauntlet/)'`.

> [!TIP]
> `test(NAME)` is a substring match; the exact-match form `test(=NAME)`
> matches the **full** nextest name, prefix included. Use
> `test(=ktstr/sched_basic_proportional)`, not the bare function name —
> `test(=sched_basic_proportional)` matches nothing.

The `{kernel}` suffix is a sanitized kernel label: `kernel_` prefix,
lowercase, non-alphanumeric characters collapsed to `_` — `6.16.1`
becomes `kernel_6_16_1`, and a path spec becomes
`kernel_path_{basename}_{hash6}` (with `_dirty` appended when the
source tree has uncommitted changes). The 6-character hash
disambiguates two source paths that share a basename.

`RUST_BACKTRACE=1` controls panic backtraces and verbose failure
output, not guest console streaming — see
[Reading Failure Output](running-tests/failures.md) for the
investigation knobs.

## Budget-based test selection

Set `KTSTR_BUDGET_SECS` to select the subset of tests that maximizes
configuration coverage within a time budget — useful for CI pipelines
and quick smoke tests:

```sh
KTSTR_BUDGET_SECS=300 cargo ktstr test --kernel ../linux
```

The selector encodes each test as a bitset of properties (scheduler,
topology class, SMT, workload characteristics) and greedily picks the
tests with the highest marginal coverage per estimated second, with
duration estimates accounting for VM boot overhead by vCPU count. A
summary is printed to stderr during budget-mode listing:

<div class="kt-term"><div class="kt-term-bar"><span class="kt-term-title">budget-mode selection summary</span></div>

<pre><span class="t-b">ktstr budget:</span> <span class="t-grn">42/1200 tests</span>, 295/300s used, 38/38 configurations covered</pre></div>

## Testing your own scheduler

Declare it with `declare_scheduler!` and reference it from
`#[ktstr_test(scheduler = ...)]` — see
[Scheduler Definitions](writing-tests/scheduler-definitions.md) and
the [Test a New Scheduler](recipes/test-new-scheduler.md) recipe.
