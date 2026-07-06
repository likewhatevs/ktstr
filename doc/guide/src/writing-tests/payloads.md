# Payloads and Included Files

Scheduler tests often need a real benchmark running alongside the
cgroup workers — `schbench` for wakeup latency, `fio` for IO
pressure, `stress-ng` for raw contention. A `Payload` declares that
binary once: its default args, how to parse its output, the metrics
it emits, the checks that gate them, and the files it needs packed
into the guest.

<div class="kt-doc-grid">
<div class="kt-doc-card"><strong>Declare</strong><p>Use <code>#[derive(Payload)]</code> to name the binary, args, metrics, and expected checks.</p></div>
<div class="kt-doc-card"><strong>Pack</strong><p>Include the executable and helper files in the guest initramfs automatically.</p></div>
<div class="kt-doc-card"><strong>Gate</strong><p>Parse JSON, exit codes, and metric leaves into sidecars and regression checks.</p></div>
</div>

## Declaring a payload {#derive-payload}

`#[derive(Payload)]` on a marker struct generates a `const Payload`.
This is a real fixture from ktstr's own test suite
(`tests/common/fixtures.rs`) — schbench with a machine-parseable
JSON summary on stdout:

```rust,ignore
use ktstr::Payload;

#[derive(Payload)]
#[payload(binary = "schbench", name = "schbench_json", output = Json)]
#[default_args("--runtime", "5", "--message-threads", "2", "--json", "-")]
#[default_check(exit_code_eq(0))]
#[metric(name = "int.rps_pct50.0", polarity = HigherBetter, unit = "rps")]
#[metric(name = "int.wakeup_latency_pct99.0", polarity = LowerBetter, unit = "us")]
#[metric(name = "int.request_latency_pct99.0", polarity = LowerBetter, unit = "us")]
pub struct SchbenchJsonPayload;
```

The derive emits `pub const SCHBENCH_JSON: Payload` — the const name
is the struct name with a trailing `Payload` stripped and converted
to SCREAMING_SNAKE_CASE (`FioPayload` → `FIO`; a suffixless
`BenchDriver` → `BENCH_DRIVER`). The const's visibility matches the
struct's.

The attributes:

- `#[payload(binary = "...", name = "...", output = ...)]` —
  `binary` (required) names the executable the include-file pipeline
  resolves and packs; `name` (default: the binary name) is the
  display label in sidecars and logs, so two fixtures can share one
  binary; `output` is `Json` (parse numeric leaves from the output)
  or `ExitCode` (status code only, the default).
- `#[default_args("--a", "--b")]` — CLI args prepended to every
  invocation; per-test `.arg(...)` calls append after them.
- `#[default_check(exit_code_eq(0))]` — a `MetricCheck` constructor
  (`min`, `max`, `range`, `exists`, `exit_code_eq`); the
  `MetricCheck::` prefix is optional. Repeat the attribute for
  several checks.
- `#[metric(name = "...", polarity = ..., unit = "...")]` — declares
  a metric the payload emits. `polarity` is `HigherBetter`,
  `LowerBetter`, `TargetValue(x)`, or `Unknown`; it drives
  `list-metrics` and comparison direction. Duplicate metric names
  are rejected at expansion.
- `#[include_files("helper", "config.json")]` — extra files packed
  into the guest alongside the binary. The `binary` itself is
  auto-prepended, so it never needs listing.

`Payload` is `#[non_exhaustive]`: downstream crates cannot use
struct-literal construction. For a binary with no declared metrics
or args, `Payload::binary(name, executable)` is the one-line
constructor; for anything richer, use the derive. Metrics extracted
with no matching `#[metric]` hint still land in the sidecar with
`Polarity::Unknown` — declare a hint for any metric a comparison
verdict should classify.

## Using a payload in a test

Reference the const from the test attribute, run it from the body:

```rust,ignore
#[ktstr_test(scheduler = MY_SCHED, payload = SCHBENCH_JSON, duration_s = 10)]
fn wakeup_latency_under_load(ctx: &Ctx) -> Result<AssertResult> {
    ctx.payload(&SCHBENCH_JSON)
        .arg("--runtime").arg("8")
        .run()
        .map(|(assert_result, _metrics)| assert_result)
}
```

`payload =` is the primary slot; `workloads = [A, B]` composes more
payloads alongside it (each runnable via `ctx.payload(&A)`). The
builder returned by `ctx.payload(...)` inherits the payload's
default args and checks; `.arg(...)` / `.args(...)` extend,
`.in_cgroup(...)` places the child, `.timeout(...)` bounds it, and
the terminal `.run()` blocks and returns
`Result<(AssertResult, PayloadMetrics)>`. Only binary-kind payloads
are runnable; the `scheduler =` slot is separate and takes a bare
`Scheduler`, never a `Payload`.

Two dedup rules: the same const may not appear in both `payload` and
`workloads` (or twice in `workloads`), but two distinct consts that
share a binary — like `FIO` and `FIO_JSON` — are not deduped and
will spawn the binary twice, each with its own argv. Pick one
fixture per binary unless two instances are the point.

## Metric extraction: stdout first, then stderr

`OutputFormat::Json` reads the payload's stdout as the primary
stream, then falls back to stderr if stdout is empty or yields no
metrics. Some benchmarks emit their numbers only to stderr —
`schbench`, for example, writes its `Wakeup Latencies percentiles` /
`Request Latencies percentiles` blocks via `fprintf(stderr, ...)`
and leaves stdout blank (pass `--json -` for a machine-parseable
summary on stdout). The fallback keeps those benchmarks usable
without a redirect.

Consequence: a payload that writes mixed output to both streams has
metrics extracted from stdout only, because the fallback fires
solely when the primary stream yields nothing parseable. If you care
about stderr-side numbers for a stdout-emitting binary, redirect
stderr into stdout at the payload layer.

`stress-ng` is the mirror trap: progress and per-stressor summaries
go to stderr and stdout is blank, so the fallback sees prose and
`OutputFormat::Json` returns zero metrics. Keep
`OutputFormat::ExitCode` for stress-ng unless the payload is wired
to emit JSON on stdout.

## Included files

Payloads declare their guest-filesystem dependencies on the
`Payload` itself via `#[include_files(...)]`, instead of relying on
the CLI `-i` / `--include-files` flag at every invocation. Specs are
resolved at test time through the same pipeline the CLI flag uses
(see [cargo ktstr shell](../running-tests/cargo-ktstr.md#shell)).

### Spec shapes

Which branch fires is decided by the shape of the string:

- **Bare name** (single component, no `/`) — looked up in the
  current working directory first, then the host `PATH`. Packed as
  `include-files/<filename>`.
  `"fio"` → host `/usr/bin/fio` → guest `/include-files/fio`.
- **Relative or absolute path** — used verbatim and must exist;
  relative paths resolve against the harness's working directory at
  test time. Packed as `include-files/<filename>`.
  `"./test-fixtures/workload.json"` → guest
  `/include-files/workload.json`.
- **Directory** — walked recursively (symlinks followed,
  non-regular files skipped); the basename becomes the root.
  `"./helpers"` containing `a.sh` and `sub/b.sh` → guest
  `/include-files/helpers/a.sh` and
  `/include-files/helpers/sub/b.sh`.

Strings in the test-level `extra_include_files` attribute follow the
same three shapes. They are not anchored to `CARGO_MANIFEST_DIR` —
they resolve against the working directory at test time, plus `PATH`
for bare names, and the attribute accepts plain string literals only
(no `concat!(env!(...))`). For fixtures shipped alongside test
source, the reliable options are a bare name placed on `PATH` by a
setup step, or a relative path rooted where the test is invoked.

### A fully declarative test

```rust,ignore
use ktstr::declare_scheduler;
use ktstr::prelude::*;

declare_scheduler!(MY_SCHED, {
    name = "my_sched",
    binary = "scx_my_sched",
    topology = (1, 1, 2, 1),
});

#[derive(Payload)]
#[payload(binary = "bench-driver")]
#[include_files("bench-helper")]
#[metric(name = "ops_per_sec", polarity = HigherBetter, unit = "ops/s")]
struct BenchDriver;

#[ktstr_test(
    scheduler = MY_SCHED,
    payload = BENCH_DRIVER,
    extra_include_files = ["test-fixtures/workload.json"],
    duration_s = 5,
)]
fn bench_driver_runs_with_declared_helpers(ctx: &Ctx) -> Result<AssertResult> {
    // bench-driver, bench-helper, and workload.json all land in the
    // guest at /include-files/ and are on the worker's PATH; no -i
    // flag on any host-side invocation.
    ctx.payload(&BENCH_DRIVER)
        .run()
        .map(|(assert_result, _metrics)| assert_result)
}
```

The declarative set — the payload's `include_files`, each
workload's, and `extra_include_files` — is aggregated at test time
and deduped on identical `(archive path, host path)` pairs. Two
declarations that resolve to the same archive slot with different
host paths are a hard error naming both host paths, rather than one
silently overwriting the other.

## Probe-wiring environment variables

Two variables pack the jemalloc allocator probe pair into the guest:
`KTSTR_JEMALLOC_PROBE_BINARY` and
`KTSTR_JEMALLOC_ALLOC_WORKER_BINARY` (absolute host paths; unset
means no probe is packed). They must be populated before ktstr's
nextest pre-dispatch runs — plain test-body code is too late — so
tests that need them set both from a `#[ctor]` constructor, using
the re-export at `ktstr::__private::ctor` to avoid a second `ctor`
crate in the dependency tree. See
[Environment Variables](../reference/environment-variables.md) for
the reference rows.
