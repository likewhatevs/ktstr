//! Ready-made [`Payload`](ktstr::Payload) fixtures for the
//! benchmark binaries that dominate scheduler-regression testing:
//! `fio` (disk IO throughput, emits JSON), `stress-ng`
//! (synthetic CPU/memory stressors, exit-code only), and
//! `schbench` (latency percentiles via `--json -`).
//!
//! Each fixture is declared via
//! [`#[derive(Payload)]`](ktstr::Payload), the same path downstream
//! test authors use — so this module doubles as an end-to-end
//! exercise of the derive macro. The emitted `const` follows the
//! derive's naming convention: `struct FooPayload` produces
//! `const FOO: Payload`.
//!
//! These fixtures live under `tests/common/` rather than inside the
//! library's `src/` tree because they are TEST SCAFFOLDING, not
//! shipped API. A downstream scheduler author who wants the same
//! `fio` / `stress-ng` / `schbench` shapes should either copy the
//! declarations below into their own crate or write their own via
//! `#[derive(Payload)]`. The library does not ship fio, stress-ng,
//! or schbench binaries — the `kind = PayloadKind::Binary(name)`
//! just declares the name; host-side include-files resolution picks
//! the path up at test time.
//!
//! The fixtures cover both
//! [`OutputFormat`](ktstr::test_support::OutputFormat) variants:
//!
//! - [`FIO`] and [`FIO_JSON`] declare `OutputFormat::Json` with a
//!   set of [`MetricHint`](ktstr::test_support::MetricHint)s
//!   describing the canonical read/write throughput + latency paths.
//!   Extracted metrics land with correct polarity/unit automatically.
//! - [`SCHBENCH_JSON`] is the third `OutputFormat::Json` fixture
//!   (alongside [`FIO`] and [`FIO_JSON`]) — schbench via `--json -`,
//!   so extraction goes through the JSON walker.
//! - [`STRESS_NG`] uses `OutputFormat::ExitCode` with a single
//!   `exit_code_eq(0)` default — stress-ng reports via exit code
//!   (bogo_ops land in stderr and are not machine-extractable
//!   without `--metrics-brief --yaml`).
//!
//! All fixtures use short, stable `name` fields matching their
//! binary names — except FIO_JSON (`"fio_json"`) and SCHBENCH_JSON
//! (`"schbench_json"`), which use distinct names so they can coexist
//! with FIO under the pairwise-dedup rule on `#[ktstr_test(workloads =
//! [...])]`. The binary names themselves (`"fio"`, `"stress-ng"`,
//! `"schbench"`) are what ktstr's include-files infrastructure
//! resolves inside the guest.
//!
//! # Polarity::Unknown downstream
//!
//! Metrics extracted from a payload are matched against the
//! payload's `metrics` table by name in
//! [`PayloadRun`](ktstr::scenario::payload_run::PayloadRun)'s post-exit pipeline;
//! names with no matching hint land with
//! [`Polarity::Unknown`](ktstr::test_support::Polarity::Unknown) and
//! an empty unit. Unknown propagates as follows:
//!
//! - **`MetricCheck` assertion pass** — [`MetricCheck`](ktstr::test_support::MetricCheck)
//!   variants (`Min`, `Max`, `Range`, `Exists`, `ExitCodeEq`) compare
//!   values to thresholds without consulting polarity. An Unknown
//!   metric fails checks the same way a hinted metric does; polarity
//!   plays no role at assert time.
//! - **`AssertResult::merge` per-key worst-case** — when multiple
//!   cgroups contribute the same ext_metric, the merge consults the
//!   crate-internal `MetricDef` from the `METRICS` registry. Names
//!   absent from the registry (the case for any Unknown metric not
//!   also registered at crate scope) fall through to
//!   `crate::stats::infer_higher_is_worse`, which infers polarity from
//!   name substrings: throughput/iops-style names fold with `min`
//!   (higher-is-better), latency/error/`_us`-style names fold with
//!   `max` (higher-is-worse), and only names matching no token default
//!   to `higher_is_worse=true` (max) — NOT a declared polarity for the
//!   metric.
//! - **`cargo ktstr stats compare` cross-run comparison** — the
//!   crate-internal `compare_rows_by` iterates the `METRICS` registry
//!   only, so Unknown metrics extracted purely via `MetricHint`
//!   absence are NOT classified as regression or improvement. They
//!   are recorded to the sidecar for later manual inspection; to
//!   surface them in a comparison verdict, register a `MetricDef` in
//!   `src/stats/metric.rs` or add a `MetricHint` on the payload with an
//!   explicit polarity.

use ktstr::Payload;

/// `fio` — flexible IO tester. Canonical workload for disk/IO
/// scheduler regressions.
///
/// Output format: JSON. Supply `--output-format=json` at the call
/// site (via `.arg(...)` on the
/// [`PayloadRun`](ktstr::scenario::payload_run::PayloadRun) builder returned by
/// `ctx.payload(&FIO)`, or via a scheduler default_args entry) or
/// use [`FIO_JSON`] which bakes it into `default_args` for the
/// common "just give me metrics" path.
///
/// **Caveat:** `FIO` leaves `default_args` empty, so invoking it
/// without `--output-format=json` causes `fio` to emit its
/// human-readable output, `extract_metrics` finds no JSON region,
/// and the check pass records every referenced metric as missing
/// without otherwise failing. Prefer [`FIO_JSON`] unless the test
/// author intentionally overrides the output mode.
///
/// Metric hints cover the first-job read/write leaf names. Fio's
/// JSON output is deeply nested (`jobs[N].read.iops`,
/// `.write.iops`, `.read.lat_ns.mean`, etc.); the hints pin the
/// four most-commonly-asserted paths. Unhinted paths land as
/// [`Polarity::Unknown`](ktstr::test_support::Polarity::Unknown)
/// and are still extracted for sidecar regression tracking.
#[derive(Payload)]
#[payload(binary = "fio", output = Json)]
#[default_check(exit_code_eq(0))]
#[metric(name = "jobs.0.read.iops", polarity = HigherBetter, unit = "iops")]
#[metric(name = "jobs.0.write.iops", polarity = HigherBetter, unit = "iops")]
#[metric(name = "jobs.0.read.lat_ns.mean", polarity = LowerBetter, unit = "ns")]
#[metric(name = "jobs.0.write.lat_ns.mean", polarity = LowerBetter, unit = "ns")]
#[allow(dead_code)]
pub struct FioPayload;

/// `fio` with `--output-format=json` pre-baked into `default_args`.
///
/// Compared to [`FIO`], this fixture differs in exactly two
/// fields:
///
/// 1. **`name`** — `"fio_json"` instead of `"fio"`. Uses a
///    distinct name so sidecar files and log output can
///    disambiguate the two fixtures. The `binary` field (the name
///    resolved by the include-files infrastructure) is still
///    `"fio"` in both.
/// 2. **`default_args`** — `&["--output-format=json"]` instead of
///    `&[]`. Everything else — `kind`, `output`, `default_checks`,
///    `metrics` — is character-for-character identical to [`FIO`].
///
/// **Caveat: simultaneous FIO + FIO_JSON.** Both fixtures have
/// `kind = PayloadKind::Binary("fio")`, so a scenario that lists
/// `#[ktstr_test(workloads = [FIO, FIO_JSON])]` spawns the `fio`
/// binary TWICE — each with its own argv set, inside whatever
/// cgroup the framework places each fixture in. The pairwise-dedup
/// on the `workloads` attribute only rejects identical Payload
/// paths; two distinct Payload constants that happen to share a
/// binary are NOT deduped. Test authors who want the same fio
/// binary once should pick ONE of the two fixtures, and extend it
/// via `ctx.payload(&FIO).arg("--output-format=json")` if the
/// `FIO_JSON` preset's args don't match their scenario.
#[derive(Payload)]
#[payload(binary = "fio", name = "fio_json", output = Json)]
#[default_args("--output-format=json")]
#[default_check(exit_code_eq(0))]
#[metric(name = "jobs.0.read.iops", polarity = HigherBetter, unit = "iops")]
#[metric(name = "jobs.0.write.iops", polarity = HigherBetter, unit = "iops")]
#[metric(name = "jobs.0.read.lat_ns.mean", polarity = LowerBetter, unit = "ns")]
#[metric(name = "jobs.0.write.lat_ns.mean", polarity = LowerBetter, unit = "ns")]
#[allow(dead_code)]
pub struct FioJsonPayload;

/// `stress-ng` — synthetic load generator (CPU, memory, IO, VM,
/// etc.). Canonical workload for exercising scheduler decisions
/// under configurable contention.
///
/// Output format: `ExitCode`. stress-ng emits human-readable
/// progress lines to stderr; its machine-readable-metrics flags
/// write structured output to a caller-named file, not to stdout
/// or stderr. Since the extraction pipeline only consumes stdout,
/// no default stress-ng invocation feeds `extract_metrics`; the
/// fixture stays in exit-code mode and the happy path is a zero
/// exit.
///
/// **Caveat:** `default_args` is empty, so invoking `STRESS_NG`
/// without at least one stressor flag (e.g. `--cpu 1`, `--vm 1`)
/// causes stress-ng to print usage and exit nonzero on some
/// versions. Always append a stressor via `.arg(...)` on the
/// [`PayloadRun`](ktstr::scenario::payload_run::PayloadRun) builder returned
/// by `ctx.payload(&STRESS_NG)`.
///
/// Tests that want bogo_ops/sec metrics should declare their own
/// custom `Payload` via [`#[derive(Payload)]`](ktstr::Payload) and
/// pair it with a post-hoc bridge that emits JSON on stdout for the
/// `OutputFormat::Json` walker. stress-ng emits bogo_ops on stderr
/// by default and its machine-readable-metrics flags write to a
/// caller-named file, not stdout, so a wrapper that captures or
/// redirects structured output onto stdout is still required.
#[derive(Payload)]
#[payload(binary = "stress-ng")]
#[default_check(exit_code_eq(0))]
#[allow(dead_code)]
pub struct StressNgPayload;

/// `schbench` with `--json -` pre-baked into `default_args`.
///
/// Schbench writes a JSON summary block to stdout when invoked with
/// `--json -` (the third argument hyphen selects stdout over a file
/// path). That block is parseable by the
/// [`OutputFormat::Json`](ktstr::test_support::OutputFormat::Json)
/// extraction pipeline — stable dotted-path metric names pinned at
/// schbench's source-level JSON schema (`write_json_stats` in
/// `schbench.c`).
///
/// schbench writes its human-readable latency tables to stderr by
/// default; `--json -` is what moves a machine-parseable summary
/// onto stdout where the JSON walker consumes it. `name` is
/// `"schbench_json"` (distinct from the `"schbench"` binary) so
/// sidecar files and log output disambiguate it from any other
/// schbench-backed fixture; the `binary` field stays `"schbench"`.
///
/// Hint paths match the JSON keys emitted by schbench's
/// `write_json_stats` in `schbench.c`. Polarity annotations follow
/// schbench convention: latency percentiles are `LowerBetter`,
/// request-per-second is `HigherBetter`. Unhinted paths still land
/// in the extracted metric set with
/// [`Polarity::Unknown`](ktstr::test_support::Polarity::Unknown),
/// so the JSON blob is surfaced in sidecar output for regression
/// tracking even when a specific percentile is not pinned here.
///
/// **`--runtime 5` tradeoff.** The 5-second `default_args` runtime
/// is sized for fast CI smoke signal, NOT for tail-latency
/// regression hunting — schbench's sample count scales with runtime
/// (`nr_samples` in `show_latencies`), so 5 s collects roughly a
/// sixth of what `--runtime 30` does and leaves p99.9+ estimates
/// dominated by variance. Override via `.arg("--runtime").arg("30")`
/// on the [`PayloadRun`](ktstr::scenario::payload_run::PayloadRun)
/// builder returned by `ctx.payload(&SCHBENCH_JSON)`. schbench
/// parses argv with `getopt_long` and each `case 'r'` overwrites
/// `runtime = atoi(optarg)`, so the trailing setting wins on
/// duplicates and the appended override takes effect.
#[derive(Payload)]
#[payload(binary = "schbench", name = "schbench_json", output = Json)]
#[default_args("--runtime", "5", "--message-threads", "2", "--json", "-")]
#[default_check(exit_code_eq(0))]
#[metric(name = "int.rps_pct50.0", polarity = HigherBetter, unit = "rps")]
#[metric(name = "int.wakeup_latency_pct99.0", polarity = LowerBetter, unit = "us")]
#[metric(name = "int.wakeup_latency_pct50.0", polarity = LowerBetter, unit = "us")]
#[metric(name = "int.request_latency_pct99.0", polarity = LowerBetter, unit = "us")]
#[metric(name = "int.request_latency_pct50.0", polarity = LowerBetter, unit = "us")]
#[allow(dead_code)]
pub struct SchbenchJsonPayload;
