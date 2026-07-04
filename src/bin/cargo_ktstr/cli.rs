//! CLI argument types for the `cargo ktstr` binary.
//!
//! Houses the clap-derived `Cargo` / `CargoSub` / `Ktstr` /
//! `KtstrCommand` / `StatsCommand` enums and structs
//! the binary entry point parses against. Pulled out of
//! [`super`] so the parent file stays focused on dispatch and
//! sub-helpers — the clap derive expansion is bulky enough to
//! dominate a single-file layout, and consumers (`Subcommand`
//! match arms, `try_parse_from` tests) only need the type
//! shapes here.

use std::path::PathBuf;

use clap::{ArgAction, Parser, Subcommand};
use ktstr::cli::KernelCommand;
use ktstr::cli::{INCLUDE_EOL_HELP, KERNEL_HELP_NO_RAW, KERNEL_HELP_RAW_OK};

#[derive(Parser)]
#[command(name = "cargo-ktstr", bin_name = "cargo")]
pub(crate) struct Cargo {
    #[command(subcommand)]
    pub(crate) command: CargoSub,
}

#[derive(Subcommand)]
pub(crate) enum CargoSub {
    /// ktstr dev workflow: build kernel + run tests.
    Ktstr(Ktstr),
}

#[derive(Parser)]
pub(crate) struct Ktstr {
    #[command(subcommand)]
    pub(crate) command: KtstrCommand,
}

// clap's derive expands every variant into a struct of `Option<T>` /
// `Vec<T>` per CLI flag; the `PerfDelta` variant's large flag set pushes
// the enum past clippy's large-variant heuristic. The enum is constructed
// once per CLI invocation and dispatched immediately; boxing every variant
// would distort the match ergonomics without measurable benefit.
#[allow(clippy::large_enum_variant)]
#[derive(Subcommand)]
pub(crate) enum KtstrCommand {
    /// Build the kernel (if needed) and run tests via cargo nextest.
    #[command(visible_alias = "nextest")]
    Test {
        /// Repeatable. See [`KERNEL_HELP_NO_RAW`] for accepted shapes
        /// (path, version, cache key, range `START..END`, git source
        /// `git+URL#tag=NAME`). Multiple `--kernel` flags fan out the
        /// gauntlet across kernels: each `(test × scenario × topology
        /// × kernel)` tuple becomes a distinct nextest test case so
        /// nextest's parallelism, retries, and `-E` filtering all
        /// apply natively.
        #[arg(long, action = ArgAction::Append, help = KERNEL_HELP_NO_RAW)]
        kernel: Vec<String>,
        /// Disable all performance mode features (flock, pinning, RT
        /// scheduling, hugepages, NUMA mbind, KVM exit suppression).
        /// For shared runners or unprivileged containers.
        /// Also settable via KTSTR_NO_PERF_MODE env var.
        #[arg(long)]
        no_perf_mode: bool,
        /// Promote hardware-driven test SKIPS to hard failures.
        /// `ResourceContention` (no LLC slot currently free / KVM fd
        /// budget exhausted -- transient), `TopologyInsufficient`
        /// (the VM can't boot on this host), and `PerfModeUnavailable`
        /// (performance_mode on a too-small host) skips become exit 1
        /// instead of silent passes. (Only an explicit cpu-budget the
        /// host can't satisfy, `CpuBudgetUnsatisfiable`, is an
        /// unconditional hard error NOT gated by this flag.) For CI
        /// environments
        /// where the hardware IS expected to support every test —
        /// a skip means the CI config is wrong, not that the test
        /// is inapplicable. Exports `KTSTR_NO_SKIP_MODE=1`.
        #[arg(long)]
        no_skip_mode: bool,
        /// Build and run tests with the release profile
        /// (`--cargo-profile release` to nextest).
        ///
        /// Release mode uses STRICTER assertion thresholds
        /// (`gap_threshold_ms` 2000 vs debug's 3000, `spread_threshold_pct`
        /// 15% vs debug's 35%) — tests that barely pass in debug may
        /// fail under `--release`. `catch_unwind`-based tests are
        /// skipped because release sets `panic = "abort"` (see
        /// `Cargo.toml [profile.release]`). Tests gated on
        /// `#[cfg(debug_assertions)]` also skip.
        ///
        /// `--release` builds the harness/test binary with the release
        /// profile. The scheduler-under-test builds release by default
        /// regardless of this flag; override its profile with
        /// `--profile <NAME>`.
        #[arg(long)]
        release: bool,
        /// Cargo BUILD profile for the scheduler-under-test (a
        /// `SchedulerSpec::Discover` package): drives `cargo build -p
        /// <scheduler> --profile <NAME>` via `KTSTR_SCHEDULER_PROFILE`.
        /// Omitted, the scheduler builds `release` — an optimized
        /// scheduler is the only sensible default. Pass `--profile dev`
        /// for a fast unoptimized scheduler build, or any custom profile
        /// from `Cargo.toml`. Independent of the harness `--release`.
        #[arg(long)]
        profile: Option<String>,
        /// NEXTEST test profile (`.config/nextest.toml`), forwarded to
        /// nextest as `--profile <NAME>`. Selects retry / timeout /
        /// output settings for the run. Distinct from `--profile` (the
        /// scheduler's cargo BUILD profile) and `--release` (the
        /// harness's cargo build profile).
        #[arg(long)]
        nextest_profile: Option<String>,
        /// Include EOL stable series in a `--kernel START..END` range
        /// expansion (shared `INCLUDE_EOL_HELP`). No effect on a single
        /// `--kernel`, a path, a cache key, or a git source.
        #[arg(long, help = INCLUDE_EOL_HELP)]
        include_eol: bool,
        /// Narrow the run to only the tests a change touches: build +
        /// introspect each declared scheduler, attribute the `base..HEAD`
        /// diff UNIONed with the working tree (uncommitted + untracked) to
        /// schedulers, and run only those schedulers' tests (ANDed with any
        /// `-E` in the passthrough args). A broad / build-graph /
        /// unattributable change runs everything (fail-safe); a strictly
        /// docs-only change (or a clean tree at `base`) runs nothing. See
        /// `cargo ktstr affected` for the same attribution rendered as a CI
        /// matrix.
        #[arg(long)]
        relevant: bool,
        /// With `--relevant`: override the baseline commit directly (skips
        /// merge-base). Ignored without `--relevant`.
        #[arg(long)]
        base: Option<String>,
        /// With `--relevant`: ref to merge-base against. Defaults to
        /// `$GITHUB_BASE_REF` (as `origin/<ref>`) on a PR, else
        /// `--default-branch`. Ignored without `--relevant`.
        #[arg(long)]
        base_ref: Option<String>,
        /// With `--relevant`: branch to merge-base against when neither
        /// `--base` / `--base-ref` nor `$GITHUB_BASE_REF` is set. Ignored
        /// without `--relevant`.
        #[arg(long, default_value = "main")]
        default_branch: String,
        /// Arguments passed through to cargo nextest run. Native flags
        /// may appear in any order relative to these (no `--` separator
        /// needed); to forward a token that shares a name with a native
        /// flag (e.g. nextest's own `--profile`), place it after a `--`.
        #[arg(last = true)]
        args: Vec<String>,
    },
    /// Build the kernel (if needed) and run tests with coverage via
    /// cargo llvm-cov nextest. For other llvm-cov subcommands
    /// (`report`, `clean`, `show-env`), use `cargo ktstr llvm-cov`.
    Coverage {
        /// Repeatable. Same shapes and multi-kernel semantics as
        /// `cargo ktstr test --kernel`: each (test × kernel) variant
        /// runs as its own nextest subprocess so cargo-llvm-cov
        /// merges every variant's profraw automatically.
        #[arg(long, action = ArgAction::Append, help = KERNEL_HELP_NO_RAW)]
        kernel: Vec<String>,
        /// Disable all performance mode features (flock, pinning, RT
        /// scheduling, hugepages, NUMA mbind, KVM exit suppression).
        /// For shared runners or unprivileged containers.
        /// Also settable via KTSTR_NO_PERF_MODE env var.
        #[arg(long)]
        no_perf_mode: bool,
        /// Promote hardware-driven test skips to hard failures.
        /// See `cargo ktstr test --no-skip-mode` for the full
        /// contract. Exports `KTSTR_NO_SKIP_MODE=1`.
        #[arg(long)]
        no_skip_mode: bool,
        /// Build and collect coverage with the release profile
        /// (`--cargo-profile release` to llvm-cov nextest).
        ///
        /// Release mode uses STRICTER assertion thresholds
        /// (`gap_threshold_ms` 2000 vs debug's 3000, `spread_threshold_pct`
        /// 15% vs debug's 35%) — tests that barely pass in debug may
        /// fail under `--release`. `catch_unwind`-based tests are
        /// skipped because release sets `panic = "abort"`.
        #[arg(long)]
        release: bool,
        /// Cargo BUILD profile for the scheduler-under-test (see `cargo
        /// ktstr test --profile`). Omitted, the scheduler builds
        /// `release`. Independent of the harness `--release`.
        #[arg(long)]
        profile: Option<String>,
        /// NEXTEST test profile forwarded to `cargo llvm-cov nextest` as
        /// `--profile <NAME>` (see `cargo ktstr test --nextest-profile`).
        #[arg(long)]
        nextest_profile: Option<String>,
        /// Include EOL stable series in a `--kernel START..END` range
        /// expansion (shared `INCLUDE_EOL_HELP`). No effect on a single
        /// `--kernel`, a path, a cache key, or a git source.
        #[arg(long, help = INCLUDE_EOL_HELP)]
        include_eol: bool,
        /// Narrow the run to only the tests a change touches (see `cargo
        /// ktstr test --relevant` for the full contract). The relevant
        /// filter is ANDed with any `-E` in the passthrough args; a broad /
        /// unattributable change runs everything, a docs-only change runs
        /// nothing.
        #[arg(long)]
        relevant: bool,
        /// With `--relevant`: override the baseline commit directly (skips
        /// merge-base). Ignored without `--relevant`.
        #[arg(long)]
        base: Option<String>,
        /// With `--relevant`: ref to merge-base against. Defaults to
        /// `$GITHUB_BASE_REF` (as `origin/<ref>`) on a PR, else
        /// `--default-branch`. Ignored without `--relevant`.
        #[arg(long)]
        base_ref: Option<String>,
        /// With `--relevant`: branch to merge-base against when neither
        /// `--base` / `--base-ref` nor `$GITHUB_BASE_REF` is set. Ignored
        /// without `--relevant`.
        #[arg(long, default_value = "main")]
        default_branch: String,
        /// Arguments passed through to cargo llvm-cov nextest. Native
        /// flags may appear in any order relative to these (no `--`
        /// separator needed); to forward a token that shares a name with
        /// a native flag (e.g. nextest's own `--profile`), place it after
        /// a `--`.
        #[arg(last = true)]
        args: Vec<String>,
    },
    /// Run `cargo llvm-cov` with arbitrary arguments.
    ///
    /// When you want `cargo llvm-cov nextest`, prefer `cargo ktstr
    /// coverage` — this subcommand is the raw passthrough for
    /// `llvm-cov` invocations that don't fit the coverage flow
    /// (e.g. `report`, `clean`, `show-env`).
    ///
    /// Note: bare `cargo ktstr llvm-cov` (no subcommand) dispatches
    /// to `cargo llvm-cov` which runs `cargo test` — not useful for
    /// ktstr tests. Always pass a subcommand.
    LlvmCov {
        /// Repeatable. Same shapes and multi-kernel semantics as
        /// `cargo ktstr test --kernel`. Profraw aggregation across
        /// kernel variants happens inside cargo-llvm-cov; this raw-
        /// passthrough hands every other argument to the user's
        /// chosen llvm-cov subcommand.
        #[arg(long, action = ArgAction::Append, help = KERNEL_HELP_NO_RAW)]
        kernel: Vec<String>,
        /// Disable all performance mode features (flock, pinning, RT
        /// scheduling, hugepages, NUMA mbind, KVM exit suppression).
        /// For shared runners or unprivileged containers.
        /// Also settable via KTSTR_NO_PERF_MODE env var.
        #[arg(long)]
        no_perf_mode: bool,
        /// Promote hardware-driven test skips to hard failures.
        /// See `cargo ktstr test --no-skip-mode` for the full
        /// contract. Exports `KTSTR_NO_SKIP_MODE=1`.
        #[arg(long)]
        no_skip_mode: bool,
        /// Include EOL stable series in a `--kernel START..END` range
        /// expansion (shared `INCLUDE_EOL_HELP`). No effect on a single
        /// `--kernel`, a path, a cache key, or a git source.
        #[arg(long, help = INCLUDE_EOL_HELP)]
        include_eol: bool,
        /// Arguments passed through to cargo llvm-cov. Native flags may
        /// appear in any order relative to these (no `--` separator
        /// needed); to forward a token that shares a name with a native
        /// flag (e.g. nextest's own `--profile`), place it after a `--`.
        #[arg(last = true)]
        args: Vec<String>,
    },
    /// Print sidecar analysis from the most recent test run.
    ///
    /// Reads sidecar JSON files from the newest subdirectory under
    /// `{CARGO_TARGET_DIR or "target"}/ktstr/` (overridable with
    /// `KTSTR_SIDECAR_DIR`) and prints gauntlet analysis, BPF
    /// verifier stats, callback profile, and KVM stats. Test runs
    /// are partitioned into `{kernel}-{project_commit}` subdirectories,
    /// where `{project_commit}` is the project HEAD short hex with
    /// `-dirty` when the worktree differs; each subdirectory is
    /// the baseline snapshot of the most recent run at that
    /// (kernel, project commit) pair (re-running at the same key
    /// pre-clears prior sidecars before writing the new run).
    ///
    /// Single-run inspection (list / list-metrics / list-values /
    /// show-host / explain-sidecar); `cargo ktstr perf-delta` diffs two
    /// commits.
    Stats {
        #[command(subcommand)]
        command: Option<StatsCommand>,
    },
    /// Re-run the failing subset of a prior sidecar pool.
    ///
    /// Scans the sidecar root for failed runs (`!passed && !skipped`),
    /// dedupes the resulting test names, and emits a `cargo nextest
    /// run`-compatible filter expression that targets exactly that
    /// subset. Default is dry-run (prints the filter expression to
    /// stdout); pass `--exec` to invoke nextest directly.
    ///
    /// Distinct from the in-VM auto-repro (`auto_repro = true` on
    /// `KtstrTestEntry`) which fires within the same test process
    /// when a primary run fails — `replay` is post-hoc, after the
    /// test process has exited, for the CI-friendly "re-run last
    /// session's failures against the new code" workflow.
    Replay {
        /// Override the sidecar root. Defaults to
        /// `test_support::runs_root()` (typically `target/ktstr/`).
        /// Same semantics as `cargo ktstr stats show-host --dir` and
        /// `cargo ktstr stats list-values --dir`: useful when
        /// inspecting an archived sidecar tree copied off a CI host.
        #[arg(long)]
        dir: Option<std::path::PathBuf>,
        /// Narrow the failed-sidecar selection by substring match
        /// on `test_name`. Case-sensitive. Useful when re-running
        /// only a specific suite under a known regression class.
        #[arg(long, short = 'E')]
        filter: Option<String>,
        /// Invoke `cargo nextest run -E <filter>` instead of
        /// printing the filter expression. Without `--exec`, the
        /// command is dry-run: the printed filter can be piped
        /// into nextest by hand, or pasted into a CI pipeline,
        /// before committing to the re-run.
        #[arg(long)]
        exec: bool,
        /// Cargo BUILD profile for the scheduler-under-test (see `cargo
        /// ktstr test --profile`). Omitted, the scheduler builds
        /// `release`. Only meaningful with `--exec` (the dry-run path
        /// runs nothing).
        #[arg(long)]
        profile: Option<String>,
        /// NEXTEST test profile forwarded to the re-run `cargo nextest
        /// run` as `--profile <NAME>` (see `cargo ktstr test
        /// --nextest-profile`). Only meaningful with `--exec`.
        #[arg(long)]
        nextest_profile: Option<String>,
        /// cargo/nextest flags forwarded verbatim to `cargo nextest run`
        /// when `--exec` re-runs the failed tests (`--features …`,
        /// `--cargo-profile …`). Native flags may appear in ANY order
        /// relative to these — the argv split (see the `argsplit` module)
        /// routes each token to ktstr or the passthrough by name, so a
        /// `--` separator is not required. To forward a token that shares
        /// a name with a native flag (e.g. this command's own `-E` /
        /// `--profile`), place it after a `--`. Ignored on the dry-run
        /// path, which prints only the filter expression.
        #[arg(last = true)]
        args: Vec<String>,
    },
    /// Compare `performance_mode` test metrics between HEAD and a
    /// baseline commit, exiting non-zero when a metric regresses past
    /// its threshold. Resolves the baseline (branch merge-base, a PR
    /// target via `$GITHUB_BASE_REF`, or an explicit `--base`), then
    /// either compares already-pooled sidecars or (with `--dual-run`)
    /// produces both commits' runs first and compares them, reusing the
    /// shared `compare_partitions` regression engine to pair on scenario and emit
    /// the verdict.
    PerfDelta {
        /// Override the baseline commit directly (skips merge-base). The
        /// testability / cached-baseline knob: diff HEAD against any
        /// chosen commit without a real branch divergence.
        #[arg(long)]
        base: Option<String>,
        /// Override the ref to diverge from (merge-base against it).
        /// Defaults to `$GITHUB_BASE_REF` (as `origin/<ref>`) on a PR,
        /// else `--default-branch`.
        #[arg(long)]
        base_ref: Option<String>,
        /// Nextest `-E` filter narrowing within the `performance_mode`
        /// test set the run selects by default.
        #[arg(long, short = 'E')]
        filter: Option<String>,
        /// Narrow the compared `performance_mode` set to only the tests the
        /// `base..HEAD` diff (UNIONed with the working tree) touches, ANDed
        /// with `--filter`. Reuses `--base` / `--base-ref` /
        /// `--default-branch` as BOTH the comparison baseline and the
        /// attribution base. A broad / unattributable change compares
        /// everything (fail-safe); a docs-only change compares nothing.
        #[arg(long)]
        relevant: bool,
        /// Branch to merge-base against when neither `--base` /
        /// `--base-ref` nor `$GITHUB_BASE_REF` is set.
        #[arg(long, default_value = "main")]
        default_branch: String,
        /// Kernel the baseline + HEAD `performance_mode` runs boot.
        /// Required with `--dual-run`; same `--kernel <SPEC>` form as
        /// `cargo ktstr test`. Unused on the cached-baseline path.
        #[arg(long)]
        kernel: Option<String>,
        /// Produce both commits' runs before comparing: check the
        /// baseline out in a detached git worktree and run its
        /// `performance_mode` tests there, run HEAD's in the working
        /// tree, then compare. Without it, compares sidecars already
        /// pooled from a prior run or a downloaded CI artifact.
        #[arg(long)]
        dual_run: bool,
        /// Uniform relative significance threshold in percent (e.g. 10
        /// for 10%), overriding every metric's registry default — the
        /// knob a CI perf-gate tightens or loosens. Sugar for a
        /// `--policy` with `{default_percent: N}`. Mutually exclusive
        /// with `--policy`.
        #[arg(long, conflicts_with = "policy")]
        threshold: Option<f64>,
        /// Path to a JSON `ComparisonPolicy` with per-metric thresholds:
        /// `{ "default_percent": <f64>, "per_metric_percent": { "<metric>":
        /// <f64>, ... } }` (metric names from `cargo ktstr stats
        /// list-metrics`). Mutually exclusive with `--threshold`.
        #[arg(long, conflicts_with = "threshold")]
        policy: Option<std::path::PathBuf>,
        /// Self-tuning noise mode: run each side N times and decide significance
        /// from whether the two sides are SEPARATED — a Welch two-sample t-test,
        /// or fully disjoint `[min, max]` bands — AND the delta is MATERIAL (the
        /// registry dual-gate), instead of a fixed `--threshold`. A high per-side
        /// spread (over `--noise-spread-threshold`) is an ADVISORY annotation
        /// only and never suppresses a confident regression. Implies the dual-run
        /// production, looped N times; commit axis only (needs `--kernel`). N must
        /// be >= 2: variance and the Welch test need at least two runs per side
        /// (>= 5 recommended for a well-powered test).
        #[arg(
            long,
            value_name = "N",
            value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(2..),
            conflicts_with_all = ["threshold", "policy", "dual_run"],
        )]
        noise_adjust: Option<usize>,
        /// Per-side relative-spread limit in percent above which `--noise-adjust`
        /// adds an ADVISORY "noisy spread" annotation to a metric's row (default
        /// 5.0). Advisory only — never suppresses a verdict. Requires
        /// `--noise-adjust`.
        #[arg(long, requires = "noise_adjust")]
        noise_spread_threshold: Option<f64>,
        /// Suppress the `--noise-adjust` per-phase spread block
        /// entirely. Requires `--noise-adjust`; mutually exclusive
        /// with every other phase flag.
        #[arg(
            long = "no-phases",
            help_heading = "Phase rendering",
            conflicts_with_all = ["phases_only", "steps_only", "phase", "phase_threshold"],
            requires = "noise_adjust",
        )]
        no_phases: bool,
        /// Show ONLY the `--noise-adjust` per-phase spread block;
        /// suppress the aggregate spread table and footer. Requires
        /// `--noise-adjust`. Composes with `--steps-only`, `--phase`,
        /// and `--phase-threshold`.
        #[arg(
            long = "phases-only",
            help_heading = "Phase rendering",
            conflicts_with = "no_phases",
            requires = "noise_adjust"
        )]
        phases_only: bool,
        /// Within the `--noise-adjust` per-phase block, suppress the
        /// BASELINE bucket (`step_index == 0`). Requires
        /// `--noise-adjust`; mutually exclusive with `--phase`.
        #[arg(
            long = "steps-only",
            help_heading = "Phase rendering",
            conflicts_with_all = ["no_phases", "phase"],
            requires = "noise_adjust",
        )]
        steps_only: bool,
        /// Show only the named per-phase index in the `--noise-adjust`
        /// per-phase block. `0` selects BASELINE; `1..=N` select
        /// scenario Step ordinals. Requires `--noise-adjust`; mutually
        /// exclusive with `--steps-only`.
        #[arg(
            long = "phase",
            help_heading = "Phase rendering",
            conflicts_with_all = ["no_phases", "steps_only"],
            requires = "noise_adjust",
        )]
        phase: Option<u16>,
        /// Per-row relative-spread gate for the `--noise-adjust`
        /// per-phase block: suppress paired rows whose
        /// `|delta-mean| / |a.mean| < PCT / 100.0` (a value from a
        /// ~zero baseline is an unbounded relative change and is
        /// always shown). Independent from `--threshold`. Requires
        /// `--noise-adjust`; mutually exclusive with `--no-phases`.
        #[arg(
            long = "phase-threshold",
            help_heading = "Phase rendering",
            conflicts_with = "no_phases",
            requires = "noise_adjust"
        )]
        phase_threshold: Option<f64>,
        /// Show EVERY compared metric row, including stable (unchanged /
        /// immaterial) and noisy (fewer than 2 usable runs) ones. Default:
        /// only meaningful rows (confident regression / improvement /
        /// informational) print; when every row is suppressed a one-line
        /// summary prints instead of an empty table. Applies to the
        /// `--noise-adjust` aggregate metrics table; the per-phase table is
        /// separately spread-gated by `--phase-threshold` (it shows every
        /// verdict kind). The fixed-threshold table already lists every
        /// changed row plus an unchanged count. Display-only — never affects
        /// the failure gate.
        #[arg(long, help_heading = "Metric rendering")]
        all_metrics: bool,
        /// Fail the run only when AT LEAST N metrics regress (default 5, so
        /// a handful of one-off noisy regressions does not flip CI red; pass
        /// `--fail-threshold 1` for fail-on-any). N = 0 never fails on the
        /// count — only `--must-fail` can then fail. Counts confident
        /// regressions; suppressed rows still count.
        #[arg(long, value_name = "N", help_heading = "Failure gating")]
        fail_threshold: Option<usize>,
        /// Comma-separated metric registry names (from `cargo ktstr stats
        /// list-metrics`) that fail the run if ANY of them regresses,
        /// regardless of `--fail-threshold` (ORed on top of the count
        /// gate). Names that could never fire the gate are rejected up front
        /// so one cannot silently disarm it: unknown names; internal rate
        /// components; per-phase-only metrics (their value never reaches the
        /// aggregate comparison, in either mode — assert them per-phase in
        /// the test instead); and — WITHOUT `--noise-adjust` — whole-run
        /// distribution metrics (read only per-run, so add `--noise-adjust`
        /// to gate on one) and informational metrics (registry polarity has
        /// no regression direction, so they never classify as a regression
        /// on the default comparison). An informational metric IS accepted
        /// under `--noise-adjust`, where a per-test direction override can
        /// classify it as a regression. Under `--noise-adjust`, a listed
        /// metric that classifies NOISY (a side had fewer than 2 usable runs)
        /// is reported but does NOT fail — raise `--noise-adjust N` for a
        /// trustworthy verdict.
        #[arg(long, value_name = "M1,M2,...", help_heading = "Failure gating")]
        must_fail: Option<String>,
        /// Cargo BUILD profile for the scheduler-under-test on BOTH
        /// sides' `cargo ktstr test` (see `cargo ktstr test --profile`).
        /// Omitted, the scheduler builds `release`. Only meaningful on the
        /// dual-run / noise-adjust production path (the cached-baseline
        /// path runs nothing).
        #[arg(long)]
        profile: Option<String>,
        /// NEXTEST test profile forwarded to BOTH sides' `cargo ktstr
        /// test` as `--nextest-profile <NAME>` (see `cargo ktstr test
        /// --nextest-profile`). Only meaningful on the dual-run /
        /// noise-adjust production path.
        #[arg(long)]
        nextest_profile: Option<String>,
        /// cargo/nextest flags forwarded verbatim to BOTH sides'
        /// `cargo ktstr test` on the dual-run / noise-adjust production
        /// path (e.g. `--features integration,wprof`). Native flags may
        /// appear in ANY order relative to these (the argv split routes
        /// by name), so no `--` separator is required; to forward a token
        /// that shares a name with a native flag (e.g. `-E` / `--profile`),
        /// place it after a `--`.
        #[arg(last = true)]
        args: Vec<String>,
    },
    /// Manage cached kernel images.
    Kernel {
        #[command(subcommand)]
        command: KernelCommand,
    },
    /// Collect BPF verifier statistics for declared schedulers.
    ///
    /// Spawns `cargo nextest run -E 'test(/^verifier/) & !test(/^verifier::/)'`
    /// (waited on via `Command::status()`, not `execvp`) — the filter
    /// matches the `verifier/<sched>/<kernel>/<preset>` cells but excludes
    /// the verifier module's own `verifier::tests::*` unit tests. Each
    /// test binary that links ktstr-test-support and has at least one
    /// `declare_scheduler!` declaration emits one nextest test per
    /// (declared scheduler × declared kernel × accepted topology preset)
    /// cell — the sweep runs each scheduler ACROSS topologies, because
    /// whether it attaches and dispatches is topology-dependent (a
    /// scheduler can attach on one topology and wedge on another). Each
    /// cell boots a VM on the topology named in the cell, loads the
    /// scheduler's BPF programs (the real kernel verifier runs), and
    /// reports per-program verified-instruction counts via host-side
    /// memory introspection.
    /// A cell PASSes only when the scheduler verifies (BPF loads),
    /// attaches (the in-guest gate confirms `/sys/kernel/sched_ext/state`
    /// reached `enabled`), AND dispatches an injected SpinWait workload
    /// (the guest confirms a worker made forward progress after attach).
    /// A scheduler that loads but never attaches, or attaches but never
    /// dispatches a runnable task, FAILs.
    /// Every cell boots with performance mode disabled (`verified_insns`
    /// is perf-mode-independent), so it takes only a shared (LOCK_SH) LLC
    /// reservation and parallel cells no longer starve each other on the
    /// LLC lock (a `performance_mode` peer's LOCK_EX can still defer a
    /// cell, resolved by nextest retry). Eevdf + KernelBuiltin scheduler
    /// variants are skipped at cell-emission time (no userspace binary to
    /// verify). `--scheduler <NAME>` restricts the sweep to a single
    /// declared scheduler across topologies. After nextest finishes, one
    /// `verified_insns` table per scheduler (rows = kernel, cols = BPF
    /// program) and a topology × scheduler PASS/FAIL grid are printed.
    ///
    /// The `declare_scheduler!` verifier cells carry no `required-features`,
    /// so they build without a feature flag — no `--features` passthrough
    /// is needed for the sweep to find them. Trailing args are forwarded
    /// verbatim to nextest (a filterset, `--cargo-profile`, ...). The
    /// scheduler-under-test builds release by default.
    Verifier {
        /// Repeatable. See [`KERNEL_HELP_NO_RAW`] for accepted shapes
        /// (path / version / cache key / range / git source). Overrides
        /// the per-scheduler declared `kernels` set when supplied.
        #[arg(long, action = ArgAction::Append, help = KERNEL_HELP_NO_RAW)]
        kernel: Vec<String>,
        /// Print raw verifier output without formatting.
        #[arg(long)]
        raw: bool,
        /// Cargo BUILD profile for the scheduler-under-test (see `cargo
        /// ktstr test --profile`). Omitted, the scheduler builds
        /// `release`. Sets `KTSTR_SCHEDULER_PROFILE` for the inner
        /// `cargo nextest run`.
        #[arg(long)]
        profile: Option<String>,
        /// NEXTEST test profile forwarded to the inner `cargo nextest
        /// run` as `--profile <NAME>` (see `cargo ktstr test
        /// --nextest-profile`).
        #[arg(long)]
        nextest_profile: Option<String>,
        /// Restrict the sweep to a single declared scheduler by name
        /// (the `declare_scheduler!` `name`). Omitted, every declared
        /// scheduler is swept across topologies. The name must be a
        /// declared BPF scheduler (`binary` / `binary_path`); `eevdf`
        /// and `kernel_builtin` schedulers have no BPF to verify and are
        /// never emitted, so naming one matches no cell.
        #[arg(long)]
        scheduler: Option<String>,
        /// Include EOL stable series in a `--kernel START..END` range
        /// expansion (shared `INCLUDE_EOL_HELP`). No effect on a single
        /// `--kernel`, a path, a cache key, or a git source.
        #[arg(long, help = INCLUDE_EOL_HELP)]
        include_eol: bool,
        /// cargo/nextest flags forwarded verbatim to the inner
        /// `cargo nextest run` — a nextest filterset, `--cargo-profile`,
        /// etc. Native flags (`--kernel` / `--raw` / `--profile` /
        /// `--nextest-profile`) may appear in ANY order relative to these
        /// (the argv split routes by name), so no `--` separator is
        /// required; to forward a token that shares a name with a native
        /// flag (e.g. nextest's own `--profile`), place it after a `--`.
        #[arg(last = true)]
        args: Vec<String>,
    },
    /// Generate shell completions for cargo-ktstr.
    Completions {
        /// Shell to generate completions for.
        shell: clap_complete::Shell,
        /// Binary name for completions.
        #[arg(long, default_value = "cargo")]
        binary: String,
    },
    /// Print the current host context used by sidecar collection:
    /// CPU identity, memory/hugepage config, transparent-hugepage
    /// policy, NUMA node count, kernel uname triple
    /// (sysname/release/machine), kernel cmdline, and every
    /// `/proc/sys/kernel/sched_*` tunable. Useful for diagnosing
    /// cross-run regressions that trace back to host-context drift
    /// (sysctl change, THP policy flip, hugepage reservation).
    ///
    /// For historical drift between archived runs, use
    /// `cargo ktstr perf-delta` — its host-delta section
    /// reports which host-context fields changed between run A
    /// and run B using the same [`ktstr::host_context::HostContext::diff`] logic.
    ShowHost,
    /// Print the resolved assertion thresholds for the named test.
    ///
    /// Dumps the merged `Assert` produced by the runtime merge chain
    /// `Assert::default_checks().merge(&entry.scheduler.assert).merge(&entry.assert)`
    /// — the same value `run_ktstr_test_inner` evaluates against
    /// worker reports. Surfaces every threshold field (or `none`
    /// when inherited / unset) so an operator can see what the test
    /// will actually check against without reading source or
    /// guessing which layer contributed each bound.
    ///
    /// Fails with an actionable message when no registered test
    /// matches the given name. Use `cargo nextest list` to
    /// enumerate test names — then pass just the FUNCTION-NAME
    /// component to `show-thresholds`, not the `<binary>::`
    /// prefix that nextest prepends to each line. The
    /// `#[ktstr_test]` registry keys on the bare function name,
    /// so `ktstr::preempt_regression_fault_under_load` (as
    /// printed by nextest) must be trimmed to
    /// `preempt_regression_fault_under_load` before it resolves.
    ShowThresholds {
        /// Function-name-only test identifier as registered in
        /// `#[ktstr_test]` (e.g. `preempt_regression_fault_under_load`).
        /// Do NOT include the `<binary>::` prefix that
        /// `cargo nextest list` prepends — strip it before
        /// invoking this command.
        test: String,
    },
    /// Emit the scheduler packages a `base..HEAD` diff affects, as a flat JSON
    /// array for a GitHub Actions dynamic matrix
    /// (`strategy.matrix.scheduler: ${{ fromJSON(...) }}` — one job per
    /// scheduler). Attributes changed paths via the cargo dependency closure
    /// and — only when a native (C/BPF) or unattributable path changed — via
    /// each scheduler's cargo dep-info (building it once to read that). A broad
    /// / build-graph / unattributable change emits every testable scheduler
    /// (fail-safe); a strictly docs-only change emits `[]`. Only Discover
    /// (cargo-package) schedulers appear; package-less schedulers (EEVDF etc.)
    /// have no matrix cell and must run in a separate unconditional CI leg.
    Affected {
        /// Override the baseline commit directly (skips merge-base).
        #[arg(long)]
        base: Option<String>,
        /// Ref to merge-base against. Defaults to `$GITHUB_BASE_REF` (as
        /// `origin/<ref>`) on a PR, else `--default-branch`.
        #[arg(long)]
        base_ref: Option<String>,
        /// Branch to merge-base against when neither `--base` / `--base-ref`
        /// nor `$GITHUB_BASE_REF` is set.
        #[arg(long, default_value = "main")]
        default_branch: String,
    },
    /// Export a registered test as a self-extracting `.run` file
    /// that reproduces the scenario on bare metal without a VM.
    ///
    /// Bundles the running ktstr binary, the scheduler binary, and
    /// every include file the test declares into a gzipped tarball
    /// embedded in a bash preamble. The preamble validates root
    /// access, sched_ext support, cgroup2 mount, sched_ext-conflict
    /// (no other scheduler attached), and topology compatibility
    /// before extracting and launching. Chmod +x on the output so
    /// the operator can execute the `.run` directly.
    ///
    /// The frozen bits (scheduler choice, scheduler args, topology)
    /// match the test as registered. Overridable on the target host:
    /// `--duration`, `--watchdog-timeout`, `--quiet` (suppress
    /// banner). NOT overridable: `--cpus`, `--topology`, `--affinity`
    /// — re-export to change those.
    ///
    /// Out of scope for v1: `host_only` tests (they orchestrate
    /// cargo / nested VMs from inside the test body), tests with
    /// `bpf_map_write` (need the framework's host-side runtime
    /// probe surface), and `KernelBuiltin` schedulers (need the
    /// `enable` / `disable` shell commands the preamble doesn't
    /// emit yet). All three are rejected with actionable errors.
    ///
    /// # Name collisions
    ///
    /// If multiple workspace test binaries register a
    /// `#[ktstr_test]` with the same name, the router visits
    /// candidates in alphabetical order by absolute binary path
    /// and the FIRST binary that admits the test wins. Use
    /// `--package` to scope the search to a specific package and
    /// disambiguate deterministically.
    Export {
        /// Function-name-only test identifier as registered in
        /// `#[ktstr_test]` (e.g. `preempt_regression_fault_under_load`).
        /// Strip the `<binary>::` prefix that
        /// `cargo nextest list` prepends — the registry keys on the
        /// bare function name.
        test: String,
        /// Output path for the `.run` file. Defaults to
        /// `<test>.run` in the current directory.
        #[arg(short = 'o', long = "output")]
        output: Option<PathBuf>,
        /// Restrict the workspace search to a specific package. When
        /// omitted, every workspace member's tests is built and
        /// scanned for a matching `#[ktstr_test]` registration.
        /// Pass-through to `cargo build --tests --package <NAME>`.
        #[arg(short = 'p', long)]
        package: Option<String>,
        /// Build the test binaries with the release profile.
        /// Stricter assertion thresholds and `panic = "abort"` —
        /// match the profile the operator will run the .run file
        /// under, otherwise the embedded binary's behavior may
        /// drift from the dev-profile test runs the operator
        /// reproduced from.
        #[arg(long)]
        release: bool,
    },
    /// Enumerate every ktstr flock held on this host.
    ///
    /// Troubleshooting companion for `--cpu-cap` contention. Scans
    /// `{KTSTR_LOCK_DIR}/ktstr-llc-*.lock`,
    /// `{KTSTR_LOCK_DIR}/ktstr-cpu-*.lock` (default `/tmp`), and
    /// `{cache_root}/.locks/*.lock`, cross-referenced against
    /// `/proc/locks` via [`ktstr::cli::list_locks`] to name the holder
    /// process (PID + cmdline) for each held lock. Read-only — does
    /// NOT attempt any flock acquire.
    Locks {
        /// Emit the snapshot as JSON (compact object under --watch,
        /// pretty-printed otherwise). Stable field names; schema
        /// documented at [`ktstr::cli::list_locks`].
        #[arg(long)]
        json: bool,
        /// Redraw the snapshot on the given interval until SIGINT.
        /// Value is parsed by `humantime`: `100ms`, `1s`, `5m`, `1h`.
        /// Human output clears and redraws in place; `--json` emits
        /// one line-terminated object per interval (ndjson-style).
        #[arg(long, value_parser = humantime::parse_duration)]
        watch: Option<std::time::Duration>,
    },
    /// Boot an interactive shell in a KVM virtual machine.
    ///
    /// Launches a VM with busybox and drops into a shell. Files and
    /// directories passed via -i are available at `/include-files/<name>`
    /// inside the guest. Directories are walked recursively, preserving
    /// structure. Dynamically-linked ELF binaries get automatic shared
    /// library resolution via ELF DT_NEEDED parsing.
    Shell {
        #[arg(long, help = KERNEL_HELP_RAW_OK)]
        kernel: Option<String>,
        /// Resolve topology + memory + extra-include-files from the
        /// named `#[ktstr_test]` registration. Probes each test
        /// binary under the workspace's `cargo build --tests` set
        /// for an entry whose `KtstrTestEntry::name == <NAME>`;
        /// ambiguous names (same NAME registered in two binaries)
        /// bail with a list of the matching binaries. The shell VM
        /// boots with the test's topology axes (numa_nodes, llcs,
        /// cores, threads), the test's memory_mib (with the wprof
        /// memory floor when the `wprof` feature is enabled and
        /// `entry.wprof` is set), and the union of the
        /// test's `extra_include_files` with operator-supplied
        /// `-i` flags. Before VM boot, prints a one-line banner to
        /// stderr naming the test + scheduler so an operator can
        /// repro the workload manually after staging debugging
        /// tools. (PS1-in-guest is a follow-up.) Mutually exclusive
        /// with `--topology` and `--memory-mib`; `-i` is additive.
        /// Note: this v1 resolves topology and extra_include_files
        /// only; the scheduler binary itself is NOT auto-staged
        /// into the guest — copy it with `-i` if you need to run
        /// it (see the banner printed on boot).
        #[arg(long, conflicts_with_all = ["topology", "memory_mib"])]
        test: Option<String>,
        /// Virtual topology as "numa_nodes,llcs,cores,threads".
        #[arg(long, default_value = "1,1,1,1")]
        topology: String,
        /// Files or directories to include in the guest. Repeatable.
        #[arg(short = 'i', long = "include-files", action = ArgAction::Append)]
        include_files: Vec<PathBuf>,
        /// Guest memory in MiB (minimum 128). When absent, estimated
        /// from payload and include file sizes.
        #[arg(long = "memory-mib", value_parser = clap::value_parser!(u32).range(128..))]
        memory_mib: Option<u32>,
        /// Forward kernel console (COM1/dmesg) to stderr in real-time.
        /// Sets loglevel=7 for verbose kernel output.
        #[arg(long)]
        dmesg: bool,
        /// Run a command in the VM instead of an interactive shell.
        /// The VM exits after the command completes.
        #[arg(long)]
        exec: Option<String>,

        /// Max wall-clock for a `--exec` payload before the VM is
        /// force-killed (a panic-less guest hang otherwise blocks
        /// forever). Parsed by humantime: `30s`, `5m`, `1h`. Ignored
        /// without `--exec`. Must exceed guest boot (a few seconds): a
        /// near-zero value force-kills before the payload runs.
        #[arg(long, value_parser = humantime::parse_duration, default_value = "120s")]
        exec_timeout: std::time::Duration,

        /// Disable all performance mode features (flock, pinning, RT
        /// scheduling, hugepages, NUMA mbind, KVM exit suppression).
        /// For shared runners or unprivileged containers.
        /// Also settable via KTSTR_NO_PERF_MODE env var.
        #[arg(long)]
        no_perf_mode: bool,

        /// Reserve only N host CPUs for the shell VM. Requires
        /// `--no-perf-mode` — perf-mode already holds every LLC
        /// exclusively, so capping under perf-mode would
        /// double-reserve. See `ktstr::cli::CPU_CAP_HELP` for the
        /// full contract.
        #[arg(long, requires = "no_perf_mode", help = ktstr::cli::CPU_CAP_HELP)]
        cpu_cap: Option<usize>,

        #[arg(long, help = ktstr::cli::DISK_HELP)]
        disk: Option<String>,
    },
}

#[derive(Subcommand)]
pub(crate) enum StatsCommand {
    /// List test runs under `{CARGO_TARGET_DIR or "target"}/ktstr/`.
    List,
    /// List the registered regression metrics and their default
    /// thresholds.
    ///
    /// Enumerates the `ktstr::stats::METRICS` registry: metric name,
    /// polarity (higher/lower better), default absolute-delta gate,
    /// default relative-delta gate, display unit, and a one-line
    /// description. Use this to see which metric names
    /// `ComparisonPolicy.per_metric_percent` keys can reference, and
    /// what each default_abs / default_rel gate starts at before an
    /// override.
    ///
    /// Default output is a human-readable table; `--json` emits a
    /// JSON array with the same fields (the row accessor function is
    /// omitted — `#[serde(skip)]` in the registry).
    ListMetrics {
        /// Emit JSON instead of a table.
        #[arg(long)]
        json: bool,
    },
    /// List the distinct values present per filterable dimension in
    /// the sidecar pool.
    ///
    /// Walks every run directory under `runs_root()` (or `--dir`),
    /// pools the sidecars, and reports the set of distinct values
    /// found across all nine filterable dimensions: `kernel`,
    /// `commit`, `kernel_commit`, `source`, `resolve_source`,
    /// `cpu_budget`, `scheduler`, `topology`, and `work_type`. The JSON keys
    /// `commit` and `source` map to the internal
    /// `SidecarResult::project_commit` / `SidecarResult::run_source` fields.
    /// Use this before a `cargo ktstr perf-delta` invocation to discover what
    /// commit / kernel values the pool actually carries — a baseline or
    /// `--kernel` that matches no pooled run fails downstream with "no rows
    /// match", and `list-values` is the upstream answer to "what do I have?".
    ///
    /// Default output renders one block per dimension with values
    /// one per line; `--json` emits a single JSON object keyed by
    /// dimension name. The five optional dimensions (`kernel`,
    /// `commit`, `kernel_commit`, `source`, `resolve_source`) surface absent values
    /// as the textual sentinel `unknown` in the table shape and as
    /// JSON `null` in the JSON shape.
    ListValues {
        /// Emit JSON instead of a per-dimension text block.
        #[arg(long)]
        json: bool,
        /// Alternate run root to walk. Defaults to
        /// `test_support::runs_root()` (typically `target/ktstr/`).
        /// Same semantics as `cargo ktstr stats show-host --dir`: useful when
        /// inspecting archived sidecar trees copied off a CI host.
        #[arg(long)]
        dir: Option<std::path::PathBuf>,
    },
    /// Print the archived host context for a specific run.
    ///
    /// Resolves `--run <id>` against `test_support::runs_root()`
    /// (or `--dir` when set), loads any sidecar file under that
    /// run directory, and renders the `host` field via
    /// `HostContext::format_human`. Useful for inspecting the
    /// CPU model, memory config, THP policy, and sched_* tunables
    /// captured at archive time — the same fingerprint
    /// `compare_partitions` uses for its host-delta section, now
    /// available on a single run.
    ///
    /// Scans sidecars in iteration order and returns the FIRST
    /// sidecar with a populated host field. Every sidecar in a
    /// single run captures the same host, but older pre-
    /// enrichment sidecars may have `host: None`; the forward
    /// scan tolerates those without false-failing as long as at
    /// least one sidecar carries the data. If NO sidecar has a
    /// populated host field, the command fails with an actionable
    /// error naming the likely cause (pre-enrichment run) rather
    /// than silently returning empty output.
    ShowHost {
        /// Run key (e.g. `6.14-abc1234` or `6.14-abc1234-dirty`;
        /// from `cargo ktstr stats list`).
        #[arg(long)]
        run: String,
        /// Alternate run root to resolve `--run` against. Defaults
        /// to `test_support::runs_root()` (typically
        /// `target/ktstr/`). Same semantics as
        /// `cargo ktstr stats show-host --dir`.
        #[arg(long)]
        dir: Option<std::path::PathBuf>,
    },
    /// Diagnose missing optional fields across a run's sidecars.
    ///
    /// Loads every `*.ktstr.json` under `--run <id>` and reports,
    /// per sidecar, which optional fields landed as null along
    /// with the documented reasons each one can be missing. Every
    /// such field carries a classification:
    ///
    /// - `expected` — null is the steady-state shape; no operator
    ///   action recovers it (e.g. payload metadata for a
    ///   scheduler-only test).
    /// - `actionable` — null indicates a recoverable gap;
    ///   re-running in a different environment (in-repo cwd,
    ///   non-tarball kernel, non-host-only test) would populate
    ///   the field.
    ///
    /// Different gauntlet variants on the same run legitimately
    /// differ on which fields populate (host-only vs VM-backed,
    /// scheduler-only vs payload-bearing), so the report is
    /// per-sidecar rather than aggregate.
    ///
    /// Sidecars are loaded verbatim. Diverges intentionally from
    /// `stats list-values` (which rewrite the
    /// `run_source` field to `"archive"` when `--dir` is set):
    /// the override would erase the only signal that surfaces a
    /// pre-rename archive whose `run_source` field was lost on
    /// load. Matches `stats show-host` semantics.
    ///
    /// Default output is per-sidecar text blocks with a header
    /// line reporting walked / parsed counts (so a corrupt
    /// `.ktstr.json` file surfaces as a parse-failure delta
    /// against the file count). Each `None` cause carries an
    /// optional `fix:` line with an operator-actionable
    /// remediation when one applies (e.g. "set KTSTR_KERNEL to
    /// a local kernel source tree" recovers `kernel_commit =
    /// None` for env-unset cases). When the walk encounters
    /// parse failures, the text output appends a trailing
    /// `corrupt sidecars (N):` block listing each corrupt path
    /// with the raw serde error message and (when applicable)
    /// an `enriched:` line with operator-facing remediation
    /// prose for known schema-drift cases. All-corrupt runs
    /// render the header + corrupt-block alone (no per-sidecar
    /// breakdown to render), preserving per-file diagnostic
    /// detail rather than collapsing to a single error line.
    ///
    /// `--json` emits a single object with three top-level
    /// keys: `_schema_version` (string version stamp —
    /// currently `"1"` — that consumers can gate on for
    /// incompatible shape changes), `_walk` (carrying the same
    /// walked / valid counts plus an `errors` array of
    /// `{path, error, enriched_message}` entries covering every
    /// parse failure; `enriched_message` is a JSON string
    /// when a known schema-drift remediation applies, JSON null
    /// otherwise), and `fields` (one entry per optional field
    /// with run-wide `none_count` + `some_count` summing to
    /// `_walk.valid`, plus the static `classification` /
    /// `causes` / `fix` catalog entry; `fix` is a JSON string
    /// when a remediation applies, JSON null otherwise). All-
    /// corrupt runs render the same shape with `valid = 0` and
    /// per-field counts at zero — never bail.
    ///
    /// Exit code is 0 even for all-corrupt runs — the
    /// diagnostic surface is the structured `_walk.errors`
    /// array (or the trailing `corrupt sidecars` text block),
    /// not the process exit code. CI scripts that need to fail
    /// on parse failures must gate on `_walk.valid > 0` or
    /// `_walk.errors.len() == 0` rather than the exit status.
    /// The only non-zero exits are missing-run-directory and
    /// empty-run (zero `.ktstr.json` files).
    ExplainSidecar {
        /// Run key (e.g. `6.14-abc1234` or `6.14-abc1234-dirty`;
        /// from `cargo ktstr stats list`).
        #[arg(long)]
        run: String,
        /// Alternate run root to resolve `--run` against.
        /// Defaults to `target/ktstr/`. Same semantics as
        /// `cargo ktstr stats show-host --dir`.
        #[arg(long)]
        dir: Option<std::path::PathBuf>,
        /// Emit aggregate JSON instead of per-sidecar text. The
        /// text shape is per-sidecar (different gauntlet variants
        /// have different None patterns); the JSON shape is
        /// across-the-run aggregate by field, suitable for
        /// dashboards and CI ingestion.
        #[arg(long)]
        json: bool,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- perf-delta phase-rendering flag mutex constraints --
    //
    // Clap's `conflicts_with` / `conflicts_with_all` annotations on
    // the 5 phase flags (`--no-phases` / `--phases-only` /
    // `--steps-only` / `--phase` / `--phase-threshold`) are
    // load-bearing — they prevent the operator from passing
    // contradictory flag combinations (e.g. `--no-phases` AND
    // `--phase 1` which simultaneously requests "no phase render"
    // and "render only phase 1"). A future refactor that drops
    // one of those annotations silently re-admits the contradiction
    // and the renderer reaches an undefined state. The tests below
    // exercise each documented conflict pair and two negative-case
    // compositions that should parse without error.
    //
    // The phase flags live on the top-level `KtstrCommand::PerfDelta`
    // variant, so the parser entry point is the top-level Cli
    // (`Cargo::try_parse_from(...)`).

    fn argv_perf_delta<'a>(extra: &[&'a str]) -> Vec<&'a str> {
        // Every phase flag `require`s `--noise-adjust` (per-phase output
        // exists only under the noise-adjusted path), so the fixtures carry
        // it; the flag under test then surfaces its conflict (or clean
        // composition) as the sole parse outcome rather than a missing-
        // requirement error.
        let mut v: Vec<&'a str> = vec!["cargo-ktstr", "ktstr", "perf-delta", "--noise-adjust", "3"];
        v.extend_from_slice(extra);
        v
    }

    /// `--no-phases` paired with `--phases-only` is the cleanest
    /// contradiction: suppress the entire phase block AND show
    /// only the phase block. Clap must reject at parse.
    #[test]
    fn perf_delta_phase_flags_no_phases_conflicts_with_phases_only() {
        let argv = argv_perf_delta(&["--no-phases", "--phases-only"]);
        let result = Cargo::try_parse_from(&argv);
        let err = match result {
            Ok(_) => panic!("--no-phases + --phases-only must be rejected"),
            Err(e) => e,
        };
        let msg = err.to_string();
        assert!(
            msg.contains("--no-phases") && msg.contains("--phases-only"),
            "clap error must name both conflicting flags; got: {msg}",
        );
    }

    /// `--no-phases` paired with `--steps-only` contradicts: the
    /// phase block is suppressed entirely, so "show only steps
    /// within the phase block" is nonsensical. Clap must reject.
    #[test]
    fn perf_delta_phase_flags_no_phases_conflicts_with_steps_only() {
        let argv = argv_perf_delta(&["--no-phases", "--steps-only"]);
        assert!(
            Cargo::try_parse_from(&argv).is_err(),
            "--no-phases + --steps-only must be rejected",
        );
    }

    /// `--no-phases` paired with `--phase 1` contradicts: suppressed
    /// block can't show a specific phase. Clap must reject.
    #[test]
    fn perf_delta_phase_flags_no_phases_conflicts_with_phase_filter() {
        let argv = argv_perf_delta(&["--no-phases", "--phase", "1"]);
        assert!(
            Cargo::try_parse_from(&argv).is_err(),
            "--no-phases + --phase must be rejected",
        );
    }

    /// `--no-phases` paired with `--phase-threshold` contradicts:
    /// suppressed block can't apply a row-significance filter.
    /// Clap must reject.
    #[test]
    fn perf_delta_phase_flags_no_phases_conflicts_with_phase_threshold() {
        let argv = argv_perf_delta(&["--no-phases", "--phase-threshold", "5"]);
        assert!(
            Cargo::try_parse_from(&argv).is_err(),
            "--no-phases + --phase-threshold must be rejected",
        );
    }

    /// `--steps-only` paired with `--phase 1` contradicts: one
    /// collapses to a single bucket, the other suppresses
    /// BASELINE — both together are confused phrasing (if N=0,
    /// `--steps-only` suppresses it; if N>=1, `--steps-only` is
    /// redundant). Clap must reject.
    #[test]
    fn perf_delta_phase_flags_steps_only_conflicts_with_phase_filter() {
        let argv = argv_perf_delta(&["--steps-only", "--phase", "1"]);
        assert!(
            Cargo::try_parse_from(&argv).is_err(),
            "--steps-only + --phase must be rejected",
        );
    }

    /// `--phases-only` + `--steps-only` + `--phase-threshold`
    /// composes — no conflict annotations gate any of the three
    /// against each other. Pins the composability contract:
    /// a refactor that adds a stricter conflict annotation
    /// would break this test, surfacing the over-restriction
    /// before it ships.
    #[test]
    fn perf_delta_phase_flags_phases_only_composes_with_steps_only_and_threshold() {
        let argv = argv_perf_delta(&["--phases-only", "--steps-only", "--phase-threshold", "5"]);
        assert!(
            Cargo::try_parse_from(&argv).is_ok(),
            "--phases-only + --steps-only + --phase-threshold must parse cleanly",
        );
    }

    /// `--phases-only` + `--phase 1` + `--phase-threshold 5`
    /// composes because all three are non-conflicting (they
    /// project on different axes: section suppression × specific
    /// phase × per-row significance gate). Sibling negative-case
    /// sentinel to the steps-only composition above.
    #[test]
    fn perf_delta_phase_flags_phases_only_composes_with_phase_filter_and_threshold() {
        let argv = argv_perf_delta(&["--phases-only", "--phase", "1", "--phase-threshold", "5"]);
        assert!(
            Cargo::try_parse_from(&argv).is_ok(),
            "--phases-only + --phase + --phase-threshold must parse cleanly",
        );
    }
}
