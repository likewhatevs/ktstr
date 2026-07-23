//! Per-run sidecar JSON — the durable record of a ktstr test outcome.
//!
//! Every test (pass, fail, or skip) writes a [`SidecarResult`] to a
//! JSON file under the run's sidecar directory; downstream analysis
//! (`cargo ktstr stats`, CI dashboards) aggregates those files to
//! compute pass/fail rates, verifier stats, callback profiles, and
//! KVM stats across gauntlet variants.
//!
//! Responsibilities owned by this module:
//! - [`SidecarResult`]: the on-disk schema. Writer-side: every field
//!   is always emitted — `null` for `None`, `[]` for empty `Vec` —
//!   with no `skip_serializing_if` and no `serde(default)`. Reader-
//!   side: serde's native `Option<T>` deserialize tolerates absence
//!   (a missing key parses as `None`); non-`Option` fields (e.g.
//!   `test_name`, `passed`, `stats`) are hard-required and a missing
//!   key fails deserialize. The contract is intentionally asymmetric
//!   so a future producer that drops an `Option` field still parses
//!   on older readers, while the current writer guarantees full
//!   round-trip symmetry. Pre-1.0: old sidecar JSON is disposable;
//!   regenerate by re-running the test rather than relying on the
//!   reader-side tolerance for migration.
//! - [`collect_sidecars`]: load every `*.ktstr.json` under a directory
//!   (one level of subdirectories for per-job gauntlet layouts).
//! - [`write_sidecar`] / [`write_skip_sidecar`]: serialize one run to
//!   disk; variant-hash the discriminating fields so gauntlet variants
//!   don't clobber each other.
//! - [`sidecar_dir`], [`runs_root`], [`newest_run_dir`]: resolve where
//!   sidecars live (env override, or
//!   `{target}/ktstr/{kernel}-{project_commit}` where
//!   `{project_commit}` is the project tree's HEAD short hex from
//!   [`detect_project_commit`], suffixed `-dirty` when the
//!   worktree differs).
//! - [`format_run_dirname`]: render the
//!   `{kernel}-{project_commit}` leaf name from the resolved
//!   kernel + commit slots, substituting the literal `unknown`
//!   when either probe returned `None` so the dirname stays
//!   filesystem-safe (see the unknown-commit collision
//!   semantics in the runs guide).
//! - [`is_run_directory`]: predicate consumed by run-listing
//!   walkers ([`newest_run_dir`] here, `sorted_run_entries` in
//!   `crate::stats`). Filters non-directories and dotfile
//!   subdirectories (notably the `.locks/` flock-sentinel
//!   subdirectory) so the lock infrastructure cannot pollute
//!   `cargo ktstr stats list` output or claim the "most recent
//!   run" bucket.
//! - [`reset_run_dir_for_session`]: shallow-wipe `*.ktstr.json`
//!   files when the run epoch changes so a re-run at the same
//!   `{kernel}-{project_commit}` key produces a last-writer-wins
//!   snapshot rather than an append-only archive. Raw no-token
//!   calls retain a process-local once gate.
//! - [`acquire_run_dir_publication_lock`]: cross-process publication
//!   rail on `{runs_root}/.locks/{key}.lock`. Writers whose run-epoch
//!   sentinel already matches take `LOCK_SH` and publish concurrently;
//!   the first writer or an epoch transition takes `LOCK_EX` for
//!   wipe -> sentinel -> first publication. Holding SH through atomic
//!   rename prevents an EX reset from crossing either the primary write
//!   or its final-verdict rewrite.
//!   The override branch (operator-chosen
//!   `KTSTR_SIDECAR_DIR`) skips the flock for the same reason
//!   it skips pre-clear: the operator owns the directory's
//!   contents.
//! - [`warn_unknown_project_commit_once`]: one-shot stderr warning
//!   on first sidecar write when `detect_project_commit` returns
//!   `None` (test process not in a git repo) so concurrent or
//!   successive non-git runs colliding on `{kernel}-unknown`
//!   surface the disambiguation hint
//!   (`KTSTR_SIDECAR_DIR=…` or place the tree under git) at
//!   first invocation rather than as a silent collision.
//! - [`format_verifier_stats`], [`format_callback_profile`],
//!   [`format_kvm_stats`]: human-readable summaries from a
//!   `Vec<SidecarResult>` for CLI output.
//! - [`detect_kernel_version`]: read the kernel version from
//!   `KTSTR_KERNEL` cache metadata for sidecar-dir naming and the
//!   `kernel_version` field, with fallback to
//!   `include/config/kernel.release` in the kernel source tree
//!   when the cache metadata is absent or does not carry a
//!   version (e.g. a raw source-tree path set in `KTSTR_KERNEL`
//!   rather than a cache key).
//! - [`detect_kernel_commit`]: read the kernel SOURCE TREE's git
//!   HEAD short hex (with `-dirty` suffix when worktree differs
//!   from the index or HEAD differs from the index) for the
//!   `kernel_commit` field. Distinct from `kernel_version`
//!   (release string from `kernel.release`) and `project_commit`
//!   (ktstr framework HEAD): this records "what kernel commit
//!   produced this run" so two runs of the same `kernel_version`
//!   but different WIP source trees compare distinctly.

use std::path::PathBuf;

use anyhow::Context;

use crate::assert::{AssertResult, ScenarioStats};
use crate::monitor::MonitorSummary;
use crate::sync::MutexExt;
use crate::test_support::PayloadMetrics;
use crate::timeline::StimulusEvent;
use crate::vmm;

use super::entry::KtstrTestEntry;
use super::timefmt::{generate_run_id, now_iso8601};

/// Which time base denominates this sidecar's workload-THROUGHPUT rate
/// metrics (`taobench_*_ops_per_cpu_sec` /
/// `schbench_*loops_per_cpu_sec` / `iteration_rate`, and any future
/// throughput rate). Some rate key names predate the denomination change, so
/// a number from a wall-era sidecar is NOT comparable to a CPU-era sidecar —
/// the compare
/// pipeline keys its row pairing and its group-averaging on this marker so
/// cross-denomination values are never silently folded or diffed (see
/// `PairingKey::from_row` and the `denomination_mismatches` counter in
/// `CompareReport`).
///
/// Event-frequency metrics (IRQ/sec, per-second RPS samples) and latency
/// metrics are wall-based in BOTH eras — a frequency's meaning is real time —
/// so they are outside this marker's scope.
///
/// `#[serde(default)]` on the carrying field + `Default = Wall`: a sidecar
/// written before the marker existed is by definition wall-era.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThroughputDenomination {
    /// Throughput rates are work / wall-second (pre-marker sidecars).
    #[default]
    Wall,
    /// Throughput rates are work / CPU-second received
    /// (`CLOCK_THREAD_CPUTIME_ID`) — the current denomination.
    CpuSec,
}

impl ThroughputDenomination {
    /// Stable string form used in pairing keys and diagnostics.
    pub fn as_str(self) -> &'static str {
        match self {
            ThroughputDenomination::Wall => "wall",
            ThroughputDenomination::CpuSec => "cpu_sec",
        }
    }
}

/// Test result sidecar written to KTSTR_SIDECAR_DIR for post-run analysis.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct SidecarResult {
    /// Fully qualified test name (matches `KtstrTestEntry::name`,
    /// the bare function name without the `ktstr/` nextest prefix).
    pub test_name: String,
    /// Rendered topology label (e.g. `1n2l4c1t`) for the variant this
    /// sidecar describes.
    pub topology: String,
    /// Scheduler name (matches `Scheduler::name`); `"eevdf"` for
    /// tests run without an scx scheduler.
    pub scheduler: String,
    /// Best-effort git commit of the scheduler binary used for this
    /// run. Currently ALWAYS `None` for every `SchedulerSpec`
    /// variant — no variant today has a reliable commit source.
    /// The field is reserved on the schema so stats tooling can
    /// enrich it once a reliable source exists (e.g. a
    /// `--version` probe or ELF-note read on the resolved
    /// scheduler binary). See
    /// [`crate::test_support::SchedulerSpec::scheduler_commit`]
    /// for the full per-variant rationale.
    ///
    /// Writer always emits (`"scheduler_commit": null` on absence).
    /// Reader-side: serde's native `Option<T>` deserialize tolerates
    /// absence (a missing key parses as `None`); see the module-level
    /// doc for the full asymmetric contract that governs every
    /// nullable on this struct.
    pub scheduler_commit: Option<String>,
    /// How the userspace scheduler binary was resolved for this run —
    /// the snake_case [`crate::test_support::ResolveSource::as_str`] tag
    /// (`"path"`, `"env_var"`, `"path_lookup"`, `"auto_built"`,
    /// `"not_found"`). Historical sidecars may carry retired tags
    /// (`"sibling_dir"`, `"target_debug"`, `"target_release"`) from the
    /// pre-built-fallback cascade the resolver no longer runs; the stats
    /// CLI reads `resolve_source` as an opaque string, so those still
    /// filter and display. Provenance, not identity: distinct from
    /// [`SidecarResult::scheduler_commit`] (the binary's git commit) —
    /// this records the resolution PATH, so the stats CLI can answer "was
    /// this run's scheduler auto-built from the workspace HEAD, or
    /// resolved from a possibly-stale `$PATH` binary?".
    /// `"auto_built"` is the only tag whose source commit is known to
    /// match the workspace tree; every other tag carries the stale-binary
    /// hazard documented on the [`crate::test_support::ResolveSource`]
    /// variant.
    ///
    /// Writer always emits (`"resolve_source": null` on absence — the
    /// skip-sidecar path resolves no binary). Reader-side: serde's native
    /// `Option<T>` deserialize tolerates absence (a missing key parses as
    /// `None`); see the module-level doc for the full asymmetric
    /// contract. Excluded from `sidecar_variant_hash` for the same
    /// cross-host grouping reason as `scheduler_commit` / `run_source`:
    /// two runs of the same semantic variant resolved via different
    /// discovery paths must still bucket together.
    pub resolve_source: Option<String>,
    /// Best-effort git HEAD of the ktstr project tree at sidecar-
    /// write time. Captured by `detect_project_commit` via
    /// `gix::discover` from the test process's current working
    /// directory; walks up to find the enclosing repo and reads
    /// HEAD short-hex, suffixing `-dirty` when index-vs-HEAD or
    /// worktree-vs-index changes are observed (submodules ignored,
    /// matching the [`crate::fetch::local_source`] dirty-detection
    /// pattern). `None` when cwd is not inside any git repo, or
    /// when the gix probe fails for any reason — this is metadata,
    /// not a gate, so probe failure must not abort the run.
    ///
    /// Distinct from [`SidecarResult::scheduler_commit`]: that
    /// field tracks the userspace scheduler binary's commit
    /// (currently always `None` per its own doc); this field
    /// tracks the ktstr framework / test-runner commit, so the
    /// stats CLI can answer "which version of the harness produced
    /// this sidecar?" without inspecting the scheduler.
    ///
    /// Writer always emits (`"project_commit": null` on absence).
    /// Reader-side: serde's native `Option<T>` deserialize tolerates
    /// absence (a missing key parses as `None`) — see the module-
    /// level doc for the full asymmetric contract. Excluded from
    /// `sidecar_variant_hash` for the same cross-host grouping
    /// reason `scheduler_commit` is excluded: two runs of the same
    /// semantic variant on different ktstr commits must still bucket
    /// together so `perf-delta` can diff them; the commit-drift
    /// detection inspects this field directly via `--project-commit`
    /// / `--a-project-commit` / `--b-project-commit`.
    pub project_commit: Option<String>,
    /// Binary payload name (matches `Payload::name` when
    /// `entry.payload` is set). `None` when the test declared no
    /// binary payload. Writer always emits (`"payload": null` on
    /// absence); reader-side, serde's native `Option<T>` deserialize
    /// tolerates absence — see the module-level doc for the full
    /// asymmetric contract.
    pub payload: Option<String>,
    /// Per-payload extracted metrics collected from `ctx.payload(X).run()`
    /// / `.spawn().wait()` call sites during the test body.
    ///
    /// One [`PayloadMetrics`] per invocation, in the order the calls
    /// ran. Empty when no payload calls were made (scheduler-only
    /// tests, or a binary-only test where the body bailed before
    /// running the payload). Writer always emits (`"metrics": []` in
    /// that case); reader-side, this `Vec` field is hard-required —
    /// non-`Option` fields fail deserialize on absence. See the
    /// module-level doc for the full contract.
    pub metrics: Vec<PayloadMetrics>,
    /// True when the run is a real pass — every assertion that
    /// ran produced a positive verdict. Mirrors
    /// [`crate::assert::AssertResult::is_pass`]. Mutually
    /// exclusive with [`Self::skipped`] and [`Self::inconclusive`]:
    /// the three bits `(passed, skipped, inconclusive)` form a
    /// strict 4-state encoding where at most one is set per
    /// record. The fourth state — Fail — is the all-false case
    /// (no dedicated bit; [`Self::is_fail`] derives it). A real
    /// pass requires `!skipped && !inconclusive` AND at least one
    /// observed assertion (the empty / all-skip case routes
    /// through [`Self::skipped`] instead).
    pub passed: bool,
    /// True when the run was skipped (e.g. topology mismatch,
    /// missing resource, in-VM `AssertResult::skip` return).
    /// Mutually exclusive with [`Self::passed`] (Pass requires a
    /// real assertion; an all-skip stream is Skip, not Pass) and
    /// with [`Self::inconclusive`]. Stats tooling subtracts
    /// `skipped` runs from "pass count" so non-executions are not
    /// reported as passes.
    pub skipped: bool,
    /// True when at least one assertion was [`Outcome::Inconclusive`](crate::assert::Outcome::Inconclusive) —
    /// the run ran but a zero-denominator ratio gate could not be
    /// evaluated (e.g. zero iterations across all workers under a
    /// `max_migration_ratio` check). Mutually exclusive with
    /// [`Self::passed`] and [`Self::skipped`]; in the
    /// `Fail > Inconclusive > Pass > Skip` lattice, Inconclusive
    /// dominates Pass/Skip but loses to Fail, so a run with both
    /// Inconclusive and Fail outcomes records `inconclusive = false,
    /// passed = false` (Fail wins) — `inconclusive = true` requires
    /// `!is_fail() && !is_pass() && !is_skip()`.
    ///
    /// Distinct from `passed = false` (Fail) and `skipped = true`
    /// (precondition unmet) so CI gates and stats tooling can
    /// triage zero-denominator runs as "workload didn't produce
    /// the signal the assertion needed" rather than misclassifying
    /// them as silent passes (prior to the [`Outcome::Inconclusive`](crate::assert::Outcome::Inconclusive)
    /// variant the zero-denominator case fell out as Pass) or as
    /// hard failures.
    pub inconclusive: bool,
    /// True when the persisted verdict (`passed`/`skipped`/
    /// `inconclusive`) is the POST-inversion FINAL outcome of a run
    /// whose underlying scenario actually failed — i.e. an
    /// `expect_err` / `expect_auto_repro` test whose induced failure was
    /// inverted to a pass. Set by the sidecar finalize
    /// (`finalize_sidecar_verdict`) after dispatch resolves the
    /// verdict; `false` for an ordinary pass/skip/fail.
    ///
    /// The verdict bits carry the FINAL outcome so the footer, `stats`
    /// analysis, and `replay` match nextest's exit code. This flag
    /// preserves the one fact that overwrite loses: that the run's
    /// telemetry is failure-mode-dominated (a deliberately short /
    /// stalled run). `perf-delta` ORs it into its exclusion guard so
    /// an inverted-to-pass row is still kept OUT of the regression math
    /// (its induced-crash telemetry is not real scheduler behavior).
    pub expected_failure: bool,
    /// Aggregate per-cgroup statistics merged across every worker.
    pub stats: ScenarioStats,
    /// Monitor summary. `None` means the monitor loop did not run
    /// (host-only tests, early VM failure) or sample collection
    /// produced no valid data. Writer always emits (`"monitor": null`
    /// on absence); reader-side, serde's native `Option<T>`
    /// deserialize tolerates absence — see the module-level doc.
    pub monitor: Option<MonitorSummary>,
    /// Periodic-capture coverage for this run: how many periodic snapshot
    /// boundaries actually fired (`periodic_fired`) out of the configured
    /// `num_snapshots` target (`periodic_target`). Carried verbatim from
    /// [`crate::prelude::VmResult`] so cross-run tooling can read the
    /// coverage off the persisted sidecar (previously only the in-memory
    /// result exposed it). `0`/`0` for runs with no periodic captures
    /// configured. Hard-required `u32` fields — old sidecars predating
    /// them re-generate on the next run (sidecar data is disposable).
    pub periodic_fired: u32,
    /// See [`Self::periodic_fired`].
    pub periodic_target: u32,
    /// Guest vCPU count and the effective host-CPU budget the vCPU threads
    /// ran on, carried verbatim from [`crate::prelude::VmResult`]. Drive
    /// the `cpu-budget` comparison Dimension (cross-budget runs are not
    /// paired — confining 32 vCPUs to 4 host CPUs measures something else)
    /// and the overcommit marker: `cpu_budget < vcpus` means the host
    /// time-sliced the guest's vCPUs, confounding the timing metrics
    /// (wake-latency / off-CPU / run-delay — schedstat run_delay tracks
    /// rq->clock, which follows the guest TSC and is not steal-adjusted,
    /// so the off-host window inflates it for tasks waiting across it).
    /// Hard-required `u32`
    /// (old sidecars re-generate; sidecar data is disposable). EXCLUDED
    /// from `sidecar_variant_hash`: a budget change is a different
    /// measurement, separated downstream by the Dimension, not the
    /// identity bucket.
    pub vcpus: u32,
    /// See [`Self::vcpus`].
    pub cpu_budget: u32,
    /// Host-side vCPU scheduling dilation for this run — the
    /// `HostVcpuSchedstat::dilation` ratio `1 + Σrun_delay/Σon_cpu`
    /// over the vCPU host threads. `> 1.0`
    /// quantifies how much the HOST time-sliced the guest's vCPUs; the
    /// direct EVIDENCE behind the `cpu_budget < vcpus` overcommit marker
    /// (a run can overcommit on paper yet see D≈1.0 if the host was
    /// otherwise idle). `None` when no vCPU ran or the host lacks
    /// `CONFIG_SCHEDSTATS` (every schedstat line reads `0 0 0`) — kept
    /// distinguishable from a genuine `D == 1.0`.
    ///
    /// `#[serde(default)]`: sidecars written before this field existed
    /// deserialize with `None` (disposable schema; a re-run repopulates).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub host_dilation: Option<f64>,
    /// Denomination of this sidecar's workload-throughput rate metrics.
    /// `#[serde(default)]` = [`ThroughputDenomination::Wall`] for
    /// pre-marker sidecars; the writer stamps the current
    /// [`ThroughputDenomination::CpuSec`]. Keys row pairing and group
    /// averaging in the compare pipeline so cross-denomination values are
    /// never silently compared (see the enum doc).
    #[serde(default)]
    pub throughput_denomination: ThroughputDenomination,
    /// Ordered stimulus events published by the guest step executor
    /// while the scenario ran.
    pub stimulus_events: Vec<StimulusEvent>,
    /// WorkSpec type label used for post-hoc filtering and A/B comparison
    /// (distinct from the `WorkType` enum — this is the text name).
    pub work_type: String,
    /// Per-BPF-program verifier statistics captured from the VM's
    /// scheduler (when one was loaded). Empty when no scheduler
    /// programs were inspected. Writer always emits as
    /// `"verifier_stats": []` in that case; reader-side, this `Vec`
    /// field is hard-required (non-`Option` fields fail deserialize
    /// on absence). See the module-level doc.
    pub verifier_stats: Vec<crate::monitor::bpf_prog::ProgVerifierStats>,
    /// Aggregate per-vCPU KVM stats read after VM exit. `None` when
    /// the VM did not run (host-only tests) or KVM stats were
    /// unavailable. Writer always emits (`"kvm_stats": null` on
    /// absence); reader-side, serde's native `Option<T>` deserialize
    /// tolerates absence — see the module-level doc.
    pub kvm_stats: Option<crate::vmm::KvmStatsTotals>,
    /// Effective sysctls active during this test run, recorded as raw
    /// `sysctl.key=value` cmdline strings. Writer always emits as
    /// `"sysctls": []` when none; reader-side, this `Vec` field is
    /// hard-required (non-`Option` fields fail deserialize on
    /// absence). See the module-level doc.
    pub sysctls: Vec<String>,
    /// Effective kernel command-line args active during this test run.
    /// Writer always emits as `"kargs": []` when none; reader-side,
    /// this `Vec` field is hard-required (non-`Option` fields fail
    /// deserialize on absence). See the module-level doc.
    pub kargs: Vec<String>,
    /// Kernel version of the VM under test (from cache metadata,
    /// e.g. `"6.14.2"`). Populated from the cache entry's
    /// `metadata.json` version field, with fallback to the kernel
    /// source tree's `include/config/kernel.release` when
    /// `KTSTR_KERNEL` points at a raw source path rather than a
    /// cache key; `None` for host-only tests or when neither
    /// source yields a version string. The host's running kernel
    /// release is carried separately in `host.kernel_release`.
    /// Writer always emits (`"kernel_version": null` on absence);
    /// reader-side, serde's native `Option<T>` deserialize tolerates
    /// absence — see the module-level doc for the full asymmetric
    /// contract.
    pub kernel_version: Option<String>,
    /// Kernel SOURCE TREE git HEAD short hex (7 chars via
    /// `oid::to_hex_with_len(7)`), with `-dirty` suffix appended
    /// when HEAD-vs-index or index-vs-worktree changes are
    /// observed. Probes via `gix::open` against the kernel
    /// directory resolved from `KTSTR_KERNEL` (not `gix::discover`
    /// — the kernel dir is explicit, not walked-up). Captured by
    /// `detect_kernel_commit` at sidecar-write time.
    ///
    /// Distinct from sibling fields:
    /// - [`SidecarResult::kernel_version`] — release string read
    ///   from cache metadata or `include/config/kernel.release`,
    ///   e.g. `"6.14.2"`. Two runs of `6.14.2` from a clean
    ///   tree and a `-dirty` worktree at the same HEAD share
    ///   `kernel_version` but differ on `kernel_commit`.
    /// - [`SidecarResult::project_commit`] — ktstr framework
    ///   HEAD captured from the test process's cwd. Tracks
    ///   "what version of the harness produced this sidecar?"
    ///   independently of the kernel under test.
    /// - [`SidecarResult::scheduler_commit`] — userspace
    ///   scheduler binary's commit (currently always `None`).
    ///
    /// `None` when:
    /// - `KTSTR_KERNEL` is unset or empty;
    /// - the resolved `KernelId` is `Version` / `CacheKey` whose
    ///   underlying source is `Tarball` / `Git` (no source tree
    ///   on disk to probe);
    /// - the resolved kernel directory is not a git repository
    ///   (`gix::open` fails);
    /// - HEAD cannot be read (unborn HEAD on a fresh `git init`
    ///   with zero commits);
    /// - any other gix probe failure — metadata, not a gate.
    ///
    /// Writer always emits (`"kernel_commit": null` on absence);
    /// reader-side, serde's native `Option<T>` deserialize tolerates
    /// absence — see the module-level doc for the full asymmetric
    /// contract. Excluded from `sidecar_variant_hash` for the same
    /// cross-host grouping reason `scheduler_commit` and
    /// `project_commit` are excluded: two runs of the same semantic
    /// variant on different kernel-source HEADs must still bucket
    /// together so `perf-delta` can diff them; the commit-drift
    /// detection inspects this field directly via the
    /// `--kernel-commit` filter.
    pub kernel_commit: Option<String>,
    /// ISO 8601 timestamp of when this test run started.
    pub timestamp: String,
    /// Unique identifier for the test run. Composed as
    /// `{run_id_timestamp}-{counter}` — the `YYYYMMDDTHHMMSSZ`
    /// process-start stamp followed by a process-local monotonic
    /// counter. Every sidecar produced in one `cargo ktstr test`
    /// invocation shares the same timestamp prefix; the counter
    /// distinguishes concurrent gauntlet variants within that
    /// invocation. Distinct from the run DIRECTORY name (keyed
    /// `{kernel}-{project_commit}`, see [`sidecar_dir`]) — the
    /// directory groups runs by what they tested, the `run_id`
    /// groups sidecars by which process emitted them.
    pub run_id: String,
    /// Host context — static-ish runtime state (CPU model,
    /// memory size, THP policy, kernel release, host cmdline,
    /// scheduler tunables). Populated by production sidecar
    /// writers.
    ///
    /// `None` causes:
    /// - **test-fixture path**: not the production sidecar
    ///   writer (production writers always populate `host`).
    /// - **pre-enrichment archive**: sidecar predates the
    ///   host-context landing — re-run the test to regenerate
    ///   under the current schema (no migration shim exists
    ///   per the pre-1.0 disposable-data contract).
    ///
    /// Deliberately excluded from the variant hash so
    /// gauntlet variants on different hosts collapse into the same
    /// hash bucket.
    ///
    /// No serde attributes: writer always emits (`"host": null` when
    /// `None`); reader-side, serde's native `Option<T>` deserialize
    /// tolerates absence (a missing key parses as `None`). The
    /// asymmetric contract is crate-wide — see the module-level doc.
    /// Pre-1.0, sidecar data is disposable, so regenerate by
    /// re-running the test rather than carrying a compat shim for
    /// older JSON; the reader-side tolerance exists so an in-flight
    /// schema rename of an `Option` field does not break parsing of
    /// older sidecars during the same producer-version, not as a
    /// long-term migration story.
    pub host: Option<crate::host_context::HostContext>,
    /// Wall-clock milliseconds spent in
    /// `KtstrVm::collect_results` — the host-side
    /// teardown window from BSP exit through SHM drain (mirrors
    /// [`VmResult::cleanup_duration`](crate::vmm::VmResult::cleanup_duration);
    /// `Duration` is converted to `u64` ms here because every other
    /// timing field on this struct that lands in a sidecar-comparison
    /// CLI uses integer ms or seconds, and JSON has no native
    /// `Duration`). `None` when the run was killed by the watchdog
    /// before `collect_results` returned, or for the `host_only` /
    /// host-only-stub paths that never boot a VM. Writer always emits
    /// (`"cleanup_duration_ms": null` on absence); reader-side,
    /// serde's native `Option<T>` deserialize tolerates absence — see
    /// the module-level doc for the full asymmetric contract.
    pub cleanup_duration_ms: Option<u64>,
    /// Provenance tag for this sidecar — distinguishes a developer's
    /// local run from a CI run so cross-environment comparisons in
    /// `perf-delta` can narrow on (or contrast across) the run
    /// environment without inferring it from `host`.
    ///
    /// Recorded by `detect_run_source` at sidecar-write time:
    /// - `Some("ci")` when `KTSTR_CI_ENV` is set non-empty (CI runner
    ///   scripts export it before invoking the test binary; local
    ///   runs never set it).
    /// - `Some("local")` otherwise — the default for any sidecar
    ///   produced by a developer-driven invocation.
    /// - The third documented value (`"archive"`) is NEVER written
    ///   here: a sidecar cannot know it will later be archived. The
    ///   stats CLI applies the `"archive"` tag at LOAD time when its
    ///   `--dir` flag points at a non-default pool root, overriding
    ///   whatever was on disk via `apply_archive_source_override`.
    ///
    /// `Option<String>` (rather than an enum) keeps the schema
    /// extensible without a serde-version bump if a future producer
    /// wants a new tag (e.g. `"benchmark"`); the consumer side
    /// treats unknown values the same as known ones — they are
    /// strings the operator can pass via `--run-source` to filter on.
    /// Writer always emits (`"run_source": null` on absence);
    /// reader-side, serde's native `Option<T>` deserialize tolerates
    /// absence — see the module-level doc for the full asymmetric
    /// contract. Excluded from `sidecar_variant_hash` for the same
    /// cross-host grouping reason `host` is excluded — two runs of
    /// the same semantic variant from different environments must
    /// still bucket together so `perf-delta` can pair them; `--run-source`
    /// is the explicit knob for source-aware narrowing.
    ///
    /// Field name `run_source` (renamed from `source`) disambiguates
    /// from [`crate::cache::KernelSource`] / `KernelMetadata.source`
    /// — those describe the kernel build's input (tarball / git /
    /// local), this describes the run-environment provenance.
    ///
    /// **On-disk JSON key changed from `"source"` to `"run_source"`
    /// in the field rename.** No `#[serde(alias = "source")]` is
    /// in place: archived sidecars written before the rename carry
    /// the `"source"` key, which the current schema treats as an
    /// unknown field. Because `SidecarResult`'s derive does NOT
    /// set `deny_unknown_fields`, the deserialize does not fail
    /// outright — instead serde silently DROPS the stale `"source"`
    /// payload and lands `run_source = None` (since `Option<T>`'s
    /// "tolerate absence" rule kicks in for the missing
    /// `"run_source"` field). The data is lost, not preserved. This
    /// is deliberate per the project's pre-1.0 disposable-data
    /// contract: re-running tests regenerates sidecars under the
    /// new key rather than carrying compat shims forward. Consumers
    /// who need the run-source classification on archived JSON
    /// must either rename the key in-place before deserialize, or
    /// re-run the test to regenerate the sidecar with the new
    /// schema. Tooling that runs against the renamed schema and
    /// observes a `None` `run_source` cannot distinguish "sidecar
    /// pre-dates the field" from "sidecar pre-dates the rename and
    /// lost its tag" — both lower-bound at `None` for filter
    /// purposes.
    pub run_source: Option<String>,
    /// Per-test [`crate::test_support::PerfDeltaAssertion`]s declared on the
    /// entry, serialized so `cargo ktstr perf-delta --noise-adjust`'s host-side
    /// compare can enforce them across commits (the entry registry in the parent
    /// process describes only HEAD's tests, not a baseline/cached sidecar's
    /// commit, so the declaration must travel WITH the run). Empty when the test
    /// declared none. Inert here — a normal `cargo ktstr test` writes them but
    /// never gates on them; only the `--noise-adjust` compare consults them (the
    /// scalar compare warns that declared gates were skipped).
    ///
    /// Writer always emits (`"perf_delta_assertions": []` on absence); reader-
    /// side this `Vec` field is hard-required (non-`Option` fails deserialize on
    /// absence) — see the module-level doc for the full contract.
    pub perf_delta_assertions: Vec<PerfDeltaAssertionRecord>,
}

/// Owned, serialized mirror of [`crate::test_support::PerfDeltaAssertion`]. The
/// public declaration type uses `&'static str` (so it stays const/E0493-safe on
/// the entry) and therefore cannot `Deserialize` into an owned value; this
/// `String`-backed record is the sidecar carrier the perf-delta compare reads.
/// `pub` because it is a field of the `pub` [`SidecarResult`] (constructed
/// across the workspace, including by the `cargo-ktstr` binary crate); the
/// author-facing declaration type is [`crate::test_support::PerfDeltaAssertion`].
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PerfDeltaAssertionRecord {
    /// Registry metric name this assertion gates (see `stats list-metrics`).
    pub metric: String,
    /// Pinned regression direction, or `None` to inherit the registry polarity.
    pub direction: Option<crate::test_support::Polarity>,
    /// Relative-regression override (percent), or `None` for `default_rel`.
    pub max_regression_pct: Option<f64>,
    /// Absolute-materiality override, or `None` for `default_abs`.
    pub min_abs: Option<f64>,
    /// Phase scope (`step_index`), or `None` to gate the aggregate value.
    pub phase: Option<u16>,
}

impl From<&crate::test_support::PerfDeltaAssertion> for PerfDeltaAssertionRecord {
    fn from(a: &crate::test_support::PerfDeltaAssertion) -> Self {
        Self {
            metric: a.metric().to_string(),
            direction: a.direction(),
            max_regression_pct: a.max_regression_pct(),
            min_abs: a.min_abs(),
            phase: a.phase(),
        }
    }
}

impl SidecarResult {
    /// Convenience accessor mirroring
    /// [`crate::assert::AssertResult::is_pass`]. SidecarResult is the
    /// wire-format mirror of an AssertResult; this method exposes the
    /// same is_pass / is_fail / is_skip / is_inconclusive vocabulary
    /// so consumers can swap between the two without re-learning
    /// field names.
    ///
    /// Returns true only when the run reached a real Pass — neither
    /// skipped, inconclusive, nor failed. The triple-conjunct guard
    /// matches AssertResult's `Fail > Inconclusive > Pass > Skip`
    /// dominance under the strict 4-state mutex this struct encodes.
    /// CI gates that want "ship-on-pass" semantics call this method
    /// and only this method.
    ///
    /// Part of the `is_pass` / `is_fail` / `is_inconclusive` /
    /// `is_skip` vocabulary uniform across the verdict surfaces:
    /// [`crate::assert::AssertResult::is_pass`] / `Self::is_pass` /
    /// [`crate::assert::Outcome::is_pass`] / `MonitorVerdict::is_pass`
    /// (in the `monitor` module, which is `pub(crate)`) /
    /// `Verdict::is_pass` (re-exported at [`crate::assert::Verdict`]) /
    /// `GauntletRow::is_pass` (in the `stats` module, which is
    /// `pub(crate)`).
    pub fn is_pass(&self) -> bool {
        self.passed && !self.skipped && !self.inconclusive
    }
    /// Convenience accessor mirroring
    /// [`crate::assert::AssertResult::is_fail`]. The four-state
    /// encoding uses three stored bits `(passed, skipped,
    /// inconclusive)` in strict mutual exclusion (at most one
    /// set); Fail is the all-false derived state, no dedicated
    /// bit. `is_fail` reads "none of the three bits are set",
    /// which under `Fail > Inconclusive > Pass > Skip` dominance
    /// correctly resolves a mixed Fail+Inconclusive stream as
    /// Fail.
    pub fn is_fail(&self) -> bool {
        !self.passed && !self.skipped && !self.inconclusive
    }
    /// Convenience accessor mirroring
    /// [`crate::assert::AssertResult::is_skip`].
    pub fn is_skip(&self) -> bool {
        self.skipped
    }
    /// Convenience accessor mirroring
    /// [`crate::assert::AssertResult::is_inconclusive`]. True when
    /// the run could not be evaluated (zero-denominator ratio gate);
    /// false on real Pass, real Fail, or Skip. CI gates that gate
    /// on "did we get a real verdict?" should test
    /// `r.is_pass() || r.is_fail()` and treat both `is_skip()` and
    /// `is_inconclusive()` as "couldn't measure".
    pub fn is_inconclusive(&self) -> bool {
        self.inconclusive
    }
}

#[cfg(test)]
impl SidecarResult {
    /// Populated [`SidecarResult`] for unit tests. Every field has a
    /// reasonable default so call sites only spell out what they want
    /// to vary via struct-update syntax:
    ///
    /// ```ignore
    /// let sc = SidecarResult {
    ///     test_name: "my_test".to_string(),
    ///     passed: false,
    ///     ..SidecarResult::test_fixture()
    /// };
    /// ```
    ///
    /// Defaults model a passing EEVDF run on a minimal `1n1l1c1t`
    /// topology with no payload and no VM telemetry: `test_name="t"`,
    /// `topology="1n1l1c1t"`, `scheduler="eevdf"`, `work_type="SpinWait"`,
    /// `passed=true`, `skipped=false`, `inconclusive=false`, every
    /// [`Option`] `None`, every [`Vec`] empty, `stats` is
    /// `ScenarioStats::default()`, and both `timestamp`/`run_id` are
    /// empty strings.
    ///
    /// **Prefer this over local `base = || SidecarResult { ... }`
    /// closures.** A local closure duplicates the default set and
    /// drifts the moment [`SidecarResult`] grows a field; this fixture
    /// is the single place those defaults live.
    ///
    /// **Hash-stability tests must not rely on these defaults for
    /// hash-participating fields** (`topology`, `scheduler`, `payload`,
    /// `work_type`, `sysctls`, `kargs`). Tests that pin
    /// a [`sidecar_variant_hash`] output against a literal constant
    /// must spell every hash-participating field out explicitly so a
    /// future change to these defaults cannot silently shift the
    /// pinned value.
    pub(crate) fn test_fixture() -> SidecarResult {
        SidecarResult {
            test_name: "t".to_string(),
            perf_delta_assertions: Vec::new(),
            topology: "1n1l1c1t".to_string(),
            scheduler: "eevdf".to_string(),
            scheduler_commit: None,
            resolve_source: None,
            project_commit: None,
            payload: None,
            metrics: Vec::new(),
            passed: true,
            skipped: false,
            inconclusive: false,
            expected_failure: false,
            stats: crate::assert::ScenarioStats::default(),
            monitor: None,
            periodic_fired: 0,
            periodic_target: 0,
            vcpus: 1,
            cpu_budget: 1,
            host_dilation: None,
            throughput_denomination: ThroughputDenomination::CpuSec,
            stimulus_events: Vec::new(),
            work_type: "SpinWait".to_string(),
            verifier_stats: Vec::new(),
            kvm_stats: None,
            sysctls: Vec::new(),
            kargs: Vec::new(),
            kernel_version: None,
            kernel_commit: None,
            timestamp: String::new(),
            run_id: String::new(),
            host: None,
            cleanup_duration_ms: None,
            run_source: None,
        }
    }
}

/// Predicate: is `path` a ktstr sidecar JSON filename?
///
/// True iff the path's extension is `json` AND the path's
/// FILENAME COMPONENT (`Path::file_name`) contains `.ktstr.` —
/// matching the on-disk shape produced by [`write_sidecar`]
/// (`<test>-<variant_hash>.ktstr.json`). Both gates are required:
/// bare `*.json` files (cargo cache, stray fixtures) and non-json
/// files whose name happens to contain `.ktstr.` (e.g. a log)
/// are excluded.
///
/// The filename-component check (rather than full-path string)
/// is load-bearing: a parent directory like
/// `target/foo.ktstr.bar/extra.json` would falsely match a
/// whole-path `contains(".ktstr.")` while NOT being a sidecar.
/// `Path::file_name()` returns only the trailing component, so
/// `.ktstr.` in any ancestor segment cannot trigger the predicate.
///
/// Single source of truth for "is this file a sidecar?" — used
/// by [`collect_sidecars_with_errors`]'s parsing walker and by
/// the explain-sidecar file-count walker
/// (`crate::cli::stats_cmds::explain_sidecar::count_sidecar_files`). Both
/// walkers MUST agree on the predicate so `walked` (count) and
/// `valid + errors` (parse outcomes) reconcile against each
/// other; a divergence would let a file count toward `walked`
/// without contributing to either bucket, manifesting as a
/// silent-drop count that has no source.
pub(crate) fn is_sidecar_filename(path: &std::path::Path) -> bool {
    path.extension().and_then(|e| e.to_str()) == Some("json")
        && path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.contains(".ktstr."))
}

/// Scan a directory for ktstr sidecar JSON files. Recurses one level
/// into subdirectories to handle per-job gauntlet layouts.
///
/// Convenience wrapper over [`collect_sidecars_with_errors`] for
/// single-directory callers that only need the parsed sidecars and not
/// the per-file parse-failure list. Emits ONE aggregated
/// [`warn_skipped_sidecars`] summary for that directory when stale
/// sidecars were dropped. Multi-directory walkers must NOT use this in a
/// loop (it would print one summary per directory) — they call
/// [`collect_sidecars_with_errors`] per directory and aggregate the counts
/// into a single summary (see `collect_pool`).
pub(crate) fn collect_sidecars(dir: &std::path::Path) -> Vec<SidecarResult> {
    let (sidecars, parse_errors, _io_errors) = collect_sidecars_with_errors(dir);
    warn_skipped_sidecars(dir, parse_errors.len());
    sidecars
}

/// Emit a single aggregated summary for the stale/unparseable sidecars a
/// walk skipped, or nothing when `skipped == 0`. Sidecars written before a
/// schema field was added fail to deserialize and are dropped (sidecar data
/// is disposable — re-running regenerates it); this collapses what would
/// otherwise be one `eprintln!` per file into one line. Per-file detail
/// stays available through [`collect_sidecars_with_errors`]'s parse-error
/// Vec, which `cargo ktstr stats explain-sidecar` renders.
///
/// `pub(crate)` so multi-directory walkers outside this module (e.g.
/// `stats::analyze::sorted_run_entries`) can accumulate `parse_errors.len()`
/// across run directories and emit ONE pool-wide summary rather than one
/// per directory.
pub(crate) fn warn_skipped_sidecars(dir: &std::path::Path, skipped: usize) {
    if skipped > 0 {
        eprintln!(
            "ktstr_test: skipped {skipped} stale sidecar(s) under {} (older \
             schema — re-run the affected tests to regenerate; \
             `cargo ktstr stats explain-sidecar --run <run>` shows per-file \
             detail)",
            dir.display(),
        );
    }
}

/// Per-file parse-failure record returned by
/// [`collect_sidecars_with_errors`] and threaded through
/// `crate::cli::WalkStats::errors` to the renderers.
///
/// Named-field struct (rather than a `(PathBuf, String,
/// Option<String>)` tuple) so call sites read fields by name —
/// pattern-matching `for err in errors` and accessing
/// `err.path` / `err.raw_error` / `err.enriched_message`
/// resists the tuple-position-swap class of bug where positional
/// fields could destructure in either order without compiler help.
pub(crate) struct SidecarParseError {
    /// On-disk path of the sidecar JSON that failed to parse.
    pub path: std::path::PathBuf,
    /// Verbatim serde-error string. Kept raw for
    /// grep-friendly parse-error tracking and surfaced through
    /// the JSON channel as the `error` key.
    pub raw_error: String,
    /// Operator-facing remediation prose computed by
    /// [`enriched_parse_error_message`]. `Some(...)` for known
    /// schema-drift cases (currently the `host` missing-field
    /// pattern), `None` otherwise. Surfaced through the JSON
    /// channel as `enriched_message`.
    pub enriched_message: Option<String>,
}

/// Per-file IO-failure record returned by
/// [`collect_sidecars_with_errors`] and threaded through
/// `crate::cli::WalkStats::io_errors` to the renderers.
///
/// Captures files where the filename predicate matched but
/// `std::fs::read_to_string` failed before parsing could begin —
/// permission denied, mid-rotate truncation, broken symlink,
/// etc. Distinct from [`SidecarParseError`] (which represents
/// "file read OK but JSON parse failed"); separating the two
/// lets dashboard consumers triage filesystem incidents apart
/// from schema drift.
///
/// Named-field struct mirroring [`SidecarParseError`]'s shape so
/// the renderer side can iterate by field name without tuple-
/// position fragility. No `enriched_message` field — there is no
/// remediation catalog for IO failures (causes vary per host:
/// fix permissions, fix the filesystem, retry the test).
pub(crate) struct SidecarIoError {
    /// On-disk path the predicate matched as a sidecar candidate.
    pub path: std::path::PathBuf,
    /// Verbatim `std::io::Error` Display string. Surfaced through
    /// the JSON channel as the `error` key on
    /// `crate::cli::WalkIoError` entries and through the text
    /// channel as the `error: ...` line under the `io errors`
    /// trailing block.
    pub raw_error: String,
}

/// Test-only re-export of [`enriched_parse_error_message`] so
/// `cli::tests` can verify the enrichment-pattern logic
/// directly against synthetic error strings. The helper itself
/// stays private so production code routes through
/// [`collect_sidecars_with_errors`].
#[cfg(test)]
pub(crate) fn enriched_parse_error_message_for_test(
    path: &std::path::Path,
    raw_error: &str,
) -> Option<String> {
    enriched_parse_error_message(path, raw_error)
}

/// Compute the operator-prose enrichment for a serde parse-error
/// message, when one applies. Today the only enriched case is the
/// `host` missing-field schema-drift diagnostic; the function
/// returns `None` for any other shape so consumers can branch on
/// "enrichment exists" without re-implementing the match.
///
/// Pulled out of [`collect_sidecars_with_errors`]'s parse path so the
/// enrichment prose is computed in one place and stored in the returned
/// [`SidecarParseError`]'s `enriched_message` field — parse failures are
/// surfaced only through that Vec, not a separate stderr channel.
///
/// Matching on the Display text is deliberate: serde's typed-error
/// surface for `missing field "X"` is not stable across
/// serde_json versions, but the rendered message is — a
/// forward-compat regression-resilient check costs one string
/// search.
fn enriched_parse_error_message(path: &std::path::Path, raw_error: &str) -> Option<String> {
    let is_missing_host = raw_error.contains("missing field") && raw_error.contains("`host`");
    if is_missing_host {
        Some(format!(
            "ktstr_test: skipping {}: {raw_error} — the `host` field \
             was added to SidecarResult; pre-1.0 policy is \
             disposable-sidecar: re-run the test to regenerate this \
             file under the current schema (no migration shim exists)",
            path.display(),
        ))
    } else {
        None
    }
}

/// Scan a directory for ktstr sidecar JSON files, returning the
/// parsed sidecars, a [`SidecarParseError`] record (named fields
/// `path`, `raw_error`, `enriched_message`) for every file that
/// passed the filename predicate but failed to deserialize, and a
/// [`SidecarIoError`] record (named fields `path`, `raw_error`)
/// for every file that passed the predicate but whose
/// `read_to_string` failed before parsing could begin. Recurses
/// one level into subdirectories to handle per-job gauntlet
/// layouts.
///
/// Parse failures are captured ONLY in the returned parse-errors vec —
/// this walker no longer logs per file. Each failure is a
/// [`SidecarParseError`] record (named fields `path`, `raw_error`,
/// `enriched_message`) for structured callers (`explain-sidecar`'s walker
/// output). Both raw and enriched are exposed so dashboard consumers can
/// pick: raw for parse-error grepping, enriched for human-facing
/// remediation prose. Callers that only need the sidecars aggregate
/// `parse_errors.len()` and emit one [`warn_skipped_sidecars`] summary
/// (see [`collect_sidecars`] / `collect_pool`) rather than one line per
/// file.
///
/// IO failures (third return) get a single eprintln line plus a
/// structured [`SidecarIoError`] record. Distinguished from
/// parse failures so dashboard consumers can triage filesystem
/// incidents (permission denied, mid-rotate truncation, broken
/// symlink) apart from schema drift. With this third channel,
/// every predicate-matching file lands in exactly one of the
/// three returned vecs — the prior implicit
/// `walked - valid - parse_errors.len()` silent-drop count is
/// now zero by construction.
///
/// Callers that don't need structured errors should use
/// [`collect_sidecars`].
pub(crate) fn collect_sidecars_with_errors(
    dir: &std::path::Path,
) -> (
    Vec<SidecarResult>,
    Vec<SidecarParseError>,
    Vec<SidecarIoError>,
) {
    let mut sidecars = Vec::new();
    let mut parse_errors: Vec<SidecarParseError> = Vec::new();
    let mut io_errors: Vec<SidecarIoError> = Vec::new();
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => {
            tracing::warn!(
                dir = %dir.display(),
                error = %e,
                "ktstr_test: collect_sidecars_with_errors cannot read root dir",
            );
            return (sidecars, parse_errors, io_errors);
        }
    };
    let mut subdirs = Vec::new();
    let try_load = |path: &std::path::Path,
                    out: &mut Vec<SidecarResult>,
                    parse_errs: &mut Vec<SidecarParseError>,
                    io_errs: &mut Vec<SidecarIoError>| {
        if !is_sidecar_filename(path) {
            return;
        }
        let data = match std::fs::read_to_string(path) {
            Ok(d) => d,
            Err(e) => {
                let raw = e.to_string();
                eprintln!("ktstr_test: cannot read {}: {raw}", path.display());
                io_errs.push(SidecarIoError {
                    path: path.to_path_buf(),
                    raw_error: raw,
                });
                return;
            }
        };
        match serde_json::from_str::<SidecarResult>(&data) {
            Ok(sc) => out.push(sc),
            Err(e) => {
                let raw = e.to_string();
                let enriched = enriched_parse_error_message(path, &raw);
                // Capture (do not log) the per-file skip: callers emit one
                // aggregated `warn_skipped_sidecars` summary so a directory
                // of stale sidecars produces a single line, not a flood.
                // `cargo ktstr stats explain-sidecar --run <run>` renders the
                // per-file detail (raw + enriched remediation) from this Vec.
                parse_errs.push(SidecarParseError {
                    path: path.to_path_buf(),
                    raw_error: raw,
                    enriched_message: enriched,
                });
            }
        }
    };
    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!(
                    dir = %dir.display(),
                    error = %e,
                    "ktstr_test: skipping unreadable DirEntry while collecting sidecars",
                );
                continue;
            }
        };
        let path = entry.path();
        if path.is_dir() {
            subdirs.push(path);
            continue;
        }
        try_load(&path, &mut sidecars, &mut parse_errors, &mut io_errors);
    }
    for sub in subdirs {
        let sub_entries = match std::fs::read_dir(&sub) {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!(
                    subdir = %sub.display(),
                    error = %e,
                    "ktstr_test: skipping unreadable subdirectory while collecting sidecars",
                );
                continue;
            }
        };
        for entry in sub_entries {
            let entry = match entry {
                Ok(e) => e,
                Err(e) => {
                    tracing::warn!(
                        subdir = %sub.display(),
                        error = %e,
                        "ktstr_test: skipping unreadable DirEntry in sidecar subdirectory",
                    );
                    continue;
                }
            };
            try_load(
                &entry.path(),
                &mut sidecars,
                &mut parse_errors,
                &mut io_errors,
            );
        }
    }
    (sidecars, parse_errors, io_errors)
}

/// Pool every sidecar JSON under every run directory at `root`.
///
/// Walks each immediate subdirectory of `root` (one per run, named
/// `{kernel}-{project_commit}` by [`sidecar_dir`] where
/// `{project_commit}` is the project tree's HEAD short hex with
/// `-dirty` suffix when the worktree differs from HEAD) and
/// concatenates the sidecars each one yields via
/// `collect_sidecars_with_errors` (per directory, so the per-directory
/// stale-sidecar skip counts aggregate into one pool-wide summary). The
/// result is a flat
/// `Vec<SidecarResult>` covering every recorded run on disk —
/// `cargo ktstr perf-delta`'s pool-driven sourcing reads it
/// once, applies the typed `--a-*` / `--b-*` filters in memory,
/// and partitions the survivors into A/B sides.
///
/// `root` is typically [`runs_root`]; pass an alternate path when
/// comparing archived sidecar trees copied off a CI host (the
/// `--dir` escape hatch on `perf-delta`).
///
/// Returns an empty Vec when `root` does not exist or contains no
/// run directories. Per-run failure (a corrupt sidecar, a partial
/// directory) is counted and skipped — pool-collection never aborts
/// on a single bad file, and emits ONE aggregated
/// `warn_skipped_sidecars` summary for the whole walk rather than a
/// per-file line.
///
/// Performance: this is a full filesystem walk over `root`. On a
/// host with many archived runs (dozens to hundreds), each
/// invocation re-reads every sidecar JSON. The cost is acceptable
/// for the current operator workflow (one comparison per
/// session) but is taskifyable if it becomes a hot path — a
/// directory-name fast-path could skip runs whose
/// `{kernel}-{project_commit}` prefix does not match the active
/// `--a-kernel` / `--b-kernel` filter.
pub fn collect_pool(root: &std::path::Path) -> Vec<SidecarResult> {
    let entries = match std::fs::read_dir(root) {
        Ok(e) => e,
        Err(e) => {
            tracing::warn!(
                root = %root.display(),
                error = %e,
                "ktstr_test: collect_pool cannot read root; returning empty pool",
            );
            return Vec::new();
        }
    };
    let mut pool = Vec::new();
    let mut skipped = 0usize;
    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!(
                    root = %root.display(),
                    error = %e,
                    "ktstr_test: skipping unreadable DirEntry while collecting pool",
                );
                continue;
            }
        };
        let path = entry.path();
        if path.is_dir() {
            // `collect_sidecars_with_errors` already handles "one level of
            // subdirectories for per-job gauntlet layouts" inside each run
            // directory, so the two-level `{root}/{run_dir}/{job_subdir}`
            // shape works without a third walker level. Use the
            // error-returning variant (not `collect_sidecars`, which emits
            // its own per-directory summary) so the skip counts aggregate
            // into ONE pool-wide summary below.
            let (sidecars, parse_errors, _io_errors) = collect_sidecars_with_errors(&path);
            pool.extend(sidecars);
            skipped += parse_errors.len();
        }
    }
    warn_skipped_sidecars(root, skipped);
    pool
}

/// BPF verifier complexity limit (BPF_COMPLEXITY_LIMIT_INSNS).
const VERIFIER_INSN_LIMIT: u32 = 1_000_000;

/// Percentage of the verifier limit that triggers a warning.
const VERIFIER_WARN_PCT: f64 = 75.0;

/// Aggregate BPF verifier stats across sidecars into a summary table.
///
/// verified_insns is deterministic for a given binary, so per-program
/// values are deduplicated (max across observations). Flags programs
/// using >=75% of the 1M verifier complexity limit.
pub(crate) fn format_verifier_stats(sidecars: &[SidecarResult]) -> String {
    use std::collections::BTreeMap;

    let mut by_name: BTreeMap<&str, u32> = BTreeMap::new();
    for sc in sidecars {
        for info in &sc.verifier_stats {
            let entry = by_name.entry(&info.name).or_insert(0);
            *entry = (*entry).max(info.verified_insns);
        }
    }

    if by_name.is_empty() {
        return String::new();
    }

    let mut out = String::from("\n=== BPF VERIFIER STATS ===\n\n");
    out.push_str(&format!(
        "  {:<24} {:>12} {:>8}\n",
        "program", "verified", "limit%"
    ));
    out.push_str(&format!("  {:-<24} {:-<12} {:-<8}\n", "", "", ""));

    let mut warnings = Vec::new();
    let mut total: u64 = 0;

    for (&name, &verified_insns) in &by_name {
        let pct = (verified_insns as f64 / VERIFIER_INSN_LIMIT as f64) * 100.0;
        let flag = if pct >= VERIFIER_WARN_PCT { " !" } else { "" };
        out.push_str(&format!(
            "  {:<24} {:>12} {:>7.1}%{flag}\n",
            name, verified_insns, pct,
        ));
        if pct >= VERIFIER_WARN_PCT {
            warnings.push(format!(
                "  {name}: {pct:.1}% of 1M limit ({verified_insns} verified insns)",
            ));
        }
        total += verified_insns as u64;
    }

    out.push_str(&format!("\n  total verified insns: {total}\n"));

    if !warnings.is_empty() {
        out.push_str("\nWARNING: programs near verifier complexity limit:\n");
        for w in &warnings {
            out.push_str(w);
            out.push('\n');
        }
    }

    out
}

/// Per-test BPF callback profile from monitor prog_stats_deltas.
///
/// Shows per-program invocation count, total CPU time, and average
/// nanoseconds per call. Each test's profile is printed independently.
pub(crate) fn format_callback_profile(sidecars: &[SidecarResult]) -> String {
    let mut out = String::new();

    for sc in sidecars {
        let deltas = match sc
            .monitor
            .as_ref()
            .and_then(|m| m.prog_stats_deltas.as_ref())
        {
            Some(d) if !d.is_empty() => d,
            _ => continue,
        };

        if out.is_empty() {
            out.push_str("\n=== BPF CALLBACK PROFILE ===\n");
        }
        out.push_str(&format!("\n  {} ({}):\n", sc.test_name, sc.topology));
        out.push_str(&format!(
            "    {:<24} {:>12} {:>14} {:>12}\n",
            "program", "cnt", "total_ns", "avg_ns"
        ));
        out.push_str(&format!(
            "    {:-<24} {:-<12} {:-<14} {:-<12}\n",
            "", "", "", ""
        ));
        for d in deltas {
            out.push_str(&format!(
                "    {:<24} {:>12} {:>14} {:>12.0}\n",
                d.name, d.cnt, d.nsecs, d.nsecs_per_call,
            ));
        }
    }

    out
}

/// Aggregate KVM stats across sidecars into a compact summary.
///
/// Averages each stat across all tests that returned `Some(KvmStatsTotals)`.
/// Tests without KVM stats (non-VM tests, old kernels) are excluded
/// from the denominator.
pub(crate) fn format_kvm_stats(sidecars: &[SidecarResult]) -> String {
    let with_stats: Vec<&crate::vmm::KvmStatsTotals> = sidecars
        .iter()
        .filter_map(|sc| sc.kvm_stats.as_ref())
        .collect();

    if with_stats.is_empty() {
        return String::new();
    }

    let n_vms = with_stats.len();

    // Compute cross-VM averages for each stat.
    let vm_avg = |name: &str| -> u64 {
        let sum: u64 = with_stats.iter().map(|d| d.avg(name)).sum();
        sum / n_vms as u64
    };

    let exits = vm_avg("exits");
    let halt = vm_avg("halt_exits");
    let halt_wait_ns = vm_avg("halt_wait_ns");
    let preempted = vm_avg("preemption_reported");
    let signal = vm_avg("signal_exits");
    let hypercalls = vm_avg("hypercalls");

    // Halt poll efficiency across all vCPUs and VMs.
    let total_poll_ok: u64 = with_stats
        .iter()
        .map(|d| d.sum("halt_successful_poll"))
        .sum();
    let total_poll_try: u64 = with_stats
        .iter()
        .map(|d| d.sum("halt_attempted_poll"))
        .sum();

    if exits == 0 {
        return String::new();
    }

    let halt_wait_ms = halt_wait_ns as f64 / 1_000_000.0;
    let poll_pct = if total_poll_try > 0 {
        (total_poll_ok as f64 / total_poll_try as f64) * 100.0
    } else {
        0.0
    };

    let mut out = format!("\n=== KVM STATS (avg across {n_vms} VMs) ===\n\n");
    out.push_str(&format!(
        "  exits/vcpu  {:>7}   halt/vcpu     {:>5}   halt_wait_ms {:>7.1}\n",
        exits, halt, halt_wait_ms,
    ));
    out.push_str(&format!(
        "  poll_ok%    {:>6.1}%   preempted/vcpu {:>4}   signal/vcpu  {:>7}\n",
        poll_pct, preempted, signal,
    ));
    if hypercalls > 0 {
        out.push_str(&format!("  hypercalls/vcpu {:>4}\n", hypercalls));
    }

    // Trust warnings.
    if preempted > 0 {
        let total: u64 = with_stats
            .iter()
            .map(|d| d.sum("preemption_reported"))
            .sum();
        out.push_str(&format!(
            "\n  WARNING: {total} host preemptions detected \
             -- timing results may be unreliable\n",
        ));
    }

    out
}

/// Resolve the sidecar output directory for the current test process.
///
/// Override: `KTSTR_SIDECAR_DIR` (used as-is when non-empty). When
/// the override is set, `serialize_and_write_sidecar` ALSO skips
/// the per-directory pre-clear so any pre-existing sidecars in
/// the operator-chosen directory are preserved verbatim — see
/// `sidecar_dir_override`.
///
/// Default: `{CARGO_TARGET_DIR or "target"}/ktstr/{kernel}-{project_commit}/`,
/// where `{kernel}` is the version detected from `KTSTR_KERNEL`'s
/// metadata (or `"unknown"` when no kernel is set / detection fails)
/// and `{project_commit}` is the project-tree HEAD short hex from
/// `detect_project_commit` (with `-dirty` suffix when the worktree
/// differs from HEAD), or `"unknown"` when the test process is not
/// running inside a git repository or the probe fails. Every sidecar
/// written from the same `cargo ktstr test` invocation lands in the
/// same directory; two runs sharing the same kernel + project commit
/// (e.g. re-running the same suite without committing changes) reuse
/// the same directory, with the second run pre-clearing any
/// `*.ktstr.json` files left by the first via
/// `reset_run_dir_for_session` — the directory is a last-writer-wins
/// snapshot keyed on (kernel, project commit), not an append-only
/// archive of every invocation.
pub fn sidecar_dir() -> PathBuf {
    sidecar_dir_override().unwrap_or_else(resolve_default_sidecar_dir)
}

/// Compute the default-path sidecar directory:
/// `{runs_root}/{kernel}-{project_commit}` where `{kernel}` and
/// `{project_commit}` come from [`detect_kernel_version`] and
/// [`detect_project_commit`] respectively, with `"unknown"`
/// substituted via [`format_run_dirname`] when either probe
/// returns `None`. Emits the one-shot
/// [`warn_unknown_project_commit_once`] stderr warning when the
/// project commit probe falls back to `"unknown"` (operators in
/// this state lose the per-commit run-directory discriminator).
///
/// Shared by [`sidecar_dir`] and the default-path branch of
/// [`serialize_and_write_sidecar`] so both call sites resolve the
/// same kernel/commit/warn/format chain through one place.
/// `serialize_and_write_sidecar` cannot call [`sidecar_dir`]
/// directly because it needs a single-read of
/// [`sidecar_dir_override`] (gated against the env-var flipping
/// mid-call between the dir-resolve and the pre-clear gate); the
/// helper supplies the default-branch body so the override read
/// stays at one site.
fn resolve_default_sidecar_dir() -> PathBuf {
    let kernel = detect_kernel_version();
    let commit = detect_project_commit();
    if commit.is_none() {
        warn_unknown_project_commit_once();
    }
    runs_root().join(format_run_dirname(kernel.as_deref(), commit.as_deref()))
}

/// Build the run-directory leaf name from optional kernel and commit
/// components. `None` collapses to the literal `"unknown"` sentinel
/// in either slot, so a non-git cwd produces `"{kernel}-unknown"`
/// and a missing kernel produces `"unknown-{project_commit}"`. Pure
/// function over the two inputs — no I/O — so unit tests can pin
/// every shape (clean, dirty, missing-kernel, missing-commit, both
/// missing) without driving the [`detect_kernel_version`] /
/// [`detect_project_commit`] OnceLocks.
///
/// SENTINEL ASYMMETRY: the on-disk dirname uses `"unknown"` for
/// missing values, but the in-memory [`SidecarResult::project_commit`]
/// / [`SidecarResult::kernel_version`] fields stay `None` (`null`
/// in JSON). a `project_commit` filter for a specific commit
/// will NOT match a sidecar whose `project_commit` is `None` —
/// omit the filter to include `None`-commit rows. The asymmetry
/// is deliberate: the dirname needs a filesystem-safe sentinel,
/// while the JSON field preserves the original probe outcome for
/// downstream tooling that distinguishes "no probe ran" from
/// "probe ran but found nothing."
fn format_run_dirname(kernel: Option<&str>, commit: Option<&str>) -> String {
    let kernel = kernel.unwrap_or("unknown");
    let commit = commit.unwrap_or("unknown");
    format!("{kernel}-{commit}")
}

/// Resolve the parent directory that holds all test-run subdirectories.
///
/// Resolution order:
/// 1. [`crate::KTSTR_RUNS_ROOT_ENV`] (absolute) — the `cargo ktstr`
///    orchestrator stamps this once at startup so its footer / `stats`
///    / `replay` reads AND the child test processes' sidecar writes
///    resolve the SAME directory regardless of CWD. This is the
///    primary path under `cargo ktstr`.
/// 2. `{cargo target directory}/ktstr` — the raw `cargo test` /
///    `cargo nextest run` fallback, with the target directory resolved
///    by `cargo_target_dir` (which asks cargo, so a
///    `.cargo/config [build] target-dir` is honored, not just
///    `CARGO_TARGET_DIR`). Only this unpinned path pays the
///    one-per-process `cargo metadata` spawn; the orchestrator pins the
///    absolute override above and never reaches it.
///
/// Used by `cargo ktstr stats` / `replay` and the post-run footer to
/// enumerate runs without reconstructing a specific run key.
pub fn runs_root() -> PathBuf {
    if let Some(root) = std::env::var_os(crate::KTSTR_RUNS_ROOT_ENV).filter(|v| !v.is_empty()) {
        return PathBuf::from(root);
    }
    cargo_target_dir().join("ktstr")
}

/// The cargo target directory for the unpinned [`runs_root`] fallback.
///
/// Asks cargo via `cargo metadata --no-deps` and reads
/// `target_directory`, so the answer reflects `CARGO_TARGET_DIR`, a
/// `.cargo/config [build] target-dir`, AND the workspace location — a
/// bare `CARGO_TARGET_DIR`-or-`"target"` read honors the env var but
/// silently ignores a config `target-dir`, landing sidecars under a
/// `target/ktstr` that no other tool writes to. Memoized in a
/// `OnceLock`: the answer depends only on the process's CWD + config
/// (stable for a suite run) and the subprocess spawn is far too costly
/// to repeat per sidecar write. Only the raw `cargo test` / nextest
/// path reaches here; the `cargo ktstr` orchestrator pins
/// [`crate::KTSTR_RUNS_ROOT_ENV`] and short-circuits [`runs_root`]
/// before this runs.
///
/// On any `cargo metadata` failure (cargo absent, non-zero exit,
/// unparsable JSON, missing `target_directory`) it falls back to the
/// bare behavior: `CARGO_TARGET_DIR` when set non-empty, else the
/// CWD-relative `"target"`.
fn cargo_target_dir() -> PathBuf {
    static TARGET_DIR: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    if let Some(cached) = TARGET_DIR.get() {
        return cached.clone();
    }
    let resolved = std::env::current_dir()
        .ok()
        .and_then(|cwd| cargo_metadata_target_dir(&cwd))
        .unwrap_or_else(|| {
            std::env::var("CARGO_TARGET_DIR")
                .ok()
                .filter(|d| !d.is_empty())
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("target"))
        });
    let _ = TARGET_DIR.set(resolved.clone());
    resolved
}

/// Run `cargo metadata --no-deps` in `dir` and return its
/// `target_directory`. `None` on any failure (cargo absent, non-zero
/// exit, unparsable JSON, missing key) so the caller can fall back.
/// The subprocess CWD is set on the `Command` from `dir`, never via a
/// process-wide `chdir`, so tests drive it against a tempdir-workspace
/// without racing the ambient CWD of concurrent tests.
fn cargo_metadata_target_dir(dir: &std::path::Path) -> Option<PathBuf> {
    let output = std::process::Command::new("cargo")
        .args(["metadata", "--format-version", "1", "--no-deps"])
        .current_dir(dir)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).ok()?;
    value.get("target_directory")?.as_str().map(PathBuf::from)
}

/// Predicate: is `entry` a candidate run directory under
/// [`runs_root`]?
///
/// True iff `entry`'s path is a directory AND its filename does
/// NOT begin with a `.` byte. The dotfile filter excludes the
/// flock sentinel subdirectory ([`crate::flock::LOCK_DIR_NAME`] =
/// `.locks`) plus any other operator-created or filesystem-
/// reserved dotfile directories from run-listing walkers
/// ([`newest_run_dir`] here, `sorted_run_entries` in
/// `crate::stats`) so the lock infrastructure does not pollute
/// `cargo ktstr stats list` output or claim the "most recent
/// run" bucket. Checking the first byte directly via
/// `as_encoded_bytes` is OS-string-safe (no UTF-8 round-trip)
/// and short-circuits cleanly on non-UTF-8 names that would
/// confuse a `to_str().starts_with('.')` chain.
///
/// Single source of truth for "is this a run-dir entry?" — both
/// run-listing call sites must pipe through this predicate so a
/// future relocation of `.locks/` (or any other added reserved
/// dotfile) updates one place.
pub(crate) fn is_run_directory(entry: &std::fs::DirEntry) -> bool {
    let path = entry.path();
    if !path.is_dir() {
        return false;
    }
    path.file_name()
        .and_then(|n| n.as_encoded_bytes().first().copied())
        .is_none_or(|b| b != b'.')
}

/// Find the most recently modified run directory under [`runs_root`].
///
/// Used by `cargo ktstr stats last-run` when neither `--dir`,
/// `KTSTR_SIDECAR_DIR`, nor `--kernel` is set: the stats command
/// doesn't itself run a kernel, so it can't reconstruct the
/// `{kernel}-{project_commit}` key that the test process used.
/// Picking the newest subdirectory by mtime mirrors "show me the
/// report from my last test run."
///
/// Dotfile-prefixed entries (notably the flock sentinel
/// subdirectory `.locks/`) are excluded via `is_run_directory`
/// so the lock infrastructure cannot claim the "most recent
/// run" bucket — `.locks/`'s mtime tracks per-write flock
/// activity and would otherwise eclipse the actual newest run
/// dir on every default-path sidecar write.
pub fn newest_run_dir() -> Option<PathBuf> {
    let root = runs_root();
    let entries = std::fs::read_dir(&root).ok()?;
    entries
        .filter_map(|e| e.ok())
        .filter(is_run_directory)
        .max_by_key(|e| e.metadata().and_then(|m| m.modified()).ok())
        .map(|e| e.path())
}

/// One failed test's on-disk artifacts within a single run directory,
/// for the `cargo ktstr test` post-run footer.
///
/// `scheduler` / `topology` come from a FAILING variant's
/// `.ktstr.json` sidecar and are `None` when the test failed BEFORE
/// writing one — e.g. a scheduler BPF-load failure that produced only
/// a placeholder `.failure-dump.json` via
/// [`crate::test_support::eval`] and never reached [`write_sidecar`].
/// Each `Option` path is `Some` only when that artifact exists AND
/// was written in the current run (the mtime gate in
/// [`summarize_run_artifacts`]).
pub(crate) struct FailedTest {
    /// Bare test function name (the artifact filename prefix).
    pub(crate) test_name: String,
    /// Scheduler under test, from a FAILING variant's `.ktstr.json`
    /// sidecar; `None` when no variant sidecar recorded a failure (a
    /// dump-only pre-sidecar failure).
    pub(crate) scheduler: Option<String>,
    /// Topology label, from the same failing variant as `scheduler`;
    /// `None` under the same condition. For a gauntlet test with
    /// multiple failing variants this is a representative one; the
    /// full per-variant set is in `stats_sidecars`.
    pub(crate) topology: Option<String>,
    /// Every primary failure dump for this test, sorted. This includes
    /// attempt-qualified nextest retry archives as well as the final
    /// attempt's canonical path.
    pub(crate) failure_dumps: Vec<PathBuf>,
    /// Every auto-repro failure dump for this test, sorted, including
    /// attempt-qualified nextest retry archives.
    pub(crate) repro_failure_dumps: Vec<PathBuf>,
    /// Every `{test}-{variant_hash}.ktstr.json` stats sidecar for
    /// this test, sorted — one per gauntlet variant (distinct
    /// variant hashes coexist). Empty for a dump-only failure.
    pub(crate) stats_sidecars: Vec<PathBuf>,
    /// `{test}-{variant_hash}.wprof.pb`.
    pub(crate) wprof: Option<PathBuf>,
    /// `{test}-{variant_hash}.repro.wprof.pb` (auto-repro retry).
    pub(crate) repro_wprof: Option<PathBuf>,
    /// True when ANY of this test's variant sidecars is `is_fail()`,
    /// so `cargo ktstr replay --filter <name>` (which selects from
    /// `is_fail` sidecars — see `replay.rs::select_failed_names`)
    /// will re-run it. False for dump-only failures (no sidecar), for
    /// which replay's pool selection finds nothing.
    pub(crate) replayable: bool,
}

/// Per-run-directory artifact summary for the post-run footer.
pub(crate) struct RunDirSummary {
    /// The `{runs_root}/{kernel}-{project_commit}` run directory.
    pub(crate) dir: PathBuf,
    /// Failed tests in this dir, ordered by `test_name`.
    pub(crate) failed: Vec<FailedTest>,
    /// Count of `.ktstr.json` stats sidecars written this run
    /// (every executed VM test that reached [`write_sidecar`],
    /// pass or fail).
    pub(crate) stats_sidecars: usize,
    /// Count of `.wprof.pb` traces written this run (excludes the
    /// `.repro.wprof.pb` auto-repro variant).
    pub(crate) wprof_traces: usize,
    /// Tests skipped because THIS HOST cannot run them, as
    /// `(test_name, class)` where `class` is the host-insufficiency
    /// tag written by [`write_host_skip_marker`]
    /// (`topology_insufficient` / `resource_contention` /
    /// `perf_mode_unavailable`). Ordered by `test_name`. The footer
    /// groups these by class so an operator sees, per class, how many
    /// tests this host could not run and which.
    pub(crate) host_skips: Vec<(String, String)>,
    /// Tests whose auto-repro probe pipeline had a problem, as
    /// `(test_name, reason)` written by [`write_probe_health_marker`]
    /// (trigger failed to load/attach, kprobes attached 0 while
    /// functions were traceable, or probes attached but captured 0
    /// events). Ordered by `test_name`; one footer line each.
    pub(crate) probe_issues: Vec<(String, String)>,
    /// `expect_err` tests whose inversion was satisfied by a scheduler
    /// LOAD/startup failure (not the intended runtime error), written by
    /// [`write_expect_err_load_marker`]. Ordered by `test_name`. The
    /// footer names them on one line so a suite that is silently green
    /// because no scheduler could load becomes visible.
    pub(crate) expect_err_load: Vec<String>,
}

/// The per-test artifact shapes a run directory holds.
enum RunArtifactKind {
    FailureDump,
    ReproFailureDump,
    StatsSidecar,
    Wprof,
    ReproWprof,
    /// `{test}-{hash}.host-skip.json` — a host-insufficiency skip
    /// marker ([`write_host_skip_marker`]). Its JSON body carries the
    /// skip `class`.
    HostSkip,
    /// `{test}-{hash}.probe-health.json` — an auto-repro probe-pipeline
    /// problem marker ([`write_probe_health_marker`]). Its JSON body
    /// carries a short `reason`.
    ProbeHealth,
    /// `{test}-{hash}.expect-err-load.json` — an `expect_err` inversion
    /// that was satisfied by a scheduler LOAD/startup failure rather than
    /// the runtime scheduler error the test intends
    /// ([`write_expect_err_load_marker`]). The presence of the marker is
    /// the signal; its JSON body carries only `test_name`. Surfaced so a
    /// suite that looks green on a kernel where no scheduler can load
    /// (every `expect_err` test's inversion satisfied by the load failure,
    /// not the intended stall/crash) becomes visible.
    ExpectErrLoad,
}

/// Split a `{test}-{16-hex variant hash}` stem into `(test, hash)`.
///
/// Test function names are Rust identifiers (never contain `-`), so the
/// LAST `-` is the variant-hash separator. Falls back to `(stem, 0)`
/// when the trailing token is not a valid 16-hex hash — so a NON-variant
/// dump (a stale pre-variant-keying file, or a future writer that omits
/// the hash) still classifies by its full prefix instead of vanishing
/// (the "no silent drops" rule). The mtime gate already excludes stale
/// prior-run files; the fallback removes the silent-drop risk entirely.
fn split_variant_stem(stem: &str) -> (&str, u64) {
    if let Some((test, hash)) = stem.rsplit_once('-')
        && hash.len() == 16
        && let Ok(h) = u64::from_str_radix(hash, 16)
    {
        (test, h)
    } else {
        (stem, 0)
    }
}

/// Parse a run-directory filename into `(test_name, variant_hash, kind)`.
///
/// Returns `None` for filenames that are not a recognized per-test
/// artifact — `.ktstr.json.tmp.<pid>.<run_id>` atomic-write staging
/// residue, stray non-ktstr files, or a `.ktstr.json` whose stem
/// lacks the `-{16-hex variant hash}` suffix [`write_sidecar`]
/// always appends.
///
/// The `variant_hash` lets the footer correlate each artifact with the
/// SAME-variant sidecar (a gauntlet test's per-preset dumps + sidecars
/// carry distinct hashes): a failure dump whose variant has no parsed
/// sidecar is a per-variant pre-sidecar failure even when a sibling
/// preset passed. failure-dump / wprof names fall back to `(stem, 0)`
/// when un-hashed (see [`split_variant_stem`]); a `.ktstr.json` sidecar
/// is ALWAYS variant-keyed by [`write_sidecar`], so a non-hashed one is
/// malformed and is dropped (`None`).
///
/// Failure dumps may include an attempt archive suffix:
/// `{test}-{hash}.attempt-N.failure-dump.json` or
/// `{test}-{hash}.repro.attempt-N.failure-dump.json`. The attempt is
/// intentionally not part of variant identity: every retry is the
/// same nextest test variant and is correlated with the same stats
/// sidecar hash.
fn classify_run_artifact(name: &str) -> Option<(&str, u64, RunArtifactKind)> {
    if let Some(stem) = name.strip_suffix(".host-skip.json") {
        let (test, hash) = split_variant_stem(stem);
        return Some((test, hash, RunArtifactKind::HostSkip));
    }
    if let Some(stem) = name.strip_suffix(".probe-health.json") {
        let (test, hash) = split_variant_stem(stem);
        return Some((test, hash, RunArtifactKind::ProbeHealth));
    }
    if let Some(stem) = name.strip_suffix(".expect-err-load.json") {
        let (test, hash) = split_variant_stem(stem);
        return Some((test, hash, RunArtifactKind::ExpectErrLoad));
    }
    if let Some(stem) = name.strip_suffix(".failure-dump.json") {
        let stem = match stem.rsplit_once(".attempt-") {
            Some((base, attempt))
                if attempt
                    .parse::<u32>()
                    .ok()
                    .is_some_and(|attempt| attempt > 0) =>
            {
                base
            }
            _ => stem,
        };
        let (stem, kind) = match stem.strip_suffix(".repro") {
            Some(base) => (base, RunArtifactKind::ReproFailureDump),
            None => (stem, RunArtifactKind::FailureDump),
        };
        let (test, hash) = split_variant_stem(stem);
        return Some((test, hash, kind));
    }
    if let Some(stem) = name.strip_suffix(".repro.wprof.pb") {
        let (test, hash) = split_variant_stem(stem);
        return Some((test, hash, RunArtifactKind::ReproWprof));
    }
    if let Some(stem) = name.strip_suffix(".wprof.pb") {
        let (test, hash) = split_variant_stem(stem);
        return Some((test, hash, RunArtifactKind::Wprof));
    }
    if let Some(stem) = name.strip_suffix(".ktstr.json") {
        // A sidecar is ALWAYS `{test}-{16-hex}` ({:016x} in
        // serialize_and_write_sidecar). A stem without a valid hash
        // suffix is a hand-named / malformed file — drop it (unlike the
        // dump arms, there's no un-hashed-sidecar writer to be lenient
        // for).
        let (test, hash) = stem.rsplit_once('-')?;
        if hash.len() == 16
            && let Ok(h) = u64::from_str_radix(hash, 16)
        {
            return Some((test, h, RunArtifactKind::StatsSidecar));
        }
    }
    None
}

/// Summarize the per-test artifacts a single run directory holds,
/// counting only files written at or after `since`.
///
/// The mtime gate is the freshness boundary: a run directory is
/// keyed `{kernel}-{project_commit}` (see [`sidecar_dir`]), so
/// re-running the same suite reuses the directory, and
/// [`reset_run_dir_for_session`] wipes only `*.ktstr.json` — stale
/// `*.failure-dump.json` / `*.wprof.pb` from an earlier run linger.
/// Filtering on `mtime >= since` (where `since` is captured before
/// the nextest build+run begins, so genuine artifacts — written
/// after the build — sort comfortably after it) keeps a stale dump
/// from a prior run from surfacing as a current failure.
///
/// Returns `None` when the directory holds no fresh artifacts (it
/// belongs to an earlier run, or cannot be read).
fn summarize_one_run_dir(
    dir: &std::path::Path,
    since: std::time::SystemTime,
) -> Option<RunDirSummary> {
    use std::collections::{BTreeMap, BTreeSet};
    #[derive(Default)]
    struct Acc {
        // Every per-variant dump, including nextest attempt archives.
        // The fail signal below keys off `dump_hashes`; these vectors
        // retain all concrete evidence paths for the footer.
        failure_dumps: Vec<PathBuf>,
        repro_failure_dumps: Vec<PathBuf>,
        wprof: Option<PathBuf>,
        repro_wprof: Option<PathBuf>,
        // EVERY variant's stats sidecar (distinct variant-hash
        // filenames coexist), so a passing variant cannot mask a
        // failing sibling.
        stats_sidecars: Vec<PathBuf>,
        // OR of `is_fail` across all of this name's variant sidecars.
        // Post-finalize the sidecar carries the FINAL (post-inversion)
        // verdict, so a passing expect_err / expect_auto_repro test
        // reads `false` here even though its scenario failed.
        any_fail: bool,
        // Variant hashes whose stats sidecar PARSED, and variant hashes
        // that left a failure dump. The gate is PER-VARIANT: a dump whose
        // variant has no parsed sidecar is a pre-sidecar failure
        // (scheduler load / VM boot crash) for THAT preset and flags
        // FAILED even when a sibling preset's sidecar parsed; a dump whose
        // variant DID parse a (final, non-failing) sidecar is an
        // expected-failure run whose dump must NOT flag — the sidecar's
        // finalized verdict already classified it. (A gauntlet test's
        // per-preset dumps + sidecars carry distinct variant hashes.)
        parsed_sidecar_hashes: BTreeSet<u64>,
        dump_hashes: BTreeSet<u64>,
        // (scheduler, topology) of the FIRST failing variant seen,
        // for the FAILED block header; `None` when no variant sidecar
        // parsed as a failure (a dump-only pre-sidecar failure).
        fail_variant: Option<(String, String)>,
        // Host-insufficiency skip class from a `.host-skip.json` marker
        // (whichever variant scanned last — a plain skip has one).
        host_skip_class: Option<String>,
        // Short auto-repro probe-pipeline problem reason from a
        // `.probe-health.json` marker (last variant scanned wins).
        probe_health_reason: Option<String>,
        // Presence of an `.expect-err-load.json` marker: this test's
        // `expect_err` inversion was satisfied by a scheduler load
        // failure (any variant sets it — a gauntlet test that load-failed
        // on one preset still surfaces).
        expect_err_load: bool,
    }
    let entries = std::fs::read_dir(dir).ok()?;
    let mut by_test: BTreeMap<String, Acc> = BTreeMap::new();
    let mut stats_sidecars = 0usize;
    let mut wprof_traces = 0usize;
    for entry in entries.flatten() {
        let Ok(meta) = entry.metadata() else {
            continue;
        };
        if !meta.is_file() {
            continue;
        }
        match meta.modified() {
            Ok(m) if m >= since => {}
            _ => continue,
        }
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let Some((test, variant_hash, kind)) = classify_run_artifact(name) else {
            continue;
        };
        let acc = by_test.entry(test.to_string()).or_default();
        match kind {
            RunArtifactKind::FailureDump => {
                acc.dump_hashes.insert(variant_hash);
                acc.failure_dumps.push(path);
            }
            RunArtifactKind::ReproFailureDump => {
                acc.dump_hashes.insert(variant_hash);
                acc.repro_failure_dumps.push(path);
            }
            RunArtifactKind::Wprof => {
                wprof_traces += 1;
                acc.wprof = Some(path);
            }
            RunArtifactKind::ReproWprof => acc.repro_wprof = Some(path),
            RunArtifactKind::StatsSidecar => {
                stats_sidecars += 1;
                // Accumulate EVERY variant (never overwrite by bare
                // name) and OR the fail signal, so one gauntlet
                // variant's pass cannot mask a failing sibling.
                match std::fs::read_to_string(&path)
                    .ok()
                    .and_then(|s| serde_json::from_str::<SidecarResult>(&s).ok())
                {
                    Some(sc) => {
                        acc.parsed_sidecar_hashes.insert(variant_hash);
                        if sc.is_fail() {
                            acc.any_fail = true;
                            if acc.fail_variant.is_none() {
                                acc.fail_variant = Some((sc.scheduler, sc.topology));
                            }
                        }
                    }
                    None => {
                        // Counted in `stats_sidecars` but
                        // unclassifiable. Warn so the count and the
                        // failed list cannot silently disagree (a
                        // corrupt `is_fail` sidecar would otherwise be
                        // swallowed).
                        tracing::warn!(
                            path = %path.display(),
                            "ktstr footer: unreadable/unparseable stats sidecar — \
                             counted but not classified",
                        );
                    }
                }
                acc.stats_sidecars.push(path);
            }
            RunArtifactKind::HostSkip => {
                // Read the marker's `class` field. A malformed / absent
                // body leaves the class unset — the marker's presence
                // alone is not enough to name a class, so it is dropped
                // rather than rendered as an unlabelled skip.
                if let Some(class) = std::fs::read_to_string(&path)
                    .ok()
                    .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
                    .and_then(|v| v.get("class").and_then(|c| c.as_str()).map(String::from))
                {
                    acc.host_skip_class = Some(class);
                }
            }
            RunArtifactKind::ProbeHealth => {
                if let Some(reason) = std::fs::read_to_string(&path)
                    .ok()
                    .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
                    .and_then(|v| v.get("reason").and_then(|r| r.as_str()).map(String::from))
                {
                    acc.probe_health_reason = Some(reason);
                }
            }
            RunArtifactKind::ExpectErrLoad => {
                // Presence alone is the signal — the body carries only
                // `test_name` (already the map key). A malformed body does
                // not suppress it: the marker exists because the writer
                // observed an `expect_err` inversion satisfied by a load
                // failure, and that observability must not hinge on a
                // parse.
                acc.expect_err_load = true;
            }
        }
    }
    if by_test.is_empty() {
        return None;
    }
    let mut failed = Vec::new();
    let mut host_skips = Vec::new();
    let mut probe_issues = Vec::new();
    let mut expect_err_load = Vec::new();
    for (test_name, mut acc) in by_test {
        // Host-insufficiency skip, probe-pipeline problem, and an
        // expect_err inversion satisfied by a load failure are all
        // orthogonal to the FAILED gate below (a host skip is not a
        // failure, a probe problem rides an otherwise-passing auto-repro
        // run, and an expect_err-load test's FINAL verdict is a PASS so it
        // never reaches `failed`), so collect them BEFORE the failure
        // `continue`.
        if let Some(class) = acc.host_skip_class.take() {
            host_skips.push((test_name.clone(), class));
        }
        if let Some(reason) = acc.probe_health_reason.take() {
            probe_issues.push((test_name.clone(), reason));
        }
        if acc.expect_err_load {
            expect_err_load.push(test_name.clone());
        }
        // A test FAILED this run if ANY of its variant sidecars records
        // `is_fail` (the FINAL post-inversion verdict), OR it left a
        // failure dump WITHOUT any parsed sidecar. A dump with no sidecar
        // is a pre-sidecar failure (scheduler load / VM boot) that never
        // reached `write_sidecar` — it must still flag. But a dump
        // alongside a parsed, non-failing sidecar is an expected-failure
        // run (expect_err / expect_auto_repro) whose induced-crash dump
        // must NOT flag: the sidecar's finalized verdict is authoritative.
        // The `is_fail` aggregate covers in-VM assertion failures,
        // including one failing gauntlet variant among passing siblings.
        // Per-variant dump gate: a failure dump whose variant has NO
        // parsed sidecar is a pre-sidecar failure for that preset and
        // flags — even when a SIBLING preset's sidecar parsed. (If every
        // dump-variant has a parsed sidecar, dump_hashes ⊆ parsed and the
        // sidecars' finalized verdicts are authoritative.) Closes the
        // mixed-gauntlet masking the old test-name-granularity gate had.
        let dump_only_failure = !acc.dump_hashes.is_subset(&acc.parsed_sidecar_hashes);
        if !acc.any_fail && !dump_only_failure {
            continue;
        }
        let (scheduler, topology) = match acc.fail_variant {
            Some((sch, topo)) => (Some(sch), Some(topo)),
            None => (None, None),
        };
        // Sort so the rendered footer is deterministic regardless of
        // `read_dir` order.
        acc.failure_dumps.sort();
        acc.repro_failure_dumps.sort();
        acc.stats_sidecars.sort();
        failed.push(FailedTest {
            test_name,
            scheduler,
            topology,
            failure_dumps: acc.failure_dumps,
            repro_failure_dumps: acc.repro_failure_dumps,
            stats_sidecars: acc.stats_sidecars,
            wprof: acc.wprof,
            repro_wprof: acc.repro_wprof,
            replayable: acc.any_fail,
        });
    }
    Some(RunDirSummary {
        dir: dir.to_path_buf(),
        failed,
        stats_sidecars,
        wprof_traces,
        host_skips,
        probe_issues,
        expect_err_load,
    })
}

/// Summarize the artifacts every run directory directly under
/// `runs_root` holds, keeping only files written at or after
/// `since`. Each [`RunDirSummary`] names its failed tests and the
/// concrete artifact path for each, so the `cargo ktstr test`
/// footer can point an operator at the exact file for the exact
/// test that failed rather than a directory + glob legend.
///
/// `since` is the wall-clock instant captured before the nextest
/// build+run; the mtime gate it drives is what excludes stale
/// artifacts left in a reused run directory (see
/// [`summarize_one_run_dir`]). Directories are returned sorted by
/// path so multi-kernel gauntlet output renders deterministically.
pub(crate) fn summarize_run_artifacts(
    runs_root: &std::path::Path,
    since: std::time::SystemTime,
) -> Vec<RunDirSummary> {
    let Ok(entries) = std::fs::read_dir(runs_root) else {
        return Vec::new();
    };
    let mut out: Vec<RunDirSummary> = entries
        .flatten()
        .filter(is_run_directory)
        .filter_map(|e| summarize_one_run_dir(&e.path(), since))
        .collect();
    out.sort_by(|a, b| a.dir.cmp(&b.dir));
    out
}

/// Render the `cargo ktstr test` post-run footer: for each run
/// directory written at or after `since`, name every FAILED test
/// and the concrete path to each of its artifacts (failure dump,
/// auto-repro dump, stats sidecar, wprof trace), plus a per-dir
/// count of stats sidecars and wprof traces.
///
/// Returns the empty string when no run directory under `runs_root`
/// holds fresh artifacts — a host-only run (no VM tests) writes no
/// sidecars, so there is nothing to point at and the caller emits
/// no footer.
///
/// A test is listed FAILED when it left a failure dump (real or
/// placeholder) or an `is_fail` stats sidecar. This is NOT an
/// exhaustive failure list: a failure that writes neither — a
/// `builder.build()` / `vm.run()` error, a pre-build host error
/// (kvm probe, kernel/scheduler resolve, validation), a host panic,
/// or an unparseable guest result — leaves no on-disk artifact and
/// no entry here. The caller (`cargo_ktstr::run_cargo`) treats
/// nextest's own exit status as the authoritative pass/fail signal
/// and notes, when nextest reports failures, that any failure
/// without an entry left no artifact.
///
/// This replaces a directory + `*.glob` legend that carried no
/// test attribution: a reused run directory mixes artifacts from
/// many tests (and, before the mtime gate, prior runs), so a glob
/// legend pointed an operator at the directory and left them to
/// guess which `*.failure-dump.json` belonged to the test that
/// just failed.
pub fn format_run_artifact_footer(
    runs_root: &std::path::Path,
    since: std::time::SystemTime,
) -> String {
    let summaries = summarize_run_artifacts(runs_root, since);
    if summaries.is_empty() {
        return String::new();
    }
    let mut out = String::new();
    out.push_str("\ncargo ktstr: test outputs\n");
    for s in &summaries {
        out.push_str(&format!("  {}\n", s.dir.display()));
        for f in &s.failed {
            // scheduler/topology are set together (both from one
            // failing variant) or both absent (dump-only) — see
            // `summarize_one_run_dir`; no mixed arm is reachable.
            let variant = match (&f.scheduler, &f.topology) {
                (Some(sch), Some(topo)) => format!("  [{sch} {topo}]"),
                _ => String::new(),
            };
            out.push_str(&format!("    FAILED  {}{variant}\n", f.test_name));
            for p in &f.failure_dumps {
                out.push_str(&format!("      {:<13} {}\n", "failure dump", p.display()));
            }
            for p in &f.repro_failure_dumps {
                out.push_str(&format!("      {:<13} {}\n", "repro dump", p.display()));
            }
            for p in &f.stats_sidecars {
                out.push_str(&format!("      {:<13} {}\n", "stats", p.display()));
            }
            if let Some(p) = &f.wprof {
                out.push_str(&format!("      {:<13} {}\n", "wprof", p.display()));
            }
            if let Some(p) = &f.repro_wprof {
                out.push_str(&format!("      {:<13} {}\n", "repro wprof", p.display()));
            }
            if f.replayable {
                out.push_str(&format!(
                    "      {:<13} cargo ktstr replay --filter {} --exec\n",
                    "replay", f.test_name,
                ));
            }
        }
        out.push_str(&render_host_skips(&s.host_skips));
        out.push_str(&render_probe_issues(&s.probe_issues));
        out.push_str(&render_expect_err_load(&s.expect_err_load));
        // Fold the "run stats-last-run for the analysis" discoverability
        // hint into the existing count line, but only when this run
        // actually wrote a stats sidecar — pointing at an empty analysis
        // would be noise. The gauntlet analysis blob is no longer
        // auto-printed after a test run; this line is how an operator
        // finds it.
        if s.stats_sidecars > 0 {
            out.push_str(&format!(
                "    ({} stats sidecar(s), {} wprof trace(s) written this run) \
                 — run `cargo ktstr stats last-run` for the gauntlet analysis\n",
                s.stats_sidecars, s.wprof_traces,
            ));
        } else {
            out.push_str(&format!(
                "    ({} stats sidecar(s), {} wprof trace(s) written this run)\n",
                s.stats_sidecars, s.wprof_traces,
            ));
        }
    }
    out
}

/// Render the host-topology-skip block: one line per skip class, each
/// naming its count and the tests this host could not run. Returns the
/// empty string when there are no host skips (the common case), so the
/// footer stays silent unless this host actually forced a skip.
///
/// `skips` is `(test_name, class)` collected from the run dir's
/// `.host-skip.json` markers; classes are grouped (BTreeMap →
/// class-alphabetical, deterministic) and each group's tests are listed
/// in the input order (already `test_name`-sorted by
/// `summarize_one_run_dir`'s BTreeMap walk).
fn render_host_skips(skips: &[(String, String)]) -> String {
    if skips.is_empty() {
        return String::new();
    }
    let mut by_class: std::collections::BTreeMap<&str, Vec<&str>> =
        std::collections::BTreeMap::new();
    for (test, class) in skips {
        by_class.entry(class).or_default().push(test);
    }
    let mut out = String::from("    host-skipped (this host cannot run):\n");
    for (class, tests) in by_class {
        out.push_str(&format!(
            "      {} ({}): {}\n",
            class,
            tests.len(),
            tests.join(", "),
        ));
    }
    out
}

/// Render the auto-repro probe-health block: one line per test whose
/// probe pipeline had a problem, with its short reason. Empty string
/// when every probe pipeline was healthy, so the footer says nothing on
/// a clean run.
fn render_probe_issues(issues: &[(String, String)]) -> String {
    if issues.is_empty() {
        return String::new();
    }
    let mut out = String::from("    probe pipeline problems:\n");
    for (test, reason) in issues {
        out.push_str(&format!("      {test}: {reason}\n"));
    }
    out
}

/// Render the `expect_err`-satisfied-by-load-failure block: one line
/// naming every test whose `expect_err` inversion passed because the
/// scheduler FAILED TO LOAD/attach, not because it hit the runtime error
/// the test intends. Empty string when there are none, so the footer is
/// silent on a normal run.
///
/// This is an ADVISORY, not a failure: each named test still PASSED
/// (`expect_err` inverted the error). But a whole suite in this block is
/// the tell that no scheduler could load on this kernel — the suite is
/// green without having exercised anything. See
/// [`write_expect_err_load_marker`].
fn render_expect_err_load(tests: &[String]) -> String {
    if tests.is_empty() {
        return String::new();
    }
    format!(
        "    expect_err satisfied by scheduler load failure (not a runtime error): {}\n",
        tests.join(", "),
    )
}

/// Detect the kernel version associated with the current test run.
///
/// Routes through [`crate::ktstr_kernel_env`] for the raw env value
/// and [`crate::kernel_path::KernelId`] for variant dispatch so the
/// three [`crate::kernel_path::KernelId`] variants are honoured symmetrically:
///
/// - `KernelId::Path(dir)`: read `metadata.json` (cache entry
///   layout) or `include/config/kernel.release` (source tree
///   layout). Unchanged from the previous behaviour.
/// - `KernelId::Version(ver)`: the user asked for a specific
///   version — return it directly. No cache access needed; a
///   version string IS a version string.
/// - `KernelId::CacheKey(key)`: look up the cache entry and
///   return `entry.metadata.version`. The previous code path
///   silently treated the key as a directory name and read
///   `<cwd>/<key>/metadata.json`, which never matched — producing
///   `None` + `sidecar_dir()` using the `"unknown"` fallback even
///   though the cache metadata already carried the version.
///
/// Returns `None` when the env var is unset, or when the env
/// resolves to a variant whose underlying source doesn't yield a
/// version string (e.g. a Path whose metadata.json / kernel.release
/// are both absent, or a CacheKey with no cache hit).
pub(crate) fn detect_kernel_version() -> Option<String> {
    use crate::kernel_path::KernelId;
    let raw = crate::ktstr_kernel_env()?;
    match KernelId::parse(&raw) {
        KernelId::Path(_) => {
            let p = std::path::Path::new(&raw);
            let meta_path = p.join("metadata.json");
            if let Ok(data) = std::fs::read_to_string(&meta_path)
                && let Ok(meta) = serde_json::from_str::<crate::cache::KernelMetadata>(&data)
            {
                return meta.version;
            }
            let ver_path = p.join("include/config/kernel.release");
            if let Ok(v) = std::fs::read_to_string(ver_path) {
                let v = v.trim();
                if !v.is_empty() {
                    return Some(v.to_string());
                }
            }
            None
        }
        KernelId::Version(ver) => Some(ver),
        KernelId::CacheKey(key) => {
            let cache = crate::cache::CacheDir::new().ok()?;
            let entry = cache.lookup(&key)?;
            entry.metadata.version
        }
        // Multi-kernel specs in KTSTR_KERNEL never reach this
        // function in production — `find_kernel`'s env reader bails
        // before sidecar writing happens. This arm is defensive: if
        // the env value is somehow a range or git spec, return
        // `None` rather than guessing one endpoint, and the sidecar
        // record will leave `kernel_version` as null.
        KernelId::Range { .. }
        | KernelId::Git { .. }
        | KernelId::Package { .. }
        | KernelId::Distro { .. } => None,
    }
}

/// Detect the ktstr project's git HEAD at sidecar-write time.
///
/// Walks up from the test process's current working directory via
/// `gix::discover` to find an enclosing repository, then reads HEAD
/// short-hex (7 chars via `oid::to_hex_with_len(7)`) and appends
/// `-dirty` when index-vs-HEAD or worktree-vs-index changes are
/// observed. Submodules are ignored
/// (`Submodule::Given { ignore: All }`).
///
/// Dirt-detection runs through the shared [`repo_is_dirty`]
/// helper (peel HEAD to its tree, diff tree-vs-index, then
/// `status()` for worktree-vs-index, submodules skipped); see its
/// doc for cascade details. The cascade is similar in spirit to
/// [`crate::fetch::local_source`]'s dirt probe but deliberately
/// diverges in missing-index handling: the sidecar path silently
/// degrades a missing index leg to "treat as clean" so metadata
/// probes never gate sidecar writes, whereas `local_source`'s
/// cache-key path treats every leg as load-bearing. The HASH
/// REPRESENTATION also DIFFERS: `fetch::local_source` DROPS the
/// short hash entirely on dirty (returns `None`) because the
/// commit no longer describes the build input the cache key
/// embeds — publishing a stale hash there would misidentify the
/// build. This helper KEEPS the hash with a `-dirty` suffix
/// instead because the sidecar's `project_commit` is a debugging
/// breadcrumb (operator-readable identity, not a cache-key input);
/// the hash plus dirty flag carries strictly more information
/// than `None` for the operator's "which ktstr commit did this
/// sidecar come from?" question.
///
/// Returns `None` when:
/// - `current_dir()` cannot be resolved (process has no valid
///   cwd — extremely rare; happens only for processes whose cwd
///   was rmdir'd while alive);
/// - cwd is not inside any git repository (`gix::discover` fails);
/// - HEAD cannot be read (an unborn HEAD on a fresh `git init`
///   with zero commits, or a corrupt repository).
///
/// Returns `Some(short_hash)` (without the `-dirty` suffix) when
/// the HEAD read succeeds but a downstream dirt-detection call
/// fails — including a missing index, an unreadable working tree,
/// or `head_tree()` failure. Each failed leg degrades to "treat
/// as clean" rather than aborting the probe, because metadata
/// must not gate sidecar writes.
///
/// `None` is the documented fallback — sidecar writing must not
/// abort because of a metadata probe failure. Stats tooling that
/// reads `project_commit` already tolerates `None` rows by
/// treating them as wildcards (no `--project-commit` filter narrowing
/// applies).
///
/// `gix::discover` is preferred over `gix::open` because tests can
/// be launched from a subdirectory of the repo (e.g.
/// `cd src && cargo test`); `discover` walks parents until it
/// finds the `.git` marker, while `open` requires the exact root
/// path. The walk is cheap — a few stat() calls bounded by the
/// depth of the cwd inside the repo.
///
/// `env!("CARGO_MANIFEST_DIR")` is deliberately NOT used here:
/// `env!` resolves at compile time and bakes the build-host's
/// absolute manifest path into the binary's read-only data
/// segment, leaking the build environment into every published
/// artifact. Resolving cwd at runtime instead means the recorded
/// commit reflects the project tree the test was launched FROM —
/// for a scheduler crate using ktstr as a dev-dependency, this is
/// the scheduler crate's commit, not ktstr's. That is the more
/// accurate semantic anyway: "what code produced this sidecar"
/// depends on the cwd at test launch (which crate is exercising
/// ktstr), not the build host.
#[doc(hidden)]
pub fn detect_project_commit() -> Option<String> {
    // Explicit override: an orchestrator (perf-delta) that checked the
    // project tree out WITHOUT a `.git` — a plain gix checkout of a baseline
    // commit into a temp dir — passes the commit label via
    // KTSTR_PROJECT_COMMIT_ENV so the sidecar records it verbatim instead of
    // a `gix::discover` that would resolve to the wrong repo (or none). It is
    // also set on the HEAD run so the recorded `project_commit` equals the
    // exact label perf-delta filters the pool on, closing the -dirty-suffix
    // mismatch between the filter (`short_hash`) and this recorder. Empty is
    // treated as unset. Mirrors the KTSTR_KERNEL_COMMIT_ENV override.
    if let Ok(explicit) = std::env::var(crate::KTSTR_PROJECT_COMMIT_ENV)
        && !explicit.is_empty()
    {
        return Some(explicit);
    }
    // Per-process memoization of the SUCCESS case only.
    //
    // The cwd is stable for the lifetime of a test process (no
    // caller mutates it), and the project tree's HEAD plus dirty
    // state cannot change underneath us without an explicit user
    // action that's outside the scope of any individual sidecar
    // write. Gauntlet runs invoke this function once per sidecar —
    // thousands of times per process — so caching the resolved
    // hash collapses every post-first successful call to a
    // `Clone`. The probe itself does ~3 syscalls (gix discover +
    // head_id + status) which dominate the sidecar-write critical
    // path; eliminating that cost on the hot path is the only
    // meaningful perf win available here.
    //
    // FAILURE IS NOT CACHED: a `None` probe outcome (no git repo
    // discoverable from cwd, unborn HEAD, transient FS / gix open
    // failure) does NOT seed the OnceLock. A FIRST call from a
    // momentarily-broken context (e.g. a test that swapped CWD via
    // some indirect path before ever calling
    // `detect_project_commit`, or a transient I/O hiccup during
    // `gix::discover`) would otherwise lock in `None` for the
    // rest of the process — every subsequent sidecar would land
    // under `target/ktstr/{kernel}-unknown/` even though the
    // commit IS resolvable from a healthy cwd. Retrying on failure
    // costs the same ~3 syscalls the success case pays once; the
    // re-probe only fires while the answer is still unknown.
    //
    // CACHE DOES NOT INVALIDATE on success: a user who commits /
    // amends / resets the project tree mid-run and expects the
    // new HEAD to surface in subsequent sidecars will see stale
    // values. This is acceptable — the
    // project tree is treated as stable-enough for a single suite
    // run; callers mutating the tree during a run own the
    // consequences.
    static PROJECT_COMMIT: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    if let Some(cached) = PROJECT_COMMIT.get() {
        return Some(cached.clone());
    }
    let cwd = std::env::current_dir().ok()?;
    let probed = detect_commit_at(&cwd)?;
    // `set` on a hot OnceLock is a no-op `Err` — safe to ignore.
    // First successful caller wins; a second concurrent caller's
    // identical hash discards harmlessly.
    let _ = PROJECT_COMMIT.set(probed.clone());
    Some(probed)
}

/// Path-taking core of [`detect_project_commit`]. Factored out so
/// unit tests can drive the full branch matrix (clean repo, dirty
/// repo, non-git directory, unborn HEAD, concurrent calls) against
/// `gix::init`-built fixtures in tempdirs without mutating the
/// process-wide `current_dir`. The public entry point reads `cwd`
/// once and delegates here.
///
/// `gix::discover` walks parents until it finds a `.git` marker —
/// tests can be launched from a subdirectory of the repo (e.g.
/// `cd src && cargo test`); the parent walk handles that, where
/// `gix::open` would require the exact root. The
/// open-vs-discover distinction is the ONLY difference between
/// this function and [`detect_kernel_commit`]; the post-open
/// "read HEAD, format short hex, append `-dirty` on dirt" body
/// lives in the shared [`commit_with_dirty_suffix`] helper.
fn detect_commit_at(path: &std::path::Path) -> Option<String> {
    let repo = gix::discover(path).ok()?;
    commit_with_dirty_suffix(&repo)
}

/// Shared post-open body for [`detect_commit_at`] and
/// [`detect_kernel_commit`]: read `repo.head_id()`, format the
/// 7-char short hex, and append `-dirty` when [`repo_is_dirty`]
/// returns `Some(true)`.
///
/// Returns `None` when `head_id()` fails (unborn HEAD on a fresh
/// `gix::init` with zero commits, or a corrupt repository) — the
/// short-hex cannot be formed.
///
/// Returns `Some(short_hash)` (without `-dirty`) when the HEAD
/// read succeeds but the [`repo_is_dirty`] probe returns `None`
/// (HEAD-tree peel failure). This matches the documented "treat
/// as clean on probe failure" degradation: metadata probes must
/// not gate sidecar writes, so a probe failure flows through as
/// "clean" rather than aborting.
///
/// `to_hex_with_len(7)` produces a `HexDisplay` that formats 7
/// hex chars without the 40-char intermediate `format!("{}")`
/// allocation. `Id` derefs to `oid` (gix-hash) which owns the
/// method.
///
/// CALL SITES diverge ONLY on the open mode (`gix::discover` for
/// the project commit, `gix::open` for the kernel commit). The
/// helper takes a `&Repository` so each caller picks the open
/// strategy that matches its semantics: project commit walks
/// parents (cwd may be inside a subdir of the repo); kernel
/// commit demands the explicit root (the kernel directory is
/// not walked-up to avoid resolving the parent ktstr repo).
fn commit_with_dirty_suffix(repo: &gix::Repository) -> Option<String> {
    let head = repo.head_id().ok()?;
    let short_hash = head.to_hex_with_len(7).to_string();
    if repo_is_dirty(repo).unwrap_or(false) {
        Some(format!("{short_hash}-dirty"))
    } else {
        Some(short_hash)
    }
}

/// Probe whether a gix repository's working tree differs from its
/// HEAD commit, ignoring submodules.
///
/// Returns `Some(true)` when the index differs from the HEAD tree
/// or the worktree differs from the index for any tracked file;
/// `Some(false)` when neither leg observed a difference; `None`
/// when the HEAD-tree peel itself failed (HEAD points at something
/// that cannot be read as a tree).
///
/// Callers in [`detect_commit_at`] / [`detect_kernel_commit`]
/// degrade `None` to "treat as clean" via `unwrap_or(false)` so
/// metadata probes never gate sidecar writes.
///
/// PROBE LEGS:
/// - tree-vs-index: peel HEAD to its tree, then `tree_index_status`
///   diff against the on-disk index. `repo.index()` returning Err
///   (missing index — partially-checked-out clones, or fresh
///   `git init` before the first commit) silently leaves the
///   index-dirty leg false. `index_or_empty()` is deliberately
///   NOT used because it would substitute an empty index and the
///   diff would flag every tracked file as "deleted from index",
///   tripping false-dirty.
/// - index-vs-worktree: `repo.status()` configured with
///   `Submodule::Given { ignore: All }` so submodule worktree
///   state is skipped. Short-circuited when the tree-vs-index leg
///   already flipped dirty: the result only needs one positive
///   signal, so a known-dirty index makes the worktree walk
///   redundant. Matches the equivalent short-circuit in
///   [`crate::fetch::local_source`].
///
/// FAILURE DEGRADATION: any individual leg failure (missing index,
/// `repo.status()` failure, `into_index_worktree_iter()` failure)
/// silently degrades that leg to "no signal" rather than aborting.
/// The function only returns `None` when the HEAD-tree peel
/// fails, because at that point neither leg can run at all.
///
/// `pub` (not `pub(crate)`) because `cargo-ktstr.rs` is a
/// separate `[[bin]]` crate that consumes `ktstr` as an
/// external dependency and needs this helper to compute the
/// `-dirty` suffix in
/// the baseline/HEAD commit in `cargo ktstr perf-delta`. Hidden
/// from rustdoc via `#[doc(hidden)]` because it is a probe-
/// style helper without a stable API contract — external
/// consumers should not depend on it.
#[doc(hidden)]
pub fn repo_is_dirty(repo: &gix::Repository) -> Option<bool> {
    let head_tree_id = repo.head_tree().ok()?.id;

    let mut index_dirty = false;
    if let Ok(index) = repo.index() {
        let _ = repo.tree_index_status(
            &head_tree_id,
            &index,
            None,
            gix::status::tree_index::TrackRenames::Disabled,
            |_, _, _| {
                index_dirty = true;
                Ok::<_, std::convert::Infallible>(std::ops::ControlFlow::Break(()))
            },
        );
    }

    let worktree_dirty = if index_dirty {
        false
    } else {
        repo.status(gix::progress::Discard)
            .ok()
            .and_then(|s| {
                s.index_worktree_rewrites(None)
                    .index_worktree_submodules(gix::status::Submodule::Given {
                        ignore: gix::submodule::config::Ignore::All,
                        check_dirty: false,
                    })
                    .index_worktree_options_mut(|opts| {
                        configure_dirty_status_options(opts);
                    })
                    .into_index_worktree_iter(Vec::new())
                    .ok()
                    .map(crate::git_status::consume_has_any)
            })
            .unwrap_or(false)
    };

    Some(index_dirty || worktree_dirty)
}

/// Configure the tracked-file-only sidecar probe while reusing the shared
/// gix worker policy.
fn configure_dirty_status_options(options: &mut gix::status::index_worktree::Options) {
    options.dirwalk_options = None;
    crate::git_status::configure_index_worktree_parallelism(options);
}

/// Detect the kernel SOURCE TREE's git HEAD at sidecar-write time.
///
/// `kernel_dir` is the explicit kernel source directory — typically
/// resolved from `KTSTR_KERNEL` for `KernelId::Path`, or from the
/// cache entry's `KernelSource::Local::source_tree_path` when
/// `KTSTR_KERNEL` is a Version / CacheKey whose underlying build
/// recorded a local tree. Uses `gix::open(kernel_dir)` (NOT
/// `gix::discover`) because the kernel directory is explicit, not
/// walked-up: the parent walk that `discover` performs would
/// resolve to whichever ancestor `.git` it found first, which
/// might be the ktstr project's repo when `kernel_dir` is a
/// non-git subdirectory inside it. `open` requires `kernel_dir`
/// itself to be the repo root, which is the documented invariant
/// for kernel checkouts.
///
/// Reads HEAD short-hex (7 chars via `oid::to_hex_with_len(7)`)
/// and appends `-dirty` when index-vs-HEAD or worktree-vs-index
/// changes are observed. Dirt-detection runs through the shared
/// [`repo_is_dirty`] helper (submodules skipped via
/// `Submodule::Given { ignore: All }`); see its doc for cascade
/// details. The cascade matches [`detect_project_commit`] and is
/// similar in spirit to [`crate::fetch::local_source`] but
/// deliberately diverges in missing-index handling: the sidecar
/// path silently degrades a missing index leg to "treat as
/// clean" so metadata probes never gate sidecar writes, whereas
/// `local_source`'s cache-key path treats every leg as
/// load-bearing. Same "treat as clean on probe failure"
/// degradation rules apply otherwise: a missing index, an
/// unreadable worktree, or `head_tree()` failure each fall
/// through as "clean" rather than aborting the probe — metadata
/// must not gate sidecar writes.
///
/// HASH REPRESENTATION matches [`detect_project_commit`]: keeps
/// the hash with `-dirty` appended (operator-readable identity).
/// Distinct from [`crate::fetch::local_source`], which DROPS the
/// hash on dirty because the commit no longer describes the
/// build INPUT for cache-key purposes.
///
/// Returns `None` when:
/// - `kernel_dir` is not a git repository (`gix::open` fails);
/// - HEAD cannot be read (unborn HEAD on a fresh `git init` with
///   zero commits, or a corrupt repository).
///
/// Returns `Some(short_hash)` (without the `-dirty` suffix) when
/// the HEAD read succeeds but a downstream dirt-detection call
/// fails — including a missing index, an unreadable working
/// tree, or `head_tree()` failure. Each failed leg degrades to
/// "treat as clean" rather than aborting the probe, because
/// metadata must not gate sidecar writes.
///
/// `pub` (not `pub(crate)`) + `#[doc(hidden)]` because cargo-ktstr is a
/// separate `[[bin]]` crate that consumes `ktstr` as a dependency. Its
/// resolved-kernel provenance helper falls back to this walk for raw source
/// trees and legacy cache entries. Hidden from rustdoc — a probe helper with
/// no stable API contract.
#[doc(hidden)]
pub fn detect_kernel_commit(kernel_dir: &std::path::Path) -> Option<String> {
    // Per-process, path-keyed memoization of the SUCCESS case
    // only. Same rationale as `detect_project_commit`: gauntlet
    // runs invoke this function once per sidecar — thousands of
    // times — and the kernel tree's HEAD plus dirty state cannot
    // change underneath us mid-suite without an explicit user
    // action outside any sidecar's control. The path key handles
    // the fixture-test case where unit tests rotate through
    // synthetic `tempfile::TempDir` kernel paths in the same
    // process; each distinct path memoizes independently.
    //
    // `Mutex<HashMap>` rather than `OnceLock` because the input
    // is parameterized on `kernel_dir` — a `OnceLock` collapses
    // every input to one cached result, which would conflate
    // different kernel directories into a single value.
    // Contention is bounded: post-warm reads are O(1) hash
    // lookups against a near-empty map (in production typically
    // ONE kernel per process), and the mutex is held only for
    // the duration of the lookup + insert.
    //
    // FAILURE IS NOT CACHED: a `None` probe outcome (kernel_dir
    // is not a git repo, unborn HEAD, transient `gix::open`
    // failure) does NOT seed the cache. Caching `None` would lock
    // in `unknown` for every subsequent sidecar even after the
    // condition resolves (e.g. a kernel directory that becomes a
    // valid checkout mid-suite, or a flaky FS that recovers).
    // Re-probing on failure costs the same gix-open + dirt-walk
    // the success case pays once; the re-probe only fires while
    // the answer is still unknown for that path.
    //
    // Mutex poisoning recovery: a panic mid-probe could poison
    // the lock; acquiring via
    // [`crate::sync::MutexExt::lock_unpoisoned`] returns the
    // guard regardless of poison state so a future caller doesn't
    // fail catastrophically. The cached map is just a HashMap of
    // owned strings; no invariant beyond "key→value mapping" can
    // be broken by an interrupted probe.
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::sync::{Mutex, OnceLock};
    static KERNEL_COMMIT_CACHE: OnceLock<Mutex<HashMap<PathBuf, String>>> = OnceLock::new();
    // Canonicalize the cache key so two paths that resolve to the
    // same on-disk directory share one entry. Without this, a
    // symlinked alias (`./linux` symlinked to `/abs/.../linux`)
    // and the resolved target would each populate their own slot,
    // re-running the gix-open + dirt-walk on every alias and
    // defeating the memoization. `canonicalize` resolves symlinks,
    // collapses `..` / `.`, and yields the absolute path the
    // kernel actually lives at. Falls back to the raw path on
    // canonicalize failure (e.g. caller passed a non-existent
    // `kernel_dir`) — gix::open will fail downstream and re-probe
    // each call until the path becomes resolvable.
    let cache_key = kernel_dir
        .canonicalize()
        .unwrap_or_else(|_| kernel_dir.to_path_buf());
    let cache = KERNEL_COMMIT_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    {
        let guard = cache.lock_unpoisoned();
        if let Some(cached) = guard.get(&cache_key) {
            return Some(cached.clone());
        }
    }
    // `gix::open` (NOT `gix::discover`) — `kernel_dir` must BE the
    // repo root. Without this the parent walk could resolve to the
    // ktstr project's own `.git` when `kernel_dir` is a non-git
    // subdirectory inside the ktstr checkout. The
    // open-vs-discover distinction is the ONLY difference between
    // this function and [`detect_commit_at`]; the post-open
    // "read HEAD, format short hex, append `-dirty` on dirt" body
    // lives in the shared [`commit_with_dirty_suffix`] helper.
    //
    // Open against `kernel_dir` (the caller-supplied path) rather
    // than `cache_key`. The two paths point at the same on-disk
    // repo by construction (canonicalize resolves to the same
    // place), so gix opens the same repository either way; passing
    // the original keeps any user-facing diagnostics (gix's
    // internal error chain) consistent with the input shape.
    let result = gix::open(kernel_dir)
        .ok()
        .and_then(|repo| commit_with_dirty_suffix(&repo));
    if let Some(ref hash) = result {
        let mut guard = cache.lock_unpoisoned();
        // First successful caller wins; a concurrent caller's
        // identical hash would overwrite harmlessly because
        // success is deterministic for a given (canonicalized
        // path, HEAD, dirty state) tuple.
        guard.insert(cache_key, hash.clone());
    }
    result
}

/// Environment variable CI runners set to mark sidecars they produce
/// as `"ci"`-source. Any non-empty value flips the tag; empty string
/// is treated as unset so a defensively-cleared variable does not
/// accidentally classify a developer run as CI.
///
/// Read at sidecar-write time by [`detect_run_source`]; matches the
/// `KTSTR_KERNEL` / `KTSTR_CACHE_DIR` env-name convention so the
/// full set of ktstr-controlled env vars is `KTSTR_*`-prefixed.
pub const KTSTR_CI_ENV: &str = "KTSTR_CI";

/// Tag value written to [`SidecarResult::run_source`] for sidecars
/// produced under [`KTSTR_CI_ENV`].
pub const SIDECAR_RUN_SOURCE_CI: &str = "ci";

/// Tag value written to [`SidecarResult::run_source`] for sidecars
/// produced without [`KTSTR_CI_ENV`] — the developer-machine
/// default.
pub const SIDECAR_RUN_SOURCE_LOCAL: &str = "local";

/// Tag value applied to [`SidecarResult::run_source`] /
/// [`GauntletRow::run_source`](crate::stats::GauntletRow::run_source)
/// at LOAD time when the consumer pulls sidecars from a non-default
/// pool root via `cargo ktstr stats show-host --dir` /
/// `cargo ktstr stats list-values --dir`. NEVER written by
/// [`write_sidecar`] — the writer cannot know the file will later
/// be moved off-host. See [`apply_archive_source_override`].
pub const SIDECAR_RUN_SOURCE_ARCHIVE: &str = "archive";

/// Read [`KTSTR_CI_ENV`] and classify the run as `"ci"` (when the
/// env var is set non-empty) or `"local"` (the default for any
/// developer-driven invocation). Empty-string env values count as
/// unset — see [`KTSTR_CI_ENV`] for rationale.
///
/// Returns `Some(_)` unconditionally because every sidecar producer
/// is, by construction, either local or CI; an `Option` return
/// keeps the field shape symmetric with the other nullable
/// `SidecarResult` fields and reserves room for a future "unknown"
/// arm without a serde-version bump.
pub(crate) fn detect_run_source() -> Option<String> {
    match std::env::var(KTSTR_CI_ENV) {
        Ok(v) if !v.is_empty() => Some(SIDECAR_RUN_SOURCE_CI.to_string()),
        _ => Some(SIDECAR_RUN_SOURCE_LOCAL.to_string()),
    }
}

/// Override every sidecar's `run_source` field to
/// [`SIDECAR_RUN_SOURCE_ARCHIVE`] when the consumer pulled the pool
/// from a non-default root via `--dir`. Called at the boundary
/// between [`collect_pool`] and the downstream stats pipeline so
/// on-disk values stay untouched while the in-memory pool reflects
/// the operator's intent: "these sidecars were copied off another
/// host; treat them as archives, not as the local-machine record."
///
/// Mutation strategy is in-place rewrite of the entire `run_source`
/// field — the `"local"` / `"ci"` distinction is meaningful on the
/// PRODUCING host but irrelevant once the sidecars have been
/// moved off, where the only useful classification is "archived
/// elsewhere." Operators who need to retain the producer-side
/// distinction inside an archive bucket can keep `--dir`
/// untargeted (read from the default root) and let the on-disk
/// values pass through.
pub(crate) fn apply_archive_source_override(pool: &mut [SidecarResult]) {
    for sc in pool {
        sc.run_source = Some(SIDECAR_RUN_SOURCE_ARCHIVE.to_string());
    }
}

/// Resolve the kernel source-tree path for [`detect_kernel_commit`]
/// from the [`crate::KTSTR_KERNEL_ENV`] env var.
///
/// Routes through [`crate::ktstr_kernel_env`] for the raw env
/// value and [`crate::kernel_path::KernelId`] for variant
/// dispatch:
///
/// - `KernelId::Path(p)`: probes the path's `metadata.json` first
///   — `cargo-ktstr`'s `--kernel /path/to/linux` resolver routes
///   clean source trees through the cache pipeline (see
///   [`crate::cli::resolve_kernel_dir_to_entry`]) and exports the
///   CACHE ENTRY directory through `KTSTR_KERNEL`, not the
///   literal source tree. When `metadata.json` parses and carries
///   a `KernelSource::Local::source_tree_path`, that path is the
///   underlying source tree and is returned. When parsing fails
///   (the path IS the source tree, the dirty-tree path that
///   skipped the cache store), falls back to using the raw env
///   value verbatim — that path is itself the source tree.
/// - `KernelId::Version(ver)`: looks for a Local cache entry
///   whose `metadata.version == ver` carrying a
///   `source_tree_path`. The tarball-shaped key (`{ver}-tarball-
///   {arch}-kc{suffix}`) is checked first because it is the
///   most-common form a Version-shaped env points at; on miss
///   (or hit yielding `Tarball` / `Git` source, both of which
///   are transient with no on-disk tree to probe), the function
///   falls back to scanning every valid cache entry for a Local
///   match on version. Without this fallback,
///   a cache populated by `kernel build --kernel
///   /path/to/linux` (a Local entry with source_tree_path) is
///   never found by a sidecar writer that has
///   `KTSTR_KERNEL=6.14.2`, even though the local tree is
///   exactly what the kernel_commit field needs to probe.
/// - `KernelId::CacheKey(k)`: uses `k` verbatim — the cache key
///   already carries every detail (source-type prefix, arch,
///   kconfig hash). On hit, returns
///   `KernelSource::Local::source_tree_path` if set, else
///   `None` (Tarball / Git entries are transient and have no
///   persisted source tree).
/// - `KernelId::Range { .. }` / `KernelId::Git { .. }`:
///   multi-kernel specs in `KTSTR_KERNEL` never reach this
///   helper in production (find_kernel's env reader bails
///   before sidecar writing). Defensive: returns `None`.
///
/// Returns `None` when the env var is unset, when no source
/// tree path is recoverable, or when the cache lookup fails.
#[cfg(test)]
fn resolve_kernel_source_dir() -> Option<std::path::PathBuf> {
    source_dir_for(&crate::ktstr_kernel_env()?)
}

/// Resolve a `KTSTR_KERNEL` identifier string to the on-disk SOURCE
/// tree whose git HEAD is the kernel's commit (or `None` for transient
/// Range/Git specs or an unrecoverable cache lookup).
///
/// `pub` + `#[doc(hidden)]` because the cargo-ktstr `[[bin]]` is a separate
/// crate. [`kernel_commit_for_resolved`] uses this for raw source trees and
/// legacy Local cache entries after taking the cache-metadata fast path.
#[doc(hidden)]
pub fn source_dir_for(raw: &str) -> Option<std::path::PathBuf> {
    use crate::kernel_path::KernelId;
    let id = KernelId::parse(raw);
    match id {
        KernelId::Path(_) => {
            let p = std::path::Path::new(raw);
            // Cache-entry layout: `metadata.json` carries the
            // `KernelSource::Local::source_tree_path` recorded at
            // build time. Source-tree layout (dirty path that
            // skipped cache store): no metadata, so the env value
            // IS the source tree. The shared helper handles both.
            crate::cache::recover_local_source_tree(p)
                .or_else(|| Some(std::path::PathBuf::from(raw)))
        }
        KernelId::Version(_) | KernelId::CacheKey(_) => {
            let cache = crate::cache::CacheDir::new().ok()?;
            resolve_kernel_source_dir_with_cache(&id, &cache)
        }
        KernelId::Range { .. }
        | KernelId::Git { .. }
        | KernelId::Package { .. }
        | KernelId::Distro { .. } => None,
    }
}

/// Resolve the commit which built one already-resolved kernel directory.
///
/// A normal cargo-ktstr resolution returns a kernel cache-entry directory.
/// Its `metadata.json` already records the source commit captured when the
/// kernel was built, so that value is both more authoritative and much
/// cheaper than walking the current source worktree again. In particular,
/// concurrent CI lanes no longer each scan the same Linux checkout before
/// starting nextest.
///
/// Local cache metadata written before `git_hash` was populated retains its
/// historical fallback: probe the recorded `source_tree_path`. A raw source
/// path (no valid cache metadata) likewise uses [`source_dir_for`] followed by
/// [`detect_kernel_commit`]. Valid non-local metadata without a commit has no
/// on-disk source to probe and returns `None`.
///
/// `pub` + `#[doc(hidden)]` lets the cargo-ktstr binary and sidecar writer use
/// the exact same provenance decision when producing/consuming the shared
/// [`crate::KTSTR_KERNEL_COMMIT_ENV`] map.
#[doc(hidden)]
pub fn kernel_commit_for_resolved(raw: &str) -> Option<String> {
    let resolved = std::path::Path::new(raw);
    if let Ok(bytes) = std::fs::read(resolved.join("metadata.json"))
        && let Ok(metadata) = serde_json::from_slice::<crate::cache::KernelMetadata>(&bytes)
    {
        return match metadata.source {
            crate::cache::KernelSource::Local {
                git_hash: Some(hash),
                ..
            }
            | crate::cache::KernelSource::Git {
                git_hash: Some(hash),
                ..
            } => Some(hash),
            crate::cache::KernelSource::Local {
                source_tree_path: Some(source),
                git_hash: None,
            } => detect_kernel_commit(&source),
            crate::cache::KernelSource::Local { .. }
            | crate::cache::KernelSource::Git { .. }
            | crate::cache::KernelSource::Tarball
            | crate::cache::KernelSource::DistroPackage { .. }
            | crate::cache::KernelSource::LocalPackage { .. } => None,
        };
    }
    source_dir_for(raw).and_then(|source| detect_kernel_commit(&source))
}

/// Pure helper for [`source_dir_for`] that takes the
/// parsed `KernelId` and an opened `CacheDir`, returning the source
/// tree path if recoverable.
///
/// Split out from [`source_dir_for`] so tests can pin a
/// `CacheDir` at a tempdir root without mutating env vars (which
/// would race other tests reading `KTSTR_KERNEL` /
/// `KTSTR_CACHE_DIR`).
///
/// Lookup order for [`crate::kernel_path::KernelId::Version`]:
/// 1. Tarball-shaped cache key (`{ver}-tarball-{arch}-kc{suffix}`),
///    direct lookup. Returns `Some` only if the entry is a
///    `KernelSource::Local` carrying a `source_tree_path`.
/// 2. Fallback scan: every valid cache entry whose
///    `metadata.version == ver`. First match with
///    `KernelSource::Local::source_tree_path` set wins. Handles
///    the case where the user built `--kernel /path/to/linux`
///    (a Local cache entry without the tarball cache-key prefix)
///    but later set `KTSTR_KERNEL=6.14.2` for the test run —
///    without this fallback, the local source tree would be
///    invisible to the sidecar writer.
///
/// `KernelSource::Tarball` and `KernelSource::Git` entries are
/// skipped at every step because their source trees are transient
/// (deleted by the cache pipeline after build), so probing them
/// for a `kernel_commit` would always fail.
///
/// For [`crate::kernel_path::KernelId::CacheKey`], performs a single direct lookup —
/// the cache key already encodes every detail (source-type
/// prefix, arch, kconfig hash) so no fallback scan is needed.
fn resolve_kernel_source_dir_with_cache(
    id: &crate::kernel_path::KernelId,
    cache: &crate::cache::CacheDir,
) -> Option<std::path::PathBuf> {
    use crate::kernel_path::KernelId;
    match id {
        KernelId::Version(ver) => {
            let arch = std::env::consts::ARCH;
            let tarball_key = format!("{ver}-tarball-{arch}-kc{}", crate::cache_key_suffix());
            if let Some(entry) = cache.lookup(&tarball_key)
                && let crate::cache::KernelSource::Local {
                    source_tree_path: Some(p),
                    ..
                } = &entry.metadata.source
            {
                return Some(p.clone());
            }
            let entries = cache.list().ok()?;
            for listed in entries {
                let crate::cache::ListedEntry::Valid(entry) = listed else {
                    continue;
                };
                if entry.metadata.version.as_deref() != Some(ver.as_str()) {
                    continue;
                }
                if let crate::cache::KernelSource::Local {
                    source_tree_path: Some(p),
                    ..
                } = &entry.metadata.source
                {
                    return Some(p.clone());
                }
            }
            None
        }
        KernelId::CacheKey(k) => {
            let entry = cache.lookup(k)?;
            match entry.metadata.source {
                crate::cache::KernelSource::Local {
                    source_tree_path: Some(ref p),
                    ..
                } => Some(p.clone()),
                _ => None,
            }
        }
        // Path / Range / Git callers do not reach this helper —
        // resolve_kernel_source_dir handles them inline. Defensive
        // None covers any future caller that adds a new arm.
        _ => None,
    }
}

/// The kernel commit recorded in a sidecar: the env fast-path first,
/// then resolved-kernel provenance metadata or the raw-source fallback.
///
/// cargo-ktstr pre-probes every resolved kernel's HEAD once and exports
/// a `dir=commit;...` map in [`crate::KTSTR_KERNEL_COMMIT_ENV`], keyed
/// by the dir it also exports as `KTSTR_KERNEL`. This process looks
/// itself up by its own [`crate::ktstr_kernel_env`] value — string-equal
/// to the map key by construction, since cargo-ktstr built both from the
/// same resolved dir. A hit skips `detect_kernel_commit`'s gix HEAD +
/// dirty-walk, which is memoized per process but NOT across the per-test
/// nextest processes (so without the map each of N processes re-pays
/// it).
///
/// Keying on `ktstr_kernel_env()` (the raw `KTSTR_KERNEL`) is deliberate —
/// that is exactly the key cargo-ktstr used. The map's commit VALUE matches
/// this function's own fallback because both call
/// [`kernel_commit_for_resolved`].
///
/// Miss / empty commit → the shared provenance resolver. Optimization only.
fn kernel_commit_for_sidecar() -> Option<String> {
    let self_dir = crate::ktstr_kernel_env()?;
    if let Ok(raw) = std::env::var(crate::KTSTR_KERNEL_COMMIT_ENV) {
        for seg in raw.split(';') {
            if let Some((dir, commit)) = seg.rsplit_once('=')
                && dir == self_dir
                && !commit.is_empty()
            {
                return Some(commit.to_string());
            }
        }
    }
    kernel_commit_for_resolved(&self_dir)
}

/// Compute a stable 64-bit discriminator over the fields that
/// distinguish gauntlet variants of the same test. Used to suffix
/// the sidecar filename so concurrent variants do not clobber each
/// other's output.
///
/// Uses [`siphasher::sip::SipHasher13`] with zero keys for the same
/// cross-toolchain stability reason as the other zero-keyed
/// SipHasher13 sites (`build.rs`, `runtime.rs` `content_hash`) —
/// the discriminator
/// must be the same across Rust toolchain versions or downstream
/// tooling that groups variants by filename breaks.
///
/// # Host-state collision caveat
///
/// The hash is over test-identity fields (topology, scheduler,
/// payload, work_type, sysctls, kargs) — NOT over
/// [`crate::host_context::HostContext`], NOT over `scheduler_commit`, NOT over
/// `project_commit`, NOT over `kernel_commit`, NOT over
/// `run_source`, NOT over `resolve_source`, and NOT over
/// `cpu_budget` / `vcpus`. The
/// [`crate::host_context::HostContext`] exclusion is pinned by
/// `sidecar_variant_hash_excludes_host_context`; the
/// `scheduler_commit` exclusion by
/// `sidecar_variant_hash_excludes_scheduler_commit`; the
/// `project_commit` exclusion by
/// `sidecar_variant_hash_excludes_project_commit`; the
/// `kernel_commit` exclusion by
/// `sidecar_variant_hash_excludes_kernel_commit`; the
/// `run_source` exclusion by
/// `sidecar_variant_hash_excludes_run_source`; the
/// `resolve_source` exclusion by
/// `sidecar_variant_hash_excludes_resolve_source`; the
/// `cpu_budget` / `vcpus` exclusion by
/// `sidecar_variant_hash_excludes_cpu_budget`. All seven are
/// deliberate for the same cross-host grouping reason — a
/// gauntlet rebuilt against a different userspace scheduler
/// commit, a bumped ktstr checkout, a kernel source tree at a
/// different HEAD, a different CI runner / developer machine, a
/// run that resolved its scheduler via a different discovery
/// path, or a run that confined its vCPUs to a different
/// host-CPU budget must still bucket with the same-named variant so
/// `compare_partitions` can diff two runs of the "same" test
/// without the commit hash, run-source tag, or budget shattering
/// them into one-row-per-commit islands. `cpu_budget` / `vcpus`
/// are instead surfaced as the [`crate::stats::Dimension::CpuBudget`]
/// pairing axis, which separates cross-budget runs at compare time
/// rather than at the identity bucket. Callers that want to detect
/// a commit drift or compare across run environments inspect
/// [`SidecarResult::scheduler_commit`] /
/// [`SidecarResult::project_commit`] /
/// [`SidecarResult::kernel_commit`] /
/// [`SidecarResult::run_source`] /
/// [`SidecarResult::resolve_source`] directly (via
/// `--project-commit` / `--kernel-commit` / `--run-source` /
/// `--resolve-source` on `perf-delta`); the filename stays stable
/// across commits and run environments by design.
///
/// The corollary of the HostContext exclusion: if the host's
/// observable state mutates mid-suite — NUMA hotplug, hugepage
/// reconfiguration, a `sysctl -w` from a parallel process — two
/// runs of the same test will produce the same sidecar filename
/// and the later write clobbers the earlier. ktstr treats host
/// state as stable-enough for a single suite run; callers
/// mutating host state during a run own the ordering themselves
/// (e.g. by writing to a different `KTSTR_SIDECAR_DIR` per host
/// snapshot).
/// The single canonical-JSON + siphash site for the variant hash.
///
/// [`sidecar_variant_hash`] (from a written [`SidecarResult`]) and
/// [`variant_hash_from_parts`] (from a test entry + resolved topology +
/// work_type, before any sidecar exists) both route through this so the
/// two derivations can never drift. `sysctls`/`kargs` are sorted here
/// for order-independence.
fn variant_hash_of(
    topology: &str,
    scheduler: &str,
    payload: Option<&str>,
    work_type: &str,
    sysctls: &[String],
    kargs: &[String],
) -> u64 {
    use siphasher::sip::SipHasher13;
    use std::hash::Hasher;
    let mut sorted_sysctls = sysctls.to_vec();
    sorted_sysctls.sort();
    let mut sorted_kargs = kargs.to_vec();
    sorted_kargs.sort();
    let canonical = serde_json::json!({
        "topology": topology,
        "scheduler": scheduler,
        "payload": payload,
        "work_type": work_type,
        "sysctls": sorted_sysctls,
        "kargs": sorted_kargs,
    });
    let bytes = serde_json::to_vec(&canonical).expect("json serialization cannot fail for strings");
    let mut h = SipHasher13::new_with_keys(0, 0);
    h.write(&bytes);
    h.finish()
}

pub(crate) fn sidecar_variant_hash(sidecar: &SidecarResult) -> u64 {
    variant_hash_of(
        &sidecar.topology,
        &sidecar.scheduler,
        sidecar.payload.as_deref(),
        &sidecar.work_type,
        &sidecar.sysctls,
        &sidecar.kargs,
    )
}

/// The variant hash for a test entry's run at a given resolved topology
/// and `work_type`, computed BEFORE any sidecar exists — the
/// failure-dump path (and the Ctx/VmResult `variant_hash` stamp) need
/// the identity at VM-build time. Mirrors [`write_sidecar`]'s field
/// derivation (topology = the resolved topology, scheduler/sysctls/kargs
/// = [`scheduler_fingerprint`], payload = `entry.payload`) so the dump
/// filename carries the SAME variant hash the sidecar will. Pinned
/// equal to [`sidecar_variant_hash`] by a roundtrip test.
pub(crate) fn variant_hash_from_parts(
    entry: &KtstrTestEntry,
    resolved_topology: &crate::vmm::topology::Topology,
    work_type: &str,
) -> u64 {
    let fp = scheduler_fingerprint(entry);
    variant_hash_of(
        &resolved_topology.to_string(),
        &fp.scheduler,
        entry.payload.map(|p| p.name),
        work_type,
        &fp.sysctls,
        &fp.kargs,
    )
}

/// Entry-derived scheduler metadata that every sidecar carries
/// regardless of pass/fail/skip.
///
/// Both write paths ([`write_sidecar`] and [`write_skip_sidecar`])
/// thread the same materialized fields through to their
/// `SidecarResult` constructors; keeping the derivation in a
/// named struct (rather than a 4-tuple) means a new
/// scheduler-level field shows up as a named field at both
/// writer sites and in every call-site binding, instead of as
/// an additional anonymous tuple slot that readers have to
/// remember the ordering of.
///
/// `pub(crate)` rather than `pub`: the intermediate struct is a
/// write-path detail, not a public API surface. No serde — this
/// is not a persisted shape, just a grouped return value.
///
/// Derives `Debug` for `assert_eq!` diagnostics, `Clone` so tests
/// can materialize a fixture once and reuse it across assertions,
/// and `PartialEq`/`Eq` so tests can compare whole fingerprints
/// in one statement rather than destructuring and asserting on
/// each field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SchedulerFingerprint {
    /// Pretty scheduler name (matches `SidecarResult::scheduler`),
    /// e.g. `"eevdf"` or a scheduler-kind payload's declared name.
    pub(crate) scheduler: String,
    /// Best-effort userspace scheduler commit; `None` for every
    /// current variant per
    /// [`crate::test_support::SchedulerSpec::scheduler_commit`].
    pub(crate) scheduler_commit: Option<String>,
    /// Formatted `sysctl.<key>=<value>` lines derived from the
    /// scheduler's declared `sysctls()`.
    pub(crate) sysctls: Vec<String>,
    /// Kernel command-line args declared by the scheduler,
    /// forwarded verbatim.
    pub(crate) kargs: Vec<String>,
}

/// Materialize the [`SchedulerFingerprint`] for a test entry.
///
/// A change to the sidecar schema (e.g. a new scheduler-level
/// field) extends this function + [`SchedulerFingerprint`] in
/// one place and every writer picks it up automatically.
fn scheduler_fingerprint(entry: &KtstrTestEntry) -> SchedulerFingerprint {
    let scheduler = entry.scheduler.name.to_string();
    // `SchedulerSpec::scheduler_commit()` returns `None` for every
    // variant (Eevdf, Discover, Path, KernelBuiltin) — the commit
    // string is not carried in the static spec; it comes from the
    // sidecar's run-time git probe instead. This call is here only
    // to surface the slot in the fingerprint so a future spec
    // variant carrying a commit would flow through automatically.
    let scheduler_commit = entry
        .scheduler
        .binary
        .scheduler_commit()
        .map(|s| s.to_string());
    let sysctls: Vec<String> = entry
        .scheduler
        .sysctls
        .iter()
        .map(|s| format!("sysctl.{}={}", s.key(), s.value()))
        .collect();
    let kargs: Vec<String> = entry
        .scheduler
        .kargs
        .iter()
        .map(|s| s.to_string())
        .collect();
    SchedulerFingerprint {
        scheduler,
        scheduler_commit,
        sysctls,
        kargs,
    }
}

/// Compute the per-variant sidecar path and serialize + write the
/// result to disk.
///
/// Gauntlet variants of the same test differ by work_type, flags
/// (via scheduler args → sysctls/kargs), scheduler, and topology. A
/// filename of just `{test_name}.ktstr.json` causes variants to
/// overwrite each other, erasing all but the last-written result.
/// `sidecar_variant_hash` hashes the discriminating fields into a
/// short stable suffix so each variant gets its own sidecar file.
///
/// On an orchestrated run-epoch mismatch,
/// [`acquire_run_dir_publication_lock`] removes any pre-existing
/// `*.ktstr.json` files before publishing the new sentinel so the
/// run is a clean snapshot rather than a mosaic of prior invocations.
/// Matching-epoch writes append concurrently.
///
/// Pre-clear is SKIPPED when `KTSTR_SIDECAR_DIR` is set: the
/// operator chose that directory and owns its contents — silent
/// data loss is not acceptable on an explicit override. When the
/// override is unset (the default-path branch),
/// `std::fs::create_dir_all` materializes the directory BEFORE
/// pre-clear runs so the helper's canonicalize step always sees
/// an existing on-disk path; without this ordering, a missing
/// dir on the very first call would key the cache against the
/// raw path while a later call (after the dir exists) would key
/// against the canonicalized absolute path, splitting the cache
/// and causing the second call to re-fire pre-clear and wipe the
/// first call's sidecars.
///
/// CROSS-PROCESS PUBLICATION: on the default path (override unset),
/// writers with a matching [`crate::KTSTR_RUN_EPOCH_ENV`] sentinel
/// acquire advisory `LOCK_SH` on
/// `{runs_root}/.locks/{key}.lock`, recheck the token under that
/// lock, and retain SH through their atomic rename. Matching writers
/// therefore publish concurrently. A first writer or epoch change
/// acquires `LOCK_EX`, rechecks, then performs wipe -> sentinel ->
/// first publication. Since EX conflicts with every matching
/// writer's SH, an epoch reset cannot cross the validated-token ->
/// rename interval. The override path skips the lock for the same
/// reason it skips pre-clear: operator-chosen directories are owned
/// by the operator, so we do not place a `.locks/` sibling inside
/// (or above) their custom layout.
///
/// A later peer whose sentinel already matches bypasses pre-clear
/// and takes only the shared publication rail. Without that sentinel
/// a later peer's pre-clear would delete an earlier peer's freshly
/// written sidecar — silent stats loss. Raw `cargo nextest run` sets
/// no token, so its peers retain the serialized wipe-everything
/// behavior.
///
/// PER-FILE ATOMICITY (both branches): the JSON is written to a
/// `<final>.tmp.<pid>.<run_id>` sibling and then `rename(2)`'d into
/// place. POSIX `rename` is atomic for same-directory destinations,
/// so a peer reader (`collect_sidecars`) never observes a partial
/// JSON payload — either the old contents stay or the new contents
/// replace them in one filesystem step. Two concurrent writers that
/// both target the same `{test_name}-{variant_hash}.ktstr.json`
/// (override path: two CI jobs sharing one operator-chosen dir;
/// default path: a torn-write window inside the flock body that the
/// flock would otherwise have to cover) cannot leave a half-written
/// JSON behind — last-rename-wins, both files are individually
/// well-formed. The `.tmp.<pid>.<run_id>` discriminator on the
/// staging name keeps two writers from racing on the same staging
/// path even when their final destinations collide. The flock on
/// the default path remains load-bearing for the pre-clear leg
/// (atomic write only protects the write itself, not the
/// `read_dir + remove_file` walk that pre-clear runs).
///
/// `label` is a caller-supplied noun for the context message ("skip
/// sidecar" / "sidecar") so the error chain points at the right call
/// site.
fn serialize_and_write_sidecar(sidecar: &SidecarResult, label: &str) -> anyhow::Result<()> {
    // Read the override ONCE. The two branches below carry the
    // result through structurally so neither leg re-reads
    // `KTSTR_SIDECAR_DIR` — preventing the override from flipping
    // mid-call (which would otherwise let an external mutation
    // between the dir resolve and the pre-clear gate either skip
    // the wipe on a default-path dir or fire a wipe on an
    // operator-chosen one).
    let (dir, do_pre_clear) = match sidecar_dir_override() {
        Some(path) => (path, false),
        None => (resolve_default_sidecar_dir(), true),
    };
    // Materialize the directory FIRST so `pre_clear_run_dir_once`
    // can canonicalize a path that exists on disk. Without this,
    // the very first invocation in a process resolves the cache
    // key against the raw relative path (canonicalize fails on a
    // missing dir, falls back to raw); subsequent invocations
    // resolve against the canonicalized absolute path because the
    // dir now exists. Two distinct keys for the same logical dir
    // → second invocation re-fires pre-clear and wipes the first
    // invocation's sidecars. Materializing pre-pre-clear closes
    // the relative-vs-absolute split.
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("create sidecar dir {}", dir.display()))?;
    let variant_hash = sidecar_variant_hash(sidecar);
    let path = dir.join(format!(
        "{}-{:016x}.ktstr.json",
        sidecar.test_name, variant_hash
    ));
    let json = serde_json::to_string_pretty(sidecar)
        .with_context(|| format!("serialize {label} for '{}'", sidecar.test_name))?;
    // Atomic write: stage into a `.tmp.<pid>.<run_id>` sibling and
    // rename(2) into the final path. `rename` is atomic for
    // same-directory destinations on every filesystem ktstr supports
    // (ext4, btrfs, xfs, tmpfs, overlayfs); a peer reader never
    // observes a partial payload. The staging name carries the pid
    // AND the unique sidecar `run_id` so two writers in the same
    // process targeting identical final paths (e.g. two threads in
    // the budget-test stdout-capture path) cannot stomp each other's
    // staging file before either rename lands. On rename failure the
    // staging file is removed so a partial sidecar does not survive
    // as garbage in the run dir; rename success consumes the staging
    // entry and there is nothing to clean up.
    let pid = std::process::id();
    let staging = dir.join(format!(
        "{}-{:016x}.ktstr.json.tmp.{pid}.{}",
        sidecar.test_name, variant_hash, sidecar.run_id,
    ));

    // Serialize before admission to keep both the shared publication
    // interval and the exclusive epoch-reset interval limited to filesystem
    // mutation. On the orchestrated path, a matching epoch takes LOCK_SH:
    // all peer writers publish concurrently, while an epoch reset's LOCK_EX
    // cannot cross the validated-token -> rename interval. A missing or
    // mismatched epoch takes LOCK_EX, rechecks under that lock, then performs
    // wipe -> sentinel before publishing. Raw no-token runs retain the
    // historical exclusive, pre-clear-once behavior. The override branch
    // skips both coordination and pre-clear.
    let session_token = if do_pre_clear {
        run_session_token()
    } else {
        None
    };
    let _publication_lock = if do_pre_clear {
        Some(acquire_run_dir_publication_lock(
            &dir,
            session_token.as_deref(),
        )?)
    } else {
        None
    };
    if let Err(error) = std::fs::write(&staging, &json) {
        // `write` may have created or partially filled the staging inode
        // before reporting an error. Never leave that residue for the rest
        // of the epoch; the next reset is not guaranteed to happen soon.
        let _ = std::fs::remove_file(&staging);
        return Err(anyhow::Error::from(error)
            .context(format!("write {label} staging {}", staging.display())));
    }
    if let Err(e) = std::fs::rename(&staging, &path) {
        // Best-effort cleanup of the staged payload; ignore the
        // unlink error so the rename failure is what surfaces
        // (the rename error names the actual problem).
        let _ = std::fs::remove_file(&staging);
        return Err(anyhow::Error::from(e).context(format!(
            "rename {label} staging {} -> {}",
            staging.display(),
            path.display(),
        )));
    }
    LAST_SIDECAR_PATH.with(|p| *p.borrow_mut() = Some(path.clone()));
    Ok(())
}

thread_local! {
    /// Absolute path of the most recent sidecar this thread wrote (via
    /// [`serialize_and_write_sidecar`]). The dispatch run loop
    /// ([`crate::test_support::eval::run_ktstr_test_inner`]) reads and
    /// clears it after the run to finalize the persisted verdict to the
    /// test's FINAL (post-inversion) outcome. nextest is process-per-test
    /// so a run writes one sidecar; a value left from an earlier phase is
    /// overwritten by the current write, so the take always yields this
    /// run's sidecar.
    static LAST_SIDECAR_PATH: std::cell::RefCell<Option<PathBuf>> =
        const { std::cell::RefCell::new(None) };
}

/// Take (read + clear) the path of the sidecar most recently written on
/// this thread, or `None` when no sidecar was written this run (an
/// early bail before any write). See [`LAST_SIDECAR_PATH`].
///
/// MUST be drained exactly once per run — `run_ktstr_test_inner` does
/// this after each dispatch. The thread-local persists across calls in
/// a process, so a caller that writes a sidecar WITHOUT a following take
/// would leave a stale path for the next take to consume; in practice
/// only `run_ktstr_test_inner` pairs a write with a take, and a stale
/// path points at a dropped tempdir so the finalize read fails benignly.
pub(crate) fn take_last_sidecar_path() -> Option<PathBuf> {
    LAST_SIDECAR_PATH.with(|p| p.borrow_mut().take())
}

/// Overwrite a written sidecar's verdict bits with the test's FINAL
/// (post-inversion) `(passed, skipped, inconclusive)` outcome — see
/// [`crate::test_support::dispatch::Verdict::sidecar_bits`] — and set
/// [`SidecarResult::expected_failure`] when an actual scenario
/// failure/inconclusive was inverted to a pass/skip. Rewrites the file
/// atomically (temp + rename) under the same publication rail as the
/// primary write. Matching orchestrated epochs take SH; a stale token
/// returns without touching the path, raw no-token runs take EX, and
/// explicit sidecar-directory overrides remain unlocked.
///
/// A no-op when the final verdict already matches what was persisted (an
/// ordinary pass/fail/skip — no `expect_err`/`expect_auto_repro`
/// inversion). Best-effort: a read/parse/serialize/write error is
/// surfaced on stderr and swallowed so the raw sidecar stands (the
/// footer then falls back to it) rather than failing the run.
pub(crate) fn finalize_sidecar_verdict(
    path: &std::path::Path,
    passed: bool,
    skipped: bool,
    inconclusive: bool,
) {
    let coordinated = sidecar_dir_override().is_none();
    let session_token = if coordinated {
        run_session_token()
    } else {
        None
    };
    finalize_sidecar_verdict_inner(
        path,
        passed,
        skipped,
        inconclusive,
        coordinated,
        session_token.as_deref(),
        || {},
    );
}

fn finalize_sidecar_verdict_inner<F>(
    path: &std::path::Path,
    passed: bool,
    skipped: bool,
    inconclusive: bool,
    coordinated: bool,
    session_token: Option<&str>,
    before_rename: F,
) where
    F: FnOnce(),
{
    // Finalization is a second publication of the same sidecar. Keep it on
    // the same epoch rail as the primary write: an orchestrated finalizer
    // takes SH and rechecks its exact token, a raw no-token finalizer takes
    // EX, and an explicit override remains unlocked. A token mismatch means
    // another invocation reset the directory after this run's primary write;
    // returning before even reading the old path prevents a stale finalizer
    // from recreating a file on the far side of that reset.
    let _publication_lock = if coordinated {
        let Some(dir) = path.parent() else {
            eprintln!(
                "ktstr: finalize_sidecar_verdict: sidecar has no parent {}",
                path.display()
            );
            return;
        };
        match acquire_run_dir_finalize_lock(dir, session_token) {
            Ok(Some(lock)) => Some(lock),
            Ok(None) => return,
            Err(error) => {
                eprintln!(
                    "ktstr: finalize_sidecar_verdict: admission for {} failed: {error:#}",
                    path.display()
                );
                return;
            }
        }
    } else {
        None
    };

    let Ok(json) = std::fs::read_to_string(path) else {
        return;
    };
    let Ok(mut sc) = serde_json::from_str::<SidecarResult>(&json) else {
        eprintln!(
            "ktstr: finalize_sidecar_verdict: unparseable sidecar {}",
            path.display()
        );
        return;
    };
    // The run's telemetry is failure-mode-dominated when its scenario
    // actually failed/was-inconclusive but the final verdict is a
    // pass/skip (an inversion) — `perf-delta` excludes such rows.
    let raw_failed = sc.is_fail() || sc.is_inconclusive();
    let expected_failure = raw_failed && (passed || skipped);
    if sc.passed == passed
        && sc.skipped == skipped
        && sc.inconclusive == inconclusive
        && sc.expected_failure == expected_failure
    {
        return;
    }
    sc.passed = passed;
    sc.skipped = skipped;
    sc.inconclusive = inconclusive;
    sc.expected_failure = expected_failure;
    let Ok(out) = serde_json::to_string_pretty(&sc) else {
        return;
    };
    // Stage with a `.ktstr.json.tmp.…` suffix (append, NOT
    // `with_extension`, which would drop `.json`) so a hard-crash orphan
    // — write succeeded but rename did not — is reaped by
    // the epoch-reset sweep via `is_sidecar_staging_filename`, the
    // same way the primary write's staging file is.
    let pid = std::process::id();
    let mut staging = path.as_os_str().to_owned();
    staging.push(format!(".tmp.finalize.{pid}"));
    let staging = std::path::PathBuf::from(staging);
    if std::fs::write(&staging, &out).is_err() {
        let _ = std::fs::remove_file(&staging);
        return;
    }
    before_rename();
    if std::fs::rename(&staging, path).is_err() {
        let _ = std::fs::remove_file(&staging);
    }
}

/// Remove every failure-dump artifact for this exact test variant,
/// including nextest retry archives.
///
/// Called when a run's FINAL outcome is a pass/skip but it wrote NO
/// sidecar — the run crashed before the guest produced a parseable
/// result (e.g. an `expect_err` test with a host-triggered BPF crash),
/// so [`finalize_sidecar_verdict`] had nothing to finalize. The freeze
/// coordinator wrote the dump unconditionally; without a sidecar to mark
/// the pass, the footer's dump-only trigger
/// ([`summarize_one_run_dir`] flags a dump with no parsed sidecar) would
/// surface this PASSING test as FAILED. Removing the dump keeps the
/// footer consistent with nextest's pass. Best-effort: a missing dump
/// (the normal clean-pass case) is fine. A genuine pre-sidecar failure
/// (final = Fail) does NOT call this, so its dump still flags.
pub(crate) fn suppress_failure_dumps(test_name: &str, variant_hash: u64) {
    let dir = sidecar_dir();
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let Some((candidate, hash, kind)) = classify_run_artifact(name) else {
            continue;
        };
        if candidate == test_name
            && hash == variant_hash
            && matches!(
                kind,
                RunArtifactKind::FailureDump | RunArtifactKind::ReproFailureDump
            )
        {
            let _ = std::fs::remove_file(path);
        }
    }
}

/// How many periodic-snapshot files a FAILING attempt keeps per dump
/// base (primary and `.repro.` are bounded independently).
///
/// The freeze coordinator writes one `{base}.snapshot.periodic_NNN.json`
/// per fired boundary (up to [`crate::scenario::snapshot::MAX_STORED_SNAPSHOTS`] = 64),
/// each multi-megabyte. NOTHING on disk reads them back: `eval` and the
/// cast-analysis pipeline both consume the in-process
/// [`crate::scenario::snapshot::SnapshotBridge`] (which FIFO-caps at 64
/// on its own), never these files — they exist solely as CI forensic
/// artifacts for human inspection. The documented floor for *meaningful*
/// periodic coverage is `num_snapshots >= 2` (see
/// `crate::assert::temporal` / `crate::stats::metric_id`); keeping the
/// most-recent 8 leaves a 4x margin of trajectory approaching the
/// failure while bounding a 64-deep pile to 8 files. The most recent
/// samples (highest `periodic_NNN` index) are retained because they
/// bracket the failure; the periodic tag is zero-padded 3-digit, so
/// lexical index order is chronological.
pub(crate) const RETAINED_PERIODIC_SNAPSHOTS: usize = 8;

/// Filename prefix shared by every on-disk snapshot for one test
/// variant: `{test_name}-{variant_hash:016x}`. The freeze coordinator
/// derives snapshot names from the failure-dump base by stripping
/// `.failure-dump` (see `vmm::freeze_coord::snapshot::snapshot_tagged_path`),
/// so the primary base yields `{prefix}.snapshot.{tag}.json` and the
/// repro base yields `{prefix}.repro.snapshot.{tag}.json`. The trailing
/// `-{hash:016x}` makes the prefix an exact variant boundary — `foo`'s
/// prefix cannot match `foobar`'s files, and a sibling gauntlet preset
/// (different hash) is untouched.
fn snapshot_base_prefix(test_name: &str, variant_hash: u64) -> String {
    format!("{test_name}-{variant_hash:016x}")
}

/// `true` when `name` is any snapshot file (`.snapshot.{tag}.json`,
/// primary or `.repro.` base) for the variant identified by `prefix`.
fn is_snapshot_file_for(name: &str, prefix: &str) -> bool {
    let Some(rest) = name.strip_prefix(prefix) else {
        return false;
    };
    let rest = rest.strip_prefix(".repro").unwrap_or(rest);
    rest.starts_with(".snapshot.") && rest.ends_with(".json")
}

/// `Some(index)` when `name` is a PERIODIC snapshot file for `prefix`,
/// returning the zero-padded 3-digit `periodic_NNN` index for recency
/// ordering. `None` for a non-matching file or a non-periodic snapshot
/// tag (e.g. `.snapshot.mid_run.json`, `.snapshot.early-degraded.json`),
/// which are single-shot and never bounded. The 3-digit / all-ASCII-digit
/// gate matches the coordinator's `format!("periodic_{:03}", idx)`
/// emission exactly, so a user-chosen `Op::CaptureSnapshot { name:
/// "periodic_kaslr" }` tag cannot be mistaken for a periodic sample.
fn periodic_snapshot_index_for(name: &str, prefix: &str) -> Option<u32> {
    let rest = name.strip_prefix(prefix)?;
    let rest = rest.strip_prefix(".repro").unwrap_or(rest);
    let tag = rest.strip_prefix(".snapshot.")?.strip_suffix(".json")?;
    let digits = tag.strip_prefix("periodic_")?;
    if digits.len() == 3 && digits.bytes().all(|b| b.is_ascii_digit()) {
        digits.parse().ok()
    } else {
        None
    }
}

/// Remove EVERY on-disk snapshot file for this exact test variant
/// (primary + `.repro.` bases, periodic and one-shot tags alike).
///
/// Two callers, both best-effort (a missing dir or file is fine, and
/// nothing here ever fails a test):
/// - **Pass/skip finalize** ([`crate::test_support::eval`]): a run whose
///   FINAL verdict is pass/skip needs no periodic forensics, so the whole
///   pile is dropped before the process exits (and thus before CI uploads
///   artifacts). This is the dominant reclaim — most e2e tests pass, and
///   each was writing dozens of multi-MB snapshots that nothing consumed.
/// - **Attempt-prepare reap**: called before boot so a new attempt starts
///   from a clean snapshot slate. A prior attempt that fired MORE
///   boundaries — or was SIGKILL'd by the watchdog mid-run — leaves stale
///   higher-index snapshots the coordinator's tag-keyed overwrite would
///   not reclaim; this reaps them. It is the crash-safety mechanism: a
///   killed process cannot clean up after itself, so the NEXT attempt's
///   prepare does it (mirrors [`super::eval`]'s `prepare_failure_dump_path`).
pub(crate) fn cleanup_snapshots(test_name: &str, variant_hash: u64) {
    let dir = sidecar_dir();
    let prefix = snapshot_base_prefix(test_name, variant_hash);
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if is_snapshot_file_for(name, &prefix) {
            let _ = std::fs::remove_file(&path);
        }
    }
}

/// Bound the periodic snapshots for a FAILING attempt to the most recent
/// `keep` per dump base, deleting the older ones. Primary and `.repro.`
/// bases are bounded independently (each keeps its own `keep`). One-shot
/// snapshot tags (`mid_run`, degraded) are never touched — they are
/// single files, not an unbounded periodic series. Best-effort; never
/// fails a test.
///
/// The retained failure DUMP is the primary evidence and is untouched
/// here; these periodic samples are supplementary trajectory, so the
/// most-recent `keep` (bracketing the failure) suffice. See
/// [`RETAINED_PERIODIC_SNAPSHOTS`] for the floor rationale.
pub(crate) fn bound_periodic_snapshots(test_name: &str, variant_hash: u64, keep: usize) {
    let dir = sidecar_dir();
    let prefix = snapshot_base_prefix(test_name, variant_hash);
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return;
    };
    // Bucket by base so the repro series does not evict the primary's.
    let mut primary: Vec<(u32, PathBuf)> = Vec::new();
    let mut repro: Vec<(u32, PathBuf)> = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let Some(idx) = periodic_snapshot_index_for(name, &prefix) else {
            continue;
        };
        let is_repro = name
            .strip_prefix(prefix.as_str())
            .is_some_and(|rest| rest.starts_with(".repro."));
        if is_repro {
            repro.push((idx, path));
        } else {
            primary.push((idx, path));
        }
    }
    for mut set in [primary, repro] {
        if set.len() <= keep {
            continue;
        }
        // Ascending by index == chronological; drop the oldest overflow,
        // keep the newest `keep`.
        set.sort_by_key(|(idx, _)| *idx);
        let drop_count = set.len() - keep;
        for (_, path) in set.into_iter().take(drop_count) {
            let _ = std::fs::remove_file(&path);
        }
    }
}

/// `Some(path)` when `KTSTR_SIDECAR_DIR` is set non-empty,
/// returning the override path verbatim; `None` when the env
/// var is unset or empty (default-path branch). Single source
/// of truth for the override read so [`sidecar_dir`] and
/// [`serialize_and_write_sidecar`] (which gates pre-clear on
/// the override's presence) share one env-read site rather
/// than each calling `std::env::var` independently.
///
/// The `is_empty()` filter is deliberate: a defensively-cleared
/// `KTSTR_SIDECAR_DIR=""` must NOT be treated as an override
/// (joining an empty path onto the run-root would silently
/// alias the runs-root itself, contaminating the listing).
/// Empty-string aliases unset, matching the
/// `if let Ok(d) ... && !d.is_empty()` predicate the function
/// replaced.
///
/// `serialize_and_write_sidecar` interprets `Some(_)` as the
/// "operator chose this dir, do not pre-clear" gate — silent
/// data loss is unacceptable on an explicit override (the
/// override is for users who want exact control over where
/// sidecars land: test isolation, archival capture, custom CI
/// layouts).
fn sidecar_dir_override() -> Option<PathBuf> {
    std::env::var(crate::KTSTR_SIDECAR_DIR_ENV)
        .ok()
        .filter(|d| !d.is_empty())
        .map(PathBuf::from)
}

/// Emit a one-shot stderr warning when [`detect_project_commit`]
/// resolves to `None` and the run directory therefore lands at
/// `{kernel}-unknown`. Operators in this state lose the
/// `{project_commit}` discriminator on the run-directory name —
/// every non-git invocation at the same kernel collides on a
/// single directory, with the latest run pre-clearing the
/// previous one's sidecars. The warning surfaces this loss-of-isolation
/// risk so the operator can either set `KTSTR_SIDECAR_DIR` to
/// disambiguate per-run, or place the project tree under git
/// so each run carries its own commit hash.
///
/// `OnceLock<()>` gates the warning to fire EXACTLY ONCE per
/// process: every gauntlet variant resolves a sidecar directory
/// independently (via [`sidecar_dir`] and
/// [`serialize_and_write_sidecar`]), so without the gate the
/// operator would see thousands of duplicate warnings interleaved
/// with test output. Called via [`resolve_default_sidecar_dir`] —
/// which is the shared default-path body that both [`sidecar_dir`]
/// and [`serialize_and_write_sidecar`] funnel through — so the
/// warning fires only on the default-path branch. The override
/// branch in either caller returns before
/// [`resolve_default_sidecar_dir`] is reached, so an operator who
/// set `KTSTR_SIDECAR_DIR` to disambiguate non-git runs does not
/// see a misleading "commit unknown" warning that does not apply
/// to their effective directory layout.
///
/// Implementation is split into a public-facing wrapper
/// (this function) that owns the process-global `OnceLock` and
/// targets stderr, and a pure inner helper
/// [`warn_unknown_project_commit_inner`] that takes the
/// `&OnceLock<()>` gate and the `&mut dyn Write` sink as
/// parameters. The split lets tests drive the warning logic
/// against a local `OnceLock` and a `Vec<u8>` sink without
/// fighting the process-global gate or the global stderr fd —
/// the wrapper's behavior is what the inner does, just with
/// the static gate and stderr supplied.
fn warn_unknown_project_commit_once() {
    static WARNED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    let mut sink = std::io::stderr();
    warn_unknown_project_commit_inner(&WARNED, &mut sink);
}

/// Pure helper for [`warn_unknown_project_commit_once`]: gate the
/// warning on `gate` and write the warning text to `sink` exactly
/// once across the gate's lifetime. Both parameters are taken by
/// reference so call sites supply ownership semantics that match
/// their gating story:
/// - The production wrapper passes a `'static` `OnceLock<()>` so
///   the gate spans the whole process and a stderr handle so the
///   warning lands in the operator's terminal.
/// - Tests pass a local `OnceLock<()>` so each test gets a fresh
///   gate (no cross-test contamination via a process-global)
///   and a `Vec<u8>` sink so the test can read back the emitted
///   bytes and assert on the warning text.
///
/// Errors from `writeln!` are ignored via `let _ =`: a metadata
/// probe warning must not gate sidecar writes. This DEPARTS from
/// the previous `eprintln!` semantics (which panic on stderr
/// write failure per the std docs) — here we drop the write
/// error silently because a metadata probe warning must not gate
/// sidecar writes.
fn warn_unknown_project_commit_inner(
    gate: &std::sync::OnceLock<()>,
    sink: &mut dyn std::io::Write,
) {
    gate.get_or_init(|| {
        let _ = writeln!(
            sink,
            "ktstr: WARNING: project commit unavailable (cwd not in a git \
             repo, or HEAD unreadable); runs at this kernel overwrite \
             each other in target/ktstr/{{kernel}}-unknown/. Set \
             KTSTR_SIDECAR_DIR=<unique-path> per run, or run from inside a \
             git repo with at least one commit."
        );
    });
}

/// Direct-test wrapper for the process-local pre-clear behavior.
///
/// Production orchestrated writes initialize/reset epochs through
/// [`acquire_run_dir_publication_lock`]. Raw no-token writes use the
/// token-stable inner helper below while holding EX.
#[cfg(test)]
fn pre_clear_run_dir_once(dir: &std::path::Path) {
    let session_token = run_session_token();
    pre_clear_run_dir_once_for_session(dir, session_token.as_deref());
}

/// Token-stable raw/pre-clear implementation.
///
/// The serializer captures the epoch once so an unrelated environment
/// mutation cannot change identity between admission and sentinel publication.
/// Direct tests use the wrapper above, which snapshots the environment at
/// entry.
fn pre_clear_run_dir_once_for_session(dir: &std::path::Path, session_token: Option<&str>) {
    use std::collections::HashSet;
    use std::path::PathBuf;
    use std::sync::{Mutex, OnceLock};
    static PRE_CLEARED: OnceLock<Mutex<HashSet<PathBuf>>> = OnceLock::new();
    // Canonicalize so two spellings of the same on-disk dir share
    // one cache entry. Falls back to the raw path when canonicalize
    // fails (the directory may not exist yet on the very first
    // write, in which case the raw path keys the entry; subsequent
    // calls with the same raw path also miss canonicalize the
    // same way and share the entry).
    let cache_key = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());
    let cache = PRE_CLEARED.get_or_init(|| Mutex::new(HashSet::new()));
    let mut guard = cache.lock_unpoisoned();
    if guard.contains(&cache_key) {
        return;
    }
    // First time this directory has been seen — wipe sidecars while
    // the cache mutex is still held. Releasing the guard before the
    // read_dir walk would open a TOCTOU window: a sibling thread that
    // observes the now-cached entry would skip its own pre-clear,
    // proceed to write a sidecar, and the original thread's walk
    // (running after the drop) would then delete that sibling's
    // freshly-written file. The walk is one read_dir + a bounded
    // number of `*.ktstr.json` removals, so holding the lock across
    // it is brief; concurrent calls against DIFFERENT directories
    // serialize through this critical section but each does a small,
    // bounded amount of I/O, which is acceptable for a metadata
    // probe call pattern. The cache insert happens AFTER the wipe
    // completes (rather than before) so a panic mid-wipe does not
    // poison the cache with an entry whose wipe never actually ran.
    // The mutex itself enforces serialization across threads; the
    // entry only records "wipe completed for this dir" and must
    // never be observed without the wipe having succeeded. `guard`
    // is dropped at end-of-scope so the lock release happens after
    // the loop completes.
    if let Some(token) = session_token
        && session_sentinel_matches(dir, token)
    {
        // A peer test process in THIS session already cleared the dir
        // (the sentinel records the session token under the flock);
        // its and the other peers' current-session sidecars must
        // survive, so skip the wipe entirely. See CONCURRENT WRITERS.
        guard.insert(cache_key);
        return;
    }
    // Raw no-token pre-clear is deliberately best-effort. Token-bearing
    // orchestrated initialization does not use this once-gated wrapper; its
    // publication caller propagates reset and sentinel failures.
    let _ = reset_run_dir_for_session(dir, session_token);
    // Record the raw/helper attempt only after it returns. Its historical
    // best-effort contract deliberately treats filesystem errors as a
    // completed attempt; the orchestrated token-bearing path bypasses this
    // process-local cache and propagates those errors instead.
    guard.insert(cache_key);
    drop(guard);
}

/// Wipe one run directory and, when present, publish its exact epoch token.
///
/// Production callers hold the run-dir exclusive flock. Unlike
/// the process-local pre-clear helper, this deliberately has no
/// process-local once gate: an observed token mismatch is authoritative even
/// when this process wrote to the same directory earlier.
fn reset_run_dir_for_session(
    dir: &std::path::Path,
    session_token: Option<&str>,
) -> anyhow::Result<()> {
    reset_run_dir_for_session_with_remover(dir, session_token, |path| std::fs::remove_file(path))
}

/// Testable implementation of [`reset_run_dir_for_session`].
///
/// A token-bearing caller uses the sentinel as an authoritative statement
/// that the old epoch was completely removed. Every enumeration, metadata,
/// and unlink failure must therefore abort before sentinel publication. Raw
/// no-token callers preserve their historical best-effort behavior by
/// explicitly discarding this result in [`pre_clear_run_dir_once_for_session`].
fn reset_run_dir_for_session_with_remover<F>(
    dir: &std::path::Path,
    session_token: Option<&str>,
    mut remove_file: F,
) -> anyhow::Result<()>
where
    F: FnMut(&std::path::Path) -> std::io::Result<()>,
{
    let Some(token) = session_token else {
        // Preserve the raw `cargo nextest run` contract exactly: unreadable
        // directories/entries, metadata failures, and individual unlink
        // failures are ignored while the sweep continues wherever possible.
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file()
                    && (is_sidecar_filename(&path) || is_sidecar_staging_filename(&path))
                {
                    let _ = remove_file(&path);
                }
            }
        }
        return Ok(());
    };

    let entries = std::fs::read_dir(dir)
        .with_context(|| format!("enumerate run directory for epoch reset {}", dir.display()))?;
    for entry in entries {
        let entry =
            entry.with_context(|| format!("enumerate run directory entry {}", dir.display()))?;
        let path = entry.path();
        let metadata = std::fs::metadata(&path)
            .with_context(|| format!("inspect run directory entry {}", path.display()))?;
        if !metadata.is_file() {
            continue;
        }
        // Two file shapes are reaped here (current-session peers
        // were already spared by the sentinel skip above, so a
        // file reaching this point is prior-session or orphaned
        // residue):
        // - `<test>-<hash>.ktstr.json` — sidecars from a PRIOR
        //   session sharing this `{kernel}-{project_commit}` key.
        // - `<test>-<hash>.ktstr.json.tmp.<pid>.<run_id>` —
        //   orphaned staging from a writer that died between
        //   `write` and `rename` in `serialize_and_write_sidecar`
        //   (`is_sidecar_filename` excludes these — the extension
        //   is `<run_id>`, not `json` — so the staging sweep is
        //   what reaps them). The flock makes reaping an in-flight
        //   stage impossible: a live peer holds the lock we hold.
        if is_sidecar_filename(&path) || is_sidecar_staging_filename(&path) {
            remove_file(&path)
                .with_context(|| format!("remove prior-epoch artifact {}", path.display()))?;
        }
    }
    // Record this session's token so peer processes skip re-wiping.
    // Token-bearing orchestrated initialization treats this write as
    // authoritative: publishing a sidecar without its epoch would collapse
    // the herd back into repeated EX resets. Written AFTER the wipe so a
    // crash mid-wipe leaves no stale sentinel falsely claiming completion.
    let sentinel = dir.join(SESSION_SENTINEL);
    std::fs::write(&sentinel, token)
        .with_context(|| format!("publish run epoch sentinel {}", sentinel.display()))?;
    Ok(())
}

/// Whether the on-disk sentinel records `token` byte-for-byte.
fn session_sentinel_matches(dir: &std::path::Path, token: &str) -> bool {
    std::fs::read_to_string(dir.join(SESSION_SENTINEL)).is_ok_and(|recorded| recorded == token)
}

/// Filename of the per-run-directory session sentinel that records
/// the [`crate::KTSTR_RUN_EPOCH_ENV`] token of the session that last
/// cleared the dir. A dotfile so every sidecar reader ignores it
/// (`is_sidecar_filename` requires a `.json` extension and
/// `classify_run_artifact` matches none of its suffixes), and it
/// lives in the run dir itself (which the caller already
/// `create_dir_all`'d) rather than the `.locks/` sibling.
const SESSION_SENTINEL: &str = ".ktstr_run_epoch";

/// Read the `cargo ktstr test` session token from
/// [`crate::KTSTR_RUN_EPOCH_ENV`] — an opaque per-invocation value
/// the orchestrator stamps once before nextest spawns, inherited by
/// every child test process.
///
/// `None` when the variable is unset or empty (raw `cargo nextest
/// run` — no orchestrator); the publication helper then serializes
/// the process-local pre-clear behavior.
/// `Some` lets pre-clear record/match the `.ktstr_run_epoch`
/// sentinel so a later peer process skips re-wiping a dir this
/// session already cleared, sparing the peers' sidecars.
fn run_session_token() -> Option<String> {
    std::env::var(crate::KTSTR_RUN_EPOCH_ENV)
        .ok()
        .filter(|v| !v.is_empty())
}

/// Predicate: is `path` an atomic-write staging file produced by
/// [`serialize_and_write_sidecar`]?
///
/// True iff the filename matches the `<test>-<hash>.ktstr.json.tmp.…`
/// shape — `is_sidecar_filename` rejects these because the
/// extension is `<run_id>` rather than `json`, so a separate
/// predicate is needed for the epoch-reset sweep
/// that reaps orphaned staging files. Filename-component check
/// (rather than full-path string) for the same load-bearing reason
/// `is_sidecar_filename` uses `Path::file_name()`: a `.ktstr.json.tmp.`
/// substring inside an ancestor segment must not match.
fn is_sidecar_staging_filename(path: &std::path::Path) -> bool {
    path.file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|n| n.contains(".ktstr.json.tmp."))
}

/// Wall-clock timeout for [`acquire_run_dir_publication_lock`].
/// Matching writers normally take SH immediately; EX is held only
/// for an epoch reset and its first atomic publication. A holder
/// that does not release within 30 s has stalled, and surfacing that
/// as an actionable error beats hanging the test run indefinitely.
/// The timeout is asymmetric with the cache-store 300 s timeout
/// because this rail protects only bounded run-directory metadata I/O.
const RUN_DIR_LOCK_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// Compute the per-run-key flock sentinel path for `dir`.
///
/// Layout: `{dir.parent()}/.locks/{dir.file_name()}.lock`. When
/// `dir = {runs_root}/{key}` (the production default-path shape),
/// this resolves to `{runs_root}/.locks/{key}.lock`. Sourced from
/// [`crate::flock::LOCK_DIR_NAME`] so a relocation of the lock
/// subdirectory updates one place across both this surface and
/// the cache module.
///
/// Returns `None` when `dir` has no parent (root) or no
/// `file_name` component (current dir, root) — neither case is
/// reachable on the production default path
/// ([`runs_root`] always returns a non-root multi-component
/// path), but the function is total over the input domain so a
/// future caller passing an unusual path surfaces a clean `None`
/// rather than panicking on `unwrap`.
///
/// Pure function over the input path — no I/O. The caller is
/// responsible for materializing the parent `.locks/`
/// subdirectory before opening the lockfile —
/// [`crate::flock::acquire_flock_with_timeout`] handles that
/// lazily.
fn run_dir_lock_path(dir: &std::path::Path) -> Option<PathBuf> {
    let parent = dir.parent()?;
    let leaf = dir.file_name()?;
    let mut filename = std::ffi::OsString::from(leaf);
    filename.push(".lock");
    Some(parent.join(crate::flock::LOCK_DIR_NAME).join(filename))
}

/// Acquire publication ownership for a verdict rewrite.
///
/// Unlike a primary write, finalization must never initialize or reset an
/// epoch: its sidecar belongs to the epoch that published it. A matching token
/// retains SH through rename; a mismatch returns `Ok(None)` without touching
/// the file. Raw no-token finalization retains the historical EX rail.
fn acquire_run_dir_finalize_lock(
    dir: &std::path::Path,
    session_token: Option<&str>,
) -> anyhow::Result<Option<std::os::fd::OwnedFd>> {
    let Some(token) = session_token else {
        return acquire_run_dir_flock_with_timeout(dir, RUN_DIR_LOCK_TIMEOUT).map(Some);
    };
    let shared = acquire_run_dir_flock_mode_with_timeout(
        dir,
        crate::flock::FlockMode::Shared,
        RUN_DIR_LOCK_TIMEOUT,
    )?;
    if session_sentinel_matches(dir, token) {
        Ok(Some(shared))
    } else {
        drop(shared);
        Ok(None)
    }
}

/// Acquire the run-directory publication rail.
///
/// A matching orchestrated epoch takes `LOCK_SH`, validates the sentinel while
/// holding it, and returns the fd for the caller to retain through atomic
/// rename. Peer writers for that epoch therefore publish concurrently. An
/// epoch reset requires `LOCK_EX`, so it cannot cross the token-validation to
/// rename interval.
///
/// A missing or mismatched epoch makes one non-blocking `LOCK_EX` election.
/// Losers wait for SH so a same-epoch initializer wakes the whole herd; a
/// genuinely different epoch proceeds to EX, rechecks, then performs
/// wipe -> sentinel exactly once. Raw no-token callers retain the historical
/// exclusive pre-clear-once path.
fn acquire_run_dir_publication_lock(
    dir: &std::path::Path,
    session_token: Option<&str>,
) -> anyhow::Result<(std::os::fd::OwnedFd, crate::flock::FlockMode)> {
    acquire_run_dir_publication_lock_with_timeout(dir, session_token, RUN_DIR_LOCK_TIMEOUT)
}

/// Timeout-parametrizable inner of [`acquire_run_dir_publication_lock`].
fn acquire_run_dir_publication_lock_with_timeout(
    dir: &std::path::Path,
    session_token: Option<&str>,
    timeout: std::time::Duration,
) -> anyhow::Result<(std::os::fd::OwnedFd, crate::flock::FlockMode)> {
    use crate::flock::FlockMode;

    let Some(token) = session_token else {
        let fd = acquire_run_dir_flock_with_timeout(dir, timeout)?;
        pre_clear_run_dir_once_for_session(dir, None);
        return Ok((fd, FlockMode::Exclusive));
    };

    loop {
        // Unlocked read is only a fast-path hint. The matching result is
        // authoritative only after the shared acquire and recheck below.
        if session_sentinel_matches(dir, token) {
            let shared = acquire_run_dir_flock_mode_with_timeout(dir, FlockMode::Shared, timeout)?;
            if session_sentinel_matches(dir, token) {
                return Ok((shared, FlockMode::Shared));
            }
            drop(shared);
        }

        // On an uninitialized directory exactly one member of a same-epoch
        // herd wins this non-blocking EX attempt. Losers wait for SH instead
        // of queueing for EX: once the winner publishes the sentinel and
        // releases, every loser wakes as a concurrent matching writer.
        if let Some(exclusive) = try_run_dir_flock(dir, FlockMode::Exclusive)? {
            if session_sentinel_matches(dir, token) {
                drop(exclusive);
                continue;
            }
            reset_run_dir_for_session(dir, Some(token))?;
            return Ok((exclusive, FlockMode::Exclusive));
        }

        let shared = acquire_run_dir_flock_mode_with_timeout(dir, FlockMode::Shared, timeout)?;
        if session_sentinel_matches(dir, token) {
            return Ok((shared, FlockMode::Shared));
        }
        drop(shared);

        // The incompatible holder was a different epoch's shared writer, not
        // an initializer. Wait for reset ownership, then recheck because a
        // same-epoch peer may have initialized while we were parked.
        let exclusive = acquire_run_dir_flock_with_timeout(dir, timeout)?;
        if session_sentinel_matches(dir, token) {
            drop(exclusive);
            continue;
        }
        reset_run_dir_for_session(dir, Some(token))?;
        return Ok((exclusive, FlockMode::Exclusive));
    }
}

/// Test-parametrizable exclusive wrapper over
/// [`acquire_run_dir_flock_mode_with_timeout`].
///
/// Resolves the per-run-key lockfile path via [`run_dir_lock_path`]
/// then delegates to [`crate::flock::acquire_flock_with_timeout`],
/// which handles parent-directory creation, the poll loop, the
/// `tracing::debug!` contention log, and the formatted timeout
/// error. The `context` argument names the run directory and the
/// `remediation` argument supplies the operator-facing recovery
/// hint about peer cargo ktstr test processes that the shared
/// helper appends to the timeout error.
///
/// Returns `Err` on:
/// - `run_dir_lock_path(dir)` returning `None` (no parent / no
///   file_name — production default path always satisfies both,
///   so this is a defensive arm),
/// - any error from [`crate::flock::acquire_flock_with_timeout`]
///   (parent directory creation failure, `try_flock` error, or
///   wall-clock `timeout` elapsing).
///
/// Returns `Ok(OwnedFd)` on successful acquire. Caller drops the
/// fd to release the kernel-side flock; the OFD-bound semantics
/// of `flock(2)` mean no explicit unlock call is required —
/// `OwnedFd::drop` runs `close(2)` which releases the lock when
/// no other fd refers to the same OFD (the fresh `try_flock`
/// open guarantees uniqueness).
fn acquire_run_dir_flock_with_timeout(
    dir: &std::path::Path,
    timeout: std::time::Duration,
) -> anyhow::Result<std::os::fd::OwnedFd> {
    acquire_run_dir_flock_mode_with_timeout(dir, crate::flock::FlockMode::Exclusive, timeout)
}

/// Mode-parametrizable run-directory flock acquire used by publication.
fn acquire_run_dir_flock_mode_with_timeout(
    dir: &std::path::Path,
    mode: crate::flock::FlockMode,
    timeout: std::time::Duration,
) -> anyhow::Result<std::os::fd::OwnedFd> {
    let lock_path = run_dir_lock_path(dir).ok_or_else(|| {
        anyhow::anyhow!(
            "cannot derive run-dir lock path from {} (no parent or no file_name component)",
            dir.display(),
        )
    })?;
    let context = format!("run-dir {}", dir.display());
    crate::flock::acquire_flock_with_timeout(
        &lock_path,
        mode,
        timeout,
        &context,
        Some(
            "A peer cargo ktstr test process is writing sidecars to the \
             same {kernel}-{project_commit} directory; wait for it to \
             finish or kill it, then retry.",
        ),
    )
}

/// One non-blocking acquire against the run-directory lockfile.
fn try_run_dir_flock(
    dir: &std::path::Path,
    mode: crate::flock::FlockMode,
) -> anyhow::Result<Option<std::os::fd::OwnedFd>> {
    let lock_path = run_dir_lock_path(dir).ok_or_else(|| {
        anyhow::anyhow!(
            "cannot derive run-dir lock path from {} (no parent or no file_name component)",
            dir.display(),
        )
    })?;
    if let Some(parent) = lock_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create lock subdirectory {}", parent.display()))?;
    }
    crate::flock::try_flock(&lock_path, mode)
}

/// Emit a minimal sidecar for a PRE-VM-BOOT skip path.
///
/// Stats tooling enumerates sidecars to compute pass/skip/fail
/// rates; when a test bails before `run_ktstr_test_inner` reaches
/// the VM-run site that calls [`write_sidecar`], the skip is
/// invisible to post-run analysis — it shows up as a missing
/// result rather than a recorded skip.
///
/// This helper writes a sidecar flagged `skipped: true, passed: true`
/// with empty VM telemetry (no monitor, no stimulus events, no
/// verifier stats, no kvm stats, no payload metrics). Stats tooling
/// that subtracts skipped runs from the pass count treats the entry
/// correctly.
///
/// # Distinction from in-VM `AssertResult::skip` paths
///
/// There are TWO classes of skip, each with its own sidecar writer:
///
/// 1. **Pre-VM-boot skips** route through this helper
///    (`write_skip_sidecar`). Examples:
///    - `performance_mode` gated off via `KTSTR_NO_PERF_MODE`
///      (see `run_ktstr_test_inner`),
///    - `ResourceContention` at `builder.build()` or `vm.run()`
///      (all-slots-busy / transient host-resource contention — the
///      VM never booted).
///
///    These paths write a MINIMAL sidecar: empty VM telemetry,
///    `skipped: true`, and BOTH `payload` and `work_type` resolved
///    exactly as a run of this config would (the entry's declared
///    payload and [`crate::test_support::args::current_work_type`]) so
///    the skip shares the run's variant identity — a later run of the
///    same config overwrites this skip's sidecar instead of coexisting
///    with it. There is no VmResult to drain because the VM didn't boot.
///
/// 2. **In-VM `AssertResult::skip` returns** — e.g. the
///    empty-cpuset skip in `scenario::run_scenario`
///    (`AssertResult::skip("not enough CPUs/LLCs")`), or the
///    `need >= 4 CPUs` checks in `scenario::dynamic::*` — route
///    through [`write_sidecar`] at `run_ktstr_test_inner`'s end.
///    The guest VM fully booted, ran through scenario setup,
///    discovered the topology couldn't accommodate the test, and
///    returned early. The resulting sidecar carries REAL VM
///    telemetry (monitor, kvm_stats, verifier_stats) alongside
///    `skipped: true` — not a blind spot, just a richer record
///    than what this helper emits.
///
/// The asymmetry is intentional: pre-VM-boot skips have no
/// telemetry to record, while in-VM skips do. Stats tooling that
/// wants to uniformly discount skipped runs filters on
/// [`SidecarResult::skipped == true`] regardless of which writer
/// produced the entry — both set the field identically.
///
/// Returns `Err` when the sidecar directory cannot be created, the
/// JSON cannot be serialized, or the file write fails. Callers that
/// ignore the Result accept the risk of stats-tooling blind spots on
/// this run.
pub(crate) fn write_skip_sidecar(
    entry: &KtstrTestEntry,
    resolved_topology: &crate::vmm::topology::Topology,
) -> anyhow::Result<()> {
    let SchedulerFingerprint {
        scheduler,
        scheduler_commit,
        sysctls,
        kargs,
    } = scheduler_fingerprint(entry);
    let sidecar = SidecarResult {
        test_name: entry.name.to_string(),
        perf_delta_assertions: entry
            .perf_delta_assertions
            .iter()
            .map(|&a| a.into())
            .collect(),
        // The RESOLVED topology a run of this preset would boot
        // (resolve_vm_topology(entry, topo)), NOT the declared
        // entry.topology — for a topology gauntlet each preset boots a
        // distinct topology, so recording the declared value would make
        // every preset share one variant_hash and clobber. For a plain
        // test (no override) resolved == declared. The skip and the run
        // of one preset thus share a variant_hash (the run path records
        // the same resolved topology), so a flaky test that skips on one
        // attempt and runs on the retry writes one sidecar.
        topology: resolved_topology.to_string(),
        scheduler,
        scheduler_commit,
        // A skip resolves no scheduler binary (no run), so there is no
        // discovery path to record.
        resolve_source: None,
        project_commit: detect_project_commit(),
        // A skip never runs the payload. Still record the declared
        // payload name so stats tooling can attribute the skip to
        // the payload-gauntlet variant rather than losing the
        // association.
        payload: entry.payload.map(|p| p.name.to_string()),
        metrics: Vec::new(),
        passed: false,
        skipped: true,
        inconclusive: false,
        expected_failure: false,
        stats: Default::default(),
        monitor: None,
        // A skip never ran the VM, so no periodic captures fired.
        periodic_fired: 0,
        periodic_target: 0,
        // A skip never booted the VM, so it has no measured budget. 0/0
        // maps to None on the GauntletRow's cpu_budget dim (skips carry no
        // budget identity; the skipped=true flag, not a sentinel field
        // value, marks them).
        vcpus: 0,
        cpu_budget: 0,
        // A skip never booted the VM, so no vCPU schedstat was sampled.
        host_dilation: None,
        // Era marker: even a skip row carries the current denomination so a
        // pool never mixes eras within one marker value.
        throughput_denomination: ThroughputDenomination::CpuSec,
        stimulus_events: Vec::new(),
        // A skip never ran the workload, but it carries the SAME
        // work_type a run of this config would (current_work_type reads
        // the per-variant --ktstr-work-type arg, identical across nextest
        // retry attempts). That keeps the skip's variant_hash equal to
        // the run's, so a flaky test that skips on one attempt and runs
        // on the retry writes one sidecar (the retry overwrites the skip)
        // rather than two coexisting files the footer would both flag.
        // Skips stay identified by skipped=true, not by a work_type
        // sentinel (see the variant-hash + skipped-bool contract above).
        work_type: super::args::current_work_type(),
        verifier_stats: Vec::new(),
        kvm_stats: None,
        sysctls,
        kargs,
        kernel_version: detect_kernel_version(),
        kernel_commit: kernel_commit_for_sidecar(),
        timestamp: now_iso8601(),
        run_id: generate_run_id(),
        host: Some(crate::host_context::collect_host_context()),
        // Skip paths never reach `collect_results`, so cleanup
        // duration is undefined. Emit `null` per the sidecar's
        // symmetric serialize/deserialize contract.
        cleanup_duration_ms: None,
        run_source: detect_run_source(),
    };
    serialize_and_write_sidecar(&sidecar, "skip sidecar")
}

/// Best-effort write of a small per-test run marker JSON to the current
/// sidecar dir as `{test}-{variant_hash:016x}.{suffix}`. Shared by the
/// host-skip and probe-health markers the footer scans
/// ([`summarize_one_run_dir`]). Not variant-clashing with the sidecar /
/// dump artifacts (distinct suffix). Failures log at warn and are
/// swallowed — a missing marker only costs the footer one advisory line,
/// never a run's correctness. NOT atomically staged: the body is a
/// single small object a partial read would fail to parse (dropped by
/// the reader), and the mtime gate excludes any prior-run residue.
fn write_run_marker(entry_name: &str, variant_hash: u64, suffix: &str, body: &serde_json::Value) {
    let dir = sidecar_dir();
    if let Err(e) = std::fs::create_dir_all(&dir) {
        tracing::warn!(error = %e, path = %dir.display(), "ktstr: create dir for {suffix} marker");
        return;
    }
    let path = dir.join(format!("{entry_name}-{variant_hash:016x}.{suffix}"));
    match serde_json::to_string_pretty(body) {
        Ok(json) => {
            if let Err(e) = std::fs::write(&path, json) {
                tracing::warn!(error = %e, path = %path.display(), "ktstr: write {suffix} marker");
            }
        }
        Err(e) => tracing::warn!(error = %e, "ktstr: serialize {suffix} marker for '{entry_name}'"),
    }
}

/// Record a `.host-skip.json` marker for a test this HOST cannot run.
/// `class` is the host-insufficiency tag
/// (`topology_insufficient` / `resource_contention` /
/// `perf_mode_unavailable`); the footer groups by it. Keyed by the same
/// variant hash the skip's `.ktstr.json` sidecar uses so a gauntlet's
/// per-preset skips do not clobber.
pub(crate) fn write_host_skip_marker(
    entry: &KtstrTestEntry,
    resolved_topology: &crate::vmm::topology::Topology,
    class: &str,
) {
    let variant_hash =
        variant_hash_from_parts(entry, resolved_topology, &super::args::current_work_type());
    write_run_marker(
        entry.name,
        variant_hash,
        "host-skip.json",
        &serde_json::json!({ "test_name": entry.name, "class": class }),
    );
}

/// Record a `.probe-health.json` marker naming a per-test auto-repro
/// probe-pipeline problem with a short `reason`, for the footer's probe
/// block. `variant_hash` matches the run's other per-test artifacts.
pub(crate) fn write_probe_health_marker(entry_name: &str, variant_hash: u64, reason: &str) {
    write_run_marker(
        entry_name,
        variant_hash,
        "probe-health.json",
        &serde_json::json!({ "test_name": entry_name, "reason": reason }),
    );
}

/// Record an `.expect-err-load.json` marker for an `expect_err` test
/// whose inversion was satisfied by a scheduler LOAD/startup failure
/// (the scheduler never attached — [`crate::verifier::AttachOutcome::Died`]
/// / `NotAttached`) rather than the runtime error the test intends. The
/// footer names these so a suite that is silently green because no
/// scheduler could load on this kernel becomes visible. `variant_hash`
/// matches the run's other per-test artifacts.
pub(crate) fn write_expect_err_load_marker(entry_name: &str, variant_hash: u64) {
    write_run_marker(
        entry_name,
        variant_hash,
        "expect-err-load.json",
        &serde_json::json!({ "test_name": entry_name }),
    );
}

/// Write a sidecar JSON file for post-run analysis.
///
/// Output goes to the current run's sidecar directory
/// (`KTSTR_SIDECAR_DIR` override, or
/// `{CARGO_TARGET_DIR or "target"}/ktstr/{kernel}-{project_commit}/`,
/// where `{project_commit}` is the project HEAD short hex with
/// `-dirty` when the worktree differs).
///
/// `payload_metrics` is the accumulated per-invocation output from
/// `ctx.payload(X).run()` / `.spawn().wait()` calls made in the
/// test body. Empty vec when the test body never called
/// `Ctx::payload` (scheduler-only tests, host-only probes).
///
/// Returns `Err` when the sidecar directory cannot be created, the
/// JSON cannot be serialized, or the file write fails. Callers that
/// ignore the Result accept the risk of stats-tooling blind spots on
/// this run.
pub(crate) fn write_sidecar(
    entry: &KtstrTestEntry,
    vm_result: &vmm::VmResult,
    stimulus_events: &[StimulusEvent],
    check_result: &AssertResult,
    work_type: &str,
    payload_metrics: &[PayloadMetrics],
    resolved_topology: &crate::vmm::topology::Topology,
) -> anyhow::Result<()> {
    let SchedulerFingerprint {
        scheduler,
        scheduler_commit,
        sysctls,
        kargs,
    } = scheduler_fingerprint(entry);
    let sidecar = SidecarResult {
        test_name: entry.name.to_string(),
        perf_delta_assertions: entry
            .perf_delta_assertions
            .iter()
            .map(|&a| a.into())
            .collect(),
        // The RESOLVED topology this run booted (resolve_vm_topology
        // result), NOT the declared entry.topology — a topology gauntlet
        // boots a distinct topology per preset, so the declared value
        // would collapse every preset to one variant_hash. resolved ==
        // declared for a plain test (no override).
        topology: resolved_topology.to_string(),
        scheduler,
        scheduler_commit,
        // Scheduler-resolution provenance, carried on VmResult from the
        // host eval path (run_ktstr_test_inner_impl resolves the binary
        // once and stamps the source), mirroring how vcpus / cpu_budget
        // ride VmResult to this stamp.
        resolve_source: vm_result.resolve_source.clone(),
        project_commit: detect_project_commit(),
        payload: entry.payload.map(|p| p.name.to_string()),
        metrics: payload_metrics.to_vec(),
        passed: check_result.is_pass(),
        skipped: check_result.is_skip(),
        inconclusive: check_result.is_inconclusive(),
        // Raw scenario verdict at write time; the dispatch-layer
        // finalize (finalize_sidecar_verdict) overwrites these bits with
        // the post-inversion outcome and sets expected_failure.
        expected_failure: false,
        stats: check_result.stats.clone(),
        monitor: vm_result.monitor.as_ref().map(|m| m.summary.clone()),
        periodic_fired: vm_result.periodic_fired,
        periodic_target: vm_result.periodic_target,
        vcpus: vm_result.vcpus,
        cpu_budget: vm_result.cpu_budget,
        // Measured host dilation (evidence for the overcommit marker),
        // derived from the run's raw vCPU-thread schedstat totals.
        host_dilation: vm_result.host_vcpu_schedstat.and_then(|s| s.dilation()),
        // Era marker: this build's throughput rates are CPU-second
        // denominated (see ThroughputDenomination).
        throughput_denomination: ThroughputDenomination::CpuSec,
        stimulus_events: stimulus_events.to_vec(),
        work_type: work_type.to_string(),
        verifier_stats: vm_result.verifier_stats.clone(),
        kvm_stats: vm_result.kvm_stats.clone(),
        sysctls,
        kargs,
        kernel_version: detect_kernel_version(),
        kernel_commit: kernel_commit_for_sidecar(),
        timestamp: now_iso8601(),
        run_id: generate_run_id(),
        host: Some(crate::host_context::collect_host_context()),
        cleanup_duration_ms: vm_result.cleanup_duration.map(|d| d.as_millis() as u64),
        run_source: detect_run_source(),
    };
    serialize_and_write_sidecar(&sidecar, "sidecar")
}

#[cfg(test)]
mod tests;
