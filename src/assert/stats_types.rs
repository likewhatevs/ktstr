use super::*;

/// Verdict for a single test scenario.
///
/// # Reading the verdict
///
/// Inspect the terminal verdict via [`Self::outcome`] (returns the
/// folded [`Outcome`] enum) or the convenience accessors
/// [`Self::is_pass`] / [`Self::is_fail`] / [`Self::is_inconclusive`] /
/// [`Self::is_skip`]. Iterate the per-variant payloads via
/// [`Self::failure_details`] (all [`Outcome::Fail`] payloads),
/// [`Self::inconclusive_details`] (all [`Outcome::Inconclusive`]
/// payloads), and [`Self::skip_details`] (all [`Outcome::Skip`]
/// payloads). All four bool accessors mirror
/// [`Outcome::is_pass`] / [`Outcome::is_fail`] /
/// [`Outcome::is_inconclusive`] / [`Outcome::is_skip`].
///
/// # Recording outcomes
///
/// Producers use the atomic mutators [`Self::record_fail`] /
/// [`Self::record_skip`] / [`Self::record_inconclusive`] /
/// [`Self::record_pass`] (each pushes a single [`Outcome`] variant
/// onto [`Self::outcomes`]) and the escape hatch
/// [`Self::record_outcome`] for pre-folded values. Constructors
/// [`Self::pass`] / [`Self::skip`] / [`Self::fail`] seed the
/// outcomes vec with the corresponding variant; [`Self::pass`] is
/// zero-allocation (empty vec; the Pass identity element).
///
/// **Wire-format stability**: this struct is postcard-serialized as
/// part of the in-VM `MSG_TYPE_TEST_RESULT` payload and as
/// sidecar artifacts under `~/.cache/ktstr`. The wire format is
/// **not stable across crate versions** — pre-1.0, fields can be
/// added, removed, or reshaped at any time, and old sidecars must
/// be regenerated after upgrades (re-running the affected tests
/// produces a fresh sidecar). Per the project's pre-1.0 no-compat
/// stance ([`crate::scenario`] module-level doc), no
/// `#[serde(default)]` shims are added for old payloads.
#[must_use = "test verdict is lost if not checked"]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AssertResult {
    /// Recorded terminal verdicts in emission order, one entry per
    /// check that explicitly called [`Self::record_pass`],
    /// [`Self::record_skip`], [`Self::record_inconclusive`], or
    /// [`Self::record_fail`] (plus the single entry seeded by
    /// [`Self::skip`] / [`Self::fail`] constructors).
    ///
    /// **Empty `outcomes` is the Pass identity** — [`Self::pass`]
    /// constructs with `outcomes: vec![]`, [`Self::outcome`] folds
    /// the vec via [`Outcome::merge`] starting from
    /// [`Outcome::Pass`], so a never-touched accumulator naturally
    /// resolves to Pass without any allocation. `record_pass()` is
    /// for the rare case where a test explicitly records a passing
    /// check (e.g. per-check helpers that document what passed);
    /// `pass()` is the zero-state "nothing failed so far"
    /// constructor.
    ///
    /// The folded terminal verdict is computed by [`Self::outcome`]
    /// per the precedence `Fail > Inconclusive > Pass > Skip`. Use
    /// [`Self::is_pass`] / [`Self::is_fail`] /
    /// [`Self::is_inconclusive`] / [`Self::is_skip`] for bool
    /// checks; use [`Self::failure_details`] /
    /// [`Self::inconclusive_details`] / [`Self::skip_details`] to
    /// iterate the per-variant [`AssertDetail`] payloads.
    pub outcomes: Vec<Outcome>,
    /// Structured records of every passing claim. Counterpart to
    /// [`Self::outcomes`]: where `outcomes` carries terminal-verdict
    /// records (Fail/Skip/Pass per-check), `passes` carries the
    /// positive confirmations every comparator's pass arm emits via
    /// [`Verdict`]'s `record_pass_unary` / `record_pass_binary`
    /// helpers.
    /// Empty in tests that don't exercise the structured-pass path
    /// (the no-claim base case), populated whenever a [`Verdict`]
    /// records claims. The auto-repro renderer iterates both vecs
    /// to compose the bracketed phase-grouped output that surfaces
    /// passing context alongside failing assertions.
    ///
    /// **Bounded by [`MAX_RECORDED_PASSES`]** — past that count,
    /// further pushes drop on the floor and a single sentinel
    /// record named [`PASSES_TRUNCATION_SENTINEL_NAME`] appears at
    /// the tail. Use the sentinel-name check (not `len()`
    /// arithmetic) to detect truncation.
    ///
    /// **Test-author convention**: do NOT pin `result.passes` shape
    /// or contents in test assertions unless the test exists
    /// specifically to verify the structured-pass surface (e.g.
    /// the auto-repro renderer's own coverage tests). The field
    /// exists for the renderer's consumption; pinning it
    /// elsewhere makes the test surface viral — every new
    /// comparator that fires under the test starts churning the
    /// pin. Pin `outcome()`, `failure_details()`, and `measurements` for
    /// scenario verification.
    pub passes: Vec<PassDetail>,
    /// Aggregated stats from all workers in this scenario.
    pub stats: ScenarioStats,
    /// Structured measurements attached via [`Self::note_value`] /
    /// [`Verdict::note_value`]. Distinct from [`Self::outcomes`] —
    /// outcomes carry typed verdict variants with `AssertDetail`
    /// payloads for operator triage, `measurements` carries typed
    /// `(key, NoteValue)` pairs for programmatic consumption (sidecar
    /// parsers, `stats compare`, regression dashboards).
    pub measurements: std::collections::BTreeMap<String, NoteValue>,
    /// Informational annotations attached via [`Self::note`] /
    /// [`Verdict::note`]. Structurally separated from [`Self::outcomes`]
    /// so the failure stream stays purely failure-shaped: sidecar
    /// consumers iterating `details` count real failures without
    /// the "forgot to filter notes" silent-miscount class of bug
    /// that the prior `DetailKind::Note` variant on [`AssertDetail`]
    /// invited. The auto-repro renderer surfaces these alongside the
    /// failure summary so the operator still sees them on a failing
    /// run.
    pub info_notes: Vec<InfoNote>,
}

/// Per-cgroup statistics from worker telemetry.
///
/// # Percentile convention
///
/// `p99_wake_latency_us` and `median_wake_latency_us` are computed
/// by `percentile` using the NEAREST-RANK (Type 1) definition:
/// the value at `ceil(n * p) - 1` in sorted order. No interpolation
/// between samples. This matches the percentile convention used
/// throughout schbench and the BPF latency histograms the project
/// cross-references, so a `ktstr` p99 reading aligns with a
/// schbench `lat99` without adjustment. For small `n` (wake
/// reservoirs cap at `MAX_WAKE_SAMPLES = 100_000` per worker —
/// see `workload.rs`) nearest-rank is also numerically stable —
/// interpolation between the two nearest ranks would be
/// implementation-defined at sample-set boundaries.
///
/// # CV pooling scope
///
/// `wake_latency_cv` is POOLED across every sample from every
/// worker in the cgroup, not a per-worker CV averaged back. That
/// collapses per-worker dispersion into the cgroup-wide signal:
/// two workers with uniformly low jitter but different means
/// produce a high pooled CV (mean-shift between workers inflates
/// stddev), while per-worker CV would show neither worker as
/// bad. This is intentional for the fairness threshold
/// (`max_wake_latency_cv`): a scheduler that gives worker A
/// 10µs wakes and worker B 1ms wakes is failing fairness even if
/// each worker on its own is tight. Tests comparing single-worker
/// behavior should scope their assertions to per-worker data
/// rather than this aggregate.
///
/// # Derived ratios
///
/// Two metrics are DERIVED rather than measured and live as
/// `&self` methods, NOT as serde-serialized fields:
/// [`Self::wake_latency_tail_ratio`] (= p99/median) and
/// [`Self::iterations_per_worker`] (= total_iterations/num_workers).
/// Pre-1.0 cleanup eliminated the prior stored-field shadow and
/// `derive_ratios` stamper. Consumers always recompute on read,
/// so a hand-constructed fixture or a deserialized sidecar from an
/// older build cannot silently carry a stale ratio. The run-level
/// worst-cgroup tail ratio (`crate::stats::MetricKind::WakeLatencyTailRatio`,
/// an `ext_metrics` entry) and the iterations efficiencies
/// (`worst_iterations_per_worker` / `worst_iterations_per_cpu_sec`) are all
/// re-pooled POST-merge by [`populate_run_distribution_metrics`] — the tail
/// ratio as the max over [`Self::wake_latency_tail_ratio`] across per-cgroup
/// [`Self`] entries, the efficiencies lowest-wins from
/// [`Self::iterations_per_worker`] / [`Self::iterations_per_cpu_sec`].
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize, crate::Claim)]
pub struct CgroupStats {
    /// Cgroup name (the workload-handle label this telemetry belongs to),
    /// or empty for unlabeled call sites (`collect_all`, bare
    /// `assert_cgroup`). Set post-hoc by `collect_handles` (in
    /// `crate::scenario`) where the name is in scope; `cgroup_stats`
    /// itself has only the reports and leaves it empty. Lets a PASSING-run
    /// consumer say which cgroup's work landed on which CPUs.
    pub cgroup_name: String,
    /// Number of workers in this cgroup.
    pub num_workers: usize,
    /// Distinct CPUs the workers in this cgroup actually ran on (union of
    /// each [`crate::workload::WorkerReport::cpus_used`]). `num_cpus` is
    /// its length, kept for the existing rollups; this set surfaces WHICH
    /// CPUs (not just how many) on every run, pass or fail.
    pub cpus_used: BTreeSet<usize>,
    /// Distinct CPUs used across all workers in this cgroup
    /// (`cpus_used.len()`).
    pub num_cpus: usize,
    /// Mean off-CPU percentage across workers (off_cpu_ns /
    /// wall_time_ns * 100). `None` when no worker reported a
    /// positive `wall_time_ns` (off-CPU% is undefined without wall
    /// time) — distinct from `Some(0.0)`, a measured "never off
    /// CPU". The `Option` keeps a not-measured cgroup from reading
    /// as a perfectly-on-CPU one in the telemetry consumers
    /// (`ScenarioStats.cgroups`).
    pub avg_off_cpu_pct: Option<f64>,
    /// Minimum off-CPU percentage across workers. `None` under the
    /// same no-measurable-wall-time condition as `avg_off_cpu_pct`.
    pub min_off_cpu_pct: Option<f64>,
    /// Maximum off-CPU percentage across workers. `None` under the
    /// same no-measurable-wall-time condition as `avg_off_cpu_pct`.
    pub max_off_cpu_pct: Option<f64>,
    /// `max_off_cpu_pct - min_off_cpu_pct`. Measures scheduling
    /// fairness within the cgroup. `None` when off-CPU% was not
    /// measured (no worker with positive wall time) — a not-measured
    /// cgroup is inconclusive for fairness, NOT "spread 0 = perfectly
    /// fair". `Some(0.0)` means a real measured zero spread.
    pub spread: Option<f64>,
    /// Longest scheduling gap across all workers (ms).
    pub max_gap_ms: u64,
    /// CPU where the longest scheduling gap occurred.
    pub max_gap_cpu: usize,
    /// Sum of CPU migration counts across all workers.
    pub total_migrations: u64,
    /// Migrations per iteration (total_migrations / total_iterations).
    pub migration_ratio: f64,
    /// 99th percentile wake latency across all workers (microseconds).
    pub p99_wake_latency_us: f64,
    /// Median wake latency across all workers (microseconds).
    pub median_wake_latency_us: f64,
    /// Coefficient of variation (stddev / mean) of wake latencies.
    ///
    /// Computed over the POOLED latency samples from every worker in
    /// the cgroup, not as a mean of per-worker CVs. Per-worker
    /// dispersion is therefore masked: a cgroup with one tight
    /// worker and one wildly variable worker can report a moderate
    /// pooled CV that looks healthier than either constituent. Use
    /// [`WorkerReport::wake_latencies_ns`] directly if per-worker
    /// CV is needed.
    pub wake_latency_cv: f64,
    /// Sum of iteration counts across all workers.
    pub total_iterations: u64,
    /// Sum of per-worker on-CPU time (nanoseconds), from each worker's
    /// schedstat run time ([`crate::workload::WorkerReport::schedstat_cpu_time_ns`]
    /// — `task->se.sum_exec_runtime`, the FIRST `/proc/<pid>/schedstat` field
    /// (`sched_info` supplies only the run_delay/pcount fields 2/3, not the
    /// on-CPU time), the summable per-thread proxy for the cgroup's
    /// `cpu.stat usage_usec`).
    /// Denominator for [`Self::iterations_per_cpu_sec`], the
    /// overcommit-invariant per-cell rate. `0` when no worker reported on-CPU
    /// time (the accessor then returns `None`).
    pub total_cpu_time_ns: u64,
    /// Mean schedstat run delay across workers (microseconds).
    pub mean_run_delay_us: f64,
    /// Worst schedstat run delay across workers (microseconds).
    pub worst_run_delay_us: f64,
    /// Fraction of pages on the expected NUMA node(s) (0.0-1.0).
    /// Derived from `/proc/self/numa_maps` and the worker's
    /// [`MemPolicy`](crate::workload::MemPolicy).
    pub page_locality: f64,
    /// Cross-node page migration ratio from `/proc/vmstat`
    /// `numa_pages_migrated` delta divided by total allocated pages.
    pub cross_node_migration_ratio: f64,
    /// Extensible metrics for the generic comparison pipeline.
    pub ext_metrics: BTreeMap<String, f64>,
}

/// Per-phase per-cgroup raw telemetry components — the per-phase analogue of
/// [`CgroupStats`]. Holds RAW components (sample vectors + counters), NOT the
/// reduced ratios/percentiles [`CgroupStats`] computes, so whole-run and
/// cross-run aggregates RE-POOL from the components at every level (the
/// per-phase telemetry thesis: an aggregate is recomputed over the pooled
/// components, never averaged from ready-made per-phase reductions — a
/// percentile or weighted ratio cannot be recovered from per-phase scalars).
/// Covers every TYPED [`CgroupStats`] reduction: avg/min/max off-CPU% and
/// spread from `off_cpu_pcts`; p99/median/CV wake latency from
/// `wake_latencies_ns`; mean/worst run-delay from `run_delays_ns`;
/// migration_ratio, iterations_per_cpu_sec, iterations_per_worker,
/// page_locality, cross_node_migration_ratio from their counter components;
/// the COUPLED worst gap (ms + the CPU that owned it) from `max_gap_ms` /
/// `max_gap_cpu`; cpus_used / num_cpus from `cpus_used`. EXCLUDES
/// [`CgroupStats::ext_metrics`] (the generic extensible map — a per-phase
/// per-cgroup custom metric is a future extension, not part of the typed
/// carrier). Lives in [`PhaseBucket::per_cgroup`], keyed by cgroup name. The
/// structural carrier is empty until a capture path populates it per phase.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize, crate::Claim)]
pub struct PhaseCgroupStats {
    /// Worker count in this cgroup for the phase — the denominator for the
    /// re-pooled per-worker iteration rate (`iterations_per_worker` =
    /// `total_iterations` / this). This is a set CARDINALITY (`reports.len()`),
    /// not a kernel counter, but it SUMs in `merge` because a single cgroup name
    /// can emit MULTIPLE carriers in one step — `collect_handles` builds one per
    /// `WorkloadHandle`, and a `CgroupDef` with several `WorkSpec` entries
    /// (`.work(..).work(..)`) spawns one handle per `WorkSpec` under the same
    /// name (`apply_setup`). Those carriers cover DISJOINT worker subsets, so the
    /// cardinality of their union is the SUM (4 + 2 → 6), matching [`cgroup_stats`]
    /// over the pooled reports (`reports.len()`); a MAX would understate the count
    /// and inflate `iterations_per_worker`. (The disjointness is the real
    /// justification — were carriers ever to overlap, the SUM would over-count.)
    pub num_workers: usize,
    /// Distinct CPUs the cgroup's workers ran on in the phase (union of each
    /// worker's `cpus_used`). Re-pools [`CgroupStats::cpus_used`] / `num_cpus`
    /// (= the set / its length) via a set UNION.
    pub cpus_used: std::collections::BTreeSet<usize>,
    /// Pooled per-wakeup latency samples (ns) across the cgroup's workers in
    /// the phase, un-reduced so p99 / median / CV re-pool over the combined set.
    /// The POOL is reservoir-capped at `MAX_WAKE_SAMPLES` (the per-worker bound,
    /// re-applied when same-name carriers merge so the carrier payload stays
    /// bounded on the size-limited guest bulk port — without it the pool would be
    /// `workers × MAX_WAKE_SAMPLES`); `wake_sample_total` carries the true
    /// pre-cap population. The CARRIER-level reductions divide by
    /// `wake_latencies_ns.len()` (this capped pool size), NOT by
    /// `wake_sample_total`: [`Self::wake_summary`] takes p99 / median over `len`,
    /// and [`cgroup_stats`] computes `cv = stddev/mean` with
    /// `n = all_latencies.len()`. The RUN-level cross-phase re-pool
    /// ([`populate_run_distribution_metrics`]) instead population-WEIGHTS (see
    /// the PARITY CONTRACT below): its CV / mean divide by Σ per-sample weights
    /// (the reconstructed true population), which equals `len` only below the cap.
    ///
    /// PARITY CONTRACT (the one component whose parity is size-dependent): for
    /// pools ≤ `MAX_WAKE_SAMPLES` the reservoir IS the full concatenation, so the
    /// p99 / median / CV re-pool reproduces [`cgroup_stats`] VALUE-FOR-VALUE.
    /// Above the cap the carrier holds a distribution-preserving reservoir
    /// SUBSAMPLE while [`cgroup_stats`] reduces over the full per-worker concat,
    /// so the re-pool is DISTRIBUTION-EQUIVALENT, not byte-identical (the bounded
    /// bulk-port frame forbids carrying the full pool; staged reservoirs cannot be
    /// byte-identical to a single full-pool reduction). This is BY DESIGN:
    /// `cgroup_stats` stays the uncapped run-level authority (capping it to match
    /// the carrier would discard most of a multi-worker cgroup's samples to chase
    /// a sub-display-precision artifact), and the carrier's >cap merge is WEIGHTED
    /// by `wake_sample_total` (`Self::weighted_merge_reservoirs`) so the subsample
    /// is an UNBIASED sample of the combined population — no smaller-population
    /// skew. Both layers de-skew the cap: the carrier MERGE weights by
    /// `wake_sample_total` (`Self::weighted_merge_reservoirs`), and the
    /// cross-PHASE run-level pool in `populate_run_distribution_metrics` weights
    /// each phase carrier's samples by `wake_sample_total / wake_latencies_ns.len()`
    /// (so a phase that exceeded the cap contributes by true population, not
    /// capped length) and reduces with the weighted percentile / moments — the
    /// prior length-weighted concat is gone. Below the cap every weight is 1.0,
    /// so the weighted P99 / median / mean / worst are BYTE-identical to the
    /// unweighted concat; the weighted CV matches only within ~1e-9 (it sums in
    /// f64 where the unweighted path sums the mean in u64 — a weighted variance
    /// cannot keep the u64 sum).
    pub wake_latencies_ns: Vec<u64>,
    /// True wakeup count before reservoir clamping (`wake_latencies_ns` is
    /// capped), so the re-pool can report the real population size. An
    /// intentional ADDITION over [`CgroupStats`] (which has no such field), NOT
    /// a mirrored reduction — do not strip it in a strict-parity audit; it is
    /// the only source of the true wakeup population once `wake_latencies_ns` is
    /// reservoir-clamped, and it is for REPORTING, not the CV denominator.
    pub wake_sample_total: u64,
    /// Pooled per-worker schedstat run-delay samples (RAW ns) for the phase,
    /// un-reduced so mean / worst run-delay re-pool over the combined set; the
    /// re-pool converts ns → µs to match [`CgroupStats`]'s run-delay-µs fields.
    /// Stored as raw kernel ns (like `wake_latencies_ns`), not pre-converted,
    /// per the raw-component thesis. GRANULARITY: unlike `wake_latencies_ns`
    /// (one per WAKEUP), each entry here is ONE per-worker value — that
    /// worker's `sched_info.run_delay` delta over the carrier's window: the
    /// whole-run `schedstat_run_delay_ns` (end−start) for the step-local
    /// carrier, or the per-phase delta for the backdrop slice carrier. So the
    /// pool size is the worker
    /// count, the mean is the average per-worker total queued-to-run delay, and
    /// `worst_run_delay_us` selects the single worker with the largest total
    /// queued-to-run delay (NOT the worst single dispatch).
    pub run_delays_ns: Vec<u64>,
    /// Per-worker off-CPU% samples for the phase, un-reduced. Carried for the
    /// per-phase per-cgroup off-CPU% RENDER — the avg / min / max /
    /// spread of the combined set. NOT consumed by the run-level
    /// distributional re-pool: off-CPU% has no run-level Distribution metric
    /// (off-CPU%/spread is intrinsically per-cgroup, so the run-level
    /// `worst_spread` stays the cross-cgroup max of per-cgroup
    /// [`CgroupStats::spread`] via the typed [`AssertResult::merge`] fold, not a
    /// pooled distribution). An EMPTY vec is the not-measured state (no worker
    /// with positive wall time), preserving the not-measured vs measured-zero
    /// distinction [`CgroupStats`] keeps. Stored as raw samples, not pre-reduced
    /// extremes, because the mean is unrecoverable from min/max alone for >2
    /// workers. Each sample is `off_cpu_ns /
    /// wall_time_ns * 100`, where `off_cpu_ns = wall_time_ns - cpu_time_ns` and
    /// `cpu_time_ns` is the `CLOCK_THREAD_CPUTIME_ID` thread on-CPU time
    /// (workload/worker `off_cpu_ns` at report build). `total_cpu_time_ns` is a
    /// DISTINCT on-CPU measurement (`schedstat_cpu_time_ns`, the `/proc`
    /// schedstat `se.sum_exec_runtime`): both ultimately track on-CPU runtime but
    /// are sampled at different points (the `CLOCK_THREAD_CPUTIME_ID` read folds
    /// the in-flight delta; the schedstat field reads the stored value), so the
    /// two need not be byte-identical and must not be cross-wired in a re-pool.
    pub off_cpu_pcts: Vec<f64>,
    /// Sum of per-worker CPU-migration counts in the phase (Counter).
    pub total_migrations: u64,
    /// Sum of per-worker iteration counts in the phase (Counter).
    pub total_iterations: u64,
    /// Sum of per-worker on-CPU time (ns) in the phase — the
    /// overcommit-invariant rate denominator (Counter). Sourced from
    /// `schedstat_cpu_time_ns` (the `/proc` schedstat `se.sum_exec_runtime`,
    /// rq-charged on-CPU ns) — a DISTINCT on-CPU-time sample from the
    /// `CLOCK_THREAD_CPUTIME_ID` time behind `off_cpu_pcts` (different sample
    /// point; not byte-identical), so do not cross-wire the two in a re-pool.
    pub total_cpu_time_ns: u64,
    /// Pages on the expected NUMA node(s) — page-locality numerator. A genuine
    /// per-thread numa_maps count (Counter, SUM across workers/sources).
    pub numa_pages_local: u64,
    /// Total allocated pages — the SHARED denominator for BOTH page_locality
    /// (`numa_pages_local` / this) AND cross_node_migration_ratio
    /// (`cross_node_migrated` / this). A genuine per-thread numa_maps count
    /// (Counter, SUM); the kernel computes both ratios over the identical page
    /// total, so one field serves both — a separate cross_node_total would
    /// invite a silent desync.
    pub numa_pages_total: u64,
    /// Cross-node migrated pages — cross_node_migration_ratio numerator
    /// (denominator is `numa_pages_total`). A SYSTEM-WIDE
    /// `/proc/vmstat numa_pages_migrated` delta each worker observes
    /// redundantly, so this is a PEAK (MAX across workers/sources), NOT a
    /// Counter — summing would inflate it by the worker count (mirrors
    /// [`CgroupStats`]'s deliberate max-fold of the same quantity).
    pub cross_node_migrated: u64,
    /// Longest scheduling gap (ms) across the cgroup's workers in the phase,
    /// coupled with `max_gap_cpu`. A Peak folded as an ARGMAX of the (ms, cpu)
    /// pair so the worst gap and its CPU survive together — mirrors
    /// [`CgroupStats`]'s `max_gap_ms` / `max_gap_cpu` coupling (a bare
    /// independent max would desync the gap from its CPU).
    pub max_gap_ms: u64,
    /// CPU that owned the worst scheduling gap — `max_gap_ms`'s argmax
    /// companion. Folded together with `max_gap_ms`, never independently.
    pub max_gap_cpu: usize,
    /// True when this carrier's raw sample vectors (`wake_latencies_ns` /
    /// `run_delays_ns` / `off_cpu_pcts`) were dropped by
    /// `AssertResult::strip_phase_cgroup_samples` to fit the size-limited guest
    /// bulk frame — distinct from a carrier that genuinely measured no samples.
    /// The reduced counters survive; only the per-phase distribution render
    /// loses its source, so the render shows "samples stripped" rather
    /// than the not-measured "n/a". Defaults to `false` (not stripped) and is set
    /// only on a carrier that actually HAD samples to drop; ORs across `merge` so
    /// a merged carrier is stripped if either input was.
    pub stripped: bool,
    /// Per-cgroup DERIVED scalar metrics for this (phase, cgroup), keyed by
    /// `crate::stats::MetricDef` name — the per-cgroup analog of
    /// [`PhaseBucket::metrics`] (which is the pooled-across-cgroups set). Populated
    /// post-fold by `derive_phase_metrics` (both the schbench per-phase family AND
    /// the non-schbench families — wake p99/median/cv, mean/max run-delay,
    /// avg/min/max/spread off-CPU%, and the migration / iterations / locality
    /// ratios) from the SAME reducers that fill
    /// the pooled map, so a
    /// test can query "metric M of cgroup C in phase P" as readily as the phase
    /// aggregate (N cgroups -> N queryable sets + the pooled aggregate). DERIVED, not a
    /// raw component: `PhaseCgroupStats::merge` leaves it empty and it is (re)derived
    /// POST-merge, exactly as the pooled map skips is_derived keys in
    /// `merge_matched_phase_buckets`. ALWAYS serialized (no `skip_serializing_if`):
    /// PhaseCgroupStats rides the postcard bulk-TLV port, a NON-self-describing
    /// POSITIONAL format — a conditionally-omitted field desyncs the byte stream and
    /// corrupts the fields after it (here `schbench`), so the field must always be
    /// present. No `serde(default)` either: pre-1.0, old sidecar/cache data is
    /// disposable and regenerates (no compat shim). Read via [`Self::get`]; the
    /// `crate::Claim` derive skips a `BTreeMap` field (matching
    /// [`PhaseBucket::metrics`], which has no Claim accessor either).
    pub metrics: std::collections::BTreeMap<String, f64>,
    /// Per-phase schbench engine metrics for a `WorkType::Schbench` backdrop
    /// cgroup (`None` for every non-schbench carrier). Pooled across the
    /// cgroup's workers by [`PhaseCgroupStats::merge`] (histogram
    /// bucket-add + integer-add of the run-delay raw pairs). The schbench
    /// per-phase derivation reads it, pools across cgroups, and derives the
    /// per-phase percentile / run-delay-mean / loop-count scalars into
    /// [`PhaseBucket::metrics`]. `pub(crate)`: the element type is `pub(crate)`
    /// and a per-phase carrier is internal — test authors read `PhaseBucket`,
    /// not this. Non-`pub` so the `crate::Claim` derive skips it (a percentile
    /// histogram has no meaningful scalar claim accessor).
    pub(crate) schbench: Option<crate::workload::schbench::run::SchbenchPhaseStats>,
}

impl PhaseCgroupStats {
    /// Component-wise union of two per-phase per-cgroup data for the SAME
    /// cgroup name (same `step_index`). Fold rule by component class:
    /// - sample vectors (`wake_latencies_ns`, `run_delays_ns`, `off_cpu_pcts`)
    ///   CONCAT, so the re-pool sees the combined set, never a mean of
    ///   per-source reductions;
    /// - the CPU set (`cpus_used`) UNIONs;
    /// - genuine Counters (`num_workers`, `wake_sample_total`,
    ///   `total_migrations`, `total_iterations`, `total_cpu_time_ns`,
    ///   `numa_pages_local`, `numa_pages_total`) SUM — `num_workers` included,
    ///   because a multi-`WorkSpec` cgroup emits one carrier per handle covering
    ///   DISJOINT worker subsets, so summing reproduces the pooled count (see
    ///   the `num_workers` field doc);
    /// - the one Peak, `cross_node_migrated`, takes the MAX (a system-wide
    ///   vmstat delta observed redundantly per worker, so summing would inflate
    ///   it);
    /// - the COUPLED worst gap (`max_gap_ms`, `max_gap_cpu`) folds as an
    ///   ARGMAX — the pair from whichever side has the larger ms (b's on tie,
    ///   matching the builders' `max_by_key` last-wins) so the gap and its CPU
    ///   stay bound together.
    ///
    /// The counter SUMs use plain `+`: debug builds panic on overflow rather
    /// than wrapping. The realistic magnitudes (iteration / ns counts far
    /// below `u64::MAX` even pooled across a long run) keep overflow
    /// unreachable; a loud debug panic is preferred over a silently wrong
    /// re-pool denominator.
    pub(crate) fn merge(a: PhaseCgroupStats, b: PhaseCgroupStats) -> PhaseCgroupStats {
        // Merge the two capped wake-latency reservoirs. Same-name carriers (a
        // multi-`WorkSpec` cgroup's per-handle carriers) merge ON THE GUEST before
        // the AssertResult is serialized over the bulk port, so K carriers must
        // not concat to K × MAX_WAKE_SAMPLES (it could overrun the 16 MiB frame,
        // flipping a PASS to a truncated FAIL).
        //
        // ≤cap: the concatenation IS the true combined population, so it passes
        // through unchanged — value-for-value parity with cgroup_stats for small
        // pools (only >cap pools become a subsample; see the `wake_latencies_ns`
        // field doc). >cap: a WEIGHTED reservoir merge weighted by each carrier's
        // true pre-cap population (`wake_sample_total`), so the merged sample is an
        // UNBIASED uniform sample of the combined population — NOT the
        // smaller-population-skewed reservoir-of-reservoirs an unweighted
        // concat-and-re-cap produced (which weighted by reservoir LENGTH ≈ 50/50,
        // ignoring the true populations).
        let cap = crate::workload::MAX_WAKE_SAMPLES;
        let wake_latencies_ns = if a.wake_latencies_ns.len() + b.wake_latencies_ns.len() <= cap {
            let mut v = a.wake_latencies_ns;
            v.extend(b.wake_latencies_ns);
            v
        } else {
            Self::weighted_merge_reservoirs(
                &a.wake_latencies_ns,
                a.wake_sample_total,
                &b.wake_latencies_ns,
                b.wake_sample_total,
                cap,
            )
        };
        let mut run_delays_ns = a.run_delays_ns;
        run_delays_ns.extend(b.run_delays_ns);
        let mut off_cpu_pcts = a.off_cpu_pcts;
        off_cpu_pcts.extend(b.off_cpu_pcts);
        let mut cpus_used = a.cpus_used;
        cpus_used.extend(b.cpus_used);
        // Coupled worst-gap ARGMAX: take the (ms, cpu) pair together from the
        // side with the larger gap (b's on tie, matching the builders'
        // max_by_key last-wins) so the CPU stays bound to the gap it owned — a
        // bare independent max would desync them. The last-wins tie-break is
        // parity-coupled to fold order: AssertResult::merge folds same-name
        // carriers in the order reports are pooled (handle iteration order), so
        // on an equal-gap tie this yields the same CPU as a single cgroup_stats
        // over the concatenated reports. A reordered fold would break that parity.
        let (max_gap_ms, max_gap_cpu) = if b.max_gap_ms >= a.max_gap_ms {
            (b.max_gap_ms, b.max_gap_cpu)
        } else {
            (a.max_gap_ms, a.max_gap_cpu)
        };
        // Per-phase schbench: OR-with-combine. Both Some → merge the pooled
        // histograms + integer-add the run-delay raw pairs / loop_count
        // (SchbenchPhaseStats::merge, the SAME associative+commutative op the
        // guest engine uses to pool across message threads); one Some → carry
        // it; both None (non-schbench cgroup) → None.
        let schbench = match (a.schbench, b.schbench) {
            (Some(mut x), Some(y)) => {
                x.merge(&y);
                Some(x)
            }
            (Some(x), None) | (None, Some(x)) => Some(x),
            (None, None) => None,
        };
        PhaseCgroupStats {
            num_workers: a.num_workers + b.num_workers,
            cpus_used,
            wake_latencies_ns,
            wake_sample_total: a.wake_sample_total + b.wake_sample_total,
            run_delays_ns,
            off_cpu_pcts,
            total_migrations: a.total_migrations + b.total_migrations,
            total_iterations: a.total_iterations + b.total_iterations,
            total_cpu_time_ns: a.total_cpu_time_ns + b.total_cpu_time_ns,
            numa_pages_local: a.numa_pages_local + b.numa_pages_local,
            numa_pages_total: a.numa_pages_total + b.numa_pages_total,
            cross_node_migrated: a.cross_node_migrated.max(b.cross_node_migrated),
            max_gap_ms,
            max_gap_cpu,
            stripped: a.stripped || b.stripped,
            // DERIVED, not a raw component: left EMPTY here and (re)derived post-merge
            // by derive_phase_metrics (the fold runs derive AFTER merging, the
            // same reason the pooled map skips is_derived keys in
            // merge_matched_phase_buckets). Folding stale per-operand metrics would
            // double-count; the post-merge re-derive is the sole producer.
            metrics: std::collections::BTreeMap::new(),
            schbench,
        }
    }

    /// Look up this cgroup's per-phase DERIVED value for `metric_name` — the
    /// per-cgroup analog of [`PhaseBucket::get`] (see [`Self::metrics`]). `None`
    /// when this cgroup carried no finite samples for the metric (the ABSENT
    /// discipline), distinct from `Some(0.0)` (a real reducer zero).
    pub fn get(&self, metric_name: &str) -> Option<f64> {
        self.metrics.get(metric_name).copied()
    }

    /// Like [`Self::get`] but panics citing the metric keys actually present when
    /// the metric is absent — use when the caller knows this cgroup MUST carry the
    /// metric in the phase.
    pub fn expect_metric(&self, metric_name: &str) -> f64 {
        self.get(metric_name).unwrap_or_else(|| {
            panic!(
                "PhaseCgroupStats::expect_metric: metric '{}' absent from this \
                 per-cgroup carrier (num_workers={}, stripped={}); keys present: {:?}. \
                 Causes: (a) the cgroup produced no finite samples for it; (b) metric \
                 name typo; (c) a non-schbench carrier has no schbench keys.",
                metric_name,
                self.num_workers,
                self.stripped,
                self.metrics.keys().collect::<Vec<_>>(),
            )
        })
    }

    /// The per-cgroup analog of [`PhaseBucket::cgroup_counter_total`] for the
    /// three per-cgroup Counters that live ONLY on the carrier (no derived
    /// `metrics` entry): `total_migrations`, `total_iterations`, and
    /// `total_cpu_time_ns`. Lets
    /// [`crate::vmm::VmResult::phase_cgroup_metric`] expose them by metric name
    /// symmetrically with the pooled [`crate::vmm::VmResult::phase_metric`]
    /// `cgroup_counter_total` fallback (the pooled `_total` suffix marks the
    /// cross-cgroup SUM; this per-cgroup form returns one cgroup's value). `None`
    /// for any other name (derived metrics are read via [`Self::get`]).
    pub fn cgroup_counter(&self, name: &str) -> Option<f64> {
        match name {
            "total_migrations" => Some(self.total_migrations as f64),
            "total_iterations" => Some(self.total_iterations as f64),
            "total_cpu_time_ns" => Some(self.total_cpu_time_ns as f64),
            _ => None,
        }
    }

    /// Merge two CAPPED uniform reservoirs into one of size ≤ `cap` that is a
    /// uniform sample of the COMBINED population. `a` is a uniform reservoir of
    /// `w_a` true samples, `b` of `w_b` (their `wake_sample_total` weights). Each
    /// output slot is drawn from `a` with probability `w_a / (w_a + w_b)` and from
    /// `b` otherwise; within a source the index is uniform. Composing the
    /// source-level uniform reservoir with the within-source uniform draw makes
    /// each output a uniform draw from the combined population, so the merged
    /// A-fraction is the TRUE `w_a / (w_a + w_b)`. This removes the equal-slot
    /// ("reservoir-of-reservoirs") skew an unweighted concat-and-re-cap imposes:
    /// two already-capped inputs concat ≈ 50/50 by LENGTH regardless of their true
    /// populations, over-counting the smaller-population carrier. Sampling WITH
    /// replacement is the correct estimator once the inputs are capped (each
    /// reservoir element stands for `w/len` population units; the pre-cap samples
    /// are gone).
    ///
    /// DETERMINISTIC: the xorshift64 stream is seeded from the inputs (populations +
    /// lengths) so the merge is a PURE function of its arguments — unlike
    /// `crate::workload::reservoir_push`, whose stream is gettid-seeded
    /// thread-local (a merge run twice would otherwise differ). The triple-shift
    /// mirrors the codebase's inline xorshift64 (`reservoir_push` /
    /// `io::xorshift64`).
    ///
    /// Assumes `w_a + w_b < 2^64` — a realistic wake population is far below it
    /// (2^64 wakeups is physically unreachable), so the single-u64 `s % total`
    /// draw spans `[0, total)`. Callers gate on `a.len() + b.len() > cap`, which
    /// (each input ≤ cap) guarantees both sources non-empty; the per-slot guards
    /// below stay safe for a degenerate hand-built input regardless.
    pub(crate) fn weighted_merge_reservoirs(
        a: &[u64],
        w_a: u64,
        b: &[u64],
        w_b: u64,
        cap: usize,
    ) -> Vec<u64> {
        if a.is_empty() && b.is_empty() {
            return Vec::new();
        }
        // Weights are the true populations; fall back to reservoir lengths if a
        // (hand-built) carrier reports zero population alongside non-empty samples,
        // keeping the split well-defined instead of dividing by a zero total. The
        // mixed case (one weight 0, the other > 0) is left as-is: a zero weight
        // sends every draw to the other source, the only defensible split for a
        // source claiming zero population. Production maintains wake_sample_total
        // >= len (reservoir_push counts every push), so neither edge is reachable
        // on the capture path.
        let (wa, wb) = if w_a == 0 && w_b == 0 {
            (a.len() as u128, b.len() as u128)
        } else {
            (w_a as u128, w_b as u128)
        };
        let total = wa + wb;
        // Loud-panic on the documented `w_a + w_b < 2^64` assumption (a realistic
        // wake population is far below it): if total exceeded u64::MAX the
        // `s as u128 % total` draw — s spans [0, 2^64) — could not reach
        // [2^64, total) and would silently bias the source split. Matches the merge
        // SUM's debug-panic-on-overflow discipline (loud over silently wrong).
        debug_assert!(
            total <= u64::MAX as u128,
            "weighted_merge_reservoirs: w_a + w_b overflows u64 ({total}); source draw would bias",
        );
        // Golden-ratio Weyl multiplier (the codebase's standard PRNG seed mixer);
        // a non-zero, input-derived seed makes the merge deterministic. xorshift64
        // has 0 as a fixed point, hence the fallback.
        const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut s =
            (w_a ^ w_b.rotate_left(32) ^ (a.len() as u64).rotate_left(16) ^ (b.len() as u64))
                .wrapping_mul(GOLDEN);
        if s == 0 {
            s = GOLDEN;
        }
        let step = |x: u64| {
            let mut v = x;
            v ^= v << 13;
            v ^= v >> 7;
            v ^= v << 17;
            v
        };
        let mut out = Vec::with_capacity(cap);
        for _ in 0..cap {
            s = step(s);
            // Defensive empty-source guards: caller gates ensure both non-empty,
            // but a stripped / zero-population fixture must never index an empty
            // slice.
            let from_a = if a.is_empty() {
                false
            } else if b.is_empty() {
                true
            } else {
                (s as u128 % total) < wa
            };
            s = step(s);
            if from_a {
                out.push(a[(s % a.len() as u64) as usize]);
            } else {
                out.push(b[(s % b.len() as u64) as usize]);
            }
        }
        out
    }

    /// Off-CPU% reduction for the per-phase per-cgroup render:
    /// `(avg, min, max, spread)` over [`Self::off_cpu_pcts`], or `None` when
    /// the vec is empty — the NOT-measured state (no worker had positive wall
    /// time). Reduces the SAME per-worker pcts [`cgroup_stats`] reduces
    /// (off_cpu_ns / wall_time_ns × 100), so for a phase spanning the whole run
    /// it reproduces that whole-run reduction; `spread = max − min`.
    /// `Some((0.0, ..))` is a MEASURED zero (distinct from the `None`
    /// not-measured state), preserving the discipline the empty-vec contract on
    /// `off_cpu_pcts` keeps. Display-only: never written back into a re-pool.
    pub fn off_cpu_summary(&self) -> Option<(f64, f64, f64, f64)> {
        let pcts = &self.off_cpu_pcts;
        if pcts.is_empty() {
            return None;
        }
        let min = pcts.iter().cloned().reduce(f64::min).expect("non-empty");
        let max = pcts.iter().cloned().reduce(f64::max).expect("non-empty");
        let avg = pcts.iter().sum::<f64>() / pcts.len() as f64;
        Some((avg, min, max, max - min))
    }

    /// Wake-latency reduction for the per-phase render:
    /// `(p99_us, median_us)` over the pooled [`Self::wake_latencies_ns`], or
    /// `None` when the pool is empty. Nearest-rank percentile via `percentile`
    /// (ns→µs once), reproducing [`cgroup_stats`]'s p99/median value-for-value
    /// for the ≤cap pool (and the run-level re-pool's `reduce_sorted_distribution`).
    /// Above `MAX_WAKE_SAMPLES` the pool is a distribution-preserving reservoir
    /// subsample (see [`Self::wake_latencies_ns`]), so p99/median is then
    /// distribution-equivalent, NOT byte-identical, to the full-pool reduction —
    /// the rendered tail stays accurate, only exact parity is size-bounded.
    /// `None`-on-empty omits the wake segment from the render rather than
    /// painting a misleading 0µs (the display analogue of `cgroup_stats`'s
    /// 0.0-sentinel, which has no Option to carry not-measured).
    pub fn wake_summary(&self) -> Option<(f64, f64)> {
        if self.wake_latencies_ns.is_empty() {
            return None;
        }
        let mut sorted = self.wake_latencies_ns.clone();
        sorted.sort_unstable();
        let p99 = percentile(&sorted, 0.99) as f64 / 1000.0;
        let median = percentile(&sorted, 0.5) as f64 / 1000.0;
        Some((p99, median))
    }

    /// Wake-latency coefficient of variation (stddev / mean) over the pooled
    /// [`Self::wake_latencies_ns`] (`n = len`), or `None` when the pool is
    /// empty (the not-measured state — so the `run_metrics` carrier writer
    /// omits all three wake keys together). `Some(0.0)` when
    /// the mean is zero is a MEASURED zero. Reproduces [`cgroup_stats`]'s
    /// `wake_latency_cv` value-for-value for the ≤cap pool (`Σv` in `u64`,
    /// same accumulation); above `MAX_WAKE_SAMPLES` the pool is a
    /// distribution-preserving reservoir subsample, so the CV is then
    /// distribution-equivalent, not byte-identical.
    pub fn wake_cv(&self) -> Option<f64> {
        if self.wake_latencies_ns.is_empty() {
            return None;
        }
        let n = self.wake_latencies_ns.len() as f64;
        let mean_ns = self.wake_latencies_ns.iter().sum::<u64>() as f64 / n;
        if mean_ns > 0.0 {
            let variance = self
                .wake_latencies_ns
                .iter()
                .map(|&v| (v as f64 - mean_ns).powi(2))
                .sum::<f64>()
                / n;
            Some(variance.sqrt() / mean_ns)
        } else {
            Some(0.0)
        }
    }

    /// Run-delay reduction for the per-phase render:
    /// `(mean_us, worst_us)` over the per-worker [`Self::run_delays_ns`] (raw
    /// ns), or `None` when empty. Divides ns→µs ONCE on the summed / maxed ns.
    /// `worst` reproduces [`cgroup_stats`]'s value-for-value (`max(ns)/1000 ==
    /// max(ns/1000)`, division is monotone). `mean` reproduces it to f64 ULP,
    /// not bit-exactly: this f64-sums then divides once (`Σns/n/1000`), while
    /// `cgroup_stats` divides each worker's ns by 1000 first then sums
    /// (`Σ(ns/1000)/n`) — the same value reassociated, differing only
    /// sub-display-precision (a divergent-input parity test bounds it at 1e-9).
    /// Each sample is
    /// one worker's whole-phase cumulative `sched_info.run_delay` delta, so
    /// `mean` is the average per-worker total queued-to-run delay and `worst`
    /// the largest. `None`-on-empty omits the segment.
    pub fn run_delay_summary(&self) -> Option<(f64, f64)> {
        if self.run_delays_ns.is_empty() {
            return None;
        }
        let n = self.run_delays_ns.len() as f64;
        // Sum in f64, NOT u64-then-cast: matches cgroup_stats's f64 accumulation
        // and cannot integer-overflow (an f64 sum saturates toward +inf; a u64
        // sum would panic in debug / silently wrap in release on a pathological
        // pool). Values are identical within the documented 1e-9 ULP bound.
        let mean = self.run_delays_ns.iter().map(|&v| v as f64).sum::<f64>() / n / 1000.0;
        let worst = *self.run_delays_ns.iter().max().expect("non-empty") as f64 / 1000.0;
        Some((mean, worst))
    }
}

impl CgroupStats {
    /// Wake-latency tail amplification:
    /// `p99_wake_latency_us / median_wake_latency_us`. Returns `0.0`
    /// when `median_wake_latency_us <= 0.0` so the result never
    /// propagates `NaN` / `Infinity` into downstream
    /// `finite_or_zero` filters. Method-only access (no stored
    /// shadow) — recomputed every call from the raw fields.
    ///
    /// Unitless; ≥1.0 by definition of order statistics (p99 cannot
    /// undershoot the median on the same sample set). Values far
    /// above 1.0 signal a long tail — the scheduler wakes most
    /// workers promptly but occasionally stalls some, a regression
    /// axis that neither `median_*` nor `p99_*` exposes in
    /// isolation.
    pub fn wake_latency_tail_ratio(&self) -> f64 {
        if self.median_wake_latency_us > 0.0 {
            self.p99_wake_latency_us / self.median_wake_latency_us
        } else {
            0.0
        }
    }

    /// Throughput per parallel degree:
    /// `total_iterations / num_workers`. `None` when
    /// `num_workers == 0` (no worker reported, so per-worker
    /// throughput is undefined — distinct from a measured zero);
    /// `Some(0.0)` when workers ran but completed zero iterations
    /// (a real throughput collapse). The `None` / `Some(0.0)` split
    /// is load-bearing: the run-level worst-cgroup re-pool in
    /// [`populate_run_distribution_metrics`] (the
    /// `MetricKind::WorstLowest` arm) must treat a measured zero as
    /// the worst reading (it wins the "lowest" bucket) while skipping
    /// a no-data cgroup — collapsing both to `0.0` would hide a
    /// starved cgroup behind the no-data sentinel. Method-only
    /// access (no stored shadow) — recomputed every call from the
    /// raw fields.
    ///
    /// Only meaningful across runs of the SAME variant (equal
    /// scenario duration): cross-variant comparison is misleading
    /// because this metric is NOT rate-normalized — a longer-
    /// running scenario racks up more iterations per worker even if
    /// the scheduler is identical. `stats compare`-style
    /// comparisons hold scenario, topology, and work_type constant
    /// before reading this method.
    pub fn iterations_per_worker(&self) -> Option<f64> {
        crate::assert::reductions::iterations_per_worker_of(self.num_workers, self.total_iterations)
    }

    /// Worker iterations per CPU-second of on-CPU time consumed by this
    /// cgroup's workers — `total_iterations / (total_cpu_time_ns / 1e9)`.
    ///
    /// Unlike [`Self::iterations_per_worker`] (raw work, which scales with
    /// the host-CPU budget delivered to the guest) and a wall-time rate
    /// (which also drops under host oversubscription), this is
    /// OVERCOMMIT-INVARIANT: under `cpu_budget < vcpus` a cell completes
    /// proportionally fewer iterations AND consumes proportionally less
    /// on-CPU time, so the ratio cancels the lost host-CPU-time factor. Use
    /// it to compare per-cgroup throughput across `cpu_budget` settings.
    ///
    /// `None` when `num_workers == 0` (no worker — undefined, distinct from a
    /// measured zero) or `total_cpu_time_ns == 0` (no on-CPU time captured;
    /// returns inconclusive rather than `Inf`). For a pure busy-spin
    /// workload this rate is ~constant by construction, so it measures
    /// CPU-time EFFICIENCY; for the cross-cell ALLOCATION balance use
    /// [`ScenarioStats::cgroup_balance_ratio`] over `iterations_per_worker`.
    pub fn iterations_per_cpu_sec(&self) -> Option<f64> {
        crate::assert::reductions::iterations_per_cpu_sec_of(
            self.num_workers,
            self.total_cpu_time_ns,
            self.total_iterations,
        )
    }
}

/// Identifier for a scenario phase. Newtype over `u16` carrying
/// the same 1-indexed encoding documented on every other
/// phase-touching site: `Phase::BASELINE` is the pre-first-Step
/// settle window (`u16` 0); `Phase::step(k)` is scenario Step `k`
/// at 1-indexed `u16` `k + 1`. The newtype catches the bug class
/// where a raw `u16` flows between sites that disagree about
/// 0-indexed vs 1-indexed Step encoding, and gives operators
/// readable construction at consumer sites (`Phase::BASELINE` /
/// `Phase::step(2)` instead of magic `0u16` / `3u16`).
///
/// Wire-format identical to a `u16` via `#[serde(transparent)]` —
/// the on-disk sidecar shape is unchanged from the bare-`u16`
/// pipeline, and existing JSON / typeshare consumers see the same
/// scalar field. `.phase_raw()` exposes the inner `u16` for paths
/// that hand the value to a serializer or formatter that does not
/// understand the newtype.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    Default,
    serde::Serialize,
    serde::Deserialize,
)]
#[serde(transparent)]
pub struct Phase(u16);

impl Phase {
    /// Pre-first-Step settle window. The framework writes
    /// `Phase::BASELINE` to `Ctx::current_step` at scenario start
    /// (before any Step's `current_step.store` advance), so any
    /// capture taken before the first Step transition stamps with
    /// this value.
    pub const BASELINE: Self = Self(0);

    /// Construct a `Phase` for the `zero_indexed`-th scenario Step.
    /// The 1-indexed encoding (Step 0 → `u16` 1, Step 1 → `u16` 2,
    /// ...) keeps `BASELINE` unambiguous at `u16` 0. Saturates at
    /// `u16::MAX` rather than overflowing — a scenario with > 65k
    /// Steps is pathological and the saturating value still
    /// distinguishes "well past any real Step" from BASELINE.
    pub const fn step(zero_indexed: u16) -> Self {
        Self(zero_indexed.saturating_add(1))
    }

    /// True iff this is `Phase::BASELINE` (the pre-first-Step
    /// settle window).
    pub const fn is_baseline(&self) -> bool {
        self.0 == 0
    }

    /// Inner `u16`. Use this when handing the value to a
    /// serializer / formatter / external consumer that does not
    /// understand the newtype. Production callers that build a
    /// `Phase` for downstream comparison should prefer
    /// `Phase::BASELINE` / `Phase::step(k)` over wrapping a raw
    /// `u16` themselves.
    pub const fn as_u16(self) -> u16 {
        self.0
    }
}

impl std::fmt::Display for Phase {
    /// `"BASELINE"` for [`Phase::BASELINE`], `"Step[k]"` for
    /// [`Phase::step`] (decoded back via the 1-indexed
    /// encoding). Matches the labels [`PhaseBucket`] embeds in
    /// `label` so operators see consistent phase identifiers
    /// across structured-sidecar reads and ad-hoc `format!`
    /// output.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_baseline() {
            write!(f, "BASELINE")
        } else {
            write!(f, "Step[{}]", self.0 - 1)
        }
    }
}

impl From<u16> for Phase {
    /// Wrap a raw 1-indexed encoded value as a [`Phase`]. Production
    /// paths that already have the encoded value (e.g. drained from
    /// the host-side mirror of `current_step`, or read out of a
    /// deserialized sidecar) construct the typed wrapper via this
    /// conversion without re-deriving the encoding.
    fn from(value: u16) -> Self {
        Self(value)
    }
}

impl From<Phase> for u16 {
    fn from(value: Phase) -> Self {
        value.0
    }
}

/// Per-phase metric bucket — one entry per scenario phase in
/// [`ScenarioStats::phases`].
///
/// A scenario with N Steps yields `N + 1` phases: phase 0 is the
/// BASELINE (pre-first-Step settle window), and phases 1..=N
/// correspond to Step 0..Step N-1 in scenario order. The
/// 1-indexed Step encoding (instead of 0-indexed) lets BASELINE
/// own `step_index = 0` unambiguously — a `step_index = 0` sample
/// is always settle, not first-Step.
///
/// Each bucket carries the metric values reduced over the phase's
/// sample window. For `crate::stats::MetricKind::Counter`
/// metrics the reduction is `last - first` across the phase's
/// periodic samples (cumulative-counter delta); for `Gauge` /
/// `Peak` / `Timestamp` it dispatches per the kind via
/// `crate::stats::aggregate_samples`. Missing metric keys mean
/// the phase had no finite samples for that metric.
///
/// Metric keys match `crate::stats::MetricDef::name` — see
/// `crate::stats::METRICS` for the canonical list of registered
/// metric names a `get` / `phase_metric` lookup expects.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize, crate::Claim)]
pub struct PhaseBucket {
    /// Phase index. `0` = BASELINE (pre-first-Step settle window).
    /// `1..=N` align with Step ordinals (1-indexed): Step 0 of the
    /// scenario lives at `step_index = 1`, Step 1 at
    /// `step_index = 2`, etc. The encoding avoids the collision
    /// where a 0-indexed Step would share `step_index = 0` with
    /// the BASELINE settle window.
    pub step_index: u16,
    /// Human-readable label. `"BASELINE"` for `step_index = 0`,
    /// `"Step[0]"` / `"Step[1]"` / ... for `step_index = 1..=N`.
    /// Mirrors the formatting used by
    /// `crate::timeline::Timeline`'s phase rendering so operator
    /// inspection of the formatted diagnostic and the structured
    /// sidecar yield the same phase identifiers.
    pub label: String,
    /// Phase window start: the MINIMUM per-sample time anchor in the
    /// phase — each sample's `boundary_offset_ms`, falling back to its
    /// `elapsed_ms`. Samples with neither anchor (both `None` — a
    /// not-measured timestamp) are excluded from the min.
    pub start_ms: u64,
    /// Phase window end: the MAXIMUM per-sample time anchor in the
    /// phase (the same `boundary_offset_ms`-or-`elapsed_ms` key as
    /// `start_ms`). A phase whose every sample is unanchored yields the
    /// inverted window `(start_ms = u64::MAX, end_ms = 0)`, which folds
    /// no monitor samples. Downstream renderers should not assume the
    /// value is closed against a stimulus event.
    pub end_ms: u64,
    /// Number of periodic samples bucketed into this phase. Zero
    /// when the phase fired no captures: BASELINE when the settle window
    /// was shorter than the periodic interval, OR a synthesized
    /// capture-free interior step (the
    /// `build_phase_buckets_with_stimulus` seam — a `StepStart`-step
    /// whose window held no periodic boundary still gets a bucket so its
    /// capture-independent `iteration_rate` is not dropped).
    pub sample_count: usize,
    /// Per-metric phase-aggregated values. See the [`PhaseBucket`]
    /// struct doc for the registry key source and per-kind reduction
    /// dispatch; missing keys mean the phase carried no finite
    /// samples for that metric (sentinel-free: `None` from the
    /// reducer surfaces as "key absent" rather than "value 0.0").
    pub metrics: std::collections::BTreeMap<String, f64>,
    /// Per-cgroup raw telemetry components for this phase, keyed by cgroup
    /// name (see [`PhaseCgroupStats`]). Empty until a capture path populates
    /// it; the structural carrier for the per-phase per-cgroup distributional
    /// re-pool. Whole-run = aggregate of these per-phase per-cgroup components.
    ///
    /// An ORPHAN bucket — a guest carrier whose `step_index` has NO paired host
    /// bucket (a dropped/absent StepStart frame, or a stimulus-less host/fixture
    /// path; NOT merely a short step, since `build_phase_buckets_with_stimulus`
    /// synthesizes a bucket for every StepStart so a captured-but-short step
    /// takes the matched arm) — is carried by
    /// `fold_guest_per_cgroup_into_host_buckets` with the shape
    /// `(start_ms, end_ms) == (0, 0)` AND empty `metrics` AND non-empty
    /// `per_cgroup` (it carries only these components). On every non-zero-duration
    /// window that shape is the orphan arm's, so the timeline render keys on it
    /// to surface "window not measured" rather than a misleading `0ms` (see
    /// `crate::timeline::phase_from_bucket`): a captured bucket has metrics. A
    /// zero-duration step at scenario start (`StepStart==StepEnd==0`) can also
    /// produce it via the matched arm, but harmlessly — a zero-duration step has
    /// no window, so "not measured" reads the same as "0ms".
    pub per_cgroup: std::collections::BTreeMap<String, PhaseCgroupStats>,
}

impl PhaseBucket {
    /// Look up the phase-aggregated value for `metric_name` (see
    /// [`PhaseBucket::metrics`] for the registry source). Returns
    /// `None` when the phase carried no finite samples for that
    /// metric — distinct from `Some(0.0)` which means the reducer
    /// produced a real zero from finite samples.
    pub fn get(&self, metric_name: &str) -> Option<f64> {
        self.metrics.get(metric_name).copied()
    }

    /// Like [`Self::get`], but panics with a diagnostic message citing
    /// the bucket's `step_index` + `label` + `sample_count` + the set
    /// of metric keys actually present when the metric is absent. Use
    /// when the caller knows the metric MUST be in the bucket (the
    /// phase fired samples and the metric is registered — see
    /// [`PhaseBucket::metrics`]) — the panic message tells the operator whether the cause is
    /// "phase produced no samples" (sample_count of 0) or "metric key
    /// typo" (positive sample_count but the key isn't in `metrics`).
    ///
    /// ```ignore
    /// let bucket = r.stats.step(0).expect("Step[0] phase");
    /// let throughput = bucket.expect_metric("throughput");
    /// ```
    pub fn expect_metric(&self, metric_name: &str) -> f64 {
        self.get(metric_name).unwrap_or_else(|| {
            panic!(
                "PhaseBucket::expect_metric: metric '{}' absent from phase \
                 step_index={} ('{}') with sample_count={}. \
                 metric keys present in this bucket: {:?}. \
                 Possible causes: (a) phase carried 0 samples for this \
                 metric (sample_count==0 means no captures landed in the \
                 phase at all; sample_count>0 means captures landed but \
                 the metric extracted no finite values from them); \
                 (b) metric name typo (verify against \
                 ScenarioStats::is_known_metric / known_metrics).",
                metric_name,
                self.step_index,
                self.label,
                self.sample_count,
                self.metrics.keys().collect::<Vec<_>>(),
            )
        })
    }

    /// Cross-cgroup phase total for a per-cgroup Counter metric that lives
    /// in [`Self::per_cgroup`] but never in [`Self::metrics`] — currently
    /// `"total_migrations"`, `"total_iterations"`, and `"total_cpu_time_ns"`.
    ///
    /// These are registered `MetricKind::Counter`s
    /// (`crate::stats::METRICS`) whose per-sample source is absent
    /// (`crate::stats::MetricDef::read_sample` returns `None` — they are
    /// per-task guest counters not captured per tick), so
    /// [`crate::assert::build_phase_buckets`] never folds them into
    /// `metrics`; only the per-cgroup carrier ([`PhaseCgroupStats`], built
    /// by `phase_cgroup_stats`) holds them. This sums them across the
    /// phase's `per_cgroup` carriers — the SAME cross-cgroup Counter sum
    /// [`ScenarioStats::total_migrations`] takes run-level and
    /// `merge_matched_phase_buckets` takes per key — so the value is the
    /// phase total, not a per-cgroup fragment.
    ///
    /// `None` when the phase has no `per_cgroup` carriers (NOT measured —
    /// distinct from a measured `Some(0.0)` when carriers exist but counted
    /// zero) or `name` is not one of the per-cgroup-sourced counters.
    /// Surfacing them here never double-sources the run-level `ext_metrics`,
    /// but by two different mechanisms: `total_migrations` / `total_iterations`
    /// are in `TYPED_FIELD_NAMES` (`populate_run_ext_metrics_from_phases` skips
    /// them; the typed `GauntletRow` accessor stays the single run-level
    /// authority), while `total_cpu_time_ns` is NOT in `TYPED_FIELD_NAMES` —
    /// it is safe because it is never written into any pooled `metrics` map
    /// (its `MetricDef` accessor and `read_sample` both return `None`, so no
    /// run-level fold ever picks it up; it lives only on the per-cgroup
    /// carrier).
    ///
    /// Counterpart to [`Self::get`] (which reads `metrics` only).
    /// [`crate::vmm::VmResult::phase_metric`] falls back to this so a
    /// `post_vm` callback reading `phase_metric(phase, "total_migrations")`
    /// gets the value instead of a silent `None`.
    pub fn cgroup_counter_total(&self, name: &str) -> Option<f64> {
        let field: fn(&PhaseCgroupStats) -> u64 = match name {
            "total_migrations" => |c| c.total_migrations,
            "total_iterations" => |c| c.total_iterations,
            "total_cpu_time_ns" => |c| c.total_cpu_time_ns,
            _ => return None,
        };
        if self.per_cgroup.is_empty() {
            return None;
        }
        Some(self.per_cgroup.values().map(field).sum::<u64>() as f64)
    }
}

/// Merge two [`PhaseBucket`]s sharing the same `step_index` per
/// the per-MetricKind dispatch in [`crate::stats::MergeKind`].
/// Called by [`AssertResult::merge`] for matched buckets;
/// unmatched buckets are appended verbatim by the caller.
///
/// Window-invariant merge:
/// - `step_index`: equal by precondition (caller pairs buckets by
///   `step_index`), kept from `a`.
/// - `label`: kept from `a`. By construction the label is derived
///   purely from `step_index` (`"BASELINE"` / `"Step[k]"`) so both
///   sides agree.
/// - `start_ms`: `min(a.start_ms, b.start_ms)` so the merged
///   window covers the earliest start of either side.
/// - `end_ms`: `max(a.end_ms, b.end_ms)` so the merged window
///   covers the latest end. Drives the [`crate::stats::MergeKind::NonCommutative`]
///   tiebreak on Gauge(Last) / Timestamp metrics — the value
///   from the bucket whose `end_ms` is later wins.
/// - `sample_count`: `a + b`. Used as the weighting denominator
///   for the `MetricKind::Gauge(GaugeAgg::Avg)` weighted mean.
///
/// Per-metric merge dispatches on the metric's `crate::stats::MetricKind`
/// from the registry via [`crate::stats::metric_def`]:
/// - `MetricKind::Counter` → `a + b` (the two reduced values are
///   per-phase deltas; the merge across cgroups sums per-cgroup
///   contributions to the phase delta, mirroring how
///   `ScenarioStats::total_migrations` adds across cgroups).
/// - `MetricKind::Peak` and `MetricKind::Gauge(GaugeAgg::Max)` →
///   `max(a, b)` (the worst-case "peak that fired" survives).
/// - `MetricKind::Gauge(GaugeAgg::Avg)` → weighted mean
///   `(a * a_w + b * b_w) / (a_w + b_w)` where `a_w = a_count.max(1)`
///   and `b_w = b_count.max(1)` — the unbiased combination of both
///   sides' per-phase means weighted by sample population, each weight
///   floored at 1. The `.max(1)` floor (mirroring
///   `populate_run_ext_metrics_from_phases`) keeps a synthesized
///   zero-capture bucket's capture-independent Gauge(Avg) value
///   contributing one phase-observation of weight rather than being
///   zero-weighted out of
///   the merge — the silent-drop the synthesize seam exists to prevent.
///   With both counts > 0 the floor is a no-op (the plain
///   sample-population weighting); both counts zero degenerates to
///   `(a + b) / 2.0`.
/// - `MetricKind::Gauge(GaugeAgg::Last)` and `MetricKind::Timestamp`
///   → value from the bucket with the larger `end_ms`; ties keep
///   `a`'s value. Captures the "latest-sample-wins" semantic per
///   the [`crate::stats::MergeKind::NonCommutative`] contract.
/// - `MetricKind::Rate { .. }` → SKIPPED in the per-key fold and
///   re-derived from the pooled components by
///   [`crate::stats::derive_rate_metrics`] as a post-pass, so the
///   merged rate is `Σnumerator / Σdenominator` (each component
///   folds by its own kind first) rather than a fold of two
///   ready-made per-phase ratios.
///
/// Unregistered metric names (not in `crate::stats::METRICS`)
/// fall back to a commutative arithmetic mean
/// `(a + b) / 2.0`. The mean is the safest default for an unknown
/// kind: sum would over-count Gauge / Timestamp values, max would
/// lose Counter / Avg signal, and "last" requires a tiebreak the
/// caller can't compute without the kind. Producers attaching
/// unregistered metrics to a `PhaseBucket` should add them to
/// `METRICS` to get the typed merge instead of the fallback.
pub(crate) fn merge_matched_phase_buckets(a: PhaseBucket, b: PhaseBucket) -> PhaseBucket {
    assert_eq!(
        a.step_index, b.step_index,
        "merge_matched_phase_buckets: caller must pair by step_index",
    );
    let mut metrics = std::collections::BTreeMap::new();
    // Collect every key present on either side; iterate once,
    // dispatching per the kind of the key (or the unregistered
    // mean fallback) so the merge is single-pass.
    let mut keys: std::collections::BTreeSet<&String> = a.metrics.keys().collect();
    keys.extend(b.metrics.keys());
    for key in keys {
        // Derived metrics (Rate / Distribution / WorstLowest) are NOT merged
        // here: a Rate re-derives from the merged components in the post-pass
        // below, and Distribution / WorstLowest are re-pooled run-level by
        // `populate_run_distribution_metrics` (they never appear in
        // phase.metrics — `aggregate_samples_for_phase` returns None — so this
        // skip is also a structural guard). Folding a ready-made derived value
        // would lose the re-pool.
        if crate::stats::metric_def(key).is_some_and(|m| m.kind.is_derived()) {
            continue;
        }
        let av = a.metrics.get(key).copied();
        let bv = b.metrics.get(key).copied();
        let merged = match (av, bv) {
            (Some(av), Some(bv)) => {
                let kind = crate::stats::metric_def(key).map(|m| m.kind);
                merge_metric_values(
                    kind,
                    av,
                    bv,
                    a.sample_count,
                    b.sample_count,
                    a.end_ms,
                    b.end_ms,
                )
            }
            (Some(v), None) | (None, Some(v)) => v,
            (None, None) => continue,
        };
        metrics.insert(key.clone(), merged);
    }
    // Re-derive Rate metrics from the now-pooled components: each
    // component merged by its own kind above (a Counter numerator
    // summed), so the rate becomes Σnumerator / Σdenominator — the
    // correct re-pool, not a mean of the two phases' ready-made ratios.
    crate::stats::derive_rate_metrics(&mut metrics);
    // Union per_cgroup by cgroup name: a cgroup present on both sides folds
    // its raw components per PhaseCgroupStats::merge (concat samples, sum
    // counters, combine extremes); a cgroup on only one side is carried
    // verbatim. Empty ∪ empty = empty, so this is a no-op until a capture
    // path populates per_cgroup (the structural-carrier invariant).
    let mut per_cgroup = a.per_cgroup;
    for (name, b_cg) in b.per_cgroup {
        match per_cgroup.remove(&name) {
            Some(a_cg) => {
                per_cgroup.insert(name, PhaseCgroupStats::merge(a_cg, b_cg));
            }
            None => {
                per_cgroup.insert(name, b_cg);
            }
        }
    }
    PhaseBucket {
        step_index: a.step_index,
        label: a.label,
        start_ms: a.start_ms.min(b.start_ms),
        end_ms: a.end_ms.max(b.end_ms),
        sample_count: a.sample_count + b.sample_count,
        metrics,
        per_cgroup,
    }
}

/// Fold the guest-collected per-phase `per_cgroup` carriers into the
/// host-rebuilt phase buckets, keyed by `step_index`.
///
/// The host rebuilds phase buckets from the periodic-capture series
/// (window + metric folds), but those buckets carry an empty `per_cgroup`
/// by construction. The guest collects per-cgroup RAW components per step
/// ([`crate::scenario::collect_handles`] under `collect_step`) into carrier
/// buckets whose only payload is `per_cgroup` — a merge-neutral
/// `(u64::MAX, 0)` window and empty `metrics`. Guest and host `step_index`
/// are the SAME 1-indexed value: the step loop stamps
/// `phase_step_index = step_idx + 1` onto BOTH the `StepStart` frames the
/// host rebuilds buckets from AND the `collect_step` carrier, so pairing by
/// `step_index` is exact and cannot drift.
///
/// Each guest carrier whose `step_index` matches a host bucket folds its
/// `per_cgroup` in via [`merge_matched_phase_buckets`] — a no-op on the
/// host's window (`min`/`max` against `MAX`/`0`), metrics (the carrier has
/// none, so each host key is carried verbatim), and `sample_count` (`+ 0`),
/// contributing ONLY the unioned `per_cgroup`. A guest `step_index` with no
/// host bucket — a DEFENSIVE case: the carrier's `step_index` has no `StepStart`
/// frame in the host stimulus timeline (a dropped/absent stimulus frame, or a
/// stimulus-less host/fixture path), since `build_phase_buckets_with_stimulus`
/// SYNTHESIZES a capture-free bucket for every StepStart-step, so a
/// captured-but-short step takes the matched arm above, not this one — is carried
/// verbatim with its window normalized to `(0, 0)` so duration consumers
/// (`end_ms - start_ms`) never underflow the merge-neutral sentinel — no
/// `per_cgroup` datum is silently dropped. With no guest carriers (a run
/// with no step-local cgroups) the host buckets pass through unchanged. The
/// returned vec is sorted by `step_index`.
pub(crate) fn fold_guest_per_cgroup_into_host_buckets(
    host_buckets: Vec<PhaseBucket>,
    guest_buckets: Vec<PhaseBucket>,
) -> Vec<PhaseBucket> {
    let host_len = host_buckets.len();
    // No-silent-drops: host buckets have unique step_index
    // (build_phase_buckets_with_stimulus emits one bucket per step_index), but
    // fold same-step_index duplicates via merge rather than a last-wins collect so
    // a future producer that violated the invariant DEGRADES to a merge, never a
    // silent release-mode drop. The debug_assert still trips loudly in test/debug.
    let mut by_idx: std::collections::BTreeMap<u16, PhaseBucket> =
        std::collections::BTreeMap::new();
    for b in host_buckets {
        match by_idx.remove(&b.step_index) {
            Some(existing) => {
                by_idx.insert(b.step_index, merge_matched_phase_buckets(existing, b));
            }
            None => {
                by_idx.insert(b.step_index, b);
            }
        }
    }
    debug_assert_eq!(
        by_idx.len(),
        host_len,
        "host buckets must have unique step_index; a collision merged (not dropped)",
    );
    for gb in guest_buckets {
        // Every guest carrier MUST carry the merge-neutral (u64::MAX, 0) sentinel
        // window (the step_per_cgroup_bucket invariant). Validate it BEFORE the
        // match so BOTH arms are guarded: the matched arm relies on the window
        // being merge-neutral (min/max no-op against the host window), and the
        // orphan arm normalizes it to (0,0). A future caller handing a
        // real-window carrier (incl. a duplicate orphan via the matched arm)
        // trips loudly instead of silently corrupting the merged window.
        debug_assert!(
            gb.start_ms == u64::MAX && gb.end_ms == 0,
            "guest carrier must carry the merge-neutral (u64::MAX, 0) window; got ({}, {})",
            gb.start_ms,
            gb.end_ms,
        );
        match by_idx.remove(&gb.step_index) {
            Some(hb) => {
                by_idx.insert(gb.step_index, merge_matched_phase_buckets(hb, gb));
            }
            None => {
                // Orphan arm: a guest carrier whose step_index has no host bucket.
                //
                // Invariant: build_phase_buckets_with_stimulus synthesizes a host
                // bucket for every StepStart-step, so a carrier whose step has a
                // StepStart frame always takes the matched arm above. This arm is
                // reached only by a carrier whose step has NO StepStart frame —
                // defensive, not produced by normal capture.
                //
                // Normalize the merge-neutral sentinel window to (0,0) so duration
                // consumers don't underflow it. The resulting (0,0)-window +
                // empty-metrics + non-empty-per_cgroup shape is the orphan
                // signature the timeline render keys on to show "window not
                // measured" instead of a misleading 0ms — the (0,0) means "no host
                // window known", NOT a measured zero-duration step.
                //
                // A zero-duration step at scenario start (StepStart==StepEnd==0)
                // produces the same shape via the matched arm, but harmlessly: a
                // zero-duration step has no window, so the render's "not measured"
                // reads the same as "0ms". See `crate::timeline::phase_from_bucket`.
                let mut orphan = gb;
                orphan.start_ms = 0;
                orphan.end_ms = 0;
                by_idx.insert(orphan.step_index, orphan);
            }
        }
    }
    by_idx.into_values().collect()
}

/// Per-metric merge inner helper used by
/// [`merge_matched_phase_buckets`]. Dispatches on the metric's
/// `crate::stats::MetricKind` (or the unregistered fallback)
/// to combine two reduced values into one.
///
/// `a_count` / `b_count` are the source buckets' `sample_count`
/// fields, used as weights for `Gauge(Avg)`. `a_end_ms` /
/// `b_end_ms` are the source buckets' window-end timestamps,
/// used to pick the later sample for `Gauge(Last)` / `Timestamp`.
fn merge_metric_values(
    kind: Option<crate::stats::MetricKind>,
    a: f64,
    b: f64,
    a_count: usize,
    b_count: usize,
    a_end_ms: u64,
    b_end_ms: u64,
) -> f64 {
    use crate::stats::{GaugeAgg, MetricKind};
    match kind {
        // Counter (cumulative) and DeltaSum (sum of per-read deltas)
        // both merge across AssertResults by summing the reduced values
        // (commutative — see MetricKind::merge_kind).
        Some(MetricKind::Counter) | Some(MetricKind::DeltaSum) => a + b,
        Some(MetricKind::Peak) | Some(MetricKind::Gauge(GaugeAgg::Max)) => a.max(b),
        Some(MetricKind::Gauge(GaugeAgg::Avg)) => {
            // Weight by sample_count, floored at 1: a sample_count==0
            // bucket carrying a capture-independent Gauge(Avg) value must
            // contribute one phase-observation of weight, not be
            // zero-weighted out of the merge. Mirrors the .max(1) floor in
            // populate_run_ext_metrics_from_phases. With both counts > 0
            // the floor is a no-op (the prior sample_count weighting);
            // with both 0 each still floors to weight 1, giving the
            // (a+b)/2 equal-weight mean (the aggregate_samples_weighted
            // zero-total-weight fallback is unreachable from here).
            // (iteration_rate — the original synthesized zero-capture case —
            // is now a MetricKind::Rate: merge_matched_phase_buckets skips it
            // via the Rate `continue` in its key loop (above this fn) and
            // re-pools it from its summed Counter components, so a Rate value
            // never reaches this Gauge(Avg) fold.)
            let a_w = a_count.max(1) as f64;
            let b_w = b_count.max(1) as f64;
            (a * a_w + b * b_w) / (a_w + b_w)
        }
        Some(MetricKind::Gauge(GaugeAgg::Last)) | Some(MetricKind::Timestamp) => {
            if b_end_ms > a_end_ms { b } else { a }
        }
        // Derived kinds (Rate / Distribution / WorstLowest) are skipped in
        // the merge loop (see `merge_matched_phase_buckets`'s `is_derived`
        // continue) and produced post-merge (`derive_rate_metrics` /
        // `populate_run_distribution_metrics`), so a derived value never
        // reaches this per-value merge — folding a ready-made derived value
        // would lose the re-pool.
        Some(MetricKind::Rate { .. })
        | Some(MetricKind::Distribution { .. })
        | Some(MetricKind::WorstLowest { .. })
        | Some(MetricKind::WakeLatencyTailRatio)
        | Some(MetricKind::PerPhase) => unreachable!(
            "derived metrics (Rate/Distribution/WorstLowest/WakeLatencyTailRatio/PerPhase) are produced post-merge, not merged as values"
        ),
        // Unregistered metric: commutative mean fallback. Sum
        // would over-count Gauge values; max would lose Counter
        // signal; "last" needs a tiebreak the caller can't
        // compute without the kind. Mean is the safest commutative
        // default.
        None => (a + b) / 2.0,
    }
}
