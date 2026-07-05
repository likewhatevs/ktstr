use super::*;

/// Unified assertion configuration. Carries both worker checks and
/// monitor thresholds as a single composable type. Each `Option` field
/// acts as an override — `None` means "inherit from parent layer".
///
/// Construct via [`Assert::NO_OVERRIDES`] (preferred const baseline)
/// or [`Assert::default_checks`] (currently aliases NO_OVERRIDES);
/// chain builder methods on either base (all builders are `const fn`
/// except [`Assert::expect_scx_bpf_error_matches`], which compiles a
/// regex at construction). Use the resulting `Assert` value as the
/// `assert` field of a [`Scheduler`](crate::test_support::Scheduler)
/// declared via [`declare_scheduler!`](crate::declare_scheduler) — the
/// macro accepts `assert = Assert::NO_OVERRIDES.foo()`-style chains
/// at the scheduler level. The `#[ktstr_test]` proc macro does NOT
/// accept an `assert = …` attribute on test entries; per-field
/// attribute shortcuts (`max_gap_ms = N`, `not_starved = true`, …)
/// compose into the equivalent struct literal at expansion time.
///
/// Merge order: `Assert::default_checks()` -> `Scheduler.assert` -> per-test `assert`.
/// `default_checks()` is `NO_OVERRIDES` — all assertions are opt-in.
///
/// ```
/// # use ktstr::assert::Assert;
/// // Scheduler opts into imbalance checking.
/// let sched_assert = Assert::NO_OVERRIDES.max_imbalance_ratio(5.0);
///
/// // Merge: defaults <- scheduler <- test.
/// let merged = Assert::default_checks()
///     .merge(&sched_assert)
///     .merge(&Assert::NO_OVERRIDES.max_gap_ms(5000));
///
/// assert_eq!(merged.not_starved, None);              // not opted in
/// assert_eq!(merged.max_imbalance_ratio, Some(5.0)); // from sched
/// assert_eq!(merged.max_gap_ms, Some(5000));         // from test
/// ```
///
/// # Serde roundtrip — covers the 20 threshold/check fields only
///
/// The serde derive at the struct level covers the 20 threshold +
/// flag fields (every `Option<bool/u64/f64/u32/usize>` plus the bare
/// `enforce_monitor_thresholds: bool`). The two reproducer-matcher
/// fields ([`Self::expect_scx_bpf_error_contains`] and
/// [`Self::expect_scx_bpf_error_matches`]) carry `#[serde(skip)]`
/// because their `&'static str` shape cannot round-trip through a
/// borrowed deserializer — see each field's doc for the rationale.
/// Sidecar consumers comparing
/// threshold config across runs treat reproducer matcher strings as
/// part of the test identity (encoded by name in the sidecar key),
/// not part of the threshold payload, so the skip is operationally
/// transparent today.
#[must_use = "builder methods return a new Assert; discard means config is lost"]
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct Assert {
    // Worker checks
    /// Enable starvation, fairness spread, and gap checks across
    /// worker reports. `Some(true)` enables, `Some(false)` explicitly
    /// disables (overriding any enabling merge from a lower layer),
    /// `None` inherits from the merge parent.
    pub not_starved: Option<bool>,
    /// Enable per-worker CPU isolation checks (ensure workers remain
    /// within their assigned cpuset). Same tri-state semantics as
    /// `not_starved`.
    pub isolation: Option<bool>,
    /// Max per-worker scheduling gap in milliseconds. Fails the
    /// assertion if any worker's longest off-CPU stretch exceeds this.
    pub max_gap_ms: Option<u64>,
    /// Max per-cgroup fairness spread as a percentage. Fails if the
    /// range between the most- and least-served workers exceeds this
    /// fraction of their mean.
    pub max_spread_pct: Option<f64>,

    // Throughput checks
    /// Max coefficient of variation for work_units/cpu_time across workers.
    /// Catches placement unfairness where some workers get less CPU than others.
    pub max_throughput_cv: Option<f64>,
    /// Minimum work_units per CPU-second. Catches cases where all workers
    /// are equally slow (CV passes but absolute throughput is too low).
    pub min_work_rate: Option<f64>,

    // Benchmarking checks
    /// Max p99 wake latency in NANOSECONDS. Fails if the pooled
    /// p99 across every worker's `wake_latencies_ns` exceeds this.
    ///
    /// # Unit-name gotcha
    ///
    /// The threshold is `_ns`, but the paired reporting field on
    /// [`CgroupStats::p99_wake_latency_us`] and the re-pooled run-level
    /// `worst_p99_wake_latency_us` ext metric (a `MetricKind::Distribution`)
    /// are MICROSECONDS. The two surfaces are intentionally split:
    ///   - the threshold uses NS for precision (typical scheduler
    ///     wake latencies are single-digit µs, so sub-µs resolution
    ///     matters for regression gates);
    ///   - the reporting fields use US for readability in
    ///     `perf-delta` / dashboard output.
    ///
    /// Both are computed from the same underlying
    /// [`WorkerReport::wake_latencies_ns`] samples — see
    /// [`assert_benchmarks`] for the threshold path and
    /// [`assert_not_starved`] for the reporting path. A bare
    /// comparison of `max_p99_wake_latency_ns` against
    /// `CgroupStats::p99_wake_latency_us` is a unit-mismatch bug;
    /// `assert_benchmarks` never does this — it consumes the raw
    /// `wake_latencies_ns` directly — and
    /// `assert_p99_ns_threshold_compares_against_ns_latencies` pins
    /// that contract.
    pub max_p99_wake_latency_ns: Option<u64>,
    /// Max wake latency coefficient of variation. Fails if CV exceeds this.
    pub max_wake_latency_cv: Option<f64>,
    /// Minimum iterations per wall-clock second. Fails if any worker is below.
    pub min_iteration_rate: Option<f64>,
    /// Max migration ratio (migrations/iterations). Fails if any cgroup exceeds this.
    pub max_migration_ratio: Option<f64>,

    // Monitor checks
    /// Max `nr_running` / LLC imbalance ratio observed by the monitor.
    /// Fails if the worst sample's imbalance exceeds this.
    pub max_imbalance_ratio: Option<f64>,
    /// Max local DSQ depth observed by the monitor. Fails if any
    /// sampled CPU's local DSQ grew beyond this.
    pub max_local_dsq_depth: Option<u32>,
    /// Treat an rq_clock-stuck verdict from the monitor as a hard failure. Same
    /// tri-state semantics as `not_starved`.
    pub fail_on_rq_clock_stuck: Option<bool>,
    /// Minimum number of consecutive samples that must exceed the
    /// monitor threshold before a verdict is raised. Smooths out
    /// single-sample spikes.
    pub sustained_samples: Option<usize>,
    /// Max `select_cpu_fallback` rate (events/sec). Fails if the
    /// scx event counter delta over the run exceeds this rate.
    pub max_fallback_rate: Option<f64>,
    /// Max `keep_last` rate (events/sec). Fails if the scx event
    /// counter delta over the run exceeds this rate.
    pub max_keep_last_rate: Option<f64>,
    /// Promote monitor threshold violations from report-only to
    /// pass/fail. When `false` (the default), the monitor still walks
    /// every sample and records every violation in the verdict's
    /// `details`, but the verdict's `passed` stays `true`. Tests that
    /// want monitor violations to fail the run call
    /// [`Self::with_monitor_defaults`], which populates each monitor
    /// threshold from `MonitorThresholds::new()`
    /// and sets this flag to `true`.
    pub enforce_monitor_thresholds: bool,

    // NUMA checks
    /// Minimum fraction of pages on the expected NUMA node(s) (0.0-1.0).
    /// Expected nodes are derived from the worker's
    /// [`MemPolicy`](crate::workload::MemPolicy) at evaluation time.
    /// Fails if the observed locality fraction falls below this.
    pub min_page_locality: Option<f64>,
    /// Maximum ratio of NUMA-node-migrated pages to total allocated
    /// pages (0.0-1.0). Distinct from [`max_migration_ratio`](Self::max_migration_ratio)
    /// which measures CPU migrations per iteration. Fails if the
    /// observed migration ratio exceeds this.
    pub max_cross_node_migration_ratio: Option<f64>,
    /// Maximum fraction of pages on slow-tier (memory-only) NUMA nodes
    /// (0.0-1.0). For CXL memory tiering tests: fails if more than
    /// this fraction of pages land on memory-only nodes. Requires
    /// `slow_tier_nodes` to be set at evaluation time.
    pub max_slow_tier_ratio: Option<f64>,

    /// Reproducer-mode literal-substring matcher for the captured
    /// `scx_bpf_error` text. When `Some(literal)`, the eval pipeline
    /// scans the combined scheduler log + sched_ext dump for a
    /// substring match against `literal` and fails the test if the
    /// substring is absent.
    ///
    /// Use this for the common case of pinning an exact error
    /// fragment like `apply_cell_config returned -EINVAL` without
    /// having to escape regex metacharacters. For pattern matching
    /// with anchors / character classes / wildcards, use
    /// [`Self::expect_scx_bpf_error_matches`] instead — the two
    /// fields are orthogonal and can both be set (both must match).
    ///
    /// Requires the entry's `expect_err = true` — a reproducer
    /// matcher only fires on expected-error tests; setting this on
    /// a passing test would assert "the test passed AND contained
    /// this error text," which is contradictory. The eval-time
    /// validation rejects that combination with a clear diagnostic.
    ///
    /// Stored as `&'static str` so the const-fn `Self::merge`
    /// composes without cloning. Empty strings are rejected at
    /// evaluation (an empty literal would silently match every
    /// message and turn this assertion into a no-op).
    /// `#[serde(skip)]` because the field's `&'static str` shape
    /// cannot round-trip through a borrowed deserializer (no source-
    /// string lifetime to bind to). Reproducer matcher strings are
    /// test-author static literals carried in the test definition
    /// itself, not per-run data the sidecar needs to roundtrip — so
    /// skipping them on the wire keeps the JSON shape clean without
    /// losing any sidecar-consumer functionality. Skipped fields
    /// default to `None` on deserialize per `Option::default()`.
    ///
    /// The `Option<Cow<'static, str>>` alternative that WOULD
    /// roundtrip is rejected because it cascade-breaks
    /// `Scheduler::assert`'s const-fn (`Cow` has a heap destructor,
    /// which breaks the const-fn assignment path used by
    /// `declare_scheduler!` macro statics). A future decomposition
    /// into a `ReproducerMatchers` sub-config could revisit this if
    /// sidecar-loaded test definitions ever need the strings to
    /// survive end-to-end.
    #[serde(skip)]
    pub expect_scx_bpf_error_contains: Option<&'static str>,

    /// Reproducer-mode regex matcher for the captured `scx_bpf_error`
    /// text. When `Some(pattern)`, the eval pipeline compiles the
    /// pattern via the `regex` crate, scans the combined scheduler
    /// log + sched_ext dump, and fails the test if the regex does
    /// not match anywhere in the corpus.
    ///
    /// The pattern is a full regex — special characters
    /// (`. * + ? ( ) [ ] { } | ^ $ \`) carry their regex meaning.
    /// For literal-substring matching, prefer
    /// [`Self::expect_scx_bpf_error_contains`] to avoid escape
    /// footguns. The captured corpus is the multi-line concatenation
    /// of the scheduler log and `--- sched_ext dump ---`; the regex
    /// crate's default flags apply: `^` and `$` anchor to the start /
    /// end of the WHOLE corpus (not individual lines), and `.` does
    /// NOT cross `\n`. Opt in to line-level anchoring with `(?m)`
    /// (e.g. `(?m)^apply_cell_config$`) and to newline-spanning
    /// `.` with `(?s)`. A bare `apply_cell_config` matches the
    /// token anywhere in the corpus.
    ///
    /// Requires the entry's `expect_err = true` — same rationale
    /// as [`Self::expect_scx_bpf_error_contains`]. Patterns are
    /// validated at construction: empty literals, invalid regex
    /// syntax, and any pattern satisfying `is_match("")` all
    /// panic via the [`Self::expect_scx_bpf_error_matches`]
    /// builder. The `is_match("")` predicate catches two
    /// no-op classes with one check: patterns that match every
    /// position (e.g. `a?`, `.*`, `(?:)`) trivially pass against
    /// any corpus, and patterns that match only the empty string
    /// (e.g. `^$`) trivially fail against any non-empty corpus —
    /// real captured scheduler-output corpora are non-empty, so
    /// both classes are equally no-op pins. Bare `\b` (word
    /// boundary) slips the gate because the empty string
    /// contains no word characters; see the builder docstring
    /// for the operator-direction.
    /// `#[serde(skip)]` for the same reason as
    /// [`Self::expect_scx_bpf_error_contains`] above: `&'static str`
    /// doesn't roundtrip + the matcher pattern is test-definition
    /// data, not sidecar-roundtrip data. Skipped fields default to
    /// `None` on deserialize.
    #[serde(skip)]
    pub expect_scx_bpf_error_matches: Option<&'static str>,
}

impl Assert {
    /// Human-readable multi-line dump of every threshold field. Each
    /// field renders as `  name: value` (`none` when the option is
    /// `None`, i.e. inherited or unset). Used by
    /// `cargo ktstr show-thresholds <test>` to expose the resolved
    /// merged `Assert` (`default_checks().merge(&entry.scheduler.assert).
    /// merge(&entry.assert)`) without forcing the operator to read
    /// the Debug impl or source. Output is a sequence of indented
    /// `row` lines ending with a newline; the caller owns any
    /// outer section header (the `show-thresholds` CLI already
    /// prints `Test: ...` / `Scheduler: ...` / `Resolved assertion
    /// thresholds:` lines above the threshold block, and
    /// `format_human` emits no outer header of its own so it does
    /// not duplicate that banner).
    pub fn format_human(&self) -> String {
        use std::fmt::Write;
        let mut out = String::new();
        fn row<T: std::fmt::Display>(out: &mut String, name: &str, v: &Option<T>) {
            match v {
                Some(x) => writeln!(out, "  {name:<38}: {x}").unwrap(),
                None => writeln!(out, "  {name:<38}: none").unwrap(),
            }
        }
        row(&mut out, "not_starved", &self.not_starved);
        row(&mut out, "isolation", &self.isolation);
        row(&mut out, "max_gap_ms", &self.max_gap_ms);
        row(&mut out, "max_spread_pct", &self.max_spread_pct);
        row(&mut out, "max_throughput_cv", &self.max_throughput_cv);
        row(&mut out, "min_work_rate", &self.min_work_rate);
        row(
            &mut out,
            "max_p99_wake_latency_ns",
            &self.max_p99_wake_latency_ns,
        );
        row(&mut out, "max_wake_latency_cv", &self.max_wake_latency_cv);
        row(&mut out, "min_iteration_rate", &self.min_iteration_rate);
        row(&mut out, "max_migration_ratio", &self.max_migration_ratio);
        row(&mut out, "max_imbalance_ratio", &self.max_imbalance_ratio);
        row(&mut out, "max_local_dsq_depth", &self.max_local_dsq_depth);
        row(
            &mut out,
            "fail_on_rq_clock_stuck",
            &self.fail_on_rq_clock_stuck,
        );
        row(&mut out, "sustained_samples", &self.sustained_samples);
        row(&mut out, "max_fallback_rate", &self.max_fallback_rate);
        row(&mut out, "max_keep_last_rate", &self.max_keep_last_rate);
        row(&mut out, "min_page_locality", &self.min_page_locality);
        row(
            &mut out,
            "max_cross_node_migration_ratio",
            &self.max_cross_node_migration_ratio,
        );
        row(&mut out, "max_slow_tier_ratio", &self.max_slow_tier_ratio);
        row(
            &mut out,
            "expect_scx_bpf_error_contains",
            &self.expect_scx_bpf_error_contains,
        );
        row(
            &mut out,
            "expect_scx_bpf_error_matches",
            &self.expect_scx_bpf_error_matches,
        );
        out
    }

    /// Identity element for [`Assert::merge`]: every Option field is
    /// `None` and `enforce_monitor_thresholds` is `false`, so neither
    /// side of a merge with `NO_OVERRIDES` is altered.
    /// Identical to the value returned by [`Self::default_checks`] —
    /// the const is for spread-into-struct-literal composition, the
    /// const fn is the method-style entry point.
    pub const NO_OVERRIDES: Assert = Assert {
        not_starved: None,
        isolation: None,
        max_gap_ms: None,
        max_spread_pct: None,
        max_throughput_cv: None,
        min_work_rate: None,
        max_p99_wake_latency_ns: None,
        max_wake_latency_cv: None,
        min_iteration_rate: None,
        max_migration_ratio: None,
        max_imbalance_ratio: None,
        max_local_dsq_depth: None,
        fail_on_rq_clock_stuck: None,
        sustained_samples: None,
        max_fallback_rate: None,
        max_keep_last_rate: None,
        enforce_monitor_thresholds: false,
        min_page_locality: None,
        max_cross_node_migration_ratio: None,
        max_slow_tier_ratio: None,
        expect_scx_bpf_error_contains: None,
        expect_scx_bpf_error_matches: None,
    };

    /// Baseline of the runtime merge chain
    /// `default_checks().merge(&scheduler.assert).merge(&entry.assert)`.
    ///
    /// All checks are off by default — tests opt in to the assertions
    /// they care about via scheduler-level or per-test `Assert`
    /// overrides.
    ///
    /// For spread-into-struct-literal composition
    /// (`Assert { not_starved: Some(true), ..Assert::NO_OVERRIDES }`)
    /// use the equivalent const [`Self::NO_OVERRIDES`]; this const fn
    /// is the method-style entry point that pairs with `.verdict()`
    /// and the builder setters.
    pub const fn default_checks() -> Assert {
        Self::NO_OVERRIDES
    }

    /// Build a fresh [`Verdict`] under this `Assert`'s threshold
    /// config. The returned accumulator carries no claim records; call
    /// the typed `claim_<field>` methods generated by
    /// [`#[derive(Claim)]`](ktstr_macros::Claim) on stats structs as
    /// `stats.claim_<field>(&mut verdict)`, or use the
    /// [`claim!`](crate::claim) macro on a local/expression, then
    /// call [`Verdict::into_result`] to produce the final
    /// [`AssertResult`].
    ///
    /// This is the entry point of the pointwise-claim API. The
    /// `Assert` itself remains pure threshold config and stays
    /// `Copy`; per-test claims accumulate on the returned `Verdict`,
    /// which owns its own buffers (details, stats).
    ///
    /// ```
    /// # use ktstr::assert::Assert;
    /// let r = Assert::default_checks().verdict().into_result();
    /// assert!(r.is_pass(), "no claims means passing verdict");
    /// ```
    pub fn verdict(self) -> Verdict {
        Verdict::with_assert(self)
    }

    pub const fn check_not_starved(mut self) -> Self {
        self.not_starved = Some(true);
        self
    }

    pub const fn check_isolation(mut self) -> Self {
        self.isolation = Some(true);
        self
    }

    pub const fn max_gap_ms(mut self, ms: u64) -> Self {
        self.max_gap_ms = Some(ms);
        self
    }

    pub const fn max_spread_pct(mut self, pct: f64) -> Self {
        self.max_spread_pct = Some(pct);
        self
    }

    pub const fn max_throughput_cv(mut self, v: f64) -> Self {
        self.max_throughput_cv = Some(v);
        self
    }

    pub const fn min_work_rate(mut self, v: f64) -> Self {
        self.min_work_rate = Some(v);
        self
    }

    pub const fn max_p99_wake_latency_ns(mut self, v: u64) -> Self {
        self.max_p99_wake_latency_ns = Some(v);
        self
    }

    pub const fn max_wake_latency_cv(mut self, v: f64) -> Self {
        self.max_wake_latency_cv = Some(v);
        self
    }

    pub const fn min_iteration_rate(mut self, v: f64) -> Self {
        self.min_iteration_rate = Some(v);
        self
    }

    pub const fn max_migration_ratio(mut self, v: f64) -> Self {
        self.max_migration_ratio = Some(v);
        self
    }

    pub const fn max_imbalance_ratio(mut self, v: f64) -> Self {
        self.max_imbalance_ratio = Some(v);
        self
    }

    pub const fn max_local_dsq_depth(mut self, v: u32) -> Self {
        self.max_local_dsq_depth = Some(v);
        self
    }

    /// Control whether a monitor rq_clock-stuck verdict fails the assertion.
    pub const fn fail_on_rq_clock_stuck(mut self, v: bool) -> Self {
        self.fail_on_rq_clock_stuck = Some(v);
        self
    }

    /// Set the number of consecutive over-threshold samples required
    /// before the monitor raises a verdict.
    pub const fn sustained_samples(mut self, v: usize) -> Self {
        self.sustained_samples = Some(v);
        self
    }

    pub const fn max_fallback_rate(mut self, v: f64) -> Self {
        self.max_fallback_rate = Some(v);
        self
    }

    pub const fn max_keep_last_rate(mut self, v: f64) -> Self {
        self.max_keep_last_rate = Some(v);
        self
    }

    pub const fn min_page_locality(mut self, v: f64) -> Self {
        self.min_page_locality = Some(v);
        self
    }

    pub const fn max_cross_node_migration_ratio(mut self, v: f64) -> Self {
        self.max_cross_node_migration_ratio = Some(v);
        self
    }

    pub const fn max_slow_tier_ratio(mut self, v: f64) -> Self {
        self.max_slow_tier_ratio = Some(v);
        self
    }

    /// True when any worker-level check field is `Some`. A pure query on
    /// the configured checks — it no longer gates per-cgroup telemetry
    /// (telemetry is built unconditionally per handle in
    /// [`crate::scenario`]'s collect path; worker-check assertions are the
    /// only opt-in part). Retained as public API for callers composing an
    /// `Assert` who need to know whether any worker check is set.
    pub const fn has_worker_checks(&self) -> bool {
        self.not_starved.is_some()
            || self.isolation.is_some()
            || self.max_gap_ms.is_some()
            || self.max_spread_pct.is_some()
            || self.max_throughput_cv.is_some()
            || self.min_work_rate.is_some()
            || self.max_p99_wake_latency_ns.is_some()
            || self.max_wake_latency_cv.is_some()
            || self.min_iteration_rate.is_some()
            || self.max_migration_ratio.is_some()
            || self.min_page_locality.is_some()
            || self.max_cross_node_migration_ratio.is_some()
            || self.max_slow_tier_ratio.is_some()
    }

    /// Merge `other` on top of `self`. Each `Some` field in `other`
    /// overrides the corresponding field in `self`; `None` fields
    /// inherit from `self`.
    ///
    /// [`Assert::NO_OVERRIDES`] is the two-sided identity:
    /// `x.merge(&NO_OVERRIDES)` and `NO_OVERRIDES.merge(&x)` both yield
    /// `x`. The runtime composes scheduler- and test-level overrides as
    /// `Assert::default_checks().merge(&scheduler.assert).merge(&test.assert)`,
    /// so a `NO_OVERRIDES` at either override layer leaves the defaults
    /// untouched -- which means "no override," not "no checks."
    pub const fn merge(&self, other: &Assert) -> Assert {
        // `Option::or` is not yet const-stable, so each field expands
        // a match rather than calling `other.x.or(self.x)`. Keep it
        // this way until `const fn` can call `Option::or`; at that
        // point the 21 match blocks collapse to 21 `.or()` calls.
        Assert {
            not_starved: match other.not_starved {
                Some(v) => Some(v),
                None => self.not_starved,
            },
            isolation: match other.isolation {
                Some(v) => Some(v),
                None => self.isolation,
            },
            max_gap_ms: match other.max_gap_ms {
                Some(v) => Some(v),
                None => self.max_gap_ms,
            },
            max_spread_pct: match other.max_spread_pct {
                Some(v) => Some(v),
                None => self.max_spread_pct,
            },
            max_throughput_cv: match other.max_throughput_cv {
                Some(v) => Some(v),
                None => self.max_throughput_cv,
            },
            min_work_rate: match other.min_work_rate {
                Some(v) => Some(v),
                None => self.min_work_rate,
            },
            max_p99_wake_latency_ns: match other.max_p99_wake_latency_ns {
                Some(v) => Some(v),
                None => self.max_p99_wake_latency_ns,
            },
            max_wake_latency_cv: match other.max_wake_latency_cv {
                Some(v) => Some(v),
                None => self.max_wake_latency_cv,
            },
            min_iteration_rate: match other.min_iteration_rate {
                Some(v) => Some(v),
                None => self.min_iteration_rate,
            },
            max_migration_ratio: match other.max_migration_ratio {
                Some(v) => Some(v),
                None => self.max_migration_ratio,
            },
            max_imbalance_ratio: match other.max_imbalance_ratio {
                Some(v) => Some(v),
                None => self.max_imbalance_ratio,
            },
            max_local_dsq_depth: match other.max_local_dsq_depth {
                Some(v) => Some(v),
                None => self.max_local_dsq_depth,
            },
            fail_on_rq_clock_stuck: match other.fail_on_rq_clock_stuck {
                Some(v) => Some(v),
                None => self.fail_on_rq_clock_stuck,
            },
            sustained_samples: match other.sustained_samples {
                Some(v) => Some(v),
                None => self.sustained_samples,
            },
            max_fallback_rate: match other.max_fallback_rate {
                Some(v) => Some(v),
                None => self.max_fallback_rate,
            },
            max_keep_last_rate: match other.max_keep_last_rate {
                Some(v) => Some(v),
                None => self.max_keep_last_rate,
            },
            enforce_monitor_thresholds: self.enforce_monitor_thresholds
                || other.enforce_monitor_thresholds,
            min_page_locality: match other.min_page_locality {
                Some(v) => Some(v),
                None => self.min_page_locality,
            },
            max_cross_node_migration_ratio: match other.max_cross_node_migration_ratio {
                Some(v) => Some(v),
                None => self.max_cross_node_migration_ratio,
            },
            max_slow_tier_ratio: match other.max_slow_tier_ratio {
                Some(v) => Some(v),
                None => self.max_slow_tier_ratio,
            },
            expect_scx_bpf_error_contains: match other.expect_scx_bpf_error_contains {
                Some(v) => Some(v),
                None => self.expect_scx_bpf_error_contains,
            },
            expect_scx_bpf_error_matches: match other.expect_scx_bpf_error_matches {
                Some(v) => Some(v),
                None => self.expect_scx_bpf_error_matches,
            },
        }
    }

    /// Extract an `AssertPlan` for worker-side checks.
    pub(crate) fn worker_plan(&self) -> AssertPlan {
        AssertPlan {
            not_starved: self.not_starved.unwrap_or(false),
            isolation: self.isolation.unwrap_or(false),
            max_gap_ms: self.max_gap_ms,
            max_spread_pct: self.max_spread_pct,
            max_throughput_cv: self.max_throughput_cv,
            min_work_rate: self.min_work_rate,
            max_p99_wake_latency_ns: self.max_p99_wake_latency_ns,
            max_wake_latency_cv: self.max_wake_latency_cv,
            min_iteration_rate: self.min_iteration_rate,
            max_migration_ratio: self.max_migration_ratio,
            min_page_locality: self.min_page_locality,
            max_cross_node_migration_ratio: self.max_cross_node_migration_ratio,
            max_slow_tier_ratio: self.max_slow_tier_ratio,
        }
    }

    /// Run the configured worker checks against one cgroup's reports.
    ///
    /// `cpuset` is the CPU set for isolation checks. `numa_nodes` is
    /// the NUMA node IDs covered by the cpuset (for page locality and
    /// slow-tier checks). Derive via
    /// [`TestTopology::numa_nodes_for_cpuset`](crate::topology::TestTopology::numa_nodes_for_cpuset).
    pub fn assert_cgroup(
        &self,
        reports: &[crate::workload::WorkerReport],
        cpuset: Option<&BTreeSet<usize>>,
    ) -> AssertResult {
        self.worker_plan().assert_cgroup(reports, cpuset, None)
    }

    /// Run worker checks with explicit NUMA node set for page locality.
    pub fn assert_cgroup_with_numa(
        &self,
        reports: &[crate::workload::WorkerReport],
        cpuset: Option<&BTreeSet<usize>>,
        numa_nodes: Option<&BTreeSet<usize>>,
    ) -> AssertResult {
        self.worker_plan()
            .assert_cgroup(reports, cpuset, numa_nodes)
    }

    /// Run NUMA page locality check.
    ///
    /// `observed` is the fraction of pages on expected nodes (0.0-1.0).
    /// `total_pages` and `local_pages` are for diagnostics.
    pub fn assert_page_locality(
        &self,
        observed: f64,
        total_pages: u64,
        local_pages: u64,
    ) -> AssertResult {
        assert_page_locality(observed, self.min_page_locality, total_pages, local_pages)
    }

    /// Run cross-node migration ratio check.
    ///
    /// `migrated_pages` is the `/proc/vmstat` `numa_pages_migrated` delta.
    /// `total_pages` is total allocated pages from numa_maps.
    pub fn assert_cross_node_migration(
        &self,
        migrated_pages: u64,
        total_pages: u64,
    ) -> AssertResult {
        assert_cross_node_migration(
            migrated_pages,
            total_pages,
            self.max_cross_node_migration_ratio,
        )
    }

    /// Extract `MonitorThresholds` for monitor-side evaluation.
    pub(crate) fn has_monitor_thresholds(&self) -> bool {
        self.max_imbalance_ratio.is_some()
            || self.max_local_dsq_depth.is_some()
            || self.fail_on_rq_clock_stuck.is_some()
            || self.sustained_samples.is_some()
            || self.max_fallback_rate.is_some()
            || self.max_keep_last_rate.is_some()
    }

    pub(crate) fn monitor_thresholds(&self) -> crate::monitor::MonitorThresholds {
        use crate::monitor::MonitorThresholds;
        let d = MonitorThresholds::new();
        MonitorThresholds {
            max_imbalance_ratio: self.max_imbalance_ratio.unwrap_or(d.max_imbalance_ratio),
            max_local_dsq_depth: self.max_local_dsq_depth.unwrap_or(d.max_local_dsq_depth),
            fail_on_rq_clock_stuck: self
                .fail_on_rq_clock_stuck
                .unwrap_or(d.fail_on_rq_clock_stuck),
            sustained_samples: self.sustained_samples.unwrap_or(d.sustained_samples),
            max_fallback_rate: self.max_fallback_rate.unwrap_or(d.max_fallback_rate),
            max_keep_last_rate: self.max_keep_last_rate.unwrap_or(d.max_keep_last_rate),
            enforce: self.enforce_monitor_thresholds,
        }
    }

    /// Opt into pass/fail enforcement for monitor thresholds. Without
    /// this call, monitor violations are reported in the verdict's
    /// `details` but do not fail the test. With it, any monitor
    /// threshold violation fails the test.
    ///
    /// Also populates any unset monitor-threshold field with the
    /// canonical default from `MonitorThresholds::new()`
    /// — so a test that only cares about `max_keep_last_rate` can chain
    /// `.max_keep_last_rate(N).with_monitor_defaults()` and get the
    /// other five enforced at their canonical defaults.
    pub const fn with_monitor_defaults(mut self) -> Self {
        use crate::monitor::MonitorThresholds;
        let d = MonitorThresholds::new();
        if self.max_imbalance_ratio.is_none() {
            self.max_imbalance_ratio = Some(d.max_imbalance_ratio);
        }
        if self.max_local_dsq_depth.is_none() {
            self.max_local_dsq_depth = Some(d.max_local_dsq_depth);
        }
        if self.fail_on_rq_clock_stuck.is_none() {
            self.fail_on_rq_clock_stuck = Some(d.fail_on_rq_clock_stuck);
        }
        if self.sustained_samples.is_none() {
            self.sustained_samples = Some(d.sustained_samples);
        }
        if self.max_fallback_rate.is_none() {
            self.max_fallback_rate = Some(d.max_fallback_rate);
        }
        if self.max_keep_last_rate.is_none() {
            self.max_keep_last_rate = Some(d.max_keep_last_rate);
        }
        self.enforce_monitor_thresholds = true;
        self
    }

    /// Const-fn builder for [`Self::expect_scx_bpf_error_contains`].
    /// Chains with the other const-fn setters so a scheduler-def or
    /// per-test assertion block can compose
    /// `Assert::NO_OVERRIDES.expect_scx_bpf_error_contains(...).check_not_starved()`.
    ///
    /// Empty strings panic at construction (an empty literal would
    /// silently match every message and turn this assertion into a
    /// no-op); pass a non-empty fragment that should appear in the
    /// expected `scx_bpf_error` message.
    ///
    /// # Panics
    /// When `literal` is empty.
    #[must_use = "builder methods consume self; bind the result"]
    pub const fn expect_scx_bpf_error_contains(mut self, literal: &'static str) -> Self {
        assert!(
            !literal.is_empty(),
            "Assert::expect_scx_bpf_error_contains: literal must be non-empty",
        );
        self.expect_scx_bpf_error_contains = Some(literal);
        self
    }

    /// Builder for [`Self::expect_scx_bpf_error_matches`]. The
    /// pattern is a regex; special characters retain their regex
    /// meaning. For literal-substring matching, prefer
    /// [`Self::expect_scx_bpf_error_contains`] to avoid escape
    /// footguns.
    ///
    /// Validates the pattern at construction: rejects empty
    /// patterns, rejects invalid regex syntax, and rejects any
    /// pattern that satisfies `is_match("")`. The empty-string
    /// match predicate catches two related no-op classes:
    /// patterns that match every position (e.g. `a?`, `.*`,
    /// `(?:)`) trivially pass against any corpus, and patterns
    /// that match only the empty string (e.g. `^$`) trivially
    /// fail against any non-empty corpus — every real captured
    /// scheduler-output corpus is non-empty, so the latter is
    /// equally a no-op pin in practice. Both are useless;
    /// `is_match("")` catches both with one check.
    ///
    /// Bare `\b` (word boundary) slips this gate because the
    /// empty string contains no word characters, so `\b` finds
    /// no transition and `is_match("")` returns false; yet `\b`
    /// matches the first word boundary in any real corpus,
    /// turning a bare-`\b` pin into a vacuous "any non-empty
    /// log passes" assertion. Use a substring of the expected
    /// error text rather than a standalone boundary assertion.
    /// All other documented assertions (`\A`, `\z`, `^`, `$`,
    /// `\B`) match the empty string at position 0 and ARE
    /// caught by the gate.
    ///
    /// Unlike the sibling [`Self::expect_scx_bpf_error_contains`]
    /// (which is `const fn`), this builder is non-const because
    /// the construction-time regex compilation requires heap
    /// allocation. Callers needing a const builder for a regex
    /// matcher must build the `Assert` via struct literal —
    /// the evaluator's defense-in-depth catches invalid syntax
    /// reached via that bypass at first evaluation, but the
    /// vacuous-pattern gate only fires on the builder path.
    ///
    /// # Panics
    /// When `pattern` is empty, is invalid regex syntax, or
    /// matches the empty string.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn expect_scx_bpf_error_matches(mut self, pattern: &'static str) -> Self {
        assert!(
            !pattern.is_empty(),
            "Assert::expect_scx_bpf_error_matches: pattern must be non-empty",
        );
        let compiled = regex::Regex::new(pattern).unwrap_or_else(|e| {
            panic!(
                "Assert::expect_scx_bpf_error_matches: pattern {pattern:?} is not valid regex: {e}",
            )
        });
        assert!(
            !compiled.is_match(""),
            "Assert::expect_scx_bpf_error_matches: pattern {pattern:?} matches the empty \
             string (e.g. `a?`, `.*`, `(?:)`, `^$`); such patterns vacuously match any \
             corpus and turn the matcher into a no-op — use a meaningful pattern that \
             requires at least one character",
        );
        self.expect_scx_bpf_error_matches = Some(pattern);
        self
    }

    /// Evaluate the reproducer-mode `scx_bpf_error` matchers against
    /// the captured text corpus. Returns an empty Vec when no matcher
    /// is configured or when every configured matcher matches; returns
    /// a non-empty Vec of `AssertDetail` entries on failure.
    ///
    /// Each configured matcher contributes at most one detail. Both
    /// fields can be set simultaneously (`AND` semantics — both must
    /// match).
    ///
    /// Preconditions enforced by the evaluator:
    /// 1. `expect_err = true` must be set when either matcher is
    ///    configured. Setting the matcher on a passing-test contract
    ///    is a misuse — surfaced with an `expect_err = true` reminder
    ///    diagnostic.
    /// 2. The regex pattern in
    ///    [`Self::expect_scx_bpf_error_matches`] must compile via
    ///    the `regex` crate. Invalid syntax surfaces as a diagnostic
    ///    naming the pattern and the compile error.
    ///
    /// `captured_text` is the concatenation of the raw scheduler-log
    /// stream (the bulk-port merged `SchedLog` frames, or the test
    /// process `output` fallback when no frames arrived) and the
    /// `--- sched_ext dump ---` extract — both surfaces where
    /// `scx_bpf_error` printk lands. The matcher sees the WHOLE
    /// stream, not the marker-extracted section; lines outside the
    /// `SCHED_OUTPUT_START..SCHED_OUTPUT_END` markers are included.
    pub fn evaluate_scx_bpf_error_match(
        &self,
        captured_text: &str,
        expect_err: bool,
    ) -> Vec<AssertDetail> {
        let mut details = Vec::new();
        if self.expect_scx_bpf_error_contains.is_none()
            && self.expect_scx_bpf_error_matches.is_none()
        {
            return details;
        }
        if !expect_err {
            details.push(AssertDetail::new(
                DetailKind::Other,
                "expect_scx_bpf_error_contains or expect_scx_bpf_error_matches \
                 requires expect_err = true on the test entry — the matcher narrows \
                 which failure counts as the expected bug, and only applies to \
                 expected-error tests; set #[ktstr_test(expect_err = true, ...)] or \
                 drop the matcher",
            ));
            return details;
        }
        // Truncate at a UTF-8 char boundary at or below 400 bytes so
        // the excerpt length stays within the "up to 400 bytes follow:"
        // budget. `chars().take(400)` would count codepoints instead,
        // and a multi-byte corpus would exceed the byte budget — the
        // boundary walk steps back to the nearest char start.
        let excerpt = || -> &str {
            let len = captured_text.len();
            if len <= 400 {
                captured_text
            } else {
                let mut end = 400;
                while end > 0 && !captured_text.is_char_boundary(end) {
                    end -= 1;
                }
                &captured_text[..end]
            }
        };
        if let Some(literal) = self.expect_scx_bpf_error_contains
            && !captured_text.contains(literal)
        {
            details.push(AssertDetail::new(
                DetailKind::Other,
                format!(
                    "expect_scx_bpf_error_contains({literal:?}): substring not found \
                     in the scheduler log + sched_ext dump corpus (the expected bug \
                     did not fire, or its message text changed). Captured corpus \
                     {} bytes; up to 400 bytes follow:\n{}",
                    captured_text.len(),
                    excerpt(),
                ),
            ));
        }
        if let Some(pattern) = self.expect_scx_bpf_error_matches {
            match regex::Regex::new(pattern) {
                Ok(re) => {
                    if !re.is_match(captured_text) {
                        details.push(AssertDetail::new(
                            DetailKind::Other,
                            format!(
                                "expect_scx_bpf_error_matches({pattern:?}): regex did \
                                 not match the scheduler log + sched_ext dump corpus \
                                 (the expected bug did not fire, or its message text \
                                 changed). Captured corpus {} bytes; up to 400 bytes \
                                 follow:\n{}",
                                captured_text.len(),
                                excerpt(),
                            ),
                        ));
                    }
                }
                Err(e) => {
                    details.push(AssertDetail::new(
                        DetailKind::Other,
                        format!(
                            "expect_scx_bpf_error_matches({pattern:?}): regex \
                             compilation failed: {e}. Fix the pattern at the test \
                             declaration site — the matcher cannot evaluate against an \
                             invalid pattern",
                        ),
                    ));
                }
            }
        }
        details
    }
}
