//! Typed metric identifiers — a discoverable, typo-proof handle for the built-in
//! metric registry ([`super::METRICS`]), plus a first-class string escape for
//! scheduler-runtime / payload-supplied keys.
//!
//! [`BuiltinMetric`] is one variant per [`super::METRICS`] entry: the entire
//! built-in metric vocabulary in one readable, completable enum. A misspelled
//! built-in metric is a COMPILE error, not a silent runtime `None` — closing the
//! silent-wrong-answer class where a typo'd metric name yields `None` and a test
//! vacuously passes having asserted nothing.
//!
//! [`MetricId`] unifies the typed built-in path with the open scheduler-runtime
//! string path behind one `impl Into<MetricId>` accessor argument:
//! `phase_metric(BuiltinMetric::TaobenchTotalQps)` and
//! `phase_metric("scx_custom_runtime_key")` both compile through the same
//! signature — the known one typed and discoverable, the dynamic one a first-class
//! string escape (existing `&str` call sites keep compiling, canonicalized to the
//! `Builtin` variant when the string names a registered metric).
//!
//! GUARDRAIL: a `MetricId::Dynamic` whose string is NOT a registered name has no
//! declared [`super::MetricKind`] ([`MetricId::def`] returns `None`); callers must
//! never aggregate such a metric as a guessed kind (the math-protocol silent-lie
//! class). Resolve-by-name or fold conservatively — never assume `Counter`.
//!
//! The 1:1 correspondence with [`super::METRICS`] is pinned by
//! [`tests::builtin_metric_is_one_to_one_with_registry`] (both directions), so a
//! registry entry without an enum variant — or an enum variant without a registry
//! entry — fails the build.

use std::borrow::Cow;

use super::{MetricDef, metric_def};

/// Define [`BuiltinMetric`] from the single `(variant => wire_name)` source of
/// truth, generating the enum, `wire_name()`, `from_wire_name()`, and `ALL` so the
/// three can never drift from each other (the registry-vs-enum drift is caught
/// separately by the pin-test).
macro_rules! builtin_metrics {
    ($($variant:ident => $wire:literal),* $(,)?) => {
        /// One variant per `super::METRICS` entry — the discoverable built-in
        /// metric vocabulary. [`Self::wire_name`] is the stable registry / sidecar
        /// / CI string; [`Self::def`] resolves the full `MetricDef`. Pinned 1:1
        /// to `METRICS`.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub enum BuiltinMetric { $($variant),* }

        impl BuiltinMetric {
            /// The stable registry / sidecar / CI wire name — the `&str` key into
            /// `super::METRICS` and the ext-metrics maps.
            pub const fn wire_name(self) -> &'static str {
                match self { $(Self::$variant => $wire,)* }
            }

            /// Classify a wire name back to its variant, or `None` for a
            /// non-built-in (scheduler-runtime / payload) key.
            pub fn from_wire_name(name: &str) -> Option<Self> {
                match name { $($wire => Some(Self::$variant),)* _ => None }
            }

            /// Every built-in metric (registry-order-independent). Drives typed
            /// all-metrics sweeps and the 1:1 pin-test.
            pub const ALL: &'static [BuiltinMetric] = &[$(Self::$variant),*];
        }
    };
}

builtin_metrics! {
    // Run-level typed / derived metrics (Gauge / Peak / Counter / Rate /
    // Distribution / WorstLowest / WakeLatencyTailRatio).
    WorstSpread => "worst_spread",
    WorstGapMs => "worst_gap_ms",
    TotalMigrations => "total_migrations",
    WorstMigrationRatio => "worst_migration_ratio",
    MaxImbalanceRatio => "max_imbalance_ratio",
    AvgImbalanceRatio => "avg_imbalance_ratio",
    MaxDsqDepth => "max_dsq_depth",
    AvgDsqDepth => "avg_dsq_depth",
    StuckCount => "stuck_count",
    TotalFallback => "total_fallback",
    TotalKeepLast => "total_keep_last",
    TotalRunDelay => "total_run_delay",
    TotalPcount => "total_pcount",
    TotalSchedCount => "total_sched_count",
    TotalYldCount => "total_yld_count",
    TotalSchedGoidle => "total_sched_goidle",
    TotalTtwuCount => "total_ttwu_count",
    TotalTtwuLocal => "total_ttwu_local",
    TotalRunDelayNsPerSched => "total_run_delay_ns_per_sched",
    TtwuLocalFraction => "ttwu_local_fraction",
    AvgNrRunning => "avg_nr_running",
    WorstP99WakeLatencyUs => "worst_p99_wake_latency_us",
    WorstMedianWakeLatencyUs => "worst_median_wake_latency_us",
    WorstWakeLatencyCv => "worst_wake_latency_cv",
    WorstP99TimerLatencyUs => "worst_p99_timer_latency_us",
    WorstMedianTimerLatencyUs => "worst_median_timer_latency_us",
    WorstP999TimerLatencyUs => "worst_p999_timer_latency_us",
    WorstTimerLatencyUs => "worst_timer_latency_us",
    IterationRate => "iteration_rate",
    TotalIterations => "total_iterations",
    TotalPhaseIterations => "total_phase_iterations",
    TotalPhaseDurationSec => "total_phase_duration_sec",
    TotalCpuTimeSec => "total_cpu_time_sec",
    TotalIterationsPooled => "total_iterations_pooled",
    IterationsPerCpuSec => "iterations_per_cpu_sec",
    SystemTimeNs => "system_time_ns",
    UserTimeNs => "user_time_ns",
    WorstMeanRunDelayUs => "worst_mean_run_delay_us",
    WorstRunDelayUs => "worst_run_delay_us",
    WorstWakeLatencyTailRatio => "worst_wake_latency_tail_ratio",
    WorstIterationsPerWorker => "worst_iterations_per_worker",
    WorstIterationsPerCpuSec => "worst_iterations_per_cpu_sec",
    WorstPageLocality => "worst_page_locality",
    WorstCrossNodeMigrationRatio => "worst_cross_node_migration_ratio",
    TotalCpuTimeNs => "total_cpu_time_ns",

    // Per-phase scalars (MetricKind::PerPhase) — schbench / taobench engine
    // metrics, surfaced via the per-phase PhaseBucket path (const-named).
    WakeupP50LatencyUs => "wakeup_p50_latency_us",
    WakeupP90LatencyUs => "wakeup_p90_latency_us",
    WakeupP99LatencyUs => "wakeup_p99_latency_us",
    WakeupP999LatencyUs => "wakeup_p999_latency_us",
    RequestP50LatencyUs => "request_p50_latency_us",
    RequestP90LatencyUs => "request_p90_latency_us",
    RequestP99LatencyUs => "request_p99_latency_us",
    RequestP999LatencyUs => "request_p999_latency_us",
    SchedDelayMsgUs => "sched_delay_msg_us",
    SchedDelayWorkerUs => "sched_delay_worker_us",
    SchbenchLoopCount => "schbench_loop_count",
    TaobenchTotalQps => "taobench_total_qps",
    TaobenchFastQps => "taobench_fast_qps",
    TaobenchSlowQps => "taobench_slow_qps",
    TaobenchHitRatio => "taobench_hit_ratio",
    TaobenchHitRate => "taobench_hit_rate",
    WakeupMinLatencyUs => "wakeup_min_latency_us",
    WakeupMaxLatencyUs => "wakeup_max_latency_us",
    RequestMinLatencyUs => "request_min_latency_us",
    RequestMaxLatencyUs => "request_max_latency_us",
    RpsP20 => "rps_p20",
    RpsP50 => "rps_p50",
    RpsP90 => "rps_p90",
    RpsMin => "rps_min",
    RpsMax => "rps_max",

    // Per-phase scalars inserted as bare literals by the non-schbench carrier
    // derivation (run_metrics.rs write_carrier_scalars) — the per-cgroup
    // wake/run-delay/off-cpu/migration/locality family.
    P99WakeLatencyUs => "p99_wake_latency_us",
    MedianWakeLatencyUs => "median_wake_latency_us",
    WakeLatencyCv => "wake_latency_cv",
    P99TimerLatencyUs => "p99_timer_latency_us",
    MedianTimerLatencyUs => "median_timer_latency_us",
    P999TimerLatencyUs => "p999_timer_latency_us",
    MeanRunDelayUs => "mean_run_delay_us",
    MaxRunDelayUs => "max_run_delay_us",
    AvgOffCpuPct => "avg_off_cpu_pct",
    MinOffCpuPct => "min_off_cpu_pct",
    MaxOffCpuPct => "max_off_cpu_pct",
    OffCpuSpreadPct => "off_cpu_spread_pct",
    MigrationRatio => "migration_ratio",
    IterationsPerWorker => "iterations_per_worker",
    PageLocality => "page_locality",
    CrossNodeMigrationRatio => "cross_node_migration_ratio",
}

impl BuiltinMetric {
    /// The full registry definition (polarity / kind / unit / tolerances /
    /// accessor). Total — the pin-test guarantees every variant resolves in
    /// `super::METRICS`.
    pub fn def(self) -> &'static MetricDef {
        metric_def(self.wire_name())
            .expect("BuiltinMetric variant must resolve in METRICS (pinned by test)")
    }
}

/// A metric identifier accepted by every metric accessor via `impl Into<MetricId>`:
/// a typed [`BuiltinMetric`] (the common, discoverable, typo-proof case) or a
/// dynamic scheduler-runtime / payload string (the open keyspace). One call shape
/// for both — `phase_metric(BuiltinMetric::X)` and `phase_metric("runtime_key")`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MetricId {
    /// A compile-time-known built-in metric (carries its registry kind via
    /// [`BuiltinMetric::def`]).
    Builtin(BuiltinMetric),
    /// A scheduler-runtime / payload-supplied key not in the built-in registry.
    Dynamic(Cow<'static, str>),
}

impl MetricId {
    /// The wire-name key — into `super::METRICS` and the ext-metrics maps. For
    /// a `Builtin` this is the inner [`BuiltinMetric::wire_name`]; for a `Dynamic`
    /// it is the raw key string.
    pub fn as_str(&self) -> &str {
        match self {
            MetricId::Builtin(b) => b.wire_name(),
            MetricId::Dynamic(s) => s.as_ref(),
        }
    }

    /// The registry definition if this id names a REGISTERED metric — `Some` for
    /// every `Builtin`; `None` for a genuine scheduler-runtime key. Via the `From`
    /// impls a registered name canonicalizes to `Builtin`, so a `Dynamic` reached
    /// through the public `impl Into<MetricId>` path never holds a registered name
    /// and resolves to `None` here; the `Dynamic` registry lookup is a defensive
    /// fallback for a directly-constructed `MetricId::Dynamic`. The GUARDRAIL: a
    /// `None` here means no declared kind, so the caller must not aggregate the
    /// value as a guessed kind (resolve-by-name or fold conservatively).
    pub fn def(&self) -> Option<&'static MetricDef> {
        match self {
            MetricId::Builtin(b) => Some(b.def()),
            MetricId::Dynamic(s) => metric_def(s.as_ref()),
        }
    }
}

impl From<BuiltinMetric> for MetricId {
    fn from(b: BuiltinMetric) -> Self {
        MetricId::Builtin(b)
    }
}

impl From<&BuiltinMetric> for MetricId {
    /// Borrowed-`BuiltinMetric` convenience — sweeping [`BuiltinMetric::ALL`]
    /// yields `&BuiltinMetric`; it is `Copy`, so this is a trivial copy-then-wrap.
    fn from(b: &BuiltinMetric) -> Self {
        MetricId::Builtin(*b)
    }
}

impl From<&str> for MetricId {
    /// Canonicalize a string: if it names a built-in, yield the typed `Builtin`
    /// (so a known reached by string still carries the typed identity + kind);
    /// otherwise `Dynamic`. A non-built-in `&str` is owned (the dynamic path is
    /// not hot; correctness over the micro-alloc).
    fn from(s: &str) -> Self {
        match BuiltinMetric::from_wire_name(s) {
            Some(b) => MetricId::Builtin(b),
            None => MetricId::Dynamic(Cow::Owned(s.to_owned())),
        }
    }
}

impl From<String> for MetricId {
    fn from(s: String) -> Self {
        match BuiltinMetric::from_wire_name(&s) {
            Some(b) => MetricId::Builtin(b),
            None => MetricId::Dynamic(Cow::Owned(s)),
        }
    }
}

impl From<&String> for MetricId {
    /// Borrowed-`String` convenience — iterating owned metric names (a
    /// `HashMap<String, _>`'s keys, a `Vec<String>`) yields `&String`;
    /// delegates to the `&str` canonicalization.
    fn from(s: &String) -> Self {
        Self::from(s.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stats::METRICS;

    /// The enum is pinned 1:1 to the registry, both directions, so adding a
    /// METRICS entry without a BuiltinMetric variant (or vice versa) fails here.
    #[test]
    fn builtin_metric_is_one_to_one_with_registry() {
        // (1) every variant resolves to a registry entry.
        for b in BuiltinMetric::ALL {
            assert!(
                metric_def(b.wire_name()).is_some(),
                "BuiltinMetric::{b:?} (wire {:?}) is not in METRICS",
                b.wire_name()
            );
        }
        // (2) every registry entry has a variant (no entry without a typed id).
        for m in METRICS {
            assert!(
                BuiltinMetric::from_wire_name(m.name).is_some(),
                "METRICS entry {:?} has no BuiltinMetric variant",
                m.name
            );
        }
        // (3) cardinality: catches a duplicate variant or a duplicate registry
        // name that (1)+(2) would individually miss.
        assert_eq!(
            BuiltinMetric::ALL.len(),
            METRICS.len(),
            "BuiltinMetric::ALL ({}) != METRICS ({})",
            BuiltinMetric::ALL.len(),
            METRICS.len()
        );
    }

    /// wire_name <-> from_wire_name roundtrips for every variant.
    #[test]
    fn wire_name_roundtrips() {
        for b in BuiltinMetric::ALL {
            assert_eq!(BuiltinMetric::from_wire_name(b.wire_name()), Some(*b));
        }
    }

    /// The MetricId hybrid: a built-in string canonicalizes to Builtin (typed +
    /// def() resolves); a non-built-in string stays Dynamic with def() == None
    /// (the no-guessed-kind guardrail).
    #[test]
    fn metric_id_canonicalizes_builtin_and_quarantines_dynamic() {
        let typed: MetricId = BuiltinMetric::TaobenchTotalQps.into();
        let from_str: MetricId = "taobench_total_qps".into();
        assert_eq!(
            typed, from_str,
            "a built-in string canonicalizes to Builtin"
        );
        assert_eq!(typed.as_str(), "taobench_total_qps");
        assert!(typed.def().is_some(), "a built-in carries its registry def");

        let dynamic: MetricId = "scx_layered_layer0_depth".into();
        assert!(
            matches!(dynamic, MetricId::Dynamic(_)),
            "a non-registered key stays Dynamic"
        );
        assert!(
            dynamic.def().is_none(),
            "a Dynamic key has no def() -> caller must not guess its kind"
        );
    }
}
