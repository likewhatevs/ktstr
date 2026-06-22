//! Unit tests for [`PhaseBucket`] and the [`ScenarioStats`]
//! per-phase accessor surface. Verifies the
//! `Op::ReadKernel*`-style discoverability path
//! (`phase` / `phase_metric`) plus the serde round-trip every new
//! pub serialized type needs.

use super::tests_common::rpt;
use super::{
    CgroupStats, PhaseBucket, PhaseCgroupStats, ScenarioStats, cgroup_stats,
    expand_backdrop_phase_buckets, fold_guest_per_cgroup_into_host_buckets, percentile,
    phase_cgroup_stats, phase_slice_to_cgroup_stats, pool_phase_slice_stats,
    populate_run_distribution_metrics, populate_run_distribution_metrics_from,
    reduce_sorted_distribution, reduce_weighted_sorted_distribution, step_per_cgroup_bucket,
    weighted_percentile,
};
use crate::monitor::dump::{FailureDumpReport, SCHEMA_SINGLE};
use crate::scenario::sample::SampleSeries;
use crate::scenario::snapshot::{DrainedSnapshotEntry, MissingStatsReason};
use crate::stats::{MetricKind, SampleReduction, SampleSource};
use crate::workload::WorkerReport;
use std::collections::{BTreeMap, BTreeSet};

/// Carries no BPF state — `MetricDef::read_sample` returns `None`
/// for every metric on this report, so the resulting
/// `PhaseBucket.metrics` map is empty. The test exercises the
/// bucketing shape, not the metric extraction.
fn fixture_report() -> FailureDumpReport {
    FailureDumpReport {
        schema: SCHEMA_SINGLE.to_string(),
        ..Default::default()
    }
}

/// Build a synthetic `DrainedSnapshotEntry` with the given
/// `step_index` stamp and `elapsed_ms` anchor.
fn fixture_entry(tag: &str, step_index: u16, elapsed_ms: u64) -> DrainedSnapshotEntry {
    DrainedSnapshotEntry {
        tag: tag.to_string(),
        report: fixture_report(),
        stats: Err(MissingStatsReason::NoSchedulerBinary),
        elapsed_ms: Some(elapsed_ms),
        boundary_offset_ms: None,
        step_index: Some(step_index),
    }
}

/// Helper: a `ScenarioStats` whose single phase carries `per_cgroup` keyed by
/// name, plus optional run-level `cgroups` for the WorstLowest path / the
/// stripped-run Distribution fallback.
fn repool_stats(
    carriers: Vec<(&str, PhaseCgroupStats)>,
    cgroups: Vec<CgroupStats>,
) -> ScenarioStats {
    let mut bucket = PhaseBucket::default();
    for (name, pcg) in carriers {
        bucket.per_cgroup.insert(name.to_string(), pcg);
    }
    ScenarioStats {
        phases: vec![bucket],
        cgroups,
        ..ScenarioStats::default()
    }
}

mod backdrop;
mod build_pipeline;
mod display;
mod phase_guard_a;
mod phase_guard_b;
mod repool;
mod serde_tests;
mod step_local;
mod weighted;
