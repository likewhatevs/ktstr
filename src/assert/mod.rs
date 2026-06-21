//! Pass/fail evaluation of scenario results.
//!
//! Key types:
//! - [`AssertResult`] -- pass/fail status with diagnostics and statistics
//! - [`Assert`] -- composable assertion config (worker + monitor checks)
//! - [`ScenarioStats`] / [`CgroupStats`] -- aggregated telemetry
//! - [`NumaMapsEntry`] -- parsed `/proc/self/numa_maps` VMA entry
//! - [`Verdict`] -- pointwise-claim accumulator (built via
//!   [`Assert::verdict`] / [`Verdict::new`]; comparators routed through
//!   [`ClaimBuilder`] / [`SetClaim`] / [`SeqClaim`])
//!
//! NUMA assertion functions:
//! - [`parse_numa_maps`] -- parse numa_maps content into per-VMA entries
//! - [`page_locality`] -- compute page locality fraction from entries
//! - [`parse_vmstat_numa_pages_migrated`] -- extract vmstat migration counter
//! - [`assert_page_locality`] / [`assert_cross_node_migration`] -- threshold checks
//!
//! Assertion uses a three-layer merge: [`Assert::default_checks()`] ->
//! `Scheduler.assert` -> per-test `assert`.
//!
//! # Statistical conventions
//!
//! - **Percentiles / medians**: nearest-rank (see `percentile`),
//!   value at index `ceil(n * p) - 1`. Unlike interpolated
//!   percentiles, every reported p99 is an actual observed sample,
//!   not a synthetic midpoint. Consistent across every
//!   [`CgroupStats`] and [`ScenarioStats`] latency field.
//! - **CV (coefficient of variation)** is stddev/mean computed over
//!   the pooled latency samples, not as a mean of per-worker CVs —
//!   see [`CgroupStats::wake_latency_cv`] for the masking caveat.
//!
//! See the [Checking](https://likewhatevs.github.io/ktstr/guide/concepts/checking.html)
//! chapter of the guide.

use crate::workload::WorkerReport;
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};

mod builder;
mod detail;
mod phase_build;
mod plan;
mod reductions;
mod run_metrics;
mod stats_types;
mod types;

pub use builder::*;
pub use detail::*;
pub use phase_build::*;
pub use plan::*;
pub use reductions::*;
pub use run_metrics::*;
pub use stats_types::*;
pub use types::*;

pub mod claim;
pub mod temporal;

pub use claim::{ClaimBuilder, SeqClaim, SetClaim, Verdict};
pub use temporal::{EachClaim, FracPair, PhaseMapExt, SeriesField};

#[cfg(test)]
mod tests_assert;
#[cfg(test)]
mod tests_benchmarks;
#[cfg(test)]
mod tests_common;
#[cfg(test)]
mod tests_merge;
#[cfg(test)]
mod tests_note;
#[cfg(test)]
mod tests_numa;
#[cfg(test)]
mod tests_percentile;
#[cfg(test)]
mod tests_phase_bucket;
#[cfg(test)]
mod tests_plan;
#[cfg(test)]
mod tests_sched_died;
#[cfg(test)]
mod tests_serde;
#[cfg(test)]
mod tests_stats;
#[cfg(test)]
mod tests_verdict;
#[cfg(test)]
mod tests_worker;
