//! Gauntlet analysis and run-to-run comparison.
//!
//! Collects per-scenario results into hand-rolled aggregation passes
//! (group-by via `BTreeMap`, mean / std via plain iterators) for
//! statistical analysis, regression detection, and run-to-run compare
//! workflows. The earlier polars-backed implementation paid for a
//! columnar engine that was overkill for the dozens-to-low-thousands
//! of rows a gauntlet produces; the hand-rolled form gives the same
//! per-scenario means, outlier rankings, and dimension summaries
//! without polars's ~30-40 transitive crates.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

mod analyze;
mod compare;
mod group;
mod metric;
mod metric_id;
mod noise;
mod row;

pub use analyze::*;
pub use compare::*;
pub use group::*;
pub use metric::*;
pub use metric_id::*;
pub use noise::*;
pub use row::*;

#[cfg(test)]
mod tests;
