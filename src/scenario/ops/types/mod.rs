//! `Op`, `CgroupDef`, `Step`, and supporting limit types — pure data
//! model extracted from the parent [`super`] module. Re-exported by
//! the parent so external paths remain `crate::scenario::ops::Op` etc.
//! See the parent module for the full module-level documentation
//! (cgroup tooling overview, worked examples, implementation entry
//! points).
//!
//! # File layout
//!
//! The module body is split by responsibility:
//!
//! | File | Owns |
//! |------|------|
//! | [`op`] | [`Op`] variant taxonomy + [`OpKind`] discriminator + [`CpusetSpec`] enum + cpuset construction surface. |
//! | [`limits`] | [`CpuLimits`] / [`MemoryLimits`] / [`IoLimits`] / [`PidsLimits`] cgroup v2 per-controller knob bundles. |
//! | [`cgroup_def`] | [`CgroupDef`] declarative cgroup blueprint + builder methods (per-WorkSpec fan, `default_*` merge, in-place `pcomm` stamp, cpu/memory/io/pids controllers). |
//! | [`step`] | [`Setup`] / [`Step`] / [`HoldSpec`] step composition, plus the `impl Op` / `impl OpKind` blocks for the bit-index map and per-variant constructor sugar. |
//! | [`resolve`] | [`CpusetSpec`] → [`std::collections::BTreeSet`] resolution (`validate` / `resolve` / `resolve_quiet`). |
//! | `tests` | Order-independence / override-precedence coverage for the cgroup-level defaults flowing through `CgroupDef::merged_works`. |
//!
//! Cross-impl-block convention: Rust permits multiple impl blocks
//! for the same type across files in the same crate. This module
//! splits `impl Op` (constructors in [`step`], enum definition in
//! [`op`]) and `impl CpusetSpec` (constructors in [`op`], topology
//! resolution in [`resolve`]) along their natural responsibility
//! boundaries.

mod cgroup_def;
mod limits;
mod op;
mod resolve;
mod step;

pub use cgroup_def::CgroupDef;
pub use limits::{CpuLimits, IoLimits, MemoryLimits, PidsLimits};
pub use op::{CpusetSpec, Op, OpKind};
pub use step::{HoldSpec, Setup, Step};

#[cfg(test)]
mod tests;
