// SPDX-License-Identifier: Apache-2.0
//! WorkType, WorkPhase, and WorkTypeValidationError — pure data types
//! and pure-self methods extracted from the parent workload module.
//!
//! Re-exported by the parent module so external paths remain
//! `crate::workload::WorkType` etc. — the split is internal.

use std::time::Duration;

use crate::workload::config::{AluWidth, humantime_serde_helper};

/// A single phase in a [`WorkType::Sequence`] compound work pattern.
///
/// Workers loop through all phases in order, then repeat. Each phase
/// runs for its specified duration before advancing to the next.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkPhase {
    /// CPU spin for the given duration.
    Spin(#[serde(with = "humantime_serde_helper")] Duration),
    /// Sleep (thread::sleep) for the given duration.
    Sleep(#[serde(with = "humantime_serde_helper")] Duration),
    /// Yield (sched_yield) repeatedly for the given duration.
    Yield(#[serde(with = "humantime_serde_helper")] Duration),
    /// Simulated I/O (write 64 KB to a tempfile + 100 us sleep) for
    /// the given duration. The tempfile lives on whatever filesystem
    /// `std::env::temp_dir()` returns; on the ktstr guest's tmpfs the
    /// write is a page-cache memcpy and the sleep provides the
    /// blocking behavior that real disk fsync would cause.
    /// `WorkType::IoSyncWrite` (the standalone variant) is the disk-IO
    /// counterpart that opens `/dev/vda` directly.
    Io(#[serde(with = "humantime_serde_helper")] Duration),
    /// ALU-bound multiply chain for the given duration. The width
    /// selector picks the data path the same way as
    /// [`WorkType::AluHot`] — see [`AluWidth`] for the resolution
    /// rules and the AVX-512 / AMX caveats. Each visit runs
    /// `alu_hot_chain` in 1024-step batches in a deadline-bounded
    /// loop so shutdown latency stays bounded by one batch.
    ///
    /// The composable counterpart to [`WorkType::AluHot`]: use this
    /// inside a [`Sequence`](WorkType::Sequence) to express duty-cycle
    /// patterns ("ALU 90 % / Sleep 10 %") that the standalone
    /// [`WorkType::AluHot`] cannot, since the standalone variant
    /// runs ALU work for the entire scenario duration.
    AluHot {
        /// SIMD / scalar width selector for the multiply chain;
        /// resolved per phase visit via `resolve_alu_width`. See
        /// [`AluWidth`] for the per-variant data-path width and
        /// the runtime resolution rules.
        width: AluWidth,
        /// Wall-clock duration of the ALU phase. Workers retire
        /// `alu_hot_chain` batches until this deadline passes.
        #[serde(with = "humantime_serde_helper")]
        duration: Duration,
    },
}

impl WorkPhase {
    /// Construct a [`WorkPhase::AluHot`] variant. Sugar for the
    /// struct-literal form that brings the struct variant in line
    /// with the tuple-variant constructors — every other [`WorkPhase`]
    /// variant takes a single `Duration` and reads as
    /// `WorkPhase::Spin(d)` etc.; the struct variant needs an explicit
    /// `WorkPhase::AluHot { width: ..., duration: ... }` block at every
    /// call site, breaking the visual symmetry. `WorkPhase::alu_hot(w, d)`
    /// restores it so `vec![WorkPhase::Spin(d1), WorkPhase::alu_hot(w, d2),
    /// WorkPhase::Sleep(d3)]` reads consistently.
    pub const fn alu_hot(width: AluWidth, duration: Duration) -> Self {
        WorkPhase::AluHot { width, duration }
    }
}
mod methods;
mod work_type;

pub use work_type::WorkType;

/// Spawn-time validation failures for [`WorkType`] preconditions.
///
/// Returned (boxed inside [`anyhow::Error`]) by
/// `WorkloadHandle::spawn` when a per-group `WorkSpec` violates a
/// runtime invariant the variant doc declares as a precondition.
/// Tests that need to assert on a specific variant downcast via
/// `err.downcast_ref::<WorkTypeValidationError>()`; the
/// `Display` impl carries the same human-readable text the previous
/// `anyhow::bail!` strings did so call sites that match on the
/// rendered message keep working.
///
/// Each variant carries `group_idx` (the position of the offending
/// `WorkSpec` inside `WorkloadConfig::composed`; the primary
/// group is index 0) so multi-group scenarios can locate the
/// offending entry without re-parsing the message string. Variants
/// with multiple constraint inputs (depth, divisor, observed count)
/// expose those values as named fields to the same end.
#[derive(Clone, Debug, PartialEq, Eq, Hash, thiserror::Error)]
pub enum WorkTypeValidationError {
    /// [`WorkType::IdleChurn`] with `burst_duration == Duration::ZERO`.
    /// Collapses the per-iteration loop to pure nanosleep so the
    /// worker accrues no runtime — useless as a scheduler test. See
    /// the variant doc's "Spawn-time validation" section for the
    /// full rationale.
    #[error(
        "IdleChurn burst_duration must be > 0 (group {group_idx}); a zero \
         burst makes the loop pure sleep and the worker accrues \
         no runtime (see [`WorkType::IdleChurn`] variant doc)"
    )]
    ZeroBurstDuration {
        /// Index of the offending group in
        /// `WorkloadConfig::composed` (primary group = 0).
        group_idx: usize,
    },
    /// [`WorkType::IdleChurn`] with `sleep_duration == Duration::ZERO`.
    /// Collapses the per-iteration loop to a CPU-bound burst with
    /// no idle path; the kernel's `nanosleep(0)` is yield-like
    /// rather than idle-like. The diagnostic steers the caller to
    /// [`WorkType::SpinWait`] (pure CPU spin) or
    /// [`WorkType::YieldHeavy`] (the closer overlap).
    #[error(
        "IdleChurn sleep_duration must be > 0 (group {group_idx}); a zero \
         sleep collapses the loop to a CPU-bound burst. \
         Use WorkType::SpinWait for pure CPU spin, or \
         WorkType::YieldHeavy for the closer overlap \
         (nanosleep(0) is yield-like — see the variant \
         doc rationale in [`WorkType::IdleChurn`])."
    )]
    ZeroSleepDuration {
        /// Index of the offending group in
        /// `WorkloadConfig::composed` (primary group = 0).
        group_idx: usize,
    },
    /// [`WorkType::WakeChain`] with `depth < 2`. A 1-stage chain has
    /// no successor to wake, and the post-fork close-other-fds
    /// block would close the worker's own write end (deadlock).
    #[error(
        "WakeChain depth must be >= 2 (got {depth}, group {group_idx}); a 1-stage \
         chain has no successor to wake and the post-fork fd close \
         logic would close the worker's own write end \
         (see [`WorkType::WakeChain`] variant doc)"
    )]
    InsufficientWakeChainDepth {
        /// The offending `depth` value the caller supplied.
        depth: usize,
        /// Index of the offending group in
        /// `WorkloadConfig::composed` (primary group = 0).
        group_idx: usize,
    },
    /// `num_workers` is not a positive multiple of the variant's
    /// [`worker_group_size`](WorkType::worker_group_size). Affects
    /// every grouped variant (paired, fan-out, herd, contention,
    /// chain). The diagnostic names the variant via [`WorkType::name`].
    #[error(
        "{name} (group {group_idx}) requires num_workers divisible by {group_size}, got {num_workers}"
    )]
    NonDivisibleWorkerCount {
        /// PascalCase variant name from [`WorkType::name`].
        name: String,
        /// Index of the offending group in
        /// `WorkloadConfig::composed` (primary group = 0).
        group_idx: usize,
        /// Required group size (the variant's
        /// [`worker_group_size`](WorkType::worker_group_size)).
        group_size: usize,
        /// The `num_workers` count the caller supplied.
        num_workers: usize,
    },
    /// [`WorkType::IpcVariance`] with one of `hot_iters`,
    /// `cold_iters`, or `period_iters` equal to `0`. A zero in
    /// any of the three collapses the alternation: zero
    /// `hot_iters` produces a pure cold-phase memory loop, zero
    /// `cold_iters` produces a pure ALU loop (use
    /// [`WorkType::AluHot`] directly for that), and zero
    /// `period_iters` produces a worker that never advances
    /// past the first stop check. Each rejection names the
    /// offending field so the caller knows which to fix.
    #[error(
        "IpcVariance {field} must be > 0 (group {group_idx}); a zero value \
         collapses the hot/cold alternation and produces a degenerate \
         workload (see [`WorkType::IpcVariance`] variant doc)"
    )]
    ZeroIpcVarianceParam {
        /// Static name of the offending field —
        /// `"hot_iters"`, `"cold_iters"`, or `"period_iters"`.
        field: &'static str,
        /// Index of the offending group in
        /// `WorkloadConfig::composed` (primary group = 0).
        group_idx: usize,
    },
}

#[cfg(test)]
mod tests;
