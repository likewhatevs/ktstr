//! Worker process management and telemetry.
//!
//! Workers are `fork()`ed processes by default ([`CloneMode::Fork`],
//! the `#[default]`) so each can be placed in its own cgroup;
//! [`CloneMode::Thread`] instead uses [`std::thread::spawn`], so those
//! workers share the parent's `tgid`, address space, and signal-handler
//! table. Key types:
//! - [`WorkType`] -- what each worker does
//! - [`WorkloadConfig`] -- spawn configuration (count, affinity, work type, policy)
//! - [`WorkloadHandle`] -- RAII handle to spawned workers
//! - [`WorkerReport`] -- per-worker telemetry collected after stop
//! - [`AffinityIntent`] -- per-worker affinity intent (Inherit, LlcAligned, Exact, etc.)
//! - [`ResolvedAffinity`] -- resolved CPU affinity for workers
//! - [`WorkSpec`] -- workload definition for a single group of workers within a cgroup
//! - [`WorkPhase`] -- a single phase in a [`WorkType::Sequence`] compound work pattern
//! - [`SchedPolicy`] -- Linux scheduling policy for a worker process
//! - [`MemPolicy`] -- NUMA memory placement policy for worker processes
//!
//! See the [WorkSpec Types](https://likewhatevs.github.io/ktstr/guide/concepts/work-types.html)
//! and [Worker Processes](https://likewhatevs.github.io/ktstr/guide/architecture/workers.html)
//! chapters of the guide.
//!
//! # Module layout
//!
//! - `affinity` — [`AffinityIntent`] / [`ResolvedAffinity`] +
//!   the resolver and `sched_setaffinity` wrapper.
//! - `config` — declarative test-author input
//!   ([`WorkloadConfig`], [`WorkSpec`], [`SchedPolicy`],
//!   [`MemPolicy`], [`MpolFlags`], [`CloneMode`],
//!   [`FutexLockMode`], [`WakeMechanism`], [`AluWidth`]) and
//!   the `humantime_serde_helper` shared by every `Duration`
//!   field.
//! - `types` — [`WorkType`] / [`WorkPhase`] /
//!   [`WorkTypeValidationError`] and the WorkType naming
//!   surface (`from_name`, `suggest`, `ALL_NAMES`).
//! - `spawn` — runtime spawn pipeline: [`WorkloadHandle`],
//!   `SpawnGuard`, [`Migration`], [`WorkerReport`],
//!   [`WorkerExitInfo`], `build_nodemask`,
//!   `apply_mempolicy_with_flags`, `apply_nice`. Tests are
//!   co-located in `spawn/tests_*.rs` siblings with shared
//!   fixtures in `spawn/testing.rs`.
//! - `worker` — `worker_main` and the per-WorkType bodies.
//!   `worker/io.rs` holds the IO-backing RAII wrappers and
//!   `worker/sched.rs` holds the scheduler/clock/metric
//!   helpers (incl. `set_sched_policy`).
//!
//! # Naming conventions
//!
//! ## "Intent" vs "Resolved" naming
//!
//! Types named with an `Intent` suffix carry **test-author intent**
//! (the input to the workload pipeline). Types named with a
//! `Resolved` prefix carry **runtime-resolved configuration** (the
//! output of intent + topology + cgroup state). [`AffinityIntent`]
//! resolves to [`ResolvedAffinity`] at spawn time via
//! [`resolve_affinity_for_cgroup`](crate::scenario::resolve_affinity_for_cgroup).
//!
//! [`CloneMode`] is a runtime-resolved value because the test
//! author writes `CloneMode::Fork` / `CloneMode::Thread` directly
//! (no resolution layer); the `Mode` suffix denotes a single
//! kernel-facing dispatch decision rather than a two-stage
//! intent/resolved pipeline.
//!
//! [`SchedClass`] and [`SchedPolicy`] follow the same coarse-intent /
//! concrete-runtime split using legacy kernel terminology rather
//! than the `Intent`/`Resolved` naming — see [`SchedClass`] for
//! the per-class mapping.
//!
//! ## "Churn" vs "Sweep" suffixes on [`WorkType`] variants
//!
//! Variants whose names end in `Churn` cycle their target setting at
//! high frequency to exercise the kernel's per-task state machines
//! under rapid transitions. [`WorkType::AffinityChurn`] samples a
//! random CPU from the effective cpuset on every iteration
//! (`rand::rng().random_range`); [`WorkType::PageFaultChurn`] touches
//! a fresh random subset of pages each cycle (xorshift64). Most Churn
//! variants pick each value randomly and independently of the
//! previous one; [`WorkType::PolicyChurn`] is the exception — despite
//! the `Churn` name it cycles through the supported scheduling
//! policies in a fixed, ordered sequence keyed on the iteration
//! counter (`iterations % policies.len()`).
//!
//! Variants whose names end in `Sweep` rotate their target setting
//! through an **ordered list or range** — the next value is a
//! deterministic function of the iteration counter, not a random
//! pick. [`WorkType::NiceSweep`] cycles nice values from
//! `effective_min..=19` modulo the range size;
//! [`WorkType::NumaWorkingSetSweep`] rotates the working-set
//! binding through `target_nodes` in declaration order. The
//! intent is to walk a phase space evenly so every value gets
//! comparable observation time, rather than producing the
//! unbiased-random transitions Churn produces.
//!
//! Choose `Churn` when the workload's value is its
//! transition-frequency entropy; choose `Sweep` when the workload
//! must visit every phase deterministically.

mod affinity;
mod config;
pub(crate) mod schbench;
mod spawn;
pub(crate) mod taobench;
mod types;
mod worker;

pub use affinity::*;
pub use config::*;
// `spawn` uses an itemised re-export rather than `pub use spawn::*`
// because the submodule contains internal helpers (`SpawnGuard`,
// `STOP`, `apply_mempolicy_with_flags`, …) that should stay
// crate-internal. Only the test-author-visible surface is
// re-exported here. `WorkerReportClaim` is the proc-macro-
// generated companion to `WorkerReport` (see the `crate::Claim`
// derive on the `WorkerReport` struct).
pub use spawn::{
    Migration, PhaseSlice, WorkerExitInfo, WorkerReport, WorkerReportClaim, WorkloadHandle,
};
// Crate-internal re-export of the wake-latency reservoir cap + the
// Algorithm-R push so the per-phase per-cgroup carrier builder
// (`crate::assert::phase_cgroup_stats`) re-caps the POOLED samples at the
// same bound the per-worker path uses (the carrier concatenates every
// worker's vec, so the pool must be re-capped before it crosses the
// size-limited guest bulk port). The `worker` module itself stays private.
pub(crate) use worker::{MAX_WAKE_SAMPLES, reservoir_push};
// `build_nodemask` is the low-level `set_mempolicy(2)` / `mbind(2)`
// nodemask builder. It's deliberately NOT in the public surface —
// test authors express NUMA placement through the [`MemPolicy`]
// enum — but `crate::vmm::host_topology` invokes `mbind(2)` directly to
// bind guest memory regions to host NUMA nodes
// (`crate::vmm::numa_mem`'s `mbind_regions` via
// `host_topology::mbind_to_nodes`) and needs an in-crate path to
// the helper.
pub(crate) use spawn::build_nodemask;
pub use types::*;
// schbench_rs is otherwise a private submodule. Its user-facing config type is
// re-exported (the WorkType::Schbench variant carries it), as are the standalone
// host-side validation entry point and its report type / percentile labels, plus
// the pipe-mode (`-p`) throughput helper (all used by the gated
// `ktstr-schbench-validate` bin for the side-by-side comparison against the
// reference schbench).
pub use schbench::{
    PipeTransferReport, SCHBENCH_PERCENTILES, SchbenchConfig, StandaloneReport,
    pipe_transfer_report, run_standalone,
};
// taobench_rs is a private submodule like schbench_rs above. Its user-facing
// config type + the host-side validation runner/report are re-exported here (the
// WorkType::Taobench variant carries the config; the runner backs the
// ktstr-taobench-validate driver). `run_standalone` is aliased to avoid colliding
// with schbench's flat `run_standalone` re-export above.
pub use taobench::{
    TaobenchConfig, TaobenchStandaloneReport, TaobenchStats,
    run_standalone as taobench_run_standalone,
};

// `FanOutCompute` stores its u64 generation counter at offset 0 of
// a 16-byte shared region and relies on the low 4 bytes of that
// counter living at offset 0 so the futex syscall (which reads the
// raw u32 at `futex_ptr`) sees the low u32 of the u64. That layout
// assumption holds on little-endian targets (x86_64, aarch64) and
// flips on big-endian — the futex would read the high 32 bits
// instead, and an increment of the u64 would leave the low 4 bytes
// unchanged until the 2^32-th advance. Reject the big-endian build
// at compile time rather than shipping a silently-broken binary.
#[cfg(not(target_endian = "little"))]
compile_error!(
    "ktstr's FanOutCompute generation-counter layout assumes a \
     little-endian target — the u64 counter at offset 0 of the \
     shared futex region must expose its low 32 bits to the \
     futex syscall at that same offset. Porting to a big-endian \
     target requires reworking the layout so futex_wait sees the \
     incrementing low 4 bytes."
);
