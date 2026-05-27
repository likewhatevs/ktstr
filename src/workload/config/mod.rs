//! Declarative configuration types for the workload pipeline.
//!
//! Holds every type a test author writes (or that round-trips through
//! serde) without crossing the kernel boundary itself: [`WorkloadConfig`]
//! and its [`WorkSpec`] composed entries, the per-knob enums
//! ([`SchedPolicy`], [`SchedClass`], [`MemPolicy`], [`MpolFlags`],
//! [`CloneMode`], [`FutexLockMode`], [`WakeMechanism`], [`AluWidth`]),
//! the [`defaults`] constants `WorkType::from_name` consults, the
//! [`humantime_serde_helper`] module the duration fields cite, and the
//! [`resolve_work_type`] selector. The corresponding kernel-call
//! helpers live in the [`spawn`](super::spawn) submodule
//! (`apply_mempolicy_with_flags`, `apply_nice`, `build_nodemask`)
//! and the [`worker`](super::worker) submodule
//! (`set_sched_policy` in `worker/sched.rs`).
//!
//! Types are re-exported from the parent module via `pub use config::*`,
//! so existing `crate::workload::WorkloadConfig` paths continue to
//! resolve.

use super::WorkType;

mod mempolicy;
mod sched;
mod work;
mod workload;

pub use mempolicy::{MemPolicy, MpolFlags};
pub use sched::{AluWidth, FutexLockMode, SchedClass, SchedPolicy, WakeMechanism};
pub use work::WorkSpec;
pub(crate) use work::validate_task_comm_string;
pub use workload::WorkloadConfig;

/// Serde helper for [`std::time::Duration`] using human-readable
/// strings (`"100ms"`, `"5s"`, `"1h30m"`) instead of the default
/// `{secs, nanos}` object.
///
/// Wire format chosen so persisted [`WorkSpec`] / [`WorkloadConfig`]
/// values are operator-readable: a test author who exports a config
/// can edit `"work_per_hop": "100us"` directly without translating
/// from `{secs: 0, nanos: 100_000}`.
///
/// Reuses the [`humantime`] crate already pulled in for CLI flag
/// parsing — no new dependency. Use via `#[serde(with =
/// "humantime_serde_helper")]` on `Duration` fields.
pub(crate) mod humantime_serde_helper {
    use std::time::Duration;

    pub fn serialize<S: serde::Serializer>(d: &Duration, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&humantime::format_duration(*d).to_string())
    }

    pub fn deserialize<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Duration, D::Error> {
        let s = <String as serde::Deserialize>::deserialize(d)?;
        humantime::parse_duration(&s).map_err(serde::de::Error::custom)
    }
}

/// Named defaults for the parametric [`WorkType`] variants, used by
/// [`WorkType::from_name`]. Extracting the magic numbers here
/// provides a named home for the default values so tests and docs
/// (e.g. `doc/guide/src/architecture/workers.md`) can cite them by
/// constant name instead of each tracking a scattered integer
/// literal. Every value carries a single-line comment naming the
/// knob and its unit; the const names mirror the
/// `{variant_snake}_{field}` convention so renames show up as
/// compile errors in both sites.
pub mod defaults {
    // Bursty
    pub const BURSTY_BURST_DURATION: std::time::Duration = std::time::Duration::from_millis(50);
    pub const BURSTY_SLEEP_DURATION: std::time::Duration = std::time::Duration::from_millis(100);
    // PipeIo
    pub const PIPE_IO_BURST_ITERS: u64 = 1024;
    // FutexPingPong
    pub const FUTEX_PING_PONG_SPIN_ITERS: u64 = 1024;
    // CachePressure / CacheYield / CachePipe share buffer shape
    pub const CACHE_PRESSURE_SIZE_KIB: usize = 32;
    pub const CACHE_PRESSURE_STRIDE: usize = 64;
    pub const CACHE_YIELD_SIZE_KIB: usize = 32;
    pub const CACHE_YIELD_STRIDE: usize = 64;
    pub const CACHE_PIPE_SIZE_KIB: usize = 32;
    pub const CACHE_PIPE_BURST_ITERS: u64 = 1024;
    // FutexFanOut
    pub const FUTEX_FAN_OUT_FAN_OUT: usize = 4;
    pub const FUTEX_FAN_OUT_SPIN_ITERS: u64 = 1024;
    // AffinityChurn
    pub const AFFINITY_CHURN_SPIN_ITERS: u64 = 1024;
    // PolicyChurn
    pub const POLICY_CHURN_SPIN_ITERS: u64 = 1024;
    // FanOutCompute
    pub const FAN_OUT_COMPUTE_FAN_OUT: usize = 4;
    pub const FAN_OUT_COMPUTE_CACHE_FOOTPRINT_KIB: usize = 256;
    pub const FAN_OUT_COMPUTE_OPERATIONS: usize = 5;
    pub const FAN_OUT_COMPUTE_SLEEP_USEC: u64 = 100;
    // PageFaultChurn
    pub const PAGE_FAULT_CHURN_REGION_KIB: usize = 4096;
    pub const PAGE_FAULT_CHURN_TOUCHES_PER_CYCLE: usize = 256;
    pub const PAGE_FAULT_CHURN_SPIN_ITERS: u64 = 64;
    // MutexContention
    pub const MUTEX_CONTENTION_CONTENDERS: usize = 4;
    pub const MUTEX_CONTENTION_HOLD_ITERS: u64 = 256;
    pub const MUTEX_CONTENTION_WORK_ITERS: u64 = 1024;
    // ThunderingHerd
    pub const THUNDERING_HERD_WAITERS: usize = 7;
    pub const THUNDERING_HERD_BATCHES: u64 = 1_000;
    pub const THUNDERING_HERD_INTER_BATCH_MS: u64 = 5;
    // PriorityInversion
    pub const PRIORITY_INVERSION_HIGH_COUNT: usize = 1;
    pub const PRIORITY_INVERSION_MEDIUM_COUNT: usize = 1;
    pub const PRIORITY_INVERSION_LOW_COUNT: usize = 1;
    pub const PRIORITY_INVERSION_HOLD_ITERS: u64 = 4096;
    pub const PRIORITY_INVERSION_WORK_ITERS: u64 = 1024;
    pub const PRIORITY_INVERSION_PI_MODE: super::FutexLockMode = super::FutexLockMode::Plain;
    // ProducerConsumerImbalance
    pub const PRODUCER_CONSUMER_PRODUCERS: usize = 2;
    pub const PRODUCER_CONSUMER_CONSUMERS: usize = 1;
    pub const PRODUCER_CONSUMER_PRODUCE_RATE_HZ: u64 = 1_000;
    pub const PRODUCER_CONSUMER_CONSUME_ITERS: u64 = 4_096;
    pub const PRODUCER_CONSUMER_QUEUE_DEPTH_TARGET: u64 = 1024;
    // RtStarvation
    pub const RT_STARVATION_RT_WORKERS: usize = 1;
    pub const RT_STARVATION_CFS_WORKERS: usize = 1;
    pub const RT_STARVATION_RT_PRIORITY: i32 = 50;
    pub const RT_STARVATION_BURST_ITERS: u64 = 1024;
    // AsymmetricWaker
    pub const ASYMMETRIC_WAKER_BURST_ITERS: u64 = 1024;
    // WakeChain
    pub const WAKE_CHAIN_DEPTH: usize = 4;
    pub const WAKE_CHAIN_WAKE: super::WakeMechanism = super::WakeMechanism::Pipe;
    pub const WAKE_CHAIN_WORK_PER_HOP: std::time::Duration = std::time::Duration::from_micros(100);
    // NumaWorkingSetSweep
    pub const NUMA_WORKING_SET_SWEEP_REGION_KIB: usize = 4_096;
    pub const NUMA_WORKING_SET_SWEEP_SWEEP_PERIOD_MS: u64 = 100;
    // CgroupChurn
    pub const CGROUP_CHURN_GROUPS: usize = 2;
    pub const CGROUP_CHURN_CYCLE_MS: u64 = 100;
    // SignalStorm
    pub const SIGNAL_STORM_SIGNALS_PER_ITER: u64 = 16;
    pub const SIGNAL_STORM_WORK_ITERS: u64 = 1024;
    // PreemptStorm
    pub const PREEMPT_STORM_CFS_WORKERS: usize = 2;
    pub const PREEMPT_STORM_RT_BURST_ITERS: u64 = 1024;
    pub const PREEMPT_STORM_RT_SLEEP_US: u64 = 1_000;
    // EpollStorm
    pub const EPOLL_STORM_PRODUCERS: usize = 1;
    pub const EPOLL_STORM_CONSUMERS: usize = 2;
    pub const EPOLL_STORM_EVENTS_PER_BURST: u64 = 32;
    // NumaMigrationChurn
    pub const NUMA_MIGRATION_CHURN_PERIOD_MS: u64 = 100;
    // IdleChurn
    pub const IDLE_CHURN_BURST_DURATION: std::time::Duration = std::time::Duration::from_millis(1);
    pub const IDLE_CHURN_SLEEP_DURATION: std::time::Duration = std::time::Duration::from_millis(5);
    /// Default for `WorkType::IdleChurn`'s `precise_timing` field.
    /// `false` keeps the inherited 50µs `current->timer_slack_ns`
    /// the variant doc describes; opt-in callers set the field to
    /// `true` directly to call `prctl(PR_SET_TIMERSLACK, 1)`.
    pub const IDLE_CHURN_PRECISE_TIMING: bool = false;
    // AluHot
    /// Default for `WorkType::AluHot`'s `width` field. `Widest`
    /// resolves to the widest data-path the host supports at
    /// worker entry — see [`super::AluWidth`] for the resolution
    /// order.
    pub const ALU_HOT_WIDTH: super::AluWidth = super::AluWidth::Widest;
    // IpcVariance
    /// Multiply-chain steps per hot phase in `WorkType::IpcVariance`.
    /// At IPC 2.0 / 2 GHz this spans ~50µs — long enough that the
    /// scheduler's IPC-window observer sees a steady high-IPC
    /// signal before the cold phase flips it.
    pub const IPC_VARIANCE_HOT_ITERS: u64 = 100_000;
    /// Random cache-line touches per cold phase in
    /// `WorkType::IpcVariance`. 1024 touches across a 512KB
    /// working set on a typical x86 core takes ~100µs (LLC) to
    /// ~1ms (DRAM-spill).
    pub const IPC_VARIANCE_COLD_ITERS: u64 = 1024;
    /// Hot+cold pair iterations per outer loop in
    /// `WorkType::IpcVariance`. 64 keeps per-stop-check
    /// overhead at <2% while bounding shutdown latency to one
    /// outer iteration (~10ms with the defaults above).
    pub const IPC_VARIANCE_PERIOD_ITERS: u64 = 64;
}

/// Resolve a work type with an optional override.
///
/// Returns a clone of `override_wt` when `swappable` is true, an
/// override is provided, and the override's group size (if any)
/// divides `num_workers`. Otherwise returns a clone of `base`. When
/// `override_wt` is `None`, always returns `base` regardless of
/// `swappable`.
pub(crate) fn resolve_work_type(
    base: &WorkType,
    override_wt: Option<&WorkType>,
    swappable: bool,
    num_workers: usize,
) -> WorkType {
    if !swappable {
        return base.clone();
    }
    match override_wt {
        Some(wt) => {
            if let Some(gs) = wt.worker_group_size()
                && !num_workers.is_multiple_of(gs)
            {
                return base.clone();
            }
            wt.clone()
        }
        None => base.clone(),
    }
}

/// How `WorkloadHandle::spawn` creates worker tasks.
///
/// `Fork` is the default — the existing `fork(2)` path with
/// separate address space, separate thread group, and `waitpid`
/// reaping. `Thread` switches to [`std::thread::spawn`] for workers
/// that share the test runner's tgid.
///
/// # `WorkType` × `CloneMode` compatibility
///
/// Most [`WorkType`] variants compose with both clone modes. The
/// only exception is surfaced at spawn time by
/// `WorkloadHandle::spawn`:
///
/// | WorkType                | Fork | Thread |
/// |-------------------------|------|--------|
/// | All variants (default)  | OK   | OK     |
/// | [`WorkType::ForkExit`]  | OK   | reject |
///
/// `ForkExit + Thread` is rejected because the worker body calls
/// `libc::fork()` from inside a thread of the parent's tgid; the
/// child then calls `_exit(0)`, which the kernel routes through
/// `do_exit`, tearing down the entire tgid (every sibling thread
/// dies). Use [`CloneMode::Fork`] for [`WorkType::ForkExit`].
///
/// Other Thread-mode interactions worth knowing:
///
/// - [`WorkType::NiceSweep`]: `setpriority(PRIO_PROCESS, 0, …)`
///   targets the calling task only (`kernel/sys.c::sys_setpriority`
///   `case PRIO_PROCESS: if (who == 0) p = current`), so each
///   sibling thread independently sweeps its own nice. Allowed.
/// - [`WorkType::AffinityChurn`]: `sched_setaffinity(0, …)`
///   addresses the calling thread by kernel rule
///   (`kernel/sched/syscalls.c::sched_setaffinity`). Allowed; no
///   cross-thread interference.
/// - [`WorkType::PolicyChurn`]: `sched_setscheduler(0, …)` is also
///   per-task. Allowed.
/// - [`WorkType::AsymmetricWaker`] with an RT class: legal but
///   the harness still runs as its original (likely SCHED_NORMAL)
///   policy; only the worker thread is RT.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CloneMode {
    /// Plain `fork(2)`: separate address space, separate thread
    /// group (`p->tgid = p->pid`), reaped via `waitpid`. The default
    /// — preserves existing `WorkloadHandle::spawn` behavior.
    #[default]
    Fork,
    /// Same thread group as the spawning process. Implementation
    /// uses [`std::thread::spawn`]; the Rust thread runtime owns
    /// all clone-flag selection internally. Reaped via
    /// [`std::thread::JoinHandle`]. Workers share `tgid`,
    /// signal-handler table, and address space with the parent —
    /// observers like `task_struct->group_leader`, `tgid`,
    /// `real_parent` all match the parent's.
    Thread,
}

#[cfg(test)]
mod tests {
    use super::super::AffinityIntent;
    use super::super::types::WorkType;
    use super::*;
    use std::collections::BTreeSet;
    use std::time::Duration;

    #[test]
    fn sched_policy_debug_shows_variant_and_priority() {
        let s = format!("{:?}", SchedPolicy::Fifo(50));
        assert!(s.contains("Fifo"), "must show variant name");
        assert!(s.contains("50"), "must show priority value");
        let s = format!("{:?}", SchedPolicy::RoundRobin(99));
        assert!(s.contains("RoundRobin"), "must show variant name");
        assert!(s.contains("99"), "must show priority value");
        // Ensure different priorities produce different output.
        let s1 = format!("{:?}", SchedPolicy::Fifo(1));
        let s10 = format!("{:?}", SchedPolicy::Fifo(10));
        assert_ne!(
            s1, s10,
            "different priorities must produce different debug output"
        );
    }
    #[test]
    fn sched_policy_copy_preserves_priority() {
        let a = SchedPolicy::Fifo(42);
        let b = a; // Copy
        match b {
            SchedPolicy::Fifo(p) => assert_eq!(p, 42),
            _ => panic!("copy must preserve variant and priority"),
        }
    }
    // -- SchedPolicy constructors --

    #[test]
    fn sched_policy_fifo_constructor() {
        match SchedPolicy::fifo(50) {
            SchedPolicy::Fifo(p) => assert_eq!(p, 50),
            _ => panic!("expected Fifo"),
        }
    }
    #[test]
    fn sched_policy_rr_constructor() {
        match SchedPolicy::round_robin(25) {
            SchedPolicy::RoundRobin(p) => assert_eq!(p, 25),
            _ => panic!("expected RoundRobin"),
        }
    }
    // -- MemPolicy tests --

    #[test]
    fn mempolicy_default_node_set_empty() {
        assert!(MemPolicy::Default.node_set().is_empty());
    }
    #[test]
    fn mempolicy_local_node_set_empty() {
        assert!(MemPolicy::Local.node_set().is_empty());
    }
    #[test]
    fn mempolicy_bind_node_set() {
        let p = MemPolicy::Bind([0, 2].into_iter().collect());
        assert_eq!(p.node_set(), [0, 2].into_iter().collect());
    }
    #[test]
    fn mempolicy_preferred_node_set() {
        let p = MemPolicy::Preferred(1);
        assert_eq!(p.node_set(), [1].into_iter().collect());
    }
    #[test]
    fn mempolicy_interleave_node_set() {
        let p = MemPolicy::Interleave([0, 1, 3].into_iter().collect());
        assert_eq!(p.node_set(), [0, 1, 3].into_iter().collect());
    }
    #[test]
    fn mempolicy_preferred_many_node_set() {
        let p = MemPolicy::preferred_many([0, 2]);
        assert_eq!(p.node_set(), [0, 2].into_iter().collect());
    }
    #[test]
    fn mempolicy_weighted_interleave_node_set() {
        let p = MemPolicy::weighted_interleave([1, 3]);
        assert_eq!(p.node_set(), [1, 3].into_iter().collect());
    }
    #[test]
    fn mempolicy_validate_bind_empty() {
        let err = MemPolicy::Bind(BTreeSet::new()).validate().unwrap_err();
        assert!(
            err.contains("Bind") && err.contains("NUMA node"),
            "diagnostic must name the variant and required content: {err}",
        );
        // Actionable-trailer pin: the trailer points
        // at the constructor a copy-paste fix would use. A future
        // simplification that strips the trailer back to the terse
        // form would silently regress the documented inline-fix UX
        // (see MemPolicy::validate doc).
        assert!(
            err.contains("MemPolicy::bind("),
            "diagnostic must name the recommended constructor: {err}",
        );
    }
    #[test]
    fn mempolicy_validate_interleave_empty() {
        let err = MemPolicy::Interleave(BTreeSet::new())
            .validate()
            .unwrap_err();
        assert!(
            err.contains("Interleave") && err.contains("NUMA node"),
            "diagnostic must name the variant and required content: {err}",
        );
        assert!(
            err.contains("MemPolicy::interleave("),
            "diagnostic must name the recommended constructor: {err}",
        );
    }
    #[test]
    fn mempolicy_validate_preferred_many_empty() {
        let err = MemPolicy::PreferredMany(BTreeSet::new())
            .validate()
            .unwrap_err();
        assert!(
            err.contains("PreferredMany") && err.contains("NUMA node"),
            "diagnostic must name the variant and required content: {err}",
        );
        assert!(
            err.contains("MemPolicy::preferred_many("),
            "diagnostic must name the recommended constructor: {err}",
        );
    }
    #[test]
    fn mempolicy_validate_weighted_interleave_empty() {
        let err = MemPolicy::WeightedInterleave(BTreeSet::new())
            .validate()
            .unwrap_err();
        assert!(
            err.contains("WeightedInterleave") && err.contains("NUMA node"),
            "diagnostic must name the variant and required content: {err}",
        );
        assert!(
            err.contains("MemPolicy::weighted_interleave("),
            "diagnostic must name the recommended constructor: {err}",
        );
        // Phd D1 regression guard: the WeightedInterleave trailer
        // previously suggested `MemPolicy::Interleave([...])` (capital
        // I — the tuple variant) which won't compile because
        // `Interleave(BTreeSet<usize>)` cannot be constructed from
        // a literal array. The correct suggestion is the lowercase
        // `interleave(...)` function constructor. This assertion
        // pins the fix.
        assert!(
            !err.contains("MemPolicy::Interleave(["),
            "diagnostic must not suggest the non-compiling capital-I Interleave variant with a literal array: {err}",
        );
    }
    #[test]
    fn mempolicy_validate_preferred_many_ok() {
        assert!(MemPolicy::preferred_many([0]).validate().is_ok());
    }
    #[test]
    fn mempolicy_validate_weighted_interleave_ok() {
        assert!(MemPolicy::weighted_interleave([0, 1]).validate().is_ok());
    }

    #[test]
    fn workload_config_validate_accepts_default() {
        WorkloadConfig::default()
            .validate()
            .expect("WorkloadConfig::default must self-validate (mem_policy=Default)");
    }

    #[test]
    fn workload_config_validate_rejects_invalid_primary_mempolicy() {
        let cfg = WorkloadConfig::default().mem_policy(MemPolicy::Bind(BTreeSet::new()));
        let err = cfg
            .validate()
            .expect_err("empty Bind nodemask on primary must reject");
        let msg = err.to_string();
        assert!(
            msg.contains("primary") && msg.contains("Bind") && msg.contains("NUMA node"),
            "diagnostic must name the slot (primary), the variant (Bind), and the constraint (NUMA node): got {msg}",
        );
    }

    #[test]
    fn workload_config_validate_rejects_invalid_composed_mempolicy() {
        let bad = WorkSpec::default()
            .work_type(WorkType::SpinWait)
            .mem_policy(MemPolicy::Interleave(BTreeSet::new()));
        let cfg = WorkloadConfig::default().composed(vec![bad]);
        let err = cfg
            .validate()
            .expect_err("empty Interleave nodemask on composed[0] must reject");
        let msg = err.to_string();
        assert!(
            msg.contains("composed[0]")
                && msg.contains("group_idx 1")
                && msg.contains("Interleave"),
            "diagnostic must name composed[0] + group_idx 1 + Interleave: got {msg}",
        );
    }

    #[test]
    fn workload_config_validate_accepts_valid_composed_mempolicy() {
        let ok = WorkSpec::default()
            .work_type(WorkType::SpinWait)
            .mem_policy(MemPolicy::Bind([0].into_iter().collect()));
        let cfg = WorkloadConfig::default().composed(vec![ok]);
        cfg.validate()
            .expect("non-empty Bind on composed[0] must validate");
    }

    /// Pins `?` short-circuit semantics in the composed-validation
    /// loop. composed[0] is valid; composed[1] is invalid Bind;
    /// composed[2] is invalid Interleave. The first invalid entry
    /// (composed[1]) must surface; subsequent invalid entries
    /// (composed[2]) must NOT appear in the diagnostic. A regression
    /// that switched to an error-accumulator pattern (try_fold into a
    /// Vec, partition, etc.) would change which composed[N] appears,
    /// silently inverting the test-author's debugging order. Editor
    /// note: `.collect::<Result<_, _>>()` also short-circuits on the
    /// first Err, so swapping the for-loop for collect wouldn't break
    /// this assertion — only a true accumulator would.
    #[test]
    fn workload_config_validate_short_circuits_first_invalid_composed() {
        let valid_spec = WorkSpec::default()
            .work_type(WorkType::SpinWait)
            .mem_policy(MemPolicy::Bind([0].into_iter().collect()));
        let invalid_bind = WorkSpec::default()
            .work_type(WorkType::SpinWait)
            .mem_policy(MemPolicy::Bind(BTreeSet::new()));
        let invalid_interleave = WorkSpec::default()
            .work_type(WorkType::SpinWait)
            .mem_policy(MemPolicy::Interleave(BTreeSet::new()));
        let cfg =
            WorkloadConfig::default().composed(vec![valid_spec, invalid_bind, invalid_interleave]);
        let err = cfg
            .validate()
            .expect_err("multi-composed with invalid entries must reject");
        let msg = err.to_string();
        assert!(
            msg.contains("composed[1]"),
            "diagnostic must name the FIRST invalid composed entry (composed[1]): got {msg}",
        );
        assert!(
            msg.contains("Bind"),
            "diagnostic must name the first failing variant (Bind): got {msg}",
        );
        // The negative assertion is LOAD-BEARING on the short-circuit
        // semantics (`?` in the validate loop returns on first Err),
        // not on the wrap content. A future "errors-trailing-
        // suggestions" rewrite that mentions composed.len() or
        // re-formats the wrap to include sibling indices would
        // silently break this guard — at which point the right fix
        // is to assert on the structural property (e.g. count of
        // anyhow::Error frames) rather than to relax the substring
        // check.
        assert!(
            !msg.contains("composed[2]"),
            "short-circuit must not surface the second invalid entry (composed[2]): got {msg}",
        );
    }
    #[test]
    fn mpol_flags_union() {
        let f = MpolFlags::STATIC_NODES | MpolFlags::NUMA_BALANCING;
        assert_eq!(f.bits(), (1 << 15) | (1 << 13));
    }
    #[test]
    fn mpol_flags_none_is_zero() {
        assert_eq!(MpolFlags::NONE.bits(), 0);
    }
    #[test]
    fn work_mpol_flags_builder() {
        let w = WorkSpec::default().mpol_flags(MpolFlags::STATIC_NODES);
        assert_eq!(w.mpol_flags, MpolFlags::STATIC_NODES);
    }
    #[test]
    fn mpol_flags_contains_identity() {
        assert!(MpolFlags::NONE.contains(MpolFlags::NONE));
        assert!(MpolFlags::STATIC_NODES.contains(MpolFlags::STATIC_NODES));
        let composite = MpolFlags::STATIC_NODES | MpolFlags::NUMA_BALANCING;
        assert!(composite.contains(composite));
    }
    #[test]
    fn mpol_flags_contains_superset_is_true_for_subset() {
        let composite = MpolFlags::STATIC_NODES | MpolFlags::NUMA_BALANCING;
        assert!(composite.contains(MpolFlags::STATIC_NODES));
        assert!(composite.contains(MpolFlags::NUMA_BALANCING));
    }
    #[test]
    fn mpol_flags_contains_subset_is_false_for_superset() {
        let composite = MpolFlags::STATIC_NODES | MpolFlags::NUMA_BALANCING;
        assert!(!MpolFlags::STATIC_NODES.contains(composite));
        assert!(!MpolFlags::NUMA_BALANCING.contains(composite));
    }
    #[test]
    fn mpol_flags_contains_empty_is_always_true() {
        // `(x & 0) == 0` holds for every x, so every MpolFlags
        // value — including NONE itself — is a superset of NONE.
        assert!(MpolFlags::NONE.contains(MpolFlags::NONE));
        assert!(MpolFlags::STATIC_NODES.contains(MpolFlags::NONE));
        let composite = MpolFlags::STATIC_NODES | MpolFlags::NUMA_BALANCING;
        assert!(composite.contains(MpolFlags::NONE));
    }
    #[test]
    fn mpol_flags_none_does_not_contain_any_set_flag() {
        assert!(!MpolFlags::NONE.contains(MpolFlags::STATIC_NODES));
        assert!(!MpolFlags::NONE.contains(MpolFlags::RELATIVE_NODES));
        assert!(!MpolFlags::NONE.contains(MpolFlags::NUMA_BALANCING));
    }
    #[test]
    fn mpol_flags_contains_rejects_disjoint_flag() {
        // Single-flag values that share no bits must not satisfy
        // `contains` in either direction.
        assert!(!MpolFlags::STATIC_NODES.contains(MpolFlags::NUMA_BALANCING));
        assert!(!MpolFlags::NUMA_BALANCING.contains(MpolFlags::STATIC_NODES));
    }
    #[test]
    fn mpol_flags_contains_rejects_partial_overlap() {
        // Partial bit overlap must not satisfy `contains` — every
        // bit of `other` must be set in `self`, not merely some.
        let a = MpolFlags::STATIC_NODES | MpolFlags::NUMA_BALANCING;
        let b = MpolFlags::RELATIVE_NODES | MpolFlags::NUMA_BALANCING;
        assert!(!a.contains(b));
        assert!(!b.contains(a));
    }
    // -- CloneMode tests --

    #[test]
    fn clone_mode_default_is_fork() {
        // Preserves historical fork-based behavior — anything else
        // would silently change every existing caller's spawn path.
        assert!(matches!(CloneMode::default(), CloneMode::Fork));
    }
    #[test]
    fn workload_config_default_clone_mode_is_fork() {
        let c = WorkloadConfig::default();
        assert!(matches!(c.clone_mode, CloneMode::Fork));
    }
    #[test]
    fn workload_config_clone_mode_builder() {
        let cfg = WorkloadConfig::default().clone_mode(CloneMode::Thread);
        assert!(matches!(cfg.clone_mode, CloneMode::Thread));
    }
    #[test]
    fn work_mem_policy_builder() {
        let w = WorkSpec::default().mem_policy(MemPolicy::Bind([0].into_iter().collect()));
        assert!(matches!(w.mem_policy, MemPolicy::Bind(_)));
    }
    #[test]
    fn work_default_mempolicy_is_default() {
        let w = WorkSpec::default();
        assert!(matches!(w.mem_policy, MemPolicy::Default));
    }
    #[test]
    fn workload_config_default_mempolicy() {
        let wl = WorkloadConfig::default();
        assert!(matches!(wl.mem_policy, MemPolicy::Default));
    }
    /// `comm` / `uid` / `gid` / `numa_node` mirror the matcher knobs
    /// that already live on [`WorkSpec`] — ensure the top-level
    /// defaults are `None` and the builders set the field.
    #[test]
    fn workload_config_default_matcher_fields_are_none() {
        let wl = WorkloadConfig::default();
        assert!(wl.comm.is_none());
        assert!(wl.uid.is_none());
        assert!(wl.gid.is_none());
        assert!(wl.numa_node.is_none());
    }
    #[test]
    fn workload_config_matcher_field_builders() {
        let wl = WorkloadConfig::default()
            .comm("ktstr-worker")
            .uid(1001)
            .gid(1002)
            .numa_node(0);
        assert_eq!(wl.comm.as_deref(), Some("ktstr-worker"));
        assert_eq!(wl.uid, Some(1001));
        assert_eq!(wl.gid, Some(1002));
        assert_eq!(wl.numa_node, Some(0));
    }
    /// Full `WorkloadConfig` round-trip with `Default` ensures every
    /// field handles serde correctly together — no field is silently
    /// missing a derive.
    #[test]
    fn workload_config_default_roundtrips() {
        let cfg = WorkloadConfig::default();
        let json = serde_json::to_string(&cfg).unwrap();
        let back: WorkloadConfig = serde_json::from_str(&json).unwrap();
        // Compare via re-serialization since WorkloadConfig has no PartialEq.
        let json2 = serde_json::to_string(&back).unwrap();
        assert_eq!(json, json2);
    }

    // -- resolve_work_type --

    #[test]
    fn resolve_work_type_not_swappable() {
        let base = WorkType::SpinWait;
        let over = WorkType::YieldHeavy;
        let result = resolve_work_type(&base, Some(&over), false, 4);
        assert!(matches!(result, WorkType::SpinWait));
    }
    #[test]
    fn resolve_work_type_swappable_applies_override() {
        let base = WorkType::SpinWait;
        let over = WorkType::YieldHeavy;
        let result = resolve_work_type(&base, Some(&over), true, 4);
        assert!(matches!(result, WorkType::YieldHeavy));
    }
    #[test]
    fn resolve_work_type_swappable_no_override() {
        let base = WorkType::SpinWait;
        let result = resolve_work_type(&base, None, true, 4);
        assert!(matches!(result, WorkType::SpinWait));
    }
    #[test]
    fn resolve_work_type_group_size_mismatch() {
        let base = WorkType::SpinWait;
        let over = WorkType::pipe_io(100); // group_size = 2
        let result = resolve_work_type(&base, Some(&over), true, 3); // 3 not divisible by 2
        assert!(matches!(result, WorkType::SpinWait));
    }
    #[test]
    fn resolve_work_type_group_size_match() {
        let base = WorkType::SpinWait;
        let over = WorkType::pipe_io(100); // group_size = 2
        let result = resolve_work_type(&base, Some(&over), true, 4); // 4 divisible by 2
        assert!(matches!(result, WorkType::PipeIo { .. }));
    }

    // -- WorkSpec builder --

    #[test]
    fn work_builder_chain() {
        let w = WorkSpec::default()
            .workers(8)
            .work_type(WorkType::bursty(
                Duration::from_millis(10),
                Duration::from_millis(20),
            ))
            .sched_policy(SchedPolicy::Batch)
            .affinity(AffinityIntent::SingleCpu)
            .nice(7);
        assert_eq!(w.num_workers, Some(8));
        if let WorkType::Bursty {
            burst_duration,
            sleep_duration,
        } = w.work_type
        {
            assert_eq!(burst_duration, Duration::from_millis(10));
            assert_eq!(sleep_duration, Duration::from_millis(20));
        } else {
            panic!("expected Bursty variant; got {:?}", w.work_type);
        }
        assert!(matches!(w.sched_policy, SchedPolicy::Batch));
        assert!(matches!(w.affinity, AffinityIntent::SingleCpu));
        assert_eq!(w.nice, Some(7));
    }
    #[test]
    fn work_default_values() {
        let w = WorkSpec::default();
        assert_eq!(w.num_workers, None);
        assert!(matches!(w.work_type, WorkType::SpinWait));
        assert!(matches!(w.sched_policy, SchedPolicy::Normal));
        assert!(matches!(w.affinity, AffinityIntent::Inherit));
        // Default nice is None — same skip semantics as
        // [`WorkloadConfig::nice`].
        assert_eq!(w.nice, None);
    }

    /// GAP 9: pin that `SchedPolicy::fifo` / `round_robin` /
    /// `deadline` are usable in const context. A regression where
    /// any of the three dropped `const` (e.g. switched from `Self {
    /// .. }` to a builder) would silently break static
    /// `KtstrTestEntry` declarations that bake a fixed policy.
    #[test]
    fn sched_policy_constructors_usable_in_const_context() {
        const F: SchedPolicy = SchedPolicy::fifo(50);
        const RR: SchedPolicy = SchedPolicy::round_robin(99);
        const DL: SchedPolicy = SchedPolicy::deadline(
            Duration::from_millis(10),
            Duration::from_millis(20),
            Duration::from_millis(30),
        );
        assert!(matches!(F, SchedPolicy::Fifo(50)));
        assert!(matches!(RR, SchedPolicy::RoundRobin(99)));
        assert!(matches!(
            DL,
            SchedPolicy::Deadline {
                runtime,
                deadline,
                period
            } if runtime == Duration::from_millis(10)
                && deadline == Duration::from_millis(20)
                && period == Duration::from_millis(30)
        ));
    }

    /// GAP 10: pin `SchedPolicy::default() == Normal` and that
    /// every variant roundtrips through serde unchanged. Default
    /// drift would silently re-class every WorkSpec that omits
    /// `sched_policy`; serde drift would break captured config
    /// replay across the 6 variants (one per scheduling class).
    #[test]
    fn sched_policy_default_is_normal_and_serde_roundtrip_per_variant() {
        let d: SchedPolicy = Default::default();
        assert!(matches!(d, SchedPolicy::Normal));

        let variants = [
            SchedPolicy::Normal,
            SchedPolicy::Batch,
            SchedPolicy::Idle,
            SchedPolicy::Fifo(50),
            SchedPolicy::RoundRobin(99),
            SchedPolicy::Deadline {
                runtime: Duration::from_millis(10),
                deadline: Duration::from_millis(20),
                period: Duration::from_millis(30),
            },
        ];
        for original in &variants {
            let bytes = serde_json::to_vec(original).expect("serialize");
            let restored: SchedPolicy = serde_json::from_slice(&bytes).expect("deserialize");
            assert_eq!(restored, *original, "roundtrip drift for {original:?}");
        }
    }
}
