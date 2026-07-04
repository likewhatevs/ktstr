//! Per-controller cgroup v2 limit structs ([`CpuLimits`],
//! [`MemoryLimits`], [`IoLimits`], [`PidsLimits`]) — the typed
//! knob-bundles attached to [`super::CgroupDef`]. Each maps directly
//! onto one or more `cgroup.*` / `*.max` / `*.weight` files; the
//! corresponding `CgroupDef::*` builder methods (e.g. `cpu_quota`,
//! `memory_max`, `io_weight`, `pids_max`) populate these structs
//! lazily via `get_or_insert_with(*::default)`.
//!
//! [`CpuLimits::default`] is the lone hand-written `Default` impl in
//! this file; see the impl's doc for the kernel-period footgun it
//! avoids.

#[allow(unused_imports)] // referenced by intra-doc links
use super::CgroupDef;

// ---------------------------------------------------------------------------
// Cgroup v2 resource limits
// ---------------------------------------------------------------------------

/// CPU controller limits (`cpu.max` + `cpu.weight`) for a cgroup. All
/// fields default to "inherit from parent" — the framework only writes
/// each knob when its corresponding field is `Some`.
///
/// Set via [`CgroupDef::cpu_quota_pct`] / [`CgroupDef::cpu_quota`] /
/// [`CgroupDef::cpu_weight`] (clear a cap with
/// [`CgroupDef::cpu_unlimited`]). The kernel allows `quota` and `weight`
/// to coexist (per `Documentation/admin-guide/cgroup-v2.rst`,
/// "CPU Interface Files"): `weight` biases relative CPU share inside
/// `period`, `quota` enforces an absolute ceiling. Surfacing both as
/// independent options lets a test author express "this cgroup gets
/// at most 50% of one CPU AND should lose to a heavier sibling under
/// contention" in a single declaration.
///
/// Validation runs at `apply_setup` time — any violation surfaces as
/// `anyhow::bail!` so a misconfigured CgroupDef fails before any
/// worker spawns.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct CpuLimits {
    /// `cpu.max` quota and period in microseconds. `quota = None`
    /// means "max" (no upper bound). `quota = Some(q)` allows the
    /// cgroup `q` µs of CPU time per `period`. `q > period` is
    /// legal: it lets the cgroup use multiple CPUs concurrently
    /// (e.g. quota 200_000 / period 100_000 = up to 2 CPUs of
    /// throughput).
    ///
    /// `period` defaults to 100_000 µs (100 ms) when omitted via
    /// the [`CgroupDef::cpu_quota_pct`] convenience builder. Set
    /// via [`CgroupDef::cpu_quota`] when a non-default period is
    /// needed (e.g. tighter control loops with 10 ms periods for
    /// latency-sensitive scheduler tests).
    pub max_quota_us: Option<u64>,
    /// `cpu.max` period component. Required whenever `max_quota_us`
    /// is `Some`; ignored when `max_quota_us` is `None` (the
    /// framework writes `"max <period>"` so the period stays
    /// recorded for diagnostics).
    pub max_period_us: u64,
    /// `cpu.weight` relative-share weight (range 1..=10000, default
    /// 100). `None` leaves the kernel default in place. Larger
    /// values get a larger share when the parent cgroup's CPU is
    /// contended.
    pub weight: Option<u32>,
}

/// Memory controller limits (`memory.max` / `memory.high` /
/// `memory.low` / `memory.swap.max`). Each field is `None` by
/// default (inherit from parent / no limit).
///
/// Set via [`CgroupDef::memory_max`], [`CgroupDef::memory_high`],
/// [`CgroupDef::memory_low`], [`CgroupDef::memory_unlimited`],
/// [`CgroupDef::memory_swap_max`], or
/// [`CgroupDef::memory_swap_unlimited`]. Construct directly only
/// when copying a [`MemoryLimits`] across [`CgroupDef`]s — the
/// builder methods are the preferred entry point because they
/// keep test code in chain position and route the per-knob value
/// through the framework's validation seam at `apply_setup`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct MemoryLimits {
    /// `memory.max` hard ceiling in bytes. Crossing this triggers
    /// the cgroup OOM killer per `Documentation/admin-guide/
    /// cgroup-v2.rst`'s "Memory Interface Files". `None` writes
    /// `"max"` (no hard limit).
    pub max: Option<u64>,
    /// `memory.high` soft throttle threshold in bytes. Crossing
    /// this triggers reclaim throttling but NOT OOM-kill. `None`
    /// writes `"max"`.
    pub high: Option<u64>,
    /// `memory.low` soft protection threshold in bytes. The kernel
    /// preferentially reclaims FROM other cgroups before reclaiming
    /// this cgroup's memory below `low`. `None` writes `"0"` (no
    /// protection).
    pub low: Option<u64>,
    /// `memory.swap.max` ceiling on the cgroup's swap usage in bytes.
    /// `None` writes `"max"` (no swap cap, the kernel default). The
    /// kernel parses the wire value via `page_counter_memparse` —
    /// either the literal `"max"` or a decimal byte count
    /// (`swap_max_write` in `mm/memcontrol.c`).
    ///
    /// # `CONFIG_SWAP=n` kernel detection
    ///
    /// `memory.swap.max` only exists when the kernel was built with
    /// `CONFIG_SWAP=y`; on swap-disabled builds the file is absent
    /// and the wire-time write returns ENOENT. The framework only
    /// emits the write when `swap_max.is_some()` — the explicit
    /// opt-in matches the per-knob semantics of the pids block, so
    /// tests that never call [`CgroupDef::memory_swap_max`] /
    /// [`CgroupDef::memory_swap_unlimited`] succeed verbatim on a
    /// swap-disabled kernel.
    ///
    /// **`swap_max = Some(N)` on a `CONFIG_SWAP=n` kernel surfaces
    /// as a hard scenario failure**: `apply_setup` propagates the
    /// ENOENT from `set_memory_swap_max`'s `write_with_timeout` up
    /// the error chain with the `memory.swap.max` filename in the
    /// context. Test authors who target the swap controller must
    /// either (a) gate the swap_max call on a host probe, or (b)
    /// require the test kernel be built with `CONFIG_SWAP=y` and
    /// document the requirement on the test.
    ///
    /// # ktstr's kernel config and swap
    ///
    /// `ktstr.kconfig` (the project-level kernel-config fragment that
    /// `cargo ktstr` merges into the test kernel's defconfig) does
    /// NOT pin `CONFIG_SWAP=y` — swap is not a test-framework
    /// requirement, and many test scenarios run faster without it.
    /// Tests that call `memory_swap_max` therefore must either
    /// extend the per-test kconfig fragment (passed alongside
    /// `ktstr.kconfig` at kernel-build time) or detect at
    /// scenario-setup time by reading `/proc/swaps` (a missing
    /// file or empty body indicates no swap subsystem) or
    /// `/proc/config.gz` (search for `CONFIG_SWAP=y`). The framework
    /// does NOT auto-detect because host probing is policy that
    /// belongs to the test author, not the workload runner.
    pub swap_max: Option<u64>,
}

/// Pids controller limits (`pids.max`). `None` is the default
/// (inherit from parent — typically `"max"`, no ceiling).
///
/// Per the kernel's `pids_max_write`, existing tasks are NOT killed
/// when the limit lands below the current task count; only future
/// `fork()` / `clone()` calls are blocked once the cgroup's task
/// count meets the limit. Useful for fork-bomb / task-count-ceiling
/// tests.
///
/// # Per-WorkType thread-budget guidance
///
/// `pids.max` counts every task (process AND thread) inside the
/// cgroup. Sizing the limit below the workload's natural task
/// budget produces silent fork failures that surface as
/// `WorkloadConfig`-level workers refusing to start.
///
/// **Most variants spawn exactly one task per worker** — their
/// [`worker_main`](crate::workload) dispatch arm neither spawns
/// helper threads nor forks children. Two exceptions run internal
/// helper threads inside the worker process: `Schbench`
/// (`message_threads` message threads, each spawning
/// `worker_threads` worker threads, plus a control thread) and
/// `Taobench` (`client_threads` client threads + `slow_threads`
/// dispatcher threads); their per-worker task counts are
/// config/CPU-sized, not 1. Per-worker budget therefore depends on
/// [`CloneMode`](crate::workload::CloneMode) (whether each worker
/// is a process or a thread sharing the parent's tgid), the
/// variant's internal helper-thread topology, and whether the
/// variant transiently forks short-lived children inside its own
/// loop. The columns below capture all three:
///
/// | Variant | Steady-state tasks | Transient peak |
/// |---------|--------------------|----------------|
/// | `SpinWait`, `YieldHeavy`, `Mixed` | 1/worker | — |
/// | `Bursty`, `IdleChurn` | 1/worker | — |
/// | `IoSyncWrite`, `IoRandRead`, `IoConvoy` | 1/worker | — |
/// | `CachePressure`, `CacheYield`, `CachePipe` | 1/worker | — |
/// | `PageFaultChurn` | 1/worker | — |
/// | `AffinityChurn`, `PolicyChurn`, `NiceSweep` | 1/worker | — |
/// | `NumaWorkingSetSweep`, `NumaMigrationChurn`, `CgroupChurn` | 1/worker | — |
/// | `Sequence` | 1/worker | — |
/// | `AluHot`, `SmtSiblingSpin`, `IpcVariance` | 1/worker | — |
/// | `PipeIo`, `FutexPingPong`, `AsymmetricWaker`, `SignalStorm` | 1/worker | — |
/// | `FutexFanOut`, `FanOutCompute` | 1/worker | — |
/// | `ThunderingHerd`, `MutexContention`, `WakeChain` | 1/worker | — |
/// | `PriorityInversion`, `ProducerConsumerImbalance` | 1/worker | — |
/// | `RtStarvation`, `PreemptStorm`, `EpollStorm` | 1/worker | — |
/// | `CrossAffinityChurn`, `TimerLatency`, `NetTraffic`, `IrqWake` | 1/worker | — |
/// | `ForkExit` | 1/worker | +1/worker (waitpid'd before next iter) |
/// | `CgroupAttachStorm` | 1/worker | +1/worker (forked child per iter, `_exit`s + auto-reaped) |
/// | `Schbench`, `Taobench` | >1/worker (internal helper threads, config/CPU-sized) | — |
/// | `Custom` | 1/worker | depends on user closure (see below) |
///
/// **`CloneMode::Fork`** (the default): each worker is a separate
/// process placed in the cgroup. The cgroup's task count for one
/// `WorkSpec` is exactly `num_workers`; for `ForkExit` the
/// instantaneous peak is `2 × num_workers` (each parent forks one
/// child, waitpid's, repeats).
///
/// **`CloneMode::Thread`**: every worker is a thread sharing the
/// test runner's tgid. The pids controller counts each thread as
/// a task, so the cgroup's task count for one `WorkSpec` is
/// `num_workers + 1` (workers + the parent task). `ForkExit` is
/// rejected at spawn time under Thread mode (see
/// [`WorkType::ForkExit`](crate::workload::WorkType::ForkExit)).
///
/// **`Custom`**: the framework runs the user closure in a single
/// task per worker (1/worker, identical to every other variant).
/// Any fork/clone the closure issues inside its loop adds to the
/// cgroup's task count for as long as the resulting child lives;
/// `pids.max` must reserve headroom equal to the closure's peak
/// child count per worker. Under `CloneMode::Fork` the framework
/// reaps closure-spawned descendants at teardown via
/// `killpg(worker_pid, SIGKILL)` against the worker's per-process
/// group, so transient children are bounded by the closure
/// itself. Under `CloneMode::Thread` the worker shares the test
/// runner's pgid and `killpg`-based cleanup is unavailable, so
/// the closure owns whatever helpers it spawns and must reap
/// them explicitly before returning the
/// [`WorkerReport`](crate::workload::WorkerReport).
///
/// **Sizing rule**: `pids.max ≥ Σ(steady-state + transient)` for
/// every [`WorkSpec`](crate::workload::WorkSpec) in the cgroup,
/// plus headroom for `cgroup.procs` migration scratch tasks and
/// any payload-binary helper processes the test attaches via
/// [`CgroupDef::workload`] (e.g. `stress-ng` spawns one task per
/// `--cpu N`). Tests with composed `WorkSpec` groups must sum
/// across every group — the framework does NOT auto-derive a
/// budget from the work spec.
///
/// # Parent-cgroup hierarchical charging
///
/// `pids.max` is a per-cgroup ceiling, but every fork/clone
/// charges every ancestor up to (but not including) the
/// unified-hierarchy root. The kernel's `pids_can_fork` calls
/// `pids_try_charge`, which loops
/// `for (p = pids; parent_pids(p); p = parent_pids(p))` and
/// charges each level (kernel/cgroup/pids.c) — root is NOT
/// charged per the loop's `parent_pids(p)` termination
/// condition. EAGAIN propagates from the FIRST level
/// (leaf-to-root traversal order) whose post-charge counter
/// exceeds its limit, so a child cgroup with `pids.max = 1024`
/// still hits EAGAIN when a parent two levels up sits at its
/// own ceiling.
///
/// Sizing rule for nested test trees: the *effective* limit is
/// `min(pids.max)` along the path from the test cgroup up to the
/// pids-controlled root, NOT just the value set on the test
/// cgroup itself. When ktstr runs under a delegated parent slice
/// (systemd `user.slice`, container runtime cgroup, ktstr's own
/// build sandbox), inspect the parent's `pids.max` before sizing
/// the test cgroup — a generous test-cgroup setting is silently
/// shadowed by a tighter ancestor.
///
/// # `pids.max(0)` is rejected at apply_setup, not type-level
///
/// `Some(0)` would silently halt every fork/clone inside the
/// cgroup, including the worker spawn itself for `CloneMode::Fork`
/// and the `ForkExit` per-iteration child fork. The kernel accepts
/// the value (it's a legitimate `pids_max_write` input), so
/// `apply_setup` adds the bail at scenario-setup time; promoting
/// it to a type-level invariant (e.g. `NonZeroU64`) would force
/// every numeric literal through a non-`const` constructor and
/// ripple into every test fixture. The runtime bail keeps the
/// surface ergonomic while still surfacing the foot-cannon at
/// construction time (before any worker spawns).
///
/// Set via [`CgroupDef::pids_max`] or
/// [`CgroupDef::pids_unlimited`]. Construct directly only when
/// copying a [`PidsLimits`] across [`CgroupDef`]s — the builder
/// methods are the preferred entry point because they route the
/// per-knob value through the framework's validation seam at
/// `apply_setup`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct PidsLimits {
    /// `pids.max` task-count ceiling. `None` writes the literal
    /// string `"max"` (the kernel's `PIDS_MAX_STR` sentinel for
    /// unlimited). `Some(n)` writes the decimal `n`. The kernel
    /// rejects negative or `>= PIDS_MAX (PID_MAX_LIMIT + 1, typically ~4M on 64-bit)` values with
    /// EINVAL; the framework's `apply_setup` rejects `Some(0)`
    /// before the syscall (a 0 limit silently halts every fork
    /// or clone inside the cgroup, blocking both worker spawn
    /// under `CloneMode::Fork` and `ForkExit`'s per-iteration
    /// child fork).
    pub max: Option<u64>,
}

/// IO controller limits (`io.weight`). Per-device throughput caps
/// (`io.max`) are intentionally not surfaced here — the per-device
/// interface needs major:minor device-id lookup which has no
/// in-tree consumer; surface it when a concrete use case lands.
///
/// Set via [`CgroupDef::io_weight`]. Construct directly only when
/// copying an [`IoLimits`] across [`CgroupDef`]s — the builder
/// method is the preferred entry point because it routes the
/// per-knob value through the framework's validation seam at
/// `apply_setup`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct IoLimits {
    /// `io.weight` relative-share weight (range 1..=10000, default
    /// 100). `None` leaves the kernel default in place.
    pub weight: Option<u16>,
}

impl Default for CpuLimits {
    /// `cpu.max` quota off, period 100 ms (the kernel default for
    /// `cpu.max`'s second column), `cpu.weight` unset. Matches the
    /// initial state used by the four `CgroupDef::cpu_*` builders;
    /// changing the default period only edits here.
    ///
    /// The derived `Default` would produce `max_period_us: 0` which
    /// `apply_setup` rejects (kernel requires period > 0). Manual
    /// impl avoids that footgun for `..Default::default()` callers.
    fn default() -> Self {
        Self {
            max_quota_us: None,
            max_period_us: 100_000,
            weight: None,
        }
    }
}
