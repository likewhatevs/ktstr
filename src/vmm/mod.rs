//! Virtual machine monitor for booting Linux kernels in KVM to host
//! scheduler test scenarios.
//!
//! The entry point is [`KtstrVm::builder()`], which returns a
//! [`KtstrVmBuilder`] for configuring the kernel, init binary,
//! virtual topology, memory, host-side performance options, and
//! monitor thresholds. Calling `.build()?.run()?` on the result
//! boots the guest and returns a [`VmResult`] containing exit state,
//! captured console, monitor samples, and drained guest messages.
//!
//! See the [VMM architecture
//! page](https://ktstr.dev/guide/architecture/vmm.html)
//! for the boot flow and the [Performance Mode
//! page](https://ktstr.dev/guide/concepts/performance-mode.html)
//! for the isolation options the builder exposes.
//!
//! # Module layout
//!
//! `KtstrVm`'s implementation is split across several files. mod.rs
//! holds the canonical [`KtstrVm`] struct definition, [`KtstrVm::run`]
//! and [`KtstrVm::run_interactive`] entry points, and the public
//! re-exports. The remaining methods are reopened from the children
//! below via additional `impl KtstrVm` blocks:
//!
//! - [`builder`] — [`KtstrVmBuilder`], its `Default`, every setter,
//!   `build()`, and the host-resource acquisition helpers.
//! - [`setup`] — boot pipeline: virtio-blk init, KVM creation,
//!   initramfs resolution / compression / load, x86_64 + aarch64
//!   memory and FDT setup, vCPU register configuration.
//! - [`freeze_coord`] — run-loop orchestration: AP thread spawn,
//!   freeze coordinator, BPF map writer, BSP run loop, and result
//!   collection.
//! - [`contention`] — KVM EINTR retry policy, `HostResourceSnapshot`,
//!   `map_transient_to_contention`, and `create_vm_with_retry`.
//! - [`initramfs_cache`] — persistent content-addressed initramfs parts,
//!   cross-process transform election, and direct-COW range planning.
//! - [`vcpu`] — vCPU thread infrastructure: `ImmediateExitHandle`,
//!   signal handler, thread pinning, RT priority, and perf capture.
//! - [`result`] — [`VmResult`], [`KvmStatsTotals`], `VmRunState`.

// `pub mod` — public sub-API surface that downstream callers may name
// directly. The arch-conditional modules (`aarch64`, `x86_64`) are
// also `pub` but live below where the cfg-gated re-exports for
// their contents are kept together.
//
// `disk_template` is `pub` for rustdoc cross-link visibility — its
// items are referenced from `disk_config`, `rust_init`, and the
// `KtstrVmBuilder::disk` doc as the canonical home for the disk
// template lifecycle. Downstream test authors do not call into it
// directly (the public path is via `KtstrVmBuilder::disk` plus the
// `Filesystem` enum), but rustdoc requires the module path to be
// reachable for the existing intra-doc-links to resolve.
pub mod blobs;
pub mod cgroup_sandbox;
pub mod console;
pub mod disk_config;
pub mod disk_template;
pub mod host_topology;
pub mod initramfs;
pub(crate) mod kvm_stats;
pub mod topology;
#[cfg(feature = "wprof")]
pub mod wprof;

// `pub(crate) mod` — crate-internal sub-modules.
pub(crate) mod builder;
pub(crate) mod capture_numa;
pub(crate) mod capture_scx;
pub(crate) mod capture_tasks;
pub(crate) mod cast_analysis_load;
pub(crate) mod contention;
pub(crate) mod exit_dispatch;
pub(crate) mod freeze_coord;
pub(crate) mod initramfs_cache;
pub(crate) mod net_config;
pub(crate) mod numa_mem;
// The virtio-PCI transport (host bridge, ECAM/CAM decode, config space) is
// x86_64-only. aarch64 references only the `PciBus` TYPE — as an always-`None`
// `Option` in the arch-neutral run loop — so the module compiles there but its
// host-bridge/ECAM machinery is never called; allow that dead code on non-x86
// (x86_64 lints the module fully — the allow is gated off there).
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
pub(crate) mod pci;
pub(crate) mod result;
pub(crate) mod rust_init;
pub(crate) mod sched_stats;
pub(crate) mod setup;
pub(crate) mod vcpu;
pub(crate) mod virtio_blk;
pub(crate) mod virtio_console;
// Shared MSI-X interrupt-delivery state (`MsixState`/`IrqSource`/`MsixRouteSink`)
// for the virtio PCI facades (net + blk). The device cores use `MsixState` on all
// arches, but its PCI-only constructors (`new`/`set_eventfd`) are wired only on
// x86_64, so allow their dead code on non-x86 (x86_64 lints the module fully).
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
pub(crate) mod virtio_msix;
pub(crate) mod virtio_net;

// Bulk transport modules. The wire format (`wire`), the host-side
// streaming assembler (`bulk`), the guest-side typed senders
// (`guest_comms`), and the host-side typed consumers (`host_comms`)
// each carry a single responsibility. Production data (STIMULUS /
// EXIT / SCHED_EXIT / PAYLOAD_METRICS /
// SCENARIO_*) flows through the virtio-console port-1 TLV stream,
// and crash diagnostics travel via COM2.
pub(crate) mod bulk;
pub(crate) mod guest_comms;
pub(crate) mod host_comms;
pub mod wire;

// `mod` — file-private helpers.
mod memory_budget;
mod pi_mutex;
mod terminal;
mod vcpu_panic;
mod vmlinux;

// Re-export the snapshot types for users who hold a [`VmResult`]:
// `VmResult::virtio_blk_counters` and `virtio_net_counters` expose
// `Option<VirtioBlkCountersSnapshot>` / `Option<VirtioNetCountersSnapshot>`
// (plain-u64 frozen views taken at result-construction time), and
// the types themselves must be reachable from the public path for
// a user to spell them out in their own signatures (e.g. a
// `post_vm` helper `fn check(s: &VirtioBlkCountersSnapshot)`). The
// defining modules stay `pub(crate)` because the device
// implementations are internal — these snapshot re-exports are
// the single public surface for the post-mortem counter views.
// `NetConfig` is the builder-side configuration type, surfaced for
// the same public-spelling reason. The in-tree readers go through
// the prelude path so the lib build sees no direct readers of
// these names; allow unused-imports locally to keep `cargo check`
// quiet while preserving the public re-export.
#[allow(unused_imports)]
pub use net_config::NetConfig;
// `VirtioBlkCountersSnapshot` is read by the result.rs test
// fixture (cfg(test)); the lib build still sees no direct reader
// outside the prelude path, so the lint behaves the same as the
// net side. Both `pub use` lines below carry their own
// `#[allow(unused_imports)]` so a future field-name swap doesn't
// silently re-enable the warning for only one of them.
#[allow(unused_imports)]
pub use virtio_blk::VirtioBlkCountersSnapshot;
#[allow(unused_imports)]
pub use virtio_net::VirtioNetCountersSnapshot;

// Re-export public result types from the new submodule.
// `KVM_INTERESTING_STATS` is part of the public surface for stats
// tooling — sidecar consumers reference these names by content, not
// by importing the constant, so the lib build sees no in-tree
// readers. Allow the unused-import lint locally to keep cargo check
// quiet while preserving the public re-export.
pub use builder::KtstrVmBuilder;
#[allow(unused_imports)]
pub use result::KVM_INTERESTING_STATS;
pub use result::{
    BodyContentionWindow, ContentionWitness, GuestLifecyclePhase, HostVcpuSchedstat,
    KvmStatsTotals, PerPhaseSchedstat, VmResult, WatchdogKillReason,
};
#[allow(unused_imports)]
pub use sched_stats::{SchedStatsClient, SchedStatsError, StatsRequest, StatsResponse};

pub(crate) use contention::{
    create_vm_with_retry, host_resource_snapshot, map_transient_to_contention,
};
pub(crate) use pi_mutex::PiMutex;
pub(crate) use terminal::TerminalRawGuard;
pub(crate) use vcpu::{
    BpfMapWriteParams, ImmediateExitHandle, WatchBpfMapParams, register_vcpu_signal_handler,
    set_thread_cpumask, vcpu_signal,
};
pub(crate) use vmlinux::{cached_vmlinux_artifacts, cached_vmlinux_bytes, find_vmlinux};

#[cfg(target_arch = "aarch64")]
pub mod aarch64;
#[cfg(target_arch = "x86_64")]
pub mod x86_64;

/// Lower bound for canonical kernel-half virtual addresses, valid
/// on both x86_64 (4-level paging) and aarch64 (VA_BITS=48). On
/// x86_64 5-level paging the canonical lower bound is
/// `0xFF00_0000_0000_0000`, but kernel symbols always live in the
/// classic upper half so `0xFFFF_8000_0000_0000` is the right
/// conservative threshold for "this looks like a kernel KVA"
/// checks (KERN_ADDRS payload validation, watchpoint arm
/// pre-checks, etc.). x86_64's `msr_kaslr::KERNEL_HALF_CANONICAL_4LEVEL`
/// is the same value retained under the more specific name for
/// the LSTAR-derive path; this cross-arch alias exists so
/// freeze_coord and related callers don't need to reach into
/// `x86_64::msr_kaslr` (which is x86-only and breaks aarch64
/// builds).
pub(crate) const KERNEL_HALF_CANONICAL: u64 = 0xFFFF_8000_0000_0000;

// `acpi`, `boot`, `mptable` are re-exported as part of the public arch
// surface for downstream tooling. mod.rs itself does not consume them
// directly (boot/setup pipeline lives in `setup.rs` and reaches them
// via `super::x86_64::{...}`), so `unused_imports` would otherwise fire.
#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
pub use x86_64::acpi;
#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
pub use x86_64::boot;
#[cfg(target_arch = "x86_64")]
pub use x86_64::kvm;
#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
pub use x86_64::mptable;

#[cfg(target_arch = "aarch64")]
#[allow(unused_imports)]
pub use aarch64::boot;
#[cfg(target_arch = "aarch64")]
pub use aarch64::kvm;

/// Arch-neutral handle for the userspace IOAPIC. On x86_64 it is the real
/// device handle ([`x86_64::kvm::IoapicHandle`], the split-irqchip /
/// \>255-vCPU path); on other targets it is uninhabited, so the run loop's
/// `Option<&IoapicHandle>` is always `None` and the (x86-only) IOAPIC
/// dispatch arms are never compile-reached.
#[cfg(target_arch = "x86_64")]
pub(crate) use x86_64::kvm::IoapicHandle;
#[cfg(not(target_arch = "x86_64"))]
pub(crate) enum IoapicHandle {}

/// Full-irqchip shared GSI route owner ([`x86_64::kvm::FullIrqchipRouteOwner`]).
/// x86-only: it owns the device MSI-X routes on the in-kernel-irqchip path and is
/// referenced solely from x86-gated run/setup code, so there is no aarch64
/// placeholder (the GIC routes device IRQs there).
#[cfg(target_arch = "x86_64")]
pub(crate) use x86_64::kvm::FullIrqchipRouteOwner;

pub use topology::Topology;

use anyhow::{Context, Result};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Start of the guest physical address space used for RAM.
/// x86_64: PA 0 (sub-1MB legacy regions share the same PA space).
/// aarch64: device MMIO below DRAM_START, RAM above.
#[cfg(target_arch = "x86_64")]
pub(crate) const DRAM_BASE: u64 = 0;

#[cfg(target_arch = "aarch64")]
pub(crate) const DRAM_BASE: u64 = kvm::DRAM_START;

// ---------------------------------------------------------------------------
// KtstrVm — builder + run
// ---------------------------------------------------------------------------

/// Builder for creating and running VMs with custom topologies.
///
/// Methods are split across multiple files via additional
/// `impl KtstrVm` blocks: the boot pipeline lives in [`setup`]
/// (init_virtio_blk, setup_memory, setup_vcpus, plus aarch64
/// counterparts), and the run-loop orchestration lives in
/// [`freeze_coord`] (run_vm, spawn_ap_threads, start_monitor,
/// start_bpf_map_write, run_bsp_loop, collect_results).
pub struct KtstrVm {
    pub(crate) kernel: PathBuf,
    /// Whether the guest exposes the virtio-PCI transport (PCI host bridge +
    /// ECAM/CAM config access + the PCI ACPI tables). Off by default;
    /// propagated to `KtstrKvm::pci_enabled` and gates the `pci=off` cmdline
    /// token. x86_64-only in effect: aarch64 has no PCI transport, so the flag
    /// is never propagated or read there — it stays for a uniform builder API.
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    pub(crate) pci_enabled: bool,
    pub(crate) init_binary: Option<PathBuf>,
    pub(crate) scheduler_binary: Option<PathBuf>,
    /// Additional schedulers packed into the initramfs alongside
    /// `scheduler_binary` so the scheduler-lifecycle Ops can swap
    /// schedulers mid-experiment. Each entry's binary is packed
    /// into the BASE archive at
    /// `staging/schedulers/<name>/scheduler` (cache-amortized via
    /// `BaseKey`'s content hash); its `sched_args` rides the SUFFIX
    /// at `staging/schedulers/<name>/sched_args` so per-run argv
    /// changes don't invalidate the base cache.
    ///
    /// The materialized
    /// [`StagedScheduler`](crate::vmm::builder::StagedScheduler)
    /// shape is held here so `spawn_initramfs_resolve` can hand the
    /// binary paths to
    /// [`BaseKey::new`](crate::vmm::initramfs_cache::BaseKey) on
    /// the resolve thread. The companion
    /// [`Self::staged_sched_args_packed`] holds the same `(name,
    /// args)` view in a borrow-friendly tuple form so
    /// [`Self::suffix_params`] can borrow a slice typed `&[(String,
    /// Vec<String>)]` without leaking the pub(crate) `StagedScheduler`
    /// type into the `pub` `SuffixParams` field signature.
    pub(crate) staged_schedulers: Vec<crate::vmm::builder::StagedScheduler>,
    /// Pre-materialized `(name, args)` view of
    /// [`Self::staged_schedulers`] so [`Self::suffix_params`] can
    /// borrow the slice directly without an intermediate allocation
    /// per `suffix_params()` call. Cloned from `staged_schedulers`
    /// once at [`KtstrVmBuilder::build`](crate::vmm::builder::KtstrVmBuilder::build)
    /// time; the duplication is bounded (name + args, no binary
    /// path) and stays in sync because both fields are immutable
    /// after build.
    pub(crate) staged_sched_args_packed: Vec<(String, Vec<String>)>,
    pub(crate) run_args: Vec<String>,
    pub(crate) sched_args: Vec<String>,
    /// Per-test framework-owned parent cgroup for every workload
    /// cgroup the test author declares via `ctx.cgroup_def(...)`.
    /// When `Some`, the guest init mkdir's `/sys/fs/cgroup{path}`
    /// BEFORE the scheduler starts and the guest-side
    /// `CgroupManager` resolves its root to that path. When
    /// `None`, the guest falls back to the legacy resolution
    /// (`--cell-parent-cgroup` in `/sched_args` → default
    /// `/sys/fs/cgroup/ktstr`). Sourced from
    /// `KtstrTestEntry::workload_root_cgroup`; never touches the
    /// scheduler argv.
    pub(crate) workload_root_cgroup: Option<String>,
    /// Cgroup the SCHEDULER is placed in. Sourced from
    /// `Scheduler::cgroup_parent`. When `Some`, the guest init
    /// mkdir's `/sys/fs/cgroup{path}` + enables `+cpuset +cpu` on
    /// every ancestor's `subtree_control` BEFORE starting the
    /// scheduler. Distinct from [`Self::workload_root_cgroup`]
    /// (workload placement) and from `--cell-parent-cgroup` in
    /// scheduler argv (cell-aware schedulers interpret that flag
    /// independently of where the framework places the scheduler).
    pub(crate) scheduler_cgroup_parent: Option<String>,
    pub(crate) topology: Topology,
    /// Guest vCPU count (topology llcs*cores*threads). Stamped onto the
    /// sidecar so `cargo ktstr stats` can pair/compare runs by their
    /// (vcpus, cpu_budget) and flag overcommit (cpu_budget < vcpus).
    pub(crate) vcpus: u32,
    /// Effective host-CPU budget the vCPU threads actually run on: the
    /// Build-time ESTIMATE of the host-CPU budget the vCPU threads will
    /// run on: the reserved plan's CPU count under no-perf-mode; the guest
    /// vCPU count under perf-mode (always) and under the deferred default
    /// WHEN host sysfs topology is readable (both then attempt a 1:1
    /// hard-pin of each vCPU thread to one host CPU). Falls back to the
    /// process's full allowed cpuset when no affinity is applied: no-perf
    /// bypass, sysfs unreadable, or the deferred default with no cached
    /// host topology. When the deferred default cannot 1:1-pin (host too
    /// small) it overcommits at run time and `run()` overrides
    /// `VmResult.cpu_budget` with the actual masked host-CPU count, so this
    /// estimate reaches the sidecar only when it held. Below `vcpus` means
    /// the host time-slices the guest's vCPUs (overcommit), confounding the
    /// guest-scheduler timing metrics.
    pub(crate) effective_cpu_budget: u32,
    /// Guest memory in MiB. `None` = deferred: computed from actual
    /// initramfs size after the initramfs build completes.
    pub(crate) memory_mib: Option<u32>,
    /// Minimum memory in MiB for deferred allocation. When non-zero,
    /// the deferred path uses `max(computed, memory_min_mib)` so topology
    /// configs that need more memory than the initramfs floor are honored.
    pub(crate) memory_min_mib: u32,
    pub(crate) cmdline_extra: String,
    pub(crate) timeout: Duration,
    /// Thresholds for reactive SysRq-D dump. When set and the monitor
    /// detects a sustained violation, it writes the dump flag to guest SHM.
    pub(crate) monitor_thresholds: Option<crate::monitor::MonitorThresholds>,
    /// Override for `scx_sched.watchdog_timeout` in the guest kernel.
    /// Converted to jiffies via CONFIG_HZ at monitor start time and
    /// written at each monitor iteration after the scheduler attaches.
    pub(crate) watchdog_timeout: Option<Duration>,
    /// Override for the freeze coordinator's per-rendezvous wait
    /// timeout. `None` means use `FREEZE_RENDEZVOUS_TIMEOUT` (30 s) from
    /// `src/vmm/freeze_coord/state.rs`. Lowering this drives the
    /// rendezvous-timeout Degraded emit path for E2E coverage of the
    /// silent-drop fixes; production callers should not override.
    pub(crate) rendezvous_timeout: Option<Duration>,
    /// Host-side BPF map writes. Empty slice disables the thread.
    /// When non-empty, a thread polls for BPF map discoverability,
    /// waits for scenario start via SHM ring, then writes each
    /// `u32` value at its specified map/offset. All writes complete
    /// before the guest is signaled via SHM slot 0, so the guest
    /// sees a single unblock regardless of how many writes ran.
    pub(crate) bpf_map_writes: Vec<BpfMapWriteParams>,
    /// Named scheduler BPF-map fields to read observer-effect-free from the
    /// free-running monitor into run-level metrics (lowered from the test
    /// entry's `watch_bpf_maps`). The monitor resolves + reads each lazily
    /// after the scheduler attaches; values surface via
    /// [`crate::vmm::result::VmResult::run_metric`] under
    /// `<scheduler-obj>_<label>` / `_<label>_{avg,max}`.
    pub(crate) watch_bpf_maps: Vec<WatchBpfMapParams>,
    /// Performance mode: vCPU pinning to host LLCs, hugepage-backed
    /// guest memory, NUMA mbind, and RT scheduling on both
    /// architectures. On x86_64, additionally: KVM_HINTS_REALTIME
    /// CPUID hint, PAUSE and HLT VM exit disabling via
    /// KVM_CAP_X86_DISABLE_EXITS, and KVM_CAP_HALT_POLL skipped
    /// (guest haltpoll cpuidle disables host halt polling via
    /// MSR_KVM_POLL_CONTROL). Oversubscription validation at build
    /// time on both architectures.
    pub(crate) performance_mode: bool,
    /// Whether the builder was invoked with `no_perf_mode(true)`. The
    /// flag is needed at `run()` time so the lock-acquisition switch
    /// can distinguish "no-perf-mode bypass / degraded-sysfs" (no
    /// locks, no plans, no acquire) from the default-else path that
    /// takes an LLC `LOCK_SH` set via `compute_pinning` +
    /// `acquire_resource_locks` (per LLC offset) whenever neither
    /// `performance_mode` nor `no_perf_mode` is in effect.
    /// Persisted on KtstrVm so the deferred-lock contract in
    /// `KtstrVm::run` does not need to re-read the env every spawn.
    pub(crate) no_perf_mode: bool,
    /// Pinning plan computed during build() when performance_mode is
    /// enabled. The flock fds carried by the plan are stripped at
    /// build time — `KtstrVm::run` re-acquires the LLC flocks via
    /// [`host_topology::acquire_resource_locks_waiting`] at the TOP
    /// of the run (before initramfs/KVM/kernel setup, so a queued
    /// cell holds no guest resources) and releases them on return, so
    /// concurrent peers see the LLCs free as soon as the run
    /// completes (and the window between `build()` and `run()`
    /// carries no locks). The `assignments` /
    /// `service_cpu` / `llc_indices` payload is what `setup` and
    /// `freeze_coord` consume during the run.
    pub(crate) pinning_plan: Option<host_topology::PinningPlan>,
    /// Per-guest-NUMA-node host NUMA nodes for mbind. Indexed by guest
    /// node ID. Each entry is the set of host NUMA nodes that the guest
    /// node's vCPUs are pinned to. Empty when performance_mode is off.
    pub(crate) mbind_node_map: Vec<Vec<usize>>,
    /// No-perf-mode resource plan. Populated for every no-perf-mode
    /// VM — either the operator-set CPU count
    /// (`--cpu-cap N` / `KTSTR_CPU_CAP=N`) or the 30%-of-allowed
    /// default when neither is present.
    ///
    /// The plan's build-time `LOCK_SH` flock fds are HELD, not
    /// stripped — but they live on [`Self::no_perf_build_locks`], NOT on
    /// this plan: `build()` moves them off `LlcPlan.locks` into that
    /// dedicated slot so `acquire_run_locks` (`&self`) can release them
    /// once the run-time replan adopts its own locks. Holding them is
    /// what makes a concurrent peer's DISCOVER of these LLCs observe a
    /// truthful holder count. `build()` used to strip them immediately
    /// and re-plan-free at run time; that left the holder-count feedback
    /// dead, so a concurrent Spread sweep saw zero holders and
    /// re-clustered every cell onto the same low LLCs. Now `run()`'s
    /// `acquire_run_locks` re-plans against the (now-truthful) live
    /// holder counts, adopting the fresh plan and its fds BEFORE the
    /// build-time fds in `no_perf_build_locks` drop so peers never see an
    /// empty window (see [`Self::no_perf_effective_cap`]). The plan's
    /// `cpus` slice still drives the build-time `setup` paths that must
    /// know a no-perf CPU mask before `run()` — the run-time replan then
    /// threads its refreshed CPUs to those same consumers so mask + lock
    /// identity holds by construction.
    ///
    /// `None` only in the degraded-sysfs case (no-perf-mode on a
    /// host whose `/sys/devices/system/cpu` cannot be read AND no
    /// explicit cap was set — the build bails with an error when
    /// a cap IS set under the same sysfs failure), and for
    /// perf-mode (which uses `pinning_plan`). The two paths are
    /// orthogonal — perf-mode hard-pins single CPUs, --cpu-cap
    /// soft-masks a pool.
    #[allow(dead_code)]
    pub(crate) no_perf_plan: Option<host_topology::LlcPlan>,
    /// Effective `CpuCap` the build-time [`Self::no_perf_plan`] targeted.
    /// `acquire_run_locks` re-plans at run time against exactly this
    /// budget — reusing the build-time cap rather than recomputing keeps
    /// the run-time reservation the same SIZE the build sized memory /
    /// warnings against, even if the process's allowed set shifted since
    /// build. `None` whenever `no_perf_plan` is `None` (degraded sysfs /
    /// bypass) and for perf-mode; those paths never replan.
    #[allow(dead_code)]
    pub(crate) no_perf_effective_cap: Option<host_topology::CpuCap>,
    /// Build-time `LOCK_SH` flock fds for the no-perf [`Self::no_perf_plan`],
    /// moved off `LlcPlan.locks` at `build()` time so they can be released
    /// under a shared borrow.
    ///
    /// These are the reservations that let a concurrent peer's DISCOVER
    /// see a truthful holder count on the LLCs this VM planned against.
    /// They are held from plan time through the setup window, then dropped
    /// the instant `acquire_run_locks`' no-perf replan arm has acquired its
    /// OWN fresh locks — acquire-before-release, so retained LLCs never
    /// flicker free. Kept in a `Mutex` (not on the plan) because
    /// `acquire_run_locks` takes `&self`: the plan itself is immutable
    /// there, but the interior-mutable slot lets the replan `mem::take` and
    /// drop the stale fds without waiting for `KtstrVm` drop. Were they left
    /// on the plan, a cell whose replan moved to different LLCs would
    /// double-report — peers counting its pid on both the abandoned
    /// build-time LLCs and the adopted run-time LLCs for the whole run.
    ///
    /// Empty (a `Mutex<Vec<_>>` holding no fds) whenever `no_perf_plan` is
    /// `None` (degraded sysfs / bypass) and for perf-mode / the deferred
    /// default; those paths never populate it.
    #[allow(dead_code)]
    pub(crate) no_perf_build_locks: std::sync::Mutex<Vec<std::os::fd::OwnedFd>>,
    /// Cached host topology snapshot read once at `build()` time
    /// from `/sys/devices/system/cpu`. `KtstrVm::run`'s default-else
    /// branch threads this through `compute_pinning` per LLC offset,
    /// then takes `LOCK_SH` via `acquire_resource_locks`, without
    /// re-reading sysfs. `None` only on
    /// the degraded-sysfs no-perf-mode branch, where no LLC
    /// reservation is possible to begin with — `KtstrVm::run`
    /// short-circuits to "no locks" in that case.
    #[allow(dead_code)]
    pub(crate) host_topo: Option<host_topology::HostTopology>,
    /// Shell commands to run in the guest to enable a kernel-built scheduler.
    pub(crate) sched_enable_cmds: Vec<String>,
    /// Shell commands to run in the guest to disable a kernel-built scheduler.
    pub(crate) sched_disable_cmds: Vec<String>,
    /// Files to include in the guest initramfs at their archive paths.
    /// Each entry is (archive_path, host_path).
    pub(crate) include_files: Vec<(String, PathBuf)>,
    /// Raw (already-decompressed) `.ko` images the guest loads via
    /// `finit_module(2)` before touching any virtio device, in load
    /// order. Threaded verbatim into [`initramfs::SuffixParams::kernel_modules`]
    /// by [`Self::suffix_params`]; empty for ktstr-built kernels
    /// (virtio =y), populated for prebuilt distro kernels (virtio =m).
    /// Stored as an independently content-addressed initramfs part, so cells
    /// with the same module set reuse both archive construction and
    /// compression.
    pub(crate) kernel_modules: Vec<PathBuf>,
    /// Initrd compression the guest kernel can unpack. LZ4 for
    /// ktstr-built kernels; prebuilt distro kernels without
    /// `CONFIG_RD_LZ4` get a format from their own `CONFIG_RD_*` set
    /// (see [`initramfs::InitrdCompression`]). Every supported format uses
    /// the persistent prepared-initrd CAS and direct-COW loader.
    pub(crate) initrd_compression: initramfs::InitrdCompression,
    /// The optional single virtio-blk disk, rendered as `/dev/vda`. ktstr
    /// wires one blk device; multi-disk would be N PCI functions (re-addable
    /// pre-1.0). The backing file is produced by the template-VM lifecycle
    /// (one-time guest-side `mkfs.<fstype>` against a sparse image, cached
    /// alongside the kernel; per-test reflink-copy at fan-out). Per-test
    /// boots populate the backing via the `Raw` tempfile or `Btrfs`
    /// cache-clone branches in [`KtstrVm::open_blk_backing`]; the
    /// disk-template-build VM driver overrides both branches via
    /// [`Self::template_staging_image`] so it can format a host-staged
    /// image without re-entering its own cache.
    pub(crate) disk: Option<disk_config::DiskConfig>,
    /// Network devices. Empty skips virtio-net entirely (no NIC function /
    /// FDT node, no IRQ). Each entry attaches one virtio-net device whose
    /// backend is the in-VMM loopback (TX bytes echoed back into RX). On
    /// x86_64 each is a virtio-pci function on its own slot/GSI
    /// (`kvm::virtio_net_pci_slot` / `kvm::virtio_net_gsi`, bounded by
    /// `kvm::MAX_VIRTIO_NICS`); aarch64 supports a single virtio-MMIO NIC.
    pub(crate) networks: Vec<net_config::NetConfig>,
    /// Internal-only override for `init_virtio_blk`'s per-test
    /// backing-file allocation. `Some(path)` makes the device open
    /// `path` directly instead of allocating a fresh `tempfile()`
    /// or invoking [`disk_template::ensure_template`]. Set
    /// exclusively by [`KtstrVmBuilder::template_staging_image`] for
    /// the disk-template-build VM driver in
    /// `disk_template::build_template_via_vm`; `None` for every
    /// other code path. See the builder field's doc for the full
    /// recursion-break rationale.
    pub(crate) template_staging_image: Option<PathBuf>,
    /// Busybox bytes packed at `bin/busybox`. `None` skips packing;
    /// `Some(bytes)` writes the provided bytes. Sourced from
    /// [`crate::vmm::blobs::load_busybox_bytes`].
    pub(crate) busybox_bytes: Option<Vec<u8>>,
    #[cfg(feature = "wprof")]
    pub(crate) wprof: Option<crate::vmm::wprof::WprofConfig>,
    /// Forward COM1 (kernel console) to stderr in real-time during
    /// interactive shell mode. Useful for watching virtio probe and
    /// kernel messages alongside the shell session.
    pub(crate) dmesg: bool,
    /// Command to execute non-interactively in shell mode (--exec).
    /// Passed to the guest via /exec_cmd in the initramfs.
    pub(crate) exec_cmd: Option<String>,
    /// Wall-clock bound for a shell `--exec` payload before the VM is
    /// force-killed. A panic-less guest hang otherwise blocks the BSP
    /// run loop ~forever; the `run_interactive` watchdog kicks the vCPU
    /// after this deadline. Consulted only in exec_mode runs.
    pub(crate) exec_timeout: Duration,
    /// Optional host path to `ktstr-jemalloc-probe`. When `Some`, the
    /// probe is packed into the guest initramfs as an extra binary at
    /// `bin/ktstr-jemalloc-probe`. Consumed by `spawn_initramfs_resolve`.
    pub(crate) jemalloc_probe_binary: Option<PathBuf>,
    /// Optional host path to `ktstr-jemalloc-alloc-worker`. When
    /// `Some`, the worker is packed alongside the probe as an
    /// extra. The cross-process closed-loop test in
    /// `tests/jemalloc_probe_tests.rs` spawns it as a background
    /// payload and probes its pid.
    pub(crate) jemalloc_alloc_worker_binary: Option<PathBuf>,
    /// Where the freeze coordinator writes the JSON-pretty
    /// `monitor::dump::FailureDumpReport` when an error-class
    /// SCX exit fires. `None` disables the file sink (the dump
    /// still goes to `tracing::error` regardless). The test
    /// framework sets this to a per-test path under the run's
    /// sidecar directory so operators find the structured JSON
    /// alongside `*.ktstr.json` without needing an env var; CLI /
    /// library callers that want the dump on disk set the path
    /// explicitly via [`KtstrVmBuilder::failure_dump_path`].
    pub(crate) failure_dump_path: Option<PathBuf>,
    /// Capture two BPF-state snapshots per VM run: an early one when
    /// the host-side `runnable_at` scanner observes any task with
    /// `jiffies - p->scx.runnable_at > watchdog_timeout/2`
    /// (mirrors the kernel's `check_rq_for_timeouts`), and a late
    /// one at the same `ktstr_err_exit_detected` latch as the
    /// single-snapshot path. Emits
    /// `monitor::dump::DualFailureDumpReport` instead of the
    /// single-snapshot `FailureDumpReport`. Only the late snapshot
    /// is required — the early one is `None` when the stall fires
    /// before the half-way threshold trips, and the file is not
    /// written at all when only the early snapshot is captured (the
    /// run completed without a stall, so the early snapshot is not
    /// useful as a standalone artifact).
    ///
    /// Set by `crate::test_support::probe::attempt_auto_repro` for
    /// the repro VM only. Primary VMs leave this `false`; their
    /// freeze coordinator emits a `monitor::dump::FailureDumpReport`
    /// directly, matching the existing single-snapshot behaviour.
    pub(crate) dual_snapshot: bool,
    /// Workload time budget. When set, the host-side watchdog
    /// resets its hard deadline to `now + workload_duration` the
    /// first time the monitor observes `*scx_root` transition from
    /// null to non-null in guest memory — so boot + BPF verifier
    /// time do not eat into the workload's budget. Bounded above
    /// by the original `timeout`-derived deadline (the watchdog
    /// uses `min(reset, original)`). `None` disables the reset and
    /// the watchdog uses `timeout` as a single deadline counted
    /// from VM boot. Populated via
    /// [`KtstrVmBuilder::workload_duration`].
    #[allow(dead_code)]
    pub(crate) workload_duration: Option<Duration>,
    /// Periodic snapshot count: when non-zero, the freeze
    /// coordinator slices the capturable window into `num_snapshots`
    /// equally-spaced boundaries (a 10 % pre / 10 % post buffer
    /// reserves the ramp edges) and fires a host-side
    /// `freeze_and_dispatch(FreezeMode::Capture { gate_on_exit_kind: false })`
    /// at each one, tagged
    /// `"periodic_NNN"` and stored on the host's
    /// [`crate::scenario::snapshot::SnapshotBridge`]. The window is
    /// `[max(scenario_start, prereqs_ready), scenario_start +
    /// workload_duration]`: the start floats to the prereq-ready
    /// moment so cold-boot accessor/attach latency cannot strand
    /// boundaries in the already-past pre-ready region, and the end
    /// is CLAMPED to the workload end so a cold-boot-late start
    /// cannot push captures into post-workload idle. On a warm boot
    /// the start equals `scenario_start`, so the window is the full
    /// `[scenario_start, scenario_start + workload_duration]`. `0`
    /// (the default) disables the periodic-capture loop entirely.
    /// Plumbed through
    /// [`KtstrVmBuilder::num_snapshots`]; the test-entry plumbing
    /// comes from [`crate::test_support::KtstrTestEntry::num_snapshots`].
    pub(crate) num_snapshots: u32,
    /// Lazy on-demand BPF cast-analysis handle for the scheduler
    /// binary's embedded BPF object(s). Populated by
    /// [`KtstrVmBuilder::build`] with the scheduler binary path
    /// (or `None` for no-scheduler runs). The handle is cheap to
    /// construct — it captures the path in a `OnceLock` slot and
    /// runs no file I/O or analyzer work at builder time. The
    /// actual analyzer (file read + ELF + BTF + register walk)
    /// runs only when
    /// `super::cast_analysis_load::LazyCastMap::get_full` is
    /// first called from the failure-dump path. Tests that pass
    /// without dumping never trigger analyzer work — the dominant
    /// case under nextest's process-per-test execution model.
    ///
    /// `.get_full()` returns the richer
    /// `super::cast_analysis_load::CastAnalysisOutput` which
    /// carries four pieces:
    /// 1. `cast_maps`: a `Vec<Arc<CastMap>>`, one per embedded
    ///    object, each a `(parent_btf_id, member_offset) -> CastHit`
    ///    map the instruction-level analyzer recovered.
    /// 2. `btfs`: every parsed embedded BPF object's program BTF
    ///    (one entry per object inside `.bpf.objs`).
    /// 3. `fwd_index`: a `name -> (btfs index, type_id)` index
    ///    over every complete (`!is_fwd`) struct/union across
    ///    `btfs`, used by the renderer's
    ///    [`crate::monitor::btf_render::MemReader::cross_btf_resolve_fwd`]
    ///    override to chase a `BTF_KIND_FWD` whose body lives in
    ///    a sibling embedded object's BTF.
    /// 4. `alloc_size_types`: unique `(alloc_size, struct_name)`
    ///    pairs captured from `scx_static_alloc_internal` call
    ///    sites, threaded to the renderer as a last-resort fallback
    ///    for deferred-resolve arena chases whose `CastHit` has
    ///    `alloc_size: None`.
    ///
    /// When `.get_full()` fires, results are cached process-wide
    /// by a u64 ahash content hash of the binary bytes so two VMs
    /// in the same
    /// process resolving to the same scheduler binary content
    /// share one analyzer run (auto-repro path, future in-process
    /// multi-test drivers). The cast map is threaded into
    /// [`crate::monitor::dump::DumpContext::cast_map`] and the
    /// `(btfs, fwd_index)` pair into
    /// `crate::monitor::dump::DumpContext::cross_btf_fwd_index` at freeze
    /// time so the failure-dump renderer can promote `u64` fields
    /// the analyzer flagged into typed-pointer renders via
    /// [`crate::monitor::btf_render::MemReader::cast_lookup`] and
    /// recover Fwd-pointee bodies via
    /// [`crate::monitor::btf_render::MemReader::cross_btf_resolve_fwd`].
    ///
    /// `.get_full()` returns `None` for every degraded case (no
    /// scheduler binary, file read failed, analyzer surfaced an
    /// empty cast map AND empty cross-BTF index — no `.bpf.objs`,
    /// BTF parse failure, no recovered casts and no complete
    /// struct/union definitions). All `None` paths render every
    /// `u64` as a plain unsigned counter and skip Fwd pointee
    /// chases with the legacy "forward declaration" skip path,
    /// matching the pre-integration default.
    pub(crate) cast_map: std::sync::Arc<crate::vmm::cast_analysis_load::LazyCastMap>,
}

struct RunLocks {
    #[allow(dead_code)]
    locks: Vec<std::os::fd::OwnedFd>,
    default_cpu_mask: Option<Vec<usize>>,
    pinning_plan: Option<host_topology::PinningPlan>,
    /// Refreshed no-perf CPU list from the run-time replan. `Some` only
    /// on the `no_perf_mode` arm that re-plans against live holder
    /// counts; the run threads it (in preference to the stale
    /// build-time `no_perf_plan.cpus`) to `run_vm`'s affinity-mask
    /// consumers — the vCPU-thread mask, the BSP mask, and the
    /// virtio-blk worker placement — so the mask every consumer applies
    /// matches the LLCs the run-scoped fds in `locks` actually protect.
    /// `None` on every other arm, where the consumers fall back to
    /// `no_perf_plan` / `default_cpu_mask` exactly as before.
    no_perf_cpus: Option<Vec<usize>>,
}

/// Process-wide gate for the VMM's verbose host-side diagnostics.
///
/// Returns `true` when either [`crate::KTSTR_DEBUG_ENV`] or
/// [`crate::RUNNER_DEBUG_ENV`] is set to exactly `"1"`. The per-vCPU
/// affinity-mask lines, the BSP run-loop trace, the `CLEANUP:`
/// teardown timings, the `KtstrVm::run` VM-setup timing lines, and the
/// watchdog lifecycle lines (started / BSP-done / scheduler-attach
/// reset — but not the timeout/kick diagnostics on the failure path)
/// route through this gate so they stay out of normal
/// CI logs — where they otherwise bury the scheduler test failure a
/// run is investigating — yet reappear on demand. `RUNNER_DEBUG=1` is
/// what GitHub Actions exports when a job is re-run with "Enable debug
/// logging", so a failed CI job re-run in debug mode gets the full
/// diagnostics for free.
///
/// The env read is memoized in a [`OnceLock`](std::sync::OnceLock)
/// because these lines are emitted from the hot per-vCPU paths; the
/// value is fixed for the process lifetime.
pub(crate) fn debug_logging_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        [crate::KTSTR_DEBUG_ENV, crate::RUNNER_DEBUG_ENV]
            .iter()
            .any(|k| std::env::var(k).is_ok_and(|v| v == "1"))
    })
}

/// Human-readable summary of device-IRQ routing-install failures, for
/// `run_interactive`'s teardown. `None` when there were none. `n` is a count of
/// `KVM_SET_GSI_ROUTING` installs that errored, each leaving a device IRQ
/// unrouted (the device hangs on first use). Called for BOTH x86 route owners:
/// the split-irqchip userspace IOAPIC (`IoapicHandle::routing_failures`) and the
/// full-irqchip MSI-X route owner (`FullIrqchipRouteOwner::routing_failures`);
/// the body is transport-neutral and summarizes either. Surfaced in interactive
/// mode because the operator's terminal shows the guest console, not the host's
/// per-failure tracing, so an unrouted IRQ would otherwise be a silent device
/// hang. x86-only: both route owners are x86 (on aarch64 the GIC routes device
/// IRQs directly and `IoapicHandle` is an empty-enum placeholder), so there is
/// nothing to summarize.
#[cfg(target_arch = "x86_64")]
fn routing_failure_summary(n: u64) -> Option<String> {
    (n > 0).then(|| {
        format!(
            "WARNING: {n} device-IRQ routing failure(s) during this run \
             (KVM_SET_GSI_ROUTING errored) — affected devices' interrupts \
             were not delivered, so those devices may have hung"
        )
    })
}

impl KtstrVm {
    pub fn builder() -> KtstrVmBuilder {
        KtstrVmBuilder::default()
    }

    /// Borrow this VM's per-invocation initramfs-suffix inputs into an
    /// [`initramfs::SuffixParams`]. Centralizes the `run_args` /
    /// `sched_args` / sched-enable / sched-disable / `exec_cmd`
    /// bundling so both x86_64 and aarch64 paths construct the suffix
    /// from the same source of truth.
    ///
    /// The `staged_sched_args` slot pulls per-name args from the
    /// pre-materialized [`Self::staged_sched_args_packed`] view so
    /// `SuffixParams` can borrow `&[(String, Vec<String>)]` directly
    /// without exposing the [`crate::vmm::builder::StagedScheduler`]
    /// type into the `pub` `SuffixParams` field signature.
    fn suffix_params(&self) -> initramfs::SuffixParams<'_> {
        // Production invariant: the initramfs path is reached only with an
        // init_binary (spawn_initramfs_resolve bails otherwise), so payload
        // is always Some here. A None would make build_suffix emit an
        // /init-less, unbootable image silently — trip it in debug/test.
        debug_assert!(
            self.init_binary.is_some(),
            "suffix_params: production initramfs path requires init_binary (the /init payload)"
        );
        initramfs::SuffixParams {
            payload: self.init_binary.as_deref(),
            args: &self.run_args,
            sched_args: &self.sched_args,
            sched_enable: &self.sched_enable_cmds,
            sched_disable: &self.sched_disable_cmds,
            exec_cmd: self.exec_cmd.as_deref(),
            staged_sched_args: &self.staged_sched_args_packed,
            workload_root_cgroup: self.workload_root_cgroup.as_deref(),
            scheduler_cgroup_parent: self.scheduler_cgroup_parent.as_deref(),
            kernel_modules: &self.kernel_modules,
        }
    }

    /// Boot the VM, run until shutdown/timeout, return captured output.
    pub fn run(&self) -> Result<VmResult> {
        let start = Instant::now();
        let dbg = debug_logging_enabled();

        // Acquire the run-scoped host reservation FIRST — before the
        // initramfs build, KVM VM creation, kernel load, and vCPU
        // setup. Under the lock-dir queue ("second-order scheduler"
        // — see host_topology::protocol), nextest admits every cell
        // at once and cells park here until capacity frees; a queued
        // cell must be CHEAP: what it holds is this process and a
        // blocked flock ticket (plus one inotify fd only while it is
        // queue head) — NOT a guest memory map with a resident kernel
        // image, not vCPU fds. Acquisition WAITS on contention
        // (wait = true) for the authoritative flock release rather
        // than converting a transient peer hold into a host-skip.
        let run_locks = self.acquire_run_locks(true)?;

        let initramfs_handle = self.spawn_initramfs_resolve()?;
        if dbg {
            eprintln!("  initramfs spawn: {:?}", start.elapsed());
        }
        let (mut vm, kernel_result) = self.create_vm_and_load_kernel()?;
        if dbg {
            eprintln!("  kvm+kernel: {:?}", start.elapsed());
        }

        #[cfg(target_arch = "x86_64")]
        let _kernel_result = {
            let kr = self.setup_memory(&mut vm, kernel_result, initramfs_handle)?;
            if dbg {
                eprintln!("  setup_memory (joins initramfs): {:?}", start.elapsed());
            }
            self.setup_vcpus(&vm, kr.entry)?;
            if dbg {
                eprintln!("  setup_vcpus: {:?}", start.elapsed());
            }
            kr
        };
        #[cfg(target_arch = "aarch64")]
        let _kernel_result = {
            let kr = self.setup_memory_aarch64(&mut vm, kernel_result, initramfs_handle)?;
            self.setup_vcpus_aarch64(&vm, kr.entry)?;
            kr
        };

        let stats_ctx = kvm_stats::open_stats_context(&vm.vcpus);
        if stats_ctx.is_none() {
            tracing::debug!("KVM_GET_STATS_FD not supported, skipping stats collection");
        }

        if dbg {
            eprintln!("VM setup total: {:?}", start.elapsed());
        }
        tracing::debug!(elapsed_us = start.elapsed().as_micros(), "total_setup");

        // Run-phase clock approximates the watchdog's hard_deadline
        // (both post-setup; the watchdog computes its deadline slightly
        // later, inside the spawned thread) so the BSP loop and monitor
        // thread don't charge VM setup overhead against the guest's
        // timeout budget. The lock-queue wait happened BEFORE setup
        // (top of this fn), so neither queue time nor setup time is
        // charged against the guest.
        let run_start = Instant::now();

        // Per-VM halt-poll (W2f): keyed on the run-lock outcome now that it is
        // resolved. `default_cpu_mask` is set only on the default-path
        // overcommit fallback (see `build_overcommit_run_locks`), so it is the
        // overcommit signal. `KVM_CAP_HALT_POLL` is settable at any time
        // (api.rst 7.20), so applying it here — after vCPU create, before the
        // run loop — is valid. `None` leaves the host module default.
        if let Some(ns) = setup::halt_poll_policy(
            self.no_perf_mode,
            self.performance_mode,
            run_locks.default_cpu_mask.is_some(),
        ) {
            vm.set_halt_poll(ns);
        }

        let effective_plan = run_locks
            .pinning_plan
            .as_ref()
            .or(self.pinning_plan.as_ref());
        // No-perf run-time replan output: the fresh plan's CPUs, taken in
        // preference to the stale build-time `no_perf_plan.cpus` so every
        // affinity mask inside `run_vm` binds to the LLCs the run-scoped
        // `run_locks.locks` actually hold. `None` on every non-no-perf /
        // degraded arm, where `run_vm` keeps its `no_perf_plan` /
        // `default_cpu_mask` fallback.
        let effective_no_perf_cpus = run_locks.no_perf_cpus.as_deref();
        let run = self.run_vm(
            run_start,
            vm,
            run_locks.default_cpu_mask.as_deref(),
            effective_plan,
            effective_no_perf_cpus,
        )?;
        // The default-overcommit path masks every vCPU thread to a CPU set
        // SMALLER than vcpus (host too small for a 1:1 pin). build()'s
        // effective_cpu_budget assumed this path either 1:1-pins or aborts, so
        // it stamped the full vcpu count; capture the true masked host-CPU
        // count here (clamped >= 1, mirroring the builder.rs sentinel) so the
        // sidecar records the real overcommit and render_overcommit_warning's
        // b<v test fires. None on every non-overcommit path, so the build-time
        // effective_cpu_budget stands.
        let overcommit_budget = run_locks
            .default_cpu_mask
            .as_ref()
            .map(|m| (m.len() as u32).max(1));
        drop(run_locks);

        let mut result = self.collect_results(start, run)?;
        if let Some(budget) = overcommit_budget {
            result.cpu_budget = budget;
        }

        // Read cumulative KVM stats after VM exit.
        if let Some(ctx) = stats_ctx {
            result.kvm_stats = Some(ctx.read_stats());
        }

        // Persist any coverage-profraw the guest /init flushed
        // (`try_flush_profraw`) and the host bucketed into
        // `guest_messages`. Every `KtstrVm::run` caller — the eval
        // path, the auto-repro path, and direct callers — funnels
        // through here, so this is the single profraw-persistence
        // site: the eval/probe per-frame dispatch intentionally does
        // NOT re-extract `Profraw` frames (a second write would make
        // `llvm-profdata merge` double-count the counters).
        if let Some(ref msgs) = result.guest_messages {
            crate::test_support::persist_guest_profraw(msgs);
        }

        Ok(result)
    }

    /// Acquire the run-scoped flock fds the VM needs for the
    /// duration of [`Self::run`] / [`Self::run_interactive`].
    /// For perf-mode and the deferred default, `build()` strips every
    /// flock from the cached pinning / LLC plan and this fn re-takes
    /// them at the TOP of `run()` — before the initramfs build, KVM
    /// VM creation, and kernel load — so a cell parked in the
    /// acquisition queue is CHEAP (a ticket flock, plus one inotify
    /// fd only as head; no guest memory). The no-perf path additionally
    /// HOLDS its build-time `LOCK_SH` fds (parked in
    /// [`Self::no_perf_build_locks`]) so peers' holder counts stay
    /// truthful, releasing them either the instant its replan adopts
    /// fresh locks or BEFORE queueing on contention (a queued waiter
    /// must hold nothing — see the no-perf arm). The returned
    /// `Vec<OwnedFd>` is dropped at the end of the run, releasing
    /// every run-scoped lock for concurrent peers.
    ///
    /// `wait` selects the contention policy: the test path
    /// ([`Self::run`]) passes `true`, so a contended reservation joins
    /// the cross-invocation acquisition queue and completes under the
    /// head license (the resource-budget model: budgeted/shared
    /// acquirers wait for exclusive holders; see
    /// [`host_topology::protocol`]) instead of surfacing contention;
    /// the interactive path passes `false` for the single-shot
    /// fast-path-only behaviour. The holder's VM watchdog and
    /// nextest process rail own lifecycle bounds; the lock layer
    /// never turns a slow live holder into `ResourceContention`.
    ///
    /// Branch table (mirrors `build()`'s plan switch):
    /// * `no_perf_mode` + cached `no_perf_plan`: re-runs the full
    ///   `acquire_llc_plan` (DISCOVER+PLAN+ACQUIRE) against the
    ///   `no_perf_effective_cap` budget, now that the build-time
    ///   `LOCK_SH` fds are still held and peers' holder counts are
    ///   truthful. The fresh plan's fds ride the run; its refreshed
    ///   `cpus` thread through `RunLocks::no_perf_cpus` to every mask
    ///   consumer so mask + lock identity holds by construction.
    /// * `no_perf_mode` + missing plan (bypass / degraded sysfs):
    ///   returns an empty Vec — `build()` already warned, no
    ///   coordination is possible on this host.
    /// * `performance_mode` + `pinning_plan`: reuses the stored
    ///   plan's `llc_indices` to take `LOCK_EX` via
    ///   `acquire_resource_locks_waiting` (fast path, then queue +
    ///   head accumulation under `wait`).
    /// * default else: walks LLC offsets, computing a
    ///   `compute_pinning` candidate per offset and taking `LOCK_SH`
    ///   on the LLC plus `LOCK_EX` on each assigned CPU via
    ///   `acquire_resource_locks` until one succeeds (the LLC `LOCK_SH`
    ///   is compatible with other non-perf `LOCK_SH` holders, blocked
    ///   by perf-mode `LOCK_EX`). If a 1:1 candidate exists but every
    ///   offset is busy (a perf-mode `LOCK_EX` peer on the LLC, or a
    ///   non-perf peer on the per-CPU `LOCK_EX` set) it queues and,
    ///   as head, probes every candidate on each lock-dir wake until
    ///   the authoritative flock release. If NO offset can
    ///   map the topology (the host is too small), or no host topology
    ///   was cached at build time (sysfs unreadable), it OVERCOMMITS
    ///   instead of skipping (via `acquire_default_run_locks` ->
    ///   `build_overcommit_run_locks`). This is the path
    ///   test fixtures take when neither `--perf-mode` nor `--no-perf-mode`
    ///   is in effect.
    fn acquire_run_locks(&self, wait: bool) -> Result<RunLocks> {
        if self.no_perf_mode {
            // Re-PLAN at run time rather than reusing the build-time
            // LLC selection. The old code forwarded `no_perf_plan`'s
            // stored `locked_llcs` straight into `acquire_resource_locks`
            // on the theory that replanning could drift the LLC set out
            // from under masks already computed from the build plan.
            // That reasoning is now inverted by two facts: (1) the
            // build-time `LOCK_SH` fds are HELD through this point (see
            // `KtstrVm::no_perf_plan`), so a full DISCOVER here finally
            // sees peers' reservations — the holder-count feedback the
            // Spread policy needs was dead as long as `build()` stripped
            // those fds before any peer planned; and (2) mask/lock
            // identity is preserved by CONSTRUCTION, not by freezing the
            // plan — the fresh plan's own `cpus` are threaded (via
            // `RunLocks::no_perf_cpus`) to every downstream mask consumer
            // (`run_vm`'s vCPU mask, the BSP mask, the virtio-blk worker
            // placement), so those masks match the fresh plan's locks,
            // not the stale build plan's. The fresh plan's fds are
            // acquired BEFORE the build-time fds release: this arm takes
            // the fresh locks, then `mem::take`s and drops the build-time
            // fds out of `self.no_perf_build_locks` (interior-mutable, so
            // a `&self` borrow can release them), so a peer never observes
            // an empty window on a retained LLC — and, crucially, an LLC
            // the replan ABANDONED stops showing our pid the moment we
            // return, instead of holding a phantom reservation until
            // `KtstrVm` drop. When the plan is `None` (degraded-sysfs /
            // bypass branch in `build()`), no coordination is possible —
            // return empty locks; `build()` already emitted the
            // diagnostic. `acquire_llc_plan` inherits the
            // `KTSTR_CARGO_TEST_MODE` short-circuit (a degenerate
            // lock-free plan naming the full allowed cpuset), so bare
            // `cargo test` needs no special-casing here.
            match (self.no_perf_plan.as_ref(), self.host_topo.as_ref()) {
                (Some(_build_plan), Some(host_topo)) => {
                    // Fresh `TestTopology` for the run-time distance
                    // matrix, derived the same way `build()` derived the
                    // one it fed `acquire_llc_plan` — a distinct probe of
                    // the same static topology, not a stashed handle.
                    let test_topo = crate::topology::TestTopology::from_system()?;
                    // `no_perf_effective_cap` is the exact budget the
                    // build-time plan targeted, so the replan reserves the
                    // same SIZE. Two-stage under `wait`: a non-blocking
                    // fast-path replan first (build-time `LOCK_SH` fds
                    // still held — peers' holder counts stay truthful);
                    // on contention, RELEASE the build-time fds BEFORE
                    // queueing. A queued waiter must hold NO resource
                    // locks: parking on the queue while holding `LOCK_SH`
                    // on an LLC a perf-mode head wants `LOCK_EX` would be
                    // a real deadlock cycle (head waits for our SH; we
                    // wait for the head's queue). The truthfulness loss
                    // is honest — a queued cell owns nothing yet.
                    // Residual `ResourceContention` (and every other
                    // `acquire_llc_plan` error) propagates verbatim for
                    // the classifier's retryable-failure routing.
                    let fresh = match host_topology::acquire_llc_plan(
                        host_topo,
                        &test_topo,
                        self.no_perf_effective_cap,
                        host_topology::PlacementPolicy::spread_for_process(),
                        false,
                    ) {
                        Ok(plan) => plan,
                        Err(e)
                            if wait
                                && e.downcast_ref::<host_topology::ResourceContention>()
                                    .is_some() =>
                        {
                            drop(std::mem::take(
                                &mut *self.no_perf_build_locks.lock().unwrap(),
                            ));
                            host_topology::acquire_llc_plan(
                                host_topo,
                                &test_topo,
                                self.no_perf_effective_cap,
                                host_topology::PlacementPolicy::spread_for_process(),
                                true,
                            )?
                        }
                        Err(e) => return Err(e),
                    };
                    // Move the RAII fds into `RunLocks` so they release at
                    // end-of-run, and carry the refreshed CPUs so the mask
                    // consumers bind to exactly these locked LLCs.
                    let host_topology::LlcPlan { cpus, locks, .. } = fresh;
                    // Fresh locks are now held; release the build-time
                    // `LOCK_SH` fds. Acquire-before-release keeps any LLC
                    // retained across the replan continuously reserved,
                    // while any LLC the replan abandoned stops listing our
                    // pid immediately rather than at `KtstrVm` drop.
                    drop(std::mem::take(
                        &mut *self.no_perf_build_locks.lock().unwrap(),
                    ));
                    Ok(RunLocks {
                        locks,
                        default_cpu_mask: None,
                        pinning_plan: None,
                        no_perf_cpus: Some(cpus),
                    })
                }
                _ => Ok(RunLocks {
                    locks: Vec::new(),
                    default_cpu_mask: None,
                    pinning_plan: None,
                    no_perf_cpus: None,
                }),
            }
        } else if self.performance_mode {
            if let Some(ref plan) = self.pinning_plan {
                // Perf mode reserves a FIXED set. The grain must MATCH the
                // build-time reservation: whole-LLC `LOCK_EX` on validated
                // small-LLC hosts, or per-CPU grain (shared LLC + per-CPU
                // `LOCK_EX` over the plan's exact CPUs) on a host whose LLC
                // dwarfs the cell. `perf_llc_lock_mode` is a pure function
                // of the stored plan + host topology, so it re-derives the
                // same mode `acquire_slot_with_locks` chose at build (the
                // grain plan's per-LLC footprint still clears the ratio +
                // floor). Without a cached host topology (degraded sysfs /
                // cargo-test mode, where the acquire short-circuits anyway)
                // fall back to whole-LLC Exclusive. Under `wait`, queue up
                // and accumulate the set as head rather than skip; `wait ==
                // false` (interactive) keeps the single-shot behaviour.
                let llc_mode = self
                    .host_topo
                    .as_ref()
                    .map(|ht| host_topology::perf_llc_lock_mode(ht, plan))
                    .unwrap_or(host_topology::LlcLockMode::Exclusive);
                match host_topology::acquire_resource_locks_waiting(
                    plan,
                    &plan.llc_indices,
                    llc_mode,
                    wait,
                )? {
                    host_topology::LockOutcome::Acquired { locks, .. } => Ok(RunLocks {
                        locks,
                        default_cpu_mask: None,
                        pinning_plan: None,
                        no_perf_cpus: None,
                    }),
                    host_topology::LockOutcome::Unavailable(reason) => {
                        Err(anyhow::Error::new(host_topology::ResourceContention {
                            reason,
                        }))
                    }
                }
            } else {
                Ok(RunLocks {
                    locks: Vec::new(),
                    default_cpu_mask: None,
                    pinning_plan: None,
                    no_perf_cpus: None,
                })
            }
        } else {
            // Default: try each LLC offset with LOCK_SH until one
            // succeeds. LOCK_SH is compatible with other LOCK_SH
            // holders (multiple non-perf VMs share), but blocked by
            // perf-mode's LOCK_EX. On contention, move to the next
            // offset. If every offset is busy but a 1:1 plan exists,
            // ResourceContention → nextest retries after the holding
            // peer finishes; if no offset maps the topology, overcommit
            // (see the produced_candidate split below).
            Self::acquire_default_run_locks(self.host_topo.as_ref(), &self.topology, wait)
        }
    }

    /// Run-lock acquisition for the interactive / one-shot shell path
    /// ([`Self::run_interactive`]). Passes `wait = false` so
    /// acquisition does NOT queue on a contended reservation — a human on a
    /// busy shared host should boot immediately, not park in the
    /// acquisition queue. On the resulting transient
    /// `ResourceContention` (every candidate LLC slot busy under a concurrent
    /// peer's `LOCK_EX`) it degrades to a lock-free best-effort boot instead
    /// of failing. The test path ([`Self::run`]) instead passes `wait = true`
    /// and queues the holder out (see [`Self::acquire_run_locks`]).
    /// Non-contention errors still propagate.
    fn acquire_interactive_run_locks(&self) -> Result<RunLocks> {
        Self::degrade_contention_to_overcommit(self.acquire_run_locks(false))
    }

    /// Map an [`Self::acquire_run_locks`] result for the one-shot shell path: a
    /// transient `ResourceContention` becomes a lock-free best-effort
    /// `RunLocks`; every other outcome (including non-contention errors) passes
    /// through unchanged. Associated fn over the input `Result` so the degrade
    /// policy is unit-testable — `acquire_resource_locks` short-circuits to
    /// Acquired in cargo-test mode, so the contention branch is otherwise
    /// unreachable from a test.
    fn degrade_contention_to_overcommit(acquired: Result<RunLocks>) -> Result<RunLocks> {
        match acquired {
            Err(e)
                if e.downcast_ref::<host_topology::ResourceContention>()
                    .is_some() =>
            {
                eprintln!(
                    "ktstr: host CPUs busy ({e}); booting the shell without an \
                     exclusive CPU reservation"
                );
                Ok(RunLocks {
                    locks: Vec::new(),
                    default_cpu_mask: None,
                    pinning_plan: None,
                    no_perf_cpus: None,
                })
            }
            other => other,
        }
    }

    /// Default (non-perf, non-no-perf) run-lock acquisition: try each LLC offset
    /// for a 1:1 LOCK_SH pin. The offset scan is the acquisition
    /// protocol's non-blocking FAST PATH — all-or-nothing per
    /// candidate, claim-subtracted (candidates targeting a live queue
    /// head's claimed LLCs/CPUs are skipped), one bounce per offset,
    /// then the verdict:
    ///
    /// * A candidate acquired → 1:1 pinned run.
    /// * NO offset maps the topology, or there is no cached host
    ///   topology → the host is too small for a 1:1 pin → OVERCOMMIT
    ///   (the host-gate policy: make it work; NOT contention).
    /// * Candidates map but every offset bounced → genuine
    ///   contention. `wait == false` (interactive) surfaces
    ///   `ResourceContention` immediately for the overcommit degrade.
    ///   `wait == true` (the TEST run path) joins the
    ///   cross-invocation queue and, as head, RE-PLANS ON EVERY WAKE:
    ///   each lock-dir event re-probes EVERY candidate all-or-nothing
    ///   (so a fully-freed alternative is taken immediately — never
    ///   kept waiting on the original choice), and while none
    ///   completes it accumulates the primary candidate (maximum
    ///   overlap with already-held locks, ties to the pid-staggered
    ///   scan order) under the head license, publishing it as the
    ///   claim. It waits for the authoritative flock release; the
    ///   holder's VM watchdog and nextest process rail own lifecycle
    ///   bounds, so host preemption cannot be misclassified as
    ///   contention.
    ///
    /// Associated fn (no `&self`) over host_topo/topology so
    /// the overcommit-vs-contention decision is unit-testable with a
    /// synthetic HostTopology (acquire_resource_locks short-circuits
    /// to Acquired in cargo-test mode, so a fitting host yields a pin
    /// and a too-small host yields overcommit).
    fn acquire_default_run_locks(
        host_topo: Option<&host_topology::HostTopology>,
        topology: &Topology,
        wait: bool,
    ) -> Result<RunLocks> {
        let Some(host_topo) = host_topo else {
            // No cached host topology (sysfs unreadable at build time): no
            // topology to plan a 1:1 pin against -> overcommit directly.
            return Ok(Self::build_overcommit_run_locks(
                host_topology::host_allowed_cpus(),
                topology.total_cpus() as usize,
            ));
        };
        let num_llcs = host_topo.llc_groups.len();
        let llcs_needed = (topology.llcs as usize).max(1);
        let max_slots = num_llcs.checked_div(llcs_needed).unwrap_or(1).max(1);
        // Candidate list, computed once — candidates are a pure
        // function of the static host topology. TWO dimensions:
        // the LLC offset (which LLC group set) × the INTRA slot
        // (which disjoint CPU window inside those LLCs — see
        // `compute_pinning_at`). Without the intra dimension every
        // 1-vCPU cell contends for one CPU prefix per LLC and a
        // 64-CPU host runs default cells only `num_llcs` wide; with
        // it, effective parallelism scales with CPUs. Enumeration is
        // intra-major within each LLC offset, and the whole flat list
        // is pid-rotated so concurrent processes spread across
        // candidates instead of scanning in the same order.
        let mut flat: Vec<host_topology::PinningPlan> = Vec::new();
        for slot in 0..max_slots {
            let offset = slot * llcs_needed;
            for intra in 0.. {
                match host_topo.compute_pinning_at(topology, false, offset, intra) {
                    Ok(c) => flat.push(c),
                    Err(_) => break, // window no longer fits — slots exhausted
                }
            }
        }
        let start = host_topology::pid_window_offset(std::process::id(), flat.len().max(1));
        let candidates: Vec<host_topology::PinningPlan> = (0..flat.len())
            .map(|i| flat[(start + i) % flat.len()].clone_unlocked())
            .collect();
        drop(flat);
        if candidates.is_empty() {
            // No offset could map the topology: host too small for a 1:1
            // placement. Overcommit instead of skipping.
            return Ok(Self::build_overcommit_run_locks(
                host_topology::host_allowed_cpus(),
                topology.total_cpus() as usize,
            ));
        }
        // FAST PATH: one non-blocking, all-or-nothing bounce per
        // candidate (claim subtraction happens inside).
        for candidate in &candidates {
            match host_topology::acquire_resource_locks(
                candidate,
                &candidate.llc_indices,
                host_topology::LlcLockMode::Shared,
            )? {
                host_topology::LockOutcome::Acquired { locks, .. } => {
                    return Ok(RunLocks {
                        locks,
                        default_cpu_mask: None,
                        pinning_plan: Some(candidate.clone_unlocked()),
                        no_perf_cpus: None,
                    });
                }
                host_topology::LockOutcome::Unavailable(_) => continue,
            }
        }
        if !wait {
            return Err(anyhow::Error::new(host_topology::ResourceContention {
                reason: format!(
                    "all {max_slots} LLC slots busy (LOCK_SH)\n  \
                     hint: a performance_mode test may hold LOCK_EX"
                ),
            }));
        }
        // Contended: queue up (holding NOTHING — every bounce above
        // released everything), then acquire as head.
        Self::acquire_default_as_head(&candidates)
    }

    /// Queue-and-head phase of [`Self::acquire_default_run_locks`]:
    /// see that method's doc for the probe-all / accumulate-primary
    /// re-plan model. Split out so the wait machinery reads as one
    /// unit.
    fn acquire_default_as_head(candidates: &[host_topology::PinningPlan]) -> Result<RunLocks> {
        use host_topology::protocol;
        let _queue = protocol::wait_for_queue_turn()?;
        // Pre-compute each candidate's canonical lock order and claim.
        let targets: Vec<Vec<(String, crate::flock::FlockMode)>> = candidates
            .iter()
            .map(|c| {
                let mut cpus: Vec<usize> = c.assignments.iter().map(|&(_, cpu)| cpu).collect();
                cpus.extend(c.service_cpu);
                protocol::canonical_lock_order(
                    &c.llc_indices,
                    crate::flock::FlockMode::Shared,
                    &cpus,
                )
            })
            .collect();
        let outcome = protocol::acquire_as_head(|held| {
            // RE-PLAN on every wake: probe EVERY candidate
            // all-or-nothing (reusing held locks) so a fully-freed
            // alternative is taken the moment it appears — the head
            // never keeps waiting on its original choice while a
            // different sufficient set sits free.
            for (i, target) in targets.iter().enumerate() {
                if let Some(locks) = held.probe_complete(target)? {
                    return Ok(protocol::HeadStep::Complete((i, locks)));
                }
            }
            // None complete: accumulate toward the PRIMARY candidate —
            // maximum overlap with what we already hold (dropping
            // partials the fresh choice still needs would be wasted
            // makespan), ties to pid-staggered scan order.
            let held_paths = held.held_paths();
            let primary = (0..targets.len())
                .max_by_key(|&i| {
                    let overlap = targets[i]
                        .iter()
                        .filter(|(p, _)| held_paths.contains(p))
                        .count();
                    // Stable tie-break toward scan order.
                    (overlap, std::cmp::Reverse(i))
                })
                .expect("candidates is non-empty — checked by caller");
            let target = &targets[primary];
            held.retain_paths(&target.iter().map(|(p, _)| p.clone()).collect());
            held.sweep(target)?;
            if held.covers(target) {
                // A holder released between the probe pass and this
                // sweep and the sweep finished the set — complete now
                // rather than sleeping on a satisfied target.
                let locks = held.take(target);
                return Ok(protocol::HeadStep::Complete((primary, locks)));
            }
            let candidate = &candidates[primary];
            let mut claim_cpus: std::collections::BTreeSet<usize> =
                candidate.assignments.iter().map(|&(_, cpu)| cpu).collect();
            claim_cpus.extend(candidate.service_cpu);
            Ok(protocol::HeadStep::Waiting {
                claim: protocol::ClaimSet::new(
                    candidate.llc_indices.iter().copied(),
                    claim_cpus,
                    crate::flock::FlockMode::Shared,
                ),
            })
        })?;
        match outcome {
            protocol::HeadOutcome::Acquired((i, locks)) => Ok(RunLocks {
                locks,
                default_cpu_mask: None,
                pinning_plan: Some(candidates[i].clone_unlocked()),
                no_perf_cpus: None,
            }),
            protocol::HeadOutcome::Aborted { reason } => {
                Err(anyhow::Error::new(host_topology::ResourceContention {
                    reason,
                }))
            }
        }
    }

    /// Build the overcommit `RunLocks` from a resolved allowed-CPU set: no
    /// resource locks, the allowed cpuset as the default mask, and an
    /// `overcommit_warning` when the mask is narrower than the vCPU count.
    ///
    /// This is the default (non-perf, non-no-perf) path's fallback when no 1:1
    /// pinning plan can map the topology onto the host (the host is too small,
    /// or no host topology was cached at build time). Rather than skipping, the
    /// VM runs OVERCOMMITTED — every vCPU thread masked to the host's allowed
    /// CPUs (oversubscribed + time-sliced). The mask is the run-time budget
    /// source: the orchestrator (`run`) reads the returned `default_cpu_mask`
    /// and overrides `VmResult.cpu_budget` with the masked host-CPU count so the
    /// sidecar records the overcommit (not the build-time vCPU count). This is
    /// the only place the default path fires the overcommit warning — `build()`
    /// computes the `no_perf_plan` warning only under `no_perf_mode`. When the
    /// allowed set is empty (a host that cannot enumerate its own affinity),
    /// `set_thread_cpumask` no-ops and the warning is the sole signal.
    ///
    /// Pure (the allowed set is passed in) so the mask + warning policy is
    /// unit-testable without reading the host cpuset.
    fn build_overcommit_run_locks(allowed: Vec<usize>, vcpus: usize) -> RunLocks {
        if let Some(w) = host_topology::overcommit_warning(allowed.len(), vcpus, false) {
            eprintln!("{w}");
        }
        RunLocks {
            locks: Vec::new(),
            default_cpu_mask: Some(allowed),
            pinning_plan: None,
            no_perf_cpus: None,
        }
    }

    /// Boot the VM with bidirectional stdin/stdout forwarding via virtio-console.
    ///
    /// Sets the host terminal to raw mode, spawns threads for stdin->hvc0
    /// and hvc0->stdout forwarding, and runs until the guest shuts down.
    /// Terminal state is restored on all exit paths including panic and
    /// process-killing signals (SIGINT, SIGTERM, SIGQUIT, SIGABRT,
    /// SIGFPE).
    ///
    /// Builder settings ignored in interactive mode: `monitor_thresholds`,
    /// `watchdog_timeout`, `bpf_map_write`, `performance_mode` pinning,
    /// and KVM stats collection. These are test-specific features that
    /// do not apply to interactive shell sessions.
    pub fn run_interactive(&self) -> Result<Option<i32>> {
        let start = Instant::now();

        let initramfs_handle = self.spawn_initramfs_resolve()?;
        let (mut vm, kernel_result) = self.create_vm_and_load_kernel()?;

        #[cfg(target_arch = "x86_64")]
        {
            let kr = self.setup_memory(&mut vm, kernel_result, initramfs_handle)?;
            self.setup_vcpus(&vm, kr.entry)?;
        }
        #[cfg(target_arch = "aarch64")]
        {
            let kr = self.setup_memory_aarch64(&mut vm, kernel_result, initramfs_handle)?;
            self.setup_vcpus_aarch64(&vm, kr.entry)?;
        }

        let com1 = Arc::new(PiMutex::new(console::Serial::new(console::COM1_BASE)));
        let com2 = Arc::new(PiMutex::new(console::Serial::new(console::COM2_BASE)));

        // Userspace IOAPIC handle for the split-irqchip path (>254 max APIC ID),
        // mirroring run_vm: the device + the raw VM fd, threaded into
        // spawn_ap_threads + run_bsp_loop so the interactive shell's serial /
        // virtio-console IRQs route via the userspace IOAPIC. `None` for
        // <=254 max APIC ID (in-kernel IOAPIC).
        // x86-only (mirrors run_vm): aarch64 has no userspace IOAPIC — the
        // GIC routes device IRQs and `IoapicHandle` is the uninhabited
        // placeholder — so the handle is always `None` there.
        #[cfg(target_arch = "x86_64")]
        let ioapic_handle: Option<Arc<crate::vmm::IoapicHandle>> = vm.ioapic.as_ref().map(|io| {
            Arc::new(crate::vmm::IoapicHandle::new(
                io.clone(),
                std::os::unix::io::AsRawFd::as_raw_fd(&*vm.vm_fd),
            ))
        });
        #[cfg(not(target_arch = "x86_64"))]
        let ioapic_handle: Option<Arc<crate::vmm::IoapicHandle>> = None;

        // Full-irqchip shared GSI route owner (inverse of `ioapic_handle`):
        // `Some` only on the full in-kernel irqchip path with PCI enabled, where
        // it owns the device MSI-X routes. `install_defaults` installs an explicit
        // default routing table route-identical to KVM's implicit one, so the
        // legacy INTx routes survive the first MSI-X `KVM_SET_GSI_ROUTING`
        // (whole-table replace). `None` on split-irqchip / non-PCI shells.
        #[cfg(target_arch = "x86_64")]
        let full_route_owner: Option<Arc<crate::vmm::FullIrqchipRouteOwner>> =
            (!vm.split_irqchip && vm.pci_enabled).then(|| {
                Arc::new(crate::vmm::FullIrqchipRouteOwner::new(
                    std::os::unix::io::AsRawFd::as_raw_fd(&*vm.vm_fd),
                ))
            });
        #[cfg(target_arch = "x86_64")]
        if let Some(owner) = full_route_owner.as_ref() {
            owner
                .install_defaults()
                .context("install full-irqchip default GSI routing")?;
        }

        // PCI host bridge handle (virtio-PCI transport), constructed only when
        // this VM enables PCI; `None` keeps non-PCI shells byte-identical.
        // Mirrors the run_vm construction. x86-only for now.
        #[cfg(target_arch = "x86_64")]
        let pci_bus_handle: Option<Arc<PiMutex<pci::PciBus>>> = if vm.pci_enabled {
            Some(Arc::new(PiMutex::new(pci::PciBus::new(
                kvm::PCI_ECAM_BASE,
                kvm::PCI_ECAM_SIZE,
            ))))
        } else {
            None
        };
        #[cfg(not(target_arch = "x86_64"))]
        let pci_bus_handle: Option<Arc<PiMutex<pci::PciBus>>> = None;

        // Virtio-console for shell I/O via /dev/hvc0.
        let mut vc = virtio_console::VirtioConsole::new();
        vc.set_mem((*vm.guest_mem).clone());
        let virtio_con = Arc::new(PiMutex::new(vc));

        // Register serial + virtio-console irqfds. On x86 split-irqchip
        // (>254 APIC IDs) the routes are installed by the userspace IOAPIC
        // when the guest programs its RTEs (ioapic_handle above is threaded
        // into the run loops); on the in-kernel-irqchip (x86 <=254) and
        // aarch64 (GIC) paths the kernel routes the GSIs directly.
        #[cfg(target_arch = "x86_64")]
        {
            vm.vm_fd
                .register_irqfd(com1.lock().irq_evt(), console::COM1_IRQ)
                .context("register COM1 irqfd")?;
            vm.vm_fd
                .register_irqfd(com2.lock().irq_evt(), console::COM2_IRQ)
                .context("register COM2 irqfd")?;
            vm.vm_fd
                .register_irqfd(virtio_con.lock().irq_evt(), kvm::VIRTIO_CONSOLE_IRQ)
                .context("register virtio-console irqfd")?;
        }
        #[cfg(target_arch = "aarch64")]
        {
            vm.vm_fd
                .register_irqfd(com1.lock().irq_evt(), kvm::SERIAL_IRQ)
                .context("register serial irqfd")?;
            vm.vm_fd
                .register_irqfd(com2.lock().irq_evt(), kvm::SERIAL2_IRQ)
                .context("register serial2 irqfd")?;
            vm.vm_fd
                .register_irqfd(virtio_con.lock().irq_evt(), kvm::VIRTIO_CONSOLE_IRQ)
                .context("register virtio-console irqfd")?;
        }

        // Optional virtio-blk for shell mode. Transport is arch-split
        // (mirrors virtio-net): x86_64 installs a virtio-pci function on
        // `pci_bus_handle` (`virtio_blk_pci_slot()`, below); the MMIO handle is
        // `None`, so the dispatch loops' MMIO blk arm goes inert and guest BAR
        // accesses route through the PCI bus. The installed PciBus function owns
        // the device Arc — shell mode needs neither counters (no `VmResult`) nor
        // worker pause (no freeze coordinator); the INTx resample eventfd is kept
        // alive below. aarch64 keeps the MMIO handle that drives the dispatch
        // loops (aarch64 PCI is a later increment). `None` when no disk is
        // attached.
        #[cfg(target_arch = "x86_64")]
        let virtio_blk: Option<Arc<PiMutex<virtio_blk::VirtioBlk>>> = None;
        // Interactive path: no run-time replan runs (see
        // `acquire_interactive_run_locks`), so the worker placement uses
        // the build-time `no_perf_plan` fallback — pass `None`.
        #[cfg(not(target_arch = "x86_64"))]
        let virtio_blk = self.init_virtio_blk(&vm, None)?;

        // Optional virtio-net for shell mode. Transport is arch-split
        // (mirrors `run_vm`): x86_64 installs a virtio-pci function on
        // `pci_bus_handle` (slot 1) and keeps the INTx resample eventfd
        // alive for the run (KVM holds its raw fd); the MMIO handle is
        // `None`, so the dispatch loops' MMIO net arm goes inert and
        // guest BAR accesses route through the PCI bus. aarch64 keeps the
        // MMIO handle that drives the dispatch loops (aarch64 PCI is a
        // later increment). Shell mode discards the counters Arc (no
        // `VmResult`). `None` when the builder has no `NetConfig`.
        #[cfg(target_arch = "x86_64")]
        let virtio_net: Option<Arc<PiMutex<virtio_net::VirtioNet>>> = None;
        // The active GSI-route owner as the MSI-X route sink: `IoapicHandle` on
        // split-irqchip, `FullIrqchipRouteOwner` on full (both impl
        // `MsixRouteSink`). Threaded into `init_virtio_net_pci` so MSI-X is
        // offered on both irqchip paths; cloned (Arc) so each owner stays
        // available for the run loops + teardown routing diagnostics.
        #[cfg(target_arch = "x86_64")]
        let msix_sink: Option<Arc<dyn virtio_msix::MsixRouteSink>> = if vm.split_irqchip {
            ioapic_handle
                .clone()
                .map(|h| h as Arc<dyn virtio_msix::MsixRouteSink>)
        } else {
            full_route_owner
                .clone()
                .map(|o| o as Arc<dyn virtio_msix::MsixRouteSink>)
        };
        #[cfg(target_arch = "x86_64")]
        let _net_resample_evts: Vec<_> = match pci_bus_handle.as_ref() {
            // One resample eventfd per NIC (Some only on the full-irqchip path);
            // all are held alive for the shell run so KVM can de-assert each
            // level GSI on guest EOI. Shell mode discards the counters.
            Some(bus) => self
                .init_virtio_net_pci(&vm, bus, msix_sink.clone())?
                .into_iter()
                .map(|h| h.resample_evt)
                .collect(),
            // No PCI bus => no NIC. The builder forces pci_enabled=true whenever
            // a NetConfig is attached on x86_64, so this arm is only reached with
            // no networks; assert that loudly so a future refactor decoupling
            // networks from pci_enabled cannot silently drop NICs (no eth0).
            None => {
                debug_assert!(
                    self.networks.is_empty(),
                    "x86_64 NetConfig(s) attached but no PCI bus — NICs would be \
                     silently dropped; pci_enabled must follow network()"
                );
                Vec::new()
            }
        };
        // virtio-blk PCI install (x86_64). The device Arc is owned by the
        // installed PciBus function; only the INTx resample eventfd is retained
        // (held alive so KVM can de-assert the level GSI on guest EOI). `None`
        // on split-irqchip or when no disk is attached.
        // Interactive path: `None` no-perf override (build-time fallback);
        // no run-time replan runs here.
        #[cfg(target_arch = "x86_64")]
        let _blk_resample_evt = match pci_bus_handle.as_ref() {
            Some(bus) => self
                .init_virtio_blk_pci(&vm, bus, msix_sink, None)?
                .and_then(|h| h.resample_evt),
            None => None,
        };
        #[cfg(not(target_arch = "x86_64"))]
        let virtio_net = self.init_virtio_net(&vm)?;

        // Non-interactive exec mode (--exec) does not need a TTY.
        let exec_mode = self.exec_cmd.is_some();

        // Pre-flight: verify stdin is a tty, enter raw mode, and create
        // the wakeup pipe before spawning threads. Failing after thread
        // spawn would abandon AP threads.
        if !exec_mode {
            use std::os::unix::io::AsRawFd;
            let stdin_fd = std::io::stdin().as_raw_fd();
            let borrowed = unsafe { std::os::unix::io::BorrowedFd::borrow_raw(stdin_fd) };
            anyhow::ensure!(
                nix::unistd::isatty(borrowed).unwrap_or(false),
                "stdin must be a terminal for interactive shell mode",
            );
        }

        // Set host terminal to raw mode. TerminalRawGuard restores on drop
        // and installs signal handlers for SIGINT, SIGTERM, SIGQUIT,
        // SIGABRT, and SIGFPE so every terminating signal routes through
        // the terminal-restore path before the process exits (see
        // `src/vmm/terminal.rs`). Skip for exec mode — no interactive
        // terminal needed.
        let _raw_guard = if exec_mode {
            None
        } else {
            Some(TerminalRawGuard::enter().context("failed to set terminal to raw mode")?)
        };

        // Wakeup pipe: write end signals the stdin reader to exit when
        // the kill flag is set, avoiding a blocking read that prevents join.
        let (wakeup_r, wakeup_w) = nix::unistd::pipe().context("create stdin wakeup pipe")?;

        let kill = Arc::new(AtomicBool::new(false));
        // Companion eventfd for `kill`. The interactive shell has no
        // epoll consumer for it (kicks land via `pthread_kill` +
        // `immediate_exit`), but spawn_ap_threads requires a non-None
        // eventfd in its signature; allocate a sentinel and let it
        // drop with the function frame.
        let kill_evt = Arc::new(
            vmm_sys_util::eventfd::EventFd::new(libc::EFD_NONBLOCK)
                .context("create shell kill eventfd")?,
        );
        // Interactive shell does not arm the failure-dump freeze
        // pipeline (no monitor thread requesting freezes). Construct
        // sentinel flags that stay false for the lifetime of the
        // session so vcpu_run_loop_unified / run_bsp_loop see a stable
        // freeze=false on every iteration and never enter the park
        // path.
        let freeze = Arc::new(AtomicBool::new(false));
        // Interactive shell never runs the freeze coordinator, so
        // `request_kva` stays 0 and `self_arm_watchpoint` is a no-op
        // on every iteration. Allocated only to satisfy the
        // spawn_ap_threads / run_bsp_loop signatures shared with the
        // failure-dump path.
        let watchpoint =
            Arc::new(vcpu::WatchpointArm::new().context("create WatchpointArm.hit_evt EventFd")?);
        let bsp_parked = Arc::new(AtomicBool::new(false));
        let bsp_regs: Arc<std::sync::Mutex<Option<exit_dispatch::VcpuRegSnapshot>>> =
            Arc::new(std::sync::Mutex::new(None));
        let has_immediate_exit = vm.has_immediate_exit;
        let mut vcpus = std::mem::take(&mut vm.vcpus);
        let mut bsp = vcpus.remove(0);

        let ap_pins = vec![None; vcpus.len()];
        // Acquire run-scoped flock fds via the same path
        // [`Self::run`] uses. `build()` strips the locks from the
        // cached plan; this re-take covers exactly the vCPU
        // thread lifetime, releasing every fd when the function
        // returns. Held via a binding so RAII drop fires on
        // every exit path (including the `?` early-returns
        // below).
        let _run_locks = self.acquire_interactive_run_locks()?;
        // Shell/interactive path mirrors run_vm: no-perf + --cpu-cap
        // applies the LlcPlan's CPU list as a sched_setaffinity mask
        // on every vCPU thread. Perf-mode's pin_targets doesn't
        // apply here — interactive shell runs under no-perf by
        // convention, and `pin_targets` is empty in this branch.
        let no_perf_mask: Option<&[usize]> = self.no_perf_plan.as_ref().map(|p| p.cpus.as_slice());
        // Interactive shell does not run a freeze coordinator, so
        // discard the freeze-handle Vecs. Interactive mode also skips
        // the perf-counter capture path; allocate empty TID slots so
        // the spawn signature is honored without producing values
        // anything reads.
        let n_aps = vcpus.len();
        let ap_tid_slots: Vec<(Arc<AtomicI32>, Arc<crate::sync::Latch>)> = (0..n_aps)
            .map(|_| {
                (
                    Arc::new(AtomicI32::new(0)),
                    Arc::new(crate::sync::Latch::new()),
                )
            })
            .collect();
        // The shared AP spawner gates each AP progressively before creating
        // the next one, including on the interactive path. This keeps a large
        // shell topology from recreating the same host-thread thundering herd
        // as a test VM. There is no separate outer all-AP gate here: every
        // latch has already fired when `spawn_ap_threads` returns.
        let ap_boot_latches: Vec<Arc<crate::sync::Latch>> = (0..n_aps)
            .map(|_| Arc::new(crate::sync::Latch::new()))
            .collect();
        let (ap_threads, _ap_freeze) = self.spawn_ap_threads(
            vcpus,
            has_immediate_exit,
            &com1,
            &com2,
            Some(&virtio_con),
            virtio_blk.as_ref(),
            virtio_net.as_ref(),
            ioapic_handle.as_ref(),
            pci_bus_handle.as_ref(),
            &kill,
            &kill_evt,
            &freeze,
            &watchpoint,
            &ap_pins,
            no_perf_mask,
            &ap_tid_slots,
            &ap_boot_latches,
            // Interactive shell does not run a freeze coordinator,
            // so no parked_evt / thaw_evt to plumb. The
            // `vcpu_run_loop_unified` honours `freeze` only when it
            // flips, which never happens in this path; the
            // eventfds remain unused.
            None,
            None,
        )?;

        // BSP kick handles for the stdin escape sequence. The stdin thread
        // needs to force the BSP out of KVM_RUN when Ctrl+A X is pressed.
        let bsp_ie_for_stdin = if has_immediate_exit {
            Some(ImmediateExitHandle::from_vcpu(&mut bsp))
        } else {
            None
        };
        let bsp_tid = unsafe { libc::pthread_self() };

        // Bound a `--exec` payload's wall-clock. A panic-less guest
        // hang leaves the guest halted, so the BSP `bsp.run()` blocks in
        // KVM_RUN indefinitely — flipping `kill` alone never unblocks it
        // (the loop only re-checks kill after a vCPU exit). The watchdog
        // mirrors the stdin Ctrl+A X kick (kill + immediate_exit +
        // SIGRTMIN) on a deadline. Gated on exec_mode — interactive
        // sessions have no deadline (the human drives Ctrl+A X). Joined in
        // the teardown below BEFORE `bsp` drops, so its `ImmediateExitHandle`
        // (a Copy of `bsp_ie_for_stdin`, pointing into `bsp`'s kvm_run
        // mmap) never writes through a freed mapping.
        let timed_out = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let exec_watchdog = if exec_mode {
            let bsp_ie_for_wd = bsp_ie_for_stdin;
            let kill_for_wd = kill.clone();
            let timed_out_for_wd = timed_out.clone();
            let deadline = self.exec_timeout;
            Some(
                std::thread::Builder::new()
                    .name("ktstr-exec-wdog".into())
                    .spawn(move || {
                        let start = std::time::Instant::now();
                        loop {
                            // Normal completion / Ctrl+A X flips `kill`.
                            if kill_for_wd.load(Ordering::Acquire) {
                                return;
                            }
                            if start.elapsed() >= deadline {
                                // Re-check: the payload may have completed
                                // in the poll gap; only kick if still live.
                                if kill_for_wd.load(Ordering::Acquire) {
                                    return;
                                }
                                timed_out_for_wd.store(true, Ordering::Release);
                                kill_for_wd.store(true, Ordering::Release);
                                if let Some(ref ie) = bsp_ie_for_wd {
                                    ie.set(1);
                                    std::sync::atomic::fence(Ordering::Release);
                                }
                                // SAFETY: bsp_tid is the BSP thread (this
                                // function's own thread); the watchdog is
                                // joined before the function returns, so the
                                // tid is live for the kick.
                                unsafe {
                                    libc::pthread_kill(bsp_tid, vcpu_signal());
                                }
                                return;
                            }
                            std::thread::sleep(std::time::Duration::from_millis(100));
                        }
                    })
                    .context("spawn ktstr-exec-wdog (interactive exec watchdog) thread")?,
            )
        } else {
            None
        };

        // UAF safety: the watchdog AND the stdin thread (spawned
        // below) each hold a Copy of `bsp_ie_for_stdin` — a raw pointer
        // into `bsp`'s kvm_run mmap (vcpu::ImmediateExitHandle) — plus
        // `bsp_tid`, and each kicks the BSP (ie.set(1) + pthread_kill) on
        // its trigger (watchdog deadline / Ctrl+A X). Both MUST be joined
        // before `bsp` drops, on EVERY exit path: the normal teardown, the
        // `?` early-returns below (stdin/stdout/dmesg spawn + eventfds),
        // and a panic-unwind (the test profile unwinds). A bare teardown
        // join covers only the normal path, so wrap both handles in an RAII
        // guard whose Drop sets `kill` (each thread re-checks it within its
        // <=100ms poll/sleep and returns WITHOUT kicking) then joins.
        // Declared after `bsp` (above) so it drops — and joins — BEFORE
        // bsp's kvm_run unmaps, on the normal path AND every early-return /
        // unwind.
        struct CrossThreadKickGuard {
            watchdog: Option<std::thread::JoinHandle<()>>,
            stdin: Option<std::thread::JoinHandle<()>>,
            kill: std::sync::Arc<std::sync::atomic::AtomicBool>,
        }
        impl Drop for CrossThreadKickGuard {
            fn drop(&mut self) {
                self.kill.store(true, std::sync::atomic::Ordering::Release);
                if let Some(h) = self.watchdog.take() {
                    let _ = h.join();
                }
                if let Some(h) = self.stdin.take() {
                    let _ = h.join();
                }
            }
        }
        let mut kick_guard = CrossThreadKickGuard {
            watchdog: exec_watchdog,
            stdin: None,
            kill: kill.clone(),
        };

        // Stdin reader thread: host stdin -> virtio-console RX queue.
        // The guest reads stdin from /dev/hvc0 (virtio-console), never
        // from COM2. pending_rx buffers input until the guest activates
        // the RX queue. Uses poll() on both stdin and the wakeup pipe
        // so the thread can be cleanly joined on shutdown.
        //
        // Escape sequence: Ctrl+A X (0x01 followed by 'x' or 'X') triggers
        // host-side VM teardown without guest cooperation.
        let vc_for_stdin = virtio_con.clone();
        let kill_for_stdin = kill.clone();
        let stdin_thread = std::thread::Builder::new()
            .name("interactive-stdin".into())
            .spawn(move || {
                use std::io::Read;
                use std::os::unix::io::{AsFd, AsRawFd};

                // wakeup_r is an OwnedFd moved into this closure; closed on exit.
                let wakeup_fd = wakeup_r;
                let stdin_fd = std::io::stdin().as_raw_fd();
                let mut buf = [0u8; 4096];
                let mut saw_ctrl_a = false;

                loop {
                    if kill_for_stdin.load(Ordering::Acquire) {
                        break;
                    }

                    // Poll stdin and wakeup fd with 100ms timeout.
                    let stdin_borrowed =
                        unsafe { std::os::unix::io::BorrowedFd::borrow_raw(stdin_fd) };
                    let wakeup_borrowed = wakeup_fd.as_fd();
                    let mut fds = [
                        nix::poll::PollFd::new(stdin_borrowed, nix::poll::PollFlags::POLLIN),
                        nix::poll::PollFd::new(wakeup_borrowed, nix::poll::PollFlags::POLLIN),
                    ];
                    match nix::poll::poll(&mut fds, 100u16) {
                        Ok(0) => continue, // timeout
                        Err(nix::errno::Errno::EINTR) => continue,
                        Err(_) => break,
                        Ok(_) => {}
                    }

                    // Wakeup fd readable means shutdown requested.
                    if fds[1]
                        .revents()
                        .is_some_and(|r| r.intersects(nix::poll::PollFlags::POLLIN))
                    {
                        break;
                    }

                    // Stdin readable.
                    if fds[0]
                        .revents()
                        .is_some_and(|r| r.intersects(nix::poll::PollFlags::POLLIN))
                    {
                        let mut stdin = std::io::stdin().lock();
                        match stdin.read(&mut buf) {
                            Ok(0) => break,
                            Ok(n) => {
                                // Scan for Ctrl+A X escape sequence. Filter
                                // escape bytes from the forwarded input so
                                // neither the 0x01 nor 'x'/'X' reaches the
                                // guest.
                                let mut forward_start = 0usize;
                                for i in 0..n {
                                    if saw_ctrl_a {
                                        saw_ctrl_a = false;
                                        if buf[i] == b'x' || buf[i] == b'X' {
                                            // Trigger host-side teardown. Bytes
                                            // before the Ctrl+A were already
                                            // flushed when saw_ctrl_a was set.
                                            eprintln!("\r\nTerminated.");
                                            kill_for_stdin.store(true, Ordering::Release);
                                            if let Some(ref ie) = bsp_ie_for_stdin {
                                                ie.set(1);
                                                std::sync::atomic::fence(Ordering::Release);
                                            }
                                            unsafe {
                                                libc::pthread_kill(bsp_tid, vcpu_signal());
                                            }
                                            return;
                                        }
                                        // Not 'x'/'X' after Ctrl+A: the 0x01
                                        // was a real keystroke. Flush any
                                        // unflushed bytes preceding this point
                                        // first so the deferred 0x01 lands in
                                        // chronological order, then queue the
                                        // 0x01, then continue processing from
                                        // `i` onward (current byte may itself
                                        // be 0x01).
                                        if forward_start < i {
                                            vc_for_stdin.lock().queue_input(&buf[forward_start..i]);
                                            forward_start = i;
                                        }
                                        vc_for_stdin.lock().queue_input(&[0x01]);
                                    }
                                    if buf[i] == 0x01 {
                                        // Flush bytes before the Ctrl+A.
                                        if forward_start < i {
                                            vc_for_stdin.lock().queue_input(&buf[forward_start..i]);
                                        }
                                        saw_ctrl_a = true;
                                        forward_start = i + 1;
                                        continue;
                                    }
                                }
                                // Forward remaining bytes.
                                if forward_start < n {
                                    vc_for_stdin.lock().queue_input(&buf[forward_start..n]);
                                }
                            }
                            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                            Err(_) => break,
                        }
                    }
                }
            })
            .context("spawn stdin reader thread")?;
        // Hand the stdin handle to the kick guard so it is joined before
        // `bsp` drops on every path (see CrossThreadKickGuard above). Any
        // `?` after this point (stdout/dmesg spawn + eventfds) now joins
        // both cross-thread holders via the guard's Drop.
        kick_guard.stdin = Some(stdin_thread);

        // Stdout writer thread: virtio-console TX -> host stdout.
        // Polls tx_evt for zero-latency wakeup when guest writes data.
        // On write errors (including BrokenPipe), sets kill flag and exits
        // to stop the VM rather than polling a dead pipe until timeout.
        let vc_for_stdout = virtio_con.clone();
        let kill_for_stdout = kill.clone();
        let stdout_thread: JoinHandle<bool> = std::thread::Builder::new()
            .name("interactive-stdout".into())
            .spawn(move || {
                use std::io::Write;

                let mut wrote_any = false;

                // Cache the raw fd for poll. The eventfd lives as long as
                // VirtioConsole which is behind Arc<PiMutex> — valid for
                // the thread's lifetime.
                let tx_evt_raw_fd = {
                    let guard = vc_for_stdout.lock();
                    std::os::unix::io::AsRawFd::as_raw_fd(guard.tx_evt())
                };
                let mut stdout = std::io::stdout().lock();
                loop {
                    if kill_for_stdout.load(Ordering::Acquire) {
                        break;
                    }
                    let borrowed =
                        unsafe { std::os::unix::io::BorrowedFd::borrow_raw(tx_evt_raw_fd) };
                    let mut fds = [nix::poll::PollFd::new(
                        borrowed,
                        nix::poll::PollFlags::POLLIN,
                    )];
                    match nix::poll::poll(&mut fds, 50u16) {
                        Ok(0) => continue,
                        Err(nix::errno::Errno::EINTR) => continue,
                        Err(_) => break,
                        Ok(_) => {
                            // Consume eventfd counter.
                            let _ = vc_for_stdout.lock().tx_evt().read();
                        }
                    }
                    // Re-check kill after poll. During shutdown the
                    // dying guest may enqueue a stray byte into the
                    // virtio TX queue (from kernel hvc_close flushing
                    // n_outbuf via tty_wait_until_sent → hvc_push →
                    // put_chars). That byte passes from_utf8 (valid
                    // single-byte UTF-8) but is unprintable, producing
                    // a garbled character on the terminal.
                    if kill_for_stdout.load(Ordering::Acquire) {
                        break;
                    }
                    let data = vc_for_stdout.lock().drain_output();
                    if !data.is_empty() {
                        // Write only valid UTF-8 prefix. Trailing
                        // incomplete sequences (from guest shutdown
                        // mid-write) are dropped to prevent garbled
                        // output.
                        let valid_len = match std::str::from_utf8(&data) {
                            Ok(_) => data.len(),
                            Err(e) => e.valid_up_to(),
                        };
                        if valid_len > 0 {
                            if stdout.write_all(&data[..valid_len]).is_err()
                                || stdout.flush().is_err()
                            {
                                kill_for_stdout.store(true, Ordering::Release);
                                break;
                            }
                            wrote_any = true;
                        }
                    }
                }
                // Final drain: the guest may have flushed output just
                // before shutdown that hasn't been polled yet.
                let data = vc_for_stdout.lock().drain_output();
                if !data.is_empty() {
                    let valid_len = match std::str::from_utf8(&data) {
                        Ok(_) => data.len(),
                        Err(e) => e.valid_up_to(),
                    };
                    if valid_len > 0 {
                        let _ = stdout.write_all(&data[..valid_len]);
                        let _ = stdout.flush();
                        wrote_any = true;
                    }
                }
                wrote_any
            })
            .context("spawn stdout writer thread")?;

        // Optional dmesg thread: COM1 -> stderr in real-time.
        // Only spawned when --dmesg is active. Gives the user kernel
        // messages (including virtio probe results) alongside the shell.
        //
        // The thread blocks in `epoll_wait` on two fds:
        //   * `data_evt` — bumped by `Serial::handle_out` whenever a
        //     guest port write appends a byte to COM1's captured-output
        //     buffer (see `Serial::install_data_evt`). Fires on every
        //     guest-side console write.
        //   * `dmesg_wakeup_evt` — a shutdown wakeup the BSP-cleanup
        //     code below pulses after flipping `kill` so the thread
        //     exits the wait promptly without sleep-polling.
        // Replaces a 50ms sleep+poll loop on `drain_output`.
        let (dmesg_thread, dmesg_wakeup_evt) = if self.dmesg {
            use std::os::unix::io::AsRawFd;
            use vmm_sys_util::epoll::{ControlOperation, Epoll, EpollEvent, EventSet};
            use vmm_sys_util::eventfd::{EFD_NONBLOCK, EventFd};

            let data_evt = com1
                .lock()
                .install_data_evt()
                .context("install COM1 dmesg data eventfd")?;
            let wakeup_evt =
                Arc::new(EventFd::new(EFD_NONBLOCK).context("create dmesg wakeup eventfd")?);
            let com1_for_dmesg = com1.clone();
            let kill_for_dmesg = kill.clone();
            let wakeup_for_thread = wakeup_evt.clone();
            const DATA_TOKEN: u64 = 0;
            const WAKEUP_TOKEN: u64 = 1;
            let handle = std::thread::Builder::new()
                .name("interactive-dmesg".into())
                .spawn(move || {
                    use std::io::Write;
                    let epoll = match Epoll::new() {
                        Ok(e) => e,
                        Err(e) => {
                            tracing::warn!(%e, "interactive-dmesg: epoll_create1 failed");
                            return;
                        }
                    };
                    if let Err(e) = epoll.ctl(
                        ControlOperation::Add,
                        data_evt.as_raw_fd(),
                        EpollEvent::new(EventSet::IN, DATA_TOKEN),
                    ) {
                        tracing::warn!(%e, "interactive-dmesg: add data_evt to epoll");
                        return;
                    }
                    if let Err(e) = epoll.ctl(
                        ControlOperation::Add,
                        wakeup_for_thread.as_raw_fd(),
                        EpollEvent::new(EventSet::IN, WAKEUP_TOKEN),
                    ) {
                        tracing::warn!(%e, "interactive-dmesg: add wakeup to epoll");
                        return;
                    }
                    let mut events = [EpollEvent::default(); 2];
                    // Lock stderr per-write, not for the whole loop.
                    // Holding the lock blocks Ctrl+A X's eprintln.
                    loop {
                        if kill_for_dmesg.load(Ordering::Acquire) {
                            break;
                        }
                        match epoll.wait(-1, &mut events) {
                            Ok(_) => {}
                            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                            Err(e) => {
                                tracing::warn!(%e, "interactive-dmesg: epoll_wait failed");
                                break;
                            }
                        }
                        // Drain both eventfd counters (counter mode —
                        // a single read returns the accumulated count
                        // and resets it; spurious EAGAIN from a racing
                        // refill is harmless).
                        let _ = data_evt.read();
                        let _ = wakeup_for_thread.read();
                        let data = com1_for_dmesg.lock().drain_output();
                        if !data.is_empty() {
                            let mut stderr = std::io::stderr().lock();
                            let _ = stderr.write_all(&data);
                            let _ = stderr.flush();
                        }
                    }
                    // Final drain.
                    let data = com1_for_dmesg.lock().drain_output();
                    if !data.is_empty() {
                        let mut stderr = std::io::stderr().lock();
                        let _ = stderr.write_all(&data);
                        let _ = stderr.flush();
                    }
                })
                .context("spawn dmesg thread")?;
            (Some(handle), Some(wakeup_evt))
        } else {
            (None, None)
        };

        // BSP run loop (same shutdown detection as run()).
        // Interactive sessions are user-controlled; the builder's timeout
        // (default 60s) must not kill the shell. Use 24 hours as a
        // practical upper bound.
        //
        // Apply the no-perf + --cpu-cap mask to the BSP thread so
        // interactive `ktstr shell --no-perf-mode --cpu-cap N` runs
        // inside the reserved LLCs just like run_vm's BSP. No pin
        // here — perf-mode doesn't apply to interactive shell:
        // `--cpu-cap` requires `--no-perf-mode` on Shell (clap
        // `requires` attribute on the cpu_cap field).
        if let Some(mask) = self.no_perf_plan.as_ref().map(|p| p.cpus.as_slice()) {
            set_thread_cpumask(mask, "BSP (shell)");
        }
        register_vcpu_signal_handler();
        let interactive_timeout = Duration::from_secs(24 * 60 * 60);
        // Exit code and timeout flag are unused here: interactive
        // exit status is not derived from the run-loop sentinel
        // (shell mode returns Ok(None)), and the 24h pseudo-timeout
        // above means `timed_out` never flips. The exit reason is
        // kept for the post-restore diagnostic line below.
        let (_exit_code, _timed_out, bsp_exit_reason) = self.run_bsp_loop(
            &mut bsp,
            &com1,
            &com2,
            Some(&virtio_con),
            virtio_blk.as_ref(),
            virtio_net.as_ref(),
            ioapic_handle.as_ref(),
            pci_bus_handle.as_ref(),
            &kill,
            &freeze,
            &watchpoint,
            &bsp_parked,
            &bsp_regs,
            has_immediate_exit,
            start,
            interactive_timeout,
            // Interactive shell never sets `freeze`, so the
            // handle_freeze branch is unreachable in this path.
            // Pass None for the wake-fd handles — the legacy
            // park_timeout cadence is the safe-by-construction
            // fallback.
            None,
            None,
            None,
            // Interactive shell does not construct a GuestKernel
            // for monitor / BPF map writes, so no TCR_EL1 cache
            // is needed.
            None,
            // CR3 cache: unused in interactive shell (no monitor
            // thread, no phys_base resolution).
            &std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
            // Interactive shell has no watchdog (24-hour timeout
            // is effectively disabled), so this flag never flips
            // and the BSP returns `timed_out=false` cleanly.
            &std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            // Interactive shell does not run the monitor/dump
            // pipeline that consumes the virt-KASLR offset, so the
            // BSP loop's MSR_LSTAR derive has no consumer. Pass a
            // throwaway Arc + EventFd (never read by any thread)
            // and a 0 link KVA to short-circuit the derive attempt.
            &std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
            &vmm_sys_util::eventfd::EventFd::new(0)
                .expect("eventfd for interactive-shell kern_virt_kaslr publish"),
            0,
        );

        // Shutdown.
        kill.store(true, Ordering::Release);

        // Wake the stdin reader so it exits poll() and can be joined.
        let _ = nix::unistd::write(&wakeup_w, &[0u8]);
        drop(wakeup_w);

        // Wake the dmesg thread so it exits epoll_wait promptly and
        // can be joined. The kill load above the loop short-circuits
        // any pending iteration; this bump ensures the wait returns
        // immediately rather than blocking on the next byte from the
        // guest after teardown.
        if let Some(ref evt) = dmesg_wakeup_evt {
            let _ = evt.write(1);
        }

        for vt in &ap_threads {
            if !vt.exited.load(Ordering::Acquire) {
                vt.kick();
            }
        }
        for vt in ap_threads {
            vt.wait_for_exit(Duration::from_secs(5));
            let _ = vt.handle.join();
        }

        let stdout_wrote = stdout_thread.join().unwrap_or(false);
        // The stdin reader and the exec watchdog are joined by
        // `kick_guard`'s Drop (which also covers the `?` early-returns +
        // panic-unwind); `kill` set above makes each return without
        // kicking. The guard drops before `bsp` (declared earlier), so
        // the joins precede the kvm_run unmap on every path.
        if let Some(dt) = dmesg_thread {
            let _ = dt.join();
        }
        drop(dmesg_wakeup_evt);

        // _raw_guard drops here, restoring terminal and signal handlers.
        drop(_raw_guard);

        // Terminate the guest's final echo line, deferred until after
        // the raw-mode restore: raw mode has ONLCR off, and the guest's
        // shell usually powers off before the hvc0->stdout forwarder
        // drains the echoed newline of the final `exit`, leaving the
        // cursor mid-line — without this the host prompt (or the
        // diagnostic below) glues onto the guest's final echo.
        if !exec_mode {
            eprintln!();
        }
        // Exit-reason diagnostic, only for abnormal exits (Fatal /
        // RunError / GuestPanic). A routine poweroff reaches here as
        // Shutdown or ExternalKill (see `BspExitReason::is_abnormal`)
        // and stays silent.
        if bsp_exit_reason.is_abnormal() {
            eprintln!("BSP: loop exit reason={bsp_exit_reason:?}");
        }

        // Surface any device-IRQ routing failures the userspace IOAPIC hit
        // during the run. The operator's terminal showed the guest console,
        // not the host's per-failure tracing, so an unrouted IRQ would
        // otherwise be a silent device hang. Printed after the raw-mode
        // restore so it lands on the operator's terminal. x86-only: the
        // userspace IOAPIC exists only on the split-irqchip path (aarch64
        // routes device IRQs via the GIC, and `IoapicHandle` is an
        // empty-enum placeholder there with no `routing_failures`).
        #[cfg(target_arch = "x86_64")]
        if let Some(io) = &ioapic_handle
            && let Some(msg) = routing_failure_summary(io.routing_failures())
        {
            eprintln!("{msg}");
        }

        // Same surfacing for the full-irqchip MSI-X route owner (the inverse of
        // ioapic_handle: Some only on the in-kernel-irqchip path). A failed
        // KVM_SET_GSI_ROUTING for an MSI-X vector leaves that vector unrouted —
        // the NIC hangs on first use — so an unrouted MSI-X install must not be
        // silent on the operator's terminal in shell mode either.
        #[cfg(target_arch = "x86_64")]
        if let Some(owner) = &full_route_owner
            && let Some(msg) = routing_failure_summary(owner.routing_failures())
        {
            eprintln!("{msg}");
        }

        // Exec mode fallback: if virtio-console produced no output
        // (kernel lacks CONFIG_VIRTIO_CONSOLE, guest fell back to
        // COM2), print COM2 output to stdout so the caller sees it.
        // No sentinel appears on COM2: the exec exit is a typed
        // `MSG_TYPE_EXEC_EXIT` bulk-port frame (see the inner
        // comment below), so the COM2 bytes are written verbatim.
        if exec_mode && !stdout_wrote {
            let app_output = com2.lock().output();
            if !app_output.is_empty() {
                use std::io::Write;
                let mut stdout = std::io::stdout().lock();
                // Pre-bulk-port-migration the guest emitted a
                // `KTSTR_EXEC_EXIT=N` sentinel line on COM2 that
                // needed filtering out of this stdout copy. The
                // exec exit is now a typed `MSG_TYPE_EXEC_EXIT`
                // frame on the bulk data port (see
                // `crate::vmm::guest_comms::send_exec_exit`), so
                // the sentinel never appears in COM2 — no filter
                // needed. Write the captured bytes verbatim.
                let _ = stdout.write_all(app_output.as_bytes());
                let _ = stdout.flush();
            }
        }

        // Print kernel console output (COM1) to stderr if non-empty.
        // Skip when --dmesg was active (already streamed to stderr).
        // `output_display` (not `output`) so the serial8250 IRQ-probe
        // 0xFF byte doesn't surface: on a quiet boot it is COM1's only
        // content, and a raw dump printed a lone `�` here.
        if !self.dmesg {
            let console_output = com1.lock().output_display();
            if !console_output.is_empty() {
                eprintln!("{console_output}");
            }
        }

        if !exec_mode {
            return Ok(None);
        }

        // Exec mode: recover the payload's exit code from the framed
        // `MSG_TYPE_EXEC_EXIT` the guest published on the bulk port
        // (port 1) just before reboot (`guest_comms::send_exec_exit`
        // writes `code.to_le_bytes()`). Shell mode does NOT run the
        // freeze-coordinator's bulk dispatch, so this is the sole
        // host-side consumer of the frame on the interactive path —
        // without it the payload's exit code is silently lost and the
        // CLI always reports success. `final_drain` walks port 1's
        // avail ring once to pick up any chain published without a
        // trailing QUEUE_NOTIFY, then returns the accumulated TX bytes.
        let bulk = virtio_con.lock().final_drain();
        let entries = crate::vmm::host_comms::parse_tlv_stream(&bulk).entries;
        match Self::exec_exit_from_entries(&entries) {
            Some(code) => Ok(Some(code)),
            // no-silent-drops: the guest always emits a CRC-valid
            // ExecExit before reboot in `--exec` mode, so its absence
            // means the exit code was lost (the guest panicked or
            // rebooted before send_exec_exit). Fail loud rather than
            // masking it as exit 0 — that silent default is the exact
            // regression this consumer closes.
            None => {
                // A watchdog timeout is the most likely cause of a
                // missing EXEC_EXIT frame (the guest was force-killed
                // mid-payload) — report it distinctly and actionably.
                if timed_out.load(Ordering::Acquire) {
                    anyhow::bail!(
                        "shell --exec '{}' exceeded the {:?} exec-timeout and \
                         was force-killed; the payload's exit code is unknown. \
                         Raise --exec-timeout for a legitimately long-running \
                         payload.",
                        self.exec_cmd.as_deref().unwrap_or("?"),
                        self.exec_timeout,
                    )
                }
                // Surface the likely cause from the captured guest
                // console rather than a bare frame-missing error — the
                // OOM/panic/abort line is in COM1/COM2 even when no
                // EXEC_EXIT frame arrived.
                let com1_out = com1.lock().output();
                let com2_out = com2.lock().output();
                anyhow::bail!(
                    "shell --exec '{}' finished but the guest delivered no CRC-valid \
                     MSG_TYPE_EXEC_EXIT frame; the payload's exit code is unknown.{}",
                    self.exec_cmd.as_deref().unwrap_or("?"),
                    Self::detect_guest_failure(&com1_out, &com2_out)
                )
            }
        }
    }

    /// Recover the shell `--exec` payload's exit code from the drained
    /// bulk-port frames: the LAST CRC-valid 4-byte `MSG_TYPE_EXEC_EXIT`
    /// frame, decoded as a little-endian `i32` (matching the guest's
    /// `send_exec_exit` `to_le_bytes`). Returns `None` when no such
    /// frame is present — the caller treats that as a lost exit code and
    /// fails loud. CRC-failed and wrong-length frames are skipped: a
    /// torn frame must never promote into a bogus exit code. Last-wins
    /// is defensive (the guest sends exactly one per exec, then reboots)
    /// and mirrors the freeze coordinator's MSG_TYPE_EXIT walker.
    fn exec_exit_from_entries(entries: &[crate::vmm::wire::ShmEntry]) -> Option<i32> {
        entries
            .iter()
            .rev()
            .find(|e| {
                e.msg_type == crate::vmm::wire::MSG_TYPE_EXEC_EXIT
                    && e.crc_ok
                    && e.payload.len() == 4
            })
            .map(|e| i32::from_le_bytes(e.payload[..4].try_into().unwrap()))
    }

    /// Scan captured guest console output (COM1 kernel console + COM2
    /// init/app stderr) for a failure signature, returning a cause
    /// suffix for the no-EXEC_EXIT-frame error. Falls back to the
    /// generic panic/reboot hint when no known signature matches.
    ///
    /// Markers, in priority order: the Rust alloc-error
    /// ("memory allocation of N bytes failed" — the guest /init aborting
    /// on a failed allocation; lands on COM2 in shell mode), then a
    /// kernel panic ("Kernel panic - not syncing" / "Attempted to kill
    /// init", anchored so a log line mentioning the bare words cannot
    /// false-match). The returned string is appended verbatim to the
    /// error and so begins with a space.
    fn detect_guest_failure(com1: &str, com2: &str) -> String {
        const ALLOC_FAIL: &str = "memory allocation of";
        const PANIC: &str = "Kernel panic - not syncing";
        const INIT_KILL: &str = "Attempted to kill init";
        // The echoed line is attacker-influenced guest console output:
        // `lines()` splits on '\n', so a newline-free line can run up to
        // the console's OUTPUT_CAP_BYTES (4 MiB) cap. Bound it so a
        // pathological guest cannot bloat the error string to megabytes;
        // the marker sits near the line start, so the head is the
        // informative part. char-boundary-safe (no byte slicing).
        fn trunc(line: &str) -> String {
            const MAX_CHARS: usize = 200;
            let t = line.trim();
            if t.chars().count() > MAX_CHARS {
                let head: String = t.chars().take(MAX_CHARS).collect();
                format!("{head}…")
            } else {
                t.to_string()
            }
        }
        // COM2 first: the Rust alloc-error is the most actionable and
        // lands there in shell mode (init stdio is dup2'd to COM2).
        for hay in [com2, com1] {
            if let Some(line) = hay.lines().find(|l| l.contains(ALLOC_FAIL)) {
                return format!(
                    " The guest /init aborted on a failed allocation: '{}'. The \
                     /init is the test binary itself — raise memory_mib or check \
                     the guest overcommit policy (vm.overcommit_memory).",
                    trunc(line)
                );
            }
        }
        for hay in [com1, com2] {
            if let Some(line) = hay
                .lines()
                .find(|l| l.contains(PANIC) || l.contains(INIT_KILL))
            {
                return format!(" Guest kernel panic: '{}'.", trunc(line));
            }
        }
        " (the guest may have panicked or rebooted before send_exec_exit)".to_string()
    }
}

#[cfg(test)]
mod tests;
