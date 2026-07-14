//! [`KtstrVmBuilder`] — public configuration surface for [`super::KtstrVm`].
//!
//! Test authors compose a VM by chaining the setters defined here, then
//! call [`KtstrVmBuilder::build`] to produce a runnable [`super::KtstrVm`].
//! The builder is the only path that constructs a VM — every field on
//! the runtime [`super::KtstrVm`] struct flows through one of the setters
//! plus the `build()` validator, which performs host-resource gating
//! (LLC reservation, hugepage probe, memory_mib sanity check) before
//! handing the VM back to the caller.
//!
//! Helpers `build_per_node_map` and `acquire_slot_with_locks` live next
//! to `build()` because they execute as part of the build pipeline:
//! `build_per_node_map` is called from `resolve_run_plans` and
//! `acquire_slot_with_locks` from `validate_performance_mode` (neither
//! directly from `build()`), and they cooperate with the
//! [`super::host_topology`] flock primitives to reserve the LLC slots
//! the resulting VM will pin against.

use anyhow::{Context, Result};
use std::path::PathBuf;
use std::time::Duration;

use super::host_topology;
use super::net_config;
use super::topology::{self, Topology};
use super::vcpu::{BpfMapWriteParams, WatchBpfMapParams};
use super::{KtstrVm, disk_config, initramfs};

/// Builder for [`super::KtstrVm`].
///
/// Obtain via [`super::KtstrVm::builder()`], configure with the chained
/// setters below, then call [`build`](Self::build) to validate the
/// configuration and materialise a `KtstrVm`. Required inputs are a
/// `kernel` source directory or image, an `init_binary`, and either
/// a `run_args` payload (for test runs) or an `exec_cmd` / shell
/// configuration (for `ktstr shell`). Everything else is optional.
///
/// # Defaults
///
/// Field defaults applied by [`Default::default`]:
/// - `memory_mib` — 256 MiB (overridden by [`memory_mib`](Self::memory_mib))
/// - `timeout` — 12 s (overridden by [`timeout`](Self::timeout))
/// - `watchdog_timeout` — 5 s (overridden by [`watchdog_timeout`](Self::watchdog_timeout))
/// - `topology` — 1 NUMA node × 1 LLC × 1 core × 1 thread (overridden
///   by [`topology`](Self::topology))
/// - `performance_mode` — `false` (operator opts in via
///   [`performance_mode`](Self::performance_mode))
///
/// `Clone` lets a cell rebuild a fresh VM from the same configuration
/// across boot attempts — the AP-bring-up-gap boot retry
/// (`crate::test_support::run_vm_with_ap_gap_retry`) clones before each
/// `build()` because `build` consumes the builder. The clone is cheap on
/// a test-mode VM (no busybox / large embedded blobs).
#[derive(Clone)]
pub struct KtstrVmBuilder {
    kernel: Option<PathBuf>,
    init_binary: Option<PathBuf>,
    scheduler_binary: Option<PathBuf>,
    /// Additional schedulers packed into the initramfs alongside
    /// the boot-time `scheduler_binary` so future scheduler-
    /// lifecycle ops can swap mid-experiment. Empty for the common
    /// single-scheduler case — pays zero initramfs cost when not
    /// populated. See [`StagedScheduler`] for the per-entry shape
    /// and the doc on
    /// [`Self::staged_scheduler`](#method.staged_scheduler) for
    /// the builder-level contract.
    staged_schedulers: Vec<StagedScheduler>,
    run_args: Vec<String>,
    sched_args: Vec<String>,
    pub(crate) topology: Topology,
    pub(crate) memory_mib: Option<u32>,
    memory_min_mib: u32,
    /// Per-test no-perf host-CPU budget override (`#[ktstr_test(cpu_budget)]`).
    /// `None` auto-sizes to the vCPU count; `Some(n)` forces that budget
    /// (n < vcpus → overcommit). An explicit `--cpu-cap` still wins.
    pub(crate) cpu_budget: Option<u32>,
    pub(crate) cmdline_extra: String,
    pub(crate) timeout: Duration,
    pub(crate) monitor_thresholds: Option<crate::monitor::MonitorThresholds>,
    pub(crate) watchdog_timeout: Option<Duration>,
    pub(crate) rendezvous_timeout: Option<Duration>,
    bpf_map_writes: Vec<BpfMapWriteParams>,
    watch_bpf_maps: Vec<WatchBpfMapParams>,
    pub(crate) performance_mode: bool,
    /// Whether to expose the virtio-PCI transport in the guest. See the
    /// `pci` setter.
    pub(crate) pci_enabled: bool,
    no_perf_mode: bool,
    sched_enable_cmds: Vec<String>,
    sched_disable_cmds: Vec<String>,
    include_files: Vec<(String, PathBuf)>,
    /// Raw (already-decompressed) `.ko` images to load in the guest,
    /// in load order, before it touches any virtio device. Empty for
    /// ktstr-built kernels (virtio =y); populated for prebuilt distro
    /// kernels (virtio =m). See [`KtstrVmBuilder::kernel_modules`].
    kernel_modules: Vec<PathBuf>,
    /// Initrd compression the guest kernel can unpack. `Lz4` for
    /// ktstr-built kernels (ktstr.kconfig pins `CONFIG_RD_LZ4=y`);
    /// prebuilt distro kernels that lack `RD_LZ4` need a format from
    /// their own `CONFIG_RD_*` set. See
    /// [`KtstrVmBuilder::initrd_compression`].
    initrd_compression: initramfs::InitrdCompression,
    /// v0 holds at most one DiskConfig; rendered as `/dev/vda`.
    /// The optional single virtio-blk disk (set by `.disk()`). ktstr wires one
    /// blk device; a single backing disk suffices for scheduler tests.
    /// Multi-disk would mean N virtio-blk PCI functions (mirroring multi-NIC)
    /// and can be re-introduced pre-1.0 if a real consumer appears. See
    /// [`super::KtstrVm::disk`].
    disk: Option<disk_config::DiskConfig>,
    /// Network devices (appended by `.network()`). Empty skips virtio-net
    /// entirely (no PCI NIC function / FDT node, no IRQ). Each entry attaches
    /// one virtio-net device; the in-VMM loopback backend echoes TX bytes back
    /// to RX. On x86_64 each NIC is a virtio-pci function on its own slot/GSI
    /// (`kvm::virtio_net_pci_slot` / `kvm::virtio_net_gsi`), bounded by
    /// `kvm::MAX_VIRTIO_NICS`; aarch64 supports a single MMIO NIC. See
    /// [`super::KtstrVm::networks`].
    networks: Vec<net_config::NetConfig>,
    /// Busybox bytes to pack at `bin/busybox`. `None` skips packing
    /// (test-mode VMs do not need shell utilities). `Some(bytes)`
    /// embeds the provided bytes — the library never owns busybox
    /// itself; bytes come from
    /// [`crate::vmm::blobs::load_busybox_bytes`] (which reads the
    /// `KTSTR_BUSYBOX_PATH` env var that `cargo-ktstr` sets at
    /// startup).
    pub(crate) busybox_bytes: Option<Vec<u8>>,
    #[cfg(feature = "wprof")]
    pub(crate) wprof: Option<crate::vmm::wprof::WprofConfig>,
    dmesg: bool,
    exec_cmd: Option<String>,
    /// Wall-clock bound for a shell `--exec` payload. A panic-less
    /// guest hang otherwise blocks the BSP run loop ~forever; the
    /// `run_interactive` watchdog kicks the vCPU after this deadline.
    /// Default 120s; consulted only in `--exec` (exec_mode) runs.
    exec_timeout: Duration,
    /// Optional host path to the `ktstr-jemalloc-probe` binary.
    /// When `Some`, the probe is packed into the guest initramfs at
    /// `bin/ktstr-jemalloc-probe` and becomes spawnable by bare name
    /// inside the guest — used by the closed-loop probe tests in
    /// `tests/jemalloc_probe_tests.rs`.
    jemalloc_probe_binary: Option<PathBuf>,
    /// Optional host path to `ktstr-jemalloc-alloc-worker`. When
    /// `Some`, packed into the initramfs at `bin/ktstr-jemalloc-
    /// alloc-worker`. Used together with `jemalloc_probe_binary` for the
    /// cross-process closed-loop test.
    jemalloc_alloc_worker_binary: Option<PathBuf>,
    /// File path where the freeze coordinator writes the
    /// JSON-pretty failure-dump report. `None` disables the file
    /// sink — the dump still emits via `tracing::error`. See
    /// [`Self::failure_dump_path`].
    failure_dump_path: Option<PathBuf>,
    /// Capture two BPF-state snapshots per VM run instead of one.
    /// See the runtime field of the same name on [`super::KtstrVm`] for
    /// the full contract; the builder field flows through `build`
    /// unchanged.
    dual_snapshot: bool,
    /// When set, `KtstrVm::init_virtio_blk` opens this path
    /// directly as the virtio-blk backing file instead of allocating
    /// a fresh `tempfile()` (Raw branch) or invoking
    /// [`super::disk_template::ensure_template`] (Btrfs branch). The
    /// path-supplied backing exists exclusively for the
    /// disk-template-build VM driver in
    /// `super::disk_template::build_template_via_vm`: that driver
    /// materialises a sparse staging image, points the template VM
    /// at it via this field, and recovers the now-formatted file
    /// after VM exit for [`super::disk_template::store_atomic`] to
    /// publish. Setting this from any other code path bypasses the
    /// template cache and is ALMOST CERTAINLY a mistake —
    /// per-test runs want the `Raw` tempfile or `Btrfs` cache
    /// branches in `init_virtio_blk`. `None` is the production
    /// default.
    template_staging_image: Option<PathBuf>,
    /// Workload time budget (the test's `duration`), distinct from
    /// the outer kill `timeout`. When set, the host-side watchdog
    /// resets its hard deadline to `now + workload_duration` the
    /// first time the host monitor observes `*scx_root` transition
    /// from null to non-null in guest memory — i.e. the moment a
    /// scheduler attaches and the workload's clock should start.
    /// The reset CAN extend past the original `timeout`-derived
    /// deadline (the watchdog uses `reset.unwrap_or(original)` with
    /// no min clamp), so boot-time delays do not eat into the
    /// workload budget. `None` (the default) disables the reset and
    /// the watchdog uses `timeout` as a single deadline counted from
    /// VM boot.
    workload_duration: Option<Duration>,
    /// Periodic snapshot count plumbed onto [`super::KtstrVm`]; see
    /// the runtime field for the full contract. `0` disables the
    /// periodic-capture loop in the freeze coordinator entirely
    /// (the default).
    num_snapshots: u32,
    /// Optional per-test workload-cgroup root. Sourced from
    /// [`crate::test_support::KtstrTestEntry::workload_root_cgroup`].
    /// When set, the guest init mkdir's the path BEFORE starting
    /// the scheduler and the guest CgroupManager uses it as the
    /// parent for every workload cgroup the test declares; when
    /// unset (the default), the guest falls back to its legacy
    /// `--cell-parent-cgroup`-or-default resolution.
    workload_root_cgroup: Option<String>,
    /// Per-scheduler cgroup the scheduler process is placed in.
    /// Sourced from
    /// [`crate::test_support::Scheduler::cgroup_parent`]. When
    /// set, the guest init mkdir's the path + enables `+cpuset
    /// +cpu` on every ancestor's `subtree_control` BEFORE starting
    /// the scheduler. Distinct from
    /// [`Self::workload_root_cgroup`] (workload placement); the
    /// two slots cover different concerns and either, both, or
    /// neither may be set.
    scheduler_cgroup_parent: Option<String>,
}

/// One scheduler staged into the guest initramfs alongside the
/// boot-time `scheduler_binary` so the scheduler-lifecycle Ops
/// (`Op::AttachScheduler` / `Op::ReplaceScheduler`) can swap to a
/// different scheduler mid-experiment without rebooting the VM.
///
/// `name` is the [`Scheduler::name`](crate::test_support::Scheduler::name)
/// of the source entry — must satisfy the
/// [`crate::test_support::staged::validate_staged_scheduler_name`]
/// shape rules (callers pre-validate at
/// `KtstrTestEntry::validate` time, before any `KtstrVmBuilder`
/// staging surface fires). `binary` is the host-side resolved
/// PathBuf the initramfs pipeline copies into the guest at
/// `/staging/schedulers/<name>/scheduler` per the
/// [`crate::test_support::staged::staged_scheduler_binary_path`]
/// mapping. `sched_args` is the per-scheduler CLI argv that
/// future Op-dispatch reads from
/// `/staging/schedulers/<name>/sched_args` at spawn time.
#[derive(Debug, Clone)]
pub(crate) struct StagedScheduler {
    pub(crate) name: String,
    pub(crate) binary: PathBuf,
    pub(crate) sched_args: Vec<String>,
}

impl Default for KtstrVmBuilder {
    /// Minimal-viable VM seed — `1 LLC × 1 core × 1 thread × 1 NUMA
    /// node = 1 vCPU`, 256 MiB guest RAM, no kernel/init/scheduler
    /// binaries set yet (those are Required-Before-Build per
    /// `Self::build` validation). The 1×1×1×1 topology is the
    /// smallest legal value (`Topology::new` rejects any zero
    /// dimension); test authors override this via `Self::topology(...)`
    /// or attribute-built entries (`#[ktstr_test(llcs=N, cores=M,
    /// threads=K)]`). The 256-MiB memory floor matches the
    /// guest-init initramfs RAM cost; tests needing larger workloads
    /// raise it via `Self::memory_mib(...)`. Every other field
    /// (timeouts, watchdog, bpf_map_writes, ...) defaults to either
    /// `None` (deferred) or an empty collection — no kernel-write
    /// values, so no rejected-by-kernel risk.
    fn default() -> Self {
        KtstrVmBuilder {
            kernel: None,
            init_binary: None,
            scheduler_binary: None,
            staged_schedulers: Vec::new(),
            run_args: Vec::new(),
            sched_args: Vec::new(),
            topology: Topology {
                llcs: 1,
                cores_per_llc: 1,
                threads_per_core: 1,
                numa_nodes: 1,
                nodes: None,
                distances: None,
                llc_cores: None,
            },
            memory_mib: Some(256),
            memory_min_mib: 0,
            cpu_budget: None,
            cmdline_extra: String::new(),
            timeout: Duration::from_secs(12),
            monitor_thresholds: None,
            watchdog_timeout: Some(Duration::from_secs(5)),
            rendezvous_timeout: None,
            bpf_map_writes: Vec::new(),
            watch_bpf_maps: Vec::new(),
            performance_mode: false,
            pci_enabled: false,
            no_perf_mode: false,
            sched_enable_cmds: Vec::new(),
            sched_disable_cmds: Vec::new(),
            include_files: Vec::new(),
            kernel_modules: Vec::new(),
            initrd_compression: initramfs::InitrdCompression::Lz4,
            disk: None,
            networks: Vec::new(),
            busybox_bytes: None,
            #[cfg(feature = "wprof")]
            wprof: None,
            dmesg: false,
            exec_cmd: None,
            exec_timeout: Duration::from_secs(120),
            jemalloc_probe_binary: None,
            jemalloc_alloc_worker_binary: None,
            failure_dump_path: None,
            dual_snapshot: false,
            template_staging_image: None,
            workload_duration: None,
            num_snapshots: 0,
            workload_root_cgroup: None,
            scheduler_cgroup_parent: None,
        }
    }
}

/// Run-time CPU/memory placement plans resolved by
/// [`KtstrVmBuilder::resolve_run_plans`].
struct RunPlans {
    pinning_plan: Option<host_topology::PinningPlan>,
    mbind_node_map: Vec<Vec<usize>>,
    no_perf_plan: Option<host_topology::LlcPlan>,
    host_topo: Option<host_topology::HostTopology>,
    /// Effective `CpuCap` the no-perf build-time plan targeted, so the
    /// run-time replan in `acquire_run_locks` reuses the same budget.
    /// `None` on every non-no-perf / degraded path.
    no_perf_effective_cap: Option<host_topology::CpuCap>,
}

impl KtstrVmBuilder {
    /// Path to the guest kernel IMAGE file (a prebuilt `bzImage` /
    /// `Image`); [`build`](Self::build) loads it verbatim and does NOT
    /// extract a source tree, so a directory here fails at load with
    /// "Unable to read bzImage header". For a source-tree ROOT use
    /// [`kernel_dir`](Self::kernel_dir) (resolves `arch/<arch>/boot/…`),
    /// or resolve the image yourself first (e.g.
    /// `crate::kernel_path::find_image_in_dir`, which also handles the
    /// cache `<dir>/bzImage` layout).
    pub fn kernel(mut self, path: impl Into<PathBuf>) -> Self {
        self.kernel = Some(path.into());
        self
    }

    /// Path to the userspace init binary run as PID 1 inside the
    /// guest (typically the current test binary).
    pub fn init_binary(mut self, path: impl Into<PathBuf>) -> Self {
        self.init_binary = Some(path.into());
        self
    }

    /// Path to an optional scheduler binary loaded alongside the
    /// init binary; the init spawns it before dispatching the test.
    pub fn scheduler_binary(mut self, path: impl Into<PathBuf>) -> Self {
        self.scheduler_binary = Some(path.into());
        self
    }

    /// Stage one additional scheduler into the guest initramfs at
    /// `/staging/schedulers/<name>/scheduler` + per-scheduler args
    /// at `/staging/schedulers/<name>/sched_args`. Future
    /// scheduler-lifecycle ops (`Op::AttachScheduler` /
    /// `Op::ReplaceScheduler`) resolve a `&'static Scheduler` to
    /// its staged path via
    /// [`crate::test_support::staged::staged_scheduler_binary_path`].
    ///
    /// Caller responsibility: pre-validate `name` via the
    /// [`crate::test_support::staged::validate_staged_scheduler_name`]
    /// shape rules — `KtstrTestEntry::validate` is the production
    /// gate. The builder accepts whatever passes through; it does
    /// NOT re-validate. Duplicate names within the staged set
    /// would land at the same guest path and silently overwrite —
    /// the validate gate must catch them upstream.
    ///
    /// Idempotent only by collection-level semantics: calling
    /// `staged_scheduler` twice with the SAME name pushes two
    /// entries that the initramfs packer would resolve to the
    /// same guest path. The packing pipeline (follow-up work)
    /// rejects such duplicates at build time as the
    /// final-line-of-defense beyond the validate gate.
    pub fn staged_scheduler(
        mut self,
        name: impl Into<String>,
        binary: impl Into<PathBuf>,
        sched_args: Vec<String>,
    ) -> Self {
        self.staged_schedulers.push(StagedScheduler {
            name: name.into(),
            binary: binary.into(),
            sched_args,
        });
        self
    }

    /// CLI argv passed to the init binary inside the guest (typically
    /// the per-test dispatch string like `--ktstr-test-fn NAME`).
    pub fn run_args(mut self, args: &[String]) -> Self {
        self.run_args = args.to_vec();
        self
    }

    /// Extra CLI arguments appended to the scheduler binary invocation.
    #[allow(dead_code)]
    pub fn sched_args(mut self, args: &[String]) -> Self {
        self.sched_args = args.to_vec();
        self
    }

    /// Resolve the kernel image from a source-tree root (sets
    /// `kernel` to `arch/<arch>/boot/<image>`).
    #[allow(dead_code)]
    pub fn kernel_dir(mut self, path: impl Into<PathBuf>) -> Self {
        let dir: PathBuf = path.into();
        #[cfg(target_arch = "x86_64")]
        {
            self.kernel = Some(dir.join("arch/x86/boot/bzImage"));
        }
        #[cfg(target_arch = "aarch64")]
        {
            self.kernel = Some(dir.join("arch/arm64/boot/Image"));
        }
        self
    }

    /// Set the virtual CPU topology.
    ///
    /// For uniform topologies, build with [`Topology::new`]. For
    /// per-node configuration (asymmetric memory, CXL nodes, custom
    /// distances), use [`Topology::with_nodes`] / [`Topology::distances`].
    pub fn topology(mut self, topo: Topology) -> Self {
        self.topology = topo;
        self
    }

    /// Pin guest memory to an explicit MiB value and clear the
    /// deferred-sizing hint. Use `memory_deferred` when the payload
    /// size should drive the allocation. Only tests pin memory this
    /// way now; every production VM sizes via `memory_deferred_min`.
    #[allow(dead_code)]
    pub fn memory_mib(mut self, mib: u32) -> Self {
        self.memory_mib = Some(mib);
        self.memory_min_mib = 0;
        self
    }

    /// Defer memory allocation until after the initramfs is built.
    ///
    /// Memory will be computed from the actual initramfs size. Use this
    /// when no explicit `--memory` override is provided.
    pub fn memory_deferred(mut self) -> Self {
        self.memory_mib = None;
        self.memory_min_mib = 0;
        self
    }

    /// Defer memory allocation with a minimum floor. The deferred path
    /// computes memory from actual initramfs size, then takes the max
    /// of that and `min_mib`. Use when the topology needs more memory
    /// than the initramfs alone requires (e.g. NUMA tests with 4096 MiB).
    pub fn memory_deferred_min(mut self, min_mib: u32) -> Self {
        self.memory_mib = None;
        self.memory_min_mib = min_mib;
        self
    }

    /// Override the no-perf host-CPU budget — the number of host CPUs the
    /// VM's vCPU threads share. The default (unset) auto-sizes to the VM's
    /// vCPU count; setting `budget` < vcpus forces CPU overcommit (used by
    /// `#[ktstr_test(cpu_budget = N)]` for contention tests). Only takes
    /// effect on the no-perf path; an explicit `--cpu-cap` / `KTSTR_CPU_CAP`
    /// overrides it. This setter stores `budget` verbatim; a value of 0 is
    /// clamped to >= 1 only when `build()` resolves the no-perf CPU cap, so
    /// no zero-CPU mask is ever produced. The `#[ktstr_test]` macro and
    /// `KtstrTestEntry::validate` reject 0 before it reaches this setter.
    pub fn cpu_budget(mut self, budget: u32) -> Self {
        self.cpu_budget = Some(budget);
        self
    }

    /// Append extra tokens to the guest kernel command line. Useful
    /// for one-off debug knobs (e.g. enabling extra subsystem
    /// verbosity) that shouldn't live in `ktstr.kconfig`.
    #[allow(dead_code)]
    pub fn cmdline(mut self, extra: &str) -> Self {
        self.cmdline_extra = extra.to_string();
        self
    }

    /// Host-side watchdog timeout. The VM is killed if it has not
    /// exited on its own within this duration; the `VmResult`
    /// returned will have `timed_out = true`.
    pub fn timeout(mut self, t: Duration) -> Self {
        self.timeout = t;
        self
    }

    /// Workload time budget (the test's `duration`). When set, the
    /// host-side watchdog resets its hard deadline to
    /// `now + workload_duration` the first time the monitor
    /// observes `*scx_root` transition from null to non-null —
    /// i.e. the moment a scheduler attaches and the workload's
    /// clock should start. The reset CAN extend past the original
    /// `timeout`-derived deadline (no min clamp), so boot-time
    /// delays do not eat into the workload budget. `None` (the
    /// default) disables the reset.
    pub fn workload_duration(mut self, d: Duration) -> Self {
        self.workload_duration = Some(d);
        self
    }

    /// Override the `MonitorThresholds` used for stuck detection and
    /// verdict rendering. Defaults to `MonitorThresholds::new()`.
    #[allow(dead_code)]
    pub fn monitor_thresholds(mut self, thresholds: crate::monitor::MonitorThresholds) -> Self {
        self.monitor_thresholds = Some(thresholds);
        self
    }

    /// File path where the freeze coordinator writes the JSON-pretty
    /// [`crate::monitor::dump::FailureDumpReport`] when an
    /// error-class SCX exit fires. `None` (the default) disables
    /// the file sink — the dump still emits via `tracing::error`
    /// regardless. The test framework's primary dispatch path in
    /// `test_support::eval` sets this per-test under the run's
    /// sidecar directory so structured failure data sits alongside
    /// `*.ktstr.json`; the auto-repro path in
    /// `test_support::probe::attempt_auto_repro` overrides it to a
    /// `.repro.failure-dump.json` sibling; CLI / library callers
    /// that want the dump on disk set it explicitly here.
    ///
    /// Pure setter — no filesystem side effects. Stale-file
    /// pre-clear is the dispatch layer's responsibility (primary:
    /// `test_support::eval`, which clears BOTH the primary path
    /// AND the repro path on every dispatch so a passing rerun
    /// is not masked by either of the prior failure's leftovers;
    /// auto-repro: `test_support::probe::attempt_auto_repro`
    /// implicitly relies on the primary dispatch's pre-clear of
    /// the repro path before falling into the repro VM build).
    pub fn failure_dump_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.failure_dump_path = Some(path.into());
        self
    }

    /// Enable the dual-snapshot freeze-coordinator path. With
    /// `enabled = true` the coordinator runs an additional per-CPU
    /// `runnable_at` scanner alongside the existing
    /// `ktstr_err_exit_detected` poll: when any task crosses the
    /// `watchdog_timeout/2` half-way mark it triggers an extra
    /// freeze + dump cycle. Both snapshots are emitted as a single
    /// [`crate::monitor::dump::DualFailureDumpReport`] file at
    /// [`Self::failure_dump_path`] (the late snapshot at the same
    /// trigger as the single-snapshot path; the early snapshot is
    /// optional). Used by the auto-repro path to capture BPF state
    /// deltas across a stall window.
    ///
    /// Default off — two reasons:
    /// 1. **Scanner cost.** The early-trigger path walks the
    ///    kernel's global `scx_tasks` list AND every per-CPU
    ///    `rq->scx.runnable_list` once per scan tick (250 ms),
    ///    reading each task's `task_struct.scx.runnable_at` via
    ///    direct-mapped guest memory. On a 64-vCPU host with
    ///    hundreds of runnable tasks the steady-state cost is
    ///    non-negligible — a primary VM doesn't pay it unless
    ///    the run already failed and an auto-repro is being
    ///    attempted.
    /// 2. **Consumer compatibility.** The on-disk shape changes
    ///    from [`crate::monitor::dump::FailureDumpReport`] to
    ///    [`crate::monitor::dump::DualFailureDumpReport`], a
    ///    different JSON schema. Any consumer reading the dump
    ///    file must handle both schemas (gated on the `schema`
    ///    field). Keeping the primary path on the single-snapshot
    ///    shape means existing consumers (e.g.
    ///    `tests/failure_dump_e2e.rs`) keep working without
    ///    awareness of the dual-snapshot wrapper.
    pub fn dual_snapshot(mut self, enabled: bool) -> Self {
        self.dual_snapshot = enabled;
        self
    }

    /// Number of equally-spaced periodic snapshots to fire inside
    /// the capturable window's 10%–90% slice. `0` (the default)
    /// disables periodic capture entirely. The window is
    /// `[max(scenario_start, prereqs_ready), scenario_start +
    /// workload_duration]`: the start floats to the prereq-ready
    /// moment (kaslr + BPF-accessor attach) so cold-boot latency
    /// cannot strand boundaries pre-ready, and the end is CLAMPED
    /// to the workload end so captures never spill into
    /// post-workload idle. On a warm boot the start equals
    /// `scenario_start`, so the window is the full
    /// `[scenario_start, scenario_start + workload_duration]` and
    /// boot + verifier time do not eat the budget; on a cold boot
    /// the window starts later and is shorter, so cross-run compares
    /// of the window-averaged keys `avg_cpu_util_comp_scale` /
    /// `avg_task_lat_cri` should use `--noise-adjust`. Each fire
    /// runs the same
    /// `freeze_and_dispatch(FreezeMode::Capture { gate_on_exit_kind: false })`
    /// path the
    /// on-demand `Op::CaptureSnapshot` handler uses and stores under
    /// `"periodic_NNN"` on the host's
    /// [`crate::scenario::snapshot::SnapshotBridge`]. Bounded above
    /// by [`crate::scenario::snapshot::MAX_STORED_SNAPSHOTS`] —
    /// `KtstrTestEntry::validate` rejects higher values so the
    /// bridge's FIFO eviction never silently drops periodic
    /// samples.
    pub fn num_snapshots(mut self, n: u32) -> Self {
        self.num_snapshots = n;
        self
    }

    /// Set the per-test workload-cgroup root. The guest init
    /// mkdir's `/sys/fs/cgroup{path}` BEFORE starting the
    /// scheduler and the guest CgroupManager uses it as the parent
    /// for every workload cgroup declared via
    /// [`Ctx::cgroup_def`](crate::scenario::Ctx::cgroup_def).
    ///
    /// `path` must be an absolute cgroup path (leading `/`,
    /// not bare `/`); programmatic callers should pass values
    /// already validated against
    /// [`crate::test_support::CgroupPath::new`].
    pub fn workload_root_cgroup(mut self, path: impl Into<String>) -> Self {
        self.workload_root_cgroup = Some(path.into());
        self
    }

    /// Set the per-scheduler cgroup the scheduler process is
    /// placed in. The guest init mkdir's the path + enables
    /// `+cpuset +cpu` on every ancestor BEFORE starting the
    /// scheduler. Distinct from
    /// [`Self::workload_root_cgroup`] (workload placement).
    ///
    /// `path` must be an absolute cgroup path (leading `/`,
    /// not bare `/`); programmatic callers should pre-validate
    /// via [`crate::test_support::CgroupPath::new`].
    pub fn scheduler_cgroup_parent(mut self, path: impl Into<String>) -> Self {
        self.scheduler_cgroup_parent = Some(path.into());
        self
    }

    /// Override the guest scx watchdog timeout. Applied via
    /// `scx_sched.watchdog_timeout` (7.1+) or the static
    /// `scx_watchdog_timeout` symbol (pre-7.1); silently no-ops on
    /// kernels where neither path is available.
    #[allow(dead_code)]
    pub fn watchdog_timeout(mut self, timeout: Duration) -> Self {
        self.watchdog_timeout = Some(timeout);
        self
    }

    /// Override the freeze coordinator's per-rendezvous wait timeout
    /// (default: 30 s via `FREEZE_RENDEZVOUS_TIMEOUT` in
    /// `freeze_coord::state`). Lowering this drives the rendezvous's
    /// Degraded emit path — a `DegradedFailureDumpReport` carrying
    /// `REASON_DEGRADED_RENDEZVOUS_TIMEOUT` or
    /// `REASON_DEGRADED_KILL_DURING_RENDEZVOUS` — without waiting the
    /// full 30 s. Primarily a test-fixture knob; production callers
    /// should not override, as the 30 s default sits well above
    /// worst-case healthy rendezvous and any real timeout indicates
    /// a wedged vCPU.
    #[allow(dead_code)]
    pub fn rendezvous_timeout(mut self, timeout: Duration) -> Self {
        self.rendezvous_timeout = Some(timeout);
        self
    }

    /// Schedule a host-side write into a named BPF map after the
    /// scheduler is loaded. `map_name_suffix` is matched against
    /// `bpf_map.name` (kernel truncates to 15 chars); `field` names a
    /// global in the map's section (e.g. `"crash"`, `"stall"`), resolved
    /// to a byte offset against the map's program BTF at write time so
    /// the offset tracks the compiler's `.bss` layout instead of a
    /// fragile hardcoded constant; `value` is a `u32` written
    /// little-endian (via `to_ne_bytes`, which is LE on the x86_64 /
    /// aarch64 hosts ktstr targets — the guest reads the `.bss` global as
    /// a plain LE int).
    ///
    /// Repeated calls queue additional writes; all queued writes run
    /// sequentially on the same `BpfMapAccessor` after the scheduler
    /// attaches, with a single guest-side unblock once every write
    /// completes. Order of calls is preserved.
    pub fn bpf_map_write(mut self, map_name_suffix: &str, field: &str, value: u32) -> Self {
        self.bpf_map_writes.push(BpfMapWriteParams {
            map_name_suffix: map_name_suffix.to_string(),
            field: field.to_string(),
            value,
        });
        self
    }

    /// Register a named scheduler BPF-map field to read observer-effect-free
    /// from the free-running monitor into a run-level metric. Mirrors
    /// [`Self::bpf_map_write`]; the monitor resolves + reads each target
    /// lazily after the scheduler attaches.
    pub fn watch_bpf_map(
        mut self,
        map_name_suffix: &str,
        field: &str,
        agg: crate::test_support::BpfMapAgg,
        label: &str,
    ) -> Self {
        // A target's run-level metric key is `<scheduler-obj>_<label>`; the obj
        // prefix is uniform per run, so two targets with the same label resolve
        // to one key and collide in the fold — silently averaging (same agg) or
        // flipping the fold class (gauge vs counter). Reject duplicate labels
        // loudly at build time rather than emit a wrong metric.
        assert!(
            !self.watch_bpf_maps.iter().any(|p| p.label == label),
            "duplicate watch_bpf_maps label {label:?}: two targets resolve to the \
             same run-level metric key — declare distinct labels",
        );
        self.watch_bpf_maps.push(WatchBpfMapParams {
            map_name_suffix: map_name_suffix.to_string(),
            field: field.to_string(),
            agg,
            label: label.to_string(),
        });
        self
    }

    /// Enable performance mode: vCPU pinning to host LLCs,
    /// hugepage-backed guest memory, NUMA mbind, and RT scheduling
    /// on both architectures. On x86_64, additionally:
    /// KVM_HINTS_REALTIME CPUID hint (disables PV spinlocks, PV TLB
    /// flush, PV sched_yield; enables haltpoll cpuidle), PAUSE + HLT
    /// VM exit disabling via KVM_CAP_X86_DISABLE_EXITS (HLT falls
    /// back to PAUSE-only when mitigate_smt_rsb is active), and
    /// KVM_CAP_HALT_POLL skipped (guest haltpoll cpuidle disables
    /// host halt polling via MSR_KVM_POLL_CONTROL). On aarch64, KVM
    /// exit suppression and CPUID hints are not available. Validated
    /// at build time -- a host with too few CPUs / LLC groups for the
    /// requested perf topology returns `PerfModeUnavailable` (a
    /// host-insufficiency: a visible skip by default, promoted to a hard
    /// fail under `KTSTR_NO_SKIP_MODE`); busy LLC slots return
    /// `ResourceContention` (transient — a retryable failure nextest
    /// re-runs); insufficient
    /// hugepages is a warning.
    #[allow(dead_code)]
    pub fn performance_mode(mut self, enabled: bool) -> Self {
        self.performance_mode = enabled;
        self
    }

    /// Expose the virtio-PCI transport in the guest: a PCI host bridge with
    /// ECAM + legacy CAM config access and the PCI ACPI tables (MCFG + a
    /// `_SB.PCI0` host bridge in the DSDT). Off by default — the guest boots
    /// with `pci=off` and only the virtio-MMIO devices (console/block).
    #[allow(dead_code)]
    pub fn pci(mut self, enabled: bool) -> Self {
        self.pci_enabled = enabled;
        self
    }

    /// Skip flock topology reservation and force `performance_mode=false`
    /// (disables pinning, RT scheduling, hugepages, NUMA mbind, KVM exit
    /// suppression). For shared runners or unprivileged containers.
    pub fn no_perf_mode(mut self, enabled: bool) -> Self {
        self.no_perf_mode = enabled;
        self
    }

    /// Shell commands run inside the guest before the scenario to
    /// switch on a kernel-builtin scheduler (mirrors
    /// `SchedulerSpec::KernelBuiltin::enable`).
    pub fn sched_enable_cmds(mut self, cmds: &[&str]) -> Self {
        self.sched_enable_cmds = cmds.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Shell commands run inside the guest after the scenario to
    /// revert a kernel-builtin scheduler change (mirrors
    /// `SchedulerSpec::KernelBuiltin::disable`).
    pub fn sched_disable_cmds(mut self, cmds: &[&str]) -> Self {
        self.sched_disable_cmds = cmds.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Add files to include in the guest initramfs.
    /// Each entry is `(archive_path, host_path)`.
    pub fn include_files(mut self, files: Vec<(String, PathBuf)>) -> Self {
        self.include_files = files;
        self
    }

    /// Kernel modules to load in the guest before it touches any virtio
    /// device, in load order. Each path is a raw (already-decompressed)
    /// `.ko`; the guest's `rust_init` loads them via `finit_module(2)`
    /// right after mounting devtmpfs. Required for prebuilt distro
    /// kernels that ship virtio (blk/console/net) as modules; leave
    /// empty for ktstr-built kernels (virtio =y). The caller owns
    /// dependency ordering — modules are loaded exactly as given.
    #[allow(dead_code)]
    pub fn kernel_modules(mut self, modules: Vec<PathBuf>) -> Self {
        self.kernel_modules = modules;
        self
    }

    /// Initrd compression format the guest kernel can unpack
    /// (`CONFIG_RD_*`). Defaults to LZ4 — correct for ktstr-built
    /// kernels and raw images. For a prebuilt distro kernel pass
    /// [`crate::cache::initrd_compression_for_image`]'s answer, derived
    /// from the config cached beside the image; a kernel booted with an
    /// initrd format it cannot decode panics before reaching userspace
    /// ("Initramfs unpacking failed").
    #[allow(dead_code)]
    pub fn initrd_compression(mut self, comp: initramfs::InitrdCompression) -> Self {
        self.initrd_compression = comp;
        self
    }

    /// Attach a disk to the VM. Each call replaces any previously
    /// attached disk; the framework reserves a single MMIO + IRQ
    /// pair, so today the VM exposes at most one virtio-blk device
    /// at `/dev/vda`.
    ///
    /// Per-test backing is allocated by
    /// `KtstrVm::init_virtio_blk`:
    /// - `Filesystem::Raw` (the default): a fresh sparse
    ///   `tempfile()` per test, the kernel reclaims storage on
    ///   device drop.
    /// - `Filesystem::Btrfs`: a host-cached, guest-formatted
    ///   template image produced by a one-shot template VM
    ///   (`super::disk_template::build_template_via_vm`) is
    ///   reflink-cloned via `FICLONE` for the per-test backing.
    ///   The host never execs mkfs against a real backing file;
    ///   the kernel inside the template VM is the on-disk-format
    ///   authority.
    ///
    /// # Visible cache + per-test fan-out
    ///
    /// For `Filesystem::Btrfs`, the cache is a real on-disk
    /// directory under the ktstr cache root (resolved via
    /// `KTSTR_CACHE_DIR` / `XDG_CACHE_HOME` / `$HOME/.cache`; see
    /// [`super::disk_template::cache_root`]) so operators can
    /// inspect what's been built, GC stale entries by hand, and warm
    /// the cache out-of-band by running a Btrfs test once. The cache
    /// is keyed by `(filesystem_tag, capacity_mib, mkfs_fingerprint)` and the
    /// directory layout is `<cache>/disk_templates/<key>/template.img`
    /// — see [`super::disk_template`] module docs for the full encoding.
    ///
    /// Per-test fan-out goes through
    /// [`super::disk_template::clone_to_per_test`], which uses the
    /// `FICLONE` ioctl to reflink-copy the cached template image
    /// into a tempfile for the test VM. `FICLONE` is `O(metadata)`
    /// and copy-on-write at the extent level: per-test fan-out is
    /// independent of disk capacity and per-test writes never
    /// modify the cached template. The cache directory MUST live
    /// on a btrfs or xfs filesystem;
    /// [`super::disk_template::verify_cache_dir_supports_reflink`]
    /// checks `statfs.f_type` up front and bails with an actionable
    /// diagnostic when the cache filesystem cannot reflink, so
    /// operators see the constraint at first use rather than
    /// debugging a cryptic ioctl errno.
    pub fn disk(mut self, disk: disk_config::DiskConfig) -> Self {
        self.disk = Some(disk);
        // On x86_64 the disk is a virtio-pci function (see `init_virtio_blk_pci`)
        // and needs the host bridge + ECAM; aarch64 routes it over virtio-MMIO
        // (no PCI yet), so leave pci_enabled untouched there. Mirrors `network()`.
        #[cfg(target_arch = "x86_64")]
        {
            self.pci_enabled = true;
        }
        self
    }

    /// Attach one virtio-net device with the given configuration. The
    /// v0 backend is in-VMM loopback: TX bytes are echoed back into
    /// the RX queue inside the VMM, generating real virtio TX kicks
    /// and real `vring_interrupt` → `NET_RX_SOFTIRQ` activity that
    /// scheduler-test scenarios can observe. There is no host
    /// networking — IP-layer self-traffic is intercepted by the
    /// guest kernel's `RTN_LOCAL` route onto `lo`, so AF_PACKET raw
    /// sockets bound by `ifindex` are the path that exercises the
    /// virtio device.
    ///
    /// APPENDS a NIC; call it once per device. On x86_64 each NIC is a
    /// virtio-pci function on its own slot/GSI, so the count is bounded by
    /// `kvm::MAX_VIRTIO_NICS` (enforced at [`Self::build`], which errors on
    /// overflow rather than silently dropping); aarch64 supports a single MMIO
    /// NIC (build errors on >1). On x86_64 attaching a NIC enables the PCI host
    /// bridge automatically — the transport is an implementation detail the
    /// test author does not select. aarch64 keeps the virtio-MMIO NIC transport
    /// (aarch64 PCI is a later increment), so PCI stays off there. Reached via
    /// the `#[ktstr_test(networks = [...])]` attribute
    /// (`test_support::runtime::build_vm_builder_base` calls this per entry
    /// NIC), or directly by raw-library callers.
    pub fn network(mut self, network: net_config::NetConfig) -> Self {
        self.networks.push(network);
        // On x86_64 a NIC is a virtio-pci function and needs the host
        // bridge + ECAM; aarch64 routes the NIC over virtio-MMIO (no PCI
        // yet), so leave pci_enabled untouched there.
        #[cfg(target_arch = "x86_64")]
        {
            self.pci_enabled = true;
        }
        self
    }

    /// Override `KtstrVm::init_virtio_blk`'s per-test
    /// backing-file allocation with `path`. Internal-only: this is
    /// the seam the disk-template-build VM driver
    /// (`super::disk_template::build_template_via_vm`) uses to
    /// point a template-build guest at a host-staged sparse image,
    /// run `mkfs.<fstype>` against it inside the guest, and recover
    /// the now-formatted bytes after VM exit.
    ///
    /// When set, `init_virtio_blk` opens `path` for read+write and
    /// hands the resulting [`std::fs::File`] to the device — neither
    /// the `Raw` tempfile branch nor the `Btrfs` ensure_template
    /// branch executes, so a template-build VM cannot recursively
    /// re-enter the disk-template cache it is itself populating.
    /// The first attached disk's
    /// [`super::disk_config::DiskConfig::capacity_bytes`] still
    /// drives the device's advertised capacity; the staging image
    /// must already be sized to match.
    ///
    /// Production test paths leave this `None`. Setting it from a
    /// per-test build silently disables the template cache and would
    /// surface as a wrong-content backing file — the `Raw`/`Btrfs`
    /// branches in `init_virtio_blk` exist exactly to satisfy
    /// per-test isolation.
    pub(crate) fn template_staging_image(mut self, path: PathBuf) -> Self {
        self.template_staging_image = Some(path);
        self
    }

    /// Host path to `ktstr-jemalloc-probe`. When set, the probe is
    /// packed into the guest initramfs as an extra binary under
    /// `bin/` and resolves by bare name on the guest `PATH`. Tests
    /// that target the jemalloc TLS probe from a guest-side
    /// `ctx.payload(&PROBE)` invocation must set this to the host
    /// path obtained via `env!("CARGO_BIN_EXE_ktstr-jemalloc-probe")`.
    ///
    /// The probe attaches to a separately-spawned
    /// `ktstr-jemalloc-alloc-worker` via `--pid <worker_pid>`; the
    /// worker ships with DWARF, which is what the probe resolves
    /// offsets against, so the init binary does NOT need to retain
    /// DWARF. An earlier
    /// design attempted to preserve DWARF on the init binary so the
    /// probe could resolve offsets against the running init; that
    /// inflated the initramfs past practical VM memory budgets (the
    /// unstripped test binary is ~1 GB) and was abandoned in favor
    /// of routing DWARF through the probe and worker binaries.
    pub fn jemalloc_probe_binary(mut self, path: impl Into<PathBuf>) -> Self {
        self.jemalloc_probe_binary = Some(path.into());
        self
    }

    /// Host path to `ktstr-jemalloc-alloc-worker`. When set, the
    /// worker is packed alongside the probe in the guest initramfs
    /// as `/bin/ktstr-jemalloc-alloc-worker`. Used by the
    /// cross-process closed-loop test — spawned as a background
    /// payload that allocates a known number of bytes on the
    /// huge-size path (the jemalloc code path that unconditionally
    /// updates `thread_allocated` regardless of tcache state), then
    /// probed externally. The worker is much smaller than the full
    /// ktstr test binary (a single `fn main` linked against
    /// tikv-jemallocator) so shipping it keeps the initramfs well
    /// inside VM memory budgets — the init-DWARF approach that
    /// inflated the archive past those budgets was abandoned in
    /// favor of per-binary DWARF on the probe and worker.
    pub fn jemalloc_alloc_worker_binary(mut self, path: impl Into<PathBuf>) -> Self {
        self.jemalloc_alloc_worker_binary = Some(path.into());
        self
    }

    /// Embed busybox bytes in the initramfs at `bin/busybox` for
    /// shell mode. `None` skips packing; `Some(bytes)` writes the
    /// provided bytes. The library does not own the bytes — most
    /// callers source them from
    /// [`crate::vmm::blobs::load_busybox_bytes`] which reads the
    /// `KTSTR_BUSYBOX_PATH` env var that `cargo-ktstr` sets at
    /// startup.
    #[allow(dead_code)]
    pub fn busybox(mut self, bytes: Option<Vec<u8>>) -> Self {
        self.busybox_bytes = bytes;
        self
    }

    /// Embed the wprof tracer binary at `bin/wprof` and record the
    /// invocation args on the guest cmdline.
    #[cfg(feature = "wprof")]
    pub fn wprof(mut self, config: Option<crate::vmm::wprof::WprofConfig>) -> Self {
        self.wprof = config;
        self
    }

    /// Stream the guest kernel console (COM1/dmesg) to stderr in
    /// real time. Also bumps `loglevel=7` for verbose kernel output.
    pub fn dmesg(mut self, enabled: bool) -> Self {
        self.dmesg = enabled;
        self
    }

    /// Run a single command inside the guest instead of an
    /// interactive shell; the VM exits when the command completes.
    /// Requires `busybox(true)` and is typically paired with
    /// `KtstrVm::new_shell`.
    #[allow(dead_code)]
    pub fn exec_cmd(mut self, cmd: impl Into<String>) -> Self {
        self.exec_cmd = Some(cmd.into());
        self
    }

    /// Wall-clock bound for a `--exec` payload before the VM is
    /// force-killed (a panic-less guest hang otherwise blocks the BSP
    /// run loop ~forever). Default 120s. Consulted only in `--exec`
    /// runs; interactive shell sessions are unbounded.
    #[allow(dead_code)]
    pub fn exec_timeout(mut self, t: Duration) -> Self {
        self.exec_timeout = t;
        self
    }

    /// Validate the builder configuration and materialise a [`super::KtstrVm`].
    ///
    /// Returns `Err` for missing required inputs (kernel, init binary),
    /// invalid topology, or host resources insufficient to satisfy
    /// `performance_mode` requirements (a too-small host surfaces as
    /// `PerfModeUnavailable` — a host-insufficiency: skip-class by default,
    /// promoted to a hard fail under `KTSTR_NO_SKIP_MODE`; busy LLC slots
    /// surface as `ResourceContention`, a transient retryable failure). An operator
    /// over-budget `--cpu-cap` surfaces as `CpuBudgetUnsatisfiable` (a hard
    /// error — a concrete number the host cannot satisfy); an author's per-test
    /// `cpu_budget` over the allowance instead surfaces as
    /// `TopologyInsufficient` (skip-class, like any topology requirement a
    /// bigger host would satisfy).
    pub fn build(mut self) -> Result<KtstrVm> {
        // Periodic capture's boundary computation requires
        // `workload_duration` to slice. Without it the
        // freeze coordinator's run-loop never even computes
        // boundaries, so a `num_snapshots > 0` value would
        // silently never fire. Reject at build() so
        // misconfiguration surfaces during VM construction
        // rather than as zero captures on a passing run.
        if self.num_snapshots > 0 && self.workload_duration.is_none() {
            anyhow::bail!(
                "KtstrVmBuilder: num_snapshots = {} requires \
                 workload_duration to be set (the periodic-capture \
                 path needs a duration to slice into the 10%-90% \
                 window). Call .workload_duration(d) or set \
                 num_snapshots = 0.",
                self.num_snapshots,
            );
        }
        let no_perf_mode = self.no_perf_mode;
        if no_perf_mode {
            self.performance_mode = false;
        }

        let RunPlans {
            pinning_plan,
            mbind_node_map,
            mut no_perf_plan,
            host_topo: cached_host_topo,
            no_perf_effective_cap,
        } = self.resolve_run_plans(no_perf_mode)?;

        // Move the no-perf plan's build-time `LOCK_SH` fds off the plan
        // into their own slot so `acquire_run_locks` (which sees `&self`)
        // can drop them once its run-time replan holds fresh locks. Held
        // on the plan they would outlive the replan — a cell that replans
        // to different LLCs would keep phantom holds on the abandoned
        // build-time LLCs until `KtstrVm` drop, undercutting the truthful
        // holder counts this reservation exists to establish. The plan
        // keeps its shape fields (`cpus` / `locked_llcs` / `mems`) for the
        // build-time setup consumers. `None` / non-no-perf plans yield an
        // empty slot.
        let no_perf_build_locks = no_perf_plan
            .as_mut()
            .map(|p| std::mem::take(&mut p.locks))
            .unwrap_or_default();

        let kernel = self.kernel.context("kernel path required")?;
        anyhow::ensure!(kernel.exists(), "kernel not found: {}", kernel.display());
        let t = &self.topology;
        anyhow::ensure!(t.llcs > 0, "llcs must be > 0");
        anyhow::ensure!(t.cores_per_llc > 0, "cores_per_llc must be > 0");
        anyhow::ensure!(t.threads_per_core > 0, "threads_per_core must be > 0");
        anyhow::ensure!(t.numa_nodes > 0, "numa_nodes must be > 0");
        // NIC count is bounded by the transport: x86_64 routes each NIC over a
        // virtio-pci function on its own INTx GSI, capped by the IOAPIC line
        // budget ([`kvm::MAX_VIRTIO_NICS`]); aarch64's single virtio-MMIO slot
        // takes one. Error loudly rather than silently dropping NICs past the
        // limit (the .network() setter appends unconditionally).
        #[cfg(target_arch = "x86_64")]
        anyhow::ensure!(
            self.networks.len() <= super::kvm::MAX_VIRTIO_NICS,
            "too many NICs: {} attached, max {} (virtio-pci INTx GSI budget)",
            self.networks.len(),
            super::kvm::MAX_VIRTIO_NICS,
        );
        #[cfg(not(target_arch = "x86_64"))]
        anyhow::ensure!(
            self.networks.len() <= 1,
            "aarch64 supports a single virtio-MMIO NIC ({} attached); \
             multi-NIC needs the PCI transport (x86_64)",
            self.networks.len(),
        );
        // `memory_mib == Some(0)` would forward a literal `-m 0` to the
        // VMM backend (KVM rejects it at ioctl time with an opaque
        // error). Catch it here with a clear message so the caller
        // learns they set 0 explicitly rather than seeing a generic
        // kvm failure later. `None` falls back to the default (256 MiB).
        if matches!(self.memory_mib, Some(0)) {
            anyhow::bail!(
                "memory_mib must be > 0 (a VM with zero memory cannot boot); \
                 omit `.memory_mib(...)` to use the builder default"
            );
        }
        if let Some(ref bin) = self.init_binary
            && !bin.starts_with("/proc/")
        {
            anyhow::ensure!(bin.exists(), "init binary not found: {}", bin.display());
        }
        if let Some(ref bin) = self.scheduler_binary {
            anyhow::ensure!(
                bin.exists(),
                "scheduler binary not found: {}",
                bin.display()
            );
        }

        // Build a lazy on-demand BPF cast-analysis handle for the
        // scheduler binary. NO file I/O and NO analyzer work runs
        // here — the handle just captures the scheduler binary
        // path and a `OnceLock` slot. The actual analyzer (file
        // read + raw ELF parse + BTF parse + register-state walk
        // over BPF instructions; no libbpf, no kernel interaction,
        // no CAP_BPF) defers until the failure-dump path first
        // calls
        // [`super::cast_analysis_load::LazyCastMap::get_full`]
        // (the sole accessor).
        // Schedulers whose tests pass never trigger analyzer
        // work — the dominant case for nextest's process-per-test
        // execution model where steady-state tests boot a VM,
        // run, and exit without ever touching the dump path.
        //
        // When `.get_full()` does fire, it consults the process-
        // wide content-hash cache via
        // [`super::cast_analysis_load::cached_cast_analysis_for_scheduler`].
        // Within a single process (auto-repro after a primary
        // failure, future in-process multi-test drivers), two VMs
        // resolving to the same scheduler binary content share
        // one analyzer run.
        let cast_map = std::sync::Arc::new(super::cast_analysis_load::LazyCastMap::new(
            self.scheduler_binary.clone(),
        ));

        // Pre-materialize the (name, args) tuple view so the VM's
        // `suffix_params()` helper can borrow it without leaking
        // `StagedScheduler` into the `pub SuffixParams` field
        // signature. Cheap clone (name is short, args are small);
        // the duplication is the price for keeping the public
        // initramfs-suffix surface free of crate-private types.
        let staged_sched_args_packed: Vec<(String, Vec<String>)> = self
            .staged_schedulers
            .iter()
            .map(|s| (s.name.clone(), s.sched_args.clone()))
            .collect();

        let vcpus = t.total_cpus();
        let effective_cpu_budget =
            resolve_effective_cpu_budget(&no_perf_plan, cached_host_topo.is_some(), vcpus);

        Ok(KtstrVm {
            kernel,
            init_binary: self.init_binary,
            scheduler_binary: self.scheduler_binary,
            staged_schedulers: self.staged_schedulers,
            staged_sched_args_packed,
            run_args: self.run_args,
            sched_args: self.sched_args,
            topology: self.topology,
            vcpus,
            effective_cpu_budget,
            memory_mib: self.memory_mib,
            memory_min_mib: self.memory_min_mib,
            cmdline_extra: self.cmdline_extra,
            timeout: self.timeout,
            monitor_thresholds: self.monitor_thresholds,
            watchdog_timeout: self.watchdog_timeout,
            rendezvous_timeout: self.rendezvous_timeout,
            bpf_map_writes: self.bpf_map_writes,
            watch_bpf_maps: self.watch_bpf_maps,
            performance_mode: self.performance_mode,
            // A NIC or disk is a virtio-pci function on x86_64, so any attached
            // network OR disk implies the PCI transport — enforce the invariant
            // at the sole KtstrVm construction point so no assembly path (or a
            // stray `.pci(false)`) can yield a non-empty `networks` or an
            // attached `disk` with
            // pci_enabled=false, which would emit `pci=off` and skip installing
            // the NICs / disk (the guest would see no eth0 / no `/dev/vda`).
            // aarch64 keeps both on virtio-MMIO.
            pci_enabled: self.pci_enabled
                || (cfg!(target_arch = "x86_64")
                    && (!self.networks.is_empty() || self.disk.is_some())),
            no_perf_mode,
            pinning_plan,
            mbind_node_map,
            no_perf_plan,
            no_perf_effective_cap,
            no_perf_build_locks: std::sync::Mutex::new(no_perf_build_locks),
            host_topo: cached_host_topo,
            sched_enable_cmds: self.sched_enable_cmds,
            sched_disable_cmds: self.sched_disable_cmds,
            include_files: self.include_files,
            kernel_modules: self.kernel_modules,
            initrd_compression: self.initrd_compression,
            disk: self.disk,
            networks: self.networks,
            busybox_bytes: self.busybox_bytes,
            #[cfg(feature = "wprof")]
            wprof: self.wprof,
            dmesg: self.dmesg,
            exec_cmd: self.exec_cmd,
            exec_timeout: self.exec_timeout,
            jemalloc_probe_binary: self.jemalloc_probe_binary,
            jemalloc_alloc_worker_binary: self.jemalloc_alloc_worker_binary,
            failure_dump_path: self.failure_dump_path,
            dual_snapshot: self.dual_snapshot,
            template_staging_image: self.template_staging_image,
            workload_duration: self.workload_duration,
            num_snapshots: self.num_snapshots,
            workload_root_cgroup: self.workload_root_cgroup,
            scheduler_cgroup_parent: self.scheduler_cgroup_parent,
            cast_map,
        })
    }

    /// Resolve the run-time CPU/memory placement plans: the no-perf
    /// CPU-budget LLC reservation, the perf-mode pinning plan + NUMA
    /// mbind map, or the deferred-default (neither) path, and cache the
    /// host topology for [`KtstrVm::run`]'s deferred-lock branch. Returns
    /// the three plan slots plus the cached topology bundled in
    /// [`RunPlans`]. `no_perf_mode` selects the first arm; otherwise
    /// `self.performance_mode` selects the perf-mode arm, else the
    /// deferred default.
    fn resolve_run_plans(&mut self, no_perf_mode: bool) -> Result<RunPlans> {
        // `host_topo` is cached on KtstrVm so `KtstrVm::run`'s
        // default-else branch (neither perf-mode nor no-perf-mode)
        // can call `compute_pinning` per LLC offset and take `LOCK_SH`
        // via `acquire_resource_locks` without re-reading sysfs.
        // The no-perf-mode and perf-mode branches reuse their
        // stored plans' `locked_llcs` / `llc_indices` directly
        // through `acquire_resource_locks` and do not need the
        // topology at run time.
        let mut cached_host_topo: Option<host_topology::HostTopology> = None;
        // No-perf run-time replan budget: the exact effective `CpuCap`
        // the build-time plan targeted, carried onto `KtstrVm` so
        // `acquire_run_locks` re-plans against the SAME budget (not one
        // recomputed from a possibly-shifted allowed set) after the
        // build-time holds have made peers' holder counts truthful.
        // `None` for perf-mode and the degraded-sysfs / bypass no-perf
        // branches, which never replan.
        let mut no_perf_effective_cap: Option<host_topology::CpuCap> = None;

        let (pinning_plan, mbind_node_map, no_perf_plan) = if no_perf_mode {
            // No-perf-mode VMs would otherwise have unrestricted vCPU
            // affinity — the host kernel places their threads on any
            // online CPU, including ones a perf-mode peer has flocked
            // and bound its RT-FIFO vCPUs to. Injecting that thread
            // competition destroys perf-mode's measurement contract.
            // The coordination mechanism is an LLC-level flock set
            // (same as `kernel_build_pipeline`) so perf-mode's required
            // `LOCK_EX` blocks on any of them and fails over cleanly.
            //
            // `--cpu-cap` (or `KTSTR_CPU_CAP`) is a CPU-count budget:
            // the planner walks whole LLCs in contention- / NUMA-aware
            // order, filtered to the calling process's allowed cpuset
            // (sched_getaffinity), and accumulates until N CPUs are
            // reserved. `acquire_llc_plan` returns the selected LLC
            // list + flat `cpus` (intersection with allowed) + RAII
            // flock fds. The `cpus` are threaded into `no_perf_plan`
            // so `run_vm` can `sched_setaffinity` every vCPU thread
            // onto that pool. `KtstrVm::run` re-acquires fresh
            // flocks just before vCPU spawn — `build()` does not
            // hold flocks across the post-build setup window so
            // concurrent peers see the LLCs free until the run
            // actually starts.
            //
            // When the cap is absent (`CpuCap::resolve(None) ==
            // Ok(None)`), `resolve_cpu_budget` below auto-sizes the
            // budget to the VM's own vCPU count via `no_perf_cpu_budget`
            // (NOT the 30%-of-allowed `default_cpu_budget`, which is the
            // kernel-build default) and passes it to the planner as an
            // explicit `Some(cap)`. The resulting plan reserves a subset
            // of host LLCs sized to the guest topology — a 2-vCPU VM asks
            // ~2 CPUs, not ~30% — so no-perf-mode VMs never over-reserve
            // and fight concurrent builds or perf-mode peers for the full
            // host, regardless of whether the user set the flag.
            //
            // `cached` returning `Err` (non-Linux, sysfs absent — the
            // process-wide cache replays the first sysfs probe's
            // failure on every call) still forces the no-cap branch;
            // `acquire_llc_plan` is skipped, no coordination is
            // possible, but the VM still runs. `KTSTR_BYPASS_LLC_LOCKS=1`
            // bypasses both paths.
            //
            // The CLI binaries reject `--cpu-cap` + bypass at parse
            // time (see `ktstr::cli::CPU_CAP_HELP` and the Shell/
            // kernel-build dispatch checks in src/bin/ktstr.rs and
            // src/bin/cargo-ktstr.rs), but library consumers building
            // a `KtstrVmBuilder` directly with both env vars set
            // would silently lose the cap under a bare `if bypass
            // { return None-plan }`. Mirror the CLI check here so
            // the enforcement contract holds for every entry point,
            // not just the ones that go through the binaries.
            let bypass = crate::bypass_llc_locks_active();
            let cpu_cap = host_topology::CpuCap::resolve(None)?;
            if bypass {
                if cpu_cap.is_some() {
                    anyhow::bail!(
                        "no-perf-mode: KTSTR_CPU_CAP conflicts with \
                         KTSTR_BYPASS_LLC_LOCKS=1; unset one of them. \
                         KTSTR_CPU_CAP is a resource contract; bypass \
                         disables the contract entirely."
                    );
                }
                (None, Vec::new(), None)
            } else if let Ok(host_topo) = host_topology::HostTopology::cached() {
                let test_topo = crate::topology::TestTopology::from_system()?;
                // Effective budget: an explicit --cpu-cap / KTSTR_CPU_CAP
                // wins; otherwise size the budget to the VM's own vCPU count
                // so a wide VM's boot-time parallel AP bringup is not
                // throttled by the 30% default mask (the "8 vs 200" boot
                // oversubscription). Computed here rather than folded into
                // `cpu_cap` so the bypass-conflict check above still keys on
                // the *explicit* cap only.
                let effective_cap = resolve_cpu_budget(
                    cpu_cap,
                    self.cpu_budget,
                    host_topology::host_allowed_cpus().len(),
                    self.topology.total_cpus() as usize,
                )?;
                // Oversubscription warning: when the resolved host-CPU
                // budget is below the guest vCPU count the host time-slices
                // the vCPU threads, confounding guest-scheduler measurement
                // (see host_topology::overcommit_warning). Computed HERE —
                // after effective_cap resolves — so the explicit
                // --cpu-cap arm (which short-circuits the match above and
                // never reaches the vcpus comparison) is covered too, not
                // just the auto-size arm. `explicit` keys severity:
                // cpu_budget / --cpu-cap is an opt-in; an auto-collapse to a
                // too-small process cpuset is the silent case.
                if let Some(cap) = effective_cap {
                    let allowed = host_topology::host_allowed_cpus().len();
                    let vcpus = self.topology.total_cpus() as usize;
                    let eff = cap.effective_count(allowed).unwrap_or(allowed);
                    let explicit = cpu_cap.is_some() || self.cpu_budget.is_some();
                    if let Some(msg) = host_topology::overcommit_warning(
                        eff,
                        vcpus,
                        explicit,
                        self.watchdog_timeout.map(|d| d.as_secs()),
                    ) {
                        // KTSTR_CARGO_TEST_MODE does not enforce the budget
                        // (acquire_llc_plan masks to the full allowed cpuset
                        // and ignores cpu_cap), so the would-be-overcommit
                        // warning is misleading there — the stamped budget
                        // shows no overcommit and the sidecar marker stays
                        // silent. Suppress the build-time warning to match.
                        if !crate::cargo_test_mode::cargo_test_mode_active() {
                            eprintln!("{msg}");
                        }
                    }
                }
                // Spread placement: concurrent no-perf VMs fan out
                // across least-held LLCs (pid-rotated tiebreak)
                // instead of Consolidate-stacking onto the same
                // LLC-0-upward prefix — the clustering that piled a
                // 30-cell verifier sweep's affinity masks onto the
                // low LLCs of a 16-LLC runner. Builds keep
                // Consolidate (see `PlacementPolicy`).
                // Build-time reservation stays non-blocking (TOCTOU-only
                // fast path): the queue-and-wait policy lives in the
                // run-time replan (`acquire_run_locks`' no-perf arm),
                // which re-plans against live holder counts and waits
                // there. Build-time CONTENTION (a perf `LOCK_EX` head or
                // its claim covering the pool) therefore degrades to a
                // PLAN-ONLY selection — same DISCOVER→PLAN, no locks —
                // instead of failing the build: the build plan only
                // shapes setup (budget size, masks) and the run replan
                // re-acquires honestly through the queue. Failing here
                // was a retry-storm source: nextest's second-scale
                // backoff can never outlive a queued perf head's claim.
                // The cost is holder-count truthfulness during THIS
                // cell's setup window (peers see no reservation), which
                // the run-time replan restores.
                let plan = match host_topology::acquire_llc_plan(
                    &host_topo,
                    &test_topo,
                    effective_cap,
                    host_topology::PlacementPolicy::spread_for_process(),
                    false,
                ) {
                    Ok(plan) => plan,
                    Err(e)
                        if e.downcast_ref::<host_topology::ResourceContention>()
                            .is_some() =>
                    {
                        tracing::debug!(
                            "no-perf build-time reservation contended ({e:#}); \
                             proceeding with a lock-free plan — the run-time \
                             replan queues for real locks"
                        );
                        host_topology::plan_llc_selection_only(
                            &host_topo,
                            &test_topo,
                            effective_cap,
                            host_topology::PlacementPolicy::spread_for_process(),
                        )?
                    }
                    Err(e) => return Err(e),
                };
                host_topology::warn_if_cross_node_spill(&plan, &host_topo);
                // Keep the plan's `LOCK_SH` flock fds ALIVE — do NOT
                // strip them. Those held fds are precisely what makes a
                // concurrent peer's DISCOVER of these LLCs see a truthful
                // holder count. The historical `drop(take(&mut
                // plan.locks))` here released every reservation before any
                // peer reached its own PLAN phase, so a concurrent no-perf
                // sweep observed zero holders and every cell Spread-planned
                // as if it were alone: the holder-count feedback was dead
                // and the cells re-clustered onto the same low LLCs the
                // Spread policy exists to avoid. `build()` moves these fds
                // off `plan.locks` into `KtstrVm::no_perf_build_locks`,
                // where they are held through the setup window and released
                // the moment `run()`'s `acquire_run_locks` re-plans against
                // the now-truthful counts and adopts the fresh plan's own
                // fds — acquire-before-release, so retained LLCs never
                // flicker free (see `KtstrVm::no_perf_effective_cap` for
                // the replan budget). `plan.cpus` / `locked_llcs` / `mems`
                // stay populated for the build-time setup paths regardless.
                no_perf_effective_cap = effective_cap;
                cached_host_topo = Some(host_topo);
                (None, Vec::new(), Some(plan))
            } else {
                if cpu_cap.is_some() {
                    anyhow::bail!(
                        "--cpu-cap set but host LLC topology unreadable from \
                         sysfs — cannot enforce the resource budget. Run on a \
                         host with /sys/devices/system/cpu populated, or drop \
                         --cpu-cap to run without enforcement."
                    );
                }
                tracing::warn!(
                    "no-perf-mode: could not read host LLC topology from sysfs; \
                     skipping CPU-budget LLC reservation. Concurrent perf-mode \
                     runs on this host will NOT be serialized against this VM"
                );
                (None, Vec::new(), None)
            }
        } else if self.performance_mode {
            let (mut plan, host_topo) = self.validate_performance_mode()?;
            let node_map = build_per_node_map(&plan, &host_topo, &self.topology);
            // Strip the flock fds — `run()` re-acquires via
            // `acquire_resource_locks` using `plan.llc_indices`.
            // The build-time setup paths read `assignments` /
            // `service_cpu` / `llc_indices`, which all stay
            // populated.
            drop(std::mem::take(&mut plan.locks));
            cached_host_topo = Some(host_topo);
            (Some(plan), node_map, None)
        } else {
            // Default: defer pinning to run() which tries each LLC
            // offset with LOCK_SH. Cache the host topology so run()
            // can compute plans; no plan or locks at build time.
            cached_host_topo = host_topology::HostTopology::cached().ok();
            (None, Vec::new(), None)
        };

        Ok(RunPlans {
            pinning_plan,
            mbind_node_map,
            no_perf_plan,
            host_topo: cached_host_topo,
            no_perf_effective_cap,
        })
    }

    /// Validate host resources for performance_mode and compute the
    /// pinning plan. Returns both the plan and the host topology (needed
    /// for NUMA node discovery). Returns `PerfModeUnavailable` when the
    /// host has too few CPUs / LLC groups for the requested perf topology
    /// (the explicit isolation guarantee cannot be honored — a permanent
    /// host-insufficiency the dispatch/macro treat as a SKIP by default,
    /// promoted to a hard FAIL under `KTSTR_NO_SKIP_MODE`; from the pre-check
    /// here and via the `compute_pinning` re-map in `acquire_slot_with_locks`),
    /// or `ResourceContention` when the host is
    /// big enough but all LLC slots are currently busy (transient →
    /// retryable failure, nextest re-runs). Warnings are printed for degraded conditions
    /// (hugepages, host load).
    fn validate_performance_mode(
        &mut self,
    ) -> Result<(host_topology::PinningPlan, host_topology::HostTopology)> {
        let host_topo = host_topology::HostTopology::cached()
            .context("performance_mode: read host topology")?;

        let t = &self.topology;
        let total_vcpus = t.total_cpus();

        // Validate LLC exclusivity: each virtual LLC should map to
        // its own physical LLC group. Sum actual per-group CPU counts
        // to handle asymmetric LLCs.
        let llcs_needed = t.llcs as usize;
        let reserved: usize = host_topo
            .llc_groups
            .iter()
            .take(llcs_needed)
            .map(|g| g.cpus.len())
            .sum();
        let total_reserved = reserved + 1; // +1 for service CPU
        if total_reserved > host_topo.total_cpus() {
            // The host has fewer CPUs than perf-mode must reserve: the
            // explicitly-requested isolation guarantee cannot be honored on
            // this host. PerfModeUnavailable — a host-insufficiency the
            // dispatch/macro treat as a SKIP by default (FAIL under
            // KTSTR_NO_SKIP_MODE); the operator provisions a bigger host,
            // narrows the topology, or drops --perf-mode.
            return Err(anyhow::Error::new(host_topology::PerfModeUnavailable {
                reason: format!(
                    "performance_mode: need {} CPUs ({} across {} LLCs + 1 service) \
                     but only {} host CPUs available\n  \
                     hint: pass --no-perf-mode or set KTSTR_NO_PERF_MODE=1 to run without CPU reservation",
                    total_reserved,
                    reserved,
                    llcs_needed,
                    host_topo.total_cpus(),
                ),
            }));
        }

        let plan = acquire_slot_with_locks(&host_topo, t)?;

        // WARN: hugepages (only when memory is known upfront).
        if let Some(mib) = self.memory_mib {
            let free = host_topology::hugepages_free();
            let needed = host_topology::hugepages_needed(mib);
            if free == 0 {
                eprintln!(
                    "performance_mode: WARNING: no 2MB hugepages available, \
                     guest memory will use regular pages",
                );
            } else if free < needed {
                eprintln!(
                    "performance_mode: WARNING: need {} 2MB hugepages, \
                     only {} free — falling back to regular pages",
                    needed, free,
                );
            }
        }

        // WARN: host load.
        if let Some((running, total)) = host_topology::host_load_estimate() {
            let threshold = (total_vcpus as f64 * 0.5) as usize;
            if running > threshold {
                eprintln!(
                    "performance_mode: WARNING: {} processes running on {} CPUs \
                     (threshold {} for {} vCPUs) — results may be noisy",
                    running, total, threshold, total_vcpus,
                );
            }
        }

        Ok((plan, host_topo))
    }
}

/// Build per-guest-NUMA-node host NUMA node mapping from a pinning plan.
fn build_per_node_map(
    plan: &host_topology::PinningPlan,
    host_topo: &host_topology::HostTopology,
    topo: &crate::vmm::topology::Topology,
) -> Vec<Vec<usize>> {
    let n = topo.numa_nodes as usize;
    let mut map: Vec<std::collections::BTreeSet<usize>> =
        vec![std::collections::BTreeSet::new(); n];
    let cpus_per_llc = topo.cores_per_llc * topo.threads_per_core;
    for &(vcpu_id, host_cpu) in &plan.assignments {
        let llc_id = vcpu_id / cpus_per_llc;
        let guest_node = topo.numa_node_of(llc_id) as usize;
        let host_node = host_topo.cpu_to_node.get(&host_cpu).copied().unwrap_or(0);
        if guest_node < n {
            map[guest_node].insert(host_node);
        }
    }
    map.into_iter().map(|s| s.into_iter().collect()).collect()
}

// Stamp the run's guest vCPU count + the EFFECTIVE host-CPU budget
// for the sidecar (budget Dimension + overcommit marker) — the
// number of distinct host CPUs the vCPU threads actually run on.
// no-perf reserves a CPU budget (the no_perf_plan's cpus) and masks
// every vCPU thread onto it (the overcommit-relevant path: budget
// may be < vcpus). Under KTSTR_CARGO_TEST_MODE the plan reserves
// nothing and its cpus == the full allowed cpuset (a no-op mask), so
// the stamp records the unrestricted set the vCPUs floated across —
// still the true CPU count the threads ran on.
// perf-mode AND the deferred default both attempt a 1:1 pinning
// plan at run time — perf-mode via `validate_performance_mode`, the
// default via `run()`'s LOCK_SH offset search — hard-pinning each
// vCPU thread to one distinct host CPU (`compute_pinning` emits
// exactly `vcpus` 1:1 assignments). Both cache the host topology, so
// `cached_host_topo.is_some()` predicts a 1:1 pin and the build-time
// budget is the vCPU count. Two run-time outcomes diverge from that
// estimate: perf-mode aborts with ResourceContention if its LOCK_EX
// is unavailable (no sidecar written), and the default path
// OVERCOMMITS when no offset can map the topology (host too small) —
// `run()` then overrides `VmResult.cpu_budget` with the actual
// masked host-CPU count (`RunLocks::default_cpu_mask` length), so a
// too-small host stamps the real overcommit, not this `vcpus`
// estimate. Only when no affinity is applied (no-perf bypass, sysfs
// unreadable, or the deferred default with no cached host topology)
// do the vCPU threads fall to the allowed-cpuset size below. The
// earlier `no_perf_plan` arm wins first, so the `cached_host_topo`
// arm is only reached with no no-perf plan (perf-mode / deferred
// default), never the no-perf masked path.
fn resolve_effective_cpu_budget(
    no_perf_plan: &Option<host_topology::LlcPlan>,
    has_cached_host_topo: bool,
    vcpus: u32,
) -> u32 {
    if let Some(p) = no_perf_plan {
        p.cpus.len() as u32
    } else if has_cached_host_topo {
        vcpus
    } else {
        // No affinity applied (bypass / sysfs-unreadable): the threads
        // float across the allowed cpuset. host_allowed_cpus() returns
        // empty only when BOTH sched_getaffinity AND /proc/self/status
        // fail (a host that can barely run); clamp to >= 1 so a genuinely
        // booted run never stamps 0, which sidecar_to_row maps to None
        // and explain renders as the "skip; VM not booted" sentinel —
        // misclassifying a real run as a skip.
        (host_topology::host_allowed_cpus().len() as u32).max(1)
    }
}

/// Resolve the effective per-VM CPU cap from an explicit cap, a per-test
/// `cpu_budget`, and the host allowance.
///
/// - An explicit `--cpu-cap`/`KTSTR_CPU_CAP` (`cpu_cap = Some`) wins verbatim.
/// - Otherwise a per-test `cpu_budget` (`#[ktstr_test]`) is honored as a
///   topology REQUIREMENT: a budget exceeding `allowed` host CPUs yields
///   [`host_topology::TopologyInsufficient`] (a host-dependent SKIP — the
///   scenario needs a bigger host; a `#[ktstr_test]` attribute is an author
///   capability request, not the operator's concrete `--cpu-cap` number, so a
///   too-small host skips it rather than reddening CI, and `--no-skip-mode`
///   promotes it to a fail). At or below the allowance it stands (floored at 1)
///   so a test can force overcommit (`cpu_budget < vcpus`). This is the
///   provenance split: the operator knob (`--cpu-cap`, validated in
///   [`host_topology::CpuCap::effective_count`]) FAILS when unsatisfiable; the
///   author attribute SKIPS.
/// - Absent both, the budget auto-sizes to the VM's vCPU count via
///   [`host_topology::no_perf_cpu_budget`] so a wide VM's boot-time parallel AP
///   bringup is not throttled by the 30% default mask.
///
/// Extracted from `build()` as a pure function so the budget-resolution policy
/// is unit-testable without booting a VM.
fn resolve_cpu_budget(
    cpu_cap: Option<host_topology::CpuCap>,
    per_test_budget: Option<u32>,
    allowed: usize,
    vcpus: usize,
) -> Result<Option<host_topology::CpuCap>> {
    match cpu_cap {
        Some(c) => Ok(Some(c)),
        None => {
            let budget = match per_test_budget {
                Some(n) => {
                    let n = n as usize;
                    if n > allowed {
                        // A per-test `cpu_budget` is an AUTHOR-declared topology
                        // REQUIREMENT (the scenario needs `n` CPUs to be
                        // meaningful), not an operator's concrete `--cpu-cap`
                        // assertion. A host with fewer allowed CPUs cannot run
                        // it, but a bigger one can — so this is a host-dependent
                        // SKIP (`TopologyInsufficient`), like any other topology
                        // requirement, NOT the hard `CpuBudgetUnsatisfiable` a
                        // too-large `--cpu-cap` raises. Provenance splits the
                        // verdict: an author attribute skips; an operator knob
                        // fails.
                        return Err(anyhow::Error::new(host_topology::TopologyInsufficient {
                            reason: format!(
                                "cpu_budget = {n} exceeds the {allowed} CPUs this \
                                 process is allowed on (from sched_getaffinity / \
                                 Cpus_allowed_list): this test needs a host with at \
                                 least {n} allowed CPUs. Provision a larger host, \
                                 lower the test's cpu_budget, or release the \
                                 cgroup/taskset constraint restricting this process."
                            ),
                        }));
                    }
                    n.max(1)
                }
                None => host_topology::no_perf_cpu_budget(allowed, vcpus),
            };
            Ok(Some(host_topology::CpuCap::new(budget)?))
        }
    }
}

/// Try each LLC slot, compute a pinning plan, and acquire resource
/// locks (non-blocking). Single pass through all available slots.
/// Returns `PerfModeUnavailable` when `compute_pinning` reports the host is
/// too small for the perf topology (the isolation guarantee cannot be
/// honored — a permanent host-insufficiency: a SKIP by default, a hard FAIL
/// under `KTSTR_NO_SKIP_MODE`).
///
/// When the host FITS but every slot is currently busy, this is NOT a
/// failure: the build-time acquire is a free-offset PROBE — `build()`
/// strips the returned fds immediately and `run()` re-acquires the
/// plan's `llc_indices` through the acquisition queue (fast path,
/// then head accumulation with progress-based patience). So on
/// all-busy the probe returns the FIRST mappable candidate with an
/// empty lock set and lets the run path queue for it. Bailing here —
/// the old behaviour — made perf cells fail instantly (~0.2 s,
/// pre-boot) whenever the suite's default cells covered the LLCs with
/// `LOCK_SH`, and nextest's second-scale retry backoff could never
/// outlive that coverage: a retry-exhaustion storm the queue exists
/// to prevent.
fn acquire_slot_with_locks(
    host_topo: &host_topology::HostTopology,
    topo: &topology::Topology,
) -> Result<host_topology::PinningPlan> {
    // Grain decision from the offset-0 mapping, which doubles as the
    // host-fits gate: a `TopologyInsufficient` here is the host-too-small
    // case → `PerfModeUnavailable` (the isolation guarantee cannot be
    // honored; a SKIP by default, hard FAIL under `KTSTR_NO_SKIP_MODE`).
    // The grain (whole-LLC-Exclusive vs per-CPU) is a function of host-LLC
    // size vs the cell's per-LLC vCPU footprint — constant across LLC
    // offsets on a homogeneous host — so one probe settles it for the
    // whole scan. The two branches below re-probe as needed; this probe's
    // only lasting output is the mode.
    let base = match host_topo.compute_pinning(topo, true, 0) {
        Ok(c) => c,
        Err(e)
            if e.downcast_ref::<host_topology::TopologyInsufficient>()
                .is_some() =>
        {
            return Err(anyhow::Error::new(host_topology::PerfModeUnavailable {
                reason: format!("performance_mode: {e:#}"),
            }));
        }
        Err(e) => return Err(e).context("performance_mode: topology mapping"),
    };

    match host_topology::perf_llc_lock_mode(host_topo, &base) {
        // Host LLC dwarfs the cell (e.g. the Graviton's 96-CPU L3):
        // reserve at per-CPU grain so disjoint perf cells coexist.
        host_topology::LlcLockMode::Shared => acquire_grain_slot_with_locks(host_topo, topo),
        // Validated small-LLC hosts (dev box, x86 CI, native arm): the
        // unchanged whole-LLC-Exclusive reservation the dilation campaign
        // measured.
        host_topology::LlcLockMode::Exclusive => acquire_exclusive_slot_with_locks(host_topo, topo),
    }
}

/// Whole-LLC-Exclusive perf reservation — the historical path, unchanged.
/// Scans LLC offsets, taking `LOCK_EX` on the whole LLC set; on all-busy
/// hands the first mappable candidate (no locks) to the run path, which
/// queues for its `llc_indices`.
fn acquire_exclusive_slot_with_locks(
    host_topo: &host_topology::HostTopology,
    topo: &topology::Topology,
) -> Result<host_topology::PinningPlan> {
    let num_llcs = host_topo.llc_groups.len();
    let llcs_needed = topo.llcs as usize;
    let max_slots = num_llcs.checked_div(llcs_needed).unwrap_or(num_llcs).max(1);
    let llc_mode = host_topology::LlcLockMode::Exclusive;

    let mut first_mappable: Option<host_topology::PinningPlan> = None;
    for slot in 0..max_slots {
        let offset = slot * llcs_needed;

        let candidate = match host_topo.compute_pinning(topo, true, offset) {
            Ok(c) => c,
            // compute_pinning returns TopologyInsufficient when the host has
            // too few CPUs/LLCs for the requested perf topology. For a
            // perf-mode test that means the isolation guarantee cannot be
            // honored here -> PerfModeUnavailable, a host-insufficiency the
            // dispatch/macro SKIP by default (FAIL under KTSTR_NO_SKIP_MODE);
            // distinct from the transient all-slots-busy ResourceContention
            // below.
            Err(e)
                if e.downcast_ref::<host_topology::TopologyInsufficient>()
                    .is_some() =>
            {
                return Err(anyhow::Error::new(host_topology::PerfModeUnavailable {
                    reason: format!("performance_mode: {e:#}"),
                }));
            }
            Err(e) => return Err(e).context("performance_mode: topology mapping"),
        };

        match host_topology::acquire_resource_locks(&candidate, &candidate.llc_indices, llc_mode)? {
            host_topology::LockOutcome::Acquired { locks, .. } => {
                let mut plan = candidate;
                plan.locks = locks;
                eprintln!(
                    "performance_mode: reserved LLC slot {} (offset {}, max {})",
                    slot, offset, max_slots,
                );
                return Ok(plan);
            }
            host_topology::LockOutcome::Unavailable(_) => {
                if first_mappable.is_none() {
                    first_mappable = Some(candidate);
                }
                continue;
            }
        }
    }

    // Host fits, every slot busy: hand the first mappable candidate
    // (no locks) to the run path, which queues for its llc_indices.
    let plan = first_mappable.expect(
        "all-busy implies at least one candidate mapped — the \
         no-candidate case returned PerfModeUnavailable above",
    );
    eprintln!(
        "performance_mode: all {max_slots} LLC slots busy at build time; \
         the run will queue for offset {} via the lock-dir acquisition \
         queue",
        plan.llc_indices.first().copied().unwrap_or(0),
    );
    Ok(plan)
}

/// Per-CPU-GRAIN perf reservation — the huge-LLC path. Enumerates
/// disjoint `(vcpus_per_llc + 1)`-CPU grain BLOCKS
/// ([`host_topology::HostTopology::compute_pinning_grain`]) and takes,
/// per block, a SHARED (`LOCK_SH`) LLC lock plus per-CPU `LOCK_EX` over
/// exactly the block's cores + service CPU (the [`LlcLockMode::Shared`]
/// composition). Because distinct blocks are disjoint by construction,
/// two perf cells on different blocks acquire non-overlapping per-CPU
/// sets and COEXIST on the shared LLC — the parallelism that whole-LLC
/// `LOCK_EX` denies on a monolithic L3.
///
/// Block probing is pid-ROTATED so simultaneously-launched cells start
/// on DIFFERENT blocks instead of all bouncing on block 0's per-CPU
/// locks (mirrors the default run path's intra-slot rotation). On
/// all-busy it hands the first mappable block (no locks) to the run
/// path, which queues for that block's exact CPUs via the acquisition
/// queue — identical contract to the exclusive path, at CPU granularity.
fn acquire_grain_slot_with_locks(
    host_topo: &host_topology::HostTopology,
    topo: &topology::Topology,
) -> Result<host_topology::PinningPlan> {
    let llc_mode = host_topology::LlcLockMode::Shared;

    // Enumerate every grain block that fits (block windows shrink the
    // available CPUs by one `(V+1)` stride each — `compute_pinning_grain`
    // returns `TopologyInsufficient` past the last, our end-of-blocks
    // signal). Block 0 mapped in the caller's grain probe, so non-empty.
    let mut blocks: Vec<host_topology::PinningPlan> = Vec::new();
    for block in 0.. {
        match host_topo.compute_pinning_grain(topo, block) {
            Ok(c) => blocks.push(c),
            Err(_) => break,
        }
    }

    // pid-rotate the probe order so concurrent cells fan across blocks.
    let start = host_topology::pid_window_offset(std::process::id(), blocks.len().max(1));
    let order: Vec<usize> = (0..blocks.len())
        .map(|i| (start + i) % blocks.len())
        .collect();

    let mut first_mappable: Option<host_topology::PinningPlan> = None;
    for &bi in &order {
        let candidate = blocks[bi].clone_unlocked();
        match host_topology::acquire_resource_locks(&candidate, &candidate.llc_indices, llc_mode)? {
            host_topology::LockOutcome::Acquired { locks, .. } => {
                let mut plan = candidate;
                plan.locks = locks;
                eprintln!(
                    "performance_mode: reserved per-CPU-grain block {} of {} \
                     on host LLC(s) {:?} (LLC dwarfs the cell; cells coexist \
                     via shared LLC + exclusive per-CPU locks)",
                    bi,
                    blocks.len(),
                    plan.llc_indices,
                );
                return Ok(plan);
            }
            host_topology::LockOutcome::Unavailable(_) => {
                if first_mappable.is_none() {
                    first_mappable = Some(candidate);
                }
            }
        }
    }

    let plan = first_mappable
        .expect("grain enumeration yields at least block 0 — the caller's probe mapped it");
    let cpus: Vec<usize> = plan.assignments.iter().map(|&(_, c)| c).collect();
    eprintln!(
        "performance_mode: all {} per-CPU-grain blocks busy at build time; \
         the run will queue for block CPUs {:?} via the acquisition queue",
        blocks.len(),
        cpus,
    );
    Ok(plan)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_default() {
        let b = KtstrVmBuilder::default();
        assert_eq!(b.memory_mib, Some(256));
        assert_eq!(b.topology.total_cpus(), 1);
    }

    /// resolve_cpu_budget: an explicit cap wins verbatim and ignores the
    /// per-test cpu_budget — even a per-test budget that would otherwise be a
    /// hard error (999 > allowed 10) is bypassed by the explicit cap.
    #[test]
    fn resolve_cpu_budget_explicit_cap_wins() {
        let cap = host_topology::CpuCap::new(5).unwrap();
        let resolved = resolve_cpu_budget(Some(cap), Some(999), 10, 8)
            .unwrap()
            .expect("explicit cap resolves to Some");
        assert_eq!(resolved.effective_count(10).unwrap(), 5);
    }

    /// resolve_cpu_budget: a per-test cpu_budget at or below the host allowance
    /// stands (floored at 1) so a test can force overcommit (budget < vcpus).
    #[test]
    fn resolve_cpu_budget_per_test_budget_within_allowance_stands() {
        let resolved = resolve_cpu_budget(None, Some(4), 10, 8)
            .unwrap()
            .expect("budget resolves to Some");
        assert_eq!(resolved.effective_count(10).unwrap(), 4);
    }

    /// resolve_cpu_budget over-allowance gate: a per-test cpu_budget exceeding
    /// the host allowance is a TYPED TopologyInsufficient SKIP (an author
    /// attribute is a topology requirement a bigger host would satisfy), NOT a
    /// silent clamp and NOT the hard CpuBudgetUnsatisfiable an operator
    /// `--cpu-cap` raises (that provenance split is the point). The complement —
    /// an operator `--cpu-cap` over the allowance still hard-failing — is pinned
    /// by the CpuCap::effective_count tests, since that path is validated there,
    /// before resolve_cpu_budget sees the cap.
    #[test]
    fn resolve_cpu_budget_per_test_budget_over_allowance_skips() {
        let err = resolve_cpu_budget(None, Some(100), 10, 8)
            .expect_err("budget 100 > allowed 10 must yield a skip error");
        assert!(
            err.downcast_ref::<host_topology::TopologyInsufficient>()
                .is_some(),
            "must be a typed TopologyInsufficient (skip), got: {err:#}",
        );
        assert!(
            err.downcast_ref::<host_topology::CpuBudgetUnsatisfiable>()
                .is_none(),
            "an author cpu_budget must NOT raise the operator-only \
             CpuBudgetUnsatisfiable hard error: {err:#}",
        );
    }

    /// resolve_cpu_budget auto-size default: absent both an explicit cap and a
    /// per-test budget, the budget auto-sizes via no_perf_cpu_budget (so a wide
    /// VM is not throttled by the 30% default mask). Pins the DELEGATION to
    /// no_perf_cpu_budget, not a re-derived constant.
    #[test]
    fn resolve_cpu_budget_auto_sizes_to_no_perf_budget() {
        let allowed = 100;
        let vcpus = 50;
        let resolved = resolve_cpu_budget(None, None, allowed, vcpus)
            .unwrap()
            .expect("auto-size resolves to Some");
        assert_eq!(
            resolved.effective_count(allowed).unwrap(),
            host_topology::no_perf_cpu_budget(allowed, vcpus),
            "absent-both must delegate to no_perf_cpu_budget",
        );
    }

    /// resolve_cpu_budget regression: the exact CI-failure shape — a
    /// 2-vCPU interactive shell (no explicit cap, no per-test budget) on a
    /// synthetic ~192-CPU host must resolve to 3 CPUs (vCPUs + 1 service), NEVER
    /// the 30% (58) that the old `default_cpu_budget` floor produced and
    /// that exhausted `acquire_llc_plan` under peer `LOCK_EX` contention.
    #[test]
    fn resolve_cpu_budget_small_vm_on_large_host_asks_vcpus_not_30_percent() {
        let allowed = 192;
        let vcpus = 2;
        let resolved = resolve_cpu_budget(None, None, allowed, vcpus)
            .unwrap()
            .expect("auto-size resolves to Some");
        assert_eq!(
            resolved.effective_count(allowed).unwrap(),
            3,
            "2-vCPU shell on a 192-CPU host asks 3 CPUs (vCPUs + 1 service), not the 30% (58) floor",
        );
    }

    /// resolve_cpu_budget: an explicit `--cpu-cap` is honored as the budget
    /// verbatim — a CEILING the operator set, not overridden by the
    /// vCPU-sized auto default. A cap ABOVE the vCPU count still stands (the
    /// operator asked for headroom); a cap the host cannot satisfy is caught
    /// later at `effective_count`, not here.
    #[test]
    fn resolve_cpu_budget_explicit_cap_is_the_ceiling() {
        let cap = host_topology::CpuCap::new(8).unwrap();
        let resolved = resolve_cpu_budget(Some(cap), None, 192, 2)
            .unwrap()
            .expect("explicit cap resolves to Some");
        assert_eq!(
            resolved.effective_count(192).unwrap(),
            8,
            "explicit --cpu-cap wins verbatim over the vCPU-sized default",
        );
    }

    /// resolve_cpu_budget: a WIDE no-perf topology still gets its full
    /// vCPU count plus the service CPU (test validity — budget >= vcpus so
    /// the guest scheduler isn't confounded by host oversubscription;
    /// the +1 keeps sensing threads off the vCPUs' CPUs).
    #[test]
    fn resolve_cpu_budget_wide_topo_gets_full_vcpu_count() {
        let allowed = 192;
        let vcpus = 50;
        let resolved = resolve_cpu_budget(None, None, allowed, vcpus)
            .unwrap()
            .expect("auto-size resolves to Some");
        assert_eq!(
            resolved.effective_count(allowed).unwrap(),
            51,
            "a 50-vCPU no-perf VM gets its 50 host CPUs + 1 service CPU",
        );
    }

    /// acquire_slot_with_locks all-busy fallback: when the host FITS the
    /// perf topology but every slot's `LOCK_EX` is currently held, the
    /// build-time probe must return the first mappable candidate with an
    /// EMPTY lock set — the run path queues for it — instead of the old
    /// `ResourceContention` bail, which made perf cells fail pre-boot in
    /// ~0.2 s whenever suite `LOCK_SH` coverage never left a free window
    /// (a retry-exhaustion storm nextest backoff cannot outlive).
    #[test]
    fn acquire_slot_with_locks_all_busy_returns_unlocked_candidate() {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let llc_prefix = format!("{}/llc-", tmp.path().display());
        let cpu_prefix = format!("{}/cpu-", tmp.path().display());
        host_topology::LLC_LOCK_PREFIX_OVERRIDE.with(|p| *p.borrow_mut() = Some(llc_prefix));
        host_topology::CPU_LOCK_PREFIX_OVERRIDE.with(|p| *p.borrow_mut() = Some(cpu_prefix));

        // One-LLC host, 1/1/1/1 perf topology (+ service CPU) fits.
        let host = host_topology::HostTopology::new_for_tests(&[(vec![0, 1], 0)]);
        let topo = topology::Topology::new(1, 1, 1, 1);
        // Peer holds the only LLC slot EX.
        let peer = crate::flock::try_flock(
            host_topology::llc_lock_path(0),
            crate::flock::FlockMode::Exclusive,
        )
        .expect("open")
        .expect("peer EX on fresh pool");

        let plan = acquire_slot_with_locks(&host, &topo)
            .expect("all-busy must yield an unlocked candidate, not contention");
        assert!(
            plan.locks.is_empty(),
            "all-busy fallback carries no locks — the run path queues",
        );
        assert_eq!(plan.llc_indices, vec![0], "first mappable candidate");
        drop(peer);
        host_topology::LLC_LOCK_PREFIX_OVERRIDE.with(|p| *p.borrow_mut() = None);
        host_topology::CPU_LOCK_PREFIX_OVERRIDE.with(|p| *p.borrow_mut() = None);
    }

    /// acquire_slot_with_locks perf-mode re-map: when the host is too small
    /// for the requested perf topology, compute_pinning's TopologyInsufficient
    /// is re-mapped to a TYPED PerfModeUnavailable (a permanent
    /// host-insufficiency — the isolation guarantee cannot be honored on ANY
    /// slot of this host), distinct from the transient ResourceContention.
    /// Host = 1 LLC / 2 CPUs; request = 4 vCPUs. The shortfall is detected by
    /// compute_pinning BEFORE any resource lock, so the synthetic host needs
    /// no flock fixture.
    #[test]
    fn acquire_slot_with_locks_host_too_small_is_perf_mode_unavailable() {
        let host = host_topology::HostTopology::new_for_tests(&[(vec![0, 1], 0)]);
        let topo = topology::Topology::new(1, 1, 4, 1);
        let err = acquire_slot_with_locks(&host, &topo)
            .expect_err("4 vCPUs on a 2-CPU host cannot satisfy the perf topology");
        assert!(
            err.downcast_ref::<host_topology::PerfModeUnavailable>()
                .is_some(),
            "host-too-small must re-map TopologyInsufficient -> PerfModeUnavailable \
             (a host-insufficiency, distinct from the transient ResourceContention): {err:#}",
        );
    }

    /// Explicit `memory_mib(0)` must be rejected at build time rather
    /// than surfacing as an opaque KVM ioctl failure later. The
    /// builder default (None→256) passes.
    #[test]
    fn builder_rejects_explicit_zero_memory() {
        // build()'s no-perf path reads KTSTR_BYPASS_LLC_LOCKS + KTSTR_CPU_CAP
        // before the memory_mib guard. Under the shared env lock, pin
        // bypass=1 + cpu_cap unset so build() short-circuits the slot/LLC
        // acquire path (no acquire_llc_plan contention; cpu_cap=None avoids
        // the bypass+cpu_cap bail), leaving the memory_mib(0) rejection.
        use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
        let _l = lock_env();
        let _g = EnvVarGuard::set(crate::KTSTR_BYPASS_LLC_LOCKS_ENV, "1");
        let _c = EnvVarGuard::remove(crate::KTSTR_CPU_CAP_ENV);
        // Point at a real file so the kernel-existence check
        // (which runs before the memory_mib guard) does not short-
        // circuit. /bin/true exists on every host the tests care
        // about; its contents don't matter for this check.
        let kernel = std::path::PathBuf::from("/bin/true");
        let result = KtstrVmBuilder::default()
            .kernel(&kernel)
            .memory_mib(0)
            .no_perf_mode(true)
            .build();
        let err = match result {
            Err(e) => e,
            Ok(_) => panic!("build() must reject memory_mib(0)"),
        };
        let msg = format!("{err:#}");
        assert!(
            msg.contains("memory_mib") && msg.contains("> 0"),
            "error must name the field and constraint: {msg}"
        );
    }

    #[test]
    fn builder_topology() {
        let b = KtstrVmBuilder::default().topology(Topology::new(1, 2, 4, 2));
        assert_eq!(b.topology.total_cpus(), 16);
        assert_eq!(b.topology.llcs, 2);
    }

    #[test]
    fn builder_cpu_budget_setter() {
        assert_eq!(KtstrVmBuilder::default().cpu_budget, None);
        let b = KtstrVmBuilder::default().cpu_budget(16);
        assert_eq!(b.cpu_budget, Some(16));
    }

    #[test]
    fn builder_requires_kernel() {
        let result = KtstrVmBuilder::default().build();
        assert!(result.is_err());
    }

    #[test]
    fn builder_rejects_missing_kernel() {
        let result = KtstrVmBuilder::default()
            .kernel("/nonexistent/vmlinuz")
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn builder_chain() {
        let b = KtstrVmBuilder::default()
            .topology(Topology::new(1, 2, 2, 2))
            .memory_mib(4096)
            .cmdline("root=/dev/sda")
            .timeout(Duration::from_secs(300));
        assert_eq!(b.memory_mib, Some(4096));
        assert_eq!(b.topology.total_cpus(), 8);
        assert_eq!(b.cmdline_extra, "root=/dev/sda");
        assert_eq!(b.timeout, Duration::from_secs(300));
    }

    #[test]
    fn builder_with_init_binary() {
        let exe = crate::resolve_current_exe().unwrap();
        let b = KtstrVmBuilder::default().init_binary(&exe);
        assert_eq!(b.init_binary.as_deref(), Some(exe.as_path()));
    }

    #[test]
    fn builder_rejects_missing_init_binary() {
        let result = KtstrVmBuilder::default()
            .kernel("/nonexistent/vmlinuz")
            .init_binary("/nonexistent/binary")
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn builder_rejects_missing_scheduler_binary() {
        let exe = crate::resolve_current_exe().unwrap();
        let result = KtstrVmBuilder::default()
            .kernel(&exe)
            .scheduler_binary("/nonexistent/scheduler")
            .build();
        assert!(result.is_err());
    }

    /// x86_64: build() rejects more than `MAX_VIRTIO_NICS` NICs (the virtio-pci
    /// INTx GSI budget) BEFORE any VM spawn, and exactly `MAX_VIRTIO_NICS` does
    /// NOT trip the cap. Pins the advertised limit — `.network()` appends
    /// unconditionally, so the cap is the only guard against routing a NIC to a
    /// GSI past the IOAPIC line budget.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn builder_rejects_more_than_max_virtio_nics() {
        let exe = crate::resolve_current_exe().unwrap();
        let net = crate::vmm::net_config::NetConfig::DEFAULT;

        // MAX + 1 NICs: rejected, naming the cap.
        let mut over = KtstrVmBuilder::default().kernel(&exe);
        for _ in 0..=crate::vmm::kvm::MAX_VIRTIO_NICS {
            over = over.network(net);
        }
        // KtstrVm has no Debug, so match rather than expect_err.
        let msg = match over.build() {
            Ok(_) => panic!("MAX+1 NICs must be rejected"),
            Err(e) => format!("{e:#}"),
        };
        assert!(
            msg.contains("too many NICs") && msg.contains("max"),
            "expected NIC-cap diagnostic, got: {msg}"
        );

        // Exactly MAX NICs must NOT trip the cap (this minimal fixture may fail
        // later for other reasons, but never on the NIC budget).
        let mut at_cap = KtstrVmBuilder::default().kernel(&exe);
        for _ in 0..crate::vmm::kvm::MAX_VIRTIO_NICS {
            at_cap = at_cap.network(net);
        }
        if let Err(e) = at_cap.build() {
            let m = format!("{e:#}");
            assert!(
                !m.contains("too many NICs"),
                "MAX NICs must not trip the cap, got: {m}"
            );
        }
    }

    /// aarch64: build() rejects more than one NIC — the virtio-MMIO transport
    /// installs a single NIC; multi-NIC needs the x86_64 PCI transport. Pins
    /// the arch-specific advertised limit.
    #[test]
    #[cfg(not(target_arch = "x86_64"))]
    fn builder_rejects_multiple_nics_on_aarch64() {
        let exe = crate::resolve_current_exe().unwrap();
        let net = crate::vmm::net_config::NetConfig::DEFAULT;
        // KtstrVm has no Debug, so match rather than expect_err.
        let msg = match KtstrVmBuilder::default()
            .kernel(&exe)
            .network(net)
            .network(net)
            .build()
        {
            Ok(_) => panic!("2 NICs on aarch64 must be rejected"),
            Err(e) => format!("{e:#}"),
        };
        assert!(
            msg.contains("single virtio-MMIO NIC"),
            "expected aarch64 single-NIC diagnostic, got: {msg}"
        );
    }

    #[test]
    fn builder_run_args() {
        let b = KtstrVmBuilder::default().run_args(&["run".into(), "--json".into()]);
        assert_eq!(b.run_args, vec!["run", "--json"]);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn builder_kernel_dir_resolves_bzimage() {
        let b = KtstrVmBuilder::default().kernel_dir("/some/linux");
        assert_eq!(
            b.kernel.as_deref(),
            Some(std::path::Path::new("/some/linux/arch/x86/boot/bzImage"))
        );
    }

    #[test]
    #[should_panic(expected = "invalid Topology")]
    fn builder_rejects_zero_llcs() {
        KtstrVmBuilder::default().topology(Topology::new(1, 0, 2, 2));
    }

    #[test]
    #[should_panic(expected = "invalid Topology")]
    fn builder_rejects_zero_cores() {
        KtstrVmBuilder::default().topology(Topology::new(1, 2, 0, 2));
    }

    #[test]
    #[should_panic(expected = "invalid Topology")]
    fn builder_rejects_zero_threads() {
        KtstrVmBuilder::default().topology(Topology::new(1, 2, 2, 0));
    }

    #[test]
    fn builder_watchdog_timeout_default() {
        let b = KtstrVmBuilder::default();
        assert_eq!(b.watchdog_timeout, Some(Duration::from_secs(5)));
    }

    #[test]
    fn builder_watchdog_timeout_override() {
        let b = KtstrVmBuilder::default().watchdog_timeout(Duration::from_secs(5));
        assert_eq!(b.watchdog_timeout, Some(Duration::from_secs(5)));
    }

    #[test]
    fn builder_rendezvous_timeout_default() {
        let b = KtstrVmBuilder::default();
        assert_eq!(b.rendezvous_timeout, None);
    }

    #[test]
    fn builder_rendezvous_timeout_override() {
        let b = KtstrVmBuilder::default().rendezvous_timeout(Duration::from_millis(100));
        assert_eq!(b.rendezvous_timeout, Some(Duration::from_millis(100)));
    }

    #[test]
    fn builder_exec_timeout_default() {
        let b = KtstrVmBuilder::default();
        assert_eq!(b.exec_timeout, Duration::from_secs(120));
    }

    #[test]
    fn builder_exec_timeout_override() {
        let b = KtstrVmBuilder::default().exec_timeout(Duration::from_secs(30));
        assert_eq!(b.exec_timeout, Duration::from_secs(30));
    }

    #[test]
    fn builder_monitor_thresholds_sets() {
        let t = crate::monitor::MonitorThresholds {
            max_imbalance_ratio: 2.0,
            ..Default::default()
        };
        let b = KtstrVmBuilder::default().monitor_thresholds(t);
        assert!(b.monitor_thresholds.is_some());
    }

    #[test]
    fn builder_sched_args() {
        let b = KtstrVmBuilder::default().sched_args(&["--enable-borrow".into()]);
        assert_eq!(b.sched_args, vec!["--enable-borrow"]);
    }

    #[test]
    fn builder_performance_mode_default_false() {
        let b = KtstrVmBuilder::default();
        assert!(!b.performance_mode);
    }

    #[test]
    fn builder_performance_mode_set() {
        let b = KtstrVmBuilder::default().performance_mode(true);
        assert!(b.performance_mode);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn builder_kernel_dir_resolves_image() {
        let b = KtstrVmBuilder::default().kernel_dir("/some/linux");
        assert_eq!(
            b.kernel.as_deref(),
            Some(std::path::Path::new("/some/linux/arch/arm64/boot/Image"))
        );
    }
}
