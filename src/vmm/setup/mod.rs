//! Boot pipeline for `KtstrVm`: virtio-blk wiring, KVM creation,
//! initramfs resolution and compression, COW overlay, deferred memory
//! computation, x86_64 / aarch64 memory and FDT layout, vCPU register
//! setup.
//!
//! These methods run on the calling thread (no vCPU work yet) and
//! produce a [`KtstrKvm`](super::kvm::KtstrKvm) ready for the
//! [`KtstrVm::run_vm`](super::KtstrVm::run_vm) loop. They are reopened
//! as additional [`impl KtstrVm`](super::KtstrVm) blocks; the canonical
//! struct definition lives in [`super`].

use anyhow::{Context, Result};
use std::path::PathBuf;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Instant;
use vm_memory::{Bytes, GuestAddress, GuestMemory, GuestMemoryMmap};

use super::KtstrVm;
use super::initramfs_cache::{BaseKey, BaseRef, get_or_build_base, get_or_compress_base_shm};
use super::memory_budget::{
    MemoryBudget, TmpfsFraction, initramfs_min_memory_mib, read_kernel_init_size,
    read_kernel_version, read_kernel_version_from_metadata_sidecar,
};
use super::pi_mutex::PiMutex;
use super::{disk_config, disk_template, host_topology, initramfs, virtio_blk, virtio_net};

#[cfg(target_arch = "aarch64")]
use super::aarch64;
#[cfg(target_arch = "aarch64")]
use super::aarch64::boot;
#[cfg(target_arch = "aarch64")]
use super::aarch64::kvm;
#[cfg(target_arch = "x86_64")]
use super::virtio_console;
#[cfg(target_arch = "x86_64")]
use super::x86_64::{acpi, boot, kvm, mptable};

/// Address where initramfs is loaded in guest memory.
#[cfg(target_arch = "x86_64")]
const INITRD_ADDR: u64 = 0x800_0000; // 128 MiB

/// Compute initramfs load address at the high end of DRAM, just below
/// the FDT. Matches Firecracker/Cloud Hypervisor placement pattern —
/// avoids conflicts with early kernel allocations near the kernel image.
///
/// Aligned to the host page size (not a hardcoded 4 KB). On Apple
/// Silicon hosts the kernel runs with 16 KB pages and the COW
/// `MAP_FIXED` mmap rejects targets that aren't host-page-aligned
/// with `EINVAL` — a 4 KB-aligned guest address that happens to fall
/// mid-host-page would clobber unrelated mappings if the kernel
/// accepted it, so the kernel correctly refuses. Round down here so
/// the overlay path reaches `mmap` with a valid alignment regardless
/// of host page size.
#[cfg(target_arch = "aarch64")]
fn aarch64_initrd_addr(memory_mib: u32, total_cpus: u32, initrd_max_size: u64) -> Result<u64> {
    // Ceiling is the PVTIME carve base, NOT the FDT address. setup_pvtime
    // registers the per-vCPU steal-time IPAs in [pvtime_base, fdt_addr) and
    // write_memory shrinks the advertised /memory to END at pvtime_base. An
    // initrd whose top enters that carve is corrupted two independent ways:
    // (1) on the FIRST KVM_RUN, before any guest code executes, the host
    // writes the 8-byte stolen_time field at steal_base+8 (the kvm_update_
    // stolen_time call from check_vcpu_requests) — the full 64-byte struct
    // is zeroed later, guest-triggered via the PV_TIME_ST hypercall, after
    // initrd unpack; and (2) the carve is outside advertised RAM, so the
    // guest kernel never memblock-reserves those pages. Either clobbers the
    // initramfs and the guest's /init never starts. Anchor the whole initrd
    // below pvtime_base so it stays in advertised RAM, clear of the carve.
    let ceiling = aarch64::fdt::pvtime_base(memory_mib, total_cpus);
    let page_size = host_page_size();
    let mask = !(page_size - 1);
    // Place initrd just below the PVTIME carve, host-page-aligned. Use
    // checked_sub: a compressed initramfs larger than the advertised RAM
    // span [DRAM_START, pvtime_base) would otherwise wrap the u64 (debug:
    // panic; release with overflow-checks off: a near-u64::MAX value that
    // would PASS the >= DRAM_START check and advertise a bogus
    // linux,initrd-start). The min-memory budget sizes RAM for the
    // tmpfs/init constraint, not for 'initrd fits below pvtime_base', so
    // this bound is payload-reachable. The initrd must reside entirely
    // within advertised RAM: an initrd above pvtime_base is outside the
    // advertised /memory and the guest kernel never memblock-reserves it
    // (see this function's header comment).
    let load_addr = ceiling
        .checked_sub(initrd_max_size)
        .map(|top| top & mask)
        .with_context(|| {
            format!(
                "compressed initrd ({initrd_max_size} bytes) exceeds the \
                 RAM span below the PVTIME carve (pvtime_base={ceiling:#x}): \
                 reduce initramfs size or increase VM memory"
            )
        })?;
    anyhow::ensure!(
        load_addr >= kvm::DRAM_START,
        "initrd load address {load_addr:#x} underflows DRAM_START {:#x} \
         (compressed initrd {initrd_max_size} bytes, pvtime_base {ceiling:#x}): \
         reduce initramfs size or increase VM memory",
        kvm::DRAM_START,
    );
    Ok(load_addr)
}

/// Host page size in bytes. Reads from `sysconf(_SC_PAGESIZE)` once
/// per process and caches the result via `OnceLock`; subsequent calls
/// hit the cache. The kernel reports the actual MMU page size (4 KB
/// on x86_64 / common aarch64, 16 KB on Apple Silicon and some
/// aarch64 server SKUs). Falls back to 4 KB only when `sysconf`
/// returns an error code (≤0), which would itself indicate a libc bug
/// — the fallback exists so a downstream alignment computation never
/// produces 0.
#[allow(dead_code)]
pub(crate) fn host_page_size() -> u64 {
    static CACHED: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        // SAFETY: sysconf is a thread-safe libc function that takes a
        // constant integer argument and returns a long. No invariants
        // on the caller side.
        let sz = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if sz > 0 { sz as u64 } else { 0x1000 }
    })
}

/// Build the auto-mount cmdline tokens for one disk. Returns an
/// empty string when no auto-mount is requested (Raw filesystem,
/// or `no_auto_mount` opt-out); otherwise returns the
/// space-prefixed `KTSTR_DISK0_FS=... KTSTR_DISK0_MOUNT=...`
/// pair, with `KTSTR_DISK0_RO=1` appended when `read_only` is
/// set.
///
/// Free fn so cfg(test) unit tests cover all branches without
/// driving a full `setup_memory` call.
///
/// Token contract (consumed by
/// `crate::vmm::rust_init::auto_mount_data_disks`):
/// * `KTSTR_DISK0_FS=<cache_tag>` — fstype string for the
///   `mount(2)` syscall. Reuses `Filesystem::cache_tag()` so the
///   on-disk-format identifier and the cmdline value stay in
///   lockstep.
/// * `KTSTR_DISK0_MOUNT=<path>` — guest-side mount point. Driven
///   by `DiskConfig::auto_mount_path` (`/mnt/<name>` when
///   `name` is set, `/mnt/disk0` otherwise).
/// * `KTSTR_DISK0_RO=1` — emitted only when `read_only` is set
///   (matches the host-side virtio-blk F_RO advertisement). The
///   guest sets `MS_RDONLY` proactively rather than letting the
///   kernel fail with -EROFS when bdev RO meets RW mount.
#[allow(dead_code)]
pub(crate) fn disk_auto_mount_cmdline_tokens(disk: &disk_config::DiskConfig) -> String {
    if disk.filesystem == disk_config::Filesystem::Raw || disk.no_auto_mount {
        return String::new();
    }
    let mut s = format!(
        " KTSTR_DISK0_FS={} KTSTR_DISK0_MOUNT={}",
        disk.filesystem.cache_tag(),
        disk.auto_mount_path(),
    );
    if disk.read_only {
        s.push_str(" KTSTR_DISK0_RO=1");
    }
    s
}

/// Guest kernel cmdline flags common to both arches, with the
/// arch-specific tail spliced in. Centralized so a flag added once
/// applies to x86_64 AND aarch64: a per-arch drift here previously left
/// `sysctl.vm.overcommit_memory=1` on x86 only, OOM-ing the aarch64
/// guest /init on its allocator reservation.
///
/// `arch_extra` is the per-arch tail (x86_64: no_timer_check /
/// clocksource / i8042 / pci=off / reboot=k; aarch64: kfence). Callers
/// append dynamic tokens (earlycon, loglevel, rdinit, disk auto-mount,
/// numa, wprof, cmdline_extra) after this base. Cmdline params are
/// order-independent, so the common-then-arch ordering is irrelevant.
fn base_guest_cmdline(arch_extra: &str) -> String {
    // KASLR is ON by default — ktstr.kconfig pins CONFIG_RANDOMIZE_BASE=y
    // (text-image slide; x86 also CONFIG_RANDOMIZE_MEMORY=y for the
    // direct-map slides). The host derives the runtime virt-KASLR offset
    // (x86: MSR_LSTAR readback + KERN_ADDRS _text; aarch64: KERN_ADDRS
    // _text only — no MSR_LSTAR) and threads it via coord_kaslr_offset()
    // into every kaslr-aware site. Tests opt out via
    // #[ktstr_test(kaslr = false)] / Scheduler::kargs(&["nokaslr"]).
    //
    // vm.overcommit_memory=1 (OVERCOMMIT_ALWAYS): the guest /init is a
    // jemalloc-backed test binary that maps more virtual address space
    // than its resident set. Under the default heuristic (mode 0)
    // __vm_enough_memory rejects a single mapping larger than free RAM,
    // so on a deferred-sized guest the /init aborts with "memory
    // allocation of N bytes failed" before the workload runs. ALWAYS mode
    // admits the mapping; the resident set then stays within guest RAM
    // so no OOM-kill follows (a coverage-instrumented /init's larger
    // resident set — the live __llvm_prf_cnts + __llvm_prf_data
    // sections — is covered by the memory_budget coverage reserve).
    // (arm64's arch_mm_preinit auto-enables
    // ALWAYS only for PAGE_SIZE>=16K with <=128 physpages.)
    format!(
        "console=ttyS0 nomodules mitigations=off random.trust_cpu=on \
         swiotlb=noforce panic=-1 lockdown=none \
         sysctl.kernel.unprivileged_bpf_disabled=0 \
         sysctl.kernel.sched_schedstats=1 delayacct \
         sysctl.kernel.task_delayacct=1 sysctl.vm.overcommit_memory=1 \
         {arch_extra} KTSTR_GUEST=1"
    )
}

/// NUMA-balancing cmdline token. The kernel handler
/// `setup_numabalancing` (mm/mempolicy.c) accepts ONLY the strings
/// "enable" / "disable" via `strcmp`; any other value (e.g. "0")
/// leaves the parse result 0, logs `pr_warn("Unable to parse
/// numa_balancing=")`, and is IGNORED — so the kernel keeps its
/// compiled CONFIG_NUMA_BALANCING_DEFAULT_ENABLED state instead of
/// being turned off. Memory-only (CXL) topologies want balancing ON to
/// migrate pages toward CPU-bearing nodes; the uniform/default case
/// wants it OFF to keep scheduler measurements free of migration
/// noise. Extracted (like `base_guest_cmdline`) so the x86_64
/// (`build_guest_cmdline`) and aarch64 (`finish_aarch64_setup`) sites
/// share one definition and a host unit test can pin both branches.
fn numa_balancing_cmdline_token(topology: &crate::vmm::topology::Topology) -> &'static str {
    if topology.has_memory_only_nodes() {
        " numa_balancing=enable"
    } else {
        " numa_balancing=disable"
    }
}

/// Pure helper: assemble the `extras` slice and the [`BaseKey`] from
/// the resolved scheduler/probe/worker/staged-binary paths. Extracted
/// out of [`KtstrVm::spawn_initramfs_resolve`] so the staged-extras
/// path-format contract, the per-staged iteration order, and the
/// shell-mode-vs-non-shell BaseKey threading can be unit-tested
/// without spawning the resolve thread or running the full
/// initramfs build.
///
/// Caller responsibilities:
/// - Pre-compute `staged_extras_names` as
///   `format!("{}/scheduler", staged_scheduler_archive_dir(&s.name))`
///   for each staged scheduler (the helper indexes into this vec by
///   position, so caller MUST keep order identical to
///   `staged_schedulers`). Materialized externally so the borrow
///   lifetime ties to the caller's owned Vec.
/// - Pre-compute `merged_includes` (operator's `include_files` plus
///   the optional alloc-worker binary).
/// - Pre-compute `has_jemalloc_extras` = `probe.is_some() ||
///   worker.is_some()` for shell-mode determination.
///
/// Returns `(extras, base_key)`. The extras vec borrows from
/// `scheduler`, `probe`, `staged_extras_names`, and
/// `staged_schedulers` — all `'a`-tied to the caller's lifetimes.
/// The base_key is owned `BaseKey`.
///
/// `#[allow(clippy::too_many_arguments)]` — the parameter set is
/// intrinsically flat (binaries + staging slice + flags); folding
/// into a builder or struct here would just rename the same
/// positional ordering. Sibling precedent: `build_vm_builder_base`
/// in `src/test_support/runtime.rs` uses the same allow for the
/// same reason.
#[allow(clippy::too_many_arguments)]
pub(crate) fn assemble_extras_and_key<'a>(
    payload: &'a std::path::Path,
    scheduler: Option<&'a std::path::Path>,
    probe: Option<&'a std::path::Path>,
    worker: Option<&'a std::path::Path>,
    staged_schedulers: &'a [crate::vmm::builder::StagedScheduler],
    staged_extras_names: &'a [String],
    merged_includes: &'a [(String, PathBuf)],
    busybox_bytes: Option<&[u8]>,
    has_jemalloc_extras: bool,
) -> Result<(Vec<(&'a str, &'a std::path::Path)>, BaseKey)> {
    debug_assert_eq!(
        staged_schedulers.len(),
        staged_extras_names.len(),
        "staged_schedulers and staged_extras_names must be co-indexed; \
         caller mis-built the extras-names slice"
    );

    let mut extras: Vec<(&str, &std::path::Path)> = Vec::new();
    if let Some(s) = scheduler {
        extras.push(("scheduler", s));
    }
    if let Some(p) = probe {
        extras.push(("bin/ktstr-jemalloc-probe", p));
    }
    for (idx, staged) in staged_schedulers.iter().enumerate() {
        extras.push((staged_extras_names[idx].as_str(), staged.binary.as_path()));
    }

    // Shell-mode determination: busybox flag, non-empty includes,
    // or any jemalloc extras (probe / worker present). Mirrors the
    // pre-extraction logic in spawn_initramfs_resolve — kept
    // explicit here so the helper is a closed unit under test
    // without a hidden dependency on the caller's shell_mode
    // computation.
    let shell_mode = busybox_bytes.is_some() || !merged_includes.is_empty() || has_jemalloc_extras;

    let staged_for_key: Vec<(&str, &std::path::Path)> = staged_schedulers
        .iter()
        .map(|s| (s.name.as_str(), s.binary.as_path()))
        .collect();

    let key = if shell_mode {
        BaseKey::new_shell(
            payload,
            scheduler,
            probe,
            worker,
            &staged_for_key,
            merged_includes,
            busybox_bytes,
        )?
    } else {
        BaseKey::new(payload, scheduler, probe, worker, &staged_for_key)?
    };

    Ok((extras, key))
}

impl KtstrVm {
    /// Construct the optional virtio-blk device for the configured
    /// disk in `self.disks`. Returns `Ok(None)` when no disk is
    /// attached.
    ///
    /// On `Ok(Some(_))`, the returned `Arc<PiMutex<VirtioBlk>>` has:
    ///   - the backing file open (sparse temp file when
    ///     `disk.backing_path` is `None`, otherwise the operator-supplied
    ///     path),
    ///   - the file extended to `disk.capacity_bytes()` (so unallocated
    ///     reads return zeros via short-read padding in `handle_read`),
    ///   - the throttle wired in,
    ///   - the irqfd registered with the VM,
    ///   - guest memory set so subsequent `process_requests` calls can
    ///     read/write descriptor data.
    ///
    /// The framework reserves a single MMIO base + IRQ pair
    /// (`VIRTIO_BLK_MMIO_BASE` / `VIRTIO_BLK_IRQ`); the builder's
    /// `.disk()` enforces the single-disk constraint by overwriting
    /// any previous disk on each call.
    pub(super) fn init_virtio_blk(
        &self,
        vm: &kvm::KtstrKvm,
    ) -> Result<Option<Arc<PiMutex<virtio_blk::VirtioBlk>>>> {
        if self.disks.is_empty() {
            return Ok(None);
        }
        let disk = &self.disks[0];
        let capacity = disk.capacity_bytes();

        // Throttle sanity gate. `DiskThrottle::validate` rejects
        // burst capacities below their refill rate (which would
        // silently cap the steady-state at the lower capacity
        // instead of the configured rate) and burst capacities set
        // without a refill rate (a one-shot bucket that never
        // refills). Run BEFORE allocating any backing-file resources
        // so a misconfigured throttle bails before disk-side host
        // commitments.
        //
        // The typed `DiskThrottleValidationError` carries the
        // failing dimension (iops/bytes) so callers downcasting via
        // `err.downcast_ref::<DiskThrottleValidationError>()` can
        // route a programmatic recovery without parsing the
        // rendered message.
        disk.throttle
            .validate()
            .map_err(|e| anyhow::anyhow!(e).context("invalid disk throttle"))?;

        // Per-test backing-file allocation forks on the configured
        // [`disk_config::Filesystem`], with one override for the
        // template-build VM driver:
        //
        //  - **`template_staging_image` set** (internal-only — see
        //    [`KtstrVmBuilder::template_staging_image`]): open the
        //    caller-supplied path RW and hand it to the device. This
        //    branch exists exclusively for
        //    [`disk_template::build_template_via_vm`]: the driver
        //    materialises a sparse staging image, points the
        //    template-build guest at it via this field, and recovers
        //    the now-formatted file after VM exit for
        //    [`disk_template::store_atomic`]. Bypasses both the
        //    `Raw` tempfile and `Btrfs` ensure_template branches so
        //    the template-build VM cannot recursively re-enter the
        //    cache it is itself populating.
        //
        //  - `Raw`: anonymous sparse `tempfile()`. The kernel
        //    reclaims storage when the device drops the File. No
        //    cache, no FICLONE.
        //
        //  - `Btrfs`: FICLONE-clones the host-cached, guest-formatted
        //    template into a per-test tempfile under the cache root
        //    (so FICLONE source and dest share a filesystem), unlinks
        //    the dest immediately after open so the device sees the
        //    same anonymous-file semantics as the `Raw` path, and
        //    hands the open `File` to the `VirtioBlk` device. See
        //    [`crate::vmm::disk_template`] module docs.
        let backing = if let Some(staging) = self.template_staging_image.as_ref() {
            let f = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(staging)
                .with_context(|| {
                    format!(
                        "open template staging image {} for virtio-blk",
                        staging.display(),
                    )
                })?;
            // Enforce the file-size = advertised-capacity invariant.
            // The in-tree caller (`disk_template::build_template_via_vm`)
            // sizes the staging file via
            // `create_and_size_staging_image` before invoking the
            // builder, so this is normally a no-op. Calling `set_len`
            // here makes the contract local to the device-init path
            // — a caller-supplied staging image that is too small or
            // too large is normalised to `capacity` instead of
            // letting virtio-blk advertise a size that disagrees with
            // the backing file. Sparse-file semantics match the Raw
            // branch above: holes don't consume disk space until
            // written.
            f.set_len(capacity)
                .context("set template staging image length to capacity")?;
            f
        } else {
            match disk.filesystem {
                disk_config::Filesystem::Raw => {
                    let f = tempfile::tempfile()
                        .context("create virtio-blk sparse temp backing file")?;
                    // Make sure the file covers the advertised capacity.
                    // set_len creates a sparse file: holes don't consume
                    // disk space until written.
                    f.set_len(capacity)
                        .context("set virtio-blk backing file length")?;
                    f
                }
                disk_config::Filesystem::Btrfs => {
                    let template =
                        disk_template::ensure_template(disk_config::Filesystem::Btrfs, capacity)
                            .context("ensure btrfs disk template")?;
                    let cache_root = disk_template::cache_root()
                        .context("resolve disk-template cache root for per-test clone")?;
                    std::fs::create_dir_all(&cache_root)
                        .with_context(|| format!("create cache root {cache_root:?}"))?;
                    // Generate a unique per-test path under the cache
                    // root. Use pid + timestamp_ns + random_u64 so
                    // concurrent tests in the same process and across
                    // processes never collide.
                    let dest = cache_root.join(format!(
                        ".per-test-{pid}-{ns:x}-{rnd:x}.img",
                        pid = std::process::id(),
                        ns = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map(|d| d.as_nanos())
                            .unwrap_or(0),
                        rnd = rand::random::<u64>(),
                    ));
                    let f = disk_template::clone_to_per_test(&template, &dest)
                        .context("FICLONE template into per-test backing")?;
                    // Unlink the dest path immediately. The open File
                    // keeps the inode alive for the device's lifetime;
                    // the kernel reclaims storage on drop, matching the
                    // `tempfile()` semantics of the Raw branch.
                    //
                    // If the unlink fails (very rare — ENOENT means a
                    // peer beat us to it, EACCES means the operator's
                    // cache permissions are broken, EBUSY can come from
                    // some FUSE backings), we keep the open File and
                    // warn — the device still works on the open fd, the
                    // only consequence is a stale path on disk that the
                    // next cache GC sweeps. Do NOT propagate the error,
                    // because the device's per-test backing is already
                    // valid and aborting VM init would be a regression
                    // versus the Raw branch where `tempfile::tempfile()`
                    // returns an already-unlinked file with no failure
                    // mode.
                    if let Err(e) = std::fs::remove_file(&dest) {
                        tracing::warn!(
                            path = %dest.display(),
                            error = %e,
                            "failed to unlink per-test btrfs backing after \
                             FICLONE; the open File still backs the device, \
                             but the leftover path will accumulate in the \
                             cache directory until manual cleanup or the \
                             next disk-template cache GC pass."
                        );
                    }
                    f
                }
            }
        };

        let mut blk =
            virtio_blk::VirtioBlk::with_options(backing, capacity, disk.throttle, disk.read_only);
        // Worker placement extracted from the host-topology plan.
        // Perf-mode produces `pinning_plan.service_cpu` (a dedicated
        // host CPU reserved away from vCPU pins) — the worker pins
        // there to keep its cache footprint out of the workload-
        // measured cpuset. Non-perf + `--cpu-cap` produces
        // `no_perf_plan.cpus` (the LLC mask shared with vCPUs); the
        // worker shares the LLC but stays inside the resource budget.
        // The two paths are orthogonal (perf-mode never has
        // `no_perf_plan` and vice versa); both `None` means inherit
        // the parent's affinity (degraded-sysfs / non-cap-set
        // fallback). The setter only takes effect on the next worker
        // spawn — `with_options` deferred initial spawn to DRIVER_OK
        // (matching the respawn path), so this call lands inside the
        // window and the first worker observes the placement.
        let placement = virtio_blk::WorkerPlacement {
            service_cpu: self.pinning_plan.as_ref().and_then(|p| p.service_cpu),
            no_perf_cpus: self.no_perf_plan.as_ref().map(|p| p.cpus.clone()),
        };
        blk.set_worker_placement(placement);
        blk.set_mem((*vm.guest_mem).clone());
        let blk_arc = Arc::new(PiMutex::new(blk));

        // irqfd registration. On x86's split-irqchip path (max APIC ID > 254)
        // the device IRQ is delivered via the userspace IOAPIC: register_irqfd
        // binds the GSI's eventfd, and the guest's IOAPIC RTE write installs
        // the matching MSI route (see super::x86_64::ioapic + IoapicHandle).
        // On the in-kernel-irqchip (x86 <=254) and aarch64 (GIC) paths the
        // kernel routes the GSI directly. The call is identical on both arches.
        vm.vm_fd
            .register_irqfd(blk_arc.lock().irq_evt(), kvm::VIRTIO_BLK_IRQ)
            .context("register virtio-blk irqfd")?;

        Ok(Some(blk_arc))
    }

    /// Construct the optional virtio-net device for the configured
    /// network in `self.network`. Returns `Ok(None)` when no network
    /// is attached.
    ///
    /// On `Ok(Some(_))`, the returned `Arc<PiMutex<VirtioNet>>` has:
    ///   - the configured MAC baked into config space,
    ///   - guest memory set so subsequent `process_tx_loopback` calls
    ///     can read TX descriptor data and write into RX descriptors,
    ///   - the irqfd registered with the VM (delivered via the userspace
    ///     IOAPIC on x86 split-irqchip, the in-kernel IOAPIC/GIC otherwise).
    ///
    /// The framework reserves a single MMIO base + IRQ pair
    /// (`VIRTIO_NET_MMIO_BASE` / `VIRTIO_NET_IRQ`); the builder's
    /// `.network()` enforces the single-device constraint by
    /// overwriting any previous network on each call.
    pub(super) fn init_virtio_net(
        &self,
        vm: &kvm::KtstrKvm,
    ) -> Result<Option<Arc<PiMutex<virtio_net::VirtioNet>>>> {
        let Some(cfg) = self.network else {
            return Ok(None);
        };
        let mut dev = virtio_net::VirtioNet::new(cfg);
        dev.set_mem((*vm.guest_mem).clone());
        let net_arc = Arc::new(PiMutex::new(dev));

        // irqfd registration. On x86's split-irqchip path (>254 APIC IDs) the
        // device IRQ routes through the userspace IOAPIC (the guest's RTE write
        // installs the MSI route); on the in-kernel-irqchip (x86 <=254) and
        // aarch64 (GIC) paths the kernel routes the GSI. Identical on both.
        vm.vm_fd
            .register_irqfd(net_arc.lock().irq_evt(), kvm::VIRTIO_NET_IRQ)
            .context("register virtio-net irqfd")?;

        Ok(Some(net_arc))
    }

    /// Create the KVM VM and optionally load the kernel.
    ///
    /// When `memory_mib` is `Some`, allocates guest memory and loads the
    /// kernel immediately (existing path). When `None` (deferred), creates
    /// the VM without memory — allocation and kernel loading happen later
    /// in `setup_memory` after the actual initramfs size is known.
    pub(super) fn create_vm_and_load_kernel(
        &self,
    ) -> Result<(kvm::KtstrKvm, Option<boot::KernelLoadResult>)> {
        let t0 = Instant::now();
        let use_hugepages = self.performance_mode
            && self.memory_mib.is_some_and(|mib| {
                host_topology::hugepages_free() >= host_topology::hugepages_needed(mib)
            });

        let mut vm = match self.memory_mib {
            Some(mib) => {
                if use_hugepages {
                    kvm::KtstrKvm::new_with_hugepages(self.topology, mib, self.performance_mode)
                        .context("create VM with hugepages")?
                } else {
                    kvm::KtstrKvm::new(self.topology, mib, self.performance_mode)
                        .context("create VM")?
                }
            }
            None => {
                kvm::KtstrKvm::new_deferred(self.topology, use_hugepages, self.performance_mode)
                    .context("create VM (deferred memory)")?
            }
        };
        tracing::debug!(elapsed_us = t0.elapsed().as_micros(), "kvm_create");

        // Propagate the builder's PCI-enable flag to the VM so the run loops
        // construct the PCI host bridge and the ACPI/cmdline gate on it.
        vm.pci_enabled = self.pci_enabled;

        // When memory is already allocated (non-deferred path), do mbind
        // and load kernel now. Deferred path does this in setup_memory.
        let kernel_result = if self.memory_mib.is_some() {
            if self.performance_mode && !self.mbind_node_map.is_empty() {
                let layout = vm.numa_layout.as_ref().expect(
                    "numa_layout is Some on the non-deferred allocation path: \
                     allocate_and_register_memory ran during `vm_new` because \
                     memory_mib was provided up front, and that call sets \
                     numa_layout to Some(...) in src/vmm/{x86_64,aarch64}/kvm.rs",
                );
                layout.mbind_regions(&vm.guest_mem, &self.mbind_node_map);
            }

            let t0 = Instant::now();
            let kr = boot::load_kernel(&vm.guest_mem, &self.kernel).context("load kernel")?;
            tracing::debug!(elapsed_us = t0.elapsed().as_micros(), "load_kernel");
            Some(kr)
        } else {
            None
        };

        Ok((vm, kernel_result))
    }

    /// Spawn initramfs resolution on a background thread.
    /// Returns the handle to join later (after KVM creation completes).
    pub(super) fn spawn_initramfs_resolve(&self) -> Option<JoinHandle<Result<(BaseRef, BaseKey)>>> {
        let bin = self.init_binary.as_ref()?;
        let payload = bin.clone();
        let scheduler = self.scheduler_binary.clone();
        let probe = self.jemalloc_probe_binary.clone();
        let worker = self.jemalloc_alloc_worker_binary.clone();
        let include_files = self.include_files.clone();
        let staged_schedulers = self.staged_schedulers.clone();
        let busybox_bytes = self.busybox_bytes.clone();
        #[cfg(feature = "wprof")]
        let wprof_host_path: Option<PathBuf> = self.wprof.as_ref().map(|w| w.host_path.clone());
        std::thread::Builder::new()
            .name("initramfs-resolve".into())
            .spawn(move || -> Result<(BaseRef, BaseKey)> {
                // Extras are stripped by `build_initramfs_base`
                // before write. The scheduler and probe can lose
                // their DWARF without functional impact — the probe
                // resolves `tsd_s.thread_allocated` offsets against
                // the TARGET process's `/proc/<pid>/exe`, not against
                // its own binary, so its own DWARF is dead weight.
                // The worker (the probe's target) MUST retain DWARF:
                // a stripped worker has no DWARF for the probe to
                // walk. Route scheduler + probe through `extras`
                // (stripped), worker through `include_files`
                // (verbatim). Packing the probe unstripped inflated
                // the initramfs by ~900MB per run in debug builds,
                // which was enough to time out VM init before the
                // test binary loaded.
                //
                // Staged schedulers ride the same `extras` path,
                // packed under `staging/schedulers/<name>/scheduler`
                // so the cpio extractor's silent parent-dir
                // requirement gets satisfied via the auto-registered
                // ancestor entries (see `build_initramfs_base`'s
                // `register_parent_dirs` loop). Each staged binary
                // contributes its own DT_NEEDED set to the shared-lib
                // resolution chain — schedulers built against
                // different libbpf revisions are correctly handled
                // without operator intervention.
                let staged_extras_names: Vec<String> = staged_schedulers
                    .iter()
                    .map(|s| {
                        format!(
                            "{}/scheduler",
                            crate::test_support::staged::staged_scheduler_archive_dir(&s.name),
                        )
                    })
                    .collect();
                let has_jemalloc_extras = probe.as_deref().is_some() || worker.as_deref().is_some();

                // Merge include_files with worker so both the cache
                // key and the actual archive build see the same
                // worker entry; the probe is added to extras inside
                // `assemble_extras_and_key`. wprof (when set) also
                // rides include_files so DT_NEEDED resolution pulls
                // its dynamic dependencies (libelf, libz, blazesym
                // C ABI) into the archive alongside the binary;
                // without that, wprof fails to load inside the
                // guest.
                let mut merged_includes: Vec<(String, PathBuf)> = include_files.clone();
                if let Some(w) = worker.as_deref() {
                    merged_includes.push((
                        "bin/ktstr-jemalloc-alloc-worker".to_string(),
                        w.to_path_buf(),
                    ));
                }
                #[cfg(feature = "wprof")]
                if let Some(wprof_path) = wprof_host_path.as_deref() {
                    merged_includes.push(("bin/wprof".to_string(), wprof_path.to_path_buf()));
                }

                let (extras, key) = assemble_extras_and_key(
                    &payload,
                    scheduler.as_deref(),
                    probe.as_deref(),
                    worker.as_deref(),
                    &staged_schedulers,
                    &staged_extras_names,
                    &merged_includes,
                    busybox_bytes.as_deref(),
                    has_jemalloc_extras,
                )?;

                let include_refs: Vec<(&str, &std::path::Path)> = merged_includes
                    .iter()
                    .map(|(a, p)| (a.as_str(), p.as_path()))
                    .collect();
                let base =
                    get_or_build_base(&payload, &extras, &include_refs, busybox_bytes, &key)?;
                Ok((base, key))
            })
            .ok()
    }

    /// Compress base+suffix as separate LZ4 legacy streams, load into
    /// guest memory via COW overlay (falling back to write_slice), and
    /// verify the write. Returns `total_compressed_size`.
    ///
    /// On a successful COW overlay, the returned `CowOverlayGuard` is
    /// pushed onto `vm.cow_overlay_guards` IMMEDIATELY — before any
    /// subsequent fallible operation (suffix write, read-back verify)
    /// runs. This is deliberate: if a later `?` unwinds this function
    /// after the MAP_FIXED overlay is in place, a locally-held guard
    /// would drop first, releasing `LOCK_SH` while the COW VMAs are
    /// still live. A concurrent writer could then take `LOCK_EX` and
    /// truncate the segment → SIGBUS on the mapped pages. Pushing the
    /// guard onto `vm` transfers ownership to the VM, where Drop
    /// order is structurally enforced (guard drops AFTER
    /// `_reservation` munmaps the COW VMAs).
    fn compress_and_load_initrd(
        &self,
        vm: &mut kvm::KtstrKvm,
        base_bytes: &[u8],
        suffix: &[u8],
        key: &BaseKey,
        load_addr: u64,
    ) -> Result<u32> {
        let uncompressed_size = base_bytes.len() + suffix.len();

        // Compress base and suffix as separate LZ4 legacy streams. The
        // kernel initramfs decompressor handles concatenated LZ4 natively
        // (re-encountering the magic mid-stream resets the decoder).
        // Keeping them separate lets us COW-map the base from SHM.
        let t0 = Instant::now();
        let lz4_base = self.get_or_compress_base(base_bytes, key)?;
        let lz4_suffix = initramfs::lz4_legacy_compress(suffix);
        let total_compressed = lz4_base.len() + lz4_suffix.len();
        tracing::debug!(
            elapsed_us = t0.elapsed().as_micros(),
            uncompressed = uncompressed_size,
            lz4_base = lz4_base.len(),
            lz4_suffix = lz4_suffix.len(),
            ratio = format!("{:.1}x", uncompressed_size as f64 / total_compressed as f64),
            "lz4_initramfs",
        );

        tracing::debug!(
            base_magic = format!(
                "{:02x}{:02x}{:02x}{:02x}",
                lz4_base[0], lz4_base[1], lz4_base[2], lz4_base[3]
            ),
            suffix_magic = format!(
                "{:02x}{:02x}{:02x}{:02x}",
                lz4_suffix[0], lz4_suffix[1], lz4_suffix[2], lz4_suffix[3]
            ),
            base_len = lz4_base.len(),
            suffix_len = lz4_suffix.len(),
            total = total_compressed,
            load_addr = format!("{:#x}", load_addr),
            suffix_addr = format!("{:#x}", load_addr + lz4_base.len() as u64),
            "initrd_load_debug",
        );

        // Try COW overlay: mmap compressed base from SHM fd directly
        // into guest memory, sharing physical pages across VMs.
        let t0 = Instant::now();
        let cow_guard = Self::try_cow_overlay(&vm.guest_mem, key, lz4_base.len(), load_addr);
        // IMPORTANT: stash the guard on the VM IMMEDIATELY — before
        // any fallible operation below. If a `?` unwinds this function
        // with a locally-held guard still on the stack, the guard
        // drops first, releasing LOCK_SH while the COW VMAs are still
        // live. Owned by `vm`, the guard drops with the VM's
        // declared-order Drop, which is strictly after
        // `_reservation` (and thus the COW VMAs). See
        // `try_cow_overlay_rejects_cross_region_span` and the C4
        // comment on `cow_overlay_guards` in kvm.rs.
        let cow_active = cow_guard.is_some();
        if let Some(guard) = cow_guard {
            vm.cow_overlay_guards.push(guard);
        }
        if cow_active {
            vm.guest_mem
                .write_slice(&lz4_suffix, GuestAddress(load_addr + lz4_base.len() as u64))
                .context("write lz4 suffix after COW base")?;
            tracing::debug!(
                elapsed_us = t0.elapsed().as_micros(),
                cow = true,
                "initrd_write"
            );
        } else {
            initramfs::load_initramfs_parts(&vm.guest_mem, &[&lz4_base, &lz4_suffix], load_addr)?;
            tracing::debug!(
                elapsed_us = t0.elapsed().as_micros(),
                cow = false,
                "initrd_write"
            );
        }

        // Read back first 8 bytes from guest memory to check write.
        let mut check_buf = [0u8; 8];
        vm.guest_mem
            .read_slice(&mut check_buf, GuestAddress(load_addr))
            .context("read-back initrd check")?;
        tracing::debug!(
            first_8 = format!(
                "{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
                check_buf[0],
                check_buf[1],
                check_buf[2],
                check_buf[3],
                check_buf[4],
                check_buf[5],
                check_buf[6],
                check_buf[7]
            ),
            expected_magic = "02214c18",
            "initrd_verify",
        );

        Ok(total_compressed as u32)
    }

    /// Select the guest rootfs tmpfs fraction for the budget formula by
    /// reading the kernel's version and gating on the honoring versions
    /// (mainline 6.18+ or a stable series at/above its backport floor)
    /// via [`TmpfsFraction::for_kernel_version`].
    ///
    /// Mirrors [`Self::init_payload_coverage_reserve`]: a `&self` accessor
    /// that derives a conservatively-defaulting value threaded into
    /// [`MemoryBudget`] at every budget call site. The version is read
    /// from the image's own setup_header where it is embedded
    /// ([`read_kernel_version`], the x86_64 bzImage), falling
    /// back to the cache `metadata.json` sidecar
    /// ([`read_kernel_version_from_metadata_sidecar`]) for images without
    /// an embedded version — notably the aarch64 `Image`. Both sources
    /// return `None` on any uncertainty, so
    /// [`TmpfsFraction::for_kernel_version`] yields the safe
    /// [`TmpfsFraction::Half`] unless the kernel is positively confirmed
    /// to honor the token — 90% is never taken on a guess.
    fn tmpfs_fraction(&self) -> TmpfsFraction {
        let version = read_kernel_version(&self.kernel)
            .or_else(|| read_kernel_version_from_metadata_sidecar(&self.kernel));
        TmpfsFraction::for_kernel_version(version)
    }

    /// Probe the `/init` payload for LLVM coverage instrumentation and,
    /// when instrumented, compute the extra resident memory the
    /// instrumented runtime needs.
    ///
    /// Returns `(instrumented, reserve_bytes)`:
    /// - `instrumented` is `true` when the payload's `.symtab` carries
    ///   `__llvm_profile_write_buffer` / `__llvm_profile_get_size_for_buffer`
    ///   (the same function-shaped symbols
    ///   [`crate::test_support::try_flush_profraw`] resolves; chosen over
    ///   the bare `__llvm_profile_runtime` marker because the marker can
    ///   be dead-stripped entirely under `--gc-sections` while these
    ///   function symbols are kept alive by that flush call's link
    ///   reference). This probes the
    ///   PAYLOAD bytes (`self.init_binary`), NOT the host process — the
    ///   guest `/init` may be instrumented even when the host VMM binary
    ///   is not (and vice versa), so
    ///   `current_binary_is_coverage_instrumented` (which probes
    ///   `/proc/self/exe`) is the wrong signal here.
    /// - `reserve_bytes` is the summed `sh_size` of the payload's
    ///   `__llvm_prf_cnts` + `__llvm_prf_data` sections — the live
    ///   coverage-counter and profile-metadata arrays. This is the floor
    ///   on the heap buffer `__llvm_profile_write_buffer` serializes into
    ///   at flush time. `0` (and `instrumented = false`) on any failure
    ///   path (no payload, unreadable file, ELF parse error, symbols
    ///   absent) — the conservative outcome, leaving the budget at its
    ///   non-instrumented size.
    ///
    /// Reads the payload via `memmap2::Mmap` (matching the
    /// `try_flush_profraw` idiom) so a ~1 GiB coverage binary is not
    /// copied into the heap just to read its section table.
    fn init_payload_coverage_reserve(&self) -> (bool, u64) {
        let Some(path) = self.init_binary.as_ref() else {
            return (false, 0);
        };
        let Ok(file) = std::fs::File::open(path) else {
            return (false, 0);
        };
        // SAFETY: `path` is the test-author-supplied `/init` payload;
        // ktstr never writes to it during a VM build, satisfying
        // memmap2's no-concurrent-modification invariant. The fd pins
        // the inode for the mapping's lifetime.
        let Ok(mmap) = (unsafe { memmap2::Mmap::map(&file) }) else {
            return (false, 0);
        };
        let Ok(elf) = goblin::elf::Elf::parse(&mmap) else {
            return (false, 0);
        };

        let vaddrs = crate::test_support::find_symbol_vaddrs(
            &elf,
            &[
                "__llvm_profile_write_buffer",
                "__llvm_profile_get_size_for_buffer",
            ],
        );
        let instrumented = vaddrs.iter().any(|v| matches!(v, Some(va) if *va != 0));
        if !instrumented {
            return (false, 0);
        }

        // Sum the live profile sections' resident sizes. `sh_size` is
        // the in-memory size (the counter/metadata arrays the runtime
        // keeps resident and serializes at flush time).
        let mut reserve_bytes: u64 = 0;
        for sh in &elf.section_headers {
            if let Some(name) = elf.shdr_strtab.get_at(sh.sh_name)
                && (name == "__llvm_prf_cnts" || name == "__llvm_prf_data")
            {
                reserve_bytes = reserve_bytes.saturating_add(sh.sh_size);
            }
        }
        (true, reserve_bytes)
    }

    /// Join the initramfs thread and load the result into guest memory.
    /// Memory must already be allocated (non-deferred path). Validates
    /// that allocated memory is sufficient for the initramfs.
    ///
    /// x86_64-only: aarch64 uses
    /// `Self::join_and_load_initramfs_aarch64`, which computes the
    /// FDT-relative load address from the compressed size after the
    /// suffix is built (the address depends on `memory_mib` AND the
    /// total compressed size, neither of which is known until after
    /// the suffix and compression run).
    #[cfg(target_arch = "x86_64")]
    fn join_and_load_initramfs(
        &self,
        vm: &mut kvm::KtstrKvm,
        handle: JoinHandle<Result<(BaseRef, BaseKey)>>,
        load_addr: u64,
    ) -> Result<(Option<u64>, Option<u32>)> {
        let t0 = Instant::now();
        let (base, key) = handle
            .join()
            .map_err(|_| anyhow::anyhow!("initramfs-resolve thread panicked"))??;
        tracing::debug!(elapsed_us = t0.elapsed().as_micros(), "initramfs_join");
        let base_bytes: &[u8] = base.as_ref();

        let t0 = Instant::now();
        let suffix = initramfs::build_suffix(base_bytes.len(), &self.suffix_params())?;
        let uncompressed_size = base_bytes.len() + suffix.len();
        tracing::debug!(
            elapsed_us = t0.elapsed().as_micros(),
            base_bytes = base_bytes.len(),
            suffix_bytes = suffix.len(),
            "build_suffix",
        );

        // Enforce minimum memory for initramfs extraction.
        // This path is only reached when memory_mib was set explicitly.
        let memory_mib = self.memory_mib.expect(
            "join_and_load_initramfs called in deferred mode; \
             use join_compute_memory_and_load instead",
        );
        // Compress first to get actual compressed size for validation.
        let lz4_base = self.get_or_compress_base(base_bytes, &key)?;
        let lz4_suffix = initramfs::lz4_legacy_compress(&suffix);
        let compressed_size = lz4_base.len() + lz4_suffix.len();
        let kernel_init_size = read_kernel_init_size(&self.kernel)?;
        let (init_coverage_instrumented, instrumented_reserve_bytes) =
            self.init_payload_coverage_reserve();
        let budget = MemoryBudget {
            uncompressed_initramfs_bytes: uncompressed_size as u64,
            compressed_initrd_bytes: compressed_size as u64,
            kernel_init_size,
            init_coverage_instrumented,
            instrumented_reserve_bytes,
            tmpfs_fraction: self.tmpfs_fraction(),
        };
        let min_mib = initramfs_min_memory_mib(&budget);
        if memory_mib < min_mib {
            anyhow::bail!(
                "VM memory {}MiB insufficient for initramfs \
                 (uncompressed={}MiB, compressed={}MiB, \
                 init_size={}MiB): need {}MiB",
                memory_mib,
                uncompressed_size >> 20,
                compressed_size >> 20,
                kernel_init_size >> 20,
                min_mib,
            );
        }

        let size = self.compress_and_load_initrd(vm, base_bytes, &suffix, &key, load_addr)?;
        Ok((Some(load_addr), Some(size)))
    }

    /// Deferred memory path: join initramfs, compute memory from actual
    /// size, allocate guest memory, then load initramfs.
    ///
    /// Returns `(initrd_addr, initrd_size, memory_mib)`.
    ///
    /// x86_64-only: aarch64 uses
    /// `Self::join_compute_memory_and_load_aarch64`, which orders
    /// the load_addr computation after `allocate_and_register_memory`
    /// (the FDT-relative initrd address depends on `memory_mib`,
    /// which is itself computed from the post-compress total size).
    #[cfg(target_arch = "x86_64")]
    fn join_compute_memory_and_load(
        &self,
        vm: &mut kvm::KtstrKvm,
        handle: JoinHandle<Result<(BaseRef, BaseKey)>>,
        load_addr: u64,
    ) -> Result<(Option<u64>, Option<u32>, u32)> {
        let t0 = Instant::now();
        let (base, key) = handle
            .join()
            .map_err(|_| anyhow::anyhow!("initramfs-resolve thread panicked"))??;
        tracing::debug!(elapsed_us = t0.elapsed().as_micros(), "initramfs_join");
        let base_bytes: &[u8] = base.as_ref();

        let t0 = Instant::now();
        let suffix = initramfs::build_suffix(base_bytes.len(), &self.suffix_params())?;
        let uncompressed_size = base_bytes.len() + suffix.len();
        tracing::debug!(
            elapsed_us = t0.elapsed().as_micros(),
            base_bytes = base_bytes.len(),
            suffix_bytes = suffix.len(),
            "build_suffix",
        );

        let t0_compress = Instant::now();
        let lz4_base = self.get_or_compress_base(base_bytes, &key)?;
        let lz4_suffix = initramfs::lz4_legacy_compress(&suffix);
        let compressed_size = lz4_base.len() + lz4_suffix.len();
        tracing::debug!(
            elapsed_us = t0_compress.elapsed().as_micros(),
            uncompressed = uncompressed_size,
            compressed = compressed_size,
            ratio = format!("{:.1}x", uncompressed_size as f64 / compressed_size as f64),
            "deferred_lz4_compress",
        );

        // Compute memory from actual sizes, honoring the
        // topology-requested minimum when non-zero.
        let kernel_init_size = read_kernel_init_size(&self.kernel)?;
        let (init_coverage_instrumented, instrumented_reserve_bytes) =
            self.init_payload_coverage_reserve();
        let budget = MemoryBudget {
            uncompressed_initramfs_bytes: uncompressed_size as u64,
            compressed_initrd_bytes: compressed_size as u64,
            kernel_init_size,
            init_coverage_instrumented,
            instrumented_reserve_bytes,
            tmpfs_fraction: self.tmpfs_fraction(),
        };
        let memory_mib = initramfs_min_memory_mib(&budget).max(self.memory_min_mib);
        tracing::debug!(
            uncompressed_mib = uncompressed_size >> 20,
            compressed_mib = compressed_size >> 20,
            init_size_mib = kernel_init_size >> 20,
            coverage_instrumented = init_coverage_instrumented,
            coverage_reserve_mib = instrumented_reserve_bytes >> 20,
            memory_min_mib = self.memory_min_mib,
            memory_mib,
            "deferred_memory_computed",
        );

        // Allocate and register guest memory.
        vm.allocate_and_register_memory(memory_mib)
            .with_context(|| format!("allocate deferred memory ({memory_mib}MiB)"))?;

        // Load pre-compressed data into guest memory. The base is already
        // in the LZ4 SHM cache from get_or_compress_base above, so
        // compress_and_load_initrd will hit the cache.
        let size = self.compress_and_load_initrd(vm, base_bytes, &suffix, &key, load_addr)?;
        Ok((Some(load_addr), Some(size), memory_mib))
    }

    pub(super) fn effective_memory_mib(&self, guest_mem: &GuestMemoryMmap) -> u32 {
        use vm_memory::GuestMemoryRegion;
        match self.memory_mib {
            Some(mib) => mib,
            None => {
                let total_bytes: u64 = guest_mem.iter().map(|r| r.len()).sum();
                (total_bytes >> 20) as u32
            }
        }
    }

    /// Get or build the compressed base, delegating to the SHM
    /// one-compressor election in `initramfs_cache`. Thin wrapper: the
    /// SHM logic uses no builder state, only the content hash.
    fn get_or_compress_base(&self, base_bytes: &[u8], key: &BaseKey) -> Result<Vec<u8>> {
        Ok(get_or_compress_base_shm(key.0, base_bytes))
    }

    /// Try to COW-overlay the compressed base from LZ4 SHM into guest
    /// memory. Returns `Some(CowOverlayGuard)` on success — the guard
    /// owns the SHM fd and holds `LOCK_SH` for the mapping's lifetime,
    /// and MUST be kept alive as long as the COW overlay is in use
    /// (typically the VM lifetime). Validates the segment starts with
    /// LZ4 legacy magic to reject stale data from a previous
    /// compression format.
    ///
    /// Associated function (no `&self`): the COW path is a pure
    /// transform of `(guest_mem, key, expected_len, load_addr)` —
    /// it reads the SHM segment keyed by `key.0` and maps it into
    /// `guest_mem`, touching no VM instance state. Keeping it
    /// `self`-free lets the unit test drive the real overlay logic
    /// without constructing a full `KtstrVm`.
    fn try_cow_overlay(
        guest_mem: &GuestMemoryMmap,
        key: &BaseKey,
        expected_len: usize,
        load_addr: u64,
    ) -> Option<initramfs::CowOverlayGuard> {
        let (fd, len) = initramfs::shm_open_lz4(key.0)?;
        if len != expected_len {
            initramfs::shm_close_fd(fd);
            return None;
        }
        // Validate LZ4 legacy magic before COW-mapping. pread the
        // first 4 bytes directly — no need to mmap the entire segment
        // just to peek at the header.
        use std::os::fd::AsRawFd;
        let mut magic = [0u8; 4];
        // SAFETY: `fd` is owned by `shm_open_lz4` and remains valid
        // until `shm_close_fd` below; `magic` is a 4-byte stack buffer
        // and the read length is exactly 4. The fd refers to a SHM
        // segment with `len >= expected_len` bytes (verified above and
        // by `shm_open_lz4`'s fstat check).
        let n = unsafe {
            libc::pread(
                fd.as_raw_fd(),
                magic.as_mut_ptr() as *mut libc::c_void,
                4,
                0,
            )
        };
        if n != 4 {
            initramfs::shm_close_fd(fd);
            return None;
        }
        if magic != initramfs::LZ4_LEGACY_MAGIC {
            tracing::warn!(
                magic = format!(
                    "{:02x}{:02x}{:02x}{:02x}",
                    magic[0], magic[1], magic[2], magic[3]
                ),
                "stale compressed shm segment in COW path, skipping"
            );
            initramfs::shm_close_fd(fd);
            return None;
        }
        // Refuse zero-length: mmap(len=0) is EINVAL and serves no
        // purpose; the suffix-write fallback handles empty bases
        // trivially. Also refuse load_addr + len overflow before
        // bounds-checking, since GuestAddress arithmetic wraps
        // silently on u64 overflow.
        if len == 0 || load_addr.checked_add(len as u64).is_none() {
            tracing::debug!(
                load_addr = format!("{:#x}", load_addr),
                len,
                "cow_overlay: invalid range (zero-length or overflow), falling back"
            );
            initramfs::shm_close_fd(fd);
            return None;
        }
        // The MAP_FIXED mmap rounds `len` up to the next host page
        // boundary internally — Apple Silicon kernels run with 16 KB
        // pages, so a 5000-byte segment mapped against a 16 KB-page
        // host actually clobbers 16384 bytes of host VA. Bounds-check
        // against the rounded-up length so we don't accept a mapping
        // that overruns the guest region, and reject load_addr that
        // isn't host-page-aligned (mmap returns EINVAL otherwise).
        #[cfg(target_arch = "aarch64")]
        let host_page = host_page_size();
        // x86_64 hosts always run with 4 KB pages, and the call sites
        // page-align load_addr to 4 KB; the rounded-up length matches
        // `len` exactly. Use the constant instead of paying for a
        // sysconf(2) on every overlay attempt.
        #[cfg(target_arch = "x86_64")]
        let host_page: u64 = 0x1000;
        if load_addr & (host_page - 1) != 0 {
            tracing::debug!(
                load_addr = format!("{:#x}", load_addr),
                host_page,
                "cow_overlay: load_addr not host-page-aligned, falling back"
            );
            initramfs::shm_close_fd(fd);
            return None;
        }
        let rounded_len = (len as u64)
            .checked_add(host_page - 1)
            .map(|v| v & !(host_page - 1));
        let Some(rounded_len) = rounded_len else {
            tracing::debug!(
                load_addr = format!("{:#x}", load_addr),
                len,
                "cow_overlay: rounded length overflows u64, falling back"
            );
            initramfs::shm_close_fd(fd);
            return None;
        };
        // Bounds-check [load_addr, load_addr + rounded_len) against
        // guest memory BEFORE the MAP_FIXED mmap. `get_host_address`
        // only validates the start address — without a length check,
        // MAP_FIXED would silently overwrite whatever host VA happens
        // to follow the region (other guest regions, reserved VA, or
        // unrelated mappings). `get_slice` fails if the range extends
        // past the region's end or spans a region boundary, which is
        // exactly the guarantee MAP_FIXED needs.
        let rounded_usize = match usize::try_from(rounded_len) {
            Ok(v) => v,
            Err(_) => {
                tracing::debug!(
                    load_addr = format!("{:#x}", load_addr),
                    rounded_len,
                    "cow_overlay: rounded length exceeds usize, falling back"
                );
                initramfs::shm_close_fd(fd);
                return None;
            }
        };
        if guest_mem
            .get_slice(GuestAddress(load_addr), rounded_usize)
            .is_err()
        {
            tracing::debug!(
                load_addr = format!("{:#x}", load_addr),
                len,
                rounded_len,
                "cow_overlay: range exceeds guest memory region, falling back"
            );
            initramfs::shm_close_fd(fd);
            return None;
        }
        let Ok(host_addr) = guest_mem.get_host_address(GuestAddress(load_addr)) else {
            initramfs::shm_close_fd(fd);
            return None;
        };
        // cow_overlay takes ownership of `fd` on both Some and None
        // paths: on success the guard carries it; on failure
        // cow_overlay itself closes it. Do NOT call shm_close_fd here.
        unsafe { initramfs::cow_overlay(host_addr, len, fd) }
    }

    /// Write cmdline, boot params, and topology tables to guest memory.
    ///
    /// When `kernel_result` is `None` (deferred memory mode), this method
    /// first joins the initramfs thread to learn the actual size, allocates
    /// guest memory from that size, does mbind, and loads the kernel — all
    /// before proceeding with the normal initramfs load and boot param setup.
    #[cfg(target_arch = "x86_64")]
    pub(super) fn setup_memory(
        &self,
        vm: &mut kvm::KtstrKvm,
        kernel_result: Option<boot::KernelLoadResult>,
        initramfs_handle: Option<JoinHandle<Result<(BaseRef, BaseKey)>>>,
    ) -> Result<boot::KernelLoadResult> {
        // Deferred memory path: join initramfs first to learn its size,
        // then allocate memory, load kernel, and load initramfs — all in
        // one shot with no estimation.
        let (kernel_result, initrd_addr, initrd_size) = if let Some(kr) = kernel_result {
            // Non-deferred: memory already allocated, kernel already loaded.
            // compress_and_load_initrd transfers the CowOverlayGuard
            // directly onto vm.cow_overlay_guards before any fallible
            // operation, so a mid-function `?` cannot drop the guard
            // before the COW VMAs are torn down.
            let (initrd_addr, initrd_size) = match initramfs_handle {
                Some(handle) => self.join_and_load_initramfs(vm, handle, INITRD_ADDR)?,
                None => (None, None),
            };
            (kr, initrd_addr, initrd_size)
        } else {
            // Deferred memory path: join initramfs first to learn its size,
            // then allocate memory, load kernel, and load initramfs — all in
            // one shot with no estimation.
            let (initrd_addr, initrd_size, _memory_mib) = match initramfs_handle {
                Some(handle) => self.join_compute_memory_and_load(vm, handle, INITRD_ADDR)?,
                None => {
                    // No initramfs — allocate minimum memory.
                    let memory_mib = 256u32;
                    vm.allocate_and_register_memory(memory_mib)
                        .context("allocate deferred memory (no initramfs)")?;
                    (None, None, memory_mib)
                }
            };

            if self.performance_mode && !self.mbind_node_map.is_empty() {
                let layout = vm.numa_layout.as_ref().expect(
                    "numa_layout is Some after the deferred allocate_and_register_memory \
                     call above: that call sets numa_layout to Some(...) in \
                     src/vmm/{x86_64,aarch64}/kvm.rs before this branch can reach here",
                );
                layout.mbind_regions(&vm.guest_mem, &self.mbind_node_map);
            }

            // Load kernel into the freshly allocated memory.
            let t0 = Instant::now();
            let kr = boot::load_kernel(&vm.guest_mem, &self.kernel).context("load kernel")?;
            tracing::debug!(elapsed_us = t0.elapsed().as_micros(), "load_kernel");

            (kr, initrd_addr, initrd_size)
        };

        // Resolve effective memory_mib for boot params / ACPI / SHM.
        let memory_mib = self.effective_memory_mib(&vm.guest_mem);

        let cmdline = self.build_guest_cmdline();

        let t0 = Instant::now();
        boot::write_cmdline(&vm.guest_mem, &cmdline)?;
        boot::write_boot_params(
            &vm.guest_mem,
            &cmdline,
            memory_mib,
            initrd_addr,
            initrd_size,
            kernel_result.setup_header.as_ref(),
        )?;
        tracing::debug!(elapsed_us = t0.elapsed().as_micros(), "cmdline_boot_params");

        let t0 = Instant::now();
        mptable::setup_mptable(&vm.guest_mem, &self.topology)?;
        let _acpi_layout = acpi::setup_acpi(
            &vm.guest_mem,
            &self.topology,
            vm.numa_layout.as_ref().expect(
                "numa_layout is Some by the time setup_acpi runs: \
                 memory allocation (whether deferred or not) ran earlier \
                 in this function and set numa_layout via \
                 allocate_and_register_memory in src/vmm/x86_64/kvm.rs",
            ),
            vm.pci_enabled,
        )?;
        tracing::debug!(elapsed_us = t0.elapsed().as_micros(), "mptable_acpi");

        Ok(kernel_result)
    }

    /// Build the guest kernel cmdline (base flags + per-device `virtio_mmio.device=` tokens).
    #[cfg(target_arch = "x86_64")]
    fn build_guest_cmdline(&self) -> String {
        // Kernel cmdline rationale (per flag):
        //   console=ttyS0        — serial console for host-visible output.
        //   nomodules            — no out-of-tree modules are shipped; skip modprobe paths.
        //   mitigations=off      — skip Spectre/Meltdown mitigations for VM perf.
        //   no_timer_check       — suppress APIC timer-calibration failure under KVM.
        //   clocksource=kvm-clock — stable paravirt clock; avoid TSC drift under KVM.
        //   random.trust_cpu=on  — seed RNG from RDRAND so userspace doesn't block on entropy.
        //   swiotlb=noforce      — skip the IOMMU bounce buffer — no passthrough devices.
        //   i8042.*=noaux/nomux/nopnp/dumbkbd — skip legacy PS/2 probing; no keyboard/mouse in VM.
        //   pci=off              — (virtio-MMIO-only guests) skip the PCI scan; dropped when the virtio-PCI transport is enabled.
        //   reboot=k             — use keyboard-controller reset method.
        //   panic=-1             — reboot immediately on panic; host detects via exit.
        //   lockdown=none        — permit /dev/mem and unrestricted BPF needed by the test runtime.
        //   sysctl.kernel.unprivileged_bpf_disabled=0 — allow BPF load from the test runtime.
        //   sysctl.kernel.sched_schedstats=1          — enable /proc/schedstat for workload reports.
        //   delayacct                                 — bare boot param consumed by the
        //                                              kernel's `__setup("delayacct", ...)`
        //                                              handler at kernel/delayacct.c:43-48.
        //                                              The handler sets `delayacct_on = 1`
        //                                              during EARLY boot, BEFORE
        //                                              `delayacct_init()` (line 50-55) reads
        //                                              the variable to decide whether to
        //                                              enable the static branch. This is the
        //                                              authoritative way to turn the
        //                                              delayacct subsystem on at boot.
        //   sysctl.kernel.task_delayacct=1            — backup runtime toggle that flips the
        //                                              delayacct_key static_branch via the
        //                                              `kernel.task_delayacct` sysctl declared
        //                                              at kernel/delayacct.c:80. This path
        //                                              fires later via deferred sysctl
        //                                              registration + proc_handler invocation,
        //                                              which has timing fragility relative to
        //                                              the early-boot increment paths
        //                                              (delayacct_blkio_start/_end gated by
        //                                              static_branch_unlikely(&delayacct_key)
        //                                              at kernel/delayacct.c). Both forms are
        //                                              specified — belt and suspenders — so
        //                                              the runtime toggle is on regardless of
        //                                              whether the early-boot or the deferred
        //                                              sysctl path runs first. Without either,
        //                                              /proc/<tid>/stat field 42 and the
        //                                              taskstats delay-accounting fields stay
        //                                              zero on every kernel built with
        //                                              CONFIG_TASK_DELAY_ACCT=y but boot-time
        //                                              off (the upstream default since v5.14).
        // `pci=off` is dropped when the virtio-PCI transport is enabled so the
        // guest enumerates the PCI host bridge; otherwise it skips the scan.
        let pci_flag = if self.pci_enabled { "" } else { "pci=off " };
        let mut cmdline = base_guest_cmdline(&format!(
            "no_timer_check clocksource=kvm-clock i8042.noaux i8042.nomux \
             i8042.nopnp i8042.dumbkbd {pci_flag}reboot=k"
        ));
        let verbose = std::env::var(crate::KTSTR_VERBOSE_ENV)
            .map(|v| v == "1")
            .unwrap_or(false)
            || std::env::var("RUST_BACKTRACE").is_ok_and(|v| v == "1" || v == "full");
        if verbose {
            cmdline.push_str(" earlyprintk=serial loglevel=7");
        } else {
            cmdline.push_str(" loglevel=0");
        }
        if self.init_binary.is_some() {
            cmdline.push_str(" rdinit=/init initramfs_options=size=90%");
        }
        // Virtio-console MMIO device on the kernel cmdline. The kernel's
        // virtio_mmio_cmdline_devices driver parses this to register the
        // MMIO transport at the given base address and IRQ.
        cmdline.push_str(&format!(
            " virtio_mmio.device={:#x}@{:#x}:{}",
            virtio_console::VIRTIO_MMIO_SIZE,
            kvm::VIRTIO_CONSOLE_MMIO_BASE,
            kvm::VIRTIO_CONSOLE_IRQ,
        ));
        // Virtio-block MMIO device — appended only when the builder
        // attached at least one disk. The kernel's virtio_mmio_cmdline
        // parser registers a MMIO transport per `virtio_mmio.device=`
        // token; the order on the cmdline determines the device-probe
        // order, which in turn determines the `/dev/vd{a,b,...}`
        // assignment. Console-first then blk matches the expected
        // `/dev/vda = first disk` mapping.
        if !self.disks.is_empty() {
            cmdline.push_str(&format!(
                " virtio_mmio.device={:#x}@{:#x}:{}",
                virtio_blk::VIRTIO_MMIO_SIZE,
                kvm::VIRTIO_BLK_MMIO_BASE,
                kvm::VIRTIO_BLK_IRQ,
            ));
            // Auto-mount handshake. Emit a `KTSTR_DISK0_FS=<tag>`
            // token whenever the first disk has been pre-formatted so
            // the guest init at
            // [`crate::vmm::rust_init::auto_mount_data_disks`]
            // can mount `/dev/vda` at `/mnt/disk0` before the test
            // dispatch runs. `Filesystem::Raw` skips the emission
            // because there is no on-disk fs to mount; the guest
            // sees only the absent token and short-circuits the
            // mount path.
            //
            // `KTSTR_DISK0_RO=1` is emitted when the disk is
            // configured `read_only`. The virtio_blk device
            // advertises `VIRTIO_BLK_F_RO` for that case so the
            // guest's gendisk is RO; mounting RW would fail with
            // `-EROFS` (kernel `do_mount` path: `__btrfs_open_devices`
            // probes the bdev's `bdev_read_only` and returns EROFS
            // when the RW mount tries to write). The token lets the
            // guest set `MS_RDONLY` proactively, surfacing the
            // intent in the cmdline and avoiding the kernel-side
            // EROFS path.
            //
            // The cache_tag() value is reused as the fstype string
            // because it is already kebab-free, ≤8 chars, and
            // matches the on-disk-format identifier the host
            // selected — using the same value for both keeps the
            // guest mount and host cache key in lockstep, so a
            // future `Filesystem` variant rename only has to update
            // one place (the `cache_tag` match in disk_config.rs)
            // and the cmdline / mount automatically follow.
            let disk = &self.disks[0];
            cmdline.push_str(&disk_auto_mount_cmdline_tokens(disk));
        }
        // Virtio-net MMIO device — appended only when the builder
        // attached a `NetConfig`. The kernel's virtio_mmio_cmdline
        // parser registers a MMIO transport per `virtio_mmio.device=`
        // token; placing this after virtio-blk does not affect device
        // ordering on the guest's network stack (ifindex is assigned
        // independently of cmdline order).
        if self.network.is_some() {
            cmdline.push_str(&format!(
                " virtio_mmio.device={:#x}@{:#x}:{}",
                virtio_net::VIRTIO_MMIO_SIZE,
                kvm::VIRTIO_NET_MMIO_BASE,
                kvm::VIRTIO_NET_IRQ,
            ));
        }
        cmdline.push_str(numa_balancing_cmdline_token(&self.topology));
        #[cfg(feature = "wprof")]
        if let Some(wprof) = self.wprof.as_ref() {
            cmdline.push_str(" KTSTR_WPROF_ARGS=");
            cmdline.push_str(&wprof.args_cmdline());
        }
        if !self.cmdline_extra.is_empty() {
            cmdline.push(' ');
            cmdline.push_str(&self.cmdline_extra);
        }
        cmdline
    }

    /// Configure BSP and AP vCPUs.
    #[cfg(target_arch = "x86_64")]
    pub(super) fn setup_vcpus(&self, vm: &kvm::KtstrKvm, kernel_entry: u64) -> Result<()> {
        let t0 = Instant::now();
        boot::setup_sregs(&vm.guest_mem, &vm.vcpus[0], vm.split_irqchip)?;
        boot::setup_regs(&vm.vcpus[0], kernel_entry)?;
        boot::setup_fpu(&vm.vcpus[0])?;
        boot::setup_msrs(&vm.vcpus[0], None)?;
        boot::setup_lapic(&vm.vcpus[0], true)?;
        vm.vcpus[0]
            .set_mp_state(kvm_bindings::kvm_mp_state {
                mp_state: kvm_bindings::KVM_MP_STATE_RUNNABLE,
            })
            .context("set BSP mp_state")?;
        tracing::debug!(elapsed_us = t0.elapsed().as_micros(), "bsp_setup");

        let t0 = Instant::now();
        for vcpu in &vm.vcpus[1..] {
            boot::setup_fpu(vcpu)?;
            boot::setup_lapic(vcpu, false)?;
            vcpu.set_mp_state(kvm_bindings::kvm_mp_state {
                mp_state: kvm_bindings::KVM_MP_STATE_UNINITIALIZED,
            })
            .context("set AP mp_state")?;
        }
        tracing::debug!(
            elapsed_us = t0.elapsed().as_micros(),
            ap_count = vm.vcpus.len().saturating_sub(1),
            "ap_setup"
        );

        Ok(())
    }
}

#[cfg(target_arch = "aarch64")]
impl KtstrVm {
    /// Allocate and register guest memory regions for aarch64, including
    /// NUMA-aware placement.
    ///
    /// Uses the same LZ4 SHM compress cache and COW overlay path as the
    /// x86_64 `Self::setup_memory` flow. The shared helpers
    /// ([`Self::get_or_compress_base`], [`Self::compress_and_load_initrd`],
    /// [`Self::try_cow_overlay`]) are arch-neutral; this function differs
    /// from the x86_64 driver only in (a) computing the initrd load
    /// address from the dynamic FDT placement (`aarch64_initrd_addr`)
    /// instead of the fixed `INITRD_ADDR`, and (b) handing off to
    /// `finish_aarch64_setup` for FDT writing instead of boot_params /
    /// ACPI emission.
    pub(super) fn setup_memory_aarch64(
        &self,
        vm: &mut kvm::KtstrKvm,
        kernel_result: Option<boot::KernelLoadResult>,
        initramfs_handle: Option<JoinHandle<Result<(BaseRef, BaseKey)>>>,
    ) -> Result<boot::KernelLoadResult> {
        // Deferred memory path for aarch64.
        let (kernel_result, initrd_addr, initrd_size) = if let Some(kr) = kernel_result {
            // Non-deferred: memory already allocated, kernel already loaded.
            // compress_and_load_initrd transfers the CowOverlayGuard
            // directly onto vm.cow_overlay_guards before any fallible
            // operation, so a mid-function `?` cannot drop the guard
            // before the COW VMAs are torn down.
            let (initrd_addr, initrd_size) = match initramfs_handle {
                Some(handle) => {
                    // `self.memory_mib` is required on the non-deferred
                    // path: deferred boots take the early-return branch
                    // below, so we only reach this site after the builder
                    // accepted a concrete `memory_mib`. Surface it as an
                    // error rather than `unwrap()` so a future refactor
                    // that drops the deferred guard fails loudly with an
                    // actionable diagnostic instead of an opaque panic.
                    let memory_mib = self.memory_mib.context(
                        "internal: non-deferred aarch64 path requires memory_mib to be set",
                    )?;
                    self.join_and_load_initramfs_aarch64(vm, handle, memory_mib)?
                }
                None => (None, None),
            };
            (kr, initrd_addr, initrd_size)
        } else {
            // Deferred memory path: join initramfs first to learn its
            // size, allocate memory, then load kernel and initramfs.
            let (initrd_addr, initrd_size) = match initramfs_handle {
                Some(handle) => self.join_compute_memory_and_load_aarch64(vm, handle)?,
                None => {
                    // No initramfs — allocate minimum memory.
                    let memory_mib = 256u32;
                    vm.allocate_and_register_memory(memory_mib)
                        .context("allocate deferred memory (no initramfs, aarch64)")?;
                    (None, None)
                }
            };

            // Load kernel into the freshly allocated memory.
            let t0 = Instant::now();
            let kr =
                boot::load_kernel(&vm.guest_mem, &self.kernel).context("load kernel (aarch64)")?;
            tracing::debug!(elapsed_us = t0.elapsed().as_micros(), "load_kernel");

            (kr, initrd_addr, initrd_size)
        };

        self.finish_aarch64_setup(vm, kernel_result, initrd_addr, initrd_size)
    }

    /// Non-deferred aarch64 initramfs load: join handle, build suffix,
    /// compress base+suffix via the LZ4 SHM cache to learn the
    /// compressed size, validate that `memory_mib` is sufficient, compute
    /// the FDT-relative load address, then COW-or-copy the compressed
    /// stream into guest memory via the shared
    /// [`Self::compress_and_load_initrd`] path.
    fn join_and_load_initramfs_aarch64(
        &self,
        vm: &mut kvm::KtstrKvm,
        handle: JoinHandle<Result<(BaseRef, BaseKey)>>,
        memory_mib: u32,
    ) -> Result<(Option<u64>, Option<u32>)> {
        let t0 = Instant::now();
        let (base, key) = handle
            .join()
            .map_err(|_| anyhow::anyhow!("initramfs-resolve thread panicked"))??;
        tracing::debug!(elapsed_us = t0.elapsed().as_micros(), "initramfs_join");
        let base_bytes: &[u8] = base.as_ref();

        let t0 = Instant::now();
        let suffix = initramfs::build_suffix(base_bytes.len(), &self.suffix_params())?;
        let uncompressed_size = base_bytes.len() + suffix.len();
        tracing::debug!(
            elapsed_us = t0.elapsed().as_micros(),
            base_bytes = base_bytes.len(),
            suffix_bytes = suffix.len(),
            "build_suffix",
        );

        // Compress to learn the compressed size for the load_addr
        // calculation. Primes the LZ4 SHM cache so the subsequent
        // compress_and_load_initrd call hits the cache instead of
        // recompressing.
        let lz4_base = self.get_or_compress_base(base_bytes, &key)?;
        let lz4_suffix = initramfs::lz4_legacy_compress(&suffix);
        let compressed_size = lz4_base.len() + lz4_suffix.len();

        // Validate the operator-supplied memory_mib against the
        // initramfs budget. Mirrors the x86_64 join_and_load_initramfs
        // contract: a builder with too-small memory_mib fails fast here
        // instead of OOMing during boot.
        let kernel_init_size = read_kernel_init_size(&self.kernel)?;
        let (init_coverage_instrumented, instrumented_reserve_bytes) =
            self.init_payload_coverage_reserve();
        let budget = MemoryBudget {
            uncompressed_initramfs_bytes: uncompressed_size as u64,
            compressed_initrd_bytes: compressed_size as u64,
            kernel_init_size,
            init_coverage_instrumented,
            instrumented_reserve_bytes,
            tmpfs_fraction: self.tmpfs_fraction(),
        };
        let min_mib = initramfs_min_memory_mib(&budget);
        if memory_mib < min_mib {
            anyhow::bail!(
                "VM memory {}MiB insufficient for initramfs \
                 (uncompressed={}MiB, compressed={}MiB, \
                 init_size={}MiB): need {}MiB",
                memory_mib,
                uncompressed_size >> 20,
                compressed_size >> 20,
                kernel_init_size >> 20,
                min_mib,
            );
        }

        let load_addr = aarch64_initrd_addr(
            memory_mib,
            self.topology.total_cpus(),
            compressed_size as u64,
        )?;
        let size = self.compress_and_load_initrd(vm, base_bytes, &suffix, &key, load_addr)?;
        Ok((Some(load_addr), Some(size)))
    }

    /// Deferred aarch64 initramfs load: join handle, build suffix,
    /// compress (priming the LZ4 SHM cache), compute memory budget,
    /// allocate guest memory, then load initramfs via the shared
    /// COW-overlay path. Returns `(Some(load_addr), Some(size))`.
    fn join_compute_memory_and_load_aarch64(
        &self,
        vm: &mut kvm::KtstrKvm,
        handle: JoinHandle<Result<(BaseRef, BaseKey)>>,
    ) -> Result<(Option<u64>, Option<u32>)> {
        let t0 = Instant::now();
        let (base, key) = handle
            .join()
            .map_err(|_| anyhow::anyhow!("initramfs-resolve thread panicked"))??;
        tracing::debug!(elapsed_us = t0.elapsed().as_micros(), "initramfs_join");
        let base_bytes: &[u8] = base.as_ref();

        let t0 = Instant::now();
        let suffix = initramfs::build_suffix(base_bytes.len(), &self.suffix_params())?;
        let uncompressed_size = base_bytes.len() + suffix.len();
        tracing::debug!(
            elapsed_us = t0.elapsed().as_micros(),
            base_bytes = base_bytes.len(),
            suffix_bytes = suffix.len(),
            "build_suffix",
        );

        // Compress before computing memory so the budget formula uses
        // actual compressed size. Primes the LZ4 SHM cache so the
        // subsequent compress_and_load_initrd call hits it.
        let t0_compress = Instant::now();
        let lz4_base = self.get_or_compress_base(base_bytes, &key)?;
        let lz4_suffix = initramfs::lz4_legacy_compress(&suffix);
        let compressed_size = lz4_base.len() + lz4_suffix.len();
        tracing::debug!(
            elapsed_us = t0_compress.elapsed().as_micros(),
            uncompressed = uncompressed_size,
            compressed = compressed_size,
            ratio = format!("{:.1}x", uncompressed_size as f64 / compressed_size as f64),
            "deferred_lz4_compress",
        );

        let kernel_init_size = read_kernel_init_size(&self.kernel)?;
        let (init_coverage_instrumented, instrumented_reserve_bytes) =
            self.init_payload_coverage_reserve();
        let budget = MemoryBudget {
            uncompressed_initramfs_bytes: uncompressed_size as u64,
            compressed_initrd_bytes: compressed_size as u64,
            kernel_init_size,
            init_coverage_instrumented,
            instrumented_reserve_bytes,
            tmpfs_fraction: self.tmpfs_fraction(),
        };
        let memory_mib = initramfs_min_memory_mib(&budget).max(self.memory_min_mib);
        tracing::debug!(
            uncompressed_mib = uncompressed_size >> 20,
            compressed_mib = compressed_size >> 20,
            init_size_mib = kernel_init_size >> 20,
            coverage_instrumented = init_coverage_instrumented,
            coverage_reserve_mib = instrumented_reserve_bytes >> 20,
            memory_min_mib = self.memory_min_mib,
            memory_mib,
            "deferred_memory_computed",
        );

        vm.allocate_and_register_memory(memory_mib)
            .with_context(|| format!("allocate deferred memory ({memory_mib}MiB, aarch64)"))?;

        // Compute load_addr only AFTER memory_mib is known: it determines
        // the FDT position and thus pvtime_base, and the initrd now sits
        // just below pvtime_base (the PVTIME carve), not the FDT.
        let load_addr = aarch64_initrd_addr(
            memory_mib,
            self.topology.total_cpus(),
            compressed_size as u64,
        )?;

        let size = self.compress_and_load_initrd(vm, base_bytes, &suffix, &key, load_addr)?;
        Ok((Some(load_addr), Some(size)))
    }

    #[cfg(target_arch = "aarch64")]
    fn finish_aarch64_setup(
        &self,
        vm: &kvm::KtstrKvm,
        kernel_result: boot::KernelLoadResult,
        initrd_addr: Option<u64>,
        initrd_size: Option<u32>,
    ) -> Result<boot::KernelLoadResult> {
        let memory_mib = self.effective_memory_mib(&vm.guest_mem);

        // Kernel cmdline rationale (per flag) — aarch64 subset of the
        // x86_64 block above. Flags present on both arches carry the
        // same justification; see the x86_64 comment for details.
        // aarch64-specific:
        //   kfence.sample_interval=0 — disable KFENCE sampling; no real
        //                              driver faults to catch in the
        //                              test VM, and KFENCE adds boot-time
        //                              page-allocation pressure.
        let mut cmdline = base_guest_cmdline("kfence.sample_interval=0");
        // earlycon is always enabled so the kernel has a console from
        // the earliest boot stage. Without it, stdout-path auto-detection
        // is the only path to early output — and that can fail silently
        // if the FDT node isn't matched by OF_EARLYCON_DECLARE.
        // earlycon base is derived from SERIAL_MMIO_BASE so it tracks the
        // device-window placement (aarch64/kvm.rs) and can never drift.
        cmdline.push_str(&format!(
            " earlycon=uart,mmio,{:#x}",
            aarch64::kvm::SERIAL_MMIO_BASE
        ));
        let verbose = std::env::var(crate::KTSTR_VERBOSE_ENV)
            .map(|v| v == "1")
            .unwrap_or(false)
            || std::env::var("RUST_BACKTRACE").is_ok_and(|v| v == "1" || v == "full");
        if verbose {
            cmdline.push_str(" loglevel=7");
        } else {
            cmdline.push_str(" loglevel=0");
        }
        if self.init_binary.is_some() {
            cmdline.push_str(" rdinit=/init initramfs_options=size=90%");
        }
        // Auto-mount tokens for the configured disk. aarch64 advertises
        // the virtio-blk MMIO transport via FDT (see
        // `create_fdt(..., !self.disks.is_empty(), ...)` below), so the
        // `virtio_mmio.device=` cmdline form used on x86_64 is omitted.
        // The `KTSTR_DISK0_*` tokens, however, are env-style markers
        // consumed by the guest init at
        // `crate::vmm::rust_init::auto_mount_data_disks` — they are
        // arch-neutral and required on aarch64 for the same auto-mount
        // contract as x86_64.
        if !self.disks.is_empty() {
            let disk = &self.disks[0];
            cmdline.push_str(&disk_auto_mount_cmdline_tokens(disk));
        }
        cmdline.push_str(numa_balancing_cmdline_token(&self.topology));
        #[cfg(feature = "wprof")]
        if let Some(wprof) = self.wprof.as_ref() {
            cmdline.push_str(" KTSTR_WPROF_ARGS=");
            cmdline.push_str(&wprof.args_cmdline());
        }
        if !self.cmdline_extra.is_empty() {
            cmdline.push(' ');
            cmdline.push_str(&self.cmdline_extra);
        }

        let t0 = Instant::now();
        boot::validate_cmdline(&cmdline)?;

        let fdt_addr = aarch64::fdt::fdt_address(memory_mib);

        // Wire KVM PV stolen-time so the guest's /proc/stat steal
        // advances under cpu_budget overcommit. The region is carved
        // from the top of guest RAM (just below the FDT); create_fdt
        // below shrinks the /memory node to pvtime_base via the same
        // helper so the guest never reuses it. setup_pvtime gates on
        // host support (has_device_attr) and skips cleanly otherwise.
        let pvtime_base = aarch64::fdt::pvtime_base(memory_mib, self.topology.total_cpus());
        anyhow::ensure!(
            pvtime_base >= aarch64::kvm::DRAM_START && pvtime_base < fdt_addr,
            "guest RAM too small to carve the PVTIME region \
             (pvtime_base={pvtime_base:#x}, fdt_addr={fdt_addr:#x})"
        );
        vm.setup_pvtime(pvtime_base)
            .context("wire KVM PV stolen-time")?;

        let mpidrs =
            aarch64::topology::read_mpidrs(&vm.vcpus).context("read vCPU MPIDRs for FDT")?;
        let hw_cache_level = aarch64::topology::host_cache_levels();
        let guest_l1_unified = aarch64::topology::host_l1_is_unified();
        let dtb = aarch64::fdt::create_fdt(
            &self.topology,
            &mpidrs,
            memory_mib,
            &cmdline,
            initrd_addr,
            initrd_size,
            hw_cache_level,
            guest_l1_unified,
            vm.numa_layout.as_ref().expect(
                "numa_layout is Some by the time FDT creation runs: \
                 memory allocation (whether deferred or not) ran earlier \
                 in this function and set numa_layout via \
                 allocate_and_register_memory in src/vmm/aarch64/kvm.rs",
            ),
            !self.disks.is_empty(),
            self.network.is_some(),
            vm.has_pmu,
        )
        .context("create FDT")?;
        vm.guest_mem
            .write_slice(&dtb, GuestAddress(fdt_addr))
            .context("write FDT to guest memory")?;
        tracing::debug!(
            elapsed_us = t0.elapsed().as_micros(),
            fdt_addr,
            fdt_len = dtb.len(),
            "cmdline_fdt",
        );

        Ok(kernel_result)
    }

    #[cfg(target_arch = "aarch64")]
    pub(super) fn setup_vcpus_aarch64(&self, vm: &kvm::KtstrKvm, kernel_entry: u64) -> Result<()> {
        let t0 = Instant::now();
        let memory_mib = self.effective_memory_mib(&vm.guest_mem);
        let fdt_addr = aarch64::fdt::fdt_address(memory_mib);
        boot::setup_regs(&vm.vcpus[0], kernel_entry, fdt_addr)?;
        tracing::debug!(elapsed_us = t0.elapsed().as_micros(), "bsp_setup");
        // APs start powered off via PSCI — no register setup needed.
        Ok(())
    }
}

#[cfg(test)]
mod tests;
