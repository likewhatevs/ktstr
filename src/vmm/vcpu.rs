//! vCPU thread infrastructure: signal-based kicks, immediate_exit
//! handles, affinity / RT scheduling helpers, and the freeze
//! coordinator's per-AP state.
//!
//! Each vCPU runs on its own host thread inside `KVM_RUN`. Kicking a
//! vCPU out of guest mode requires (a) writing the
//! `kvm_run.immediate_exit` byte from outside the thread (the
//! Firecracker pattern) and (b) sending the dedicated `SIGRTMIN`
//! signal so the in-progress ioctl returns `EINTR`. This module owns
//! the cross-thread handle ([`ImmediateExitHandle`]), the signal
//! handler registration, and the `VcpuThread` struct used by the run
//! orchestrator.
//!
//! Affinity helpers ([`pin_current_thread`], [`set_thread_cpumask`])
//! and RT priority ([`set_rt_priority`]) live here too — they're
//! shared between the BSP / AP run loops.

use std::ops::Deref;
#[cfg(test)]
use std::os::unix::io::AsRawFd;
use std::os::unix::thread::JoinHandleExt;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicPtr, AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Weak};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use vmm_sys_util::eventfd::{EFD_NONBLOCK, EventFd};
#[cfg(test)]
use vmm_sys_util::{
    epoll::{ControlOperation, Epoll, EpollEvent, EventSet},
    timerfd::TimerFd,
};

use super::exit_dispatch;
use super::result::HostVcpuSchedstat;
use crate::monitor;
use crate::sync::Latch;

// ---------------------------------------------------------------------------
// ImmediateExitHandle — cross-thread access to kvm_run.immediate_exit
// ---------------------------------------------------------------------------

/// Handle for setting the `immediate_exit` field in a vCPU's mmap'd `kvm_run`
/// struct from outside the vCPU thread.
///
/// The `kvm_run` page is `MAP_SHARED` between kernel and userspace; the
/// `immediate_exit` field is a single byte read by KVM atomically before
/// entering `KVM_RUN`. Setting it to 1 causes the next `KVM_RUN` to return
/// immediately with `EINTR`.
///
/// Each handle holds a Weak reference to an [`ImmediateExitState`] owned by the
/// run-loop's [`ImmediateExitVcpu`]. A set/read upgrades it to a transient Arc,
/// which pins the state's second MAP_SHARED `kvm_run` mapping for the complete
/// byte access. Linux VMAs retain a file reference, so an in-flight access
/// remains valid even if the run-loop completes and its `VcpuFd` is returned
/// or dropped concurrently. Idle stale handles do not prolong the VM's
/// lifetime. The extra VMA maps the same physical run pages and costs only
/// VMA/page-table metadata.
///
/// This follows the KVM API's offset-zero vCPU mapping contract and Linux's
/// `kvm_vcpu_mmap` implementation: each VMA maps the same `vcpu->run` pages,
/// the VMA's file reference delays `kvm_vcpu_release`, and KVM samples
/// `immediate_exit` with `READ_ONCE`.
///
/// [`ImmediateExitVcpu`] also publishes an active flag before dropping its
/// strong Arc. That flag is deliberately not the memory-safety proof: a set
/// may upgrade, observe `true`, lose a race with owner completion, and safely
/// finish through its transient Arc mapping. Later upgrades fail and no-op.
///
/// Handles cannot be constructed from a bare `&mut VcpuFd`. The only
/// constructor is [`ImmediateExitVcpu::new`], which consumes the fd and
/// requires the owning VM's kernel-reported vCPU mmap size.
#[derive(Clone)]
pub(crate) struct ImmediateExitHandle {
    state: Weak<ImmediateExitState>,
}

struct ImmediateExitState {
    ptr: *mut u8,
    active: AtomicBool,
    // Owns the independent VMA addressed by `ptr`. Construct the pointer once
    // while this mapping is uniquely owned, then never expose or dereference
    // the wrapper again: every Rust byte access goes through AtomicU8 below.
    _mapping: kvm_ioctls::KvmRunWrapper,
}

// SAFETY: `_mapping` owns the offset-zero vCPU VMA containing `ptr` for the
// state's entire lifetime. Linux retains the vCPU file ref for that VMA, so
// kvm_vcpu_release cannot retire `vcpu->run` before `_mapping` unmaps it.
// Every Rust access through `ptr` uses AtomicU8, matching KVM's READ_ONCE
// access. The mapping cannot move its virtual address and is not exposed after
// the single construction-time `as_mut_ref` used to extract `ptr`.
unsafe impl Send for ImmediateExitState {}
unsafe impl Sync for ImmediateExitState {}

impl ImmediateExitHandle {
    /// Set `immediate_exit` to the given value. Returns `false` without
    /// touching the pointer if the run-loop owner has completed.
    pub(crate) fn set(&self, val: u8) -> bool {
        let Some(state) = self.state.upgrade() else {
            return false;
        };
        if !state.active.load(Ordering::Acquire) {
            return false;
        }
        // SAFETY: the transient `state` Arc pins `_mapping`, including across
        // a race where the active load above returns true just before owner
        // completion. AtomicU8 has size/alignment 1, matching the ABI field.
        unsafe { AtomicU8::from_ptr(state.ptr).store(val, Ordering::Release) };
        true
    }

    #[cfg(test)]
    pub(crate) fn read_byte(&self) -> Option<u8> {
        let state = self.state.upgrade()?;
        if !state.active.load(Ordering::Acquire) {
            return None;
        }
        // SAFETY: identical structural mapping-lifetime argument to `set`.
        Some(unsafe { AtomicU8::from_ptr(state.ptr).load(Ordering::Acquire) })
    }

    #[cfg(test)]
    fn set_after_test_barrier(&self, val: u8, acquired: &Latch, release: &Latch) -> Option<u8> {
        let state = self.state.upgrade()?;
        if !state.active.load(Ordering::Acquire) {
            return None;
        }
        acquired.set();
        release.wait();
        // SAFETY: the barrier deliberately allows owner completion and the
        // original VcpuFd mapping's destruction before this store. This
        // transient Arc still structurally owns the independent mapping.
        let byte = unsafe { AtomicU8::from_ptr(state.ptr) };
        byte.store(val, Ordering::Release);
        Some(byte.load(Ordering::Acquire))
    }
}

/// Owns the run-loop `VcpuFd` and the logical-active reference for every
/// [`ImmediateExitHandle`] clone.
///
/// Drop and [`Self::into_inner`] publish inactive before destroying or
/// returning the fd. Memory safety does not depend on that ordering: the
/// in-flight handle access owns a transient Arc to the shared mapping until
/// that access completes.
pub(crate) struct ImmediateExitVcpu {
    vcpu: Option<kvm_ioctls::VcpuFd>,
    state: Option<Arc<ImmediateExitState>>,
}

impl ImmediateExitVcpu {
    /// Consume one vCPU fd and, when `enabled`, create an independently-owned
    /// shared mapping for cross-thread access to its immediate-exit byte.
    pub(crate) fn new(
        vcpu: kvm_ioctls::VcpuFd,
        enabled: bool,
        run_size: usize,
    ) -> vmm_sys_util::errno::Result<(Self, Option<ImmediateExitHandle>)> {
        let state = if enabled {
            let mut mapping = kvm_ioctls::KvmRunWrapper::mmap_from_fd(&vcpu, run_size)?;
            let ptr: *mut u8 = &mut mapping.as_mut_ref().immediate_exit;
            Some(Arc::new(ImmediateExitState {
                ptr,
                active: AtomicBool::new(true),
                _mapping: mapping,
            }))
        } else {
            None
        };
        let handle = state.as_ref().map(|state| ImmediateExitHandle {
            state: Arc::downgrade(state),
        });
        Ok((
            Self {
                vcpu: Some(vcpu),
                state,
            },
            handle,
        ))
    }

    fn deactivate(&mut self) {
        if let Some(state) = self.state.take() {
            state.active.store(false, Ordering::Release);
        }
    }

    /// Enter KVM through the fd paired with this wrapper's immediate-exit
    /// mapping. Keeping this operation on the wrapper prevents safe code from
    /// replacing the inner fd while handles still name the original vCPU.
    pub(crate) fn run(&mut self) -> vmm_sys_util::errno::Result<kvm_ioctls::VcpuExit<'_>> {
        self.vcpu
            .as_mut()
            .expect("ImmediateExitVcpu owns its VcpuFd")
            .run()
    }

    /// Atomically update the run-loop mapping's immediate-exit byte.
    pub(crate) fn set_immediate_exit(&mut self, val: u8) {
        set_vcpu_immediate_exit(
            self.vcpu
                .as_mut()
                .expect("ImmediateExitVcpu owns its VcpuFd"),
            val,
        );
    }

    #[cfg(test)]
    fn read_immediate_exit(&mut self) -> u8 {
        let ptr: *mut u8 = &mut self
            .vcpu
            .as_mut()
            .expect("ImmediateExitVcpu owns its VcpuFd")
            .get_kvm_run()
            .immediate_exit;
        // SAFETY: the wrapper owns the primary kvm_run mapping for the
        // complete load and AtomicU8 matches the one-byte ABI field.
        unsafe { AtomicU8::from_ptr(ptr).load(Ordering::Acquire) }
    }

    /// Mark every handle inactive, drop the owner's strong mapping reference,
    /// then return the still-live raw fd. Any access that already upgraded its
    /// Weak reference keeps the mapping pinned through completion; later
    /// upgrades fail and no-op.
    pub(crate) fn into_inner(mut self) -> kvm_ioctls::VcpuFd {
        self.deactivate();
        self.vcpu.take().expect("ImmediateExitVcpu owns its VcpuFd")
    }
}

// Shared-reference ioctl methods remain available through Deref. There is
// intentionally no DerefMut/AsMut implementation: safe code must not
// `mem::replace` the fd while the secondary mapping and its Weak handles still
// identify the original vCPU. Every mutable operation is forwarded explicitly
// above so the fd/mapping pairing remains structural.
impl Deref for ImmediateExitVcpu {
    type Target = kvm_ioctls::VcpuFd;

    fn deref(&self) -> &Self::Target {
        self.vcpu
            .as_ref()
            .expect("ImmediateExitVcpu owns its VcpuFd")
    }
}

impl Drop for ImmediateExitVcpu {
    fn drop(&mut self) {
        self.deactivate();
    }
}

/// Atomically update `kvm_run.immediate_exit` from the owning vCPU thread.
/// Cross-thread handles use the same `AtomicU8` representation, avoiding a
/// mixed atomic/non-atomic Rust data race on the shared byte.
fn set_vcpu_immediate_exit(vcpu: &mut kvm_ioctls::VcpuFd, val: u8) {
    let ptr: *mut u8 = &mut vcpu.get_kvm_run().immediate_exit;
    // SAFETY: AtomicU8 has size/alignment 1 and `ptr` names a live u8 field in
    // the caller-owned kvm_run mapping.
    unsafe { AtomicU8::from_ptr(ptr).store(val, Ordering::Release) };
}

// ---------------------------------------------------------------------------
// Signal handling — Firecracker/libkrun pattern: SIGRTMIN + immediate_exit
// ---------------------------------------------------------------------------

/// Convert a host-side `Duration` to guest jiffies, using the
/// guest kernel's CONFIG_HZ.
///
/// Computed as `(d.as_millis() * hz) / 1000` rather than
/// `d.as_secs() * hz` so sub-second durations don't truncate to 0
/// — a 500 ms watchdog with HZ=1000 should land at 500 jiffies, not
/// at 0 (the bug that masked the early-trigger path before this
/// helper existed). Truncation is to the jiffies tick boundary
/// (1000/HZ ms), which is the kernel's own arithmetic precision.
///
/// Two call sites today: the freeze coordinator's
/// `half_threshold_jiffies` (compares against scanned per-task
/// runnable-age in jiffies) and the `watchdog_override` setup
/// (writes a jiffies count into `scx_sched.watchdog_timeout` in
/// guest memory). Both pre-existed scattered as inline expressions;
/// centralising the conversion keeps the precision rule in one
/// place and eliminates drift opportunities.
pub(crate) fn duration_to_jiffies(d: Duration, hz: u64) -> u64 {
    // saturating_mul guards against the theoretical overflow of
    // pathologically-large `Duration` * pathologically-large `hz`.
    // Real ktstr inputs (watchdog_timeout in seconds, HZ in 100..1000)
    // never approach the u64 boundary, but a `Duration::MAX` /
    // `u64::MAX` HZ pair would otherwise wrap and silently produce a
    // garbage jiffies value. Saturating to u64::MAX (then `/ 1000`)
    // at least keeps the threshold check semantics "this jiffies
    // count is unreachable" rather than "this jiffies count is small,
    // so the trigger fires immediately".
    (d.as_millis() as u64).saturating_mul(hz) / 1000
}

/// Signal used to kick vCPU threads out of KVM_RUN.
/// All three Rust reference VMMs (Firecracker, Cloud Hypervisor, libkrun)
/// use SIGRTMIN. SIGUSR1/SIGUSR2 conflict with application-level signals.
pub(crate) fn vcpu_signal() -> libc::c_int {
    libc::SIGRTMIN()
}

/// Resolve the byte offset of `ktstr_err_exit_detected` within the
/// probe BPF program's `.bss` section by walking the program's BTF
/// Datasec. Returns `None` when any step fails (program BTF not yet
/// loaded, struct btf untranslatable, blob short-read, BTF parse
/// reject, no matching VarSecinfo).
///
/// `btf_kva` is the kernel KVA of the probe map's `struct btf`;
/// `base` is the host's parsed vmlinux BTF used as the split-BTF
/// base when the program BTF is split. Lives next to
/// [`vcpu_signal`] because the freeze coordinator is the sole
/// consumer.
pub(crate) fn load_probe_bss_offset(
    kernel: &crate::monitor::guest::GuestKernel,
    btf_kva: u64,
    base: &btf_rs::Btf,
    offsets: &crate::monitor::btf_offsets::BpfMapOffsets,
) -> Option<u32> {
    let mem = kernel.mem();
    let walk = kernel.walk_context();
    let btf_pa = crate::monitor::idr::translate_any_kva(
        mem,
        walk.cr3_pa,
        walk.page_offset,
        btf_kva,
        walk.l5,
        walk.tcr_el1,
    )?;
    let data_kva = mem.read_u64(btf_pa, offsets.btf_data);
    let data_size = mem.read_u32(btf_pa, offsets.btf_data_size) as usize;
    let base_kva = mem.read_u64(btf_pa, offsets.btf_base_btf);
    if data_kva == 0 || data_size == 0 {
        return None;
    }
    if data_size > crate::monitor::dump::MAX_BTF_BLOB {
        return None;
    }
    // The chunked vmalloc reader handles per-page translate + bulk
    // copy and honours all-or-nothing on short reads — the previous
    // hand-rolled loop here duplicated `GuestKernel::read_kva_bytes_chunked`
    // for no benefit.
    let blob = kernel.read_kva_bytes_chunked(data_kva, data_size)?;
    let prog_btf = if base_kva != 0 {
        btf_rs::Btf::from_split_bytes(&blob, base).ok()?
    } else {
        btf_rs::Btf::from_bytes(&blob).ok()?
    };
    crate::monitor::btf_offsets::resolve_var_offset_in_section(
        &prog_btf,
        ".bss",
        "ktstr_err_exit_detected",
    )
}

/// Signal handler — Firecracker pattern.
/// The handler itself is a no-op; its sole purpose is to cause KVM_RUN
/// to return with EINTR. The fence ensures that a write to
/// `kvm_run.immediate_exit` from another thread (via ImmediateExitHandle)
/// is visible when KVM_RUN returns. This Acquire fence pairs with the
/// proximal `Ordering::Release` fence in each cross-thread kicker:
///
/// - [`super::freeze_coord`]'s freeze coordinator fences between pass 1
///   (writing every `kvm_run.immediate_exit`) and pass 2 (delivering SIGRTMIN);
/// - the interactive relay fences after its BSP immediate-exit write and before
///   each SIGRTMIN retry.
///
/// Each Release fence publishes the immediate-exit byte before its signal is
/// delivered; the Acquire fence here runs in the receiving vCPU thread.
/// Without the pair, a vCPU could process its signal, re-enter KVM_RUN, and
/// miss the byte that was supposed to short-circuit guest entry.
extern "C" fn vcpu_signal_handler(_: libc::c_int, _: *mut libc::siginfo_t, _: *mut libc::c_void) {
    std::sync::atomic::fence(Ordering::Acquire);
}

/// Register the vCPU signal handler and unblock the signal in this thread.
/// Must be called from each vCPU thread before entering the run loop.
/// Follows Firecracker's register_kick_signal_handler + QEMU's
/// kvm_init_cpu_signals: register SA_SIGINFO handler, then unblock via
/// pthread_sigmask so the signal is deliverable inside KVM_RUN.
///
/// # Panics
///
/// Panics if `libc::sigaction` or `libc::pthread_sigmask` returns
/// non-zero. Both calls are infallible for the SIGRTMIN argument we
/// pass on every supported kernel (the signum is reserved by glibc
/// for application use, the `SA_SIGINFO` handler shape is universally
/// accepted, and `SIG_UNBLOCK` with a single-signal set has no error
/// path beyond "invalid signum"). Silent failure here would leave the
/// vCPU thread unable to break out of `KVM_RUN` on `SIGRTMIN` — every
/// `VcpuThread::kick()` call becomes a no-op and the thread blocks
/// forever, with no diagnostic. Panicking surfaces the broken
/// invariant the moment it occurs and routes through the panic hook
/// that ships a crash diagnostic to COM2 before reboot. Mirrors the
/// `SigchldDispositionGuard::install` discipline in
/// `crate::vmm::rust_init`.
pub(crate) fn register_vcpu_signal_handler() {
    unsafe {
        let mut sa: libc::sigaction = std::mem::zeroed();
        sa.sa_sigaction = vcpu_signal_handler as *const () as usize;
        sa.sa_flags = libc::SA_SIGINFO;
        libc::sigemptyset(&mut sa.sa_mask);
        let rc = libc::sigaction(vcpu_signal(), &sa, std::ptr::null_mut());
        assert_eq!(
            rc,
            0,
            "register_vcpu_signal_handler: sigaction(SIGRTMIN, SA_SIGINFO) failed: {} \
             — vCPU kicks would silently no-op and KVM_RUN would block forever",
            std::io::Error::last_os_error(),
        );

        // Unblock the signal in this thread so pthread_kill can deliver it.
        let mut set: libc::sigset_t = std::mem::zeroed();
        libc::sigemptyset(&mut set);
        libc::sigaddset(&mut set, vcpu_signal());
        let rc = libc::pthread_sigmask(libc::SIG_UNBLOCK, &set, std::ptr::null_mut());
        assert_eq!(
            rc,
            0,
            "register_vcpu_signal_handler: pthread_sigmask(SIG_UNBLOCK, SIGRTMIN) failed: {} \
             — signal would stay blocked and pthread_kick deliveries would queue forever",
            std::io::Error::from_raw_os_error(rc),
        );
    }
}

// ---------------------------------------------------------------------------
// vCPU affinity
// ---------------------------------------------------------------------------

/// Pin the calling thread to a single host CPU via sched_setaffinity(0, ...).
/// Logs success (debug-gated) or warning; does not fail the VM.
pub(crate) fn pin_current_thread(cpu: usize, label: &str) {
    let mut cpuset = nix::sched::CpuSet::new();
    if let Err(e) = cpuset.set(cpu) {
        let (width, online) = cpu_set_diag_context();
        eprintln!(
            "performance_mode: WARNING: cpuset.set({cpu}) for {label}: {e}. \
             cpu_set bitmap holds CPU IDs 0..{width} (libc CPU_SETSIZE) and \
             host reports {online} online CPUs (sysconf _SC_NPROCESSORS_ONLN). \
             Either the cpuset spec was resolved against a stale topology or \
             the bitmap cap needs raising on this build."
        );
        return;
    }
    match nix::sched::sched_setaffinity(nix::unistd::Pid::from_raw(0), &cpuset) {
        Ok(()) => {
            // Success is per-vCPU chatter (N lines per VM); route it through
            // the debug gate like the other per-vCPU diagnostics. Failure
            // stays unconditional — it explains degraded timing.
            if crate::vmm::debug_logging_enabled() {
                eprintln!("performance_mode: pinned {label} to host CPU {cpu}");
            }
        }
        Err(e) => eprintln!("performance_mode: WARNING: pin {label} to CPU {cpu}: {e}"),
    }
}

/// Snapshot the host's CPU-set bitmap width (libc `CPU_SETSIZE`) and
/// the online-CPU count (sysconf `_SC_NPROCESSORS_ONLN`) for use in
/// rich `cpuset.set` failure diagnostics. Renders sysconf failure as
/// `"unavailable"` rather than `0` so a degenerate sysconf result is
/// disambiguated from a legitimately zero count.
fn cpu_set_diag_context() -> (usize, std::borrow::Cow<'static, str>) {
    let width = libc::CPU_SETSIZE as usize;
    let online: std::borrow::Cow<'static, str> =
        match nix::unistd::sysconf(nix::unistd::SysconfVar::_NPROCESSORS_ONLN)
            .ok()
            .flatten()
        {
            Some(n) => format!("{n}").into(),
            None => "unavailable".into(),
        };
    (width, online)
}

/// Set the calling thread's CPU mask to the supplied set. Distinct
/// from [`pin_current_thread`]: that one locks a thread to a single
/// CPU (the perf-mode contract), this one constrains a thread to a
/// pool without picking a specific CPU. The kernel picks a runnable
/// CPU from the mask.
///
/// Used by the no-perf + `--cpu-cap` path at
/// [`crate::vmm::KtstrVmBuilder::build`]: every vCPU thread gets the reserved
/// LLC's CPUs as its mask so the vCPU runs inside the resource
/// budget without fighting the kernel scheduler for a hard pin it
/// doesn't actually need.
///
/// Logs success or warning; does not fail the VM.
///
/// Best-effort partial-mask semantics: a single bad CPU (out of
/// `CpuSet`'s static bitmap range) does NOT abort the whole call.
/// The bad entry is logged and skipped, and the resulting mask
/// reflects every CPU that fit. This is preferable to the
/// alternative — silently inheriting whatever overly-narrow mask
/// the thread already had (often a single-CPU perf-mode pin) and
/// quietly losing the broadening the caller asked for. The only
/// case that early-returns is "every requested CPU was rejected,"
/// which would otherwise call `sched_setaffinity` with an empty
/// mask and block the thread forever.
///
/// `pub(crate)` so the sibling vmm modules (the BSP/AP run loops in
/// `freeze_coord`, the shell-mode BSP in `vmm::mod`, and the
/// virtio-blk worker) can share the same affinity primitive.
pub(crate) fn set_thread_cpumask(cpus: &[usize], label: &str) {
    // Build the cpuset by adding every CPU we can. A bad CPU
    // (out-of-range for `CpuSet`'s static bitmap, currently 1024 on
    // x86_64) skips that single entry and continues the loop rather
    // than aborting the whole call. The early-return form gave us
    // the worst of both worlds: the thread inherited whatever
    // overly-narrow mask was already in place (e.g. a single-CPU
    // perf-mode pin) and the caller silently lost the broadening
    // it asked for. A partial mask — every CPU that fit, minus the
    // bad one — preserves most of the intent and remains observable
    // via the per-skip warning + the post-loop summary.
    let mut cpuset = nix::sched::CpuSet::new();
    let mut applied: Vec<usize> = Vec::with_capacity(cpus.len());
    let mut skipped: Vec<usize> = Vec::new();
    // Snapshot the bitmap width + online-CPU count once before the
    // loop so a per-CPU diagnostic on overflow carries both numbers
    // without re-syscalling sysconf on every failing iteration.
    let (cpuset_bitmap_width, online_cpus_str) = cpu_set_diag_context();
    for &cpu in cpus {
        match cpuset.set(cpu) {
            Ok(()) => applied.push(cpu),
            Err(e) => {
                eprintln!(
                    "no_perf_mode: WARNING: cpuset.set({cpu}) for {label}: {e}. \
                     cpu_set bitmap holds CPU IDs 0..{cpuset_bitmap_width} \
                     (libc CPU_SETSIZE) and host reports {online_cpus_str} \
                     online CPUs (sysconf _SC_NPROCESSORS_ONLN). Either the \
                     cpuset spec was resolved against a stale topology or \
                     the bitmap cap needs raising on this build. Skipping {cpu}."
                );
                skipped.push(cpu);
            }
        }
    }
    if !skipped.is_empty() {
        eprintln!(
            "no_perf_mode: {label}: skipped {} of {} requested CPUs ({skipped:?}); proceeding with {applied:?}",
            skipped.len(),
            cpus.len(),
        );
    }
    // If every requested CPU failed to bind we have nothing to apply
    // — calling sched_setaffinity with an empty mask would block the
    // thread forever. Bail rather than mask to zero.
    if applied.is_empty() {
        eprintln!(
            "no_perf_mode: WARNING: {label}: no valid CPUs to mask (every requested entry failed)"
        );
        return;
    }
    let n = applied.len();
    // Range-collapse the CPU list so contiguous spans render as
    // "a-b" and non-contiguous CPUs render with explicit
    // commas: [0,1,2,5,7,8] → "0-2,5,7-8". A bare min-max range
    // ("0-8") would be misleading when CPUs 3, 4, 6 are excluded.
    // `applied` is sorted by construction in the loop above
    // (each `cpu` is pushed in iteration order from a sorted
    // `cpus` slice).
    let cpu_list_str = {
        let mut parts: Vec<String> = Vec::new();
        let mut start = applied[0];
        let mut end = applied[0];
        for &cpu in &applied[1..] {
            if cpu == end + 1 {
                end = cpu;
            } else {
                if start == end {
                    parts.push(format!("{start}"));
                } else {
                    parts.push(format!("{start}-{end}"));
                }
                start = cpu;
                end = cpu;
            }
        }
        if start == end {
            parts.push(format!("{start}"));
        } else {
            parts.push(format!("{start}-{end}"));
        }
        parts.join(",")
    };
    match nix::sched::sched_setaffinity(nix::unistd::Pid::from_raw(0), &cpuset) {
        Ok(()) => {
            if crate::vmm::debug_logging_enabled() {
                eprintln!("no_perf_mode: mask {label} to {n} CPUs ({cpu_list_str})")
            }
        }
        Err(e) => {
            eprintln!("no_perf_mode: WARNING: mask {label} to {n} CPUs ({cpu_list_str}): {e}")
        }
    }
}

/// Set the calling thread to SCHED_FIFO at the given priority.
/// Logs success or warning via tracing; does not fail the VM.
///
/// Callers: vCPU / BSP threads (FIFO-1) only under `performance_mode`,
/// and the watchdog + monitor service threads (FIFO-2) UNCONDITIONALLY
/// (their sensing must not dilate with the load it measures — see the
/// freeze-coordinator spawn sites). So this is no longer a perf-mode-only
/// path, and the messages no longer say so.
///
/// Uses `tracing::info!` / `tracing::warn!` rather than `eprintln!`
/// so the warn-without-CAP_SYS_NICE branch is observable by tests
/// that install a tracing subscriber (e.g. `tracing-test`).
/// Previously `eprintln!` made the warning invisible to any test
/// that didn't fork + redirect fd 2.
///
/// The failure warning is emitted at most ONCE per process: the two
/// service threads now call this on every VM cell, and in
/// `performance_mode` every vCPU calls it too, so on a host without
/// CAP_SYS_NICE a single cell would otherwise emit N+2 identical warns —
/// and a CI fleet running one cell per process would drown in them. The
/// capability failure is uniform for the whole process, so the first
/// warn tells the operator everything the rest would repeat.
pub(crate) fn set_rt_priority(priority: i32, label: &str) {
    let param = libc::sched_param {
        sched_priority: priority,
    };
    let rc = unsafe { libc::sched_setscheduler(0, libc::SCHED_FIFO, &param) };
    if rc == 0 {
        tracing::info!(
            label = label,
            priority = priority,
            "{label} set to SCHED_FIFO priority {priority}",
        );
    } else {
        let err = std::io::Error::last_os_error();
        static WARN_ONCE: std::sync::Once = std::sync::Once::new();
        WARN_ONCE.call_once(|| {
            tracing::warn!(
                label = label,
                priority = priority,
                err = %err,
                "WARNING: SCHED_FIFO for {label}: {err} (need CAP_SYS_NICE)",
            );
        });
    }
}

/// Wait for every vCPU thread's TID to publish into its slot, then
/// open per-vCPU `perf_event_open` counters bound to those TIDs. The
/// returned [`monitor::perf_counters::PerfCountersCapture`] is shared
/// (via `Arc`) by the monitor sampling loop and the freeze
/// coordinator so the per-tick timeline and the freeze-instant
/// snapshot read through the same fds — opening twice would burn
/// twice the perf slots and produce two slightly-different time
/// bases.
///
/// `vcpu_tid_slots[i]` pairs the AP-thread-published TID for vCPU
/// `i` with a [`Latch`] the producer fires after storing the TID
/// (0 = BSP, written synchronously before this function runs and
/// shipped with a pre-set latch). The function blocks each slot's
/// latch with a shared 1 s deadline instead of sleep-polling the
/// `AtomicI32`. Any slot still 0 at the deadline is treated as "no
/// perf data for that vCPU"; the whole capture returns `None` so
/// the timeline + freeze paths consume `Option::as_ref()` and emit
/// `None` per-CPU.
///
/// Failure paths (perf_event_paranoid too high, missing
/// CAP_PERFMON, hardware lacks the requested counter) log a warning
/// via `tracing::warn!` and return `None`. The dump pipeline still
/// runs without per-vCPU perf data.
pub(crate) fn open_vcpu_perf_capture(
    vcpu_tid_slots: &[(Arc<AtomicI32>, Arc<Latch>)],
) -> Option<monitor::perf_counters::PerfCountersCapture> {
    let overall_deadline = Instant::now() + Duration::from_secs(1);
    let mut tids: Vec<libc::pid_t> = Vec::with_capacity(vcpu_tid_slots.len());
    for (slot, latch) in vcpu_tid_slots {
        // Block until the AP publishes its TID (or the deadline
        // elapses). The producer side stores the TID with `Release`
        // ordering before calling `Latch::set`, so a successful
        // `Latch::wait_timeout` happens-before the `slot.load`
        // observes the published value.
        let now = Instant::now();
        let remaining = overall_deadline.saturating_duration_since(now);
        if remaining.is_zero() {
            tids.push(slot.load(Ordering::Acquire));
            continue;
        }
        latch.wait_timeout(remaining);
        tids.push(slot.load(Ordering::Acquire));
    }
    if !tids.iter().all(|&t| t > 0) {
        let missing: Vec<usize> = tids
            .iter()
            .enumerate()
            .filter_map(|(i, &t)| (t == 0).then_some(i))
            .collect();
        tracing::warn!(
            ?missing,
            "vCPU TID slots never published; per-vCPU perf capture disabled"
        );
        return None;
    }
    match monitor::perf_counters::PerfCountersCapture::open(&tids) {
        Ok(cap) => Some(cap),
        Err(e) => {
            tracing::warn!(
                err = %e,
                "perf_event_open failed; per-vCPU IPC/cache-miss capture disabled"
            );
            None
        }
    }
}

// ---------------------------------------------------------------------------
// VcpuThread — Cloud Hypervisor pattern with Firecracker's immediate_exit
// ---------------------------------------------------------------------------

/// Per-vCPU thread lifecycle and targeting state shared by the coordinator and
/// bounded AP teardown.
pub(crate) struct VcpuThread {
    pub(crate) handle: JoinHandle<kvm_ioctls::VcpuFd>,
    /// Test-only mirror set after the thread exits the KVM_RUN loop.
    /// Production teardown uses `JoinHandle::is_finished()` because the panic
    /// hook may publish this edge before stack unwinding has completed.
    #[cfg(test)]
    pub(crate) exited: Arc<AtomicBool>,
    /// Handle to set `kvm_run.immediate_exit` from outside the vCPU thread.
    /// `None` when KVM_CAP_IMMEDIATE_EXIT is not available.
    pub(crate) immediate_exit: Option<ImmediateExitHandle>,
    /// Eventfd bumped after the run-loop exit edge. Production
    /// `freeze_coord::kick_and_join_ap_threads` epoll-waits on it before
    /// re-checking `JoinHandle::is_finished()`; the test-only
    /// `VcpuThread::wait_for_exit` helper exercises the same wake edge. The
    /// panic hook also signals it, which wakes teardown promptly without
    /// claiming unwind has finished. Counter mode (not semaphore): the value
    /// is unused; only the transition from zero to non-zero matters.
    pub(crate) exit_evt: Arc<EventFd>,
    /// Logical participation flag for cross-thread AP kick loops.
    /// Initialised to `true` at spawn and flipped to `false` by the AP panic
    /// hook before unwinding. Readers use it to avoid work after that edge,
    /// but it is not an mmap-lifetime proof: a `true` load cannot pin memory
    /// across a later pointer access. [`ImmediateExitHandle`] supplies its
    /// own lifetime synchronization for every actual byte access.
    pub(crate) alive: Arc<AtomicBool>,
    /// Final cumulative schedstat sample taken by the AP on its own thread
    /// immediately before exit. `/proc/self/task/<tid>` disappears when the
    /// thread returns, so teardown consumers use this slot only when their
    /// live proc read races that disappearance. One snapshot per vCPU
    /// lifetime; never sampled from a timer path.
    pub(crate) schedstat_at_exit: Arc<std::sync::Mutex<Option<HostVcpuSchedstat>>>,
}

/// Per-AP freeze-rendezvous state held outside `VcpuThread`. Cloned
/// out of `spawn_ap_threads` and into the freeze coordinator at run
/// startup; not needed for teardown (kick/join), so it lives apart
/// from `VcpuThread` to keep that struct minimal.
pub(crate) struct ApFreezeHandles {
    /// Per-AP `parked` ack flags. Set by the AP thread when it has
    /// completed the freeze drain dance and is parked, awaiting
    /// clearance to resume. The freeze coordinator polls each entry
    /// with Acquire ordering before reading guest memory; the
    /// thread's prior Release store synchronizes-with that load,
    /// providing the happens-before edge that makes host-side
    /// guest-memory reads consistent on weakly-ordered
    /// architectures.
    pub(crate) parked: Vec<Arc<AtomicBool>>,
    /// Per-AP register-snapshot slots captured at freeze time
    /// (RIP/RSP/CR3 on x86_64, PC/SP/TTBR1+TTBR0 on aarch64). Written
    /// by the AP thread on its own thread (KVM_GET_REGS is fd-bound
    /// and not safe cross-thread) just before the `parked` Release
    /// store; read by the freeze coordinator after the rendezvous
    /// Acquire. `None` until the first freeze fires; reset to
    /// `None` on thaw is NOT done — a successive freeze overwrites
    /// with the new capture.
    pub(crate) regs: Vec<Arc<std::sync::Mutex<Option<exit_dispatch::VcpuRegSnapshot>>>>,
}

/// Shared watchpoint-arming and hit-detection state for the
/// failure-dump freeze trigger.
///
/// Adds a hardware data-write watchpoint on `*scx_root->exit_kind`
/// (the kernel's authoritative SCX_EXIT_* latch) as the primary
/// late-trigger signal, alongside the existing
/// `ktstr_err_exit_detected` BPF .bss poll which remains active as
/// fallback. The freeze coordinator resolves the field's KVA lazily
/// (after `*scx_root` becomes non-NULL) and publishes it via
/// [`Self::request_kva`]; each vCPU thread polls this slot before
/// each `KVM_RUN` and self-arms via [`self_arm_watchpoint`], which
/// emits the appropriate per-arch `KVM_SET_GUEST_DEBUG` payload:
///
///   - x86_64: `debugreg[0] = exit_kind_kva` and `debugreg[7]`
///     configured for "trap on 4-byte writes" (DR7 control bits
///     `0x000D0703` = bit 10 reserved-1, bits 0-1 L0+G0 enable,
///     bits 8-9 LE+GE exact, bits 16-17 R/W0 = write-only,
///     bits 18-19 LEN0 = 4-byte).
///   - aarch64: `dbg_wvr[0] = exit_kind_kva & ~0x7` (8-byte
///     aligned base) and `dbg_wcr[0]` configured for "trap on
///     4-byte writes" (E=1, PAC=0b11 EL0+EL1, LSC=0b10 write-only,
///     BAS = 0xF shifted by `kva & 0x7` for 4-byte selection).
///     The aarch64 watchpoint trap is taken BEFORE the offending
///     store retires (ARM ARM D2.10.5), so after a fire the run
///     loop transitions WCR.E to 0 on the fired slot AND asserts
///     `KVM_GUESTDBG_SINGLESTEP` for one KVM_RUN; the next
///     `KVM_EXIT_DEBUG` carries `EC=ESR_ELx_EC_SOFTSTP_LOW (0x32)`
///     which signals "the watched store retired, the slot may be
///     re-armed" — the loop then drops `KVM_GUESTDBG_SINGLESTEP`
///     and restores WCR.E=1. Without this dance KVM_RUN replays
///     the same store and re-trips the watchpoint forever.
///
/// Once armed, a guest store to the field traps via
/// `KVM_EXIT_DEBUG`; the dispatch path sets [`Self::hit`], which
/// the freeze coordinator polls alongside the BPF .bss latch.
///
/// Why a hardware watchpoint: the BPF .bss poll requires a full
/// guest-memory page-walk every 250 ms iteration AND a parallel BPF
/// program writing the latch. The watchpoint is delivered
/// synchronously by hardware the instant the kernel sets `exit_kind`
/// (e.g. `kernel/sched/ext.c` `scx_exit` path), with no host-side
/// polling overhead and no dependency on the probe BPF program being
/// loaded. It also fires on ANY exit_kind transition — including
/// SCX_EXIT_ERROR_BPF / SCX_EXIT_ERROR_STALL paths the .bss probe might miss
/// when its selected hook ran before kernel teardown.
/// The .bss path remains because the watchpoint can be unavailable
/// (no `scx_root` symbol on pre-6.16, BTF stripped of `scx_sched`,
/// or `KVM_SET_GUEST_DEBUG` rejected by the host).
pub(crate) struct WatchpointArm {
    /// KVA the freeze coordinator wants armed in slot 0
    /// (`debugreg[0]` on x86_64, `dbg_wvr[0]`/`dbg_wcr[0]` on
    /// aarch64). `0` means "no arm requested yet" — the coordinator
    /// publishes this once it has resolved
    /// `*scx_root + exit_kind_offset`. After publication the value
    /// is monotonic for the VM run (the kernel scx_sched lifetime
    /// spans every err_exit transition we care about).
    pub(crate) request_kva: AtomicU64,
    /// Host pointer to the same `exit_kind` field. Published by the
    /// coordinator alongside `request_kva` so the vCPU thread can
    /// `read_volatile` the post-store value at fire time without
    /// needing its own `GuestMem` plumbing. `null_mut` until the
    /// coordinator publishes; valid for the VM lifetime once set
    /// (the underlying guest-DRAM page is mapped through
    /// `vm.guest_mem`, which is dropped only by `collect_results`
    /// AFTER every vCPU thread has joined — so the host mapping
    /// strictly outlives every reader of this pointer).
    ///
    /// SAFETY: deref is sound only after a paired `Acquire` load on
    /// `request_kva` returns non-zero — the coordinator's
    /// `Release` store on `request_kva` orders this pointer's
    /// publication. After that point the host-side guest-DRAM
    /// mapping at this address stays mapped for the VM run because
    /// `vm.guest_mem` is dropped only after `collect_results` joins
    /// every vCPU thread (so no read can outlive the unmap), and
    /// the kernel's `scx_sched` slab page is not freed until well
    /// after the `exit_kind != 0` transition we care about. The
    /// vCPU only ever reads (`read_volatile`), never writes, so
    /// there is no torn-update concern beyond the guest's own
    /// `atomic_set` write — which is the ONE write the watchpoint
    /// catches.
    pub(crate) kind_host_ptr: AtomicPtr<u32>,
    /// Set by the vCPU thread that observed `KVM_EXIT_DEBUG` AND
    /// confirmed the post-store `exit_kind` value indicates an
    /// error-class exit (`>= SCX_EXIT_ERROR == 1024`). The
    /// `KIND -> SCX_EXIT_DONE` transition on a clean shutdown
    /// (`scx_unregister`) also writes `exit_kind` and trips the
    /// watchpoint, but its post-store value is `1` (`SCX_EXIT_DONE`)
    /// and MUST NOT trigger the failure-dump freeze — emitting a
    /// dump on every clean test exit is a regression. The freeze
    /// coordinator polls `hit` with Acquire ordering once the
    /// watchpoint is armed; the vCPU's prior Release store
    /// synchronizes-with that load. Mirrors the prior
    /// `cached_bss_pa != 0` poll semantics so the late-trigger
    /// state machine stays unchanged.
    pub(crate) hit: AtomicBool,
    /// EventFd written alongside every `hit.store(true, Release)` so
    /// the freeze coordinator's epoll set wakes immediately on a
    /// late-trigger fire instead of waiting for the next epoll
    /// timeout. EFD_NONBLOCK so spurious additional writes never
    /// stall the writer (an overflowing counter would only happen if
    /// the coordinator never drained — in which case it's already
    /// servicing the trigger). The vCPU thread's `Release` store on
    /// `hit` happens-before the eventfd write to libc; an Acquire
    /// load on `hit` after the coordinator drains the eventfd
    /// observes the store on weakly-ordered architectures.
    pub(crate) hit_evt: EventFd,
    /// User-watchpoint slot state for slots 1..=3 (slot 0 is the
    /// `*scx_root->exit_kind` trigger above and never appears in
    /// this array). The array index `+ 1` is the per-arch hardware
    /// slot:
    ///   - x86_64: `user[0]` -> DR1, `user[1]` -> DR2, `user[2]` ->
    ///     DR3 (`debugreg[1..=3]` plus DR7 enable bits).
    ///   - aarch64: `user[0]` -> watchpoint 1, `user[1]` ->
    ///     watchpoint 2, `user[2]` -> watchpoint 3
    ///     (`dbg_wvr[1..=3]` and `dbg_wcr[1..=3]`).
    ///
    /// Each slot is updated by `Op::WatchSnapshot` after the freeze
    /// coordinator publishes the resolved KVA; the vCPU's
    /// `self_arm_watchpoint` arms every requested slot on the next
    /// loop iteration. A `KVM_EXIT_DEBUG` identifies which slot
    /// fired (DR6 bits B0..B3 on x86_64; FAR vs armed-WVR range
    /// match on aarch64) and stores `true` into the corresponding
    /// `hit` flag.
    pub(crate) user: [WatchpointSlot; 3],
    /// Fast-path gate for `self_arm_watchpoint`. `0` until any
    /// publisher (the freeze coordinator's err_exit publish or
    /// `arm_user_watchpoint`) writes a non-zero KVA into ANY slot;
    /// then flipped to `1` and never reset for the run. The vCPU
    /// loop loads this once with `Relaxed` before each KVM_RUN and
    /// short-circuits the four `Acquire` loads on `request_kva`
    /// when no arm has ever been requested. Without this gate every
    /// vCPU iteration eats four cross-thread atomic loads even
    /// before the watchpoint becomes interesting (the common case
    /// for tests that never trigger sched_ext error transitions
    /// AND register no `Op::WatchSnapshot` slots).
    ///
    /// `Relaxed` is correct here because the gate's only purpose is
    /// to skip the per-slot `request_kva` reads. When the gate
    /// flips false→true the publishers also issue a Release store
    /// on the slot's `request_kva`; once a vCPU sees the gate set,
    /// it falls through to the `Acquire` load on `request_kva`
    /// which carries the synchronizes-with edge. The gate itself
    /// never publishes data — it only authorises the slow path.
    /// `AtomicU8` instead of `AtomicBool` so a future second flag
    /// (e.g. "any disarm requested") can pack into the same word
    /// without touching the call sites.
    pub(crate) any_armed: AtomicU8,
}

/// Per-user-watchpoint slot state. One slot per hardware
/// breakpoint/watchpoint register pair (DR1/DR2/DR3 on x86_64;
/// watchpoint 1/2/3 on aarch64).
pub(crate) struct WatchpointSlot {
    /// Resolved KVA the coordinator wants armed. `0` = unallocated.
    /// Published by the freeze coordinator's `arm_user_watchpoint`
    /// handler (in `crate::vmm::freeze_coord`) after it resolves the
    /// symbol path through BTF + kallsyms. Once non-zero, every vCPU
    /// thread arms its corresponding hardware slot on its next loop
    /// iteration.
    pub(crate) request_kva: AtomicU64,
    /// Set by a vCPU when it observes a `KVM_EXIT_DEBUG` whose
    /// arch-specific identifier matches this slot (DR6 bit
    /// `B{1,2,3}` on x86_64; `far` falling within `[wvr_base,
    /// wvr_base + 4)` of an armed slot on aarch64). The freeze
    /// coordinator's epoll loop polls all three `hit` flags with
    /// Acquire on each `WATCHPOINT` token wake, runs
    /// `freeze_and_dispatch(FreezeMode::Capture { gate_on_exit_kind: false })` on any trip, and stores the
    /// report under the registered tag in the bridge.
    pub(crate) hit: AtomicBool,
    /// Snapshot tag the bridge stores the captured report under.
    /// Mutex-locked so the host-side watch-register handler can
    /// publish the tag alongside the request_kva atomically. The
    /// coordinator reads this when latching a fire to look up the
    /// bridge key. `String::new()` until the slot is allocated.
    pub(crate) tag: std::sync::Mutex<String>,
}

impl WatchpointSlot {
    fn new() -> Self {
        Self {
            request_kva: AtomicU64::new(0),
            hit: AtomicBool::new(false),
            tag: std::sync::Mutex::new(String::new()),
        }
    }
}

/// `SCX_EXIT_ERROR` from `enum scx_exit_kind` in
/// `kernel/sched/ext_internal.h`. Values below this threshold are
/// clean-exit classes (`SCX_EXIT_NONE = 0`, `SCX_EXIT_DONE = 1`,
/// `SCX_EXIT_UNREG = 64`, etc.) — the kernel writes them to
/// `sch->exit_kind` during normal `scx_unregister` flow. Values
/// `>= 1024` are error classes (`SCX_EXIT_ERROR`,
/// `SCX_EXIT_ERROR_BPF`, `SCX_EXIT_ERROR_STALL`) and are the only
/// transitions the failure-dump freeze cares about. Pinned per
/// `kernel/sched/ext_internal.h::scx_exit_kind::SCX_EXIT_ERROR =
/// 1024`.
pub(crate) const SCX_EXIT_ERROR_THRESHOLD: u32 = 1024;

impl WatchpointArm {
    pub(crate) fn new() -> std::io::Result<Self> {
        Ok(Self {
            request_kva: AtomicU64::new(0),
            kind_host_ptr: AtomicPtr::new(std::ptr::null_mut()),
            hit: AtomicBool::new(false),
            hit_evt: EventFd::new(EFD_NONBLOCK)?,
            user: [
                WatchpointSlot::new(),
                WatchpointSlot::new(),
                WatchpointSlot::new(),
            ],
            any_armed: AtomicU8::new(0),
        })
    }

    /// Mark the arm-fast-path gate as live. Idempotent — every
    /// publisher (freeze coordinator's err_exit publish,
    /// `arm_user_watchpoint`) calls this after the Release store on
    /// `request_kva`. `Relaxed` is sufficient: the gate only
    /// authorises the per-slot `Acquire` loads in
    /// `self_arm_watchpoint`, which carry their own
    /// synchronizes-with edge from the publisher's Release.
    pub(crate) fn mark_armed(&self) {
        self.any_armed.store(1, Ordering::Relaxed);
    }

    /// Disarm slot 0 (the freeze coordinator's
    /// `*scx_root->exit_kind` watchpoint): clear `request_kva`, null
    /// `kind_host_ptr`, then clear the sticky `hit`. Used whenever the
    /// `scx_sched` the slot watches is being torn down — the scheduler
    /// detach / rebind the scan-tick poll detects, AND the explicit
    /// guest swap-notify the freeze coordinator processes synchronously
    /// — so a stale DR0 can no longer fire into the now-RCU-freed (and
    /// possibly slab-recycled) page.
    ///
    /// Store order is load-bearing: `request_kva` FIRST (Release), then
    /// `kind_host_ptr` (Release), then `hit` (Release). Clearing
    /// `request_kva` first makes a racing vCPU's Acquire load return 0
    /// so its next `self_arm_watchpoint` reissues
    /// `KVM_SET_GUEST_DEBUG` without slot 0's enable bits (clearing
    /// DR0 / DR7 L0/G0) BEFORE the host pointer is nulled; an in-flight
    /// `read_volatile` on the previously-observed pointer stays safe
    /// because the host mapping (`vm.guest_mem`) outlives every vCPU
    /// thread. Clearing `hit` prevents the late-trigger arm from
    /// re-observing a stale fire from the scheduler whose teardown this
    /// disarm closes out; real errors still latch via the BPF `.bss`
    /// `err_triggered` path, which is independent of `hit`.
    pub(crate) fn disarm(&self) {
        self.request_kva.store(0, Ordering::Release);
        self.kind_host_ptr
            .store(std::ptr::null_mut(), Ordering::Release);
        self.hit.store(false, Ordering::Release);
    }

    /// Latch `hit=true` AND wake the freeze coordinator's epoll loop
    /// — but only on the false→true transition. Used on every
    /// `KVM_EXIT_DEBUG` site that confirms an error-class write to
    /// `*scx_root->exit_kind`.
    ///
    /// `compare_exchange` on `hit` makes the latch idempotent
    /// across two race patterns:
    ///   - Cross-vCPU concurrent stores: hardware data-write
    ///     watchpoints trap only on the executing vCPU (DR0..DR3
    ///     are per-vCPU on x86_64; `dbg_wvr/dbg_wcr` are per-vCPU
    ///     on aarch64), so a single store cannot fire on more than
    ///     one vCPU. But two vCPUs writing to the watched address
    ///     in close succession each produce a `KVM_EXIT_DEBUG`;
    ///     only the first to win the CAS publishes the eventfd
    ///     edge. Peer fires see the slot already latched and skip
    ///     the eventfd write — preventing the freeze coordinator
    ///     from rendezvousing twice for what should be one logical
    ///     event.
    ///   - Re-fires before reset: if a vCPU fires again before the
    ///     freeze coordinator's reset path runs, the second fire
    ///     CAS true→true and skips the eventfd write. The
    ///     coordinator only resets `hit` on the slot-0
    ///     suppression / rendezvous-timeout path
    ///     (`crate::vmm::freeze_coord::run_vm`'s coordinator
    ///     loop, the
    ///     `None if watchpoint_only_trigger` arm — slot 0 alone
    ///     resets on suppression so the next genuine error-class
    ///     write retriggers; on a successful dump slot 0 stays
    ///     latched and `freeze_state = Done` ends the run); user
    ///     slots 1..=3 reset via `swap(false)` in the per-iteration
    ///     user-slot dispatch loop.
    ///
    /// `Release` ordering on the success path synchronizes-with
    /// the coordinator's `Acquire` load on `hit`. `Relaxed` on the
    /// failure path is safe because the slot is already latched —
    /// no new data is published.
    ///
    /// A failed eventfd write is logged but non-fatal: the `hit`
    /// flag still trips the next epoll tick (timerfd or timeout),
    /// so the trigger eventually fires either way.
    pub(crate) fn latch_hit(&self) {
        if self
            .hit
            .compare_exchange(false, true, Ordering::Release, Ordering::Relaxed)
            .is_err()
        {
            // Already latched by a peer vCPU or a prior fire on
            // this vCPU; the eventfd edge is already pending or
            // has already woken the coordinator.
            return;
        }
        if let Err(e) = self.hit_evt.write(1) {
            tracing::warn!(
                error = %e,
                "WatchpointArm::latch_hit: eventfd write failed; \
                 coordinator will still trip on next epoll timeout"
            );
        }
    }

    /// Latch a user-watchpoint slot fire — but only on the
    /// false→true transition. `idx` selects the DR1/DR2/DR3 slot
    /// (0..=2 mapping to DR1..=DR3). Same dedup rationale as
    /// [`Self::latch_hit`]: cross-vCPU concurrent stores and re-
    /// fires before the coordinator's `swap(false)` reset all
    /// converge on a single eventfd edge.
    ///
    /// Out-of-range `idx` is a programming error caught at the
    /// boundary — the helper returns silently (no eventfd write,
    /// no latch) so a bug in the dispatch loop's index arithmetic
    /// surfaces as "no fire" in test logs rather than a panic on
    /// the kernel-boundary path.
    pub(crate) fn latch_user_hit(&self, idx: usize) {
        if idx >= self.user.len() {
            return;
        }
        if self.user[idx]
            .hit
            .compare_exchange(false, true, Ordering::Release, Ordering::Relaxed)
            .is_err()
        {
            return;
        }
        if let Err(e) = self.hit_evt.write(1) {
            tracing::warn!(
                error = %e,
                idx,
                "WatchpointArm::latch_user_hit: eventfd write failed; \
                 coordinator will still trip on next epoll timeout"
            );
        }
    }
}

/// Maximum consecutive non-EINTR failures from `KVM_SET_GUEST_DEBUG`
/// before the watchpoint arm path gives up and stops retrying. EINTR
/// failures (transient — signal interrupted the ioctl, e.g.
/// SIGRTMIN-driven kick race) do NOT count toward this cap. Only
/// permanent errors (unsupported cap, EINVAL on the debug struct,
/// hardware DR0 unavailable on this host) accumulate. Three retries
/// gives one cycle of headroom for transient ioctl interactions
/// before falling back; after the budget is exhausted the BPF .bss
/// latch path carries the late-trigger signal and the watchpoint
/// stays disabled for the rest of the run.
#[allow(dead_code)]
pub(crate) const WATCHPOINT_MAX_NON_EINTR_FAILURES: u8 = 3;

/// Whether the kernel is still holding guest debug on this vCPU.
///
/// `armed_slots` is NOT a proxy for this. It mirrors the KVAs we
/// *requested*, and an all-zero request set is a live posture:
/// `WatchpointArm::disarm` (scheduler detach / A→B rebind /
/// sched-swap notify) and the coordinator's post-capture user-slot
/// release both zero `request_kva` while vCPU threads run. Before
/// this type existed the aarch64 arm path answered that state with
/// `KVM_GUESTDBG_ENABLE` and no DBGWCR.E bits, which leaves
/// `vcpu->guest_debug` non-zero — and `arch/arm64/kvm/debug.c
/// ::kvm_arm_setup_mdcr_el2` keys MDCR_EL2.TDE off exactly that, so
/// the guest's own BRKs kept routing to EL2 while `armed_slots` read
/// all-zero. Track engagement from the ioctls we actually issued.
///
/// Inert on x86_64: that arm path never releases guest debug (a
/// guest #BP is intercepted only under `KVM_GUESTDBG_USE_SW_BP`,
/// which ktstr never sets, so there is no wedge to recover from).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum GuestDebugState {
    /// No `KVM_SET_GUEST_DEBUG` carrying `KVM_GUESTDBG_ENABLE` is
    /// outstanding — `vcpu->guest_debug` is zero and debug
    /// exceptions go to the guest's own vector.
    Released,
    /// An ioctl carrying `KVM_GUESTDBG_ENABLE` was accepted and no
    /// successful release has followed.
    Engaged,
    /// The release ioctl kept failing and we stopped retrying at
    /// `WATCHPOINT_MAX_NON_EINTR_FAILURES`. The kernel still holds
    /// guest debug; this is deliberately distinct from `Released` so
    /// "we gave up" never reads as "nothing is programmed".
    // Never constructed on x86_64 — that arm path has no release.
    #[allow(dead_code)]
    ReleaseFailed,
}

impl GuestDebugState {
    /// State implied by a `KVM_SET_GUEST_DEBUG` that returned `Ok`
    /// with `control`. A control word without `KVM_GUESTDBG_ENABLE`
    /// makes `kvm_arch_vcpu_ioctl_set_guest_debug` zero
    /// `vcpu->guest_debug` outright.
    #[allow(dead_code)]
    pub(crate) fn after_ok(control: u32) -> Self {
        if control & kvm_bindings::KVM_GUESTDBG_ENABLE != 0 {
            Self::Engaged
        } else {
            Self::Released
        }
    }

    /// Whether `disarm_guest_debug` still has an ioctl to issue.
    #[allow(dead_code)]
    pub(crate) fn needs_release(self) -> bool {
        matches!(self, Self::Engaged)
    }
}

/// Self-arm a hardware data-write watchpoint on `kva` if the per-vCPU
/// state shows the requested KVA changed.
///
/// `armed` tracks the KVA currently programmed into the vCPU's
/// `debugreg[0]` (`0` = no watchpoint armed yet). `request` is the
/// shared atomic the coordinator publishes the resolved
/// `exit_kind_kva` into. When the two diverge, this issues
/// `KVM_SET_GUEST_DEBUG`; once successful, `*armed` is updated to
/// match `request` so the next call is a no-op.
///
/// `failures` counts consecutive non-EINTR failures. EINTR (signal
/// race against `SIGRTMIN`-driven kicks) is transient and does NOT
/// stamp `*armed`; the next iteration retries. Other errors are
/// counted; once `*failures >= WATCHPOINT_MAX_NON_EINTR_FAILURES`
/// we stamp `*armed = req` so the loop stops re-issuing the doomed
/// ioctl. A successful arm resets `*failures` to 0.
///
/// Returns `true` if the call landed a new arm, `false` if no work
/// was needed or the ioctl failed (callers may surface a single
/// warn — failure is non-fatal: the BPF .bss fallback continues to
/// work).
///
/// x86_64 implementation. The DR0/DR7 layout is Intel SDM Vol. 3B
/// Chapter 17. aarch64 has its own DBGWCR/DBGWVR encoding implemented
/// in the `cfg(target_arch = "aarch64")` sibling below; both share
/// this signature and the same per-slot semantics.
///
/// Arms ALL requested slots (slot 0 for `*scx_root->exit_kind`, plus
/// slots 1..=3 for user `Op::WatchSnapshot` registrations) in a
/// single `KVM_SET_GUEST_DEBUG` ioctl. `armed_slots` tracks the
/// currently-armed KVA in each slot; whenever any slot's
/// `request_kva` differs from its `armed_slots` entry the helper
/// rebuilds the full debugreg + DR7 (x86_64) or `dbg_wcr/dbg_wvr`
/// arrays (aarch64) and re-issues the ioctl.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn self_arm_watchpoint(
    vcpu: &kvm_ioctls::VcpuFd,
    watchpoint: &WatchpointArm,
    armed_slots: &mut [u64; 4],
    failures: &mut u8,
    single_step_pending: bool,
    single_step_slot: usize,
    armed_single_step: &mut bool,
    foreign_debug: bool,
    guest_debug_state: &mut GuestDebugState,
) -> bool {
    // Single-step bookkeeping is aarch64-only (the ARM watchpoint trap
    // fires BEFORE the store retires, so re-entering KVM_RUN replays
    // the same instruction → infinite-fire without
    // KVM_GUESTDBG_SINGLESTEP). On x86_64 the trap is taken AFTER the
    // store (Intel SDM Vol. 3B 17.2.4) and re-entry advances normally,
    // so these inputs are unused; consume them here to keep the
    // signature shared with the aarch64 sibling.
    //
    // `foreign_debug` and `guest_debug_state` are aarch64-only because
    // the wedge they recover from is. `vmx_update_exception_bitmap`
    // sets the BP_VECTOR intercept only under
    // `KVM_GUESTDBG_USE_SW_BP`, and we arm with
    // `KVM_GUESTDBG_USE_HW_BP`, so a guest kprobe's `int3` goes
    // straight to the guest IDT — there is no x86 analogue of arm64's
    // BRK-swallowed-by-MDCR_EL2.TDE hang. DB_VECTOR is a different
    // matter: `vmx_update_exception_bitmap` intercepts it
    // unconditionally, and under `KVM_GUESTDBG_USE_HW_BP`
    // `handle_exception_nmi` reflects a guest #DB to userspace instead
    // of reinjecting it. That cannot wedge the vCPU (#DB is
    // trap-class, so the instruction retired), and ktstr's guests do
    // not generate one today: x86 kprobes no longer single-step with
    // EFLAGS.TF (`arch/x86/kernel/kprobes/core.c::setup_singlestep`)
    // and guest DR writes are shadowed while we hold guest debug. So
    // this path never needs to hand debug back.
    let _ = (
        single_step_pending,
        single_step_slot,
        &mut *armed_single_step,
        foreign_debug,
        &mut *guest_debug_state,
    );
    // Fast-path gate: short-circuit when no publisher has flipped
    // `any_armed`. The freeze coordinator's err_exit publish and
    // `arm_user_watchpoint` set the gate (via
    // `WatchpointArm::mark_armed`) AFTER their Release on
    // `request_kva`; until then no slot can carry a non-zero KVA
    // and the per-slot `Acquire` reads below are guaranteed to
    // return zeros. Skipping them saves four cross-thread atomic
    // loads per KVM_RUN iteration on every vCPU thread. On x86_64
    // TSO makes Acquire loads cheap (plain MOV), but the gate
    // still removes four cache-coherent reads from the run-loop
    // hot path on the common case where no test arms a
    // watchpoint.
    if watchpoint.any_armed.load(Ordering::Relaxed) == 0 {
        return false;
    }
    let mut requests = [0u64; 4];
    requests[0] = watchpoint.request_kva.load(Ordering::Acquire);
    for i in 0..3 {
        requests[i + 1] = watchpoint.user[i].request_kva.load(Ordering::Acquire);
    }
    if requests == *armed_slots {
        return false;
    }
    use kvm_bindings::{KVM_GUESTDBG_ENABLE, KVM_GUESTDBG_USE_HW_BP, kvm_guest_debug};
    let mut debug_struct = kvm_guest_debug {
        control: KVM_GUESTDBG_ENABLE | KVM_GUESTDBG_USE_HW_BP,
        pad: 0,
        arch: kvm_bindings::kvm_guest_debug_arch::default(),
    };
    // DR7 base: GE (0x200) + MBS (0x400) + LE (0x100). Per-DR enable
    // and R/W/LEN encodings get OR'd in for each requested slot.
    //
    // The per-slot loop can legitimately emit nothing: `any_armed`
    // stays set for the run lifetime once flipped, but slots ARE
    // released while vCPU threads run — `WatchpointArm::disarm`
    // (scheduler detach / A→B rebind / sched-swap notify) and the
    // coordinator's post-capture user-slot release both zero
    // `request_kva`. An all-zero request set here leaves DR7 with
    // its GE/MBS/LE base and no L/G enable bits, so no debug
    // register fires; combined with BP_VECTOR being intercepted only
    // under `KVM_GUESTDBG_USE_SW_BP`, the leftover
    // `KVM_GUESTDBG_ENABLE` costs nothing on x86. The aarch64
    // sibling has to issue control 0 in the same state because
    // MDCR_EL2.TDE keys off `vcpu->guest_debug` alone.
    let mut dr7: u64 = 0x400 | 0x200 | 0x100;
    for (i, kva) in requests.iter().enumerate() {
        if *kva == 0 {
            continue;
        }
        debug_struct.arch.debugreg[i] = *kva;
        // Per-slot DR7 layout (Intel SDM Vol. 3B 17.2.4):
        //   bit 2*i     = L<i>        → local enable across task switches
        //   bit 2*i+1   = G<i>        → global enable
        //   bits 16+4*i .. 17+4*i = R/W<i> = 0b01 (trap on data writes only)
        //   bits 18+4*i .. 19+4*i = LEN<i> = 0b11 (4-byte length)
        // 4-byte LEN matches the existing DR0 setup (the kernel writes
        // `*scx_root->exit_kind` as a u32; user-arm targets are also
        // u32 / u64-aligned scalars). Mismatched access widths still
        // fire a watchpoint when ANY byte of the access overlaps the
        // DR_LEN range, so 4-byte LEN catches u32 / u64 / pointer-width
        // writes equally.
        dr7 |= (0b11) << (2 * i); // L<i> + G<i>
        dr7 |= (0b01) << (16 + 4 * i); // R/W<i> = data-write
        dr7 |= (0b11) << (18 + 4 * i); // LEN<i> = 4-byte
    }
    debug_struct.arch.debugreg[7] = dr7;
    match vcpu.set_guest_debug(&debug_struct) {
        Ok(()) => {
            *armed_slots = requests;
            *failures = 0;
            true
        }
        Err(e) => {
            // EINTR is transient (SIGRTMIN kick raced the ioctl).
            // Do NOT stamp `armed` and do NOT increment `failures`
            // — the next iteration's call retries the same KVAs.
            if e.errno() == libc::EINTR {
                tracing::debug!(
                    err = %e,
                    requests = ?requests,
                    "self_arm_watchpoint: EINTR — will retry next iteration"
                );
                return false;
            }
            *failures = failures.saturating_add(1);
            tracing::warn!(
                err = %e,
                requests = ?requests,
                failures = *failures,
                "self_arm_watchpoint: KVM_SET_GUEST_DEBUG failed"
            );
            if *failures >= WATCHPOINT_MAX_NON_EINTR_FAILURES {
                tracing::warn!(
                    requests = ?requests,
                    failures = *failures,
                    "self_arm_watchpoint: hit retry cap, suppressing further \
                     attempts; falling back to BPF .bss poll for failure-dump \
                     trigger"
                );
                *armed_slots = requests;
            }
            false
        }
    }
}

/// Release guest debug on this vCPU by issuing
/// `KVM_SET_GUEST_DEBUG` with control 0. The kernel's
/// `kvm_arch_vcpu_ioctl_set_guest_debug` treats a control word
/// without `KVM_GUESTDBG_ENABLE` as "clear everything" — it zeroes
/// `vcpu->guest_debug` and drops the pending-single-step flag — so
/// the next entry runs with MDCR_EL2.TDE clear and guest debug
/// exceptions go to the guest's own EL1 vector again.
///
/// Idempotent: returns immediately unless `guest_debug_state` says
/// the kernel is actually holding guest debug, so the caller can
/// invoke it on every loop iteration. The gate is deliberately NOT
/// `armed_slots == [0; 4]` — the mirror tracks requested KVAs, and a
/// released slot set with `KVM_GUESTDBG_ENABLE` still outstanding is
/// exactly the state that needs this ioctl (see `GuestDebugState`).
///
/// Error handling mirrors the arm path — EINTR retries on the next
/// iteration, other errors count toward
/// `WATCHPOINT_MAX_NON_EINTR_FAILURES` and then park the state at
/// `ReleaseFailed` to stop an unbounded ioctl storm in the run loop
/// without claiming the release succeeded.
#[cfg(target_arch = "aarch64")]
fn disarm_guest_debug(
    vcpu: &kvm_ioctls::VcpuFd,
    armed_slots: &mut [u64; 4],
    failures: &mut u8,
    armed_single_step: &mut bool,
    guest_debug_state: &mut GuestDebugState,
) -> bool {
    if !guest_debug_state.needs_release() {
        return false;
    }
    let debug_struct = kvm_bindings::kvm_guest_debug {
        control: 0,
        pad: 0,
        arch: kvm_bindings::kvm_guest_debug_arch::default(),
    };
    match vcpu.set_guest_debug(&debug_struct) {
        Ok(()) => {
            let released = *armed_slots;
            *armed_slots = [0u64; 4];
            *armed_single_step = false;
            *guest_debug_state = GuestDebugState::Released;
            *failures = 0;
            // Log the slot set we gave up so the line distinguishes
            // "lost nothing that mattered" from "lost a user watch".
            // Slot 0 degrades to the BPF `.bss` poll; slots 1..=3 are
            // `Op::WatchSnapshot` arms with NO fallback — their hits
            // are latched only from KVM_EXIT_DEBUG dispatch.
            tracing::warn!(
                released = ?released,
                "disarmed guest debug after a guest-owned debug exception; \
                 this vCPU's hardware watchpoints are gone for the run — \
                 the failure-dump trigger falls back to the BPF .bss poll, \
                 and any user Op::WatchSnapshot slot armed here can no \
                 longer fire at all"
            );
            true
        }
        Err(e) => {
            if e.errno() == libc::EINTR {
                tracing::debug!(
                    err = %e,
                    "disarm_guest_debug: EINTR — will retry next iteration"
                );
                return false;
            }
            *failures = failures.saturating_add(1);
            tracing::error!(
                err = %e,
                failures = *failures,
                "disarm_guest_debug: KVM_SET_GUEST_DEBUG failed; the guest \
                 cannot take its own debug exceptions while we hold \
                 MDCR_EL2.TDE"
            );
            if *failures >= WATCHPOINT_MAX_NON_EINTR_FAILURES {
                // Stop re-issuing a doomed ioctl on every KVM_RUN. The
                // vCPU stays wedged, but the watchdog surfaces that; a
                // hot ioctl loop would only bury the evidence. Park at
                // `ReleaseFailed` rather than `Released`: nothing was
                // released, and the mirrors below only record that we
                // no longer consider any slot ours to re-program.
                *armed_slots = [0u64; 4];
                *armed_single_step = false;
                *guest_debug_state = GuestDebugState::ReleaseFailed;
            }
            false
        }
    }
}

/// Build the `KVM_SET_GUEST_DEBUG` control word for a requested
/// slot set.
///
/// An all-zero request set with no step pending is the RELEASED
/// posture, not "enabled with nothing armed". `arch/arm64/kvm/
/// debug.c::kvm_arm_setup_mdcr_el2` sets MDCR_EL2.TDE for any
/// non-zero `vcpu->guest_debug`, and `arch/arm64/kvm/guest.c
/// ::kvm_arch_vcpu_ioctl_set_guest_debug` does not check that any
/// DBGWCR.E is set, so `KVM_GUESTDBG_ENABLE` over an empty
/// `dbg_wcr` array keeps routing the guest's own BRKs to EL2 while
/// watching nothing. Control 0 takes the kernel's
/// `!(dbg->control & KVM_GUESTDBG_ENABLE)` branch, which zeroes
/// `vcpu->guest_debug` and drops the pending-single-step flag.
///
/// A pending single-step never takes the release branch: dropping
/// `KVM_GUESTDBG_SINGLESTEP` mid-dance clears the kernel's
/// pending-step state, so the SOFTSTP_LOW exit that clears
/// `single_step_pending` would never arrive and the run loop would
/// stay latched in step with a stale slot mask.
///
/// `KVM_GUESTDBG_SINGLESTEP` is also what makes
/// `arch/arm64/kvm/debug.c::setup_external_mdscr` write
/// MDSCR_EL1.SS; without it the cpsr SS-bit dance
/// (`kvm_handle_guest_debug` → DBG_SPSR_SS) never re-arms.
#[cfg(target_arch = "aarch64")]
pub(crate) fn watchpoint_control_word(requests: &[u64; 4], single_step_pending: bool) -> u32 {
    use kvm_bindings::{KVM_GUESTDBG_ENABLE, KVM_GUESTDBG_SINGLESTEP, KVM_GUESTDBG_USE_HW};
    if *requests == [0u64; 4] && !single_step_pending {
        return 0;
    }
    let mut control = KVM_GUESTDBG_ENABLE | KVM_GUESTDBG_USE_HW;
    if single_step_pending {
        control |= KVM_GUESTDBG_SINGLESTEP;
    }
    control
}

/// aarch64 implementation. Arms ALL requested slots
/// (`watchpoint.request_kva` for slot 0, `watchpoint.user[i]
/// .request_kva` for slot 1..=3) by populating the
/// `dbg_wcr` / `dbg_wvr` arrays of `kvm_guest_debug_arch` and
/// issuing `KVM_SET_GUEST_DEBUG` with control flags
/// `KVM_GUESTDBG_ENABLE | KVM_GUESTDBG_USE_HW`.
///
/// DBGWCR encoding per ARM DDI 0487 D7.3.11 (and verified
/// against QEMU `insert_hw_watchpoint` in
/// target/arm/hyp_gdbstub.c):
///
/// ```text
///  31  29 28   24 23  21  20  19 16 15 14  13   12  5 4   3 2   1  0
/// +------+-------+------+----+-----+-----+-----+-----+-----+-----+---+
/// | RES0 |  MASK | RES0 | WT | LBN | SSC | HMC | BAS | LSC | PAC | E |
/// +------+-------+------+----+-----+-----+-----+-----+-----+-----+---+
/// ```
///
///   bit 0       E   = 1 (enable)
///   bits \[2:1\]  PAC = 0b11 (EL0+EL1, any security state)
///   bits \[4:3\]  LSC = 0b10 (store/write only — matches the
///                          x86 R/W=01 semantic the freeze
///                          coordinator already encodes)
///   bits \[12:5\] BAS = which bytes of the 8-byte block at
///                     DBGWVR fire. For a 4-byte watch on a
///                     4-byte aligned KVA, BAS = 0xF
///                     shifted left by `kva & 0x7`.
///   bits \[15:13\] HMC = 0
///   bits \[19:16\] SSC = 0
///   bit 20       WT  = 0 (unlinked)
///   bits \[23:21\] LBN = 0
///   bits \[28:24\] MASK = 0 (no address mask; we never use
///                          larger ranges)
///
/// Concrete WCR values:
///   - 4-byte write at doubleword offset 0:
///     `0x1 | (3 << 1) | (2 << 3) | (0xF << 5)` = `0x1F7`
///   - 4-byte write at doubleword offset 4:
///     `0x1 | (3 << 1) | (2 << 3) | (0xF << 9)` = `0x1E17`
///
/// DBGWVR holds bits VA\[52:2\] in the architectural form
/// `RESS | VA[52:49] | VA[48:2] | 0 0`; the kernel sign-
/// extends as required. We pass `kva & ~0x7` so the bottom
/// 3 bits are zero (8-byte aligned base), and BAS picks the
/// 4 bytes inside that block we actually want to watch. ARM
/// requires DBGWVR's bottom 2 bits be zero; the upstream
/// `arm_user_watchpoint` validator already rejects KVAs
/// whose bottom 2 bits are set, so this layer never sees a
/// misaligned target.
///
/// Slot semantics match the x86_64 path exactly:
///   - `armed_slots[i]` mirrors the requested KVA so a
///     no-change iteration short-circuits.
///   - EINTR is transient and does NOT count toward the
///     non-EINTR failure cap.
///   - On hitting `WATCHPOINT_MAX_NON_EINTR_FAILURES`, the
///     slot stamps to suppress further retries.
///
/// `request_kva` IS reset while vCPU threads run, so an
/// all-zero request set is a normal posture rather than an
/// impossible one: `WatchpointArm::disarm` clears slot 0 on
/// scheduler detach, on an A→B rebind, and on the synchronous
/// sched-swap notify, and the freeze coordinator zeroes a user
/// slot's `request_kva` after every `Op::WatchSnapshot`
/// capture. `watchpoint_control_word` answers that state with
/// control 0 so the kernel drops `vcpu->guest_debug` (and with
/// it MDCR_EL2.TDE) instead of holding an enabled-but-empty
/// watchpoint set. `guest_debug_state` — not `armed_slots` —
/// is the only authority on whether the kernel still holds
/// guest debug.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn self_arm_watchpoint(
    vcpu: &kvm_ioctls::VcpuFd,
    watchpoint: &WatchpointArm,
    armed_slots: &mut [u64; 4],
    failures: &mut u8,
    single_step_pending: bool,
    single_step_slot: usize,
    armed_single_step: &mut bool,
    foreign_debug: bool,
    guest_debug_state: &mut GuestDebugState,
) -> bool {
    // Sticky disarm. `dispatch_watchpoint_hit` sets `foreign_debug`
    // when a KVM_EXIT_DEBUG cannot be attributed to one of our own
    // slots — in practice a guest kprobe's `brk`, which ktstr plants
    // as core functionality.
    //
    // Why the arm is what breaks the guest: on arm64 ANY non-zero
    // `vcpu->guest_debug` makes KVM set MDCR_EL2.TDE
    // (`arch/arm64/kvm/debug.c::kvm_arm_setup_mdcr_el2`), routing
    // every guest debug exception to EL2. `arch/arm64/kvm/
    // handle_exit.c` maps `ESR_ELx_EC_BRK64` to
    // `kvm_handle_guest_debug`, which reflects the exception back
    // into the guest ONLY when `!vcpu->guest_debug`; otherwise it
    // returns KVM_EXIT_DEBUG to userspace. So while we hold guest
    // debug, the guest's own BRK never reaches its EL1 debug vector,
    // the PC never advances, and the vCPU re-executes the BRK on
    // every KVM_RUN. The x86 BRK analogue does not wedge: KVM
    // intercepts a guest #BP only under `KVM_GUESTDBG_USE_SW_BP`, so
    // an `int3` goes straight to the guest IDT.
    //
    // Handing debug back is therefore the only recovery. Control 0
    // (no `KVM_GUESTDBG_ENABLE`) makes
    // `kvm_arch_vcpu_ioctl_set_guest_debug` zero `vcpu->guest_debug`
    // outright, clearing TDE on the next entry so the re-executed
    // BRK traps to the guest.
    //
    // Sticky because the alternative re-wedges instantly: the slot
    // KVAs are republished on the next attach, so any later
    // iteration would re-arm and swallow the next kprobe. Peer vCPUs
    // keep their arms — the release is per-vCPU. It is not, though,
    // a rare degradation: ktstr's Phase-B kprobes sit on
    // `enqueue_task_scx`, which runs on every CPU, so on aarch64
    // with probes live expect EVERY vCPU to release shortly after
    // arming. The hardware watchpoint is a best-effort early-window
    // mechanism there, not a run-long one — slot 0 falls back to the
    // BPF `.bss` poll, and user `Op::WatchSnapshot` slots have no
    // fallback at all.
    if foreign_debug {
        return disarm_guest_debug(
            vcpu,
            armed_slots,
            failures,
            armed_single_step,
            guest_debug_state,
        );
    }
    // Fast-path gate: short-circuit when no publisher has flipped
    // `any_armed`. The freeze coordinator's err_exit publish and
    // `arm_user_watchpoint` set the gate (via
    // `WatchpointArm::mark_armed`) AFTER their Release on
    // `request_kva`; until then no slot can carry a non-zero KVA
    // and the per-slot `Acquire` reads below are guaranteed to
    // return zeros. Skipping them saves four cross-thread atomic
    // loads per KVM_RUN iteration on every vCPU thread, which is
    // material on aarch64 where each load is an `LDAR` with an
    // associated barrier.
    //
    // `Relaxed` is correct: the gate's only role is to authorise
    // the per-slot `Acquire` loads below. Once the gate is set,
    // those loads carry the synchronizes-with edge from the
    // publisher's `Release` on `request_kva` directly. The gate
    // itself never publishes data — it only signals "the slow
    // path is now interesting."
    if watchpoint.any_armed.load(Ordering::Relaxed) == 0 {
        return false;
    }
    let mut requests = [0u64; 4];
    requests[0] = watchpoint.request_kva.load(Ordering::Acquire);
    for i in 0..3 {
        requests[i + 1] = watchpoint.user[i].request_kva.load(Ordering::Acquire);
    }
    // Re-issue when EITHER the requested slot KVAs changed OR the
    // single-step posture flipped (transition INTO step → disable
    // every matched slot's WCR.E and assert KVM_GUESTDBG_SINGLESTEP;
    // transition OUT OF step → restore WCR.E=1 on every slot and
    // drop SINGLESTEP). The two-conditions form keeps the no-arm
    // fast path intact for the common no-op iteration.
    if requests == *armed_slots && *armed_single_step == single_step_pending {
        return false;
    }
    use kvm_bindings::kvm_guest_debug;
    let control = watchpoint_control_word(&requests, single_step_pending);
    let mut debug_struct = kvm_guest_debug {
        control,
        pad: 0,
        arch: kvm_bindings::kvm_guest_debug_arch::default(),
    };
    // `single_step_slot` carries a 4-bit bitmap of slot indices
    // that fired on this dispatch (bit i set ⇒ slot i was matched
    // by the FAR range check in `dispatch_watchpoint_hit`).
    // Truncate to u8 — only the bottom four bits are defined.
    let step_mask: u8 = (single_step_slot & 0xF) as u8;
    for (i, kva) in requests.iter().enumerate() {
        if *kva == 0 {
            continue;
        }
        // 8-byte aligned base. ARM DDI 0487 D7.3.10 requires
        // DBGWVR's bottom 2 bits be zero; setting bottom 3
        // bits to zero (8-byte align) keeps BAS as the sole
        // byte selector and matches QEMU's
        // `wvr = addr & (~0x7ULL)`.
        debug_struct.arch.dbg_wvr[i] = *kva & !0x7u64;
        // BAS picks the 4 contiguous bytes of the 8-byte
        // block that the watch targets. `byte_offset` is the
        // byte offset of `kva` within that 8-byte block; the
        // 4-byte BAS bitmap (0b1111 = 0xF) shifts left by
        // that offset. For 4-byte aligned KVAs `byte_offset`
        // is 0 or 4 — both valid placements (BAS=0x0F or
        // BAS=0xF0).
        let byte_offset = (*kva & 0x7u64) as u32;
        let bas: u64 = 0xFu64 << byte_offset;
        // PAC=0b11 (bits 2:1) | LSC=0b10 (bits 4:3,
        // write-only) | BAS (bits 12:5). The E bit (bit 0) is
        // cleared on EVERY slot whose bit is set in the step
        // mask — overlapping arms (`arm_user_watchpoint` does
        // not reject duplicate KVAs) can produce simultaneous
        // matches, and EVERY matched slot must have WCR.E
        // cleared during the single-step pass to avoid the
        // offending store re-tripping the watchpoint on its
        // replay. Peer slots that did NOT match keep E=1 so
        // their watches stay live during the step. We diverge
        // from the kernel `arch/arm64/kernel/hw_breakpoint.c
        // ::toggle_bp_registers(AARCH64_DBG_REG_WCR, el, 0)`
        // pattern, which disables WCR.E on EVERY watchpoint
        // slot at the matching exception level during
        // single-step regardless of which slot fired. KVM
        // userspace only programs slots ktstr explicitly arms,
        // so peer slots are always ktstr's own watches; keeping
        // them active during the step preserves the post-step
        // rearm contract (no extra ioctl to restore peer E=1
        // bits) and the watched store is on a matched slot, so
        // peer slots cannot re-trip on the replay.
        let e = if single_step_pending && (step_mask & (1u8 << i)) != 0 {
            0u64
        } else {
            1u64
        };
        let wcr: u64 = e | (0b11u64 << 1) | (0b10u64 << 3) | (bas << 5);
        debug_struct.arch.dbg_wcr[i] = wcr;
    }
    match vcpu.set_guest_debug(&debug_struct) {
        Ok(()) => {
            *armed_slots = requests;
            *armed_single_step = single_step_pending;
            *guest_debug_state = GuestDebugState::after_ok(control);
            *failures = 0;
            true
        }
        Err(e) => {
            // EINTR is transient (SIGRTMIN kick raced the
            // ioctl). Do NOT stamp `armed_slots` /
            // `armed_single_step` and do NOT increment
            // `failures` — the next iteration's call retries
            // with the same posture.
            if e.errno() == libc::EINTR {
                tracing::debug!(
                    err = %e,
                    requests = ?requests,
                    "self_arm_watchpoint: EINTR — will retry next iteration"
                );
                return false;
            }
            *failures = failures.saturating_add(1);
            tracing::warn!(
                err = %e,
                requests = ?requests,
                failures = *failures,
                "self_arm_watchpoint: KVM_SET_GUEST_DEBUG failed"
            );
            if *failures >= WATCHPOINT_MAX_NON_EINTR_FAILURES {
                tracing::warn!(
                    requests = ?requests,
                    failures = *failures,
                    "self_arm_watchpoint: hit retry cap, suppressing further \
                     attempts; falling back to BPF .bss poll for failure-dump \
                     trigger"
                );
                // Stamp the request mirrors only. `guest_debug_state`
                // is left alone deliberately: the ioctl FAILED, so
                // whatever the kernel held before this call it still
                // holds. Clearing it here is the exact conflation
                // that made an all-zero mirror read as "guest debug
                // is off" — a later `foreign_debug` release must
                // still issue its control-0 ioctl.
                *armed_slots = requests;
                *armed_single_step = single_step_pending;
            }
            false
        }
    }
}

impl VcpuThread {
    /// Kick a vCPU out of KVM_RUN. If immediate_exit is available, sets the
    /// flag before sending the signal (Firecracker pattern). Otherwise falls
    /// back to signal-only (the signal handler causes EINTR).
    ///
    /// The `alive` load is only an early no-op optimization after the panic
    /// edge; it is not a lifetime proof. A stale `true` is safe because
    /// `ImmediateExitHandle::set` upgrades its Weak mapping reference to a
    /// transient Arc before checking active, so a stale participation read
    /// remains safe even if the run-loop VcpuFd completes concurrently.
    pub(crate) fn kick(&self) {
        if let Some(ref ie) = self.immediate_exit
            && self.alive.load(Ordering::Acquire)
        {
            let _ = ie.set(1);
            std::sync::atomic::fence(Ordering::Release);
        }
        self.signal();
    }

    /// Send the kick signal to interrupt a blocked KVM_RUN.
    pub(crate) fn signal(&self) {
        unsafe {
            libc::pthread_kill(self.handle.as_pthread_t() as libc::pthread_t, vcpu_signal());
        }
    }

    /// Wait for the thread to exit, retrying an immediate-exit + signal wake
    /// every 10ms. `unpark` additionally releases a userspace park.
    ///
    /// Implementation: blocks in `epoll_wait` on `self.exit_evt`
    /// (bumped by the AP thread after `exited.store(true)` and by
    /// the panic hook on a panic-classified shutdown) plus a
    /// 10ms-interval `timerfd` for the periodic re-kick. The outer
    /// `start.elapsed()` deadline caps the total wait at `timeout`
    /// without an explicit timeout fd. A spurious wake (EINTR or a
    /// stale eventfd-counter drain) loops back without dropping the
    /// kick cadence.
    #[cfg(test)]
    pub(crate) fn wait_for_exit(&self, timeout: Duration) {
        if self.exited.load(Ordering::Acquire) {
            return;
        }

        let epoll = match Epoll::new() {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!(%e, "wait_for_exit: epoll_create1 failed");
                return;
            }
        };
        const EXIT_TOKEN: u64 = 0;
        const KICK_TOKEN: u64 = 1;
        if let Err(e) = epoll.ctl(
            ControlOperation::Add,
            self.exit_evt.as_raw_fd(),
            EpollEvent::new(EventSet::IN, EXIT_TOKEN),
        ) {
            tracing::warn!(%e, "wait_for_exit: add exit_evt to epoll");
            return;
        }
        let mut kick_timer = match TimerFd::new() {
            Ok(t) => t,
            Err(e) => {
                tracing::warn!(%e, "wait_for_exit: timerfd_create failed");
                return;
            }
        };
        let kick_interval = Duration::from_millis(10);
        if let Err(e) = kick_timer.reset(kick_interval, Some(kick_interval)) {
            tracing::warn!(%e, "wait_for_exit: timerfd_settime failed");
            return;
        }
        if let Err(e) = epoll.ctl(
            ControlOperation::Add,
            kick_timer.as_raw_fd(),
            EpollEvent::new(EventSet::IN, KICK_TOKEN),
        ) {
            tracing::warn!(%e, "wait_for_exit: add timerfd to epoll");
            return;
        }

        let start = Instant::now();
        let mut events = [EpollEvent::default(); 2];
        loop {
            if self.exited.load(Ordering::Acquire) {
                return;
            }
            let elapsed = start.elapsed();
            if elapsed >= timeout {
                return;
            }
            let remaining_ms = (timeout - elapsed).as_millis().min(i32::MAX as u128) as i32;
            match epoll.wait(remaining_ms, &mut events) {
                Ok(0) => return, // overall timeout
                Ok(n) => {
                    for ev in &events[..n] {
                        if ev.data() == KICK_TOKEN {
                            // Drain timerfd expiry counter (counter
                            // mode); the read value is uninteresting.
                            let _ = kick_timer.wait();
                            self.kick();
                            self.handle.thread().unpark();
                        }
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => {
                    tracing::warn!(%e, "wait_for_exit: epoll_wait failed");
                    return;
                }
            }
        }
    }
}

/// Parameters for a host-side BPF map write during VM execution.
#[derive(Clone)]
pub(crate) struct BpfMapWriteParams {
    pub(crate) map_name_suffix: String,
    /// Name of the global to write, resolved to a byte offset against the
    /// map's program BTF at write time (see
    /// `freeze_coord::start_bpf_map_write` phase 2). Mirrors
    /// `WatchBpfMapParams::field`; replaces a padding-fragile hardcoded
    /// byte offset.
    pub(crate) field: String,
    pub(crate) value: u32,
}

/// Owned runtime form of a [`crate::test_support::WatchBpfMap`] target —
/// the coordinator/monitor's copy of a named scheduler BPF-map field to read
/// observer-effect-free into a run-level metric. Lowered from the
/// `&'static [&'static WatchBpfMap]` test-entry list (mirrors how
/// [`BpfMapWriteParams`] lowers `BpfMapWrite`).
#[derive(Clone)]
pub(crate) struct WatchBpfMapParams {
    /// Map-name suffix matched via `ends_with` (`.bss`, `cpu_ctx_stor`).
    pub(crate) map_name_suffix: String,
    /// Dot-path into the map's value type (`sys_stat.avg_lat_cri`, or a bare
    /// member like `lat_headroom`).
    pub(crate) field: String,
    /// Scalar vs per-CPU aggregation.
    pub(crate) agg: crate::test_support::BpfMapAgg,
    /// Metric-key leaf appended to the scheduler-obj prefix.
    pub(crate) label: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vmm::kvm;
    use crate::vmm::topology::Topology;

    #[test]
    fn vcpu_signal_is_sigrtmin() {
        let sig = vcpu_signal();
        assert!(sig >= libc::SIGRTMIN(), "signal should be >= SIGRTMIN");
        assert!(sig <= libc::SIGRTMAX(), "signal should be <= SIGRTMAX");
    }

    /// [`VcpuThread::wait_for_exit`] must block on its `exit_evt`
    /// epoll until the AP signals exit, then return promptly. Drives
    /// the real wait path (epoll + 10ms-kick timerfd + `exited`
    /// Acquire load) end-to-end: the AP thread sleeps briefly, then
    /// performs the production exit handshake (`exited.store(true)`
    /// THEN `exit_evt.write(1)` — the same order the AP loop and the
    /// panic hook use) and the parent's `wait_for_exit` must observe
    /// the eventfd wake and return well under its timeout.
    #[test]
    fn vcpu_wait_for_exit_returns_on_exit_signal() {
        use std::sync::Barrier;
        // SIGRTMIN handler so the periodic re-kick `pthread_kill`
        // inside wait_for_exit does not terminate the test process.
        register_vcpu_signal_handler();
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
            llc_cores: None,
        };
        let mut vm = kvm::KtstrKvm::new(topo, 64, false).unwrap();
        let probe_vcpu = vm.vcpus.remove(0);

        let exited = Arc::new(AtomicBool::new(false));
        let exit_evt = Arc::new(EventFd::new(EFD_NONBLOCK).unwrap());
        let alive = Arc::new(AtomicBool::new(true));

        // The AP thread waits for a go signal, sleeps a beat so the
        // parent is already blocked in epoll_wait, then signals exit
        // in the production order before handing back its VcpuFd.
        let go = Arc::new(Barrier::new(2));
        let go_ap = go.clone();
        let exited_ap = Arc::clone(&exited);
        let exit_evt_ap = Arc::clone(&exit_evt);
        let handle = std::thread::Builder::new()
            .name("wait-for-exit-stub".into())
            .spawn(move || {
                go_ap.wait();
                std::thread::sleep(Duration::from_millis(20));
                // Production exit handshake: flip the flag, then bump
                // the eventfd so the epoll wake and the Acquire load
                // both observe the exit.
                exited_ap.store(true, Ordering::Release);
                exit_evt_ap.write(1).unwrap();
                probe_vcpu
            })
            .unwrap();

        let vt = VcpuThread {
            handle,
            exited,
            immediate_exit: None,
            exit_evt,
            alive,
            schedstat_at_exit: Arc::new(std::sync::Mutex::new(None)),
        };

        // Before the AP signals, wait_for_exit must NOT short-circuit:
        // the `exited` flag is still false, so the fast-path return at
        // the top of wait_for_exit does not fire and the call would
        // block. Release the AP and time the real wait.
        assert!(!vt.exited.load(Ordering::Acquire), "pre-signal: not exited");
        go.wait();
        let start = Instant::now();
        // Generous ceiling vs the ~20ms AP delay; a regression that
        // failed to wake on exit_evt would burn the full timeout.
        vt.wait_for_exit(Duration::from_secs(5));
        let waited = start.elapsed();
        assert!(
            vt.exited.load(Ordering::Acquire),
            "wait_for_exit must only return once exited is observed true",
        );
        assert!(
            waited < Duration::from_secs(2),
            "wait_for_exit must wake on exit_evt promptly, not burn the \
             timeout: waited {waited:?}",
        );
        let _ = vt.handle.join();
    }

    /// Pin the millisecond-precision Duration→jiffies conversion.
    /// Sub-second inputs must NOT truncate to 0 (the bug that masked
    /// the freeze-coord early trigger before this helper existed),
    /// whole-second inputs must scale by HZ, and HZ != 1000 must
    /// scale correctly down to the jiffies tick boundary.
    #[test]
    fn duration_to_jiffies_basic() {
        // 500 ms at HZ=1000 → 500 jiffies (the bug case: as_secs()
        // would yield 0 here).
        assert_eq!(duration_to_jiffies(Duration::from_millis(500), 1000), 500);
        // 1500 ms at HZ=1000 → 1500 jiffies (the fractional-second
        // input path must not truncate the integer-seconds component
        // either).
        assert_eq!(duration_to_jiffies(Duration::from_millis(1500), 1000), 1500);
        // 4 s at HZ=250 → 1000 jiffies (lower HZ tick rate; the
        // ms→jiffies arithmetic should land on the same answer as
        // the as_secs()*hz form for whole seconds).
        assert_eq!(duration_to_jiffies(Duration::from_secs(4), 250), 1000);
        // Zero duration → zero jiffies (no UB, no spurious tick).
        assert_eq!(duration_to_jiffies(Duration::from_millis(0), 1000), 0);
        // Degenerate HZ=0 → zero jiffies. Guards against an
        // unresolvable guest-side CONFIG_HZ where
        // `monitor::guest_kernel_hz` falls back to 0; the resulting
        // `half_threshold_jiffies` of 0 means "early-trigger threshold
        // never fires," which is the right degradation — better than
        // a divide-by-zero or an unbounded sentinel that would fire
        // on every iteration.
        assert_eq!(duration_to_jiffies(Duration::from_secs(1), 0), 0);
    }

    #[test]
    fn immediate_exit_secondary_mapping_aliases_run_loop_mapping() {
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
            llc_cores: None,
        };
        let mut vm = kvm::KtstrKvm::new(topo, 64, false).unwrap();
        let run_size = vm.vm_fd.run_size();
        let (mut vcpu, handle) =
            ImmediateExitVcpu::new(vm.vcpus.remove(0), true, run_size).unwrap();
        let handle = handle.unwrap();

        // Initial state should be 0.
        assert_eq!(
            vcpu.read_immediate_exit(),
            0,
            "immediate_exit should start at 0"
        );

        // Set via handle, verify via VcpuFd.
        assert!(handle.set(1));
        assert_eq!(
            vcpu.read_immediate_exit(),
            1,
            "handle.set(1) should be visible via the run-loop mapping"
        );

        // Clear via VcpuFd, verify.
        vcpu.set_immediate_exit(0);
        assert_eq!(
            vcpu.read_immediate_exit(),
            0,
            "owner-side atomic clear should clear the flag"
        );
        assert_eq!(
            handle.read_byte(),
            Some(0),
            "owner-side AtomicU8 clear must be visible through the secondary mapping",
        );
    }

    #[test]
    fn immediate_exit_handle_cross_vcpu() {
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 2,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
            llc_cores: None,
        };
        let mut vm = kvm::KtstrKvm::new(topo, 64, false).unwrap();
        let run_size = vm.vm_fd.run_size();
        let (mut vcpu1, h1) =
            ImmediateExitVcpu::new(vm.vcpus.pop().unwrap(), true, run_size).unwrap();
        let (mut vcpu0, h0) =
            ImmediateExitVcpu::new(vm.vcpus.pop().unwrap(), true, run_size).unwrap();
        let h0 = h0.unwrap();
        let h1 = h1.unwrap();

        // Setting one vCPU's handle should not affect the other.
        assert!(h0.set(1));
        assert_eq!(vcpu0.read_immediate_exit(), 1);
        assert_eq!(
            vcpu1.read_immediate_exit(),
            0,
            "setting vcpu0 handle should not affect vcpu1"
        );

        assert!(h1.set(1));
        assert_eq!(vcpu1.read_immediate_exit(), 1);

        // Clear both.
        assert!(h0.set(0));
        assert!(h1.set(0));
        assert_eq!(vcpu0.read_immediate_exit(), 0);
        assert_eq!(vcpu1.read_immediate_exit(), 0);
    }

    #[test]
    fn immediate_exit_mapping_survives_owner_drop_during_inflight_set() {
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
            llc_cores: None,
        };
        let mut vm = kvm::KtstrKvm::new(topo, 64, false).unwrap();
        let run_size = vm.vm_fd.run_size();
        let (owner, handle) = ImmediateExitVcpu::new(vm.vcpus.remove(0), true, run_size).unwrap();
        let handle = handle.unwrap();
        let acquired = Arc::new(Latch::new());
        let release = Arc::new(Latch::new());
        let acquired_writer = Arc::clone(&acquired);
        let release_writer = Arc::clone(&release);
        let writer_handle = handle.clone();
        let writer = std::thread::spawn(move || {
            writer_handle.set_after_test_barrier(1, &acquired_writer, &release_writer)
        });
        if !acquired.wait_timeout(Duration::from_secs(2)) {
            release.set();
            let _ = writer.join();
            panic!("in-flight immediate-exit writer never acquired its transient mapping Arc");
        }

        // Reproduce the old false-read/deschedule interleaving exactly:
        // writer has observed active=true but is paused before dereference;
        // owner Drop then deactivates the state and destroys the original
        // VcpuFd mapping before the writer resumes. Drop the entire VM owner
        // too: the independent VMA's file reference—not an accidentally-live
        // VmFd—must pin the kernel vCPU/run pages across the in-flight access.
        let (drop_done_tx, drop_done_rx) = std::sync::mpsc::channel();
        let dropper = std::thread::spawn(move || {
            drop(owner);
            drop(vm);
            drop_done_tx.send(()).unwrap();
        });
        drop_done_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("owner Drop must not wait on the paused handle");
        dropper.join().unwrap();
        assert_eq!(handle.read_byte(), None, "stale Weak handle must no-op");

        release.set();
        assert_eq!(
            writer.join().unwrap(),
            Some(1),
            "the in-flight store and read complete through the transient Arc",
        );

        assert!(
            !handle.set(1),
            "a new post-owner set must observe inactive and no-op",
        );
        assert_eq!(handle.read_byte(), None);
    }

    #[test]
    fn immediate_exit_stale_handle_noops_after_owner_return_and_vcpu_drop() {
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
            llc_cores: None,
        };
        let mut vm = kvm::KtstrKvm::new(topo, 64, false).unwrap();
        let run_size = vm.vm_fd.run_size();
        let (owner, handle) = ImmediateExitVcpu::new(vm.vcpus.remove(0), true, run_size).unwrap();
        let handle = handle.unwrap();
        assert!(handle.set(1));

        let vcpu = owner.into_inner();
        assert!(!handle.set(0), "owner return must deactivate future sets");
        assert_eq!(handle.read_byte(), None);

        drop(vcpu);
        assert!(!handle.set(0));
        assert_eq!(handle.read_byte(), None);
    }

    #[test]
    fn immediate_exit_disabled_owner_avoids_secondary_mapping() {
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
            llc_cores: None,
        };
        let mut vm = kvm::KtstrKvm::new(topo, 64, false).unwrap();
        let run_size = vm.vm_fd.run_size();
        let (owner, handle) = ImmediateExitVcpu::new(vm.vcpus.remove(0), false, run_size).unwrap();
        assert!(handle.is_none());
        let mut vcpu = owner.into_inner();
        assert_eq!(vcpu.get_kvm_run().immediate_exit, 0);
    }

    /// The panic-edge `alive=false` gate remains a useful early no-op, but
    /// mapping safety is supplied by `ImmediateExitVcpu`, not this atomic.
    #[test]
    fn vcpu_thread_kick_skips_ie_when_alive_false() {
        use std::sync::Barrier;
        // Register the SIGRTMIN handler before any `kick()` runs.
        // Default disposition for realtime signals is "terminate
        // process", and `kick()` calls `pthread_kill(tid, SIGRTMIN)`
        // — without a registered handler the test would die with
        // SIGRTMIN. Idempotent across repeated calls (sigaction is
        // process-wide).
        register_vcpu_signal_handler();
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
            llc_cores: None,
        };
        let mut vm = kvm::KtstrKvm::new(topo, 64, false).unwrap();
        let run_size = vm.vm_fd.run_size();
        let (probe_vcpu, ie) = ImmediateExitVcpu::new(vm.vcpus.remove(0), true, run_size).unwrap();
        let ie = ie.unwrap();
        // Spawn a dummy thread we can hand into a `JoinHandle<VcpuFd>`.
        // The thread parks on a barrier — kick() fires the signal at
        // it; the signal-handler default for SIGRTMIN is no-op-with-
        // EINTR, so the thread is unaffected. After we return from
        // kick(), drop the barrier and let it exit.
        let barrier = Arc::new(Barrier::new(2));
        let barrier_thread = barrier.clone();
        let handle = std::thread::Builder::new()
            .name("kick-test-stub".into())
            .spawn(move || {
                barrier_thread.wait();
                probe_vcpu.into_inner()
            })
            .unwrap();
        let exited = Arc::new(AtomicBool::new(false));
        let exit_evt = Arc::new(EventFd::new(EFD_NONBLOCK).unwrap());
        let alive = Arc::new(AtomicBool::new(false));
        let vt = VcpuThread {
            handle,
            exited,
            immediate_exit: Some(ie),
            exit_evt,
            alive,
            schedstat_at_exit: Arc::new(std::sync::Mutex::new(None)),
        };
        // Sanity: byte starts at 0 and alive is false — the test's
        // pre-condition.
        // Note: the spawned VcpuFd is moved into the closure above,
        // so we read the byte through the same shared `ie` we
        // captured before the move (handle dereferences the same
        // MAP_SHARED page).
        let read_byte = || vt.immediate_exit.as_ref().unwrap().read_byte();
        assert_eq!(read_byte(), Some(0));
        vt.kick();
        // alive=false ⇒ ie.set(1) is gated off ⇒ byte stays 0.
        assert_eq!(
            read_byte(),
            Some(0),
            "kick() should skip an unnecessary IE write after the panic edge",
        );
        // Release the stub and drain.
        barrier.wait();
        let _ = vt.handle.join();
    }

    /// Counterpart pinning the kick semantics when alive is `true`:
    /// the byte is written and observable. Together with
    /// `vcpu_thread_kick_skips_ie_when_alive_false` this fully
    /// pins the gate's truth table.
    #[test]
    fn vcpu_thread_kick_writes_ie_when_alive_true() {
        use std::sync::Barrier;
        register_vcpu_signal_handler();
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
            llc_cores: None,
        };
        let mut vm = kvm::KtstrKvm::new(topo, 64, false).unwrap();
        let run_size = vm.vm_fd.run_size();
        let (probe_vcpu, ie) = ImmediateExitVcpu::new(vm.vcpus.remove(0), true, run_size).unwrap();
        let ie = ie.unwrap();
        let barrier = Arc::new(Barrier::new(2));
        let barrier_thread = barrier.clone();
        let handle = std::thread::Builder::new()
            .name("kick-test-stub-alive".into())
            .spawn(move || {
                barrier_thread.wait();
                probe_vcpu.into_inner()
            })
            .unwrap();
        let exited = Arc::new(AtomicBool::new(false));
        let exit_evt = Arc::new(EventFd::new(EFD_NONBLOCK).unwrap());
        let alive = Arc::new(AtomicBool::new(true));
        let vt = VcpuThread {
            handle,
            exited,
            immediate_exit: Some(ie),
            exit_evt,
            alive,
            schedstat_at_exit: Arc::new(std::sync::Mutex::new(None)),
        };
        let read_byte = || vt.immediate_exit.as_ref().unwrap().read_byte();
        assert_eq!(read_byte(), Some(0));
        vt.kick();
        assert_eq!(
            read_byte(),
            Some(1),
            "kick() must write ie.set(1) when alive == true",
        );
        barrier.wait();
        let _ = vt.handle.join();
    }

    // -- RT scheduling tests --

    #[test]
    fn set_rt_priority_applies_when_capable() {
        // Probe CAP_SYS_NICE via a direct sched_setscheduler call
        // first: RT policies require the capability, and CI
        // containers frequently drop it. If the probe fails, skip
        // rather than fail — the permission check is the feature
        // under test.
        let param = libc::sched_param { sched_priority: 1 };
        let rc = unsafe { libc::sched_setscheduler(0, libc::SCHED_FIFO, &param) };
        if rc != 0 {
            skip!("no CAP_SYS_NICE capability available");
        }
        let policy = unsafe { libc::sched_getscheduler(0) };
        assert_eq!(policy, libc::SCHED_FIFO);
        let mut out_param: libc::sched_param = unsafe { std::mem::zeroed() };
        unsafe { libc::sched_getparam(0, &mut out_param) };
        assert_eq!(out_param.sched_priority, 1);
        // Restore SCHED_OTHER so later tests in the same nextest
        // process don't inherit this thread's RT policy.
        let restore = libc::sched_param { sched_priority: 0 };
        unsafe { libc::sched_setscheduler(0, libc::SCHED_OTHER, &restore) };
    }

    /// `set_rt_priority` emits a `tracing::warn!` with the
    /// "need CAP_SYS_NICE" substring when `sched_setscheduler`
    /// returns an error — the warn-and-proceed invariant that keeps
    /// vCPU threads running in unprivileged containers with the
    /// default scheduling policy instead of failing the VM.
    ///
    /// Captures tracing output via `tracing_test::traced_test` so the
    /// assertion observes the actual warn event (not just "the call
    /// did not panic"). Runs ONLY when the test process lacks
    /// CAP_SYS_NICE — if the capability is present, the success
    /// branch fires instead and the warn is never emitted, leaving
    /// nothing to assert; in that case we restore SCHED_OTHER on
    /// the probe thread and skip.
    #[test]
    #[tracing_test::traced_test]
    fn set_rt_priority_warns_without_cap() {
        // Probe CAP_SYS_NICE: if we CAN set SCHED_FIFO, the test
        // can't exercise the warn path. Restore SCHED_OTHER and
        // skip — we can't observe the warn event without actually
        // failing the syscall.
        let probe = libc::sched_param { sched_priority: 1 };
        let rc = unsafe { libc::sched_setscheduler(0, libc::SCHED_FIFO, &probe) };
        if rc == 0 {
            // Restore SCHED_OTHER so later tests don't inherit RT.
            let restore = libc::sched_param { sched_priority: 0 };
            unsafe { libc::sched_setscheduler(0, libc::SCHED_OTHER, &restore) };
            skip!("CAP_SYS_NICE present — cannot exercise warn path");
        }
        // Now we know the syscall will fail. Call set_rt_priority
        // and assert the warn event fires with the expected
        // substring. `logs_contain` is injected into the test by
        // the `#[traced_test]` macro and scans the per-test tracing
        // buffer.
        set_rt_priority(1, "test-thread");
        assert!(
            logs_contain("need CAP_SYS_NICE"),
            "warn event must include the 'need CAP_SYS_NICE' hint \
             so operators reading stderr know what permission to \
             grant",
        );
        assert!(
            logs_contain("SCHED_FIFO"),
            "warn event must name the policy whose attachment failed",
        );
        assert!(
            logs_contain("test-thread"),
            "warn event must name the label so operators can attribute \
             the warning to a specific vCPU / monitor / watchdog thread",
        );
    }
}
