//! Per-thread jemalloc TLS counter probe — the engine module.
//!
//! Reads the jemalloc per-thread TSD counters
//! (`tsd_s.thread_allocated` / `tsd_s.thread_deallocated`) out of a
//! running jemalloc-linked process by attaching via ptrace, reading
//! the thread pointer, computing the TLS address from
//! DWARF-resolved offsets, and pulling the counters with
//! `process_vm_readv`. jemalloc maintains the per-thread TSD
//! counters unconditionally on its alloc/dalloc fast and slow
//! paths, so attaching late does not lose data — every read is
//! cumulative from thread creation.
//!
//! Two consumers share this single source of truth for per-thread
//! counter math:
//!
//! - The standalone CLI binary at `src/bin/jemalloc_probe.rs`
//!   source-shares this file via `#[path = "../host_thread_probe.rs"]`
//!   rather than `use ktstr::host_thread_probe`, to avoid linking
//!   the ktstr library (which would pull in its early-dispatch ctor
//!   and bloat the initramfs image used by integration tests). The
//!   compiled bytes match exactly; the bin invokes
//!   [`attach_jemalloc`] + [`probe_thread_with_cache`] directly.
//! - The ctprof capture pipeline at [`ktstr::ctprof`]
//!   consumes the engine through the in-crate `pub` API, calling
//!   [`attach_jemalloc_at`] for each probed tgid and
//!   [`probe_thread`] for each tid behind the
//!   `try_attach_probe_for_tgid_at` / `probe_thread_recording`
//!   wrappers that thread the per-snapshot `ProbeSummary` tally.
//!
//! # Scope
//!
//! - Linux only. ptrace + procfs gating; portable to BSDs would
//!   require a different attach mechanism.
//! - x86_64 and aarch64 host (and target — ptrace is same-arch). A
//!   probe built for x86_64 cannot probe an aarch64 target.
//! - Static-linked jemalloc only. The TLS symbol must live in the
//!   target's main executable (the binary `/proc/<pid>/exe` points
//!   at). Dynamic-TLS symbols in DSOs require a DTV walk that this
//!   module does not implement; [`AttachError::JemallocInDso`] is
//!   the closed bucket for that case so callers can surface a
//!   targeted remediation.
//! - Requires DWARF debuginfo reachable from the target ELF —
//!   either inline `.debug_info` on the executable itself, or a
//!   paired external `.debug` file discovered via
//!   `.gnu_debuglink` / `NT_GNU_BUILD_ID`.
//!
//! # Privilege
//!
//! `ptrace(PTRACE_SEIZE)` must succeed against the target. Under
//! `kernel.yama.ptrace_scope=0` (the KVM-guest default and the
//! happy path for ktstr's own integration tests) any same-uid
//! process attaches. Under `=1` (Debian/Ubuntu host default) the
//! tracer must be an ancestor of the target, the target must have
//! opted in via `prctl(PR_SET_PTRACER)`, or the tracer must carry
//! `CAP_SYS_PTRACE`. `=2` and `=3` raise the bar further; this
//! module surfaces the underlying `EPERM` / `ESRCH` through
//! [`ProbeError::PtraceSeize`] without papering over it.
//!
//! # Self-attach
//!
//! `PTRACE_SEIZE` rejects a process attaching to its own tgid.
//! [`attach_jemalloc`] does NOT pre-filter for self-pid because the
//! capture pipeline already excludes the calling process from its
//! per-tgid walk. A direct caller passing `std::process::id()` will
//! see the EPERM surface through [`AttachError`] — informative
//! enough for the caller's own self-detection if needed.

use std::borrow::Cow;
use std::fs;
use std::io::IoSliceMut;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use gimli::{AttributeValue, EndianSlice, LittleEndian, Reader, Unit};
use goblin::elf::Elf;
use nix::sys::ptrace;
use nix::sys::ptrace::Options;
#[cfg(target_arch = "x86_64")]
use nix::sys::ptrace::regset::NT_PRSTATUS;
use nix::sys::uio::{RemoteIoVec, process_vm_readv};
use nix::sys::wait::{WaitPidFlag, WaitStatus, waitpid};
use nix::unistd::Pid;

// ---------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------

/// Per-thread jemalloc counter snapshot. Both fields are cumulative
/// u64 byte counts since thread creation — they can only grow over
/// the lifetime of a thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThreadCounters {
    pub allocated_bytes: u64,
    pub deallocated_bytes: u64,
}

/// Attached probe context: the resolved jemalloc TLS symbol + the
/// DWARF-derived `tsd_s` field offsets. Cheap to clone (a few u64s
/// plus a PathBuf); pass by reference to keep the per-thread fast
/// path zero-allocation.
///
/// Construct via [`attach_jemalloc`]; the constructor performs the
/// expensive ELF parse + DWARF walk so per-thread reads are
/// amortized across every tid that shares a tgid.
#[derive(Debug, Clone)]
pub struct JemallocProbe {
    symbol: TsdTlsSymbol,
    offsets: CounterOffsets,
}

/// Pre-probe failures that prevent any per-thread read. Distinct
/// variants so callers can decide whether to surface the failure
/// (e.g. arch mismatch) or skip silently (e.g. target is not
/// jemalloc-linked at all).
#[derive(Debug)]
#[non_exhaustive]
pub enum AttachError {
    /// `/proc/<pid>` does not exist or is unreachable.
    PidMissing(anyhow::Error),
    /// `readlink(/proc/<pid>/exe)` failed — typically the target
    /// exited mid-attach, or permission is denied under the
    /// active ptrace policy.
    ReadlinkFailure(anyhow::Error),
    /// `/proc/<pid>/maps` could not be read.
    MapsReadFailure(anyhow::Error),
    /// No jemalloc TSD TLS symbol turned up in any executable
    /// mapping. The target is either not jemalloc-linked at all,
    /// or links a jemalloc build whose symbol-name prefix is not
    /// in the recognized set.
    JemallocNotFound(anyhow::Error),
    /// jemalloc TSD TLS symbol was found, but in a shared object
    /// rather than the target's main executable. Static-TLS
    /// addressing only reaches the main executable's TLS image;
    /// dynamic-TLS lookups in DSOs need a DTV walk the engine
    /// does not implement.
    JemallocInDso(anyhow::Error),
    /// Target ELF's architecture does not match the probe binary's.
    /// ptrace is same-arch; a cross-arch attach cannot read the
    /// thread pointer.
    ArchMismatch(anyhow::Error),
    /// DWARF parse failed — target was stripped without external
    /// debuginfo, debuglink CRC mismatched, or `tsd_s` / its
    /// member fields were absent from the DWARF tree.
    DwarfParseFailure(anyhow::Error),
}

impl AttachError {
    /// Returns the underlying `anyhow::Error` for diagnostics.
    pub fn source(&self) -> &anyhow::Error {
        match self {
            Self::PidMissing(e)
            | Self::ReadlinkFailure(e)
            | Self::MapsReadFailure(e)
            | Self::JemallocNotFound(e)
            | Self::JemallocInDso(e)
            | Self::ArchMismatch(e)
            | Self::DwarfParseFailure(e) => e,
        }
    }

    /// Stable token for log output — single-word identifier
    /// downstream consumers can match against. Adding a new
    /// variant is always safe; renaming a token breaks consumer
    /// matching, so the variant→tag mapping is the wire contract.
    pub fn tag(&self) -> &'static str {
        match self {
            Self::PidMissing(_) => "pid-missing",
            Self::ReadlinkFailure(_) => "readlink-failure",
            Self::MapsReadFailure(_) => "maps-read-failure",
            Self::JemallocNotFound(_) => "jemalloc-not-found",
            Self::JemallocInDso(_) => "jemalloc-in-dso",
            Self::ArchMismatch(_) => "arch-mismatch",
            Self::DwarfParseFailure(_) => "dwarf-parse-failure",
        }
    }
}

impl std::fmt::Display for AttachError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {:#}", self.tag(), self.source())
    }
}

impl std::error::Error for AttachError {}

/// Per-thread probe failures. The variant carries a structural
/// classifier; the wrapped `anyhow::Error` carries the human-
/// readable rendering (errno text, address details, etc.).
#[derive(Debug)]
#[non_exhaustive]
pub enum ProbeError {
    /// `ptrace(PTRACE_SEIZE)` failed. Common causes: tid exited
    /// between enumeration and attach (ESRCH), tracer lacks
    /// privilege (EPERM), or another tracer is already attached
    /// (EBUSY).
    ///
    /// In the ctprof capture pipeline, ESRCH from a
    /// race-with-thread-death is the dominant production case:
    /// `iter_task_ids_at` enumerates the tgid's task directory
    /// upfront, then capture probes each tid in turn. Threads
    /// that exit between enumeration and the per-tid attach are
    /// the rule, not the exception, on a busy host with
    /// short-lived helper threads. Capture absorbs this variant
    /// into the absent-counter contract so a thread that
    /// vanished mid-snapshot lands at allocated_bytes=0 rather
    /// than failing the whole capture.
    PtraceSeize(anyhow::Error),
    /// `ptrace(PTRACE_INTERRUPT)` failed after a successful seize.
    PtraceInterrupt(anyhow::Error),
    /// `waitpid` after `PTRACE_INTERRUPT` returned an unexpected
    /// status, or a syscall error.
    Waitpid(anyhow::Error),
    /// `ptrace(PTRACE_GETREGSET, ...)` failed when reading the
    /// thread pointer.
    GetRegset(anyhow::Error),
    /// `process_vm_readv` failed or returned a short read.
    ProcessVmReadv(anyhow::Error),
    /// TLS address arithmetic overflowed or underflowed — the
    /// target's TLS layout is malformed in a way that should not
    /// occur on a well-formed jemalloc binary.
    TlsArithmetic(anyhow::Error),
}

impl ProbeError {
    pub fn source(&self) -> &anyhow::Error {
        match self {
            Self::PtraceSeize(e)
            | Self::PtraceInterrupt(e)
            | Self::Waitpid(e)
            | Self::GetRegset(e)
            | Self::ProcessVmReadv(e)
            | Self::TlsArithmetic(e) => e,
        }
    }

    pub fn tag(&self) -> &'static str {
        match self {
            Self::PtraceSeize(_) => "ptrace-seize",
            Self::PtraceInterrupt(_) => "ptrace-interrupt",
            Self::Waitpid(_) => "waitpid",
            Self::GetRegset(_) => "get-regset",
            Self::ProcessVmReadv(_) => "process-vm-readv",
            Self::TlsArithmetic(_) => "tls-arithmetic",
        }
    }
}

impl std::fmt::Display for ProbeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {:#}", self.tag(), self.source())
    }
}

impl std::error::Error for ProbeError {}

/// Resolve jemalloc symbols + DWARF offsets for `pid`. The expensive
/// one-time work for a process: ELF parse, symbol lookup, DWARF
/// walk. Per-thread reads call [`probe_thread`] with the returned
/// [`JemallocProbe`].
///
/// `pid` must be live (`/proc/<pid>` exists) and must point at a
/// statically-linked jemalloc executable with reachable DWARF.
/// Failures are classified into [`AttachError`] variants so the
/// caller can decide which are worth surfacing — `JemallocNotFound`
/// is the expected outcome for the vast majority of system processes
/// and is usually skipped silently; `DwarfParseFailure` against a
/// known-jemalloc target is worth logging.
#[allow(dead_code)]
pub fn attach_jemalloc(pid: i32) -> std::result::Result<JemallocProbe, AttachError> {
    attach_jemalloc_at(Path::new("/proc"), pid)
}

/// `proc_root`-parameterised variant of [`attach_jemalloc`]. Lets
/// tests stage a synthetic `/proc/<pid>/{exe,maps}` shape under a
/// tempdir without touching the real procfs. Production code uses
/// the default-root [`attach_jemalloc`] wrapper above; this variant
/// is exposed as `pub` so downstream harnesses that stand up
/// alternate procfs shapes (containers with mount namespaces,
/// pid-namespaced probes, fuzz fixtures) can drive the same
/// detection / attach codepath without duplicating it.
///
/// Failure classification is identical to [`attach_jemalloc`]:
/// non-existent `<proc_root>/<pid>` directory → `PidMissing`;
/// unreadable `<proc_root>/<pid>/exe` symlink → `ReadlinkFailure`;
/// unreadable `<proc_root>/<pid>/maps` → `MapsReadFailure`; jemalloc
/// TLS symbol absent → `JemallocNotFound`; symbol present in a DSO
/// rather than the main exe → `JemallocInDso`; etc.
pub fn attach_jemalloc_at(
    proc_root: &Path,
    pid: i32,
) -> std::result::Result<JemallocProbe, AttachError> {
    let pid_dir = proc_root.join(pid.to_string());
    if !pid_dir.exists() {
        return Err(AttachError::PidMissing(anyhow!(
            "{} does not exist",
            pid_dir.display(),
        )));
    }
    let (symbol, offsets) = find_jemalloc_via_maps_at(proc_root, pid)?;
    Ok(JemallocProbe { symbol, offsets })
}

/// Maximum wall-clock the probe is willing to wait for the target
/// thread to deliver its post-`PTRACE_INTERRUPT` stop event before
/// giving up. Bounded so a target in an unexpected state (another
/// tracer attached, kernel scheduling delay, the rare "interrupt
/// signal not delivered" case observed in the wild) does not
/// produce an unbounded waitpid hang inside the capture path.
///
/// 250 ms covers a generous slack over the typical kernel delay
/// for SIGSTOP delivery on a non-pathologically-loaded host (sub-
/// millisecond) plus the round-trip overhead of waitpid's poll
/// path. If the stop has not landed by then, surface
/// [`ProbeError::Waitpid`] so the caller can move on rather than
/// blocking the rest of the snapshot.
const WAIT_FOR_STOP_TIMEOUT: Duration = Duration::from_millis(250);

/// Polling interval for the non-blocking waitpid loop. 1 ms is
/// short enough that a fast stop is observed near-immediately and
/// long enough that the polling overhead does not dominate the
/// per-thread cost on the slow path.
const WAIT_FOR_STOP_POLL_INTERVAL: Duration = Duration::from_millis(1);

/// Read one thread's `thread_allocated` and `thread_deallocated`
/// counters via the cold-path ptrace dance: SEIZE → INTERRUPT →
/// bounded-poll-waitpid → GETREGSET → process_vm_readv → DETACH.
///
/// Thin wrapper over [`probe_thread_with_cache`] passing
/// `cached_tp = None` (always traverses the slow path). Callers that
/// take a single one-shot snapshot per tid stay on this entry point;
/// callers running repeat snapshots against the same tid use
/// [`probe_thread_with_cache`] to skip the ptrace dance after the
/// first observation.
pub fn probe_thread(
    probe: &JemallocProbe,
    tid: i32,
) -> std::result::Result<ThreadCounters, ProbeError> {
    probe_thread_with_cache(probe, tid, None).map(|(c, _)| c)
}

/// Read one thread's `thread_allocated` and `thread_deallocated`
/// counters with optional thread-pointer caching.
///
/// - `cached_tp = None` (cache miss / first observation): runs the
///   full ptrace dance (SEIZE → INTERRUPT → bounded-poll-waitpid →
///   GETREGSET → process_vm_readv → DETACH) and returns the
///   counters plus the observed TP. Callers cache the TP keyed on
///   tid for subsequent snapshots.
/// - `cached_tp = Some(tp)` (cache hit / snapshots 2..N against the
///   same tid): skips the ptrace dance entirely and reads counters
///   directly via `process_vm_readv` against the cached TP. Returns
///   the counters and the same `tp` (echoed for caller convenience —
///   one signature serves both arms).
///
/// Detach is unconditional via the [`ScopeDetach`] guard on the
/// cache-miss arm: any fallible step after seize still detaches on
/// scope exit so the target thread is never left stopped. The
/// cache-hit arm has no ptrace session so no detach is needed.
///
/// The caller is responsible for tid liveness — a tid that exited
/// between the caller's enumeration and this call surfaces as
/// [`ProbeError::PtraceSeize`] with ESRCH on the cache-miss arm, or
/// as [`ProbeError::ProcessVmReadv`] with ESRCH on the cache-hit
/// arm; the engine does not retry.
///
/// Bounded waitpid (cache-miss only): the post-interrupt stop is
/// awaited via [`WaitPidFlag::WNOHANG`] in a poll loop bounded by
/// [`WAIT_FOR_STOP_TIMEOUT`]. A target whose stop event does not
/// land within the budget surfaces as [`ProbeError::Waitpid`]
/// rather than blocking the snapshot indefinitely.
///
/// **Cache-hit consistency caveat**: the `process_vm_readv` on the
/// cache-hit arm races the target's ongoing counter updates. Each
/// naturally-aligned u64 load is torn-read-free on x86_64/aarch64,
/// so `allocated_bytes` and `deallocated_bytes` individually remain
/// consistent — but the pair is sampled inside one remote read of
/// the 24-byte counter span, and the target can mutate the span
/// between the two sub-loads. Cumulative monotonic counters tolerate
/// that skew (no `allocated >= deallocated` invariant rides on the
/// pair snapshot). Exotic runtimes that relocate TLS mid-lifetime
/// (Wine, some Go builds, hand-rolled libc, direct
/// `arch_prctl(ARCH_SET_FS)` / `WRFSBASE` calls under
/// `CR4.FSGSBASE`) would desync the cache; callers controlling tid
/// enumeration are expected to evict cache entries for tids that
/// drop out of the live enumeration so a kernel-recycled tid does
/// not hit a stale entry from a prior thread.
pub fn probe_thread_with_cache(
    probe: &JemallocProbe,
    tid: i32,
    cached_tp: Option<u64>,
) -> std::result::Result<(ThreadCounters, u64), ProbeError> {
    let pid = Pid::from_raw(tid);

    // `_detach` lives only on the slow path — the fast path never
    // seizes, so it has nothing to detach. Bound on both arms with
    // the uniform `Option<ScopeDetach>` type so the guard's
    // Drop-on-scope-exit semantic spans the entire fn body, including
    // the `read_counters_at_thread_pointer` call below — matching the
    // pre-cache behavior where the slow-path detach was held across
    // the counter read.
    let (thread_pointer, _detach) = match cached_tp {
        Some(tp) => (tp, None),
        None => {
            ptrace::seize(pid, Options::empty()).map_err(|e| {
                ProbeError::PtraceSeize(anyhow!("ptrace(PTRACE_SEIZE) on tid {tid}: {e}"))
            })?;
            // Construct the detach guard IMMEDIATELY after a
            // successful seize so any subsequent fallible step still
            // detaches on scope exit. Drop runs even on panic.
            let guard = ScopeDetach(tid);

            ptrace::interrupt(pid).map_err(|e| {
                ProbeError::PtraceInterrupt(anyhow!("ptrace(PTRACE_INTERRUPT) on tid {tid}: {e}"))
            })?;
            wait_for_stop(pid, tid)?;

            let tp = arch::read_thread_pointer_ptrace(pid).map_err(|e| {
                ProbeError::GetRegset(anyhow!(
                    "ptrace(PTRACE_GETREGSET, {}) on tid {tid}: {e}",
                    arch::REGSET_NAME,
                ))
            })?;
            (tp, Some(guard))
        }
    };

    let counters =
        read_counters_at_thread_pointer(thread_pointer, &probe.symbol, &probe.offsets, tid)?;
    Ok((counters, thread_pointer))
}

/// Bounded-poll waitpid loop. Returns `Ok(())` when the target is
/// observed in the expected `Stopped` / `PtraceEvent` state, or
/// surfaces [`ProbeError::Waitpid`] on syscall error / unexpected
/// status / timeout. Factored out of [`probe_thread`] so the
/// polling discipline has one source of truth.
fn wait_for_stop(pid: Pid, tid: i32) -> std::result::Result<(), ProbeError> {
    let deadline = Instant::now() + WAIT_FOR_STOP_TIMEOUT;
    loop {
        match waitpid(pid, Some(WaitPidFlag::WNOHANG)) {
            Ok(WaitStatus::Stopped(_, _) | WaitStatus::PtraceEvent(_, _, _)) => return Ok(()),
            // WNOHANG returns StillAlive when no status is pending
            // yet. Spin-poll with a short sleep until the deadline
            // bites. Other Ok variants (Exited, Signaled, etc.)
            // mean the target is gone — surface as Waitpid so the
            // caller distinguishes "race-exit during attach" from
            // a successful stop.
            Ok(WaitStatus::StillAlive) => {
                if Instant::now() >= deadline {
                    return Err(ProbeError::Waitpid(anyhow!(
                        "waitpid on tid {tid} did not observe the post-interrupt \
                         stop within {:?}; the target may have a conflicting \
                         tracer or the kernel delayed signal delivery beyond \
                         the budget",
                        WAIT_FOR_STOP_TIMEOUT,
                    )));
                }
                std::thread::sleep(WAIT_FOR_STOP_POLL_INTERVAL);
            }
            Ok(other) => {
                return Err(ProbeError::Waitpid(anyhow!(
                    "waitpid on tid {tid} returned unexpected status: {other:?}"
                )));
            }
            Err(e) => return Err(ProbeError::Waitpid(anyhow!("waitpid on tid {tid}: {e}"))),
        }
    }
}

// ---------------------------------------------------------------------
// Symbol-name registry
// ---------------------------------------------------------------------

/// Suffix-match predicate for jemalloc's per-thread TSD TLS symbol.
///
/// jemalloc's build system emits the per-thread TSD as `tsd_tls`
/// optionally prefixed by whatever the `--with-jemalloc-prefix=…`
/// flag specifies. Observed prefixes in the wild:
/// - bare `tsd_tls` (unprefixed builds, older jemalloc versions)
/// - `je_tsd_tls` (default `--with-jemalloc-prefix=je_`)
/// - `_rjem_je_tsd_tls` (what `tikv-jemallocator-sys` bakes in to
///   keep the Rust global allocator from colliding with system
///   libc malloc symbols at link time)
/// - `jemalloc_je_tsd_tls` (`--with-jemalloc-prefix=jemalloc_`,
///   seen in some large statically-linked binaries)
///
/// Maintaining an exhaustive list churned every time a downstream
/// project picked a new prefix. The suffix match accepts any
/// symbol that is exactly `tsd_tls` OR has a non-empty prefix
/// followed by `_tsd_tls`. The non-empty-prefix gate (rather than
/// a bare `ends_with("_tsd_tls")`) keeps the predicate from
/// accepting the degenerate `_tsd_tls` (empty prefix, just the
/// separator) — no real jemalloc build emits that, and rejecting
/// it preserves the "any clean prefix" contract the user-facing
/// docs imply. The trailing `_` separator avoids false positives
/// like `mytsd_tls` (no underscore separator) or
/// `something_tsd_tls_v2` (no trailing terminator).
///
/// False-positive defense: a name match alone is not the only
/// gate. The downstream DWARF walk insists on `struct tsd_s` with
/// the `cant_access_tsd_items_directly_use_a_getter_or_setter_*`
/// mangled member names, which are jemalloc-specific. A non-
/// jemalloc symbol whose name happens to satisfy this predicate
/// would fail the DWARF walk and surface as
/// `AttachError::DwarfParseFailure`, not as a silent garbage
/// read.
fn is_jemalloc_tsd_tls_symbol(name: &str) -> bool {
    if name == "tsd_tls" {
        return true;
    }
    // `name.ends_with("_tsd_tls")` is true for `_tsd_tls` (length
    // 8 == len("_tsd_tls")) but the prefix would be empty. Require
    // strictly more characters so the prefix is non-empty.
    name.len() > "_tsd_tls".len() && name.ends_with("_tsd_tls")
}

/// DWARF struct name for jemalloc's per-thread state.
const TSD_STRUCT_NAME: &str = "tsd_s";

/// jemalloc mangles `tsd_s` field names with this fixed prefix via
/// the `TSD_MANGLE` macro so direct field access in C code triggers
/// a compile-time symbol-lookup failure, forcing callers to go
/// through the `tsd_*_get` / `tsd_*_set` accessor macros. The DWARF
/// emitted by the compiler carries the mangled names verbatim — the
/// engine matches on the full prefixed name to avoid accidental
/// false positives on substring overlaps like
/// `thread_allocated_last_event_key`.
macro_rules! tsd_mangle_prefix {
    () => {
        "cant_access_tsd_items_directly_use_a_getter_or_setter_"
    };
}

#[allow(dead_code)]
const TSD_MANGLE_PREFIX: &str = tsd_mangle_prefix!();
const ALLOCATED_FIELD: &str = concat!(tsd_mangle_prefix!(), "thread_allocated");
const DEALLOCATED_FIELD: &str = concat!(tsd_mangle_prefix!(), "thread_deallocated");

// ---------------------------------------------------------------------
// Per-arch primitives
// ---------------------------------------------------------------------

mod arch {
    use super::*;

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    compile_error!(
        "host_thread_probe supports only x86_64 and aarch64 targets; \
         ptrace is same-arch and the TLS address math is arch-specific \
         (Variant II on x86_64, Variant I on aarch64)"
    );

    #[cfg(target_arch = "x86_64")]
    pub const EXPECTED_E_MACHINE: u16 = goblin::elf::header::EM_X86_64;
    #[cfg(target_arch = "aarch64")]
    pub const EXPECTED_E_MACHINE: u16 = goblin::elf::header::EM_AARCH64;

    #[cfg(target_arch = "x86_64")]
    pub const ARCH_NAME: &str = "x86_64";
    #[cfg(target_arch = "aarch64")]
    pub const ARCH_NAME: &str = "aarch64";

    #[cfg(target_arch = "x86_64")]
    pub const REGSET_NAME: &str = "NT_PRSTATUS";
    #[cfg(target_arch = "aarch64")]
    pub const REGSET_NAME: &str = "NT_ARM_TLS";

    /// `NT_ARM_TLS` regset number from `linux/include/uapi/linux/elf.h`.
    /// `nix` does not expose this regset — its `RegisterSetValue` enum
    /// is closed and only carries a curated subset — so the aarch64
    /// read path calls `libc::ptrace` directly with the raw value.
    #[cfg(target_arch = "aarch64")]
    pub const NT_ARM_TLS: libc::c_int = 0x401;

    /// Read the stopped target thread's TP (thread pointer) via ptrace.
    ///
    /// - x86_64: `PTRACE_GETREGSET, NT_PRSTATUS` — returns
    ///   `user_regs_struct.fs_base`.
    /// - aarch64: `PTRACE_GETREGSET, NT_ARM_TLS` — returns
    ///   `[tpidr_el0, tpidr2_el0]` on kernels with TPIDR2 support, or a
    ///   single `tpidr_el0` on older kernels. We request only the
    ///   first 8 bytes via the iovec's `iov_len`, so the read is
    ///   version-stable across both.
    #[cfg(target_arch = "x86_64")]
    pub fn read_thread_pointer_ptrace(pid: Pid) -> std::result::Result<u64, nix::errno::Errno> {
        let regs = ptrace::getregset::<NT_PRSTATUS>(pid)?;
        Ok(regs.fs_base)
    }

    #[cfg(target_arch = "aarch64")]
    pub fn read_thread_pointer_ptrace(pid: Pid) -> std::result::Result<u64, nix::errno::Errno> {
        let mut tpidr: u64 = 0;
        let mut iov = libc::iovec {
            iov_base: (&mut tpidr as *mut u64).cast::<libc::c_void>(),
            iov_len: std::mem::size_of::<u64>(),
        };
        // SAFETY: libc::ptrace is variadic; the addresses passed
        // must be valid for the duration of the call. iov.iov_base
        // points at a stack u64 and &mut iov points at a stack
        // iovec — both live for the entire call. No concurrent
        // reads or writes of either `tpidr` or `iov` may occur
        // during the syscall: `tpidr` and `iov` are stack-local to
        // this fn (no aliases handed out), the unsafe block is the
        // sole writer for the syscall window, and the kernel
        // performs the iov-driven write before returning. The
        // post-syscall `iov.iov_len` read on the next line and the
        // `tpidr` read in the trailing `Ok(tpidr)` therefore
        // observe a fully-ordered, single-threaded write — no
        // atomic / fence / volatile required.
        let res = unsafe {
            libc::ptrace(
                libc::PTRACE_GETREGSET,
                pid.as_raw(),
                NT_ARM_TLS as libc::c_long,
                &mut iov as *mut libc::iovec,
            )
        };
        if res == -1 {
            return Err(nix::errno::Errno::last());
        }
        // PTRACE_GETREGSET writes the actual byte count back into
        // iov.iov_len. A short write means the kernel emitted fewer
        // bytes than our u64 slot — surface EIO so the caller's
        // GetRegset arm reports the real cause instead of a
        // downstream malformed-address failure.
        if iov.iov_len < std::mem::size_of::<u64>() {
            return Err(nix::errno::Errno::EIO);
        }
        Ok(tpidr)
    }
}

// ---------------------------------------------------------------------
// ELF + DWARF resolution
// ---------------------------------------------------------------------

/// Thread-local symbol lookup result — enough to compute the
/// per-thread address of the TLS image containing jemalloc's
/// `tsd_tls`.
#[derive(Debug, Clone)]
struct TsdTlsSymbol {
    /// Absolute path of the ELF containing the symbol.
    elf_path: PathBuf,
    /// `st_value` of the symbol — offset within the TLS image.
    st_value: u64,
    /// `round_up(PT_TLS.p_memsz, PT_TLS.p_align)` — Variant II's
    /// TP-to-TLS-image delta on x86_64.
    tls_image_aligned_size: u64,
    /// Raw `PT_TLS.p_align` value — Variant I (aarch64) needs it
    /// to compute `round_up(TCB_SIZE_AARCH64, p_align)`.
    p_align: u64,
    /// ELF `e_machine` for arch matching.
    e_machine: u16,
}

/// Offsets of the two counters inside `struct tsd_s`, resolved from
/// DWARF. Computed once per ELF; shared across every thread of a
/// tgid via the [`JemallocProbe`] struct.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CounterOffsets {
    thread_allocated: u64,
    thread_deallocated: u64,
}

impl CounterOffsets {
    /// Construct, enforcing `thread_allocated < thread_deallocated`.
    /// jemalloc's TSD_DATA_FAST block lays them out in that order
    /// with `thread_allocated_next_event_fast` between them. A
    /// reversed pair means the DWARF walk picked up a different
    /// struct or the layout has drifted; either way the combined-
    /// read math would underflow and read garbage, so fail fast.
    fn new(thread_allocated: u64, thread_deallocated: u64) -> Result<Self> {
        if thread_allocated >= thread_deallocated {
            bail!(
                "unexpected tsd_s layout: thread_allocated ({thread_allocated}) \
                 must precede thread_deallocated ({thread_deallocated}) per \
                 jemalloc TSD_DATA_FAST ordering",
            );
        }
        Ok(Self {
            thread_allocated,
            thread_deallocated,
        })
    }

    /// Byte span covering both counters plus the
    /// `thread_allocated_next_event_fast` u64 between them. The
    /// length of the remote iov for one `process_vm_readv` while
    /// the target thread is stopped.
    fn combined_read_span(&self) -> u64 {
        self.thread_deallocated + 8 - self.thread_allocated
    }
}

/// Locate jemalloc's TSD TLS symbol inside the given ELF. Returns
/// the symbol's `st_value` plus the PT_TLS-aligned image size
/// needed for TP-relative addressing. Symbol-name matching uses
/// the suffix predicate [`is_jemalloc_tsd_tls_symbol`] so any
/// `--with-jemalloc-prefix` variant resolves without an explicit
/// registry entry.
///
/// `.symtab` is searched first, then `.dynsym`. Within a table,
/// the first symbol whose name satisfies the predicate wins —
/// jemalloc's static-TLS layout gives every build exactly one
/// `tsd_tls` (the bare or prefixed form), so the first-hit policy
/// matches the realistic ELF shape.
fn find_tsd_tls(elf: &Elf<'_>, elf_path: &Path) -> Result<TsdTlsSymbol> {
    let e_machine = elf.header.e_machine;
    let (tls_image_aligned_size, p_align) = extract_pt_tls_layout(elf)?;

    let tables: [(&str, &goblin::elf::Symtab<'_>, &goblin::strtab::Strtab<'_>); 2] = [
        (".symtab", &elf.syms, &elf.strtab),
        (".dynsym", &elf.dynsyms, &elf.dynstrtab),
    ];
    for (_table_name, syms, strs) in tables {
        if let Some(st_value) = find_jemalloc_tsd_tls_in_table(syms, strs) {
            return Ok(TsdTlsSymbol {
                elf_path: elf_path.to_path_buf(),
                st_value,
                tls_image_aligned_size,
                p_align,
                e_machine,
            });
        }
    }

    Err(anyhow!(
        "no jemalloc TLS symbol (bare `tsd_tls` or any `<prefix>_tsd_tls`) \
         found in .symtab or .dynsym of {}",
        elf_path.display(),
    ))
}

/// Walk one symbol table's name index for the first symbol whose
/// name satisfies [`is_jemalloc_tsd_tls_symbol`]. Returns the
/// symbol's `st_value`, or `None` if no matching symbol exists in
/// this table.
fn find_jemalloc_tsd_tls_in_table(
    syms: &goblin::elf::Symtab<'_>,
    strs: &goblin::strtab::Strtab<'_>,
) -> Option<u64> {
    for sym in syms.iter() {
        if let Some(name) = strs.get_at(sym.st_name)
            && is_jemalloc_tsd_tls_symbol(name)
        {
            return Some(sym.st_value);
        }
    }
    None
}

/// Round `value` up to a multiple of `align`, returning `None` on
/// overflow. `align` must be a power of two (or zero, clamped to 1);
/// callers encoding the ELF power-of-two invariant rely on
/// `& !(align - 1)` rather than `% align`.
fn round_up_pow2(value: u64, align: u64) -> Option<u64> {
    let align = align.max(1);
    value.checked_add(align - 1).map(|v| v & !(align - 1))
}

/// Extract `(round_up(p_memsz, p_align), p_align)` from the ELF's
/// `PT_TLS` program header. The first is Variant II's TP-to-image
/// delta; the second feeds Variant I's TCB-to-image offset.
fn extract_pt_tls_layout(elf: &Elf<'_>) -> Result<(u64, u64)> {
    let tls_hdr = elf
        .program_headers
        .iter()
        .find(|ph| ph.p_type == goblin::elf::program_header::PT_TLS)
        .ok_or_else(|| anyhow!("ELF has no PT_TLS segment — target does not use static TLS"))?;
    debug_assert!(
        tls_hdr.p_align == 0 || tls_hdr.p_align.is_power_of_two(),
        "PT_TLS.p_align must be 0 or a power of two, got {}",
        tls_hdr.p_align,
    );
    let align = tls_hdr.p_align.max(1);
    let rounded = round_up_pow2(tls_hdr.p_memsz, align)
        .ok_or_else(|| anyhow!("PT_TLS size arithmetic overflow"))?;
    Ok((rounded, align))
}

/// Resolve the byte offsets of `thread_allocated` and
/// `thread_deallocated` inside `struct tsd_s` by walking DWARF on
/// the target ELF, or on an external debuginfo file discovered via
/// `.gnu_debuglink` / `NT_GNU_BUILD_ID` when the target is stripped.
#[allow(dead_code)]
fn resolve_field_offsets(elf_path: &Path) -> Result<CounterOffsets> {
    let data = fs::read(elf_path)
        .with_context(|| format!("re-read {} for DWARF inspection", elf_path.display()))?;
    let elf = Elf::parse(&data).with_context(|| format!("parse ELF {}", elf_path.display()))?;

    if section_is_populated(&elf, &data, ".debug_info") {
        return resolve_field_offsets_from_bytes(&data, elf_path);
    }

    let debuglink = read_gnu_debuglink(&elf, &data);
    let build_id = read_build_id(&elf, &data);
    let debuglink_name = debuglink.as_ref().map(|(n, _)| n.as_str());
    let build_id_hex = build_id.as_deref();

    let candidates = candidate_debuginfo_paths(elf_path, debuglink_name, build_id_hex);
    if candidates.is_empty() {
        // Distinguish "no pointer at all" from "pointer present but
        // rejected as unsafe": all three cases reach the
        // empty-candidates branch, but the operator's remediation
        // differs. Genuinely-absent debuginfo asks them to rebuild
        // with `-g` / install -dbg; an unsafe `.gnu_debuglink` or
        // build-id asks them to investigate why their toolchain
        // emitted malformed metadata (which only happens with a
        // hostile or corrupt ELF, since binutils itself rejects
        // path-bearing debuglink names per
        // `bfd/opncls.c::bfd_get_debug_link_info` and
        // `read_build_id` always emits clean lowercase hex).
        //
        // The order below matters for the diagnostic:
        // unsafe-debuglink and unsafe-build-id are independent
        // failure surfaces, so we surface whichever fired and let
        // the operator chase that lead first. A future caller that
        // poisons both at once will see only the debuglink message;
        // that's acceptable because both pointers being malformed
        // is one underlying cause (caller bypassed the parsers).
        if let Some(name) = debuglink_name
            && !debuglink_name_is_safe(name)
        {
            anyhow::bail!(
                "{} has no populated .debug_info and its \
                 .gnu_debuglink filename `{}` was rejected as unsafe \
                 (carries path separators, NUL bytes, or `.`/`..` \
                 traversal forms). A well-formed `.gnu_debuglink` \
                 holds only a bare basename. Inspect the target ELF \
                 with `objdump --section .gnu_debuglink` to confirm \
                 the on-disk content is what your toolchain emitted; \
                 if it is, the toolchain is broken or the ELF was \
                 tampered with.",
                elf_path.display(),
                name,
            );
        }
        if let Some(hex) = build_id_hex
            && hex.len() >= 2
            && !build_id_hex_is_safe(hex)
        {
            anyhow::bail!(
                "{} has no populated .debug_info and its \
                 NT_GNU_BUILD_ID hex `{}` was rejected as unsafe \
                 (must be even-length lowercase hex per \
                 `read_build_id`'s output format; uppercase, \
                 non-hex bytes, path separators, NUL bytes, or odd \
                 length all fail the gate). Inspect the target ELF \
                 with `readelf -n` to confirm the on-disk note is \
                 what your toolchain emitted; if it is, the \
                 toolchain is broken or the ELF was tampered with.",
                elf_path.display(),
                hex,
            );
        }
        anyhow::bail!(
            "{} has no populated .debug_info and carries neither a \
             .gnu_debuglink section nor an NT_GNU_BUILD_ID note — there \
             is no pointer to external debuginfo. Rebuild the target \
             with `-g`, ship a paired `.debug` file, or install the \
             distro's -dbg / -debuginfo package.",
            elf_path.display(),
        );
    }

    let mut tried: Vec<String> = Vec::new();
    for candidate in &candidates {
        let debug_data = match fs::read(candidate) {
            Ok(d) => d,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                tried.push(format!("{} (not found)", candidate.display()));
                continue;
            }
            Err(e) => {
                tried.push(format!("{}: {e}", candidate.display()));
                continue;
            }
        };
        if let Some((_, expected_crc)) = debuglink.as_ref() {
            let actual = crc32fast::hash(&debug_data);
            if actual != *expected_crc {
                tried.push(format!(
                    "{} (CRC mismatch: expected {:#010x}, got {:#010x})",
                    candidate.display(),
                    expected_crc,
                    actual,
                ));
                continue;
            }
        }
        return resolve_field_offsets_from_bytes(&debug_data, candidate);
    }
    anyhow::bail!(
        "{} is stripped; searched for external debuginfo via \
         debuglink={debuglink_name:?} build_id={build_id_hex:?} but \
         no candidate was readable or CRC-matched. Tried: {}",
        elf_path.display(),
        tried.join("; "),
    );
}

fn resolve_field_offsets_from_bytes(data: &[u8], source_path: &Path) -> Result<CounterOffsets> {
    let elf = Elf::parse(data).with_context(|| format!("parse ELF {}", source_path.display()))?;

    let load_section = |id: gimli::SectionId| -> Result<Cow<'_, [u8]>> {
        let name = id.name();
        for sh in &elf.section_headers {
            if let Some(section_name) = elf.shdr_strtab.get_at(sh.sh_name)
                && section_name == name
            {
                let range = sh.file_range().unwrap_or(0..0);
                return Ok(Cow::Borrowed(&data[range]));
            }
        }
        Ok(Cow::Borrowed(&[]))
    };

    let dwarf_sections = gimli::DwarfSections::load(load_section)?;
    let dwarf = dwarf_sections.borrow(|bytes| EndianSlice::new(bytes, LittleEndian));

    // Fast path: .debug_pubtypes accelerated lookup. If the ELF
    // carries a .debug_pubtypes section, scan it for "tsd_s" to
    // get the exact unit + DIE offset, then parse only THAT unit.
    // This turns an O(all_units) walk into O(pubtypes_entries) +
    // O(1_unit), cutting minutes to milliseconds on large binaries.
    if let Some(pubtypes_data) = find_section_slice(&elf, data, ".debug_pubtypes") {
        let pubtypes = gimli::DebugPubTypes::new(pubtypes_data, LittleEndian);
        let mut items = pubtypes.items();
        while let Ok(Some(entry)) = items.next() {
            let name_bytes = entry.name().to_slice().ok();
            if name_bytes.as_ref().map(|s| s.as_ref()) == Some(TSD_STRUCT_NAME.as_bytes()) {
                let unit_offset = entry.unit_header_offset();
                if let Ok(header) = dwarf.debug_info.header_from_offset(unit_offset)
                    && let Ok(unit) = dwarf.unit(header)
                    && let Ok(Some((a, d))) = struct_member_offsets_in_unit(&dwarf, &unit)
                {
                    let allocated = a.ok_or_else(|| {
                        anyhow!(
                            ".debug_pubtypes fast path: found tsd_s but {} missing",
                            ALLOCATED_FIELD,
                        )
                    })?;
                    let deallocated = d.ok_or_else(|| {
                        anyhow!(
                            ".debug_pubtypes fast path: found tsd_s but {} missing",
                            DEALLOCATED_FIELD,
                        )
                    })?;
                    return CounterOffsets::new(allocated, deallocated);
                }
            }
        }
    }

    let mut allocated: Option<u64> = None;
    let mut deallocated: Option<u64> = None;

    let mut units = dwarf.units();
    while let Some(header) = units.next()? {
        let unit = dwarf.unit(header)?;
        if let Some((a, d)) = struct_member_offsets_in_unit(&dwarf, &unit)? {
            if let Some(v) = a {
                allocated.get_or_insert(v);
            }
            if let Some(v) = d {
                deallocated.get_or_insert(v);
            }
            if allocated.is_some() && deallocated.is_some() {
                break;
            }
        }
    }

    let allocated = allocated.ok_or_else(|| {
        anyhow!(
            "DWARF walk of {} did not find field '{}' in struct '{}'",
            source_path.display(),
            ALLOCATED_FIELD,
            TSD_STRUCT_NAME,
        )
    })?;
    let deallocated = deallocated.ok_or_else(|| {
        anyhow!(
            "DWARF walk of {} did not find field '{}' in struct '{}'",
            source_path.display(),
            DEALLOCATED_FIELD,
            TSD_STRUCT_NAME,
        )
    })?;
    CounterOffsets::new(allocated, deallocated)
}

/// Resolve field offsets via a DWARF Package (.dwp) file using
/// gimli's `DwarfPackage` with proper CU index support. Requires
/// the parent (skeleton) DWARF to extract DWO IDs from skeleton
/// units.
fn resolve_field_offsets_from_dwp(
    parent_data: &[u8],
    dwp_data: &[u8],
    source_path: &Path,
) -> Result<CounterOffsets> {
    let parent_elf = Elf::parse(parent_data)
        .with_context(|| format!("parse parent ELF for DWP {}", source_path.display()))?;
    let dwp_elf =
        Elf::parse(dwp_data).with_context(|| format!("parse DWP ELF {}", source_path.display()))?;

    let load_parent = |id: gimli::SectionId| -> Result<Cow<'_, [u8]>> {
        let name = id.name();
        for sh in &parent_elf.section_headers {
            if let Some(sn) = parent_elf.shdr_strtab.get_at(sh.sh_name)
                && sn == name
            {
                let range = sh.file_range().unwrap_or(0..0);
                return Ok(Cow::Borrowed(&parent_data[range]));
            }
        }
        Ok(Cow::Borrowed(&[]))
    };
    let parent_sections = gimli::DwarfSections::load(load_parent)?;
    let parent = parent_sections.borrow(|bytes| EndianSlice::new(bytes, LittleEndian));

    let load_dwp = |id: gimli::SectionId| -> Result<EndianSlice<'_, LittleEndian>> {
        let dwo_name = format!("{}.dwo", id.name());
        for sh in &dwp_elf.section_headers {
            if let Some(sn) = dwp_elf.shdr_strtab.get_at(sh.sh_name)
                && (sn == dwo_name || sn == id.name())
            {
                let range = sh.file_range().unwrap_or(0..0);
                let bytes = dwp_data.get(range).unwrap_or(&[]);
                return Ok(EndianSlice::new(bytes, LittleEndian));
            }
        }
        Ok(EndianSlice::new(&[], LittleEndian))
    };
    let empty = EndianSlice::new(&[], LittleEndian);
    let dwp = gimli::DwarfPackage::load(load_dwp, empty)?;

    // Walk skeleton units, extract DWO IDs, find split units in DWP.
    let mut skel_units = parent.units();
    while let Some(skel_header) = skel_units.next()? {
        let skel_unit = parent.unit(skel_header)?;
        let dwo_id = match skel_unit.dwo_id {
            Some(id) => id,
            None => continue,
        };
        let Some(split_dwarf) = dwp.find_cu(dwo_id, &parent)? else {
            continue;
        };
        let mut split_units = split_dwarf.units();
        while let Some(split_header) = split_units.next()? {
            let split_unit = split_dwarf.unit(split_header)?;
            if let Some((a, d)) = struct_member_offsets_in_unit(&split_dwarf, &split_unit)? {
                let allocated =
                    a.ok_or_else(|| anyhow!("DWP: found tsd_s but {} missing", ALLOCATED_FIELD,))?;
                let deallocated =
                    d.ok_or_else(
                        || anyhow!("DWP: found tsd_s but {} missing", DEALLOCATED_FIELD,),
                    )?;
                return CounterOffsets::new(allocated, deallocated);
            }
        }
    }
    anyhow::bail!(
        "DWP walk of {} visited all skeleton units but did not find \
         field '{}' in struct '{}'",
        source_path.display(),
        ALLOCATED_FIELD,
        TSD_STRUCT_NAME,
    )
}

fn section_is_populated(elf: &Elf, data: &[u8], name: &str) -> bool {
    for sh in &elf.section_headers {
        if let Some(n) = elf.shdr_strtab.get_at(sh.sh_name)
            && n == name
        {
            let range = sh.file_range().unwrap_or(0..0);
            let len = data.get(range).map(<[u8]>::len).unwrap_or(0);
            return len > 0;
        }
    }
    false
}

/// Parse a `.gnu_debuglink` section. Layout per binutils
/// `bfd/opncls.c`:
///
/// - NUL-terminated filename, padded to a 4-byte boundary
/// - 4-byte little-endian CRC32 of the debug file
fn read_gnu_debuglink(elf: &Elf, data: &[u8]) -> Option<(String, u32)> {
    let bytes = find_section_slice(elf, data, ".gnu_debuglink")?;
    let nul = bytes.iter().position(|&b| b == 0)?;
    let name = std::str::from_utf8(&bytes[..nul]).ok()?.to_string();
    let after_name = (nul + 1).next_multiple_of(4);
    if after_name + 4 > bytes.len() {
        return None;
    }
    let crc = u32::from_le_bytes(bytes[after_name..after_name + 4].try_into().ok()?);
    Some((name, crc))
}

/// Parse an `NT_GNU_BUILD_ID` note inside `.note.gnu.build-id` and
/// return the build-id as lowercase hex.
fn read_build_id(elf: &Elf, data: &[u8]) -> Option<String> {
    let bytes = find_section_slice(elf, data, ".note.gnu.build-id")?;
    // Note layout: namesz(u32) descsz(u32) type(u32) name(padded) desc.
    if bytes.len() < 12 {
        return None;
    }
    let namesz = u32::from_le_bytes(bytes[0..4].try_into().ok()?) as usize;
    let descsz = u32::from_le_bytes(bytes[4..8].try_into().ok()?) as usize;
    let name_start = 12;
    let desc_start = name_start + namesz.next_multiple_of(4);
    let desc_end = desc_start + descsz;
    if desc_end > bytes.len() {
        return None;
    }
    let mut hex = String::with_capacity(descsz * 2);
    for &b in &bytes[desc_start..desc_end] {
        use std::fmt::Write;
        let _ = write!(&mut hex, "{b:02x}");
    }
    Some(hex)
}

/// Reject a debuglink filename that carries directory components or
/// is otherwise unsafe to splice into a candidate path. Per binutils
/// `bfd/opncls.c`, `.gnu_debuglink` is specified to hold a bare
/// filename (no path separators, no NUL bytes mid-string). A
/// malformed or hostile ELF could embed `../../etc/passwd` or
/// `/etc/shadow`, and `PathBuf::join` REPLACES the receiver with an
/// absolute right-hand side — so a single unvalidated splice would
/// produce a candidate path the caller would happily try to mmap as
/// debuginfo. Reject any name containing `/`, an embedded NUL, the
/// empty string, or the literal `"."`/`".."` traversal forms.
fn debuglink_name_is_safe(name: &str) -> bool {
    !name.is_empty() && !name.contains('/') && !name.contains('\0') && name != "." && name != ".."
}

/// Validate that `hex` matches the format [`read_build_id`] emits:
/// non-empty, even length, lowercase hex digits only (`[0-9a-f]`).
/// `read_build_id` is the only in-tree producer and writes via the
/// `{:02x}` formatter byte-by-byte, so any deviation from this
/// shape implies the upstream parser was bypassed (caller threading
/// untrusted text through `candidate_debuginfo_paths` directly) or
/// a future `read_build_id` regression.
///
/// The format-match check is strict enough that the eventual
/// `/usr/lib/debug/.build-id/<head>/<tail>.debug` path is bounded
/// to `[0-9a-f]/` segments — no `/`, `..`, NUL, or any other byte
/// that could traverse the candidate-path tree. Even-length is a
/// `read_build_id` invariant (each input byte produces exactly two
/// hex chars); pinning it here keeps a hypothetical odd-length
/// caller-bypass from emitting a malformed candidate.
fn build_id_hex_is_safe(hex: &str) -> bool {
    !hex.is_empty()
        && hex.len().is_multiple_of(2)
        && hex
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn candidate_debuginfo_paths(
    elf_path: &Path,
    debuglink_name: Option<&str>,
    build_id_hex: Option<&str>,
) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Some(hex) = build_id_hex
        && hex.len() >= 2
        && build_id_hex_is_safe(hex)
    {
        let (head, tail) = hex.split_at(2);
        out.push(PathBuf::from(format!(
            "/usr/lib/debug/.build-id/{head}/{tail}.debug"
        )));
    }
    if let Some(name) = debuglink_name
        && debuglink_name_is_safe(name)
        && let Some(parent) = elf_path.parent()
    {
        out.push(parent.join(name));
        out.push(parent.join(".debug").join(name));
        // /usr/lib/debug/<absolute target dir>/<name> — the
        // distro debuginfo convention is keyed off the absolute
        // install path of the ELF (e.g. /usr/bin/foo →
        // /usr/lib/debug/usr/bin/foo.debug). For a relative
        // target the "absolute install path" is undefined, so
        // skip this candidate rather than synthesizing
        // /usr/lib/debug/./foo.debug, which neither rpm nor deb
        // ever populate. `strip_prefix("/")` succeeds iff the
        // parent is absolute on Unix, so the Ok-arm doubles as
        // both the absoluteness gate and the relativization the
        // join below needs.
        if let Ok(rel) = parent.strip_prefix("/") {
            out.push(PathBuf::from("/usr/lib/debug").join(rel).join(name));
        }
    }
    out
}

fn find_section_slice<'a>(elf: &Elf, data: &'a [u8], name: &str) -> Option<&'a [u8]> {
    for sh in &elf.section_headers {
        if let Some(n) = elf.shdr_strtab.get_at(sh.sh_name)
            && n == name
        {
            let range = sh.file_range().unwrap_or(0..0);
            return data.get(range);
        }
    }
    None
}

fn struct_member_offsets_in_unit<R: Reader>(
    dwarf: &gimli::Dwarf<R>,
    unit: &Unit<R>,
) -> Result<Option<(Option<u64>, Option<u64>)>> {
    // Two strategies:
    // 1. Named struct: DW_TAG_structure_type with DW_AT_name "tsd_s"
    //    then walk direct DW_TAG_member children.
    // 2. Field scan: walk ALL DW_TAG_member entries looking for the
    //    unique field names directly. Handles typedef'd anonymous
    //    structs, C++ namespaced structs, or renamed struct types.
    //    The field names are unique enough (84-char mangled prefix)
    //    that false positives are impossible.
    let mut allocated: Option<u64> = None;
    let mut deallocated: Option<u64> = None;

    let mut entries = unit.entries();
    while let Some(entry) = entries.next_dfs()? {
        let tag = entry.tag();
        let name_attr = entry.attr_value(gimli::DW_AT_name);
        let loc_attr = entry.attr_value(gimli::DW_AT_data_member_location);

        // Strategy 1: named struct match — walk children.
        if tag == gimli::DW_TAG_structure_type {
            if let Some(name_val) = name_attr
                && let Ok(name_str) = dwarf.attr_string(unit, name_val)
            {
                let name_bytes = name_str.to_slice().ok();
                if name_bytes.as_ref().map(|s| s.as_ref()) == Some(TSD_STRUCT_NAME.as_bytes()) {
                    let struct_depth = entries.depth();
                    while let Some(child) = entries.next_dfs()? {
                        let child_tag = child.tag();
                        let child_name_attr = child.attr_value(gimli::DW_AT_name);
                        let child_loc_attr = child.attr_value(gimli::DW_AT_data_member_location);
                        let child_depth = entries.depth();
                        if child_depth <= struct_depth {
                            break;
                        }
                        if child_depth != struct_depth + 1 || child_tag != gimli::DW_TAG_member {
                            continue;
                        }
                        check_member_field(
                            dwarf,
                            unit,
                            child_name_attr,
                            child_loc_attr,
                            &mut allocated,
                            &mut deallocated,
                        )?;
                        if allocated.is_some() && deallocated.is_some() {
                            return Ok(Some((allocated, deallocated)));
                        }
                    }
                    if allocated.is_some() || deallocated.is_some() {
                        return Ok(Some((allocated, deallocated)));
                    }
                    continue;
                }
            }
            continue;
        }

        // Strategy 2: direct field scan — any DW_TAG_member with
        // the target field name, regardless of parent struct.
        if tag == gimli::DW_TAG_member {
            check_member_field(
                dwarf,
                unit,
                name_attr,
                loc_attr,
                &mut allocated,
                &mut deallocated,
            )?;
            if allocated.is_some() && deallocated.is_some() {
                return Ok(Some((allocated, deallocated)));
            }
            continue;
        }
    }
    if allocated.is_some() || deallocated.is_some() {
        return Ok(Some((allocated, deallocated)));
    }
    Ok(None)
}

fn check_member_field<R: Reader>(
    dwarf: &gimli::Dwarf<R>,
    unit: &Unit<R>,
    name_attr: Option<gimli::AttributeValue<R>>,
    loc_attr: Option<gimli::AttributeValue<R>>,
    allocated: &mut Option<u64>,
    deallocated: &mut Option<u64>,
) -> Result<()> {
    let name = match name_attr {
        Some(v) => v,
        None => return Ok(()),
    };
    let name_str = dwarf.attr_string(unit, name)?;
    let bytes = name_str.to_slice()?;
    let as_str = bytes.as_ref();
    let is_allocated = as_str == ALLOCATED_FIELD.as_bytes();
    let is_deallocated = as_str == DEALLOCATED_FIELD.as_bytes();
    if !is_allocated && !is_deallocated {
        return Ok(());
    }
    let offset = member_offset(loc_attr)?;
    if is_allocated && allocated.is_none() {
        *allocated = offset;
    }
    if is_deallocated && deallocated.is_none() {
        *deallocated = offset;
    }
    Ok(())
}

fn member_offset<R: Reader>(attr: Option<AttributeValue<R>>) -> Result<Option<u64>> {
    let Some(attr) = attr else { return Ok(None) };
    match attr {
        AttributeValue::Udata(v) => Ok(Some(v)),
        AttributeValue::Data1(v) => Ok(Some(v as u64)),
        AttributeValue::Data2(v) => Ok(Some(v as u64)),
        AttributeValue::Data4(v) => Ok(Some(v as u64)),
        AttributeValue::Data8(v) => Ok(Some(v)),
        AttributeValue::Sdata(v) if v >= 0 => Ok(Some(v as u64)),
        other => Err(anyhow!(
            "unexpected DW_AT_data_member_location form: {:?} — \
             DWARF expression forms are not supported for field-offset resolution",
            other
        )),
    }
}

// ---------------------------------------------------------------------
// TLS address arithmetic (Variant I + II)
// ---------------------------------------------------------------------

/// Reserved-area size at the low end of AArch64's Variant I thread-
/// control block — 2 words before the TLS image, per the AArch64 ELF
/// Linux ABI (IHI 0056D §4.1). Only consumed by the aarch64 path;
/// the cfg gate keeps the x86_64 build clean of dead-code warnings
/// without losing the constant from the source for cross-arch
/// reading.
#[cfg(target_arch = "aarch64")]
const TCB_SIZE_AARCH64: u64 = 16;

/// Variant II TLS address (x86_64). The thread pointer points to the
/// END of the static TLS block; the executable's TLS image sits at
/// `fs_base - tls_image_aligned_size`.
#[cfg(target_arch = "x86_64")]
fn compute_tls_address_variant_ii(
    fs_base: u64,
    tls_image_aligned_size: u64,
    st_value: u64,
    field_offset: u64,
) -> Result<u64> {
    let image_base = fs_base.checked_sub(tls_image_aligned_size).ok_or_else(|| {
        anyhow!(
            "fs_base ({fs_base:#x}) is below the aligned TLS image size \
             ({tls_image_aligned_size:#x}); target likely has no static \
             TLS initialized yet"
        )
    })?;
    image_base
        .checked_add(st_value)
        .and_then(|v| v.checked_add(field_offset))
        .ok_or_else(|| anyhow!("TLS address arithmetic overflow"))
}

/// Variant I TLS address (aarch64). `TPIDR_EL0` points to the
/// BEGINNING of the thread-control block; the TLS image sits at
/// `TP + round_up(TCB_SIZE_AARCH64, p_align)`.
#[cfg(target_arch = "aarch64")]
fn compute_tls_address_variant_i(
    tpidr_el0: u64,
    p_align: u64,
    st_value: u64,
    field_offset: u64,
) -> Result<u64> {
    let image_offset = round_up_pow2(TCB_SIZE_AARCH64, p_align).ok_or_else(|| {
        anyhow!(
            "TLS image offset overflow: tcb={} align={p_align:#x}",
            TCB_SIZE_AARCH64,
        )
    })?;
    tpidr_el0
        .checked_add(image_offset)
        .and_then(|v| v.checked_add(st_value))
        .and_then(|v| v.checked_add(field_offset))
        .ok_or_else(|| anyhow!("TLS address arithmetic overflow"))
}

#[cfg(target_arch = "x86_64")]
fn compute_tls_address(
    tp: u64,
    tls_image_aligned_size: u64,
    _p_align: u64,
    st_value: u64,
    field_offset: u64,
) -> Result<u64> {
    compute_tls_address_variant_ii(tp, tls_image_aligned_size, st_value, field_offset)
}

#[cfg(target_arch = "aarch64")]
fn compute_tls_address(
    tp: u64,
    _tls_image_aligned_size: u64,
    p_align: u64,
    st_value: u64,
    field_offset: u64,
) -> Result<u64> {
    compute_tls_address_variant_i(tp, p_align, st_value, field_offset)
}

// ---------------------------------------------------------------------
// /proc/<pid>/maps walking
// ---------------------------------------------------------------------

/// Scan `<proc_root>/<pid>/maps` for a mapping whose on-disk ELF
/// contains a jemalloc TLS symbol. Returns the (symbol info, DWARF
/// offsets) pair for the main executable match.
///
/// The match must live in the binary `<proc_root>/<pid>/exe` points
/// at (static TLS); a hit in a DSO returns `JemallocInDso` without
/// attempting the read.
fn find_jemalloc_via_maps_at(
    proc_root: &Path,
    pid: i32,
) -> std::result::Result<(TsdTlsSymbol, CounterOffsets), AttachError> {
    let exe_link = proc_root.join(pid.to_string()).join("exe");
    let exe_path = fs::read_link(&exe_link).map_err(|e| {
        AttachError::ReadlinkFailure(anyhow::Error::from(e).context(format!(
            "readlink {} (need it to gate static-TLS match)",
            exe_link.display(),
        )))
    })?;

    let maps_path = proc_root.join(pid.to_string()).join("maps");
    let contents = fs::read_to_string(&maps_path).map_err(|e| {
        AttachError::MapsReadFailure(
            anyhow::Error::from(e).context(format!("read {}", maps_path.display())),
        )
    })?;

    // Fast path: try the main executable directly via the /proc
    // magic symlink. This always works even for containerized
    // processes (the kernel resolves across mount namespaces),
    // unlike the maps-path open below which uses container-relative
    // paths that may not exist on the host filesystem. Reuses
    // `exe_link` from the readlink above rather than constructing
    // a duplicate PathBuf — the path is identical and the buffer
    // is still in scope.
    let mut last_symbol_err: Option<anyhow::Error> = None;
    let exe_mmap = std::fs::File::open(&exe_link).and_then(|f| unsafe { memmap2::Mmap::map(&f) });
    if let Ok(ref data) = exe_mmap
        && let Ok(elf) = Elf::parse(data)
    {
        match find_tsd_tls(&elf, &exe_path) {
            Ok(symbol) => {
                if symbol.e_machine != arch::EXPECTED_E_MACHINE {
                    return Err(AttachError::ArchMismatch(anyhow!(
                        "probe is {}-only; target ELF {} carries e_machine={:#x}. \
                             Use a probe binary matching the target's architecture \
                             (ptrace is same-arch).",
                        arch::ARCH_NAME,
                        symbol.elf_path.display(),
                        symbol.e_machine,
                    )));
                }
                // Cascade: fastest sources first, slow walk last.
                let ns_root = proc_root.join(pid.to_string()).join("root");
                let exe_rel = exe_path.strip_prefix("/").unwrap_or(&exe_path);

                // 1. Inline .debug_pubtypes accelerated (instant).
                if section_is_populated(&elf, data, ".debug_info")
                    && section_is_populated(&elf, data, ".debug_pubtypes")
                    && let Ok(offsets) = resolve_field_offsets_from_bytes(data, &exe_path)
                {
                    return Ok((symbol, offsets));
                }

                // 2. DWP (split DWARF — likely where tsd_s lives
                // for binaries compiled with -gsplit-dwarf).
                // The parent (skeleton) DWARF might be in the main
                // binary OR in the external debuginfo. Try both.
                let dwp_candidates = [
                    ns_root.join(format!("{}.dwp", exe_rel.display())),
                    PathBuf::from(format!("{}.dwp", exe_path.display())),
                ];
                let debuginfo_parent_candidates = [
                    ns_root.join(format!("{}.debuginfo", exe_rel.display())),
                    ns_root.join(format!("{}.debug", exe_rel.display())),
                ];
                if let Some(offsets) = try_resolve_dwp_offsets(
                    data,
                    &dwp_candidates,
                    &debuginfo_parent_candidates,
                    pid,
                ) {
                    return Ok((symbol, offsets));
                }

                // 3. External debuginfo (.debuginfo, .debug, debuglink, build-id).
                let debuginfo_candidates: Vec<PathBuf> = {
                    let debuglink = read_gnu_debuglink(&elf, data);
                    let build_id = read_build_id(&elf, data);
                    let mut c = Vec::new();
                    c.push(ns_root.join(format!("{}.debuginfo", exe_rel.display())));
                    c.push(ns_root.join(format!("{}.debug", exe_rel.display())));
                    let host_candidates = candidate_debuginfo_paths(
                        &exe_path,
                        debuglink.as_ref().map(|(n, _)| n.as_str()),
                        build_id.as_deref(),
                    );
                    for hc in &host_candidates {
                        if let Ok(rel) = hc.strip_prefix("/") {
                            c.push(ns_root.join(rel));
                        }
                        c.push(hc.clone());
                    }
                    c
                };
                if let Some(r) = try_resolve_debuginfo_offsets(&debuginfo_candidates, pid) {
                    return Ok((symbol, r));
                }

                // 4. Inline DWARF slow walk (last resort, bounded).
                // Only reaches here if no pubtypes, no DWP, no
                // external debuginfo resolved. Bounded to 200
                // units to prevent multi-minute hangs.
                if section_is_populated(&elf, data, ".debug_info")
                    && let Ok(offsets) = resolve_field_offsets_from_bytes(data, &exe_path)
                {
                    return Ok((symbol, offsets));
                }

                // 5. Nothing — report what we tried.
                let mut tried: Vec<String> = Vec::new();
                for p in &dwp_candidates {
                    tried.push(p.display().to_string());
                }
                for p in &debuginfo_candidates {
                    tried.push(p.display().to_string());
                }
                return Err(AttachError::DwarfParseFailure(anyhow!(
                    "jemalloc TSD symbol found in {} but no usable DWARF: \
                         inline .debug_info absent, no .dwp or external debuginfo \
                         resolved field offsets for struct '{}'. \
                         Rebuild with -g, supply a .dwp, or install the debuginfo \
                         package. Searched: {}",
                    exe_path.display(),
                    TSD_STRUCT_NAME,
                    tried.join(", "),
                )));
            }
            Err(e) => {
                last_symbol_err = Some(e);
            }
        }
    }

    // Slow path: walk maps for DSO matches (covers dynamically-linked
    // jemalloc, though static-TLS guard rejects those today).
    let mut seen: std::collections::BTreeSet<PathBuf> = std::collections::BTreeSet::new();
    for line in contents.lines() {
        let Some(path) = parse_maps_elf_path(line) else {
            continue;
        };
        if path == exe_path {
            continue; // already tried via /proc/<pid>/exe above
        }
        if !seen.insert(path.clone()) {
            continue;
        }
        let data = match fs::read(&path) {
            Ok(d) => d,
            Err(_) => continue,
        };
        let elf = match Elf::parse(&data) {
            Ok(e) => e,
            Err(_) => continue,
        };
        let _symbol = match find_tsd_tls(&elf, &path) {
            Ok(s) => s,
            Err(e) => {
                last_symbol_err = Some(e);
                continue;
            }
        };
        // Static-TLS guard: the match must be in the main executable.
        return Err(AttachError::JemallocInDso(anyhow!(
            "jemalloc TLS symbol found in {} but static-TLS probe requires \
             the match be in the main executable ({}); dynamic-TLS lookups \
             in shared objects are not supported. Remediation: relink \
             the target to embed jemalloc statically (e.g. build against \
             tikv-jemallocator-sys rather than a system libjemalloc.so).",
            path.display(),
            exe_path.display(),
        )));
    }

    let context = last_symbol_err
        .map(|e| format!(" — last per-ELF error: {e}"))
        .unwrap_or_default();
    Err(AttachError::JemallocNotFound(anyhow!(
        "jemalloc TLS symbol ({}) not found in any r-x mapping \
         under {}. Either the target is not jemalloc-linked, or the \
         symbol prefix is not in the recognized set.{}",
        "tsd_tls / je_tsd_tls / _rjem_je_tsd_tls / *_tsd_tls",
        maps_path.display(),
        context,
    )))
}

/// Walk the DWP (split-DWARF) candidates for the TSD field offsets.
/// For each `.dwp`, try the main binary as the skeleton-unit parent
/// first, then each external-debuginfo parent (skeleton units live
/// there when the main binary is fully stripped). Returns the first
/// resolved offsets, or `None` if no `.dwp` yields the field offsets.
fn try_resolve_dwp_offsets(
    data: &[u8],
    dwp_candidates: &[PathBuf],
    debuginfo_parent_candidates: &[PathBuf],
    pid: i32,
) -> Option<CounterOffsets> {
    for dwp_path in dwp_candidates {
        let dwp_mmap =
            match std::fs::File::open(dwp_path).and_then(|f| unsafe { memmap2::Mmap::map(&f) }) {
                Ok(m) => m,
                Err(e) => {
                    tracing::debug!(
                        pid, ?dwp_path, err = %e,
                        "ctprof probe: DWP not readable",
                    );
                    continue;
                }
            };
        tracing::debug!(
            pid,
            ?dwp_path,
            bytes = dwp_mmap.len(),
            "ctprof probe: trying DWP (mmap)",
        );
        // Try main binary as parent first.
        if let Ok(offsets) = resolve_field_offsets_from_dwp(data, &dwp_mmap, dwp_path) {
            return Some(offsets);
        }
        // Try debuginfo files as parent (skeleton
        // units live here when the main binary is
        // fully stripped).
        for dbg_parent_path in debuginfo_parent_candidates {
            let dbg_parent = match std::fs::File::open(dbg_parent_path)
                .and_then(|f| unsafe { memmap2::Mmap::map(&f) })
            {
                Ok(m) => m,
                Err(_) => continue,
            };
            tracing::debug!(
                pid,
                ?dbg_parent_path,
                bytes = dbg_parent.len(),
                "ctprof probe: trying DWP with debuginfo parent",
            );
            if let Ok(offsets) = resolve_field_offsets_from_dwp(&dbg_parent, &dwp_mmap, dwp_path) {
                return Some(offsets);
            }
        }
        tracing::debug!(
            pid,
            ?dwp_path,
            "ctprof probe: DWP found but no parent had skeleton units",
        );
    }
    None
}

/// Walk the external-debuginfo candidates (`.debuginfo`, `.debug`,
/// debuglink, build-id resolved) for the TSD field offsets, returning
/// the first that resolves or `None`.
fn try_resolve_debuginfo_offsets(
    debuginfo_candidates: &[PathBuf],
    pid: i32,
) -> Option<CounterOffsets> {
    for candidate in debuginfo_candidates {
        let dbg_mmap =
            std::fs::File::open(candidate).and_then(|f| unsafe { memmap2::Mmap::map(&f) });
        if let Ok(ref dbg_data) = dbg_mmap {
            tracing::debug!(
                pid,
                ?candidate,
                bytes = dbg_data.len(),
                "ctprof probe: trying debuginfo (mmap)",
            );
            if let Ok(r) = resolve_field_offsets_from_bytes(dbg_data, candidate) {
                return Some(r);
            }
        }
    }
    None
}

/// Extract the on-disk ELF path from a `/proc/<pid>/maps` line, or
/// `None` if the line is a non-file mapping (anon, `[stack]`, …) or
/// not executable. Returning only `r-x` mappings avoids re-opening
/// the same ELF for each of its segments.
fn parse_maps_elf_path(line: &str) -> Option<PathBuf> {
    let mut iter = line.split_whitespace();
    let _range = iter.next()?;
    let perms = iter.next()?;
    if !perms.contains('x') {
        return None;
    }
    let _offset = iter.next()?;
    let _dev = iter.next()?;
    let _inode = iter.next()?;
    let path = iter.next()?;
    if !path.starts_with('/') {
        return None;
    }
    Some(PathBuf::from(path))
}

// ---------------------------------------------------------------------
// Counter read after TP resolution
// ---------------------------------------------------------------------

/// Compute the TLS address from the resolved thread pointer and
/// pull the counters with one `process_vm_readv`. Factored out of
/// [`probe_thread`] so the post-attach math is testable and
/// reusable by callers that already have a TP from another path.
fn read_counters_at_thread_pointer(
    thread_pointer: u64,
    symbol: &TsdTlsSymbol,
    offsets: &CounterOffsets,
    tid: i32,
) -> std::result::Result<ThreadCounters, ProbeError> {
    let pid = Pid::from_raw(tid);
    let addr = compute_tls_address(
        thread_pointer,
        symbol.tls_image_aligned_size,
        symbol.p_align,
        symbol.st_value,
        offsets.thread_allocated,
    )
    .map_err(ProbeError::TlsArithmetic)?;

    let span = offsets.combined_read_span();
    debug_assert!(
        addr % 8 == 0,
        "process_vm_readv remote base must be 8-byte aligned (jemalloc \
         tsd_s.thread_allocated is a u64); got addr={addr:#x}",
    );
    let mut buf = vec![0u8; span as usize];
    let remote = RemoteIoVec {
        base: addr as usize,
        len: span as usize,
    };
    let mut local = [IoSliceMut::new(&mut buf)];
    let n = process_vm_readv(pid, &mut local, &[remote]).map_err(|e| {
        ProbeError::ProcessVmReadv(anyhow!("process_vm_readv on tid {tid} at {addr:#x}: {e}"))
    })?;
    if n != span as usize {
        return Err(ProbeError::ProcessVmReadv(anyhow!(
            "short process_vm_readv on tid {tid}: got {n} bytes, expected {span}"
        )));
    }

    let allocated = u64::from_le_bytes(buf[0..8].try_into().unwrap());
    // bytes 8..16 are thread_allocated_next_event_fast (discarded).
    let dealloc_offset = (offsets.thread_deallocated - offsets.thread_allocated) as usize;
    let deallocated =
        u64::from_le_bytes(buf[dealloc_offset..dealloc_offset + 8].try_into().unwrap());

    Ok(ThreadCounters {
        allocated_bytes: allocated,
        deallocated_bytes: deallocated,
    })
}

// ---------------------------------------------------------------------
// Detach guard
// ---------------------------------------------------------------------

/// Drop guard that detaches the tid on scope exit so a mid-read
/// failure or panic doesn't leave the target thread stopped.
struct ScopeDetach(i32);

impl Drop for ScopeDetach {
    fn drop(&mut self) {
        let pid = Pid::from_raw(self.0);
        let _ = ptrace::detach(pid, None);
    }
}

#[cfg(test)]
#[path = "host_thread_probe_tests.rs"]
mod tests;
