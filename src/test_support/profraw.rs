//! Guest-side LLVM coverage profraw flush + host-side write-out.
//!
//! Under `-C instrument-coverage`, the compiler inserts profile counters
//! and registers an atexit handler via `.init_array` that writes
//! `.profraw` at process exit. Inside a ktstr guest VM, `std::process::exit`
//! bypasses the atexit handler when the ktstr `#[ctor]` runs first
//! (the ordering between `.init_array` entries is unspecified). To keep
//! coverage data from being dropped, [`try_flush_profraw`] calls the
//! compiler-rt buffer API (`__llvm_profile_get_size_for_buffer` +
//! `__llvm_profile_write_buffer`) directly under `cfg(coverage)`,
//! serializes profraw into a heap buffer, and publishes it through the
//! guest-to-host bulk channel under `MSG_TYPE_PROFRAW`.
//!
//! VP data scope: the buffer flush covers coverage counters and
//! bitmaps only; PGO value-profile data is not preserved.
//! `__llvm_profile_write_buffer` passes a NULL `VPDataReader` to
//! `lprofWriteData` (defined in
//! `compiler-rt/lib/profile/InstrProfilingBuffer.c`),
//! whereas the file-based `__llvm_profile_write_file` path passes
//! `lprofGetVPDataReader()` (`InstrProfilingFile.c`) and DOES
//! capture VP records. This matches the current `-C instrument-coverage`
//! use case, which does not emit VP data. Combining coverage with PGO
//! (`-C profile-generate`) in the same binary would silently lose VP
//! records on this path; switch back to the file-based serializer if
//! that combination becomes a requirement.
//!
//! On the host, [`write_profraw`] receives those bytes via the SHM ring
//! and writes them into `LLVM_COV_TARGET_DIR` (or a fallback sibling
//! directory next to the test binary) as
//! `ktstr-test-{pid}-{counter}.profraw`.
//!
//! # Host atexit profraw redirect
//!
//! The host-side atexit path (the OS-managed dump that fires on
//! `std::process::exit` for non-VM-dispatch test runs — including
//! every test run via `cargo nextest run` directly, without a
//! `cargo ktstr` wrapper) reads `LLVM_PROFILE_FILE` once during the
//! LLVM runtime's `.init_array` initializer; if unset, the compiler-rt
//! default is `default.profraw` in the process cwd. When the operator
//! launched the test from a kernel source tree, that cwd points at the
//! source tree and the dump leaks into someone else's directory.
//!
//! [`redirect_default_profraw_path`] is a `priority = 0` ctor that
//! runs BEFORE the LLVM runtime's `.init_array` entry (which has no
//! priority and lands at the default `.init_array` slot per glibc
//! ordering rules) and points `LLVM_PROFILE_FILE` at the same
//! workspace-local target directory the cargo-ktstr wrapper already
//! injects, so a directly-invoked `cargo nextest run` no longer drops
//! `default.profraw` in cwd. The redirect is a no-op when:
//!   - getpid() == 1 (in-VM init; the SHM-ring flush above owns
//!     guest-side coverage and the env is irrelevant inside the VM
//!     because `std::process::exit` bypasses atexit anyway).
//!   - `LLVM_PROFILE_FILE` is already set (operator override or
//!     wrapper injection takes precedence — same `existing_env.is_some()`
//!     short-circuit `cargo-ktstr.rs::profraw_inject_for` applies).
//!   - The target binary is NOT coverage-instrumented. Detection is
//!     a symtab probe for `__llvm_profile_runtime` — the same
//!     compiler-rt symbol [`try_flush_profraw`] resolves for the
//!     guest-side flush. Non-instrumented binaries that link the
//!     ktstr lib (e.g. `cargo-ktstr` itself in a non-coverage build)
//!     must NOT set the env, otherwise the env propagates to spawned
//!     child test binaries, which then short-circuit their own
//!     redirect on the inherited value and write profraw into the
//!     PARENT's target dir rather than their own per-binary one
//!     (cargo-ktstr's exe lives in `target/{profile}/` while test
//!     binaries live in `target/{profile}/deps/`, so the two
//!     `current_exe`-relative target dirs differ).
//!
//! Supporting helper:
//! - [`find_symbol_vaddrs`] walks `.symtab` in one pass for multiple
//!   symbols at once, used by the coverage-instrumentation detection
//!   probes (in-process and on the host-side `/init` payload).
//!
//! Those probes read the binary via `memmap2::Mmap` rather than
//! `std::fs::read` so the kernel page cache backs the bytes goblin
//! parses; for coverage-instrumented binaries (hundreds of MiB up to
//! ~1 GiB) this avoids the heap allocation + copy of the entire binary
//! just to read its symbol table. [`try_flush_profraw`] itself no
//! longer parses the ELF — under `cfg(coverage)` it calls the
//! buffer-API entry points directly.

use anyhow::{Context, Result};
use std::fs::File;
use std::path::{Path, PathBuf};

#[cfg(coverage)]
use crate::vmm;

/// Flush LLVM coverage profraw to the host through the bulk channel.
///
/// Under `-C instrument-coverage` (cargo-llvm-cov sets `cfg(coverage)`)
/// the compiler-rt profile runtime is linked, so the buffer-API entry
/// points `__llvm_profile_get_size_for_buffer` and
/// `__llvm_profile_write_buffer` are defined. This calls them directly:
/// it allocates a buffer of the reported size, serializes the live
/// profile counters into it, and publishes the buffer through the
/// virtio-console bulk port for host-side extraction.
///
/// Calling `__llvm_profile_write_buffer` directly is also what keeps it
/// alive under `--gc-sections`: the call site is a link-time reference.
/// Resolving it by name through the ELF `.symtab` at runtime instead
/// (an earlier approach) was NOT a link reference, so the linker
/// dead-stripped `write_buffer` — nothing else in the retained graph
/// called it (the runtime's own `write_file` path is stripped too) — and
/// the flush silently no-op'd, leaving guest coverage at 0%.
///
/// No-op when not coverage-instrumented (the `cfg(coverage)` body is
/// absent, so the symbols are never referenced and the build links
/// without the profile runtime) or when called from host context.
pub(crate) fn try_flush_profraw() {
    #[cfg(coverage)]
    {
        if !vmm::guest_comms::is_guest() {
            return;
        }

        // Flush at most once per process. The guest `/init` (pid 1) can
        // reach `try_flush_profraw` from several paths in the same
        // process — the post-dispatch site (rust_init Phase 5), the
        // probe result-publish path, and the ctor / nextest `--exact`
        // dispatch paths (which flush then `process::exit`). A second
        // flush emits a second `Profraw` frame and `llvm-profdata merge`
        // would double-count the counters. First flush per process wins.
        {
            use std::sync::atomic::{AtomicBool, Ordering};
            static FLUSHED: AtomicBool = AtomicBool::new(false);
            if FLUSHED.swap(true, Ordering::SeqCst) {
                return;
            }
        }

        // SAFETY: both are stable compiler-rt buffer-API entry points,
        // defined whenever `-C instrument-coverage` linked the profile
        // runtime (guaranteed under `cfg(coverage)`). `get_size` is
        // `uint64_t (void)`; `write_buffer` is `int (char *)`, returning
        // 0 on success after serializing the live counters into the
        // caller's buffer. The dispatch context is single-threaded
        // (guest `/init`, post-dispatch).
        unsafe extern "C" {
            fn __llvm_profile_get_size_for_buffer() -> u64;
            fn __llvm_profile_write_buffer(buf: *mut std::os::raw::c_char) -> std::os::raw::c_int;
        }

        let needed = unsafe { __llvm_profile_get_size_for_buffer() } as usize;
        if needed == 0 {
            return;
        }

        let mut buf: Vec<u8> = vec![0u8; needed];
        // `__llvm_profile_write_buffer` returns 0 on success.
        if unsafe { __llvm_profile_write_buffer(buf.as_mut_ptr().cast::<std::os::raw::c_char>()) }
            != 0
        {
            return;
        }

        vmm::guest_comms::send_profraw(&buf);
    }
}

/// Resolve multiple symbol virtual addresses in a single pass through
/// the ELF .symtab. Returns addresses in the same order as `names`.
pub(crate) fn find_symbol_vaddrs(elf: &goblin::elf::Elf<'_>, names: &[&str]) -> Vec<Option<u64>> {
    let mut results = vec![None; names.len()];
    let mut remaining = names.len();

    for sym in elf.syms.iter() {
        if remaining == 0 {
            break;
        }
        if sym.st_size == 0 {
            continue;
        }
        let sym_name = match elf.strtab.get_at(sym.st_name) {
            Some(n) => n,
            None => continue,
        };
        for (i, name) in names.iter().enumerate() {
            if results[i].is_none() && sym_name == *name {
                results[i] = Some(sym.st_value);
                remaining -= 1;
                break;
            }
        }
    }
    results
}

static PROFRAW_COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Persist every coverage-profraw frame in a post-run guest bulk drain
/// to the llvm-cov-target directory.
///
/// Walks the [`crate::vmm::host_comms::BulkDrainResult`] the host
/// bucketed into [`crate::vmm::result::VmResult::guest_messages`] and,
/// for each [`MsgType::Profraw`](crate::vmm::wire::MsgType::Profraw)
/// frame that passed its per-frame CRC and carries a non-empty payload,
/// calls [`write_profraw`]. Mirrors the CRC + non-empty gate the
/// per-frame eval/probe dispatch applied so a corrupted or empty frame
/// is never written.
///
/// Called from [`crate::vmm::KtstrVm::run`] so the direct
/// `KtstrVm::run()` path persists guest coverage like the
/// eval (`run_ktstr_test_inner`) and auto-repro (`probe`) paths do —
/// previously the direct path silently dropped the profraw the guest
/// `/init` flushed. The eval and probe paths funnel through
/// `KtstrVm::run`, so they no longer extract `Profraw` frames
/// themselves; doing so here AND there would write the same payload
/// twice and `llvm-profdata merge` would double-count the counters.
pub(crate) fn persist_guest_profraw(messages: &crate::vmm::host_comms::BulkDrainResult) {
    use crate::vmm::wire::MsgType;
    for entry in &messages.entries {
        if MsgType::from_wire(entry.msg_type) == Some(MsgType::Profraw)
            && entry.crc_ok
            && !entry.payload.is_empty()
            && let Err(e) = write_profraw(&entry.payload)
        {
            eprintln!("ktstr_test: persist guest profraw: {e}");
        }
    }
}

/// Write profraw data to the llvm-cov-target directory.
pub(crate) fn write_profraw(data: &[u8]) -> Result<()> {
    let target_dir = target_dir();
    std::fs::create_dir_all(&target_dir)
        .with_context(|| format!("create profraw dir: {}", target_dir.display()))?;
    let id = PROFRAW_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let path = target_dir.join(format!("ktstr-test-{}-{}.profraw", std::process::id(), id));
    std::fs::write(&path, data).with_context(|| format!("write profraw: {}", path.display()))?;
    Ok(())
}

/// Resolve the llvm-cov-target directory for profraw output.
///
/// Cascade:
/// 1. `LLVM_COV_TARGET_DIR` — explicit operator override.
/// 2. `LLVM_PROFILE_FILE`'s parent directory — when an outer harness
///    (cargo-llvm-cov, or the cargo-ktstr `LLVM_PROFILE_FILE` injection
///    that prevents host-side `default.profraw` leakage from the
///    `cargo ktstr test` path) has already pinned the output location.
/// 3. `<current_exe parent>/llvm-cov-target/` — workspace-local
///    fallback so an instrumented binary invoked without any
///    coordination still drops profraw next to the build output
///    rather than in cwd.
///
/// `pub` rather than `pub(crate)` so the cargo-ktstr binary can
/// resolve the same directory before exec-ing `cargo nextest run`,
/// keeping host-side and guest-side profraw output co-located in
/// one tree without cargo-ktstr re-implementing the cascade.
pub fn target_dir() -> PathBuf {
    if let Ok(d) = std::env::var("LLVM_COV_TARGET_DIR") {
        return PathBuf::from(d);
    }
    // `LLVM_PROFILE_FILE` may be a bare filename (e.g. `default.profraw`)
    // — `Path::parent` returns `Some("")` in that shape, which would
    // otherwise propagate a structurally-empty `PathBuf` through the
    // cascade and surface as an unusable target dir downstream
    // (`std::fs::create_dir_all("")` errors with EINVAL on Linux).
    // The empty-os-str filter forces those bare-filename cases to fall
    // through to the `current_exe`-relative fallback below.
    if let Some(parent) = std::env::var("LLVM_PROFILE_FILE")
        .ok()
        .as_ref()
        .and_then(|p| Path::new(p).parent())
        .filter(|p| !p.as_os_str().is_empty())
    {
        return parent.to_path_buf();
    }
    let mut p = crate::resolve_current_exe().unwrap_or_else(|_| std::env::temp_dir());
    p.pop(); // remove binary name
    p.push("llvm-cov-target");
    p
}

/// Pure decision logic for [`redirect_default_profraw_path`]: given the
/// current `LLVM_PROFILE_FILE` value, the current pid, the coverage
/// instrumentation marker, and the workspace-local target dir
/// resolved from a callable, return the pattern to set on
/// `LLVM_PROFILE_FILE` or `None` to leave the env untouched.
/// Mirrors the `cargo-ktstr.rs::profraw_inject_for` predicate so
/// the directly-invoked `cargo nextest run` and `cargo ktstr test`
/// paths agree on when to inject.
///
/// Returns `Some(pattern)` only when:
///   - `pid != 1` (a test binary running as `/init` inside the guest VM
///     is owned by the SHM-ring flush; setting host-side env in that
///     context would still be a no-op because `std::process::exit`
///     bypasses atexit, but the early return keeps the in-VM startup
///     trace clean of an irrelevant env mutation).
///   - `existing` is `None` (operator-supplied
///     `LLVM_PROFILE_FILE` or wrapper-injected value takes precedence —
///     identical short-circuit to the cargo-ktstr wrapper).
///   - `is_coverage_instrumented` is `true` (the calling binary has
///     the LLVM compiler-rt profile runtime linked in). Without this
///     guard, the redirect would fire in `cargo-ktstr` and other
///     non-instrumented binaries that link the ktstr lib, polluting
///     the env passed to child test binaries — those child binaries
///     would then see a pre-set `LLVM_PROFILE_FILE` and short-circuit
///     their own redirect, writing profraw into the parent's target
///     dir rather than their own per-binary one. Detecting
///     instrumentation via the runtime marker symbol scopes the
///     redirect to the binaries that actually emit profraw.
///
/// `target_dir` is taken as a callable so the test suite can drive
/// the predicate against a synthetic target without building the
/// real `<current_exe parent>/llvm-cov-target/` path. `%p` (process
/// id) and `%m` (binary hash) are LLVM runtime expansions that keep
/// parallel-test output files distinct — the same pattern shape
/// `cargo-ktstr.rs::profraw_inject_for` emits.
fn redirect_pattern_for(
    pid: libc::pid_t,
    existing: Option<std::ffi::OsString>,
    is_coverage_instrumented: bool,
    target_dir: impl FnOnce() -> PathBuf,
) -> Option<PathBuf> {
    if pid == 1 || existing.is_some() || !is_coverage_instrumented {
        return None;
    }
    Some(target_dir().join("default-%p-%m.profraw"))
}

/// Detect whether the running binary has LLVM compiler-rt's profile
/// runtime linked in by probing for `__llvm_profile_runtime` —
/// the symbol that
/// `compiler-rt/lib/profile/InstrProfilingRuntime.cpp` defines as
/// `INSTR_PROF_PROFILE_RUNTIME_VAR`. Coverage-instrumented binaries
/// link this symbol; non-instrumented binaries (e.g. `cargo-ktstr`
/// in a normal release build) do not.
///
/// Probes via the symtab walk that profraw flushing already uses
/// (the symbol has hidden visibility, so dlsym would not find it).
/// Returns `false` on any failure path (binary mmap, ELF parse,
/// symbol absent) — false negatives leave the redirect off, which
/// is the conservative outcome: the binary writes
/// `default.profraw` in cwd just as before, no regression.
fn is_coverage_instrumented_binary() -> bool {
    let exe_file = match File::open("/proc/self/exe") {
        Ok(f) => f,
        Err(_) => return false,
    };
    // SAFETY: same invariants as `try_flush_profraw`'s mmap — see
    // the SAFETY block there. The `/proc/self/exe` mapping pins
    // the binary inode for the mmap's lifetime, and no part of
    // ktstr writes to its own binary during process startup.
    let mmap = match unsafe { memmap2::Mmap::map(&exe_file) } {
        Ok(m) => m,
        Err(_) => return false,
    };
    let elf = match goblin::elf::Elf::parse(&mmap) {
        Ok(e) => e,
        Err(_) => return false,
    };
    // Probe for the profile-write-buffer entry point rather than
    // the bare `__llvm_profile_runtime` marker. The marker is
    // declared `int __llvm_profile_runtime;` in compiler-rt and
    // can lose its `.symtab` size on `--gc-sections` /
    // `-Wl,--strip-debug` paths some toolchains apply to
    // coverage builds, which trips
    // [`find_symbol_vaddrs`]'s `st_size == 0` skip and silently
    // hides instrumented binaries from this probe. The
    // function-shaped symbols [`try_flush_profraw`] already
    // resolves (`__llvm_profile_get_size_for_buffer` and
    // `__llvm_profile_write_buffer`) keep their non-zero
    // `st_size` because every linker path emits the function
    // body — they are the reliable presence signal for
    // instrumented binaries, proved empirically by the fact that
    // coverage profraw collection in CI succeeds via the same
    // symtab probe.
    let vaddrs = find_symbol_vaddrs(
        &elf,
        &[
            "__llvm_profile_write_buffer",
            "__llvm_profile_get_size_for_buffer",
        ],
    );
    vaddrs.iter().any(|v| matches!(v, Some(va) if *va != 0))
}

/// Process-wide cached version of [`is_coverage_instrumented_binary`]:
/// whether the HOST process (`/proc/self/exe`) is built with
/// `-C instrument-coverage`. The symbol-table walk runs once per
/// process and is memoised in a `OnceLock<bool>` so repeated probes
/// only pay the ELF parse once.
///
/// History: VM-booting tests once used this to skip themselves under
/// coverage, because the instrumented `current_exe` used as the guest
/// `/init` OOMed early in boot (the budget in
/// `crate::vmm::memory_budget` was payload-agnostic, sizing the
/// non-instrumented case). That skip list is gone:
/// [`crate::vmm::memory_budget::initramfs_min_memory_mib`] now
/// detects an instrumented `/init` payload and reserves the extra
/// resident memory (`__llvm_prf_cnts` + `__llvm_prf_data`), so the
/// instrumented `/init` boots and its coverage is captured via
/// [`persist_guest_profraw`].
///
/// This probes the HOST process, not the `/init` payload — the budget
/// path probes the payload bytes directly (see
/// `KtstrVm::init_payload_coverage_reserve`). Retained as a
/// `#[doc(hidden)]` `pub` capability for out-of-tree consumers that
/// want to branch on host-process instrumentation.
///
/// `pub` (not `pub(crate)`) so integration tests in `tests/*.rs`
/// can reach the helper. `#[doc(hidden)]` keeps it out of the
/// crate's rendered docs — the helper is intentionally internal
/// to the test surface and the docs surface should not expose it.
#[doc(hidden)]
pub fn current_binary_is_coverage_instrumented() -> bool {
    use std::sync::OnceLock;
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(is_coverage_instrumented_binary)
}

ctor::declarative::ctor! {
/// Set `LLVM_PROFILE_FILE` to the workspace-local target directory
/// before the LLVM compiler-rt runtime reads it.
///
/// `priority = 0` lands this ctor in `.init_array.0`, which the
/// glibc startup loop walks BEFORE the unprioritized `.init_array`
/// slot that compiler-rt's `INSTR_PROF_PROFILE_RUNTIME_VAR` static
/// initializer (`InstrProfilingRuntime.cpp`) lives in. By the time
/// `__llvm_profile_initialize_file` runs and calls
/// `getenv("LLVM_PROFILE_FILE")`, our `set_var` has already landed.
///
/// See the module-level "Host atexit profraw redirect" section for
/// the full motivation. This ctor is intentionally separate from
/// [`crate::test_support::dispatch::ktstr_test_early_dispatch`] (the
/// unprioritized ctor that handles VM dispatch and SHM-ring flushes)
/// because that ctor must NOT acquire the priority slot — its
/// gauntlet-expansion and dispatch logic is order-insensitive
/// relative to compiler-rt, but pinning a low priority on it would
/// risk surprising interactions with future `.init_array.NN` entries.
/// Keeping the redirect in its own minimal ctor scopes the priority
/// promise to one well-understood operation.
///
/// The set_var call is sound in this ctor context: glibc invokes
/// `.init_array` entries on the main thread before any user code
/// has spawned an additional thread, so the env-block mutation is
/// race-free.
///
/// ctor 1.0's `priority` documentation flags the 0..100 range as
/// platform-reserved for the C runtime's own startup, so accessing
/// libc/std services from a constructor with such a priority "may
/// not be safe" in portable terms. On Linux/glibc the dynamic
/// linker finishes libc initialization before walking
/// `.init_array.0`, so `std::env::set_var` (which lowers to glibc's
/// `setenv`) is safe here. The priority retains the .init_array.0
/// placement that the compiler-rt ordering above depends on; other
/// platforms would need re-validation.
///
/// This site uses ctor's declarative `ctor::declarative::ctor! { ... }`
/// form; ctor 1.0 also ships `#[ctor::ctor(...)]` (proc-macro attribute)
/// re-exported under `crate::__private::ctor::ctor` for downstream
/// consumers. The declarative form is the in-tree convention because
/// it avoids the TT-muncher recursion-limit cost on the ktstr_test
/// expansion path.
#[ctor(unsafe, priority = 0)]
fn redirect_default_profraw_path() {
    // Cheap precondition checks first — pid (one syscall) and env
    // (one var_os call) — so the ELF parse only runs in the
    // direct-`cargo nextest run`-with-no-env case where the ctor
    // actually has a decision to make. cargo-ktstr-wrapped runs and
    // cargo-llvm-cov runs both pre-set `LLVM_PROFILE_FILE`, so
    // `existing.is_some()` short-circuits before
    // `current_binary_is_coverage_instrumented` mmaps `/proc/self/exe`
    // and walks the symtab (first call only; memoised). pid=1 (in-VM
    // init) similarly avoids the
    // probe — the SHM-ring flush owns guest-side coverage.
    let pid = unsafe { libc::getpid() };
    let existing = std::env::var_os("LLVM_PROFILE_FILE");
    if pid == 1 || existing.is_some() {
        return;
    }
    let instrumented = current_binary_is_coverage_instrumented();
    if let Some(pattern) = redirect_pattern_for(pid, existing, instrumented, target_dir) {
        // SAFETY: this ctor runs from `.init_array.0`, before any
        // user thread has spawned. The env block is single-writer,
        // single-reader at this moment, so `set_var` is sound. The
        // `set_var` API was deprecated in Rust 2024 for thread
        // unsafety in non-startup contexts, but ctor-time mutation
        // is exactly the protected case the deprecation guidance
        // carves out via `unsafe`.
        unsafe {
            std::env::set_var("LLVM_PROFILE_FILE", &pattern);
        }
    }
}
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::{EnvVarGuard, lock_env};
    use super::*;

    // -- target_dir --

    #[test]
    fn target_dir_with_env_var() {
        let _lock = lock_env();
        let _env = EnvVarGuard::set("LLVM_COV_TARGET_DIR", "/tmp/my-cov-dir");
        let dir = target_dir();
        assert_eq!(dir, PathBuf::from("/tmp/my-cov-dir"));
    }

    #[test]
    fn target_dir_from_llvm_profile_file() {
        let _lock = lock_env();
        let _env_cov = EnvVarGuard::remove("LLVM_COV_TARGET_DIR");
        let _env_prof =
            EnvVarGuard::set("LLVM_PROFILE_FILE", "/tmp/cov-target/ktstr-%p-%m.profraw");
        let dir = target_dir();
        assert_eq!(dir, PathBuf::from("/tmp/cov-target"));
    }

    #[test]
    fn target_dir_without_env_var() {
        let _lock = lock_env();
        let _env_cov = EnvVarGuard::remove("LLVM_COV_TARGET_DIR");
        let _env_prof = EnvVarGuard::remove("LLVM_PROFILE_FILE");
        let dir = target_dir();
        // Falls back to current_exe parent + "llvm-cov-target".
        assert!(
            dir.ends_with("llvm-cov-target"),
            "expected path ending in llvm-cov-target, got: {}",
            dir.display()
        );
    }

    /// `LLVM_PROFILE_FILE` set to a bare filename (no parent
    /// directory component, e.g. `default.profraw`) must fall
    /// through to the `current_exe`-relative fallback rather than
    /// surfacing a structurally-empty `PathBuf` through the
    /// cascade. `Path::new("default.profraw").parent()` returns
    /// `Some("")`; without the empty-os-str filter,
    /// `target_dir` would return `PathBuf::from("")` and downstream
    /// `create_dir_all` calls fail with EINVAL.
    #[test]
    fn target_dir_bare_filename_llvm_profile_file_falls_through() {
        let _lock = lock_env();
        let _g_cov = EnvVarGuard::remove("LLVM_COV_TARGET_DIR");
        let _g_prof = EnvVarGuard::set("LLVM_PROFILE_FILE", "default.profraw");
        let dir = target_dir();
        assert!(
            !dir.as_os_str().is_empty(),
            "bare-filename LLVM_PROFILE_FILE must fall through to the \
             current_exe fallback, not return an empty PathBuf",
        );
        assert!(
            dir.ends_with("llvm-cov-target"),
            "fallback must land at the current_exe-relative llvm-cov-target \
             dir, got: {}",
            dir.display(),
        );
    }

    // -- redirect_pattern_for (host-side LLVM_PROFILE_FILE redirect predicate) --

    /// Pid 1 (in-VM init) must short-circuit. Even when no
    /// `LLVM_PROFILE_FILE` is set and the binary is instrumented,
    /// the SHM-ring flush owns guest-side coverage and the
    /// host-side env redirect is irrelevant.
    #[test]
    fn redirect_pattern_for_pid_1_returns_none() {
        let pattern =
            redirect_pattern_for(1, None, true, || PathBuf::from("/should/not/be/called"));
        assert!(
            pattern.is_none(),
            "pid=1 (guest init) must skip env redirect"
        );
    }

    /// An already-set `LLVM_PROFILE_FILE` (operator override or
    /// cargo-ktstr/cargo-llvm-cov wrapper injection) takes
    /// precedence — the redirect must be a no-op so it does not
    /// stomp on the outer harness's profile location.
    #[test]
    fn redirect_pattern_for_existing_env_returns_none() {
        let pattern = redirect_pattern_for(
            42,
            Some(std::ffi::OsString::from("/operator/picked/path.profraw")),
            true,
            || PathBuf::from("/should/not/be/called"),
        );
        assert!(
            pattern.is_none(),
            "existing LLVM_PROFILE_FILE must take precedence"
        );
    }

    /// Empty `LLVM_PROFILE_FILE` (`Some("")` from a shell that did
    /// `export LLVM_PROFILE_FILE=`) is a degenerate but possible
    /// shape. `var_os` returns `Some` for an empty value, so the
    /// existing-env short-circuit fires and we leave it alone — the
    /// LLVM runtime treats empty as "fall through to default" so
    /// `default.profraw` lands in cwd, but that is the operator's
    /// choice once they explicitly assigned the variable. Pinned
    /// here so a future "treat empty as unset" change is a deliberate
    /// decision rather than a silent drift.
    #[test]
    fn redirect_pattern_for_empty_env_short_circuits() {
        let pattern = redirect_pattern_for(42, Some(std::ffi::OsString::new()), true, || {
            PathBuf::from("/should/not/be/called")
        });
        assert!(
            pattern.is_none(),
            "Some(\"\") in LLVM_PROFILE_FILE counts as set; redirect must defer"
        );
    }

    /// Non-instrumented binaries (cargo-ktstr in normal builds, the
    /// `ktstr` standalone CLI) must not set the env. Otherwise the
    /// inherited env in spawned child test binaries pre-empts their
    /// own redirect and they write profraw into the parent's target
    /// dir instead of their own per-binary one.
    #[test]
    fn redirect_pattern_for_non_instrumented_binary_returns_none() {
        let pattern =
            redirect_pattern_for(42, None, false, || PathBuf::from("/should/not/be/called"));
        assert!(
            pattern.is_none(),
            "non-coverage-instrumented binary must not pollute the env passed \
             to children"
        );
    }

    /// Host-pid + unset env + instrumented binary produces a
    /// redirect to the workspace-local target dir with the LLVM
    /// `%p`/`%m` expansions baked into the filename.
    #[test]
    fn redirect_pattern_for_host_unset_returns_target_pattern() {
        let target = PathBuf::from("/synthetic/llvm-cov-target");
        let pattern = redirect_pattern_for(42, None, true, || target.clone())
            .expect("host pid + unset env + instrumented must produce a redirect pattern");
        assert_eq!(
            pattern,
            PathBuf::from("/synthetic/llvm-cov-target/default-%p-%m.profraw"),
        );
    }

    /// The pattern shape matches what
    /// `cargo-ktstr.rs::profraw_inject_for` emits — both paths
    /// inject `default-%p-%m.profraw` so coverage merge tools
    /// (cargo-llvm-cov) see a uniform filename suffix regardless of
    /// which entry point launched the test binary.
    #[test]
    fn redirect_pattern_for_filename_matches_cargo_ktstr_wrapper() {
        let target = PathBuf::from("/x");
        let pattern = redirect_pattern_for(42, None, true, || target.clone()).unwrap();
        assert_eq!(
            pattern.file_name().and_then(|n| n.to_str()),
            Some("default-%p-%m.profraw"),
            "filename suffix must match cargo-ktstr's profraw_inject_for",
        );
    }

    // -- find_symbol_vaddrs --

    #[test]
    fn find_symbol_vaddrs_resolves_known_symbol() {
        let exe = crate::resolve_current_exe().unwrap();
        let data = std::fs::read(&exe).unwrap();
        let elf = goblin::elf::Elf::parse(&data).unwrap();
        // "main" is present in the symtab of any Rust test binary.
        let results = find_symbol_vaddrs(&elf, &["main"]);
        assert_eq!(results.len(), 1);
        assert!(
            results[0].is_some(),
            "main symbol should be resolved in test binary"
        );
        assert_ne!(results[0].unwrap(), 0, "main address should be nonzero");
    }

    #[test]
    fn find_symbol_vaddrs_missing_symbol_returns_none() {
        let exe = crate::resolve_current_exe().unwrap();
        let data = std::fs::read(&exe).unwrap();
        let elf = goblin::elf::Elf::parse(&data).unwrap();
        let results = find_symbol_vaddrs(&elf, &["__nonexistent_symbol_xyz__"]);
        assert_eq!(results.len(), 1);
        assert!(results[0].is_none());
    }

    #[test]
    fn find_symbol_vaddrs_mixed_results() {
        let exe = crate::resolve_current_exe().unwrap();
        let data = std::fs::read(&exe).unwrap();
        let elf = goblin::elf::Elf::parse(&data).unwrap();
        let results = find_symbol_vaddrs(&elf, &["main", "__nonexistent_symbol_xyz__"]);
        assert_eq!(results.len(), 2);
        assert!(results[0].is_some(), "main should resolve");
        assert!(results[1].is_none(), "nonexistent should not resolve");
    }

    // -- profile buffer-API retention (regression) --

    /// `--gc-sections` dead-strips `__llvm_profile_write_buffer` unless a
    /// link-time reference keeps it — [`try_flush_profraw`]'s direct call
    /// under `cfg(coverage)` is that reference. This test references the
    /// symbol by NAME only (a `.symtab` lookup, not a link reference of
    /// its own), so it fails if that call is ever removed and the linker
    /// strips the function — the exact regression that left guest
    /// coverage at 0% before the direct-call fix. Coverage-only: the
    /// symbol does not exist in non-instrumented builds.
    #[cfg(coverage)]
    #[test]
    fn write_buffer_symbol_retained_under_coverage() {
        let exe = crate::resolve_current_exe().unwrap();
        let data = std::fs::read(&exe).unwrap();
        let elf = goblin::elf::Elf::parse(&data).unwrap();
        let v = find_symbol_vaddrs(&elf, &["__llvm_profile_write_buffer"]);
        assert!(
            v[0].is_some(),
            "__llvm_profile_write_buffer must survive --gc-sections under \
             coverage; without it the guest flush silently no-ops",
        );
    }
}
