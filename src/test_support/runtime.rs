//! Runtime configuration primitives shared by `eval` and `probe`.
//!
//! `eval` calls `probe::attempt_auto_repro` from its failure path,
//! so items shared between the two siblings live here to avoid a
//! circular import chain. All items are `pub(crate)` and remain
//! internal to `test_support`.

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Duration;

use super::entry::KtstrTestEntry;

/// Stable PathBuf for the process-owned config scratch directory.
///
/// Populated once by [`scratch_dir`] on first access. Kept in a
/// separate `OnceLock` from the `TempDir` itself so the `atexit`
/// cleanup handler can read the path through `extern "C"` without
/// involving the `tempfile::TempDir` value (whose `Drop` would
/// otherwise never run — see the "leak bound" note on
/// [`scratch_dir`]).
static SCRATCH_PATH: OnceLock<PathBuf> = OnceLock::new();

/// Process-owned scratch directory for all inline-config tempfile
/// writes — both [`config_content_parts`] (in-VM eval path) and
/// [`crate::export::export_test`]'s `config_content_addition`
/// (host-side .run packaging path).
///
/// Created lazily on first access via `tempfile::Builder` with
/// explicit `0o700` mode (overrides the crate default of umask-
/// restricted `0o777`-via-`mkdir(2)`, which on a standard
/// `umask=0o022` host yields `0o755` and would expose directory
/// listings + filename predictability to other uids). The
/// directory is a random-suffixed subdirectory of
/// `std::env::temp_dir()`, owned by the current uid.
///
/// Both call sites share this single directory because the
/// security and leak-bound properties are identical for both
/// purposes, and a single `OnceLock` + single `atexit` handler is
/// simpler than maintaining parallel scratch dirs that diverge
/// silently. Filenames are independently prefixed at each call
/// site (`ktstr-config-{hash:016x}.json` for the eval path,
/// `ktstr-export-config-{hash:016x}-{basename}` for the export
/// path) so the two purposes can be visually distinguished inside
/// the same directory.
///
/// Two properties matter:
///
/// 1. **Symlink defense.** /tmp is sticky-bit world-writable, so an
///    attacker can pre-plant a symlink at the predictable content-
///    addressed path and have us write to wherever it points. A
///    per-process 0o700 subdirectory blocks every cross-uid access
///    mode (read, list, write, traverse); only our process can
///    create or replace files inside it, which eliminates the
///    symlink-attack surface for the tempfile-write path.
///
/// 2. **Leak bound.** Rust does NOT run `Drop` impls on values
///    stored in `static` slots at process exit — so the
///    `tempfile::TempDir`'s built-in cleanup would never fire here.
///    Instead, the path is registered with `libc::atexit`
///    (POSIX-spec process-exit handler) so a clean exit
///    (`exit(3)`, fall-off-`main`) triggers
///    [`std::fs::remove_dir_all`] on the directory. Crash, abort,
///    SIGKILL, or panic-`abort` skip the atexit handler and leak
///    the directory; the residual is bounded by the number of
///    such ungraceful exits and the directory contents are
///    text-sized config files. The tempdir's random suffix
///    prevents collisions across runs, so accumulated leak dirs
///    don't interfere with future runs.
pub(crate) fn scratch_dir() -> &'static Path {
    SCRATCH_PATH
        .get_or_init(|| {
            let td = tempfile::Builder::new()
                .prefix("ktstr-config-")
                .permissions(std::fs::Permissions::from_mode(0o700))
                .tempdir()
                .expect("create ktstr config scratch directory");
            // `keep()` consumes the TempDir without running its
            // Drop's cleanup (it flips the cleanup flag and returns
            // the bare PathBuf we own). The atexit registration
            // below takes over cleanup responsibility.
            let path = td.keep();
            // SAFETY: `cleanup_scratch_dir` has the required
            // `extern "C" fn()` signature that `libc::atexit`
            // accepts. The `unsafe` block here is required because
            // `libc::atexit` itself is an `unsafe extern "C"` FFI
            // call (the callback signature itself is plain
            // `extern "C" fn()`, not `unsafe`). Registering more
            // than once is the caller's responsibility;
            // `OnceLock::get_or_init` guarantees this runs exactly
            // once per process.
            let rc = unsafe { libc::atexit(cleanup_scratch_dir) };
            assert_eq!(
                rc, 0,
                "libc::atexit registration for ktstr config scratch dir failed"
            );
            path
        })
        .as_path()
}

/// Process-exit handler registered via `libc::atexit` by
/// [`scratch_dir`] on first init. Removes the scratch directory and
/// every config file inside it. Errors are ignored — by the time
/// this runs the process is exiting and there is nowhere to surface
/// a failure (no `eprintln!` ordering guarantees from inside an
/// atexit handler, and panicking would be unsound across the C ABI
/// boundary).
extern "C" fn cleanup_scratch_dir() {
    if let Some(path) = SCRATCH_PATH.get() {
        let _ = std::fs::remove_dir_all(path);
    }
}

/// True when `RUST_BACKTRACE` is set to `"1"` or `"full"`.
///
/// Controls whether the full guest kernel console is appended to the
/// `--- diagnostics ---` section of a failed test, and whether
/// auto-repro forwards the repro VM's COM1/COM2 output to the host
/// terminal in real time. The scheduler-log and sched_ext-dump
/// sections of a failure are always emitted regardless of this flag.
pub(crate) fn verbose() -> bool {
    std::env::var("RUST_BACKTRACE")
        .map(|v| v == "1" || v == "full")
        .unwrap_or(false)
}

/// True when `KTSTR_NO_PERF_MODE` is set to a NON-EMPTY value.
///
/// Centralises the perf-mode-disabled check used by the dispatch
/// gauntlet routes (`run_named_test`, `run_gauntlet_test`, and the
/// verifier-cell listing route `list_verifier_cells_all`, all in
/// `super::dispatch`) and the eval entry path
/// (`super::eval::run_ktstr_test_inner_impl`). All four sites
/// previously called `std::env::var("KTSTR_NO_PERF_MODE").is_ok()`
/// directly, which returned true for `KTSTR_NO_PERF_MODE=` (empty
/// string set, e.g. via `unset`/`set` interplay in CI shells, or a
/// `--env KTSTR_NO_PERF_MODE` Docker pass-through with no value) —
/// silently skipping every `performance_mode` test. Requiring a
/// non-empty value matches operator intent ("set it to something to
/// disable perf mode") and rejects the empty-string accident.
///
/// `cargo ktstr test --no-perf-mode` exports `KTSTR_NO_PERF_MODE=1`
/// (a non-empty value), so the existing CLI surface is unaffected.
pub(crate) fn no_perf_mode_active() -> bool {
    std::env::var(crate::KTSTR_NO_PERF_MODE_ENV)
        .map(|v| !v.is_empty())
        .unwrap_or(false)
}

/// True when `KTSTR_BYPASS_LLC_LOCKS` is set to a NON-EMPTY value.
///
/// Centralises the bypass check used at 7 reader sites:
/// `vmm/builder.rs:1199`, `cli/kernel_build/build.rs:102` +
/// `:488` (the latter the inverse `!bypass_llc_locks_active()`
/// form), `bin/cargo_ktstr/kernel/mod.rs:720`,
/// `bin/cargo_ktstr/misc/shell.rs:181`, and `bin/ktstr.rs:652` +
/// `:1267`. All sites previously spelled the same
/// `.ok().is_some_and(|v| !v.is_empty())` inline; centralising
/// eliminates the drift hazard and matches the
/// `no_perf_mode_active` shape so the empty-string contract is
/// uniformly enforced.
///
/// Set via `--bypass-llc-locks` CLI flag or
/// `KTSTR_BYPASS_LLC_LOCKS=1` direct export. Empty
/// (`KTSTR_BYPASS_LLC_LOCKS=` from a Docker `--env` pass-through
/// without value) does NOT activate per the empty-as-unset
/// contract — preventing a stray export from silently disabling
/// LLC flock contention enforcement in CI.
pub fn bypass_llc_locks_active() -> bool {
    std::env::var(crate::KTSTR_BYPASS_LLC_LOCKS_ENV)
        .ok()
        .is_some_and(|v| !v.is_empty())
}

/// Effective no-perf-mode for a given test entry. The env override
/// `KTSTR_NO_PERF_MODE` and the per-entry [`KtstrTestEntry::no_perf_mode`]
/// attribute are OR'd: either source forces the no-perf path
/// (cpuset/LLC locking still applies, but vCPU pinning, hugepages,
/// NUMA mbind, RT scheduling, and KVM exit suppression are all
/// skipped). The env override is the operator-level switch; the
/// per-entry attribute lets a test author opt the test out
/// permanently — e.g. tests that exercise wild virtual topologies
/// the host hardware can't possibly satisfy under perf-mode pinning.
pub(crate) fn no_perf_mode_for_entry(entry: &KtstrTestEntry) -> bool {
    no_perf_mode_active() || entry.no_perf_mode
}

/// True when `KTSTR_PERF_ONLY` is set to a NON-EMPTY value.
///
/// Mirrors [`no_perf_mode_active`]'s empty-as-unset contract (see
/// [`crate::KTSTR_PERF_ONLY_ENV`]): any non-empty value restricts the
/// run to `performance_mode` tests, an empty value does not.
/// Consulted at the dispatch named/gauntlet routes and the eval entry
/// to skip non-perf entries before VM boot. Set by the mergebase
/// perf-delta subcommand.
pub(crate) fn perf_only_active() -> bool {
    std::env::var(crate::KTSTR_PERF_ONLY_ENV)
        .map(|v| !v.is_empty())
        .unwrap_or(false)
}

/// Whether `perf_only_active()` requires SKIPPING this entry: perf-only
/// mode is on and the entry is not a `performance_mode` test. A
/// `performance_mode` entry is always kept (it is the selection
/// target); every other entry is skipped so a perf-delta run measures
/// only the perf-configured tests.
pub(crate) fn perf_only_skips_entry(entry: &KtstrTestEntry) -> bool {
    perf_only_active() && !entry.performance_mode
}

/// Derive initramfs archive path, host path, and guest path from a
/// scheduler's static `config_file`. This scheduler-only projection is shared
/// by ordinary test VMs and generated verifier cells so both launch the same
/// declaration instead of silently dropping the verifier's config.
pub(crate) fn scheduler_config_file_parts(
    scheduler: &super::entry::Scheduler,
) -> Option<(String, PathBuf, String)> {
    let config_path = scheduler.config_file?;
    let file_name = Path::new(config_path)
        .file_name()
        .and_then(|n| n.to_str())
        .expect("config_file must have a valid filename");
    let archive_path = format!("include-files/{file_name}");
    let guest_path = format!("/include-files/{file_name}");
    Some((archive_path, PathBuf::from(config_path), guest_path))
}

/// Entry-shaped compatibility wrapper for the ordinary test launch paths.
pub(crate) fn config_file_parts(entry: &KtstrTestEntry) -> Option<(String, PathBuf, String)> {
    scheduler_config_file_parts(entry.scheduler)
}

/// Stable u64 hash of arbitrary string content.
///
/// Used by the config-content tempfile path code, but suitable for
/// any content-addressed naming site that needs determinism across
/// rustc bumps.
///
/// Uses `siphasher::sip::SipHasher13::new_with_keys(0, 0)` rather
/// than `std::collections::hash_map::DefaultHasher` because the std
/// algorithm is explicitly unspecified across rustc versions (see
/// workspace `Cargo.toml` for the dep-line rationale). The explicit
/// `new_with_keys(0, 0)` form matches the project's other
/// stable-hash sites (`src/test_support/sidecar/mod.rs`, `build.rs`)
/// so a future audit of zero-keyed SipHasher13 callers finds every
/// instance via one grep. Same content always produces the same u64
/// across toolchain upgrades, so cached artifacts stay reproducible
/// across machines and rustc bumps.
pub(crate) fn content_hash(content: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = siphasher::sip::SipHasher13::new_with_keys(0, 0);
    content.hash(&mut hasher);
    hasher.finish()
}

/// Resolve inline config content into a temp file on disk, returning
/// `(archive_path, host_path, guest_path, sched_args)` where
/// `sched_args` are the CLI args derived from the scheduler's
/// `config_file_def` arg template. Returns `None` when the entry has
/// no `config_content`.
pub(crate) fn config_content_parts(
    entry: &KtstrTestEntry,
) -> Option<(String, PathBuf, String, Vec<String>)> {
    use std::io::Write as _;
    let content = entry.config_content?;
    let (arg_template, guest_path) = entry.scheduler.config_file_def?;
    let archive_path = guest_path.trim_start_matches('/').to_string();
    let hash = content_hash(content);
    let dir = scratch_dir();
    // Write to a uniquely-named scratch file, then atomic-rename to the
    // canonical content-addressed path:
    //   - Scratch acquisition via `NamedTempFile::new_in` uses
    //     `mkstemp(3)` semantics: random suffix, opened O_EXCL so no
    //     pre-existing file can be subverted as the write target.
    //   - The atomic `persist` rename is the cross-thread / cross-process
    //     race fix. Two writers of the same content race their renames
    //     to the canonical path; the last writer wins, but since `hash`
    //     is content-addressed both wrote byte-identical content, so the
    //     winner's bytes match the loser's. No torn writes are possible
    //     because `rename(2)` is atomic at the inode level — readers
    //     either see the old inode or the new one, never a partial blend.
    //   - On panic between `new_in` and `persist`, NamedTempFile's `Drop`
    //     unlinks the scratch file. No `/tmp` leak from in-process aborts.
    let canonical = dir.join(format!("ktstr-config-{hash:016x}.json"));
    let mut scratch =
        tempfile::NamedTempFile::new_in(dir).expect("create ktstr config scratch file");
    scratch
        .as_file_mut()
        .write_all(content.as_bytes())
        .expect("write ktstr config content to scratch");
    scratch
        .persist(&canonical)
        .expect("atomic-rename ktstr config scratch to canonical path");
    let expanded = arg_template.replace("{file}", guest_path);
    let sched_args: Vec<String> = expanded.split_whitespace().map(|s| s.to_string()).collect();
    Some((archive_path, canonical, guest_path.to_string(), sched_args))
}

/// Build the shared `cmdline=` string appended to every ktstr_test
/// guest boot. Per-scheduler sysctls, per-scheduler kargs,
/// `RUST_BACKTRACE` / `RUST_LOG` propagation, and the host-resolved
/// `KTSTR_SIDECAR_DIR` so the guest's `sidecar_dir()` returns the
/// SAME path the host's freeze coordinator writes to. Without that
/// propagation, host and guest each compute the run directory
/// independently — the host walks `gix::discover` from a real
/// workspace cwd and produces `{kernel}-{commit}` whereas the
/// guest's cwd is `/` (no git repo, no kernel env), yielding the
/// `unknown-unknown` fallback. Anything the two VM-launch sites
/// (`run_ktstr_test_inner` and `attempt_auto_repro`) previously
/// re-implemented side-by-side lives here.
/// Ordered guest-kernel cmdline tokens owned by a scheduler declaration.
///
/// Keep this projection independent of `KtstrTestEntry`: generated verifier
/// cells have a scheduler but no test entry, and booting them without these
/// tokens can change scheduler startup, BPF rodata, and verifier instruction
/// counts. Sysctls precede raw kargs exactly as on the ordinary test path.
pub(crate) fn scheduler_cmdline_tokens(scheduler: &super::entry::Scheduler) -> Vec<String> {
    let mut parts = Vec::new();
    for s in scheduler.sysctls {
        parts.push(format!("sysctl.{}={}", s.key(), s.value()));
    }
    for &karg in scheduler.kargs {
        parts.push(karg.to_string());
    }
    parts
}

fn build_cmdline_extra_with_probe_dump_gate(
    entry: &KtstrTestEntry,
    include_probe_dump_gate: bool,
) -> String {
    let mut parts = scheduler_cmdline_tokens(entry.scheduler);
    // Framework-owned readiness authority. Scheduler kargs are otherwise
    // passed through verbatim, but they may neither arm nor disarm this
    // watchdog-accounting protocol: only the typed test-entry field decides
    // whether the primary VM emits the exact `=1` token. This also prevents
    // auto-repro from inheriting a raw scheduler-provided gate.
    parts.retain(|part| {
        part != "KTSTR_AWAIT_PROBE_DUMP_READY" && !part.starts_with("KTSTR_AWAIT_PROBE_DUMP_READY=")
    });
    // Per-test KASLR opt-out (see `KtstrTestEntry.kaslr` doc). The base
    // cmdline `base_guest_cmdline` at `src/vmm/setup/mod.rs` does NOT
    // inject `nokaslr` by default — KASLR is on. A test that needs determinism sets `kaslr = false` in
    // its `#[ktstr_test]` attribute; that lands the token here, where it
    // composes with any operator-supplied `Scheduler::kargs(&["nokaslr"])`
    // above (kernel parses the flag as a bool; duplicates are harmless).
    // Mirrored guest-side by `vmm::rust_init::create_cgroup_parent_from_sched_args`
    // and `monitor::symbols::resolve_page_offset`, both of which handle the
    // `nokaslr` case via the live-publisher fall back to `DEFAULT_PAGE_OFFSET`.
    if !entry.kaslr {
        parts.push("nokaslr".to_string());
    }
    // Periodic-capture window anchoring: a run that declares periodic
    // captures gates its FIRST ScenarioStart (which opens the capture
    // window) on the host's periodic-prereqs-ready signal, so the
    // window opens over the live workload with the full declared
    // duration instead of racing the KASLR publish (see
    // `vmm::rust_init` Phase 5 and `SIGNAL_PERIODIC_READY`). Only
    // periodic runs pay the wait; non-periodic runs never see the flag.
    if entry.num_snapshots > 0 {
        parts.push("KTSTR_AWAIT_PERIODIC_READY=1".to_string());
    }
    // Diagnostic scheduler-start gate: only explicitly opted-in tests
    // pay the host's full probe-counter decode and guest wait. The
    // host publishes the matching edge only after the exact dump
    // reader succeeds, so scheduler-relative fault timers start with
    // the diagnostic substrate already readable.
    if include_probe_dump_gate && entry.probe_dump_ready_gate {
        parts.push("KTSTR_AWAIT_PROBE_DUMP_READY=1".to_string());
    }
    if let Ok(bt) = std::env::var("RUST_BACKTRACE") {
        parts.push(format!("RUST_BACKTRACE={bt}"));
    }
    if let Ok(log) = std::env::var("RUST_LOG") {
        parts.push(format!("RUST_LOG={log}"));
    }
    // Propagate the host-resolved sidecar dir so the guest scenario
    // computes the same path the host's freeze coordinator wrote to
    // (e.g. when a test reads `sidecar_dir().join("foo.json")` from
    // inside the guest, the path matches the host's writer site).
    // The host resolves via the OnceLock-cached project commit walk
    // from the workspace cwd; the guest's cwd is `/` and would
    // otherwise fall back to `unknown-unknown`. Sidecar dir paths
    // are filesystem-safe ASCII (kernel version + 7-char hex
    // commit, optional `-dirty` suffix), so the cmdline-as-token
    // shape is sound — no escaping needed for whitespace.
    //
    // Absolutize via `current_dir().join()` when the resolved path
    // is relative (the default-branch shape:
    // `target/ktstr/{kernel}-{commit}` against the host cwd). The
    // guest's cwd is `/`, so a relative token would resolve there
    // instead of at the host's workspace root — the propagation
    // must carry the FULL absolute path so the guest's
    // `sidecar_dir()` reports the same string the host's writer
    // site used. Falls back to the raw resolved path when the cwd
    // probe fails (extremely rare; happens only when the process's
    // cwd was rmdir'd while alive — a metadata probe has no
    // recourse, leave the path as-is).
    let resolved = super::sidecar::sidecar_dir();
    let absolute = if resolved.is_absolute() {
        resolved
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(&resolved))
            .unwrap_or(resolved)
    };
    if let Some(s) = absolute.to_str() {
        parts.push(format!("KTSTR_SIDECAR_DIR={s}"));
    }
    parts.join(" ")
}

/// Build the primary VM's cmdline additions.
///
/// An explicitly opted-in primary carries the probe-dump readiness gate. Its
/// guest args also carry the matching minimal probe request; see
/// `append_primary_probe_dump_arg`.
pub(crate) fn build_cmdline_extra(entry: &KtstrTestEntry) -> String {
    build_cmdline_extra_with_probe_dump_gate(entry, true)
}

/// Build an auto-repro VM's cmdline additions without primary-only launch
/// gates.
///
/// Stall auto-repro intentionally skips probe attachment, so inheriting a
/// primary cell's probe-dump gate would make guest init wait for an edge the
/// host can never publish.
pub(crate) fn build_auto_repro_cmdline_extra(entry: &KtstrTestEntry) -> String {
    build_cmdline_extra_with_probe_dump_gate(entry, false)
}

#[cfg(feature = "wprof")]
pub(crate) fn attach_wprof_if_requested(
    builder: crate::vmm::KtstrVmBuilder,
    entry: &KtstrTestEntry,
    label: &'static str,
) -> anyhow::Result<crate::vmm::KtstrVmBuilder> {
    if !entry.wprof {
        return Ok(builder);
    }
    let mut config = crate::vmm::wprof::WprofConfig::from_env().map_err(|e| {
        anyhow::anyhow!(
            "ktstr_test: {label}: wprof requested by \
             #[ktstr_test(wprof)] but WprofConfig::from_env failed: \
             {e:#}. Ensure cargo-ktstr's install_env exported \
             KTSTR_WPROF_PATH and the path is readable."
        )
    })?;
    if let Some(custom_args) = entry.wprof_args {
        config.args = custom_args.split_whitespace().map(String::from).collect();
    }
    Ok(builder.wprof(Some(config)))
}

/// Per-cpu memory-scaling core shared by every VM sizing site.
///
/// Returns `(cpus * 64).max(256)` — 64 MiB per vCPU, floored at
/// 256 MiB. 64 MiB/cpu matches `derive_test_memory_mib`'s established
/// scaling (it was cut from 256 to 64 MiB/cpu when the compressed
/// initramfs landed and dropped peak guest memory; commit 5b6063a0);
/// no per-cpu budget beyond that heuristic is documented.
///
/// The returned value is a LOWER BOUND: every call site feeds it to
/// `.memory_deferred_min(mib)`, which raises the actual allocation to
/// fit the real initramfs, so the final boot memory may exceed this.
pub(crate) fn cpu_scaled_memory_mib(cpus: u32) -> u32 {
    checked_cpu_scaled_memory_mib(cpus)
        .expect("validated live topology overflows the 64 MiB/vCPU memory floor")
}

/// Fallible scalar form used when decoding topology dimensions from an ELF.
pub(crate) fn checked_cpu_scaled_memory_mib(cpus: u32) -> Option<u32> {
    cpus.checked_mul(64).map(|scaled| scaled.max(256))
}

/// Memory floor for one verifier topology preset.
///
/// Verifier VMs use deferred sizing, so the actual initramfs budget may raise
/// this floor. The preset budget caps only the topology-derived CPU scaling:
/// a 252-vCPU synthetic topology must not advertise 16 GiB merely because it
/// has many vCPUs when the preset explicitly budgets 4 GiB. Small presets keep
/// the ordinary 64 MiB/vCPU floor.
pub(crate) fn verifier_preset_memory_min_mib(cpus: u32, preset_memory_mib: usize) -> u32 {
    checked_verifier_preset_memory_min_mib(cpus, preset_memory_mib)
        .expect("validated live verifier topology overflows the 64 MiB/vCPU memory floor")
}

/// Fallible verifier-memory projection for topology data decoded from an ELF.
pub(crate) fn checked_verifier_preset_memory_min_mib(
    cpus: u32,
    preset_memory_mib: usize,
) -> Option<u32> {
    let preset_cap = u32::try_from(preset_memory_mib).unwrap_or(u32::MAX);
    checked_cpu_scaled_memory_mib(cpus).map(|scaled| scaled.min(preset_cap))
}

/// wprof BPF ringbuf sizing baked into `WprofConfig::default_args`.
///
/// wprof allocates `WPROF_DEFAULT_RINGBUF_CNT` BPF ring buffers, each
/// `--ringbuf-size` KiB rounded up to a power-of-two byte size; the
/// tracer faults the whole arena in on capture, so guest RAM must
/// cover it. These two constants are the single source of truth:
/// `crate::vmm::wprof::WprofConfig::default_args` renders them into
/// the `--ringbuf-size=`/`--ringbuf-cnt=` flags AND
/// [`WPROF_MIN_MEMORY_MIB`] derives the guest-memory floor from them,
/// so changing the sizing here moves the flags and the floor together.
///
/// The default is deliberately minimal — a single 16 MiB ring buffer,
/// which is wprof's own per-buffer default (`DEFAULT_RINGBUF_SZ`) and
/// comfortably holds the short sched-event captures ktstr's wprof
/// tests run. ktstr only tests that wprof writes valid data
/// end-to-end; oversizing merely inflates the per-cell fault-in cost.
/// Larger captures pass their own `#[ktstr_test(wprof_args = "...")]`
/// and, if the resulting arena exceeds the guest's normal memory,
/// their own `memory_mib`.
pub(crate) const WPROF_DEFAULT_RINGBUF_SIZE_KB: u32 = 16 * 1024;
pub(crate) const WPROF_DEFAULT_RINGBUF_CNT: u32 = 1;

/// Guest-memory floor (MiB) for a stamped wprof attachment, derived
/// from the default ringbuf arena above.
///
/// Always compiled because the cargo-ktstr admission runner reads a
/// stamped wprof bit from an ELF built with a different feature set
/// than its own, so this projection cannot be `wprof`-gated. The
/// public wprof API remains feature-gated and aliases this value.
///
/// At the minimal default the arena (16 MiB) sits below the universal
/// 256 MiB memory floor from [`cpu_scaled_memory_mib`], so this floor
/// is subsumed and never bumps a real VM; it re-engages only if the
/// default ringbuf sizing above ever grows the arena past that floor.
pub(crate) const WPROF_MIN_MEMORY_MIB: u32 = {
    let per_rb_bytes = wprof_next_pow2_u64(WPROF_DEFAULT_RINGBUF_SIZE_KB as u64 * 1024);
    let arena_bytes = per_rb_bytes * WPROF_DEFAULT_RINGBUF_CNT as u64;
    (arena_bytes / (1024 * 1024)) as u32
};

/// Round `n` up to the next power of two, mirroring wprof's
/// `round_pow_of_2` on the ringbuf byte size. `const fn` so the
/// memory floor is a compile-time constant.
const fn wprof_next_pow2_u64(n: u64) -> u64 {
    let mut p: u64 = 1;
    while p < n {
        p <<= 1;
    }
    p
}

/// Apply the feature-independent memory floor encoded by a stamped wprof bit.
pub(crate) fn apply_wprof_memory_floor(raw_mib: u32, wprof: bool) -> u32 {
    if wprof && raw_mib < WPROF_MIN_MEMORY_MIB {
        WPROF_MIN_MEMORY_MIB
    } else {
        raw_mib
    }
}

/// Compute the exact deferred-memory floor from admission-stamp scalars.
///
/// This is the process-independent core of [`derive_test_memory_mib`].  The
/// pre-exec target runner has only the final ELF stamp, not a live
/// [`KtstrTestEntry`], and must publish the same lower bound before it starts
/// the heavyweight test process. Keeping the scalar projection here prevents
/// admission and VM construction from acquiring different memory weights.
pub(crate) fn derive_test_memory_min_mib(cpus: u32, declared_memory_mib: u32, wprof: bool) -> u32 {
    checked_derive_test_memory_min_mib(cpus, declared_memory_mib, wprof)
        .expect("validated live topology overflows the 64 MiB/vCPU memory floor")
}

/// Fallible admission-memory projection for topology data decoded from an ELF.
pub(crate) fn checked_derive_test_memory_min_mib(
    cpus: u32,
    declared_memory_mib: u32,
    wprof: bool,
) -> Option<u32> {
    let raw = checked_cpu_scaled_memory_mib(cpus)?.max(declared_memory_mib);
    Some(apply_wprof_memory_floor(raw, wprof))
}

/// Derive the test VM's memory floor from a CPU count + entry.
///
/// Returns `max(cpu_scaled_memory_mib(cpus), entry.memory_mib)`. When
/// `entry.wprof` is true, bumps to the feature-independent stamped wprof
/// memory floor if below it. The tracer attachment itself remains gated by
/// the `wprof` feature.
///
/// The returned value is the LOWER BOUND on guest memory; the
/// VM builder ultimately uses `.memory_deferred_min(mib)` which
/// also accounts for the initramfs size, so the final boot memory
/// may exceed this value.
pub(crate) fn derive_test_memory_mib(cpus: u32, entry: &KtstrTestEntry) -> u32 {
    let memory_min_mib = derive_test_memory_min_mib(cpus, entry.memory_mib, entry.wprof);
    #[cfg(feature = "wprof")]
    {
        use crate::vmm::wprof::WPROF_MIN_MEMORY_MIB;
        let raw = cpu_scaled_memory_mib(cpus).max(entry.memory_mib);
        if memory_min_mib != raw {
            tracing::info!(
                test = %entry.name,
                requested_mib = raw,
                floored_mib = WPROF_MIN_MEMORY_MIB,
                "wprof enabled; memory_mib floored to \
                 WPROF_MIN_MEMORY_MIB"
            );
        }
    }
    memory_min_mib
}

/// Resolve the VM topology and memory size from an optional
/// TopoOverride.
///
/// Returns `(topology, memory_mib)` where `topology` is the
/// `vmm::topology::Topology` passed to the VM builder and `memory_mib`
/// is the LOWER BOUND on guest memory (the builder's
/// `.memory_deferred_min(mib)` may raise the actual allocation
/// to fit the initramfs). When `topo` is `Some`, both come from
/// the override and the memory is honored verbatim (per the
/// override-is-verbatim contract — see `topo.rs:42-44`). When
/// `topo` is `None`, the topology comes from `entry.topology` and
/// memory is derived by [`derive_test_memory_mib`]. Shared with
/// `attempt_auto_repro` so the repro VM always sizes memory the
/// same way as the first VM — reproducibility requires identical
/// topology, including the wprof floor when applicable.
///
/// When the `wprof` feature is enabled and `entry.wprof` is true,
/// a TopoOverride with memory_mib below the wprof floor triggers
/// a warn-level log but is still honored verbatim.
pub(crate) fn resolve_vm_topology(
    entry: &KtstrTestEntry,
    topo: Option<&super::topo::TopoOverride>,
) -> (crate::vmm::topology::Topology, u32) {
    match topo {
        Some(t) => {
            #[cfg(feature = "wprof")]
            if entry.wprof && t.memory_mib < crate::vmm::wprof::WPROF_MIN_MEMORY_MIB {
                tracing::warn!(
                    test = %entry.name,
                    override_mib = t.memory_mib,
                    wprof_min_mib = crate::vmm::wprof::WPROF_MIN_MEMORY_MIB,
                    "wprof enabled with TopoOverride.memory_mib below \
                     WPROF_MIN_MEMORY_MIB; honoring the override per the \
                     override-is-verbatim contract, but wprof may OOM-kill \
                     mid-run"
                );
            }
            (crate::vmm::topology::Topology::from(t), t.memory_mib)
        }
        None => {
            let cpus = entry.topology.total_cpus();
            let mem = derive_test_memory_mib(cpus, entry);
            (entry.topology, mem)
        }
    }
}

/// Append per-scheduler `sched_args` entries shared by both VM-launch
/// paths: `--config <guest_path>` if the scheduler declared one, the
/// cgroup-parent switch, the scheduler's own fixed args, and
/// per-entry extra args. Active-flag dispatch and probe-specific args
/// remain at the call site because they differ between the paths.
///
/// The caller owns the `include_files` binding on the builder;
/// `config_file_parts` and the guest-path push are returned separately
/// so the caller decides whether to attach include files (production
/// does, probe-only repro pipelines that already pass `include_files`
/// can skip it).
/// Concrete absolute-path example used by the panic messages that
/// reject malformed `--cell-parent-cgroup` values — names the
/// scheduler's declared default when one exists, falls back to a
/// canonical `/ktstr` literal otherwise. The operator gets a
/// copy-pasteable shape regardless of whether the scheduler is
/// cell-aware. Centralised so both rejection arms (Value-invalid and
/// MissingValue) display the same example.
fn cgroup_parent_example(entry: &KtstrTestEntry) -> String {
    entry
        .scheduler
        .cgroup_parent
        .map(|p| p.as_str().to_string())
        .unwrap_or_else(|| "/ktstr".to_string())
}

pub(crate) fn append_base_sched_args(entry: &KtstrTestEntry, args: &mut Vec<String>) {
    // Fail-fast on a malformed user-supplied `--cell-parent-cgroup`
    // value before the auto-inject branch. The host-side consumer
    // `resolve_cgroup_root` (defined in `test_support::args`, used by
    // the probe/setup path at `probe.rs::build_dispatch_ctx_parts`)
    // interpolates the value into a
    // `/sys/fs/cgroup{path}` literal and hands the result to
    // `CgroupManager::new`, which has NO host-root guard — any path
    // that doesn't start with `/` lands inside the host cgroup root
    // (e.g. `""` → `/sys/fs/cgroup`, `"my_test"` →
    // `/sys/fs/cgroupmy_test`) and corrupts unrelated cgroup state
    // when subsequent `cgroups.setup(...)` calls run. The guest-side
    // sibling `vmm::rust_init::create_cgroup_parent_from_sched_args`
    // happens to be safe-by-coincidence for the empty case because
    // `enable_subtree_controllers_to` early-returns when leaf equals
    // the cgroup root — but probe.rs has no such gate, so the host
    // fail-fast is what actually protects against corruption.
    //
    // The check is universal — independent of whether the scheduler
    // declares a default `cgroup_parent` — because both routes
    // (`extra_sched_args` from the test author, `sched_args` from
    // the scheduler def) flow through the same parse + chain below,
    // and the corruption risk is identical regardless of who
    // supplied the bad value. Operator sees the message at test
    // setup time, before any cgroup ops run.
    match super::args::parse_cell_parent_cgroup(
        entry
            .scheduler
            .sched_args
            .iter()
            .chain(entry.extra_sched_args.iter())
            .copied(),
    ) {
        super::args::CellParentCgroupArg::Value(path)
            if !super::args::cell_parent_path_is_valid(path) =>
        {
            let example = cgroup_parent_example(entry);
            let mut fixes = format!(
                "supply an absolute path under `/` with at least one non-`.`/`..` \
                 segment (e.g. `{example}`) for the per-test cgroup root"
            );
            if let Some(default) = entry.scheduler.cgroup_parent {
                fixes.push_str(&format!(
                    " or omit the flag entirely (the framework will auto-inject \
                     the scheduler's default `cgroup_parent = {default}`)"
                ));
            }
            panic!(
                "test `{}` supplies `--cell-parent-cgroup` with a value `{:?}` \
                 (via `extra_sched_args` on the test or `sched_args` in the \
                 scheduler def) that does not start with `/`, is `/` alone, or \
                 contains `.`/`..` segments that normalize back to the host \
                 cgroup root; {fixes}. Empty, bare `/`, relative, or paths \
                 like `/.`, `/foo/..`, `/./bar/..` all resolve to a path \
                 equal to or inside `/sys/fs/cgroup` (e.g. empty → \
                 `/sys/fs/cgroup`, `/` → `/sys/fs/cgroup/`, `/.` → \
                 `/sys/fs/cgroup` after canonicalization) and corrupt \
                 unrelated cgroup state when the probe-side `CgroupManager` \
                 operates on the resolved path. This gate mirrors the \
                 const-eval check in `CgroupPath::new` so runtime values \
                 share the validation contract that compile-time \
                 declarations already pass.",
                entry.name, path,
            );
        }
        super::args::CellParentCgroupArg::MissingValue => {
            let example = cgroup_parent_example(entry);
            let mut fixes = format!(
                "either remove the bare `--cell-parent-cgroup` and let the \
                 framework auto-inject the scheduler's default (when one is \
                 declared), or supply a value (e.g. `--cell-parent-cgroup={example}` \
                 in combined form, or `--cell-parent-cgroup` followed by an \
                 absolute path in two-token form)"
            );
            if entry.scheduler.cgroup_parent.is_none() {
                fixes.push_str(
                    "; the scheduler in this test declares no default \
                     `cgroup_parent`, so an absolute-path value is required",
                );
            }
            panic!(
                "test `{}` supplies a bare `--cell-parent-cgroup` (via \
                 `extra_sched_args` on the test or `sched_args` in the \
                 scheduler def) with no following value; {fixes}. The \
                 framework intercepts this here because letting it through \
                 would silently combine with the framework's auto-inject \
                 (when a default exists) and trip clap's `cannot be used \
                 multiple times` diagnostic — a confusing error that buries \
                 the actual missing-value mistake.",
                entry.name,
            );
        }
        super::args::CellParentCgroupArg::Value(_) => {
            // User-supplied valid path — flows through the
            // `args.extend(...)` calls below. Skip the auto-inject so
            // clap doesn't reject the duplicate flag with `cannot be
            // used multiple times`.
        }
        super::args::CellParentCgroupArg::Absent => {
            // `cgroup_parent` controls the cgroup root where the
            // framework places test cgroups (`resolve_cgroup_root`
            // returns `/sys/fs/cgroup{cgroup_parent}` for guest
            // CgroupManager). It does NOT auto-inject
            // `--cell-parent-cgroup` into the scheduler's argv —
            // cell-aware schedulers (scx_mitosis et al.) interpret
            // that flag by enabling userspace_managed_cell_mode and
            // starting an inotify-driven CellManager that can
            // interfere with the host-side periodic-capture
            // pipeline. If a scheduler genuinely needs
            // `--cell-parent-cgroup`, the scheduler declaration's
            // own `sched_args` array (or the per-test
            // `extra_sched_args`) must include it explicitly. The
            // guest-side `create_cgroup_parent_from_sched_args`
            // mkdir + subtree-controller setup still fires when
            // `--cell-parent-cgroup` is present in `/sched_args` —
            // it's gated on the flag's presence, not on whether the
            // framework injected it vs. the user added it manually.
        }
    }
    args.extend(entry.scheduler.sched_args.iter().map(|s| s.to_string()));
    args.extend(entry.extra_sched_args.iter().map(|s| s.to_string()));
}

/// Retry budget for the guest's `vmm::rust_init::send_sys_rdy_with_retry`
/// loop. Boot-to-readiness wall time is a fixed base PLUS per-vCPU
/// work: the virtio-console multiport handshake (DEVICE_READY →
/// PORT_ADD → PORT_READY → PORT_OPEN per `drivers/char/virtio_console.c`)
/// issues per-CPU work whose wall time grows roughly linearly with
/// topology size, on top of a fixed device-enumeration / first-CPU
/// cost. The budget is therefore ADDITIVE — `BASE_MS + vcpus *
/// PER_VCPU_MS` — not `max(BASE, scaled)`.
///
/// The earlier `max` form left the per-vCPU term dead below ~67 vCPUs
/// (since `vcpus * 150` only clears the 10 s floor at 67 vCPUs), so a
/// 64-vCPU VM got the same 10 s budget as a 1-vCPU VM. Under host
/// contention a 64-vCPU handshake was observed at ~10 s and timed out
/// by ~8 ms — the disk-template gauntlet flake. The additive base
/// gives every topology ~10 s of fixed headroom on top of its linear
/// per-vCPU term (64 vCPUs → 19.6 s), so a slow handshake under load
/// no longer races the floor.
///
/// Capped at 90 s as a sanity bound on a genuinely-stuck boot's
/// guest-side retry loop — NOT to protect the host watchdog. The
/// watchdog deadline IS derived from this budget ([`vm_boot_headroom`]
/// feeds [`vm_timeout_from_entry`]), so the budget can never "blow" a
/// deadline it defines; the old 30 s cap's stated rationale was
/// inverted. That 30 s cap truncated the additive term above
/// ~133 vCPUs — a 256-vCPU guest wants `10_000 + 256×150 = 48_400` ms
/// but was clamped to 30_000 ms, starving the widest topologies of
/// boot budget exactly where overcommit makes the boot slowest (the
/// wide-SMP boot-timeout class). 90 s admits the full additive budget
/// up to the 512-vCPU `MAX_VCPUS` (`10_000 + 512×150 = 86_800` ms);
/// only pathological counts above 533 vCPUs clip.
///
/// The const-fn signature lets both the host (`vm_boot_headroom`,
/// `vm_timeout_from_entry`) and the guest (`vmm::rust_init`) compute
/// the same budget without trans-VM coordination — the guest reads
/// its own vCPU count from `/sys/devices/system/cpu/online`. The guest
/// uses this UN-scaled budget (it cannot read host overcommit); on an
/// oversubscribed boot its `send_sys_rdy_with_retry` loop may exhaust
/// and WARN, but that is non-fatal (the host monitor's `data_valid`
/// gate keeps reads safe). The host's [`vm_timeout_from_entry`] is the
/// Tier-3 dead-man deadline this budget feeds; it multiplies the
/// headroom by a flat [`DEADMAN_HEADROOM_MULT`] and no longer scales by
/// the host overcommit ratio — the progress watchdog's Tier-1/2 rules,
/// not the wall deadline, now absorb an oversubscribed host.
pub(crate) const fn sys_rdy_budget_ms(vcpus: u32) -> u64 {
    const BASE_MS: u64 = 10_000;
    const CAP_MS: u64 = 90_000;
    const PER_VCPU_MS: u64 = 150;
    let scaled = (vcpus as u64).saturating_mul(PER_VCPU_MS);
    let total = BASE_MS.saturating_add(scaled);
    if total > CAP_MS { CAP_MS } else { total }
}

/// Headroom for kernel init, scheduler attach, and BPF verifier time
/// — the post-sys_rdy phase of guest startup. Distinct from
/// [`sys_rdy_budget_ms`]'s base + per-vCPU budget (the pre-sys_rdy
/// virtio-console handshake budget); the two add together to form
/// the full [`vm_boot_headroom`].
const KERNEL_INIT_HEADROOM_MS: u64 = 10_000;

/// The authoritative boot allowance in milliseconds. Kept as a const
/// integer helper so the wall-clock VM deadline and the progress
/// watchdog's Boot CPU budget consume the same contract instead of
/// maintaining independently calibrated width formulas.
pub(crate) const fn vm_boot_headroom_ms(vcpus: u32) -> u64 {
    KERNEL_INIT_HEADROOM_MS.saturating_add(sys_rdy_budget_ms(vcpus))
}

/// Total boot headroom: covers kernel init + scheduler attach + BPF
/// verifier time ([`KERNEL_INIT_HEADROOM_MS`]) plus the guest's scaled
/// `send_sys_rdy` retry loop ([`sys_rdy_budget_ms`]) before the
/// workload phase begins. Scales with vCPU count so the host timeout
/// doesn't fire while the guest is still inside its sys_rdy budget.
pub(crate) fn vm_boot_headroom(vcpus: u32) -> Duration {
    Duration::from_millis(vm_boot_headroom_ms(vcpus))
}

/// Per-phase CPU-time budget (nanoseconds) charged against a single
/// lifecycle phase before the progress watchdog's Tier-1 rule
/// (`vmm::freeze_coord::watchdog_step`) treats "CPU burned without
/// progress" as a wedge. `phase` is the [`crate::monitor::LifecycleStage`]
/// discriminant (Boot=0, Attach=1, Dispatch=2, Body=3, Teardown=4);
/// this fn is deliberately phrased over the raw `u8` so it carries no
/// dependency on that enum and stays testable in isolation.
///
/// Tier-1's evidence is the MAX per-vCPU CPU burned in-phase (the monitor's
/// `max_vcpu_cpu_in_phase_ns`), not the SUM across vCPUs. A spinning wedge
/// is one (or a few) hot thread(s), so the sound budget is what a SINGLE
/// vCPU may legitimately burn in the phase — a per-vCPU-linear budget
/// against a summed evidence was the wide-SMP false-kill bug (256 idle
/// vCPUs summed ~141 s of diffuse background burn and crossed a 64 s
/// budget with zero wedge).
///
/// Boot is the deliberate exception to otherwise width-independent
/// per-vCPU budgets. The BSP serially enumerates and initializes every AP,
/// so the busiest-vCPU signal contains legitimate O(vCPU-count) work even
/// though the signal itself is a MAX rather than a SUM. Its budget reuses
/// [`vm_boot_headroom_ms`], the existing semantic allowance that already
/// governs kernel init, scheduler attach, BPF verification, and the
/// vCPU-scaled sys-rdy handshake. This keeps one source of truth for a
/// healthy boot instead of fitting another base-plus-slope formula to
/// whichever wide distro image happened to fail last. The pthread
/// currency widening is applied by the watchdog on top.
///
/// Every other phase stays flat because its legitimate busiest-vCPU work
/// does not scale with guest width:
///   - Attach (35 s): single-threaded BPF load + verifier time dominates
///     and lands on one vCPU; host-side cold-BTF/vmlinux waits accrue NO
///     guest CPU (the guest is blocked, not spinning), so even a cold
///     parse under host-compile contention leaves the max per-vCPU burn
///     well under this.
///   - Dispatch (8 s): the probe's first-dispatch warm-up on the busiest
///     worker vCPU; 8 s is generous for a single vCPU reaching first
///     dispatch.
///   - Body (`u64::MAX`): the workload phase has no CPU-time budget — a
///     body test is *supposed* to burn CPU, so Tier-1 is structurally
///     disabled here via this sentinel (the watchdog never charges a CPU
///     budget it can exceed). Body wedges are caught by the existing
///     deadline logic, not Tier-1.
///   - Teardown (8 s): unwind / dump / VM-exit on the busiest vCPU.
///
/// Unknown phases (>4) return `u64::MAX` — an id this fn does not model
/// must never be killed on a CPU budget it cannot justify. The pthread
/// currency's 3/2 widening (`widen_budget_for_currency`) still applies on
/// top of these in the watchdog.
pub(crate) const fn phase_cpu_budget_ns(phase: u8, vcpus: u32) -> u64 {
    const S_TO_NS: u64 = 1_000_000_000;
    match phase {
        // Boot: share the authoritative boot-headroom contract.
        0 => vm_boot_headroom_ms(vcpus).saturating_mul(1_000_000),
        // Attach
        1 => 35u64.saturating_mul(S_TO_NS),
        // Dispatch
        2 => 8u64.saturating_mul(S_TO_NS),
        // Body — Tier-1 off (see doc).
        3 => u64::MAX,
        // Teardown
        4 => 8u64.saturating_mul(S_TO_NS),
        // Unknown phase → never kill.
        _ => u64::MAX,
    }
}

/// Per-phase WALL-time backstop (nanoseconds) for the progress
/// watchdog's Tier-2 rule (`vmm::freeze_coord::watchdog_step`): the
/// fully-quiesced idle-wedge case where a phase has made no progress for
/// this long *and* the guest shows no runnable demand. Distinct from the
/// CPU budget above — this is wall clock, so it fires when a phase is
/// stuck making no forward progress regardless of whether CPU is being
/// burned (a spinning wedge is Tier-1's; a *silent* wedge is Tier-2's).
///
/// Each backstop is set STRICTLY ABOVE the existing wall budget that
/// governs the same phase, so Tier-2 is a last-resort backstop that only
/// fires after the phase's own in-band deadline has already had its
/// chance — never a competing, earlier kill:
///   - Boot: at least 45s and always 5s above
///     [`vm_boot_headroom_ms`]. A genuinely silent boot (no epochs,
///     no demand) past this is wedged, not merely slow.
///   - Attach 40s: above [`COLD_BTF_PHASE1_BUDGET`] (30s) — the host's
///     own cold-BTF accessor-build deadline — plus slack, so a slow but
///     live BTF parse is never mistaken for a silent attach wedge.
///   - Dispatch 35s: above the guest probe's `DISPATCH_DEADLINE_CAP`
///     (30s, `vmm::rust_init::verifier_workload`), so the probe verdict
///     — not Tier-2 — decides a live-but-slow dispatch.
///   - Teardown 15s: a short backstop; teardown that makes no progress
///     for this long with no demand is stuck unwinding.
///   - Body (`u64::MAX`): the workload phase has no wall backstop —
///     Tier-2 is structurally off for Body via this sentinel (a body
///     test may legitimately sit quiescent; its deadline is the
///     authoritative bound).
///
/// Unknown phases (>4) return `u64::MAX` — never kill on a wall backstop
/// this fn does not model.
pub(crate) const fn phase_wall_backstop_ns(phase: u8, vcpus: u32) -> u64 {
    const S_TO_NS: u64 = 1_000_000_000;
    const MS_TO_NS: u64 = 1_000_000;
    let s = match phase {
        // Boot is width-sensitive. Preserve the historical 45s floor for
        // small guests while keeping Tier-2 strictly after the authoritative
        // boot allowance for wide guests.
        0 => {
            let scaled_ms = vm_boot_headroom_ms(vcpus).saturating_add(5_000);
            let backstop_ms = if scaled_ms > 45_000 {
                scaled_ms
            } else {
                45_000
            };
            return backstop_ms.saturating_mul(MS_TO_NS);
        }
        // Attach — > COLD_BTF_PHASE1_BUDGET (30s).
        1 => 40u64,
        // Dispatch — > guest DISPATCH_DEADLINE_CAP (30s).
        2 => 35u64,
        // Body — Tier-2 off (see doc).
        3 => return u64::MAX,
        // Teardown
        4 => 15u64,
        // Unknown phase → never kill.
        _ => return u64::MAX,
    };
    s.saturating_mul(S_TO_NS)
}

/// Worst-case host-side latency the guest's `wait_for_map_write` latch
/// blocks on before a `bpf_map_write` test's workload runs: the host
/// builds the BPF-map accessor (ELF + BTF parse + symbol HashMap, ~4 s
/// on a debug vmlinux per the freeze-coord accessor-init comment) in a
/// retry loop bounded by a 30 s `phase1_deadline`. Under heavy `-j16`
/// host-compile contention the parse scales and a cold vmlinux read adds
/// seconds, so the latch can block up to that deadline. Added to the
/// workload budget for any entry declaring a `bpf_map_write` — a
/// framework cost every such test pays, not a per-test concern the
/// author must remember to budget for.
const COLD_BTF_PHASE1_BUDGET: Duration = Duration::from_secs(30);

/// Oversubscription ratio at or beyond which a default/no-perf (auto)
/// overcommit is SKIPPED rather than booted. Above it the host
/// time-slices the vCPU threads so heavily the boot cannot make forward
/// progress, so the dispatch
/// (`run_ktstr_test_inner_impl`) skips with a "host topology
/// insufficient" signal instead of hard-failing. Set above the 4× the
/// `cpu_budget` overcommit tests deliberately exercise (those carry an
/// explicit `cpu_budget`, so they are NOT auto-collapse and never skip
/// on this ratio) and far above the ~1.3× a 256-vCPU guest hits on a
/// 192-CPU CI runner — which therefore RUNS and validates wide-SMP boot
/// rather than skipping, so the boot invariant is never masked.
pub(crate) const OVERCOMMIT_SKIP_RATIO: f64 = 6.0;

/// Flat multiplier on the vCPU-scaled [`vm_boot_headroom`] term of the
/// Tier-3 dead-man deadline ([`vm_timeout_from_entry`] and
/// [`verifier_vm_timeout`]). The approved ~2.5x rounded up to an
/// integer — integer math, and the extra generosity is free here.
///
/// This is NOT a timeout bump. The progress watchdog's Tier-1
/// (CPU-burned-without-progress) and Tier-2 (silent idle-wedge) rules in
/// `vmm::freeze_coord::watchdog_step` are the primary hang
/// detectors for INFRA phases: a wedge dies on those tiers inside a bounded
/// per-phase budget no matter how loose this deadline is. Tier-3 owns the
/// remaining degradation paths: dead monitoring, an inert cell, and an
/// active Body livelock after it burns the whole effective-deadline
/// busiest-vCPU budget. The CPU backstop, like Tier-1, stretches with host
/// starvation; no wall-clock oversub multiplier is needed. A flat generous
/// headroom factor therefore suffices for the deadline itself.
const DEADMAN_HEADROOM_MULT: u32 = 3;

/// Stricter skip ratio for the `expect_auto_repro` chain. That inversion
/// boots a SECOND wide-SMP VM which must replay the forced failure and land
/// a shape-valid `.repro.wprof.pb` — a far more fragile path under host
/// time-slicing than a single boot: the repro VM's system-wide wprof
/// capture over hundreds of vCPUs stops reliably producing a transportable
/// trace once the host oversubscribes (the boots themselves still finish
/// inside the Tier-3 dead-man deadline; the trace transport is what breaks).
/// So this chain auto-skips well below the generic [`OVERCOMMIT_SKIP_RATIO`],
/// while single-VM wide-SMP BOOT tests keep running (and validating boot) up
/// to the generic cap. Tuned to sit between the ~1.3x a 256-vCPU guest hits
/// on the 192-CPU wide-SMP design-target runner (which still RUNS the
/// auto-repro hop, so it is validated there) and the ~2.7x of a 96-CPU host
/// (which skips cleanly instead of hard-failing — the "overcommit OR
/// auto-skip, never hard-fail" contract).
pub(crate) const EXPECT_AUTO_REPRO_SKIP_RATIO: f64 = 2.0;

/// Host overcommit ratio for a `vcpus`-wide guest given the host's
/// allowed-CPU count and the test's optional explicit `cpu_budget`:
/// vCPUs divided by the host CPUs the vCPU threads actually land on.
/// With an explicit `cpu_budget` the threads collapse onto
/// `min(cpu_budget, allowed)` (the per-test cap); without one the
/// default/no-perf path collapses onto the whole allowed cpuset
/// (`no_perf_cpu_budget`'s `(vcpus + 1).min(allowed)` clamp when
/// `allowed < vcpus + 1`, else a fitting 1:1 pin). Under
/// `KTSTR_CARGO_TEST_MODE` the planner ignores the explicit budget and
/// masks to the full allowed cpuset, so with a `cpu_budget` the
/// returned ratio is an UPPER bound there (CI runs `cargo ktstr test`
/// with that mode OFF, where the budget IS enforced and the ratio is
/// exact). Floored at 1.0 (a fitting host is never under-subscribed) and
/// divide-by-zero-guarded (an unenumerable cpuset yields 1.0).
///
/// Feeds only the boot-or-skip gate ([`overcommit_skip_reason`]) and the
/// operator overcommit warning — NOT the Tier-3 dead-man deadline, which
/// no longer scales with this ratio (the progress watchdog's Tier-1/2
/// rules absorb a loaded host). Pure over `(vcpus, allowed_cpus,
/// cpu_budget)` so the skip boundary is unit-testable without reading the
/// host cpuset.
pub(crate) fn overcommit_ratio(vcpus: u32, allowed_cpus: usize, cpu_budget: Option<u32>) -> f64 {
    let allowed = allowed_cpus.max(1);
    let effective = match cpu_budget {
        Some(b) => (b as usize).min(allowed),
        None => allowed,
    }
    .max(1);
    (vcpus as f64 / effective as f64).max(1.0)
}

/// Reason to auto-skip an over-oversubscribed default/no-perf run, or
/// `None` to run it. The default/no-perf path collapses the vCPU threads
/// onto the allowed cpuset (`build_overcommit_run_locks` /
/// `no_perf_cpu_budget`); at or beyond [`OVERCOMMIT_SKIP_RATIO`] the host
/// time-slices so hard the boot cannot make forward progress, so the
/// dispatch
/// (`run_ktstr_test_inner_impl`) skips with this reason — the "overcommit
/// OR auto-skip, never hard-fail" contract. Returns `None` (runs) for:
/// the fitting / mildly-oversubscribed case (< the ratio, e.g. a 256-vCPU
/// guest at ~1.3x on a 192-CPU CI runner, so wide-SMP boot is VALIDATED
/// there, never masked); an explicit `cpu_budget` (a deliberate
/// oversubscription opt-in for contention testing always runs, its deeper
/// ratio being the author's choice); and an empty (unenumerable) cpuset
/// (no ratio is computable, so the overcommit warning is the sole
/// signal). Pure over `(vcpus, allowed_cpus, cpu_budget)` so the skip
/// boundary is unit-testable without booting a VM.
pub(crate) fn overcommit_skip_reason(
    vcpus: u32,
    allowed_cpus: usize,
    cpu_budget: Option<u32>,
    expect_auto_repro: bool,
) -> Option<String> {
    if cpu_budget.is_some() || allowed_cpus == 0 {
        return None;
    }
    let oversub = overcommit_ratio(vcpus, allowed_cpus, None);
    // The two-VM expect_auto_repro inversion uses a much stricter cap
    // ([`EXPECT_AUTO_REPRO_SKIP_RATIO`]) than a single-VM boot test: its
    // repro-VM wprof-trace transport is fragile under time-slicing, so it
    // skips at an oversubscription a boot-only wide-SMP test still runs at.
    let skip_ratio = if expect_auto_repro {
        EXPECT_AUTO_REPRO_SKIP_RATIO
    } else {
        OVERCOMMIT_SKIP_RATIO
    };
    if oversub < skip_ratio {
        return None;
    }
    let chain = if expect_auto_repro {
        " for the expect_auto_repro inversion chain"
    } else {
        ""
    };
    Some(format!(
        "host topology insufficient: {vcpus} vCPUs auto-collapse onto \
         {allowed_cpus} allowed host CPUs = {oversub:.1}x oversubscription \
         (>= {skip_ratio:.0}x skip cap{chain}); widen the process cpuset \
         or shrink the guest topology"
    ))
}

/// Derive the host-side VM timeout from the test entry's watchdog and
/// duration. This is the watchdog's TIER-3 dead-man's switch — the
/// generous wall-clock backstop, NOT the primary hang detector. The
/// primary detectors are the progress watchdog's Tier-1
/// (CPU-burned-without-progress) and Tier-2 (silent idle-wedge) rules in
/// `vmm::freeze_coord::watchdog_step`: a wedge dies on those
/// tiers inside a bounded per-phase budget. Tier-3 also owns a dead
/// monitor, an inert cell, and an active Body livelock after it consumes
/// the whole effective-deadline busiest-vCPU CPU budget; that last bound
/// remains dilation-immune because a starved cell accrues CPU slowly.
///
/// Shape: `base + vm_boot_headroom(booted_vcpus) * DEADMAN_HEADROOM_MULT`.
/// The vCPU-scaled boot headroom covers a slow boot on a large topology;
/// the flat [`DEADMAN_HEADROOM_MULT`] makes it generous. This is NOT a
/// timeout bump and does NOT scale with the host overcommit ratio — the
/// old oversub multiplier existed to stop wall-clock false-timeouts on a
/// loaded host, a job Tier-1/2 now own, so a flat generous factor
/// replaces it (and the overcommit-scaling machinery is retired from this
/// path). `base` is the guest's own workload/watchdog budget:
/// [`COLD_BTF_PHASE1_BUDGET`] is added when the entry declares a
/// `bpf_map_write` (the guest blocks on the host's cold-BTF accessor
/// build before the workload starts), and
/// [`crate::vmm::freeze_coord::WPROF_SHIP_GRACE`] when it declares
/// `wprof` (a crashing scheduler's late Phase-5 trace ship is held for
/// that window before teardown). The soft (reset) deadline rides this
/// Tier-3 value via `max(reset, hard_deadline)`.
///
/// `booted_vcpus` is the vCPU count of the topology the VM actually
/// boots (`resolve_vm_topology(entry, topo).0.total_cpus()`), NOT the
/// declared `entry.topology`: under a `TopoOverride` (gauntlet preset /
/// `--ktstr-topo`) they diverge, and the vCPU-scaled boot headroom must
/// scale to the topology that boots — otherwise the deadline is sized for
/// a smaller-than-booted preset.
pub(crate) fn vm_timeout_from_entry(
    entry: &super::entry::KtstrTestEntry,
    booted_vcpus: u32,
) -> Duration {
    let mut base = entry
        .watchdog_timeout
        .max(entry.duration)
        .max(Duration::from_secs(1));
    if !entry.bpf_map_write.is_empty() {
        base += COLD_BTF_PHASE1_BUDGET;
    }
    // A wprof entry's scheduler may crash; on an error-class exit the
    // freeze coordinator holds the VM open up to `WPROF_SHIP_GRACE` for
    // the guest's late Phase-5 wprof trace ship before killing. Add that
    // window to the host budget so a late crash's full ship grace fits
    // inside the watchdog deadline (mirrors COLD_BTF_PHASE1_BUDGET).
    if entry.wprof {
        base += crate::vmm::freeze_coord::WPROF_SHIP_GRACE;
    }
    base + vm_boot_headroom(booted_vcpus) * DEADMAN_HEADROOM_MULT
}

/// Base host-side budget for a verifier sweep cell's guest lifecycle
/// past boot: scheduler spawn, BPF load/verify, attach gate, the
/// dispatch probe, and teardown. The boot phase is budgeted
/// separately by the vCPU-scaled dead-man headroom in
/// [`verifier_vm_timeout`], mirroring [`vm_timeout_from_entry`]'s
/// base-vs-headroom split.
pub(crate) const VERIFIER_BASE_TIMEOUT: Duration = Duration::from_secs(120);

/// Post-attach budget for a verifier cell, wired into the builder's
/// `workload_duration` so the watchdog's scheduler-attach reset arms:
/// a cell whose boot ate most of the deadline still gets this much
/// wall time AFTER attach for the dispatch probe + teardown (the
/// reset is extend-only — `max(reset, hard_deadline)`). Sized above
/// the guest probe's `DISPATCH_DEADLINE_CAP` (30s, see
/// `vmm::rust_init::verifier_workload`) plus teardown/dump slack.
/// Before this, verifier cells passed no `workload_duration` at all,
/// which made the attach-reset dead code on the sweep path — the
/// watchdog dump always read `reset_armed_by=none`.
pub(crate) const VERIFIER_WORKLOAD_BUDGET: Duration = Duration::from_secs(60);

/// Host-side VM timeout for a verifier sweep cell: the flat
/// [`VERIFIER_BASE_TIMEOUT`] plus the same vCPU-scaled dead-man boot
/// headroom the `#[ktstr_test]` path gets from [`vm_timeout_from_entry`]
/// — its Tier-3 backstop, not the primary hang detector (Tier-1/2 in
/// `vmm::freeze_coord::watchdog_step` own that). The sweep
/// previously hard-coded 120s with no headroom at all, so a
/// wide-topology cell (128 vCPUs) that boots fine but slowly under CI
/// concurrency was killed mid-attach — reported by the scx verifier
/// sweep as a VM timeout (now worded "hung with no confirmed scheduler
/// attach" for that shape).
pub(crate) fn verifier_vm_timeout(booted_vcpus: u32) -> Duration {
    VERIFIER_BASE_TIMEOUT + vm_boot_headroom(booted_vcpus) * DEADMAN_HEADROOM_MULT
}

/// Configure the ktstr_test VM builder prefix shared by the main
/// test path ([`super::eval::run_ktstr_test_inner`]) and the
/// auto-repro path ([`super::probe::attempt_auto_repro`]).
///
/// Applies, in order: kernel, init binary, topology, memory floor,
/// guest cmdline, SHM size, guest argv, host-side timeout, perf-mode
/// disable flag, optional scheduler binary, every queued BPF map
/// write, and the scheduler watchdog timeout.
///
/// The caller owns the divergent tail. `run_ktstr_test_inner`
/// additionally wires `performance_mode`,
/// `sched_enable_cmds`/`sched_disable_cmds` for kernel-built
/// schedulers, and `monitor_thresholds`. `attempt_auto_repro`
/// additionally wires `include_files` plus base `sched_args`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_vm_builder_base(
    entry: &KtstrTestEntry,
    kernel: &Path,
    ktstr_bin: &Path,
    scheduler: Option<&Path>,
    staged_schedulers: &[(String, std::path::PathBuf, Vec<String>)],
    vm_topology: crate::vmm::topology::Topology,
    memory_mib: u32,
    cmdline_extra: &str,
    guest_args: &[String],
    no_perf_mode: bool,
) -> crate::vmm::KtstrVmBuilder {
    // The base builder deliberately does NOT set
    // `failure_dump_path` — the per-VM target is caller-specific
    // (primary vs auto-repro). Stale-file pre-clear lives at the
    // dispatch sites (`test_support::eval` for primary;
    // `test_support::probe::attempt_auto_repro` for repro), not
    // inside the setter or this base call. The setter is pure
    // (no FS side effects); placing the pre-clear in the dispatch
    // layer prevents the auto-repro path's reuse of this base
    // builder from accidentally erasing the primary dump that
    // just landed.
    let mut builder = crate::vmm::KtstrVm::builder()
        .kernel(kernel)
        // Prebuilt distro kernels ship virtio as modules; embed the
        // ordered boot-module set from the cache entry beside the image
        // (empty for built kernels — a no-op).
        .kernel_modules(crate::cache::boot_modules_for_image(kernel))
        .initrd_compression(crate::cache::initrd_compression_for_image(kernel))
        .init_binary(ktstr_bin)
        .topology(vm_topology)
        .memory_deferred_min(memory_mib)
        .cmdline(cmdline_extra)
        .run_args(guest_args)
        .timeout(vm_timeout_from_entry(entry, vm_topology.total_cpus()))
        .workload_duration(entry.duration)
        .no_perf_mode(no_perf_mode);

    // Per-test no-perf CPU budget override (#[ktstr_test(cpu_budget = N)]).
    // None leaves the builder's auto-size (vCPU count plus one service CPU,
    // clamped to the allowed cpuset) in place; only the no-perf path consumes it.
    if let Some(budget) = entry.cpu_budget {
        builder = builder.cpu_budget(budget);
    }

    if let Some(sched_path) = scheduler {
        builder = builder.scheduler_binary(sched_path);
    }

    // Push each pre-resolved staged scheduler into the builder's
    // staging set. Caller is responsible for running each entry
    // through resolve_scheduler so this fn stays
    // infallible (sibling to the boot-time `scheduler: Option<&Path>`
    // shape which is also caller-resolved). KernelBuiltin / Eevdf
    // staged entries (no binary to resolve) are skipped at the
    // caller side; only resolved (name, host_binary, sched_args)
    // tuples reach this loop.
    for (name, host_path, sched_args) in staged_schedulers {
        builder = builder.staged_scheduler(name.clone(), host_path.clone(), sched_args.clone());
    }

    // Opt-in jemalloc-probe wiring. An integration test that needs
    // the probe (see `tests/jemalloc_probe_tests.rs`) sets
    // `KTSTR_JEMALLOC_PROBE_BINARY` to the absolute host path of
    // `ktstr-jemalloc-probe` via `#[ctor]` before the test harness
    // dispatches. When set, the probe is packed into every VM's
    // base initramfs; the init binary stays stripped because the
    // paired alloc-worker carries DWARF. Absent env var = existing
    // behavior (no probe).
    //
    // Required ctor shape in a new test file that needs the probe
    // in the guest — paste verbatim, adjust the two binary names.
    // Either ctor form works (ktstr re-exports both): the proc-macro
    // attribute shown below, or the declarative
    // `::ktstr::__private::ctor::declarative::ctor! { ... }` block
    // form (ktstr's own in-tree sites use the declarative form per
    // src/test_support/dispatch.rs).
    //
    // ```ignore
    // #[::ktstr::__private::ctor::ctor(unsafe, crate_path = ::ktstr::__private::ctor)]
    // fn set_probe_binary_env_var() {
    //     // SAFETY: ctor runs before any `#[ktstr_test]` thread or
    //     // probe thread spawns; glibc's `__environ` mutation is
    //     // single-threaded here.
    //     unsafe {
    //         std::env::set_var(
    //             ::ktstr::KTSTR_JEMALLOC_PROBE_BINARY_ENV,
    //             env!("CARGO_BIN_EXE_ktstr-jemalloc-probe"),
    //         );
    //         std::env::set_var(
    //             ::ktstr::KTSTR_JEMALLOC_ALLOC_WORKER_BINARY_ENV,
    //             env!("CARGO_BIN_EXE_ktstr-jemalloc-alloc-worker"),
    //         );
    //     }
    // }
    // ```
    //
    // Declarative-form equivalent (no `crate_path = ` plumbing required
    // because the macro_rules! expansion resolves paths via `$crate`):
    //
    // ```ignore
    // ::ktstr::__private::ctor::declarative::ctor! {
    // #[ctor(unsafe)]
    // fn set_probe_binary_env_var() {
    //     // SAFETY: same as proc-macro form above.
    //     unsafe {
    //         std::env::set_var(
    //             ::ktstr::KTSTR_JEMALLOC_PROBE_BINARY_ENV,
    //             env!("CARGO_BIN_EXE_ktstr-jemalloc-probe"),
    //         );
    //         std::env::set_var(
    //             ::ktstr::KTSTR_JEMALLOC_ALLOC_WORKER_BINARY_ENV,
    //             env!("CARGO_BIN_EXE_ktstr-jemalloc-alloc-worker"),
    //         );
    //     }
    // }
    // }
    // ```
    //
    // The `crate_path = ::ktstr::__private::ctor` argument is
    // non-negotiable: `#[ctor::ctor(unsafe)]` without the
    // re-export path panics at compile time because the `ctor`
    // crate is not listed in the test crate's direct deps. ktstr
    // re-exports `ctor` under `__private::ctor` exactly so test
    // authors do not need to add it themselves. ctor 1.0 also
    // mandates the `unsafe` marker as the first attribute
    // argument; bare `#[ctor::ctor]` no longer compiles.
    if let Ok(probe_path) = std::env::var(crate::KTSTR_JEMALLOC_PROBE_BINARY_ENV)
        && !probe_path.is_empty()
    {
        // Pack the probe binary into the guest initramfs at
        // `/bin/ktstr-jemalloc-probe`. Closed-loop probe tests run
        // the probe via `--pid <alloc_worker_pid>` against the
        // paired `ktstr-jemalloc-alloc-worker` target; DWARF comes
        // from the worker's own ELF, not the init's.
        builder = builder.jemalloc_probe_binary(std::path::PathBuf::from(probe_path));
    }
    if let Ok(worker_path) = std::env::var(crate::KTSTR_JEMALLOC_ALLOC_WORKER_BINARY_ENV)
        && !worker_path.is_empty()
    {
        // Pack the jemalloc-alloc-worker binary alongside the
        // probe. Only the cross-process closed-loop test sets
        // this; scheduler-only tests leave the env var unset and
        // skip the wiring.
        builder = builder.jemalloc_alloc_worker_binary(std::path::PathBuf::from(worker_path));
    }

    for bpf_write in entry.bpf_map_write {
        builder = builder.bpf_map_write(
            bpf_write.map_name_suffix(),
            bpf_write.field(),
            bpf_write.value(),
        );
    }

    for watch in entry.watch_bpf_maps {
        builder = builder.watch_bpf_map(
            watch.map_name_suffix(),
            watch.field(),
            watch.agg(),
            watch.label(),
        );
    }

    if let Some(disk_cfg) = entry.disk.clone() {
        builder = builder.disk(disk_cfg);
    }

    for net_cfg in entry.networks {
        builder = builder.network(*net_cfg);
    }

    builder = builder.num_snapshots(entry.num_snapshots);

    if let Some(root) = entry.workload_root_cgroup {
        builder = builder.workload_root_cgroup(root.as_str().to_string());
    }
    if let Some(parent) = entry.scheduler.cgroup_parent {
        builder = builder.scheduler_cgroup_parent(parent.as_str().to_string());
    }

    builder.watchdog_timeout(entry.watchdog_timeout)
}

#[cfg(test)]
mod tests {
    use super::super::entry::Scheduler;
    use super::super::test_helpers::{EnvVarGuard, lock_env};
    use super::*;

    #[test]
    fn vm_timeout_from_entry_adds_cold_btf_budget_for_bpf_map_write() {
        use super::super::entry::{BpfMapWrite, KtstrTestEntry};
        static W: BpfMapWrite = BpfMapWrite::new(".bss", "crash", 0);
        static WS: &[&BpfMapWrite] = &[&W];
        let no_write = KtstrTestEntry {
            name: "no_write",
            ..KtstrTestEntry::DEFAULT
        };
        let with_write = KtstrTestEntry {
            name: "with_write",
            bpf_map_write: WS,
            ..KtstrTestEntry::DEFAULT
        };
        // The cold-BTF phase-1 budget is added to the workload base only
        // when the entry declares a host-side bpf_map_write; the delta
        // between an otherwise-identical pair is exactly that budget.
        assert_eq!(
            vm_timeout_from_entry(&with_write, with_write.topology.total_cpus()),
            vm_timeout_from_entry(&no_write, no_write.topology.total_cpus())
                + COLD_BTF_PHASE1_BUDGET,
            "bpf_map_write entries get the cold-BTF phase-1 budget added",
        );
    }

    #[test]
    fn no_perf_mode_active_true_when_env_set_to_value() {
        let _l = lock_env();
        let _g = EnvVarGuard::set(crate::KTSTR_NO_PERF_MODE_ENV, "1");
        assert!(no_perf_mode_active());
    }

    #[test]
    fn no_perf_mode_active_false_when_env_unset() {
        let _l = lock_env();
        let _g = EnvVarGuard::remove(crate::KTSTR_NO_PERF_MODE_ENV);
        assert!(!no_perf_mode_active());
    }

    /// Regression pin: empty-string-as-unset contract. Before the
    /// env-var-sweep cleanup, the bare `is_ok()` reader returned
    /// true on
    /// `KTSTR_NO_PERF_MODE=` (set but empty — e.g. Docker
    /// `--env KTSTR_NO_PERF_MODE` pass-through fired without a
    /// value), silently flipping perf-mode OFF for every
    /// `performance_mode` test. The fix at L146 treats
    /// empty-as-unset; this test pins that contract for ALL
    /// consumer sites (shell-mode VM at lib.rs, verifier at
    /// verifier.rs, dispatch + eval) since they all route
    /// through this helper.
    #[test]
    fn no_perf_mode_active_false_when_env_set_to_empty_string() {
        let _l = lock_env();
        let _g = EnvVarGuard::set(crate::KTSTR_NO_PERF_MODE_ENV, "");
        assert!(
            !no_perf_mode_active(),
            "empty-string env must be treated as UNSET — a regression \
             here flips perf-mode for every consumer that routes \
             through no_perf_mode_active",
        );
    }

    #[test]
    fn perf_only_active_true_when_env_set_to_value() {
        let _l = lock_env();
        let _g = EnvVarGuard::set(crate::KTSTR_PERF_ONLY_ENV, "1");
        assert!(perf_only_active());
    }

    #[test]
    fn perf_only_active_false_when_env_unset() {
        let _l = lock_env();
        let _g = EnvVarGuard::remove(crate::KTSTR_PERF_ONLY_ENV);
        assert!(!perf_only_active());
    }

    /// Empty-as-unset contract (mirrors `no_perf_mode_active`): a
    /// `KTSTR_PERF_ONLY=` pass-through must NOT silently skip every
    /// non-perf test.
    #[test]
    fn perf_only_active_false_when_env_set_to_empty_string() {
        let _l = lock_env();
        let _g = EnvVarGuard::set(crate::KTSTR_PERF_ONLY_ENV, "");
        assert!(
            !perf_only_active(),
            "empty-string env must be treated as UNSET",
        );
    }

    /// Selection logic: with perf-only active, a non-performance_mode
    /// entry is skipped while a performance_mode entry is kept. When
    /// perf-only is inactive, neither is skipped.
    #[test]
    fn perf_only_skips_entry_keeps_perf_skips_others() {
        use super::super::entry::KtstrTestEntry;
        let perf = KtstrTestEntry {
            name: "perf",
            performance_mode: true,
            ..KtstrTestEntry::DEFAULT
        };
        let plain = KtstrTestEntry {
            name: "plain",
            performance_mode: false,
            ..KtstrTestEntry::DEFAULT
        };

        let _l = lock_env();
        {
            let _g = EnvVarGuard::set(crate::KTSTR_PERF_ONLY_ENV, "1");
            assert!(
                !perf_only_skips_entry(&perf),
                "a performance_mode test is the selection target, never skipped",
            );
            assert!(
                perf_only_skips_entry(&plain),
                "a non-performance_mode test must be skipped under perf-only",
            );
        }
        let _g = EnvVarGuard::remove(crate::KTSTR_PERF_ONLY_ENV);
        assert!(!perf_only_skips_entry(&perf));
        assert!(
            !perf_only_skips_entry(&plain),
            "perf-only inactive => nothing is skipped on this axis",
        );
    }

    #[test]
    fn bypass_llc_locks_active_true_when_env_set_to_value() {
        let _l = lock_env();
        let _g = EnvVarGuard::set(crate::KTSTR_BYPASS_LLC_LOCKS_ENV, "1");
        assert!(bypass_llc_locks_active());
    }

    #[test]
    fn bypass_llc_locks_active_false_when_env_unset() {
        let _l = lock_env();
        let _g = EnvVarGuard::remove(crate::KTSTR_BYPASS_LLC_LOCKS_ENV);
        assert!(!bypass_llc_locks_active());
    }

    /// Regression pin: empty-string-as-unset contract for
    /// KTSTR_BYPASS_LLC_LOCKS. A bare `KTSTR_BYPASS_LLC_LOCKS=`
    /// (CI shell / Docker `--env` pass-through without value)
    /// must NOT activate the bypass. The helper enforces this
    /// uniformly for all 7 reader sites (vmm/builder.rs,
    /// cli/kernel_build/build.rs ×2, bin/ktstr.rs ×2,
    /// bin/cargo_ktstr/{kernel/mod, misc/shell}) — a regression
    /// here flips the contention contract for every caller.
    #[test]
    fn bypass_llc_locks_active_false_when_env_set_to_empty_string() {
        let _l = lock_env();
        let _g = EnvVarGuard::set(crate::KTSTR_BYPASS_LLC_LOCKS_ENV, "");
        assert!(
            !bypass_llc_locks_active(),
            "empty-string env must be treated as UNSET per the contract \
             shared with no_perf_mode_active — a regression here flips \
             LLC flock contention enforcement for every reader",
        );
    }

    #[test]
    fn config_file_parts_nested_path() {
        static SCHED: Scheduler = Scheduler::named("cfg").config_file("configs/my_sched.toml");
        let entry = KtstrTestEntry {
            name: "cfg_test",
            scheduler: &SCHED,
            ..KtstrTestEntry::DEFAULT
        };
        let (archive, host, guest) = config_file_parts(&entry).unwrap();
        assert_eq!(archive, "include-files/my_sched.toml");
        assert_eq!(host, PathBuf::from("configs/my_sched.toml"));
        assert_eq!(guest, "/include-files/my_sched.toml");
    }

    #[test]
    fn config_file_parts_bare_filename() {
        static SCHED: Scheduler = Scheduler::named("cfg").config_file("config.toml");
        let entry = KtstrTestEntry {
            name: "cfg_bare",
            scheduler: &SCHED,
            ..KtstrTestEntry::DEFAULT
        };
        let (archive, host, guest) = config_file_parts(&entry).unwrap();
        assert_eq!(archive, "include-files/config.toml");
        assert_eq!(host, PathBuf::from("config.toml"));
        assert_eq!(guest, "/include-files/config.toml");
    }

    #[test]
    fn config_file_parts_none_when_unset() {
        let entry = KtstrTestEntry {
            name: "no_cfg",
            ..KtstrTestEntry::DEFAULT
        };
        assert!(config_file_parts(&entry).is_none());
    }

    // -- build_cmdline_extra --

    use super::super::entry::{KtstrTestEntry, Sysctl};

    #[test]
    fn build_cmdline_extra_default_is_sidecar_only() {
        let _lock = lock_env();
        // Make sure the env does not inject spurious RUST_BACKTRACE /
        // RUST_LOG entries that would break the default assertion.
        let _env_bt = EnvVarGuard::remove("RUST_BACKTRACE");
        let _env_log = EnvVarGuard::remove("RUST_LOG");
        // Pin KTSTR_SIDECAR_DIR so the propagation token shape is
        // stable across tests; without the override, the call falls
        // through to the `{kernel}-{commit}` resolver whose output
        // depends on the test process's git state.
        let _env_sd = EnvVarGuard::set(crate::KTSTR_SIDECAR_DIR_ENV, "/tmp/ktstr-test");

        let entry = KtstrTestEntry {
            name: "cmdline_test",
            ..KtstrTestEntry::DEFAULT
        };
        let out = build_cmdline_extra(&entry);
        assert_eq!(out, "KTSTR_SIDECAR_DIR=/tmp/ktstr-test");
    }

    #[test]
    fn build_cmdline_extra_appends_sysctls_kargs() {
        let _lock = lock_env();
        let _env_bt = EnvVarGuard::remove("RUST_BACKTRACE");
        let _env_log = EnvVarGuard::remove("RUST_LOG");
        let _env_sd = EnvVarGuard::set(crate::KTSTR_SIDECAR_DIR_ENV, "/tmp/ktstr-test");

        static SYSCTLS: &[Sysctl] = &[Sysctl::new("kernel.foo", "1")];
        static SCHED: Scheduler = Scheduler::named("s").sysctls(SYSCTLS).kargs(&["quiet"]);
        let entry = KtstrTestEntry {
            name: "cmd",
            scheduler: &SCHED,
            ..KtstrTestEntry::DEFAULT
        };
        let out = build_cmdline_extra(&entry);
        assert_eq!(
            out,
            "sysctl.kernel.foo=1 quiet KTSTR_SIDECAR_DIR=/tmp/ktstr-test"
        );
    }

    #[test]
    fn build_cmdline_extra_emits_probe_dump_gate_only_when_opted_in() {
        let _lock = lock_env();
        let _env_bt = EnvVarGuard::remove("RUST_BACKTRACE");
        let _env_log = EnvVarGuard::remove("RUST_LOG");
        let _env_sd = EnvVarGuard::set(crate::KTSTR_SIDECAR_DIR_ENV, "/tmp/ktstr-test");

        let default = KtstrTestEntry {
            name: "probe_dump_gate_default",
            ..KtstrTestEntry::DEFAULT
        };
        assert!(
            !build_cmdline_extra(&default).contains("KTSTR_AWAIT_PROBE_DUMP_READY"),
            "the default path must not pay the probe-dump readiness gate"
        );

        let opted_in = KtstrTestEntry {
            name: "probe_dump_gate_enabled",
            probe_dump_ready_gate: true,
            ..KtstrTestEntry::DEFAULT
        };
        assert_eq!(
            build_cmdline_extra(&opted_in),
            "KTSTR_AWAIT_PROBE_DUMP_READY=1 KTSTR_SIDECAR_DIR=/tmp/ktstr-test"
        );
    }

    #[test]
    fn build_auto_repro_cmdline_extra_omits_primary_probe_dump_gate() {
        let _lock = lock_env();
        let _env_bt = EnvVarGuard::remove("RUST_BACKTRACE");
        let _env_log = EnvVarGuard::remove("RUST_LOG");
        let _env_sd = EnvVarGuard::set(crate::KTSTR_SIDECAR_DIR_ENV, "/tmp/ktstr-test");

        let opted_in = KtstrTestEntry {
            name: "probe_dump_gate_auto_repro",
            probe_dump_ready_gate: true,
            ..KtstrTestEntry::DEFAULT
        };
        assert_eq!(
            build_auto_repro_cmdline_extra(&opted_in),
            "KTSTR_SIDECAR_DIR=/tmp/ktstr-test",
            "auto-repro must not inherit the primary-only probe-dump gate"
        );
    }

    #[test]
    fn scheduler_kargs_cannot_forge_or_disable_probe_dump_gate() {
        let _lock = lock_env();
        let _env_bt = EnvVarGuard::remove("RUST_BACKTRACE");
        let _env_log = EnvVarGuard::remove("RUST_LOG");
        let _env_sd = EnvVarGuard::set(crate::KTSTR_SIDECAR_DIR_ENV, "/tmp/ktstr-test");

        static SCHED: Scheduler = Scheduler::named("probe-gate-kargs").kargs(&[
            "quiet",
            "KTSTR_AWAIT_PROBE_DUMP_READY=0",
            "KTSTR_AWAIT_PROBE_DUMP_READY=1",
        ]);
        let disabled = KtstrTestEntry {
            name: "probe_gate_kargs_disabled",
            scheduler: &SCHED,
            ..KtstrTestEntry::DEFAULT
        };
        assert_eq!(
            build_cmdline_extra(&disabled),
            "quiet KTSTR_SIDECAR_DIR=/tmp/ktstr-test"
        );

        let enabled = KtstrTestEntry {
            name: "probe_gate_kargs_enabled",
            scheduler: &SCHED,
            probe_dump_ready_gate: true,
            ..KtstrTestEntry::DEFAULT
        };
        let primary = build_cmdline_extra(&enabled);
        assert_eq!(
            primary,
            "quiet KTSTR_AWAIT_PROBE_DUMP_READY=1 KTSTR_SIDECAR_DIR=/tmp/ktstr-test"
        );
        assert_eq!(
            primary
                .split_ascii_whitespace()
                .filter(|token| *token == "KTSTR_AWAIT_PROBE_DUMP_READY=1")
                .count(),
            1
        );
        assert_eq!(
            build_auto_repro_cmdline_extra(&enabled),
            "quiet KTSTR_SIDECAR_DIR=/tmp/ktstr-test"
        );
    }

    #[test]
    fn build_cmdline_extra_propagates_rust_env() {
        let _lock = lock_env();
        let _env_bt = EnvVarGuard::set("RUST_BACKTRACE", "1");
        let _env_log = EnvVarGuard::set("RUST_LOG", "debug");
        let _env_sd = EnvVarGuard::set(crate::KTSTR_SIDECAR_DIR_ENV, "/tmp/ktstr-test");

        let entry = KtstrTestEntry {
            name: "cmd",
            ..KtstrTestEntry::DEFAULT
        };
        let out = build_cmdline_extra(&entry);
        assert!(
            out.contains("RUST_BACKTRACE=1"),
            "expected RUST_BACKTRACE propagation: {out}"
        );
        assert!(
            out.contains("RUST_LOG=debug"),
            "expected RUST_LOG propagation: {out}"
        );
        assert!(
            out.contains("KTSTR_SIDECAR_DIR=/tmp/ktstr-test"),
            "expected KTSTR_SIDECAR_DIR propagation: {out}"
        );
    }

    #[test]
    fn build_cmdline_extra_propagates_sidecar_dir() {
        let _lock = lock_env();
        let _env_bt = EnvVarGuard::remove("RUST_BACKTRACE");
        let _env_log = EnvVarGuard::remove("RUST_LOG");
        // Explicit override path proves the token shape is exactly
        // `KTSTR_SIDECAR_DIR=<path>` and uses the override verbatim
        // (host's `sidecar_dir()` honours the env var as the
        // operator-chosen override slot).
        let _env_sd = EnvVarGuard::set(crate::KTSTR_SIDECAR_DIR_ENV, "/explicit/sidecar/dir");

        let entry = KtstrTestEntry {
            name: "cmd",
            ..KtstrTestEntry::DEFAULT
        };
        let out = build_cmdline_extra(&entry);
        assert_eq!(out, "KTSTR_SIDECAR_DIR=/explicit/sidecar/dir");
    }

    // -- resolve_vm_topology --

    #[test]
    fn resolve_vm_topology_override_is_verbatim() {
        let entry = KtstrTestEntry {
            name: "topo_test",
            ..KtstrTestEntry::DEFAULT
        };
        let over = super::super::topo::TopoOverride {
            numa_nodes: 2,
            llcs: 4,
            cores: 8,
            threads: 2,
            memory_mib: 4096,
        };
        let (topo, mem) = resolve_vm_topology(&entry, Some(&over));
        assert_eq!(mem, 4096);
        assert_eq!(topo.llcs, 4);
        assert_eq!(topo.cores_per_llc, 8);
        assert_eq!(topo.threads_per_core, 2);
        assert_eq!(topo.numa_nodes, 2);
    }

    #[test]
    fn resolve_vm_topology_default_memory_can_stay_below_two_gib() {
        // The default topology has two vCPUs, so cpu scaling yields
        // 128 MiB and the framework-wide 256-MiB floor wins. This
        // specifically guards against restoring a stale multi-GiB
        // entry default that would defeat deferred payload sizing.
        let entry = KtstrTestEntry {
            name: "tiny",
            ..KtstrTestEntry::DEFAULT
        };
        let (_topo, mem) = resolve_vm_topology(&entry, None);
        assert_eq!(mem, 256, "memory floor = 256 MiB, got {mem}");
        assert!(mem < 2048, "default memory must not pin every VM at 2 GiB");
    }

    #[test]
    fn resolve_vm_topology_none_honors_entry_memory_mib() {
        // Entry with explicit memory_mib above the cpu*64 and 256 floors.
        let entry = KtstrTestEntry {
            name: "mem",
            memory_mib: 1536,
            ..KtstrTestEntry::DEFAULT
        };
        let (_topo, mem) = resolve_vm_topology(&entry, None);
        assert_eq!(
            mem, 1536,
            "a real explicit override must win even when it is not the old 2-GiB default"
        );
    }

    #[cfg(feature = "wprof")]
    #[test]
    fn resolve_vm_topology_wprof_does_not_oversize_at_minimal_default() {
        // wprof=true with the minimal default arena must NOT oversize:
        // the derived wprof floor (16 MiB) is below the entry's own
        // memory, so the entry-derived path sizes the VM exactly like
        // the same non-wprof entry. Guards against restoring a
        // multi-GiB wprof floor that pinned every wprof cell high.
        let entry = KtstrTestEntry {
            name: "wprof_minimal",
            memory_mib: 512,
            wprof: true,
            ..KtstrTestEntry::DEFAULT
        };
        // Compile-time invariant: at the minimal default the wprof
        // floor stays at/under the universal 256 MiB floor, so it is
        // subsumed and cannot oversize a VM.
        const { assert!(crate::vmm::wprof::WPROF_MIN_MEMORY_MIB <= 256) };
        let (_topo, mem) = resolve_vm_topology(&entry, None);
        assert_eq!(
            mem, 512,
            "wprof must not oversize past the entry's own 512 MiB; got {mem}"
        );
    }

    #[cfg(feature = "wprof")]
    #[test]
    fn resolve_vm_topology_wprof_no_bump_when_already_above_floor() {
        // Entry already above the wprof floor — must be honored unchanged.
        let entry = KtstrTestEntry {
            name: "wprof_high",
            memory_mib: 8192,
            wprof: true,
            ..KtstrTestEntry::DEFAULT
        };
        let (_topo, mem) = resolve_vm_topology(&entry, None);
        assert_eq!(
            mem, 8192,
            "memory_mib above WPROF_MIN_MEMORY_MIB must be honored \
             unchanged, got {mem}"
        );
    }

    #[test]
    fn resolve_vm_topology_wprof_disabled_does_not_floor() {
        // wprof=false: the wprof floor must NOT apply, even when
        // entry.memory_mib falls below WPROF_MIN_MEMORY_MIB. Only
        // the universal 256 floor + cpu*64 derivation apply.
        let entry = KtstrTestEntry {
            name: "no_wprof",
            memory_mib: 512,
            wprof: false,
            ..KtstrTestEntry::DEFAULT
        };
        let (_topo, mem) = resolve_vm_topology(&entry, None);
        assert_eq!(
            mem, 512,
            "wprof=false must not invoke the WPROF_MIN_MEMORY_MIB \
             floor, got {mem}"
        );
    }

    #[test]
    fn derive_test_memory_mib_baseline_without_wprof() {
        let entry = KtstrTestEntry {
            name: "baseline",
            ..KtstrTestEntry::DEFAULT
        };
        let mem = derive_test_memory_mib(2, &entry);
        assert_eq!(mem, 256, "2 cpus * 64 = 128, floor 256 wins");
        assert_eq!(
            derive_test_memory_min_mib(2, entry.memory_mib, entry.wprof),
            mem,
            "the admission-stamp scalar projection must equal entry-based VM sizing",
        );
    }

    #[test]
    fn cpu_scaled_memory_mib_scales_and_floors() {
        // Shared scaling core used by both derive_test_memory_mib and
        // the verifier cell. Below the crossover (256/64 = 4 cpus) the
        // 256 MiB floor wins; at/above it the 64 MiB/cpu term wins.
        assert_eq!(cpu_scaled_memory_mib(1), 256, "1 cpu * 64 = 64, floor 256");
        assert_eq!(cpu_scaled_memory_mib(4), 256, "4 cpus * 64 = 256, tie");
        assert_eq!(cpu_scaled_memory_mib(8), 512, "8 cpus * 64 = 512 wins");
        assert_eq!(cpu_scaled_memory_mib(16), 1024, "16 cpus * 64 = 1024 wins");
    }

    #[test]
    fn stamped_wprof_memory_floor_is_feature_independent() {
        // The floor is applied by value, not by cfg!(feature = "wprof"):
        // a below-floor raw is bumped, an at/above-floor raw passes
        // through — identically whether or not the reader binary was
        // built with wprof.
        assert_eq!(
            apply_wprof_memory_floor(WPROF_MIN_MEMORY_MIB - 1, true),
            WPROF_MIN_MEMORY_MIB,
            "a raw below the derived floor must be bumped up to it",
        );
        assert_eq!(
            apply_wprof_memory_floor(WPROF_MIN_MEMORY_MIB, true),
            WPROF_MIN_MEMORY_MIB,
            "a raw at the floor passes through (strict-less-than)",
        );
        assert_eq!(apply_wprof_memory_floor(4096, true), 4096);
        assert_eq!(apply_wprof_memory_floor(4096, false), 4096);
        // At the minimal default arena the floor (16 MiB) is far below
        // the universal 256 MiB floor, so admission sizing is dominated
        // by the latter regardless of the wprof bit — the whole point
        // of the minimal default is that wprof no longer oversizes.
        assert_eq!(derive_test_memory_min_mib(2, 768, true), 768);
        assert_eq!(derive_test_memory_min_mib(2, 768, false), 768);
    }

    #[test]
    fn checked_admission_memory_scaling_rejects_overflow() {
        assert_eq!(checked_cpu_scaled_memory_mib(u32::MAX), None);
        assert_eq!(
            checked_derive_test_memory_min_mib(u32::MAX, 256, false),
            None
        );
        assert_eq!(checked_verifier_preset_memory_min_mib(u32::MAX, 4096), None);
    }

    #[test]
    fn verifier_preset_memory_caps_synthetic_cpu_scaling() {
        assert_eq!(verifier_preset_memory_min_mib(4, 2048), 256);
        assert_eq!(verifier_preset_memory_min_mib(32, 2048), 2048);
        assert_eq!(verifier_preset_memory_min_mib(64, 2048), 2048);
        assert_eq!(verifier_preset_memory_min_mib(252, 4096), 4096);
    }

    #[test]
    fn cpu_scaled_memory_mib_backs_derive_test_memory_mib() {
        // The entry-aware wrapper must reduce to the shared core when no
        // per-entry override raises it — the invariant that lets the
        // direct verifier API reuse cpu_scaled_memory_mib directly.
        let entry = KtstrTestEntry {
            name: "shared",
            ..KtstrTestEntry::DEFAULT
        };
        for cpus in [1u32, 4, 8, 32] {
            assert_eq!(
                derive_test_memory_mib(cpus, &entry),
                cpu_scaled_memory_mib(cpus),
                "derive_test_memory_mib must equal the shared core for \
                 {cpus} cpus when the entry uses the framework minimum"
            );
        }
    }

    #[cfg(feature = "wprof")]
    #[test]
    fn resolve_vm_topology_wprof_universal_floor_dominates_tiny_arena() {
        // With the minimal default arena the wprof floor is far below
        // the universal 256 MiB floor, so the latter dominates even
        // when the entry declares exactly WPROF_MIN_MEMORY_MIB. The
        // strict-less-than direction of the floor condition itself is
        // pinned directly in `stamped_wprof_memory_floor_is_feature_independent`.
        let entry = KtstrTestEntry {
            name: "wprof_exact",
            memory_mib: crate::vmm::wprof::WPROF_MIN_MEMORY_MIB,
            wprof: true,
            ..KtstrTestEntry::DEFAULT
        };
        let (_topo, mem) = resolve_vm_topology(&entry, None);
        assert_eq!(
            mem, 256,
            "universal 256 MiB floor dominates the tiny wprof arena; got {mem}"
        );
    }

    #[cfg(feature = "wprof")]
    #[test]
    fn resolve_vm_topology_wprof_zero_entry_memory_mib_uses_universal_floor() {
        // Edge case: entry.memory_mib=0 with wprof=true. The raw
        // derivation `max(cpus*64, 256, 0)` resolves to 256 on the
        // default 1-CPU topology; the minimal wprof arena is below
        // that, so the universal 256 MiB floor is what the VM gets.
        let entry = KtstrTestEntry {
            name: "wprof_zero_mib",
            memory_mib: 0,
            wprof: true,
            ..KtstrTestEntry::DEFAULT
        };
        let (_topo, mem) = resolve_vm_topology(&entry, None);
        assert_eq!(
            mem, 256,
            "entry.memory_mib=0 with wprof=true resolves to the \
             universal 256 MiB floor; got {mem}"
        );
    }

    #[cfg(feature = "wprof")]
    #[test]
    fn derive_test_memory_mib_helper_applies_wprof_floor() {
        // Direct test of the derivation helper used by BOTH
        // resolve_vm_topology AND the dispatch.rs sites that
        // construct TopoOverride from CLI / preset topology
        // (run_ktstr_test_with_topo_str, run_gauntlet_test).
        // Pins that the helper applies the wprof floor — a
        // regression that re-inlined the formula at the dispatch
        // sites without the wprof check would silently bypass
        // the floor when `cargo ktstr test --ktstr-topo` runs
        // against a wprof-tagged test.
        let entry = KtstrTestEntry {
            name: "helper",
            memory_mib: 0,
            wprof: true,
            ..KtstrTestEntry::DEFAULT
        };
        let mem = derive_test_memory_mib(2, &entry);
        assert_eq!(
            mem, 256,
            "helper applies the wprof floor (16 MiB) but it is subsumed \
             by the universal 256 MiB floor for a 2-cpu VM; got {mem}"
        );
        assert_eq!(
            derive_test_memory_min_mib(2, entry.memory_mib, entry.wprof),
            mem,
            "pre-exec admission must apply the same wprof floor as VM sizing",
        );

        // wprof=false: derivation returns the raw formula
        // without any floor.
        let entry_no_wprof = KtstrTestEntry {
            wprof: false,
            ..entry
        };
        let mem = derive_test_memory_mib(2, &entry_no_wprof);
        assert_eq!(
            mem, 256,
            "helper with wprof=false must NOT apply the floor; \
             expected max(2*64, 256, 0)=256, got {mem}"
        );
    }

    #[cfg(feature = "wprof")]
    #[test]
    fn resolve_vm_topology_override_with_wprof_honors_override_verbatim() {
        // The override-is-verbatim contract: a TopoOverride with
        // memory_mib below WPROF_MIN_MEMORY_MIB is honored as the
        // operator's explicit choice. A warn-level log fires (not
        // verified in this unit test — tracing capture is out of
        // scope here) but the boot memory matches the override.
        let entry = KtstrTestEntry {
            name: "override_wprof",
            wprof: true,
            ..KtstrTestEntry::DEFAULT
        };
        let over = super::super::topo::TopoOverride {
            numa_nodes: 1,
            llcs: 1,
            cores: 1,
            threads: 1,
            memory_mib: 512,
        };
        let (_topo, mem) = resolve_vm_topology(&entry, Some(&over));
        assert_eq!(
            mem, 512,
            "TopoOverride.memory_mib must be honored verbatim even \
             with wprof enabled, got {mem}"
        );
    }

    // -- append_base_sched_args --

    #[test]
    fn append_base_sched_args_empty_when_none_set() {
        let entry = KtstrTestEntry {
            name: "nosched",
            ..KtstrTestEntry::DEFAULT
        };
        let mut args = Vec::new();
        append_base_sched_args(&entry, &mut args);
        assert!(args.is_empty(), "no sched args expected: {args:?}");
    }

    /// `cgroup_parent` does NOT auto-inject `--cell-parent-cgroup`
    /// into the scheduler argv — the two concerns are decoupled.
    /// The scheduler-def `sched_args` and the per-test
    /// `extra_sched_args` flow through unchanged; the `cgroup_parent`
    /// setting controls the framework's cgroup root but never
    /// modifies the scheduler's CLI invocation.
    #[test]
    fn append_base_sched_args_does_not_auto_inject_cell_parent_cgroup() {
        static SCHED: Scheduler = Scheduler::named("s")
            .cgroup_parent("/sys/fs/cgroup/ktstr")
            .sched_args(&["-v", "--flag"]);
        let entry = KtstrTestEntry {
            name: "sched",
            scheduler: &SCHED,
            extra_sched_args: &["--extra"],
            ..KtstrTestEntry::DEFAULT
        };
        let mut args = Vec::new();
        append_base_sched_args(&entry, &mut args);
        assert_eq!(
            args,
            vec![
                "-v".to_string(),
                "--flag".to_string(),
                "--extra".to_string(),
            ],
            "cgroup_parent must not auto-inject --cell-parent-cgroup; \
             only sched_args + extra_sched_args reach the scheduler"
        );
    }

    /// User-passed `--cell-parent-cgroup /user` via `extra_sched_args`
    /// suppresses the auto-inject so clap inside the scheduler binary
    /// doesn't reject the duplicate.
    #[test]
    fn append_base_sched_args_dedupes_extra_split_form() {
        static SCHED: Scheduler = Scheduler::named("s").cgroup_parent("/sys/fs/cgroup/ktstr");
        let entry = KtstrTestEntry {
            name: "sched",
            scheduler: &SCHED,
            extra_sched_args: &["--cell-parent-cgroup", "/user"],
            ..KtstrTestEntry::DEFAULT
        };
        let mut args = Vec::new();
        append_base_sched_args(&entry, &mut args);
        assert_eq!(
            args,
            vec!["--cell-parent-cgroup".to_string(), "/user".to_string()],
            "auto-inject must be skipped when extra_sched_args carries \
             --cell-parent-cgroup in two-token form"
        );
    }

    /// Combined form (`--cell-parent-cgroup=/user`) must also suppress
    /// the auto-inject.
    #[test]
    fn append_base_sched_args_dedupes_extra_combined_form() {
        static SCHED: Scheduler = Scheduler::named("s").cgroup_parent("/sys/fs/cgroup/ktstr");
        let entry = KtstrTestEntry {
            name: "sched",
            scheduler: &SCHED,
            extra_sched_args: &["--cell-parent-cgroup=/user"],
            ..KtstrTestEntry::DEFAULT
        };
        let mut args = Vec::new();
        append_base_sched_args(&entry, &mut args);
        assert_eq!(
            args,
            vec!["--cell-parent-cgroup=/user".to_string()],
            "auto-inject must be skipped when extra_sched_args carries \
             --cell-parent-cgroup in combined `=` form"
        );
    }

    /// Scheduler-def `sched_args` carrying `--cell-parent-cgroup`
    /// also suppresses the auto-inject.
    #[test]
    fn append_base_sched_args_dedupes_scheduler_sched_args() {
        static SCHED: Scheduler = Scheduler::named("s")
            .cgroup_parent("/sys/fs/cgroup/ktstr")
            .sched_args(&["--cell-parent-cgroup", "/user"]);
        let entry = KtstrTestEntry {
            name: "sched",
            scheduler: &SCHED,
            ..KtstrTestEntry::DEFAULT
        };
        let mut args = Vec::new();
        append_base_sched_args(&entry, &mut args);
        assert_eq!(
            args,
            vec!["--cell-parent-cgroup".to_string(), "/user".to_string()],
            "auto-inject must be skipped when scheduler.sched_args carries \
             --cell-parent-cgroup"
        );
    }

    /// Scheduler-def `sched_args` carrying the combined `=` form also
    /// suppresses the auto-inject — completes the {source × form}
    /// 2×2 matrix.
    #[test]
    fn append_base_sched_args_dedupes_scheduler_sched_args_combined_form() {
        static SCHED: Scheduler = Scheduler::named("s")
            .cgroup_parent("/sys/fs/cgroup/ktstr")
            .sched_args(&["--cell-parent-cgroup=/user"]);
        let entry = KtstrTestEntry {
            name: "sched",
            scheduler: &SCHED,
            ..KtstrTestEntry::DEFAULT
        };
        let mut args = Vec::new();
        append_base_sched_args(&entry, &mut args);
        assert_eq!(
            args,
            vec!["--cell-parent-cgroup=/user".to_string()],
            "auto-inject must be skipped when scheduler.sched_args carries \
             --cell-parent-cgroup in combined `=` form"
        );
    }

    /// When BOTH scheduler.sched_args AND extra_sched_args carry
    /// `--cell-parent-cgroup`, the framework's auto-inject is
    /// suppressed (`.any()` short-circuits on first match) but the
    /// user's duplicates flow through unchanged. The framework does
    /// not dedupe user-supplied duplicates — clap inside the
    /// scheduler binary will reject them with "cannot be used
    /// multiple times", as it should. Pin: the framework correctly
    /// avoids ADDING a third copy.
    #[test]
    fn append_base_sched_args_does_not_dedupe_user_dupes() {
        static SCHED: Scheduler = Scheduler::named("s")
            .cgroup_parent("/sys/fs/cgroup/ktstr")
            .sched_args(&["--cell-parent-cgroup", "/sched"]);
        let entry = KtstrTestEntry {
            name: "sched",
            scheduler: &SCHED,
            extra_sched_args: &["--cell-parent-cgroup", "/extra"],
            ..KtstrTestEntry::DEFAULT
        };
        let mut args = Vec::new();
        append_base_sched_args(&entry, &mut args);
        assert_eq!(
            args,
            vec![
                "--cell-parent-cgroup".to_string(),
                "/sched".to_string(),
                "--cell-parent-cgroup".to_string(),
                "/extra".to_string(),
            ],
            "framework auto-inject is suppressed; both user-supplied \
             entries flow through unchanged (user owns the dup)"
        );
    }

    /// Empty combined value (`--cell-parent-cgroup=`) is rejected at
    /// the framework gate with an actionable panic that names the
    /// offending test and points the operator at the right fix.
    /// Empty values would resolve to `/sys/fs/cgroup` (the host
    /// cgroup root) downstream — guaranteed to corrupt unrelated
    /// cgroup state — so the framework rejects rather than letting
    /// clap surface a generic "value required" error after the
    /// cgroup hierarchy has already been built.
    #[test]
    #[should_panic(expected = "that does not start with `/`")]
    fn append_base_sched_args_panics_on_empty_combined_value_via_extra() {
        static SCHED: Scheduler = Scheduler::named("s").cgroup_parent("/sys/fs/cgroup/ktstr");
        let entry = KtstrTestEntry {
            name: "sched",
            scheduler: &SCHED,
            extra_sched_args: &["--cell-parent-cgroup="],
            ..KtstrTestEntry::DEFAULT
        };
        let mut args = Vec::new();
        append_base_sched_args(&entry, &mut args);
    }

    /// Two-token form with an empty value as the second token
    /// (`["--cell-parent-cgroup", ""]`) is rejected by the same gate.
    /// Covers the second route into `parse_cell_parent_cgroup` so a
    /// future refactor that switches the empty-detection logic on
    /// only one form gets caught.
    #[test]
    #[should_panic(expected = "that does not start with `/`")]
    fn append_base_sched_args_panics_on_empty_two_token_value_via_extra() {
        static SCHED: Scheduler = Scheduler::named("s").cgroup_parent("/sys/fs/cgroup/ktstr");
        let entry = KtstrTestEntry {
            name: "sched_two_token",
            scheduler: &SCHED,
            extra_sched_args: &["--cell-parent-cgroup", ""],
            ..KtstrTestEntry::DEFAULT
        };
        let mut args = Vec::new();
        append_base_sched_args(&entry, &mut args);
    }

    /// Bad value via the scheduler-def's own `sched_args` rather than
    /// the test's `extra_sched_args` — the chain at the parser site
    /// covers both sources, so the gate fires regardless of origin.
    /// Pins both the combined form and the scheduler origin.
    #[test]
    #[should_panic(expected = "that does not start with `/`")]
    fn append_base_sched_args_panics_on_empty_combined_value_via_scheduler_sched_args() {
        static SCHED: Scheduler = Scheduler::named("s")
            .cgroup_parent("/sys/fs/cgroup/ktstr")
            .sched_args(&["--cell-parent-cgroup="]);
        let entry = KtstrTestEntry {
            name: "sched_in_def",
            scheduler: &SCHED,
            ..KtstrTestEntry::DEFAULT
        };
        let mut args = Vec::new();
        append_base_sched_args(&entry, &mut args);
    }

    /// Two-token form via the scheduler-def origin — completes the
    /// 2-source × 2-form matrix together with the three siblings.
    #[test]
    #[should_panic(expected = "that does not start with `/`")]
    fn append_base_sched_args_panics_on_empty_two_token_value_via_scheduler_sched_args() {
        static SCHED: Scheduler = Scheduler::named("s")
            .cgroup_parent("/sys/fs/cgroup/ktstr")
            .sched_args(&["--cell-parent-cgroup", ""]);
        let entry = KtstrTestEntry {
            name: "sched_in_def_two_token",
            scheduler: &SCHED,
            ..KtstrTestEntry::DEFAULT
        };
        let mut args = Vec::new();
        append_base_sched_args(&entry, &mut args);
    }

    /// Empty-value gate fires even when the scheduler-def has no
    /// `cgroup_parent` default. Without the universal gate the empty
    /// value would slip through and corrupt unrelated host cgroup
    /// state at the downstream `resolve_cgroup_root` interpolation.
    #[test]
    #[should_panic(expected = "that does not start with `/`")]
    fn append_base_sched_args_panics_on_empty_combined_value_no_scheduler_cgroup_parent() {
        static SCHED: Scheduler = Scheduler::named("s");
        let entry = KtstrTestEntry {
            name: "no_default_cgroup",
            scheduler: &SCHED,
            extra_sched_args: &["--cell-parent-cgroup="],
            ..KtstrTestEntry::DEFAULT
        };
        let mut args = Vec::new();
        append_base_sched_args(&entry, &mut args);
    }

    /// Two-token form, no scheduler default — completes the
    /// no-default matrix together with the combined-form sibling.
    #[test]
    #[should_panic(expected = "that does not start with `/`")]
    fn append_base_sched_args_panics_on_empty_two_token_value_no_scheduler_cgroup_parent() {
        static SCHED: Scheduler = Scheduler::named("s");
        let entry = KtstrTestEntry {
            name: "no_default_cgroup_two_token",
            scheduler: &SCHED,
            extra_sched_args: &["--cell-parent-cgroup", ""],
            ..KtstrTestEntry::DEFAULT
        };
        let mut args = Vec::new();
        append_base_sched_args(&entry, &mut args);
    }

    /// Relative path (no leading `/`) is rejected by the same gate.
    /// Pins the broader contract (the message explicitly promises
    /// "absolute path under `/`"); empty is just one case of
    /// non-absolute.
    #[test]
    #[should_panic(expected = "that does not start with `/`")]
    fn append_base_sched_args_panics_on_relative_path_value() {
        static SCHED: Scheduler = Scheduler::named("s").cgroup_parent("/sys/fs/cgroup/ktstr");
        let entry = KtstrTestEntry {
            name: "relative_path",
            scheduler: &SCHED,
            extra_sched_args: &["--cell-parent-cgroup=my_test"],
            ..KtstrTestEntry::DEFAULT
        };
        let mut args = Vec::new();
        append_base_sched_args(&entry, &mut args);
    }

    /// Two-token form of the relative-path case. Closes the matrix
    /// gap: combined-form was pinned by the sibling above but a
    /// future refactor that split path validation between the
    /// combined and two-token branches could regress one form
    /// without test catching.
    #[test]
    #[should_panic(expected = "that does not start with `/`")]
    fn append_base_sched_args_panics_on_relative_path_value_two_token() {
        static SCHED: Scheduler = Scheduler::named("s").cgroup_parent("/sys/fs/cgroup/ktstr");
        let entry = KtstrTestEntry {
            name: "relative_path_two_token",
            scheduler: &SCHED,
            extra_sched_args: &["--cell-parent-cgroup", "my_test"],
            ..KtstrTestEntry::DEFAULT
        };
        let mut args = Vec::new();
        append_base_sched_args(&entry, &mut args);
    }

    /// `/.` is absolute and has more than one character, so a naive
    /// `starts_with('/') && len > 1` check passes — but the kernel
    /// canonicalizes `/sys/fs/cgroup/.` back to `/sys/fs/cgroup`
    /// (host cgroup root), corrupting unrelated cgroup state.
    /// `Path::components` strips the trailing `.`, yielding `[RootDir]`
    /// — the validator rejects via the "has no Normal component"
    /// check, not the CurDir arm (see `cell_parent_path_is_valid`).
    #[test]
    #[should_panic(expected = "contains `.`/`..` segments")]
    fn append_base_sched_args_panics_on_dot_normalizing_to_root() {
        static SCHED: Scheduler = Scheduler::named("s").cgroup_parent("/sys/fs/cgroup/ktstr");
        let entry = KtstrTestEntry {
            name: "dot_normalize",
            scheduler: &SCHED,
            extra_sched_args: &["--cell-parent-cgroup=/."],
            ..KtstrTestEntry::DEFAULT
        };
        let mut args = Vec::new();
        append_base_sched_args(&entry, &mut args);
    }

    /// `/foo/..` canonicalizes back to `/` → `/sys/fs/cgroup`. Same
    /// host-root corruption risk as the empty/bare-slash cases. The
    /// component-based gate rejects any `..` (ParentDir) segment.
    #[test]
    #[should_panic(expected = "contains `.`/`..` segments")]
    fn append_base_sched_args_panics_on_parent_dir_normalizing_to_root() {
        static SCHED: Scheduler = Scheduler::named("s").cgroup_parent("/sys/fs/cgroup/ktstr");
        let entry = KtstrTestEntry {
            name: "parent_dir_normalize",
            scheduler: &SCHED,
            extra_sched_args: &["--cell-parent-cgroup=/foo/.."],
            ..KtstrTestEntry::DEFAULT
        };
        let mut args = Vec::new();
        append_base_sched_args(&entry, &mut args);
    }

    /// Mixed `/./bar/..` — both kinds of normalizing segment in one
    /// path. `Path::components` strips the leading `/.`, yielding
    /// `[RootDir, Normal("bar"), ParentDir]`; the validator reaches
    /// the `ParentDir` and rejects via that arm. The `/.` never
    /// surfaces as a CurDir component.
    #[test]
    #[should_panic(expected = "contains `.`/`..` segments")]
    fn append_base_sched_args_panics_on_mixed_normalize_segments() {
        static SCHED: Scheduler = Scheduler::named("s").cgroup_parent("/sys/fs/cgroup/ktstr");
        let entry = KtstrTestEntry {
            name: "mixed_normalize",
            scheduler: &SCHED,
            extra_sched_args: &["--cell-parent-cgroup=/./bar/.."],
            ..KtstrTestEntry::DEFAULT
        };
        let mut args = Vec::new();
        append_base_sched_args(&entry, &mut args);
    }

    /// `/foo/./bar` is ACCEPTED — `Path::components` normalizes away
    /// every `CurDir` segment (see `cell_parent_path_is_valid` for
    /// the full per-position behavior); the canonical form
    /// `/foo/bar` is a real non-root path. Pin the accept path so a
    /// future refactor to a stricter `.contains("/./")` text check
    /// is caught. Also assert the user value flows through verbatim
    /// — a regression that canonicalized the path before forwarding
    /// would silently rewrite `/foo/./bar` to `/foo/bar`.
    #[test]
    fn append_base_sched_args_accepts_embedded_dot_segment() {
        static SCHED: Scheduler = Scheduler::named("s").cgroup_parent("/sys/fs/cgroup/ktstr");
        let entry = KtstrTestEntry {
            name: "embedded_dot_ok",
            scheduler: &SCHED,
            extra_sched_args: &["--cell-parent-cgroup=/foo/./bar"],
            ..KtstrTestEntry::DEFAULT
        };
        let mut args = Vec::new();
        append_base_sched_args(&entry, &mut args);
        assert!(
            args.iter().any(|a| a == "--cell-parent-cgroup=/foo/./bar"),
            "user value must pass through verbatim (no canonicalization); args: {args:?}",
        );
    }

    /// Bare `/..` is the most damaging path-normalize edge:
    /// downstream interpolation `/sys/fs/cgroup/..` canonicalizes to
    /// `/sys/fs` — escapes the cgroup hierarchy entirely. The
    /// component walk hits `ParentDir` immediately after `RootDir`
    /// (no Normal segment between them) and rejects via the
    /// ParentDir arm.
    #[test]
    #[should_panic(expected = "contains `.`/`..` segments")]
    fn append_base_sched_args_panics_on_bare_parent_dir() {
        static SCHED: Scheduler = Scheduler::named("s").cgroup_parent("/sys/fs/cgroup/ktstr");
        let entry = KtstrTestEntry {
            name: "bare_parent_dir",
            scheduler: &SCHED,
            extra_sched_args: &["--cell-parent-cgroup=/.."],
            ..KtstrTestEntry::DEFAULT
        };
        let mut args = Vec::new();
        append_base_sched_args(&entry, &mut args);
    }

    /// Mid-path `/foo/../bar` — ParentDir sits BETWEEN Normal
    /// segments. Different shape from `/foo/..` (trailing
    /// ParentDir): a regression that bailed only on
    /// `path.ends_with("/..")` would slip this past. Downstream
    /// interpolation `/sys/fs/cgroup/foo/../bar` canonicalizes to
    /// `/sys/fs/cgroup/bar` — an unintended sibling directory the
    /// test author didn't ask for. Component walk catches ParentDir
    /// in any position.
    #[test]
    #[should_panic(expected = "contains `.`/`..` segments")]
    fn append_base_sched_args_panics_on_mid_path_parent_dir() {
        static SCHED: Scheduler = Scheduler::named("s").cgroup_parent("/sys/fs/cgroup/ktstr");
        let entry = KtstrTestEntry {
            name: "mid_path_parent_dir",
            scheduler: &SCHED,
            extra_sched_args: &["--cell-parent-cgroup=/foo/../bar"],
            ..KtstrTestEntry::DEFAULT
        };
        let mut args = Vec::new();
        append_base_sched_args(&entry, &mut args);
    }

    /// Bare `/` slips a naive `starts_with('/')` check but resolves
    /// downstream to `/sys/fs/cgroup/` — semantically the host cgroup
    /// root, same corruption risk as the empty case. The gate mirrors
    /// `CgroupPath::new`'s const-eval contract (rejects both
    /// no-leading-slash AND `"/"` alone) so runtime values share the
    /// same validation as compile-time declarations.
    #[test]
    #[should_panic(expected = "is `/` alone")]
    fn append_base_sched_args_panics_on_bare_slash_value() {
        static SCHED: Scheduler = Scheduler::named("s").cgroup_parent("/sys/fs/cgroup/ktstr");
        let entry = KtstrTestEntry {
            name: "bare_slash",
            scheduler: &SCHED,
            extra_sched_args: &["--cell-parent-cgroup=/"],
            ..KtstrTestEntry::DEFAULT
        };
        let mut args = Vec::new();
        append_base_sched_args(&entry, &mut args);
    }

    /// Combined-form empty value via scheduler-def `sched_args`
    /// when the scheduler also has NO `cgroup_parent` default. Closes
    /// the matrix intersection: a future refactor that gates the
    /// scheduler-def-source check on `cgroup_parent.is_some()` would
    /// pass the other 6 empty tests but regress this cell.
    #[test]
    #[should_panic(expected = "that does not start with `/`")]
    fn append_base_sched_args_panics_on_empty_combined_value_in_scheduler_sched_args_no_default() {
        static SCHED: Scheduler = Scheduler::named("s").sched_args(&["--cell-parent-cgroup="]);
        let entry = KtstrTestEntry {
            name: "scheduler_def_origin_no_default",
            scheduler: &SCHED,
            ..KtstrTestEntry::DEFAULT
        };
        let mut args = Vec::new();
        append_base_sched_args(&entry, &mut args);
    }

    /// Two-token-form sibling of the above — completes the
    /// 2-form coverage for the scheduler-def-origin × no-default
    /// intersection.
    #[test]
    #[should_panic(expected = "that does not start with `/`")]
    fn append_base_sched_args_panics_on_empty_two_token_value_in_scheduler_sched_args_no_default() {
        static SCHED: Scheduler = Scheduler::named("s").sched_args(&["--cell-parent-cgroup", ""]);
        let entry = KtstrTestEntry {
            name: "scheduler_def_origin_two_token_no_default",
            scheduler: &SCHED,
            ..KtstrTestEntry::DEFAULT
        };
        let mut args = Vec::new();
        append_base_sched_args(&entry, &mut args);
    }

    /// Bare `--cell-parent-cgroup` flag with no following token
    /// (two-token form, trailing in argv) is rejected at the
    /// framework gate via the `CellParentCgroupArg::MissingValue`
    /// arm. Previously this shape parsed as "absent", triggered the
    /// auto-inject, and produced two copies of the flag in the final
    /// argv that clap then rejected with a confused "cannot be used
    /// multiple times" diagnostic. The gate intercepts here so the
    /// operator gets a "missing value" message anchored to their
    /// declaration.
    #[test]
    #[should_panic(expected = "supplies a bare `--cell-parent-cgroup`")]
    fn append_base_sched_args_panics_on_missing_value_via_extra() {
        static SCHED: Scheduler = Scheduler::named("s").cgroup_parent("/sys/fs/cgroup/ktstr");
        let entry = KtstrTestEntry {
            name: "missing_value_extra",
            scheduler: &SCHED,
            extra_sched_args: &["--cell-parent-cgroup"],
            ..KtstrTestEntry::DEFAULT
        };
        let mut args = Vec::new();
        append_base_sched_args(&entry, &mut args);
    }

    /// Bare flag preceded by an unrelated trailing token still trips
    /// the MissingValue arm — the parser walks the chain in order,
    /// hits the bare flag, and `iter.next()` returns None at end of
    /// stream regardless of which unrelated tokens came before it.
    #[test]
    #[should_panic(expected = "supplies a bare `--cell-parent-cgroup`")]
    fn append_base_sched_args_panics_on_missing_value_after_other_flag() {
        static SCHED: Scheduler = Scheduler::named("s").cgroup_parent("/sys/fs/cgroup/ktstr");
        let entry = KtstrTestEntry {
            name: "missing_value_after_other",
            scheduler: &SCHED,
            extra_sched_args: &["--other-flag", "--cell-parent-cgroup"],
            ..KtstrTestEntry::DEFAULT
        };
        let mut args = Vec::new();
        append_base_sched_args(&entry, &mut args);
    }

    /// Bare flag in the scheduler-def's `sched_args` also trips
    /// MissingValue — the parser chains both sources and the
    /// universal gate handles them identically.
    #[test]
    #[should_panic(expected = "supplies a bare `--cell-parent-cgroup`")]
    fn append_base_sched_args_panics_on_missing_value_in_scheduler_sched_args() {
        static SCHED: Scheduler = Scheduler::named("s")
            .cgroup_parent("/sys/fs/cgroup/ktstr")
            .sched_args(&["--cell-parent-cgroup"]);
        let entry = KtstrTestEntry {
            name: "missing_value_scheduler_def",
            scheduler: &SCHED,
            ..KtstrTestEntry::DEFAULT
        };
        let mut args = Vec::new();
        append_base_sched_args(&entry, &mut args);
    }

    /// Bare flag with no scheduler default `cgroup_parent`. The
    /// universal gate must still fire — the panic message in this
    /// case omits the "let the framework auto-inject" suggestion
    /// (no default to inject) and adds a hint that an absolute path
    /// is required for cell-aware schedulers without a declared
    /// default.
    #[test]
    #[should_panic(expected = "supplies a bare `--cell-parent-cgroup`")]
    fn append_base_sched_args_panics_on_missing_value_no_scheduler_cgroup_parent() {
        static SCHED: Scheduler = Scheduler::named("s");
        let entry = KtstrTestEntry {
            name: "missing_value_no_default",
            scheduler: &SCHED,
            extra_sched_args: &["--cell-parent-cgroup"],
            ..KtstrTestEntry::DEFAULT
        };
        let mut args = Vec::new();
        append_base_sched_args(&entry, &mut args);
    }

    /// Bare flag via scheduler-def `sched_args` with no default
    /// `cgroup_parent`. Closes the matrix intersection: a future
    /// refactor that gated the MissingValue check on
    /// `cgroup_parent.is_some()` (mirroring an earlier regression
    /// fixed for Value-invalid) would pass the other 4 MissingValue
    /// tests but regress this cell.
    #[test]
    #[should_panic(expected = "supplies a bare `--cell-parent-cgroup`")]
    fn append_base_sched_args_panics_on_missing_value_in_scheduler_sched_args_no_default() {
        static SCHED: Scheduler = Scheduler::named("s").sched_args(&["--cell-parent-cgroup"]);
        let entry = KtstrTestEntry {
            name: "missing_value_scheduler_def_no_default",
            scheduler: &SCHED,
            ..KtstrTestEntry::DEFAULT
        };
        let mut args = Vec::new();
        append_base_sched_args(&entry, &mut args);
    }

    /// Bare flag after another flag, with no scheduler default.
    /// Completes the after-other-flag × default matrix together with
    /// the sibling test that has a default.
    #[test]
    #[should_panic(expected = "supplies a bare `--cell-parent-cgroup`")]
    fn append_base_sched_args_panics_on_missing_value_after_other_flag_no_default() {
        static SCHED: Scheduler = Scheduler::named("s");
        let entry = KtstrTestEntry {
            name: "missing_value_after_other_no_default",
            scheduler: &SCHED,
            extra_sched_args: &["--other-flag", "--cell-parent-cgroup"],
            ..KtstrTestEntry::DEFAULT
        };
        let mut args = Vec::new();
        append_base_sched_args(&entry, &mut args);
    }

    // -- build_vm_builder_base --

    /// Kernel-path surfaces in the builder's "kernel not found" error.
    /// Proves the `kernel()` setter is wired through the helper.
    #[test]
    fn build_vm_builder_base_propagates_kernel_path() {
        // build()'s no-perf path reads KTSTR_BYPASS_LLC_LOCKS + KTSTR_CPU_CAP
        // before the validation checks. Under the shared env lock, pin
        // bypass=1 + cpu_cap unset so build() short-circuits the slot/LLC
        // acquire path (no acquire_llc_plan contention; cpu_cap=None avoids
        // the bypass+cpu_cap bail), leaving the asserted error the only outcome.
        let _l = lock_env();
        let _g = EnvVarGuard::set(crate::KTSTR_BYPASS_LLC_LOCKS_ENV, "1");
        let _c = EnvVarGuard::remove(crate::KTSTR_CPU_CAP_ENV);
        let entry = KtstrTestEntry {
            name: "vmb_kernel_path",
            ..KtstrTestEntry::DEFAULT
        };
        let exe = crate::resolve_current_exe().unwrap();
        let missing_kernel =
            PathBuf::from("/nonexistent/build_vm_builder_base_test_kernel.bzImage");
        let result = build_vm_builder_base(
            &entry,
            &missing_kernel,
            &exe,
            None,
            &[],
            crate::vmm::topology::Topology::new(1, 1, 1, 1),
            256,
            "",
            &["run".to_string()],
            true,
        )
        .build();
        // `KtstrVm` does not implement Debug, so `.unwrap_err()` is not
        // available — collapse Ok into a panic to extract the error by hand.
        let err = match result {
            Ok(_) => panic!("builder.build() unexpectedly succeeded for missing kernel"),
            Err(e) => e,
        };
        let msg = format!("{err}");
        assert!(
            msg.contains("kernel not found"),
            "expected kernel not found error, got: {msg}",
        );
        assert!(
            msg.contains("build_vm_builder_base_test_kernel"),
            "expected the fake kernel path to appear in the error, got: {msg}",
        );
    }

    /// A zero-`llcs` topology is forwarded to the builder and surfaces
    /// as a validation error. Proves `topology()` is wired through.
    #[test]
    fn build_vm_builder_base_propagates_topology_validation() {
        // See build_vm_builder_base_propagates_kernel_path: pin bypass=1 +
        // cpu_cap unset under the shared env lock so build() short-circuits
        // the no-perf slot/LLC path and the asserted error is deterministic.
        let _l = lock_env();
        let _g = EnvVarGuard::set(crate::KTSTR_BYPASS_LLC_LOCKS_ENV, "1");
        let _c = EnvVarGuard::remove(crate::KTSTR_CPU_CAP_ENV);
        let entry = KtstrTestEntry {
            name: "vmb_topology",
            ..KtstrTestEntry::DEFAULT
        };
        let exe = crate::resolve_current_exe().unwrap();
        let bad_topology = crate::vmm::topology::Topology {
            llcs: 0,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
            llc_cores: None,
        };
        let result = build_vm_builder_base(
            &entry,
            &exe,
            &exe,
            None,
            &[],
            bad_topology,
            256,
            "",
            &["run".to_string()],
            true,
        )
        .build();
        let err = match result {
            Ok(_) => panic!("builder.build() unexpectedly succeeded for zero-llcs topology"),
            Err(e) => e,
        };
        let msg = format!("{err}");
        assert!(
            msg.contains("llcs must be > 0"),
            "expected topology validation error, got: {msg}",
        );
    }

    /// An optional scheduler binary is attached when `Some(path)`
    /// is supplied, surfacing as a "scheduler binary not found"
    /// error when the path is missing.
    #[test]
    fn build_vm_builder_base_propagates_scheduler_binary() {
        // See build_vm_builder_base_propagates_kernel_path: pin bypass=1 +
        // cpu_cap unset under the shared env lock so build() short-circuits
        // the no-perf slot/LLC path and the asserted error is deterministic.
        let _l = lock_env();
        let _g = EnvVarGuard::set(crate::KTSTR_BYPASS_LLC_LOCKS_ENV, "1");
        let _c = EnvVarGuard::remove(crate::KTSTR_CPU_CAP_ENV);
        let entry = KtstrTestEntry {
            name: "vmb_scheduler",
            ..KtstrTestEntry::DEFAULT
        };
        let exe = crate::resolve_current_exe().unwrap();
        let missing_scheduler = PathBuf::from("/nonexistent/build_vm_builder_base_test_scheduler");
        let result = build_vm_builder_base(
            &entry,
            &exe,
            &exe,
            Some(&missing_scheduler),
            &[],
            crate::vmm::topology::Topology::new(1, 1, 1, 1),
            256,
            "",
            &["run".to_string()],
            true,
        )
        .build();
        let err = match result {
            Ok(_) => panic!("builder.build() unexpectedly succeeded for missing scheduler"),
            Err(e) => e,
        };
        let msg = format!("{err}");
        assert!(
            msg.contains("scheduler binary not found"),
            "expected scheduler binary error, got: {msg}",
        );
        assert!(
            msg.contains("build_vm_builder_base_test_scheduler"),
            "expected the fake scheduler path to appear, got: {msg}",
        );
    }

    // -- vm_timeout_from_entry tests --

    // The Tier-3 dead-man deadline is `base + vm_boot_headroom(vcpus) *
    // DEADMAN_HEADROOM_MULT`; the headroom no longer scales by the host
    // overcommit ratio, so these are host-cpuset-independent (no pin).

    #[test]
    fn vm_timeout_from_entry_uses_watchdog_when_largest() {
        // DEFAULT topology = 2 vCPUs → vm_boot_headroom(2) = 20.3 s ×3 =
        // 60.9 s. base = max(60s, 30s, 1s) = 60s → 120.9 s.
        let entry = KtstrTestEntry {
            name: "wdog",
            watchdog_timeout: Duration::from_secs(60),
            duration: Duration::from_secs(30),
            ..KtstrTestEntry::DEFAULT
        };
        assert_eq!(
            vm_timeout_from_entry(&entry, entry.topology.total_cpus()),
            Duration::from_millis(120_900)
        );
    }

    #[test]
    fn vm_timeout_from_entry_uses_duration_when_largest() {
        let entry = KtstrTestEntry {
            name: "dur",
            watchdog_timeout: Duration::from_secs(5),
            duration: Duration::from_secs(120),
            ..KtstrTestEntry::DEFAULT
        };
        // base = max(5s, 120s, 1s) = 120s; headroom(2)×3 = 60.9 s.
        assert_eq!(
            vm_timeout_from_entry(&entry, entry.topology.total_cpus()),
            Duration::from_millis(180_900)
        );
    }

    #[test]
    fn vm_timeout_from_entry_floor_when_both_small() {
        // base floors at 1 s; headroom(2)×3 = 60.9 s → 61.9 s.
        let entry = KtstrTestEntry {
            name: "tiny",
            watchdog_timeout: Duration::from_millis(10),
            duration: Duration::from_millis(50),
            ..KtstrTestEntry::DEFAULT
        };
        assert_eq!(
            vm_timeout_from_entry(&entry, entry.topology.total_cpus()),
            Duration::from_millis(61_900)
        );
    }

    #[test]
    fn vm_timeout_from_default_entry() {
        // DEFAULT watchdog = 5 s, duration = 12 s → base = 12 s.
        // headroom(2)×3 = 60.9 s → 72.9 s total.
        let entry = KtstrTestEntry {
            name: "default",
            ..KtstrTestEntry::DEFAULT
        };
        assert_eq!(
            vm_timeout_from_entry(&entry, entry.topology.total_cpus()),
            Duration::from_millis(72_900)
        );
    }

    #[test]
    fn vm_timeout_from_entry_scales_headroom_with_topology() {
        // A reported case: numa=1, llcs=7, cores=9, threads=2 → 126 vCPUs.
        // vm_boot_headroom(126) = 38.9 s ×3 = 116.7 s. base = max(5 s
        // watchdog, 12 s duration, 1 s) = 12 s → total = 128.7 s.
        // Pins the `entry.topology.total_cpus()` → `vm_boot_headroom`
        // wiring; the flat ×3 headroom is host-cpuset-independent.
        let entry = KtstrTestEntry {
            name: "large_topo",
            topology: crate::vmm::topology::Topology {
                llcs: 7,
                cores_per_llc: 9,
                threads_per_core: 2,
                numa_nodes: 1,
                nodes: None,
                distances: None,
                llc_cores: None,
            },
            ..KtstrTestEntry::DEFAULT
        };
        assert_eq!(
            vm_timeout_from_entry(&entry, entry.topology.total_cpus()),
            Duration::from_millis(128_700)
        );
    }

    #[test]
    fn vm_timeout_from_entry_scales_on_booted_not_declared_vcpus() {
        // Under a TopoOverride the VM boots a different vCPU count than
        // entry.topology declares; the boot-headroom deadline must scale
        // to the BOOTED count passed in, not entry.topology. A default
        // 2-vCPU entry "booted" at 126 vCPUs must get the 126-vCPU
        // headroom (128.7 s, matching the declared-126 case above), not
        // the declared 2-vCPU headroom (72.9 s). The declared-vs-booted
        // gap is otherwise untested — every other vm_timeout test passes
        // entry.topology.total_cpus() (declared == booted).
        let entry = KtstrTestEntry {
            name: "booted_override",
            ..KtstrTestEntry::DEFAULT
        };
        assert_eq!(
            vm_timeout_from_entry(&entry, 126),
            Duration::from_millis(128_700),
        );
        assert_eq!(
            vm_timeout_from_entry(&entry, entry.topology.total_cpus()),
            Duration::from_millis(72_900),
        );
        assert_ne!(
            vm_timeout_from_entry(&entry, 126),
            vm_timeout_from_entry(&entry, entry.topology.total_cpus()),
            "the deadline must key on the booted count, not the declared entry.topology",
        );
    }

    // -- overcommit_ratio / oversub-scaled vm_timeout --

    #[test]
    fn overcommit_ratio_floors_at_one_for_fitting_host() {
        // vCPUs <= allowed → not oversubscribed → 1.0 (never < 1).
        assert_eq!(overcommit_ratio(8, 192, None), 1.0);
        assert_eq!(overcommit_ratio(192, 192, None), 1.0);
    }

    #[test]
    fn overcommit_ratio_auto_collapse_uses_allowed_cpuset() {
        // No explicit cpu_budget: the vCPU threads collapse onto the
        // whole allowed cpuset. 256 vCPUs on 192 allowed = the CI
        // wide-SMP case (~1.33x).
        let r = overcommit_ratio(256, 192, None);
        assert!((r - 256.0 / 192.0).abs() < 1e-9, "got {r}");
    }

    #[test]
    fn overcommit_ratio_explicit_budget_collapses_onto_min_budget_allowed() {
        // Explicit cpu_budget caps the host CPUs the vCPU threads land
        // on (the deliberate _overcommit test): 256 / min(64, 192) = 4x.
        assert_eq!(overcommit_ratio(256, 192, Some(64)), 4.0);
        // A budget wider than the allowed set clamps to allowed.
        let r = overcommit_ratio(256, 192, Some(1000));
        assert!((r - 256.0 / 192.0).abs() < 1e-9, "got {r}");
    }

    #[test]
    fn overcommit_ratio_guards_empty_cpuset() {
        // An unenumerable cpuset (allowed_cpus = 0) must not divide by
        // zero — treat it as a 1-CPU host.
        assert_eq!(overcommit_ratio(8, 0, None), 8.0);
    }

    #[test]
    fn overcommit_skip_reason_skips_severe_auto_collapse() {
        // 256 vCPUs auto-collapse onto 8 host CPUs = 32x ≥ 6x → skip.
        // (boot-only path: expect_auto_repro = false.)
        let r = overcommit_skip_reason(256, 8, None, false);
        assert!(
            r.as_deref()
                .is_some_and(|m| m.contains("host topology insufficient")),
            "32x auto-collapse must skip with the typed reason, got {r:?}",
        );
    }

    #[test]
    fn overcommit_skip_reason_runs_ci_wide_smp_ratio() {
        // 256 vCPUs on a 192-CPU CI runner = 1.33x < 6x → RUNS (None),
        // so wide-SMP boot is validated there, never masked.
        assert_eq!(overcommit_skip_reason(256, 192, None, false), None);
    }

    #[test]
    fn overcommit_skip_reason_never_skips_explicit_budget() {
        // An explicit cpu_budget is a deliberate oversubscription opt-in
        // (contention testing): even 256 vCPUs on 8 host CPUs runs.
        assert_eq!(overcommit_skip_reason(256, 8, Some(4), false), None);
    }

    #[test]
    fn overcommit_skip_reason_runs_on_empty_cpuset() {
        // An unenumerable cpuset (allowed = 0) cannot compute a ratio →
        // does not skip; the overcommit warning is the sole signal there.
        assert_eq!(overcommit_skip_reason(256, 0, None, false), None);
    }

    #[test]
    fn overcommit_skip_reason_boundary_is_inclusive_at_cap() {
        // ≥ cap skips, just-below runs. 48 vCPUs on 8 = exactly 6.0x → skip;
        // 47 on 8 = 5.875x < 6.0 → run. (boot-only path.)
        assert!(overcommit_skip_reason(48, 8, None, false).is_some());
        assert_eq!(overcommit_skip_reason(47, 8, None, false), None);
    }

    #[test]
    fn overcommit_skip_reason_expect_auto_repro_uses_stricter_cap() {
        // The expect_auto_repro inversion chain skips at a much lower
        // ratio (EXPECT_AUTO_REPRO_SKIP_RATIO = 2.0x) than a boot-only
        // wide test. Pins the failing CI case: 256 vCPUs on a 96-CPU host
        // = 2.67x.
        //   - 2.67x WITH expect_auto_repro -> SKIP (the two-VM wprof chain
        //     cannot run cleanly under that time-slicing).
        let skip = overcommit_skip_reason(256, 96, None, true);
        assert!(
            skip.as_deref().is_some_and(
                |m| m.contains("host topology insufficient") && m.contains("expect_auto_repro")
            ),
            "2.67x with expect_auto_repro must skip naming the chain, got {skip:?}",
        );
        //   - same 2.67x WITHOUT expect_auto_repro -> RUNS (a single-VM
        //     wide-SMP boot test still validates boot at 2.67x < 6.0x).
        assert_eq!(overcommit_skip_reason(256, 96, None, false), None);
        //   - the 192-CPU design-target runner (256/192 = 1.33x) RUNS the
        //     auto-repro hop even with expect_auto_repro (1.33x < 2.0x), so
        //     the >255 inversion is still validated there.
        assert_eq!(overcommit_skip_reason(256, 192, None, true), None);
        //   - an explicit cpu_budget stays a deliberate opt-in: no skip
        //     even with expect_auto_repro.
        assert_eq!(overcommit_skip_reason(256, 8, Some(4), true), None);
        //   - boundary: exactly 2.0x with expect_auto_repro skips (>= is
        //     inclusive); just below runs.
        assert!(overcommit_skip_reason(16, 8, None, true).is_some());
        assert_eq!(overcommit_skip_reason(15, 8, None, true), None);
    }

    #[test]
    fn vm_timeout_headroom_is_flat_deadman_mult() {
        // Tier-3 dead-man: the headroom term is a FLAT
        // vm_boot_headroom(vcpus) × DEADMAN_HEADROOM_MULT, with NO host
        // overcommit scaling. 256 vCPUs (16 LLCs × 16 cores):
        // vm_boot_headroom(256) = 58.4 s ×3 = 175.2 s; base = max(5 s
        // watchdog, 12 s duration, 1 s) = 12 s → 187.2 s. The old path
        // multiplied this headroom by the host overcommit ratio; it no
        // longer does (Tier-1/2 absorb a loaded host), so the value is
        // independent of the host cpuset.
        let entry = KtstrTestEntry {
            name: "flat",
            topology: crate::vmm::topology::Topology {
                llcs: 16,
                cores_per_llc: 16,
                threads_per_core: 1,
                numa_nodes: 1,
                nodes: None,
                distances: None,
                llc_cores: None,
            },
            ..KtstrTestEntry::DEFAULT
        };
        // 12_000 + 58_400 × 3 = 187_200 ms.
        assert_eq!(
            vm_timeout_from_entry(&entry, entry.topology.total_cpus()),
            Duration::from_millis(187_200)
        );
    }

    #[test]
    fn vm_timeout_headroom_does_not_scale_with_overcommit() {
        // The deadline reads no host cpuset: the headroom above `base` is
        // EXACTLY vm_boot_headroom(vcpus) × DEADMAN_HEADROOM_MULT for any
        // vcpu count, so an oversubscribed host cannot lengthen it. Pins
        // the structural shape (base = 12 s = max(5 s wdog, 12 s dur, 1 s)
        // for a DEFAULT entry).
        let entry = KtstrTestEntry {
            name: "no_oversub",
            ..KtstrTestEntry::DEFAULT
        };
        let base = Duration::from_secs(12);
        for vcpus in [2u32, 64, 128, 256] {
            assert_eq!(
                vm_timeout_from_entry(&entry, vcpus) - base,
                vm_boot_headroom(vcpus) * DEADMAN_HEADROOM_MULT,
                "headroom must be the flat dead-man term at {vcpus} vCPUs",
            );
        }
    }

    #[test]
    fn verifier_vm_timeout_is_flat_deadman() {
        // Verifier Tier-3 dead-man: VERIFIER_BASE_TIMEOUT (120 s) plus a
        // FLAT vm_boot_headroom(vcpus) × DEADMAN_HEADROOM_MULT, no host
        // overcommit scaling. 128 vCPUs: vm_boot_headroom(128) = 10 s
        // kernel init + (10_000 + 128×150 = 29_200 ms) sys_rdy = 39.2 s
        // ×3 = 117.6 s → 237.6 s. The old flat 120 s left ZERO boot
        // headroom for exactly this shape — the mid-attach VM-timeout
        // class from the scx verifier sweep.
        assert_eq!(verifier_vm_timeout(128), Duration::from_millis(237_600));
        // Structural: the headroom above the base is exactly the flat
        // dead-man term for any vcpu count (no cpuset read).
        for vcpus in [2u32, 128, 256] {
            assert_eq!(
                verifier_vm_timeout(vcpus) - VERIFIER_BASE_TIMEOUT,
                vm_boot_headroom(vcpus) * DEADMAN_HEADROOM_MULT,
            );
        }
        // The worst-case verifier deadline (widest realistic 256-vCPU
        // preset) plus the extend-only attach-reset extension must still
        // fit the 240s x 2 = 480s nextest verifier override.
        assert!(
            verifier_vm_timeout(256) + VERIFIER_WORKLOAD_BUDGET < Duration::from_secs(480),
            "worst-case verifier deadline (incl. attach-reset extension) \
             must fit the 240s x 2 nextest override",
        );
    }

    // -- sys_rdy_budget_ms / vm_boot_headroom --

    #[test]
    fn sys_rdy_budget_ms_base_plus_linear_per_vcpu() {
        // Additive: 10_000 ms base + vcpus × 150. Every topology gets
        // the base PLUS its per-vCPU term — no dead floor below 67
        // vCPUs (the bug that gave a 64-vCPU VM the same 10 s as a
        // 1-vCPU VM).
        assert_eq!(sys_rdy_budget_ms(1), 10_150);
        assert_eq!(sys_rdy_budget_ms(32), 14_800);
        assert_eq!(sys_rdy_budget_ms(66), 19_900);
    }

    #[test]
    fn sys_rdy_budget_ms_scales_linearly_in_band() {
        // 10_000 ms base + vcpus × 150, in the band below the 90 s cap.
        assert_eq!(sys_rdy_budget_ms(67), 20_050);
        // The 126-vCPU case lands at 28.9 s.
        assert_eq!(sys_rdy_budget_ms(126), 28_900);
        // 256-vCPU wide-SMP gets its FULL additive budget (48.4 s) — the
        // case the old 30 s cap truncated to 30 s, starving the boot.
        assert_eq!(sys_rdy_budget_ms(256), 48_400);
    }

    #[test]
    fn sys_rdy_budget_ms_caps_at_ninety_seconds() {
        // The 512-vCPU MAX_VCPUS topology gets its full additive budget
        // (10_000 + 512×150 = 86_800 ms), comfortably under the cap.
        assert_eq!(sys_rdy_budget_ms(512), 86_800);
        // 533 vCPUs is the last under the 90 s cap (10_000 + 533×150 =
        // 89_950); 534 is the first clipped (10_000 + 534×150 = 90_100
        // → 90_000). Only pathological >533-vCPU counts clip.
        assert_eq!(sys_rdy_budget_ms(533), 89_950);
        assert_eq!(sys_rdy_budget_ms(534), 90_000);
        assert_eq!(sys_rdy_budget_ms(u32::MAX), 90_000);
    }

    #[test]
    fn sys_rdy_budget_ms_zero_returns_base() {
        // Guest fallback when /sys/devices/system/cpu/online is missing:
        // 0 vCPUs → the bare 10_000 ms base (no per-vCPU term).
        assert_eq!(sys_rdy_budget_ms(0), 10_000);
    }

    #[test]
    fn vm_boot_headroom_is_ten_plus_sys_rdy_budget() {
        // KERNEL_INIT_HEADROOM (10 s) + sys_rdy_budget_ms(vcpus).
        assert_eq!(vm_boot_headroom(1), Duration::from_millis(20_150));
        assert_eq!(vm_boot_headroom(126), Duration::from_millis(38_900));
        // 256-vCPU wide-SMP: 10 s + 48.4 s = 58.4 s. Host overcommit
        // does not change this flat deadman component.
        assert_eq!(vm_boot_headroom(256), Duration::from_millis(58_400));
        // 512-vCPU MAX_VCPUS budget (86.8 s) → 96.8 s headroom, uncapped
        // under the 90 s ceiling.
        assert_eq!(vm_boot_headroom(512), Duration::from_millis(96_800));
    }

    // -- phase_cpu_budget_ns / phase_wall_backstop_ns --

    #[test]
    fn phase_cpu_budget_ns_boot_reuses_vm_boot_headroom() {
        const S: u64 = 1_000_000_000;
        // Boot consumes the same allowance as the VM deadline. The
        // remaining phases stay flat against max-per-vCPU evidence.
        assert_eq!(phase_cpu_budget_ns(0, 1), 20_150_000_000);
        assert_eq!(phase_cpu_budget_ns(0, 240), 56 * S);
        assert_eq!(
            phase_cpu_budget_ns(0, 240),
            vm_boot_headroom(240).as_nanos() as u64,
        );
        assert_eq!(phase_cpu_budget_ns(1, 1), 35 * S);
        assert_eq!(phase_cpu_budget_ns(1, 240), 35 * S);
        assert_eq!(phase_cpu_budget_ns(2, 1), 8 * S);
        assert_eq!(phase_cpu_budget_ns(2, 240), 8 * S);
        assert_eq!(phase_cpu_budget_ns(4, 1), 8 * S);
        assert_eq!(phase_cpu_budget_ns(4, 240), 8 * S);
    }

    #[test]
    fn phase_cpu_budget_ns_body_and_unknown_are_sentinel() {
        // Body (3): Tier-1 structurally off — the sentinel makes any
        // `cpu > budget` comparison unsatisfiable.
        assert_eq!(phase_cpu_budget_ns(3, 240), u64::MAX);
        // Unknown phase ids never carry a killable budget.
        assert_eq!(phase_cpu_budget_ns(5, 240), u64::MAX);
        assert_eq!(phase_cpu_budget_ns(u8::MAX, 240), u64::MAX);
    }

    #[test]
    fn phase_wall_backstop_ns_values_sit_above_in_band_deadlines() {
        const S: u64 = 1_000_000_000;
        assert_eq!(phase_wall_backstop_ns(0, 1), 45 * S);
        assert_eq!(phase_wall_backstop_ns(0, 240), 61 * S);
        assert_eq!(phase_wall_backstop_ns(0, 512), 101_800_000_000);
        for vcpus in [1, 240, 512] {
            assert!(
                phase_wall_backstop_ns(0, vcpus) > vm_boot_headroom(vcpus).as_nanos() as u64,
                "Boot Tier-2 must sit after the authoritative allowance at {vcpus} vCPUs",
            );
        }
        // Attach backstop must clear the host cold-BTF phase-1 budget.
        assert_eq!(phase_wall_backstop_ns(1, 1), 40 * S);
        assert!(phase_wall_backstop_ns(1, 512) > COLD_BTF_PHASE1_BUDGET.as_nanos() as u64);
        // Dispatch backstop must clear the guest probe's 30s cap.
        assert_eq!(phase_wall_backstop_ns(2, 1), 35 * S);
        assert_eq!(phase_wall_backstop_ns(4, 1), 15 * S);
    }

    #[test]
    fn phase_wall_backstop_ns_body_and_unknown_are_sentinel() {
        // Body (3) and unknown ids: Tier-2 structurally off.
        assert_eq!(phase_wall_backstop_ns(3, 240), u64::MAX);
        assert_eq!(phase_wall_backstop_ns(5, 240), u64::MAX);
        assert_eq!(phase_wall_backstop_ns(u8::MAX, 240), u64::MAX);
    }

    /// Two calls to `content_hash` with the same input must return
    /// the same u64. Pins the within-process determinism invariant
    /// against a future regression that swaps in a per-call-seeded
    /// hasher — e.g. `std::hash::RandomState::new().build_hasher()`,
    /// which increments its keys per call within a process, or any
    /// time/thread-id-seeded scheme. Note: swapping to std's
    /// `DefaultHasher::new()` would NOT regress this test —
    /// `DefaultHasher` is itself `SipHasher13::new_with_keys(0, 0)`
    /// and therefore deterministic; the cross-rustc-version
    /// stability regression class is caught by the value-pin
    /// follow-up, not this assertion.
    #[test]
    fn content_hash_is_deterministic_across_calls() {
        let input = "scheduler config payload";
        assert_eq!(content_hash(input), content_hash(input));
    }

    /// Distinct inputs must produce distinct hashes. Catches a trivial
    /// regression (constant-returning hasher) that the determinism
    /// test alone would silently accept.
    #[test]
    fn content_hash_differs_for_distinct_inputs() {
        assert_ne!(content_hash("alpha"), content_hash("beta"));
    }

    /// Cross-toolchain stability pin: every `content_hash` output must
    /// equal the SipHasher13(keys=0,0) value emitted at commit time.
    /// Pins the algorithm choice — a future swap to a different
    /// stable hasher (e.g. xxhash, fxhash) would silently regenerate
    /// every content-addressed cache filename on disk, breaking cache
    /// hit rates without surfacing as a failed test. The companion
    /// `content_hash_is_deterministic_across_calls` pin guards
    /// within-process determinism; this pin guards cross-process /
    /// cross-toolchain / cross-machine stability.
    #[test]
    fn content_hash_value_pin() {
        // SipHasher13(keys=0,0) over the four corpora below. If any
        // assertion fails, the algorithm or its seeding changed —
        // STOP. `content_hash` names the inline-config tempfile in
        // `config_content_parts` at src/test_support/runtime.rs and
        // the export-config tempfile in `export.rs`; flipping the
        // hashes silently regenerates those filenames on every
        // process, breaking any future scheme that tries to dedup
        // across runs and breaking intra-run reproducibility if a
        // caller comes to depend on stable byte equality across
        // identical inputs. Update only after intentional algorithm
        // migration. The four corpora — empty + two short ASCII +
        // one realistic config payload — span the cases the
        // algorithm needs to handle correctly.
        assert_eq!(content_hash(""), 0x30406ea523c53def);
        assert_eq!(content_hash("alpha"), 0x3c87f3c3317bd39a);
        assert_eq!(content_hash("beta"), 0xbb8fd2aa1487d7ac);
        assert_eq!(content_hash("scheduler config payload"), 0xc678971ba48d5f80);
    }

    /// Per-content-hash inline-config files MUST land inside the
    /// per-process `scratch_dir()` subtree, NOT bare
    /// `std::env::temp_dir()`. The 0o700 process-owned subdirectory
    /// blocks the cross-uid symlink-replacement attack on
    /// predictable content-addressed filenames in shared `/tmp`. A
    /// future "simplification" that reverts the path to bare
    /// `std::env::temp_dir().join(...)` silently restores the
    /// attack surface; this test fails loudly first.
    #[test]
    fn config_content_parts_writes_inside_process_scratch_dir() {
        use crate::assert::Assert;
        use crate::scenario::Ctx;
        use crate::test_support::entry::{
            KtstrTestEntry, Scheduler, SchedulerSpec, TopologyConstraints,
        };
        use crate::vmm::topology::Topology;

        static SCHED: Scheduler = Scheduler {
            name: "config_parts_test_sched",
            binary: SchedulerSpec::Discover("nope"),
            sysctls: &[],
            kargs: &[],
            assert: Assert::NO_OVERRIDES,
            cgroup_parent: None,
            sched_args: &[],
            topology: Topology {
                llcs: 1,
                cores_per_llc: 1,
                threads_per_core: 1,
                numa_nodes: 1,
                nodes: None,
                distances: None,
                llc_cores: None,
            },
            constraints: TopologyConstraints::DEFAULT,
            config_file: None,
            config_file_def: Some(("--config={file}", "/include-files/p.json")),
            kernels: &[],
            verifier_exclude_topologies: &[],
            manifest_dir: env!("CARGO_MANIFEST_DIR"),
        };
        fn func(_: &Ctx) -> anyhow::Result<crate::assert::AssertResult> {
            Ok(crate::assert::AssertResult::pass())
        }
        let entry = KtstrTestEntry {
            name: "scratch_dir_path_test",
            func,
            scheduler: &SCHED,
            config_content: Some("{\"sentinel\":42}"),
            ..KtstrTestEntry::DEFAULT
        };
        let (_, host_path, _, _) =
            config_content_parts(&entry).expect("config_content_parts returns Some");
        assert!(
            host_path.starts_with(scratch_dir()),
            "config tempfile must live inside the process-owned scratch dir, \
             not bare std::env::temp_dir(): got host_path={host_path:?}, \
             scratch_dir={:?}",
            scratch_dir()
        );
    }

    /// Two same-content calls produce the SAME canonical path
    /// (content-addressed naming idempotence). Callers using the
    /// returned PathBuf for downstream dedup decisions rely on this
    /// — a regression that breaks the content-hash → path mapping
    /// would silently spam the scratch dir with per-call distinct
    /// names instead of reusing the canonical entry.
    #[test]
    fn config_content_parts_same_content_same_canonical_path() {
        use crate::assert::Assert;
        use crate::scenario::Ctx;
        use crate::test_support::entry::{
            KtstrTestEntry, Scheduler, SchedulerSpec, TopologyConstraints,
        };
        use crate::vmm::topology::Topology;

        static SCHED: Scheduler = Scheduler {
            name: "config_parts_idempotent_sched",
            binary: SchedulerSpec::Discover("nope"),
            sysctls: &[],
            kargs: &[],
            assert: Assert::NO_OVERRIDES,
            cgroup_parent: None,
            sched_args: &[],
            topology: Topology {
                llcs: 1,
                cores_per_llc: 1,
                threads_per_core: 1,
                numa_nodes: 1,
                nodes: None,
                distances: None,
                llc_cores: None,
            },
            constraints: TopologyConstraints::DEFAULT,
            config_file: None,
            config_file_def: Some(("--config={file}", "/include-files/p.json")),
            kernels: &[],
            verifier_exclude_topologies: &[],
            manifest_dir: env!("CARGO_MANIFEST_DIR"),
        };
        fn func(_: &Ctx) -> anyhow::Result<crate::assert::AssertResult> {
            Ok(crate::assert::AssertResult::pass())
        }
        let entry = KtstrTestEntry {
            name: "idempotent_path_test",
            func,
            scheduler: &SCHED,
            config_content: Some("{\"idempotent\":true}"),
            ..KtstrTestEntry::DEFAULT
        };
        let (_, p1, _, _) = config_content_parts(&entry).expect("first call returns Some");
        let (_, p2, _, _) = config_content_parts(&entry).expect("second call returns Some");
        assert_eq!(
            p1, p2,
            "same content_content -> same canonical path; content-addressed naming \
             must be idempotent across calls"
        );
        // The filename component encodes the content hash via the
        // `ktstr-config-{hash:016x}.json` template; verify the prefix
        // so a future filename-template change is caught.
        let name = p1.file_name().and_then(|n| n.to_str()).unwrap_or("");
        assert!(
            name.starts_with("ktstr-config-") && name.ends_with(".json"),
            "canonical filename must follow `ktstr-config-{{hash}}.json` template, got: {name}"
        );
    }
}
