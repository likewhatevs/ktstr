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
/// gauntlet routes (`run_named_test`, `run_gauntlet_test` in
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
    std::env::var("KTSTR_NO_PERF_MODE")
        .map(|v| !v.is_empty())
        .unwrap_or(false)
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

/// True when `KTSTR_CARGO_TEST_MODE` is set to a NON-EMPTY value.
///
/// Marks a test invocation that runs without the cargo-ktstr wrapper
/// (typically `KTSTR_KERNEL=... KTSTR_CARGO_TEST_MODE=1 cargo test
/// -- some_test`). When active, the harness:
///
/// 1. Skips the cross-process initramfs SHM cache and builds the
///    initramfs inline per VM run (no `shm_open`, no `flock`-based
///    builder election; the process-local HashMap still memoises).
/// 2. Skips host-topology LLC / per-CPU flock acquisition — tests
///    run on whatever CPUs the OS schedules them onto. Acceptable
///    for development iteration; perf-mode tests still use their
///    measurement contract internally but no peer-coordination
///    flocks are taken.
/// 3. Skips gauntlet variant expansion in nextest discovery —
///    each `#[ktstr_test]` runs once with its declared topology.
///    No multi-kernel fan-out via `KTSTR_KERNEL_LIST`.
/// 4. Resolves `SchedulerSpec::Discover(name)` via `$PATH` first
///    (before the sibling-dir / target-dir / build chain) so a
///    user can install scx_layered on PATH and run their test
///    without driving the cargo-ktstr build pipeline.
///
/// Empty-string rejection mirrors [`no_perf_mode_active`]: a stray
/// `KTSTR_CARGO_TEST_MODE=` from CI shell quirks must not silently
/// flip the harness into degraded coordination mode.
pub(crate) fn cargo_test_mode_active() -> bool {
    std::env::var("KTSTR_CARGO_TEST_MODE")
        .map(|v| !v.is_empty())
        .unwrap_or(false)
}

/// Derive initramfs archive path, host path, and guest path from a
/// scheduler's `config_file`. Returns `None` when no config file is set.
pub(crate) fn config_file_parts(entry: &KtstrTestEntry) -> Option<(String, PathBuf, String)> {
    let config_path = entry.scheduler.config_file?;
    let file_name = Path::new(config_path)
        .file_name()
        .and_then(|n| n.to_str())
        .expect("config_file must have a valid filename");
    let archive_path = format!("include-files/{file_name}");
    let guest_path = format!("/include-files/{file_name}");
    Some((archive_path, PathBuf::from(config_path), guest_path))
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
pub(crate) fn build_cmdline_extra(entry: &KtstrTestEntry) -> String {
    let mut parts: Vec<String> = Vec::new();
    for s in entry.scheduler.sysctls {
        parts.push(format!("sysctl.{}={}", s.key, s.value));
    }
    for &karg in entry.scheduler.kargs {
        parts.push(karg.to_string());
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

/// Resolve the VM topology and memory size from an optional
/// TopoOverride.
///
/// Returns `(topology, memory_mib)` where `topology` is the
/// `vmm::topology::Topology` passed to the VM builder and `memory_mib`
/// is the memory allocation in megabytes. When `topo` is `Some`, both
/// come from the override. When `topo` is `None`, the topology comes
/// from `entry.topology` and memory is `max(total_cpus * 64, 256,
/// entry.memory_mib)`. Shared with `attempt_auto_repro` so the repro
/// VM always sizes memory the same way as the first VM.
pub(crate) fn resolve_vm_topology(
    entry: &KtstrTestEntry,
    topo: Option<&super::topo::TopoOverride>,
) -> (crate::vmm::topology::Topology, u32) {
    match topo {
        Some(t) => (crate::vmm::topology::Topology::from(t), t.memory_mib),
        None => {
            let cpus = entry.topology.total_cpus();
            let mem = (cpus * 64).max(256).max(entry.memory_mib);
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
    // `resolve_cgroup_root` (used by the probe/setup path at
    // `probe.rs::run_scenario_probe`) interpolates the value into a
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
            // No user-supplied flag — auto-inject the scheduler's
            // default `cgroup_parent` when one is declared. The same
            // parser is used guest-side by `resolve_cgroup_root` and
            // `create_cgroup_parent_from_sched_args`, so the auto-
            // injected two-token form produces identical guest
            // cgroup-tree creation to a user-supplied value.
            if let Some(cgroup_path) = entry.scheduler.cgroup_parent {
                args.push(super::args::CELL_PARENT_CGROUP_FLAG.to_string());
                args.push(cgroup_path.to_string());
            }
        }
    }
    args.extend(entry.scheduler.sched_args.iter().map(|s| s.to_string()));
    args.extend(entry.extra_sched_args.iter().map(|s| s.to_string()));
}

/// Retry budget for the guest's `send_sys_rdy` loop in
/// `vmm::rust_init`. Scales with vCPU count because the virtio-console
/// multiport handshake (DEVICE_READY → PORT_ADD → PORT_READY →
/// PORT_OPEN per `drivers/char/virtio_console.c`) issues per-CPU work
/// whose wall time grows roughly linearly with topology size. A
/// 126-vCPU test under host contention has been observed to need
/// ~10 s for the handshake alone; 150 ms/vCPU gives ~50 % headroom.
///
/// Floored at 10 s (preserves the prior single-CPU default) and
/// capped at 30 s so pathological topologies (512 vCPUs would naively
/// land at 76 s) don't blow the watchdog budget.
///
/// The const-fn signature lets both the host (`vm_boot_headroom`,
/// `vm_timeout_from_entry`) and the guest (`vmm::rust_init`) compute
/// the same budget without trans-VM coordination — the guest reads
/// its own vCPU count from `/sys/devices/system/cpu/online`.
pub(crate) const fn sys_rdy_budget_ms(vcpus: u32) -> u64 {
    const FLOOR_MS: u64 = 10_000;
    const CAP_MS: u64 = 30_000;
    const PER_VCPU_MS: u64 = 150;
    let scaled = (vcpus as u64).saturating_mul(PER_VCPU_MS);
    let bounded = if scaled > CAP_MS { CAP_MS } else { scaled };
    if bounded > FLOOR_MS {
        bounded
    } else {
        FLOOR_MS
    }
}

/// Headroom for kernel init, scheduler attach, and BPF verifier time
/// — the post-sys_rdy phase of guest startup. Distinct from
/// [`sys_rdy_budget_ms`]'s floor (which is the pre-sys_rdy
/// virtio-console handshake budget); the two add together to form
/// the full [`vm_boot_headroom`].
const KERNEL_INIT_HEADROOM: Duration = Duration::from_secs(10);

/// Total boot headroom: covers kernel init + scheduler attach + BPF
/// verifier time ([`KERNEL_INIT_HEADROOM`]) plus the guest's scaled
/// `send_sys_rdy` retry loop ([`sys_rdy_budget_ms`]) before the
/// workload phase begins. Scales with vCPU count so the host timeout
/// doesn't fire while the guest is still inside its sys_rdy budget.
pub(crate) fn vm_boot_headroom(vcpus: u32) -> Duration {
    KERNEL_INIT_HEADROOM + Duration::from_millis(sys_rdy_budget_ms(vcpus))
}

/// Derive the host-side VM timeout from the test entry's watchdog
/// and duration. Adds vCPU-scaled boot headroom so the workload gets
/// its full duration even after a slow boot on a large topology.
pub(crate) fn vm_timeout_from_entry(entry: &super::entry::KtstrTestEntry) -> Duration {
    let base = entry
        .watchdog_timeout
        .max(entry.duration)
        .max(Duration::from_secs(1));
    base + vm_boot_headroom(entry.topology.total_cpus())
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
        .init_binary(ktstr_bin)
        .topology(vm_topology)
        .memory_deferred_min(memory_mib)
        .cmdline(cmdline_extra)
        .run_args(guest_args)
        .timeout(vm_timeout_from_entry(entry))
        .workload_duration(entry.duration)
        .no_perf_mode(no_perf_mode);

    if let Some(sched_path) = scheduler {
        builder = builder.scheduler_binary(sched_path);
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
    //             "KTSTR_JEMALLOC_PROBE_BINARY",
    //             env!("CARGO_BIN_EXE_ktstr-jemalloc-probe"),
    //         );
    //         std::env::set_var(
    //             "KTSTR_JEMALLOC_ALLOC_WORKER_BINARY",
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
    //             "KTSTR_JEMALLOC_PROBE_BINARY",
    //             env!("CARGO_BIN_EXE_ktstr-jemalloc-probe"),
    //         );
    //         std::env::set_var(
    //             "KTSTR_JEMALLOC_ALLOC_WORKER_BINARY",
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
    if let Ok(probe_path) = std::env::var("KTSTR_JEMALLOC_PROBE_BINARY")
        && !probe_path.is_empty()
    {
        // Pack the probe binary into the guest initramfs at
        // `/bin/ktstr-jemalloc-probe`. Closed-loop probe tests run
        // the probe via `--pid <alloc_worker_pid>` against the
        // paired `ktstr-jemalloc-alloc-worker` target; DWARF comes
        // from the worker's own ELF, not the init's.
        builder = builder.jemalloc_probe_binary(std::path::PathBuf::from(probe_path));
    }
    if let Ok(worker_path) = std::env::var("KTSTR_JEMALLOC_ALLOC_WORKER_BINARY")
        && !worker_path.is_empty()
    {
        // Pack the jemalloc-alloc-worker binary alongside the
        // probe. Only the cross-process closed-loop test sets
        // this; scheduler-only tests leave the env var unset and
        // skip the wiring.
        builder = builder.jemalloc_alloc_worker_binary(std::path::PathBuf::from(worker_path));
    }

    for bpf_write in entry.bpf_map_write {
        builder =
            builder.bpf_map_write(bpf_write.map_name_suffix, bpf_write.offset, bpf_write.value);
    }

    if let Some(disk_cfg) = entry.disk.clone() {
        builder = builder.disk(disk_cfg);
    }

    builder = builder.num_snapshots(entry.num_snapshots);

    builder.watchdog_timeout(entry.watchdog_timeout)
}

#[cfg(test)]
mod tests {
    use super::super::entry::Scheduler;
    use super::*;

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
    use super::super::test_helpers::{EnvVarGuard, lock_env};

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
        let _env_sd = EnvVarGuard::set("KTSTR_SIDECAR_DIR", "/tmp/ktstr-test");

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
        let _env_sd = EnvVarGuard::set("KTSTR_SIDECAR_DIR", "/tmp/ktstr-test");

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
    fn build_cmdline_extra_propagates_rust_env() {
        let _lock = lock_env();
        let _env_bt = EnvVarGuard::set("RUST_BACKTRACE", "1");
        let _env_log = EnvVarGuard::set("RUST_LOG", "debug");
        let _env_sd = EnvVarGuard::set("KTSTR_SIDECAR_DIR", "/tmp/ktstr-test");

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
        let _env_sd = EnvVarGuard::set("KTSTR_SIDECAR_DIR", "/explicit/sidecar/dir");

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
    fn resolve_vm_topology_none_floors_memory_at_256() {
        // Tiny topology: 1*1*1=1 cpu -> 64 MiB raw, entry.memory_mib=0,
        // floor = max(64, 256, 0) = 256.
        //
        // Override memory_mib explicitly to 0 — KtstrTestEntry::DEFAULT
        // sets memory_mib=2048, which would bypass the floor entirely
        // and leave this test vacuously passing regardless of the
        // max(…, 256, …) branch. Setting memory_mib=0 makes the 256
        // floor the exact lower bound the assertion verifies.
        let entry = KtstrTestEntry {
            name: "tiny",
            memory_mib: 0,
            ..KtstrTestEntry::DEFAULT
        };
        let (_topo, mem) = resolve_vm_topology(&entry, None);
        assert_eq!(mem, 256, "memory floor = 256 MiB, got {mem}");
    }

    #[test]
    fn resolve_vm_topology_none_honors_entry_memory_mib() {
        // Entry with explicit memory_mib above the cpu*64 and 256 floors.
        let entry = KtstrTestEntry {
            name: "mem",
            memory_mib: 8192,
            ..KtstrTestEntry::DEFAULT
        };
        let (_topo, mem) = resolve_vm_topology(&entry, None);
        assert_eq!(mem, 8192);
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

    #[test]
    fn append_base_sched_args_includes_cgroup_parent_and_sched_args() {
        use super::super::entry::CgroupPath;
        static CG: CgroupPath = CgroupPath::new("/sys/fs/cgroup/ktstr");
        static SCHED: Scheduler = Scheduler::named("s")
            .cgroup_parent("/sys/fs/cgroup/ktstr")
            .sched_args(&["-v", "--flag"]);
        // Touch the static so the compiler doesn't drop it; verifies
        // the path we store matches what cgroup_parent produces.
        let _ = &CG;
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
                "--cell-parent-cgroup".to_string(),
                "/sys/fs/cgroup/ktstr".to_string(),
                "-v".to_string(),
                "--flag".to_string(),
                "--extra".to_string(),
            ],
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
        };
        let result = build_vm_builder_base(
            &entry,
            &exe,
            &exe,
            None,
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

    // -- cargo_test_mode_active --

    /// `KTSTR_CARGO_TEST_MODE=1` (any non-empty value) flips the
    /// flag to true. Pins the activation contract against the
    /// dispatch sites that gate gauntlet emission, scheduler PATH
    /// lookup, initramfs-cache SHM bypass, and host-topology flock
    /// bypass on this exact predicate.
    #[test]
    fn cargo_test_mode_active_set_non_empty() {
        let _lock = lock_env();
        let _env = EnvVarGuard::set("KTSTR_CARGO_TEST_MODE", "1");
        assert!(cargo_test_mode_active());
        let _env_word = EnvVarGuard::set("KTSTR_CARGO_TEST_MODE", "yes");
        assert!(cargo_test_mode_active());
    }

    /// Empty-string rejection: `KTSTR_CARGO_TEST_MODE=` does NOT
    /// flip the flag. Mirrors `no_perf_mode_active`'s behavior so
    /// CI shells / docker pass-through that set the var without a
    /// value never accidentally degrade the harness.
    #[test]
    fn cargo_test_mode_active_empty_string_rejected() {
        let _lock = lock_env();
        let _env = EnvVarGuard::set("KTSTR_CARGO_TEST_MODE", "");
        assert!(
            !cargo_test_mode_active(),
            "empty-string KTSTR_CARGO_TEST_MODE must NOT activate; \
             treating it as truthy would silently degrade SHM / \
             flock coordination on a stray `--env` pass-through"
        );
    }

    /// Unset env var → false.
    #[test]
    fn cargo_test_mode_active_unset() {
        let _lock = lock_env();
        let _env = EnvVarGuard::remove("KTSTR_CARGO_TEST_MODE");
        assert!(!cargo_test_mode_active());
    }

    // -- vm_timeout_from_entry tests --

    #[test]
    fn vm_timeout_from_entry_uses_watchdog_when_largest() {
        // DEFAULT topology = 2 vCPUs → sys_rdy_budget_ms = 10_000 (floor)
        // → vm_boot_headroom = 20 s; base = max(60s, 30s, 1s) = 60s.
        let entry = KtstrTestEntry {
            name: "wdog",
            watchdog_timeout: Duration::from_secs(60),
            duration: Duration::from_secs(30),
            ..KtstrTestEntry::DEFAULT
        };
        assert_eq!(vm_timeout_from_entry(&entry), Duration::from_secs(80));
    }

    #[test]
    fn vm_timeout_from_entry_uses_duration_when_largest() {
        let entry = KtstrTestEntry {
            name: "dur",
            watchdog_timeout: Duration::from_secs(5),
            duration: Duration::from_secs(120),
            ..KtstrTestEntry::DEFAULT
        };
        assert_eq!(vm_timeout_from_entry(&entry), Duration::from_secs(140));
    }

    #[test]
    fn vm_timeout_from_entry_floor_when_both_small() {
        // base floors at 1 s; vm_boot_headroom for 2 vCPUs is 20 s.
        let entry = KtstrTestEntry {
            name: "tiny",
            watchdog_timeout: Duration::from_millis(10),
            duration: Duration::from_millis(50),
            ..KtstrTestEntry::DEFAULT
        };
        assert_eq!(vm_timeout_from_entry(&entry), Duration::from_secs(21));
    }

    #[test]
    fn vm_timeout_from_default_entry() {
        // DEFAULT watchdog = 5 s, duration = 12 s → base = 12 s.
        // vm_boot_headroom for 2 vCPUs = 20 s → 32 s total.
        let entry = KtstrTestEntry {
            name: "default",
            ..KtstrTestEntry::DEFAULT
        };
        assert_eq!(vm_timeout_from_entry(&entry), Duration::from_secs(32));
    }

    #[test]
    fn vm_timeout_from_entry_scales_headroom_with_topology() {
        // The B2 reporter's case: numa=1, llcs=7, cores=9, threads=2 → 126 vCPUs.
        // sys_rdy_budget_ms(126) = 18_900 ms → vm_boot_headroom = 28.9 s.
        // base = max(5 s watchdog, 12 s duration, 1 s) = 12 s → total = 40.9 s.
        // Pins the `entry.topology.total_cpus()` → `vm_boot_headroom` wiring.
        let entry = KtstrTestEntry {
            name: "large_topo",
            topology: crate::vmm::topology::Topology {
                llcs: 7,
                cores_per_llc: 9,
                threads_per_core: 2,
                numa_nodes: 1,
                nodes: None,
                distances: None,
            },
            ..KtstrTestEntry::DEFAULT
        };
        assert_eq!(vm_timeout_from_entry(&entry), Duration::from_millis(40_900));
    }

    // -- sys_rdy_budget_ms / vm_boot_headroom --

    #[test]
    fn sys_rdy_budget_ms_floor_holds_for_small_topologies() {
        assert_eq!(sys_rdy_budget_ms(1), 10_000);
        assert_eq!(sys_rdy_budget_ms(32), 10_000);
        assert_eq!(sys_rdy_budget_ms(66), 10_000);
    }

    #[test]
    fn sys_rdy_budget_ms_scales_linearly_in_band() {
        // 67 vCPUs is the first to exceed the 10 s floor (67*150 = 10050).
        assert_eq!(sys_rdy_budget_ms(67), 10_050);
        // The reporter's 126-vCPU case lands at 18.9 s.
        assert_eq!(sys_rdy_budget_ms(126), 18_900);
        // 192 vCPUs still under the cap.
        assert_eq!(sys_rdy_budget_ms(192), 28_800);
    }

    #[test]
    fn sys_rdy_budget_ms_caps_at_thirty_seconds() {
        // 200 vCPUs = exactly the cap (200*150 = 30000).
        assert_eq!(sys_rdy_budget_ms(200), 30_000);
        // Pathological topologies clip — no unbounded budget.
        assert_eq!(sys_rdy_budget_ms(512), 30_000);
        assert_eq!(sys_rdy_budget_ms(u32::MAX), 30_000);
    }

    #[test]
    fn sys_rdy_budget_ms_zero_returns_floor() {
        // Guest fallback when /sys/devices/system/cpu/online is missing.
        assert_eq!(sys_rdy_budget_ms(0), 10_000);
    }

    #[test]
    fn vm_boot_headroom_is_ten_plus_sys_rdy_budget() {
        assert_eq!(vm_boot_headroom(1), Duration::from_secs(20));
        assert_eq!(vm_boot_headroom(126), Duration::from_millis(28_900));
        assert_eq!(vm_boot_headroom(512), Duration::from_secs(40));
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
            },
            constraints: TopologyConstraints::DEFAULT,
            config_file: None,
            config_file_def: Some(("--config={file}", "/include-files/p.json")),
            kernels: &[],
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
            },
            constraints: TopologyConstraints::DEFAULT,
            config_file: None,
            config_file_def: Some(("--config={file}", "/include-files/p.json")),
            kernels: &[],
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
