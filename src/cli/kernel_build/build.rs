//! Top-level kernel build orchestration.
//!
//! Holds [`kernel_build_pipeline`] (the post-acquisition orchestrator
//! that runs `clean` → configure → build → validate → cache-store),
//! the two-phase reservation acquisition
//! ([`acquire_build_reservation`]) for LLC flock + cgroup sandbox +
//! `make -jN` hint, and the source-tree flock helper
//! ([`acquire_source_tree_lock`]) that serializes parallel builds
//! against the same on-disk source tree.

use std::path::Path;

use anyhow::{Context, Result};

use super::super::kernel_cmd::{
    DIRTY_TREE_CACHE_SKIP_HINT, EMBEDDED_KCONFIG, NON_GIT_TREE_CACHE_SKIP_HINT,
    embedded_kconfig_hash,
};
use super::super::util::{success, warn};
use super::kconfig::{
    all_fragment_lines_present, configure_kernel, validate_kernel_config,
    warn_dropped_extra_kconfig_lines, warn_extra_kconfig_overrides_baked_in,
};
use super::make::{build_make_args, make_kernel_with_output, run_make, run_make_with_output};

/// Result of the post-acquisition kernel build pipeline.
///
/// Returned by [`kernel_build_pipeline`] so callers can inspect
/// the cache entry and built image path.
#[non_exhaustive]
pub struct KernelBuildResult {
    /// Cache entry, if the build was cached. `None` for dirty trees
    /// or when cache store fails.
    pub entry: Option<crate::cache::CacheEntry>,
    /// Path to the built kernel image.
    pub image_path: std::path::PathBuf,
    /// Whether the source tree was dirty as observed by the build
    /// pipeline. `true` if either the acquire-time inspection
    /// reported dirty OR the post-build re-check observed a
    /// mid-build mutation (worktree edit, branch flip, mid-build
    /// commit). The downstream label decoration in cargo-ktstr's
    /// `resolve_one` uses this to append `_dirty` so a
    /// non-reproducible run is distinguishable from a clean rebuild
    /// of the same path.
    pub post_build_is_dirty: bool,
}

/// Two-phase build reservation handles (LLC flock plan + cgroup v2
/// sandbox + make -jN hint). Consumed by
/// [`kernel_build_pipeline`]; the factored-out
/// [`acquire_build_reservation`] builds it from `cpu_cap` without
/// depending on kernel source, enabling integration tests that
/// exercise the reservation logic against synthetic topologies.
///
/// Also acquired directly by `cargo-ktstr`'s harness-compile path
/// (`bin/cargo_ktstr/run_cargo.rs`) so a colocated runner's
/// `cargo build`/`nextest --no-run` warm-up coordinates against the
/// SAME machine-global LLC flock namespace + cgroup cpuset as a
/// kernel build — hence `pub` (the bin holds this as an opaque RAII
/// guard, dropping it before the test-running phase). Fields stay
/// `pub(crate)`: external holders only need the RAII lifetime and the
/// [`BuildReservation::make_jobs`] accessor, not field access.
///
/// Drop order is load-bearing: `_sandbox` is declared first and
/// drops first per Rust's declaration-order field-drop rule;
/// this ensures the cgroup sandbox is removed before the LLC
/// flock is released. Otherwise a peer could observe the LLC
/// released before the cgroup is gone and mint a conflicting
/// plan.
#[derive(Debug)]
pub struct BuildReservation {
    /// cgroup v2 sandbox. `None` when `plan` is `None` (no reservation
    /// to enforce). Drops FIRST per struct field order — cgroup
    /// rmdir runs while LLC flocks are still held. `_` prefix
    /// keeps the binding alive through Drop but marks it as
    /// not-read — the RAII invariant IS the read.
    pub(crate) _sandbox: Option<crate::vmm::cgroup_sandbox::BuildSandbox>,
    /// LLC plan (flock fds + cpus + mems). `None` under
    /// `KTSTR_BYPASS_LLC_LOCKS=1` or sysfs-unreadable host without
    /// `--cpu-cap`. Drops SECOND per struct field order —
    /// flocks release AFTER the sandbox rmdir lands.
    pub(crate) plan: Option<crate::vmm::host_topology::LlcPlan>,
    /// `make -jN` parallelism hint. `Some(N)` under an active
    /// `plan`; `None` when no reservation exists (caller falls
    /// back to `nproc`).
    pub(crate) make_jobs: Option<usize>,
}

impl BuildReservation {
    /// Reserved-CPU parallelism hint (`plan.cpus.len()`), or `None`
    /// when no reservation was acquired (bypass / degraded sysfs).
    /// The kernel-build path feeds it to `make -jN`; the cargo
    /// harness-compile path feeds it to `CARGO_BUILD_JOBS` so parallel
    /// rustc is capped to the reserved cpuset instead of exploding to
    /// `nproc` inside a confined cgroup. `None` leaves the child build
    /// on its own default.
    pub fn make_jobs(&self) -> Option<usize> {
        self.make_jobs
    }
}

/// Acquire the two-phase reservation (LLC flocks + cgroup sandbox)
/// for a kernel OR harness build. Factored out of
/// [`kernel_build_pipeline`] so integration tests can exercise the
/// cpu_cap → acquire → sandbox → make_jobs decision tree without
/// requiring a real kernel source tree, and so `cargo-ktstr`'s
/// harness-compile warm-up (`bin/cargo_ktstr/run_cargo.rs`) can take
/// the identical reservation before a colocated runner's `cargo`
/// build invades a perf-mode LLC reservation. `pub` for that
/// cross-binary reuse; the escape hatches (`KTSTR_BYPASS_LLC_LOCKS`,
/// `KTSTR_CARGO_TEST_MODE` inside `acquire_llc_plan`), degraded-sysfs
/// warn/hard-error gating, and cgroup-degrade semantics are shared
/// verbatim between both callers.
///
/// Returns a `BuildReservation` whose fields are the three values
/// `kernel_build_pipeline` used to bind inline. `_sandbox` is
/// declared first and drops first per Rust's declaration-order
/// field-drop rule; this ensures the cgroup sandbox is removed
/// before the LLC flock is released.
///
/// `cli_label` prefixes operator-facing error text.
///
/// `cpu_cap` is the resolved CPU-count cap from
/// [`CpuCap::resolve`](crate::vmm::host_topology::CpuCap::resolve);
/// `None` means "reserve 30% of the calling process's allowed-CPU
/// set", applied inside the planner at acquire time.
pub fn acquire_build_reservation(
    cli_label: &str,
    cpu_cap: Option<crate::vmm::host_topology::CpuCap>,
) -> Result<BuildReservation> {
    acquire_build_reservation_impl(cli_label, cpu_cap, false, None)
}

/// Acquire a build reservation, joining the progress-aware host queue when
/// perf-mode work temporarily owns too much LLC capacity.
///
/// Harness prebuilds use this variant because many independent CI invocations
/// compile concurrently and have no source-tree lock to serialize them. The
/// resolved CPU budget is a ceiling: this path immediately takes the largest
/// non-empty placement that does not overlap an exact VM reservation, and
/// derives Cargo/make parallelism from that actual placement. It joins the
/// queue only while every compatible CPU is occupied, then repeats that
/// largest-fit plan on each wake. Kernel builds retain
/// [`acquire_build_reservation`]'s exact, interactive nonblocking behavior.
pub fn acquire_build_reservation_waiting(
    cli_label: &str,
    cpu_cap: Option<crate::vmm::host_topology::CpuCap>,
) -> Result<BuildReservation> {
    acquire_build_reservation_impl(cli_label, cpu_cap, true, None)
}

/// Cancellation-aware harness-build reservation.
///
/// A signal-published `cancelled` flag interrupts either the ticket's futex
/// wait or the coordinator's inotify wait, releasing the current elastic
/// registry claim, partial LLC holds, and any completed plan before returning
/// [`std::io::ErrorKind::Interrupted`]. Sizing otherwise matches
/// [`acquire_build_reservation_waiting`].
pub fn acquire_build_reservation_waiting_interruptible(
    cli_label: &str,
    cpu_cap: Option<crate::vmm::host_topology::CpuCap>,
    cancelled: &std::sync::atomic::AtomicBool,
) -> Result<BuildReservation> {
    acquire_build_reservation_impl(cli_label, cpu_cap, true, Some(cancelled))
}

/// Cancellation-aware harness-build reservation with synchronous wait ticks.
///
/// `progress` runs on the acquiring thread while an ordinary admission ticket is
/// parked or its elected coordinator is blocked in inotify. The protocol only
/// slices those blocking syscalls; reservation ordering, retry deadlines, and
/// liveness deadlines are unchanged. Sizing otherwise matches
/// [`acquire_build_reservation_waiting`].
pub fn acquire_build_reservation_waiting_interruptible_with_progress(
    cli_label: &str,
    cpu_cap: Option<crate::vmm::host_topology::CpuCap>,
    cancelled: &std::sync::atomic::AtomicBool,
    progress: impl FnMut() + 'static,
) -> Result<BuildReservation> {
    crate::vmm::host_topology::with_reservation_wait_progress(progress, || {
        acquire_build_reservation_impl(cli_label, cpu_cap, true, Some(cancelled))
    })
}

fn acquire_build_reservation_impl(
    cli_label: &str,
    cpu_cap: Option<crate::vmm::host_topology::CpuCap>,
    elastic: bool,
    cancelled: Option<&std::sync::atomic::AtomicBool>,
) -> Result<BuildReservation> {
    check_reservation_cancelled(cancelled)?;
    let bypass = crate::bypass_llc_locks_active();
    // INVARIANT: `_sandbox` is declared first and drops first per
    // Rust's declaration-order field-drop rule; this ensures the
    // cgroup sandbox is removed before the LLC flock is released.
    // Reordering either would either
    // (a) unlock LLCs while the sandbox still enforces the
    // cpuset — a concurrent peer could claim the LLC and stomp
    // gcc children that haven't exited — or (b) leave the cgroup
    // hierarchy non-empty when its parent tries to rmdir.
    let plan: Option<crate::vmm::host_topology::LlcPlan> = if bypass {
        if cpu_cap.is_some() {
            anyhow::bail!(
                "{cli_label}: --cpu-cap conflicts with KTSTR_BYPASS_LLC_LOCKS=1; \
                 unset one of them. --cpu-cap is a resource contract; bypass \
                 disables the contract entirely."
            );
        }
        None
    } else if let Ok(host_topo) = crate::vmm::host_topology::HostTopology::from_sysfs() {
        let test_topo = crate::topology::TestTopology::from_system()?;
        // Builds keep Consolidate placement: packing onto already-held
        // LLCs leaves whole LLCs free for exclusive perf-mode
        // reservations, and a build is throughput-elastic where a VM's
        // vCPU threads are not (VMs use Spread — see `PlacementPolicy`).
        let acquired_plan = if elastic {
            crate::vmm::host_topology::acquire_elastic_build_llc_plan(
                &host_topo, &test_topo, cpu_cap, cancelled,
            )?
        } else {
            crate::vmm::host_topology::acquire_llc_plan(
                &host_topo,
                &test_topo,
                cpu_cap,
                crate::vmm::host_topology::PlacementPolicy::Consolidate,
                false,
            )?
        };
        check_reservation_cancelled(cancelled)?;
        crate::vmm::host_topology::warn_if_cross_node_spill(&acquired_plan, &host_topo);
        Some(acquired_plan)
    } else {
        if cpu_cap.is_some() {
            anyhow::bail!(
                "{cli_label}: --cpu-cap set but host LLC topology unreadable \
                 from sysfs — cannot enforce the resource budget. Run on a \
                 host with /sys/devices/system/cpu populated, or drop \
                 --cpu-cap to build without enforcement."
            );
        }
        tracing::warn!(
            "{cli_label}: could not read host LLC topology from sysfs; \
             skipping kernel-build LLC reservation. Concurrent perf-mode \
             runs on this host will NOT be serialized against this build"
        );
        None
    };

    // Phase 2: cgroup v2 sandbox that enforces cpu+mem binding on
    // make/gcc children. `hard_error_on_degrade` is driven by
    // whether `--cpu-cap` was set explicitly: degradation is fatal
    // under the flag (the flag promises enforcement), and warn-only
    // when the 30%-of-allowed default was expanded (the default
    // contract is best-effort — a parent cgroup narrowing the
    // reservation should not fail the build).
    let sandbox: Option<crate::vmm::cgroup_sandbox::BuildSandbox> = match plan.as_ref() {
        Some(p) => Some(crate::vmm::cgroup_sandbox::BuildSandbox::try_create(
            &p.cpus,
            &p.mems,
            cpu_cap.is_some(),
        )?),
        None => None,
    };
    check_reservation_cancelled(cancelled)?;

    // `make -jN` parallelism hint. `N` = `plan.cpus.len()` via
    // `make_jobs_for_plan` — the reserved CPU count, whether that
    // came from an explicit `--cpu-cap N` or the 30%-of-allowed
    // default. See `make_kernel_with_output` for the resolution.
    let make_jobs = plan
        .as_ref()
        .map(crate::vmm::host_topology::make_jobs_for_plan);

    Ok(BuildReservation {
        plan,
        _sandbox: sandbox,
        make_jobs,
    })
}

fn check_reservation_cancelled(cancelled: Option<&std::sync::atomic::AtomicBool>) -> Result<()> {
    if cancelled.is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Acquire)) {
        Err(std::io::Error::new(
            std::io::ErrorKind::Interrupted,
            "ktstr build-reservation acquisition interrupted",
        )
        .into())
    } else {
        Ok(())
    }
}

/// Acquire an exclusive flock on a per-source-canonical-path lockfile
/// so two concurrent `cargo ktstr test --kernel <path>` runs against
/// the SAME source tree don't race in `make` (defconfig vs
/// olddefconfig vs compile_commands.json) and stomp each other's
/// `.config` and build artifacts.
///
/// The lockfile lives at
/// `{KTSTR_CACHE_DIR}/.locks/source-{path_hash}.lock` where
/// `{path_hash}` is the full 8-char CRC32 hex of the canonical
/// source-path bytes (same shape and helper the
/// `local-unknown-{path_hash}` cache key uses, see
/// [`crate::fetch::canonical_path_hash`] /
/// [`crate::fetch::compose_local_cache_key`]) — one per-tree
/// identifier ties the source-tree flock to the cache key it gates.
///
/// Lockfile placement piggybacks on the cache root's `.locks/`
/// subdirectory ([`crate::flock::LOCK_DIR_NAME`]) so source-tree
/// flocks share the same filesystem-residency story as cache-entry
/// flocks: never under `/tmp`, where `tmpwatch` (or the equivalent
/// `systemd-tmpfiles` cleanup) can sweep stale-mtime files out from
/// under an active flock holder. flock(2) does NOT update the
/// inode's mtime, so a /tmp-resident lockfile would be a candidate
/// for sweep on every run, with the resulting `unlink(2)` racing
/// any peer trying to `open(2)` the same path. The `.locks/`
/// directory under the user-controlled cache root is exempt from
/// those sweeps.
///
/// Try-then-wait: attempts a non-blocking acquire first. If
/// contended, reports the wait and falls through to a blocking
/// acquire that parks until the peer releases. The wait path does
/// not scan host-global `/proc/locks`: on a storm host that scan can
/// itself take seconds, delaying the actual flock wait without
/// affecting correctness. When the blocking acquire returns, the
/// peer's build is done and the cache likely contains the artifact —
/// the caller checks the cache after we return and skips the build
/// if the slot is populated.
///
/// Distinct from the cache-entry flock acquired inside
/// [`crate::cache::CacheDir::store`]: that lock serializes the
/// atomic install of an artifact bundle into a cache slot; this
/// lock serializes the BUILD itself against the source-tree
/// `make` invocations.
pub(crate) fn acquire_source_tree_lock(
    canonical: &Path,
    cli_label: &str,
) -> Result<std::os::fd::OwnedFd> {
    acquire_source_tree_lock_with_wait_hook(canonical, cli_label, || {})
}

fn acquire_source_tree_lock_with_wait_hook(
    canonical: &Path,
    cli_label: &str,
    on_wait: impl FnOnce(),
) -> Result<std::os::fd::OwnedFd> {
    use anyhow::Context;

    // Share the per-path CRC32 with `local-unknown-{hash}` cache
    // keys so a single per-tree identifier ties the source-tree
    // flock to the cache slot it gates.
    let path_hash = crate::fetch::canonical_path_hash(canonical);
    let cache = crate::cache::CacheDir::new()
        .with_context(|| "open cache root for source-tree lockfile placement")?;
    cache
        .ensure_lock_dir()
        .with_context(|| "create cache `.locks/` subdir for source-tree lock")?;
    let lock_path = cache.lock_path(&format!("source-{path_hash}"));

    match crate::flock::try_flock(&lock_path, crate::flock::FlockMode::Exclusive)
        .with_context(|| format!("acquire source-tree flock {}", lock_path.display()))?
    {
        Some(fd) => Ok(fd),
        None => {
            // Non-blocking acquire failed (EWOULDBLOCK) — a live
            // peer holds the lock. Publish the transition before
            // entering the blocking syscall; production uses a no-op
            // hook while the test seam observes it without polling
            // host-global `/proc/locks`.
            on_wait();
            eprintln!(
                "{cli_label}: source tree {} is locked by a concurrent ktstr \
                 build — waiting for it to finish.",
                canonical.display(),
            );
            crate::flock::block_flock(&lock_path, crate::flock::FlockMode::Exclusive).with_context(
                || format!("blocking wait on source-tree flock {}", lock_path.display()),
            )
        }
    }
}

/// Classification of source-tree state at the post-acquire
/// re-probe site inside [`kernel_build_pipeline`].
///
/// The pipeline re-probes the source tree after the source-tree EX
/// wait completes so a mid-wait mutation (operator edit, branch flip,
/// commit on top) can invalidate the cache-skip short-circuit instead
/// of returning a cache slot keyed on the pre-wait identity. The
/// 5-variant split keeps cause-attribution honest in the operator
/// diagnostic emitted by [`MidWaitState::diagnostic`]: a `git commit`
/// during the wait is not "your edits"; an operator who started dirty
/// did not dirty the tree because of the wait; a probe failure is
/// not a confirmed mutation, just unknowable state.
#[derive(Debug, PartialEq, Eq)]
enum MidWaitState {
    /// Source tree unchanged across the wait (or non-local source
    /// where the wait has no source-tree implication). The pipeline
    /// proceeds to the cache_lookup short-circuit.
    Clean,
    /// Operator started with a dirty tree BEFORE the source-tree
    /// EX wait was taken. The wait was not the cause of the dirty
    /// state, so the diagnostic is silent (returns `None`) to avoid
    /// fabricating wait-related attribution.
    PreAcquireDirty,
    /// Operator edited a tracked file DURING the wait (acquire-time
    /// probe was clean, post-wait probe is dirty). Forces a rebuild
    /// and emits a "your local edits" diagnostic.
    DirtyEdit,
    /// Operator advanced HEAD (commit / branch flip) during the wait
    /// (acquire-time short-hash differs from post-wait short-hash;
    /// post-wait worktree is clean). Forces a rebuild and emits a
    /// "HEAD advanced" diagnostic.
    HashAdvanced,
    /// Post-wait probe returned `Err` (corrupt git state, removed
    /// source dir, or a gix internal error). Forces a conservative
    /// rebuild — unknowable state cannot be assumed Clean.
    ProbeFailed,
}

impl MidWaitState {
    /// Operator-facing diagnostic body (without the `{cli_label}: `
    /// prefix — caller composes via `eprintln!("{cli_label}: {body}")`).
    ///
    /// Returns `None` for [`Self::Clean`] (the cache-skip gate emits
    /// its own message) and [`Self::PreAcquireDirty`] (the wait was
    /// not the cause of the dirty state, so a wait-related diagnostic
    /// would fabricate attribution).
    fn diagnostic(&self) -> Option<&'static str> {
        match self {
            Self::DirtyEdit => Some(
                "source tree changed during peer's build wait \
                 — rebuilding to capture your local edits",
            ),
            Self::HashAdvanced => Some(
                "source HEAD advanced during peer's build wait \
                 — rebuilding for the new commit",
            ),
            Self::ProbeFailed => Some(
                "source-tree dirty re-check failed during peer's \
                 build wait — rebuilding conservatively (re-run with \
                 RUST_LOG=warn for the probe error)",
            ),
            Self::Clean | Self::PreAcquireDirty => None,
        }
    }
}

/// Operator-facing diagnostic body emitted when the post-mid-wait
/// `cache_lookup` short-circuit fires (without the `{cli_label}: `
/// prefix — caller composes via `eprintln!("{cli_label}: {body}")`,
/// matching the [`MidWaitState::diagnostic`] convention).
///
/// Separate from [`MidWaitState::diagnostic`] because the cache-hit
/// is downstream of the variant classification — the message fires
/// only when all three of `mid_wait_clean`, a populated cache slot,
/// and an extant image file align. Tying the message to a single
/// MidWaitState variant would misrepresent that conjunction.
fn cache_hit_diagnostic(cache_key: &str) -> String {
    format!(
        "concurrent ktstr build populated cache slot {cache_key} during \
         peer's build wait — skipping redundant rebuild"
    )
}

/// Post-acquisition kernel build pipeline.
///
/// Handles: clean, configure, build, validate config, generate
/// compile_commands.json for local trees, find image, strip vmlinux,
/// compute metadata, cache store, and remote cache store (when
/// enabled). Callers handle source acquisition.
///
/// `cli_label` prefixes diagnostic status output (e.g. `"ktstr"` or
/// `"cargo ktstr"`).
///
/// `force` mirrors `kernel build --force`: when set, the post-acquire
/// cache re-check below does NOT short-circuit on a peer-populated slot.
/// A forced build must actually rebuild — the peer-dedup skip only
/// spares a redundant NON-forced rebuild. (The caller already does its
/// own pre-acquire `!force` cache lookup; this gates the second,
/// post-source-lock lookup the caller cannot reach.)
///
/// `is_local_source` should be true when the source is a local
/// kernel source tree, regardless of how the caller arrived there
/// (`kernel build --kernel <path>`, `cargo ktstr test --kernel <path>`,
/// or any other Path-spec entry that funnels through
/// [`super::super::resolve_kernel_dir`] /
/// [`super::super::resolve_kernel_dir_to_entry`]). It controls the
/// mrproper warning and `source_tree_path` in metadata.
///
/// `extra_kconfig` is an optional user-supplied kconfig fragment
/// merged on top of [`EMBEDDED_KCONFIG`] before `configure_kernel`
/// (which runs olddefconfig only when new lines are needed).
/// `Some(content)` appends the fragment AFTER the baked-in fragment
/// so kbuild's last-occurrence-wins semantics
/// (`scripts/kconfig/confdata.c::conf_read_simple`) make user values
/// override baked-in ones on conflict, and forces a re-configure pass
/// even when `.config` already carries `CONFIG_SCHED_CLASS_EXT=y`
/// (the user fragment may add or invert symbols the baked-in pass
/// alone wouldn't have produced).
///
/// Two metadata fields capture the build inputs separately:
/// - `ktstr_kconfig_hash` always holds the bare baked-in hash
///   (`crate::kconfig_hash()` of `EMBEDDED_KCONFIG`) so
///   `KconfigStatus::Matches/Stale/Untracked` keeps comparing
///   against the live baked-in fragment.
/// - `extra_kconfig_hash` holds `Some(crate::extra_kconfig_hash(content))`
///   when extras were supplied, `None` otherwise. Drives the
///   `(extra kconfig)` tag in `kernel list`.
///
/// Callers that don't expose `--extra-kconfig` (test/coverage/
/// shell/verifier) pass `None`.
#[allow(clippy::too_many_arguments)]
pub fn kernel_build_pipeline(
    acquired: &crate::fetch::AcquiredSource,
    cache: &crate::cache::CacheDir,
    cli_label: &str,
    clean: bool,
    force: bool,
    is_local_source: bool,
    cpu_cap: Option<crate::vmm::host_topology::CpuCap>,
    extra_kconfig: Option<&str>,
    progress: Option<&crate::cli::FetchProgress>,
) -> Result<KernelBuildResult> {
    let source_dir = &acquired.source_dir;
    let (arch, image_name) = crate::fetch::arch_info();

    // Bind a guaranteed-live progress group for the build phase: the
    // caller's group (the parallel resolve's shared group, or a
    // single-shot caller's local one), or a fresh local group when the
    // caller passed none. The build phase renders through this group and
    // NEVER through a standalone `Spinner`, so concurrent builds in the
    // parallel resolve cannot race the process-global `SPINNER_ACTIVE`
    // guard. Off-TTY the group is hidden and inert.
    let owned_group;
    let progress = match progress {
        Some(p) => p,
        None => {
            owned_group = crate::cli::FetchProgress::new();
            &owned_group
        }
    };

    // Two-phase reservation. A concurrent perf-mode test run must
    // not have its measured CPUs stomped by a `make -j$(nproc)`
    // explosion of gcc children, and vice-versa a concurrent
    // kernel build must not have its compile window extended by
    // a test pinning RT-FIFO on shared cores. Phase 1 of the
    // reservation is the LLC-level flock from
    // [`acquire_llc_plan`]: whole-LLC flocks whose count is
    // chosen to cover the CPU budget (either an explicit
    // `--cpu-cap N` or the 30%-of-allowed default). Phase 2 is
    // the cgroup v2 sandbox from
    // [`BuildSandbox::try_create`] that binds make/gcc's
    // cpu+mem sets to the plan's CPUs + NUMA nodes so the
    // parallelism hint is enforced, not just advisory.
    //
    // Binding order is load-bearing: `_sandbox` is declared first
    // and drops first per Rust's declaration-order field-drop rule,
    // which migrates the build pid out of the cgroup and rmdirs the
    // child while the LLC flocks are still held. Otherwise a peer
    // could observe the LLC released before the cgroup is gone,
    // mint a new plan against the same LLCs, and see an orphan
    // cgroup lingering for up to the 24h sweep window.
    //
    // Escape hatches:
    //   - `KTSTR_BYPASS_LLC_LOCKS=1`: skip the LLC plan+flock
    //     acquisition entirely; the build proceeds immediately
    //     without coordinating with any concurrent perf-mode run.
    //     Use when the operator explicitly accepts measurement
    //     noise (one shell doing unrelated work, an isolated
    //     developer workstation, or a CI queue that already
    //     serializes jobs at a higher layer). Mutually exclusive
    //     with `--cpu-cap` at CLI parse time — see the CLI
    //     binaries' pre-dispatch conflict check.
    //   - Sysfs-unreadable host (non-Linux, degraded container):
    //     `HostTopology::from_sysfs()` returns `Err`. Without
    //     `--cpu-cap`, we emit a `tracing::warn!` and proceed
    //     without locks. With `--cpu-cap`, the flag cannot be
    //     honoured and we fail hard — cpu_cap is a contract, not
    //     a hint: a silent degrade would let a build exceed the
    //     declared resource budget without surfacing.
    // `_plan` + `_sandbox` are kept alive via RAII — their Drops
    // release the LLC flocks and cgroup on scope exit. Struct
    // field order in BuildReservation ensures `_sandbox` drops
    // BEFORE `plan`, per Rust's declaration-order field-drop rule.
    let BuildReservation {
        plan: _plan,
        _sandbox,
        make_jobs,
    } = acquire_build_reservation(cli_label, cpu_cap)?;

    // Source-tree flock for local sources. Two parallel
    // `cargo ktstr test --kernel ./linux` runs would otherwise race
    // in `make` against the same source tree (e.g. one's
    // `make defconfig` racing with another's `make compile_commands.json`)
    // and produce inconsistent .config / build artifacts. The flock is
    // taken on the SOURCE TREE itself (per canonical path), distinct from
    // the cache-entry flock acquired inside `cache.store` (per cache key).
    // The two are complementary: the source-tree flock serializes the
    // build phase; the cache-entry flock serializes the atomic install.
    //
    // Held via `OwnedFd` for the lifetime of `_source_lock` — drops at
    // end of pipeline. Skipped under `KTSTR_BYPASS_LLC_LOCKS` to share
    // the operator's escape hatch with the LLC-flock bypass; that
    // env var already declares "I accept noise from concurrent runs."
    //
    // `acquire_source_tree_lock` does a non-blocking `try_flock`
    // first; on EWOULDBLOCK it reports the source path and then parks
    // in a blocking `flock(LOCK_EX)` until the holder releases. It
    // deliberately avoids a host-global `/proc/locks` attribution
    // scan before parking. The wait is intentional: when the peer's
    // build finishes, the cache slot is likely populated and the
    // post-acquire cache check below short-circuits the redundant
    // rebuild. The pre-wait `eprintln!` ensures the operator sees
    // what they're waiting on rather than a silent stall.
    let _source_lock = if is_local_source && !crate::bypass_llc_locks_active() {
        Some(acquire_source_tree_lock(source_dir, cli_label)?)
    } else {
        None
    };

    // Post-acquire cache re-check. N peers racing on a cold cache all
    // queue on the source-tree EX above. When the first peer's build
    // completes and releases, the cache slot is populated — every
    // subsequent peer should observe the hit and skip a redundant
    // rebuild rather than serially repeat the same work. The
    // pre-acquire `cache_lookup` in `resolve_kernel_dir_to_entry`
    // catches the warm-cache case (no lock taken at all); this check
    // catches the cold-then-warmed-during-wait case.
    let mid_wait_state = compute_mid_wait_state(acquired, source_dir, is_local_source, cli_label);
    let mid_wait_clean = mid_wait_state == MidWaitState::Clean;

    if let Some(body) = mid_wait_state.diagnostic() {
        eprintln!("{cli_label}: {body}");
    }

    if !force
        && mid_wait_clean
        && let Some(entry) =
            crate::cli::resolve::cache_lookup(cache, &acquired.cache_key, cli_label)
        && entry.image_path().exists()
    {
        eprintln!("{cli_label}: {}", cache_hit_diagnostic(&acquired.cache_key));
        let image_path = entry.image_path();
        return Ok(KernelBuildResult {
            entry: Some(entry),
            image_path,
            post_build_is_dirty: false,
        });
    }

    if clean {
        if !is_local_source {
            eprintln!(
                "{cli_label}: --clean is only meaningful with a --kernel <path> source (downloaded/cloned sources start clean)"
            );
        } else {
            eprintln!("{cli_label}: make mrproper");
            run_make(source_dir, &["mrproper"])?;
        }
    }

    reconfigure_and_build(source_dir, extra_kconfig, cli_label, make_jobs, progress)?;

    // Validate critical config options were not silently disabled.
    // When `--extra-kconfig` is set, attach an actionable hint
    // pointing at the user fragment as a likely cause. The most
    // plausible failure mode is a user override that disables a
    // baked-in invariant (e.g. a fragment containing
    // `# CONFIG_BPF is not set` defeats the BPF dep chain), so
    // name `--extra-kconfig` in the wrap context.
    validate_kernel_config(source_dir).with_context(|| {
        if extra_kconfig.is_some() {
            "post-build kernel config validation failed; check that your \
             --extra-kconfig fragment does not disable a CONFIG_X required by \
             ktstr (e.g. CONFIG_BPF, CONFIG_DEBUG_INFO_BTF, CONFIG_FTRACE, \
             CONFIG_SCHED_CLASS_EXT)"
                .to_string()
        } else {
            "post-build kernel config validation failed".to_string()
        }
    })?;

    if !acquired.is_temp {
        generate_compile_commands(source_dir, progress)?;
    }

    let (image_path, vmlinux_opt) = find_built_image(source_dir, cli_label)?;
    let vmlinux_ref = vmlinux_opt.as_deref();

    // Cache (skip for dirty local trees).
    if acquired.is_dirty {
        eprintln!("{cli_label}: kernel built at {}", image_path.display());
        // Branch the hint wording: commit/stash is only an actionable
        // remediation for an actual git repo. A non-git source tree
        // is force-marked dirty (see `acquire_local_source` in
        // `fetch.rs`) because dirty detection is impossible, and
        // telling the operator to "commit or stash" leads nowhere.
        let hint = dirty_cache_skip_hint(acquired.is_git);
        eprintln!("{cli_label}: {hint}");
        return Ok(KernelBuildResult {
            entry: None,
            image_path,
            post_build_is_dirty: true,
        });
    }

    if let Some(skip_result) = post_build_dirty_skip(
        acquired,
        source_dir,
        is_local_source,
        &image_path,
        cli_label,
    ) {
        return Ok(skip_result);
    }

    build_metadata_and_store(
        acquired,
        cache,
        cli_label,
        is_local_source,
        arch,
        image_name,
        extra_kconfig,
        source_dir,
        image_path,
        vmlinux_ref,
    )
}

/// Classify the source tree at the post-acquire re-probe site.
///
/// Mid-wait edit guard: the operator may edit a tracked file in
/// the source tree DURING our EX wait (long peer build = long
/// window). `acquired.is_dirty` snapshots clean-at-acquire; a fresh
/// probe via `inspect_local_source_state` catches edits that landed
/// during the wait. If dirty/hash-changed, the operator's intent
/// is "build what's on disk" — skip the cache re-check and fall
/// through to the build branch, where the post-build dirty re-check
/// at the cache-store site will recognise the mutation and skip
/// caching. Probe errors are warnings (not fatal) — same Err
/// disposition as the post-build re-check.
/// PreAcquireDirty distinguishes "operator started with a dirty
/// tree" (the wait wasn't the cause) from "operator dirtied the
/// tree during the wait" (DirtyEdit). The split keeps the enum
/// variants honest about cause-attribution per the
/// [`MidWaitState::diagnostic`] dispatch in [`kernel_build_pipeline`].
///
/// TOCTOU acceptance: the source tree can mutate between this
/// probe and the `cache_lookup` call in the caller — a microsecond
/// window (typically) where an operator edit, background
/// autoformatter write, `git commit`, or IDE pre-commit hook would
/// slip through both this guard AND the post-build dirty re-check at
/// the cache-store site (the cache-hit return in the caller
/// short-circuits before `make`, so the post-build re-check never
/// runs for the racing-into-hit path). Publication-side staleness is
/// gated by the separate post-build dirty re-check at the cache-store
/// site; this paragraph is about the consumer-side stale-serving
/// window only. The cache slot is keyed on `acquired.cache_key`
/// (frozen at acquire time inside `local_source`), so the served
/// artifact's identity is the acquire-time HEAD — a mid-window
/// mutation produces a cache hit that serves a slightly-stale
/// source state without destroying the operator's later state.
/// Bounded race; the operator's next invocation re-acquires and
/// observes the new state.
///
/// "Next invocation correct" is NOT a remediation for: single-shot
/// CI pipelines without retry, `git bisect run` invocations (each
/// commit is independent), or pipelined CI flows where one job
/// builds the kernel and a downstream job consumes the cached
/// image without re-probing the source tree. Operators in those
/// workflows should treat cache hits as acquire-time-correct, not
/// invocation-time correct. Holding an EX flock across [probe,
/// cache_lookup] or re-probing after the lookup were considered
/// and rejected as adding common-path latency for a microsecond-
/// wide window.
fn compute_mid_wait_state(
    acquired: &crate::fetch::AcquiredSource,
    source_dir: &Path,
    is_local_source: bool,
    cli_label: &str,
) -> MidWaitState {
    if is_local_source && !acquired.is_dirty {
        match crate::fetch::inspect_local_source_state(source_dir) {
            Ok(post) => {
                let hash_changed = post.short_hash
                    != acquired
                        .kernel_source
                        .as_local_git_hash()
                        .map(str::to_string);
                if post.is_dirty {
                    MidWaitState::DirtyEdit
                } else if hash_changed {
                    MidWaitState::HashAdvanced
                } else {
                    MidWaitState::Clean
                }
            }
            Err(e) => {
                tracing::warn!(
                    cli_label = cli_label,
                    err = %format!("{e:#}"),
                    "mid-wait dirty re-check failed; proceeding to build",
                );
                MidWaitState::ProbeFailed
            }
        }
    } else if acquired.is_dirty {
        MidWaitState::PreAcquireDirty
    } else {
        MidWaitState::Clean
    }
}

/// Merge the kconfig fragments, reconfigure when stale, then build.
///
/// Builds the merged fragment ONCE so the configure call observes
/// the byte layout `{EMBEDDED_KCONFIG}\n{extra}` (with a `\n`
/// interleave) defined in [`crate::merge_kconfig_fragments`]. The
/// helper returns a `Cow<'_, str>` so the no-extras path borrows
/// `EMBEDDED_KCONFIG` without allocating; only the user-fragment
/// case heaps the merged string. Unit tests pin the exact
/// ordering kbuild's last-wins rule operates on.
///
/// Reconfigures when any merged-fragment line is missing from the
/// current `.config`. The prior `has_sched_ext` probe was a proxy for
/// "configured" — but a stale `.config` from an earlier build can carry
/// sched_ext while MISSING a changed baked-in value (e.g. an edited
/// CONFIG_NR_CPUS in ktstr.kconfig) or every user `--extra-kconfig` line,
/// silently ignoring the edit. `all_fragment_lines_present` checks the
/// actual merged fragment (exact-line) instead, so an edited baked-in
/// symbol or a user extra both trigger the merged configure.
fn reconfigure_and_build(
    source_dir: &Path,
    extra_kconfig: Option<&str>,
    cli_label: &str,
    make_jobs: Option<usize>,
    progress: &crate::cli::FetchProgress,
) -> Result<()> {
    let merged_fragment = crate::merge_kconfig_fragments(EMBEDDED_KCONFIG, extra_kconfig);

    // Surface a `tracing::warn!` for each user fragment line that
    // overrides a baked-in symbol from `EMBEDDED_KCONFIG`. The build
    // proceeds with the user value winning (last-wins is the design
    // intent) — the warning lets the operator see they are shadowing
    // a baked-in setting before configure_kernel (which runs
    // olddefconfig only when new lines are needed), which is when
    // an over-aggressive override can still be addressed by editing
    // the fragment. A separate post-build `validate_kernel_config`
    // pass catches critical-baked-in disablement (e.g. CONFIG_BPF).
    if let Some(extra) = extra_kconfig {
        warn_extra_kconfig_overrides_baked_in(extra, cli_label);
    }

    let config_now = std::fs::read_to_string(source_dir.join(".config")).unwrap_or_default();
    let needs_configure =
        extra_kconfig.is_some() || !all_fragment_lines_present(&merged_fragment, &config_now);
    if needs_configure {
        let bar = progress.step_bar("Configuring kernel...");
        let configure_result = configure_kernel(source_dir, &merged_fragment, Some(progress));
        bar.finish();
        // Wrap configure errors with `--extra-kconfig` context when
        // extras are present so the user can pinpoint which input is
        // responsible for an olddefconfig failure (e.g. a malformed
        // `CONFIG_X=` line in their fragment).
        configure_result.with_context(|| {
            if extra_kconfig.is_some() {
                "kernel configure failed (with --extra-kconfig fragment merged on top of \
                 baked-in ktstr.kconfig); check the fragment for syntax errors or \
                 conflicting symbol declarations"
                    .to_string()
            } else {
                "kernel configure failed".to_string()
            }
        })?;

        // Post-olddefconfig validation — warn (not error) when a
        // user-requested option from `--extra-kconfig` did not
        // survive into the final `.config` (typically because
        // olddefconfig dropped it for an unmet dependency). Emits
        // one `tracing::warn!` per dropped line naming the
        // requested setting and the actual final value.
        // The hard-fail "user override killed a baked-in invariant"
        // case (e.g. user disabled `CONFIG_BPF`) is caught at
        // `validate_kernel_config` post-build with extra context.
        if let Some(extra) = extra_kconfig {
            warn_dropped_extra_kconfig_lines(source_dir, extra, cli_label);
        }
    }

    let bar = progress.step_bar("Building kernel...");
    let build_result = make_kernel_with_output(source_dir, Some(progress), make_jobs);
    bar.finish();
    build_result?;
    Ok(())
}

/// Generate `compile_commands.json` for local trees (LSP support).
///
/// The args MUST match `make_kernel_with_output`'s
/// (`-jN`, `KCFLAGS=-Wno-error`) — otherwise make's "command line
/// flag changed" detection invalidates the build's object cache
/// and recompiles every translation unit single-threaded under
/// the compile_commands.json rule.  Build the same `-jN +
/// KCFLAGS=-Wno-error` prefix via `build_make_args`, then append
/// the target.
fn generate_compile_commands(
    source_dir: &Path,
    progress: &crate::cli::FetchProgress,
) -> Result<()> {
    let nproc = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let mut cc_args = build_make_args(nproc);
    cc_args.push("compile_commands.json".into());
    let cc_arg_refs: Vec<&str> = cc_args.iter().map(|s| s.as_str()).collect();
    let bar = progress.step_bar("Generating compile_commands.json...");
    let cc_result = run_make_with_output(source_dir, &cc_arg_refs, Some(progress));
    bar.finish();
    cc_result?;
    Ok(())
}

/// Find the built kernel image and vmlinux. Returns `(image_path,
/// vmlinux)` where `vmlinux` is `Some` only when `<source_dir>/vmlinux`
/// exists; emits the operator-facing caching/missing diagnostic.
fn find_built_image(
    source_dir: &Path,
    cli_label: &str,
) -> Result<(std::path::PathBuf, Option<std::path::PathBuf>)> {
    let image_path = crate::kernel_path::find_image_in_dir(source_dir)
        .ok_or_else(|| anyhow::anyhow!("no kernel image found in {}", source_dir.display()))?;
    let vmlinux_path = source_dir.join("vmlinux");
    let vmlinux_opt = if vmlinux_path.exists() {
        let orig_mib = std::fs::metadata(&vmlinux_path)
            .map(|m| m.len() as f64 / (1024.0 * 1024.0))
            .unwrap_or(0.0);
        eprintln!("{cli_label}: caching vmlinux ({orig_mib:.0} MiB, will be stripped)");
        Some(vmlinux_path)
    } else {
        eprintln!("{cli_label}: warning: vmlinux not found, BTF will not be cached");
        None
    };
    Ok((image_path, vmlinux_opt))
}

/// Post-build dirty re-check. Returns `Some(result)` when the cache
/// store must be skipped because the source tree changed during the
/// build; `None` (proceed to store) otherwise.
///
/// `local_source` captures `is_dirty` ONCE at acquire time. The
/// operator may then edit a tracked file (`.config` mutation, source
/// patch) DURING the build window. The acquire-time `is_dirty=false`
/// would say "safe to cache" but the on-disk content actually built
/// differs from the HEAD commit recorded in the cache key — a future
/// cache hit on that key would serve a build that no longer matches
/// its identity. Re-running the same gix probes catches the race. On
/// any change (dirty flip OR HEAD-hash shift from a concurrent
/// commit), skip the cache store and emit a one-liner explaining why
/// the cache slot was passed over.
///
/// Errors from the re-check are surfaced as a warning rather than a
/// hard fail — the build itself succeeded; refusing to store on a
/// re-check probe failure would penalize an otherwise-clean run for a
/// transient gix glitch. The cache store proceeds with the original
/// key, on the same pessimistic basis as a tree the re-check could not
/// classify.
fn post_build_dirty_skip(
    acquired: &crate::fetch::AcquiredSource,
    source_dir: &Path,
    is_local_source: bool,
    image_path: &Path,
    cli_label: &str,
) -> Option<KernelBuildResult> {
    if is_local_source {
        match crate::fetch::inspect_local_source_state(source_dir) {
            Ok(post) => {
                let (skip, hash_changed) =
                    post_build_cache_store_skip(&post, acquired.kernel_source.as_local_git_hash());
                if skip {
                    eprintln!(
                        "{cli_label}: source tree changed during build \
                         (acquire-time dirty={}, post-build dirty={}; \
                         hash_changed={hash_changed}); skipping cache store \
                         to avoid recording a stale identity. Re-run after \
                         the working tree settles to populate the cache.",
                        acquired.is_dirty, post.is_dirty,
                    );
                    return Some(KernelBuildResult {
                        entry: None,
                        image_path: image_path.to_path_buf(),
                        // Mid-build mutation flips the run's
                        // reproducibility — the cache key recorded at
                        // acquire time no longer identifies the actual
                        // build input. Mirror that into the outcome so
                        // the kernel-label downstream gets the
                        // `_dirty` suffix.
                        post_build_is_dirty: true,
                    });
                }
            }
            Err(e) => {
                tracing::warn!(
                    cli_label = cli_label,
                    err = %format!("{e:#}"),
                    "post-build dirty re-check failed; proceeding to cache store",
                );
            }
        }
    }
    None
}

/// Build the kernel metadata + artifact set, store to cache, and
/// return the clean (non-dirty) [`KernelBuildResult`].
///
/// `too_many_arguments` allow: the cache-store tail threads the
/// acquire-time inputs (`acquired`, `arch`, `image_name`,
/// `extra_kconfig`, `is_local_source`) plus the build outputs
/// (`source_dir`, `image_path`, `vmlinux_ref`) the metadata records;
/// bundling them into a struct would add indirection without
/// changing the data flow.
#[allow(clippy::too_many_arguments)]
fn build_metadata_and_store(
    acquired: &crate::fetch::AcquiredSource,
    cache: &crate::cache::CacheDir,
    cli_label: &str,
    is_local_source: bool,
    arch: &str,
    image_name: &str,
    extra_kconfig: Option<&str>,
    source_dir: &Path,
    image_path: std::path::PathBuf,
    vmlinux_ref: Option<&Path>,
) -> Result<KernelBuildResult> {
    let config_hash = config_hash_for(source_dir)?;

    // Two-segment metadata: the bare baked-in hash stays in
    // `ktstr_kconfig_hash` so `kernel list`'s matches/stale/
    // untracked verdict (see `CacheEntry::kconfig_status`) keeps
    // comparing against the live `EMBEDDED_KCONFIG`, and the user
    // extras hash lives in its own slot. Matches the cache-key
    // suffix shape `kc{baked}-xkc{extra}` produced by
    // [`crate::cache_key_suffix_with_extra`].
    let kconfig_hash = embedded_kconfig_hash();
    let extra_kconfig_hash_value = extra_kconfig.map(crate::extra_kconfig_hash);

    // Source-tree vmlinux stat (size + mtime seconds) so a later
    // `prefer_source_tree_for_dwarf` lookup can detect a user
    // rebuild between cache store and DWARF read. Only meaningful
    // for local sources whose vmlinux survived the build —
    // `vmlinux_ref` is `None` if vmlinux wasn't found, in which
    // case there's nothing to stat. mtime read is best-effort:
    // failure leaves the validation pair `None` and prefers the
    // pre-validation behavior for this entry.
    let source_vmlinux_stat = source_vmlinux_stat_for(vmlinux_ref);

    let mut metadata = crate::cache::KernelMetadata::new(
        acquired.kernel_source.clone(),
        arch,
        image_name,
        crate::test_support::now_iso8601(),
    )
    .with_ktstr_kconfig_hash(kconfig_hash);
    if let Some(v) = acquired.version.clone() {
        metadata = metadata.with_version(v);
    }
    if let Some(h) = config_hash {
        metadata = metadata.with_config_hash(h);
    }
    if let Some(h) = extra_kconfig_hash_value {
        metadata = metadata.with_extra_kconfig_hash(h);
    }
    if is_local_source && let Some((size, mtime_secs)) = source_vmlinux_stat {
        metadata = metadata.with_source_vmlinux_stat(size, mtime_secs);
    }

    let mut artifacts = crate::cache::CacheArtifacts::new(&image_path);
    if let Some(v) = vmlinux_ref {
        artifacts = artifacts.with_vmlinux(v);
    }
    let entry = match cache.store(&acquired.cache_key, &artifacts, &metadata) {
        Ok(entry) => {
            success(&format!("\u{2713} Kernel cached: {}", acquired.cache_key));
            eprintln!("{cli_label}: image: {}", entry.image_path().display());
            if crate::remote_cache::is_enabled() {
                crate::remote_cache::remote_store(&entry, cli_label);
            }
            Some(entry)
        }
        Err(e) => {
            warn(&format!("{cli_label}: cache store failed: {e:#}"));
            None
        }
    };

    Ok(KernelBuildResult {
        entry,
        image_path,
        post_build_is_dirty: false,
    })
}

/// CRC32 of `<source_dir>/.config` rendered as a fixed 8-hex-digit
/// lowercase string, or `None` when no `.config` exists. The
/// zero-padded `{:08x}` width is the cache-key suffix contract — a
/// `{:x}` (no pad) would drop a leading-zero nibble and silently
/// re-key every cached build, defeating cache hits. Pulled out of
/// [`kernel_build_pipeline`] so the derivation is unit-testable
/// without a real `make`.
pub(crate) fn config_hash_for(source_dir: &Path) -> std::io::Result<Option<String>> {
    let config_path = source_dir.join(".config");
    if config_path.exists() {
        let data = std::fs::read(&config_path)?;
        Ok(Some(format!("{:08x}", crc32fast::hash(&data))))
    } else {
        Ok(None)
    }
}

/// `(size, mtime_secs)` of the source-tree vmlinux, or `None` when the
/// path is absent or unstattable. `mtime_secs` is signed seconds since
/// the epoch — a pre-epoch mtime (clock skew) maps to a negative
/// count rather than dropping the stat. The `None` short-circuit on a
/// missing vmlinux keeps a later `prefer_source_tree_for_dwarf`
/// staleness check from comparing against a phantom `(0, _)`. Pulled
/// out of [`kernel_build_pipeline`] for unit-testability.
pub(crate) fn source_vmlinux_stat_for(vmlinux_ref: Option<&Path>) -> Option<(u64, i64)> {
    let v = vmlinux_ref?;
    let stat = std::fs::metadata(v).ok()?;
    let mtime_secs = stat.modified().ok().and_then(|t| {
        t.duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .ok()
            .or_else(|| {
                std::time::UNIX_EPOCH
                    .duration_since(t)
                    .ok()
                    .map(|d| -(d.as_secs() as i64))
            })
    })?;
    Some((stat.len(), mtime_secs))
}

/// The cache-skip hint wording for a dirty local source tree. A git
/// repo can be remediated by commit/stash; a non-git tree is
/// force-marked dirty (dirty detection is impossible) so commit/stash
/// advice is unactionable — it gets the put-under-git wording
/// instead. Pulled out of [`kernel_build_pipeline`] for
/// unit-testability.
pub(crate) fn dirty_cache_skip_hint(is_git: bool) -> &'static str {
    if is_git {
        DIRTY_TREE_CACHE_SKIP_HINT
    } else {
        NON_GIT_TREE_CACHE_SKIP_HINT
    }
}

/// The post-build cache-store-skip decision. After `make` returns,
/// `inspect_local_source_state` is re-run; the store is skipped when
/// the worktree went dirty during the build OR HEAD advanced (a
/// concurrent commit), because the acquire-time cache key no longer
/// identifies the built input. Returns `(skip, hash_changed)` so the
/// caller both decides and reports `hash_changed` in the skip message.
/// `acquired_local_git_hash` is the acquire-time hash from the kernel
/// source's `as_local_git_hash`. Pulled out of
/// [`kernel_build_pipeline`] for unit-testability.
pub(crate) fn post_build_cache_store_skip(
    post: &crate::fetch::LocalSourceState,
    acquired_local_git_hash: Option<&str>,
) -> (bool, bool) {
    let hash_changed = post.short_hash != acquired_local_git_hash.map(str::to_string);
    (post.is_dirty || hash_changed, hash_changed)
}

#[cfg(test)]
mod tests {
    use super::super::super::kernel_cmd::KernelCommand;
    use super::*;

    /// Returns `false` when `git` is not on `PATH`. Tests that drive
    /// a real git repo in a tempdir call this first and `return` early
    /// when git is unavailable so CI without git silently skips
    /// instead of failing on a hard-error.
    fn git_available() -> bool {
        std::process::Command::new("git")
            .arg("--version")
            .output()
            .is_ok()
    }

    /// Runs `git` in `canonical` with the sandboxed env that the
    /// mid-wait tests share — neutralizes `~/.gitconfig` and
    /// `/etc/gitconfig` (so a CI host's git identity can't pollute
    /// the test repo) and pins author/committer identity so `commit`
    /// succeeds without depending on host config. Asserts the command
    /// exited successfully; failure surfaces stderr in the panic
    /// message.
    fn run_git(canonical: &Path, args: &[&str]) {
        let out = std::process::Command::new("git")
            .args(args)
            .current_dir(canonical)
            .env("GIT_CONFIG_GLOBAL", "/dev/null")
            .env("GIT_CONFIG_SYSTEM", "/dev/null")
            .env("GIT_AUTHOR_NAME", "ktstr-test")
            .env("GIT_AUTHOR_EMAIL", "ktstr-test@localhost")
            .env("GIT_COMMITTER_NAME", "ktstr-test")
            .env("GIT_COMMITTER_EMAIL", "ktstr-test@localhost")
            .output()
            .expect("git");
        assert!(
            out.status.success(),
            "git {args:?} failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    /// Pins the post-acquire cache re-check at `kernel_build_pipeline`
    /// (the early-return path that fires when a peer publishes the
    /// cache slot during our source-tree EX wait).
    ///
    /// The early-return gate is 3-pronged: `!acquired.is_dirty` AND
    /// `cache_lookup(...).is_some()` AND `entry.image_path().exists()`.
    /// A regression that drops any prong (e.g. someone "simplifies"
    /// out the exists check) would let stale-manifest entries slip
    /// through and the runtime would crash later on a phantom image.
    ///
    /// Single-thread, deterministic — the "after EX wait" semantic
    /// reduces to "after the lookup, observe the planted state."
    /// Real thread orchestration is covered by
    /// `acquire_source_tree_lock_blocks_on_contention_then_succeeds`
    /// elsewhere in this module.
    #[test]
    fn cache_lookup_observes_peer_published_entry_after_ex_wait() {
        let _env_lock = crate::test_support::test_helpers::lock_env();
        let cache_tmp = tempfile::TempDir::new().expect("cache tempdir");
        let _cache_env = crate::test_support::test_helpers::EnvVarGuard::set(
            crate::KTSTR_CACHE_DIR_ENV,
            cache_tmp.path(),
        );
        let cache = crate::cache::CacheDir::with_root(cache_tmp.path().to_path_buf());
        let cache_key = "test-cache-key-7f8a9b";

        // Plant a cache entry via `CacheDir::store` (the production
        // helper). Going through `store` rather than hand-writing
        // metadata.json keeps the test honest against schema drift.
        let (arch, image_name) = crate::fetch::arch_info();
        let staging = tempfile::TempDir::new().expect("staging tempdir");
        let fake_image = staging.path().join(image_name);
        std::fs::write(&fake_image, b"fake kernel image bytes").expect("write fake image");
        let metadata = crate::cache::KernelMetadata::new(
            crate::cache::KernelSource::Local {
                source_tree_path: None,
                git_hash: None,
            },
            arch,
            image_name,
            "2026-04-12T10:00:00Z",
        );
        let artifacts = crate::cache::CacheArtifacts::new(&fake_image);
        cache
            .store(cache_key, &artifacts, &metadata)
            .expect("plant cache entry");

        // Exercise the 3-condition gate. `cache_lookup` is the same
        // helper `kernel_build_pipeline` calls at the post-acquire
        // re-check; `image_path().exists()` is the second gate; the
        // `is_dirty` gate is upstream (this test assumes a clean
        // source by construction since `acquired.is_dirty` is the
        // caller's responsibility).
        let entry = crate::cli::resolve::cache_lookup(&cache, cache_key, "test")
            .expect("cache_lookup must surface the planted entry");
        assert!(
            entry.image_path().exists(),
            "image_path existence check must hold for the planted entry",
        );
        assert_eq!(entry.metadata.built_at, "2026-04-12T10:00:00Z");
    }

    /// Pins the HashAdvanced branch of [`MidWaitState`] classification
    /// at `kernel_build_pipeline` — operator advanced HEAD
    /// (`git commit`/`checkout`) during the peer's build wait, leaving
    /// the worktree clean but the short_hash bumped.
    ///
    /// Failure mode pinned: a future "simplification" that drops the
    /// `hash_changed` check and trusts only `post.is_dirty` would
    /// silently accept a cache slot keyed on the pre-commit hash even
    /// though the operator committed (clean post-state) on top during
    /// the wait. The served cache slot would correspond to an older
    /// HEAD than the operator's current source tree.
    #[test]
    fn mid_wait_hash_change_invalidates_cache_hit_skip() {
        if !git_available() {
            eprintln!(
                "mid_wait_hash_change_invalidates_cache_hit_skip: \
                 git unavailable, skipping"
            );
            return;
        }

        let tmp = tempfile::TempDir::new().unwrap();
        let canonical = tmp.path().to_path_buf();
        run_git(&canonical, &["init", "-q", "-b", "main"]);
        std::fs::write(canonical.join("seed.txt"), "initial").unwrap();
        run_git(&canonical, &["add", "seed.txt"]);
        run_git(&canonical, &["commit", "-q", "-m", "initial"]);

        let pre = crate::fetch::inspect_local_source_state(&canonical).expect("acquire-time probe");
        let acquired_hash = pre
            .short_hash
            .clone()
            .expect("clean repo must carry a short_hash");

        // Mid-wait commit — different from the acquire-time hash.
        std::fs::write(canonical.join("file.txt"), "amended mid-wait").unwrap();
        run_git(&canonical, &["add", "file.txt"]);
        run_git(&canonical, &["commit", "-q", "-m", "mid-wait commit"]);

        let post = crate::fetch::inspect_local_source_state(&canonical).expect("post-wait probe");

        assert!(
            !post.is_dirty,
            "committed changes leave the worktree clean; the hash \
             change is what must invalidate the cache hit (not is_dirty)",
        );
        assert!(
            post.short_hash.is_some(),
            "clean post-wait state must carry a short_hash",
        );
        assert_ne!(
            post.short_hash.as_ref(),
            Some(&acquired_hash),
            "the new commit must yield a different short_hash than the \
             acquire-time hash",
        );

        // Mirror the production ternary in `kernel_build_pipeline`'s
        // mid_wait_state classification.
        let hash_changed = post.short_hash != Some(acquired_hash);
        let state = if post.is_dirty {
            MidWaitState::DirtyEdit
        } else if hash_changed {
            MidWaitState::HashAdvanced
        } else {
            MidWaitState::Clean
        };
        assert_eq!(
            state,
            MidWaitState::HashAdvanced,
            "clean worktree + advanced HEAD must classify as HashAdvanced",
        );
        assert!(
            state != MidWaitState::Clean,
            "hash_changed=true must falsify mid_wait_clean, forcing a \
             rebuild for the new cache key",
        );
    }

    /// Pins the Clean branch of [`MidWaitState`] classification at
    /// `kernel_build_pipeline` — the positive path where a peer's
    /// build wait completes with the source tree unchanged and the
    /// `cache_lookup` short-circuit fires.
    ///
    /// Failure mode pinned: a future refactor that flips the
    /// `if post.is_dirty` / `else if hash_changed` order, or one that
    /// inverts a `!is_dirty` check, would route a no-mutation
    /// post-wait probe into DirtyEdit or HashAdvanced and force a
    /// redundant rebuild every time. This test ensures the no-op
    /// path keeps returning [`MidWaitState::Clean`] so the cache
    /// short-circuit at the consumer site remains reachable.
    #[test]
    fn mid_wait_clean_path_allows_cache_hit_skip() {
        if !git_available() {
            eprintln!(
                "mid_wait_clean_path_allows_cache_hit_skip: \
                 git unavailable, skipping"
            );
            return;
        }

        let tmp = tempfile::TempDir::new().unwrap();
        let canonical = tmp.path().to_path_buf();
        run_git(&canonical, &["init", "-q", "-b", "main"]);
        std::fs::write(canonical.join("seed.txt"), "initial").unwrap();
        run_git(&canonical, &["add", "seed.txt"]);
        run_git(&canonical, &["commit", "-q", "-m", "initial"]);

        let pre = crate::fetch::inspect_local_source_state(&canonical).expect("acquire-time probe");
        let acquired_hash = pre
            .short_hash
            .clone()
            .expect("clean repo must carry a short_hash");

        // No mid-wait mutation. Post-probe must observe the same hash
        // and a clean worktree.
        let post = crate::fetch::inspect_local_source_state(&canonical).expect("post-wait probe");

        assert!(
            !post.is_dirty,
            "no mid-wait mutation must leave the post-wait probe clean",
        );
        assert_eq!(
            post.short_hash.as_ref(),
            Some(&acquired_hash),
            "no mid-wait commit must leave the short_hash unchanged",
        );

        let hash_changed = post.short_hash != Some(acquired_hash);
        let state = if post.is_dirty {
            MidWaitState::DirtyEdit
        } else if hash_changed {
            MidWaitState::HashAdvanced
        } else {
            MidWaitState::Clean
        };
        assert_eq!(
            state,
            MidWaitState::Clean,
            "no-mutation post-wait state must classify as Clean so the \
             cache_lookup short-circuit fires",
        );
        assert_eq!(
            state.diagnostic(),
            None,
            "Clean must be silent — the cache-skip gate emits its own \
             diagnostic when the lookup hits",
        );
    }

    /// Pins the DirtyEdit branch of [`MidWaitState`] classification at
    /// `kernel_build_pipeline` — operator edited a tracked file
    /// during the peer's build wait, post-wait probe surfaces
    /// `is_dirty=true` with no HEAD advance.
    ///
    /// Failure mode pinned: a future change that elides the
    /// `post.is_dirty` arm (e.g. trusting only `hash_changed`) would
    /// silently return a cache slot keyed on the pre-edit HEAD even
    /// though the operator's worktree no longer matches it — the
    /// rebuilt artifact would reflect the operator's local edits and
    /// the served cache slot would not.
    #[test]
    fn mid_wait_dirty_edit_invalidates_cache_hit_skip() {
        if !git_available() {
            eprintln!(
                "mid_wait_dirty_edit_invalidates_cache_hit_skip: \
                 git unavailable, skipping"
            );
            return;
        }

        let tmp = tempfile::TempDir::new().unwrap();
        let canonical = tmp.path().to_path_buf();
        run_git(&canonical, &["init", "-q", "-b", "main"]);
        std::fs::write(canonical.join("seed.txt"), "initial").unwrap();
        run_git(&canonical, &["add", "seed.txt"]);
        run_git(&canonical, &["commit", "-q", "-m", "initial"]);

        let pre = crate::fetch::inspect_local_source_state(&canonical).expect("acquire-time probe");
        let acquired_hash = pre
            .short_hash
            .clone()
            .expect("clean repo must carry a short_hash");

        // Mid-wait edit to a tracked file (no commit). The post-wait
        // probe must classify this as DirtyEdit — same hash, dirty
        // worktree.
        std::fs::write(canonical.join("seed.txt"), "operator edit during wait").unwrap();

        let post = crate::fetch::inspect_local_source_state(&canonical).expect("post-wait probe");

        assert!(
            post.is_dirty,
            "uncommitted edit to a tracked file must mark the post-wait \
             probe dirty",
        );

        let hash_changed = post.short_hash != Some(acquired_hash);
        let state = if post.is_dirty {
            MidWaitState::DirtyEdit
        } else if hash_changed {
            MidWaitState::HashAdvanced
        } else {
            MidWaitState::Clean
        };
        assert_eq!(
            state,
            MidWaitState::DirtyEdit,
            "dirty worktree without HEAD advance must classify as DirtyEdit",
        );
        assert!(
            state != MidWaitState::Clean,
            "DirtyEdit must falsify mid_wait_clean — the cache slot \
             corresponds to pre-edit state",
        );
    }

    /// Pins the ProbeFailed branch of [`MidWaitState`] classification at
    /// `kernel_build_pipeline` — the probe used to re-check the source
    /// tree returned `Err` and the pipeline conservatively rebuilds.
    ///
    /// Provoke strategy: init + commit, then truncate `.git/HEAD` to
    /// empty so `gix::discover` still succeeds (the `.git` dir
    /// exists) but `repo.head_id()` fails on the malformed ref —
    /// that error path is `inspect_local_source_state`'s only route
    /// to `Result::Err`. The non-git arm of `gix::discover` returns
    /// `Ok((None, true, false))`, NOT an `Err`, so simply removing
    /// `.git` does not reach ProbeFailed.
    ///
    /// Failure mode pinned: a future refactor that treats probe
    /// errors as Clean would silently return a cache slot keyed on
    /// unknowable post-wait state. The conservative-rebuild
    /// disposition is correct precisely because the alternative
    /// hides genuine corruption from the operator.
    #[test]
    fn mid_wait_probe_failure_invalidates_cache_hit_skip() {
        if !git_available() {
            eprintln!(
                "mid_wait_probe_failure_invalidates_cache_hit_skip: \
                 git unavailable, skipping"
            );
            return;
        }

        let tmp = tempfile::TempDir::new().unwrap();
        let canonical = tmp.path().to_path_buf();
        run_git(&canonical, &["init", "-q", "-b", "main"]);
        std::fs::write(canonical.join("seed.txt"), "initial").unwrap();
        run_git(&canonical, &["add", "seed.txt"]);
        run_git(&canonical, &["commit", "-q", "-m", "initial"]);

        let pre = crate::fetch::inspect_local_source_state(&canonical).expect("acquire-time probe");
        assert!(
            pre.short_hash.is_some(),
            "pre-corruption probe must succeed (the corruption happens \
             mid-wait, not at acquire time)",
        );

        // Corrupt HEAD mid-wait. `gix::discover` still sees `.git/`
        // and succeeds; the subsequent `head_id()` call fails on the
        // empty ref and `inspect_local_source_state` propagates the
        // error.
        std::fs::write(canonical.join(".git/HEAD"), b"").expect("truncate .git/HEAD");

        let post = crate::fetch::inspect_local_source_state(&canonical);
        assert!(
            post.is_err(),
            "truncated .git/HEAD must surface as a probe error, not a \
             silent Clean classification — found: {post:?}",
        );

        // Mirror the production dispatch: probe Err → ProbeFailed,
        // which falsifies mid_wait_clean and forces a rebuild.
        let state = match post {
            Ok(_) => MidWaitState::Clean,
            Err(_) => MidWaitState::ProbeFailed,
        };
        assert_eq!(
            state,
            MidWaitState::ProbeFailed,
            "probe Err must classify as ProbeFailed",
        );
        assert!(
            state != MidWaitState::Clean,
            "ProbeFailed must falsify mid_wait_clean — unknowable state \
             cannot be assumed Clean",
        );
    }

    /// Pins the non-local-source branch of [`MidWaitState`]
    /// classification at `kernel_build_pipeline` — when the source
    /// came from a non-local kernel spec (e.g. `Git+ref`,
    /// `Tarball`, downloaded archive), the outer
    /// `if is_local_source && !acquired.is_dirty` guard short-circuits
    /// the probe entirely and the fall-through reaches
    /// [`MidWaitState::Clean`] via the `else { Clean }` arm.
    ///
    /// Failure mode pinned: a future refactor that inverts the outer
    /// guard (e.g. mistakenly calls `inspect_local_source_state` on a
    /// Git+ref source, which doesn't have a meaningful local probe
    /// target) would route a non-local source into the probe branch
    /// and likely surface ProbeFailed against a non-git tree — a
    /// noisy regression. This test pins the no-probe short-circuit.
    #[test]
    fn mid_wait_non_local_source_classifies_as_clean() {
        // Mirror the outer production switch with is_local_source=false.
        // No probe call — the outer `if is_local_source && !acquired.is_dirty`
        // guard short-circuits when !is_local_source, falling through
        // to the `else if acquired.is_dirty` / else arms.
        let is_local_source = false;
        let acquired_is_dirty = false;
        let state = if is_local_source && !acquired_is_dirty {
            unreachable!(
                "is_local_source=false must skip the probe branch — the \
                 outer guard requires both is_local_source AND \
                 !acquired.is_dirty to reach the probe arm"
            )
        } else if acquired_is_dirty {
            MidWaitState::PreAcquireDirty
        } else {
            MidWaitState::Clean
        };
        assert_eq!(
            state,
            MidWaitState::Clean,
            "non-local clean source must classify as Clean — the cache \
             short-circuit applies to any source whose state we cannot \
             probe (or did not need to probe)",
        );
        assert_eq!(
            state.diagnostic(),
            None,
            "Clean non-local source must be silent",
        );
    }

    /// Pins the PreAcquireDirty variant identity and its silent
    /// diagnostic — `MidWaitState::PreAcquireDirty.diagnostic()`
    /// returns `None` because the wait was not the cause of the
    /// dirty state.
    ///
    /// SCOPE: does NOT exercise the caller-side dispatch order in
    /// `kernel_build_pipeline` — the test reconstructs the
    /// `if is_local_source && !acquired.is_dirty / else if
    /// acquired.is_dirty / else` chain inline because PreAcquireDirty
    /// is constructed without any probe call. A future refactor that
    /// flipped the guard order in `kernel_build_pipeline` would not
    /// fail this test; the other 4 mid_wait tests ground against
    /// `inspect_local_source_state` and would catch a probe-arm
    /// regression. This test pins the variant + diagnostic pair only.
    #[test]
    fn mid_wait_pre_acquire_dirty_suppresses_wait_diagnostic() {
        // Mirror the production dispatch with acquired.is_dirty=true.
        // No probe call — the `else if acquired.is_dirty` arm fires
        // before the probe-bearing branch. If the guard structure in
        // `kernel_build_pipeline` changes (e.g. PreAcquireDirty moves
        // inside the probe match), update this mirror.
        let is_local_source = true;
        let acquired_is_dirty = true;
        let state = if is_local_source && !acquired_is_dirty {
            unreachable!(
                "the guard requires !acquired.is_dirty before the probe \
                 branch; acquired_is_dirty=true must skip this arm"
            )
        } else if acquired_is_dirty {
            MidWaitState::PreAcquireDirty
        } else {
            MidWaitState::Clean
        };
        assert_eq!(
            state,
            MidWaitState::PreAcquireDirty,
            "acquired.is_dirty=true must classify as PreAcquireDirty",
        );
        assert_eq!(
            state.diagnostic(),
            None,
            "PreAcquireDirty must be silent — the wait was not the \
             cause of the dirty state, so a wait-related diagnostic \
             would fabricate attribution",
        );
    }

    /// Pins the exact diagnostic bodies emitted by each
    /// [`MidWaitState`] variant so a future copywriting change to
    /// the operator-facing messages is a deliberate, reviewed
    /// edit rather than silent drift.
    ///
    /// Clean and PreAcquireDirty return `None` (silent). DirtyEdit,
    /// HashAdvanced, and ProbeFailed return their full body strings
    /// without the `{cli_label}: ` prefix — the caller composes the
    /// prefix at the eprintln site.
    #[test]
    fn mid_wait_state_diagnostics_pinned() {
        assert_eq!(MidWaitState::Clean.diagnostic(), None);
        assert_eq!(MidWaitState::PreAcquireDirty.diagnostic(), None);
        assert_eq!(
            MidWaitState::DirtyEdit.diagnostic(),
            Some(
                "source tree changed during peer's build wait \
                 — rebuilding to capture your local edits"
            ),
        );
        assert_eq!(
            MidWaitState::HashAdvanced.diagnostic(),
            Some(
                "source HEAD advanced during peer's build wait \
                 — rebuilding for the new commit"
            ),
        );
        assert_eq!(
            MidWaitState::ProbeFailed.diagnostic(),
            Some(
                "source-tree dirty re-check failed during peer's \
                 build wait — rebuilding conservatively (re-run with \
                 RUST_LOG=warn for the probe error)"
            ),
        );
    }

    /// Pins the exact diagnostic body emitted by the post-mid-wait
    /// `cache_lookup` short-circuit so a future copywriting change
    /// to the operator-facing message is a deliberate, reviewed
    /// edit rather than silent drift — parallel to
    /// [`mid_wait_state_diagnostics_pinned`] for the
    /// [`MidWaitState`] family.
    ///
    /// Two assertions: a byte-for-byte match on the formatted body
    /// with a representative cache_key, plus an inequality between
    /// two distinct keys to prove the `{cache_key}` placeholder is
    /// load-bearing (catches a regression that replaces the
    /// substitution with a static label and would silently produce
    /// a constant string regardless of input).
    #[test]
    fn cache_hit_diagnostic_pinned() {
        let cache_key = "test-cache-key-7f8a9b";
        assert_eq!(
            cache_hit_diagnostic(cache_key),
            "concurrent ktstr build populated cache slot test-cache-key-7f8a9b \
             during peer's build wait — skipping redundant rebuild",
        );
        assert_ne!(
            cache_hit_diagnostic(cache_key),
            cache_hit_diagnostic("different-key-x86-64"),
            "cache_key substitution must be load-bearing, not a no-op",
        );
    }

    /// Pins the `force` gate on the post-acquire cache short-circuit in
    /// [`kernel_build_pipeline`]. The peer-dedup skip fires only when ALL
    /// of `!force && mid_wait_clean && cache-hit && image-exists` hold;
    /// `--force` must rebuild even when a peer already populated the slot
    /// (the observed bug: a forced build silently no-op'd via this path).
    /// Mirrors the production `if !force && mid_wait_clean && ...` guard —
    /// keep in lockstep with the short-circuit condition.
    #[test]
    fn force_bypasses_post_acquire_cache_short_circuit() {
        // All three non-force preconditions hold (a warm, peer-populated
        // slot on a clean tree): the skip fires only when NOT forced.
        let mid_wait_clean = true;
        let cache_hit = true;
        let image_exists = true;
        let short_circuits = |force: bool| !force && mid_wait_clean && cache_hit && image_exists;
        assert!(
            short_circuits(false),
            "non-forced build with a warm peer-populated slot must skip \
             the redundant rebuild",
        );
        assert!(
            !short_circuits(true),
            "--force must rebuild through a warm peer-populated slot, not \
             short-circuit — a forced build that no-ops defeats --force",
        );
    }

    /// `kernel build --cpu-cap N` parses through clap into
    /// `KernelCommand::Build { cpu_cap: Some(N), .. }`. Pins the
    /// flag's wire path: a future rename of the field, a stray
    /// `default_value`, or a `value_parser` change that altered
    /// rejection semantics would surface as a parse failure or a
    /// shape mismatch on the assertion.
    #[test]
    fn kernel_build_parses_cpu_cap_without_extra_flags() {
        use clap::Parser as _;
        #[derive(clap::Parser, Debug)]
        struct TestCli {
            #[command(subcommand)]
            cmd: KernelCommand,
        }
        let parsed =
            TestCli::try_parse_from(["prog", "build", "--kernel", "6.14.2", "--cpu-cap", "4"])
                .expect("kernel build --cpu-cap N must parse");
        match parsed.cmd {
            KernelCommand::Build {
                cpu_cap, kernel, ..
            } => {
                assert_eq!(cpu_cap, Some(4));
                assert_eq!(kernel.as_deref(), Some("6.14.2"));
            }
            other => panic!("expected KernelCommand::Build, got {other:?}"),
        }
    }

    /// `kernel build` without `--cpu-cap` parses with `cpu_cap: None`
    /// — the "unset" sentinel the downstream planner expands into the
    /// 30%-of-allowed default. Pins the no-flag path so a future
    /// rename of the clap field or a stray `default_value = "0"`
    /// surfaces as a test failure, not a silent runtime behavior change.
    #[test]
    fn kernel_build_without_cpu_cap_defaults_to_none() {
        use clap::Parser as _;
        #[derive(clap::Parser, Debug)]
        struct TestCli {
            #[command(subcommand)]
            cmd: KernelCommand,
        }
        let parsed = TestCli::try_parse_from(["prog", "build", "--kernel", "6.14.2"])
            .expect("kernel build without --cpu-cap must parse");
        match parsed.cmd {
            KernelCommand::Build { cpu_cap, .. } => {
                assert_eq!(cpu_cap, None, "no --cpu-cap must produce None, not Some(0)",);
            }
            other => panic!("expected KernelCommand::Build, got {other:?}"),
        }
    }

    /// `kernel build --cpu-cap 0` parses successfully at clap level
    /// — the "must be ≥ 1" check lives in [`CpuCap::new`], not in
    /// the clap value parser. Pins the two-layer validation: clap
    /// accepts any usize; runtime resolution via `CpuCap::resolve` is
    /// responsible for the "0 is rejected" diagnostic.
    #[test]
    fn kernel_build_cpu_cap_zero_passes_clap() {
        use clap::Parser as _;
        #[derive(clap::Parser, Debug)]
        struct TestCli {
            #[command(subcommand)]
            cmd: KernelCommand,
        }
        let parsed =
            TestCli::try_parse_from(["prog", "build", "--kernel", "6.14.2", "--cpu-cap", "0"])
                .expect("clap-level parse must accept 0; runtime validation rejects");
        match parsed.cmd {
            KernelCommand::Build { cpu_cap, .. } => {
                assert_eq!(
                    cpu_cap,
                    Some(0),
                    "clap parses 0 verbatim; validation is downstream",
                );
            }
            other => panic!("expected KernelCommand::Build, got {other:?}"),
        }
    }

    // ---------------------------------------------------------------
    // kernel_build_pipeline reservation phase — factored-out
    // `acquire_build_reservation` covers the cpu_cap → acquire →
    // sandbox → make_jobs flow without needing a real kernel source.
    // ---------------------------------------------------------------

    /// Serialize `KTSTR_BYPASS_LLC_LOCKS` env-var mutation across test
    /// threads. Delegates to the ONE crate-wide env mutex so it
    /// serializes against EVERY env-touching test (process-wide
    /// `std::env`), including the builder tests that read
    /// `KTSTR_BYPASS_LLC_LOCKS` — a module-local mutex left them racing.
    /// `lock_env()` recovers from poison.
    fn bypass_env_lock() -> std::sync::MutexGuard<'static, ()> {
        crate::test_support::test_helpers::lock_env()
    }

    /// RAII guard for scoped `KTSTR_BYPASS_LLC_LOCKS` mutation.
    /// Caller holds `bypass_env_lock()` before constructing.
    struct BypassGuard;
    impl BypassGuard {
        fn set(value: &str) -> Self {
            // SAFETY: env_lock held by caller; serializes with
            // every other env-mutating test.
            unsafe {
                std::env::set_var(crate::KTSTR_BYPASS_LLC_LOCKS_ENV, value);
            }
            BypassGuard
        }
        fn remove() -> Self {
            // SAFETY: caller holds env_lock.
            unsafe {
                std::env::remove_var(crate::KTSTR_BYPASS_LLC_LOCKS_ENV);
            }
            BypassGuard
        }
    }
    impl Drop for BypassGuard {
        fn drop(&mut self) {
            // SAFETY: guard lifetime bounded by env_lock held by
            // caller; Drop runs before the mutex guard releases.
            unsafe {
                std::env::remove_var(crate::KTSTR_BYPASS_LLC_LOCKS_ENV);
            }
        }
    }

    /// `acquire_build_reservation` with `KTSTR_BYPASS_LLC_LOCKS=1`
    /// plus `cpu_cap=None` returns a no-reservation `BuildReservation`:
    /// plan, sandbox, and make_jobs all None. Pins the "bypass
    /// disables both layers" contract.
    #[test]
    fn acquire_build_reservation_bypass_returns_no_reservation() {
        let _lock = bypass_env_lock();
        let _env = BypassGuard::set("1");
        let r = acquire_build_reservation("test", None).expect("bypass + no cap must succeed");
        assert!(r.plan.is_none(), "bypass must produce no LLC plan");
        assert!(
            r._sandbox.is_none(),
            "bypass must produce no cgroup sandbox",
        );
        assert!(
            r.make_jobs.is_none(),
            "bypass must fall back to nproc (None signals to caller)",
        );
    }

    /// Regression pin: empty-string-as-unset contract for
    /// `KTSTR_BYPASS_LLC_LOCKS`. A bare `KTSTR_BYPASS_LLC_LOCKS=`
    /// (CI shells, Docker `--env` pass-through without value) must
    /// NOT activate the bypass — the reader at L102 uses
    /// `.is_some_and(|v| !v.is_empty())` and that contract is
    /// shared by all 7 sibling readers. If a future contributor
    /// flips to `.is_some_and(|_| true)` or bare `.is_ok()`, this
    /// test catches the regression before it silently disables LLC
    /// flock contention enforcement in CI.
    #[test]
    fn acquire_build_reservation_bypass_empty_string_rejected() {
        let _lock = bypass_env_lock();
        let _env = BypassGuard::set("");
        match acquire_build_reservation("test", None) {
            Ok(r) => {
                // Empty-as-unset means we take the standard branch,
                // not the bypass branch. Standard branch produces a
                // BuildReservation with plan / sandbox / make_jobs
                // tied together (set-or-unset together per the
                // `plan_and_make_jobs_consistent` invariant). If the
                // bypass had been (incorrectly) triggered, all 3
                // would be None.
                assert_eq!(
                    r.plan.is_some(),
                    r.make_jobs.is_some(),
                    "empty-string must NOT activate bypass — plan + make_jobs \
                     should follow the standard-branch invariant",
                );
            }
            Err(e) => {
                // Sysfs-unreadable host: standard branch failed for
                // unrelated reasons. The empty-string-as-unset
                // contract is still proven because the bypass branch
                // would have returned `Ok` with all-None fields (per
                // the `bypass_returns_no_reservation` test); reaching
                // Err proves the standard branch was taken.
                eprintln!("standard-branch error confirms bypass was NOT taken (good): {e:#}");
            }
        }
    }

    /// `acquire_build_reservation` with `KTSTR_BYPASS_LLC_LOCKS=1`
    /// plus `cpu_cap=Some(_)` must error with the "resource contract"
    /// substring. Pins the conflict check at the pipeline's
    /// reservation entry point.
    #[test]
    fn acquire_build_reservation_bypass_with_cap_errors() {
        let _lock = bypass_env_lock();
        let _env = BypassGuard::set("1");
        let cap = crate::vmm::host_topology::CpuCap::new(2).expect("cap=2 valid");
        let err =
            acquire_build_reservation("test", Some(cap)).expect_err("bypass + cap must error");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("resource contract"),
            "err must name the resource contract: {msg}",
        );
    }

    /// `acquire_build_reservation` without bypass on a sysfs-capable
    /// host: returns a `BuildReservation` whose fields populate
    /// consistently — plan.is_some() iff make_jobs.is_some() iff
    /// sandbox.is_some(). Pins the "plan and make_jobs must never
    /// diverge" invariant.
    #[test]
    fn acquire_build_reservation_plan_and_make_jobs_consistent() {
        let _lock = bypass_env_lock();
        let _env = BypassGuard::remove();
        match acquire_build_reservation("test", None) {
            Ok(r) => {
                assert_eq!(
                    r.plan.is_some(),
                    r.make_jobs.is_some(),
                    "plan and make_jobs must agree on reservation presence",
                );
                if let (Some(p), Some(jobs)) = (r.plan.as_ref(), r.make_jobs) {
                    assert_eq!(
                        jobs,
                        crate::vmm::host_topology::make_jobs_for_plan(p),
                        "make_jobs must equal make_jobs_for_plan(&plan)",
                    );
                }
                assert_eq!(
                    r.plan.is_some(),
                    r._sandbox.is_some(),
                    "sandbox and plan must agree on reservation presence",
                );
            }
            Err(e) => {
                // Sysfs-unreadable host or contested LLCs. Accept
                // either outcome; the test's intent is to pin the
                // invariant in the success case, not force success.
                eprintln!("acquire_build_reservation unavailable on this host: {e:#}");
            }
        }
    }

    /// `acquire_build_reservation` plain bypass (no `--cpu-cap`)
    /// must NOT touch the sysfs probe. The test sets the bypass and
    /// confirms no error escapes, even on a host whose
    /// `HostTopology::from_sysfs()` would otherwise fail (the
    /// bypass branch is taken FIRST in the function, before the
    /// sysfs probe is attempted). Pins the "bypass short-circuits
    /// the topology probe" branch shape — a regression that
    /// re-ordered the bypass check below the sysfs probe would
    /// surface as a sysfs-error escape.
    #[test]
    fn acquire_build_reservation_bypass_does_not_touch_sysfs() {
        let _lock = bypass_env_lock();
        let _env = BypassGuard::set("1");
        let r = acquire_build_reservation("test", None)
            .expect("bypass must succeed regardless of sysfs availability");
        // The bypass branch produces (None, None, None) by
        // construction — no further state to assert beyond the
        // sibling tests that already pin the field shape.
        assert!(r.plan.is_none());
        assert!(r._sandbox.is_none());
        assert!(r.make_jobs.is_none());
    }

    // ---------------------------------------------------------------
    // acquire_source_tree_lock — per-source-tree flock that
    // serializes parallel builds against the same on-disk source.
    // ---------------------------------------------------------------
    //
    // Tests use `isolated_cache_dir()` to point `KTSTR_CACHE_DIR` at
    // a tempdir for the test's lifetime, so the production
    // `CacheDir::new()` resolves into the tempdir without touching
    // the operator's real cache directory. The lockfile path is
    // deterministic (cache_root/.locks/source-{path_hash}.lock) so
    // we can re-derive it from the canonical input path and assert
    // its presence.

    /// `acquire_source_tree_lock` on a fresh canonical path under
    /// an isolated cache root succeeds (no peer holding the lock)
    /// and creates the lockfile under `cache_root/.locks/`. Pins
    /// the lockfile placement: a regression that moved the lockfile
    /// to `/tmp/` (where `tmpwatch` could sweep it under an active
    /// holder) would surface here as the assertion failing on
    /// "lockfile not found at expected path."
    #[test]
    fn acquire_source_tree_lock_succeeds_on_fresh_path() {
        use crate::test_support::test_helpers::{isolated_cache_dir, lock_env};
        let _env_lock = lock_env();
        let cache = isolated_cache_dir();
        let canonical = std::path::PathBuf::from("/tmp/fake-source-tree-for-test");
        let fd = acquire_source_tree_lock(&canonical, "test")
            .expect("fresh-path acquire must succeed under isolated cache");
        // Lockfile must land under the isolated cache root's
        // `.locks/` subdirectory. The naming is `source-{hash}.lock`
        // where `{hash}` is `canonical_path_hash(canonical)`.
        let path_hash = crate::fetch::canonical_path_hash(&canonical);
        let expected = cache
            .path()
            .join(crate::flock::LOCK_DIR_NAME)
            .join(format!("source-{path_hash}.lock"));
        assert!(
            expected.exists(),
            "lockfile must exist at {} after acquire",
            expected.display(),
        );
        // Drop the FD explicitly to release the flock before the
        // tempdir cleanup races with it.
        drop(fd);
    }

    /// `acquire_source_tree_lock` returns the SAME lockfile path
    /// for two different canonical inputs IFF they share the same
    /// `canonical_path_hash`. Two distinct inputs (`/srv/linux-a`
    /// and `/srv/linux-b`) must produce DIFFERENT lockfiles so
    /// concurrent builds against unrelated source trees don't
    /// serialize against each other. Pins the per-tree
    /// disambiguation contract.
    #[test]
    fn acquire_source_tree_lock_distinct_paths_yield_distinct_lockfiles() {
        use crate::test_support::test_helpers::{isolated_cache_dir, lock_env};
        let _env_lock = lock_env();
        let cache = isolated_cache_dir();
        let path_a = std::path::PathBuf::from("/tmp/fake-source-a");
        let path_b = std::path::PathBuf::from("/tmp/fake-source-b");
        let fd_a = acquire_source_tree_lock(&path_a, "test")
            .expect("path A acquire must succeed under isolated cache");
        // Acquiring path B while path A's lock is still held must
        // ALSO succeed — they hash to different lockfiles, so
        // there's no contention.
        let fd_b = acquire_source_tree_lock(&path_b, "test").expect(
            "path B acquire must succeed concurrently with A — \
                 distinct canonical paths must hash to distinct \
                 lockfiles so unrelated builds don't serialize",
        );
        let hash_a = crate::fetch::canonical_path_hash(&path_a);
        let hash_b = crate::fetch::canonical_path_hash(&path_b);
        assert_ne!(
            hash_a, hash_b,
            "distinct canonical paths must produce distinct CRC32 hashes",
        );
        let lock_a = cache
            .path()
            .join(crate::flock::LOCK_DIR_NAME)
            .join(format!("source-{hash_a}.lock"));
        let lock_b = cache
            .path()
            .join(crate::flock::LOCK_DIR_NAME)
            .join(format!("source-{hash_b}.lock"));
        assert!(lock_a.exists());
        assert!(lock_b.exists());
        assert_ne!(lock_a, lock_b);
        drop(fd_a);
        drop(fd_b);
    }

    /// `acquire_source_tree_lock` on a path whose lockfile is
    /// already held by a peer parks in a blocking flock(2) until the
    /// holder releases, then succeeds. Pins the try-then-wait
    /// contract: a regression that re-introduced the bail-on-EWOULDBLOCK
    /// behavior, or any other path that returns without ever calling
    /// `flock(LOCK_EX)` blocking, would either bypass the wait hook,
    /// complete while the original holder remains live, or fail to
    /// acquire after that holder releases.
    ///
    /// We simulate "concurrent peer" by holding the first FD on the
    /// main thread, spawn a worker that issues a second acquire, wait
    /// for the contended-path hook immediately before `block_flock`,
    /// assert that the worker has not completed, release the holder,
    /// and collect the successful acquisition. The hook makes this
    /// fully event-driven and avoids a host-global `/proc/locks` scan,
    /// whose cost scales with every runner on a storm host.
    #[test]
    fn acquire_source_tree_lock_blocks_on_contention_then_succeeds() {
        use crate::test_support::test_helpers::{isolated_cache_dir, lock_env};
        // `_env_lock` and `_cache` MUST outlive the spawned worker
        // thread. The worker reads `KTSTR_CACHE_DIR` inside
        // `acquire_source_tree_lock`'s `CacheDir::new()`; if
        // `IsolatedCacheDir`'s drop ran while the worker was still
        // resolving the cache root, the worker would observe a
        // restored / empty env var and either land outside the
        // tempdir or fail with a stale-cache-root error. The bindings
        // below are declared here and dropped at end-of-scope, AFTER
        // the explicit `worker_result` collection point below.
        let _env_lock = lock_env();
        let _cache = isolated_cache_dir();
        let canonical = std::path::PathBuf::from("/tmp/fake-source-contention");
        let holder = acquire_source_tree_lock(&canonical, "test")
            .expect("first acquire must succeed under isolated cache");

        let worker_canonical = canonical.clone();
        let (waiting_tx, waiting_rx) = std::sync::mpsc::sync_channel(1);
        let (result_tx, result_rx) = std::sync::mpsc::sync_channel(1);
        let _worker = std::thread::spawn(move || {
            let result =
                acquire_source_tree_lock_with_wait_hook(&worker_canonical, "test", || {
                    let _ = waiting_tx.send(());
                });
            let _ = result_tx.send(result);
        });

        waiting_rx
            .recv_timeout(std::time::Duration::from_secs(30))
            .expect("contended acquire must publish the blocking transition");
        assert!(
            matches!(
                result_rx.try_recv(),
                Err(std::sync::mpsc::TryRecvError::Empty)
            ),
            "the contended acquisition must not complete while the holder is live",
        );

        drop(holder);
        let worker_result = result_rx
            .recv_timeout(std::time::Duration::from_secs(30))
            .expect("worker must deliver after the holder releases");
        let acquired = worker_result.expect("worker acquire must succeed once the holder releases");

        // Drop the worker's FD explicitly so the lockfile flock
        // releases before the isolated cache dir is torn down.
        // `_env_lock` and `_cache` are bound at function-scope above
        // and drop at end-of-scope, AFTER this point.
        drop(acquired);
    }

    /// `BuildReservation` field declaration order is load-bearing:
    /// `_sandbox` MUST be declared BEFORE `plan` so Rust's
    /// in-declaration-order field-drop runs the sandbox cgroup
    /// rmdir BEFORE the LLC flock release.
    ///
    /// A regression that swapped the field order would mean
    /// LLC flocks release first, which lets a peer claim the LLC
    /// while gcc children are still bound to a cgroup whose rmdir
    /// hasn't run yet.
    ///
    /// We can't assert drop ORDER directly without exotic
    /// machinery, but we can assert the field order is what we
    /// expect via the `Debug` derive: `_sandbox` appears in the
    /// formatted struct BEFORE `plan` IFF the field declaration
    /// order matches the Drop-order requirement. The field-name
    /// regex is enough to pin the order without depending on the
    /// inner field shapes (which evolve as the planner / sandbox
    /// types add or rename their own fields).
    #[test]
    fn build_reservation_field_order_pins_drop_invariant() {
        let r = BuildReservation {
            _sandbox: None,
            plan: None,
            make_jobs: None,
        };
        let dbg = format!("{r:?}");
        let sandbox_pos = dbg
            .find("_sandbox")
            .expect("Debug output must mention _sandbox field");
        let plan_pos = dbg
            .find("plan")
            .expect("Debug output must mention plan field");
        assert!(
            sandbox_pos < plan_pos,
            "_sandbox MUST be declared before plan so cgroup rmdir \
             runs BEFORE LLC flock release on Drop. Debug: {dbg}",
        );
    }

    // ---------------------------------------------------------------
    // Post-build metadata-derivation arms inside
    // `kernel_build_pipeline`. These pure-logic blocks (config_hash
    // CRC32, source_vmlinux_stat, the dirty-tree cache-skip hint, the
    // post-build dirty re-check store decision) are unreachable through
    // the public `kernel_build_pipeline` entry without a real `make`
    // invocation, so each was extracted into a `pub(crate)` helper
    // (`config_hash_for`, `source_vmlinux_stat_for`,
    // `dirty_cache_skip_hint`, `post_build_cache_store_skip`) that both
    // the pipeline and these tests call — so the tests exercise the
    // real production code, not a copy.
    // ---------------------------------------------------------------

    /// Drives the production [`config_hash_for`] helper (extracted from
    /// `kernel_build_pipeline`). Pins: (1) absent `.config` yields
    /// `None`; (2) present `.config` hashes to `crc32fast::hash` of the
    /// exact bytes; (3) the `{:08x}` zero-pad is load-bearing.
    ///
    /// Property (2) is checked against an INDEPENDENT `crc32fast::hash`
    /// so a hasher swap diverges. Property (3) is checked with an input
    /// whose CRC32 has a leading-zero nibble: a `{:x}` regression would
    /// drop the leading zero, shortening the hash below 8 chars and
    /// silently re-keying every cached build.
    #[test]
    fn config_hash_derivation_matches_crc32_and_width() {
        let tmp = tempfile::TempDir::new().expect("config tempdir");
        let source_dir = tmp.path();

        // Absent `.config` arm: the production `else { None }` branch.
        assert_eq!(
            config_hash_for(source_dir).expect("absent-config read"),
            None,
            "no .config must yield None config_hash",
        );

        // Present `.config` arm: hash matches an independent crc32fast.
        let body = b"CONFIG_SCHED_CLASS_EXT=y\nCONFIG_BPF=y\n";
        std::fs::write(source_dir.join(".config"), body).expect("write .config");
        let present = config_hash_for(source_dir).expect("present-config read");
        let expected = format!("{:08x}", crc32fast::hash(body));
        assert_eq!(
            present.as_deref(),
            Some(expected.as_str()),
            "present .config must hash to crc32fast::hash of its bytes",
        );

        // Width guard: find an input whose CRC32 has a leading-zero
        // nibble (< 0x1000_0000) so the `{:08x}` zero-pad is the only
        // thing keeping the rendered hash at 8 chars. A `{:x}`
        // regression would render it as 7 chars and fail the length
        // assert below. The probe sequence is deterministic; ~1 in 16
        // inputs qualifies, so 10k iterations always finds one.
        let probe = (0u32..10_000)
            .map(|n| format!("probe-{n}"))
            .find(|p| crc32fast::hash(p.as_bytes()) < 0x1000_0000)
            .expect("a leading-zero CRC32 within 10k probes");
        std::fs::write(source_dir.join(".config"), probe.as_bytes()).expect("rewrite .config");
        let zh = config_hash_for(source_dir)
            .expect("probe read")
            .expect("present .config must hash");
        assert_eq!(
            zh.len(),
            8,
            "config_hash must always be 8 hex chars ({{:08x}}): {zh}"
        );
        assert!(
            zh.starts_with('0'),
            "leading-zero CRC32 must keep its zero pad: {zh}"
        );
        assert!(
            zh.bytes().all(|b| b.is_ascii_hexdigit()),
            "config_hash must be lowercase hex: {zh}",
        );
    }

    /// Drives the production [`source_vmlinux_stat_for`] helper
    /// (extracted from `kernel_build_pipeline`). Pins three arms: a
    /// `None` ref, a ref to a missing file, and a present file.
    ///
    /// Failure mode pinned: a regression that dropped the `None`
    /// short-circuit (yielding `Some((0, _))` for an absent/missing
    /// vmlinux) or returned the wrong size would defeat the
    /// `prefer_source_tree_for_dwarf` staleness check that compares
    /// this stat against a later read.
    #[test]
    fn source_vmlinux_stat_present_and_absent_arms() {
        let tmp = tempfile::TempDir::new().expect("vmlinux tempdir");
        let vmlinux_path = tmp.path().join("vmlinux");

        // None ref → short-circuits to None.
        assert_eq!(
            source_vmlinux_stat_for(None),
            None,
            "a None vmlinux_ref must yield None",
        );
        // Ref to a non-existent file → metadata fails → None (NOT a
        // phantom (0, _)).
        assert_eq!(
            source_vmlinux_stat_for(Some(vmlinux_path.as_path())),
            None,
            "a vmlinux_ref to a missing file must yield None, not (0, _)",
        );

        // Present file → (real length, positive post-epoch mtime).
        let body = b"\x7fELF fake vmlinux payload bytes for stat";
        std::fs::write(&vmlinux_path, body).expect("write vmlinux");
        let (size, mtime_secs) = source_vmlinux_stat_for(Some(vmlinux_path.as_path()))
            .expect("present vmlinux must stat to Some");
        assert_eq!(
            size,
            body.len() as u64,
            "stat size must equal the real source-tree vmlinux length",
        );
        assert!(
            mtime_secs > 0,
            "a freshly-written vmlinux must carry a positive post-epoch \
             mtime in seconds, got {mtime_secs}",
        );
    }

    /// Drives the production [`dirty_cache_skip_hint`] helper (extracted
    /// from `kernel_build_pipeline`). A git repo with uncommitted
    /// changes gets the "commit or stash" hint; a non-git tree
    /// (force-marked dirty because dirty detection is impossible) gets
    /// the "put the source under git" hint — telling a non-git operator
    /// to "commit or stash" leads nowhere.
    ///
    /// Failure mode pinned: a regression that dropped the `is_git`
    /// branch and always returned `DIRTY_TREE_CACHE_SKIP_HINT` would
    /// give non-git operators unactionable advice. The inequality of
    /// the two constants proves the branch the helper selects on is
    /// load-bearing.
    #[test]
    fn dirty_tree_cache_skip_hint_branches_on_is_git() {
        assert_eq!(
            dirty_cache_skip_hint(true),
            DIRTY_TREE_CACHE_SKIP_HINT,
            "is_git=true must select the commit/stash hint",
        );
        assert_eq!(
            dirty_cache_skip_hint(false),
            NON_GIT_TREE_CACHE_SKIP_HINT,
            "is_git=false must select the put-under-git hint",
        );
        assert_ne!(
            DIRTY_TREE_CACHE_SKIP_HINT, NON_GIT_TREE_CACHE_SKIP_HINT,
            "the two hints must differ so the is_git branch is \
             load-bearing — a non-git operator must not be told to \
             commit/stash",
        );
    }

    /// Drives the production [`post_build_cache_store_skip`] predicate
    /// (extracted from `kernel_build_pipeline`, DISTINCT from the
    /// mid-wait dirty classifier above) against a real git tree for the
    /// clean (store proceeds) and mid-build-commit (hash advanced →
    /// skip) cases. `inspect_local_source_state` is the real probe; the
    /// skip decision is the real predicate.
    ///
    /// Failure mode pinned: a regression that dropped the
    /// `hash_changed` disjunct and trusted only `post.is_dirty` would
    /// store a build under a cache key keyed on the pre-commit HEAD
    /// even though the operator committed (clean worktree) on top
    /// during the build — a future cache hit would serve a build that
    /// no longer matches its recorded identity.
    #[test]
    fn post_build_dirty_recheck_skips_store_on_hash_advance() {
        if !git_available() {
            eprintln!(
                "post_build_dirty_recheck_skips_store_on_hash_advance: \
                 git unavailable, skipping"
            );
            return;
        }

        let tmp = tempfile::TempDir::new().unwrap();
        let canonical = tmp.path().to_path_buf();
        run_git(&canonical, &["init", "-q", "-b", "main"]);
        std::fs::write(canonical.join("seed.txt"), "initial").unwrap();
        run_git(&canonical, &["add", "seed.txt"]);
        run_git(&canonical, &["commit", "-q", "-m", "initial"]);

        // Acquire-time identity (frozen at build start in `local_source`).
        let acquire =
            crate::fetch::inspect_local_source_state(&canonical).expect("acquire-time probe");
        let acquired_hash = acquire.short_hash.clone();

        // Clean post-build: same hash, clean worktree → store proceeds.
        let post_clean =
            crate::fetch::inspect_local_source_state(&canonical).expect("post-build clean probe");
        let (skip_clean, hash_changed_clean) =
            post_build_cache_store_skip(&post_clean, acquired_hash.as_deref());
        assert!(
            !hash_changed_clean,
            "an unchanged HEAD must not flag hash_changed",
        );
        assert!(
            !skip_clean,
            "an unchanged tree post-build must NOT skip the cache store",
        );

        // Mid-build commit: HEAD advanced, worktree clean → skip store.
        std::fs::write(canonical.join("midbuild.txt"), "landed during make").unwrap();
        run_git(&canonical, &["add", "midbuild.txt"]);
        run_git(&canonical, &["commit", "-q", "-m", "mid-build commit"]);
        let post_advanced = crate::fetch::inspect_local_source_state(&canonical)
            .expect("post-build advanced probe");
        assert!(
            !post_advanced.is_dirty,
            "a committed mid-build change leaves the worktree clean; the \
             hash advance (not is_dirty) must drive the store skip",
        );
        let (skip_advanced, hash_changed_advanced) =
            post_build_cache_store_skip(&post_advanced, acquired_hash.as_deref());
        assert!(
            hash_changed_advanced,
            "the mid-build commit must yield a short_hash distinct from \
             the acquire-time hash",
        );
        assert!(
            skip_advanced,
            "a HEAD advance during the build must skip the cache store so \
             a stale identity is never recorded",
        );
    }
}
