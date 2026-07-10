//! Scheduler-binary resolution: maps a `SchedulerSpec` to a path plus a
//! `ResolveSource` provenance (the discovery cascade, PATH lookup,
//! staged-scheduler ordering) and dedups include-file lists. Split out
//! of eval/mod.rs to keep the module under the size ceiling.

use super::*;

/// Dedupe a resolved include-file list produced by unioning the
/// per-payload `include_files` specs through
/// [`crate::cli::resolve_include_files`] and appending the scheduler
/// config file entry. Each input tuple carries an `origin` label
/// (e.g. `"declarative"`, `"scheduler config_file"`) that is
/// surfaced in conflict diagnostics so the operator can trace which
/// declaration contributed each side of a collision.
///
/// Policy:
///
/// - Identical `(archive_path, host_path)` pairs collapse silently
///   (the same host file declared twice is harmless). Comparison
///   uses [`Path::canonicalize`] so two spellings of the same real
///   file (e.g. `./fio` vs `/usr/bin/fio` when `./fio` is a
///   symlink) are treated as equal. Canonicalization failure
///   (missing path, permission denied) falls back to byte-for-byte
///   PathBuf comparison; literal duplicates still collapse, and a
///   genuine conflict still surfaces.
/// - Two entries sharing an `archive_path` but resolving to
///   different canonical `host_path`s are a genuine ambiguity — a
///   scheduler's and a payload's `include_files` both claiming
///   `include-files/config.json` but pointing at different host
///   paths means one of the two would silently overwrite the other
///   in the initramfs. Bail with a diagnostic naming both host
///   paths AND their origin labels so the author can rename one
///   archive slot.
///
/// Case-sensitivity: `archive_path` keys are compared
/// byte-for-byte (via `BTreeMap<String, _>`), so on a case-
/// insensitive host filesystem (macOS HFS+, NTFS with the
/// `case-insensitive` mount flag) two archive paths spelled
/// `include-files/Helper` and `include-files/helper` are treated
/// as distinct here even though the host filesystem would
/// conflate them. This is intentional: `archive_path` is the
/// path inside the guest initramfs, which is tmpfs / ext4-
/// equivalent (always case-sensitive), so the guest-side
/// identity is what governs.
///
/// Order is stabilized via `BTreeMap`'s sorted iteration so the
/// emitted slice is deterministic regardless of which caller
/// appended first. Extracted from `run_ktstr_test_inner` so the
/// policy can be unit-tested without constructing a whole
/// KtstrTestEntry + VmBuilder.
pub(crate) fn dedupe_include_files(
    resolved: &[(String, std::path::PathBuf, &'static str)],
) -> Result<Vec<(String, std::path::PathBuf)>> {
    let mut seen: std::collections::BTreeMap<String, (std::path::PathBuf, &'static str)> =
        std::collections::BTreeMap::new();
    for (archive, host, origin) in resolved {
        if let Some((existing, existing_origin)) = seen.get(archive) {
            // Canonicalize both sides before comparing so
            // symlink-equivalent spellings collapse. A failed
            // canonicalize (missing path, permission denied) falls
            // back to the uncanonicalized value so the structural
            // compare still runs — literal duplicates still collapse
            // and genuine conflicts still surface.
            let existing_canon = existing.canonicalize().unwrap_or_else(|_| existing.clone());
            let host_canon = host.canonicalize().unwrap_or_else(|_| host.clone());
            if existing_canon != host_canon {
                anyhow::bail!(
                    "include_files conflict for archive path '{archive}': sources disagree \
                     on host path ({} [origin: {existing_origin}] vs {} [origin: {origin}]). \
                     Remove the duplicate declaration or rename one of the archive entries.",
                    existing.display(),
                    host.display(),
                );
            }
        } else {
            seen.insert(archive.clone(), (host.clone(), origin));
        }
    }
    Ok(seen
        .into_iter()
        .map(|(archive, (host, _origin))| (archive, host))
        .collect())
}

/// Provenance of a scheduler binary returned by [`resolve_scheduler`].
///
/// Each variant identifies the discovery branch that produced the
/// path, so downstream tooling (sidecar, cache-key construction, log
/// lines) can distinguish "we found a pre-built binary in a target
/// directory whose git hash we don't control" from "we just built
/// this binary from HEAD in the current workspace and therefore know
/// its source commit is the workspace HEAD."
///
/// Only the [`AutoBuilt`](Self::AutoBuilt) variant carries an honest
/// source-commit guarantee: every other branch locates an *existing*
/// file whose provenance is outside this process's knowledge.
/// Callers that need to stamp a sidecar with a scheduler-specific
/// commit must discard the hash for every non-`AutoBuilt` resolution
/// — a stale `target/debug/` binary looks identical to a fresh
/// `AutoBuilt` one but can be arbitrarily old.
///
/// `Eevdf` / `KernelBuiltin` / `Path` resolutions do not go through
/// the discovery cascade:
/// - `Eevdf` / `KernelBuiltin` → [`NotFound`](Self::NotFound) (no
///   user-space binary involved; the tuple's `Option<PathBuf>` is
///   `None`).
/// - `Path(p)` → [`Path`](Self::Path) (the caller named the binary
///   explicitly in the test entry — no env-var or filesystem search
///   runs).
///
/// The variant ordering in the enum mirrors the discovery cascade
/// order in [`resolve_scheduler`] so a reviewer can scan both lists
/// in lockstep.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolveSource {
    /// Resolved via the literal path the caller supplied as
    /// `SchedulerSpec::Path(p)`. No env-var or filesystem search
    /// involved — the path arrived in the test entry directly.
    /// Trusted to the extent the caller trusts the argument; git-
    /// hash provenance is UNKNOWN to this process.
    Path,
    /// Resolved via the `KTSTR_SCHEDULER` environment variable on the
    /// `SchedulerSpec::Discover` arm. Trusted to the extent the
    /// caller trusts the variable; git-hash provenance is UNKNOWN
    /// to this process.
    EnvVar,
    /// Resolved via a `$PATH` lookup. Only produced when
    /// `KTSTR_CARGO_TEST_MODE` is active and a binary by the
    /// requested name was found on the user's `$PATH` in front of
    /// the sibling-dir / target-dir cascade. Git-hash provenance
    /// UNKNOWN — the binary on PATH may be a system-wide install,
    /// a prior build, or a custom one the user staged for this run.
    PathLookup,
    /// Resolved via a sibling of `crate::resolve_current_exe`
    /// (same directory, or the sibling of a `deps/` directory for
    /// integration tests / nextest). Git-hash provenance UNKNOWN
    /// — the binary may be from any previous build.
    SiblingDir,
    /// Resolved via a fallback search in `target/debug/`. Git-hash
    /// provenance UNKNOWN — a stale binary from an older tree
    /// passes this check identically to a fresh one.
    TargetDebug,
    /// Resolved via a fallback search in `target/release/`. Git-hash
    /// provenance UNKNOWN — same stale-binary hazard as
    /// [`TargetDebug`](Self::TargetDebug).
    TargetRelease,
    /// Built on demand by [`crate::build_and_find_binary`] inside this
    /// process. The build targets the current workspace's HEAD by
    /// construction — the ONLY variant where the source commit is
    /// known to match the workspace tree the tests run from.
    AutoBuilt,
    /// No user-space binary path was produced. Returned for
    /// `SchedulerSpec::Eevdf` and `SchedulerSpec::KernelBuiltin` (the
    /// kernel supplies the scheduler — no binary to locate). The
    /// tuple's `Option<PathBuf>` is always `None` for this variant.
    NotFound,
}

impl ResolveSource {
    /// Stable snake_case tag for the sidecar `resolve_source` field and
    /// the `stats` `--resolve-source` filter — the string analog of the
    /// variant, mirroring the `run_source` tag convention so the
    /// persisted JSON shape does not depend on this enum's Rust
    /// representation. Variant order matches the discovery cascade.
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Path => "path",
            Self::EnvVar => "env_var",
            Self::PathLookup => "path_lookup",
            Self::SiblingDir => "sibling_dir",
            Self::TargetDebug => "target_debug",
            Self::TargetRelease => "target_release",
            Self::AutoBuilt => "auto_built",
            Self::NotFound => "not_found",
        }
    }
}

/// Walk `$PATH` directories in order looking for an executable
/// named `name`. Returns the first match that is a regular file
/// with at least one execute permission bit set. None when `PATH`
/// is unset, empty, or contains no matching executable.
///
/// Mirrors the semantics of `which(1)` and the
/// `crate::export::search_path_for` helper without pulling in a
/// new crate dependency. Used by [`resolve_scheduler`] only when
/// `KTSTR_CARGO_TEST_MODE` is active so the existing nextest /
/// `cargo ktstr test` discovery cascade stays in front of any
/// system-wide install on PATH for the production test path.
fn find_on_path(name: &str) -> Option<PathBuf> {
    use std::os::unix::fs::PermissionsExt;
    let path_var = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        let candidate = dir.join(name);
        if !candidate.is_file() {
            continue;
        }
        let executable = candidate
            .metadata()
            .map(|m| m.permissions().mode() & 0o111 != 0)
            .unwrap_or(false);
        if executable {
            return Some(candidate);
        }
    }
    None
}

/// Resolve every entry in `entry.staged_schedulers` via a caller-
/// supplied resolver, propagating resolver errors strictly (suitable
/// for the primary-dispatch path where a missing staged binary is a
/// hard failure operator should see at dispatch time, not later at
/// Op-dispatch inside the VM). KernelBuiltin / Eevdf staged entries
/// — whose resolver returns `Ok(None)` — are silently dropped:
/// they have no binary to stage and the lifecycle ops resolve them
/// via shell-script slots instead.
///
/// Returns `(name, resolved_host_path, sched_args)` tuples in the
/// SAME order as `entry.staged_schedulers` iteration. Ordering is
/// load-bearing: the initramfs packer iterates the result
/// to emit per-scheduler `/staging/schedulers/<name>/` archive
/// entries, and parent-directory dependencies are encounter-order
/// sensitive. Tests pin the order-preservation against a future
/// refactor that uses `.collect::<HashMap<_,_>>().into_iter()`
/// (would silently scramble).
///
/// `resolver` is a closure rather than a direct call to
/// [`resolve_scheduler`] so unit tests can drive the order-
/// preservation contract with a synthetic resolver that returns
/// known paths without touching the host filesystem. It receives the
/// whole staged [`Scheduler`](crate::test_support::Scheduler) so the
/// production closure can forward `manifest_dir` to `resolve_scheduler`.
pub(crate) fn resolve_staged_schedulers_strict<F>(
    entry: &KtstrTestEntry,
    mut resolver: F,
) -> Result<Vec<(String, PathBuf, Vec<String>)>>
where
    F: FnMut(&crate::test_support::Scheduler) -> Result<Option<PathBuf>>,
{
    let mut out = Vec::with_capacity(entry.staged_schedulers.len());
    for staged in entry.staged_schedulers {
        let Some(host_path) = resolver(staged)? else {
            continue;
        };
        out.push((
            staged.name.to_string(),
            host_path,
            staged.sched_args.iter().map(|s| s.to_string()).collect(),
        ));
    }
    Ok(out)
}

/// True when `KTSTR_SCHEDULER_ALLOW_STALE_FALLBACK` is set to a
/// NON-EMPTY value — the knowing-operator opt-out that lets a failed
/// orchestrated `cargo build -p <sched>` fall back to a pre-built
/// sibling / `target/{debug,release}/` binary AS-IS instead of failing
/// the test. Default (unset / empty) refuses the stale fallback so a
/// build that fails for a new reason cannot silently validate against an
/// old scheduler. Empty-string rejection mirrors
/// [`crate::cargo_test_mode::cargo_test_mode_active`] so a stray
/// `KTSTR_SCHEDULER_ALLOW_STALE_FALLBACK=` from a CI shell cannot
/// re-enable the hazard.
fn allow_stale_scheduler_fallback() -> bool {
    std::env::var(crate::KTSTR_SCHEDULER_ALLOW_STALE_FALLBACK_ENV)
        .map(|v| !v.is_empty())
        .unwrap_or(false)
}

/// Resolve a scheduler binary from a `SchedulerSpec`.
///
/// Returns the resolved path (if any) paired with the
/// [`ResolveSource`] naming the discovery branch that produced it.
/// The source is load-bearing for downstream provenance: only
/// [`ResolveSource::AutoBuilt`] guarantees the binary matches the
/// current workspace tree; every other variant locates a
/// pre-existing file whose git hash is UNKNOWN to this process.
///
/// `manifest_dir` is the declaring crate's `CARGO_MANIFEST_DIR` (from
/// [`Scheduler::manifest_dir`](crate::test_support::Scheduler::manifest_dir)),
/// forwarded to [`crate::build_and_find_binary`] so the on-demand
/// `Discover` build runs in the scheduler's OWN workspace.
///
/// Variant mapping:
/// - `Eevdf` / `KernelBuiltin { .. }` → `(None, NotFound)` (no
///   user-space binary).
/// - `Path(p)` → `(Some(p), Path)` (explicit caller-named path;
///   validated for existence).
/// - `Discover(name)` → cascade through `KTSTR_SCHEDULER` env
///   ([`EnvVar`](ResolveSource::EnvVar)), `$PATH` lookup when
///   `KTSTR_CARGO_TEST_MODE` is active
///   ([`PathLookup`](ResolveSource::PathLookup)), sibling of
///   `current_exe` ([`SiblingDir`](ResolveSource::SiblingDir)),
///   `target/debug/` ([`TargetDebug`](ResolveSource::TargetDebug)),
///   `target/release/` ([`TargetRelease`](ResolveSource::TargetRelease)),
///   on-demand build ([`AutoBuilt`](ResolveSource::AutoBuilt)). In the
///   orchestrated (non-cargo-test) flow the on-demand build runs FIRST
///   and a build FAILURE REFUSES (returns the error) rather than serving
///   a stale pre-built binary, unless
///   [`KTSTR_SCHEDULER_ALLOW_STALE_FALLBACK`](crate::KTSTR_SCHEDULER_ALLOW_STALE_FALLBACK_ENV)
///   is set. Exhausting every branch is a hard error. The PATH lookup is
///   only enabled in cargo-test mode so the existing nextest /
///   `cargo ktstr test` discovery cascade remains canonical
///   (sibling-of-test-binary first) — pulling a system-wide
///   `scx_layered` ahead of a workspace-built one would corrupt
///   gauntlet runs whose results must reflect the in-tree
///   scheduler revision.
pub fn resolve_scheduler(
    spec: &SchedulerSpec,
    manifest_dir: &str,
) -> Result<(Option<PathBuf>, ResolveSource)> {
    match spec {
        SchedulerSpec::Eevdf | SchedulerSpec::KernelBuiltin { .. } => {
            Ok((None, ResolveSource::NotFound))
        }
        SchedulerSpec::Path(p) => {
            let path = PathBuf::from(p);
            anyhow::ensure!(
                path.exists(),
                "scheduler binary at '{p}' does not exist on disk. \
                 SchedulerSpec::Path treats its argument as an \
                 already-built binary — build the scheduler first \
                 (e.g. cargo build -p scx_<name>) and pass its \
                 target/debug/scx_<name> path, or correct the path if \
                 it has shifted."
            );
            Ok((Some(path), ResolveSource::Path))
        }
        SchedulerSpec::Discover(name) => {
            // 1. KTSTR_SCHEDULER env var (global / coarse fallback —
            // applies to every Discover scheduler regardless of name).
            if let Ok(p) = std::env::var(crate::KTSTR_SCHEDULER_ENV) {
                let path = PathBuf::from(&p);
                if path.exists() {
                    return Ok((Some(path), ResolveSource::EnvVar));
                }
            }

            // 1b. KTSTR_CARGO_TEST_MODE: try $PATH lookup so a user
            // who installed scx_layered (or scx-ktstr) on PATH can
            // run the test without going through the cargo-ktstr
            // wrapper or having a target/debug/ build of the
            // scheduler. Only active in cargo-test mode — outside
            // that mode the sibling-dir / target-dir cascade below
            // remains authoritative so gauntlet runs land on the
            // workspace-built scheduler revision.
            if crate::cargo_test_mode::cargo_test_mode_active()
                && let Some(found) = find_on_path(name)
            {
                return Ok((Some(found), ResolveSource::PathLookup));
            }

            // 1c. Orchestrated (non-cargo-test-mode) flow: prefer a
            // fresh workspace build. `cargo build -p {name}` rebuilds
            // the scheduler when its sources (incl. src/bpf/*.bpf.c via
            // its build.rs) changed and is a fast no-op when
            // up-to-date, so an edited scheduler never runs stale. The
            // sibling / target-dir cascade below returns a pre-built
            // binary AS-IS with no staleness check, so serving it after
            // a build that was expected to succeed would silently
            // validate the test against a stale scheduler. Therefore a
            // build FAILURE here REFUSES by default — it returns the
            // error, which propagates to a hard test failure on the
            // result surface (not a swallowed eprintln). The cascade
            // below is reached on this path ONLY when
            // KTSTR_SCHEDULER_ALLOW_STALE_FALLBACK is set (the knowing-
            // operator opt-out for a momentarily-broken cargo). build_*
            // also errors when cargo is absent or no bin artifact is
            // produced; the refusal covers all three — a run that cannot
            // produce a fresh binary must not silently use a stale one.
            // cargo-test-mode is excluded entirely: it targets an
            // installed scheduler (PATH lookup above) without a
            // workspace build, so its cascade is legitimate.
            if !crate::cargo_test_mode::cargo_test_mode_active() {
                match crate::build_and_find_binary(name, manifest_dir) {
                    Ok(path) => return Ok((Some(path), ResolveSource::AutoBuilt)),
                    Err(e) => {
                        if !allow_stale_scheduler_fallback() {
                            // Attach the SchedulerBuildRefused marker (inner) so
                            // dispatch forces a hard FAIL even under expect_err,
                            // then the operator-facing message (outer, shown first
                            // by {e:#}). build_and_find_binary's cargo-stderr stays
                            // innermost in the chain.
                            return Err(e
                                .context(crate::test_support::eval::SchedulerBuildRefused)
                                .context(format!(
                                    "ktstr_test: workspace build of scheduler \
                                     '{name}' failed; refusing to validate against \
                                     a possibly-stale pre-built binary. Fix the \
                                     build, or set \
                                     KTSTR_SCHEDULER_ALLOW_STALE_FALLBACK=1 to fall \
                                     back to a pre-built sibling/target-dir binary \
                                     AS-IS."
                                )));
                        }
                        eprintln!(
                            "ktstr_test: workspace build of scheduler '{name}' \
                             failed ({e:#}); KTSTR_SCHEDULER_ALLOW_STALE_FALLBACK \
                             set — falling back to a pre-built binary if present"
                        );
                    }
                }
            }

            // 2. Sibling of current executable (or parent of deps/)
            if let Ok(exe) = crate::resolve_current_exe()
                && let Some(dir) = exe.parent()
            {
                let candidate = dir.join(name);
                if candidate.exists() {
                    return Ok((Some(candidate), ResolveSource::SiblingDir));
                }
                // Integration tests and nextest place test binaries in
                // target/{debug,release}/deps/. The scheduler binary is
                // one level up in target/{debug,release}/.
                if dir.file_name().is_some_and(|d| d == "deps")
                    && let Some(parent) = dir.parent()
                {
                    let candidate = parent.join(name);
                    if candidate.exists() {
                        return Ok((Some(candidate), ResolveSource::SiblingDir));
                    }
                }
            }

            // 3-4. target/{debug,release}/ pre-built fallbacks (reached
            // only when the build-first step could not run). Probe the
            // profile-matching dir FIRST: the scheduler defaults to the
            // release profile (see `build_and_find_binary`), so prefer
            // target/release/ over a possibly-stale target/debug/ binary
            // unless KTSTR_SCHEDULER_PROFILE=dev explicitly selects the
            // debug tree.
            let prefer_release = crate::scheduler_profile_name() != "dev";
            for (dir, source) in target_dir_probe_order(prefer_release) {
                let candidate = PathBuf::from(dir).join(name);
                if candidate.exists() {
                    return Ok((Some(candidate), source));
                }
            }

            // 5. Build the scheduler package on demand — ONLY in
            // cargo-test-mode, which skips the build-first step 1c, so this
            // is its FIRST build attempt when the PATH / sibling / target-dir
            // lookups all miss. The non-cargo-test flow already ran the build
            // in step 1c (returning Ok, refusing on failure, or — under the
            // opt-out — falling through here intending a PRE-BUILT binary), so
            // re-running the build here would be redundant; skip straight to
            // the bail.
            if crate::cargo_test_mode::cargo_test_mode_active() {
                match crate::build_and_find_binary(name, manifest_dir) {
                    Ok(path) => return Ok((Some(path), ResolveSource::AutoBuilt)),
                    Err(e) => {
                        eprintln!("ktstr_test: auto-build scheduler '{name}' failed: {e:#}")
                    }
                }
            }

            anyhow::bail!(
                "scheduler '{name}' not found. Set KTSTR_SCHEDULER or \
                 place it next to the test binary or in target/{{debug,release}}/"
            )
        }
    }
}

/// Order to probe the pre-built `target/{debug,release}/` scheduler
/// binaries in the `Discover` cascade fallback: the profile-matching
/// directory first. With `prefer_release` (the release-profile default —
/// see [`crate::scheduler_profile_name`] — unless `KTSTR_SCHEDULER_PROFILE=dev`)
/// the release build is the intended one, so `target/release/` is probed
/// before a possibly-stale opposite-profile `target/debug/` binary. Pure +
/// `pub(crate)` so the reorder is unit-testable without staging a CWD.
pub(crate) fn target_dir_probe_order(prefer_release: bool) -> [(&'static str, ResolveSource); 2] {
    if prefer_release {
        [
            ("target/release", ResolveSource::TargetRelease),
            ("target/debug", ResolveSource::TargetDebug),
        ]
    } else {
        [
            ("target/debug", ResolveSource::TargetDebug),
            ("target/release", ResolveSource::TargetRelease),
        ]
    }
}
