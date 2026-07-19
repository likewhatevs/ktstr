//! Scheduler-binary resolution: maps a `SchedulerSpec` to a path plus a
//! `ResolveSource` provenance (the orchestrator manifest, env override,
//! `$PATH` lookup, or on-demand workspace build) and dedups include-file
//! lists. Split out of eval/mod.rs to keep the module under the size ceiling.

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
/// Each variant identifies the resolution branch that produced the
/// path, so downstream tooling (sidecar, cache-key construction, log
/// lines) can distinguish "we located an *existing* binary whose git
/// hash we don't control" from "we just built this binary in the
/// declaring crate's workspace and therefore know its source commit
/// matches that workspace's HEAD."
///
/// Only the [`AutoBuilt`](Self::AutoBuilt) variant carries an in-child
/// source-commit guarantee. A manifest artifact was elected by the parent and
/// can represent either a workspace build or the global operator override, so
/// the child records that fact without guessing its source commit.
///
/// `Eevdf` / `KernelBuiltin` / `Path` resolutions do not go through
/// the `Discover` flow:
/// - `Eevdf` / `KernelBuiltin` → [`NotFound`](Self::NotFound) (no
///   user-space binary involved; the tuple's `Option<PathBuf>` is
///   `None`).
/// - `Path(p)` → [`Path`](Self::Path) (the caller named the binary
///   explicitly in the test entry — no env-var or filesystem search
///   runs).
///
/// The variant ordering in the enum mirrors the resolution order in
/// [`resolve_scheduler`] so a reviewer can scan both lists in lockstep.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolveSource {
    /// Resolved through the authoritative parent-written scheduler artifact
    /// manifest. No per-child search or Cargo invocation ran.
    Manifest,
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
    /// requested name was found on the user's `$PATH` ahead of the
    /// on-demand workspace build. Git-hash provenance UNKNOWN — the
    /// binary on PATH may be a system-wide install, a prior build,
    /// or a custom one the user staged for this run.
    PathLookup,
    /// Built on demand by [`crate::build_and_find_binary`] inside this
    /// process, in the declaring crate's workspace. The build targets
    /// that workspace's HEAD by construction — the ONLY variant where
    /// the source commit is known to match the workspace tree the
    /// tests run from.
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
    /// representation. Variant order matches the resolution order.
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Manifest => "manifest",
            Self::Path => "path",
            Self::EnvVar => "env_var",
            Self::PathLookup => "path_lookup",
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
/// `KTSTR_CARGO_TEST_MODE` is active, so an installed scheduler
/// resolves without a workspace build; the orchestrated path skips
/// it and always builds so gauntlet runs land on the workspace-built
/// scheduler revision rather than a system-wide install on PATH.
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

/// Resolve a scheduler binary from a `SchedulerSpec`.
///
/// Returns the resolved path (if any) paired with the
/// [`ResolveSource`] naming the branch that produced it. The source is
/// load-bearing for downstream provenance: only
/// [`ResolveSource::AutoBuilt`] guarantees the binary matches the
/// declaring crate's workspace tree; every other variant locates a
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
/// - `Discover(name)` → four steps:
///   1. exact parent manifest lookup
///      ([`Manifest`](ResolveSource::Manifest)). Presence is authoritative:
///      invalid or missing data is a hard error and cannot fall through.
///   2. `KTSTR_SCHEDULER` env override
///      ([`EnvVar`](ResolveSource::EnvVar)) — a global, name-agnostic
///      binary path applying to every `Discover` scheduler.
///   3. cargo-test mode only: `$PATH` lookup
///      ([`PathLookup`](ResolveSource::PathLookup)), so an installed
///      scheduler resolves without a workspace build.
///   4. `cargo build -p <name>` in the declaring crate's workspace
///      ([`AutoBuilt`](ResolveSource::AutoBuilt)), in BOTH modes.
///
///   Cargo owns freshness: it rebuilds when the scheduler's sources
///   (incl. `src/bpf/*.bpf.c` via its build.rs) change, is a no-op when
///   they are up to date, and fails when the scheduler cannot build. A
///   failed build is a hard error, full stop — a scheduler that will
///   not build is a failed test, never a silently-stale pre-built
///   binary. The error carries the `SchedulerBuildRefused`
///   marker so dispatch forces a hard FAIL even under `expect_err`: a
///   host-side build fault must not invert to PASS. The `$PATH` lookup
///   is gated to cargo-test mode so gauntlet runs always land on the
///   workspace-built scheduler revision rather than a system-wide
///   install.
pub fn resolve_scheduler(
    spec: &SchedulerSpec,
    manifest_dir: &str,
) -> Result<(Option<PathBuf>, ResolveSource)> {
    if matches!(spec, SchedulerSpec::Discover(_) | SchedulerSpec::Path(_)) {
        match crate::scheduler_artifact::scheduler_artifact_from_env(None, spec, manifest_dir) {
            Ok(Some(path)) => return Ok((Some(path), ResolveSource::Manifest)),
            Ok(None) => {}
            Err(error) => {
                return Err(error
                    .context(FrameworkInfrastructureFailure)
                    .context(
                        "ktstr_test: parent scheduler artifact handoff is invalid",
                    ));
            }
        }
    }

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
            // 2. KTSTR_SCHEDULER env override (global / name-agnostic —
            // applies to every Discover scheduler regardless of name).
            if let Ok(p) = std::env::var(crate::KTSTR_SCHEDULER_ENV) {
                let path = PathBuf::from(&p);
                if path.exists() {
                    return Ok((Some(path), ResolveSource::EnvVar));
                }
            }

            // 3. cargo-test mode only: $PATH lookup, so a user who
            // installed the scheduler on PATH can run the test without a
            // workspace build. Gated to cargo-test mode — the
            // orchestrated path skips it and always builds so gauntlet
            // runs land on the workspace-built scheduler revision, not a
            // system-wide install.
            if crate::cargo_test_mode::cargo_test_mode_active()
                && let Some(found) = find_on_path(name)
            {
                return Ok((Some(found), ResolveSource::PathLookup));
            }

            // 4. Build the scheduler in its declaring crate's workspace.
            // Cargo owns freshness: `cargo build -p {name}` rebuilds on
            // source change, no-ops when fresh, and errors when the
            // scheduler cannot build (or cargo is absent, or no bin
            // artifact is produced). A build FAILURE is a hard error —
            // there is no pre-built fallback: a scheduler that will not
            // build is a failed test, not a reason to validate against a
            // possibly-stale binary. Attach the SchedulerBuildRefused
            // marker (inner) so dispatch forces a hard FAIL even under
            // expect_err, then the operator-facing message (outer, shown
            // first by {e:#}). build_and_find_binary's cargo-stderr stays
            // innermost in the chain.
            match crate::build_and_find_binary(name, manifest_dir) {
                Ok(path) => Ok((Some(path), ResolveSource::AutoBuilt)),
                Err(e) => Err(e
                    .context(crate::test_support::eval::SchedulerBuildRefused)
                    .context(format!(
                        "ktstr_test: workspace build of scheduler '{name}' \
                         failed; a scheduler that cannot be built is a failed \
                         test. Fix the build."
                    ))),
            }
        }
    }
}
