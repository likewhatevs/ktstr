//! Kernel resolution and build helpers for the `cargo ktstr` binary.
//!
//! Houses the spec-resolution pipeline that converts `--kernel
//! <SPEC>` arguments into bootable kernel directories, the
//! `kernel build` subcommand dispatcher
//! (`kernel_build` / `kernel_build_one` / `cache_lookup`), and the
//! `format_built_age` cache-hit log helper.
//!
//! The flat `(label, kernel_dir)` list this module emits is what
//! the `test` / `coverage` / `llvm-cov` dispatchers (in
//! [`super::run_cargo`]) hand to the test binary as the kernel
//! dimension of the gauntlet expansion via the
//! [`ktstr::KTSTR_KERNEL_LIST_ENV`] wire format.
//!
//! Pure label emission, the `KTSTR_KERNEL_LIST` wire encoder, and
//! the dedup / collision-detection helpers live in the
//! [`wire_format`] submodule — that subsystem is independently
//! unit-testable without driving the rayon resolve pipeline (every
//! `resolve_one` arm performs real I/O).

pub(crate) mod wire_format;

use std::path::{Path, PathBuf};

use ktstr::cache::{CacheDir, CacheEntry};
use ktstr::cli;
use ktstr::fetch;

pub(crate) use wire_format::{
    cache_key_to_version_label, decorate_path_label_for_dirty, dedupe_resolved,
    detect_label_collisions, encode_kernel_list, git_kernel_label, path_kernel_label,
    preflight_collision_check,
};

/// Resolve a `KernelId::Path` to a directory suitable for export
/// via [`ktstr::KTSTR_KERNEL_ENV`] plus the dirty-tree flag the
/// caller uses to decorate the kernel label.
///
/// Routes Path specs through [`cli::resolve_kernel_dir_to_entry`]
/// so they share the same cache pipeline as Version / CacheKey /
/// Git specs:
///   - Clean source tree, cache miss → build, store at
///     `local-{hash7}-{arch}-kc{suffix}`, return cache entry dir
///     with `is_dirty=false`.
///   - Clean source tree, cache hit → skip build, log a
///     `tracing::info!` cache-hit line referencing the user's raw
///     input path, the resolved cache key, and the build age
///     (rendered on stderr only when `RUST_LOG` enables the `info`
///     level; suppressed under the default `warn` filter), then
///     return cache entry dir with `is_dirty=false`.
///   - Dirty source tree → build in source, skip cache store,
///     return canonical source dir with `is_dirty=true`. The
///     caller appends `_dirty` to the kernel label so the test
///     report distinguishes the non-reproducible run from a
///     subsequent clean rebuild of the same tree.
///
/// Both directory shapes are valid inputs to
/// [`ktstr::kernel_path::find_image_in_dir`]'s child consumers;
/// the cache-entry layout (`<dir>/<image_name>`) and source-tree
/// layout (`<dir>/arch/<arch>/boot/<image_name>`) are both probed.
///
/// `raw_input` is the verbatim user-supplied `--kernel` argument
/// before canonicalization — used in the cache-hit `tracing::info!`
/// line so the operator sees the path they actually typed (e.g.
/// `../linux`) rather than the resolved canonical form, and in
/// the resolve-failure error so a typo names whatever the user
/// supplied.
///
/// On canonicalize / source-tree-validation failure, the inner
/// error is re-wrapped with the user's raw input + the standard
/// `KTSTR_KERNEL_HINT` so the diagnostic shape matches the
/// behaviour the previous inline `canonicalize` call provided.
/// The single canonicalize then lives inside
/// [`ktstr::fetch::local_source`] (called via
/// [`cli::resolve_kernel_dir_to_entry`]); doing it twice in this
/// function and again in `local_source` produced redundant
/// syscalls without changing the resulting path.
pub(crate) fn resolve_path_kernel(p: &Path, raw_input: &str) -> Result<(PathBuf, bool), String> {
    // Boundary bridge: `cli::resolve_kernel_dir_to_entry` returns
    // `anyhow::Result<KernelDirOutcome>` while this function
    // returns `Result<_, String>`, so we stringify at the call
    // site. A broader anyhow migration across cargo-ktstr.rs is
    // pending and would drop this last bridge.
    let outcome = cli::resolve_kernel_dir_to_entry(p, "cargo ktstr", None).map_err(|e| {
        format!(
            "--kernel {raw_input}: {e:#}. {hint}",
            hint = ktstr::KTSTR_KERNEL_HINT,
        )
    })?;
    if let Some(hit) = outcome.cache_hit {
        tracing::info!(
            "cargo ktstr: cache hit for {raw_input} ({key}{age})",
            key = hit.cache_key,
            age = format_built_age(&hit.built_at),
        );
    }
    Ok((outcome.dir, outcome.is_dirty))
}

/// Resolve `--kernel <SPEC>` (or absence) to a bootable image path
/// for the `shell` and `verifier` subcommands.
///
/// `KERNEL_POLICY` is declared as a function-local const because
/// this is the sole consumer — the policy describes cargo-ktstr's
/// host-side conventions for [`cli::resolve_kernel_image`]:
///   - `accept_raw_image: true` allows `--kernel /path/to/bzImage`
///     to short-circuit the source-tree / cache-key path resolution
///     pipeline (the test harness routes through
///     [`resolve_kernel_set`] which only accepts directory inputs by
///     construction, so per-test labels are deterministic).
///   - `cli_label: "cargo ktstr"` is the user-facing prefix that
///     [`cli::resolve_kernel_image`] embeds in its diagnostic
///     messages so failures cite the binary the operator invoked.
///
/// On `Some(spec)` the call delegates to
/// [`cli::resolve_kernel_image`] which dispatches on the parsed
/// [`ktstr::kernel_path::KernelId`] variant (Path / Version /
/// CacheKey / Git). On `None` the same helper falls back to
/// [`ktstr::find_kernel`] (cache-then-filesystem auto-discovery)
/// followed by a kernel.org download if nothing is found.
///
/// Errors stringify the underlying anyhow chain via `{e:#}` so the
/// shell / verifier dispatchers stay on the `Result<_, String>`
/// surface this binary uses end-to-end.
pub(crate) fn resolve_kernel_image(kernel: Option<&str>) -> Result<PathBuf, String> {
    /// Policy for cargo-ktstr's shell + verifier kernel resolution:
    /// accept raw image files, use "cargo ktstr" as the CLI label.
    const KERNEL_POLICY: cli::KernelResolvePolicy<'static> = cli::KernelResolvePolicy {
        accept_raw_image: true,
        cli_label: "cargo ktstr",
    };
    cli::resolve_kernel_image(kernel, &KERNEL_POLICY).map_err(|e| format!("{e:#}"))
}

/// Format a cache entry's `built_at` ISO-8601 timestamp as a
/// human-readable age suffix for the cache-hit log line.
///
/// Returns `, built {age} ago` (with the leading comma+space) on
/// successful parse + elapsed-since-now computation, so the call
/// site can splice it directly into the parenthesised message:
/// `(local-..., built 2h 15m ago)`. Returns the empty string when
/// either the timestamp can't be parsed (malformed metadata) or
/// the build moment is in the future relative to local clock
/// (clock skew on a shared cache); callers see `(local-...)` with
/// no age suffix in those degenerate cases.
pub(crate) fn format_built_age(built_at: &str) -> String {
    let Ok(parsed) = humantime::parse_rfc3339(built_at) else {
        return String::new();
    };
    let Ok(elapsed) = std::time::SystemTime::now().duration_since(parsed) else {
        return String::new();
    };
    // Truncate to whole-second granularity. `format_duration` on
    // sub-second remainders renders nanos that aren't useful
    // ("2h 15m 32s 184ms 7us 12ns") and clutter the cache-hit
    // line.
    let elapsed = std::time::Duration::from_secs(elapsed.as_secs());
    format!(", built {} ago", humantime::format_duration(elapsed))
}

/// Canonicalize a cache-entry directory before exporting it via
/// [`ktstr::KTSTR_KERNEL_ENV`] / [`ktstr::KTSTR_KERNEL_LIST_ENV`].
/// `CacheDir` roots at the XDG cache home (or `KTSTR_CACHE_DIR`),
/// both typically absolute — but an operator-supplied
/// `KTSTR_CACHE_DIR=./cache` would produce a relative path here and
/// reach the same cwd-divergence bug the `Path` branch defends
/// against. `canonicalize` resolves that from the parent's cwd; a
/// failure means the cache dir was removed between lookup and
/// export (rare race), in which case we fall back to the original
/// path rather than bailing — the child will re-enter its own cache
/// lookup and surface the real missing-entry error.
pub(crate) fn canonicalize_cache_dir(cache_dir: PathBuf) -> PathBuf {
    std::fs::canonicalize(&cache_dir).unwrap_or(cache_dir)
}

/// Resolve one already-validated [`ktstr::kernel_path::KernelId`]
/// (NOT `Range` — the caller fans Range out to per-version
/// `Version` ids before calling here) to a `(label, dir)` tuple.
///
/// Extracted from `resolve_kernel_set`'s rayon body so the per-
/// spec match arm is one shared function rather than five inline
/// arms: [`resolve_one_spec`] fans a Range out to per-version
/// `Version` ids and calls here once per version (all on the same
/// bounded rayon pool), and every other spec calls here once.
///
/// Range fan-out lives on the caller because the
/// `expand_kernel_range` step yields a `Vec<String>` that has to
/// be expanded into the same parallel pool — `flat_map_iter` is
/// the wrong shape for "fan out N items into the parent
/// iterator." See the parallel comment block in
/// [`resolve_kernel_set`] for the full strategy.
pub(crate) fn resolve_one(
    id: ktstr::kernel_path::KernelId,
    mp: Option<&ktstr::cli::FetchProgress>,
) -> Result<(String, PathBuf), String> {
    use ktstr::kernel_path::KernelId;
    match id {
        KernelId::Path(p) => {
            // Capture the user's raw input string before any
            // canonicalization so cache-hit diagnostics inside
            // `resolve_path_kernel` can name the path they
            // actually typed (`../linux`) instead of the
            // resolved canonical form.
            let raw_input = p.display().to_string();
            // Compute the BASE label from the CANONICAL SOURCE TREE
            // path, NOT the directory `resolve_path_kernel`
            // returns. The returned dir may be a cache entry
            // (`<cache>/local-{hash7}-{arch}-kc{suffix}`); a
            // basename-derived label off that would render as
            // `path_local-{hash7}-{arch}-kc{suffix}_{hash6}` and
            // change between cache-miss runs (when
            // `path_kernel_label` would have observed the source
            // tree dir) and cache-hit runs (when it would have
            // observed the cache entry dir). Pinning the label
            // to the canonical SOURCE path keeps the operator-
            // facing identifier stable across cache states for
            // the same `--kernel /path/to/linux` invocation.
            //
            // The dirty-tree flag from `resolve_path_kernel`
            // appends a `_dirty` suffix when `local_source`
            // observed uncommitted modifications. The dirty-tree
            // build skips the cache store, so a `path_linux_a3b1c2`
            // row in the test report is not interchangeable with
            // a `path_linux_a3b1c2_dirty` row — the former is
            // reproducible from the recorded git hash, the latter
            // is not. Surfacing the divergence in the label keeps
            // the gauntlet output honest about which runs the
            // operator can re-run from cache.
            let canon_input = std::fs::canonicalize(&p).map_err(|e| {
                format!(
                    "--kernel {}: path does not exist or cannot be \
                     canonicalized ({e:#}). {hint}",
                    p.display(),
                    hint = ktstr::KTSTR_KERNEL_HINT,
                )
            })?;
            let base_label = path_kernel_label(&canon_input);
            let (dir, is_dirty) = resolve_path_kernel(&p, &raw_input)?;
            let label = decorate_path_label_for_dirty(&base_label, is_dirty);
            Ok((label, dir))
        }
        KernelId::Version(ref ver) => {
            let cache_dir = ktstr::cli::resolve_cached_kernel(&id, "cargo ktstr", mp)
                .map_err(|e| format!("{e:#}"))?;
            let dir = canonicalize_cache_dir(cache_dir);
            Ok((ver.clone(), dir))
        }
        KernelId::CacheKey(ref key) => {
            let cache_dir = ktstr::cli::resolve_cached_kernel(&id, "cargo ktstr", mp)
                .map_err(|e| format!("{e:#}"))?;
            let dir = canonicalize_cache_dir(cache_dir);
            // Extract a discriminating label from the cache key —
            // tarball keys yield the version prefix
            // (`6.14.2-tarball-…` → `6.14.2`), content-addressed git
            // keys yield a compact kind/ref-hash label, and local keys
            // yield `local_{hash6}` (or `local_unknown` for non-git
            // trees). See [`cache_key_to_version_label`] for the full
            // per-shape contract and fallback behavior.
            let label = cache_key_to_version_label(key).to_string();
            Ok((label, dir))
        }
        KernelId::Git {
            ref url,
            ref git_ref,
            ref_kind,
        } => {
            // Auto-discovery test path: no build-flag overrides (force /
            // clean / cpu_cap / extra_kconfig are `cargo ktstr kernel
            // build --kernel git+…`-only).
            let cache_dir = ktstr::cli::resolve_git_kernel(
                url,
                git_ref,
                ref_kind,
                "cargo ktstr",
                mp,
                false,
                false,
                None,
                None,
            )
            .map_err(|e| format!("resolve git+{url}#{git_ref}: {e:#}"))?;
            let dir = canonicalize_cache_dir(cache_dir);
            let label = git_kernel_label(url, git_ref, ref_kind);
            Ok((label, dir))
        }
        KernelId::Range { start, end, .. } => {
            // Defensive: the caller fans Range out to per-version
            // Version ids before calling here. This arm exists
            // only so the compiler accepts the exhaustive match;
            // hitting it indicates a programming error in the
            // caller's flat-map shape rather than a user-visible
            // condition, so the diagnostic is descriptive enough
            // to point a developer at the wrong call site.
            Err(format!(
                "internal: resolve_one called with Range {start}..{end}; \
                 caller must expand Range via `expand_kernel_range` and \
                 call `resolve_one` per version"
            ))
        }
        ref id @ (KernelId::Package { .. } | KernelId::Distro { .. }) => {
            // Local packages and distro kernels acquire into the cache
            // (download + extract, or unpack local files) and resolve to
            // the entry directory the gauntlet boots. Validate first so a
            // malformed distro release surfaces its specific diagnostic.
            id.validate().map_err(|e| format!("--kernel {id}: {e}"))?;
            match id {
                KernelId::Package { path } => {
                    let dir =
                        ktstr::distro::acquire::acquire_package_kernel(std::slice::from_ref(path))
                            .map_err(|e| format!("--kernel {id}: {e:#}"))?;
                    let label = path
                        .file_stem()
                        .map(|s| s.to_string_lossy().into_owned())
                        .unwrap_or_else(|| "package".to_string());
                    Ok((label, canonicalize_cache_dir(dir)))
                }
                KernelId::Distro { kind, release } => {
                    let dir = ktstr::distro::acquire::acquire_distro_kernel(
                        *kind,
                        release.as_deref(),
                        "cargo ktstr",
                        mp,
                    )
                    .map_err(|e| format!("--kernel {id}: {e:#}"))?;
                    Ok((id.to_string(), canonicalize_cache_dir(dir)))
                }
                _ => unreachable!("arm guarded to Package | Distro"),
            }
        }
    }
}

/// Resolve every `--kernel` spec to a flat list of `(kernel_label,
/// kernel_dir)` pairs. Each Range expands to one entry per release
/// in the interval; each Path / Version / CacheKey / Git produces
/// exactly one entry.
///
/// The flat list is what `cargo ktstr test` (and `coverage` /
/// `llvm-cov`) hand to the test binary as the kernel dimension of
/// the gauntlet expansion: every (test × scenario × topology ×
/// kernel) tuple becomes a distinct nextest test case so
/// nextest's parallelism, retries, and `-E` filtering apply
/// natively. A single `cargo nextest run` (or `cargo llvm-cov
/// nextest`) invocation services every variant; profraw lands per-
/// child so cargo-llvm-cov merges all of them automatically.
///
/// Build / download / clone failures abort the resolution before
/// any test runs — there's no useful state to continue from
/// (a missing kernel can't be tested, and continuing would mask
/// which kernel was requested-but-unavailable in the operator-
/// visible error stream).
///
/// `kernel_label` for each entry is a semantic, operator-readable
/// identity:
/// - Path → `path_{basename}_{hash6}` (basename + 6-char hash of the
///   canonical path so two distinct directories with the same name
///   don't collide).
/// - Version / Range expansion → the version string verbatim
///   (e.g. `6.14.2`, `6.15-rc3`).
/// - CacheKey → the compact semantic label produced by
///   `cache_key_to_version_label` for tarball, content-addressed git,
///   and local key shapes.
/// - Git → `git_{owner}_{repo}_{kind}_{ref}` extracted from the URL,
///   the ref kind (tag/branch/sha), and the git ref.
///
/// The downstream [`ktstr::test_support::sanitize_kernel_label`]
/// applies the `kernel_` prefix and `[a-z0-9_]+` normalisation; this
/// label is the human-meaningful payload it operates on.
pub(crate) fn resolve_kernel_set(
    specs: &[String],
    include_eol: bool,
) -> Result<Vec<(String, PathBuf)>, String> {
    preflight_collision_check(specs)?;
    // One progress group shared across every parallel worker: each
    // download/clone adds its own bar to the group's MultiProgress
    // (concurrent bars coordinate; off-TTY they are hidden). Cleared
    // after the parallel resolve — on success and on error — so no
    // bars linger into the build/test phase. `?` is applied AFTER the
    // clear so an error path still tidies the terminal.
    let mp = ktstr::cli::FetchProgress::new();
    let resolved = resolve_specs_parallel(specs, &mp, include_eol);
    mp.clear();
    let resolved = dedupe_resolved(resolved?);
    detect_label_collisions(&resolved)?;
    Ok(resolved)
}

/// Resolve one trimmed `--kernel` spec into the `(label, path)`
/// results it expands to.
///
/// Returns a `Vec` so the caller can splice it into the rayon
/// `flat_map_iter` parent iterator via `.into_iter()`: a Range
/// yields one element per expanded version, every other spec
/// yields exactly one element. Each element is an opaque
/// `Result<(String, PathBuf), String>` driven by
/// [`resolve_one_with_progress`]; the caller's `collect` on
/// `Result` short-circuits on the first error.
///
/// Each spec resolves independently:
///   - Path → cache lookup → maybe build (no network).
///     Clean source trees hit the local-source cache key
///     `local-{hash7}-{arch}-kc{suffix}`; cache miss reaches
///     the same `kernel_build_pipeline` Version/CacheKey/Git
///     specs use, with the result stored back at the same key.
///     Dirty trees skip the cache store and build in place.
///   - Version / CacheKey → cache lookup → maybe download +
///     build.
///   - Range → fetch releases.json once, then per-version
///     cache lookup → maybe download + build for each
///     expanded version.
///   - Git branch/tag → one exact-ref gix prepare → content-cache /
///     builder election → receive only for the winner → maybe build.
///     An explicit GitHub SHA uses codeload.
///
/// Two phases of work happen behind the per-spec resolvers:
/// (1) network I/O — kernel.org tarball download, an explicit-SHA
///     GitHub codeload snapshot, or an exact-ref gix shallow fetch — which is
///     independent across specs and overlaps freely.
/// (2) build — `make -j$(nproc)` invoked under an LLC flock
///     plus a cgroup v2 sandbox (`acquire_build_reservation`
///     in `kernel_build_pipeline`). The LLC flock is taken
///     SHARED (`LOCK_SH`), so it does NOT serialize concurrent
///     builders against each other — distinct `--kernel` specs
///     build concurrently across rayon workers. The shared lock
///     coordinates a build against a perf-mode test run (which
///     takes the LLC `LOCK_EX`), not build-vs-build.
///
/// Net effect: parallelizing `resolve_kernel_set` overlaps the
/// download / clone phase AND concurrent builds. Because builds
/// run concurrently, the build phase renders its progress through
/// the shared `FetchProgress` group (whose `MultiProgress::add`
/// is `RwLock`-serialized) rather than a process-global `Spinner`.
/// Builds of the SAME source path still serialize via a separate
/// exclusive source-tree flock, and concurrent stores against
/// different cache keys are safe (the cache's per-key exclusive
/// store lock).
///
/// Concurrent resolves of the SAME spec (e.g. a duplicated
/// `--kernel 6.14.2` flag) racing on the same cache key are
/// also safe — the cache's exclusive store lock means the
/// second resolver re-checks the cache after acquiring its
/// own lock and finds the just-written entry, skipping the
/// redundant build.
///
/// A Range spec expands to N versions; this fn resolves them
/// concurrently via [`resolve_versions_parallel`] rather than
/// serializing versions against itself. That is the same
/// parallelism peer specs (other top-level `--kernel` arguments)
/// already get: builds do NOT serialize against each other (the
/// LLC flock is shared — see above), and each expanded version is
/// a distinct tarball cache key hitting its own per-key store
/// lock, so a range's versions are as independent as separate
/// `--kernel` flags.
///
/// Fail-fast is unchanged from the pre-parallel form: the helper's
/// `collect` gathers every version's `Result`, then the outer
/// `collect::<Result<_, _>>` in `resolve_specs_parallel` aborts on
/// the first Err it observes — which version surfaces is not
/// deterministic under concurrency, but a single failure still
/// aborts the whole resolve just as the sequential loop did.
#[derive(Clone, Copy)]
enum ResolveExecution {
    Parallel,
    Sequential,
}

fn resolve_one_spec(
    trimmed: String,
    mp: &ktstr::cli::FetchProgress,
    include_eol: bool,
    execution: ResolveExecution,
) -> Vec<Result<(String, PathBuf), String>> {
    use ktstr::kernel_path::KernelId;

    // Validation runs first so an inverted Range bails
    // before any I/O — same diagnostic timing the
    // pre-parallel loop preserved.
    let id = KernelId::parse(&trimmed);
    if let Err(e) = id.validate() {
        return vec![Err(format!("--kernel {id}: {e}"))];
    }
    match id {
        KernelId::Range { start, end, .. } => {
            match ktstr::cli::expand_kernel_range(&start, &end, "cargo ktstr", include_eol) {
                Ok(versions) => {
                    let resolve_version = |ver: &str| {
                        resolve_one_with_progress(KernelId::Version(ver.to_string()), mp)
                            .map_err(|e| format!("resolve kernel {ver}: {e}"))
                    };
                    match execution {
                        ResolveExecution::Parallel => {
                            resolve_versions_parallel(versions, resolve_version)
                        }
                        ResolveExecution::Sequential => {
                            resolve_versions_sequential(versions, resolve_version)
                        }
                    }
                }
                Err(e) => vec![Err(format!("{e:#}"))],
            }
        }
        other => vec![resolve_one_with_progress(other, mp)],
    }
}

/// Resolve every version of an expanded Range concurrently,
/// preserving input (version) order in the returned Vec.
///
/// `Vec::into_par_iter` is an `IndexedParallelIterator`, so
/// `collect` writes each result into its input slot — the output
/// is in the same order as `versions` regardless of which worker
/// finishes first (so a downstream `resolved[0] -> KTSTR_KERNEL`
/// choice stays deterministic). The nested `into_par_iter` runs on
/// whatever rayon pool the caller is installed under
/// ([`resolve_specs_parallel`]'s bounded pool), sharing it via
/// work-stealing rather than spawning a second pool, so
/// `KTSTR_KERNEL_PARALLELISM` still bounds total width. `collect`
/// gathers EVERY version's `Result` (Ok and Err alike); the
/// caller's outer `collect::<Result<_, _>>` is what aborts the
/// cohort on the first Err.
///
/// The per-version `resolve` closure is injected so this
/// concurrency + order contract can be unit-tested without
/// network / build I/O.
fn resolve_versions_parallel<R>(
    versions: Vec<String>,
    resolve: R,
) -> Vec<Result<(String, PathBuf), String>>
where
    R: Fn(&str) -> Result<(String, PathBuf), String> + Sync,
{
    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    versions.into_par_iter().map(|ver| resolve(&ver)).collect()
}

/// Resource-exhaustion counterpart to [`resolve_versions_parallel`].
///
/// Used only when the enclosing bounded pool could not create its
/// workers. A Range must stay sequential too: calling the parallel
/// helper outside `pool.install` would lazily initialize Rayon's
/// host-sized global pool and recreate the failure at the nested level.
fn resolve_versions_sequential<R>(
    versions: Vec<String>,
    resolve: R,
) -> Vec<Result<(String, PathBuf), String>>
where
    R: Fn(&str) -> Result<(String, PathBuf), String>,
{
    versions.into_iter().map(|ver| resolve(&ver)).collect()
}

/// `resolve_one` plus per-resolve progress feedback.
///
/// A user passing `--kernel 6.10..6.20` (10+ versions) sees
/// `cargo ktstr: resolved kernel "6.10"` lines as each version
/// finishes its download+build cycle, instead of staring at
/// silence for the multi-minute resolve. Emitted at the Ok-arm
/// of each `resolve_one` call so failures still propagate via
/// the existing fail-fast `collect::<Result<_, _>>?` chain
/// upstream — only successful resolves print. Single-kernel
/// runs emit ONE line; that's negligible noise versus the
/// multi-kernel UX gain. Output is `eprintln!` (stderr) so
/// it doesn't pollute stdout pipelines that consume the
/// tool's other output (e.g. shell scripts piping through
/// jq).
///
/// `tracing::info!` would respect `RUST_LOG`, but the
/// command spends most of its wall time in
/// `resolve_kernel_set` and operators expect progress
/// visibility by default — gating it behind a verbosity
/// flag would defeat the point. Keep it as unconditional
/// `eprintln!` matching the pattern other long-running
/// helpers (`expand_kernel_range`, `kernel_build_pipeline`)
/// already use.
fn resolve_one_with_progress(
    id: ktstr::kernel_path::KernelId,
    mp: &ktstr::cli::FetchProgress,
) -> Result<(String, PathBuf), String> {
    let result = resolve_one(id, Some(mp));
    if let Ok((label, _)) = &result {
        eprintln!("cargo ktstr: resolved kernel {label:?}");
    }
    result
}

/// Resolve every spec in parallel under a bounded rayon ThreadPool,
/// returning the flat (label, path) list before dedup / collision
/// detection.
///
/// Result-collecting fail-fast: rayon's `collect` on
/// `Result<_, _>` aborts on the first Err it observes — which
/// failing spec surfaces is not deterministic under concurrency —
/// discarding the other in-flight results. A single failure still
/// aborts the whole resolve, matching the pre-parallel loop's `?`
/// propagation (that loop surfaced errors in spec order). Peers
/// still in flight clean up via their tempdirs going out of scope
/// (see `download_and_cache_version` / `resolve_git_kernel` for
/// the `tempfile::TempDir`-driven teardown).
fn resolve_specs_parallel(
    specs: &[String],
    mp: &ktstr::cli::FetchProgress,
    include_eol: bool,
) -> Result<Vec<(String, PathBuf)>, String> {
    resolve_specs_parallel_with_pool_builder(specs, mp, include_eol, |max_threads| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(max_threads)
            .build()
            .map_err(|e| e.to_string())
    })
}

/// [`resolve_specs_parallel`] with an injected bounded-pool builder.
///
/// The seam lets tests force the `pthread_create` failure path without
/// changing process limits. Production passes Rayon's real builder.
fn resolve_specs_parallel_with_pool_builder<B>(
    specs: &[String],
    mp: &ktstr::cli::FetchProgress,
    include_eol: bool,
    build_pool: B,
) -> Result<Vec<(String, PathBuf)>, String>
where
    B: FnOnce(usize) -> Result<rayon::ThreadPool, String>,
{
    use rayon::iter::{IntoParallelIterator, ParallelIterator};

    // Cap rayon parallelism via a bounded ThreadPool installed
    // ONLY for this resolve pipeline. Without the cap, an
    // operator passing `--kernel A --kernel B ... --kernel Z`
    // (10+ specs) would saturate the global rayon pool with
    // simultaneous git_clone + tarball downloads. Each download
    // / clone is network-bound and largely cooperative on local
    // CPU, but the spawn cost (rayon worker steal-and-park,
    // tempdir creation, gix or reqwest init) compounds in
    // proportion to spec count, and a contended local network
    // (the kernel.org CDN's per-IP throttle, a developer's home
    // ISP, a CI runner's shared NIC) degrades when too many
    // streams overlap.
    //
    // The cap defaults to `available_parallelism()` — the host's
    // logical CPU count, std-lib provided so no extra dependency
    // is pulled in. Saturating local parallelism is the right
    // ceiling: download streams shouldn't outnumber the threads
    // the host can drive without thrashing. (Concurrent builds are
    // NOT serialized against each other — the LLC flock is shared
    // (LOCK_SH) — but `make -j$(nproc)` already saturates CPU, so
    // additional in-flight downloads wouldn't speed the builds up.)
    //
    // Operators can override the cap via the
    // `KTSTR_KERNEL_PARALLELISM` env var (see
    // [`ktstr::KTSTR_KERNEL_PARALLELISM_ENV`]) — useful when the
    // default is wrong for the host: a fast NIC + slow CPU
    // benefits from more in-flight downloads; a contended CI
    // runner with concurrent jobs benefits from a lower cap to
    // leave bandwidth for siblings. Parsing rules and fallback
    // behavior live in [`ktstr::cli::resolve_kernel_parallelism`]
    // so a typoed export (`=abc`, `=0`) silently degrades to the
    // host-CPU default rather than disabling parallelism.
    //
    // Bounded ThreadPool via `pool.install(|| ...)` scopes the
    // cap to this pipeline only — the global rayon pool is
    // unaffected. If construction fails because pthread creation
    // is already exhausted, the entire pipeline (including versions
    // nested under a Range) runs sequentially. Falling through to a
    // parallel iterator here would lazily create Rayon's host-sized
    // global pool at exactly the point thread creation is failing.
    let max_threads = ktstr::cli::resolve_kernel_parallelism();
    let pool_result = build_pool(max_threads);
    let nonempty_spec = |raw: &String| {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    };

    let resolve_in_pool = || -> Result<Vec<(String, PathBuf)>, String> {
        specs
            .into_par_iter()
            .filter_map(&nonempty_spec)
            .flat_map_iter(|trimmed| {
                resolve_one_spec(trimmed, mp, include_eol, ResolveExecution::Parallel).into_iter()
            })
            .collect::<Result<Vec<_>, _>>()
    };
    match pool_result {
        Ok(pool) => pool.install(resolve_in_pool),
        Err(e) => {
            tracing::warn!(
                error = %e,
                max_threads,
                "rayon ThreadPoolBuilder failed; falling back to sequential kernel resolution"
            );
            specs
                .iter()
                .filter_map(&nonempty_spec)
                .flat_map(|trimmed| {
                    resolve_one_spec(trimmed, mp, include_eol, ResolveExecution::Sequential)
                        .into_iter()
                })
                .collect::<Result<Vec<_>, _>>()
        }
    }
}

/// Resolve the `--cpu-cap` flag into a build reservation plan, rejecting
/// the `KTSTR_BYPASS_LLC_LOCKS` conflict up front so operators see the
/// parse-time error, not an opaque pipeline bail later. Shared by the
/// tarball / source path ([`kernel_build_one`]) and the git path
/// ([`ktstr::cli::resolve_git_kernel`]).
fn resolve_cpu_cap(cpu_cap: Option<usize>) -> Result<Option<cli::CpuCap>, String> {
    if cpu_cap.is_some() && ktstr::bypass_llc_locks_active() {
        return Err(
            "--cpu-cap conflicts with KTSTR_BYPASS_LLC_LOCKS=1; unset one of them. \
             --cpu-cap is a resource contract; bypass disables the contract entirely."
                .to_string(),
        );
    }
    cli::CpuCap::resolve(cpu_cap).map_err(|e| format!("{e:#}"))
}

/// Acquire source, configure, build, and cache a kernel image from a
/// single unified `--kernel` spec.
///
/// The spec parses to a [`ktstr::kernel_path::KernelId`] whose variant
/// selects the source: a `Version` (`6.14.2`, or a `MAJOR.MINOR` prefix
/// resolving to the latest patch) or omitted `--kernel` downloads a
/// tarball; a `Range` (`START..END`) expands against kernel.org's
/// `releases.json` and builds every `stable` / `longterm` release in
/// the inclusive interval; a `Path` builds a local source tree; a `Git`
/// source is fetched and built via [`ktstr::cli::resolve_git_kernel`]; a
/// `CacheKey` is rejected (it names an already-built entry). Range mode
/// collects per-version errors as a best-effort summary — a build
/// failure on one version is reported and the iteration continues, so a
/// stale endpoint doesn't block the rest of the range from caching.
pub(crate) fn kernel_build(
    kernel: Option<String>,
    force: bool,
    clean: bool,
    cpu_cap: Option<usize>,
    extra_kconfig: Option<PathBuf>,
    skip_sha256: bool,
    include_eol: bool,
) -> Result<(), String> {
    ktstr::cli::check_tools(&["make"]).map_err(|e| format!("{e:#}"))?;
    // Read the extra-kconfig fragment ONCE up front so a range
    // expansion doesn't re-read the same file per version (and so a
    // bad path surfaces before any download / build work fires).
    // [`ktstr::cli::read_extra_kconfig`] does the 4-arm error
    // classification (ENOENT/EISDIR/EACCES/UTF-8) and emits an
    // empty-file warning so a 0-byte fragment doesn't silently
    // produce an "extras present but nothing merged" build.
    let extra_content: Option<String> = match extra_kconfig.as_ref() {
        Some(p) => Some(cli::read_extra_kconfig(p, "cargo ktstr")?),
        None => None,
    };

    use ktstr::kernel_path::KernelId;
    // The unified `--kernel` spec parses to a KernelId whose variant
    // selects the source; omitted (`None`) builds the latest stable
    // release via the tarball path. Validate before any I/O so an
    // inverted range surfaces the "swap the endpoints" diagnostic ahead
    // of any download.
    let id = kernel.as_deref().map(KernelId::parse);
    if let Some(id) = &id {
        id.validate().map_err(|e| format!("--kernel {id}: {e}"))?;
    }

    match id {
        Some(KernelId::Range { start, end, .. }) => {
            let versions =
                ktstr::cli::expand_kernel_range(&start, &end, "cargo ktstr", include_eol)
                    .map_err(|e| format!("{e:#}"))?;
            let total = versions.len();
            let mut failures: Vec<(String, String)> = Vec::new();
            for (i, ver) in versions.iter().enumerate() {
                eprintln!("cargo ktstr: [{}/{total}] kernel build {ver}", i + 1);
                if let Err(e) = kernel_build_one(
                    Some(ver.clone()),
                    None,
                    force,
                    clean,
                    cpu_cap,
                    extra_content.as_deref(),
                    skip_sha256,
                ) {
                    eprintln!("cargo ktstr: {ver}: {e}");
                    failures.push((ver.clone(), e));
                }
            }
            if failures.is_empty() {
                Ok(())
            } else {
                // Continue-on-error is the right default for ranges (a
                // stale endpoint shouldn't gate the rest of the cohort);
                // a non-zero exit still flags the cohort as partial and
                // the summary lists each failing version for scraping.
                Err(format!(
                    "kernel build range {start}..{end}: {failed}/{total} \
                     version(s) failed: {names}",
                    failed = failures.len(),
                    names = failures
                        .iter()
                        .map(|(v, _)| v.as_str())
                        .collect::<Vec<_>>()
                        .join(", "),
                ))
            }
        }
        Some(KernelId::Path(path)) => kernel_build_one(
            None,
            Some(path),
            force,
            clean,
            cpu_cap,
            extra_content.as_deref(),
            skip_sha256,
        ),
        Some(KernelId::Version(v)) => kernel_build_one(
            Some(v),
            None,
            force,
            clean,
            cpu_cap,
            extra_content.as_deref(),
            skip_sha256,
        ),
        None => kernel_build_one(
            None,
            None,
            force,
            clean,
            cpu_cap,
            extra_content.as_deref(),
            skip_sha256,
        ),
        Some(KernelId::Git {
            url,
            git_ref,
            ref_kind,
        }) => {
            // Git builds route through the SAME resolver the test path
            // uses (codeload snapshot for GitHub, kind-directed clone
            // otherwise), now threading `kernel build`'s force/clean/
            // cpu_cap/extra_kconfig flags so a git build honors them.
            let resolved_cap = resolve_cpu_cap(cpu_cap)?;
            let fetch_progress = cli::FetchProgress::new();
            let dir = ktstr::cli::resolve_git_kernel(
                &url,
                &git_ref,
                ref_kind,
                "cargo ktstr",
                Some(&fetch_progress),
                force,
                clean,
                resolved_cap,
                extra_content.as_deref(),
            )
            .map_err(|e| format!("build git+{url}#{git_ref}: {e:#}"))?;
            eprintln!("cargo ktstr: kernel cached at {}", dir.display());
            Ok(())
        }
        Some(KernelId::CacheKey(key)) => Err(format!(
            "--kernel {key}: a cache key names an already-built kernel, so \
             there is nothing to build. Pass a version (`6.14.2`), a range \
             (`6.11..6.14`), a source path (`./linux`), or a git source \
             (`git+URL#tag=NAME`). A relative source directory must be \
             spelled `./{key}` to be read as a path, not a cache key. Run \
             `kernel list` to see cached entries.",
        )),
        // For a local package or distro spec, "build" means acquire the
        // prebuilt kernel into the cache (download + extract, or unpack
        // local files) — there is nothing to compile. Validated above,
        // before this match.
        Some(id @ (KernelId::Package { .. } | KernelId::Distro { .. })) => {
            let dir = match &id {
                KernelId::Package { path } => {
                    ktstr::distro::acquire::acquire_package_kernel(std::slice::from_ref(path))
                }
                KernelId::Distro { kind, release } => {
                    ktstr::distro::acquire::acquire_distro_kernel(
                        *kind,
                        release.as_deref(),
                        "cargo ktstr",
                        None,
                    )
                }
                _ => unreachable!("arm guarded to Package | Distro"),
            }
            .map_err(|e| format!("acquire --kernel {id}: {e:#}"))?;
            eprintln!("cargo ktstr: prebuilt kernel cached at {}", dir.display());
            Ok(())
        }
    }
}

/// Build one tarball (explicit version / `MAJOR.MINOR` prefix / latest
/// stable) or a `--kernel <path>` local source tree. Split from
/// [`kernel_build`] so the range loop can reuse the download + cache +
/// build pipeline per resolved version without duplicating it. The
/// git-source path is handled separately by
/// [`ktstr::cli::resolve_git_kernel`].
///
/// `extra_kconfig` is the pre-loaded user fragment from
/// `--extra-kconfig PATH` (the file is read once in [`kernel_build`]
/// before fanning out to per-version invocations). `Some(content)`
/// folds into the cache key suffix via
/// [`ktstr::cache_key_suffix_with_extra`] and into the configure
/// pass via the Cow merge construction in
/// [`ktstr::cli::kernel_build_pipeline`].
fn kernel_build_one(
    version: Option<String>,
    source: Option<PathBuf>,
    force: bool,
    clean: bool,
    cpu_cap: Option<usize>,
    extra_kconfig: Option<&str>,
    skip_sha256: bool,
) -> Result<(), String> {
    let resolved_cap = resolve_cpu_cap(cpu_cap)?;

    let cache = CacheDir::new().map_err(|e| format!("open cache: {e:#}"))?;

    // Temporary directory for tarball/git source extraction.
    let tmp_dir = tempfile::TempDir::new().map_err(|e| format!("create temp dir: {e:#}"))?;

    // Acquire source.
    let client = fetch::shared_client();
    // Progress group for this build: hosts the download bar AND
    // (via `kernel_build_pipeline`) the build phase, so one renderer
    // covers the whole operation. A `--kernel <path>` source adds no
    // fetch bar but the same group still hosts the build bar.
    let fetch_progress = cli::FetchProgress::new();
    let mut acquired = if let Some(ref src_path) = source {
        fetch::local_source(src_path).map_err(|e| format!("{e:#}"))?
    } else {
        // Tarball download: explicit version, prefix, or latest stable.
        let ver = match version {
            Some(v) if fetch::is_major_minor_prefix(&v) => {
                // Major.minor prefix (e.g., "6.12") — resolve latest patch.
                fetch::fetch_version_for_prefix(client, &v, "cargo ktstr")
                    .map_err(|e| format!("{e:#}"))?
            }
            Some(v) => v,
            None => fetch::fetch_latest_stable_version(client, "cargo ktstr")
                .map_err(|e| format!("{e:#}"))?,
        };
        // Check cache before downloading. Cache key folds in the
        // merged-kconfig hash so an `--extra-kconfig` build looks up
        // a distinct slot from a vanilla baked-in-only build —
        // `cache_key_suffix_with_extra(None)` equals
        // `cache_key_suffix()` so the no-extra path is byte-identical
        // to pre-flag behavior.
        let (arch, _) = fetch::arch_info();
        let cache_key = format!(
            "{ver}-tarball-{arch}-kc{}",
            ktstr::cache_key_suffix_with_extra(extra_kconfig),
        );
        if !force && let Some(entry) = cache_lookup(&cache, &cache_key) {
            eprintln!("cargo ktstr: cached kernel found: {}", entry.path.display());
            eprintln!("cargo ktstr: use --force to rebuild");
            return Ok(());
        }
        let result = fetch::download_tarball(
            client,
            &ver,
            tmp_dir.path(),
            "cargo ktstr",
            skip_sha256,
            Some(&fetch_progress),
        );
        let mut acquired = result.map_err(|e| format!("{e:#}"))?;
        // `download_tarball` builds its `cache_key` using the bare
        // `cache_key_suffix()` (see `fetch::download_tarball`).
        // Override with the merged-suffix key we looked up under so
        // the post-build cache store lands at the same slot we'd
        // hit on a re-run.
        acquired.cache_key = cache_key;
        acquired
    };

    // For a `--kernel <path>` source, `local_source` builds
    // `acquired.cache_key` against the bare `cache_key_suffix()` —
    // already shaped `...-kc{baked_hash}`. With `--extra-kconfig` set,
    // lift the `-xkc{extra_hash}` append to
    // [`cli::append_extra_kconfig_suffix`] so the cache lookup +
    // post-build store both target the extras-aware slot. (The tarball
    // path already looked up under the merged key above.)
    if source.is_some() {
        cli::append_extra_kconfig_suffix(&mut acquired.cache_key, extra_kconfig);
    }

    // Check cache for a `--kernel <path>` source (tarball already
    // checked pre-download above).
    if !force
        && source.is_some()
        && !acquired.is_dirty
        && let Some(entry) = cache_lookup(&cache, &acquired.cache_key)
    {
        eprintln!("cargo ktstr: cached kernel found: {}", entry.path.display());
        eprintln!("cargo ktstr: use --force to rebuild");
        return Ok(());
    }

    // `--force` fail-fast pre-check: if tests are actively holding
    // the cache-entry lock, bail with the PID list rather than
    // silently waiting to stomp the in-use entry. The guard drops
    // at the end of this `if` before `kernel_build_pipeline` runs.
    if force {
        let _force_check = cache
            .try_acquire_exclusive_lock(&acquired.cache_key)
            .map_err(|e| format!("{e:#}"))?;
    }

    cli::kernel_build_pipeline(
        &acquired,
        &cache,
        "cargo ktstr",
        clean,
        force,
        source.is_some(),
        resolved_cap,
        extra_kconfig,
        Some(&fetch_progress),
    )
    .map_err(|e| format!("{e:#}"))?;

    Ok(())
}

/// Look up a cache key, checking local first, then remote (if enabled).
fn cache_lookup(cache: &CacheDir, cache_key: &str) -> Option<CacheEntry> {
    cli::cache_lookup(cache, cache_key, "cargo ktstr")
}

#[cfg(test)]
mod tests {
    use super::*;

    const POOL_FAILURE_THREAD_PROBE_ENV: &str = "KTSTR_KERNEL_POOL_FAILURE_THREAD_PROBE";
    const POOL_FAILURE_THREAD_PROBE_MARKER: &str = "kernel-pool-failure-thread-probe-ok";

    // ---------------------------------------------------------------
    // format_built_age — cache-hit log line age suffix
    // ---------------------------------------------------------------
    //
    // The helper renders the persisted `built_at` ISO-8601 stamp as
    // `, built {age} ago`. It must produce the empty string on
    // unparseable inputs (so the cache-hit line still renders
    // gracefully without a malformed suffix), and must include a
    // leading comma+space prefix on the success path so the call
    // site can splice it directly into `(cache_key{age})`.

    #[test]
    fn format_built_age_unparseable_returns_empty_string() {
        // Malformed timestamp must not panic and must not yield a
        // half-formed suffix. The cache-hit log line stays valid
        // even when metadata is corrupt: `(cache_key)` with no
        // age portion.
        assert_eq!(format_built_age("not-a-timestamp"), "");
        assert_eq!(format_built_age(""), "");
        // Almost-valid RFC 3339 (missing trailing Z) must also
        // collapse to empty rather than returning a partial.
        assert_eq!(format_built_age("2026-01-02T03:04:05"), "");
    }

    #[test]
    fn format_built_age_future_timestamp_returns_empty_string() {
        // A timestamp far in the future fails
        // `duration_since` because the build moment hasn't
        // occurred yet relative to local clock — clock skew on a
        // shared cache between two hosts can produce this. The
        // helper collapses to empty rather than rendering
        // `built -2h ago` or panicking.
        assert_eq!(format_built_age("9999-12-31T23:59:59Z"), "");
    }

    #[test]
    fn format_built_age_past_timestamp_includes_leading_comma_and_seconds() {
        // A reachable past timestamp must produce the
        // `, built ... ago` shape. We don't pin the exact age
        // string (it depends on `SystemTime::now()` at test time),
        // but assert the structural invariants:
        //   * non-empty
        //   * starts with `, built ` (the leading comma+space lets
        //     the caller splice into `(cache_key{age})` without a
        //     conditional separator)
        //   * ends with ` ago` (the trailing keyword renders the
        //     duration as relative-past in human language)
        let one_hour_ago = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
            .saturating_sub(3600);
        let timestamp = humantime::format_rfc3339(
            std::time::UNIX_EPOCH + std::time::Duration::from_secs(one_hour_ago),
        )
        .to_string();
        let age = format_built_age(&timestamp);
        assert!(
            age.starts_with(", built "),
            "age suffix must start with the splice prefix `, built `, got {age:?}",
        );
        assert!(
            age.ends_with(" ago"),
            "age suffix must end with the relative-past keyword ` ago`, got {age:?}",
        );
    }

    // ---------------------------------------------------------------
    // resolve_path_kernel — Path-spec error diagnostics
    // ---------------------------------------------------------------
    //
    // The diagnostic shape `--kernel {raw}: {inner}. {KTSTR_KERNEL_HINT}`
    // is what the user sees when `--kernel /path/...` fails. The
    // raw input must appear verbatim so a typo names the exact
    // string they passed; the inner error must come from the
    // shared resolution pipeline (currently
    // `cli::resolve_kernel_dir_to_entry`); the hint must guide
    // the user toward the supported `--kernel` shapes.

    /// Nonexistent path: the source-tree validation
    /// (`Makefile + Kconfig` exist) fails inside
    /// [`cli::resolve_kernel_dir_to_entry`] (via
    /// `acquire_local_source_tree`), and `resolve_path_kernel`
    /// re-wraps the error with the user's raw input + the
    /// standard hint. Pins the `--kernel {raw}: ...` prefix and
    /// the trailing hint marker so a regression that dropped
    /// either surfaces here.
    #[test]
    fn resolve_path_kernel_nonexistent_returns_actionable_error() {
        let raw = "/this/path/should/not/exist/under/test";
        let result = resolve_path_kernel(std::path::Path::new(raw), raw);
        let err = result.expect_err("nonexistent path must surface as Err");
        assert!(
            err.contains(&format!("--kernel {raw}")),
            "error must lead with `--kernel {{raw_input}}:` so a typo \
             names the exact string the user passed. got: {err}",
        );
        // The hint string carries the documented `--kernel` value
        // shapes; pin its presence rather than its prose so a
        // future hint rewrite doesn't break this test.
        assert!(
            err.contains(ktstr::KTSTR_KERNEL_HINT),
            "error must end with KTSTR_KERNEL_HINT so the user sees \
             the supported `--kernel` shapes. got: {err}",
        );
    }

    /// Empty tempdir (real directory, no Makefile or Kconfig):
    /// `acquire_local_source_tree` rejects it as "not a kernel
    /// source tree" and `resolve_path_kernel` re-wraps with the
    /// user's raw input + hint. Distinct from
    /// `resolve_path_kernel_nonexistent_returns_actionable_error`
    /// because the inner error path differs (existence-check
    /// pass, content-shape fail) — both must surface the same
    /// outer wrapping.
    #[test]
    fn resolve_path_kernel_empty_tempdir_returns_not_a_source_tree_error() {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let raw = tmp.path().display().to_string();
        let result = resolve_path_kernel(tmp.path(), &raw);
        let err = result.expect_err("empty tempdir must surface as Err");
        assert!(
            err.contains(&format!("--kernel {raw}")),
            "error must lead with `--kernel {{raw_input}}:`. got: {err}",
        );
        assert!(
            err.contains("not a kernel source tree"),
            "error must include the `not a kernel source tree` phrase \
             from `acquire_local_source_tree`'s diagnostic. got: {err}",
        );
        assert!(
            err.contains(ktstr::KTSTR_KERNEL_HINT),
            "error must end with KTSTR_KERNEL_HINT. got: {err}",
        );
    }

    // ---------------------------------------------------------------
    // resolve_one — defensive Range arm (internal caller contract)
    // ---------------------------------------------------------------
    //
    // `resolve_one` is only ever called by `resolve_kernel_set` after
    // it fans every Range out to per-version `Version` ids, so the
    // Range arm is the sole arm reachable WITHOUT real I/O — it is a
    // programming-error guard, not a user-facing path. Every other
    // arm (Path / Version / CacheKey / Git) drives canonicalize +
    // build / cache lookup / clone and is flagged not-host-testable.

    /// The Range arm must surface the exact internal-error string —
    /// `internal:` discriminator plus the `expand_kernel_range`
    /// remediation pointer — so a developer who wires a Range into
    /// `resolve_one` directly is pointed at the wrong call site. Pins
    /// the full interpolated message (start/end splice into the
    /// `{start}..{end}` placeholders) by exact equality, not substring.
    #[test]
    fn resolve_one_range_arm_returns_internal_caller_contract_error() {
        use ktstr::kernel_path::KernelId;
        let err = resolve_one(
            KernelId::Range {
                start: "6.10".to_string(),
                end: "6.12".to_string(),
                syntax_inclusive: false,
            },
            None,
        )
        .expect_err("Range arm must be Err — it is the defensive internal-error path");
        assert_eq!(
            err,
            "internal: resolve_one called with Range 6.10..6.12; \
             caller must expand Range via `expand_kernel_range` and \
             call `resolve_one` per version",
        );
    }

    // ---------------------------------------------------------------
    // resolve_versions_parallel — intra-range concurrency contract
    // ---------------------------------------------------------------
    //
    // The Range arm of `resolve_one_spec` resolves expanded versions
    // through this helper. Its two guarantees — output order == input
    // order, and every version attempted (no inner short-circuit) —
    // are what the doc asserts and what a downstream `resolved[0] ->
    // KTSTR_KERNEL` choice relies on. Real Range resolves need network
    // + build, so the per-version resolver is injected here to exercise
    // the concurrency contract in isolation.

    /// `Vec::into_par_iter` is indexed, so `collect` restores input
    /// order regardless of completion order. A regression that swapped
    /// the helper to an unindexed adaptor (e.g. `par_bridge`) would
    /// shuffle the output and fail this.
    #[test]
    fn resolve_versions_parallel_preserves_input_order() {
        let versions: Vec<String> = (0..64).map(|i| format!("6.{i}")).collect();
        let out = resolve_versions_parallel(versions.clone(), |ver| {
            Ok((
                ver.to_string(),
                std::path::PathBuf::from(format!("/fake/{ver}")),
            ))
        });
        let labels: Vec<String> = out.into_iter().map(|r| r.unwrap().0).collect();
        assert_eq!(
            labels, versions,
            "parallel resolve must return results in input (version) order"
        );
    }

    /// The helper's `collect` gathers EVERY version's `Result` — it
    /// does NOT short-circuit on the first Err (that is the caller's
    /// outer `collect::<Result<_, _>>`'s job). So a failing version
    /// leaves all N resolves attempted and its Err at its input index.
    #[test]
    fn resolve_versions_parallel_attempts_every_version_and_keeps_error_position() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let versions: Vec<String> = (0..32).map(|i| format!("6.{i}")).collect();
        let attempts = AtomicUsize::new(0);
        let out = resolve_versions_parallel(versions.clone(), |ver| {
            attempts.fetch_add(1, Ordering::Relaxed);
            if ver == "6.7" {
                Err(format!("boom {ver}"))
            } else {
                Ok((ver.to_string(), std::path::PathBuf::from(ver)))
            }
        });
        assert_eq!(
            attempts.load(Ordering::Relaxed),
            versions.len(),
            "every version must be attempted; the inner collect does not short-circuit"
        );
        assert_eq!(out.len(), versions.len());
        assert!(
            out.iter().enumerate().all(|(i, r)| (i == 7) == r.is_err()),
            "only the 6.7 slot (input index 7) is Err; order preserved"
        );
    }

    #[test]
    fn resolve_versions_sequential_preserves_order_and_calling_thread() {
        let caller = std::thread::current().id();
        let versions: Vec<String> = (0..16).map(|i| format!("6.{i}")).collect();
        let out = resolve_versions_sequential(versions.clone(), |ver| {
            assert_eq!(
                std::thread::current().id(),
                caller,
                "sequential Range fallback escaped onto a worker"
            );
            Ok((ver.to_string(), std::path::PathBuf::from(ver)))
        });
        let labels: Vec<String> = out.into_iter().map(|result| result.unwrap().0).collect();
        assert_eq!(labels, versions);
    }

    /// Exercise the real outer filtering / validation / error path with
    /// a deterministically failed pool builder in an isolated process.
    /// The child task count catches any fallback to a global parallel
    /// iterator without depending on test order in this large binary.
    #[test]
    fn resolve_specs_pool_build_failure_is_sequential() {
        if std::env::var_os(POOL_FAILURE_THREAD_PROBE_ENV).is_some() {
            return;
        }
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .env(POOL_FAILURE_THREAD_PROBE_ENV, "1")
            .arg("--exact")
            .arg("kernel::tests::resolve_specs_pool_build_failure_is_sequential_child")
            .arg("--nocapture")
            .arg("--test-threads=1")
            .output()
            .expect("spawn isolated kernel resolver pool-failure probe");
        let transcript = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            output.status.success(),
            "isolated kernel resolver pool-failure probe failed:\n{transcript}"
        );
        assert!(
            transcript.contains(POOL_FAILURE_THREAD_PROBE_MARKER),
            "isolated helper did not run (wrong libtest exact name?):\n{transcript}"
        );
    }

    #[test]
    fn resolve_specs_pool_build_failure_is_sequential_child() {
        if std::env::var_os(POOL_FAILURE_THREAD_PROBE_ENV).is_none() {
            return;
        }
        let specs = vec!["   ".to_string(), " 6.20..6.10 ".to_string()];
        let progress = ktstr::cli::FetchProgress::new();
        let requested_threads = std::cell::Cell::new(0);
        let task_count = || {
            std::fs::read_dir("/proc/self/task")
                .expect("read /proc/self/task")
                .count()
        };
        let before = task_count();
        let err = resolve_specs_parallel_with_pool_builder(&specs, &progress, false, |threads| {
            requested_threads.set(threads);
            Err("forced pool build failure".to_string())
        })
        .expect_err("inverted Range must retain its validation error");
        let after = task_count();
        assert!(
            requested_threads.get() >= 1,
            "pool builder seam was not called"
        );
        assert!(
            err.contains("6.20") && err.contains("6.10"),
            "sequential fallback lost the original validation error: {err}"
        );
        assert_eq!(
            after, before,
            "failed bounded-pool construction initialized retained global workers: \
             before={before}, after={after}"
        );
        eprintln!("{POOL_FAILURE_THREAD_PROBE_MARKER}");
    }

    // ---------------------------------------------------------------
    // canonicalize_cache_dir — both arms of the single `unwrap_or`
    // ---------------------------------------------------------------
    //
    // `canonicalize_cache_dir` defends against a relative
    // `KTSTR_CACHE_DIR=./cache` reaching the child with a
    // cwd-divergent path: `canonicalize` resolves a real entry to an
    // absolute, symlink-resolved path; a failure (the documented
    // remove-between-lookup-and-export race) falls back to the input
    // path verbatim rather than bailing. Both arms are pure std::fs,
    // host-isolable, and currently untested.

    /// Existing directory: the Ok arm returns the std-canonicalized
    /// (absolute, symlink-resolved) form, NOT the input verbatim. To
    /// genuinely distinguish the Ok arm from the `unwrap_or` fallback,
    /// the input is a SYMLINK to a real dir — so the resolved form
    /// (the real dir) differs from the input (the symlink path). A
    /// plain already-canonical tempdir cannot exercise this divergence
    /// on a host whose tempdir has no symlink component, so the
    /// fallback (returning the input verbatim) would pass undetected.
    #[test]
    fn canonicalize_cache_dir_existing_dir_returns_absolute_canonical_path() {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let real = tmp.path().join("real_cache");
        std::fs::create_dir(&real).expect("create real dir");
        let link = tmp.path().join("link_cache");
        std::os::unix::fs::symlink(&real, &link).expect("create symlink");
        let resolved = std::fs::canonicalize(&real).expect("canonicalize real dir");

        let got = canonicalize_cache_dir(link.clone());
        assert_eq!(
            got, resolved,
            "Ok arm must return the symlink-resolved real dir, not the input",
        );
        assert_ne!(
            got, link,
            "must NOT fall through to the unwrap_or fallback (input verbatim)",
        );
        assert!(got.is_absolute(), "canonicalized path must be absolute");
    }

    /// Nonexistent directory: `canonicalize` fails and the
    /// `unwrap_or` fallback returns the original `PathBuf` unchanged.
    /// Exact-equality: the output is byte-identical to the input
    /// (proves the documented fall-back-rather-than-bail behaviour,
    /// not a panic and not a transformed path).
    #[test]
    fn canonicalize_cache_dir_nonexistent_falls_back_to_input_path() {
        let missing =
            std::path::PathBuf::from("/this/cache/dir/does/not/exist/ktstr-canonicalize-test");
        assert_eq!(canonicalize_cache_dir(missing.clone()), missing);
    }

    // ---------------------------------------------------------------
    // resolve_kernel_set — preflight gate fires before any I/O
    // ---------------------------------------------------------------
    //
    // `resolve_kernel_set` runs `preflight_collision_check(specs)?` as
    // its first statement — before the bounded rayon ThreadPool is
    // built and before any `resolve_one` I/O. `preflight_collision_check`
    // is unit-tested directly in `wire_format.rs`; these two tests pin
    // the INTEGRATION — that `resolve_kernel_set` wires it as the first
    // gate, so a colliding pair or an inverted range errors locally
    // with zero download / build / clone contact.

    /// Two specs whose sanitized labels collide (`6.14.2` → Version
    /// label, `6-14-2` → CacheKey label, both sanitize to
    /// `kernel_6_14_2`) must abort at the preflight gate. The
    /// preflight-distinctive prefix and the shared sanitized id pin
    /// the gate: a regression that ran preflight AFTER resolve would
    /// download / build before erroring (and could surface the
    /// post-resolve `detect_label_collisions` wording, which has no
    /// `pre-flight` prefix), failing the first assertion.
    #[test]
    fn resolve_kernel_set_colliding_specs_fail_at_preflight_before_io() {
        let specs = vec!["6.14.2".to_string(), "6-14-2".to_string()];
        let err = resolve_kernel_set(&specs, false)
            .expect_err("colliding sanitized labels must abort at preflight");
        assert!(
            err.contains("pre-flight check found collision"),
            "must be the preflight diagnostic, not the post-resolve one. got: {err}",
        );
        assert!(
            err.contains("kernel_6_14_2"),
            "must name the shared sanitized id. got: {err}",
        );
    }

    /// An inverted Range (`6.15..6.14`) fails `KernelId::validate`
    /// inside the preflight gate, before `expand_kernel_range` would
    /// fire its `releases.json` fetch. Pins that `resolve_kernel_set`
    /// rejects the inversion with zero network contact: a regression
    /// that moved validation after `expand_kernel_range` would attempt
    /// a fetch instead of erroring locally. Asserts the `--kernel`
    /// framing (`--kernel {id}: {e}` from the preflight gate) and the
    /// inverted-range diagnostic.
    #[test]
    fn resolve_kernel_set_inverted_range_fails_validation_before_io() {
        let specs = vec!["6.15..6.14".to_string()];
        let err = resolve_kernel_set(&specs, false)
            .expect_err("inverted range must fail validation pre-resolve");
        assert!(
            err.contains("--kernel"),
            "must carry `--kernel` framing. got: {err}",
        );
        assert!(
            err.contains("inverted kernel range"),
            "must surface the inversion diagnostic. got: {err}",
        );
    }
}
