//! Kernel source acquisition: tarball download, GitHub codeload
//! snapshot, git clone, local tree.
//!
//! The acquisition entry points each return an [`AcquiredSource`]
//! carrying the source directory, cache key, and metadata the caller
//! needs to proceed to configuration and build: [`download_tarball`]
//! (kernel.org stable/RC), `download_github_archive` (a GitHub codeload
//! commit snapshot), `git_clone_kinded` (a kind-directed shallow clone
//! that dispatches to `git_clone_tag` / [`git_clone`]), and
//! [`local_source`] (an on-disk tree).

use std::collections::{HashSet, VecDeque};
use std::io::Read;
use std::net::{SocketAddr, ToSocketAddrs};
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};
use std::{hash::BuildHasher, hash::Hasher};

use ::gix;
use anyhow::{Context, Result, anyhow};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64_STANDARD};
use md5::{Digest as _, Md5};
use reqwest::Url;
use reqwest::blocking::{Client, ClientBuilder};
use sha2::{Digest, Sha256};

#[path = "../build_support/gix_policy.rs"]
mod gix_policy;

/// Process-wide [`reqwest::blocking::Client`] lazily initialized on
/// first access via [`shared_client`]. Keeping a single `Client`
/// instance across the fetch-family reuses its TCP connection pool
/// and TLS session cache across repeated calls to the same host
/// within a CLI run. Cross-host fetches in the same run still
/// re-handshake because reqwest's connection pool keys on host.
static SHARED_CLIENT: OnceLock<Client> = OnceLock::new();

/// Process interruption observed by every in-process source-acquisition
/// operation. cargo-ktstr's signal handler updates this through the
/// compatibility entry point [`set_git_operation_interrupted`]; binaries
/// retaining the default signal disposition terminate directly and do not
/// need the bridge.
static SOURCE_OPERATION_INTERRUPTED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Bridge a process-level signal epoch into in-process source acquisition.
///
/// This performs one atomic store and is safe to call from
/// cargo-ktstr's SIGINT/SIGTERM handler. `false` starts a new guard epoch;
/// `true` stops active and subsequent HTTP request/backoff/read/extract work
/// as well as gix ref-map/receive/checkout work.
#[doc(hidden)]
pub fn set_git_operation_interrupted(interrupted: bool) {
    SOURCE_OPERATION_INTERRUPTED.store(interrupted, Ordering::SeqCst);
}

/// Read the process-level source-acquisition interruption bridge.
#[doc(hidden)]
pub fn git_operation_interrupted() -> bool {
    SOURCE_OPERATION_INTERRUPTED.load(Ordering::SeqCst)
}

fn ensure_source_operation_not_interrupted(phase: &str) -> Result<()> {
    if SOURCE_OPERATION_INTERRUPTED.load(Ordering::Acquire) {
        anyhow::bail!("source acquisition interrupted {phase}");
    }
    Ok(())
}

fn interrupted_io_error(phase: &str) -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::Interrupted,
        format!("source acquisition interrupted {phase}"),
    )
}

/// Connect-phase timeout for [`shared_client`]: bounds the TCP + TLS
/// handshake before reqwest gives up on a dead peer or blackholed route.
const SHARED_CLIENT_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Default blocking-operation timeout for [`shared_client`].
///
/// Reqwest's blocking client applies this budget to connection, request-write,
/// and response-read waits. Keeping it explicit here means a response that
/// delivered headers but then stopped producing body bytes cannot wedge a
/// caller that forgot a per-request override. Small metadata/index/probe
/// requests also pass this value explicitly; large archive and package
/// downloads override it with [`DOWNLOAD_REQUEST_READ_TIMEOUT`].
const SMALL_RESPONSE_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

/// Return the process-wide shared [`reqwest::blocking::Client`]. First
/// call constructs it via `Client::builder()` with
/// [`SHARED_CLIENT_CONNECT_TIMEOUT`] and
/// [`SMALL_RESPONSE_REQUEST_TIMEOUT`] applied; every subsequent call returns
/// a reference to the same instance. This helper is for top-level CLI entries
/// that want the default client.
///
/// Tests that need to verify a network round-trip (rather than a
/// cache hit) must NOT pass `shared_client()` to a cache-routed
/// helper (`cached_releases`, `cached_releases_with`,
/// [`fetch_latest_stable_version`], [`fetch_version_for_prefix`]) —
/// `RELEASES_CACHE` may already be populated by a peer test, in
/// which case the helper returns cached data and the network is
/// never touched. Construct a local `Client` and pass it to the
/// cache-routed helper to skip the cache; the pointer-equality gate
/// in `cached_releases_with` routes a non-singleton client to a
/// direct `fetch_releases` call against `RELEASES_URL` (the
/// production URL — the bypass skips the cache, NOT the URL). For
/// full URL injection (e.g. localhost mock server testing), call
/// either `fetch_releases` directly with the mock URL — see
/// `fetch_releases_against_localhost_mock_returns_parsed` — or use
/// the cache-aware seam `cached_releases_with_url`, which routes
/// the non-singleton bypass branch through the supplied URL while
/// preserving the singleton/cache routing identical to
/// `cached_releases_with`.
///
/// # Panics
///
/// Panics on the first call if `Client::builder().build()` fails to
/// construct a client. Documented failure modes include TLS backend
/// initialization (e.g. rustls/native-tls subsystem unreachable) and
/// system-resolver config load failure; both are treated as setup
/// bugs rather than runtime errors. The
/// `expect` here, rather than propagating the error, mirrors the
/// inherited behavior of `reqwest::blocking::Client::new()` (which
/// is itself an infallible wrapper around `builder().build().expect`).
pub fn shared_client() -> &'static Client {
    SHARED_CLIENT.get_or_init(|| {
        default_http_client_builder()
            .build()
            .expect("build shared reqwest client")
    })
}

/// Construct the common reqwest client builder used by the process-wide
/// client and by retry-only DNS-edge clients.
///
/// Keeping the builder policy in one place makes a routed retry retain the
/// shared client's connect timeout, TLS defaults, and automatic system-proxy
/// handling. A routed client adds only a per-origin DNS override; it does not
/// rewrite the URL, Host header, TLS SNI, or proxy configuration.
fn default_http_client_builder() -> ClientBuilder {
    Client::builder()
        .connect_timeout(SHARED_CLIENT_CONNECT_TIMEOUT)
        .timeout(SMALL_RESPONSE_REQUEST_TIMEOUT)
}

/// Process-wide cache of the parsed `releases.json` payload.
/// Populated by [`cached_releases_with`] on its first successful
/// singleton-path fetch; every subsequent singleton call returns a
/// clone of the cached vector without re-issuing the HTTP request.
/// Lifetime matches the process — `releases.json` does not change
/// underneath a single CLI invocation, so a per-process cache
/// cannot serve stale data in any way the user would notice.
///
/// Failures are NOT cached: a transient kernel.org outage that
/// errors the first call must allow a later caller to retry, since
/// the underlying network condition may have cleared. Storing
/// `Vec<Release>` rather than `Result<Vec<Release>>` enforces this
/// at the type level — there's no way to populate the cache with
/// a failure.
///
/// Companion to [`SHARED_CLIENT`]: both amortize per-invocation
/// network cost across the resolve pipeline. Without this cache,
/// `cargo ktstr test --kernel 6.10..6.12 --kernel 6.14..6.16`
/// fetches `releases.json` twice — once per Range spec — under
/// the rayon par_iter that drives `resolve_kernel_set`. With
/// the cache the first Range to reach `expand_kernel_range`
/// populates the slot; the second observes the populated slot
/// and skips the network entirely.
static RELEASES_CACHE: OnceLock<Vec<Release>> = OnceLock::new();

/// Cache for the gregkh stable-mirror release tags — the `X.Y.Z`
/// version strings parsed from its `refs/tags/vX.Y.Z` advertisement.
/// Companion to [`RELEASES_CACHE`]: `--include-eol` may expand several
/// `A..B` ranges under one `resolve_kernel_set`, and each would
/// otherwise re-ls-remote the mirror. Populated on the first successful
/// enumeration; a failed ls-remote leaves it empty so the next caller
/// retries (`Vec`, not `Result`, mirroring `RELEASES_CACHE`).
static STABLE_TAGS_CACHE: OnceLock<Vec<String>> = OnceLock::new();

/// Fetch `releases.json` via the process-wide [`shared_client`],
/// routing through [`RELEASES_CACHE`].
///
/// Thin wrapper for callers that don't already thread a `&Client`
/// — top-level CLI entries like [`crate::cli::expand_kernel_range`]
/// (under the rayon-driven `cargo ktstr` resolve pipeline) and
/// `crate::cli::fetch_active_prefixes` (the EOL-annotation pass).
/// Caching, race semantics, and fault-injection routing are all
/// documented on [`cached_releases_with`].
pub(crate) fn cached_releases() -> Result<Vec<Release>> {
    cached_releases_with(shared_client())
}

/// Pointer-equality against the [`OnceLock`]-backed
/// [`shared_client`] singleton is the correct predicate because
/// `shared_client()` returns a stable `&'static Client` address.
/// The [`cached_releases_with`] gate uses this predicate to
/// decide whether to consult [`RELEASES_CACHE`]: the singleton
/// hits the cache, every other (test-constructed) `Client`
/// bypasses it and exercises the underlying [`fetch_releases`]
/// path.
///
/// Caveat: `shared_client().clone()` produces a distinct
/// `Client` at a different address even though it shares the
/// singleton's connection pool via the inner `Arc`, so the
/// clone bypasses the cache. Always pass `shared_client()`
/// directly — never a clone — when cache routing is desired.
///
/// Side-effect-free when [`SHARED_CLIENT`] is uninitialized:
/// no client can equal a not-yet-allocated singleton, so we
/// return `false` without triggering `get_or_init` — tests
/// that pass a local `Client` before any production code path
/// has touched the singleton skip the construction entirely.
fn is_shared_client(client: &Client) -> bool {
    match SHARED_CLIENT.get() {
        Some(singleton) => std::ptr::eq(client, singleton),
        None => false,
    }
}

/// Unified cache-aware entry point for `releases.json`. Routes
/// the process-wide [`shared_client`] singleton through
/// [`RELEASES_CACHE`]; any other (test-constructed) `Client`
/// bypasses [`RELEASES_CACHE`] and calls [`fetch_releases`] with
/// [`RELEASES_URL`] directly — the cache is skipped but the
/// production URL is used.
///
/// Used by every in-file caller that already threads a `&Client`
/// — [`fetch_latest_stable_version`], [`fetch_version_for_prefix`],
/// [`latest_in_series`] — so production callers reuse
/// [`RELEASES_CACHE`] and tests still get cache-bypass via the
/// pointer-equality gate. [`cached_releases`] is the no-`Client`
/// wrapper for top-level CLI entries.
///
/// Tests that need URL injection on the bypass branch (e.g.
/// localhost mock server testing) call
/// [`cached_releases_with_url`] directly with their mock URL —
/// the URL-injectable form preserves identical routing
/// semantics. This wrapper is the production entry point and
/// pins the URL to [`RELEASES_URL`]; production code MUST go
/// through this wrapper. A singleton call with a non-RELEASES_URL
/// would otherwise populate [`RELEASES_CACHE`] with
/// non-production data and corrupt every later production
/// call — the singleton-path branch in
/// [`cached_releases_with_url`] guards against this in both
/// dev (`debug_assert!`) and release builds (fall back to
/// bypass), but routing every production call through this
/// wrapper makes the misuse impossible by construction.
/// Caching, race semantics, and the bypass-vs-cache routing
/// are fully documented on [`cached_releases_with_url`].
fn cached_releases_with(client: &Client) -> Result<Vec<Release>> {
    cached_releases_with_url(client, RELEASES_URL)
}

/// URL-injectable form of [`cached_releases_with`]. Production
/// always reaches this through the [`cached_releases_with`]
/// wrapper, which pins `url` to [`RELEASES_URL`]; the explicit
/// `url` parameter exists so the bypass-branch test can route
/// the non-singleton path through a localhost
/// [`std::net::TcpListener`]-backed mock instead of hitting real
/// kernel.org. Without this seam, the bypass test would either
/// (a) require a real network round-trip on every run, or
/// (b) accept a 5s timeout penalty on offline hosts to surface
/// `Err` as a bypass-confirmation signal — both costs the seam
/// eliminates.
///
/// Cache contract is identical to [`cached_releases_with`]:
/// non-singleton clients bypass [`RELEASES_CACHE`] and call
/// [`fetch_releases`] with `url`; the singleton routes through
/// the cache only when `url == RELEASES_URL` (consulting via
/// `OnceLock::get`, populating via `OnceLock::set` on miss). A
/// singleton call with a non-RELEASES_URL trips the
/// `debug_assert!` in dev builds and falls back to the bypass
/// behavior in release builds — fetches directly via `url`,
/// returns the result, never touches [`RELEASES_CACHE`]. The
/// cache only ever stores data fetched from the singleton +
/// RELEASES_URL combination, so a test that injects a mock URL
/// on either branch cannot pollute the production cache.
///
/// Failures are propagated without populating [`RELEASES_CACHE`],
/// so a transient kernel.org outage on the first call lets the
/// next caller retry. Storing `Vec<Release>` (not
/// `Result<Vec<Release>>`) enforces this at the type level.
///
/// Concurrent population on the singleton path is safe via the
/// `OnceLock::set` race: the loser's `set` returns `Err(clone)`
/// (the cloned vector that was passed in is moved back), the
/// returned `Err` is discarded via `let _ = …`, and the loser
/// returns its own original `fresh` vector. Both winner and
/// loser return content-equivalent data since both fetched the
/// same `releases.json`. Worst case under concurrent first
/// calls: both callers issue the network round-trip, only one
/// populates [`RELEASES_CACHE`]; every later call — from any
/// thread — observes the populated slot via the `get` fast-path
/// and skips the network.
fn cached_releases_with_url(client: &Client, url: &str) -> Result<Vec<Release>> {
    // Non-singleton clients bypass the cache (test fault injection).
    if !is_shared_client(client) {
        return fetch_releases(client, url);
    }
    // Cache-poison guard: the singleton path populates
    // RELEASES_CACHE on miss. A test author that mistakenly
    // passes a non-production URL with shared_client() would
    // fill the cache with non-production data and corrupt every
    // later production call (which reaches the cache via
    // get-fast-path). Catch the misuse at debug-build time —
    // production callers always thread RELEASES_URL through the
    // `cached_releases_with` wrapper, so the assertion is a
    // no-op for them; only a future test author wiring this
    // function up with shared_client() and a mock URL would trip
    // it.
    debug_assert!(
        url == RELEASES_URL,
        "cached_releases_with_url: shared_client() must use RELEASES_URL \
         to avoid RELEASES_CACHE pollution — got url={url:?}, expected \
         RELEASES_URL ({RELEASES_URL:?}). Tests that need URL injection \
         must pass a non-singleton Client (which takes the bypass branch \
         above and never touches the cache).",
    );
    // Release-build guard: `debug_assert!` is stripped in
    // optimized builds, so a non-RELEASES_URL on the singleton
    // path would otherwise reach the populate-on-miss path below
    // and persistently poison RELEASES_CACHE for every later
    // production caller. Mirror the bypass-branch behavior
    // (fetch directly, do not touch the cache) so the misuse
    // degrades to a slow per-call fetch instead of a permanently
    // wrong cache. The debug_assert above still fires loudly in
    // dev builds; this branch only catches the misuse that
    // slipped through to release.
    if url != RELEASES_URL {
        return fetch_releases(client, url);
    }
    if let Some(cached) = RELEASES_CACHE.get() {
        return Ok(cached.clone());
    }
    let fresh = fetch_releases(client, url)?;
    // Race-loss: `set` returns `Err(clone)` carrying back the
    // clone we passed in; we discard it and return the original
    // `fresh` below. See the rustdoc above for full semantics.
    let _ = RELEASES_CACHE.set(fresh.clone());
    Ok(fresh)
}

/// Downloaded/cloned kernel source ready for building.
#[non_exhaustive]
pub struct AcquiredSource {
    /// Path to the kernel source directory.
    pub source_dir: PathBuf,
    /// Cache key for this source (e.g. "6.14.2-tarball-x86_64-kc{kconfig_hash}").
    pub cache_key: String,
    /// Version string if known (e.g. "6.14.2", "6.15-rc3").
    pub version: Option<String>,
    /// How the source was acquired, with per-variant payload
    /// (git hash/ref for `Git`, source tree path and git hash for
    /// `Local`).
    pub kernel_source: crate::cache::KernelSource,
    /// Whether the source is a temporary directory that should be
    /// cleaned up after building.
    pub is_temp: bool,
    /// For local sources: whether the working tree is dirty.
    /// Dirty trees must not be cached.
    pub is_dirty: bool,
    /// For local sources: whether the source is an actual git
    /// repository. `true` when `gix::discover` succeeded and the
    /// crate could compute index + worktree dirty state; `false`
    /// for non-git source trees (tarball-extracted, rsync'd,
    /// hand-assembled) where dirty detection is impossible and
    /// the source is always cache-skipped pessimistically. Lets
    /// the cache-skip hint branch on whether `commit` / `stash`
    /// are actionable remediations (they aren't for non-git
    /// sources).
    ///
    /// For non-local sources (tarball, git clone) the field is
    /// set to `true` by convention — these paths are always
    /// `is_dirty = false`, so the cache-skip branch that reads
    /// `is_git` is never reached and the value is inert. Pinning
    /// to `true` (rather than leaving the field meaningless)
    /// keeps the invariant "is_git is meaningful only when
    /// is_dirty is true, but always set" so a future code path
    /// that reaches `is_git` outside the cache-skip context does
    /// not trip on an `is_git = false` under a known-good source.
    pub is_git: bool,
}

/// Target architecture string and boot image name.
pub fn arch_info() -> (&'static str, &'static str) {
    #[cfg(target_arch = "x86_64")]
    {
        ("x86_64", "bzImage")
    }
    #[cfg(target_arch = "aarch64")]
    {
        ("aarch64", "Image")
    }
}

/// Parse a version string into its major version for URL construction.
///
/// "6.14.2" -> 6, "6.15-rc3" -> 6.
fn major_version(version: &str) -> Result<u32> {
    let major_str = version
        .split('.')
        .next()
        .ok_or_else(|| anyhow!("invalid version: {version}"))?;
    major_str
        .parse::<u32>()
        .with_context(|| format!("invalid major version in {version}"))
}

/// Determine if a version string represents an RC release.
///
/// RC releases use a different URL pattern and gzip compression
/// (vs xz for stable).
fn is_rc(version: &str) -> bool {
    version.contains("-rc")
}

/// One (`moniker`, `version`) row from kernel.org's `releases.json`.
///
/// A named struct instead of a bare `(String, String)` tuple so every
/// call site reads its field by name (`r.moniker`, `r.version`) rather
/// than positional destructuring — the two strings are trivially
/// swappable at a tuple-destructure call site, and a silent swap
/// would mis-drive `is_skippable_release_moniker` while the
/// now-misnamed "moniker" string flows into `version_prefix`
/// downstream. Naming the fields removes that class of bug at the
/// type-checker level and shows up in IDE hints on every iteration
/// site.
///
/// Both fields are owned `String` (not `&str`) because the values are
/// parsed out of a `reqwest::Response` body whose lifetime ends when
/// `fetch_releases` returns; downstream callers iterate the vector
/// long after that borrow would dangle.
#[derive(Clone, Debug)]
pub(crate) struct Release {
    /// releases.json `moniker` field — stable / longterm / mainline /
    /// linux-next / etc. Consumed by
    /// [`is_skippable_release_moniker`] and by
    /// [`fetch_latest_stable_version`]'s stable/longterm filter.
    pub moniker: String,
    /// releases.json `version` field — e.g. `"6.14.2"`, `"6.15-rc3"`,
    /// `"6.16-rc2-next-20260420"`. Consumed by
    /// [`version_tuple`], [`patch_level`], and
    /// `cli::version_prefix`.
    pub version: String,
}

/// Is this releases.json moniker one that the version-resolution
/// pipeline should skip?
///
/// `linux-next` is a rolling integration branch whose version strings
/// carry a date suffix rather than a stable tag, so it does not fit
/// the major.minor.patch resolution model used by `latest_in_series`,
/// `fetch_version_for_prefix`, and `cli::fetch_active_prefixes`. The
/// release iteration in all three sites filters it out; this helper
/// is the single point of truth for that decision so a future moniker
/// that also warrants skipping can be added in one place.
pub(crate) fn is_skippable_release_moniker(moniker: &str) -> bool {
    moniker == "linux-next"
}

/// Find the latest version in the same major.minor series from releases.json.
///
/// Returns `Some("6.14.10")` for prefix `"6.14"` if that series exists in
/// releases.json. Returns `None` if the series is not found (EOL or invalid).
fn latest_in_series(client: &Client, version: &str) -> Option<String> {
    let prefix = {
        let parts: Vec<&str> = version.split('.').collect();
        if parts.len() >= 2 {
            format!("{}.{}", parts[0], parts[1])
        } else {
            return None;
        }
    };

    // Routes through [`RELEASES_CACHE`] for the singleton; see
    // [`cached_releases_with`] for the bypass gate.
    let releases = cached_releases_with(client).ok()?;
    let mut best: Option<(String, (u32, u32, u32))> = None;
    for r in &releases {
        if is_skippable_release_moniker(&r.moniker) {
            continue;
        }
        if !r.version.starts_with(&prefix) {
            continue;
        }
        if r.version.len() != prefix.len() && r.version.as_bytes()[prefix.len()] != b'.' {
            continue;
        }
        if let Some(tuple) = version_tuple(&r.version)
            && (best.is_none() || tuple > best.as_ref().unwrap().1)
        {
            best = Some((r.version.clone(), tuple));
        }
    }
    best.map(|(v, _)| v)
}

/// Build a user-facing error message for a version that was not found.
///
/// Suggests the latest version in the same major.minor series when
/// releases.json contains one.
fn version_not_found_msg(client: &Client, version: &str) -> String {
    let parts: Vec<&str> = version.split('.').collect();
    let prefix = if parts.len() >= 2 {
        format!("{}.{}", parts[0], parts[1])
    } else {
        version.to_string()
    };
    match latest_in_series(client, version) {
        Some(latest) if latest != version => {
            format!("version {version} not found. latest {prefix}.x: {latest}")
        }
        _ => format!("version {version} not found"),
    }
}

/// Reject responses where the server returned HTML instead of a binary
/// archive. Some CDN error pages return 200 with text/html.
fn reject_html_response(response: &reqwest::blocking::Response, url: &str) -> Result<()> {
    if let Some(ct) = response.headers().get(reqwest::header::CONTENT_TYPE)
        && let Ok(ct_str) = ct.to_str()
        && ct_str.contains("text/html")
    {
        anyhow::bail!(
            "download {url}: server returned HTML instead of tarball (URL may be invalid)"
        );
    }
    Ok(())
}

/// Print download size from Content-Length header if available.
///
/// `cli_label` prefixes the diagnostic line so the message matches the
/// binary the user invoked (`"ktstr"` vs `"cargo ktstr"`).
fn print_download_size(
    response: &reqwest::blocking::Response,
    url: &str,
    cli_label: &str,
    mp: Option<&crate::cli::FetchProgress>,
) {
    let line = if let Some(len) = response.content_length() {
        let mib = len as f64 / (1024.0 * 1024.0);
        format!("{cli_label}: downloading {url} ({mib:.1} MiB)")
    } else {
        format!("{cli_label}: downloading {url}")
    };
    // Route through the progress group so the line coordinates with
    // concurrent bars on a TTY (and still reaches piped/CI stderr when
    // the group is hidden); raw `eprintln!` when no group is present.
    match mp {
        Some(fp) => fp.println(&line),
        None => eprintln!("{line}"),
    }
}

/// Maximum tolerated stretch of "no body bytes received" before a
/// streaming download is declared stalled. Catches a TCP connection
/// that completed handshake (so connect_timeout doesn't fire) but
/// then silently stops delivering body data — a common CDN failure
/// mode where keepalive holds the socket open while the upstream
/// origin is unreachable. The 60s value is generous enough that a
/// real slow uplink delivering chunks every few seconds never
/// triggers it, but tight enough that a wedged connection surfaces
/// before the run's overall test timeout.
const DOWNLOAD_NO_PROGRESS_TIMEOUT: Duration = Duration::from_secs(60);

/// Streaming `Read` adapter for kernel tarball downloads.
///
/// Wraps the [`reqwest::blocking::Response`] body to do two things
/// the bare response cannot:
///
/// 1. **Body-progress watchdog.** Tracks `last_progress` (the
///    instant of the last successful read with `n > 0`) and errors
///    when more than [`DOWNLOAD_NO_PROGRESS_TIMEOUT`] elapses
///    between byte-producing reads. Without this, a CDN edge that
///    keepalives the socket but stops delivering body bytes would
///    only surface after reqwest's per-request read timeout
///    ([`DOWNLOAD_REQUEST_READ_TIMEOUT`], 300s), which bounds a
///    single stalled `read()`; the watchdog applies the tighter
///    60s no-progress bound across successive reads. The check fires
///    BEFORE the inner `read()` so a stalled inner reader cannot
///    out-block the watchdog.
///
/// 2. **Streaming checksums.** Updates SHA-256 and MD5 hashers with
///    every byte that flows past, so kernel.org/distro SHA-256 and
///    Google Cloud Storage MD5 metadata can both use this one
///    watchdog/progress stream without a second pass over the data.
///
/// Sits between [`reqwest::blocking::Response`] and the
/// decompression layer (`XzDecoder` / `GzDecoder`); both
/// decompressors expose `into_inner()` so the wrapper can be
/// recovered after extraction completes (see
/// [`Self::finalize`]).
struct DownloadStream<R: Read> {
    /// Underlying reqwest response body. Owned because `XzDecoder`
    /// and `GzDecoder` take ownership of their inner reader, so
    /// the wrapper must hold the response by value rather than by
    /// reference.
    inner: R,
    /// Running SHA-256 hasher updated on every byte-producing read.
    /// Consumed by [`DownloadStream::finalize`] (which takes `self`
    /// by value); the call site recovers the wrapper from inside
    /// the decoder + tar archive chain via `into_inner` before
    /// finalizing.
    sha256: Sha256,
    /// Running MD5 hasher for GCS `md5Hash` verification. GCS exposes
    /// this digest as standard base64 rather than hexadecimal.
    md5: Md5,
    /// Total body bytes read so far. Surfaced in the watchdog
    /// error message so an operator triaging "no progress" can see
    /// how many bytes did arrive before the stall — distinguishing
    /// "connection dropped after a few bytes" from "connection
    /// dropped after most of the payload".
    bytes_total: u64,
    /// `Instant` of the last successful read with `n > 0`. Set at
    /// construction (not on first read) so a connection that wins
    /// the handshake but never delivers any body bytes still
    /// trips the watchdog after [`DOWNLOAD_NO_PROGRESS_TIMEOUT`]
    /// rather than waiting for an indeterminate pre-data window.
    last_progress: Instant,
    /// Tolerated stretch of zero-progress time. Pinned at
    /// construction from [`DOWNLOAD_NO_PROGRESS_TIMEOUT`]; held in
    /// the struct rather than read from the constant on every
    /// `read()` so a future per-call override (e.g. shorter
    /// timeouts in tests) lands without touching the watchdog
    /// logic.
    no_progress_timeout: Duration,
    /// Optional indicatif download bar, advanced by `inc(n)` on
    /// every byte-producing read in lockstep with `bytes_total`.
    /// `None` is the no-bar path (non-TTY, or no progress group
    /// threaded in) and carries zero per-read overhead beyond the
    /// `Option` check. Advancing here — the single byte-accounting
    /// site — guarantees `bar.position() == finalize().1`, so the
    /// bar can never drift from the bytes the hasher and watchdog
    /// observed.
    progress: Option<indicatif::ProgressBar>,
}

impl<R: Read> DownloadStream<R> {
    /// Construct a streaming wrapper around `inner` with the production
    /// no-progress budget, optionally attaching an indicatif progress
    /// bar. `last_progress` is set to "now" so the watchdog clock starts
    /// at construction; the downstream decoder may take an indeterminate
    /// time before the first `read()`, but any actual progress resets
    /// the clock. The optional bar is advanced by `inc(n)` on every
    /// byte-producing read (see the `progress` field); `progress = None`
    /// is the non-TTY / no-group path (no bar). The bar is a pure
    /// observer — it never affects the watchdog gate or the streaming
    /// sha256, so a stalled or truncated download still surfaces its
    /// error unchanged.
    fn with_progress(inner: R, progress: Option<indicatif::ProgressBar>) -> Self {
        Self {
            inner,
            sha256: Sha256::new(),
            md5: Md5::new(),
            bytes_total: 0,
            last_progress: Instant::now(),
            no_progress_timeout: DOWNLOAD_NO_PROGRESS_TIMEOUT,
            progress,
        }
    }

    /// Consume the wrapper and return `(hex_digest, bytes_total)`.
    /// Lowercase hex matches the format kernel.org publishes in
    /// `sha256sums.asc`, so the caller can do a direct
    /// `eq_ignore_ascii_case` comparison without re-encoding.
    fn finalize(self) -> (String, u64) {
        (hex::encode(self.sha256.finalize()), self.bytes_total)
    }

    /// Consume the wrapper and return the GCS-style standard-base64
    /// MD5 digest plus the byte total.
    fn finalize_md5_base64(self) -> (String, u64) {
        (
            BASE64_STANDARD.encode(self.md5.finalize()),
            self.bytes_total,
        )
    }
}

impl<R: Read> Read for DownloadStream<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if SOURCE_OPERATION_INTERRUPTED.load(Ordering::Acquire) {
            return Err(interrupted_io_error("before reading response body"));
        }
        // Watchdog gate: trip BEFORE delegating to the inner reader
        // so a stalled inner read does not get a fresh chance to
        // run after the no-progress window has already expired. The
        // wrapper cannot interrupt a `read()` that is currently
        // blocked in a syscall — that protection comes from the
        // per-request timeout configured via
        // `RequestBuilder::timeout` — but it can refuse to issue
        // the next call once the cumulative no-progress window
        // crosses the bound.
        let elapsed = self.last_progress.elapsed();
        if elapsed > self.no_progress_timeout {
            return Err(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                format!(
                    "download stalled: no body bytes for {}s after {} bytes received",
                    elapsed.as_secs(),
                    self.bytes_total,
                ),
            ));
        }
        match self.inner.read(buf) {
            Ok(0) => {
                // EOF: do NOT update last_progress — a 0-byte read
                // is not progress, and updating here would let a
                // decoder that polls past EOF reset the watchdog
                // indefinitely.
                Ok(0)
            }
            Ok(n) => {
                // A signal can land while the inner blocking read owns this
                // thread. Reject the bytes before feeding a decoder, checksum,
                // or destination file so the caller's transaction rolls back
                // instead of publishing after cancellation.
                if SOURCE_OPERATION_INTERRUPTED.load(Ordering::Acquire) {
                    return Err(interrupted_io_error("while reading response body"));
                }
                self.sha256.update(&buf[..n]);
                self.md5.update(&buf[..n]);
                self.bytes_total += n as u64;
                self.last_progress = Instant::now();
                // Advance the bar in lockstep with `bytes_total` (same
                // `n`, same reads) so `position()` and `finalize().1`
                // never diverge. No-op when no bar is attached.
                if let Some(pb) = &self.progress {
                    pb.inc(n as u64);
                }
                Ok(n)
            }
            Err(e) => Err(e),
        }
    }
}

/// Per-request body-stream timeout passed to
/// [`reqwest::blocking::RequestBuilder::timeout`] for tarball
/// downloads. The blocking client treats this as a per-`read()`
/// deadline (reset on every successful read), so it complements the
/// [`DownloadStream`] watchdog: reqwest's deadline kills a single
/// stalled syscall, and the watchdog observes the cumulative
/// no-progress window across multiple reads. Set generously
/// (5 minutes) because a slow but progressing connection can
/// legitimately take that long for a single read on a large CDN
/// chunk; the watchdog provides the tighter 60s no-progress bound.
const DOWNLOAD_REQUEST_READ_TIMEOUT: Duration = Duration::from_secs(300);

/// Retry policy for [`get_with_transient_retry`]: attempt count and
/// the unit the exponential backoff scales from (sleep =
/// `backoff_unit << attempt`, i.e. 2s/4s/8s for a 1s unit — the same
/// timing as `build_helpers::retry_with_backoff`, which covers the
/// build-script blob downloads; this covers the runtime kernel-source
/// fetches). A struct rather than two bare params so the production
/// sites share one const and the fetch_tests seam can inject a
/// zero-backoff policy without sleeping through real backoff.
struct HttpRetry {
    attempts: u32,
    backoff_unit: Duration,
}

/// Production retry policy: 4 attempts, 2s/4s/8s between them.
/// Matches `MAX_TARBALL_ATTEMPTS`/`MAX_CLONE_ATTEMPTS` in build.rs so
/// every network fetch in the project tolerates the same outage
/// window (~14s) before giving up.
const TRANSIENT_HTTP_RETRY: HttpRetry = HttpRetry {
    attempts: 4,
    backoff_unit: Duration::from_secs(1),
};

/// Whether an HTTP status is worth retrying: 429 (rate limit) and the
/// gateway trio 502/503/504. kernel.org fronts cdn.kernel.org with
/// Varnish, which returns bursts of `503` with `retry-after: 0`
/// during cache churn — the exact failure that killed every
/// `just kernel-build` job in one CI run. 500 is deliberately NOT
/// retried: it signals a server bug rather than a transient edge
/// condition, and the existing contract (pinned by
/// `download_stable_tarball_500_is_hard_error_without_marker`) treats
/// it as an immediate hard error.
fn is_transient_http_status(status: reqwest::StatusCode) -> bool {
    matches!(status.as_u16(), 429 | 502 | 503 | 504)
}

/// Whether a retryable status identifies a failed HTTP gateway/CDN edge.
///
/// 429 remains retryable but deliberately does not select another address:
/// changing peers to evade a server's rate limit is incorrect. The gateway
/// trio instead says that a connectable peer could not serve the request, so
/// another address published for the same hostname is the useful next route.
fn is_dns_edge_failure_status(status: reqwest::StatusCode) -> bool {
    matches!(status.as_u16(), 502..=504)
}

/// Remaining DNS addresses for one origin during a single retry loop.
struct HttpRetryRoutes {
    host: String,
    port: u16,
    remaining: VecDeque<SocketAddr>,
}

impl HttpRetryRoutes {
    fn matches(&self, host: &str, port: u16) -> bool {
        self.host == host && self.port == port
    }

    fn matches_url(&self, url: &Url) -> bool {
        url.host_str()
            .zip(url.port_or_known_default())
            .is_some_and(|(host, port)| self.matches(host, port))
    }

    fn mark_failed(&mut self, addr: SocketAddr) {
        self.remaining
            .retain(|candidate| candidate.ip() != addr.ip());
    }
}

/// Resolve every address currently published for an HTTP origin.
fn resolve_http_origin(host: &str, port: u16) -> std::io::Result<Vec<SocketAddr>> {
    (host, port).to_socket_addrs().map(Iterator::collect)
}

/// Build a fresh client with an ordered set of origin-address candidates while
/// retaining the hostname in the request URL.
///
/// Passing the whole remaining tail lets reqwest retain its connector-level
/// multi-address/Happy-Eyeballs fallback within one attempt. The URL hostname
/// still supplies the Host header and TLS SNI. If a system proxy intercepts
/// the request, reqwest connects to that proxy instead; these origin
/// candidates then remain advisory and do not claim which upstream peer the
/// proxy selects.
fn build_routed_http_client(host: &str, addrs: &[SocketAddr]) -> reqwest::Result<Client> {
    default_http_client_builder()
        .resolve_to_addrs(host, addrs)
        .build()
}

/// Replace the retry-address queue with a fresh resolution of `request_url`.
///
/// Returning `false` means the URL has no routable HTTP origin (host or known
/// port), so the caller should preserve the ordinary same-client retry.
fn reset_http_retry_routes<R>(
    routes: &mut Option<HttpRetryRoutes>,
    request_url: &Url,
    resolve: &R,
) -> std::io::Result<bool>
where
    R: Fn(&str, u16) -> std::io::Result<Vec<SocketAddr>>,
{
    let Some(host) = request_url.host_str() else {
        *routes = None;
        return Ok(false);
    };
    let Some(port) = request_url.port_or_known_default() else {
        *routes = None;
        return Ok(false);
    };

    let mut seen = HashSet::new();
    let remaining: VecDeque<_> = resolve(host, port)?
        .into_iter()
        .filter(|addr| seen.insert(*addr))
        .collect();
    let has_candidates = !remaining.is_empty();
    *routes = Some(HttpRetryRoutes {
        host: host.to_owned(),
        port,
        remaining,
    });
    Ok(has_candidates)
}

/// Populate or update the alternate-address queue after a gateway response.
fn record_failed_http_edge<R>(
    routes: &mut Option<HttpRetryRoutes>,
    response_url: &Url,
    failed_remote: Option<SocketAddr>,
    resolve: &R,
) -> std::io::Result<bool>
where
    R: Fn(&str, u16) -> std::io::Result<Vec<SocketAddr>>,
{
    let Some(host) = response_url.host_str() else {
        return Ok(false);
    };
    let Some(port) = response_url.port_or_known_default() else {
        return Ok(false);
    };

    if routes
        .as_ref()
        .is_none_or(|existing| !existing.matches(host, port))
    {
        reset_http_retry_routes(routes, response_url, resolve)?;
    }
    if let (Some(existing), Some(failed)) = (routes.as_mut(), failed_remote) {
        existing.mark_failed(failed);
    }

    Ok(routes
        .as_ref()
        .is_some_and(|existing| !existing.remaining.is_empty()))
}

/// Prepare the next routed attempt after a transport error.
///
/// [`next_routed_http_client`] removes the current primary before issuing the
/// request, leaving only untried candidates in `routes.remaining`. When the
/// error URL still names that origin, preserve the advanced tail rather than
/// re-resolving and retrying the same primary again. A missing route set
/// (initial-client failure) or a different error origin (redirect failure)
/// gets a fresh resolution for the URL that actually failed.
fn prepare_routes_after_transport_error<R>(
    routes: &mut Option<HttpRetryRoutes>,
    failed_url: &Url,
    resolve: &R,
) -> std::io::Result<bool>
where
    R: Fn(&str, u16) -> std::io::Result<Vec<SocketAddr>>,
{
    if routes
        .as_ref()
        .is_some_and(|existing| existing.matches_url(failed_url))
    {
        return Ok(routes
            .as_ref()
            .is_some_and(|existing| !existing.remaining.is_empty()));
    }
    reset_http_retry_routes(routes, failed_url, resolve)
}

/// Route-relevant facts from one failed HTTP attempt.
///
/// Keeping this independent of reqwest's concrete response/error objects gives
/// the retry loop one transition point for both failure classes and lets tests
/// drive the route state deterministically without scheduling local sockets.
enum HttpRetryRouteFailure<'a> {
    Response {
        status: reqwest::StatusCode,
        response_url: &'a Url,
        remote_addr: Option<SocketAddr>,
    },
    Transport {
        failed_url: &'a Url,
    },
}

/// Update retry-route state from one failed attempt.
///
/// Non-gateway responses (notably 429) deliberately leave routing unchanged.
/// Gateway responses discard the observed peer after resolving the response
/// origin, while transport errors initialize or preserve the candidate tail
/// according to [`prepare_routes_after_transport_error`].
fn prepare_http_retry_route<R>(
    routes: &mut Option<HttpRetryRoutes>,
    failure: HttpRetryRouteFailure<'_>,
    resolve: &R,
) -> std::io::Result<bool>
where
    R: Fn(&str, u16) -> std::io::Result<Vec<SocketAddr>>,
{
    match failure {
        HttpRetryRouteFailure::Response {
            status,
            response_url,
            remote_addr,
        } if is_dns_edge_failure_status(status) => {
            record_failed_http_edge(routes, response_url, remote_addr, resolve)
        }
        HttpRetryRouteFailure::Response { .. } => Ok(false),
        HttpRetryRouteFailure::Transport { failed_url } => {
            prepare_routes_after_transport_error(routes, failed_url, resolve)
        }
    }
}

/// Build a retry client for the next untried primary address plus every
/// remaining fallback candidate.
///
/// Client-construction failure is non-fatal: skipping that primary address
/// preserves the original retry contract rather than replacing an HTTP outage
/// with a TLS/backend setup error.
fn next_routed_http_client<B>(
    routes: &mut Option<HttpRetryRoutes>,
    build_client: &B,
) -> Option<(Client, Vec<SocketAddr>)>
where
    B: Fn(&str, &[SocketAddr]) -> reqwest::Result<Client>,
{
    let routes = routes.as_mut()?;
    while let Some(primary) = routes.remaining.pop_front() {
        let candidates: Vec<_> = std::iter::once(primary)
            .chain(routes.remaining.iter().copied())
            .collect();
        match build_client(&routes.host, &candidates) {
            Ok(client) => return Some((client, candidates)),
            Err(err) => {
                tracing::warn!(
                    host = %routes.host,
                    ?candidates,
                    %err,
                    "failed to build an HTTP retry client for resolved candidates; \
                     trying a shorter candidate tail",
                );
            }
        }
    }
    None
}

/// GET `url`, retrying transient failures (transport errors and
/// [`is_transient_http_status`] statuses) per `retry`.
///
/// The failure surface is IDENTICAL to a bare
/// `client.get(url).send()`: every non-transient response — success,
/// 404, or any other status — is returned to the caller untouched, so
/// each call site keeps its own status routing (the 404 →
/// [`TarballNotFound`] git-fallback marker, the RC/codeload bespoke
/// 404 messages) and its exact error wording. A transient status on
/// the FINAL attempt is likewise returned (the caller's status gate
/// bails with the same `HTTP 503` text an unretried call would have
/// produced); a transport error on the final attempt propagates with
/// the same `"{what} {url}"` context the call sites previously
/// attached. Retries are purely additive.
///
/// `what` is the call site's error-context verb ("download" /
/// "fetch") so the transport-error context matches the wording each
/// site used before retries existed.
///
/// Only the GET is retried — response-body streaming failures are
/// out of scope (the tarball sites stream through [`DownloadStream`]
/// with its own watchdog, and a mid-stream abort after a 200 leaves
/// staging state a plain re-GET could not resume anyway).
fn get_with_transient_retry(
    client: &Client,
    url: &str,
    timeout: Option<Duration>,
    what: &str,
    retry: &HttpRetry,
) -> Result<reqwest::blocking::Response> {
    get_with_transient_retry_and_headers(
        client,
        url,
        timeout,
        what,
        retry,
        &reqwest::header::HeaderMap::new(),
    )
}

/// [`get_with_transient_retry`] with headers reapplied to every attempt.
///
/// This is used by the ranged URL-existence probe. Keeping request
/// construction inside the retry loop is important: a retry must carry the
/// same `Range` contract as the first attempt instead of silently becoming a
/// full-body GET.
fn get_with_transient_retry_and_headers(
    client: &Client,
    url: &str,
    timeout: Option<Duration>,
    what: &str,
    retry: &HttpRetry,
    headers: &reqwest::header::HeaderMap,
) -> Result<reqwest::blocking::Response> {
    get_with_transient_retry_and_routes(
        client,
        url,
        timeout,
        what,
        retry,
        headers,
        HttpRouteFailover {
            enabled: is_shared_client(client),
            resolve: resolve_http_origin,
            build_client: build_routed_http_client,
        },
    )
}

struct HttpRouteFailover<R, B> {
    enabled: bool,
    resolve: R,
    build_client: B,
}

/// Implementation seam for [`get_with_transient_retry`].
///
/// Production enables address failover only for [`shared_client`], whose
/// builder configuration is known and can be reproduced safely. A
/// caller-provided client may carry custom roots, proxies, or redirect policy,
/// so its retries remain on that exact client. Tests inject deterministic
/// address resolution and a no-proxy routed-client builder here.
fn get_with_transient_retry_and_routes<R, B>(
    client: &Client,
    url: &str,
    timeout: Option<Duration>,
    what: &str,
    retry: &HttpRetry,
    headers: &reqwest::header::HeaderMap,
    route_failover: HttpRouteFailover<R, B>,
) -> Result<reqwest::blocking::Response>
where
    R: Fn(&str, u16) -> std::io::Result<Vec<SocketAddr>>,
    B: Fn(&str, &[SocketAddr]) -> reqwest::Result<Client>,
{
    assert!(retry.attempts > 0, "HttpRetry.attempts must be >= 1");
    let mut routes = None;
    let mut routed_client: Option<(Client, Vec<SocketAddr>)> = None;

    for attempt in 1..=retry.attempts {
        ensure_source_operation_not_interrupted("before HTTP request")?;
        let request_client = routed_client
            .as_ref()
            .map(|(client, _)| client)
            .unwrap_or(client);
        let mut req = request_client.get(url).headers(headers.clone());
        if let Some(t) = timeout {
            req = req.timeout(t);
        }
        let mut advance_route = false;
        match req.send() {
            Ok(response) => {
                ensure_source_operation_not_interrupted("after HTTP response headers")?;
                let status = response.status();
                if !is_transient_http_status(status) || attempt == retry.attempts {
                    return Ok(response);
                }
                let remote_addr = response.remote_addr();
                if route_failover.enabled {
                    match prepare_http_retry_route(
                        &mut routes,
                        HttpRetryRouteFailure::Response {
                            status,
                            response_url: response.url(),
                            remote_addr,
                        },
                        &route_failover.resolve,
                    ) {
                        Ok(has_routes) => advance_route = has_routes,
                        Err(err) => tracing::warn!(
                            url = %response.url(),
                            %err,
                            "failed to resolve alternate HTTP addresses; retrying normally",
                        ),
                    }
                }
                tracing::warn!(
                    %url, %status, ?remote_addr, attempt, max_attempts = retry.attempts,
                    "{what} hit a transient HTTP status; retrying",
                );
            }
            Err(e) => {
                if attempt == retry.attempts {
                    return Err(anyhow::Error::new(e).context(format!("{what} {url}")));
                }
                if route_failover.enabled {
                    let failed_url = e.url().cloned().or_else(|| Url::parse(url).ok());
                    if let Some(failed_url) = failed_url {
                        match prepare_http_retry_route(
                            &mut routes,
                            HttpRetryRouteFailure::Transport {
                                failed_url: &failed_url,
                            },
                            &route_failover.resolve,
                        ) {
                            Ok(has_routes) => advance_route = has_routes,
                            Err(resolve_err) => tracing::warn!(
                                url = %failed_url,
                                err = %resolve_err,
                                "failed to refresh HTTP candidates after a transport error; \
                                 retrying normally",
                            ),
                        }
                    }
                }
                tracing::warn!(
                    %url, error_url = ?e.url(), err = %e,
                    attempt, max_attempts = retry.attempts,
                    "{what} failed in transport; retrying",
                );
            }
        }
        // Reached only on a transient failure with attempts left:
        // both match arms return on the final attempt.
        source_operation_backoff(retry.backoff_unit * (1u32 << attempt))?;
        if advance_route {
            routed_client = next_routed_http_client(&mut routes, &route_failover.build_client);
            if let Some((_, candidates)) = routed_client.as_ref() {
                tracing::warn!(
                    %url,
                    ?candidates,
                    next_attempt = attempt + 1,
                    "retrying with alternate resolved origin candidates; \
                     an active proxy may select a different upstream route",
                );
            }
        }
    }
    unreachable!("the attempt == retry.attempts arms above return on the final iteration")
}

/// Sleep between HTTP attempts without making cancellation wait for the whole
/// exponential-backoff interval. The 100 ms quantum matches the flock wait
/// poller and is far below the shortest production backoff (2 s).
fn source_operation_backoff(duration: Duration) -> Result<()> {
    const QUANTUM: Duration = Duration::from_millis(100);
    let deadline = Instant::now() + duration;
    loop {
        ensure_source_operation_not_interrupted("during HTTP retry backoff")?;
        let now = Instant::now();
        if now >= deadline {
            return Ok(());
        }
        std::thread::sleep((deadline - now).min(QUANTUM));
    }
}

/// GET `url` through the process-wide [`shared_client`] with the
/// standard [`TRANSIENT_HTTP_RETRY`] policy and return the full
/// response body as bytes.
///
/// The distro repo-metadata resolver ([`crate::distro::repo`]) fetches
/// repomd.xml, primary.xml.{gz,zst,xz}, `Packages.gz`, and `mirror.list`
/// through this so every metadata fetch rides the same retry/backoff
/// seam as the kernel-source downloads. `what` is the error-context
/// verb ("fetch"). A non-success status is a hard error carrying the
/// status; the caller adds its own context. The explicit
/// [`SMALL_RESPONSE_REQUEST_TIMEOUT`] bounds both the wait for response
/// headers and every blocking body read, including the headers-then-stall
/// failure mode.
pub(crate) fn fetch_metadata_bytes(url: &str, what: &str) -> Result<Vec<u8>> {
    fetch_metadata_bytes_with(
        shared_client(),
        url,
        what,
        &TRANSIENT_HTTP_RETRY,
        SMALL_RESPONSE_REQUEST_TIMEOUT,
    )
}

/// Injectable core of [`fetch_metadata_bytes`] for timeout/retry tests.
fn fetch_metadata_bytes_with(
    client: &Client,
    url: &str,
    what: &str,
    retry: &HttpRetry,
    timeout: Duration,
) -> Result<Vec<u8>> {
    let response = get_with_transient_retry(client, url, Some(timeout), what, retry)?;
    if !response.status().is_success() {
        anyhow::bail!("{what} {url}: HTTP {}", response.status());
    }
    read_response_bytes(response, url)
}

/// GET `url` like [`fetch_metadata_bytes`] but decode the body as
/// UTF-8 text — for the plain-text metadata endpoints (`mirror.list`,
/// `meta-release-lts`).
pub(crate) fn fetch_metadata_text(url: &str, what: &str) -> Result<String> {
    let bytes = fetch_metadata_bytes(url, what)?;
    String::from_utf8(bytes).with_context(|| format!("decode body of {url} as UTF-8"))
}

/// Ranged GET used by live distro-resolution tests to prove a resolved
/// package URL still exists upstream without downloading the package.
///
/// The range header is rebuilt on every transient retry, and the small-response
/// timeout bounds both header and body waits. Returning the final status keeps
/// 404 and other permanent failures visible to the caller's assertion.
#[cfg(test)]
pub(crate) fn probe_url_status(url: &str) -> Result<reqwest::StatusCode> {
    probe_url_status_with(
        shared_client(),
        url,
        &TRANSIENT_HTTP_RETRY,
        SMALL_RESPONSE_REQUEST_TIMEOUT,
    )
}

#[cfg(test)]
fn probe_url_status_with(
    client: &Client,
    url: &str,
    retry: &HttpRetry,
    timeout: Duration,
) -> Result<reqwest::StatusCode> {
    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert(
        reqwest::header::RANGE,
        reqwest::header::HeaderValue::from_static("bytes=0-0"),
    );
    let response = get_with_transient_retry_and_headers(
        client,
        url,
        Some(timeout),
        "probe",
        retry,
        &headers,
    )?;
    Ok(response.status())
}

/// Construct the cdn.kernel.org `sha256sums.asc` URL for a stable
/// major series:
/// `https://cdn.kernel.org/pub/linux/kernel/v{major}.x/sha256sums.asc`.
/// Single source of truth for the manifest URL shape, used by
/// [`resolve_expected_sha256`] (production) and shared with the
/// URL-injection test seam so the two never drift.
fn sha256sums_url(major: u32) -> String {
    format!("https://cdn.kernel.org/pub/linux/kernel/v{major}.x/sha256sums.asc")
}

/// GET the cleartext SHA-256 manifest at `url` and return its body.
///
/// Returns the file body as a `String` on success. Any error
/// (transport failure, non-2xx status, non-UTF-8 body) is
/// propagated; the caller treats failure as "no expected hash
/// available" and downgrades verification to a warning.
///
/// Takes the full `url` rather than a `major` so the GET-and-status
/// mechanics are reachable with an injected URL (a localhost mock)
/// without a real cdn.kernel.org round-trip — mirrors the
/// [`fetch_releases`] / [`cached_releases_with_url`] seam. Production
/// reaches this only via [`resolve_expected_sha256_from_url`], whose
/// URL is pinned by [`sha256sums_url`].
fn fetch_sha256sums_from_url(client: &Client, url: &str) -> Result<String> {
    tracing::info!(%url, "fetching kernel tarball sha256sums (requires network)");
    let response = get_with_transient_retry(
        client,
        url,
        Some(SMALL_RESPONSE_REQUEST_TIMEOUT),
        "fetch",
        &TRANSIENT_HTTP_RETRY,
    )?;
    if !response.status().is_success() {
        anyhow::bail!("fetch {url}: HTTP {}", response.status());
    }
    let bytes = read_response_bytes(response, url)?;
    String::from_utf8(bytes).with_context(|| format!("decode body of {url} as UTF-8"))
}

/// Read a short HTTP response through the same cancellation-aware
/// streaming adapter as large archives. Metadata endpoints do not need
/// a visible byte bar, but they must observe the shared signal epoch
/// before and after every blocking body read.
fn read_response_bytes(response: reqwest::blocking::Response, url: &str) -> Result<Vec<u8>> {
    let mut stream = DownloadStream::with_progress(response, None);
    let mut bytes = Vec::new();
    stream
        .read_to_end(&mut bytes)
        .with_context(|| format!("read body of {url}"))?;
    ensure_source_operation_not_interrupted("after reading HTTP response body")?;
    Ok(bytes)
}

/// Extract the SHA-256 hex digest for `target_filename` from the
/// cleartext-signed `sha256sums.asc` body.
///
/// kernel.org publishes `sha256sums.asc` as a PGP-cleartext-signed
/// document: a `-----BEGIN PGP SIGNED MESSAGE-----` header, an
/// optional `Hash:` line, a blank line, the cleartext body
/// (`<64-hex-chars>  <filename>` per line), then a
/// `-----BEGIN PGP SIGNATURE-----` block. We only need the
/// cleartext body — signature verification is a separate concern
/// (the user-facing instruction is "If no expected hash available,
/// log warning", not "require signature").
///
/// Returns `Some(lowercase_hex)` on first match. Returns `None` if
/// the target filename does not appear in the manifest (e.g. the
/// upstream rotated or removed the entry).
fn parse_sha256_for_file(manifest: &str, target_filename: &str) -> Option<String> {
    // Strip the PGP signature trailer if present. Everything after
    // the signature marker is binary noise that never contains
    // checksum lines.
    let body = manifest
        .split_once("-----BEGIN PGP SIGNATURE-----")
        .map(|(before, _)| before)
        .unwrap_or(manifest);
    for line in body.lines() {
        let line = line.trim();
        // sha256sum format: `<64-hex-chars><whitespace><filename>`.
        // Split on whitespace; require exactly two tokens and a
        // 64-char hex first token.
        let mut parts = line.split_whitespace();
        let Some(hash) = parts.next() else { continue };
        let Some(name) = parts.next() else { continue };
        if name != target_filename {
            continue;
        }
        if hash.len() != 64 || !hash.chars().all(|c| c.is_ascii_hexdigit()) {
            continue;
        }
        return Some(hash.to_ascii_lowercase());
    }
    None
}

/// Verify `actual_hex` against `expected_hex` (case-insensitive).
/// Returns `Ok(())` on match, `Err` with a diagnostic message on
/// mismatch. Pulled out of the call site so the comparison logic
/// has one home and the diagnostic carries both digests in lowercase
/// hex for direct copy-paste reuse.
fn verify_sha256(actual_hex: &str, expected_hex: &str, url: &str) -> Result<()> {
    if actual_hex.eq_ignore_ascii_case(expected_hex) {
        Ok(())
    } else {
        anyhow::bail!(
            "sha256 mismatch for {url}: expected {}, got {}. \
             If cdn.kernel.org updated this tarball in-place, \
             retry with --skip-sha256 to bypass verification.",
            expected_hex.to_ascii_lowercase(),
            actual_hex.to_ascii_lowercase(),
        );
    }
}

/// Resolve the expected SHA-256 digest for a stable tarball from
/// cdn.kernel.org's `sha256sums.asc` manifest.
///
/// Three outcomes:
/// - `Some(hex)` — manifest fetched and the entry for `tarball_name`
///   was parsed cleanly.
/// - `None` with no warning (only when `skip_sha256 = true`) —
///   operator explicitly opted out of verification; emits a single
///   security-sensitive bypass warning instead.
/// - `None` with a per-cause warning (manifest fetch failed, or
///   manifest fetched but entry missing) — best-effort fallback so
///   a transient cdn.kernel.org outage / schema drift does not
///   gate the whole download.
///
/// The fallback path is deliberately permissive: we trade strict
/// authentication for build availability. A network-path attacker
/// who can deny `sha256sums.asc` while serving a poisoned
/// `linux-{version}.tar.xz` could exploit this; operators who
/// require strict verification should pin the source via a
/// `--kernel <path>` or `--kernel git+…` source rather than the
/// download path. The bypass warnings
/// surface on the operator's diagnostic stream so the lost
/// guarantee is visible to ops triage.
///
/// Extracted from [`download_stable_tarball`] so the gate is
/// directly unit-testable without mocking network calls — the
/// caller-supplied `client` reaches a `Client::get` only when
/// `skip_sha256 == false`, so a `skip_sha256 = true` test does not
/// need a configured `Client`.
fn resolve_expected_sha256(
    client: &Client,
    major: u32,
    tarball_name: &str,
    skip_sha256: bool,
) -> Option<String> {
    resolve_expected_sha256_from_url(client, &sha256sums_url(major), tarball_name, skip_sha256)
}

/// URL-injectable core of [`resolve_expected_sha256`]: the skip-gate,
/// fetch-then-parse, and per-cause warn-and-downgrade logic, against
/// an arbitrary `sha256sums_url`. Production reaches this only via
/// [`resolve_expected_sha256`], which pins the URL to
/// [`sha256sums_url`]; the seam exists so the no-skip arm's
/// fetch-and-parse path is testable against a localhost mock without a
/// real cdn.kernel.org round-trip — mirrors [`cached_releases_with_url`].
fn resolve_expected_sha256_from_url(
    client: &Client,
    sha256sums_url: &str,
    tarball_name: &str,
    skip_sha256: bool,
) -> Option<String> {
    if skip_sha256 {
        tracing::warn!(
            tarball = %tarball_name,
            "--skip-sha256: bypassing checksum verification — the \
             downloaded tarball will not be authenticated against \
             cdn.kernel.org's sha256sums.asc manifest. Use only when \
             upstream has updated a tarball in-place and the manifest \
             is mismatched.",
        );
        return None;
    }
    // Best-effort expected-hash lookup: any failure (network,
    // status, parse, missing entry) downgrades to a warning so the
    // download still proceeds. The warning surfaces the cause so an
    // operator triaging "kernel build went weird" can spot that
    // verification was skipped.
    match fetch_sha256sums_from_url(client, sha256sums_url) {
        Ok(manifest) => match parse_sha256_for_file(&manifest, tarball_name) {
            Some(hex) => Some(hex),
            None => {
                tracing::warn!(
                    tarball = %tarball_name,
                    "sha256sums.asc fetched but no entry for {tarball_name}; \
                     download will proceed without checksum verification. \
                     Pass --skip-sha256 to bypass the manifest fetch when \
                     the entry is known to be absent.",
                );
                None
            }
        },
        Err(err) => {
            tracing::warn!(
                error = %format!("{err:#}"),
                "failed to fetch sha256sums.asc; download will proceed \
                 without checksum verification. Pass --skip-sha256 to \
                 bypass the manifest fetch when the manifest is known \
                 to be unavailable.",
            );
            None
        }
    }
}

/// GitHub mirror of the linux-stable tree — comprehensive (stable +
/// base-release `vX.Y.Z` tags back to v2.6) and the authoritative
/// source for tags whose `.tar.xz` is no longer on cdn.kernel.org.
///
/// cdn.kernel.org keeps only the LATEST tarball of each series
/// currently in `releases.json`; every superseded point release AND
/// every tag of an EOL series is pruned (a GET for the tarball 404s,
/// verified empirically — and HEAD is not a dependable existence probe
/// on the CDN). The gregkh mirror still carries every `vX.Y.Z` tag, and
/// codeload serves each tag's snapshot as a `tar.gz`, so a codeload
/// download recovers the source a pruned tarball would have provided —
/// no clone. Its `ls-refs` advertises every release tag, which
/// `--include-eol` enumerates to surface EOL series absent from
/// `releases.json` (see [`cached_stable_tags`]) and which
/// [`fetch_version_for_prefix`] resolves for an EOL/unreleased series.
/// github.com advertises allow-sha + a ref-prefix filter and a codeload
/// CDN; git.kernel.org offers neither. Used by [`download_tarball`]'s
/// [`TarballNotFound`] fallback and the prefix resolver.
const STABLE_MIRROR_URL: &str = "https://github.com/gregkh/linux";

/// Marker error attached to a stable-tarball download failure when
/// cdn.kernel.org returns HTTP 404.
///
/// A 404 means the tarball is pruned — an EOL series (absent from
/// `releases.json`) or a superseded point release (the CDN retains
/// only each maintained series' latest). [`download_tarball`] detects
/// this via `downcast_ref` (the context-aware anyhow accessor — a
/// `chain().any(..is::<T>())` walk would MISS a context-wrapped
/// marker) and falls back to a codeload snapshot of the tag from the
/// gregkh mirror ([`STABLE_MIRROR_URL`]). Any other HTTP status is a
/// hard error with no fallback.
#[derive(Debug)]
struct TarballNotFound;

impl std::fmt::Display for TarballNotFound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("stable tarball pruned from cdn.kernel.org (EOL or superseded point release)")
    }
}

impl std::error::Error for TarballNotFound {}

/// Download a stable kernel tarball (.tar.xz) from cdn.kernel.org.
///
/// Returns a [`TarballNotFound`] error (downcast-detectable) when the
/// CDN 404s the tarball — see that type for the pruning semantics and
/// [`download_tarball`] for the git-tag fallback it triggers.
///
/// Streams the body through a [`DownloadStream`] watchdog so a
/// stalled connection (no body bytes for
/// [`DOWNLOAD_NO_PROGRESS_TIMEOUT`]) surfaces as an error rather
/// than blocking indefinitely. Computes SHA-256 over the streamed
/// bytes and verifies against the digest in
/// `sha256sums.asc` for the matching `linux-{version}.tar.xz`
/// entry; if the manifest fetch / parse fails (transient outage,
/// schema drift, missing entry), logs a warning and continues
/// without verification rather than failing the whole download.
///
/// `skip_sha256 = true` bypasses the manifest fetch entirely and
/// emits a single bypass warning. Intended for the case where
/// cdn.kernel.org has updated a tarball in-place (a new point
/// release reusing the same URL) and the manifest is stale or
/// mismatched. Unverified downloads are a security-sensitive
/// fallback — the bypass warning surfaces the lost guarantee on
/// the operator's diagnostic stream.
fn download_stable_tarball(
    client: &Client,
    version: &str,
    dest_dir: &Path,
    cli_label: &str,
    skip_sha256: bool,
    mp: Option<&crate::cli::FetchProgress>,
) -> Result<PathBuf> {
    let major = major_version(version)?;
    let url = format!("https://cdn.kernel.org/pub/linux/kernel/v{major}.x/linux-{version}.tar.xz");
    download_stable_tarball_from_url(client, &url, version, dest_dir, cli_label, skip_sha256, mp)
}

/// URL-injectable core of [`download_stable_tarball`]: the GET, the
/// 404→[`TarballNotFound`] / other-status→hard-error status gate, and
/// the stream→verify→extract pipeline, against an arbitrary tarball
/// `url`. Production reaches this only via [`download_stable_tarball`],
/// which pins the cdn.kernel.org URL; the seam exists so the status
/// routing (404 marker vs hard error) is unit-testable against a
/// localhost mock without a real cdn round-trip — mirrors
/// [`resolve_expected_sha256_from_url`] / [`fetch_releases`].
fn download_stable_tarball_from_url(
    client: &Client,
    url: &str,
    version: &str,
    dest_dir: &Path,
    cli_label: &str,
    skip_sha256: bool,
    mp: Option<&crate::cli::FetchProgress>,
) -> Result<PathBuf> {
    let major = major_version(version)?;
    let tarball_name = format!("linux-{version}.tar.xz");

    let expected_sha256 = resolve_expected_sha256(client, major, &tarball_name, skip_sha256);

    tracing::info!(%url, "downloading stable kernel tarball (requires network)");
    let response = get_with_transient_retry(
        client,
        url,
        Some(DOWNLOAD_REQUEST_READ_TIMEOUT),
        "download",
        &TRANSIENT_HTTP_RETRY,
    )?;
    if !response.status().is_success() {
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            // Pruned tarball (EOL series or superseded point release).
            // Return the downcast-detectable marker so `download_tarball`
            // falls back to a codeload snapshot of the tag from the
            // gregkh mirror (`STABLE_MIRROR_URL`) rather than failing
            // outright.
            return Err(anyhow::Error::new(TarballNotFound));
        }
        anyhow::bail!("download {url}: HTTP {}", response.status());
    }
    reject_html_response(&response, url)?;
    print_download_size(&response, url, cli_label, mp);
    // Capture the total before `response` is moved into the stream so a
    // determinate (percent + ETA) bar can be built; `None` when the
    // server sent no Content-Length, in which case the bar degrades to
    // a live byte counter.
    let total = response.content_length();

    // Route status lines through the progress group (see
    // `print_download_size`); `eprintln!` when no group is threaded in.
    let status = |line: &str| match mp {
        Some(fp) => fp.println(line),
        None => eprintln!("{line}"),
    };
    status(&format!("{cli_label}: extracting tarball (xz)"));
    // Stage extraction inside `dest_dir` (same filesystem) so the
    // final `RENAME_NOREPLACE` into place is atomic and a verification
    // failure leaves `dest_dir` untouched. A bad mirror that serves
    // a wrong-version archive — or sneaks stray top-level entries
    // alongside `linux-{version}/` — gets caught after extraction
    // but before anything lands in `dest_dir`. The TempDir's Drop
    // sweeps every entry the malicious archive deposited.
    let staging =
        tempfile::TempDir::new_in(dest_dir).with_context(|| "create extraction staging dir")?;
    let download_bar = mp.map(|fp| fp.download_bar(version, total));
    let stream = DownloadStream::with_progress(response, download_bar.as_ref().map(|b| b.bar()));
    let decoder = xz2::read::XzDecoder::new(stream);
    let mut archive = tar::Archive::new(decoder);
    ensure_source_operation_not_interrupted("before extracting stable kernel tarball")?;
    archive
        .unpack(staging.path())
        .with_context(|| "extract tarball")?;
    ensure_source_operation_not_interrupted("after extracting stable kernel tarball")?;

    // Recover the watchdog wrapper from inside the decoder/archive
    // chain to read the streaming digest. `into_inner` on tar +
    // xz2 each peel one layer of the chain. Done after a successful
    // unpack so we don't compute over a partial stream.
    let stream = archive.into_inner().into_inner();
    let (actual_hex, bytes_total) = stream.finalize();
    // Download is complete (every byte streamed) — clear the bar
    // before emitting the verification status so the two don't overlap.
    if let Some(bar) = &download_bar {
        bar.finish();
    }
    if let Some(expected) = expected_sha256.as_deref() {
        verify_sha256(&actual_hex, expected, url)?;
        status(&format!(
            "{cli_label}: sha256 verified ({bytes_total} bytes, hash {actual_hex})"
        ));
    } else if !skip_sha256 {
        // Skip path already emitted its bespoke bypass warning
        // before the download; firing again here under "no
        // expected sha256 available" would mislead — that wording
        // implies a fallback, not an explicit operator opt-out.
        tracing::warn!(
            url = %url,
            bytes = bytes_total,
            sha256 = %actual_hex,
            "no expected sha256 available for {url}; computed digest \
             {actual_hex} over {bytes_total} bytes is unverified",
        );
    }

    ensure_source_operation_not_interrupted("before promoting stable kernel source")?;
    let promoted = promote_staged_kernel_tree_transaction(&staging, dest_dir, version)?;
    if let Err(err) =
        ensure_source_operation_not_interrupted("after promoting stable kernel source")
    {
        return Err(promoted.rollback(err));
    }
    Ok(promoted.commit())
}

/// Verify a kernel tarball's staged extraction contains exactly one
/// top-level entry named `linux-{version}/` and atomically rename it
/// into `dest_dir/linux-{version}`. Bails — leaving `dest_dir`
/// untouched — when the staging dir holds a stray entry, when the
/// expected inner directory is missing, or when the rename fails.
/// The caller's `TempDir` outlives this helper, so its Drop sweeps
/// any residual staging contents whether this returns Ok or Err.
#[cfg(test)]
fn promote_staged_kernel_tree(
    staging: &tempfile::TempDir,
    dest_dir: &Path,
    version: &str,
) -> Result<PathBuf> {
    Ok(promote_staged_kernel_tree_transaction(staging, dest_dir, version)?.commit())
}

fn promote_staged_kernel_tree_transaction(
    staging: &tempfile::TempDir,
    dest_dir: &Path,
    version: &str,
) -> Result<PromotedPath> {
    let expected_name = format!("linux-{version}");
    let mut found_inner = false;
    for entry in std::fs::read_dir(staging.path()).with_context(|| "read staging dir entries")? {
        let entry = entry.with_context(|| "iterate staging dir entry")?;
        let name = entry.file_name();
        if name == std::ffi::OsStr::new(&expected_name) {
            found_inner = true;
        } else {
            anyhow::bail!(
                "tarball contains unexpected top-level entry {name:?}; \
                 expected only {expected_name}/"
            );
        }
    }
    if !found_inner {
        anyhow::bail!("expected directory {expected_name} after extraction");
    }
    let inner = staging.path().join(&expected_name);
    let source_dir = dest_dir.join(&expected_name);
    promote_path_noreplace(&inner, &source_dir, PublishedPathKind::Directory)
}

/// Promote the single top-level directory a codeload archive extracts
/// out of `staging` into `dest_dir/{canonical}`, so it survives
/// `staging`'s `Drop`.
///
/// Unlike [`promote_staged_kernel_tree`], the top-dir name is not
/// `linux-{version}` — GitHub derives it from the ref (`linux-6.11.11`
/// for a tag, `linux-{sha}` for a commit, `linux-{branch}` for a
/// branch), so this promotes the SOLE entry by structure rather than by
/// a fixed name, renaming it to a caller-supplied `canonical` name that
/// keys off the resolved commit (collision-free across refs). A hostile
/// or malformed snapshot that deposits zero or several top-level
/// entries — or a top-level entry that is not a plain directory (a
/// regular file, or a symlink, which the directory-entry file-type
/// check rejects rather than following) — is rejected before anything
/// lands in `dest_dir`; the `TempDir`'s `Drop` sweeps every entry the
/// archive left.
#[cfg(test)]
fn promote_single_kernel_tree(
    staging: &tempfile::TempDir,
    dest_dir: &Path,
    canonical: &str,
) -> Result<PathBuf> {
    Ok(promote_single_kernel_tree_transaction(staging, dest_dir, canonical)?.commit())
}

fn promote_single_kernel_tree_transaction(
    staging: &tempfile::TempDir,
    dest_dir: &Path,
    canonical: &str,
) -> Result<PromotedPath> {
    let mut entries = Vec::new();
    for entry in std::fs::read_dir(staging.path()).with_context(|| "read staging dir entries")? {
        entries.push(entry.with_context(|| "iterate staging dir entry")?);
    }
    if entries.len() != 1 {
        anyhow::bail!(
            "codeload archive must contain exactly one top-level entry; found {}",
            entries.len()
        );
    }
    let inner = entries[0].path();
    // Use the DIRECTORY-ENTRY file type (does NOT follow symlinks) so a
    // top-level symlink-to-directory is rejected rather than promoted:
    // `Path::is_dir()` would follow the link and accept an
    // attacker-chosen target, and renameat2 moves the symlink itself
    // (it never dereferences), leaving the build reading through it.
    let entry_type = entries[0]
        .file_type()
        .with_context(|| "stat codeload top-level entry")?;
    if !entry_type.is_dir() {
        anyhow::bail!(
            "codeload archive top-level entry is not a plain directory: {}",
            inner.display()
        );
    }
    let source_dir = dest_dir.join(canonical);
    promote_path_noreplace(&inner, &source_dir, PublishedPathKind::Directory)
}

#[derive(Clone, Copy)]
enum PublishedPathKind {
    File,
    Directory,
}

struct PromotedPath {
    path: PathBuf,
    kind: PublishedPathKind,
    committed: bool,
}

impl PromotedPath {
    fn commit(mut self) -> PathBuf {
        self.committed = true;
        self.path.clone()
    }

    fn rollback(mut self, primary: anyhow::Error) -> anyhow::Error {
        match self.remove() {
            Ok(()) => primary,
            Err(rollback_err) => primary.context(format!(
                "also failed to roll back transaction-owned publication {}: {rollback_err}",
                self.path.display()
            )),
        }
    }

    fn remove(&mut self) -> std::io::Result<()> {
        let result = match self.kind {
            PublishedPathKind::File => std::fs::remove_file(&self.path),
            PublishedPathKind::Directory => std::fs::remove_dir_all(&self.path),
        };
        if result.is_ok()
            || result
                .as_ref()
                .is_err_and(|err| err.kind() == std::io::ErrorKind::NotFound)
        {
            self.committed = true;
            Ok(())
        } else {
            result
        }
    }
}

impl Drop for PromotedPath {
    fn drop(&mut self) {
        if !self.committed {
            let _ = self.remove();
        }
    }
}

/// Atomically publish a transaction-owned path without replacing an
/// existing destination. A signal landing across the rename removes
/// only the path this successful `RENAME_NOREPLACE` created; failure to
/// roll it back is attached to the cancellation error instead of being
/// silently discarded.
fn promote_path_noreplace(
    staged: &Path,
    destination: &Path,
    kind: PublishedPathKind,
) -> Result<PromotedPath> {
    ensure_source_operation_not_interrupted("before publishing acquired source")?;
    rustix::fs::renameat_with(
        rustix::fs::CWD,
        staged,
        rustix::fs::CWD,
        destination,
        rustix::fs::RenameFlags::NOREPLACE,
    )
    .map_err(|err| {
        anyhow!(
            "renameat2(RENAME_NOREPLACE) {} -> {}: {err}",
            staged.display(),
            destination.display()
        )
    })?;

    if let Err(interrupted) =
        ensure_source_operation_not_interrupted("after publishing acquired source")
    {
        let promoted = PromotedPath {
            path: destination.to_path_buf(),
            kind,
            committed: false,
        };
        return Err(promoted.rollback(interrupted));
    }
    Ok(PromotedPath {
        path: destination.to_path_buf(),
        kind,
        committed: false,
    })
}

/// Download an RC kernel tarball (.tar.gz) from git.kernel.org.
///
/// Streams the body through a [`DownloadStream`] watchdog so a
/// stalled connection surfaces as an error rather than blocking
/// indefinitely. RC tarballs are dynamically generated by gitweb
/// at request time and have no published `sha256sums` manifest, so
/// this path always logs a warning that the digest is unverified —
/// it is computed and surfaced for diagnostic value (operators can
/// pin it manually) but never compared to an authoritative source.
fn download_rc_tarball(
    client: &Client,
    version: &str,
    dest_dir: &Path,
    cli_label: &str,
    mp: Option<&crate::cli::FetchProgress>,
) -> Result<PathBuf> {
    let url = format!("https://git.kernel.org/torvalds/t/linux-{version}.tar.gz");
    tracing::info!(%url, "downloading RC kernel tarball (requires network)");

    let response = get_with_transient_retry(
        client,
        &url,
        Some(DOWNLOAD_REQUEST_READ_TIMEOUT),
        "download",
        &TRANSIENT_HTTP_RETRY,
    )?;
    if response.status() == reqwest::StatusCode::NOT_FOUND {
        anyhow::bail!(
            "RC tarball not found: {url}\n  \
             RC releases are removed from git.kernel.org after the stable version ships."
        );
    }
    if !response.status().is_success() {
        anyhow::bail!("download {url}: HTTP {}", response.status());
    }
    reject_html_response(&response, &url)?;
    print_download_size(&response, &url, cli_label, mp);
    // RC tarballs are gitweb-generated and often arrive without a
    // Content-Length, so `total` is frequently `None` and the bar
    // degrades to a live byte counter (rate, no ETA).
    let total = response.content_length();

    let status = |line: &str| match mp {
        Some(fp) => fp.println(line),
        None => eprintln!("{line}"),
    };
    status(&format!("{cli_label}: extracting tarball (gzip)"));
    // Stage extraction inside `dest_dir` (same filesystem) so the
    // final atomic rename keeps `dest_dir` clean when a bad mirror
    // serves a wrong-version archive or sneaks stray top-level
    // entries past the archive boundary. RC tarballs have no
    // upstream sha256 manifest, so structural verification is the
    // only defence against a hostile gitweb response.
    let staging =
        tempfile::TempDir::new_in(dest_dir).with_context(|| "create extraction staging dir")?;
    let download_bar = mp.map(|fp| fp.download_bar(version, total));
    let stream = DownloadStream::with_progress(response, download_bar.as_ref().map(|b| b.bar()));
    let decoder = flate2::read::GzDecoder::new(stream);
    let mut archive = tar::Archive::new(decoder);
    ensure_source_operation_not_interrupted("before extracting RC kernel tarball")?;
    archive
        .unpack(staging.path())
        .with_context(|| "extract tarball")?;
    ensure_source_operation_not_interrupted("after extracting RC kernel tarball")?;

    // Surface the streamed digest as a warning. RC tarballs have
    // no upstream manifest, so verification is impossible — but
    // emitting the hash gives an operator a value they can
    // capture for offline pinning if they want to detect drift on
    // re-fetch.
    let stream = archive.into_inner().into_inner();
    let (actual_hex, bytes_total) = stream.finalize();
    if let Some(bar) = &download_bar {
        bar.finish();
    }
    tracing::warn!(
        url = %url,
        bytes = bytes_total,
        sha256 = %actual_hex,
        "no expected sha256 available for {url} (RC tarballs are \
         dynamically generated by git.kernel.org and have no \
         published manifest); computed digest {actual_hex} over \
         {bytes_total} bytes is unverified",
    );

    ensure_source_operation_not_interrupted("before promoting RC kernel source")?;
    let promoted = promote_staged_kernel_tree_transaction(&staging, dest_dir, version)?;
    if let Err(err) = ensure_source_operation_not_interrupted("after promoting RC kernel source") {
        return Err(promoted.rollback(err));
    }
    Ok(promoted.commit())
}

/// Download a GitHub source snapshot for `git_ref` as a codeload
/// `tar.gz` and extract it, returning an [`AcquiredSource`] keyed as an
/// explicit immutable-SHA snapshot ([`git_cache_key`] over the resolved
/// `commit_hash`). Archive SHA, branch, and tag identities remain
/// distinct even when they currently point at the same commit.
///
/// GitHub serves a gzip snapshot for any tag/branch/commit via
/// codeload; the caller supplies the `archive_url`
/// ([`github_archive_url`]) and the pre-resolved `commit_hash`
/// ([`resolve_ref_commit`]) — the snapshot has no `.git`, so the
/// commit cannot be read back from the tree. Modeled on
/// [`download_rc_tarball`] (gzip decode; codeload carries no sha256
/// manifest, so extraction is structurally verified —
/// [`promote_single_kernel_tree_transaction`] rejects any top level that is not a
/// single plain directory (multi-entry, a file, or a symlink) — and
/// the streamed digest is logged, not compared).
pub(crate) fn download_github_archive(
    client: &Client,
    archive_url: &str,
    git_ref: &str,
    commit_hash: &str,
    dest_dir: &Path,
    cli_label: &str,
    mp: Option<&crate::cli::FetchProgress>,
) -> Result<AcquiredSource> {
    let progress_label = format!("{cli_label}: {git_ref}");
    let snapshot_progress = match mp {
        Some(fp) => fp.operation_progress(&progress_label, "fetching snapshot", "snapshot fetch"),
        None => crate::cli::progress::CloneProgress::standalone_operation(
            &progress_label,
            "fetching snapshot",
            "snapshot fetch",
        ),
    };
    // Create the phase before issuing the request. The blocking HTTP
    // header wait itself has no byte counter, so this live item is what
    // lets CloneProgress emit its first phase immediately and a
    // ten-second heartbeat for as long as the server stays silent.
    let request_phase = snapshot_progress.item_named("requesting snapshot");
    tracing::info!(%archive_url, "downloading GitHub codeload snapshot (requires network)");
    let response = get_with_transient_retry(
        client,
        archive_url,
        Some(DOWNLOAD_REQUEST_READ_TIMEOUT),
        "download",
        &TRANSIENT_HTTP_RETRY,
    )?;
    drop(request_phase);
    if response.status() == reqwest::StatusCode::NOT_FOUND {
        anyhow::bail!(
            "codeload snapshot not found: {archive_url}\n  \
             the ref may not exist on the remote, or the repo is private"
        );
    }
    if !response.status().is_success() {
        anyhow::bail!("download {archive_url}: HTTP {}", response.status());
    }
    reject_html_response(&response, archive_url)?;
    print_download_size(&response, archive_url, cli_label, mp);
    // codeload responses are dynamically generated and often arrive
    // without a Content-Length, so `total` is frequently `None` and the
    // bar degrades to a live byte counter.
    let total = response.content_length();

    let status = |line: &str| match mp {
        Some(fp) => fp.println(line),
        None => eprintln!("{line}"),
    };
    status(&format!("{cli_label}: extracting snapshot (gzip)"));
    // Stage extraction inside `dest_dir` (same filesystem) so the final
    // atomic rename keeps `dest_dir` clean when a bad response serves a
    // malformed archive or sneaks stray top-level entries. codeload
    // snapshots have no upstream sha256 manifest, so structural
    // verification (single top-level dir) is the only defence against a
    // hostile response.
    let staging =
        tempfile::TempDir::new_in(dest_dir).with_context(|| "create extraction staging dir")?;
    let short_hash: String = commit_hash.chars().take(7).collect();
    let download_bar = mp.map(|fp| fp.download_bar(git_ref, total));
    // tar::Archive streams decompression directly from the response, so
    // download and extraction intentionally overlap instead of writing
    // a second full archive to disk. Name that fused phase truthfully;
    // the reporter continues to heartbeat even while gzip/tar is doing
    // CPU or filesystem work between response reads.
    let stream_phase = snapshot_progress.item_named("streaming and extracting snapshot");
    let stream = DownloadStream::with_progress(response, download_bar.as_ref().map(|b| b.bar()));
    let decoder = flate2::read::GzDecoder::new(stream);
    let mut archive = tar::Archive::new(decoder);
    ensure_source_operation_not_interrupted("before extracting GitHub snapshot")?;
    archive
        .unpack(staging.path())
        .with_context(|| "extract snapshot")?;
    ensure_source_operation_not_interrupted("after extracting GitHub snapshot")?;

    // Drain the watchdog to read the streamed digest. codeload has no
    // published manifest, so the digest cannot be verified — log it so
    // an operator can capture it for offline pinning. `into_inner` peels
    // the tar then the gz layer, recovering the `DownloadStream`.
    let stream = archive.into_inner().into_inner();
    let (actual_hex, bytes_total) = stream.finalize();
    drop(stream_phase);
    if let Some(bar) = &download_bar {
        bar.finish();
    }
    tracing::info!(
        url = %archive_url,
        bytes = bytes_total,
        sha256 = %actual_hex,
        "codeload snapshot extracted (unverified: codeload archives have \
         no published sha256 manifest)",
    );

    // Name the promoted tree by the resolved commit so distinct refs
    // never collide in `dest_dir` (the tree is temporary — `is_temp`).
    let canonical = format!("linux-git-{short_hash}");
    ensure_source_operation_not_interrupted("before promoting GitHub snapshot")?;
    let promoted = promote_single_kernel_tree_transaction(&staging, dest_dir, &canonical)?;
    if let Err(err) = ensure_source_operation_not_interrupted("after promoting GitHub snapshot") {
        return Err(promoted.rollback(err));
    }
    let source_dir = promoted.path.clone();
    ensure_source_operation_not_interrupted("before reading GitHub snapshot metadata")?;
    let version = read_makefile_version(&source_dir);
    if let Err(err) =
        ensure_source_operation_not_interrupted("after reading GitHub snapshot metadata")
    {
        return Err(promoted.rollback(err));
    }

    let acquired = AcquiredSource {
        source_dir,
        cache_key: git_cache_key(crate::kernel_path::GitRefKind::Sha, git_ref, commit_hash),
        version,
        kernel_source: crate::cache::KernelSource::git(short_hash, git_ref),
        is_temp: true,
        is_dirty: false,
        is_git: true,
    };
    let _source_dir = promoted.commit();
    snapshot_progress.finish();
    Ok(acquired)
}

/// Download a kernel tarball (stable or RC) and extract it.
///
/// `cli_label` prefixes diagnostic status output (e.g. `"ktstr"` or
/// `"cargo ktstr"`).
///
/// `skip_sha256` propagates to `download_stable_tarball` only —
/// stable tarballs publish a `sha256sums.asc` manifest the flag
/// bypasses. RC tarballs (`download_rc_tarball`) have no published
/// manifest so verification is impossible regardless of the flag;
/// the RC path always runs unverified and emits its own warning,
/// so `skip_sha256` is a no-op on the RC arm. `--kernel <path>` and
/// `--kernel git+…` sources do not reach this function at all.
///
/// `mp` is the progress group the determinate download bar is added
/// to; `None` disables the bar (the single-shot `kernel build` paths
/// and unit tests pass `None`).
pub fn download_tarball(
    client: &Client,
    version: &str,
    dest_dir: &Path,
    cli_label: &str,
    skip_sha256: bool,
    mp: Option<&crate::cli::FetchProgress>,
) -> Result<AcquiredSource> {
    let (arch, _) = arch_info();
    let source_dir = if is_rc(version) {
        download_rc_tarball(client, version, dest_dir, cli_label, mp)?
    } else {
        match download_stable_tarball(client, version, dest_dir, cli_label, skip_sha256, mp) {
            Ok(dir) => dir,
            // Pruned tarball (EOL series or superseded point release):
            // cdn.kernel.org keeps only each maintained series' latest
            // .tar.xz. Recover the source from the stable tree's
            // `v{version}` tag via a shallow (depth-1) clone. The kernel
            // built from this source is cached by the caller under the
            // SAME `{version}-tarball-...` key returned below, so a
            // re-run hits that cache and never re-clones.
            Err(e) if e.downcast_ref::<TarballNotFound>().is_some() => {
                let tag = format!("v{version}");
                // A 404 says the tarball is gone, not why. cdn.kernel.org
                // keeps only the latest tarball per series, but the gregkh
                // GitHub mirror carries every `vX.Y.Z` release tag and
                // codeload serves the tag's snapshot as a tar.gz — no
                // clone, and a commit-pinned snapshot. Resolve the tag to
                // its commit first (kind-directed, so a tag never aliases
                // a same-named branch); a tag absent there means the
                // version simply does not exist — surface the friendly
                // "not found" suggestion (with the latest in-series patch)
                // instead of a cryptic fetch failure.
                let Some(commit_id) = resolve_ref_commit(
                    STABLE_MIRROR_URL,
                    &tag,
                    crate::kernel_path::GitRefKind::Tag,
                    cli_label,
                    mp,
                )?
                else {
                    anyhow::bail!("{}", version_not_found_msg(client, version));
                };
                let commit_hash = format!("{commit_id}");
                let archive_url = github_archive_url(STABLE_MIRROR_URL, &commit_hash)
                    .expect("STABLE_MIRROR_URL is a github.com URL");
                let msg = format!(
                    "{cli_label}: {version} not on cdn.kernel.org (pruned/EOL); \
                     using the GitHub source-snapshot fallback for gregkh mirror tag {tag}"
                );
                match mp {
                    Some(fp) => fp.println(&msg),
                    None => eprintln!("{msg}"),
                }
                download_github_archive(
                    client,
                    &archive_url,
                    &tag,
                    &commit_hash,
                    dest_dir,
                    cli_label,
                    mp,
                )?
                .source_dir
            }
            Err(e) => return Err(e),
        }
    };

    Ok(AcquiredSource {
        source_dir,
        cache_key: format!("{version}-tarball-{arch}-kc{}", crate::cache_key_suffix()),
        version: Some(version.to_string()),
        kernel_source: crate::cache::KernelSource::Tarball,
        is_temp: true,
        is_dirty: false,
        is_git: true,
    })
}

/// Download a single file from `url` to `dest`, streaming through the
/// `DownloadStream` watchdog + progress bar and verifying the
/// resulting SHA-256 against `expected_sha256` (hex, case-insensitive).
///
/// Used by prebuilt-distro-kernel acquisition
/// (`crate::distro::acquire`) to fetch each `.rpm`/`.deb` package and
/// its (up to ~1 GiB) debuginfo. Rides the same transient-retry seam
/// (`TRANSIENT_HTTP_RETRY`) and no-progress watchdog
/// (`DOWNLOAD_NO_PROGRESS_TIMEOUT`, which tracks body progress rather
/// than total time so a large but progressing download is never
/// aborted) as the kernel-source tarball path. Unlike the tarball
/// downloaders it does not extract — the bytes land verbatim at `dest`
/// for the caller to unpack. `label` names the package on its progress
/// bar; `cli_label` prefixes the size status line.
pub fn download_verified_file(
    url: &str,
    dest: &Path,
    expected_sha256: &str,
    label: &str,
    cli_label: &str,
    mp: Option<&crate::cli::FetchProgress>,
) -> Result<()> {
    download_verified_file_with_checksum(
        url,
        dest,
        DownloadChecksum::Sha256(expected_sha256),
        label,
        cli_label,
        mp,
    )
}

/// Download a single file with the shared retry/watchdog/progress
/// pipeline and verify its Google Cloud Storage `md5Hash` (standard
/// base64). This is the GKE artifact counterpart to
/// [`download_verified_file`]; the transport and progress behavior are
/// deliberately identical.
pub(crate) fn download_verified_md5_file(
    url: &str,
    dest: &Path,
    expected_md5_base64: &str,
    label: &str,
    cli_label: &str,
    mp: Option<&crate::cli::FetchProgress>,
) -> Result<()> {
    download_verified_file_with_checksum(
        url,
        dest,
        DownloadChecksum::Md5Base64(expected_md5_base64),
        label,
        cli_label,
        mp,
    )
}

#[derive(Clone, Copy)]
enum DownloadChecksum<'a> {
    Sha256(&'a str),
    Md5Base64(&'a str),
}

fn download_verified_file_with_checksum(
    url: &str,
    dest: &Path,
    checksum: DownloadChecksum<'_>,
    label: &str,
    cli_label: &str,
    mp: Option<&crate::cli::FetchProgress>,
) -> Result<()> {
    let client = shared_client();
    tracing::info!(%url, "downloading verified kernel artifact (requires network)");
    let response = get_with_transient_retry(
        client,
        url,
        Some(DOWNLOAD_REQUEST_READ_TIMEOUT),
        "download",
        &TRANSIENT_HTTP_RETRY,
    )?;
    if response.status() == reqwest::StatusCode::NOT_FOUND {
        anyhow::bail!("package not found: {url}");
    }
    if !response.status().is_success() {
        anyhow::bail!("download {url}: HTTP {}", response.status());
    }
    reject_html_response(&response, url)?;
    print_download_size(&response, url, cli_label, mp);
    let total = response.content_length();
    let download_bar = mp.map(|fp| fp.download_bar(label, total));
    let parent = dest
        .parent()
        .ok_or_else(|| anyhow!("download destination has no parent: {}", dest.display()))?;
    let staging = tempfile::TempDir::new_in(parent)
        .with_context(|| format!("create download staging directory in {}", parent.display()))?;
    let staged_file = staging.path().join("artifact");
    let staged_result = (|| -> Result<()> {
        let mut stream =
            DownloadStream::with_progress(response, download_bar.as_ref().map(|b| b.bar()));
        let mut file = std::fs::File::create(&staged_file)
            .with_context(|| format!("create {}", staged_file.display()))?;
        std::io::copy(&mut stream, &mut file)
            .with_context(|| format!("stream {url} to {}", staged_file.display()))?;
        ensure_source_operation_not_interrupted("after writing downloaded artifact")?;
        if let Some(bar) = &download_bar {
            bar.finish();
        }
        match checksum {
            DownloadChecksum::Sha256(expected) => {
                let (actual, _bytes) = stream.finalize();
                verify_sha256(&actual, expected, url)?;
            }
            DownloadChecksum::Md5Base64(expected) => {
                let (actual, _bytes) = stream.finalize_md5_base64();
                verify_md5_base64(&actual, expected, url)?;
            }
        }
        drop(file);
        Ok(())
    })();
    if let Err(err) = staged_result {
        return Err(close_private_staging(staging, err));
    }
    let promoted = match promote_path_noreplace(&staged_file, dest, PublishedPathKind::File) {
        Ok(promoted) => promoted,
        Err(err) => return Err(close_private_staging(staging, err)),
    };
    drop(staging);
    promoted.commit();
    Ok(())
}

fn verify_md5_base64(actual: &str, expected: &str, url: &str) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        anyhow::bail!(
            "md5 mismatch for {url}: expected {expected}, got {actual}. \
             The object generation may have changed; retry so ktstr can \
             resolve fresh Google Cloud Storage metadata."
        );
    }
}

/// Parse the patch level from a kernel version string.
/// "6.12.8" → Some(8), "7.0" → Some(0), "abc" → None.
fn patch_level(version: &str) -> Option<u32> {
    let parts: Vec<&str> = version.split('.').collect();
    match parts.len() {
        2 => Some(0), // "7.0" has patch level 0
        3 => parts[2].parse().ok(),
        _ => None,
    }
}

/// Production URL for `releases.json`. Tests call [`fetch_releases`] directly with a localhost mock URL.
pub(crate) const RELEASES_URL: &str = "https://www.kernel.org/releases.json";

/// Fetch `releases.json` from `url` and return a vector of
/// [`Release`] records. Issues an HTTP GET unconditionally — no
/// cache consultation.
///
/// Production callers reach this function via
/// [`cached_releases_with`] (or [`cached_releases`]) which pass
/// [`RELEASES_URL`]; the cache helper only invokes
/// `fetch_releases` on a cache miss for the singleton path or on
/// the bypass branch for non-singleton clients. Tests that need
/// to exercise the underlying GET directly — without the cache
/// layer — call this function with a locally-constructed `Client`
/// and a localhost URL pointed at a TcpListener-backed mock that
/// returns canned `releases.json` content.
pub(crate) fn fetch_releases(client: &Client, url: &str) -> Result<Vec<Release>> {
    tracing::info!(%url, "fetching kernel.org releases index (requires network)");
    let response = get_with_transient_retry(
        client,
        url,
        Some(SMALL_RESPONSE_REQUEST_TIMEOUT),
        "fetch",
        &TRANSIENT_HTTP_RETRY,
    )?;
    if !response.status().is_success() {
        anyhow::bail!("fetch {url}: HTTP {}", response.status());
    }
    let body = String::from_utf8(read_response_bytes(response, url)?)
        .with_context(|| "decode response body as UTF-8")?;
    parse_releases_body(&body)
}

fn parse_releases_body(body: &str) -> Result<Vec<Release>> {
    let json: serde_json::Value =
        serde_json::from_str(body).with_context(|| "parse releases.json")?;
    let releases = json
        .get("releases")
        .and_then(|r| r.as_array())
        .ok_or_else(|| anyhow!("releases.json: missing releases array"))?;
    let input_rows = releases.len();
    let parsed: Vec<Release> = releases
        .iter()
        .filter_map(|r| {
            let moniker = r.get("moniker")?.as_str()?;
            let version = r.get("version")?.as_str()?;
            Some(Release {
                moniker: moniker.to_string(),
                version: version.to_string(),
            })
        })
        .collect();
    // Per-row tolerance: a corrupt row is silently dropped via the
    // filter_map `?` chain so a single bad entry does not abort the
    // whole fetch (see `fetch_releases_row_missing_moniker_drops_row`
    // and siblings). The drop is also a hazard: the truncated vector
    // gets cached in [`RELEASES_CACHE`] for the rest of the process
    // lifetime via the singleton path, so a transient malformed row
    // at fetch time persists as a partial snapshot for every later
    // cache-hit caller. Surface the drop count so an operator
    // tailing logs sees that releases.json arrived partial — without
    // this, the symptom (a missing version on resolve) is invisible
    // until it propagates as "version not found" elsewhere.
    let dropped = input_rows - parsed.len();
    if dropped > 0 {
        tracing::warn!(
            input_rows,
            parsed_rows = parsed.len(),
            dropped,
            "releases.json: dropped {dropped} of {input_rows} row(s) \
             missing moniker/version (or non-string values); cached \
             snapshot will reflect this for the process lifetime"
        );
    }
    Ok(parsed)
}

/// Fetch the latest stable kernel version from kernel.org.
///
/// Selects from the `releases` array (moniker "stable" or "longterm"),
/// requiring patch version >= 8 to avoid brand-new major versions
/// that may have build issues on CI runners.
///
/// When `client` is the process-wide [`shared_client`] singleton,
/// routes through `RELEASES_CACHE`; other clients bypass the
/// cache via pointer-equality and exercise `fetch_releases`
/// directly — see `cached_releases_with` for details.
///
/// `cli_label` prefixes diagnostic status output (e.g. `"ktstr"` or
/// `"cargo ktstr"`).
pub fn fetch_latest_stable_version(client: &Client, cli_label: &str) -> Result<String> {
    eprintln!("{cli_label}: fetching latest kernel version");
    let releases = cached_releases_with(client)?;

    let mut best: Option<&str> = None;
    for r in &releases {
        if r.moniker != "stable" && r.moniker != "longterm" {
            continue;
        }
        if patch_level(&r.version).unwrap_or(0) < 8 {
            continue;
        }
        // Pick the first matching release — releases.json is ordered
        // newest first, so the first stable with patch >= 8 is the best.
        best = Some(r.version.as_str());
        break;
    }

    let version =
        best.ok_or_else(|| anyhow!("no stable kernel with patch >= 8 found in releases.json"))?;
    eprintln!("{cli_label}: latest stable kernel: {version}");
    Ok(version.to_string())
}

/// Parse a version string into numeric components for comparison.
/// "6.14.2" → Some((6, 14, 2)), "6.14" → Some((6, 14, 0)),
/// "7.0" → Some((7, 0, 0)). Returns None for unparseable versions.
fn version_tuple(version: &str) -> Option<(u32, u32, u32)> {
    let parts: Vec<&str> = version.split('.').collect();
    match parts.len() {
        2 => {
            let major = parts[0].parse().ok()?;
            let minor = parts[1].parse().ok()?;
            Some((major, minor, 0))
        }
        3 => {
            let major = parts[0].parse().ok()?;
            let minor = parts[1].parse().ok()?;
            let patch = parts[2].parse().ok()?;
            Some((major, minor, patch))
        }
        _ => None,
    }
}

/// Return true when `s` is a kernel major.minor prefix like
/// `"6.14"` (as opposed to a full patch version `"6.14.2"` or an rc
/// tag `"6.15-rc3"`). Callers use this to decide whether the input
/// needs prefix resolution via [`fetch_version_for_prefix`].
///
/// Accepts any string with fewer than 2 dots and no `-rc` substring,
/// so `"7"` (single-segment) and `""` both return true. This matches
/// the historical inline check used by kernel-build dispatchers.
pub fn is_major_minor_prefix(s: &str) -> bool {
    s.matches('.').count() < 2 && !s.contains("-rc")
}

/// Resolve the highest version matching a prefix.
///
/// E.g., "6.12" → "6.12.81", "6" → "6.19.12" (highest 6.x.y).
///
/// Scans all monikers in releases.json except linux-next. On no active
/// match (an EOL or unreleased series, absent from releases.json),
/// resolves the highest `vX.Y.z` stable patch from the gregkh mirror's
/// git tags; if the series has NO stable point release yet (only a base
/// tag), falls back to the bare `{prefix}` mainline base — see
/// `latest_patch_from_git_tags`.
///
/// When `client` is the process-wide [`shared_client`] singleton,
/// routes through `RELEASES_CACHE`; other clients bypass the
/// cache via pointer-equality and exercise `fetch_releases`
/// directly — see `cached_releases_with` for details. Cache
/// scope is releases.json only; the EOL-series git-tag fallback in
/// `latest_patch_from_git_tags` always hits the network.
///
/// `cli_label` prefixes diagnostic status output (e.g. `"ktstr"` or
/// `"cargo ktstr"`).
pub fn fetch_version_for_prefix(client: &Client, prefix: &str, cli_label: &str) -> Result<String> {
    eprintln!("{cli_label}: fetching latest {prefix}.x kernel version");
    let releases = cached_releases_with(client)?;

    let mut best: Option<(&str, (u32, u32, u32))> = None;
    for r in &releases {
        if is_skippable_release_moniker(&r.moniker) {
            continue;
        }
        if !r.version.starts_with(prefix) {
            continue;
        }
        if r.version.len() != prefix.len() && r.version.as_bytes()[prefix.len()] != b'.' {
            continue;
        }
        let Some(tuple) = version_tuple(&r.version) else {
            continue;
        };
        if best.is_none() || tuple > best.unwrap().1 {
            best = Some((r.version.as_str(), tuple));
        }
    }

    if let Some((version, _)) = best {
        eprintln!("{cli_label}: latest {prefix}.x kernel: {version}");
        return Ok(version.to_string());
    }

    eprintln!(
        "{cli_label}: {prefix}.x not in releases.json (EOL or unreleased series); \
         resolving latest patch via the gregkh mirror tags"
    );
    match latest_patch_from_git_tags(STABLE_MIRROR_URL, prefix, cli_label)? {
        Some(version) => Ok(version),
        None => {
            // No stable point release for this series — fall back to the
            // mainline base (the `{prefix}` release itself, e.g. a series
            // just cut with no `.1` yet, per the "only if there is no
            // X.Y.z stable use X.Y mainline" rule). The base tarball is
            // fetched by the normal download path (cdn.kernel.org,
            // falling back to the gregkh mirror snapshot); torvalds is
            // the mainline authority the gregkh mirror tracks.
            eprintln!(
                "{cli_label}: no {prefix}.x stable point release; using {prefix} mainline base"
            );
            Ok(prefix.to_string())
        }
    }
}

/// Resolve a series' latest stable patch by ls-remote-ing the gregkh
/// GitHub mirror's `refs/tags/v{prefix}.{patch}` tags and taking the
/// highest patch. Returns `Ok(None)` when the series has NO stable
/// point release (no `v{prefix}.N` tag) — the caller then falls back to
/// the mainline base.
///
/// The gregkh mirror is the RELIABLE EOL-resolution source: it carries
/// every `vX.Y.Z` release tag (back to v2.6) and its codeload CDN
/// serves each tag's tarball, so resolution and the pruned-tarball
/// download (see [`download_tarball`]'s fallback) share ONE
/// comprehensive mirror. cdn.kernel.org cannot be used here: its
/// `v{major}.x/` directory index 404s, and its `sha256sums.asc` is
/// served inconsistently per CDN edge (200 from some nodes, 404 from
/// others — the 404 nodes break CI runners while the tarball fetch on
/// those same nodes still succeeds).
fn latest_patch_from_git_tags(url: &str, prefix: &str, cli_label: &str) -> Result<Option<String>> {
    eprintln!("{cli_label}: resolving {prefix}.x release tags via {url}");
    let refs = ls_remote_refs(url, &format!("{cli_label}: {prefix}.x tags"))
        .with_context(|| format!("ls-remote {url} for {prefix}.x release tags"))?;
    match max_tag_patch(refs.iter().map(ref_full_name), prefix) {
        Some(patch) => {
            let version = format!("{prefix}.{patch}");
            eprintln!("{cli_label}: latest {prefix}.x kernel (from git tags): {version}");
            Ok(Some(version))
        }
        None => Ok(None),
    }
}

/// The advertised full ref name (`refs/...`), as raw bytes, of a
/// protocol handshake ref.
fn ref_full_name(r: &gix::protocol::handshake::Ref) -> &[u8] {
    use gix::protocol::handshake::Ref::{Direct, Peeled, Symbolic, Unborn};
    match r {
        Peeled { full_ref_name, .. }
        | Direct { full_ref_name, .. }
        | Symbolic { full_ref_name, .. }
        | Unborn { full_ref_name, .. } => full_ref_name.as_ref(),
    }
}

/// Highest `{patch}` among `refs/tags/v{prefix}.{patch}` ref names.
///
/// gix folds an annotated tag's peeled entry into a single
/// `Ref::Peeled` whose `full_ref_name` is the BASE name — no `^{}`
/// suffix — and a lightweight tag arrives as a `Ref::Direct` with the
/// base name too, so every tag advertises its base
/// `refs/tags/v{prefix}.{patch}` name for the needle to match. The
/// `^{}` strip below is therefore a defensive no-op on real gix output
/// (it only affects a raw wire ref name gix never emits; the base
/// entry supplies the patch regardless). Pure (no network) so it is
/// unit-testable with synthetic ref names.
///
/// The trailing `.` in the `refs/tags/v{prefix}.` needle keeps a
/// `6.14` prefix from matching a `6.140` series, and the numeric-only
/// patch tail rejects `-rc` and other non-release tags.
fn max_tag_patch<'a>(ref_names: impl Iterator<Item = &'a [u8]>, prefix: &str) -> Option<u32> {
    let needle = format!("refs/tags/v{prefix}.");
    let mut best: Option<u32> = None;
    for name in ref_names {
        let Some(rest) = name.strip_prefix(needle.as_bytes()) else {
            continue;
        };
        let rest = rest.strip_suffix(b"^{}").unwrap_or(rest);
        if let Ok(s) = std::str::from_utf8(rest)
            && let Ok(patch) = s.parse::<u32>()
        {
            best = Some(best.map_or(patch, |b| b.max(patch)));
        }
    }
    best
}

/// ls-remote the gregkh stable mirror ([`STABLE_MIRROR_URL`]) once and
/// cache the release version strings (`X.Y.Z`) parsed from its
/// `refs/tags/vX.Y.Z` advertisement, for `--include-eol` range
/// expansion. Returns EVERY release-tag version verbatim (including
/// `-rc*` and old series); the caller
/// (`crate::cli::select_series_latest_in_range`) does the
/// range / rc / per-series filtering. `None` on ls-remote failure —
/// not cached, so the next caller retries. gregkh/linux mirrors
/// linux-stable comprehensively (tags back to v2.6), so this surfaces
/// EOL series that `releases.json` has dropped.
pub(crate) fn cached_stable_tags() -> Option<&'static [String]> {
    if let Some(tags) = STABLE_TAGS_CACHE.get() {
        return Some(tags.as_slice());
    }
    let refs = ls_remote_refs(STABLE_MIRROR_URL, "ktstr: stable release tags").ok()?;
    let tags: Vec<String> = refs
        .iter()
        .filter_map(|r| {
            // Base tag name only: gix folds an annotated tag's peeled
            // entry into one `Ref::Peeled` carrying the base name, and a
            // lightweight tag is a `Ref::Direct` with the base name, so
            // `^{}` never appears on real gix output — the strip is a
            // defensive no-op. Non-`refs/tags/v*` refs are skipped.
            let name = ref_full_name(r);
            let v = name.strip_prefix(b"refs/tags/v")?;
            let v = v.strip_suffix(b"^{}").unwrap_or(v);
            std::str::from_utf8(v).ok().map(|s| s.to_string())
        })
        .collect();
    // Loser of a concurrent race discards its clone (both fetched the
    // same advertisement, so the cached content is equivalent).
    let _ = STABLE_TAGS_CACHE.set(tags);
    STABLE_TAGS_CACHE.get().map(|v| v.as_slice())
}

/// Cache key for a git-acquired kernel: the explicit ref kind, a
/// fixed-seed ahash of the length-delimited raw ref bytes, the resolved
/// commit's full hash, the target arch, and the kconfig-fragment suffix.
///
/// The kind is part of the identity because `#tag=next`,
/// `#branch=next`, and `#sha=...` are distinct acquisition contracts
/// even when they currently resolve to the same commit. Hashing the raw
/// bytes avoids both filesystem-hostile ref spellings and collisions
/// introduced by sanitizing (`a/b` versus `a_b`) while keeping the
/// cache key well below `NAME_MAX`. The zero-seeded ahash matches
/// ktstr's other fast content-addressed caches; this is an identity
/// accelerator, not a cryptographic integrity boundary.
///
/// The FULL 40-hex commit hash keys the entry (not a 7-hex prefix): a
/// branch/tag tip moves over time, so the `{git_ref}` segment alone
/// cannot distinguish successive commits — only the hash does. A 7-hex
/// (28-bit) prefix would let a moved tip whose new commit shares the
/// first 7 hex with the cached old commit hit the stale entry and serve
/// the wrong kernel build under the new ref. The full id removes that
/// collision class; the probe and clone both render full lowercase hex
/// before any truncation, so keying on it is drift-free.
pub(crate) fn git_cache_key(
    ref_kind: crate::kernel_path::GitRefKind,
    git_ref: &str,
    commit_hash: &str,
) -> String {
    use crate::kernel_path::GitRefKind;

    let (arch, _) = arch_info();
    let (kind_name, kind_byte) = match ref_kind {
        GitRefKind::Tag => ("tag", 0u8),
        GitRefKind::Branch => ("branch", 1u8),
        GitRefKind::Sha => ("sha", 2u8),
        GitRefKind::Unknown => ("unknown", 3u8),
    };
    let canonical_ref = (ref_kind == GitRefKind::Sha).then(|| git_ref.to_ascii_lowercase());
    let ref_bytes = canonical_ref.as_deref().unwrap_or(git_ref).as_bytes();
    let mut hasher = ahash::RandomState::with_seeds(0, 0, 0, 0).build_hasher();
    hasher.write(&[kind_byte]);
    hasher.write(&(ref_bytes.len() as u64).to_le_bytes());
    hasher.write(ref_bytes);
    let ref_hash = hasher.finish();
    format!(
        "git-{kind_name}-{ref_hash:016x}-{}-{arch}-kc{}",
        commit_hash.to_ascii_lowercase(),
        crate::cache_key_suffix()
    )
}

/// If `url` is a GitHub remote, build the codeload archive URL for the
/// resolved `commit_hash`: `github.com/OWNER/REPO/archive/<commit>.tar.gz`
/// (302 → codeload.github.com, its CDN) serves a gzip source snapshot
/// for any commit — verified empirically. `resolve_git_kernel` invokes
/// this only for an explicit immutable SHA; branches and tags use the
/// exact shallow gix path even on GitHub. `None` for a non-GitHub URL
/// (self-hosted / GitLab / …), where commit-SHA acquisition is not
/// supported.
///
/// The caller supplies an immutable `commit_hash` (an explicit SHA, or
/// the strict stable-tag fallback's resolved commit), so the download
/// fetches the exact commit the cache/source record names.
/// `commit_hash` is lowercased to align with `git_cache_key`.
///
/// Accepts the https/http/ssh/git and scp-style GitHub remotes, each
/// with an optional trailing `/` and `.git`; the host is matched
/// case-insensitively (DNS hostnames are case-insensitive).
pub(crate) fn github_archive_url(url: &str, commit_hash: &str) -> Option<String> {
    // Match the github.com scheme+host CASE-INSENSITIVELY (DNS
    // hostnames are case-insensitive, so `GitHub.com` is a GitHub URL),
    // keeping the OWNER/REPO path verbatim. Accept the https/http/ssh/git
    // schemes (with an optional `git@` userinfo) and the scp-style
    // git@github.com:OWNER/REPO, each with an optional trailing `.git`.
    let mut path = None;
    for prefix in [
        "https://github.com/",
        "http://github.com/",
        "ssh://git@github.com/",
        "ssh://github.com/",
        "git://github.com/",
        "git@github.com:",
    ] {
        if url
            .get(..prefix.len())
            .is_some_and(|head| head.eq_ignore_ascii_case(prefix))
        {
            path = Some(&url[prefix.len()..]);
            break;
        }
    }
    let path = path?;
    // Trim trailing slashes (a common copy-paste artifact) before the
    // `.git` strip so `OWNER/REPO/` and `OWNER/REPO.git/` still resolve
    // to codeload rather than misrouting to the clone path.
    let path = path.trim_end_matches('/');
    let path = path.strip_suffix(".git").unwrap_or(path);
    // Exactly OWNER/REPO — reject deeper paths (a stray extra segment
    // is not a repo root, so fall through to the clone path).
    let mut segs = path.split('/');
    let owner = segs.next().filter(|s| !s.is_empty())?;
    let repo = segs.next().filter(|s| !s.is_empty())?;
    if segs.next().is_some() {
        return None;
    }
    // Always the resolved COMMIT (lowercased) — never a ref-name
    // snapshot — so the extracted tree matches git_cache_key's commit
    // exactly regardless of a concurrent branch-tip move. codeload
    // serves any commit case-insensitively.
    Some(format!(
        "https://github.com/{owner}/{repo}/archive/{}.tar.gz",
        commit_hash.to_ascii_lowercase()
    ))
}

/// The object id the advertised ref named exactly `target` points at,
/// or `None` if no ref matches. For an annotated tag (`Ref::Peeled`)
/// this is the PEELED commit (`object`), never the tag object;
/// `Ref::Unborn` carries no commit and never matches. Used by the
/// kind-directed [`resolve_ref_commit`] so tag-peeling and
/// unborn-skipping stay consistent.
fn pick_ref_object(
    refs: &[gix::protocol::handshake::Ref],
    target: &str,
) -> Option<gix::hash::ObjectId> {
    refs.iter().find_map(|r| {
        let (name, object) = match r {
            gix::protocol::handshake::Ref::Peeled {
                full_ref_name,
                object,
                ..
            }
            | gix::protocol::handshake::Ref::Direct {
                full_ref_name,
                object,
            }
            | gix::protocol::handshake::Ref::Symbolic {
                full_ref_name,
                object,
                ..
            } => (full_ref_name, object),
            gix::protocol::handshake::Ref::Unborn { .. } => return None,
        };
        (*name == target).then_some(*object)
    })
}

/// Resolve `git_ref` to its full commit hash under `ref_kind`, via a
/// kind-directed exact-ref discovery. This standalone helper serves
/// stable-tag fallback and metadata callers that genuinely need
/// ls-remote semantics. Normal branch/tag acquisition instead reads the
/// peeled id from its single receive `Prepare`, avoiding a second
/// handshake. An explicit SHA resolves offline.
///
/// A `Sha` ref IS the commit (lowercased to match `git_clone`'s
/// rendering) and resolves offline — no handshake. `Tag`/`Branch`
/// match ONLY the fully-qualified `refs/tags/{ref}` / `refs/heads/{ref}`
/// so a tag never aliases a same-named branch (a bare-name DWIM lookup
/// would resolve either). `Ok(None)` means the exact ref was not
/// advertised or the kind is `Unknown`. Transport, authentication,
/// protocol, and cancellation failures propagate to the caller.
pub(crate) fn resolve_ref_commit(
    url: &str,
    git_ref: &str,
    ref_kind: crate::kernel_path::GitRefKind,
    cli_label: &str,
    mp: Option<&crate::cli::FetchProgress>,
) -> Result<Option<gix::ObjectId>> {
    use crate::kernel_path::GitRefKind;
    let target = match ref_kind {
        GitRefKind::Sha => {
            let canonical = git_ref.to_ascii_lowercase();
            return gix::ObjectId::from_hex(canonical.as_bytes())
                .with_context(|| format!("parse explicit git commit {git_ref}"))
                .map(Some);
        }
        GitRefKind::Tag => format!("refs/tags/{git_ref}"),
        GitRefKind::Branch => format!("refs/heads/{git_ref}"),
        GitRefKind::Unknown => return Ok(None),
    };
    match gix_policy::classify_source(url).map_err(anyhow::Error::msg)? {
        gix_policy::InProcessSource::Local(path) => {
            return resolve_local_ref_commit(&path, &target, &SOURCE_OPERATION_INTERRUPTED);
        }
        gix_policy::InProcessSource::Http | gix_policy::InProcessSource::Https => {}
    }
    let progress_label = format!("{cli_label}: {git_ref}");
    let ref_progress = match mp {
        Some(fp) => fp.operation_progress(&progress_label, "resolving git ref", "ref discovery"),
        None => crate::cli::progress::CloneProgress::standalone_operation(
            &progress_label,
            "resolving git ref",
            "ref discovery",
        ),
    };
    let refs = discover_remote_refs(
        url,
        Some(&target),
        &ref_progress,
        &SOURCE_OPERATION_INTERRUPTED,
    )?;
    let object = pick_ref_object(&refs, &target);
    ref_progress.finish();
    Ok(object)
}

fn resolve_local_ref_commit(
    repository_path: &Path,
    target: &str,
    interrupt: &std::sync::atomic::AtomicBool,
) -> Result<Option<gix::ObjectId>> {
    ensure_git_operation_not_interrupted(interrupt, "before opening local repository")?;
    let repo = gix::open_opts(repository_path, anon_open_opts())
        .with_context(|| format!("open local source repository {}", repository_path.display()))?;
    ensure_git_operation_not_interrupted(interrupt, "after opening local repository")?;
    let Some(mut reference) = repo
        .try_find_reference(target)
        .with_context(|| format!("find local source ref {target}"))?
    else {
        return Ok(None);
    };
    let commit_id = reference
        .peel_to_commit()
        .with_context(|| format!("peel local source ref {target} to a commit"))?
        .id;
    #[cfg(test)]
    if TEST_INTERRUPT_AFTER_REF_DISCOVERY.swap(false, Ordering::AcqRel) {
        interrupt.store(true, Ordering::Release);
    }
    ensure_git_operation_not_interrupted(interrupt, "after reading local ref")?;
    Ok(Some(commit_id))
}

/// ls-remote `url` and return EVERY advertised ref WITHOUT fetching a
/// pack. Callers decide explicitly whether failure is fatal:
/// correctness-sensitive resolution propagates it, while optional
/// historical-tag enrichment degrades with `.ok()`.
///
/// The ad-hoc repo (`init_opts` on a tempdir, with repo-local git config
/// only — see `anon_open_opts`) carries no working tree and fetches no
/// pack. Remote-side ref-prefix filtering is
/// DISABLED: gix's default (`prefix_from_spec_as_filter_on_remote =
/// true`) derives protocol-v2 `ls-refs` `ref-prefix` filters from the
/// remote's fetch refspecs; an anonymous `remote_at` has none, and
/// `fetch_tags = Included` injects only `refs/tags/*`, so the server
/// would return TAGS ONLY and `refs/heads/*` would never arrive.
/// Disabling the filter returns all refs, so a branch, tag, or HEAD
/// all resolve.
fn ls_remote_refs(url: &str, progress_label: &str) -> Result<Vec<gix::protocol::handshake::Ref>> {
    let ref_progress = crate::cli::progress::CloneProgress::standalone_operation(
        progress_label,
        "resolving git refs",
        "ref discovery",
    );
    let refs = discover_remote_refs(url, None, &ref_progress, &SOURCE_OPERATION_INTERRUPTED)?;
    ref_progress.finish();
    Ok(refs)
}

/// Discover either one exact fully-qualified ref or the remote's full
/// advertisement without fetching a pack.
///
/// `exact_ref = Some` installs one exact fetch refspec and leaves
/// protocol-v2 prefix filtering enabled, so a normal branch/tag cache
/// probe asks the server only for that namespace-qualified ref. The
/// `None` form deliberately disables filtering for stable-tag
/// enumeration. Both forms create their progress phase before
/// `connect`: the first blocking network wait is therefore visible and
/// receives CloneProgress's ten-second heartbeat.
fn discover_remote_refs(
    url: &str,
    exact_ref: Option<&str>,
    progress: &crate::cli::progress::CloneProgress,
    interrupt: &std::sync::atomic::AtomicBool,
) -> Result<Vec<gix::protocol::handshake::Ref>> {
    ensure_git_operation_not_interrupted(interrupt, "before preparing ref discovery")?;
    match gix_policy::classify_source(url).map_err(anyhow::Error::msg)? {
        gix_policy::InProcessSource::Http | gix_policy::InProcessSource::Https => {}
        gix_policy::InProcessSource::Local(path) => anyhow::bail!(
            "local ref discovery must open {} directly instead of constructing a remote transport",
            path.display()
        ),
    }
    let tmp = tempfile::TempDir::new().with_context(|| "create ref discovery staging dir")?;
    ensure_git_operation_not_interrupted(interrupt, "after preparing ref discovery")?;
    let repo = gix::ThreadSafeRepository::init_opts(
        tmp.path(),
        gix::create::Kind::WithWorktree,
        gix::create::Options::default(),
        anon_open_opts(),
    )
    .with_context(|| "prepare ref discovery repository")?
    .to_thread_local();
    ensure_git_operation_not_interrupted(interrupt, "after initializing ref discovery")?;
    let mut remote = repo
        .remote_at_without_url_rewrite(url)
        .with_context(|| format!("prepare ref discovery remote {url}"))?;
    ensure_git_operation_not_interrupted(interrupt, "after preparing ref discovery remote")?;

    if let Some(source_ref) = exact_ref {
        // A private throwaway destination is enough to derive the exact
        // protocol-v2 ref-prefix; no ref is written because this is only
        // a ref-map operation.
        const DISCOVERY_REF: &str = "refs/ktstr/discovery";
        let refspec = format!("+{source_ref}:{DISCOVERY_REF}");
        remote = remote.with_fetch_tags(gix::remote::fetch::Tags::None);
        remote
            .replace_refspecs([refspec.as_str()], gix::remote::Direction::Fetch)
            .with_context(|| format!("set exact discovery refspec for {source_ref}"))?;
        ensure_git_operation_not_interrupted(interrupt, "after setting discovery refspec")?;
    }

    // This phase exists before connect(), whose handshake has no
    // progress or interrupt parameter. The shared process interrupt is
    // checked on both sides of that non-interruptible boundary.
    let discovery_progress = progress.item_named("discovering refs");
    ensure_git_operation_not_interrupted(interrupt, "before connecting for ref discovery")?;
    let mut connection = remote
        .connect(gix::remote::Direction::Fetch)
        .with_context(|| format!("connect to {url} for ref discovery"))?;
    connection.set_credentials(gix_policy::reject_credentials);
    ensure_git_operation_not_interrupted(interrupt, "after connecting for ref discovery")?;
    let options = gix::remote::ref_map::Options {
        prefix_from_spec_as_filter_on_remote: exact_ref.is_some(),
        ..Default::default()
    };
    ensure_git_operation_not_interrupted(interrupt, "before reading remote refs")?;
    let (refmap, _handshake) = connection
        .ref_map(discovery_progress, options)
        .with_context(|| format!("read remote refs from {url}"))?;
    #[cfg(test)]
    if TEST_INTERRUPT_AFTER_REF_DISCOVERY.swap(false, Ordering::AcqRel) {
        interrupt.store(true, Ordering::Release);
    }
    ensure_git_operation_not_interrupted(interrupt, "after reading remote refs")?;
    Ok(refmap.remote_refs)
}

/// Shared hermetic options for runtime ref discovery, remote acquisition,
/// and directly opened local repositories.
fn anon_open_opts() -> gix::open::Options {
    gix_policy::open_options()
}

/// Shallow-clone a git repository at a BRANCH ref.
///
/// `cli_label` prefixes diagnostic status output (e.g. `"ktstr"` or
/// `"cargo ktstr"`).
///
/// `mp` is the progress group a determinate clone bar is added to.
/// Without a group, the clone still drives a hidden CloneProgress tree
/// and emits escape-free ten-second heartbeats, so single-shot
/// `kernel build` cannot disappear during a slow handshake or pack
/// index. The bar shows real object/file counts + ETA during the
/// receiving / resolving / checkout phases that gix reports a bounded
/// total for; see the `crate::cli::progress` module.
///
/// For a TAG ref use the test-visible `git_clone_tag` wrapper. Both entry points share the
/// same lower-level exact-ref fetch; the wrapper selects the native
/// `refs/heads/*` or `refs/tags/*` source namespace.
pub fn git_clone(
    url: &str,
    git_ref: &str,
    dest_dir: &Path,
    cli_label: &str,
    mp: Option<&crate::cli::FetchProgress>,
) -> Result<AcquiredSource> {
    git_clone_inner(
        url,
        git_ref,
        crate::kernel_path::GitRefKind::Branch,
        dest_dir,
        cli_label,
        mp,
    )
}

/// Shallow-clone a git repository at a TAG ref (e.g. `v6.14.11`).
///
/// The exact fetch maps `refs/tags/{tag}` to ktstr's private fetched
/// ref, peels annotated tags to their commit for checkout, and retains
/// the original tag object under its native name for `git describe` /
/// `setlocalversion`. The `#tag=` source (via
/// [`git_clone_kinded`]) uses this path for every remote, including
/// GitHub. The pruned/EOL tarball recovery is a separately reported
/// archive fallback in [`download_tarball`].
#[cfg(test)]
pub(crate) fn git_clone_tag(
    url: &str,
    tag: &str,
    dest_dir: &Path,
    cli_label: &str,
    mp: Option<&crate::cli::FetchProgress>,
) -> Result<AcquiredSource> {
    git_clone_inner(
        url,
        tag,
        crate::kernel_path::GitRefKind::Tag,
        dest_dir,
        cli_label,
        mp,
    )
}

/// Clone a git source at `git_ref`, dispatching on `ref_kind` to the
/// correct clone path. Branches and tags use this exact shallow gix
/// path uniformly, including well-formed GitHub remotes. Only an
/// explicit immutable GitHub `Sha` is routed to
/// [`download_github_archive`] by [`crate::cli::resolve_git_kernel`].
///
/// - `Tag` → [`git_clone_tag`] (exact `refs/tags/*` source).
/// - `Branch` → [`git_clone`] (exact `refs/heads/*` source).
/// - `Sha` → a hard error: gix cannot fetch a bare commit, and a
///   self-hosted server generally lacks allow-sha-in-want. The
///   actionable message points at GitHub (codeload serves any sha) or a
///   tag/branch.
/// - `Unknown` → a hard error; [`crate::kernel_path::KernelId::validate`]
///   rejects it upstream, so this is a defensive backstop.
#[cfg(test)]
pub(crate) fn git_clone_kinded(
    url: &str,
    git_ref: &str,
    ref_kind: crate::kernel_path::GitRefKind,
    dest_dir: &Path,
    cli_label: &str,
    mp: Option<&crate::cli::FetchProgress>,
) -> Result<AcquiredSource> {
    use crate::kernel_path::GitRefKind;
    match ref_kind {
        GitRefKind::Tag => git_clone_tag(url, git_ref, dest_dir, cli_label, mp),
        GitRefKind::Branch => git_clone(url, git_ref, dest_dir, cli_label, mp),
        GitRefKind::Sha => anyhow::bail!(
            "git+{url}#sha={git_ref}: fetching this source by commit sha is \
             not supported — gix cannot fetch a bare commit and the remote \
             lacks allow-sha-in-want. Use a github.com/OWNER/REPO URL \
             (codeload serves any commit) or pin a #tag= / #branch= instead."
        ),
        GitRefKind::Unknown => anyhow::bail!(
            "git+{url}: ref kind could not be determined; use #tag=NAME, \
             #branch=NAME, or #sha=<40-hex>"
        ),
    }
}

/// Shared shallow-clone implementation for [`git_clone`] (branch) and
/// `git_clone_tag` (tag).
///
/// This deliberately uses gix's lower-level fetch API rather than
/// `clone::PrepareFetch`. The clone helper adds the remote's `HEAD` to
/// every fetch and, for its ordinary shallow branch path, changes tag
/// handling to `Tags::All`. On a kernel repository that turns a
/// depth-one branch checkout into a fetch and pack-index of every
/// release tag. An exact fully-qualified refspec keeps negotiation to
/// the one branch or tag the caller requested. Branches and tags use
/// the same fetch and checkout path; only their source namespace
/// differs.
fn git_clone_inner(
    url: &str,
    git_ref: &str,
    ref_kind: crate::kernel_path::GitRefKind,
    dest_dir: &Path,
    cli_label: &str,
    mp: Option<&crate::cli::FetchProgress>,
) -> Result<AcquiredSource> {
    match git_clone_inner_gated(url, git_ref, ref_kind, dest_dir, cli_label, mp, |_| {
        Ok(ExactRefGate::Fetch(()))
    })? {
        GitCloneOutcome::Fetched { acquired, .. } => Ok(acquired),
        GitCloneOutcome::Skipped { .. } => unreachable!("unconditional clone cannot skip"),
    }
}

pub(crate) enum GitCloneOutcome<T> {
    Fetched { acquired: AcquiredSource, token: T },
    Skipped { commit_hash: String, token: T },
}

/// Exact branch/tag acquisition with a cache/election callback executed
/// after the single gix ref-map and before receive. Used by
/// `resolve_git_kernel` to avoid a separate discovery handshake.
pub(crate) fn git_clone_kinded_gated<T>(
    url: &str,
    git_ref: &str,
    ref_kind: crate::kernel_path::GitRefKind,
    dest_dir: &Path,
    cli_label: &str,
    mp: Option<&crate::cli::FetchProgress>,
    gate: impl FnOnce(gix::ObjectId) -> Result<ExactRefGate<T>>,
) -> Result<GitCloneOutcome<T>> {
    use crate::kernel_path::GitRefKind;
    match ref_kind {
        GitRefKind::Branch | GitRefKind::Tag => {
            git_clone_inner_gated(url, git_ref, ref_kind, dest_dir, cli_label, mp, gate)
        }
        GitRefKind::Sha => anyhow::bail!(
            "git+{url}#sha={git_ref}: fetching this source by commit sha is \
             not supported — use a github.com/OWNER/REPO URL or pin a \
             #tag= / #branch= instead"
        ),
        GitRefKind::Unknown => anyhow::bail!(
            "git+{url}: ref kind could not be determined; use #tag=NAME, \
             #branch=NAME, or #sha=<40-hex>"
        ),
    }
}

fn close_private_staging(staging: tempfile::TempDir, primary: anyhow::Error) -> anyhow::Error {
    match staging.close() {
        Ok(()) => primary,
        Err(cleanup_err) => primary.context(format!(
            "also failed to remove private source-acquisition staging directory: {cleanup_err}"
        )),
    }
}

fn git_clone_inner_gated<T>(
    url: &str,
    git_ref: &str,
    ref_kind: crate::kernel_path::GitRefKind,
    dest_dir: &Path,
    cli_label: &str,
    mp: Option<&crate::cli::FetchProgress>,
    gate: impl FnOnce(gix::ObjectId) -> Result<ExactRefGate<T>>,
) -> Result<GitCloneOutcome<T>> {
    let cloning = format!("{cli_label}: cloning {url} (ref: {git_ref}, depth: 1)");
    match mp {
        Some(fp) => fp.println(&cloning),
        None => eprintln!("{cloning}"),
    }

    let clone_dir = dest_dir.join("linux");
    let staging = tempfile::TempDir::new_in(dest_dir)
        .with_context(|| format!("create private clone staging dir in {}", dest_dir.display()))?;

    // Drive a determinate clone bar from gix's progress tree (see
    // [`crate::cli::progress::CloneProgress`]). A shared group renders
    // coordinated bars; a command without one still gets the same
    // plain-text ten-second heartbeat through a standalone reporter.
    // Every phase polls the process-level signal bridge set by
    // cargo-ktstr's interrupt handler.
    let progress_label = format!("{cli_label}: {git_ref}");
    let clone_progress = match mp {
        Some(fp) => fp.clone_progress(&progress_label),
        None => crate::cli::progress::CloneProgress::standalone_operation(
            &progress_label,
            "cloning",
            "clone",
        ),
    };
    let fetched = match fetch_exact_ref_and_checkout_gated(
        url,
        git_ref,
        ref_kind,
        staging.path(),
        Some(&clone_progress),
        &SOURCE_OPERATION_INTERRUPTED,
        gate,
    ) {
        Ok(outcome) => outcome,
        Err(err) => {
            return Err(close_private_staging(staging, err));
        }
    };
    let (commit_id, token) = match fetched {
        ExactRefFetch::Skipped { commit_id, token } => {
            let commit_hash = format!("{commit_id}");
            staging.close().with_context(
                || "remove private clone staging directory after cache/election skip",
            )?;
            return Ok(GitCloneOutcome::Skipped { commit_hash, token });
        }
        ExactRefFetch::Fetched { commit_id, token } => (commit_id, token),
    };

    let acquired = (|| -> Result<AcquiredSource> {
        // FULL commit hash keys the cache (see `git_cache_key` — a 7-hex
        // prefix risks a moved-tip collision serving a stale build); the
        // 7-hex `short_hash` is kept only for the human-facing source
        // record.
        let commit_hash = format!("{commit_id}");
        let short_hash = commit_hash.chars().take(7).collect::<String>();
        let cache_key = git_cache_key(ref_kind, git_ref, &commit_hash);

        // Record the kernel version from the checked-out source
        // Makefile, as local_source does. The filesystem read is the
        // final non-interruptible boundary after index publication, so
        // observe the shared epoch on both sides before making this
        // source visible to the cache/build pipeline.
        ensure_source_operation_not_interrupted("before reading cloned source metadata")?;
        let version = read_makefile_version(staging.path());
        ensure_source_operation_not_interrupted("after reading cloned source metadata")?;

        Ok(AcquiredSource {
            source_dir: clone_dir.clone(),
            cache_key,
            version,
            kernel_source: crate::cache::KernelSource::git(short_hash, git_ref),
            is_temp: true,
            is_dirty: false,
            is_git: true,
        })
    })();
    match acquired {
        Ok(acquired) => {
            let promoted = match promote_path_noreplace(
                staging.path(),
                &clone_dir,
                PublishedPathKind::Directory,
            ) {
                Ok(promoted) => promoted,
                Err(err) => return Err(close_private_staging(staging, err)),
            };
            #[cfg(test)]
            if TEST_INTERRUPT_AFTER_CLONE_PROMOTE.swap(false, Ordering::AcqRel) {
                SOURCE_OPERATION_INTERRUPTED.store(true, Ordering::Release);
            }
            if let Err(err) = ensure_source_operation_not_interrupted("after promoting exact clone")
            {
                drop(staging);
                return Err(promoted.rollback(err));
            }
            drop(staging);
            let source_dir = promoted.commit();
            debug_assert_eq!(source_dir, acquired.source_dir);
            clone_progress.finish();
            Ok(GitCloneOutcome::Fetched { acquired, token })
        }
        Err(err) => Err(close_private_staging(staging, err)),
    }
}

/// Measured ceiling for one gix pack-index or checkout operation.
///
/// A live sched_ext `for-next` depth-one fetch on the 64-CPU
/// development host took 55.0s / 705MB with all CPUs, 57.6s / 454MB
/// with eight, and 58.0s / 591MB with sixteen. Eight therefore keeps
/// essentially all transfer throughput while avoiding hundreds of MB
/// of per-clone worker state. Smaller hosts use their available CPU
/// count instead.
const GIX_WORKERS_PER_OPERATION: usize = 8;

/// Machine-wide, non-blocking lease for checkout's optional extra
/// workers. Every checkout always has its calling thread. It then tries
/// to flock up to seven numbered tokens in `KTSTR_LOCK_DIR` (or `/tmp`)
/// and uses one worker per acquired token. Contending processes simply
/// get fewer extras; no source acquisition is ever queued behind this
/// performance optimization.
struct GixCheckoutWorkerLease {
    extra_workers: Vec<std::os::fd::OwnedFd>,
}

impl GixCheckoutWorkerLease {
    fn acquire() -> Self {
        let wanted = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1)
            .min(GIX_WORKERS_PER_OPERATION)
            .saturating_sub(1);
        let lock_dir = crate::cache::resolve_lock_dir();
        Self::acquire_in(&lock_dir, wanted)
    }

    fn acquire_in(lock_dir: &Path, wanted: usize) -> Self {
        let wanted = wanted.min(GIX_WORKERS_PER_OPERATION.saturating_sub(1));
        if let Err(err) = std::fs::create_dir_all(lock_dir) {
            tracing::warn!(
                %err,
                path = %lock_dir.display(),
                "could not create gix worker-token directory; checkout will use its caller thread",
            );
            return Self {
                extra_workers: Vec::new(),
            };
        }
        let mut extra_workers = Vec::with_capacity(wanted);
        // `wanted` is also the size of the machine-wide namespace.
        // Scanning all seven ceiling slots on a two-CPU host would let
        // seven processes each lease a different extra despite the
        // host-wide budget being one.
        for slot in 0..wanted {
            if extra_workers.len() == wanted {
                break;
            }
            let path = lock_dir.join(format!("ktstr-gix-checkout-worker-{slot}.lock"));
            match crate::flock::try_flock(&path, crate::flock::FlockMode::Exclusive) {
                Ok(Some(fd)) => extra_workers.push(fd),
                Ok(None) => {}
                Err(err) => {
                    tracing::warn!(
                        %err,
                        path = %path.display(),
                        "could not lease gix checkout worker token; continuing with fewer workers",
                    );
                }
            }
        }
        Self { extra_workers }
    }

    fn workers(&self) -> usize {
        1 + self.extra_workers.len()
    }
}

/// Fetch one fully-qualified branch or tag at depth one and materialize
/// its peeled commit into `clone_dir`.
pub(crate) enum ExactRefGate<T> {
    Fetch(T),
    Skip(T),
}

enum ExactRefFetch<T> {
    Fetched { commit_id: gix::ObjectId, token: T },
    Skipped { commit_id: gix::ObjectId, token: T },
}

#[cfg(test)]
fn fetch_exact_ref_and_checkout(
    url: &str,
    git_ref: &str,
    ref_kind: crate::kernel_path::GitRefKind,
    clone_dir: &Path,
    clone_progress: Option<&crate::cli::progress::CloneProgress>,
    interrupt: &std::sync::atomic::AtomicBool,
) -> Result<gix::ObjectId> {
    match fetch_exact_ref_and_checkout_gated(
        url,
        git_ref,
        ref_kind,
        clone_dir,
        clone_progress,
        interrupt,
        |_| Ok(ExactRefGate::Fetch(())),
    )? {
        ExactRefFetch::Fetched { commit_id, .. } => Ok(commit_id),
        ExactRefFetch::Skipped { .. } => unreachable!("unconditional exact fetch cannot skip"),
    }
}

/// Prepare one exact gix fetch, expose its advertised peeled commit to
/// the caller before receiving a pack, and reuse that same `Prepare`
/// when the caller wins the content-key builder election.
fn fetch_exact_ref_and_checkout_gated<T>(
    url: &str,
    git_ref: &str,
    ref_kind: crate::kernel_path::GitRefKind,
    clone_dir: &Path,
    clone_progress: Option<&crate::cli::progress::CloneProgress>,
    interrupt: &std::sync::atomic::AtomicBool,
    gate: impl FnOnce(gix::ObjectId) -> Result<ExactRefGate<T>>,
) -> Result<ExactRefFetch<T>> {
    use crate::kernel_path::GitRefKind;

    let source_ref = match ref_kind {
        GitRefKind::Branch => format!("refs/heads/{git_ref}"),
        GitRefKind::Tag => format!("refs/tags/{git_ref}"),
        GitRefKind::Sha | GitRefKind::Unknown => {
            anyhow::bail!("exact shallow fetch requires a branch or tag")
        }
    };
    if let gix_policy::InProcessSource::Local(repository_path) =
        gix_policy::classify_source(url).map_err(anyhow::Error::msg)?
    {
        return materialize_local_exact_ref_gated(
            &repository_path,
            &source_ref,
            clone_dir,
            clone_progress,
            interrupt,
            gate,
        );
    }
    // A fixed private destination means arbitrary valid branch/tag names
    // never have to be re-encoded as a local branch name.
    const FETCHED_REF: &str = "refs/ktstr/source";
    let refspec = format!("+{source_ref}:{FETCHED_REF}");

    ensure_git_operation_not_interrupted(interrupt, "before preparing clone")?;
    let mut repo = gix::ThreadSafeRepository::init_opts(
        clone_dir,
        gix::create::Kind::WithWorktree,
        gix::create::Options::default(),
        anon_open_opts(),
    )
    .with_context(|| "prepare clone")?
    .to_thread_local();
    ensure_git_operation_not_interrupted(interrupt, "after preparing clone")?;

    // Pack indexing runs on the caller thread. Checkout separately
    // leases machine-wide optional workers immediately before it starts;
    // no worker token is held across connect or ref mapping.
    ensure_git_operation_not_interrupted(interrupt, "before configuring pack workers")?;
    repo.config_snapshot_mut()
        .set_value(&gix::config::tree::Pack::THREADS, "1")
        .with_context(|| "configure single-threaded gix pack indexing")?;
    ensure_git_operation_not_interrupted(interrupt, "after configuring pack workers")?;

    ensure_git_operation_not_interrupted(interrupt, "before preparing clone remote")?;
    let mut remote = repo
        .remote_at_without_url_rewrite(url)
        .with_context(|| "prepare clone remote")?
        .with_fetch_tags(gix::remote::fetch::Tags::None);
    ensure_git_operation_not_interrupted(interrupt, "after preparing clone remote")?;
    ensure_git_operation_not_interrupted(interrupt, "before setting exact fetch refspec")?;
    remote
        .replace_refspecs([refspec.as_str()], gix::remote::Direction::Fetch)
        .with_context(|| format!("set exact fetch refspec for {source_ref}"))?;
    ensure_git_operation_not_interrupted(interrupt, "after setting exact fetch refspec")?;

    // prepare_fetch owns this item later, but create it before connect:
    // connect performs the first network handshake and accepts no
    // progress/interrupt handle of its own.
    let discovery_progress =
        clone_progress.map(|progress| progress.item_named("discovering exact ref"));
    ensure_git_operation_not_interrupted(interrupt, "before connecting")?;
    let connection = remote
        .connect(gix::remote::Direction::Fetch)
        .with_context(|| format!("connect to {url}"))?;
    let mut connection = connection;
    connection.set_credentials(gix_policy::reject_credentials);
    ensure_git_operation_not_interrupted(interrupt, "after connecting")?;
    ensure_git_operation_not_interrupted(interrupt, "before mapping exact remote ref")?;
    let prepare = match discovery_progress {
        Some(progress) => connection
            .prepare_fetch(progress, gix::remote::ref_map::Options::default())
            .with_context(|| format!("map exact remote ref {source_ref}"))?,
        None => connection
            .prepare_fetch(
                gix::progress::Discard,
                gix::remote::ref_map::Options::default(),
            )
            .with_context(|| format!("map exact remote ref {source_ref}"))?,
    };
    #[cfg(test)]
    TEST_EXACT_PREPARE_COUNT.fetch_add(1, Ordering::AcqRel);
    ensure_git_operation_not_interrupted(interrupt, "after mapping exact remote ref")?;
    let prepare = prepare.with_shallow(gix::remote::fetch::Shallow::DepthAtRemote(
        NonZeroU32::new(1).expect("1 is nonzero"),
    ));
    ensure_git_operation_not_interrupted(interrupt, "after configuring shallow fetch")?;

    // `Prepare` already owns the one connect + ref-map handshake. Read
    // the exact mapping it will receive and elect/cache-probe before any
    // pack bytes move; the winner continues with this same value.
    let advertised_commit = prepare
        .ref_map()
        .mappings
        .iter()
        .find_map(|mapping| mapping.remote.peeled_id().map(ToOwned::to_owned))
        .with_context(|| format!("remote did not advertise exact ref {source_ref}"))?;
    ensure_git_operation_not_interrupted(interrupt, "before exact-ref cache election")?;
    let token = match gate(advertised_commit)? {
        ExactRefGate::Fetch(token) => token,
        ExactRefGate::Skip(token) => {
            return Ok(ExactRefFetch::Skipped {
                commit_id: advertised_commit,
                token,
            });
        }
    };
    ensure_git_operation_not_interrupted(interrupt, "after exact-ref cache election")?;

    ensure_git_operation_not_interrupted(interrupt, "before receiving the pack")?;
    match clone_progress {
        Some(progress) => prepare
            .receive(
                progress.item_named("receiving and indexing exact ref"),
                interrupt,
            )
            .with_context(|| format!("fetch exact remote ref {source_ref}"))?,
        None => prepare
            .receive(gix::progress::Discard, interrupt)
            .with_context(|| format!("fetch exact remote ref {source_ref}"))?,
    };
    ensure_git_operation_not_interrupted(interrupt, "after receiving the pack")?;
    ensure_git_operation_not_interrupted(interrupt, "before reading fetched ref")?;
    let mut fetched_ref = repo
        .find_reference(FETCHED_REF)
        .with_context(|| format!("find fetched ref {source_ref}"))?;
    ensure_git_operation_not_interrupted(interrupt, "after reading fetched ref")?;
    let unpeeled_id = fetched_ref.id().detach();
    // Preserve exactly the requested ref in its native namespace. In
    // particular, retaining the selected tag keeps setlocalversion and
    // git-describe semantics without importing any unrelated tag.
    ensure_git_operation_not_interrupted(interrupt, "before preserving exact source ref")?;
    repo.reference(
        source_ref.as_str(),
        unpeeled_id,
        gix::refs::transaction::PreviousValue::Any,
        "clone: preserve exact source ref",
    )
    .with_context(|| format!("preserve fetched ref {source_ref}"))?;
    ensure_git_operation_not_interrupted(interrupt, "after preserving exact source ref")?;
    ensure_git_operation_not_interrupted(interrupt, "before peeling exact source ref")?;
    let commit_id = fetched_ref
        .peel_to_commit()
        .with_context(|| format!("peel fetched ref {source_ref} to a commit"))?
        .id;
    ensure_git_operation_not_interrupted(interrupt, "after peeling exact source ref")?;
    if commit_id != advertised_commit {
        anyhow::bail!(
            "exact ref {source_ref} changed during its single fetch handshake \
             (advertised {advertised_commit}, received {commit_id})"
        );
    }

    // Both branch and tag clones are snapshots for ktstr, so a detached
    // HEAD is the uniform representation of the exact peeled commit.
    ensure_git_operation_not_interrupted(interrupt, "before publishing exact HEAD")?;
    repo.reference(
        "HEAD",
        commit_id,
        gix::refs::transaction::PreviousValue::Any,
        "clone: checkout exact ktstr source",
    )
    .with_context(|| "set cloned repository HEAD")?;
    ensure_git_operation_not_interrupted(interrupt, "after publishing exact HEAD")?;

    match clone_progress {
        Some(progress) => {
            checkout_exact_commit(
                &repo,
                commit_id,
                progress.item_named("checking out exact ref"),
                interrupt,
            )?;
        }
        None => {
            checkout_exact_commit(&repo, commit_id, gix::progress::Discard, interrupt)?;
        }
    }
    Ok(ExactRefFetch::Fetched { commit_id, token })
}

fn materialize_local_exact_ref_gated<T>(
    repository_path: &Path,
    source_ref: &str,
    clone_dir: &Path,
    clone_progress: Option<&crate::cli::progress::CloneProgress>,
    interrupt: &std::sync::atomic::AtomicBool,
    gate: impl FnOnce(gix::ObjectId) -> Result<ExactRefGate<T>>,
) -> Result<ExactRefFetch<T>> {
    const FETCHED_REF: &str = "refs/ktstr/source";

    ensure_git_operation_not_interrupted(interrupt, "before opening local exact source")?;
    let source = gix::open_opts(repository_path, anon_open_opts())
        .with_context(|| format!("open local source repository {}", repository_path.display()))?;
    ensure_git_operation_not_interrupted(interrupt, "after opening local exact source")?;
    let mut reference = source
        .find_reference(source_ref)
        .with_context(|| format!("find local source ref {source_ref}"))?;
    let unpeeled_id = reference.id().detach();
    let commit_id = reference
        .peel_to_commit()
        .with_context(|| format!("peel local source ref {source_ref} to a commit"))?
        .id;
    ensure_git_operation_not_interrupted(interrupt, "before exact-ref cache election")?;
    let token = match gate(commit_id)? {
        ExactRefGate::Fetch(token) => token,
        ExactRefGate::Skip(token) => {
            return Ok(ExactRefFetch::Skipped { commit_id, token });
        }
    };
    ensure_git_operation_not_interrupted(interrupt, "after exact-ref cache election")?;

    let _copy_progress =
        clone_progress.map(|progress| progress.item_named("copying local exact ref"));
    ensure_git_operation_not_interrupted(interrupt, "before preparing local clone")?;
    let repo = gix::ThreadSafeRepository::init_opts(
        clone_dir,
        gix::create::Kind::WithWorktree,
        gix::create::Options::default(),
        anon_open_opts(),
    )
    .with_context(|| "prepare local clone")?
    .to_thread_local();
    ensure_git_operation_not_interrupted(interrupt, "before copying local exact objects")?;
    copy_local_exact_ref_objects(&source, &repo, unpeeled_id, commit_id, interrupt)?;
    ensure_git_operation_not_interrupted(interrupt, "after copying local exact objects")?;
    drop(_copy_progress);

    repo.reference(
        FETCHED_REF,
        unpeeled_id,
        gix::refs::transaction::PreviousValue::Any,
        "clone: stage exact local source ref",
    )
    .with_context(|| "stage exact local source ref")?;
    repo.reference(
        source_ref,
        unpeeled_id,
        gix::refs::transaction::PreviousValue::Any,
        "clone: preserve exact local source ref",
    )
    .with_context(|| format!("preserve local source ref {source_ref}"))?;
    repo.reference(
        "HEAD",
        commit_id,
        gix::refs::transaction::PreviousValue::Any,
        "clone: checkout exact local source",
    )
    .with_context(|| "set local clone HEAD")?;
    std::fs::write(repo.git_dir().join("shallow"), format!("{commit_id}\n"))
        .with_context(|| "write local clone shallow boundary")?;
    ensure_git_operation_not_interrupted(interrupt, "after publishing local exact refs")?;

    match clone_progress {
        Some(progress) => checkout_exact_commit(
            &repo,
            commit_id,
            progress.item_named("checking out local exact ref"),
            interrupt,
        )?,
        None => checkout_exact_commit(&repo, commit_id, gix::progress::Discard, interrupt)?,
    }
    Ok(ExactRefFetch::Fetched { commit_id, token })
}

fn copy_local_exact_ref_objects(
    source: &gix::Repository,
    destination: &gix::Repository,
    unpeeled_id: gix::ObjectId,
    commit_id: gix::ObjectId,
    interrupt: &std::sync::atomic::AtomicBool,
) -> Result<()> {
    let mut copied = std::collections::HashSet::new();
    let mut current = unpeeled_id;
    loop {
        ensure_git_operation_not_interrupted(interrupt, "while copying local exact ref")?;
        if !copied.insert(current) {
            anyhow::bail!("local exact ref contains a tag cycle at {current}");
        }
        let object = source
            .find_object(current)
            .with_context(|| format!("read local exact-ref object {current}"))?;
        write_local_object(destination, &object)?;
        match object.kind {
            gix::objs::Kind::Tag => {
                current = object
                    .try_into_tag()
                    .with_context(|| format!("decode local tag object {current}"))?
                    .target_id()
                    .with_context(|| format!("read target of local tag object {current}"))?
                    .detach();
            }
            gix::objs::Kind::Commit => {
                if current != commit_id {
                    anyhow::bail!(
                        "local exact ref peeled inconsistently: expected {commit_id}, found {current}"
                    );
                }
                let tree_id = object
                    .try_into_commit()
                    .with_context(|| format!("decode local commit {commit_id}"))?
                    .tree_id()
                    .with_context(|| format!("read tree of local commit {commit_id}"))?
                    .detach();
                copy_local_tree_objects(source, destination, tree_id, &mut copied, interrupt)?;
                return Ok(());
            }
            kind => anyhow::bail!("local exact ref resolves to {kind}, expected a tag or commit"),
        }
    }
}

fn copy_local_tree_objects(
    source: &gix::Repository,
    destination: &gix::Repository,
    root_tree: gix::ObjectId,
    copied: &mut std::collections::HashSet<gix::ObjectId>,
    interrupt: &std::sync::atomic::AtomicBool,
) -> Result<()> {
    let mut pending = vec![root_tree];
    while let Some(object_id) = pending.pop() {
        #[cfg(test)]
        if TEST_INTERRUPT_DURING_LOCAL_COPY.swap(false, Ordering::AcqRel) {
            interrupt.store(true, Ordering::Release);
        }
        ensure_git_operation_not_interrupted(interrupt, "while copying local checkout tree")?;
        if !copied.insert(object_id) {
            continue;
        }
        let object = source
            .find_object(object_id)
            .with_context(|| format!("read local checkout object {object_id}"))?;
        write_local_object(destination, &object)?;
        match object.kind {
            gix::objs::Kind::Tree => {
                let tree = object
                    .try_into_tree()
                    .with_context(|| format!("decode local tree {object_id}"))?;
                for entry in tree.iter() {
                    let entry =
                        entry.with_context(|| format!("decode entry in local tree {object_id}"))?;
                    if entry.kind() != gix::objs::tree::EntryKind::Commit {
                        pending.push(entry.object_id());
                    }
                }
            }
            gix::objs::Kind::Blob => {}
            kind => anyhow::bail!(
                "local tree {root_tree} references unexpected {kind} object {object_id}"
            ),
        }
    }
    Ok(())
}

fn write_local_object(destination: &gix::Repository, object: &gix::Object<'_>) -> Result<()> {
    let written = gix::objs::Write::write_buf(destination, object.kind, &object.data)
        .map_err(anyhow::Error::from_boxed)
        .with_context(|| format!("write local source object {}", object.id))?;
    if written != object.id {
        anyhow::bail!(
            "local source object {} changed identity while materializing (wrote {written})",
            object.id
        );
    }
    Ok(())
}

fn ensure_git_operation_not_interrupted(
    interrupt: &std::sync::atomic::AtomicBool,
    phase: &str,
) -> Result<()> {
    if interrupt.load(Ordering::Acquire) {
        anyhow::bail!("git operation interrupted {phase}");
    }
    Ok(())
}

#[cfg(test)]
static TEST_INTERRUPT_AFTER_CHECKOUT: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

#[cfg(test)]
static TEST_INTERRUPT_AFTER_INDEX_WRITE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

#[cfg(test)]
static TEST_INTERRUPT_AFTER_REF_DISCOVERY: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

#[cfg(test)]
static TEST_INTERRUPT_DURING_LOCAL_COPY: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

#[cfg(test)]
static TEST_EXACT_PREPARE_COUNT: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

#[cfg(test)]
static TEST_INTERRUPT_AFTER_CLONE_PROMOTE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Materialize `commit_id` into the main worktree and persist its index.
fn checkout_exact_commit(
    repo: &gix::Repository,
    commit_id: gix::ObjectId,
    mut progress: impl gix::Progress,
    interrupt: &std::sync::atomic::AtomicBool,
) -> Result<()> {
    ensure_git_operation_not_interrupted(interrupt, "before locating exact clone worktree")?;
    let workdir = repo
        .workdir()
        .with_context(|| "exact clone repository has no worktree")?;
    ensure_git_operation_not_interrupted(interrupt, "after locating exact clone worktree")?;
    ensure_git_operation_not_interrupted(interrupt, "before resolving fetched commit tree")?;
    let tree_id = repo
        .find_object(commit_id)
        .with_context(|| format!("find fetched commit {commit_id}"))?
        .peel_to_tree()
        .with_context(|| format!("peel fetched commit {commit_id} to its tree"))?
        .id;
    ensure_git_operation_not_interrupted(interrupt, "after resolving fetched commit tree")?;
    ensure_git_operation_not_interrupted(interrupt, "before building exact clone index")?;
    let mut index = repo
        .index_from_tree(&tree_id)
        .with_context(|| format!("build index from fetched tree {tree_id}"))?;
    ensure_git_operation_not_interrupted(interrupt, "after building exact clone index")?;
    ensure_git_operation_not_interrupted(interrupt, "before building checkout options")?;
    let mut opts = repo
        .checkout_options(gix::worktree::stack::state::attributes::Source::IdMapping)
        .with_context(|| "build exact clone checkout options")?;
    ensure_git_operation_not_interrupted(interrupt, "after building checkout options")?;
    opts.destination_is_initially_empty = true;
    opts.keep_going = false;
    let checkout_workers = GixCheckoutWorkerLease::acquire();
    opts.thread_limit = Some(checkout_workers.workers());

    progress.init(Some(index.entries().len()), gix::progress::count("files"));
    ensure_git_operation_not_interrupted(interrupt, "before building checkout object handle")?;
    let objects = repo
        .objects
        .clone()
        .into_arc()
        .with_context(|| "build thread-safe object handle for exact clone checkout")?;
    ensure_git_operation_not_interrupted(interrupt, "after building checkout object handle")?;
    ensure_git_operation_not_interrupted(interrupt, "before checkout")?;
    gix::worktree::state::checkout(
        &mut index,
        workdir,
        objects,
        &progress,
        &gix::progress::Discard,
        interrupt,
        opts,
    )
    .with_context(|| format!("check out fetched commit {commit_id}"))?;
    #[cfg(test)]
    if TEST_INTERRUPT_AFTER_CHECKOUT.swap(false, Ordering::AcqRel) {
        interrupt.store(true, Ordering::Release);
    }
    // gix checkout returns Ok with a partial worktree when its
    // interrupt flag becomes set. Never persist that partial index or
    // report success; the caller removes the entire clone directory.
    ensure_git_operation_not_interrupted(interrupt, "during checkout")?;
    drop(checkout_workers);

    ensure_git_operation_not_interrupted(interrupt, "before writing exact clone index")?;
    index
        .write(Default::default())
        .with_context(|| "write exact clone index")?;
    #[cfg(test)]
    if TEST_INTERRUPT_AFTER_INDEX_WRITE.swap(false, Ordering::AcqRel) {
        interrupt.store(true, Ordering::Release);
    }
    // Publication is not complete until the persisted index has been
    // followed by one final observation of the shared signal epoch.
    // Otherwise a signal landing inside index.write() could return an
    // apparently successful clone whose caller proceeds to cache it.
    ensure_git_operation_not_interrupted(interrupt, "after writing exact clone index")?;
    Ok(())
}

/// Use a local kernel source tree.
///
/// Dirty detection uses gix `tree_index_status` (HEAD-vs-index) and
/// `status().into_index_worktree_iter()` (index-vs-worktree) to check
/// for modifications to tracked files. Submodule checks are skipped
/// entirely. Untracked files do not affect the dirty flag.
///
/// When the tree is dirty, the HEAD commit does not describe the
/// source actually being built, so `git_hash` is dropped — no
/// commit identifies a dirty worktree. `is_dirty=true` carries that
/// fact forward; callers (see [`crate::cli`]) use it to bypass the
/// kernel cache entirely.
///
/// No diagnostic output: all operator-visible messaging for a
/// local source is routed through `kernel_build_pipeline`'s
/// cache-skip hint (`DIRTY_TREE_CACHE_SKIP_HINT` /
/// `NON_GIT_TREE_CACHE_SKIP_HINT`), which has the full context
/// to emit a single informational line rather than two redundant
/// warnings. Sibling entries (`download_tarball`, `git_clone`)
/// still take a `cli_label` because they genuinely print
/// progress lines — `local_source` does not.
pub fn local_source(source_path: &Path) -> Result<AcquiredSource> {
    let (arch, _) = arch_info();

    if !source_path.is_dir() {
        anyhow::bail!("{}: not a directory", source_path.display());
    }

    let canonical = source_path
        .canonicalize()
        .with_context(|| format!("canonicalize {}", source_path.display()))?;

    let LocalSourceState {
        short_hash,
        is_dirty,
        is_git,
    } = inspect_local_source_state(&canonical)?;

    // User .config is folded into the cache key so two builds of the
    // same HEAD with different `.config` files do NOT collide on the
    // same key — see [`config_hash_for_key`] for the encoding.
    // Read at `local_source` time (rather than at the post-build
    // store site) so cache LOOKUP and cache STORE see the same key.
    let user_config_hash = config_hash_for_key(&canonical);

    let cache_key =
        compose_local_cache_key(arch, &short_hash, &canonical, user_config_hash.as_deref());

    // Record the kernel version from the source-tree Makefile so the
    // tmpfs-fraction gate (TmpfsFraction::for_kernel_version, via the
    // cache metadata.json sidecar) recognizes a locally-built honoring
    // kernel — symmetric with the tarball path. None when the Makefile
    // is unreadable/unparsable, which keeps the conservative 50% default.
    let version = read_makefile_version(&canonical);

    Ok(AcquiredSource {
        source_dir: canonical.clone(),
        cache_key,
        version,
        kernel_source: crate::cache::KernelSource::Local {
            source_tree_path: Some(canonical),
            git_hash: short_hash,
        },
        is_temp: false,
        is_dirty,
        is_git,
    })
}

/// Parse the kernel `MAJOR.MINOR.PATCH` version from a source tree's
/// top-level `Makefile` (`VERSION` / `PATCHLEVEL` / `SUBLEVEL`) — the
/// authoritative version of a locally-built kernel, mirroring the
/// version a tarball acquisition records. Returns `None` if the
/// `Makefile` is unreadable or any of the three fields is absent or
/// non-numeric, so the caller records no version and the rootfs tmpfs
/// fraction conservatively defaults to 50% (the honoring gate
/// `TmpfsFraction::for_kernel_version` keys on a positively-known
/// version). `EXTRAVERSION` (e.g. `-rc7`) is intentionally ignored: the
/// gate keys on `MAJOR.MINOR.PATCH` only.
fn read_makefile_version(source_dir: &Path) -> Option<String> {
    let text = std::fs::read_to_string(source_dir.join("Makefile")).ok()?;
    // Each field is a top-of-file `NAME = N` assignment; take the first
    // matching line and require a bare integer (a trailing comment or
    // non-numeric value yields None for that field, hence overall None).
    let field = |name: &str| -> Option<u16> {
        text.lines().find_map(|line| {
            line.trim()
                .strip_prefix(name)?
                .trim_start()
                .strip_prefix('=')?
                .trim()
                .parse::<u16>()
                .ok()
        })
    };
    Some(format!(
        "{}.{}.{}",
        field("VERSION")?,
        field("PATCHLEVEL")?,
        field("SUBLEVEL")?
    ))
}

/// Result of [`inspect_local_source_state`] — git hash and dirty/git
/// classification of a canonical source-tree path. Pulled out of
/// [`local_source`] so the post-build dirty re-check (a second call
/// from [`crate::cli::kernel_build_pipeline`]) reuses the exact same
/// gix path.
#[derive(Debug, Clone)]
pub struct LocalSourceState {
    /// HEAD short hash (7 chars). `None` when the tree is dirty
    /// (HEAD doesn't describe the actual source) or non-git (no
    /// HEAD at all). Mirrors the `git_hash` field on
    /// [`AcquiredSource::kernel_source`] for [`crate::cache::KernelSource::Local`].
    pub short_hash: Option<String>,
    /// Tracked-file dirt: HEAD-vs-index disagreement OR
    /// index-vs-worktree disagreement. Always `true` for non-git
    /// trees (dirty detection is impossible without git, so the
    /// pessimistic stance is dirty).
    pub is_dirty: bool,
    /// `true` when `gix::discover` succeeded (the tree is a git
    /// repo); `false` otherwise. Lets the cache-skip hint branch
    /// on whether `commit` / `stash` is actionable.
    pub is_git: bool,
}

/// Inspect a canonical source-tree path for git hash + dirty state.
///
/// Submodule checks are skipped (false positives on kernel trees
/// with uninitialized submodules). The non-git arm returns
/// `(None, true, false)` so the caller's cache-skip hint can
/// distinguish "dirty git repo" from "not a git repo at all".
///
/// Called twice per build by [`crate::cli::kernel_build_pipeline`]:
/// once at acquire time (via [`local_source`]) and again after
/// `make` returns to detect mid-build worktree edits, branch flips,
/// or commits that would otherwise let a racing-write build land in
/// the cache under a stale identity. Both calls share the same gix
/// path so the post-build comparison is apples-to-apples.
///
/// Non-atomic against concurrent git operations: the probe runs
/// six sequential gix calls (`discover` → `head_id` → `head_tree`
/// → `index_or_empty` → `tree_index_status` → `status`), each a
/// separate filesystem read with no transactional bracket. A
/// concurrent `git commit`, `git add`, or worktree write between
/// any two calls can produce internally-inconsistent results —
/// e.g. `head_id` reads commit C0, a peer commit lands C1, then
/// `head_tree` reads C1's root tree and the diff against the
/// post-add index reports unexpected dirt. Git itself serializes
/// its own writes via per-resource lockfiles under `.git/`
/// (`index.lock` for staging operations, `HEAD.lock` and
/// `refs/heads/<branch>.lock` for ref updates), so peer `git`
/// processes wait on whichever lockfile their operation touches;
/// the genuinely-unsynchronized class is worktree-only writes
/// (autoformatter, IDE-on-save) which the index-worktree status
/// step catches regardless of timing.
///
/// The disposition is intentionally pessimistic so inconsistency is
/// safe: any `Err` propagates to the caller, which treats it as a
/// rebuild signal (`MidWaitState::ProbeFailed` in the mid-wait
/// caller); any spurious dirty signal falls into DirtyEdit /
/// HashAdvanced, both forcing a rebuild. The cost of a false-
/// positive rebuild is one extra `make`; the cost of a false-
/// negative would be a cache slot keyed on a HEAD that no longer
/// describes the source — the asymmetry is the reason for the
/// pessimistic disposition. Callers should treat the returned
/// state as a best-effort approximation of probe-time, not an
/// instantaneous snapshot.
pub fn inspect_local_source_state(canonical: &Path) -> Result<LocalSourceState> {
    let (short_hash, is_dirty, is_git) = match gix::discover(canonical) {
        Ok(repo) => {
            let head = repo.head_id().with_context(|| "read HEAD")?;
            let short_hash = format!("{}", head).chars().take(7).collect::<String>();

            // tree_index_status compares a TREE id against the index;
            // the HEAD commit id is not itself a tree, so peel HEAD
            // to its root tree before diffing or the diff silently
            // returns an error and index dirt goes undetected.
            let head_tree = repo.head_tree().with_context(|| "read HEAD tree")?;
            let head_tree_id = head_tree.id;

            // Check HEAD-vs-index for tracked file changes.
            let mut index_dirty = false;
            let index = repo.index_or_empty().with_context(|| "open index")?;
            let _ = repo.tree_index_status(
                &head_tree_id,
                &index,
                None,
                gix::status::tree_index::TrackRenames::Disabled,
                |_, _, _| {
                    index_dirty = true;
                    Ok::<_, std::convert::Infallible>(std::ops::ControlFlow::Break(()))
                },
            );

            // Check index-vs-worktree for modified tracked files,
            // skipping submodules entirely (Ignore::All).
            let worktree_dirty = if !index_dirty {
                repo.status(gix::progress::Discard)
                    .with_context(|| "status")?
                    .index_worktree_rewrites(None)
                    .index_worktree_submodules(gix::status::Submodule::Given {
                        ignore: gix::submodule::config::Ignore::All,
                        check_dirty: false,
                    })
                    .index_worktree_options_mut(|opts| {
                        opts.dirwalk_options = None;
                        crate::git_status::configure_index_worktree_parallelism(opts);
                    })
                    .into_index_worktree_iter(Vec::new())
                    .map(crate::git_status::consume_has_any)
                    .unwrap_or(false)
            } else {
                false
            };

            let is_dirty = index_dirty || worktree_dirty;
            // Drop the HEAD hash when dirty — the commit does not
            // describe the actual source being built, so publishing
            // it via git_hash / cache_key would misidentify the
            // build input.
            let hash = if is_dirty { None } else { Some(short_hash) };
            (hash, is_dirty, true)
        }
        Err(_) => {
            // The downstream kernel_build_pipeline (cli::kernel_build_pipeline)
            // emits `NON_GIT_TREE_CACHE_SKIP_HINT` — a single
            // informational line that names both the cause and the
            // remediation paths — once the is_dirty=true branch
            // decides to skip the cache. Emitting a second
            // "not a git repository" warning here duplicated that
            // content for every non-git `--kernel <path>` run. The
            // `(None, true, false)` tuple silently communicates
            // the non-git state to the cache-skip decision site;
            // no separate stderr line is needed on this path.
            (None, true, false)
        }
    };
    Ok(LocalSourceState {
        short_hash,
        is_dirty,
        is_git,
    })
}

/// Compose the cache key for a local source given its arch, optional
/// HEAD short hash, canonical source path, and optional user
/// `.config` hash.
///
/// Three shapes:
/// - `local-{hash7}-{arch}-kc{suffix}` — clean git tree, no user
///   `.config` (plain `make defconfig` path or no config file yet)
/// - `local-{hash7}-{arch}-cfg{user_config}-kc{suffix}` — clean git
///   tree with a user `.config` whose hash differs from `defconfig`
/// - `local-unknown-{path_hash}-{arch}-kc{suffix}` — dirty / non-git
///   tree (HEAD does not describe the source; the path-derived
///   crc32 salt keeps two distinct dirty trees from colliding on the
///   same `local-unknown-...` slot)
///
/// `path_hash` is the full 8-char (32-bit) lowercase-hex CRC32 of
/// the canonical source-path bytes. CRC32 keeps the per-path
/// disambiguator stable across runs without pulling in a
/// crypto-grade hash for what is fundamentally a slot disambiguator.
///
/// `user_config_hash` is `None` whenever the source tree has no
/// `.config` file yet (the build will run `make defconfig` and
/// produce one). This collapses the user-config branch back into the
/// hash-only key so a fresh checkout's first build still hits a
/// later cache lookup keyed without the cfg segment.
pub fn compose_local_cache_key(
    arch: &str,
    short_hash: &Option<String>,
    canonical: &Path,
    user_config_hash: Option<&str>,
) -> String {
    let suffix = crate::cache_key_suffix();
    match short_hash {
        Some(hash) => match user_config_hash {
            Some(cfg) => format!("local-{hash}-{arch}-cfg{cfg}-kc{suffix}"),
            None => format!("local-{hash}-{arch}-kc{suffix}"),
        },
        None => {
            let path_hash = canonical_path_hash(canonical);
            format!("local-unknown-{path_hash}-{arch}-kc{suffix}")
        }
    }
}

/// CRC32 of the canonical source-path bytes, lowercase hex
/// (full 8-char width — the entire 32-bit value). Disambiguates
/// `local-unknown-...` cache keys and per-source-tree lockfile
/// names across distinct dirty / non-git source trees so two
/// parallel `cargo ktstr test --kernel ./linux-a` and
/// `--kernel ./linux-b` runs can't write each other's vmlinux into
/// the same cache slot or share a single source-tree flock.
///
/// Full 32 bits (8 hex chars) of CRC32 keep collision risk
/// negligible against the practical population (handful of source
/// trees per host) while staying human-readable. The earlier
/// 6-char (24-bit) form left ~6× the collision surface for the
/// same key shape; truncation served no purpose other than visual
/// brevity. Path bytes are taken via `OsStr::as_encoded_bytes` so
/// a non-UTF-8 component (rare on Linux but possible) doesn't lose
/// entropy through a UTF-8 lossy conversion.
pub(crate) fn canonical_path_hash(canonical: &Path) -> String {
    let bytes = canonical.as_os_str().as_encoded_bytes();
    format!("{:08x}", crc32fast::hash(bytes))
}

/// Read `<canonical>/.config` and return its CRC32 as a lowercase
/// hex string suitable for embedding in the cache key. Returns
/// `None` when no `.config` exists (a fresh tree before the build
/// runs `make defconfig`).
///
/// Distinct from the `config_hash` written into [`crate::cache::KernelMetadata`]
/// at store time — that records the FINAL `.config` after
/// configuration runs, for diagnostic display in `kernel list`.
/// This helper records the PRE-BUILD `.config` so the cache key
/// reflects what the operator's tree currently has on disk; the
/// same `.config` content always maps to the same key, even if the
/// downstream `make olddefconfig` step elaborates additional
/// defaults.
fn config_hash_for_key(canonical: &Path) -> Option<String> {
    let config_path = canonical.join(".config");
    let data = std::fs::read(&config_path).ok()?;
    Some(format!("{:08x}", crc32fast::hash(&data)))
}

#[cfg(test)]
#[path = "fetch_tests.rs"]
mod tests;
