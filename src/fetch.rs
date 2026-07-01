//! Kernel source acquisition: tarball download, git clone, local tree.
//!
//! Three entry points — [`download_tarball`], [`git_clone`], and
//! [`local_source`] — each return an [`AcquiredSource`] carrying the
//! source directory, cache key, and metadata the caller needs to
//! proceed to configuration and build.

use std::io::Read;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow};
use reqwest::blocking::Client;
use sha2::{Digest, Sha256};

/// Process-wide [`reqwest::blocking::Client`] lazily initialized on
/// first access via [`shared_client`]. Keeping a single `Client`
/// instance across the fetch-family reuses its TCP connection pool
/// and TLS session cache across repeated calls to the same host
/// within a CLI run. Cross-host fetches in the same run still
/// re-handshake because reqwest's connection pool keys on host.
static SHARED_CLIENT: OnceLock<Client> = OnceLock::new();

/// Connect-phase timeout for [`shared_client`]: bounds the time spent
/// in the TCP + TLS handshake before reqwest gives up on a peer.
/// Bounds the dead-route case — a CDN edge that accepts the SYN but
/// stalls the handshake, or a route that blackholes outright —
/// without putting any ceiling on the response body's streaming
/// duration once the connection is up.
///
/// No total request `.timeout()` is set: the same client serves both
/// short requests (directory listings, releases.json) and large
/// tarball streams ([`download_stable_tarball`],
/// [`download_rc_tarball`]), where a 130–180 MiB compressed payload
/// over a slow uplink can take minutes of wall-clock to deliver.
/// Capping that with a per-request timeout would abort legitimate
/// downloads; bounding only the connect phase preserves the
/// dead-route guarantee while letting
/// the body stream as long as the upstream is making forward
/// progress.
const SHARED_CLIENT_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Return the process-wide shared [`reqwest::blocking::Client`]. First
/// call constructs it via `Client::builder()` with
/// `SHARED_CLIENT_CONNECT_TIMEOUT` applied; every subsequent call
/// returns a reference to the same instance. This helper is for
/// top-level CLI entries that want the default client.
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
/// construct a client. The documented failure modes are TLS backend
/// initialization (e.g. rustls/native-tls subsystem unreachable) and
/// are treated as setup bugs rather than runtime errors. The
/// `expect` here, rather than propagating the error, mirrors the
/// inherited behavior of `reqwest::blocking::Client::new()` (which
/// is itself an infallible wrapper around `builder().build().expect`).
pub fn shared_client() -> &'static Client {
    SHARED_CLIENT.get_or_init(|| {
        Client::builder()
            .connect_timeout(SHARED_CLIENT_CONNECT_TIMEOUT)
            .build()
            .expect("build shared reqwest client")
    })
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
///    leave the download blocked indefinitely (reqwest's per-read
///    timeout reset on every empty wakeup, and the connect-phase
///    timeout already passed during handshake). The check fires
///    BEFORE the inner `read()` so a stalled inner reader cannot
///    out-block the watchdog.
///
/// 2. **Streaming SHA-256.** Updates a [`Sha256`] hasher with every
///    byte that flows past, so the caller can verify the finalized
///    digest against an expected value (parsed out of
///    `sha256sums.asc`) without a second pass over the data. The
///    hasher only sees bytes that were actually consumed by the
///    decoder + tar extractor, which is the same set of bytes that
///    landed on disk — so a partial download that errored midway
///    produces a hash over only what we successfully streamed,
///    preventing false-positive verifications on truncated input.
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
    hasher: Sha256,
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
            hasher: Sha256::new(),
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
        (hex::encode(self.hasher.finalize()), self.bytes_total)
    }
}

impl<R: Read> Read for DownloadStream<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
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
                self.hasher.update(&buf[..n]);
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

/// Total request timeout for [`fetch_sha256sums_from_url`]: bounds
/// the wall-clock window for the single small-body GET that
/// retrieves the cleartext-signed checksum manifest. The body is
/// the `sha256sums.asc` cleartext block — typically a few KiB of
/// `<hash>  <filename>` lines plus a PGP signature trailer — so a
/// tight 30 s ceiling fits the realistic case (sub-second on a
/// healthy CDN edge) while still bounding the failure mode this
/// guards against: a stalled CDN that accepts the connection but
/// never delivers bytes. Without a per-request timeout the
/// shared client only carries [`SHARED_CLIENT_CONNECT_TIMEOUT`]
/// (handshake-only), so a stalled body read would hang the build
/// indefinitely. The caller treats any error from this function
/// as "no expected hash available" and downgrades verification
/// to a warning, so a 30 s timeout that fires on a hung CDN
/// surfaces as an unverified-but-progressing download rather
/// than a wedged build.
const SHA256SUMS_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

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
    let response = client
        .get(url)
        .timeout(SHA256SUMS_REQUEST_TIMEOUT)
        .send()
        .with_context(|| format!("fetch {url}"))?;
    if !response.status().is_success() {
        anyhow::bail!("fetch {url}: HTTP {}", response.status());
    }
    response
        .text()
        .with_context(|| format!("read body of {url}"))
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
/// require strict verification should pin the source via `--source`
/// or `--git` rather than the download path. The bypass warnings
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

/// Download a stable kernel tarball (.tar.xz) from cdn.kernel.org.
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
    let tarball_name = format!("linux-{version}.tar.xz");
    let url = format!("https://cdn.kernel.org/pub/linux/kernel/v{major}.x/{tarball_name}");

    let expected_sha256 = resolve_expected_sha256(client, major, &tarball_name, skip_sha256);

    tracing::info!(%url, "downloading stable kernel tarball (requires network)");
    let response = client
        .get(&url)
        .timeout(DOWNLOAD_REQUEST_READ_TIMEOUT)
        .send()
        .with_context(|| format!("download {url}"))?;
    if !response.status().is_success() {
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            anyhow::bail!("{}", version_not_found_msg(client, version));
        }
        anyhow::bail!("download {url}: HTTP {}", response.status());
    }
    reject_html_response(&response, &url)?;
    print_download_size(&response, &url, cli_label, mp);
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
    // final `fs::rename` into place is atomic and a verification
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
    archive
        .unpack(staging.path())
        .with_context(|| "extract tarball")?;

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
        verify_sha256(&actual_hex, expected, &url)?;
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

    let source_dir = promote_staged_kernel_tree(&staging, dest_dir, version)?;
    Ok(source_dir)
}

/// Verify a kernel tarball's staged extraction contains exactly one
/// top-level entry named `linux-{version}/` and atomically rename it
/// into `dest_dir/linux-{version}`. Bails — leaving `dest_dir`
/// untouched — when the staging dir holds a stray entry, when the
/// expected inner directory is missing, or when the rename fails.
/// The caller's `TempDir` outlives this helper, so its Drop sweeps
/// any residual staging contents whether this returns Ok or Err.
fn promote_staged_kernel_tree(
    staging: &tempfile::TempDir,
    dest_dir: &Path,
    version: &str,
) -> Result<PathBuf> {
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
    std::fs::rename(&inner, &source_dir)
        .with_context(|| format!("rename {} -> {}", inner.display(), source_dir.display()))?;
    Ok(source_dir)
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

    let response = client
        .get(&url)
        .timeout(DOWNLOAD_REQUEST_READ_TIMEOUT)
        .send()
        .with_context(|| format!("download {url}"))?;
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
    archive
        .unpack(staging.path())
        .with_context(|| "extract tarball")?;

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

    let source_dir = promote_staged_kernel_tree(&staging, dest_dir, version)?;
    Ok(source_dir)
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
/// so `skip_sha256` is a no-op on the RC arm. `--source` and
/// `--git` callers do not reach this function at all.
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
        download_stable_tarball(client, version, dest_dir, cli_label, skip_sha256, mp)?
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
    let response = client
        .get(url)
        .send()
        .with_context(|| format!("fetch {url}"))?;
    if !response.status().is_success() {
        anyhow::bail!("fetch {url}: HTTP {}", response.status());
    }
    let body = response.text().with_context(|| "read response body")?;
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
/// Scans all monikers in releases.json except linux-next. If no
/// match is found (EOL series), fetches the cdn.kernel.org directory
/// listing to find the highest patch version with a tarball.
///
/// When `client` is the process-wide [`shared_client`] singleton,
/// routes through `RELEASES_CACHE`; other clients bypass the
/// cache via pointer-equality and exercise `fetch_releases`
/// directly — see `cached_releases_with` for details. Cache
/// scope is releases.json only; the EOL-series directory-listing
/// fallback in `probe_latest_patch` always hits the network.
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

    eprintln!("{cli_label}: {prefix}.x not in releases.json (EOL series), probing cdn.kernel.org");
    probe_latest_patch(client, prefix, cli_label)
}

/// Find the latest patch version for an EOL series by fetching the
/// CDN directory listing.
///
/// GETs the `v{major}.x/` directory index from cdn.kernel.org and
/// extracts `linux-{prefix}.{patch}.tar.xz` filenames to find the
/// highest patch. One GET replaces the former parallel-HEAD probe
/// which failed in CI environments that block or mishandle HEAD
/// requests to the CDN.
fn probe_latest_patch(client: &Client, prefix: &str, cli_label: &str) -> Result<String> {
    let major = major_version(prefix)?;
    let url = format!("https://cdn.kernel.org/pub/linux/kernel/v{major}.x/");
    eprintln!("{cli_label}: fetching directory listing from {url}");
    let body = client
        .get(&url)
        .send()
        .with_context(|| format!("GET {url}"))?
        .error_for_status()
        .with_context(|| format!("GET {url}"))?
        .text()
        .with_context(|| format!("reading body from {url}"))?;

    let needle = format!("linux-{prefix}.");
    let mut best_patch: Option<u32> = None;
    for line in body.lines() {
        let Some(pos) = line.find(&needle) else {
            continue;
        };
        let after = &line[pos + needle.len()..];
        let Some(dot) = after.find(".tar.xz") else {
            continue;
        };
        let patch_str = &after[..dot];
        if let Ok(patch) = patch_str.parse::<u32>()
            && best_patch.is_none_or(|b| patch > b)
        {
            best_patch = Some(patch);
        }
    }

    match best_patch {
        Some(patch) => {
            let version = format!("{prefix}.{patch}");
            eprintln!("{cli_label}: latest {prefix}.x kernel (from cdn listing): {version}");
            Ok(version)
        }
        None => {
            anyhow::bail!(
                "no tarball matching {prefix}.x found in cdn.kernel.org \
                 directory listing at {url}"
            );
        }
    }
}

/// Cache key for a git-cloned kernel: the raw user ref verbatim, the
/// resolved commit's FULL hash, the target arch, and the
/// kconfig-fragment suffix. The SINGLE construction site, shared by
/// [`git_clone`] (post-clone, from `head_id`) and the pre-clone
/// ls-remote cache probe in `resolve_git_kernel` — a drift between the
/// two would make the probe miss the entry the clone wrote and defeat
/// the clone-skip.
///
/// The FULL 40-hex commit hash keys the entry (not a 7-hex prefix): a
/// branch/tag tip moves over time, so the `{git_ref}` segment alone
/// cannot distinguish successive commits — only the hash does. A 7-hex
/// (28-bit) prefix would let a moved tip whose new commit shares the
/// first 7 hex with the cached old commit hit the stale entry and serve
/// the wrong kernel build under the new ref. The full id removes that
/// collision class; the probe and clone both render full lowercase hex
/// before any truncation, so keying on it is drift-free.
pub fn git_cache_key(git_ref: &str, commit_hash: &str) -> String {
    let (arch, _) = arch_info();
    format!(
        "{git_ref}-git-{commit_hash}-{arch}-kc{}",
        crate::cache_key_suffix()
    )
}

/// Resolve `git_ref` to its tip COMMIT's full hash among the `refs` a
/// remote advertised, for the pre-clone cache probe. Precedence: an
/// explicit `refs/` path, else `refs/heads/{ref}`, else
/// `refs/tags/{ref}`, else `HEAD`. For an annotated tag (`Ref::Peeled`)
/// the peeled commit (`object`) is used, never the tag object;
/// `Ref::Unborn` carries no commit and is skipped. `None` when no ref
/// matches. Pure over `refs` so the precedence + tag-peeling are
/// unit-testable without a network handshake.
///
/// This is a best-effort guess at the commit a clone would land on; it
/// does NOT perfectly mirror the shallow clone in every case. This probe
/// applies heads-before-tags precedence, whereas the shallow clone
/// ([`git_clone`], `DepthAtRemote(1)`) resolves the bare ref via gix's
/// DWIM across heads AND tags and errors `RefNameAmbiguous` when a remote
/// advertises both a branch and a tag of that name — where this probe
/// instead returns the branch commit. A mismatch only costs a redundant
/// clone (the post-clone re-check in `resolve_git_kernel` is
/// authoritative); it never serves a wrong build, since [`git_cache_key`]
/// embeds the full commit hash returned here.
fn match_ref_commit_hash(refs: &[gix::protocol::handshake::Ref], git_ref: &str) -> Option<String> {
    let pick = |target: &str| -> Option<gix::hash::ObjectId> {
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
    };
    let object = git_ref
        .starts_with("refs/")
        .then(|| pick(git_ref))
        .flatten()
        .or_else(|| pick(&format!("refs/heads/{git_ref}")))
        .or_else(|| pick(&format!("refs/tags/{git_ref}")))
        .or_else(|| (git_ref == "HEAD").then(|| pick("HEAD")).flatten())?;
    Some(format!("{object}"))
}

/// True when `git_ref` is a full 40-char hex commit id — recognizable
/// as a sha without a remote handshake. A 39/41-char ref, or any
/// 40-char ref carrying a non-hex byte, is a name (branch/tag) and
/// falls through to ls-remote. Case is not normalized here (the caller
/// lowercases the full hash to match `git_clone`'s rendering).
fn is_full_sha(git_ref: &str) -> bool {
    git_ref.len() == 40 && git_ref.bytes().all(|b| b.is_ascii_hexdigit())
}

/// Resolve `git_ref` to its tip commit's FULL hash WITHOUT cloning, via
/// an ls-remote handshake, so [`crate::cli::resolve_git_kernel`] can
/// probe the cache before the expensive full clone. Returns `None`
/// (best-effort) on ANY failure — network, auth, or an unadvertised ref
/// — so the caller falls through to the clone, which resolves the ref
/// authoritatively.
///
/// A full 40-hex `git_ref` is taken directly (no handshake): it matches
/// [`git_clone`]'s post-clone `head_id` byte-for-byte (gix `Id` /
/// `ObjectId` `Display` both emit the full lowercase hex), so it builds
/// the key a sha-clone would write — see [`git_cache_key`]. [`git_clone`]
/// currently REJECTS a raw-sha `git_ref` (a bare commit cannot be
/// fetched), so for a sha this probe always misses and the clone then
/// surfaces the actionable error; the branch is kept because it avoids a
/// handshake on that reject path and is ready for a future sha-capable
/// clone. It trusts the hex verbatim (no remote validation) — harmless
/// while sha-clone is rejected, since an unbuildable sha has no cache
/// entry. The ad-hoc bare repo (`gix::init` on a tempdir) carries no
/// working tree and fetches no pack; `ref_map` lists every advertised ref
/// (remote-side ref-prefix filtering is disabled — see the body) without
/// fetching a pack. See `match_ref_commit_hash` for the rare branch/tag
/// same-name caveat where this guess and the clone diverge.
pub fn ls_remote_commit_hash(url: &str, git_ref: &str) -> Option<String> {
    if is_full_sha(git_ref) {
        return Some(git_ref.to_ascii_lowercase());
    }
    let tmp = tempfile::TempDir::new().ok()?;
    let repo = gix::init(tmp.path()).ok()?;
    let remote = repo.remote_at(url).ok()?;
    let conn = remote.connect(gix::remote::Direction::Fetch).ok()?;
    let (refmap, _handshake) = conn
        .ref_map(
            gix::progress::Discard,
            gix::remote::ref_map::Options {
                // ls-remote semantics: list EVERY advertised ref. gix's
                // default (`prefix_from_spec_as_filter_on_remote = true`)
                // derives protocol-v2 `ls-refs` `ref-prefix` filters from the
                // remote's fetch refspecs; an anonymous `remote_at` has none,
                // and `fetch_tags = Included` injects only `refs/tags/*`, so
                // the server returns TAGS ONLY and `refs/heads/*` never
                // arrive — a branch `git_ref` then resolves to `None` and the
                // clone-skip never fires (every run re-clones). Disabling the
                // remote-side filter returns all refs, so a branch, tag, or
                // HEAD `git_ref` all resolve.
                prefix_from_spec_as_filter_on_remote: false,
                ..Default::default()
            },
        )
        .ok()?;
    match_ref_commit_hash(&refmap.remote_refs, git_ref)
}

/// Clone a git repository with shallow depth.
///
/// `cli_label` prefixes diagnostic status output (e.g. `"ktstr"` or
/// `"cargo ktstr"`).
///
/// `mp` is the progress group a determinate clone bar is added to;
/// `None` disables the bar and passes `gix::progress::Discard` to gix
/// exactly as before (the single-shot `kernel build` paths and unit
/// tests pass `None`). The bar shows real object/file counts + ETA
/// during the receiving / resolving / checkout phases that gix reports
/// a bounded total for; see the `crate::cli::progress` module.
pub fn git_clone(
    url: &str,
    git_ref: &str,
    dest_dir: &Path,
    cli_label: &str,
    mp: Option<&crate::cli::FetchProgress>,
) -> Result<AcquiredSource> {
    // A raw 40-hex commit SHA cannot be cloned here: gix's
    // `prepare_clone(...).with_ref_name(<object-id>)` panics at
    // `fetch_then_checkout` (gix `clone/access.rs`), and fetching a bare
    // commit needs server-side allow-sha-in-want support this path does
    // not implement. Reject with an actionable error (use a branch or
    // tag) rather than panic. Placed at the single clone entry so every
    // caller (auto-discovery + `kernel build`) is covered.
    if is_full_sha(git_ref) {
        anyhow::bail!(
            "git+{url}#{git_ref}: cannot fetch a kernel by raw commit SHA — \
             use a branch or tag name instead (the resolved commit is recorded \
             in the cache key after a branch/tag clone)"
        );
    }
    let cloning = format!("{cli_label}: cloning {url} (ref: {git_ref}, depth: 1)");
    match mp {
        Some(fp) => fp.println(&cloning),
        None => eprintln!("{cloning}"),
    }

    let clone_dir = dest_dir.join("linux");

    let mut prep = gix::prepare_clone(url, &clone_dir)
        .with_context(|| "prepare clone")?
        .with_shallow(gix::remote::fetch::Shallow::DepthAtRemote(
            NonZeroU32::new(1).expect("1 is nonzero"),
        ))
        .with_ref_name(Some(git_ref))
        .with_context(|| "set ref name")?;

    // Drive a determinate clone bar from gix's progress tree (see
    // [`crate::cli::progress::CloneProgress`]). `None` when no progress
    // group is threaded in; the gix calls then pass `Discard` exactly
    // as before. One interrupt flag (never set) is shared by both
    // phases, matching the prior per-call `AtomicBool::new(false)`.
    let clone_progress = mp.map(|fp| fp.clone_progress(git_ref));
    let interrupt = std::sync::atomic::AtomicBool::new(false);

    let (mut checkout, _outcome) = match &clone_progress {
        Some(cp) => prep
            .fetch_then_checkout(cp.item(), &interrupt)
            .with_context(|| "clone fetch")?,
        None => prep
            .fetch_then_checkout(gix::progress::Discard, &interrupt)
            .with_context(|| "clone fetch")?,
    };

    let (_repo, _outcome) = match &clone_progress {
        Some(cp) => checkout
            .main_worktree(cp.item(), &interrupt)
            .with_context(|| "checkout")?,
        None => checkout
            .main_worktree(gix::progress::Discard, &interrupt)
            .with_context(|| "checkout")?,
    };

    // Clone + checkout done — stop the poll thread, join it, clear the
    // bar. On any error path above, `clone_progress` is dropped
    // instead, and its `Drop` performs the same shutdown (leak-proof).
    if let Some(cp) = clone_progress {
        cp.finish();
    }

    let repo = gix::open(&clone_dir).with_context(|| "open cloned repo")?;
    let head = repo.head_id().with_context(|| "read HEAD")?;
    // FULL commit hash keys the cache (see `git_cache_key` — a 7-hex
    // prefix risks a moved-tip collision serving a stale build); the
    // 7-hex `short_hash` is kept only for the human-facing source record.
    let commit_hash = format!("{head}");
    let short_hash = commit_hash.chars().take(7).collect::<String>();

    let cache_key = git_cache_key(git_ref, &commit_hash);

    // Record the kernel version from the checked-out source Makefile, as
    // local_source does — the worktree is fully checked out here, so a
    // git-clone-acquired honoring kernel also earns the 90% tmpfs reclaim
    // via the metadata.json sidecar. None on an unreadable/unparsable
    // Makefile, which keeps the conservative 50% default.
    let version = read_makefile_version(&clone_dir);

    Ok(AcquiredSource {
        source_dir: clone_dir,
        cache_key,
        version,
        kernel_source: crate::cache::KernelSource::git(short_hash, git_ref),
        is_temp: true,
        is_dirty: false,
        is_git: true,
    })
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
                    })
                    .into_index_worktree_iter(Vec::new())
                    .map(|mut iter| iter.next().is_some())
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
            // content for every non-git `--source` run. The
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
