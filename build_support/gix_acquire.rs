//! Shared, build-script-safe source acquisition.
//!
//! Both `build.rs` and `scx-ktstr/build.rs` include this module directly so
//! their network path cannot drift: exact depth-one gix fetches, an immutable
//! per-repository source CAS, jobserver-bounded breadth-first recursion, and
//! one heartbeat reporter. This module never invokes a `git` or `gix`
//! executable or constructs a helper-capable transport.

use std::collections::{BTreeMap, HashMap};
use std::fs::OpenOptions;
use std::hash::{BuildHasher, Hasher};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, mpsc};
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::gix_acquire_ahash as ahash;
use crate::gix_acquire_fs2::FileExt;
use crate::gix_acquire_gix as gix;
use crate::gix_acquire_jobserver as jobserver;
use gix::bstr::ByteSlice;

#[path = "gix_policy.rs"]
mod gix_policy;

const CACHE_SCHEMA: &str = "ktstr-build-content-v1";
const CACHE_SENTINEL: &str = ".ktstr-content-key";
const SOURCE_NODE_SCHEMA: &str = "ktstr-source-node-v1";
const SOURCE_NODE_REPOSITORY: &str = "repository.git";
const FETCHED_REF: &str = "refs/ktstr/source";
const HEARTBEAT: Duration = Duration::from_secs(10);
const POLL: Duration = Duration::from_millis(250);
const MAX_FETCH_ATTEMPTS: u32 = 4;
const MAX_PARALLEL_SOURCE_NODES: usize = 8;
const MAX_SUBMODULE_DEPTH: usize = 32;

/// Return ktstr's persistent cache root, if the host exposes an absolute
/// XDG/HOME cache location.
pub(crate) fn cache_root(namespace: &str) -> Option<PathBuf> {
    let root = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .filter(|path| path.is_absolute())
        .or_else(|| {
            std::env::var_os("HOME")
                .map(PathBuf::from)
                .filter(|path| path.is_absolute())
                .map(|home| home.join(".cache"))
        })?;
    Some(root.join("ktstr").join("content-v1").join(namespace))
}

/// Fixed-seed, non-cryptographic content identifier.
///
/// Length-prefixing each component makes the input tuple unambiguous; the
/// fixed seeds keep the path stable across processes and toolchain updates.
pub(crate) fn content_id(parts: &[&str]) -> String {
    let state = ahash::RandomState::with_seeds(
        0x4b54_5354_522d_4341,
        0x532d_4749_582d_5631,
        0xa076_1d64_78bd_642f,
        0xe703_7ed1_a0b4_28db,
    );
    let mut hasher = state.build_hasher();
    hasher.write_u64(CACHE_SCHEMA.len() as u64);
    hasher.write(CACHE_SCHEMA.as_bytes());
    for part in parts {
        hasher.write_u64(part.len() as u64);
        hasher.write(part.as_bytes());
    }
    format!("{:016x}", hasher.finish())
}

fn content_manifest(parts: &[&str]) -> String {
    let mut manifest = String::from(CACHE_SCHEMA);
    for part in parts {
        manifest.push('\n');
        manifest.push_str(&part.len().to_string());
        manifest.push(':');
        manifest.push_str(part);
    }
    manifest
}

/// The final directory used by [`ensure_cached`].
pub(crate) fn cache_entry(root: &Path, parts: &[&str]) -> PathBuf {
    root.join(content_id(parts))
}

/// Test-facing adapter around the fully cancellable cache implementation.
///
/// Production callers use the cancellable entry points below. Keeping this
/// adapter test-only lets the cross-process integration fixture exercise the
/// same election and publication path without carrying an otherwise unused
/// build-script symbol.
#[cfg(test)]
pub(crate) fn ensure_cached<Complete, Build>(
    root: &Path,
    parts: &[&str],
    label: &str,
    complete: Complete,
    build: Build,
) -> Result<PathBuf, String>
where
    Complete: Fn(&Path) -> bool,
    Build: FnOnce(&Path, &ProgressReporter) -> Result<(), String>,
{
    let cancelled = AtomicBool::new(false);
    let progress = ProgressReporter::new(label);
    let result = ensure_cached_with(
        root,
        parts,
        label,
        &complete,
        &progress,
        &cancelled,
        |stage, progress, _cancelled| build(stage, progress),
    );
    if result.is_ok() {
        progress.finish();
    }
    result
}

fn ensure_cached_with<Complete, Build>(
    root: &Path,
    parts: &[&str],
    label: &str,
    complete: &Complete,
    progress: &ProgressReporter,
    cancelled: &AtomicBool,
    build: Build,
) -> Result<PathBuf, String>
where
    Complete: Fn(&Path) -> bool,
    Build: FnOnce(&Path, &ProgressReporter, &AtomicBool) -> Result<(), String>,
{
    let manifest = content_manifest(parts);
    let entry = cache_entry(root, parts);
    let is_complete = |path: &Path| {
        complete(path)
            && std::fs::read_to_string(path.join(CACHE_SENTINEL))
                .is_ok_and(|value| value == manifest)
    };
    if is_complete(&entry) {
        return Ok(entry);
    }

    std::fs::create_dir_all(root)
        .map_err(|err| format!("create content cache {}: {err}", root.display()))?;
    let id = content_id(parts);
    let lock_path = root.join(format!(".{id}.lock"));
    let lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&lock_path)
        .map_err(|err| format!("open content-cache lock {}: {err}", lock_path.display()))?;

    progress.set_phase("waiting for the shared content builder");
    loop {
        match FileExt::try_lock_exclusive(&lock) {
            Ok(()) => break,
            Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                ensure_not_cancelled(cancelled, "while waiting for a shared source builder")?;
                std::thread::sleep(POLL);
            }
            Err(err) => {
                return Err(format!(
                    "acquire content-cache lock {}: {err}",
                    lock_path.display()
                ));
            }
        }
    }

    if is_complete(&entry) {
        progress.set_phase("reusing the shared published result");
        return Ok(entry);
    }

    if entry.exists() {
        std::fs::remove_dir_all(&entry)
            .map_err(|err| format!("remove incomplete cache entry {}: {err}", entry.display()))?;
    }

    remove_stale_stages(root, &id)?;
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let stage = root.join(format!(".{id}.work-{}-{nonce}", std::process::id()));
    if stage.exists() {
        std::fs::remove_dir_all(&stage)
            .map_err(|err| format!("remove stale cache stage {}: {err}", stage.display()))?;
    }
    std::fs::create_dir_all(&stage)
        .map_err(|err| format!("create cache stage {}: {err}", stage.display()))?;
    let guard = RemoveDirOnDrop(stage.clone());

    progress.set_phase("building the shared content entry");
    ensure_not_cancelled(cancelled, "before building a shared content entry")?;
    build(&stage, progress, cancelled)?;
    ensure_not_cancelled(cancelled, "after building a shared content entry")?;
    std::fs::write(stage.join(CACHE_SENTINEL), &manifest)
        .map_err(|err| format!("write cache completion sentinel: {err}"))?;
    if !is_complete(&stage) {
        return Err(format!(
            "{label}: builder returned without producing a complete cache entry"
        ));
    }
    std::fs::rename(&stage, &entry).map_err(|err| {
        format!(
            "atomically publish content cache {} -> {}: {err}",
            stage.display(),
            entry.display()
        )
    })?;
    guard.disarm();
    FileExt::unlock(&lock)
        .map_err(|err| format!("unlock content-cache lock {}: {err}", lock_path.display()))?;
    Ok(entry)
}

fn remove_stale_stages(root: &Path, id: &str) -> Result<(), String> {
    let prefix = format!(".{id}.work-");
    for candidate in std::fs::read_dir(root)
        .map_err(|err| format!("scan content cache {}: {err}", root.display()))?
    {
        let candidate = candidate.map_err(|err| format!("read content-cache entry: {err}"))?;
        if !candidate.file_name().to_string_lossy().starts_with(&prefix) {
            continue;
        }
        let path = candidate.path();
        if candidate
            .file_type()
            .map_err(|err| format!("stat stale cache stage {}: {err}", path.display()))?
            .is_dir()
        {
            std::fs::remove_dir_all(&path)
                .map_err(|err| format!("remove stale cache stage {}: {err}", path.display()))?;
        } else {
            std::fs::remove_file(&path)
                .map_err(|err| format!("remove stale cache stage {}: {err}", path.display()))?;
        }
    }
    Ok(())
}

/// One immutable repository revision in the shared source graph.
///
/// A node deliberately excludes compiler and target state. The same fetched
/// object database is therefore reused by every wprof/scx binary build whose
/// exact source tuple matches.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct SourceNode {
    canonical_url: String,
    selector: String,
    commit: String,
}

impl SourceNode {
    pub(crate) fn new(url: &str, selector: &str, commit: &str) -> Result<Self, String> {
        if selector.is_empty() {
            return Err("exact source selector is empty".to_string());
        }
        let selector = if selector.starts_with("refs/") {
            selector.to_string()
        } else {
            normalize_full_commit(selector)?
        };
        Ok(Self {
            canonical_url: canonical_https_url(url)?,
            selector,
            commit: normalize_full_commit(commit)?,
        })
    }

    fn key_parts(&self) -> [&str; 4] {
        [
            SOURCE_NODE_SCHEMA,
            self.canonical_url.as_str(),
            self.selector.as_str(),
            self.commit.as_str(),
        ]
    }

    pub(crate) fn id(&self) -> String {
        content_id(&self.key_parts())
    }
}

#[derive(Clone, Debug)]
struct SourceChild {
    path: PathBuf,
    node: SourceNode,
}

#[derive(Clone, Debug)]
struct AcquiredSourceNode {
    source: SourceNode,
    cache_entry: PathBuf,
    children: Vec<SourceChild>,
}

#[derive(Clone, Debug)]
struct SourceOccurrence {
    source: SourceNode,
    destination: PathBuf,
    depth: usize,
}

#[derive(Clone, Debug)]
struct ResolvedSourceOccurrence {
    source: SourceNode,
    cache_entry: PathBuf,
    destination: PathBuf,
}

/// Canonicalize the public HTTPS identity used by the source-node CAS.
///
/// GitHub treats the optional `.git` suffix as the same repository. Keeping a
/// single canonical spelling prevents wprof/scx and their relative submodule
/// URLs from downloading duplicate object databases.
pub(crate) fn canonical_https_url(url: &str) -> Result<String, String> {
    if !matches!(
        gix_policy::classify_source(url)?,
        gix_policy::InProcessSource::Https
    ) {
        return Err(format!(
            "build-time exact source acquisition requires https://; refusing {url}"
        ));
    }
    let mut parsed = gix::Url::from_bytes(url.as_bytes().as_bstr())
        .map_err(|err| format!("parse source URL {url}: {err}"))?;
    if parsed.user.is_some() || parsed.password.is_some() {
        return Err(format!(
            "build-time source URL must not contain credentials: {url}"
        ));
    }
    let host = parsed
        .host
        .as_deref()
        .ok_or_else(|| format!("HTTPS source URL has no host: {url}"))?
        .to_ascii_lowercase();
    parsed.host = Some(host.clone());
    if parsed.port == Some(443) {
        parsed.port = None;
    }
    parsed.serialize_alternative_form = false;

    let mut path = parsed
        .path
        .to_str()
        .map_err(|err| format!("source URL path is not UTF-8: {err}"))?
        .trim_end_matches('/')
        .to_string();
    if path.is_empty() || path == "/" {
        return Err(format!("HTTPS source URL has no repository path: {url}"));
    }
    if host == "github.com" {
        if let Some(without_suffix) = path.strip_suffix(".git") {
            path = without_suffix.to_string();
        }
        path.push_str(".git");
    }
    parsed.path = path.into();
    parsed
        .to_bstring()
        .to_str()
        .map(str::to_owned)
        .map_err(|err| format!("canonical source URL is not UTF-8: {err}"))
}

fn normalize_full_commit(commit: &str) -> Result<String, String> {
    if commit.len() != 40 {
        return Err(format!(
            "exact source commit must be a full 40-digit SHA-1, got {commit}"
        ));
    }
    gix::ObjectId::from_hex(commit.as_bytes())
        .map_err(|err| format!("parse exact source commit {commit}: {err}"))
        .map(|id| id.to_string())
}

/// Fetch and assemble one exact repository without following submodules.
///
/// The cache contains a bare, immutable object database. The destination is a
/// private mutable worktree produced from those objects and never contains
/// repository metadata or hardlinks back into the cache.
#[allow(dead_code)] // scx uses this; wprof uses the recursive peer.
pub(crate) fn assemble_exact_cached(
    source_cache_root: &Path,
    url: &str,
    selector: &str,
    commit: &str,
    destination: &Path,
    progress: &ProgressReporter,
) -> Result<(), String> {
    let source = SourceNode::new(url, selector, commit)?;
    let cancelled = AtomicBool::new(false);
    let acquired = acquire_source_node(source, source_cache_root, progress, &cancelled)?;
    reset_private_destination(destination)?;
    materialize_source_node(&acquired, destination, &cancelled)
}

/// Fetch an exact recursive repository graph into immutable per-node entries,
/// then assemble one private mutable worktree.
///
/// Graph expansion is breadth-first. Identical nodes are fetched once even if
/// they appear at several paths, while every committed placement is still
/// materialized. Cargo's jobserver grants every worker beyond the build
/// script's implicit caller slot; a source fetch itself always uses one gix
/// pack/check-out worker.
#[allow(dead_code)] // wprof uses this; scx uses the non-recursive peer.
pub(crate) fn assemble_exact_recursive_cached(
    source_cache_root: &Path,
    url: &str,
    selector: &str,
    commit: &str,
    destination: &Path,
    progress: &ProgressReporter,
) -> Result<(), String> {
    let source = SourceNode::new(url, selector, commit)?;
    let cancelled = AtomicBool::new(false);
    // The root is necessarily a singleton. Acquire it with the build script's
    // implicit jobserver slot, discover its first level, and only then borrow
    // extra tokens that can immediately back real sibling fetches.
    let root_source = acquire_source_node(source.clone(), source_cache_root, progress, &cancelled)?;
    let (width, permits) = cargo_source_parallelism()?;
    let resolved = walk_source_graph(
        source,
        vec![root_source],
        source_cache_root,
        width,
        progress,
        &cancelled,
        |source, cache_root, progress, cancelled| {
            acquire_source_node(source, cache_root, progress, cancelled)
        },
    )?;
    drop(permits);

    reset_private_destination(destination)?;
    for occurrence in resolved {
        if !occurrence.destination.as_os_str().is_empty() {
            remove_path_if_present(&destination.join(&occurrence.destination))?;
        }
        materialize_source_node(
            &AcquiredSourceNode {
                source: occurrence.source,
                cache_entry: occurrence.cache_entry,
                children: Vec::new(),
            },
            &destination.join(occurrence.destination),
            &cancelled,
        )?;
    }
    Ok(())
}

fn acquire_source_node(
    source: SourceNode,
    source_cache_root: &Path,
    progress: &ProgressReporter,
    cancelled: &AtomicBool,
) -> Result<AcquiredSourceNode, String> {
    let parts = source.key_parts();
    let label = format!("source {}@{}", source.canonical_url, source.selector);
    let complete = |entry: &Path| source_node_complete(entry, &source.commit);
    let entry = ensure_cached_with(
        source_cache_root,
        &parts,
        &label,
        &complete,
        progress,
        cancelled,
        |stage, progress, cancelled| {
            fetch_source_node_with_retry(
                &source,
                &stage.join(SOURCE_NODE_REPOSITORY),
                progress,
                cancelled,
            )
        },
    )?;
    let children = source_node_children(&source, &entry)?;
    Ok(AcquiredSourceNode {
        source,
        cache_entry: entry,
        children,
    })
}

fn source_node_complete(entry: &Path, expected_commit: &str) -> bool {
    let Ok(repo) = gix::open_opts(entry.join(SOURCE_NODE_REPOSITORY), open_options()) else {
        return false;
    };
    repo.is_bare() && verify_head_commit(&repo, expected_commit).is_ok()
}

fn source_node_children(
    source: &SourceNode,
    cache_entry: &Path,
) -> Result<Vec<SourceChild>, String> {
    let repo = gix::open_opts(cache_entry.join(SOURCE_NODE_REPOSITORY), open_options())
        .map_err(|err| format!("open cached source node {}: {err}", source.id()))?;
    verify_head_commit(&repo, &source.commit)?;
    submodule_checkouts(&repo, &source.canonical_url)?
        .into_iter()
        .map(|child| {
            Ok(SourceChild {
                path: child.path,
                node: SourceNode::new(&child.url, &child.commit, &child.commit)?,
            })
        })
        .collect()
}

fn fetch_source_node_with_retry(
    source: &SourceNode,
    destination: &Path,
    progress: &ProgressReporter,
    cancelled: &AtomicBool,
) -> Result<(), String> {
    let mut last_error = None;
    for attempt in 1..=MAX_FETCH_ATTEMPTS {
        ensure_not_cancelled(cancelled, "before an exact source fetch")?;
        remove_path_if_present(destination)?;
        progress.set_phase(&format!(
            "fetching {}@{} (attempt {attempt}/{MAX_FETCH_ATTEMPTS})",
            source.canonical_url, source.selector
        ));
        match fetch_source_node_once(source, destination, progress, cancelled) {
            Ok(()) => return Ok(()),
            Err(err) => {
                last_error = Some(err);
                if attempt < MAX_FETCH_ATTEMPTS {
                    let backoff = Duration::from_secs(1u64 << attempt);
                    progress.set_phase(&format!(
                        "source fetch failed; retrying in {}s",
                        backoff.as_secs()
                    ));
                    wait_or_cancel(backoff, cancelled)?;
                }
            }
        }
    }
    Err(last_error.expect("at least one source-node fetch attempt ran"))
}

fn fetch_source_node_once(
    source: &SourceNode,
    destination: &Path,
    progress: &ProgressReporter,
    cancelled: &AtomicBool,
) -> Result<(), String> {
    let mut repo = gix::ThreadSafeRepository::init_opts(
        destination,
        gix::create::Kind::Bare,
        gix::create::Options::default(),
        open_options(),
    )
    .map_err(|err| format!("initialize exact bare source node: {err}"))?
    .to_thread_local();
    repo.config_snapshot_mut()
        .set_value(&gix::config::tree::Pack::THREADS, "1")
        .map_err(|err| format!("limit source-node pack workers: {err}"))?;

    let refspec = format!("+{}:{FETCHED_REF}", source.selector);
    let mut remote = repo
        .remote_at_without_url_rewrite(source.canonical_url.as_str())
        .map_err(|err| format!("prepare exact remote {}: {err}", source.canonical_url))?
        .with_fetch_tags(gix::remote::fetch::Tags::None);
    remote
        .replace_refspecs([refspec.as_str()], gix::remote::Direction::Fetch)
        .map_err(|err| format!("configure exact refspec {}: {err}", source.selector))?;

    ensure_not_cancelled(cancelled, "before connecting an exact source fetch")?;
    let negotiate = progress.item(&format!(
        "negotiating {}@{}",
        source.canonical_url, source.selector
    ));
    let mut connection = remote
        .connect(gix::remote::Direction::Fetch)
        .map_err(|err| format!("connect exact remote {}: {err}", source.canonical_url))?;
    connection.set_credentials(gix_policy::reject_credentials);
    let prepare = connection
        .prepare_fetch(negotiate, gix::remote::ref_map::Options::default())
        .map_err(|err| format!("map exact selector {}: {err}", source.selector))?
        .with_shallow(gix::remote::fetch::Shallow::DepthAtRemote(
            1.try_into().expect("one is non-zero"),
        ));
    ensure_not_cancelled(cancelled, "before receiving an exact source pack")?;
    let receive = progress.item(&format!(
        "receiving {}@{}",
        source.canonical_url, source.selector
    ));
    prepare
        .receive(receive, cancelled)
        .map_err(|err| format!("fetch exact selector {}: {err}", source.selector))?;
    ensure_not_cancelled(cancelled, "after receiving an exact source pack")?;

    let mut fetched_ref = repo
        .find_reference(FETCHED_REF)
        .map_err(|err| format!("find fetched selector {}: {err}", source.selector))?;
    let commit = fetched_ref
        .peel_to_commit()
        .map_err(|err| format!("peel fetched selector {}: {err}", source.selector))?
        .id;
    verify_commit_id(commit, &source.commit)?;
    repo.reference(
        "HEAD",
        commit,
        gix::refs::transaction::PreviousValue::Any,
        "ktstr source cache: exact detached commit",
    )
    .map_err(|err| format!("set exact source-node HEAD: {err}"))?;
    Ok(())
}

fn walk_source_graph<Acquire>(
    root: SourceNode,
    initial: Vec<AcquiredSourceNode>,
    source_cache_root: &Path,
    width: usize,
    progress: &ProgressReporter,
    cancelled: &AtomicBool,
    acquire: Acquire,
) -> Result<Vec<ResolvedSourceOccurrence>, String>
where
    Acquire: Fn(SourceNode, &Path, &ProgressReporter, &AtomicBool) -> Result<AcquiredSourceNode, String>
        + Sync,
{
    let mut acquired: HashMap<String, AcquiredSourceNode> = initial
        .into_iter()
        .map(|source| (source.source.id(), source))
        .collect();
    let mut resolved = Vec::new();
    let mut level = vec![SourceOccurrence {
        source: root,
        destination: PathBuf::new(),
        depth: 0,
    }];

    while !level.is_empty() {
        ensure_not_cancelled(cancelled, "while expanding the source graph")?;
        let mut missing = BTreeMap::<String, SourceNode>::new();
        for occurrence in &level {
            let id = occurrence.source.id();
            if !acquired.contains_key(&id) {
                missing
                    .entry(id)
                    .or_insert_with(|| occurrence.source.clone());
            }
        }
        let fetched = run_bounded(
            missing.into_values().collect(),
            width,
            cancelled,
            |source, cancelled| acquire(source, source_cache_root, progress, cancelled),
        )?;
        for source in fetched {
            acquired.insert(source.source.id(), source);
        }

        let mut next = Vec::new();
        for occurrence in level {
            let cached = acquired
                .get(&occurrence.source.id())
                .expect("every source node in a level was acquired");
            resolved.push(ResolvedSourceOccurrence {
                source: occurrence.source,
                cache_entry: cached.cache_entry.clone(),
                destination: occurrence.destination.clone(),
            });
            if !cached.children.is_empty() && occurrence.depth >= MAX_SUBMODULE_DEPTH {
                return Err(format!(
                    "submodule nesting exceeds {MAX_SUBMODULE_DEPTH} levels below {}",
                    cached.source.canonical_url
                ));
            }
            for child in &cached.children {
                next.push(SourceOccurrence {
                    source: child.node.clone(),
                    destination: occurrence.destination.join(&child.path),
                    depth: occurrence.depth + 1,
                });
            }
        }
        level = next;
    }
    Ok(resolved)
}

fn run_bounded<T, R, Work>(
    items: Vec<T>,
    width: usize,
    cancelled: &AtomicBool,
    work: Work,
) -> Result<Vec<R>, String>
where
    T: Send,
    R: Send,
    Work: Fn(T, &AtomicBool) -> Result<R, String> + Sync,
{
    let width = width.max(1);
    let mut pending = items.into_iter().enumerate();
    let mut completed = Vec::new();
    loop {
        ensure_not_cancelled(cancelled, "before starting a source-node batch")?;
        let batch: Vec<_> = pending.by_ref().take(width).collect();
        if batch.is_empty() {
            break;
        }
        let failure = Mutex::new(None::<String>);
        let mut outcomes = std::thread::scope(|scope| {
            let (send, receive) = mpsc::channel();
            let mut batch = batch.into_iter();
            let first = batch.next().expect("non-empty source-node batch");
            for (index, item) in batch {
                let send = send.clone();
                let failure = &failure;
                let work = &work;
                scope.spawn(move || {
                    let outcome = run_one_bounded(item, cancelled, failure, work);
                    let _ = send.send((index, outcome));
                });
            }
            let first = (
                first.0,
                run_one_bounded(first.1, cancelled, &failure, &work),
            );
            drop(send);
            let mut outcomes = vec![first];
            outcomes.extend(receive);
            outcomes
        });
        outcomes.sort_by_key(|(index, _)| *index);
        if let Some(error) = failure.into_inner().unwrap_or_else(|err| err.into_inner()) {
            return Err(error);
        }
        for (_, outcome) in outcomes {
            completed.push(outcome?);
        }
    }
    Ok(completed)
}

fn run_one_bounded<T, R, Work>(
    item: T,
    cancelled: &AtomicBool,
    failure: &Mutex<Option<String>>,
    work: &Work,
) -> Result<R, String>
where
    Work: Fn(T, &AtomicBool) -> Result<R, String>,
{
    ensure_not_cancelled(cancelled, "before starting a source node")?;
    match work(item, cancelled) {
        Ok(value) => Ok(value),
        Err(error) => {
            let mut first = failure.lock().unwrap_or_else(|err| err.into_inner());
            if first.is_none() {
                *first = Some(error.clone());
                cancelled.store(true, Ordering::Release);
            }
            Err(error)
        }
    }
}

fn cargo_source_parallelism() -> Result<(usize, Vec<jobserver::Acquired>), String> {
    // SAFETY: Cargo owns and exports the authenticated jobserver descriptors
    // inherited by this build script. We only duplicate them through the
    // jobserver crate and keep every acquired token alive until all workers
    // have joined.
    let Some(client) = (unsafe { jobserver::Client::from_env() }) else {
        return Ok((1, Vec::new()));
    };
    let mut permits = Vec::new();
    while permits.len() + 1 < MAX_PARALLEL_SOURCE_NODES {
        match client.try_acquire() {
            Ok(Some(permit)) => permits.push(permit),
            Ok(None) => break,
            Err(err) if err.kind() == std::io::ErrorKind::Unsupported => break,
            Err(err) => return Err(format!("query Cargo jobserver capacity: {err}")),
        }
    }
    Ok((1 + permits.len(), permits))
}

fn materialize_source_node(
    source: &AcquiredSourceNode,
    destination: &Path,
    cancelled: &AtomicBool,
) -> Result<(), String> {
    ensure_not_cancelled(cancelled, "before assembling a cached source node")?;
    std::fs::create_dir_all(destination).map_err(|err| {
        format!(
            "create private source tree {}: {err}",
            destination.display()
        )
    })?;
    let repo = gix::open_opts(
        source.cache_entry.join(SOURCE_NODE_REPOSITORY),
        open_options(),
    )
    .map_err(|err| format!("open cached source node {}: {err}", source.source.id()))?;
    materialize_repository_commit(&repo, &source.source.commit, destination, cancelled)
}

fn materialize_repository_commit(
    repo: &gix::Repository,
    expected_commit: &str,
    destination: &Path,
    cancelled: &AtomicBool,
) -> Result<(), String> {
    std::fs::create_dir_all(destination).map_err(|err| {
        format!(
            "create private source tree {}: {err}",
            destination.display()
        )
    })?;
    verify_head_commit(repo, expected_commit)?;
    let commit = repo
        .head_commit()
        .map_err(|err| format!("resolve cached source-node HEAD: {err}"))?;
    let tree_id = commit
        .tree_id()
        .map_err(|err| format!("resolve cached source-node tree: {err}"))?
        .detach();
    let mut index = repo
        .index_from_tree(&tree_id)
        .map_err(|err| format!("build private source-tree index: {err}"))?;
    let mut options = repo
        .checkout_options(gix::worktree::stack::state::attributes::Source::IdMapping)
        .map_err(|err| format!("configure private source-tree checkout: {err}"))?;
    options.destination_is_initially_empty = true;
    options.keep_going = false;
    options.thread_limit = Some(1);
    let objects = repo
        .objects
        .clone()
        .into_arc()
        .map_err(|err| format!("open cached source-node object database: {err}"))?;
    gix::worktree::state::checkout(
        &mut index,
        destination,
        objects,
        &gix::progress::Discard,
        &gix::progress::Discard,
        cancelled,
        options,
    )
    .map_err(|err| format!("assemble cached source commit {expected_commit}: {err}"))?;
    ensure_not_cancelled(cancelled, "after assembling a cached source node")
}

#[cfg(test)]
pub(crate) fn materialize_repository_commit_for_test(
    repo: &gix::Repository,
    commit: gix::ObjectId,
    destination: &Path,
) -> Result<(), String> {
    repo.reference(
        "HEAD",
        commit,
        gix::refs::transaction::PreviousValue::Any,
        "ktstr test: exact detached commit",
    )
    .map_err(|err| format!("set fixture HEAD: {err}"))?;
    materialize_repository_commit(
        repo,
        &commit.to_string(),
        destination,
        &AtomicBool::new(false),
    )
}

#[cfg(test)]
pub(crate) fn walk_source_graph_for_test<Children>(
    root: SourceNode,
    width: usize,
    cancelled: &AtomicBool,
    children: Children,
) -> Result<Vec<(String, PathBuf)>, String>
where
    Children: Fn(&SourceNode, &AtomicBool) -> Result<Vec<(PathBuf, SourceNode)>, String> + Sync,
{
    let progress = ProgressReporter::new("source graph fixture");
    walk_source_graph(
        root,
        Vec::new(),
        Path::new("fixture-cache"),
        width,
        &progress,
        cancelled,
        |source, _cache, _progress, cancelled| {
            let source_children = children(&source, cancelled)?
                .into_iter()
                .map(|(path, node)| SourceChild { path, node })
                .collect();
            Ok(AcquiredSourceNode {
                cache_entry: PathBuf::from(source.id()),
                source,
                children: source_children,
            })
        },
    )
    .map(|occurrences| {
        occurrences
            .into_iter()
            .map(|occurrence| (occurrence.source.id(), occurrence.destination))
            .collect()
    })
}

#[cfg(test)]
pub(crate) fn run_bounded_for_test<T, R, Work>(
    items: Vec<T>,
    width: usize,
    cancelled: &AtomicBool,
    work: Work,
) -> Result<Vec<R>, String>
where
    T: Send,
    R: Send,
    Work: Fn(T, &AtomicBool) -> Result<R, String> + Sync,
{
    run_bounded(items, width, cancelled, work)
}

fn reset_private_destination(destination: &Path) -> Result<(), String> {
    remove_path_if_present(destination)?;
    std::fs::create_dir_all(destination).map_err(|err| {
        format!(
            "create private source destination {}: {err}",
            destination.display()
        )
    })
}

fn remove_path_if_present(path: &Path) -> Result<(), String> {
    let metadata = match std::fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(err) => {
            return Err(format!(
                "inspect {} before replacement: {err}",
                path.display()
            ));
        }
    };
    if metadata.is_dir() {
        std::fs::remove_dir_all(path)
            .map_err(|err| format!("remove directory {}: {err}", path.display()))
    } else {
        std::fs::remove_file(path).map_err(|err| format!("remove file {}: {err}", path.display()))
    }
}

fn wait_or_cancel(duration: Duration, cancelled: &AtomicBool) -> Result<(), String> {
    let deadline = Instant::now() + duration;
    while Instant::now() < deadline {
        ensure_not_cancelled(cancelled, "during source-fetch retry backoff")?;
        std::thread::sleep(POLL.min(deadline.saturating_duration_since(Instant::now())));
    }
    Ok(())
}

fn ensure_not_cancelled(cancelled: &AtomicBool, phase: &str) -> Result<(), String> {
    if cancelled.load(Ordering::Acquire) {
        Err(format!("source acquisition cancelled {phase}"))
    } else {
        Ok(())
    }
}

/// Materialize one exact revision and all of its gitlink submodules.
///
/// Each repository is fetched independently at depth one. Recursive
/// submodules use the exact object id recorded by their parent's committed
/// tree, never a branch tip from `.gitmodules`.
#[allow(dead_code)] // scx uses the non-recursive peer; wprof uses this one.
pub(crate) fn checkout_exact_recursive(
    url: &str,
    revision: &str,
    destination: &Path,
    progress: &ProgressReporter,
) -> Result<(), String> {
    checkout_exact_inner(
        url,
        revision,
        None,
        destination,
        progress,
        0,
        &checkout_exact,
    )
}

/// Materialize an advertised root ref recursively and require its peeled
/// commit to equal `expected_commit`.
///
/// This keeps pinned roots on the normal advertised-ref protocol path while
/// making the immutable commit pin authoritative.
#[allow(dead_code)] // wprof uses this; scx uses the non-recursive peer.
pub(crate) fn checkout_exact_recursive_verified(
    url: &str,
    revision: &str,
    expected_commit: &str,
    destination: &Path,
    progress: &ProgressReporter,
) -> Result<(), String> {
    checkout_exact_inner(
        url,
        revision,
        Some(expected_commit),
        destination,
        progress,
        0,
        &checkout_exact,
    )
}

/// Materialize one exact revision without initializing submodules.
pub(crate) fn checkout_exact(
    url: &str,
    revision: &str,
    destination: &Path,
    progress: &ProgressReporter,
) -> Result<(), String> {
    clone_one_with_retry(url, revision, destination, progress)
}

/// Materialize an advertised ref without submodules and require its peeled
/// commit to equal `expected_commit`.
#[allow(dead_code)] // scx uses this; wprof uses the recursive peer.
pub(crate) fn checkout_exact_verified(
    url: &str,
    revision: &str,
    expected_commit: &str,
    destination: &Path,
    progress: &ProgressReporter,
) -> Result<(), String> {
    checkout_exact(url, revision, destination, progress)?;
    let repo = gix::open_opts(destination, open_options())
        .map_err(|err| format!("open checkout {}: {err}", destination.display()))?;
    verify_head_commit(&repo, expected_commit)
}

#[cfg(test)]
pub(crate) fn checkout_exact_recursive_with(
    url: &str,
    revision: &str,
    expected_commit: &str,
    destination: &Path,
    progress: &ProgressReporter,
    materialize: &MaterializeRevision<'_>,
) -> Result<(), String> {
    checkout_exact_inner(
        url,
        revision,
        Some(expected_commit),
        destination,
        progress,
        0,
        materialize,
    )
}

type MaterializeRevision<'a> =
    dyn Fn(&str, &str, &Path, &ProgressReporter) -> Result<(), String> + 'a;

fn checkout_exact_inner(
    url: &str,
    revision: &str,
    expected_commit: Option<&str>,
    destination: &Path,
    progress: &ProgressReporter,
    depth: usize,
    materialize: &MaterializeRevision<'_>,
) -> Result<(), String> {
    if depth > MAX_SUBMODULE_DEPTH {
        return Err(format!(
            "submodule nesting exceeds {MAX_SUBMODULE_DEPTH} levels at {url}@{revision}"
        ));
    }
    materialize(url, revision, destination, progress)?;

    let repo = gix::open_opts(destination, open_options())
        .map_err(|err| format!("open checkout {}: {err}", destination.display()))?;
    if let Some(expected_commit) = expected_commit {
        verify_head_commit(&repo, expected_commit)?;
    }
    let children = submodule_checkouts(&repo, url)?;
    drop(repo);

    for child in children {
        let child_destination = destination.join(&child.path);
        checkout_exact_inner(
            &child.url,
            &child.commit,
            Some(&child.commit),
            &child_destination,
            progress,
            depth + 1,
            materialize,
        )
        .map_err(|err| {
            format!(
                "initialize submodule {} at {}: {err}",
                child.path.display(),
                child.commit
            )
        })?;
    }
    Ok(())
}

#[derive(Debug, Eq, PartialEq)]
pub(crate) struct SubmoduleCheckout {
    pub(crate) path: PathBuf,
    pub(crate) url: String,
    pub(crate) commit: String,
}

pub(crate) fn submodule_checkouts(
    repo: &gix::Repository,
    parent_url: &str,
) -> Result<Vec<SubmoduleCheckout>, String> {
    let mut children = Vec::new();
    if let Some(submodules) = repo
        .submodules()
        .map_err(|err| format!("read submodules: {err}"))?
    {
        for submodule in submodules {
            let path = submodule
                .path()
                .map_err(|err| format!("read submodule path: {err}"))?;
            let path = gix::path::from_bstr(path).into_owned();
            validate_relative_submodule_path(&path)?;
            let child_url = submodule
                .url()
                .map_err(|err| format!("read submodule URL for {}: {err}", path.display()))?;
            let child_url = resolve_submodule_url(parent_url, &child_url)?;
            let commit = submodule
                .head_id()
                .map_err(|err| format!("read gitlink id for {}: {err}", path.display()))?
                .ok_or_else(|| format!("submodule {} has no committed gitlink", path.display()))?;
            children.push(SubmoduleCheckout {
                path,
                url: child_url,
                commit: commit.to_string(),
            });
        }
    }
    Ok(children)
}

fn verify_head_commit(repo: &gix::Repository, expected: &str) -> Result<(), String> {
    let actual = repo
        .head_commit()
        .map_err(|err| format!("resolve peeled checkout HEAD: {err}"))?
        .id;
    verify_commit_id(actual, expected)
}

fn verify_commit_id(actual: gix::ObjectId, expected: &str) -> Result<(), String> {
    let expected = gix::ObjectId::from_hex(expected.as_bytes())
        .map_err(|err| format!("parse expected peeled commit {expected}: {err}"))?;
    if actual != expected {
        return Err(format!(
            "advertised ref peeled to {actual}, expected pinned commit {expected}"
        ));
    }
    Ok(())
}

#[cfg(test)]
pub(crate) fn verify_reference_commit(
    repo: &gix::Repository,
    reference: &str,
    expected: &str,
) -> Result<(), String> {
    let actual = repo
        .find_reference(reference)
        .map_err(|err| format!("find fixture reference {reference}: {err}"))?
        .peel_to_commit()
        .map_err(|err| format!("peel fixture reference {reference}: {err}"))?
        .id;
    verify_commit_id(actual, expected)
}

fn clone_one_with_retry(
    url: &str,
    revision: &str,
    destination: &Path,
    progress: &ProgressReporter,
) -> Result<(), String> {
    require_build_https(url)?;
    let mut last_error = None;
    for attempt in 1..=MAX_FETCH_ATTEMPTS {
        if destination.exists() {
            std::fs::remove_dir_all(destination).map_err(|err| {
                format!(
                    "remove partial checkout {} before attempt {attempt}: {err}",
                    destination.display()
                )
            })?;
        }
        progress.set_phase(&format!(
            "fetching {url}@{revision} (attempt {attempt}/{MAX_FETCH_ATTEMPTS})"
        ));
        match clone_one(url, revision, destination, progress) {
            Ok(()) => return Ok(()),
            Err(err) => {
                last_error = Some(err);
                if attempt < MAX_FETCH_ATTEMPTS {
                    let backoff = Duration::from_secs(1u64 << attempt);
                    progress
                        .set_phase(&format!("fetch failed; retrying in {}s", backoff.as_secs()));
                    std::thread::sleep(backoff);
                }
            }
        }
    }
    Err(last_error.expect("at least one exact-fetch attempt ran"))
}

fn require_build_https(url: &str) -> Result<(), String> {
    match gix_policy::classify_source(url)? {
        gix_policy::InProcessSource::Https => Ok(()),
        gix_policy::InProcessSource::Http => Err(format!(
            "build-time exact source acquisition requires https://; refusing {url}"
        )),
        gix_policy::InProcessSource::Local(_) => Err(format!(
            "build-time exact source acquisition requires an HTTPS remote \
             handled in-process; refusing local repository {url}"
        )),
    }
}

fn clone_one(
    url: &str,
    revision: &str,
    destination: &Path,
    progress: &ProgressReporter,
) -> Result<(), String> {
    clone_one_with_options(url, revision, destination, progress, open_options())
}

fn clone_one_with_options(
    url: &str,
    revision: &str,
    destination: &Path,
    progress: &ProgressReporter,
    open_options: gix::open::Options,
) -> Result<(), String> {
    match gix_policy::classify_source(url)? {
        gix_policy::InProcessSource::Http | gix_policy::InProcessSource::Https => {}
        gix_policy::InProcessSource::Local(_) => {
            return Err(format!(
                "build-time acquisition will not construct gix's local remote \
                 transport for {url}"
            ));
        }
    }
    let interrupt = AtomicBool::new(false);
    let mut repo = gix::ThreadSafeRepository::init_opts(
        destination,
        gix::create::Kind::WithWorktree,
        gix::create::Options::default(),
        open_options,
    )
    .map_err(|err| format!("initialize exact checkout: {err}"))?
    .to_thread_local();

    // Callers that still use this non-CAS compatibility seam get the same
    // bounded worker shape as source nodes. Recursive production acquisition
    // runs through the jobserver-coordinated graph above.
    let workers = 1;
    repo.config_snapshot_mut()
        .set_value(
            &gix::config::tree::Pack::THREADS,
            workers.to_string().as_str(),
        )
        .map_err(|err| format!("configure exact-checkout pack workers: {err}"))?;

    let source = if revision.starts_with("refs/") {
        revision.to_string()
    } else {
        gix::ObjectId::from_hex(revision.as_bytes())
            .map_err(|err| format!("parse exact revision {revision}: {err}"))?
            .to_string()
    };
    let refspec = format!("+{source}:{FETCHED_REF}");
    let mut remote = repo
        .remote_at_without_url_rewrite(url)
        .map_err(|err| format!("prepare exact remote {url}: {err}"))?
        .with_fetch_tags(gix::remote::fetch::Tags::None);
    remote
        .replace_refspecs([refspec.as_str()], gix::remote::Direction::Fetch)
        .map_err(|err| format!("configure exact refspec {source}: {err}"))?;

    let negotiate = progress.item("negotiating exact shallow fetch");
    let mut connection = remote
        .connect(gix::remote::Direction::Fetch)
        .map_err(|err| format!("connect exact remote {url}: {err}"))?;
    connection.set_credentials(gix_policy::reject_credentials);
    let prepare = connection
        .prepare_fetch(negotiate, gix::remote::ref_map::Options::default())
        .map_err(|err| format!("map exact revision {revision}: {err}"))?
        .with_shallow(gix::remote::fetch::Shallow::DepthAtRemote(
            1.try_into().expect("one is non-zero"),
        ));
    let receive = progress.item("receiving and indexing shallow pack");
    prepare
        .receive(receive, &interrupt)
        .map_err(|err| format!("fetch exact revision {revision}: {err}"))?;

    let mut fetched_ref = repo
        .find_reference(FETCHED_REF)
        .map_err(|err| format!("find fetched revision {revision}: {err}"))?;
    let commit_id = fetched_ref
        .peel_to_commit()
        .map_err(|err| format!("peel fetched revision {revision} to commit: {err}"))?
        .id;
    materialize_commit(&repo, commit_id, destination, workers, progress, &interrupt)
}

#[cfg(test)]
pub(crate) struct HttpTransportLimitsForTest<'a> {
    pub connect_timeout_ms: u64,
    pub low_speed_limit: u32,
    pub low_speed_time_seconds: u64,
    pub proxy: &'a str,
}

#[cfg(test)]
pub(crate) fn clone_one_with_transport_limits_for_test(
    url: &str,
    revision: &str,
    destination: &Path,
    progress: &ProgressReporter,
    transport: HttpTransportLimitsForTest<'_>,
) -> Result<(), String> {
    clone_one_with_options(
        url,
        revision,
        destination,
        progress,
        open_options_with_transport_limits(
            transport.connect_timeout_ms,
            transport.low_speed_limit,
            transport.low_speed_time_seconds,
            Some(transport.proxy),
            Some(""),
        ),
    )
}

fn materialize_commit(
    repo: &gix::Repository,
    commit_id: gix::ObjectId,
    destination: &Path,
    workers: usize,
    progress: &ProgressReporter,
    interrupt: &AtomicBool,
) -> Result<(), String> {
    repo.reference(
        "HEAD",
        commit_id,
        gix::refs::transaction::PreviousValue::Any,
        "ktstr build: exact detached checkout",
    )
    .map_err(|err| format!("set detached HEAD for exact commit {commit_id}: {err}"))?;

    let tree_id = repo
        .find_object(commit_id)
        .map_err(|err| format!("find exact commit {commit_id}: {err}"))?
        .peel_to_tree()
        .map_err(|err| format!("peel exact commit {commit_id} to tree: {err}"))?
        .id;
    let mut index = repo
        .index_from_tree(&tree_id)
        .map_err(|err| format!("build exact-checkout index: {err}"))?;
    let mut options = repo
        .checkout_options(gix::worktree::stack::state::attributes::Source::IdMapping)
        .map_err(|err| format!("configure exact checkout: {err}"))?;
    options.destination_is_initially_empty = true;
    options.keep_going = false;
    options.thread_limit = Some(workers);
    let objects = repo
        .objects
        .clone()
        .into_arc()
        .map_err(|err| format!("prepare exact-checkout object database: {err}"))?;
    let checkout = progress.item("checking out exact tree");
    gix::worktree::state::checkout(
        &mut index,
        destination,
        objects,
        &checkout,
        &gix::progress::Discard,
        interrupt,
        options,
    )
    .map_err(|err| format!("check out exact tree {commit_id}: {err}"))?;
    index
        .write(Default::default())
        .map_err(|err| format!("write exact-checkout index: {err}"))?;
    Ok(())
}

#[cfg(test)]
pub(crate) fn materialize_commit_for_test(
    repo: &gix::Repository,
    commit_id: gix::ObjectId,
    destination: &Path,
    progress: &ProgressReporter,
) -> Result<(), String> {
    materialize_commit(
        repo,
        commit_id,
        destination,
        1,
        progress,
        &AtomicBool::new(false),
    )
}

pub(crate) fn open_options() -> gix::open::Options {
    gix_policy::open_options()
}

#[cfg(test)]
pub(crate) use gix_policy::reject_credentials as reject_credentials_for_test;

#[cfg(test)]
fn open_options_with_transport_limits(
    connect_timeout_ms: u64,
    low_speed_limit: u32,
    low_speed_time_seconds: u64,
    proxy: Option<&str>,
    no_proxy: Option<&str>,
) -> gix::open::Options {
    gix_policy::open_options_with_transport_limits(
        connect_timeout_ms,
        low_speed_limit,
        low_speed_time_seconds,
        proxy,
        no_proxy,
    )
}

#[cfg(test)]
pub(crate) fn http_proxy_fixture_open_options_for_test(proxy: &str) -> gix::open::Options {
    open_options_with_transport_limits(20_000, 1024, 30, Some(proxy), Some(""))
}

fn validate_relative_submodule_path(path: &Path) -> Result<(), String> {
    if path.as_os_str().is_empty() {
        return Err("submodule path is empty".to_string());
    }
    for component in path.components() {
        match component {
            Component::Normal(_) => {}
            Component::CurDir
            | Component::ParentDir
            | Component::RootDir
            | Component::Prefix(_) => {
                return Err(format!(
                    "submodule path must stay below its parent checkout: {}",
                    path.display()
                ));
            }
        }
    }
    Ok(())
}

fn resolve_submodule_url(parent: &str, child: &gix::Url) -> Result<String, String> {
    let serialized = child.to_bstring();
    let child_text = serialized
        .to_str()
        .map_err(|err| format!("submodule URL is not UTF-8: {err}"))?;
    if !child_text.starts_with("./") && !child_text.starts_with("../") {
        return Ok(child_text.to_string());
    }

    let mut parent = gix::Url::from_bytes(parent.as_bytes().as_bstr())
        .map_err(|err| format!("parse parent URL {parent}: {err}"))?;
    let parent_path = parent
        .path
        .to_str()
        .map_err(|err| format!("parent URL path is not UTF-8: {err}"))?;
    let leading_slash = parent_path.starts_with('/');
    let mut components: Vec<&str> = parent_path
        .split('/')
        .filter(|component| !component.is_empty())
        .collect();
    for component in child_text.split('/') {
        match component {
            "" | "." => {}
            ".." => {
                components.pop().ok_or_else(|| {
                    format!("relative submodule URL escapes parent URL root: {child_text}")
                })?;
            }
            component => components.push(component),
        }
    }
    let mut path = components.join("/");
    if leading_slash {
        path.insert(0, '/');
    }
    parent.path = path.into();
    Ok(parent.to_bstring().to_string())
}

struct RemoveDirOnDrop(PathBuf);

impl RemoveDirOnDrop {
    fn disarm(self) {
        std::mem::forget(self);
    }
}

impl Drop for RemoveDirOnDrop {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A lightweight prodash-to-`cargo:warning` bridge.
///
/// Phase changes are printed immediately. While a phase is blocked in gix or a
/// cross-process lock, the reporter snapshots gix's real counters and emits an
/// escape-free heartbeat at least every ten seconds.
pub(crate) struct ProgressReporter {
    label: String,
    started: Instant,
    phase: Arc<Mutex<String>>,
    root: Arc<gix::progress::tree::Root>,
    stop: Arc<(Mutex<bool>, Condvar)>,
    thread: Option<JoinHandle<()>>,
}

type ProgressTask = Box<dyn FnOnce() + Send + 'static>;

impl ProgressReporter {
    pub(crate) fn new(label: &str) -> Self {
        Self::new_with_spawn(label, |task| {
            std::thread::Builder::new()
                .name("ktstr-gix-progress".to_string())
                .spawn(task)
        })
    }

    fn new_with_spawn(
        label: &str,
        spawn: impl FnOnce(ProgressTask) -> std::io::Result<JoinHandle<()>>,
    ) -> Self {
        let root = gix::progress::tree::Root::new();
        let phase = Arc::new(Mutex::new(String::from("starting")));
        let stop = Arc::new((Mutex::new(false), Condvar::new()));
        let started = Instant::now();
        println!("cargo:warning={label}: starting");
        let task: ProgressTask = Box::new({
            let label = label.to_string();
            let phase = Arc::clone(&phase);
            let root = Arc::clone(&root);
            let stop = Arc::clone(&stop);
            move || heartbeat_loop(label, phase, root, stop)
        });
        let thread = match spawn(task) {
            Ok(thread) => Some(thread),
            Err(error) => {
                println!(
                    "cargo:warning={label}: heartbeat thread unavailable ({error}); \
                     continuing with phase updates"
                );
                None
            }
        };
        Self {
            label: label.to_string(),
            started,
            phase,
            root,
            stop,
            thread,
        }
    }

    #[cfg(test)]
    pub(crate) fn with_failed_spawn_for_test(label: &str) -> Self {
        Self::new_with_spawn(label, |_task| {
            Err(std::io::Error::other(
                "injected build-progress thread spawn failure",
            ))
        })
    }

    pub(crate) fn set_phase(&self, phase: &str) {
        *self.phase.lock().unwrap_or_else(|err| err.into_inner()) = phase.to_string();
        println!("cargo:warning={}: {phase}", self.label);
    }

    fn item(&self, phase: &str) -> gix::progress::tree::Item {
        self.set_phase(phase);
        self.root.add_child(phase.to_string())
    }

    #[cfg(test)]
    pub(crate) fn heartbeat_thread_active_for_test(&self) -> bool {
        self.thread.is_some()
    }

    fn shutdown(&mut self) {
        {
            let (lock, wake) = &*self.stop;
            *lock.lock().unwrap_or_else(|err| err.into_inner()) = true;
            wake.notify_all();
        }
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }

    pub(crate) fn finish(mut self) {
        self.shutdown();
        println!(
            "cargo:warning={}: complete in {}",
            self.label,
            format_elapsed(self.started.elapsed())
        );
    }
}

impl Drop for ProgressReporter {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn heartbeat_loop(
    label: String,
    phase: Arc<Mutex<String>>,
    root: Arc<gix::progress::tree::Root>,
    stop: Arc<(Mutex<bool>, Condvar)>,
) {
    let mut snapshot = Vec::new();
    let mut last_heartbeat = Instant::now();
    let started = Instant::now();
    loop {
        let (lock, wake) = &*stop;
        let guard = lock.lock().unwrap_or_else(|err| err.into_inner());
        let (guard, _) = wake
            .wait_timeout(guard, POLL)
            .unwrap_or_else(|err| err.into_inner());
        if *guard {
            return;
        }
        drop(guard);
        if last_heartbeat.elapsed() < HEARTBEAT {
            continue;
        }
        root.sorted_snapshot(&mut snapshot);
        let phase = phase.lock().unwrap_or_else(|err| err.into_inner()).clone();
        let detail = snapshot
            .iter()
            .rev()
            .find_map(|(_, task)| {
                task.progress.as_ref().map(|value| {
                    let step = value.step.load(Ordering::Relaxed);
                    match value.done_at {
                        Some(total) if total > 0 => {
                            format!("{}: {step}/{total} ({}%)", task.name, step * 100 / total)
                        }
                        _ => format!("{}: {step}", task.name),
                    }
                })
            })
            .unwrap_or_else(|| "working".to_string());
        println!(
            "cargo:warning={label}: {phase}; {detail}; elapsed {}",
            format_elapsed(started.elapsed())
        );
        last_heartbeat = Instant::now();
    }
}

fn format_elapsed(elapsed: Duration) -> String {
    let seconds = elapsed.as_secs();
    format!("{}m{:02}s", seconds / 60, seconds % 60)
}
