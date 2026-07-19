use crate::monitor::cast_analysis::{AddrSpace, CastHit, CastMap};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::fs::{File, OpenOptions};
use std::path::PathBuf;
use std::time::{Duration, SystemTime};

use super::FwdIndexEntry;

// Explicit on-disk FORMAT version for PersistedCastAnalysis. Bump only
// for wire-layout changes (a field add/remove/retype). Analyzer BEHAVIOR
// changes no longer need a manual bump: they self-invalidate via
// ANALYZER_FINGERPRINT (a build.rs hash of the cast-analysis source,
// folded into cache_path below). v13 was the last manual bump — it
// invalidated caches written before the arena_confirmed deferred-resolve
// loop in src/monitor/cast_analysis/mod.rs landed (pre-v13 cast maps lack
// the `target_type_id == 0` arena entries, so a stale v12 cache rendered
// the affected BSS u64 fields as plain integers instead of typed
// pointers).
const SCHEMA_VERSION: u32 = 14;
const CACHE_LOCK_DIR: &str = ".locks-v14";
const CACHE_GC_STAMP: &str = ".gc-v14";
const CACHE_GC_INTERVAL: Duration = Duration::from_secs(60 * 60);
const CACHE_MAX_AGE: Duration = Duration::from_secs(30 * 24 * 60 * 60);
const CACHE_MAX_BYTES: u64 = 256 << 20;

#[derive(Serialize, Deserialize)]
struct PersistedAddrSpace(u8);

impl From<AddrSpace> for PersistedAddrSpace {
    fn from(a: AddrSpace) -> Self {
        match a {
            AddrSpace::Arena => Self(0),
            AddrSpace::Kernel => Self(1),
        }
    }
}

impl PersistedAddrSpace {
    fn into_addr_space(self) -> Option<AddrSpace> {
        match self.0 {
            0 => Some(AddrSpace::Arena),
            1 => Some(AddrSpace::Kernel),
            _ => None,
        }
    }
}

#[derive(Serialize, Deserialize)]
struct PersistedCastHit {
    target_type_id: u32,
    addr_space: PersistedAddrSpace,
    alloc_size: Option<u64>,
}

impl From<CastHit> for PersistedCastHit {
    fn from(h: CastHit) -> Self {
        Self {
            target_type_id: h.target_type_id,
            addr_space: h.addr_space.into(),
            alloc_size: h.alloc_size,
        }
    }
}

impl PersistedCastHit {
    fn into_cast_hit(self) -> Option<CastHit> {
        Some(CastHit {
            target_type_id: self.target_type_id,
            addr_space: self.addr_space.into_addr_space()?,
            alloc_size: self.alloc_size,
        })
    }
}

#[derive(Serialize, Deserialize)]
struct PersistedFwdIndexEntry {
    btfs_idx: u32,
    type_id: u32,
}

impl From<&FwdIndexEntry> for PersistedFwdIndexEntry {
    fn from(e: &FwdIndexEntry) -> Self {
        Self {
            btfs_idx: e.btfs_idx as u32,
            type_id: e.type_id,
        }
    }
}

impl PersistedFwdIndexEntry {
    fn into_fwd_index_entry(self) -> FwdIndexEntry {
        FwdIndexEntry {
            btfs_idx: self.btfs_idx as usize,
            type_id: self.type_id,
        }
    }
}

#[derive(Serialize, Deserialize)]
struct PersistedCastAnalysis {
    schema_version: u32,
    content_hash: u64,
    cast_maps: Vec<Vec<((u32, u32), PersistedCastHit)>>,
    fwd_entries: Vec<(String, PersistedFwdIndexEntry)>,
    btf_count: u32,
    alloc_size_types: Vec<(u64, String)>,
}

pub(super) struct CachedCastAnalysis {
    pub(super) cast_maps: Vec<CastMap>,
    pub(super) fwd_index: HashMap<String, FwdIndexEntry>,
    pub(super) btf_count: usize,
    pub(super) alloc_size_types: Vec<(u64, String)>,
}

fn cache_dir() -> Result<PathBuf> {
    let dir = crate::cache::resolve_cache_root_with_suffix("cast_analysis")
        .context("resolve cast-analysis cache root")?;
    // Reclaim `*.bin.tmp.<pid>` staging files left by interrupted prior
    // runs, once per process on first cache access. try_save writes a
    // pid-suffixed temp then renames; a process that dies between the write
    // and the rename orphans the temp, and nothing else reclaims it.
    static SWEEP_ONCE: std::sync::Once = std::sync::Once::new();
    SWEEP_ONCE.call_once(|| sweep_stale_tmp(&dir));
    maybe_gc_cache(&dir)?;
    Ok(dir)
}

fn cache_file_name(hash: u64) -> String {
    format!(
        "v{SCHEMA_VERSION}_{ANALYZER_FINGERPRINT}_{CARGO_LOCK_FINGERPRINT}_{hash:016x}.bin"
    )
}

fn cache_lock_name(hash: u64) -> String {
    format!("{ANALYZER_FINGERPRINT}_{CARGO_LOCK_FINGERPRINT}_{hash:016x}.lock")
}

fn current_hash_from_cache_name(name: &str) -> Option<u64> {
    let prefix =
        format!("v{SCHEMA_VERSION}_{ANALYZER_FINGERPRINT}_{CARGO_LOCK_FINGERPRINT}_");
    let hash = name.strip_prefix(&prefix)?.strip_suffix(".bin")?;
    (hash.len() == 16).then(|| u64::from_str_radix(hash, 16).ok())?
}

fn open_gc_lock(path: &std::path::Path) -> Result<File> {
    OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(path)
        .with_context(|| format!("open cast-analysis cache lock {}", path.display()))
}

fn cache_entry_is_old(metadata: &std::fs::Metadata, now: SystemTime) -> bool {
    metadata
        .modified()
        .ok()
        .and_then(|modified| now.duration_since(modified).ok())
        .is_some_and(|age| age >= CACHE_MAX_AGE)
}

fn remove_if_present(path: &std::path::Path) -> Result<bool> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error)
            .with_context(|| format!("remove cast-analysis cache {}", path.display())),
    }
}

fn try_remove_current_entry(
    lock_dir: &std::path::Path,
    hash: u64,
    path: &std::path::Path,
) -> Result<bool> {
    let lock_path = lock_dir.join(cache_lock_name(hash));
    let lock = open_gc_lock(&lock_path)?;
    match crate::cache::content::flock_retry(
        &lock,
        rustix::fs::FlockOperation::NonBlockingLockExclusive,
    ) {
        Ok(()) => {
            let removed = remove_if_present(path)?;
            remove_if_present(&lock_path)?;
            Ok(removed)
        }
        Err(error) if error == rustix::io::Errno::WOULDBLOCK => Ok(false),
        Err(error) => Err(error).with_context(|| {
            format!(
                "try-lock cast-analysis cache entry for cleanup {}",
                lock_path.display()
            )
        }),
    }
}

fn gc_cache_at(root: &std::path::Path, now: SystemTime, max_bytes: u64) -> Result<()> {
    std::fs::create_dir_all(root)
        .with_context(|| format!("create cast-analysis cache dir {}", root.display()))?;
    let lock_dir = root.join(CACHE_LOCK_DIR);
    std::fs::create_dir_all(&lock_dir)
        .with_context(|| format!("create cast-analysis lock dir {}", lock_dir.display()))?;
    let namespace_gate_path = lock_dir.join("namespace.lock");
    let namespace_gate = open_gc_lock(&namespace_gate_path)?;
    match crate::cache::content::flock_retry(
        &namespace_gate,
        rustix::fs::FlockOperation::NonBlockingLockExclusive,
    ) {
        Ok(()) => {}
        Err(error) if error == rustix::io::Errno::WOULDBLOCK => return Ok(()),
        Err(error) => {
            return Err(error).context("lock cast-analysis namespace for cleanup");
        }
    }

    struct Candidate {
        path: PathBuf,
        hash: Option<u64>,
        modified: SystemTime,
        len: u64,
        expired: bool,
    }
    let mut candidates = Vec::new();
    let mut total_bytes = 0u64;
    for entry in std::fs::read_dir(root)
        .with_context(|| format!("scan cast-analysis cache {}", root.display()))?
    {
        let entry = entry.context("read cast-analysis cache entry")?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        if !name.ends_with(".bin") || name.contains(".tmp.") {
            continue;
        }
        let metadata = entry
            .metadata()
            .with_context(|| format!("stat cast-analysis cache {}", entry.path().display()))?;
        if !metadata.is_file() {
            continue;
        }
        total_bytes = total_bytes.saturating_add(metadata.len());
        candidates.push(Candidate {
            path: entry.path(),
            hash: current_hash_from_cache_name(name),
            modified: metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH),
            len: metadata.len(),
            expired: cache_entry_is_old(&metadata, now),
        });
    }
    candidates.sort_by_key(|candidate| candidate.modified);
    for candidate in candidates {
        if !candidate.expired && total_bytes <= max_bytes {
            continue;
        }
        let removed = match candidate.hash {
            Some(hash) => try_remove_current_entry(&lock_dir, hash, &candidate.path)?,
            // An older record has no writer in the current namespace. Atomic
            // rename/open-file semantics make unlink safe even if an old
            // process still has the inode open.
            None => remove_if_present(&candidate.path)?,
        };
        if removed {
            total_bytes = total_bytes.saturating_sub(candidate.len);
        }
    }

    let current_lock_prefix =
        format!("{ANALYZER_FINGERPRINT}_{CARGO_LOCK_FINGERPRINT}_");
    for entry in std::fs::read_dir(&lock_dir)
        .with_context(|| format!("scan cast-analysis locks {}", lock_dir.display()))?
    {
        let entry = entry.context("read cast-analysis lock entry")?;
        let name = entry.file_name();
        let Some(hash) = name
            .to_str()
            .and_then(|name| name.strip_prefix(&current_lock_prefix))
            .and_then(|name| name.strip_suffix(".lock"))
            .filter(|hash| hash.len() == 16)
            .and_then(|hash| u64::from_str_radix(hash, 16).ok())
        else {
            continue;
        };
        let data_path = root.join(cache_file_name(hash));
        let metadata = entry
            .metadata()
            .with_context(|| format!("stat cast-analysis lock {}", entry.path().display()))?;
        if !data_path.exists() && cache_entry_is_old(&metadata, now) {
            try_remove_current_entry(&lock_dir, hash, &data_path)?;
        }
    }

    let stamp = root.join(CACHE_GC_STAMP);
    let stamp_file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&stamp)
        .with_context(|| format!("update cast-analysis GC stamp {}", stamp.display()))?;
    stamp_file
        .sync_data()
        .with_context(|| format!("sync cast-analysis GC stamp {}", stamp.display()))?;
    Ok(())
}

fn maybe_gc_cache(root: &std::path::Path) -> Result<()> {
    let stamp = root.join(CACHE_GC_STAMP);
    if stamp.metadata().ok().is_some_and(|metadata| {
        !metadata
            .modified()
            .ok()
            .and_then(|modified| SystemTime::now().duration_since(modified).ok())
            .is_some_and(|age| age >= CACHE_GC_INTERVAL)
    }) {
        return Ok(());
    }
    gc_cache_at(root, SystemTime::now(), CACHE_MAX_BYTES)
}

/// Remove `*.bin.tmp.<pid>` staging files in `dir` whose owning pid is no
/// longer alive (`kill(pid, 0)` -> ESRCH). A file owned by a LIVE pid is a
/// concurrent run's in-flight write and is left untouched; the final
/// `.bin` cache files (no `.tmp.<pid>` suffix) are never matched.
fn sweep_stale_tmp(dir: &std::path::Path) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let self_pid = std::process::id();
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(pid) = name
            .to_str()
            .and_then(|n| n.rsplit_once(".bin.tmp."))
            .and_then(|(_, suffix)| suffix.split('.').next())
            .and_then(|pid| pid.parse::<i32>().ok())
        else {
            continue;
        };
        // Skip non-positive pids (kill(0)/kill(-N) probe process GROUPS)
        // and our own in-flight writes.
        if pid <= 0 || pid == self_pid as i32 {
            continue;
        }
        let dead = matches!(
            nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid), None),
            Err(nix::errno::Errno::ESRCH),
        );
        if dead {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

/// Compile-time fingerprint of the cast-analysis source, emitted by
/// build.rs (`cast_analyzer_fingerprint`) from a SipHash-13 of every
/// non-test `.rs` under `src/monitor/cast_analysis`,
/// `src/vmm/cast_analysis_load`, `src/monitor/sdt_alloc` (which
/// resolves the cached `alloc_size_types`), plus `src/monitor/btf_render` and
/// `src/monitor/bpf_map` (whose modifier-peel / struct-resolve helpers
/// resolve every cast's terminal type). Folding it into the cache key makes the
/// cache self-invalidate whenever the analyzer's behavior changes, with
/// no manual `SCHEMA_VERSION` bump — closing the footgun where a stale
/// cache served the old analyzer's output and masked a fixed bug as a
/// flake.
const ANALYZER_FINGERPRINT: &str = env!("KTSTR_CAST_ANALYZER_FINGERPRINT");

/// Compile-time fingerprint of the whole `Cargo.lock`, emitted by
/// build.rs (`cargo_lock_fingerprint`). Folded into the cache key
/// alongside [`ANALYZER_FINGERPRINT`] so a dependency bump — a `btf-rs`
/// (BTF parsing) or `libbpf-rs` / `libbpf-sys` (BPF-opcode constants)
/// version change that can alter the cast map — invalidates the cache
/// even when the analyzer source is unchanged. Only the cast-analysis
/// cache folds this in: the kernels / models / disk_template caches are
/// produced by external tools (Kbuild, downloads, in-VM mkfs) and are
/// dependency-independent, so fingerprinting them would force pointless
/// rebuilds.
const CARGO_LOCK_FINGERPRINT: &str = env!("KTSTR_CARGO_LOCK_FINGERPRINT");

fn cache_path(hash: u64) -> Result<PathBuf> {
    Ok(cache_dir()?.join(cache_file_name(hash)))
}

pub(super) struct CoordinationPaths {
    pub(super) lock: PathBuf,
    pub(super) namespace_gate: PathBuf,
}

pub(super) fn coordination_paths(hash: u64) -> Result<CoordinationPaths> {
    let lock_dir = cache_dir()?.join(CACHE_LOCK_DIR);
    std::fs::create_dir_all(&lock_dir)
        .with_context(|| format!("create cast-analysis lock dir {}", lock_dir.display()))?;
    Ok(CoordinationPaths {
        lock: lock_dir.join(cache_lock_name(hash)),
        namespace_gate: lock_dir.join("namespace.lock"),
    })
}

pub(super) fn try_load(hash: u64) -> Result<Option<CachedCastAnalysis>> {
    let path = cache_path(hash)?;
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(error)
                .with_context(|| format!("read cast-analysis cache {}", path.display()));
        }
    };
    let persisted: PersistedCastAnalysis = postcard::from_bytes(&bytes)
        .with_context(|| format!("decode cast-analysis cache {}", path.display()))?;

    anyhow::ensure!(
        persisted.schema_version == SCHEMA_VERSION,
        "cast-analysis cache schema mismatch in {}",
        path.display()
    );
    anyhow::ensure!(
        persisted.content_hash == hash,
        "cast-analysis cache content hash mismatch in {}",
        path.display()
    );
    let mut cast_maps = Vec::with_capacity(persisted.cast_maps.len());
    for persisted_map in persisted.cast_maps {
        let mut cast_map = BTreeMap::new();
        for (key, hit) in persisted_map {
            let hit = hit.into_cast_hit().with_context(|| {
                format!(
                    "invalid address space in cast-analysis cache {}",
                    path.display()
                )
            })?;
            cast_map.insert(key, hit);
        }
        cast_maps.push(cast_map);
    }

    let mut fwd_index = HashMap::new();
    for (name, entry) in persisted.fwd_entries {
        fwd_index.insert(name, entry.into_fwd_index_entry());
    }

    tracing::info!(
        casts = cast_maps.iter().map(BTreeMap::len).sum::<usize>(),
        fwd = fwd_index.len(),
        path = %path.display(),
        "cast_analysis: loaded from disk cache"
    );
    Ok(Some(CachedCastAnalysis {
        cast_maps,
        fwd_index,
        btf_count: persisted.btf_count as usize,
        alloc_size_types: persisted.alloc_size_types,
    }))
}

pub(super) fn try_save(
    hash: u64,
    cast_maps: &[std::sync::Arc<CastMap>],
    fwd_index: &HashMap<String, FwdIndexEntry>,
    btf_count: usize,
    alloc_size_types: &[(u64, String)],
) -> Result<()> {
    let path = cache_path(hash)?;
    let btf_count = u32::try_from(btf_count).context("cast-analysis BTF count exceeds u32")?;

    let persisted = PersistedCastAnalysis {
        schema_version: SCHEMA_VERSION,
        content_hash: hash,
        cast_maps: cast_maps
            .iter()
            .map(|cast_map| {
                cast_map
                    .iter()
                    .map(|(&key, &hit)| (key, hit.into()))
                    .collect()
            })
            .collect(),
        fwd_entries: fwd_index
            .iter()
            .map(|(k, v)| (k.clone(), v.into()))
            .collect(),
        btf_count,
        alloc_size_types: alloc_size_types.to_vec(),
    };

    let encoded = postcard::to_stdvec(&persisted).context("encode cast analysis for disk cache")?;

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create cast-analysis cache dir {}", parent.display()))?;
    }

    let mut temporary = tempfile::Builder::new()
        .prefix(&format!(
            ".cast-analysis-{hash:016x}.bin.tmp.{}.",
            std::process::id()
        ))
        .tempfile_in(
            path.parent()
                .context("cast-analysis cache path has no parent")?,
        )
        .context("create cast-analysis cache temp")?;
    use std::io::Write as _;
    temporary
        .write_all(&encoded)
        .context("write cast-analysis cache temp")?;
    temporary
        .as_file()
        .sync_all()
        .context("sync cast-analysis cache temp")?;
    temporary
        .persist(&path)
        .map_err(|error| error.error)
        .with_context(|| format!("publish cast-analysis cache {}", path.display()))?;
    crate::cache::fsync_parent(&path).context("sync cast-analysis cache parent")?;
    tracing::debug!(
        path = %path.display(),
        bytes = encoded.len(),
        "cast_analysis: saved to disk cache"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::test_helpers::{isolated_cache_dir, lock_env};

    #[test]
    fn sweep_removes_dead_pid_tmp_keeps_live_and_final() {
        // A `*.bin.tmp.<pid>` staging file owned by a dead pid is reclaimed;
        // one owned by our own (live) pid and the final `.bin` cache file
        // are kept. pid 2147483647 (i32::MAX) is above pid_max -> ESRCH.
        let base = std::env::temp_dir().join(format!("ktstr-castsweep-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).expect("mk temp root");
        let dead = base.join("abc.bin.tmp.2147483647");
        let live = base.join(format!("abc.bin.tmp.{}", std::process::id()));
        let final_cache = base.join("abc.bin");
        for p in [&dead, &live, &final_cache] {
            std::fs::write(p, b"x").expect("write fixture");
        }

        sweep_stale_tmp(&base);

        assert!(!dead.exists(), "a dead-owner .bin.tmp file is reclaimed");
        assert!(live.exists(), "our own (live) .bin.tmp file is kept");
        assert!(
            final_cache.exists(),
            "the final .bin cache file is untouched"
        );

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn gc_bounds_records_and_current_locks_without_unlinking_live_builder() {
        let root = tempfile::TempDir::new().unwrap();
        let lock_dir = root.path().join(CACHE_LOCK_DIR);
        std::fs::create_dir_all(&lock_dir).unwrap();
        let removable_hash = 0x1111_1111_1111_1111;
        let live_hash = 0x2222_2222_2222_2222;
        let obsolete = root.path().join("v13_obsolete.bin");
        std::fs::write(
            root.path().join(cache_file_name(removable_hash)),
            vec![0u8; 4096],
        )
        .unwrap();
        std::fs::write(lock_dir.join(cache_lock_name(removable_hash)), b"").unwrap();
        std::fs::write(
            root.path().join(cache_file_name(live_hash)),
            vec![0u8; 4096],
        )
        .unwrap();
        let live_lock_path = lock_dir.join(cache_lock_name(live_hash));
        let live_lock = open_gc_lock(&live_lock_path).unwrap();
        crate::cache::content::flock_retry(
            &live_lock,
            rustix::fs::FlockOperation::LockExclusive,
        )
        .unwrap();
        std::fs::write(&obsolete, vec![0u8; 4096]).unwrap();

        let future = SystemTime::now() + CACHE_MAX_AGE + Duration::from_secs(1);
        gc_cache_at(root.path(), future, 0).unwrap();
        assert!(
            !root.path().join(cache_file_name(removable_hash)).exists()
        );
        assert!(!lock_dir.join(cache_lock_name(removable_hash)).exists());
        assert!(!obsolete.exists(), "obsolete schema records must be bounded");
        assert!(
            root.path().join(cache_file_name(live_hash)).exists(),
            "GC must skip a record whose builder lock is live"
        );
        assert!(live_lock_path.exists());

        drop(live_lock);
        gc_cache_at(root.path(), future, 0).unwrap();
        assert!(!root.path().join(cache_file_name(live_hash)).exists());
        assert!(!live_lock_path.exists());
        assert!(lock_dir.join("namespace.lock").exists());
        assert!(root.path().join(CACHE_GC_STAMP).exists());
    }

    #[test]
    fn roundtrip_save_load() {
        let _env_lock = lock_env();
        let _cache = isolated_cache_dir();

        let mut cast_map = BTreeMap::new();
        cast_map.insert(
            (2, 8),
            CastHit {
                target_type_id: 5,
                addr_space: AddrSpace::Arena,
                alloc_size: None,
            },
        );
        cast_map.insert(
            (3, 16),
            CastHit {
                target_type_id: 7,
                addr_space: AddrSpace::Kernel,
                alloc_size: None,
            },
        );
        let mut fwd_index = HashMap::new();
        fwd_index.insert(
            "cgx_target".to_string(),
            FwdIndexEntry {
                btfs_idx: 1,
                type_id: 4,
            },
        );

        let hash = 0xDEAD_BEEF_CAFE_1234u64;
        let cast_maps = vec![
            std::sync::Arc::new(cast_map),
            std::sync::Arc::new(BTreeMap::new()),
        ];
        try_save(hash, &cast_maps, &fwd_index, 2, &[]).expect("save cast analysis");

        let loaded = try_load(hash).expect("load cast analysis");
        assert!(loaded.is_some(), "roundtrip must succeed");
        let loaded = loaded.unwrap();
        let loaded_map = &loaded.cast_maps[0];
        assert_eq!(loaded_map.len(), 2);
        assert_eq!(loaded_map.get(&(2, 8)).unwrap().target_type_id, 5);
        assert_eq!(
            loaded_map.get(&(2, 8)).unwrap().addr_space,
            AddrSpace::Arena
        );
        assert_eq!(
            loaded_map.get(&(3, 16)).unwrap().addr_space,
            AddrSpace::Kernel
        );
        assert!(loaded.cast_maps[1].is_empty());
        assert_eq!(loaded.btf_count, 2);
        assert_eq!(loaded.fwd_index.len(), 1);
        assert_eq!(loaded.fwd_index["cgx_target"].btfs_idx, 1);
        assert_eq!(loaded.fwd_index["cgx_target"].type_id, 4);
    }

    #[test]
    fn roundtrip_preserves_independent_btf_count() {
        let _env_lock = lock_env();
        let _cache = isolated_cache_dir();

        let mut cast_map = BTreeMap::new();
        cast_map.insert(
            (1, 0),
            CastHit {
                target_type_id: 9,
                addr_space: AddrSpace::Arena,
                alloc_size: None,
            },
        );
        let fwd_index = HashMap::new();
        let hash = 0x1234_5678_9ABC_DEF0u64;
        try_save(hash, &[std::sync::Arc::new(cast_map)], &fwd_index, 3, &[])
            .expect("save independent BTF count");

        let loaded = try_load(hash)
            .expect("load independent BTF count")
            .expect("persisted entry");
        assert_eq!(loaded.btf_count, 3);
    }

    #[test]
    fn load_nonexistent_returns_none() {
        let _env_lock = lock_env();
        assert!(
            try_load(0xFFFF_FFFF_FFFF_FFFFu64)
                .expect("nonexistent cache lookup")
                .is_none()
        );
    }

    #[test]
    fn try_save_persists_empty_result_as_negative_entry() {
        let _env_lock = lock_env();
        let _cache = isolated_cache_dir();

        // Empty analysis is still a completed content-addressed result.
        // Persisting it lets cross-process waiters distinguish a negative hit
        // from a miss instead of serially repeating the expensive analysis.
        let cast_map = BTreeMap::new();
        let fwd_index = HashMap::new();
        let hash = 0xABCD_1234_5678_9999u64;
        try_save(hash, &[std::sync::Arc::new(cast_map)], &fwd_index, 2, &[])
            .expect("persist negative entry");

        let loaded = try_load(hash)
            .expect("load negative entry")
            .expect("empty negative result must be persisted");
        assert!(loaded.cast_maps.iter().all(BTreeMap::is_empty));
        assert!(loaded.fwd_index.is_empty());
        assert!(loaded.alloc_size_types.is_empty());
    }
}
