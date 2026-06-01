use crate::monitor::cast_analysis::{AddrSpace, CastHit, CastMap};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

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
const SCHEMA_VERSION: u32 = 13;

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
    cast_entries: Vec<((u32, u32), PersistedCastHit)>,
    fwd_entries: Vec<(String, PersistedFwdIndexEntry)>,
    btf_count: u32,
    alloc_size_types: Vec<(u64, String)>,
}

fn cache_dir() -> Option<PathBuf> {
    crate::cache::resolve_cache_root_with_suffix("cast_analysis").ok()
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

fn cache_path(hash: u64) -> Option<PathBuf> {
    cache_dir()
        .map(|d| d.join(format!("v{SCHEMA_VERSION}_{ANALYZER_FINGERPRINT}_{hash:016x}.bin")))
}

#[allow(clippy::type_complexity)]
pub(super) fn try_load(
    hash: u64,
    expected_btf_count: usize,
) -> Option<(CastMap, HashMap<String, FwdIndexEntry>, Vec<(u64, String)>)> {
    let path = cache_path(hash)?;
    let bytes = std::fs::read(&path).ok()?;
    let persisted: PersistedCastAnalysis = postcard::from_bytes(&bytes).ok()?;

    if persisted.schema_version != SCHEMA_VERSION {
        return None;
    }
    if persisted.content_hash != hash {
        return None;
    }
    if persisted.btf_count as usize != expected_btf_count {
        tracing::debug!(
            expected = expected_btf_count,
            cached = persisted.btf_count,
            "cast_analysis: disk cache btf_count mismatch; treating as miss"
        );
        return None;
    }

    let mut cast_map = BTreeMap::new();
    for (key, hit) in persisted.cast_entries {
        cast_map.insert(key, hit.into_cast_hit()?);
    }

    let mut fwd_index = HashMap::new();
    for (name, entry) in persisted.fwd_entries {
        fwd_index.insert(name, entry.into_fwd_index_entry());
    }

    tracing::info!(
        casts = cast_map.len(),
        fwd = fwd_index.len(),
        path = %path.display(),
        "cast_analysis: loaded from disk cache"
    );
    Some((cast_map, fwd_index, persisted.alloc_size_types))
}

pub(super) fn try_save(
    hash: u64,
    cast_map: &CastMap,
    fwd_index: &HashMap<String, FwdIndexEntry>,
    btf_count: usize,
    alloc_size_types: &[(u64, String)],
) {
    // Symmetric with get_full's read-side collapse (mod.rs:401/431):
    // a result with no cast entries AND an empty fwd index loads back
    // as None, so persisting it only wastes a disk write plus a
    // recompute on every subsequent run. Skip the write. (alloc_size_types
    // alone never resurrects a cached result -- the read-side collapse
    // ignores it -- so it does not gate this guard.)
    if cast_map.is_empty() && fwd_index.is_empty() {
        return;
    }
    let Some(path) = cache_path(hash) else { return };

    let persisted = PersistedCastAnalysis {
        schema_version: SCHEMA_VERSION,
        content_hash: hash,
        cast_entries: cast_map.iter().map(|(&k, &v)| (k, v.into())).collect(),
        fwd_entries: fwd_index
            .iter()
            .map(|(k, v)| (k.clone(), v.into()))
            .collect(),
        btf_count: btf_count as u32,
        alloc_size_types: alloc_size_types.to_vec(),
    };

    let encoded = match postcard::to_stdvec(&persisted) {
        Ok(v) => v,
        Err(e) => {
            tracing::debug!(error = %e, "cast_analysis: failed to encode for disk cache");
            return;
        }
    };

    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    let tmp = path.with_extension(format!("bin.tmp.{}", std::process::id()));
    if std::fs::write(&tmp, &encoded).is_ok() {
        if std::fs::rename(&tmp, &path).is_err() {
            let _ = std::fs::remove_file(&tmp);
        } else {
            tracing::debug!(
                path = %path.display(),
                bytes = encoded.len(),
                "cast_analysis: saved to disk cache"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::test_helpers::{isolated_cache_dir, lock_env};

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
        try_save(hash, &cast_map, &fwd_index, 2, &[]);

        let loaded = try_load(hash, 2);
        assert!(loaded.is_some(), "roundtrip must succeed");
        let (loaded_map, loaded_fwd, _alloc_types) = loaded.unwrap();
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
        assert_eq!(loaded_fwd.len(), 1);
        assert_eq!(loaded_fwd["cgx_target"].btfs_idx, 1);
        assert_eq!(loaded_fwd["cgx_target"].type_id, 4);
    }

    #[test]
    fn load_wrong_btf_count_returns_none() {
        let _env_lock = lock_env();
        let _cache = isolated_cache_dir();

        // Non-empty cast map so try_save actually persists -- the
        // empty-result guard would otherwise skip the write and this test
        // would pass vacuously via a nonexistent-file miss instead of
        // exercising the btf_count check it names.
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
        try_save(hash, &cast_map, &fwd_index, 3, &[]);

        assert!(
            try_load(hash, 5).is_none(),
            "btf_count mismatch must return None"
        );
    }

    #[test]
    fn load_nonexistent_returns_none() {
        let _env_lock = lock_env();
        assert!(try_load(0xFFFF_FFFF_FFFF_FFFFu64, 1).is_none());
    }

    #[test]
    fn try_save_skips_empty_result() {
        let _env_lock = lock_env();
        let _cache = isolated_cache_dir();

        // An empty result (no cast entries, empty fwd index) loads back as
        // None via get_full's collapse, so try_save must not persist it --
        // otherwise every run pays a disk write plus a recompute on the
        // next load. Verify the write is skipped: try_load finds no file.
        let cast_map = BTreeMap::new();
        let fwd_index = HashMap::new();
        let hash = 0xABCD_1234_5678_9999u64;
        try_save(hash, &cast_map, &fwd_index, 2, &[]);

        assert!(
            try_load(hash, 2).is_none(),
            "empty result must not be persisted"
        );
    }
}
