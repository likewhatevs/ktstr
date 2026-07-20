//! Locate the `vmlinux` ELF that pairs with a guest kernel image.
//!
//! Used by the host monitor and BPF reader to resolve symbols and
//! BTF offsets against the running guest kernel.

use crate::sync::RwLockExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock, RwLock};

/// Process-global cache of vmlinux ELF bytes keyed by canonical path.
///
/// `collect_verifier_stats` is called once per failure-dump cycle; in a
/// nextest test process running many `#[ktstr_test]` cases that boot
/// fresh VMs against the same kernel, repeating the file read for every
/// VM costs 50-340 MB of disk I/O on the freeze-coord cleanup critical
/// path. Caching the bytes once per canonical path collapses every
/// subsequent VM's read to a hash lookup + `Arc::clone`.
///
/// The cached entry pairs the bytes with the file's mtime at read
/// time. On every lookup we re-stat the file: if the stat'd mtime
/// matches the cached mtime, the cached bytes are reused; otherwise
/// the entry is replaced with a fresh read. This catches the case
/// where a developer rebuilds vmlinux mid-process — the user gets
/// the new bytes instead of stale cached bytes that would mismatch
/// against the running guest kernel. Stat-cost is microseconds vs
/// ~100ms for the cached read it gates, so the invalidation check
/// is effectively free on the hot path.
///
/// The cache key is the canonicalized path so symlinks across cache
/// / source-tree layouts collapse to one entry. A `canonicalize` or
/// `metadata` failure (EACCES, missing target) skips the cache and
/// falls through to the direct read. The error case is not cached:
/// a transient EACCES (e.g. a half-written cache entry whose
/// permissions arrive on the next ms) should not poison the cache
/// for the rest of the process.
static VMLINUX_BYTES_CACHE: OnceLock<RwLock<std::collections::HashMap<PathBuf, CachedEntry>>> =
    OnceLock::new();

/// One slot in [`VMLINUX_BYTES_CACHE`]. The mtime gates the bytes —
/// a mismatch on lookup invalidates and triggers a re-read.
struct CachedEntry {
    mtime: std::time::SystemTime,
    bytes: Arc<Vec<u8>>,
}

/// Return the cached vmlinux ELF bytes for `path`, populating the cache
/// on first read and invalidating on file modification.
///
/// Returns `None` when `path` is unreadable (stat or read failure).
/// The error case is not cached: a transient EACCES (e.g. a
/// half-written cache entry whose permissions arrive on the next
/// ms) should not poison the cache for the rest of the process.
pub(crate) fn cached_vmlinux_bytes(path: &Path) -> Option<Arc<Vec<u8>>> {
    let canon = std::fs::canonicalize(path)
        .ok()
        .unwrap_or_else(|| path.to_path_buf());
    // mtime captured before the read so a concurrent write that
    // finishes mid-read produces a mtime-bumped entry on the NEXT
    // lookup (this insert may carry the M1 mtime with mid-stream
    // bytes; the next lookup sees M2 ≠ M1 and re-reads cleanly).
    let mtime = std::fs::metadata(&canon).and_then(|m| m.modified()).ok()?;
    let slot = VMLINUX_BYTES_CACHE.get_or_init(|| RwLock::new(std::collections::HashMap::new()));
    {
        let read = slot.read_unpoisoned();
        if let Some(entry) = read.get(&canon)
            && entry.mtime == mtime
        {
            return Some(Arc::clone(&entry.bytes));
        }
    }
    // Read outside the write lock so a slow read doesn't block other
    // canonical paths' lookups. A racing second reader will pay the
    // same read once each — acceptable: if mtime matches both reads
    // produce the same bytes.
    let bytes = std::fs::read(&canon).ok()?;
    let arc = Arc::new(bytes);
    let mut write = slot.write_unpoisoned();
    // Always overwrite: if no entry, insert; if entry exists with
    // matching mtime (racing reader won the insert race), our overwrite
    // is identical; if entry exists with stale mtime (file rewrote
    // between our read-lock release and our write-lock acquire), the
    // stale entry is replaced.
    write.insert(
        canon,
        CachedEntry {
            mtime,
            bytes: Arc::clone(&arc),
        },
    );
    Some(arc)
}

/// Process-global cache of the *parsed* vmlinux products, keyed the
/// same way as [`VMLINUX_BYTES_CACHE`] (canonical path + mtime).
///
/// A VM run parses the host vmlinux twice: the freeze-coord inline path
/// derives the link-time text/syscall KVAs from `KernelSymbols`, and
/// `start_monitor` re-derives `KernelSymbols` plus the BTF-backed
/// `KernelOffsets`/`BpfProgOffsets`/`PsiGroupOffsets`. The file read is
/// already deduped by [`VMLINUX_BYTES_CACHE`], but each consumer still
/// ran its own `goblin::elf::Elf::parse` (and the monitor its own BTF
/// parse) over ~50 MB+ of debug vmlinux — seconds of pure CPU repeated
/// per VM. Caching the *derived* products (all owned `u64`s / small
/// structs / an `Arc<Btf>`, so no self-referential borrow of the byte
/// buffer) collapses every consumer past the first to a hash lookup +
/// clone, once per (canonical path, mtime) per process.
static VMLINUX_ARTIFACTS_CACHE: OnceLock<
    RwLock<std::collections::HashMap<PathBuf, CachedArtifacts>>,
> = OnceLock::new();

/// One slot in [`VMLINUX_ARTIFACTS_CACHE`]. The mtime gates the parsed
/// products exactly as [`CachedEntry`] gates the bytes.
struct CachedArtifacts {
    mtime: std::time::SystemTime,
    artifacts: Arc<VmlinuxArtifacts>,
}

/// Parsed vmlinux products shared between the freeze-coord inline
/// link-KVA resolution and the monitor thread.
///
/// Every field is owned (no borrow of the underlying ELF bytes), so a
/// single parse can be cached and cloned by both consumers.
pub(crate) struct VmlinuxArtifacts {
    /// ELF symbol addresses. Present whenever the ELF parsed and the
    /// mandatory symbols resolved. The inline link-KVA path needs only
    /// this; a symbol-parse failure fails the whole artifact (returns
    /// `None`), matching the pre-cache inline path's `.ok()` degrade.
    pub symbols: crate::monitor::symbols::KernelSymbols,
    /// Every defined ELF symbol, keyed by name.
    ///
    /// The freeze coordinator needs this complete table for arbitrary
    /// `Op::WatchSnapshot` names and for guest-memory accessor bootstrap.
    /// Keeping it in the same derived artifact as [`Self::symbols`] and BTF
    /// offsets means cross-process sidecar hits never re-read or reparse the
    /// full vmlinux just to rebuild a coordinator-local HashMap.
    pub all_symbols: Arc<std::collections::HashMap<String, u64>>,
    /// BTF-derived products the monitor thread needs. `None` when BTF
    /// load or `KernelOffsets` resolution failed — `start_monitor`
    /// then returns no monitor thread, exactly as its pre-cache
    /// per-consumer parse did, while the inline path still gets
    /// `symbols`.
    pub monitor: Option<MonitorArtifacts>,
    /// Guest `CONFIG_HZ` scanned from the embedded IKCONFIG blob, or
    /// `None` when the vmlinux carries no IKCONFIG (CONFIG_IKCONFIG off
    /// or a stripped image). Derived from the same bytes as `symbols`
    /// so `guest_kernel_hz` reads it off the shared parse (and off the
    /// `.artifacts` sidecar on a cross-process hit) instead of a second
    /// full `std::fs::read` + IKCONFIG scan of the ELF.
    pub guest_hz: Option<u64>,
    /// This vmlinux's GNU build-id (`.note.gnu.build-id`), or `None` when
    /// the note is absent. Compared host-side against the booted kernel's
    /// build-id (published by the guest via `MSG_TYPE_KERN_BUILD_ID`) to
    /// catch a stale/mismatched cache entry — a vmlinux whose layout or
    /// symbol addresses differ from the running Image — before its
    /// offsets silently mis-read guest memory. A pure function of the ELF
    /// bytes, so it rides the same parse + sidecar as `symbols`.
    pub build_id: Option<Vec<u8>>,
}

/// Immutable vmlinux inputs prepared before a VM acquires host CPU/LLC
/// admission or allocates guest memory.
///
/// A cold [`cached_vmlinux_artifacts`] call may wait behind the one
/// cross-process sidecar builder for this kernel. Carrying both the resolved
/// path and derived products through the run keeps that wait out of the
/// admitted VM lifetime and lets every later consumer use the exact same
/// artifact without another path lookup or cache rendezvous.
pub(crate) struct PreparedVmlinux {
    pub(crate) path: PathBuf,
    pub(crate) artifacts: Arc<VmlinuxArtifacts>,
}

/// Resolve and derive the immutable vmlinux products needed by one VM run.
///
/// `None` preserves the existing graceful degradation when no matching
/// vmlinux exists or its ELF/symbol products cannot be read.
pub(crate) fn prepare_vmlinux(kernel_path: &Path) -> Option<PreparedVmlinux> {
    let path = find_vmlinux(kernel_path)?;
    let artifacts = cached_vmlinux_artifacts(&path)?;
    Some(PreparedVmlinux { path, artifacts })
}

/// Extract the GNU build-id (`NT_GNU_BUILD_ID`, owner `"GNU"`) from a
/// parsed vmlinux ELF's note headers. `None` when the note is absent
/// (a build without `--build-id`) or unreadable.
fn extract_vmlinux_build_id(elf: &goblin::elf::Elf, data: &[u8]) -> Option<Vec<u8>> {
    const NT_GNU_BUILD_ID: u32 = 3;
    let notes = elf.iter_note_headers(data)?;
    for note in notes.flatten() {
        if note.n_type == NT_GNU_BUILD_ID && note.name == "GNU" {
            // Truncate to the same cap the guest applies to its published
            // build-id ([`crate::vmm::wire::KERN_BUILD_ID_MAX`]) so a
            // build-id longer than the cap compares equal on both sides
            // instead of spuriously mismatching.
            let end = note.desc.len().min(crate::vmm::wire::KERN_BUILD_ID_MAX);
            return Some(note.desc[..end].to_vec());
        }
    }
    None
}

/// The monitor-only subset of [`VmlinuxArtifacts`]: the BTF-backed
/// offset tables plus the shared `Btf` handle.
pub(crate) struct MonitorArtifacts {
    pub offsets: crate::monitor::btf_offsets::KernelOffsets,
    pub prog_offsets: Option<crate::monitor::btf_offsets::BpfProgOffsets>,
    pub psi_offsets: Option<crate::monitor::btf_offsets::PsiGroupOffsets>,
    pub btf: Arc<btf_rs::Btf>,
}

/// Parse the vmlinux bytes into [`VmlinuxArtifacts`], deriving the ELF
/// symbols once and the BTF-backed offsets once. `None` when the ELF
/// parse or mandatory-symbol resolution fails (both consumers degrade);
/// a BTF/offset failure leaves `symbols` intact with `monitor: None`.
fn parse_vmlinux_artifacts(data: &[u8], path: &Path) -> Option<VmlinuxArtifacts> {
    let elf = goblin::elf::Elf::parse(data).ok()?;
    let symbols = crate::monitor::symbols::KernelSymbols::from_elf(&elf).ok()?;
    const SHN_UNDEF: usize = 0;
    let mut all_symbols = std::collections::HashMap::new();
    for symbol in elf.syms.iter() {
        if symbol.st_shndx == SHN_UNDEF {
            continue;
        }
        if let Some(name) = elf.strtab.get_at(symbol.st_name) {
            // Preserve the old `VmlinuxSymbolCache::from_path` semantics for
            // duplicate names: the last defined symbol wins.
            all_symbols.insert(name.to_string(), symbol.st_value);
        }
    }
    // BTF-backed products fail independently of `symbols`: a closure so
    // any early `?` yields `monitor: None` without dropping `symbols`.
    let monitor = (|| {
        let btf = crate::monitor::btf_offsets::load_btf_from_elf(&elf, data, path).ok()?;
        let offsets = crate::monitor::btf_offsets::KernelOffsets::from_btf(&btf).ok()?;
        let prog_offsets = crate::monitor::btf_offsets::BpfProgOffsets::from_btf(&btf).ok();
        let psi_offsets = crate::monitor::btf_offsets::PsiGroupOffsets::from_btf(&btf).ok();
        Some(MonitorArtifacts {
            offsets,
            prog_offsets,
            psi_offsets,
            btf: Arc::new(btf),
        })
    })();
    // Scan IKCONFIG for CONFIG_HZ off the same bytes. Independent of
    // the symbol/BTF parse: a vmlinux can carry IKCONFIG without the
    // scheduler symbols and vice versa, so a failed scan just yields
    // `None` (guest_kernel_hz then falls back to its .config paths).
    let guest_hz = crate::monitor::hz_from_vmlinux_bytes(data);
    let build_id = extract_vmlinux_build_id(&elf, data);
    Some(VmlinuxArtifacts {
        symbols,
        all_symbols: Arc::new(all_symbols),
        monitor,
        guest_hz,
        build_id,
    })
}

/// Magic + layout-version tag stamped as the first field of every
/// `<vmlinux>.artifacts` sidecar.
///
/// The magic disambiguates the file from any other postcard blob; the
/// embedded `CARGO_PKG_VERSION` gates the LAYOUT of the mirrored offset
/// structs. postcard is not self-describing, so a ktstr build whose
/// offset-struct fields changed shape would mis-decode an older
/// sidecar's later fields. Pinning the tag to the crate version means
/// any such build stamps a new tag, the version check on load rejects
/// the stale sidecar, and the artifacts are re-derived + rewritten.
/// (Sidecar content is a pure function of the vmlinux bytes, so the
/// mtime freshness rule covers vmlinux changes; only ktstr-version
/// layout drift needs this tag.)
const ARTIFACTS_SIDECAR_VERSION: &str =
    concat!("ktstr-vmlinux-artifacts-v4 ", env!("CARGO_PKG_VERSION"));

/// Plain-old-data mirror of the derived half of [`VmlinuxArtifacts`],
/// serialized to the `<vmlinux>.artifacts` sidecar via postcard.
///
/// Deliberately a parallel struct rather than serde on
/// `VmlinuxArtifacts` itself: the live type carries an `Arc<Btf>`
/// ([`MonitorArtifacts::btf`]) that is not serializable — it is
/// reconstructed from the paired `.btf` sidecar on load. `offsets`
/// being `Some` is the "monitor was present" marker, mirroring
/// `VmlinuxArtifacts.monitor.is_some()`; `prog_offsets` / `psi_offsets`
/// are the monitor's optional sub-tables.
#[derive(serde::Serialize, serde::Deserialize)]
struct ArtifactsSidecar {
    /// Magic + ktstr-version tag ([`ARTIFACTS_SIDECAR_VERSION`]). First
    /// field so a version mismatch is caught before any later field is
    /// trusted.
    version: String,
    symbols: crate::monitor::symbols::KernelSymbols,
    /// Sorted mirror of [`VmlinuxArtifacts::all_symbols`]. A Vec gives the
    /// sidecar deterministic bytes; serializing a HashMap directly would
    /// reshuffle entries after every decode because its RandomState changes.
    /// The live artifact rebuilds the O(1) lookup map once on load.
    all_symbols: Vec<(String, u64)>,
    /// `Some` iff the original parse produced a [`MonitorArtifacts`]
    /// (BTF loaded and `KernelOffsets` resolved).
    offsets: Option<crate::monitor::btf_offsets::KernelOffsets>,
    prog_offsets: Option<crate::monitor::btf_offsets::BpfProgOffsets>,
    psi_offsets: Option<crate::monitor::btf_offsets::PsiGroupOffsets>,
    guest_hz: Option<u64>,
    /// Mirror of [`VmlinuxArtifacts::build_id`].
    build_id: Option<Vec<u8>>,
}

/// Borrowed serialization mirror of [`ArtifactsSidecar`].
///
/// Its field order and serialized shapes intentionally match the owned decode
/// type. Avoiding a cloned complete symbol map while writing matters on the
/// one cold-builder path: that map can contain hundreds of thousands of
/// owned names, and the whole point of the sidecar is to collapse peak work
/// and memory across a verifier storm.
#[derive(serde::Serialize)]
struct ArtifactsSidecarRef<'a> {
    version: &'a str,
    symbols: &'a crate::monitor::symbols::KernelSymbols,
    all_symbols: Vec<(&'a String, &'a u64)>,
    offsets: Option<&'a crate::monitor::btf_offsets::KernelOffsets>,
    prog_offsets: Option<&'a crate::monitor::btf_offsets::BpfProgOffsets>,
    psi_offsets: Option<&'a crate::monitor::btf_offsets::PsiGroupOffsets>,
    guest_hz: &'a Option<u64>,
    build_id: &'a Option<Vec<u8>>,
}

/// Sidecar path for a vmlinux: append `.artifacts` to the filename so
/// it sits next to vmlinux (and next to the `.btf` sidecar) in the same
/// cache-entry directory. Append-suffix (not `with_extension`) mirrors
/// the `.btf` sidecar and preserves any existing extension.
fn artifacts_sidecar_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_os_string();
    name.push(".artifacts");
    PathBuf::from(name)
}

/// Cross-process single-builder lock for [`artifacts_sidecar_path`].
///
/// The vmlinux cache is shared by many nextest processes. Atomic sidecar
/// writes prevent corruption, but without a builder lock every process that
/// observes the same cold miss independently reads and parses the full ELF.
/// Holding this flock across the post-lock recheck and derivation collapses
/// that storm to one builder; waiters reconstruct the same derived artifact.
fn artifacts_sidecar_lock_path(path: &Path) -> PathBuf {
    let mut name = artifacts_sidecar_path(path).into_os_string();
    name.push(".lock");
    PathBuf::from(name)
}

/// Atomically write `bytes` to `dest` via a tempfile in the same
/// directory + rename, so a concurrent reader sees either the old file
/// or the new one, never a partial write. Mirrors the `.btf` sidecar's
/// write path.
fn atomic_write_sidecar(dest: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    let parent = dest.parent().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "artifacts sidecar path has no parent directory",
        )
    })?;
    let mut tmp = tempfile::NamedTempFile::new_in(parent)?;
    tmp.write_all(bytes)?;
    tmp.as_file().sync_all()?;
    tmp.persist(dest).map_err(|e| e.error)?;
    Ok(())
}

/// Try to assemble [`VmlinuxArtifacts`] from the `<vmlinux>.artifacts`
/// sidecar WITHOUT reading or parsing the multi-hundred-MB vmlinux ELF.
///
/// `canon` is the canonicalized vmlinux path (the shared cache key),
/// used both for the sidecar path and — when the sidecar carries
/// monitor offsets — to reconstruct the `Arc<Btf>` from the paired
/// `.btf` sidecar.
///
/// Returns `None` (a miss the caller resolves with a full parse) on:
/// cache-root membership or freshness failure, read/decode failure, a
/// version-tag mismatch (silently re-derived per the version gate), OR
/// a monitor-present sidecar whose paired `.btf` sidecar cannot rebuild
/// the BTF (the offsets are unusable without it). A monitor-absent
/// sidecar needs no BTF and assembles directly.
fn load_artifacts_sidecar(canon: &Path) -> Option<VmlinuxArtifacts> {
    // Same gate as the writer: source trees / distro debug paths are
    // never trusted for a sidecar they would not have written.
    if !crate::cache::path_inside_cache_root(canon) {
        return None;
    }
    let sidecar = artifacts_sidecar_path(canon);
    if !crate::monitor::btf_offsets::sidecar_fresh(&sidecar, canon) {
        return None;
    }
    let bytes = std::fs::read(&sidecar).ok()?;
    let decoded: ArtifactsSidecar = postcard::from_bytes(&bytes).ok()?;
    // Version gate: a tag mismatch means the sidecar was written by a
    // ktstr build with a different offset-struct layout (or is a
    // foreign blob). Ignore and let the caller re-derive + rewrite.
    if decoded.version != ARTIFACTS_SIDECAR_VERSION {
        return None;
    }
    let ArtifactsSidecar {
        version: _,
        symbols,
        all_symbols,
        offsets,
        prog_offsets,
        psi_offsets,
        guest_hz,
        build_id,
    } = decoded;
    let monitor = match offsets {
        Some(offsets) => {
            // Monitor was present at parse time — rebuild the shared
            // `Arc<Btf>` from the paired `.btf` sidecar. Its absence
            // makes the stored offsets unusable, so treat the whole
            // load as a miss and fall back to a full parse (which
            // rewrites both sidecars).
            let btf = crate::monitor::btf_offsets::load_btf_from_sidecar(canon)?;
            Some(MonitorArtifacts {
                offsets,
                prog_offsets,
                psi_offsets,
                btf: Arc::new(btf),
            })
        }
        None => None,
    };
    Some(VmlinuxArtifacts {
        symbols,
        all_symbols: Arc::new(all_symbols.into_iter().collect()),
        monitor,
        guest_hz,
        build_id,
    })
}

/// Best-effort write of the `<vmlinux>.artifacts` sidecar from a
/// freshly-parsed `artifacts`. Failures are swallowed (logged): the
/// parse already succeeded; we just miss the cross-process cache on the
/// next load.
fn write_artifacts_sidecar(sidecar: &Path, artifacts: &VmlinuxArtifacts) {
    let mut all_symbols: Vec<_> = artifacts.all_symbols.iter().collect();
    all_symbols.sort_unstable_by(|(left, _), (right, _)| left.cmp(right));
    let payload = ArtifactsSidecarRef {
        version: ARTIFACTS_SIDECAR_VERSION,
        symbols: &artifacts.symbols,
        all_symbols,
        offsets: artifacts.monitor.as_ref().map(|m| &m.offsets),
        prog_offsets: artifacts
            .monitor
            .as_ref()
            .and_then(|m| m.prog_offsets.as_ref()),
        psi_offsets: artifacts
            .monitor
            .as_ref()
            .and_then(|m| m.psi_offsets.as_ref()),
        guest_hz: &artifacts.guest_hz,
        build_id: &artifacts.build_id,
    };
    let bytes = match postcard::to_allocvec(&payload) {
        Ok(b) => b,
        Err(e) => {
            tracing::warn!(err = %e, "vmlinux .artifacts sidecar encode failed");
            return;
        }
    };
    if let Err(e) = atomic_write_sidecar(sidecar, &bytes) {
        tracing::warn!(
            path = %sidecar.display(),
            err = %e,
            "vmlinux .artifacts sidecar write failed; artifacts re-derived on next load",
        );
    }
}

/// Return the cached parsed products for `path`, deriving them once on
/// the first request for a given (canonical path, mtime) and cloning
/// the `Arc` on every subsequent hit.
///
/// Two cache tiers back this:
///
/// - **L1** — the process-global [`VMLINUX_ARTIFACTS_CACHE`] map. Keying,
///   mtime invalidation, and the not-cached error path mirror
///   [`cached_vmlinux_bytes`].
/// - **L2** — the cross-process `<vmlinux>.artifacts` sidecar. nextest
///   runs each CI cell in its own process, so without L2 every cell
///   pays a full `fs::read` + goblin ELF parse + BTF parse of a
///   50-340 MB debug vmlinux. On an L1 miss we first try the sidecar,
///   which assembles the artifacts from a small postcard blob (+ the
///   `.btf` sidecar for the `Arc<Btf>`) WITHOUT touching the ELF. Only a
///   sidecar miss falls through to the full parse, which then best-effort
///   writes the sidecar for sibling processes.
///
/// `None` when the file is unreadable or the ELF/symbol parse fails.
pub(crate) fn cached_vmlinux_artifacts(path: &Path) -> Option<Arc<VmlinuxArtifacts>> {
    let canon = std::fs::canonicalize(path)
        .ok()
        .unwrap_or_else(|| path.to_path_buf());
    let mtime = std::fs::metadata(&canon).and_then(|m| m.modified()).ok()?;
    let slot =
        VMLINUX_ARTIFACTS_CACHE.get_or_init(|| RwLock::new(std::collections::HashMap::new()));
    {
        let read = slot.read_unpoisoned();
        if let Some(entry) = read.get(&canon)
            && entry.mtime == mtime
        {
            return Some(Arc::clone(&entry.artifacts));
        }
    }
    // L2: cross-process sidecar. Assembles the artifacts without reading
    // the ELF; on hit, promote into L1 and return.
    if let Some(artifacts) = load_artifacts_sidecar(&canon) {
        let arc = Arc::new(artifacts);
        slot.write_unpoisoned().insert(
            canon,
            CachedArtifacts {
                mtime,
                artifacts: Arc::clone(&arc),
            },
        );
        return Some(arc);
    }
    // A cold shared-cache miss must have exactly one parser. Atomic rename
    // protects readers from partial bytes but does not prevent every nextest
    // process from doing the same multi-hundred-MB read + ELF/BTF parse before
    // racing to that rename. Serialize only builders of this exact derived
    // artifact, then recheck both cache tiers after the flock lands.
    let _builder_lock = if crate::cache::path_inside_cache_root(&canon) {
        let lock_path = artifacts_sidecar_lock_path(&canon);
        match crate::flock::block_flock(&lock_path, crate::flock::FlockMode::Exclusive) {
            Ok(lock) => Some(lock),
            Err(error) => {
                tracing::warn!(
                    path = %lock_path.display(),
                    %error,
                    "vmlinux artifacts single-builder flock unavailable; \
                     deriving without cross-process deduplication"
                );
                None
            }
        }
    } else {
        None
    };
    if _builder_lock.is_some() {
        {
            let read = slot.read_unpoisoned();
            if let Some(entry) = read.get(&canon)
                && entry.mtime == mtime
            {
                return Some(Arc::clone(&entry.artifacts));
            }
        }
        if let Some(artifacts) = load_artifacts_sidecar(&canon) {
            let arc = Arc::new(artifacts);
            slot.write_unpoisoned().insert(
                canon,
                CachedArtifacts {
                    mtime,
                    artifacts: Arc::clone(&arc),
                },
            );
            return Some(arc);
        }
    }
    // Miss: full parse outside the write lock so a slow parse doesn't
    // block other canonical paths' lookups. The byte read is served by
    // `cached_vmlinux_bytes` (its own cache + mtime gate).
    let data = cached_vmlinux_bytes(&canon)?;
    let artifacts = Arc::new(parse_vmlinux_artifacts(&data, &canon)?);
    // Best-effort write the sidecar for sibling processes, gated on
    // cache-root membership (never pollute source trees / distro paths).
    if crate::cache::path_inside_cache_root(&canon) {
        // Pair constraint: a monitor-present sidecar is only usable if a
        // fresh `.btf` sidecar can reconstruct its `Arc<Btf>` on load.
        // `parse_vmlinux_artifacts`' `load_btf_from_elf` already wrote
        // that `.btf` sidecar (inside the cache) on a successful BTF
        // load, so this normally holds; the check guards the case where
        // that write failed — skip the `.artifacts` write rather than
        // strand offsets that can never be rehydrated. A monitor-absent
        // parse needs no BTF and always writes.
        let btf_ok = artifacts.monitor.is_none()
            || crate::monitor::btf_offsets::btf_sidecar_fresh_for(&canon);
        if btf_ok {
            write_artifacts_sidecar(&artifacts_sidecar_path(&canon), &artifacts);
        }
    }
    slot.write_unpoisoned().insert(
        canon,
        CachedArtifacts {
            mtime,
            artifacts: Arc::clone(&artifacts),
        },
    );
    Some(artifacts)
}

/// Clear every cached entry. Used by `#[cfg(test)]` tests that need
/// to assert against a clean cache state without inheriting entries
/// from prior tests in the same process — a regular use case for
/// invalidation-coverage tests where we want to compare cache-miss
/// vs cache-hit behaviour deterministically.
#[cfg(test)]
pub(crate) fn clear_vmlinux_cache_for_tests() {
    if let Some(slot) = VMLINUX_BYTES_CACHE.get() {
        slot.write_unpoisoned().clear();
    }
    if let Some(slot) = VMLINUX_ARTIFACTS_CACHE.get() {
        slot.write_unpoisoned().clear();
    }
}

/// Find the vmlinux ELF next to a kernel image path.
///
/// Shared across x86_64 and aarch64. Both architectures follow the
/// kernel build's `<root>/arch/<arch>/boot/<image>` layout, so
/// stepping 3 directories up from `kernel_path` lands on `<root>`
/// where `vmlinux` sits. Distro paths diverge: x86_64 ships debug
/// vmlinux at `/usr/lib/debug/boot/vmlinux-<version>`, aarch64 splits
/// between `/boot/vmlinux-<version>` and
/// `/lib/modules/<version>/build/vmlinux`. Both distro layouts are
/// probed regardless of arch — the arch-specific filename prefix
/// (`bzImage` vs `Image`) only tells us where to look, not which
/// layout owns the match.
pub(crate) fn find_vmlinux(kernel_path: &Path) -> Option<PathBuf> {
    let dir = kernel_path.parent()?;
    let candidate = dir.join("vmlinux");
    if candidate.exists() {
        return Some(candidate);
    }
    // Kernel build tree: <root>/arch/<arch>/boot/<image> -> <root>/vmlinux.
    if let Ok(root) = dir.join("../../..").canonicalize() {
        let candidate = root.join("vmlinux");
        if candidate.exists() {
            return Some(candidate);
        }
    }
    // Distro layouts keyed by the image's version suffix
    // (`vmlinuz-<version>`).
    if let Some(name) = kernel_path.file_name().and_then(|n| n.to_str()) {
        let version = name.strip_prefix("vmlinuz-").unwrap_or(name);
        for candidate in [
            PathBuf::from(format!("/usr/lib/debug/boot/vmlinux-{version}")),
            PathBuf::from(format!("/boot/vmlinux-{version}")),
            PathBuf::from(format!("/lib/modules/{version}/build/vmlinux")),
        ] {
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }
    // `/lib/modules/<version>/vmlinuz` layout: version is the parent
    // directory name, and the sibling `build/vmlinux` is the target.
    if let Some(parent_name) = dir.file_name().and_then(|n| n.to_str()) {
        for candidate in [
            dir.join("build/vmlinux"),
            PathBuf::from(format!("/boot/vmlinux-{parent_name}")),
        ] {
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn find_vmlinux_from_bzimage_path() {
        // Create a temp dir simulating <root>/arch/x86/boot/bzImage with vmlinux at <root>.
        let tmp = tempfile::TempDir::new().unwrap();
        let boot_dir = tmp.path().join("arch/x86/boot");
        std::fs::create_dir_all(&boot_dir).unwrap();
        let vmlinux = tmp.path().join("vmlinux");
        std::fs::write(&vmlinux, b"ELF").unwrap();
        let bzimage = boot_dir.join("bzImage");
        std::fs::write(&bzimage, b"kernel").unwrap();

        let found = find_vmlinux(&bzimage);
        assert_eq!(found, Some(vmlinux));
    }

    #[test]
    fn find_vmlinux_sibling() {
        // vmlinux in the same directory as the kernel image.
        let tmp = tempfile::TempDir::new().unwrap();
        let vmlinux = tmp.path().join("vmlinux");
        std::fs::write(&vmlinux, b"ELF").unwrap();
        let kernel = tmp.path().join("bzImage");
        std::fs::write(&kernel, b"kernel").unwrap();

        let found = find_vmlinux(&kernel);
        assert_eq!(found, Some(vmlinux));
    }

    #[test]
    fn find_vmlinux_bare_filename() {
        // A bare filename — parent is "" so no vmlinux sibling found.
        assert_eq!(find_vmlinux(Path::new("vmlinuz")), None);
    }

    #[test]
    fn find_vmlinux_root_parent() {
        // /vmlinuz has parent "/" — no vmlinux there (or if there is, fine).
        // The function should not panic.
        let result = find_vmlinux(Path::new("/vmlinuz"));
        // /vmlinux almost certainly doesn't exist; if it does, that's still valid.
        if !Path::new("/vmlinux").exists() {
            assert_eq!(result, None);
        }
    }

    #[test]
    fn find_vmlinux_missing_returns_none() {
        let tmp = tempfile::TempDir::new().unwrap();
        let kernel = tmp.path().join("bzImage");
        std::fs::write(&kernel, b"kernel").unwrap();

        assert_eq!(find_vmlinux(&kernel), None);
    }

    /// First call reads from disk; second call returns a clone of the
    /// cached `Arc<Vec<u8>>`, proving the cache hit path does not re-
    /// read. `Arc::ptr_eq` is the load-bearing assertion: the bytes
    /// would compare equal even from a re-read, but only the cache
    /// hit returns the same allocation.
    #[test]
    fn cached_vmlinux_bytes_hits_on_second_call() {
        let tmp = tempfile::TempDir::new().unwrap();
        let vmlinux = tmp.path().join("vmlinux-test-cache");
        std::fs::write(&vmlinux, b"FAKE_VMLINUX_BYTES").unwrap();

        let first = cached_vmlinux_bytes(&vmlinux).expect("first read populates cache");
        let second = cached_vmlinux_bytes(&vmlinux).expect("second read hits cache");
        assert_eq!(first.as_slice(), b"FAKE_VMLINUX_BYTES");
        assert!(
            Arc::ptr_eq(&first, &second),
            "cache hit must return the same Arc; got fresh allocations on each call"
        );
    }

    /// Unreadable path returns `None` without populating the cache;
    /// a subsequent successful path is unaffected.
    #[test]
    fn cached_vmlinux_bytes_missing_returns_none() {
        let tmp = tempfile::TempDir::new().unwrap();
        let nonexistent = tmp.path().join("missing-xyzzy");
        assert!(cached_vmlinux_bytes(&nonexistent).is_none());
    }

    /// Two distinct symlink paths pointing at the SAME real file must
    /// dedup to one cache entry — canonicalize collapses both keys to
    /// the same canonical PathBuf, so the second lookup hits the cache
    /// populated by the first and returns a clone of the same `Arc`.
    /// Verified via `Arc::ptr_eq` rather than byte equality: the bytes
    /// would compare equal even from a re-read; only a true cache hit
    /// returns the same allocation.
    #[test]
    #[cfg(unix)]
    fn cached_vmlinux_bytes_dedups_symlinks_to_same_target() {
        let tmp = tempfile::TempDir::new().unwrap();
        let real = tmp.path().join("vmlinux-real");
        std::fs::write(&real, b"SYMLINK_DEDUP_BYTES").unwrap();
        let link_a = tmp.path().join("vmlinux-link-a");
        let link_b = tmp.path().join("vmlinux-link-b");
        std::os::unix::fs::symlink(&real, &link_a).unwrap();
        std::os::unix::fs::symlink(&real, &link_b).unwrap();

        let via_a = cached_vmlinux_bytes(&link_a).expect("read via symlink A");
        let via_b = cached_vmlinux_bytes(&link_b).expect("read via symlink B");
        assert!(
            Arc::ptr_eq(&via_a, &via_b),
            "two symlinks to the same target must canonicalize to the \
             same cache key and return the same Arc; got fresh \
             allocations, suggesting the canonicalize-then-key path \
             regressed to keying on the raw symlink path."
        );
    }

    /// A dangling symlink (target deleted before any read) makes
    /// `canonicalize` fail. The function falls back to using the
    /// symlink's own path as the cache key, then `fs::read` fails to
    /// open the dangling target and returns `None`. The cache is not
    /// populated for the dangling path.
    #[test]
    #[cfg(unix)]
    fn cached_vmlinux_bytes_dangling_symlink_returns_none() {
        let tmp = tempfile::TempDir::new().unwrap();
        let target = tmp.path().join("vmlinux-gone");
        let link = tmp.path().join("vmlinux-dangling");
        std::fs::write(&target, b"ELF").unwrap();
        std::os::unix::fs::symlink(&target, &link).unwrap();
        std::fs::remove_file(&target).unwrap();

        assert!(cached_vmlinux_bytes(&link).is_none());
    }

    /// Rewriting the file with new bytes between two lookups must
    /// invalidate the cache and surface the new bytes on the
    /// second lookup. Catches the "stale cached bytes after a
    /// rebuild" regression that the pre-mtime version had: a
    /// developer who rebuilds vmlinux while a long-lived test
    /// process is running would get the stale bytes forever
    /// without this invalidation. Verifies via NON-`Arc::ptr_eq`
    /// (the new bytes must be in a fresh allocation) plus a byte-
    /// content comparison (the new content actually reached the
    /// reader). Bumps mtime explicitly via `libc::utimes` (rather
    /// than sleeping for FS-granularity) so the test runs in
    /// microseconds and survives FS variants with 1-second mtime
    /// resolution.
    #[test]
    #[cfg(unix)]
    fn cached_vmlinux_bytes_invalidates_on_mtime_change() {
        let tmp = tempfile::TempDir::new().unwrap();
        let vmlinux = tmp.path().join("vmlinux-mtime-test");
        std::fs::write(&vmlinux, b"FIRST_BYTES").unwrap();
        clear_vmlinux_cache_for_tests();

        let first = cached_vmlinux_bytes(&vmlinux).expect("first read");
        assert_eq!(first.as_slice(), b"FIRST_BYTES");

        // Rewrite with new content, then bump mtime to a sentinel
        // value far in the past via libc::utimes so the captured
        // mtime is guaranteed != the cached one regardless of FS
        // mtime resolution. Setting both atime and mtime to
        // 1970-01-02T00:00:00 (86400 sec since epoch) makes the
        // pre-write mtime (now-ish) vs post-utimes mtime
        // (1970-01-02) trivially distinct.
        std::fs::write(&vmlinux, b"SECOND_BYTES_DIFFERENT").unwrap();
        let path_c = std::ffi::CString::new(vmlinux.as_os_str().as_encoded_bytes()).unwrap();
        let sentinel = libc::timeval {
            tv_sec: 86_400,
            tv_usec: 0,
        };
        let times = [sentinel, sentinel];
        // SAFETY: path_c is a valid NUL-terminated path; times is a
        // 2-element timeval array (atime, mtime) as utimes(2) requires.
        let rc = unsafe { libc::utimes(path_c.as_ptr(), times.as_ptr()) };
        assert_eq!(rc, 0, "libc::utimes must succeed on the temp file");

        let second = cached_vmlinux_bytes(&vmlinux).expect("second read");
        assert_eq!(
            second.as_slice(),
            b"SECOND_BYTES_DIFFERENT",
            "mtime change must invalidate cache and surface the rewritten bytes"
        );
        assert!(
            !Arc::ptr_eq(&first, &second),
            "post-rewrite second lookup must return a fresh Arc, \
             not the stale cached one — Arc::ptr_eq returning true \
             means the invalidation path didn't fire."
        );
    }

    // -- `<vmlinux>.artifacts` sidecar --
    //
    // These exercise the cross-process sidecar (encode/decode, the
    // version gate, and the mtime freshness rule) directly with
    // synthetic offset structs — the repo ships no vmlinux fixture with
    // BTF to drive the full parse path, and the monitor-present Arc<Btf>
    // reconstruction needs a real `.btf` sidecar. The monitor-absent
    // path (`monitor: None`) rehydrates without any BTF, so it round-
    // trips through the real filesystem helpers here.

    fn synthetic_symbols() -> crate::monitor::symbols::KernelSymbols {
        crate::monitor::symbols::KernelSymbols {
            runqueues: 0x1000,
            per_cpu_offset: 0x2000,
            page_offset_base_kva: Some(0x3000),
            memstart_addr_kva: None,
            phys_base_kva: None,
            scx_root: Some(0x4000),
            scx_tasks: None,
            init_top_pgt: Some(0x5000),
            pgtable_l5_enabled: None,
            prog_idr: Some(0x6000),
            scx_watchdog_timeout: None,
            scx_watchdog_timestamp: Some(0x7000),
            scx_watchdog_interval: None,
            jiffies_64: Some(0x8000),
            entry_syscall_64_kva: None,
            kernel_text_kva: Some(0x9000),
            kernel_cpustat: None,
            kstat: Some(0xa000),
            tick_cpu_sched: None,
            node_data: Some(0xb000),
            psi_system: Some(0xc000),
            cgrp_dfl_root: None,
        }
    }

    fn synthetic_all_symbols() -> Arc<std::collections::HashMap<String, u64>> {
        Arc::new(std::collections::HashMap::from([
            ("runqueues".to_string(), 0x1000),
            ("__per_cpu_offset".to_string(), 0x2000),
            ("arbitrary_watch_target".to_string(), 0xd000),
        ]))
    }

    fn synthetic_all_symbols_sidecar() -> Vec<(String, u64)> {
        let mut symbols: Vec<_> = synthetic_all_symbols()
            .iter()
            .map(|(name, kva)| (name.clone(), *kva))
            .collect();
        symbols.sort_unstable_by(|(left, _), (right, _)| left.cmp(right));
        symbols
    }

    fn synthetic_kernel_offsets() -> crate::monitor::btf_offsets::KernelOffsets {
        use crate::monitor::btf_offsets::{
            KernelOffsets, SchedstatOffsets, ScxEventOffsets, ScxWatchdogOffsets,
        };
        use crate::monitor::btf_offsets::{SchedDomainOffsets, SchedDomainStatsOffsets};
        KernelOffsets {
            rq_cpu: 1,
            rq_nr_running: 2,
            rq_clock: 3,
            rq_scx: 4,
            scx_rq_nr_running: 5,
            scx_rq_local_dsq: 6,
            scx_rq_flags: 7,
            dsq_nr: 8,
            rq_avg_irq_util_avg: Some(9),
            event_offsets: Some(ScxEventOffsets {
                percpu_ptr_off: 10,
                event_stats_off: 11,
                ev_select_cpu_fallback: 12,
                ev_dispatch_local_dsq_offline: 13,
                ev_dispatch_keep_last: 14,
                ev_enq_skip_exiting: 15,
                ev_enq_skip_migration_disabled: 16,
                ev_reenq_immed: Some(17),
                ev_reenq_local_repeat: None,
                ev_refill_slice_dfl: Some(18),
                ev_bypass_duration: None,
                ev_bypass_dispatch: Some(19),
                ev_bypass_activate: None,
                ev_insert_not_owned: Some(20),
                ev_sub_bypass_dispatch: None,
            }),
            schedstat_offsets: Some(SchedstatOffsets {
                rq_sched_info: 21,
                sched_info_run_delay: 22,
                sched_info_pcount: 23,
                rq_yld_count: 24,
                rq_sched_count: 25,
                rq_sched_goidle: 26,
                rq_ttwu_count: 27,
                rq_ttwu_local: 28,
            }),
            sched_domain_offsets: Some(SchedDomainOffsets {
                rq_sd: 29,
                sd_parent: 30,
                sd_level: 31,
                sd_name: 32,
                sd_flags: 33,
                sd_span_weight: 34,
                sd_balance_interval: 35,
                sd_nr_balance_failed: 36,
                sd_newidle_call: Some(37),
                sd_newidle_success: Some(38),
                sd_newidle_ratio: None,
                sd_max_newidle_lb_cost: 39,
                stats_offsets: Some(SchedDomainStatsOffsets {
                    sd_lb_count: 40,
                    sd_lb_failed: 41,
                    sd_lb_balanced: 42,
                    sd_lb_imbalance_load: 43,
                    sd_lb_imbalance_util: 44,
                    sd_lb_imbalance_task: 45,
                    sd_lb_imbalance_misfit: 46,
                    sd_lb_gained: 47,
                    sd_lb_hot_gained: 48,
                    sd_lb_nobusyg: 49,
                    sd_lb_nobusyq: 50,
                    sd_alb_count: 51,
                    sd_alb_failed: 52,
                    sd_alb_pushed: 53,
                    sd_sbe_count: 54,
                    sd_sbe_balanced: 55,
                    sd_sbe_pushed: 56,
                    sd_sbf_count: 57,
                    sd_sbf_balanced: 58,
                    sd_sbf_pushed: 59,
                    sd_ttwu_wake_remote: 60,
                    sd_ttwu_move_affine: 61,
                    sd_ttwu_move_balance: 62,
                }),
            }),
            watchdog_offsets: Some(ScxWatchdogOffsets {
                scx_sched_watchdog_timeout_off: 63,
            }),
        }
    }

    fn synthetic_prog_offsets() -> crate::monitor::btf_offsets::BpfProgOffsets {
        crate::monitor::btf_offsets::BpfProgOffsets {
            prog_type: 1,
            prog_aux: 2,
            aux_verified_insns: 3,
            aux_name: 4,
            aux_used_maps: 5,
            aux_used_map_cnt: 6,
            xa_node_slots: 7,
            xa_node_shift: 8,
            idr_xa_head: 9,
            idr_next: 10,
            prog_stats: 11,
            stats_cnt: 12,
            stats_nsecs: 13,
            stats_misses: 14,
        }
    }

    fn synthetic_psi_offsets() -> crate::monitor::btf_offsets::PsiGroupOffsets {
        crate::monitor::btf_offsets::PsiGroupOffsets {
            psi_group_total: 100,
            psi_group_avg: 200,
            psi_irq_full_idx: Some(6),
        }
    }

    /// Preparing a VM resolves the sibling vmlinux and promotes its
    /// cross-process sidecar into the process cache before admission. A
    /// second preparation must reuse the exact same immutable artifact
    /// allocation instead of parsing (or even decoding) it again.
    #[test]
    fn prepare_vmlinux_resolves_sidecar_and_reuses_artifacts() {
        use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
        let _lock = lock_env();
        let tmp = tempfile::TempDir::new().unwrap();
        let _guard = EnvVarGuard::set(crate::KTSTR_CACHE_DIR_ENV, tmp.path());
        let entry = tmp.path().join("kentry");
        std::fs::create_dir_all(&entry).unwrap();
        let kernel = entry.join("bzImage");
        std::fs::write(&kernel, b"fake-kernel").unwrap();
        let vmlinux = entry.join("vmlinux");
        std::fs::write(&vmlinux, b"not-an-elf").unwrap();
        let canon = std::fs::canonicalize(&vmlinux).unwrap();

        let original = VmlinuxArtifacts {
            symbols: synthetic_symbols(),
            all_symbols: synthetic_all_symbols(),
            monitor: None,
            guest_hz: Some(1000),
            build_id: Some(vec![0xaa, 0xbb]),
        };
        write_artifacts_sidecar(&artifacts_sidecar_path(&canon), &original);
        clear_vmlinux_cache_for_tests();

        let first = prepare_vmlinux(&kernel).expect("fresh sidecar prepares vmlinux");
        assert_eq!(first.path, vmlinux);
        assert_eq!(first.artifacts.guest_hz, Some(1000));
        assert_eq!(first.artifacts.build_id.as_deref(), Some(&[0xaa, 0xbb][..]));
        assert_eq!(first.artifacts.all_symbols, original.all_symbols);

        let second = prepare_vmlinux(&kernel).expect("second prepare hits process cache");
        assert_eq!(second.path, first.path);
        assert!(
            Arc::ptr_eq(&first.artifacts, &second.artifacts),
            "the second preparation must reuse the promoted artifact allocation",
        );
    }

    /// Write the sidecar from a monitor-absent `VmlinuxArtifacts`, then
    /// load it back through the real filesystem helpers and assert the
    /// reconstruction is byte-identical — the derived products survive a
    /// process boundary without re-touching the ELF.
    #[test]
    fn artifacts_sidecar_roundtrip_monitor_absent() {
        use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
        let _lock = lock_env();
        let tmp = tempfile::TempDir::new().unwrap();
        let _guard = EnvVarGuard::set(crate::KTSTR_CACHE_DIR_ENV, tmp.path());
        let entry = tmp.path().join("kentry");
        std::fs::create_dir_all(&entry).unwrap();
        let vmlinux = entry.join("vmlinux");
        std::fs::write(&vmlinux, b"fake-elf").unwrap();
        let canon = std::fs::canonicalize(&vmlinux).unwrap();

        let original = VmlinuxArtifacts {
            symbols: synthetic_symbols(),
            all_symbols: synthetic_all_symbols(),
            monitor: None,
            guest_hz: Some(1000),
            build_id: None,
        };
        let sidecar = artifacts_sidecar_path(&canon);
        write_artifacts_sidecar(&sidecar, &original);
        assert!(sidecar.exists(), "sidecar must be written next to vmlinux");

        let loaded = load_artifacts_sidecar(&canon).expect("fresh in-cache sidecar must load");
        assert!(loaded.monitor.is_none());
        assert_eq!(loaded.guest_hz, Some(1000));
        assert_eq!(
            postcard::to_allocvec(&loaded.symbols).unwrap(),
            postcard::to_allocvec(&original.symbols).unwrap(),
            "reconstructed symbols must equal the written ones",
        );
        assert_eq!(
            loaded.all_symbols, original.all_symbols,
            "the complete symbol table must survive the process boundary",
        );
    }

    /// A sidecar whose version tag differs from this build's is silently
    /// ignored (the caller re-derives + rewrites) — this is how a
    /// ktstr-version offset-layout change invalidates stale sidecars.
    #[test]
    fn artifacts_sidecar_version_mismatch_ignored() {
        use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
        let _lock = lock_env();
        let tmp = tempfile::TempDir::new().unwrap();
        let _guard = EnvVarGuard::set(crate::KTSTR_CACHE_DIR_ENV, tmp.path());
        let entry = tmp.path().join("kentry");
        std::fs::create_dir_all(&entry).unwrap();
        let vmlinux = entry.join("vmlinux");
        std::fs::write(&vmlinux, b"fake-elf").unwrap();
        let canon = std::fs::canonicalize(&vmlinux).unwrap();

        let payload = ArtifactsSidecar {
            version: "ktstr-vmlinux-artifacts-v1 0.0.0-stale".to_string(),
            symbols: synthetic_symbols(),
            all_symbols: synthetic_all_symbols_sidecar(),
            offsets: None,
            prog_offsets: None,
            psi_offsets: None,
            guest_hz: Some(250),
            build_id: None,
        };
        let bytes = postcard::to_allocvec(&payload).unwrap();
        atomic_write_sidecar(&artifacts_sidecar_path(&canon), &bytes).unwrap();

        assert!(
            load_artifacts_sidecar(&canon).is_none(),
            "a version-tag mismatch must be treated as a miss",
        );
    }

    /// A sidecar older than its vmlinux is stale and ignored, catching
    /// the "vmlinux rebuilt, sidecar not" case. The freshness rule is
    /// the same mtime comparison the `.btf` sidecar uses.
    #[test]
    #[cfg(unix)]
    fn artifacts_sidecar_stale_mtime_ignored() {
        use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
        let _lock = lock_env();
        let tmp = tempfile::TempDir::new().unwrap();
        let _guard = EnvVarGuard::set(crate::KTSTR_CACHE_DIR_ENV, tmp.path());
        let entry = tmp.path().join("kentry");
        std::fs::create_dir_all(&entry).unwrap();
        let vmlinux = entry.join("vmlinux");
        std::fs::write(&vmlinux, b"fake-elf").unwrap();
        let canon = std::fs::canonicalize(&vmlinux).unwrap();

        let original = VmlinuxArtifacts {
            symbols: synthetic_symbols(),
            all_symbols: synthetic_all_symbols(),
            monitor: None,
            guest_hz: Some(1000),
            build_id: None,
        };
        let sidecar = artifacts_sidecar_path(&canon);
        write_artifacts_sidecar(&sidecar, &original);
        assert!(
            load_artifacts_sidecar(&canon).is_some(),
            "fresh sidecar must load before staling",
        );

        // Push the sidecar's mtime to 1970-01-02 so vmlinux (written
        // now) is strictly newer -> stale. libc::utimes rather than a
        // sleep so the test runs in microseconds regardless of FS mtime
        // resolution.
        let sidecar_c = std::ffi::CString::new(sidecar.as_os_str().as_encoded_bytes()).unwrap();
        let past = libc::timeval {
            tv_sec: 86_400,
            tv_usec: 0,
        };
        let times = [past, past];
        // SAFETY: sidecar_c is a valid NUL-terminated path; times is a
        // 2-element (atime, mtime) array as utimes(2) requires.
        let rc = unsafe { libc::utimes(sidecar_c.as_ptr(), times.as_ptr()) };
        assert_eq!(rc, 0, "libc::utimes must succeed on the sidecar");

        assert!(
            load_artifacts_sidecar(&canon).is_none(),
            "a sidecar older than its vmlinux must be treated as stale",
        );
    }

    /// Every nested offset struct survives a postcard encode/decode
    /// roundtrip losslessly — proves the serde derives on `KernelOffsets`
    /// and all of `ScxEventOffsets` / `SchedstatOffsets` /
    /// `SchedDomainOffsets` / `SchedDomainStatsOffsets` /
    /// `ScxWatchdogOffsets` / `BpfProgOffsets` / `PsiGroupOffsets` are
    /// wired up. No filesystem: this pins the wire mirror, not the cache.
    #[test]
    fn artifacts_sidecar_postcard_roundtrip_all_offsets() {
        let payload = ArtifactsSidecar {
            version: ARTIFACTS_SIDECAR_VERSION.to_string(),
            symbols: synthetic_symbols(),
            all_symbols: synthetic_all_symbols_sidecar(),
            offsets: Some(synthetic_kernel_offsets()),
            prog_offsets: Some(synthetic_prog_offsets()),
            psi_offsets: Some(synthetic_psi_offsets()),
            guest_hz: Some(1000),
            build_id: None,
        };
        let encoded = postcard::to_allocvec(&payload).unwrap();
        let decoded: ArtifactsSidecar = postcard::from_bytes(&encoded).unwrap();
        assert_eq!(decoded.version, ARTIFACTS_SIDECAR_VERSION);
        assert!(decoded.offsets.is_some());
        let re_encoded = postcard::to_allocvec(&decoded).unwrap();
        assert_eq!(
            encoded, re_encoded,
            "lossless postcard roundtrip through every nested offset struct",
        );
    }
}
