//! Host-side BPF cast analysis driver for the scheduler binary.
//!
//! Bridges the path-based scheduler-binary input (a libbpf-rs / scx-built
//! ELF that embeds its compiled BPF objects into a `.bpf.objs` PROGBITS
//! section) and the pure-data [`crate::monitor::cast_analysis::analyze_casts`]
//! pass that turns BPF instructions plus a parsed [`btf_rs::Btf`] into a
//! [`crate::monitor::cast_analysis::CastMap`].
//!
//! # Pipeline
//!
//! 1. Read the scheduler binary from disk.
//! 2. Parse it as a host ELF via [`goblin::elf::Elf::parse`]; locate the
//!    `.bpf.objs` PROGBITS section. scx schedulers (the only producers
//!    we target) embed their compiled BPF object(s) inline at that
//!    section via the libbpf-rs / scx skel codegen. Each `STT_OBJECT`
//!    symbol in the outer ELF whose containing section is `.bpf.objs`
//!    points at a contiguous embedded ELF blob — the BPF object that
//!    the scheduler will hand to `bpf_object__load` at runtime.
//! 3. For each embedded ELF, parse its `.BTF` (and `.BTF.ext` when
//!    present) plus every program text section (any PROGBITS section
//!    flagged `SHF_EXECINSTR`).
//! 4. Concatenate the program texts in section-header order. Decode each
//!    8-byte slot through [`crate::monitor::cast_analysis::BpfInsn::from_le_bytes`].
//! 5. Walk `.BTF.ext`'s `func_info` and build the [`FuncEntry`] table:
//!    every record's `insn_off` (in BYTES) becomes a function-entry PC
//!    once divided by 8 and offset into the concatenated stream by the
//!    base of the section the record belongs to. The record's `type_id`
//!    is the BTF id of `BTF_KIND_FUNC` whose `func.type` is the
//!    [`btf_rs::Type::FuncProto`] the analyzer reseeds R1..R5 from.
//! 6. Run [`analyze_casts`]; merge the result into a single
//!    [`CastMap`] aggregating every embedded BPF object's findings.
//!
//! # Error policy
//!
//! Any failure returns an empty [`CastMap`]. The log level depends on
//! the failure kind: scheduler-binary read errors, outer ELF parse
//! failures, missing `.bpf.objs`, inner ELF parse failures, and
//! malformed `.BTF` log at `warn!` (these indicate a likely bug in
//! the scheduler build); a missing `.BTF` section and an inner ELF
//! with no executable BPF program sections log at `debug!` (these
//! shapes are valid for non-scx binaries that ship a `.bpf.objs` for
//! unrelated reasons). The dump path is best-effort — a missing
//! cast map silently disables typed-pointer promotion in the renderer
//! (every `u64` field renders as a plain counter, the pre-integration
//! default).
//!
//! No libbpf calls, no kernel BPF interaction, no CAP_BPF needed — this
//! runs purely on the on-disk binary bytes.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};

use crate::monitor::cast_analysis::{
    BPF_PSEUDO_CALL, BPF_PSEUDO_KFUNC_CALL, BpfInsn, CastMap, DatasecPointer, FuncEntry,
    SubprogReturn, analyze_casts,
};

use btf_rs::{Btf, Type};

/// One BPF instruction's wire size (bytes). Mirrors `sizeof(struct
/// bpf_insn)` in the kernel's UAPI and the [`BpfInsn::from_le_bytes`]
/// 8-byte input. Used to translate `.BTF.ext`-reported byte offsets
/// (`bpf_func_info::insn_off`) into instruction indices for
/// [`FuncEntry::insn_offset`].
const BPF_INSN_SIZE: usize = 8;

/// Resolve a string offset against the BTF string table embedded in
/// the `.BTF` section blob. Per kernel `include/uapi/linux/btf.h`,
/// the BTF header is: magic(2) + version(1) + flags(1) + hdr_len(4)
/// + type_off(4) + type_len(4) + str_off(4) + str_len(4) = 24 bytes.
///
/// The string table starts at `hdr_len + str_off` within the blob.
fn btf_str_at(btf_bytes: &[u8], str_off: u32) -> Option<&str> {
    if btf_bytes.len() < 24 {
        return None;
    }
    let hdr_len = u32::from_le_bytes(btf_bytes[4..8].try_into().ok()?) as usize;
    let str_section_off = u32::from_le_bytes(btf_bytes[16..20].try_into().ok()?) as usize;
    let str_section_len = u32::from_le_bytes(btf_bytes[20..24].try_into().ok()?) as usize;
    let str_start = hdr_len + str_section_off;
    let off = str_off as usize;
    if off >= str_section_len {
        return None;
    }
    let base = str_start + off;
    if base >= btf_bytes.len() {
        return None;
    }
    let strtab_end = (str_start + str_section_len).min(btf_bytes.len());
    if base >= strtab_end {
        return None;
    }
    let end = btf_bytes[base..strtab_end]
        .iter()
        .position(|&b| b == 0)
        .map(|p| base + p)
        .unwrap_or(strtab_end);
    std::str::from_utf8(&btf_bytes[base..end]).ok()
}

/// `.BTF.ext` magic — `0xEB9F` in native byte order.
///
/// Same magic as the `.BTF` section. A mismatch here (truncation,
/// foreign-endian, corruption) triggers the silent-empty-result path:
/// the cast analyzer never sees garbage data.
const BTF_MAGIC: u16 = 0xEB9F;

/// Minimum `.BTF.ext` header byte size. Per kernel
/// `tools/lib/bpf/btf.c:btf_ext_parse`, the minimum is
/// `offsetofend(struct btf_ext_header, line_info_len)` = 24 bytes:
/// magic(2) + version(1) + flags(1) + hdr_len(4) + func_info_off(4)
/// + func_info_len(4) + line_info_off(4) + line_info_len(4).
const BTF_EXT_HEADER_MIN_LEN: u32 = 24;

/// One entry in the cross-BTF Fwd resolution index — locates a
/// complete struct/union body by `(BTF index, type id)`.
///
/// `btfs_idx` selects which entry of [`CastAnalysisOutput::btfs`]
/// carries the body; `type_id` is the type id WITHIN that BTF's
/// own id space (distinct from the entry BTF's id space the
/// renderer's chase entered with).
///
/// Used as the value type of [`CastAnalysisOutput::fwd_index`] —
/// the renderer's
/// [`crate::monitor::btf_render::MemReader::cross_btf_resolve_fwd`]
/// override looks the entry up by name, picks `btfs[btfs_idx]`,
/// and recurses against `type_id`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FwdIndexEntry {
    /// Index into [`CastAnalysisOutput::btfs`] selecting which
    /// embedded BPF object's parsed program BTF carries the body.
    pub(crate) btfs_idx: usize,
    /// Type id within `btfs[btfs_idx]`'s own id space. Distinct
    /// from the entry BTF's id space; the chase code switches the
    /// rendering BTF before resolving the id.
    pub(crate) type_id: u32,
}

/// Output of one full pass of host-side scheduler cast analysis: the
/// `(parent_struct, member_offset) -> CastHit` map, the list of every
/// embedded BPF object's program BTF, and a name-keyed index over
/// every complete (`!is_fwd`) struct/union/typedef across those BTFs.
///
/// The renderer's chase paths consult the cross-BTF index when a
/// declared `BTF_KIND_FWD` pointee has no complete sibling in its
/// own BTF: the index points at the `(btfs[idx], type_id)` pair where
/// the body lives, so a `cgx_target __arena *` declared in object A
/// (Fwd-only) renders as the full `struct cgx_target { ... }` body
/// from object B without dropping into the "forward declaration; body
/// not in this BTF" skip.
///
/// Built once per scheduler binary per process via
/// [`cached_cast_analysis_for_scheduler`] and shared across VMs by
/// content hash. The `btfs` vec is `Arc<Btf>` so the rendered
/// borrows live for the full dump pass without copying the parsed
/// BTF.
pub(crate) struct CastAnalysisOutput {
    /// `(parent_btf_id, member_offset) -> CastHit` recovered by the
    /// instruction-level cast analyzer. The renderer's
    /// [`crate::monitor::btf_render::MemReader::cast_lookup`] hits
    /// against the per-program BTF the rendered map was loaded from.
    /// Even when the cast hit is empty, the wrapping output is still
    /// retained because the cross-BTF `fwd_index` is independently
    /// useful — a scheduler whose Fwd pointers all live in
    /// non-typed-pointer-bearing maps still benefits from the index
    /// when the renderer chases those maps' [`Type::Ptr`] arms.
    pub(crate) cast_maps: Vec<Arc<CastMap>>,
    /// Every embedded BPF object's parsed program BTF, in the same
    /// order [`iter_embedded_bpf_objects`] yielded the slices. Index
    /// 0 is the first symbol-driven slice (or the fallback whole-
    /// section blob), index 1 is the next, and so on. Empty when no
    /// BTF parsed successfully — the renderer falls back to the
    /// per-map vmlinux BTF for any cross-BTF resolution that would
    /// have hit this index.
    pub(crate) btfs: Vec<Arc<Btf>>,
    /// `struct_or_union_name -> FwdIndexEntry` for every complete
    /// (`!is_fwd`) [`btf_rs::Type::Struct`] / [`btf_rs::Type::Union`]
    /// across `btfs`. `Typedef` is NOT indexed — typedefs add no
    /// body and the chase path peels through them via
    /// `peel_modifiers_with_id` before consulting the index.
    ///
    /// First-write-wins: when the same name appears in multiple
    /// BTFs the index keeps the first-seen entry. Two distinct
    /// programs declaring `struct foo` with conflicting layouts
    /// would each see their own program BTF resolve correctly via
    /// the renderer's local Fwd-resolving peel; the cross-BTF index
    /// only fires when the local resolve failed. The first-write-
    /// wins policy keeps the index deterministic across re-runs of
    /// the analyzer on the same binary.
    ///
    /// Anonymous structs/unions are not indexed (no name to key on);
    /// the chase falls through to the existing "forward declaration;
    /// body not in this BTF" skip path for those.
    pub(crate) fwd_index: HashMap<String, FwdIndexEntry>,
    /// Unique alloc_sizes captured from `scx_static_alloc_internal`
    /// call sites via [`build_subprog_returns`]. Threaded to the
    /// renderer as a last-resort fallback for deferred-resolve
    /// arena chases whose CastHit has `alloc_size: None`.
    /// `(alloc_size, struct_name)` pairs: for each captured alloc_size
    /// from `scx_static_alloc_internal`, the struct name that
    /// `discover_payload_btf_id` resolved uniquely in the embedded
    /// `.bpf.o` BTF. The renderer uses the name with
    /// `cross_btf_resolve_fwd` to find the struct body at chase time.
    /// Empty when no sizes resolved or no embedded BTF was available.
    pub(crate) alloc_size_types: Vec<(u64, String)>,
}

/// Per-`KtstrVm` lazy on-demand BPF cast-analysis handle.
///
/// Captures the scheduler binary path at VM build time (no analyzer
/// work runs here) and exposes a lazy accessor (`.get_full()`)
/// that runs the analysis on first call and caches the result
/// inside an [`OnceLock`]. The failure-dump path is the only
/// production caller, so a test that passes without ever dumping
/// pays zero analyzer cost. A test that triggers multiple dumps
/// in the same VM (e.g. periodic-capture + final freeze) only
/// runs the analyzer once.
///
/// # Cross-VM sharing
///
/// `.get_full()` consults the process-wide content-hash cache via
/// [`cached_cast_analysis_for_scheduler`], so two VMs in the same
/// process that share a scheduler binary share one analyzed
/// `Arc<CastAnalysisOutput>`. Production runs under nextest use
/// process-per-test by default, so the cross-VM share helps mostly
/// for the auto-repro path (which boots a second VM in the same
/// process after a primary-test failure) and for any future
/// in-process multi-test driver.
///
/// # Concurrency
///
/// `OnceLock::get_or_init` serialises concurrent first-callers in
/// the same VM: the second caller blocks while the first runs the
/// analysis, then both observe the cached
/// `Option<Arc<CastAnalysisOutput>>`. The inner
/// [`cached_cast_analysis_for_scheduler`] additionally dedupes work
/// across VMs by content hash and uses an inner `OnceLock` per
/// cache entry to avoid the thundering-herd shape where two VMs
/// find the cache empty under the same lock and both run the
/// analyzer after releasing it.
pub(crate) struct LazyCastMap {
    /// Scheduler binary path captured at VM build time. `None`
    /// when the builder had no scheduler binary; `.get_full()`
    /// returns `None` immediately in that case.
    scheduler_binary: Option<std::path::PathBuf>,
    /// One-shot per-VM cache of the analysis result. Populated by
    /// the first `.get_full()` caller via
    /// [`cached_cast_analysis_for_scheduler`]; `None` is cached
    /// when no scheduler binary was set OR the analyzer produced
    /// neither cast findings nor cross-BTF index entries.
    inner: OnceLock<Option<Arc<CastAnalysisOutput>>>,
}

impl LazyCastMap {
    /// Construct a lazy handle for `scheduler_binary`. No file I/O
    /// or analyzer work runs here — both defer to
    /// [`Self::get_full`].
    pub(crate) fn new(scheduler_binary: Option<std::path::PathBuf>) -> Self {
        Self {
            scheduler_binary,
            inner: OnceLock::new(),
        }
    }

    /// Force the lazy analysis (or return the cached result) and
    /// hand back the full [`CastAnalysisOutput`] including the
    /// cross-BTF Fwd index.
    ///
    /// First call runs [`cached_cast_analysis_for_scheduler`] on
    /// the captured path, which itself consults the process-wide
    /// content-hash cache — so two VMs that share a scheduler
    /// binary path produce one analyzer run per process.
    /// Subsequent `.get_full()` calls on the same VM hit the inner
    /// `OnceLock` and return immediately.
    ///
    /// Returns `None` when no scheduler binary was set, the file
    /// read failed, or the analyzer produced neither cast findings
    /// nor cross-BTF index entries.
    pub(crate) fn get_full(&self) -> Option<Arc<CastAnalysisOutput>> {
        self.inner
            .get_or_init(|| {
                self.scheduler_binary
                    .as_deref()
                    .and_then(cached_cast_analysis_for_scheduler)
            })
            .clone()
    }
}

/// Process-wide cache entry: scheduler binary content hash →
/// `Arc<OnceLock<Option<Arc<CastAnalysisOutput>>>>`. The outer
/// `OnceLock` is the deduplication primitive — two VMs that hash
/// to the same content but find the entry uninitialized both call
/// `entry.get_or_init(...)`, which runs the analyzer exactly once.
/// The entry's eventual value is the collapsed
/// `Option<Arc<CastAnalysisOutput>>` (`None` on empty cast map AND
/// empty cross-BTF index, `Some` on any non-empty). Without the
/// inner `OnceLock` shape, two cache misses on the same hash would
/// each release the `Mutex<HashMap>` lock, then race to run the
/// analyzer in parallel — the thundering-herd anti-pattern.
type CastCacheEntry = Arc<OnceLock<Option<Arc<CastAnalysisOutput>>>>;

fn cast_cache() -> &'static Mutex<HashMap<u64, CastCacheEntry>> {
    static CACHE: OnceLock<Mutex<HashMap<u64, CastCacheEntry>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn ahash_bytes(bytes: &[u8]) -> u64 {
    use std::hash::{BuildHasher, Hasher};
    let mut hasher = ahash::RandomState::with_seeds(0, 0, 0, 0).build_hasher();
    hasher.write(bytes);
    hasher.finish()
}

/// Process-wide content-hash-cached entry point.
///
/// Reads the scheduler binary once, hashes the bytes via ahash
/// (AES-NI accelerated, deterministic per-binary with fixed seeds),
/// and either returns the previously-analysed
/// `Option<Arc<CastAnalysisOutput>>` for that hash or runs the
/// analyzer once to populate the cache entry. The cache value is
/// `Option<Arc>` (collapsed empty → `None`) so the dump path's
/// borrow expresses "no analysis available" cleanly without an
/// emptiness check at every freeze.
///
/// # Why content-hash, not path-stat
///
/// `(path, dev, ino, mtime, len)` would be a stale-tolerant cache
/// key when scheduler binaries always rebuild with a fresh mtime,
/// but a `cp -p`-style overwrite or hardlinked rotation can
/// preserve mtime AND length while the bytes change, hitting a
/// stale entry and rendering the wrong cast map for a
/// just-replaced binary. Content-hash over the actual bytes is
/// the only key that is correct for every overwrite shape. The
/// hash cost is dominated by the file read which has to happen
/// anyway.
///
/// # Concurrency
///
/// Two simultaneous misses for the same hash do NOT both run the
/// analyzer — they share an `Arc<OnceLock<...>>` and the second
/// caller blocks inside `OnceLock::get_or_init` until the first
/// finishes. Misses for different hashes proceed in parallel
/// because the `Mutex<HashMap>` is held only across the
/// hash-and-fetch step.
///
/// # Returns
///
/// `None` when the file read fails (transient I/O) OR the
/// analyzer's result is empty AND the cross-BTF index is empty.
/// Otherwise the analyzed `Arc<CastAnalysisOutput>` shared with
/// every prior caller for the same binary content.
pub(crate) fn cached_cast_analysis_for_scheduler(path: &Path) -> Option<Arc<CastAnalysisOutput>> {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            tracing::warn!(
                error = %e,
                path = %path.display(),
                "cast_analysis: read scheduler binary failed; \
                 dump renderer will fall back to plain u64 counters"
            );
            return None;
        }
    };
    let hash_t0 = std::time::Instant::now();
    let hash = ahash_bytes(&bytes);
    tracing::debug!(
        elapsed_us = hash_t0.elapsed().as_micros() as u64,
        len = bytes.len(),
        hash = format_args!("{hash:016x}"),
        "cast_analysis: scheduler binary content hash finished"
    );

    let entry: CastCacheEntry = {
        let mut cache = cast_cache().lock().unwrap();
        cache
            .entry(hash)
            .or_insert_with(|| Arc::new(OnceLock::new()))
            .clone()
    };
    entry
        .get_or_init(|| {
            // Disk cache probe: if a prior process already analyzed
            // this binary, load the result without re-running the
            // instruction walker. BTFs are reparsed from the binary
            // bytes (Btf is not serializable).
            let btfs = parse_btfs_from_bytes(&bytes);
            if let Some((cast_map, fwd_index, alloc_size_types)) =
                persist::try_load(hash, btfs.len())
            {
                tracing::debug!("cast_analysis: disk cache hit");
                let out = CastAnalysisOutput {
                    cast_maps: vec![Arc::new(cast_map)],
                    btfs,
                    fwd_index,
                    alloc_size_types,
                };
                let total: usize = out.cast_maps.iter().map(|m| m.len()).sum();
                return if total == 0 && out.fwd_index.is_empty() {
                    None
                } else {
                    Some(Arc::new(out))
                };
            }

            let analyze_t0 = std::time::Instant::now();
            let out = build_cast_analysis_from_bytes(&bytes);
            tracing::debug!(
                elapsed_ms = analyze_t0.elapsed().as_millis() as u64,
                casts = out.cast_maps.iter().map(|m| m.len()).sum::<usize>(),
                btfs = out.btfs.len(),
                fwd_index = out.fwd_index.len(),
                "cast_analysis: on-demand analysis finished"
            );
            let merged_for_cache: CastMap = out
                .cast_maps
                .iter()
                .flat_map(|m| m.iter())
                .map(|(&k, &v)| (k, v))
                .collect();
            // Do not cache a lossy multi-object merge. When >1 embedded
            // object carries casts the flat (parent_id, offset) merge
            // collides (per-object BTF id-spaces -- see the "Single-object
            // only" note on build_cast_analysis_from_bytes), which already
            // logged a loud error!. Caching the collided map would mask
            // that error on every later cache hit (the build path, and its
            // error!, is skipped on a hit), turning the loud guard into a
            // one-shot. Skip the write so the guard re-fires every run
            // until per-btf_kva selection lands.
            if objects_with_casts(&out.cast_maps) <= 1 {
                persist::try_save(
                    hash,
                    &merged_for_cache,
                    &out.fwd_index,
                    out.btfs.len(),
                    &out.alloc_size_types,
                );
            }
            let total_casts: usize = out.cast_maps.iter().map(|m| m.len()).sum();
            if total_casts == 0 && out.fwd_index.is_empty() {
                None
            } else {
                Some(Arc::new(out))
            }
        })
        .clone()
}

/// Count embedded BPF objects that produced at least one cast.
///
/// The dump renderer threads a single cast map (`cast_maps.first()`
/// in the freeze coordinator) and the disk cache merges all objects
/// into one (`get_full`); both are correct only when at most one
/// object carries casts. Per-object program BTFs each restart their
/// user-type ids at `vmlinux_last + 1`, so the same
/// `(parent_id, offset)` from two objects collides -- the merge
/// overwrites, `first()` drops. A count > 1 is therefore
/// unrenderable today; `build_cast_analysis_from_bytes` logs a loud
/// `error!` so the gap is never silent.
fn objects_with_casts(cast_maps: &[Arc<CastMap>]) -> usize {
    cast_maps.iter().filter(|m| !m.is_empty()).count()
}

/// Run the cast-analysis pipeline on already-loaded scheduler
/// binary bytes.
///
/// Locates every embedded BPF object inside `.bpf.objs`, parses
/// each object's program BTF, runs the analyzer per-object, and
/// returns one [`CastMap`] per object alongside the parsed BTFs and a
/// name-keyed cross-BTF Fwd resolution index over every complete
/// struct/union across them. The renderer's chase paths consume
/// the index when a `BTF_KIND_FWD` pointee in one BTF resolves to
/// a complete sibling in another — the typical multi-object
/// scheduler shape where one `.bpf.c` declares
/// `struct cgx_target;` (forward) and a sibling object defines
/// `struct cgx_target { ... }` (full body).
///
/// Returns an empty [`CastAnalysisOutput`] on parse failure
/// (`cast_map` empty, `btfs` empty, `fwd_index` empty). Per-stage
/// timing is emitted at `debug!` so a future regression in any
/// sub-stage is visible without re-instrumenting.
///
/// This is the lowest-level entry point; see
/// [`cached_cast_analysis_for_scheduler`] for the production
/// path-driven, content-hash-cached, lazy-on-demand wrapper.
///
/// # Single-object only (multi-object guarded)
///
/// scx schedulers ship one embedded BPF object per binary today, so
/// `cast_maps` has a single entry and the downstream single-map
/// threading is exact. Multi-object schedulers do not exist and are
/// NOT handled: [`crate::monitor::btf_render::MemReader::cast_lookup`]
/// consults one flat map keyed on `(parent_type_id, offset)`, the
/// freeze coordinator threads only `cast_maps.first()`, and the disk
/// cache would persist a single merged map. Per-object program BTFs
/// each restart user-type ids at `vmlinux_last + 1`, so the same
/// `(parent_id, offset)` from two objects collides -- the merge
/// overwrites, `first()` drops. Arena resolution is unaffected:
/// `resolve_arena_type` is already `requesting_btf_kva`-scoped; only
/// the cast lookup is flat.
///
/// `objects_with_casts` detects the multi-object case and
/// `build_cast_analysis_from_bytes` logs a loud `error!`; `get_full`
/// then skips the disk write so the `error!` re-fires every run
/// instead of being masked by a cached lossy map. Correct support
/// needs per-`btf_kva` cast-map selection, unimplemented because no
/// multi-object scheduler exists. The conservative "false negatives
/// are fine, false positives are not" stance from
/// [`crate::monitor::cast_analysis`] still applies.
pub(crate) fn build_cast_analysis_from_bytes(bytes: &[u8]) -> CastAnalysisOutput {
    let parse_t0 = std::time::Instant::now();
    let outer = match goblin::elf::Elf::parse(bytes) {
        Ok(e) => e,
        Err(e) => {
            tracing::warn!(
                error = %e,
                "cast_analysis: parse outer ELF failed; \
                 dump renderer will fall back to plain u64 counters"
            );
            return CastAnalysisOutput {
                cast_maps: vec![Arc::new(CastMap::new())],
                btfs: Vec::new(),
                fwd_index: HashMap::new(),
                alloc_size_types: Vec::new(),
            };
        }
    };
    let bpf_objs_section = match find_section(&outer, ".bpf.objs") {
        Some(s) => s,
        None => {
            tracing::debug!(
                "cast_analysis: scheduler binary has no .bpf.objs section; \
                 typed-pointer rendering disabled"
            );
            return CastAnalysisOutput {
                cast_maps: vec![Arc::new(CastMap::new())],
                btfs: Vec::new(),
                fwd_index: HashMap::new(),
                alloc_size_types: Vec::new(),
            };
        }
    };
    tracing::debug!(
        elapsed_us = parse_t0.elapsed().as_micros() as u64,
        "cast_analysis: outer ELF parse + .bpf.objs lookup finished"
    );

    let mut cast_maps: Vec<Arc<CastMap>> = Vec::new();
    let mut btfs: Vec<Arc<Btf>> = Vec::new();
    let mut all_alloc_sizes: Vec<u64> = Vec::new();
    let started = std::time::Instant::now();
    tracing::debug!("cast_analysis: starting analyze_casts pipeline");
    for inner in iter_embedded_bpf_objects(&outer, bytes, bpf_objs_section) {
        let one_t0 = std::time::Instant::now();
        let (one, btf_for_obj, obj_alloc_sizes) = analyze_one_object_with_btf(inner);
        tracing::debug!(
            elapsed_ms = one_t0.elapsed().as_millis() as u64,
            casts = one.len(),
            "cast_analysis: analyze_one_object_with_btf finished"
        );
        cast_maps.push(Arc::new(one));
        all_alloc_sizes.extend_from_slice(&obj_alloc_sizes);
        if let Some(btf) = btf_for_obj {
            btfs.push(btf);
        }
    }
    let total_casts: usize = cast_maps.iter().map(|m| m.len()).sum();
    tracing::debug!(
        elapsed_ms = started.elapsed().as_millis() as u64,
        casts = total_casts,
        btfs = btfs.len(),
        objects = cast_maps.len(),
        "cast_analysis: analyze_casts pipeline finished"
    );

    // Fail loudly on the unsupported multi-object case rather than
    // silently dropping or mis-rendering casts. See `objects_with_casts`
    // and the "Single-object only" note above: the renderer threads
    // `cast_maps.first()` and the disk cache merges every object into a
    // single flat `(parent_id, offset)` map, but per-object program BTFs
    // restart their id-space at `vmlinux_last + 1`, so casts from objects
    // 2+ collide on the merge and are dropped by `first()`. No multi-object
    // scx scheduler ships today; this guards the future.
    let cast_bearing_objects = objects_with_casts(&cast_maps);
    if cast_bearing_objects > 1 {
        tracing::error!(
            objects = cast_maps.len(),
            cast_bearing_objects,
            "cast analysis found casts in more than one embedded BPF object; \
             multi-object cast rendering is unsupported -- casts from objects \
             2+ are dropped (renderer) or overwritten (disk cache) because \
             per-object BTF id-spaces collide. correct support needs \
             per-btf_kva cast-map selection."
        );
    }

    // Build the cross-BTF Fwd resolution index over every parsed
    // BTF. `build_fwd_index` walks each BTF's id space looking for
    // complete struct/union definitions and records `name ->
    // (btfs index, type id)`; first-write-wins on duplicate names
    // (see [`CastAnalysisOutput::fwd_index`]).
    let fwd_t0 = std::time::Instant::now();
    let fwd_index = build_fwd_index(&btfs);
    tracing::debug!(
        elapsed_us = fwd_t0.elapsed().as_micros() as u64,
        entries = fwd_index.len(),
        "cast_analysis: build_fwd_index finished"
    );

    // Demote to debug! when no casts were recovered: a clean
    // analyze on a scheduler with no typed pointers is a normal
    // outcome, not an event the operator needs to see at info!
    // (which would surface as a startup line on every test run).
    // Non-empty results stay at info! so the operator sees the
    // recovery count when it matters.
    if total_casts == 0 {
        tracing::debug!(
            casts = 0,
            "cast_analysis: recovered 0 typed pointers from scheduler"
        );
    } else {
        tracing::info!(
            casts = total_casts,
            "cast_analysis: recovered typed pointers from scheduler"
        );
    }
    all_alloc_sizes.sort_unstable();
    all_alloc_sizes.dedup();
    // For each captured alloc_size, try discover_payload_btf_id
    // against every embedded BTF. The embedded BTFs carry full
    // struct bodies that may be Fwd-only in the kernel's split BTF.
    // Store (size, struct_name) so the renderer can cross-BTF-resolve
    // by name at chase time.
    //
    // Walk each BTF's struct id-space exactly once via
    // [`enumerate_named_structs`] (consecutive-fail-cap to bail at the
    // dense table's end, [`crate::monitor::sdt_alloc::MAX_BTF_ID_PROBE`]
    // backstops a sparse BTF). The cached `(size, name)` table is then
    // probed per alloc_size — replaces a quadratic per-size re-walk
    // AND the prior `take_while().last()` max-id discovery, which
    // bailed on the first id gap and undercounted on sparse split-BTF
    // tables.
    let mut alloc_size_types: Vec<(u64, String)> = Vec::with_capacity(all_alloc_sizes.len());
    let mut seen_names: std::collections::HashSet<String> = std::collections::HashSet::new();
    let per_btf_structs: Vec<Vec<(u64, String)>> = btfs
        .iter()
        .map(|ebtf| enumerate_named_structs(ebtf))
        .collect();
    for &size in &all_alloc_sizes {
        if size == 0 {
            continue;
        }
        for (ebtf, structs) in btfs.iter().zip(per_btf_structs.iter()) {
            let choice =
                super::super::monitor::sdt_alloc::discover_payload_btf_id(ebtf, size as usize, "");
            if choice.target_type_id != 0 {
                if let Ok(ty) = ebtf.resolve_type_by_id(choice.target_type_id)
                    && let Some(bt) = ty.as_btf_type()
                    && let Ok(name) = ebtf.resolve_name(bt)
                    && !name.is_empty()
                    && seen_names.insert(name.to_string())
                {
                    alloc_size_types.push((size, name.to_string()));
                }
                break;
            }
            // For ambiguous sizes, collect all scheduler-
            // convention candidates (names ending in _ctx,
            // _arena_ctx, or exact task_ctx). The cross-BTF
            // resolution at chase time disambiguates by name.
            for (struct_size, name) in structs {
                if *struct_size != size {
                    continue;
                }
                let dominated =
                    name == "task_ctx" || name.ends_with("_ctx") || name.ends_with("_arena_ctx");
                if dominated && seen_names.insert(name.clone()) {
                    alloc_size_types.push((size, name.clone()));
                }
            }
        }
    }
    CastAnalysisOutput {
        cast_maps,
        btfs,
        fwd_index,
        alloc_size_types,
    }
}

fn parse_btfs_from_bytes(bytes: &[u8]) -> Vec<Arc<Btf>> {
    let outer = match goblin::elf::Elf::parse(bytes) {
        Ok(e) => e,
        Err(_) => return Vec::new(),
    };
    let bpf_objs_section = match find_section(&outer, ".bpf.objs") {
        Some(s) => s,
        None => return Vec::new(),
    };
    let mut btfs = Vec::new();
    for inner in iter_embedded_bpf_objects(&outer, bytes, bpf_objs_section) {
        let elf = match goblin::elf::Elf::parse(inner) {
            Ok(e) => e,
            Err(_) => continue,
        };
        let btf_bytes = match find_section(&elf, ".BTF").and_then(|i| section_data(&elf, inner, i))
        {
            Some(b) => b,
            None => continue,
        };
        if let Ok(btf) = Btf::from_bytes(btf_bytes) {
            btfs.push(Arc::new(btf));
        }
    }
    btfs
}

/// Walk every parsed BTF and collect a `name -> FwdIndexEntry`
/// index of complete (`!is_fwd`) struct/union definitions for the
/// renderer's cross-BTF Fwd resolution path. First-write-wins —
/// see [`CastAnalysisOutput::fwd_index`] for the rationale.
///
/// The id-space walk uses the same `consecutive_fail` cap pattern
/// as [`crate::monitor::sdt_alloc::discover_payload_btf_id`]: real
/// BPF BTFs have dense id tables, so 256 consecutive failed
/// `resolve_type_by_id` calls is safe to treat as "table
/// exhausted". The hard ceiling
/// [`crate::monitor::sdt_alloc::MAX_BTF_ID_PROBE`] backstops a
/// pathological / synthesized BTF.
///
/// Anonymous structs/unions are silently skipped (no name to key
/// the index entry on). Type kinds that are not Struct/Union are
/// also skipped — the index is consumed by the renderer's
/// [`crate::monitor::btf_render::peel_modifiers_resolving_fwd`]
/// extension, which only looks up Fwd terminals against this
/// table.
fn build_fwd_index(btfs: &[Arc<Btf>]) -> HashMap<String, FwdIndexEntry> {
    let mut out: HashMap<String, FwdIndexEntry> = HashMap::new();
    const CONSECUTIVE_FAIL_CAP: u32 = 256;
    for (idx, btf) in btfs.iter().enumerate() {
        let mut tid: u32 = 1;
        let mut consecutive_fail: u32 = 0;
        while tid < crate::monitor::sdt_alloc::MAX_BTF_ID_PROBE {
            match btf.resolve_type_by_id(tid) {
                Ok(ty) => {
                    consecutive_fail = 0;
                    match &ty {
                        Type::Struct(s) | Type::Union(s) => {
                            if let Ok(name) = btf.resolve_name(s)
                                && !name.is_empty()
                            {
                                out.entry(name).or_insert(FwdIndexEntry {
                                    btfs_idx: idx,
                                    type_id: tid,
                                });
                            }
                        }
                        Type::Typedef(td) => {
                            if let Ok(td_name) = btf.resolve_name(td)
                                && !td_name.is_empty()
                                && let Some(pid) = <dyn btf_rs::BtfType>::get_type_id(td)
                                && let Ok(Type::Struct(s)) = btf.resolve_type_by_id(pid)
                                && btf.resolve_name(&s).map_or(true, |n| n.is_empty())
                            {
                                let base = td_name.strip_suffix("_t").unwrap_or(&td_name);
                                out.entry(base.to_string()).or_insert(FwdIndexEntry {
                                    btfs_idx: idx,
                                    type_id: pid,
                                });
                            }
                        }
                        _ => {}
                    }
                }
                Err(_) => {
                    consecutive_fail += 1;
                    if consecutive_fail >= CONSECUTIVE_FAIL_CAP {
                        break;
                    }
                }
            }
            tid += 1;
        }
    }
    out
}

/// Enumerate every named [`Type::Struct`] in one BTF as
/// `(struct_size, struct_name)` pairs.
///
/// Mirrors the consecutive-fail-cap pattern from [`build_fwd_index`]
/// and [`crate::monitor::sdt_alloc::discover_payload_btf_id`]: real
/// BPF BTFs have dense id tables, so 256 consecutive `resolve_type_by_id`
/// failures is safe to treat as "table exhausted"; the hard ceiling
/// [`crate::monitor::sdt_alloc::MAX_BTF_ID_PROBE`] backstops a
/// pathological / sparse BTF id space.
///
/// Anonymous structs (empty resolved name) and non-Struct kinds are
/// skipped — the caller looks up by name and only cares about struct
/// kinds.
fn enumerate_named_structs(btf: &Btf) -> Vec<(u64, String)> {
    const CONSECUTIVE_FAIL_CAP: u32 = 256;
    let mut out: Vec<(u64, String)> = Vec::new();
    let mut tid: u32 = 1;
    let mut consecutive_fail: u32 = 0;
    while tid < crate::monitor::sdt_alloc::MAX_BTF_ID_PROBE {
        match btf.resolve_type_by_id(tid) {
            Ok(ty) => {
                consecutive_fail = 0;
                if let Type::Struct(s) = &ty
                    && let Ok(name) = btf.resolve_name(s)
                    && !name.is_empty()
                {
                    out.push((s.size() as u64, name));
                }
            }
            Err(_) => {
                consecutive_fail += 1;
                if consecutive_fail >= CONSECUTIVE_FAIL_CAP {
                    break;
                }
            }
        }
        tid += 1;
    }
    out
}

/// Walk the outer ELF's symbol tables and yield every byte slice that
/// belongs to a `STT_OBJECT` symbol whose section is `.bpf.objs`.
///
/// scx-built schedulers emit a single such symbol per BPF object — the
/// libbpf-rs `bpf_skel::imp::DATA` slice the runtime hands to
/// `bpf_object__load`. A scheduler that statically composes multiple
/// BPF objects (theoretical; not produced by today's scx skel codegen)
/// would emit one symbol per object and the iterator would yield each
/// in turn. The fallback "one slice covering the whole section" path
/// ensures a hand-crafted scheduler that drops the symbol table still
/// gets analyzed: the section name alone is enough to identify the
/// blob.
fn iter_embedded_bpf_objects<'data>(
    outer: &goblin::elf::Elf<'_>,
    file_bytes: &'data [u8],
    bpf_objs_idx: usize,
) -> Vec<&'data [u8]> {
    let mut out: Vec<&[u8]> = Vec::new();
    // Symbol-driven path: every STT_OBJECT pointing into .bpf.objs.
    // st_value is the section-relative virtual address (the section's
    // sh_addr is the section start in the file's virtual layout); a
    // typical `.bpf.objs` is non-allocated and sh_addr matches sh_offset
    // semantics here, but we anchor on the section's file offset
    // explicitly to avoid relying on that coincidence.
    let sh = &outer.section_headers[bpf_objs_idx];
    let sec_file_start = sh.sh_offset as usize;
    let sec_file_end = sec_file_start.saturating_add(sh.sh_size as usize);
    let sec_va_start = sh.sh_addr;
    for sym in outer.syms.iter() {
        // STT_OBJECT (data symbol); section index match ties the
        // symbol to .bpf.objs. SHN_UNDEF / SHN_ABS / SHN_COMMON are
        // below the section-header range so the equality test
        // already excludes them.
        if sym.st_type() != goblin::elf::sym::STT_OBJECT {
            continue;
        }
        if sym.st_shndx != bpf_objs_idx {
            continue;
        }
        if sym.st_size == 0 {
            continue;
        }
        // Translate virtual address → file offset. For a typical
        // non-allocated `.bpf.objs` section, sh_addr is 0 and st_value
        // is the byte offset within the section. For an allocated
        // section, sh_addr is the load address and st_value is also
        // a virtual address; in either case the per-symbol offset
        // within the section is `st_value - sh_addr`, and the file
        // offset is `sec_file_start + (st_value - sh_addr)`. Using
        // checked arithmetic so a symbol whose st_value somehow
        // precedes sh_addr (corrupted ELF) is rejected rather than
        // wrapping into a wild slice index.
        let Some(rel) = sym.st_value.checked_sub(sec_va_start) else {
            continue;
        };
        let Some(start) = (sec_file_start as u64).checked_add(rel) else {
            continue;
        };
        let Some(end) = start.checked_add(sym.st_size) else {
            continue;
        };
        if (start as usize) < sec_file_start || (end as usize) > sec_file_end {
            continue;
        }
        if let Some(slice) = file_bytes.get(start as usize..end as usize) {
            out.push(slice);
        }
    }
    if out.is_empty() {
        // No matching symbol — fall back to treating the entire
        // section as one BPF object. scx-built binaries always emit
        // a covering symbol; a stripped binary or a custom scheduler
        // that omits it still gets analysis as long as the section's
        // bytes are themselves a valid BPF object ELF.
        if let Some(slice) = file_bytes.get(sec_file_start..sec_file_end) {
            out.push(slice);
        }
    }
    out
}

/// Run cast analysis on one embedded BPF object's bytes and
/// return the parsed BTF alongside the cast map.
///
/// The bytes are themselves an ELF (the BPF object); parse it, extract
/// the BTF, the `.BTF.ext`-derived [`FuncEntry`] table, and the
/// concatenated instruction stream, then call [`analyze_casts`].
///
/// The parsed BTF is returned wrapped in `Arc` so the caller can
/// retain it across the dump pass without copying. `None` for the
/// BTF position indicates a parse failure or an inner ELF without
/// a `.BTF` section — the cast map is still returned (empty in that
/// case) so the merger keeps working without distinguishing the
/// no-BTF inner from one with no recovered casts.
fn analyze_one_object_with_btf(obj_bytes: &[u8]) -> (CastMap, Option<Arc<Btf>>, Vec<u64>) {
    let elf = match goblin::elf::Elf::parse(obj_bytes) {
        Ok(e) => e,
        Err(e) => {
            tracing::warn!(
                error = %e,
                "cast_analysis: parse inner BPF object ELF failed"
            );
            return (CastMap::new(), None, Vec::new());
        }
    };

    // .BTF is mandatory — no BTF, no struct/field resolution, no
    // analysis output the renderer can use.
    let btf_bytes = match find_section(&elf, ".BTF").and_then(|i| section_data(&elf, obj_bytes, i))
    {
        Some(b) => b,
        None => {
            tracing::debug!("cast_analysis: inner ELF has no .BTF section");
            return (CastMap::new(), None, Vec::new());
        }
    };
    let btf = match Btf::from_bytes(btf_bytes) {
        Ok(b) => b,
        Err(e) => {
            tracing::warn!(
                error = ?e,
                "cast_analysis: parse .BTF failed"
            );
            return (CastMap::new(), None, Vec::new());
        }
    };
    let btf = Arc::new(btf);

    // Instruction sections in section-header order: every
    // SHF_EXECINSTR-flagged PROGBITS section. Concatenating in this
    // order matches how `.BTF.ext` records reference them — each
    // record's `insn_off` is byte-relative to its OWN section, so we
    // record each section's base index in the concatenated stream and
    // translate per-record below.
    // Pre-walk to size the concatenated instruction vec — saves a
    // sequence of growth-and-copy reallocations on schedulers with
    // large BPF programs (a single scx scheduler easily hits tens of
    // thousands of instructions). Each `chunks_exact(BPF_INSN_SIZE)`
    // pass below pushes `data.len() / BPF_INSN_SIZE` instructions.
    let total_insns: usize = elf
        .section_headers
        .iter()
        .enumerate()
        .filter(|(_, sh)| {
            sh.sh_type == goblin::elf::section_header::SHT_PROGBITS
                && sh.sh_flags & u64::from(goblin::elf::section_header::SHF_EXECINSTR) != 0
        })
        .filter_map(|(idx, _)| section_data(&elf, obj_bytes, idx))
        .filter(|d| d.len().is_multiple_of(BPF_INSN_SIZE))
        .map(|d| d.len() / BPF_INSN_SIZE)
        .sum();
    let mut text_concat: Vec<BpfInsn> = Vec::with_capacity(total_insns);
    let mut section_bases: HashMap<u32, usize> = HashMap::new();
    for (idx, sh) in elf.section_headers.iter().enumerate() {
        if sh.sh_type != goblin::elf::section_header::SHT_PROGBITS {
            continue;
        }
        if sh.sh_flags & u64::from(goblin::elf::section_header::SHF_EXECINSTR) == 0 {
            continue;
        }
        let Some(data) = section_data(&elf, obj_bytes, idx) else {
            continue;
        };
        if data.len() % BPF_INSN_SIZE != 0 {
            // Non-multiple-of-8 program section: malformed for BPF
            // bytecode. Skip rather than try to decode partial slots.
            continue;
        }
        let base = text_concat.len();
        for chunk in data.chunks_exact(BPF_INSN_SIZE) {
            let mut buf = [0u8; BPF_INSN_SIZE];
            buf.copy_from_slice(chunk);
            text_concat.push(BpfInsn::from_le_bytes(buf));
        }
        section_bases.insert(idx as u32, base);
    }
    if text_concat.is_empty() {
        tracing::debug!("cast_analysis: inner ELF has no executable BPF program sections");
        // Even on empty text we still return the parsed `Btf` so
        // the cross-BTF Fwd index can pick up its struct/union
        // definitions: a header-only object that contributes no
        // analyzer findings can still expose a complete sibling
        // for a Fwd in another object.
        return (CastMap::new(), Some(btf), Vec::new());
    }

    // .BTF.ext is optional — without it, every program function still
    // appears in the concatenated insn stream, but the analyzer cannot
    // reseed R1..R5 at function entries. Without entries the
    // analyzer cannot clear stale R6..R9 state at function
    // boundaries, which could produce false positives in theory
    // (stale typed pointer leaks via concatenation fall-through).
    // In practice all scx-built schedulers ship valid .BTF.ext.
    let func_entries = find_section(&elf, ".BTF.ext")
        .and_then(|i| section_data(&elf, obj_bytes, i))
        .map(|d| parse_btf_ext_func_entries(d, btf_bytes, &elf, &section_bases))
        .unwrap_or_default();

    // Pre-relocation .bpf.o files (the production path: an embedded
    // BPF object inside a scheduler binary that has not been through
    // libbpf's RELO_EXTERN_CALL handler yet) emit kfunc call sites
    // as `BPF_JMP|BPF_CALL` with `src_reg = BPF_PSEUDO_CALL = 1` and
    // `imm = -1`. The cast analyzer's `handle_kfunc_call` keys on
    // `src_reg = BPF_PSEUDO_KFUNC_CALL = 2` + `imm = btf_id`, so
    // every pre-relocation kfunc call is invisible to it. Patching
    // mirrors what libbpf does at load time
    // (`bpf_object__relocate_data`'s `RELO_EXTERN_CALL` arm):
    // walk the ELF relocation entries that target each program text
    // section, resolve the symbol name to a `BTF_KIND_FUNC` of
    // extern linkage in the program's own BTF, then rewrite both
    // `src_reg` and `imm` on the call instruction. After patching,
    // `analyze_casts` sees the kfunc id and `handle_kfunc_call`
    // recovers the return type — typically `Ptr -> Struct` for
    // pointer-returning kfuncs (`bpf_task_acquire`,
    // `bpf_cpumask_first`, …), which seeds R0 so the next STX of
    // R0 into a u64 slot records a `(parent, off) -> target,
    // AddrSpace::Kernel` cast entry.
    let patch_t0 = std::time::Instant::now();
    patch_kfunc_calls(&mut text_concat, btf.as_ref(), &elf, &section_bases);
    tracing::debug!(
        elapsed_us = patch_t0.elapsed().as_micros() as u64,
        insns = text_concat.len(),
        "cast_analysis: patch_kfunc_calls finished"
    );

    // BPF-to-BPF subprog call patching. libbpf-rs's Linker leaves
    // every global subprog call as `BPF_PSEUDO_CALL` with
    // `imm = -1`, paired with a `STT_FUNC` relocation. The cast
    // analyzer's `caller_arg_types` mechanism (see
    // [`crate::monitor::cast_analysis::Analyzer::analyze`])
    // computes `callee_pc = pc + 1 + insn.imm`, so an unpatched
    // `imm == -1` resolves to `pc` (the call site itself) and
    // poisons the lookup table with bogus entries. Patching
    // mirrors what libbpf does at load time
    // (`bpf_object__reloc_code` in tools/lib/bpf/libbpf.c):
    // `sub_insn_idx = sym.st_value/8 + insn.imm + 1`, with
    // `insn.imm = -1` for the global-subprog case. We rewrite
    // the placeholder `imm` in place so the analyzer's
    // `pc + 1 + imm` computation lands on the correct callee
    // entry PC in the concatenated text stream.
    let subprog_patch_t0 = std::time::Instant::now();
    patch_subprog_calls(&mut text_concat, &elf, &section_bases);
    tracing::debug!(
        elapsed_us = subprog_patch_t0.elapsed().as_micros() as u64,
        insns = text_concat.len(),
        "cast_analysis: patch_subprog_calls finished"
    );

    // BSS / DATA / RODATA datasec annotations: walk every
    // relocation section in the inner ELF and emit a
    // `DatasecPointer` per `R_BPF_64_64` reloc that targets a
    // section the program BTF exposes as a `BTF_KIND_DATASEC`.
    // The annotation gives the analyzer's `BPF_LD_IMM64` arm the
    // missing `(datasec_id, base_offset)` pair: libbpf's runtime
    // relocator would set `src_reg = BPF_PSEUDO_MAP_VALUE` and
    // patch the imm into a map fd, but the host-side cast loader
    // sees pre-relocation bytecode where the imm is the per-
    // variable byte offset within the section. We translate that
    // directly into the analyzer's `RegState::DatasecPointer`
    // representation so subsequent STX/LDX through the LD_IMM64
    // destination resolve to the right `VarSecinfo` entry via
    // `struct_member_at`.
    let datasec_t0 = std::time::Instant::now();
    let datasec_pointers = build_datasec_pointers(&text_concat, btf.as_ref(), &elf, &section_bases);
    tracing::debug!(
        elapsed_us = datasec_t0.elapsed().as_micros() as u64,
        datasec_pointers = datasec_pointers.len(),
        "cast_analysis: build_datasec_pointers finished"
    );

    // Allocator-return seeds: walk every relocation section to find
    // `BPF_PSEUDO_CALL` sites whose resolved subprog name matches
    // the arena-allocator allowlist (e.g. `scx_static_alloc_internal`).
    // Emit one [`SubprogReturn`] per matching call site so the
    // analyzer's `BPF_OP_CALL` arm tags R0 as
    // [`RegState::ArenaU64FromAlloc`] after the standard R0..=R5
    // clobber. The subsequent STX of the tagged R0 (or its
    // propagation through MOV / stack spill / LDX of an
    // already-arena-tagged slot) records `(parent, off)` as an
    // Arena cast finding via the new STX-flow path. See
    // [`build_subprog_returns`] for the relocation walk.
    let alloc_seed_t0 = std::time::Instant::now();
    let subprog_returns = build_subprog_returns(&text_concat, &elf, &section_bases);
    tracing::debug!(
        elapsed_us = alloc_seed_t0.elapsed().as_micros() as u64,
        subprog_returns = subprog_returns.len(),
        "cast_analysis: build_subprog_returns finished"
    );

    let analyze_t0 = std::time::Instant::now();
    let result = analyze_casts(
        &text_concat,
        btf.as_ref(),
        &[],
        &func_entries,
        &datasec_pointers,
        &subprog_returns,
    );
    tracing::debug!(
        elapsed_ms = analyze_t0.elapsed().as_millis() as u64,
        casts = result.len(),
        "cast_analysis: analyze_casts inner pass finished"
    );
    let mut alloc_sizes: Vec<u64> = subprog_returns
        .iter()
        .filter_map(|sr| sr.alloc_size)
        .collect();
    alloc_sizes.sort_unstable();
    alloc_sizes.dedup();
    (result, Some(btf), alloc_sizes)
}


mod reloc;
// Re-export the relocation/patch/parse layer so the staying loader code
// calls these helpers by bare name and the test module reaches them via
// `use super::super::*`. Some items are consumed only by `cfg(test)`, so
// the lib build sees part of this glob as unused.
#[allow(unused_imports)]
pub(crate) use reloc::*;

/// Find a section by exact name. Returns the section index, or `None`
/// if no section matches. Uses `shdr_strtab.get_at` directly to avoid
/// pulling section data when only the index is needed.
fn find_section(elf: &goblin::elf::Elf<'_>, name: &str) -> Option<usize> {
    for (i, sh) in elf.section_headers.iter().enumerate() {
        if let Some(n) = elf.shdr_strtab.get_at(sh.sh_name)
            && n == name
        {
            return Some(i);
        }
    }
    None
}

/// Get the byte slice covering a section's `[sh_offset, sh_offset +
/// sh_size)` range. Returns `None` if the range is out of bounds (a
/// malformed ELF whose section header points past file end).
fn section_data<'a>(
    elf: &goblin::elf::Elf<'_>,
    file_bytes: &'a [u8],
    idx: usize,
) -> Option<&'a [u8]> {
    let sh = elf.section_headers.get(idx)?;
    let start = sh.sh_offset as usize;
    let end = start.checked_add(sh.sh_size as usize)?;
    file_bytes.get(start..end)
}

mod persist;

#[cfg(test)]
mod tests;
