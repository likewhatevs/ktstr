//! BTF-resolved byte offsets within the sdt_alloc structures.
//!
//! `SdtAllocOffsets::from_btf` reads the scheduler's program BTF (not
//! vmlinux) and produces the byte offsets the walker needs to navigate
//! `struct scx_allocator` / `sdt_pool` / `sdt_desc` / `sdt_chunk` /
//! `sdt_data` from frozen guest memory. Handles `BTF_KIND_FWD` forward
//! declarations: `sdt_chunk` and `sdt_data` are tolerated as
//! forward-only (the walker fills in fallbacks from
//! `lib/sdt_task_defs.h` invariants); the other three must surface as
//! full struct definitions.

use anyhow::{Context, Result};
use btf_rs::Btf;

use crate::monitor::btf_offsets::{StructOrFwd, find_struct_or_fwd, member_byte_offset};

use super::SIZEOF_SDT_ID;

/// Byte offsets within the sdt_alloc data structures.
///
/// All resolved from the SCHEDULER'S program BTF (not vmlinux), since
/// `struct scx_allocator`, `struct sdt_pool`, `struct sdt_desc`, and
/// `struct sdt_chunk` are linked into the BPF program from
/// `lib/sdt_alloc.bpf.c` and never appear in vmlinux BTF.
#[derive(Debug, Clone)]
pub struct SdtAllocOffsets {
    /// Offset of `pool` (sdt_pool) within `struct scx_allocator`.
    pub allocator_pool: usize,
    /// Offset of `root` (sdt_desc_t *) within `struct scx_allocator`.
    pub allocator_root: usize,
    /// Total size of `struct scx_allocator`. Used by callers that
    /// need to bound a slice read of the in-bss allocator image.
    pub allocator_size: usize,
    /// Offset of `elem_size` (u64) within `struct sdt_pool`.
    pub pool_elem_size: usize,
    /// Offset of `allocated` ([u64; 8]) within `struct sdt_desc`.
    pub desc_allocated: usize,
    /// Offset of `nr_free` (u64) within `struct sdt_desc`.
    pub desc_nr_free: usize,
    /// Offset of `chunk` (`struct sdt_chunk *`) within `struct sdt_desc`.
    pub desc_chunk: usize,
    /// Offset of the union (`descs`/`data`) within `struct sdt_chunk`.
    /// Both interpretations alias at the same offset (it's a union).
    pub chunk_union: usize,
    /// Total size of `struct sdt_data` (header + zero-length payload[]).
    /// Equals 8 on all known kernels (the size of `union sdt_id`)
    /// because the flexible `payload[]` array adds no bytes to the
    /// struct's size.
    pub data_header_size: usize,
}

impl SdtAllocOffsets {
    /// Resolve sdt_alloc struct offsets from a pre-loaded program BTF.
    ///
    /// Returns `Err` when the program BTF lacks any of the required
    /// types — e.g. a scheduler that doesn't link `lib/sdt_alloc.bpf.c`
    /// into its BPF object. The dump pipeline treats this as "no
    /// sdt_alloc state to surface" and skips the walk silently rather
    /// than aborting, since not every scheduler uses the allocator.
    ///
    /// # `BTF_KIND_FWD` handling
    ///
    /// BPF program BTFs emit `BTF_KIND_FWD` (forward declaration, no
    /// body) for any struct the program references only by pointer.
    /// The four "structural" types — `scx_allocator`, `sdt_pool`,
    /// `sdt_desc`, `sdt_chunk` — must surface as full struct
    /// definitions: the walker derives member offsets from each, and
    /// a forward declaration carries no member information. A Fwd for
    /// any of those four is surfaced as `Err` so the dump pipeline
    /// records a clear diagnostic instead of crashing on missing
    /// members.
    ///
    /// `sdt_data` is the exception: lavd and other schedulers that
    /// only consume opaque allocator-returned pointers emit `sdt_data`
    /// as a `BTF_KIND_FWD`. The walker only needs the size of the
    /// header (the leading `union sdt_id`, 8 bytes; the `payload[]`
    /// flex-array contributes 0), so [`SIZEOF_SDT_ID`] is used as the
    /// fallback when `sdt_data` is a Fwd. The kernel header
    /// `lib/sdt_task_defs.h` makes this size invariant (it's the only
    /// non-flex-array member), so the fallback is correct without BTF
    /// involvement.
    ///
    /// # Panics
    ///
    /// `Struct::size()` calls below panic if the matched
    /// `Type::Struct(...)` record has a kind-vs-size-field skew —
    /// structurally invalid BTF that the producer (BPF object's
    /// program BTF) emitted. Malformed program BTF is a build-time
    /// bug in the scheduler's BPF, not a runtime recovery case;
    /// the panic points the operator at the BPF producer.
    pub fn from_btf(btf: &Btf) -> Result<Self> {
        let allocator = require_full_struct(btf, "scx_allocator").context(
            "btf: struct scx_allocator unavailable (scheduler doesn't link sdt_alloc, or BTF only carries a forward declaration)"
        )?;
        let allocator_pool = member_byte_offset(btf, &allocator, "pool")?;
        let allocator_root = member_byte_offset(btf, &allocator, "root")?;
        let allocator_size = allocator.size();

        let pool = require_full_struct(btf, "sdt_pool")
            .context("btf: struct sdt_pool unavailable for member offsets")?;
        let pool_elem_size = member_byte_offset(btf, &pool, "elem_size")?;

        let desc = require_full_struct(btf, "sdt_desc")
            .context("btf: struct sdt_desc unavailable for member offsets")?;
        let desc_allocated = member_byte_offset(btf, &desc, "allocated")?;
        let desc_nr_free = member_byte_offset(btf, &desc, "nr_free")?;
        let desc_chunk = member_byte_offset(btf, &desc, "chunk")?;

        // sdt_chunk is another type schedulers commonly emit as
        // BTF_KIND_FWD — it's only accessed internally by the
        // sdt_alloc library helpers. The struct contains a single
        // anonymous union at offset 0 (descs[] for internal nodes,
        // data[] for leaves). When only a Fwd is available, hardcode
        // chunk_union = 0 matching the kernel layout at
        // lib/sdt_task_defs.h.
        let chunk_union = match find_struct_or_fwd(btf, "sdt_chunk")
            .context("btf: struct sdt_chunk not found")?
        {
            StructOrFwd::Full(chunk) => chunk_union_offset(btf, &chunk)?,
            StructOrFwd::Fwd => 0,
        };

        // `sdt_data` is the one type the walker tolerates as a
        // `BTF_KIND_FWD`. The scheduler program never accesses its
        // members directly (the lib/sdt_alloc.bpf.c helpers do, in lib
        // BTF that may not be linked into the program BTF), so lavd
        // and similar schedulers emit it as a forward declaration.
        // The size is fixed by `lib/sdt_task_defs.h` at 8 bytes (the
        // `union sdt_id` header; `payload[]` is a flex-array
        // contributing 0 bytes), so we fall back to [`SIZEOF_SDT_ID`]
        // when the body is absent.
        let data_header_size =
            match find_struct_or_fwd(btf, "sdt_data").context("btf: struct sdt_data not found")? {
                StructOrFwd::Full(data) => data.size(),
                StructOrFwd::Fwd => SIZEOF_SDT_ID,
            };

        Ok(Self {
            allocator_pool,
            allocator_root,
            allocator_size,
            pool_elem_size,
            desc_allocated,
            desc_nr_free,
            desc_chunk,
            chunk_union,
            data_header_size,
        })
    }
}

/// Resolve a struct that the walker requires by member offset.
///
/// Wraps [`find_struct_or_fwd`] and rejects forward declarations with
/// an explicit "fwd, no body" diagnostic — distinct from "not found at
/// all" so an operator can tell whether the scheduler links sdt_alloc
/// at all (Err: not found) versus whether the program BTF stripped the
/// struct body (Err: fwd only). Returning a Fwd from this helper would
/// mean propagating an unusable struct handle whose `member_byte_offset`
/// calls would then fail with a misleading "field 'X' not found" error.
fn require_full_struct(btf: &Btf, name: &str) -> Result<btf_rs::Struct> {
    match find_struct_or_fwd(btf, name)? {
        StructOrFwd::Full(s) => Ok(s),
        StructOrFwd::Fwd => anyhow::bail!(
            "btf: struct {name} present only as BTF_KIND_FWD forward declaration; member offsets unavailable"
        ),
    }
}

/// Locate the union member offset within `struct sdt_chunk`.
///
/// The chunk struct is `struct { union { sdt_desc *descs[512];
/// sdt_data *data[512]; } }` — both arms occupy the same byte range.
/// Searching for either name returns the same offset; we accept the
/// first that resolves so a future rename of one arm doesn't cause
/// a hard failure when the other arm still exists.
fn chunk_union_offset(btf: &Btf, chunk: &btf_rs::Struct) -> Result<usize> {
    if let Ok(off) = member_byte_offset(btf, chunk, "descs") {
        return Ok(off);
    }
    if let Ok(off) = member_byte_offset(btf, chunk, "data") {
        return Ok(off);
    }
    anyhow::bail!("btf: struct sdt_chunk has neither `descs` nor `data` member")
}
