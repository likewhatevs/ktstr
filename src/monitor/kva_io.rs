//! Shared KVA-chunked I/O primitive.
//!
//! [`chunked_kva_io`] walks a contiguous KVA range one 4 KiB page at
//! a time: it translates each page's KVA to a guest PA via a
//! caller-supplied translator and hands the resolved PA plus the
//! chunk window to a caller closure that performs the actual DRAM
//! read or write.
//!
//! v0 consumer: [`super::bpf_map`] (read/write of BPF map values
//! living in vmalloc'd memory) via the local wrapper at
//! `bpf_map::chunked_kva_io`. The translator parameter abstracts WHO
//! does the KVA→PA lookup: that wrapper passes the BPF accessor's
//! `(cr3_pa, l5, tcr_el1, mem)` quadruple through
//! [`super::reader::GuestMem::translate_kva`]; the helper itself
//! stays oblivious.

/// Page chunk size for the KVA walk. 4 KiB is the smallest page
/// granule shared across x86-64 + aarch64 (4 KiB / 16 KiB / 64 KiB
/// granules); chunking at 4 KiB never straddles a leaf PTE on any
/// supported guest configuration. A coarser chunk would risk walking
/// past one PTE in a single bulk copy on a 4 KiB-granule kernel.
pub(crate) const PAGE_CHUNK: u64 = 4096;

// The chunk-boundary mask trick at the bottom of `chunked_kva_io`
// (`kva & !(PAGE_CHUNK - 1)`) only computes a correct next-page
// boundary when `PAGE_CHUNK` is a power of two. Pin the invariant
// at compile time so a future tweak to the constant fails to
// build instead of silently corrupting the chunking math.
const _: () = assert!(PAGE_CHUNK.is_power_of_two());

/// Copy a contiguous byte range to or from a kernel virtual address
/// range, chunking at 4 KiB page boundaries so each chunk takes one
/// `translate` call plus one bulk DRAM copy.
///
/// This replaces byte-by-byte loops that would issue one translate
/// per byte — a 4 KiB value read translated 4096 times and paid 4096
/// copy_nonoverlapping-of-one-byte calls. A full page now takes one
/// translate + one bulk copy (up to [`PAGE_CHUNK`] bytes); a range
/// that crosses a page boundary splits into N translate+copy pairs
/// where N is the number of pages touched.
///
/// `translate` resolves a KVA to its guest PA, returning `None` when
/// the page is not mapped. `target_kva` is the starting guest virtual
/// address; `len` is the total length. `chunk_fn` receives the
/// resolved guest PA, the offset of this chunk from the start of the
/// payload (`src_off`), and the length of this chunk (`chunk_len`).
/// The closure performs the actual memcpy.
///
/// Returns `false` when any chunk fails to translate. The chunk
/// closure is invoked for every chunk up to and including the one
/// preceding the failure; the failing chunk itself is NOT invoked
/// (translate runs first, and short-circuits before chunk_fn). The
/// caller is responsible for tracking partial-completion state
/// (e.g. by counting bytes inside the chunk closure and comparing
/// against `len` after the call).
///
/// **Caller-side invariant**: `target_kva.checked_add(len as u64)`
/// must be `Some`. The loop's page-boundary math
/// (`(kva & !(PAGE_CHUNK - 1)) + PAGE_CHUNK`) wraps silently in
/// release builds if the range extends past `u64::MAX`. Today's
/// kernel KVAs live well below that threshold (vmalloc tops out at
/// `0xffff_e8ff_ffff_ffff` on x86_64 L4 and analogous on L5/aarch64),
/// so the invariant is trivially satisfied by the existing BPF map
/// consumer + every realistic kernel-symbol target. Callers writing
/// non-kernel KVAs must check.
pub(crate) fn chunked_kva_io<T, F>(
    translate: T,
    target_kva: u64,
    len: usize,
    mut chunk_fn: F,
) -> bool
where
    T: Fn(u64) -> Option<u64>,
    F: FnMut(u64, u64, usize),
{
    let mut consumed: u64 = 0;
    let total = len as u64;
    while consumed < total {
        let kva = target_kva + consumed;
        let Some(pa) = translate(kva) else {
            return false;
        };
        // Advance at most to the next page boundary so the next
        // translate call lands on a fresh resolved page.
        let page_end = (kva & !(PAGE_CHUNK - 1)) + PAGE_CHUNK;
        let chunk_len = (page_end - kva).min(total - consumed) as usize;
        chunk_fn(pa, consumed, chunk_len);
        consumed += chunk_len as u64;
    }
    true
}
