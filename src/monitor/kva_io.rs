//! Shared KVA-chunked I/O primitive.
//!
//! [`chunked_kva_io`] walks a contiguous KVA range one 4 KiB page at
//! a time: it translates each page's KVA to a guest PA via a
//! caller-supplied translator and hands the resolved PA plus the
//! chunk window to a caller closure that performs the actual DRAM
//! read or write.
//!
//! Consumers:
//! - [`super::bpf_map`] (BPF map value read/write in vmalloc'd
//!   memory), via the local wrapper at `bpf_map::chunked_kva_io`
//!   that supplies the BPF accessor's `(cr3_pa, l5, tcr_el1, mem)`
//!   quadruple through [`super::reader::GuestMem::translate_kva`].
//! - [`super::guest::GuestKernel::write_kva_bytes_chunked`]
//!   (vmalloc'd KVA writes), which supplies the cached translator
//!   `translate_kva_cached` directly.
//!
//! The translator parameter abstracts WHO does the KVA→PA lookup;
//! the helper itself stays oblivious.

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    /// Capture every `(pa, src_off, chunk_len)` triple the helper hands
    /// back, so each test can assert the exact chunking decisions made.
    /// Pairs with a translator closure that the test configures per-case.
    fn run_with_capture<T>(
        translate: T,
        target_kva: u64,
        len: usize,
    ) -> (bool, Vec<(u64, u64, usize)>)
    where
        T: Fn(u64) -> Option<u64>,
    {
        let captured = RefCell::new(Vec::new());
        let ok = chunked_kva_io(translate, target_kva, len, |pa, src_off, chunk_len| {
            captured.borrow_mut().push((pa, src_off, chunk_len));
        });
        (ok, captured.into_inner())
    }

    #[test]
    fn chunked_kva_io_single_page_range_one_translate_one_chunk() {
        let translate_calls = RefCell::new(0u32);
        let (ok, chunks) = run_with_capture(
            |kva| {
                *translate_calls.borrow_mut() += 1;
                Some(kva ^ 0xFFFF_FFFF_F000_0000) // arbitrary PA mapping
            },
            0x1000, // page-aligned KVA
            512,    // < PAGE_CHUNK, all in one page
        );
        assert!(ok);
        assert_eq!(*translate_calls.borrow(), 1);
        assert_eq!(chunks.len(), 1);
        let (_pa, src_off, chunk_len) = chunks[0];
        assert_eq!(src_off, 0);
        assert_eq!(chunk_len, 512);
    }

    #[test]
    fn chunked_kva_io_multi_page_range_n_translate_n_chunks() {
        let translate_calls = RefCell::new(0u32);
        let (ok, chunks) = run_with_capture(
            |kva| {
                *translate_calls.borrow_mut() += 1;
                Some(kva.wrapping_sub(0x1_0000)) // any deterministic mapping
            },
            0x1000,                                // page-aligned start
            (PAGE_CHUNK as usize) * 2 + PAGE_CHUNK as usize / 2, // 2.5 pages
        );
        assert!(ok);
        assert_eq!(*translate_calls.borrow(), 3);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].2, PAGE_CHUNK as usize);
        assert_eq!(chunks[1].2, PAGE_CHUNK as usize);
        assert_eq!(chunks[2].2, PAGE_CHUNK as usize / 2);
        // src_off progresses by chunk length
        assert_eq!(chunks[0].1, 0);
        assert_eq!(chunks[1].1, PAGE_CHUNK);
        assert_eq!(chunks[2].1, PAGE_CHUNK * 2);
    }

    #[test]
    fn chunked_kva_io_unaligned_start_kva_chunks_to_first_page_boundary() {
        let offset_in_page: u64 = 0x100;
        let total: u64 = PAGE_CHUNK + 0x200; // straddles 2 pages
        let (ok, chunks) = run_with_capture(
            |_kva| Some(0xdead_0000),
            PAGE_CHUNK + offset_in_page, // unaligned start
            total as usize,
        );
        assert!(ok);
        assert_eq!(chunks.len(), 2);
        // First chunk fills to the next page boundary.
        assert_eq!(chunks[0].2 as u64, PAGE_CHUNK - offset_in_page);
        // Second chunk holds the remainder.
        assert_eq!(chunks[1].2 as u64, total - (PAGE_CHUNK - offset_in_page));
    }

    #[test]
    fn chunked_kva_io_translate_fails_mid_range_returns_false_short_chunks() {
        let translate_calls = RefCell::new(0u32);
        let (ok, chunks) = run_with_capture(
            |_kva| {
                let mut n = translate_calls.borrow_mut();
                *n += 1;
                if *n == 1 { Some(0x1000) } else { None }
            },
            0x1000,
            (PAGE_CHUNK as usize) * 2,
        );
        assert!(!ok);
        // chunk_fn fires once for the successful translate; the second
        // page's None short-circuits before chunk_fn runs.
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].1, 0);
        assert_eq!(chunks[0].2, PAGE_CHUNK as usize);
    }

    #[test]
    fn chunked_kva_io_zero_length_returns_true_no_chunk_calls() {
        let translate_calls = RefCell::new(0u32);
        let (ok, chunks) = run_with_capture(
            |_kva| {
                *translate_calls.borrow_mut() += 1;
                Some(0)
            },
            0xdead_beef,
            0,
        );
        assert!(ok);
        assert_eq!(*translate_calls.borrow(), 0);
        assert!(chunks.is_empty());
    }

    #[test]
    fn chunked_kva_io_translator_always_none_returns_false_no_chunks() {
        let (ok, chunks) = run_with_capture(|_kva| None, 0x1000, 4096);
        assert!(!ok);
        assert!(chunks.is_empty());
    }

    #[test]
    fn chunked_kva_io_chunk_at_last_byte_of_page_correctly_splits() {
        // len = PAGE_CHUNK + 1 at page-aligned start: first chunk fills
        // the page, second chunk carries exactly one byte.
        let (ok, chunks) = run_with_capture(|_kva| Some(0), 0x1000, PAGE_CHUNK as usize + 1);
        assert!(ok);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].2, PAGE_CHUNK as usize);
        assert_eq!(chunks[1].2, 1);
    }
}
