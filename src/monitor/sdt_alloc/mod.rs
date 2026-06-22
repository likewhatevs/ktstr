//! Host-side `sdt_alloc` radix-tree walker.
//!
//! `sdt_alloc` is the per-task / per-cgroup arena allocator that ships
//! in the upstream scx tree at `lib/sdt_alloc.bpf.c` (and friends in
//! `lib/sdt_task_defs.h`). Schedulers that opt into it allocate
//! per-entity contexts out of BPF arena memory, addressed via a
//! 3-level radix tree rooted at `scx_allocator.root`. The kernel
//! never exposes this layout to userspace — at freeze time the host
//! has only the raw arena page snapshot from [`super::arena`], which
//! is page-granular and structurally opaque.
//!
//! This module walks the same tree the scheduler walks, from frozen
//! guest memory, and produces structured per-allocation records that
//! the BTF renderer can turn into named field views. The result lands
//! in [`super::dump::FailureDumpReport::sdt_allocations`], distinct
//! from the page-granular snapshot so consumers can read either
//! representation.
//!
//! # Tree shape
//!
//! From `lib/sdt_task_defs.h`:
//!
//! ```text
//! scx_allocator { sdt_pool pool; sdt_desc_t *root; }
//! sdt_pool      { void *slab; u64 elem_size; u64 max_elems; u64 idx; }
//! sdt_desc      { u64 allocated[8]; u64 nr_free; sdt_chunk *chunk; }
//! sdt_chunk     { union { sdt_desc *descs[512]; sdt_data *data[512]; } }
//! sdt_data      { union sdt_id tid; u64 payload[]; }
//! sdt_id        { s32 idx; s32 genn; }   /* 8 bytes */
//! ```
//!
//! Three levels (`SDT_TASK_LEVELS = 3`), 512 entries per chunk
//! (`1 << SDT_TASK_ENTS_PER_PAGE_SHIFT`). The `allocated[8]` bitmap
//! (512 bits / 64) tracks live slots at each level. Internal levels
//! (0, 1) reach the next descriptor via `chunk->descs[pos]`; the leaf
//! level (2) reaches the user-visible payload via `chunk->data[pos]`.
//!
//! `sdt_data.tid.idx` carries the 27-bit (3 × 9) packed index that
//! produced this slot, and `tid.genn` increments on recycle so a
//! consumer can detect ABA across allocations of the same idx.
//!
//! # Liveness
//!
//! At the leaf level (level 2), `allocated[]` is the source of truth:
//! a set bit means slot `pos` carries a live `sdt_data *` in
//! `chunk->data[pos]`. We use the bitmap there because `tid.idx` and
//! `pool.idx` are both unreliable — post-free `tid.idx` is reset to 0
//! (ambiguous with slot 0), and `pool.idx` is the pool's high-water
//! mark, not the live count. `chunk->data[pos]` is also nullable for
//! pristine slots that the pool never handed out — we skip those
//! silently.
//!
//! At internal levels (0 and 1) the bitmap semantics are inverted:
//! `lib/sdt_alloc.bpf.c` only sets a parent bit once a child becomes
//! FULL (`desc_find_empty` propagates the set up only while the
//! decremented `nr_free` stays at 0) and clears it once the child
//! transitions back from full (`mark_nodes_avail` only propagates the
//! clear up while the incremented `nr_free` is still 1). So a set bit
//! at an internal level means "the descendant subtree is full"; a
//! clear bit means "partially populated, empty, or never created."
//! The common case for any scheduler with N << 512^3 live tasks is
//! "all clear", so the bitmap is unusable for enumeration at internal
//! levels. The walker enumerates internal levels by pointer non-null
//! in `chunk->descs[]` instead — every populated subtree has a
//! non-NULL desc child stored at its `pos` (`desc_find_empty` writes
//! `desc_children[pos]` whenever it allocates a new chunk), and a
//! NULL pointer is a never-created subtree we skip silently.
//!
//! # Race window
//!
//! The freeze coordinator pauses every vCPU before this walker runs,
//! but `scx_alloc_free_idx` zeroes a slot's payload BEFORE clearing
//! the bitmap. A frozen snapshot captured between those two writes
//! sees a "live" bitmap bit but a zero-filled payload. We render the
//! zeros as a "zeroed slot" rather than try to detect mid-free — the
//! consumer can recognise all-zero payloads as the race.

mod offsets;
mod payload;
mod snapshot;
mod walker;

pub use offsets::SdtAllocOffsets;
pub use payload::discover_payload_btf_id;
pub use snapshot::{SdtAllocEntry, SdtAllocatorSnapshot};
pub use walker::walk_sdt_allocator;

/// Tree depth and per-chunk fan-out from `lib/sdt_task_defs.h`.
///
/// `SDT_TASK_LEVELS = 3` and `SDT_TASK_ENTS_PER_PAGE_SHIFT = 9`. The
/// 512-entry fan-out is hard-baked into the layout (chunk arrays are
/// declared as `[SDT_TASK_ENTS_PER_CHUNK]` at file scope), so a future
/// upstream change in fan-out would re-shape `struct sdt_chunk` itself
/// and the walker would surface the divergence as a missing-field BTF
/// resolution failure during offset lookup.
pub(super) const SDT_TASK_LEVELS: usize = 3;
const SDT_TASK_ENTS_PER_PAGE_SHIFT: u32 = 9;
pub(super) const SDT_TASK_ENTS_PER_CHUNK: usize = 1 << SDT_TASK_ENTS_PER_PAGE_SHIFT; // 512
pub(super) const SDT_TASK_CHUNK_BITMAP_U64S: usize = SDT_TASK_ENTS_PER_CHUNK / 64; // 8

/// Maximum number of leaf allocations the walker will surface in a
/// single dump.
///
/// A scheduler that allocated millions of per-task contexts would OOM
/// the host renderer if the result was unbounded; cap to 4096 entries
/// (mirrors `MAX_HASH_ENTRIES` in [`super::dump`]) and surface
/// truncation via [`SdtAllocatorSnapshot::truncated`].
pub const MAX_SDT_ALLOC_ENTRIES: usize = 4096;

/// Width of `union sdt_id` in bytes — 8: `s32 idx + s32 genn`.
///
/// The kernel layout in `lib/sdt_task_defs.h` makes this a hard part
/// of the wire format: `union sdt_id { s64 val; struct { s32 idx; s32
/// genn; }; }` is exactly 8 bytes, and `struct sdt_data { union sdt_id
/// tid; u64 payload[]; }` has no other non-flex-array member, so
/// `sizeof(struct sdt_data) == 8` for every kernel that ships
/// sdt_alloc. [`SdtAllocOffsets::from_btf`] uses this as the fallback
/// for `data_header_size` when the scheduler's program BTF surfaces
/// `sdt_data` as a `BTF_KIND_FWD` forward declaration (no struct body
/// from which `.size()` could be read); unit tests pin it against
/// upstream layout drift.
pub(super) const SIZEOF_SDT_ID: usize = 8;

/// Sanity cap on `pool.elem_size` (allocation slot stride) the walker
/// will trust.
///
/// `lib/sdt_alloc.bpf.c::pool_set_size` checks `data_size % 8 == 0`
/// and bails on zero; `scx_alloc_init` rounds up to 8 then ensures
/// the chunk fits in `PAGE_SIZE`. So a real `elem_size` is always
/// `[16, 4096]` for non-degenerate allocators (16-byte minimum =
/// `sizeof(sdt_data) + 8`-byte minimum payload after `round_up(...,
/// 8)`). A torn snapshot or an uninitialized struct could surface a
/// wild value; reject anything outside this range.
pub(super) const MIN_ELEM_SIZE: u64 = 16;
pub(super) const MAX_ELEM_SIZE: u64 = 4096;

/// Upper bound on the BTF type-id walk in [`discover_payload_btf_id`].
///
/// btf-rs has no "list all types" iterator, so the heuristic walks
/// type ids 1..N and probes each with `resolve_type_by_id`. BTF can
/// have sparse id gaps (a single unresolvable id does NOT mean the
/// table is exhausted), so we don't break on the first miss — we
/// `continue` and walk up to this cap. 100k is well above the largest
/// program-BTF type tables ktstr sees in practice (~10k for a complex
/// scheduler) while still keeping the worst-case probe cost bounded.
///
/// Shared with [`super::cast_analysis`]'s candidate-search id walk:
/// both probes use the same heuristic against the same per-program
/// BTFs, so a single ceiling keeps them aligned.
pub(crate) const MAX_BTF_ID_PROBE: u32 = 100_000;

/// Read a u64 at `offset` from a byte slice, returning None when the
/// read would overflow the slice. Little-endian to match the kernel
/// layout the bytes came from.
pub(super) fn read_u64_at(bytes: &[u8], offset: usize) -> Option<u64> {
    let end = offset.checked_add(8)?;
    let slice = bytes.get(offset..end)?;
    let mut buf = [0u8; 8];
    buf.copy_from_slice(slice);
    Some(u64::from_le_bytes(buf))
}

#[cfg(test)]
mod tests {
    use super::*;
    use btf_rs::Btf;

    use crate::monitor::btf_render::RenderedValue;

    #[test]
    fn read_u64_at_basic() {
        let bytes = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0xff];
        // LE: 0x0807060504030201.
        assert_eq!(read_u64_at(&bytes, 0), Some(0x0807060504030201));
        // Out of range.
        assert_eq!(read_u64_at(&bytes, 2), None);
        assert_eq!(read_u64_at(&bytes, 100), None);
    }

    #[test]
    fn read_u64_at_handles_offset_overflow() {
        // offset.checked_add(8) overflow returns None rather than
        // panicking.
        let bytes = [0u8; 16];
        assert_eq!(read_u64_at(&bytes, usize::MAX), None);
    }

    #[test]
    fn empty_snapshot_serde() {
        let snap = SdtAllocatorSnapshot::default();
        let json = serde_json::to_string(&snap).unwrap();
        // entries / truncated / payload_type_reason skipped when at
        // default (the conditional skip predicates).
        assert!(!json.contains("\"entries\""));
        assert!(!json.contains("\"truncated\""));
        assert!(!json.contains("\"payload_type_reason\""));
        // elem_size, allocator_name, target_type_id, and
        // skipped_subtrees are always emitted — zero values carry
        // diagnostic information that suppression would mask.
        assert!(json.contains("\"elem_size\":0"));
        assert!(json.contains("\"allocator_name\":\"\""));
        assert!(json.contains("\"skipped_subtrees\":0"));
    }

    #[test]
    fn populated_snapshot_roundtrip() {
        let snap = SdtAllocatorSnapshot {
            allocator_name: "scx_task_allocator".into(),
            entries: vec![SdtAllocEntry {
                idx: 7,
                genn: 1,
                user_addr: 0x1000,
                payload: RenderedValue::Bytes {
                    hex: "de ad be ef".into(),
                },
            }],
            truncated: false,
            skipped_subtrees: 2,
            elem_size: 24,
            target_type_id: 42,
            payload_type_reason: String::new(),
            all_slot_addrs: Vec::new(),
        };
        let json = serde_json::to_string(&snap).expect("serialize");
        let parsed: SdtAllocatorSnapshot = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.entries.len(), 1);
        assert_eq!(parsed.entries[0].idx, 7);
        assert_eq!(parsed.entries[0].genn, 1);
        assert_eq!(parsed.elem_size, 24);
        assert_eq!(parsed.target_type_id, 42);
        assert_eq!(parsed.skipped_subtrees, 2);
        assert_eq!(parsed.allocator_name, "scx_task_allocator");
    }

    #[test]
    fn truncated_flag_serialises() {
        let snap = SdtAllocatorSnapshot {
            allocator_name: "x".into(),
            entries: vec![],
            truncated: true,
            skipped_subtrees: 0,
            elem_size: 24,
            target_type_id: 0,
            payload_type_reason: String::new(),
            all_slot_addrs: Vec::new(),
        };
        let json = serde_json::to_string(&snap).unwrap();
        assert!(json.contains("\"truncated\":true"));
    }

    #[test]
    fn payload_type_reason_serialises_when_nonempty() {
        let snap = SdtAllocatorSnapshot {
            allocator_name: "x".into(),
            entries: vec![],
            truncated: false,
            skipped_subtrees: 0,
            elem_size: 24,
            target_type_id: 0,
            payload_type_reason: "no candidate of size 16".into(),
            all_slot_addrs: Vec::new(),
        };
        let json = serde_json::to_string(&snap).unwrap();
        assert!(json.contains("\"payload_type_reason\":\"no candidate of size 16\""));
    }

    #[test]
    fn constants_match_upstream_layout() {
        // Pin the per-chunk fan-out and bitmap shape against
        // `lib/sdt_task_defs.h`. A future upstream change in either
        // value would re-shape `struct sdt_chunk` and surface here.
        assert_eq!(SDT_TASK_LEVELS, 3);
        assert_eq!(SDT_TASK_ENTS_PER_PAGE_SHIFT, 9);
        assert_eq!(SDT_TASK_ENTS_PER_CHUNK, 512);
        assert_eq!(SDT_TASK_CHUNK_BITMAP_U64S, 8);
        assert_eq!(SIZEOF_SDT_ID, 8);
    }

    #[test]
    fn elem_size_bounds_match_kernel() {
        // Mirror `pool_set_size`'s `data_size % 8 == 0` and
        // PAGE_SIZE chunk-fit checks. MIN/MAX must allow every
        // valid scheduler-declared payload size. Wrapped in `const`
        // blocks so the asserts run at compile time — a future drift
        // surfaces as a build failure, not a deferred test failure.
        const {
            assert!(MIN_ELEM_SIZE >= 16);
        }
        const {
            assert!(MAX_ELEM_SIZE <= 4096);
        }
        const {
            assert!(MIN_ELEM_SIZE.is_multiple_of(8));
        }
    }

    #[test]
    fn entry_display_shows_idx_genn_user_addr() {
        let entry = SdtAllocEntry {
            idx: 7,
            genn: 1,
            user_addr: 0x1000,
            payload: RenderedValue::Uint {
                bits: 32,
                value: 42,
            },
        };
        let out = format!("{entry}");
        assert!(out.contains("idx=7"), "missing idx: {out}");
        assert!(out.contains("genn=1"), "missing genn: {out}");
        assert!(out.contains("user_addr=0x1000"), "missing user_addr: {out}");
        assert!(out.contains("payload=42"), "missing payload: {out}");
    }

    #[test]
    fn snapshot_display_shows_header_and_entries() {
        let snap = SdtAllocatorSnapshot {
            allocator_name: "scx_task_allocator".into(),
            entries: vec![SdtAllocEntry {
                idx: 7,
                genn: 1,
                user_addr: 0x1000,
                payload: RenderedValue::Uint {
                    bits: 32,
                    value: 42,
                },
            }],
            truncated: false,
            skipped_subtrees: 0,
            elem_size: 24,
            target_type_id: 42,
            payload_type_reason: String::new(),
            all_slot_addrs: Vec::new(),
        };
        let out = format!("{snap}");
        assert!(
            out.contains("sdt_alloc scx_task_allocator"),
            "missing header: {out}"
        );
        assert!(out.contains("elem_size=24"), "missing elem_size: {out}");
        assert!(
            out.contains("target_type_id=42"),
            "missing target_type_id: {out}"
        );
        assert!(out.contains("1 live"), "missing entry count: {out}");
        assert!(out.contains("42"), "missing entry payload: {out}");
    }

    #[test]
    fn snapshot_display_marks_truncated_and_skipped() {
        let snap = SdtAllocatorSnapshot {
            allocator_name: "x".into(),
            entries: vec![],
            truncated: true,
            skipped_subtrees: 5,
            elem_size: 24,
            target_type_id: 0,
            payload_type_reason: "no candidate of size 16".into(),
            all_slot_addrs: Vec::new(),
        };
        let out = format!("{snap}");
        assert!(out.contains("(truncated)"), "missing truncated: {out}");
        assert!(
            out.contains("(5 subtrees skipped)"),
            "missing skipped: {out}"
        );
        assert!(
            out.contains("reason=no candidate of size 16"),
            "missing reason: {out}"
        );
    }

    // -- discover_payload_btf_id ------------------------------------
    //
    // Pure-function tests that don't need a `GuestKernel`. The
    // `walk_sdt_allocator` walker is intentionally NOT covered by
    // unit tests — it requires a live GuestKernel reading frozen
    // VM memory, which is structural integration coverage owned by
    // the existing failure_dump_e2e harness. These tests exercise
    // the BTF heuristic and offset-resolver branches that don't
    // need a kernel handle.

    /// `payload_size == 0` short-circuits without probing any BTF
    /// type ids — the heuristic correctly recognises that an
    /// allocator with zero payload bytes has no struct to discover.
    /// Pinning the early-return shape against a future regression
    /// that might silently start probing on zero (which would
    /// produce a spurious "no candidate of size 0" diagnostic).
    #[test]
    fn discover_payload_btf_id_zero_size_short_circuits() {
        let path = match crate::monitor::find_test_vmlinux() {
            Some(p) => p,
            None => {
                crate::report::test_skip("no vmlinux for BTF load");
                return;
            }
        };
        let btf = match crate::monitor::btf_offsets::load_btf_from_path(&path) {
            Ok(b) => b,
            Err(_) => {
                crate::report::test_skip("BTF load failed");
                return;
            }
        };
        let choice = discover_payload_btf_id(&btf, 0, "");
        assert_eq!(
            choice.target_type_id, 0,
            "zero-size must yield target_type_id=0"
        );
        assert_eq!(
            choice.reason, "payload_size == 0",
            "zero-size reason must be the early-return marker, got: {}",
            choice.reason
        );
    }

    /// A payload size that no real kernel struct can possibly hit —
    /// `usize::MAX / 2` is far larger than any real struct
    /// (kernel `struct task_struct` is on the order of 10 KiB, the
    /// largest plausible kernel struct is well under a megabyte).
    /// Searching for that size yields zero candidates; assert the
    /// returned reason matches the documented "no candidate of size
    /// {N}" wording so a consumer reading `payload_type_reason` can
    /// rely on the exact format.
    #[test]
    fn discover_payload_btf_id_no_candidate_path() {
        let path = match crate::monitor::find_test_vmlinux() {
            Some(p) => p,
            None => {
                crate::report::test_skip("no vmlinux for BTF load");
                return;
            }
        };
        let btf = match crate::monitor::btf_offsets::load_btf_from_path(&path) {
            Ok(b) => b,
            Err(_) => {
                crate::report::test_skip("BTF load failed");
                return;
            }
        };
        let impossible_size = usize::MAX / 2;
        let choice = discover_payload_btf_id(&btf, impossible_size, "");
        assert_eq!(choice.target_type_id, 0);
        let expected = format!("no candidate of size {impossible_size}");
        assert_eq!(
            choice.reason, expected,
            "reason must exactly match documented format: got '{}'",
            choice.reason
        );
    }

    /// SdtAllocOffsets::from_btf returns Err when `struct
    /// scx_allocator` is absent from the BTF — vmlinux BTF never
    /// contains it (sdt_alloc lives in the scheduler's program BTF,
    /// not the kernel's), so a from_btf call against vmlinux must
    /// surface the expected error and not panic. The dump pipeline
    /// reads this Err to decide "no sdt_alloc state to surface."
    /// Pin the diagnostic-string contract since callers rely on it.
    #[test]
    fn sdt_alloc_offsets_from_vmlinux_btf_returns_err() {
        let path = match crate::monitor::find_test_vmlinux() {
            Some(p) => p,
            None => {
                crate::report::test_skip("no vmlinux for BTF load");
                return;
            }
        };
        let btf = match crate::monitor::btf_offsets::load_btf_from_path(&path) {
            Ok(b) => b,
            Err(_) => {
                crate::report::test_skip("BTF load failed");
                return;
            }
        };
        let err = SdtAllocOffsets::from_btf(&btf)
            .expect_err("vmlinux BTF must NOT contain scx_allocator — from_btf must Err");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("scx_allocator"),
            "error must name the missing struct so the dump pipeline can log a useful diagnostic: '{msg}'"
        );
    }

    // -- from_btf error paths ----------------------------------------
    //
    // The four error paths in [`SdtAllocOffsets::from_btf`] each
    // surface a distinct diagnostic:
    //
    //   1. `scx_allocator` missing → `"struct scx_allocator unavailable"`
    //      (covered by `sdt_alloc_offsets_from_vmlinux_btf_returns_err`)
    //   2. `sdt_pool` missing      → `"struct sdt_pool unavailable for member offsets"`
    //   3. `sdt_desc` missing      → `"struct sdt_desc unavailable for member offsets"`
    //   4. `sdt_chunk` missing     → `"struct sdt_chunk not found"`
    //
    // The fifth code path — `sdt_data` as `BTF_KIND_FWD` — must
    // succeed (lavd and similar schedulers emit `sdt_data` as a
    // forward declaration; the walker hardcodes 8 from
    // [`SIZEOF_SDT_ID`] when the body is absent).
    //
    // The tests below build minimal synthetic BTF blobs (mirroring
    // the per-test-module pattern in
    // `cast_analysis::tests::build_btf` and
    // `test_support::btf_blob::cast_build_btf` — pared down here to only
    // the kinds `from_btf` consults: BTF_KIND_INT, BTF_KIND_STRUCT,
    // BTF_KIND_FWD) and parse them via `Btf::from_bytes`. Synthetic
    // BTF makes the four error paths reachable deterministically
    // without requiring a real scheduler program BTF on disk.
    //
    // The constants and wire format mirror linux uapi `btf.h` and
    // `Documentation/bpf/btf.rst`. The `info` u32 layout: `kind`
    // in bits 24..29, `vlen` in low 16 bits, `kind_flag` in bit 31.

    const SDTA_BTF_MAGIC: u16 = 0xEB9F;
    const SDTA_BTF_VERSION: u8 = 1;
    const SDTA_BTF_HEADER_LEN: u32 = 24;
    const SDTA_BTF_KIND_INT: u32 = 1;
    const SDTA_BTF_KIND_STRUCT: u32 = 4;
    /// `BTF_KIND_FWD = 7`. Forward declaration. Carries a name but
    /// no body; `kind_flag` selects struct (0) vs union (1) per
    /// `btf-rs::Fwd::is_struct` / `is_union`.
    const SDTA_BTF_KIND_FWD: u32 = 7;

    /// One member of a synthetic `BTF_KIND_STRUCT`. The wire format
    /// stores `bit_offset` (member offset in BITS, not bytes); the
    /// test helper converts from `byte_offset` for readability.
    #[derive(Clone, Copy)]
    struct SdtaSynMember {
        name_off: u32,
        type_id: u32,
        byte_offset: u32,
    }

    /// One synthetic BTF type. The pared-down kind set (`Int`,
    /// `Struct`, `Fwd`) is exactly what `from_btf` needs to traverse:
    /// member offsets come from `Struct`s, the `sdt_data` Fwd path
    /// exercises the [`SIZEOF_SDT_ID`] fallback, and `Int` provides
    /// terminal type ids the struct members can reference.
    enum SdtaSynType {
        /// `BTF_KIND_INT`. `encoding=0` is plain unsigned (the form
        /// `u64` / `u32` resolve to in libbpf-emitted BTF).
        Int {
            name_off: u32,
            size: u32,
            encoding: u32,
            offset: u32,
            bits: u32,
        },
        /// `BTF_KIND_STRUCT` with `kind_flag=0` — non-bitfield
        /// members. `from_btf` only consumes byte-aligned member
        /// offsets via [`member_byte_offset`], so the simpler
        /// `kind_flag=0` form suffices.
        Struct {
            name_off: u32,
            size: u32,
            members: Vec<SdtaSynMember>,
        },
        /// `BTF_KIND_FWD` (struct flavour, `kind_flag=0`). Used by
        /// the `sdt_data` Fwd-fallback test — the only path
        /// `from_btf` accepts a Fwd on, since the header size is
        /// kernel-source-fixed at `SIZEOF_SDT_ID = 8`.
        Fwd { name_off: u32 },
    }

    /// Append a NUL-terminated string to the BTF strings buffer and
    /// return its byte offset. Same shape as
    /// `cast_analysis::tests::push_name`, kept private to this test
    /// module to avoid coupling between fixtures.
    fn sdta_push_name(s: &mut Vec<u8>, name: &str) -> u32 {
        let off = s.len() as u32;
        s.extend_from_slice(name.as_bytes());
        s.push(0);
        off
    }

    /// Build a minimal BTF byte blob from a list of synthetic types
    /// and a string section.
    ///
    /// Header layout matches `cast_analysis::tests::build_btf`:
    /// 24-byte header (magic + version + flags + hdr_len + type_off
    /// + type_len + str_off + str_len) followed by the type section
    ///   then the string section. Type ids start at 1 (id 0 is Void)
    ///   and increase in `types` order.
    fn sdta_build_btf(types: &[SdtaSynType], strings: &[u8]) -> Vec<u8> {
        let mut type_section: Vec<u8> = Vec::new();
        for ty in types {
            match ty {
                SdtaSynType::Int {
                    name_off,
                    size,
                    encoding,
                    offset,
                    bits,
                } => {
                    type_section.extend_from_slice(&name_off.to_le_bytes());
                    let info = (SDTA_BTF_KIND_INT << 24) & 0x1f00_0000;
                    type_section.extend_from_slice(&info.to_le_bytes());
                    type_section.extend_from_slice(&size.to_le_bytes());
                    let int_data = (*encoding << 24) | ((*offset & 0xff) << 16) | (*bits & 0xff);
                    type_section.extend_from_slice(&int_data.to_le_bytes());
                }
                SdtaSynType::Struct {
                    name_off,
                    size,
                    members,
                } => {
                    type_section.extend_from_slice(&name_off.to_le_bytes());
                    let vlen = members.len() as u32;
                    let info = ((SDTA_BTF_KIND_STRUCT << 24) & 0x1f00_0000) | (vlen & 0xffff);
                    type_section.extend_from_slice(&info.to_le_bytes());
                    type_section.extend_from_slice(&size.to_le_bytes());
                    for m in members {
                        type_section.extend_from_slice(&m.name_off.to_le_bytes());
                        type_section.extend_from_slice(&m.type_id.to_le_bytes());
                        // Non-bitfield struct: bit_offset = byte * 8.
                        let bit_off = m.byte_offset * 8;
                        type_section.extend_from_slice(&bit_off.to_le_bytes());
                    }
                }
                SdtaSynType::Fwd { name_off } => {
                    type_section.extend_from_slice(&name_off.to_le_bytes());
                    // BTF_KIND_FWD: vlen=0, kind_flag=0 (struct
                    // flavour). size_type field is unused but is
                    // still 4 bytes wide on the wire — emit 0.
                    let info = (SDTA_BTF_KIND_FWD << 24) & 0x1f00_0000;
                    type_section.extend_from_slice(&info.to_le_bytes());
                    type_section.extend_from_slice(&0u32.to_le_bytes());
                }
            }
        }

        let type_len = type_section.len() as u32;
        let str_len = strings.len() as u32;

        let mut blob: Vec<u8> = Vec::new();
        blob.extend_from_slice(&SDTA_BTF_MAGIC.to_le_bytes());
        blob.push(SDTA_BTF_VERSION);
        blob.push(0); // flags
        blob.extend_from_slice(&SDTA_BTF_HEADER_LEN.to_le_bytes());
        blob.extend_from_slice(&0u32.to_le_bytes()); // type_off
        blob.extend_from_slice(&type_len.to_le_bytes());
        blob.extend_from_slice(&type_len.to_le_bytes()); // str_off = type_len
        blob.extend_from_slice(&str_len.to_le_bytes());
        blob.extend_from_slice(&type_section);
        blob.extend_from_slice(strings);
        blob
    }

    /// Set of names every from_btf-error-path test needs in its
    /// string section. Shared so each test's setup stays focused on
    /// the type list, not the string table mechanics.
    ///
    /// Returns `(strings, name_offsets)` where `name_offsets` is a
    /// struct of byte offsets keyed by name.
    fn sdta_strings_for_from_btf() -> (Vec<u8>, SdtaNames) {
        let mut strings: Vec<u8> = vec![0];
        let n_u64 = sdta_push_name(&mut strings, "u64");
        let n_scx_allocator = sdta_push_name(&mut strings, "scx_allocator");
        let n_sdt_pool = sdta_push_name(&mut strings, "sdt_pool");
        let n_sdt_desc = sdta_push_name(&mut strings, "sdt_desc");
        let n_sdt_chunk = sdta_push_name(&mut strings, "sdt_chunk");
        let n_sdt_data = sdta_push_name(&mut strings, "sdt_data");
        let n_pool = sdta_push_name(&mut strings, "pool");
        let n_root = sdta_push_name(&mut strings, "root");
        let n_elem_size = sdta_push_name(&mut strings, "elem_size");
        let n_allocated = sdta_push_name(&mut strings, "allocated");
        let n_nr_free = sdta_push_name(&mut strings, "nr_free");
        let n_chunk = sdta_push_name(&mut strings, "chunk");
        let n_descs = sdta_push_name(&mut strings, "descs");
        (
            strings,
            SdtaNames {
                n_u64,
                n_scx_allocator,
                n_sdt_pool,
                n_sdt_desc,
                n_sdt_chunk,
                n_sdt_data,
                n_pool,
                n_root,
                n_elem_size,
                n_allocated,
                n_nr_free,
                n_chunk,
                n_descs,
            },
        )
    }

    /// Byte offsets within the string section for the names every
    /// from_btf-error-path test references. Bundled into a struct so
    /// each test's local state stays tidy and so the order of `let`
    /// bindings matches across tests (preventing accidental skews
    /// between tests that all reference the same name table).
    struct SdtaNames {
        n_u64: u32,
        n_scx_allocator: u32,
        n_sdt_pool: u32,
        n_sdt_desc: u32,
        n_sdt_chunk: u32,
        n_sdt_data: u32,
        n_pool: u32,
        n_root: u32,
        n_elem_size: u32,
        n_allocated: u32,
        n_nr_free: u32,
        n_chunk: u32,
        n_descs: u32,
    }

    /// Build the minimal `scx_allocator` struct every from_btf path
    /// must traverse before reaching the inner-struct lookups. Two
    /// members `pool` and `root`, both typed as `u64` (type_id=1)
    /// since `from_btf` only reads each member's byte offset, not
    /// its type. `pool` at offset 0, `root` at offset 8, total size
    /// 16 — matches the kernel's `struct scx_allocator { struct
    /// sdt_pool pool; sdt_desc_t *root; }` member ORDER (the actual
    /// kernel `pool` is itself a struct, but the synthetic version
    /// only needs a name + offset for [`member_byte_offset`] to
    /// succeed).
    fn sdta_allocator_struct(names: &SdtaNames) -> SdtaSynType {
        SdtaSynType::Struct {
            name_off: names.n_scx_allocator,
            size: 16,
            members: vec![
                SdtaSynMember {
                    name_off: names.n_pool,
                    type_id: 1,
                    byte_offset: 0,
                },
                SdtaSynMember {
                    name_off: names.n_root,
                    type_id: 1,
                    byte_offset: 8,
                },
            ],
        }
    }

    /// Test 1: `scx_allocator` is present but `sdt_pool` is missing
    /// from the BTF entirely. `from_btf` must surface the
    /// `"sdt_pool unavailable for member offsets"` context — the
    /// distinct diagnostic that lets the dump pipeline distinguish
    /// "no scheduler links sdt_alloc" (test
    /// `sdt_alloc_offsets_from_vmlinux_btf_returns_err`) from
    /// "scheduler links sdt_alloc but the BTF stripped sdt_pool".
    #[test]
    fn sdt_alloc_offsets_missing_sdt_pool_distinct_error() {
        let (strings, names) = sdta_strings_for_from_btf();
        let types = vec![
            // id=1: u64 plain unsigned. Used as the type for every
            // member of the synthetic structs (see
            // `sdta_allocator_struct`'s comment).
            SdtaSynType::Int {
                name_off: names.n_u64,
                size: 8,
                encoding: 0,
                offset: 0,
                bits: 64,
            },
            // id=2: scx_allocator (full struct).
            sdta_allocator_struct(&names),
            // id=3: sdt_desc (full struct, present so we don't
            // accidentally match its error path instead).
            SdtaSynType::Struct {
                name_off: names.n_sdt_desc,
                size: 24,
                members: vec![
                    SdtaSynMember {
                        name_off: names.n_allocated,
                        type_id: 1,
                        byte_offset: 0,
                    },
                    SdtaSynMember {
                        name_off: names.n_nr_free,
                        type_id: 1,
                        byte_offset: 8,
                    },
                    SdtaSynMember {
                        name_off: names.n_chunk,
                        type_id: 1,
                        byte_offset: 16,
                    },
                ],
            },
            // id=4: sdt_chunk (full struct).
            SdtaSynType::Struct {
                name_off: names.n_sdt_chunk,
                size: 8,
                members: vec![SdtaSynMember {
                    name_off: names.n_descs,
                    type_id: 1,
                    byte_offset: 0,
                }],
            },
            // id=5: sdt_data (Fwd, the form `from_btf` accepts).
            SdtaSynType::Fwd {
                name_off: names.n_sdt_data,
            },
            // sdt_pool is intentionally OMITTED — this is the path
            // under test.
        ];
        let blob = sdta_build_btf(&types, &strings);
        let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");

        let err =
            SdtAllocOffsets::from_btf(&btf).expect_err("missing sdt_pool must surface as Err");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("sdt_pool"),
            "error must name the missing struct: '{msg}'"
        );
        assert!(
            msg.contains("unavailable for member offsets"),
            "error must carry the sdt_pool-specific context distinguishing this from sdt_chunk's 'not found' wording: '{msg}'"
        );
        // The diagnostic must NOT name unrelated structs — the
        // error path is sdt_pool-specific. A regression that
        // reordered the require_full_struct calls would surface
        // `sdt_desc` in the message instead.
        assert!(
            !msg.contains("sdt_desc"),
            "missing-sdt_pool error must not reference sdt_desc: '{msg}'"
        );
        assert!(
            !msg.contains("sdt_chunk"),
            "missing-sdt_pool error must not reference sdt_chunk: '{msg}'"
        );
    }

    /// Test 2: `scx_allocator` and `sdt_pool` are present but
    /// `sdt_desc` is missing from the BTF. The error must reach
    /// `from_btf`'s `sdt_desc` lookup specifically — surfacing the
    /// `"sdt_desc unavailable for member offsets"` context — and
    /// not collapse onto an earlier failure.
    #[test]
    fn sdt_alloc_offsets_missing_sdt_desc_distinct_error() {
        let (strings, names) = sdta_strings_for_from_btf();
        let types = vec![
            // id=1: u64 plain unsigned.
            SdtaSynType::Int {
                name_off: names.n_u64,
                size: 8,
                encoding: 0,
                offset: 0,
                bits: 64,
            },
            // id=2: scx_allocator (full struct).
            sdta_allocator_struct(&names),
            // id=3: sdt_pool (full struct, present).
            SdtaSynType::Struct {
                name_off: names.n_sdt_pool,
                size: 32,
                members: vec![SdtaSynMember {
                    name_off: names.n_elem_size,
                    type_id: 1,
                    byte_offset: 16,
                }],
            },
            // id=4: sdt_chunk (full struct, present so we reach
            // sdt_desc before sdt_chunk).
            SdtaSynType::Struct {
                name_off: names.n_sdt_chunk,
                size: 8,
                members: vec![SdtaSynMember {
                    name_off: names.n_descs,
                    type_id: 1,
                    byte_offset: 0,
                }],
            },
            // id=5: sdt_data (Fwd).
            SdtaSynType::Fwd {
                name_off: names.n_sdt_data,
            },
            // sdt_desc is intentionally OMITTED.
        ];
        let blob = sdta_build_btf(&types, &strings);
        let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");

        let err =
            SdtAllocOffsets::from_btf(&btf).expect_err("missing sdt_desc must surface as Err");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("sdt_desc"),
            "error must name the missing struct: '{msg}'"
        );
        assert!(
            msg.contains("unavailable for member offsets"),
            "error must carry the sdt_desc-specific context distinguishing this from sdt_chunk's 'not found' wording: '{msg}'"
        );
        // sdt_pool must NOT appear — sdt_pool resolved successfully
        // before from_btf reached the sdt_desc lookup. A leak
        // would mean require_full_struct's context is misordered.
        assert!(
            !msg.contains("sdt_pool"),
            "missing-sdt_desc error must not reference sdt_pool: '{msg}'"
        );
        assert!(
            !msg.contains("sdt_chunk"),
            "missing-sdt_desc error must not reference sdt_chunk: '{msg}'"
        );
    }

    /// Test 3: `scx_allocator`, `sdt_pool`, `sdt_desc` all present
    /// but `sdt_chunk` is missing. The error must surface the
    /// distinct `"sdt_chunk not found"` context — sdt_chunk goes
    /// through [`find_struct_or_fwd`] (not `require_full_struct`),
    /// so its diagnostic wording differs from sdt_pool / sdt_desc.
    #[test]
    fn sdt_alloc_offsets_missing_sdt_chunk_distinct_error() {
        let (strings, names) = sdta_strings_for_from_btf();
        let types = vec![
            // id=1: u64 plain unsigned.
            SdtaSynType::Int {
                name_off: names.n_u64,
                size: 8,
                encoding: 0,
                offset: 0,
                bits: 64,
            },
            // id=2: scx_allocator (full struct).
            sdta_allocator_struct(&names),
            // id=3: sdt_pool (full struct).
            SdtaSynType::Struct {
                name_off: names.n_sdt_pool,
                size: 32,
                members: vec![SdtaSynMember {
                    name_off: names.n_elem_size,
                    type_id: 1,
                    byte_offset: 16,
                }],
            },
            // id=4: sdt_desc (full struct, present so sdt_chunk is
            // the failing lookup).
            SdtaSynType::Struct {
                name_off: names.n_sdt_desc,
                size: 24,
                members: vec![
                    SdtaSynMember {
                        name_off: names.n_allocated,
                        type_id: 1,
                        byte_offset: 0,
                    },
                    SdtaSynMember {
                        name_off: names.n_nr_free,
                        type_id: 1,
                        byte_offset: 8,
                    },
                    SdtaSynMember {
                        name_off: names.n_chunk,
                        type_id: 1,
                        byte_offset: 16,
                    },
                ],
            },
            // id=5: sdt_data (Fwd).
            SdtaSynType::Fwd {
                name_off: names.n_sdt_data,
            },
            // sdt_chunk is intentionally OMITTED.
        ];
        let blob = sdta_build_btf(&types, &strings);
        let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");

        let err =
            SdtAllocOffsets::from_btf(&btf).expect_err("missing sdt_chunk must surface as Err");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("sdt_chunk"),
            "error must name the missing struct: '{msg}'"
        );
        // sdt_chunk uses `find_struct_or_fwd` with the context
        // `"btf: struct sdt_chunk not found"`. The inner anyhow
        // error from `with_context` always carries `"type 'X' not
        // found"`, so a contains-"not found" check alone is also
        // satisfied by the sdt_pool / sdt_desc paths via their
        // inner context. The distinguishing OUTER-context check is
        // the absence of `"unavailable for member offsets"` — that
        // wording is sdt_pool / sdt_desc-specific.
        assert!(
            msg.contains("not found"),
            "sdt_chunk error must carry the find_struct_or_fwd 'not found' wording: '{msg}'"
        );
        assert!(
            !msg.contains("unavailable for member offsets"),
            "sdt_chunk uses find_struct_or_fwd, NOT require_full_struct — the 'unavailable for member offsets' phrase is sdt_pool / sdt_desc-specific and must not appear: '{msg}'"
        );
        assert!(
            !msg.contains("sdt_pool"),
            "missing-sdt_chunk error must not reference sdt_pool: '{msg}'"
        );
        assert!(
            !msg.contains("sdt_desc"),
            "missing-sdt_chunk error must not reference sdt_desc: '{msg}'"
        );
    }

    /// Test 4: every required type present and `sdt_data` emitted
    /// as a `BTF_KIND_FWD` forward declaration. `from_btf` must
    /// succeed and `data_header_size` must equal
    /// [`SIZEOF_SDT_ID`] = 8 — the kernel-header-fixed size of the
    /// `union sdt_id` header (the only non-flex-array member in
    /// `struct sdt_data`). This is the lavd-style scheduler path:
    /// the program never accesses `sdt_data` members directly so
    /// libbpf strips the body to a Fwd, and the walker covers leaf
    /// liveness via the leaf descriptor's `allocated[]` bitmap rather
    /// than the slot's own header content.
    #[test]
    fn sdt_alloc_offsets_sdt_data_fwd_uses_sizeof_sdt_id() {
        let (strings, names) = sdta_strings_for_from_btf();
        let types = vec![
            // id=1: u64 plain unsigned.
            SdtaSynType::Int {
                name_off: names.n_u64,
                size: 8,
                encoding: 0,
                offset: 0,
                bits: 64,
            },
            // id=2: scx_allocator (full struct).
            sdta_allocator_struct(&names),
            // id=3: sdt_pool (full struct).
            SdtaSynType::Struct {
                name_off: names.n_sdt_pool,
                size: 32,
                members: vec![SdtaSynMember {
                    name_off: names.n_elem_size,
                    type_id: 1,
                    byte_offset: 16,
                }],
            },
            // id=4: sdt_desc (full struct).
            SdtaSynType::Struct {
                name_off: names.n_sdt_desc,
                size: 24,
                members: vec![
                    SdtaSynMember {
                        name_off: names.n_allocated,
                        type_id: 1,
                        byte_offset: 0,
                    },
                    SdtaSynMember {
                        name_off: names.n_nr_free,
                        type_id: 1,
                        byte_offset: 8,
                    },
                    SdtaSynMember {
                        name_off: names.n_chunk,
                        type_id: 1,
                        byte_offset: 16,
                    },
                ],
            },
            // id=5: sdt_chunk (full struct with `descs` member at
            // offset 0 — matches the kernel layout's union at
            // offset 0).
            SdtaSynType::Struct {
                name_off: names.n_sdt_chunk,
                size: 8,
                members: vec![SdtaSynMember {
                    name_off: names.n_descs,
                    type_id: 1,
                    byte_offset: 0,
                }],
            },
            // id=6: sdt_data (Fwd) — the path under test. The
            // hardcoded fallback to SIZEOF_SDT_ID must fire.
            SdtaSynType::Fwd {
                name_off: names.n_sdt_data,
            },
        ];
        let blob = sdta_build_btf(&types, &strings);
        let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");

        let offsets = SdtAllocOffsets::from_btf(&btf)
            .expect("sdt_data Fwd must NOT cause from_btf to fail — Fwd is the lavd-style path");
        assert_eq!(
            offsets.data_header_size, SIZEOF_SDT_ID,
            "data_header_size for a Fwd sdt_data must fall back to SIZEOF_SDT_ID (=8, the union sdt_id header size that lib/sdt_task_defs.h fixes)"
        );
        assert_eq!(
            offsets.data_header_size, 8,
            "literal-8 cross-check: the Fwd fallback must equal exactly 8 bytes (kernel-header-fixed)"
        );
    }

    // -- discover_payload_btf_id heuristic branches ---------------
    //
    // The existing tests (`discover_payload_btf_id_zero_size_short_circuits`,
    // `discover_payload_btf_id_no_candidate_path`) cover only the
    // payload_size=0 short-circuit and the empty-size_matches path.
    // The heuristic's actual branching logic — single-match returns
    // id, multi-match falls through pattern arms (task_ctx exact →
    // *_arena_ctx → *_task_ctx → *_ctx suffix), per-arm ambiguity,
    // anonymous-struct rejection — is entirely uncovered. Without
    // these tests, an implementation that only handles the two
    // existing edge cases passes the suite while being broken for
    // every real per-cgroup or per-task allocator.
    //
    // Per-cgroup arena pointers (cgx_raw, llcx_raw) failing to
    // chase. Tests below cover `scx_cgroup_ctx`-style names
    // matching the `*_ctx` arm, the per-arm ambiguous fallback,
    // and the continue-to-next-arm path.
    //
    // All three tests use the existing `sdta_*` BTF builder helpers
    // declared above to avoid duplicating wire-format logic.

    /// Single size-match resolves cleanly. A BTF with one
    /// 16-byte struct named `cgrp_ctx` and one 8-byte int. Calling
    /// `discover_payload_btf_id(&btf, 16, "")` finds `cgrp_ctx`
    /// as the unique size-match; size_matches.len() == 1 routes to
    /// the "single match" arm and returns the id with empty reason.
    /// Pin the contract that a single-match path bypasses the
    /// pattern-priority dispatch entirely.
    #[test]
    fn discover_payload_btf_id_single_size_match_returns_id() {
        let mut strings: Vec<u8> = vec![0];
        let n_u64 = sdta_push_name(&mut strings, "u64");
        let n_cgrp_ctx = sdta_push_name(&mut strings, "cgrp_ctx");
        let n_a = sdta_push_name(&mut strings, "a");
        let n_b = sdta_push_name(&mut strings, "b");
        let types = vec![
            // id 1: u64 (8 bytes; not a size-match for 16).
            SdtaSynType::Int {
                name_off: n_u64,
                size: 8,
                encoding: 0,
                offset: 0,
                bits: 64,
            },
            // id 2: struct cgrp_ctx { u64 a @ 0; u64 b @ 8 } size=16.
            SdtaSynType::Struct {
                name_off: n_cgrp_ctx,
                size: 16,
                members: vec![
                    SdtaSynMember {
                        name_off: n_a,
                        type_id: 1,
                        byte_offset: 0,
                    },
                    SdtaSynMember {
                        name_off: n_b,
                        type_id: 1,
                        byte_offset: 8,
                    },
                ],
            },
        ];
        let blob = sdta_build_btf(&types, &strings);
        let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");
        let choice = discover_payload_btf_id(&btf, 16, "");
        assert_eq!(
            choice.target_type_id, 2,
            "single 16-byte struct cgrp_ctx must be picked unambiguously"
        );
        assert_eq!(
            choice.reason, "",
            "single-match path must return empty reason; got {:?}",
            choice.reason
        );
    }

    /// Per-cgroup `scx_cgroup_ctx`-style name resolves via
    /// the `*_ctx` suffix arm. With one 16-byte struct named
    /// `scx_cgroup_ctx`, this is also a single size-match — the
    /// test pins that the heuristic accepts the per-cgroup name
    /// AND that an int of size 8 doesn't pollute the candidate
    /// list. The bug surface is exactly this case
    /// (per-cgroup struct fails to resolve via discover);
    /// confirming a clean single-match here pins the baseline
    /// before the multi-candidate cases below exercise the actual
    /// branching.
    #[test]
    fn discover_payload_btf_id_per_cgroup_ctx_resolves_via_ctx_suffix() {
        let mut strings: Vec<u8> = vec![0];
        let n_u64 = sdta_push_name(&mut strings, "u64");
        let n_cgrp = sdta_push_name(&mut strings, "scx_cgroup_ctx");
        let n_a = sdta_push_name(&mut strings, "a");
        let n_b = sdta_push_name(&mut strings, "b");
        let types = vec![
            SdtaSynType::Int {
                name_off: n_u64,
                size: 8,
                encoding: 0,
                offset: 0,
                bits: 64,
            },
            SdtaSynType::Struct {
                name_off: n_cgrp,
                size: 16,
                members: vec![
                    SdtaSynMember {
                        name_off: n_a,
                        type_id: 1,
                        byte_offset: 0,
                    },
                    SdtaSynMember {
                        name_off: n_b,
                        type_id: 1,
                        byte_offset: 8,
                    },
                ],
            },
        ];
        let blob = sdta_build_btf(&types, &strings);
        let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");
        let choice = discover_payload_btf_id(&btf, 16, "");
        assert_eq!(
            choice.target_type_id, 2,
            "scx_cgroup_ctx (single 16-byte size-match) must resolve via the \
             single-match arm"
        );
        assert_eq!(choice.reason, "");
    }

    /// `task_ctx` (exact name) wins over `*_ctx` suffix
    /// when both same-size structs exist. The heuristic at
    /// sdt_alloc.rs:646-651 lists arm 1 (`n == "task_ctx"`)
    /// before arm 4 (`*_ctx` suffix). Pin the priority order so
    /// a future refactor that drops the exact-name arm in favor
    /// of a single suffix-pattern walk surfaces here.
    #[test]
    fn discover_payload_btf_id_task_ctx_exact_wins_over_ctx_suffix() {
        let mut strings: Vec<u8> = vec![0];
        let n_u64 = sdta_push_name(&mut strings, "u64");
        let n_task = sdta_push_name(&mut strings, "task_ctx");
        let n_cgrp = sdta_push_name(&mut strings, "cgrp_ctx");
        let n_a = sdta_push_name(&mut strings, "a");
        let types = vec![
            SdtaSynType::Int {
                name_off: n_u64,
                size: 8,
                encoding: 0,
                offset: 0,
                bits: 64,
            },
            // id 2: task_ctx (16 bytes).
            SdtaSynType::Struct {
                name_off: n_task,
                size: 16,
                members: vec![
                    SdtaSynMember {
                        name_off: n_a,
                        type_id: 1,
                        byte_offset: 0,
                    },
                    SdtaSynMember {
                        name_off: n_a,
                        type_id: 1,
                        byte_offset: 8,
                    },
                ],
            },
            // id 3: cgrp_ctx (16 bytes — same size).
            SdtaSynType::Struct {
                name_off: n_cgrp,
                size: 16,
                members: vec![
                    SdtaSynMember {
                        name_off: n_a,
                        type_id: 1,
                        byte_offset: 0,
                    },
                    SdtaSynMember {
                        name_off: n_a,
                        type_id: 1,
                        byte_offset: 8,
                    },
                ],
            },
        ];
        let blob = sdta_build_btf(&types, &strings);
        let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");
        let choice = discover_payload_btf_id(&btf, 16, "");
        assert_eq!(
            choice.target_type_id, 2,
            "exact `task_ctx` arm (priority 1) must win over `*_ctx` suffix \
             arm (priority 4); cgrp_ctx must NOT be picked: {:?}",
            choice
        );
        assert_eq!(choice.reason, "");
    }

    /// Ambiguous at the `*_ctx` suffix arm with no upper-arm
    /// resolution. Two structs (`cgrp_ctx` and `task_data_ctx`)
    /// both 16 bytes, neither matching exact `task_ctx`,
    /// `*_arena_ctx`, or `*_task_ctx`. Arm 4 (`*_ctx`) gets 2 hits;
    /// the per-arm `_ => continue` at sdt_alloc.rs:670-674 advances
    /// past arm 4 (the last arm) and falls through to the
    /// post-loop "no unambiguous pattern winner" branch at
    /// sdt_alloc.rs:677-681, returning `target_type_id = 0` with
    /// `reason = "ambiguous: 2 candidates"`. Pin BOTH the id (0)
    /// AND the exact reason — the reason format is wire-stable
    /// (operator-visible via SdtAllocatorSnapshot::payload_type_reason).
    #[test]
    fn discover_payload_btf_id_ambiguous_at_ctx_arm_falls_through() {
        let mut strings: Vec<u8> = vec![0];
        let n_u64 = sdta_push_name(&mut strings, "u64");
        let n_a = sdta_push_name(&mut strings, "a");
        // Two structs ending in `_ctx` — both qualify ONLY at arm 4.
        let n_cgrp = sdta_push_name(&mut strings, "cgrp_ctx");
        let n_task_data = sdta_push_name(&mut strings, "task_data_ctx");
        let types = vec![
            SdtaSynType::Int {
                name_off: n_u64,
                size: 8,
                encoding: 0,
                offset: 0,
                bits: 64,
            },
            SdtaSynType::Struct {
                name_off: n_cgrp,
                size: 16,
                members: vec![
                    SdtaSynMember {
                        name_off: n_a,
                        type_id: 1,
                        byte_offset: 0,
                    },
                    SdtaSynMember {
                        name_off: n_a,
                        type_id: 1,
                        byte_offset: 8,
                    },
                ],
            },
            SdtaSynType::Struct {
                name_off: n_task_data,
                size: 16,
                members: vec![
                    SdtaSynMember {
                        name_off: n_a,
                        type_id: 1,
                        byte_offset: 0,
                    },
                    SdtaSynMember {
                        name_off: n_a,
                        type_id: 1,
                        byte_offset: 8,
                    },
                ],
            },
        ];
        let blob = sdta_build_btf(&types, &strings);
        let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");
        let choice = discover_payload_btf_id(&btf, 16, "");
        assert_eq!(
            choice.target_type_id, 0,
            "ambiguous `*_ctx` matches must fall through every arm and \
             return target_type_id=0; got {:?}",
            choice
        );
        assert_eq!(
            choice.reason, "ambiguous: 2 candidates",
            "ambiguous-fallback reason format is wire-stable (operator reads \
             SdtAllocatorSnapshot::payload_type_reason). Pin the format string \
             byte-for-byte; a refactor that changes 'ambiguous' to 'multi' or \
             'candidates' to 'matches' would silently break log scrapers."
        );
    }

    /// Per-arm continue resolves at lower arm. TWO `*_arena_ctx`
    /// structs (ambiguous at arm 2) AND ONE `*_task_ctx` struct
    /// (unambiguous at arm 3). Per the production code at
    /// sdt_alloc.rs:670-674, arm 2's `_ => continue` advances to
    /// arm 3, which has 1 hit → returns the `*_task_ctx` id.
    ///
    /// The docstring at sdt_alloc.rs:565-571 says "If 2+ structs
    /// match the *same* pattern, we fall back to hex". The code
    /// CONTRADICTS this — `continue` proceeds to the next arm
    /// rather than aborting to the post-loop fall-through. This
    /// test pins the CODE's behavior; if a future fix changes
    /// either side, the test must be updated to match the new
    /// semantics, and the doc-vs-code drift reconciled in the
    /// same commit.
    #[test]
    fn discover_payload_btf_id_per_arm_ambiguity_resolves_at_lower_arm() {
        let mut strings: Vec<u8> = vec![0];
        let n_u64 = sdta_push_name(&mut strings, "u64");
        let n_a = sdta_push_name(&mut strings, "a");
        // Two `*_arena_ctx` (collide at arm 2):
        let n_cgrp_arena = sdta_push_name(&mut strings, "cgrp_arena_ctx");
        let n_other_arena = sdta_push_name(&mut strings, "other_arena_ctx");
        // One unique `*_task_ctx` (resolves at arm 3):
        let n_my_task = sdta_push_name(&mut strings, "my_task_ctx");
        let types = vec![
            SdtaSynType::Int {
                name_off: n_u64,
                size: 8,
                encoding: 0,
                offset: 0,
                bits: 64,
            },
            // id 2: cgrp_arena_ctx (16 bytes).
            SdtaSynType::Struct {
                name_off: n_cgrp_arena,
                size: 16,
                members: vec![
                    SdtaSynMember {
                        name_off: n_a,
                        type_id: 1,
                        byte_offset: 0,
                    },
                    SdtaSynMember {
                        name_off: n_a,
                        type_id: 1,
                        byte_offset: 8,
                    },
                ],
            },
            // id 3: other_arena_ctx (16 bytes — collides with id 2 at arm 2).
            SdtaSynType::Struct {
                name_off: n_other_arena,
                size: 16,
                members: vec![
                    SdtaSynMember {
                        name_off: n_a,
                        type_id: 1,
                        byte_offset: 0,
                    },
                    SdtaSynMember {
                        name_off: n_a,
                        type_id: 1,
                        byte_offset: 8,
                    },
                ],
            },
            // id 4: my_task_ctx (16 bytes — unique at arm 3,
            // matches `*_task_ctx`).
            SdtaSynType::Struct {
                name_off: n_my_task,
                size: 16,
                members: vec![
                    SdtaSynMember {
                        name_off: n_a,
                        type_id: 1,
                        byte_offset: 0,
                    },
                    SdtaSynMember {
                        name_off: n_a,
                        type_id: 1,
                        byte_offset: 8,
                    },
                ],
            },
        ];
        let blob = sdta_build_btf(&types, &strings);
        let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");
        let choice = discover_payload_btf_id(&btf, 16, "");
        // Arm 1 (`task_ctx` exact): no hit. Arm 2 (`*_arena_ctx`):
        // 2 hits → continue. Arm 3 (`*_task_ctx`): 1 hit → return
        // id 4. Arm 4 (`*_ctx`): never reached.
        assert_eq!(
            choice.target_type_id, 4,
            "arm 2 ambiguous → continue; arm 3 unique my_task_ctx → return id 4. \
             Got {:?}. If this fails, the continue-on-arm-ambiguity semantics \
             changed — verify against the docstring at sdt_alloc.rs:565-571 \
             (which currently contradicts the code) and update both sides \
             together.",
            choice
        );
        assert_eq!(
            choice.reason, "",
            "successful pattern-arm resolution must return empty reason"
        );
    }
}
