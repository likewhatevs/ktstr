//! Per-map rendering dispatch for the failure dump.
//!
//! [`render_map`] is the per-map-type fan-out called from
//! [`super::dump_state`]. It builds one [`super::FailureDumpMap`] per
//! BPF map, dispatching on `info.map_type`:
//!
//! - ARRAY / HASH / LRU_HASH / PERCPU_*HASH / PERCPU_ARRAY — read
//!   bytes via the [`crate::monitor::bpf_map::BpfMapAccessor`] trait and
//!   render through BTF when available.
//! - ARENA — reuse the dump pre-pass snapshot or take a fresh
//!   page-granular snapshot.
//! - STRUCT_OPS — read the userspace `kvalue` shape and patch
//!   `state` from `kvalue.common.state`.
//! - TASK_STORAGE / INODE_STORAGE / SK_STORAGE / CGRP_STORAGE — walk
//!   the shared `bpf_local_storage_map` selem chain via the
//!   accessor's `iter_task_storage`. All four kernel map types share
//!   the same struct layout so they share the arm.
//! - RINGBUF / USER_RINGBUF — read producer/consumer/pending
//!   positions via [`render_ringbuf_state`].
//! - STACK_TRACE — walk `bpf_stack_map.buckets[]` via
//!   [`render_stack_traces`].
//! - FD-array families — report populated indices via
//!   [`render_fd_array_slots`].
//! - Anything else — surface a static explanation from
//!   [`MAP_TYPE_EXPLANATIONS`] or an "unknown map_type" diagnostic.
//!
//! The [`AccessorMemReader`] type implements [`MemReader`] so BTF
//! renderers can chase `__arena` pointers into a captured arena
//! snapshot. [`GuestMemMapAccessor::mem_reader`] constructs one
//! per dump pass.

use btf_rs::Btf;

use super::super::arena::{ArenaSnapshot, BpfArenaOffsets, snapshot_arena};
use super::super::bpf_map::{
    BPF_MAP_TYPE_ARENA, BPF_MAP_TYPE_ARRAY, BPF_MAP_TYPE_ARRAY_OF_MAPS, BPF_MAP_TYPE_BLOOM_FILTER,
    BPF_MAP_TYPE_CGROUP_ARRAY, BPF_MAP_TYPE_CGROUP_STORAGE, BPF_MAP_TYPE_CGRP_STORAGE,
    BPF_MAP_TYPE_CPUMAP, BPF_MAP_TYPE_DEVMAP, BPF_MAP_TYPE_DEVMAP_HASH, BPF_MAP_TYPE_HASH,
    BPF_MAP_TYPE_HASH_OF_MAPS, BPF_MAP_TYPE_INODE_STORAGE, BPF_MAP_TYPE_INSN_ARRAY,
    BPF_MAP_TYPE_LPM_TRIE, BPF_MAP_TYPE_LRU_HASH, BPF_MAP_TYPE_LRU_PERCPU_HASH,
    BPF_MAP_TYPE_PERCPU_ARRAY, BPF_MAP_TYPE_PERCPU_CGROUP_STORAGE, BPF_MAP_TYPE_PERCPU_HASH,
    BPF_MAP_TYPE_PERF_EVENT_ARRAY, BPF_MAP_TYPE_PROG_ARRAY, BPF_MAP_TYPE_QUEUE,
    BPF_MAP_TYPE_REUSEPORT_SOCKARRAY, BPF_MAP_TYPE_RINGBUF, BPF_MAP_TYPE_SK_STORAGE,
    BPF_MAP_TYPE_SOCKHASH, BPF_MAP_TYPE_SOCKMAP, BPF_MAP_TYPE_STACK, BPF_MAP_TYPE_STACK_TRACE,
    BPF_MAP_TYPE_STRUCT_OPS, BPF_MAP_TYPE_TASK_STORAGE, BPF_MAP_TYPE_USER_RINGBUF,
    BPF_MAP_TYPE_XSKMAP, BpfMapAccessor, BpfMapInfo, GuestMemMapAccessor, MAP_MATERIALIZE_MAX,
};
use super::super::btf_render::{
    ArenaResolveHit, CastHit, CrossBtfRef, FwdKind, MemReader, RenderedValue, render_value_with_mem,
};
use super::super::cast_analysis::CastMap;

use super::{
    CrossBtfFwdIndex, FailureDumpArrayEntry, FailureDumpEntry, FailureDumpFdArray, FailureDumpMap,
    FailureDumpPercpuEntry, FailureDumpPercpuHashEntry, FailureDumpRingbuf, FailureDumpStackTrace,
    FailureDumpStackTraceEntry, hex_dump,
};

/// Maximum per-CPU array key span the dump path will iterate.
///
/// `BPF_MAP_TYPE_PERCPU_ARRAY` declares `max_entries` at create-time;
/// the dump enumerates `0..min(max_entries, MAX_PERCPU_KEYS)` so a
/// scheduler that allocated a million-entry per-CPU array doesn't
/// blow up the report. Today's scx schedulers use small fixed-size
/// per-CPU arrays (one entry per topology level), so this cap is
/// generous.
pub(super) const MAX_PERCPU_KEYS: u32 = 256;

/// Maximum (key, value) pairs the dump path will pull from a HASH map.
///
/// Mirrors `super::super::btf_render::MAX_ARRAY_ELEMS` (4096): a
/// HASH map with millions of live entries would OOM the host
/// renderer if iterated unbounded, so the dump caps at 4096 and
/// surfaces an `error` describing the truncation. The unrendered
/// tail is dropped (the truncation `error` reports it) — recording it
/// would itself require unbounded memory.
///
/// Aliases [`MAP_MATERIALIZE_MAX`]: the bpf_map walkers stop
/// materializing one past this value, so the render cap and the walker
/// materialize cap are intentionally one const — a drift between them
/// would silently lose the `len > MAX_HASH_ENTRIES` truncation signal.
pub(super) const MAX_HASH_ENTRIES: usize = MAP_MATERIALIZE_MAX;

/// Maximum entries the dump path will render from a multi-entry
/// `BPF_MAP_TYPE_ARRAY` map.
///
/// Matches [`MAX_HASH_ENTRIES`] (4096), not [`MAX_PERCPU_KEYS`]
/// (256): plain ARRAY maps are not "one entry per topology level" —
/// sched_ext schedulers declare large ones (mitosis's `cells` is 256,
/// `debug_events` is 4096), so a 256 cap would silently truncate real
/// data. The OOM bound is `MAX_ARRAY_KEYS * MAX_VALUE_SIZE`; per-entry
/// size is separately capped at `MAX_VALUE_SIZE` inside
/// `read_bpf_map_array_value`. Entries beyond the cap are dropped and
/// the truncation surfaces in the map's `error`.
pub(super) const MAX_ARRAY_KEYS: u32 = 4096;

/// Maximum stack-trace bucket pointers the dump path will probe in a
/// `BPF_MAP_TYPE_STACK_TRACE` map.
///
/// The kernel rounds `n_buckets` up to the next power of two, so a
/// scheduler that asked for 1024 traces sees 1024 buckets; one that
/// asked for a million sees a million. Cap at 16k slots to bound the
/// freeze-time read cost (each slot is one 8-byte translate+read);
/// truncation surfaces as `truncated: true` in
/// [`FailureDumpStackTrace`].
pub(super) const MAX_STACK_TRACE_BUCKETS: u32 = 16_384;

/// Maximum PCs (or build-id records' worth of bytes) extracted from one
/// stack-trace bucket. Bounds the per-bucket render cost; deeper
/// stacks beyond this point are surfaced as truncated.
pub(super) const MAX_STACK_TRACE_PCS: u32 = 128;

/// Maximum FD-array slots the dump path will probe.
///
/// FD-array families (PROG_ARRAY, PERF_EVENT_ARRAY, etc.) typically
/// declare `max_entries` in the dozens, but a scheduler hosting a
/// 64k-entry tail-call table should still get a bounded scan. 4096
/// matches `MAX_HASH_ENTRIES` for symmetry across the dump path.
pub(super) const MAX_FD_ARRAY_SLOTS: u32 = 4096;

/// Maximum populated indices recorded in [`FailureDumpFdArray::indices`].
/// A fully-packed scan yields up to [`MAX_FD_ARRAY_SLOTS`] indices,
/// which is itself manageable; the explicit cap exists so a future
/// MAX_FD_ARRAY_SLOTS bump doesn't silently bloat the dump JSON.
pub(super) const MAX_FD_ARRAY_INDICES: usize = 1024;

/// `user_addr → index into arena_snapshot.pages` lookup table,
/// built once per dump pass and threaded into every
/// [`AccessorMemReader`] the per-map render constructs.
///
/// **Key shape**: full 64-bit user-side arena page address
/// (`ArenaPage::user_addr`), distinct from sibling [`ArenaSlotIndex`]
/// which keys on the LOW 32 bits of slot start. Pages are uniquely
/// identified by their full user-VA so multi-arena schedulers (each
/// arena map gets its own snapshot) don't collide on shared low-32
/// page-aligned bits.
///
/// `read_arena` is on the freeze hot path and runs for every Ptr in
/// every BTF-rendered value — a linear scan over `pages` was O(N)
/// per chase, making a large arena render O(N·M) over the snapshot.
/// The HashMap keeps the chase O(1) per lookup. Building the index
/// once in [`super::dump_state`] (instead of inside each
/// [`GuestMemMapAccessor::mem_reader`] call) drops a per-map
/// rebuild that cost O(pages) on every map enumerated.
///
/// Empty when no arena was captured by the dump pre-pass, so the
/// no-arena path costs nothing extra.
pub(super) type ArenaPageIndex = std::collections::HashMap<u64, usize>;

/// Per-slot metadata stored in the [`ArenaSlotIndex`] for each live
/// sdt_alloc allocation, supporting range-based lookup of any
/// chased pointer that lands inside the slot.
///
/// Four fields capture the slot shape the renderer needs at chase
/// time:
///
/// - `elem_size` — total byte stride of one allocator slot
///   (`sdt_pool.elem_size`). The lookup uses this to bound the
///   slot's address range as `[slot_start, slot_start + elem_size)`.
/// - `header_size` — size of the `union sdt_id` header at the front
///   of the slot (8 bytes on every kernel that ships sdt_alloc, per
///   `lib/sdt_task_defs.h`). Pointers landing in the first
///   `header_size` bytes of the slot are inside the header, not the
///   payload.
/// - `target_type_id` — the BTF type id of the payload struct
///   (everything after the header). Resolved by the sdt_alloc
///   pre-pass via [`super::super::sdt_alloc::discover_payload_btf_id`]
///   from `payload_size = elem_size - header_size`. Scoped to the
///   BTF identified by `source_btf_kva` — feeding it into any other
///   BTF would mis-resolve because BTF type id spaces are per-BTF.
/// - `source_btf_kva` — the kernel virtual address of the program
///   BTF the `target_type_id` was resolved against. The renderer
///   compares this against each requesting map's `btf_kva` and
///   suppresses the bridge hit on mismatch so a chase originating
///   from a map backed by a different program BTF cannot consume
///   an id from the wrong id space. `0` means "any BTF" — the
///   pre-pass discovered the slot before a specific scheduler BTF
///   was locked in (currently unreachable; reserved for future
///   single-arena multi-BTF shapes).
///
/// Storing `elem_size` and `header_size` (rather than precomputing a
/// single payload-start key) lets the index handle both slot-start
/// pointers (`user_addr` from `sdt_alloc()` directly) and payload-start
/// pointers (`user_addr + header_size` from `scx_task_data(p)`) with a
/// single keyed entry per slot — the lookup computes `offset_in_slot`
/// from the chased address and the slot start, then routes to the
/// correct render flavor.
///
/// Visibility matches sibling render-pass types (`ArenaPageIndex`,
/// `SdtAllocMeta`, `ArenaSlotIndex`, `RenderMapCtx`): all are
/// `pub(super)` so the `dump` module owns the surface. The
/// 20-byte POD derive set (`PartialEq, Eq`) lets tests use
/// `assert_eq!` directly. `Hash` is intentionally NOT derived —
/// the type is the VALUE side of [`ArenaSlotIndex`] (a `BTreeMap`
/// keyed by `slot_start: u32`) and never composes into a hashed
/// container itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ArenaSlotInfo {
    pub(super) elem_size: u32,
    pub(super) header_size: u32,
    pub(super) target_type_id: u32,
    pub(super) source_btf_kva: u64,
}

/// `slot_start → ArenaSlotInfo` lookup populated by the sdt_alloc
/// pre-pass for every live allocator slot.
///
/// `slot_start` is the low 32 bits of the user-side arena address
/// of the slot's first byte (the sdt_id header). Arena pointers in
/// guest memory store the user-side window's low 32 bits, so the
/// index keys on the same windowed shape as the values the renderer
/// chases.
///
/// **4 GiB-alignment invariant**: the low-32 keying is correct iff
/// the arena's `user_vm_start` is 4 GiB-aligned (the high 32 bits of
/// every slot address are constant). Every in-tree scx scheduler
/// uses a 4 GiB-aligned `map_extra` (`1 << 32`, `1 << 44`); the
/// kernel auto-pick path rounds the user VM area up to `SZ_4G`
/// before mounting. An unaligned `user_vm_start` (only reachable
/// via an out-of-tree scheduler passing custom `map_extra`) would
/// make low-32 keying ambiguous, so [`super::dump_state`] checks
/// alignment at index-build and skips the index entirely with a
/// `tracing::warn!` when violated. This index is therefore only
/// populated when the invariant holds.
///
/// **Single-arena invariant**: every allocator's slots in the index
/// are keyed against the SAME `user_vm_start` because the dump
/// pre-pass walks every sdt_alloc allocator using
/// [`super::dump_state`]'s single `arena_kern_vm_start` value
/// (derived from `shared_arena_snapshot`, which picks the first
/// `BPF_MAP_TYPE_ARENA` map only). Schedulers that declare multiple
/// arena maps backing distinct allocators are not supported by the
/// current dump pipeline — the second arena would silently
/// mis-translate because the walker uses the first arena's
/// `kern_vm_start` for every allocator. Today's scx schedulers
/// declare a single arena, so the constraint is met in practice. If
/// a multi-arena scheduler ever lands, this index's low-32 keying
/// becomes ambiguous across allocators (two slots from different
/// arenas with different `user_vm_start` bases could collide on the
/// low-32 key) and the design must move to a `(arena_id, low_32)`
/// composite key.
///
/// **Slot non-overlap invariant**: two slots in the same allocator
/// never occupy overlapping byte ranges (the kernel allocator places
/// slots back-to-back inside one `sdt_chunk` per
/// `lib/sdt_alloc.bpf.c`'s `scx_alloc_internal` and never re-uses a
/// position while the bitmap still has it marked allocated). Under
/// the single-arena invariant above, two distinct allocators
/// sharing one arena receive non-overlapping kernel-side
/// allocations from `bpf_arena_alloc_pages`, so their slot address
/// ranges also do not overlap. The dedup logic in
/// [`super::dump_state`] keys on exact `slot_start` only and emits
/// `tracing::warn!` on duplicates; an overlapping
/// `(start_a, elem_a)`, `(start_b, elem_b)` pair where
/// `start_a + elem_a > start_b > start_a` is structurally
/// impossible.
///
/// The map is consulted by
/// [`super::super::btf_render::MemReader::resolve_arena_type`] via
/// [`AccessorMemReader::resolve_arena_type`] — given a chased
/// pointer's value, the implementation uses
/// [`std::collections::BTreeMap::range`] to find the slot whose
/// `[slot_start, slot_start + elem_size)` range contains the
/// address. Once located:
///
/// - `offset_in_slot == 0` (slot-start pointer, e.g. the `data`
///   field of `scx_task_map_val` storing the raw return of
///   `sdt_alloc()`): the bridge returns the payload type id with
///   `header_skip = header_size`. The chase reads
///   `header_size + btf_size` bytes from the chased address,
///   slices off the header, and renders the payload.
/// - `offset_in_slot == header_size` (payload-start pointer, e.g.
///   the return of `scx_task_data(p)` cached in
///   `cached_taskc_raw`): the bridge returns the payload type id
///   with `header_skip = 0`. The chase reads `btf_size` bytes
///   from the chased address and renders the payload — the same
///   path the prior payload-only index supported.
/// - Any other `offset_in_slot` (mid-header or mid-payload): the
///   bridge returns `None`. Mid-header pointers carry no useful
///   payload type; mid-payload pointers would need the renderer
///   to walk back to the struct start, which is out of scope
///   today.
///
/// `BTreeMap` keeps the iteration order deterministic for tests;
/// the hot-path lookup is `O(log N)` against the small entry count
/// (capped at [`super::super::sdt_alloc::MAX_SDT_ALLOC_ENTRIES`]).
///
/// Empty when no allocator with a typed payload was discovered, so
/// the bridge stays a no-op outside the sdt_alloc-using slice of
/// schedulers — same cost-shape as [`ArenaPageIndex`].
pub(super) type ArenaSlotIndex = std::collections::BTreeMap<u32, ArenaSlotInfo>;

/// Build [`ArenaPageIndex`] from `snap.pages`. `entry().or_insert()`
/// keeps the FIRST page seen for any duplicate `user_addr` and
/// logs the collision instead of silently shadowing — duplicates
/// would indicate a corrupted snapshot (the kernel never maps two
/// arena pages at the same user-side address) so the warn is the
/// diagnostic.
pub(super) fn build_arena_page_index(
    snap: Option<&super::super::arena::ArenaSnapshot>,
) -> ArenaPageIndex {
    let mut index = ArenaPageIndex::new();
    let Some(snap) = snap else {
        return index;
    };
    for (i, p) in snap.pages.iter().enumerate() {
        match index.entry(p.user_addr) {
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert(i);
            }
            std::collections::hash_map::Entry::Occupied(o) => {
                tracing::warn!(
                    user_addr = format_args!("{:#x}", p.user_addr),
                    first_idx = *o.get(),
                    duplicate_idx = i,
                    "arena snapshot has duplicate user_addr; keeping first page",
                );
            }
        }
    }
    index
}

/// Append one allocator's slots to the per-pass [`ArenaSlotIndex`].
///
/// `entries` carries the live slot records the sdt_alloc walker
/// produced (`SdtAllocEntry::user_addr` already masked to the low
/// 32 bits — see
/// `super::super::sdt_alloc::TreeWalker::emit_leaf`).
/// `target_type_id` is the resolved per-slot payload type;
/// `header_size` and `elem_size` come from the allocator's `pool`
/// metadata via [`super::super::sdt_alloc::SdtAllocOffsets`] +
/// `walk_sdt_allocator`.
///
/// Conversions via `u32::try_from` reject `header_size` /
/// `elem_size` values that don't fit a `u32`. `data_header_size`
/// is 8 on every kernel that ships sdt_alloc and `elem_size` is
/// bounded above by `MAX_ELEM_SIZE = 4096` in the walker, so the
/// conversions never fail in practice; a future drift surfaces as
/// a no-op rather than a panic. `target_type_id == 0` short-
/// circuits — the bridge filters zero ids as "no payload type" and
/// adding such entries would just point every chase at 0.
///
/// Duplicate `slot_start` keys (two slots reporting the same
/// windowed start, indicating a stale snapshot from a freed
/// allocation racing with the freeze) keep the FIRST entry and
/// emit a `tracing::warn!` line so an operator diagnosing a
/// wrong-render can spot the collision. The kernel allocator
/// places slots back-to-back inside one `sdt_chunk` and never
/// re-uses a position while the bitmap still has it marked
/// allocated (see `lib/sdt_alloc.bpf.c::scx_alloc_internal`'s
/// bitmap-then-data ordering), so distinct slots cannot have
/// overlapping `[start, start + elem_size)` ranges — dedup-on-
/// exact-key is sufficient.
pub(super) fn append_arena_slot_index_for_allocator(
    index: &mut ArenaSlotIndex,
    allocator_name: &str,
    target_type_id: u32,
    header_size: usize,
    elem_size: u64,
    slot_addrs: &[u64],
    source_btf_kva: u64,
) {
    if target_type_id == 0 {
        return;
    }
    let Ok(header_low32) = u32::try_from(header_size) else {
        return;
    };
    let Ok(elem_low32) = u32::try_from(elem_size) else {
        return;
    };
    let info = ArenaSlotInfo {
        elem_size: elem_low32,
        header_size: header_low32,
        target_type_id,
        source_btf_kva,
    };
    for &addr in slot_addrs {
        let slot_start = addr as u32;
        match index.entry(slot_start) {
            std::collections::btree_map::Entry::Vacant(v) => {
                v.insert(info);
            }
            std::collections::btree_map::Entry::Occupied(o) => {
                // Low-32-bit collision mitigation: probability is
                // ~1.6% with 8K entries. A real duplicate signals
                // either a torn snapshot OR (more likely) a low-32
                // collision between slot starts in different
                // allocators. Surface as `warn` so an operator
                // diagnosing a wrong-render can spot the collision
                // — `debug` would have hidden the cause.
                tracing::warn!(
                    slot_start = format_args!("{:#x}", slot_start),
                    first_target_type_id = o.get().target_type_id,
                    duplicate_target_type_id = target_type_id,
                    allocator = %allocator_name,
                    "sdt_alloc bridge has duplicate slot_start (low-32 collision \
                     across allocators or torn snapshot); keeping first entry",
                );
            }
        }
    }
}

/// Free helper: check whether `addr` falls within the arena's user_vm
/// window. Factored out of [`AccessorMemReader::is_arena_addr`] so
/// unit tests can construct an [`ArenaSnapshot`] directly without
/// requiring a [`super::super::guest::GuestKernel`] (which would need
/// a full guest memory mock).
///
/// Returns `false` when `snap` is `None`, `snap.user_vm_start == 0`,
/// or `user_vm_start + 1<<32` would overflow `u64`. The 4 GiB upper
/// bound matches `kernel/bpf/arena.c::arena_map_alloc`'s `SZ_4G`
/// max.
pub(super) fn is_arena_addr_in_snapshot(
    snap: Option<&super::super::arena::ArenaSnapshot>,
    addr: u64,
) -> bool {
    let Some(snap) = snap else {
        return false;
    };
    if snap.user_vm_start == 0 {
        return false;
    }
    let Some(end) = snap.user_vm_start.checked_add(1 << 32) else {
        return false;
    };
    addr >= snap.user_vm_start && addr < end
}

/// Free helper backing [`AccessorMemReader::cast_lookup`]: look up
/// the [`CastHit`] for `(parent_type_id, member_byte_offset)` in the
/// optional [`CastMap`]. `None` map (no cast analysis supplied — the
/// dump pass ran without a scheduler binary) short-circuits before
/// any map access; a present map returns the keyed entry by value
/// ([`CastHit`] is `Copy`), or `None` when the exact key is absent
/// (no fallback to nearby offsets). Extracted (mirroring
/// [`is_arena_addr_in_snapshot`] / [`read_arena_in_snapshot`] /
/// [`resolve_arena_type_in_index`]) so unit tests exercise the real
/// short-circuit + keyed-lookup path without constructing the
/// private [`AccessorMemReader`].
pub(super) fn cast_lookup_in_map(
    cast_map: Option<&CastMap>,
    parent_type_id: u32,
    member_byte_offset: u32,
) -> Option<CastHit> {
    let map = cast_map?;
    map.get(&(parent_type_id, member_byte_offset)).copied()
}

/// Free helper backing [`AccessorMemReader::read_arena`]: read `len`
/// bytes at arena user-address `addr` from the pre-pass snapshot,
/// falling back to a live kernel-virtual read. Extracted (mirroring
/// [`is_arena_addr_in_snapshot`]) so unit tests exercise the real
/// page-index fast path, the cross-page `checked_add` / `> 4096` bound,
/// the short-captured-page fall-through, and the `kern_vm_start`-anchor
/// KVA composition without a full [`super::super::guest::GuestKernel`].
/// `read_kva` performs the slow-path read; it is invoked only when the
/// page was not captured (or was captured short) AND
/// `kern_vm_start != 0`.
pub(super) fn read_arena_in_snapshot(
    snapshot: Option<&super::super::arena::ArenaSnapshot>,
    arena_page_index: &ArenaPageIndex,
    addr: u64,
    len: usize,
    read_kva: impl FnOnce(u64, usize) -> Option<Vec<u8>>,
) -> Option<Vec<u8>> {
    let snap = snapshot?;
    let offset = (addr & 0xFFF) as usize;
    // checked_add against a hostile addr+len near u64::MAX keeps the
    // single-page bound from wrapping into a false-positive accept.
    let end = offset.checked_add(len)?;
    if end > 4096 {
        return None;
    }
    let page_addr = addr & !0xFFF;
    if let Some(&idx) = arena_page_index.get(&page_addr) {
        let page = &snap.pages[idx];
        if end <= page.bytes.len() {
            return Some(page.bytes[offset..end].to_vec());
        }
        // Captured page is short (region/DRAM truncation at capture
        // time): fall through to the live walker rather than returning
        // a short slice.
    }
    // Slow path: snapshot didn't capture this page. `kern_vm_start == 0`
    // means the pre-pass bailed before resolving the kernel-side anchor;
    // without it there's no way to compose a KVA.
    if snap.kern_vm_start == 0 {
        return None;
    }
    let kva = snap.kern_vm_start.wrapping_add(addr & 0xFFFF_FFFF);
    read_kva(kva, len)
}

/// Free helper: resolve a chased arena address to a payload BTF type
/// id and `header_skip` byte count. Factored out of
/// [`AccessorMemReader::resolve_arena_type`] so unit tests can call
/// it with a synthetic [`ArenaSlotIndex`] and an [`ArenaSnapshot`]
/// without constructing a [`super::super::guest::GuestKernel`].
///
/// `arena_snapshot` is the pre-pass arena snapshot; the helper uses
/// it via [`is_arena_addr_in_snapshot`] to gate the lookup so an
/// out-of-window address can't accidentally collide with a slot
/// whose low-32-bit-windowed start matches by happenstance.
///
/// `arena_slot_index` is the per-allocator slot index. The lookup
/// uses [`std::collections::BTreeMap::range`] to find the slot whose
/// `[slot_start, slot_start + elem_size)` range contains the
/// chased address (windowed to its low 32 bits).
///
/// `requesting_btf_kva` identifies the BTF the calling renderer is
/// using. The lookup returns `None` when the matched slot's
/// [`ArenaSlotInfo::source_btf_kva`] is non-zero and differs from
/// this value — the slot's `target_type_id` is in a different BTF's
/// id space and feeding it into the caller's BTF would either
/// resolve to the wrong struct or fail entirely. A `0`
/// `source_btf_kva` is treated as "any BTF" (legacy / synthetic
/// entries); a `0` `requesting_btf_kva` from the caller is treated
/// as "no BTF context available", which suppresses the hit
/// conservatively when the index entry is BTF-scoped.
///
/// See [`AccessorMemReader::resolve_arena_type`]'s in-trait doc for
/// the full slot-position routing semantics.
pub(super) fn resolve_arena_type_in_index(
    arena_snapshot: Option<&super::super::arena::ArenaSnapshot>,
    arena_slot_index: Option<&ArenaSlotIndex>,
    addr: u64,
    requesting_btf_kva: u64,
) -> Option<ArenaResolveHit> {
    let index = arena_slot_index?;
    if !is_arena_addr_in_snapshot(arena_snapshot, addr) {
        return None;
    }
    let key = (addr & 0xFFFF_FFFF) as u32;
    let Some((&slot_start, info)) = index.range(..=key).next_back() else {
        tracing::debug!(
            addr = format_args!("{:#x}", addr),
            key = format_args!("{:#x}", key),
            index_len = index.len(),
            "resolve_arena_type: no slot found in ArenaSlotIndex for key"
        );
        return None;
    };
    let slot_end_u64 = (slot_start as u64) + (info.elem_size as u64);
    if (key as u64) >= slot_end_u64 {
        return None;
    }
    // Cross-BTF id-space gate: the slot's `target_type_id` is keyed
    // against `info.source_btf_kva`'s id space. A caller rendering
    // against a different BTF cannot consume this id — its type id
    // space is disjoint. Suppress the hit so the renderer falls
    // through to its existing cross-BTF / Fwd-skip path instead of
    // wrong-rendering against a sibling BTF's id. A zero
    // `source_btf_kva` (synthetic / legacy entries) is treated as
    // "any BTF"; a zero `requesting_btf_kva` from the caller means
    // the renderer has no BTF context (vmlinux fallback), which is
    // suppressed conservatively.
    if info.source_btf_kva != 0
        && (requesting_btf_kva == 0 || requesting_btf_kva != info.source_btf_kva)
    {
        tracing::debug!(
            addr = format_args!("{:#x}", addr),
            slot_start = format_args!("{:#x}", slot_start),
            slot_btf_kva = format_args!("{:#x}", info.source_btf_kva),
            requesting_btf_kva = format_args!("{:#x}", requesting_btf_kva),
            "resolve_arena_type: cross-BTF id-space mismatch; suppressing hit"
        );
        return None;
    }
    let offset_in_slot = key - slot_start;
    if offset_in_slot == 0 {
        Some(ArenaResolveHit {
            target_type_id: info.target_type_id,
            header_skip: info.header_size as usize,
        })
    } else if offset_in_slot == info.header_size {
        Some(ArenaResolveHit {
            target_type_id: info.target_type_id,
            header_skip: 0,
        })
    } else {
        None
    }
}

/// Free helper: chained resolve. Consults the per-instance sdt_alloc
/// index first via [`resolve_arena_type_in_index`]; on miss, checks
/// whether `addr` falls inside a live `scx_static` bump-allocator
/// region and (if so) returns `None` deliberately — the bridge
/// recognises the address as "in scx_static memory" but cannot
/// recover a per-allocation type without a per-call-site type hook
/// from cast analysis (see the
/// [`crate::monitor::scx_static_alloc`] module-level doc).
///
/// The deliberate `None` on scx_static hit is the "no invalid data
/// made" contract: returning a guess would render the slot against
/// the wrong struct shape; returning `None` preserves the renderer's
/// existing fall-through to the historical Fwd-skip / cross-BTF
/// resolve path.
///
/// This helper composes with [`is_arena_addr_in_snapshot`] only via
/// the inner [`resolve_arena_type_in_index`] call; the scx_static
/// membership check uses the index directly because the
/// `[memory, memory + max_alloc_bytes)` region has already been
/// bounded to the arena window at walker time (every byte of an
/// scx_static region lives inside the arena by construction; see
/// `lib/sdt_alloc.bpf.c::scx_static_init` which calls
/// `bpf_arena_alloc_pages` to obtain `memory`).
///
/// Returns `Some(hit)` only when the sdt_alloc index resolved
/// completely. Returns `None` on every other path: no sdt_alloc
/// match (with or without scx_static fall-through hit), or
/// out-of-window address. Callers that need to distinguish
/// "scx_static-membership None" from "out-of-window None" can
/// consult [`super::super::scx_static_alloc::is_arena_addr_in_scx_static_index`]
/// directly.
pub(super) fn resolve_arena_type_with_static_fallback(
    arena_snapshot: Option<&super::super::arena::ArenaSnapshot>,
    arena_slot_index: Option<&ArenaSlotIndex>,
    scx_static_index: Option<&super::super::scx_static_alloc::ScxStaticRangeIndex>,
    addr: u64,
    requesting_btf_kva: u64,
) -> Option<ArenaResolveHit> {
    if let Some(hit) =
        resolve_arena_type_in_index(arena_snapshot, arena_slot_index, addr, requesting_btf_kva)
    {
        return Some(hit);
    }
    // sdt_alloc index missed. Consult the scx_static range index for
    // the diagnostic membership check; on hit, log a `tracing::trace!`
    // line so an operator diagnosing a missing typed-pointer render
    // can see the cause ("address is in scx_static memory; bridge
    // has no type") without re-deriving the membership themselves.
    // The function still returns `None` — see the doc-comment above
    // for the "no invalid data made" rationale.
    if let Some(static_index) = scx_static_index
        && is_arena_addr_in_snapshot(arena_snapshot, addr)
        && super::super::scx_static_alloc::is_arena_addr_in_scx_static_index(static_index, addr)
    {
        tracing::trace!(
            addr = format_args!("{:#x}", addr),
            "resolve_arena_type: hit scx_static range; no per-allocation \
             type recovery available, falling through to caller's \
             skip path",
        );
        return None;
    }
    // No meta fallback: the meta-based "best-effort" resolve was a
    // false positive factory — it turned ANY arena-window address
    // into a typed chase using the first allocator's payload type,
    // regardless of whether the address was an actual allocation.
    // Counters and timestamps whose values fell in the 4 GiB arena
    // window were rendered as typed structs (wrong). If the per-slot
    // ArenaSlotIndex missed (cap exceeded, snapshot truncation), the
    // chase fails closed with a clear skip reason. Correctness over
    // coverage.
    None
}

/// [`MemReader`] backed by a [`super::super::guest::GuestKernel`]
/// page-walker plus an optional [`super::super::arena::ArenaSnapshot`]
/// for `__arena` pointer chase. Constructed via
/// [`GuestMemMapAccessor::mem_reader`]; the dump path threads one
/// instance through every BTF render call so arena pointers in
/// the rendered tree resolve to typed contents.
///
/// Lifetimes: borrows the kernel from the accessor, the snapshot
/// from [`super::dump_state`]'s pre-pass, and the per-pass
/// [`ArenaPageIndex`] — all three outlive the per-map render they
/// parameterize.
struct AccessorMemReader<'a> {
    kernel: &'a super::super::guest::GuestKernel,
    /// Pre-pass snapshot from [`super::dump_state`]. `Some` when an
    /// arena map exists in the report; `is_arena_addr` and
    /// `read_arena` delegate to its
    /// [`ArenaSnapshot::user_vm_start`] +
    /// [`ArenaSnapshot::pages`] for the fast path, falling back to
    /// the kernel page-table walker via
    /// [`ArenaSnapshot::kern_vm_start`] when a Ptr targets a page
    /// the snapshot didn't capture (sequential prefix capped at
    /// MAX_ARENA_PAGES; live schedulers with many tasks point past
    /// that boundary). `None` disables arena-pointer chase entirely.
    arena_snapshot: Option<&'a super::super::arena::ArenaSnapshot>,
    /// Borrowed page-index built once per dump pass — see
    /// [`ArenaPageIndex`]. Empty index is valid (no-arena path).
    arena_page_index: &'a ArenaPageIndex,
    /// Guest's `nr_cpu_ids`, threaded through from
    /// [`RenderMapCtx::num_cpus`]. The btf_render cpumask path caps
    /// the bit walk at this value so a kernel-side cpumask slab
    /// allocation (sized to NR_CPUS at config time, e.g. 8192 bits)
    /// doesn't render bits past the actual guest CPU count as
    /// phantom cpus on top of slab padding / freelist garbage.
    num_cpus: u32,
    /// Optional BPF cast-analysis output, threaded through from
    /// [`RenderMapCtx::cast_map`]. When `Some`, the
    /// [`MemReader::cast_lookup`] override forwards
    /// `(parent_type_id, member_byte_offset)` queries into the map
    /// so the renderer can promote `u64` fields the analyzer
    /// flagged into typed-pointer renders. `None` disables the
    /// intercept (the trait default returns `None`, leaving every
    /// `u64` field as a plain unsigned counter).
    cast_map: Option<&'a CastMap>,
    /// Optional sdt_alloc payload-type index — see [`ArenaSlotIndex`].
    /// When `Some`, the [`MemReader::resolve_arena_type`] override
    /// translates a chased arena address into its allocator slot's
    /// payload BTF type id, letting the renderer chase a
    /// forward-declared pointee whose body lives in the sdt_alloc
    /// library's BTF. `None` (no allocator metadata discovered)
    /// keeps the renderer's default behaviour intact: a Fwd target
    /// without a complete sibling in the BTF surfaces as
    /// "forward declaration; body not in this BTF".
    arena_slot_index: Option<&'a ArenaSlotIndex>,
    /// Discovered sdt_alloc allocator metadata for the dump pass.
    /// Consumed only by the
    /// [`MemReader::resolve_arena_type_meta_fallback`] override
    /// (see [`AccessorMemReader::resolve_arena_type_meta_fallback`]),
    /// which `try_sdt_alloc_bridge` calls only when the chase
    /// target is `Type::Fwd(_)` AND the per-slot
    /// [`MemReader::resolve_arena_type`] returned `None`. The override
    /// returns the first metadata entry with a resolved
    /// `target_type_id` if the address lies inside the arena window,
    /// covering Fwd-pointee chases when the per-slot index missed
    /// (snapshot truncation, `MAX_SDT_ALLOC_ENTRIES` cap). The
    /// type-gating is what prevents the historical "false positive
    /// factory" behaviour — arbitrary `u64` fields whose values
    /// happen to land in the arena window do NOT trigger this
    /// fallback because they reach
    /// [`MemReader::resolve_arena_type`] (no meta consultation),
    /// not the Fwd-gated bridge path. Empty slice (no allocator with
    /// a typed payload was discovered) disables the fallback.
    sdt_alloc_metas: &'a [SdtAllocMeta],
    /// Optional cross-BTF Fwd resolution context — see
    /// [`CrossBtfFwdIndex`]. When `Some`, the
    /// [`MemReader::cross_btf_resolve_fwd`] override looks up
    /// the Fwd's `name` in the per-dump index (gated by
    /// [`FwdKind`] to enforce the struct-vs-union aggregate
    /// kind) and returns a [`CrossBtfRef`] whose `btf` borrow
    /// points at the resolving sibling. `None` (no scheduler
    /// binary, or analyzer found no complete struct/union
    /// definitions) keeps the renderer's default behaviour: a
    /// Fwd terminal with no in-BTF sibling surfaces as "forward
    /// declaration; body not in this BTF".
    cross_btf_fwd_index: Option<&'a CrossBtfFwdIndex<'a>>,
    /// Optional `scx_static` bump-allocator range index — see
    /// [`super::super::scx_static_alloc::ScxStaticRangeIndex`]. The
    /// dump pre-pass populates this from a [`crate::monitor::scx_static_alloc::ScxStaticSnapshot`]
    /// covering every live `scx_static` instance's
    /// `[memory_low32, memory_low32 + off)` region. The
    /// [`MemReader::resolve_arena_type`] override consults this
    /// index AFTER [`Self::arena_slot_index`] misses: it answers
    /// "is the chased address in scx_static memory?" with a
    /// boolean. The bridge cannot recover per-allocation BTF type
    /// ids from `scx_static` memory (no per-slot header; the
    /// analyzer would need a per-call-site type hook that does
    /// not exist today — see the
    /// [`crate::monitor::scx_static_alloc`] module-level doc),
    /// so a hit here returns `None` from `resolve_arena_type` —
    /// the "no invalid data made" contract: the renderer falls
    /// back to the historical Fwd-skip / cross-BTF behaviour
    /// rather than risking a wrong-type render.
    scx_static_index: Option<&'a super::super::scx_static_alloc::ScxStaticRangeIndex>,
    /// Optional set of `slot_start` low-32 addresses for every
    /// allocator slot the dump pre-pass already rendered into
    /// `report.sdt_allocations`. The
    /// [`MemReader::is_already_rendered`] override consults this
    /// set so the per-map renderer's arena chase can short-circuit
    /// when a TASK_STORAGE / HASH map's value points at a slot the
    /// typed-allocator surface already shows — preventing the same
    /// payload from appearing twice in the failure dump. `None`
    /// (no allocator pre-pass produced any rendered slot, or the
    /// caller chose not to dedup for this reader instance) keeps
    /// the chase pipeline's existing behaviour intact.
    rendered_slot_addrs: Option<&'a std::collections::HashSet<u32>>,
    /// Unique alloc_sizes from all Arena CastHits in the CastMap.
    /// Fallback for deferred-resolve chases with alloc_size=None.
    alloc_size_types: &'a [(u64, String)],
    /// Kernel virtual address of the program BTF the current per-map
    /// renderer is using. Threaded into the
    /// [`MemReader::resolve_arena_type`] gate so the sdt_alloc
    /// bridge can suppress hits whose
    /// [`ArenaSlotInfo::source_btf_kva`] points at a different
    /// program BTF (multi-`.bpf.o` schedulers where each object has
    /// its own BTF and its own id space). `0` is the "no per-map
    /// program BTF" sentinel (e.g. a map that fell back to the
    /// caller-supplied vmlinux BTF); the bridge suppresses hits
    /// conservatively in that case to avoid leaking an id from one
    /// program BTF into the vmlinux render context. See
    /// [`resolve_arena_type_in_index`] for the gate semantics.
    requesting_btf_kva: u64,
}

impl MemReader for AccessorMemReader<'_> {
    fn read_kva(&self, kva: u64, len: usize) -> Option<Vec<u8>> {
        let walk = self.kernel.walk_context();
        let pa = super::super::idr::translate_any_kva(
            self.kernel.mem(),
            walk.cr3_pa,
            walk.page_offset,
            kva,
            walk.l5,
            walk.tcr_el1,
        )?;
        let mut buf = vec![0u8; len];
        let read = self.kernel.mem().read_bytes(pa, &mut buf);
        if read == len { Some(buf) } else { None }
    }

    fn is_arena_addr(&self, addr: u64) -> bool {
        // Single-line forwarder to the free helper so unit tests
        // can exercise the gate without a full
        // [`super::super::guest::GuestKernel`] mock. See
        // [`is_arena_addr_in_snapshot`] for the canonical body.
        is_arena_addr_in_snapshot(self.arena_snapshot, addr)
    }

    fn read_arena(&self, addr: u64, len: usize) -> Option<Vec<u8>> {
        // Body lives in the free helper so unit tests reach it without a
        // full GuestKernel. The caller (Ptr deref in btf_render) caps
        // `len` at one page before calling, but the helper pins the
        // single-page bound locally. The slow-path closure walks the
        // kernel page tables via `read_kva` using the arena's
        // `kern_vm_start` anchor — the same formula
        // [`super::super::arena::snapshot_arena`] uses at capture time.
        read_arena_in_snapshot(
            self.arena_snapshot,
            self.arena_page_index,
            addr,
            len,
            |kva, l| self.read_kva(kva, l),
        )
    }

    fn nr_cpu_ids(&self) -> u32 {
        self.num_cpus
    }

    fn cast_lookup(&self, parent_type_id: u32, member_byte_offset: u32) -> Option<CastHit> {
        // CastMap key is `(source_btf_type_id, field_byte_offset)`,
        // value is [`CastHit`] — see
        // [`super::super::cast_analysis::CastMap`]. The cast analyzer
        // already keys on the underlying *struct* type id (see
        // `bpf_map::resolve_to_struct_id`), matching the form
        // [`super::super::btf_render::render_struct`] threads down
        // as `parent_type_id` after [`super::super::btf_render::peel_modifiers_with_id`].
        //
        // Body lives in the free helper so unit tests reach the
        // None-short-circuit and keyed-lookup path without the
        // private [`AccessorMemReader`]. See [`cast_lookup_in_map`].
        cast_lookup_in_map(self.cast_map, parent_type_id, member_byte_offset)
    }
    fn resolve_arena_type(&self, addr: u64) -> Option<ArenaResolveHit> {
        // Forwarder to the free helper so unit tests can exercise
        // slot-position routing without a full
        // [`super::super::guest::GuestKernel`] mock. See
        // [`resolve_arena_type_with_static_fallback`] for the
        // canonical body — that helper consults the per-instance
        // sdt_alloc index first, then falls through to the
        // [`super::super::scx_static_alloc::ScxStaticRangeIndex`]
        // membership check. `self.requesting_btf_kva` flows into the
        // cross-BTF id-space gate so a slot's `target_type_id`
        // cannot be consumed by a renderer running against a
        // different program BTF.
        resolve_arena_type_with_static_fallback(
            self.arena_snapshot,
            self.arena_slot_index,
            self.scx_static_index,
            addr,
            self.requesting_btf_kva,
        )
    }
    fn resolve_arena_type_meta_fallback(&self, addr: u64) -> Option<ArenaResolveHit> {
        if !is_arena_addr_in_snapshot(self.arena_snapshot, addr) {
            return None;
        }
        self.sdt_alloc_metas
            .iter()
            .find(|m| m.target_type_id != 0)
            .map(|meta| ArenaResolveHit {
                target_type_id: meta.target_type_id,
                header_skip: meta.header_size,
            })
    }
    fn cross_btf_resolve_fwd(&self, name: &str, kind: FwdKind) -> Option<CrossBtfRef<'_>> {
        // Single-line forwarder to the free helper so unit tests
        // can exercise the resolve path without a full
        // [`super::super::guest::GuestKernel`] mock. See
        // [`resolve_cross_btf_fwd_in_index`] for the canonical
        // body.
        resolve_cross_btf_fwd_in_index(self.cross_btf_fwd_index, name, kind)
    }
    fn is_already_rendered(&self, addr: u64) -> bool {
        // Mask to the low-32 windowed shape the dump pre-pass keys
        // its [`ArenaSlotIndex`] on (every slot_start in the index
        // is `addr as u32`). The set forwarded here is built once
        // per dump pass from the same keys, so the membership check
        // matches the same address space the chase target was
        // resolved against. `None` (no rendered-slot index threaded
        // in) returns `false` per the trait default semantics.
        let Some(set) = self.rendered_slot_addrs else {
            return false;
        };
        set.contains(&(addr as u32))
    }
    fn alloc_size_types(&self) -> &[(u64, String)] {
        self.alloc_size_types
    }
}

/// Free helper: resolve a `BTF_KIND_FWD` name to a complete
/// definition in a sibling BTF. Factored out of
/// [`AccessorMemReader::cross_btf_resolve_fwd`] so unit tests can
/// call it with a synthetic [`CrossBtfFwdIndex`] without
/// constructing a [`super::super::guest::GuestKernel`].
///
/// The lookup keys on `name` exactly — anonymous Fwds (empty
/// resolved name) drop in the caller before reaching this helper.
/// `kind` is preserved end-to-end via the aggregate-kind match the
/// indexer applies at build time: only complete `Type::Struct(s)`
/// definitions match [`FwdKind::Struct`], only `Type::Union(s)`
/// match [`FwdKind::Union`].
///
/// Resolution order:
/// 1. In-binary embedded `.bpf.o` BTFs via `cross.fwd_index`.
///    Scheduler-defined types live here — preferred because
///    their layout matches what the scheduler's BPF program
///    actually allocated.
/// 2. vmlinux BTF (`cross.vmlinux_btf`), when supplied. Covers
///    kernel-defined structs the scheduler references (kernel
///    `_ctx` types, `task_struct`, …) but does not redefine in
///    its own BTF.
///
/// Returns `Some(CrossBtfRef { btf, type_id })` on the first
/// tier that produces a complete body matching `kind`
/// (embedded BTF wins over vmlinux). Returns `None` when
/// neither tier carries a matching complete body — the chase
/// falls through to the historical "forward declaration; body
/// not in this BTF" skip.
pub(super) fn resolve_cross_btf_fwd_in_index<'a>(
    cross_btf_fwd_index: Option<&'a CrossBtfFwdIndex<'a>>,
    name: &str,
    kind: FwdKind,
) -> Option<CrossBtfRef<'a>> {
    use btf_rs::Type;
    let cross = cross_btf_fwd_index?;
    // Tier 1: embedded `.bpf.o` BTFs.
    if let Some(entry) = cross.fwd_index.get(name)
        && let Some(btf_arc) = cross.btfs.get(entry.btfs_idx)
        && let Ok(ty) = btf_arc.resolve_type_by_id(entry.type_id)
        && matches!(
            (&ty, kind),
            (Type::Struct(_), FwdKind::Struct) | (Type::Union(_), FwdKind::Union)
        )
    {
        return Some(CrossBtfRef {
            btf: btf_arc.as_ref(),
            type_id: entry.type_id,
        });
    }
    None
}

impl<'a> GuestMemMapAccessor<'a> {
    /// Build a [`MemReader`] that resolves kernel KVAs via this
    /// accessor's [`super::super::guest::GuestKernel`] page-walker,
    /// optionally supplemented with an
    /// [`super::super::arena::ArenaSnapshot`] for `__arena`-pointer
    /// chase.
    ///
    /// `arena_page_index` is the dump pass's pre-built lookup table
    /// keyed by `user_addr`; pass an empty index when
    /// `arena_snapshot` is `None`. Building the index once outside
    /// the renderer (via [`build_arena_page_index`]) avoids a
    /// per-map rebuild on every `mem_reader` call.
    ///
    /// Pass `Some(&snap)` once the dump's pre-pass has captured the
    /// scheduler's arena pages — every `Type::Ptr` in subsequent
    /// BTF renders whose value falls in `[snap.user_vm_start ..
    /// + 4 GiB)` will resolve into the chase pipeline:
    /// captured pages return straight from `snap.pages`; pages
    /// beyond the snapshot's `MAX_ARENA_PAGES` sequential prefix
    /// translate via `snap.kern_vm_start + (ptr & 0xFFFF_FFFF)`
    /// through the host page-table walker (mirrors the formula
    /// [`super::super::arena::snapshot_arena`] uses at capture
    /// time and [`chase_sdt_data_payload`] uses for sdt_data
    /// payload chase). `None` disables arena chase (kernel kptrs
    /// to bpf_cpumask still resolve via `read_kva`).
    ///
    /// `num_cpus` is the guest's `nr_cpu_ids`, exposed through
    /// [`MemReader::nr_cpu_ids`] so the cpumask renderer can cap
    /// the bit walk at the guest's actual CPU count instead of
    /// rendering NR_CPUS-wide slab padding as phantom cpus.
    ///
    /// `cast_map` carries the BPF cast-analysis output for the
    /// scheduler's program — `Some(&map)` lets the renderer promote
    /// `u64` fields the analyzer flagged into typed-pointer renders
    /// via [`MemReader::cast_lookup`]; `None` keeps every `u64`
    /// rendered as a plain unsigned counter (the trait default).
    ///
    /// `arena_slot_index` is the dump pass's
    /// `slot_start → ArenaSlotInfo` lookup populated by the
    /// sdt_alloc pre-pass — see [`ArenaSlotIndex`]. `Some(&idx)`
    /// lets the renderer recover a forward-declared pointee's BTF
    /// type id (and the slot's header skip) from the captured
    /// allocator metadata via the
    /// [`MemReader::resolve_arena_type`] range lookup, when the
    /// program's own BTF carries only a `BTF_KIND_FWD`. `None`
    /// disables the bridge (no sdt_alloc allocator with a typed
    /// payload was discovered).
    ///
    /// `cross_btf_fwd_index` is the dump pass's cross-BTF Fwd
    /// resolution context — see [`CrossBtfFwdIndex`]. `Some(&idx)`
    /// lets the renderer chase a `BTF_KIND_FWD` whose body lives in
    /// a sibling embedded BPF object via
    /// [`MemReader::cross_btf_resolve_fwd`]. `None` (no scheduler
    /// binary was set on the builder, or the analyzer found no
    /// complete struct/union definitions) keeps the renderer's
    /// default "forward declaration; body not in this BTF" skip
    /// path intact.
    ///
    /// `rendered_slot_addrs` is the dump pass's set of low-32
    /// windowed slot starts that already appear in
    /// `report.sdt_allocations` (the sdt_alloc pre-pass surfaced
    /// them under the typed-allocator surface). `Some(&set)` lets
    /// the per-map renderer's [`MemReader::is_already_rendered`]
    /// short-circuit the arena chase when a TASK_STORAGE / HASH
    /// map's value pointer chases back into the same slot —
    /// preventing the same payload from appearing twice in the
    /// failure dump. `None` keeps the chase pipeline's existing
    /// behaviour intact.
    /// `requesting_btf_kva` identifies the program BTF the per-map
    /// renderer will use. Threaded into the
    /// [`MemReader::resolve_arena_type`] gate so the sdt_alloc
    /// bridge can suppress hits originating from a different
    /// program BTF's id space (multi-`.bpf.o` schedulers where each
    /// embedded object has its own BTF). Pass the map's
    /// [`super::super::bpf_map::BpfMapInfo::btf_kva`] when the per-map
    /// BTF is the program BTF loaded from that map; pass `0` when the
    /// renderer falls back to the caller-supplied vmlinux BTF (the
    /// gate then suppresses cross-BTF bridge hits conservatively).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn mem_reader(
        &self,
        arena_snapshot: Option<&'a super::super::arena::ArenaSnapshot>,
        arena_page_index: &'a ArenaPageIndex,
        num_cpus: u32,
        cast_map: Option<&'a CastMap>,
        arena_slot_index: Option<&'a ArenaSlotIndex>,
        sdt_alloc_metas: &'a [SdtAllocMeta],
        cross_btf_fwd_index: Option<&'a CrossBtfFwdIndex<'a>>,
        scx_static_index: Option<&'a super::super::scx_static_alloc::ScxStaticRangeIndex>,
        rendered_slot_addrs: Option<&'a std::collections::HashSet<u32>>,
        alloc_size_types: &'a [(u64, String)],
        requesting_btf_kva: u64,
    ) -> impl MemReader + 'a {
        AccessorMemReader {
            kernel: self.kernel(),
            arena_snapshot,
            arena_page_index,
            num_cpus,
            cast_map,
            arena_slot_index,
            sdt_alloc_metas,
            cross_btf_fwd_index,
            scx_static_index,
            rendered_slot_addrs,
            alloc_size_types,
            requesting_btf_kva,
        }
    }
}

/// Decoded sdt_alloc allocator metadata threaded into the per-map
/// renderer so the TASK_STORAGE / HASH arms can chase
/// `struct sdt_data __arena *` entry pointers into typed payload
/// renders.
///
/// `target_type_id` and `header_size` come from the sdt_alloc
/// pre-pass: `discover_payload_btf_id` matches `payload_size = elem_size
/// - sizeof(sdt_data)` against program-BTF struct sizes, and
/// [`super::super::sdt_alloc::SdtAllocOffsets::data_header_size`]
/// resolves the BTF size of `struct sdt_data` itself (the `union sdt_id
///   tid` header — `payload[]` flex array contributes 0 bytes per the
///   kernel layout). `elem_size` is the per-slot stride from
/// `pool.elem_size`; the renderer reads `elem_size` bytes from the arena
/// at each entry's `data` pointer, skips `header_size` bytes, and
/// renders the remaining `payload_size = elem_size - header_size` bytes
/// against `target_type_id`.
///
/// `kern_vm_start` is the kernel-side base of the arena's user_vm
/// window — `bpf_arena.kern_vm->addr + GUARD_HALF`, the same value
/// [`super::super::arena::snapshot_arena`] computes. The arms compose
/// a kernel virtual address from each entry's user-side `data`
/// pointer: `kva = kern_vm_start + (ptr & 0xFFFF_FFFF)`. That KVA
/// goes through the host page-table walker (via `MemReader::read_kva`,
/// implemented atop `translate_any_kva` on the guest kernel handle),
/// which can read ANY mapped arena page — `MemReader::read_arena` would
/// only succeed against the small subset of pages captured into the
/// pre-pass [`super::super::arena::ArenaSnapshot`] (the snapshot caps
/// `MAX_ARENA_PAGES` and would silently miss most live entries on a
/// scheduler with many tasks).
///
/// `allocator_name` is the .bss var name the metadata was discovered
/// under (e.g. `"scx_task_allocator"`, `"scx_cgrp_allocator"`); the
/// per-map selector [`select_sdt_alloc_meta`] uses it to disambiguate
/// when a scheduler declares more than one typed allocator.
#[derive(Debug, Clone)]
pub(super) struct SdtAllocMeta {
    pub(super) allocator_name: String,
    pub(super) elem_size: u64,
    pub(super) header_size: usize,
    pub(super) target_type_id: u32,
    pub(super) kern_vm_start: u64,
}

/// Pick the [`SdtAllocMeta`] entry whose allocator name best matches
/// `map_name`. Returns the first metadata entry whose normalized
/// allocator stem appears as a substring of `map_name`; ties are
/// broken by the longest stem so `scx_task_allocator` matches
/// `scx_task_map` rather than a coincidentally-shared `scx_` prefix.
///
/// Single-allocator schedulers fall through to the unique candidate
/// when no name matches — preserving the prior behavior for
/// schedulers that ship only one typed allocator. Returns `None`
/// when `metas` is empty OR when multiple allocators exist and the
/// map name disambiguates none of them (the renderer then degrades
/// to `payload: None` rather than risking a wrong-struct decode).
///
/// Stem extraction strips the trailing `_allocator` and the leading
/// `scx_` (the convention every in-tree allocator follows); the
/// remaining stem (`task`, `cgrp`, …) is the load-bearing token
/// the local-storage / hash maps include in their own names
/// (`scx_task_map`, `scx_cgrp_map`, etc.). A scheduler that
/// declares an allocator without those affixes degrades gracefully
/// — the substring match still fires when the raw allocator name
/// appears in the map name.
pub(super) fn select_sdt_alloc_meta<'a>(
    metas: &'a [SdtAllocMeta],
    map_name: &str,
) -> Option<&'a SdtAllocMeta> {
    if metas.is_empty() {
        return None;
    }
    if metas.len() == 1 {
        return Some(&metas[0]);
    }
    let stem = |name: &str| -> String {
        let s = name.strip_suffix("_allocator").unwrap_or(name);
        s.strip_prefix("scx_").unwrap_or(s).to_string()
    };
    metas
        .iter()
        .filter(|m| {
            let s = stem(&m.allocator_name);
            !s.is_empty() && map_name.contains(&s)
        })
        .max_by_key(|m| stem(&m.allocator_name).len())
}

/// Per-map render parameters that don't vary between maps in a
/// dump pass. Bundled so [`render_map`] takes one context borrow
/// plus the per-map [`BpfMapInfo`] rather than separately-named
/// arguments. Constructed once in [`super::dump_state`] and threaded
/// into every map render call.
///
/// `shared_arena` carries the pre-pass arena snapshot the renderer
/// uses both as the `MemReader` arena context and to short-circuit
/// the `BPF_MAP_TYPE_ARENA` arm so the same map's `out.arena` reuses
/// the captured pages instead of paying a second `snapshot_arena`.
/// The `u64` second slot is the snapshot's source `BpfMapInfo::map_kva`
/// — the arena-arm reuses the snapshot only when the rendered map's
/// `map_kva` matches.
///
/// `arena_page_index` is the dump pass's pre-built `user_addr →
/// page index` lookup; threaded as a borrow so each per-map
/// `mem_reader` call avoids rebuilding the table.
///
/// `sdt_alloc_metas` carries the discovered allocator metadata from
/// the sdt_alloc pre-pass; the TASK_STORAGE / HASH arms select the
/// matching entry by map name (see [`select_sdt_alloc_meta`]) to
/// expand the per-entry `struct sdt_data __arena *` pointer into a
/// typed payload render. Empty when no allocator with a typed
/// payload was found (older scheduler that doesn't link
/// `lib/sdt_alloc.bpf.c`, pre-pass found candidates but
/// `discover_payload_btf_id` returned 0 — e.g. ambiguous candidates
/// of the matching size — etc.).
pub(super) struct RenderMapCtx<'a> {
    pub(super) accessor: &'a GuestMemMapAccessor<'a>,
    /// Per-map BTF — `None` falls back to `vmlinux_btf` inside the
    /// renderer. Set per-map by [`super::dump_state`] from the
    /// program-BTF cache.
    pub(super) btf: Option<&'a Btf>,
    pub(super) num_cpus: u32,
    pub(super) arena_offsets: Option<&'a BpfArenaOffsets>,
    pub(super) shared_arena: Option<(&'a ArenaSnapshot, u64)>,
    pub(super) arena_page_index: &'a ArenaPageIndex,
    pub(super) sdt_alloc_metas: &'a [SdtAllocMeta],
    /// Cast analysis output for the scheduler's BPF program. Threaded
    /// into every per-map [`AccessorMemReader`] so
    /// [`MemReader::cast_lookup`] can promote `u64` fields the
    /// analyzer flagged into typed-pointer renders. `None` (no
    /// analysis available, e.g. older program-BTF without
    /// instructions) leaves the renderer's existing `u64`-as-counter
    /// behavior untouched.
    pub(super) cast_map: Option<&'a CastMap>,
    /// `slot_start → ArenaSlotInfo` index populated by the sdt_alloc
    /// pre-pass. Threaded into every per-map [`AccessorMemReader`]
    /// so [`MemReader::resolve_arena_type`] can range-lookup the
    /// allocator slot a chased arena address falls in and recover
    /// the real per-task / per-cgroup payload struct id (plus a
    /// `header_skip` byte count) when the program BTF carries only
    /// a `BTF_KIND_FWD` for the declared pointee. `None` (no
    /// sdt_alloc allocator with a typed payload was discovered)
    /// leaves the chase pipeline's "forward declaration" skip path
    /// intact.
    pub(super) arena_slot_index: Option<&'a ArenaSlotIndex>,
    /// Cross-BTF Fwd resolution context populated by the cast-
    /// analysis pre-pass. Threaded into every per-map
    /// [`AccessorMemReader`] so
    /// [`MemReader::cross_btf_resolve_fwd`] can resolve a
    /// `BTF_KIND_FWD` terminal whose body lives in a sibling
    /// embedded BPF object's BTF (the typical multi-`.bpf.objs`
    /// shape). `None` (no scheduler binary, or analyzer found no
    /// complete struct/union definitions) keeps the renderer's
    /// "forward declaration; body not in this BTF" skip path
    /// intact.
    pub(super) cross_btf_fwd_index: Option<&'a CrossBtfFwdIndex<'a>>,
    /// `start_low32 → size` index covering every live `scx_static`
    /// bump-allocator region — see
    /// [`super::super::scx_static_alloc::ScxStaticRangeIndex`]. The
    /// renderer's [`MemReader::resolve_arena_type`] consults this
    /// index AFTER `arena_slot_index` misses; on a hit, the bridge
    /// recognises the address as "in scx_static memory" but
    /// fails closed (`None`) — per-allocation type recovery is not
    /// possible without a per-call-site type hook from cast
    /// analysis (see the [`crate::monitor::scx_static_alloc`]
    /// module-level doc). `None` (no scheduler with an initialised
    /// `scx_static` instance was discovered) keeps the chase
    /// pipeline's existing skip path intact.
    pub(super) scx_static_index: Option<&'a super::super::scx_static_alloc::ScxStaticRangeIndex>,
    /// `slot_start` low-32 addresses for every allocator slot the
    /// dump pre-pass already rendered into `report.sdt_allocations`.
    /// Threaded into every per-map [`AccessorMemReader`] so
    /// [`MemReader::is_already_rendered`] short-circuits the arena
    /// chase when a TASK_STORAGE / HASH map's value points at a
    /// slot the typed-allocator surface already shows — preventing
    /// the same payload from appearing twice in the failure dump.
    /// `None` (no allocator pre-pass produced any rendered slot)
    /// keeps the chase pipeline's existing behaviour intact.
    pub(super) rendered_slot_addrs: Option<&'a std::collections::HashSet<u32>>,
    pub(super) alloc_size_types: &'a [(u64, String)],
}

/// Render `bytes` via BTF when both a `Btf` is available AND the
/// `type_id` is non-zero; otherwise fall through to a hex dump.
///
/// Centralizes the value-side render gate that ARRAY, HASH/LRU_HASH,
/// PERCPU_*HASH, PERCPU_ARRAY, and STRUCT_OPS arms previously
/// duplicated. Always produces a `RenderedValue` so the caller drops
/// the `Option<>` indirection that the key-side helper carries.
pub(super) fn render_value_or_hex(
    btf: Option<&Btf>,
    type_id: u32,
    bytes: &[u8],
    mem_reader: &dyn MemReader,
) -> RenderedValue {
    match (btf, type_id) {
        (Some(b), id) if id != 0 => render_value_with_mem(b, id, bytes, mem_reader),
        _ => RenderedValue::Bytes {
            hex: hex_dump(bytes),
        },
    }
}

/// Optional BTF render — `Some(rendered)` when both a `Btf` is
/// available AND the `type_id` is non-zero, `None` otherwise. Used on
/// the key side of HASH/LRU_HASH and PERCPU_*HASH entries where the
/// renderer keeps `key_hex` regardless and surfaces the rendered key
/// only when BTF allows it.
pub(super) fn render_key_optional(
    btf: Option<&Btf>,
    type_id: u32,
    bytes: &[u8],
    mem_reader: &dyn MemReader,
) -> Option<RenderedValue> {
    match (btf, type_id) {
        (Some(b), id) if id != 0 => Some(render_value_with_mem(b, id, bytes, mem_reader)),
        _ => None,
    }
}

/// Peel BTF modifier types (`Typedef`, `Const`, `Volatile`,
/// `Restrict`, `TypeTag`, `DeclTag`) from `start`, returning the
/// first non-modifier `Type` reached or `None` on resolve failure /
/// cycle limit. Thin wrapper over the canonical
/// [`super::super::btf_render::peel_modifiers_from_type`] so the
/// modifier-peel loop has one implementation across the dump
/// pipeline. The renderer's helper applies the same hop cap as the
/// previous local copy did and recognises the same modifier kinds
/// plus `DeclTag` (which the previous local list missed but the
/// kernel BPF pipeline does emit for global-variable types).
fn peel_modifiers(btf: &Btf, start: btf_rs::Type) -> Option<btf_rs::Type> {
    super::super::btf_render::peel_modifiers_from_type(btf, start)
}

/// Locate the byte offset of a `struct sdt_data __arena *` member
/// within `value_type_id`, returning `None` when no such member exists.
///
/// Walks the struct's members and, for each one, peels modifiers + a
/// single `Ptr` layer to inspect the pointee. A pointee `Type::Struct`
/// whose name is `"sdt_data"` makes that member the entry-payload
/// pointer the TASK_STORAGE arm follows into the captured arena. Only
/// the FIRST matching member is returned — schedulers today declare a
/// single `data` field per task-storage value type, and ambiguity at
/// this layer (two arena-`sdt_data` pointers in one value struct) is
/// not a shape ktstr supports.
///
/// Returns `None` for: non-struct value types, missing pointee struct,
/// pointee not named `sdt_data`, or any BTF resolve failure. The caller
/// degrades to `payload: None` on any None — the surface struct still
/// renders, only the per-entry typed payload is suppressed.
pub(super) fn find_sdt_data_field_offset(btf: &Btf, value_type_id: u32) -> Option<usize> {
    use btf_rs::Type;
    if value_type_id == 0 {
        return None;
    }
    // Peel modifiers from the value type id before reaching the
    // underlying struct/union.
    let value_ty = peel_modifiers(btf, btf.resolve_type_by_id(value_type_id).ok()?)?;
    let s = match value_ty {
        Type::Struct(s) | Type::Union(s) => s,
        _ => return None,
    };
    for member in &s.members {
        // Bit offsets are byte-aligned for non-bitfield struct fields;
        // the existing render_member arm rejects non-byte-aligned
        // non-bitfields, so skipping them here matches the renderer's
        // own handling.
        let bit_off = member.bit_offset() as usize;
        if !bit_off.is_multiple_of(8) {
            continue;
        }
        let byte_off = bit_off / 8;
        let Ok(member_ty) = btf.resolve_chained_type(member) else {
            continue;
        };
        // Peel modifiers on the member's type before checking for Ptr.
        let Some(t) = peel_modifiers(btf, member_ty) else {
            continue;
        };
        let ptr = match t {
            Type::Ptr(p) => p,
            _ => continue,
        };
        // Resolve the pointee through modifier chains too.
        let Ok(pointee_chained) = btf.resolve_chained_type(&ptr) else {
            continue;
        };
        let Some(pointee) = peel_modifiers(btf, pointee_chained) else {
            continue;
        };
        // Schedulers vary in whether `sdt_data` resolves to a full
        // BTF_KIND_STRUCT or a BTF_KIND_FWD forward declaration —
        // libbpf emits a Fwd when the pointee struct body is not
        // in the program's own BTF (the body lives in the
        // sdt_alloc library's BTF). Accept both shapes; the
        // pointer-offset is the same regardless of body
        // availability and the post-pass arena chase resolves the
        // body separately.
        let pointee_name: Option<String> = match pointee {
            Type::Struct(ref s) => btf.resolve_name(s).ok(),
            Type::Fwd(ref fwd) if fwd.is_struct() => btf.resolve_name(fwd).ok(),
            _ => continue,
        };
        if matches!(pointee_name.as_deref(), Some("sdt_data")) {
            return Some(byte_off);
        }
    }
    None
}

/// Resolve the per-ops struct BTF type id from the wrapper id
/// `bpf_struct_ops_<name>` libbpf stores in `bpf_map.btf_vmlinux_value_type_id`.
///
/// The wrapper struct has shape:
///   ```c
///   struct bpf_struct_ops_<name> {
///       struct bpf_struct_ops_common_value common;
///       struct <user_ops_name> data;  // e.g. sched_ext_ops
///   };
///   ```
/// The dump path reads the `data` flex tail (per
/// [`super::super::btf_offsets::StructOpsOffsets`]'s `value_data`
/// offset) so the renderer wants the type id of the `data` member's
/// type, not the wrapper. Walks the wrapper's members for one named
/// `"data"` and returns its resolved type id.
///
/// Returns `None` when:
/// - `wrapper_type_id` is 0 (kernel `btf_vmlinux_value_type_id`
///   field unresolved or never populated).
/// - The id resolves to a non-struct (corrupted read).
/// - The wrapper has no `data` member (struct_ops shape changed
///   in a kernel newer than this dump renderer — the consumer
///   sees raw hex with a doc-friendly comment rather than a
///   silent mis-decode).
/// - The member's chained type fails to resolve.
fn resolve_struct_ops_payload_type_id(btf: &Btf, wrapper_type_id: u32) -> Option<u32> {
    use btf_rs::{BtfType, Type};
    if wrapper_type_id == 0 {
        return None;
    }
    let wrapper_ty = peel_modifiers(btf, btf.resolve_type_by_id(wrapper_type_id).ok()?)?;
    let s = match wrapper_ty {
        Type::Struct(s) => s,
        _ => return None,
    };
    for member in &s.members {
        let Ok(name) = btf.resolve_name(member) else {
            continue;
        };
        if name != "data" {
            continue;
        }
        // The `data` member is declared as the user struct directly
        // (no pointer indirection — the common header + flex array
        // shape — see kernel/bpf/bpf_struct_ops.c). The member's own
        // `r#type` field IS the user struct type id (modulo modifier
        // chains the BPF compiler emits for global variable types).
        // Peel modifiers via the chained type, then emit the type id
        // the renderer can use directly.
        let raw_id = member.get_type_id()?;
        let chained = btf.resolve_type_by_id(raw_id).ok()?;
        let inner = peel_modifiers(btf, chained)?;
        // Re-derive the id of the peeled type. Peeling can hop
        // through modifiers without changing the semantic struct,
        // but the renderer needs the id of whatever type the bytes
        // describe. resolve_chained_type would loop us back through
        // peel_modifiers's exit type, so we just walk the original
        // raw_id forward through identical kinds.
        return Some(match inner {
            Type::Struct(_) | Type::Union(_) | Type::Int(_) | Type::Enum(_) | Type::Enum64(_) => {
                // Walk modifiers forward until we land on a non-
                // modifier; report the peeled id. Reuses the
                // canonical helper in btf_render so this caller
                // shares the renderer's modifier-set definition
                // (Typedef / Const / Volatile / Restrict / TypeTag
                // / DeclTag) and hop cap.
                super::super::btf_render::peel_modifiers_with_id(btf, raw_id)?.1
            }
            _ => return None,
        });
    }
    None
}

/// Read the `struct sdt_data __arena *` pointer out of `value_bytes` at
/// `offset` and render its payload via `meta.target_type_id`.
///
/// Returns `None` when:
/// - `btf`, `field_offset`, or `meta` is missing (caller skipped
///   the BTF resolve, no allocator metadata was discovered, etc.).
/// - The value bytes are too short to hold an 8-byte pointer at the
///   resolved offset.
/// - The arena pointer reads as 0 (entry not yet populated by
///   `scx_task_alloc`).
/// - `meta.target_type_id` is 0 (allocator's payload type was
///   ambiguous or unresolved).
/// - `meta.elem_size` is smaller than `meta.header_size` (corrupt
///   pre-pass metadata).
/// - `meta.kern_vm_start == 0` (arena pre-pass found no kernel-side
///   anchor — the walk has no way to compute the KVA).
/// - The page-table walker returns `None` (the arena page the entry
///   points into is unmapped at freeze time).
///
/// Reads the OUTER `data` pointer via `mem_reader.read_kva` — the host
/// page-table walker, which works against every mapped guest kernel
/// page regardless of whether it was captured in the dump pre-pass
/// snapshot. This is the right primitive for the outer chase: live
/// schedulers with many tasks point most entries past the snapshot's
/// `MAX_ARENA_PAGES` sequential prefix, so a `read_arena`-only path
/// would silently miss them. Mirrors the read path
/// `super::super::sdt_alloc::TreeWalker::translate_arena_ptr` uses
/// for the sdt_alloc allocator walk.
///
/// The same `mem_reader` is threaded into `render_value_with_mem` for
/// the payload's inner render. Arena pointers embedded in the payload
/// (e.g. an arena-allocated child struct, a `cbw_cgrp_entry.cgx`
/// pointing back into the cgroup-context arena) hit the
/// btf_render Ptr arm's `is_arena_addr` branch, which calls
/// [`MemReader::read_arena`]. Per the [`AccessorMemReader`]
/// implementation, that path tries the snapshot's frozen pages
/// first (O(1) HashMap lookup) and falls through to the kernel
/// page-table walker for pages outside the captured prefix —
/// the same fallback this outer call makes directly. Recursion
/// thus has identical reach to the outer chase, not the
/// snapshot-only subset earlier docs implied.
pub(super) fn chase_sdt_data_payload(
    btf: Option<&Btf>,
    field_offset: Option<usize>,
    meta: Option<&SdtAllocMeta>,
    value_bytes: &[u8],
    mem_reader: &dyn MemReader,
) -> Option<RenderedValue> {
    let btf = btf?;
    let off = field_offset?;
    let meta = meta?;
    if meta.target_type_id == 0 {
        return None;
    }
    if meta.elem_size as usize <= meta.header_size {
        return None;
    }
    if meta.kern_vm_start == 0 {
        return None;
    }
    // Pointer read — the value bytes are little-endian per the host's
    // (x86_64 / aarch64) byte order; `__arena *` is a 64-bit pointer
    // on the supported architectures.
    let end = off.checked_add(8)?;
    let slice = value_bytes.get(off..end)?;
    let mut buf = [0u8; 8];
    buf.copy_from_slice(slice);
    let data_ptr = u64::from_le_bytes(buf);
    if data_ptr == 0 {
        return None;
    }
    // Compose the kernel virtual address from the user-side arena
    // pointer. Mirrors the formula in
    // [`super::super::sdt_alloc::TreeWalker::translate_arena_ptr`]:
    // the kernel mounts arena pages with the user_vm window's low 32
    // bits offset against `kern_vm_start`.
    let kva = meta.kern_vm_start.wrapping_add(data_ptr & 0xFFFF_FFFF);
    let elem_bytes = mem_reader.read_kva(kva, meta.elem_size as usize)?;
    let payload = elem_bytes.get(meta.header_size..)?;
    Some(render_value_with_mem(
        btf,
        meta.target_type_id,
        payload,
        mem_reader,
    ))
}

/// Static explanation strings for map types whose contents the dump
/// path doesn't decode into a typed [`RenderedValue`].
///
/// Each entry pairs the kernel-uapi `enum bpf_map_type` discriminant
/// with the operator-visible reason. Ordered by discriminant so a
/// reader can scan in numeric order. Multi-discriminant entries
/// (e.g. `DEVMAP | DEVMAP_HASH`) appear once per discriminant — the
/// look-up is exact-match, not pattern-match.
///
/// The dispatch is split between this table (shape-based reasons that
/// don't change per map) and the explicit match arms in
/// [`render_map`] (arms that read real data, walk a sub-structure,
/// or compute a per-map message). Keep the two sides lockstep when
/// adding a new map type.
/// True when `map_name` matches libbpf's `.rodata.str1.1`
/// string-literal section convention.
///
/// clang emits string literals into a `mergeable` section named
/// `.rodata.str1.1` (`SHF_MERGE | SHF_STRINGS`, ent_size=1, align=1
/// — see the ELF spec on string-merge sections); libbpf prefixes the
/// `<obj_name>` so the kernel registers the map as
/// `<obj>.rodata.str1.1`. The section has NO Datasec entry in BTF
/// (the strings aren't typed globals), which is why the dump path
/// can't BTF-render the value buffer and falls through to hex.
/// Detect by suffix so any obj_name prefix matches.
fn is_str_literal_section(map_name: &str) -> bool {
    map_name.ends_with(".rodata.str1.1")
}

/// Render `bytes` as a printable ASCII dump for the
/// `.rodata.str1.1` arm. Printable bytes (0x20..=0x7E) pass through;
/// every other byte is rendered as `\xHH` so a NUL-separated string
/// literal section still reads as concatenated literals. Non-string
/// content (e.g. a corrupted read) shows up as a long `\xHH` run
/// rather than producing UTF-8 replacement-character noise that
/// would obscure the boundary between adjacent literals.
fn ascii_str_dump(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(bytes.len());
    for &b in bytes {
        if (0x20..=0x7E).contains(&b) {
            s.push(b as char);
        } else {
            // unwrap is safe: write! to String never fails.
            let _ = write!(s, "\\x{b:02x}");
        }
    }
    s
}

/// Render the kernel `enum bpf_map_type` discriminant as the
/// matching `BPF_MAP_TYPE_<NAME>` short name (lowercase, prefix
/// stripped — e.g. `27 → "ringbuf"`). Returns `None` for
/// discriminants this dump renderer doesn't know about (kernel
/// newer than the renderer, post-INSN_ARRAY additions); callers
/// fall through to the raw number to keep diagnostics readable.
///
/// Pinned name table — adding a new map type to
/// [`super::super::bpf_map`] needs a matching entry here so the
/// failure-dump header line surfaces the symbolic name instead of
/// a bare integer the operator has to translate by hand.
pub(super) fn map_type_name(map_type: u32) -> Option<&'static str> {
    Some(match map_type {
        BPF_MAP_TYPE_HASH => "hash",
        BPF_MAP_TYPE_ARRAY => "array",
        BPF_MAP_TYPE_PROG_ARRAY => "prog_array",
        BPF_MAP_TYPE_PERF_EVENT_ARRAY => "perf_event_array",
        BPF_MAP_TYPE_PERCPU_HASH => "percpu_hash",
        BPF_MAP_TYPE_PERCPU_ARRAY => "percpu_array",
        BPF_MAP_TYPE_STACK_TRACE => "stack_trace",
        BPF_MAP_TYPE_CGROUP_ARRAY => "cgroup_array",
        BPF_MAP_TYPE_LRU_HASH => "lru_hash",
        BPF_MAP_TYPE_LRU_PERCPU_HASH => "lru_percpu_hash",
        BPF_MAP_TYPE_LPM_TRIE => "lpm_trie",
        BPF_MAP_TYPE_ARRAY_OF_MAPS => "array_of_maps",
        BPF_MAP_TYPE_HASH_OF_MAPS => "hash_of_maps",
        BPF_MAP_TYPE_DEVMAP => "devmap",
        BPF_MAP_TYPE_SOCKMAP => "sockmap",
        BPF_MAP_TYPE_CPUMAP => "cpumap",
        BPF_MAP_TYPE_XSKMAP => "xskmap",
        BPF_MAP_TYPE_SOCKHASH => "sockhash",
        BPF_MAP_TYPE_CGROUP_STORAGE => "cgroup_storage",
        BPF_MAP_TYPE_REUSEPORT_SOCKARRAY => "reuseport_sockarray",
        BPF_MAP_TYPE_PERCPU_CGROUP_STORAGE => "percpu_cgroup_storage",
        BPF_MAP_TYPE_QUEUE => "queue",
        BPF_MAP_TYPE_STACK => "stack",
        BPF_MAP_TYPE_SK_STORAGE => "sk_storage",
        BPF_MAP_TYPE_DEVMAP_HASH => "devmap_hash",
        BPF_MAP_TYPE_STRUCT_OPS => "struct_ops",
        BPF_MAP_TYPE_RINGBUF => "ringbuf",
        BPF_MAP_TYPE_INODE_STORAGE => "inode_storage",
        BPF_MAP_TYPE_TASK_STORAGE => "task_storage",
        BPF_MAP_TYPE_BLOOM_FILTER => "bloom_filter",
        BPF_MAP_TYPE_USER_RINGBUF => "user_ringbuf",
        BPF_MAP_TYPE_CGRP_STORAGE => "cgrp_storage",
        BPF_MAP_TYPE_ARENA => "arena",
        BPF_MAP_TYPE_INSN_ARRAY => "insn_array",
        _ => return None,
    })
}

/// Operator-facing error set into `FailureDumpMap.error` when a
/// BPF_MAP_TYPE_ARENA map is captured but its BTF offsets did not
/// resolve. A `const` (not an inline literal) so the byte-exact wire
/// string is pinnable by a test without constructing a full render_map
/// context.
pub(super) const ARENA_OFFSETS_UNAVAILABLE_MSG: &str =
    "arena BTF offsets unavailable (kernel lacks struct bpf_arena?)";

pub(super) const MAP_TYPE_EXPLANATIONS: &[(u32, &str)] = &[
    (
        BPF_MAP_TYPE_CGROUP_STORAGE,
        "deprecated cgroup-attached storage; use CGRP_STORAGE on newer kernels",
    ),
    (
        BPF_MAP_TYPE_PERCPU_CGROUP_STORAGE,
        "deprecated cgroup-attached storage; use CGRP_STORAGE on newer kernels",
    ),
    (
        BPF_MAP_TYPE_QUEUE,
        "QUEUE/STACK are destructive (peek shows only the head; pop consumes); \
         no enumeration API",
    ),
    (
        BPF_MAP_TYPE_STACK,
        "QUEUE/STACK are destructive (peek shows only the head; pop consumes); \
         no enumeration API",
    ),
    (
        BPF_MAP_TYPE_BLOOM_FILTER,
        "BLOOM_FILTER is a probabilistic set; no key enumeration is possible",
    ),
    (
        BPF_MAP_TYPE_LPM_TRIE,
        "LPM_TRIE walker not implemented (keyed by prefixlen + data); \
         use bpf(2) BPF_MAP_GET_NEXT_KEY for live-host iteration",
    ),
    (
        BPF_MAP_TYPE_INSN_ARRAY,
        "INSN_ARRAY stores BPF instruction targets used by the verifier \
         for indirect jumps; values are kernel-side program data",
    ),
];

pub(super) fn render_map(ctx: &RenderMapCtx<'_>, info: &BpfMapInfo) -> FailureDumpMap {
    let &RenderMapCtx {
        accessor,
        btf,
        num_cpus,
        arena_offsets,
        shared_arena,
        arena_page_index,
        sdt_alloc_metas,
        cast_map,
        arena_slot_index,
        cross_btf_fwd_index,
        scx_static_index,
        rendered_slot_addrs,
        alloc_size_types,
    } = ctx;
    let mem_reader = accessor.mem_reader(
        shared_arena.map(|(snap, _map_kva)| snap),
        arena_page_index,
        num_cpus,
        cast_map,
        arena_slot_index,
        sdt_alloc_metas,
        cross_btf_fwd_index,
        scx_static_index,
        rendered_slot_addrs,
        alloc_size_types,
        // The per-map renderer reads through `info.btf_kva`'s
        // program BTF when the map carries its own BTF (see
        // [`super::dump_state`]'s map_btf selection logic). Threading
        // this kva into the mem_reader lets
        // [`MemReader::resolve_arena_type`] gate the sdt_alloc bridge
        // against the slot's `source_btf_kva` so a hit cannot leak
        // into the wrong id space. `0` (no per-map program BTF) is
        // the conservative-suppress sentinel.
        info.btf_kva,
    );
    // Per-map allocator selection. The TASK_STORAGE / HASH chase
    // arms consult this to pick the matching allocator metadata when
    // a scheduler declares more than one (e.g. per-task and
    // per-cgroup); single-allocator schedulers fall through to the
    // unique candidate. See [`select_sdt_alloc_meta`] for the
    // matching rule.
    let info_name = info.name();
    let sdt_alloc_meta: Option<SdtAllocMeta> =
        select_sdt_alloc_meta(sdt_alloc_metas, &info_name).cloned();
    let mut out = FailureDumpMap {
        name: info_name.into_owned(),
        map_kva: info.map_kva,
        map_type: info.map_type,
        value_size: info.value_size,
        max_entries: info.max_entries,
        value: None,
        entries: Vec::new(),
        array_entries: Vec::new(),
        percpu_entries: Vec::new(),
        percpu_hash_entries: Vec::new(),
        arena: None,
        ringbuf: None,
        stack_trace: None,
        fd_array: None,
        error: None,
    };

    match info.map_type {
        BPF_MAP_TYPE_ARRAY => {
            if info.max_entries <= 1 {
                // Single-entry ARRAY: the global-section maps
                // (.bss / .data / .rodata) declare value_size as the
                // section size and max_entries = 1, so one read of the
                // whole value region renders the section.
                //
                // libbpf packs string literals into a section named
                // `<obj>.rodata.str1.1` with NO BTF Datasec entry —
                // `btf_value_type_id` is 0 and the renderer would
                // otherwise emit raw hex. The string-merge section's
                // bytes are concatenated nul-terminated ASCII (clang's
                // `.str1.1` mergeable section produces 1-byte-stride
                // 1-alignment string data per the ELF spec); render
                // them as a printable ASCII dump with non-printable
                // bytes escaped so operators see the actual literals.
                // Re-use `out.name` (computed once above) instead of
                // calling `info.name()` again — it allocates on a
                // non-UTF-8 name and even the ASCII fast path rebuilds
                // a `Cow`.
                match accessor.read_value(info, 0, info.value_size as usize) {
                    Some(bytes) => {
                        out.value = Some(if is_str_literal_section(&out.name) {
                            RenderedValue::Bytes {
                                hex: ascii_str_dump(&bytes),
                            }
                        } else {
                            render_value_or_hex(btf, info.btf_value_type_id, &bytes, &mem_reader)
                        });
                    }
                    None => {
                        out.error = Some("ARRAY value region unreadable (unmapped page?)".into());
                    }
                }
            } else {
                // Multi-entry ARRAY: entries are contiguous at
                // `bpf_array.value` with stride round_up(value_size, 8)
                // (kernel/bpf/arraymap.c:93,167-176). The BTF type id
                // `btf_value_type_id` describes one entry, so the
                // renderer is called per-key for keys
                // 0..min(max_entries, MAX_ARRAY_KEYS). A key whose page
                // is unmapped lands as `value: None` and the loop never
                // aborts, so one bad key cannot mask the rest; the
                // unreadable count and any cap truncation surface in
                // `error`.
                let limit = info.max_entries.min(MAX_ARRAY_KEYS);
                let mut unreadable: u32 = 0;
                for key in 0..limit {
                    let value = match accessor.read_array(info, key) {
                        Some(bytes) => Some(render_value_or_hex(
                            btf,
                            info.btf_value_type_id,
                            &bytes,
                            &mem_reader,
                        )),
                        None => {
                            unreadable += 1;
                            None
                        }
                    };
                    out.array_entries.push(FailureDumpArrayEntry { key, value });
                }
                // Mirror the PERCPU_ARRAY truncation prose so log
                // scrapers see a consistent shape across cap
                // diagnostics; append the per-key read-failure count
                // when any key was unmapped.
                let mut notes: Vec<String> = Vec::new();
                if info.max_entries > MAX_ARRAY_KEYS {
                    notes.push(format!(
                        "ARRAY truncated at {MAX_ARRAY_KEYS} keys (max_entries={})",
                        info.max_entries
                    ));
                }
                if unreadable > 0 {
                    notes.push(format!("{unreadable} keys unreadable"));
                }
                if !notes.is_empty() {
                    out.error = Some(notes.join("; "));
                }
            }
        }
        BPF_MAP_TYPE_HASH | BPF_MAP_TYPE_LRU_HASH => {
            // HASH and LRU_HASH share the same htab_elem layout
            // (`kernel/bpf/hashtab.c::htab_elem_value`); LRU just
            // adds an eviction policy on top, so the dump path is
            // identical.
            //
            // Hard-cap at MAX_HASH_ENTRIES to keep a million-entry
            // hash from OOMing the host renderer. `iter_hash_map`
            // already enforces its own much-larger HTAB_ITER_MAX
            // (1_000_000) inside the bucket walk, but a million
            // [`RenderedValue`] trees would still pin gigabytes
            // here — surface the truncation in `out.error` so the
            // consumer sees that the rendered slice is partial.
            let raw_entries = accessor.iter_hash_map(info);
            let truncated = raw_entries.len() > MAX_HASH_ENTRIES;
            // Resolve the sdt_data arena-pointer field offset once,
            // outside the per-entry loop. Some sched_ext schedulers
            // declare HASH-shaped maps (e.g. `scx_task_map`) whose
            // value type carries a `struct sdt_data __arena *` —
            // chase it the same way the TASK_STORAGE arm does so
            // these schedulers see their per-task payload in the
            // dump rather than just the surface struct. `None`
            // when the value type lacks the pointer, no allocator
            // metadata was discovered, or BTF resolve fails — the
            // chase below short-circuits and `payload` stays None.
            let sdt_data_field = btf
                .zip(sdt_alloc_meta.as_ref())
                .and_then(|(b, _)| find_sdt_data_field_offset(b, info.btf_value_type_id));
            for (k, v) in raw_entries.into_iter().take(MAX_HASH_ENTRIES) {
                let key = render_key_optional(btf, info.btf_key_type_id, &k, &mem_reader);
                let value = render_key_optional(btf, info.btf_value_type_id, &v, &mem_reader);
                let payload = chase_sdt_data_payload(
                    btf,
                    sdt_data_field,
                    sdt_alloc_meta.as_ref(),
                    &v,
                    &mem_reader,
                );
                out.entries.push(FailureDumpEntry {
                    key,
                    key_hex: hex_dump(&k),
                    value,
                    value_hex: hex_dump(&v),
                    payload,
                });
            }
            if truncated {
                out.error = Some(format!("hash map truncated at {MAX_HASH_ENTRIES} entries"));
            }
        }
        BPF_MAP_TYPE_PERCPU_HASH | BPF_MAP_TYPE_LRU_PERCPU_HASH => {
            // PERCPU_HASH / LRU_PERCPU_HASH: each htab_elem stores a
            // `void __percpu *` at the htab_elem_value position; we
            // dereference per-CPU through `__per_cpu_offset[cpu]`
            // (`kernel/bpf/hashtab.c::htab_percpu_map_lookup_elem`).
            // Same MAX_HASH_ENTRIES truncation policy as plain HASH.
            let raw_entries = accessor.iter_percpu_hash_map(info, num_cpus);
            let truncated = raw_entries.len() > MAX_HASH_ENTRIES;
            for (k, per_cpu_bytes) in raw_entries.into_iter().take(MAX_HASH_ENTRIES) {
                let key = render_key_optional(btf, info.btf_key_type_id, &k, &mem_reader);
                let per_cpu = per_cpu_bytes
                    .into_iter()
                    .map(|maybe_bytes| {
                        maybe_bytes.map(|b| {
                            render_value_or_hex(btf, info.btf_value_type_id, &b, &mem_reader)
                        })
                    })
                    .collect();
                out.percpu_hash_entries.push(FailureDumpPercpuHashEntry {
                    key,
                    key_hex: hex_dump(&k),
                    per_cpu,
                });
            }
            if truncated {
                out.error = Some(format!(
                    "percpu hash map truncated at {MAX_HASH_ENTRIES} entries"
                ));
            }
        }
        BPF_MAP_TYPE_PERCPU_ARRAY => {
            let limit = info.max_entries.min(MAX_PERCPU_KEYS);
            for key in 0..limit {
                let per_cpu_bytes = accessor.read_percpu_array(info, key, num_cpus);
                let per_cpu = per_cpu_bytes
                    .into_iter()
                    .map(|maybe_bytes| {
                        maybe_bytes.map(|b| {
                            render_value_or_hex(btf, info.btf_value_type_id, &b, &mem_reader)
                        })
                    })
                    .collect();
                out.percpu_entries
                    .push(FailureDumpPercpuEntry { key, per_cpu });
            }
            // Surface PERCPU_ARRAY key truncation, mirroring the
            // ARRAY (key 0 of N) and HASH (entries cap) patterns:
            // when the map declares more keys than [`MAX_PERCPU_KEYS`],
            // the dump only walks the first MAX_PERCPU_KEYS slots and
            // the consumer needs to know the rest are dropped.
            if info.max_entries > MAX_PERCPU_KEYS {
                out.error = Some(format!(
                    "PERCPU_ARRAY truncated at {MAX_PERCPU_KEYS} keys (max_entries={})",
                    info.max_entries,
                ));
            }
        }
        BPF_MAP_TYPE_ARENA => {
            // Arena maps render in two phases:
            //
            //   1. Page-granular: arena pages live in vmalloc space
            //      and translate via the existing PTE walker. Each
            //      mapped page surfaces here as a 4 KiB ArenaPage —
            //      raw bytes the operator can post-process against
            //      the program's own layout documentation.
            //
            //   2. Structured (sdt_alloc post-pass): when the
            //      scheduler links `lib/sdt_alloc.bpf.c`, the
            //      `dump_state` post-pass walks `scx_allocator`'s
            //      radix tree and produces named-field
            //      [`super::super::sdt_alloc::SdtAllocEntry`]
            //      records under
            //      [`super::FailureDumpReport::sdt_allocations`].
            //      That phase is gated on the program BTF carrying
            //      `struct scx_allocator` — schedulers that don't
            //      use sdt_alloc still get the page-granular
            //      fallback from this arm.
            //
            // Both representations land in the same dump so a
            // consumer can pick whichever fits — raw bytes for ad
            // hoc post-processing, structured records for typed
            // field views.
            match arena_offsets {
                Some(off) => {
                    // Reuse the dump-state pre-pass snapshot when this
                    // is the same arena map: the pre-pass
                    // (`shared_arena_snapshot` in
                    // [`super::dump_state`]) has already paid the
                    // freeze-path cost of walking every mapped page,
                    // and re-running `snapshot_arena` here would do
                    // the same vmalloc walk a second time on the
                    // freeze hot path. The map_kva match is exact:
                    // `BpfMapInfo::map_kva` is the kernel virtual
                    // address of the `struct bpf_map` in the IDR walk
                    // so two distinct ARENA maps cannot share it.
                    // Other arena maps (a multi-object scheduler
                    // binding two separate arenas) fall through to a
                    // fresh `snapshot_arena` — the shared pre-pass
                    // only covers the first arena which is what
                    // `__arena` pointers across the program reference.
                    let snap = match shared_arena {
                        Some((shared_snap, shared_map_kva)) if shared_map_kva == info.map_kva => {
                            shared_snap.clone()
                        }
                        _ => snapshot_arena(accessor.kernel(), info, off),
                    };
                    out.arena = Some(snap);
                }
                None => {
                    out.error = Some(ARENA_OFFSETS_UNAVAILABLE_MSG.into());
                }
            }
        }
        BPF_MAP_TYPE_STRUCT_OPS => {
            // STRUCT_OPS embeds the registered kernel struct (e.g.
            // `sched_ext_ops` for ktstr's scx-ktstr fixture) inside
            // `bpf_struct_ops_map.kvalue.data`. `find_all_bpf_maps`
            // sets `value_kva = kvalue + data_off` so this arm can
            // share the ARRAY read path.
            //
            // libbpf zeroes `map->btf_value_type_id` for STRUCT_OPS
            // (`tools/lib/bpf/libbpf.c::bpf_object__create_maps`,
            // case BPF_MAP_TYPE_STRUCT_OPS) and instead populates the
            // kernel-side `btf_vmlinux_value_type_id` with the
            // `bpf_struct_ops_<name>` wrapper id from vmlinux BTF.
            // The wrapper's `data` member's pointee type is the
            // per-ops struct (e.g. `sched_ext_ops`) — the actual
            // payload type the BTF renderer can name fields with.
            // The read length is `value_size - data_off` to match
            // the per-ops struct footprint (the kernel allocates the
            // wrapper-inclusive `vt->size` and the data region is
            // the tail flex array).
            //
            // Early-return when struct_ops_offsets are absent: a
            // `data_off=0` fallback would read `value_size` bytes
            // (the wrapper-inclusive size) under a type id that
            // describes only the data payload. The renderer would
            // then over-read past the typed footprint into the
            // common header's bytes — silent miscoding rather than
            // a clean diagnostic.
            let Some(so) = accessor.offsets().struct_ops_offsets.as_ref() else {
                out.error = Some(
                    "STRUCT_OPS value unreadable: bpf_struct_ops_map BTF offsets unresolved \
                     (kernel without struct_ops support, or vmlinux BTF stripped of \
                     bpf_struct_ops_map / bpf_struct_ops_value)."
                        .into(),
                );
                return out;
            };
            let data_off = so.value_data;
            let data_len = (info.value_size as usize).saturating_sub(data_off);
            // Resolve the per-ops struct type id. Try the order
            // libbpf populates the fields in:
            //   1. `btf_value_type_id` — non-zero only on older
            //      libbpf builds that didn't zero it.
            //   2. `btf_vmlinux_value_type_id` — wrapper id; walk
            //      `wrapper.data` to its pointee struct.
            // The renderer then decodes the data bytes against that
            // type, so each `void *` ops field surfaces as a
            // [`RenderedValue::Ptr`] hex address with the BTF-named
            // member names alongside.
            let payload_type_id = if info.btf_value_type_id != 0 {
                info.btf_value_type_id
            } else if info.btf_vmlinux_value_type_id != 0 {
                btf.and_then(|b| {
                    resolve_struct_ops_payload_type_id(b, info.btf_vmlinux_value_type_id)
                })
                .unwrap_or(0)
            } else {
                0
            };
            match accessor.read_value(info, 0, data_len) {
                Some(bytes) => {
                    out.value = Some(render_value_or_hex(
                        btf,
                        payload_type_id,
                        &bytes,
                        &mem_reader,
                    ));
                }
                None => {
                    out.error = Some(
                        "STRUCT_OPS value unreadable: value region unmapped. Live-host \
                         backend reads via BPF_MAP_LOOKUP_ELEM at key=0."
                            .into(),
                    );
                }
            }
        }
        BPF_MAP_TYPE_TASK_STORAGE
        | BPF_MAP_TYPE_INODE_STORAGE
        | BPF_MAP_TYPE_SK_STORAGE
        | BPF_MAP_TYPE_CGRP_STORAGE => {
            // Local-storage walker: iterates
            // `bpf_local_storage_map.buckets[i].list` (regular hlist —
            // NOT `hlist_nulls` like `bpf_htab`; the kernel uses
            // `INIT_HLIST_HEAD` in `bpf_local_storage_map_alloc`) and
            // surfaces each `bpf_local_storage_elem` as one
            // `FailureDumpEntry`. The "key" side carries the owning
            // object's KVA (8-byte LE in `key_hex`) — `task_struct`
            // for TASK_STORAGE, `inode` / `sock` / `cgroup` for the
            // shape-identical INODE / SK / CGRP_STORAGE variants.
            // The "value" side carries the BTF-rendered
            // `sdata.data[]` payload.
            //
            // All four kernel map types share the
            // `bpf_local_storage_map` layout
            // (`include/linux/bpf_local_storage.h`), so one walker
            // plus [`super::super::btf_offsets::TaskStorageOffsets`]
            // covers them all — the per-type difference is only the
            // owner type, which the walker treats as opaque.
            //
            // Same `MAX_HASH_ENTRIES` truncation policy as plain HASH:
            // a million-entry storage map would otherwise pin gigabytes
            // of [`RenderedValue`] trees here. The walker's own cap
            // (`TASK_STORAGE_ITER_MAX = 1_000_000`) is the safety
            // bound against pointer-cycle corruption; this cap is the
            // operator-visible report-size bound.
            //
            // Empty-result handling matches HASH: no entries means no
            // entries — either the map has no live owners attached,
            // or `task_storage_offsets` is missing on this kernel.
            // Both states produce an empty `entries` list with no
            // error.
            let raw_entries = accessor.iter_task_storage(info);
            let truncated = raw_entries.len() > MAX_HASH_ENTRIES;
            // Resolve the offset of the value type's `struct sdt_data
            // __arena *` member once, before the per-entry loop, so each
            // entry can pull the arena pointer from raw value bytes
            // without re-walking BTF. `None` when the value type isn't a
            // struct, lacks an `sdt_data` arena pointer member, or BTF
            // resolution fails — every gate degrades to `payload: None`.
            let sdt_data_field = btf
                .zip(sdt_alloc_meta.as_ref())
                .and_then(|(b, _)| find_sdt_data_field_offset(b, info.btf_value_type_id));
            for (k, v) in raw_entries.into_iter().take(MAX_HASH_ENTRIES) {
                // `k` is the 8-byte LE owner KVA produced by
                // [`super::super::bpf_map::local_storage::iter_local_storage_entries`]
                // — `task_struct` for TASK_STORAGE; `inode` /
                // `sock` / `cgroup` for INODE / SK / CGRP_STORAGE.
                // The kernel internally rekeys local-storage maps by
                // owner KVA regardless of the user-declared key type
                // (`bpf_local_storage_map_check_btf` enforces the
                // user side to be `i32` — see kernel/bpf/
                // bpf_local_storage.c). `info.btf_key_type_id` thus
                // describes a 4-byte int that doesn't match the
                // 8-byte owner-KVA bytes we hold; rendering through
                // it would produce a misleading 4-byte truncation.
                // Synthesize a [`RenderedValue::Ptr`] directly so
                // operators see "owner_kva: 0xff11000009_85df00" in
                // the rendered output instead of the raw hex
                // surfaced solely via `key_hex`. `deref` is `None`:
                // the renderer's Ptr-deref path only chases cpumask
                // kptrs and arena addresses; the owner objects
                // (task_struct, inode, sock, cgroup) are slab-
                // allocated kernel objects with no static walker
                // here.
                let owner_kva = u64::from_le_bytes(k.as_slice().try_into().unwrap_or([0u8; 8]));
                let key = Some(RenderedValue::Ptr {
                    value: owner_kva,
                    deref: None,
                    deref_skipped_reason: None,
                    cast_annotation: None,
                });
                // Value side renders through BTF when available and
                // falls back to hex when the value type id isn't
                // resolvable — `render_key_optional` would otherwise
                // drop the rendered side entirely (returning `None`)
                // when BTF is missing, leaving operators with only
                // `value_hex`. The hex bytes ARE still kept in
                // `value_hex` regardless, so the fallback adds the
                // structured render WITHOUT removing the raw bytes.
                let value = Some(render_value_or_hex(
                    btf,
                    info.btf_value_type_id,
                    &v,
                    &mem_reader,
                ));
                let payload = chase_sdt_data_payload(
                    btf,
                    sdt_data_field,
                    sdt_alloc_meta.as_ref(),
                    &v,
                    &mem_reader,
                );
                out.entries.push(FailureDumpEntry {
                    key,
                    key_hex: hex_dump(&k),
                    value,
                    value_hex: hex_dump(&v),
                    payload,
                });
            }
            if truncated {
                out.error = Some(format!(
                    "local_storage map truncated at {MAX_HASH_ENTRIES} entries"
                ));
            }
        }
        BPF_MAP_TYPE_RINGBUF | BPF_MAP_TYPE_USER_RINGBUF => {
            // Records themselves are transient; the dump path can still
            // surface the producer/consumer/pending positions and
            // capacity from `struct bpf_ringbuf` so the operator sees
            // whether the consumer is keeping up. See
            // [`render_ringbuf_state`] for the read path and
            // [`FailureDumpRingbuf`] for the rendered shape.
            match render_ringbuf_state(accessor, info) {
                Ok(rb) => {
                    out.ringbuf = Some(rb);
                }
                Err(reason) => {
                    out.error = Some(reason);
                }
            }
        }
        BPF_MAP_TYPE_STACK_TRACE => {
            // STACK_TRACE keys 0..n_buckets where n_buckets =
            // roundup_pow_of_two(max_entries). Each non-null
            // `bpf_stack_map.buckets[id]` points to a
            // `stack_map_bucket` carrying `nr` u64 PCs (or
            // `bpf_stack_build_id` records when BPF_F_STACK_BUILD_ID
            // is set). See [`render_stack_traces`].
            match render_stack_traces(accessor, info) {
                Ok(st) => {
                    out.stack_trace = Some(st);
                }
                Err(reason) => {
                    out.error = Some(reason);
                }
            }
        }
        // FD-array families: read each `bpf_array.ptrs[]` slot as a
        // `void *`. Non-zero = populated. The slot's contents are
        // kernel pointers to subsystem-specific objects (bpf_prog,
        // file, bpf_map, net_device, sock, cgroup, etc.) — the dump
        // path doesn't dereference them; just reports populated
        // indices so the operator sees which slots have entries.
        BPF_MAP_TYPE_PROG_ARRAY
        | BPF_MAP_TYPE_PERF_EVENT_ARRAY
        | BPF_MAP_TYPE_CGROUP_ARRAY
        | BPF_MAP_TYPE_ARRAY_OF_MAPS
        | BPF_MAP_TYPE_HASH_OF_MAPS
        | BPF_MAP_TYPE_DEVMAP
        | BPF_MAP_TYPE_DEVMAP_HASH
        | BPF_MAP_TYPE_SOCKMAP
        | BPF_MAP_TYPE_SOCKHASH
        | BPF_MAP_TYPE_CPUMAP
        | BPF_MAP_TYPE_XSKMAP
        | BPF_MAP_TYPE_REUSEPORT_SOCKARRAY => {
            out.fd_array = Some(render_fd_array_slots(accessor, info));
            if fd_map_is_hash_shaped(info.map_type) {
                out.error = Some(
                    "fd-array dump does not walk hash-shaped FD maps \
                     (SOCKHASH/DEVMAP_HASH/HASH_OF_MAPS); populated \
                     count is unavailable for these"
                        .into(),
                );
            }
        }
        // Every other map type (transient containers,
        // sub-structures the dump path doesn't yet decode) maps to
        // a static explanation string in [`MAP_TYPE_EXPLANATIONS`].
        // The wildcard arm catches kernels newer than the dump
        // renderer (a freshly-added uapi map type). Keep the table
        // and dispatch lockstep when adding a new map type that
        // needs a real walker — convert the table entry into an
        // explicit arm.
        other => {
            out.error = Some(
                MAP_TYPE_EXPLANATIONS
                    .iter()
                    .find(|(t, _)| *t == other)
                    .map(|(_, msg)| (*msg).to_string())
                    .unwrap_or_else(|| {
                        format!(
                            "unknown map_type {other} (kernel newer than dump renderer; \
                             update render_map dispatch)"
                        )
                    }),
            );
        }
    }

    out
}

/// Read `struct bpf_ringbuf_map.rb -> struct bpf_ringbuf` and surface
/// the consumer/producer/pending positions plus capacity.
///
/// Read path:
///   1. The `struct bpf_map` lives at `info.map_kva` (start of
///      `bpf_ringbuf_map` since `bpf_map` is its first field).
///   2. `rb` is at `info.map_kva + offsets.rbm_rb`; read as a u64.
///   3. `*rb` is heap-allocated (vmap'd) — translate via
///      `translate_any_kva`, then read mask/consumer_pos/producer_pos
///      /pending_pos at the resolved page.
///
/// Returns `Err(reason)` when:
///   - `BpfRingbufOffsets` were not resolvable from BTF (kernel
///     without ringbuf support, or BTF stripped of `bpf_ringbuf` /
///     `bpf_ringbuf_map`).
///   - The `bpf_ringbuf *` pointer is NULL or unmapped.
///   - `info.map_kva` itself doesn't translate.
pub(super) fn render_ringbuf_state(
    accessor: &GuestMemMapAccessor<'_>,
    info: &BpfMapInfo,
) -> Result<FailureDumpRingbuf, String> {
    let Some(rb_offs) = accessor.offsets().ringbuf_offsets.as_ref() else {
        return Err(
            "RINGBUF state unreadable: BTF lacks bpf_ringbuf_map / bpf_ringbuf \
             (kernel built without ringbuf, or BTF stripped). Wire format \
             remains consumer_pos/producer_pos/mask/pending_pos but the \
             struct field offsets are not resolved on this kernel."
                .to_string(),
        );
    };

    let kernel = accessor.kernel();
    let mem = kernel.mem();
    let walk = kernel.walk_context();

    // Step 1+2: read `bpf_ringbuf_map.rb` as u64 (kernel pointer).
    let map_pa = super::super::idr::translate_any_kva(
        mem,
        walk.cr3_pa,
        walk.page_offset,
        info.map_kva,
        walk.l5,
        walk.tcr_el1,
    )
    .ok_or_else(|| {
        "RINGBUF map_kva unmapped during freeze (concurrent destruction race?)".to_string()
    })?;
    let rb_kva = mem.read_u64(map_pa, rb_offs.rbm_rb);
    if rb_kva == 0 {
        return Err(
            "RINGBUF rb pointer NULL: bpf_ringbuf_alloc failed at map create time \
             (out of memory or numa mismatch); the map exists but has no backing \
             ring data area"
                .to_string(),
        );
    }

    // Step 3: dereference rb_kva to read four position fields.
    // bpf_ringbuf is vmap'd (kernel/bpf/ringbuf.c::bpf_ringbuf_area_alloc),
    // so translate_any_kva is required (page-table walk if SLAB-direct
    // fails). The mask field comes first; consumer/producer/pending are
    // page-aligned so they may straddle separate translates — the helper
    // does one translate + one scalar read per field.
    let read_at = |off: usize| -> Option<u64> {
        let kva = rb_kva.wrapping_add(off as u64);
        let pa = super::super::idr::translate_any_kva(
            mem,
            walk.cr3_pa,
            walk.page_offset,
            kva,
            walk.l5,
            walk.tcr_el1,
        )?;
        Some(mem.read_u64(pa, 0))
    };

    let mask = read_at(rb_offs.rb_mask).ok_or_else(|| {
        "RINGBUF rb->mask unmapped during freeze (rb pointer torn or unmapped)".to_string()
    })?;
    let consumer_pos = read_at(rb_offs.rb_consumer_pos)
        .ok_or_else(|| "RINGBUF rb->consumer_pos unmapped (consumer page absent)".to_string())?;
    let producer_pos = read_at(rb_offs.rb_producer_pos)
        .ok_or_else(|| "RINGBUF rb->producer_pos unmapped (producer page absent)".to_string())?;
    let pending_pos = read_at(rb_offs.rb_pending_pos)
        .ok_or_else(|| "RINGBUF rb->pending_pos unmapped (producer page absent)".to_string())?;

    // Capacity = mask + 1 (data_sz from bpf_ringbuf_alloc; always
    // power of two). Reject `mask == u64::MAX` (capacity would
    // wrap to 0) — this signals a corrupted read of `rb->mask`,
    // not a legitimate ring (kernel allocates at most 1<<31
    // bytes per `bpf_ringbuf_alloc`'s validation, leaving the
    // top bits zero).
    if mask == u64::MAX {
        return Err("RINGBUF rb->mask = u64::MAX (capacity would wrap to 0); \
             likely a corrupted read of bpf_ringbuf.mask"
            .to_string());
    }
    let capacity = mask.wrapping_add(1);
    if capacity == 0 {
        return Err("RINGBUF capacity = 0 (mask + 1 wrapped); rb->mask read \
             produced a non-power-of-two value"
            .to_string());
    }
    // Pending bytes uses unsigned wraparound — both counters are
    // monotonically advancing 64-bit values so the subtraction is
    // well-defined for any consumer/producer ordering.
    Ok(FailureDumpRingbuf {
        capacity,
        consumer_pos,
        producer_pos,
        pending_pos,
        pending_bytes: producer_pos.wrapping_sub(consumer_pos),
    })
}

/// Walk a `BPF_MAP_TYPE_STACK_TRACE` map's `bpf_stack_map.buckets[]`
/// flex array and surface every populated bucket.
///
/// Read path:
///   1. The map is `bpf_stack_map` (kernel/bpf/stackmap.c) embedding
///      `struct bpf_map` at offset 0; `info.map_kva` points there.
///   2. Read `n_buckets` (u32) at the BTF-resolved offset.
///   3. For each bucket id 0..min(n_buckets, MAX_STACK_TRACE_BUCKETS),
///      read `buckets[id]` (a `struct stack_map_bucket *`) at the
///      flex-array offset + id*8.
///   4. For each non-null pointer, dereference and read `nr` (u32),
///      then up to MAX_STACK_TRACE_PCS u64 PCs from `data[]`.
///
/// Returns `Err(reason)` when BTF offsets weren't resolvable; an empty
/// `entries` vec when the map is allocated but no traces are stored
/// (live but unused).
pub(super) fn render_stack_traces(
    accessor: &GuestMemMapAccessor<'_>,
    info: &BpfMapInfo,
) -> Result<FailureDumpStackTrace, String> {
    let Some(sm_offs) = accessor.offsets().stackmap_offsets.as_ref() else {
        return Err(
            "STACK_TRACE bucket array unreadable: BTF lacks bpf_stack_map / \
             stack_map_bucket. Each non-null bucket pointer would carry `nr` \
             u64 PCs (or bpf_stack_build_id records when BPF_F_STACK_BUILD_ID \
             is set on the map); resolve the offsets to render."
                .to_string(),
        );
    };

    let kernel = accessor.kernel();
    let mem = kernel.mem();
    let walk = kernel.walk_context();

    let map_pa = super::super::idr::translate_any_kva(
        mem,
        walk.cr3_pa,
        walk.page_offset,
        info.map_kva,
        walk.l5,
        walk.tcr_el1,
    )
    .ok_or_else(|| "STACK_TRACE map_kva unmapped during freeze".to_string())?;

    let n_buckets = mem.read_u32(map_pa, sm_offs.smap_n_buckets);
    let scan_buckets = n_buckets.min(MAX_STACK_TRACE_BUCKETS);

    // Per-entry size matches `stack_map_data_size` in
    // kernel/bpf/stackmap.c: `sizeof(struct bpf_stack_build_id)` (32:
    // s32 status + uchar build_id[20] + union { u64 offset; u64 ip; })
    // when BPF_F_STACK_BUILD_ID is set, else `sizeof(u64)` (8). The
    // flag lives on the kernel-side `bpf_map.map_flags`, captured into
    // `BpfMapInfo::map_flags` at discovery. The kernel asserts
    // `BUILD_BUG_ON(sizeof(bpf_stack_build_id) % sizeof(u64))` so the
    // entry size is always a u64-multiple.
    const BPF_F_STACK_BUILD_ID: u32 = 1 << 5;
    const STACK_BUILD_ID_SIZE: u32 = 32;
    let build_id_mode = (info.map_flags & BPF_F_STACK_BUILD_ID) != 0;
    let entry_size: u32 = if build_id_mode {
        STACK_BUILD_ID_SIZE
    } else {
        8
    };

    let mut entries = Vec::new();
    let mut any_truncated = false;
    let mut buckets_unreadable = 0u32;

    for bucket_id in 0..scan_buckets {
        // buckets[id] is a u64 pointer at smap_buckets + id*8.
        let slot_kva = info
            .map_kva
            .wrapping_add(sm_offs.smap_buckets as u64)
            .wrapping_add((bucket_id as u64) * 8);
        let Some(slot_pa) = super::super::idr::translate_any_kva(
            mem,
            walk.cr3_pa,
            walk.page_offset,
            slot_kva,
            walk.l5,
            walk.tcr_el1,
        ) else {
            // Bucket array spans across pages on large maps; an
            // unmapped page in the array itself is exotic but
            // possible. Skip rather than abort, but count it so the
            // gap surfaces instead of reading as a smaller map.
            buckets_unreadable += 1;
            continue;
        };
        let bucket_kva = mem.read_u64(slot_pa, 0);
        if bucket_kva == 0 {
            continue;
        }

        // Dereference the bucket: read nr (u32) and the data[] head.
        let Some(bucket_pa) = super::super::idr::translate_any_kva(
            mem,
            walk.cr3_pa,
            walk.page_offset,
            bucket_kva,
            walk.l5,
            walk.tcr_el1,
        ) else {
            // Non-null bucket pointer but its struct page is
            // unmapped: the bucket exists yet its contents are
            // unreadable. Count rather than silently drop.
            buckets_unreadable += 1;
            continue;
        };
        let nr = mem.read_u32(bucket_pa, sm_offs.smb_nr);

        // Per-bucket PC cap: bound the read cost regardless of how
        // deep the trace is. Surface the truncation.
        let nr_to_read = nr.min(MAX_STACK_TRACE_PCS);
        if nr > MAX_STACK_TRACE_PCS {
            any_truncated = true;
        }

        let data_byte_len = (nr_to_read as usize).saturating_mul(entry_size as usize);

        // Read raw data bytes for the hex dump path. data[] is at
        // the flex-array offset within stack_map_bucket. The bucket
        // area is allocated by `bpf_map_area_alloc` (kernel/bpf/
        // syscall.c) which kmalloc's small allocations (direct-mapped)
        // and vmalloc's larger ones (vmap'd). Translate per-page via
        // `translate_any_kva` so both backings work and so vmap'd
        // pages that aren't physically contiguous read correctly.
        let data_kva = bucket_kva.wrapping_add(sm_offs.smb_data as u64);
        let mut data_bytes = vec![0u8; data_byte_len];
        const PAGE: u64 = 4096;
        let mut consumed: u64 = 0;
        let total = data_byte_len as u64;
        let mut data_ok = true;
        while consumed < total {
            let cur_kva = data_kva.wrapping_add(consumed);
            let Some(pa) = super::super::idr::translate_any_kva(
                mem,
                walk.cr3_pa,
                walk.page_offset,
                cur_kva,
                walk.l5,
                walk.tcr_el1,
            ) else {
                data_ok = false;
                break;
            };
            // `page_end` wraps to 0 when `cur_kva` lies on the last
            // page of the 64-bit address space (e.g. an aarch64 TTBR1
            // KVA near `0xFFFF_FFFF_FFFF_F000`). A plain
            // `page_end - cur_kva` then underflows and panics in debug.
            // Modular arithmetic gives the correct chunk size — the
            // subtraction is exact mod 2^64, so `0u64.wrapping_sub(
            // 0xFFFF_FFFF_FFFF_F000) == PAGE` recovers the trailing-
            // page byte count without a special case.
            let page_end = (cur_kva & !(PAGE - 1)).wrapping_add(PAGE);
            let chunk_len = page_end.wrapping_sub(cur_kva).min(total - consumed) as usize;
            let dst = &mut data_bytes[consumed as usize..consumed as usize + chunk_len];
            let n = mem.read_bytes(pa, dst);
            if n != chunk_len {
                data_ok = false;
                break;
            }
            consumed += chunk_len as u64;
        }
        if !data_ok {
            data_bytes.clear();
        }

        let mut pcs = Vec::new();
        if !build_id_mode {
            // Decode u64 PCs. Stop at first short read.
            // Kernel writes via memcpy of `unsigned long` (native byte
            // order) to `stack_map_bucket.data[]`; on the LE-only
            // hosts ktstr supports (x86_64, aarch64) native == LE, so
            // explicit `from_le_bytes` is correct and keeps the wire
            // assumption documented at the read site.
            for chunk in data_bytes.chunks_exact(8) {
                pcs.push(u64::from_le_bytes(chunk.try_into().unwrap()));
            }
        }

        entries.push(FailureDumpStackTraceEntry {
            bucket_id,
            nr,
            pcs,
            data_hex: hex_dump(&data_bytes),
        });
    }

    Ok(FailureDumpStackTrace {
        n_buckets,
        entries,
        truncated: any_truncated || n_buckets > MAX_STACK_TRACE_BUCKETS,
        buckets_unreadable,
    })
}

/// Walk an FD-array's `bpf_array.ptrs[]` flex array and report
/// populated indices.
///
/// True for FD-map types whose slots live in a hash table (SOCKHASH,
/// DEVMAP_HASH, HASH_OF_MAPS) rather than the contiguous
/// `bpf_array.ptrs` flex array. [`render_fd_array_slots`] cannot walk
/// these; [`render_map`] surfaces the gap in `out.error`.
pub(super) fn fd_map_is_hash_shaped(map_type: u32) -> bool {
    matches!(
        map_type,
        BPF_MAP_TYPE_SOCKHASH | BPF_MAP_TYPE_DEVMAP_HASH | BPF_MAP_TYPE_HASH_OF_MAPS
    )
}

/// FD-array families (PROG_ARRAY, PERF_EVENT_ARRAY, CGROUP_ARRAY,
/// ARRAY_OF_MAPS, DEVMAP, SOCKMAP, CPUMAP, XSKMAP,
/// REUSEPORT_SOCKARRAY) all share the `bpf_array.ptrs` layout: each
/// slot is `sizeof(void *)` (8 bytes on 64-bit). Non-zero = populated.
/// The dump reports populated count + indices (truncated to
/// [`MAX_FD_ARRAY_INDICES`]).
///
/// HASH_OF_MAPS and SOCKHASH/DEVMAP_HASH are hash-shaped, not array-
/// shaped: their slots do not live in the contiguous `bpf_array.ptrs`
/// flex array this walker reads, so scanning it would report a false
/// `populated: 0` indistinguishable from a genuinely empty array map.
/// [`fd_map_is_hash_shaped`] detects them, the walker returns
/// immediately with `scanned: 0`, and [`render_map`] sets `out.error`
/// so operators see the dump path does not walk these. Accurate
/// support would walk the hash table (`iter_hash_map`) and resolve
/// each element's fd pointer.
pub(super) fn render_fd_array_slots(
    accessor: &GuestMemMapAccessor<'_>,
    info: &BpfMapInfo,
) -> FailureDumpFdArray {
    let kernel = accessor.kernel();
    let mem = kernel.mem();
    let walk = kernel.walk_context();
    let array_value_off = accessor.offsets().array_value;

    let scan = info.max_entries.min(MAX_FD_ARRAY_SLOTS);
    let truncated = info.max_entries > MAX_FD_ARRAY_SLOTS;

    let mut populated: u32 = 0;
    let mut indices: Vec<u32> = Vec::new();

    // For hash-shaped FD maps, bpf_array layout doesn't apply — the
    // ptrs flex array isn't the right slot home. Bail without scanning
    // (scanned: 0) rather than report a false populated: 0; render_map
    // sets out.error so the gap is explicit. See fd_map_is_hash_shaped.
    if fd_map_is_hash_shaped(info.map_type) {
        return FailureDumpFdArray {
            populated: 0,
            scanned: 0,
            indices: Vec::new(),
            truncated: false,
            indices_truncated: false,
            unreadable: 0,
        };
    }

    let mut unreadable: u32 = 0;
    for idx in 0..scan {
        let slot_kva = info
            .map_kva
            .wrapping_add(array_value_off as u64)
            .wrapping_add((idx as u64) * 8);
        let Some(slot_pa) = super::super::idr::translate_any_kva(
            mem,
            walk.cr3_pa,
            walk.page_offset,
            slot_kva,
            walk.l5,
            walk.tcr_el1,
        ) else {
            // Unmapped slot page: cannot read this pointer. Count it
            // rather than silently treating the slot as empty, so
            // `populated` is a visible lower bound.
            unreadable += 1;
            continue;
        };
        let ptr = mem.read_u64(slot_pa, 0);
        if ptr != 0 {
            populated += 1;
            if indices.len() < MAX_FD_ARRAY_INDICES {
                indices.push(idx);
            }
        }
    }

    FailureDumpFdArray {
        populated,
        scanned: scan,
        indices_truncated: indices.len() < populated as usize,
        indices,
        truncated,
        unreadable,
    }
}

#[cfg(test)]
mod render_map_unit_tests {
    //! In-module host-unit tests for `render_map.rs`.
    //!
    //! These live inside `render_map.rs` (rather than the sibling
    //! `dump::tests` module) because three of the functions under
    //! test are private `fn` items — `ascii_str_dump`,
    //! `is_str_literal_section`, and `resolve_struct_ops_payload_type_id`
    //! — which a sibling module cannot reach without changing
    //! production visibility. A child `#[cfg(test)]` module sees the
    //! parent module's private items, so the tests reach them without
    //! promoting any production symbol to `pub(super)`.
    //!
    //! The scene helpers (`build_*_scene`) are re-derived locally as
    //! minimal byte-buffer layouts rather than reaching into the
    //! `dump::tests` module's private fixtures — keeping every edit
    //! inside this file. They mirror the production read formula
    //! `translate_any_kva` reverses (`pa = kva - page_offset`).

    use super::*;
    use crate::monitor::btf_render::MemReader;

    // ---- pure helpers --------------------------------------------

    /// `map_type_name` maps every known discriminant to its lowercase
    /// short name and returns `None` for an unknown discriminant. The
    /// table below independently mirrors all 34 arms of the production
    /// match, so a renamed arm (the string changes) or a dropped arm
    /// (it now falls to `None`) fails this test. These names feed the
    /// operator-facing failure-dump header, where a silent regression
    /// in any single arm would mislabel a captured map.
    #[test]
    fn map_type_name_known_and_unknown() {
        // (discriminant, expected lowercase short name) for every arm.
        let table: &[(u32, &str)] = &[
            (BPF_MAP_TYPE_HASH, "hash"),
            (BPF_MAP_TYPE_ARRAY, "array"),
            (BPF_MAP_TYPE_PROG_ARRAY, "prog_array"),
            (BPF_MAP_TYPE_PERF_EVENT_ARRAY, "perf_event_array"),
            (BPF_MAP_TYPE_PERCPU_HASH, "percpu_hash"),
            (BPF_MAP_TYPE_PERCPU_ARRAY, "percpu_array"),
            (BPF_MAP_TYPE_STACK_TRACE, "stack_trace"),
            (BPF_MAP_TYPE_CGROUP_ARRAY, "cgroup_array"),
            (BPF_MAP_TYPE_LRU_HASH, "lru_hash"),
            (BPF_MAP_TYPE_LRU_PERCPU_HASH, "lru_percpu_hash"),
            (BPF_MAP_TYPE_LPM_TRIE, "lpm_trie"),
            (BPF_MAP_TYPE_ARRAY_OF_MAPS, "array_of_maps"),
            (BPF_MAP_TYPE_HASH_OF_MAPS, "hash_of_maps"),
            (BPF_MAP_TYPE_DEVMAP, "devmap"),
            (BPF_MAP_TYPE_SOCKMAP, "sockmap"),
            (BPF_MAP_TYPE_CPUMAP, "cpumap"),
            (BPF_MAP_TYPE_XSKMAP, "xskmap"),
            (BPF_MAP_TYPE_SOCKHASH, "sockhash"),
            (BPF_MAP_TYPE_CGROUP_STORAGE, "cgroup_storage"),
            (BPF_MAP_TYPE_REUSEPORT_SOCKARRAY, "reuseport_sockarray"),
            (BPF_MAP_TYPE_PERCPU_CGROUP_STORAGE, "percpu_cgroup_storage"),
            (BPF_MAP_TYPE_QUEUE, "queue"),
            (BPF_MAP_TYPE_STACK, "stack"),
            (BPF_MAP_TYPE_SK_STORAGE, "sk_storage"),
            (BPF_MAP_TYPE_DEVMAP_HASH, "devmap_hash"),
            (BPF_MAP_TYPE_STRUCT_OPS, "struct_ops"),
            (BPF_MAP_TYPE_RINGBUF, "ringbuf"),
            (BPF_MAP_TYPE_INODE_STORAGE, "inode_storage"),
            (BPF_MAP_TYPE_TASK_STORAGE, "task_storage"),
            (BPF_MAP_TYPE_BLOOM_FILTER, "bloom_filter"),
            (BPF_MAP_TYPE_USER_RINGBUF, "user_ringbuf"),
            (BPF_MAP_TYPE_CGRP_STORAGE, "cgrp_storage"),
            (BPF_MAP_TYPE_ARENA, "arena"),
            (BPF_MAP_TYPE_INSN_ARRAY, "insn_array"),
        ];
        assert_eq!(table.len(), 34, "table must mirror all 34 known arms");
        for &(disc, name) in table {
            assert_eq!(
                map_type_name(disc),
                Some(name),
                "map_type_name({disc}) must be Some({name:?})",
            );
        }
        // Unknown discriminant well past every uapi value.
        assert_eq!(map_type_name(0xDEAD_BEEF), None);
    }

    /// `ascii_str_dump` passes printable bytes (0x20..=0x7E) through and
    /// escapes every other byte as lowercase two-digit `\xHH`. DEL
    /// (0x7F) and NUL (0x00) are the boundary cases.
    #[test]
    fn ascii_str_dump_printable_passthrough_and_escape() {
        assert_eq!(ascii_str_dump(b"abc"), "abc");
        assert_eq!(ascii_str_dump(&[0x00]), "\\x00");
        assert_eq!(ascii_str_dump(&[0x41, 0x00, 0x42]), "A\\x00B");
        // DEL is just above the printable range → escaped.
        assert_eq!(ascii_str_dump(&[0x7f]), "\\x7f");
        // 0x20 (space) is the low printable bound; 0x7E (~) the high.
        assert_eq!(ascii_str_dump(&[0x20, 0x7e]), " ~");
        // 0x1f (just below space) and 0xff escape with lowercase hex.
        assert_eq!(ascii_str_dump(&[0x1f, 0xff]), "\\x1f\\xff");
    }

    /// `is_str_literal_section` is a `.rodata.str1.1` suffix check —
    /// any obj-name prefix matches, a bare `.rodata` does not, and a
    /// trailing extra char defeats the suffix.
    #[test]
    fn is_str_literal_section_suffix_match() {
        assert!(is_str_literal_section("scx_foo.rodata.str1.1"));
        assert!(is_str_literal_section(".rodata.str1.1"));
        assert!(!is_str_literal_section("scx_foo.rodata"));
        assert!(!is_str_literal_section("scx_foo.bss"));
        assert!(!is_str_literal_section("rodata.str1.1x"));
    }

    // ---- select_sdt_alloc_meta -----------------------------------

    fn mk_meta(name: &str, target_type_id: u32) -> SdtAllocMeta {
        SdtAllocMeta {
            allocator_name: name.into(),
            elem_size: 32,
            header_size: 8,
            target_type_id,
            kern_vm_start: 0xFFFF_8000_0000_0000,
        }
    }

    /// Empty metadata yields `None`; a single allocator is returned
    /// regardless of whether its stem matches the map name (single-
    /// allocator schedulers always degrade to the unique candidate).
    #[test]
    fn select_sdt_alloc_meta_empty_and_single() {
        assert!(select_sdt_alloc_meta(&[], "anything").is_none());
        let single = [mk_meta("scx_task_allocator", 7)];
        // Single meta: returned even though "task" is not a substring
        // of the map name — the len()==1 branch precedes the filter.
        let sel = select_sdt_alloc_meta(&single, "unrelated_map")
            .expect("single allocator must always be selected");
        assert_eq!(sel.allocator_name, "scx_task_allocator");
        assert_eq!(sel.target_type_id, 7);
    }

    /// With two allocators, the one whose normalized stem
    /// (`scx_<stem>_allocator` → `<stem>`) is a substring of the map
    /// name wins; a map name matching neither stem yields `None`.
    #[test]
    fn select_sdt_alloc_meta_multi_match_and_no_match() {
        let metas = [
            mk_meta("scx_task_allocator", 7),
            mk_meta("scx_cgrp_allocator", 11),
        ];
        // "task" ⊂ "scx_task_map"; "cgrp" ⊄ "scx_task_map".
        let sel = select_sdt_alloc_meta(&metas, "scx_task_map")
            .expect("the task allocator stem must match scx_task_map");
        assert_eq!(sel.allocator_name, "scx_task_allocator");
        assert_eq!(sel.target_type_id, 7);
        // Neither "task" nor "cgrp" appears in "unrelated_map".
        assert!(
            select_sdt_alloc_meta(&metas, "unrelated_map").is_none(),
            "no stem matches → None (degrade to payload: None, not a guess)",
        );
    }

    /// Tie-break: when two stems both appear in the map name, the
    /// longest stem wins (`max_by_key` on stem length). A short
    /// substring stem must not shadow a more-specific longer one.
    #[test]
    fn select_sdt_alloc_meta_longest_stem_wins() {
        let metas = [
            // stem "x" (len 1) — substring of "scx_xlong_map".
            mk_meta("scx_x_allocator", 1),
            // stem "xlong" (len 5) — also a substring, longer.
            mk_meta("scx_xlong_allocator", 99),
        ];
        let sel = select_sdt_alloc_meta(&metas, "scx_xlong_map")
            .expect("both stems substring the name; longest must be chosen");
        assert_eq!(
            sel.target_type_id, 99,
            "longer stem 'xlong' must win over shorter 'x'",
        );
        assert_eq!(sel.allocator_name, "scx_xlong_allocator");
    }

    // ---- build_arena_page_index ----------------------------------

    /// `None` snapshot yields an empty index; distinct `user_addr`
    /// pages map to their enumeration index; a duplicate `user_addr`
    /// keeps the FIRST page (lower index) and discards the second.
    #[test]
    fn build_arena_page_index_distinct_and_duplicate() {
        use super::super::super::arena::{ArenaPage, ArenaSnapshot};

        assert!(build_arena_page_index(None).is_empty());

        let distinct = ArenaSnapshot {
            pages: vec![
                ArenaPage {
                    user_addr: 0x1000,
                    bytes: vec![0u8; 16],
                },
                ArenaPage {
                    user_addr: 0x2000,
                    bytes: vec![0u8; 16],
                },
            ],
            ..ArenaSnapshot::default()
        };
        let idx = build_arena_page_index(Some(&distinct));
        assert_eq!(idx.len(), 2);
        assert_eq!(idx.get(&0x1000), Some(&0usize));
        assert_eq!(idx.get(&0x2000), Some(&1usize));

        // Two pages sharing one user_addr: the Occupied arm keeps the
        // first (index 0) and logs, leaving exactly one entry.
        let dup = ArenaSnapshot {
            pages: vec![
                ArenaPage {
                    user_addr: 0x3000,
                    bytes: vec![1u8; 16],
                },
                ArenaPage {
                    user_addr: 0x3000,
                    bytes: vec![2u8; 16],
                },
            ],
            ..ArenaSnapshot::default()
        };
        let idx = build_arena_page_index(Some(&dup));
        assert_eq!(idx.len(), 1);
        assert_eq!(
            idx.get(&0x3000),
            Some(&0usize),
            "duplicate user_addr must keep the FIRST page's index",
        );
    }

    // ---- resolve_arena_type_in_index cross-BTF + empty -----------

    /// Cross-BTF id-space gate: a slot whose `source_btf_kva` is
    /// non-zero resolves ONLY when the requesting BTF kva matches.
    /// A mismatching kva suppresses the hit; a zero requesting kva
    /// (no per-map BTF context) is suppressed conservatively. The
    /// matching-kva case proves the gate is selective, not a
    /// blanket-None.
    #[test]
    fn resolve_arena_type_in_index_cross_btf_gate() {
        use super::super::super::arena::ArenaSnapshot;
        use super::super::super::btf_render::ArenaResolveHit;

        let snap = ArenaSnapshot {
            user_vm_start: 0x10_0000_0000,
            ..ArenaSnapshot::default()
        };
        let mut index = ArenaSlotIndex::new();
        index.insert(
            0x0000_1000,
            ArenaSlotInfo {
                elem_size: 24,
                header_size: 8,
                target_type_id: 7,
                source_btf_kva: 0xAAAA,
            },
        );

        // Mismatch: slot's BTF kva 0xAAAA != requesting 0xBBBB.
        assert!(
            resolve_arena_type_in_index(Some(&snap), Some(&index), 0x10_0000_1000, 0xBBBB)
                .is_none(),
            "cross-BTF mismatch must suppress the hit",
        );
        // Zero requesting kva with non-zero slot kva: conservative
        // suppress.
        assert!(
            resolve_arena_type_in_index(Some(&snap), Some(&index), 0x10_0000_1000, 0).is_none(),
            "zero requesting_btf_kva must suppress a BTF-scoped slot",
        );
        // Match: same slot DOES resolve when the requesting kva equals
        // the slot's source_btf_kva — proving the gate is selective.
        assert_eq!(
            resolve_arena_type_in_index(Some(&snap), Some(&index), 0x10_0000_1000, 0xAAAA),
            Some(ArenaResolveHit {
                target_type_id: 7,
                header_skip: 8,
            }),
            "matching requesting_btf_kva must resolve the slot",
        );
    }

    /// Empty index with an in-window address hits the
    /// `range(..=key).next_back() == None` branch and returns `None`.
    /// Distinct from the `arena_slot_index = None` short-circuit (the
    /// index is present but carries no slots).
    #[test]
    fn resolve_arena_type_in_index_empty_index_in_window() {
        use super::super::super::arena::ArenaSnapshot;

        let snap = ArenaSnapshot {
            user_vm_start: 0x10_0000_0000,
            ..ArenaSnapshot::default()
        };
        let empty = ArenaSlotIndex::new();
        assert!(
            resolve_arena_type_in_index(Some(&snap), Some(&empty), 0x10_0000_1008, 0).is_none(),
            "empty index with in-window addr must reach the no-slot branch and return None",
        );
    }

    // ---- append_arena_slot_index_for_allocator oversized header --

    /// A `header_size` above `u32::MAX` fails the `u32::try_from`
    /// guard and short-circuits before any insert — leaving the index
    /// untouched. The baseline call with `header_size = 8` DOES
    /// insert, proving the guard rather than vacuous emptiness.
    #[test]
    fn append_arena_slot_index_oversized_header_skips() {
        // Baseline: a fitting header inserts exactly one entry.
        let mut baseline = ArenaSlotIndex::new();
        append_arena_slot_index_for_allocator(&mut baseline, "a", 7, 8, 16, &[0x1000u64], 0);
        assert_eq!(baseline.len(), 1, "fitting header_size must insert");

        // Oversized header_size (> u32::MAX): try_from fails, bail.
        let mut index = ArenaSlotIndex::new();
        append_arena_slot_index_for_allocator(
            &mut index,
            "a",
            7,
            (u32::MAX as usize) + 1,
            16,
            &[0x1000u64],
            0,
        );
        assert!(
            index.is_empty(),
            "header_size > u32::MAX must skip every entry; got {} entries",
            index.len(),
        );
    }

    // ---- synthetic BTF byte builder ------------------------------
    //
    // Mirrors the wire-format builder in `dump::tests` (the
    // `find_sdt_data_field_offset_returns_offset_for_fwd_pointee`
    // template) so the BTF-driven tests below don't depend on a host
    // vmlinux. Kept local so every edit stays inside render_map.rs.

    /// Push a NUL-terminated string into the BTF string section,
    /// returning its offset.
    fn push_str(strings: &mut Vec<u8>, name: &str) -> u32 {
        let off = strings.len() as u32;
        strings.extend_from_slice(name.as_bytes());
        strings.push(0);
        off
    }

    /// Assemble a BTF blob from a populated type section + string
    /// section (kernel uapi `struct btf_header`, magic 0xEB9F).
    fn assemble_btf(types: &[u8], strings: &[u8]) -> btf_rs::Btf {
        use std::io::Write;
        let type_len = types.len() as u32;
        let str_len = strings.len() as u32;
        let mut blob: Vec<u8> = Vec::new();
        blob.write_all(&0xEB9F_u16.to_le_bytes()).unwrap(); // magic
        blob.push(1); // version
        blob.push(0); // flags
        blob.write_all(&24u32.to_le_bytes()).unwrap(); // hdr_len
        blob.write_all(&0u32.to_le_bytes()).unwrap(); // type_off
        blob.write_all(&type_len.to_le_bytes()).unwrap();
        blob.write_all(&type_len.to_le_bytes()).unwrap(); // str_off
        blob.write_all(&str_len.to_le_bytes()).unwrap();
        blob.extend_from_slice(types);
        blob.extend_from_slice(strings);
        btf_rs::Btf::from_bytes(&blob).expect("synthetic BTF parses")
    }

    const BTF_KIND_INT: u32 = 1;
    const BTF_KIND_PTR: u32 = 2;
    const BTF_KIND_STRUCT: u32 = 4;
    const BTF_KIND_FWD: u32 = 7;

    /// Emit a `BTF_KIND_INT` type entry (name_off, size bytes,
    /// `int_data` = `(encoding<<24)|(offset<<16)|bits`). For unsigned
    /// integers `encoding == 0`, so `int_data == bits`.
    fn emit_int(types: &mut Vec<u8>, name_off: u32, size: u32, bits: u32) {
        types.extend_from_slice(&name_off.to_le_bytes());
        let info = (BTF_KIND_INT << 24) & 0x1f00_0000;
        types.extend_from_slice(&info.to_le_bytes());
        types.extend_from_slice(&size.to_le_bytes());
        types.extend_from_slice(&bits.to_le_bytes());
    }

    // ---- resolve_struct_ops_payload_type_id ----------------------

    /// Build a BTF whose wrapper struct `bpf_struct_ops_<name>` carries
    /// a `data` member pointing at the user-ops struct, and assert the
    /// resolve returns the user-ops struct id (the `data` member's
    /// type), not the wrapper id. Also covers the wrapper_id==0, the
    /// non-struct wrapper, and the missing-`data`-member None paths.
    #[test]
    fn resolve_struct_ops_payload_type_id_resolves_data_member() {
        // BTF layout:
        //   id 1: Int u64 (the `common` member's type — a stand-in)
        //   id 2: Struct user_ops { u64 a @ 0 } size 8  (the payload)
        //   id 3: Struct bpf_struct_ops_x { u64 common @ 0;
        //                                    user_ops data @ 8 } size 16
        let mut strings: Vec<u8> = vec![0];
        let n_u64 = push_str(&mut strings, "u64");
        let n_user_ops = push_str(&mut strings, "user_ops");
        let n_a = push_str(&mut strings, "a");
        let n_wrapper = push_str(&mut strings, "bpf_struct_ops_x");
        let n_common = push_str(&mut strings, "common");
        let n_data = push_str(&mut strings, "data");

        let mut types: Vec<u8> = Vec::new();
        // id 1: Int u64.
        emit_int(&mut types, n_u64, 8, 64);
        // id 2: Struct user_ops { u64 a @ 0 } size 8, vlen 1.
        types.extend_from_slice(&n_user_ops.to_le_bytes());
        let s2_info = ((BTF_KIND_STRUCT << 24) & 0x1f00_0000) | 1u32;
        types.extend_from_slice(&s2_info.to_le_bytes());
        types.extend_from_slice(&8u32.to_le_bytes()); // size
        types.extend_from_slice(&n_a.to_le_bytes());
        types.extend_from_slice(&1u32.to_le_bytes()); // member type id 1
        types.extend_from_slice(&0u32.to_le_bytes()); // bit offset 0
        // id 3: Struct bpf_struct_ops_x { u64 common @ 0;
        //        user_ops data @ 8 } size 16, vlen 2.
        types.extend_from_slice(&n_wrapper.to_le_bytes());
        let s3_info = ((BTF_KIND_STRUCT << 24) & 0x1f00_0000) | 2u32;
        types.extend_from_slice(&s3_info.to_le_bytes());
        types.extend_from_slice(&16u32.to_le_bytes()); // size
        // member 0: common @ bit 0, type id 1 (u64).
        types.extend_from_slice(&n_common.to_le_bytes());
        types.extend_from_slice(&1u32.to_le_bytes());
        types.extend_from_slice(&0u32.to_le_bytes());
        // member 1: data @ bit 64 (byte 8), type id 2 (user_ops).
        types.extend_from_slice(&n_data.to_le_bytes());
        types.extend_from_slice(&2u32.to_le_bytes());
        types.extend_from_slice(&64u32.to_le_bytes());

        let btf = assemble_btf(&types, &strings);
        let wrapper_id: u32 = 3;
        assert_eq!(
            resolve_struct_ops_payload_type_id(&btf, wrapper_id),
            Some(2),
            "resolve must return the `data` member's type id (user_ops=2), not the wrapper id",
        );
        // wrapper_type_id == 0 short-circuits.
        assert_eq!(resolve_struct_ops_payload_type_id(&btf, 0), None);
        // Non-struct wrapper id (point at the Int id 1): None.
        assert_eq!(resolve_struct_ops_payload_type_id(&btf, 1), None);
    }

    /// A wrapper struct with NO `data` member resolves to `None` — the
    /// loop walks every member, finds no `"data"`, and falls through.
    #[test]
    fn resolve_struct_ops_payload_type_id_no_data_member() {
        // id 1: Int u64; id 2: Struct wrapper { u64 common @ 0 } size 8.
        let mut strings: Vec<u8> = vec![0];
        let n_u64 = push_str(&mut strings, "u64");
        let n_wrapper = push_str(&mut strings, "bpf_struct_ops_x");
        let n_common = push_str(&mut strings, "common");

        let mut types: Vec<u8> = Vec::new();
        emit_int(&mut types, n_u64, 8, 64);
        types.extend_from_slice(&n_wrapper.to_le_bytes());
        let s_info = ((BTF_KIND_STRUCT << 24) & 0x1f00_0000) | 1u32;
        types.extend_from_slice(&s_info.to_le_bytes());
        types.extend_from_slice(&8u32.to_le_bytes());
        types.extend_from_slice(&n_common.to_le_bytes());
        types.extend_from_slice(&1u32.to_le_bytes());
        types.extend_from_slice(&0u32.to_le_bytes());

        let btf = assemble_btf(&types, &strings);
        assert_eq!(
            resolve_struct_ops_payload_type_id(&btf, 2),
            None,
            "wrapper without a `data` member must resolve to None",
        );
    }

    // ---- find_sdt_data_field_offset bitfield skip ----------------

    /// A value struct with a non-byte-aligned bitfield member before
    /// the real `struct sdt_data __arena *` member: the bitfield is
    /// skipped via the `!bit_off.is_multiple_of(8)` continue, and the
    /// sdt_data pointer at byte 16 still resolves to `Some(16)`. This
    /// adds the bitfield-skip coverage the existing happy-path test
    /// (member 0 = byte-aligned u64) does not exercise.
    #[test]
    fn find_sdt_data_field_offset_skips_non_byte_aligned_bitfield() {
        // BTF layout:
        //   id 1: Int u64
        //   id 2: Fwd struct sdt_data
        //   id 3: Ptr -> id 2
        //   id 4: Struct value_t (kind_flag=1, bitfield members) {
        //           bitfield bf : 3 @ bit 3 (non-byte-aligned);
        //           sdt_data __arena * data @ bit 128 (byte 16)
        //         } size 24
        let mut strings: Vec<u8> = vec![0];
        let n_u64 = push_str(&mut strings, "u64");
        let n_sdt_data = push_str(&mut strings, "sdt_data");
        let n_value = push_str(&mut strings, "value_t");
        let n_bf = push_str(&mut strings, "bf");
        let n_data = push_str(&mut strings, "data");

        let mut types: Vec<u8> = Vec::new();
        // id 1: Int u64.
        emit_int(&mut types, n_u64, 8, 64);
        // id 2: Fwd struct sdt_data (kind_flag=0 → struct flavour).
        types.extend_from_slice(&n_sdt_data.to_le_bytes());
        let fwd_info = (BTF_KIND_FWD << 24) & 0x1f00_0000;
        types.extend_from_slice(&fwd_info.to_le_bytes());
        types.extend_from_slice(&0u32.to_le_bytes());
        // id 3: Ptr -> id 2.
        types.extend_from_slice(&0u32.to_le_bytes());
        let ptr_info = (BTF_KIND_PTR << 24) & 0x1f00_0000;
        types.extend_from_slice(&ptr_info.to_le_bytes());
        types.extend_from_slice(&2u32.to_le_bytes());
        // id 4: Struct value_t — kind_flag=1 marks bitfield members,
        // so each member's bit_offset encodes (size<<24)|offset.
        types.extend_from_slice(&n_value.to_le_bytes());
        let struct_info =
            ((BTF_KIND_STRUCT << 24) & 0x1f00_0000) | (1u32 << 31) | 2u32; // kind_flag=1, vlen=2
        types.extend_from_slice(&struct_info.to_le_bytes());
        types.extend_from_slice(&24u32.to_le_bytes()); // size
        // member 0: bf — bitfield size 3, bit offset 3 (non-multiple
        // of 8). kind_flag=1 encoding: (bitfield_size << 24) | offset.
        types.extend_from_slice(&n_bf.to_le_bytes());
        types.extend_from_slice(&1u32.to_le_bytes()); // type id 1 (u64)
        types.extend_from_slice(&((3u32 << 24) | 3u32).to_le_bytes());
        // member 1: data — size 0 (not a bitfield), bit offset 128
        // (byte 16). Encoding: (0 << 24) | 128.
        types.extend_from_slice(&n_data.to_le_bytes());
        types.extend_from_slice(&3u32.to_le_bytes()); // type id 3 (Ptr)
        types.extend_from_slice(&128u32.to_le_bytes());

        let btf = assemble_btf(&types, &strings);
        assert_eq!(
            find_sdt_data_field_offset(&btf, 4),
            Some(16),
            "non-byte-aligned bitfield member must be skipped; the sdt_data \
             pointer at byte 16 must still resolve",
        );
    }

    // ---- chase_sdt_data_payload success render -------------------

    /// Happy path: a `MemReader` stub returns canned `elem_size` bytes
    /// (8-byte header + a u32 payload). The chase reads the arena
    /// pointer from `value_bytes`, composes the KVA, slices off the
    /// header, and renders the payload struct. Asserts the rendered
    /// first member equals the post-header bytes AND that `read_kva`
    /// was invoked with `kva == kern_vm_start + (data_ptr & 0xFFFF_FFFF)`.
    #[test]
    fn chase_sdt_data_payload_renders_payload_struct() {
        use std::cell::Cell;

        // Payload BTF: id 1 = Int u32; id 2 = Struct payload_t {
        // u32 val @ 0 } size 4. target_type_id = 2.
        let mut strings: Vec<u8> = vec![0];
        let n_u32 = push_str(&mut strings, "u32");
        let n_payload = push_str(&mut strings, "payload_t");
        let n_val = push_str(&mut strings, "val");
        let mut types: Vec<u8> = Vec::new();
        emit_int(&mut types, n_u32, 4, 32);
        types.extend_from_slice(&n_payload.to_le_bytes());
        let s_info = ((BTF_KIND_STRUCT << 24) & 0x1f00_0000) | 1u32;
        types.extend_from_slice(&s_info.to_le_bytes());
        types.extend_from_slice(&4u32.to_le_bytes()); // size
        types.extend_from_slice(&n_val.to_le_bytes());
        types.extend_from_slice(&1u32.to_le_bytes()); // member type u32
        types.extend_from_slice(&0u32.to_le_bytes()); // bit offset 0
        let btf = assemble_btf(&types, &strings);

        let kern_vm_start: u64 = 0xFFFF_8000_0000_0000;
        let meta = SdtAllocMeta {
            allocator_name: "scx_test_allocator".into(),
            elem_size: 12, // 8-byte header + 4-byte payload
            header_size: 8,
            target_type_id: 2,
            kern_vm_start,
        };
        // value_bytes: arena data pointer at field offset 0.
        let data_ptr: u64 = 0x1_2345_6010;
        let mut value_bytes = vec![0u8; 8];
        value_bytes[0..8].copy_from_slice(&data_ptr.to_le_bytes());

        // Stub: capture the requested kva, return 8 header bytes +
        // a u32 payload of 0xDEADBEEF.
        struct Stub {
            seen_kva: Cell<Option<(u64, usize)>>,
        }
        impl MemReader for Stub {
            fn read_kva(&self, kva: u64, len: usize) -> Option<Vec<u8>> {
                self.seen_kva.set(Some((kva, len)));
                let mut v = vec![0u8; 8]; // header
                v.extend_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
                assert_eq!(v.len(), 12, "stub must return elem_size bytes");
                v.truncate(len);
                Some(v)
            }
        }
        let reader = Stub {
            seen_kva: Cell::new(None),
        };

        let rendered =
            chase_sdt_data_payload(Some(&btf), Some(0), Some(&meta), &value_bytes, &reader)
                .expect("payload must render with all prereqs satisfied");
        match rendered {
            RenderedValue::Struct { members, .. } => {
                let first = members.first().expect("payload struct has one member");
                assert_eq!(first.name, "val");
                assert_eq!(
                    first.value,
                    RenderedValue::Uint {
                        bits: 32,
                        value: 0xDEAD_BEEF,
                    },
                    "the rendered field must equal the post-header payload bytes",
                );
            }
            other => panic!("expected Struct render, got {other:?}"),
        }
        let (kva, len) = reader
            .seen_kva
            .get()
            .expect("read_kva must run on the success path");
        assert_eq!(len, 12, "read length must equal elem_size");
        assert_eq!(
            kva,
            kern_vm_start.wrapping_add(data_ptr & 0xFFFF_FFFF),
            "KVA must compose as kern_vm_start + (data_ptr & 0xFFFF_FFFF)",
        );
    }

    // ---- scene helpers for the GuestMem-backed render tests ------

    const PAGE_OFFSET: u64 = super::super::super::symbols::DEFAULT_PAGE_OFFSET;

    /// Direct-mapping KVA from a host PA — the inverse of the
    /// `pa = kva - page_offset` formula `translate_any_kva` applies.
    fn pa_to_kva(pa: u64) -> u64 {
        PAGE_OFFSET.wrapping_add(pa)
    }

    /// Build a `GuestKernel` + `GuestMemMapAccessor` over `buf`,
    /// invoke `f`, and return its result. `cr3 = 0`, so reads route
    /// through the direct-map (`translate_any_kva`'s `kva - page_offset`
    /// path), never a page-table walk.
    ///
    /// SAFETY: `buf` outlives the borrow; `GuestMem::new` aliases its
    /// backing storage for the duration of the call.
    fn with_accessor<R>(
        buf: &[u8],
        offsets: &super::super::super::btf_offsets::BpfMapOffsets,
        f: impl FnOnce(&GuestMemMapAccessor<'_>) -> R,
    ) -> R {
        let mem = unsafe {
            super::super::super::reader::GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64)
        };
        let kernel = super::super::super::guest::GuestKernel::new_for_test(
            std::sync::Arc::new(mem),
            std::collections::HashMap::new(),
            PAGE_OFFSET,
            0,
            false,
        );
        let accessor = GuestMemMapAccessor::new_for_test(&kernel, offsets, 0);
        f(&accessor)
    }

    fn map_info(name: &str, map_type: u32, map_kva: u64, max_entries: u32) -> BpfMapInfo {
        let (name_bytes, name_len) = crate::monitor::test_util::name_from_str(name);
        BpfMapInfo {
            map_pa: 0,
            map_kva,
            name_bytes,
            name_len,
            map_type,
            map_flags: 0,
            key_size: 0,
            value_size: 0,
            max_entries,
            value_kva: None,
            btf_kva: 0,
            btf_value_type_id: 0,
            btf_vmlinux_value_type_id: 0,
            btf_key_type_id: 0,
        }
    }

    /// Stackmap offsets matching `dump::tests::synth_stackmap_offsets`:
    /// `n_buckets` at 0, `buckets[]` at 16, `nr` at 0, `data[]` at 16.
    fn stackmap_offsets() -> super::super::super::btf_offsets::BpfMapOffsets {
        let mut o = super::super::super::btf_offsets::BpfMapOffsets::EMPTY;
        o.stackmap_offsets = Some(super::super::super::btf_offsets::BpfStackmapOffsets {
            smap_n_buckets: 0,
            smap_buckets: 16,
            smb_nr: 0,
            smb_data: 16,
        });
        o
    }

    // ---- render_stack_traces error/truncation branches -----------

    /// A bucket reporting `nr` greater than `MAX_STACK_TRACE_PCS`
    /// (128): the entry preserves the raw `nr` (200) but caps
    /// `pcs.len()` at 128, and the result's `truncated` flag is set.
    #[test]
    fn render_stack_traces_per_bucket_pc_truncation() {
        // One bucket at PA 0x1000 (map struct), bucket struct at
        // 0x1_0000. nr = 200 PCs; the read loop bounds at 128.
        let map_pa: u64 = 0x1000;
        let bucket_pa: u64 = 0x1_0000;
        let nr: u32 = 200; // > MAX_STACK_TRACE_PCS (128)
        // Buffer large enough for bucket data: bucket_pa + 16 (data
        // offset) + 200*8 bytes + slack.
        let buf_size = (bucket_pa as usize) + 16 + (nr as usize) * 8 + 0x1000;
        let mut buf = vec![0u8; buf_size];
        let w32 = |b: &mut Vec<u8>, pa: u64, v: u32| {
            let o = pa as usize;
            b[o..o + 4].copy_from_slice(&v.to_le_bytes());
        };
        let w64 = |b: &mut Vec<u8>, pa: u64, v: u64| {
            let o = pa as usize;
            b[o..o + 8].copy_from_slice(&v.to_le_bytes());
        };
        // n_buckets = 1 at map offset 0.
        w32(&mut buf, map_pa, 1);
        // buckets[0] pointer at map + 16.
        w64(&mut buf, map_pa + 16, pa_to_kva(bucket_pa));
        // bucket struct: nr at bucket+0, data[] at bucket+16.
        w32(&mut buf, bucket_pa, nr);
        for j in 0..nr as u64 {
            w64(
                &mut buf,
                bucket_pa + 16 + j * 8,
                0xFFFF_FFFF_8000_0000 + j,
            );
        }
        let offsets = stackmap_offsets();
        let info = map_info("test_stack", BPF_MAP_TYPE_STACK_TRACE, pa_to_kva(map_pa), 1);
        let st = with_accessor(&buf, &offsets, |a| {
            render_stack_traces(a, &info).expect("truncating render must succeed")
        });
        assert_eq!(st.n_buckets, 1);
        assert_eq!(st.entries.len(), 1);
        assert_eq!(st.entries[0].nr, 200, "raw nr must be preserved");
        assert_eq!(
            st.entries[0].pcs.len(),
            MAX_STACK_TRACE_PCS as usize,
            "pcs must cap at MAX_STACK_TRACE_PCS (128)",
        );
        assert!(st.truncated, "nr > MAX_STACK_TRACE_PCS must set truncated");
        assert_eq!(st.buckets_unreadable, 0);
    }

    /// A non-null bucket pointer whose target struct page is unmapped
    /// (its KVA reverses to a PA past the buffer): the bucket is
    /// counted in `buckets_unreadable` and contributes no entry.
    #[test]
    fn render_stack_traces_unmapped_bucket_struct_counted() {
        let map_pa: u64 = 0x1000;
        let buf_size = (map_pa as usize) + 64;
        let mut buf = vec![0u8; buf_size];
        let w32 = |b: &mut Vec<u8>, pa: u64, v: u32| {
            let o = pa as usize;
            b[o..o + 4].copy_from_slice(&v.to_le_bytes());
        };
        let w64 = |b: &mut Vec<u8>, pa: u64, v: u64| {
            let o = pa as usize;
            b[o..o + 8].copy_from_slice(&v.to_le_bytes());
        };
        // n_buckets = 1; the slot pointer is non-null but points at a
        // KVA whose PA (0x100_0000) is far past the buffer end →
        // translate_any_kva fails on the bucket struct deref.
        w32(&mut buf, map_pa, 1);
        w64(&mut buf, map_pa + 16, pa_to_kva(0x100_0000));
        let offsets = stackmap_offsets();
        let info = map_info("test_stack", BPF_MAP_TYPE_STACK_TRACE, pa_to_kva(map_pa), 1);
        let st = with_accessor(&buf, &offsets, |a| {
            render_stack_traces(a, &info).expect("render must succeed despite unmapped bucket")
        });
        assert_eq!(st.n_buckets, 1);
        assert!(
            st.entries.is_empty(),
            "unmapped bucket struct must contribute no entry",
        );
        assert_eq!(
            st.buckets_unreadable, 1,
            "unmapped bucket struct must be counted, not silently dropped",
        );
    }

    /// `n_buckets` exceeding `MAX_STACK_TRACE_BUCKETS` (16384) sets the
    /// result's `truncated` flag even with no readable buckets. The
    /// scan caps at the limit; the raw `n_buckets` is preserved.
    #[test]
    fn render_stack_traces_n_buckets_over_cap_truncates() {
        let map_pa: u64 = 0x1000;
        // Only the n_buckets field is read here (all bucket slots past
        // the buffer are unreadable, but we set n_buckets above the
        // cap to force the truncated flag).
        let buf_size = (map_pa as usize) + 64;
        let mut buf = vec![0u8; buf_size];
        let huge = MAX_STACK_TRACE_BUCKETS + 1;
        let o = map_pa as usize;
        buf[o..o + 4].copy_from_slice(&huge.to_le_bytes());
        let offsets = stackmap_offsets();
        let info = map_info(
            "test_stack",
            BPF_MAP_TYPE_STACK_TRACE,
            pa_to_kva(map_pa),
            huge,
        );
        let st = with_accessor(&buf, &offsets, |a| {
            render_stack_traces(a, &info).expect("render must succeed")
        });
        assert_eq!(st.n_buckets, huge, "raw n_buckets must be preserved");
        assert!(
            st.truncated,
            "n_buckets > MAX_STACK_TRACE_BUCKETS must set truncated",
        );
    }

    // ---- render_fd_array_slots unmapped slot page ----------------

    /// FD-array offsets: `ptrs[]` at offset 16 (matches
    /// `dump::tests::synth_fd_array_offsets`).
    fn fd_array_offsets() -> super::super::super::btf_offsets::BpfMapOffsets {
        let mut o = super::super::super::btf_offsets::BpfMapOffsets::EMPTY;
        o.array_value = 16;
        o
    }

    /// A PROG_ARRAY whose scan range extends past the mapped buffer:
    /// slots within the buffer that are non-zero count as `populated`;
    /// slots whose computed KVA reverses to a PA past the buffer end
    /// count as `unreadable`. `scanned == min(max_entries,
    /// MAX_FD_ARRAY_SLOTS)`.
    #[test]
    fn render_fd_array_slots_unmapped_slot_counted() {
        let map_pa: u64 = 0x1000;
        let array_value: u64 = 16;
        // Buffer holds the map struct + 2 slots (indices 0,1). We scan
        // 4 entries, so slots 2 and 3 land past the buffer end and are
        // unreadable.
        let buf_size = (map_pa as usize) + (array_value as usize) + 2 * 8;
        let mut buf = vec![0u8; buf_size];
        let w64 = |b: &mut Vec<u8>, pa: u64, v: u64| {
            let o = pa as usize;
            b[o..o + 8].copy_from_slice(&v.to_le_bytes());
        };
        // Slot 0: populated (non-zero kernel pointer). Slot 1: zero.
        w64(&mut buf, map_pa + array_value, 0xFFFF_8000_0000_0001);
        w64(&mut buf, map_pa + array_value + 8, 0);
        let offsets = fd_array_offsets();
        // max_entries = 4 → slots 2,3 are past the buffer end.
        let info = map_info(
            "test_fd_array",
            BPF_MAP_TYPE_PROG_ARRAY,
            pa_to_kva(map_pa),
            4,
        );
        let fa = with_accessor(&buf, &offsets, |a| render_fd_array_slots(a, &info));
        assert_eq!(fa.scanned, 4, "scanned must equal min(max_entries, cap)");
        assert_eq!(
            fa.populated, 1,
            "only the readable non-zero slot 0 counts as populated",
        );
        assert_eq!(fa.indices, vec![0u32]);
        assert_eq!(
            fa.unreadable, 2,
            "the two slots past the mapped buffer must be counted unreadable",
        );
    }

    // ---- render_map feasible per-arm dispatch --------------------
    //
    // Per the coverage verdict, the value-page-SUCCESS arms (ARRAY /
    // multi-ARRAY / STRUCT_OPS read) are NOT host-feasible because
    // read_value/read_array route through the PTE-only translate_kva
    // (a real page-table walk) which the cr3=0 test kernel cannot
    // satisfy. The arms below use translate_any_kva (direct-map) or
    // perform no read, so they ARE feasible here.

    /// Build a minimal `RenderMapCtx` over `accessor` with everything
    /// optional set to None/empty.
    fn ctx<'a>(
        accessor: &'a GuestMemMapAccessor<'a>,
        arena_offsets: Option<&'a BpfArenaOffsets>,
        page_index: &'a ArenaPageIndex,
        metas: &'a [SdtAllocMeta],
    ) -> RenderMapCtx<'a> {
        RenderMapCtx {
            accessor,
            btf: None,
            num_cpus: 1,
            arena_offsets,
            shared_arena: None,
            arena_page_index: page_index,
            sdt_alloc_metas: metas,
            cast_map: None,
            arena_slot_index: None,
            cross_btf_fwd_index: None,
            scx_static_index: None,
            alloc_size_types: &[],
            rendered_slot_addrs: None,
        }
    }

    /// QUEUE map → the wildcard arm surfaces the exact
    /// `MAP_TYPE_EXPLANATIONS` string for QUEUE; no read occurs.
    #[test]
    fn render_map_queue_surfaces_explanation_string() {
        let buf = vec![0u8; 0x2000];
        let offsets = super::super::super::btf_offsets::BpfMapOffsets::EMPTY;
        let page_index = ArenaPageIndex::new();
        let metas: Vec<SdtAllocMeta> = Vec::new();
        let info = map_info("q", BPF_MAP_TYPE_QUEUE, pa_to_kva(0x1000), 16);
        let expected = MAP_TYPE_EXPLANATIONS
            .iter()
            .find(|(t, _)| *t == BPF_MAP_TYPE_QUEUE)
            .map(|(_, m)| (*m).to_string())
            .expect("QUEUE must have an explanation entry");
        let out = with_accessor(&buf, &offsets, |a| {
            let c = ctx(a, None, &page_index, &metas);
            render_map(&c, &info)
        });
        assert_eq!(
            out.error.as_deref(),
            Some(expected.as_str()),
            "QUEUE error must be the exact MAP_TYPE_EXPLANATIONS string",
        );
        assert!(out.value.is_none());
    }

    /// An unknown discriminant (not in any explicit arm nor the
    /// explanation table) surfaces the "unknown map_type N" diagnostic
    /// with the decimal discriminant.
    #[test]
    fn render_map_unknown_type_surfaces_decimal_diagnostic() {
        let buf = vec![0u8; 0x2000];
        let offsets = super::super::super::btf_offsets::BpfMapOffsets::EMPTY;
        let page_index = ArenaPageIndex::new();
        let metas: Vec<SdtAllocMeta> = Vec::new();
        let unknown: u32 = 0xDEAD_BEEF; // 3735928559
        let info = map_info("u", unknown, pa_to_kva(0x1000), 1);
        let out = with_accessor(&buf, &offsets, |a| {
            let c = ctx(a, None, &page_index, &metas);
            render_map(&c, &info)
        });
        let err = out.error.expect("unknown map_type must set an error");
        assert!(
            err.contains("unknown map_type 3735928559"),
            "error must name the decimal discriminant; got: {err}",
        );
    }

    /// ARENA map with `arena_offsets = None` surfaces the exact
    /// `ARENA_OFFSETS_UNAVAILABLE_MSG` and leaves `arena` unset (no
    /// snapshot taken).
    #[test]
    fn render_map_arena_no_offsets_surfaces_unavailable_msg() {
        let buf = vec![0u8; 0x2000];
        let offsets = super::super::super::btf_offsets::BpfMapOffsets::EMPTY;
        let page_index = ArenaPageIndex::new();
        let metas: Vec<SdtAllocMeta> = Vec::new();
        let info = map_info("arena", BPF_MAP_TYPE_ARENA, pa_to_kva(0x1000), 1);
        let out = with_accessor(&buf, &offsets, |a| {
            let c = ctx(a, None, &page_index, &metas);
            render_map(&c, &info)
        });
        assert_eq!(
            out.error.as_deref(),
            Some(ARENA_OFFSETS_UNAVAILABLE_MSG),
            "ARENA without offsets must surface the exact unavailable message",
        );
        assert!(out.arena.is_none(), "no snapshot must be taken");
    }

    /// RINGBUF map → the Ok arm sets `out.ringbuf` from the
    /// direct-map-readable `bpf_ringbuf` fields. Pins the rendered
    /// capacity/positions against the synthetic scene.
    #[test]
    fn render_map_ringbuf_ok_sets_ringbuf() {
        let map_pa: u64 = 0x1000;
        let rb_pa: u64 = 0x10_0000;
        let buf_size = (rb_pa as usize) + 0x1000;
        let mut buf = vec![0u8; buf_size];
        let w64 = |b: &mut Vec<u8>, pa: u64, v: u64| {
            let o = pa as usize;
            b[o..o + 8].copy_from_slice(&v.to_le_bytes());
        };
        // Ringbuf offsets: rbm_rb @ 0; mask @ 0, consumer @ 64,
        // producer @ 128, pending @ 192 (matches dump::tests).
        let mut offsets = super::super::super::btf_offsets::BpfMapOffsets::EMPTY;
        offsets.ringbuf_offsets =
            Some(super::super::super::btf_offsets::BpfRingbufOffsets {
                rbm_rb: 0,
                rb_mask: 0,
                rb_consumer_pos: 64,
                rb_producer_pos: 128,
                rb_pending_pos: 192,
            });
        // map.rb pointer → rb KVA.
        w64(&mut buf, map_pa, pa_to_kva(rb_pa));
        // mask = 0xFFF (capacity 4096); consumer 100, producer 200,
        // pending 150.
        w64(&mut buf, rb_pa, 0xFFF);
        w64(&mut buf, rb_pa + 64, 100);
        w64(&mut buf, rb_pa + 128, 200);
        w64(&mut buf, rb_pa + 192, 150);
        let page_index = ArenaPageIndex::new();
        let metas: Vec<SdtAllocMeta> = Vec::new();
        let info = map_info("rb", BPF_MAP_TYPE_RINGBUF, pa_to_kva(map_pa), 4096);
        let out = with_accessor(&buf, &offsets, |a| {
            let c = ctx(a, None, &page_index, &metas);
            render_map(&c, &info)
        });
        assert!(out.error.is_none(), "ringbuf render must not error");
        let rb = out.ringbuf.expect("RINGBUF Ok arm must set out.ringbuf");
        assert_eq!(rb.capacity, 4096);
        assert_eq!(rb.consumer_pos, 100);
        assert_eq!(rb.producer_pos, 200);
        assert_eq!(rb.pending_pos, 150);
        assert_eq!(rb.pending_bytes, 100, "producer - consumer = 100");
    }

    /// STACK_TRACE map → the Ok arm sets `out.stack_trace`. An
    /// all-empty (null-pointer) bucket scene renders with `n_buckets`
    /// set and no entries, exercising the STACK_TRACE dispatch arm
    /// end-to-end through `render_map`.
    #[test]
    fn render_map_stack_trace_ok_sets_stack_trace() {
        let map_pa: u64 = 0x1000;
        let buf_size = (map_pa as usize) + 64;
        let mut buf = vec![0u8; buf_size];
        // n_buckets = 2; both slot pointers left null (zero).
        let o = map_pa as usize;
        buf[o..o + 4].copy_from_slice(&2u32.to_le_bytes());
        let offsets = stackmap_offsets();
        let page_index = ArenaPageIndex::new();
        let metas: Vec<SdtAllocMeta> = Vec::new();
        let info = map_info("st", BPF_MAP_TYPE_STACK_TRACE, pa_to_kva(map_pa), 2);
        let out = with_accessor(&buf, &offsets, |a| {
            let c = ctx(a, None, &page_index, &metas);
            render_map(&c, &info)
        });
        assert!(out.error.is_none(), "empty stack-trace render must not error");
        let st = out
            .stack_trace
            .expect("STACK_TRACE Ok arm must set out.stack_trace");
        assert_eq!(st.n_buckets, 2);
        assert!(st.entries.is_empty(), "null buckets contribute no entries");
        assert!(!st.truncated);
    }
}
