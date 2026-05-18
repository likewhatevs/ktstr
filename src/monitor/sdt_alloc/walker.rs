//! Radix-tree walker that descends the 3-level sdt_alloc tree from
//! frozen guest memory and emits one [`super::SdtAllocEntry`] per
//! live leaf allocation.
//!
//! [`walk_sdt_allocator`] is the public entry point — caller hands in
//! the raw `struct scx_allocator` byte image, the kern_vm window base,
//! the resolved offsets, and a [`crate::monitor::btf_render::MemReader`]
//! used to chase arena pointers during BTF rendering.
//!
//! [`TreeWalker`] is the internal recursive descent. Level enumeration
//! flips by depth: internal levels (0, 1) iterate every non-NULL child
//! pointer in `chunk->descs[]`; leaf level (2) reads the `allocated[]`
//! bitmap and emits one [`super::SdtAllocEntry`] per set bit. See the
//! module-level "Liveness" doc on the parent module for the bitmap
//! semantics rationale.

use btf_rs::Btf;

use crate::monitor::Kva;
use crate::monitor::btf_render::{MemReader, RenderedValue, render_value_with_mem};
use crate::monitor::dump::hex_dump;
use crate::monitor::guest::GuestKernel;

use super::{
    MAX_ELEM_SIZE, MAX_SDT_ALLOC_ENTRIES, MIN_ELEM_SIZE, SDT_TASK_CHUNK_BITMAP_U64S,
    SDT_TASK_ENTS_PER_CHUNK, SDT_TASK_LEVELS, SdtAllocEntry, SdtAllocOffsets, SdtAllocatorSnapshot,
    read_u64_at,
};

/// Walk an `scx_allocator` from frozen guest memory and surface every
/// live allocation.
///
/// `allocator_bytes` is the raw byte image of the `struct
/// scx_allocator` instance — the caller reads this from the
/// scheduler's `.bss` (or wherever the allocator is declared) and
/// hands the slice in so the walker doesn't need to know symbol
/// resolution machinery.
///
/// `kern_vm_start` is the kernel-side base of the arena's user_vm
/// window — `bpf_arena.kern_vm->addr + GUARD_HALF`, the same value
/// [`crate::monitor::arena::snapshot_arena`] computes. Arena pointers in the
/// tree are `__arena` pointers whose low 32 bits index this window,
/// so every translation goes:
///   `kern_va = kern_vm_start + (ptr & 0xFFFF_FFFF)`
///
/// `target_type_id` is the BTF type id the renderer applies to
/// the bytes after the `tid` header. Pass 0 (or a discovered id from
/// [`discover_payload_btf_id`]) — 0 routes to a hex dump.
///
/// `payload_type_reason` is a human-readable string describing why
/// `target_type_id` is 0 (when it is); ignored when the id is
/// non-zero. Surfaces in [`SdtAllocatorSnapshot::payload_type_reason`]
/// so an operator can distinguish "no candidate of this size" from
/// "ambiguous candidates" without re-deriving the heuristic.
///
/// The walk is best-effort: a corrupt desc / chunk / data pointer
/// stops descent into that subtree (incrementing
/// [`SdtAllocatorSnapshot::skipped_subtrees`]) and skips to the next
/// bit, so a single stale slot can't truncate the whole dump.
//
// `clippy::too_many_arguments` is silenced here: every parameter
// is genuinely needed and bundling them into a config struct would
// just shift the surface area without reducing it. The kernel /
// BTF / offsets references must stay independent borrows so the
// caller can compose them from different sources (vmlinux BTF +
// program BTF, accessor + arena snapshot kern_vm_start, etc.).
#[allow(clippy::too_many_arguments)]
pub fn walk_sdt_allocator(
    kernel: &GuestKernel,
    kern_vm_start: u64,
    allocator_bytes: &[u8],
    offsets: &SdtAllocOffsets,
    btf: &Btf,
    target_type_id: u32,
    payload_type_reason: impl Into<String>,
    allocator_name: impl Into<String>,
    mem: &dyn MemReader,
) -> SdtAllocatorSnapshot {
    let mut snap = SdtAllocatorSnapshot {
        allocator_name: allocator_name.into(),
        entries: Vec::new(),
        truncated: false,
        skipped_subtrees: 0,
        elem_size: 0,
        target_type_id,
        payload_type_reason: payload_type_reason.into(),
        all_slot_addrs: Vec::new(),
    };

    // Read pool.elem_size from the in-bss allocator image. This is
    // the per-slot stride the leaf walker needs to know how many
    // bytes to read for each allocation.
    let pool_off = offsets.allocator_pool + offsets.pool_elem_size;
    let Some(elem_size) = read_u64_at(allocator_bytes, pool_off) else {
        return snap;
    };
    if !(MIN_ELEM_SIZE..=MAX_ELEM_SIZE).contains(&elem_size) {
        // Out-of-range: corrupt allocator or torn snapshot. Surface
        // empty entries with the captured elem_size so a consumer
        // can see the diagnostic.
        snap.elem_size = elem_size;
        return snap;
    }
    snap.elem_size = elem_size;

    // Sanity: the data header (8 bytes for sdt_id) must fit inside
    // the slot — without this, payload_size would be negative.
    let header = offsets.data_header_size;
    if elem_size < header as u64 {
        return snap;
    }
    let payload_size = (elem_size - header as u64) as usize;

    // Read the root descriptor pointer.
    let Some(root_ptr) = read_u64_at(allocator_bytes, offsets.allocator_root) else {
        return snap;
    };
    if root_ptr == 0 {
        return snap;
    }

    let mut walker = TreeWalker {
        kernel,
        kern_vm_start,
        offsets,
        btf,
        target_type_id,
        payload_size,
        mem,
        out: &mut snap,
    };
    walker.descend(root_ptr, 0);

    snap
}

/// Internal walker state. Bundles the read-only inputs the recursive
/// descent threads through every call so each function takes one
/// `&mut self` and the actual position arguments.
struct TreeWalker<'a> {
    kernel: &'a GuestKernel,
    kern_vm_start: u64,
    offsets: &'a SdtAllocOffsets,
    btf: &'a Btf,
    target_type_id: u32,
    payload_size: usize,
    /// `MemReader` used by [`render_value_with_mem`] when rendering
    /// each leaf payload — lets the BTF renderer chase `__arena`
    /// pointers within the payload (e.g. an entry holding a pointer
    /// to another arena struct) into typed contents instead of
    /// emitting raw hex.
    mem: &'a dyn MemReader,
    out: &'a mut SdtAllocatorSnapshot,
}

impl<'a> TreeWalker<'a> {
    /// Descend into a `sdt_desc` at level `level`. Levels 0 and 1
    /// scan every position in `chunk->descs[]` and recurse into each
    /// non-NULL child pointer (the bitmap is unusable for enumeration
    /// at internal levels — see the module-level "Liveness" docs).
    /// Level 2 reads `chunk->data[]` and emits one [`SdtAllocEntry`]
    /// per allocated bit in this descriptor's `allocated[]` bitmap.
    ///
    /// Increments [`SdtAllocatorSnapshot::skipped_subtrees`] on every
    /// early return that abandons descent into a non-trivial subtree
    /// — translate failures, out-of-range `nr_free`, NULL `chunk`. A
    /// NULL `chunk->descs[pos]` at an internal level is a never-created
    /// subtree and is skipped silently (NOT counted as a skipped
    /// subtree).
    fn descend(&mut self, desc_ptr: u64, level: usize) {
        if level >= SDT_TASK_LEVELS {
            return;
        }

        // Once `emit_leaf` has flipped `truncated`, the cap on entries
        // is reached. Continuing to descend would still grow
        // `all_slot_addrs` (every leaf appended unconditionally before
        // the `entries` cap check) and waste work on every remaining
        // subtree. Bail at the top of every recursive call so descent
        // halts globally, not just at the next leaf.
        if self.out.truncated {
            return;
        }

        // Translate the arena pointer to a kernel VA, then to a PA.
        let Some(desc_pa) = self.translate_arena_ptr(desc_ptr) else {
            self.out.skipped_subtrees = self.out.skipped_subtrees.saturating_add(1);
            return;
        };

        // Read the descriptor: bitmap, nr_free (sanity), chunk pointer.
        let mut allocated = [0u64; SDT_TASK_CHUNK_BITMAP_U64S];
        let mem = self.kernel.mem();
        for (i, slot) in allocated.iter_mut().enumerate() {
            *slot = mem.read_u64(desc_pa, self.offsets.desc_allocated + i * 8);
        }
        let nr_free = mem.read_u64(desc_pa, self.offsets.desc_nr_free);
        // Sanity: nr_free is u64 but bounded by 512 in the kernel.
        // A wild value indicates a torn read or an uninitialized
        // descriptor — abort descent into this subtree.
        if nr_free > SDT_TASK_ENTS_PER_CHUNK as u64 {
            self.out.skipped_subtrees = self.out.skipped_subtrees.saturating_add(1);
            return;
        }
        let chunk_ptr = mem.read_u64(desc_pa, self.offsets.desc_chunk);
        if chunk_ptr == 0 {
            self.out.skipped_subtrees = self.out.skipped_subtrees.saturating_add(1);
            return;
        }
        let Some(chunk_pa) = self.translate_arena_ptr(chunk_ptr) else {
            self.out.skipped_subtrees = self.out.skipped_subtrees.saturating_add(1);
            return;
        };

        // Enumerate slots. The leaf level (`SDT_TASK_LEVELS - 1`)
        // reads `allocated[]` directly: a set bit there means the
        // `sdt_data *` in `chunk->data[pos]` is live (per the
        // module-level "Liveness" docs). Internal levels (0 and 1)
        // ignore the bitmap entirely — `lib/sdt_alloc.bpf.c` only
        // sets a parent bit once a child subtree is FULL, so SET on
        // an internal level means "subtree full" and CLEAR is "empty,
        // partial, or never created." The common-case bitmap (few
        // tasks, deep tree) is all-zero at the internal levels, so
        // iterating set bits would surface zero allocations. Walk
        // every position and descend into any non-NULL `desc *`; a
        // NULL pointer is a never-created subtree we skip silently.
        if level == SDT_TASK_LEVELS - 1 {
            for (word_idx, &word_value) in allocated.iter().enumerate() {
                let mut word = word_value;
                while word != 0 {
                    let bit = word.trailing_zeros() as usize;
                    word &= word - 1;
                    let pos = word_idx * 64 + bit;
                    if pos >= SDT_TASK_ENTS_PER_CHUNK {
                        continue;
                    }

                    // chunk->data[pos] at chunk_pa + chunk_union +
                    // pos * 8.
                    let entry_ptr_off = self.offsets.chunk_union + pos * 8;
                    let entry_ptr = mem.read_u64(chunk_pa, entry_ptr_off);
                    if entry_ptr == 0 {
                        // Pristine slot: bit was set but
                        // `chunk->data[pos]` never got populated. The
                        // kernel allocator populates
                        // `chunk->data[pos]` after setting the bit
                        // (in `scx_alloc_internal`), so a snapshot
                        // captured between the bit set and the
                        // pointer store sees this transient state.
                        // Skip without counting as a skipped subtree
                        // — it's a legitimate transient. Surface a
                        // `tracing::debug!` so an operator
                        // diagnosing missing slots can see the race
                        // without re-deriving "where did the live
                        // bit go?".
                        tracing::debug!(
                            allocator = %self.out.allocator_name,
                            pos,
                            "sdt_alloc walker: leaf data[pos] == 0 (bit set, \
                             pointer store not yet committed — scx_alloc_internal \
                             populates the pointer after the bitmap bit)",
                        );
                        continue;
                    }

                    self.emit_leaf(entry_ptr);
                }
            }
        } else {
            // Internal level: scan every position in chunk->descs[]
            // and descend into any non-NULL child pointer.
            for pos in 0..SDT_TASK_ENTS_PER_CHUNK {
                let entry_ptr_off = self.offsets.chunk_union + pos * 8;
                let entry_ptr = mem.read_u64(chunk_pa, entry_ptr_off);
                if entry_ptr == 0 {
                    // Never-created subtree: `desc_find_empty` only
                    // writes `desc_children[pos]` when it allocates
                    // a new chunk for a previously empty slot. A
                    // NULL pointer is a legitimate gap, not an
                    // anomaly; skip without counting as a skipped
                    // subtree. `trace!` (not `debug!`) because
                    // a full tree has up to 512 NULL slots per
                    // internal node — a sparse allocator would
                    // flood the debug log otherwise.
                    tracing::trace!(
                        allocator = %self.out.allocator_name,
                        level,
                        pos,
                        "sdt_alloc walker: internal desc[pos] == 0 \
                         (never-created subtree)",
                    );
                    continue;
                }
                self.descend(entry_ptr, level + 1);
            }
        }
    }

    /// Emit one leaf allocation: read tid + payload, BTF-render.
    fn emit_leaf(&mut self, data_ptr: u64) {
        self.out.all_slot_addrs.push(data_ptr & 0xFFFF_FFFF);
        if self.out.entries.len() >= MAX_SDT_ALLOC_ENTRIES {
            self.out.truncated = true;
            return;
        }
        let Some(data_pa) = self.translate_arena_ptr(data_ptr) else {
            self.out.skipped_subtrees = self.out.skipped_subtrees.saturating_add(1);
            return;
        };
        let mem = self.kernel.mem();

        // tid: union sdt_id { s64 val; struct { s32 idx; s32 genn; } }
        // — read as two s32s.
        let idx = mem.read_u32(data_pa, 0) as i32;
        let genn = mem.read_u32(data_pa, 4) as i32;

        // Payload: read self.payload_size bytes after the
        // data_header_size offset.
        let mut payload_bytes = vec![0u8; self.payload_size];
        let n = mem.read_bytes(
            data_pa + self.offsets.data_header_size as u64,
            &mut payload_bytes,
        );
        payload_bytes.truncate(n);
        if payload_bytes.is_empty() {
            // Couldn't read a single byte of payload — chunk was at
            // end-of-DRAM or unmapped. Surface the tid alone (still
            // useful for an operator) with an Unsupported payload
            // carrying the diagnostic reason.
            self.out.entries.push(SdtAllocEntry {
                idx,
                genn,
                user_addr: data_ptr & 0xFFFF_FFFF,
                payload: RenderedValue::Unsupported {
                    reason: "payload read failed: end-of-DRAM or unmapped page".into(),
                },
            });
            return;
        }

        // Zeroed-slot race: `scx_alloc_free_idx` writes zeros to the
        // payload BEFORE clearing the bitmap (see the module-level
        // "Race window" doc). A frozen snapshot captured between
        // those two writes sees a live bitmap bit but an all-zero
        // payload. Surface a `tracing::debug!` so an operator
        // triaging "this slot's payload looks empty" can correlate
        // it with the mid-free race rather than chasing a phantom
        // bug in the renderer. The render itself still proceeds —
        // the consumer can decide whether to interpret an all-zero
        // entry as a real "freshly-allocated zero-init" allocation
        // or as a mid-free residue.
        if payload_bytes.iter().all(|&b| b == 0) {
            tracing::debug!(
                allocator = %self.out.allocator_name,
                idx,
                genn,
                user_addr = format_args!("{:#x}", data_ptr & 0xFFFF_FFFF),
                payload_len = payload_bytes.len(),
                "sdt_alloc walker: all-zero payload (mid-free race? scx_alloc_free_idx \
                 zeros payload before clearing the bitmap)",
            );
        }

        let payload = if self.target_type_id != 0 {
            render_value_with_mem(self.btf, self.target_type_id, &payload_bytes, self.mem)
        } else {
            RenderedValue::Bytes {
                hex: hex_dump(&payload_bytes),
            }
        };

        self.out.entries.push(SdtAllocEntry {
            idx,
            genn,
            user_addr: data_ptr & 0xFFFF_FFFF,
            payload,
        });
    }

    /// Translate a `__arena` pointer to a guest physical address.
    ///
    /// Mirrors the formula in [`super::arena`]: the kernel composes
    /// the actual kern-VA from the LOW 32 bits of the arena pointer
    /// added to `kern_vm_start`. Returns `None` if the translate
    /// fails (page unmapped, PA out of DRAM bounds).
    fn translate_arena_ptr(&self, ptr: u64) -> Option<u64> {
        if ptr == 0 {
            return None;
        }
        let kva = self.kern_vm_start.wrapping_add(ptr & 0xFFFF_FFFF);
        let pa = self.kernel.mem().translate_kva(
            self.kernel.cr3_pa(),
            Kva(kva),
            self.kernel.l5(),
            self.kernel.tcr_el1(),
        )?;
        // Bounds-check the PA: a corrupt PTE could point past
        // end-of-DRAM. Translate guarantees page alignment but not
        // DRAM membership beyond the first page.
        if pa >= self.kernel.mem().size() {
            return None;
        }
        Some(pa)
    }
}
