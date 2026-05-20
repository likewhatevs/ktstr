//! Output types for the sdt_alloc walker — one [`SdtAllocEntry`] per
//! live allocation, surfaced inside an [`SdtAllocatorSnapshot`] that
//! also carries the truncation flag, skipped-subtree count, and
//! payload-type-resolution diagnostics.

use serde::{Deserialize, Serialize};

use crate::monitor::btf_render::RenderedValue;

/// One leaf allocation surfaced from the sdt_alloc tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct SdtAllocEntry {
    /// `sdt_data.tid.idx` — the 27-bit packed slot index. Negative
    /// values are surfaced verbatim (the kernel uses `s32`); the host
    /// does not interpret sign here.
    pub idx: i32,
    /// `sdt_data.tid.genn` — incremented on recycle so consumers can
    /// distinguish reallocations of the same `idx`.
    pub genn: i32,
    /// Low 32 bits of the user-side arena pointer to the `sdt_data`
    /// slot. Computed by `TreeWalker::emit_leaf` as
    /// `data_ptr & 0xFFFF_FFFF`, NOT the full user-side VA — slot
    /// addresses already live in the 32-bit `arena.user_vm_start`
    /// window, so the masked low 32 bits are sufficient for
    /// correlation against pointer values an operator sees in BPF
    /// program output AND match the masking convention the renderer's
    /// arena-type bridge keys on (see
    /// [`crate::monitor::btf_render::MemReader::resolve_arena_type`]).
    ///
    /// Distinct from [`crate::monitor::arena::ArenaPage::user_addr`], which
    /// carries the FULL user-side VA — the page-snapshot consumer
    /// surfaces the unmasked address so cross-arena callers (multiple
    /// arena maps with different `user_vm_start` bases) do not collide
    /// on the same low-32 bits.
    pub user_addr: u64,
    /// BTF-rendered payload (everything after the 8-byte
    /// `union sdt_id` tid header). Falls back to a hex dump when
    /// payload type discovery failed; renders as
    /// [`RenderedValue::Unsupported`] when the payload couldn't be
    /// read at all (end-of-DRAM, unmapped page).
    pub payload: RenderedValue,
}

impl std::fmt::Display for SdtAllocEntry {
    /// Human-readable rendering: `idx=N genn=M user_addr=0x... payload=<value>`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "idx={} genn={} user_addr={:#x} payload=",
            self.idx, self.genn, self.user_addr
        )?;
        std::fmt::Display::fmt(&self.payload, f)
    }
}

/// All sdt_alloc allocations surfaced from a single allocator.
///
/// One [`SdtAllocatorSnapshot`] per allocator instance — the dump
/// pipeline today walks one allocator (the scheduler's primary) but
/// the type shape leaves room for multiple allocators to coexist in
/// the snapshot.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct SdtAllocatorSnapshot {
    /// Allocator name (e.g. the .bss symbol the allocator was read
    /// from, like `"scx_task_allocator"`). Surfaced so a consumer
    /// reading multiple allocator dumps can tell them apart.
    pub allocator_name: String,
    /// Live allocations, in tree-walk order (level 0 → 1 → 2,
    /// monotonic pos at each level). Capped at
    /// `MAX_SDT_ALLOC_ENTRIES`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub entries: Vec<SdtAllocEntry>,
    /// True when the walk stopped at `MAX_SDT_ALLOC_ENTRIES` before
    /// covering every live bit in the bitmaps.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub truncated: bool,
    /// Count of subtrees the walker abandoned mid-descent due to a
    /// pointer translate failure, an out-of-range `nr_free`, a NULL
    /// `chunk` pointer, or any other diagnostic that aborts descent
    /// into one subtree without poisoning the rest of the walk. A
    /// non-zero value here means the dump is partial — some live
    /// allocations may not be in `entries`.
    ///
    /// Always serialized — a zero value carries diagnostic information
    /// ("walker reached the end of the tree without skipping anything"),
    /// and suppressing it on default would make consumers conflate "zero
    /// skipped" with "field absent / older schema". Mirrors the
    /// always-serialize policy used by sibling `elem_size` and
    /// `target_type_id`.
    pub skipped_subtrees: u32,
    /// Diagnostic: the per-pool slot stride. Surfaces alongside the
    /// rendered entries so a consumer can spot when the rendered
    /// payload size diverges from the declared one.
    pub elem_size: u64,
    /// Diagnostic: the BTF type id used to render payload bytes.
    /// 0 when `discover_payload_btf_id` returned no candidate and
    /// the renderer fell back to hex.
    pub target_type_id: u32,
    /// Diagnostic: when `discover_payload_btf_id` returned 0, the
    /// reason (e.g. `"no candidate of size 16"`,
    /// `"ambiguous: 3 candidates"`, `"payload_size == 0"`). Empty on
    /// successful BTF resolve. Lets an operator distinguish the
    /// fallback paths without re-deriving the heuristic.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub payload_type_reason: String,
    #[serde(skip)]
    pub all_slot_addrs: Vec<u64>,
}

impl std::fmt::Display for SdtAllocatorSnapshot {
    /// Header line + one entry per allocation, indented. Diagnostic
    /// lines (truncated, skipped_subtrees) appended when non-default.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "sdt_alloc {} (elem_size={}, target_type_id={}",
            self.allocator_name, self.elem_size, self.target_type_id
        )?;
        if !self.payload_type_reason.is_empty() {
            write!(f, ", reason={}", self.payload_type_reason)?;
        }
        write!(f, "): {} live", self.entries.len())?;
        if self.truncated {
            f.write_str(" (truncated)")?;
        }
        if self.skipped_subtrees > 0 {
            write!(f, " ({} subtrees skipped)", self.skipped_subtrees)?;
        }
        for entry in &self.entries {
            f.write_str("\n")?;
            crate::monitor::btf_render::write_value_at_depth(f, &entry.payload, 2)?;
        }
        Ok(())
    }
}
