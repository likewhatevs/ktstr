//! Shared IDR/xarray walk for host-side kernel memory reads.
//!
//! The kernel's IDR uses an xarray internally. These functions walk
//! the xarray tree structure in guest physical memory to enumerate
//! entries by index. Used by both BPF map and BPF program discovery.

use super::Kva;
use super::reader::GuestMem;
use super::symbols::kva_to_pa;

/// XA_CHUNK_SHIFT = 6, XA_CHUNK_SIZE = 64.
pub(crate) const XA_CHUNK_SIZE: u64 = 64;

/// Translate a kernel virtual address to a GuestMem offset, trying
/// direct mapping first, then page table walk.
///
/// SLAB allocations live in the direct mapping (PAGE_OFFSET..PAGE_END).
/// vmalloc'd and module addresses require a page table walk.
///
/// `cr3_pa` / `page_offset` / `l5` / `tcr_el1` mirror the signature
/// of [`GuestMem::translate_kva`]; on x86_64 `tcr_el1` is ignored,
/// on aarch64 `l5` is ignored.
#[allow(clippy::too_many_arguments)]
pub(crate) fn translate_any_kva(
    mem: &GuestMem,
    cr3_pa: u64,
    page_offset: u64,
    kva: u64,
    l5: bool,
    tcr_el1: u64,
) -> Option<u64> {
    let direct_pa = kva_to_pa(kva, page_offset);
    if direct_pa < mem.size() {
        return Some(direct_pa);
    }
    mem.translate_kva(cr3_pa, Kva(kva), l5, tcr_el1)
}

/// Load an entry from an xarray by index.
///
/// xa_node structs are SLAB-allocated and live in the direct mapping,
/// so their KVAs are translated via `kva_to_pa(kva, page_offset)`.
/// `slots_off` and `shift_off` are BTF-resolved byte offsets of
/// `slots` and `shift` within `struct xa_node`.
///
/// Returns `Some(0)` for empty slots or `Some(ptr)` for populated
/// entries. Out-of-bounds reads return 0 (empty slot).
pub(crate) fn xa_load(
    mem: &GuestMem,
    page_offset: u64,
    xa_head: u64,
    index: u64,
    slots_off: usize,
    shift_off: usize,
) -> Option<u64> {
    if xa_head == 0 {
        return Some(0);
    }

    // A non-node head (direct entry, value entry, or a reserved small
    // internal marker) is a single-entry xarray — see xa_is_node.
    if !xa_is_node(xa_head) {
        // Single-entry xarray: only index 0 is valid.
        return if index == 0 { Some(xa_head) } else { Some(0) };
    }

    // xa_head is a node pointer. Clear the internal marker bits.
    let mut node_kva = xa_head & !3u64;
    let mut shift = xa_node_shift(mem, page_offset, node_kva, shift_off);

    loop {
        // Guest-supplied shift is read from `xa_node.shift` (u8) so the
        // attacker controls 0..=255. `index >> shift` is UB in Rust when
        // `shift >= 64` (debug panic, release wrap), so refuse to deref a
        // node whose advertised shift is out of range. A real kernel-side
        // xa_node never holds shift >= 64 (XA_CHUNK_SHIFT is 6 and the
        // tree depth is bounded by ulong width), so a value here means
        // either guest memory corruption or a hostile guest writing a
        // crafted xa_node — return Some(0) to surface "no entry" rather
        // than tripping the host on bogus input.
        if shift >= 64 {
            return Some(0);
        }
        let slot_idx = (index >> shift) & (XA_CHUNK_SIZE - 1);
        let slot_pa = kva_to_pa(node_kva + slots_off as u64 + slot_idx * 8, page_offset);
        let entry = mem.read_u64(slot_pa, 0);

        if entry == 0 {
            return Some(0);
        }

        if !xa_is_node(entry) {
            // Terminal entry: a leaf pointer, a value entry, or a
            // reserved small internal marker — not a node to descend.
            return Some(entry);
        }

        // Internal node — descend.
        node_kva = entry & !3u64;
        if shift < 6 {
            return Some(0);
        }
        shift -= 6; // XA_CHUNK_SHIFT
    }
}

/// Mirror of the kernel's `xa_is_node` (include/linux/xarray.h): an
/// xarray slot is a node to descend into only when it is an internal
/// entry (`entry & 3 == 2` — bit 1 set, bit 0 clear) AND its value
/// exceeds 4096. The kernel reserves internal values 0..=4096 for
/// sibling entries, the RETRY marker, and ZERO/reserved encodings, and
/// `xas_load` descends `while (xa_is_node(entry))` (lib/xarray.c), so a
/// small internal-looking marker — or a value entry (bit 0 set) — is
/// terminal, not a node. Testing only `entry & 2` would mis-descend
/// those into a bogus tiny node KVA.
fn xa_is_node(entry: u64) -> bool {
    entry & 3 == 2 && entry > 4096
}

/// Read the `shift` field from an xa_node (SLAB-allocated, direct mapping).
pub(crate) fn xa_node_shift(
    mem: &GuestMem,
    page_offset: u64,
    node_kva: u64,
    shift_off: usize,
) -> u64 {
    let pa = kva_to_pa(node_kva, page_offset);
    mem.read_u8(pa, shift_off) as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monitor::symbols::DEFAULT_PAGE_OFFSET;

    #[test]
    fn xa_node_shift_reads_shift_byte() {
        // Place an xa_node at PA 0x100 with shift=12 at byte offset 2.
        let mut buf = [0u8; 0x200];
        let shift_off = 2;
        buf[0x100 + shift_off] = 12;

        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let node_kva = DEFAULT_PAGE_OFFSET + 0x100;
        assert_eq!(
            xa_node_shift(&mem, DEFAULT_PAGE_OFFSET, node_kva, shift_off),
            12
        );
    }

    #[test]
    fn xa_node_shift_zero() {
        let mut buf = [0u8; 0x200];
        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let node_kva = DEFAULT_PAGE_OFFSET;
        // shift_off=0, buf[0]=0 -> shift=0.
        assert_eq!(xa_node_shift(&mem, DEFAULT_PAGE_OFFSET, node_kva, 0), 0);
    }

    #[test]
    fn xa_node_shift_max_u8() {
        let mut buf = [0u8; 0x100];
        buf[0x10] = 255;
        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let node_kva = DEFAULT_PAGE_OFFSET;
        assert_eq!(
            xa_node_shift(&mem, DEFAULT_PAGE_OFFSET, node_kva, 0x10),
            255
        );
    }

    // -- xa_load classification (mirrors kernel xa_is_node) --

    #[test]
    fn xa_load_empty_head_returns_zero() {
        let mut buf = [0u8; 0x100];
        // SAFETY: buf outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        assert_eq!(xa_load(&mem, DEFAULT_PAGE_OFFSET, 0, 0, 8, 0), Some(0));
    }

    #[test]
    fn xa_load_single_entry_direct_pointer() {
        // An 8-byte-aligned head (bits 0-1 clear) is not a node: it's a
        // single-entry xarray holding that pointer at index 0.
        let mut buf = [0u8; 0x100];
        // SAFETY: buf outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let ptr = DEFAULT_PAGE_OFFSET + 0x40;
        assert_eq!(xa_load(&mem, DEFAULT_PAGE_OFFSET, ptr, 0, 8, 0), Some(ptr));
        assert_eq!(xa_load(&mem, DEFAULT_PAGE_OFFSET, ptr, 1, 8, 0), Some(0));
    }

    #[test]
    fn xa_load_root_value_entry_is_single_entry() {
        // A head with bit 1 set but value <= 4096 (7 = a value entry,
        // bits 0+1 set) is NOT an xa_node (xa_is_node requires > 4096);
        // it is a single-entry xarray. The prior `xa_head & 2` test
        // would have mis-treated it as a node and descended.
        let mut buf = [0u8; 0x100];
        // SAFETY: buf outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        assert_eq!(xa_load(&mem, DEFAULT_PAGE_OFFSET, 7, 0, 8, 0), Some(7));
        assert_eq!(xa_load(&mem, DEFAULT_PAGE_OFFSET, 7, 1, 8, 0), Some(0));
    }

    #[test]
    fn xa_load_node_slot_value_entry_is_terminal() {
        // A real node (kva | 2, > 4096) whose slot 0 holds a VALUE entry
        // (7 = 0b111, bits 0+1 set, so `& 3 == 3`) must be returned
        // terminal, NOT descended. xa_is_node(7) is false via the
        // `& 3 == 2` clause — a value entry is not internal. The prior
        // `entry & 2` test (bit 1 set) mis-descended into a bogus tiny
        // node KVA (7 & !3 = 4).
        let mut buf = [0u8; 0x200];
        let pa = 0x100usize;
        buf[pa] = 0; // shift = 0 (leaf level)
        buf[pa + 8..pa + 16].copy_from_slice(&7u64.to_le_bytes()); // slot[0] = 7 (value entry)
        // SAFETY: buf outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let head = (DEFAULT_PAGE_OFFSET + pa as u64) | 2;
        assert_eq!(xa_load(&mem, DEFAULT_PAGE_OFFSET, head, 0, 8, 0), Some(7));
    }

    #[test]
    fn xa_load_node_slot_small_internal_marker_is_terminal() {
        // The `> 4096` size clause: a slot holding a true internal marker
        // (6 = 0b110, `& 3 == 2` — bit 1 set, bit 0 clear) but with value
        // <= 4096 is a kernel-reserved entry (sibling / RETRY / ZERO), NOT
        // a node. xa_is_node(6) is false via the `> 4096` clause, so it is
        // returned terminal. The prior `entry & 2` test (6 & 2 = 2)
        // mis-descended into node_kva = 6 & !3 = 4 (a bogus tiny KVA).
        let mut buf = [0u8; 0x200];
        let pa = 0x100usize;
        buf[pa] = 0; // shift = 0 (leaf level)
        buf[pa + 8..pa + 16].copy_from_slice(&6u64.to_le_bytes()); // slot[0] = 6 (internal marker, <= 4096)
        // SAFETY: buf outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let head = (DEFAULT_PAGE_OFFSET + pa as u64) | 2;
        assert_eq!(xa_load(&mem, DEFAULT_PAGE_OFFSET, head, 0, 8, 0), Some(6));
    }

    #[test]
    fn xa_load_node_slot_pointer_leaf_resolves() {
        // A node whose slot holds an 8-byte-aligned pointer returns it
        // as a leaf (real descent + leaf path stays correct).
        let mut buf = [0u8; 0x200];
        let pa = 0x100usize;
        buf[pa] = 0; // shift = 0
        let leaf = DEFAULT_PAGE_OFFSET + 0x40;
        buf[pa + 8..pa + 16].copy_from_slice(&leaf.to_le_bytes());
        // SAFETY: buf outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let head = (DEFAULT_PAGE_OFFSET + pa as u64) | 2;
        assert_eq!(
            xa_load(&mem, DEFAULT_PAGE_OFFSET, head, 0, 8, 0),
            Some(leaf)
        );
    }

    #[test]
    fn xa_load_multi_level_node_descends_to_leaf() {
        // Two real nodes (both > 4096): index 5 walks node1.slot[0] (a
        // node entry) then node2.slot[5] (the leaf). Confirms genuine
        // internal entries still descend after the xa_is_node tightening.
        let mut buf = [0u8; 0x400];
        let n1 = 0x100usize;
        let n2 = 0x200usize;
        buf[n1] = 6; // node1 shift = 6 (xa_load decrements by 6 per level)
        let node2_kva = DEFAULT_PAGE_OFFSET + n2 as u64;
        // node1.slot[(5 >> 6) & 63 = 0] = node2 (internal node entry)
        buf[n1 + 8..n1 + 16].copy_from_slice(&(node2_kva | 2).to_le_bytes());
        // node2.slot[(5 >> 0) & 63 = 5] = leaf pointer
        let leaf = DEFAULT_PAGE_OFFSET + 0x80;
        buf[n2 + 8 + 5 * 8..n2 + 8 + 5 * 8 + 8].copy_from_slice(&leaf.to_le_bytes());
        // SAFETY: buf outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let head = (DEFAULT_PAGE_OFFSET + n1 as u64) | 2;
        assert_eq!(
            xa_load(&mem, DEFAULT_PAGE_OFFSET, head, 5, 8, 0),
            Some(leaf)
        );
    }
}
