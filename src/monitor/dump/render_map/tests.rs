//! Unit tests for [`super`] (the `monitor::dump::render_map` module).
//! Co-located via the `tests` submodule pattern (sibling file).

#![cfg(test)]

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
    let struct_info = ((BTF_KIND_STRUCT << 24) & 0x1f00_0000) | (1u32 << 31) | 2u32; // kind_flag=1, vlen=2
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
        w64(&mut buf, bucket_pa + 16 + j * 8, 0xFFFF_FFFF_8000_0000 + j);
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
    offsets.ringbuf_offsets = Some(super::super::super::btf_offsets::BpfRingbufOffsets {
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
    assert!(
        out.error.is_none(),
        "empty stack-trace render must not error"
    );
    let st = out
        .stack_trace
        .expect("STACK_TRACE Ok arm must set out.stack_trace");
    assert_eq!(st.n_buckets, 2);
    assert!(st.entries.is_empty(), "null buckets contribute no entries");
    assert!(!st.truncated);
}
