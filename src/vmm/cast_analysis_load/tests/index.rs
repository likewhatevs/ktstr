use super::super::*;
use super::*;
use goblin::elf::header as h;
use goblin::elf::section_header as sh;
use goblin::elf::sym as syms;

/// Multiple BTFs: the index records the first BTF seen for any
/// duplicate name, so an entry's `(idx, type_id)` reflects the
/// first-write-wins policy. The renderer only consults the
/// cross-BTF index when local in-BTF resolution failed, so a
/// name conflict resolved locally never reaches the index.
#[test]
fn build_fwd_index_first_write_wins_on_duplicate_name() {
    // BTF #0: struct foo at id 2 (u64 at offset 0)
    let mut strings_0 = vec![0u8];
    let n_int_0 = push_btf_name(&mut strings_0, "u64");
    let n_foo_0 = push_btf_name(&mut strings_0, "foo");
    let n_x_0 = push_btf_name(&mut strings_0, "x");
    let types_0 = vec![
        SynKind::Int {
            name_off: n_int_0,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        SynKind::Struct {
            name_off: n_foo_0,
            size: 8,
            members: vec![SynMember {
                name_off: n_x_0,
                type_id: 1,
                byte_offset: 0,
            }],
        },
    ];
    let blob_0 = build_btf_full(&types_0, &strings_0);
    let btf_0 = Arc::new(Btf::from_bytes(&blob_0).expect("parse btf 0"));

    // BTF #1: also has struct foo (different layout!) at id 2.
    // Index keeps the BTF #0 entry per first-write-wins.
    let mut strings_1 = vec![0u8];
    let n_int_1 = push_btf_name(&mut strings_1, "u64");
    let n_foo_1 = push_btf_name(&mut strings_1, "foo");
    let n_y_1 = push_btf_name(&mut strings_1, "y");
    let types_1 = vec![
        SynKind::Int {
            name_off: n_int_1,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        SynKind::Struct {
            name_off: n_foo_1,
            size: 16,
            members: vec![SynMember {
                name_off: n_y_1,
                type_id: 1,
                byte_offset: 8,
            }],
        },
    ];
    let blob_1 = build_btf_full(&types_1, &strings_1);
    let btf_1 = Arc::new(Btf::from_bytes(&blob_1).expect("parse btf 1"));

    let btfs = vec![btf_0, btf_1];
    let index = build_fwd_index(&btfs);
    // Entry must point at BTF #0, not #1.
    assert_eq!(
        index.get("foo"),
        Some(&FwdIndexEntry {
            btfs_idx: 0,
            type_id: 2,
        }),
        "first-write-wins: BTF #0 wins on duplicate name"
    );
}

/// Anonymous structs (empty resolved name) are silently
/// skipped — the index keys on names, so an anonymous type has
/// nothing to look up.
#[test]
fn build_fwd_index_skips_anonymous_structs() {
    let mut strings = vec![0u8];
    let n_int = push_btf_name(&mut strings, "u64");
    let n_x = push_btf_name(&mut strings, "x");
    let types = vec![
        SynKind::Int {
            name_off: n_int,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        // Anonymous struct (name_off = 0)
        SynKind::Struct {
            name_off: 0,
            size: 8,
            members: vec![SynMember {
                name_off: n_x,
                type_id: 1,
                byte_offset: 0,
            }],
        },
    ];
    let blob = build_btf_full(&types, &strings);
    let btf = Arc::new(Btf::from_bytes(&blob).expect("parse btf"));
    let btfs = vec![btf];
    let index = build_fwd_index(&btfs);
    // Anonymous struct must NOT be indexed under the empty
    // string (would key every anonymous type to the same slot).
    assert!(
        index.is_empty(),
        "anonymous structs must not be indexed: {index:?}"
    );
}

/// Cross-BTF Fwd resolution — `BTF_KIND_FWD` entries
/// must not register in the index even when no complete body
/// shares the name. The renderer's chase consults the index to
/// resolve a Fwd terminal to a complete sibling; a Fwd-keyed
/// entry would point the chase at another Fwd (no body), defeating
/// the index. The function's `match` on the type kind routes Fwd
/// to the `_ => {}` arm, so Fwd never reaches `resolve_name`.
///
/// Fixture: BTF #0 declares `struct shared;` (Fwd) at id 2; BTF #1
/// defines `struct shared { u64 v @ 0 }` at id 2. Expected:
/// `index["shared"] = (1, 2)` — first-write-wins records BTF #1's
/// complete body, not BTF #0's Fwd. Even if BTF order were
/// reversed (Fwd-bearing BTF traversed first), the Fwd would be
/// filtered out so the complete body in the other BTF would still
/// win. This test pins the Fwd-skip behaviour explicitly.
#[test]
fn build_fwd_index_skips_fwd_when_complete_body_in_later_btf() {
    // BTF #0: forward declaration only — `struct shared;` at id 2.
    let mut strings_0 = vec![0u8];
    let n_int_0 = push_btf_name(&mut strings_0, "u64");
    let n_shared_0 = push_btf_name(&mut strings_0, "shared");
    let types_0 = vec![
        // id 1: u64 (filler so id=2 is the Fwd)
        SynKind::Int {
            name_off: n_int_0,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        // id 2: BTF_KIND_FWD `struct shared;` (kind_flag=0 -> struct)
        SynKind::Fwd {
            name_off: n_shared_0,
            kind_flag: 0,
        },
    ];
    let blob_0 = build_btf_full(&types_0, &strings_0);
    let btf_0 = Arc::new(Btf::from_bytes(&blob_0).expect("parse btf 0"));

    // BTF #1: complete `struct shared { u64 v @ 0 }` at id 2 — the
    // body the cross-BTF index keys to.
    let mut strings_1 = vec![0u8];
    let n_int_1 = push_btf_name(&mut strings_1, "u64");
    let n_shared_1 = push_btf_name(&mut strings_1, "shared");
    let n_v_1 = push_btf_name(&mut strings_1, "v");
    let types_1 = vec![
        SynKind::Int {
            name_off: n_int_1,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        SynKind::Struct {
            name_off: n_shared_1,
            size: 8,
            members: vec![SynMember {
                name_off: n_v_1,
                type_id: 1,
                byte_offset: 0,
            }],
        },
    ];
    let blob_1 = build_btf_full(&types_1, &strings_1);
    let btf_1 = Arc::new(Btf::from_bytes(&blob_1).expect("parse btf 1"));

    // Sanity: BTF #0's id 2 must be a Fwd (proves the synthesizer
    // emitted the right kind). Without this, a Struct-encoding bug
    // could pass the assertion below trivially.
    let ty_0_id_2 = btf_0
        .resolve_type_by_id(2)
        .expect("BTF #0 id 2 must resolve");
    assert!(
        matches!(ty_0_id_2, Type::Fwd(_)),
        "BTF #0 id 2 must be Fwd, got {ty_0_id_2:?}"
    );

    let btfs = vec![btf_0, btf_1];
    let index = build_fwd_index(&btfs);
    // The "shared" key must point at BTF #1 (the complete body),
    // not BTF #0 (the Fwd that should be filtered out).
    assert_eq!(
        index.get("shared"),
        Some(&FwdIndexEntry {
            btfs_idx: 1,
            type_id: 2,
        }),
        "Fwd in BTF #0 must not register; complete body in BTF #1 wins: {index:?}"
    );
    // No spurious entries from BTF #0.
    assert_eq!(
        index.len(),
        1,
        "only the BTF #1 complete body should be indexed: {index:?}"
    );
}

/// A `BTF_KIND_FWD` with `name_off = 0` (no name in
/// the strtab) must be silently skipped without panicking the
/// id-space walk. btf-rs registers a name only when `name_offset`
/// returns `Some` (cbtf.rs `btf_type::name_offset` gates on
/// `if offset > 0`, and Fwd is not in `has_anon_name`), so a
/// name_off=0 Fwd has no string-table linkage.
/// `build_fwd_index`'s `match` on `&ty` routes Fwd to the `_ => {}`
/// arm, so `resolve_name` is never called on the empty-named Fwd.
/// This test pins the no-panic guarantee — the `match` already has
/// a `Type::Typedef` arm alongside Struct/Union, and Fwd falls to
/// the `_ => {}` wildcard so `resolve_name` never runs on it.
#[test]
fn build_fwd_index_handles_empty_name_fwd_without_panic() {
    let mut strings = vec![0u8];
    let n_int = push_btf_name(&mut strings, "u64");
    let n_named = push_btf_name(&mut strings, "named");
    let n_x = push_btf_name(&mut strings, "x");
    let types = vec![
        // id 1: u64
        SynKind::Int {
            name_off: n_int,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        // id 2: BTF_KIND_FWD with name_off=0 (no strtab linkage).
        // btf-rs's `btf_type::name_offset` (cbtf.rs) returns `None`
        // here because of the `if offset > 0` gate (Fwd is not in
        // `has_anon_name`).
        SynKind::Fwd {
            name_off: 0,
            kind_flag: 0,
        },
        // id 3: a complete named struct so the index has at
        // least one positive entry, proving the walk continued
        // past the empty-named Fwd at id 2 instead of aborting.
        SynKind::Struct {
            name_off: n_named,
            size: 8,
            members: vec![SynMember {
                name_off: n_x,
                type_id: 1,
                byte_offset: 0,
            }],
        },
    ];
    let blob = build_btf_full(&types, &strings);
    let btf = Arc::new(Btf::from_bytes(&blob).expect("parse btf"));

    // Sanity: BTF id 2 must be a Fwd (synthesizer encoded the
    // kind correctly).
    let ty_id_2 = btf.resolve_type_by_id(2).expect("BTF id 2 must resolve");
    assert!(
        matches!(ty_id_2, Type::Fwd(_)),
        "BTF id 2 must be Fwd, got {ty_id_2:?}"
    );

    let btfs = vec![btf];
    // The walk must not panic even though id 2's Fwd has no
    // strtab name. Filter behaviour: Fwd kind never reaches
    // `resolve_name`, so the empty-name path is structurally
    // unreachable for the production filter.
    let index = build_fwd_index(&btfs);
    // The empty-named Fwd at id 2 must NOT be present, neither
    // under "" (anonymous) nor under any other key.
    assert!(
        !index.contains_key(""),
        "empty-string key must not appear (anonymous Fwd): {index:?}"
    );
    // The named struct at id 3 must be indexed — proves the walk
    // continued past the Fwd at id 2 rather than terminating.
    assert_eq!(
        index.get("named"),
        Some(&FwdIndexEntry {
            btfs_idx: 0,
            type_id: 3,
        }),
        "named struct at id 3 must register after the empty-named Fwd at id 2: {index:?}"
    );
    assert_eq!(
        index.len(),
        1,
        "only the named struct should be indexed: {index:?}"
    );
}

/// Two-object end-to-end: object A's BTF declares
/// `struct cgx_target;` (a `BTF_KIND_FWD`) and references it
/// via a Ptr field; object B's BTF carries the full body
/// `struct cgx_target { u64 marker @ 0 }`. The cross-BTF index
/// produced by [`build_cast_analysis_from_bytes`] indexes
/// `cgx_target -> (1, 2)` — BTF #1 (object B) at type id 2,
/// the body location.
///
/// Mirrors the deferred-resolve arena cast target shape: a
/// `__arena u64` declared in object A whose true type is the
/// `cgx_target` body in object B. The renderer's chase then
/// resolves the Fwd through the cross-BTF index and renders
/// the payload.
#[test]
fn build_cast_analysis_indexes_cross_object_struct_body() {
    // Object A: declares `struct cgx_target;` as a Fwd at id
    // 2, used as a pointee. The Fwd has no body — just the
    // forward declaration.
    let mut strings_a = vec![0u8];
    let n_int_a = push_btf_name(&mut strings_a, "u64");
    let n_cgx_a = push_btf_name(&mut strings_a, "cgx_target");
    let n_t_a = push_btf_name(&mut strings_a, "outer_a");
    let n_field_a = push_btf_name(&mut strings_a, "ptr_to_target");
    let n_func_a = push_btf_name(&mut strings_a, "func_a");
    let n_text_a = push_btf_name(&mut strings_a, ".text");
    // This test omits a `cgx_target` Fwd in object A entirely;
    // the cross-BTF index lookup is exercised purely via object
    // B's complete body. `build_fwd_index_skips_fwd_when_complete_body_in_later_btf`
    // covers the Fwd-in-A + body-in-B shape directly with the
    // `SynKind::Fwd` synthesizer. The `outer_a` struct here just
    // exists so `analyze_one_object_with_btf` has a non-empty
    // type table to traverse.
    let types_a = vec![
        // id 1: u64
        SynKind::Int {
            name_off: n_int_a,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        // id 2: outer_a (carries a u64 field; just there so
        // we have any struct at all in this BTF — the
        // cross-BTF assertion is on object B's body)
        SynKind::Struct {
            name_off: n_t_a,
            size: 8,
            members: vec![SynMember {
                name_off: n_field_a,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        // id 3: FuncProto returning void with one u64 param
        SynKind::FuncProto {
            return_type_id: 0,
            params: vec![SynParam {
                name_off: 0,
                type_id: 1,
            }],
        },
        // id 4: Func
        SynKind::Func {
            name_off: n_func_a,
            type_id: 3,
            linkage: 1,
        },
    ];
    let _ = n_cgx_a; // unused: this test omits a `cgx_target` Fwd
    let btf_blob_a = build_btf_full(&types_a, &strings_a);
    let insns_a = vec![exit_insn()];
    let text_a = insns_to_text_bytes(&insns_a);
    let btf_ext_a = build_btf_ext(n_text_a, &[(0, 3)], 8);
    let inner_a = build_full_bpf_object_elf(text_a, btf_blob_a, btf_ext_a);

    // Object B: defines `struct cgx_target { u64 marker @ 0 }`
    // as a complete struct at id 2.
    let mut strings_b = vec![0u8];
    let n_int_b = push_btf_name(&mut strings_b, "u64");
    let n_cgx_b = push_btf_name(&mut strings_b, "cgx_target");
    let n_marker_b = push_btf_name(&mut strings_b, "marker");
    let n_func_b = push_btf_name(&mut strings_b, "func_b");
    let n_text_b = push_btf_name(&mut strings_b, ".text");
    let types_b = vec![
        // id 1: u64
        SynKind::Int {
            name_off: n_int_b,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        // id 2: struct cgx_target { u64 marker @ 0 }  -- THE
        // BODY the cross-BTF index keys to.
        SynKind::Struct {
            name_off: n_cgx_b,
            size: 8,
            members: vec![SynMember {
                name_off: n_marker_b,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        SynKind::FuncProto {
            return_type_id: 0,
            params: vec![SynParam {
                name_off: 0,
                type_id: 1,
            }],
        },
        SynKind::Func {
            name_off: n_func_b,
            type_id: 3,
            linkage: 1,
        },
    ];
    let btf_blob_b = build_btf_full(&types_b, &strings_b);
    let insns_b = vec![exit_insn()];
    let text_b = insns_to_text_bytes(&insns_b);
    let btf_ext_b = build_btf_ext(n_text_b, &[(0, 3)], 8);
    let inner_b = build_full_bpf_object_elf(text_b, btf_blob_b, btf_ext_b);

    // Outer ELF wraps both inner objects in `.bpf.objs` via
    // STT_OBJECT symbols so [`iter_embedded_bpf_objects`]
    // yields them as separate slices.
    let strtab = b"\0obj_a\0obj_b\0".to_vec();
    let mut symtab = Vec::new();
    symtab.extend_from_slice(&elf64_sym(0, 0, 0, 0, 0));
    // sym for object A: name_off = 1 (b"obj_a"), st_value = 0,
    // size = inner_a.len()
    symtab.extend_from_slice(&elf64_sym(
        1,
        st_info(syms::STB_GLOBAL, syms::STT_OBJECT),
        1, // st_shndx — .bpf.objs at shdr[1]
        0,
        inner_a.len() as u64,
    ));
    // sym for object B: name_off = 7 (b"obj_b"),
    // st_value = inner_a.len(), size = inner_b.len()
    symtab.extend_from_slice(&elf64_sym(
        7,
        st_info(syms::STB_GLOBAL, syms::STT_OBJECT),
        1,
        inner_a.len() as u64,
        inner_b.len() as u64,
    ));

    // Pack both inner objects back-to-back in `.bpf.objs`.
    let mut bpf_objs_data = Vec::new();
    bpf_objs_data.extend_from_slice(&inner_a);
    bpf_objs_data.extend_from_slice(&inner_b);

    let outer = build_elf64(
        vec![
            SecSpec::new(".bpf.objs", sh::SHT_PROGBITS).data(bpf_objs_data),
            SecSpec::new(".strtab", sh::SHT_STRTAB).data(strtab),
            SecSpec::new(".symtab", sh::SHT_SYMTAB)
                .data(symtab)
                .link(2)
                .entsize(24),
        ],
        h::EM_X86_64,
        h::ET_REL,
    );

    let out = build_cast_analysis_from_bytes(&outer);
    // Both BTFs parsed.
    assert_eq!(
        out.btfs.len(),
        2,
        "both embedded objects' BTFs must be retained: {}",
        out.btfs.len()
    );
    // The cross-BTF index has cgx_target keyed at the FIRST
    // BTF that carries a complete body. Object A in this
    // fixture exposes no `cgx_target` struct (this test omits
    // the Fwd entirely), so object B's id is what gets indexed.
    let cgx_hit = out.fwd_index.get("cgx_target");
    assert_eq!(
        cgx_hit,
        Some(&FwdIndexEntry {
            btfs_idx: 1,
            type_id: 2,
        }),
        "cross-BTF index must point cgx_target to BTF #1 at type id 2: {:?}",
        out.fwd_index
    );
    // Both objects' top-level structs are also indexed.
    assert_eq!(
        out.fwd_index.get("outer_a"),
        Some(&FwdIndexEntry {
            btfs_idx: 0,
            type_id: 2,
        }),
        "object A's struct outer_a must be indexed in BTF #0 at id 2"
    );
}

// ----- LazyCastMap full-output accessor -------------------------

/// `LazyCastMap::new(None).get_full()` returns `None` without
/// touching the filesystem or the process-wide cache. Matches
/// the no-scheduler dump-path contract (every `u64` renders as
/// a plain counter) for the production [`Self::get_full`]
/// accessor that returns the full [`CastAnalysisOutput`]
/// including the cross-BTF Fwd index.
#[test]
fn lazy_cast_map_get_full_returns_none_when_no_scheduler() {
    let lazy = LazyCastMap::new(None);
    assert!(
        lazy.get_full().is_none(),
        "no-scheduler builder must short-circuit `.get_full()` to None",
    );
}

// ----- cached_cast_analysis_for_scheduler concurrency -----------

/// Multi-thread race on the same scheduler binary path: every
/// caller must observe the same `Arc<CastAnalysisOutput>` —
/// pointer-equal — proving the per-hash `OnceLock` inside the
/// process-wide cache deduplicates concurrent first-callers
/// rather than running the analyzer once per caller and
/// returning equivalent-but-distinct Arcs.
///
/// Uses [`std::thread::scope`] so the threads can borrow the
/// path; an [`Arc<std::sync::Barrier>`] coordinates the
/// release point so every thread enters
/// [`cached_cast_analysis_for_scheduler`] within microseconds
/// of one another, maximising the contention on the cache's
/// `Mutex<HashMap>` lookup AND the per-hash
/// `OnceLock::get_or_init` serialisation. Without the barrier
/// the threads might serialise naturally on creation, missing
/// the concurrent-init regression the
/// `Arc<OnceLock<...>>` shape exists to catch.
#[test]
fn cached_cast_analysis_concurrent_callers_share_one_oncelock_init() {
    use std::sync::{Arc as StdArc, Barrier};

    // Build the standard arena-cast end-to-end fixture and
    // write it to a fresh path so the content hash is unique
    // to this test run (won't collide with other tests'
    // cache entries).
    let blob = build_recovers_arena_cast_outer_elf();
    let dir = tempfile::tempdir().expect("tempdir");
    let p = dir.path().join("concurrent.bin");
    std::fs::write(&p, &blob).expect("write");

    const N_THREADS: usize = 8;
    let barrier = StdArc::new(Barrier::new(N_THREADS));
    let path = p.clone();
    let results: Vec<Arc<CastAnalysisOutput>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..N_THREADS)
            .map(|_| {
                let barrier = barrier.clone();
                let path = path.clone();
                s.spawn(move || {
                    // Synchronise the release: every thread
                    // hits `wait()` before any thread enters
                    // the cache lookup.
                    barrier.wait();
                    cached_cast_analysis_for_scheduler(&path)
                        .expect("non-empty fixture must produce Some")
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    assert_eq!(results.len(), N_THREADS);
    // Every Arc must be pointer-equal to the first — proves
    // the OnceLock dedup fired and only one analysis ran
    // across all N concurrent callers.
    let first = &results[0];
    for (i, other) in results.iter().enumerate().skip(1) {
        assert!(
            Arc::ptr_eq(first, other),
            "thread {i}: Arc must be pointer-equal to thread 0's; \
                 OnceLock dedup did NOT fire across concurrent callers",
        );
    }
}

/// `MOV r1, imm` directly before the call → recovered.
#[test]
fn recover_alloc_size_adjacent_mov_returns_imm() {
    let text = vec![BpfInsn::new(MOV_R1_CODE, 1, 0, 0, 4096), call_insn()];
    assert_eq!(
        super::super::recover_alloc_size_from_r1(&text, 1),
        Some(4096)
    );
}

/// A BPF_CALL between the MOV and the target call clobbers r1-r5, so
/// the earlier immediate is stale → None (pre-fix returned 4096).
#[test]
fn recover_alloc_size_stops_at_call_clobber_returns_none() {
    let text = vec![
        BpfInsn::new(MOV_R1_CODE, 1, 0, 0, 4096),
        call_insn(), // helper call clobbers r1
        call_insn(), // target static-alloc call
    ];
    assert_eq!(super::super::recover_alloc_size_from_r1(&text, 2), None);
}

/// An ALU64 write to r1 between the MOV and the call invalidates the
/// immediate → None (pre-fix returned the stale 4096).
#[test]
fn recover_alloc_size_stops_at_alu_clobber_returns_none() {
    let text = vec![
        BpfInsn::new(MOV_R1_CODE, 1, 0, 0, 4096),
        // BPF_ALU64 | BPF_ADD | BPF_K (0x07), dst r1: r1 += 0.
        BpfInsn::new(0x07, 1, 0, 0, 0),
        call_insn(),
    ];
    assert_eq!(super::super::recover_alloc_size_from_r1(&text, 2), None);
}

/// An atomic XCHG with src==r1 (`r1 = xchg(*dst, r1)`) writes r1 even
/// though it is class BPF_STX with dst==base, so the scan must stop
/// (None), not return the stale MOV. A dst-only writer check misses
/// this; kernel check_atomic_rmw confirms the src-register write.
#[test]
fn recover_alloc_size_stops_at_atomic_fetch_clobber_returns_none() {
    // BPF_STX(0x03) | BPF_DW(0x18) | BPF_ATOMIC(0xc0) = 0xdb;
    // imm BPF_XCHG = 0xe1 (0xe0 | BPF_FETCH). dst=base (r10/fp), src=r1.
    let text = vec![
        BpfInsn::new(MOV_R1_CODE, 1, 0, 0, 4096),
        BpfInsn::new(0xdb, 10, 1, -8, 0xe1),
        call_insn(),
    ];
    assert_eq!(super::super::recover_alloc_size_from_r1(&text, 2), None);
}

/// No MOV r1 in the window (a MOV to r2 does not write r1) → None.
#[test]
fn recover_alloc_size_no_mov_returns_none() {
    let text = vec![
        BpfInsn::new(MOV_R1_CODE, 2, 0, 0, 4096), // MOV r2, not r1
        call_insn(),
    ];
    assert_eq!(super::super::recover_alloc_size_from_r1(&text, 1), None);
}

/// call_pc == 0 has no predecessors → None.
#[test]
fn recover_alloc_size_call_pc_zero_returns_none() {
    let text = vec![BpfInsn::new(MOV_R1_CODE, 1, 0, 0, 4096)];
    assert_eq!(super::super::recover_alloc_size_from_r1(&text, 0), None);
}

/// A MOV exactly ALLOC_SIZE_LOOKBACK instructions back is in-window
/// (found); one further back is out-of-window (None).
#[test]
fn recover_alloc_size_lookback_window_boundary() {
    let lb = super::super::ALLOC_SIZE_LOOKBACK;
    let mov_r2 = || BpfInsn::new(MOV_R1_CODE, 2, 0, 0, 0); // non-r1 filler

    // In-window: MOV r1 @0, (lb-1) fillers, call @lb → found at distance lb.
    let mut text = vec![BpfInsn::new(MOV_R1_CODE, 1, 0, 0, 4096)];
    for _ in 0..(lb - 1) {
        text.push(mov_r2());
    }
    text.push(call_insn());
    assert_eq!(
        super::super::recover_alloc_size_from_r1(&text, lb),
        Some(4096)
    );

    // Out-of-window: one extra filler pushes the MOV to distance lb+1.
    let mut text = vec![BpfInsn::new(MOV_R1_CODE, 1, 0, 0, 4096)];
    for _ in 0..lb {
        text.push(mov_r2());
    }
    text.push(call_insn());
    assert_eq!(
        super::super::recover_alloc_size_from_r1(&text, lb + 1),
        None
    );
}
