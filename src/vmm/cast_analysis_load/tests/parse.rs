use super::super::*;
use super::*;
use crate::monitor::cast_analysis::{AddrSpace, CastHit};
use goblin::elf::header as h;
use goblin::elf::section_header as sh;
use goblin::elf::sym as syms;

// ----- Tests for cached_cast_analysis_for_scheduler error paths --

/// 1. Path that does not exist on the filesystem: the
///    `std::fs::read` arm fires, returns `None`.
#[test]
fn cached_cast_analysis_nonexistent_path_returns_none() {
    let p = std::path::Path::new("/tmp/ktstr-cast-analysis-nonexistent-fixture-path-do-not-create");
    // Sanity: ensure the path really does not exist so the
    // assertion below proves what it claims.
    assert!(
        !p.exists(),
        "fixture path must not exist; remove it before running this test"
    );
    assert!(cached_cast_analysis_for_scheduler(p).is_none());
}

/// 2. Empty file: `goblin::elf::Elf::parse` rejects a 0-byte
///    input; the parse arm fires; empty result collapses to `None`.
#[test]
fn cached_cast_analysis_empty_file_returns_none() {
    let dir = tempfile::tempdir().expect("tempdir");
    let p = dir.path().join("empty.bin");
    std::fs::write(&p, b"").expect("write empty file");
    assert!(cached_cast_analysis_for_scheduler(&p).is_none());
}

/// 3. Valid ELF without a `.bpf.objs` section: the section-lookup
///    arm fires, no analysis happens; empty result collapses to
///    `None`.
#[test]
fn cached_cast_analysis_no_bpf_objs_section_returns_none() {
    let blob = build_elf64(
        vec![SecSpec::new(".text", sh::SHT_PROGBITS).flags(sh::SHF_EXECINSTR.into())],
        h::EM_X86_64,
        h::ET_REL,
    );
    let dir = tempfile::tempdir().expect("tempdir");
    let p = dir.path().join("no_bpf_objs.elf");
    std::fs::write(&p, &blob).expect("write");
    assert!(cached_cast_analysis_for_scheduler(&p).is_none());
}

// ----- Tests for btf_str_at --------------------------------------

/// 4. Empty `btf_bytes`: hits the `< 24` header-length gate.
#[test]
fn btf_str_at_empty_returns_none() {
    assert!(btf_str_at(&[], 0).is_none());
    assert!(btf_str_at(&[0u8; 23], 0).is_none());
}

/// 5. `str_off` past `str_section_len`: the `off >= str_section_len`
///    gate fires.
#[test]
fn btf_str_at_offset_past_strtab_returns_none() {
    // strings: 6 bytes ("\0abc\0\0"); offset 100 is far past.
    let strings = b"\0abc\0\0";
    let blob = build_btf_blob(&[], strings);
    assert!(btf_str_at(&blob, 100).is_none());
}

/// 6. `str_off` exactly at the strtab boundary (= len): the
///    `>=` gate rejects it.
#[test]
fn btf_str_at_offset_at_boundary_returns_none() {
    let strings = b"\0abc\0";
    let blob = build_btf_blob(&[], strings);
    assert!(btf_str_at(&blob, strings.len() as u32).is_none());
}

/// 7. No null terminator in the slice from `base..strtab_end`:
///    the function returns the whole tail as a string. Use a
///    payload that ends without a `\0` to hit the `unwrap_or`
///    branch — the result is still valid UTF-8, exercising the
///    "no null terminator within bounds" path that produces a
///    string instead of `None`. The closer case for `None` is
///    invalid UTF-8 bytes; emit those to confirm `from_utf8`
///    rejection.
#[test]
fn btf_str_at_no_null_terminator_invalid_utf8_returns_none() {
    // Strings: 0xff is not valid UTF-8 as a leading byte and
    // there is no trailing `\0` — `from_utf8` rejects, function
    // returns None.
    let strings = vec![0u8, 0xff, 0xff];
    let blob = build_btf_blob(&[], &strings);
    // str_off=1 points to the first 0xff byte; the slice
    // [base..strtab_end] is `[0xff, 0xff]` (no null), so the
    // `from_utf8` call rejects.
    assert!(btf_str_at(&blob, 1).is_none());
}

/// 8. Valid lookup: returns the expected string.
#[test]
fn btf_str_at_valid_returns_string() {
    let strings = b"\0hello\0world\0";
    let blob = build_btf_blob(&[], strings);
    // Offset 1 = "hello"; offset 7 = "world".
    assert_eq!(btf_str_at(&blob, 1), Some("hello"));
    assert_eq!(btf_str_at(&blob, 7), Some("world"));
    // Offset 0 is the empty string.
    assert_eq!(btf_str_at(&blob, 0), Some(""));
}

// ----- Tests for parse_btf_ext_func_entries ----------------------

/// 9. Data shorter than the minimum 24-byte `.BTF.ext` header:
///    the length gate fires.
#[test]
fn parse_btf_ext_too_short_returns_empty() {
    let btf_bytes = build_btf_blob(&[], b"\0");
    // Build a minimal inner ELF so we can pass &elf to the
    // function (even though we never reach the section walk).
    let blob = build_elf64(vec![], h::EM_BPF, h::ET_REL);
    let elf = goblin::elf::Elf::parse(&blob).unwrap();
    let bases = HashMap::new();
    for short_len in [0usize, 23] {
        let data = vec![0u8; short_len];
        let out = parse_btf_ext_func_entries(&data, &btf_bytes, &elf, &bases);
        assert!(out.is_empty(), "len={short_len}");
    }
}

/// 10. Wrong magic: the magic check fires.
#[test]
fn parse_btf_ext_wrong_magic_returns_empty() {
    let mut data = vec![0u8; 24];
    // Magic = 0xDEAD (not 0xEB9F).
    data[0..2].copy_from_slice(&0xDEADu16.to_le_bytes());
    let btf_bytes = build_btf_blob(&[], b"\0");
    let blob = build_elf64(vec![], h::EM_BPF, h::ET_REL);
    let elf = goblin::elf::Elf::parse(&blob).unwrap();
    let bases = HashMap::new();
    let out = parse_btf_ext_func_entries(&data, &btf_bytes, &elf, &bases);
    assert!(out.is_empty());
}

/// 11. `hdr_len` below the 24-byte minimum, and `hdr_len` past
///     `data.len()`: both fire the `hdr_len < MIN || hdr_len >
///     data.len()` gate.
#[test]
fn parse_btf_ext_bad_hdr_len_returns_empty() {
    let btf_bytes = build_btf_blob(&[], b"\0");
    let blob = build_elf64(vec![], h::EM_BPF, h::ET_REL);
    let elf = goblin::elf::Elf::parse(&blob).unwrap();
    let bases = HashMap::new();

    // (a) hdr_len = 16 (< 24).
    let mut data = vec![0u8; 24];
    data[0..2].copy_from_slice(&0xEB9F_u16.to_le_bytes());
    data[4..8].copy_from_slice(&16u32.to_le_bytes());
    let out = parse_btf_ext_func_entries(&data, &btf_bytes, &elf, &bases);
    assert!(out.is_empty(), "hdr_len=16 should be rejected");

    // (b) hdr_len = 1024 (> data.len()).
    let mut data = vec![0u8; 24];
    data[0..2].copy_from_slice(&0xEB9F_u16.to_le_bytes());
    data[4..8].copy_from_slice(&1024u32.to_le_bytes());
    let out = parse_btf_ext_func_entries(&data, &btf_bytes, &elf, &bases);
    assert!(out.is_empty(), "hdr_len > data.len should be rejected");
}

/// 12. `func_info_off` + `func_info_len` overflows `data.len()`:
///     the `info_end > data.len()` gate fires.
#[test]
fn parse_btf_ext_func_info_window_oob_returns_empty() {
    let btf_bytes = build_btf_blob(&[], b"\0");
    let blob = build_elf64(vec![], h::EM_BPF, h::ET_REL);
    let elf = goblin::elf::Elf::parse(&blob).unwrap();
    let bases = HashMap::new();

    // hdr_len=24, func_info_off=0, func_info_len=10_000;
    // info window runs 24..10024 but data is only 32 bytes.
    let mut data = vec![0u8; 32];
    data[0..2].copy_from_slice(&0xEB9F_u16.to_le_bytes());
    data[4..8].copy_from_slice(&24u32.to_le_bytes()); // hdr_len
    data[8..12].copy_from_slice(&0u32.to_le_bytes()); // func_info_off
    data[12..16].copy_from_slice(&10_000u32.to_le_bytes()); // func_info_len
    let out = parse_btf_ext_func_entries(&data, &btf_bytes, &elf, &bases);
    assert!(out.is_empty());
}

/// 13. `record_size` < 8: the analyzer requires at least an
///     8-byte `bpf_func_info_min`. Smaller records are rejected.
#[test]
fn parse_btf_ext_record_size_too_small_returns_empty() {
    let btf_bytes = build_btf_blob(&[], b"\0");
    let blob = build_elf64(vec![], h::EM_BPF, h::ET_REL);
    let elf = goblin::elf::Elf::parse(&blob).unwrap();
    let bases = HashMap::new();

    // hdr_len=24, func_info_off=0, func_info_len=4 (just the
    // record_size field). record_size=4 < 8 → reject.
    let mut data = vec![0u8; 32];
    data[0..2].copy_from_slice(&0xEB9F_u16.to_le_bytes());
    data[4..8].copy_from_slice(&24u32.to_le_bytes()); // hdr_len
    data[8..12].copy_from_slice(&0u32.to_le_bytes()); // func_info_off
    data[12..16].copy_from_slice(&8u32.to_le_bytes()); // func_info_len
    // info section starts at offset 24 (hdr_len). Place a
    // record_size of 4 there.
    data[24..28].copy_from_slice(&4u32.to_le_bytes());
    let out = parse_btf_ext_func_entries(&data, &btf_bytes, &elf, &bases);
    assert!(out.is_empty());
}

/// 14. Record with `insn_off` not a multiple of 8: the entry
///     is silently skipped rather than producing a bogus
///     [`FuncEntry`].
///
/// Builds a full valid `.BTF.ext` with one section name pointing
/// at a `.text` PROGBITS+EXECINSTR section, two records — one
/// with `insn_off=8` (valid, kept) and one with `insn_off=12`
/// (not multiple of 8, dropped). Verifies the kept entry has
/// the expected `insn_offset` and the malformed one is absent.
#[test]
fn parse_btf_ext_non_multiple_insn_off_skips_entry() {
    // Build BTF strings with a "txt" entry at offset 1.
    let bytes_strs = b"\0txt\0";
    let btf_bytes = build_btf_blob(&[], bytes_strs);

    // Build inner ELF with a .text section so find_section can
    // resolve "txt"... but the BTF strtab name "txt" must match
    // the ELF section name. So name the section "txt".
    let inner = build_elf64(
        vec![SecSpec::new("txt", sh::SHT_PROGBITS).flags(sh::SHF_EXECINSTR.into())],
        h::EM_BPF,
        h::ET_REL,
    );
    let elf = goblin::elf::Elf::parse(&inner).unwrap();
    // The user section "txt" is shdr index 1 (0 is NULL).
    let mut bases: HashMap<u32, usize> = HashMap::new();
    bases.insert(1, 0);

    // Build the .BTF.ext payload:
    //   header (24 bytes): magic, ver, flags, hdr_len=24,
    //     func_info_off=0, func_info_len=24,
    //     line_info_off=24, line_info_len=0.
    //   info (24 bytes): record_size=8 + 1 sec hdr (8 bytes,
    //     sec_name_off=1 ("txt"), num_info=2) + 2 records of
    //     8 bytes each = 4 + 8 + 16 = 28? Let me recompute:
    //     record_size(4) + sec_hdr(8) + 2*8(16) = 28 bytes.
    // We need func_info_len = 28 then.
    let mut data = Vec::new();
    data.extend_from_slice(&0xEB9F_u16.to_le_bytes()); // magic
    data.push(1); // version
    data.push(0); // flags
    data.extend_from_slice(&24u32.to_le_bytes()); // hdr_len
    data.extend_from_slice(&0u32.to_le_bytes()); // func_info_off
    data.extend_from_slice(&28u32.to_le_bytes()); // func_info_len
    data.extend_from_slice(&28u32.to_le_bytes()); // line_info_off
    data.extend_from_slice(&0u32.to_le_bytes()); // line_info_len
    // func_info data:
    data.extend_from_slice(&8u32.to_le_bytes()); // record_size = 8
    data.extend_from_slice(&1u32.to_le_bytes()); // sec_name_off = "txt"
    data.extend_from_slice(&2u32.to_le_bytes()); // num_info = 2
    // record 0: insn_off=8 (valid; instruction index = 8/8 = 1)
    data.extend_from_slice(&8u32.to_le_bytes());
    data.extend_from_slice(&42u32.to_le_bytes()); // type_id = 42
    // record 1: insn_off=12 (NOT multiple of 8; skipped)
    data.extend_from_slice(&12u32.to_le_bytes());
    data.extend_from_slice(&99u32.to_le_bytes()); // type_id = 99
    let out = parse_btf_ext_func_entries(&data, &btf_bytes, &elf, &bases);
    // Only the insn_off=8 entry should land.
    assert_eq!(out.len(), 1, "got {out:?}");
    assert_eq!(out[0].insn_offset, 1);
    assert_eq!(out[0].func_proto_id, 42);
}

// ----- Tests for iter_embedded_bpf_objects -----------------------

/// 15. No `STT_OBJECT` symbols pointing into `.bpf.objs`: the
///     fallback branch fires and returns one slice covering the
///     entire section.
#[test]
fn iter_embedded_bpf_objects_no_symbols_falls_back_to_full_section() {
    // Build a scheduler-like ELF: one `.bpf.objs` section, no
    // symbol table at all.
    let payload = b"DUMMY_BPF_OBJ_BYTES".to_vec();
    let payload_len = payload.len();
    let blob = build_elf64(
        vec![SecSpec::new(".bpf.objs", sh::SHT_PROGBITS).data(payload)],
        h::EM_X86_64,
        h::ET_REL,
    );
    let elf = goblin::elf::Elf::parse(&blob).unwrap();
    // `.bpf.objs` is at section index 1 (0 = NULL).
    let bpf_objs_idx = find_section(&elf, ".bpf.objs").expect(".bpf.objs");
    let out = iter_embedded_bpf_objects(&elf, &blob, bpf_objs_idx);
    assert_eq!(out.len(), 1, "expected one fallback slice");
    assert_eq!(out[0].len(), payload_len);
    assert_eq!(out[0], b"DUMMY_BPF_OBJ_BYTES");
}

// ----- Tests for section_data ------------------------------------

/// 16. Section header with `sh_offset + sh_size` overflowing
///     `usize`: `checked_add` returns `None`, function returns
///     `None`.
///
/// Building this through the normal builder is impossible
/// (it always sets a real offset). Instead, we manually patch
/// the section header bytes after construction to set
/// `sh_offset=u64::MAX` and `sh_size=u64::MAX`. Goblin still
/// parses the header successfully; `section_data` then triggers
/// the overflow path.
#[test]
fn section_data_overflow_returns_none() {
    let payload = b"PAYLOAD".to_vec();
    let mut blob = build_elf64(
        vec![SecSpec::new(".x", sh::SHT_PROGBITS).data(payload)],
        h::EM_X86_64,
        h::ET_REL,
    );
    // Patch shdr[1] (".x") sh_offset and sh_size to u64::MAX so
    // the `start.checked_add(size)` overflows. shdr table is at
    // the end of the file; each shdr is 64 bytes; shdr[0] is
    // NULL, so shdr[1] starts at e_shoff+64.
    let elf_view = goblin::elf::Elf::parse(&blob).unwrap();
    let shoff = elf_view.header.e_shoff as usize;
    let shdr1_off = shoff + 64;
    // sh_offset is at byte 24 within the 64-byte ELF64 shdr;
    // sh_size is at byte 32.
    blob[shdr1_off + 24..shdr1_off + 32].copy_from_slice(&u64::MAX.to_le_bytes());
    blob[shdr1_off + 32..shdr1_off + 40].copy_from_slice(&u64::MAX.to_le_bytes());

    let elf = goblin::elf::Elf::parse(&blob).unwrap();
    let idx = find_section(&elf, ".x").expect(".x");
    assert!(section_data(&elf, &blob, idx).is_none());
}

/// Sanity: the unused-helper escape valves (`elf64_sym`,
/// `st_info`) are exercised by a smoke build of a symbol table
/// to keep them from rotting if a future test wants them. The
/// goblin parser must accept the symtab/strtab pair.
#[test]
fn smoke_symtab_helpers_compile() {
    // Build .strtab content: "\0bpf_obj\0".
    let strtab = b"\0bpf_obj\0".to_vec();
    // Single STT_OBJECT symbol named "bpf_obj" pointing at
    // the (theoretical) `.bpf.objs` section index 1.
    let mut symtab = Vec::new();
    // shdr[0] = NULL — the first entry of a symtab is reserved.
    symtab.extend_from_slice(&elf64_sym(0, 0, 0, 0, 0));
    symtab.extend_from_slice(&elf64_sym(
        1, // st_name: offset of "bpf_obj" in .strtab
        st_info(syms::STB_GLOBAL, syms::STT_OBJECT),
        1, // st_shndx
        0, // st_value
        8, // st_size
    ));

    let blob = build_elf64(
        vec![
            SecSpec::new(".bpf.objs", sh::SHT_PROGBITS).data(vec![0u8; 8]),
            SecSpec::new(".strtab", sh::SHT_STRTAB).data(strtab),
            SecSpec::new(".symtab", sh::SHT_SYMTAB)
                .data(symtab)
                .link(2) // sh_link → strtab is user-section index 2 = shdr index 3? wait
                .entsize(24),
        ],
        h::EM_X86_64,
        h::ET_REL,
    );
    // sh_link must reference the actual shdr index of the
    // strtab. shdr[0]=NULL, [1]=.bpf.objs, [2]=.strtab,
    // [3]=.symtab, [4]=.shstrtab. So sh_link should be 2.
    // We passed link(2) above, which matches.
    let _ = goblin::elf::Elf::parse(&blob).expect("parse");
    // The parser-level smoke completed; nothing further to
    // assert here — this test exists so the helpers stay in
    // active use.
}

// ----- Tests for find_section ------------------------------------

/// Happy path: `find_section` resolves an existing section by
/// name and returns the matching shdr index.
#[test]
fn find_section_locates_named_section() {
    let blob = build_elf64(
        vec![
            SecSpec::new(".text", sh::SHT_PROGBITS).flags(sh::SHF_EXECINSTR.into()),
            SecSpec::new(".bpf.objs", sh::SHT_PROGBITS).data(vec![0u8; 4]),
        ],
        h::EM_BPF,
        h::ET_REL,
    );
    let elf = goblin::elf::Elf::parse(&blob).unwrap();
    // shdr[0]=NULL, [1]=.text, [2]=.bpf.objs, [3]=.shstrtab.
    assert_eq!(find_section(&elf, ".text"), Some(1));
    assert_eq!(find_section(&elf, ".bpf.objs"), Some(2));
}

/// `find_section` returns `None` for a name that does not match
/// any section.
#[test]
fn find_section_missing_returns_none() {
    let blob = build_elf64(
        vec![SecSpec::new(".text", sh::SHT_PROGBITS).flags(sh::SHF_EXECINSTR.into())],
        h::EM_BPF,
        h::ET_REL,
    );
    let elf = goblin::elf::Elf::parse(&blob).unwrap();
    assert_eq!(find_section(&elf, ".nope"), None);
}

// ----- Tests for section_data happy path -------------------------

/// `section_data` returns the byte slice covering a known
/// section's `[sh_offset, sh_offset + sh_size)` range.
#[test]
fn section_data_returns_section_bytes() {
    let payload = b"section-bytes-payload-12345".to_vec();
    let payload_len = payload.len();
    let blob = build_elf64(
        vec![SecSpec::new(".x", sh::SHT_PROGBITS).data(payload)],
        h::EM_BPF,
        h::ET_REL,
    );
    let elf = goblin::elf::Elf::parse(&blob).unwrap();
    let idx = find_section(&elf, ".x").unwrap();
    let bytes = section_data(&elf, &blob, idx).expect("payload slice");
    assert_eq!(bytes.len(), payload_len);
    assert_eq!(bytes, &b"section-bytes-payload-12345"[..]);
}

/// Out-of-range section index returns `None`.
#[test]
fn section_data_out_of_range_returns_none() {
    let blob = build_elf64(
        vec![SecSpec::new(".text", sh::SHT_PROGBITS)],
        h::EM_BPF,
        h::ET_REL,
    );
    let elf = goblin::elf::Elf::parse(&blob).unwrap();
    assert!(section_data(&elf, &blob, 9999).is_none());
}

// ----- iter_embedded_bpf_objects symbol-driven path --------------

/// Symbol-driven path: a single `STT_OBJECT` symbol pointing
/// into `.bpf.objs` produces one slice covering exactly the
/// range `[st_value, st_value + st_size)`.
#[test]
fn iter_embedded_bpf_objects_uses_object_symbol() {
    let payload: Vec<u8> = (0..32u8).collect();
    let strtab = b"\0bpf_obj\0".to_vec();
    let mut symtab = Vec::new();
    symtab.extend_from_slice(&elf64_sym(0, 0, 0, 0, 0));
    symtab.extend_from_slice(&elf64_sym(
        1,
        st_info(syms::STB_GLOBAL, syms::STT_OBJECT),
        1,  // st_shndx — .bpf.objs at shdr[1]
        4,  // st_value: byte offset within .bpf.objs (sh_addr=0)
        24, // st_size
    ));
    let blob = build_elf64(
        vec![
            SecSpec::new(".bpf.objs", sh::SHT_PROGBITS).data(payload),
            SecSpec::new(".strtab", sh::SHT_STRTAB).data(strtab),
            SecSpec::new(".symtab", sh::SHT_SYMTAB)
                .data(symtab)
                .link(2)
                .entsize(24),
        ],
        h::EM_X86_64,
        h::ET_REL,
    );
    let elf = goblin::elf::Elf::parse(&blob).unwrap();
    let bpf_objs_idx = find_section(&elf, ".bpf.objs").unwrap();
    let out = iter_embedded_bpf_objects(&elf, &blob, bpf_objs_idx);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].len(), 24);
    let expected: Vec<u8> = (4..28u8).collect();
    assert_eq!(out[0], expected.as_slice());
}

/// Symbol whose `st_value + st_size` exceeds the section bounds
/// is rejected; the iterator falls back to the full section.
#[test]
fn iter_embedded_bpf_objects_rejects_oversized_symbol() {
    let payload = b"0123456789abcdef".to_vec(); // 16 bytes
    let payload_len = payload.len();
    let strtab = b"\0bpf_obj\0".to_vec();
    let mut symtab = Vec::new();
    symtab.extend_from_slice(&elf64_sym(0, 0, 0, 0, 0));
    // st_size=200 vs section size=16 → reject → fallback fires.
    symtab.extend_from_slice(&elf64_sym(
        1,
        st_info(syms::STB_GLOBAL, syms::STT_OBJECT),
        1,
        0,
        200,
    ));
    let blob = build_elf64(
        vec![
            SecSpec::new(".bpf.objs", sh::SHT_PROGBITS).data(payload),
            SecSpec::new(".strtab", sh::SHT_STRTAB).data(strtab),
            SecSpec::new(".symtab", sh::SHT_SYMTAB)
                .data(symtab)
                .link(2)
                .entsize(24),
        ],
        h::EM_X86_64,
        h::ET_REL,
    );
    let elf = goblin::elf::Elf::parse(&blob).unwrap();
    let bpf_objs_idx = find_section(&elf, ".bpf.objs").unwrap();
    let out = iter_embedded_bpf_objects(&elf, &blob, bpf_objs_idx);
    assert_eq!(out.len(), 1, "fallback yields exactly one slice");
    assert_eq!(out[0].len(), payload_len);
}

/// Symbol whose `st_type` is `STT_FUNC` (not `STT_OBJECT`) is
/// skipped — iterator falls back to the full section.
#[test]
fn iter_embedded_bpf_objects_skips_non_object_symbols() {
    let payload = b"hello-bpf-objects".to_vec();
    let payload_len = payload.len();
    let strtab = b"\0func_sym\0".to_vec();
    let mut symtab = Vec::new();
    symtab.extend_from_slice(&elf64_sym(0, 0, 0, 0, 0));
    symtab.extend_from_slice(&elf64_sym(
        1,
        st_info(syms::STB_GLOBAL, syms::STT_FUNC),
        1,
        0,
        8,
    ));
    let blob = build_elf64(
        vec![
            SecSpec::new(".bpf.objs", sh::SHT_PROGBITS).data(payload),
            SecSpec::new(".strtab", sh::SHT_STRTAB).data(strtab),
            SecSpec::new(".symtab", sh::SHT_SYMTAB)
                .data(symtab)
                .link(2)
                .entsize(24),
        ],
        h::EM_X86_64,
        h::ET_REL,
    );
    let elf = goblin::elf::Elf::parse(&blob).unwrap();
    let bpf_objs_idx = find_section(&elf, ".bpf.objs").unwrap();
    let out = iter_embedded_bpf_objects(&elf, &blob, bpf_objs_idx);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].len(), payload_len);
}

// ----- analyze_one_object_with_btf error paths -------------------

/// Inner ELF whose bytes do not start with a valid ELF magic
/// fails goblin parse — `analyze_one_object_with_btf` returns
/// empty.
#[test]
fn analyze_one_object_corrupt_elf_returns_empty() {
    let bytes = vec![0u8; 64]; // all zeros — bad ELF magic
    let (map, btf, _alloc_sizes) = analyze_one_object_with_btf(&bytes);
    assert!(map.is_empty());
    assert!(btf.is_none());
}

/// Inner ELF without a `.BTF` section returns an empty map and
/// no parsed BTF.
#[test]
fn analyze_one_object_no_btf_returns_empty() {
    let bytes = build_elf64(
        vec![
            SecSpec::new(".text", sh::SHT_PROGBITS)
                .flags(sh::SHF_EXECINSTR.into())
                .data(vec![0u8; 8]),
        ],
        h::EM_BPF,
        h::ET_REL,
    );
    let (map, btf, _alloc_sizes) = analyze_one_object_with_btf(&bytes);
    assert!(map.is_empty());
    assert!(btf.is_none());
}

/// Inner ELF whose `.BTF` bytes do not parse as valid BTF
/// returns empty.
#[test]
fn analyze_one_object_corrupt_btf_returns_empty() {
    let bytes = build_elf64(
        vec![
            SecSpec::new(".text", sh::SHT_PROGBITS)
                .flags(sh::SHF_EXECINSTR.into())
                .data(insns_to_text_bytes(&[exit_insn()])),
            SecSpec::new(".BTF", sh::SHT_PROGBITS).data(vec![0xFFu8; 32]),
        ],
        h::EM_BPF,
        h::ET_REL,
    );
    let (map, btf, _alloc_sizes) = analyze_one_object_with_btf(&bytes);
    assert!(map.is_empty());
    assert!(btf.is_none());
}

/// Inner ELF with valid BTF but no executable text section
/// produces no instructions to analyze → empty map. The parsed
/// BTF is still returned so its struct/union definitions can
/// feed the cross-BTF Fwd index.
#[test]
fn analyze_one_object_no_text_section_returns_empty() {
    let bytes = build_elf64(
        vec![SecSpec::new(".BTF", sh::SHT_PROGBITS).data(build_btf_blob(&[], b"\0"))],
        h::EM_BPF,
        h::ET_REL,
    );
    let (map, btf, _alloc_sizes) = analyze_one_object_with_btf(&bytes);
    assert!(map.is_empty());
    assert!(btf.is_some());
}

/// Text section whose byte length is not a multiple of 8 is
/// skipped during decode → empty map. As with the no-text case,
/// the parsed BTF is still returned for cross-BTF Fwd indexing.
#[test]
fn analyze_one_object_misaligned_text_skipped() {
    let bytes = build_elf64(
        vec![
            SecSpec::new(".text", sh::SHT_PROGBITS)
                .flags(sh::SHF_EXECINSTR.into())
                .data(vec![0u8; 7]),
            SecSpec::new(".BTF", sh::SHT_PROGBITS).data(build_btf_blob(&[], b"\0")),
        ],
        h::EM_BPF,
        h::ET_REL,
    );
    let (map, btf, _alloc_sizes) = analyze_one_object_with_btf(&bytes);
    assert!(map.is_empty());
    assert!(btf.is_some());
}

// ----- analyze_one_object_with_btf end-to-end recovery -----------

/// Full pipeline: BTF describes T (id=2) with a u64 field at
/// offset 8 and Q (id=3) with a u64 field at offset 0; .text
/// contains a function entry that loads T.f then dereferences
/// it as Q*; .BTF.ext seeds R1=*T at the entry. Expected:
/// CastMap maps `(2, 8) → CastHit { alloc_size: None, 3, Arena }`.
#[test]
fn analyze_one_object_recovers_arena_cast_end_to_end() {
    let mut strings = vec![0u8];
    let n_int = push_btf_name(&mut strings, "u64");
    let n_t = push_btf_name(&mut strings, "T");
    let n_q = push_btf_name(&mut strings, "Q");
    let n_f = push_btf_name(&mut strings, "f");
    let n_x = push_btf_name(&mut strings, "x");
    let n_func = push_btf_name(&mut strings, "myfunc");
    let n_text = push_btf_name(&mut strings, ".text");
    // id=1 u64, id=2 T, id=3 Q, id=4 *T, id=5 FuncProto(*T),
    // id=6 Func(myfunc@5).
    let types = vec![
        SynKind::Int {
            name_off: n_int,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        SynKind::Struct {
            name_off: n_t,
            size: 16,
            members: vec![SynMember {
                name_off: n_f,
                type_id: 1,
                byte_offset: 8,
            }],
        },
        SynKind::Struct {
            name_off: n_q,
            size: 8,
            members: vec![SynMember {
                name_off: n_x,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        SynKind::Ptr { type_id: 2 },
        SynKind::FuncProto {
            return_type_id: 0,
            params: vec![SynParam {
                name_off: 0,
                type_id: 4,
            }],
        },
        SynKind::Func {
            name_off: n_func,
            type_id: 5,
            linkage: 1,
        },
    ];
    let btf_blob = build_btf_full(&types, &strings);
    // r2 = *(u64 *)(r1 + 8); r2 = arena_cast(r2);
    // r3 = *(u64 *)(r2 + 0); exit.
    // The arena_cast adds (T, 8) to arena_confirmed (arena-evidence
    // prerequisite for the shape-inference finding).
    let insns = vec![
        ldx_dw_mem(2, 1, 8),
        addr_space_cast_insn(2, 2),
        ldx_dw_mem(3, 2, 0),
        exit_insn(),
    ];
    let text = insns_to_text_bytes(&insns);
    let btf_ext = build_btf_ext(n_text, &[(0, 5)], 8);

    let bytes = build_full_bpf_object_elf(text, btf_blob, btf_ext);
    let (map, btf, _alloc_sizes) = analyze_one_object_with_btf(&bytes);
    assert!(btf.is_some(), "valid BTF must be returned");
    let hit = map.get(&(2u32, 8u32)).copied();
    assert_eq!(
        hit,
        Some(CastHit {
            alloc_size: None,
            target_type_id: 3,
            addr_space: AddrSpace::Arena,
        }),
        "expected arena cast T.f → Q*, got {map:?}"
    );
}

// ----- cached_cast_analysis_for_scheduler error & happy paths ---

/// Outer ELF that parses successfully but whose `.bpf.objs`
/// bytes are not a valid inner ELF — outer merge is empty,
/// cache layer collapses to `None`.
#[test]
fn cached_cast_analysis_corrupt_inner_returns_none() {
    let outer = build_elf64(
        vec![SecSpec::new(".bpf.objs", sh::SHT_PROGBITS).data(b"not-an-elf".to_vec())],
        h::EM_X86_64,
        h::ET_REL,
    );
    let dir = tempfile::tempdir().expect("tempdir");
    let p = dir.path().join("bad_inner.bin");
    std::fs::write(&p, &outer).expect("write");
    assert!(cached_cast_analysis_for_scheduler(&p).is_none());
}

/// Outer ELF whose `.bpf.objs` carries an inner BPF ELF
/// without a `.BTF` section — outer merge is empty, cache
/// layer collapses to `None`.
#[test]
fn cached_cast_analysis_inner_without_btf_returns_none() {
    let inner = build_elf64(
        vec![
            SecSpec::new(".text", sh::SHT_PROGBITS)
                .flags(sh::SHF_EXECINSTR.into())
                .data(vec![0u8; 8]),
        ],
        h::EM_BPF,
        h::ET_REL,
    );
    let outer = build_elf64(
        vec![SecSpec::new(".bpf.objs", sh::SHT_PROGBITS).data(inner)],
        h::EM_X86_64,
        h::ET_REL,
    );
    let dir = tempfile::tempdir().expect("tempdir");
    let p = dir.path().join("no_inner_btf.bin");
    std::fs::write(&p, &outer).expect("write");
    assert!(cached_cast_analysis_for_scheduler(&p).is_none());
}

/// Full end-to-end through the public driver: outer host ELF
/// wraps an inner BPF ELF that recovers an arena cast.
#[test]
fn cached_cast_analysis_recovers_arena_cast_end_to_end() {
    let mut strings = vec![0u8];
    let n_int = push_btf_name(&mut strings, "u64");
    let n_t = push_btf_name(&mut strings, "T");
    let n_q = push_btf_name(&mut strings, "Q");
    let n_f = push_btf_name(&mut strings, "f");
    let n_x = push_btf_name(&mut strings, "x");
    let n_func = push_btf_name(&mut strings, "myfunc");
    let n_text = push_btf_name(&mut strings, ".text");
    let types = vec![
        SynKind::Int {
            name_off: n_int,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        SynKind::Struct {
            name_off: n_t,
            size: 16,
            members: vec![SynMember {
                name_off: n_f,
                type_id: 1,
                byte_offset: 8,
            }],
        },
        SynKind::Struct {
            name_off: n_q,
            size: 8,
            members: vec![SynMember {
                name_off: n_x,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        SynKind::Ptr { type_id: 2 },
        SynKind::FuncProto {
            return_type_id: 0,
            params: vec![SynParam {
                name_off: 0,
                type_id: 4,
            }],
        },
        SynKind::Func {
            name_off: n_func,
            type_id: 5,
            linkage: 1,
        },
    ];
    let btf_blob = build_btf_full(&types, &strings);
    // Arena-evidence mitigation: include arena_space_cast on r2 so the
    // shape-inference finding emits.
    let insns = vec![
        ldx_dw_mem(2, 1, 8),
        addr_space_cast_insn(2, 2),
        ldx_dw_mem(3, 2, 0),
        exit_insn(),
    ];
    let text = insns_to_text_bytes(&insns);
    let btf_ext = build_btf_ext(n_text, &[(0, 5)], 8);

    let inner = build_full_bpf_object_elf(text, btf_blob, btf_ext);
    let outer = build_elf64(
        vec![SecSpec::new(".bpf.objs", sh::SHT_PROGBITS).data(inner)],
        h::EM_X86_64,
        h::ET_REL,
    );
    let dir = tempfile::tempdir().expect("tempdir");
    let p = dir.path().join("full.bin");
    std::fs::write(&p, &outer).expect("write");

    let out = cached_cast_analysis_for_scheduler(&p).expect("non-empty fixture must produce Some");
    let hit = out.cast_maps[0].get(&(2u32, 8u32)).copied();
    assert_eq!(
        hit,
        Some(CastHit {
            alloc_size: None,
            target_type_id: 3,
            addr_space: AddrSpace::Arena,
        }),
        "expected arena cast T.f → Q*, got {:?}",
        out.cast_maps[0]
    );
}

/// Cache hit by content: two calls on the same bytes (different
/// paths) return the same `Arc<CastAnalysisOutput>`. Proves the
/// cache is content-keyed (SHA-256), not path-keyed.
#[test]
fn cached_cast_analysis_returns_same_arc_for_same_content() {
    let blob = build_recovers_arena_cast_outer_elf();
    let dir = tempfile::tempdir().expect("tempdir");
    let p1 = dir.path().join("first.bin");
    let p2 = dir.path().join("second.bin");
    std::fs::write(&p1, &blob).expect("write 1");
    std::fs::write(&p2, &blob).expect("write 2");

    let first = cached_cast_analysis_for_scheduler(&p1).expect("Some on non-empty analysis");
    let second = cached_cast_analysis_for_scheduler(&p2).expect("cache hit on identical content");

    assert!(
        Arc::ptr_eq(&first, &second),
        "expected pointer-equal Arc when two paths have identical content"
    );
    // Sanity: the cached output carries the recovered cast.
    assert_eq!(
        first.cast_maps[0].get(&(2u32, 8u32)).copied(),
        Some(CastHit {
            alloc_size: None,
            target_type_id: 3,
            addr_space: AddrSpace::Arena,
        }),
    );
}

/// Cache miss by content: an empty-result blob caches as
/// `None`. Proves the empty-result collapse (cast_map empty
/// AND fwd_index empty) is preserved across cache lookups.
#[test]
fn cached_cast_analysis_collapses_empty_to_none() {
    let empty_blob = build_elf64(
        vec![SecSpec::new(".text", sh::SHT_PROGBITS).flags(sh::SHF_EXECINSTR.into())],
        h::EM_X86_64,
        h::ET_REL,
    );

    let dir = tempfile::tempdir().expect("tempdir");
    let p = dir.path().join("empty.bin");
    std::fs::write(&p, &empty_blob).expect("write");

    // First call analyzes; result is empty → None.
    assert!(cached_cast_analysis_for_scheduler(&p).is_none());
    // Second call hits the same content-hash cache entry and
    // also resolves to None without re-running the analyzer.
    assert!(cached_cast_analysis_for_scheduler(&p).is_none());
}

/// Read-failure path: a non-existent path produces `None`
/// without polluting the cache. A later call after the file
/// appears must succeed and run the analyzer on demand.
#[test]
fn cached_cast_analysis_read_failure_does_not_pollute_cache() {
    let dir = tempfile::tempdir().expect("tempdir");
    let p = dir.path().join("appears_later.bin");

    assert!(!p.exists());
    assert!(cached_cast_analysis_for_scheduler(&p).is_none());

    let blob = build_recovers_arena_cast_outer_elf();
    std::fs::write(&p, &blob).expect("write");
    let out = cached_cast_analysis_for_scheduler(&p)
        .expect("post-creation read should succeed and produce a non-empty CastAnalysisOutput");
    assert_eq!(
        out.cast_maps[0].get(&(2u32, 8u32)).copied(),
        Some(CastHit {
            alloc_size: None,
            target_type_id: 3,
            addr_space: AddrSpace::Arena,
        }),
        "post-creation analysis should recover the seeded cast"
    );
}

/// Lazy wrapper: `LazyCastMap::new` runs no analysis. The
/// `OnceLock` is empty until `.get_full()` fires, and
/// `.get_full()` returns identical `Arc`s on every subsequent
/// call (the analyzer ran exactly once).
#[test]
fn lazy_cast_map_get_full_is_idempotent_and_lazy() {
    let blob = build_recovers_arena_cast_outer_elf();
    let dir = tempfile::tempdir().expect("tempdir");
    let p = dir.path().join("lazy.bin");
    std::fs::write(&p, &blob).expect("write");

    let lazy = LazyCastMap::new(Some(p.clone()));
    // Sanity: the lazy slot is empty before any `.get_full()`.
    assert!(
        lazy.inner.get().is_none(),
        "LazyCastMap::new must not run analysis"
    );

    let first = lazy.get_full().expect("non-empty result");
    let second = lazy.get_full().expect("non-empty result");
    assert!(
        Arc::ptr_eq(&first, &second),
        "OnceLock-backed `.get_full()` must return the same Arc on every call"
    );
}

/// `LazyCastMap::get_full` on a binary with no recoverable
/// casts returns `None` (the cache layer collapses empty
/// results). The renderer treats `None` identically to an
/// empty map, so this keeps the pre-integration default
/// behaviour intact.
#[test]
fn lazy_cast_map_get_full_returns_none_for_no_findings() {
    let empty_blob = build_elf64(
        vec![SecSpec::new(".text", sh::SHT_PROGBITS).flags(sh::SHF_EXECINSTR.into())],
        h::EM_X86_64,
        h::ET_REL,
    );
    let dir = tempfile::tempdir().expect("tempdir");
    let p = dir.path().join("no_findings.bin");
    std::fs::write(&p, &empty_blob).expect("write");

    let lazy = LazyCastMap::new(Some(p));
    assert!(
        lazy.get_full().is_none(),
        "no-`.bpf.objs` binary must collapse to None on `.get_full()`"
    );
}

/// `objects_with_casts` counts only cast-bearing objects -- the gate
/// `build_cast_analysis_from_bytes` uses to fail loudly on the
/// unsupported multi-object case (renderer threads `first()`, cache
/// merges into one flat map, per-object BTF id-spaces collide).
#[test]
fn objects_with_casts_counts_only_cast_bearing_objects() {
    let empty = Arc::new(CastMap::new());
    let mut m = CastMap::new();
    m.insert(
        (1, 0),
        CastHit {
            target_type_id: 2,
            addr_space: AddrSpace::Arena,
            alloc_size: None,
        },
    );
    let with_cast = Arc::new(m);

    assert_eq!(objects_with_casts(&[]), 0);
    assert_eq!(objects_with_casts(std::slice::from_ref(&empty)), 0);
    assert_eq!(objects_with_casts(std::slice::from_ref(&with_cast)), 1);
    assert_eq!(
        objects_with_casts(&[with_cast.clone(), empty.clone()]),
        1,
        "empty objects do not count toward the multi-object gate"
    );
    assert_eq!(
        objects_with_casts(&[with_cast.clone(), with_cast.clone()]),
        2,
        "two cast-bearing objects is the case the guard fires on"
    );
}

// ----- parse_btf_ext_func_entries happy paths --------------------

/// Records produce one [`FuncEntry`] each, with `insn_offset`
/// measured in instruction indices (byte offset / 8) plus the
/// section base supplied by the caller.
#[test]
fn parse_btf_ext_records_produce_func_entries() {
    let mut strings = vec![0u8];
    let n_text = push_btf_name(&mut strings, ".text");
    let btf_blob = build_btf_full(&[], &strings);

    let inner = build_elf64(
        vec![
            SecSpec::new(".text", sh::SHT_PROGBITS)
                .flags(sh::SHF_EXECINSTR.into())
                .data(vec![0u8; 32]),
        ],
        h::EM_BPF,
        h::ET_REL,
    );
    let elf = goblin::elf::Elf::parse(&inner).unwrap();
    let text_idx = find_section(&elf, ".text").expect(".text") as u32;
    let mut bases: HashMap<u32, usize> = HashMap::new();
    bases.insert(text_idx, 0);

    let data = build_btf_ext(n_text, &[(0, 11), (16, 22)], 8);
    let out = parse_btf_ext_func_entries(&data, &btf_blob, &elf, &bases);
    assert_eq!(out.len(), 2, "got {out:?}");
    assert_eq!(out[0].insn_offset, 0);
    assert_eq!(out[0].func_proto_id, 11);
    assert_eq!(out[1].insn_offset, 2);
    assert_eq!(out[1].func_proto_id, 22);
}

/// Record offsets are measured relative to the section's base
/// in the concatenated text stream.
#[test]
fn parse_btf_ext_applies_section_base_offset() {
    let mut strings = vec![0u8];
    let n_text = push_btf_name(&mut strings, ".text");
    let btf_blob = build_btf_full(&[], &strings);
    let inner = build_elf64(
        vec![
            SecSpec::new(".text", sh::SHT_PROGBITS)
                .flags(sh::SHF_EXECINSTR.into())
                .data(vec![0u8; 32]),
        ],
        h::EM_BPF,
        h::ET_REL,
    );
    let elf = goblin::elf::Elf::parse(&inner).unwrap();
    let text_idx = find_section(&elf, ".text").expect(".text") as u32;
    let mut bases: HashMap<u32, usize> = HashMap::new();
    bases.insert(text_idx, 10);
    let data = build_btf_ext(n_text, &[(16, 5)], 8);
    let out = parse_btf_ext_func_entries(&data, &btf_blob, &elf, &bases);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].insn_offset, 12);
    assert_eq!(out[0].func_proto_id, 5);
}

/// `record_size` larger than the minimum 8 bytes means
/// trailing padding the parser must skip.
#[test]
fn parse_btf_ext_handles_padded_records() {
    let mut strings = vec![0u8];
    let n_text = push_btf_name(&mut strings, ".text");
    let btf_blob = build_btf_full(&[], &strings);
    let inner = build_elf64(
        vec![
            SecSpec::new(".text", sh::SHT_PROGBITS)
                .flags(sh::SHF_EXECINSTR.into())
                .data(vec![0u8; 32]),
        ],
        h::EM_BPF,
        h::ET_REL,
    );
    let elf = goblin::elf::Elf::parse(&inner).unwrap();
    let text_idx = find_section(&elf, ".text").expect(".text") as u32;
    let mut bases: HashMap<u32, usize> = HashMap::new();
    bases.insert(text_idx, 0);
    let data = build_btf_ext(n_text, &[(0, 11), (8, 22)], 16);
    let out = parse_btf_ext_func_entries(&data, &btf_blob, &elf, &bases);
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].insn_offset, 0);
    assert_eq!(out[0].func_proto_id, 11);
    assert_eq!(out[1].insn_offset, 1);
    assert_eq!(out[1].func_proto_id, 22);
}

/// `sec_name_off` that does not resolve in the BTF strtab
/// causes records to be silently skipped.
#[test]
fn parse_btf_ext_skips_unresolvable_section_name() {
    let strings = vec![0u8];
    let btf_blob = build_btf_full(&[], &strings);
    let inner = build_elf64(
        vec![
            SecSpec::new(".text", sh::SHT_PROGBITS)
                .flags(sh::SHF_EXECINSTR.into())
                .data(vec![0u8; 32]),
        ],
        h::EM_BPF,
        h::ET_REL,
    );
    let elf = goblin::elf::Elf::parse(&inner).unwrap();
    let bases: HashMap<u32, usize> = HashMap::new();
    let data = build_btf_ext(999, &[(0, 7)], 8);
    let out = parse_btf_ext_func_entries(&data, &btf_blob, &elf, &bases);
    assert!(out.is_empty());
}

/// `sec_name_off` resolves to a name that does not match any
/// ELF section — records are skipped.
#[test]
fn parse_btf_ext_skips_section_not_in_elf() {
    let mut strings = vec![0u8];
    let n_other = push_btf_name(&mut strings, ".not_in_elf");
    let btf_blob = build_btf_full(&[], &strings);
    let inner = build_elf64(
        vec![
            SecSpec::new(".text", sh::SHT_PROGBITS)
                .flags(sh::SHF_EXECINSTR.into())
                .data(vec![0u8; 32]),
        ],
        h::EM_BPF,
        h::ET_REL,
    );
    let elf = goblin::elf::Elf::parse(&inner).unwrap();
    let bases: HashMap<u32, usize> = HashMap::new();
    let data = build_btf_ext(n_other, &[(0, 7)], 8);
    let out = parse_btf_ext_func_entries(&data, &btf_blob, &elf, &bases);
    assert!(out.is_empty());
}

/// ELF section exists but `section_bases` lacks an entry —
/// records skipped.
#[test]
fn parse_btf_ext_skips_section_without_base() {
    let mut strings = vec![0u8];
    let n_text = push_btf_name(&mut strings, ".text");
    let btf_blob = build_btf_full(&[], &strings);
    let inner = build_elf64(
        vec![
            SecSpec::new(".text", sh::SHT_PROGBITS)
                .flags(sh::SHF_EXECINSTR.into())
                .data(vec![0u8; 32]),
        ],
        h::EM_BPF,
        h::ET_REL,
    );
    let elf = goblin::elf::Elf::parse(&inner).unwrap();
    let bases: HashMap<u32, usize> = HashMap::new();
    let data = build_btf_ext(n_text, &[(0, 7)], 8);
    let out = parse_btf_ext_func_entries(&data, &btf_blob, &elf, &bases);
    assert!(out.is_empty());
}

/// `func_info_len` of zero short-circuits the record loop.
#[test]
fn parse_btf_ext_zero_func_info_len_returns_empty() {
    let btf_blob = build_btf_full(&[], b"\0");
    let inner = build_elf64(vec![], h::EM_BPF, h::ET_REL);
    let elf = goblin::elf::Elf::parse(&inner).unwrap();
    let bases = HashMap::new();
    let mut data = vec![0u8; 24];
    data[0..2].copy_from_slice(&0xEB9F_u16.to_le_bytes());
    data[4..8].copy_from_slice(&24u32.to_le_bytes());
    let out = parse_btf_ext_func_entries(&data, &btf_blob, &elf, &bases);
    assert!(out.is_empty());
}
