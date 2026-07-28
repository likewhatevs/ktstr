use super::super::*;
use super::*;
use goblin::elf::header as h;
use goblin::elf::section_header as sh;
use goblin::elf::sym as syms;

/// Test 1 — happy path: kfunc call gets rewritten.
#[test]
fn patch_kfunc_calls_happy_path_rewrites_call_site() {
    let kf_name = "bpf_task_acquire";
    let (btf_blob, expected_func_id, _t_id) = build_kfunc_btf_blob(kf_name);
    let btf = Btf::from_bytes(&btf_blob).expect("parse btf");

    let mut strtab: Vec<u8> = vec![0];
    let kf_str_off = strtab.len() as u32;
    strtab.extend_from_slice(kf_name.as_bytes());
    strtab.push(0);

    let mut symtab: Vec<u8> = Vec::new();
    symtab.extend_from_slice(&elf64_sym(0, 0, 0, 0, 0));
    symtab.extend_from_slice(&elf64_sym(
        kf_str_off,
        st_info(syms::STB_GLOBAL, syms::STT_NOTYPE),
        0,
        0,
        0,
    ));

    let mut text: Vec<u8> = Vec::new();
    text.extend_from_slice(&pre_reloc_kfunc_call_bytes());
    text.extend_from_slice(&kfunc_exit_bytes());

    let rel_data: Vec<u8> = elf64_rel(0, 1, 10).to_vec();

    let blob = build_elf64(
        vec![
            SecSpec::new(".text", sh::SHT_PROGBITS)
                .flags(sh::SHF_EXECINSTR.into())
                .data(text),
            SecSpec::new(".strtab", sh::SHT_STRTAB).data(strtab),
            SecSpec::new(".symtab", sh::SHT_SYMTAB)
                .data(symtab)
                .link(2)
                .entsize(24),
            SecSpec::new(".rel.text", sh::SHT_REL)
                .data(rel_data)
                .link(3)
                .info(1)
                .entsize(16),
            SecSpec::new(".BTF", sh::SHT_PROGBITS).data(btf_blob),
        ],
        h::EM_BPF,
        h::ET_REL,
    );
    let elf = goblin::elf::Elf::parse(&blob).expect("parse elf");

    let mut text_concat: Vec<BpfInsn> = vec![
        BpfInsn::from_le_bytes(pre_reloc_kfunc_call_bytes()),
        BpfInsn::from_le_bytes(kfunc_exit_bytes()),
    ];
    let mut section_bases: HashMap<u32, usize> = HashMap::new();
    section_bases.insert(1, 0);

    assert_eq!(text_concat[0].code, 0x85);
    assert_eq!(text_concat[0].src_reg(), BPF_PSEUDO_CALL);
    assert_eq!(text_concat[0].imm, -1);

    patch_kfunc_calls(&mut text_concat, &btf, &elf, &section_bases);

    assert_eq!(text_concat[0].code, 0x85);
    assert_eq!(
        text_concat[0].src_reg(),
        BPF_PSEUDO_KFUNC_CALL,
        "src_reg now BPF_PSEUDO_KFUNC_CALL"
    );
    assert_eq!(
        text_concat[0].imm, expected_func_id as i32,
        "imm patched to BTF Func id"
    );
    assert_eq!(text_concat[1].code, 0x95);
}

/// Test 2 — non-extern symbol must NOT trigger patching.
#[test]
fn patch_kfunc_calls_skips_non_extern_symbol() {
    let kf_name = "static_helper";
    let (btf_blob, _func_id, _) = build_kfunc_btf_blob(kf_name);
    let btf = Btf::from_bytes(&btf_blob).expect("parse btf");

    let mut strtab: Vec<u8> = vec![0];
    let name_off = strtab.len() as u32;
    strtab.extend_from_slice(kf_name.as_bytes());
    strtab.push(0);
    let mut symtab: Vec<u8> = Vec::new();
    symtab.extend_from_slice(&elf64_sym(0, 0, 0, 0, 0));
    symtab.extend_from_slice(&elf64_sym(
        name_off,
        st_info(syms::STB_LOCAL, syms::STT_NOTYPE),
        0,
        0,
        0,
    ));

    let mut text: Vec<u8> = Vec::new();
    text.extend_from_slice(&pre_reloc_kfunc_call_bytes());
    text.extend_from_slice(&kfunc_exit_bytes());
    let rel_data: Vec<u8> = elf64_rel(0, 1, 10).to_vec();

    let blob = build_elf64(
        vec![
            SecSpec::new(".text", sh::SHT_PROGBITS)
                .flags(sh::SHF_EXECINSTR.into())
                .data(text),
            SecSpec::new(".strtab", sh::SHT_STRTAB).data(strtab),
            SecSpec::new(".symtab", sh::SHT_SYMTAB)
                .data(symtab)
                .link(2)
                .entsize(24),
            SecSpec::new(".rel.text", sh::SHT_REL)
                .data(rel_data)
                .link(3)
                .info(1)
                .entsize(16),
            SecSpec::new(".BTF", sh::SHT_PROGBITS).data(btf_blob),
        ],
        h::EM_BPF,
        h::ET_REL,
    );
    let elf = goblin::elf::Elf::parse(&blob).expect("parse elf");
    let mut text_concat: Vec<BpfInsn> = vec![
        BpfInsn::from_le_bytes(pre_reloc_kfunc_call_bytes()),
        BpfInsn::from_le_bytes(kfunc_exit_bytes()),
    ];
    let mut section_bases: HashMap<u32, usize> = HashMap::new();
    section_bases.insert(1, 0);

    patch_kfunc_calls(&mut text_concat, &btf, &elf, &section_bases);

    assert_eq!(text_concat[0].src_reg(), BPF_PSEUDO_CALL);
    assert_eq!(text_concat[0].imm, -1);
}

/// Test 3 — symbol is extern but its name does NOT resolve to
/// an extern FUNC in the program BTF.
#[test]
fn patch_kfunc_calls_skips_symbol_not_in_btf() {
    let (btf_blob, _func_id, _) = build_kfunc_btf_blob("bpf_task_acquire");
    let btf = Btf::from_bytes(&btf_blob).expect("parse btf");

    let unknown = "unknown_kfunc";
    let mut strtab: Vec<u8> = vec![0];
    let name_off = strtab.len() as u32;
    strtab.extend_from_slice(unknown.as_bytes());
    strtab.push(0);
    let mut symtab: Vec<u8> = Vec::new();
    symtab.extend_from_slice(&elf64_sym(0, 0, 0, 0, 0));
    symtab.extend_from_slice(&elf64_sym(
        name_off,
        st_info(syms::STB_GLOBAL, syms::STT_NOTYPE),
        0,
        0,
        0,
    ));

    let mut text: Vec<u8> = Vec::new();
    text.extend_from_slice(&pre_reloc_kfunc_call_bytes());
    text.extend_from_slice(&kfunc_exit_bytes());
    let rel_data: Vec<u8> = elf64_rel(0, 1, 10).to_vec();

    let blob = build_elf64(
        vec![
            SecSpec::new(".text", sh::SHT_PROGBITS)
                .flags(sh::SHF_EXECINSTR.into())
                .data(text),
            SecSpec::new(".strtab", sh::SHT_STRTAB).data(strtab),
            SecSpec::new(".symtab", sh::SHT_SYMTAB)
                .data(symtab)
                .link(2)
                .entsize(24),
            SecSpec::new(".rel.text", sh::SHT_REL)
                .data(rel_data)
                .link(3)
                .info(1)
                .entsize(16),
            SecSpec::new(".BTF", sh::SHT_PROGBITS).data(btf_blob),
        ],
        h::EM_BPF,
        h::ET_REL,
    );
    let elf = goblin::elf::Elf::parse(&blob).expect("parse elf");
    let mut text_concat: Vec<BpfInsn> = vec![
        BpfInsn::from_le_bytes(pre_reloc_kfunc_call_bytes()),
        BpfInsn::from_le_bytes(kfunc_exit_bytes()),
    ];
    let mut section_bases: HashMap<u32, usize> = HashMap::new();
    section_bases.insert(1, 0);

    patch_kfunc_calls(&mut text_concat, &btf, &elf, &section_bases);

    assert_eq!(text_concat[0].src_reg(), BPF_PSEUDO_CALL);
    assert_eq!(text_concat[0].imm, -1);
}

/// Test 4 — relocation targets a section we did NOT add to
/// `section_bases` (e.g. `.maps`).
#[test]
fn patch_kfunc_calls_ignores_non_text_relocations() {
    let kf_name = "bpf_task_acquire";
    let (btf_blob, _func_id, _) = build_kfunc_btf_blob(kf_name);
    let btf = Btf::from_bytes(&btf_blob).expect("parse btf");

    let mut strtab: Vec<u8> = vec![0];
    let name_off = strtab.len() as u32;
    strtab.extend_from_slice(kf_name.as_bytes());
    strtab.push(0);
    let mut symtab: Vec<u8> = Vec::new();
    symtab.extend_from_slice(&elf64_sym(0, 0, 0, 0, 0));
    symtab.extend_from_slice(&elf64_sym(
        name_off,
        st_info(syms::STB_GLOBAL, syms::STT_NOTYPE),
        0,
        0,
        0,
    ));

    let mut text: Vec<u8> = Vec::new();
    text.extend_from_slice(&pre_reloc_kfunc_call_bytes());
    text.extend_from_slice(&kfunc_exit_bytes());
    let rel_data: Vec<u8> = elf64_rel(0, 1, 10).to_vec();

    let blob = build_elf64(
        vec![
            SecSpec::new(".text", sh::SHT_PROGBITS)
                .flags(sh::SHF_EXECINSTR.into())
                .data(text),
            SecSpec::new(".maps", sh::SHT_PROGBITS).data(vec![0u8; 8]),
            SecSpec::new(".strtab", sh::SHT_STRTAB).data(strtab),
            SecSpec::new(".symtab", sh::SHT_SYMTAB)
                .data(symtab)
                .link(3)
                .entsize(24),
            SecSpec::new(".rel.maps", sh::SHT_REL)
                .data(rel_data)
                .link(4)
                .info(2)
                .entsize(16),
            SecSpec::new(".BTF", sh::SHT_PROGBITS).data(btf_blob),
        ],
        h::EM_BPF,
        h::ET_REL,
    );
    let elf = goblin::elf::Elf::parse(&blob).expect("parse elf");
    let mut text_concat: Vec<BpfInsn> = vec![
        BpfInsn::from_le_bytes(pre_reloc_kfunc_call_bytes()),
        BpfInsn::from_le_bytes(kfunc_exit_bytes()),
    ];
    let mut section_bases: HashMap<u32, usize> = HashMap::new();
    section_bases.insert(1, 0);

    patch_kfunc_calls(&mut text_concat, &btf, &elf, &section_bases);

    assert_eq!(text_concat[0].src_reg(), BPF_PSEUDO_CALL);
    assert_eq!(text_concat[0].imm, -1);
}

/// Test 5 — relocation byte offset is past the section's end.
#[test]
fn patch_kfunc_calls_rejects_out_of_bounds_offset() {
    let kf_name = "bpf_task_acquire";
    let (btf_blob, _func_id, _) = build_kfunc_btf_blob(kf_name);
    let btf = Btf::from_bytes(&btf_blob).expect("parse btf");

    let mut strtab: Vec<u8> = vec![0];
    let name_off = strtab.len() as u32;
    strtab.extend_from_slice(kf_name.as_bytes());
    strtab.push(0);
    let mut symtab: Vec<u8> = Vec::new();
    symtab.extend_from_slice(&elf64_sym(0, 0, 0, 0, 0));
    symtab.extend_from_slice(&elf64_sym(
        name_off,
        st_info(syms::STB_GLOBAL, syms::STT_NOTYPE),
        0,
        0,
        0,
    ));

    let mut text: Vec<u8> = Vec::new();
    text.extend_from_slice(&pre_reloc_kfunc_call_bytes());
    text.extend_from_slice(&kfunc_exit_bytes());
    // r_offset = 100 (past 16-byte .text).
    let rel_data: Vec<u8> = elf64_rel(100, 1, 10).to_vec();

    let blob = build_elf64(
        vec![
            SecSpec::new(".text", sh::SHT_PROGBITS)
                .flags(sh::SHF_EXECINSTR.into())
                .data(text),
            SecSpec::new(".strtab", sh::SHT_STRTAB).data(strtab),
            SecSpec::new(".symtab", sh::SHT_SYMTAB)
                .data(symtab)
                .link(2)
                .entsize(24),
            SecSpec::new(".rel.text", sh::SHT_REL)
                .data(rel_data)
                .link(3)
                .info(1)
                .entsize(16),
            SecSpec::new(".BTF", sh::SHT_PROGBITS).data(btf_blob),
        ],
        h::EM_BPF,
        h::ET_REL,
    );
    let elf = goblin::elf::Elf::parse(&blob).expect("parse elf");
    let mut text_concat: Vec<BpfInsn> = vec![
        BpfInsn::from_le_bytes(pre_reloc_kfunc_call_bytes()),
        BpfInsn::from_le_bytes(kfunc_exit_bytes()),
    ];
    let mut section_bases: HashMap<u32, usize> = HashMap::new();
    section_bases.insert(1, 0);

    patch_kfunc_calls(&mut text_concat, &btf, &elf, &section_bases);

    assert_eq!(text_concat[0].src_reg(), BPF_PSEUDO_CALL);
    assert_eq!(text_concat[0].imm, -1);
}

/// Test 6 — the relocation lands on a non-call instruction
/// (LD_IMM64). The patcher's code-byte gate rejects.
#[test]
fn patch_kfunc_calls_rejects_non_call_instruction() {
    let kf_name = "bpf_task_acquire";
    let (btf_blob, _func_id, _) = build_kfunc_btf_blob(kf_name);
    let btf = Btf::from_bytes(&btf_blob).expect("parse btf");

    let mut strtab: Vec<u8> = vec![0];
    let name_off = strtab.len() as u32;
    strtab.extend_from_slice(kf_name.as_bytes());
    strtab.push(0);
    let mut symtab: Vec<u8> = Vec::new();
    symtab.extend_from_slice(&elf64_sym(0, 0, 0, 0, 0));
    symtab.extend_from_slice(&elf64_sym(
        name_off,
        st_info(syms::STB_GLOBAL, syms::STT_NOTYPE),
        0,
        0,
        0,
    ));

    let ld_imm64_first_slot: [u8; 8] = [0x18, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
    let ld_imm64_second_slot: [u8; 8] = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
    let mut text: Vec<u8> = Vec::new();
    text.extend_from_slice(&ld_imm64_first_slot);
    text.extend_from_slice(&ld_imm64_second_slot);
    text.extend_from_slice(&kfunc_exit_bytes());
    let rel_data: Vec<u8> = elf64_rel(0, 1, 1).to_vec();

    let blob = build_elf64(
        vec![
            SecSpec::new(".text", sh::SHT_PROGBITS)
                .flags(sh::SHF_EXECINSTR.into())
                .data(text),
            SecSpec::new(".strtab", sh::SHT_STRTAB).data(strtab),
            SecSpec::new(".symtab", sh::SHT_SYMTAB)
                .data(symtab)
                .link(2)
                .entsize(24),
            SecSpec::new(".rel.text", sh::SHT_REL)
                .data(rel_data)
                .link(3)
                .info(1)
                .entsize(16),
            SecSpec::new(".BTF", sh::SHT_PROGBITS).data(btf_blob),
        ],
        h::EM_BPF,
        h::ET_REL,
    );
    let elf = goblin::elf::Elf::parse(&blob).expect("parse elf");
    let mut text_concat: Vec<BpfInsn> = vec![
        BpfInsn::from_le_bytes(ld_imm64_first_slot),
        BpfInsn::from_le_bytes(ld_imm64_second_slot),
        BpfInsn::from_le_bytes(kfunc_exit_bytes()),
    ];
    let mut section_bases: HashMap<u32, usize> = HashMap::new();
    section_bases.insert(1, 0);

    let pre = text_concat.clone();
    patch_kfunc_calls(&mut text_concat, &btf, &elf, &section_bases);

    assert_eq!(text_concat, pre);
}

/// Test 7 — relocation entry whose `imm` is NOT `-1` (a
/// resolved subprog call). Must not be patched.
#[test]
fn patch_kfunc_calls_rejects_non_minus_one_imm() {
    let kf_name = "bpf_task_acquire";
    let (btf_blob, _func_id, _) = build_kfunc_btf_blob(kf_name);
    let btf = Btf::from_bytes(&btf_blob).expect("parse btf");

    let mut strtab: Vec<u8> = vec![0];
    let name_off = strtab.len() as u32;
    strtab.extend_from_slice(kf_name.as_bytes());
    strtab.push(0);
    let mut symtab: Vec<u8> = Vec::new();
    symtab.extend_from_slice(&elf64_sym(0, 0, 0, 0, 0));
    symtab.extend_from_slice(&elf64_sym(
        name_off,
        st_info(syms::STB_GLOBAL, syms::STT_NOTYPE),
        0,
        0,
        0,
    ));

    // imm = 42 (not -1).
    let subprog_call: [u8; 8] = [0x85, 0x10, 0x00, 0x00, 0x2a, 0x00, 0x00, 0x00];
    let mut text: Vec<u8> = Vec::new();
    text.extend_from_slice(&subprog_call);
    text.extend_from_slice(&kfunc_exit_bytes());
    let rel_data: Vec<u8> = elf64_rel(0, 1, 10).to_vec();

    let blob = build_elf64(
        vec![
            SecSpec::new(".text", sh::SHT_PROGBITS)
                .flags(sh::SHF_EXECINSTR.into())
                .data(text),
            SecSpec::new(".strtab", sh::SHT_STRTAB).data(strtab),
            SecSpec::new(".symtab", sh::SHT_SYMTAB)
                .data(symtab)
                .link(2)
                .entsize(24),
            SecSpec::new(".rel.text", sh::SHT_REL)
                .data(rel_data)
                .link(3)
                .info(1)
                .entsize(16),
            SecSpec::new(".BTF", sh::SHT_PROGBITS).data(btf_blob),
        ],
        h::EM_BPF,
        h::ET_REL,
    );
    let elf = goblin::elf::Elf::parse(&blob).expect("parse elf");
    let mut text_concat: Vec<BpfInsn> = vec![
        BpfInsn::from_le_bytes(subprog_call),
        BpfInsn::from_le_bytes(kfunc_exit_bytes()),
    ];
    let mut section_bases: HashMap<u32, usize> = HashMap::new();
    section_bases.insert(1, 0);

    patch_kfunc_calls(&mut text_concat, &btf, &elf, &section_bases);

    assert_eq!(text_concat[0].src_reg(), BPF_PSEUDO_CALL);
    assert_eq!(text_concat[0].imm, 42);
}

/// Test 8 — `find_extern_func_btf_id` only matches FUNC types,
/// not other kinds that share the same name.
#[test]
fn find_extern_func_btf_id_filters_to_func_kind() {
    let mut strings: Vec<u8> = vec![0];
    let n_u64 = strings.len() as u32;
    strings.extend_from_slice(b"u64");
    strings.push(0);
    let n_foo = strings.len() as u32;
    strings.extend_from_slice(b"foo");
    strings.push(0);

    let mut types: Vec<u8> = Vec::new();
    types.extend_from_slice(&kfunc_btf_type_header(n_u64, 1, 0, 8));
    types.extend_from_slice(&64u32.to_le_bytes());
    // BTF_KIND_VAR (kind=14) named "foo".
    types.extend_from_slice(&kfunc_btf_type_header(n_foo, 14, 0, 1));
    types.extend_from_slice(&1u32.to_le_bytes());

    let mut blob: Vec<u8> = Vec::new();
    blob.extend_from_slice(&0xEB9F_u16.to_le_bytes());
    blob.push(1);
    blob.push(0);
    blob.extend_from_slice(&24u32.to_le_bytes());
    blob.extend_from_slice(&0u32.to_le_bytes());
    blob.extend_from_slice(&(types.len() as u32).to_le_bytes());
    blob.extend_from_slice(&(types.len() as u32).to_le_bytes());
    blob.extend_from_slice(&(strings.len() as u32).to_le_bytes());
    blob.extend_from_slice(&types);
    blob.extend_from_slice(&strings);

    let btf = Btf::from_bytes(&blob).expect("parse btf");
    // VAR id is not returned (kind filter rejects).
    assert_eq!(find_extern_func_btf_id(&btf, "foo"), None);
    // Name not in BTF returns None.
    assert_eq!(find_extern_func_btf_id(&btf, "absent"), None);
}

/// Test 1 — happy path: a `BPF_PSEUDO_CALL imm=-1` against an
/// `STT_FUNC` symbol gets `imm` rewritten to point at the
/// callee entry PC.
#[test]
fn patch_subprog_calls_happy_path_rewrites_imm() {
    let callee_name = "my_subprog";
    let mut strtab: Vec<u8> = vec![0];
    let name_off = strtab.len() as u32;
    strtab.extend_from_slice(callee_name.as_bytes());
    strtab.push(0);

    // Symbol 1: STT_FUNC, defined in section 1 (.text), st_value
    // = 16 bytes (the third instruction slot, a callee entry two
    // 8-byte slots after the EXIT terminator of the caller).
    let callee_st_value: u64 = 16;
    let mut symtab: Vec<u8> = Vec::new();
    symtab.extend_from_slice(&elf64_sym(0, 0, 0, 0, 0));
    symtab.extend_from_slice(&elf64_sym(
        name_off,
        st_info(syms::STB_GLOBAL, syms::STT_FUNC),
        1, // st_shndx = section 1 (.text)
        callee_st_value,
        0,
    ));

    // Text layout:
    //   pc=0: caller's `BPF_PSEUDO_CALL imm=-1`.
    //   pc=1: caller's EXIT.
    //   pc=2: callee entry (NOP placeholder; real BPF would have
    //         the function body here, but the patcher only
    //         consults the call-site instruction).
    let mut text: Vec<u8> = Vec::new();
    text.extend_from_slice(&pre_reloc_subprog_call_bytes());
    text.extend_from_slice(&kfunc_exit_bytes());
    text.extend_from_slice(&subprog_nop_bytes());
    let rel_data: Vec<u8> = elf64_rel(0, 1, 10).to_vec();

    let blob = build_elf64(
        vec![
            SecSpec::new(".text", sh::SHT_PROGBITS)
                .flags(sh::SHF_EXECINSTR.into())
                .data(text),
            SecSpec::new(".strtab", sh::SHT_STRTAB).data(strtab),
            SecSpec::new(".symtab", sh::SHT_SYMTAB)
                .data(symtab)
                .link(2)
                .entsize(24),
            SecSpec::new(".rel.text", sh::SHT_REL)
                .data(rel_data)
                .link(3)
                .info(1)
                .entsize(16),
        ],
        h::EM_BPF,
        h::ET_REL,
    );
    let elf = goblin::elf::Elf::parse(&blob).expect("parse elf");

    let mut text_concat: Vec<BpfInsn> = vec![
        BpfInsn::from_le_bytes(pre_reloc_subprog_call_bytes()),
        BpfInsn::from_le_bytes(kfunc_exit_bytes()),
        BpfInsn::from_le_bytes(subprog_nop_bytes()),
    ];
    let mut section_bases: HashMap<u32, usize> = HashMap::new();
    section_bases.insert(1, 0);

    assert_eq!(text_concat[0].imm, -1);
    patch_subprog_calls(&mut text_concat, &elf, &section_bases);

    // callee_pc = base(.text) + st_value/8 = 0 + 16/8 = 2.
    // call_pc = 0. new_imm = 2 - 0 - 1 = 1. After patch, the
    // analyzer's `pc + 1 + imm` = 0 + 1 + 1 = 2 = callee entry PC.
    assert_eq!(
        text_concat[0].imm, 1,
        "imm patched to callee_pc - call_pc - 1"
    );
    assert_eq!(
        text_concat[0].src_reg(),
        BPF_PSEUDO_CALL,
        "src_reg untouched (subprog calls keep BPF_PSEUDO_CALL)"
    );
    assert_eq!(text_concat[0].code, 0x85, "opcode untouched");
}

/// Test 2 — non-`-1` imm: a static-subprog call already carrying
/// the correct PC-relative offset must NOT be patched.
#[test]
fn patch_subprog_calls_skips_non_minus_one_imm() {
    let callee_name = "static_subprog";
    let mut strtab: Vec<u8> = vec![0];
    let name_off = strtab.len() as u32;
    strtab.extend_from_slice(callee_name.as_bytes());
    strtab.push(0);

    let mut symtab: Vec<u8> = Vec::new();
    symtab.extend_from_slice(&elf64_sym(0, 0, 0, 0, 0));
    symtab.extend_from_slice(&elf64_sym(
        name_off,
        st_info(syms::STB_LOCAL, syms::STT_FUNC),
        1,
        0,
        0,
    ));

    // Pre-set imm = 5 (already encoded by clang for a static
    // subprog). The patcher must leave it alone.
    let mut call = pre_reloc_subprog_call_bytes();
    call[4..8].copy_from_slice(&5i32.to_le_bytes());

    let mut text: Vec<u8> = Vec::new();
    text.extend_from_slice(&call);
    text.extend_from_slice(&kfunc_exit_bytes());
    let rel_data: Vec<u8> = elf64_rel(0, 1, 10).to_vec();

    let blob = build_elf64(
        vec![
            SecSpec::new(".text", sh::SHT_PROGBITS)
                .flags(sh::SHF_EXECINSTR.into())
                .data(text),
            SecSpec::new(".strtab", sh::SHT_STRTAB).data(strtab),
            SecSpec::new(".symtab", sh::SHT_SYMTAB)
                .data(symtab)
                .link(2)
                .entsize(24),
            SecSpec::new(".rel.text", sh::SHT_REL)
                .data(rel_data)
                .link(3)
                .info(1)
                .entsize(16),
        ],
        h::EM_BPF,
        h::ET_REL,
    );
    let elf = goblin::elf::Elf::parse(&blob).expect("parse elf");

    let mut text_concat: Vec<BpfInsn> = vec![
        BpfInsn::from_le_bytes(call),
        BpfInsn::from_le_bytes(kfunc_exit_bytes()),
    ];
    let mut section_bases: HashMap<u32, usize> = HashMap::new();
    section_bases.insert(1, 0);

    assert_eq!(text_concat[0].imm, 5);
    patch_subprog_calls(&mut text_concat, &elf, &section_bases);
    assert_eq!(text_concat[0].imm, 5, "non-(-1) imm must stay untouched");
}

/// Test 2b — `STT_SECTION` reloc (the `bpftool gen object` cross-
/// section shape): a subprog call whose reloc names the callee's
/// SECTION (not the callee FUNC) carries the callee's in-section
/// instruction index as `imm + 1`. Unlike the `STT_FUNC` non-`-1`
/// case (Test 2, left untouched), this one MUST be rebased into the
/// concatenated PC space — the pre-existing `imm` is relative to the
/// callee section start, not to our concatenation.
#[test]
fn patch_subprog_calls_rebases_section_symbol_call() {
    // One `.text` section, base 0, four insns:
    //   pc=0: NOP filler (so the call is not at pc 0 and the base
    //         offset in the new_imm arithmetic is observable).
    //   pc=1: the `BPF_PSEUDO_CALL` (reloc target).
    //   pc=2: EXIT.
    //   pc=3: callee entry.
    // The `.text` SECTION symbol has st_value=0; the callee's
    // in-section index (3) is carried as `imm = 3 - 1 = 2`.
    // Section symbols conventionally carry an empty name; the patcher
    // keys on the symbol's TYPE + shndx, not its name.
    let strtab: Vec<u8> = vec![0];
    let name_off = 0u32;

    let mut symtab: Vec<u8> = Vec::new();
    symtab.extend_from_slice(&elf64_sym(0, 0, 0, 0, 0));
    symtab.extend_from_slice(&elf64_sym(
        name_off,
        st_info(syms::STB_LOCAL, syms::STT_SECTION),
        1, // st_shndx = section 1 (.text)
        0, // st_value = 0 (section symbol)
        0,
    ));

    let mut call = pre_reloc_subprog_call_bytes();
    call[4..8].copy_from_slice(&2i32.to_le_bytes()); // imm = callee_index - 1

    let mut text: Vec<u8> = Vec::new();
    text.extend_from_slice(&subprog_nop_bytes());
    text.extend_from_slice(&call);
    text.extend_from_slice(&kfunc_exit_bytes());
    text.extend_from_slice(&subprog_nop_bytes());
    // Reloc targets the call at byte offset 8 (pc=1), symbol index 1.
    let rel_data: Vec<u8> = elf64_rel(8, 1, 10).to_vec();

    let blob = build_elf64(
        vec![
            SecSpec::new(".text", sh::SHT_PROGBITS)
                .flags(sh::SHF_EXECINSTR.into())
                .data(text),
            SecSpec::new(".strtab", sh::SHT_STRTAB).data(strtab),
            SecSpec::new(".symtab", sh::SHT_SYMTAB)
                .data(symtab)
                .link(2)
                .entsize(24),
            SecSpec::new(".rel.text", sh::SHT_REL)
                .data(rel_data)
                .link(3)
                .info(1)
                .entsize(16),
        ],
        h::EM_BPF,
        h::ET_REL,
    );
    let elf = goblin::elf::Elf::parse(&blob).expect("parse elf");

    let mut text_concat: Vec<BpfInsn> = vec![
        BpfInsn::from_le_bytes(subprog_nop_bytes()),
        BpfInsn::from_le_bytes(call),
        BpfInsn::from_le_bytes(kfunc_exit_bytes()),
        BpfInsn::from_le_bytes(subprog_nop_bytes()),
    ];
    let mut section_bases: HashMap<u32, usize> = HashMap::new();
    section_bases.insert(1, 0);

    assert_eq!(text_concat[1].imm, 2);
    patch_subprog_calls(&mut text_concat, &elf, &section_bases);

    // callee_pc = base(.text=0) + (imm + 1) = 0 + 3 = 3.
    // call_pc = 1. new_imm = 3 - 1 - 1 = 1. `pc + 1 + imm` =
    // 1 + 1 + 1 = 3 = callee entry.
    assert_eq!(
        text_concat[1].imm, 1,
        "STT_SECTION call rebased: imm = callee_pc - call_pc - 1"
    );
    assert_eq!(
        text_concat[1].src_reg(),
        BPF_PSEUDO_CALL,
        "src_reg untouched"
    );
}

/// Test 3 — `STT_NOTYPE` extern symbol (the kfunc shape) must
/// NOT trigger subprog patching. `patch_kfunc_calls` owns that
/// pipeline; a subprog patch here would corrupt the BTF id.
#[test]
fn patch_subprog_calls_skips_stt_notype_symbol() {
    let kf_name = "bpf_some_kfunc";
    let mut strtab: Vec<u8> = vec![0];
    let name_off = strtab.len() as u32;
    strtab.extend_from_slice(kf_name.as_bytes());
    strtab.push(0);

    let mut symtab: Vec<u8> = Vec::new();
    symtab.extend_from_slice(&elf64_sym(0, 0, 0, 0, 0));
    // STT_NOTYPE + SHN_UNDEF — the extern kfunc shape.
    symtab.extend_from_slice(&elf64_sym(
        name_off,
        st_info(syms::STB_GLOBAL, syms::STT_NOTYPE),
        0,
        0,
        0,
    ));

    let mut text: Vec<u8> = Vec::new();
    text.extend_from_slice(&pre_reloc_subprog_call_bytes());
    text.extend_from_slice(&kfunc_exit_bytes());
    let rel_data: Vec<u8> = elf64_rel(0, 1, 10).to_vec();

    let blob = build_elf64(
        vec![
            SecSpec::new(".text", sh::SHT_PROGBITS)
                .flags(sh::SHF_EXECINSTR.into())
                .data(text),
            SecSpec::new(".strtab", sh::SHT_STRTAB).data(strtab),
            SecSpec::new(".symtab", sh::SHT_SYMTAB)
                .data(symtab)
                .link(2)
                .entsize(24),
            SecSpec::new(".rel.text", sh::SHT_REL)
                .data(rel_data)
                .link(3)
                .info(1)
                .entsize(16),
        ],
        h::EM_BPF,
        h::ET_REL,
    );
    let elf = goblin::elf::Elf::parse(&blob).expect("parse elf");

    let mut text_concat: Vec<BpfInsn> = vec![
        BpfInsn::from_le_bytes(pre_reloc_subprog_call_bytes()),
        BpfInsn::from_le_bytes(kfunc_exit_bytes()),
    ];
    let mut section_bases: HashMap<u32, usize> = HashMap::new();
    section_bases.insert(1, 0);

    patch_subprog_calls(&mut text_concat, &elf, &section_bases);
    assert_eq!(
        text_concat[0].imm, -1,
        "STT_NOTYPE / SHN_UNDEF kfunc shape must not be touched"
    );
}

/// Test 4 — symbol's section is NOT in `section_bases`. A
/// subprog defined in a section we did not concatenate must
/// not be patched: we cannot compute a callee PC.
#[test]
fn patch_subprog_calls_skips_callee_section_outside_section_bases() {
    let callee_name = "subprog_in_other_section";
    let mut strtab: Vec<u8> = vec![0];
    let name_off = strtab.len() as u32;
    strtab.extend_from_slice(callee_name.as_bytes());
    strtab.push(0);

    // Symbol points at section 5 (.other) which we will NOT
    // include in `section_bases`.
    let mut symtab: Vec<u8> = Vec::new();
    symtab.extend_from_slice(&elf64_sym(0, 0, 0, 0, 0));
    symtab.extend_from_slice(&elf64_sym(
        name_off,
        st_info(syms::STB_GLOBAL, syms::STT_FUNC),
        5,
        0,
        0,
    ));

    let mut text: Vec<u8> = Vec::new();
    text.extend_from_slice(&pre_reloc_subprog_call_bytes());
    text.extend_from_slice(&kfunc_exit_bytes());
    let rel_data: Vec<u8> = elf64_rel(0, 1, 10).to_vec();

    let blob = build_elf64(
        vec![
            SecSpec::new(".text", sh::SHT_PROGBITS)
                .flags(sh::SHF_EXECINSTR.into())
                .data(text),
            SecSpec::new(".strtab", sh::SHT_STRTAB).data(strtab),
            SecSpec::new(".symtab", sh::SHT_SYMTAB)
                .data(symtab)
                .link(2)
                .entsize(24),
            SecSpec::new(".rel.text", sh::SHT_REL)
                .data(rel_data)
                .link(3)
                .info(1)
                .entsize(16),
            SecSpec::new(".other", sh::SHT_PROGBITS).data(vec![0u8; 8]),
        ],
        h::EM_BPF,
        h::ET_REL,
    );
    let elf = goblin::elf::Elf::parse(&blob).expect("parse elf");

    let mut text_concat: Vec<BpfInsn> = vec![
        BpfInsn::from_le_bytes(pre_reloc_subprog_call_bytes()),
        BpfInsn::from_le_bytes(kfunc_exit_bytes()),
    ];
    // section_bases includes only section 1 (.text), NOT section 5.
    let mut section_bases: HashMap<u32, usize> = HashMap::new();
    section_bases.insert(1, 0);

    patch_subprog_calls(&mut text_concat, &elf, &section_bases);
    assert_eq!(
        text_concat[0].imm, -1,
        "callee section outside section_bases must skip patching"
    );
}

/// Test 1 — happy path. Symbol is `STT_FUNC` global non-extern,
/// name is on `ALLOC_SUBPROG_NAMES`, call is `BPF_PSEUDO_CALL`.
/// Must emit exactly one [`SubprogReturn`] at the call PC.
#[test]
fn build_subprog_returns_happy_path_emits_one() {
    let (blob, text_concat, section_bases) = build_subprog_test_scaffold(
        "scx_alloc_internal",
        st_info(syms::STB_GLOBAL, syms::STT_FUNC),
        1, // st_shndx — .text at shdr[1]
        pseudo_call_bytes(123),
    );
    let elf = goblin::elf::Elf::parse(&blob).expect("parse elf");
    let out = build_subprog_returns(&text_concat, &elf, &section_bases);
    assert_eq!(out.len(), 1, "happy path: expected 1 entry, got {out:?}");
    assert_eq!(
        out[0].insn_offset, 0,
        "SubprogReturn must point at the call PC"
    );
}

/// Test 2 — gate skip: `BPF_PSEUDO_KFUNC_CALL` site. Even though
/// the symbol is `STT_FUNC` and the name is on the allowlist,
/// the call's `src_reg = 2` (kfunc) must be rejected. Kfunc
/// arena allocators are tagged via
/// [`crate::monitor::cast_analysis::ARENA_ALLOC_KFUNC_NAMES`]
/// inside [`crate::monitor::cast_analysis::Analyzer::handle_kfunc_call`],
/// not via SubprogReturn.
#[test]
fn build_subprog_returns_skips_pseudo_kfunc_call() {
    let (blob, text_concat, section_bases) = build_subprog_test_scaffold(
        "scx_alloc_internal",
        st_info(syms::STB_GLOBAL, syms::STT_FUNC),
        1,
        pseudo_kfunc_call_bytes(0),
    );
    let elf = goblin::elf::Elf::parse(&blob).expect("parse elf");
    let out = build_subprog_returns(&text_concat, &elf, &section_bases);
    assert!(
        out.is_empty(),
        "BPF_PSEUDO_KFUNC_CALL must not seed a SubprogReturn: {out:?}"
    );
}

/// Test 3 — gate skip: `STT_OBJECT` symbol. A data symbol
/// (`STT_OBJECT`) referenced by a reloc on a call site is
/// malformed input — the relocation walks over a call PC but
/// the resolved symbol is not a subprog. The
/// `sym.st_type() == STT_FUNC` gate must reject it.
#[test]
fn build_subprog_returns_skips_stt_object() {
    let (blob, text_concat, section_bases) = build_subprog_test_scaffold(
        "scx_alloc_internal",
        st_info(syms::STB_GLOBAL, syms::STT_OBJECT),
        1,
        pseudo_call_bytes(0),
    );
    let elf = goblin::elf::Elf::parse(&blob).expect("parse elf");
    let out = build_subprog_returns(&text_concat, &elf, &section_bases);
    assert!(
        out.is_empty(),
        "STT_OBJECT symbol must not seed a SubprogReturn: {out:?}"
    );
}

/// Test 4 — gate skip: `STT_FUNC` symbol whose name is NOT on
/// `ALLOC_SUBPROG_NAMES`. A regular BPF-to-BPF call to a
/// non-allocator subprog must not seed an arena tag. The
/// allowlist keeps the arena finding path strictly scoped.
#[test]
fn build_subprog_returns_skips_non_allowlist_name() {
    let (blob, text_concat, section_bases) = build_subprog_test_scaffold(
        "ktstr_some_unrelated_helper",
        st_info(syms::STB_GLOBAL, syms::STT_FUNC),
        1,
        pseudo_call_bytes(0),
    );
    let elf = goblin::elf::Elf::parse(&blob).expect("parse elf");
    let out = build_subprog_returns(&text_concat, &elf, &section_bases);
    assert!(
        out.is_empty(),
        "non-allowlist subprog name must not seed a SubprogReturn: {out:?}"
    );
}

/// Gate 1 (R_BPF_64_64 type): a relocation whose `r_type` is
/// not `R_BPF_64_64` (= 1) is silently dropped — the function
/// produces no `DatasecPointer` even though every other gate
/// would pass.
#[test]
fn build_datasec_pointers_rejects_non_r_bpf_64_64() {
    let (blob, btf_blob, text_concat, section_bases) = build_datasec_test_scaffold(
        ".bss",
        ".bss",
        10, // r_type != R_BPF_64_64 (= 1)
        0,
        0,
        1, // st_shndx = .bss (idx 1)
        st_info(syms::STB_GLOBAL, syms::STT_OBJECT),
        0,
    );
    let elf = goblin::elf::Elf::parse(&blob).expect("parse elf");
    let btf = Btf::from_bytes(&btf_blob).expect("parse btf");
    let out = build_datasec_pointers(&text_concat, &btf, &elf, &section_bases);
    assert!(out.is_empty(), "non-R_BPF_64_64 reloc must be skipped");
}

/// Gate 2 (`r_offset` alignment): a relocation whose `r_offset`
/// is not a multiple of 8 cannot reference an LD_IMM64
/// instruction (BPF instructions are 8-byte aligned). The
/// alignment gate fires before any other check.
#[test]
fn build_datasec_pointers_rejects_non_multiple_of_8_offset() {
    let (blob, btf_blob, text_concat, section_bases) = build_datasec_test_scaffold(
        ".bss",
        ".bss",
        1,
        4, // r_offset = 4 (not a multiple of 8)
        0,
        1,
        st_info(syms::STB_GLOBAL, syms::STT_OBJECT),
        0,
    );
    let elf = goblin::elf::Elf::parse(&blob).expect("parse elf");
    let btf = Btf::from_bytes(&btf_blob).expect("parse btf");
    let out = build_datasec_pointers(&text_concat, &btf, &elf, &section_bases);
    assert!(
        out.is_empty(),
        "r_offset=4 (not multiple of 8) must be rejected"
    );
}

/// Gate 3 (`r_offset` past section end): a relocation whose
/// `r_offset >= section_byte_size` cannot possibly land on a
/// real instruction. The bounds gate fires.
#[test]
fn build_datasec_pointers_rejects_offset_past_section_size() {
    // Text section size = 24 bytes (3 BPF instructions). An
    // r_offset of 100 is far past the end and must be rejected.
    let (blob, btf_blob, text_concat, section_bases) = build_datasec_test_scaffold(
        ".bss",
        ".bss",
        1,
        100, // r_offset >= section_byte_size (= 24)
        0,
        1,
        st_info(syms::STB_GLOBAL, syms::STT_OBJECT),
        0,
    );
    let elf = goblin::elf::Elf::parse(&blob).expect("parse elf");
    let btf = Btf::from_bytes(&btf_blob).expect("parse btf");
    let out = build_datasec_pointers(&text_concat, &btf, &elf, &section_bases);
    assert!(
        out.is_empty(),
        "r_offset past section size must be rejected"
    );
}

/// Gate 4 (instruction opcode): a relocation that lands on an
/// instruction whose `code` byte is not `BPF_LD_IMM64` (= 0x18)
/// is silently dropped. The renderer relies on the LD_IMM64
/// arm to apply datasec annotations; a reloc on an EXIT or
/// LDX would mis-route the analyzer state.
#[test]
fn build_datasec_pointers_rejects_non_ld_imm64_opcode() {
    // r_offset = 16 → instruction index 2 (the EXIT slot, not
    // an LD_IMM64). The opcode-byte gate fires.
    let (blob, btf_blob, text_concat, section_bases) = build_datasec_test_scaffold(
        ".bss",
        ".bss",
        1,
        16, // EXIT slot, not LD_IMM64
        0,
        1,
        st_info(syms::STB_GLOBAL, syms::STT_OBJECT),
        0,
    );
    let elf = goblin::elf::Elf::parse(&blob).expect("parse elf");
    let btf = Btf::from_bytes(&btf_blob).expect("parse btf");
    let out = build_datasec_pointers(&text_concat, &btf, &elf, &section_bases);
    assert!(
        out.is_empty(),
        "reloc on non-LD_IMM64 opcode must be rejected"
    );
}

/// Gate 5 (symbol section binding): symbols with `st_shndx`
/// set to `SHN_UNDEF` (0), `SHN_ABS` (0xFFF1), or `SHN_COMMON`
/// (0xFFF2) are not bound to a real section index; the
/// function rejects all three.
#[test]
fn build_datasec_pointers_rejects_special_section_index_symbols() {
    for shndx in [0u16, 0xFFF1, 0xFFF2] {
        let (blob, btf_blob, text_concat, section_bases) = build_datasec_test_scaffold(
            ".bss",
            ".bss",
            1,
            0,
            0,
            shndx,
            st_info(syms::STB_GLOBAL, syms::STT_OBJECT),
            0,
        );
        let elf = goblin::elf::Elf::parse(&blob).expect("parse elf");
        let btf = Btf::from_bytes(&btf_blob).expect("parse btf");
        let out = build_datasec_pointers(&text_concat, &btf, &elf, &section_bases);
        assert!(
            out.is_empty(),
            "symbol with st_shndx={shndx:#x} must be rejected"
        );
    }
}

/// Gate 6 (BTF datasec lookup): a section name that resolves
/// in the ELF but does NOT exist as a `BTF_KIND_DATASEC` in the
/// program BTF is rejected. Even if the section name is well-
/// formed (`.bss`), without a matching BTF datasec the
/// annotation cannot be emitted — the analyzer would have no
/// VarSecinfo entries to walk.
#[test]
fn build_datasec_pointers_rejects_section_not_in_btf() {
    // ELF section name = `.bss`, BTF datasec name = `.rodata`.
    // The BTF lookup at the section name `.bss` finds no
    // matching datasec → drop.
    let (blob, btf_blob, text_concat, section_bases) = build_datasec_test_scaffold(
        ".bss",
        ".rodata", // BTF datasec name mismatches ELF section name
        1,
        0,
        0,
        1,
        st_info(syms::STB_GLOBAL, syms::STT_OBJECT),
        0,
    );
    let elf = goblin::elf::Elf::parse(&blob).expect("parse elf");
    let btf = Btf::from_bytes(&btf_blob).expect("parse btf");
    let out = build_datasec_pointers(&text_concat, &btf, &elf, &section_bases);
    assert!(
        out.is_empty(),
        "section name not in BTF as DATASEC must be rejected"
    );
}

/// Gate 7 (`sym.st_value` overflow): if `sym.st_value`
/// exceeds `u32::MAX`, the offset cannot be represented in the
/// `base_offset: u32` field of [`DatasecPointer`]. The gate
/// rejects.
#[test]
fn build_datasec_pointers_rejects_st_value_past_u32_max() {
    let (blob, btf_blob, text_concat, section_bases) = build_datasec_test_scaffold(
        ".bss",
        ".bss",
        1,
        0,
        (u32::MAX as u64) + 1, // st_value > u32::MAX
        1,
        st_info(syms::STB_GLOBAL, syms::STT_OBJECT),
        0,
    );
    let elf = goblin::elf::Elf::parse(&blob).expect("parse elf");
    let btf = Btf::from_bytes(&btf_blob).expect("parse btf");
    let out = build_datasec_pointers(&text_concat, &btf, &elf, &section_bases);
    assert!(out.is_empty(), "sym.st_value > u32::MAX must be rejected");
}

/// Gate 8 (happy path): every gate passes, the function emits
/// exactly one [`DatasecPointer`] with the expected
/// `insn_offset`, `datasec_type_id`, and `base_offset`.
/// The `base_offset` is the sum of `insn.imm` and
/// `sym.st_value`, mirroring the libbpf convention for
/// `STT_OBJECT` symbols carrying the per-variable offset in
/// `st_value` and `STT_SECTION` symbols using `imm`.
#[test]
fn build_datasec_pointers_happy_path_emits_pointer() {
    // `imm = 16`, `st_value = 0`: STT_SECTION-style
    // pre-relocation form where the byte offset of the
    // referenced global is encoded in the LD_IMM64 imm field.
    let (blob, btf_blob, text_concat, section_bases) = build_datasec_test_scaffold(
        ".bss",
        ".bss",
        1, // R_BPF_64_64
        0, // r_offset = 0 (LD_IMM64 first slot)
        0, // st_value = 0
        1, // st_shndx = .bss (idx 1)
        st_info(syms::STB_GLOBAL, syms::STT_OBJECT),
        16, // LD_IMM64 imm = 16 (offset within .bss)
    );
    let elf = goblin::elf::Elf::parse(&blob).expect("parse elf");
    let btf = Btf::from_bytes(&btf_blob).expect("parse btf");
    let out = build_datasec_pointers(&text_concat, &btf, &elf, &section_bases);
    assert_eq!(out.len(), 1, "all gates pass → exactly one entry");
    assert_eq!(out[0].insn_offset, 0, "PC = base + r_offset/8 = 0");
    assert_eq!(
        out[0].datasec_type_id, 2,
        "datasec id is 2 (per build_datasec_btf_blob)"
    );
    assert_eq!(
        out[0].base_offset, 16,
        "base_offset = imm (16) + st_value (0) = 16"
    );
}

/// `find_datasec_btf_id` filters its results to
/// `BTF_KIND_DATASEC` only — a name shared by a `BTF_KIND_VAR`
/// or `BTF_KIND_INT` does not match. Mirrors the kind-filter
/// invariant in [`find_extern_func_btf_id_filters_to_func_kind`]
/// for the kfunc helper.
#[test]
fn find_datasec_btf_id_filters_to_datasec_kind() {
    // Build a BTF with three types named `.bss`:
    //   id 1: BTF_KIND_INT named ".bss" (size=4, bits=32)
    //   id 2: BTF_KIND_VAR named ".bss" (linkage=1)
    //   id 3: BTF_KIND_DATASEC named ".bss" (size=8)
    // The lookup must return id 3 — not id 1 (Int) or id 2
    // (Var) — even though all three share the same name.
    let mut strings: Vec<u8> = vec![0];
    let n_bss = strings.len() as u32;
    strings.extend_from_slice(b".bss");
    strings.push(0);

    let mut types: Vec<u8> = Vec::new();
    // id 1: INT
    types.extend_from_slice(&kfunc_btf_type_header(n_bss, 1, 0, 4));
    let int_data: u32 = 32;
    types.extend_from_slice(&int_data.to_le_bytes());
    // id 2: VAR (kind=14, vlen=0). size_or_type = wrapped int id (1).
    types.extend_from_slice(&kfunc_btf_type_header(n_bss, 14, 0, 1));
    let var_linkage: u32 = 1; // global
    types.extend_from_slice(&var_linkage.to_le_bytes());
    // id 3: DATASEC (kind=15, vlen=0). size_or_type = section
    // byte size (8).
    append_btf_datasec(&mut types, n_bss, 8, &[]);

    let mut blob: Vec<u8> = Vec::new();
    blob.extend_from_slice(&0xEB9F_u16.to_le_bytes());
    blob.push(1);
    blob.push(0);
    blob.extend_from_slice(&24u32.to_le_bytes());
    blob.extend_from_slice(&0u32.to_le_bytes());
    blob.extend_from_slice(&(types.len() as u32).to_le_bytes());
    blob.extend_from_slice(&(types.len() as u32).to_le_bytes());
    blob.extend_from_slice(&(strings.len() as u32).to_le_bytes());
    blob.extend_from_slice(&types);
    blob.extend_from_slice(&strings);

    let btf = Btf::from_bytes(&blob).expect("parse btf");
    // The datasec is id 3; the helper must filter past Int (1)
    // and Var (2) to return it.
    assert_eq!(
        find_datasec_btf_id(&btf, ".bss"),
        Some(3),
        "kind filter must skip past Int/Var to the Datasec",
    );
    // A name not present in the BTF returns None.
    assert_eq!(find_datasec_btf_id(&btf, ".rodata"), None);
}

/// `patch_kfunc_calls` already-relocated gate: a call whose
/// `src_reg == BPF_PSEUDO_KFUNC_CALL` (= 2) and `imm == 42`
/// has already been rewritten by some prior relocation pass
/// (e.g. an scheduler binary that captures a post-load BPF
/// object). The patcher must NOT overwrite the kernel BTF id
/// already in `imm` — doing so would replace a kernel id with
/// a program-BTF id, sending the analyzer to the wrong BTF
/// universe. Both `src_reg` and `imm` survive unmodified.
#[test]
fn patch_kfunc_calls_skips_already_relocated_src_reg() {
    let kf_name = "bpf_task_acquire";
    let (btf_blob, _expected_func_id, _t_id) = build_kfunc_btf_blob(kf_name);
    let btf = Btf::from_bytes(&btf_blob).expect("parse btf");

    let mut strtab: Vec<u8> = vec![0];
    let kf_str_off = strtab.len() as u32;
    strtab.extend_from_slice(kf_name.as_bytes());
    strtab.push(0);

    let mut symtab: Vec<u8> = Vec::new();
    symtab.extend_from_slice(&elf64_sym(0, 0, 0, 0, 0));
    symtab.extend_from_slice(&elf64_sym(
        kf_str_off,
        st_info(syms::STB_GLOBAL, syms::STT_NOTYPE),
        0,
        0,
        0,
    ));

    // Already-relocated kfunc call:
    //   code = 0x85 (BPF_JMP | BPF_CALL)
    //   dst = 0, src = BPF_PSEUDO_KFUNC_CALL (= 2)
    //   off = 0, imm = 42 (some kernel BTF id)
    // The packed regs byte: dst=0 (low 4) | src=2 (high 4) = 0x20.
    let already_relocated_call: [u8; 8] = [0x85, 0x20, 0x00, 0x00, 42, 0x00, 0x00, 0x00];

    let mut text: Vec<u8> = Vec::new();
    text.extend_from_slice(&already_relocated_call);
    text.extend_from_slice(&kfunc_exit_bytes());
    let rel_data: Vec<u8> = elf64_rel(0, 1, 10).to_vec();

    let blob = build_elf64(
        vec![
            SecSpec::new(".text", sh::SHT_PROGBITS)
                .flags(sh::SHF_EXECINSTR.into())
                .data(text),
            SecSpec::new(".strtab", sh::SHT_STRTAB).data(strtab),
            SecSpec::new(".symtab", sh::SHT_SYMTAB)
                .data(symtab)
                .link(2)
                .entsize(24),
            SecSpec::new(".rel.text", sh::SHT_REL)
                .data(rel_data)
                .link(3)
                .info(1)
                .entsize(16),
            SecSpec::new(".BTF", sh::SHT_PROGBITS).data(btf_blob),
        ],
        h::EM_BPF,
        h::ET_REL,
    );
    let elf = goblin::elf::Elf::parse(&blob).expect("parse elf");
    let mut text_concat: Vec<BpfInsn> = vec![
        BpfInsn::from_le_bytes(already_relocated_call),
        BpfInsn::from_le_bytes(kfunc_exit_bytes()),
    ];
    let mut section_bases: HashMap<u32, usize> = HashMap::new();
    section_bases.insert(1, 0);

    // Sanity: pre-call state matches the already-relocated form.
    assert_eq!(text_concat[0].code, 0x85);
    assert_eq!(text_concat[0].src_reg(), BPF_PSEUDO_KFUNC_CALL);
    assert_eq!(text_concat[0].imm, 42);

    patch_kfunc_calls(&mut text_concat, &btf, &elf, &section_bases);

    // Both fields must survive unmodified — the imm gate
    // (`imm != -1`) fires before any BTF lookup, preserving
    // the kernel id intact.
    assert_eq!(
        text_concat[0].src_reg(),
        BPF_PSEUDO_KFUNC_CALL,
        "src_reg must survive unmodified",
    );
    assert_eq!(
        text_concat[0].imm, 42,
        "imm must survive unmodified — kernel BTF id preserved",
    );
}

// ----- build_fwd_index tests -----------------------------------

/// Single BTF carrying complete `Type::Struct` entries indexes
/// each name to `(0, type_id)` — the fwd-resolution index is
/// the input the renderer's cross-BTF chase consults when a
/// `BTF_KIND_FWD` terminal needs a body lookup.
#[test]
fn build_fwd_index_indexes_single_btf_structs() {
    let mut strings = vec![0u8];
    let n_int = push_btf_name(&mut strings, "u64");
    let n_foo = push_btf_name(&mut strings, "foo");
    let n_bar = push_btf_name(&mut strings, "bar");
    let n_x = push_btf_name(&mut strings, "x");
    let types = vec![
        // id 1: u64 (skipped by the indexer — Int is not a Struct/Union/Typedef)
        SynKind::Int {
            name_off: n_int,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        // id 2: struct foo { u64 x @ 0 }
        SynKind::Struct {
            name_off: n_foo,
            size: 8,
            members: vec![SynMember {
                name_off: n_x,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        // id 3: struct bar { u64 x @ 0 }
        SynKind::Struct {
            name_off: n_bar,
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
    assert_eq!(
        index.get("foo"),
        Some(&FwdIndexEntry {
            btfs_idx: 0,
            type_id: 2,
        })
    );
    assert_eq!(
        index.get("bar"),
        Some(&FwdIndexEntry {
            btfs_idx: 0,
            type_id: 3,
        })
    );
    assert!(!index.contains_key("u64"), "Int names must not be indexed");
}
