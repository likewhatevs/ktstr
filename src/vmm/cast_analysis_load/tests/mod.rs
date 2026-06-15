//! Error-path coverage for the host-side BPF cast-analysis driver.
//!
//! Every public function in this module returns an empty
//! [`CastMap`] (or an empty `Vec<FuncEntry>`) on malformed input;
//! tests below exercise each early-return so an unintentionally
//! tightened gate (one that panics or aborts) shows up as a test
//! failure rather than a runtime crash on a stripped scheduler
//! binary.
//!
//! Fixtures are byte arrays built in-test with the
//! [`Elf64Builder`] helper — minimal ELF64 little-endian, only
//! the fields the cast loader inspects (section headers, the
//! shstrtab, `.bpf.objs`, `.BTF`, `.BTF.ext`, `SHF_EXECINSTR`
//! PROGBITS sections, and an optional `.symtab`/`.strtab`
//! pair). The builder produces blobs that pass
//! [`goblin::elf::Elf::parse`].
use super::*;
use goblin::elf::header as h;
use goblin::elf::section_header as sh;
use std::io::Write;

// ----- ELF fixture builder ----------------------------------------

/// One section in a synthetic ELF64. Matches the fields the
/// cast loader reads (`sh_type`, `sh_flags`, `sh_addr`,
/// `sh_offset`, `sh_size`) plus a `name` so the builder can
/// own the shstrtab.
struct SecSpec {
    name: &'static str,
    sh_type: u32,
    sh_flags: u64,
    sh_addr: u64,
    /// Section payload bytes. Empty payload still produces a
    /// section header (e.g. a NULL/SHT_NULL section).
    data: Vec<u8>,
    /// `sh_link` field (for symtab → strtab back-reference, or
    /// for a rel/rela section's symtab back-reference).
    sh_link: u32,
    /// `sh_info` field. For SHT_REL / SHT_RELA sections this is
    /// the index of the section being relocated (per ELF spec).
    /// For SHT_SYMTAB it is one greater than the index of the
    /// last local symbol; we leave it at 0 for tests since no
    /// production code in this module reads SYMTAB sh_info.
    sh_info: u32,
    /// `sh_entsize` field (24 for symtab on ELF64; 16 for SHT_REL,
    /// 24 for SHT_RELA).
    sh_entsize: u64,
}

impl SecSpec {
    fn new(name: &'static str, sh_type: u32) -> Self {
        Self {
            name,
            sh_type,
            sh_flags: 0,
            sh_addr: 0,
            data: Vec::new(),
            sh_link: 0,
            sh_info: 0,
            sh_entsize: 0,
        }
    }
    fn flags(mut self, f: u64) -> Self {
        self.sh_flags = f;
        self
    }
    fn data(mut self, d: Vec<u8>) -> Self {
        self.data = d;
        self
    }
    fn link(mut self, l: u32) -> Self {
        self.sh_link = l;
        self
    }
    fn info(mut self, i: u32) -> Self {
        self.sh_info = i;
        self
    }
    fn entsize(mut self, e: u64) -> Self {
        self.sh_entsize = e;
        self
    }
}

/// Build a synthetic ELF64 little-endian byte blob from a list
/// of [`SecSpec`]s.
///
/// Layout:
/// 1. ELF header at offset 0 (64 bytes).
/// 2. Section data packed back-to-back starting at offset 64.
/// 3. shstrtab (auto-generated) appended after the user data.
/// 4. Section header table appended last.
///
/// A leading `SHT_NULL` section is prepended automatically (the
/// ELF spec mandates `shdr[0]` is null). The shstrtab section is
/// appended automatically and `e_shstrndx` points at it.
fn build_elf64(sections: Vec<SecSpec>, e_machine: u16, e_type: u16) -> Vec<u8> {
    // 1. Build the shstrtab payload up front so each section's
    //    sh_name offset is known.
    let mut shstrtab: Vec<u8> = vec![0u8]; // ELF: index 0 is the empty string.
    let null_name_off = 0u32;
    let mut sec_name_offs: Vec<u32> = Vec::new();
    for s in &sections {
        sec_name_offs.push(shstrtab.len() as u32);
        shstrtab.extend_from_slice(s.name.as_bytes());
        shstrtab.push(0);
    }
    let shstrtab_self_name_off = shstrtab.len() as u32;
    shstrtab.extend_from_slice(b".shstrtab");
    shstrtab.push(0);

    // 2. ELF64 sizes per goblin: SIZEOF_EHDR=64, SIZEOF_SHDR=64.
    let ehdr_size: usize = 64;
    let shdr_size: usize = 64;

    // 3. Lay out section data at growing file offsets after the
    //    header. NULL section (index 0) has zero size and is at
    //    offset 0 (convention).
    let mut data_blob: Vec<u8> = Vec::new();
    let mut sec_file_off: Vec<u64> = Vec::new();
    // Index 0: NULL — placed at offset 0 with size 0.
    sec_file_off.push(0);
    // Indices 1..N: user sections, packed after the ELF header.
    let mut cursor: u64 = ehdr_size as u64;
    for s in &sections {
        sec_file_off.push(cursor);
        data_blob.extend_from_slice(&s.data);
        cursor += s.data.len() as u64;
    }
    // shstrtab section file offset.
    let shstrtab_file_off = cursor;
    data_blob.extend_from_slice(&shstrtab);
    cursor += shstrtab.len() as u64;
    // Section header table file offset.
    let shoff = cursor;

    // 4. Total section count: NULL + user + shstrtab.
    let shnum = (1 + sections.len() + 1) as u16;
    let shstrndx = (1 + sections.len()) as u16;

    // 5. ELF header.
    let mut blob: Vec<u8> = Vec::with_capacity(ehdr_size);
    // e_ident[16]
    blob.extend_from_slice(h::ELFMAG); // \x7FELF
    blob.push(h::ELFCLASS64); // EI_CLASS=2
    blob.push(h::ELFDATA2LSB); // EI_DATA=1
    blob.push(h::EV_CURRENT); // EI_VERSION=1
    blob.push(0); // EI_OSABI=0 (System V)
    blob.push(0); // EI_ABIVERSION
    // EI_PAD: 7 bytes of 0.
    blob.extend_from_slice(&[0u8; 7]);
    // e_type, e_machine, e_version
    blob.extend_from_slice(&e_type.to_le_bytes());
    blob.extend_from_slice(&e_machine.to_le_bytes());
    blob.extend_from_slice(&1u32.to_le_bytes()); // EV_CURRENT=1
    blob.extend_from_slice(&0u64.to_le_bytes()); // e_entry
    blob.extend_from_slice(&0u64.to_le_bytes()); // e_phoff (no program headers)
    blob.extend_from_slice(&shoff.to_le_bytes()); // e_shoff
    blob.extend_from_slice(&0u32.to_le_bytes()); // e_flags
    blob.extend_from_slice(&(ehdr_size as u16).to_le_bytes()); // e_ehsize
    blob.extend_from_slice(&0u16.to_le_bytes()); // e_phentsize
    blob.extend_from_slice(&0u16.to_le_bytes()); // e_phnum
    blob.extend_from_slice(&(shdr_size as u16).to_le_bytes()); // e_shentsize
    blob.extend_from_slice(&shnum.to_le_bytes()); // e_shnum
    blob.extend_from_slice(&shstrndx.to_le_bytes()); // e_shstrndx

    // 6. Section data + shstrtab payload.
    blob.extend_from_slice(&data_blob);

    // 7. Section header table.
    let mut write_shdr = |sh_name: u32,
                          sh_type: u32,
                          sh_flags: u64,
                          sh_addr: u64,
                          sh_offset: u64,
                          sh_size: u64,
                          sh_link: u32,
                          sh_info: u32,
                          sh_addralign: u64,
                          sh_entsize: u64| {
        blob.write_all(&sh_name.to_le_bytes()).unwrap();
        blob.write_all(&sh_type.to_le_bytes()).unwrap();
        blob.write_all(&sh_flags.to_le_bytes()).unwrap();
        blob.write_all(&sh_addr.to_le_bytes()).unwrap();
        blob.write_all(&sh_offset.to_le_bytes()).unwrap();
        blob.write_all(&sh_size.to_le_bytes()).unwrap();
        blob.write_all(&sh_link.to_le_bytes()).unwrap();
        blob.write_all(&sh_info.to_le_bytes()).unwrap();
        blob.write_all(&sh_addralign.to_le_bytes()).unwrap();
        blob.write_all(&sh_entsize.to_le_bytes()).unwrap();
    };
    // shdr[0] = NULL.
    write_shdr(null_name_off, sh::SHT_NULL, 0, 0, 0, 0, 0, 0, 0, 0);
    // User sections.
    for (i, s) in sections.iter().enumerate() {
        write_shdr(
            sec_name_offs[i],
            s.sh_type,
            s.sh_flags,
            s.sh_addr,
            sec_file_off[i + 1],
            s.data.len() as u64,
            s.sh_link,
            s.sh_info,
            1,
            s.sh_entsize,
        );
    }
    // shstrtab section.
    write_shdr(
        shstrtab_self_name_off,
        sh::SHT_STRTAB,
        0,
        0,
        shstrtab_file_off,
        shstrtab.len() as u64,
        0,
        0,
        1,
        0,
    );

    blob
}

/// Build an ELF64 symbol table entry (24 bytes, little-endian).
///
/// Layout per `goblin::elf::sym::sym64::Sym`:
/// `st_name(4) st_info(1) st_other(1) st_shndx(2) st_value(8) st_size(8)`.
fn elf64_sym(st_name: u32, st_info: u8, st_shndx: u16, st_value: u64, st_size: u64) -> [u8; 24] {
    let mut out = [0u8; 24];
    out[0..4].copy_from_slice(&st_name.to_le_bytes());
    out[4] = st_info;
    out[5] = 0; // st_other (visibility) = STV_DEFAULT
    out[6..8].copy_from_slice(&st_shndx.to_le_bytes());
    out[8..16].copy_from_slice(&st_value.to_le_bytes());
    out[16..24].copy_from_slice(&st_size.to_le_bytes());
    out
}

/// Pack the symbol-binding (high 4 bits) and symbol-type (low 4
/// bits) into the `st_info` byte. Mirrors `ELF64_ST_INFO(b,t)`
/// from the SysV ELF spec.
fn st_info(bind: u8, ty: u8) -> u8 {
    (bind << 4) | (ty & 0x0f)
}

// ----- BTF fixture builder ----------------------------------------

/// Build a minimal `.BTF` blob.
///
/// Mirrors the BTF wire format documented in
/// `include/uapi/linux/btf.h`: 24-byte header (magic, version,
/// flags, hdr_len, type_off, type_len, str_off, str_len) +
/// type section + string section. The `types` payload is opaque
/// to these tests — `btf_str_at` only consults the header
/// fields and the string section, so an empty type section is
/// fine.
fn build_btf_blob(types: &[u8], strings: &[u8]) -> Vec<u8> {
    let type_len = types.len() as u32;
    let str_len = strings.len() as u32;
    let mut blob = Vec::new();
    blob.write_all(&0xEB9F_u16.to_le_bytes()).unwrap(); // magic
    blob.push(1); // version
    blob.push(0); // flags
    blob.write_all(&24u32.to_le_bytes()).unwrap(); // hdr_len
    blob.write_all(&0u32.to_le_bytes()).unwrap(); // type_off
    blob.write_all(&type_len.to_le_bytes()).unwrap(); // type_len
    blob.write_all(&type_len.to_le_bytes()).unwrap(); // str_off (= type_len)
    blob.write_all(&str_len.to_le_bytes()).unwrap(); // str_len
    blob.extend_from_slice(types);
    blob.extend_from_slice(strings);
    blob
}

// ----- BPF instruction encoding helpers --------------------------
//
// `BpfInsn` exposes [`BpfInsn::new`] and [`BpfInsn::from_le_bytes`]
// but not a writer. The end-to-end tests below need wire bytes to
// populate `.text` sections, so we re-encode by mirroring the
// little-endian layout the decoder reads.
fn insn_to_bytes(i: BpfInsn) -> [u8; 8] {
    // `regs` field is private in production; rebuild the packed
    // byte from `dst_reg()` (low 4 bits) and `src_reg()` (high
    // 4 bits) — exactly the layout `BpfInsn::new` produces.
    let regs_byte = (i.dst_reg() & 0x0f) | ((i.src_reg() & 0x0f) << 4);
    let mut buf = [0u8; 8];
    buf[0] = i.code;
    buf[1] = regs_byte;
    buf[2..4].copy_from_slice(&i.off.to_le_bytes());
    buf[4..8].copy_from_slice(&i.imm.to_le_bytes());
    buf
}

fn insns_to_text_bytes(insns: &[BpfInsn]) -> Vec<u8> {
    let mut out = Vec::with_capacity(insns.len() * 8);
    for ins in insns {
        out.extend_from_slice(&insn_to_bytes(*ins));
    }
    out
}

// BPF opcode field values (kernel uapi `bpf.h`):
//   class low 3 bits: LDX=1, JMP=5
//   size bits 3..4: DW=0x18
//   mode bits 5..7: MEM=0x60
//   op  bits 4..7: EXIT=0x90
const OP_LDX_DW_MEM: u8 = 0x01 | 0x18 | 0x60; // 0x79
const OP_JMP_EXIT: u8 = 0x05 | 0x90; // 0x95

fn ldx_dw_mem(dst: u8, src: u8, off: i16) -> BpfInsn {
    BpfInsn::new(OP_LDX_DW_MEM, dst, src, off, 0)
}

fn exit_insn() -> BpfInsn {
    BpfInsn::new(OP_JMP_EXIT, 0, 0, 0, 0)
}

/// `BPF_ADDR_SPACE_CAST` for the arena-evidence mitigation: ALU64 | MOV | X
/// with `off=1, imm=1` is the as(1)→as(0) (arena→kernel) cast
/// the analyzer treats as `arena_confirmed` evidence on the
/// source's `(struct, field_offset)` slot.
fn addr_space_cast_insn(dst: u8, src: u8) -> BpfInsn {
    use libbpf_rs::libbpf_sys as bs;
    let code = (bs::BPF_ALU64 | bs::BPF_MOV | bs::BPF_X) as u8;
    BpfInsn::new(code, dst, src, 1, 1)
}

// ----- Synthesizers for full BTF (ints, structs, ptr, FuncProto, Func)
//
// The error-path tests above only need empty BTFs. The
// analyze_one_object_with_btf end-to-end test needs a real BTF
// whose types the analyzer can intersect. This builder mirrors
// `cast_analysis::tests::build_btf` in shape but is local to this
// module so the two test fixtures stay decoupled.
const SYN_BTF_KIND_INT: u32 = 1;
const SYN_BTF_KIND_PTR: u32 = 2;
const SYN_BTF_KIND_STRUCT: u32 = 4;
const SYN_BTF_KIND_FWD: u32 = 7;
const SYN_BTF_KIND_FUNC: u32 = 12;
const SYN_BTF_KIND_FUNC_PROTO: u32 = 13;

/// Append `name` plus a trailing NUL to `s`; return the offset
/// at which it was written. Standard BTF strtab convention.
fn push_btf_name(s: &mut Vec<u8>, name: &str) -> u32 {
    let off = s.len() as u32;
    s.extend_from_slice(name.as_bytes());
    s.push(0);
    off
}

/// Member of a synthetic struct (non-bitfield, byte-aligned).
#[derive(Clone, Copy)]
struct SynMember {
    name_off: u32,
    type_id: u32,
    byte_offset: u32,
}

/// FuncProto parameter record.
#[derive(Clone, Copy)]
struct SynParam {
    name_off: u32,
    type_id: u32,
}

enum SynKind {
    Int {
        name_off: u32,
        size: u32,
        encoding: u32,
        offset: u32,
        bits: u32,
    },
    Ptr {
        type_id: u32,
    },
    Struct {
        name_off: u32,
        size: u32,
        members: Vec<SynMember>,
    },
    /// `BTF_KIND_FWD` (kind 7) — forward declaration with no body.
    /// Layout per `include/uapi/linux/btf.h` is the 12-byte common
    /// `struct btf_type` header (name_off + info + size_type) with
    /// no trailing payload. `kind_flag` (bit 31 of info) is 0 for
    /// struct-kind, 1 for union-kind; `Fwd::is_struct()` /
    /// `Fwd::is_union()` in btf-rs read it back. The third u32 of
    /// the common header is unused for Fwd and emitted as 0.
    Fwd {
        name_off: u32,
        kind_flag: u32,
    },
    Func {
        name_off: u32,
        type_id: u32,
        linkage: u32,
    },
    FuncProto {
        return_type_id: u32,
        params: Vec<SynParam>,
    },
}

/// Encode `types` and `strings` into a BTF byte blob.
fn build_btf_full(types: &[SynKind], strings: &[u8]) -> Vec<u8> {
    let mut type_section = Vec::new();
    for ty in types {
        match ty {
            SynKind::Int {
                name_off,
                size,
                encoding,
                offset,
                bits,
            } => {
                type_section.extend_from_slice(&name_off.to_le_bytes());
                let info = (SYN_BTF_KIND_INT << 24) & 0x1f00_0000;
                type_section.extend_from_slice(&info.to_le_bytes());
                type_section.extend_from_slice(&size.to_le_bytes());
                let int_data = (*encoding << 24) | ((*offset & 0xff) << 16) | (*bits & 0xff);
                type_section.extend_from_slice(&int_data.to_le_bytes());
            }
            SynKind::Ptr { type_id } => {
                type_section.extend_from_slice(&0u32.to_le_bytes());
                let info = (SYN_BTF_KIND_PTR << 24) & 0x1f00_0000;
                type_section.extend_from_slice(&info.to_le_bytes());
                type_section.extend_from_slice(&type_id.to_le_bytes());
            }
            SynKind::Struct {
                name_off,
                size,
                members,
            } => {
                type_section.extend_from_slice(&name_off.to_le_bytes());
                let vlen = members.len() as u32;
                let info = ((SYN_BTF_KIND_STRUCT << 24) & 0x1f00_0000) | (vlen & 0xffff);
                type_section.extend_from_slice(&info.to_le_bytes());
                type_section.extend_from_slice(&size.to_le_bytes());
                for m in members {
                    type_section.extend_from_slice(&m.name_off.to_le_bytes());
                    type_section.extend_from_slice(&m.type_id.to_le_bytes());
                    let bit_off = m.byte_offset * 8;
                    type_section.extend_from_slice(&bit_off.to_le_bytes());
                }
            }
            SynKind::Fwd {
                name_off,
                kind_flag,
            } => {
                type_section.extend_from_slice(&name_off.to_le_bytes());
                // Fwd info: kind in bits 24-28, kind_flag in bit
                // 31 (struct-vs-union tag), no vlen. The third u32
                // of btf_type (size_type) is unused for Fwd.
                let info = ((SYN_BTF_KIND_FWD << 24) & 0x1f00_0000) | ((*kind_flag & 1) << 31);
                type_section.extend_from_slice(&info.to_le_bytes());
                type_section.extend_from_slice(&0u32.to_le_bytes());
            }
            SynKind::Func {
                name_off,
                type_id,
                linkage,
            } => {
                type_section.extend_from_slice(&name_off.to_le_bytes());
                let info = ((SYN_BTF_KIND_FUNC << 24) & 0x1f00_0000) | (*linkage & 0xffff);
                type_section.extend_from_slice(&info.to_le_bytes());
                type_section.extend_from_slice(&type_id.to_le_bytes());
            }
            SynKind::FuncProto {
                return_type_id,
                params,
            } => {
                type_section.extend_from_slice(&0u32.to_le_bytes());
                let vlen = params.len() as u32;
                let info = ((SYN_BTF_KIND_FUNC_PROTO << 24) & 0x1f00_0000) | (vlen & 0xffff);
                type_section.extend_from_slice(&info.to_le_bytes());
                type_section.extend_from_slice(&return_type_id.to_le_bytes());
                for p in params {
                    type_section.extend_from_slice(&p.name_off.to_le_bytes());
                    type_section.extend_from_slice(&p.type_id.to_le_bytes());
                }
            }
        }
    }
    // Header.
    let type_len = type_section.len() as u32;
    let str_len = strings.len() as u32;
    let mut blob = Vec::new();
    blob.write_all(&0xEB9F_u16.to_le_bytes()).unwrap();
    blob.push(1); // version
    blob.push(0); // flags
    blob.write_all(&24u32.to_le_bytes()).unwrap(); // hdr_len
    blob.write_all(&0u32.to_le_bytes()).unwrap(); // type_off
    blob.write_all(&type_len.to_le_bytes()).unwrap(); // type_len
    blob.write_all(&type_len.to_le_bytes()).unwrap(); // str_off
    blob.write_all(&str_len.to_le_bytes()).unwrap();
    blob.extend_from_slice(&type_section);
    blob.extend_from_slice(strings);
    blob
}

/// Build a `.BTF.ext` blob describing one `func_info` section
/// with `records` entries `(insn_off, type_id)`.
fn build_btf_ext(section_name_off: u32, records: &[(u32, u32)], record_size: u32) -> Vec<u8> {
    let header_len = 24u32;
    let info_len = 4 + 4 + 4 + records.len() as u32 * record_size;
    let mut info = Vec::new();
    info.extend_from_slice(&record_size.to_le_bytes());
    info.extend_from_slice(&section_name_off.to_le_bytes());
    info.extend_from_slice(&(records.len() as u32).to_le_bytes());
    for (insn_off, type_id) in records {
        info.extend_from_slice(&insn_off.to_le_bytes());
        info.extend_from_slice(&type_id.to_le_bytes());
        let pad = record_size.saturating_sub(8) as usize;
        info.extend(std::iter::repeat_n(0, pad));
    }
    let mut out = Vec::new();
    out.extend_from_slice(&0xEB9F_u16.to_le_bytes());
    out.push(1);
    out.push(0);
    out.extend_from_slice(&header_len.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes()); // func_info_off
    out.extend_from_slice(&info_len.to_le_bytes());
    out.extend_from_slice(&info_len.to_le_bytes()); // line_info_off (unused)
    out.extend_from_slice(&0u32.to_le_bytes()); // line_info_len
    out.extend_from_slice(&info);
    out
}

/// Construct a BPF object ELF with `.text`, `.BTF`, and
/// `.BTF.ext` sections — the canonical scx-built shape minus
/// the relocations the loader does not consume.
fn build_full_bpf_object_elf(text: Vec<u8>, btf: Vec<u8>, btf_ext: Vec<u8>) -> Vec<u8> {
    build_elf64(
        vec![
            SecSpec::new(".text", sh::SHT_PROGBITS)
                .flags(sh::SHF_EXECINSTR.into())
                .data(text),
            SecSpec::new(".BTF", sh::SHT_PROGBITS).data(btf),
            SecSpec::new(".BTF.ext", sh::SHT_PROGBITS).data(btf_ext),
        ],
        h::EM_BPF,
        h::ET_REL,
    )
}

// ----- Tests for cached_cast_analysis_for_scheduler --------------

/// Helper: build the same arena-cast end-to-end fixture used by
/// `cached_cast_analysis_recovers_arena_cast_end_to_end`,
/// returning the outer ELF bytes. Centralised so cache tests
/// share a fixture shape with the path-driven test.
fn build_recovers_arena_cast_outer_elf() -> Vec<u8> {
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
    build_elf64(
        vec![SecSpec::new(".bpf.objs", sh::SHT_PROGBITS).data(inner)],
        h::EM_X86_64,
        h::ET_REL,
    )
}

// ----- kfunc-relocation patcher tests ----------------------------
//
// Coverage strategy: for `patch_kfunc_calls` we run end-to-end
// tests that synthesize a complete BPF object (program text +
// BTF + .symtab/.strtab + a SHT_REL section) and feed the
// patcher the same `(text_concat, btf, elf, section_bases)`
// tuple `analyze_one_object_with_btf` produces. After patching we re-
// decode the call instruction and assert the analyzer-visible
// state — this is the exact contract `analyze_casts` consumes
// downstream, so any drift between the patcher and the analyzer
// surfaces here.

/// Encode a single BTF type record header. Mirrors the wire
/// format from linux uapi `btf.h`:
/// `name_off(4) info(4) size_or_type(4)`.
fn kfunc_btf_type_header(name_off: u32, kind: u32, vlen: u32, size_or_type: u32) -> [u8; 12] {
    let info = ((kind << 24) & 0x1f00_0000) | (vlen & 0xffff);
    let mut out = [0u8; 12];
    out[0..4].copy_from_slice(&name_off.to_le_bytes());
    out[4..8].copy_from_slice(&info.to_le_bytes());
    out[8..12].copy_from_slice(&size_or_type.to_le_bytes());
    out
}

/// Build a minimal `.BTF` blob containing a single extern FUNC
/// with a FuncProto that returns a struct pointer.
/// Returns the byte blob plus the BTF id of the extern Func
/// (always 5) and the BTF id of struct T (always 2).
fn build_kfunc_btf_blob(kf_name: &str) -> (Vec<u8>, u32, u32) {
    let mut strings: Vec<u8> = vec![0];
    let push_name = |s: &mut Vec<u8>, name: &str| -> u32 {
        let off = s.len() as u32;
        s.extend_from_slice(name.as_bytes());
        s.push(0);
        off
    };
    let n_u64 = push_name(&mut strings, "u64");
    let n_t = push_name(&mut strings, "T");
    let n_x = push_name(&mut strings, "x");
    let n_func = push_name(&mut strings, kf_name);

    let mut types: Vec<u8> = Vec::new();
    const BTF_KIND_INT: u32 = 1;
    const BTF_KIND_PTR: u32 = 2;
    const BTF_KIND_STRUCT: u32 = 4;
    const BTF_KIND_FUNC: u32 = 12;
    const BTF_KIND_FUNC_PROTO: u32 = 13;
    const BTF_FUNC_EXTERN: u32 = 2;

    // id 1: BTF_KIND_INT u64.
    types.extend_from_slice(&kfunc_btf_type_header(n_u64, BTF_KIND_INT, 0, 8));
    let int_data: u32 = 64;
    types.extend_from_slice(&int_data.to_le_bytes());

    // id 2: BTF_KIND_STRUCT T { u64 x @ 0 } size=8 vlen=1.
    types.extend_from_slice(&kfunc_btf_type_header(n_t, BTF_KIND_STRUCT, 1, 8));
    types.extend_from_slice(&n_x.to_le_bytes());
    types.extend_from_slice(&1u32.to_le_bytes());
    types.extend_from_slice(&0u32.to_le_bytes());

    // id 3: BTF_KIND_PTR -> id 2.
    types.extend_from_slice(&kfunc_btf_type_header(0, BTF_KIND_PTR, 0, 2));

    // id 4: BTF_KIND_FUNC_PROTO returning id 3, no params.
    types.extend_from_slice(&kfunc_btf_type_header(0, BTF_KIND_FUNC_PROTO, 0, 3));

    // id 5: BTF_KIND_FUNC kf_name -> id 4 (proto), linkage=extern.
    types.extend_from_slice(&kfunc_btf_type_header(
        n_func,
        BTF_KIND_FUNC,
        BTF_FUNC_EXTERN,
        4,
    ));

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
    (blob, 5, 2)
}

/// Build an ELF64 `Elf64_Rel` entry (16 bytes, little-endian).
/// `Elf64_Rel { r_offset(8), r_info(8) }` where
/// `r_info = (sym_idx << 32) | r_type`.
fn elf64_rel(r_offset: u64, sym_idx: u64, r_type: u32) -> [u8; 16] {
    let mut out = [0u8; 16];
    out[0..8].copy_from_slice(&r_offset.to_le_bytes());
    let r_info = (sym_idx << 32) | (r_type as u64);
    out[8..16].copy_from_slice(&r_info.to_le_bytes());
    out
}

/// Encode a `BPF_JMP|BPF_CALL` with the clang-emitted pre-
/// relocation kfunc form: `code=0x85`, `dst=0`,
/// `src=BPF_PSEUDO_CALL=1`, `off=0`, `imm=-1`.
fn pre_reloc_kfunc_call_bytes() -> [u8; 8] {
    [0x85, 0x10, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff]
}

/// Encode an EXIT instruction (`code=0x95`).
fn kfunc_exit_bytes() -> [u8; 8] {
    [0x95, 0, 0, 0, 0, 0, 0, 0]
}

// ----- patch_subprog_calls tests ---------------------------------
//
// Coverage strategy mirrors the kfunc patcher: each test
// synthesises a minimal BPF object (program text + symtab/strtab
// + SHT_REL section) and feeds the analyzer-visible tuple
// `(text_concat, elf, section_bases)` to
// [`patch_subprog_calls`]. The fixtures pin the gate set the
// function applies before rewriting `imm`:
//
//   1. Happy path — `BPF_PSEUDO_CALL`, `imm == -1`, defined
//      `STT_FUNC` symbol whose section we concatenated. After
//      patching, `imm` must equal `callee_pc - call_pc - 1` so
//      the analyzer's `pc + 1 + imm` lands on the callee entry.
//   2. Gate skip — non-`-1` imm. Static (file-local) subprog
//      calls already carry the correct PC-relative offset and
//      must not be touched.
//   3. Gate skip — `STT_NOTYPE` symbol (the kfunc shape).
//      `patch_kfunc_calls` handles that pipeline; a subprog
//      patch on a kfunc reloc would silently corrupt the imm.
//   4. Gate skip — symbol's section is NOT in `section_bases`.
//      A subprog defined in a section we did not concatenate
//      cannot be resolved to a callee PC and is skipped.
//
// These pin the false-negative-safe boundary: a regression in
// any gate either drops a subprog patch (caller_arg_types stays
// poisoned) or stomps a non-subprog call site with a bogus imm.

/// Encode a `BPF_JMP|BPF_CALL` with the pre-relocation global
/// subprog form: `code=0x85`, `dst=0`,
/// `src=BPF_PSEUDO_CALL=1`, `off=0`, `imm=-1`.
fn pre_reloc_subprog_call_bytes() -> [u8; 8] {
    [0x85, 0x10, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff]
}

/// Encode a NOP-shaped instruction (`BPF_ALU64 | BPF_MOV | BPF_X`,
/// dst=R0, src=R0). Suitable as filler insns for callee bodies in
/// fixtures — the patcher ignores everything but the tagged call
/// site.
fn subprog_nop_bytes() -> [u8; 8] {
    // BPF_ALU64 | BPF_MOV | BPF_X = 0x07 | 0xb0 | 0x08 = 0xbf.
    // dst=0, src=0, off=0, imm=0.
    [0xbf, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
}

// ----- build_subprog_returns tests -------------------------------
//
// Coverage strategy: each test synthesises a minimal BPF object
// (program text + symtab/strtab + SHT_REL section) and feeds the
// analyzer-visible tuple `(text_concat, elf, section_bases)` to
// [`build_subprog_returns`]. The four test cases pin the four
// gates the function applies before recording a [`SubprogReturn`]:
//
//   1. Happy path — symbol is `STT_FUNC` + non-`SHN_UNDEF` +
//      `BPF_PSEUDO_CALL` site + name on `ALLOC_SUBPROG_NAMES`.
//      The result must contain exactly one entry pointing at the
//      call PC.
//   2. Gate skip — `BPF_PSEUDO_KFUNC_CALL` (src_reg = 2). Kfunc
//      calls are handled by `handle_kfunc_call` separately; a
//      subprog-return seed at this site would mis-route the
//      arena tag.
//   3. Gate skip — `STT_OBJECT` symbol (data, not function). The
//      relocation might target a call PC by coincidence but the
//      symbol is not a subprog.
//   4. Gate skip — `STT_FUNC` symbol whose name is NOT on
//      `ALLOC_SUBPROG_NAMES`. A regular subprog call must not
//      seed an arena tag — the analyzer's `BPF_OP_CALL` arm
//      relies on the seed list to disambiguate.
//
// These four gates compose the "false-negative-safe" boundary of
// the SubprogReturn pipeline; a regression in any one of them
// would either drop allocator-call sites silently (false
// negative — surfaces as a missing chase) or seed an arena tag
// on an unrelated subprog call (false positive — produces a
// misleading render). The tests below pin each gate independently.

/// Encode a `BPF_PSEUDO_CALL` (src_reg = 1) call instruction:
/// `code=0x85`, `dst=0`, `src=1`, `off=0`, `imm=any`.
/// Mirrors clang's pre-relocation BPF-to-BPF call shape.
fn pseudo_call_bytes(imm: i32) -> [u8; 8] {
    let mut out = [0u8; 8];
    out[0] = 0x85; // BPF_JMP | BPF_CALL
    out[1] = 0x10; // dst=0, src=1 (BPF_PSEUDO_CALL)
    out[2..4].copy_from_slice(&0i16.to_le_bytes());
    out[4..8].copy_from_slice(&imm.to_le_bytes());
    out
}

/// Encode a `BPF_PSEUDO_KFUNC_CALL` (src_reg = 2) call
/// instruction. Used by the gate-skip test to confirm kfunc
/// call sites do not seed SubprogReturn entries.
fn pseudo_kfunc_call_bytes(imm: i32) -> [u8; 8] {
    let mut out = [0u8; 8];
    out[0] = 0x85;
    out[1] = 0x20; // dst=0, src=2 (BPF_PSEUDO_KFUNC_CALL)
    out[2..4].copy_from_slice(&0i16.to_le_bytes());
    out[4..8].copy_from_slice(&imm.to_le_bytes());
    out
}

/// Build a `(elf, text_concat, section_bases)` triple that
/// [`build_subprog_returns`] consumes. The input is a minimal
/// BPF object with one program text section (call + EXIT) plus
/// a SHT_REL section pointing at the call PC. The symbol the
/// reloc references is parameterised so each gate test can
/// vary it independently. Returns the three tuple elements.
#[allow(clippy::too_many_arguments)]
fn build_subprog_test_scaffold(
    sym_name: &str,
    sym_st_type_bind: u8,
    sym_st_shndx: u16,
    call_bytes: [u8; 8],
) -> (Vec<u8>, Vec<BpfInsn>, HashMap<u32, usize>) {
    let mut strtab: Vec<u8> = vec![0];
    let n_sym = strtab.len() as u32;
    strtab.extend_from_slice(sym_name.as_bytes());
    strtab.push(0);

    let mut symtab: Vec<u8> = Vec::new();
    symtab.extend_from_slice(&elf64_sym(0, 0, 0, 0, 0));
    symtab.extend_from_slice(&elf64_sym(n_sym, sym_st_type_bind, sym_st_shndx, 0, 0));

    let mut text: Vec<u8> = Vec::new();
    text.extend_from_slice(&call_bytes);
    text.extend_from_slice(&kfunc_exit_bytes());
    // r_offset = 0 (call is at the first slot), sym_idx = 1
    // (the synthesised symbol), r_type = 1 (R_BPF_64_64 — value
    // does not matter for the SubprogReturn walk; the function
    // does not gate on r_type, only on the resolved instruction
    // and symbol shape).
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
        ],
        h::EM_BPF,
        h::ET_REL,
    );
    let text_concat: Vec<BpfInsn> = vec![
        BpfInsn::from_le_bytes(call_bytes),
        BpfInsn::from_le_bytes(kfunc_exit_bytes()),
    ];
    let mut section_bases: HashMap<u32, usize> = HashMap::new();
    section_bases.insert(1, 0); // .text at section index 1
    (blob, text_concat, section_bases)
}

// ----- build_datasec_pointers tests ------------------------------
//
// The eight gates inside `build_datasec_pointers` reject malformed
// input and surface only well-formed `DatasecPointer` annotations
// for `R_BPF_64_64` relocations whose target instruction is a
// `BPF_LD_IMM64` referencing a `BTF_KIND_DATASEC` section. The
// tests below construct one `(elf, btf, section_bases)` tuple per
// gate, run [`build_datasec_pointers`], and assert the gate fired
// (empty result) or did not fire (one result with the expected
// fields).

/// Encode `BPF_LD_IMM64` first-slot wire bytes:
/// `code=0x18`, `dst_reg=0`, `src_reg=0`, `off=0`, `imm`.
/// libbpf-style pre-relocation: the LD_IMM64 second slot
/// (also 8 bytes, all zero except trailing imm-high) is appended
/// separately by callers — only the first slot opcode matters
/// for the `build_datasec_pointers` gate.
fn ld_imm64_first_slot_bytes(imm: i32) -> [u8; 8] {
    // `BPF_LD | BPF_DW | BPF_IMM` = 0x18 in linux uapi `bpf.h`.
    let mut out = [0u8; 8];
    out[0] = 0x18;
    out[1] = 0; // regs byte: dst=0, src=0
    out[2..4].copy_from_slice(&0i16.to_le_bytes());
    out[4..8].copy_from_slice(&imm.to_le_bytes());
    out
}

/// `BPF_LD_IMM64` second slot — 8 bytes with the imm-high field
/// cleared. Production paths use this slot for the high 32 bits
/// of a 64-bit immediate; the test only needs a non-call slot
/// the patcher will skip.
fn ld_imm64_second_slot_bytes() -> [u8; 8] {
    [0u8; 8]
}

/// Append a single `BTF_KIND_DATASEC` type to `types`. Each
/// datasec entry is `name_off(4) info(4) size(4)` (12 bytes,
/// the standard btf_type header) plus N * 12 bytes for each
/// VarSecinfo (`type(4) offset(4) size(4)`). `vsi_entries` is a
/// slice of `(type_id, offset, size)` triples — empty list is
/// allowed (vlen=0), giving a name-only datasec.
fn append_btf_datasec(
    types: &mut Vec<u8>,
    name_off: u32,
    section_size: u32,
    vsi_entries: &[(u32, u32, u32)],
) {
    // BTF_KIND_DATASEC = 15. info packs `(kind << 24) | vlen`.
    const BTF_KIND_DATASEC: u32 = 15;
    let vlen = vsi_entries.len() as u32;
    let info = ((BTF_KIND_DATASEC << 24) & 0x1f00_0000) | (vlen & 0xffff);
    types.extend_from_slice(&name_off.to_le_bytes());
    types.extend_from_slice(&info.to_le_bytes());
    // size_or_type field carries the section's total byte size
    // for DATASEC (matches kernel `btf_type::size_or_type` union
    // when `kind == BTF_KIND_DATASEC`).
    types.extend_from_slice(&section_size.to_le_bytes());
    for (type_id, offset, size) in vsi_entries {
        types.extend_from_slice(&type_id.to_le_bytes());
        types.extend_from_slice(&offset.to_le_bytes());
        types.extend_from_slice(&size.to_le_bytes());
    }
}

/// Build a minimal `.BTF` blob containing one `BTF_KIND_DATASEC`
/// named `sec_name` plus one `BTF_KIND_INT u64` (id=1). Returns
/// the byte blob and the datasec id (always 2). The integer is
/// the underlying type for any VarSecinfo entries the caller adds.
fn build_datasec_btf_blob(sec_name: &str) -> (Vec<u8>, u32) {
    let mut strings: Vec<u8> = vec![0];
    let n_u64 = strings.len() as u32;
    strings.extend_from_slice(b"u64");
    strings.push(0);
    let n_sec = strings.len() as u32;
    strings.extend_from_slice(sec_name.as_bytes());
    strings.push(0);

    let mut types: Vec<u8> = Vec::new();
    // id 1: BTF_KIND_INT u64 size=8 bits=64 (encoding=0).
    types.extend_from_slice(&kfunc_btf_type_header(n_u64, 1, 0, 8));
    let int_data: u32 = 64;
    types.extend_from_slice(&int_data.to_le_bytes());
    // id 2: BTF_KIND_DATASEC named `sec_name`, no VSI entries.
    // `build_datasec_pointers` only resolves the section name
    // to a datasec id; it does NOT walk the VSI list (that's
    // the analyzer's job during STX/LDX). An empty VSI list is
    // acceptable for these gate-focused tests.
    append_btf_datasec(&mut types, n_sec, 32, &[]);

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
    (blob, 2)
}

/// Construct the standard scaffold the `build_datasec_pointers`
/// gate tests share: an inner ELF with a `.bss`-named PROGBITS
/// section (the "datasec target"), a `.text` section with one
/// LD_IMM64 + EXIT, a `.symtab` + `.strtab`, and an `SHT_REL`
/// section relocating `.text`. Returns `(blob, btf_blob,
/// text_concat, section_bases)` ready for [`build_datasec_pointers`].
///
/// `r_type` selects the relocation type byte (1 = R_BPF_64_64);
/// `r_offset` selects which `.text` slot the reloc lands on
/// (must be 0 for the LD_IMM64 first slot); `sym_st_value`,
/// `sym_st_shndx`, and `sym_st_type_bind` parameterize the
/// referenced symbol; `imm_value` is the LD_IMM64 first-slot
/// `imm` field. `sec_name_in_btf` controls whether the BTF
/// datasec's name matches the ELF section name.
#[allow(clippy::too_many_arguments)]
fn build_datasec_test_scaffold(
    bss_name: &'static str,
    sec_name_in_btf: &str,
    r_type: u32,
    r_offset: u64,
    sym_st_value: u64,
    sym_st_shndx: u16,
    sym_st_type_bind: u8,
    imm_value: i32,
) -> (Vec<u8>, Vec<u8>, Vec<BpfInsn>, HashMap<u32, usize>) {
    // BTF blob: one datasec whose name is `sec_name_in_btf`.
    let (btf_blob, _ds_id) = build_datasec_btf_blob(sec_name_in_btf);

    // ELF strtab: just the symbol name (we use a single named
    // symbol pointing into `.bss`).
    let mut strtab: Vec<u8> = vec![0];
    let n_sym = strtab.len() as u32;
    strtab.extend_from_slice(b"global_var");
    strtab.push(0);

    // Symtab: shdr[0] is the always-null sentinel; shdr[1] is
    // the variable symbol. `st_info` packs (bind, type) per
    // ELF64. The caller controls both via `sym_st_type_bind`.
    let mut symtab: Vec<u8> = Vec::new();
    symtab.extend_from_slice(&elf64_sym(0, 0, 0, 0, 0));
    symtab.extend_from_slice(&elf64_sym(
        n_sym,
        sym_st_type_bind,
        sym_st_shndx,
        sym_st_value,
        0,
    ));

    // Text section: one LD_IMM64 + an EXIT slot. The LD_IMM64
    // uses two 8-byte slots; we encode a third slot for EXIT
    // so the section byte size is 24 — matching what the BPF
    // loader sees for a real LD_IMM64 followed by an exit.
    let mut text: Vec<u8> = Vec::new();
    text.extend_from_slice(&ld_imm64_first_slot_bytes(imm_value));
    text.extend_from_slice(&ld_imm64_second_slot_bytes());
    text.extend_from_slice(&kfunc_exit_bytes());

    // SHT_REL entry: `r_offset = r_offset` (caller-controlled),
    // `r_sym = 1` (our named symbol), `r_type = r_type`.
    let rel_data: Vec<u8> = elf64_rel(r_offset, 1, r_type).to_vec();

    // ELF layout (caller-controlled section names so tests can
    // exercise the "unknown section name" gate). Section
    // indices: 1 = `.bss`-named (`bss_name`); 2 = `.text`;
    // 3 = `.strtab`; 4 = `.symtab`; 5 = `.rel.text`; 6 = `.BTF`.
    let blob = build_elf64(
        vec![
            SecSpec::new(bss_name, sh::SHT_PROGBITS).data(vec![0u8; 32]),
            SecSpec::new(".text", sh::SHT_PROGBITS)
                .flags(sh::SHF_EXECINSTR.into())
                .data(text),
            SecSpec::new(".strtab", sh::SHT_STRTAB).data(strtab),
            SecSpec::new(".symtab", sh::SHT_SYMTAB)
                .data(symtab)
                .link(3)
                .entsize(24),
            SecSpec::new(".rel.text", sh::SHT_REL)
                .data(rel_data)
                .link(4)
                .info(2) // info = target section idx (.text)
                .entsize(16),
            SecSpec::new(".BTF", sh::SHT_PROGBITS).data(btf_blob.clone()),
        ],
        h::EM_BPF,
        h::ET_REL,
    );

    // Decoded text — three 8-byte instructions:
    //   slot 0: LD_IMM64 first half (the reloc target)
    //   slot 1: LD_IMM64 second half (zeros)
    //   slot 2: EXIT
    let text_concat: Vec<BpfInsn> = vec![
        BpfInsn::from_le_bytes(ld_imm64_first_slot_bytes(imm_value)),
        BpfInsn::from_le_bytes(ld_imm64_second_slot_bytes()),
        BpfInsn::from_le_bytes(kfunc_exit_bytes()),
    ];

    // section_bases: only the .text section (idx 2 here). The
    // base index is 0 because the test object only has one text
    // section, so its instructions start at concat-idx 0.
    let mut section_bases: HashMap<u32, usize> = HashMap::new();
    section_bases.insert(2, 0);

    (blob, btf_blob, text_concat, section_bases)
}

// ---- recover_alloc_size_from_r1 backward scan ----------------------
//
// The scan must stop at the MOST RECENT write to R1: a `MOV r1, imm`
// yields the surviving sizeof; any OTHER write to r1 (ALU/LDX/MOV-from-
// reg/LD into r1, an atomic-fetch into r1, or a BPF_CALL clobbering
// caller-saved r1-r5) must miss conservatively rather than return a
// stale immediate from an earlier MOV. The clobber cases fail on the
// pre-fix code (which scanned past the clobber and returned the stale
// value).

const MOV_R1_CODE: u8 = super::BPF_MOV64_IMM_CODE;
fn call_insn() -> BpfInsn {
    BpfInsn::new(
        super::cast_analysis_load_consts::BPF_JMP_CALL_CODE,
        0,
        0,
        0,
        0,
    )
}

// Test groups extracted from the original flat tests.rs; the shared
// ELF/BTF fixture builders above stay here so every group reaches
// them as a child module via `use super::*`.
mod parse;
mod patch;
mod index;
