use super::*;
use std::io::Write;

/// Append a NUL-terminated string to the BTF strings buffer and
/// return its byte offset. Shared across the cast_analysis test
/// fixtures so each `let push_name = |s, name| { ... }` closure
/// stays out of the per-test setup.
fn push_name(s: &mut Vec<u8>, name: &str) -> u32 {
    let off = s.len() as u32;
    s.extend_from_slice(name.as_bytes());
    s.push(0);
    off
}

// ----- BTF synthesizers ---------------------------------------
//
// The BTF library is read-only; tests build a raw BTF byte blob
// with a small writer and parse it via Btf::from_bytes. The
// BTF wire format (header + type section + string section) is
// documented in linux Documentation/bpf/btf.rst. The helpers
// below cover only the kinds the cast analyzer needs to see:
// BTF_KIND_INT (1), BTF_KIND_PTR (2), BTF_KIND_STRUCT (4),
// BTF_KIND_FUNC (12), BTF_KIND_FUNC_PROTO (13).

const BTF_MAGIC: u16 = 0xEB9F;
const BTF_VERSION: u8 = 1;
const BTF_HEADER_LEN: u32 = 24;

const BTF_KIND_INT: u32 = 1;
const BTF_KIND_PTR: u32 = 2;
/// `BTF_KIND_ARRAY = 3` — fixed-length array. Wire layout:
/// header + `btf_array { u32 type, u32 index_type, u32 nelems }`
/// (12 bytes after the 12-byte common header). Per linux uapi
/// `btf.h` and `btf-rs::cbtf::btf_array`, an Array's `size_type`
/// field in the common header is 0 (storage size derives from
/// `elem_size * nelems`, not the header).
const BTF_KIND_ARRAY: u32 = 3;
const BTF_KIND_STRUCT: u32 = 4;
const BTF_KIND_UNION: u32 = 5;
const BTF_KIND_FWD: u32 = 7;
const BTF_KIND_TYPEDEF: u32 = 8;
const BTF_KIND_VOLATILE: u32 = 9;
const BTF_KIND_CONST: u32 = 10;
const BTF_KIND_FUNC: u32 = 12;
const BTF_KIND_FUNC_PROTO: u32 = 13;
/// `BTF_KIND_VAR = 14` — global variable declaration.
/// References its underlying type via the post-header `type`
/// u32 and carries a linkage u32 (static / global / extern).
const BTF_KIND_VAR: u32 = 14;
/// `BTF_KIND_DATASEC = 15` — global section (`.bss`,
/// `.data`, `.rodata`, `.data.<name>`). Carries a list of
/// `VarSecinfo` records that point at `BTF_KIND_VAR` entries.
const BTF_KIND_DATASEC: u32 = 15;

/// `info` field bit 31: encodes bitfield-style member offset / vs
/// regular union for Fwd. Per linux uapi `btf.h` and
/// `btf-rs::cbtf::btf_type::kind_flag`. Splitting the constant out
/// keeps the OR expressions in `build_btf` readable when emitting
/// kind_flag=1 structs / fwd-union.
const KIND_FLAG_BIT: u32 = 1 << 31;

/// One member in a synthetic struct.
#[derive(Clone, Copy)]
struct SynMember {
    name_off: u32,
    type_id: u32,
    /// Byte offset; converted to bit offset on emit.
    byte_offset: u32,
}

/// One parameter in a synthetic FuncProto. `type_id` is the BTF
/// type id of the parameter's type (0 + name_off=0 marks the
/// variadic sentinel — tests don't emit it).
#[derive(Clone, Copy)]
struct SynParam {
    name_off: u32,
    type_id: u32,
}

/// One member in a synthetic struct/union built with kind_flag=1.
/// Encodes `bit_offset` in the low 24 bits and `bitfield_size_bits`
/// in the upper 8 bits of the member's `offset` u32 (per linux uapi
/// `btf.h` and `btf-rs::Member::bitfield_size`). Used by tests that
/// exercise bitfield handling in `build_layout_index` and
/// `struct_member_at`.
#[derive(Clone, Copy)]
struct SynMemberBits {
    name_off: u32,
    type_id: u32,
    /// Member offset in BITS (NOT bytes). For non-bit-aligned
    /// members the test sets a bit position that is not a multiple
    /// of 8.
    bit_offset: u32,
    /// 0 for a non-bitfield member in a kind_flag=1 struct, > 0
    /// for an actual bitfield. Production skips members with
    /// `bitfield_size > 0` AND members whose `bit_offset % 8 != 0`.
    bitfield_size_bits: u32,
}

/// One synthetic BTF type. Tests build a Vec<SynType>; the
/// writer assigns ids starting at 1 (id 0 is Void) and emits
/// the type section in order.
#[allow(dead_code)] // not all variants are used by every test
enum SynType {
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
    /// `BTF_KIND_ARRAY` — fixed-length array. Wire layout:
    /// header (with `name_off=0`, `vlen=0`, `kind_flag=0`,
    /// `size_type=0` per linux uapi `btf.h`) followed by
    /// `btf_array { u32 type, u32 index_type, u32 nelems }`
    /// (12 bytes), matching `btf-rs::cbtf::btf_array::from_reader`.
    /// `type_id` is the element type, `index_type_id` is the
    /// index type (typically `u32`), `nelems` is the array length.
    Array {
        type_id: u32,
        index_type_id: u32,
        nelems: u32,
    },
    Struct {
        name_off: u32,
        size: u32,
        members: Vec<SynMember>,
    },
    /// `BTF_KIND_UNION` — same wire layout as Struct (members are
    /// `btf_member` records). `btf-rs` aliases `Union = Struct`,
    /// so production code paths walk both via the same
    /// `Type::Struct(s) | Type::Union(s)` arm.
    Union {
        name_off: u32,
        size: u32,
        members: Vec<SynMember>,
    },
    /// `BTF_KIND_STRUCT` with kind_flag=1 — member `offset` u32
    /// packs `bit_offset` in low 24 bits and `bitfield_size` in
    /// upper 8 bits. Real C structs that contain ANY bitfield
    /// member have kind_flag=1 even for non-bitfield members in
    /// the same struct (per linux uapi `btf.h`).
    StructBitfields {
        name_off: u32,
        size: u32,
        members: Vec<SynMemberBits>,
    },
    /// `BTF_KIND_FWD` — forward declaration. No payload after the
    /// `btf_type` header. `kind_flag` selects struct (0) vs union
    /// (1) per `btf-rs::Fwd::is_struct` / `is_union`. Used by
    /// tests that probe `member_size_bytes` for unsupported
    /// terminals via a struct member typed as Fwd.
    Fwd {
        name_off: u32,
        kind_flag: u32,
    },
    /// `BTF_KIND_TYPEDEF` — `typedef X T`. Same wire layout as
    /// `Ptr`: header followed by a `u32 type` referencing the
    /// underlying type id. `peel_modifiers` peels through it.
    Typedef {
        name_off: u32,
        type_id: u32,
    },
    /// `BTF_KIND_VOLATILE` — `volatile T`. Same wire layout as
    /// `Ptr`. `name_off` is 0 per the BTF spec.
    Volatile {
        type_id: u32,
    },
    /// `BTF_KIND_CONST` — `const T`. Same wire layout as `Ptr`.
    /// `name_off` is 0 per the BTF spec.
    Const {
        type_id: u32,
    },
    /// `BTF_KIND_FUNC`. `type_id` is the FuncProto id; `vlen`
    /// encodes BTF_FUNC_STATIC/GLOBAL/EXTERN (0/1/2).
    Func {
        name_off: u32,
        type_id: u32,
        linkage: u32,
    },
    /// `BTF_KIND_FUNC_PROTO`. `return_type_id` is the BTF id
    /// of the return type (0 = void); `params` enumerate the
    /// parameter list. `name_off` is always 0 per the BTF
    /// spec (FuncProto types are anonymous).
    FuncProto {
        return_type_id: u32,
        params: Vec<SynParam>,
    },
    /// `BTF_KIND_VAR` — global variable declaration. Wire
    /// layout: header + `u32 type` + `u32 linkage`. Used as
    /// the entry-pointed-to from a `BTF_KIND_DATASEC`
    /// `VarSecinfo`, since libbpf always emits Datasec
    /// entries pointing at a Var (the Var carries the
    /// variable's name and references its underlying C type).
    /// `linkage` mirrors `BTF_VAR_STATIC=0` /
    /// `BTF_VAR_GLOBAL_ALLOCATED=1` /
    /// `BTF_VAR_GLOBAL_EXTERN=2`.
    Var {
        name_off: u32,
        type_id: u32,
        linkage: u32,
    },
    /// `BTF_KIND_DATASEC` — a global section (`.bss`,
    /// `.data`, `.rodata`, `.data.<name>`). Wire layout:
    /// header + `u32 size` + per-VarSecinfo records of
    /// `{u32 type; u32 offset; u32 size}`. Each VarSecinfo
    /// references a `BTF_KIND_VAR` whose underlying type is
    /// the global's C type.
    Datasec {
        name_off: u32,
        size: u32,
        entries: Vec<SynVarSecinfo>,
    },
}

/// One entry in a synthetic `BTF_KIND_DATASEC`. Wire layout:
/// `{u32 type_id; u32 offset; u32 size}` per VarSecinfo. The
/// `type_id` references a `BTF_KIND_VAR` whose underlying
/// type is the global's C type.
#[derive(Clone, Copy)]
struct SynVarSecinfo {
    type_id: u32,
    /// Byte offset of the variable within the section.
    offset: u32,
    /// Byte size of the variable's storage (matches the
    /// underlying type's `type_size`).
    size: u32,
}

/// Build a minimal BTF byte blob for testing.
///
/// `strings` is the string section payload (must start with
/// `\0`). Type ids start at 1 and increase in `types` order.
fn build_btf(types: &[SynType], strings: &[u8]) -> Vec<u8> {
    let mut type_section = Vec::new();
    for ty in types {
        match ty {
            SynType::Int {
                name_off,
                size,
                encoding,
                offset,
                bits,
            } => {
                type_section.extend_from_slice(&name_off.to_le_bytes());
                let info = (BTF_KIND_INT << 24) & 0x1f00_0000;
                type_section.extend_from_slice(&info.to_le_bytes());
                type_section.extend_from_slice(&size.to_le_bytes());
                let int_data = (*encoding << 24) | ((*offset & 0xff) << 16) | (*bits & 0xff);
                type_section.extend_from_slice(&int_data.to_le_bytes());
            }
            SynType::Ptr { type_id } => {
                let name_off: u32 = 0;
                type_section.extend_from_slice(&name_off.to_le_bytes());
                let info = (BTF_KIND_PTR << 24) & 0x1f00_0000;
                type_section.extend_from_slice(&info.to_le_bytes());
                type_section.extend_from_slice(&type_id.to_le_bytes());
            }
            SynType::Array {
                type_id,
                index_type_id,
                nelems,
            } => {
                // Per linux uapi `btf.h` (`btf_array_check_meta`):
                // ARRAY's `name_off`, `vlen`, `kind_flag`, and
                // `size_type` are all 0; the trailing `btf_array`
                // record carries `type`, `index_type`, `nelems`.
                let name_off: u32 = 0;
                type_section.extend_from_slice(&name_off.to_le_bytes());
                let info = (BTF_KIND_ARRAY << 24) & 0x1f00_0000;
                type_section.extend_from_slice(&info.to_le_bytes());
                let size_type: u32 = 0;
                type_section.extend_from_slice(&size_type.to_le_bytes());
                // Trailing `btf_array` (12 bytes) — matches
                // `btf-rs::cbtf::btf_array::from_reader` field order.
                type_section.extend_from_slice(&type_id.to_le_bytes());
                type_section.extend_from_slice(&index_type_id.to_le_bytes());
                type_section.extend_from_slice(&nelems.to_le_bytes());
            }
            SynType::Struct {
                name_off,
                size,
                members,
            } => {
                type_section.extend_from_slice(&name_off.to_le_bytes());
                let vlen = members.len() as u32;
                let info = ((BTF_KIND_STRUCT << 24) & 0x1f00_0000) | (vlen & 0xffff);
                type_section.extend_from_slice(&info.to_le_bytes());
                type_section.extend_from_slice(&size.to_le_bytes());
                for m in members {
                    type_section.extend_from_slice(&m.name_off.to_le_bytes());
                    type_section.extend_from_slice(&m.type_id.to_le_bytes());
                    // Non-bitfield: bit_offset = byte * 8.
                    let bit_off = m.byte_offset * 8;
                    type_section.extend_from_slice(&bit_off.to_le_bytes());
                }
            }
            SynType::Union {
                name_off,
                size,
                members,
            } => {
                type_section.extend_from_slice(&name_off.to_le_bytes());
                let vlen = members.len() as u32;
                // Same wire encoding as Struct, only the kind id
                // differs. kind_flag=0 (regular union, members
                // carry plain bit_offset).
                let info = ((BTF_KIND_UNION << 24) & 0x1f00_0000) | (vlen & 0xffff);
                type_section.extend_from_slice(&info.to_le_bytes());
                type_section.extend_from_slice(&size.to_le_bytes());
                for m in members {
                    type_section.extend_from_slice(&m.name_off.to_le_bytes());
                    type_section.extend_from_slice(&m.type_id.to_le_bytes());
                    // Union members all sit at bit offset 0 in
                    // real C, but the wire format still carries a
                    // u32 offset. Allow tests to set arbitrary
                    // byte_offset values to exercise the
                    // struct_member_at lookup logic; production
                    // matches on byte position regardless of kind.
                    let bit_off = m.byte_offset * 8;
                    type_section.extend_from_slice(&bit_off.to_le_bytes());
                }
            }
            SynType::StructBitfields {
                name_off,
                size,
                members,
            } => {
                type_section.extend_from_slice(&name_off.to_le_bytes());
                let vlen = members.len() as u32;
                // kind_flag=1 (bit 31 set): member offset packs
                // (bitfield_size << 24) | (bit_offset & 0xffffff).
                let info =
                    (((BTF_KIND_STRUCT << 24) & 0x1f00_0000) | (vlen & 0xffff)) | KIND_FLAG_BIT;
                type_section.extend_from_slice(&info.to_le_bytes());
                type_section.extend_from_slice(&size.to_le_bytes());
                for m in members {
                    type_section.extend_from_slice(&m.name_off.to_le_bytes());
                    type_section.extend_from_slice(&m.type_id.to_le_bytes());
                    let packed =
                        ((m.bitfield_size_bits & 0xff) << 24) | (m.bit_offset & 0x00ff_ffff);
                    type_section.extend_from_slice(&packed.to_le_bytes());
                }
            }
            SynType::Fwd {
                name_off,
                kind_flag,
            } => {
                type_section.extend_from_slice(&name_off.to_le_bytes());
                // BTF_KIND_FWD: vlen is unused (set to 0); the
                // kind_flag bit encodes struct (0) vs union (1).
                // size_type field is also unused per the kernel
                // wire format but is still 4 bytes long; emit 0.
                let info = ((BTF_KIND_FWD << 24) & 0x1f00_0000) | ((*kind_flag & 0x1) << 31);
                type_section.extend_from_slice(&info.to_le_bytes());
                type_section.extend_from_slice(&0u32.to_le_bytes());
            }
            SynType::Typedef { name_off, type_id } => {
                type_section.extend_from_slice(&name_off.to_le_bytes());
                let info = (BTF_KIND_TYPEDEF << 24) & 0x1f00_0000;
                type_section.extend_from_slice(&info.to_le_bytes());
                type_section.extend_from_slice(&type_id.to_le_bytes());
            }
            SynType::Volatile { type_id } => {
                let name_off: u32 = 0;
                type_section.extend_from_slice(&name_off.to_le_bytes());
                let info = (BTF_KIND_VOLATILE << 24) & 0x1f00_0000;
                type_section.extend_from_slice(&info.to_le_bytes());
                type_section.extend_from_slice(&type_id.to_le_bytes());
            }
            SynType::Const { type_id } => {
                let name_off: u32 = 0;
                type_section.extend_from_slice(&name_off.to_le_bytes());
                let info = (BTF_KIND_CONST << 24) & 0x1f00_0000;
                type_section.extend_from_slice(&info.to_le_bytes());
                type_section.extend_from_slice(&type_id.to_le_bytes());
            }
            SynType::Func {
                name_off,
                type_id,
                linkage,
            } => {
                type_section.extend_from_slice(&name_off.to_le_bytes());
                // BTF_KIND_FUNC encodes the linkage in vlen
                // (0=static, 1=global, 2=extern).
                let info = ((BTF_KIND_FUNC << 24) & 0x1f00_0000) | (*linkage & 0xffff);
                type_section.extend_from_slice(&info.to_le_bytes());
                type_section.extend_from_slice(&type_id.to_le_bytes());
            }
            SynType::FuncProto {
                return_type_id,
                params,
            } => {
                let name_off: u32 = 0;
                type_section.extend_from_slice(&name_off.to_le_bytes());
                let vlen = params.len() as u32;
                let info = ((BTF_KIND_FUNC_PROTO << 24) & 0x1f00_0000) | (vlen & 0xffff);
                type_section.extend_from_slice(&info.to_le_bytes());
                type_section.extend_from_slice(&return_type_id.to_le_bytes());
                for p in params {
                    type_section.extend_from_slice(&p.name_off.to_le_bytes());
                    type_section.extend_from_slice(&p.type_id.to_le_bytes());
                }
            }
            SynType::Var {
                name_off,
                type_id,
                linkage,
            } => {
                type_section.extend_from_slice(&name_off.to_le_bytes());
                // BTF_KIND_VAR = 14. Header followed by
                // `u32 type` + `u32 linkage`.
                let info = (BTF_KIND_VAR << 24) & 0x1f00_0000;
                type_section.extend_from_slice(&info.to_le_bytes());
                type_section.extend_from_slice(&type_id.to_le_bytes());
                type_section.extend_from_slice(&linkage.to_le_bytes());
            }
            SynType::Datasec {
                name_off,
                size,
                entries,
            } => {
                type_section.extend_from_slice(&name_off.to_le_bytes());
                let vlen = entries.len() as u32;
                // BTF_KIND_DATASEC = 15. Header followed by
                // `u32 size` + per-entry `{u32 type, u32
                // offset, u32 size}`.
                let info = ((BTF_KIND_DATASEC << 24) & 0x1f00_0000) | (vlen & 0xffff);
                type_section.extend_from_slice(&info.to_le_bytes());
                type_section.extend_from_slice(&size.to_le_bytes());
                for e in entries {
                    type_section.extend_from_slice(&e.type_id.to_le_bytes());
                    type_section.extend_from_slice(&e.offset.to_le_bytes());
                    type_section.extend_from_slice(&e.size.to_le_bytes());
                }
            }
        }
    }

    let type_len = type_section.len() as u32;
    let str_len = strings.len() as u32;

    let mut blob = Vec::new();
    // Header (24 bytes).
    blob.write_all(&BTF_MAGIC.to_le_bytes()).unwrap();
    blob.push(BTF_VERSION);
    blob.push(0); // flags
    blob.write_all(&BTF_HEADER_LEN.to_le_bytes()).unwrap();
    blob.write_all(&0u32.to_le_bytes()).unwrap(); // type_off
    blob.write_all(&type_len.to_le_bytes()).unwrap();
    blob.write_all(&type_len.to_le_bytes()).unwrap(); // str_off (= type_len)
    blob.write_all(&str_len.to_le_bytes()).unwrap();
    blob.extend_from_slice(&type_section);
    blob.extend_from_slice(strings);
    blob
}

// BTF int encoding flags: signed = 1, char = 2, bool = 4. The
// synthesizer uses 0 for plain unsigned.

/// Helper: build a BTF with `task_struct`-like source struct
/// `T` (id=2) and target struct `Q` (id=3). T has a u64 field
/// at byte offset `field_off` named `f`. Q has a u32 at byte
/// offset `target_off`. Returns the byte blob and the (T_id,
/// Q_id) pair.
fn btf_with_source_and_target(field_off: u32, target_off: u32) -> (Vec<u8>, u32, u32) {
    // Strings: null + names. Order matters since name_offs
    // index into this byte vector.
    let mut strings: Vec<u8> = vec![0];
    let n_int = push_name(&mut strings, "u64");
    let n_t = push_name(&mut strings, "T");
    let n_q = push_name(&mut strings, "Q");
    let n_f = push_name(&mut strings, "f");
    let n_x = push_name(&mut strings, "x");

    let types = vec![
        // id 1: int u64 (size=8, bits=64).
        SynType::Int {
            name_off: n_int,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        // id 2: struct T { ... f at field_off ... } size = field_off + 8.
        SynType::Struct {
            name_off: n_t,
            size: field_off + 8,
            members: vec![SynMember {
                name_off: n_f,
                type_id: 1,
                byte_offset: field_off,
            }],
        },
        // id 3: struct Q { u64 x at target_off }, size = target_off + 8.
        SynType::Struct {
            name_off: n_q,
            size: target_off + 8,
            members: vec![SynMember {
                name_off: n_x,
                type_id: 1,
                byte_offset: target_off,
            }],
        },
    ];
    (build_btf(&types, &strings), 2, 3)
}

/// Like [`btf_with_source_and_target`] but with TWO same-shape targets
/// `Q1` (id=3) and `Q2` (id=4), each a struct with a `u64` at offset 0.
/// A deref of the cast result at offset 0 matches BOTH, so shape
/// inference's `candidates.len() == 1` check fails and drops the slot --
/// exercising the deferred-resolve emit path. Source struct `T` (id=2)
/// has a `u64` at `field_off`. Returns the blob and the `T` id.
fn btf_source_and_two_targets(field_off: u32) -> (Vec<u8>, u32) {
    let mut strings: Vec<u8> = vec![0];
    let n_int = push_name(&mut strings, "u64");
    let n_t = push_name(&mut strings, "T");
    let n_q1 = push_name(&mut strings, "Q1");
    let n_q2 = push_name(&mut strings, "Q2");
    let n_f = push_name(&mut strings, "f");
    let n_x = push_name(&mut strings, "x");
    let types = vec![
        // id 1: int u64.
        SynType::Int {
            name_off: n_int,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        // id 2: struct T { u64 f at field_off }.
        SynType::Struct {
            name_off: n_t,
            size: field_off + 8,
            members: vec![SynMember {
                name_off: n_f,
                type_id: 1,
                byte_offset: field_off,
            }],
        },
        // id 3: struct Q1 { u64 x at 0 }.
        SynType::Struct {
            name_off: n_q1,
            size: 8,
            members: vec![SynMember {
                name_off: n_x,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        // id 4: struct Q2 { u64 x at 0 } -- same shape as Q1.
        SynType::Struct {
            name_off: n_q2,
            size: 8,
            members: vec![SynMember {
                name_off: n_x,
                type_id: 1,
                byte_offset: 0,
            }],
        },
    ];
    (build_btf(&types, &strings), 2)
}

/// Small helper to emit a single [`BpfInsn`] with given fields.
/// Uses `BpfInsn::new` directly so dst/src register packing is
/// done by the constructor — no `let mut x = X::default(); x.f = …`
/// clippy footgun, no separate setter calls.
fn mk_insn(code: u8, dst: u8, src: u8, off: i16, imm: i32) -> BpfInsn {
    BpfInsn::new(code, dst, src, off, imm)
}

fn ldx(size: u8, dst: u8, src: u8, off: i16) -> BpfInsn {
    mk_insn(BPF_CLASS_LDX | size | BPF_MODE_MEM, dst, src, off, 0)
}

/// `*(size *)(r_dst + off) = r_src`. Plain memory store (BPF_MEM
/// mode), the spill / kptr-write encoding the analyzer cares
/// about. dst is the address-base register, src is the value.
fn stx(size: u8, dst: u8, src: u8, off: i16) -> BpfInsn {
    mk_insn(BPF_CLASS_STX | size | BPF_MODE_MEM, dst, src, off, 0)
}

fn mov_x(dst: u8, src: u8) -> BpfInsn {
    mk_insn(BPF_CLASS_ALU64 | BPF_OP_MOV | BPF_SRC_X, dst, src, 0, 0)
}

fn mov_k(dst: u8, imm: i32) -> BpfInsn {
    mk_insn(BPF_CLASS_ALU64 | BPF_OP_MOV, dst, 0, 0, imm)
}

/// Generic plain-helper `BPF_CALL` for tests that exercise the
/// R0..R5 clobber + spill/reload behaviour without engaging the
/// helper-return arm. `imm == 1` happens to coincide with
/// `BPF_FUNC_map_lookup_elem`, but the helper-return arm only
/// fires when R1 is a `RegState::DatasecPointer{".maps", ..}` at
/// the call site — none of the call sites that consume this
/// helper seed R1 that way, so R0 still ends up Unknown after the
/// clobber as those tests assert. Tests that need a different
/// helper id (e.g. to confirm the arm rejects non-allowlist ids)
/// build the call instruction inline via `helper_call` instead.
fn call() -> BpfInsn {
    mk_insn(BPF_CLASS_JMP | BPF_OP_CALL, 0, 0, 0, 1)
}

/// `BPF_CALL` with `src_reg == BPF_PSEUDO_KFUNC_CALL` and
/// `imm == kfunc_btf_id`. Models the relocated call form where
/// the imm carries the BTF id of a `BTF_KIND_FUNC`.
fn kfunc_call(kfunc_btf_id: u32) -> BpfInsn {
    mk_insn(
        BPF_CLASS_JMP | BPF_OP_CALL,
        0,
        BPF_PSEUDO_KFUNC_CALL,
        0,
        kfunc_btf_id as i32,
    )
}

fn exit() -> BpfInsn {
    mk_insn(BPF_CLASS_JMP | BPF_OP_EXIT, 0, 0, 0, 0)
}

/// `BPF_ADDR_SPACE_CAST` (kernel/bpf/verifier.c::check_alu_op):
/// `ALU64 | MOV | X` with `off=1, imm=1` is the as(1)→as(0)
/// cast — arena→kernel. `imm=1<<16` is the as(0)→as(1) cast in
/// the other direction. Tests use the arena→kernel form to
/// surface arena_confirmed evidence for shape-inference findings
/// (arena-evidence mitigation).
fn addr_space_cast(dst: u8, src: u8, imm: i32) -> BpfInsn {
    mk_insn(BPF_CLASS_ALU64 | BPF_OP_MOV | BPF_SRC_X, dst, src, 1, imm)
}

// ----- Cross-group BTF / instruction helpers ------------------
//
// `btf_kptr_base` and `ld_imm64` are each used by more than one
// test submodule, so they live in the facade where every
// submodule reaches them via `use super::*`.

/// Build a BTF blob with:
/// - id 1: u64
/// - id 2: struct T { u64 x @ 0 }   ("task_struct" stand-in)
/// - id 3: T*  (pointer to id=2)
/// - id 4: struct P { u64 slot @ slot_off }
///
/// Returns (blob, T_id, P_id, T_ptr_id). Tests that need a
/// FuncProto add it on top of this blob.
fn btf_kptr_base(slot_off: u32) -> (Vec<u8>, u32, u32, u32) {
    let mut strings: Vec<u8> = vec![0];
    let n_u64 = push_name(&mut strings, "u64");
    let n_t = push_name(&mut strings, "T");
    let n_p = push_name(&mut strings, "P");
    let n_x = push_name(&mut strings, "x");
    let n_slot = push_name(&mut strings, "slot");
    let types = vec![
        // id 1: u64
        SynType::Int {
            name_off: n_u64,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        // id 2: struct T { u64 x @ 0 }
        SynType::Struct {
            name_off: n_t,
            size: 8,
            members: vec![SynMember {
                name_off: n_x,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        // id 3: T*
        SynType::Ptr { type_id: 2 },
        // id 4: struct P { u64 slot @ slot_off }
        SynType::Struct {
            name_off: n_p,
            size: slot_off + 8,
            members: vec![SynMember {
                name_off: n_slot,
                type_id: 1,
                byte_offset: slot_off,
            }],
        },
    ];
    let blob = build_btf(&types, &strings);
    (blob, 2, 4, 3)
}

/// Helper: emit `BPF_LD_IMM64 dst, imm` as the two-instruction
/// pseudo. The second slot's `code` is 0 per linux uapi
/// `bpf.h`. The analyzer's `skip_next` flag swallows the
/// second slot.
fn ld_imm64(dst: u8, imm: i32) -> [BpfInsn; 2] {
    let lo = mk_insn(BPF_CLASS_LD | BPF_SIZE_DW | BPF_MODE_IMM, dst, 0, 0, imm);
    let hi = mk_insn(0, 0, 0, 0, 0);
    [lo, hi]
}

mod atomic_stack;
mod bss_jumps_funcentry;
mod btf_stack_kfunc_edge;
mod cast_tracking;
mod conflict_oob_finalize;
mod frameaddr_kfunc_stx_edge;
mod helper_map_crossfn;
mod kptr_addrspace;
mod register_alu_cast_misc;
mod stress;
mod stx_flow;
