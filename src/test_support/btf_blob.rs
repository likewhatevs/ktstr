//! Synthetic BTF-blob builder shared across `#[cfg(test)]` suites.
//!
//! Hand-assembles a minimal BTF section (24-byte header + type
//! section + string section) from a list of [`CastSynType`] records,
//! producing a `Vec<u8>` that `btf_rs::Btf::from_bytes` parses. Lets
//! tests pin BTF-driven code paths (struct/member offset resolution,
//! the renderer's cast intercept, bitfield decoding) without a real
//! vmlinux — no `CONFIG_DEBUG_INFO_BTF` kernel, no `KTSTR_KERNEL`
//! resolution, no sidecar cache.
//!
//! Originally lived in `src/monitor/btf_render/tests/mod.rs` (the
//! `cast_*` renderer suites); hoisted here so the
//! `kernel_op_dispatch` BTF-gated dispatch tests
//! (`resolve_per_cpu_field_pa`, `resolve_and_validate_task_field`)
//! can build the same blobs. The renderer suites consume it through a
//! `pub use crate::test_support::btf_blob::*` re-export in their `mod`,
//! so their `super::*` imports keep resolving unchanged.

const CAST_BTF_MAGIC: u16 = 0xEB9F;
const CAST_BTF_VERSION: u8 = 1;
const CAST_BTF_HEADER_LEN: u32 = 24;
const CAST_BTF_KIND_INT: u32 = 1;
/// `BTF_KIND_PTR` per `btf-rs::obj::resolve` — kind 2 maps to
/// `Type::Ptr`. Used by the Fwd-pointee chase tests so the Type::Ptr
/// arm hits a forward-declared pointee.
const CAST_BTF_KIND_PTR: u32 = 2;
const CAST_BTF_KIND_STRUCT: u32 = 4;
/// `BTF_KIND_FWD` per `btf-rs::obj::resolve` — kind 7 maps to
/// `Type::Fwd`. Used by the Fwd-pointee chase tests; libbpf emits
/// this for structs whose body lives in a separate BTF (e.g.
/// `struct sdt_data` defined in the sdt_alloc library and referenced
/// from a scheduler that doesn't include the full body).
const CAST_BTF_KIND_FWD: u32 = 7;
/// `BTF_KIND_TYPEDEF` per `btf-rs::obj::resolve` — kind 8 maps to
/// `Type::Typedef`. Used by the modifier-chain integration test.
const CAST_BTF_KIND_TYPEDEF: u32 = 8;
/// `BTF_KIND_CONST` per `btf-rs::obj::resolve` — kind 10 maps to
/// `Type::Const`. Used by the modifier-chain integration test.
const CAST_BTF_KIND_CONST: u32 = 10;
/// `BTF_KIND_TYPE_TAG` per `btf-rs::obj::resolve` — kind 18 maps to
/// `Type::TypeTag`. Models a `__kptr` tag wrapping a pointer, the
/// realistic shape of a `struct bpf_cpumask __kptr *` member/global.
const CAST_BTF_KIND_TYPE_TAG: u32 = 18;
/// `BTF_KIND_ENUM` per `btf-rs` — kind 6 maps to `Type::Enum`. Used
/// by the bitfield tests with a signed-enum base.
const CAST_BTF_KIND_ENUM: u32 = 6;
/// `BTF_KIND_ENUM64` per `btf-rs` — kind 19 maps to `Type::Enum64`.
/// Used by the bitfield tests with a signed-enum64 base.
const CAST_BTF_KIND_ENUM64: u32 = 19;
/// `BTF_KIND_FLOAT` per `btf-rs::cbtf` — kind 16 maps to
/// `Type::Float`. Used by the float-decode tests so the renderer's
/// `render_float` byte-decode arms run against a real BTF float
/// instead of a hand-constructed `RenderedValue::Float`.
const CAST_BTF_KIND_FLOAT: u32 = 16;

/// Build a minimal BTF blob containing `types` (id=1..) and a
/// string-section payload `strings` (must start with `\0`). The
/// header layout matches `cast_analysis::tests::build_btf`:
/// 24-byte header, type section, string section. Supports the BTF
/// kinds the renderer tests exercise: Int, Struct, BitfieldStruct,
/// Enum, Enum64, Typedef, Const, TypeTag, Ptr, Fwd.
pub(crate) fn cast_build_btf(types: &[CastSynType], strings: &[u8]) -> Vec<u8> {
    let mut type_section = Vec::new();
    for ty in types {
        match ty {
            CastSynType::Int {
                name_off,
                size,
                encoding,
                offset,
                bits,
            } => {
                type_section.extend_from_slice(&name_off.to_le_bytes());
                let info = (CAST_BTF_KIND_INT << 24) & 0x1f00_0000;
                type_section.extend_from_slice(&info.to_le_bytes());
                type_section.extend_from_slice(&size.to_le_bytes());
                let int_data = (*encoding << 24) | ((*offset & 0xff) << 16) | (*bits & 0xff);
                type_section.extend_from_slice(&int_data.to_le_bytes());
            }
            CastSynType::Struct {
                name_off,
                size,
                members,
            } => {
                type_section.extend_from_slice(&name_off.to_le_bytes());
                let vlen = members.len() as u32;
                let info = ((CAST_BTF_KIND_STRUCT << 24) & 0x1f00_0000) | (vlen & 0xffff);
                type_section.extend_from_slice(&info.to_le_bytes());
                type_section.extend_from_slice(&size.to_le_bytes());
                for m in members {
                    type_section.extend_from_slice(&m.name_off.to_le_bytes());
                    type_section.extend_from_slice(&m.type_id.to_le_bytes());
                    let bit_off = m.byte_offset * 8;
                    type_section.extend_from_slice(&bit_off.to_le_bytes());
                }
            }
            CastSynType::Typedef { name_off, type_id } => {
                // BTF_KIND_TYPEDEF wire layout: name_off (4) + info (4)
                // + size_type (4) where size_type holds the wrapped
                // type id. Per `cbtf::btf_type::kind`, the kind is
                // bits 24..29 of `info`; vlen is 0 for Typedef.
                type_section.extend_from_slice(&name_off.to_le_bytes());
                let info = (CAST_BTF_KIND_TYPEDEF << 24) & 0x1f00_0000;
                type_section.extend_from_slice(&info.to_le_bytes());
                type_section.extend_from_slice(&type_id.to_le_bytes());
            }
            CastSynType::Const { type_id } => {
                // BTF_KIND_CONST wire layout: name_off (4, always 0) +
                // info (4) + size_type (4, the wrapped type id). Per
                // the BTF spec, Const types are anonymous so name_off
                // is unused.
                let name_off: u32 = 0;
                type_section.extend_from_slice(&name_off.to_le_bytes());
                let info = (CAST_BTF_KIND_CONST << 24) & 0x1f00_0000;
                type_section.extend_from_slice(&info.to_le_bytes());
                type_section.extend_from_slice(&type_id.to_le_bytes());
            }
            CastSynType::TypeTag { name_off, type_id } => {
                // BTF_KIND_TYPE_TAG wire layout: name_off (4) + info (4)
                // + type (4, the tagged type id). Same shape as
                // Typedef; the kind byte selects TypeTag. Models a
                // `__kptr` tag wrapping a pointer.
                type_section.extend_from_slice(&name_off.to_le_bytes());
                let info = (CAST_BTF_KIND_TYPE_TAG << 24) & 0x1f00_0000;
                type_section.extend_from_slice(&info.to_le_bytes());
                type_section.extend_from_slice(&type_id.to_le_bytes());
            }
            CastSynType::Ptr { type_id } => {
                // BTF_KIND_PTR wire layout: name_off (4, always 0) +
                // info (4) + size_type (4, the pointee type id). Ptr
                // types are anonymous per the BTF spec.
                let name_off: u32 = 0;
                type_section.extend_from_slice(&name_off.to_le_bytes());
                let info = (CAST_BTF_KIND_PTR << 24) & 0x1f00_0000;
                type_section.extend_from_slice(&info.to_le_bytes());
                type_section.extend_from_slice(&type_id.to_le_bytes());
            }
            CastSynType::Fwd { name_off, is_union } => {
                // BTF_KIND_FWD wire layout: name_off (4) + info (4) +
                // size_type (4, unused — emit 0). Per
                // `btf-rs::Fwd::is_union`, the kind_flag (bit 31 of
                // info) selects struct (0) vs union (1) for the
                // forward declaration's referent.
                type_section.extend_from_slice(&name_off.to_le_bytes());
                let kind_flag = if *is_union { 1u32 << 31 } else { 0 };
                let info = ((CAST_BTF_KIND_FWD << 24) & 0x1f00_0000) | kind_flag;
                type_section.extend_from_slice(&info.to_le_bytes());
                type_section.extend_from_slice(&0u32.to_le_bytes());
            }
            CastSynType::BitfieldStruct {
                name_off,
                size,
                members,
            } => {
                // A BTF_KIND_STRUCT emitted with kind_flag == 1 (info
                // bit 31): btf-rs then decodes each member offset word
                // as bitfield_size<<24 | (bit_offset & 0xffffff) and
                // Member::bitfield_size returns Some.
                type_section.extend_from_slice(&name_off.to_le_bytes());
                let vlen = members.len() as u32;
                let info =
                    ((CAST_BTF_KIND_STRUCT << 24) & 0x1f00_0000) | (vlen & 0xffff) | (1u32 << 31);
                type_section.extend_from_slice(&info.to_le_bytes());
                type_section.extend_from_slice(&size.to_le_bytes());
                for m in members {
                    type_section.extend_from_slice(&m.name_off.to_le_bytes());
                    type_section.extend_from_slice(&m.type_id.to_le_bytes());
                    let off_word = (m.bitfield_size << 24) | (m.bit_offset & 0x00ff_ffff);
                    type_section.extend_from_slice(&off_word.to_le_bytes());
                }
            }
            CastSynType::Enum {
                name_off,
                size,
                signed,
                members,
            } => {
                // BTF_KIND_ENUM: name_off + info + size_type, then vlen *
                // btf_enum{ name_off(4), val(4) }. `signed` sets info
                // bit 31, which btf-rs Enum::is_signed reads.
                type_section.extend_from_slice(&name_off.to_le_bytes());
                let vlen = members.len() as u32;
                let kf = if *signed { 1u32 << 31 } else { 0 };
                let info = ((CAST_BTF_KIND_ENUM << 24) & 0x1f00_0000) | (vlen & 0xffff) | kf;
                type_section.extend_from_slice(&info.to_le_bytes());
                type_section.extend_from_slice(&size.to_le_bytes());
                for (m_name_off, val) in members {
                    type_section.extend_from_slice(&m_name_off.to_le_bytes());
                    type_section.extend_from_slice(&val.to_le_bytes());
                }
            }
            CastSynType::Enum64 {
                name_off,
                size,
                signed,
                members,
            } => {
                // BTF_KIND_ENUM64: name_off + info + size_type, then
                // vlen * btf_enum64{ name_off(4), val_lo32(4),
                // val_hi32(4) }. `signed` sets info bit 31
                // (Enum64::is_signed); val() reconstructs (hi<<32)|lo.
                type_section.extend_from_slice(&name_off.to_le_bytes());
                let vlen = members.len() as u32;
                let kf = if *signed { 1u32 << 31 } else { 0 };
                let info = ((CAST_BTF_KIND_ENUM64 << 24) & 0x1f00_0000) | (vlen & 0xffff) | kf;
                type_section.extend_from_slice(&info.to_le_bytes());
                type_section.extend_from_slice(&size.to_le_bytes());
                for (m_name_off, val) in members {
                    type_section.extend_from_slice(&m_name_off.to_le_bytes());
                    let lo = (*val & 0xffff_ffff) as u32;
                    let hi = (*val >> 32) as u32;
                    type_section.extend_from_slice(&lo.to_le_bytes());
                    type_section.extend_from_slice(&hi.to_le_bytes());
                }
            }
            CastSynType::Float { name_off, size } => {
                // BTF_KIND_FLOAT wire layout: name_off (4) + info (4)
                // + size_type (4, the byte width). Per `cbtf`, kind 16
                // has `has_size()` true so `btf_type::size()` returns
                // `size_type`, which `btf_rs::Float::size()` surfaces;
                // vlen is 0 (no trailing records). The renderer's
                // `render_float` reads this size to pick the f32 / f64
                // decode arm.
                type_section.extend_from_slice(&name_off.to_le_bytes());
                let info = (CAST_BTF_KIND_FLOAT << 24) & 0x1f00_0000;
                type_section.extend_from_slice(&info.to_le_bytes());
                type_section.extend_from_slice(&size.to_le_bytes());
            }
        }
    }

    let type_len = type_section.len() as u32;
    let str_len = strings.len() as u32;

    let mut blob = Vec::new();
    // Header (24 bytes): magic (2) + version (1) + flags (1)
    // + hdr_len (4) + type_off (4) + type_len (4)
    // + str_off (4) + str_len (4).
    blob.extend_from_slice(&CAST_BTF_MAGIC.to_le_bytes());
    blob.push(CAST_BTF_VERSION);
    blob.push(0); // flags
    blob.extend_from_slice(&CAST_BTF_HEADER_LEN.to_le_bytes());
    blob.extend_from_slice(&0u32.to_le_bytes()); // type_off
    blob.extend_from_slice(&type_len.to_le_bytes());
    blob.extend_from_slice(&type_len.to_le_bytes()); // str_off = type_len
    blob.extend_from_slice(&str_len.to_le_bytes());
    blob.extend_from_slice(&type_section);
    blob.extend_from_slice(strings);
    blob
}

#[derive(Clone, Copy)]
pub(crate) struct CastSynMember {
    pub(crate) name_off: u32,
    pub(crate) type_id: u32,
    pub(crate) byte_offset: u32,
}

/// A bitfield member for [`CastSynType::BitfieldStruct`]. The parent
/// struct is emitted with `kind_flag == 1`, so `btf-rs`
/// `Member::bitfield_size` returns `Some(bitfield_size)` and
/// `Member::bit_offset` returns `bit_offset` (the low 24 bits of the
/// packed offset word). `bit_offset` is a RAW bit offset (not bytes);
/// `bitfield_size` is the field width in bits.
#[derive(Clone, Copy)]
pub(crate) struct CastSynBitMember {
    pub(crate) name_off: u32,
    pub(crate) type_id: u32,
    pub(crate) bit_offset: u32,
    pub(crate) bitfield_size: u32,
}

pub(crate) enum CastSynType {
    /// `BTF_KIND_INT`. encoding=0 = plain unsigned (not signed,
    /// not char, not bool — the gate the cast intercept requires).
    Int {
        name_off: u32,
        size: u32,
        encoding: u32,
        offset: u32,
        bits: u32,
    },
    Struct {
        name_off: u32,
        size: u32,
        members: Vec<CastSynMember>,
    },
    /// `BTF_KIND_TYPEDEF` (kind=8). Wraps another type id with a
    /// name. The renderer's
    /// [`crate::monitor::btf_render::peel_modifiers_with_id`] peels
    /// through it; the analyzer's
    /// [`crate::monitor::bpf_map::resolve_to_struct_id`] peels through
    /// it too. Used by the modifier-chain integration test to verify
    /// both peel paths agree on the underlying struct id the
    /// [`crate::monitor::cast_analysis::CastMap`] keys on.
    Typedef { name_off: u32, type_id: u32 },
    /// `BTF_KIND_CONST` (kind=10). Anonymous wrapper around another
    /// type id. Same renderer / analyzer peel treatment as Typedef.
    /// `name_off` is always 0 per the BTF spec (Const types are
    /// anonymous), but the field is still emitted for wire-format
    /// completeness.
    Const { type_id: u32 },
    /// `BTF_KIND_TYPE_TAG` (kind=18). Named tag wrapping `type_id`.
    /// Models `__kptr` on a pointer (`struct bpf_cpumask __kptr *`);
    /// [`crate::monitor::btf_render::peel_modifiers_with_id`] peels it
    /// to reach the wrapped Ptr.
    TypeTag { name_off: u32, type_id: u32 },
    /// `BTF_KIND_PTR` (kind=2). Anonymous pointer-to-`type_id`. Used
    /// to model a Type::Ptr field whose pointee is a forward-
    /// declared aggregate (the scenario the Fwd chase test exercises).
    Ptr { type_id: u32 },
    /// `BTF_KIND_FWD` (kind=7). Forward declaration of a struct
    /// (`is_union: false`) or union (`is_union: true`). Carries a
    /// name but no body — `type_size` returns `None`. Models the
    /// scenario where a scheduler library defines the struct (e.g.
    /// `struct sdt_data` in the sdt_alloc library) and the using
    /// program only references it via pointer; the program's BTF
    /// then carries `Fwd` rather than the full `Struct`.
    Fwd { name_off: u32, is_union: bool },
    /// A `BTF_KIND_STRUCT` emitted with `kind_flag == 1` so its members
    /// are bitfields: `btf-rs` decodes each member offset word as
    /// `bitfield_size << 24 | bit_offset` and `Member::bitfield_size`
    /// returns `Some`. The plain [`Struct`](Self::Struct) variant keeps
    /// `kind_flag == 0` (byte-aligned members); this variant isolates
    /// the bitfield encoding so non-bitfield tests stay noise-free.
    BitfieldStruct {
        name_off: u32,
        size: u32,
        members: Vec<CastSynBitMember>,
    },
    /// `BTF_KIND_ENUM` (kind=6). 32-bit-valued enum. `signed` sets the
    /// info-word `kind_flag` (bit 31), which `btf-rs` `Enum::is_signed`
    /// reads. Each `(name_off, val)` is a `btf_enum` record.
    Enum {
        name_off: u32,
        size: u32,
        signed: bool,
        members: Vec<(u32, u32)>,
    },
    /// `BTF_KIND_ENUM64` (kind=19). 64-bit-valued enum. `signed` sets
    /// the info-word `kind_flag` (bit 31), read by `Enum64::is_signed`.
    /// Each `(name_off, val)` is a `btf_enum64` record (val split into
    /// lo/hi 32-bit halves on the wire).
    Enum64 {
        name_off: u32,
        size: u32,
        signed: bool,
        members: Vec<(u32, u64)>,
    },
    /// `BTF_KIND_FLOAT` (kind=16). IEEE-754 float of `size` bytes
    /// (`btf_rs::Float::size`). The renderer's `render_float` decodes
    /// 4-byte slices as `f32` and 8-byte slices as `f64`; any other
    /// declared `size` routes to the `Unsupported` arm. Used by the
    /// float-decode tests so the byte-decode path runs through a real
    /// BTF float rather than a hand-built `RenderedValue::Float`.
    Float { name_off: u32, size: u32 },
}
