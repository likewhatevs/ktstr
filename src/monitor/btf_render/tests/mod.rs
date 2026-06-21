use super::super::cast_analysis::AddrSpace;
use super::*;

/// Build a Btf instance using the project's standard vmlinux
/// resolver (`find_test_vmlinux`) and BTF loader
/// (`load_btf_from_path`). Both honour the `KTSTR_KERNEL` env var,
/// the local-tree fallbacks, and the BTF sidecar cache that real
/// monitor code relies on, so tests don't drift onto a different
/// resolution path that masks bugs in the production loader.
///
/// Returns `None` only when `find_test_vmlinux` decides to skip;
/// it surfaces a `test_skip` message in that path so the user sees
/// the reason rather than a silent no-op.
fn test_btf() -> Option<Btf> {
    let path = crate::monitor::find_test_vmlinux()?;
    crate::monitor::btf_offsets::load_btf_from_path(&path).ok()
}

/// Compact constructor for [`RenderedValue::Enum`] in tests.
/// Mirrors `uint_v` in src/scenario/snapshot/tests.rs:3443 — folds
/// the 5-field struct literal (now that `is_signed` is part of the
/// shape) into a one-line call so fixture sites stay readable as
/// the variant grows.
fn enum_v(bits: u32, value: i64, variant: Option<&str>, is_signed: bool) -> RenderedValue {
    RenderedValue::Enum {
        bits,
        value,
        variant: variant.map(String::from),
        is_signed,
    }
}


// ----- Shared cast-test fixtures (hoisted from cast_intercept) -----

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

/// Build a minimal BTF blob containing `types` (id=1..) and a
/// string-section payload `strings` (must start with `\0`). The
/// header layout matches `cast_analysis::tests::build_btf`:
/// 24-byte header, type section, string section. Supports the BTF
/// kinds the renderer tests exercise: Int, Struct, BitfieldStruct,
/// Enum, Enum64, Typedef, Const, TypeTag, Ptr, Fwd.
fn cast_build_btf(types: &[CastSynType], strings: &[u8]) -> Vec<u8> {
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
struct CastSynMember {
    name_off: u32,
    type_id: u32,
    byte_offset: u32,
}

/// A bitfield member for [`CastSynType::BitfieldStruct`]. The parent
/// struct is emitted with `kind_flag == 1`, so `btf-rs`
/// `Member::bitfield_size` returns `Some(bitfield_size)` and
/// `Member::bit_offset` returns `bit_offset` (the low 24 bits of the
/// packed offset word). `bit_offset` is a RAW bit offset (not bytes);
/// `bitfield_size` is the field width in bits.
#[derive(Clone, Copy)]
struct CastSynBitMember {
    name_off: u32,
    type_id: u32,
    bit_offset: u32,
    bitfield_size: u32,
}

enum CastSynType {
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
    /// name. The renderer's [`peel_modifiers_with_id`] peels through
    /// it; the analyzer's [`super::super::bpf_map::resolve_to_struct_id`]
    /// peels through it too. Used by the modifier-chain integration
    /// test to verify both peel paths agree on the underlying
    /// struct id the [`CastMap`] keys on.
    Typedef { name_off: u32, type_id: u32 },
    /// `BTF_KIND_CONST` (kind=10). Anonymous wrapper around another
    /// type id. Same renderer / analyzer peel treatment as Typedef.
    /// `name_off` is always 0 per the BTF spec (Const types are
    /// anonymous), but the field is still emitted for wire-format
    /// completeness.
    Const { type_id: u32 },
    /// `BTF_KIND_TYPE_TAG` (kind=18). Named tag wrapping `type_id`.
    /// Models `__kptr` on a pointer (`struct bpf_cpumask __kptr *`);
    /// `peel_modifiers_with_id` peels it to reach the wrapped Ptr.
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
}

// -- render_bitfield coverage --
//
// render_bitfield (mod.rs ~4490) decodes a struct bitfield member. It
// is reached from render_member only when the owning struct has
// kind_flag == 1 and the member width > 0, which the synthetic builder
// expresses via CastSynType::BitfieldStruct + CastSynBitMember. The
// width == 0 branch is unreachable through render_value (render_member
// filters width == 0), so it is exercised by a direct render_bitfield
// call.


/// Helper: build a string section + name offsets for the names
/// used across cast tests. Returns `(strings, n_int_name, n_t,
/// n_q, n_f, n_x)` where `n_*` are the byte offsets of each name
/// inside the string section.
fn cast_strings_for_t_q() -> (Vec<u8>, u32, u32, u32, u32, u32) {
    let mut strings: Vec<u8> = vec![0];
    let push = |s: &mut Vec<u8>, name: &str| -> u32 {
        let off = s.len() as u32;
        s.extend_from_slice(name.as_bytes());
        s.push(0);
        off
    };
    let n_int = push(&mut strings, "u64");
    let n_t = push(&mut strings, "T");
    let n_q = push(&mut strings, "Q");
    let n_f = push(&mut strings, "f");
    let n_x = push(&mut strings, "x");
    (strings, n_int, n_t, n_q, n_f, n_x)
}

/// Build a BTF blob with: id=1 plain-unsigned u64 (size=8,bits=64),
/// id=2 struct T { u64 f at offset 0; } size=8, id=3 struct Q
/// { u64 x at offset 0; } size=8. T_id=2, Q_id=3.
fn cast_btf_t_and_q() -> (Vec<u8>, u32, u32) {
    let (strings, n_int, n_t, n_q, n_f, n_x) = cast_strings_for_t_q();
    let types = vec![
        CastSynType::Int {
            name_off: n_int,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        CastSynType::Struct {
            name_off: n_t,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_f,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        CastSynType::Struct {
            name_off: n_q,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_x,
                type_id: 1,
                byte_offset: 0,
            }],
        },
    ];
    (cast_build_btf(&types, &strings), 2, 3)
}

/// Stub `MemReader` for cast-intercept tests. Two cast-lookup
/// modes:
///
/// - `cast_map = Some(map)` — looks up
///   `(parent_type_id, member_byte_offset)` in a real
///   [`crate::monitor::cast_analysis::CastMap`] (typically produced by
///   [`crate::monitor::cast_analysis::analyze_casts`]) and returns
///   the matching [`CastHit`]. The integration tests use this mode
///   to wire actual analyzer output into the renderer.
/// - `cast_map = None` — returns the fixed `hit` (or `None` when
///   `hit` is `None`) for every query. The unit tests for the
///   intercept gate use this mode because they only need the
///   intercept to fire or not fire on a single (parent, offset)
///   pair.
///
/// `arena_bytes_at` and `kva_bytes_at` drive the address-space
/// dispatch; tests that don't exercise reads leave the maps empty.
///
/// `arena_type_at` carries the sdt_alloc bridge entries the
/// renderer's [`MemReader::resolve_arena_type`] override consults
/// — `addr → btf_type_id`. Mirrors the production
/// `AccessorMemReader::resolve_arena_type` shape (the `dump/render_map.rs`
/// override masks the address with `0xFFFF_FFFF` and looks up in
/// the per-pass index); the stub keeps full addresses keyed
/// directly so tests can use the actual chased value.
#[derive(Default)]
struct CastStubReader {
    /// Fixed [`CastHit`] returned by `cast_lookup` when `cast_map`
    /// is `None`. Universal-match avoids hand-keying the same
    /// (id, offset) pair into every gate-focused test.
    hit: Option<CastHit>,
    /// Real cast map consulted when `Some`. The
    /// `(parent_type_id, member_byte_offset)` lookup mirrors
    /// `AccessorMemReader::cast_lookup` in `dump/render_map.rs`
    /// (the production path), so the integration tests cover the
    /// same shape.
    cast_map: Option<crate::monitor::cast_analysis::CastMap>,
    arena_window: Option<(u64, u64)>,
    arena_bytes_at: std::collections::HashMap<u64, Vec<u8>>,
    kva_bytes_at: std::collections::HashMap<u64, Vec<u8>>,
    /// `addr → ArenaResolveHit` lookup the stub returns from
    /// [`MemReader::resolve_arena_type`]. Empty by default — the
    /// trait method then surfaces the trait-default `None` for
    /// every query, matching every existing test that does not
    /// exercise the sdt_alloc bridge. The
    /// [`ArenaResolveHit::header_skip`] field is the byte count
    /// the chase must skip from `addr` before the payload struct
    /// begins (0 for payload-start chases, the slot's header size
    /// for slot-start chases) — see
    /// [`MemReader::resolve_arena_type`] for the production
    /// contract.
    arena_type_at: std::collections::HashMap<u64, ArenaResolveHit>,
    /// Owned BTFs the stub holds for cross-BTF Fwd resolution.
    /// `cross_btf_resolve_fwd` returns a borrow into this vec when
    /// `cross_btf_index` has a hit. None / empty disables the
    /// trait method's response (default `None`).
    cross_btf_btfs: Vec<std::sync::Arc<Btf>>,
    /// `name -> (cross_btf_btfs index, type_id, want_struct)` for
    /// cross-BTF Fwd resolution. The `bool` is the
    /// aggregate-kind flag the trait gates on
    /// (`true = Type::Struct`, `false = Type::Union`); a
    /// stored entry only fires when the query's `kind`
    /// matches.
    cross_btf_index: std::collections::HashMap<String, (usize, u32, bool)>,
    /// Set of low-32 windowed slot starts the dump pre-pass would
    /// have already rendered. The
    /// [`MemReader::is_already_rendered`] override returns `true`
    /// when `addr as u32` lies in this set so the chase
    /// short-circuits to a `deref: None` Ptr with the "already
    /// rendered" reason — mirrors the production
    /// `AccessorMemReader` dedup wired through the
    /// `rendered_slot_addrs` field. Empty by default so existing
    /// tests stay untouched.
    rendered_slot_addrs: std::collections::HashSet<u32>,
}

impl MemReader for CastStubReader {
    fn read_kva(&self, kva: u64, len: usize) -> Option<Vec<u8>> {
        let bytes = self.kva_bytes_at.get(&kva)?;
        if bytes.len() < len {
            return None;
        }
        Some(bytes[..len].to_vec())
    }
    fn is_arena_addr(&self, addr: u64) -> bool {
        match self.arena_window {
            Some((lo, hi)) => addr >= lo && addr < hi,
            None => false,
        }
    }
    fn read_arena(&self, addr: u64, len: usize) -> Option<Vec<u8>> {
        let bytes = self.arena_bytes_at.get(&addr)?;
        if bytes.len() < len {
            return None;
        }
        Some(bytes[..len].to_vec())
    }
    fn cast_lookup(&self, parent_type_id: u32, member_byte_offset: u32) -> Option<CastHit> {
        // CastMap mode: look up (parent, offset) in the analyzer's
        // output. Mirrors the production `AccessorMemReader::cast_lookup`
        // so the integration tests cover the same key/value shape.
        if let Some(map) = &self.cast_map {
            return map.get(&(parent_type_id, member_byte_offset)).copied();
        }
        // Fixed-hit mode (default): return the canned hit
        // regardless of (parent, offset). Used by gate-focused
        // unit tests above.
        self.hit
    }
    fn resolve_arena_type(&self, addr: u64) -> Option<ArenaResolveHit> {
        self.arena_type_at.get(&addr).copied()
    }
    fn cross_btf_resolve_fwd(
        &self,
        name: &str,
        kind: super::FwdKind,
    ) -> Option<super::CrossBtfRef<'_>> {
        let &(idx, type_id, idx_is_struct) = self.cross_btf_index.get(name)?;
        let idx_kind = super::FwdKind::from_is_struct(idx_is_struct);
        if idx_kind != kind {
            return None;
        }
        let btf = self.cross_btf_btfs.get(idx)?;
        Some(super::CrossBtfRef {
            btf: btf.as_ref(),
            type_id,
        })
    }
    fn is_already_rendered(&self, addr: u64) -> bool {
        self.rendered_slot_addrs.contains(&(addr as u32))
    }
}


// ----- Shared RenderedValue constructor -----

fn uint(value: u64) -> RenderedValue {
    RenderedValue::Uint { bits: 64, value }
}

mod display_basics;
mod datasec_cpumask;
mod predicates;
mod templates_arrays_cycles;
mod cast_intercept;
mod cast_pipeline;
mod cast_kernel_arm;
mod fwd_sibling;
mod chase_edge_cases;
mod sdt_bridge;
mod rendered_value_accessors;
mod cpumask_render;
mod typed_arrays_dedup;
