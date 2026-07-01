use super::super::cast_analysis::AddrSpace;
use super::*;

/// Build a Btf instance using the project's standard vmlinux
/// resolver (`find_test_vmlinux`) and BTF loader
/// (`load_btf_from_path`). Both honour the `KTSTR_KERNEL` env var,
/// the local-tree fallbacks, and the BTF sidecar cache that real
/// monitor code relies on, so tests don't drift onto a different
/// resolution path that masks bugs in the production loader.
///
/// Returns `None` when `find_test_vmlinux` skips (surfacing a
/// `test_skip` message in that path so the user sees the reason) or
/// when `load_btf_from_path` fails to load/parse the BTF (via `.ok()`,
/// silently).
fn test_btf() -> Option<Btf> {
    let path = crate::monitor::find_test_vmlinux()?;
    crate::monitor::btf_offsets::load_btf_from_path(&path).ok()
}

/// Compact constructor for [`RenderedValue::Enum`] in tests.
/// Mirrors the `uint_v` helper in `scenario::snapshot`'s test module — folds
/// the 4-field struct literal (now that `is_signed` is part of the
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

// ----- Shared cast-test fixtures (hoisted to test_support::btf_blob) -----
//
// The synthetic BTF builder (`cast_build_btf`), its input records
// (`CastSynType`, `CastSynMember`, `CastSynBitMember`), and the
// internal `CAST_BTF_*` kind constants now live in
// `crate::test_support::btf_blob` so the `kernel_op_dispatch`
// BTF-gated dispatch tests can build the same blobs. Re-exported here
// so the `cast_*` submodules below keep resolving them through
// `super::*` unchanged. The `CAST_BTF_*` consts stay private to
// `btf_blob` (only `cast_build_btf` uses them).
pub(crate) use crate::test_support::btf_blob::{
    CastSynBitMember, CastSynMember, CastSynType, cast_build_btf,
};

// -- render_bitfield coverage --
//
// render_bitfield decodes a struct bitfield member. It
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

fn struct_with(members: Vec<(&str, RenderedValue)>) -> RenderedValue {
    RenderedValue::Struct {
        type_name: None,
        members: members
            .into_iter()
            .map(|(n, v)| RenderedMember {
                name: n.to_string(),
                value: v,
            })
            .collect(),
    }
}

/// Pull the sole member's value out of a rendered single-member
/// struct, transparently peeling the `Truncated` wrapper that
/// `render_struct` adds when `bytes.len() < struct size`.
fn sole_member_value(v: &RenderedValue) -> &RenderedValue {
    match v {
        RenderedValue::Struct { members, .. } => {
            assert_eq!(members.len(), 1, "expected exactly one member");
            &members[0].value
        }
        RenderedValue::Truncated { partial, .. } => sole_member_value(partial),
        other => panic!("expected Struct/Truncated, got {other:?}"),
    }
}

mod cast_intercept;
mod cast_kernel_arm;
mod cast_pipeline;
mod chase_edge_cases;
mod cpumask_render;
mod datasec_cpumask;
mod display_basics;
mod fwd_sibling;
mod predicates;
mod render_value_coverage;
mod rendered_value_accessors;
mod sdt_bridge;
mod templates_arrays_cycles;
mod typed_arrays_dedup;
