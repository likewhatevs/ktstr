//! Unit tests for [`super`] (the `probe::btf` module).
//! Co-located via the `tests` submodule pattern (sibling file).

#![cfg(test)]

use super::*;

// -- STRUCT_FIELDS constant --

#[test]
fn struct_fields_has_task_struct() {
    let entry = STRUCT_FIELDS.iter().find(|(s, _)| *s == "task_struct");
    assert!(entry.is_some());
    let (_, fields) = entry.unwrap();
    assert!(fields.iter().any(|(_, k)| *k == "pid"));
    assert!(fields.iter().any(|(_, k)| *k == "dsq_id"));
    assert!(fields.iter().any(|(_, k)| *k == "cpumask_0"));
}

#[test]
fn struct_fields_has_rq() {
    let entry = STRUCT_FIELDS.iter().find(|(s, _)| *s == "rq");
    assert!(entry.is_some());
    let (_, fields) = entry.unwrap();
    assert!(fields.iter().any(|(_, k)| *k == "cpu"));
}

// -- parse_btf_functions (requires /sys/kernel/btf/vmlinux) --

#[test]
fn parse_btf_known_kernel_func() {
    if !std::path::Path::new("/sys/kernel/btf/vmlinux").exists() {
        skip!("/sys/kernel/btf/vmlinux not present");
    }
    let funcs = parse_btf_functions(&["do_exit"], None);
    if funcs.is_empty() {
        skip!("BTF load returned no functions — host BTF may lack do_exit");
    }
    assert_eq!(funcs[0].name, "do_exit");
    assert!(!funcs[0].params.is_empty());
}

#[test]
fn parse_btf_unknown_func_returns_empty() {
    let funcs = parse_btf_functions(&["__totally_fake_function_name__"], None);
    assert!(funcs.is_empty());
}

// -- discover_bpf_symbols --

#[test]
fn discover_bpf_symbols_no_scheduler() {
    // Empty stack_names gates off the non-StructOps discovery branch, so
    // every returned entry must be a StructOps program (is_bpf). The set
    // is host-dependent (the StructOps progs loaded on the test host),
    // so emptiness/count can't be asserted — but the empty-input
    // contract (StructOps-only) can.
    let result = discover_bpf_symbols(&[]);
    assert!(
        result.iter().all(|f| f.is_bpf),
        "empty stack_names must yield only StructOps (is_bpf) entries",
    );
}

// -- STRUCT_FIELDS invariants --

#[test]
fn struct_fields_all_entries_have_fields() {
    for (name, fields) in STRUCT_FIELDS {
        assert!(!fields.is_empty(), "struct {name} has no fields");
    }
}

#[test]
fn struct_fields_no_duplicate_structs() {
    let names: Vec<&str> = STRUCT_FIELDS.iter().map(|(n, _)| *n).collect();
    let unique: std::collections::HashSet<&&str> = names.iter().collect();
    assert_eq!(
        names.len(),
        unique.len(),
        "duplicate struct names in STRUCT_FIELDS"
    );
}

#[test]
fn struct_fields_no_duplicate_keys_per_struct() {
    for (name, fields) in STRUCT_FIELDS {
        let keys: Vec<&str> = fields.iter().map(|(_, k)| *k).collect();
        let unique: std::collections::HashSet<&&str> = keys.iter().collect();
        assert_eq!(
            keys.len(),
            unique.len(),
            "struct {name} has duplicate field keys"
        );
    }
}

#[test]
fn struct_fields_has_scx_dispatch_q() {
    let entry = STRUCT_FIELDS.iter().find(|(s, _)| *s == "scx_dispatch_q");
    assert!(entry.is_some());
}

#[test]
fn struct_fields_has_scx_exit_info() {
    let entry = STRUCT_FIELDS.iter().find(|(s, _)| *s == "scx_exit_info");
    assert!(entry.is_some());
}

#[test]
fn struct_fields_has_scx_init_task_args() {
    let entry = STRUCT_FIELDS
        .iter()
        .find(|(s, _)| *s == "scx_init_task_args");
    assert!(entry.is_some());
}

#[test]
fn struct_fields_has_scx_cgroup_init_args() {
    let entry = STRUCT_FIELDS
        .iter()
        .find(|(s, _)| *s == "scx_cgroup_init_args");
    assert!(entry.is_some());
}

// -- BtfParam/BtfFunc construction --

#[test]
fn btf_param_debug_display() {
    let p = BtfParam {
        name: "test".into(),
        struct_name: Some("task_struct".into()),
        is_ptr: true,
        ..Default::default()
    };
    let dbg = format!("{:?}", p);
    assert!(dbg.contains("task_struct"));
    assert!(dbg.contains("test"));
}

// -- BtfFunc construction --

#[test]
fn btf_func_empty_params() {
    let f = BtfFunc {
        name: "empty".into(),
        params: vec![],
        is_variadic: false,
    };
    assert_eq!(f.name, "empty");
    assert!(f.params.is_empty());
}

#[test]
fn btf_func_multiple_params() {
    let f = BtfFunc {
        name: "multi".into(),
        params: vec![
            BtfParam {
                name: "a".into(),
                struct_name: Some("task_struct".into()),
                is_ptr: true,
                ..Default::default()
            },
            BtfParam {
                name: "b".into(),
                struct_name: Some("rq".into()),
                is_ptr: true,
                ..Default::default()
            },
            BtfParam {
                name: "c".into(),
                struct_name: None,
                is_ptr: false,
                ..Default::default()
            },
        ],
        is_variadic: false,
    };
    assert_eq!(f.params.len(), 3);
    assert!(f.params[0].is_ptr);
    assert!(f.params[2].struct_name.is_none());
}

// -- parse_btf_functions with multiple names --

#[test]
fn parse_btf_multiple_unknown_funcs() {
    let funcs = parse_btf_functions(&["__fake_a__", "__fake_b__", "__fake_c__"], None);
    assert!(funcs.is_empty());
}

// -- STRUCT_FIELDS field access patterns --

#[test]
fn struct_fields_has_cpumask_words() {
    let entry = STRUCT_FIELDS.iter().find(|(s, _)| *s == "task_struct");
    let (_, fields) = entry.unwrap();
    assert!(fields.iter().any(|(_, k)| *k == "cpumask_0"));
    assert!(fields.iter().any(|(_, k)| *k == "cpumask_1"));
    assert!(fields.iter().any(|(_, k)| *k == "cpumask_2"));
    assert!(fields.iter().any(|(_, k)| *k == "cpumask_3"));
}

#[test]
fn struct_fields_cpumask_access_patterns() {
    let entry = STRUCT_FIELDS.iter().find(|(s, _)| *s == "task_struct");
    let (_, fields) = entry.unwrap();
    assert!(fields.iter().any(|(a, _)| *a == "->cpus_ptr->bits[0]"));
    assert!(fields.iter().any(|(a, _)| *a == "->cpus_ptr->bits[1]"));
    assert!(fields.iter().any(|(a, _)| *a == "->cpus_ptr->bits[2]"));
    assert!(fields.iter().any(|(a, _)| *a == "->cpus_ptr->bits[3]"));
}

#[test]
fn struct_fields_task_struct_field_count() {
    let entry = STRUCT_FIELDS.iter().find(|(s, _)| *s == "task_struct");
    let (_, fields) = entry.unwrap();
    // pid + 4 cpumask words + dsq_id + enq_flags + slice + vtime +
    // weight + sticky_cpu + scx_flags = 12
    assert_eq!(fields.len(), 12);
}

#[test]
fn struct_fields_auto_discover_cpumask_pattern() {
    let ts_fields = STRUCT_FIELDS
        .iter()
        .find(|(s, _)| *s == "task_struct")
        .unwrap()
        .1;
    let cpumask_accesses: Vec<&&str> = ts_fields
        .iter()
        .filter(|(a, _)| a.contains("bits["))
        .map(|(a, _)| a)
        .collect();
    assert_eq!(cpumask_accesses.len(), 4);
    for (i, a) in cpumask_accesses.iter().enumerate() {
        assert!(
            a.contains(&format!("bits[{i}]")),
            "expected bits[{i}] in {a}",
        );
    }
}

#[test]
fn struct_fields_access_patterns_start_with_arrow() {
    for (name, fields) in STRUCT_FIELDS {
        for (access, _key) in *fields {
            assert!(
                access.starts_with("->"),
                "struct {name} field access '{access}' should start with '->'"
            );
        }
    }
}

// -- parse_btf_functions / resolve_field_specs against ELF vmlinux --
//
// btf_rs::Btf::from_file only accepts raw-BTF magic and fails on an
// ELF vmlinux. The previous implementation swallowed that failure
// and silently returned an empty Vec. The regression guard: give
// both entry points an ELF vmlinux and assert they produce
// non-empty results.

#[test]
fn parse_btf_functions_accepts_elf_vmlinux() {
    let Some(path) = crate::monitor::find_test_vmlinux() else {
        skip!("no vmlinux found; {}", crate::KTSTR_KERNEL_HINT);
    };
    if path.starts_with("/sys/") {
        skip!("vmlinux is raw BTF, this test exercises the ELF path");
    }
    let funcs = parse_btf_functions(&["do_exit"], Some(path.to_str().unwrap()));
    assert!(
        !funcs.is_empty(),
        "ELF vmlinux should yield a BtfFunc for do_exit"
    );
    assert_eq!(funcs[0].name, "do_exit");
    assert!(
        !funcs[0].params.is_empty(),
        "do_exit should have at least one parameter resolved from BTF"
    );
}

#[test]
fn resolve_field_specs_accepts_elf_vmlinux() {
    let Some(path) = crate::monitor::find_test_vmlinux() else {
        skip!("no vmlinux found; {}", crate::KTSTR_KERNEL_HINT);
    };
    if path.starts_with("/sys/") {
        skip!("vmlinux is raw BTF, this test exercises the ELF path");
    }
    // Minimal BtfFunc that maps a STRUCT_FIELDS-eligible param so
    // resolve_field_specs has something to resolve against the BTF.
    let func = BtfFunc {
        name: "test_fn".to_string(),
        params: vec![BtfParam {
            name: "p".to_string(),
            struct_name: Some("task_struct".to_string()),
            is_ptr: true,
            ..Default::default()
        }],
        is_variadic: false,
    };
    let specs = resolve_field_specs(&func, Some(path.to_str().unwrap()));
    assert!(
        !specs.is_empty(),
        "task_struct STRUCT_FIELDS should resolve from ELF vmlinux BTF"
    );
}

// ===================================================================
// Synthetic raw-BTF builder + host-only parser tests.
//
// These exercise the pure btf_rs parsers ([`resolve_field_specs_with_btf`],
// [`resolve_member_offset`], [`resolve_type_size`], [`resolve_pointed_struct`],
// [`parse_btf_functions`], [`discover_vmlinux_struct_fields`],
// [`classify_vmlinux_member`], [`emit_member_field`]) against
// hand-built raw-BTF blobs parsed via [`btf_rs::Btf::from_bytes`].
// No kernel, no vmlinux, no VM — fully host-runnable.
//
// Layout encoded here follows include/uapi/linux/btf.h and the
// btf-rs cbtf decoders: 24-byte header (magic 0xEB9F), then the type
// section, then the string section. Type ids start at 1 (id 0 = Void).
// ===================================================================

/// btf-rs kind ids (include/uapi/linux/btf.h).
const K_INT: u32 = 1;
const K_PTR: u32 = 2;
const K_ARRAY: u32 = 3;
const K_STRUCT: u32 = 4;
const K_UNION: u32 = 5;
const K_ENUM: u32 = 6;
const K_TYPEDEF: u32 = 8;
const K_VOLATILE: u32 = 9;
const K_CONST: u32 = 10;
const K_FUNC: u32 = 12;
const K_FUNC_PROTO: u32 = 13;
const K_ENUM64: u32 = 19;

/// BTF_INT encoding bits (cbtf::BTF_INT_*).
const INT_SIGNED: u32 = 1 << 0;
const INT_BOOL: u32 = 1 << 2;

/// `BTF_FUNC_EXTERN` linkage (vlen field of a `Func` type).
const FUNC_EXTERN: u32 = 2;

/// A struct/union member: (name_offset, type_id, byte_offset).
#[derive(Clone, Copy)]
struct M {
    name_off: u32,
    type_id: u32,
    byte_off: u32,
}

/// A FuncProto parameter: (name_offset, type_id). A `(0, 0)` pair is
/// the variadic sentinel.
#[derive(Clone, Copy)]
struct P {
    name_off: u32,
    type_id: u32,
}

/// One synthetic BTF type. `BtfBuilder` appends these in order; the
/// first added type is id 1.
enum Ty {
    /// `size` = byte width (1/2/4/8); `encoding` = BTF_INT_* bits.
    Int {
        size: u32,
        encoding: u32,
    },
    Ptr {
        type_id: u32,
    },
    /// `elem` = element type id; `nelems` = element count.
    Array {
        elem: u32,
        nelems: u32,
    },
    Struct {
        name_off: u32,
        size: u32,
        members: Vec<M>,
    },
    Union {
        name_off: u32,
        size: u32,
        members: Vec<M>,
    },
    /// `size` = byte width stored in size_type (read back by
    /// `Enum::size()`).
    Enum {
        name_off: u32,
        size: u32,
    },
    Enum64 {
        name_off: u32,
    },
    /// Const/Volatile/Typedef qualifier referencing `type_id`.
    Const {
        type_id: u32,
    },
    Volatile {
        type_id: u32,
    },
    Typedef {
        name_off: u32,
        type_id: u32,
    },
    /// FuncProto returning `ret` with the given params.
    FuncProto {
        ret: u32,
        params: Vec<P>,
    },
    /// Func named `name_off` pointing at FuncProto id `type_id`.
    Func {
        name_off: u32,
        type_id: u32,
    },
}

/// Builds a raw-BTF blob plus a string table for [`btf_rs::Btf::from_bytes`].
struct BtfBuilder {
    types: Vec<Ty>,
    strings: Vec<u8>,
}

impl BtfBuilder {
    fn new() -> Self {
        // strtab[0] must be NUL so name_off 0 resolves to "" (anonymous).
        BtfBuilder {
            types: Vec::new(),
            strings: vec![0],
        }
    }

    /// Intern `name`, returning its strtab offset.
    fn name(&mut self, name: &str) -> u32 {
        let off = self.strings.len() as u32;
        self.strings.extend_from_slice(name.as_bytes());
        self.strings.push(0);
        off
    }

    /// Append a type, returning its 1-based id.
    fn add(&mut self, ty: Ty) -> u32 {
        self.types.push(ty);
        self.types.len() as u32
    }

    /// Common 12-byte `btf_type` header: name_off, info, size_type.
    fn push_hdr(section: &mut Vec<u8>, name_off: u32, kind: u32, vlen: u32, size_type: u32) {
        section.extend_from_slice(&name_off.to_le_bytes());
        let info = ((kind << 24) & 0x1f00_0000) | (vlen & 0xffff);
        section.extend_from_slice(&info.to_le_bytes());
        section.extend_from_slice(&size_type.to_le_bytes());
    }

    /// Encode header + string sections into a parseable blob.
    fn build(&self) -> Vec<u8> {
        let mut sec = Vec::new();
        for ty in &self.types {
            match ty {
                Ty::Int { size, encoding } => {
                    Self::push_hdr(&mut sec, 0, K_INT, 0, *size);
                    // btf_int.data: encoding<<24 | offset<<16 | bits.
                    let data = (*encoding << 24) | (size * 8);
                    sec.extend_from_slice(&data.to_le_bytes());
                }
                Ty::Ptr { type_id } => Self::push_hdr(&mut sec, 0, K_PTR, 0, *type_id),
                Ty::Array { elem, nelems } => {
                    Self::push_hdr(&mut sec, 0, K_ARRAY, 0, 0);
                    // btf_array: type, index_type, nelems.
                    sec.extend_from_slice(&elem.to_le_bytes());
                    sec.extend_from_slice(&K_INT.to_le_bytes()); // index_type (any int)
                    sec.extend_from_slice(&nelems.to_le_bytes());
                }
                Ty::Struct {
                    name_off,
                    size,
                    members,
                }
                | Ty::Union {
                    name_off,
                    size,
                    members,
                } => {
                    let kind = if matches!(ty, Ty::Union { .. }) {
                        K_UNION
                    } else {
                        K_STRUCT
                    };
                    Self::push_hdr(&mut sec, *name_off, kind, members.len() as u32, *size);
                    for m in members {
                        sec.extend_from_slice(&m.name_off.to_le_bytes());
                        sec.extend_from_slice(&m.type_id.to_le_bytes());
                        sec.extend_from_slice(&(m.byte_off * 8).to_le_bytes());
                    }
                }
                Ty::Enum { name_off, size } => {
                    // vlen 0 (no enumerators needed for size/classification).
                    Self::push_hdr(&mut sec, *name_off, K_ENUM, 0, *size);
                }
                Ty::Enum64 { name_off } => {
                    Self::push_hdr(&mut sec, *name_off, K_ENUM64, 0, 8);
                }
                Ty::Const { type_id } => Self::push_hdr(&mut sec, 0, K_CONST, 0, *type_id),
                Ty::Volatile { type_id } => Self::push_hdr(&mut sec, 0, K_VOLATILE, 0, *type_id),
                Ty::Typedef { name_off, type_id } => {
                    Self::push_hdr(&mut sec, *name_off, K_TYPEDEF, 0, *type_id);
                }
                Ty::FuncProto { ret, params } => {
                    Self::push_hdr(&mut sec, 0, K_FUNC_PROTO, params.len() as u32, *ret);
                    for p in params {
                        sec.extend_from_slice(&p.name_off.to_le_bytes());
                        sec.extend_from_slice(&p.type_id.to_le_bytes());
                    }
                }
                Ty::Func { name_off, type_id } => {
                    // vlen carries linkage for Func.
                    Self::push_hdr(&mut sec, *name_off, K_FUNC, FUNC_EXTERN, *type_id);
                }
            }
        }

        let type_len = sec.len() as u32;
        let str_len = self.strings.len() as u32;
        let mut blob = Vec::new();
        blob.extend_from_slice(&0xEB9F_u16.to_le_bytes()); // magic
        blob.push(1); // version
        blob.push(0); // flags
        blob.extend_from_slice(&24u32.to_le_bytes()); // hdr_len
        blob.extend_from_slice(&0u32.to_le_bytes()); // type_off
        blob.extend_from_slice(&type_len.to_le_bytes()); // type_len
        blob.extend_from_slice(&type_len.to_le_bytes()); // str_off
        blob.extend_from_slice(&str_len.to_le_bytes()); // str_len
        blob.extend_from_slice(&sec);
        blob.extend_from_slice(&self.strings);
        blob
    }

    /// Build and parse, panicking on a malformed blob.
    fn parse(&self) -> btf_rs::Btf {
        btf_rs::Btf::from_bytes(&self.build()).expect("synthetic BTF should parse")
    }

    /// Emit a bespoke blob: an `u32` (id 1) plus a struct holding one
    /// member at a raw `bit_off` (NOT multiplied by 8). Used to place a
    /// member at a non-byte-aligned offset, which `build()` cannot
    /// express (it multiplies `byte_off` by 8). `self` is consumed for
    /// its string table (which already holds the names).
    fn build_with_raw_member(
        &self,
        struct_name_off: u32,
        member_name_off: u32,
        member_type_id: u32,
        bit_off: u32,
    ) -> Vec<u8> {
        let mut sec = Vec::new();
        // id 1: u32.
        Self::push_hdr(&mut sec, 0, K_INT, 0, 4);
        sec.extend_from_slice(&32u32.to_le_bytes()); // bits=32
        // id 2: struct { member @ bit_off }.
        Self::push_hdr(&mut sec, struct_name_off, K_STRUCT, 1, 8);
        sec.extend_from_slice(&member_name_off.to_le_bytes());
        sec.extend_from_slice(&member_type_id.to_le_bytes());
        sec.extend_from_slice(&bit_off.to_le_bytes());

        let type_len = sec.len() as u32;
        let str_len = self.strings.len() as u32;
        let mut blob = Vec::new();
        blob.extend_from_slice(&0xEB9F_u16.to_le_bytes());
        blob.push(1);
        blob.push(0);
        blob.extend_from_slice(&24u32.to_le_bytes());
        blob.extend_from_slice(&0u32.to_le_bytes());
        blob.extend_from_slice(&type_len.to_le_bytes());
        blob.extend_from_slice(&type_len.to_le_bytes());
        blob.extend_from_slice(&str_len.to_le_bytes());
        blob.extend_from_slice(&sec);
        blob.extend_from_slice(&self.strings);
        blob
    }
}

/// The synthetic builder must round-trip through `btf_rs::Btf::from_bytes`:
/// a named int is resolvable by name and reports its size.
#[test]
fn synthetic_btf_parses_and_resolves() {
    let mut b = BtfBuilder::new();
    let id_u32 = b.add(Ty::Int {
        size: 4,
        encoding: 0,
    }); // id 1
    let sn = b.name("holder");
    let xn = b.name("x");
    b.add(Ty::Struct {
        name_off: sn,
        size: 4,
        members: vec![M {
            name_off: xn,
            type_id: id_u32,
            byte_off: 0,
        }],
    });
    let btf = b.parse();
    let s = resolve_struct_type(&btf, "holder").expect("holder resolves");
    assert_eq!(s.size(), 4);
    assert_eq!(s.members.len(), 1);
}

// ---- resolve_struct_type ----

#[test]
fn resolve_struct_type_found_and_missing() {
    let mut b = BtfBuilder::new();
    let id_u64 = b.add(Ty::Int {
        size: 8,
        encoding: 0,
    }); // id1
    let sn = b.name("rq");
    let cn = b.name("cpu");
    b.add(Ty::Struct {
        name_off: sn,
        size: 8,
        members: vec![M {
            name_off: cn,
            type_id: id_u64,
            byte_off: 0,
        }],
    });
    let btf = b.parse();
    assert!(resolve_struct_type(&btf, "rq").is_some());
    assert!(resolve_struct_type(&btf, "no_such_struct").is_none());
}

// ---- resolve_member_offset ----

/// Helper: build a `task_struct`-shaped BTF with an embedded `scx`
/// sub-struct and a `cpus_ptr` pointer to a `cpumask { bits[4] }`.
/// Returns the parsed BTF. Layout (ids as added):
///   id1 u64, id2 s32,
///   id3 u64[4], id4 cpumask { u64 bits[4] @0 } size 32,
///   id5 ptr->cpumask,
///   id6 scx_entity { u64 slice @0; u64 dsq_vtime @8; s32 weight @16 } size 24,
///   id7 task_struct { s32 pid @0; ptr cpus_ptr @8; scx_entity scx @16 } size 40
fn build_task_struct_btf() -> (btf_rs::Btf, BtfBuilder) {
    let mut b = BtfBuilder::new();
    let id_u64 = b.add(Ty::Int {
        size: 8,
        encoding: 0,
    }); // id1
    let id_s32 = b.add(Ty::Int {
        size: 4,
        encoding: INT_SIGNED,
    }); // id2
    let id_arr = b.add(Ty::Array {
        elem: id_u64,
        nelems: 4,
    }); // id3 u64[4]
    let cm_name = b.name("cpumask");
    let bits_name = b.name("bits");
    let id_cpumask = b.add(Ty::Struct {
        name_off: cm_name,
        size: 32,
        members: vec![M {
            name_off: bits_name,
            type_id: id_arr,
            byte_off: 0,
        }],
    }); // id4
    let id_cpumask_ptr = b.add(Ty::Ptr {
        type_id: id_cpumask,
    }); // id5
    let scx_name = b.name("scx_entity");
    let slice_n = b.name("slice");
    let vtime_n = b.name("dsq_vtime");
    let weight_n = b.name("weight");
    let id_scx = b.add(Ty::Struct {
        name_off: scx_name,
        size: 24,
        members: vec![
            M {
                name_off: slice_n,
                type_id: id_u64,
                byte_off: 0,
            },
            M {
                name_off: vtime_n,
                type_id: id_u64,
                byte_off: 8,
            },
            M {
                name_off: weight_n,
                type_id: id_s32,
                byte_off: 16,
            },
        ],
    }); // id6
    let ts_name = b.name("task_struct");
    let pid_n = b.name("pid");
    let cpus_n = b.name("cpus_ptr");
    let scxm_n = b.name("scx");
    b.add(Ty::Struct {
        name_off: ts_name,
        size: 40,
        members: vec![
            M {
                name_off: pid_n,
                type_id: id_s32,
                byte_off: 0,
            },
            M {
                name_off: cpus_n,
                type_id: id_cpumask_ptr,
                byte_off: 8,
            },
            M {
                name_off: scxm_n,
                type_id: id_scx,
                byte_off: 16,
            },
        ],
    }); // id7
    let btf = b.parse();
    (btf, b)
}

/// Helper: BTF with a single `rq { int cpu @0 }` struct (id 2; the
/// int is id 1). `rq` is in STRUCT_FIELDS as `->cpu` -> a single spec.
fn build_rq_btf() -> btf_rs::Btf {
    let mut b = BtfBuilder::new();
    let id_int = b.add(Ty::Int {
        size: 4,
        encoding: 0,
    }); // id1
    let rn = b.name("rq");
    let cn = b.name("cpu");
    b.add(Ty::Struct {
        name_off: rn,
        size: 4,
        members: vec![M {
            name_off: cn,
            type_id: id_int,
            byte_off: 0,
        }],
    });
    b.parse()
}

#[test]
fn resolve_member_offset_simple_scalar() {
    let (btf, _b) = build_task_struct_btf();
    let ts = resolve_struct_type(&btf, "task_struct").unwrap();
    // pid @0, size 4.
    assert_eq!(resolve_member_offset(&btf, &ts, "pid"), Some((0, 4)));
}

#[test]
fn resolve_member_offset_nested_dot_path() {
    let (btf, _b) = build_task_struct_btf();
    let ts = resolve_struct_type(&btf, "task_struct").unwrap();
    // scx @16, slice @0 within scx -> total 16, size 8.
    assert_eq!(resolve_member_offset(&btf, &ts, "scx.slice"), Some((16, 8)));
    // scx @16, dsq_vtime @8 -> 24, size 8.
    assert_eq!(
        resolve_member_offset(&btf, &ts, "scx.dsq_vtime"),
        Some((24, 8))
    );
    // scx @16, weight @16 -> 32, size 4.
    assert_eq!(
        resolve_member_offset(&btf, &ts, "scx.weight"),
        Some((32, 4))
    );
}

#[test]
fn resolve_member_offset_array_index() {
    let (btf, _b) = build_task_struct_btf();
    let cpumask = resolve_struct_type(&btf, "cpumask").unwrap();
    // bits @0, element size 8, index 2 -> offset 16, size 8.
    assert_eq!(
        resolve_member_offset(&btf, &cpumask, "bits[2]"),
        Some((16, 8))
    );
    assert_eq!(
        resolve_member_offset(&btf, &cpumask, "bits[0]"),
        Some((0, 8))
    );
}

#[test]
fn resolve_member_offset_unknown_member() {
    let (btf, _b) = build_task_struct_btf();
    let ts = resolve_struct_type(&btf, "task_struct").unwrap();
    assert_eq!(resolve_member_offset(&btf, &ts, "nonexistent"), None);
    // Nested path with a missing intermediate also yields None.
    assert_eq!(resolve_member_offset(&btf, &ts, "scx.bogus"), None);
}

#[test]
fn resolve_member_offset_bitfield_is_skipped() {
    // A struct whose member sits at a non-byte-aligned bit offset
    // (bit 4) must be rejected by resolve_member_offset.
    let mut raw = BtfBuilder::new();
    let u32_id = raw.add(Ty::Int {
        size: 4,
        encoding: 0,
    });
    let bf_name = raw.name("bf");
    let flag_name = raw.name("flag");
    // struct "bf" { u32 flag @ bit 4 } — bit offset not a multiple of 8.
    let blob = raw.build_with_raw_member(bf_name, flag_name, u32_id, 4);
    let btf = btf_rs::Btf::from_bytes(&blob).expect("parse");
    let bf = resolve_struct_type(&btf, "bf").unwrap();
    assert_eq!(
        resolve_member_offset(&btf, &bf, "flag"),
        None,
        "non-byte-aligned member must be rejected"
    );
}

#[test]
fn resolve_member_offset_through_embedded_union() {
    // A nested dot-path that descends through an embedded union member
    // must follow the Union branch of follow_to_struct_or_union.
    // struct outer { union inner u @8 } where union inner { u64 a @0; u32 b @0 }.
    let mut b = BtfBuilder::new();
    let id_u64 = b.add(Ty::Int {
        size: 8,
        encoding: 0,
    }); // id1
    let id_u32 = b.add(Ty::Int {
        size: 4,
        encoding: 0,
    }); // id2
    let un = b.name("inner");
    let a_n = b.name("a");
    let b_n = b.name("b");
    let id_union = b.add(Ty::Union {
        name_off: un,
        size: 8,
        members: vec![
            M {
                name_off: a_n,
                type_id: id_u64,
                byte_off: 0,
            },
            M {
                name_off: b_n,
                type_id: id_u32,
                byte_off: 0,
            },
        ],
    }); // id3
    let on = b.name("outer");
    let u_n = b.name("u");
    b.add(Ty::Struct {
        name_off: on,
        size: 16,
        members: vec![M {
            name_off: u_n,
            type_id: id_union,
            byte_off: 8,
        }],
    }); // id4
    let btf = b.parse();
    let outer = resolve_struct_type(&btf, "outer").unwrap();
    // u @8, a @0 within union -> total 8, size 8.
    assert_eq!(resolve_member_offset(&btf, &outer, "u.a"), Some((8, 8)));
    // u @8, b @0 within union -> total 8, size 4.
    assert_eq!(resolve_member_offset(&btf, &outer, "u.b"), Some((8, 4)));
}

// ---- resolve_type_size (exercised via resolve_member_offset last-part) ----

#[test]
fn resolve_type_size_ptr_enum_enum64() {
    let mut b = BtfBuilder::new();
    let id_u32 = b.add(Ty::Int {
        size: 4,
        encoding: 0,
    }); // id1
    let id_ptr = b.add(Ty::Ptr { type_id: id_u32 }); // id2
    let en = b.name("color");
    let id_enum = b.add(Ty::Enum {
        name_off: en,
        size: 4,
    }); // id3
    let en64 = b.name("bigenum");
    let id_enum64 = b.add(Ty::Enum64 { name_off: en64 }); // id4
    let sn = b.name("kinds");
    let p = b.name("p");
    let e = b.name("e");
    let e64 = b.name("e64");
    b.add(Ty::Struct {
        name_off: sn,
        size: 24,
        members: vec![
            M {
                name_off: p,
                type_id: id_ptr,
                byte_off: 0,
            },
            M {
                name_off: e,
                type_id: id_enum,
                byte_off: 8,
            },
            M {
                name_off: e64,
                type_id: id_enum64,
                byte_off: 16,
            },
        ],
    });
    let btf = b.parse();
    let s = resolve_struct_type(&btf, "kinds").unwrap();
    // Ptr always reads 8 bytes.
    assert_eq!(resolve_member_offset(&btf, &s, "p"), Some((0, 8)));
    // Enum reports its declared size (4).
    assert_eq!(resolve_member_offset(&btf, &s, "e"), Some((8, 4)));
    // Enum64 always reads 8 bytes.
    assert_eq!(resolve_member_offset(&btf, &s, "e64"), Some((16, 8)));
}

#[test]
fn resolve_type_size_follows_qualifier_chain() {
    // A member typed `const volatile u32` must still resolve to size 4
    // (resolve_type_size peels Const/Volatile/Typedef).
    let mut b = BtfBuilder::new();
    let id_u32 = b.add(Ty::Int {
        size: 4,
        encoding: 0,
    }); // id1
    let id_vol = b.add(Ty::Volatile { type_id: id_u32 }); // id2
    let id_const = b.add(Ty::Const { type_id: id_vol }); // id3
    let td = b.name("my_u32");
    let id_td = b.add(Ty::Typedef {
        name_off: td,
        type_id: id_const,
    }); // id4
    let sn = b.name("q");
    let f = b.name("v");
    b.add(Ty::Struct {
        name_off: sn,
        size: 4,
        members: vec![M {
            name_off: f,
            type_id: id_td,
            byte_off: 0,
        }],
    });
    let btf = b.parse();
    let s = resolve_struct_type(&btf, "q").unwrap();
    assert_eq!(resolve_member_offset(&btf, &s, "v"), Some((0, 4)));
}

// ---- resolve_pointed_struct ----

#[test]
fn resolve_pointed_struct_follows_pointer() {
    let (btf, _b) = build_task_struct_btf();
    let ts = resolve_struct_type(&btf, "task_struct").unwrap();
    // cpus_ptr is ptr->cpumask.
    let pointed = resolve_pointed_struct(&btf, &ts, "cpus_ptr").expect("cpus_ptr -> cpumask");
    assert_eq!(btf.resolve_name(&pointed).unwrap(), "cpumask");
    // A scalar member has no pointed struct.
    assert!(resolve_pointed_struct(&btf, &ts, "pid").is_none());
    // An unknown member yields None.
    assert!(resolve_pointed_struct(&btf, &ts, "nope").is_none());
}

// ---- resolve_field_specs_with_btf: STRUCT_FIELDS curated path ----

#[test]
fn resolve_field_specs_struct_fields_simple_and_chained() {
    let (btf, _b) = build_task_struct_btf();
    // task_struct is in STRUCT_FIELDS; param 0 is a task_struct ptr.
    let func = BtfFunc {
        name: "test".into(),
        params: vec![BtfParam {
            name: "p".into(),
            struct_name: Some("task_struct".into()),
            is_ptr: true,
            ..Default::default()
        }],
        is_variadic: false,
    };
    let specs = resolve_field_specs_with_btf(&func, &btf);

    // STRUCT_FIELDS["task_struct"] order is:
    //  0 ->pid, 1..4 ->cpus_ptr->bits[0..3], 5 ->scx.ddsp_dsq_id,
    //  6 ->scx.ddsp_enq_flags, 7 ->scx.slice, 8 ->scx.dsq_vtime,
    //  9 ->scx.weight, 10 ->scx.sticky_cpu, 11 ->scx.flags.
    // Our synthetic scx only has slice/dsq_vtime/weight, and cpumask
    // has bits[0..3]; the rest of the curated members are absent and
    // produce no spec (but still consume field_idx slots).

    // pid: offset 0, size 4, field_idx 0, single-level.
    let pid = specs.iter().find(|s| s.field_idx == 0).expect("pid spec");
    assert_eq!(pid.param_idx, 0);
    assert_eq!(pid.offset, 0);
    assert_eq!(pid.size, 4);
    assert_eq!(pid.ptr_offset, 0);

    // cpus_ptr->bits[0]: chained. cpus_ptr @8, bits[0] @0 size 8.
    let b0 = specs
        .iter()
        .find(|s| s.field_idx == 1)
        .expect("bits[0] spec");
    assert_eq!(b0.ptr_offset, 8, "intermediate ptr offset = cpus_ptr @8");
    assert_eq!(b0.offset, 0, "bits[0] @0 within cpumask");
    assert_eq!(b0.size, 8);

    // cpus_ptr->bits[2]: same ptr, target offset 16.
    let b2 = specs
        .iter()
        .find(|s| s.field_idx == 3)
        .expect("bits[2] spec");
    assert_eq!(b2.ptr_offset, 8);
    assert_eq!(b2.offset, 16);
    assert_eq!(b2.size, 8);

    // scx.slice is curated field_idx 7 (offset 16, size 8 in our layout).
    let slice = specs
        .iter()
        .find(|s| s.field_idx == 7)
        .expect("scx.slice spec");
    assert_eq!(slice.offset, 16);
    assert_eq!(slice.size, 8);
    assert_eq!(slice.ptr_offset, 0, "scx.slice is a single-level dot path");

    // Curated members absent from our synthetic struct (ddsp_dsq_id @5,
    // sticky_cpu @10, flags @11) produce no spec.
    assert!(specs.iter().all(|s| s.field_idx != 5));
    assert!(specs.iter().all(|s| s.field_idx != 10));
}

#[test]
fn resolve_field_specs_unknown_struct_keeps_field_alignment() {
    // A STRUCT_FIELDS param whose struct is missing from BTF must
    // advance field_idx by the curated field count so later params
    // stay aligned with build_field_keys, and emit no specs for it.
    let btf = build_rq_btf();
    // Param 0: task_struct (NOT present in this BTF) — 12 curated fields.
    // Param 1: rq (present) — ->cpu at field_idx 12.
    let func = BtfFunc {
        name: "test".into(),
        params: vec![
            BtfParam {
                name: "p".into(),
                struct_name: Some("task_struct".into()),
                is_ptr: true,
                ..Default::default()
            },
            BtfParam {
                name: "rq".into(),
                struct_name: Some("rq".into()),
                is_ptr: true,
                ..Default::default()
            },
        ],
        is_variadic: false,
    };
    let specs = resolve_field_specs_with_btf(&func, &btf);
    // No specs for the missing task_struct.
    assert!(specs.iter().all(|s| s.param_idx != 0));
    // rq->cpu lands at field_idx 12 (after the 12 task_struct slots).
    let cpu = specs.iter().find(|s| s.param_idx == 1).expect("rq->cpu");
    assert_eq!(cpu.field_idx, 12);
    assert_eq!(cpu.offset, 0);
    assert_eq!(cpu.size, 4);
}

#[test]
fn resolve_field_specs_scalar_param_consumes_one_slot() {
    // A non-ptr scalar param takes exactly one field slot and emits
    // no spec; a following STRUCT_FIELDS param starts at field_idx 1.
    let btf = build_rq_btf();
    let func = BtfFunc {
        name: "test".into(),
        params: vec![
            BtfParam {
                name: "cpu".into(),
                struct_name: None,
                is_ptr: false,
                ..Default::default()
            },
            BtfParam {
                name: "rq".into(),
                struct_name: Some("rq".into()),
                is_ptr: true,
                ..Default::default()
            },
        ],
        is_variadic: false,
    };
    let specs = resolve_field_specs_with_btf(&func, &btf);
    let cpu = specs.iter().find(|s| s.param_idx == 1).expect("rq->cpu");
    assert_eq!(cpu.field_idx, 1, "scalar param consumed slot 0");
}

#[test]
fn resolve_field_specs_caps_at_six_params() {
    // Only the first 6 params are processed; a 7th STRUCT_FIELDS param
    // must produce no spec.
    let btf = build_rq_btf();
    // 6 scalar params then a 7th rq param.
    let mut params: Vec<BtfParam> = (0..6)
        .map(|i| BtfParam {
            name: format!("a{i}"),
            struct_name: None,
            is_ptr: false,
            ..Default::default()
        })
        .collect();
    params.push(BtfParam {
        name: "rq".into(),
        struct_name: Some("rq".into()),
        is_ptr: true,
        ..Default::default()
    });
    let func = BtfFunc {
        name: "test".into(),
        params,
        is_variadic: false,
    };
    let specs = resolve_field_specs_with_btf(&func, &btf);
    assert!(
        specs.iter().all(|s| s.param_idx != 6),
        "7th param must be ignored (6-param cap)"
    );
}

// ---- resolve_field_specs_with_btf: auto_fields path ----

#[test]
fn resolve_field_specs_auto_fields_single_and_chained() {
    // A param with auto_fields + type_name pointing at a synthetic
    // struct: one scalar field and one cpumask chained field.
    let (btf, _b) = build_task_struct_btf();
    let func = BtfFunc {
        name: "test".into(),
        params: vec![BtfParam {
            name: "t".into(),
            struct_name: None, // not in STRUCT_FIELDS -> auto path
            is_ptr: true,
            type_name: Some("task_struct".into()),
            auto_fields: vec![
                ("pid".into(), "->pid".into(), RenderHint::Signed),
                (
                    "cpus_ptr".into(),
                    "->cpus_ptr->bits[0]".into(),
                    RenderHint::Hex,
                ),
            ],
            ..Default::default()
        }],
        is_variadic: false,
    };
    let specs = resolve_field_specs_with_btf(&func, &btf);
    assert_eq!(specs.len(), 2);
    // pid single-level: offset 0 size 4 ptr_offset 0.
    assert_eq!(specs[0].offset, 0);
    assert_eq!(specs[0].size, 4);
    assert_eq!(specs[0].ptr_offset, 0);
    assert_eq!(specs[0].field_idx, 0);
    // cpus_ptr->bits[0] chained: ptr_offset 8 (cpus_ptr), offset 0, size 8.
    assert_eq!(specs[1].ptr_offset, 8);
    assert_eq!(specs[1].offset, 0);
    assert_eq!(specs[1].size, 8);
    assert_eq!(specs[1].field_idx, 1);
}

#[test]
fn resolve_field_specs_auto_fields_missing_type_name_aligns() {
    // auto_fields with type_name=None must skip field slots equal to
    // the auto_fields count, keeping a later param aligned.
    let btf = build_rq_btf();
    let func = BtfFunc {
        name: "test".into(),
        params: vec![
            BtfParam {
                name: "x".into(),
                struct_name: None,
                is_ptr: true,
                type_name: None, // missing -> skip auto_fields.len() slots
                auto_fields: vec![
                    ("a".into(), "->a".into(), RenderHint::Hex),
                    ("b".into(), "->b".into(), RenderHint::Hex),
                    ("c".into(), "->c".into(), RenderHint::Hex),
                ],
                ..Default::default()
            },
            BtfParam {
                name: "rq".into(),
                struct_name: Some("rq".into()),
                is_ptr: true,
                ..Default::default()
            },
        ],
        is_variadic: false,
    };
    let specs = resolve_field_specs_with_btf(&func, &btf);
    let cpu = specs.iter().find(|s| s.param_idx == 1).expect("rq->cpu");
    assert_eq!(cpu.field_idx, 3, "3 skipped auto slots before rq");
}

// ---- parse_btf_functions over synthetic ELF... (raw-BTF via from_bytes) ----
//
// parse_btf_functions takes a path; to exercise it host-only we write
// the synthetic raw-BTF blob to a temp file (load_btf_from_path accepts
// raw BTF). This drives the full Func->FuncProto->param resolution,
// known-struct vs auto-discover branching, char* detection, variadic
// sentinel handling, and the is_ptr flag.

/// Write `blob` to a fresh temp file, returning its path. The file
/// persists for the test's lifetime via the returned NamedTempFile.
fn blob_to_tempfile(blob: &[u8]) -> tempfile::NamedTempFile {
    use std::io::Write as _;
    let mut f = tempfile::NamedTempFile::new().expect("temp file");
    f.write_all(blob).expect("write blob");
    f.flush().expect("flush");
    f
}

/// Build a BTF containing one Func `probe_fn(task_struct *p, struct
/// foo *f, const char *name, int cpu)` plus the struct/int types.
/// `task_struct` is in STRUCT_FIELDS (-> struct_name); `foo` is not
/// (-> auto_fields + type_name); `name` is char* (-> is_string_ptr);
/// `cpu` is a scalar int.
fn build_probe_fn_btf() -> Vec<u8> {
    let mut b = BtfBuilder::new();
    let id_int = b.add(Ty::Int {
        size: 4,
        encoding: INT_SIGNED,
    }); // id1 int
    let id_char = b.add(Ty::Int {
        size: 1,
        encoding: 0,
    }); // id2 char
    let id_char_const = b.add(Ty::Const { type_id: id_char }); // id3 const char
    let id_char_ptr = b.add(Ty::Ptr {
        type_id: id_char_const,
    }); // id4 const char *
    // task_struct { int pid @0 }.
    let ts_n = b.name("task_struct");
    let pid_n = b.name("pid");
    let id_ts = b.add(Ty::Struct {
        name_off: ts_n,
        size: 8,
        members: vec![M {
            name_off: pid_n,
            type_id: id_int,
            byte_off: 0,
        }],
    }); // id5
    let id_ts_ptr = b.add(Ty::Ptr { type_id: id_ts }); // id6
    // foo { int bar @0 } — not in STRUCT_FIELDS.
    let foo_n = b.name("foo");
    let bar_n = b.name("bar");
    let id_foo = b.add(Ty::Struct {
        name_off: foo_n,
        size: 4,
        members: vec![M {
            name_off: bar_n,
            type_id: id_int,
            byte_off: 0,
        }],
    }); // id7
    let id_foo_ptr = b.add(Ty::Ptr { type_id: id_foo }); // id8
    // FuncProto(p: task_struct*, f: foo*, name: const char*, cpu: int).
    let p_n = b.name("p");
    let f_n = b.name("f");
    let name_n = b.name("name");
    let cpu_n = b.name("cpu");
    let id_proto = b.add(Ty::FuncProto {
        ret: id_int,
        params: vec![
            P {
                name_off: p_n,
                type_id: id_ts_ptr,
            },
            P {
                name_off: f_n,
                type_id: id_foo_ptr,
            },
            P {
                name_off: name_n,
                type_id: id_char_ptr,
            },
            P {
                name_off: cpu_n,
                type_id: id_int,
            },
        ],
    }); // id9
    let fn_n = b.name("probe_fn");
    b.add(Ty::Func {
        name_off: fn_n,
        type_id: id_proto,
    }); // id10
    b.build()
}

#[test]
fn parse_btf_functions_resolves_param_kinds() {
    let blob = build_probe_fn_btf();
    let tf = blob_to_tempfile(&blob);
    let funcs = parse_btf_functions(&["probe_fn"], Some(tf.path().to_str().unwrap()));
    assert_eq!(funcs.len(), 1);
    let f = &funcs[0];
    assert_eq!(f.name, "probe_fn");
    assert!(!f.is_variadic);
    assert_eq!(f.params.len(), 4);

    // p: task_struct ptr -> known struct, no auto_fields.
    assert_eq!(f.params[0].name, "p");
    assert!(f.params[0].is_ptr);
    assert_eq!(f.params[0].struct_name.as_deref(), Some("task_struct"));
    assert!(f.params[0].auto_fields.is_empty());
    assert!(!f.params[0].is_string_ptr);

    // f: foo ptr -> not in STRUCT_FIELDS, auto-discovered, type_name set.
    assert_eq!(f.params[1].name, "f");
    assert!(f.params[1].is_ptr);
    assert!(f.params[1].struct_name.is_none());
    assert_eq!(f.params[1].type_name.as_deref(), Some("foo"));
    // foo has one int member -> one auto field "->bar".
    assert_eq!(
        f.params[1].auto_fields,
        vec![("bar".to_string(), "->bar".to_string(), RenderHint::Signed)]
    );

    // name: const char * -> is_string_ptr.
    assert_eq!(f.params[2].name, "name");
    assert!(f.params[2].is_ptr);
    assert!(f.params[2].is_string_ptr);

    // cpu: scalar int -> not ptr.
    assert_eq!(f.params[3].name, "cpu");
    assert!(!f.params[3].is_ptr);
    assert!(!f.params[3].is_string_ptr);
    assert!(f.params[3].struct_name.is_none());
    assert!(f.params[3].auto_fields.is_empty());
}

#[test]
fn parse_btf_functions_detects_variadic() {
    // FuncProto with a trailing (0,0) sentinel param is variadic; the
    // sentinel is dropped from params.
    let mut b = BtfBuilder::new();
    let id_int = b.add(Ty::Int {
        size: 4,
        encoding: INT_SIGNED,
    });
    let fmt_n = b.name("fmt");
    let cn = b.name("c");
    let id_char = b.add(Ty::Int {
        size: 1,
        encoding: 0,
    });
    let id_const = b.add(Ty::Const { type_id: id_char });
    let id_char_ptr = b.add(Ty::Ptr { type_id: id_const });
    let id_proto = b.add(Ty::FuncProto {
        ret: id_int,
        params: vec![
            P {
                name_off: fmt_n,
                type_id: id_char_ptr,
            },
            P {
                name_off: 0,
                type_id: 0,
            }, // variadic sentinel
        ],
    });
    let _ = cn;
    let fn_n = b.name("printk_like");
    b.add(Ty::Func {
        name_off: fn_n,
        type_id: id_proto,
    });
    let tf = blob_to_tempfile(&b.build());
    let funcs = parse_btf_functions(&["printk_like"], Some(tf.path().to_str().unwrap()));
    assert_eq!(funcs.len(), 1);
    assert!(funcs[0].is_variadic);
    // Sentinel dropped -> only the fmt param remains.
    assert_eq!(funcs[0].params.len(), 1);
    assert_eq!(funcs[0].params[0].name, "fmt");
    assert!(funcs[0].params[0].is_string_ptr);
}

// ---- discover_vmlinux_struct_fields ----

#[test]
fn discover_vmlinux_struct_fields_emits_expected_access() {
    // Struct foo { int sc @0; bool ok @4; enum e @8; cpumask *cm @16;
    //              other *ptr @24 }.
    // Expect: sc -> ->sc/Signed, ok -> ->ok/Bool, e -> ->e/Hex,
    //         cm -> ->cm->bits[0]/Hex, ptr (non-cpumask) skipped.
    let mut b = BtfBuilder::new();
    let id_signed = b.add(Ty::Int {
        size: 4,
        encoding: INT_SIGNED,
    }); // id1
    let id_bool = b.add(Ty::Int {
        size: 1,
        encoding: INT_BOOL,
    }); // id2
    let en = b.name("e_kind");
    let id_enum = b.add(Ty::Enum {
        name_off: en,
        size: 4,
    }); // id3
    // cpumask struct + ptr.
    let cm_n = b.name("cpumask");
    let id_u64 = b.add(Ty::Int {
        size: 8,
        encoding: 0,
    }); // id4
    let id_arr = b.add(Ty::Array {
        elem: id_u64,
        nelems: 4,
    }); // id5
    let bits_n = b.name("bits");
    let id_cpumask = b.add(Ty::Struct {
        name_off: cm_n,
        size: 32,
        members: vec![M {
            name_off: bits_n,
            type_id: id_arr,
            byte_off: 0,
        }],
    }); // id6
    let id_cm_ptr = b.add(Ty::Ptr {
        type_id: id_cpumask,
    }); // id7
    // other struct + ptr (NOT a cpumask).
    let other_n = b.name("other");
    let z_n = b.name("z");
    let id_other = b.add(Ty::Struct {
        name_off: other_n,
        size: 4,
        members: vec![M {
            name_off: z_n,
            type_id: id_signed,
            byte_off: 0,
        }],
    }); // id8
    let id_other_ptr = b.add(Ty::Ptr { type_id: id_other }); // id9
    // foo.
    let foo_n = b.name("foo");
    let sc_n = b.name("sc");
    let ok_n = b.name("ok");
    let e_n = b.name("e");
    let cm_m = b.name("cm");
    let ptr_m = b.name("ptr");
    let id_foo = b.add(Ty::Struct {
        name_off: foo_n,
        size: 32,
        members: vec![
            M {
                name_off: sc_n,
                type_id: id_signed,
                byte_off: 0,
            },
            M {
                name_off: ok_n,
                type_id: id_bool,
                byte_off: 4,
            },
            M {
                name_off: e_n,
                type_id: id_enum,
                byte_off: 8,
            },
            M {
                name_off: cm_m,
                type_id: id_cm_ptr,
                byte_off: 16,
            },
            M {
                name_off: ptr_m,
                type_id: id_other_ptr,
                byte_off: 24,
            },
        ],
    }); // id10
    let btf = b.parse();
    let fields = discover_vmlinux_struct_fields(&btf, id_foo);
    assert_eq!(
        fields,
        vec![
            ("sc".to_string(), "->sc".to_string(), RenderHint::Signed),
            ("ok".to_string(), "->ok".to_string(), RenderHint::Bool),
            ("e".to_string(), "->e".to_string(), RenderHint::Hex),
            (
                "cm".to_string(),
                "->cm->bits[0]".to_string(),
                RenderHint::Hex
            ),
            // `ptr` (pointer to non-cpumask struct) is skipped.
        ]
    );
}

#[test]
fn discover_vmlinux_struct_fields_through_pointer_arg() {
    // discover_vmlinux_struct_fields peels a Ptr to find the struct,
    // so a type_id naming a `ptr->struct` resolves the struct's fields.
    let mut b = BtfBuilder::new();
    let id_u32 = b.add(Ty::Int {
        size: 4,
        encoding: 0,
    }); // id1
    let sn = b.name("s");
    let mn = b.name("m");
    let id_s = b.add(Ty::Struct {
        name_off: sn,
        size: 4,
        members: vec![M {
            name_off: mn,
            type_id: id_u32,
            byte_off: 0,
        }],
    }); // id2
    let id_ptr = b.add(Ty::Ptr { type_id: id_s }); // id3
    let btf = b.parse();
    let fields = discover_vmlinux_struct_fields(&btf, id_ptr);
    assert_eq!(
        fields,
        vec![("m".to_string(), "->m".to_string(), RenderHint::Hex)]
    );
}

// ---- classify_vmlinux_member ----

#[test]
fn classify_vmlinux_member_variants() {
    let mut b = BtfBuilder::new();
    let id_hex = b.add(Ty::Int {
        size: 8,
        encoding: 0,
    }); // id1 unsigned
    let id_signed = b.add(Ty::Int {
        size: 4,
        encoding: INT_SIGNED,
    }); // id2
    let id_bool = b.add(Ty::Int {
        size: 1,
        encoding: INT_BOOL,
    }); // id3
    let en = b.name("ek");
    let id_enum = b.add(Ty::Enum {
        name_off: en,
        size: 4,
    }); // id4
    let en64 = b.name("e64k");
    let id_enum64 = b.add(Ty::Enum64 { name_off: en64 }); // id5
    // const u8 chain -> Int (hex).
    let id_u8 = b.add(Ty::Int {
        size: 1,
        encoding: 0,
    }); // id6
    let id_const_u8 = b.add(Ty::Const { type_id: id_u8 }); // id7
    // cpumask ptr.
    let cm_n = b.name("cpumask");
    let bits_n = b.name("bits");
    let id_cpumask = b.add(Ty::Struct {
        name_off: cm_n,
        size: 8,
        members: vec![M {
            name_off: bits_n,
            type_id: id_hex,
            byte_off: 0,
        }],
    }); // id8
    let id_cm_ptr = b.add(Ty::Ptr {
        type_id: id_cpumask,
    }); // id9
    // cpumask_t typedef -> struct, ptr.
    let cmt_n = b.name("cpumask_t");
    let bits2_n = b.name("bits2");
    let id_cmt = b.add(Ty::Struct {
        name_off: cmt_n,
        size: 8,
        members: vec![M {
            name_off: bits2_n,
            type_id: id_hex,
            byte_off: 0,
        }],
    }); // id10
    let id_cmt_ptr = b.add(Ty::Ptr { type_id: id_cmt }); // id11
    // non-cpumask ptr.
    let widget_n = b.name("widget");
    let w_n = b.name("w");
    let id_other = b.add(Ty::Struct {
        name_off: widget_n,
        size: 4,
        members: vec![M {
            name_off: w_n,
            type_id: id_signed,
            byte_off: 0,
        }],
    }); // id12
    let id_other_ptr = b.add(Ty::Ptr { type_id: id_other }); // id13
    let btf = b.parse();

    // Int hex.
    assert!(matches!(
        classify_vmlinux_member(&btf, id_hex),
        Some(MemberClass::Int(RenderHint::Hex))
    ));
    // Int signed.
    assert!(matches!(
        classify_vmlinux_member(&btf, id_signed),
        Some(MemberClass::Int(RenderHint::Signed))
    ));
    // Int bool.
    assert!(matches!(
        classify_vmlinux_member(&btf, id_bool),
        Some(MemberClass::Int(RenderHint::Bool))
    ));
    // Enum / Enum64.
    assert!(matches!(
        classify_vmlinux_member(&btf, id_enum),
        Some(MemberClass::Enum)
    ));
    assert!(matches!(
        classify_vmlinux_member(&btf, id_enum64),
        Some(MemberClass::Enum)
    ));
    // const u8 chain still classifies as Int.
    assert!(matches!(
        classify_vmlinux_member(&btf, id_const_u8),
        Some(MemberClass::Int(RenderHint::Hex))
    ));
    // cpumask / cpumask_t pointers -> CpumaskPtr.
    assert!(matches!(
        classify_vmlinux_member(&btf, id_cm_ptr),
        Some(MemberClass::CpumaskPtr)
    ));
    assert!(matches!(
        classify_vmlinux_member(&btf, id_cmt_ptr),
        Some(MemberClass::CpumaskPtr)
    ));
    // pointer to a non-cpumask struct -> None (skipped).
    assert!(classify_vmlinux_member(&btf, id_other_ptr).is_none());
    // A bare struct (not int/enum/ptr) -> None.
    assert!(classify_vmlinux_member(&btf, id_other).is_none());
}

// ---- emit_member_field (all MemberClass variants) ----

#[test]
fn emit_member_field_all_variants() {
    let mut fields = Vec::new();
    emit_member_field(&mut fields, "sc", MemberClass::Int(RenderHint::Signed));
    emit_member_field(&mut fields, "ok", MemberClass::Int(RenderHint::Bool));
    emit_member_field(&mut fields, "e", MemberClass::Enum);
    emit_member_field(&mut fields, "cm", MemberClass::CpumaskPtr);
    emit_member_field(&mut fields, "bm", MemberClass::BpfCpumaskPtr);
    assert_eq!(
        fields,
        vec![
            ("sc".to_string(), "->sc".to_string(), RenderHint::Signed),
            ("ok".to_string(), "->ok".to_string(), RenderHint::Bool),
            ("e".to_string(), "->e".to_string(), RenderHint::Hex),
            (
                "cm".to_string(),
                "->cm->bits[0]".to_string(),
                RenderHint::Hex
            ),
            (
                "bm".to_string(),
                "->bm->cpumask.bits[0]".to_string(),
                RenderHint::Hex
            ),
        ]
    );
}

// ---- infer_scalar_param_name ----

#[test]
fn infer_scalar_param_name_known_ops() {
    // select_cpu's param 1 is prev_cpu, param 2 is wake_flags.
    assert_eq!(infer_scalar_param_name("ktstr_select_cpu", 1), "prev_cpu");
    assert_eq!(infer_scalar_param_name("ktstr_select_cpu", 2), "wake_flags");
    // dispatch's param 0 is cpu.
    assert_eq!(infer_scalar_param_name("sched_dispatch", 0), "cpu");
    // update_idle: cpu, idle.
    assert_eq!(infer_scalar_param_name("x_update_idle", 0), "cpu");
    assert_eq!(infer_scalar_param_name("x_update_idle", 1), "idle");
    // set_weight: index 0 is empty in the table -> falls through to arg0.
    assert_eq!(infer_scalar_param_name("foo_set_weight", 0), "arg0");
    assert_eq!(infer_scalar_param_name("foo_set_weight", 1), "weight");
}

#[test]
fn infer_scalar_param_name_fallback() {
    // Unknown op or out-of-range position -> arg{pos}.
    assert_eq!(infer_scalar_param_name("mystery_callback", 0), "arg0");
    assert_eq!(infer_scalar_param_name("mystery_callback", 3), "arg3");
    // Known op but position past its table -> arg{pos}.
    assert_eq!(infer_scalar_param_name("x_dispatch", 5), "arg5");
}
